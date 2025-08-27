import os,math,json
from dataclasses import dataclass,field
from typing import Optional
import torch
from transformers import TrainerCallback
from .pipeline import CodecConfig,TransformSpec,CheckpointCompressor

@dataclass
class CallbackConfig:
    method:str="warp_q"
    avg_bitwidth:float=4.0
    target_nz_ratio:float=0.10
    enforce_target_nz:bool=True
    rate_mode:str="nz"
    target_bpp:Optional[float]=None
    bpp_tol:float=1e-3
    quant_start_step:int=400
    quant_freq:int=400
    gop_size:int=1
    i_frame_scale:float=1.0
    p_frame_scale:float=0.8
    apply_to_model:bool=False
    save_dir:Optional[str]=None
    save_payload:bool=True
    transform:TransformSpec=field(default_factory=TransformSpec)
    qkv_group_size:int=128
    qkv_clip_pct:float=0.99
    qkv_use_dwt:bool=False
    momentum_bitwidth:int=8
    momentum_clip_pct:float=0.999
    m2w_gate:bool=True
    pi_disable_threshold:float=0.1
    mask_from:str="post"
    dwt_alloc_mode:str="fixed"
    dwt_alloc:str="A:0.6,LH:0.15,HL:0.15,HH:0.1"
    dwt_bits:str="A:5,D:2"
    use_codebook:bool=False
    cb_bits_min:int=0
    cb_bits_max:int=6
    cb_A_min:int=4
    coeff_mask_mode:str="energy"
    cb_sample:int=50000
    cb_iters:int=8

def _to_ccfg(c:CallbackConfig)->CodecConfig:
    return CodecConfig(
        avg_bitwidth=c.avg_bitwidth,
        target_nz_ratio=c.target_nz_ratio,
        enforce_target_nz=c.enforce_target_nz,
        rate_mode=c.rate_mode,
        target_bpp=c.target_bpp,
        bpp_tol=c.bpp_tol,
        qkv_group_size=c.qkv_group_size,
        qkv_clip_pct=c.qkv_clip_pct,
        momentum_bitwidth=c.momentum_bitwidth,
        momentum_clip_pct=c.momentum_clip_pct,
        m2w_gate=c.m2w_gate,
        pi_disable_threshold=c.pi_disable_threshold,
        mask_from=c.mask_from,
        transform=c.transform,
        qkv_use_dwt=c.qkv_use_dwt,
        dwt_alloc_mode=c.dwt_alloc_mode,
        dwt_alloc=c.dwt_alloc,
        dwt_bits=c.dwt_bits,
        save_compact=True,
        zstd_level=8,
        use_codebook=c.use_codebook,
        cb_bits_min=c.cb_bits_min,
        cb_bits_max=c.cb_bits_max,
        cb_A_min=c.cb_A_min,
        cb_sample=c.cb_sample,
        cb_iters=c.cb_iters,
        coeff_mask_mode=c.coeff_mask_mode
    )

def _mb(x:int)->float: return float(x)/(1024.0*1024.0)

class WaRPQCallback(TrainerCallback):
    def __init__(self,cfg:CallbackConfig): self.cfg=cfg; self.engine=None; self._prepared=False; self._optimizer_ref=None; self._save_root=None
    def on_train_begin(self,args,state,control,model=None,optimizer=None,**k):
        self._optimizer_ref=optimizer
        self._save_root=self.cfg.save_dir if self.cfg.save_dir else os.path.join(args.output_dir,"ckpts")
        os.makedirs(self._save_root,exist_ok=True)
        print(f"[WaRPQ] start_at={self.cfg.quant_start_step}, freq={self.cfg.quant_freq}, apply_to_model={self.cfg.apply_to_model}, save_dir={self._save_root}")
        if self.engine is None:
            self.engine=CheckpointCompressor(_to_ccfg(self.cfg))
            self.engine.prepare(model,optimizer=optimizer)
            self._prepared=True
        return control
    def on_step_end(self,args,state,control,model=None,optimizer=None,**k):
        if self.engine is None: return control
        gs=state.global_step
        if gs<self.cfg.quant_start_step: return control
        if (gs-self.cfg.quant_start_step)%self.cfg.quant_freq!=0: return control
        opt=optimizer if optimizer is not None else self._optimizer_ref
        compact_path=os.path.join(self._save_root,f"step{int(gs):08d}.warpq")
        stats=self.engine.compress_step(model,gs,optimizer=opt,apply_to_model=self.cfg.apply_to_model,save_dir=self._save_root,compact_path=compact_path)
        warpq_b=_mb(stats.get("compact_bytes",0))
        fw=_mb(stats.get("fullW_bytes",0)); fo=_mb(stats.get("fullO_bytes",0))
        cr_ckpt=stats.get("ratio_ckpt_WplusO",None)
        p=stats.get("p_eff",0.0); bavg=stats.get("bavg_eff",0.0); bpp=stats.get("bpp",0.0); crw=stats.get("cr_weight",0.0); snr=stats.get("snr_db",0.0)
        ckpt_str=f"CR_ckpt≈{cr_ckpt:.1f}x (W+O)" if cr_ckpt else "CR_ckpt=NA"
        print(f"[WaRPQ] step={gs} frame=I | p={p:.4f} bavg={bavg:.2f} bpp={bpp:.3f} CR≈{crw:.1f}x SNR={snr:.2f}dB | file warpq={warpq_b:.2f}MB, fullW≈{fw:.2f}MB, fullO≈{fo:.2f}MB | {ckpt_str} -> {self._save_root}")
        return control
