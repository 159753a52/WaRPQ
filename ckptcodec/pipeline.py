# pipeline.py
import os,re,math,json,time
from dataclasses import dataclass,field
from typing import Dict,Any,Tuple,List,Optional
import numpy as np
import torch,torch.nn as nn,torch.nn.functional as F
from .io import WarpqWriter,pack_mask,pack_uintegers,to_bytes_fp16
from .cbq import band_codebook_quantize

def _H(p:float)->float:
    if p<=0.0 or p>=1.0: return 0.0
    return float(-p*math.log2(p)-(1-p)*math.log2(1-p))
def _sd(a: float, b: float, eps: float = 1e-12) -> float: return float(a / (b + eps))

def project_topk_time(x:torch.Tensor,p_keep:float):
    if p_keep>=1.0: return x,torch.ones_like(x,dtype=x.dtype)
    if p_keep<=0.0: return torch.zeros_like(x),torch.zeros_like(x,dtype=x.dtype)
    n=x.numel();k=max(1,int(round(p_keep*n)));flat=x.abs().reshape(-1);thr=torch.kthvalue(flat,n-k+1).values
    m=(x.abs()>=thr).to(dtype=x.dtype);return x*m,m

@torch.no_grad()
def quantize_uniform_symmetric_q(y:torch.Tensor,n_bits:int,vmax:torch.Tensor):
    if n_bits<=1:
        z=torch.zeros_like(y); return z,torch.zeros_like(vmax,dtype=torch.int32),torch.zeros_like(vmax)
    qmax=2**(n_bits-1)-1;s=(vmax/qmax).clamp_min(1e-8);q=torch.clamp(torch.round(y/s),min=-qmax,max=qmax).to(torch.int32)
    return (q*s),q,s

@torch.no_grad()
def quantize_pergroup_uniform_q(t:torch.Tensor,n_bits:int,group:int,dim:int=-1,clip:float=0.995):
    if n_bits<=0 or t.ndim==0 or t.numel()==0:
        z=torch.zeros(1,device=t.device,dtype=t.dtype)
        return t.clone(),torch.zeros(1,device=t.device,dtype=torch.int32),z
    dtype=t.dtype;dim=dim if dim>=0 else t.ndim+dim
    p=list(range(t.ndim));p[dim],p[-1]=p[-1],p[dim]
    x=t.permute(*p).contiguous();sh=x.shape;L=sh[-1];x=x.view(-1,L)
    pad=(group-(L%group))%group
    if pad>0: x=torch.cat([x,torch.zeros(x.shape[0],pad,device=x.device,dtype=x.dtype)],1)
    G=x.shape[1]//group;x=x.view(-1,G,group)
    vmax=torch.quantile(x.abs(),q=clip,dim=-1,keepdim=True).clamp_min(1e-8);qmax=2**(n_bits-1)-1
    q=torch.clamp(torch.round(x/(vmax/qmax)),min=-qmax,max=qmax).to(torch.int32);deq=q*(vmax/qmax)
    deq=deq.view(-1,G*group);deq=deq[:,:L] if pad>0 else deq;deq=deq.view(*sh[:-1],L)
    inv=list(range(t.ndim));inv[dim],inv[-1]=inv[-1],inv[dim]
    deq=deq.permute(*inv).contiguous().to(dtype=dtype)
    return deq,q,vmax

def _pad_pow2(x,J):
    m=1<<J;T=x.shape[-1];pad=(-T)%m;return (F.pad(x,(0,pad)) if pad>0 else x),pad
def haar1d_fwd_vec(x,J):
    y,_=_pad_pow2(x,J);dets=[]
    for _ in range(J):
        even=y[...,::2];odd=y[...,1::2];cA=(even+odd)/math.sqrt(2.0);cD=(even-odd)/math.sqrt(2.0);dets.append(cD);y=cA
    return y,dets
def haar1d_inv_vec(cA,dets):
    y=cA
    for cD in reversed(dets):
        even=(y+cD)/math.sqrt(2.0);odd=(y-cD)/math.sqrt(2.0);y=torch.stack([even,odd],dim=-1).reshape(*even.shape[:-1],even.shape[-1]*2)
    return y
def blockify_1d(x:torch.Tensor,L:int,dim:int,J:int):
    d=dim if dim>=0 else x.ndim+dim
    perm=list(range(x.ndim));perm[d],perm[-1]=perm[-1],perm[d]
    inv=[0]*len(perm)
    for i,p in enumerate(perm): inv[p]=i
    X=x.permute(*perm).contiguous();prefix=list(X.shape[:-1]);T=X.shape[-1]
    lead=int(np.prod(prefix)) if prefix else 1;Y=X.view(lead,T);mul=L*(1<<J);pad=(-T)%mul
    if pad>0: Y=F.pad(Y,(0,pad))
    nb=Y.shape[1]//L;B=Y.view(-1,nb,L).reshape(-1,L)
    return B,{"perm":perm,"invperm":inv,"prefix":prefix,"T":T,"pad":pad,"L":L,"nb":nb,"lead":lead}
def unblockify_1d(B:torch.Tensor,meta:Dict[str,Any]):
    L,nb,lead,T,pad=meta["L"],meta["nb"],meta["lead"],meta["T"],meta["pad"]
    Y=B.view(lead,nb,L).reshape(lead,nb*L)
    if pad>0: Y=Y[:,:nb*L-pad]
    Y=Y[:,:T];X=Y.view(*meta["prefix"],T)
    return X.permute(*meta["invperm"]).contiguous()

@torch.no_grad()
def lifting1d_fwd_ls(B:torch.Tensor,J:int):
    y,_=_pad_pow2(B,J);dets=[];ab=[]
    cur=y
    for _ in range(J):
        even=cur[...,::2];odd=cur[...,1::2]
        a=((even*odd).sum(-1,keepdim=True)/((odd*odd).sum(-1,keepdim=True).clamp_min(1e-8)))
        d=even-a*odd
        b=-( (odd*d).sum(-1,keepdim=True) / ((d*d).sum(-1,keepdim=True).clamp_min(1e-8)) )
        s=odd+b*d;dets.append(d);ab.append((a,b));cur=s
    return cur,dets,ab

def _encode_tensor_wavelet(writer,name,T,*,axis_dim:int,L:int,J:int,bitsA:int,bitsD:int,
                           mask_bool_np,meta1:dict,perm:torch.Tensor=None,inv:torch.Tensor=None,
                           sqrt:bool=False):
    X=torch.sqrt(T.clamp_min(0)) if sqrt else T
    if perm is not None: X=X.index_select(axis_dim,perm.to(X.device))
    B,meta=blockify_1d(X,L=L,dim=axis_dim,J=J)
    cS,d_list,_=lifting1d_fwd_ls(B,J)
    vS=torch.quantile(cS.abs(),q=0.9995 if sqrt else 0.999,dim=1,keepdim=True).clamp_min(1e-8)
    qS_deq,qS_int,sA=quantize_uniform_symmetric_q(cS,bitsA,vS)
    qbytesA=pack_uintegers((qS_int.reshape(-1).cpu().numpy()+(1<<(bitsA-1))).astype(np.int64),bitsA)
    sbytesA=to_bytes_fp16(sA.cpu().numpy())
    qbytesDs,sbytesDs=[],[]
    for di in d_list:
        vD=torch.quantile(di.abs(),q=0.99,dim=1,keepdim=True).clamp_min(1e-8)
        _,qD_int,sD=quantize_uniform_symmetric_q(di,bitsD,vD)
        qbytesDs.append(pack_uintegers((qD_int.reshape(-1).cpu().numpy()+(1<<(bitsD-1))).astype(np.int64),bitsD))
        sbytesDs.append(to_bytes_fp16(sD.cpu().numpy()))
    blobs={"mask":pack_mask(mask_bool_np.astype(np.uint8)),"qA":qbytesA,"sA":sbytesA}
    for i,(qb,sb) in enumerate(zip(qbytesDs,sbytesDs)): blobs[f"qD{i+1}"]=qb; blobs[f"sD{i+1}"]=sb
    if perm is not None: blobs["perm"]=perm.cpu().numpy().astype(np.int32).tobytes()
    if inv  is not None: blobs["inv"]=inv.cpu().numpy().astype(np.int32).tobytes()
    meta_comp={"codec":"haar1d","J":int(J),"axis":int(axis_dim),"L":int(L),"bits":[int(bitsA),int(bitsD)],
               "has_perm":perm is not None,"has_inv":inv is not None,
               "perm_len":int(perm.numel()) if perm is not None else 0,
               "inv_len":int(inv.numel()) if inv is not None else 0,
               "meta1":meta,"shape":list(T.shape),"sqrt":bool(sqrt)}
    writer.add(name,meta_comp,blobs)

@dataclass
class TransformSpec:
    kind:str="g2lwt"; tile_h:int=32; tile_w:int=32; dwt_levels:int=1
@dataclass
class CodecConfig:
    avg_bitwidth:float=4.0; target_nz_ratio:float=0.10; enforce_target_nz:bool=True
    rate_mode:str="nz"; target_bpp:Optional[float]=None; bpp_tol:float=1e-3
    qkv_group_size:int=128; qkv_clip_pct:float=0.99
    momentum_bitwidth:int=8; momentum_clip_pct:float=0.999; m2w_gate:bool=True
    pi_disable_threshold:float=0.1; mask_from:str="post"
    transform:TransformSpec=field(default_factory=TransformSpec)
    qkv_use_dwt:bool=False; dwt_alloc_mode:str="fixed"; dwt_alloc:str="A:0.6,LH:0.15,HL:0.15,HH:0.1"; dwt_bits:str="A:5,D:2"
    save_compact:bool=True; zstd_level:int=8
    use_codebook:bool=False; cb_bits_min:int=0; cb_bits_max:int=6; cb_A_min:int=4
    cb_sample:int=50000; cb_iters:int=8; coeff_mask_mode:str="energy"

class CheckpointCompressor:
    def __init__(self,config:CodecConfig):
        self.cfg=config;self.baseline_state:Dict[str,torch.Tensor]={};self.nz_eps=1e-12;self._pa_cache:Dict[str,Any]={};self._optimizer_ref=None
    def prepare(self,model:nn.Module,optimizer=None):
        self.baseline_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        self._optimizer_ref=optimizer
        if self.cfg.transform.kind in ("pa_adwt","g2lwt"):
            with torch.no_grad():
                for name,p in model.named_parameters():
                    if self._ok(name,p):
                        try: self._prepare_pa_for_param(name,p.data)
                        except: pass
    @torch.no_grad()
    def compress_step(self,model:nn.Module,step:int,optimizer=None,apply_to_model:bool=False,save_dir:Optional[str]=None,compact_path:Optional[str]=None)->Dict[str,Any]:
        dev=next(model.parameters()).device;opt=optimizer if optimizer is not None else self._optimizer_ref;t0=time.perf_counter()
        if save_dir is not None:
            os.makedirs(save_dir,exist_ok=True)
            if self.baseline_state is not None:
                torch.save(self.baseline_state,os.path.join(save_dir,f"baseline_step{int(step):08d}_pre.pt"))
        writer=None
        if compact_path and self.cfg.save_compact: writer=WarpqWriter(compact_path,level=self.cfg.zstd_level)
        stats={"nnz_eff":0,"tot_eff":0,"bsum_eff":0,"bcnt_eff":0,"nmse":0.0,"varws":0.0,"qkvL":0,"qkve":0}
        rec_totals:Dict[str,torch.Tensor]={}
        for name,param in model.named_parameters():
            if name not in self.baseline_state or not self._ok(name,param): continue
            base=self.baseline_state[name].to(device=dev,dtype=param.dtype);w=param.data
            st=opt.state.get(param,{}) if opt is not None else {};m=st.get("exp_avg",None);v=st.get("exp_avg_sq",None);lr=self._get_lr(opt,param)
            pred=torch.zeros_like(w) if (m is None or v is None or lr is None) else (-lr*(m.to(device=w.device,dtype=w.dtype)/(torch.sqrt(v.to(device=w.device,dtype=w.dtype)+1e-12))))
            E=(w-base)-pred
            rec_E,mask,per,pack_meta,pack_blobs,ctx=self._compress_param(name,E,base,model)
            d=(E.float()-rec_E.float());msqe=float(torch.mean(d*d).item());varE=float(E.float().var(unbiased=False).item())
            stats["nmse"]+=msqe*E.numel();stats["varws"]+=varE*E.numel();stats["tot_eff"]+=per["tot_eff"];stats["nnz_eff"]+=per["nnz_eff"];stats["bsum_eff"]+=per["bsum_eff"];stats["bcnt_eff"]+=per["bcnt_eff"];stats["qkvL"]+=per.get("qkvL",0);stats["qkve"]+=per.get("qkve",0)
            rec_total=pred+rec_E;rec_totals[name]=rec_total.detach().cpu()
            if writer and pack_meta is not None:
                pack_meta["shape"]=list(E.shape); writer.add(name,pack_meta,pack_blobs)
            if apply_to_model: param.data.copy_(base+rec_total.to(device=base.device,dtype=base.dtype))
            self.baseline_state[name]=(base+rec_total).detach().cpu()
            if opt is not None and (param in opt.state): self._maybe_prune_and_quantize_momentum(opt.state[param],E,rec_E)
            if writer is not None and opt is not None and (param in opt.state) and ctx is not None:
                mb=(mask!=0).bool().detach().cpu().numpy()
                bitsA,bitsD=ctx["bits"][0],ctx["bits"][1]
                if "exp_avg" in opt.state[param] and isinstance(opt.state[param]["exp_avg"],torch.Tensor):
                    _encode_tensor_wavelet(writer,f"{name}/momentum/exp_avg",
                        opt.state[param]["exp_avg"].to(device=E.device,dtype=E.dtype)*mask,
                        axis_dim=ctx["axis_dim"],L=ctx["L"],J=ctx["J"],bitsA=max(6,bitsA),bitsD=bitsD,
                        mask_bool_np=mb,meta1=ctx["meta1"],perm=ctx["perm"],inv=ctx["inv"],sqrt=False)
                if "exp_avg_sq" in opt.state[param] and isinstance(opt.state[param]["exp_avg_sq"],torch.Tensor):
                    _encode_tensor_wavelet(writer,f"{name}/momentum/exp_avg_sq",
                        opt.state[param]["exp_avg_sq"].to(device=E.device,dtype=E.dtype)*mask,
                        axis_dim=ctx["axis_dim"],L=ctx["L"],J=ctx["J"],bitsA=max(6,bitsA),bitsD=bitsD,
                        mask_bool_np=mb,meta1=ctx["meta1"],perm=ctx["perm"],inv=ctx["inv"],sqrt=True)
        bavg=(stats["bsum_eff"]/stats["bcnt_eff"]) if stats["bcnt_eff"]>0 else float(self.cfg.avg_bitwidth)
        p=_sd(stats["nnz_eff"],stats["tot_eff"]);bpp=p*bavg+_H(p);cr=32.0/max(bpp,1e-9);nmse=_sd(stats["nmse"],stats["varws"]);snr=10.0*math.log10(max(1.0/max(nmse,1e-12),1e-12));t1=time.perf_counter()
        out={"step":int(step),"p_eff":float(p),"bavg_eff":float(bavg),"bpp":float(bpp),"cr_weight":float(cr),"nmse":float(nmse),"snr_db":float(snr),"time_s":float(t1-t0),"qkv_fast_layers":int(stats["qkvL"])}

        pt_path=None
        if save_dir is not None:
            with open(os.path.join(save_dir,"compress_stats.jsonl"),"a") as f: f.write(json.dumps(out)+"\n")
            pt_path=os.path.join(save_dir,f"rcp_step{int(step):08d}.pt")
            torch.save({"step":int(step),"delta":rec_totals},pt_path)

        if writer: writer.close()
        compact_bytes=(os.path.getsize(compact_path) if compact_path and os.path.exists(compact_path) else 0)
        pt_bytes=(os.path.getsize(pt_path) if pt_path and os.path.exists(pt_path) else 0)

        fullW_bytes=0
        with torch.no_grad():
            for p_ in model.parameters(): fullW_bytes+=p_.numel()*p_.element_size()
        fullO_bytes=0
        if opt is not None:
            for _,st_ in opt.state.items():
                for v in st_.values():
                    if isinstance(v,torch.Tensor): fullO_bytes+=v.numel()*v.element_size()

        full_ckpt_bytes=fullW_bytes+fullO_bytes
        compact_ckpt_bytes=compact_bytes

        out.update({
            "compact_bytes":int(compact_bytes),
            "pt_delta_bytes":int(pt_bytes),
            "fullW_bytes":int(fullW_bytes),
            "fullO_bytes":int(fullO_bytes),
            "full_ckpt_bytes":int(full_ckpt_bytes),
            "ratio_fullW_vs_compactW":(float(fullW_bytes)/float(compact_bytes)) if compact_bytes>0 else None,
            "ratio_deltaPT_vs_compactW":(float(pt_bytes)/float(compact_bytes)) if compact_bytes>0 else None,
            "ratio_ckpt_WplusO":(float(full_ckpt_bytes)/float(compact_ckpt_bytes)) if compact_ckpt_bytes>0 else None
        })
        return out

    def _compress_param(self,name:str,E:torch.Tensor,base:torch.Tensor,model:nn.Module):
        cfg=self.cfg;per={"tot_eff":E.numel(),"nnz_eff":0,"bsum_eff":0,"bcnt_eff":0}; pack_meta=None; pack_blobs=None
        if self._is_qkv(name) and not cfg.qkv_use_dwt:
            bq=int(round(cfg.avg_bitwidth))
            deq,q_int,vmax=quantize_pergroup_uniform_q(E,n_bits=bq,group=cfg.qkv_group_size,dim=-1,clip=cfg.qkv_clip_pct)
            p_keep=(self._solve_p_for_bpp(bq) if cfg.rate_mode=="bpp" else cfg.target_nz_ratio)
            if cfg.enforce_target_nz:
                if cfg.mask_from=="pre": _,mask=project_topk_time(E,p_keep);deq=deq*mask
                else: deq,mask=project_topk_time(deq,p_keep)
            else: mask=torch.ones_like(deq,dtype=deq.dtype)
            nz=int((mask!=0).sum().item());per["nnz_eff"]+=nz;per["bsum_eff"]+=bq*nz;per["bcnt_eff"]+=nz;per["qkvL"]=1;per["qkve"]=E.numel()
            deq=self._trust_blend(E,deq,name,cfg.pi_disable_threshold)
            pack_ctx={"axis_dim":-1,"L":cfg.qkv_group_size,"J":1,"bits":[bq,bq],"perm":None,"inv":None,"meta1":{}}
            if cfg.save_compact:
                u=(q_int.reshape(-1).to(torch.int32).cpu().numpy()+ (1<<(bq-1))).astype(np.int64)
                mb=(mask!=0).bool().cpu().numpy(); vmaxnp=vmax.cpu().numpy()
                pack_meta={"codec":"qgroup","bits":bq,"group":cfg.qkv_group_size,"dim":-1,"vmax_shape":list(vmax.shape),"clip":cfg.qkv_clip_pct}
                pack_blobs={"mask":pack_mask(mb),"q":pack_uintegers(u,nbits=bq),"vmax":to_bytes_fp16(vmaxnp)}
            return deq,mask,per,pack_meta,pack_blobs,pack_ctx

        kind=cfg.transform.kind;axis_dim=-1 if kind=="haar1d_in" else -2 if kind=="haar1d_out" else -1
        perm=inv=None;X_for=E
        if kind in ("pa_adwt","g2lwt"):
            entry=self._pa_cache.get(name,None) or self._prepare_pa_for_param(name,base);axis_dim=entry["dim"];perm=entry["perm"].to(E.device);inv=entry["inv"].to(E.device);X_for=E.index_select(axis_dim,perm)
        L=self._pick_L_eff(X_for.shape[axis_dim],(cfg.transform.tile_w if axis_dim==-1 else cfg.transform.tile_h),cfg.transform.dwt_levels)

        if cfg.use_codebook:
            _Btmp,meta1=blockify_1d(X_for,L=L,dim=axis_dim,J=cfg.transform.dwt_levels)
            rec_x,meta_cb,blobs_cb,bstats=band_codebook_quantize(
                X_for,L=L,J=cfg.transform.dwt_levels,axis_dim=axis_dim,
                avg_bits=cfg.avg_bitwidth,keep_ratio=cfg.target_nz_ratio,
                bmin=cfg.cb_bits_min,bmax=cfg.cb_bits_max,A_min=cfg.cb_A_min,
                mask_mode=cfg.coeff_mask_mode,cb_sample=cfg.cb_sample,cb_iters=cfg.cb_iters,
                perm=None,inv=None)
            rec_E=(rec_x if perm is None else rec_x.index_select(axis_dim,inv))
            mask=torch.ones_like(rec_E,dtype=rec_E.dtype)
            kept=int(bstats.get("kept",0)); ibits=int(bstats.get("ibits",0))
            per["nnz_eff"]+=kept; per["bsum_eff"]+=ibits; per["bcnt_eff"]+=max(kept,1)
            pack_meta,pack_blobs=meta_cb,blobs_cb
            bits_mom=[max(6,cfg.cb_A_min), max(2,min(cfg.cb_bits_max,3))]
            pack_ctx={"axis_dim":axis_dim,"L":L,"J":cfg.transform.dwt_levels,"bits":bits_mom,"perm":perm,"inv":inv,"meta1":meta1}
            return rec_E,mask,per,pack_meta,pack_blobs,pack_ctx

        B1,meta1=blockify_1d(X_for,L=L,dim=axis_dim,J=cfg.transform.dwt_levels);cA,dets=haar1d_fwd_vec(B1,cfg.transform.dwt_levels)
        b_base=int(round(cfg.avg_bitwidth));bits=self._bits_for_haar1d(b_base,cfg.transform.dwt_levels)
        vA=torch.quantile(cA.abs(),q=0.995,dim=1,keepdim=True).clamp_min(1e-8);rcA,qA,sA=quantize_uniform_symmetric_q(cA,bits[0],vA)
        rD=[];qDs=[];sDs=[]
        for d in dets:
            vm=torch.quantile(d.abs(),q=0.990,dim=1,keepdim=True).clamp_min(1e-8);dq,qi,si=quantize_uniform_symmetric_q(d,bits[1],vm);rD.append(dq);qDs.append(qi);sDs.append(si)
        RB=haar1d_inv_vec(rcA,rD);rb=RB[...,:B1.shape[-1]];rec_perm=unblockify_1d(rb,meta1).reshape_as(X_for);rec_E=(rec_perm if perm is None else rec_perm.index_select(axis_dim,inv))
        if cfg.enforce_target_nz:
            p_keep=(self._solve_p_for_bpp(b_base) if cfg.rate_mode=="bpp" else cfg.target_nz_ratio)
            if cfg.mask_from=="pre": _,mask=project_topk_time(E,p_keep);rec_E=rec_E*mask
            else: rec_E,mask=project_topk_time(rec_E,p_keep)
        else: mask=torch.ones_like(rec_E,dtype=rec_E.dtype)
        rec_E=self._trust_blend(E,rec_E,name,max(0.2,cfg.pi_disable_threshold));nz=int((mask!=0).sum().item())
        per["nnz_eff"]+=nz;per["bsum_eff"]+=int(round(b_base))*nz;per["bcnt_eff"]+=nz
        if cfg.save_compact:
            qa=(qA.reshape(-1).to(torch.int32).cpu().numpy()+(1<<(bits[0]-1))).astype(np.int64); qbytesA=pack_uintegers(qa,bits[0]); sbytesA=to_bytes_fp16(sA.cpu().numpy())
            qbytesDs=[]; sbytesDs=[]
            for qi,si in zip(qDs,sDs):
                qn=(qi.reshape(-1).to(torch.int32).cpu().numpy()+(1<<(bits[1]-1))).astype(np.int64)
                qbytesDs.append(pack_uintegers(qn,bits[1])); sbytesDs.append(to_bytes_fp16(si.cpu().numpy()))
            mb=(mask!=0).bool().cpu().numpy()
            perm_bytes=(perm.cpu().numpy().astype(np.int32).tobytes() if perm is not None else b"")
            inv_bytes =(inv.cpu().numpy().astype(np.int32).tobytes()  if inv  is not None else b"")
            has_perm,has_inv=(len(perm_bytes)>0),(len(inv_bytes)>0)
            perm_len=(perm.numel() if perm is not None else 0); inv_len=(inv.numel() if inv is not None else 0)
            meta={"codec":"haar1d","J":cfg.transform.dwt_levels,"axis":axis_dim,"L":L,"bits":[bits[0],bits[1]],
                  "has_perm":has_perm,"has_inv":has_inv,"perm_len":int(perm_len),"inv_len":int(inv_len),
                  "meta1":meta1,"shape":list(E.shape)}
            blobs={"mask":pack_mask(mb),"qA":qbytesA,"sA":sbytesA}
            for i,(qb,sb) in enumerate(zip(qbytesDs,sbytesDs)): blobs[f"qD{i+1}"]=qb; blobs[f"sD{i+1}"]=sb
            if has_perm: blobs["perm"]=perm_bytes
            if has_inv:  blobs["inv"]=inv_bytes
            pack_meta,pack_blobs=meta,blobs
        pack_ctx={"axis_dim":axis_dim,"L":L,"J":cfg.transform.dwt_levels,"bits":bits,"perm":perm,"inv":inv,"meta1":meta1}
        return rec_E,mask,per,pack_meta,pack_blobs,pack_ctx

    def _ok(self,n:str,p:torch.nn.Parameter):
        if not p.requires_grad: return False
        l=n.lower()
        if l.endswith(".bias") or ".bias" in l: return False
        if "layernorm" in l or re.search(r"\bnorm\b",l) or "ln_" in l: return False
        if "embed" in l or "embedding" in l or "wte" in l or "wpe" in l: return False
        if p.ndim<=1: return False
        return True
    def _is_qkv(self,name):
        l=name.lower()
        return ("c_attn.weight" in name) or ("attention.query_key_value.weight" in l) or ("q_proj" in l) or ("k_proj" in l) or ("v_proj" in l)
    def _find_group_for_param(self,opt,param):
        if opt is None: return None
        for g in opt.param_groups:
            for p in g.get("params",[]):
                if p is param: return g
        return None
    def _get_lr(self,opt,param):
        g=self._find_group_for_param(opt,param); return None if g is None else g.get("lr",None)
    def _solve_p_for_bpp(self,b_eff:int):
        if self.cfg.rate_mode!="bpp" or self.cfg.target_bpp is None: return self.cfg.target_nz_ratio
        lo,hi=1e-6,0.5
        for _ in range(16):
            mid=(lo+hi)/2;bpp=mid*b_eff+_H(mid)
            if bpp>self.cfg.target_bpp: hi=mid
            else: lo=mid
            if abs(bpp-self.cfg.target_bpp)<=self.cfg.bpp_tol: break
        return (lo+hi)/2
    def _pca_seriation(self,W_2d:torch.Tensor,dim:int):
        Wcpu=W_2d.detach().float().cpu()
        if dim==0: U,S,V=torch.pca_lowrank(Wcpu,q=1,center=True); vals=U[:,0]
        else: U,S,V=torch.pca_lowrank(Wcpu.t(),q=1,center=True); vals=U[:,0]
        perm=torch.argsort(vals,dim=0); inv=torch.empty_like(perm); inv[perm]=torch.arange(perm.numel(),dtype=perm.dtype)
        return perm.to(W_2d.device),inv.to(W_2d.device)
    def _haar_comp_score(self,X_2d:torch.Tensor,axis:int,L:int,J:int,perm:torch.Tensor=None,frac:float=0.10):
        Xp=X_2d.index_select(axis,perm) if perm is not None else X_2d
        B,meta=blockify_1d(Xp,L=L,dim=axis,J=J);cA,d=haar1d_fwd_vec(B,J)
        C=[cA]+d;C=torch.cat([t.reshape(t.shape[0],-1).abs_()**2 for t in C],1)
        k=max(1,int(math.ceil(frac*C.shape[1])));topk,_=torch.topk(C,k,dim=1,largest=True,sorted=False)
        return (topk.sum(1)/(C.sum(1)+1e-12)).mean().item()
    def _prepare_pa_for_param(self,name:str,W:torch.Tensor):
        if W.ndim<2: return
        M,N=W.shape[-2],W.shape[-1];W2d=W.detach().reshape(M,N)
        pr,ir=self._pca_seriation(W2d,0);pc,ic=self._pca_seriation(W2d,1)
        s_out=self._haar_comp_score(W2d,-2,self.cfg.transform.tile_h,self.cfg.transform.dwt_levels,perm=pr,frac=0.10)
        s_in =self._haar_comp_score(W2d,-1,self.cfg.transform.tile_w,self.cfg.transform.dwt_levels,perm=pc,frac=0.10)
        if s_in>=s_out: entry={"axis":"in","perm":pc,"inv":ic,"dim":-1,"L":self.cfg.transform.tile_w,"scores":(s_in,s_out)}
        else: entry={"axis":"out","perm":pr,"inv":ir,"dim":-2,"L":self.cfg.transform.tile_h,"scores":(s_in,s_out)}
        self._pa_cache[name]=entry;return entry
    def _pick_L_eff(self,T:int,L:int,J:int):
        m=1<<J
        if T<m: return m
        Le=min(L,(T//m)*m);return max(m,Le)
    def _trust_blend(self,E:torch.Tensor,rec_E:torch.Tensor,name:str,thresh:float):
        diff=(rec_E-E).float();den=(E.float().pow(2).mean()+1e-12);rel=(diff.pow(2).mean()/den).item()
        if not math.isfinite(rel) or rel<=thresh: return rec_E
        try: cos=F.cosine_similarity(E.view(1,-1).float(),rec_E.view(1,-1).float(),dim=1).item()
        except: cos=1.0
        cos=max(0.0,float(cos));alpha=min(1.0,max(0.15,0.5*cos/math.sqrt(rel/max(thresh,1e-12))));return alpha*rec_E+(1.0-alpha)*E
    def _maybe_prune_and_quantize_momentum(self,st:Dict[str,Any],E:torch.Tensor,rec_E:torch.Tensor):
        fm=(rec_E.abs()>self.nz_eps).to(device=E.device,dtype=E.dtype)
        if "exp_avg" in st and isinstance(st["exp_avg"],torch.Tensor):
            d=(rec_E-E).abs().view(-1);thr=torch.quantile(d,0.90);ch=(d.view_as(rec_E)>=thr)
            st["exp_avg"][ch]=0
            if "exp_avg_sq" in st and isinstance(st["exp_avg_sq"],torch.Tensor): st["exp_avg_sq"][ch]=0
        for k in ["exp_avg","exp_avg_sq"]:
            if k in st and isinstance(st[k],torch.Tensor):
                mom=st[k].to(device=E.device,dtype=E.dtype)*fm
                if self.cfg.momentum_bitwidth<=0: st[k].copy_(mom); continue
                g=1024;mf=mom.reshape(-1);pad=(g-(mf.numel()%g))%g
                if pad>0: mf=F.pad(mf,(0,pad))
                G=mf.view(-1,g);vmax=torch.quantile(G.abs(),q=self.cfg.momentum_clip_pct,dim=-1,keepdim=True).clamp_min(1e-8);qmax=2**(self.cfg.momentum_bitwidth-1)-1
                qm=torch.clamp(torch.round(G/(vmax/qmax)),min=-qmax,max=qmax);deq=qm*(vmax/qmax)
                st[k].copy_(deq.view(-1)[: mom.numel()].reshape_as(mom))
    def _parse_kv(self,s,cast):
        r={}
        for t in s.split(","):
            if not t: continue
            k,v=t.split(":"); r[k.strip()]=cast(v.strip())
        return r
    def _bits_for_haar1d(self,b_base:int,J:int):
        mp=self._parse_kv(self.cfg.dwt_bits,int);A=mp.get("A",max(5,b_base));D=mp.get("D",max(2,b_base-1));return [A,D]
    def _alloc_for_haar1d(self,cA:torch.Tensor,dets:List[torch.Tensor],mode:str="fixed"):
        if mode=="energy":
            e=[float((cA.float().pow(2)).mean().item())]+[float((d.float().pow(2)).mean().item()) for d in dets]
            w=np.maximum(np.array(e,dtype=np.float64),1e-12);w=w/w.sum();res={"A":float(w[0])}
            for i,_ in enumerate(dets): res[f"D{i+1}"]=float(w[i+1])
            res["D"]=float(1.0-sum([res.get("A",0.0)]));return res
        mp=self._parse_kv(self.cfg.dwt_alloc,float)
        if "A" in mp: res={"A":mp["A"]}
        else: res={"A":0.6}
        if "D" in mp:
            res["D"]=mp["D"]
            for i,_ in enumerate(dets): res[f"D{i+1}"]=mp["D"]/max(1,len(dets))
        else:
            for i,k in enumerate(["LH","HL","HH"]):
                if k in mp: res[f"D{i+1}"]=mp[k]
            if len(dets)>3:
                rest=1.0-res.get("A",0.6)-sum(res.get(f"D{i+1}",0.0) for i in range(3))
                eq=rest/max(1,(len(dets)-3))
                for i in range(3,len(dets)): res[f"D{i+1}"]=eq
        if "D" not in res: res["D"]=float(1.0-res.get("A",0.6))
        return res
