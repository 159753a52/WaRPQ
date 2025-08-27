import os,re,json,math,argparse,random,time
from typing import Dict,Any,Tuple,List
import numpy as np,torch,torch.nn as nn,torch.nn.functional as F
from datasets import load_dataset
from transformers import (AutoTokenizer,AutoModelForCausalLM,DataCollatorForLanguageModeling,TrainingArguments,Trainer,TrainerCallback,EarlyStoppingCallback)
from ckptcodec.hf import WaRPQCallback,CallbackConfig
from ckptcodec.pipeline import TransformSpec,CheckpointCompressor,CodecConfig

def set_seed(s:int):
    os.environ["PYTHONHASHSEED"]=str(s);random.seed(s);np.random.seed(s);torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False

def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).lower()
    if s in ("yes","true","t","y","1"):  return True
    if s in ("no","false","f","n","0"):  return False
    raise argparse.ArgumentTypeError("boolean expected")

class JsonLoggingCallback(TrainerCallback):
    def __init__(self,p): self.p=p; self.h=[]
    def on_train_begin(self,args,state,control,**k):
        os.makedirs(os.path.dirname(self.p),exist_ok=True)
        open(self.p,"w").write("[]")
    def on_log(self,args,state,control,logs=None,**k):
        if logs is None: return
        e={**logs,"step":state.global_step}
        self.h.append(e)
        open(self.p,"w").write(json.dumps(self.h,indent=2))

@torch.no_grad()
def strided_perplexity(raw,tok,model,stride=512):
    dev=next(model.parameters()).device;text="\n\n".join(raw["text"]);enc=tok(text,return_tensors="pt");ids=enc.input_ids.to(dev)
    ml=getattr(model.config,"n_positions",tok.model_max_length);ml=tok.model_max_length if (ml is None or ml>tok.model_max_length) else ml
    nll=0.0;nt=0;prev=0;L=ids.size(1)
    for b in range(0,L,stride):
        e=min(b+ml,L);trg=e-prev;inp=ids[:,b:e];tgt=inp.clone();tgt[:,:-trg]=-100
        out=model(inp,labels=tgt);nv=(tgt!=-100).sum().item();bs=tgt.size(0);nvt=nv-bs;nll+=out.loss*nvt;nt+=nvt;prev=e
        if e==L: break
    a=nll/nt;ppl=torch.exp(a).item();return a.item(),ppl

def _pick_tokenizer_name(a)->str:
    if a.tokenizer_name: return a.tokenizer_name
    if os.path.isdir(a.model_name):
        tok_files=["tokenizer.json","vocab.json","merges.txt","tokenizer_config.json","special_tokens_map.json"]
        if any(os.path.exists(os.path.join(a.model_name,f)) for f in tok_files): return a.model_name
    return "openai-community/gpt2"

@torch.no_grad()
def lm_loss_on_split(raw_split,tok,model,block_size=256,batch_size=2,max_batches=200):
    def _tok(b): return tok(b["text"],return_special_tokens_mask=True)
    tokenized=raw_split.map(_tok,batched=True,remove_columns=["text"]);bs=block_size
    def group(ex):
        c={k:sum(ex[k],[]) for k in ex.keys()};t=(len(c["input_ids"])//bs)*bs
        r={k:[v[i:i+bs] for i in range(0,t,bs)] for k,v in c.items()};r["labels"]=r["input_ids"].copy();return r
    lm_ds=tokenized.map(group,batched=True);coll=DataCollatorForLanguageModeling(tok,mlm=False)
    dev=next(model.parameters()).device;dl=torch.utils.data.DataLoader(lm_ds,batch_size=batch_size,shuffle=False,collate_fn=coll)
    model.eval();total_tokens=0;total_loss=0.0
    for i,batch in enumerate(dl):
        if i>=max_batches: break
        batch={k:v.to(dev) for k,v in batch.items()};out=model(**batch);labels=batch["labels"];nt=int((labels!=-100).sum().item());total_loss+=float(out.loss.item())*nt;total_tokens+=nt
    if total_tokens==0: return float("nan"),float("nan")
    avg_nll=total_loss/total_tokens;return avg_nll,math.exp(avg_nll)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--method",type=str,default="warp_q",choices=["none","warp_q"])
    p.add_argument("--avg_bitwidth",type=float,default=4.0)
    p.add_argument("--target_nz_ratio",type=float,default=0.10)
    p.add_argument("--enforce_target_nz",action="store_true",default=True)

    p.add_argument("--quant_start_step",type=int,default=400)
    p.add_argument("--quant_freq",type=int,default=400)
    p.add_argument("--apply_to_model",action="store_true",default=False)

    p.add_argument("--gop_size",type=int,default=1)
    p.add_argument("--i_frame_scale",type=float,default=1.0)
    p.add_argument("--p_frame_scale",type=float,default=0.8)

    p.add_argument("--qkv_group_size",type=int,default=128)
    p.add_argument("--qkv_clip_pct",type=float,default=0.99)

    p.add_argument("--momentum_bitwidth",type=int,default=8)
    p.add_argument("--momentum_clip_pct",type=float,default=0.999)
    p.add_argument("--m2w_gate",action="store_true",default=True)

    p.add_argument("--ema_decay",type=float,default=0.99)

    p.add_argument("--tile_h",type=int,default=32)
    p.add_argument("--tile_w",type=int,default=32)
    p.add_argument("--dwt_levels",type=int,default=1)
    p.add_argument("--dwt_wave",type=str,default="haar")

    p.add_argument("--save_payload",action="store_true",default=True)
    p.add_argument("--save_dir",type=str,default=None)

    p.add_argument("--use_predictor",action="store_true",default=False)
    p.add_argument("--predictor_beta",type=float,default=0.9)

    p.add_argument("--pi_disable_threshold",type=float,default=0.1)
    p.add_argument("--mask_from",type=str,default="post",choices=["pre","post","coeff"])
    p.add_argument("--rate_mode",type=str,default="nz",choices=["nz","bpp"])
    p.add_argument("--target_bpp",type=float,default=None)
    p.add_argument("--bpp_tol",type=float,default=1e-3)

    p.add_argument("--dwt_alloc_mode",type=str,default="fixed",choices=["fixed","energy"])
    p.add_argument("--dwt_alloc",type=str,default="A:0.6,LH:0.15,HL:0.15,HH:0.1")
    p.add_argument("--dwt_bits",type=str,default="A:5,D:2")
    p.add_argument("--qkv_use_dwt",action="store_true",default=False)

    # ========= 新增：码本量化（CBQ）相关 =========
    # 支持：--use_codebook  或  --use_codebook true/false
    p.add_argument("--use_codebook", type=str2bool, nargs="?", const=True, default=False,
                   help="启用波段码本量化（band codebook quantization）")
    p.add_argument("--cb_bits_min", type=int, default=0, help="码本索引的最小 bit")
    p.add_argument("--cb_bits_max", type=int, default=6, help="码本索引的最大 bit")
    p.add_argument("--cb_A_min",   type=int, default=4, help="低频(Approx)的最小 bit 下限")
    p.add_argument("--coeff_mask_mode", type=str, default="energy", help="小波系数保留策略：energy/topk/none")
    p.add_argument("--cb_sample", type=int, default=50000, help="学习码本的采样点数")
    p.add_argument("--cb_iters",  type=int, default=8, help="码本 KMeans 迭代次数")
    # ========================================

    p.add_argument("--max_steps",type=int,default=-1)
    p.add_argument("--disable_early_stop",action="store_true",default=True)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--model_name",type=str,default="openai-community/gpt2")
    p.add_argument("--output_dir",type=str,default="./runs/exp")
    p.add_argument("--lm_block_size",type=int,default=256)
    p.add_argument("--per_device_train_batch_size",type=int,default=2)
    p.add_argument("--per_device_eval_batch_size",type=int,default=2)
    p.add_argument("--grad_accum",type=int,default=8)
    p.add_argument("--lr",type=float,default=1e-4)
    p.add_argument("--epochs",type=int,default=5)
    p.add_argument("--warmup_ratio",type=float,default=0.03)
    p.add_argument("--weight_decay",type=float,default=0.1)
    p.add_argument("--lr_scheduler_type",type=str,default="cosine")
    p.add_argument("--eval_freq",type=int,default=200)
    p.add_argument("--save_freq",type=int,default=200)
    p.add_argument("--early_stop_patience",type=int,default=3)
    p.add_argument("--strided_stride",type=int,default=512)
    p.add_argument("--single_shot",action="store_true",default=False)
    p.add_argument("--single_shot_step",type=int,default=800)
    p.add_argument("--single_shot_apply",action="store_true",default=True)
    p.add_argument("--single_shot_eval",action="store_true",default=True)
    p.add_argument("--single_shot_residual",action="store_true",default=False)
    p.add_argument("--baseline_path",type=str,default=None)
    p.add_argument("--tokenizer_name",type=str,default=None)
    p.add_argument("--transform",type=str,default="g2lwt",
                   choices=["none","dct2","dwt2","haar1d_in","haar1d_out","pa_adwt","g2lwt"])
    a=p.parse_args()


    set_seed(a.seed)
    ds=load_dataset("wikitext","wikitext-2-raw-v1")
    tok=AutoTokenizer.from_pretrained(_pick_tokenizer_name(a),use_fast=True)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(a.model_name);model.config.pad_token_id=tok.pad_token_id

    def _tok(b): return tok(b["text"],return_special_tokens_mask=True)
    tokenized=ds.map(_tok,batched=True,remove_columns=["text"]);bs=a.lm_block_size
    def group(ex):
        c={k:sum(ex[k],[]) for k in ex.keys()};t=(len(c["input_ids"])//bs)*bs
        r={k:[v[i:i+bs] for i in range(0,t,bs)] for k,v in c.items()};r["labels"]=r["input_ids"].copy();return r
    lm=tokenized.map(group,batched=True);coll=DataCollatorForLanguageModeling(tok,mlm=False)

    ta=TrainingArguments(output_dir=a.output_dir,seed=a.seed,eval_strategy="steps",eval_steps=a.eval_freq,save_strategy="steps",save_steps=a.save_freq,save_total_limit=5,load_best_model_at_end=True,metric_for_best_model="eval_loss",greater_is_better=False,per_device_train_batch_size=a.per_device_train_batch_size,per_device_eval_batch_size=a.per_device_eval_batch_size,gradient_accumulation_steps=a.grad_accum,learning_rate=a.lr,num_train_epochs=a.epochs,max_steps=a.max_steps,warmup_ratio=a.warmup_ratio,weight_decay=a.weight_decay,lr_scheduler_type=a.lr_scheduler_type,fp16=torch.cuda.is_available(),dataloader_num_workers=2,logging_steps=50,logging_first_step=True,report_to="none",save_safetensors=True,group_by_length=True)

    cb = WaRPQCallback(CallbackConfig(
        method=a.method,
        avg_bitwidth=a.avg_bitwidth,
        target_nz_ratio=a.target_nz_ratio,
        enforce_target_nz=a.enforce_target_nz,
        rate_mode=a.rate_mode,
        target_bpp=a.target_bpp,
        bpp_tol=a.bpp_tol,
        quant_start_step=a.quant_start_step,
        quant_freq=a.quant_freq,
        gop_size=a.gop_size,
        i_frame_scale=a.i_frame_scale,
        p_frame_scale=a.p_frame_scale,
        apply_to_model=a.apply_to_model,
        save_dir=a.save_dir,
        save_payload=a.save_payload,
        transform=TransformSpec(kind=a.transform, tile_h=a.tile_h, tile_w=a.tile_w, dwt_levels=a.dwt_levels),
        qkv_group_size=a.qkv_group_size,
        qkv_clip_pct=a.qkv_clip_pct,
        momentum_bitwidth=a.momentum_bitwidth,
        momentum_clip_pct=a.momentum_clip_pct,
        m2w_gate=a.m2w_gate,
        pi_disable_threshold=a.pi_disable_threshold,
        mask_from=a.mask_from,
        qkv_use_dwt=a.qkv_use_dwt,
        dwt_alloc_mode=a.dwt_alloc_mode,
        dwt_alloc=a.dwt_alloc,
        dwt_bits=a.dwt_bits,
        # —— 新增：码本量化相关传参 ——
        use_codebook=a.use_codebook,
        cb_bits_min=a.cb_bits_min,
        cb_bits_max=a.cb_bits_max,
        cb_A_min=a.cb_A_min,
        coeff_mask_mode=a.coeff_mask_mode,
        cb_sample=a.cb_sample,
        cb_iters=a.cb_iters
    ))

    cbs=[JsonLoggingCallback(os.path.join(a.output_dir,"training_log.json")),cb]
    if (not a.disable_early_stop) and a.early_stop_patience>0: cbs.append(EarlyStoppingCallback(early_stopping_patience=a.early_stop_patience,early_stopping_threshold=0.0))

    if a.single_shot:
        set_seed(a.seed);dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tok=AutoTokenizer.from_pretrained(_pick_tokenizer_name(a),use_fast=True)
        if tok.pad_token is None: tok.pad_token=tok.eos_token
        model=AutoModelForCausalLM.from_pretrained(a.model_name);model.config.pad_token_id=tok.pad_token_id;model.to(dev).eval()
        if a.single_shot_eval:
            tr_nll0,tr_ppl0=lm_loss_on_split(ds["train"],tok,model,block_size=a.lm_block_size,batch_size=a.per_device_eval_batch_size,max_batches=200)
            print(f"[OneShot] TRAIN BEFORE  loss={tr_nll0:.4f}, ppl={tr_ppl0:.2f}")
        engine = CheckpointCompressor(CodecConfig(
            avg_bitwidth=a.avg_bitwidth,
            target_nz_ratio=a.target_nz_ratio,
            enforce_target_nz=a.enforce_target_nz,
            rate_mode=a.rate_mode,
            target_bpp=a.target_bpp,
            bpp_tol=a.bpp_tol,
            qkv_group_size=a.qkv_group_size,
            qkv_clip_pct=a.qkv_clip_pct,
            momentum_bitwidth=0,  # one-shot 默认不改动量，如需可改回 a.momentum_bitwidth
            momentum_clip_pct=a.momentum_clip_pct,
            m2w_gate=a.m2w_gate,
            pi_disable_threshold=a.pi_disable_threshold,
            mask_from=a.mask_from,
            transform=TransformSpec(kind=a.transform, tile_h=a.tile_h, tile_w=a.tile_w, dwt_levels=a.dwt_levels),
            qkv_use_dwt=a.qkv_use_dwt,
            dwt_alloc_mode=a.dwt_alloc_mode,
            dwt_alloc=a.dwt_alloc,
            dwt_bits=a.dwt_bits,
            # —— 新增：码本量化相关传参 ——
            use_codebook=a.use_codebook,
            cb_bits_min=a.cb_bits_min,
            cb_bits_max=a.cb_bits_max,
            cb_A_min=a.cb_A_min,
            coeff_mask_mode=a.coeff_mask_mode,
            cb_sample=a.cb_sample,
            cb_iters=a.cb_iters
        ))
        engine.prepare(model,optimizer=None)
        if a.single_shot_residual and a.baseline_path and os.path.exists(a.baseline_path):
            bl=torch.load(a.baseline_path,map_location="cpu")
            engine.baseline_state={k:bl.get(k,torch.zeros_like(v,device="cpu")).to(dtype=v.dtype).cpu() for k,v in model.state_dict().items()}
            print(f"[OneShot] Loaded baseline from: {a.baseline_path}")
        else:
            engine.baseline_state={k:torch.zeros_like(v).cpu() for k,v in model.state_dict().items()}
        stats=engine.compress_step(model,step=a.single_shot_step,optimizer=None,apply_to_model=a.single_shot_apply,save_dir=a.output_dir);print(f"[OneShot] stats: {stats}")
        if a.single_shot_eval:
            tr_nll1,tr_ppl1=lm_loss_on_split(ds["train"],tok,model,block_size=a.lm_block_size,batch_size=a.per_device_eval_batch_size,max_batches=200)
            print(f"[OneShot] TRAIN AFTER   loss={tr_nll1:.4f}, ppl={tr_ppl1:.2f}")
            print(f"[OneShot] Δtrain_loss={tr_nll1-tr_nll0:+.4f}, Δtrain_ppl={tr_ppl1-tr_ppl0:+.2f}")
        if a.single_shot_apply:
            out=os.path.join(a.output_dir,"oneshot_compressed");os.makedirs(out,exist_ok=True);model.save_pretrained(out,safe_serialization=True);tok.save_pretrained(out);print(f"[OneShot] Compressed model saved to {out}")
        return

    tr=Trainer(model=model,args=ta,train_dataset=lm["train"],eval_dataset=lm["validation"],data_collator=coll,callbacks=cbs)
    print("--- Starting Training (GPT-2 on WikiText-2, method:",a.method,") ---");tr.train()
    mv=tr.evaluate();ev=mv["eval_loss"];print(f"[Trainer.evaluate] valid: eval_loss={ev:.4f}, ppl={math.exp(ev):.2f}")
    model.eval();nll,ppl=strided_perplexity(ds["validation"],tok,model,stride=a.strided_stride);print(f"[Strided] valid: avg_nll={nll:.4f}, ppl={ppl:.2f}")
    if "test" in ds:
        nllt,pplt=strided_perplexity(ds["test"],tok,model,stride=a.strided_stride);print(f"[Strided]  test: avg_nll={nllt:.4f}, ppl={pplt:.2f}")
    out=os.path.join(a.output_dir,"best_model_final");tr.save_model(out);print(f"--- Best model saved to {out} ---")

if __name__=="__main__": main()


#可拆分版本，为之后的扩展实验做准备 跑了一下实际压缩率，不是很高，只有三十多倍的压缩
# python train_gpt2_wikitext2_3.py \
#   --method warp_q --transform g2lwt --dwt_levels 2 --tile_w 128 --tile_h 128 \
#   --qkv_use_dwt --dwt_alloc_mode energy --dwt_bits "A:6,D:2" \
#   --avg_bitwidth 4.0 --target_nz_ratio 0.07 --pi_disable_threshold 0.20 \
#   --quant_start_step 400 --quant_freq 400 --apply_to_model \
#   --output_dir runs/warpq_g2lwt_train \
#   --use_codebook true --cb_bits_min 0 --cb_bits_max 6 --cb_A_min 4 \
#   --coeff_mask_mode energy --cb_sample 50000 --cb_iters 8