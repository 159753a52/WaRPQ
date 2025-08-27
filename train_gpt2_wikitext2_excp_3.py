import os,io,re,lzma,json,math,time,argparse,random,base64
from dataclasses import dataclass
from typing import Dict,Any,List,Tuple,Optional
import numpy as np,torch,torch.nn as nn,torch.nn.functional as F
from torch.optim import Optimizer
from datasets import load_dataset
from transformers import (AutoTokenizer,AutoModelForCausalLM,DataCollatorForLanguageModeling,TrainingArguments,Trainer,TrainerCallback,TrainerControl,TrainerState,EarlyStoppingCallback)
try:
    import faiss
    _HAS_FAISS=True
except Exception:
    _HAS_FAISS=False

def _json_sanitize(o):
    if isinstance(o,(bytes,bytearray)): return base64.b64encode(o).decode("ascii")
    if isinstance(o,torch.Tensor): return o.detach().cpu().tolist()
    if isinstance(o,np.ndarray): return o.tolist()
    if isinstance(o,dict): return {k:_json_sanitize(v) for k,v in o.items()}
    if isinstance(o,(list,tuple)): return [_json_sanitize(v) for v in o]
    return o

def set_seed(s:int):
    os.environ["PYTHONHASHSEED"]=str(s);random.seed(s);np.random.seed(s);torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False

def _ok(n:str,p:torch.Tensor)->bool:
    return p.requires_grad  # 包含LN、bias、embedding等

def _H(p:float)->float: return 0.0 if p<=0.0 or p>=1.0 else (-p*math.log2(p)-(1-p)*math.log2(1-p))
def _pick_tokenizer_name(a)->str:
    if getattr(a,"tokenizer_name",None): return a.tokenizer_name
    if os.path.isdir(a.model_name):
        for f in ["tokenizer.json","vocab.json","merges.txt","tokenizer_config.json","special_tokens_map.json"]:
            if os.path.exists(os.path.join(a.model_name,f)): return a.model_name
    return "openai-community/gpt2"

@torch.no_grad()
def lm_loss_on_split(raw_split,tok,model,block_size=256,batch_size=2,max_batches=200):
    def _tok(b): return tok(b["text"],return_special_tokens_mask=True)
    tokenized=raw_split.map(_tok,batched=True,remove_columns=["text"])
    bs=block_size
    def group(ex):
        c={k:sum(ex[k],[]) for k in ex.keys()}
        t=(len(c["input_ids"])//bs)*bs
        r={k:[v[i:i+bs] for i in range(0,t,bs)] for k,v in c.items()}
        r["labels"]=r["input_ids"].copy();return r
    lm_ds=tokenized.map(group,batched=True)
    coll=DataCollatorForLanguageModeling(tok,mlm=False)
    dev=next(model.parameters()).device
    dl=torch.utils.data.DataLoader(lm_ds,batch_size=batch_size,shuffle=False,collate_fn=coll)
    model.eval();total_tokens=0;total_loss=0.0
    for i,b in enumerate(dl):
        if i>=max_batches: break
        b={k:v.to(dev) for k,v in b.items()}
        out=model(**b);labels=b["labels"];nt=int((labels!=-100).sum().item())
        total_loss+=float(out.loss.item())*nt;total_tokens+=nt
    if total_tokens==0: return float("nan"),float("nan")
    avg_nll=total_loss/total_tokens;return avg_nll,math.exp(avg_nll)

def sizeof_fmt(num:float,suffix:str="B")->str:
    for u in ["","K","M","G","T","P","E","Z"]:
        if abs(num)<1024.0: return f"{num:3.1f}{u}{suffix}"
        num/=1024.0
    return f"{num:.1f}Y{suffix}"

def robust_quantile(x:torch.Tensor,q:float,sample_cap:int=2_000_000)->torch.Tensor:
    xf=x.reshape(-1);n=xf.numel()
    if n==0: return torch.tensor(0.0,device=x.device,dtype=x.dtype)
    if n<=sample_cap:
        k=max(1,int(math.ceil(q*n)));return torch.kthvalue(xf,k).values
    idx=torch.randint(low=0,high=n,size=(sample_cap,),device=x.device);xs=xf[idx]
    k=max(1,int(math.ceil(q*sample_cap)));return torch.kthvalue(xs,k).values

def _pack_nibbles(idx:torch.Tensor)->bytes:
    assert idx.dtype==torch.uint8
    if idx.numel()%2==1: idx=torch.cat([idx,torch.zeros(1,dtype=torch.uint8,device=idx.device)])
    hi=idx.view(-1,2)[:,0];lo=idx.view(-1,2)[:,1];packed=(hi<<4)|lo
    return bytes(packed.cpu().numpy().tobytes())

def _unpack_nibbles(buf:bytes,total:int)->torch.Tensor:
    a=np.frombuffer(buf,dtype=np.uint8);a=torch.from_numpy(a.copy())
    hi=(a>>4)&0xF;lo=a&0xF;idx=torch.stack([hi,lo],dim=1).view(-1)
    if idx.numel()>total: idx=idx[:total]
    return idx

@dataclass
class QuantizedTensor:
    shape:Tuple[int,...];centers:torch.Tensor;packed_idx:bytes;nbits:int
    def to_bytes(self)->Dict[str,Any]:
        ctr_b=self.centers.detach().cpu().numpy().astype(np.float16).tobytes()
        return {"shape":list(self.shape),"K":int(self.centers.numel()),"nbits":int(self.nbits),"centers_dtype":"float16","centers_b64":base64.b64encode(ctr_b).decode("ascii"),"packed_idx_b64":base64.b64encode(self.packed_idx).decode("ascii")}
    @staticmethod
    def from_bytes(d:Dict[str,Any],device:torch.device)->"QuantizedTensor":
        shape=tuple(d["shape"]);K=int(d["K"])
        ctr=np.frombuffer(base64.b64decode(d["centers_b64"]),dtype=np.float16).astype(np.float32)
        centers=torch.from_numpy(ctr).to(device).view(K);packed=base64.b64decode(d["packed_idx_b64"])
        return QuantizedTensor(shape=shape,centers=centers,packed_idx=packed,nbits=int(d["nbits"]))
    def dequantize(self,device:torch.device)->torch.Tensor:
        total=int(np.prod(self.shape));idx=_unpack_nibbles(self.packed_idx,total).to(device)
        out=torch.zeros(total,device=device,dtype=torch.float32);nz=idx>0
        if nz.any(): out[nz]=self.centers[(idx[nz]-1).long()]
        return out.view(*self.shape)

def _raw_to_bytes(x:torch.Tensor)->Dict[str,Any]:
    b=x.detach().cpu().numpy().astype(np.float16).tobytes()
    return {"shape":list(x.shape),"raw_dtype":"float16","raw_b64":base64.b64encode(b).decode("ascii"),"raw":True}

def _raw_from_bytes(d:Dict[str,Any],device:torch.device)->torch.Tensor:
    sh=tuple(d["shape"]);arr=np.frombuffer(base64.b64decode(d["raw_b64"]),dtype=np.float16).astype(np.float32)
    return torch.from_numpy(arr).to(device).view(*sh)

def _kmeans1d_faiss(vals:torch.Tensor,K:int,max_iters:int)->torch.Tensor:
    x=vals.detach().cpu().view(-1,1).numpy().astype(np.float32)
    km=faiss.Kmeans(d=1, k=K, niter=max_iters, gpu=False,
                      min_points_per_centroid=1,  # 关键
                      verbose=False, seed=123)
    km.train(x)
    _,I=km.index.search(x,1)
    centers=torch.from_numpy(km.centroids.copy()).to(vals.device).view(K).to(torch.float32)
    labels=torch.from_numpy(I.astype(np.int64).reshape(-1)).to(vals.device)
    return centers,labels

def _kmeans1d_bucketize(vals:torch.Tensor,K:int,max_iters:int)->torch.Tensor:
    device=vals.device;centers=torch.linspace(vals.quantile(0.005),vals.quantile(0.995),K,device=device,dtype=torch.float32)
    for _ in range(max_iters):
        b=(centers[1:]+centers[:-1])/2
        labels=torch.bucketize(vals,b)
        counts=torch.bincount(labels,minlength=K)
        sums=torch.bincount(labels,weights=vals,minlength=K)
        nonempty=counts>0
        newc=centers.clone()
        newc[nonempty]=sums[nonempty]/counts[nonempty]
        empty=~nonempty
        if empty.any():
            lo,hi=vals.min(),vals.max()
            newc[empty]=torch.linspace(lo,hi,int(empty.sum().item()),device=device)
        if torch.allclose(newc,centers,atol=1e-6,rtol=0): centers=newc;break
        centers=newc
    b=(centers[1:]+centers[:-1])/2
    labels=torch.bucketize(vals,b)
    return centers,labels

def kmeans_nonuniform_quantize(x:torch.Tensor,nbits:int,max_iters:int=12)->QuantizedTensor:
    assert nbits>=2
    flat=x.reshape(-1).to(torch.float32);device=flat.device
    nz_mask=flat!=0;nz=flat[nz_mask];K=(1<<nbits)-1
    if nz.numel()==0:
        centers=torch.zeros(K,device=device,dtype=torch.float32);idx=torch.zeros_like(flat,dtype=torch.uint8);return QuantizedTensor(tuple(x.shape),centers,_pack_nibbles(idx),nbits)
    if nz.numel()<=K: return QuantizedTensor(tuple(x.shape),torch.zeros(0,device=device),_pack_nibbles(torch.zeros_like(flat,dtype=torch.uint8)),nbits if False else nbits).__class__(tuple(x.shape),torch.zeros(0,device=device),_pack_nibbles(torch.zeros_like(flat,dtype=torch.uint8)),nbits) if False else QuantizedTensor(tuple(x.shape),torch.zeros(0,device=device),_pack_nibbles(torch.zeros_like(flat,dtype=torch.uint8)),nbits)  # placeholder to satisfy type checkers
    if nz.numel()<=K:  # 少量非零直接存原值
        qt=QuantizedTensor(tuple(x.shape),torch.zeros(0,device=device),b"",nbits)
        qt.to_bytes=lambda: _raw_to_bytes(x)  # type: ignore
        qt.dequantize=lambda device: x.to(device).to(torch.float32)  # type: ignore
        return qt
    if _HAS_FAISS:
        centers,labels=_kmeans1d_faiss(nz,K,max_iters)
    else:
        centers,labels=_kmeans1d_bucketize(nz,K,max_iters)
    idx=torch.zeros_like(flat,dtype=torch.uint8)
    idx[nz_mask]=(labels.to(torch.int64)+1).to(torch.uint8)
    return QuantizedTensor(tuple(x.shape),centers,_pack_nibbles(idx),nbits)

class JsonLoggingCallback(TrainerCallback):
    def __init__(self,p): self.p=p; self.buf=[]
    def on_train_begin(self,args,state,control,**k):
        os.makedirs(os.path.dirname(self.p),exist_ok=True); open(self.p,"w").write("[]")
    def on_log(self,args,state,control,logs=None,**k):
        if logs is None: return
        e={**logs,"step":int(state.global_step)}; self.buf.append(e); open(self.p,"w").write(json.dumps(self.buf,indent=2))

@torch.no_grad()
def project_topk_tensor(x:torch.Tensor,p_keep:float):
    if p_keep>=1.0: return x,torch.ones_like(x,dtype=torch.bool)
    if p_keep<=0.0: return torch.zeros_like(x),torch.zeros_like(x,dtype=torch.bool)
    flat=x.abs().reshape(-1);k=max(1,int(round(p_keep*flat.numel())))
    if k>=flat.numel(): m=torch.ones_like(x,dtype=torch.bool);return x,m
    thr=torch.kthvalue(flat,flat.numel()-k+1).values;m=(x.abs()>=thr);return x*m,m

class ExCPCheckpointer(TrainerCallback):
    def __init__(self,cfg:"ExCPConfig"):
        self.cfg=cfg; self._last_reconstructed:Dict[str,torch.Tensor]={}; self._step0_saved=False
    @staticmethod
    def _find_group_for_param(opt:Optional[Optimizer],param:torch.Tensor):
        if opt is None: return None
        for g in opt.param_groups:
            for p in g.get("params",[]):
                if p is param: return g
        return None
    def _ensure_dirs(self):
        os.makedirs(self.cfg.save_dir,exist_ok=True); os.makedirs(os.path.join(self.cfg.save_dir,"ckpts"),exist_ok=True)
    def on_train_begin(self,args,state:TrainerState,control:TrainerControl,model=None,**k):
        self._ensure_dirs()
        if self.cfg.keep_init_state and not self._step0_saved:
            init_path=os.path.join(self.cfg.save_dir,"init_weights.pt")
            if not os.path.exists(init_path):
                torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},init_path)
            self._last_reconstructed={k:v.detach().to(next(model.parameters()).device) for k,v in model.state_dict().items()}
            self._step0_saved=True
        meta={"nbits":self.cfg.nbits,"alpha":self.cfg.alpha,"beta":self.cfg.beta,"compress_every":self.cfg.compress_every,"start_step":self.cfg.start_step}
        open(os.path.join(self.cfg.save_dir,"meta.json"),"w").write(json.dumps(meta,indent=2))
    def on_step_end(self,args:TrainingArguments,state:TrainerState,control:TrainerControl,model=None,optimizer:Optional[Optimizer]=None,**k):
        step=int(state.global_step)
        if step<self.cfg.start_step: return
        if step%self.cfg.compress_every!=0: return
        self._compress_now(step,model,optimizer)
    @torch.no_grad()
    def _compress_now(self,step:int,model:nn.Module,optimizer:Optional[Optimizer]):
        t0=time.perf_counter();dev=next(model.parameters()).device
        if not self._last_reconstructed: self._last_reconstructed={k:v.detach().to(dev) for k,v in model.state_dict().items()}
        name_to_param=dict(model.named_parameters()); id2name={id(p):n for n,p in name_to_param.items()}
        optim_states:Dict[str,Dict[str,torch.Tensor]]={}
        if optimizer is not None:
            for g in optimizer.param_groups:
                for p in g.get("params",[]):
                    n=id2name.get(id(p)); 
                    if n is None: continue
                    st=optimizer.state.get(p,{})
                    m=st.get("exp_avg",None); v=st.get("exp_avg_sq",None)
                    if isinstance(m,torch.Tensor): optim_states.setdefault(n,{})["m"]=m.detach().to(dev).float()
                    if isinstance(v,torch.Tensor): optim_states.setdefault(n,{})["v"]=v.detach().to(dev).float()
        payload_w:Dict[str,Any]={}; payload_o:Dict[str,Any]={}
        total_params=0; kept_params=0
        for name,p_obj in name_to_param.items():
            if not _ok(name,p_obj): continue
            W=p_obj.detach().to(dev).float(); W_last=self._last_reconstructed.get(name,W.clone())
            dW=(W-W_last); st=optim_states.get(name,{})
            if self.cfg.enforce_target:
                dWq,keep_mask=project_topk_tensor(dW,self.cfg.target_nz)
            else:
                v_mean=float(st.get("v",torch.zeros(1,device=dev)).mean().item()) if "v" in st else 0.0
                med=float(dW.abs().median().item())
                scale=min(2.5,self.cfg.alpha/math.sqrt(v_mean+1e-12))
                thr=med*scale
                keep_mask=(dW.abs()>thr); dWq=dW*keep_mask
            bits=int(self.cfg.qkv_bits or self.cfg.nbits)
            qobj=kmeans_nonuniform_quantize(dWq,nbits=bits,max_iters=int(self.cfg.kmeans_iters))
            rec_dW=qobj.dequantize(device=dev)
            payload_w[name]=qobj.to_bytes() if qobj.centers.numel()>0 or qobj.packed_idx!=b"" else _raw_to_bytes(dWq)
            if "m" in st:
                m1=st["m"]; mo_thr=self.cfg.beta*float(m1.abs().mean().item()); Mo=(m1.abs()>mo_thr)&keep_mask
                m1p=(m1*Mo).to(torch.float32); qt_m=kmeans_nonuniform_quantize(m1p,nbits=int(self.cfg.nbits),max_iters=int(self.cfg.kmeans_iters)); payload_o.setdefault(name,{})["m"]=qt_m.to_bytes() if qt_m.centers.numel()>0 or qt_m.packed_idx!=b"" else _raw_to_bytes(m1p)
                if "v" in st and isinstance(st["v"],torch.Tensor):
                    v2p=(st["v"]*Mo).to(torch.float32); qt_v=kmeans_nonuniform_quantize(v2p,nbits=int(self.cfg.nbits),max_iters=int(self.cfg.kmeans_iters)); payload_o[name]["v"]=qt_v.to_bytes() if qt_v.centers.numel()>0 or qt_v.packed_idx!=b"" else _raw_to_bytes(v2p)
            W_next=(W_last+rec_dW).to(torch.float32); self._last_reconstructed[name]=W_next.clone()
            if self.cfg.apply_to_model: p_obj.data.copy_(W_next.to(p_obj.dtype))
            if self.cfg.apply_to_model and optimizer is not None and self.cfg.momentum_policy=="reset_changed":
                st_=optimizer.state.get(p_obj,{})
                if "exp_avg" in st_ and isinstance(st_["exp_avg"],torch.Tensor):
                    delta=(rec_dW-dW).abs(); thr=robust_quantile(delta,0.90,sample_cap=2_000_000); ch=(delta>=thr); st_["exp_avg"].masked_fill_(ch,0)
                    if "exp_avg_sq" in st_ and isinstance(st_["exp_avg_sq"],torch.Tensor): st_["exp_avg_sq"].masked_fill_(ch,0)
            n=dW.numel(); kk=int(keep_mask.sum().item()); total_params+=n; kept_params+=kk
        pt={"step":int(step),"nbits":self.cfg.nbits,"alpha":self.cfg.alpha,"beta":self.cfg.beta,"target_nz":self.cfg.target_nz,"enforce_target":self.cfg.enforce_target,"weights":payload_w,"optimizer":payload_o}
        raw=json.dumps(pt).encode("utf-8"); comp=lzma.compress(raw,preset=9|lzma.PRESET_EXTREME); out=os.path.join(self.cfg.save_dir,"ckpts",f"excp_{step:08d}.xz")
        os.makedirs(os.path.dirname(out),exist_ok=True); open(out,"wb").write(comp)
        p_keep=kept_params/max(1,total_params); bpp=p_keep*(self.cfg.nbits)+_H(p_keep); cr=32.0/max(bpp,1e-9); t1=time.perf_counter()
        stats={"step":int(step),"bytes":len(comp),"human":sizeof_fmt(len(comp)),"p_keep":float(p_keep),"bpp":float(bpp),"cr_weight":float(cr),"time_s":(t1-t0)}
        open(os.path.join(self.cfg.save_dir,"compress_stats.jsonl"),"a").write(json.dumps(stats)+"\n")
        print(f"[ExCP] step={step} | keep={p_keep*100:.2f}% | bpp={bpp:.3f} | CR≈{cr:.1f}x | wrote {sizeof_fmt(len(comp))} in {t1-t0:.2f}s")

@torch.no_grad()
def excp_reconstruct_to_state_dict(compressed_dir:str,upto_step:Optional[int]=None,init_weights_path:Optional[str]=None,device:Optional[torch.device]=None)->Dict[str,torch.Tensor]:
    assert init_weights_path is not None and os.path.exists(init_weights_path)
    if device is None: device=torch.device("cpu")
    base=torch.load(init_weights_path,map_location="cpu"); recon={k:v.clone().to(device) for k,v in base.items()}
    xs=sorted([x for x in os.listdir(os.path.join(compressed_dir,"ckpts")) if x.endswith(".xz")])
    for fn in xs:
        m=re.match(r"excp_(\d+)\.xz",fn); 
        if not m: continue
        step=int(m.group(1))
        if upto_step is not None and step>upto_step: break
        d=json.loads(lzma.decompress(open(os.path.join(compressed_dir,"ckpts",fn),"rb").read()).decode("utf-8"))
        for name,qd in d["weights"].items():
            if isinstance(qd,dict) and qd.get("raw",False):
                recon[name]=recon[name].to(torch.float32); recon[name]+=_raw_from_bytes(qd,device=device)
            else:
                qt=QuantizedTensor.from_bytes(qd,device=device); recon[name]=recon[name].to(torch.float32); recon[name]+=qt.dequantize(device=device)
    return recon

@torch.no_grad()
def excp_reconstruct_optimizer_state(compressed_dir:str,upto_step:Optional[int]=None,init_weights_path:Optional[str]=None,device:Optional[torch.device]=None)->Dict[str,Dict[str,torch.Tensor]]:
    assert init_weights_path is not None and os.path.exists(init_weights_path)
    if device is None: device=torch.device("cpu")
    base=torch.load(init_weights_path,map_location="cpu")
    opt_m={k:torch.zeros_like(v,device=device,dtype=torch.float32) for k,v in base.items()}
    opt_v={k:torch.zeros_like(v,device=device,dtype=torch.float32) for k,v in base.items()}
    xs=sorted([x for x in os.listdir(os.path.join(compressed_dir,"ckpts")) if x.endswith(".xz")])
    for fn in xs:
        m=re.match(r"excp_(\d+)\.xz",fn)
        if not m: continue
        step=int(m.group(1))
        if upto_step is not None and step>upto_step: break
        d=json.loads(lzma.decompress(open(os.path.join(compressed_dir,"ckpts",fn),"rb").read()).decode("utf-8"))
        opt=d.get("optimizer",{})
        for name,od in opt.items():
            if "m" in od:
                if isinstance(od["m"],dict) and od["m"].get("raw",False): opt_m[name]=_raw_from_bytes(od["m"],device=device)
                else: qm=QuantizedTensor.from_bytes(od["m"],device=device); opt_m[name]=qm.dequantize(device=device)
            if "v" in od:
                if isinstance(od["v"],dict) and od["v"].get("raw",False): opt_v[name]=_raw_from_bytes(od["v"],device=device)
                else: qv=QuantizedTensor.from_bytes(od["v"],device=device); opt_v[name]=qv.dequantize(device=device)
    return {k:{"m":opt_m.get(k,None),"v":opt_v.get(k,None)} for k in base.keys()}

@torch.no_grad()
def strided_perplexity(raw,tok,model,stride=512):
    dev=next(model.parameters()).device;text="\n\n".join(raw["text"]);enc=tok(text,return_tensors="pt");ids=enc.input_ids.to(dev)
    ml=getattr(model.config,"n_positions",tok.model_max_length); ml=tok.model_max_length if (ml is None or ml>tok.model_max_length) else ml
    nll=0.0;nt=0;prev=0;L=ids.size(1)
    for b in range(0,L,stride):
        e=min(b+ml,L);trg=e-prev;inp=ids[:,b:e];tgt=inp.clone();tgt[:,:-trg]=-100
        out=model(inp,labels=tgt);nv=(tgt!=-100).sum().item();nll+=out.loss*nv;nt+=nv;prev=e
        if e==L: break
    a=nll/nt;ppl=torch.exp(a).item();return a.item(),ppl

def _layer_group(name:str)->str:
    l=name.lower()
    if ("c_attn.weight" in l) or ("attention.query_key_value.weight" in l) or ("q_proj" in l) or ("k_proj" in l) or ("v_proj" in l): return "attn_qkv"
    if ("attn.c_proj.weight" in l) or ("attention.out_proj.weight" in l) or ("o_proj" in l): return "attn_out"
    if (".mlp.c_fc.weight" in l) or ("mlp.fc_in.weight" in l) or ("gate_proj" in l) or ("up_proj" in l): return "mlp_fc"
    if (".mlp.c_proj.weight" in l) or ("mlp.fc_out.weight" in l) or ("down_proj" in l): return "mlp_proj"
    return "others"

def _parse_group_keep_spec(spec:str)->Dict[str,float]:
    out={}
    if not spec: return out
    for it in spec.split(","):
        it=it.strip()
        if not it: continue
        k,v=it.split("="); out[k.strip()]=float(v)
    return out

@torch.no_grad()
def global_threshold_T(model:nn.Module,baseline:Dict[str,torch.Tensor],opt_moments:Dict[str,Dict[str,torch.Tensor]],target_keep:float,sample_cap:int=5_000_000)->float:
    pool=[]; total=0
    for name,p in model.named_parameters():
        if not _ok(name,p): continue
        W=p.data.detach(); B=baseline.get(name,torch.zeros_like(W,device="cpu")).to(W.device,dtype=W.dtype); E=(W-B).float()
        v_mean=float(opt_moments.get(name,{}).get("v",torch.zeros(1,device=W.device)).mean().item()) if name in opt_moments and ("v" in opt_moments[name]) else 1.0
        z=E.abs()*math.sqrt(v_mean+1e-12); flat=z.reshape(-1); total+=flat.numel()
        take=min(flat.numel(),max(1,sample_cap//max(1,len(pool)+1)))
        if take<flat.numel(): idx=torch.randint(0,flat.numel(),(take,),device=flat.device); pool.append(flat[idx].detach().cpu())
        else: pool.append(flat.detach().cpu())
    if not pool: return 0.0
    Z=torch.cat(pool,dim=0); k=max(1,int(round(target_keep*Z.numel()))); thr=torch.kthvalue(Z,Z.numel()-k+1).values.item(); return float(thr)

@torch.no_grad()
def groupwise_threshold_T(model:nn.Module,baseline:Dict[str,torch.Tensor],opt_moments:Dict[str,Dict[str,torch.Tensor]],default_keep:float,keep_spec:Optional[Dict[str,float]]=None,sample_cap_per_group:int=2_000_000)->Dict[str,float]:
    keep_spec=keep_spec or {}; buckets:Dict[str,List[torch.Tensor]]={}
    for name,p in model.named_parameters():
        if not _ok(name,p): continue
        g=_layer_group(name); W=p.data.detach()
        B=baseline.get(name,torch.zeros_like(W,device="cpu")).to(W.device,dtype=W.dtype); E=(W-B).float()
        v_mean=float(opt_moments.get(name,{}).get("v",torch.zeros(1,device=W.device)).mean().item()) if name in opt_moments and ("v" in opt_moments[name]) else 1.0
        z=E.abs()*math.sqrt(v_mean+1e-12); flat=z.reshape(-1); take=min(sample_cap_per_group,flat.numel())
        if take<flat.numel(): idx=torch.randint(0,flat.numel(),(take,),device=flat.device); s=flat[idx].detach().cpu()
        else: s=flat.detach().cpu()
        buckets.setdefault(g,[]).append(s)
    T:Dict[str,float]={}
    for g,parts in buckets.items():
        s=torch.cat(parts,dim=0)
        if s.numel()==0: T[g]=0.0; continue
        pk=keep_spec.get(g,default_keep); k=max(1,int(round(pk*s.numel()))); thr=torch.kthvalue(s,s.numel()-k+1).values.item(); T[g]=float(thr)
    return T

@dataclass
class ExCPConfig:
    save_dir:str; compress_every:int=400; start_step:int=400; nbits:int=4; alpha:float=5e-5; beta:float=2.0
    keep_init_state:bool=True; include_bias_and_ln:bool=True; target_nz:float=0.10; enforce_target:bool=False
    apply_to_model:bool=True; momentum_policy:str="none"; qkv_bits:Optional[int]=4
    qkv_clip_pct:float=0.99; kmeans_iters:int=12; kmeans_qkv:bool=True

def _is_qkv(name:str)->bool:
    l=name.lower()
    return ("c_attn.weight" in l) or ("attention.query_key_value.weight" in l) or ("q_proj" in l) or ("k_proj" in l) or ("v_proj" in l)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model_name",type=str,default="openai-community/gpt2")
    ap.add_argument("--output_dir",type=str,default="./runs/excp")
    ap.add_argument("--lm_block_size",type=int,default=256)
    ap.add_argument("--per_device_train_batch_size",type=int,default=2)
    ap.add_argument("--per_device_eval_batch_size",type=int,default=2)
    ap.add_argument("--grad_accum",type=int,default=8)
    ap.add_argument("--lr",type=float,default=1e-4)
    ap.add_argument("--epochs",type=int,default=5)
    ap.add_argument("--warmup_ratio",type=float,default=0.03)
    ap.add_argument("--weight_decay",type=float,default=0.1)
    ap.add_argument("--lr_scheduler_type",type=str,default="cosine")
    ap.add_argument("--eval_freq",type=int,default=200)
    ap.add_argument("--save_freq",type=int,default=200)
    ap.add_argument("--early_stop_patience",type=int,default=3)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--max_steps",type=int,default=-1)
    ap.add_argument("--disable_early_stop",action="store_true",default=True)
    ap.add_argument("--excp_save_dir",type=str,default=None)
    ap.add_argument("--excp_start_step",type=int,default=400)
    ap.add_argument("--excp_every",type=int,default=400)
    ap.add_argument("--excp_nbits",type=int,default=4)
    ap.add_argument("--excp_alpha",type=float,default=5e-5)
    ap.add_argument("--excp_beta",type=float,default=2.0)
    ap.add_argument("--excp_keep_init",action="store_true",default=True)
    ap.add_argument("--excp_apply_to_model",action="store_true",default=False)
    ap.add_argument("--excp_momentum_policy",type=str,default="none",choices=["none","reset_changed"])
    ap.add_argument("--excp_target_nz",type=float,default=0.07)
    ap.add_argument("--excp_enforce_target",action="store_true",default=False)
    ap.add_argument("--excp_qkv_bits",type=int,default=4)
    ap.add_argument("--excp_kmeans_iters",type=int,default=12)
    ap.add_argument("--reconstruct_only",action="store_true",default=False)
    ap.add_argument("--reconstruct_upto",type=int,default=None)
    ap.add_argument("--reconstruct_out",type=str,default=None)
    ap.add_argument("--reconstruct_optimizer",action="store_true",default=False)
    ap.add_argument("--reconstruct_opt_out",type=str,default=None)
    ap.add_argument("--single_shot",action="store_true",default=False)
    ap.add_argument("--single_shot_step",type=int,default=800)
    ap.add_argument("--single_shot_apply",action="store_true",default=True)
    ap.add_argument("--single_shot_eval",action="store_true",default=True)
    ap.add_argument("--single_shot_residual",action="store_true",default=True)
    ap.add_argument("--baseline_path",type=str,default=None)
    ap.add_argument("--tokenizer_name",type=str,default=None)
    ap.add_argument("--max_eval_batches",type=int,default=200)
    ap.add_argument("--excp_use_opt_threshold",action="store_true",default=True)
    ap.add_argument("--optimizer_path",type=str,default=None)
    ap.add_argument("--excp_calibrate_keep",type=float,default=None)
    ap.add_argument("--excp_groupwise_calibrate",action="store_true",default=False)
    ap.add_argument("--excp_min_keep_floor",type=float,default=0.02)
    ap.add_argument("--excp_group_keep_spec",type=str,default="")
    a=ap.parse_args(); set_seed(a.seed)
    if a.reconstruct_only:
        assert a.excp_save_dir is not None,"--excp_save_dir required"
        init_path=os.path.join(a.excp_save_dir,"init_weights.pt")
        state=excp_reconstruct_to_state_dict(a.excp_save_dir,upto_step=a.reconstruct_upto,init_weights_path=init_path)
        out=a.reconstruct_out or os.path.join(a.excp_save_dir,f"reconstructed_{a.reconstruct_upto or 'final'}.pt")
        torch.save(state,out); print(f"Reconstructed state_dict saved to: {out}")
        if a.reconstruct_optimizer:
            opt=excp_reconstruct_optimizer_state(a.excp_save_dir,upto_step=a.reconstruct_upto,init_weights_path=init_path)
            out_opt=a.reconstruct_opt_out or os.path.join(a.excp_save_dir,f"reconstructed_optimizer_{a.reconstruct_upto or 'final'}.pt")
            torch.save(opt,out_opt); print(f"Reconstructed optimizer moments saved to: {out_opt}")
        return
    ds=load_dataset("wikitext","wikitext-2-raw-v1")
    tok=AutoTokenizer.from_pretrained(a.model_name,use_fast=True)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(a.model_name); model.config.pad_token_id=tok.pad_token_id
    def _tok(b): return tok(b["text"],return_special_tokens_mask=True)
    tokenized=ds.map(_tok,batched=True,remove_columns=["text"])
    bs=a.lm_block_size
    def group(ex):
        c={k:sum(ex[k],[]) for k in ex.keys()}; t=(len(c["input_ids"])//bs)*bs
        r={k:[v[i:i+bs] for i in range(0,t,bs)] for k,v in c.items()}; r["labels"]=r["input_ids"].copy(); return r
    lm=tokenized.map(group,batched=True); coll=DataCollatorForLanguageModeling(tok,mlm=False)
    ta=TrainingArguments(output_dir=a.output_dir,seed=a.seed,eval_strategy="steps",eval_steps=a.eval_freq,save_strategy="steps",save_steps=a.save_freq,save_total_limit=5,load_best_model_at_end=True,metric_for_best_model="eval_loss",greater_is_better=False,per_device_train_batch_size=a.per_device_train_batch_size,per_device_eval_batch_size=a.per_device_eval_batch_size,gradient_accumulation_steps=a.grad_accum,learning_rate=a.lr,num_train_epochs=a.epochs,max_steps=a.max_steps,warmup_ratio=a.warmup_ratio,weight_decay=a.weight_decay,lr_scheduler_type=a.lr_scheduler_type,fp16=torch.cuda.is_available(),dataloader_num_workers=2,logging_steps=50,logging_first_step=True,report_to="none",save_safetensors=True,group_by_length=True)
    excp_dir=a.excp_save_dir or os.path.join(a.output_dir,"excp")
    cfg=ExCPConfig(save_dir=excp_dir,compress_every=a.excp_every,start_step=a.excp_start_step,nbits=a.excp_nbits,alpha=a.excp_alpha,beta=a.excp_beta,keep_init_state=a.excp_keep_init,apply_to_model=a.excp_apply_to_model,momentum_policy=a.excp_momentum_policy,target_nz=a.excp_target_nz,enforce_target=a.excp_enforce_target,qkv_bits=a.excp_qkv_bits,kmeans_iters=a.excp_kmeans_iters,kmeans_qkv=True)
    excp_cb=ExCPCheckpointer(cfg)
    cbs=[JsonLoggingCallback(os.path.join(a.output_dir,"training_log.json")),excp_cb]
    if (not a.disable_early_stop) and a.early_stop_patience>0: cbs.append(EarlyStoppingCallback(early_stopping_patience=a.early_stop_patience,early_stopping_threshold=0.0))
    if a.single_shot:
        set_seed(a.seed); dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tok=AutoTokenizer.from_pretrained(_pick_tokenizer_name(a),use_fast=True)
        if tok.pad_token is None: tok.pad_token=tok.eos_token
        model=AutoModelForCausalLM.from_pretrained(a.model_name); model.config.pad_token_id=tok.pad_token_id; model.to(dev).eval()
        ds=load_dataset("wikitext","wikitext-2-raw-v1")
        if a.single_shot_eval:
            nll0,ppl0=lm_loss_on_split(ds["train"],tok,model,block_size=a.lm_block_size,batch_size=a.per_device_eval_batch_size,max_batches=a.max_eval_batches)
            print(f"[ExCP OneShot] TRAIN BEFORE  loss={nll0:.4f}, ppl={ppl0:.2f}")
        sd=model.state_dict()
        if a.single_shot_residual:
            if a.baseline_path and os.path.exists(a.baseline_path):
                bl=torch.load(a.baseline_path,map_location="cpu")
                baseline={k:bl.get(k,torch.zeros_like(v,device="cpu")).to(dtype=v.dtype).cpu() for k,v in sd.items()}
                print(f"[ExCP OneShot] Loaded baseline from: {a.baseline_path}")
            else:
                print("[ExCP OneShot] WARN: baseline_path not found; use baseline=0."); baseline={k:torch.zeros_like(v).cpu() for k,v in sd.items()}
        else:
            baseline={k:torch.zeros_like(v).cpu() for k,v in sd.items()}
        step=int(a.single_shot_step)
        kept,total=0,0; opt_moments={}
        src_opt_dir=a.optimizer_path if a.optimizer_path is not None else a.model_name
        if src_opt_dir is not None:
            try:
                obj=torch.load(os.path.join(src_opt_dir,"optimizer.pt"),map_location="cpu"); st=obj.get("state",{}); groups=obj.get("param_groups",[]); opt_ids=[]; [opt_ids.extend(list(g.get("params",[]))) for g in groups]
                names=[n for n,_ in model.named_parameters()]; params=[p for _,p in model.named_parameters()]
                for i,pid in enumerate(opt_ids):
                    if i>=len(params): break
                    name, p = names[i], params[i]; s=st.get(pid,None)
                    if s is None: continue
                    em=s.get("exp_avg",None); ev=s.get("exp_avg_sq",None); ent={}
                    if isinstance(em,torch.Tensor) and em.numel()==p.numel(): ent["m"]=em.to(device=dev,dtype=torch.float32).view_as(p)
                    if isinstance(ev,torch.Tensor) and ev.numel()==p.numel(): ent["v"]=ev.to(device=dev,dtype=torch.float32).view_as(p)
                    if ent: opt_moments[name]=ent
            except Exception as e:
                print(f"[ExCP OneShot] WARN: failed to load optimizer.pt: {e}")
        t0=time.perf_counter()
        with torch.no_grad():
            for name,p in model.named_parameters():
                if not _ok(name,p): continue
                W=p.data.detach().to(dev).float(); B=baseline.get(name,torch.zeros_like(W,device="cpu")).to(dev,dtype=W.dtype); E=W-B
                if a.excp_enforce_target:
                    E_kept,keep_mask=project_topk_tensor(E,a.excp_target_nz)
                else:
                    v_mean=float(opt_moments.get(name,{}).get("v",torch.zeros(1,device=dev)).mean().item()) if name in opt_moments and ("v" in opt_moments[name]) else 0.0
                    med=float(E.abs().median().item()); thr=min(2.5,a.excp_alpha/math.sqrt(v_mean+1e-12))*med
                    keep_mask=(E.abs()>thr); E_kept=E*keep_mask
                b_qkv=int(a.excp_qkv_bits or a.excp_nbits)
                rec_E=kmeans_nonuniform_quantize(E_kept,nbits=b_qkv,max_iters=int(a.excp_kmeans_iters)).dequantize(device=dev)
                W_rec=(B+rec_E).to(W.dtype)
                if a.single_shot_apply: p.data.copy_(W_rec)
                n=E.numel(); total+=n; kept+=int(keep_mask.sum().item())
        p_keep=kept/max(1,total); bpp=p_keep*a.excp_nbits+_H(p_keep); cr=32.0/max(bpp,1e-9); t1=time.perf_counter()
        print(f"[ExCP OneShot] step={step} | keep={p_keep*100:.2f}% | nbits={a.excp_nbits} | bpp={bpp:.3f} | CR≈{cr:.1f}x | {t1-t0:.2f}s")
        if a.single_shot_eval:
            nll1,ppl1=lm_loss_on_split(ds["train"],tok,model,block_size=a.lm_block_size,batch_size=a.per_device_eval_batch_size,max_batches=a.max_eval_batches)
            print(f"[ExCP OneShot] TRAIN AFTER   loss={nll1:.4f}, ppl={ppl1:.2f}")
        if a.single_shot_apply:
            out=os.path.join(a.output_dir,"oneshot_excp"); os.makedirs(out,exist_ok=True); model.save_pretrained(out,safe_serialization=True); tok.save_pretrained(out); print(f"[ExCP OneShot] Compressed model saved to {out}")
        return
    tr=Trainer(model=model,args=ta,train_dataset=lm["train"],eval_dataset=lm["validation"],data_collator=coll,callbacks=cbs)
    print("--- Starting Training (GPT-2 on WikiText-2) with ExCP periodic compression/restore ---"); tr.train()
    mv=tr.evaluate(); ev=mv["eval_loss"]; print(f"[Trainer.evaluate] valid: eval_loss={ev:.4f}, ppl={math.exp(ev):.2f}")
    model.eval(); nll,ppl=strided_perplexity(ds["validation"],tok,model,stride=512); print(f"[Strided] valid: avg_nll={nll:.4f}, ppl={ppl:.2f}")
    if "test" in ds:
        nllt,pplt=strided_perplexity(ds["test"],tok,model,stride=512); print(f"[Strided]  test: avg_nll={nllt:.4f}, ppl={pplt:.2f}")
    out=os.path.join(a.output_dir,"best_model_final"); tr.save_model(out); tok.save_pretrained(out); print(f"--- Best model saved to {out} ---"); print(f"ExCP compressed artifacts under: {excp_dir}")

if __name__=="__main__": main()


# 比较严格的按照了官方代码库实现了ExCP。但是运行得到的效果还是没有那么好。精度下降的比较多。猜测有可能是因为参数太少。
# 可以转移到更大的模型去尝试训练。