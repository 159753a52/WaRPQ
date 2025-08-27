# train_gpt2_wikitext2.py
import os, re, json, math, time, argparse, random
from typing import Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling,
                          TrainingArguments, Trainer, TrainerCallback, TrainerState, TrainerControl,
                          EarlyStoppingCallback)

def set_seed(s:int):
    os.environ["PYTHONHASHSEED"]=str(s); random.seed(s); np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s); torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def quantize_uniform_symmetric(y:torch.Tensor,n_bits:int,vmax:torch.Tensor):
    if n_bits<=1: return torch.zeros_like(y), torch.tensor(0.,device=y.device)
    qmax=2**(n_bits-1)-1; s=(vmax/qmax).clamp_min(1e-8); q=torch.clamp(torch.round(y/s),min=-qmax,max=qmax); return q*s, qmax

def quantize_pergroup_uniform(t:torch.Tensor,n_bits:int,group:int,dim:int=-1,clip:float=0.995):
    if n_bits<=0 or t.ndim==0 or t.numel()==0: return t.clone()
    dtype=t.dtype; dim=dim if dim>=0 else t.ndim+dim
    p=list(range(t.ndim)); p[dim],p[-1]=p[-1],p[dim]; x=t.permute(*p).contiguous(); sh=x.shape; L=sh[-1]; x=x.view(-1,L)
    pad=(group-(L%group))%group
    if pad>0: x=torch.cat([x, torch.zeros(x.shape[0],pad,device=x.device,dtype=x.dtype)],1)
    G=x.shape[1]//group; x=x.view(-1,G,group)
    vmax=torch.quantile(x.abs(),q=clip,dim=-1,keepdim=True).clamp_min(1e-8); qmax=2**(n_bits-1)-1
    q=torch.clamp(torch.round(x/(vmax/qmax)),min=-qmax,max=qmax); deq=q*(vmax/qmax)
    deq=deq.view(-1,G*group); deq=deq[:,:L] if pad>0 else deq; deq=deq.view(*sh[:-1],L)
    inv=list(range(t.ndim)); inv[dim],inv[-1]=inv[-1],inv[dim]; return deq.permute(*inv).contiguous().to(dtype=dtype)

def _H(p:float)->float: 
    return 0.0 if p<=0.0 or p>=1.0 else float(-p*math.log2(p)-(1-p)*math.log2(1-p))
def _sd(a:float,b:float,eps:float=1e-12)->float: return float(a/(b+eps))

class JsonLoggingCallback(TrainerCallback):
    def __init__(self,p): self.p=p; self.h=[]
    def on_train_begin(self,args,state,control,**k):
        os.makedirs(os.path.dirname(self.p),exist_ok=True); open(self.p,"w").write("[]")
    def on_log(self,args,state,control,logs=None,**k):
        if logs is None: return
        e={**logs,"step":state.global_step}; self.h.append(e); open(self.p,"w").write(json.dumps(self.h,indent=2))

def project_topk_time(x:torch.Tensor,p_keep:float,eps:float=1e-12):
    if p_keep>=1.0: return x, torch.ones_like(x,dtype=x.dtype)
    if p_keep<=0.0: return torch.zeros_like(x), torch.zeros_like(x,dtype=x.dtype)
    n=x.numel(); k=max(1,int(round(p_keep*n))); flat=x.abs().reshape(-1); thr=torch.kthvalue(flat,n-k+1).values
    m=(x.abs()>=thr).to(dtype=x.dtype); return x*m, m

class SubbandPredictor:
    def __init__(self,beta:float=0.9,mode:str="ar1"): self.b=beta; self.mode=mode; self.st={}
    def _g(self,k):
        if k not in self.st: self.st[k]={"mu":0.0,"s0":1e-6,"s1":0.0,"last":None}; return self.st[k]
    def update_and_predict(self,k,cur,is_i):
        s=self._g(k); mu,s0,s1,last=s["mu"],s["s0"],s["s1"],s["last"]; x=cur.mean()
        pred=x if last is None else (self.b*last+(1-self.b)*mu if self.mode=="ema" else (s1/max(s0,1e-6))*(last-mu)+mu)
        d=(x-mu); mu=0.99*mu+0.01*float(x.item()); s0=0.99*s0+0.01*float(d*d); 
        if last is not None: s1=0.99*s1+0.01*float(d*(last-mu))
        s["mu"],s["s0"],s["s1"],s["last"]=mu,s0,s1,float(x.item()); 
        return float(pred), float(1.0-_sd((x-pred)*(x-pred),max(x*x,1e-12)))

def make_dct_mat(N,dev,dtype):
    n=torch.arange(N,device=dev,dtype=dtype).reshape(1,-1); k=torch.arange(N,device=dev,dtype=dtype).reshape(-1,1)
    M=torch.cos(math.pi*(n+0.5)*k/N); M[0,:]=M[0,:]/math.sqrt(2.0); M=M*math.sqrt(2.0/N); return M

class TwoDTransforms:
    def __init__(self, kind:str, tile_h:int, tile_w:int, dwt_wave:str="haar", dwt_levels:int=1):
        self.kind=kind; self.tile_h=tile_h; self.tile_w=tile_w; self.wave=dwt_wave; self.J=dwt_levels
        self._dct={}; self._dwt={}
    def _get_dct(self,dev,dtype):
        k=(dev,dtype,self.tile_h,self.tile_w)
        if k not in self._dct: self._dct[k]=(make_dct_mat(self.tile_h,dev,dtype), make_dct_mat(self.tile_w,dev,dtype))
        return self._dct[k]
    def _get_dwt(self,dev):
        k=(dev,self.wave,self.J)
        if k not in self._dwt:
            from pytorch_wavelets import DWTForward, DWTInverse
            self._dwt[k]=(DWTForward(J=self.J,wave=self.wave,mode="symmetric").to(dev),
                          DWTInverse(wave=self.wave,mode="symmetric").to(dev))
        return self._dwt[k]
    def fwd(self,blk):
        if self.kind=="dct2":
            Mh,Mw=self._get_dct(blk.device,blk.dtype); y=torch.einsum("hi,biw->bhw",Mh,blk); y=torch.einsum("bhw,wj->bhj",y,Mw.t()); return y,None
        if self.kind=="dwt2":
            dwt,_=self._get_dwt(blk.device); yl,yh=dwt(blk.unsqueeze(1)); return (yl.squeeze(1), yh), None
        return blk,None
    def inv(self,cf):
        if self.kind=="dct2":
            Mh,Mw=self._get_dct(cf.device,cf.dtype); y=torch.einsum("hi,biw->bhw",Mh.t(),cf); y=torch.einsum("bhw,wj->bhj",y,Mw); return y
        if self.kind=="dwt2":
            _,idwt=self._get_dwt(cf[0].device); cA,yh=cf; return idwt((cA.unsqueeze(1),yh)).squeeze(1)
        return cf

def blockify_2d(x:torch.Tensor,h:int,w:int):
    s=x.shape
    if x.ndim<2: X=x.reshape(1,-1); H=1; W=X.shape[-1]; X=X.reshape(1,H,W)
    else: H=x.shape[-2]; W=x.shape[-1]; X=x.reshape(-1,H,W)
    ph=(h-(H%h))%h; pw=(w-(W%w))%w
    if ph>0 or pw>0: X=F.pad(X,(0,pw,0,ph),value=0.0)
    H2,W2=X.shape[-2],X.shape[-1]; bh,bw=H2//h,W2//w
    X=X.view(-1,bh,h,bw,w).permute(0,1,3,2,4).reshape(-1,h,w)
    return X, {"orig_shape":list(s),"H":H,"W":W,"bh":bh,"bw":bw}

def unblockify_2d(blk:torch.Tensor, meta:Dict[str,Any]):
    h,w=blk.shape[-2],blk.shape[-1]; bh,bw=meta["bh"],meta["bw"]
    X=blk.view(-1,bh,bw,h,w).permute(0,1,3,2,4).reshape(-1,bh*h,bw*w); H,W=meta["H"],meta["W"]; X=X[:,:H,:W]
    s=meta["orig_shape"]; 
    if len(s)>=2: lead=int(np.prod(s[:-2])) if len(s)>2 else 1; X=X.view(lead,H,W).reshape(s)
    else: X=X.reshape(s)
    return X

class WaRPQCallback(TrainerCallback):
    def __init__(self, method:str, avg_bitwidth:float, quant_freq:int, quant_start_step:int=200,
                 apply_to_model:bool=False, target_nz_ratio:float=0.10, enforce_target_nz:bool=True,
                 gop_size:int=1, i_frame_scale:float=1.0, p_frame_scale:float=0.8,
                 qkv_group_size:int=128, qkv_clip_pct:float=0.99, momentum_bitwidth:int=8,
                 momentum_clip_pct:float=0.999, m2w_gate:bool=True, ema_decay:float=0.99,
                 transform:str="dct2", tile_h:int=32, tile_w:int=32, dwt_levels:int=1, dwt_wave:str="haar",
                 save_dir:str=None, save_payload:bool=True, use_predictor:bool=True, predictor_beta:float=0.9,
                 pi_disable_threshold:float=0.1, mask_from:str="post",
                 rate_mode:str="nz", target_bpp:float=None, bpp_tol:float=1e-3,
                 dwt_alloc_mode:str="fixed", dwt_alloc:str="A:0.6,LH:0.15,HL:0.15,HH:0.1",
                 dwt_bits:str="A:5,LH:3,HL:3,HH:2"):
        self.method=method; self.avg_bitwidth=avg_bitwidth; self.quant_freq=quant_freq; self.quant_start_step=quant_start_step
        self.apply_to_model=apply_to_model; self.target_nz_ratio=float(max(0,min(1,target_nz_ratio))); self.enforce_target_nz=enforce_target_nz
        self.gop_size=gop_size; self.i_frame_scale=i_frame_scale; self.p_frame_scale=p_frame_scale
        self.qkv_group_size=qkv_group_size; self.qkv_clip_pct=qkv_clip_pct
        self.momentum_bitwidth=momentum_bitwidth; self.momentum_clip_pct=momentum_clip_pct; self.m2w_gate=m2w_gate
        self.ema_decay=ema_decay
        self.transforms=TwoDTransforms(kind=transform, tile_h=tile_h, tile_w=tile_w, dwt_wave=dwt_wave, dwt_levels=dwt_levels)
        self.save_dir=save_dir; self.save_payload=save_payload
        self.use_predictor=use_predictor; self.predictor=SubbandPredictor(beta=predictor_beta,mode="ar1") if use_predictor else None
        self.pi_disable_threshold=pi_disable_threshold; self.mask_from=mask_from
        self.rate_mode=rate_mode; self.target_bpp=target_bpp; self.bpp_tol=bpp_tol
        self.dwt_alloc_mode=dwt_alloc_mode; self.dwt_alloc=dwt_alloc; self.dwt_bits=dwt_bits
        self.baseline_state_dict=None; self._optimizer_ref=None; self._last_gop_start=0; self._ef_buf={}; self.nz_eps=1e-12

    def _ok(self,n,p):
        if not p.requires_grad or self.method=="none": return False
        l=n.lower()
        if l.endswith(".bias") or ".bias" in l: return False
        if "layernorm" in l or re.search(r"\bnorm\b",l) or "ln_" in l: return False
        if "embed" in l or "embedding" in l or "wte" in l or "wpe" in l: return False
        if p.ndim<=1: return False
        return True

    def _frame_scale(self,step:int):
        if self.gop_size<=1: return 1.0, True
        period=self.gop_size*self.quant_freq
        if (step-self._last_gop_start)>=period or step==0: self._last_gop_start=step
        idx=(step-self._last_gop_start)//self.quant_freq
        return (self.i_frame_scale,True) if idx==0 else (self.p_frame_scale,False)

    def on_train_begin(self,args,state,control,model=None,**k):
        if self.method=="none": return
        self.baseline_state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        self._optimizer_ref=k.get("optimizer",None)
        if self.save_dir is None: self.save_dir=os.path.join(args.output_dir,"ckpts")
        os.makedirs(self.save_dir,exist_ok=True); open(os.path.join(self.save_dir,"meta.json"),"w").write(json.dumps({"method":self.method}))

    def on_optimizer_step(self,args,state,control,optimizer=None,**k):
        if optimizer is not None: self._optimizer_ref=optimizer

    def _find_group_for_param(self,opt,param):
        if opt is None: return None
        for g in opt.param_groups:
            for p in g.get("params",[]):
                if p is param: return g
        return None

    def _get_lr(self,opt,param):
        g=self._find_group_for_param(opt,param); return None if g is None else g.get("lr",None)

    def _is_qkv(self,name):
        l=name.lower()
        return ("c_attn.weight" in name) or ("attention.query_key_value.weight" in l) or ("q_proj" in l) or ("k_proj" in l) or ("v_proj" in l)

    def _parse_kv(self,s,cast=float):
        r={}
        for t in s.split(","):
            if not t: continue
            k,v=t.split(":"); r[k.strip()]=cast(v.strip())
        return r

    def _solve_p_for_bpp(self,b_eff):
        if self.rate_mode!="bpp" or self.target_bpp is None: return self.target_nz_ratio
        lo,hi=1e-6,0.5
        for _ in range(16):
            mid=(lo+hi)/2; bpp=mid*b_eff+_H(mid)
            if bpp>self.target_bpp: hi=mid
            else: lo=mid
            if abs(bpp-self.target_bpp)<=self.bpp_tol: break
        return max(1e-6,min(hi,self.target_nz_ratio if self.rate_mode=="nz" else hi)) if False else (lo+hi)/2

    def _alloc_for_dwt(self, bands):
        if self.dwt_alloc_mode=="energy":
            e=[float((b.float()*b.float()).mean().item())+1e-12 for _,b in bands]; s=sum(e); w=[v/s for v in e]
            return w
        mp=self._parse_kv(self.dwt_alloc,float)
        J=len([1 for k,_ in bands if k.startswith("L") and "_LH" in k])//1 if len(bands)>1 else 1
        aA=mp.get("A",0.6); aLH=mp.get("LH",0.15); aHL=mp.get("HL",0.15); aHH=mp.get("HH",0.10)
        per={"A":aA,"LH":aLH/J,"HL":aHL/J,"HH":aHH/J}
        w=[per["A"] if k=="A" else per["LH"] if k.endswith("_LH") else per["HL"] if k.endswith("_HL") else per["HH"] for k,_ in bands]
        s=sum(w); return [v/s for v in w]

    def _bits_for_dwt(self, bands, b_base:int):
        mp=self._parse_kv(self.dwt_bits,int); a=mp.get("A",max(5,b_base)); dl=max(2,b_base-1)
        m={"LH":mp.get("LH",dl),"HL":mp.get("HL",dl),"HH":mp.get("HH",dl)}
        return [a if k=="A" else m["LH"] if k.endswith("_LH") else m["HL"] if k.endswith("_HL") else m["HH"] for k,_ in bands]

    def _compress_step(self, model:nn.Module, optimizer, state:TrainerState, tag:str):
        dev=next(model.parameters()).device; sc,is_I=self._frame_scale(state.global_step)
        ab=max(1.5,min(6.0,self.avg_bitwidth*sc)); t0=time.perf_counter()
        tot,nnz,bsum,bcnt,qkvL,qkve,msews,varws=0,0,0.0,0,0,0,0.0,0.0; sb_stat={}
        def add_stat(lbl,orig,rec):
            n=orig.numel()
            if n==0: return
            d=(orig.float()-rec.float()); mse=float((d*d).sum().item()); var=float(orig.float().var(unbiased=False).item())*n
            a=sb_stat.get(lbl,[0.0,0.0,0]); a[0]+=mse; a[1]+=max(var,1e-12); a[2]+=n; sb_stat[lbl]=a

        for name,param in model.named_parameters():
            if name not in self.baseline_state_dict or not self._ok(name,param): continue
            base=self.baseline_state_dict[name].to(device=dev,dtype=param.dtype); w=param.data
            st=optimizer.state.get(param,{}) if optimizer is not None else {}
            m=st.get("exp_avg",None); v=st.get("exp_avg_sq",None); lr=self._get_lr(optimizer,param)
            pred=torch.zeros_like(w) if (m is None or v is None or lr is None) else -lr*(m.to(device=w.device,dtype=w.dtype)/(torch.sqrt(v.to(device=w.device,dtype=w.dtype)+1e-12)))
            E=(w-base)-pred; ef=self._ef_buf.get(name,None); E_tilde=E+(torch.zeros_like(E) if ef is None else ef); n=E.numel()

            if self._is_qkv(name):
                qkvL+=1; qkve+=n; bq=int(round(ab)); p_keep=self._solve_p_for_bpp(bq) if self.rate_mode=="bpp" else self.target_nz_ratio
                rec_E=quantize_pergroup_uniform(E_tilde,n_bits=bq,group=self.qkv_group_size,dim=-1,clip=self.qkv_clip_pct)
                rec_E,mask=project_topk_time(rec_E,p_keep) if self.enforce_target_nz else (rec_E,(rec_E.abs()>self.nz_eps).to(dtype=E.dtype))
            else:
                blk,meta=blockify_2d(E_tilde,self.transforms.tile_h,self.transforms.tile_w)
                if self.transforms.kind=="dct2":
                    Mh,Mw=self.transforms._get_dct(blk.device,blk.dtype); Y=torch.einsum("hi,biw->bhw",Mh,blk); Y=torch.einsum("bhw,wj->bhj",Y,Mw.t())
                    b=int(round(ab)); p_keep=self._solve_p_for_bpp(b) if self.rate_mode=="bpp" else self.target_nz_ratio
                    vmax=torch.quantile(Y.abs().reshape(Y.shape[0],-1),q=0.995,dim=1,keepdim=True).clamp_min(1e-8).view(-1,1,1)
                    Yq,_=quantize_uniform_symmetric(Y,b,vmax); Yq,mask=project_topk_time(Yq,p_keep) if self.enforce_target_nz else (Yq,(Yq.abs()>self.nz_eps).to(dtype=Y.dtype))
                    rb=torch.einsum("hi,biw->bhw",Mh.t(),Yq); rb=torch.einsum("bhw,wj->bhj",rb,Mw)
                    rec_E=unblockify_2d(rb,meta).reshape_as(E_tilde)
                elif self.transforms.kind=="dwt2":
                    (cA,yh),_=self.transforms.fwd(blk); b_base=int(round(ab))
                    bands=[("A",cA)]
                    for i,d in enumerate(yh):
                        D=d.squeeze(1); bands+=[(f"L{i+1}_LH",D[:,0]),(f"L{i+1}_HL",D[:,1]),(f"L{i+1}_HH",D[:,2])]
                    w_alloc=self._alloc_for_dwt(bands); bits=self._bits_for_dwt(bands,b_base)
                    b_eff=sum([w_alloc[i]*bits[i] for i in range(len(bands))]); p_tot=self._solve_p_for_bpp(b_eff) if self.rate_mode=="bpp" else self.target_nz_ratio
                    K_tot=max(1,int(round(p_tot*E_tilde.numel())))
                    sizes=[b[1].numel() for b in bands]; S=sum(sizes); K=[max(0,min(sizes[i],int(round(w_alloc[i]*K_tot*S/max(1,S))))) for i in range(len(bands))]
                    rec_coeff=[]
                    for i,(lbl,B) in enumerate(bands):
                        vmax=torch.quantile(B.abs().reshape(B.shape[0],-1),q=0.995 if lbl=="A" else 0.990,dim=1,keepdim=True).clamp_min(1e-8)
                        vmax=vmax.view(B.shape[0],*([1]*(B.ndim-1))); dq,_=quantize_uniform_symmetric(B,bits[i],vmax)
                        if self.enforce_target_nz:
                            p_i=float(K[i])/max(1,sizes[i]); dq,mask_i=project_topk_time(dq,p_i)
                        else:
                            mask_i=(dq.abs()>self.nz_eps).to(dtype=dq.dtype)
                        add_stat(lbl,B,dq); rec_coeff.append((lbl,dq))
                    rcA=[x[1] for x in rec_coeff if x[0]=="A"][0]
                    ry=[]
                    for i,d in enumerate(yh):
                        D=torch.zeros_like(d.squeeze(1)); D[:,0]=[x[1] for x in rec_coeff if x[0]==f"L{i+1}_LH"][0]
                        D[:,1]=[x[1] for x in rec_coeff if x[0]==f"L{i+1}_HL"][0]; D[:,2]=[x[1] for x in rec_coeff if x[0]==f"L{i+1}_HH"][0]
                        ry.append(D.unsqueeze(1))
                    rb=self.transforms.inv((rcA,ry)); rec_E=unblockify_2d(rb,meta).reshape_as(E_tilde)
                    mask=(rec_E.abs()>self.nz_eps).to(dtype=E_tilde.dtype)
                else:
                    b=int(round(ab)); p_keep=self._solve_p_for_bpp(b) if self.rate_mode=="bpp" else self.target_nz_ratio
                    vmax=torch.quantile(blk.abs().reshape(blk.shape[0],-1),q=0.995,dim=1,keepdim=True).clamp_min(1e-8).view(-1,1,1)
                    rb,_=quantize_uniform_symmetric(blk,b,vmax); rb,maskb=project_topk_time(rb,p_keep) if self.enforce_target_nz else (rb,(rb.abs()>self.nz_eps).to(dtype=rb.dtype))
                    rec_E=unblockify_2d(rb,meta).reshape_as(E_tilde); mask=maskb

            if self.apply_to_model: param.data.copy_(base+pred+rec_E)
            self._ef_buf[name]=(E_tilde-rec_E).detach()

            if optimizer is not None and (param in optimizer.state):
                st=optimizer.state[param]
                if "exp_avg" in st and isinstance(st["exp_avg"],torch.Tensor):
                    d=(rec_E-E).abs().view(-1); thr=torch.quantile(d,0.90); ch=(d.view_as(rec_E)>=thr); st["exp_avg"][ch]=0
                    if "exp_avg_sq" in st and isinstance(st["exp_avg_sq"],torch.Tensor): st["exp_avg_sq"][ch]=0
                if self.momentum_bitwidth>0:
                    fm=mask
                    for k in ["exp_avg","exp_avg_sq"]:
                        if k in st and isinstance(st[k],torch.Tensor):
                            mom=st[k].to(device=E.device,dtype=E.dtype)*fm; g=1024; mf=mom.reshape(-1); pad=(g-(mf.numel()%g))%g
                            if pad>0: mf=F.pad(mf,(0,pad))
                            G=mf.view(-1,g); vmax=torch.quantile(G.abs(),q=self.momentum_clip_pct,dim=-1,keepdim=True).clamp_min(1e-8)
                            qmax=2**(self.momentum_bitwidth-1)-1; qm=torch.clamp(torch.round(G/(vmax/qmax)),min=-qmax,max=qmax); deq=qm*(vmax/qmax)
                            st[k].copy_(deq.view(-1)[:mom.numel()].reshape_as(mom))

            new=(base+pred+rec_E).detach(); ema=self.ema_decay
            if self.baseline_state_dict is not None:
                self.baseline_state_dict[name]=(self.baseline_state_dict[name].to(new.device,dtype=new.dtype)*ema+new*(1.0-ema)).detach().cpu()

            nz=int((mask>0).sum().item()); tot+=n; nnz+=nz; b_l=int(round(ab)); bsum+=b_l*nz; bcnt+=nz
            err=(E.float()-rec_E.float()); msqe=float(torch.mean(err*err).item()); varE=float(E.float().var(unbiased=False).item()); msews+=msqe*n; varws+=varE*n

        bavg=(bsum/bcnt) if bcnt>0 else float(self.avg_bitwidth); p=_sd(nnz,tot); bpp=p*bavg+_H(p); cr=32.0/max(bpp,1e-9)
        nmse=_sd(msews,varws); snr=10.0*math.log10(max(1.0/max(nmse,1e-12),1e-12)); t1=time.perf_counter()
        print(f"    - Frame: {'I' if is_I else 'P'} | p={p:.4f} | bpp={bpp:.3f} | CR≈{cr:.1f}x | SNR={snr:.2f}dB | QKV={qkvL}({qkve/1e6:.2f}M) | {t1-t0:.2f}s")
        sb_out={k:[_sd(v[0],v[1]), 10.0*math.log10(max(1.0/max(_sd(v[0],v[1]),1e-12),1e-12))] for k,v in sb_stat.items()}
        rec={"step":int(state.global_step),"bpp":float(bpp),"cr_weight":float(cr),"nmse":float(nmse),"snr_db":float(snr),
             "time_s":float(t1-t0),"p":float(p),"avg_bit":float(bavg),"transform":self.transforms.kind,
             "tile":[self.transforms.tile_h,self.transforms.tile_w],"qkv_fast_layers":int(qkvL),"subbands":sb_out}
        with open(os.path.join(self.save_dir,"compress_stats.jsonl"),"a") as f: f.write(json.dumps(rec)+"\n")

    def on_step_end(self,args,state,control,model=None,optimizer=None,**k):
        if self.baseline_state_dict is None or self.method=="none": return
        if state.global_step<self.quant_start_step: return
        if state.global_step>0 and state.global_step%self.quant_freq==0:
            self._compress_step(model, optimizer if optimizer is not None else self._optimizer_ref, state, f"{state.global_step:08d}")

@torch.no_grad()
def strided_perplexity(raw, tok, model, stride=512):
    dev=next(model.parameters()).device; text="\n\n".join(raw["text"]); enc=tok(text,return_tensors="pt"); ids=enc.input_ids.to(dev)
    ml=getattr(model.config,"n_positions", tok.model_max_length); ml=tok.model_max_length if (ml is None or ml>tok.model_max_length) else ml
    nll=0.0; nt=0; prev=0; L=ids.size(1)
    for b in range(0,L,stride):
        e=min(b+ml,L); trg=e-prev; inp=ids[:,b:e]; tgt=inp.clone(); tgt[:,:-trg]=-100
        out=model(inp,labels=tgt); nv=(tgt!=-100).sum().item(); bs=tgt.size(0); nvt=nv-bs; nll+=out.loss*nvt; nt+=nvt; prev=e
        if e==L: break
    a=nll/nt; ppl=torch.exp(a).item(); return a.item(), ppl

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
    p.add_argument("--transform",type=str,default="dct2",choices=["none","dct2","dwt2"])
    p.add_argument("--tile_h",type=int,default=32)
    p.add_argument("--tile_w",type=int,default=32)
    p.add_argument("--dwt_levels",type=int,default=1)
    p.add_argument("--dwt_wave",type=str,default="haar")
    p.add_argument("--save_payload",action="store_true",default=True)
    p.add_argument("--save_dir",type=str,default=None)
    p.add_argument("--use_predictor",action="store_true",default=False)
    p.add_argument("--predictor_beta",type=float,default=0.9)
    p.add_argument("--pi_disable_threshold",type=float,default=0.1)
    p.add_argument("--mask_from",type=str,default="post",choices=["pre","post"])
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

    # 在 ArgumentParser 中追加
    p.add_argument("--rate_mode",type=str,default="nz",choices=["nz","bpp"])
    p.add_argument("--target_bpp",type=float,default=None)
    p.add_argument("--bpp_tol",type=float,default=1e-3)
    p.add_argument("--dwt_alloc_mode",type=str,default="fixed",choices=["fixed","energy"])
    p.add_argument("--dwt_alloc",type=str,default="A:0.6,LH:0.15,HL:0.15,HH:0.1")
    p.add_argument("--dwt_bits",type=str,default="A:5,LH:3,HL:3,HH:2")

    a=p.parse_args()

    set_seed(a.seed)
    ds=load_dataset("wikitext","wikitext-2-raw-v1")
    tok=AutoTokenizer.from_pretrained(a.model_name,use_fast=True); 
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(a.model_name); model.config.pad_token_id=tok.pad_token_id

    def _tok(b): return tok(b["text"],return_special_tokens_mask=True)
    tokenized=ds.map(_tok,batched=True,remove_columns=["text"])
    bs=a.lm_block_size
    def group(ex):
        c={k:sum(ex[k],[]) for k in ex.keys()}; t=(len(c["input_ids"])//bs)*bs
        r={k:[v[i:i+bs] for i in range(0,t,bs)] for k,v in c.items()}; r["labels"]=r["input_ids"].copy(); return r
    lm=tokenized.map(group,batched=True)
    coll=DataCollatorForLanguageModeling(tok,mlm=False)

    ta=TrainingArguments(output_dir=a.output_dir,seed=a.seed,eval_strategy="steps",eval_steps=a.eval_freq,
                         save_strategy="steps",save_steps=a.save_freq,save_total_limit=5,load_best_model_at_end=True,
                         metric_for_best_model="eval_loss",greater_is_better=False,per_device_train_batch_size=a.per_device_train_batch_size,
                         per_device_eval_batch_size=a.per_device_eval_batch_size,gradient_accumulation_steps=a.grad_accum,
                         learning_rate=a.lr,num_train_epochs=a.epochs,max_steps=a.max_steps,warmup_ratio=a.warmup_ratio,
                         weight_decay=a.weight_decay,lr_scheduler_type=a.lr_scheduler_type,fp16=torch.cuda.is_available(),
                         dataloader_num_workers=2,logging_steps=50,logging_first_step=True,report_to="none",save_safetensors=True,group_by_length=True)

    # 在 WaRPQCallback(...) 实例化时把新参数也传入
    cb=WaRPQCallback(
        method=a.method,avg_bitwidth=a.avg_bitwidth,quant_freq=a.quant_freq,quant_start_step=a.quant_start_step,
        apply_to_model=a.apply_to_model,target_nz_ratio=a.target_nz_ratio,enforce_target_nz=a.enforce_target_nz,
        gop_size=a.gop_size,i_frame_scale=a.i_frame_scale,p_frame_scale=a.p_frame_scale,qkv_group_size=a.qkv_group_size,
        qkv_clip_pct=a.qkv_clip_pct,momentum_bitwidth=a.momentum_bitwidth,momentum_clip_pct=a.momentum_clip_pct,
        m2w_gate=a.m2w_gate,ema_decay=a.ema_decay,transform=a.transform,tile_h=a.tile_h,tile_w=a.tile_w,
        dwt_levels=a.dwt_levels,dwt_wave=a.dwt_wave,save_dir=a.save_dir,save_payload=a.save_payload,
        use_predictor=a.use_predictor,predictor_beta=a.predictor_beta,pi_disable_threshold=a.pi_disable_threshold,
        mask_from=a.mask_from,rate_mode=a.rate_mode,target_bpp=a.target_bpp,bpp_tol=a.bpp_tol,
        dwt_alloc_mode=a.dwt_alloc_mode,dwt_alloc=a.dwt_alloc,dwt_bits=a.dwt_bits
    )


    cbs=[JsonLoggingCallback(os.path.join(a.output_dir,"training_log.json")),cb]
    if (not a.disable_early_stop) and a.early_stop_patience>0:
        cbs.append(EarlyStoppingCallback(early_stopping_patience=a.early_stop_patience,early_stopping_threshold=0.0))

    tr=Trainer(model=model,args=ta,train_dataset=lm["train"],eval_dataset=lm["validation"],data_collator=coll,callbacks=cbs)
    print("--- Starting Training (GPT-2 on WikiText-2, method:",a.method,") ---"); tr.train()
    mv=tr.evaluate(); ev=mv["eval_loss"]; print(f"[Trainer.evaluate] valid: eval_loss={ev:.4f}, ppl={math.exp(ev):.2f}")
    model.eval(); nll,ppl=strided_perplexity(ds["validation"],tok,model,stride=a.strided_stride)
    print(f"[Strided] valid: avg_nll={nll:.4f}, ppl={ppl:.2f}")
    if "test" in ds:
        nllt,pplt=strided_perplexity(ds["test"],tok,model,stride=a.strided_stride); print(f"[Strided]  test: avg_nll={nllt:.4f}, ppl={pplt:.2f}")
    out=os.path.join(a.output_dir,"best_model_final"); tr.save_model(out); print(f"--- Best model saved to {out} ---")

if __name__=="__main__": main()
