import os,re,json,math,time,argparse,random
from typing import Dict,Any,Tuple,List
import numpy as np,torch,torch.nn as nn,torch.nn.functional as F
from datasets import load_dataset
from transformers import (AutoTokenizer,AutoModelForCausalLM,DataCollatorForLanguageModeling,TrainingArguments,Trainer,TrainerCallback,EarlyStoppingCallback)

def set_seed(s:int):
    os.environ["PYTHONHASHSEED"]=str(s);random.seed(s);np.random.seed(s);torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False

def quantize_uniform_symmetric(y:torch.Tensor,n_bits:int,vmax:torch.Tensor):
    if n_bits<=1: return torch.zeros_like(y),torch.tensor(0.,device=y.device)
    qmax=2**(n_bits-1)-1;s=(vmax/qmax).clamp_min(1e-8);q=torch.clamp(torch.round(y/s),min=-qmax,max=qmax);return q*s,qmax

def quantize_pergroup_uniform(t:torch.Tensor,n_bits:int,group:int,dim:int=-1,clip:float=0.995):
    if n_bits<=0 or t.ndim==0 or t.numel()==0: return t.clone()
    dtype=t.dtype;dim=dim if dim>=0 else t.ndim+dim
    p=list(range(t.ndim));p[dim],p[-1]=p[-1],p[dim]
    x=t.permute(*p).contiguous();sh=x.shape;L=sh[-1];x=x.view(-1,L)
    pad=(group-(L%group))%group
    if pad>0: x=torch.cat([x,torch.zeros(x.shape[0],pad,device=x.device,dtype=x.dtype)],1)
    G=x.shape[1]//group;x=x.view(-1,G,group)
    vmax=torch.quantile(x.abs(),q=clip,dim=-1,keepdim=True).clamp_min(1e-8);qmax=2**(n_bits-1)-1
    q=torch.clamp(torch.round(x/(vmax/qmax)),min=-qmax,max=qmax);deq=q*(vmax/qmax)
    deq=deq.view(-1,G*group);deq=deq[:,:L] if pad>0 else deq;deq=deq.view(*sh[:-1],L)
    inv=list(range(t.ndim));inv[dim],inv[-1]=inv[-1],inv[dim]
    return deq.permute(*inv).contiguous().to(dtype=dtype)

def _H(p:float)->float: return 0.0 if p<=0.0 or p>=1.0 else float(-p*math.log2(p)-(1-p)*math.log2(1-p))
def _sd(a: float, b: float, eps: float = 1e-12) -> float: return float(a / (b + eps))

class JsonLoggingCallback(TrainerCallback):
    def __init__(self,p): self.p=p; self.h=[]
    def on_train_begin(self,args,state,control,**k): os.makedirs(os.path.dirname(self.p),exist_ok=True); open(self.p,"w").write("[]")
    def on_log(self,args,state,control,logs=None,**k):
        if logs is None: return
        e={**logs,"step":state.global_step}; self.h.append(e); open(self.p,"w").write(json.dumps(self.h,indent=2))

def project_topk_time(x:torch.Tensor,p_keep:float,eps:float=1e-12):
    if p_keep>=1.0: return x,torch.ones_like(x,dtype=x.dtype)
    if p_keep<=0.0: return torch.zeros_like(x),torch.zeros_like(x,dtype=x.dtype)
    n=x.numel();k=max(1,int(round(p_keep*n)));flat=x.abs().reshape(-1);thr=torch.kthvalue(flat,n-k+1).values
    m=(x.abs()>=thr).to(dtype=x.dtype);return x*m,m

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

class Transforms:
    def __init__(self,kind:str,tile_h:int,tile_w:int,dwt_wave:str="haar",dwt_levels:int=1):
        self.kind=kind;self.tile_h=tile_h;self.tile_w=tile_w;self.wave=dwt_wave;self.J=dwt_levels

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

@torch.no_grad()
def lifting1d_inv_ls(cA:torch.Tensor,dets:List[torch.Tensor],ab:List[Tuple[torch.Tensor,torch.Tensor]]):
    y=cA
    for (d,(a,b)) in reversed(list(zip(dets,ab))):
        s=y;odd=s-b*d;even=d+a*odd;y=torch.stack([even,odd],dim=-1).reshape(*even.shape[:-1],even.shape[-1]*2)
    return y

class SubbandPredictor:
    def __init__(self,beta:float=0.9,mode:str="ar1"): self.b=beta;self.mode=mode;self.st={}
    def _g(self,k):
        if k not in self.st: self.st[k]={"mu":0.0,"s0":1e-6,"s1":0.0,"last":None};return self.st[k]
    def update_and_predict(self,k,cur,is_i):
        s=self._g(k);mu,s0,s1,last=s["mu"],s["s0"],s["s1"],s["last"];x=cur.mean()
        pred=x if last is None else (self.b*last+(1-self.b)*mu if self.mode=="ema" else (s1/max(s0,1e-6))*(last-mu)+mu)
        d=(x-mu);mu=0.99*mu+0.01*float(x.item());s0=0.99*s0+0.01*float(d*d)
        if last is not None: s1=0.99*s1+0.01*float(d*(last-mu))
        s["mu"],s["s0"],s["s1"],s["last"]=mu,s0,s1,float(x.item());return float(pred),float(1.0-_sd((x-pred)*(x-pred),max(x*x,1e-12)))

class WaRPQCallback(TrainerCallback):
    def __init__(self,method:str,avg_bitwidth:float,quant_freq:int,quant_start_step:int=200,apply_to_model:bool=False,target_nz_ratio:float=0.10,enforce_target_nz:bool=True,gop_size:int=1,i_frame_scale:float=1.0,p_frame_scale:float=0.8,qkv_group_size:int=128,qkv_clip_pct:float=0.99,momentum_bitwidth:int=8,momentum_clip_pct:float=0.999,m2w_gate:bool=True,ema_decay:float=0.99,transform:str="dct2",tile_h:int=32,tile_w:int=32,dwt_levels:int=1,dwt_wave:str="haar",save_dir:str=None,save_payload:bool=True,use_predictor:bool=True,predictor_beta:float=0.9,pi_disable_threshold:float=0.1,mask_from:str="post",rate_mode:str="nz",target_bpp:float=None,bpp_tol:float=1e-3,dwt_alloc_mode:str="fixed",dwt_alloc:str="A:0.6,LH:0.15,HL:0.15,HH:0.1",dwt_bits:str="A:5,LH:3,HL:3,HH:2",qkv_use_dwt:bool=False):
        self.method=method;self.avg_bitwidth=avg_bitwidth;self.quant_freq=quant_freq;self.quant_start_step=quant_start_step
        self.apply_to_model=apply_to_model;self.target_nz_ratio=float(max(0,min(1,target_nz_ratio)));self.enforce_target_nz=enforce_target_nz
        self.gop_size=gop_size;self.i_frame_scale=i_frame_scale;self.p_frame_scale=p_frame_scale
        self.qkv_group_size=qkv_group_size;self.qkv_clip_pct=qkv_clip_pct
        self.momentum_bitwidth=momentum_bitwidth;self.momentum_clip_pct=momentum_clip_pct;self.m2w_gate=m2w_gate
        self.ema_decay=ema_decay
        self.transforms=Transforms(kind=transform,tile_h=tile_h,tile_w=tile_w,dwt_wave=dwt_wave,dwt_levels=dwt_levels)
        self.save_dir=save_dir;self.save_payload=save_payload
        self.use_predictor=use_predictor;self.predictor=SubbandPredictor(beta=predictor_beta,mode="ar1") if use_predictor else None
        self.pi_disable_threshold=pi_disable_threshold;self.mask_from=mask_from
        self.rate_mode=rate_mode;self.target_bpp=target_bpp;self.bpp_tol=bpp_tol
        self.dwt_alloc_mode=dwt_alloc_mode;self.dwt_alloc=dwt_alloc;self.dwt_bits=dwt_bits
        self.qkv_use_dwt=qkv_use_dwt
        self.baseline_state_dict=None;self._optimizer_ref=None;self._last_gop_start=0;self._ef_buf={};self.nz_eps=1e-12
        self._pa_cache={};self._pa_topk_frac=0.10

    def _ok(self,n,p):
        if not p.requires_grad or self.method=="none": return False
        l=n.lower()
        if l.endswith(".bias") or ".bias" in l: return False
        if "layernorm" in l or re.search(r"\bnorm\b",l) or "ln_" in l: return False
        if "embed" in l or "embedding" in l or "wte" in l or "wpe" in l: return False
        if p.ndim<=1: return False
        return True

    def _frame_scale(self,step:int):
        if self.gop_size<=1: return 1.0,True
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
        if self.transforms.kind in ("pa_adwt","g2lwt"):
            with torch.no_grad():
                for name,p in model.named_parameters():
                    if self._ok(name,p):
                        try: self._prepare_pa_for_param(name,p.data)
                        except: pass

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
            mid=(lo+hi)/2;bpp=mid*b_eff+_H(mid)
            if bpp>self.target_bpp: hi=mid
            else: lo=mid
            if abs(bpp-self.target_bpp)<=self.bpp_tol: break
        return (lo+hi)/2

    def _pca_seriation(self,W_2d:torch.Tensor,dim:int):
        Wcpu=W_2d.detach().float().cpu()
        if dim==0: U,S,V=torch.pca_lowrank(Wcpu,q=1,center=True); vals=U[:,0]
        else: U,S,V=torch.pca_lowrank(Wcpu.t(),q=1,center=True); vals=U[:,0]
        perm=torch.argsort(vals,dim=0); inv=torch.empty_like(perm); inv[perm]=torch.arange(perm.numel(),dtype=perm.dtype)
        return perm.to(W_2d.device),inv.to(W_2d.device)

    def _haar_comp_score(self,X_2d:torch.Tensor,axis:int,L:int,J:int,perm:torch.Tensor=None,frac:float=0.10):
        if L<=0: L=64
        Xp=X_2d.index_select(axis,perm) if perm is not None else X_2d
        B,meta=blockify_1d(Xp,L=L,dim=axis,J=J);cA,d=haar1d_fwd_vec(B,J)
        C=[cA]+d;C=torch.cat([t.reshape(t.shape[0],-1).abs_()**2 for t in C],1)
        k=max(1,int(math.ceil(frac*C.shape[1])));topk,_=torch.topk(C,k,dim=1,largest=True,sorted=False)
        return (topk.sum(1)/(C.sum(1)+1e-12)).mean().item()

    def _prepare_pa_for_param(self,name:str,W:torch.Tensor):
        if W.ndim<2: return
        M,N=W.shape[-2],W.shape[-1];W2d=W.detach().reshape(M,N)
        pr,ir=self._pca_seriation(W2d,0);pc,ic=self._pca_seriation(W2d,1)
        s_out=self._haar_comp_score(W2d,-2,self.transforms.tile_h,self.transforms.J,perm=pr,frac=self._pa_topk_frac)
        s_in =self._haar_comp_score(W2d,-1,self.transforms.tile_w,self.transforms.J,perm=pc,frac=self._pa_topk_frac)
        if s_in>=s_out: entry={"axis":"in","perm":pc,"inv":ic,"dim":-1,"L":self.transforms.tile_w,"scores":(s_in,s_out)}
        else: entry={"axis":"out","perm":pr,"inv":ir,"dim":-2,"L":self.transforms.tile_h,"scores":(s_in,s_out)}
        self._pa_cache[name]=entry;return entry

    def _pick_L_eff(self,T:int,L:int,J:int):
        m=1<<J
        if T<m: return m
        Le=min(L,(T//m)*m);return max(m,Le)

    def _alloc_for_haar1d(self,J:int):
        mp=self._parse_kv(self.dwt_alloc,float);aA=mp.get("A",0.6);rest=max(1e-9,1.0-aA);return [aA]+[rest/J]*J

    def _alloc_energy(self,bands:List[Tuple[str,torch.Tensor]])->List[float]:
        e=[float((t.float().pow(2)).mean().item()) for _,t in bands];w=np.array(e,dtype=np.float64);w=np.maximum(w,1e-12);w=w/w.sum();return w.tolist()

    def _bits_for_haar1d(self,b_base:int,J:int):
        mp=self._parse_kv(self.dwt_bits,int);a=mp.get("A",max(5,b_base));d=mp.get("D",max(2,b_base-1));return [a]+[d]*J

    def _trust_blend(self,E:torch.Tensor,rec_E:torch.Tensor,name:str,thresh:float):
        diff=(rec_E-E).float();den=(E.float().pow(2).mean()+1e-12);rel=(diff.pow(2).mean()/den).item()
        if not math.isfinite(rel) or rel<=thresh: return rec_E
        try: cos=F.cosine_similarity(E.view(1,-1).float(),rec_E.view(1,-1).float(),dim=1).item()
        except: cos=1.0
        cos=max(0.0,float(cos));alpha=min(1.0,max(0.15,0.5*cos/math.sqrt(rel/max(thresh,1e-12))))
        return alpha*rec_E+(1.0-alpha)*E

    def _compress_step(self,model:nn.Module,optimizer,state,tag:str):
        dev=next(model.parameters()).device;sc,is_I=self._frame_scale(state.global_step);ab=max(1.5,min(6.0,self.avg_bitwidth*sc));t0=time.perf_counter()
        if self.save_payload and self.baseline_state_dict is not None:
            os.makedirs(self.save_dir,exist_ok=True)
            torch.save(self.baseline_state_dict,os.path.join(self.save_dir,f"baseline_step{int(state.global_step):08d}_pre.pt"))
        tot_eff=nnz_eff=0;bsum_eff=0.0;bcnt_eff=0;qkvL=qkve=0;msews=varws=0.0;sb_stat={}
        def add_stat(lbl,orig,rec):
            n=orig.numel()
            if n==0: return
            d=(orig.float()-rec.float());mse=float((d*d).sum().item());var=float(orig.float().var(unbiased=False).item())*n
            a=sb_stat.get(lbl,[0.0,0.0,0]);a[0]+=mse;a[1]+=max(var,1e-12);a[2]+=n;sb_stat[lbl]=a
        rec_totals={};mom_dump={}
        for name,param in model.named_parameters():
            if name not in self.baseline_state_dict or not self._ok(name,param): continue
            base=self.baseline_state_dict[name].to(device=dev,dtype=param.dtype);w=param.data
            st=optimizer.state.get(param,{}) if optimizer is not None else {}
            m=st.get("exp_avg",None);v=st.get("exp_avg_sq",None);lr=self._get_lr(optimizer,param)
            pred=torch.zeros_like(w) if (m is None or v is None or lr is None) else (-lr*(m.to(device=w.device,dtype=w.dtype)/(torch.sqrt(v.to(device=w.device,dtype=w.dtype)+1e-12))))
            E=(w-base)-pred;ef=self._ef_buf.get(name,None);E_tilde=E+(torch.zeros_like(E) if ef is None else ef);n_elem=E.numel();mask_param=torch.ones_like(E_tilde,dtype=E_tilde.dtype)
            rec_E=None
            if self._is_qkv(name) and self.qkv_use_dwt and self.transforms.kind in ("g2lwt","haar1d_in","haar1d_out"):
                qkvL+=1;qkve+=n_elem;b_base=int(round(ab));bits=self._bits_for_haar1d(b_base,self.transforms.J);bits[0]=max(bits[0],6)
                M,N=E_tilde.shape[-2],E_tilde.shape[-1];n_heads=getattr(getattr(model,"config",None),"num_attention_heads",None)
                if n_heads and (N%(3*n_heads)==0):
                    hd=N//(3*n_heads);X=E_tilde.reshape(M,3,n_heads,hd);axis_dim=-1;L=self._pick_L_eff(hd,max(64,min(256,hd)),self.transforms.J)
                    B1,meta1=blockify_1d(X,L=L,dim=axis_dim,J=self.transforms.J);cS,d_list,ab_params=lifting1d_fwd_ls(B1,self.transforms.J)
                    alloc=self._alloc_for_haar1d(self.transforms.J)
                    if self.dwt_alloc_mode=="energy": alloc=self._alloc_energy([("S",cS)]+[(f"D{i+1}",d) for i,d in enumerate(d_list)])
                    rec_list=[]
                    for (lbl,x),bi in zip([("S",cS)]+[(f"D{i+1}",d) for i,d in enumerate(d_list)],bits):
                        q=0.999 if lbl=="S" else 0.99;vmax=torch.quantile(x.abs(),q=q,dim=1,keepdim=True).clamp_min(1e-8);qx,_=quantize_uniform_symmetric(x,bi,vmax);rec_list.append(qx)
                    rS,rDs=rec_list[0],rec_list[1:];RB=lifting1d_inv_ls(rS,rDs,ab_params);rb=RB[...,:B1.shape[-1]]
                    rec_perm=unblockify_1d(rb,meta1).reshape_as(X);rec_perm=rec_perm.reshape(M,N)
                    p_keep=self._solve_p_for_bpp(bits[0]) if self.rate_mode=="bpp" else self.target_nz_ratio
                    rec_E,mask_param=project_topk_time(rec_perm,p_keep);rec_E=self._trust_blend(E,rec_E,name,max(0.2,self.pi_disable_threshold))
                    nz=int((mask_param!=0).sum().item());tot_eff+=n_elem;nnz_eff+=nz;bsum_eff+=int(round(ab))*nz;bcnt_eff+=nz
                else:
                    bq=int(round(ab));p_keep=self._solve_p_for_bpp(bq) if self.rate_mode=="bpp" else self.target_nz_ratio
                    rec_E=quantize_pergroup_uniform(E_tilde,n_bits=bq,group=self.qkv_group_size,dim=-1,clip=self.qkv_clip_pct)
                    if self.enforce_target_nz:
                        if self.mask_from=="pre": _,mask_param=project_topk_time(E_tilde,p_keep);rec_E=rec_E*mask_param
                        else: rec_E,mask_param=project_topk_time(rec_E,p_keep)
                    rec_E=self._trust_blend(E,rec_E,name,self.pi_disable_threshold)
                    nz=int((mask_param!=0).sum().item());tot_eff+=n_elem;nnz_eff+=nz;bsum_eff+=bq*nz;bcnt_eff+=nz
            else:
                kind=self.transforms.kind
                if kind=="g2lwt":
                    entry=self._pa_cache.get(name,None) or self._prepare_pa_for_param(name,base);axis_dim=entry["dim"];perm=entry["perm"].to(E_tilde.device);inv=entry["inv"].to(E_tilde.device)
                    Xp=E_tilde.index_select(axis_dim,perm);T=Xp.shape[axis_dim];L=self._pick_L_eff(T,entry["L"],self.transforms.J)
                    B1,meta1=blockify_1d(Xp,L=L,dim=axis_dim,J=self.transforms.J);cS,d_list,ab_params=lifting1d_fwd_ls(B1,self.transforms.J)
                    b_base=int(round(ab));bits=self._bits_for_haar1d(b_base,self.transforms.J);bits[0]=max(bits[0],6)
                    alloc=self._alloc_for_haar1d(self.transforms.J)
                    if self.dwt_alloc_mode=="energy": alloc=self._alloc_energy([("S",cS)]+[(f"D{i+1}",d) for i,d in enumerate(d_list)])
                    if self.mask_from=="coeff" and self.enforce_target_nz:
                        rec_list=[];bands=[("S",cS)]+[(f"D{i+1}",d) for i,d in enumerate(d_list)];total_coeff=sum(x.numel() for _,x in bands);alpha_keep=1.5 if re.search(r"(c_proj|out_proj|fc2)\.weight",name) else 1.0
                        for (lbl,x),bi,ai in zip(bands,bits,alloc):
                            q=0.999 if lbl=="S" else 0.99;v=torch.quantile(x.abs(),q=q,dim=1,keepdim=True).clamp_min(1e-8);qx,_=quantize_uniform_symmetric(x,bi,v)
                            k_goal=int(round(alpha_keep*self.target_nz_ratio*ai*total_coeff));flat=qx.abs().reshape(-1);k_goal=min(max(1,k_goal),flat.numel())
                            if k_goal<flat.numel(): thr=torch.kthvalue(flat,flat.numel()-k_goal+1).values;qx=qx*(qx.abs()>=thr)
                            nnz_band=int((qx!=0).sum().item());bsum_eff+=bi*nnz_band;bcnt_eff+=nnz_band;nnz_eff+=nnz_band;tot_eff+=qx.numel();rec_list.append(qx);add_stat(f"G2_{lbl}",x,qx)
                        rS,rDs=rec_list[0],rec_list[1:];RB=lifting1d_inv_ls(rS,rDs,ab_params);rb=RB[...,:B1.shape[-1]]
                        rec_perm=unblockify_1d(rb,meta1).reshape_as(Xp);rec_E=rec_perm.index_select(axis_dim,inv);mask_param=(rec_E.abs()>self.nz_eps).to(dtype=E.dtype)
                        rec_E=self._trust_blend(E,rec_E,name,max(0.2,self.pi_disable_threshold))
                    else:
                        vS=torch.quantile(cS.abs(),q=0.999,dim=1,keepdim=True).clamp_min(1e-8);rS,_=quantize_uniform_symmetric(cS,bits[0],vS);add_stat("G2_S",cS,rS)
                        rD=[]
                        for i,(d,bi) in enumerate(zip(d_list,bits[1:])):
                            vd=torch.quantile(d.abs(),q=0.99,dim=1,keepdim=True).clamp_min(1e-8);dq,_=quantize_uniform_symmetric(d,bi,vd);rD.append(dq);add_stat(f"G2_D{i+1}",d,dq)
                        RB=lifting1d_inv_ls(rS,rD,ab_params);rb=RB[...,:B1.shape[-1]]
                        rec_perm=unblockify_1d(rb,meta1).reshape_as(Xp);rec_E=rec_perm.index_select(axis_dim,inv)
                        if self.enforce_target_nz:
                            p_keep=min(1.0,(self.target_nz_ratio if self.rate_mode!="bpp" else self._solve_p_for_bpp(b_base)))
                            if self.mask_from=="pre": _,mask_param=project_topk_time(E_tilde,p_keep);rec_E=rec_E*mask_param
                            else: rec_E,mask_param=project_topk_time(rec_E,p_keep)
                        else: mask_param=torch.ones_like(rec_E,dtype=rec_E.dtype)
                        rec_E=self._trust_blend(E,rec_E,name,max(0.2,self.pi_disable_threshold))
                        nz=int((mask_param!=0).sum().item());tot_eff+=n_elem;nnz_eff+=nz;bsum_eff+=int(round(ab))*nz;bcnt_eff+=nz
                else:
                    axis_dim=-1 if kind=="haar1d_in" else -2 if kind=="haar1d_out" else -1
                    perm=inv=None;X_for=E_tilde
                    if kind=="pa_adwt":
                        entry=self._pa_cache.get(name,None) or self._prepare_pa_for_param(name,base);axis_dim=entry["dim"];perm=entry["perm"].to(E_tilde.device);inv=entry["inv"].to(E_tilde.device);X_for=E_tilde.index_select(axis_dim,perm)
                    L=self._pick_L_eff(X_for.shape[axis_dim],(self.transforms.tile_w if axis_dim==-1 else self.transforms.tile_h),self.transforms.J)
                    B1,meta1=blockify_1d(X_for,L=L,dim=axis_dim,J=self.transforms.J);b_base=int(round(ab));cA,dets=haar1d_fwd_vec(B1,self.transforms.J);bits=self._bits_for_haar1d(b_base,self.transforms.J)
                    if self.mask_from=="coeff" and self.enforce_target_nz:
                        alloc=self._alloc_for_haar1d(self.transforms.J);bands=[("A",cA)]+[(f"D{i+1}",d) for i,d in enumerate(dets)];total_coeff=sum(x.numel() for _,x in bands);rec_list=[]
                        for (lbl,x),bi,ai in zip(bands,bits,alloc):
                            q=0.995 if lbl=="A" else 0.990;v=torch.quantile(x.abs(),q=q,dim=1,keepdim=True).clamp_min(1e-8);qx,_=quantize_uniform_symmetric(x,bi,v)
                            k_goal=int(round(self.target_nz_ratio*ai*total_coeff));flat=qx.abs().reshape(-1);k_goal=min(max(1,k_goal),flat.numel())
                            if k_goal<flat.numel(): thr=torch.kthvalue(flat,flat.numel()-k_goal+1).values;qx=qx*(qx.abs()>=thr)
                            nnz_band=int((qx!=0).sum().item());bsum_eff+=bi*nnz_band;bcnt_eff+=nnz_band;nnz_eff+=nnz_band;tot_eff+=qx.numel();rec_list.append(qx);add_stat(f"{'PA_' if kind=='pa_adwt' else 'A1D_'}{lbl}",x,qx)
                        rcA,rDs=rec_list[0],rec_list[1:];RB=haar1d_inv_vec(rcA,rDs);rb=RB[...,:B1.shape[-1]];rec_perm=unblockify_1d(rb,meta1).reshape_as(X_for);rec_E=(rec_perm if perm is None else rec_perm.index_select(axis_dim,inv));mask_param=(rec_E.abs()>self.nz_eps).to(dtype=E.dtype)
                    else:
                        vA=torch.quantile(cA.abs(),q=0.995,dim=1,keepdim=True).clamp_min(1e-8);rcA,_=quantize_uniform_symmetric(cA,bits[0],vA);add_stat("A1D_A",cA,rcA)
                        rD=[]
                        for i,(d,bi) in enumerate(zip(dets,bits[1:])):
                            vm=torch.quantile(d.abs(),q=0.990,dim=1,keepdim=True).clamp_min(1e-8);dq,_=quantize_uniform_symmetric(d,bi,vm);rD.append(dq);add_stat(f"A1D_D{i+1}",d,dq)
                        RB=haar1d_inv_vec(rcA,rD);rb=RB[...,:B1.shape[-1]];rec_perm=unblockify_1d(rb,meta1).reshape_as(X_for);rec_E=(rec_perm if perm is None else rec_perm.index_select(axis_dim,inv))
                        if self.enforce_target_nz:
                            p_keep=self.target_nz_ratio if self.rate_mode!="bpp" else self._solve_p_for_bpp(b_base)
                            if self.mask_from=="pre": _,mask_param=project_topk_time(E_tilde,p_keep);rec_E=rec_E*mask_param
                            else: rec_E,mask_param=project_topk_time(rec_E,p_keep)
                        else: mask_param=torch.ones_like(rec_E,dtype=rec_E.dtype)
                        rec_E=self._trust_blend(E,rec_E,name,self.pi_disable_threshold)
                        nz=int((mask_param!=0).sum().item());tot_eff+=n_elem;nnz_eff+=nz;bsum_eff+=int(round(ab))*nz;bcnt_eff+=nz
            rec_total=pred+rec_E;rec_totals[name]=rec_total.detach().cpu()
            if self.apply_to_model: param.data.copy_(base+rec_total.to(device=base.device,dtype=base.dtype))
            self.baseline_state_dict[name]=(base+rec_total).detach().cpu()
            self._ef_buf[name]=(E_tilde-rec_E).detach()
            if optimizer is not None and (param in optimizer.state):
                st=optimizer.state[param]
                if "exp_avg" in st and isinstance(st["exp_avg"],torch.Tensor):
                    d=(rec_E-E).abs().view(-1);thr=torch.quantile(d,0.90);ch=(d.view_as(rec_E)>=thr);st["exp_avg"][ch]=0
                    if "exp_avg_sq" in st and isinstance(st["exp_avg_sq"],torch.Tensor): st["exp_avg_sq"][ch]=0
                if self.momentum_bitwidth>0:
                    fm=(rec_E.abs()>self.nz_eps).to(device=E.device,dtype=E.dtype)
                    for k in ["exp_avg","exp_avg_sq"]:
                        if k in st and isinstance(st[k],torch.Tensor):
                            mom=st[k].to(device=E.device,dtype=E.dtype)*fm;g=1024;mf=mom.reshape(-1);pad=(g-(mf.numel()%g))%g
                            if pad>0: mf=F.pad(mf,(0,pad))
                            G=mf.view(-1,g);vmax=torch.quantile(G.abs(),q=self.momentum_clip_pct,dim=-1,keepdim=True).clamp_min(1e-8);qmax=2**(self.momentum_bitwidth-1)-1
                            qm=torch.clamp(torch.round(G/(vmax/qmax)),min=-qmax,max=qmax);deq=qm*(vmax/qmax);st[k].copy_(deq.view(-1)[: mom.numel()].reshape_as(mom))
                if "exp_avg" in st: mom_dump[name]={"exp_avg":st["exp_avg"].detach().cpu()}
                if "exp_avg_sq" in st: mom_dump[name] = {**mom_dump.get(name,{}),"exp_avg_sq":st["exp_avg_sq"].detach().cpu()}
            err=(E.float()-rec_E.float());msqe=float(torch.mean(err*err).item());varE=float(E.float().var(unbiased=False).item());msews+=msqe*n_elem;varws+=varE*n_elem
        bavg=(bsum_eff/bcnt_eff) if bcnt_eff>0 else float(self.avg_bitwidth);p=_sd(nnz_eff,tot_eff);bpp=p*bavg+_H(p);cr=32.0/max(bpp,1e-9);nmse=_sd(msews,varws);snr=10.0*math.log10(max(1.0/max(nmse,1e-12),1e-12));t1=time.perf_counter()
        print(f"    - Frame: {'I' if is_I else 'P'} | p={p:.4f} | bavg={bavg:.2f} | bpp={bpp:.3f} | CR≈{cr:.1f}x | SNR={snr:.2f}dB | QKV={qkvL}({qkve/1e6:.2f}M) | {t1-t0:.2f}s")
        sb_out={k:[_sd(v[0],v[1]),10.0*math.log10(max(1.0/max(_sd(v[0],v[1]),1e-12),1e-12))] for k,v in sb_stat.items()}
        rec={"step":int(state.global_step),"p_eff":float(p),"bavg_eff":float(bavg),"bpp":float(bpp),"cr_weight":float(cr),"nmse":float(nmse),"snr_db":float(snr),"time_s":float(t1-t0),"transform":self.transforms.kind,"tile":[self.transforms.tile_h,self.transforms.tile_w],"qkv_fast_layers":int(qkvL),"subbands":sb_out}
        with open(os.path.join(self.save_dir,"compress_stats.jsonl"),"a") as f: f.write(json.dumps(rec)+"\n")
        if self.save_payload and self.baseline_state_dict is not None:
            torch.save(self.baseline_state_dict,os.path.join(self.save_dir,f"baseline_step{int(state.global_step):08d}_post.pt"))
            torch.save({"step":int(state.global_step),"delta":rec_totals,"moments":mom_dump},os.path.join(self.save_dir,f"rcp_step{int(state.global_step):08d}.pt"))

    def on_step_end(self,args,state,control,model=None,optimizer=None,**k):
        if self.baseline_state_dict is None or self.method=="none": return
        if state.global_step<self.quant_start_step: return
        if state.global_step>0 and state.global_step%self.quant_freq==0:
            self._compress_step(model,optimizer if optimizer is not None else self._optimizer_ref,state,f"{state.global_step:08d}")

class _SimpleState:
    def __init__(self,step:int): self.global_step=step

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
    tokenized=raw_split.map(_tok,batched=True,remove_columns=["text"])
    bs=block_size
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
    p.add_argument("--dwt_bits",type=str,default="A:5,LH:3,HL:3,HH:2")
    p.add_argument("--qkv_use_dwt",action="store_true",default=False)
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
    p.add_argument("--transform",type=str,default="g2lwt",choices=["none","dct2","dwt2","haar1d_in","haar1d_out","pa_adwt","g2lwt"])
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

    cb=WaRPQCallback(method=a.method,avg_bitwidth=a.avg_bitwidth,quant_freq=a.quant_freq,quant_start_step=a.quant_start_step,apply_to_model=a.apply_to_model,target_nz_ratio=a.target_nz_ratio,enforce_target_nz=a.enforce_target_nz,gop_size=a.gop_size,i_frame_scale=a.i_frame_scale,p_frame_scale=a.p_frame_scale,qkv_group_size=a.qkv_group_size,qkv_clip_pct=a.qkv_clip_pct,momentum_bitwidth=a.momentum_bitwidth,momentum_clip_pct=a.momentum_clip_pct,m2w_gate=a.m2w_gate,ema_decay=a.ema_decay,transform=a.transform,tile_h=a.tile_h,tile_w=a.tile_w,dwt_levels=a.dwt_levels,dwt_wave=a.dwt_wave,save_dir=a.save_dir,save_payload=a.save_payload,use_predictor=a.use_predictor,predictor_beta=a.predictor_beta,pi_disable_threshold=a.pi_disable_threshold,mask_from=a.mask_from,rate_mode=a.rate_mode,target_bpp=a.target_bpp,bpp_tol=a.bpp_tol,dwt_alloc_mode=a.dwt_alloc_mode,dwt_alloc=a.dwt_alloc,dwt_bits=a.dwt_bits,qkv_use_dwt=a.qkv_use_dwt)

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
        cb=WaRPQCallback(method="warp_q",avg_bitwidth=a.avg_bitwidth,quant_freq=1,quant_start_step=0,apply_to_model=a.single_shot_apply,target_nz_ratio=a.target_nz_ratio,enforce_target_nz=a.enforce_target_nz,gop_size=1,i_frame_scale=1.0,p_frame_scale=1.0,qkv_group_size=a.qkv_group_size,qkv_clip_pct=a.qkv_clip_pct,momentum_bitwidth=0,momentum_clip_pct=a.momentum_clip_pct,m2w_gate=a.m2w_gate,ema_decay=a.ema_decay,transform=a.transform,tile_h=a.tile_h,tile_w=a.tile_w,dwt_levels=a.dwt_levels,dwt_wave=a.dwt_wave,save_dir=a.output_dir,save_payload=True,use_predictor=False,predictor_beta=a.predictor_beta,pi_disable_threshold=a.pi_disable_threshold,mask_from=a.mask_from,rate_mode=a.rate_mode,target_bpp=a.target_bpp,bpp_tol=a.bpp_tol,dwt_alloc_mode=a.dwt_alloc_mode,dwt_alloc=a.dwt_alloc,dwt_bits=a.dwt_bits,qkv_use_dwt=a.qkv_use_dwt)
        class _ArgsNS: 
            def __init__(self,out): self.output_dir=out
        cb.on_train_begin(_ArgsNS(a.output_dir),None,None,model=model)
        if a.single_shot_residual:
            if a.baseline_path and os.path.exists(a.baseline_path):
                bl=torch.load(a.baseline_path,map_location="cpu")
                cb.baseline_state_dict={k:bl.get(k,torch.zeros_like(v,device="cpu")).to(dtype=v.dtype).cpu() for k,v in model.state_dict().items()}
                print(f"[OneShot] Loaded baseline from: {a.baseline_path}")
            else:
                print("[OneShot] WARN: --baseline_path not found; use zero baseline.")
                cb.baseline_state_dict={k:torch.zeros_like(v).cpu() for k,v in model.state_dict().items()}
        else:
            cb.baseline_state_dict={k:torch.zeros_like(v).cpu() for k,v in model.state_dict().items()}
        if getattr(cb.transforms,"kind","") in ("pa_adwt","g2lwt"):
            with torch.no_grad():
                for name,p in model.named_parameters():
                    if cb._ok(name,p):
                        try: cb._prepare_pa_for_param(name,p.data)
                        except Exception as e: print(f"[OneShot] prepare cache skip {name}: {e}")
        fake_state=_SimpleState(a.single_shot_step);cb._compress_step(model,optimizer=None,state=fake_state,tag="oneshot")
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

#稳定可对比版本
# python train_gpt2_wikitext2_2.py   --method warp_q   --transform g2lwt --dwt_levels 2 --tile_w 128 --tile_h 128   --qkv_use_dwt   --dwt_alloc_mode energy --dwt_bits "A:6,D:2"   --avg_bitwidth 4.0 --target_nz_ratio 0.07   --pi_disable_threshold 0.20   --quant_start_step 400 --quant_freq 400   --apply_to_model   --output_dir runs/warpq_g2lwt_train