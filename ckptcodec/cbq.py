# ckptcodec/cbq.py
import math, numpy as np, torch, torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from .io import pack_mask, pack_uintegers, to_bytes_fp16

def _pad_pow2(x,J):
    m=1<<J;T=x.shape[-1];pad=(-T)%m
    return (F.pad(x,(0,pad)) if pad>0 else x),pad

def haar1d_fwd_vec(x,J):
    y,_=_pad_pow2(x,J);dets=[]
    for _ in range(J):
        even=y[...,::2];odd=y[...,1::2]
        cA=(even+odd)/math.sqrt(2.0);cD=(even-odd)/math.sqrt(2.0)
        dets.append(cD);y=cA
    return y,dets

def haar1d_inv_vec(cA,dets):
    y=cA
    for cD in reversed(dets):
        even=(y+cD)/math.sqrt(2.0);odd=(y-cD)/math.sqrt(2.0)
        y=torch.stack([even,odd],dim=-1).reshape(*even.shape[:-1],even.shape[-1]*2)
    return y

def blockify_1d(x,L,dim,J):
    d=dim if dim>=0 else x.ndim+dim
    perm=list(range(x.ndim));perm[d],perm[-1]=perm[-1],perm[d]
    inv=[0]*len(perm)
    for i,p in enumerate(perm): inv[p]=i
    X=x.permute(*perm).contiguous();prefix=list(X.shape[:-1]);T=X.shape[-1]
    lead=int(np.prod(prefix)) if prefix else 1;Y=X.view(lead,T);mul=L*(1<<J);pad=(-T)%mul
    if pad>0: Y=F.pad(Y,(0,pad))
    nb=Y.shape[1]//L;B=Y.view(-1,nb,L).reshape(-1,L)
    return B,{"perm":perm,"invperm":inv,"prefix":prefix,"T":T,"pad":pad,"L":L,"nb":nb,"lead":lead}

def unblockify_1d(B,meta):
    L,nb,lead,T,pad=meta["L"],meta["nb"],meta["lead"],meta["T"],meta["pad"]
    Y=B.view(lead,nb,L).reshape(lead,nb*L)
    if pad>0: Y=Y[:,:nb*L-pad]
    Y=Y[:,:T];X=Y.view(*meta["prefix"],T)
    return X.permute(*meta["invperm"]).contiguous()

def _dynamic_bits_allocation(vars_list:List[float], counts_list:List[int],
                             avg_bits:float, bmin:int, bmax:int, A_min:int)->List[int]:
    v=np.maximum(np.array(vars_list,dtype=np.float64),1e-12)
    n=np.array(counts_list,dtype=np.float64)
    def mean_bits(lmbd):
        b=np.clip(lmbd+0.5*np.log2(v),bmin,bmax); b[0]=max(b[0],A_min)
        return float((b*n).sum()/n.sum())
    lo,hi=-10.0,10.0
    for _ in range(32):
        mid=(lo+hi)/2.0
        if mean_bits(mid)>avg_bits: hi=mid
        else: lo=mid
    b=np.clip(lo+0.5*np.log2(v),bmin,bmax); b[0]=max(b[0],A_min)
    return np.round(b).astype(int).tolist()

@torch.no_grad()
def _kmeans_codebook(x:torch.Tensor,K:int,iters:int=8,sample:int=10000)->Tuple[torch.Tensor,torch.Tensor]:
    if K<=1: return torch.zeros(1,device=x.device),torch.zeros_like(x,dtype=torch.long)
    xf=x.reshape(-1)
    if sample and xf.numel()>sample:
        idx=torch.randperm(xf.numel(),device=xf.device)[:sample]; data=xf[idx].clone()
    else: data=xf
    qs=torch.linspace(0,1,steps=K+2,device=xf.device)[1:-1]
    c=torch.quantile(data,qs).reshape(K)
    for _ in range(iters):
        d=(data[:,None]-c[None,:]).abs_(); a=d.argmin(1)
        for k in range(K):
            sel=data[a==k]
            if sel.numel()>0: c[k]=sel.mean()
    d_full=(xf[:,None]-c[None,:]).abs_(); a_full=d_full.argmin(1)
    return c,a_full.view_as(x)

def _codebook_bits_fp16(K:int)->int: return K*16

@torch.no_grad()
def band_codebook_quantize(E:torch.Tensor, *, L:int, J:int, axis_dim:int,
                           avg_bits:float, keep_ratio:float,
                           bmin:int, bmax:int, A_min:int,
                           mask_mode:str, cb_sample:int, cb_iters:int,
                           perm:Optional[torch.Tensor]=None, inv:Optional[torch.Tensor]=None):
    X=E if (perm is None) else E.index_select(axis_dim,perm)
    B,meta=blockify_1d(X,L=L,dim=axis_dim,J=J); cA,dets=haar1d_fwd_vec(B,J)
    bands=[("A",cA)]+[(f"D{i+1}",d) for i,d in enumerate(dets)]
    vlist=[float(b[1].float().var(unbiased=False).item()) for b in bands]
    nlist=[b[1].numel() for b in bands]
    bits=_dynamic_bits_allocation(vlist,nlist,avg_bits,bmin,bmax,A_min)
    tot=sum(nlist); energy=np.array([v*n for v,n in zip(vlist,nlist)],dtype=np.float64); w=energy/energy.sum()
    if mask_mode=="uniform": keeps=[int(round(keep_ratio*n)) for n in nlist]
    else: keeps=[min(int(round(keep_ratio*tot*wi)),ni) for wi,ni in zip(w,nlist)]
    recons=[]; ibits=cbits=mbits=0; blobs={}
    for (lbl,Xb),bi,ki in zip(bands,bits,keeps):
        Xf=Xb.reshape(-1)
        if ki<=0 or bi<=0:
            Y=torch.zeros_like(Xf); recons.append(Y.view_as(Xb)); mbits+=Xf.numel()
            blobs[f"mask{lbl}"]=pack_mask(np.zeros(Xf.numel(),dtype=np.uint8)); blobs[f"idx{lbl}"]=b""; blobs[f"cb{lbl}"]=b""
            continue
        ki=min(ki,Xf.numel()); thr=Xf.abs().kthvalue(Xf.numel()-ki+1).values; M=(Xf.abs()>=thr)
        kept=Xf[M]; K=(1<<bi)
        C,assign=_kmeans_codebook(kept,K=K,iters=cb_iters,sample=min(cb_sample,kept.numel()))
        Yk=C[assign]; Y=torch.zeros_like(Xf); Y[M]=Yk; recons.append(Y.view_as(Xb))
        idx_bits=pack_uintegers(assign.to(torch.int64).add_(0).cpu().numpy(),nbits=bi)
        blobs[f"mask{lbl}"]=pack_mask(M.to(torch.uint8).cpu().numpy()); blobs[f"idx{lbl}"]=idx_bits; blobs[f"cb{lbl}"]=to_bytes_fp16(C.detach().cpu().numpy())
        ibits+=ki*bi; cbits+=_codebook_bits_fp16(K); mbits+=Xf.numel()
    RB=haar1d_inv_vec(recons[0],recons[1:]); RB=RB[...,:B.shape[-1]]
    Xrec=unblockify_1d(RB,meta).reshape_as(X); rec_E=Xrec if (inv is None) else Xrec.index_select(axis_dim,inv)
    meta_comp={"codec":"haar1d_cb","J":int(J),"axis":int(axis_dim),"L":int(L),
               "bits":bits,"shape":list(E.shape),"has_perm":perm is not None,"has_inv":inv is not None,
               "perm_len": int(perm.numel()) if perm is not None else 0, "inv_len": int(inv.numel()) if inv is not None else 0}
    if perm is not None: blobs["perm"]=perm.cpu().numpy().astype(np.int32).tobytes()
    if inv  is not None: blobs["inv"]=inv.cpu().numpy().astype(np.int32).tobytes()
    return rec_E, meta_comp, blobs, {"ibits":ibits,"cbits":cbits,"mbits":mbits,"kept":sum(keeps),"bits_vec":bits}
