# tools/cbq_demo.py
import torch, numpy as np, math
from ckptcodec.cbq import blockify_1d,unblockify_1d,haar1d_fwd_vec,haar1d_inv_vec, band_codebook_quantize

torch.manual_seed(7); np.random.seed(7)

def synth_residual(m=256,n=256,rank=6,noise=0.03):
    U=torch.randn(m,rank); V=torch.randn(rank,n)
    E=(U@V)+noise*torch.randn(m,n); return E/(E.std()+1e-8)

@torch.no_grad()
def baseline_uniform_dense(E,L=128,J=2,A_bits=6,D_bits=2,keep_ratio=0.07):
    B,meta=blockify_1d(E,L=L,dim=-1,J=J); cA,dets=haar1d_fwd_vec(B,J)
    def uq(X,b,qq):
        if b<=0: return torch.zeros_like(X),0,0
        vmax=torch.quantile(X.abs(),q=qq,dim=1,keepdim=True).clamp_min(1e-8)
        qmax=2**(b-1)-1; Q=torch.clamp(torch.round(X/(vmax/qmax)),-qmax,qmax)
        Xq=Q*(vmax/qmax); return Xq, X.numel()*b, vmax.numel()*16
    rc=[uq(cA,A_bits,0.999)]; ibit,sbit=rc[0][1],rc[0][2]
    rr=[]; 
    for d in dets:
        xq,ib,sb=uq(d,D_bits,0.99); rr.append(xq); ibit+=ib; sbit+=sb
    RB=haar1d_inv_vec(rc[0][0],rr)[...,:B.shape[-1]]
    Eq=unblockify_1d(RB,meta).reshape_as(E)
    flat=Eq.abs().view(-1); k=max(1,int(round(keep_ratio*flat.numel())))
    thr=flat.kthvalue(flat.numel()-k+1).values; M=(Eq.abs()>=thr); Erec=Eq*M
    mbits=E.numel(); err=(E-Erec).float(); nmse=(err.pow(2).mean()/(E.float().pow(2).mean()+1e-12)).item()
    snr=10.0*math.log10(max(1.0/max(nmse,1e-12),1e-12)); bpp=(ibit+sbit+mbits)/E.numel()
    return {"method":"baseline(A6,D2)+param_mask","bpp":bpp,"snr_db":snr}

def main():
    E=synth_residual()
    base=baseline_uniform_dense(E)
    print(base)
    for avg in [3.0,4.0,5.0]:
        rec,meta,blobs,bstats=band_codebook_quantize(
            E,L=128,J=2,axis_dim=-1,avg_bits=avg,keep_ratio=0.07,
            bmin=0,bmax=6,A_min=4,mask_mode="energy",cb_sample=10000,cb_iters=6)
        err=(E-rec).float(); nmse=(err.pow(2).mean()/(E.float().pow(2).mean()+1e-12)).item()
        snr=10.0*math.log10(max(1.0/max(nmse,1e-12),1e-12))
        bpp=(bstats["ibits"]+bstats["cbits"]+bstats["mbits"])/E.numel()
        print({"method":f"band_codebook(avg={avg})","bits":bstats["bits_vec"],"bpp":bpp,"snr_db":snr})
if __name__=="__main__": main()
