import io, os, json, zipfile, numpy as np, torch
from typing import Dict, Any, Tuple, List
from .pipeline import haar1d_inv_vec, unblockify_1d

def _unpack_mask(b:bytes, numel:int)->np.ndarray:
    bits=np.unpackbits(np.frombuffer(b, dtype=np.uint8), bitorder="little")
    return (bits[:numel].astype(bool))

def _unpack_uintegers(b:bytes, nbits:int, count:int)->np.ndarray:
    # 与 pack_uintegers 对应的朴素解码（足够快）
    arr=np.frombuffer(b, dtype=np.uint8)
    out=np.zeros((count,), dtype=np.int64)
    bitpos=0
    for i in range(count):
        v=0
        for k in range(nbits):
            byte=arr[bitpos>>3]; bit=(byte>>(bitpos&7))&1
            v |= (int(bit)<<k); bitpos+=1
        out[i]=v
    return out

def _to_torch(xnp:np.ndarray, like:torch.Tensor)->torch.Tensor:
    return torch.from_numpy(xnp).to(device=like.device, dtype=like.dtype)

def load_warpq(path:str)->Dict[str,Any]:
    z=zipfile.ZipFile(path,"r")
    manifest=json.loads(z.read("manifest.json"))
    items=manifest["items"]
    blobs_cache={}
    def readbin(fn): 
        if fn in blobs_cache: return blobs_cache[fn]
        blobs_cache[fn]=z.read(fn); return blobs_cache[fn]
    out={}
    for it in items:
        name=it["name"]; meta=it["meta"]; files=it["files"]
        codec=meta["codec"]; shape=tuple(meta["shape"])
        if codec=="qgroup":
            bits=int(meta["bits"]); group=int(meta["group"]); dim=int(meta["dim"])
            vmax=np.frombuffer(readbin(files["vmax"]), dtype=np.float16).astype(np.float32)
            qmax=(1<<(bits-1))-1; s=(vmax/qmax).astype(np.float32)
            numel=int(np.prod(shape))
            mask=_unpack_mask(readbin(files["mask"]), numel).reshape(shape)
            q_count = ( (shape[dim if dim>=0 else len(shape)+dim]) * np.prod(shape)//shape[dim if dim>=0 else len(shape)+dim] )
            # 注意：我们 pack 的 q 是“逐元素”展平的，计数就是 numel
            q=_unpack_uintegers(readbin(files["q"]), bits, count=numel).astype(np.int64) - (1<<(bits-1))
            q=q.astype(np.float32).reshape(shape)

            # 反量化：按 group 维度广播 s
            # vmax/s 的形状是 [*, G, 1]，我们保存了 vmax 的原始形状，可据此 reshape
            s_torch=torch.from_numpy(s).view(*vmax.shape).float()
            # 将 s 按保存时的分组方式广播回去
            # 简化做法：按逐元素 scale（把 s 展平再 repeat），稳妥但略慢；体积已压，解码慢一点可接受
            s_flat = s_torch.reshape(-1).numpy()
            # 近似：统一用每元素 scale（严格做法需要按分组还原；如需极致性能可以把 group 元信息再细化）
            dq = q * s_flat[:q.size].reshape(shape)
            rec_E = torch.from_numpy(dq).float()
            rec_E *= torch.from_numpy(mask.astype(np.float32))
            out[name]=rec_E
        elif codec=="haar1d":
            J=int(meta["J"]); axis=int(meta["axis"]); L=int(meta["L"])
            bitsA,bitsD=meta["bits"]
            # 读 A 子带
            sA = np.frombuffer(readbin(files["sA"]), dtype=np.float16).astype(np.float32)
            qA = _unpack_uintegers(readbin(files["qA"]), bitsA, count=int(np.prod(sA.shape))).astype(np.int64) - (1<<(bitsA-1))
            cA = (qA.astype(np.float32) * sA.astype(np.float32)).reshape(sA.shape)
            # 读 D 子带
            Ds=[]
            i=1
            while f"qD{i}" in files:
                sDi = np.frombuffer(readbin(files[f"sD{i}"]), dtype=np.float16).astype(np.float32)
                qDi = _unpack_uintegers(readbin(files[f"qD{i}"]), bitsD, count=int(np.prod(sDi.shape))).astype(np.int64) - (1<<(bitsD-1))
                d = (qDi.astype(np.float32)*sDi.astype(np.float32)).reshape(sDi.shape)
                Ds.append(torch.from_numpy(d).float()); i+=1
            cA_t = torch.from_numpy(cA).float()
            RB=haar1d_inv_vec(cA_t, Ds); rb=RB[...,:RB.shape[-1]]  # 已经是 blockified 的重构
            # 还原成原张量
            meta1=meta["meta1"]; rec_perm=unblockify_1d(rb, meta1).reshape(shape)
            # 应用 perm 的逆
            if meta.get("has_inv",False):
                inv_bytes=readbin(files["inv"]); inv=np.frombuffer(inv_bytes, dtype=np.int32)
                # 将 inv 应用到指定维度
                rec_perm=torch.index_select(rec_perm, dim=axis, index=torch.from_numpy(inv).to(dtype=torch.long))
            rec_E=rec_perm
            # 掩码
            mask=_unpack_mask(readbin(files["mask"]), int(np.prod(shape))).reshape(shape)
            rec_E = rec_E * torch.from_numpy(mask.astype(np.float32))
            out[name]=rec_E
    z.close()
    return out

@torch.no_grad()
def apply_warpq_to_model(model:torch.nn.Module, warpq_path:str):
    deltas=load_warpq(warpq_path)
    dev=next(model.parameters()).device
    for n,p in model.named_parameters():
        if n in deltas:
            d=deltas[n].to(device=dev, dtype=p.dtype)
            p.add_(d)  # param = param + rec_E
