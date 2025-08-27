import io,os,zipfile,json,struct
import numpy as np
try:
    import zstandard as zstd
except Exception:
    zstd=None

def _comp(b,level):
    if zstd is None or level<=0: return b,False
    c=zstd.ZstdCompressor(level=level).compress(b); return c,True

def pack_mask(mask_bool_np:np.ndarray)->bytes:
    return np.packbits(mask_bool_np.reshape(-1).astype(np.uint8),bitorder="little").tobytes()

def pack_uintegers(u:np.ndarray,nbits:int)->bytes:
    u=u.astype(np.uint64).reshape(-1); total_bits=int(u.size*nbits)
    out=np.zeros(((total_bits+7)//8,),dtype=np.uint8); bitpos=0
    for v in u:
        vv=int(v)
        for b in range(nbits):
            if (vv>>b)&1: out[bitpos>>3]|=(1<<(bitpos&7))
            bitpos+=1
    return out.tobytes()

def to_bytes_fp16(xnp:np.ndarray)->bytes:
    return xnp.astype(np.float16).tobytes()

class WarpqWriter:
    def __init__(self,path:str,level:int=5):
        self.z=zipfile.ZipFile(path,"w",compression=zipfile.ZIP_STORED,allowZip64=True)
        self.manifest={"version":1,"items":[],"has_zstd":bool(zstd and level>0)}
        self.level=level; self.idx=0
        self._closed=False; self._manifest_written=False
    def add(self,name:str,meta:dict,blobs:dict):
        if self._closed: return
        rec={"name":name,"meta":meta,"files":{}}
        for k,b in blobs.items():
            comp,flag=_comp(b,self.level); fn=f"{self.idx:05d}.{name}.{k}{'.zst' if flag else '.bin'}"
            self.z.writestr(fn,comp); rec["files"][k]=fn
        self.manifest["items"].append(rec); self.idx+=1
    def close(self):
        if self._closed: return
        # zipfile.ZipFile 关闭后 fp=None；防止二次 writestr
        if getattr(self.z,"fp",None) is not None and not self._manifest_written:
            self.z.writestr("manifest.json",json.dumps(self.manifest,separators=(",",":")))
            self._manifest_written=True
        if getattr(self.z,"fp",None) is not None:
            self.z.close()
        self._closed=True
