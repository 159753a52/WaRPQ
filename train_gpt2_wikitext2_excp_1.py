# train_gpt2_wikitext2_excp_1.py
import os
import io
import re
import lzma
import json
import math
import time
import argparse
import random
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
import base64
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    EarlyStoppingCallback,
)

def _json_sanitize(obj):
    if isinstance(obj, (bytes, bytearray)):
        return base64.b64encode(obj).decode("ascii")
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    return obj

def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def _ok(name: str, p: torch.Tensor) -> bool:
    l = name.lower()
    if not p.requires_grad: return False
    if l.endswith(".bias") or ".bias" in l: return False
    if "layernorm" in l or re.search(r"\bnorm\b", l) or "ln_" in l: return False
    if "embed" in l or "embedding" in l or "wte" in l or "wpe" in l: return False
    if p.ndim <= 1: return False
    return True

def _H(p: float) -> float:
    return 0.0 if p <= 0.0 or p >= 1.0 else (-p*math.log2(p) - (1-p)*math.log2(1-p))

def _pick_tokenizer_name(args) -> str:
    if getattr(args, "tokenizer_name", None):
        return args.tokenizer_name
    if os.path.isdir(args.model_name):
        for f in ["tokenizer.json","vocab.json","merges.txt","tokenizer_config.json","special_tokens_map.json"]:
            if os.path.exists(os.path.join(args.model_name, f)):
                return args.model_name
    return "openai-community/gpt2"

@torch.no_grad()
def lm_loss_on_split(raw_split, tok, model, block_size=256, batch_size=2, max_batches=200):
    def _tok(b):
        return tok(b["text"], return_special_tokens_mask=True)
    tokenized = raw_split.map(_tok, batched=True, remove_columns=["text"])

    bs = block_size
    def group(ex):
        c = {k: sum(ex[k], []) for k in ex.keys()}
        t = (len(c["input_ids"]) // bs) * bs
        r = {k: [v[i:i+bs] for i in range(0, t, bs)] for k, v in c.items()}
        r["labels"] = r["input_ids"].copy()
        return r
    lm_ds = tokenized.map(group, batched=True)

    coll = DataCollatorForLanguageModeling(tok, mlm=False)
    dev = next(model.parameters()).device
    dl = torch.utils.data.DataLoader(lm_ds, batch_size=batch_size, shuffle=False, collate_fn=coll)

    model.eval()
    total_tokens, total_loss = 0, 0.0
    for i, batch in enumerate(dl):
        if i >= max_batches: break
        batch = {k: v.to(dev) for k, v in batch.items()}
        out = model(**batch)
        labels = batch["labels"]
        nt = int((labels != -100).sum().item())
        total_loss += float(out.loss.item()) * nt
        total_tokens += nt
    if total_tokens == 0:
        return float("nan"), float("nan")
    avg_nll = total_loss / total_tokens
    return avg_nll, math.exp(avg_nll)

def sizeof_fmt(num: float, suffix: str = "B") -> str:
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Y{suffix}"

def robust_quantile(x: torch.Tensor, q: float, sample_cap: int = 2_000_000) -> torch.Tensor:
    xf = x.reshape(-1)
    n = xf.numel()
    if n == 0:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)
    if n <= sample_cap:
        k = max(1, int(math.ceil(q * n)))
        return torch.kthvalue(xf, k).values
    idx = torch.randint(low=0, high=n, size=(sample_cap,), device=x.device)
    xs = xf[idx]
    k = max(1, int(math.ceil(q * sample_cap)))
    return torch.kthvalue(xs, k).values

def _pack_nibbles(idx: torch.Tensor) -> bytes:
    assert idx.dtype == torch.uint8
    if idx.numel() % 2 == 1:
        idx = torch.cat([idx, torch.zeros(1, dtype=torch.uint8, device=idx.device)])
    hi = idx.view(-1, 2)[:, 0]
    lo = idx.view(-1, 2)[:, 1]
    packed = (hi << 4) | lo
    return bytes(packed.cpu().numpy().tobytes())

def _unpack_nibbles(buf: bytes, total: int) -> torch.Tensor:
    a = np.frombuffer(buf, dtype=np.uint8)
    a = torch.from_numpy(a.copy())
    hi = (a >> 4) & 0xF
    lo = a & 0xF
    idx = torch.stack([hi, lo], dim=1).view(-1)
    if idx.numel() > total:
        idx = idx[:total]
    return idx

@dataclass
class QuantizedTensor:
    shape: Tuple[int, ...]
    centers: torch.Tensor
    packed_idx: bytes
    nbits: int
    def to_bytes(self) -> Dict[str, Any]:
        ctr_b = self.centers.detach().cpu().numpy().astype(np.float16).tobytes()
        return {
            "shape": list(self.shape),
            "K": int(self.centers.numel()),
            "nbits": int(self.nbits),
            "centers_dtype": "float16",
            "centers_b64": base64.b64encode(ctr_b).decode("ascii"),
            "packed_idx_b64": base64.b64encode(self.packed_idx).decode("ascii"),
        }
    @staticmethod
    def from_bytes(d: Dict[str, Any], device: torch.device) -> "QuantizedTensor":
        shape = tuple(d["shape"])
        K = int(d["K"])
        ctr = np.frombuffer(base64.b64decode(d["centers_b64"]), dtype=np.float16).astype(np.float32)
        centers = torch.from_numpy(ctr).to(device).view(K)
        packed = base64.b64decode(d["packed_idx_b64"])
        return QuantizedTensor(shape=shape, centers=centers, packed_idx=packed, nbits=int(d["nbits"]))
    def dequantize(self, device: torch.device) -> torch.Tensor:
        total = int(np.prod(self.shape))
        idx = _unpack_nibbles(self.packed_idx, total).to(device)
        out = torch.zeros(total, device=device, dtype=torch.float32)
        nz = idx > 0
        if nz.any():
            out[nz] = self.centers[(idx[nz] - 1).long()]
        return out.view(*self.shape)

class JsonLoggingCallback(TrainerCallback):
    def __init__(self, json_log_path): self.json_log_path=json_log_path; self.buf=[]
    def on_train_begin(self, args, state, control, **kwargs):
        os.makedirs(os.path.dirname(self.json_log_path), exist_ok=True)
        with open(self.json_log_path, "w") as f: json.dump([], f)
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None: return
        e = {**logs, "step": int(state.global_step)}
        self.buf.append(e)
        with open(self.json_log_path, "w") as f: json.dump(self.buf, f, indent=2)

def kmeans_nonuniform_quantize(
    x: torch.Tensor,
    nbits: int,
    max_iters: int = 12,
    assign_chunk: int = 2_000_000,
) -> QuantizedTensor:
    assert nbits >= 2, "nbits must be >= 2"
    x = x.contiguous()
    flat = x.reshape(-1).to(torch.float32)
    device = flat.device

    nz_mask = flat != 0
    nz = flat[nz_mask]
    K = (1 << nbits) - 1

    if nz.numel() == 0:
        centers = torch.zeros(K, device=device, dtype=torch.float32)
        idx = torch.zeros_like(flat, dtype=torch.uint8)
        return QuantizedTensor(tuple(x.shape), centers, _pack_nibbles(idx), nbits)

    q_lo = torch.quantile(nz, 0.005)
    q_hi = torch.quantile(nz, 0.995)
    if not torch.isfinite(q_lo): q_lo = nz.min()
    if not torch.isfinite(q_hi): q_hi = nz.max()
    if q_hi <= q_lo:
        q_lo = nz.min()
        q_hi = nz.max() if nz.max() > nz.min() else (q_lo + 1e-6)

    centers = torch.linspace(q_lo, q_hi, K, device=device, dtype=torch.float32)

    for _ in range(max_iters):
        counts = torch.zeros(K, device=device, dtype=torch.long)
        sums   = torch.zeros(K, device=device, dtype=torch.float32)

        N = nz.numel()
        for s in range(0, N, assign_chunk):
            e = min(s + assign_chunk, N)
            vals = nz[s:e]
            d = (vals.unsqueeze(1) - centers.unsqueeze(0)) ** 2
            a = torch.argmin(d, dim=1)
            counts += torch.bincount(a, minlength=K)
            sums   += torch.bincount(a, weights=vals, minlength=K)

        nonempty = counts > 0
        if nonempty.any():
            centers[nonempty] = sums[nonempty] / counts[nonempty]

        empty = ~nonempty
        if empty.any():
            lo, hi = nz.min(), nz.max()
            centers[empty] = torch.linspace(lo, hi, empty.sum(), device=device)

    idx = torch.zeros_like(flat, dtype=torch.uint8)
    nz_idx = torch.where(nz_mask)[0]
    N = nz_idx.numel()
    for s in range(0, N, assign_chunk):
        e = min(s + assign_chunk, N)
        sel = nz_idx[s:e]
        vals = flat[sel]
        d = (vals.unsqueeze(1) - centers.unsqueeze(0)) ** 2
        a = torch.argmin(d, dim=1)
        idx[sel] = (a + 1).to(torch.uint8)

    return QuantizedTensor(tuple(x.shape), centers, _pack_nibbles(idx), nbits)

@dataclass
class ExCPConfig:
    save_dir: str
    compress_every: int = 1000
    start_step: int = 0
    nbits: int = 4
    alpha: float = 5e-5
    beta: float = 2.0
    keep_init_state: bool = True
    include_bias_and_ln: bool = True
    target_nz: float = 0.10
    enforce_target: bool = False
    apply_to_model: bool = False
    momentum_policy: str = "none"

class ExCPCheckpointer(TrainerCallback):
    def __init__(self, cfg: ExCPConfig):
        self.cfg = cfg
        self._last_reconstructed: Dict[str, torch.Tensor] = {}
        self._step0_saved = False

    @staticmethod
    def _find_group_for_param(optimizer: Optional[Optimizer], param: torch.Tensor):
        if optimizer is None:
            return None
        for g in optimizer.param_groups:
            for p in g.get("params", []):
                if p is param:
                    return g
        return None

    def _ensure_dirs(self):
        os.makedirs(self.cfg.save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.cfg.save_dir, "ckpts"), exist_ok=True)

    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        self._ensure_dirs()
        if self.cfg.keep_init_state and not self._step0_saved:
            init_path = os.path.join(self.cfg.save_dir, "init_weights.pt")
            if not os.path.exists(init_path):
                torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, init_path)
                self._last_reconstructed = {k: v.detach().to(next(model.parameters()).device) for k, v in model.state_dict().items()}
                self._step0_saved = True
        meta = {
            "nbits": self.cfg.nbits,
            "alpha": self.cfg.alpha,
            "beta": self.cfg.beta,
            "compress_every": self.cfg.compress_every,
            "start_step": self.cfg.start_step,
        }
        with open(os.path.join(self.cfg.save_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, model=None, optimizer: Optional[Optimizer] = None, **kwargs):
        step = int(state.global_step)
        if step < self.cfg.start_step:
            return
        if step % self.cfg.compress_every != 0:
            return
        self._compress_now(step, model, optimizer)

    @torch.no_grad()
    def _compress_now(self, step: int, model: nn.Module, optimizer: Optional[Optimizer]):
        t0 = time.perf_counter()
        dev = next(model.parameters()).device

        if not self._last_reconstructed:
            self._last_reconstructed = {k: v.detach().to(dev) for k, v in model.state_dict().items()}

        name_to_param = dict(model.named_parameters())
        id2name = {id(p): n for n, p in name_to_param.items()}

        optim_states: Dict[str, Dict[str, torch.Tensor]] = {}
        if optimizer is not None:
            for g in optimizer.param_groups:
                for p in g.get("params", []):
                    n = id2name.get(id(p))
                    if n is None:
                        continue
                    st = optimizer.state.get(p, {})
                    m = st.get("exp_avg", None)
                    v = st.get("exp_avg_sq", None)
                    if isinstance(m, torch.Tensor):
                        optim_states.setdefault(n, {})["m"] = m.detach().to(dev).float()
                    if isinstance(v, torch.Tensor):
                        optim_states.setdefault(n, {})["v"] = v.detach().to(dev).float()

        payload_w: Dict[str, Any] = {}
        payload_o: Dict[str, Any] = {}

        total_params = 0
        kept_params  = 0

        for name, p_obj in name_to_param.items():
            w_cur = p_obj.detach().to(dev).float()
            w_last = self._last_reconstructed.get(name, w_cur.clone())
            dW = (w_cur - w_last)

            st = optim_states.get(name, {})
            v_mean = float(st.get("v", torch.zeros(1, device=dev)).mean().item()) if "v" in st else 0.0
            rw = self.cfg.alpha / math.sqrt(v_mean + 1e-12)
            Mw = (dW.abs() > rw)

            if self.cfg.enforce_target:
                flat = dW.abs().view(-1)
                k = max(1, int(round(self.cfg.target_nz * flat.numel())))
                if k < flat.numel():
                    thr = torch.kthvalue(flat, flat.numel() - k + 1).values
                    Mk = (dW.abs() >= thr)
                else:
                    Mk = torch.ones_like(dW, dtype=torch.bool)
                keep_mask = Mk
            else:
                keep_mask = Mw

            dW_kept = dW * keep_mask

            m1p = None; v2p = None
            if "m" in st:
                m1 = st["m"]
                mo_thr = self.cfg.beta * float(m1.abs().mean().item())
                Mo = (m1.abs() > mo_thr) & keep_mask
                m1p = (m1 * Mo).to(torch.float32)
                if "v" in st and isinstance(st["v"], torch.Tensor):
                    v2p = (st["v"] * Mo).to(torch.float32)

            qt_dW = kmeans_nonuniform_quantize(dW_kept, nbits=self.cfg.nbits)
            payload_w[name] = qt_dW.to_bytes()

            if m1p is not None:
                qt_m = kmeans_nonuniform_quantize(m1p, nbits=self.cfg.nbits)
                payload_o[name] = {"m": qt_m.to_bytes()}
                if v2p is not None:
                    qt_v = kmeans_nonuniform_quantize(v2p, nbits=self.cfg.nbits)
                    payload_o[name]["v"] = qt_v.to_bytes()

            recon_dW = qt_dW.dequantize(device=dev)
            w_next   = (w_last + recon_dW).to(torch.float32)
            self._last_reconstructed[name] = w_next.clone()

            if self.cfg.apply_to_model:
                p_obj.data.copy_(w_next.to(p_obj.dtype))

            if self.cfg.apply_to_model and optimizer is not None and self.cfg.momentum_policy == "reset_changed":
                st_ = optimizer.state.get(p_obj, {})
                if "exp_avg" in st_ and isinstance(st_["exp_avg"], torch.Tensor):
                    delta = (recon_dW - dW).abs()
                    thr = robust_quantile(delta, 0.90, sample_cap=2_000_000)
                    changed = (delta >= thr)
                    st_["exp_avg"].masked_fill_(changed, 0)
                    if "exp_avg_sq" in st_ and isinstance(st_["exp_avg_sq"], torch.Tensor):
                        st_["exp_avg_sq"].masked_fill_(changed, 0)

            n = dW.numel()
            k_kept = int(keep_mask.sum().item())
            total_params += n
            kept_params  += k_kept

        pt = {
            "step": int(step),
            "nbits": self.cfg.nbits,
            "alpha": self.cfg.alpha,
            "beta": self.cfg.beta,
            "target_nz": self.cfg.target_nz,
            "enforce_target": self.cfg.enforce_target,
            "weights": payload_w,
            "optimizer": payload_o,
        }
        raw = json.dumps(pt).encode("utf-8")
        comp = lzma.compress(raw, preset=9 | lzma.PRESET_EXTREME)
        out = os.path.join(self.cfg.save_dir, "ckpts", f"excp_{step:08d}.xz")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "wb") as f:
            f.write(comp)

        p_keep = kept_params / max(1, total_params)
        def H(p):
            return 0.0 if p <= 0.0 or p >= 1.0 else (-p*math.log2(p) - (1-p)*math.log2(1-p))
        bpp = p_keep * self.cfg.nbits + H(p_keep)
        cr  = 32.0 / max(bpp, 1e-9)
        t1 = time.perf_counter()
        stats = {
            "step": int(step),
            "bytes": len(comp),
            "human": sizeof_fmt(len(comp)),
            "p_keep": float(p_keep),
            "bpp": float(bpp),
            "cr_weight": float(cr),
            "time_s": (t1 - t0),
        }
        with open(os.path.join(self.cfg.save_dir, "compress_stats.jsonl"), "a") as f:
            f.write(json.dumps(stats) + "\n")

        print(f"[ExCP] step={step} | keep={p_keep*100:.2f}% | bpp={bpp:.3f} | CR≈{cr:.1f}x | wrote {sizeof_fmt(len(comp))} in {t1-t0:.2f}s")

@torch.no_grad()
def excp_reconstruct_to_state_dict(compressed_dir: str, upto_step: Optional[int] = None, init_weights_path: Optional[str] = None, device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
    assert init_weights_path is not None and os.path.exists(init_weights_path), "init_weights.pt is required"
    if device is None:
        device = torch.device("cpu")
    base = torch.load(init_weights_path, map_location="cpu")
    recon = {k: v.clone().to(device) for k, v in base.items()}
    xs = sorted([x for x in os.listdir(os.path.join(compressed_dir, "ckpts")) if x.endswith(".xz")])
    for fn in xs:
        m = re.match(r"excp_(\d+)\.xz", fn)
        if not m:
            continue
        step = int(m.group(1))
        if upto_step is not None and step > upto_step:
            break
        with open(os.path.join(compressed_dir, "ckpts", fn), "rb") as f:
            buf = f.read()
        d = json.loads(lzma.decompress(buf).decode("utf-8"))
        for name, qd in d["weights"].items():
            qt = QuantizedTensor.from_bytes(qd, device=device)
            recon[name] = recon[name].to(torch.float32)
            recon[name] += qt.dequantize(device=device)
    return recon

@torch.no_grad()
def strided_perplexity(raw_split, tokenizer, model, stride=512):
    device = next(model.parameters()).device
    text = "\n\n".join(raw_split["text"])
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids.to(device)
    max_length = getattr(model.config, "n_positions", tokenizer.model_max_length)
    if max_length is None or max_length > tokenizer.model_max_length:
        max_length = tokenizer.model_max_length
    nll_sum = 0.0
    n_tokens = 0
    prev_end = 0
    seq_len = input_ids.size(1)
    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        trg_len = end - prev_end
        inp = input_ids[:, begin:end]
        tgt = inp.clone()
        tgt[:, :-trg_len] = -100
        outputs = model(inp, labels=tgt)
        num_valid = (tgt != -100).sum().item()
        batch = tgt.size(0)
        num_loss_tokens = num_valid - batch
        nll_sum += outputs.loss * num_loss_tokens
        n_tokens += num_loss_tokens
        prev_end = end
        if end == seq_len:
            break
    avg_nll = nll_sum / n_tokens
    ppl = torch.exp(avg_nll).item()
    return avg_nll.item(), ppl

def _is_qkv(name: str) -> bool:
    l = name.lower()
    return ("c_attn.weight" in l) or ("attention.query_key_value.weight" in l) \
           or ("q_proj" in l) or ("k_proj" in l) or ("v_proj" in l)

@torch.no_grad()
def quantize_pergroup_uniform(t: torch.Tensor, n_bits: int, group: int, dim: int = -1, clip: float = 0.995):
    if n_bits <= 0 or t.ndim == 0 or t.numel() == 0:
        return t.clone()
    dtype = t.dtype
    dim = dim if dim >= 0 else t.ndim + dim
    p = list(range(t.ndim)); p[dim], p[-1] = p[-1], p[dim]
    x = t.permute(*p).contiguous(); sh = x.shape; L = sh[-1]; x = x.view(-1, L)
    pad = (group - (L % group)) % group
    if pad > 0:
        x = torch.cat([x, torch.zeros(x.shape[0], pad, device=x.device, dtype=x.dtype)], 1)
    G = x.shape[1] // group; x = x.view(-1, G, group)
    vmax = torch.quantile(x.abs(), q=clip, dim=-1, keepdim=True).clamp_min(1e-8)
    qmax = 2 ** (n_bits - 1) - 1
    q = torch.clamp(torch.round(x / (vmax / qmax)), min=-qmax, max=qmax)
    deq = q * (vmax / qmax)
    deq = deq.view(-1, G * group); deq = deq[:, :L] if pad > 0 else deq; deq = deq.view(*sh[:-1], L)
    inv = list(range(t.ndim)); inv[dim], inv[-1] = inv[-1], inv[dim]
    return deq.permute(*inv).contiguous().to(dtype=dtype)

@torch.no_grad()
def project_topk_tensor(x: torch.Tensor, p_keep: float):
    if p_keep >= 1.0: return x, torch.ones_like(x, dtype=torch.bool)
    if p_keep <= 0.0: return torch.zeros_like(x), torch.zeros_like(x, dtype=torch.bool)
    flat = x.abs().reshape(-1)
    k = max(1, int(round(p_keep * flat.numel())))
    if k >= flat.numel():
        m = torch.ones_like(x, dtype=torch.bool)
        return x, m
    thr = torch.kthvalue(flat, flat.numel() - k + 1).values
    m = (x.abs() >= thr)
    return x * m, m

@torch.no_grad()
def load_optimizer_moments_by_order(model: nn.Module, ckpt_dir: str, device: torch.device):
    opt_pt = os.path.join(ckpt_dir, "optimizer.pt")
    if not os.path.exists(opt_pt):
        print(f"[ExCP OneShot] WARN: optimizer.pt not found under: {ckpt_dir}")
        return {}
    try:
        obj = torch.load(opt_pt, map_location="cpu")
        state: dict = obj.get("state", {})
        groups: list = obj.get("param_groups", [])
        opt_param_ids = []
        for g in groups:
            opt_param_ids.extend(list(g.get("params", [])))
        moments = {}
        names = [n for n, _ in model.named_parameters()]
        params = [p for _, p in model.named_parameters()]
        i = 0
        for pid in opt_param_ids:
            if i >= len(params): break
            name, p = names[i], params[i]
            st = state.get(pid, None)
            if st is not None:
                m = st.get("exp_avg", None)
                v = st.get("exp_avg_sq", None)
                entry = {}
                if isinstance(m, torch.Tensor) and m.numel() == p.numel():
                    entry["m"] = m.to(device=device, dtype=torch.float32).view_as(p)
                if isinstance(v, torch.Tensor) and v.numel() == p.numel():
                    entry["v"] = v.to(device=device, dtype=torch.float32).view_as(p)
                if entry:
                    moments[name] = entry
            i += 1
        n_all = sum(1 for _ in model.named_parameters())
        n_have = len(moments)
        n_have_v = sum(1 for k in moments if "v" in moments[k])
        print(f"[ExCP OneShot] Loaded optimizer moments for {n_have}/{n_all} params (with v: {n_have_v}). Source: {opt_pt}")
        return moments
    except Exception as e:
        print(f"[ExCP OneShot] WARN: failed to parse optimizer moments: {e}")
        return {}

@torch.no_grad()
def global_threshold_T(model: nn.Module,
                       baseline: Dict[str, torch.Tensor],
                       opt_moments: Dict[str, Dict[str, torch.Tensor]],
                       target_keep: float,
                       sample_cap: int = 5_000_000) -> float:
    pool = []
    total = 0
    for name, p in model.named_parameters():
        if not _ok(name, p): continue
        W = p.data.detach()
        B = baseline.get(name, torch.zeros_like(W, device="cpu")).to(W.device, dtype=W.dtype)
        E = (W - B).float()
        if name in opt_moments and ("v" in opt_moments[name]):
            v_mean = float(opt_moments[name]["v"].mean().item())
        else:
            v_mean = 1.0
        z = E.abs() * math.sqrt(v_mean + 1e-12)
        flat = z.reshape(-1)
        total += flat.numel()
        take = min(flat.numel(), max(1, sample_cap // max(1, len(pool)+1)))
        if take < flat.numel():
            idx = torch.randint(0, flat.numel(), (take,), device=flat.device)
            pool.append(flat[idx].detach().cpu())
        else:
            pool.append(flat.detach().cpu())
    if not pool:
        return 0.0
    Z = torch.cat(pool, dim=0)
    k = max(1, int(round(target_keep * Z.numel())))
    thr = torch.kthvalue(Z, Z.numel() - k + 1).values.item()
    return float(thr)

def _layer_group(name: str) -> str:
    l = name.lower()
    if ("c_attn.weight" in l) or ("attention.query_key_value.weight" in l) or ("q_proj" in l) or ("k_proj" in l) or ("v_proj" in l):
        return "attn_qkv"
    if ("attn.c_proj.weight" in l) or ("attention.out_proj.weight" in l) or ("o_proj" in l):
        return "attn_out"
    if (".mlp.c_fc.weight" in l) or ("mlp.fc_in.weight" in l) or ("gate_proj" in l) or ("up_proj" in l):
        return "mlp_fc"
    if (".mlp.c_proj.weight" in l) or ("mlp.fc_out.weight" in l) or ("down_proj" in l):
        return "mlp_proj"
    return "others"

def _parse_group_keep_spec(spec: str) -> Dict[str, float]:
    out = {}
    if not spec: return out
    for item in spec.split(","):
        item = item.strip()
        if not item: continue
        k, v = item.split("=")
        out[k.strip()] = float(v)
    return out

@torch.no_grad()
def groupwise_threshold_T(model: nn.Module,
                          baseline: Dict[str, torch.Tensor],
                          opt_moments: Dict[str, Dict[str, torch.Tensor]],
                          default_keep: float,
                          keep_spec: Optional[Dict[str, float]] = None,
                          sample_cap_per_group: int = 2_000_000) -> Dict[str, float]:
    keep_spec = keep_spec or {}
    buckets: Dict[str, List[torch.Tensor]] = {}
    for name, p in model.named_parameters():
        if not _ok(name, p):
            continue
        g = _layer_group(name)
        W = p.data.detach()
        B = baseline.get(name, torch.zeros_like(W, device="cpu")).to(W.device, dtype=W.dtype)
        E = (W - B).float()
        if name in opt_moments and ("v" in opt_moments[name]):
            v_mean = float(opt_moments[name]["v"].mean().item())
        else:
            v_mean = 1.0
        z = E.abs() * math.sqrt(v_mean + 1e-12)
        flat = z.reshape(-1)
        take = min(sample_cap_per_group, flat.numel())
        if take < flat.numel():
            idx = torch.randint(0, flat.numel(), (take,), device=flat.device)
            s = flat[idx].detach().cpu()
        else:
            s = flat.detach().cpu()
        buckets.setdefault(g, []).append(s)

    T: Dict[str, float] = {}
    for g, parts in buckets.items():
        s = torch.cat(parts, dim=0)
        if s.numel() == 0:
            T[g] = 0.0
            continue
        pk = keep_spec.get(g, default_keep)
        k = max(1, int(round(pk * s.numel())))
        thr = torch.kthvalue(s, s.numel() - k + 1).values.item()
        T[g] = float(thr)
    return T

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="openai-community/gpt2")
    ap.add_argument("--output_dir", type=str, default="./runs/excp")
    ap.add_argument("--lm_block_size", type=int, default=256)
    ap.add_argument("--per_device_train_batch_size", type=int, default=2)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--lr_scheduler_type", type=str, default="cosine")
    ap.add_argument("--eval_freq", type=int, default=200)
    ap.add_argument("--save_freq", type=int, default=200)
    ap.add_argument("--early_stop_patience", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--disable_early_stop", action="store_true", default=True)

    ap.add_argument("--excp_save_dir", type=str, default=None)
    ap.add_argument("--excp_start_step", type=int, default=400)
    ap.add_argument("--excp_every", type=int, default=1000)
    ap.add_argument("--excp_nbits", type=int, default=4)
    ap.add_argument("--excp_alpha", type=float, default=5e-5)
    ap.add_argument("--excp_beta", type=float, default=2.0)
    ap.add_argument("--excp_keep_init", action="store_true", default=True)

    ap.add_argument("--reconstruct_only", action="store_true", default=False)
    ap.add_argument("--reconstruct_upto", type=int, default=None)
    ap.add_argument("--reconstruct_out", type=str, default=None)

    ap.add_argument("--excp_apply_to_model", action="store_true", default=False)
    ap.add_argument("--excp_momentum_policy", type=str, default="none", choices=["none", "reset_changed"])
    ap.add_argument("--excp_target_nz", type=float, default=0.10)
    ap.add_argument("--excp_enforce_target", action="store_true", default=False)

    ap.add_argument("--single_shot", action="store_true", default=False)
    ap.add_argument("--single_shot_step", type=int, default=800)
    ap.add_argument("--single_shot_apply", action="store_true", default=True)
    ap.add_argument("--single_shot_eval", action="store_true", default=True)
    ap.add_argument("--single_shot_residual", action="store_true", default=True)
    ap.add_argument("--baseline_path", type=str, default=None)
    ap.add_argument("--tokenizer_name", type=str, default=None)
    ap.add_argument("--max_eval_batches", type=int, default=200)

    ap.add_argument("--excp_kmeans_iters", type=int, default=12)
    ap.add_argument("--excp_kmeans_sample", type=int, default=1_000_000)
    ap.add_argument("--excp_qkv_bits", type=int, default=None)
    ap.add_argument("--excp_qkv_group", type=int, default=128)
    ap.add_argument("--excp_qkv_clip_pct", type=float, default=0.99)
    ap.add_argument("--print_top_layers", type=int, default=12)
    ap.add_argument("--excp_kmeans_qkv", action="store_true", default=False)

    ap.add_argument("--excp_use_opt_threshold", action="store_true", default=True,
                    help="Use optimizer v-stat to set rw = T/sqrt(mean(v)) instead of global top-k.")
    ap.add_argument("--optimizer_path", type=str, default=None)
    ap.add_argument("--excp_calibrate_keep", type=float, default=None,
                    help="If set in (0,1], calibrate T by |E|*sqrt(mean(v)) quantile so keep≈this ratio.")
    ap.add_argument("--excp_groupwise_calibrate", action="store_true", default=False,
                    help="Calibrate T per layer-group instead of globally.")
    ap.add_argument("--excp_min_keep_floor", type=float, default=0.02,
                    help="Per-layer minimum keep floor to avoid starving a whole layer.")
    ap.add_argument("--excp_group_keep_spec", type=str, default="",
                    help="Optional: 'attn_qkv=0.12,attn_out=0.10,mlp_fc=0.08,mlp_proj=0.08'")

    args = ap.parse_args()
    set_seed(args.seed)

    if args.reconstruct_only:
        assert args.excp_save_dir is not None, "--excp_save_dir required"
        init_path = os.path.join(args.excp_save_dir, "init_weights.pt")
        state = excp_reconstruct_to_state_dict(args.excp_save_dir, upto_step=args.reconstruct_upto, init_weights_path=init_path)
        out = args.reconstruct_out or os.path.join(args.excp_save_dir, f"reconstructed_{args.reconstruct_upto or 'final'}.pt")
        torch.save(state, out)
        print(f"Reconstructed state_dict saved to: {out}")
        return

    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.config.pad_token_id = tokenizer.pad_token_id

    def tok(batch):
        return tokenizer(batch["text"], return_special_tokens_mask=True)

    tokenized = ds.map(tok, batched=True, remove_columns=["text"])

    block_size = args.lm_block_size
    def group_texts(examples):
        concat = {k: sum(examples[k], []) for k in examples.keys()}
        total = (len(concat["input_ids"]) // block_size) * block_size
        result = {k: [t[i : i + block_size] for i in range(0, total, block_size)] for k, t in concat.items()}
        result["labels"] = result["input_ids"].copy()
        return result

    lm_ds = tokenized.map(group_texts, batched=True)
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        seed=args.seed,
        eval_strategy="steps",
        eval_steps=args.eval_freq,
        save_strategy="steps",
        save_steps=args.save_freq,
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        logging_steps=50,
        logging_first_step=True,
        report_to="none",
        save_safetensors=True,
        group_by_length=True,
    )

    excp_dir = args.excp_save_dir or os.path.join(args.output_dir, "excp")
    cfg = ExCPConfig(
        save_dir=excp_dir,
        compress_every=args.excp_every,
        start_step=args.excp_start_step,
        nbits=args.excp_nbits,
        alpha=args.excp_alpha,
        beta=args.excp_beta,
        keep_init_state=args.excp_keep_init,
        apply_to_model=args.excp_apply_to_model,
        momentum_policy=args.excp_momentum_policy,
        target_nz=args.excp_target_nz,
        enforce_target=args.excp_enforce_target,
    )
    excp_cb = ExCPCheckpointer(cfg)

    callbacks = [JsonLoggingCallback(os.path.join(args.output_dir, "training_log.json")), excp_cb]
    if (not args.disable_early_stop) and args.early_stop_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stop_patience, early_stopping_threshold=0.0))

    if args.single_shot:
        set_seed(args.seed)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tok = AutoTokenizer.from_pretrained(_pick_tokenizer_name(args), use_fast=True)
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(args.model_name)
        model.config.pad_token_id = tok.pad_token_id
        model.to(dev).eval()

        ds = load_dataset("wikitext", "wikitext-2-raw-v1")

        if args.single_shot_eval:
            nll0, ppl0 = lm_loss_on_split(
                ds["train"], tok, model,
                block_size=args.lm_block_size,
                batch_size=args.per_device_eval_batch_size,
                max_batches=args.max_eval_batches
            )
            print(f"[ExCP OneShot] TRAIN BEFORE  loss={nll0:.4f}, ppl={ppl0:.2f}")

        sd = model.state_dict()
        if args.single_shot_residual:
            if args.baseline_path and os.path.exists(args.baseline_path):
                bl = torch.load(args.baseline_path, map_location="cpu")
                baseline = {k: bl.get(k, torch.zeros_like(v, device="cpu")).to(dtype=v.dtype).cpu()
                            for k, v in sd.items()}
                print(f"[ExCP OneShot] Loaded baseline from: {args.baseline_path}")
            else:
                print("[ExCP OneShot] WARN: baseline_path not found; use baseline=0.")
                baseline = {k: torch.zeros_like(v).cpu() for k, v in sd.items()}
        else:
            baseline = {k: torch.zeros_like(v).cpu() for k, v in sd.items()}

        step = int(args.single_shot_step)
        print(f"[ExCP OneShot] QKV quantizer = {'k-means' if args.excp_kmeans_qkv else 'per-group uniform'}, "
              f"qkv_bits={args.excp_qkv_bits or args.excp_nbits}, target_nz={args.excp_target_nz}")

        kept, total = 0, 0
        mse_sum, var_sum = 0.0, 0.0
        per_layer_err = []

        opt_moments = {}
        src_opt_dir = args.optimizer_path if args.optimizer_path is not None else args.model_name
        if src_opt_dir is not None:
            opt_moments = load_optimizer_moments_by_order(model, src_opt_dir, dev)

        global_T = None
        group_T: Dict[str, float] = {}
        keep_spec = _parse_group_keep_spec(args.excp_group_keep_spec)
        if args.excp_use_opt_threshold and (args.excp_calibrate_keep is not None) and (0.0 < args.excp_calibrate_keep <= 1.0):
            if args.excp_groupwise_calibrate:
                group_T = groupwise_threshold_T(model, baseline, opt_moments,
                                                default_keep=float(args.excp_calibrate_keep),
                                                keep_spec=keep_spec)
                print(f"[ExCP OneShot] Group-wise v-calibrated thresholds: { {k: round(v,6) for k,v in group_T.items()} }")
            else:
                global_T = global_threshold_T(model, baseline, opt_moments, float(args.excp_calibrate_keep))
                print(f"[ExCP OneShot] Global v-calibrated threshold T computed for target keep={args.excp_calibrate_keep:.2f}")

        t0 = time.perf_counter()
        with torch.no_grad():
            for name, p in model.named_parameters():
                if not _ok(name, p):
                    continue
                W = p.data.detach().to(dev).float()
                B = baseline.get(name, torch.zeros_like(W, device="cpu")).to(dev, dtype=W.dtype)
                E = W - B

                if args.excp_use_opt_threshold:
                    if name in opt_moments and ("v" in opt_moments[name]):
                        v_mean = float(opt_moments[name]["v"].mean().item())
                    else:
                        v_mean = 1.0
                    if group_T:
                        Tg = group_T.get(_layer_group(name), 0.0)
                        rw = Tg / math.sqrt(v_mean + 1e-12)
                    elif global_T is not None:
                        rw = global_T / math.sqrt(v_mean + 1e-12)
                    else:
                        rw = args.excp_alpha / math.sqrt(v_mean + 1e-12)
                    keep_mask = (E.abs() > rw)
                    E_kept = E * keep_mask
                else:
                    if args.excp_enforce_target:
                        E_kept, keep_mask = project_topk_tensor(E, args.excp_target_nz)
                    else:
                        keep_mask = (E.abs() > args.excp_alpha)
                        E_kept = E * keep_mask

                if args.excp_min_keep_floor > 0.0:
                    want = int(math.ceil(args.excp_min_keep_floor * E.numel()))
                    have = int(keep_mask.sum().item())
                    if have < want:
                        flat = E.abs().reshape(-1)
                        k = want
                        thr_l = torch.kthvalue(flat, flat.numel() - k + 1).values
                        keep_mask = keep_mask | (E.abs() >= thr_l)
                        E_kept = E * keep_mask

                def _kmeans_deq(x: torch.Tensor, nbits: int, iters: int, dev: torch.device):
                    x = x.contiguous()
                    qt = kmeans_nonuniform_quantize(x, nbits=nbits, max_iters=iters)
                    return qt.dequantize(device=dev)

                if dev.type == "cuda": torch.cuda.synchronize()
                tq0 = time.perf_counter()

                b_qkv = int(args.excp_qkv_bits or args.excp_nbits)
                if _is_qkv(name):
                    if args.excp_kmeans_qkv:
                        if "c_attn.weight" in name and E_kept.ndim == 2:
                            D3 = E_kept.shape[-1]
                            assert D3 % 3 == 0, "unexpected fused QKV shape"
                            D = D3 // 3
                            q, k, v = torch.split(E_kept, D, dim=-1)
                            q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
                            rq = _kmeans_deq(q, nbits=b_qkv, iters=int(args.excp_kmeans_iters), dev=dev)
                            rk = _kmeans_deq(k, nbits=b_qkv, iters=int(args.excp_kmeans_iters), dev=dev)
                            rv = _kmeans_deq(v, nbits=b_qkv, iters=int(args.excp_kmeans_iters), dev=dev)
                            rec_E = torch.cat([rq, rk, rv], dim=-1)
                        else:
                            rec_E = _kmeans_deq(E_kept, nbits=b_qkv, iters=int(args.excp_kmeans_iters), dev=dev)
                    else:
                        rec_E = quantize_pergroup_uniform(
                            E_kept, n_bits=b_qkv, group=args.excp_qkv_group, dim=-1, clip=args.excp_qkv_clip_pct
                        )
                else:
                    rec_E = _kmeans_deq(E_kept, nbits=int(args.excp_nbits), iters=int(args.excp_kmeans_iters), dev=dev)

                W_rec = (B + rec_E).to(W.dtype)
                if args.single_shot_apply:
                    p.data.copy_(W_rec)

                n = E.numel()
                total += n
                kk = int(keep_mask.sum().item())
                kept += kk
                err = (E - rec_E).float()
                mse = float((err * err).mean().item())
                var = float(E.float().var(unbiased=False).item())
                rel = 0.0 if var <= 0 else mse / var
                per_layer_err.append((rel, name))

                if dev.type == "cuda": torch.cuda.synchronize()
                tq1 = time.perf_counter()
                print(f"[ExCP OneShot][{name}] shape={tuple(E.shape)} nnz={int((E!=0).sum().item())} kept={kk} ({100.0*kk/max(1,E.numel()):.2f}%)")
                print(f"[ExCP OneShot][{name}] kmeans_time={tq1 - tq0:.3f}s")

        p_keep = kept / max(1, total)
        bpp = p_keep * args.excp_nbits + _H(p_keep)
        cr = 32.0 / max(bpp, 1e-9)

        nmse = 0.0
        vsum = 0.0
        for rel, _nm in per_layer_err:
            pass
        nmse = 0.0

        t1 = time.perf_counter()
        print(f"[ExCP OneShot] step={step} | keep={p_keep*100:.2f}% | nbits={args.excp_nbits} "
              f"| bpp={bpp:.3f} | CR≈{cr:.1f}x | {t1-t0:.2f}s")

        if args.single_shot_eval:
            nll1, ppl1 = lm_loss_on_split(
                ds["train"], tok, model,
                block_size=args.lm_block_size,
                batch_size=args.per_device_eval_batch_size,
                max_batches=args.max_eval_batches
            )
            print(f"[ExCP OneShot] TRAIN AFTER   loss={nll1:.4f}, ppl={ppl1:.2f}")
            print(f"[ExCP OneShot] Δtrain_loss={nll1-nll0:+.4f}, Δtrain_ppl={ppl1-ppl0:+.2f}")

        if args.print_top_layers and per_layer_err:
            per_layer_err.sort(reverse=True)
            kshow = min(args.print_top_layers, len(per_layer_err))
            print(f"[ExCP OneShot] Top-{kshow} layers by relative error:")
            for i in range(kshow):
                rel, nm = per_layer_err[i]
                print(f"  {i+1:2d}. rel_mse={rel:.4e}  |  {nm}")

        if args.single_shot_apply:
            out = os.path.join(args.output_dir, "oneshot_excp")
            os.makedirs(out, exist_ok=True)
            model.save_pretrained(out, safe_serialization=True)
            tok.save_pretrained(out)
            print(f"[ExCP OneShot] Compressed model saved to {out}")
        return

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_ds["train"],
        eval_dataset=lm_ds["validation"],
        data_collator=collator,
        callbacks=callbacks,
    )

    print("--- Starting Training (GPT-2 on WikiText-2) with ExCP compression ---")
    trainer.train()

    metrics_val = trainer.evaluate()
    eval_loss = metrics_val["eval_loss"]
    ppl_val = math.exp(eval_loss)
    print(f"[Trainer.evaluate] valid: eval_loss={eval_loss:.4f}, ppl={ppl_val:.2f}")
    model.eval()
    avg_nll_valid, ppl_valid = strided_perplexity(ds["validation"], tokenizer, model, stride=512)
    print(f"[Strided] valid: avg_nll={avg_nll_valid:.4f}, ppl={ppl_valid:.2f}")
    if "test" in ds:
        avg_nll_test, ppl_test = strided_perplexity(ds["test"], tokenizer, model, stride=512)
        print(f"[Strided]  test: avg_nll={avg_nll_test:.4f}, ppl={ppl_test:.2f}")

    out = os.path.join(args.output_dir, "best_model_final")
    trainer.save_model(out)
    tokenizer.save_pretrained(out)
    print(f"--- Best model saved to {out} ---")
    print(f"ExCP compressed artifacts under: {excp_dir}")

if __name__ == "__main__":
    main()

# (warpq) root@v100:~/test# python train_gpt2_wikitext2_excp_1.py   --single_shot --single_shot_residual --single_shot_apply --single_shot_eval   --single_shot_step 1200   --model_name runs/exp_ckpt_gpt2/checkpoint-1200   --tokenizer_name openai-community/gpt2   --baseline_path runs/exp_ckpt_gpt2/ckpts/baseline_step00000900_pre.pt   --optimizer_path runs/exp_ckpt_gpt2/c
# heckpoint-1200   --per_device_eval_batch_size 2   --lm_block_size 256 --max_eval_batches 1000   --excp_nbits 8 --excp_kmeans_qkv --excp_kmeans_iters 30   --excp_use_opt_threshold   --ex
# cp_groupwise_calibrate   --excp_group_keep_spec "mlp_proj=0.14,mlp_fc=0.09,attn_qkv=0.09,attn_out=0.07"   --excp_min_keep_floor 0.02   --excp_calibrate_keep 0.10   --output_dir runs/one
# shot_excp_step1200_from900
# `loss_type=None` was set in the config but it is unrecognised.Using the default loss: `ForCausalLMLoss`.
# [ExCP OneShot] TRAIN BEFORE  loss=2.9024, ppl=18.22
# [ExCP OneShot] Loaded baseline from: runs/exp_ckpt_gpt2/ckpts/baseline_step00000900_pre.pt
# [ExCP OneShot] QKV quantizer = k-means, qkv_bits=8, target_nz=0.1
# [ExCP OneShot] Loaded optimizer moments for 48/148 params (with v: 48). Source: runs/exp_ckpt_gpt2/checkpoint-1200/optimizer.pt
# [ExCP OneShot] Group-wise v-calibrated thresholds: {'attn_qkv': 0.000365, 'attn_out': 0.000384, 'mlp_fc': 0.000377, 'mlp_proj': 0.000266}
# [ExCP OneShot][transformer.h.0.attn.c_attn.weight] shape=(768, 2304) nnz=1769443 kept=161616 (9.13%)
# [ExCP OneShot][transformer.h.0.attn.c_attn.weight] kmeans_time=0.138s
# [ExCP OneShot][transformer.h.0.attn.c_proj.weight] shape=(768, 768) nnz=589823 kept=37914 (6.43%)
# [ExCP OneShot][transformer.h.0.attn.c_proj.weight] kmeans_time=0.028s
# [ExCP OneShot][transformer.h.0.mlp.c_fc.weight] shape=(768, 3072) nnz=2359246 kept=183636 (7.78%)
# [ExCP OneShot][transformer.h.0.mlp.c_fc.weight] kmeans_time=0.056s
# [ExCP OneShot][transformer.h.0.mlp.c_proj.weight] shape=(3072, 768) nnz=2359273 kept=47187 (2.00%)
# [ExCP OneShot][transformer.h.0.mlp.c_proj.weight] kmeans_time=0.032s
# [ExCP OneShot][transformer.h.1.attn.c_attn.weight] shape=(768, 2304) nnz=1769442 kept=139171 (7.87%)
# [ExCP OneShot][transformer.h.1.attn.c_attn.weight] kmeans_time=0.089s
# [ExCP OneShot][transformer.h.1.attn.c_proj.weight] shape=(768, 768) nnz=589819 kept=35154 (5.96%)
# [ExCP OneShot][transformer.h.1.attn.c_proj.weight] kmeans_time=0.028s
# [ExCP OneShot][transformer.h.1.mlp.c_fc.weight] shape=(768, 3072) nnz=2359260 kept=188913 (8.01%)
# [ExCP OneShot][transformer.h.1.mlp.c_fc.weight] kmeans_time=0.057s
# [ExCP OneShot][transformer.h.1.mlp.c_proj.weight] shape=(3072, 768) nnz=2359265 kept=47186 (2.00%)
# [ExCP OneShot][transformer.h.1.mlp.c_proj.weight] kmeans_time=0.032s
# [ExCP OneShot][transformer.h.2.attn.c_attn.weight] shape=(768, 2304) nnz=1769444 kept=136579 (7.72%)
# [ExCP OneShot][transformer.h.2.attn.c_attn.weight] kmeans_time=0.089s
# [ExCP OneShot][transformer.h.2.attn.c_proj.weight] shape=(768, 768) nnz=589820 kept=35449 (6.01%)
# [ExCP OneShot][transformer.h.2.attn.c_proj.weight] kmeans_time=0.027s
# [ExCP OneShot][transformer.h.2.mlp.c_fc.weight] shape=(768, 3072) nnz=2359263 kept=200383 (8.49%)
# [ExCP OneShot][transformer.h.2.mlp.c_fc.weight] kmeans_time=0.058s
# [ExCP OneShot][transformer.h.2.mlp.c_proj.weight] shape=(3072, 768) nnz=2359265 kept=47186 (2.00%)
# [ExCP OneShot][transformer.h.2.mlp.c_proj.weight] kmeans_time=0.031s
# [ExCP OneShot][transformer.h.3.attn.c_attn.weight] shape=(768, 2304) nnz=1769449 kept=151501 (8.56%)
# [ExCP OneShot][transformer.h.3.attn.c_attn.weight] kmeans_time=0.091s
# [ExCP OneShot][transformer.h.3.attn.c_proj.weight] shape=(768, 768) nnz=589817 kept=38823 (6.58%)
# [ExCP OneShot][transformer.h.3.attn.c_proj.weight] kmeans_time=0.028s
# [ExCP OneShot][transformer.h.3.mlp.c_fc.weight] shape=(768, 3072) nnz=2359258 kept=197985 (8.39%)
# [ExCP OneShot][transformer.h.3.mlp.c_fc.weight] kmeans_time=0.058s
# [ExCP OneShot][transformer.h.3.mlp.c_proj.weight] shape=(3072, 768) nnz=2359271 kept=47186 (2.00%)
# [ExCP OneShot][transformer.h.3.mlp.c_proj.weight] kmeans_time=0.031s
# [ExCP OneShot][transformer.h.4.attn.c_attn.weight] shape=(768, 2304) nnz=1769437 kept=156979 (8.87%)
# [ExCP OneShot][transformer.h.4.attn.c_attn.weight] kmeans_time=0.092s
# [ExCP OneShot][transformer.h.4.attn.c_proj.weight] shape=(768, 768) nnz=589816 kept=40219 (6.82%)
# [ExCP OneShot][transformer.h.4.attn.c_proj.weight] kmeans_time=0.028s
# [ExCP OneShot][transformer.h.4.mlp.c_fc.weight] shape=(768, 3072) nnz=2359259 kept=202161 (8.57%)
# [ExCP OneShot][transformer.h.4.mlp.c_fc.weight] kmeans_time=0.059s
# [ExCP OneShot][transformer.h.4.mlp.c_proj.weight] shape=(3072, 768) nnz=2359274 kept=428151 (18.15%)
# [ExCP OneShot][transformer.h.4.mlp.c_proj.weight] kmeans_time=0.099s
# [ExCP OneShot][transformer.h.5.attn.c_attn.weight] shape=(768, 2304) nnz=1769449 kept=155760 (8.80%)
# [ExCP OneShot][transformer.h.5.attn.c_attn.weight] kmeans_time=0.091s
# [ExCP OneShot][transformer.h.5.attn.c_proj.weight] shape=(768, 768) nnz=589816 kept=40002 (6.78%)
# [ExCP OneShot][transformer.h.5.attn.c_proj.weight] kmeans_time=0.028s
# [ExCP OneShot][transformer.h.5.mlp.c_fc.weight] shape=(768, 3072) nnz=2359262 kept=204847 (8.68%)
# [ExCP OneShot][transformer.h.5.mlp.c_fc.weight] kmeans_time=0.059s
# [ExCP OneShot][transformer.h.5.mlp.c_proj.weight] shape=(3072, 768) nnz=2359276 kept=429091 (18.19%)
# [ExCP OneShot][transformer.h.5.mlp.c_proj.weight] kmeans_time=0.098s
# [ExCP OneShot][transformer.h.6.attn.c_attn.weight] shape=(768, 2304) nnz=1769451 kept=163269 (9.23%)
# [ExCP OneShot][transformer.h.6.attn.c_attn.weight] kmeans_time=0.093s
# [ExCP OneShot][transformer.h.6.attn.c_proj.weight] shape=(768, 768) nnz=589815 kept=41497 (7.04%)
# [ExCP OneShot][transformer.h.6.attn.c_proj.weight] kmeans_time=0.029s
# [ExCP OneShot][transformer.h.6.mlp.c_fc.weight] shape=(768, 3072) nnz=2359276 kept=211558 (8.97%)
# [ExCP OneShot][transformer.h.6.mlp.c_fc.weight] kmeans_time=0.060s
# [ExCP OneShot][transformer.h.6.mlp.c_proj.weight] shape=(3072, 768) nnz=2359259 kept=454176 (19.25%)
# [ExCP OneShot][transformer.h.6.mlp.c_proj.weight] kmeans_time=0.103s
# [ExCP OneShot][transformer.h.7.attn.c_attn.weight] shape=(768, 2304) nnz=1769447 kept=167201 (9.45%)
# [ExCP OneShot][transformer.h.7.attn.c_attn.weight] kmeans_time=0.093s
# [ExCP OneShot][transformer.h.7.attn.c_proj.weight] shape=(768, 768) nnz=589816 kept=44866 (7.61%)
# [ExCP OneShot][transformer.h.7.attn.c_proj.weight] kmeans_time=0.029s
# [ExCP OneShot][transformer.h.7.mlp.c_fc.weight] shape=(768, 3072) nnz=2359266 kept=218345 (9.25%)
# [ExCP OneShot][transformer.h.7.mlp.c_fc.weight] kmeans_time=0.061s
# [ExCP OneShot][transformer.h.7.mlp.c_proj.weight] shape=(3072, 768) nnz=2359260 kept=469044 (19.88%)
# [ExCP OneShot][transformer.h.7.mlp.c_proj.weight] kmeans_time=0.106s
# [ExCP OneShot][transformer.h.8.attn.c_attn.weight] shape=(768, 2304) nnz=1769444 kept=167828 (9.48%)
# [ExCP OneShot][transformer.h.8.attn.c_attn.weight] kmeans_time=0.095s
# [ExCP OneShot][transformer.h.8.attn.c_proj.weight] shape=(768, 768) nnz=589813 kept=43690 (7.41%)
# [ExCP OneShot][transformer.h.8.attn.c_proj.weight] kmeans_time=0.029s
# [ExCP OneShot][transformer.h.8.mlp.c_fc.weight] shape=(768, 3072) nnz=2359261 kept=220292 (9.34%)
# [ExCP OneShot][transformer.h.8.mlp.c_fc.weight] kmeans_time=0.062s
# [ExCP OneShot][transformer.h.8.mlp.c_proj.weight] shape=(3072, 768) nnz=2359253 kept=489302 (20.74%)
# [ExCP OneShot][transformer.h.8.mlp.c_proj.weight] kmeans_time=0.109s
# [ExCP OneShot][transformer.h.9.attn.c_attn.weight] shape=(768, 2304) nnz=1769445 kept=165082 (9.33%)
# [ExCP OneShot][transformer.h.9.attn.c_attn.weight] kmeans_time=0.094s
# [ExCP OneShot][transformer.h.9.attn.c_proj.weight] shape=(768, 768) nnz=589818 kept=45956 (7.79%)
# [ExCP OneShot][transformer.h.9.attn.c_proj.weight] kmeans_time=0.030s
# [ExCP OneShot][transformer.h.9.mlp.c_fc.weight] shape=(768, 3072) nnz=2359263 kept=230619 (9.77%)
# [ExCP OneShot][transformer.h.9.mlp.c_fc.weight] kmeans_time=0.063s
# [ExCP OneShot][transformer.h.9.mlp.c_proj.weight] shape=(3072, 768) nnz=2359253 kept=522770 (22.16%)
# [ExCP OneShot][transformer.h.9.mlp.c_proj.weight] kmeans_time=0.115s
# [ExCP OneShot][transformer.h.10.attn.c_attn.weight] shape=(768, 2304) nnz=1769444 kept=168855 (9.54%)
# [ExCP OneShot][transformer.h.10.attn.c_attn.weight] kmeans_time=0.095s
# [ExCP OneShot][transformer.h.10.attn.c_proj.weight] shape=(768, 768) nnz=589817 kept=46372 (7.86%)
# [ExCP OneShot][transformer.h.10.attn.c_proj.weight] kmeans_time=0.030s
# [ExCP OneShot][transformer.h.10.mlp.c_fc.weight] shape=(768, 3072) nnz=2359267 kept=237114 (10.05%)
# [ExCP OneShot][transformer.h.10.mlp.c_fc.weight] kmeans_time=0.064s
# [ExCP OneShot][transformer.h.10.mlp.c_proj.weight] shape=(3072, 768) nnz=2359253 kept=568564 (24.10%)
# [ExCP OneShot][transformer.h.10.mlp.c_proj.weight] kmeans_time=0.124s
# [ExCP OneShot][transformer.h.11.attn.c_attn.weight] shape=(768, 2304) nnz=1769445 kept=177055 (10.01%)
# [ExCP OneShot][transformer.h.11.attn.c_attn.weight] kmeans_time=0.097s
# [ExCP OneShot][transformer.h.11.attn.c_proj.weight] shape=(768, 768) nnz=589816 kept=45508 (7.72%)
# [ExCP OneShot][transformer.h.11.attn.c_proj.weight] kmeans_time=0.030s
# [ExCP OneShot][transformer.h.11.mlp.c_fc.weight] shape=(768, 3072) nnz=2359268 kept=251164 (10.65%)
# [ExCP OneShot][transformer.h.11.mlp.c_fc.weight] kmeans_time=0.067s
# [ExCP OneShot][transformer.h.11.mlp.c_proj.weight] shape=(3072, 768) nnz=2359230 kept=601463 (25.49%)
# [ExCP OneShot][transformer.h.11.mlp.c_proj.weight] kmeans_time=0.130s
# [ExCP OneShot] step=1200 | keep=10.72% | nbits=8 | bpp=1.349 | CR≈23.7x | 3.45s
# [ExCP OneShot] TRAIN AFTER   loss=2.9173, ppl=18.49
# [ExCP OneShot] Δtrain_loss=+0.0149, Δtrain_ppl=+0.27
# [ExCP OneShot] Top-12 layers by relative error:
#    1. rel_mse=4.0211e+00  |  transformer.h.11.mlp.c_proj.weight
#    2. rel_mse=3.7224e+00  |  transformer.h.10.mlp.c_proj.weight
#    3. rel_mse=3.5441e+00  |  transformer.h.9.mlp.c_proj.weight
#    4. rel_mse=3.4685e+00  |  transformer.h.8.mlp.c_proj.weight
#    5. rel_mse=3.3675e+00  |  transformer.h.7.mlp.c_proj.weight
#    6. rel_mse=3.3197e+00  |  transformer.h.6.mlp.c_proj.weight
#    7. rel_mse=3.2497e+00  |  transformer.h.4.mlp.c_proj.weight
#    8. rel_mse=3.2488e+00  |  transformer.h.5.mlp.c_proj.weight
#    9. rel_mse=2.4143e+00  |  transformer.h.11.mlp.c_fc.weight
#   10. rel_mse=2.3824e+00  |  transformer.h.10.mlp.c_fc.weight
#   11. rel_mse=2.3595e+00  |  transformer.h.11.attn.c_proj.weight
#   12. rel_mse=2.3541e+00  |  transformer.h.11.attn.c_attn.weight
# [ExCP OneShot] Compressed model saved to runs/oneshot_excp_step1200_from900/oneshot_excp
# (warpq) root@v100:~/test# 