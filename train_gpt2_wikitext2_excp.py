# train_gpt2_wikitext2_excp.py
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

def sizeof_fmt(num: float, suffix: str = "B") -> str:
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Y{suffix}"

def robust_quantile(x: torch.Tensor, q: float, sample_cap: int = 2_000_000) -> torch.Tensor:
    """
    返回张量 x 的 q 分位数阈值，内存/规模安全：
    - 小张量：用 kthvalue 精确计算（不做全排序）
    - 超大张量：随机抽样到 sample_cap 后再用 kthvalue（近似，但足够稳）
    """
    xf = x.reshape(-1)
    n = xf.numel()
    if n == 0:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)
    if n <= sample_cap:
        k = max(1, int(math.ceil(q * n)))
        return torch.kthvalue(xf, k).values
    # 抽样（有放回），避免构造巨大乱序索引
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

def kmeans_nonuniform_quantize(x: torch.Tensor, nbits: int, max_iters: int = 12, sample: int = 1_000_000) -> QuantizedTensor:
    assert nbits >= 2, "nbits must be >= 2"
    flat = x.view(-1)
    device = flat.device
    nz_mask = flat != 0
    nz = flat[nz_mask]
    K = (1 << nbits) - 1
    if nz.numel() == 0:
        centers = torch.zeros(K, device=device, dtype=torch.float32)
        idx = torch.zeros_like(flat, dtype=torch.uint8)
        return QuantizedTensor(tuple(x.shape), centers, _pack_nibbles(idx), nbits)
    if nz.numel() > sample:
        sel = torch.randperm(nz.numel(), device=device)[:sample]
        samp = nz[sel].clone()
    else:
        samp = nz.clone()
    lo = torch.quantile(samp, 0.005)
    hi = torch.quantile(samp, 0.995)
    centers = torch.linspace(lo, hi, K, device=device, dtype=torch.float32)
    for _ in range(max_iters):
        d = (samp.unsqueeze(1) - centers.unsqueeze(0)).abs()
        a = torch.argmin(d, dim=1)
        for k in range(K):
            m = a == k
            if m.any():
                centers[k] = samp[m].mean()
    idx = torch.zeros_like(flat, dtype=torch.uint8)
    if nz.numel() > 0:
        CH = 2_000_000
        nz_idx = torch.where(nz_mask)[0]
        for i in range(0, nz_idx.numel(), CH):
            sel = nz_idx[i : i + CH]
            vals = flat[sel]
            d = (vals.unsqueeze(1) - centers.unsqueeze(0)).abs()
            a = torch.argmin(d, dim=1)
            idx[sel] = (a + 1).to(torch.uint8)
    qt = QuantizedTensor(tuple(x.shape), centers=centers, packed_idx=_pack_nibbles(idx), nbits=nbits)
    return qt

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
    momentum_policy: str = "none"   # ['none','reset_changed']

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

            # —— GPU 上量化 —— #
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

            # 动量策略：把变化较大的位置的动量清零
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

def main():
    ap = argparse.ArgumentParser()
    # training
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
    # 固定步数 + 禁用早停
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--disable_early_stop", action="store_true", default=True)

    # ExCP
    ap.add_argument("--excp_save_dir", type=str, default=None)
    ap.add_argument("--excp_start_step", type=int, default=400)
    ap.add_argument("--excp_every", type=int, default=1000)
    ap.add_argument("--excp_nbits", type=int, default=4)
    ap.add_argument("--excp_alpha", type=float, default=5e-5)
    ap.add_argument("--excp_beta", type=float, default=2.0)
    ap.add_argument("--excp_keep_init", action="store_true", default=True)

    # Reconstruction utility
    ap.add_argument("--reconstruct_only", action="store_true", default=False)
    ap.add_argument("--reconstruct_upto", type=int, default=None)
    ap.add_argument("--reconstruct_out", type=str, default=None)

    # 与 WaRP-Q 对齐的控制项
    ap.add_argument("--excp_apply_to_model", action="store_true", default=False)
    ap.add_argument("--excp_momentum_policy", type=str, default="none", choices=["none", "reset_changed"])
    ap.add_argument("--excp_target_nz", type=float, default=0.10)
    ap.add_argument("--excp_enforce_target", action="store_true", default=True)

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
        max_steps=args.max_steps,  # 固定总步数
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
