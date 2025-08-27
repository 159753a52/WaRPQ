import os, sys, math, json, argparse, random
from typing import Optional, List
from dataclasses import dataclass

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import pywt
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup, AutoConfig

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def bytes_of(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()

def snr_db(x: torch.Tensor, x_hat: torch.Tensor, eps=1e-12) -> float:
    s = torch.mean(x.float()**2).item()
    n = torch.mean((x.float() - x_hat.float())**2).item()
    return 10.0 * math.log10((s+eps)/(n+eps))

@dataclass
class CompressCfg:
    wavelet: str = "haar"
    level: int = 1
    prune_ratio: float = 0.2
    nbits: int = 4
    quant: str = "uniform"
    separate_zero: bool = True
    process_optimizer: bool = True
    target_modules: Optional[List[str]] = None

def _flatten(arrs): return np.concatenate([a.ravel() for a in arrs])
def _split_by_shapes(vec, shapes):
    parts, i = [], 0
    for shp in shapes:
        n = int(np.prod(shp)); parts.append(vec[i:i+n].reshape(shp)); i += n
    return parts

def dwt_flatten(x: np.ndarray, wavelet="haar", level=1):
    coeffs = pywt.wavedec(x, wavelet=wavelet, level=level, mode="periodization")
    shapes = [c.shape for c in coeffs]
    flat = _flatten(coeffs)
    return flat, (wavelet, level, shapes)

def idwt_from_flat(flat: np.ndarray, ctx):
    wavelet, level, shapes = ctx
    parts = _split_by_shapes(flat, shapes)
    x_hat = pywt.waverec(parts, wavelet=wavelet, mode="periodization")
    return x_hat

def magnitude_prune(vec: np.ndarray, prune_ratio: float):
    if prune_ratio <= 0: return vec
    thr = np.quantile(np.abs(vec), 1.0 - prune_ratio)
    out = vec.copy(); out[np.abs(out) < thr] = 0.0
    return out

def quant_uniform(vec: np.ndarray, nbits: int):
    if nbits >= 16:
        return vec.copy(), {"meta":{"scheme":"identity"}, "bits": vec.size*32}
    qmax = (1 << (nbits-1)) - 1
    scale = (np.max(np.abs(vec))+1e-12) / qmax
    q = np.round(vec/scale).astype(np.int32)
    q = np.clip(q, -qmax-1, qmax)
    dq = q.astype(np.float32) * scale
    return dq, {"meta":{"scheme":"uniform_sym","scale":float(scale),"nbits":nbits}, "bits": vec.size*nbits}

def kmeans_lite(vec: np.ndarray, nbits: int, iters=8, separate_zero=True):
    K = (1 << nbits)
    x = vec.reshape(-1,1).astype(np.float32)
    if separate_zero:
        mask = np.abs(x[:,0]) > 0
        if not mask.any():
            return vec.copy(), {"meta":{"scheme":"km_sep0_allzero"},"bits":0}
        xw = x[mask]
        qs = np.linspace(0,1,K-1,endpoint=False)[1:]
        centers = np.quantile(xw, qs).reshape(-1,1)
        for _ in range(iters):
            d = (xw - centers.T)**2
            idx = d.argmin(axis=1)
            for k in range(centers.shape[0]):
                pts = xw[idx==k]
                if pts.size: centers[k,0] = pts.mean()
        d_all = (x - centers.T)**2
        idx_all = d_all.argmin(axis=1)
        x_hat = centers[idx_all,0]
        x_hat[~mask] = 0.0
        bits = int(math.ceil(math.log2(K))) * x.size
        return x_hat.astype(np.float32).reshape(vec.shape), {"meta":{"scheme":"km_sep0","centers":centers.squeeze().tolist()}, "bits":bits}
    else:
        qs = np.linspace(0,1,K,endpoint=False)[1:]
        centers = np.quantile(x, qs).reshape(-1,1)
        for _ in range(iters):
            d = (x - centers.T)**2
            idx = d.argmin(axis=1)
            for k in range(centers.shape[0]):
                pts = x[idx==k]
                if pts.size: centers[k,0] = pts.mean()
        idx = ((x - centers.T)**2).argmin(axis=1)
        x_hat = centers[idx,0]
        bits = int(math.ceil(math.log2(K))) * x.size
        return x_hat.astype(np.float32).reshape(vec.shape), {"meta":{"scheme":"km","centers":centers.squeeze().tolist()}, "bits":bits}

def dwt_prune_quant_restore(t: torch.Tensor, cfg: CompressCfg):
    device, dtype = t.device, t.dtype
    x = t.detach().float().cpu().numpy().ravel()
    flat, ctx = dwt_flatten(x, cfg.wavelet, cfg.level)
    flat = magnitude_prune(flat, cfg.prune_ratio)
    if cfg.quant == "uniform":
        dq, qinfo = quant_uniform(flat, cfg.nbits)
    else:
        dq, qinfo = kmeans_lite(flat, cfg.nbits, separate_zero=cfg.separate_zero)
    x_hat = idwt_from_flat(dq, ctx)[:x.size].reshape(t.shape)
    x_hat_t = torch.from_numpy(x_hat).to(device=device, dtype=dtype)
    orig_bytes = bytes_of(t)
    comp_bytes = max(1, qinfo.get("bits", t.numel()*cfg.nbits)//8)
    cr = orig_bytes / comp_bytes
    snr = snr_db(t, x_hat_t)
    return x_hat_t, {"cr":cr, "snr_db":snr, "orig_bytes":orig_bytes, "comp_bytes":comp_bytes, "q":qinfo.get("meta",{})}

def should_process(name: str, targets: Optional[List[str]]):
    if not targets: return True
    return any(k in name for k in targets)

def periodic_compress_and_restore(model, optimizer, cfg: CompressCfg, global_step: int, log_path: str):
    model.eval()
    metrics = {"step": int(global_step), "items":[]}
    with torch.no_grad():
        for n, p in model.named_parameters():
            if not p.requires_grad or p.is_sparse: continue
            if not should_process(n, cfg.target_modules): continue
            rec, m = dwt_prune_quant_restore(p.data, cfg)
            p.data.copy_(rec)
            metrics["items"].append({"type":"weight","name":n, **m})
        if cfg.process_optimizer:
            for n, p in model.named_parameters():
                st = optimizer.state.get(p, None)
                if st is None: continue
                for key in ("exp_avg","exp_avg_sq"):
                    if key in st and isinstance(st[key], torch.Tensor) and should_process(n, cfg.target_modules):
                        rec, m = dwt_prune_quant_restore(st[key].data, cfg)
                        st[key].data.copy_(rec)
                        metrics["items"].append({"type":key,"name":n, **m})
    if metrics["items"]:
        cr = float(np.mean([it["cr"] for it in metrics["items"]]))
        snr = float(np.mean([it["snr_db"] for it in metrics["items"]]))
        metrics["avg_cr"], metrics["avg_snr_db"] = cr, snr
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
    model.train()

def build_dataset(args, tokenizer):
    if args.hf_dataset:
        if args.hf_config:
            ds = load_dataset(args.hf_dataset, args.hf_config, split="train")
        else:
            ds = load_dataset(args.hf_dataset, split="train")
        text_col = args.text_column or "text"
    else:
        if not args.data_files:
            raise ValueError("Provide --hf_dataset or --data_files")
        if args.data_files.endswith(".json") or args.data_files.endswith(".jsonl"):
            ds = load_dataset("json", data_files=args.data_files, split="train")
            text_col = args.text_column or "text"
        else:
            ds = load_dataset("text", data_files=args.data_files, split="train")
            text_col = "text"
    def tok(batch):
        return tokenizer(batch[text_col], return_attention_mask=False, add_special_tokens=False)
    keep_cols = [c for c in ds.column_names if c != text_col]
    ds_tok = ds.map(tok, batched=True, remove_columns=keep_cols or None)
    block = args.block_size
    def group(ex):
        concat = []
        for ids in ex["input_ids"]:
            concat.extend(ids)
        total = (len(concat)//block)*block
        chunks = [concat[i:i+block] for i in range(0, total, block)]
        return {"input_ids": chunks, "labels": [list(c) for c in chunks]}
    ds_group = ds_tok.map(group, batched=True, remove_columns=ds_tok.column_names)
    ds_group.set_format(type="torch", columns=["input_ids","labels"])
    return ds_group

class Collator:
    def __init__(self, pad): self.pad = pad
    def __call__(self, batch):
        input_ids = torch.stack([b["input_ids"] for b in batch])
        labels = torch.stack([b["labels"] for b in batch])
        attn = (input_ids != self.pad).long()
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="EleutherAI/pythia-70m")
    ap.add_argument("--hf_dataset", type=str, default="Skylion007/openwebtext")
    ap.add_argument("--data_files", type=str, default="")
    ap.add_argument("--text_column", type=str, default="")
    ap.add_argument("--out", type=str, default="out_pythia70m_dwtpq")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--block_size", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--warmup_steps", type=int, default=400)
    ap.add_argument("--max_steps", type=int, default=3000)
    ap.add_argument("--save_steps", type=int, default=1000)
    ap.add_argument("--restore_period", type=int, default=1000)
    ap.add_argument("--log_interval", type=int, default=50)
    ap.add_argument("--wavelet", type=str, default="haar")
    ap.add_argument("--dwt_level", type=int, default=1)
    ap.add_argument("--prune_ratio", type=float, default=0.2)
    ap.add_argument("--nbits", type=int, default=4)
    ap.add_argument("--quant", type=str, default="uniform", choices=["uniform","kmeans"])
    ap.add_argument("--compress_moments", action="store_true")
    ap.add_argument("--targets", type=str, default="")
    ap.add_argument("--hf_config", type=str, default="")
    ap.add_argument("--amp", type=str, default="none", choices=["none","fp16"])
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    config = AutoConfig.from_pretrained(args.model)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            config=config,
            torch_dtype=torch.float32,
            attn_implementation="sdpa",
        ).to(device)
    except ValueError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            config=config,
            torch_dtype=torch.float32,
            attn_implementation="eager",
        ).to(device)

    ds = build_dataset(args, tok)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=Collator(tok.pad_token_id))

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    sched = get_linear_schedule_with_warmup(optim, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps)
    use_amp = (args.amp == "fp16")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    os.makedirs(args.out, exist_ok=True)
    log_jsonl = os.path.join(args.out, "compress_log.jsonl")

    comp_cfg = CompressCfg(wavelet=args.wavelet, level=args.dwt_level, prune_ratio=args.prune_ratio,
                           nbits=args.nbits, quant=args.quant, separate_zero=True,
                           process_optimizer=args.compress_moments,
                           target_modules=[s for s in args.targets.split(",") if s] if args.targets else None)

    model.train()
    global_step = 0
    ce_ema = None
    pbar = tqdm(total=args.max_steps, dynamic_ncols=True, desc="train")

    while global_step < args.max_steps:
        for batch in dl:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].to(device)

            if use_amp:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    out = model(**inputs, use_cache=False)
                    logits = out.logits
            else:
                out = model(**inputs, use_cache=False)
                logits = out.logits

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            with torch.cuda.amp.autocast(enabled=False):
                ce = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)).float(),
                    shift_labels.view(-1),
                    ignore_index=-100
                )
                loss = ce / args.grad_accum

            if not torch.isfinite(ce):
                raise RuntimeError(f"Non-finite CE at step {global_step}")

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (global_step + 1) % args.grad_accum == 0:
                if use_amp:
                    scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if use_amp:
                    scaler.step(optim); scaler.update()
                else:
                    optim.step()
                optim.zero_grad(set_to_none=True)
                sched.step()

            if ce_ema is None:
                ce_ema = ce.item()
            else:
                ce_ema = 0.9 * ce_ema + 0.1 * ce.item()
            if global_step % args.log_interval == 0:
                ppl = math.exp(ce_ema)
                print(f"[step {global_step}] CE(avg-token)={ce.item():.4f} | CE(EMA)={ce_ema:.4f} | PPL≈{ppl:.1f}")

            if (global_step > 0) and (global_step % args.save_steps == 0):
                ck = os.path.join(args.out, f"ckpt_{global_step}")
                os.makedirs(ck, exist_ok=True)
                model.save_pretrained(ck); tok.save_pretrained(ck)
                torch.save({"optimizer": optim.state_dict(), "scheduler": sched.state_dict(),
                            "scaler": scaler.state_dict(), "step": global_step},
                           os.path.join(ck, "trainer_state.pt"))

            if (global_step > 0) and (global_step % args.restore_period == 0):
                print(f"[step {global_step}] periodic DWT-prune-{args.nbits}b & restore")
                periodic_compress_and_restore(model, optim, comp_cfg, global_step, log_jsonl)

            global_step += 1
            pbar.update(1)
            if global_step >= args.max_steps: break

    model.save_pretrained(args.out); tok.save_pretrained(args.out)
    torch.save({"optimizer": optim.state_dict(), "scheduler": sched.state_dict(),
                "scaler": scaler.state_dict(), "step": global_step},
               os.path.join(args.out, "trainer_state.pt"))
    print("Done. Artifacts in:", args.out)

if __name__ == "__main__":
    main()