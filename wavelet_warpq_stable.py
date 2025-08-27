import os
import re
import io
import lzma
import json
import math
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, TrainerCallback, TrainerState, TrainerControl
import evaluate
from pytorch_wavelets import DWT1D, IDWT1D

def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def mu_law_compand(x: torch.Tensor, mu: float, s: torch.Tensor):
    x = torch.clamp(x, min=-s, max=s)
    y = torch.sign(x) * torch.log1p(mu * torch.abs(x) / (s + 1e-12)) / math.log(1.0 + mu)
    return torch.clamp(y, -1.0, 1.0)

def mu_law_expand(y: torch.Tensor, mu: float, s: torch.Tensor):
    x = torch.sign(y) * (s * (torch.pow(1.0 + mu, torch.abs(y)) - 1.0) / mu)
    return x

def quantize_uniform_symmetric(y: torch.Tensor, n_bits: int):
    if n_bits <= 1:
        return torch.zeros_like(y), 0.0
    qmax = 2 ** (n_bits - 1) - 1
    q = torch.clamp(torch.round(y * qmax), min=-qmax, max=qmax)
    return q, qmax

def dequantize_uniform_symmetric(q: torch.Tensor, qmax: float):
    if qmax == 0:
        return torch.zeros_like(q)
    return q / qmax

def packbits_int16(arr: np.ndarray):
    bio = io.BytesIO()
    np.save(bio, arr.astype(np.int16), allow_pickle=False)
    return bio.getvalue()

def packfloats(arr: np.ndarray):
    bio = io.BytesIO()
    np.save(bio, arr.astype(np.float32), allow_pickle=False)
    return bio.getvalue()

def lzma_compress_bytes(b: bytes, level: int = 9):
    return lzma.compress(b, preset=level | lzma.PRESET_EXTREME)

def bit_allocation_rd(energies: List[float], sens: List[float], counts: List[int], avg_bits: float, choices: List[int]):
    J = len(energies)
    Ntot = sum(counts)
    budget = int(round(avg_bits * Ntot))
    b = [min(choices) for _ in range(J)]
    used = sum(c * bb for c, bb in zip(counts, b))
    def rd_gain(j, bj):
        Cj = max(energies[j], 1e-12) * max(sens[j], 1e-6)
        return Cj * (2.0 ** (-2.0 * (bj + 1)) - 2.0 ** (-2.0 * bj))
    while used + 1 <= budget:
        best = None
        for j in range(J):
            if b[j] >= max(choices):
                continue
            gain = rd_gain(j, b[j]) * counts[j]
            if (best is None) or (gain < best[0]):
                best = (gain, j)
        if best is None:
            break
        j = best[1]
        b[j] += 1
        used += counts[j]
    return b

def bit_allocation_rd_lagrange(energies: List[float], sens: List[float], counts: List[int], avg_bits: float, choices: List[int]):
    J = len(energies)
    Ntot = sum(counts)
    budget = avg_bits * Ntot
    C = [max(energies[j], 1e-12) * max(sens[j], 1e-6) for j in range(J)]
    bmin, bmax = min(choices), max(choices)
    lam_lo, lam_hi = 1e-9, 1e9
    def alloc(lam):
        b = []
        used = 0.0
        for j in range(J):
            if C[j] <= 0:
                bj = bmin
            else:
                bj = 0.5 * math.log2(C[j] / lam)
                bj = int(max(bmin, min(bmax, math.floor(bj))))
            b.append(bj)
            used += bj * counts[j]
        return b, used
    for _ in range(50):
        lam = math.sqrt(lam_lo * lam_hi)
        b, used = alloc(lam)
        if used > budget:
            lam_lo = lam
        else:
            lam_hi = lam
    b, _ = alloc(lam_hi)
    return b

def blockify_flat(x: torch.Tensor, block_size: int):
    xflat = x.reshape(-1)
    pad = (block_size - (xflat.numel() % block_size)) % block_size
    if pad > 0:
        xflat = F.pad(xflat.unsqueeze(0), (0, pad), mode="constant", value=0).squeeze(0)
    blocks = xflat.view(-1, block_size)
    return blocks, pad

def unblockify_flat(blocks: torch.Tensor, orig_numel: int):
    xflat = blocks.reshape(-1)
    if xflat.numel() > orig_numel:
        xflat = xflat[:orig_numel]
    return xflat

def dwt1d_blocks(blocks: torch.Tensor, dwt_levels: int):
    dwt = DWT1D(wave="haar", J=dwt_levels, mode="symmetric").to(blocks.device)
    idwt = IDWT1D(wave="haar", mode="symmetric").to(blocks.device)
    cA, cD_list = dwt(blocks.unsqueeze(1))
    coeffs = [cA.squeeze(1)] + [d.squeeze(1) for d in cD_list]
    return coeffs, idwt

def idwt1d_blocks(coeffs: List[torch.Tensor], idwt):
    rec = idwt((coeffs[0].unsqueeze(1), [c.unsqueeze(1) for c in coeffs[1:]])).squeeze(1)
    return rec

def soft_threshold(x: torch.Tensor, tau: torch.Tensor):
    return torch.sign(x) * torch.clamp(torch.abs(x) - tau, min=0.0)

def median_abs(x: torch.Tensor, dim: int = -1, keepdim: bool = True):
    med = x.abs().median(dim=dim, keepdim=keepdim).values
    return med

def percentile_abs(x: torch.Tensor, q: float, dim: int = -1, keepdim: bool = True):
    k = max(1, int(round(q * x.shape[dim])))
    vals, _ = torch.topk(x.abs(), k, dim=dim, largest=True, sorted=True)
    thr = vals.narrow(dim, k - 1, 1)
    return thr

def quantize_subband_mulaw(x: torch.Tensor, b: int, clip_pct: float, mu: float):
    if x.numel() == 0:
        return x.clone(), torch.tensor(0.0, device=x.device), torch.tensor(1.0, device=x.device), torch.zeros_like(x)
    vmax = torch.quantile(torch.abs(x).reshape(x.shape[0], -1), q=clip_pct, dim=-1, keepdim=True)
    vmax = torch.clamp(vmax, min=1e-8)
    y = mu_law_compand(x, mu, vmax)
    q, qmax = quantize_uniform_symmetric(y, b)
    deq = dequantize_uniform_symmetric(q, qmax)
    xrec = mu_law_expand(deq, mu, vmax)
    return xrec, vmax.squeeze(-1), torch.tensor(mu, device=x.device), q

def serialize_param_payload(name: str, subband_bits: List[int], vmax_list: List[torch.Tensor], mu_list: List[torch.Tensor], q_list: List[torch.Tensor], shapes: Dict[str, Any], extra_meta: Dict[str, Any]):
    arrs = {}
    for idx, q in enumerate(q_list):
        arrs[f"q_{idx}"] = q.detach().cpu().to(torch.int16).numpy()
    for idx, v in enumerate(vmax_list):
        arrs[f"vmax_{idx}"] = v.detach().cpu().to(torch.float32).numpy()
    mus = np.array([float(mu_list[idx].item() if isinstance(mu_list[idx], torch.Tensor) else mu_list[idx]) for idx in range(len(mu_list))], dtype=np.float32)
    bits = np.array(subband_bits, dtype=np.int16)
    meta = {
        "name": name,
        "bits": bits.tolist(),
        "mus": mus.tolist(),
        "shapes": shapes,
        "extra": extra_meta
    }
    bio = io.BytesIO()
    np.savez(bio, **arrs, bits=bits, mus=mus, meta=np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8))
    raw = bio.getvalue()
    comp = lzma_compress_bytes(raw, level=9)
    return comp, len(raw), len(comp), meta

def quantize_pergroup_uniform(tensor: torch.Tensor, n_bits: int, group_size: int, dim: int = -1, clip_pct: float = 0.995):
    if n_bits <= 0 or tensor.ndim == 0 or tensor.numel() == 0:
        return tensor.clone()
    dtype = tensor.dtype
    dim = dim if dim >= 0 else tensor.ndim + dim
    perm = list(range(tensor.ndim))
    perm[dim], perm[-1] = perm[-1], perm[dim]
    x = tensor.permute(*perm).contiguous()
    shape = x.shape
    L = shape[-1]
    x = x.reshape(-1, L)
    pad = (group_size - (L % group_size)) % group_size
    if pad > 0:
        pad_tensor = torch.zeros(x.shape[0], pad, device=x.device, dtype=x.dtype)
        x = torch.cat([x, pad_tensor], dim=1)
    G = x.shape[1] // group_size
    x = x.view(-1, G, group_size)
    absx = x.abs()
    vmax = torch.quantile(absx, q=clip_pct, dim=-1, keepdim=True)
    vmax = torch.clamp(vmax, min=1e-8)
    qmax = 2 ** (n_bits - 1) - 1
    q = torch.clamp(torch.round(x / (vmax / qmax)), min=-qmax, max=qmax)
    deq = q * (vmax / qmax)
    deq = deq.view(-1, G * group_size)
    if pad > 0:
        deq = deq[:, :L]
    deq = deq.view(*shape[:-1], L)
    inv_perm = list(range(tensor.ndim))
    inv_perm[dim], inv_perm[-1] = inv_perm[-1], inv_perm[dim]
    return deq.permute(*inv_perm).contiguous().to(dtype=dtype)

def _binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p))

def _safe_div(a: float, b: float, eps: float = 1e-12) -> float:
    return float(a / (b + eps))

class JsonLoggingCallback(TrainerCallback):
    def __init__(self, json_log_path):
        self.json_log_path = json_log_path
        self.log_history = []
    def on_train_begin(self, args, state, control, **kwargs):
        if not os.path.exists(os.path.dirname(self.json_log_path)):
            os.makedirs(os.path.dirname(self.json_log_path), exist_ok=True)
        with open(self.json_log_path, "w") as f:
            json.dump([], f)
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            log_entry = {**logs, "step": state.global_step}
            self.log_history.append(log_entry)
            with open(self.json_log_path, "w") as f:
                json.dump(self.log_history, f, indent=2)

class SubbandPredictor:
    def __init__(self, mode: str = "ar1", beta: float = 0.9):
        self.mode = mode
        self.beta = float(beta)
        self.state: Dict[Tuple[str, int], Dict[str, Any]] = {}
    def _get_state(self, key):
        if key not in self.state:
            self.state[key] = {"mu": 0.0, "s0": 1e-6, "s1": 0.0, "last": None}
        return self.state[key]
    def update_and_predict(self, key: Tuple[str, int], c_rows: torch.Tensor, is_i_frame: bool):
        st = self._get_state(key)
        mu, s0, s1, last = st["mu"], st["s0"], st["s1"], st["last"]
        cur = c_rows.mean(dim=1, keepdim=True)
        if last is None:
            pred = cur.clone()
        else:
            if self.mode == "ema":
                pred = self.beta * last + (1.0 - self.beta) * torch.tensor(mu, device=cur.device, dtype=cur.dtype)
            else:
                rho = s1 / max(s0, 1e-6)
                pred = rho * (last - torch.tensor(mu, device=cur.device, dtype=cur.dtype)) + torch.tensor(mu, device=cur.device, dtype=cur.dtype)
        diff = (cur - mu)
        mu = 0.99 * mu + 0.01 * cur.mean().item()
        s0 = 0.99 * s0 + 0.01 * (diff**2).mean().item()
        if last is not None:
            s1 = 0.99 * s1 + 0.01 * (diff * (last - mu)).mean().item()
        st["mu"], st["s0"], st["s1"], st["last"] = mu, s0, s1, cur.detach()
        pred_full = pred.repeat(1, c_rows.shape[1])
        var_c = float(c_rows.float().var(unbiased=False).item()) if c_rows.numel() > 0 else 0.0
        var_d = float((c_rows.float() - pred_full.float()).var(unbiased=False).item()) if c_rows.numel() > 0 else 0.0
        PI = 1.0 - _safe_div(var_d, max(var_c, 1e-12))
        return pred_full, PI

class WaRPQCallback(TrainerCallback):
    def __init__(
        self,
        method: str,
        avg_bitwidth: float,
        quant_freq: int,
        block_size: int = 256,
        dwt_levels: int = 3,
        alpha_w2m: float = 5e-5,
        beta_m2w: float = 2.0,
        clip_pct: float = 0.995,
        mu_law_mu: float = 255.0,
        momentum_bitwidth: int = 8,
        momentum_clip_pct: float = 0.999,
        momentum_mask_min: float = 0.5,
        rd_bit_choices: str = "2,3,4,5",
        sparsity_warmup_steps: int = 1000,
        skip_layernorm_and_bias: bool = True,
        qkv_special_handling: bool = True,
        qkv_group_size: int = 64,
        qkv_clip_pct: float = 0.995,
        quant_start_step: int = 200,
        apply_to_model: bool = False,
        gop_size: int = 4,
        i_frame_scale: float = 1.0,
        p_frame_scale: float = 0.8,
        save_dir: str = "./compressed_ckpts",
        save_payload: bool = True,
        target_nz_ratio: float = 0.1,
        enforce_target_nz: bool = True,
        m2w_gate: bool = True,
        m2w_gate_q: float = 0.9,
        ema_decay: float = 0.99,
        use_predictor: bool = True,
        predictor_mode: str = "ar1",
        predictor_beta: float = 0.9,
        pi_disable_threshold: float = 0.1,
        use_lagrange_alloc: bool = True,
        min_apply_snr_db: float = 8.0,
        enable_final_topk: bool = True
    ):
        self.method = method
        self.avg_bitwidth = avg_bitwidth
        self.quant_freq = quant_freq
        self.block_size = block_size
        self.dwt_levels = dwt_levels
        self.alpha_w2m = alpha_w2m
        self.beta_m2w = beta_m2w
        self.clip_pct = clip_pct
        self.mu_law_mu = mu_law_mu
        self.momentum_bitwidth = momentum_bitwidth
        self.momentum_clip_pct = momentum_clip_pct
        self.momentum_mask_min = momentum_mask_min
        self.rd_bit_choices = [int(x) for x in rd_bit_choices.split(",") if len(x) > 0]
        self.sparsity_warmup_steps = sparsity_warmup_steps
        self.skip_layernorm_and_bias = skip_layernorm_and_bias
        self.qkv_special_handling = qkv_special_handling
        self.qkv_group_size = qkv_group_size
        self.qkv_clip_pct = qkv_clip_pct
        self.quant_start_step = quant_start_step
        self.apply_to_model = apply_to_model
        self.gop_size = gop_size
        self.i_frame_scale = i_frame_scale
        self.p_frame_scale = p_frame_scale
        self.save_dir = save_dir
        self.save_payload = save_payload
        self.target_nz_ratio = float(max(0.0, min(1.0, target_nz_ratio)))
        self.enforce_target_nz = enforce_target_nz
        self.m2w_gate = m2w_gate
        self.m2w_gate_q = m2w_gate_q
        self.ema_decay = ema_decay
        self.use_predictor = use_predictor
        self.predictor_mode = predictor_mode
        self.predictor_beta = predictor_beta
        self.pi_disable_threshold = pi_disable_threshold
        self.use_lagrange_alloc = use_lagrange_alloc
        self.min_apply_snr_db = float(min_apply_snr_db)
        self.enable_final_topk = bool(enable_final_topk)
        self.baseline_state_dict = None
        self._optimizer_ref = None
        self._printed_optim_info = False
        self._last_gop_start = 0
        self._stats = []
        self.nz_eps = 1e-12
        self._prev_eval_acc = None
        self._last_quant_step = -1
        self.predictor = SubbandPredictor(mode=self.predictor_mode, beta=self.predictor_beta) if self.use_predictor else None
        os.makedirs(self.save_dir, exist_ok=True)

    def _should_process(self, name: str, param: nn.Parameter) -> bool:
        if not param.requires_grad:
            return False
        if self.method == "none":
            return False
        lname = name.lower()
        if self.skip_layernorm_and_bias:
            if "layernorm" in lname or re.search(r"\bnorm\b", lname) or "ln_" in lname:
                return False
            if lname.endswith(".bias") or lname.endswith("_bias") or ".bias" in lname:
                return False
        if "embed" in lname or "embedding" in lname or "pos" in lname or "rope" in lname or "rotary" in lname:
            return False
        if lname.startswith("score.") or ".score." in lname:
            return False
        if param.ndim <= 1:
            return False
        return True

    def _current_frame_scale(self, step: int):
        if self.gop_size <= 1:
            return 1.0, True
        period = self.gop_size * self.quant_freq
        if (step - self._last_gop_start) >= period or step == 0:
            self._last_gop_start = step
        idx = (step - self._last_gop_start) // self.quant_freq
        if idx == 0:
            return self.i_frame_scale, True
        return self.p_frame_scale, False

    def _get_lr_for_param(self, optimizer, param):
        if optimizer is None:
            return None
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p is param:
                    return group.get("lr", None)
        return None

    def _ema_update_tensor(self, old_cpu: torch.Tensor, new_dev: torch.Tensor, decay: float):
        new_cpu = new_dev.detach().cpu()
        if old_cpu is None or old_cpu.numel() == 0:
            return new_cpu.clone()
        return (decay * old_cpu + (1.0 - decay) * new_cpu).clone()

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model: nn.Module, **kwargs):
        if self.method != "none":
            self.baseline_state_dict = {name: param.detach().cpu().clone() for name, param in model.state_dict().items()}
            self._optimizer_ref = kwargs.get("optimizer", None)
            with open(os.path.join(self.save_dir, "meta.json"), "w") as f:
                json.dump({"method": self.method}, f)

    def on_optimizer_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, optimizer=None, **kwargs):
        if optimizer is not None:
            self._optimizer_ref = optimizer

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        acc = metrics.get("eval_accuracy", None)
        if acc is not None:
            if self._prev_eval_acc is None:
                print(f"--- Eval @ step {state.global_step}: acc={acc:.4f}")
            else:
                delta = acc - self._prev_eval_acc
                print(f"--- Eval @ step {state.global_step}: acc={acc:.4f} (Δ{delta:+.4f} from last eval)")
            self._prev_eval_acc = acc

    def _hard_topk_mask_qkv(self, E: torch.Tensor, v: torch.Tensor, p_keep: float):
        n = E.numel()
        if n == 0:
            return torch.zeros_like(E, dtype=E.dtype)
        if p_keep >= 1.0:
            return torch.ones_like(E, dtype=E.dtype)
        if p_keep <= 0.0:
            return torch.zeros_like(E, dtype=E.dtype)
        if v is None:
            imp = E.abs()
        else:
            vbar = torch.mean(v.to(device=E.device, dtype=E.dtype))
            imp = E.abs() * torch.sqrt(torch.clamp(vbar, min=1e-12))
        k_keep = max(1, int(round(p_keep * n)))
        flat = imp.reshape(-1)
        k_small = max(1, n - k_keep + 1)
        thr = torch.kthvalue(flat, k_small).values
        mask = (imp >= thr).to(dtype=E.dtype)
        return mask

    def _enforce_topk_in_wavelet(self, coeffs_E: List[torch.Tensor], sens: List[float], p_keep: float):
        if p_keep >= 1.0:
            return [torch.ones_like(c, dtype=c.dtype) for c in coeffs_E]
        if p_keep <= 0.0:
            return [torch.zeros_like(c, dtype=c.dtype) for c in coeffs_E]
        imps = []
        for j, c in enumerate(coeffs_E):
            vbar = torch.tensor(max(sens[j], 1e-8), device=c.device, dtype=c.dtype)
            imp = c.abs() * torch.sqrt(vbar)
            imps.append(imp.reshape(-1))
        all_imp = torch.cat(imps, dim=0)
        total = all_imp.numel()
        k_keep = max(1, int(round(p_keep * total)))
        k_small = max(1, total - k_keep + 1)
        thr = torch.kthvalue(all_imp, k_small).values
        masks = []
        for j, c in enumerate(coeffs_E):
            vbar = torch.tensor(max(sens[j], 1e-8), device=c.device, dtype=c.dtype)
            imp = c.abs() * torch.sqrt(vbar)
            m = (imp >= thr).to(dtype=c.dtype)
            masks.append(m)
        return masks

    def _enforce_topk_spatial(self, x: torch.Tensor, p_keep: float):
        if p_keep >= 1.0:
            return x
        if p_keep <= 0.0:
            return torch.zeros_like(x)
        n = x.numel()
        k = max(1, int(round(p_keep * n)))
        flat = x.abs().reshape(-1)
        thr = torch.topk(flat, k, largest=True).values.min()
        return x * (x.abs() >= thr).to(x.dtype)

    def _momentum_gate_mask(self, mtmp: torch.Tensor):
        q = float(getattr(self, "m2w_gate_q", 0.9))
        ro = torch.quantile(mtmp.abs().float(), q)
        return (mtmp.abs() >= ro.to(device=mtmp.device, dtype=mtmp.dtype)).to(dtype=mtmp.dtype)

    def _compress_step(self, model: nn.Module, optimizer, state: TrainerState, save_tag: str):
        device = next(model.parameters()).device
        frame_scale, is_I = self._current_frame_scale(state.global_step)
        avg_bit = max(1.5, min(6.0, self.avg_bitwidth * frame_scale))
        alpha = self.alpha_w2m * frame_scale
        beta = self.beta_m2w
        warm = max(1, getattr(self, "sparsity_warmup_steps", 1000))
        steps_since_start = max(0, state.global_step - self.quant_start_step)
        progress = min(1.0, steps_since_start / float(warm)) if state.global_step >= self.quant_start_step else 0.0
        current_keep = 1.0 - (1.0 - self.target_nz_ratio) * progress
        micro_eval_enabled = hasattr(self, "diag_micro_eval_size") and getattr(self, "diag_micro_eval_size") > 0 and hasattr(self, "calib_batch") and self.calib_batch is not None
        train_was = model.training
        model.eval()
        with torch.no_grad():
            if micro_eval_enabled:
                batch = {k: v.to(device) for k, v in self.calib_batch.items() if k != "labels"}
                labels = self.calib_batch.get("labels", None)
                if labels is not None:
                    labels = labels.to(device)
                out_pre = model(**batch)
                logits_pre = out_pre.logits.detach()
                if labels is not None:
                    acc_pre = (logits_pre.argmax(dim=-1) == labels).float().mean().item()
                else:
                    acc_pre = None
        if train_was:
            model.train()
        total_raw = 0
        total_comp = 0
        payload_paths = []
        t0 = time.perf_counter()
        total_elems = 0
        total_nnz = 0
        qkv_fast_layers = 0
        qkv_fast_elems = 0
        mse_weighted_sum = 0.0
        var_weighted_sum = 0.0
        layer_stats = []
        nmse_qkv_sum = 0.0
        var_qkv_sum = 0.0
        nmse_oth_sum = 0.0
        var_oth_sum = 0.0
        mask_means_per_layer = []
        alloc_bits_sum = 0.0
        alloc_bits_count = 0
        per_layer_band_logs = []
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name not in self.baseline_state_dict:
                    continue
                if not self._should_process(name, param):
                    continue
                base = self.baseline_state_dict[name].to(device=device, dtype=param.dtype)
                w_now = param.data
                pstate = optimizer.state.get(param, {}) if optimizer is not None else {}
                m = pstate.get("exp_avg", None)
                v = pstate.get("exp_avg_sq", None)
                lr = self._get_lr_for_param(optimizer, param)
                if m is None or v is None or lr is None:
                    pred = torch.zeros_like(w_now)
                else:
                    pred = -lr * (m.to(device=param.device, dtype=param.dtype) / (torch.sqrt(v.to(device=param.device, dtype=param.dtype) + 1e-12)))
                resid_true = w_now - base
                E = resid_true - pred
                n_param = E.numel()
                is_qkv = self.qkv_special_handling and ("attention.query_key_value.weight" in name or "q_proj" in name or "k_proj" in name or "v_proj" in name)
                if is_qkv:
                    qkv_fast_layers += 1
                    qkv_fast_elems += n_param
                    bq = int(round(avg_bit))
                    if self.enforce_target_nz:
                        mask = self._hard_topk_mask_qkv(E, v, current_keep)
                        E_masked = E * mask
                    else:
                        E_masked = E
                        mask = (E_masked.abs() > self.nz_eps).to(dtype=E_masked.dtype)
                    rec_E = quantize_pergroup_uniform(E_masked, n_bits=bq, group_size=self.qkv_group_size, dim=-1, clip_pct=self.qkv_clip_pct)
                    keep_mask = (rec_E.abs() > self.nz_eps).to(dtype=E.dtype)
                    err = (E.float() - rec_E.float())
                    msqe = float(torch.mean(err * err).item())
                    varE = float(E.float().var(unbiased=False).item())
                    snr_db_layer = 10.0 * np.log10(max(varE / max(msqe, 1e-12), 1e-12))
                    apply_this_layer = self.apply_to_model and (snr_db_layer >= self.min_apply_snr_db)
                    if apply_this_layer:
                        param.data.copy_(base + pred + rec_E)
                    nnz = int((keep_mask > 0).sum().item())
                    total_elems += n_param
                    total_nnz += nnz
                    mse_weighted_sum += msqe * n_param
                    var_weighted_sum += varE * n_param
                    keep_ratio = float(nnz) / float(max(n_param, 1))
                    layer_stats.append({
                        "name": name,
                        "is_qkv": True,
                        "numel": int(n_param),
                        "keep_ratio": keep_ratio,
                        "b_avg_layer": float(bq) if nnz > 0 else 0.0,
                        "mse": msqe,
                        "var": varE,
                        "snr_db": snr_db_layer
                    })
                    nmse_qkv_sum += msqe * n_param
                    var_qkv_sum += varE * n_param
                    mask_means_per_layer.append(float(keep_mask.float().mean().item()))
                    alloc_bits_sum += float(bq) * nnz
                    alloc_bits_count += nnz
                    continue
                blocks_E, pad_E = blockify_flat(E, self.block_size)
                coeffs_E, idwt = dwt1d_blocks(blocks_E, self.dwt_levels)
                energies = [float(torch.mean(c.float() ** 2).item()) for c in coeffs_E]
                if v is None:
                    sens = [1.0 for _ in coeffs_E]
                else:
                    blocks_v, _ = blockify_flat(v.to(device=E.device, dtype=E.dtype), self.block_size)
                    coeffs_v, _ = dwt1d_blocks(blocks_v, self.dwt_levels)
                    sens = [float(torch.mean(cv.float()).item() + 1e-8) for cv in coeffs_v]
                coeffs_orig = [c.clone() for c in coeffs_E]
                band_logs = []
                if self.use_predictor and not is_I and (self.predictor is not None):
                    preds = []
                    Pis = []
                    for j, c in enumerate(coeffs_E):
                        pred_full, PI = self.predictor.update_and_predict((name, j), c, is_I)
                        use_pred = PI >= self.pi_disable_threshold
                        if not use_pred:
                            pred_full = torch.zeros_like(c)
                        preds.append(pred_full)
                        Pis.append(float(PI))
                    for j, c in enumerate(coeffs_E):
                        coeffs_E[j] = c - preds[j]
                else:
                    preds = [torch.zeros_like(c) for c in coeffs_E]
                    Pis = [None for _ in coeffs_E]
                for j, c in enumerate(coeffs_E):
                    med = median_abs(c, dim=-1, keepdim=True)
                    vbar = torch.tensor(max(sens[j], 1e-8), device=c.device, dtype=c.dtype)
                    tau = alpha * med / (torch.sqrt(vbar) + 1e-12)
                    coeffs_E[j] = soft_threshold(c, tau)
                    band_logs.append({"j": j, "tau_mean": float(tau.mean().item())})
                if self.enforce_target_nz:
                    masks_wavelet = self._enforce_topk_in_wavelet(coeffs_E, sens, current_keep)
                    for j in range(len(coeffs_E)):
                        coeffs_E[j] = coeffs_E[j] * masks_wavelet[j]
                energies_after = [float(torch.mean(c.float() ** 2).item()) for c in coeffs_E]
                counts_nz = [int((c.abs() > self.nz_eps).sum().item()) for c in coeffs_E]
                active = [i for i, nz in enumerate(counts_nz) if nz > 0]
                if active:
                    e_act = [energies_after[i] for i in active]
                    s_act = [sens[i] for i in active]
                    n_act = [counts_nz[i] for i in active]
                    if self.use_lagrange_alloc:
                        b_act = bit_allocation_rd_lagrange(e_act, s_act, n_act, avg_bit, self.rd_bit_choices)
                    else:
                        b_act = bit_allocation_rd(e_act, s_act, n_act, avg_bit, self.rd_bit_choices)
                    bits = [0] * len(coeffs_E)
                    for j, i in enumerate(active):
                        bits[i] = b_act[j]
                else:
                    bits = [0] * len(coeffs_E)
                nnz_sum_layer = 0
                b_sum_layer = 0
                vmax_list, mu_list, q_list, deq_coeffs = [], [], [], []
                for j, c in enumerate(coeffs_E):
                    b = int(bits[j])
                    if b <= 1:
                        xrec = torch.zeros_like(c)
                        vmax = torch.tensor(0.0, device=c.device, dtype=c.dtype).repeat(c.shape[0], 1)
                        q = torch.zeros_like(c, dtype=torch.int16)
                        vmax_list.append(vmax)
                        mu_list.append(torch.tensor(self.mu_law_mu, device=c.device))
                        q_list.append(q)
                        xrec_plus = xrec + preds[j]
                        deq_coeffs.append(xrec_plus)
                    else:
                        xrec, vmax, muval, q = quantize_subband_mulaw(c, b, self.clip_pct, self.mu_law_mu)
                        vmax_list.append(vmax)
                        mu_list.append(muval)
                        q_list.append(q.to(torch.int16))
                        xrec_plus = xrec + preds[j]
                        deq_coeffs.append(xrec_plus)
                    if b > 1:
                        nnz_sub = int((c.abs() > self.nz_eps).sum().item())
                        nnz_sum_layer += nnz_sub
                        b_sum_layer += b * nnz_sub
                if nnz_sum_layer > 0:
                    b_avg_layer = b_sum_layer / nnz_sum_layer
                else:
                    b_avg_layer = 0.0
                alloc_bits_sum += b_sum_layer
                alloc_bits_count += nnz_sum_layer
                rec_blocks = idwt1d_blocks(deq_coeffs, idwt)
                rec_flat = unblockify_flat(rec_blocks, E.numel())
                rec_E = rec_flat.reshape_as(E)
                if self.enable_final_topk and self.enforce_target_nz and (0.0 < current_keep < 1.0):
                    flat = rec_E.abs().reshape(-1)
                    k = max(1, int(round(current_keep * flat.numel())))
                    kth = torch.kthvalue(flat, flat.numel() - k + 1).values
                    rec_E = rec_E * (rec_E.abs() >= kth).to(rec_E.dtype)
                keep_mask = (rec_E.abs() > self.nz_eps).to(dtype=E.dtype)
                err = (E.float() - rec_E.float())
                msqe = float(torch.mean(err * err).item())
                varE = float(E.float().var(unbiased=False).item())
                snr_db_layer = 10.0 * np.log10(max(varE / max(msqe, 1e-12), 1e-12))
                apply_this_layer = self.apply_to_model and (snr_db_layer >= self.min_apply_snr_db)
                if apply_this_layer:
                    param.data.copy_(base + pred + rec_E)
                if self.save_payload:
                    meta_shapes = {"orig_shape": list(E.shape), "block_size": self.block_size, "pad": pad_E, "levels": self.dwt_levels}
                    extra_meta = {"alpha": alpha, "beta": beta, "avg_bit": avg_bit, "is_I": is_I, "current_keep": current_keep}
                    comp, raw_sz, comp_sz, meta = serialize_param_payload(name, bits, vmax_list, mu_list, q_list, meta_shapes, extra_meta)
                    step_dir = os.path.join(self.save_dir, f"step_{save_tag}")
                    os.makedirs(step_dir, exist_ok=True)
                    out_path = os.path.join(step_dir, f"{name.replace('.', '_')}.npz.lzma")
                    with open(out_path, "wb") as f:
                        f.write(comp)
                    total_raw += raw_sz
                    total_comp += comp_sz
                    payload_paths.append(out_path)
                nnz = int((keep_mask > 0).sum().item())
                total_elems += n_param
                total_nnz += nnz
                mse_weighted_sum += msqe * n_param
                var_weighted_sum += varE * n_param
                keep_ratio = float(nnz) / float(max(n_param, 1))
                layer_stats.append({
                    "name": name,
                    "is_qkv": False,
                    "numel": int(n_param),
                    "keep_ratio": keep_ratio,
                    "b_avg_layer": float(b_avg_layer),
                    "mse": msqe,
                    "var": varE,
                    "snr_db": snr_db_layer
                })
                nmse_oth_sum += msqe * n_param
                var_oth_sum += varE * n_param
                mask_means_per_layer.append(float(keep_mask.float().mean().item()))
                band_detail = []
                for j in range(len(coeffs_E)):
                    var_c = float(coeffs_orig[j].float().var(unbiased=False).item()) if coeffs_orig[j].numel() > 0 else 0.0
                    var_d = float((coeffs_orig[j].float()).var(unbiased=False).item()) if coeffs_orig[j].numel() > 0 else 0.0
                    mse_q = float(((coeffs_orig[j].float() - (deq_coeffs[j].float())).pow(2)).mean().item()) if coeffs_orig[j].numel() > 0 else 0.0
                    band_detail.append({
                        "band": j,
                        "energy_pre": float(energies[j]),
                        "energy_post": float(energies_after[j]),
                        "bits": int(bits[j]),
                        "PI": None if Pis[j] is None else float(Pis[j]),
                        "var_c": var_c,
                        "var_pred": var_d,
                        "mse_quant": mse_q
                    })
                per_layer_band_logs.append({
                    "step": int(state.global_step),
                    "layer": name,
                    "is_I": bool(is_I),
                    "current_keep": float(current_keep),
                    "bands": band_detail
                })
        if self.save_payload:
            idx_path = os.path.join(self.save_dir, f"step_{save_tag}", "index.json")
            with open(idx_path, "w") as f:
                json.dump({"step": state.global_step, "total_raw_bytes": int(total_raw), "total_comp_bytes": int(total_comp), "paths": payload_paths}, f, indent=2)
        self._stats.append({"step": int(state.global_step), "raw_bytes": int(total_raw), "comp_bytes": int(total_comp)})
        if alloc_bits_count > 0:
            b_avg = alloc_bits_sum / alloc_bits_count
        else:
            b_avg = float(self.avg_bitwidth)
        p = _safe_div(total_nnz, total_elems)
        bits_per_param = p * b_avg + _binary_entropy(p)
        cr_weight = 32.0 / max(bits_per_param, 1e-9)
        nmse = _safe_div(mse_weighted_sum, var_weighted_sum)
        snr_db = 10.0 * np.log10(max(1.0 / max(nmse, 1e-12), 1e-12))
        mask_mean = float(np.mean(mask_means_per_layer)) if mask_means_per_layer else 1.0
        mask_min = float(np.min(mask_means_per_layer)) if mask_means_per_layer else 1.0
        mask_max = float(np.max(mask_means_per_layer)) if mask_means_per_layer else 1.0
        nmse_qkv = _safe_div(nmse_qkv_sum, var_qkv_sum) if var_qkv_sum > 0 else 0.0
        nmse_oth = _safe_div(nmse_oth_sum, var_oth_sum) if var_oth_sum > 0 else 0.0
        t1 = time.perf_counter()
        per_million = (t1 - t0) * 1000.0 / max(total_elems / 1e6, 1e-9)
        print(f"    - Frame: {'I' if is_I else 'P'} | Keep(target→now): {self.target_nz_ratio:.3f}->{current_keep:.3f} | Actual p: {p:.4f}")
        print(f"    - Est. rate: b_avg≈{b_avg:.2f} bits, bits/param≈{bits_per_param:.3f}, CR(weight)≈{cr_weight:.1f}x")
        print(f"    - Residual NMSE: {nmse:.3e} | SNR: {snr_db:.2f} dB")
        print(f"    - Momentum Mask (layer-avg keep): mean={mask_mean:.3f}, min={mask_min:.3f}, max={mask_max:.3f}")
        print(f"    - QKV fast path: {qkv_fast_layers} layers, {qkv_fast_elems/1e6:.2f}M elems")
        print(f"    - Group NMSE: QKV={nmse_qkv:.3e}, OTHER={nmse_oth:.3e}")
        print(f"    - Compress wall time: {(t1 - t0):.2f}s ({per_million:.2f} ms / 1M params)")
        if layer_stats:
            worst = sorted(layer_stats, key=lambda d: d["snr_db"])[:5]
            for w in worst:
                print(f"      [worst] {w['name']:<48} | SNR {w['snr_db']:+6.2f} dB | keep {w['keep_ratio']:.3f} | b≈{w['b_avg_layer']:.2f}")
        if getattr(self, "diag_layerwise_json", False):
            step_dir = os.path.join(self.save_dir, f"step_{save_tag}")
            os.makedirs(step_dir, exist_ok=True)
            with open(os.path.join(step_dir, "layer_stats.json"), "w") as f:
                json.dump({
                    "step": int(state.global_step),
                    "overall": {
                        "frame": "I" if is_I else "P",
                        "current_keep": current_keep,
                        "b_avg": b_avg,
                        "bits_per_param": bits_per_param,
                        "nmse": nmse,
                        "snr_db": snr_db,
                        "nmse_qkv": nmse_qkv,
                        "nmse_other": nmse_oth
                    },
                    "layers": layer_stats,
                    "per_layer_bands": per_layer_band_logs
                }, f, indent=2)
        if micro_eval_enabled:
            model.eval()
            with torch.no_grad():
                batch = {k: v.to(device) for k, v in self.calib_batch.items() if k != "labels"}
                labels = self.calib_batch.get("labels", None)
                if labels is not None:
                    labels = labels.to(device)
                out_post = model(**batch)
                logits_post = out_post.logits.detach()
                logit_mse = float(torch.mean((logits_post - logits_pre)**2).item())
                p_pre = torch.softmax(logits_pre, dim=-1).clamp_min(1e-12)
                p_post = torch.softmax(logits_post, dim=-1).clamp_min(1e-12)
                kl = float(torch.mean(torch.sum(p_pre * (p_pre.log() - p_post.log()), dim=-1)).item())
                if labels is not None:
                    acc_post = (logits_post.argmax(dim=-1) == labels).float().mean().item()
                    print(f"    - MicroEval ({getattr(self, 'diag_micro_eval_size', 0)} ex): Δacc={acc_post - acc_pre:+.4f}, KL={kl:.4e}, logitMSE={logit_mse:.4e}")
                else:
                    print(f"    - MicroEval ({getattr(self, 'diag_micro_eval_size', 0)} ex): KL={kl:.4e}, logitMSE={logit_mse:.4e}")
            if train_was:
                model.train()

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model: nn.Module = None, optimizer: torch.optim.Optimizer = None, **kwargs):
        if self.baseline_state_dict is None or self.method == "none":
            return
        if state.global_step < self.quant_start_step:
            return
        if state.global_step > 0 and state.global_step % self.quant_freq == 0:
            tag = f"{state.global_step:08d}"
            self._compress_step(model, optimizer if optimizer is not None else self._optimizer_ref, state, tag)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="warp_q", choices=["none", "warp_q"])
    parser.add_argument("--avg_bitwidth", type=float, default=4.0)
    parser.add_argument("--momentum_bitwidth", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--dwt_levels", type=int, default=3)
    parser.add_argument("--alpha_w2m", type=float, default=5e-5)
    parser.add_argument("--beta_m2w", type=float, default=2.0)
    parser.add_argument("--clip_pct", type=float, default=0.995)
    parser.add_argument("--mu_law_mu", type=float, default=255.0)
    parser.add_argument("--momentum_clip_pct", type=float, default=0.999)
    parser.add_argument("--momentum_mask_min", type=float, default=0.5)
    parser.add_argument("--rd_bit_choices", type=str, default="2,3,4,5")
    parser.add_argument("--sparsity_warmup_steps", type=int, default=1000)
    parser.add_argument("--skip_layernorm_and_bias", action="store_true", default=True)
    parser.add_argument("--qkv_special_handling", action="store_true", default=False)
    parser.add_argument("--qkv_group_size", type=int, default=64)
    parser.add_argument("--qkv_clip_pct", type=float, default=0.995)
    parser.add_argument("--quant_start_step", type=int, default=200)
    parser.add_argument("--quant_freq", type=int, default=200)
    parser.add_argument("--apply_to_model", action="store_true", default=False)
    parser.add_argument("--gop_size", type=int, default=4)
    parser.add_argument("--i_frame_scale", type=float, default=1.0)
    parser.add_argument("--p_frame_scale", type=float, default=0.8)
    parser.add_argument("--save_dir", type=str, default="./compressed_ckpts")
    parser.add_argument("--save_payload", action="store_true", default=False)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--eval_freq", type=int, default=200)
    parser.add_argument("--save_freq", type=int, default=200)
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-410m")
    parser.add_argument("--dataset_name", type=str, default="glue")
    parser.add_argument("--dataset_config", type=str, default="sst2")
    parser.add_argument("--output_dir", type=str, default="./results_warpq")
    parser.add_argument("--target_nz_ratio", type=float, default=0.1)
    parser.add_argument("--enforce_target_nz", action="store_true", default=True)
    parser.add_argument("--m2w_gate", action="store_true", default=False)
    parser.add_argument("--m2w_gate_q", type=float, default=0.9)
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--diag_micro_eval_size", type=int, default=0)
    parser.add_argument("--diag_layerwise_json", action="store_true", default=True)
    parser.add_argument("--use_predictor", action="store_true", default=True)
    parser.add_argument("--predictor_mode", type=str, default="ar1", choices=["ema", "ar1"])
    parser.add_argument("--predictor_beta", type=float, default=0.9)
    parser.add_argument("--pi_disable_threshold", type=float, default=0.1)
    parser.add_argument("--use_lagrange_alloc", action="store_true", default=True)
    parser.add_argument("--min_apply_snr_db", type=float, default=8.0)
    parser.add_argument("--enable_final_topk", action="store_true", default=True)
    args = parser.parse_args()
    set_seed(args.seed)
    MODEL_NAME = args.model_name
    DATASET_NAME, DATASET_CONFIG = args.dataset_name, args.dataset_config
    OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.config.pad_token_id = model.config.eos_token_id
    raw_datasets = load_dataset(DATASET_NAME, DATASET_CONFIG)
    tokenized_datasets = raw_datasets.map(lambda ex: tokenizer(ex["sentence"], truncation=True, padding="longest"), batched=True)
    accuracy_metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        return accuracy_metric.compute(predictions=np.argmax(eval_pred.predictions, axis=1), references=eval_pred.label_ids)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=args.lr,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_strategy="steps",
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=args.eval_freq,
        save_strategy="steps",
        save_steps=args.save_freq,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        bf16=False,
        report_to="none",
        group_by_length=True,
    )
    calib_batch = None
    if args.diag_micro_eval_size > 0:
        n = min(args.diag_micro_eval_size, len(raw_datasets["validation"]))
        sample = raw_datasets["validation"].select(range(n))
        possible_text_keys = ["sentence", "text", "premise", "question", "content", "sentence1"]
        text_key = None
        for k in possible_text_keys:
            if k in sample.column_names:
                text_key = k
                break
        if text_key is None:
            raise ValueError(f"Could not find a text column in validation set. Available columns: {sample.column_names}")
        texts = sample[text_key]
        if not isinstance(texts, (list, tuple)):
            texts = list(texts)
        texts = [str(t) for t in texts]
        toks = tokenizer(
            text=texts,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )
        label_key = "label" if "label" in sample.column_names else ("labels" if "labels" in sample.column_names else None)
        if label_key is not None:
            labels = torch.tensor(list(sample[label_key]), dtype=torch.long)
            calib_batch = {"input_ids": toks["input_ids"], "attention_mask": toks["attention_mask"], "labels": labels}
        else:
            calib_batch = {"input_ids": toks["input_ids"], "attention_mask": toks["attention_mask"]}
    callback = WaRPQCallback(
        method=args.method,
        avg_bitwidth=args.avg_bitwidth,
        quant_freq=args.quant_freq,
        block_size=args.block_size,
        dwt_levels=args.dwt_levels,
        alpha_w2m=args.alpha_w2m,
        beta_m2w=args.beta_m2w,
        clip_pct=args.clip_pct,
        mu_law_mu=args.mu_law_mu,
        momentum_bitwidth=args.momentum_bitwidth,
        momentum_clip_pct=args.momentum_clip_pct,
        momentum_mask_min=args.momentum_mask_min,
        rd_bit_choices=args.rd_bit_choices,
        sparsity_warmup_steps=args.sparsity_warmup_steps,
        skip_layernorm_and_bias=args.skip_layernorm_and_bias,
        qkv_special_handling=args.qkv_special_handling,
        qkv_group_size=args.qkv_group_size,
        qkv_clip_pct=args.qkv_clip_pct,
        quant_start_step=args.quant_start_step,
        apply_to_model=args.apply_to_model,
        gop_size=args.gop_size,
        i_frame_scale=args.i_frame_scale,
        p_frame_scale=args.p_frame_scale,
        save_dir=args.save_dir,
        save_payload=args.save_payload,
        target_nz_ratio=args.target_nz_ratio,
        enforce_target_nz=args.enforce_target_nz,
        m2w_gate=args.m2w_gate,
        ema_decay=args.ema_decay,
        use_predictor=args.use_predictor,
        predictor_mode=args.predictor_mode,
        predictor_beta=args.predictor_beta,
        pi_disable_threshold=args.pi_disable_threshold,
        use_lagrange_alloc=args.use_lagrange_alloc,
        min_apply_snr_db=args.min_apply_snr_db,
        enable_final_topk=args.enable_final_topk
    )
    setattr(callback, "diag_micro_eval_size", int(args.diag_micro_eval_size))
    setattr(callback, "diag_layerwise_json", bool(args.diag_layerwise_json))
    setattr(callback, "calib_batch", calib_batch)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[JsonLoggingCallback(os.path.join(OUTPUT_DIR, "training_log.json")), callback],
    )
    print("--- Starting Training ---")
    print(f"Method: {args.method}")
    print(f"Avg bitwidth: {args.avg_bitwidth}, Momentum bitwidth: {args.momentum_bitwidth}")
    print(f"DWT levels: {args.dwt_levels}, Block size: {args.block_size}, Clip pct: {args.clip_pct}, mu: {args.mu_law_mu}")
    print(f"RD bit choices: {args.rd_bit_choices}, GOP size: {args.gop_size}, I-scale: {args.i_frame_scale}, P-scale: {args.p_frame_scale}")
    print(f"Use predictor: {args.use_predictor} ({args.predictor_mode}), PI thr: {args.pi_disable_threshold}")
    print(f"Apply back to model: {args.apply_to_model}, Save payload: {args.save_payload}, Save dir: {args.save_dir}")
    print(f"Target nz ratio: {args.target_nz_ratio}, Warmup steps: {args.sparsity_warmup_steps}, FinalTopK: {args.enable_final_topk}")
    print(f"Min apply SNR dB: {args.min_apply_snr_db}")
    print(f"Output Directory: {OUTPUT_DIR}")
    trainer.train()
    print("--- Training Finished ---")
    eval_results = trainer.evaluate()
    print(f"Final evaluation results: {eval_results}")
    trainer.save_model(os.path.join(OUTPUT_DIR, "best_model_final"))
    print(f"--- Best model saved to {os.path.join(OUTPUT_DIR, 'best_model_final')} ---")

if __name__ == "__main__":
    main()
