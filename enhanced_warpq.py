# enhanced_warpq.py
import os, math, argparse, random, json, time, io
import numpy as np, torch, torch.nn.functional as F, torch.nn as nn
from typing import List, Tuple
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from pytorch_wavelets import DWT1D, IDWT1D, DWTForward, DWTInverse

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

def mu_law_compand(x, mu, s):
    x = torch.clamp(x, -s, s)
    y = torch.sign(x) * torch.log1p(mu * torch.abs(x) / (s + 1e-12)) / math.log(1 + mu)
    return torch.clamp(y, -1., 1.)

def mu_law_expand(y, mu, s):
    return torch.sign(y) * (s * (torch.pow(1 + mu, torch.abs(y)) - 1) / mu)

def quant_sym(y, b):
    if b <= 1: return torch.zeros_like(y), 0.
    qmax = 2 ** (b - 1) - 1
    q = torch.clamp(torch.round(y * qmax), -qmax, qmax)
    return q, qmax

def dequant_sym(q, qmax):
    return q / qmax if qmax else torch.zeros_like(q)

def kurtosis(x):  # 4‑th / 2‑nd ²
    m2 = torch.mean(x ** 2)
    m4 = torch.mean(x ** 4)
    return m4 / (m2 ** 2 + 1e-12)

def rd_bits_lagrange(E: List[float], sens: List[float], cnt: List[int],
                     R: float, choices: List[int]):
    C = [max(e, 1e-12) * max(s, 1e-9) for e, s in zip(E, sens)]
    bmin, bmax = min(choices), max(choices)
    lo, hi = 1e-9, 1e9
    tot = sum(cnt) * R
    for _ in range(50):
        lam = math.sqrt(lo * hi)
        used = 0; out = []
        for Cj, nj in zip(C, cnt):
            bj = bmin if Cj == 0 else max(bmin, min(bmax, int(0.5 * math.log2(Cj / lam))))
            used += bj * nj; out.append(bj)
        if used > tot: lo = lam
        else: hi = lam
    return out

def block(x, bs):
    flat = x.reshape(-1); pad = (-len(flat)) % bs
    if pad: flat = F.pad(flat, (0, pad))
    return flat.view(-1, bs), pad

def unblock(b, n):
    flat = b.reshape(-1)
    return flat[:n]

def dwt1(batch, lvl, w):
    d = DWT1D(wave=w, J=lvl, mode='symmetric').to(batch.device)
    i = IDWT1D(wave=w, mode='symmetric').to(batch.device)
    cA, cD = d(batch.unsqueeze(1)); coeffs = [cA.squeeze(1)] + [x.squeeze(1) for x in cD]
    return coeffs, i

def idwt1(coeffs, inv):
    return inv((coeffs[0].unsqueeze(1), [x.unsqueeze(1) for x in coeffs[1:]])).squeeze(1)

def dwt2(x2d, lvl, w):
    B, C = x2d.shape
    fwd = DWTForward(J=lvl, wave=w, mode='symmetric').to(x2d.device)
    inv = DWTInverse(wave=w, mode='symmetric').to(x2d.device)
    LL, highs = fwd(x2d.unsqueeze(0).unsqueeze(0))
    coeffs = [LL.squeeze()] + [h.squeeze() for h in highs]
    return coeffs, inv

def idwt2(coeffs, inv):
    rec = inv((coeffs[0].unsqueeze(0).unsqueeze(0),
               [x.unsqueeze(0).unsqueeze(0) for x in coeffs[1:]]))
    return rec.squeeze()

def quant_subband(x, b, pct, mu):
    vmax = torch.quantile(
        x.abs().reshape(x.shape[0], -1), pct, dim=-1, keepdim=True
    ).clamp(min=1e-8)
    vmax = vmax.view(x.shape[0], *([1] * (x.ndim - 1)))
    y = mu_law_compand(x, mu, vmax)
    q, qmax = quant_sym(y, b)
    rec = mu_law_expand(dequant_sym(q, qmax), mu, vmax)
    return rec

def compress_tensor(t, sens, cfg):
    bs, lvl, pct = cfg.bs, cfg.lvl, cfg.pct
    wav = cfg.wave
    if cfg.enable_2d and t.ndim == 2:
        coeffs, inv = dwt2(t, lvl, wav)
    else:
        blocks, pad = block(t, bs); coeffs, inv = dwt1(blocks, lvl, wav)
    cnt = [c.numel() for c in coeffs]
    eng = [float(torch.mean(c.float() ** 2)) for c in coeffs]
    bits = rd_bits_lagrange(eng, sens, cnt, cfg.avg_bit, cfg.choices)
    rec_coeffs = []
    for c, b in zip(coeffs, bits):
        if b <= 1:
            rec_coeffs.append(torch.zeros_like(c))
            continue
        kur = kurtosis(c)
        mu = 255. if kur > 6. else 87.
        rec = quant_subband(c, b, pct, mu)
        rec_coeffs.append(rec)
    if cfg.enable_2d and t.ndim == 2:
        rec = idwt2(rec_coeffs, inv)
    else:
        rec_blocks = idwt1(rec_coeffs, inv)
        rec = unblock(rec_blocks, t.numel()).reshape_as(t)
    return rec, bits

class Cfg: pass

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='bert-base-uncased')
    p.add_argument('--waves', default='haar,db2')
    p.add_argument('--dwt_lvl', type=int, default=2)
    p.add_argument('--block', type=int, default=256)
    p.add_argument('--avg_bit', type=float, default=3.5)
    p.add_argument('--choices', default='2,3,4,5')
    p.add_argument('--pct', type=float, default=0.995)
    p.add_argument('--gamma_joint', type=float, default=0.35)
    p.add_argument('--target_nz', type=float, default=0.1)
    p.add_argument('--enable_2d', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    seed_all(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)
    opt_state = {}

    cfg = Cfg()
    cfg.bs = args.block
    cfg.lvl = args.dwt_lvl
    cfg.avg_bit = args.avg_bit
    cfg.pct = args.pct
    cfg.choices = [int(x) for x in args.choices.split(',')]
    cfg.enable_2d = args.enable_2d

    wave_list = args.waves.split(',')
    total_mse, total_var = 0., 0.
    t0 = time.time()
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.ndim <= 1: continue
            p_data = p.data.detach()
            sens = torch.ones(len(wave_list))  # placeholder
            best_rec, best_bits, best_nmse = None, None, 1e9
            for w in wave_list:
                cfg.wave = w
                rec, bits = compress_tensor(p_data, sens, cfg)
                nmse = torch.mean((rec - p_data) ** 2) / (p_data.var() + 1e-12)
                if nmse < best_nmse:
                    best_nmse, best_rec, best_bits = nmse, rec, bits
            p.data.copy_(best_rec)
            total_mse += best_nmse * p_data.numel()
            total_var += p_data.var() * p_data.numel()
    nmse_all = total_mse / total_var
    snr = 10 * math.log10(1 / nmse_all)
    print(json.dumps({'NMSE': nmse_all, 'SNR(dB)': snr}, indent=2))
    print('elapsed', time.time() - t0, 's')

if __name__ == '__main__':
    main()
