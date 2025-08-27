import os, json, math, glob
import matplotlib.pyplot as plt

OUTDIR = "./runs/compare"
os.makedirs(OUTDIR, exist_ok=True)

RUNS = {
    "FP32": "./runs/fp32_fixed",
    "ExCP": "./runs/excp_fixed",
    "DCT+Z+DZ": "./runs/warp_dct2_zigzag_dz",
}

def load_eval(run_dir):
    path = os.path.join(run_dir, "training_log.json")
    if not os.path.isfile(path):
        return [], [], []
    logs = json.load(open(path))
    steps, evl, ppl = [], [], []
    for e in logs:
        if "eval_loss" in e:
            steps.append(e["step"])
            evl.append(e["eval_loss"])
            ppl.append(math.exp(e["eval_loss"]))
    return steps, evl, ppl

def find_stats_file(run_dir):
    cands = glob.glob(os.path.join(run_dir, "**", "compress_stats.jsonl"), recursive=True)
    return cands[0] if cands else None

def load_stats(stats_path):
    if not stats_path or not os.path.isfile(stats_path):
        return {}
    S = {"step": [], "bpp": [], "cr": [], "energy_keep": [], "dz_zero": []}
    with open(stats_path) as f:
        for line in f:
            try:
                j = json.loads(line)
            except:
                continue
            S["step"].append(j.get("step"))
            S["bpp"].append(j.get("bpp", None))
            S["cr"].append(j.get("cr_weight", None))
            S["energy_keep"].append(j.get("spec_energy_keep", None))
            S["dz_zero"].append(j.get("dz_zero_frac", None))
    return S

# 1) eval loss / ppl 叠加
plt.figure()
for name, d in RUNS.items():
    s, evl, ppl = load_eval(d)
    if s:
        plt.plot(s, evl, label=name)
plt.xlabel("step"); plt.ylabel("eval_loss"); plt.title("Validation Loss"); plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "overlay_eval_loss.png"), dpi=160)

plt.figure()
for name, d in RUNS.items():
    s, evl, ppl = load_eval(d)
    if s:
        plt.plot(s, ppl, label=name)
plt.xlabel("step"); plt.ylabel("perplexity"); plt.title("Validation PPL"); plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "overlay_eval_ppl.png"), dpi=160)

# 2) 压缩相关曲线叠加（ExCP 与 DCT+Z+DZ 可能有，FP32通常没有）
plt.figure()
for name, d in RUNS.items():
    sp = find_stats_file(d)
    S = load_stats(sp)
    if S.get("step"):
        plt.plot(S["step"], S["bpp"], label=f"{name} bpp")
plt.xlabel("step"); plt.ylabel("bpp"); plt.title("Bits per Parameter (estimated)")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "overlay_bpp.png"), dpi=160)

plt.figure()
for name, d in RUNS.items():
    sp = find_stats_file(d)
    S = load_stats(sp)
    if S.get("step"):
        plt.plot(S["step"], S["cr"], label=f"{name} CR")
plt.xlabel("step"); plt.ylabel("Compression Ratio (weight)"); plt.title("Compression Ratio (estimated)")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "overlay_cr.png"), dpi=160)

# 3) 频域能量与 dead-zone 零值率
plt.figure()
for name, d in RUNS.items():
    sp = find_stats_file(d)
    S = load_stats(sp)
    if S.get("step") and any(x is not None for x in S["energy_keep"]):
        y = [x if x is not None else float("nan") for x in S["energy_keep"]]
        plt.plot(S["step"], y, label=f"{name} energy_keep")
plt.xlabel("step"); plt.ylabel("energy_keep")
plt.title("Spectral Energy Retention (after Z-mask)")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "overlay_energy_keep.png"), dpi=160)

plt.figure()
for name, d in RUNS.items():
    sp = find_stats_file(d)
    S = load_stats(sp)
    if S.get("step") and any(x is not None for x in S["dz_zero"]):
        y = [x if x is not None else float("nan") for x in S["dz_zero"]]
        plt.plot(S["step"], y, label=f"{name} deadzone_zero")
plt.xlabel("step"); plt.ylabel("deadzone zero fraction")
plt.title("Dead-zone Zero Fraction")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "overlay_deadzone_zero.png"), dpi=160)

print("Saved to:", OUTDIR)
