import os, json, math
import matplotlib.pyplot as plt

OUTDIR = "./runs/compare"
os.makedirs(OUTDIR, exist_ok=True)

RUNS = {
    "FP32": "./runs/fp32_fixed",
    "ExCP": "./runs/excp_fixed",
    "DCT+Z+DZ": "./runs/warp_dct2_zigzag_dz",
}

def load_train(run_dir):
    path = os.path.join(run_dir, "training_log.json")
    if not os.path.isfile(path): return [], []
    logs = json.load(open(path))
    s, l = [], []
    for e in logs:
        if "step" in e and "loss" in e:  # Trainer 的训练日志项
            s.append(e["step"])
            l.append(e["loss"])
    return s, l

def ema(y, alpha=0.1):
    m=None; out=[]
    for v in y:
        m = v if m is None else alpha*v + (1-alpha)*m
        out.append(m)
    return out

plt.figure()
for name, d in RUNS.items():
    s, l = load_train(d)
    if s:
        # 只画 EMA，噪声更小；如果想看原始曲线，把下一行注释打开
        plt.plot(s, l, alpha=0.25, linewidth=1, label=f"{name} (raw)")
        plt.plot(s, ema(l, 0.1), label=f"{name}")

plt.xlabel("step"); plt.ylabel("train_loss")
plt.title("Training Loss (EMA, α=0.1)")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "overlay_train_loss.png"), dpi=160)
print("Saved:", os.path.join(OUTDIR, "overlay_train_loss.png"))
