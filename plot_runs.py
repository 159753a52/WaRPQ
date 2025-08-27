
import os, json, math, argparse
import matplotlib.pyplot as plt

def load_trainer_state(run_dir):
    # 兼容性查找
    ts_path = os.path.join(run_dir, "trainer_state.json")
    if not os.path.exists(ts_path):
        # 兜底：在子目录里找
        for root, _, files in os.walk(run_dir):
            if "trainer_state.json" in files:
                ts_path = os.path.join(root, "trainer_state.json")
                break
    if not os.path.exists(ts_path):
        print(f"[WARN] No trainer_state.json under {run_dir}")
        return [], [], []
    with open(ts_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logs = data.get("log_history", [])
    train_steps, train_loss = [], []
    eval_steps, eval_loss = [], []
    for r in logs:
        s = r.get("step")
        if "loss" in r and s is not None:  # 训练步日志
            train_steps.append(s); train_loss.append(r["loss"])
        if "eval_loss" in r and s is not None:  # 评估日志
            eval_steps.append(s); eval_loss.append(r["eval_loss"])
    return (train_steps, train_loss), (eval_steps, eval_loss)

def plot_overlay(series_map, xlabel, ylabel, title, out_png):
    plt.figure(figsize=(8,5))
    for name, (xs, ys) in series_map.items():
        if xs and ys:
            plt.plot(xs, ys, label=name, linewidth=1.8)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()
    print(f"[SAVE] {out_png}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excp", type=str, default=None, help="ExCP run dir, e.g. ./runs/excp_fixed")
    ap.add_argument("--warpq", type=str, default=None, help="WARP-Q run dir, e.g. ./runs/warp_dct2_fixed")
    ap.add_argument("--fp32", type=str, default=None, help="FP32 run dir, e.g. ./runs/fp32_fixed")
    ap.add_argument("--outdir", type=str, default="./plots")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    runs = []
    if args.excp:  runs.append(("ExCP", args.excp))
    if args.warpq: runs.append(("WARP-Q (DCT2)", args.warpq))
    if args.fp32:  runs.append(("FP32 (baseline)", args.fp32))

    train_map, eval_loss_map, eval_ppl_map = {}, {}, {}
    for name, rd in runs:
        (tr_s, tr_l), (ev_s, ev_l) = load_trainer_state(rd)
        if tr_s:
            train_map[name] = (tr_s, tr_l)
        if ev_s:
            eval_loss_map[name] = (ev_s, ev_l)
            # ppl = exp(eval_loss)
            eval_ppl_map[name] = (ev_s, [math.exp(v) for v in ev_l])

    # 训练 loss
    plot_overlay(train_map, "Step", "Train loss",
                 "Train loss (WikiText-2, GPT-2)", os.path.join(args.outdir, "overlay_train_loss.png"))
    # 验证 loss
    plot_overlay(eval_loss_map, "Step", "Eval loss",
                 "Eval loss (WikiText-2, GPT-2)", os.path.join(args.outdir, "overlay_eval_loss.png"))
    # 验证 ppl
    plot_overlay(eval_ppl_map, "Step", "Perplexity",
                 "Eval perplexity (WikiText-2, GPT-2)", os.path.join(args.outdir, "overlay_eval_ppl.png"))

    # 控制台顺便输出末次指标
    for name, (xs, ys) in eval_loss_map.items():
        last_loss = ys[-1]
        last_ppl = math.exp(last_loss)
        print(f"[{name}] last eval_loss={last_loss:.4f}, ppl≈{last_ppl:.2f}")

if __name__ == "__main__":
    main()