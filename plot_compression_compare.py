# plot_compression_compare.py
import os, json
import matplotlib.pyplot as plt

def read_jsonl(path):
    xs=[]
    with open(path,'r') as f:
        for line in f:
            if line.strip():
                xs.append(json.loads(line))
    return xs

def human_mb(x): return x/1024/1024

def main(warp_stats, excp_dir, out_prefix):
    # WaRP-Q: 读取权重压缩比
    warp = read_jsonl(warp_stats) if warp_stats and os.path.exists(warp_stats) else []
    w_steps = [r["step"] for r in warp if "step" in r]
    w_cr    = [r.get("cr_weight", None) for r in warp]

    # ExCP: 读取每次压缩的字节数 & 计算压缩比（相对一次 full-fp32 权重）
    excp_stats = os.path.join(excp_dir, "compress_stats.jsonl")
    init_path  = os.path.join(excp_dir, "init_weights.pt")
    e = read_jsonl(excp_stats)
    e_steps = [r["step"] for r in e]
    e_bytes = [int(r["bytes"]) for r in e]

    # full-fp32 大小（直接用 init_weights.pt 文件大小）
    full_bytes = os.path.getsize(init_path)

    # 每次事件的 ckpt 压缩比（越大越好）
    e_cr_each = [full_bytes / b for b in e_bytes]

    # 累积体积对比（ExCP 累积 vs baseline 保存 full-fp32 的累积）
    cum_excp = [sum(e_bytes[:i+1]) + full_bytes for i in range(len(e_bytes))]
    cum_full = [full_bytes*(i+1) for i in range(len(e_bytes))]  # baseline: 每次都存全量

    # --- Plot 1: 压缩比曲线 ---
    plt.figure(figsize=(8,5))
    if w_steps:
        plt.plot(w_steps, w_cr, marker='o', label='WaRP-Q CR(weight)')
    plt.plot(e_steps, e_cr_each, marker='s', label='ExCP CR(ckpt per event)')
    plt.xlabel('step'); plt.ylabel('compression ratio (×)')
    plt.title('Compression Ratio')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(out_prefix+'_cr.png')

    # --- Plot 2: 累积体积 ---
    plt.figure(figsize=(8,5))
    plt.plot(e_steps, list(map(human_mb, cum_full)), marker='o', label='Baseline FP32 cumulative (MB)')
    plt.plot(e_steps, list(map(human_mb, cum_excp)), marker='s', label='ExCP cumulative (MB)')
    plt.xlabel('step'); plt.ylabel('size (MB)')
    plt.title('Cumulative Checkpoint Size')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(out_prefix+'_cumulative.png')

    print('[saved]', out_prefix+'_cr.png')
    print('[saved]', out_prefix+'_cumulative.png')

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--warp_stats', type=str, default='./runs/warp_dct2_topk/ckpts/compress_stats.jsonl')
    ap.add_argument('--excp_dir',   type=str, default='./runs/excp_gpt2/excp')
    ap.add_argument('--out',        type=str, default='compare_compress')
    args = ap.parse_args()
    main(args.warp_stats, args.excp_dir, args.out)
