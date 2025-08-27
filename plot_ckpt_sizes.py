import os, re, sys
import matplotlib.pyplot as plt

def scan_sizes(ckpt_dir):
    pat = re.compile(r".*\.(7z|zip|rar|pt|bin)$", re.I)
    files = sorted([f for f in os.listdir(ckpt_dir) if pat.match(f)])
    steps, sizes = [], []
    for f in files:
        fp = os.path.join(ckpt_dir, f)
        steps.append(int(re.findall(r"\d+", f)[-1]) if re.findall(r"\d+", f) else len(steps))
        sizes.append(os.path.getsize(fp) / (1024*1024.0))
    return steps, sizes

def plot(ckpt_dirs):
    plt.figure()
    for d in ckpt_dirs:
        if not os.path.isdir(d):
            continue
        steps, sizes = scan_sizes(d)
        if steps:
            plt.plot(steps, sizes, marker="o", label=os.path.basename(d))
    plt.xlabel("step (parsed from filename)"); plt.ylabel("size (MB)")
    plt.title("Compressed Checkpoint Size")
    plt.legend()
    plt.savefig("ckpt_sizes.png", dpi=150)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/plot_ckpt_sizes.py CKPT_DIR [CKPT_DIR ...]")
        sys.exit(1)
    plot(sys.argv[1:])
