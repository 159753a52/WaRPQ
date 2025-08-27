import json, math, sys, os
import matplotlib.pyplot as plt

def load_loghist(run_dir):
    fp = os.path.join(run_dir, "trainer_state.json")
    with open(fp, "r") as f:
        st = json.load(f)
    return st.get("log_history", [])

def plot(run_dirs):
    fig1 = plt.figure()
    for rd in run_dirs:
        hist = load_loghist(rd)
        steps = [h["step"] for h in hist if "loss" in h or "eval_loss" in h or "train_loss" in h]
        tr_loss = [h.get("loss", None) for h in hist]
        plt.plot([i for i,h in enumerate(tr_loss) if h is not None],
                 [h for h in tr_loss if h is not None],
                 label=f"{os.path.basename(rd)}:train_loss")
    plt.xlabel("log index"); plt.ylabel("train loss"); plt.legend(); plt.title("Train Loss")
    fig1.savefig("train_loss.png", dpi=150)

    fig2 = plt.figure()
    for rd in run_dirs:
        hist = load_loghist(rd)
        eval_steps, eval_loss, ppl = [], [], []
        for h in hist:
            if "eval_loss" in h:
                eval_steps.append(h.get("step", len(eval_steps)))
                ev = h["eval_loss"]
                eval_loss.append(ev)
                ppl.append(math.exp(ev))
        if eval_steps:
            plt.plot(eval_steps, ppl, label=f"{os.path.basename(rd)}")
    plt.xlabel("global step"); plt.ylabel("perplexity (exp(eval_loss))"); plt.legend(); plt.title("Eval PPL")
    fig2.savefig("eval_ppl.png", dpi=150)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/plot_trainer_curves.py RUN_DIR [RUN_DIR ...]")
        sys.exit(1)
    plot(sys.argv[1:])
