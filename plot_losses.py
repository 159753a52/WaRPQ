# plot_losses.py
import json, os
import numpy as np
import matplotlib.pyplot as plt

def load(path):
    with open(path, "r") as f:
        logs = json.load(f)
    steps, loss, eval_steps, eval_loss = [], [], [], []
    for e in logs:
        if "loss" in e:
            steps.append(e["step"]); loss.append(e["loss"])
        if "eval_loss" in e:
            eval_steps.append(e["step"]); eval_loss.append(e["eval_loss"])
    return np.array(steps), np.array(loss), np.array(eval_steps), np.array(eval_loss)

def overlay(run_a, run_b, name_a, name_b, out_prefix):
    s0,l0,es0,el0 = load(os.path.join(run_a, "training_log.json"))
    s1,l1,es1,el1 = load(os.path.join(run_b, "training_log.json"))
    plt.figure(); plt.plot(s0, l0, label=name_a); plt.plot(s1, l1, label=name_b); plt.xlabel("step"); plt.ylabel("train loss"); plt.legend(); plt.tight_layout(); plt.savefig(out_prefix+"_train.png", dpi=200)
    plt.figure(); plt.plot(es0, el0, label=name_a+" eval"); plt.plot(es1, el1, label=name_b+" eval"); plt.xlabel("step"); plt.ylabel("eval loss"); plt.legend(); plt.tight_layout(); plt.savefig(out_prefix+"_eval.png", dpi=200)
    x = np.interp(s1, s0, l0) if len(s0)>1 and len(s1)>1 else l0
    xc = np.corrcoef(x[:min(len(x),len(l1))], l1[:min(len(x),len(l1))])[0,1] if len(x)>1 and len(l1)>1 else np.nan
    y = np.interp(es1, es0, el0) if len(es0)>1 and len(es1)>1 else el0
    yc = np.corrcoef(y[:min(len(y),len(el1))], el1[:min(len(y),len(el1))])[0,1] if len(y)>1 and len(el1)>1 else np.nan
    print("train_corr", xc, "eval_corr", yc)

if __name__ == "__main__":
    import sys
    overlay(sys.argv[1], sys.argv[2], "A", "B", sys.argv[3])
