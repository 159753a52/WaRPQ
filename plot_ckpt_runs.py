# plot_ckpt_runs.py
import os, json, argparse, math, glob
from collections import defaultdict, OrderedDict
from typing import List, Dict, Any
import matplotlib.pyplot as plt

def read_json(path):
    with open(path, "r") as f:
        return json.load(f)

def read_jsonl(path):
    out=[]
    with open(path,"r") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try: out.append(json.loads(line))
            except: pass
    return out

def moving_avg(xs, ys, k=5):
    if k<=1 or len(xs)<=k: return xs, ys
    s=[]; t=[]
    half=k//2
    for i in range(len(xs)):
        lo=max(0,i-half); hi=min(len(xs),i+half+1)
        t.append(xs[i]); s.append(sum(ys[lo:hi])/(hi-lo))
    return t, s

def nearest_leq(sorted_steps, target):
    import bisect
    i = bisect.bisect_right(sorted_steps, target) - 1
    if i < 0: return None
    return sorted_steps[i]

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def load_run(run_dir: str) -> Dict[str, Any]:
    d={"run": run_dir, "label": os.path.basename(run_dir)}
    tlog=os.path.join(run_dir,"training_log.json")
    cs=os.path.join(run_dir,"ckpts","compress_stats.jsonl")
    d["train"] = read_json(tlog) if os.path.exists(tlog) else []
    d["comp"]  = read_jsonl(cs)   if os.path.exists(cs)   else []
    return d

def extract_series_train(run):
    tl=run["train"]
    tr_steps=[]; tr_loss=[]
    ev_steps=[]; ev_loss=[]
    for e in tl:
        if "loss" in e:
            tr_steps.append(int(e.get("step",0))); tr_loss.append(float(e["loss"]))
        if "eval_loss" in e:
            ev_steps.append(int(e.get("step",0))); ev_loss.append(float(e["eval_loss"]))
    return (tr_steps,tr_loss),(ev_steps,ev_loss)

def extract_series_comp(run):
    cs=run["comp"]
    steps=[]; bpp=[]; cr=[]; snr=[]; subbands_list=[]
    for e in cs:
        steps.append(int(e.get("step",0)))
        bpp.append(float(e.get("bpp", float("nan"))))
        cr.append(float(e.get("cr_weight", float("nan"))))
        if "snr_db" in e: snr.append(float(e["snr_db"]))
        else: snr.append(float("nan"))
        subbands_list.append(e.get("subbands", None))
    return steps,bpp,cr,snr,subbands_list

def align_bpp_with_eval(run):
    (ts,_),(es,ev)=extract_series_train(run)
    comp_steps,bpp,cr,_,_=extract_series_comp(run)
    if not es or not comp_steps: return [],[],[]
    es_sorted=sorted(set(es))
    x=[]; y=[]; r=[]
    for s,b,c in zip(comp_steps,bpp,cr):
        s2=nearest_leq(es_sorted,s)
        if s2 is None: continue
        # 找对应 eval_loss
        # 取该 step 的最后一次 eval_loss
        vals=[(ss,ll) for ss,ll in zip(es,ev) if ss==s2]
        if not vals: 
            # 兜底最近邻
            import numpy as np
            idx=min(range(len(es)), key=lambda i: abs(es[i]-s))
            y.append(ev[idx]); x.append(b); r.append((s,c))
        else:
            y.append(vals[-1][1]); x.append(b); r.append((s,c))
    return x,y,r

def aggregate_subbands(run):
    steps,bpp,cr,snr,subs=extract_series_comp(run)
    agg=defaultdict(lambda: [0.0,0.0,0])  # label -> sum_nmse, sum_var? here nmse not stored; we stored [nmse,snr] in value
    have=False
    for d in subs:
        if not isinstance(d, dict): continue
        have=True
        for k,v in d.items():
            # v = [nmse, snr_db]
            if isinstance(v, list) and len(v)>=2:
                nmse=float(v[0]); sdb=float(v[1])
                a=agg[k]; a[0]+=nmse; a[1]+=sdb; a[2]+=1; agg[k]=a
    if not have: return None
    # 计算均值
    out={}
    for k,(snmse, ssnr, n) in agg.items():
        if n==0: continue
        out[k]={"nmse": snmse/max(n,1), "snr_db": ssnr/max(n,1)}
    # 按层级排序：A, L1_LH/HL/HH, L2_*, ...
    def keyfun(s):
        if s=="A": return (0,0)
        if s.startswith("L"):
            try:
                parts=s.split("_")
                lvl=int(parts[0][1:])
                band=parts[1] if len(parts)>1 else ""
                order={"LH":0,"HL":1,"HH":2}.get(band,3)
                return (lvl, order)
            except: pass
        return (999, 999)
    out_sorted=OrderedDict(sorted(out.items(), key=lambda kv: keyfun(kv[0])))
    return out_sorted

def plot_train_eval(runs, outdir, smooth=7):
    ensure_dir(outdir)
    plt.figure(); 
    for r in runs:
        (ts,tl),(es,ev)=extract_series_train(r)
        if ts:
            xs,ys=moving_avg(ts,tl,k=smooth)
            plt.plot(xs,ys,label=r["label"])
    plt.xlabel("Step"); plt.ylabel("Train loss"); plt.title("Train loss over steps"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir,"train_loss.png"), dpi=180); plt.close()

    plt.figure();
    for r in runs:
        (ts,tl),(es,ev)=extract_series_train(r)
        if es:
            plt.plot(es,ev,marker="o",markersize=3,linewidth=1,label=r["label"])
    plt.xlabel("Step"); plt.ylabel("Eval loss"); plt.title("Eval loss over steps"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir,"eval_loss.png"), dpi=180); plt.close()

def plot_rate_curves(runs, outdir):
    ensure_dir(outdir)
    plt.figure();
    for r in runs:
        s,b,c,_,_=extract_series_comp(r)
        if s:
            plt.plot(s,b,marker="o",linewidth=1,label=r["label"])
    plt.xlabel("Step"); plt.ylabel("bpp"); plt.title("Bits-per-param @ checkpoint"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir,"bpp_step.png"), dpi=180); plt.close()

    plt.figure();
    for r in runs:
        s,b,c,_,_=extract_series_comp(r)
        if s:
            plt.plot(s,c,marker="o",linewidth=1,label=r["label"])
    plt.xlabel("Step"); plt.ylabel("Estimated CR (×)"); plt.title("Compression ratio @ checkpoint"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir,"cr_step.png"), dpi=180); plt.close()

def plot_rate_distortion(runs, outdir):
    ensure_dir(outdir)
    plt.figure()
    for r in runs:
        xb,yl,_ = align_bpp_with_eval(r)
        if xb:
            plt.plot(xb,yl,marker="o",linewidth=1,label=r["label"])
    plt.xlabel("bpp"); plt.ylabel("Eval loss"); plt.title("Rate–Distortion (bpp vs eval loss)"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir,"rate_distortion.png"), dpi=180); plt.close()

def plot_subband_bars(runs, outdir):
    ensure_dir(outdir)
    for r in runs:
        agg=aggregate_subbands(r)
        if not agg: continue
        labels=list(agg.keys())
        snr=[agg[k]["snr_db"] for k in labels]
        nmse=[agg[k]["nmse"] for k in labels]
        plt.figure(figsize=(max(6,len(labels)*0.8),3.2))
        plt.bar(range(len(labels)), snr)
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
        plt.ylabel("Avg SNR (dB)"); plt.title(f"Subband SNR – {r['label']}"); plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"subband_snr_bar_{r['label']}.png"), dpi=180); plt.close()

        plt.figure(figsize=(max(6,len(labels)*0.8),3.2))
        plt.bar(range(len(labels)), nmse)
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
        plt.ylabel("Avg NMSE"); plt.title(f"Subband NMSE – {r['label']}"); plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"subband_nmse_bar_{r['label']}.png"), dpi=180); plt.close()

def export_summary(runs, outdir):
    ensure_dir(outdir)
    rows=[]
    for r in runs:
        s,b,c,_,_=extract_series_comp(r)
        _,_,(pairs)=align_bpp_with_eval(r)
        final_eval=None
        (ts,tl),(es,ev)=extract_series_train(r)
        if ev: final_eval=ev[-1]
        row={
            "run": r["run"],
            "label": r["label"],
            "bpp_mean": (sum([x for x in b if not math.isnan(x)]) / max(1, len([x for x in b if not math.isnan(x)]))) if b else float("nan"),
            "bpp_last": b[-1] if b else float("nan"),
            "cr_last": c[-1] if c else float("nan"),
            "eval_last": final_eval if final_eval is not None else float("nan"),
            "num_ckpts": len(s)
        }
        rows.append(row)
    # 写 CSV/JSON（无 pandas 也能导出）
    jpath=os.path.join(outdir,"summary.json")
    with open(jpath,"w") as f: json.dump(rows, f, indent=2)
    try:
        import pandas as pd
        pd.DataFrame(rows).to_csv(os.path.join(outdir,"summary.csv"), index=False)
    except Exception:
        # 简易 CSV
        hdr=list(rows[0].keys()) if rows else []
        with open(os.path.join(outdir,"summary.csv"),"w") as f:
            if hdr: f.write(",".join(hdr)+"\n")
            for r in rows:
                f.write(",".join(str(r[k]) for k in hdr)+"\n")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--out", type=str, default="plots")
    ap.add_argument("--smooth", type=int, default=7)
    args=ap.parse_args()

    runs=[]
    for i, rd in enumerate(args.runs):
        r=load_run(rd)
        if args.labels and i<len(args.labels): r["label"]=args.labels[i]
        runs.append(r)

    os.makedirs(args.out, exist_ok=True)
    plot_train_eval(runs, args.out, smooth=args.smooth)
    plot_rate_curves(runs, args.out)
    plot_rate_distortion(runs, args.out)
    plot_subband_bars(runs, args.out)
    export_summary(runs, args.out)
    print(f"[OK] Figures and summary written to: {args.out}")

if __name__=="__main__":
    main()
