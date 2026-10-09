"""Scaling experiments: N (state dim) and L (depth) dependence of the collapse law."""
import json, os, sys, time, argparse
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_one import run_config


def sig(cfg):
    keys = ["N", "D", "alpha", "L", "S", "mean_dt", "delta_mode", "dt_cv", "g",
            "spectrum", "seed", "dt_clip", "budget", "steps", "l1", "n_eval"]
    return json.dumps({k: cfg.get(k) for k in keys}, sort_keys=True)


ap = argparse.ArgumentParser()
ap.add_argument("--out", default="results/scaling.jsonl")
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
done = set()
if os.path.exists(args.out):
    for line in open(args.out):
        try:
            done.add(sig(json.loads(line)["config"]))
        except Exception:
            pass

base = dict(batch=32, budget=400_000, n_eval=128, steps=4000, l1=1e-3,
            dt_clip=[0.25, 4.0], dt_cv=0.5, g=1.0, seed=0)
todo = []
# N scaling at fixed transition-region configs (iid mode, hetero spectrum)
for N in [32, 128, 256]:
    for alpha in [2, 10, 50]:
        for dt in [0.01, 0.1]:
            for S in [3e-3, 3e-2]:
                todo.append(dict(base, N=N, D=N, L=2048, alpha=alpha, mean_dt=dt, S=S,
                                 delta_mode="iid", spectrum="hetero"))
# L scaling (depth) at fixed N=64
for L in [256, 1024, 8192]:
    for alpha in [2, 10, 50]:
        for dt in [0.003, 0.1]:
            for S in [3e-3, 3e-2]:
                todo.append(dict(base, N=64, D=64, L=L, alpha=alpha, mean_dt=dt, S=S,
                                 delta_mode="iid", spectrum="hetero"))
todo = [c for c in todo if sig(c) not in done]
if args.limit:
    todo = todo[:args.limit]
print(len(todo), "configs")
t0 = time.time()
for i, cfg in enumerate(todo):
    try:
        res = run_config(cfg, device)
        with open(args.out, "a") as fh:
            fh.write(json.dumps(res) + "\n")
        print(f"[{i+1}/{len(todo)}] N={cfg['N']} L={cfg['L']} a={cfg['alpha']} dt={cfg['mean_dt']} "
              f"S={cfg['S']} probe={res['probe_r2_w']:.3f} ({time.time()-t0:.0f}s)", flush=True)
    except Exception as e:
        print("FAILED", cfg, repr(e), flush=True)
