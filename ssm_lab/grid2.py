"""Focused control-grid: modes noise/const/selective(iid ref) x S x dt, hetero spectrum."""
import json, os, sys, time, argparse
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_one import run_config

def sig(cfg):
    keys = ["N","D","alpha","L","S","mean_dt","delta_mode","dt_cv","g",
            "spectrum","seed","dt_clip","budget","steps","l1","n_eval"]
    return json.dumps({k: cfg.get(k) for k in keys}, sort_keys=True)

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="results/controls.jsonl")
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
done = set()
if os.path.exists(args.out):
    for line in open(args.out):
        try: done.add(sig(json.loads(line)["config"]))
        except Exception: pass

base = dict(N=64, D=64, L=2048, batch=32, budget=400_000, n_eval=128,
            steps=4000, l1=1e-3, dt_clip=[0.25, 4.0], spectrum="hetero")
todo = []
for mode in ["noise", "const"]:
    for S in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]:
        for dt in [3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]:
            for alpha in [2, 5, 10, 20, 50]:
                todo.append(dict(base, S=S, mean_dt=dt, alpha=alpha,
                                 delta_mode=mode, dt_cv=0.5, g=1.0, seed=0))
todo = [c for c in todo if sig(c) not in done]
if args.limit: todo = todo[:args.limit]
print(len(todo), "configs")
t0 = time.time()
for i, cfg in enumerate(todo):
    try:
        res = run_config(cfg, device)
        with open(args.out, "a") as fh: fh.write(json.dumps(res) + "\n")
        if (i+1) % 20 == 0 or i == len(todo)-1:
            print(f"[{i+1}/{len(todo)}] {cfg['delta_mode']} S={cfg['S']} dt={cfg['mean_dt']} "
                  f"a={cfg['alpha']} probe={res['probe_r2_w']:.3f} ({time.time()-t0:.0f}s)", flush=True)
    except Exception as e:
        print("FAILED", cfg, e, flush=True)
