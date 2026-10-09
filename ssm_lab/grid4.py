"""Phase B: (1) fine transition scan around the collapse boundary,
(2) budget robustness (400k vs 1.2M samples), (3) N-scaling replication seeds."""
import json, os, sys, time, argparse
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_one import run_config


def sig(cfg):
    keys = ["N", "D", "alpha", "L", "S", "mean_dt", "delta_mode", "dt_cv", "g",
            "spectrum", "seed", "dt_clip", "budget", "steps", "l1", "n_eval"]
    return json.dumps({k: cfg.get(k) for k in keys}, sort_keys=True)


ap = argparse.ArgumentParser()
ap.add_argument("--out", default="results/phaseB.jsonl")
ap.add_argument("--part", default="all", choices=["fine", "budget", "nscaling", "all"])
args = ap.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
done = set()
if os.path.exists(args.out):
    for line in open(args.out):
        try:
            done.add(sig(json.loads(line)["config"]))
        except Exception:
            pass

base = dict(N=64, D=64, L=2048, batch=32, n_eval=128, steps=4000, l1=1e-3,
            dt_clip=[0.25, 4.0], dt_cv=0.5, g=1.0, spectrum="hetero")
todo = []

# (1) fine S-grid around the transition, iid + selective
if args.part in ("fine", "all"):
    for mode in ["iid", "selective"]:
        for alpha in [5, 10, 20]:
            for dt in [0.03, 0.1, 0.3]:
                for S in [0.004, 0.006, 0.008, 0.010, 0.013, 0.017, 0.022, 0.028, 0.036]:
                    for seed in [0, 1]:
                        todo.append(dict(base, S=S, mean_dt=dt, alpha=alpha,
                                         delta_mode=mode, seed=seed))

# (2) budget robustness: same configs at 3x budget
if args.part in ("budget", "all"):
    for mode in ["iid", "selective", "noise", "const"]:
        for alpha in [5, 20]:
            for dt in [0.01, 0.1]:
                for S in [1e-3, 1e-2]:
                    todo.append(dict(base, S=S, mean_dt=dt, alpha=alpha,
                                     delta_mode=mode, seed=0, budget=1_200_000))

# (3) N-scaling with extra seeds
if args.part in ("nscaling", "all"):
    for N in [32, 128, 256]:
        for alpha in [2, 10, 50]:
            for dt in [0.01, 0.1]:
                for S in [3e-3, 3e-2]:
                    for seed in [1, 2]:
                        todo.append(dict(base, N=N, D=N, L=2048, alpha=alpha, mean_dt=dt,
                                         S=S, delta_mode="iid", seed=seed))

todo = [c for c in todo if sig(c) not in done]
print(len(todo), "configs to run")
t0 = time.time()
for i, cfg in enumerate(todo):
    try:
        res = run_config(cfg, device)
        with open(args.out, "a") as fh:
            fh.write(json.dumps(res) + "\n")
        if (i + 1) % 25 == 0 or i == len(todo) - 1:
            print(f"[{i+1}/{len(todo)}] {cfg['delta_mode']} S={cfg['S']} dt={cfg['mean_dt']} "
                  f"a={cfg['alpha']} N={cfg['N']} probe={res['probe_r2_w']:.3f} "
                  f"({(time.time()-t0)/60:.0f}min)", flush=True)
    except Exception as e:
        print("FAILED", cfg, repr(e), flush=True)
print(f"done in {(time.time()-t0)/60:.1f} min")
