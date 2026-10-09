"""Phase 1 grid: const-Delta x {homo, hetero} — the theory-critical missing cells.
Also iid x homo fine (to isolate variability from warping)."""
import json, os, sys, time, argparse
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_one import run_config


def sig(cfg):
    keys = ["N", "D", "alpha", "L", "S", "mean_dt", "delta_mode", "dt_cv", "g",
            "spectrum", "seed", "dt_clip", "budget", "steps", "l1", "n_eval"]
    return json.dumps({k: cfg.get(k) for k in keys}, sort_keys=True)


ap = argparse.ArgumentParser()
ap.add_argument("--out", default="results/phase1.jsonl")
args = ap.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
done = set()
if os.path.exists(args.out):
    for line in open(args.out):
        try:
            done.add(sig(json.loads(line)["config"]))
        except Exception:
            pass

base = dict(N=64, D=64, L=2048, batch=32, budget=400_000, n_eval=128,
            steps=4000, l1=1e-3, dt_clip=[0.25, 4.0], dt_cv=0.5, g=1.0, seed=0)
todo = []
for spec in ["homo", "hetero"]:
    for mode in ["const", "iid"]:
        for S in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]:
            for dt in [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
                for alpha in [2, 5, 10, 20, 50]:
                    todo.append(dict(base, S=S, mean_dt=dt, alpha=alpha,
                                     delta_mode=mode, spectrum=spec))
todo = [c for c in todo if sig(c) not in done]
print(len(todo), "configs")
t0 = time.time()
for i, cfg in enumerate(todo):
    try:
        res = run_config(cfg, device)
        with open(args.out, "a") as fh:
            fh.write(json.dumps(res) + "\n")
        if (i + 1) % 50 == 0 or i == len(todo) - 1:
            print(f"[{i+1}/{len(todo)}] {cfg['delta_mode']}/{cfg['spectrum']} "
                  f"S={cfg['S']} dt={cfg['mean_dt']} a={cfg['alpha']} "
                  f"probe={res['probe_r2_w']:.3f} ({(time.time()-t0)/60:.0f}min)", flush=True)
    except Exception as e:
        print("FAILED", cfg, repr(e), flush=True)
print(f"done {(time.time()-t0)/60:.1f} min")
