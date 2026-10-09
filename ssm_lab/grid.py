"""Grid sweep driver: generates configs, runs them sequentially, appends to results file.
Skips configs already present (matching on config signature)."""
import json, os, sys, time, itertools, argparse
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_one import run_config


def config_signature(cfg):
    keys = ["N", "D", "alpha", "L", "S", "mean_dt", "delta_mode", "dt_cv", "g",
            "spectrum", "seed", "dt_clip", "budget", "steps", "l1", "n_eval"]
    return json.dumps({k: cfg.get(k) for k in keys}, sort_keys=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/results.jsonl")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=1)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as fh:
            for line in fh:
                try:
                    done.add(config_signature(json.loads(line)["config"]))
                except Exception:
                    pass

    base = dict(N=64, D=64, L=2048, batch=32, budget=400_000, n_eval=128,
                steps=4000, l1=1e-3, dt_clip=[0.25, 4.0])

    grid = []
    for S in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]:
        for dt in [3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]:
            for alpha in [2, 5, 10, 20, 50]:
                for mode in ["iid", "selective"]:
                    for spec in ["hetero", "homo"]:
                        # NOTE: dt_cv identical across modes => moment-matched marginals;
                        # iid additionally clipped to [0.25,4]x mean.
                        grid.append(dict(base, S=S, mean_dt=dt, alpha=alpha,
                                         delta_mode=mode, dt_cv=0.5,
                                         g=1.0, spectrum=spec))

    todo = []
    for g in grid:
        for s in range(args.seeds):
            cfg = dict(g, seed=s)
            if config_signature(cfg) not in done:
                todo.append(cfg)
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(todo)} configs to run (device={device})")

    t_start = time.time()
    for i, cfg in enumerate(todo):
        t0 = time.time()
        try:
            res = run_config(cfg, device)
            with open(args.out, "a") as fh:
                fh.write(json.dumps(res) + "\n")
            print(f"[{i+1}/{len(todo)}] S={cfg['S']} dt={cfg['mean_dt']} a={cfg['alpha']} "
                  f"{cfg['delta_mode']}/{cfg['spectrum']} "
                  f"probe={res.get('probe_r2_w', -1):.3f} auc={res.get('auc', -1):.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"[{i+1}/{len(todo)}] FAILED {cfg}: {e}", flush=True)
    print(f"done in {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
