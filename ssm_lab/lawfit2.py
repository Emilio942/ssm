import json, sys
import numpy as np
import pandas as pd
sys.path.insert(0, "ssm_lab")
from analyze import load, flat_cols

df = flat_cols(load("results/results.jsonl"))
df = df[df["c_full"].notna()].copy()

def row_feats(r):
    E = np.array(r["geometry.atom_energy"])
    P = np.array(r["geometry.dir_persist_by_lag"][:len(E)])
    e0 = max(E[0], 1e-18)
    rq = E / e0                                  # r^2(d)
    rr = np.sqrt(rq)
    S, alpha, w = r["config.S"], r["config.alpha"], r["window"]
    A = float((rr * P).sum())                    # coherent aligned sum over lags
    return pd.Series({
        "S": S, "alpha": alpha, "window": w,
        "c_cross": S * w * alpha / 64.0,
        "c_coh_cross": S * alpha / 64.0 * A ** 2,     # coherent cross interference
        "c_coh_self": S * A ** 2,                     # coherent self echo
        "y": (1 - r["probe_r2_w"]) / max(r["probe_r2_w"], 1e-4),
        "mode": r["config.delta_mode"],
    })

X = df.apply(row_feats, axis=1)
X["log_y"] = np.log(X["y"].clip(lower=1e-4))
for m in ["iid", "selective"]:
    d = X[X["mode"] == m]
    print(m, {c: round(np.corrcoef(np.log(d[c].clip(lower=1e-12)), d["log_y"])[0, 1], 3)
              for c in ["c_cross", "c_coh_cross", "c_coh_self"]})

print("\n=== power laws ===")
for m in ["iid", "selective"]:
    for c in ["c_cross", "c_coh_cross", "c_coh_self"]:
        d = X[X["mode"] == m]
        A_ = np.vstack([np.ones(len(d)), np.log(d[c].clip(lower=1e-12))]).T
        coef, *_ = np.linalg.lstsq(A_, d["log_y"], rcond=None)
        pred = A_ @ coef
        r2 = 1 - ((d["log_y"] - pred) ** 2).sum() / ((d["log_y"] - d["log_y"].mean()) ** 2).sum()
        cstar = np.exp(-coef[0] / coef[1])
        print(f"{m:9s} {c:12s}: y = {np.exp(coef[0]):6.2f} c^{coef[1]:.2f}  R2={r2:.3f}  y=1 at c*={cstar:.4f}")

print("\n=== two-predictor (c_coh_cross + c_coh_self) ===")
for m in ["iid", "selective"]:
    d = X[X["mode"] == m]
    A_ = np.vstack([np.ones(len(d)), np.log(d["c_coh_cross"].clip(lower=1e-12)),
                    np.log(d["c_coh_self"].clip(lower=1e-12))]).T
    coef, *_ = np.linalg.lstsq(A_, d["log_y"], rcond=None)
    pred = A_ @ coef
    r2 = 1 - ((d["log_y"] - pred) ** 2).sum() / ((d["log_y"] - d["log_y"].mean()) ** 2).sum()
    print(f"{m}: y = {np.exp(coef[0]):.2f} ccoh_cross^{coef[1]:.2f} ccoh_self^{coef[2]:.2f} (fitR2={r2:.3f})")
