import sys
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
    rr = np.sqrt(E / e0)
    A = float((rr * P).sum())
    return pd.Series({
        "S": r["config.S"], "alpha": r["config.alpha"], "window": r["window"],
        "spec": r["config.spectrum"], "mode": r["config.delta_mode"],
        "A": A,
        "c_coh_cross": r["config.S"] * r["config.alpha"] / 64.0 * A ** 2,
        "c_coh_self": r["config.S"] * A ** 2,
        "R2": r["probe_r2_w"],
    })

X = df.apply(row_feats, axis=1)
X["y"] = (1 - X["R2"]) / X["R2"].clip(lower=1e-4)

for spec in ["homo", "hetero"]:
    for m in ["iid", "selective"]:
        d = X[(X["spec"] == spec) & (X["mode"] == m)]
        # floor estimate: 5th percentile of y at low c
        low = d[d["c_coh_cross"] < d["c_coh_cross"].quantile(0.2)]
        y0 = low["y"].quantile(0.25)
        yy = (d["y"] - y0).clip(lower=1e-4)
        A_ = np.vstack([np.ones(len(d)), np.log(d["c_coh_cross"].clip(lower=1e-12))]).T
        coef, *_ = np.linalg.lstsq(A_, np.log(yy), rcond=None)
        pred = A_ @ coef
        r2 = 1 - ((np.log(yy) - pred) ** 2).sum() / ((np.log(yy) - np.log(yy).mean()) ** 2).sum()
        cstar = np.exp(-coef[0] / coef[1])
        print(f"{spec:7s} {m:9s}: floor_y0={y0:.3f}  y-y0 = {np.exp(coef[0]):.2f} c^ {coef[1]:.2f}"
              f"  fitR2={r2:.3f}  transition c*={cstar:.4f}")
