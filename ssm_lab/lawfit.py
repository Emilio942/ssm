"""Law fitting: which collapse variable explains (1-R2)/R2?"""
import json, sys
import numpy as np
import pandas as pd
sys.path.insert(0, "ssm_lab")
from analyze import load, flat_cols

df = flat_cols(load("results/results.jsonl"))
df = df[df["c_full"].notna()].copy()

# candidate per-run predictors
def row_feats(r):
    E = np.array(r["geometry.atom_energy"])
    P = np.array(r["geometry.dir_persist_by_lag"][:len(E)])
    e0 = max(E[0], 1e-18)
    r2 = E / e0                       # relative energies r^2(d)
    S, alpha, w = r["config.S"], r["config.alpha"], r["window"]
    pbar2 = (E / max(E.sum(), 1e-18) * P ** 2).sum()
    pbar1 = (E / max(E.sum(), 1e-18) * P).sum()          # linear-weighted overlap
    return pd.Series({
        "S": S, "alpha": alpha, "window": w,
        "c_cross": S * w * alpha / 64.0,                 # cross-feature leakage
        "c_self_p2": S * (r2 * P ** 2).sum(),            # self-echo, p^2 weighted
        "c_self_p1": S * (r2 * P).sum(),                 # self-echo, p weighted
        "c_self_full": S * r2.sum(),                     # self-echo, no direction loss
        "c_sum": S * (r2 * (P ** 2 + alpha / 64.0)).sum(),
        "y": (1 - r["probe_r2_w"]) / max(r["probe_r2_w"], 1e-4),
        "mode": r["config.delta_mode"],
    })

X = df.apply(row_feats, axis=1)
X["log_y"] = np.log(X["y"].clip(lower=1e-4))

print("=== correlations with log y (per mode) ===")
cands = ["c_cross", "c_self_p2", "c_self_p1", "c_self_full", "c_sum"]
for m in ["iid", "selective"]:
    d = X[X["mode"] == m]
    print(m, {c: round(np.corrcoef(np.log(d[c].clip(lower=1e-9)), d["log_y"])[0, 1], 3)
              for c in cands})

# 2-parameter power-law fit: log y = b0 + b1 * log(c)
from numpy.polynomial import polynomial as P
print("\n=== power-law fits: y = exp(b0) * c^b1 ===")
for m in ["iid", "selective"]:
    for c in ["c_cross", "c_sum", "c_self_p1"]:
        d = X[(X["mode"] == m)].copy()
        A = np.vstack([np.ones(len(d)), np.log(d[c].clip(lower=1e-9))]).T
        coef, *_ = np.linalg.lstsq(A, d["log_y"], rcond=None)
        pred = A @ coef
        r2 = 1 - ((d["log_y"] - pred) ** 2).sum() / ((d["log_y"] - d["log_y"].mean()) ** 2).sum()
        print(f"{m:9s} {c:12s}: y = {np.exp(coef[0]):.3f} * c^{coef[1]:.3f}   R2={r2:.3f}")

# 2-predictor fit: log y = b0 + b1 log c_cross + b2 log c_self_p1
print("\n=== two-predictor fits ===")
for m in ["iid", "selective"]:
    d = X[X["mode"] == m]
    A = np.vstack([np.ones(len(d)),
                   np.log(d["c_cross"].clip(lower=1e-12)),
                   np.log(d["c_self_p1"].clip(lower=1e-12))]).T
    coef, *_ = np.linalg.lstsq(A, d["log_y"], rcond=None)
    pred = A @ coef
    r2 = 1 - ((d["log_y"] - pred) ** 2).sum() / ((d["log_y"] - d["log_y"].mean()) ** 2).sum()
    print(f"{m}: log y = {coef[0]:.2f} + {coef[1]:.2f} log c_cross + {coef[2]:.2f} log c_self_p1   (fit R2={r2:.3f})")

# collapse figure with fitted law per mode
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, m in zip(axes, ["iid", "selective"]):
    d = X[X["mode"] == m]
    # combined variable from two-predictor fit
    A = np.vstack([np.ones(len(d)),
                   np.log(d["c_cross"].clip(lower=1e-12)),
                   np.log(d["c_self_p1"].clip(lower=1e-12))]).T
    coef, *_ = np.linalg.lstsq(A, d["log_y"], rcond=None)
    chat = np.exp(coef[0]) * (d["c_cross"].clip(lower=1e-12) ** coef[1]) * \
           (d["c_self_p1"].clip(lower=1e-12) ** coef[2])
    sc = ax.scatter(chat, d["y"], c=np.log10(d["alpha"]), s=12, cmap="plasma", alpha=0.8)
    xs = np.logspace(np.log10(max(chat.min(), 1e-6)), np.log10(chat.max()), 100)
    ax.plot(xs, xs, "k--", lw=1, label="y = c_eff")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"fitted $\hat c = %.2f\, c_{cross}^{%.2f}\, c_{self}^{%.2f}$" % tuple(np.exp(coef[0:1]).tolist() + [coef[1], coef[2]]))
    ax.set_ylabel("(1-R²)/R²")
    ax.set_title(m); ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.colorbar(sc, ax=axes, label="log10 alpha")
fig.suptitle("Empirical collapse law")
fig.tight_layout()
fig.savefig("results/figs/collapse_fit.png", dpi=140)
print("\nfigure: results/figs/collapse_fit.png")
