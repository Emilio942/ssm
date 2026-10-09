"""Final validation: (1) empirical S* crossings from fine scan vs law prediction,
(2) budget robustness, (3) law constants with proper A on 3 seeds."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "ssm_lab")
from analyze import load, flat_cols

df = flat_cols(pd.concat([load("results/results.jsonl"), load("results/controls.jsonl"),
                          load("results/scaling.jsonl"), load("results/phaseB.jsonl")],
                         ignore_index=True))
het = df[df["config.spectrum"] == "hetero"].copy()
N = 64

def A_of(r):
    E = r["geometry.atom_energy"]; P = r["geometry.dir_persist_by_lag"]
    if not isinstance(E, list) or len(E) < 2: return np.nan
    E = np.array(E); P = np.array(P[:len(E)])
    return float((np.sqrt(E / E[0]) * P).sum())

het["A"] = het.apply(A_of, axis=1)
het["c"] = het["config.S"] * (het["config.alpha"] / N) * het["A"]**2

# ---------- (1) S* crossing from fine scan ----------
print("=== empirical transition S* (probe=0.5, linear interp on fine S-grid, 2 seeds) ===")
print(f"{'mode':10s} {'alpha':>5s} {'dt':>5s} | {'S*_emp':>8s} {'c*_emp':>8s} | {'A':>5s}")
cstars = {}
for (mode, alpha, dt), g in het[het["config.seed"].isin([0,1])].groupby(
        ["config.delta_mode","config.alpha","config.mean_dt"]):
    g = g.groupby("config.S")["probe_r2_w"].mean().sort_index()
    if g.max() < 0.5 or g.min() > 0.5 or len(g) < 5:
        continue
    svals, pvals = g.index.values, g.values
    # first crossing from above 0.5 to below
    Sstar = np.nan
    for i in range(len(pvals)-1):
        if pvals[i] >= 0.5 > pvals[i+1]:
            frac = (pvals[i]-0.5)/(pvals[i]-pvals[i+1])
            Sstar = svals[i] + frac*(svals[i+1]-svals[i])
            break
    if np.isnan(Sstar): continue
    Amid = het[(het["config.delta_mode"]==mode)&(het["config.alpha"]==alpha)&
               (het["config.mean_dt"]==dt)]["A"].mean()
    cstar = Sstar * alpha/N * Amid**2
    cstars.setdefault(mode, []).append(cstar)
    print(f"{mode:10s} {alpha:5d} {dt:5.2f} | {Sstar:8.4f} {cstar:8.4f} | {Amid:5.2f}")
for m, v in cstars.items():
    print(f"{m}: c* = {np.mean(v):.4f} ± {np.std(v):.4f} (n={len(v)})")

# ---------- (2) budget robustness ----------
print("\n=== budget robustness (400k vs 1.2M samples, seed 0) ===")
sub = het[het["config.seed"]==0]
big = sub[sub["config.budget"]==1_200_000]
pairs = []
for _, r in big.iterrows():
    m = sub[(sub["config.budget"]==400_000)&(sub["config.S"]==r["config.S"])&
            (sub["config.mean_dt"]==r["config.mean_dt"])&(sub["config.alpha"]==r["config.alpha"])&
            (sub["config.delta_mode"]==r["config.delta_mode"])]
    if len(m):
        pairs.append((m["probe_r2_w"].iloc[0], r["probe_r2_w"]))
p = np.array(pairs)
if len(p):
    d = p[:,1]-p[:,0]
    print(f"n={len(p)}  probe(1.2M) - probe(400k): mean {d.mean():+.4f}, "
          f"std {d.std():.4f}, max |d| {np.abs(d).max():.4f}")

# ---------- (3) law constants, all seeds, proper A ----------
print("\n=== collapse law (all seeds, proper A) ===")
L = het[het["A"].notna() & (het["config.L"]==2048) & (het["config.budget"]==400_000)].copy()
L["y"] = (1-L["probe_r2_w"])/L["probe_r2_w"].clip(lower=1e-4)
for m in ["iid","selective","noise","const"]:
    d = L[L["config.delta_mode"]==m]
    if len(d) < 30: continue
    y0 = d[d["c"]<d["c"].quantile(0.2)]["y"].quantile(0.25)
    yy = (d["y"]-y0).clip(lower=1e-4)
    A_ = np.vstack([np.ones(len(d)), np.log(d["c"].clip(lower=1e-12))]).T
    coef,*_ = np.linalg.lstsq(A_, np.log(yy), rcond=None)
    pred = A_ @ coef
    r2 = 1-((np.log(yy)-pred)**2).sum()/((np.log(yy)-np.log(yy).mean())**2).sum()
    cstar = np.exp(-coef[0]/coef[1])
    print(f"{m:10s}: y-y0 = {np.exp(coef[0]):5.1f} c^{coef[1]:.2f}  (y0={y0:.2f}, fitR2={r2:.2f}, "
          f"n={len(d)})  c*(y=1) = {cstar:.4f}")
