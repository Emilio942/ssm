"""Phase C: seed-aggregated analysis with error bars; verify headline findings."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "ssm_lab")
from analyze import load, flat_cols

def load_all():
    dfs = []
    for f in ["results/results.jsonl", "results/controls.jsonl",
              "results/scaling.jsonl", "results/phaseB.jsonl"]:
        try:
            dfs.append(load(f))
        except FileNotFoundError:
            pass
    return flat_cols(pd.concat(dfs, ignore_index=True))

df = load_all()
print(f"total runs: {len(df)}, seeds: {sorted(df['config.seed'].unique())}")

# ---------- seed reliability of probe_r2_w ----------
g = df.groupby(["config.S","config.mean_dt","config.alpha","config.delta_mode",
                "config.spectrum"])["probe_r2_w"].agg(["mean","std","count"])
multi = g[g["count"] >= 2]
print(f"\nconfigs with >=2 seeds: {len(multi)}")
print(f"per-config seed std of probe R2_w: mean {multi['std'].mean():.4f}, "
      f"median {multi['std'].median():.4f}, p90 {multi['std'].quantile(0.9):.4f}")

# ---------- headline finding 1: paired selective-iid with seed error bars ----------
het = df[df["config.spectrum"]=="hetero"]
piv = het.pivot_table(index=["config.S","config.mean_dt","config.alpha","config.seed"],
                      columns="config.delta_mode", values="probe_r2_w").dropna()
for a_, b_ in [("selective","iid"), ("selective","noise"), ("const","iid"), ("noise","iid")]:
    if a_ in piv and b_ in piv:
        d = (piv[a_]-piv[b_]).dropna()
        # per-seed means then across-seed error
        per_seed = d.groupby(level=3).mean()
        print(f"{a_:9s}-{b_:9s}: {d.mean():+.4f}  (per-seed: "
              + ", ".join(f"s{k}={v:+.4f}" for k,v in per_seed.items())
              + f"; SEM {d.groupby(level=3).mean().std()/max(np.sqrt(len(per_seed)),1):.4f})")

# ---------- headline law: refit with all seeds, per mode ----------
rows = []
for _, r in df.iterrows():
    E = r["geometry.atom_energy"]; P = r["geometry.dir_persist_by_lag"]
    if not isinstance(E, list) or len(E) < 2:
        continue
    E = np.array(E); P = np.array(P[:len(E)])
    A = float((np.sqrt(E/E[0]) * P).sum())
    rows.append((r["config.S"], r["config.alpha"], r["config.N"], r["window"],
                 r["probe_r2_w"], r["config.delta_mode"], r["config.spectrum"]))
L = pd.DataFrame(rows, columns=["S","alpha","N","window","R2","mode","spec"])
L = L[L["spec"]=="hetero"]
L["A2"] = L["window"]**2   # approx: A ~ tau_eff*pbar, use measured below
# compute A properly needs persist; do coarse version with window
L["c"] = L["S"] * (L["alpha"]/L["N"]) * L["A2"] * 1.0
L["y"] = (1-L["R2"])/L["R2"].clip(lower=1e-4)
for m in ["iid","selective"]:
    d = L[L["mode"]==m]
    y0 = d[d["c"]<d["c"].quantile(0.2)]["y"].quantile(0.25)
    yy = (d["y"]-y0).clip(lower=1e-4)
    Amat = np.vstack([np.ones(len(d)), np.log(d["c"].clip(lower=1e-12))]).T
    coef,*_ = np.linalg.lstsq(Amat, np.log(yy), rcond=None)
    print(f"law[{m}]: y-y0={np.exp(coef[0]):.1f} c^{coef[1]:.2f} (y0={y0:.2f}, n={len(d)}) "
          f"-> R2=0.5 at c*={(1-y0)/np.exp(coef[0])**(1/coef[1]) if coef[1]!=0 else float('nan'):.4f}")
