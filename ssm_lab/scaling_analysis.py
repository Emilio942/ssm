import sys
import numpy as np, pandas as pd
sys.path.insert(0, "ssm_lab")
from analyze import load, flat_cols

sc = flat_cols(load("results/scaling.jsonl"))

# ---- L scaling: probe at fixed config vs L ----
print("=== L scaling (N=64, iid/hetero) ===")
d = sc[sc["config.N"]==64]
piv = d.pivot_table(index=["config.alpha","config.mean_dt","config.S"], columns="config.L", values="probe_r2_w")
print(piv.round(3).to_string())

# ---- N scaling ----
print("\n=== N scaling (L=2048, iid/hetero): probe R2_w by N ===")
d = sc[sc["config.L"]==2048]
piv = d.pivot_table(index=["config.alpha","config.mean_dt","config.S"], columns="config.N", values="probe_r2_w")
print(piv.round(3).to_string())

# ---- does y follow S*(alpha/N)*A^2 across N? ----
print("\n=== coherent-law check across N (should be approx constant y at same c) ===")
d = sc[sc["config.L"]==2048].copy()
def _A(r):
    E = r["geometry.atom_energy"]; P = r["geometry.dir_persist_by_lag"]
    return float((np.sqrt(np.array(E)) * np.array(P[:len(E)])).sum())
d["A"] = d.apply(_A, axis=1)
d["c_coh"] = d["config.S"] * (d["config.alpha"]/d["config.N"]) * d["A"]**2
d["y"] = (1-d["probe_r2_w"])/d["probe_r2_w"].clip(lower=1e-3)
d["Nn"] = d["config.N"].astype(int)
for (alpha, dt, S), g in d.groupby(["config.alpha","config.mean_dt","config.S"]):
    row = " ".join(f"N{n}: y={y:.2f} c={c:.3f}" for n, y, c in
                   zip(g["Nn"], g["y"], g["c_coh"]))
    print(f"a={alpha} dt={dt} S={S}: {row}")
