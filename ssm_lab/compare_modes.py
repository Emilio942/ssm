import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "ssm_lab")
from analyze import load, flat_cols

df = flat_cols(load("results/results.jsonl"))
metrics = ["probe_r2_w", "probe_r2_tr", "auc", "extract_corr", "nre", "ctim", "window"]
piv = df.pivot_table(index=["config.S","config.mean_dt","config.alpha","config.spectrum"],
                     columns="config.delta_mode", values=metrics)
print("=== iid vs selective, paired differences (mean over grid) ===")
for m in metrics:
    if (m, "iid") in piv.columns and (m, "selective") in piv.columns:
        dlt = (piv[(m,"selective")] - piv[(m,"iid")]).dropna()
        print(f"{m:14s}: delta = {dlt.mean():+.4f}  (std {dlt.std():.3f})")

# homo vs hetero (direction drift effect) within iid
print("\n=== homo vs hetero (iid mode): direction-warping cost ===")
p2 = df[df["config.delta_mode"]=="iid"].pivot_table(
    index=["config.S","config.mean_dt","config.alpha"],
    columns="config.spectrum", values="probe_r2_w").dropna()
d2 = p2["hetero"] - p2["homo"]
print(f"probe_r2_w delta (hetero-homo): {d2.mean():+.4f} (std {d2.std():.3f}, n={len(d2)})")

# SAE vs probe gap: how much of linear capacity does the SAE capture?
print("\n=== SAE extraction vs linear probe (ratio), by mode ===")
df["sae_ratio"] = df["extract_corr"] / df["probe_r2_w"].clip(lower=1e-3)
print(df.groupby(["config.delta_mode","config.spectrum"])["sae_ratio"].mean().round(3))
