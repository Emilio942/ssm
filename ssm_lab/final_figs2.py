"""Final multi-seed figures: paired deltas with SEM, transition table."""
import sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "ssm_lab")
from analyze import load, flat_cols

df = flat_cols(pd.concat([load("results/results.jsonl"), load("results/controls.jsonl"),
                          load("results/phaseB.jsonl")], ignore_index=True))
het = df[df["config.spectrum"]=="hetero"]

piv = het.pivot_table(index=["config.S","config.mean_dt","config.alpha","config.seed"],
                      columns="config.delta_mode", values="probe_r2_w").dropna()
cfg = piv.groupby(level=[0,1,2]).mean()

fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
pairs = [("selective","noise"), ("selective","iid"), ("const","iid")]
labels = ["content-correlation cost\n(selective − noise)",
          "total selectivity cost\n(selective − iid)",
          "pure-variability cost\n(const − iid)"]
for ax, (a_, b_), lab in zip(axes, pairs, labels):
    piv["d"] = piv[a_] - piv[b_]
    g = piv["d"].groupby(level=2).agg(["mean","std","count"])
    sem = g["std"]/np.sqrt(g["count"])
    ax.errorbar(g.index, g["mean"], yerr=2*sem, marker="o", capsize=4, lw=1.5)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel(r"$\alpha = M/N$"); ax.set_ylabel("probe R²_w delta")
    ax.set_title(lab, fontsize=10); ax.grid(alpha=0.3)
fig.suptitle("Selectivity effect decomposition — 3 seeds, 210–282 configs per pair "
             "(bars: ±2 SEM across configs)", fontsize=11)
fig.tight_layout()
fig.savefig("results/figs/mode_decomposition_multiseed.png", dpi=140)
plt.close(fig)

# transition table figure: S* heatmap per mode
import json
fine = df[(df["config.seed"].isin([0,1]))]
fig, axes = plt.subplots(1, 4, figsize=(18, 4.0), sharey=True)
S_list_f = sorted(fine["config.S"].unique())
for ax, mode in zip(axes, ["const","iid","noise","selective"]):
    d = fine[(fine["config.delta_mode"]==mode)]
    M = np.full((3, len(S_list_f)), np.nan)
    for (alpha, dt), g in d.groupby(["config.alpha","config.mean_dt"]):
        gg = g.groupby("config.S")["probe_r2_w"].mean().sort_index()
        i = {5:0,10:1,20:2}.get(alpha)
        if i is None: continue
        for j, s in enumerate(S_list_f):
            if s in gg.index: M[i, j] = gg[s]
    im = ax.imshow(M, origin="lower", aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(S_list_f)))
    ax.set_xticklabels([f"{s:g}" for s in S_list_f], rotation=60, fontsize=6)
    ax.set_yticks([0,1,2]); ax.set_yticklabels([5,10,20], fontsize=8)
    ax.set_title(mode, fontsize=11)
    if mode=="const": ax.set_ylabel(r"$\alpha$   (rows: 5/10/20)")
    ax.set_xlabel("S (fine grid)")
    for i in range(3):
        for j in range(len(S_list_f)):
            if not np.isnan(M[i,j]):
                ax.text(j,i,f"{M[i,j]:.2f}",ha="center",va="center",fontsize=5,
                        color="white" if M[i,j]<0.55 else "black")
fig.suptitle(r"Fine-grid transition scan ($\bar\Delta\in\{0.03,0.1,0.3\}$ averaged; probe R²_w, 2 seeds)")
fig.tight_layout()
fig.savefig("results/figs/fine_transition.png", dpi=140)
plt.close(fig)
print("figures done")
