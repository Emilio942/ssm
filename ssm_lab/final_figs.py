import sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "ssm_lab")
from analyze import load, flat_cols

df = flat_cols(load("results/results.jsonl"))
ctrl = flat_cols(load("results/controls.jsonl"))
all_ = pd.concat([df, ctrl], ignore_index=True)
het = all_[all_["config.spectrum"] == "hetero"]

S_list = sorted(all_["config.S"].unique())
dt_list = sorted(all_["config.mean_dt"].unique())
alphas = sorted(all_["config.alpha"].unique())
modes = ["const", "iid", "noise", "selective"]

# ============ FIG 1: phase diagrams, all 4 modes, alpha=10 ============
fig, axes = plt.subplots(2, 4, figsize=(17, 7.2), sharex=True, sharey=True)
vmin, vmax = 0, 1
for ci, mode in enumerate(modes):
    for ri, (metric, ttl, vmax_) in enumerate([
            ("probe_r2_w", "linear probe R²_w", 1.0),
            ("auc", "SAE detection AUC", 1.0)]):
        ax = axes[ri, ci]
        d = het[(het["config.alpha"] == 10) & (het["config.delta_mode"] == mode)]
        M = np.full((len(dt_list), len(S_list)), np.nan)
        for _, r in d.iterrows():
            M[dt_list.index(r["config.mean_dt"]), S_list.index(r["config.S"])] = r[metric]
        im = ax.imshow(M, origin="lower", aspect="auto", vmin=vmin, vmax=vmax_, cmap="viridis")
        ax.set_xticks(range(len(S_list))); ax.set_xticklabels([f"{s:g}" for s in S_list], rotation=60, fontsize=7)
        ax.set_yticks(range(len(dt_list))); ax.set_yticklabels([f"{v:g}" for v in dt_list], fontsize=7)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=6,
                            color="white" if M[i, j] < 0.55 else "black")
        if ri == 0: ax.set_title(mode, fontsize=11)
        if ci == 0: ax.set_ylabel(f"{ttl}\n$\\bar\\Delta$")
        if ri == 1: ax.set_xlabel("sparsity S")
fig.suptitle("Phase diagrams at $\\alpha = M/N = 10$, $N=64$, hetero spectrum  "
             "(rows: metrics; cols: $\\Delta_t$ modes)", fontsize=12)
fig.tight_layout()
fig.savefig("results/figs/phase_all_modes.png", dpi=140)
plt.close(fig)

# ============ FIG 2: mode decomposition by alpha ============
piv = het.pivot_table(index=["config.S", "config.mean_dt", "config.alpha"],
                      columns="config.delta_mode", values="probe_r2_w").dropna().reset_index()
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
for ax, (a_, b_) in enumerate([("selective", "noise"), ("const", "iid")]):
    piv["d"] = piv[a_] - piv[b_]
    g = piv.groupby("config.alpha")["d"].agg(["mean", "std"])
    axes[ax].errorbar(g.index, g["mean"], yerr=g["std"], marker="o", capsize=4)
    axes[ax].set_xlabel(r"over-completeness $\alpha = M/N$")
    axes[ax].set_ylabel(f"probe R²_w  ({a_} − {b_})")
    axes[ax].set_title(["content-correlation cost\n(selective − noise, moments matched)",
                        "pure-variability cost\n(const − iid)"][ax])
    axes[ax].grid(alpha=0.3)
    axes[ax].axhline(0, color="k", lw=0.6)
fig.suptitle("Decomposing the selectivity effect (paired, hetero spectrum)")
fig.tight_layout()
fig.savefig("results/figs/mode_decomposition.png", dpi=140)
plt.close(fig)

# ============ FIG 3: sweet-spot region (success criteria) ============
# success: probe >= 0.5 AND auc >= 0.7
all2 = all_.copy()
all2["success"] = (all2["probe_r2_w"] >= 0.5) & (all2["auc"] >= 0.7)
fig, axes = plt.subplots(1, 5, figsize=(19, 3.6), sharey=True)
for ai, alpha in enumerate(alphas):
    ax = axes[ai]
    d = all2[(all2["config.alpha"] == alpha) & (all2["config.spectrum"] == "hetero")
             & (all2["config.delta_mode"] == "iid")]
    M = np.full((len(dt_list), len(S_list)), np.nan)
    for _, r in d.iterrows():
        M[dt_list.index(r["config.mean_dt"]), S_list.index(r["config.S"])] = float(r["success"])
    im = ax.imshow(M, origin="lower", aspect="auto", vmin=0, vmax=1, cmap="RdYlGn")
    ax.set_xticks(range(len(S_list))); ax.set_xticklabels([f"{s:g}" for s in S_list], rotation=60, fontsize=7)
    ax.set_yticks(range(len(dt_list))); ax.set_yticklabels([f"{v:g}" for v in dt_list], fontsize=7)
    ax.set_xlabel("S"); ax.set_title(f"alpha={alpha}", fontsize=10)
axes[0].set_ylabel(r"$\bar\Delta$")
fig.suptitle("Sweet-spot region (green): probe R²_w ≥ 0.5 AND SAE AUC ≥ 0.7 — iid mode, hetero", fontsize=11)
fig.tight_layout()
fig.savefig("results/figs/sweet_spot.png", dpi=140)
plt.close(fig)

print("figures written")
# summary counts for the log
for mode in modes:
    d = all2[(all2["config.delta_mode"] == mode) & (all2["config.spectrum"] == "hetero")]
    print(f"{mode:10s}: success fraction over grid = {d['success'].mean():.3f}")
