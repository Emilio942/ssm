"""Analysis of sweep results: phase diagrams, boundary extraction, iid-vs-selective."""
import json, os, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    df = pd.json_normalize(rows)
    return df


def flat_cols(df):
    df = df.copy()
    df["ctim"] = df["geometry.ctim_mean_abs_cos"]
    df["window"] = df["config.eff_window"]
    df["alive"] = df["config.alive_eff"]
    df["sinr"] = df["geometry.sinr_proxy"]

    def persist_calc(v):
        if not isinstance(v, list) or len(v) < 2:
            return pd.Series({"persist16": np.nan, "persist128": np.nan, "pbar2": np.nan})
        return pd.Series({"persist16": v[min(4, len(v) - 1)], "persist128": v[-1],
                          "pbar2": np.nan})
    p = df["geometry.dir_persist_by_lag"].apply(persist_calc)
    df = pd.concat([df, p], axis=1)

    # full collapse predictor:  c = S * tau_eff * (alpha/N + pbar^2)
    # pbar^2 = sum_d E(d) p(d)^2 / sum_d E(d)   (energy-weighted self-overlap)
    def collapse(row):
        E = row["geometry.atom_energy"]
        P = row["geometry.dir_persist_by_lag"]
        if not isinstance(E, list) or not isinstance(P, list) or len(E) < 2:
            return np.nan
        E = np.array(E[:len(P)]); P = np.array(P[:len(E)])
        w = E / max(E.sum(), 1e-18)
        pbar2 = float((w * P ** 2).sum())
        return row["config.S"] * row["window"] * (row["config.alpha"] / 64.0 + pbar2)
    df["c_full"] = df.apply(collapse, axis=1)
    return df


def phase_axis(ax, d, value_col, title, S_list, dt_list, alpha, spec, mode,
               vmin=0, vmax=1, cmap="viridis", fmt="{:.2f}"):
    d = d[(d["config.alpha"] == alpha) & (d["config.spectrum"] == spec) &
          (d["config.delta_mode"] == mode)]
    M_ = np.full((len(dt_list), len(S_list)), np.nan)
    for _, r in d.iterrows():
        i = dt_list.index(r["config.mean_dt"]) if r["config.mean_dt"] in dt_list else None
        j = S_list.index(r["config.S"]) if r["config.S"] in S_list else None
        if i is not None and j is not None:
            M_[i, j] = r[value_col]
    im = ax.imshow(M_, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks(range(len(S_list))); ax.set_xticklabels([f"{s:g}" for s in S_list], rotation=45)
    ax.set_yticks(range(len(dt_list))); ax.set_yticklabels([f"{v:g}" for v in dt_list])
    ax.set_xlabel("sparsity S"); ax.set_ylabel("mean step  $\\bar\\Delta$")
    ax.set_title(title, fontsize=9)
    for i in range(M_.shape[0]):
        for j in range(M_.shape[1]):
            if not np.isnan(M_[i, j]):
                ax.text(j, i, fmt.format(M_[i, j]), ha="center", va="center",
                        fontsize=6, color="white" if M_[i, j] < (vmin+vmax)/2 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/results.jsonl")
    ap.add_argument("--outdir", default="results/figs")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = flat_cols(load(args.results))
    print(f"loaded {len(df)} runs")
    if len(df) == 0:
        return
    print("\n--- realized delta stats by mode ---")
    print(df.groupby("config.delta_mode")[["delta_mean_realized", "delta_std_realized"]]
          .mean().round(4))

    S_list = sorted(df["config.S"].unique())
    dt_list = sorted(df["config.mean_dt"].unique())

    # -------- phase diagrams: probe_r2_w over (S, dt) for each (alpha, mode) --------
    alphas = sorted(df["config.alpha"].unique())
    for spec in ["hetero", "homo"]:
        nrow, ncol = len(alphas), 2
        fig, axes = plt.subplots(nrow, ncol, figsize=(9, 2.6 * nrow))
        for r, alpha in enumerate(alphas):
            for c, mode in enumerate(["iid", "selective"]):
                ax = axes[r, c] if nrow > 1 else axes[c]
                phase_axis(ax, df, "probe_r2_w",
                           f"alpha={alpha} {spec} {mode}", S_list, dt_list, alpha, spec, mode)
        fig.suptitle("Linear recoverability  R²_w (fresh features)  — phase diagrams")
        fig.tight_layout()
        fig.savefig(f"{args.outdir}/phase_probe_{spec}.png", dpi=140)
        plt.close(fig)

    # -------- SAE AUC diagrams --------
    for spec in ["hetero", "homo"]:
        nrow, ncol = len(alphas), 2
        fig, axes = plt.subplots(nrow, ncol, figsize=(9, 2.6 * nrow))
        for r, alpha in enumerate(alphas):
            for c, mode in enumerate(["iid", "selective"]):
                ax = axes[r, c] if nrow > 1 else axes[c]
                phase_axis(ax, df, "auc", f"SAE AUC alpha={alpha} {spec} {mode}",
                           S_list, dt_list, alpha, spec, mode, vmin=0.4, vmax=1.0)
        fig.suptitle("SAE feature-detection AUC (twin-pair readout)")
        fig.tight_layout()
        fig.savefig(f"{args.outdir}/phase_auc_{spec}.png", dpi=140)
        plt.close(fig)

    # -------- collapse-collapse: probe vs crowding (alive_eff) --------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for c, mode in enumerate(["iid", "selective"]):
        ax = axes[c]
        d = df[(df["config.delta_mode"] == mode) & (df["config.spectrum"] == "hetero")]
        for alpha in alphas:
            dd = d[d["config.alpha"] == alpha].sort_values("alive")
            ax.scatter(dd["alive"], dd["probe_r2_w"], s=14, label=f"alpha={alpha}", alpha=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("crowding  $\\rho_{alive} = S\\,M\\,\\tau_{eff}$")
        ax.set_ylabel("probe R²_w")
        ax.set_title(f"{mode}")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle("Capacity collapse vs measured crowding (hetero spectrum)")
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/collapse_vs_crowding.png", dpi=140)
    plt.close(fig)

    # -------- iid vs selective paired difference --------
    piv = df.pivot_table(index=["config.S", "config.mean_dt", "config.alpha", "config.spectrum"],
                         columns="config.delta_mode", values="probe_r2_w").dropna()
    if {"iid", "selective"} <= set(piv.columns):
        piv["delta_probe"] = piv["selective"] - piv["iid"]
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        ax.scatter(piv["iid"], piv["selective"], s=12, alpha=0.6)
        lim = [0, max(1.0, piv[["iid", "selective"]].max().max())]
        ax.plot(lim, lim, "k--", lw=0.8)
        ax.set_xlabel("probe R²_w  (iid Δ_t)")
        ax.set_ylabel("probe R²_w  (selective Δ_t)")
        ax.set_title("Matched-marginal comparison")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{args.outdir}/iid_vs_selective.png", dpi=140)
        plt.close(fig)
        print("\n--- selective minus iid (probe R²_w), by alpha ---")
        print(piv.reset_index().rename(columns={"config.alpha": "alpha"})
              .groupby("alpha")["delta_probe"].agg(["mean", "min", "max"]).round(3))

    # -------- collapse vs full predictor c = S*tau_eff*(alpha/N + pbar^2) --------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for c_, mode in enumerate(["iid", "selective"]):
        ax = axes[c_]
        d = df[(df["config.delta_mode"] == mode) & df["c_full"].notna()]
        sc = ax.scatter(d["c_full"], d["probe_r2_w"], c=np.log10(d["config.alpha"]),
                        s=14, cmap="plasma", alpha=0.85)
        xs = np.logspace(-4, 3, 100)
        ax.plot(xs, 1 / (1 + xs), "k--", lw=1, label="1/(1+c)")
        ax.set_xscale("log")
        ax.set_xlabel(r"predictor  $c = S\,\tau_{eff}\,(\alpha/N + \bar p^2)$")
        ax.set_ylabel("probe R²_w")
        ax.set_title(f"{mode}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.colorbar(sc, ax=axes, label=r"$\log_{10}\alpha$")
    fig.suptitle("Linear recoverability collapse (hetero+homo pooled)")
    fig.savefig(f"{args.outdir}/collapse_full_predictor.png", dpi=140)
    plt.close(fig)

    # residual structure: R²_w measured vs 1/(1+c)
    dd = df[df["c_full"].notna()].copy()
    dd["pred"] = 1 / (1 + dd["c_full"])
    dd["resid"] = dd["probe_r2_w"] - dd["pred"]
    print("\n--- collapse fit: R2_w vs 1/(1+c_full) ---")
    print(dd.groupby("config.delta_mode")[["resid"]]
          .agg(["mean", "std"]).round(3))
    print("corr(c_full, 1-R2_w):",
          round(np.corrcoef(dd["c_full"], 1 - dd["probe_r2_w"])[0, 1], 3))

    print("\nfigures written to", args.outdir)


if __name__ == "__main__":
    main()
