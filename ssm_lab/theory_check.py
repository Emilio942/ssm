"""Exact static-theory check (vectorized):
   const-Delta => atoms w_j(d) = (bbar*V_j) * exp(-a*dt*d)  (deterministic per lag)
   G = sum_d W_d W_d^T = (W^T W) . (V V^T elementwise)
   c_j(d) = w_i(0)^T G^{-1} w_j(d)  =>  C = (W .* u) V^T,  u = G^{-1} w_i(0)
   y = S * (sum_{(j,d)!=(i,0)} C[d,j]^2) / C[0,i]^2
Closed form (homo): y = S*alpha/(1-q^2) - S."""
import sys, math
import numpy as np, pandas as pd
import torch
sys.path.insert(0, "ssm_lab")
from analyze import load
from sim import make_decay

device = "cuda"

def y_static_exact(N, M, D, S, dt, spectrum, a, W_B, Phi, tail=1e-4):
    d_max = int(min(20000, max(300, 6.0 / (2 * dt * max(float(a.min()), 1e-3)))))
    V = W_B @ Phi                                             # (N, M)
    d = torch.arange(d_max, device=device, dtype=torch.float32)
    W = (1.0 - torch.exp(-dt * a)) / a * torch.exp(-a.unsqueeze(0) * dt * d.unsqueeze(-1))  # (D,N)
    G = (W.T @ W) * (V @ V.T)                                  # (W^TW) o (VV^T)
    Ginv = torch.linalg.inv(G + 1e-8 * G.diagonal().mean() * torch.eye(N, device=device))
    feats = torch.randperm(M, device=device)[:48]
    ys = []
    for i in feats.tolist():
        u = Ginv @ (V[:, i] * W[0])
        C = (W * u.unsqueeze(0)) @ V                          # (d_max, M)
        c0 = float(C[0, i])
        tot = float((C * C).sum()) - float(C[0, i]) ** 2
        ys.append(S * tot / c0 ** 2)
    return float(np.mean(ys))

def main():
    df = load("results/phase1.jsonl")
    rows = []
    for _, r in df.iterrows():
        if r["config.delta_mode"] != "const":
            continue
        gen = torch.Generator(device=device); gen.manual_seed(10_000 + r["config.seed"])
        a = make_decay(r["config.N"], r["config.spectrum"], device, gen)
        W_B = torch.randn(r["config.N"], r["config.D"], device=device, generator=gen) / math.sqrt(r["config.D"])
        Phi = torch.randn(r["config.D"], r["config.M"], device=device, generator=gen) / math.sqrt(r["config.D"])
        yt = y_static_exact(r["config.N"], r["config.M"], r["config.D"], r["config.S"],
                            r["config.mean_dt"], r["config.spectrum"], a, W_B, Phi)
        ym = (1 - r["probe_r2_w"]) / max(r["probe_r2_w"], 1e-3)
        q = math.exp(-r["config.mean_dt"])
        ycf = (r["config.S"] * r["config.alpha"] / (1 - q ** 2) - r["config.S"]
               if r["config.spectrum"] == "homo" else None)
        rows.append(dict(S=r["config.S"], alpha=r["config.alpha"], dt=r["config.mean_dt"],
                         spec=r["config.spectrum"], y_meas=ym, y_theory=yt, y_closed=ycf))
    T = pd.DataFrame(rows)
    T.to_csv("results/theory_check.csv", index=False)
    print(f"configs checked: {len(T)}")
    for spec in ["homo", "hetero"]:
        t = T[T["spec"] == spec]
        ok = t[(t["y_meas"] > 0.03) & (t["y_theory"] > 1e-5)]
        if len(ok) < 5:
            continue
        corr = np.corrcoef(np.log10(ok["y_meas"].clip(lower=1e-3)),
                           np.log10(ok["y_theory"].clip(lower=1e-6)))[0, 1]
        ratio = ok["y_meas"] / ok["y_theory"].clip(lower=1e-9)
        print(f"{spec}: n={len(ok)} corr(log)={corr:.3f}  "
              f"y_meas/y_theory: median {ratio.median():.2f} "
              f"[p25 {ratio.quantile(0.25):.2f}, p75 {ratio.quantile(0.75):.2f}]")
        if spec == "homo":
            ok2 = t[(t["y_closed"].notna()) & (t["y_meas"] > 0.03) & (t["y_closed"] > 1e-4)]
            r2 = ok2["y_meas"] / ok2["y_closed"]
            print(f"  closed form: y_meas/y_closed median {r2.median():.2f} "
                  f"[p25 {r2.quantile(0.25):.2f}, p75 {r2.quantile(0.75):.2f}]  n={len(ok2)}")

if __name__ == "__main__":
    main()

def part2():
    """iid rows: ratio y_meas/y_static_theory = pure variability effect."""
    df = load("results/phase1.jsonl")
    rows = []
    for _, r in df.iterrows():
        if r["config.delta_mode"] != "iid":
            continue
        gen = torch.Generator(device=device); gen.manual_seed(10_000 + r["config.seed"])
        a = make_decay(r["config.N"], r["config.spectrum"], device, gen)
        W_B = torch.randn(r["config.N"], r["config.D"], device=device, generator=gen) / math.sqrt(r["config.D"])
        Phi = torch.randn(r["config.D"], r["config.M"], device=device, generator=gen) / math.sqrt(r["config.D"])
        yt = y_static_exact(r["config.N"], r["config.M"], r["config.D"], r["config.S"],
                            r["config.mean_dt"], r["config.spectrum"], a, W_B, Phi)
        ym = (1 - r["probe_r2_w"]) / max(r["probe_r2_w"], 1e-3)
        rows.append(dict(S=r["config.S"], alpha=r["config.alpha"], dt=r["config.mean_dt"],
                         spec=r["config.spectrum"], y_meas=ym, y_theory=yt))
    T = pd.DataFrame(rows)
    T.to_csv("results/theory_check_iid.csv", index=False)
    ok = T[(T["y_meas"] > 0.03) & (T["y_theory"] > 1e-5)]
    ratio = ok["y_meas"] / ok["y_theory"].clip(lower=1e-9)
    print(f"iid: n={len(ok)} corr(log)="
          f"{np.corrcoef(np.log10(ok['y_meas'].clip(lower=1e-3)), np.log10(ok['y_theory'].clip(lower=1e-6)))[0,1]:.3f}")
    print(f"  y_meas/y_static_theory: median {ratio.median():.2f} "
          f"[p25 {ratio.quantile(0.25):.2f}, p75 {ratio.quantile(0.75):.2f}]")

if len(sys.argv) > 1 and sys.argv[1] == "iid":
    part2()
