"""
Single-configuration experiment (v3):
Primary metrics
  - probe_r2_w   : active-weighted linear recoverability R^2 of FRESH features (ridge)
  - probe_r2_tr  : linear recoverability R^2 of the ACCUMULATED decayed trace
  - sae_auc      : AUC of matched SAE latents detecting fresh feature activity
  - nre          : normalized reconstruction error (proposal eq. 10)
  - frr          : proposal eq. 11 at fixed tau (prior-capped; secondary)
Geometry
  - CTIM, atom energy profile, effective memory window, dictionary drift, SINR proxy
"""
import json, math, os, time, argparse
import torch
from sim import make_decay, run_ssm, geometry_metrics
from sae import train_sae, nre, frr, greedy_match, evaluate_sae


@torch.no_grad()
def ridge_predict(h, targets, device, lam_rel=1e-3, max_train=131072):
    """Fit ridge h->targets; returns predictions on the same rows (in-sample, high n)."""
    n, N = h.shape
    idx = torch.randperm(n, device=device)[:max_train]
    X = torch.cat([h[idx], torch.ones(idx.numel(), 1, device=device)], 1)
    A = targets[idx].float()
    reg = lam_rel * (X.T @ X).diagonal().mean() * torch.eye(N + 1, device=device)
    W = torch.linalg.solve(X.T @ X + reg, X.T @ A)
    return torch.cat([h, torch.ones(n, 1, device=device)], 1) @ W


@torch.no_grad()
def weighted_r2(pred, targets, rescale=True):
    """R^2 over active rows, with per-feature optimal linear rescaling of the
    prediction (removes systematic shrinkage of underdetermined maps)."""
    num, den = 0.0, 0.0
    for i in range(targets.shape[1]):
        act = targets[:, i] != 0
        if act.sum() < 5:
            continue
        a, p = targets[act, i], pred[act, i]
        if rescale:
            vp = p.var()
            c = ((a - a.mean()) * (p - p.mean())).mean() / max(vp.item(), 1e-12)
            p = p * c
        num += (a - p).pow(2).sum().item()
        den += a.pow(2).sum().item()
    return max(0.0, 1.0 - num / max(den, 1e-12))


@torch.no_grad()
def plain_r2(pred, targets, max_n=20000):
    idx = torch.randperm(pred.shape[0], device=pred.device)[:max_n]
    p, t = pred[idx], targets[idx]
    return 1.0 - (p - t).pow(2).sum(0).div(t.pow(2).sum(0) + 1e-12).clamp(0, 1).mean().item()


def collect_data(cfg, a, W_B, Phi, M, device, gen, eval_idx):
    N, D, L = cfg["N"], cfg["D"], cfg["L"]
    budget = cfg.get("budget", 400_000)
    Bs = cfg["batch"]
    if Bs * L * M > 3.0e7:
        Bs = max(2, int(3.0e7 // (L * M)))
    hs, ls, ts, dsum, dsumsq, dcnt = [], [], [], 0.0, 0.0, 0
    got = 0
    while got < budget:
        out = run_ssm(N, M, D, L, Bs, cfg["S"], cfg["mean_dt"], cfg["delta_mode"],
                      cfg.get("dt_cv", 1.0), cfg.get("g", 1.0), a, W_B, Phi,
                      device, gen, eval_idx=eval_idx, dt_clip=cfg.get("dt_clip"))
        hs.append(out["h"]); ls.append(out["a_t"])
        if out["trace"] is not None:
            ts.append(out["trace"])
        d = out["deltas"].float()
        dsum += d.sum().item(); dsumsq += d.pow(2).sum().item(); dcnt += d.numel()
        got += out["h"].shape[0]
    delta_mean = dsum / dcnt
    delta_std = max(dsumsq / dcnt - delta_mean ** 2, 0.0) ** 0.5
    return (torch.cat(hs)[:budget], torch.cat(ls)[:budget],
            torch.cat(ts)[:budget] if ts else None,
            {"delta_mean": delta_mean, "delta_std": delta_std})


def run_config(cfg, device):
    gen = torch.Generator(device=device); gen.manual_seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    N, D = cfg["N"], cfg["D"]
    M = max(2, int(round(cfg["alpha"] * N)))
    L = cfg["L"]

    a = make_decay(N, cfg["spectrum"], device, gen,
                   cfg.get("a_min", 0.5), cfg.get("a_max", 2.0))
    W_B = torch.randn(N, D, device=device, generator=gen) / math.sqrt(D)
    Phi = torch.randn(D, M, device=device, generator=gen) / math.sqrt(D)

    n_eval = min(M, cfg.get("n_eval", 128))
    eval_idx = torch.randperm(M, device=device, generator=gen)[:n_eval]

    t0 = time.time()
    h, lab, trace, dstats = collect_data(cfg, a, W_B, Phi, M, device, gen, eval_idx)
    t_sim = time.time() - t0
    res = {"delta_mean_realized": dstats["delta_mean"],
           "delta_std_realized": dstats["delta_std"]}

    n = h.shape[0]
    perm = torch.randperm(n, device=device, generator=gen)
    n_tr = int(0.8 * n)
    h_tr, h_va = h[perm[:n_tr]], h[perm[n_tr:]]
    lab_va = lab[perm[n_tr:]].float()
    n_act = (lab_va != 0).sum(0)

    # ---- linear probe (fresh targets, active-weighted) ----
    pred_fresh = ridge_predict(h_va, lab_va, device)
    res["probe_r2_w"] = weighted_r2(pred_fresh, lab_va)
    if trace is not None:
        tr_va = trace[perm[n_tr:]].float()
        pred_tr = ridge_predict(h_va, tr_va, device)
        res["probe_r2_tr"] = plain_r2(pred_tr, tr_va)
        del pred_tr, tr_va
    del pred_fresh

    # ---- SAE (adaptive capacity: latents scale with dictionary, k with crowding) ----
    t0 = time.time()
    P = cfg.get("p_mult", None)
    if P is None:
        P = int(min(4 * M + 128, 8192))
    k = cfg.get("topk", None)
    if k is None:
        a_mean0 = a.mean().item()
        alive_naive = cfg["S"] * M / max(2.0 * a_mean0 * cfg["mean_dt"], 1e-9)
        k = int(min(max(round(1.5 * alive_naive), 6), 64))
    sae, hm, hs = train_sae(h_tr, N, P, device, gen,
                            steps=cfg.get("steps", 4000), l1=cfg.get("l1", 1e-3),
                            topk=k)
    res["t_train_s"] = time.time() - t0
    res["nre"] = nre(sae, h_va, hm, hs, device)

    n_metric = min(h_va.shape[0], 40000)
    h_met, lab_met = h_va[:n_metric], lab_va[:n_metric]
    with torch.no_grad():
        Fs = []
        for i in range(0, h_met.shape[0], 8192):
            _, f = sae((h_met[i:i + 8192] - hm) / hs)
            Fs.append(f)
        f_met = torch.cat(Fs, 0).float()
    ev = evaluate_sae(f_met, lab_met, device)
    res.update(ev)
    fr = frr(sae, h_met, lab_met, hm, hs, device, tau=cfg.get("tau", 0.05))
    res["frr"], res["fpr"] = fr["frr"], fr["fpr"]
    del f_met

    # ---- geometry ----
    geo = geometry_metrics(N, M, D, L, cfg["S"], cfg["mean_dt"], cfg["delta_mode"],
                           cfg.get("dt_cv", 1.0), cfg.get("g", 1.0),
                           a, W_B, Phi, device, gen)
    res["geometry"] = geo

    a_mean = a.mean().item()
    res["config"] = dict(cfg)
    res["config"]["M"] = M
    res["config"]["sae_P"] = P
    res["config"]["sae_k"] = k
    res["config"]["eff_window"] = geo.get("eff_window")
    res["config"]["alive_eff"] = (cfg["S"] * M * geo["eff_window"]
                                  if geo.get("eff_window") else None)
    res["config"]["min_active_per_eval_feat"] = int(n_act.min())
    res["config"]["frac_eval_feats_seen"] = (n_act > 0).float().mean().item()
    res["config"]["t_sim_s"] = t_sim
    del h, lab, trace, h_tr, h_va, lab_va, h_met, lab_met
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/results.jsonl")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cfg = json.loads(open(args.config).read())
    res = run_config(cfg, device)
    with open(args.out, "a") as fh:
        fh.write(json.dumps(res) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k not in ("geometry", "config")}, indent=2))


if __name__ == "__main__":
    main()
