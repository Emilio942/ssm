"""
Selective SSM simulation with synthetic sparse feature injection.

Model (diagonal A for speed; equivalent to S6 with diagonal A as in Mamba):
    h_t = exp(-Delta_t * a) * h_{t-1} + bbar_t * (W_B x_t)
    bbar_t = (1 - exp(-Delta_t * a)) / a          (exact ZOH for diagonal A)
    x_t = Phi a_t,   a_t ~ Bernoulli(S) * N(0,1)  (sparse ground-truth features)

Delta_t modes:
    'iid'       : LogNormal, mean = mean_dt, cv = dt_cv (context-independent noise)
    'selective' : softplus(g * s_t + b) with s_t a standardized random projection of
                  x_t, b tuned so E[Delta_t] = mean_dt  (true input-dependent selectivity)

Decay spectra:
    'hetero' : a_i ~ U(a_min, a_max)   (heterogeneous decay -> direction warping)
    'homo'   : a_i = 1                  (scalar decay -> atoms keep direction)
"""
import math
import torch


def make_decay(N, spectrum, device, gen, a_min=0.5, a_max=2.0):
    if spectrum == "homo":
        return torch.ones(N, device=device)
    return a_min + (a_max - a_min) * torch.rand(N, device=device, generator=gen)


def _softplus(x):
    return torch.nn.functional.softplus(x)


def calibrate_selective_bias(s_flat, g, target_mean, device):
    """Find b such that mean(softplus(g*s + b)) == target_mean (bisection)."""
    lo, hi = -60.0, 60.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        m = _softplus(g * s_flat + mid).mean().item()
        if m < target_mean:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def calibrate_selective_moments(s_flat, target_mean, target_cv, device):
    """Find (g, b) so that softplus(g*s+b) has mean=target_mean, cv=target_cv.
    Outer bisection on g (std monotone increasing in g), inner on b (mean monotone)."""
    smin, smax = s_flat.min().item(), s_flat.max().item()
    lo_g, hi_g = 1e-4, 50.0
    for _ in range(40):
        g = 0.5 * (lo_g + hi_g)
        b = calibrate_selective_bias(s_flat, g, target_mean, device)
        d = _softplus(g * s_flat + b)
        cv = (d.std() / d.mean().clamp_min(1e-12)).item()
        if cv < target_cv:
            lo_g = g
        else:
            hi_g = g
    g = 0.5 * (lo_g + hi_g)
    b = calibrate_selective_bias(s_flat, g, target_mean, device)
    return g, b


def sample_deltas(mode, mean_dt, dt_cv, g, X, proj_u, s_std, device, gen, dt_clip=None):
    """X: (B,L,D) inputs. Returns deltas (B,L) > 0 with empirical mean ~= mean_dt.
    'selective' mode moment-matches (mean, cv) of the iid mode so that any
    performance difference is attributable to input-dependence, not marginals."""
    B, L, _ = X.shape
    if mode == "iid":
        cv = max(dt_cv, 1e-6)
        sig2 = math.log(1.0 + cv * cv)
        mu = math.log(max(mean_dt, 1e-12)) - sig2 / 2.0
        z = torch.randn(B, L, device=device, generator=gen)
        d = torch.exp(mu + math.sqrt(sig2) * z)
        if dt_cv > 0 and dt_clip is not None:
            lo, hi = dt_clip
            d = d.clamp(mean_dt * lo, mean_dt * hi)
        return d
    elif mode == "selective":
        # coupling-strength parameterization: s_t = g*s_content + sqrt(1-g^2)*s_noise
        # (both standardized) => Corr(Delta path, content) grows with g while the
        # marginal moments of Delta stay fixed by the (gg, bb) calibration.
        s_content = (X @ proj_u) / s_std
        s_noise = torch.randn(B, L, device=device, generator=gen)
        s_noise = (s_noise - s_noise.mean()) / s_noise.std().clamp_min(1e-6)
        gg_eff = max(min(g, 1.0), 0.0)
        s = gg_eff * s_content + math.sqrt(max(1.0 - gg_eff ** 2, 0.0)) * s_noise
        gg, bb = calibrate_selective_moments(s.reshape(-1).float(), mean_dt,
                                             max(dt_cv, 1e-6), device)
        return _softplus(gg * s + bb)
    elif mode == "noise":
        # control: same softplus(g*s+b) family, but s is independent noise
        # (moment-matched to mean/cv) -> isolates content-correlation effect
        s = torch.randn(B, L, device=device, generator=gen)
        gg, bb = calibrate_selective_moments(s.reshape(-1).float(), mean_dt,
                                             max(dt_cv, 1e-6), device)
        return _softplus(gg * s + bb)
    elif mode == "const":
        return torch.full((B, L), float(mean_dt), device=device)
    raise ValueError(mode)


@torch.no_grad()
def run_ssm(N, M, D, L, B_batch, S, mean_dt, delta_mode, dt_cv, g,
            a, W_B, Phi, device, gen, label_dtype=torch.float16, eval_idx=None, dt_clip=None):
    """Run batched selective SSM. Returns dict with h, labels a_t, deltas, extras.
    If eval_idx is given, labels are returned only for those feature columns."""
    # --- sparse ground-truth features ---
    mask = torch.rand(B_batch, L, M, device=device, generator=gen) < S
    vals = torch.randn(B_batch, L, M, device=device, generator=gen)
    a_t = (mask.to(vals.dtype)) * vals                       # (B,L,M)
    X = a_t @ Phi.T                                          # (B,L,D)

    # --- selectivity ---
    proj_u = torch.randn(D, device=device, generator=gen)
    proj_u = proj_u / proj_u.norm()
    s_flat = ((X.reshape(-1, D)) @ proj_u)
    s_std = s_flat.std().clamp_min(1e-6).item()
    deltas = sample_deltas(delta_mode, mean_dt, dt_cv, g, X, proj_u, s_std, device, gen, dt_clip=dt_clip)

    # --- discretization (exact ZOH, diagonal A = -diag(a)) ---
    eda = torch.exp(-deltas.unsqueeze(-1) * a)               # (B,L,N)
    bbar = (1.0 - eda) / a                                   # (B,L,N) -> *Delta as Delta->0
    WX = X @ W_B.T                                           # (B,L,N)

    # --- recurrence ---
    h = torch.empty(B_batch, L, N, device=device, dtype=torch.float32)
    hp = torch.zeros(B_batch, N, device=device)
    a_mean = a.mean()
    # accumulated (decayed) activity trace for eval features: proxy gamma_i(t)
    tr = None
    tr_out = None
    if eval_idx is not None:
        tr = torch.zeros(B_batch, len(eval_idx), device=device)
        tr_out = torch.empty(B_batch, L, len(eval_idx), device=device, dtype=torch.float32)
    decay_avg = torch.exp(-deltas * a_mean)                  # (B,L) scalar proxy
    fresh_eval = a_t[:, :, eval_idx] if eval_idx is not None else None
    for t in range(L):
        hp = eda[:, t] * hp + bbar[:, t] * WX[:, t]
        h[:, t] = hp
        if tr is not None:
            tr = decay_avg[:, t].unsqueeze(-1) * tr + fresh_eval[:, t]
            tr_out[:, t] = tr

    return {
        "h": h.reshape(-1, N),
        "a_t": (a_t.reshape(-1, M) if eval_idx is None
                else a_t.reshape(-1, M)[:, eval_idx]).to(label_dtype),
        "trace": None if tr_out is None else tr_out.reshape(-1, tr_out.shape[-1]).to(label_dtype),
        "deltas": deltas.reshape(-1),
    }


@torch.no_grad()
def geometry_metrics(N, M, D, L, S, mean_dt, delta_mode, dt_cv, g,
                     a, W_B, Phi, device, gen, n_feat=32, n_t=64,
                     lag_grid=(0, 1, 2, 4, 8, 16, 32, 64, 128)):
    """
    Direct measurement of the warped dictionary  atom_i(k,t) without SAE:
        atom_i(k,t) = (bbar_k * (W_B Phi_i)) * exp(-a * T_{k,t}),
        T_{k,t} = sum_{j=k+1..t} Delta_j  (cumulative decay path)
    Reports: energy decay profile, CTIM (mean/max |cos| between distinct atoms),
    per-feature direction drift, and an SINR-style capacity proxy.
    """
    # one sequence of deltas
    x = torch.randn(1, L, D, device=device, generator=gen) * (S ** 0.5)  # scale like E x
    proj_u = torch.randn(D, device=device, generator=gen); proj_u /= proj_u.norm()
    s = (x.reshape(-1, D) @ proj_u); s_std = s.std().clamp_min(1e-6).item()
    deltas = sample_deltas(delta_mode, mean_dt, dt_cv, g, x, proj_u, s_std, device, gen)[0]  # (L,)
    cs = torch.cumsum(deltas, 0)                              # T up to t (inclusive)

    feats = torch.randperm(M, device=device, generator=gen)[:min(n_feat, M)]
    V = (W_B @ Phi[:, feats])                                 # (N, F) static atoms
    lags = torch.tensor([l for l in lag_grid if l < L], device=device)
    ts = torch.arange(L - 1, max(lags.max().item(), 1), -1, device=device)[:n_t]
    if ts.numel() == 0:
        ts = torch.tensor([L - 1], device=device)

    bbar = (1.0 - torch.exp(-deltas.unsqueeze(-1) * a)) / a   # (L,N)

    energies, coss, persist = [], [], []
    mean_dirs = {}   # lag -> (F,N) mean atom direction (normalized)
    for d in lags.tolist():
        At = ts[ts - 1 - d >= 0] if d > 0 else ts             # injection indices k
        if At.numel() == 0:
            continue
        k = At - d
        T = cs[At] - cs[k]                                    # (Tn,) decay path length
        decay = torch.exp(-T.unsqueeze(-1) * a)               # (Tn,N)
        bb = bbar[k]                                          # (Tn,N)
        # atom[f, tn, :] = decay * bb * V[:, f]
        atoms = torch.einsum("tn,tn,nf->ftn", decay, bb, V)   # (F,Tn,N)
        e = atoms.pow(2).sum(-1)                              # (F,Tn)
        energies.append(e.mean().item())
        # CTIM: cos between different features (same lag, same t)
        An = atoms / atoms.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        Fdim = An.shape[0]
        if Fdim > 1:
            C = torch.einsum("ftn,gtn->fgt", An, An)          # (F,F,Tn)
            mask = ~torch.eye(Fdim, dtype=torch.bool, device=device)
            coss.append(C[mask].abs().mean().item())
        # mean direction per feature at this lag (for lag-persistence curve)
        md = atoms.mean(dim=1)                                # (F,N)
        md = md / md.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        mean_dirs[d] = md

    # direction persistence: cos(atom(d), atom(d0)) as a function of lag d
    d0 = lags[0].item() if len(lags) else 0
    if d0 in mean_dirs:
        m0 = mean_dirs[d0]
        for d in lags.tolist():
            if d in mean_dirs:
                c = (mean_dirs[d] * m0).sum(-1)               # (F,)
                persist.append(c.mean().item())

    # SINR proxy at lag 0: signal energy of newest features vs interference energy
    # interference ~ S * M * sum_d E||atom(d)||^2  (all other active features, all lags)
    sinr = None
    eff_window = None
    if len(energies) >= 2:
        e0 = energies[0]
        etot = sum(energies)  # crude upper bound of accumulated other-energy per feature
        sinr = e0 / max(S * M * etot, 1e-12)
        eff_window = etot / max(e0, 1e-18)   # effective memory window in steps

    return {
        "lag_grid": [int(l) for l in lags.tolist()],
        "atom_energy": energies,
        "ctim_mean_abs_cos": sum(coss) / max(len(coss), 1),
        "dir_persist_by_lag": persist,
        "sinr_proxy": sinr,
        "eff_window": eff_window,
        "mean_T_end": cs[-1].item(),
        "var_T_end": cs[-1].var(unbiased=False).item() if cs.numel() > 1 else 0.0,
    }
