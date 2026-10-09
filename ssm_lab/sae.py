"""
Sparse Autoencoder (standard, ReLU + L1) for h_t decomposition,
plus NRE / FRR metrics with greedy feature-to-latent matching.
"""
import torch
import torch.nn as nn


class SAE(nn.Module):
    def __init__(self, n, p):
        super().__init__()
        self.enc = nn.Linear(n, p)
        self.dec = nn.Linear(p, n)
        nn.init.kaiming_uniform_(self.enc.weight, nonlinearity="relu")
        nn.init.zeros_(self.enc.bias)
        with torch.no_grad():
            w = torch.randn(p, n)
            w = w / w.norm(dim=1, keepdim=True)
            self.dec.weight.copy_(w.T)
        nn.init.zeros_(self.dec.bias)

    def forward(self, x):
        f = nn.functional.relu(self.enc(x))
        return self.dec(f), f


class TopKSAE(nn.Module):
    """TopK sparse autoencoder (Gao et al. 2024): keep k largest pre-activations."""
    def __init__(self, n, p, k):
        super().__init__()
        self.k = k
        self.enc = nn.Linear(n, p)
        self.dec = nn.Linear(p, n)
        nn.init.kaiming_uniform_(self.enc.weight, nonlinearity="relu")
        nn.init.zeros_(self.enc.bias)
        with torch.no_grad():
            w = torch.randn(p, n)
            w = w / w.norm(dim=1, keepdim=True)
            self.dec.weight.copy_(w.T)
        nn.init.zeros_(self.dec.bias)

    def forward(self, x):
        pre = self.enc(x)
        v, idx = torch.topk(pre, min(self.k, pre.shape[-1]), dim=-1)
        mask = torch.zeros_like(pre).scatter_(-1, idx, 1.0)
        f = torch.relu(pre) * mask
        return self.dec(f), f


def train_sae(h_train, n, p, device, gen, steps=3000, batch=1024, lr=1e-3, l1=1e-3,
              h_mean=None, h_std=None, log_every=500, topk=None):
    """h_train: (n_samples, N) float32 on device. Returns (sae, h_mean, h_std)."""
    if h_mean is None:
        h_mean = h_train.mean(0, keepdim=True)
        h_std = h_train.std(0, keepdim=True).clamp_min(1e-6)
    X = ((h_train - h_mean) / h_std)
    sae = (TopKSAE(n, p, topk) if topk else SAE(n, p)).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    n_samp = X.shape[0]
    for it in range(steps):
        idx = torch.randint(0, n_samp, (min(batch, n_samp),), device=device, generator=gen)
        xb = X[idx]
        xhat, f = sae(xb)
        loss = (xhat - xb).pow(2).sum(-1).mean() / n
        if not topk:
            loss = loss + l1 * f.sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # keep decoder columns unit-norm (standard SAE practice)
        with torch.no_grad():
            sae.dec.weight.mul_(1.0 / sae.dec.weight.norm(dim=0, keepdim=True).clamp_min(1e-8))
        opt.step()
    return sae, h_mean, h_std


@torch.no_grad()
def nre(sae, h, h_mean, h_std, device, batch=8192):
    """Normalized reconstruction error (eq. 10 of proposal) on standardized scale."""
    sae.eval()
    num, den = 0.0, 0.0
    X = (h - h_mean) / h_std
    for i in range(0, X.shape[0], batch):
        xb = X[i:i + batch]
        xhat, _ = sae(xb)
        num += (xb - xhat).pow(2).sum(-1).sum().item()
        den += xb.pow(2).sum(-1).sum().item()
    return num / max(den, 1e-12)


@torch.no_grad()
def greedy_match(a_val, f_val, device, max_corr_only=False):
    """
    Match M ground-truth features to P SAE latents by activation correlation.
    a_val: (n, M), f_val: (n, P). Returns (match: list len M -> latent or -1,
            corr_of_match: tensor (M,), corr matrix).
    """
    n, M = a_val.shape
    P = f_val.shape[1]
    A = a_val - a_val.mean(0, keepdim=True)
    F = f_val - f_val.mean(0, keepdim=True)
    A = A / A.std(0, keepdim=True).clamp_min(1e-6)
    F = F / F.std(0, keepdim=True).clamp_min(1e-6)
    # guard memory: chunk over features
    C = torch.empty(M, P, device=device)
    for i in range(0, M, 512):
        C[i:i + 512] = (A[:, i:i + 512].T @ F) / n
    C = torch.nan_to_num(C, nan=0.0)
    if max_corr_only:
        return None, C.abs().max(dim=1).values, C
    # greedy: process features by descending best corr, assign best unused latent
    best, best_p = C.max(dim=1)
    order = torch.argsort(best, descending=True)
    used = torch.zeros(P, dtype=torch.bool, device=device)
    match = torch.full((M,), -1, dtype=torch.long, device=device)
    match_corr = torch.zeros(M, device=device)
    for i in order.tolist():
        row = C[i].clone()
        row[used] = -2.0
        p = int(row.argmax().item())
        if row[p].item() <= -1.9:
            continue
        used[p] = True
        match[i] = p
        match_corr[i] = C[i, p].item()
    return match, match_corr, C


@torch.no_grad()
def frr(sae, h_val, a_val, h_mean, h_std, device, tau=0.05, batch=8192):
    """
    Feature Recovery Rate (eq. 11): fraction of active GT features whose matched
    latent fires above tau (latents live in standardized-h units).
    Also returns false-positive rate (latent fires while GT silent).
    """
    sae.eval()
    Fs = []
    for i in range(0, h_val.shape[0], batch):
        _, f = sae((h_val[i:i + batch] - h_mean) / h_std)
        Fs.append(f)
    f_val = torch.cat(Fs, 0).float()
    match, match_corr, _ = greedy_match(a_val.float(), f_val, device)
    hits, tot = 0, 0
    fp, tn = 0, 0
    for i in range(a_val.shape[1]):
        p = int(match[i].item())
        if p < 0:
            continue
        active = a_val[:, i] != 0
        if active.any():
            hits += (f_val[active, p] > tau).sum().item()
            tot += int(active.sum().item())
            fp += (f_val[~active, p] > tau).sum().item()
            tn += int((~active).sum().item())
    return {
        "frr": hits / max(tot, 1),
        "fpr": fp / max(tn, 1),
        "matched_frac": (match >= 0).float().mean().item(),
        "mean_match_corr": match_corr.mean().item(),
    }


@torch.no_grad()
def auc_matched(f_val, a_val, match, max_n=20000):
    """AUC of matched latent separating active vs inactive, per feature (prior-robust)."""
    n = f_val.shape[0]
    idx = torch.randperm(n, device=f_val.device)[:max_n]
    f, a = f_val[idx], a_val[idx]
    aucs = []
    for i in range(a.shape[1]):
        p = int(match[i].item())
        if p < 0:
            continue
        act = a[:, i] != 0
        n_act, n_ina = int(act.sum()), int((~act).sum())
        if n_act < 5 or n_ina < 5:
            continue
        ranks = f[:, p].argsort().argsort().float() + 1.0        # average ranks via ties-ignore
        auc = (ranks[act].sum().item() - n_act * (n_act + 1) / 2) / (n_act * n_ina)
        aucs.append(auc)
    return sum(aucs) / max(len(aucs), 1)


@torch.no_grad()
def evaluate_sae(f_val, a_val, device, max_n=40000):
    """
    Sign-robust SAE evaluation for signed sparse features:
      for each GT feature i, find latent p maximizing |corr(f_p, a_i)|;
      orientation s = sign(corr).
    Returns dict with:
      match_abs_corr : mean_i |corr(f_p*, a_i)|            (overall alignment)
      extract_corr   : mean_i |corr(f_p*, a_i) | active|   (extraction given active)
      auc            : AUC of oriented latent detecting activity
      matched_frac   : fraction of features with |corr| > 0.1
    """
    n, M = a_val.shape
    idx = torch.randperm(n, device=device)[:max_n]
    f, a = f_val[idx].float(), a_val[idx].float()
    A = a - a.mean(0, keepdim=True)
    F = f - f.mean(0, keepdim=True)
    A = A / A.std(0, keepdim=True).clamp_min(1e-6)
    F = F / F.std(0, keepdim=True).clamp_min(1e-6)
    C = torch.empty(M, f.shape[1], device=device)
    for i in range(0, M, 512):
        C[i:i + 512] = (A[:, i:i + 512].T @ F) / f.shape[0]
    C = torch.nan_to_num(C, nan=0.0)
    absC = C.abs()
    best2 = absC.topk(min(2, absC.shape[1]), dim=1)      # (M,2) twin-pair candidates
    best_p = best2.indices                                # (M,2)
    best_c = C.gather(1, best_p)                          # (M,2) signed corrs
    match_abs = best_c[:, 0].abs().mean().item()

    ext, aucs, matched = [], [], 0
    for i in range(M):
        p1, p2 = int(best_p[i, 0].item()), int(best_p[i, 1].item())
        s1 = 1.0 if best_c[i, 0] >= 0 else -1.0
        s2 = 1.0 if best_c[i, 1] >= 0 else -1.0
        act = a[:, i] != 0
        n_act = int(act.sum())
        if n_act < 5:
            continue
        # twin-pair linear readout of the signed coefficient
        read = s1 * f[:, p1] + s2 * f[:, p2]
        ext.append(abs(torch.corrcoef(torch.stack([read[act], a[act, i]]))[0, 1]).item())
        # detection from combined firing of the twin pair
        score = f[:, p1] + f[:, p2]
        ranks = score.argsort().argsort().float() + 1.0
        n_ina = f.shape[0] - n_act
        aucs.append((ranks[act].sum().item() - n_act * (n_act + 1) / 2) / (n_act * n_ina))
        if abs(best_c[i, 0].item()) > 0.1:
            matched += 1
    return {
        "match_abs_corr": match_abs,
        "extract_corr": sum(ext) / max(len(ext), 1),
        "auc": sum(aucs) / max(len(aucs), 1),
        "matched_frac": matched / M,
    }
