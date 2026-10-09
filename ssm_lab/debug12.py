import torch, math, torch.nn as nn
from sae import TopKSAE, greedy_match, auc_matched
device = "cuda"
gen = torch.Generator(device=device); gen.manual_seed(0)
N, M, S, n = 64, 320, 0.01, 262144
V = torch.randn(N, M, device=device, generator=gen) / math.sqrt(N)
mask = torch.rand(n, M, device=device, generator=gen) < S
a = mask.float() * torch.randn(n, M, device=device, generator=gen)
h = a @ V.T
hm, hs_ = h.mean(0, keepdim=True), h.std(0, keepdim=True).clamp_min(1e-6)
X = (h - hm) / hs_

def train_topk(P, k, steps, renorm, lr=1e-3):
    torch.manual_seed(0)
    sae = TopKSAE(N, P, k).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    for it in range(steps):
        idx = torch.randint(0, n, (1024,), device=device, generator=gen)
        xb = X[idx]
        xhat, f = sae(xb)
        loss = (xhat - xb).pow(2).sum(-1).mean() / N
        opt.zero_grad(set_to_none=True); loss.backward()
        if renorm:
            with torch.no_grad():
                sae.dec.weight.mul_(1.0/sae.dec.weight.norm(dim=0,keepdim=True).clamp_min(1e-8))
        opt.step()
    with torch.no_grad():
        _, f = sae(X)
    f = f.float()
    match, mc, C = greedy_match(a.abs(), f, device)
    xhat, _ = sae(X[:8192])
    nre = ((xhat - X[:8192]).pow(2).sum(-1) / X[:8192].pow(2).sum(-1)).mean()
    print(f"P={P} k={k} steps={steps} renorm={renorm}: corr={mc.mean():.3f} auc={auc_matched(f,a.abs(),match):.3f} nre={nre:.4f}")

train_topk(1024, 6, 8000, renorm=False)
train_topk(1024, 6, 30000, renorm=False)
train_topk(512, 6, 30000, renorm=True)
