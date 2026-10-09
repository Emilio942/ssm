import torch, math
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

def run(P, k, steps):
    torch.manual_seed(0)
    sae = TopKSAE(N, P, k).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    for it in range(steps):
        idx = torch.randint(0, n, (1024,), device=device, generator=gen)
        xb = X[idx]
        xhat, f = sae(xb)
        loss = (xhat - xb).pow(2).sum(-1).mean() / N
        opt.zero_grad(set_to_none=True); loss.backward()
        with torch.no_grad():
            sae.dec.weight.mul_(1.0/sae.dec.weight.norm(dim=0,keepdim=True).clamp_min(1e-8))
        opt.step()
    with torch.no_grad():
        _, f = sae(X)
    f = f.float()
    match, mc, C = greedy_match(a.abs(), f, device)
    # decoder alignment to true atoms (up to sign)
    D = sae.dec.weight.T            # (P, N)
    Dn = D / D.norm(dim=1, keepdim=True)
    Vn = V / V.norm(dim=0, keepdim=True)
    align = (Dn @ Vn).abs()          # (P, M)
    best = align.max(0).values
    print(f"P={P} k={k}: corr={mc.mean():.3f} auc={auc_matched(f,a.abs(),match):.3f} "
          f"dec_align_med={best.median():.3f} dec_align_p10={best.quantile(0.1):.3f}")

run(512, 6, 8000)
run(2048, 6, 8000)
