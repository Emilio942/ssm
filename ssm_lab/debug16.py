import torch, math
from sae import TopKSAE
device = "cuda"
gen = torch.Generator(device=device); gen.manual_seed(0)
N, M, S, n = 64, 320, 0.01, 262144
V = torch.randn(N, M, device=device, generator=gen) / math.sqrt(N)
mask = torch.rand(n, M, device=device, generator=gen) < S
a = mask.float() * torch.randn(n, M, device=device, generator=gen)
h = a @ V.T
hm, hs_ = h.mean(0, keepdim=True), h.std(0, keepdim=True).clamp_min(1e-6)
X = (h - hm) / hs_
torch.manual_seed(0)
sae = TopKSAE(N, 2048, 6).to(device)
opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
for it in range(8000):
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

def topk_agg_corr(f, a, topm):
    A = a - a.mean(0, True); F = f - f.mean(0, True)
    A = A / A.std(0, True).clamp_min(1e-6); F = F / F.std(0, True).clamp_min(1e-6)
    C = (A.T @ F) / a.shape[0]
    vals = []
    for i in range(a.shape[1]):
        top = C[i].abs().topk(topm).values
        vals.append(top.max())
    return torch.stack(vals).mean().item()

for m in [1, 2, 3, 5, 8]:
    print(f"top-{m} latent max-corr with |a|: {topk_agg_corr(f, a.abs(), m):.3f}")
# oracle: decoder-direction reading
D = sae.dec.weight.T
Dn = D / D.norm(dim=1, keepdim=True)
align = (Dn @ (V / V.norm(dim=0,keepdim=True))).abs()   # (P, M)
best_p = align.argmax(0)
print("direct decode via best-aligned latent corr:",
      topk_agg_corr(f[:, best_p], a.abs(), 1))
