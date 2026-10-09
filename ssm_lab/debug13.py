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
print("||x||^2 mean:", X[:1024].pow(2).sum(-1).mean().item())

torch.manual_seed(0)
sae = TopKSAE(N, 512, 6).to(device)
opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
for it in range(8001):
    idx = torch.randint(0, n, (1024,), device=device, generator=gen)
    xb = X[idx]
    xhat, f = sae(xb)
    loss = (xhat - xb).pow(2).sum(-1).mean() / N
    opt.zero_grad(set_to_none=True); loss.backward()
    with torch.no_grad():
        sae.dec.weight.mul_(1.0/sae.dec.weight.norm(dim=0,keepdim=True).clamp_min(1e-8))
    opt.step()
    if it % 1000 == 0:
        with torch.no_grad():
            fw = sae(xb)
            print(f"it={it} loss={loss.item():.4f} f_sum={f.sum(-1).mean():.2f} "
                  f"dec_norm={sae.dec.weight.norm(dim=0).mean():.3f} enc_wstd={sae.enc.weight.std():.3f}")
