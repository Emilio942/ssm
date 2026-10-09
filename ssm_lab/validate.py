import torch, math
from sae import train_sae, evaluate_sae
from run_one import ridge_predict, weighted_r2

device = "cuda"
gen = torch.Generator(device=device); gen.manual_seed(0)
N, M, S, n = 64, 320, 0.01, 262144
V = torch.randn(N, M, device=device, generator=gen) / math.sqrt(N)
mask = torch.rand(n, M, device=device, generator=gen) < S
a = mask.float() * torch.randn(n, M, device=device, generator=gen)
h = a @ V.T
pred = ridge_predict(h, a, device)
print("STATIC probe_r2_w:", round(weighted_r2(pred, a), 3), " (theory ~0.96)")
sae, hm, hs = train_sae(h, N, 2048, device, gen, steps=8000, topk=6)
with torch.no_grad():
    _, f = sae((h - hm) / hs)
print("STATIC SAE:", {k: round(v, 3) for k, v in evaluate_sae(f, a, device).items()})
