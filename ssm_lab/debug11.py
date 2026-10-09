import torch, math
from sae import train_sae, greedy_match, auc_matched
device = "cuda"
gen = torch.Generator(device=device); gen.manual_seed(0)
N, M, S, n = 64, 320, 0.01, 262144
V = torch.randn(N, M, device=device, generator=gen) / math.sqrt(N)
mask = torch.rand(n, M, device=device, generator=gen) < S
a = mask.float() * torch.randn(n, M, device=device, generator=gen)
h = a @ V.T
for P, k, steps in [(1024, 4, 12000), (512, 4, 12000), (1024, 8, 8000)]:
    sae, hm, hs = train_sae(h, N, P, device, gen, steps=steps, topk=k)
    with torch.no_grad():
        _, f = sae((h - hm) / hs)
    f = f.float()
    match, mc, C = greedy_match(a.abs(), f, device)
    print(f"P={P} k={k} steps={steps}: match_corr={mc.mean():.3f}  auc={auc_matched(f, a.abs(), match):.3f}")
