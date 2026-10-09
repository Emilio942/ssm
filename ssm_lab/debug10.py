import torch, math
from sae import train_sae, greedy_match
device = "cuda"
gen = torch.Generator(device=device); gen.manual_seed(0)
N, M, S, n = 64, 320, 0.01, 262144
V = torch.randn(N, M, device=device, generator=gen) / math.sqrt(N)
mask = torch.rand(n, M, device=device, generator=gen) < S
a = mask.float() * torch.randn(n, M, device=device, generator=gen)
h = a @ V.T

for k in [4, 8, 16]:
    sae, hm, hs = train_sae(h, N, 512, device, gen, steps=8000, topk=k)
    with torch.no_grad():
        _, f = sae((h - hm) / hs)
    f = f.float()
    match, mc, C = greedy_match(a.abs(), f, device)
    # detection AUC vs |a|
    from sae import auc_matched
    print(f"k={k}: match_corr(|a|)={mc.mean():.3f}  auc(|a|)={auc_matched(f, a.abs(), match):.3f}")
