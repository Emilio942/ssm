import torch, math
from run_one import ridge_predict, weighted_r2
from sae import train_sae, greedy_match, auc_matched

device = "cuda"
gen = torch.Generator(device=device); gen.manual_seed(0)
N, M, S, n = 64, 320, 0.01, 262144
V = torch.randn(N, M, device=device, generator=gen) / math.sqrt(N)
mask = torch.rand(n, M, device=device, generator=gen) < S
a = mask.float() * torch.randn(n, M, device=device, generator=gen)
h = a @ V.T
pred = ridge_predict(h, a, device)
print("static ridge weighted_r2 (rescaled):", weighted_r2(pred, a))

sae, hm, hs = train_sae(h, N, 512, device, gen, steps=4000, topk=32)
with torch.no_grad():
    _, f = sae((h - hm) / hs)
f = f.float()
match, mc, C = greedy_match(a, f, device)
print("static TopK-SAE: match_corr=%.3f  auc=%.3f" % (mc.mean(), auc_matched(f, a, match)))
