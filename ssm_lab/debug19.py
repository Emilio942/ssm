import torch, math
from sae import train_sae
device = "cuda"
gen = torch.Generator(device=device); gen.manual_seed(0)
N, M, S, n = 64, 320, 0.01, 262144
V = torch.randn(N, M, device=device, generator=gen) / math.sqrt(N)
mask = torch.rand(n, M, device=device, generator=gen) < S
a = mask.float() * torch.randn(n, M, device=device, generator=gen)
h = a @ V.T
sae, hm, hs = train_sae(h, N, 2048, device, gen, steps=8000, topk=6)
with torch.no_grad():
    _, f = sae((h - hm) / hs)
f = f.float()
print("dead latent frac:", (f.std(0) < 1e-6).float().mean().item())
i = 7
mu_f, mu_a = f.mean(0, keepdim=True), a[:, i].mean()
F = (f - mu_f) / f.std(0, keepdim=True).clamp_min(1e-6)
A = (a[:, i] - mu_a) / a[:, i].std()
C = ((F * A.unsqueeze(1)).mean(0))
C = torch.nan_to_num(C)
p = int(C.abs().argmax())
print(f"best latent {p}, corr {C[p]:.4f}; #latents |corr|>0.3: {(C.abs()>0.3).sum().item()}")
act = a[:, i] != 0
score = C[p].sign()*f[:, p]
print("score: act mean %.3f  inact mean %.3f  act p90 %.3f  inact p90 %.3f" %
      (score[act].mean(), score[~act].mean(), score[act].quantile(0.9), score[~act].quantile(0.9)))
ranks = score.argsort().argsort().float() + 1.0
n_act = int(act.sum())
auc = (ranks[act].sum().item() - n_act*(n_act+1)/2)/(n_act*(len(act)-n_act))
print("manual AUC:", round(auc, 4))
# extraction on active
cc = torch.corrcoef(torch.stack([score[act], a[act, i]]))[0,1]
print("extract corr:", cc.item())
