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
i = 7
C = torch.stack([((f[:,p]-f[:,p].mean())*(a[:,i]-a[:,i].mean())).mean()/(f[:,p].std()*a[:,i].std()) for p in range(0, 2048, 1)])
absC = C.abs()
p = int(absC.argmax()); s = 1.0 if C[p] >= 0 else -1.0
print(f"best latent {p}, corr {C[p]:.3f}")
score = s*f[:, p]
act = a[:, i] != 0
print("score mean active %.3f, inactive %.3f" % (score[act].mean(), score[~act].mean()))
ranks = score.argsort().argsort().float() + 1.0
n_act = int(act.sum()); n_ina = (~act).sum()
auc = (ranks[act].sum().item() - n_act*(n_act+1)/2)/(n_act*n_ina)
print("manual AUC:", auc)
