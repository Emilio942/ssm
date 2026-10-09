import torch, math
from run_one import ridge_predict
device = "cuda"
gen = torch.Generator(device=device); gen.manual_seed(0)
N, M, S, n = 64, 320, 0.01, 262144
V = torch.randn(N, M, device=device, generator=gen) / math.sqrt(N)
mask = torch.rand(n, M, device=device, generator=gen) < S
a = mask.float() * torch.randn(n, M, device=device, generator=gen)
h = a @ V.T
pred = ridge_predict(h, a, device)
i = 7
act = a[:, i] != 0
aa, pp = a[act, i], pred[act, i]
print("n_active:", int(act.sum()), " a.std %.3f  p.std %.3f" % (aa.std(), pp.std()))
c = ((aa-aa.mean())*(pp-pp.mean())).sum()/max(pp.var().item(),1e-12)
print("fitted c = %.3f  -> resid std %.3f (a std %.3f)" % (c, (aa-c*pp).std(), aa.std()))
