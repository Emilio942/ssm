import torch, math
from run_one import ridge_predict, weighted_r2

device = "cuda"
gen = torch.Generator(device=device); gen.manual_seed(0)
N, M, S, n = 64, 320, 0.01, 262144
V = torch.randn(N, M, device=device, generator=gen) / math.sqrt(N)
mask = torch.rand(n, M, device=device, generator=gen) < S
a = mask.float() * torch.randn(n, M, device=device, generator=gen)
h = a @ V.T

# exact projection predictions
G = V @ V.T
P = V.T @ torch.linalg.solve(G + 1e-9*torch.eye(N, device=device), V)
ap = a @ P.T
print("exact-pinv weighted_r2:", weighted_r2(ap, a))

# per-feature rescaled pinv
scale = 1.0 / P.diagonal().clamp_min(1e-6)
print("rescaled  weighted_r2:", weighted_r2(ap * scale, a))

# ridge pipeline
pred = ridge_predict(h, a, device)
print("ridge     weighted_r2:", weighted_r2(pred, a))

# correlation check on one feature
i = 7
sel = a[:, i] != 0
print("feat7: P_ii=%.3f  corr(pred,a|act)=%.3f" % (P[i,i],
      torch.corrcoef(torch.stack([pred[sel,i], a[sel,i]]))[0,1]))
