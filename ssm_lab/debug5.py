import torch, math
from run_one import ridge_predict, weighted_r2, plain_r2

device = "cuda"
gen = torch.Generator(device=device); gen.manual_seed(0)
N, M, S, n = 64, 320, 0.01, 262144
V = torch.randn(N, M, device=device, generator=gen) / math.sqrt(N)
mask = torch.rand(n, M, device=device, generator=gen) < S
a = mask.float() * torch.randn(n, M, device=device, generator=gen)
h = a @ V.T
pred = ridge_predict(h, a, device)
print("STATIC control: weighted_r2 =", weighted_r2(pred, a), " plain_r2 =", plain_r2(pred, a))
