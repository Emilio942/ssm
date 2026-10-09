import torch, math
device = "cuda"
gen = torch.Generator(device=device); gen.manual_seed(0)
N, M, S, n = 64, 320, 0.01, 262144
V = torch.randn(N, M, device=device, generator=gen) / math.sqrt(N)
mask = torch.rand(n, M, device=device, generator=gen) < S
a = mask.float() * torch.randn(n, M, device=device, generator=gen)
h = a @ V.T
i = 7
vhat = V[:, i] / V[:, i].norm()
pre = h @ vhat                                    # oracle reading direction
ai = a[:, i].abs()
act = ai > 0
def corr(x, y):
    x = x - x.mean(); y = y - y.mean()
    return (x*y).mean()/(x.std()*y.std())
print("oracle corr(pre, |a_i|) all:", corr(pre, ai).item())
print("oracle corr(pre, |a_i|) active-only:", corr(pre[act], ai[act]).item())
print("oracle corr(pre, a_i) active-only:", corr(pre[act], a[act, i]).item())
# with interference from concurrent features removed? count others
k = mask.sum(1)
print("mean others when i active:", (k[act]-1).float().mean().item())
