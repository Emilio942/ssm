import torch, math
device = "cuda"
gen = torch.Generator(device=device); gen.manual_seed(0)
N, M, S, n = 64, 320, 0.01, 100000
V = torch.randn(N, M, device=device, generator=gen) / math.sqrt(N)
mask = torch.rand(n, M, device=device, generator=gen) < S
a = mask.float() * torch.randn(n, M, device=device, generator=gen)
h = a @ V.T
# exact min-norm solution per row: a_hat = a V^T (V V^T)^{-1} V = P a
G = V @ V.T                                   # (N,N)
P = V.T @ torch.linalg.solve(G + 1e-9*torch.eye(N, device=device), V)   # (M,M) projector
Ap = a @ P.T                                  # min-norm predictions
act = a != 0
sig = Ap[act].std(); noise = (Ap - a)[act].std()
print(f"oracle pinv: conditional signal std {sig:.3f}, residual std {noise:.3f}, "
      f"R2_w ~ {1-(noise/sig)**2:.3f}")
k = mask.sum(1).float()
print("active per row: mean %.2f" % k.mean())
# residual of feature i vs number of OTHER active features in same row
i = 5
oth = (k - mask[:, i].float())
for lo, hi in [(0,1),(1,3),(3,6),(6,12)]:
    sel = act[:, i] & (oth >= lo) & (oth < hi)
    if sel.sum() > 10:
        print(f"  n_other in [{lo},{hi}): resid std { (Ap-a)[sel, i].std():.3f}  (n={int(sel.sum())})")
