import math, torch
from sim import make_decay, run_ssm

device = "cuda"
gen = torch.Generator(device=device); gen.manual_seed(0); torch.manual_seed(0)
N, D, L, B, S = 64, 64, 1024, 32, 1e-3
M = 320
a = make_decay(N, "homo", device, gen)          # all decay = 1
W_B = torch.randn(N, D, device=device, generator=gen) / math.sqrt(D)
Phi = torch.randn(D, M, device=device, generator=gen) / math.sqrt(D)

out = run_ssm(N, M, D, L, B, S, 0.1, "iid", 0.0, 1.0, a, W_B, Phi, device, gen, label_dtype=torch.float32)
h, a_t = out["h"], out["a_t"].float()
print("h norm mean %.4f" % h.norm(dim=1).mean())

V = W_B @ Phi   # (N, M) static mixing matrix
# static model check: h_t should be ~ V @ gamma_t  with gamma_t = sum_k c(t-k) a_{k,i}
# Recover gamma by least squares on V (M >> N, use ridge on gram of columns)
G = V.T @ V
n = h.shape[0]
idx = torch.randperm(n, device=device, generator=gen)[:8192]
lam = 1e-6 * G.diagonal().mean()
gam = torch.linalg.solve(G + lam*torch.eye(M, device=device), V.T @ h[idx].T).T   # (n, M)
Ac = a_t[idx]
# correlation between gamma (recovered accumulated activity) and TRUE accumulated activity
# true accumulated: convolve a_t with decay kernel c(d)=exp(-d*0.1*1)
c = torch.exp(-torch.arange(200, device=device)*0.1)
acc = torch.zeros_like(Ac)
for d in range(1, 200):
    acc[d:] += c[d]*Ac[:-d]
g1 = gam[:, :64]; t1 = acc[:, :64]
corr = ((g1-g1.mean(0))*(t1-t1.mean(0))).sum(0) / (g1.std(0)*t1.std(0)*g1.shape[0]+1e-12)
print("corr(recovered gamma, true accumulated activity): mean %.3f" % corr.mean())

# direct: does fresh activity correlate with h projection?
proj = h[idx] @ V[:, :64]
fresh = Ac[:, :64]
cc = ((proj-proj.mean(0))*(fresh-fresh.mean(0))).sum(0)/(proj.std(0)*fresh.std(0)*proj.shape[0])
print("corr(<h,v_i>, fresh a_i): mean %.4f  max %.3f" % (cc.mean(), cc.max()))

# rank/energy check on V
sv = torch.linalg.svdvals(V)
print("V singular values: max %.3f min %.3f  (cond %.1f)" % (sv[0], sv[-1], sv[0]/sv[-1]))
