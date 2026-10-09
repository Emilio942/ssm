"""Debug: do GT features exist linearly in h at all? Compare direct probes vs SAE latents."""
import json, math, torch
from sim import make_decay, run_ssm
from sae import train_sae, greedy_match

device = "cuda"
gen = torch.Generator(device=device); gen.manual_seed(0)
torch.manual_seed(0)

N, D, alpha, L, B, S, mean_dt = 64, 64, 5.0, 1024, 32, 0.01, 0.1
M = int(alpha * N)
a = make_decay(N, "hetero", device, gen, 0.5, 2.0)
W_B = torch.randn(N, D, device=device, generator=gen) / math.sqrt(D)
Phi = torch.randn(D, M, device=device, generator=gen) / math.sqrt(D)

out = run_ssm(N, M, D, L, B, S, mean_dt, "iid", 1.0, 1.0, a, W_B, Phi, device, gen)
h, a_t = out["h"].float(), out["a_t"].float()
print("h stats: mean|%.4f| std|%.4f| norm|%.4f|" % (h.mean(), h.std(), h.norm(dim=1).mean()))

# --- Direct linear probe: ridge regression a_t -> h, per-feature R^2 ---
n = h.shape[0]
Xa = torch.cat([h, torch.ones(n, 1, device=device)], 1)
# solve (Xa^T Xa + lam I) W = Xa^T a_t on a subsample
idx = torch.randperm(n, device=device, generator=gen)[:16384]
Xs, As = Xa[idx], a_t[idx]
lam = 1e-2 * n
Wp = torch.linalg.solve(Xs.T @ Xs + lam * torch.eye(N + 1, device=device), Xs.T @ As)
pred = Xs @ Wp
resid = As - pred
r2 = 1 - resid.pow(2).sum(0) / (As.pow(2).sum(0) + 1e-12)
print("linear-probe R^2 per feature: mean %.3f, median %.3f, p90 %.3f" %
      (r2.mean(), r2.median(), r2.quantile(0.9)))

# --- probe via fresh-atom projection ---
v = W_B @ Phi  # (N,M)
proj = (h @ v)  # (n,M) correlation-ish score
# corr of a_{t,i} with proj
Ac = a_t - a_t.mean(0, keepdim=True); Pc = proj - proj.mean(0, keepdim=True)
Ac = Ac / Ac.std(0, keepdim=True).clamp_min(1e-6); Pc = Pc / Pc.std(0, keepdim=True).clamp_min(1e-6)
corr = (Ac.T @ Pc) / n
print("fresh-atom corr: mean %.3f max %.3f" % (corr.mean(), corr.max()))

# --- SAE latents ---
sae, hm, hs = train_sae(h, N, 512, device, gen, steps=3000, l1=1e-3)
Xs_ = (h[idx] - hm) / hs
with torch.no_grad():
    _, f = sae(Xs_)
f = f.float()
match, mc, C = greedy_match(As, f, device)
print("SAE: f active frac %.3f  mean f-sum %.2f  maxcorr %.3f  matched-corr mean %.3f" %
      ((f > 0).float().mean(), f.sum(1).mean(), C.max(1).values.mean(), mc.mean()))
# no-L1 variant
sae2, hm2, hs2 = train_sae(h, N, 512, device, gen, steps=3000, l1=0.0)
Xs2 = (h[idx] - hm2) / hs2
with torch.no_grad():
    _, f2 = sae2(Xs2)
f2 = f2.float()
match2, mc2, C2 = greedy_match(As, f2, device)
print("SAE no-L1: active frac %.3f  maxcorr %.3f  matched-corr mean %.3f" %
      ((f2 > 0).float().mean(), C2.max(1).values.mean(), mc2.mean()))
