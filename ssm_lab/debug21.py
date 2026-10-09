import math, torch
from sim import make_decay, run_ssm
from run_one import ridge_predict, weighted_r2
device = "cuda"
gen = torch.Generator(device=device); gen.manual_seed(0)
N, D, L, M, S, dt = 64, 64, 2048, 640, 1e-4, 1.0
a = make_decay(N, "hetero", device, gen)
W_B = torch.randn(N, D, device=device, generator=gen) / math.sqrt(D)
Phi = torch.randn(D, M, device=device, generator=gen) / math.sqrt(D)
eval_idx = torch.randperm(M, device=device, generator=gen)[:128]
out = run_ssm(N, M, D, L, 32, S, dt, "const", 0.5, 1.0, a, W_B, Phi, device, gen,
              eval_idx=eval_idx, dt_clip=[0.25,4.0])
h, lab = out["h"].float(), out["a_t"].float()
print("||h|| mean %.4f  frac exactly-zero rows %.3f" % (h.norm(dim=1).mean(), (h.norm(dim=1)<1e-7).float().mean()))
n_act = (lab!=0).sum(0)
print("actives per eval feature: min %d med %.0f" % (n_act.min(), n_act.float().median()))
pred = ridge_predict(h, lab, device)
r2w = weighted_r2(pred, lab)
print("probe R2_w:", round(r2w, 3))
# single-feature manual check
i = int(torch.argmax(n_act))
v = (W_B @ Phi[:, eval_idx[i]]); v = v/v.norm()
pr = h @ v
act = lab[:, i] != 0
cc = torch.corrcoef(torch.stack([pr[act], lab[act, i]]))[0,1]
print(f"feature {i}: manual corr(atom-projection, a|active) = {cc:.3f}, n_act={int(act.sum())}")
# residual of regression for that feature
resid = (lab[act, i] - pred[act, i])
print("resid std %.3f vs target std %.3f" % (resid.std(), lab[act,i].std()))
