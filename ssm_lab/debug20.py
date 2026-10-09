import json, math, torch
from sim import make_decay, run_ssm, sample_deltas, calibrate_selective_bias
from run_one import ridge_predict, weighted_r2, collect_data
device = "cuda"
gen = torch.Generator(device=device); gen.manual_seed(0)
N, D, L, M, S, dt = 64, 64, 2048, 128, 1e-4, 0.003
a = make_decay(N, "homo", device, gen)
W_B = torch.randn(N, D, device=device, generator=gen) / math.sqrt(D)
Phi = torch.randn(D, M, device=device, generator=gen) / math.sqrt(D)
eval_idx = torch.arange(M)
h, lab, tr, ds = collect_data(dict(N=N,D=D,L=L,batch=32,budget=400000,S=S,mean_dt=dt,
    delta_mode="selective",dt_cv=1.0,g=1.0), a, W_B, Phi, M, device, gen, eval_idx)
print("realized delta:", ds)
print("h norm:", h.norm(dim=1).mean().item(), " h std:", h.std().item())
print("label actives per feat:", (lab!=0).float().sum(0)[:10])
pred = ridge_predict(h_va := h, lab.float(), device)
print("probe_r2_w:", weighted_r2(pred, lab.float()))
