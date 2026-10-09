import json, math, torch
from sim import make_decay, run_ssm
from sae import train_sae, greedy_match

device = "cuda"
def probe(S, alpha, mean_dt, spectrum="hetero", mode="iid", seed=0):
    gen = torch.Generator(device=device); gen.manual_seed(seed); torch.manual_seed(seed)
    N, D, L, B = 64, 64, 1024, 32
    M = int(alpha * N)
    a = make_decay(N, spectrum, device, gen, 0.5, 2.0)
    W_B = torch.randn(N, D, device=device, generator=gen) / math.sqrt(D)
    Phi = torch.randn(D, M, device=device, generator=gen) / math.sqrt(D)
    out = run_ssm(N, M, D, L, B, S, mean_dt, mode, 1.0, 1.0, a, W_B, Phi, device, gen)
    h, a_t = out["h"].float(), out["a_t"].float()
    n = h.shape[0]
    idx = torch.randperm(n, device=device, generator=gen)[:16384]
    Xa = torch.cat([h, torch.ones(n,1,device=device)],1)[idx]
    As = a_t[idx]
    Wp = torch.linalg.solve(Xa.T@Xa + 1e-2*n*torch.eye(N+1,device=device), Xa.T@As)
    r2 = 1 - (As-Xa@Wp).pow(2).sum(0)/(As.pow(2).sum(0)+1e-12)
    sae, hm, hs = train_sae(h, N, 512, device, gen, steps=2000, l1=1e-3)
    with torch.no_grad():
        _, f = sae((h[idx]-hm)/hs)
    match, mc, C = greedy_match(As, f.float(), device)
    print(f"S={S:<8} a={alpha:<4} dt={mean_dt:<7} alive~{S*M/(2*1.25*mean_dt):8.2f}  "
          f"probeR2={r2.mean():.3f}  maxcorr={C.max(1).values.mean():.3f}")

probe(1e-4, 5, 0.1)
probe(1e-3, 5, 0.1)
probe(1e-2, 5, 0.1)
probe(1e-3, 5, 1.0)
probe(1e-2, 5, 1.0)
probe(1e-3, 20, 0.1)
