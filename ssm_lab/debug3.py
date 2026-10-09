import math, torch
from sim import make_decay, run_ssm
from sae import train_sae, greedy_match
from run_one import ridge_probe

device = "cuda"
def probe(S, alpha, mean_dt, spectrum="hetero", mode="iid", seed=0, budget=400_000):
    gen = torch.Generator(device=device); gen.manual_seed(seed); torch.manual_seed(seed)
    N, D, L = 64, 64, 2048
    M = int(alpha * N)
    a = make_decay(N, spectrum, device, gen, 0.5, 2.0)
    W_B = torch.randn(N, D, device=device, generator=gen) / math.sqrt(D)
    Phi = torch.randn(D, M, device=device, generator=gen) / math.sqrt(D)
    eval_idx = torch.randperm(M, device=device, generator=gen)[:min(M,128)]
    hs, ls = [], []
    got = 0
    while got < budget:
        out = run_ssm(N, M, D, L, 32, S, mean_dt, mode, 1.0, 1.0, a, W_B, Phi, device, gen, eval_idx=eval_idx)
        hs.append(out["h"]); ls.append(out["a_t"]); got += out["h"].shape[0]
    h, lab = torch.cat(hs)[:budget], torch.cat(ls)[:budget].float()
    alive = S*M/(2*1.25*mean_dt)
    pm, pmed = ridge_probe(h, lab, device)
    n = h.shape[0]
    idx = torch.randperm(n, device=device, generator=gen)[:16384]
    sae, hm, hs2 = train_sae(h, N, 512, device, gen, steps=3000, l1=1e-3)
    with torch.no_grad(): _, f = sae((h[idx]-hm)/hs2)
    match, mc, C = greedy_match(lab[idx], f.float(), device)
    print(f"S={S:<8} a={alpha:<4} dt={mean_dt:<7} alive~{alive:9.2f}  probeR2={pm:.3f}  "
          f"maxcorr={C.max(1).values.mean():.3f}  matchcorr={mc.mean():.3f}")

probe(1e-4, 5, 0.1)
probe(1e-3, 5, 0.1)
probe(1e-2, 5, 0.1)
probe(1e-3, 5, 1.0)
probe(1e-2, 5, 1.0)
probe(3e-2, 5, 1.0)
probe(1e-1, 5, 1.0)
