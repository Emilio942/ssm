"""Honest (out-of-sample) probe vs in-sample probe on matched configs."""
import sys, torch
import numpy as np
sys.path.insert(0, "ssm_lab")
from analyze import load, flat_cols
from sim import make_decay, run_ssm
device = "cuda"

df = flat_cols(load("results/phaseB.jsonl"))
big = df[(df["config.budget"]==1_200_000) & (df["config.seed"]==0)]
print("configs:", len(big))

def ridge_fit_pred(h_tr, a_tr, h_va, N, lam_rel=1e-3):
    Xtr = torch.cat([h_tr, torch.ones(len(h_tr),1,device=device)],1)
    reg = lam_rel*(Xtr.T@Xtr).diagonal().mean()*torch.eye(N+1,device=device)
    W = torch.linalg.solve(Xtr.T@Xtr+reg, Xtr.T@a_tr)
    Xva = torch.cat([h_va, torch.ones(len(h_va),1,device=device)],1)
    return Xva @ W

def wr2(pred, targets):
    num=den=0.0
    for i in range(targets.shape[1]):
        act = targets[:,i]!=0
        if act.sum()<5: continue
        a,p = targets[act,i], pred[act,i]
        c = ((a-a.mean())*(p-p.mean())).mean()/max(p.var().item(),1e-12)
        p = p*c
        num += (a-p).pow(2).sum().item(); den += a.pow(2).sum().item()
    return max(0.0, 1-num/max(den,1e-12))

res = []
for _, r in big.iterrows():
    N = int(r["config.N"]); M = int(r["config.M"]); D = 64
    gen = torch.Generator(device=device); gen.manual_seed(int(r["config.seed"])+999)
    a = make_decay(N, r["config.spectrum"], device, gen)
    W_B = torch.randn(N, D, device=device, generator=gen)/np.sqrt(D)
    Phi = torch.randn(D, M, device=device, generator=gen)/np.sqrt(D)
    eval_idx = torch.randperm(M, device=device, generator=gen)[:128]
    hs, ls = [], []
    got = 0
    while got < 400_000:  # same 400k as main runs
        out = run_ssm(N, M, D, int(r["config.L"]), 32, r["config.S"], r["config.mean_dt"],
                      r["config.delta_mode"], 0.5, 1.0, a, W_B, Phi, device, gen,
                      eval_idx=eval_idx, dt_clip=[0.25,4.0])
        hs.append(out["h"]); ls.append(out["a_t"]); got += out["h"].shape[0]
    h = torch.cat(hs)[:400_000]; lab = torch.cat(ls)[:400_000].float()
    n = len(h); perm = torch.randperm(n, device=device, generator=gen)
    n_tr = int(0.8*n)
    # BLOCKED split: train = first 80% of sequences within each chunk (row blocks)
    # simple: use random half of SEQUENCES: rows are chunk-major (B,L)
    B, Ltot = 32, int(r["config.L"])
    hb = h.view(-1, B, Ltot, N) if False else None
    # sequence-blocked split: keep rows of whole sequences
    seq_tr = torch.zeros(B, dtype=torch.bool); seq_tr[:int(0.8*B)] = True
    mask_tr = seq_tr.repeat(Ltot)  # wrong ordering: h is (B*L) chunk-major => (b,l)
    # reconstruct sequence index per row
    seq_idx = torch.arange(n, device=device) % Ltot  # position within seq
    seq_id = (torch.arange(n, device=device) // Ltot) % B
    tr_m = torch.isin(seq_id, torch.where(seq_tr)[0].to(seq_id.device))
    pred_oos = ridge_fit_pred(h[tr_m], lab[tr_m], h[~tr_m], N)
    r2_oos = wr2(pred_oos, lab[~tr_m])
    pred_ins = ridge_fit_pred(h[~tr_m], lab[~tr_m], h[~tr_m], N)
    r2_ins = wr2(pred_ins, lab[~tr_m])
    res.append((r["probe_r2_w"], r2_ins, r2_oos))
    print(f"S={r['config.S']:<7} dt={r['config.mean_dt']:<5} a={r['config.alpha']:<3} "
          f"{r['config.delta_mode']:9s} sweep={r['probe_r2_w']:.3f} insample={r2_ins:.3f} "
          f"blocked-OOS={r2_oos:.3f}")
res = np.array(res)
print(f"\nmean: sweep(in-sample 80k)={res[:,0].mean():.3f}  in-sample(48k)={res[:,1].mean():.3f}  "
      f"sequence-blocked OOS={res[:,2].mean():.3f}")
print(f"optimism (in-sample - OOS): {np.mean(res[:,1]-res[:,2]):+.4f}")
