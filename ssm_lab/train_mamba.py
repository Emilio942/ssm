"""Train a small diagonal selective SSM on persistent-feature prediction,
then measure feature recoverability (probe/SAE) + geometry on the TRAINED system.

Task: x_t = Phi a_t with persistent features (switch process). Predict x_{t+1}
from h_t => the state must carry feature memory. Learnable: decay rates a
(softplus-parameterized, >0), W_B, W_delta (scalar Delta_t = softplus(w. x + b)),
readout W_out.
"""
import json, math, os, sys, time, argparse
import torch
import torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim import make_decay
from run_one import ridge_predict, weighted_r2
from sae import train_sae, evaluate_sae, nre as sae_nre


class SelectiveSSM(nn.Module):
    def __init__(self, N, D, a_init):
        super().__init__()
        self.N, self.D = N, D
        self.a_raw = nn.Parameter(torch.full((N,), float(torch.log(torch.expm1(torch.tensor(a_init))))))
        self.W_B = nn.Parameter(torch.randn(N, D) / math.sqrt(D))
        self.w_delta = nn.Parameter(torch.randn(D) / math.sqrt(D))
        self.b_delta = nn.Parameter(torch.tensor(-2.0))
        self.W_out = nn.Parameter(torch.zeros(D, N))
        self.b_out = nn.Parameter(torch.zeros(D))

    def forward(self, X):
        """X: (B, L, D). Returns h: (B, L, N), yhat: (B, L, D) predicting X shifted by 1."""
        a = torch.nn.functional.softplus(self.a_raw) + 1e-4
        B, L, _ = X.shape
        pre = X @ self.w_delta
        deltas = torch.nn.functional.softplus(pre + self.b_delta)      # (B,L)
        eda = torch.exp(-deltas.unsqueeze(-1) * a)
        bbar = (1.0 - eda) / a
        WX = X @ self.W_B.T
        h = torch.empty(B, L, self.N, device=X.device)
        hp = torch.zeros(B, self.N, device=X.device)
        for t in range(L):
            hp = eda[:, t] * hp + bbar[:, t] * WX[:, t]
            h[:, t] = hp
        yhat = h @ self.W_out.T + self.b_out
        return h, yhat, a, deltas


def gen_persistent_features(B, L, M, S, p_off, device, gen, flash=True):
    """flash=True: features flash for exactly ONE step (pure memory task).
    flash=False: switch process (turn-on w.p. S, mean duration 1/p_off)."""
    mask = torch.zeros(B, L, M, device=device)
    if flash:
        mask = (torch.rand(B, L, M, device=device, generator=gen) < S) * \
               torch.randn(B, L, M, device=device, generator=gen)
        return mask
    on = torch.zeros(B, M, device=device, dtype=torch.bool)
    val = torch.zeros(B, M, device=device)
    for t in range(L):
        turn_on = (torch.rand(B, M, device=device, generator=gen) < S) & ~on
        val = torch.where(turn_on, torch.randn(B, M, device=device, generator=gen), val)
        on = on | turn_on
        off = on & (torch.rand(B, M, device=device, generator=gen) < p_off)
        on = on & ~off
        mask[:, t] = on.float() * val
    return mask


def make_batch(B, L, M, S, p_off, Phi, device, gen):
    a_t = gen_persistent_features(B, L, M, S, p_off, device, gen)
    X = a_t @ Phi.T
    return a_t, X


def train_model(N, D, M, S, p_off, L, steps, lr, device, seed, Phi, delay=16):
    gen = torch.Generator(device=device); gen.manual_seed(seed)
    torch.manual_seed(seed)
    a_init = 1.0
    model = SelectiveSSM(N, D, a_init).to(device)
    W_tar = torch.randn(M, N, device=device) / math.sqrt(N)   # target readout (fixed)
    opt = torch.optim.Adam(list(model.parameters()) + [W_tar], lr=lr)
    fails = 0
    with torch.set_grad_enabled(True):
        for it in range(steps):
            try:
                a_t, X = make_batch(16, L, M, S, p_off, Phi, device, gen)
                h, _, a, deltas = model(X)
                tgt = a_t[:, :-delay]                         # flash at time t-delay
                yhat = h[:, delay:] @ W_tar.T                 # state at time t
                loss = (yhat - tgt).pow(2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                fails = 0
            except RuntimeError as e:
                fails += 1
                print(f"    [it {it}] runtime error ({e}); fail {fails}", flush=True)
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                if fails > 30:
                    raise
                continue
            if it % 500 == 0:
                print(f"    step {it}: loss {loss.item():.4f}", flush=True)
    del opt
    torch.cuda.empty_cache()
    return model


def evaluate_trained(model, N, D, M, S, p_off, L, Phi, device, gen, n_eval=128,
                     budget=200_000, delay=16):
    a_fixed = (torch.nn.functional.softplus(model.a_raw) + 1e-4).detach()
    eval_idx = torch.randperm(M, device=device, generator=gen)[:n_eval]
    hs, ls = [], []
    got = 0
    model.eval()
    with torch.no_grad():
        while got < budget:
            a_t, X = make_batch(16, L, M, S, p_off, Phi, device, gen)
            h, _, _, _ = model(X)                  # (16, L, N)
            hs.append(h.reshape(-1, N).cpu())
            ls.append(a_t.reshape(-1, M)[:, eval_idx].cpu())
            got += h.shape[0] * h.shape[1]
    H = torch.cat(hs)[:budget + 2048].to(device)
    A = torch.cat(ls)[:budget + 2048].float().to(device)
    lag = 8
    h = H[lag:]
    lab = A[:-lag]            # flash that happened `lag` steps before state h_t
    n = len(h)
    perm = torch.randperm(n, device=device, generator=gen)
    n_tr = int(0.8 * n)
    h_tr, h_va, lab_va = h[perm[:n_tr]], h[perm[n_tr:]], lab[perm[n_tr:]]
    res = {}
    pred = ridge_predict(h_va, lab_va, device)
    res["probe_r2_w"] = weighted_r2(pred, lab_va)
    P = int(min(4 * M + 128, 8192))
    sae, hm, hs_ = train_sae(h_tr, N, P, device, gen, steps=3000, topk=16)
    res["nre"] = sae_nre(sae, h_va, hm, hs_, device)
    with torch.no_grad():
        Fs = []
        for i in range(0, len(h_va), 8192):
            _, f = sae((h_va[i:i + 8192] - hm) / hs_)
            Fs.append(f)
        f_va = torch.cat(Fs, 0).float()
    ev = evaluate_sae(f_va, lab_va, device)
    res.update(ev)
    # geometry of trained system
    with torch.no_grad():
        a_t, X = make_batch(4, L, M, S, p_off, Phi, device, gen)
        _, _, _, deltas = model(X)
        deltas_list = deltas.reshape(-1)
    res["delta_mean"] = float(deltas_list.mean())
    res["delta_std"] = float(deltas_list.std())
    res["a_mean"] = float(a_fixed.mean())
    res["a_min"], res["a_max"] = float(a_fixed.min()), float(a_fixed.max())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/trained.jsonl")
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            try:
                done.add(json.loads(line)["sig"])
            except Exception:
                pass

    todo = []
    for M in [64, 128, 256, 640]:
        for seed in [0, 1]:
            todo.append(dict(M=M, seed=seed))
    todo.append(dict(M=128, seed=0, untrained=True))   # random-weight control
    todo.append(dict(M=256, seed=0, untrained=True))
    todo = [t for t in todo if json.dumps(t, sort_keys=True) not in done]
    if args.limit:
        todo = todo[:args.limit]

    N, D, L = 64, 64, 512
    S, p_off = 0.01, 0.02
    t0 = time.time()
    for cfg in todo:
        M, seed = cfg["M"], cfg["seed"]
        g = torch.Generator(device=device); g.manual_seed(seed)
        Phi = torch.randn(D, M, device=device, generator=g) / math.sqrt(D)
        print(f"=== M={M} seed={seed} untrained={cfg.get('untrained', False)}", flush=True)
        if cfg.get("untrained"):
            model = SelectiveSSM(N, D, 1.0).to(device)     # random init, no training
        else:
            model = train_model(N, D, M, S, p_off, L, args.steps, 3e-3, device, seed, Phi,
                                delay=cfg.get("delay", 16))
        res = evaluate_trained(model, N, D, M, S, p_off, L, Phi, device,
                               torch.Generator(device=device), budget=150_000)
        torch.cuda.empty_cache()
        res["sig"] = json.dumps(cfg, sort_keys=True)
        res["config"] = dict(cfg, N=N, D=D, L=L, S=S, p_off=p_off)
        with open(args.out, "a") as fh:
            fh.write(json.dumps(res) + "\n")
        print("   ", {k: round(v, 3) for k, v in res.items() if isinstance(v, float)},
              flush=True)
    print(f"done {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
