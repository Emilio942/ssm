# Feature Superposition in Selective State Space Models — Theory, Phase Diagrams & Trained-Model Validation

Empirical research program on **mathematical capacity bounds and the geometry of
feature superposition in selective state space models** (Mamba/S6-type SSMs with
input-dependent timescales Δ_t), in the spirit of mechanistic interpretability.

This repository contains the full experimental apparatus — simulation core,
sparse-autoencoder evaluation, sweep drivers, analysis and theory-verification
scripts — plus all raw results (~4,100 runs) and figures produced in an
autonomous research session on a single RTX 3060.

---

## The research question

> How does input-dependent selectivity Δ_t modify the phase-transition boundary
> of feature superposition capacity in selective SSMs, and what is the
> hyper-parameter domain (the "sweet spot") within which temporally accumulated
> features remain linearly disentangleable by sparse autoencoders (SAEs)?

**Setup.** Sparse ground-truth features `a_t ~ Bernoulli(S)·N(0,1)` are embedded
through a random dictionary `Φ` (`x_t = Φa_t`) and injected into a diagonal
selective SSM with exact zero-order-hold discretization,

```
h_t = exp(−Δ_t ⊙ a) ⊙ h_{t−1} + ((1 − exp(−Δ_t ⊙ a))/a) ⊙ (W_B x_t),
Δ_t = softplus(w_Δ · x_t + b_Δ)
```

Each feature injected at step k reaches step t through the warped atom
`w_j(d) = (b̄ ⊙ v_j) ⊙ e^{−a·T_{k,t}}` with the cumulative decay path
`T_{k,t} = Σ Δ_j` — the temporal analogue of superposition's dictionary warp.
A TopK-SAE and a matched linear probe then attempt to recover the features from
`h_t`.

## Headline results

1. **Static-interference theorem (exact, verified to 3–5 %).** For constant Δ
   the system is an exact static mixing, and the recoverability of feature i
   obeys
   `y := (1−R²)/R² = S · Σ_{(j,d)≠(i,0)} c_j(d)² / c_i(0)²`,
   with `c_j(d) = w_i(0)ᵀ G⁻¹ w_j(d)`, `G = Σ_d W_d W_dᵀ`.
   Closed form for homogeneous decay: **y = S·α/(1−q²) − S**, q = e^{−Δ}.
   Verified over 340 configs: median measured/theory = 0.95–0.99,
   corr(log) = 0.98. Direction warping needs no separate "amplification" — it
   is inside G. Deviations from the theorem directly measure non-constant Δ.

2. **The collapse law with a variability multiplier and a coupling floor.**
   With i.i.d. Δ_t (moments matched) the theorem still explains the data up to
   a factor ≈ 1.5 [1.07, 5.12]. With content-correlated Δ_t an extraction noise
   floor appears that rises ~linearly with the Δ–content correlation g
   (y₀: 0.245 → 0.520 for g = 0 → 1) and caps peak recoverability (R² ≲ 0.7).

3. **Where the boundary sits.** The R² = 1/2 crossing sits at the coherent
   crowding variable `c = S·(α/N)·A² ≈ 0.03` (A = coherent lag-sum of atom
   energy × direction persistence, measured per run) — i.e. collapse begins at
   **O(1) concurrently-alive features**, ~30× earlier than the isotropic-√N
   estimate, because same-interferer echoes accumulate coherently across lags.

4. **What does *not* matter.** Direction warping per se (heterogeneous decay
   spectra) is benign — paired Δ = +0.018 in favor of heterogeneity. Sequence
   depth L is irrelevant once L ≫ τ_eff. SAEs reach the linear-probe bound when
   given fair dictionary capacity (P ≈ 4M latents, TopK).

5. **Trained SSMs reproduce the law end-to-end.** Small SSMs trained on a
   flash-recall task (16-step delay) learn to slow their decay ~4× to extend
   the memory window to the task horizon — but only while crowding allows
   (learned τ: 34 → 8.6 steps as M grows 64 → 640). Gradient descent moves
   *along* the law's iso-quality curve, not around it. At intermediate crowding
   a heterogeneous decay spectrum emerges spontaneously.

**Practical sweet-spot recipe:** keep the alive-trace mass
`ρ_alive = S·M·τ_eff ≲ 2–3` (at N = 64), decouple Δ_t from feature content (or
regularize the coupling), use heterogeneous decay spectra freely, and give the
SAE ≈ 4M latents with k ≈ 1.5·ρ_alive.

## Metric methodology (why this was non-trivial)

Four subtle measurement traps had to be fixed before the physics was visible
(each documented in `RESEARCH_LOG.md`):

- fixed-threshold FRR is **prior-capped** (perfect extraction yields FRR ≈ 13 %
  at S = 1e-3) → replaced by active-weighted R² and detection AUC;
- underdetermined linear maps shrink predictions by ≈ N/M → per-feature optimal
  rescaling (validated against the analytic static control: 0.960 vs 0.96);
- signed features need **twin-pair** latent readouts (`s₁f_p1 + s₂f_p2`);
- dead latents produce NaN correlations that poison greedy matching.

## Repository layout

```
ssm_lab/
  sim.py           SSM simulation + exact geometry metrics (atoms, CTIM, τ_eff)
  sae.py           TopK-SAE, sign-robust evaluation (AUC, twin-pair readout)
  run_one.py       one config → one JSON line (probe, SAE, geometry)
  grid*.py         sweep drivers (main grid, controls, scaling, replication)
  analyze.py       phase diagrams, collapse curves, paired comparisons
  theory_check.py  exact static-theorem verification (vectorized projector)
  lawfit*.py       empirical law fitting
  train_mamba.py   trainable SSM (flash-recall task) + end-to-end evaluation
  honest_probe.py  blocked out-of-sample probe audit
results/
  *.jsonl          ~4,100 raw runs (main, controls, scaling, replication, theory)
  figs/            phase diagrams, collapse laws, decomposition, sweet spots
RESEARCH_LOG.md    full narrative log: methodology, traps, findings, theorem
ssm.md             original research proposal + hypothesis-verification history
```

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy scipy pandas matplotlib torch --index-url https://download.pytorch.org/whl/cu124

# single experiment
python ssm_lab/run_one.py --config config.json --out results/my.jsonl

# full main sweep (resumable; skips completed signatures)
python ssm_lab/grid.py --out results/results.jsonl --seeds 3

# verify the static-interference theorem yourself
python ssm_lab/theory_check.py

# train an SSM end-to-end on the flash-recall task
python ssm_lab/train_mamba.py --steps 2500
```

Reference hardware: RTX 3060 (12 GB); every sweep in this repo ran on it.
A full main-grid sweep (840 configs × 3 seeds) takes ≈ 4.3 h.

## Reproducibility notes

- All sweeps are signature-resumable and seed-explicit; headline numbers are
  3-seed replicated (per-config seed std of probe R²: median 0.005).
- The probe was audited against sequence-blocked out-of-sample evaluation
  (in-sample optimism ≈ +0.005 — sweep numbers are honest).
- Δ_t modes are moment-matched (mean and cv) by 2-parameter bisection, so
  mode comparisons isolate the dependence structure, not the marginals.

## License

See LICENSE.
