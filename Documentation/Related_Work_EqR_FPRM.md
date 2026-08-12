# Related Work Deep-Read: Equilibrium Reasoners & FPRM (2026-08-12)

**The two closest lines to ours, read completely; every borrowable mechanism
and every point of differentiation, documented for the scaling sessions.
Companions: Design_Ledger.md (our evidence), Research_Brainstorm.md
(clusters N-S).**

- EqR: "Equilibrium Reasoners" (arXiv:2605.21488) — attractor reasoning with
  depth/breadth scaling; Sudoku-Extreme 99.8% @ 5.03M params (D=64, B=128).
- FPRM: fixed-point reasoning with signal-propagation stabilizers
  (arXiv:2606.18206) — ARC-1 47.5% @ 7M, NO TTT, NO augmentation.

## 1. The convergences (independent confirmations of our line)

| Theirs | Ours | Note |
|---|---|---|
| EqR update z+= (1−λ)r(z)+βε, λ=0.05 hand-set | iterate_eq y+=η(p−y), η LEARNED = 0.058 | Same equation; our training discovered their hand-tuned damping — mutual validation |
| EqR SOT (segment, supervise, detach) | deep supervision + carry detach | already ours |
| FPRM damped iteration + residual halting τ=0.1 | our damped update + res_tau (built, unused in training) | adopt their τ discipline |
| EqR depth-before-breadth interaction (breadth pays only at D≥4) | T6 count dividend; depth-first design | matches |
| FPRM: convergence-residual tracks correctness AFTER landscape shaping | our E3-era stability-scoring failure was on the UNSHAPED substrate | retest on basin-trained substrates (below) |

## 2. What we take (each → a registered arm or upgrade)

1. **RI — randomized-initialization TRAINING rows (EqR's biggest lever:
   Maze 44.9→68.6 from RI alone).** z₀~N(0,σ₀) per trajectory in training →
   trained path-independence → multi-init breadth works at test. **We have
   never trained from anything but VOID starts — this is the missing
   mechanism behind cluster M's weak coverage** (our samplers explore an
   untrained init-distribution). → pretrain-13 arm: RI rows (the
   init-distribution sibling of our anchor rows). Prediction: multi-init
   coverage jumps from +2-3pp toward pool-level; kill: coverage flat ⇒
   path-dependence is architectural, not training-distributional.
2. **NI — per-step noise in TRAINING dynamics (β=0.01)** — distinct from
   our corrupt-target anchor INIT: they perturb every step; claims fewer
   spurious attractors. → pretrain-13 arm variant; instrument readout: does
   wrong-stable count (135/144 legacy) drop?
3. **FPRM coupled residual scaling (α₁,α₂ + Theorem-1 boundedness).** Their
   ablation: 83.4%→94.2% from initializing two scalars (α₁=.75, α₂=.25 —
   "initialize contractive"). We scale T at 5-75M; deep unrolls without
   their boundedness coupling is asking for the pre-norm divergence they
   document. → eq-core upgrade at pretrain-13, CI-gated (inertness at
   α₁=1,α₂ recovering our current graph).
4. **FPOpt adaptive damping at inference** (η decays on residual-plateau,
   γ≈.95-.98) — accuracy scales with iteration budget under decay. →
   inference upgrade; interacts with cluster Q (barrier spectroscopy sets
   the temperature; FPOpt sets the quench schedule).
5. **Truncated-Neumann implicit gradients (K-term)** — their alternative to
   full BPTT with error bound σ^K/(1−σ). Matters for T≥8 at 5-75M memory
   budgets. → DP-trainer option, tested against BPTT-equivalence at small K.
6. **Residual-based candidate selection retest** — EqR's Top-1-Converged
   works after shaping; our 08-08 rejection of stability-scoring predates
   basin training. → one battery: rank candidates by fixed-point residual
   on the record-radius substrate; compare vs PoE and vote.

## 3. What neither has (our defended ground, sharpened)

- **Basin measurement**: no retention, no corruption-ladder code-distance
  spectra, no radius/count laws, no family-transfer gates. Both INFER
  attractors from convergence; we MEASURE them. The four laws + packing
  frontier (cluster N) have no counterpart in either paper.
- **Information pricing**: no flux ledger, no throat, no S1/S2 physics. The
  count-vs-radius mechanistic split (dynamical capacity vs optimization
  under constraint) is invisible without our instruments.
- **The conversion stack**: FPRM reaches ARC-1 47.5% with NO TTT, NO
  augmentation, NO ensembling — pure dynamics. Our populations, PoE,
  snap-decoding, and (post-[H-12]) metric-aware adaptation sit ON TOP of
  that ceiling. EqR's breadth is our cluster-M with training support; their
  aggregation (majority/Top-1-residual) is strictly weaker than PoE.
- **Efficiency positioning**: both live at 5-7M; our laws were measured at
  78k-449k with the same phenomenology — the scaling sessions test whether
  the laws BRIDGE the two regimes (they should: the throat is
  task-determined).

## 4. Consolidated pretrain-13 design (all inputs now on record)

d-and-T scaled together (steps(d) calibrated from the 20k/40k pair), priced
per-scale/free-bits, C20 corpus + curriculum, **+ RI rows + NI arm +
coupled residual scaling (α₁,α₂) + FPOpt inference damping + residual-
selection battery**, on the v6e pod (DP trainer; Neumann-K if T≥8). Convert
phase order: cluster S (all-eq self-decode with RI-trained multi-init
candidates + PoE) → metric-aware TTT (cluster P gates it) → eval-6 (PI).
