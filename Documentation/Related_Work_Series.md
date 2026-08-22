# Related Work Deep-Read: the series HRM → TRM → EqR → FPRM (2026-08-21)

**The field's evolution read as ONE line, because that is how it happened (PI directive, course
correction #3): each paper inherits the previous one's object (recursive/recurrent reasoning on
tiny-data puzzle benchmarks), changes one or two mechanisms, and — the thread that matters for
paper 1 — takes a different stance on FIXED POINTS. HRM assumes them; TRM shows they are not reached
and drops the assumption; EqR and FPRM build them for real (stabilized attractors) and scale test-time
depth/breadth on top; we MEASURE them (basins, reachability, code geometry, priced information) and
show on which landscape class each mechanism pays. Companions: `Related_Work_EqR_FPRM.md` (the EqR/FPRM
deep read, 2026-08-12 — not repeated here beyond the series table), `Design_Ledger.md`.**

Sources: HRM arXiv:2506.21734 (full text), TRM arXiv:2510.04871 (full text + repo README), EqR
arXiv:2605.21488 and FPRM arXiv:2606.18206 (per the 08-12 deep read), HRM/TRM repos for the
Sudoku-Extreme protocol (read 2026-08-21). Numbers below are quoted from those sources; items marked
⚠ must be re-verified against the PDFs at writing time.

---

## 1. HRM — Hierarchical Reasoning Model (Wang et al., Jun 2025; 27M params)

**Object.** Two recurrent modules — a high-level H (slow, "abstract planning") and a low-level L
(fast, "detailed computation") — run N high-level cycles of T low-level timesteps (forward = N×T
steps); transformer-encoder blocks (RoPE, GLU, RMSNorm); a learnable task token prepended; initial
hidden states sampled from a truncated normal (σ=1, trunc 2).

**The fixed-point stance (the load-bearing assumption).** Training avoids BPTT through the recursion
with a *one-step gradient*: "the gradient of the last state of each module, treating other states as
constant" — justified by the implicit function theorem under the assumption that L "converges to a
local fixed point z*_L during each cycle" (and (I−J)^−1 ≈ I, a truncated Neumann series). The paper is
explicit that this is idealized: "the actual HRM does not require L to fully converge — the one-step
gradient approximation is applied after T timesteps regardless." Plus deep supervision across segments
(state detached between segments) and ACT halting learned by Q-learning (halt/continue head; M_min/M_max
with stochastic M_min).

**Data regime (the benchmark convention the series inherits).** ~1,000 training examples per task,
no pretraining, no CoT; Sudoku augmented by band and digit permutations; ARC by translations /
rotations / flips / color permutations with 1000 augmented variants per test input and a 2-prediction
vote; exact-match scoring; Sudoku/Maze single inference pass.

**Reported.** ARC-AGI-1 40.3% (vs o3-mini-high 34.5%, Claude 3.7 8K 21.2%); Sudoku-Extreme "near-perfect"
and Maze-Hard "~100%" in the paper's own framing ⚠ (see the discrepancy note in §2); direct-prediction
transformer baselines 0% on Sudoku-Extreme(1k) / 16.9% on Sudoku-Extreme-Full. **Inference-time
scaling: trained with M_max=8, gains continue at M_max=16 on Sudoku — and "extra computational
resources yield minimal gains" on ARC-AGI.** That last sentence is our landscape-class dissociation
observed inside the series itself (depth pays on CSP, not on ARC) — without an explanation; ours
explains it (basins absent vs present; depth-limited vs inventory-limited). Also: a
participation-ratio "dimensionality hierarchy" (PR z_L 30.2 vs z_H 90.0) offered as a biological
analogy; forward residuals "remain high over many steps" (their own Figure 3 — the L-module
"repeatedly converges within cycles before being reset by H").

## 2. TRM — Tiny Recursive Model (Jolicoeur-Martineau, Oct 2025; 7M / 5M params)

**Object.** ONE tiny network (2 layers) recursing on (x, y, z): y = the current answer, z = the latent
reasoning state; n inner recursions (n=6 vs HRM's effective 2) and T supervision steps (T=3) for up to
N_sup=16 improvement steps; MLP-token-mixing variant for Sudoku (5M), attention variant for Maze/ARC (7M).

**What it removed and why — the series' first self-correction.** (i) the hierarchy and the two
networks ("both tasks can be replaced by a single network … better generalization": 82.4→87.4%);
(ii) **the one-step/IFT gradient: "there is no guarantee that a fixed-point is reached … HRM is only
doing 4 recursions before stopping to apply the one-step approximation"; empirically "the residual for
z_H is clearly well above 0 at every step" and "z_L is very far from converged after one f_L evaluation at
T cycles, which is when the fixed-point is assumed to be reached"** — TRM backpropagates through the
full n+1 recursions instead; the one-step gradient alone costs **87.4 → 56.5%** (its largest ablation);
(iii) 4 layers → 2 ("smaller networks are better" with recursion; overfitting on small data: 79.5 vs 87.4);
(iv) ACT simplified (no Q-learning continue pass; −1.3 with ACT); (v) EMA added (79.9 → 87.4%); (vi) the
biological narrative replaced by "z = reasoning, y = answer". Other ablations: T=2,n=2 73.7%;
self-attention on Sudoku 74.7% (MLP 87.4%).

**Data.** Sudoku-Extreme 1,000 examples × 1000 shuffling augmentations; Maze-Hard 1,000 mazes × 8
dihedral; ARC-AGI-1 800 tasks / ARC-AGI-2 1,120 tasks × 1000 augmentations (color permutation,
dihedral, translations). Sudoku-Extreme build: `--subsample-size 1000 --num-aug 1000`; evaluation = exact
accuracy on the full test set (repo: "around 87% (±2) MLP-T, around 75% (±2) attention").

**Reported (TRM Tables 4–5).** Sudoku-Extreme: **HRM 55.0% · TRM-MLP 87.4% · TRM-Att 74.7%** (R1 0,
o3-mini-high 0); Maze-Hard: HRM 74.5 · TRM-Att 85.3; ARC-AGI-1: HRM 40.3 · TRM-Att **44.6** (R1 15.8,
o3-mini-high 34.5, Gemini 2.5 Pro 37.0); ARC-AGI-2: HRM 5.0 · TRM 7.8. ⚠ **Discrepancy to resolve at
writing:** HRM's paper describes Sudoku-Extreme as near-perfect; TRM's Table 4 lists HRM at 55.0% under
the 1k-example protocol — check TRM's text for the provenance of the HRM row (re-run vs reported) before
quoting either as "HRM's number"; our comparisons key off TRM's protocol, which is the one the benchmark
convention follows.

## 3. EqR — Equilibrium Reasoners (May 2026; 5.03M) and 4. FPRM (Jun 2026; 7M) — the series builds the fixed points for real

From the 08-12 deep read (full notes there): **EqR** makes the equilibrium genuine — the update is
*our* damped iterate (z += (1−λ)r(z)+βε with λ=0.05 hand-set; our learned η=0.058), trained with
segment-supervise-detach, **randomized-init training rows** (RI; Maze 44.9→68.6, their biggest lever) and
**per-step noise** (NI, β=.01, "fewer spurious attractors"), then scales **depth D** and **breadth B
(multi-init)** at test time: Sudoku-Extreme **99.8% @ D=64, B=128**; depth-before-breadth (breadth pays only
at D≥4). **FPRM** supplies the stabilizer theory for deep unrolls — pre-norm, coupled residual scaling
(α₁,α₂ with a boundedness theorem; "initialize contractive" 83.4→94.2), damped iteration, residual
halting τ=0.1, FPOpt adaptive inference damping (γ≈.95–.98), truncated-Neumann K-term gradients — and
posts **ARC-1 47.5% @ 7M with no TTT, no augmentation, no ensembling**: stabilized genuine equilibria beat
TRM at the same scale, pure dynamics.

---

## 5. The series in one table (what each introduced; what it assumed about fixed points; what we measured)

| | HRM | TRM | EqR | FPRM | **ours** |
|---|---|---|---|---|---|
| scale | 27M | 7M / 5M | 5.03M | 7M | 45k–760k (d12–d64); Sudoku 77k–? |
| recursion | H/L two-timescale | one net, (y, z), n×T | equilibrium iterate, D deep | stabilized fixed-point iterate | equilibrium core (damped, learned η), T6–24 |
| fixed points | **assumed** (one-step/IFT gradient) | **shown not reached**; assumption dropped, full BPTT | **built** (stabilized attractors) | **built** (+ stabilizer theorems) | **measured** (retention N, ladders r̄, reachability, e1e3) and graded: H-2 falsified-as-deployed → H-2′ basin-trained |
| training lever | deep supervision, ACT | EMA, fewer layers, n↑ | RI rows, NI, SOT | coupled residual scaling, halting | anchors (basin objectives), NI, pricing, S9 |
| test-time scaling | M_max↑ pays on Sudoku, not ARC (unexplained) | — | depth D + breadth B | FPOpt damping with iteration budget | depth pays on Sudoku (+16/40 at t=64), +0 on ARC (t=96) — **explained** by landscape class |
| information | — | — | — | — | priced flux / throat / two-profile RG structure; Law 1–4; H-42 |
| symmetry | aug | aug (1000×) | aug | aug | **exact S9 = Sudoku's digit symmetry by construction** (aug-free) |
| Sudoku-Extreme | ~55 (per TRM) ⚠ | 87.4 (MLP) | 99.8 (D64,B128) | — | sprint S2: bands M1 ≥50 / M2 ≥85 / M3 ≥95 |
| ARC-AGI-1 | 40.3 | 44.6 | — | 47.5 (no TTT) | paper 2: ≤1M-param lane; conversion mechanisms |

**The story paper 1 tells with this table.** The series moved from *assuming* equilibria (HRM) to
*not needing* them (TRM) to *engineering* them (EqR/FPRM) — and still evaluates Sudoku and ARC as one
benchmark family, noting only in passing that test-time compute pays on one and not the other. Our
contribution on this axis is to **measure the attractor landscape those equilibria live in** and show it
is a different object on the two domains: single-attractor CSP (retention = solve at depth; RI/multi-init
pay; depth converts near-misses) vs inventory-limited ARC (basins exist for ~30% of instances, cold
starts miss them, depth/RI/temperature buy nothing, candidate supply × basin decoding converts). The
sprint's arms (S0–S7) are precisely the series' levers — RI, NI, training depth, inference depth,
breadth with free verification, stabilized damping — run on one substrate with the instruments on.

## 6. Items owed before writing (⚠)

1. HRM's Sudoku-Extreme provenance (paper "near-perfect" vs TRM's 55.0%) — read TRM's footnote/appendix.
2. EqR's exact Sudoku-Extreme protocol (train size, aug, D/B test-time budget per number) — re-open the
   PDF at the table; confirm the 40k-layer unroll figure in the ledger refers to their extreme test-time
   scaling axis, not the 99.8% config.
3. FPRM on Sudoku/Maze (if reported) — the 08-12 note recorded ARC only.
4. The ARC Prize Foundation's independent HRM analysis (augmentation/ACT attribution) — cite if used.
