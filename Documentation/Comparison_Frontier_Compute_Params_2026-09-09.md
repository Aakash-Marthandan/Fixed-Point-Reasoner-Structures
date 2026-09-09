# The frontier comparison, organized by parameters and by inference compute (2026-09-09)

Every accuracy carries its training-data column (the 1k convention = 1,000 base puzzles × augmentation; the full split = 2.7–3.8M puzzles), its protocol (single pass / restarts / search; the depth; the test set) and a compute tier: **M** = analytic multiply-accumulates per puzzle from our accounting (our cells and the TRM-MLP cell: 70.0 GMAC per outer step at w384, 20.4 at w192, 15.2 for the field's 5M cell; TFLOPs = 2 × TMAC); **R** = read from the paper's own compute axis; **I** = inferred from the parameter count × 81 tokens × the reported iterations (a lower bound, no attention terms); **–** = not computed. Our numbers: EMA weights, bf16, val-selected inside the run on train-file puzzles; the field's released weights through the same evaluator on identical puzzles. Sets: FULL = 422,786; 100k / 20k / 5k = uniform subsamples (binomial SE 0.04 / 0.07 / 0.09 pp at 99 %).

## 1. By parameter count (the thousand-puzzle convention unless the training column says otherwise; the headline single-pass number at the paper's own depth)

| params | system | training data | protocol | accuracy | inference compute per puzzle (tier) |
|---|---|---|---|---|---|
| 789,125 | **DEC-w192 (C5, ours, one seed)** | 1k × position aug | single pass D64, 100k | **99.10** | 1.31 TMAC = 2.6 TFLOPs (M); D16 95.91 at 0.33 TMAC; D128 99.42 at 2.61; D256 99.60 at 5.22 |
| 796,937 | Sotaku v2 (GitHub) | **2.7M puzzles** | single pass, 1,024 iterations, a 25k subset | 99.12 | ≈ 0.07 TMAC (I: 81 × 0.8M × 1,024); 96.29 at 128 iterations ≈ 0.008 |
| 800,000 | Lattice Deduction Transformer | 1k × (digit + D4) aug | learned elimination + branching search; 300 puzzles | 100 (2 timeouts) | – (search) |
| 2,000,000 | SE-RRM ✓ | 1k × aug (digit permutation) | single pass, 16 supervision steps | 93.73 | – (attention over 9 symbols × 81 positions; not computed) |
| 2,780,933 | **DEC-w384 seed triple (C0 C1 C2, ours)** | 1k × position aug | single pass D64, 100k; mean ± half-spread of 3 seeds | **98.18 ± 0.61** | 4.48 TMAC = 9.0 TFLOPs (M); D16 95.02 ± 0.65 at 1.12; D128 98.70 at 8.96; D256 98.97 at 17.9 |
| 2,977,541 | DEC-w384 + set attention (C4, ours, one seed) | 1k × position aug | single pass D64, 100k | 98.90 | 4.48 TMAC + the attention's q/k (M, lower bound); D16 96.70 |
| 5,000,000 | CMM | 1k × aug ⚠ | single pass, iterations unstated | 93.7 | – |
| 5,000,000 | PTRM ✓ (its Table 3; an earlier note said 7M) | 1k × aug | best-Q of 100 stochastic rollouts, D64 | 98.75 (87.28 deterministic) | ≈ 97 TMAC (M: 100 rollouts × 0.97 on the TRM-MLP cell) |
| 5,028,866 | alphaXiv TRM-MLP (released grid, our evaluator) | 1k × aug 1000 | single pass D16 / D64, FULL | 79.32 / 83.45 | 0.24 / 0.97 TMAC (M); the paper's 87.4 at D16 |
| 5,037,058 | CGAR (released, our evaluator) ✓ | 1k × aug | single pass D16 / D64, FULL | 86.10 / 91.71 | 0.24 / 0.97 TMAC (M); the paper's 86.02 |
| 5,037,058 | EqR EMA (released, our evaluator) | 1k × aug | single pass D16 / D64 / D128 / D256 (FULL / FULL / 20k / 5k) | 86.52 / 93.15 / 95.02 / 95.88 | 0.24 / 0.97 / 1.95 / 3.89 TMAC (M); the paper's 84.8 / 93.0 |
| 7,000,000 | FPRM ✓ | 1k × aug 1000 | pass@1, adaptive halting, FULL | 94.2 | – (their curve sits at ≈ 1–10 TFLOPs per puzzle in FRM's Fig. 5; R) |
| 7,000,000 | Flow Reasoning Models v3 ✓ | 1k (difficulty-balanced) × Sudoku symmetries | one stochastic rollout; PEAK over inference compute; a 1,000-puzzle test subset; 3 seeds | 99.5 | ≈ 1–10 TFLOPs = 0.5–5 TMAC at the peak (R: their Fig. 5, PyTorch operator FLOPs × realized NFE) |
| 10,000,000 | GRAM | 1k × aug ⚠ | 20 sampled trajectories + majority vote, 16 iterations, a 1,000-puzzle test set | 97.0 | – (20 × a 16-iteration pass) |
| 27,000,000 | HRM | 1k × aug | single pass, up to 16 ACT segments | 55.0 (TRM's table; 54.9 our read) | – |
| 27,000,000 | Attractor Models | ≈ 1k | fixed point by implicit differentiation | 91.4 | – |
| n/r | Diffusion curriculum ✓ (hidden-128 looped transformer; params not reported) | **the full 3.83M-puzzle split** | one stochastic rollout, K = 10,000 steps, ≈ 423k test, 3 seeds | 99.90 | – (10,000 denoiser passes; I: ≈ 1–2 TMAC if the block is 1–2M params) |

## 2. The compute–accuracy ladder under the thousand-puzzle convention, single pass (one deterministic rollout; sorted by inference compute per puzzle; tier M unless noted)

| TMAC / puzzle | TFLOPs | system, depth | params | set | accuracy |
|---|---|---|---|---|---|
| 0.24 | 0.5 | TRM-MLP paper, D16 (their number) | 5M | FULL | 87.4 |
| 0.24 | 0.5 | CGAR paper, D16 | 5M | FULL | 86.02 |
| 0.24 | 0.5 | EqR paper, D16 | 5.03M | FULL | 84.8 |
| 0.24 | 0.5 | alphaXiv TRM-MLP, D16 | 5.04M | FULL | 79.32 |
| 0.24 | 0.5 | CGAR, D16 | 5.04M | FULL | 86.10 |
| 0.24 | 0.5 | EqR, D16 | 5.04M | FULL | 86.52 |
| 0.33 | 0.7 | **DEC-w192 C5, D16** | 0.79M | FULL | **95.91** |
| 0.97 | 1.9 | alphaXiv TRM-MLP, D64 | 5.04M | FULL | 83.45 |
| 0.97 | 1.9 | CGAR, D64 | 5.04M | FULL | 91.71 |
| 0.97 | 1.9 | EqR, D64 | 5.04M | FULL | 93.15 |
| 0.97 | 1.9 | EqR paper, D64 (B = 1) | 5.03M | FULL | 93.0 |
| 1.12 | 2.2 | **DEC-w384 triple, D16** | 2.78M | FULL | **95.02 ± 0.65** |
| 1.12 | 2.2 | DEC-w384 C2 (champion by rule), D16 | 2.78M | FULL | 95.63 |
| 1.12 | 2.2 | DEC-w384 + set attention C4, D16 (+ q/k, lower bound) | 2.98M | FULL | 96.70 |
| 1.31 | 2.6 | **DEC-w192 C5, D64** | 0.79M | 100k | **99.10** |
| 1.95 | 3.9 | alphaXiv TRM-MLP, D128 | 5.04M | 20k | 84.97 |
| 1.95 | 3.9 | CGAR, D128 | 5.04M | 20k | 93.17 |
| 1.95 | 3.9 | EqR, D128 | 5.04M | 20k | 95.02 |
| 2.61 | 5.2 | **DEC-w192 C5, D128** | 0.79M | 20k | **99.42** |
| 3.89 | 7.8 | alphaXiv TRM-MLP, D256 | 5.04M | 5k | 86.26 |
| 3.89 | 7.8 | CGAR, D256 | 5.04M | 5k | 94.10 |
| 3.89 | 7.8 | EqR, D256 | 5.04M | 5k | 95.88 |
| 4.48 | 9.0 | **DEC-w384 triple, D64** | 2.78M | 100k | **98.18 ± 0.61** |
| 4.48 | 9.0 | DEC-w384 C2 (champion by rule), D64 | 2.78M | 100k | 98.67 |
| 4.48 | 9.0 | DEC-w384 + set attention C4, D64 (+ q/k, lower bound) | 2.98M | 100k | 98.90 |
| 5.22 | 10.4 | **DEC-w192 C5, D256** | 0.79M | 5k | **99.60** |
| 8.96 | 17.9 | **DEC-w384 triple, D128** | 2.78M | 20k | **98.70 ± 0.43** |
| 8.96 | 17.9 | DEC-w384 C2 (champion by rule), D128 | 2.78M | 20k | 99.00 |
| 8.96 | 17.9 | DEC-w384 + set attention C4, D128 (+ q/k, lower bound) | 2.98M | 20k | 99.15 |
| 17.92 | 35.8 | **DEC-w384 triple, D256** | 2.78M | 5k | **98.97 ± 0.46** |
| 17.92 | 35.8 | DEC-w384 C2 (champion by rule), D256 | 2.78M | 5k | 99.34 |
| 17.92 | 35.8 | DEC-w384 + set attention C4, D256 (+ q/k, lower bound) | 2.98M | 5k | 99.32 |
| ≈ 0.5–5 (R) | ≈ 1–10 | Flow Reasoning Models v3, peak over compute (a 1,000-puzzle test subset; one stochastic rollout) | 7.0M | 1k subset | 99.5 |
| – | – | FPRM pass@1 with adaptive halting (≈ 1–10 TFLOPs in FRM's Fig. 5, R) | 7M | FULL | 94.2 |
| – | – | SE-RRM, 16 steps | 2M | ? | 93.73 |
| – | – | CMM | 5M | ? | 93.7 |
| – | – | Attractor Models | 27M | ? | 91.4 |
| – | – | HRM | 27M | FULL | 55.0 |

## 3. The restart, vote and search columns (the thousand-puzzle convention; sorted by inference compute per puzzle)

| TMAC / puzzle | system | params | set | accuracy | column |
|---|---|---|---|---|---|
| 43 | **DEC-w192 C5, k = 32 at D64 (residual-selected = verified)** | 0.79M | 5k | **99.82** / 99.82 | verification-free residual selection / the free verifier |
| 97 | PTRM, best-Q of 100 rollouts at D64 (σ .3) | 5M | FULL | 98.75 (pass@100 99.06) | learned Q-head selection |
| 125 | EqR (released, our evaluator), k = 128 at D64 | 5.04M | 20k | 98.85 / 98.89 | residual / verifier |
| 125 | EqR paper, B = 128 residual selection at D64 | 5.03M | FULL | 99.8 | residual (their headline) |
| 125 | CGAR (released, our evaluator), k = 128 at D64 | 5.04M | 20k | 71.34 / 99.31 | residual (broken: 30.9 % spurious) / verifier |
| 125 | alphaXiv TRM-MLP (released, our evaluator), k = 128 at D64 | 5.04M | 20k | 93.84 / 97.50 | residual / verifier |
| 148 | DEC-w384 triple, k = 32 at D64 | 2.78M | 5k | 99.28–99.70 residual = verified | residual / verifier |
| 168 | **DEC-w192 C5, k = 128 at D64** | 0.79M | 5k | **99.88** / 99.88 | residual / verifier |
| 578 | DEC-w384 C0, k = 128 at D64 | 2.78M | 5k | 99.76 / 99.76 | residual / verifier |
| – | GRAM, 20 sampled trajectories + majority vote at 16 iterations | 10M | 1,000 puzzles | 97.0 | unverified vote |
| – | Guided reasoning (TRM base), halting-head reweighted trajectories | 5M | ? | 98.0 | halting-head selection |
| – | Speed is Confidence, halt-first selection over an ensemble | TRM ensembles | ? | 97 | ensemble |
| – | EqR, test-time scaling to 40,000 equivalent layers (abstract) | 5.03M | ? | > 99 | depth + breadth |
| – | Hypothesis-Pinning Search (blog), hundreds of rollouts on the tail | 7M | test set | 100 | search |
| – | Lattice Deduction Transformer, branching search | 0.8M | 300 puzzles | 100 | search |

## 4. The other training regime (the full split; a different problem: large-data solvers, not small-data inductive bias)

| TMAC / puzzle (tier) | system | params | training puzzles | protocol | accuracy |
|---|---|---|---|---|---|
| ≈ 0.07 (I) | Sotaku v2 | 0.80M | 2.7M | single pass, 1,024 iterations, 25k subset | 99.12 (99.05 at 2,048; 98.63 at 4,096) |
| ≈ 1–2 (I) | Diffusion curriculum | not reported (hidden 128) | 3.83M | one stochastic rollout, K = 10,000, ≈ 423k test | 99.90 |

## 5. What the tables say

1. **Per parameter, the DEC is the smallest model at the top of every thousand-puzzle column:** the w192 DEC (0.79M) is 16 % of EqR / PTRM (5M), 11 % of FPRM / FRM (7M), 8 % of GRAM (10M) and 3 % of HRM (27M), and reads 99.10 at D64 against their 93.0–94.2 single pass; the w384 triple (2.78M) is 55 % of the 5M cell at 98.18 ± 0.61. Only Sotaku (0.80M) and LDT (0.8M) are of the w192's size, and both sit in other columns (2.7M training puzzles; search).
2. **Per unit of inference arithmetic under the thousand-puzzle convention, the ladder is ours from 0.33 TMAC upward:** at 0.24–0.33 TMAC the released 5M weights read 79–87 (D16) and the w192 DEC 95.91 (D16); at ≈ 1 TMAC EqR's D64 93.15 vs the w192's D64 99.10 (1.31); at 2–4 TMAC EqR's D128 / D256 95.02 / 95.88 vs the w192's D128 99.42 (2.61) and the triple's D16 95.02 (1.12); every DEC point above 1 TMAC is above every field point at any depth. FRM's 99.5 peak sits at ≈ 0.5–5 TMAC on their own axis, on a 1,000-puzzle subset, within error of the w192's 99.42–99.60 rows on different subsets: the two systems share the top of this ladder, and neither is measured on the other's set.
3. **The w384 DEC buys its lead with arithmetic, the w192 does not:** at D64 the w384 triple costs 4.6× EqR's arithmetic for +5.1 pp; the w192 costs 1.34× for +6.0 pp on identical puzzles. The compute story of the paper is the w192 row; the w384 rows are the seeded claim.
4. **On the restart columns the compute is comparable and the DEC is at or above the field:** 43 TMAC buys 99.82 verification-free at k = 32 on the w192 (EqR needs ≈ 125 TMAC for 99.8 in its paper and reads 98.85 through our evaluator on the 20k; PTRM ≈ 97 TMAC for 98.75); at matched k = 128 the w192 reads 99.88 at 169 TMAC. These are coverage-class numbers on a 5k subsample, one seed for the w192.
5. **The full-split solvers are cheaper AND more accurate, and they are a different problem:** Sotaku's ≈ 0.07 TMAC for 99.12 and the diffusion curriculum's 99.90 are trained on 2,700–3,800× more puzzles; under the thousand-puzzle convention the DEC leads, and no thousand-puzzle model has been shown to reach those numbers. The paper claims the small-data column and says so.
6. **What is not computed:** SE-RRM's, CMM's, FPRM's, GRAM's, HRM's and the Attractor Model's arithmetic per puzzle (no operator counts in their papers; FPRM's and FRM's read from FRM's Fig. 5 only), and C4's attention term. The MAC instrument that would replace the analytic column is owed.
