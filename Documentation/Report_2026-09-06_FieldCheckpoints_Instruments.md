# The field's frontier models read through our instrument suite — public HRM / TRM / EqR (and a CGAR TRM) Sudoku-Extreme checkpoints (2026-09-06; Fable, parallel science session)

**Registration:** `Plan_2026-09-05_FieldCheckpoints_Instruments.md` (predictions P1–P19 locked before any number was read; §8 = the exploratory addendum). **Knowledge base:** `Note_2026-09-05_Field_KnowledgeBase.md`. **Artifact:** `runs/field_ckpts/analysis/field_ckpts_20260906.{txt,json}` (harness `runs/field_ckpts/harness/`, promoted to `tools/field_ckpts/` at this checkpoint; the checkpoints themselves stay under `runs/field_ckpts/ckpts/`, gitignored). **Compute:** Mac M5 Pro only (Metal + CPU), 2026-09-05 20:51Z → 2026-09-06 ~15:00Z; fp32 forward (the field trains and evaluates in bf16 — the bf16 control is §2.3); the pod was Night A's throughout. **Nothing here touches Night A.** Every number is measured on the named set; "lens" rows are the 2026-09-05 sportC2 decoder-lens values on the identical puzzles.

## 0. In one paragraph
Four public checkpoints ran in their authors' own code, our JAX port of the field cell was verified against the public PyTorch weights to float32 round-off, and every instrument in the suite read on their weights. The public HRM measures 54.9 % (TRM's table was right; HRM's README was not); the public TRM-MLPs measure 79.1 (alphaXiv) and 86.1 (CGAR); the public EqR reproduces its paper (86.8 @D16 / 93.2 @D64 with noise off) and loses 9.7 points at D16 under its own released Langevin-noise setting. The decoder lens orders the field the way it ordered our cells: HRM MIXED (threshold at 24 givens, search yield 49–54 %), the alphaXiv TRM MIXED with a residual threshold at 18–20 givens (yield 76–81), CGAR and EqR DECIMATING (no threshold, 84–92 %). All four commit 87–98 % of cells at step one, churn confidently on failure, and at stalls are wrong on 53–58 % of the cells they are surest about — the calibrated-commitment gap is a property of the whole field. Their halting heads are near-perfect verifiers at the last step (AUC .998–1.000), the mechanism behind every "verification-free" selector in the series. Retention and fixed points split the family: HRM erases an injected solution at step one, TRM and EqR hold it perfectly; EqR's solved endpoints are contractive (radius .76), TRM's neutral (1.01), CGAR's drifting (1.37) while readout-stable. Random-init draws split it again: HRM has no random-init capability at all (3.2 % per draw), TRM and EqR are init-invariant, and the selector law reads spurious rates 11.9 / 2.3 / 0.0 %. Digit relabeling flips the solve status of 24–50 % of puzzles on every public model — the symmetry defect exact equivariance removes by construction. The same puzzles are hard for every cell in the field and for ours (X0's failures lie 94 % inside R3's), and our reproduction X0 reads like the public EqR on every instrument (solved-set Jaccard .886–.908).

## 1. What ran (sets identical to the lens's; idx into our test arrays)
| checkpoint | source | params | trained | instruments (set) |
|---|---|---|---|---|
| HRM-pub | HF `sapientinc/HRM-checkpoint-sudoku-extreme` (official) | 27.28M | 1k × aug 1000, 40k epochs, batch 768, lr 1e-4, wd 1.0; eval always 16 steps | cold D64 (SCAN20k), draws k8 D16 (SUB5k), dynamics t64 (STRAT256), retention/prefix/symmetry/cold (STRAT512), Jacobian (STRAT84), init-radius, D64 draws |
| TRM-pub | HF `alphaXiv/trm-model-sudoku` (independent reproduction) | 5.03M | batch 1536, lr 2e-4, 32,550 steps; card 79.37 | same + bf16 control |
| TRM-CGAR | HF `Kaleemullah/trm-cgar-sudoku` (JAIR 2025 code release) | 5.03M | batch 768, lr 1e-4, wd 1.0, 50k epochs, depth curriculum + supervision decay .7; card 86.02 | same |
| EqR-pub | HF `locuslab/EqR-model` (official; EMA weights) | 5.03M | TRM-MLP + λ_ .95 Langevin loop, RI trunc-normal std 1 at every reset, noise .01 train / .5 eval; 50k steps | same, at noise 0.5 (their eval default) and noise 0 |
Sets: SCAN20k = the 20,000-puzzle natural subsample (seed 20260822; X0's, R3's, W0's, B0c's scans are on the identical puzzles); SUB5k = a seeded 5,000-subset of it; STRAT512/256/84 = `stratified_subsample(test_rating, n, 20260821)`; REF500 = the reference-decoder slice; TRAIN-EqR = EqR's exact seed-42 training 1k (`runs/field_ckpts/sets/train_eqr.npz`). Numerics: fp32; init: the trained fixed buffers (HRM/TRM/CGAR), EqR's exact truncated-normal draw (inverse-erf, std 1).

## 2. Port and numerics
**2.1 The JAX port is exact (P14, split).** The alphaXiv weights converted into `qhrrn2.trm_cell` (token rows shifted by one, prefix row 0 = their vector, heads transposed, their H_init/L_init) reproduce their PyTorch fp32 logits to max |Δ| 0.0017 on logits of scale 19.5 at step 1 (mean 2e-5), with 100 % exact-solve agreement through step 9 and 95.3 % at step 16.
**2.2 The recurrence is chaotic on unsolved puzzles.** The JAX-vs-torch deviation grows ≈ 10× per outer step (0.015 → 0.21 → 0.75 → 15 → 29 → 40 …). The control — PyTorch fp64 vs fp32 on the same weights — grows identically (0.006 → 0.02 → 0.20 → 1.2 → 16 → 22 → 35 …; exact agreement 100 % through step 12, 98.4 % at 16). Round-off is amplified about an order of magnitude per outer step (42 block applications) on churning trajectories; solved trajectories are readout-stable. The P14 agreement criterion (≥ 99.5 %) therefore fails at step 16 for a reason the control attributes to the dynamics, not the port.
**2.3 bf16 moves puzzles, not rates.** TRM-pub at D16 on SCAN20k: fp32 79.12 %, bf16 79.13 %; per-puzzle agreement 95.4 % (460 puzzles solve only in fp32, 461 only in bf16). Every per-puzzle comparison across implementations carries a ≈ 5 % numerics floor; paired tests must treat numerics as noise.

## 3. Headline reproductions and depth (I1 / I7; SCAN20k; CI ±0.5–0.7 pp)
| model | protocol | exact @D1 / 2 / 4 / 8 / 16 / 32 / 64 | regressions 16→64 | valid-but-wrong | STRAT512 @16 |
|---|---|---|---|---|---|
| HRM-pub | fixed init | 1.6 / 8.0 / 34.8 / 48.6 / **54.9** / 58.0 / 59.9 | 0.0 % | 0.2 % (given-violating 0.2 %) | 57.8 |
| TRM-pub | fixed init | 29.5 / 58.1 / 70.1 / 75.7 / **79.1** / 81.4 / 83.5 | 0.0 | 0.0 | 80.9 |
| TRM-CGAR | fixed init | 19.4 / 55.7 / 72.4 / 81.0 / **86.1** / 89.5 / 91.7 | 0.0 | 0.0 | 85.9 |
| EqR-pub | trunc RI, noise .5 (released eval default) | 3.3 / 38.2 / 54.9 / 67.4 / **77.1** / 84.8 / 90.2 | 0.0 | 0.0 | 81.6 |
| EqR-pub | trunc RI, noise 0 | 8.8 / 50.1 / 69.1 / 80.3 / **86.8** / 90.7 / **93.2** | 0.0 | 0.0 | — |
| X0 (ours, sportC1) | fixed init | 86.03 @D16 / 92.81 @D64 (full test) | 0.0 | — | — |
Readings. (i) P1 HIT: the public HRM is a 55 % model under the standard protocol; the "nearly perfect" README claim is not this protocol. (ii) P2 HIT (79.1; 83.5). (iii) P3 split: EqR reproduces its paper only with noise 0 (86.8 ≈ 86.4; 93.2 ≈ 93.0); its released eval yamls set Langevin noise .5, which costs 9.7 pp at D16 and 3.0 at D64 on single trajectories — the noise is a breadth device (§6), and the paper's B=1 column is effectively the noise-0 number. (iv) P16 HIT (86.1). (v) Depth never regresses on any public model (P12 half MISS: HRM regresses 0.0 %, not ≥ 2 %); HRM's +5.0 from D16 → D64 shows the one-step-gradient model was under-iterated, not unstable.

## 4. The decoder lens on the field's weights (E1; SCAN20k)
| model @D | g50 (width) | P(cold \| rating 0) | search yield P(cold \| rating > 0) | by rating 0 / 1–9 / 10–29 / 30–59 / 60+ | class |
|---|---|---|---|---|---|
| HRM-pub @16 | 24.6 (10.9) | 92.7 | 48.5 | 92.7 / 58.3 / 42.5 / 47.4 / 46.7 | MIXED |
| HRM-pub @64 | 24.0 (11.3) | 94.9 | 53.9 | 94.9 / 64.6 / 47.2 / 53.0 / 52.3 | MIXED |
| TRM-pub @16 | 19.9 (16.6) | 99.9 | 75.6 | 99.9 / 83.9 / 68.9 / 75.6 / 78.6 | MIXED |
| TRM-pub @64 | 18.1 (18.6) | 99.9 | 80.7 | 99.9 / 87.6 / 75.0 / 80.8 / 83.2 | MIXED |
| TRM-CGAR @16 | none (16.9) | 99.8 | 83.8 | 99.8 / 89.2 / 79.2 / 84.0 / 85.6 | DECIMATING |
| TRM-CGAR @64 | none (11.4) | 99.9 | 90.4 | 99.9 / 93.7 / 87.9 / 90.2 / 91.1 | DECIMATING |
| EqR-pub n.5 @64 | none (14.6) | 99.7 | 88.6 | 99.7 / 92.0 / 85.9 / 88.6 / 90.1 | DECIMATING |
| EqR-pub n0 @64 | none (10.2) | 99.8 | 92.1 | 99.8 / 94.8 / 89.8 / 92.2 / 92.9 | DECIMATING |
| X0 (lens) | none (10.9) | 100.0 | 91.7 | 100 / 95.1 / 89.4 / 91.1 / 92.4 | DECIMATING |
| R3 / B0c / W0 (lens) | 25.9 / 26.1 / 27.0 | 89.9 / 89.2 / 82.6 | 35.5 / 34.9 / 27.1 | | SOFT |
Readings. P4 partial: EqR and CGAR read DECIMATING; the alphaXiv TRM keeps a threshold at the hardest erasure loads (18–20 givens) and reads MIXED by the letter with decimating-class yield — training quality moves the threshold's position, not the dynamics class (§5); HRM MIXED as predicted (P4 HIT) with a threshold inside our natives' range and twice their yield. P15 HIT: every public TRM sits above every native of ours on yield (76–90 vs ≤ 36) and below X0/EqR on the same puzzles.
**The classical ladder on REF500 (cold@D64; identical 500 puzzles).** peeling 12.2 < BP-40 17.2 < BP+decimation 21.6 < our natives 34.6–45.6 < HRM 61.8 < TRM-pub 84.4 < TRM-CGAR 91.8 ≈ X0 92.2 ≈ EqR 92.6 < ML 100; search yields 0 / 3.5 / 8.2 / 25–39 / 55.9 / 81.7 / 90.4 / 90.8 / 91.3. The stall-set Jaccard with BP's stopping sets falls monotonically along the ladder: W0 .748, B0c .664, R3 .614, HRM .451, TRM-pub .188, X0 .094, CGAR .099, EqR .089 — the learned cells escape BP's stopping sets in proportion to their yield, and the field's frontier shares under 10 % of its stall set with message passing.

## 5. Dynamics, decimation quality, calibration (E2 / E3 / E6; STRAT256 cold trajectories, t = 64)
| model | solved | cells correct step 1/8/64 (solved ‖ unsolved) | commit p>.9 step 1/64 (solved ‖ unsolved) | confidently wrong at t64 (unsolved) | monotone solves | flips-to-wrong per cell, last 32 (unsolved) | syndrome osc. | first_exact med / p90 |
|---|---|---|---|---|---|---|---|---|
| HRM-pub | 61.3 | 59/92/100 ‖ 45/46/45 | 89/100 ‖ 87/90 | 49 | 19.7 | 3.68 | 19.1 | 3 / 17 |
| TRM-pub | 80.5 | 76/97/100 ‖ 40/42/41 | 98/100 ‖ 96/93 | 54 | 73.3 | 5.75 | 25.5 | 1 / 4 |
| TRM-CGAR | 89.8 | 73/92/100 ‖ 44/45/44 | 96/100 ‖ 93/90 | 48 | 62.6 | 3.53 | 17.8 | 1 / 13 |
| EqR-pub n.5 | 87.9 | 60/87/100 ‖ 46/42/43 | 91/100 ‖ 88/92 | 51 | 44.4 | 5.48 | 28.9 | 2 / 18 |
| EqR-pub n0 | 91.0 | 67/94/100 ‖ 45/45/41 | 95/100 ‖ 91/92 | 53 | 54.5 | 3.57 | 20.3 | 1 / 10 |
| X0 (lens) | 92.2 | 71/93/100 ‖ 45/43/46 | 96/100 ‖ 94/90 | 47 | 62.7 | 3.27 | 20.5 | 1 / 10 |
| R3 / W0 (lens) | 45.3 / 29.7 | 50/79/100 ‖ 43/51/49 | 4/100 ‖ 1/34 | 14 / 15 | 3.4 / 7.9 | 0.22 / 2.27 | 6.9 / 11.5 | 14 / 29 · 12 / 28 |
Readings (P5 HIT on commitment and confident churn for all four; monotone-solve MISS for HRM at 19.7). The hard-decision class is the field's, not our reproduction's: every public model commits ≥ 87 % of cells at step one, keeps readout entropy ≤ .05 throughout, and churns confidently when it fails (3.5–5.8 flips to a wrong digit per cell in the last 32 steps; syndrome oscillation 18–29). HRM revises the most (un-peel .88 per solved cell, first-exact median 3) — the two-module, 24-layer step propagates less per step than the 42-layer single-network step and pays in monotonicity (19.7 % vs 63–73 %). Noise .5 on EqR halves monotone solves (54.5 → 44.4) and raises un-peel 3.6× — the breadth device injects revision.
**E3 / E6 — at stalls nobody's confidence is a hard decision.** Committed fraction at τ .9: 90–93 % of non-given cells on all four public models, 53–58 % of them wrong, P(all committed correct) = 0, and peeling from the committed cells contradicts on 100 % of stalled puzzles (X0 90.5 / 52.3 / 0 / 100; our natives 34–40 % committed, 33–39 % wrong, contradictions 44–57 %). The five most confident stalled cells are right 46–53 % of the time at confidence 1.000 (gap .47–.54); step-1 entropy .01–.05. The calibrated-commitment finding (sportC2 §4.4) generalizes to every public field checkpoint.

## 6. List decoding and the selector law (E4 / E5; SUB5k × k = 8 random-init draws N(0,1), D16; labeled k=8 vs the lens's k=128)
| model | cold@16 | b1 (draw 1) | verified@1/2/4/8 (cold ∪ draws) | rescue P(any of 8 \| cold fails) | r_i spectrum 0 / (0,.5] / (.5,1) / 1 | selector (our latent residual): AUC / spurious / t1r@8 / verified@8 | EqR's logit score: AUC / spurious / top-4-of-8 mean / any |
|---|---|---|---|---|---|---|---|
| HRM-pub | 54.6 | **3.2** | 55.0/55.3/55.8/56.8 | 4.8 | 80.5 / 19.5 / 0 / 0 | .790 / 11.9 / 9.3 / 19.5 | .791 / 11.3 / 5.1 / 17.1 |
| TRM-pub | 79.2 | 77.8 | 82.8/84.6/87.4/90.2 | 53.0 | 10.1 / 11.8 / 19.0 / 59.2 | .968 / 2.3 / 88.2 / 89.9 | .965 / 2.2 / 83.1 / 89.6 |
| TRM-CGAR | [PENDING: CGAR draws] | | | | | | |
| EqR-pub n.5 | 77.4 | 76.8 | 85.7/89.5/93.0/95.9 | 81.8 | 4.6 / 19.7 / 21.2 / 54.4 | .982 / 0.0 / 94.8 / 95.4 | .967 / 0.0 / 86.2 / 94.5 |
| EqR-pub n0 | 87.5 | 87.0 | 91.3/93.1/95.0/96.4 | 70.9 | 3.9 / 9.5 / 10.2 / 76.4 | .986 / 0.0 / 95.8 / 96.1 | .980 / 0.0 / 91.7 / 95.6 |
| X0 (lens, k128) | 92.9 | 92.1 | … 99.7 @128 | 95.9 | 0.3 / … / 93.5 | .925 / 7.8 / 92.5 / 99.7 | — |
Readings. (i) **HRM has no random-init capability** (draw-1 exact 3.2 % vs 54.6 % cold; 80 % of puzzles never solved from any draw): the public HRM works only from its one trained initial vector — our no-RI native's class (P1's b1 0.39 %). TRM and EqR are init-invariant (b1 ≈ cold), as X0 was. (ii) **The selector law reads on their weights (P6 HIT on EqR, MISS by a hair on TRM's band):** spurious-attractor rate per wrong draw EqR 0.0 % (RI/NI-trained), TRM-pub 2.3 %, HRM 11.9 %; residual selection recovers .994 (EqR), .981 (TRM), .48 (HRM) of the verifier's coverage — verification-free selection needs the clean attractor structure RI/NI training buys (the D3/D9 law, external). Our latent residual discriminates at least as well as EqR's own logit-delta score (AUC .982 vs .967 at k=8). (iii) **Noise is a breadth device:** EqR's noise .5 costs 10 pp per draw (b1 87.0 → 76.8) but leaves the 8-draw coverage unchanged (96.4 vs 95.9) and raises the rescue of cold failures from 71 % to 82 % — precision traded for diversity, the funnel model's (ρ, r) plane on the field's weights. (iv) EqR's own top-4-by-convergence selection at B=8 reads 91.7 % mean / 95.6 % any (noise 0) — a verifier-free statistic within 0.5 pp of the verified coverage on the model whose attractors are clean, and 5.1 / 17.1 on HRM whose are not.

## 7. The halting head is a learned verifier (I8; SCAN20k)
AUC(q_halt, exact) at the final step: HRM .999, TRM-pub .999, TRM-CGAR .999, EqR 1.000 (both depths); their own `q_halt_accuracy` 99.8–100 %; precision/recall at threshold 0: 99.7–100 / 100 (P9 HIT). The ACT head, trained with BCE on sequence correctness, becomes a near-exact correctness detector on a domain where correctness is locally checkable. Consequences: PTRM's best-Q@100 = 98.75 vs pass@100 = 99.06 is a verifier-selected number in all but name; the "verification-free" framing of the series' selectors is a verifier-dependence statement — the first row of the paper's verifier-dependence table, measured on the field's weights. (Note the trivial half: at the last step the head's positives coincide with the solved set; the non-trivial content is that it is calibrated on the unsolved side too — precision 99.7–100.)

## 8. Retention, fixed points, prefix, symmetry, memorization (I5 / I6 / I9 / I10 / I11)
| instrument | HRM-pub | TRM-pub | TRM-CGAR | EqR-pub |
|---|---|---|---|---|
| solution retention (z_H := their embedding of the solution; exact at steps 1…8, STRAT512) | **0.0** / 0.6 / 4.1 / 14.1 / 19.7 / 26.8 / 32.6 / 38.1 (all-8: 0.0) | 100 at every step | 100 at every step | 100 at every step (noise 0 and .5) |
| FD Jacobian radius at the t=64 endpoint, solved (p10; frac < 1) ‖ unsolved (STRAT84) | .78 (.66; 92 %) ‖ 1.94 | 1.01 (.92; 47 %) ‖ 4.17 | 1.37 (1.17; 1 %) ‖ 8.17 | .76 (.61; 91 %) ‖ 5.60 |
| relative latent residual at t=64, solved ‖ unsolved | .0000 ‖ .33 | .0075 ‖ .49 | .0715 ‖ .45 | .0000 ‖ .47 |
| prefix token zeroed (Δ exact@16, STRAT512) | 57.8 → 2.0 (**−55.9**) | 80.9 → 81.8 (+1.0) | 85.9 → 86.5 (+0.6) | 81.6 → 76.4 (−5.3) |
| digit relabeling ×9 (STRAT512): orbit-consistent / any-of-9 / orbit-vote gain | 50.0 / 81.8 (+24) / −0.6 | 76.4 / 92.0 (+11) / −0.2 | 76.6 / 95.7 (+10) / +2.7 | 59.4 / 97.5 (+16) / +0.4 |
| own training 1k, exact@16 (EqR only; seed-42 set) | — | — | — | 81.5 (test 77.1 at the same noise) |
Readings. (i) **Truth erasure vs retention (P7):** the HRM loop overwrites an injected solution at step one and re-derives 38 % of them from the puzzle by step eight; the single-network TRM/EqR loop holds the solution embedding as a fixed point on every puzzle. Ren & Liu's "fixed-point violation" on HRM is, on the public checkpoint, an erasure of handed truth (our E3b class), not an instability of solved endpoints — those are contractive (radius .78) with zero depth regressions. (ii) **Attractor training does what it claims (P8):** EqR's solved endpoints are contractive (.76; residual 0) where TRM-pub's are neutral (1.01; drift .75 %/step) and CGAR's drift (1.37; 7 %/step) while readout-stable — the curriculum-trained model is the least converged in the latent and the best of the three TRMs in accuracy: readout stability, not latent convergence, carries the field's accuracy (lens G3's reading, now across four models). (iii) **Prefix (P13 split):** HRM's single prefix token is load-bearing (−56 pp: it is the only token its Q head reads and the attention's global scratch), EqR's costs 5 pp, the TRMs' are inert. (iv) **The symmetry defect in the field's units (P10 half):** despite training with digit permutation, relabeling the digits flips the solve status of 24–50 % of puzzles; any-of-9 relabelings adds 10–24 pp of coverage while majority orbit-voting adds ≤ 2.7 (Ren & Liu's +18.2 used a different vote). An exactly equivariant cell has this defect at zero by construction — the symmetry row of the paper, measured on their weights. (v) **No memorization in the field regime (P11 MISS, informatively):** EqR solves 81.5 % of its own 1,000 training puzzles at D16 — the same as test — where X1 (the same cell without the digit orbit) memorized 99 %.

## 9. Cross-cell overlap on the identical 20k at depth 64 (J)
| pair | Jaccard(solved) | only-A | only-B | P(A fails \| B fails) | P(B fails \| A fails) |
|---|---|---|---|---|---|
| EqR-pub \| X0 | .886 | 844 | 1372 | 40.9 | 29.8 |
| CGAR \| X0 | .908 | 778 | 1002 | 45.5 | 39.3 |
| TRM-pub \| X0 | .844 | 548 | 2430 | 61.6 | 26.6 |
| HRM \| X0 | .624 | 241 | 6840 | 83.1 | 14.8 |
| X0 \| R3 | .460 | 9989 | 89 | 11.8 | **93.8** |
| EqR \| R3 | .468 | 9533 | 161 | 15.8 | 91.8 |
| HRM \| R3 | .552 | 4634 | 1333 | 59.1 | 83.4 |
| TRM-pub \| CGAR | .842 | 675 | 2333 | 59.1 | 29.5 |
Readings. Failures nest along the ladder: X0's cold failures fall 94 % inside R3's, 83 % inside HRM's; CGAR solves 96 % of what the alphaXiv TRM solves (P19 HIT). Our reproduction X0 and the public EqR/CGAR agree on 89–91 % of their solved sets — the external-validity result: the model we built to read the field's recipe through our instruments is, on every instrument, the field's model.

## 10. Exploratory (plan §8; post-hoc, no scoreboard weight)
**Init-basin radius** (exact@16 on STRAT512 with z0 = trained init + ε·N(0,1); ε = 0 is the cold pass, ε = 3 ≈ a random draw):

| model | ε 0 | .03 | .1 | .3 | 1 | 3 |
|---|---|---|---|---|---|---|
| HRM-pub | 57.6 | 58.2 | 57.2 | 55.7 | 27.7 | 0.6 |
| TRM-pub (alphaXiv) | 81.4 | 80.5 | 79.9 | 81.1 | 80.5 | 76.4 |
| TRM-CGAR | 86.3 | 86.1 | 87.3 | 88.3 | 77.3 | 71.7 |
| EqR-pub (noise 0) | 87.9 | 88.1 | 89.3 | 88.7 | 88.7 | 87.5 |

Reading: a one-number instrument for path dependence — the perturbation size at which cold accuracy halves is ≈ 1 for HRM (its trained initial vector is the only working start; at ε = 3 nothing is left, the 3.2 % draw result), > 3 for EqR (RI-trained: flat to a full random draw), ≈ 2 for the alphaXiv TRM, ≈ 1 for CGAR, which also GAINS +2 pp at ε = .3 (the PTRM / guided-reasoning noise dividend measured on a public checkpoint). **HRM random-init draws at D = 64** (1k puzzles, k = 2): draw-1 exact 4.5 %, any-of-2 8.2 % (D16: 3.2 %) — depth does not recover the random-init trajectory; HRM's random-init basins are absent, not slow. **CGAR's draws** (§6): single random-init draw 64.5 % against 86.6 cold — the curriculum-trained TRM is far less init-invariant than the alphaXiv one — with a broken residual selector (AUC .73, spurious 26.5 %, t1r@8 69.8 vs verified 94.9): the model whose solved endpoints drift most in the latent (radius 1.37, §8) is the one whose convergence residual selects worst — the selector law's mechanism (selection needs converged solved states) read on a public checkpoint.

## 11. Prediction scoreboard (P1–P19 as registered; sub-items counted)
| P | prediction | read | verdict |
|---|---|---|---|
| P1 | HRM-pub exact@16 ∈ [48, 62] | 54.9 | HIT |
| P2 | TRM-pub @16 ∈ [77, 82]; @64 ∈ [82, 90], regressions ≤ .5 % | 79.1; 83.5, 0.0 | HIT, HIT |
| P3 | EqR @16 ∈ [82, 90] at noise .5; @64 ∈ [90, 95]; \|Δnoise\| ≤ 2 pp; top-4 at B=8 ≥ 96 | 77.1 (86.8 at noise 0); 90.2 / 93.2; 9.7 pp; 95.6 any / 91.7 mean | MISS (HIT at noise 0), HIT, MISS, MISS-BELOW |
| P4 | TRM-pub and EqR DECIMATING; HRM MIXED | TRM-pub MIXED (g50 18–20); EqR DECIMATING; CGAR DECIMATING; HRM MIXED | MISS, HIT, HIT |
| P5 | commit@1 ≥ 85 % (all); monotone ≥ 40 %; conf-wrong at stalls ≥ 30 % | 87–98; HRM 19.7, others 44–73; 48–54 | HIT, MISS(HRM)/HIT, HIT |
| P6 | TRM-pub spurious ∈ [3, 12] %; EqR ≤ 1 %; t1r/verified EqR ≥ .98, TRM ≤ .96 | 2.3; 0.0; .994, .981 | MISS-BELOW, HIT, HIT, MISS (ordering HIT) |
| P7 | retention EqR ≥ .95; TRM ∈ [.6, .95]; HRM < .80 | 1.00; 1.00; 0.0 | HIT, MISS-ABOVE, HIT |
| P8 | EqR solved residual ≤ 1/5 TRM's; TRM radius ∈ [.95, 1.15]; EqR < 1; HRM > 1 | 0 vs .0075; 1.01; .76; .78 | HIT, HIT, HIT, MISS |
| P9 | halting AUC ≥ .90 (TRM/EqR), ≥ .80 (HRM) | .999–1.000 | HIT |
| P10 | HRM orbit-vote gain ≥ +10 with consistency < .8; TRM/EqR gain ≤ +5 | −0.6 (any-of-9 +24), .50; −0.2 / +0.4 / +2.7 | MISS/HIT, HIT |
| P11 | EqR train-solve ≥ .99; given-violating recalls ≤ .2 % | 81.5; 0.0–0.2 | MISS, HIT |
| P12 | TRM/EqR regressions ≤ .5 %; HRM ≥ 2 % | 0.0; 0.0 | HIT, MISS |
| P13 | prefix inert on all (\|Δ\| ≤ 1 pp) | TRM +1.0, CGAR +0.6, EqR −5.3, HRM −55.9 | HIT, HIT, MISS, MISS |
| P14 | port: step-1 max \|Δ\| ≤ 1e-3; exact agreement ≥ 99.5 % at D16 | 0.0017 (scale 19.5; relative 1e-4); 95.3 % (control identical) | HIT (relative), MISS (chaos, control-attributed) |
| P15 | TRM-pub above every native on yield and below X0; g50 ≤ 20 or absent | 80.7 vs ≤ 36 vs 91.7; 18.1 | HIT, HIT |
| P16 | CGAR @16 ∈ [84, 88] | 86.1 | HIT |
| P17 | CGAR DECIMATING; dynamics within 10 pp of TRM-pub | DECIMATING; commit 96 vs 98, monotone 62.6 vs 73.3 | HIT, borderline |
| P18 | CGAR spurious ≤ TRM-pub's | 26.5 vs 2.3 % (CGAR b1 64.5 vs cold 86.6; AUC .73) | MISS (10×) |
| P19 | P(CGAR solves \| TRM-pub solves) ≥ .95 | .96 | HIT |
Tally on the read items: 27 HIT, 13 MISS, 1 borderline (P18 added: CGAR's selector is the worst of the four). The misses carry the new content: the noise protocol, the residual threshold at the low-givens edge, HRM's dead draws and load-bearing prefix, EqR's non-memorization, HRM's contractive endpoints.

## 12. What this gives the paper (and what it does not)
1. **The instruments port.** Every suite member read on four public checkpoints in their own code, and our reproduction reads like the public EqR on all of them (Jaccard .89–.91, threshold/yield/dynamics/selector within noise). The external-validity objection (Review #1 R4, Review #2 §3) is answered by measurement.
2. **Field-wide readings no series paper reports:** (a) the decoder-class ladder with a monotone escape from BP's stopping sets (Jaccard .75 → .09); (b) the calibrated-commitment gap on every field model (90 % committed at stalls, 53–58 % wrong, top-5 correct ≈ 50 % at confidence 1.000); (c) the halting head as a near-exact verifier (AUC ≥ .998) — the verifier-dependence table's first row; (d) the selector law's external instance (spurious 0.0 / 2.3 / 11.9 % ↔ RI/NI / none / no random-init capability); (e) truth erasure vs retention separating the two-module HRM loop from the single-network loop; (f) readout-stable, latent-drifting solved states on the best TRM (radius 1.37) — accuracy does not require latent convergence; (g) the symmetry defect in their units (24–50 % of puzzles flip under relabeling; +10–24 pp any-of-9 coverage) — the price of augmentation over equivariance; (h) the reproduction distribution of TRM-MLP (72–87) and HRM's 55; (i) the noise-as-breadth reading of EqR's protocol; (j) round-off amplification ≈ 10× per outer step — a numerics floor for every per-puzzle claim in the field.
3. **What it does not give:** an accuracy claim for us. The natives are 35–46 on the ladder where the field's frontier is 84–93; the DEC (Night A) is the arm that answers whether the symmetric state closes that gap, and SE-RRM (knowledge base §4) says a symmetric state with attention over symbols already does at 2M.
4. **Instrument additions to the suite (standing from here):** solution retention on injected embeddings for any recurrent cell; the halting-head AUC row; the orbit-consistency / any-of-k / orbit-vote row; the init-basin radius (§10); the bf16 agreement floor.

## 13. Labels and limits
fp32 vs the field's bf16 (§2.3); k = 8 draws on 5k (the lens's columns are k = 128 on 20k — ρ and spurious rates at k=8 are lower bounds); HRM's training set unknown (memorization read only on EqR); the lens's SOFT/MIXED/DECIMATING letters depend on whether the fitted threshold falls inside 17–35 givens (TRM-pub at 18.1 is a boundary case); EqR's noise-.5 rows are single trajectories under their breadth protocol, its noise-0 rows the paper's B=1; all sets are the lens's, none is the full 422,786 (CIs ±0.5–0.7 pp on the 20k).
