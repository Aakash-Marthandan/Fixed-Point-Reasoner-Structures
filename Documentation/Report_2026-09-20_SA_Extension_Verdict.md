# The attention arms at the paper's budget — verdict (2026-09-20)

**Registration:** `Plan_2026-09-19_SA_Extension.md` (§2–§5 locked at f2ebf57; the PI's page-one choice A + B recorded at 75ed4f7; §3 and §4 verified byte-identical since). **Analyzer:** `tools/analyze_saext.py`, frozen at f2ebf57 — it and its import verified by content hash immediately before the run (sha256 1cb297db… / 029f9f08…), selftest 30/30, run FIRST and byte-untouched. **Data:** `runs/_saext_pull/` (132 objects, 1.51 GB, crc32c 132/132). **Cost:** $319.49 (page one ≈ $320, cap $380), 15 h 53 min. Every number below is printed by a script; outputs in `runs/_saext_pull/analysis/`.

## 0. The short version

The three attention arms (SE-RRM's two mixers inside our block, loop and recipe) were resumed from their banked 30k ladder states to the paper's 50,000-step budget and read with the full champion battery plus the four rows the paper's width-192 seeds carry. **On all 422,786 test puzzles — the set SE-RRM reports on:**

| arm | parameters | 16 iterations | 64 iterations | selected grid |
|---|---|---|---|---|
| SA128 | 0.62M | 95.45 | 99.48 | 42,000 |
| SA192 | 1.06M | 97.31 | 99.62 | 40,000 |
| SA256 | 1.97M | 97.93 | 99.62 | 46,000 |
| our width-192 triple (the paper's model) | 0.79M | **95.41 ± 0.54** | **99.05 ± 0.18** | 46,000 each |
| SE-RRM, published | 2M | 93.73 main text · **95.4 appendix Table A6** | 98.22 | one run |

| question | answer |
|---|---|
| Did the extension raise the arms? | **No, not beyond the floor.** R-SE-1 is INSIDE at both depths on all three (+0.10 / +0.52 / −0.22 pp at 16 on the identical 5,000). The ladder's budget-edge rows were not materially lower bounds. |
| Did it fix the edge problem? | **Yes.** R-SE-2: every arm now selects INTERIOR (42,000 / 40,000 / 46,000) where the ladder had SA128 and SA192 at the 30k edge and SA256 at 28k. The rows carry no "lower bound" caveat. |
| Are the attention arms better than our model? | **Not resolved at one seed.** R-SE-3 is INSIDE at both depths for all three. SA256's +2.53 pp at 16 sits **0.05 pp inside the 2.58 floor**; at 64 the arms are +0.43 to +0.57 with a 2.44 floor. The direction is positive at both depths on all three arms. |
| Against SE-RRM's published numbers? | SA192 and SA256 read **ABOVE** their main-text 93.73 at 16 (+3.58, +4.20); all three are **WITHIN** the floor of their appendix variant's 95.4 and **WITHIN** at 64 (+1.26 to +1.40 over 98.22). |
| The selector? | R-SE-5 **CLEAN on all three at k32 and k128** (spurious ≤ 0.33 %). |
| The width order? | R-SE-6: monotone at 16 (SA128 < SA192 < SA256) but every gap is INSIDE the floor, so unresolved at one seed; at 64 SA192 and SA256 are tied to 0.00 pp. |

## 1. Integrity and provenance

**INTEGRITY PASS** on all three arms, with one note. Per arm the analyzer re-derived, from the pulled artefacts: the run's argv equals the banked (ladder) argv in **`steps` alone**, 30,000 → 50,000, and the banked config is the ladder's own; the `EXTENDED` record reads 30,000 → 50,000; 30,000 is in the resume record; the monitor rows are the 25 grids 2,000 … 50,000 and the first 15 are the ladder's rows unchanged; the last training step is 50,000; the selection re-derived with the registered rule equals `val_best.txt`; every row sits on the selected grid (the final row on `ckpt_latest`, the raw row without EMA) at its registered size; the 5,000-puzzle rows carry the ladder's ids; the D16 full row covers 422,786 distinct ids.
**The note:** SA128 carries a second resume at 41,500 — its spot preemption. **Its selected grid, 42,000, is the first grid after that resume**, i.e. 500 steps after a SOT carry reset. Integrity-safe and labelled, as SA192's ladder reset at 25,500 was; stated here because the selected checkpoint sits adjacent to it.
No NaN on any arm (0 non-finite loss rows). The resume guard's line — `RESUME-FLAGS-IDENTICAL 91 keys compared; allowed changes: steps 30000->50000` — is in all three manifests.

## 2. The registered letters (verbatim)

```
INTEGRITY                PASS [notes: SA128 resumes besides the join: [41500] (chain recycles; the SOT carry reset there, labelled)]
  SA128  selected 42000  16: 95.45 (n 422786)   64: 99.48 (n 422786)
  SA192  selected 40000  16: 97.31 (n 422786)   64: 99.62 (n 422786)
  SA256  selected 46000  16: 97.93 (n 422786)   64: 99.62 (n 422786)
R-SE-1 SA128 @16           INSIDE (+0.10 pp, only-new 159, only-ladder 154, n 5000, p 0.82; ladder grid 30000, new grid 42000)
R-SE-1 SA128 @64           INSIDE (+0.10 pp, only-new 24, only-ladder 19, n 5000, p 0.54; ladder grid 30000, new grid 42000)
R-SE-2 SA128               INTERIOR (selected 42000)
R-SE-1 SA192 @16           INSIDE (+0.52 pp, only-new 102, only-ladder 76, n 5000, p 0.061; ladder grid 30000, new grid 40000)
R-SE-1 SA192 @64           INSIDE (+0.00 pp, only-new 13, only-ladder 13, n 5000, p 1; ladder grid 30000, new grid 40000)
R-SE-2 SA192               INTERIOR (selected 40000)
R-SE-1 SA256 @16           INSIDE (-0.22 pp, only-new 52, only-ladder 63, n 5000, p 0.35; ladder grid 28000, new grid 46000)
R-SE-1 SA256 @64           INSIDE (-0.06 pp, only-new 11, only-ladder 14, n 5000, p 0.69; ladder grid 28000, new grid 46000)
R-SE-2 SA256               INTERIOR (selected 46000)
R-SE-3 SA128 @16           INSIDE (95.45 vs the width-192 triple 95.41; +0.04 pp)
R-SE-3 SA128 @64           INSIDE (99.48 vs the width-192 triple 99.05; +0.43 pp)
R-SE-4 SA128 @16 vs 93.73  WITHIN (95.45 vs SE-RRM's published 93.73 (main text); +1.72 pp; their one run)
R-SE-4 SA128 @16 vs 95.4   WITHIN (95.45 vs SE-RRM's published 95.4 (appendix Table A6); +0.05 pp; their one run)
R-SE-4 SA128 @64 vs 98.22  WITHIN (99.48 vs SE-RRM's published 98.22; +1.26 pp; their one run)
R-SE-3 SA192 @16           INSIDE (97.31 vs the width-192 triple 95.41; +1.91 pp)
R-SE-3 SA192 @64           INSIDE (99.62 vs the width-192 triple 99.05; +0.57 pp)
R-SE-4 SA192 @16 vs 93.73  ABOVE (97.31 vs SE-RRM's published 93.73 (main text); +3.58 pp; their one run)
R-SE-4 SA192 @16 vs 95.4   WITHIN (97.31 vs SE-RRM's published 95.4 (appendix Table A6); +1.91 pp; their one run)
R-SE-4 SA192 @64 vs 98.22  WITHIN (99.62 vs SE-RRM's published 98.22; +1.40 pp; their one run)
R-SE-3 SA256 @16           INSIDE (97.93 vs the width-192 triple 95.41; +2.53 pp)
R-SE-3 SA256 @64           INSIDE (99.62 vs the width-192 triple 99.05; +0.57 pp)
R-SE-4 SA256 @16 vs 93.73  ABOVE (97.93 vs SE-RRM's published 93.73 (main text); +4.20 pp; their one run)
R-SE-4 SA256 @16 vs 95.4   WITHIN (97.93 vs SE-RRM's published 95.4 (appendix Table A6); +2.53 pp; their one run)
R-SE-4 SA256 @64 vs 98.22  WITHIN (99.62 vs SE-RRM's published 98.22; +1.40 pp; their one run)
R-SE-5 SA128 k32           CLEAN (spurious 0.00 %)      R-SE-5 SA128 k128   CLEAN (spurious 0.00 %)
R-SE-5 SA192 k32           CLEAN (spurious 0.33 %)      R-SE-5 SA192 k128   CLEAN (spurious 0.21 %)
R-SE-5 SA256 k32           CLEAN (spurious 0.00 %)      R-SE-5 SA256 k128   CLEAN (spurious 0.21 %)
R-SE-6 SA192 vs SA256 @16  NARROWER-BEHIND / INSIDE (-0.62 pp, p 1.6e-105, n 422786)
R-SE-6 SA192 vs SA256 @64  NARROWER-BEHIND / INSIDE (-0.00 pp, p 0.94, n 422786)
R-SE-6 SA128 vs SA192 @16  NARROWER-BEHIND / INSIDE (-1.86 pp, p 0, n 422786)
R-SE-6 SA128 vs SA192 @64  NARROWER-BEHIND / INSIDE (-0.14 pp, p 1.6e-24, n 422786)
DESCRIPTIVE SA128        D128 20k 99.74, D256 5k 99.88, D128 50k 99.75, D256 50k 99.89; final 95.32, raw 93.51
DESCRIPTIVE SA192        D128 20k 99.80, D256 5k 99.86, D128 50k 99.79, D256 50k 99.87; final 97.46, raw 95.47
DESCRIPTIVE SA256        D128 20k 99.74, D256 5k 99.74, D128 50k 99.76, D256 50k 99.83; final 97.94, raw 96.72
```

## 3. Prediction scoreboard (credences written before any run)

| | prediction | credence | outcome |
|---|---|---|---|
| P1 | INTEGRITY PASS on all three | 0.85 | **held** |
| P2 | SA128 selected past 30,000 | 0.75 | **held** (42,000) |
| P3 | SA192 selected past 30,000 | 0.65 | **held** (40,000) |
| P4 | SA256 selected past 30,000 | 0.45 | **held** (46,000) |
| P5 | R-SE-1 INSIDE at both depths on SA192 and SA256 | 0.70 | **held** |
| P6 | R-SE-1 ABOVE-BEYOND at 16 on SA128 | 0.25 | failed (INSIDE, +0.10) |
| P7 | R-SE-3: SA256 ABOVE the width-192 triple at 16 (> 97.99) | 0.60 | **failed by 0.06 pp** (97.93) |
| P8 | R-SE-3: SA128 INSIDE at 16 | 0.75 | **held** |
| P9 | R-SE-3: all three INSIDE at 64 | 0.90 | **held** |
| P10 | R-SE-4: SA256 and SA192 ABOVE 93.73 at 16 | 0.80 | **held** |
| P11 | R-SE-4: SA128 WITHIN the floor of 95.4 at 16 | 0.70 | **held** (+0.05) |
| P12 | R-SE-5: CLEAN on all three at k32 | 0.75 | **held** |
| P13 | no NaN, no SCAN-DEADLOCK, no guard refusal | 0.90 | **held** |

**Brier 0.100** over the thirteen (0.25 = coin flips). Page one's expected accuracies: SA128 95.45 (expected 95.0–96.5 ✓), SA192 97.31 (96.8–97.8 ✓), SA256 97.93 (98.0–98.6, **0.07 low**); at 64 all three inside their bands. The one real miss is P7, and it missed by 0.06 pp on a 2.58 pp floor — a knife-edge, not a surprise about the model.

## 4. Post-results critique

**C1. The extension bought provenance, not accuracy.** R-SE-1 is INSIDE on every arm at both depths, so the ladder's 30k rows were already at their plateau; the honest statement is "the 50k budget moves every selection off the budget edge and changes the numbers by at most half a point", not "the arms improve with more training". What the run actually delivers: interior selections, full-set rows, and the four Table-1 rows.
**C2. SA256 vs our model at 16 is a knife-edge, and the floor is borrowed.** +2.53 pp against a 2.58 pp floor: INSIDE by 0.05 pp. The floor is twice the largest seeded spread of the MLP-mixer recipe; the attention arms' own seed spread is unmeasured (one seed each). Read it as "not resolved at one seed", never as "no difference" — and never as "the attention arm wins".
**C3. Puzzle-sampling significance is not seed resolution.** R-SE-6's p-values on 422,786 paired puzzles are astronomically small (p = 0 at 16 for SA128 vs SA192) while every gap is INSIDE the floor. The exact McNemar answers "would another draw of puzzles flip this?"; the floor answers "would another seed?". Only the second licenses a claim here.
**C4. The comparison to SE-RRM is now same-set but still cross-run.** Our rows and their published numbers are on the identical 422,786 puzzles, which removes the evaluation-set caveat the ladder carried; what remains is one run on each side, their selection rule unstated, and two published numbers for their model (93.73 main text with RoPE2D, 95.4 in appendix Table A6 with 1-D RoPE). Quote the appendix number beside the main-text one, as the letters do.
**C5. The extension's effect is measured on the 5,000, not the full set.** R-SE-1 pairs the new grid against the ladder's 30k grid on the ladder's 5,000 ids, because the ladder never ran a full-set row. The full-set numbers therefore have no 30k counterpart; "the extension changed little" is a statement about the 5,000.
**C6. The selected grid is not obviously the best grid.** SA192's final grid reads 97.46 and SA256's 97.94 **on the 50,000-puzzle subsample**, against selected rows of 97.31 and 97.93 **on the full set** — different sets, so these do not compare, and nothing here says the selection was wrong. Settling it would need the final grid on the full set (not run; ≈ 25–40 min per arm).
**C7. SA128's selected grid sits 500 steps after its preemption resume.** The carry reset at 41,500 is labelled and integrity-safe, but the selected checkpoint (42,000) is the first grid after it. If a reviewer asks, the neighbouring grids (40,000 and 44,000) are banked and can be read.
**C8. The depth rows saturate.** D128 and D256 read 99.74–99.89 on all three arms: at this depth the arms are at the ceiling of the measurement, and differences there are not interpretable at one seed.
**C9. Compute is not matched.** The attention arms cost ≈ 1.6 × (128), 2.2 × (192) and 2.45 × (256) our width-192 model's inference per row (measured on the k32 scans) and more per training step. Any row quoted beside the width-192 model must carry that.

## 5. Descriptive read (EXPLORATORY — `tools/lens_saext_read.py`, selftest 3/3; after the letters, moves none)

Accuracy by iteration on all 422,786 puzzles (the fixed start, EMA weights, the selected grid):

| | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| SA128 | 18.15 | 45.69 | 71.45 | 87.77 | 95.45 | 98.46 | 99.48 |
| SA192 | 32.68 | 65.85 | 81.90 | 92.40 | 97.31 | 99.05 | 99.62 |
| SA256 | 26.72 | 62.97 | 85.04 | 94.29 | 97.93 | 99.21 | 99.62 |
| our width-192, seed 0 | 17.05 | 45.71 | 72.65 | 89.31 | 95.91 | 98.24 | 99.10 |
| SE-RRM, published (one run) | 16.05 | 62.06 | 77.31 | 87.38 | 93.73 | 96.82 | 98.22 |

The attention arms commit far more in the FIRST iteration (SA192 32.7 %, SA256 26.7 % against our 17.1 % and SE-RRM's 16.1 %) and stay ahead of SE-RRM's curve from iteration 4 on. Our width-192 model overtakes SA128 at 8 and 16 iterations and is passed again by 64. The raw (non-EMA) weights read 93.51 / 95.47 / 96.72 against the selected rows: the EMA is worth 2–4 points on this model class, as in every earlier campaign.

### 5b. Restarts (EXPLORATORY; the PI asked, 2026-09-20; `tools/lens_saext_read.py` T3, selftest 5/5)

On the identical 5,000 puzzles, 64 iterations, EMA weights, the selected grid. "selected" = the draw with the smallest residual (the paper's rule); "verified" = at least one draw exact.

| arm / scan | fixed start | 1 random start | sel k4 | sel k8 | sel k32 | sel k128 | verified |
|---|---|---|---|---|---|---|---|
| SA128 k32 | 99.48 | 99.36 | 99.90 | 99.94 | 99.94 | — | 99.94 |
| SA128 k128 | 99.46 | 99.42 | 99.88 | 99.92 | 99.94 | 99.96 | 99.96 |
| SA192 k32 | 99.60 | 99.60 | 99.84 | 99.88 | 99.98 | — | 99.98 |
| SA192 k128 | 99.60 | 99.60 | 99.84 | 99.88 | 99.98 | 99.98 | 100.00 |
| SA256 k32 | 99.60 | 99.68 | 99.84 | 99.92 | 99.96 | — | 99.96 |
| SA256 k128 | 99.54 | 99.62 | 99.86 | 99.94 | 99.94 | 99.92 | 99.98 |
| our width-192 k128 triple (the paper's restart column) | | | | | | **99.85 ± 0.04** | 99.89 ± 0.03 |
| EqR at k128 | | | | | | 98.84 | |

Restarts are worth ≈ 0.4 pp over the fixed start on every arm and nearly all of it arrives by k 4–8. At k128 the arms read 99.92–99.98 selected against the triple's 99.85 ± 0.04. **SUPERSEDED BY THE REGISTERED RE-READ** (`Note_2026-09-20_Restart_Column.md`, at the PI's word; `tools/analyze_restarts.py` frozen at 8d95854, selftest 12/12): the floor computed from the three width-192 seeds on this same instrument is **0.160 pp selected / 0.120 pp verified**, and **every arm is INSIDE both** (+0.067 to +0.127 selected), so the restart column carries NO letter — although every arm is ahead of every seed pairwise. The sentence in this section that called the gaps "two to three times the seed half-spread" read as more margin than the rule allows and is withdrawn. SA192 solved all 5,000 with at least one of its 128 restarts (verified 100.00 on this sample). The selector does not degrade as restarts are added on any arm — the opposite of the recipe ablation's start-up arm, where the residual-selected accuracy FELL from k1 to k32.

## 6. What this means for the paper

1. **The attention arms can now stand as full rows** (the PI's open decision D7): same budget, same selection rule, same full test set, the same battery and the same Table-1 cells as the width-192 seeds, with interior selections and CLEAN selectors at k32 and k128.
2. **The supportable comparison to our own model:** at the paper's budget, on all 422,786 puzzles, SE-RRM's mixers inside our recipe read 95.45 / 97.31 / 97.93 at 16 iterations against our width-192 triple's 95.41 ± 0.54 — **every difference inside the seed floor, so unresolved at one seed**, with the direction positive and the compute 1.6–2.45 × higher. The narrowing claim, which is about our MLP-mixer design, is untouched by this run.
3. **The supportable comparison to SE-RRM:** their mixers inside our recipe, at 2.0M parameters, read 97.93 / 99.62 where their paper reports 93.73 (main text) or 95.4 (appendix) at 16 and 98.22 at 64 — one run on each side, their selection rule unstated. This replaces the ladder's 5,000-puzzle statement with one on their own evaluation set.
4. **One sentence the recipe ablation already licenses stays attached:** of the three recipe items tested there, only the optimizer's batch and learning rate moved accuracy, so the gap to their published numbers is not the mixer and not our damping, noise or start-up levers.
5. **Not claimed:** that the attention arms beat our model (C2), that wider is better among them (C3), that the extension raised accuracy (C1), or anything about depth beyond 64 (C8).

**Open, cheap, not run:** a restart column read as a LETTER would need its own registered floor (the triple's k128 half-spread is ±0.04) and ideally a second seed; as it stands §5b is descriptive. And the final grid on the full set for SA192 and SA256 (C6; ≈ 25–40 min per arm on one pod) would settle whether the 512-puzzle selector leaves anything on the table at this budget.
