# The attention arms: rows and wording for the paper (hand-off, 2026-09-20)

**From the ops/analysis session to the paper session.** Everything here is banked, analyzed under frozen rules, and traceable to a commit. Nothing in `paper/` was touched. Provenance: registration f2ebf57 (`Plan_2026-09-19_SA_Extension.md`), verdict 8e1fcea (`Report_2026-09-20_SA_Extension_Verdict.md`), the restart column's registration 8d95854 and result 2774228 (`Note_2026-09-20_Restart_Column.md`), analyzers `tools/analyze_saext.py` (sha256 1cb297db…) and `tools/analyze_restarts.py` (8307f56b…), data `runs/_saext_pull/` (crc32c 132/132), outputs `runs/_saext_pull/analysis/`.

## 1. What the arms are (one sentence for the methods section)

**SA128 / SA192 / SA256 are OUR reimplementation of SE-RRM's two mixers — self-attention over the 81 cells with 2D rotary positions, and attention across the nine digit fields in place of our mean — inside OUR block, loop and recipe**, at hidden size 128 / 192 / 256, seed 0, trained to 50,000 steps (the same budget as the paper's width-192 model) and selected by the same rule (the 512 validation puzzles every 2,000 steps, maximum with the earliest tie). They are NOT SE-RRM's model: their layer layout, dropout 0.2, batch 272, learning rate, 10,000 epochs and random early stop are not ported.

## 2. The rows (every cell: one seed, EMA weights, the registered fixed start)

| | SA128 | SA192 | SA256 | our width-192 triple | SE-RRM published |
|---|---|---|---|---|---|
| parameters | 623,397 | 1,057,957 | 1,967,653 | 790k (0.79M) | 2M |
| selected step (of 50,000) | 42,000 | 40,000 | 46,000 | 46,000 each | — |
| **16 iterations, all 422,786** | **95.45** | **97.31** | **97.93** | **95.41 ± 0.54** | 93.73 main text · 95.4 appendix A6 |
| **64 iterations, all 422,786** | **99.48** | **99.62** | **99.62** | **99.05 ± 0.18** | 98.22 |
| 128 iterations, 50,000 puzzles | 99.75 | 99.79 | 99.76 | (the paper's existing cell) | 98.84 |
| 256 iterations, 50,000 puzzles | 99.89 | 99.87 | 99.83 | (the paper's existing cell) | — |
| k128 restarts, residual-selected, 5,000 | 99.96 | 99.98 | 99.92 | 99.85 ± 0.04 | — |
| k128 restarts, verified (any draw), 5,000 | 99.96 | 100.00 | 99.98 | 99.89 ± 0.03 | — |
| k32 restarts, residual-selected, 5,000 | 99.94 | 99.98 | 99.96 | — | — |

Per-iteration on all 422,786 (for a figure or an appendix table; the same fixed start, EMA):

| | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| SA128 | 18.15 | 45.69 | 71.45 | 87.77 | 95.45 | 98.46 | 99.48 |
| SA192 | 32.68 | 65.85 | 81.90 | 92.40 | 97.31 | 99.05 | 99.62 |
| SA256 | 26.72 | 62.97 | 85.04 | 94.29 | 97.93 | 99.21 | 99.62 |
| our width-192, seed 0 | 17.05 | 45.71 | 72.65 | 89.31 | 95.91 | 98.24 | 99.10 |
| SE-RRM published (one run) | 16.05 | 62.06 | 77.31 | 87.38 | 93.73 | 96.82 | 98.22 |

**Protocol strings for the caption, exact:** 16 and 64 iterations on all 422,786 Sudoku-Extreme test puzzles; 128 and 256 iterations on a 50,000-puzzle uniform subsample (seed 20260822), the same subsample the width-192 cells use; restart columns on the 5,000-puzzle subsample, 64 iterations, 128 randomised starts, the draw with the smallest residual ("selected") or any exact draw ("verified"). One seed per attention arm; the width-192 row is three seeds.

## 3. The comparisons, with their labels (these are the frozen letters)

**Floors** (twice the largest seeded spread of the recipe, on the same instrument): 2.58 pp at 16 iterations, 2.44 pp at 64, **0.160 pp for the k128 selected column and 0.120 pp for verified** (computed from C5 / C7 / C8's own k128 rows: 99.88 / 99.88 / 99.80 and 99.88 / 99.92 / 99.86).

| comparison | result |
|---|---|
| each arm vs the width-192 triple, 16 and 64 iterations | **INSIDE the floor, every arm, both depths** (+0.04 to +2.53 at 16; +0.43 to +0.57 at 64). Unresolved at one seed. |
| each arm vs the triple, k128 selected and verified | **INSIDE the floor, every arm, both columns** (+0.067 to +0.127 selected), although each arm is ahead of all three seeds pairwise. No claim. |
| SA192 / SA256 vs SE-RRM's main-text 93.73 at 16 | **ABOVE** (+3.58, +4.20) |
| every arm vs SE-RRM's appendix Table A6 95.4 at 16 | **WITHIN** the floor |
| every arm vs SE-RRM's 98.22 at 64 | **WITHIN** the floor (+1.26 to +1.40) |
| restart selector (spurious rate) | **CLEAN on all three at k32 and k128** (≤ 0.33 %) |
| width order among the arms | monotone at 16, every gap INSIDE the floor; tied at 64. Unresolved. |

## 4. Wording you can use verbatim

1. **The row's introduction:** "We reimplemented SE-RRM's two mixers inside our block, loop and recipe and trained them at our budget and selection rule (one seed each at hidden 128, 192 and 256)."
2. **Against our model:** "At 50,000 steps on all 422,786 test puzzles, these attention-mixed models read 95.45 / 97.31 / 97.93 at 16 iterations and 99.48 / 99.62 / 99.62 at 64, against our width-192 model's 95.41 ± 0.54 and 99.05 ± 0.18. Every difference is inside the seed floor of our recipe (2.58 and 2.44 points), so at one seed per arm we do not resolve a difference; the direction is positive at both depths, and the attention arms cost 1.6–2.45× more per evaluated row."
3. **Against SE-RRM:** "On the same test set, our reimplementation of their mixers inside our recipe reads 97.93 at 16 iterations and 99.62 at 64 at 2.0M parameters, where their paper reports 93.73 (main text) or 95.4 (appendix Table A6) and 98.22 — one run on each side, and their checkpoint-selection rule is not stated."
4. **What the recipe ablation adds (already banked, `Report_2026-09-19_Recipe_Ablation_Verdict.md`):** "Reverting one recipe item at a time toward SE-RRM's published setup, only the batch size and learning rate of their hyperparameter table moved accuracy (−12.8 points at 16 iterations); damping, path noise and our start-up levers moved it by at most 0.2. The difference from their published number is therefore not the mixer."
5. **Restarts:** "With 128 randomised starts and residual selection the attention arms read 99.92–99.98 on the 5,000-puzzle subsample against our width-192 triple's 99.85 ± 0.04 — inside the seeded floor of that column (0.16 points), so we report the row without claiming a difference."
6. **Depth:** "At 128 and 256 iterations all rows sit between 99.75 and 99.89, at the ceiling of this measurement."
7. **The narrowing statement, scoped (from the ladder's verdict):** "Our narrowing result concerns our MLP-mixer design, whose cell mixer holds 124,416 parameters at every hidden size; it is not a claim about attention-mixed models, whose cross-cell parameters grow with width."
8. **Decimation, if you want it (from `Note_2026-09-19_Ladder_Adversarial_Pass.md` §3):** "The commit-then-repair signature holds under the attention mixers as well: after one iteration 93.7–95.2 % of empty cells are committed and about a quarter of them are wrong on puzzles that end up solved, repaired by iteration eight."

## 5. What must NOT be said

- **Not** that the attention arms beat our model, or that any of them is the best model: every comparison to the width-192 triple is inside the floor at one seed, at both depths and in both restart columns.
- **Not** that wider (or narrower) is better among the attention arms: monotone at 16 but inside the floor; tied at 64. The tiny p-values there are puzzle-sampling, not seed resolution.
- **Not** that the extension improved the arms: moving 30,000 → 50,000 changed the 5,000-puzzle numbers by at most 0.52 points (all inside the floor). What it bought is that every selection is now interior to the budget, so no row carries a "lower bound" caveat, plus the full-set and Table-1 cells.
- **Not** that SE-RRM is under-tuned or would reach our numbers under other settings: our optimizer arm landed 8.3 points BELOW their published figure, and their own two sources disagree on the Sudoku learning rate (paper Table A2 says 5e-4; their released Sudoku command passes none and their config default is 1e-4).
- **Not** "SE-RRM's model reads X in our recipe": these are our mixers-only reimplementation, not their model.
- **Do not quote 93.73 alone.** Their appendix Table A6 reports 95.4 for their 1-D RoPE variant; quote both or quote the appendix number.

## 6. Compute, to state beside any attention row

Measured on the same pod type: **per evaluated row, 1.6× (SA128), 2.2× (SA192) and 2.45× (SA256)** the width-192 model's wall time (from the k32 restart scans, the longest common row: 39 / 53 / 60 min against ~24.5). Training wall pace at the same batch: 3.25 / 2.32 / 1.96 steps/s. The ladder verdict's C2 also quotes per-training-step trainer ratios (1.5 / 2.1 / 4.9 against 14.2 steps/s); the second adversarial audit disputes the SA256 figure there, so **prefer the inference ratio above**, which I measured directly from row walls, and say which quantity you mean.

## 7. Open, cheap, if you want more

- **A second seed** on one arm (≈ $45, ≈ 6 h on one pod) is what would turn any of the INSIDE labels into a claim; nothing else will.
- **The final grid on the full set** for SA192 / SA256 (≈ 25–40 min each, one pod) would settle whether the 512-puzzle selector leaves anything on the table at this budget (verdict C6).
- Neither is needed for the rows above to go in as they stand.
