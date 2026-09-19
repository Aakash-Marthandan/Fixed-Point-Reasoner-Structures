# The recipe ablation — verdict (2026-09-19)

**Registration:** `Plan_2026-09-19_Recipe_Ablation.md` (§2–§4 locked at 3cee53b; the PI: "Change the three things to locate the advantage we have"). **Analyzer:** `tools/analyze_sablate.py`, frozen at 3cee53b — it, its two imports and the plan verified identical BY CONTENT HASH immediately before the run (75ad164b… / bcbb66cb… / 029f9f08… / 908e9584…), selftest 25/25, run byte-untouched and FIRST. **Data:** `runs/_sablate_pull/` (18 objects, crc32c 18/18); the reference = the ladder's banked SA256 rows, read, never re-run. **Spend:** ≈ $110 (plan $120–135, cap $170). Every number below is printed by a script; outputs in `runs/_sablate_pull/analysis/` (`sablate_verdict.txt`, `frozen_stdout.keep.txt`, `descriptive_read.txt`).

## 0. The short version

The ladder showed that SE-RRM's mixers inside OUR recipe read 98.24 / 99.60 where SE-RRM's paper reports 93.73 / 98.22, so the mixer is not the difference. This run moved ONE recipe item at a time to SE-RRM's published side (seed 0, hidden 256, fixed 30,000 steps, the identical 5,000 test puzzles).

| arm | the one item moved | 16 iterations | 64 iterations | against SA256 (paired) |
|---|---|---|---|---|
| SA256 (reference) | — | 98.24 | 99.60 | — |
| SA256L | EqR's damping and path noise OFF | 98.04 | 99.54 | **INSIDE** (−0.20 / −0.06 pp) |
| SA256S | our randomized init and anchor rows OFF | 98.42 | 99.76 | **INSIDE** (+0.18 / +0.16 pp) |
| SA256O | batch 768 → 272, lr 1e-4 → 5e-4 (SE-RRM's) | **85.42** | **91.50** | **BELOW-BEYOND** (−12.82 / −8.10 pp) |

**LOCATED: the optimizer's batch size and learning rate.** It is the only one of the three items that moves accuracy, and it moves it by five times the floor. Those settings are **TRM's** (batch 768, lr 1e-4, wd 1.0), which our recipe inherits; they are not a contribution of ours. EqR's damping and noise, and our two start-up levers, are worth nothing measurable in accuracy on this model at this budget.

**What it does NOT show:** that SE-RRM would reach 98 with TRM's settings. The arm lands 8.3 points BELOW SE-RRM's own published number, so inside THEIR recipe the same settings do much better than inside ours; their other choices (dropout 0.2, their block layout, their schedule, none ablatable today) interact with the optimizer. We moved our recipe toward theirs, never theirs toward ours.

**A source check made for this verdict changes how the optimizer arm should be read (2026-09-19, at the sources).** TRM's released settings are batch 768, lr 1e-4 constant after 2,000 warm-up steps, wd 1.0, β2 0.95, EMA (its README's Sudoku command + `config/cfg_pretrain.yaml`): our recipe's optimizer IS TRM's, item for item. SE-RRM's PAPER (arXiv 2603.02193, Table A2, Sudoku) states batch 272, **lr 0.0005**, wd 1, warm-up + constant, 10,000 epochs — the arm is faithful to that table. But SE-RRM's RELEASED Sudoku command sets `global_batch_size=272` and **no `lr=`**, and their `config/cfg_pretrain.yaml` default is **1e-4** (their ARC and Maze commands do pass `lr=0.0005`). Their two sources disagree on the Sudoku learning rate. If their run used 1e-4, our arm moved the learning rate further than they did, which would by itself explain why it lands 8 points below their number. The arm that would settle it — batch 272 at lr 1e-4, their released command's setting — was not registered and has not been run.

**What the start-up levers DO buy (one registered letter + an exploratory read):** the restart selector. Without them the selector is DIRTY (27.4 % spurious against 0.00 %), and adding restarts makes the residual-selected accuracy WORSE (99.12 → 98.58 from 1 to 32 restarts) where with them it improves (99.58 → 99.92).

## 1. Integrity and provenance

- **INTEGRITY PASS** on all three arms: each arm's trainer argv AND model config differ from SA256's banked ones in exactly the registered keys at the registered values; seed 0; 30,000 steps, no extension marker; n 5,000, EMA, the right depth, puzzle ids identical to SA256's on every row; each arm's rows and scan on one grid = its selection.
- No NaN on any arm (0 non-finite loss rows), no resume on any arm, no rematerialization retry. SA256O's first full-batch run was the on-chip preflight (the Mac cannot hold that batch); it passed.
- Selected grids: SA256L 20k · **SA256S 30k = the budget's last grid (a lower bound)** · SA256O 22k · (SA256 28k).
- Ops disclosure: no test-set value and no validation value was read before the analyzer ran. The tick mask was inert under macOS's BSD sed until 06:04Z; the lines that passed through it carried the TRAINING loss only (no VALBEST line had been printed yet). The structure checks read the first field of `val_best.txt` (the grid) only.

## 2. The registered letters (verbatim)

```
INTEGRITY                PASS
  SA256  16: 98.24 (n 5000)   64: 99.60 (n 5000)
  SA256L 16: 98.04 (n 5000)   64: 99.54 (n 5000)   [the LOOP: damping and path noise off]
  SA256S 16: 98.42 (n 5000)   64: 99.76 (n 5000)   [the START-UP levers: randomized initialization and anchor rows off]
  SA256O 16: 85.42 (n 5000)   64: 91.50 (n 5000)   [the OPTIMIZER: batch 272, learning rate 5e-4]
R-AB-1 SA256L vs SA256 @16     INSIDE (-0.20 pp, only-SA256L 60, only-SA256 70, n 5000, p 0.43)
R-AB-2 SA256L vs SE-RRM @16    ABOVE (98.04 vs the published 93.73; +4.31 pp; different evaluation sets, their one run)
R-AB-1 SA256L vs SA256 @64     INSIDE (-0.06 pp, only-SA256L 15, only-SA256 18, n 5000, p 0.73)
R-AB-2 SA256L vs SE-RRM @64    WITHIN (99.54 vs the published 98.22; +1.32 pp; different evaluation sets, their one run)
R-AB-1 SA256S vs SA256 @16     INSIDE (+0.18 pp, only-SA256S 67, only-SA256 58, n 5000, p 0.47)
R-AB-2 SA256S vs SE-RRM @16    ABOVE (98.42 vs the published 93.73; +4.69 pp; different evaluation sets, their one run)
R-AB-1 SA256S vs SA256 @64     INSIDE (+0.16 pp, only-SA256S 16, only-SA256 8, n 5000, p 0.15)
R-AB-2 SA256S vs SE-RRM @64    WITHIN (99.76 vs the published 98.22; +1.54 pp; different evaluation sets, their one run)
R-AB-1 SA256O vs SA256 @16     BELOW-BEYOND (-12.82 pp, only-SA256O 17, only-SA256 658, n 5000, p 3.8e-170)
R-AB-2 SA256O vs SE-RRM @16    BELOW (85.42 vs the published 93.73; -8.31 pp; different evaluation sets, their one run)
R-AB-1 SA256O vs SA256 @64     BELOW-BEYOND (-8.10 pp, only-SA256O 2, only-SA256 407, n 5000, p 1.3e-118)
R-AB-2 SA256O vs SE-RRM @64    BELOW (91.50 vs the published 98.22; -6.72 pp; different evaluation sets, their one run)
R-AB-3 LOCATED @16             LOCATED: SA256O (-12.82 pp = 284 % of the 4.51 pp to SE-RRM)
R-AB-3 LOCATED @64             LOCATED: SA256O (-8.10 pp)
R-AB-4 SELECTOR SA256          CLEAN (spurious k32 0.00 %)
R-AB-4 SELECTOR SA256L         CLEAN (spurious k32 0.36 %)
R-AB-4 SELECTOR SA256S         DIRTY (spurious k32 27.38 %)
R-AB-4 SELECTOR SA256O         CLEAN (spurious k32 0.00 %)
STABILITY SA256L         selected 020000; validation max 98.44, end 98.05; scan fixed start 99.62, one random start 99.54 (gap -0.08 pp), verified 99.96; non-finite loss rows 0
STABILITY SA256S         selected 030000 EDGE (a lower bound: the budget ended on the maximum); validation max 98.24, end 98.24; scan fixed start 99.74, one random start 99.12 (gap -0.62 pp), verified 99.98; non-finite loss rows 0
STABILITY SA256O         selected 022000; validation max 90.23, end 87.89; scan fixed start 91.52, one random start 91.74 (gap +0.22 pp), verified 97.38; non-finite loss rows 0
```

## 3. Prediction scoreboard (credences written before any run)

| | prediction | credence | outcome |
|---|---|---|---|
| P1 | INTEGRITY PASS on all three | 0.85 | **held** |
| P2 | SA256S INSIDE at both depths | 0.70 | **held** (+0.18 / +0.16) |
| P2b | SA256S: one random start ≥ 5 points below its fixed start | 0.65 | **failed** (−0.62 pp: the basin is wide without randomized-init training) |
| P2c | SA256S selector DIRTY | 0.55 | **held** (27.38 %) |
| P3 | SA256L INSIDE at both depths | 0.60 | **held** (−0.20 / −0.06) |
| P4 | SA256O BELOW-BEYOND at 16 | 0.50 | **held** (−12.82; far larger than expected: the stated band was 93–98) |
| P4b | SA256O aborts on a NaN | 0.15 | did not happen |
| P4c | SA256O selected at the budget edge | 0.40 | did not happen (22k, on a flat noisy plateau) |
| P5 | an arm WITHIN the floor of SE-RRM's 93.73 at 16 | 0.35 | **failed by the letter** (R-AB-2 prints BELOW for SA256O, ABOVE for L and S); the plan's parenthesis "(i.e. ≤ 96.31)" would count SA256O — the prediction was worded two ways (see C8) |
| P6 | NOT-LOCATED at 16 | 0.30 | failed (LOCATED) |
| P7 | NOT-LOCATED at 64 | 0.80 | **failed**: I reasoned that a difference inside the floor cannot be located; the arm fell by six times that difference |

Brier score 0.198 over the eleven (0.25 = coin flips). The two large misses share one cause: I priced the optimizer arm as a mild change (expected 93–98 at 16) and it was the largest effect this program has measured from a single recipe item.

## 4. Post-results critique

**C1. The optimizer arm moved four quantities together, and the letter cannot separate them.** Batch (768 → 272), learning rate (1e-4 → 5e-4), training rows seen at a fixed step budget (23.0M → 8.2M, registered), and — not registered — the per-step weight decay, which under decoupled AdamW at wd 1.0 scales with the learning rate (×5). §5 T1 addresses the rows descriptively; batch against learning rate against decay needs arms this run does not have.
**C2. LOCATED is a statement about OUR recipe, not about SE-RRM's run.** The arm reads 8.3 points below their published 93.73, so their settings are worth far more inside their recipe than inside ours. The supportable sentence: "in our recipe the optimizer's batch size and learning rate is the only one of the three items that carries accuracy". Not supportable: "SE-RRM is under-tuned", "SE-RRM would reach 98 under TRM's settings". Verified at the sources for this verdict: their paper's Table A2 gives warm-up + constant learning rate at 0.0005 for Sudoku, while their released Sudoku command omits `lr=` and their config's default is 1e-4 — so which learning rate produced their 93.73 is not determinable from public sources, and the arm tests their TABLE, not necessarily their RUN. Their 10,000 epochs are, if an epoch is one pass over the 1,000 puzzles as in the HRM codebase they build on (not verified by running their code), ≈ 10M training rows — close to the arm's 8.2M, so the registered rows confound is also a feature of their setup rather than an artifact of ours.
**C3. The floors are borrowed.** 2.58 / 2.44 pp are twice the seed spread of the MLP-mixer model; the attention model's own seed spread is unmeasured. The verdict does not depend on it: −12.8 is five floors, ±0.2 is a thirteenth of one.
**C4. SA256S is a lower bound and it already matches the reference**, so INSIDE stands however its plateau resolves.
**C5. SA256O's selection rode noise, and the test row says so honestly.** Its validation curve is flat at 86–88 % from step 6,000 to 30,000 with one 90.23 spike at 22k (512 puzzles: one standard error ≈ 1.4 pp; the maximum of thirteen flat grids sits ≈ 2–3 pp above the plateau). The test row of that grid, 85.42, is an unbiased read of the selected checkpoint; the 4.8-point validation-to-test gap is the winner's curse, not leakage (the other arms' gaps are −0.4 and +0.2).
**C6. The arm under-fits; it does not memorize.** Train exact 14.4 % against 33.9 %, train loss 0.654 against 0.512, and the RAW weights validate at 62.9 % beside the EMA's 87.9 % (the reference: 94.3 beside 97.9). The raw iterate is bouncing: the step is too large for the batch under a constant schedule in this recipe.
**C7. A rule-design defect: R-AB-3 printed "284 % of the difference".** A share is meaningful in [0, 100]; past 100 the honest label is OVERSHOOTS (the arm falls below the external number). The floor guard I wrote covered the denominator, not the range. Lesson entered.
**C8. P5 was worded two ways** ("WITHIN the floor of 93.73" and "≤ 96.31"), which disagree exactly when an arm falls below the band. A prediction names the analyzer letter it is scored on. Lesson entered.
**C9. The reduced battery stops at 64 iterations and 30,000 steps.** EqR's damping may matter for stability deeper (128, 256 iterations) or later in training; this run cannot see either. "Damping carries no accuracy" is scoped to ≤ 64 iterations at 30k, one seed.

## 5. Descriptive read (EXPLORATORY — `tools/lens_sablate_read.py`, selftest 5/5; written after the letters; moves none)

**T1. Fewer training rows do not explain the fall.** SA256O's validation (EMA, 16 iterations) reaches 87.3 % by step 6,000 (1.6M rows) and then stays between 85.5 and 90.2 for the remaining 24,000 steps: it is on a plateau for four fifths of its budget, not still climbing. At matched rows it leads the reference early (its steps are five times larger) and trails it from ≈ 5M rows on: −2.5 pp at 5.4M, −8.5 to −9.3 at 6.5–7.6M, −7.4 at 8.2M, while the reference is still rising (95.3 at 8.2M rows, 98.6 at 21.5M). A rows-matched rerun (84.7k steps) would most likely sit on the same plateau.
**T2. The train side:** as in C6. SA256S fits the training set slightly better without the anchor rows (37.4 % train exact against 33.9 %).
**T3. Accuracy by iteration, the identical 5,000:**

| | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| SA256 | 25.60 | 65.18 | 83.76 | 94.00 | 98.24 | 99.16 | 99.60 |
| SA256L | 44.98 | 70.30 | 85.64 | 94.14 | 98.04 | 99.26 | 99.54 |
| SA256S | 47.30 | 73.04 | 86.82 | 94.74 | 98.42 | 99.42 | 99.76 |
| SA256O | 29.96 | 60.14 | 72.48 | 80.16 | 85.42 | 88.94 | 91.50 |
| SE-RRM (published; full test set, one run) | 16.05 | 62.06 | 77.31 | 87.38 | 93.73 | 96.82 | 98.22 |

Removing the damping, or removing the randomized start in training, nearly doubles the share solved in ONE iteration (25.6 → 45–47 %) and changes nothing from iteration 8 on. SA256O is below SE-RRM's published curve from iteration 2 on.
**T4. The scan (32 restarts, 64 iterations):**

| | fixed start | one random start | residual-selected at 1 / 8 / 32 restarts | verified | spurious |
|---|---|---|---|---|---|
| SA256 | 99.60 | 99.58 | 99.58 / 99.82 / 99.92 | 99.94 | 0.00 % |
| SA256L | 99.62 | 99.54 | 99.54 / 99.94 / 99.96 | 99.96 | 0.36 % |
| SA256S | 99.74 | 99.12 | 99.12 / 99.30 / **98.58** | 99.98 | **27.38 %** |
| SA256O | 91.52 | 91.74 | 91.74 / 95.90 / 97.34 | 97.38 | 0.00 % |

Without the start-up levers the solution is still reached from almost every random start (verified 99.98 %), but wrong endpoints become as stationary as right ones, so picking the smallest residual gets WORSE as restarts are added. This is the X5-long selector inversion in miniature, on the attention-mixed model, produced by removing two flags. The levers' product is a trustworthy selector, not fixed-start accuracy — the reading the MLP-mixer model gave in Night A (accuracy flat), now on a second mixer. One seed.

## 6. What this means for the paper

1. **The SE-RRM paragraph gains one sentence and loses nothing.** "SE-RRM's mixers inside our recipe reach 98.2 / 99.6 (2.0M parameters, one seed, 5,000 puzzles). Reverting one recipe item at a time toward SE-RRM's published setup, only the batch size and learning rate of SE-RRM's hyperparameter table (272, 5e-4, against TRM's 768, 1e-4, which we inherit) moves accuracy (−12.8 pp at 16 iterations); EqR's damping and noise and our start-up levers move it by at most 0.2 pp." No statement about what SE-RRM's model would reach under other settings.
2. **Credit, stated plainly:** the accuracy of the attention-mixed model in our recipe is SE-RRM's mixers + TRM's optimizer settings. Our contributions on that model are the ones the paper already claims for the MLP-mixer model: the levers make restarts selectable (R-AB-4 + T4), they do not raise fixed-start accuracy.
3. **No headline number changes.** The paper's model remains the width-192 MLP-mixer triple; the attention arms remain one-seed rows with their compute stated.
4. **Optional, appendix:** the selector reading replicates on a second mixer (T4), and "commit, then repair" holds under it (`Note_2026-09-19_Ladder_Adversarial_Pass.md` §3).

**One follow-up is worth the PI's decision (not launched):** SA256 at **batch 272, lr 1e-4** — SE-RRM's released Sudoku command's optimizer. One pod, ≈ 3 h, ≈ $25 at SA256O's measured pace. If it reads near SA256, the fall is the learning rate's alone and the paper's sentence becomes "…only the learning rate stated in SE-RRM's hyperparameter table moves accuracy; their released command's batch size does not"; if it reads near SA256O, the batch size carries it. Either way it separates the two quantities the letter cannot (C1), and it removes the ambiguity in what "SE-RRM's setting" means. Without it, item 1's sentence must say "the batch size and learning rate of SE-RRM's hyperparameter table (their released command omits the learning rate)". Separating the decay coupling is for our own understanding or paper 2.
