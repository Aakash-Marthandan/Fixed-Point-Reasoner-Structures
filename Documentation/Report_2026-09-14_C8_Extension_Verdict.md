# The C8 extension — verdict and what we learn (2026-09-14)

**Registration:** `Documentation/Plan_2026-09-14_C8_Extension.md` (commit 543e7bf). **The PI's reporting decision (B):** recorded 07:05Z, before any data (63a3917, plan §8).

**Tools:**
- **The frozen analyzer:** `tools/analyze_c8x.py` + `tools/c8x_valcurve.py`, 0-diff against 543e7bf, selftest 19/19 → `runs/_c8x_pull/analysis/c8x_verdict.{txt,json}`.
- **Descriptive lenses** (no rules): `tools/lens_c8x_learn.py` (selftest 5/5) → `runs/_c8x_pull/analysis/learn.txt`; `tools/lens_width_seed.py` on the budget-matched staging root → `runs/_c8x_pull/stage_bm/analysis/width_seed.txt`.

## 1. The run and the close

**Node and training.** One spot v6e-8 in asia-south1-c, created 07:02:23Z, no preemption.
- **The resume, verified on the node:** the chain restored C8's banked state from `champ/C8_pretrain.tgz` read-only; the trainer logged `RESUMED from ckpt_latest.pkl at step 30000` (8 devices × 96 rows, the paper-final layout).
- **Timeline:** training reached 50k at 08:26Z, the battery finished at 10:02Z, the 63 held-out grids at 10:58Z, the filler rows at 12:47Z, and the 48k extra rows at 14:02Z.
- **Completion:** `CHAIN-C8X-COMPLETE` + `c8x_final.tgz` at 14:05:47Z; the supervisor then tore the node down (`Deleted tpu`, `down rc=0` at 14:07:32Z).

**Fleet zero** was verified at the source in six zones, with nothing queued. **Spend:** 7.09 node-hours ≈ **$57** at about $8/h (inferred), inside the $45–56 projection plus the extra rows.

**Data integrity (the PI's directive).**
- **Pulls:** all 83 objects under `c8x/` were pulled with `gcloud storage` and crc32c-compared into `runs/_c8x_pull/`, never over `runs/`.
- **`champ/` untouched:** its listing (2,182 objects with sizes, timestamps and paths) is **byte-identical** before launch and after the run. Nothing of the earlier runs was written, moved or deleted.

## 2. The registered reading

**INTEGRITY PASS.** All seven checks hold:
- **I1:** argv equals the paper-final C8's except out and steps.
- **I2:** monitor rows ≤ 30k identical, the extension's rows complete, resume at 30000.
- **I3:** the 22k and 30k grids are sha256-identical to the paper-final's.
- **I4:** every n-gate holds.
- **I5:** one checkpoint for the monitor pick's rows.
- **I6:** 21 held-out grids per model.
- **I7:** identical paired sets.

| rule | reading |
|---|---|
| R-X1 MONITOR-PICK | **MOVED(46k)** (provenance = replay) |
| R-X2 VAL-PEAK | **PEAK-LATE(48k)** · **22K-BELOW** (48k − 22k +2.58 pp on the 10k, p 5e-23) · **HYP-OUT**: 46k is 53 puzzles under the peak, just outside the 50-puzzle plateau {48k, 50k} · **HYP-ABOVE-22K** (+2.05 pp, p 4e-14) |
| R-X3 BETTER-POINT | g_val = 48k **BETTER**: D16 full 95.73 vs 92.98 (+2.74), D64 full 99.14 vs 97.97 (+1.17), both p ≈ 0. g_mon = 46k **BETTER**: +2.50 / +1.21 |
| R-X4 SEED-OR-BUDGET | **BUDGET-MOSTLY**: R = 1.38 with g_val, 1.43 with g_mon. The extra budget recovers more than C8's whole D64 gap to C7 |
| R-X5 TRIPLE-SENS | **HOLDS**: 99.05 ± 0.18 with C8 := g_mon (99.04 ± 0.17 with g_val) |
| R-X6 W192-PLATEAU | **SHARED-PEAK**: held-out peaks C5 46k, C7 48k, C8 48k (span 2k) · **MONITOR-IN-PLATEAU 1/3**. The plateaus: C5 {46, 48, 50}k contains its pick; C7 {42, 44, 48, 50}k does not (46k is 61 puzzles under); C8 {48, 50}k does not. |

The analyzer prints C7's plateau as the range "42000..50000". The plateau is a set of grids, non-contiguous here, and the count reads the set correctly.

**Scoreboard: 5 of 11 HIT.**
- **Hits:** MOVED, PEAK-LATE, 22K-BELOW, g_val BETTER, SHARED-PEAK.
- **Misses:** HYP-IN-PLATEAU, BOTH (it read BUDGET-MOSTLY), SOFTENS (it read HOLDS), MONITOR-IN-PLATEAU ≥ 2/3, and both g_val accuracy bands, which missed ABOVE.

The extension paid more than predicted.

## 3. C8's rows

EMA single pass; full = 422,786:

| grid | D16 full | D64 100k | D64 full | D128 20k / 50k | D256 5k / 50k | k32 residual / verified | mean first-exact step |
|---|---|---|---|---|---|---|---|
| 22k (the registered protocol) | 92.98 | 97.91 | 97.97 | 98.47 / 98.46 | 98.80 / 98.75 | 99.74 / 99.82 | 5.99 |
| **46k (the monitor's pick; the paper under (B))** | **95.49** | 99.15 | **99.18** | 99.45 / 99.51 | 99.64 / 99.64 | 99.82 / 99.84 | 4.54 |
| 48k (the held-out pick) | 95.73 | – | 99.14 | – | – | – | 4.48 |

## 4. What we learn (descriptive; `learn.txt`)

**(A) The PI's hypothesis, split in two.**
- **Where the best point is: confirmed.** All three width-192 seeds peak at 46–48k on the 10k held-out set.
- **That monitor noise stopped C8: not supported.**
  - Replayed at 30k under the held-out instrument (about 20× tighter), C8's argmax by 30k is still 22k (9,337 of 10,000, against 9,269 at 30k), so the extension would not have fired either.
  - C8 truly plateaued and dipped from 22k to 30k, then rose again after 30k (+326 puzzles to 48k).
  - What failed is the rule's assumption that a peak before the last 4k means training is done. A trajectory with a second rise defeats any rule that only looks inside the budget.

**How precise the monitor is.**
- **Ranking:** it ranks grids well over the whole run (Spearman with the held-out curve +0.90, +0.94, +0.94) but only moderately on the late plateau where selection happens (+0.70, +0.71, +0.62).
- **Regret:** its pick costs 0, 61 and 53 held-out puzzles (0 / 0.61 / 0.53 pp).
- **On test, for C8:** the held-out pick minus the monitor pick is +0.24 pp at D16 (p 8e-12) and −0.04 pp at D64 (p 6e-3). A better selector buys a little at 16 iterations and nothing at the paper's headline depth.

**(B) Every width-192 seed collapses mid-training, then peaks late** (the held-out single pass, exact of 10,000):

| seed | peak before collapse | collapse (drawdown) | 30k | late peak (30k → peak) |
|---|---|---|---|---|
| C5 | 14k 9,186 | 22k 1,523 (−7,663) | 9,544 | 46k 9,598 (+54) |
| C7 | 18k 9,262 | 22k 7,778 (−1,484) | 9,428 | 48k 9,540 (+112) |
| C8 | 12k 8,966 | 16k 7,862 (−1,104) | 9,269 | 48k 9,595 (+326) |

- The registered excursion detector (raw 512-monitor under 50 %) fired on 2 of 3; the held-out curves show the collapse on 3 of 3, at different depths.
- Every seed recovers; the late rise is largest for the seed that collapsed earliest.

**(C) Extra training makes the cell decide faster.** C8's exact accuracy by iteration on the full set, 22k → 46k:

| step | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| 22k | 1.18 | 9.80 | 57.58 | 83.38 | 92.98 | 96.59 | 97.97 |
| 46k | 3.13 | 46.14 | 74.71 | 89.02 | 95.49 | 98.20 | 99.18 |
| gain | +1.94 | **+36.34** | +17.13 | +5.63 | +2.50 | +1.60 | +1.21 |

The mean first-exact step falls from 5.99 to 4.54. Most of the extension's return is the same answers reached in fewer iterations; the D64 gain is the residual.

**(D) The remaining errors are mostly seed-specific, and width 192's shared core is half of width 384's.** At D64 on the full set:

| triple | failures per seed | failed by all three | any seed solves | shared share of a seed's failures | mean rating, shared failures (all puzzles 22.1) |
|---|---|---|---|---|---|
| width 192, budget-matched | 3,571 / 5,002 / 3,480 | **969 (0.23 %)** | 99.77 % | 24 % | 31.1 |
| width 192, registered (C8 at 22k) | 3,571 / 5,002 / 8,578 | 1,048 (0.25 %) | 99.75 % | 18 % | 31.3 |
| width 384 | 6,465 / 10,371 / 5,626 | **2,215 (0.52 %)** | 99.48 % | 30 % | 29.6 |

"Any seed solves" is a coverage column: a three-model portfolio needs a verifier to choose, and Sudoku's constraints are one.

**(E) No memorization by 50k at width 192.**
- The 50k grid is +0.29 pp over 46k on the identical 50k test subsample at D16 (p 5e-3).
- The held-out set at 50k reads 9,557 against the 9,595 peak.
- The training-batch exact rate is flat: 0.237 at 30k, 0.264 at 46k, 0.258 at 50k.

The recipe may still be short of its best at 50k.

## 5. Width, budget-matched (`width_seed.txt` on `stage_bm`)

**Paired width 192 − width 384, per seed, on identical puzzles:**

| depth | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| D16 full | +0.81 | +0.49 | −0.14 | +0.39 |
| **D64 full** | +0.68 | +1.27 | +0.51 | **+0.82** |
| D128 20k | +0.47 | +1.05 | +0.46 | +0.66 |
| D256 5k | +0.46 | +1.08 | +0.30 | +0.61 |

Every p ≤ 6e-3.

**Width buys earlier decisions:**
- **The wide cell is ahead at step 4 on every seed** (79–81 % vs 72–75 %).
- **The narrow cell passes it** at steps 10, 14 and 18, and gains more from 16 to 64 iterations (+3.25 / +3.99 / +3.69 vs +3.38 / +3.21 / +3.04).
- **Mean first-exact step:** 3.7–3.9 at width 384 against 4.4–4.8 at width 192.

**At equal arithmetic per puzzle, width 192 wins on every seed** (measured 14.82 vs 49.95 GMAC per iteration):

| budget per puzzle | width 384 | width 192 | per seed |
|---|---|---|---|
| ≈ 950 GMAC | 19 iterations, 95.74 % | 64 iterations, 99.05 % | +3.28 / +3.81 / +2.86 |
| ≈ 400 GMAC | 8 iterations, 90.22 % | 27 iterations, 97.62 % | – |

**The extra width is spent memorizing the 1,000 training puzzles:**
- **Width 384:** training-batch exact 0.36–0.39 at 30k; the width-384 runs trained to 50k reach 0.52–0.78 while their held-out monitor falls 7–12 pp.
- **Width 192 at 50k:** 0.19–0.28, with the monitor within 1.4 pp of its peak.

**Scope:**
- One task and one training recipe (lr 1e-4, wd 1.0: the field's setting for its 5M models, not tuned per width).
- Three seeds per width and a thousand training puzzles.

## 6. The paper

Decision (B) is applied: `Report_2026-09-14_Paper_Final_Verdict.md` §13. The width-192 row is 95.41 ± 0.54 / 99.05 ± 0.18 / 99.40 ± 0.13 / 99.56 ± 0.10; the abstract reads 99.0 ± 0.2. With the 48k rows in, the readings above are final and change none of the paper's numbers, which use the monitor's pick.

## 7. Lessons for the next registration

- **Replace the extension rule.** "The selected grid inside the last 4k of the budget" cannot see a second rise, and the width-192 recipe has one on every seed. Fix the budget per width (≥ 50k for width 192, where no seed has started memorizing), or extend while the held-out curve's running maximum is still rising.
- **Select on a large held-out set.** The 10k held-out set resolves about 0.2 pp against the monitor's 1 pp, and never touches the test set. At the headline depth the monitor's pick was already within noise, so the larger set matters most at shallow depth.
- **Narrow recursive cells are non-monotone early in training.** Read collapse-and-recover on the held-out curve, not the raw 512-puzzle monitor, which caught 2 of 3.
