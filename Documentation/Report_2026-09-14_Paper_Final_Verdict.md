# The paper's final Sudoku runs — verdict (2026-09-14)

**Registration:** `Documentation/Plan_2026-09-13_Paper_Final_Runs.md` (commit 7a99d9e). **Analyzer:** `tools/analyze_paperfinal.py`, 0-diff against 7a99d9e (sha256 029f9f08…), selftest 25/25, run on a staging root of symlinks (`runs/_paperfinal_pull/stage`; nothing extracted over `runs/`). **Output:** `runs/_paperfinal_pull/stage/analysis/paperfinal_verdict.{txt,json}`. **Descriptive lens (no rules):** `tools/lens_selection_seed.py` (selftest 6/6) → `stage/analysis/selection_seed.txt`.

## 1. What ran, and the close

One spot v6e-8 in asia-south1-c, node created 12:13:32Z Sep 13 and deleted 03:35:56Z Sep 14 by the supervisor on `CHAIN-PAPERFINAL-COMPLETE` (03:31:06Z). No preemption. That is 15.4 node-hours, **about $123** at about $8/h (inferred rate), under the registration's $180–230 projection. Fleet zero was verified at the source in all six zones with nothing queued.

Everything registered was banked: C7 (extended to 50k, selected 46k) and C8 (not extended, selected 22k) through the full champion battery; their D64 rows on all 422,786 puzzles and their D128/D256 rows on the 50k subsample; EqR's released weights at k 128 on the identical 5k; and C4's D64 row on the full set. All 37 objects were pulled to `runs/_paperfinal_pull/` and crc32c-verified against GCS.

**Two pull defects, both caught by the crc check before any reading:**
- A `gsutil cp` hung after writing all of `C7_pretrain.tgz`. The two pretrain tarballs were re-fetched with `gcloud storage` into `tgz_gc/` and verified.
- The local `paperfinal_final.tgz` was a 919-byte leftover from the first zsh pull attempt, byte-identical to `calib_C7_vsel.tgz`. The skip-if-present rule had kept it. It is renamed `STALE_…bad`; the real 113,288-byte object is in `tgz_gc/`, verified.

The hung gsutil process (pid 33005 and its children) is still alive and holds only a temp file. It is the PI's to kill.

## 2. INTEGRITY: the frozen reader fails on a format defect; the registered check passes

The frozen analyzer prints `INTEGRITY FAIL: C7: no config argv; C8: no config argv`.

**The data is fine.** The trainer stores `config.json["argv"]` as the parsed argparse namespace (a dict). The reader accepts only a token list: the selftest fixture wrote a list, and the pre-registration validation on real rows never touched this gate. As a result the reader never compared any recipe, C5's included.

**The registered comparison, run on the dict form (addendum, reader untouched):** C7's and C8's argv equal C5's in every key except `seed` (1 / 2), `out`, `steps` and `remat` (C5 ran with the registered OOM retry; C7 and C8 without it). The only other differences are four keys the DEC-ARC build added to the trainer after C5 trained:
- `table_lr` = None
- `table_wd` = None
- `w_void` = None
- `decarc_heads` = 4

On the Sudoku DEC path each is inert by construction. With both table knobs None, the optimizer takes the unchanged default branch, and `w_void` / `decarc_heads` are read only under `cell == "decarc"`. The model configs differ only in those keys, and both give n_params 789,125. Every other INTEGRITY clause passed: every n-gate, one checkpoint per arm across all vsel rows, k/t on the scans, and EqR's 5k idx set equal to all seven champion arms' k128 sets.

**Reading.** The registration says no letter is read if INTEGRITY fails. The letters below are read **conditional on this addendum**, labeled, because the failure is a reader-format defect and the registered condition itself holds. **The PI confirms or overrides this reading.** If overridden, the paper keeps the one-seed labels (plan §5).

## 3. The registered letters

| rule | letter | numbers |
|---|---|---|
| R-PF-1 SEEDS-W192 | **WIDE** | D16 full spread 2.92 pp (C5 95.91, C7 94.82, C8 92.98); D64 full spread 1.18 pp |
| R-PF-2 HEADLINE-W192 | **SOFTENS** | the width-192 triple: D16 full 94.57 ± 1.46 · **D64 full 98.65 ± 0.59** · D128 50k 99.05 ± 0.49 · D256 50k 99.26 ± 0.41 |
| R-PF-3 WIDTH-PAIRED | **PARITY** | D64 full mean Δ +0.42 pp: C5−C0 +0.68 (4,651/1,757), C7−C1 +1.27 (7,761/2,392), C8−C2 −0.70 (3,562/6,514); each p ≤ 1e-192. At D16 (descriptive), PARITY with mean −0.45 |
| R-PF-4 SELECTOR-W192 | **SELECTOR-CLEAN** | C7: spurious 0.20 %, t1r@32 99.80 = verified 99.80. C8: spurious 0.23 %, t1r 99.74 vs verified 99.82 (inside the .1 pp tolerance) |
| R-PF-5 DEPTH-W192 | **MONOTONE + ZERO-REGRESSION** on both | 0 puzzles lost from D16 to D64 on the full set, for each arm |
| R-PF-6 MEMORIZATION | **NOT-MEMORIZED** on all three | vsel − final: C5 −0.12, C7 −0.07, C8 +0.87 pp; end CE .53–.55 |
| R-PF-7 EXTENSION | C7 **EXTENDED+PAID** (46k); C8 **NOT-EXTENDED** (22k) | C5 EXTENDED; the reader shows "selected None" for C5 only because its `val_best.txt` is not in the local champion pull (its grid is 46k by the provenance field) |
| R-PF-8 EXCURSION | C5 EXCURSION 16–28k (the detector is valid); C7 **EXCURSION** 22–26k; C8 NONE | C7 also dips at 36–38k (two grids; the reader reports the longest run) |
| R-PF-9 RESTART-PAIRED | EqR on the identical 5k: t1r@128 **98.84** = verified 98.84, spurious 0.00 %, **CONSISTENT** with its 20k 98.85. All seven arms **AHEAD** | C0 +0.92, C1 +0.64, C2 +0.94, C3 +0.68, C4 +0.82, **C5 +1.04 (53/1, p 6e-15)**, C6 +0.90; each p ≤ 6e-5 |
| R-PF-10 C4-FULL | **CONSISTENT** | full 98.93 vs 100k 98.90; vs C0 on the full set +0.46 pp (3,944/2,002, p 2e-142) |

**Scoreboard: 12 of 18 clauses HIT.** All six misses are on the width-192 seed pair's accuracy, and all miss in the same direction, below:
- C7 D16 and C8 D16 fall under [94.9, 96.9]
- C8 D64 falls under [98.6, 99.5]
- SEEDS TIGHT (credence 65 %)
- HEADLINE HOLDS (55 %; SOFTENS carried 35 %)
- WIDTH PAYS (45 %; PARITY carried 40 %)

The hits:
- C7 D64
- SELECTOR-CLEAN, DEPTH, NOT-MEMORIZED
- ≥ 1 EXTENDED, ≥ 1 EXCURSION
- EqR's band, CONSISTENT and spurious clauses
- C5 AHEAD
- C4's band and CONSISTENT

## 4. The rows (EMA, single pass unless a scan; from the summaries)

| row | C0 | C1 | C2 | **C5** | **C7** | **C8** | C4 |
|---|---|---|---|---|---|---|---|
| D16 full (422,786) | 95.09 | 94.34 | 95.63 | 95.91 | 94.82 | 92.98 | 96.70 |
| D16 final grid (50k) | 94.76 | 81.26 | 95.17 | 96.03 | 94.89 | 92.11 | 96.41 |
| D16 raw weights (50k) | 92.73 | 91.13 | 93.45 | 88.91 | 89.74 | 87.21 | 93.92 |
| D64 100k | 98.42 | 97.46 | 98.68 | 99.10 | 98.79 | 97.91 | 98.90 |
| **D64 full** | 98.47 | 97.55 | 98.67 | 99.16 | 98.82 | 97.97 | **98.93** |
| D128 20k / 50k | 98.95 / – | 98.14 / – | 99.00 / – | 99.43 / 99.43 | 99.19 / 99.25 | 98.47 / 98.46 | 99.15 / – |
| D256 5k / 50k | 99.14 / – | 98.42 / – | 99.34 / – | 99.60 / 99.58 | 99.50 / 99.45 | 98.80 / 98.75 | 99.32 / – |
| k32 on 5k: cold / t1r@32 | | | | | 98.82 / 99.80 | 98.00 / 99.74 | |
| selected grid | 16k | 28k | 24k | 46k | 46k | 22k | 20k |

The width-384 triple on the full set: D16 95.02 ± 0.65, D64 98.23 ± 0.56. The width-192 triple: D16 94.57 ± 1.46, D64 98.65 ± 0.59. Census: 0 exploded on C7 and C8, at both the selected and final grids.

## 5. The PI's question: is checkpoint selection seed-dependent? (descriptive; existing rows only)

**Monitor curves** (512 puzzles, EMA, 2k grids; one SE ≈ 1 pp at .95).

Width 384 (C0, C1, C2):
- **Argmax** (earliest tie): 16k / 28k / 24k.
- **Plateau** (within 1 pp of the max): 16–22k / 16–30k / 14–30k. The plateaus overlap at 16–22k.
- **After the plateau:** C1, extended, has a sustained end at 34k. The width-384 treatments with 50k budgets end at 44k (C3, C6).

Width 192 (C5, C7, C8):
- **The rise is slower:** each arm reads under 90 % at 10k, against about 94–95 % for the width-384 seeds.
- **Plateaus:** C5 28–50k, C7 30–48k, C8 22–26k.
- **C5 and C7 both select 46k.**

**(B) Within an arm, the step choice inside the plateau costs little at large n.** Selected grid vs final grid on the identical 50k, paired:

| arm | selected vs final | Δ, pp | p |
|---|---|---|---|
| C0 | 16k vs 30k | +0.30 | 4e-3 |
| C2 | 24k vs 30k | +0.46 | |
| C4 | 20k vs 30k | +0.19 | |
| C5 | 46k vs 50k | −0.18 | .08 |
| C7 | 46k vs 50k | −0.01 | .96 |
| C8 | 22k vs 30k | +0.86 | |

Selection matters only once the curve leaves its plateau. The memorized finals cost 5–13 pp (C1 +12.97, C3 +10.04, C6 +5.48), and that is the error the registered monitor exists to avoid.

**(C) The seed spread lives in the maps, not in the selection.**
- **Width 192, same step:** C5 and C7 were both selected at 46k, yet differ by +1.08 pp at D16 on the full set (15,215/10,638, p 4e-179). At the common 50k grid they differ by +1.14 pp.
- **Width 384:** C0 − C2 is −0.54 pp at their different selected grids (16k vs 24k) and −0.41 pp at the common 30k.

Moving to a common step changes a seed gap by about 0.1 pp.

**What is seed-dependent is the budget, through the extension rule.** The rule reads the monitor's argmax, which is noise-level inside a plateau:
- C2's 24k and 30k grids tie at 96.5, so the earliest tie kept it at 30k.
- C8's argmax fell at 22k, so it never extended.

C5 and C7 extended and kept rising past 30k on the monitor (+0.6 and +1.0 pp to 46k). **So the width-192 triple mixes budgets of 50k, 50k and 30k, and C8's deficit (D16 −1.8 pp vs C7, D64 −0.85 pp) cannot be split into seed and budget with the rows that exist.**
- **Against a pure budget effect:** at the common 28–30k grids C8's monitor reads 91.8 / 93.2 against C7's 95.3 / 95.5 and C5's 97.1 / 96.5, about 1.7–3.9 SE lower on a difference of two 512-puzzle readings. Its 30k grid is also 0.86 pp below its own 22k grid at n 50k.
- **For some budget effect:** the other two seeds gained after 30k.

**Answer.** The plateau is width-set, the argmax inside it is seed noise, and the step inside the plateau costs ≤ 0.5 pp. The seed spread is a property of the trained maps. The seed-dependent part of the protocol is the extension decision.

The common-step test fixed before C8's curve was read (46/48/50k for width 192) **cannot run for C8**: those grids do not exist, because it trained to 30k. The width-384 20k grids have only 512-puzzle screens (C0 95.12, C1 95.51, C2 94.92; spread 0.6 pp, inside noise).

## 6. Post-results critique

- **C8 is a legitimate seed outcome under the registered protocol and is reported as one** (plan §7), labeled with its 30k budget. The one clean way to separate budget from seed is C8 resumed from `ckpt_latest` (30k) to 50k with the constant-lr recipe, exactly what the extension would have done, plus D16 on the 50k and D64 on the full set: about 3 h on a v6e-8, about $25, labeled a counterfactual. No letter would change.
- **Eight-chip vs four-chip data parallelism** (C7/C8 vs C5) cannot be separated from the seed at n = 1 per configuration. C7 on 8 chips sits between C5 and C8, so no one-directional sign is visible. Labeled, not a variable.
- **The width-192 cell's seed variance at D16 is 2.3× the width-384 triple's** (2.92 vs 1.29 pp). Part of it is C8's budget. At D64 the two widths' spreads are equal (1.18 vs 1.12 pp): depth absorbs most of the seed variance, as it absorbs the width gap.
- **The excursion is a width-192 regularity at 2 of 3 seeds** (C5 16–28k, C7 22–26k and 36–38k) and absent at width 384. It recovered on both arms and left no trace in the selected grids' selector or depth rows.
- **The selector law now holds on all nine champion grids** (SELECTOR-CLEAN 7/7 on 2026-09-09 plus C7 and C8 here, at spurious 0.20 / 0.23 % and t1r within 0.08 pp of verified at k 32). Depth never unsolves a solved puzzle on either new arm (0 of 422,786 lost from D16 to D64).
- **EqR on the identical 5k reproduces its 20k reading** (98.84 vs 98.85). Residual selection equals its verified coverage, with spurious 0.00 %. Every champion arm leads it by 0.6–1.0 pp on the same puzzles.

## 7. What changes in the paper (the reporting rules fixed at registration; the writing session applies them)

- **Table 1's width-192 row becomes three seeds:**
  - D16 full 94.6 ± 1.5
  - D64 full **98.6 ± 0.6**
  - D128 50k 99.0 ± 0.5
  - D256 50k 99.3 ± 0.4

  Footnote: one seed at a 30k budget (the extension rule did not fire), and the triple trained on 4 vs 8 chips.
- **The abstract's 0.8M sentence uses 98.6** in place of C5's single-seed 99.2. The letter is SOFTENS, so the verb drops "above 99" language.
- **§4's compute sentence (PARITY):** the width-192 cell "matches its seed-matched width-384 twin" (mean +0.4 pp at D64 on the full set, 2 of 3 seeds ahead).
- **The restart column:** EqR 98.84 on the identical 5k with the paired letters (all arms AHEAD). The 20k row moves to Appendix D.
- **C4's dagger goes:** 98.93 on the full set.
- **Seed counts:** the memorization clock, not memorized by 50k, now carries 3/3 width-192 seeds; the excursion paragraph carries 2/3 width-192 seeds and 0/3 width-384.

## 8. Lessons

- **A selftest fixture must be built from a real artifact's schema** (copy a real `config.json`), never a hand-written guess. The argv gate passed 25/25 on a list the trainer never writes. Next registration: the analyzer's no-data run on a real prior config asserts each INTEGRITY clause actually executed.
- **The crc check earned its keep twice in one pull.** Skip-if-present without a hash comparison can bank a wrong file under the right name.
- **A budget rule keyed on a noisy argmax makes the budget seed-dependent.** Next time, read the extension on the plateau (e.g., max over the last 4k within 1 SE of the running max), or fix the budget per width.
