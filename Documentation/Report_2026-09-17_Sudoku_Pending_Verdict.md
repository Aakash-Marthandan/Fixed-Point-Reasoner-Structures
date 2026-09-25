# The pending Sudoku runs for paper 1 — verdict (2026-09-17)

> **Status (2026-09-25):** historical record, preserved verbatim; it predates the 2026-09-19 adversarial audit and the paper's frozen account (v4.3). "DEC" below names the digit-field cell; the paper calls these runs MLP 192 and the TRM-cell control (X5). The current statement of the results is the root `README.md` §2–§3; the correction record is `Design_Ledger.md` §5 (the 2026-09-19 audit entry and the 2026-09-24 documentation entry).

**Registration:** `Plan_2026-09-17_Sudoku_Pending_Runs.md` (§3 rules and §4 credences locked at 8990ccf). **Analyzer:** `tools/analyze_sudokupend.py`, frozen at 208345d. **Data:** `runs/_sudokupend_pull/` (23 objects, CRC32C 23/23). **Spend:** ≈ $44.60. Every number below is printed by a script; the outputs are in `runs/_sudokupend_pull/analysis/`.

## 0. The short version

| Question | Answer |
|---|---|
| Does the 99.8 headline survive as a triple? | **Yes.** At k = 128 on the identical 5,000 test puzzles every width-192 seed is ahead of EqR's released weights: selected 99.88 / 99.88 / 99.80 (mean 99.85, half-spread 0.04), verified 99.88 / 99.92 / 99.86 (mean 99.89, half-spread 0.03), EqR 98.84. Discordant pairs 53:1, 55:3, 53:5; p ≤ 3.5e-11 on each seed. |
| Does TRM's cell reach the DEC at matched parameters under the full recipe? | **Not within the matched 50k-step budget:** 45.04 at 16 iterations (422,786 puzzles) and 47.47 at 64 (100k), against the triple's 95.41 and 99.01. **But the row is a lower bound, not TRM's cell's level at this size** — its monitor is still at its maximum on the last grid, it used about one nineteenth of the DEC's training compute, and it is one seed (§4). |
| What should the paper do? | Take the k128 triple now. Hold the X5 row as "at matched parameters and matched steps" with its caveats, or run it to its plateau first (§6) — the PI's call. |

## 1. Integrity and provenance

- The pull: 23/23 CRC32C. The final archive (`sudokupend_final.tgz`, summaries only) agrees with the per-row banks on every file they share (0 differing files).
- The staging root `runs/_sudokupend_pull/stage/runs` is symlinks only. The triple's rows are the budget-matched ones the paper uses, all on `ckpt_046000`: C5 from `runs/`, C7 from the paper-final pull, C8 from the extension pull (never its pre-extension 22k grid). D16 n = 422,786 and D64 n = 100,000 on all three. EqR's k128 row is the paper-final pull's.
- X5 verified at the source: `cell trm`, hidden 160, 795,909 + 32 parameters (×1.009 of the DEC's 789,122), digit augmentation on, anchor k 1 / ε 0.2 / frac 0.25, randomized-init σ 1.0, batch 768, lr 1e-4, wd 1.0, EMA .999, 50,000 steps, seed 0. One grid (`ckpt_050000`) across its D16, D64, k32 and k128 rows.
- The analyzer was byte-identical to 208345d when first run; its selftest passed 16/16.

**The reader-format addendum.** The frozen run printed `INTEGRITY FAIL: X5 argv lacks [...]` — a false alarm disclosed before the data (ledger, 2026-09-17): the trainer records its arguments as a dict, the frozen check expected a token list. The fix is one helper, `argv_missing`: the dict form is read flag by flag against its recorded value, which is stricter than the registered token membership; the list form is read exactly as before. No rule, threshold or registry entry changed. Five new selftest cases in the trainer's real format (21/21). **The two runs differ in exactly one output line** (INTEGRITY: FAIL → PASS); both are kept (`frozen_run.txt`, `addendum_run.txt`).

## 2. The registered letters (verbatim, the addendum run)

```
INTEGRITY                PASS
R-SP-1 PARAMS            MATCHED (795,941 vs 789,122, x1.009; config.json (bulk + the task table))
R-SP-2 PARITY-16         BELOW (X 45.04 vs the triple's mean 95.41, d -50.36 pp, floor 0.54 read)
R-SP-2 PARITY-64         BELOW (X 47.47 vs the triple's mean 99.01, d -51.55 pp, floor 0.18 read)
R-SP-3 SELECTOR          CLEAN (spurious k32 0.01 %; k128 0.01 %)
R-SP-4 K128-TRIPLE       C5 AHEAD (selected 99.88 vs EqR 98.84, p 6.1e-15; verified 99.88) | C7 AHEAD (selected 99.88 vs EqR 98.84, p 2.3e-13; verified 99.92) | C8 AHEAD (selected 99.80 vs EqR 98.84, p 3.5e-11; verified 99.86) | TRIPLE-AHEAD (selected 99.80–99.88, verified 99.86–99.92)
R-SP-5 ONSET             NOT-BY-END
EXCURSION (descriptive)  NONE
X k128 (descriptive)     selected 58.18, verified 58.32
```

The D64 mean reads 99.01 because R-SP-2 compares like for like on the 100k rows; the paper's 99.05 ± 0.18 is the full-set row (99.16 / 98.82 / 99.18), unchanged.

## 3. Prediction scoreboard (credences locked before the data)

| Prediction | Credence | Outcome |
|---|---|---|
| P-SP-1 MATCHED | 99 % | hit |
| P-SP-2 BELOW at 16 | 70 % | hit |
| P-SP-2 BELOW at 64 | 65 % | hit |
| P-SP-3 CLEAN | 70 % | hit |
| P-SP-4 TRIPLE-AHEAD | 85 % | hit |
| selected ≥ 99.7 on every seed | 80 % | hit (99.80 the lowest) |
| verified ≥ 99.8 on every seed | 75 % | hit (99.86 the lowest) |
| P-SP-5 NOT-BY-END | 55 % | hit |

Eight of eight. The direction was called; **the size was not**. The registration gave PARITY a 25 % credence at 16 iterations, so a gap of 50 points was outside what it had in mind. It also stated the run's goal but no expected accuracy for X5 — a shortfall against the goal-disclosure rule, recorded here. Had the number been written down, "why would a 0.8 M TRM cell finish inside 50k steps?" would have been asked before the launch instead of after.

## 4. Post-results critique

**C1. X5's budget binds; the DEC's does not.** X5's monitor maximum (46.68) sits only on the last grid. The curve ignites late — flat near 17–25 from 8k to 32k, then 25 → 47 between 34k and 50k — and gains +4.7 points over the last 10k steps. Under the champion chain's extension rule this arm would have been extended; the registration fixed its budget to match the triple's steps. The triple first reaches its maximum inside the budget, and the width-192 long run found no better point out to 150k. So 45.04 is **a lower bound at matched steps**. Where TRM's cell plateaus at this size is unmeasured.

**C2. Matched parameters are not matched compute.** On the same 8-chip pod type X5 trains at a median 272.7 steps/s against the DEC's 14.2 (C7 and C8; C5's 5.3 is a 4-chip pace and is not like for like) — X5 works on 97 tokens, the DEC on 729. Its whole 50k pretrain took about ten minutes of pod wall (the banked grids' timestamps); a DEC seed takes about an hour at its recorded pace. At matched training compute TRM's cell would have roughly nineteen times the steps, ≈ 1 M. A similar gap holds at inference. *(Corrected 2026-09-17: the first version of this paragraph compared X5 with C5's 4-chip pace and said fifty times.)* A reviewer will raise this, and the row as it stands has no answer to it.

**C3. One seed, read mid-ignition.** A curve that is climbing ten points per 8k steps makes the value at a fixed step sensitive to when ignition happens, which may move by thousands of steps between seeds. The BELOW letter is safe from this (−50 points against a 0.54 floor). The number 45.04 is not a stable property of the architecture.

**C4. What the gap is made of.** X5 differs from TRM as published in three ways: the anchor row, randomized init, and width 512 → 160. At 5 M parameters on this same loop each of the first two costs TRM's cell under two points (Night A: 87.79 plain, 86.46 with the anchor, 85.65 with randomized init, all selected at 40–48k). So within this budget the narrowing is where the 40 points go — confounded with C1, since the 5 M cell finishes inside 50k steps and the 0.8 M cell does not.

**C5. Which letters could not have failed.** R-SP-1 (the count was known from the CPU smoke) and EXCURSION (it needs an EMA of 90 %, which X5 never approaches — its NONE is vacuous). R-SP-4's margins are far above the count floor (53:1, 55:3, 53:5). The selected-minus-verified gaps (0, 2 and 3 puzzles) are below it and carry no evidence.

**C6. A registration slip, harmless.** The plan's S2 line quoted C5's existing k128 row as "selected 99.16, verified 99.20". Those are its cold-start-scale numbers; the row reads 99.88 / 99.88, as the paper has it. No rule or credence depended on the misquote.

**C7. Protocol label for R-SP-4.** EqR's row is its released weights under our evaluator at 64 iterations and k = 128 restarts with residual selection, on the same 5,000 puzzles. The paper already labels C5's comparison this way; the triple inherits the label.

## 5. Descriptive read (EXPLORATORY — `tools/lens_sudokupend_read.py`, selftest 12/12)

**The k128 triple.** Unsolved after selection: 6, 6 and 10 of 5,000. The union is 15 puzzles; only 2 are unsolved by all three seeds, so some seed solves 4,998 of 5,000. EqR leaves 58 unsolved; 8 of our 15 are among them. Restarts lift the DEC from 99.09 cold to 99.85 selected and EqR from 93.50 to 98.84. The DEC's wrong draws look converged 0.2–0.4 % of the time (EqR 0.000 %), well inside the 1 % CLEAN line; the selector loses 0, 2 and 3 puzzles to it.

**X5's failures are a coverage problem, not a selection problem.** Restarts lift it only from 47.78 to 58.18, and 2,084 of 5,000 puzzles have no exact draw among 128. Its selector is as clean as EqR's (0.012 %) — randomized init did its job; the basins are missing. Depth buys little: 45.04 → 47.47 → 47.81 → 48.96 at 16 / 64 / 128 / 256 iterations.

**The training side.** X5's training loss flattens at 0.733 from 20k on and its training-exact reaches 3.9 %; the DEC seed reads 0.561 and 25.1 % on the same loop and regime. The cell is under-fit at this step count, consistent with C1.

## 6. What this means for the paper, and the run that would close it

**Ready now.** Table 1's selected / verified column becomes a triple at k = 128: selected 99.85 ± 0.04, verified 99.89 ± 0.03, each seed ahead of EqR (98.84) at p ≤ 3.5e-11 on identical puzzles. The best-seed objection to the 99.8 headline is closed.

**Not ready as a capacity claim.** The X5 row supports exactly this sentence: *at matched parameters, recipe and training steps, TRM's cell reaches 45.0 / 47.5 where the symmetric DEC reaches 95.4 / 99.0; it was still improving when its budget ended.* It does not support "TRM's cell cannot use 0.8 M parameters", and it does not yet credit the symmetric state over narrowness against the compute objection.

**The run that would make it defensible — X5 to its plateau (the PI's decision; nothing is launched).** Resume the same arm and seed from its banked 50k state (constant learning rate, so a resume is clean) under a registered stopping rule: stop when the EMA monitor has not improved for a fixed window, capped at the compute-matched step count, ≈ 1 M steps (19 × 50k). At the measured wall pace (≈ 90 steps/s with the 2k-step monitors and 500-step checkpoint writes; faster on a sparser cadence) that is at most ≈ 3 hours of one spot v6e-8, ≈ $25. The battery adds ≈ $10–25. If the plateau lands far below the triple, the row answers both the budget and the compute objection in one line. If it lands near the triple, the paper needs to know that before a reviewer finds it. Seeds 1 and 2 matter only in the second case.
