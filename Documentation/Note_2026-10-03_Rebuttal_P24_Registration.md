# Note 2026-10-03 — P24: is the single-state model's stall equally memoryless and chaotic? P22's protocol on SA256U and SA256 (registration, before any row)

**Goal (page one).** A MEASUREMENT, with no accuracy target, no cost and nothing trained.
- Inference only, on the Mac's CPU.
- Receivers: SE-RRM attribution round 2's matched pair, offered by session d378a5 with its PI's go.
  - **SA256U** at its selected 14k: the single recurrent state, a one-carry loop.
  - **SA256** at its selected 28k: the two-state design.
  - **SA256** at 14k: matched to SA256U's training stage.
- Tool: `tools/rebuttal_p24.py`.
  - It runs on the MAIN repo's `src/qhrrn2` and `tools/eval_sudoku_extreme.py`, using the evaluator's own loader recipe. The release code has no single-state branch.
  - It imports P22's statistics, letters and reader helpers unchanged.
  - Outputs: `runs/analysis/rebuttal_20261003g/`.
- Checkpoints: `runs/_p13_ckpts/` (read-only; pulled from the GCS live runs by d378a5).

## Why

P22 confirmed that the two-state attention models' stalled correction is **memoryless** and **chaotic**. Round 2 found that the single state commits faster at the first iterations and then repairs little. A natural contrast follows. In the two-state design, the stall is a live search that keeps re-rolling. In the one-carry loop it might instead freeze or age:
- trajectories that stop moving;
- R below 1;
- small rotations that leave outcomes unchanged.

Either answer bears on what the slow/fast split buys at test time.

## Design

P22's design, with these differences:
- **Code:** the main repo, with a loop that mirrors `run_batch` (carry z ← z + η_z(z_f − z); canvas y ← y + η(p − y); η = η_z = 1). The display is the digit argmax, as in P17–P23.
- **Population:**
  - SA256U: a seeded draw of 160 from its own 5,000-puzzle record's unsolved-at-16 puzzles (seed `[20261006, 24, 0]`).
  - SA256, both checkpoints: one shared seeded draw of 160 from SA256's full-test record at 46k (the manuscript's attention-256 run), taking puzzles whose first exact iteration is never or after 16. P17's, P20's and P22's attention-256 pools are excluded (seed `[20261006, 24, 1]`). Puzzles are thus paired across the two SA256 checkpoints.
  - On every receiver: the CPU trigger (an invalid display at iteration 16), keeping the first 80 triggered puzzles in sorted-id order to bound cost.
- **Arms:** as P22, with one addition.
  - A0: intact, 32 iterations. The number of display cells changing at each iteration is also recorded.
  - Eight 60° re-rolls, 32 iterations each (seed `[20261006, ID, 101, k]`).
  - Four 1° re-rolls and four 0.1° re-rolls, 16 iterations each (salts 102 and 103).
- **Gates:**
  - Each checkpoint's sha256 has the prefix d378a5 reported.
  - The configuration matches its receiver: a DEC cell, width 256, η = η_z = 1, and single-state only for SA256U.
  - **G1:** on each receiver's first batch, the loop's per-iteration exact flags equal the main evaluator's `run_batch` flags bitwise over 16 iterations from the fixed start.
  - The trigger's equivalence with CPU-unsolved.
  - Every rotation changes no score by more than 1e-2.
  - Every realized angle is within 1 % of its nominal angle.
  - A gates-only smoke on the first 16 drawn puzzles of SA256U (11:13Z, after the note and tool were committed and pushed, before launch) passed every gate: the sha256 prefix, single-state at step 14000, G1 (`run_batch` bitwise), the trigger's equivalence (15 triggered, all kept), the largest rotation score shift 5.2e-7, the largest realized-angle deviation 0.0004 %, and every arm with its designed length. It printed no outcome counts. Those 16 puzzles are part of the registered population.

## Rules (per receiver; P22's, unchanged)

**A — memory.** R with a puzzle-bootstrap interval:
- AGING if the interval's upper end is below 1;
- ACCUMULATING if its lower end is above 1;
- otherwise MEMORYLESS if the interval lies within [0.67, 1.5];
- otherwise UNRESOLVED.

**B — chaos.** D(1°)/D(60°):
- CHAOTIC if ≥ 0.8;
- REGULAR if ≤ 0.5;
- INTERMEDIATE otherwise.

**The registered pattern across receivers.**

| pattern | condition |
|---|---|
| SAME-TAIL | SA256U and SA256 at 28k are both MEMORYLESS and CHAOTIC |
| SINGLE-STATE-DIFFERS | SA256U's pair of letters differs from both SA256 checkpoints' |
| MIXED | otherwise |

**Exploratory.** P22's list, plus the stalled intact trajectories' display changes: the fraction of frozen steps and the median number of cells changed.

## Predictions and credences (before any row)

| receiver | rule A | rule B |
|---|---|---|
| SA256 at 28k and at 14k | MEMORYLESS 0.65 | CHAOTIC 0.65 |
| SA256U | MEMORYLESS 0.35, AGING 0.35, UNRESOLVED 0.20, ACCUMULATING 0.10 | CHAOTIC 0.40, INTERMEDIATE 0.30, REGULAR 0.30 |

| pattern | credence |
|---|---|
| SAME-TAIL | 0.35 |
| SINGLE-STATE-DIFFERS | 0.40 |
| MIXED | 0.25 |

## Wording under each outcome

- **SAME-TAIL:** "The single-state recurrence's stall is also memoryless and chaotic: the tail law does not depend on the two-state design."
- **SINGLE-STATE-DIFFERS:** "The single-state recurrence's stall differs from the two-state design's: [its letters, with R and the chaos ratio]."
- **MIXED:** each receiver's letters, as measured.
- Under every outcome:
  - re-roll directions are random;
  - the trigger uses no answer key;
  - SA256U and SA256 differ in structure and in training length, and SA256 at 14k is the stage-matched control;
  - the result holds for these checkpoints and populations, at P22's horizon.

## Plan

1. Commit and push this note and the tool before any row.
2. Run the gates-only smoke.
3. Queue the experiment behind session d378a5's P13, waiting on its queue's PID; P13 is expected to finish around 13:45Z.
4. Run the three receivers in parallel at nice 5.
5. Produce the report with the tool.
6. When it is read, append the Outcome here and write a ledger line.

---

## P24 Outcome (2026-10-03 14:45Z; `runs/analysis/rebuttal_20261003g/report.{txt,json}`; every gate passed; no shared tool edited during the runs)

**Integrity.**
- Every checkpoint carried its reported sha256 prefix and the expected configuration (single-state only for SA256U).
- **G1 passed on all three receivers.** The main-repo loop reproduced `run_batch`'s exact flags bitwise.
- On every receiver the trigger was asserted equivalent to CPU-unsolved. Triggered: 131 / 56 / 67 of 160, kept 80 / 56 / 67 (SA256U 14k / SA256 28k / SA256 14k).
- Every rotation changed no score by more than 1e-2 (maximum 7.8e-7), and every realized angle was within 0.002 % of nominal.
- No solved grid un-solved.
- An independent recount agrees with the report on R, both ratios, the event counts, every letter, the frozen-step measure and the pattern.

**Results.**

| receiver | kept | re-roll completion within 16 / 32 | completions early / late | R [95 %] | A | D(60°) | D(1°) | D(0.1°) | D(1°)/D(60°) [95 %] | B |
|---|---|---|---|---|---|---|---|---|---|---|
| SA256U 14k (one state) | 80 | 0.116 / 0.181 | 74 / 42 | 1.00 [0.68, 1.40] | MEMORYLESS | 0.148 | 0.119 | 0.098 | 0.80 [0.50, 1.25] | CHAOTIC |
| SA256 28k | 56 | 0.328 / 0.487 | 147 / 71 | 1.07 [0.82, 1.37] | MEMORYLESS | 0.301 | 0.262 | 0.295 | 0.87 [0.63, 1.20] | CHAOTIC |
| SA256 14k | 67 | 0.297 / 0.422 | 159 / 67 | 0.86 [0.66, 1.08] | UNRESOLVED | 0.274 | 0.256 | 0.224 | 0.94 [0.68, 1.26] | CHAOTIC |

**Pattern.** **SAME-TAIL**: SA256U and SA256 at 28k are both MEMORYLESS and CHAOTIC.

**Borderline disclosure.** Two memory letters sit on the equivalence margin, in opposite directions:
- SA256U's registered interval lower end is 0.680; the recount's separately seeded bootstrap gives 0.659, just outside 0.67.
- SA256 at 14k is 0.661 against 0.677 in the recount.

SA256U's chaos ratio, 0.80, is also at its threshold, with a wide interval. The letters stand by the registered bootstrap. The width reflects how rarely SA256U's stalled trajectories complete (116 completions in 640 re-rolls), not a departure from memorylessness: its point estimate is R = 1.000.

**Predictions scored.**
- SA256: MEMORYLESS (0.65) HIT at 28k, MISS at 14k (UNRESOLVED); CHAOTIC (0.65) HIT on both.
- SA256U: MEMORYLESS (0.35) HIT; CHAOTIC (0.40) HIT.
- Pattern: SAME-TAIL (0.35) HIT; SINGLE-STATE-DIFFERS (0.40) MISS.

**The registered sentence that applies.** SAME-TAIL: "The single-state recurrence's stall is also memoryless and chaotic: the tail law does not depend on the two-state design." Measured qualification: the single state's letters are borderline, for lack of completions.

**Exploratory.**
1. *The single state's stall is a live search, not a frozen one.* On stalled intact steps, no display is frozen (0.000 / 0.000 / 0.001), and a median of 13 / 12 / 11 cells change per iteration.
2. *What the single state lacks is completion rate, not search.* A stalled re-roll completes within 16 iterations 0.116 of the time against 0.328 (SA256 28k) and 0.297 (SA256 14k), and within 32 iterations 0.181 against 0.487 and 0.422. 44 of SA256U's 80 kept puzzles had none of eight re-rolls solved by 32, against 9 of 56 and 16 of 67. This fits round 2's finding that the single state repairs little: it keeps searching at the same character but finds far less often.
3. *Intact continuation behaves like one more re-roll on every receiver.*
   - Completed within 16 iterations: 11 / 21 / 20, against 9.2 / 18.4 / 19.9 expected.
   - Within 32: 15 / 29 / 27, against memoryless predictions of 13.8 / 25.6 / 28.4.

**Reading.**
- The memoryless, chaotic character of stalled correction does not come from the slow/fast split: the one-carry recurrence has it too, at its point estimate.
- What the two-state design buys in the stalled regime is a higher per-instance completion rate, about three times the single state's per sixteen iterations. The search is not a different kind.
- Together with P22 and P23, the memoryless letter holds on eight of nine receivers, across two token mixers and both recurrence structures. The ninth, SA256 at 14k, is UNRESOLVED at the margin. Small-angle chaos is strong on attention and somewhat weaker on the MLP mixer.
