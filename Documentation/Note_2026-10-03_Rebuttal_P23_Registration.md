# Note 2026-10-03 — P23: does the memoryless chaotic tail transfer to another token mixer? P22's protocol on MLP 192, three seeds (registration, before any row)

**Goal (page one).** A MEASUREMENT. No accuracy target, no cost, and nothing trained.
- Inference only, on the Mac's CPU, with the manuscript's banked MLP 192 checkpoints (seeds 0, 1, 2).
- Tool: `tools/rebuttal_p23.py`. It imports P22's statistics, letters and reader helpers and P21's MLP loader unchanged. The run loop is P22's, with the MLP pool and its own seed.
- Outputs: `runs/analysis/rebuttal_20261003f/`.
- Tests prediction 4 of plan §7 (transfer) for the law itself.

## Why

P22 confirmed on all three attention widths that stalled correction is **memoryless** and **chaotic**:
- R = 1.08, 0.91 and 1.08;
- D(1°)/D(60°) = 0.93, 0.97 and 0.82.

P21 showed, exploratorily, the same pattern on MLP 192: survivors of 32 completed at the fresh re-roll rate (0.257 vs 0.256). The question now is whether the law transfers to a different token mixer and three more independent initializations.

## Design

**Identical to P22's**, except for the following.
- **Receivers:** MLP 192 seeds 0, 1 and 2, each loaded with P21's MLP loader, whose checkpoint hash is gated against the release manifest.
- **Population:** per seed, a fresh seeded uniform draw of 160 test indices from that model's own record's unsolved-at-16 pool, excluding P21's pool for that seed (seed `[20261005, 23, seed]`).
- **Rotation seeds:** `[20261005, puzzle ID, 101 / 102 / 103, k]`.

As in P22:
- Trigger: an invalid display at iteration 16 on the CPU, asserted equivalent to CPU-unsolved.
- Arms: A0 intact (32 iterations); eight 60° re-rolls (32 iterations); four 1° and four 0.1° re-rolls (16 iterations).
- Gates: rotation score shift ≤ 1e-2; realized angle within 1 % of nominal; trigger equivalence. A gates-only smoke on 16 puzzles of seed 0 is run before launch.

## Rules

P22's rules, unchanged, with each seed as a receiver. Each letter is confirmed when it holds on at least two of the three seeds.

- **A — memory.** From R = Σ d_L ÷ Σ (K − d_E) d_E / (K − 1), with a puzzle bootstrap 95 % interval:
  - AGING if the interval's upper end is below 1;
  - ACCUMULATING if its lower end is above 1;
  - otherwise MEMORYLESS if the interval lies within [0.67, 1.5];
  - otherwise UNRESOLVED.
- **B — chaos.** From D(1°)/D(60°): CHAOTIC if ≥ 0.8; REGULAR if ≤ 0.5; INTERMEDIATE otherwise.
- **Exploratory:** P22's list, the same as there.

## Predictions and credences (before any row)

| rule | letter | credence |
|---|---|---|
| A | MEMORYLESS | 0.65 |
| A | AGING | 0.10 |
| A | ACCUMULATING | 0.10 |
| A | UNRESOLVED | 0.15 |
| B | CHAOTIC | 0.65 |
| B | INTERMEDIATE | 0.25 |
| B | REGULAR | 0.10 |

## Wording under each outcome

- **MEMORYLESS / CHAOTIC:** P22's sentences, with "in a recursive reasoner with a different token mixer, across three independent initializations" added.
- **Any other letter:** "The memoryless chaotic tail found on the attention models does not transfer to the MLP mixer as a [letter] pattern", together with the measured R and ratio.
- Under every outcome: P22's qualifications.

## Plan

1. Commit this note and the tool before any row; push.
2. Run the gates-only smoke.
3. Launch the three seeds in parallel at nice 5. The other session has said the Mac's CPU is free until about 15:00Z.
4. Produce the report with the tool.
5. When it is read, append the Outcome here and write a ledger line.
