# Note 2026-10-03 — P21: does the answer-preserving hidden-state rotation transfer to another token mixer and to independent initializations? (registration, before any row and before any P20 result)

**Goal (page one).** A MEASUREMENT. There is no accuracy target, no cost, and nothing is trained: inference only, on the Mac's CPU, using the manuscript's banked MLP 192 checkpoints (seeds 0, 1, 2).
- Tool: `tools/rebuttal_p21.py`. It imports P17's runner unchanged. In its own process only, it replaces the base loader with one for the named MLP checkpoint. Outputs go to `runs/analysis/rebuttal_20261003d/`.
- This note is registered before any P21 row and before any P20 result was seen, so the design cannot depend on P20's outcome.
- It tests prediction 4 of plan §7 (transfer).

## Why

P17's exploratory arms found a 60° score- and norm-preserving rotation of the readout-invisible slow state to be the strongest arm on all three attention widths.
- Solved by 32: 90 / 75 / 72, against 71 / 60 / 47 for intact continuation (exact McNemar p 0.015 / 0.049 / 6e-4).
- Attention 256 had not been seen when P20 was registered.
- P20 tests the rotation on fresh attention puzzles.

The open question is whether the effect belongs to the attention models or to this class of recursive reasoner. MLP 192 has the same two-state recurrence, readout and training recipe with a different token mixer (an MLP over cells with mean coupling). It was trained from three independent initializations. The test therefore spans both an architecture change and initialization.

**Loader check (prototype, before this note).** The MLP loader reproduces the release manifest's checkpoint hash. On 128 puzzles of seed 0, the CPU fixed-start loop agrees with the saved benchmark record about as well as the validated attention loader does:
- puzzles the record solved by iteration 8 were solved at 16 on the CPU in 63 / 64 (Attention 128 with the reference loader: 61 / 64);
- 24 / 64 record-unsolved puzzles stayed unsolved (Attention 128: 27 / 64).

The record (accelerator) and the CPU differ per puzzle in the same way for both loaders. That is why the trigger is computed on the CPU run, as in P17 and P20.

## Design

**Population.** Per seed:
- a fresh seeded uniform draw of 384 test indices from that model's saved full-test record's unsolved-at-16 pool (seed `[20261003, 21, seed]`);
- sixteen fixed-start iterations through the release step on the CPU.

Trigger (no answer key): the displayed grid at iteration 16 is invalid, i.e. it has a duplicate in a row, column or box, or an altered given. This is asserted equivalent to CPU-unsolved.

Strata: intact continuation to iteration 64 labels each triggered puzzle *transient* (solved by 64) or *trapped*.

**Arms.** Every arm starts from the iteration-16 state.
- **A0** intact, continued to iteration 64.
- **R60, pre-specified.** P20's operation and angle: the readout-invisible slow part at every cell is rotated by 60° toward a random direction orthogonal to the readout and to itself, keeping scores and norms (seed `[20261003, puzzle ID, 91, 60]`). Continued to iteration 64.
- **R45 and R75**: the same operation at 45° and 75°. Exploratory, to map the operating range. Continued to 32.
- **A5**: a fresh Gaussian restart run for 16 iterations (seed `[20261003, puzzle ID, 92]`).

**Gates.**
- The checkpoint's hash equals the release manifest's.
- The mixer and the selected step are as listed in the model inventory.
- Every rotation changes no digit score by more than 1e-2.
- The trigger's equivalence with CPU-unsolved is asserted.
- A smoke on 16 puzzles of seed 0 is run before launch.

## Rules

All rules count puzzles solved by iteration 32 among triggered puzzles, using exact McNemar tests. They are P20's rules, with each seed as a receiver.

- **ROTATION-HELPS** per seed if R60 solves more than A0 with p < 0.05. ROTATION-HURTS if it solves fewer with p < 0.05; NO-DIFFERENCE otherwise. **TRANSFERS** if ROTATION-HELPS holds on at least two of the three seeds.
- **BEATS-RESTART** per seed if R60 solves more than A5 with p < 0.05; confirmed on at least two of three seeds.
- **TRAPPED-RESCUE** if R60 solves at least 10 % of the trapped puzzles on at least two of three seeds. A0 by definition solves none of them by 64.
- Reported beside the rules:
  - every arm by stratum;
  - losses per arm against A0;
  - the R45 / R60 / R75 operating range;
  - the long horizon (exploratory): R60 against A0 solved by 48 and by 64, and the iteration at which A0 first reaches R60's count at 32 (iterations of waiting saved).

**Predictions and credences (before any row).**

| prediction | credence |
|---|---|
| TRANSFERS | 0.45 |
| BEATS-RESTART confirmed | 0.40 |
| TRAPPED-RESCUE | 0.55 |
| R60 still ahead of A0 at iteration 64 on at least two seeds (rescue, not only speed-up) | 0.45 |

**Wording under each outcome.**
- TRANSFERS: "The same answer-preserving rotation helps stalled trajectories in a recursive reasoner with a different token mixer, across three independent initializations."
- NO-DIFFERENCE on at least two seeds: "The rotation's benefit does not transfer to the MLP mixer."
- Under every outcome:
  - the rotation's direction is random;
  - the trigger uses no answer key;
  - the angle was chosen on the attention models (P17's exploratory arms), as disclosed here;
  - the result holds for the tested checkpoints and population.

## Plan

1. Commit this note and the tool before any row. The commit stays local: pushing waits for the other session's local-only commit to clear.
2. Queue the run behind P20's queue, waiting on its PID.
3. Run the three seeds in parallel at nice 5.
4. Produce the report with the tool.
5. When it is read, append the Outcome here and write a ledger line.
