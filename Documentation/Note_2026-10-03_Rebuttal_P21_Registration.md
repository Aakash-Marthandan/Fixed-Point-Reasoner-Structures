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
- A smoke on the first 16 drawn puzzles of seed 0 (03:49Z, after the note and tool were committed, before launch) passed every gate: the checkpoint hash equal to the manifest's, the trigger's equivalence (7 triggered), every arm with its designed length, and the largest rotation score shift 6.3e-7. It printed no outcome counts. Those 16 puzzles are part of the registered population.

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

---

## P21 Outcome (2026-10-03 07:55Z; `runs/analysis/rebuttal_20261003d/report.{txt,json}`; every gate passed; no shared tool edited during the runs)

**Integrity.**
- Every checkpoint's hash equalled the release manifest's (rechecked from each output's metadata by the recount).
- On every seed the trigger was asserted equivalent to CPU-unsolved. Triggered: 163 / 191 / 168 of 384 (seeds 0 / 1 / 2).
- Every rotation changed no digit score by more than 1e-2; the realized maximum was 7.9e-7 / 8.0e-7 / 7.5e-7.
- No solved grid un-solved in any arm.
- An independent recount from the raw arrays agrees with the report on every arm, stratum, loss count, letter, the long horizon and the aggregate.

**Results (solved by iteration 32 among triggered puzzles).**

| MLP 192 | triggered (transient / trapped) | A0 intact | R45 | R60 | R75 | A5 restart | R60 vs intact | R60 vs restart | trapped solved by R60 |
|---|---|---|---|---|---|---|---|---|---|
| seed 0 | 163 (90 / 73) | 59 | 51 | 58 | 63 | 67 | 21/22, p 1.0 | 25/34, p 0.30 | 11 (0.15) |
| seed 1 | 191 (128 / 63) | 72 | 79 | 78 | 76 | 67 | 38/32, p 0.55 | 39/28, p 0.22 | 8 (0.13) |
| seed 2 | 168 (119 / 49) | 76 | 67 | 70 | 69 | 63 | 23/29, p 0.49 | 33/26, p 0.44 | 9 (0.18) |

Losses against A0, seeds 0 / 1 / 2:
- R45: 27 / 24 / 30
- R60: 22 / 32 / 29
- R75: 21 / 31 / 33
- restart: 22 / 34 / 38

Trapped puzzles solved by the restart: 16 / 9 / 8.

**Letters.**
- **NO-DIFFERENCE** on every seed, so TRANSFERS does not hold.
- BEATS-RESTART: not shown on any seed.
- **TRAPPED-RESCUE holds** (0.15 / 0.13 / 0.18).

**Predictions scored.**
- TRANSFERS (0.45): MISS.
- BEATS-RESTART (0.40): MISS.
- TRAPPED-RESCUE (0.55): HIT.
- R60 ahead of A0 at 64 on at least two seeds (0.45): MISS (one of three).

**The registered sentence that applies, with its qualification.** NO-DIFFERENCE: "The rotation's benefit does not transfer to the MLP mixer." P20, read after this registration, found no benefit on the attention models either, so there was no benefit to transfer. What transfers is the behaviour P20 found: across a different token mixer and three independent initializations, the rotation re-rolls stalled trajectories without improving them.

**Exploratory.**
1. *The long horizon.* The rotated trajectory tracks intact continuation through iteration 64.

   | | seed 0 | seed 1 | seed 2 |
   |---|---|---|---|
   | solved by 48, rotated / intact | 85 / 78 | 107 / 108 | 94 / 102 |
   | solved by 64, rotated / intact | 97 / 90 | 116 / 128 | 114 / 119 |
   | McNemar at 64 (rotated-only / intact-only) | 20/13, p 0.30 | 15/27, p 0.09 | 18/23, p 0.53 |

   The two arms' per-16-iteration completion rates follow the same falling curve (seed 0: 0.36 / 0.26 / 0.15 rotated against 0.36 / 0.18 / 0.14 intact).
2. *Survivors behave like fresh re-rolls.* On the puzzles intact continuation had not solved by 32 (315 pooled), its completion rate over iterations 33–48 is 0.257. The same puzzles' re-roll rate (16 iterations after the kick at 16, four perturbed arms) is 0.256. This is the memoryless pattern P22 tests directly.
3. *Parallel against sequential at matched compute (48 iterations).* Puzzles solved by at least one of three perturbed 16-iteration runs: 99 / 122 / 111. Intact continuation for 48 iterations: 90 / 128 / 119.

**Reading.** On a second architecture and three more initializations, an answer-preserving perturbation of the hidden state neither helps nor hurts stalled trajectories at any horizon up to iteration 64. It resamples them. The stalled phase behaves like a memoryless chaotic search on both mixers. P22 tests the memoryless law directly on the attention models.
