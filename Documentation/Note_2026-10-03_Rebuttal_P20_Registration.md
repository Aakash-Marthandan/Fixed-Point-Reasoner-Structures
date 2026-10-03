# Note 2026-10-03 — P20: a confirmatory test of a moderate rotation of the whole hidden state on stalled trajectories (registration, before any row)

**Goal (page one).** A MEASUREMENT. No accuracy target; $0; the Mac's CPU; inference only on the manuscript's banked attention checkpoints; nothing trained. Tool: `tools/rebuttal_p20.py` (imports P17's runner unchanged; outputs `runs/analysis/rebuttal_20261003c/`). Registered before any P20 row and before Attention 256's P17 results were seen.

## Why

P17 (`Note_2026-10-02_Rebuttal_P15_P16_P17_Registration.md`, Amendments 4–5) ran its dose arms descriptively. On Attention 128 and 192 (complete; 256 rerunning after a bug), the strongest arm by iteration 32 was a 60° norm- and score-preserving rotation of the readout-invisible slow state, applied once at iteration 16 to stalled trajectories: 90 against 71 solved by intact continuation on 128 (37 gained, 18 lost, exact McNemar p = 0.015) and 75 against 60 on 192 (33 / 18, p = 0.049); smaller rotations did not help consistently and a full random replacement traded rescues of never-solved puzzles for losses among those that would have completed anyway. The pre-specified guided arm of P17 reduced to a single kick on those widths, because stalled trajectories there never freeze. The 60° rotation is therefore an exploratory finding selected among several arms; this experiment tests it on fresh puzzles with the rotation pre-specified.

## Design

**Population.** Per receiver (Attention 128, 192, 256): a fresh seeded uniform draw of 384 test indices from the saved full-test record's unsolved-at-16 pool, excluding P12's pool and that receiver's P17 pool (seed `[20261003, 20, width]`). Sixteen fixed-start iterations through the release step on the CPU. Trigger (no answer key): the displayed grid at iteration 16 is invalid (a duplicate in a row, column or box, or an altered given); asserted equivalent to CPU-unsolved. Strata: intact continuation to iteration 64 on the same run labels each triggered puzzle transient (solved by 64) or trapped.

**Arms**, from the iteration-16 state, each continued to iteration 32: A0 intact (to 64 for the strata); **R60, pre-specified**: the readout-invisible slow part at every cell rotated by 60° toward a random direction orthogonal to the readout and to itself, scores and norms kept (P10/P17's operation; seed `[20261003, puzzle ID, 81, 60]`); R45 and R75 (the same with 45° and 75°, exploratory: the operating range); A5 a fresh Gaussian restart run for 16 iterations (seed `[20261003, puzzle ID, 82]`). Gates: every rotation changes no digit score by more than 1e-2; the trigger's equivalence asserted. A smoke on 16 puzzles of Attention 128 passed both.

## Rules (solved by iteration 32 among triggered puzzles; exact McNemar)

- **ROTATION-HELPS** per receiver if R60 solves more than A0 with p < 0.05 (ROTATION-HURTS if fewer with p < 0.05; NO-DIFFERENCE otherwise). The confirmed result requires ROTATION-HELPS on at least two of the three receivers.
- **BEATS-RESTART** per receiver if R60 solves more than A5 with p < 0.05; confirmed on at least two of three.
- **TRAPPED-RESCUE** if R60 solves at least 10 % of the trapped puzzles (which A0 by definition does not solve by 64) on at least two of three receivers.
- Reported beside: every arm by stratum, losses per arm against A0, the R45 / R60 / R75 operating range.

**Predictions and credences (before any row).** ROTATION-HELPS confirmed 0.55; BEATS-RESTART confirmed 0.50; TRAPPED-RESCUE 0.55; R60 the best of the three angles on at least two receivers 0.40.

**Wording under each outcome.** ROTATION-HELPS: "One moderate rotation of the hidden state, with every displayed score kept, solves more stalled puzzles within sixteen further iterations than continuing unchanged." With TRAPPED-RESCUE: "... including puzzles that continuing does not solve within 48." NO-DIFFERENCE: "The gain seen in the exploratory arms does not replicate on fresh puzzles." Under every outcome: the rotation's direction is random; the trigger uses no answer key; the angle was chosen from P17's exploratory arms, disclosed here.

## Plan

Commit this note and the tool before any row (locally; pushing waits for the other session's local-only commit to clear); queued behind P17's width-256 rerun (PID-based); three widths in parallel at nice 5; report by the tool; the Outcome appended here and a ledger line written when read.
