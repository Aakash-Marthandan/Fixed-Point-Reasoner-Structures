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

---

**Amendment 1 — the strata label (2026-10-03 02:10Z, before any P20 row).** P20 imports P17's runner, whose intact arm runs to iteration 80 rather than 64 (P17 Amendment 6). P20's reader now labels *transient* and *trapped* at iteration 64 as registered (index 47) and reports the iteration-80 count beside it as exploratory. The run is unchanged. No rule, arm, seed or population changed.

---

## P20 Outcome (2026-10-03 05:55Z; `runs/analysis/rebuttal_20261003c/report.{txt,json}`; every gate passed; no shared tool edited during the runs)

**Integrity.**
- On every width the trigger was asserted equivalent to CPU-unsolved. Triggered: 160 / 155 / 145 of 384 (Attention 256 / 192 / 128).
- Every rotation changed no digit score by more than 1e-2; the realized maximum was 7.2e-7 / 8.3e-7 / 6.0e-7.
- Strata use the registered label at iteration 64 (Amendment 1).
- An independent recount from the raw arrays agrees with the report on every arm, stratum, loss count, letter and the aggregate.

**Results (solved by iteration 32 among triggered puzzles).**

| receiver | triggered (transient / trapped) | A0 intact | R45 | R60 | R75 | A5 restart | R60 vs intact | R60 vs restart | trapped solved by R60 |
|---|---|---|---|---|---|---|---|---|---|
| Attention 256 | 160 (111 / 49) | 67 | 73 | 68 | 60 | 64 | 27/26, p 1.0 | 26/22, p 0.67 | 8 (0.16) |
| Attention 192 | 155 (119 / 36) | 83 | 75 | 80 | 71 | 63 | 23/26, p 0.78 | 38/21, p 0.036 | 6 (0.17) |
| Attention 128 | 145 (116 / 29) | 75 | 76 | 71 | 67 | 71 | 22/26, p 0.67 | 25/25, p 1.0 | 4 (0.14) |

Losses against A0 (solved by intact continuation at 32, not under the arm), 256 / 192 / 128:
- R45: 23 / 30 / 27
- R60: 26 / 26 / 26
- R75: 26 / 31 / 31
- restart: 29 / 37 / 27

Trapped puzzles solved by the restart: 9 / 5 / 6.

**Letters.**
- ROTATION-HELPS: **NO-DIFFERENCE** on every width; not confirmed.
- BEATS-RESTART: on Attention 192 only; not confirmed.
- **TRAPPED-RESCUE: holds** (R60 solves 16 % / 17 % / 14 % of the trapped puzzles).

**Predictions scored.**
- ROTATION-HELPS confirmed (0.55): MISS.
- BEATS-RESTART confirmed (0.50): MISS.
- TRAPPED-RESCUE (0.55): HIT.
- R60 the best angle on at least two receivers (0.40): MISS. The best angle was 45° on 256 and 128 and 60° on 192.

**The registered sentence that applies.** NO-DIFFERENCE: "The gain seen in the exploratory arms does not replicate on fresh puzzles."

TRAPPED-RESCUE's registered wording extends the ROTATION-HELPS sentence, which failed, so it is stated here on its own, with its cost: the rotation completes 14–17 % of the puzzles that continuing does not complete within 48 more iterations; it loses as many puzzles that continuing completes; and a fresh restart completes about as many trapped puzzles.

Under every outcome: the rotation's direction is random; the trigger uses no answer key; the angle was chosen from P17's exploratory arms.

**Exploratory.**
1. *Why P17 looked different.*
   - P17's intact continuation solved fewer of its triggered puzzles by 32 than P20's did: pooled over widths, 178 / 438 (0.41) against 225 / 460 (0.49), z = 2.5.
   - The 60° arm's own rate barely moved (0.54 against 0.48).
   - On P17's Attention 256 draw, every perturbed arm beat intact continuation.
   - With the 60° arm selected as the best of eight, a weak intact draw is enough to produce P17's exploratory result. The two populations are uniform draws from the same pool, through identical code paths for the intact arm (checked).
2. *Sensitive, without direction.* About a third of stalled puzzles change outcome by 32 under a rotation that keeps every displayed score (53 / 49 / 48 discordant of 160 / 155 / 145), equally often in each direction. Each perturbed arm's 16-iteration solve rate (0.38–0.52) matches intact continuation's rate over iterations 17–32 (0.42–0.54).
3. *Re-rolls in parallel equal continuing.*
   - Puzzles solved by 32 under at least one of the four perturbed arms (four times 16 iterations): 112 / 121 / 120.
   - Puzzles solved by intact continuation to iteration 80 (64 iterations): 116 / 124 / 120.
   - Intact continuation's per-16-iteration solve rate falls over time: 256: 0.42, 0.35, 0.18, 0.10; 192: 0.54, 0.31, 0.28, 0.14; 128: 0.52, 0.44, 0.26, 0.14.
   - Since fresh perturbed runs do no better than continuing, the falling rate is consistent with heterogeneous puzzle difficulty rather than individual trajectories getting stuck. This reading is untested by a late kick.

**Reading.**
- One answer-preserving perturbation of the hidden state does not improve stalled trajectories on fresh puzzles at any tested angle. Control by a single global kick is not established.
- What the data show is a property of the dynamics: a stalled trajectory is highly sensitive to its readout-invisible state, but a perturbation re-rolls its outcome rather than improving it.
- Together with P17's finding that stalled displays never freeze, stalled correction behaves like a chaotic search whose tail reflects which puzzles are hard.
- P21, registered before this result, runs as registered on the MLP seeds.
