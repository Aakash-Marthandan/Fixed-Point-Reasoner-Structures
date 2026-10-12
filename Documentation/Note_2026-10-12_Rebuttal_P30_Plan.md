# Note 2026-10-12 — P30: a bounded exploratory check for a selection signal in the P28 bank (plan, written before any feature is computed)

**Goal (page one).** EXPLORATORY. No new outcome experiment: only a bounded, deterministic replay for features.

The question: on P28's bank-failing Sudoku puzzles, does information available *before continuation* distinguish which intervention succeeds **within the same puzzle**, rather than identifying easier puzzles? Does selecting by it beat **equally informed** selection among matched controls and among restarts, once the work of probing every inspected candidate is charged?

A positive signal would justify a fresh, registered confirmation on new puzzles. **It is not a performance claim.** P29 motivates the question but gives no evidence that selection will work. P28 found the selection opportunity generic: every strategy completes about 17–19 % of rollouts per 16-iteration window.

## Data (all from P28; `runs/analysis/rebuttal_20261014/attention_{192,256}/p28.npz`)

- **Puzzles:** 128 bank-failing puzzles per receiver. Attention 192 is primary; Attention 256 is a replicate.
- **Candidates per puzzle:**
  - 12 ordered pairs (recipient i ← donor j over the bank's four histories) × {leading transplant, its matched control, full transplant, its control};
  - 12 equal-work restarts.
- **Outcome:** success = a verified readout within the 16-iteration window (`first` ≥ 0), as registered in P28.

## Features (solution-free, frozen now)

**Recipient-level.** From the bank history i at iteration 16; the bank is regenerated deterministically.
- Constraint violations of the displayed grid (duplicate digits in rows, columns, boxes).
- Score margin: the mean, and the 10th percentile over cells, of top-1 minus top-2.
- Cells changed between iterations 15 and 16, and over iterations 13–16.
- Stalled: display(16) = display(15).
- Start type: fixed or Gaussian.

**Pair-level** (transplants and their controls; the same values for a transplant and its control, because the control shares the pair's geometry):
- the donor's violations and margins;
- recipient–donor display disagreement at iteration 16;
- ‖Δ‖/‖h‖ for the full and leading parts;
- the energy of Δ in P26's frozen basis.

**Restart-level.** After a **probe of 2 iterations** (charged): violations, margins and changes at the probe's end.

No feature uses the solution or any post-continuation quantity. Features are computed and saved before they are joined to outcomes.

## Analysis (frozen now)

**Model.** L2 logistic regression on standardized features.
- Standardization, feature use and the penalty (chosen from {0.01, 0.1, 1, 10} by inner puzzle-grouped 4-fold cross-validation) are all fitted **inside** the training folds.
- Outer evaluation: 8 puzzle-grouped folds. All candidates of a puzzle stay together.
- One model per candidate family: transplants, controls, restarts.

**Within-puzzle discrimination.**
- **Primary readout:** held-out within-puzzle AUC, over pairs of candidates in the same puzzle with discordant outcomes. A puzzle-bootstrap interval uses 2,000 draws.
- Puzzles with no discordant pair contribute nothing, and their count is reported.

**Selection.**
- Per puzzle, the top-1 candidate by held-out score. Success rate of the pick against a random pick from the same family (the family's base rate), and against the family oracle; puzzle-bootstrap intervals.
- **Equally informed comparators:**
  - controls are ranked by the same recipient and pair features;
  - restarts by their probe features.
- **Compute accounting**, reported for every strategy, in model iterations per puzzle:
  - transplants and controls: the bank (4 × 16 = 64), plus construction (no forward work), plus the continuation (16);
  - restarts: 12 probes × 2, plus the remaining 14 iterations of the chosen restart.

  Both views are shown: "bank charged" and "bank sunk", the latter as in P28's framing of existing histories.

**Signal rule (exploratory, the gate for any confirmation).** A selection signal is reported only if, on Attention 192:
1. the transplant family's held-out within-puzzle AUC has a 95 % lower bound above 0.5; **and**
2. the top-1 gain over random among transplants exceeds the equally informed gain among controls (paired by puzzle, interval above 0); **and**
3. it exceeds the gain among restarts at the reported compute.

Attention 256 is reported as a replicate. Anything else reads "no selection signal at this size". A signal leads only to a proposal for a registered confirmation on fresh puzzles.

## Compute and order

- **Replay** (deterministic; the P28 runner's `bank` and the restarts' first 2 iterations): about 20–30 min per receiver on the Mac's CPU (the float32 audit's measured bank time), plus minutes for features and models.
- **Order:** it runs after the P31 replication, if P31 is approved first, so the two do not share the CPU. Otherwise it can start now.
- **Outputs:** `runs/analysis/rebuttal_20261016/` (features `p30_features_{w}.npz`, then `p30_report.{txt,json}`).

## P30 Outcome (2026-10-12 ~18:45Z; EXPLORATORY; `runs/analysis/rebuttal_20261016/p30_report.{txt,json}`; plan 47c2fb4 unchanged)

**Replay.** Deterministic, outcome-blind; 1,728 s (192) and 2,733 s (256).
- **Gate:** the regenerated banks reproduce P28's screening. No bank history verified on any of the 256 puzzles.
- The features were saved before outcomes were joined.
- One fix was needed before the analysis ran: the cross-validation seeds had to be flattened. The selftest had used a scalar seed.

**Held-out results (8 puzzle-grouped outer folds; inner 4-fold choice of the L2 penalty).**

| receiver | family | base rate | within-puzzle AUC [95 %] | top-1 gain over a random pick [95 %] |
|---|---|---|---|---|
| 192 | leading transplant | 0.193 | 0.506 [0.467, 0.547] | −0.005 [−0.061, +0.059] |
| 192 | matched control | 0.189 | 0.501 [0.451, 0.548] | +0.030 [−0.025, +0.089] |
| 192 | full transplant | 0.184 | 0.533 [0.486, 0.579] | +0.035 [−0.024, +0.098] |
| 192 | restart (2-iteration probe) | 0.172 | 0.533 [0.491, 0.574] | +0.031 [−0.030, +0.093] |
| 256 | leading transplant | 0.169 | 0.524 [0.473, 0.570] | −0.012 [−0.069, +0.049] |
| 256 | full transplant | 0.166 | 0.454 [0.409, 0.500] | −0.057 [−0.111, +0.000] |
| 256 | restart | 0.162 | 0.465 [0.422, 0.514] | +0.057 [−0.007, +0.118] |

- **Equally informed comparisons on 192:**
  - the leading-transplant gain minus the matched-control gain, paired by puzzle: −0.035 [−0.120, +0.049];
  - the leading pick minus the restart pick: −0.016 [−0.102, +0.070].
- **Compute per puzzle:** transplant or control selection costs 80 model iterations (16 if the bank is treated as sunk); restart selection costs 38.
- Inner cross-validation chose the strongest penalty (C = 0.01) in almost every fold, as expected when the features carry little information.

**The plan's signal rule:** on 192 the AUC, beats-controls and beats-restarts conditions all fail → **no selection signal at this size.** Attention 256 agrees.

**Sanity check after the frozen analysis** (`feature_check.py`, exploratory).
- Every feature varies, except "stalled" (no bank history stalls at iteration 16).
- The univariate within-puzzle AUCs of the transplant features lie between 0.46 and 0.53 on both receivers.
- The one larger value is the restart probe's mean margin on 256: 0.576 [0.533, 0.621]. It is the best of 5 features, uncorrected, and absent on 192 (0.510). It is at most a lead about restarts, not about transplants.

**Reading.**
- Information available before continuation does not identify which transplant succeeds *within* a puzzle. This holds for these solution-free features (recipient state, donor state, pair geometry and the energy in P26's basis) on this bank. The selectors do no better than equally informed selection among matched controls or among restarts.
- This agrees with P28's finding that the selection opportunity is generic: each strategy succeeds in about 17–19 % of 16-iteration windows.
- The transferable correction response (P26/P27, P29) is therefore, so far, a mechanistic property without a known rule for choosing beneficial configurations.
- **Not shown:** that no feature could work. Richer probes, longer windows or learned representations remain untested, and a positive future signal would still need a registered confirmation on fresh puzzles.
