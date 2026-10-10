# Note 2026-10-10 — P25: a state-dependent collective repair law (registration, before any row; approved by the PI)

**Goal (page one).** A MEASUREMENT.
- Tool: `tools/rebuttal_p25.py`, which imports the P15 runner, P16's relabel and commitment, and Stage 1's statistics unchanged.
- Frozen laws: `tools/p25_frozen_laws.json` (sha256 f5f43556…), fitted before any P25 row.
- Outputs: `runs/analysis/rebuttal_20261011/`.
- Inference only, on the Mac's CPU. No training.
- Receivers: Attention 256 (discovery and primary) and Attention 192 (primary; a frozen replication for Part 2); Attention 128 is reported.

The target is the ambitious version of the result:

> From the present state, we can predict how much compatible correction the network needs. An identified hidden component changes that requirement. Manipulating it makes fresh unsolved states easier to complete.

P25 tests the first two links. Compactness and beneficial control are staged into P26 and P27, with their criteria stated here.

## Why

Stage 1 (`Note_2026-10-10_SeqA_Explanation_Comparison_Plan.md`, including its correction addendum) found, on already-inspected banks:
- The amount of correction predicts repair at cells that were not edited.
- The tested configuration features and hidden summaries add little.

It also left four questions open:
1. The plotted dose curve mixes dose with population: the evaluated cells and contributing states change with dose.
2. The strongest feature, y0, uses the unedited continuation, i.e. future information.
3. Many separate additive contributions crossing a decision boundary were not distinguished from a collective interaction.
4. Hidden coupling was not tested broadly.

Collective reconstruction and hidden coupling are compatible: hidden interactions could implement a collective law. P25 tests the law prospectively on a fixed population, and tests whether retained state sets it.

## Part 1 — the frozen law's numerical predictions on a fixed evaluation population

**Population.**
- 128 fresh test puzzles per receiver: 16 from each of the full test set's eight rating octiles, as the study's 512 were stratified (rank-based bins).
- Excluded: the study's 512, and the P12 / P17 / P20 / P22 attention pools.
- Seeds `[20261011, 25, width, octile]`.
- Intact fixed-start states after iterations 1 and 2, computed through the release step.
- Kept if unsolved with at least 10 wrong cells.

**Fixed evaluation cells.** For each state, the evaluation set E is 4 wrong cells chosen uniformly *first*. E is never edited, in any arm. Corrections are sampled only from the remaining wrong cells R.

**Arms.** Every correction is P16/P18/P19's cell relabel in the network's own coordinates.
- No edit.
- Doses: fractions of R of 0.1, 0.3, 0.5, 0.7 and 0.9, with three random sets per dose. The realized fraction |S| / (number of wrong cells) is recorded.
- Interaction panels: two disjoint sets A and B of equal size, at two sizes (|A| = |B| ≈ 0.15|R| and ≈ 0.25|R|). Each panel has arms A, B and A ∪ B, with two independent panels per size.

Every arm continues four iterations through the release step. For each arm, the displayed grid and all nine digit scores at the E cells are saved.

**The frozen laws.** Both are fitted per width on every (P19 variant with f > 0, wrong cell it does not edit) row at k = 1, before any P25 row: 61,076 / 54,554 / 80,418 rows. Each is a logistic regression on training-standardized features, with L2 1e-3 on slopes. Their functional form, coefficients, preprocessing and uncertainty procedure are committed in `tools/p25_frozen_laws.json`:

| law | kind | features |
|---|---|---|
| **L_c** | conditional (sees the no-edit continuation first) | y0, f, f², y0·f, the state's wrong-cell count, t |
| **L_p** | present-state (nothing from any continuation) | f, f², the evaluated cell's current margin, its retained-commitment index, its visible conflicts, its wrong peers, the wrong-cell count, t |

L_p is the stronger predictive target. L_c is the direct test of Stage 1, and is never described as present-state prediction. Both laws use ground truth through the correction design, so both are diagnostic, not deployable.

**The additive comparator, on the score scale.** Δm(S) is the change, against no edit, of an E cell's margin toward its solution digit (solution-digit score minus the best other score) at k = 1. The additive model predicts Δm(A ∪ B) = Δm(A) + Δm(B), and correctness as the sign of m₀ + Δm(A) + Δm(B). Curvature in binary correctness alone can therefore arise from thresholding additive score changes. Additivity is tested on the margin.

Each law's frozen baseline (M0) is fitted the same way: **B_c** = y0, wrong-cell count and t; **B_p** = wrong-cell count and t. Cross-validated on P19 (log-loss / AUC):

| receiver | L_c | B_c | L_p | B_p |
|---|---|---|---|---|
| Attention 256 | 0.465 / 0.836 | 0.511 / 0.792 | 0.558 / 0.740 | 0.601 / 0.665 |
| Attention 192 | 0.459 / 0.855 | 0.543 / 0.769 | 0.526 / 0.801 | 0.614 / 0.687 |
| Attention 128 | 0.516 / 0.818 | 0.532 / 0.800 | 0.606 / 0.723 | 0.630 / 0.689 |

**Primary quantities and letters (k = 1; puzzle-bootstrap 95 % intervals, 2,000 resamples, seeds `[20261011, 25, width, i]`).**

- **P1a, rates on the fixed cells.** Held-out log-loss of each law against the baseline M0 (state and y0 for L_c; state alone for L_p), and absolute calibration error by dose.

  | letter | condition |
  |---|---|
  | CALIBRATED | the law beats its baseline with an interval excluding 0, and has calibration error ≤ 0.05 at ≥ 4 of 5 doses |
  | MISCALIBRATED | beats its baseline, but fails the calibration bar |
  | NO-PREDICTION | otherwise |

  Lettered separately for L_c and L_p.
- **P1b, variation across states.** Spearman correlation between per-state predicted and observed repair on E, pooled over doses. STATE-PREDICTED if ρ ≥ 0.30 with an interval excluding 0.
- **P1c, the joint response.** The mean margin interaction I = Δm(A ∪ B) − Δm(A) − Δm(B) over E cells and panels:

  | letter | condition |
  |---|---|
  | SUPER-ADDITIVE | the interval's lower end is above 0 |
  | SUB-ADDITIVE | the upper end is below 0 |
  | ADDITIVE | the interval lies within ±δ, with δ = 10 % of the mean |Δm(A)| + |Δm(B)| |
  | UNRESOLVED | otherwise |

  Beside it: which of L_c and the additive-threshold model better predicts binary correctness in A ∪ B (accuracy, with a paired bootstrap).

## Part 2 — does retained state set the collective response?

**Construction** (decided by the feasibility probe below; natural matched pairs do not occur). P25 registers the alternative construction explicitly, never treating it as naturally occurring. The recipient is a history H_A at iteration t; the donor is a Gaussian-start history H_B of the same puzzle at the same t. The **hybrid** keeps H_A's readout-parallel slow part (so every displayed score is kept exactly) and takes H_B's readout-invisible slow part.
- Transfers are reciprocal (B ← A as well).

**Pairs.** For each puzzle, the recipient A is the fixed-start state after iteration 1. The donor B is that puzzle's Gaussian-start history after iteration 1 (start seed `[20261011, puzzle ID, 7]`, through the same release loop). Both must be unsolved with at least 10 wrong cells. Each direction (A ← B and B ← A) has its own four E cells and masks (seeds `[20261011, ID, 4, direction]` and `[…, 5, direction, dose, set]`). The rotation's direction is seeded `[20261011, ID, 6, direction]`.

**Conditions per recipient** (each with its own no-edit continuation as its baseline):

| condition | what it is | expectation |
|---|---|---|
| intact | the recipient unchanged | — |
| transfer | the donor's readout-invisible slow part | the test |
| self-transfer | the recipient's own, through the same code path | bitwise equal to intact |
| restoration | transfer, then restore | within 1e-4 of intact, per element (two float32 roundings) |
| angle- and norm-matched random rotation | each vector's readout-invisible part rotated by the same angle that separates it from the donor's, toward a random direction, and scaled to the donor's norm | matches the change exactly in angle and size, without the donor's content |

Each condition gets P1's dose arms on the recipient's fixed E cells: doses 0.1 / 0.5 / 0.9, two sets each, the same masks across conditions.

**The question is whether hidden state transfers the response to correction, not just a better baseline.** Two models are fitted with puzzle-grouped cross-validation on rows of (E cell × condition × arm):

| model | what the condition may change |
|---|---|
| **B (baseline only)** | its own no-edit outcome only; features y0 (the condition's own), f, f², wrong-cell count, with a shared dose response |
| **D (dose response)** | the dose response as well: B plus indicators for transfer and random, and their products with f |

Both are fitted on intact, transfer and random rows; the self-transfer and restoration controls are checked by the gates.

**Letters.**

| letter | condition |
|---|---|
| STATE-SETS-RESPONSE | D beats B with a held-out log-loss interval excluding 0, on 256 (discovery) and replicating on 192 (frozen) |
| BASELINE-ONLY | B suffices on both |
| UNRESOLVED | otherwise |

**TRANSFER-CARRIES** holds if the transfer moves the recipient's dose response toward the donor's intact one, beyond baseline, significantly more than the angle-matched random rotation does. It is measured per (pair, direction, dose) on the response beyond baseline, where response = mean repair minus mean no-edit repair over the E cells and sets. The transfer's change from intact is regressed (ordinary least squares) on the donor's intact response minus the recipient's intact one, the donor measured in the reciprocal direction at the same dose. The same is done for the random rotation. TRANSFER-CARRIES requires the transfer slope's puzzle-bootstrap interval to exclude 0, and the transfer-minus-random slope's interval to exclude 0.

The controllable quantity is not presumed to be a threshold. Gain, saturation, and the dose needed for 50 % induced repair are each reported.

**Feasibility (probed before this note; display matching only, no correction outcome).** On Attention 256: 64 fresh puzzles unsolved at 16 in the record, each with the fixed start and seven Gaussian starts run for 8 iterations.
- 74 % of displays were unsolved, with a median of 31 wrong cells.
- Among 8,801 same-iteration pairs of unsolved displays of the same puzzle: **0** exact matches, 5 within one cell, 46 within three.
- Across all pairs of iterations: 1 exact match in 67,805.

Natural matched unsolved histories are therefore unavailable. **P25 uses the hybrid construction, registered as such.** Its outputs are matched by construction, because the recipient's readout-parallel part, and so every displayed score, is kept. Hybrids can be atypical states. That is why every condition is judged against its own no-edit continuation, and why self-transfer, restoration and the angle-matched random rotation are included.

## Staged next registrations (criteria stated now)

- **P26, compactness.**
  - Discovery (256): reduced components of the transferred slow part, i.e. top-k directions of the donor-minus-recipient difference, and field or cell subsets.
  - Confirmation: fresh puzzles and correction masks not used for identification.
  - Success: a component at most 1/8 of the full part's dimension carries at least half of the full transfer's effect on the dose response, against equally sized random components.
- **P27, beneficial control.** On fresh stalled states, the component predicted to lower the required correction (chosen without the solution) is compared with its opposite, a magnitude-matched random direction, intact continuation and restoration. It is measured on both the predicted intermediate response and completion, at matched compute including identification cost.

## Predictions and credences (before any P25 row)

| quantity | prediction | credence |
|---|---|---|
| P1a, L_c | CALIBRATED | 0.50 |
| P1a, L_p | CALIBRATED | 0.25 |
| P1b | STATE-PREDICTED | 0.50 |
| P1c | SUPER-ADDITIVE | 0.45 |
| P1c | ADDITIVE | 0.30 |
| P1c | SUB-ADDITIVE | 0.10 |
| P1c | UNRESOLVED | 0.15 |
| Part 2 | STATE-SETS-RESPONSE | 0.35 |
| Part 2 | BASELINE-ONLY | 0.40 |
| Part 2 | UNRESOLVED | 0.25 |
| Part 2 | TRANSFER-CARRIES | 0.30 |

## Gates

- States: the release step reproduces itself bitwise on two runs.
- Every relabel produces its designed display.
- No E cell is edited in any arm.
- Transfers and rotations change no displayed score by more than 1e-3 (float64).
- Self-transfer is bitwise equal to intact; restoration is within 1e-4 of intact per element.
- Every rotation's realized angle is within 1 % of its target, and its norm within 1 % of the donor's.
- A gates-only smoke on 4 puzzles of Attention 256 runs before launch, after this note, the tool and the frozen laws are committed.

## Compute

Historical: P19 on Attention 256 ran 5,360 four-iteration variants in about 37 minutes. Expected per width: about 200 Part-1 states × 28 arms, and about 110 pairs × 2 directions × 5 conditions × 7 arms. The three widths run in parallel at nice 5. Timing is to be measured on the smoke.

## Plan

1. Commit and push this note, the tool and the frozen laws before any row.
2. Run the gates-only smoke.
3. Launch the three receivers.
4. Produce the report with the tool's frozen reader.
5. When it is read, append the Outcome here and write a ledger line.

---

**Amendment 1 — a pre-launch fix and the smoke (2026-10-10, before any row of the registered population was analysed).**
- The first smoke stopped immediately on a name clash. The tool's method `roll` was shadowed by the instance attribute `self.roll`, which the P15 base class binds to its jitted rollout. The method is renamed `roll_variants`; no logic changed.
- The gates-only smoke on the first 4 puzzles of Attention 256 then passed every gate:
  - 6 Part-1 states and 168 variants; 4 Part-2 recipients and 140 variants;
  - no display failures, and no evaluation cell edited;
  - largest score shift 5.5e-7;
  - self-transfer bitwise equal to intact;
  - restoration within 3.0e-8;
  - rotation angle and norm within 1.5e-8 and 1.0e-8.

  It took 3.5 minutes. It printed no outcome counts.
- The frozen reader was run once on the smoke file with its output suppressed, to check that it completes; it did. No values were viewed. Those 4 puzzles are part of the registered population.

---

## P25 Outcome (2026-10-10 09:30Z; `runs/analysis/rebuttal_20261011/report.{txt,json}`; every gate passed; no shared tool edited during the runs)

**Integrity.**
- On every width: no display failure; no evaluation cell edited in any arm; largest score shift from a transfer, restoration or rotation 7.3e-7; self-transfer bitwise equal to intact; restoration within 3.0e-8; rotation angle and norm within 6.5e-8.
- The release states reproduced bitwise.
- Part 1: 129 / 121 / 164 states (Attention 256 / 192 / 128). Part 2: 118 / 142 / 202 recipients.
- An independent recount agrees exactly on every law gain, calibration error, correlation, margin interaction and transfer slope. A separately implemented B-vs-D comparison (its own optimizer and folds) gives −0.002 / −0.0002 / +0.0002, also ≈ 0.

**Part 1 — the frozen collective law predicts fresh, fixed cells.**

| receiver | L_c gain over B_c [95 %], AUC (baseline) | L_p gain over B_p [95 %], AUC (baseline) | P1a letters (L_c / L_p) | per-state ρ, L_c / L_p | P1b | margin interaction I [95 %] (δ) | P1c |
|---|---|---|---|---|---|---|---|
| Attention 256 | +0.074 [+0.053, +0.095], 0.843 (0.770) | +0.062 [+0.043, +0.079], 0.785 (0.684) | CALIBRATED / CALIBRATED | 0.80 / 0.59 | STATE-PREDICTED (both) | +9.4 [+3.2, +15.4] (4.8) | **SUPER-ADDITIVE** |
| Attention 192 | +0.118 [+0.087, +0.149], 0.898 (0.801) | +0.110 [+0.085, +0.134], 0.851 (0.730) | CALIBRATED / CALIBRATED | 0.83 / 0.68 | STATE-PREDICTED (both) | +6.0 [+0.4, +11.4] (3.5) | **SUPER-ADDITIVE** |
| Attention 128 | +0.017 [+0.008, +0.027], 0.822 (0.807) | +0.016 [+0.004, +0.026], 0.718 (0.703) | CALIBRATED / CALIBRATED | 0.79 / 0.56 | STATE-PREDICTED (both) | +1.9 [−1.7, +5.6] (3.2) | UNRESOLVED |

- Calibration error by dose was ≤ 0.05 at at least 4 of 5 doses for every law and width. The exceptions are L_p at dose 0.1 on 256 (0.080) and dose 0.9 on 128 (0.062 / 0.053).
- *Beside P1c.* For binary correctness in A ∪ B, the additive-threshold model is at least as accurate as L_c: 0.758 vs 0.734, 0.811 vs 0.768 and 0.785 vs 0.764, with paired intervals including or touching 0. That model uses the measured single-set margin changes at the same cells, which the law does not see.

**Part 2 — retained state and the response to correction.**
- **B against D:** the D-over-B held-out gain is −0.0005 [−0.0018, +0.0008] / −0.0000 [−0.0010, +0.0009] / +0.0005 [−0.0012, +0.0022]. The letter is **BASELINE-ONLY**: allowing transfer or rotation, as a condition, to change the dose response adds no predictive value beyond each condition's own baseline.
- **TRANSFER-CARRIES**, as the transfer slope [95 %] against the random slope, with the interval for transfer minus random:

  | receiver | transfer | random | transfer minus random | letter |
  |---|---|---|---|---|
  | Attention 256 | +0.572 [+0.474, +0.662] | +0.384 | [+0.073, +0.289] | **TRANSFER-CARRIES** |
  | Attention 192 | +0.561 [+0.456, +0.663] | +0.231 | [+0.190, +0.480] | **TRANSFER-CARRIES** |
  | Attention 128 | +0.546 | +0.502 | [−0.066, +0.180] | NOT-SHOWN |

- *Descriptive.* Pooled induced repair by dose is nearly identical across conditions (intact, transfer, random: 0.21 / 0.16 / 0.22 at dose 0.1 and 0.95 / 0.90 / 0.95 at 0.9 on Attention 256). This matches BASELINE-ONLY: the movement is pair-specific, not a common shift.

**Predictions scored.**
- P1a L_c CALIBRATED (0.50): HIT. P1a L_p CALIBRATED (0.25): HIT.
- P1b STATE-PREDICTED (0.50): HIT.
- P1c SUPER-ADDITIVE (0.45): HIT on 256 and 192.
- Part 2 STATE-SETS-RESPONSE (0.35): MISS; BASELINE-ONLY (0.40): HIT.
- TRANSFER-CARRIES (0.30): HIT on 256 and 192.

**Exploratory (after the read).** A split-half check of TRANSFER-CARRIES addresses the shared-noise concern. The recipient's measured intact response enters both axes of the registered regression, which biases any re-rolled condition toward a positive slope. Using disjoint evaluation cells and disjoint mask sets on the two axes:

| receiver | transfer | random | transfer minus random |
|---|---|---|---|
| Attention 256 | +0.33 [+0.21, +0.44] | +0.18 [+0.09, +0.26] | [+0.05, +0.25] |
| Attention 192 | +0.39 [+0.26, +0.50] | +0.14 [+0.03, +0.22] | [+0.12, +0.38] |
| Attention 128 | +0.31 | +0.27 | [−0.05, +0.16] |

The donor-specific excess survives on both primary receivers. The random rotation keeps a smaller positive slope, consistent with a re-rolled trajectory moving toward typical behaviour. That generic part is what the registered comparison removes.

**The registered sentences, read plainly.**
- *Part 1:* "A collective law fitted on earlier interventions predicts, on fresh puzzles and fixed never-edited cells, how often untouched errors are repaired as more of the configuration is corrected, and which states repair more — also from the present state alone. Joint corrections are super-additive in the network's own margins on the two wider models. The interaction is not only thresholded additive score changes."
- *Part 2:* "The readout-invisible slow state does not shift the law as a whole, but a same-puzzle donor's readout-invisible part carries part of that donor's specific response to correction into the recipient, beyond an exactly angle- and norm-matched random change, on the two wider models."
- *Under every outcome:* the corrections are oracle-guided diagnostic edits; the hybrids are constructed states; one checkpoint per receiver; random directions; the measured horizon is one iteration.

**Reading.**
- The collective repair law is predictive, not only descriptive.
- Retained state carries transferable, state-specific information about how the configuration will take up a correction. It is a pair-specific "parameter" of the response rather than a uniform change of the law.
- This satisfies the first two links of the target chain: present-state prediction of the needed correction, and a transferable hidden component that changes the response.
- P26 (compactness) is the next registration. Can a component at most 1/8 of the slow part's dimension carry at least half of this donor-specific transfer, against equally sized random components, confirmed on fresh puzzles and new masks?
