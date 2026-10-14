# Note 2026-10-13 — P32: can a compact description predict unmeasured, correction-specific responses, and can a controlled state edit change them by a predicted amount? (plan, revised after the PI's review; Stage 1 frozen before it runs)

**Goal (page one).** A bounded prediction-and-intervention study, **developed on Attention 192** and then frozen for Attention 256.
- The chain it tests: measured structure → prediction of previously unmeasured, correction-specific responses → accurate prediction of how a controlled state edit changes them.
- **Not an accuracy run:** no repair benefit has been shown (P27, P29, P31). No training; inference on the Mac's CPU.
- Stage 1 is exploratory model selection on data already inspected, so it makes no confirmatory claim. Stages 2 and 3 are registered after Stage 1, with every threshold frozen beforehand.

## The PI's review of the draft (2026-10-13), adopted

1. **The decisive test is correction-specific.** Better overall error could come from merely knowing which states respond strongly to larger corrections. The primary target is therefore the *differences between unmeasured masks of the same size, on the same state and evaluation cell*. With one probe per dose, three masks per dose remain. The overall response prediction is reported separately.
2. **M3 represents the proposed correction explicitly.** It sees the edited cells, their old and new digits, and the edit itself (in this model a correction swaps the old and new digits' field vectors at the cell, in the slow and the fast state), together with their interaction with the state. Its view is not restricted to cells sharing a unit with the evaluation cell: it also has a global state summary and a representation of the whole correction set. It is a small regularized model. A bilinear low-rank interaction model competes, and it too must predict a new mask from its description alone.
3. **Strong, equally calibrated baselines.** The strongest baseline without hidden-state features receives the same three probe responses as M3.
   - Features available before continuation (the initial scores) are free.
   - The no-edit continuation is a rollout, charged to the probe panel (3 masked + 1 no-edit rollout per state).
   - Every feature that uses the solution digit is labelled **oracle-assisted** and cannot support a deployable selection claim.
   - M1 and M2 (rollout-assisted) are references only.
4. **The split around P26's frozen basis.** U was built after selecting P26's discovery states by their response outcomes, so the **104 discovery puzzles are training-only** for any predictor that uses U. They never appear in validation or test scoring, the inner folds included. Both histories, every mask and every intervention are grouped by puzzle. Scaling, regularization and any dimensionality reduction are fitted inside the training folds.
5. **Hidden-state reconstruction.** The archives hold outcomes, not hidden tensors. Stage 1 regenerates the states by a bounded deterministic replay of P26's and P27's pair construction, in a scratch output directory, with provenance checks against the archives (below). "Minutes" is not assumed; the replay is timed.
6. **Stage 3 tests accurate intervention predictions, not correlation.**
   - The predictor and its calibration are frozen before any new intervention response is observed.
   - The strongest form is calibrated on the **intact recipient's** probes, and that calibration is reused for every transplant and control.
   - It predicts the response *change* each intervention causes, scored by squared error and calibration slope, including the donor-minus-control contrast.
   - The controls use P27's construction: per row, the displacement norm and the inner product with the recipient are matched inside the chosen subspace. Each control is drawn once and scaled across λ.
   - A monotone donor coefficient in λ is secondary.
7. **Practical details are frozen before confirmation:** eligible (not merely screened) sample sizes, insufficient-variation rules, exact thresholds, the role of λ = 1.5, and, for width 256, the distinction between replicating the learning procedure and transferring fitted coefficients.

## Stage 1 (exploratory model selection; frozen now, before it runs)

**Data.** Attention 192, from three archives (outcomes saved; states regenerated):
- P26 stage D: 104 discovery puzzles, 208 pair-directions. **Training only.**
- P26 stage C: 104 puzzles, 208 pair-directions.
- P27: 192 puzzles, 384 pair-directions.

Scored puzzles: P26 C and P27, 296 puzzles. P27's hybrid conditions are used only for the Stage-3 prototype.

**Provenance gates.**
- The regenerated pairs reproduce each archive's puzzle ids, directions, evaluation cells, displayed grids, solutions and every mask's cells exactly.
- For a sample of pair-directions in every archive, the regenerated states re-rolled through the archive's own rollout reproduce the saved evaluation-cell scores of the no-edit arm and one mask per dose (deviation reported; bitwise expected).
- The basis hash is asserted. The replay writes nothing into the archives' directories.

**Target.** The evaluation cell's solution-digit margin (solution score minus the largest other) one update after the change, k = 1. Δ(m, e) = margin(m, e) − margin(no edit, e).
- **Primary target:** for the three unmeasured masks of a dose, Δ centered among themselves. The no-edit term cancels, so the primary target needs no no-edit rollout.
- **Probes:** mask 0 of each dose is measured; masks 1–3 are unmeasured.
- **Insufficient variation:** a (state, cell, dose) group whose centered target has RMS < 1e-4 is excluded from scoring and counted.

**Models** (each is a two-part ridge regression):
- a *dose-level* part, fitted on the mean of a dose's masks;
- a *specific* part, fitted on within-dose centered targets with within-dose centered features, using all four masks of each dose in the training puzzles.

| model | features | label |
|---|---|---|
| B0 | amount and geometry: dose, mask size, corrected cells sharing the evaluation cell's row / column / box, unrelated count | hidden-free, oracle-free |
| B1 | B0 + display-level digit relations (related corrections whose new or old digit equals the evaluation cell's displayed digit or its runner-up) + the evaluation cell's initial score gaps + the corrected cells' old-vs-new score gaps + displayed duplicates and grid violations | hidden-free, oracle-free |
| B1o | B1 + the evaluation cell's initial solution-digit margin + related corrections whose old digit is the cell's solution digit + the number of wrong cells | hidden-free, **oracle-assisted** |
| M3 | B1 + hidden features (U-coordinates, readout-invisible): the evaluation cell's displayed-digit row and field mean; the correction set's mean swap displacement in U (related and unrelated); inner products between the evaluation cell's rows and the corrected cells' old and new rows (related and unrelated); a global summary (mean displayed-digit row; mean of all rows); the evaluation row scaled by the related count | hidden, oracle-free |
| M3o | M3 + B1o + the evaluation cell's solution-digit row in U and its inner products with the corrected rows | hidden, **oracle-assisted** |
| M3b (exploratory) | M3 + bilinear interactions: the evaluation row ⊗ the related corrections' summed swap displacement (576) | hidden, oracle-free |
| Rother (reference) | the other history's measured centered responses to the same masks, times a gain fitted in training | **rollout-assisted** (needs the donor's rollouts) |

- **Calibration, identical for every model:** per (state, evaluation cell), a gain on the specific part, fitted from the three probes' deviations from the model's own dose-level prediction, shrunk toward 1 by a training-fold prior.
- The uncalibrated predictions are reported too.
- **Fitting:** features are standardized and ridge strength is chosen from {0.1, 1, 10, 100, 1000} by inner 4-fold puzzle-grouped cross-validation. P26-D puzzles are in every inner training fold and never in an inner validation fold. There are 8 outer folds over the 296 scored puzzles.

**Scoring and the Stage-1 rule.**
- The primary metric is the held-out SSE of the centered unmeasured-mask targets.
- **The rule:** M3 (oracle-free) qualifies for Stage 2 if its error is ≥ 10 % below the strongest hidden-free oracle-free baseline (the lower of B0 and B1), with a 2,000-draw puzzle-bootstrap lower bound > 0. The same rule applies to M3o against the stronger of B1 and B1o, labelled oracle-assisted.
- The Stage-2 candidate is the oracle-free M3 if it qualifies, otherwise M3o (labelled). If neither qualifies: "no compact predictive description of correction-specific responses at this level", and Stages 2 and 3 are not run in this form.
- **Reported:** the overall Δ prediction (it charges the no-edit rollout), k = 2, Rother, M3b, and the excluded-group counts.

**Stage-3 prototype (exploratory, P27's hybrids).**
- The intact-trained models predict the responses of the leading transplant, its control, the full transplant and the combined control, with the **intact recipient's** probe calibration.
- They are scored on the response change each intervention causes on the unmeasured masks, against the zero change every hidden-free model must predict (the readout, and so every visible feature, is unchanged). Also scored: the calibration slope and the donor-minus-control contrast.

## Stages 2 and 3 (to be registered after Stage 1)

- **Stage 2:** fresh puzzles in P27's construction with new seeds. Eligible sample size and thresholds are frozen beforehand. The frozen model and calibration predict the unmeasured masks before their rollouts are read. **Test:** the primary within-dose error reduction against the frozen strongest baseline, with a lower bound > 0 at the registered margin.
- **Stage 3:**
  - h_λ = h_r + λ (h_d − h_r) U Uᵀ for λ ∈ {0.5, 1.0} (primary) and 1.5 (secondary, extrapolation);
  - P27-construction controls drawn once and scaled by λ;
  - each condition's own unedited continuation.
  - Frozen predictions of each intervention's response change are scored by squared error against zero change, by the calibration slope and by the donor-minus-control contrast.
- **Attention 256:** the frozen *procedure* is replicated (refit on 256's data). The fitted *coefficients* cannot be transferred, because the widths and bases differ.

**Compute.** Stage 1 is a replay (timed) plus model fitting. Stages 2 and 3 are P27-sized local runs, priced on a smoke before registration.

## P32 Stage 1 Outcome (2026-10-13 ~13:55Z; EXPLORATORY; `runs/analysis/rebuttal_20261018/p32_stage1_k{1,2}.{txt,json}`; plan and tool committed e46f848 before any number)

**Replay and provenance** (389 s; scratch metadata; nothing written into the archives).
- All three archives were regenerated exactly: P26 D 208, P26 C 208 and P27 384 pair-directions.
- Puzzle ids, directions, evaluation cells, displayed grids, solutions and every mask's cells match with zero mismatches.
- The sampled re-rolls through each archive's own rollout reproduce the saved evaluation-cell scores **bitwise**.
- P27's five hybrid conditions display exactly the recipient's digits.

**The registered result (k = 1; primary within-dose target; 7,104 scored groups, none excluded).**

| comparison (frozen rule: ≥ 10 % with a lower bound > 0) | error reduction [95 %] |
|---|---|
| M3 vs B1 (oracle-free; B1 the stronger visible baseline) | +5.2 % [−0.6, +11.2] |
| M3o vs B1o (oracle-assisted) | +4.9 % [−0.8, +10.7] |
| M3b (bilinear, exploratory) vs B1 | −1.5 % [−7.5, +4.7] |

**→ Neither qualifies: "no compact predictive description of correction-specific responses at this level."** Stages 2 and 3 are not run in this form.
- **Secondary, k = 2:** M3 vs B1 +2.8 % [−1.5, +7.0]; M3o vs B1o +2.5 %. Neither qualifies.
- Inner cross-validation chose the strongest ridge penalty in the grid (1,000) almost everywhere.

**A design flaw in the frozen calibration (disclosed; post hoc, `diag_stage1.py`).**
- The per-(state, cell) gain fitted from three probes was unstable. Its 5–95 % range was −3.4 to +6.0 for M3 and −95 to +107 for B0.
- **Every probe-calibrated model therefore predicted worse than zero:** M3 −20.6 % [−26.8, −14.7] and B1 −27.2 % [−35.9, −20.1] relative to predicting zero. The frozen comparison was between two predictors worse than zero, and the rule should have included zero and the uncalibrated predictors as references.
- **The conclusion does not depend on the flaw.** Uncalibrated, M3 improves on B1 by **+2.7 % [+1.6, +3.8]** and on zero by **+3.7 % [+2.5, +5.0]**. Hidden state adds a small, resolved amount, nowhere near 10 %.

**What the target looks like.** The within-dose differences are heavy-tailed: |T| has a median of 2.4, a 90th percentile of 45, a 99th of 84 and a maximum of 101. A minority of large swings dominate the squared error, and no tested description predicts them.

**The rollout-assisted reference.** Using the other history's measured responses to the same masks improves on zero by only about 1.6 %. Little of the correction-specific response is shared even between two histories of the same puzzle. (Its "+22.7 % vs B1" reflects B1's calibration failure.)

**Stage-3 prototype (exploratory, P27's hybrids, intact-trained M3).**
- With the frozen calibration, the predicted intervention changes are worse than predicting no change, by 7–24 %.
- Without the gain (post hoc, `diag_proto.py`), they are still not better than zero (−0.6 % to −5.5 %).
- They carry a weak direction: correlations 0.17 for the leading transplant, 0.09 for its control, 0.14 for the full transplant, 0.10 for the combined control, and **0.16 for the leading − control difference**; slopes 0.24–0.47.
- The magnitude is badly missed: the observed change has RMS ≈ 58 against a predicted 20.

**Reading.**
- The transferable response structure (P26/P27, P29/P31) is real. But *individual* correction-specific responses at untouched cells are not compactly predictable by linear descriptions of the visible state, the readout-invisible state in P26's frozen basis, and an explicit representation of the correction.
- Hidden-state features add about 3 % beyond visible ones. About 96 % of the within-dose variance stays unexplained, concentrated in large swings.
- This agrees with the synthesis's "sensitivity without control" (P20–P24).
- The chain structure → prediction → controlled realization is blocked at the prediction step for this description class.
- Not shown: that no description could work. Nonlinear models, the network's own local linear response, or a target built around the large swings remain untested.

**Lessons.**
1. A calibration fitted from a few probes per unit must be validated inside the training folds before it enters a frozen rule. The rule must include predicting zero and the uncalibrated predictor as references.
2. A ridge grid chosen at its edge everywhere signals a weak signal and should have extended.
3. For a heavy-tailed target, the error measure and its robust companion should be fixed before the run.

## P32 CLOSED — scope addendum after the PI's review (2026-10-14 ~04:45Z; exploratory; every figure reproduced, `runs/analysis/rebuttal_20261018/review_check.py`)

**Decision.** P32 closes under its original rule. Stages 2 and 3 do not run, and no broad nonlinear-model search follows. The PI's independent reconstruction reproduces the uncalibrated M3 error almost exactly and confirms that P26's discovery puzzles stayed training-only.

**The conclusion, narrowed.**

> These pooled features of the selected slow-state subspace, with the tested regression and calibration procedure, did not meet the prespecified forecasting threshold.

- The "about 96 % unexplained" figure is this predictor's held-out squared error. It is **not** an estimate of irreducible unpredictability, or of information absent from the hidden state.
- **Withdrawn:** "the chain structure → prediction → control is blocked at the prediction step." The tested route to an *accurate numerical response model* failed. A narrower route through *action selection* remains open (below).

**Three qualifications.**
1. **Most prediction error concerns whether a correction succeeds.** 26.9 % of the scored (state, cell, dose) groups contain unmeasured masks that disagree on whether the untouched cell becomes correct, and those groups hold **86.8 %** of the target's squared magnitude. The largest 10 % of |target| values hold 69.3 %. The hard cases mostly cross the correct/incorrect decision boundary. This identifies a concrete behavioural event; it does not establish competing attractors or a global rule switch.
2. **P32 tested a narrower state description than its outcome wording suggested.**
   - Within-dose centering removes **97 of the 176 hidden features** from the specific-response regression: the evaluation cell's displayed-digit row and field mean, its norm, and both global summaries. They are constant within a (state, cell, dose) group.
   - Such summaries can only affect within-dose predictions through interactions with the proposed correction; they enter only the level and calibration parts.
   - The features use the slow state, while the correction edits both the slow and the fast state.
3. **The same uncalibrated predictor can rank corrections** (post hoc diagnostic; the fitted model unchanged). For each history and dose, choose one of the three unmeasured masks by its predicted mean response over the four untouched cells, then score correctness on those cells:

| selection | untouched-cell correctness |
|---|---|
| random among the three masks | 67.63 % |
| visible-feature baseline B1 | 70.13 % |
| hidden-feature model M3 | **71.88 %** |

- M3 − random: **+4.24 pp [+3.33, +5.16]**. M3 − B1: **+1.75 pp [+0.68, +2.79]**. B1 − random: +2.50 pp.
- Among 3,828 discordant mask pairs (one corrects the cell and the other does not), M3 ranks the correcting mask higher **64.0 %** of the time; B1 59.0 %.
- **Scope:** this is *selection among oracle-supplied corrections*. The candidate corrections and the evaluation population were built from the solution. It is not autonomous solving, not whole-puzzle improvement, not hidden-state steering, and it does not overturn P30.
- **Poor prediction of response magnitude does not imply useless prediction of action ordering.**

**Three capabilities, kept distinct.**
- Transferring correction-response structure: supported (P26/P27, P29/P31).
- Forecasting individual response magnitudes accurately: P32's tested approach fails.
- Using state information to choose between corrective actions: a modest exploratory signal (above).

**Next (the PI).**
1. P32 is closed.
2. The ranking result is recorded as exploratory. If more local time is spent here, it gets one small, separately registered fresh-puzzle confirmation with the existing predictor frozen (hidden vs visible vs random selection), run alongside the ARC work and never delaying it.
3. **ARC is the priority** (`Plan_2026-10-13_ARC_Error_Correction_Pilot_DRAFT.md`). Its central test is whether feedback helps choose or construct revisions that improve unseen answers.
4. On ARC, demonstration-derived feedback is available and query solutions are not. A result that depends on oracle query corrections cannot satisfy that goal.
