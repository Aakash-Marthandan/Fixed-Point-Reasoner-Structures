# Note 2026-10-12 — P29: does a score-preserving retained-state transplant transfer differential responses to identical corrections on ARC? (registration, before any confirmation row; the PI's three reviews incorporated)

**Goal (page one).** A MEASUREMENT of task and architecture generality, on the public TRM ARC-AGI-1 checkpoint. Inference only, on the Mac's CPU; no training.

**Registered (summary; details in the sections below).**
- **Primary:** option B. The correct-token margin on four never-edited evaluation tokens, one update after the change (k = 1), standardized per query by its pooled intact response scale.
- **Population:** every eligible query of the 352 confirmation tasks; the 48 pilot tasks are excluded.
- **Headline S-ARC:** β(full) > 0 *and* β(full) − β(full_ctrl) > 0 (both 95 % lower bounds; 20,000 task-clustered draws).
- **Secondary:** the raw margin and token correctness.
- **Frozen task-influence sensitivity analysis.**
- **Tool:** `tools/rebuttal_p29.py`, with FROZEN = {E 4, margin_std, tolerances 3e-3 / 1.5e-3}.

**Read before the result (from the control-only calibration, no donor transplant).**
- The frozen launch rule passed by a hair. Its precision criterion 1 reads 0.149 against 0.15 at the central projection of 143 eligible queries, and 0.186 at the low end (92).
- One random control alone has a resolved positive donor coefficient (+0.209). The donor-aligned half of S-ARC is not specific by itself; the "beyond the control" half carries the claim.
- Standardized control responses are heavy-tailed (up to 21 times a query's intact scale).
- The calibration assumes transplants are no more variable than controls.

The candidate result, if positive: at fixed current answer scores, transplanting retained state transfers part of a history's characteristic response to subsequent corrections, on a second task and a second architecture. A positive result would **not** establish universal applicability, generalization to unseen tasks, or improved inference (P28 already separated response transfer from usefulness).

**Expected accuracy level (goal-disclosure rule).** Not an accuracy run. For reference, the checkpoint's own numbers: its card claims 43 % pass@2 with 1,000-view voting; single view from the fixed start, 128 of the 419 ARC-1 evaluation queries are exact at depth 16, and 124 of those are already exact after the first update (`paper/code/evidence/arc/cold.jsonl`, archived). Gaussian starts (std 1) are functioning histories for this model: 1,060 of 3,352 archived random-start draws are exact at depth 16 (31.6 %; `restarts.jsonl`).

## The PI's decisions on the draft (2026-10-12), adopted

1. **Model.** The public TRM ARC checkpoint is the primary and only model. DEC-ARC D0 is deferred: its weakness (5 of 144 val-hard; one fixed point per input) would make a negative result uninterpretable.
2. **Feasibility first.** A small pilot on separate ARC tasks checks that the measurement can vary before any confirmation row. Its tasks are excluded from confirmation. Settings are chosen from intact responses only and frozen before confirmation (protocol below, written before the pilot ran).
3. **"Identical corrections" made precise** (below), with colour-to-colour corrections as the primary population and the decoded shape saved beside token accuracy.
4. **The full transplant kept as the primary test, with what it preserves stated.** A token-only transplant is a later localization test, not part of P29.
5. **Success rule tightened.** The headline transfer claim needs a positive donor-aligned coefficient for the full transfer relative to intact, *and* its advantage over the control. Both estimates are reported with intervals.
6. **Gates tightened** (below); 20,000 bootstrap draws, task-clustered.

**A justification withdrawn.** The earlier draft argued from the readout-retention run (0 of 28 complete corrected answers kept) that responses here are short-lived. That run shows only that complete corrected answers were not maintained. It does not show that every corrected token, or its downstream influence, was erased. It also intervened after 16 updates and corrected the whole answer; P29 intervenes after one update with partial corrections. It says nothing about P29's responsiveness; the pilot measures that directly.

## Model, state and update

- **Checkpoint:** alphaXiv TRM ARC-AGI-1 h3l4, `runs/field_ckpts/ckpts/trm_arc1/step_259320_arc_ag1_attn_type_h3l4`, run in its own vendored code (TRM_alphaxiv at 413f2f5) through `paper/code/adapters/arc/field_models_arc.py`; environment `runs/field_ckpts/venv` (torch 2.14), CPU, float32.
- **State:** slow state zH and fast state zL, each (916, 512): 16 puzzle-prefix positions, then the 900 answer positions (a 30 × 30 canvas).
- **One update** = the native inner recurrence: 3 H-cycles of (4 L-updates of zL from zH + input, then one update of zH from zL).
- **Readout:** the displayed token at answer position p is the argmax of `lm_head(zH[16 + p])`, a linear map of rank 12 (vocabulary: 0 padding, 1 end-of-row, 2–11 the ten colours). The decoder crops the largest top-left rectangle free of non-colour tokens.
- **Continuation:** fixed depth through the adapter's update. The halting head (which reads prefix position 0) never determines the rollout.

## Pairs

- Per query: history A = the fixed start (H_init, L_init) after one update; history B = a Gaussian start (std 1, seed `[20261015, query, 7]`) after one update.
- **Kept when** both are unsolved (decoded grid not exact) and they share at least |E| + 8 **colour-to-colour** wrong tokens W: inside the target grid (the label is a colour), both histories display a colour, and both are wrong.
- Structural errors (in-grid tokens displaying padding or end-of-row; colours displayed outside the grid) are counted, not corrected. If the pilot shows the colour-to-colour population is too small, a structural population is declared separately before confirmation, never mixed in after.

## Aligned challenges and the correction

- **Evaluation tokens E:** the first |E| tokens of a seeded permutation of W (seed `[20261015, query, 1]`). Never edited.
- **Masks:** doses 0.25, 0.5 and 0.75 of the remaining shared tokens R = W − E, four equal-size masks per dose (seeds `[20261015, query, 2, dose, mask]`). Every challenge is identical for both histories of a pair.
- **What "identical correction" means.** The two histories receive corrections at the same tokens, toward the same target colours. At each corrected token, the recipient's own displayed-colour and target-colour logits are swapped, realized by the minimum-norm change of zH (the readout's pseudoinverse). These are **aligned semantic corrections with history-specific numerical magnitudes**: the change in zH differs between the two histories because their logits differ. Correction magnitudes are recorded for both histories.
- **Within a recipient, the correction is built once** from its intact state and the identical float32 change is added to its intact, transplanted and control states. Transplants and controls preserve every logit, so the same change realizes the same target display in every condition (checked on the executed tensors).
- **Shape.** A colour-to-colour correction does not change which positions hold non-colour tokens, so it cannot change the decoded shape at the moment of the edit. Later updates can; the decoded shape is saved at every update beside token accuracy.

## Conditions (both directions; each with its own no-edit continuation and the 12 masks)

| condition | the recipient's state |
|---|---|
| intact | unchanged |
| full | zH: at all 900 answer positions (padding included), the donor's readout-invisible component (the 500-dimensional nullspace of lm_head) replaces the recipient's, so all 12 logits at every position are preserved; at the 16 prefix positions, the donor's whole vectors. zL, the inputs and the task embedding remain the recipient's (the two histories are of the same query) |
| full_ctrl | per answer token, the recipient's invisible component rotated inside the nullspace to the donor's angle from it and scaled to the donor's norm; per prefix token, the recipient's vector rotated likewise in the full space. Drawn once per pair-direction (seed `[20261015, query, 6, direction]`) and reused for the no-edit arm and every mask |

A positive result concerns the transplanted slow state **collectively, including its prefix**: the prefix is invisible to the answer readout but is part of the retained computation.

**Exact restoration:** after every other variant of a block is built, the saved recipient tensors are written back and rolled (the no-edit arm and one mask per dose); all four updates must reproduce the intact rollouts bitwise (displays and evaluation logits).

## The primary statistic and rules

**Statistic:** P27's all-four-mask centered donor coefficient at k = 1 (one update after the change), on E. Within each (state, dose) the four masks' outcomes are centered; the centered difference (Y_X − Y_ref) is regressed on the centered donor intact outcomes and the centered recipient intact outcomes (OLS); β(X) is the donor coefficient with ref = intact. The outcome Y and |E| are frozen by the pilot (protocol below).

**Intervals:** 95 % bootstrap, **20,000 draws, clustered by task** (both directions, every challenge and every query of a task together); joint quantities bootstrapped jointly.

**S-ARC (headline; both parts required):**
- **donor-aligned transfer:** β(full) has a lower bound > 0, and
- **beyond the control:** β(full) − β(full_ctrl) has a lower bound > 0.

Both estimates are reported with intervals under every outcome. A positive difference with β(full) unresolved is reported as "beats the control, donor alignment unresolved": it could arise from the control moving responses away from the donor.

**Secondary (reported, no letters):** β(full_ctrl); P26's R1 and BASELINE; k = 2–4; the correct-token margin (if correctness is primary) or correctness (if the margin is); decoded-shape correctness by condition; transplant and correction magnitudes.

**Credences** (to be restated after the pilot, before confirmation): S-ARC 0.30; β(full) resolved 0.40; the difference resolved 0.35.

## Gates (per block; any failure stops the run)

- **Native recurrence:** the adapter's update equals the native `TinyRecursiveReasoningModel_ACTV1_Inner.forward` from identical states over four updates, bitwise, for fixed-start, Gaussian-start and transplanted states (zH, zL and logits).
- **Batch invariance:** a row rolled alone equals the same row inside a batch, bitwise.
- **The decoder:** our crop equals TRM's `_crop_np` on every screened display.
- **Corrections on the executed float32 tensors**, in every condition: the designed display at the masked tokens, every other position's logits bitwise unchanged, and the realized logits within a tolerance declared from the pilot.
- **Transplants and controls on the executed float32 tensors:** zero displayed-token changes at all 900 positions; logit deviation within a tolerance declared from the pilot; the full prefix equal to the donor's; control norms and angles matched (float64) and inside the nullspace.
- **Restoration** bitwise over the multistep continuation.
- The control draw is made once per pair-direction and reused.
- Source hashes (adapter, vendored model code, data, checkpoint) unchanged from start to end.

## Feasibility pilot (protocol written before the pilot ran)

**Tasks.** A frozen seeded order of the 400 task names (seed `[20261015, 29, 0]`); the first 48 tasks are the pilot. All their queries are excluded from confirmation; the confirmation population is every query of the remaining 352 tasks.

**What runs.** For each candidate |E| ∈ {4, 8, 16}, its own design (eligibility |W| ≥ |E| + 8; E and masks drawn as above) on the pilot's eligible queries: **intact rollouts only** (no-edit and 12 masks, both histories, four updates). Transplant and control states are constructed and gated but not rolled out, so no transplant effect is observed. Timing is measured.

**What is read (intact responses only).**
- Eligibility: queries screened, both unsolved, shared colour-to-colour wrong tokens, the same count for any display type, structural tokens and decoded shapes.
- Variation: the share of (state, dose) units whose four masks give different outcomes; the R² of the donor's centered responses on the recipient's (their coefficients are separable only if the donor has variation beyond the recipient's).
- A precision proxy for the donor coefficient: SE ≈ √(2·mean(c_R²) / Σ r_D²), with r_D the donor's centered responses residualized on the recipient's. Calibrated on P27's intact rows: the proxy's noise term matches the actual dependent's variance within 0.82–0.92, and the bootstrap SE is 1.15–1.53 times the proxy. The projection uses a 1.5 clustering factor and the projected confirmation size; MDE = 2.8 × projected SE (80 % power, two-sided 5 %). For reference, P27's realized four-mask SEs were 0.032 (Attention 192) and 0.027 (256); varying shares 0.46 and 0.59; R² 0.006 and 0.002.
- Correction magnitudes in both histories; transplant magnitudes; CPU cost.

**Decision rules (frozen now; rules 2–3 rely on a precision proxy the PI's second review withdrew, so they are not applied automatically — see "Pilot outcome").**
1. **Fixed by design, not chosen:** one update before the change; doses; four masks; R_MIN = 8; colour-to-colour population; k = 1 primary.
2. **|E|:** the candidate with the smallest projected MDE for token correctness at k = 1 (within 0.005, the smaller |E|).
3. **Primary outcome:** token correctness at k = 1 if, at the chosen |E|, the varying share is ≥ 0.25 and the projected MDE is ≤ 0.15. Otherwise the correct-token margin at k = 1 becomes primary (its |E| chosen by rule 2 applied to margins) if its projected MDE is ≤ 0.15, with correctness secondary; the registration records the switch and the reason. Otherwise the test is **uninformative at this design**: no confirmation runs, and the revision (for example a later intervention depth, another pair construction, or a declared structural population) goes to the PI. A flat measurement yields "uninformative test", never a conclusion that ARC lacks the mechanism.
4. **Population:** every eligible confirmation query, if the projected wall from the measured cost is ≤ 4 h; otherwise the first eligible queries in the frozen order that fit 4 h, with the MDE recomputed (and put to the PI if it exceeds 0.15).
5. **Tolerances** (superseded by the PI's second review; see below): the executed float32 logit tolerances for corrections and transplants are 10 times the pilot's maxima, rounded up to a power of ten. A pair-direction whose executed transplant or control would change any displayed token is excluded from all conditions and counted, rather than stopping the run.

## Pilot outcome (2026-10-12; intact responses only — no transplant or control state was rolled out)

Files: `runs/analysis/rebuttal_20261015/pilot_report.{txt,json}` (the tool's report, run after the repairs below); `pilot_concentration.py` and `bracket_calibration_p27.py` there (exploratory, intact-only). The pilot ran with the tool at sha256 795bc8ab… (blocks carry it); the reader repairs below touch no rollout code.

**Gates (all three designs, 21 blocks): all passed.**
- The native recurrence was bitwise over four updates from fixed-start, Gaussian-start and transplanted states. A row alone equals the same row inside a batch, bitwise.
- Restoration was bitwise over the multistep continuation. The crop matched TRM's `_crop_np` on every display.
- Corrections: the designed display in every condition, other positions' logits bitwise unchanged; realized-logit deviation ≤ 7.1e-4.
- Transplants and controls: zero displayed-token changes at all 900 positions; executed logit deviation ≤ 1.46e-3 (full) and 1.59e-3 (control). Controls' norms and angles matched to ≤ 5e-15, inside the nullspace to ≤ 8e-14.
- **The smallest top-two logit gap over the pilot is 3.6e-4** (the smoke's 0.020 does not describe the pilot). No token flipped, but a flip is possible in confirmation; the registered exclusion rule covers it.

**Eligibility** (49 queries from the 48 pilot tasks):
- 12 queries solved after one update from each start; 36 both unsolved.
- Shared colour-to-colour wrong tokens among those: median 12, quartiles 6 / 23 (any display: median 14).
- Eligible queries:

| \|E\| | eligible (colour-to-colour) | any display type | projected over 370 confirmation queries (task-bootstrap 95 %) |
|---|---|---|---|
| 4 | 19 | 22 | 143 [92, 193] |
| 8 | 14 | 16 | 106 [62, 152] |
| 16 | 9 | 11 | 68 [31, 109] |

The structural population is barely larger; colour-to-colour stays the population.

**Where the responses are (k = 1).**

| measure | \|E\| | groups varying across masks | tasks with residual donor information | largest task's share | effective tasks |
|---|---|---|---|---|---|
| correctness | 4 | 7.0 % | 5 of 19 | 0.606 | 2.2 |
| correctness | 8 | 14.3 % | 5 of 14 | 0.910 | 1.2 |
| correctness | 16 | 20.4 % | 5 of 9 | 0.347 | 3.7 |
| margin | 4 | 100 % | 19 of 19 | 0.545 | 2.6 |
| margin | 8 | 100 % | 14 of 14 | 0.912 | 1.2 |
| margin | 16 | 100 % | 9 of 9 | 0.757 | 1.7 |

- **Token correctness fails the registered variation requirement at every |E|** (7–20 % against 25 %). At these errors TRM is very confident: the mean evaluation-token margin without an edit is about −110 logits at |E| = 4.
- **Margins respond everywhere, but the raw-logit information is concentrated by scale.** The two dominant tasks (1c02dbbe, cb227835) move their evaluation margins about 20 times more than the median task (response RMS 47 and 43 logits against a median of 2.5). The same tasks carry most of the correctness information.
- Correction magnitudes: ‖δ‖/‖h‖ per corrected token is 0.69 in both histories; for the same mask, the fixed/Gaussian norm ratio has median 1.01 (quartiles 0.81 / 1.38). Transplant: ‖Δ‖/‖h‖ 0.30 at answer positions, 0.49 at the prefix.

**The precision proxy is withdrawn as a decision tool (the PI's review).**
- The Sudoku-calibrated proxy projected MDE ≈ 0.10 for correctness and 0.11 for margins at |E| = 4, although correctness varies in 7 % of groups.
- With both directions included, the donor and recipient columns have equal total variance, so the proxy is nearly scale-free. It reflects the row count and the donor–recipient correlation. It reads 0.066 at every horizon, and 0.068 even with the task holding 61 % of the information omitted.
- **A concentration-aware set of assumed noise scenarios** replaces it (scenarios, not statistical bounds). It uses Var(β) = Σ_tasks (Σ_rows r_D e)² / (Σ r_D²)², with noise variance twice the pair's local response variance, independent (the optimistic scenario) or perfectly correlated (the pessimistic scenario) within a task.
  - On P27's intact rows, the optimistic scenario reproduces P27's realized SEs: 0.0315 against 0.032 on 192; 0.032 against 0.027–0.029 on 256. Sudoku had about 110–120 effective puzzles. The pessimistic scenario is about 3.4 times too large there.
  - Projection for the confirmation under the optimistic scenario (MDE = 2.8 SE, per coefficient): correctness |E| = 4 about 0.25; margin |E| = 4 about 0.24; correctness |E| = 16 about 0.18. Across the eligibility interval the margin |E| = 4 figure spans about 0.20–0.29.
  - Sudoku's transfer effects were 0.08–0.23.
- **The headline is a conjunction of two lower bounds;** it does not inherit even the single-coefficient power.

**What the pre-stated rules would have done, and why they are not applied automatically.** Rule 3 fails correctness at every |E| and falls back to margins. By the withdrawn proxy, margins at |E| = 4 would have "passed" (0.106 ≤ 0.15), and confirmation would have been launched. Under the optimistic noise scenario they do not (≈ 0.24). The PI's review stopped the automatic trigger; the choice below is the PI's.

## The PI's second review (2026-10-12), adopted

1. **The flat-response bug is fixed.** Flat responses, or donor responses with no variation beyond the recipient's, now return UNINFORMATIVE, never zero uncertainty. The selftest reproduces the PI's constant input and also tests collinear and single-task cases.
2. **Rank-deficient bootstrap draws are declared.** A draw in which a donor coefficient is not identified (its residual after the recipient's column and the intercept ≤ 1e-9 of its own sum of squares) is counted. It sits outside *both* ends of the interval, so it can only widen it. It is neither dropped nor set to zero. A point estimate that is not identified reads NOT-RESOLVED (uninformative).
3. **Precision is a planning heuristic.** The task concentration, the leave-one-task-out precision and the eligibility interval are reported above.
4. **A task-influence sensitivity analysis is frozen for confirmation.** Two analyses, both with 20,000 draws:
   - the primary contrasts recomputed without the task holding the largest share of the *intact-only* residual donor information, with its letter;
   - the leave-one-task-out ranges of β(full) and β(full) − β(full_ctrl).

   No task is discarded from the primary analysis.
5. **Tighter tolerances**, replacing rule 5's "10 times, rounded up to a power of ten" (which would have allowed 0.1):
   - executed transplant and control logit deviation ≤ **3e-3** (pilot maximum 1.59e-3);
   - realized correction-logit deviation ≤ **1.5e-3** (pilot maximum 7.1e-4).

   For scale, 1.59e-3 is 8e-4 of the median across-mask SD of the intact evaluation margin (1.93 logits) and 4e-3 of its 10th percentile. Any displayed-token change excludes the query (both directions) and is counted.
6. **Two achievements kept distinct.**
   - Under a margin primary, a positive P29 establishes: "Retained state transfers history-specific changes in confidence produced by subsequent corrections."
   - Only a replicated effect on token correctness would support the stronger "transfers which answers get corrected".

## Decision owed to the PI (nothing launched; superseded by the PI's third review below)

Model, pairs, colour-only challenges and the held-out confirmation tasks stay as they are. The open choice is the primary measure, and therefore whose responses the coefficient averages over:

- **(A) Raw correct-token margin, |E| = 4, k = 1** (the registered fallback).
  - It weights tasks by their logit response scale: two or three tasks carry most of the information (about 20 effective tasks projected in confirmation).
  - Calibrated MDE about 0.24 (0.20–0.29).
  - A null result is uninformative below that size; a positive one is a statement driven by a minority of tasks (the task-influence analysis will show how much).
- **(B) The margin standardized per query** (both histories and every condition divided by the query's pooled RMS of centered intact responses, fixed before any transplant outcome), |E| = 4, k = 1.
  - It equalizes each query's intact response energy (not exactly an equal-weight average of task-specific effects). In the pilot this spreads the *predictors'* information over 18.3 effective tasks of 19 (largest share 0.07); the projected MDE of about 0.08 assumed that intervention variability follows intact variability, which the PI's third review showed fails on Sudoku (below). It is withdrawn as a launch justification.
  - **This changes the estimand**, from logit-scale responses to an equal-weight average of mostly small confidence changes (median response RMS 2.5 logits on margins near −110). It is not a cure for weak responses.
  - Raw margin and correctness would be registered secondaries.
  - The PI cautioned against changing the primary statistic to improve diagnostics; this option is offered only because the task-generality question is about tasks, not about logit scale.
- **(C) Revise the design before confirmation** (for example the intervention depth) with a second intact-only pilot round on the same pilot tasks.
- **(D) Stop:** record P29 as weakly informative at this checkpoint (informative discrete responses in 5 of 19 eligible tasks).

**Recommendation: (B), with (A) and correctness as registered secondaries** and the task-influence analysis on all three. The question is whether the property appears across ARC tasks, and only (B) can answer it at this size. (A) mostly asks it of two or three tasks.

**Cost**, either way, at |E| = 4:
- 143 [92, 193] projected eligible queries, about 12,300 variants (43 per pair-direction).
- Measured throughput: 0.70–0.75 s per variant with one process; 1.30–1.39 s per process with three concurrent processes (about 2.2 variants/s in aggregate).
- About 1.6 h (1.0–2.1 h) on three processes.
- AC power and an open lid are needed.

## The PI's third review (2026-10-12): option B, conditional on a control-only precision calibration

**The PI's decision.** Keep the model, pairs, colour-only challenges and held-out confirmation tasks. Option B (the per-query standardized margin, |E| = 4, k = 1) is the candidate, but its projected 0.08 is not a launch justification. A short control-only calibration on the existing pilot queries decides. If it supports a useful test, B launches with four evaluation tokens and k = 1. If not, no several-hour confirmation runs under the claim that the 0.15 precision target was met.

**Why (the PI's exploratory check on P27, reproduced below before the calibration).** The standardization describes the *predictors*. Precision also depends on how variable the transplanted and control responses become. A query with small intact responses and large post-intervention responses is amplified by its small intact scale. On P27's saved scores, with the proposed normalization and 20,000 puzzle-bootstrap draws:
- the noise model predicts SE 0.029 (192) and 0.031 (256);
- the measured clustered SEs are 0.097 and 0.092, about 3.4 and 3.0 times larger;
- the normalized full − control effect is +0.294 [0.095, 0.477] on 192 and +0.131 [−0.051, 0.309] on 256.

These are new exploratory analyses, not replacements for P27's registered findings. The raw-margin contrast is resolved on both widths.

**Wording for B.**
- It equalizes each query's intact response energy. It does not make the pooled coefficient exactly an equal-weight average of task-specific effects.
- A positive result establishes transfer of **correction-induced logit-margin responses**. It does not by itself establish transfer of corrected answers or of calibrated confidence.

**The normalization (frozen before the calibration).**
- One scale per query: s_q = the RMS of the 24 centered intact responses of both histories (3 doses × 4 masks × 2 directions; centered within each history and dose, as in the statistic). The response is the mean correct-token margin over E at k = 1.
- Every condition and both directions of the query are divided by s_q. The same k = 1 scale is used at every horizon.
- **Small scales:** a query with s_q < 0.01 logits is excluded from the standardized analysis and counted. At that scale the executed float32 tolerance (3e-3) would be at least 30 % of the scale.
- Raw margin and correctness stay as secondary outcomes.

**The calibration (protocol frozen before any calibration row).**
- **Population:** the 19 eligible pilot queries at |E| = 4 (38 pair-directions), with their existing E and masks. No confirmation query is touched.
- **Conditions:**
  - intact, re-rolled; it must reproduce the pilot's intact rollouts bitwise (a determinism gate);
  - full_ctrl, the registered matched control (seed `[20261015, query, 6, direction]`);
  - ctrl_b, an independent draw of the same construction (seed `[20261015, query, 16, direction]`).

  Each control is drawn once per pair-direction and reused for the no-edit arm and all 12 masks. **The genuine donor transplant is not rolled out**, so no donor-transfer effect is observed. Every gate of the confirmation applies, with the tightened tolerances.
- **Measured (standardized margin, k = 1):**
  - β(full_ctrl) and β(ctrl_b) relative to intact (the noise of the donor-aligned part β(full));
  - β(full_ctrl) − β(ctrl_b) (the noise of the "beyond the control" part β(full) − β(full_ctrl)).

  Both use task-clustered bootstrap intervals, 20,000 draws (SE = interval width / 3.92), and task-jackknife SEs.
- **Also reported, not used by the rule:**
  - task influence on the control-versus-control contrast (the largest task's share of the jackknife influence, and the contrast with that task omitted);
  - the distribution over pair-directions of the control responses' RMS relative to s_q (median, 90th percentile, maximum);
  - the same quantities for the raw margin and for correctness.
- **Projection:** SE_conf = SE_pilot × √(19 / n), with n = 143 eligible confirmation queries (task-bootstrap interval 92–193); MDE = 2.8 × SE_conf for a single coefficient.
- **Stated assumptions:**
  - genuine transplants are no more variable than matched controls (the calibration cannot measure the transplant);
  - confirmation queries resemble the pilot queries;
  - precision scales with the square root of the number of queries;
  - a 19-query SE is itself uncertain, so the larger of the bootstrap and jackknife SEs is used.

  The conjunction of two lower bounds has less power than either coefficient alone; no joint power is claimed.

**Launch rule (frozen before reading the calibration).** Launch B (|E| = 4, k = 1, standardized margin) on the confirmation population if and only if all of the following hold:
1. At n = 143, MDE ≤ 0.15 for the control-versus-control contrast, using the larger of its bootstrap and jackknife SEs.
2. At n = 143, MDE ≤ 0.15 for the larger of the two control-versus-intact coefficients' SEs.
3. The control-versus-control contrast is itself not resolved (its 95 % interval contains 0). Two draws of the same construction should be exchangeable; a resolved difference would make the noise model suspect.
4. Fewer than 20 % of the pilot queries are excluded for small scales.
5. Every gate passes, including the bitwise reproduction of the pilot's intact rollouts.

Otherwise nothing launches and the result goes to the PI. The MDE at 92 and 193 queries is reported beside the central figure either way.

## Calibration outcome (2026-10-12; `runs/analysis/rebuttal_20261015/report_calib.{txt,json}`; no donor transplant rolled out)

**Gates.**
- The re-rolled intact rollouts reproduce the pilot's bitwise (494 variants).
- No query was excluded for flips.
- Executed transplant and control logit deviation ≤ 1.71e-3 (the second draw included; cap 3e-3).
- Every other gate passed in all 7 blocks.

**Option B (standardized margin, k = 1; 19 queries; none excluded for small scale).**
- β(full_ctrl) = +0.060 [−0.128, +0.225]; SE 0.090 (bootstrap and jackknife alike).
- β(ctrl_b) = **+0.209 [+0.047, +0.439]**; SE 0.100 (bootstrap), 0.112 (jackknife).
- Control − control = −0.149 [−0.428, +0.114]; SE 0.138 (bootstrap), 0.146 (jackknife).
- Projected MDE at 143 / 92 / 193 queries:
  - control vs intact: 0.114 / 0.143 / 0.099;
  - control vs control: **0.149** / 0.186 / 0.129.
- **Task influence:** da515329 holds 0.62 of the jackknife influence on the control-versus-control contrast. Without it the contrast is −0.031 [−0.234, +0.168].
- **Centered response RMS per pair-direction, in units of the query's scale:**

| condition | median | 90th percentile | max |
|---|---|---|---|
| intact | 1.00 | 1.38 | 1.41 |
| full_ctrl | 0.93 | 1.36 | **20.95** |
| ctrl_b | 0.88 | 1.66 | **13.09** |

This is the failure mode the PI predicted: small intact scales amplify large intervention responses.

**Raw margin (secondary).**
- Control − control +0.201 [−0.081, +0.603]; MDE 0.230 at 143 queries.
- One task (cb227835) holds 0.87 of the influence.

**Correctness (secondary).**
- Control − control +0.167 [0.000, +0.524]; MDE 0.297 at 143 queries.
- Most groups do not vary.

**The frozen launch rule:** all five criteria hold (criterion 1 at 0.149 ≤ 0.15) → **LAUNCH B**.

**What this means for the reading.**
- The confirmation is powered, under the stated assumptions, for single-coefficient effects of about 0.15 at the central projection. It is not powered at the low end of the eligibility interval, and it carries no joint-power guarantee for the conjunction.
- A random matched perturbation can produce a resolved positive donor coefficient on this checkpoint, so β(full) > 0 alone means little. The registered headline needs the difference from the control as well.
- The confirmation draws one control per pair-direction; the calibration shows two draws can differ by about 0.15 at this sample size.
- A null result is "not resolved at this precision", never "absent".

## Registered: credences and compute (before any confirmation row)

| event (option B, k = 1) | credence |
|---|---|
| S-ARC (both lower bounds > 0) | 0.20 |
| β(full) resolved > 0 | 0.40 |
| β(full) − β(full_ctrl) resolved > 0 | 0.25 |
| given S-ARC, it survives omitting the top-information task | 0.50 |

**Compute.** About 143 eligible queries (92–193) × 2 directions × 43 variants ≈ 12,300 variants on three processes. The measured aggregate throughput is about 2.2 variants per second, so roughly 1.6 h (1.0–2.1 h). Blocks of 8 queries are checkpointed and resumable. AC power is attached.

**Wording under each outcome.**
- **S-ARC:** "On ARC, at fixed current answer scores, transplanting a history's retained slow state transfers part of its correction-induced logit-margin responses beyond a matched random change." Add the task-influence reading. This is not transfer of corrected answers and not of calibrated confidence.
- **Only the difference from the control resolved:** "beats the control, donor alignment unresolved."
- **Only β(full) resolved:** "donor-aligned but not beyond a matched random change". On this checkpoint, random changes can do that too.
- **Neither resolved:** "not resolved at this precision (single-coefficient MDE ≈ 0.15 at the realized size)". This does not say ARC lacks the property.

## P29 Outcome (2026-10-12; `runs/analysis/rebuttal_20261015/report_conf.{txt,json}`; every gate passed; recount exact)

**Run.** Launched 06:30Z (12:00 IST) by the system clock; the registration entry's "~09:10Z" label was wrong, see below. Three processes; 13,731 s of rollout time; 10,148 variants.
- All 370 confirmation queries were screened; 254 were unsolved after one update from both starts.
- **118 queries were eligible, one per task (118 tasks, 236 pair-directions).** That is inside the projection interval [92, 193], below the central 143.
- Gates (native recurrence and batch invariance on each shard's first block; restoration and the intervention checks on every applicable variant — wording corrected 2026-10-13):
  - the native recurrence and batch invariance were bitwise, and restoration was bitwise over four updates;
  - zero displayed-token changes under transplants or controls, and no query excluded;
  - executed logit deviation ≤ 2.2e-3 for transplants and controls (cap 3e-3) and ≤ 1.19e-3 for corrections (cap 1.5e-3); the smallest top-two gap was 2.7e-4;
  - no small-scale exclusions, and no unidentified bootstrap draws.
- **Two reader-side fixes after the run; neither touches a rollout or a statistic.**
  1. `merge` now drops empty blocks before concatenation. Block 35 had no eligible query, and its placeholder evaluation-token array was 16 wide instead of 4. The earlier merged files' hashes are unchanged.
  2. `load` now reads every array into memory once. The lazy npz access re-decompressed the display array on every variant access, and the first reader run, which also lacked `caffeinate` while the Mac idle-slept on battery, stalled for hours.

  The independent recount (`recount.py`, own code) reproduces every number below.

**The registered result (standardized margin, |E| = 4, k = 1; 20,000 task-clustered draws).**

| quantity | estimate | 95 % interval |
|---|---|---|
| β(full), donor-aligned | **+0.354** | [+0.251, +0.488] |
| β(full_ctrl), the matched control | +0.171 | [+0.073, +0.276] |
| β(full) − β(full_ctrl), beyond the control | **+0.183** | [+0.017, +0.389] |

**→ S-ARC.** Both lower bounds are above zero. The second lower bound is thin.
- In four other 20,000-draw seeds it reads +0.015 to +0.018, with 1.4–1.7 % of draws at or below zero (`seed_check.py`, exploratory).
- The recount's own resampler gives [+0.018, +0.388].

**Task influence (registered).**
- The information is spread: all 118 tasks carry residual donor information, effectively 110.4 tasks. The top task (a934301b) holds 0.014.
- Without that task the result still reads S-ARC: β(full) +0.358 [+0.252, +0.493], difference +0.182 [+0.012, +0.391].
- Leave-one-task-out ranges: β(full) [+0.305, +0.370]; difference [+0.108, +0.225].

**Secondary (registered, no letters).**
- BASELINE (full − control slope): +0.318 [+0.126, +0.781]. R1: +0.309 [+0.129, +0.470].
- **Raw margin:** β(full) +0.136 [+0.035, +0.371], control +0.117; difference **+0.019 [−0.060, +0.182], not resolved**. The logit-scale-weighted average, dominated by the few tasks with very large logit responses, shows no transfer beyond the control. The standardized result lives in the typical query.
- **Token correctness:** β(full) +0.142 [+0.042, +0.280], control +0.037; difference **+0.105 [+0.007, +0.248]**. This is resolved at k = 1, but it is a secondary, and most groups do not vary. It needs its own replication before any claim that transfer extends to which answers get corrected.
- **Horizons, standardized margin, difference from the control:** k = 2 +0.194 [+0.058, +0.255]; k = 3 +0.146 [+0.018, +0.288]; k = 4 +0.085 [−0.060, +0.271], not resolved. The raw margin and correctness are unresolved at k = 2–4.
- The decoded shape is correct for 85.6 % of variants at k = 1 in every condition. Exact answers are ≤ 0.8 %.
- Magnitudes: transplant ‖Δ‖/‖h‖ 0.29 at answer positions and 0.55 at the prefix. The correction ‖δ‖/‖h‖ is 0.73 in both histories.

**Predictions.**

| event | credence | outcome |
|---|---|---|
| S-ARC | 0.20 | yes |
| β(full) resolved > 0 | 0.40 | yes |
| difference resolved > 0 | 0.25 | yes |
| S-ARC survives omitting the top-information task | 0.50 | yes |

Brier score 0.453. The transfer was under-predicted, as on Sudoku P26.

**Reading.**
- **Registered wording:** "On ARC, at fixed current answer scores, transplanting a history's retained slow state transfers part of its correction-induced logit-margin responses beyond a matched random change."
- It is spread across tasks, not carried by one, and persists for two to three updates on the standardized margin.
- It is **not** transfer of corrected answers: correctness is only a resolved secondary. It is **not** transfer of calibrated confidence.
- **What limits it.**
  - A random retained-state change of the same geometry already moves responses toward the donor's (+0.171). The donor-specific part is the difference, +0.18, whose lower bound sits near zero.
  - The weighting matters: under the raw logit scale the transfer beyond the control is not resolved.
  - The confirmation drew one control per pair-direction. The calibration showed two draws can differ, so part of the interval's width is control-draw noise.
- **With Sudoku P26/P27:** the same property, that retained state carries a transferable, history-specific response to subsequent corrections, now appears on a second task and a second architecture (TRM on ARC-AGI-1). This holds for the response measure registered here; Sudoku's basis or coordinates were not used.

**Clock correction.** The ledger entry for this registration is labelled "2026-10-12 ~09:10Z". The commit is 18c010d at 06:30Z (12:00 IST); my time labels ran about three hours ahead.

## P29 SCOPE ADDENDUM after the PI’s review (2026-10-12 ~17:50Z; every number reproduced; exploratory; `runs/analysis/rebuttal_20261015/review_check.py`)

**What stands.** The registered decision stands: S-ARC, with β(full) − β(full_ctrl) = +0.183 [+0.017, +0.389]. The provenance holds:
- the registered runner was used;
- the pilot tasks were excluded;
- all 370 queries were screened and 118 retained;
- the gates pass;
- the two later code changes touch only loading and merging.

The PI's independent bootstrap gives about [+0.014, +0.388]. The positive lower bound is reproducible, though close to zero.

**Corrected: predictor information is not outcome influence.**
- "About 110 effective tasks" measures how broadly the *intact donor-response predictors* vary. It does not measure how evenly tasks contribute to the estimated transplant effect or to its uncertainty.
- **One task, b9630600, holds 0.62 of the jackknife variance** of the full − control contrast.
  - Omitting it moves the contrast from +0.183 to +0.108 [−0.015, +0.213].
  - Every single-task omission leaves a positive point estimate (range +0.108 to +0.225).
  - **The effect's direction is more stable than its statistical significance.**
- The registered omission removed a different task, chosen by predictor information before any outcome, and passes as reported. The b9630600 omission was chosen after seeing outcomes. It is a sensitivity analysis: it neither replaces the primary test nor justifies discarding the task.
- On b9630600, responses after intervention reach **30.1 times** the query's intact scale for the control (one direction) and 14.9 times for the full transplant (the other direction); the intact responses are 0.3–1.3. This is the failure mode the calibration anticipated: standardization balances the predictors and leaves large intervention responses in place.
- **Withdrawn:** "spread across tasks, not carried by one". **Supported:** predictor information is broadly distributed, while significance remains sensitive to individual-task influence.

**Two interpretations tightened.**
- **Raw margin.** "Not resolved" is not "nothing transfers": the interval [−0.060, +0.182] allows a positive effect. Disagreement between the raw and standardized analyses does not establish that the effect "lives in the typical query". **That phrase is withdrawn.**
- **Persistence.** The standardized contrast is positive with intervals excluding zero at k = 2 and 3. **Say "detected through the third update; unresolved at the fourth."** Where the effect ends is not established.

**No repair benefit** (challenged arms, mean evaluation-token correctness at k = 1): intact 9.80 %, full transplant 9.42 %, matched control 10.58 %. Response transfer is not better repair.

**Correctness secondary.** +0.105 [+0.007, +0.248] is encouraging. Only 6.2 % of history–dose groups (44 of 708) vary in correctness across masks, and its information is much less broadly distributed than the margin's. It needs confirmation.

**The mechanism this supports.** The intervention preserves the recipient's current answer logits to numerical tolerance, and the evaluation tokens are never edited. Even so, replacing retained state changes *which equal-sized corrections elsewhere produce donor-like responses*, beyond a matched random change:

> Retained state configures how subsequent corrections propagate, and part of that configuration can be transferred between histories.

P29 supplies evidence for this in TRM on ARC, alongside the Sudoku findings (P26/P27). The generalizable contribution is the intervention method together with the observed transfer of correction responses. **Not tested on ARC:** compactness, represented content, and a rule for choosing beneficial configurations.

**Candidate manuscript wording (the PI's):** "On 118 eligible ARC-AGI-1 evaluation queries, answer-logit-preserving slow-state transplants in a public TRM checkpoint transferred correction-specific responses beyond matched random perturbations under a preregistered, query-normalized margin analysis (Δβ = 0.183; 95% task-bootstrap CI [0.017, 0.389]). This extends the response-transfer finding to a second task and architecture, although statistical significance was sensitive to individual-task influence."
- The appendix carries the raw-margin analysis, the correctness secondary, the later horizons, and the distinction between predictor concentration and outcome influence.
- It is a defensible positive addition, **not** a robust improvement in reasoning performance.

**Next (the PI's decisions).**
1. **One registered replication of the primary finding, with correctness explicitly included** (P31, drafted separately). It uses fresh history and mask seeds, averages a fixed small number of independent matched controls, and predeclares the task-influence analysis and the roles of the two outcomes. Reusing the ARC task set tests reproducibility across histories and challenges, not generalization to new tasks.
2. Correctness alone is not the next headline; both outcomes are measured in the same runs.
3. **P30 proceeds as a bounded analysis of the existing Sudoku bank:** can information available before continuation identify a beneficial intervention? It keeps the equally informed control and restart comparisons and the compute accounting. P29 motivates it but gives no evidence that selection will work.

## Scope (under every outcome)

- Oracle-guided diagnostic corrections through the readout's pseudoinverse: not a validated semantic inverse.
- Constructed hybrids of two histories of the same query.
- One public checkpoint, single view, no voting; fixed-depth continuation.
- A one-update primary horizon.
- Colour-to-colour corrections only.

## Staged alongside: P30 as a bounded exploratory analysis

- **Question:** do observable features distinguish a beneficial choice *within* a puzzle, rather than identifying easier puzzles?
- **No new outcome experiment; bounded replay for features.** Regenerating P28's iteration-16 states and probing restarts both require forward computation. The replay is deterministic from the banks and adds no new outcome.
- **Features:** solution-free, available before continuation (each history's constraint violations, score margins and recent display changes; each pair's display disagreement, ‖Δ‖ and energy in U).
- **Charging for probing.** Any selection comparison charges the model work of probing every candidate inspected, restart prefixes included, against equal-work alternatives.
- **Comparators:** equally informed selection among the matched controls and among restarts.
- **Evaluation:** feature selection and tuning only inside puzzle-grouped training folds; all candidates of a puzzle kept together.
- **What a signal would justify:** a fresh, registered confirmation, not a performance claim.

## P28 reporting qualification (and its extension)

The P28 addendum's executed float32 audit (score change ≤ 7.4e-7 on 192 and 6.9e-7 on 256, no flips) covered two eligible puzzles per receiver (12 pairs × 4 arms). Its maxima were the maxima observed in that audit, not a bound over the run.

**Extended to every eligible puzzle** (exploratory; `runs/analysis/rebuttal_20261014/float32_check_all.{py,json}`): 128 puzzles × 12 ordered pairs × 4 arms per receiver.
- Attention 192: maximum 8.9e-7 (lead_ctrl). Attention 256: maximum 8.5e-7 (full).
- Zero displayed-digit flips on either receiver.

These replace the two-puzzle figures. The ledger's unscoped "≤ 7.4e-7" is corrected in the P29 registration entry.
