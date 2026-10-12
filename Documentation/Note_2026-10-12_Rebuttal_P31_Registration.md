# Note 2026-10-12 — P31: a registered replication of P29's response transfer on ARC, with correctness included (registration, before any P31 row; the PI's decisions incorporated)

**Goal (page one).** A MEASUREMENT: does P29's result reproduce across fresh histories and fresh challenges?
- P29's result: on the public TRM ARC-AGI-1 checkpoint, an answer-logit-preserving slow-state transplant transfers correction responses beyond a matched random change (standardized margin, Δβ = +0.183 [+0.017, +0.389]).
- Same model, tool and measurement. Inference only, on the Mac's CPU.
- Not an accuracy run: P29 showed no repair benefit (9.80 / 9.42 / 10.58 %).
- It reuses P29's 352 confirmation tasks. The fixed-start state is unchanged. It therefore tests **reproducibility across new pairings, challenges and controls, not generalization to new tasks**.
- **Tool:** `tools/rebuttal_p31.py`, over P29's runner (seed-parametrized; independent controls). **Outputs:** `runs/analysis/rebuttal_20261017/`.
  - The merged file keeps the runner's name `p29_conf.npz`.
  - The P29 runner's task split now uses its own fixed seed, so the 352 tasks are identical to P29's.

## What changes from P29 (the PI's decisions of 2026-10-12, both rounds)

1. **Fresh history and mask seeds.**
   - History B is a new Gaussian draw: seed `[20261016, query, 7]`.
   - New evaluation tokens and masks: seeds `[20261016, query, 1]` and `[20261016, query, 2, dose, mask]`.
   - History A, the fixed start after one update, is deterministic and therefore the same state as in P29. Its *pairing*, challenges and controls are all new.
   - Eligibility is recomputed. Expect a count near P29's 118, not the same queries.
2. **Averaged controls.**
   - M = 4 independent matched controls per pair-direction (seeds `[20261016, query, 6, direction, j]`, j = 0..3). Each is drawn once and reused for the no-edit arm and all 12 masks.
   - **Each control's outcome (margin, standardized margin, correctness) is computed from its own rollout, then the four outcomes are averaged.** Logits are never averaged. Because the regression is linear in the dependent variable, β(averaged control) is the mean of the four β(control_j). All controls of a query stay in its task-bootstrap cluster.
   - Four independent draws halve the conditional SD of the control-draw component relative to one draw. They do not halve the experiment's total uncertainty.
3. **Both outcomes have predeclared roles** (fixed-sequence testing, so the two confirmatory tests together keep the 5 % error rate):
   - **Primary, R-MARGIN.** P29's S-ARC on the standardized margin at k = 1, |E| = 4: β(full) > 0 **and** β(full) − β(control) > 0 (both 95 % lower bounds; 20,000 task-clustered draws).
   - **Second confirmatory, R-CORRECT,** awarded **only if R-MARGIN holds.** The same two conditions on token correctness at k = 1. Correctness is reported either way; if R-MARGIN fails it carries no label.
   - **Both decisions rest on P31 alone.** Nothing is pooled with P29.
   - Correctness is coarse (6.2 % of P29's groups varied); a null R-CORRECT reads "not resolved at this precision", not "absent".
4. **A predeclared task-influence analysis** (sensitivity diagnostics, not part of either letter), separately for each confirmatory outcome:
   - **the selected task:** the task with the largest share of the jackknife variance of **full − averaged control** for that outcome;
   - both contrasts with it omitted, labelled **"survives the selected task omission"** or **"does not survive"**. This does not establish that every possible single-task omission would preserve significance;
   - the predictor-information omission, as in P29;
   - the **complete** leave-one-task-out ranges of β(full) and of the difference.

   The selected-omission intervals remain sensitivity diagnostics even though the selection rule is declared now. The registered wording always states influence beside significance.
5. **Heavy tails: a registered secondary, clipping followed by recentering.**
   - The original intact-derived query scale is used.
   - Each condition's standardized responses are centered within each history–dose group, then clipped at ±10. Each control is clipped separately, before the controls are averaged.
   - The statistic then recenters. The final values need not stay within ±10.
   - The report counts the responses clipped and the queries affected.
   - **The uncapped analysis determines the letters.** The clipped result cannot rescue a failed primary test. No other threshold is examined.

## Unchanged from P29

- The model, adapter and gates. The native recurrence and batch invariance must be bitwise, and restoration bitwise over four updates.
- The executed float32 caps (3e-3 for transplants and controls, 1.5e-3 for corrections). Zero displayed-token changes, or the query is excluded and counted.
- Colour-to-colour population; doses 0.25 / 0.5 / 0.75 with four masks each.
- The full transplant: the nullspace at all 900 answer positions plus the prefix.
- Corrections built once per recipient and added unchanged to every condition.
- The per-query scale; the small-scale exclusion below 0.01.
- Secondaries: the raw margin, k = 2–4, BASELINE, R1. Decoded shape and the challenged-arm repair means are reported.

## Credences (before any P31 row)

| event | credence |
|---|---|
| R-MARGIN | 0.45 |
| β(full) resolved > 0 (margin) | 0.75 |
| R-CORRECT (conditional on R-MARGIN) | 0.30 |
| survives the selected task omission (margin), given R-MARGIN | 0.45 |

The reasoning: P29's lower bound sat at +0.017. A replication at a similar size, with the control noise reduced, is near a coin flip for the conjunction.

## Compute

Per pair-direction: (2 + M) × 13 + 4 = 82 variants, against P29's 43. For about 118 queries that is about 19,400 variants; at the measured ~2.2 variants/s on three processes, about **2.5 h (1.8–3.2 h)**. M = 3 would take about 2.1 h. AC power, an open lid, and `caffeinate` for every step, readers included.

## The PI's decisions (2026-10-12), adopted; no further approval round

1. **M = 4 controls**, about 25 minutes more than M = 3.
2. **The ±10 clipping secondary is kept**, implemented as above.
3. **The primary population stays the 352 confirmation tasks.** The 48 pilot tasks helped set the measurement and the calibration. Eligibility is recomputed under the new histories, every newly eligible query is retained, and the overlap with P29's eligible queries is reported.

**Selftests** (`tools/rebuttal_p31.py --selftest`, synthetic merged files):
- planted margin and correctness transfer → R-MARGIN, then R-CORRECT;
- no margin transfer → correctness reported without a label even when planted;
- a single large control response is counted by the clipped secondary, and clipping happens before averaging.

**What a positive result would and would not mean.**
- A replicated margin result strengthens the cross-task mechanism claim. A gated correctness result would extend it to which answers get corrected.
- Neither, by itself, establishes a repair-performance gain.

## Order

- P31 runs after P30's feature replay finishes, so the two do not share the CPU.
- Before launch, the seed-parametrized runner must reproduce one P29 pilot block bitwise (a regression gate on the runner change).
- Then three shards, under `caffeinate`, with AC power and an open lid.
