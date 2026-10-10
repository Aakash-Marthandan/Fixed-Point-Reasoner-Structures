# Note 2026-10-10 — Sequence A: which explanation predicts the network's responses to corrections? (exploratory analysis plan, written before any fit)

**What this is.** This is the first step of `Note_2026-10-10_Rebuttal_Contribution_and_Predictive_Mechanism.md`: making the existing intervention banks compete between explanations.
- It is an **exploratory** analysis of already-inspected data. It selects the next experiment and is not a confirmatory test.
- Its model ladder and its reading rules are fixed here, before any model is fitted, so that the result is interpretable.
- Local, read-only. No inference and no training.
- Tool: `tools/analyze_seqA.py`. Outputs: `runs/analysis/seqA_20261010/`.

## The three explanations

| explanation | what it predicts |
|---|---|
| **Independent local correction** | A corrected cell is kept, and an untouched cell is repaired, independently of what else is corrected at the same time. |
| **Collective reconstruction** | How much compatible correction is supplied determines acceptance and repair elsewhere. Which equally sized set is corrected adds little. |
| **Coordination by configuration or state** | Which cells are corrected together matters, through their relations to the cell being predicted (visible configuration) or through retained hidden quantities (state). |

## Data

All three receivers (Attention 256, 192, 128), with 256 and 192 primary.

- **P19.** 268 / 245 / 329 unsolved states on 177 / 162 / 198 puzzles. For each state: one no-edit continuation, and for each fraction f ∈ {0.1, …, 1.0}, two independent random sets of wrong cells relabelled to their solution digits in the network's own coordinates. Each variant's predicted grids are saved for four iterations.
- **P18.** The same states. For each small closure (2–6 cells): the closure, one of its cells alone (`single`), and the closure minus one cell (`partial`).
- **P16.** Per-cell margin and the retained-commitment index for every one of these states.

## A1 — amount against identity, on a common evaluation set

**Evaluation set.** For each state s and fraction f, the evaluation set U(s, f) is the wrong cells that *neither* draw edits. Both draws are evaluated on the same cells, so the cells being predicted are held fixed while the edited set changes.

**Outcome.** y = 1 if a cell in U displays its solution digit k iterations after the edit (k = 1 primary, k = 4 reported). The no-edit continuation's value for the same cell, y0, is a feature.

**Model ladder.** Logistic regression on standardized features, each tier adding to the previous one:

| tier | features |
|---|---|
| **M0 (state)** | y0, the state's wrong-cell count, the iteration t |
| **M1 (+ amount)** | f and the number of cells edited |
| **M2 (+ visible configuration)** | for the evaluated cell c and the edited set E: edited peers of c; edited peers whose new digit equals c's current wrong digit (the edit makes c visibly conflict); edited peers whose old digit was c's solution digit (the edit removes a visible blocker); wrong peers remaining after the edit |
| **M3 (+ hidden quantities)** | c's margin and retained-commitment index, and the mean commitment index of its edited peers (P16's per-cell arrays) |

**Validation.**
- Five-fold cross-validation grouped by puzzle: every state and variant of a puzzle stays in one fold.
- Metrics: held-out log-loss and AUC.
- The improvement of each tier over the previous one, as the per-row held-out log-loss difference, gets a 95 % puzzle-bootstrap interval (1,000 resamples).

**The identity test proper (paired).** Consider the cells of U whose outcome differs between the two draws. Does the difference in M2's configuration features between the draws predict which draw repairs the cell? This conditions away the cell, the state and the amount entirely. Measured by the AUC of the feature difference, with a puzzle-bootstrap interval.

**Aggregate decomposition.** For each (s, f, draw), r is the repaired share of U. Report the share of r's variance lying between the two draws within (s, f), against the share explained by f and by s.

## A2 — independent against cooperative correction (P18)

For each small closure, its `single` cell c was edited alone, and also within the whole closure. Paired across closures, compare c's retention at k = 1:
- alone: P(c shows its solution | single);
- with its partners: P(c shows its solution | closure).

Do the same for `partial` (the closure minus one cell): its cells' retention, against the same cells' retention in the full closure.

Exact McNemar on the pairs; no-edit retention reported beside.

## Reading rules (fixed before any fit)

- **Independent local correction is rejected** if retention with partners exceeds retention alone with p < 0.05 on both primary receivers (A2), or if amount (M1) improves on M0 for the untouched cells with an interval excluding 0 (A1).
- **Visible configuration adds** if M2 improves on M1 with an interval excluding 0 *and* the paired identity AUC is ≥ 0.60, on both primary receivers. **Hidden quantities add** if M3 improves on M2 by the same standard and its AUC gain is ≥ 0.02.
- **Interpretation** (each is a direction for the next experiment, not a claim):
  - amount dominates, with neither configuration nor hidden adding: pursue *collective reconstruction*, i.e. a law relating the corrected share and its compatibility to the response;
  - configuration adds: pursue the visible coordination structure, and test whether hidden quantities modulate it;
  - hidden adds: pursue a compact state-dependent coupling, sequence B's transfer target;
  - nothing generalizes: these coordinates and this level of description are insufficient, so do not relabel a flexible fit as a theory.
- Everything here has been seen before in summary form. Its findings choose the next registration; they do not stand as confirmed results.
