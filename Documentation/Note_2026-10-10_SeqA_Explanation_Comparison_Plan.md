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

---

## Results (2026-10-10; `runs/analysis/seqA_20261010/report.{txt,json}`; exploratory)

The tool was run once after the plan and the tool were committed (f521ad2). One fix was needed before any output: at f = 1.0, P19 saved a single draw and has no unedited cell, so that fraction is skipped.

**A1, k = 1.** Rows are (variant, cell neither draw edits) pairs: 38,698 / 34,540 / 50,840 on 175 / 159 / 198 puzzles. Held-out log-loss gain of each tier over the previous, with its 95 % puzzle-bootstrap interval and the AUC gain:

| receiver | M1 amount over M0 | M2 configuration over M1 | M3 hidden over M2 | paired identity AUC | same-cell agreement between draws |
|---|---|---|---|---|---|
| Attention 256 | +0.021 [+0.015, +0.028], AUC +0.023 | +0.0005 [0.000, +0.001], AUC +0.001 | +0.002 [−0.000, +0.004], AUC +0.002 | 0.53 [0.52, 0.54] | 0.77 |
| Attention 192 | +0.047 [+0.036, +0.059], AUC +0.050 | −0.000 [−0.001, +0.001], AUC +0.000 | +0.006 [+0.002, +0.009], AUC +0.004 | 0.55 [0.53, 0.56] | 0.79 |
| Attention 128 | +0.004 [+0.002, +0.006], AUC +0.007 | +0.000 [−0.000, +0.001], AUC +0.000 | +0.007 [+0.005, +0.009], AUC +0.008 | 0.53 [0.51, 0.54] | 0.77 |

The full models reach AUC 0.83 / 0.85 / 0.83. Most of that comes from the cell's own no-edit outcome; M0 alone has AUC 0.81 / 0.79 / 0.82. At k = 4 the amount gain shrinks (+0.004, +0.016, −0.000), and configuration still adds nothing.

**The shape of the amount effect** (descriptive). Among untouched wrong cells that the no-edit continuation does *not* repair, the share repaired one iteration after the edit, by corrected fraction:

| fraction | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 |
|---|---|---|---|---|---|
| Attention 256 | 0.22 | 0.33 | 0.50 | 0.76 | 0.92 |
| Attention 192 | 0.19 | 0.28 | 0.59 | 0.83 | 0.94 |
| Attention 128 | 0.15 | 0.20 | 0.30 | 0.39 | 0.62 |

Among cells the no-edit continuation *does* repair, 14–18 % are lost after a small correction elsewhere (256 and 192, f = 0.1–0.3). More are kept after a large one (0.96–1.00 at f ≥ 0.7).

**A2 (P18).** A closure's single cell kept alone vs with its partners, at k = 1:

| receiver | alone | with partners | no edit | McNemar (only-with-partners / only-alone) |
|---|---|---|---|---|
| Attention 256 | 0.660 | 0.696 | 0.611 | 31 / 20, p 0.16 |
| Attention 192 | 0.617 | 0.682 | 0.538 | 31 / 14, p 0.016 |
| Attention 128 | 0.604 | 0.606 | 0.444 | 52 / 51, p 1.0 |

Partial against full closure, on the same cells: no difference on 256 and 192, and the partial slightly ahead on 128 (p 0.028).

**Reading by the rules fixed above.**
- **Independent local correction is rejected through A1.** On both primary receivers, the amount of correction elsewhere improves prediction at cells that were not edited, with intervals excluding 0, while the cells' local relations to the edited set (M2) add nothing. A2 alone shows cooperation only on 192.
- **Visible configuration does not add** (M2 ≈ M1; paired identity AUC 0.53–0.55, below 0.60). Which equally sized set is corrected does not predict which untouched cells repair beyond the amount. About 21–23 % of the same untouched cells nevertheless change outcome between two same-size draws. That variation is real, but neither the configuration features nor the state-level hidden quantities predict it. Predicting it would need features of the post-edit state, i.e. replay.
- **Hidden quantities** (margin, commitment index) add a small, significant amount on 192 and 128, but AUC +0.004 to +0.008 is below the 0.02 bar. They do not add by the rule.
- **Direction for the next experiment: collective reconstruction.** Untouched cells are repaired in proportion to how much of the configuration is corrected at once, on top of each cell's own propensity, not through which cells are corrected. The open question for sequence B becomes whether *retained state sets this law's parameters*. For a fixed displayed configuration and a fixed correction, do different hidden histories of the same puzzle give different dose–response curves? That would be a state-dependent collective law, transferable as a parameter, rather than a compact coupling operator between particular decisions.

---

**Addendum — corrections and a narrower reading (2026-10-10, after an external review; the quantities were rechecked from the code and the arrays).**

1. *Variance decomposition.* A pair's contribution to the population variance is (r₁ − r₂)² / 4, not / 2. The exact between-draw shares at k = 1 are **0.229 / 0.177 / 0.217**, not 0.457 / 0.354 / 0.434; at k = 4 they are 0.340 / 0.276 / 0.292. The regression results are unaffected.
2. *A2 clustering.* Pairs repeat within puzzles: two closures, two iterations. Puzzle-bootstrap 95 % intervals for the paired retention difference, with partners minus alone:

   | receiver | single cell | partial vs closure |
   |---|---|---|
   | Attention 256 | +0.036 [−0.016, +0.090] | +0.004 [−0.029, +0.040] |
   | Attention 192 | **+0.064 [+0.012, +0.117]** | +0.021 [−0.012, +0.056] |
   | Attention 128 | +0.002 [−0.044, +0.041] | −0.028 [−0.058, +0.000] |

   Positive on 192; unresolved on 256 and 128. The 128 partial result is not significant once clustering is respected.
3. *The dose curve mixes dose with population.* The cells neither draw edits thin out as the fraction rises. On Attention 256 the evaluation set is 263 states and 11,000 cell-draw observations at f = 0.1, 263 and 3,336 at f = 0.5, and **73 and 160 at f = 0.9**. Only 76 of those 160 are among cells the no-edit continuation does not repair. The plotted averages are therefore not a clean within-state response law: states with few wrong cells drop out at high doses.
4. *y0 is information from the future.* The strongest feature, y0, is whether the cell is correct after the *unedited continuation*. The models are conditional ones (given the no-edit continuation), not predictions from the present state alone. They also use ground truth (which cells are wrong), so they are diagnostic, not deployable.
5. *The narrower reading, which supersedes "independent local correction is rejected" and the chat summary's "which cells doesn't matter":*

   > Across the tested interventions, the amount of correction predicts repair at untouched cells. The particular local configuration features and the hidden summaries tested add little predictive information.

   - Set identity matters causally: changing the corrected set changes about 21–23 % of the same untouched cells' outcomes. The tested features predict those differences poorly. A useful statistical law averaged over correction sets is still possible.
   - What is rejected is the strict claim that correction elsewhere has no effect. Many separate additive contributions whose sum crosses a decision boundary are not excluded; that needs the A / B / A+B design, with additivity tested on the score scale.
   - Hidden coupling was not tested broadly. M3 held margins, the previously unsuccessful commitment index and its average.
   - Collective reconstruction and hidden coupling are compatible: hidden interactions could implement the collective law.
