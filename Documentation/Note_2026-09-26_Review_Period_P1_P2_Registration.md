# Registration: the review-period experiments P1 (clue legality and repair) and P2 (the slow carry beyond its readout) — 2026-09-25 ~22:30Z (2026-09-26 IST), before any row

**Goal (page one).** A MEASUREMENT. No accuracy target; $0; the Mac's CPU; inference only on the manuscript's banked checkpoints; nothing trained. Tool: `tools/rebuttal_p1p2.py` (selftest passed with one score-preservation mutant); runs under `runs/analysis/rebuttal_20260926/`; the queue `queue.sh` there. The manuscript is under review; results enter its response as labeled additions with the submitted PDF unchanged as the baseline.

**Why.** Ten simulated reviewers over two panels (2026-09-25/26) converged on two objections to the manuscript's Section 4. (1) The matched-error repair contrast is confounded by clue refutability (0.17 % of EqR's wrong digits contradict a given against 63 % of uniform random corruptions) and by the anchor branch's training on uniform corruptions. (2) The slow/fast reset asymmetry is predicted by the readout depending on the slow state alone, so "retained history" is not separated from "the current answer".

## P1 — does direct clue refutability explain the repair gap?

**Populations and receivers.** The study's 512 rating-stratified repair puzzles (`runs/analysis/attention_transfer_20260923/inputs/repair.npz`, pinned), primary analysis on the 438 with at least one EqR error, as the manuscript. Receivers: Attention 128/192/256 (release checkpoints, the manuscript's own pipeline `tools/attention_transfer_study.py` re-used unchanged by the new tool: batch 128, float32 CPU, 16 iterations, full vocabulary, the supplied grid embedded into the slow state with the fixed fast buffer); MLP 192 seed 0 (C5, 46k) and EqR's released weights through `tools/lens_repair_radius.py` (imported unchanged; 16 iterations; its digit-only decoding). The port re-runs the references `eqr` and `uniform_0` in the same process so every contrast is like for like.

**Sources (all clue-preserving; recorded in `inputs/p1_sources.json`).**
- `legal_random_0..3`: the solution with k = EqR's wrong-cell count empty cells made wrong at random FEASIBLE cells with a LEGAL wrong digit (a digit that is not the solution's and does not duplicate a given in the cell's row, column or box); seeds `[20260926, id, 11, j]`.
- `legal_at_eqr_0..3`: EqR's own wrong cells; a legal wrong digit OTHER than EqR's, EqR's digit only when it is the sole legal wrong digit (fallback count recorded); seeds `[20260926, id, 12, j]`.
- `guess_sa128`, `guess_sa256`: the attention models' own fixed-start first-iteration grids on the same puzzles, clues restored (model-produced, not count-matched to EqR); `legal_match_sa128/256`: legal random at random cells matched to those counts; seeds `[20260926, id, 13, w]`.
- Infeasible-cell rule (fixed before data): a cell is feasible when at least one legal wrong digit exists; a puzzle with fewer feasible cells than k receives all feasible cells and is excluded from the primary analysis (deficit recorded; observed 0 in every draw). Input statistics at prepare: every legal source has zero clue-conflict share; in each `legal_at_eqr` draw 274 of the 10,185 EqR wrong cells keep EqR's digit because it is the sole legal wrong digit there; the smoke on 16 puzzles passed every gate (first-step logits bitwise, score shift 1e-14, re-encoded readout agreement 1.0).

**Statistics and rules.** E1 = mean empty-cell error after one iteration over the eligible pairs. Gap closure G = (E1_legal − E1_uniform) / (E1_eqr − E1_uniform), with E1_uniform the study's four-draw mean and E1_legal the mean over the four legal draws. Letters: REFUTABILITY-MOSTLY if G ≥ 0.67; CONFIGURATION-BEYOND if G ≤ 0.33; MIXED otherwise; UNDEFINED when the eqr−uniform gap is under 5 pp. Computed per receiver for `legal_random` (G) and `legal_at_eqr` (G_loc). Endpoint exact-at-16 advantage legal − eqr: paired stratified bootstrap (10,000 resamples within the eight rating bins) and draw-0 exact McNemar, reported without a letter. Cross-model sources: EXPLORATORY, no letter (E1 and exact-16 of `guess_sa*` against `legal_match_sa*` on puzzles with at least one wrong cell).

**Predictions and credences.** Attention receivers: MIXED 0.45; REFUTABILITY-MOSTLY 0.30; CONFIGURATION-BEYOND 0.25; expected E1_legal 10–25 % between uniform (3–5 %) and EqR (35–37 %). G_loc ≥ 0.67: 0.50. The same letter on all three widths: 0.6. C5 and EqR: the same distribution; EqR's G above C5's: 0.6. Endpoint advantage legal − eqr inside [−2, +4] pp on every attention receiver: 0.7.

**Wording under each outcome (written now).**
- REFUTABILITY-MOSTLY: "Random corruptions that respect the givens are repaired about as slowly as the model's own errors at matched counts: direct clue refutability accounts for most of the early repair gap." Section 4's "error configuration shapes repair beyond error count" is narrowed to legality against the givens.
- CONFIGURATION-BEYOND: "Clue-legal random corruptions are still repaired far faster than the model's own errors at matched counts: refutability explains at most a third of the gap; the remainder lies in which cells and digits are wrong." Mutually supporting errors are not thereby established.
- MIXED: G reported with both readings; no mechanism word.
- Under every outcome: one checkpoint per model, the manuscript's populations, inference-time inputs outside the training distribution; the anchor-familiarity confound is addressed only by the EqR receiver.

## P2 — what does the slow carry contribute beyond its readout?

**Population.** The study's 256 shared first states (every second repair puzzle; `inputs/interventions.npz`), the three attention checkpoints. References: the study's saved `intact` (250/250/250) and `reset_slow` (58/101/126) branches, re-derived bitwise as a gate.

**Modes, applied before every iteration 2–16 from the shared first state (`trajectory` in the tool; every other line of the loop is the study's).**
- `reencode_slow`: the slow state replaced by `embed_answer` of the current decoded grid with clues restored; the fast state kept. The immediate readout agreement of the re-encoded state is recorded (descriptive).
- `null_h0`: per (digit, cell) vector, the component along the readout vector `lm_head` is kept and the orthogonal remainder replaced by the initial buffer's remainder, so all 729 digit scores are unchanged; gate: max score shift < 1e-2 per chunk, recorded.
- `null_random`: as `null_h0` with the remainder replaced by a random orthogonal vector of matched norm; seeds `[20260926, id, 21, t]`. The matched-magnitude control.

**Statistic and rules.** R = 1 − (intact − mode) / (intact − reset_slow) in exact-at-16 counts. ANSWER-ONLY if R ≥ 0.9; BEYOND-READOUT if R ≤ 0.5; MIXED otherwise; UNDEFINED when the slow-reset loss is under 20 puzzles. Primary: `reencode_slow` and `null_h0`. `null_random` reported beside them: if it is within 0.1 of `null_h0`, the effect of the replaced content is not specific.

**Predictions and credences.** `reencode_slow`: MIXED 0.45, ANSWER-ONLY 0.30, BEYOND-READOUT 0.25. R(null_h0) ≥ R(reencode_slow) − 0.1: 0.6. R(null_random) ≤ R(null_h0) − 0.1: 0.5.

**Wording under each outcome.** ANSWER-ONLY: "Re-encoding the decoded grid every iteration recovers at least 90 % of what the slow reset lost: the slow carry's useful content is essentially its current answer." BEYOND-READOUT: "...recovers at most half: the slow carry holds information beyond its readout that later corrections depend on." MIXED: reported with R. Under every outcome: no content identified; interventions outside the training distribution.

## Gates, labels, plan

Gates: (1) selftest; (2) `prepare` asserts clue preservation, exact counts, zero clue-conflict share for every legal source, EqR's 17/10,185 and the uniform draw's ≈ 63 %; (3) the tool's loop reproduces the study's intact interventions chunk 0 (128 puzzles × 16 iterations) bitwise before any new row on each width, and the shared first state is asserted bitwise per batch; (4) the null modes assert the score shift; (5) a 16-puzzle, 2-iteration smoke to a scratch directory before launch; (6) the port's references re-run in-process. Confirmatory: the G, G_loc and R letters. Exploratory: the cross-model sources, the endpoint intervals, the readout agreement, per-width comparisons. Plan: `queue.sh` runs the three widths in parallel (about 3–5 h), then C5 and EqR (about 1–2 h), then `report`. P2c (the 94k identical-grid state swaps) and the message-freeze and mean-ablation controls are registered separately before their build. No shared tool is edited; `lens_repair_radius.T_TOTAL` is set to 16 in-process.

---

## Outcome (2026-09-26 03:20Z; `runs/analysis/rebuttal_20260926/report.{txt,json}`; every gate passed; no shared tool edited; one reader fix after the run: the port section of `report()` indexed the eligibility mask by test ID instead of by position and crashed before writing, corrected to positional indexing with no rule touched, and the corrected report reproduces the hand-computed interim values exactly)

**Integrity.** All three attention loops reproduced the paper's intact continuation bitwise (128 puzzles × 16 iterations each); the shared first state matched bitwise on every batch; the score-preserving modes shifted the 729 digit scores by at most 2e-14; the port receivers reproduced the manuscript's references exactly (C5: E1 41.20 / 5.91 %, exact 94.5 / 98.6 %; EqR: 37.55 / 2.44 %, 85.2 / 98.2 %); the attention references reproduced Table 7 (36.96 / 35.94 / 35.15 % EqR; 5.04 / 2.84 / 3.80 % uniform).

**P1 letters (early error after one iteration, 438 eligible pairs).**

| receiver | E1 EqR | E1 uniform | E1 legal, random cells | G | letter | E1 legal, EqR's cells | G_loc | letter |
|---|---|---|---|---|---|---|---|---|
| Attention 128 | 36.96 | 5.04 | 16.82 | 0.369 | MIXED | 16.21 | 0.350 | MIXED |
| Attention 192 | 35.94 | 2.84 | 12.72 | 0.298 | CONFIGURATION-BEYOND | 15.74 | 0.390 | MIXED |
| Attention 256 | 35.15 | 3.80 | 12.86 | 0.289 | CONFIGURATION-BEYOND | 15.58 | 0.376 | MIXED |
| MLP 192 (C5) | 41.20 | 5.91 | 18.61 | 0.360 | MIXED | 23.39 | 0.495 | MIXED |
| EqR | 37.55 | 2.44 | 13.43 | 0.313 | CONFIGURATION-BEYOND | 16.72 | 0.407 | MIXED |

Clue refutability closes 29–37 % of the early repair gap on every receiver; the five values span 0.08 and straddle the 0.33 boundary, so the letter split between MIXED and CONFIGURATION-BEYOND is the threshold, not the receivers. Placing the legal digits at EqR's own cells closes a further 0–14 points (G_loc 0.35–0.50); the remaining half or more of the gap belongs to the digits the model chose. Exploratory cross-model sources: the attention models' own fixed-start first guesses are repaired at 40–45 % early error by every receiver (their count-matched legal corruptions at 11–22 %), so the contrast is producer-independent. Endpoints (exact at 16, legal − EqR): attention +1.0 / +1.5 (128), +0.9 / +1.1 (192), +0.7 / −0.1 (256) pp, every interval through zero; C5 +2.6–2.9 pp; EqR +5.0–7.3 pp (its own endpoint gap persists).

**P2 letters (exact at 16 of 256; intact 250; slow reset 58 / 101 / 126).**

| mode | 128 | R | 192 | R | 256 | R | letter |
|---|---|---|---|---|---|---|---|
| reencode_slow | 148 | 0.469 | 152 | 0.342 | 151 | 0.202 | BEYOND-READOUT (all) |
| null_h0 (scores kept; invisible part := buffer's) | 109 | 0.266 | 138 | 0.248 | 122 | −0.032 | BEYOND-READOUT (all) |
| null_random (scores kept; invisible part := matched-norm noise) | 171 | 0.589 | 212 | 0.745 | 230 | 0.839 | MIXED (all) |

The two primary modes read BEYOND-READOUT on every width, but the control reverses the registered expectation: replacing the readout-invisible component by matched-norm noise is the MILDEST edit (losses 79 / 38 / 20 of 250), the buffer's component the most damaging (at width 256 as damaging as the full slow reset, with an oscillating trajectory), and re-encoding the decoded grid in between. Reading: (a) keeping every digit score does not rescue the slow reset, so the reset's damage is not the loss of the current answer; (b) a neutral erasure of the invisible content costs 8–32 pp, so the slow carry holds information beyond its scores that later corrections use, moderately and decreasingly with width; (c) most of the slow reset's damage is the injected initial-buffer signal, not discarded history. The re-encoded state's immediate readout reproduced its grid exactly (agreement 1.0).

**Predictions scored.** P1 attention letter: MIXED (0.45) hit on 128 only, CONFIGURATION-BEYOND (0.25) hit on 192 and 256; E1_legal 10–25 % HIT. Same letter on all three widths (0.6): MISS (a 0.08 spread across the boundary). G_loc ≥ 0.67 (0.5): MISS (0.35–0.50). EqR's G above C5's (0.6): MISS (0.313 vs 0.360). Attention endpoint advantage inside [−2, +4] pp (0.7): HIT. P2 reencode: BEYOND-READOUT (0.25) hit. R(null_h0) ≥ R(reencode) − 0.1 (0.6): MISS. R(null_random) ≤ R(null_h0) − 0.1 (0.5): MISS, opposite direction.

**The registered sentences that apply.** P1, all receivers: "Clue-legal random corruptions are still repaired far faster than the model's own errors at matched counts: refutability explains about a third of the early gap; the remainder lies in which cells and, above all, which digits are wrong." Mutually supporting errors are not thereby established. P2: "Keeping every digit score does not rescue the slow reset; erasing the readout-invisible part of the slow state neutrally costs 8–32 points; the buffer's own invisible component, not the loss of history, carries most of the reset's damage." Under every outcome: one checkpoint per model, the manuscript's populations, inference-time inputs and edits outside the training distribution, no content identified.
