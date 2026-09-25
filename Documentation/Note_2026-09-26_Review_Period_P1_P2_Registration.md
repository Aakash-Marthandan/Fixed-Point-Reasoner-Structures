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
