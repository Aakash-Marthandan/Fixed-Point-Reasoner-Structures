# Registration: the review-period experiments P1c (mutually consistent corruptions) and P2b (message controls) — 2026-09-26 ~04:20Z, before any row

**Goal (page one).** A MEASUREMENT. No accuracy target; $0; the Mac's CPU; inference only on the manuscript's banked checkpoints; nothing trained. Follows `Note_2026-09-26_Review_Period_P1_P2_Registration.md` (P1/P2, read 03:25Z). P1c tool: `tools/rebuttal_p1c.py` (imports the P1 pipeline unchanged; outputs `runs/analysis/rebuttal_20260926b/`). P2b tool: `tools/rebuttal_p2b.py`, to be built after P1c launches; its design and rules are fixed here first.

## P1c — are coordinated random errors as hard to repair as the model's own?

**Why.** P1 showed that clue refutability explains about a third of the early repair gap and that the model's own digits explain the rest. EqR's wrong digits are conflict-free with everything else in the grid 54 % of the time (their solution-holders in the row, column and box are themselves wrong), against 6 % for the legal random draws and 3 % for uniform ones. This is the combinatorial signature the manuscript describes and the conjecture's "mutually supporting errors" premise would produce. The test: random wrong digits with that structure.

**Source.** `consistent_random_0..3`: k = EqR's wrong-cell count; a chain-closure procedure (a random empty cell takes a legal wrong digit; every unit peer that holds that digit in the solution is corrupted next with the legal digit that opens the fewest further holders; new chains start when the queue empties) followed by conflict-minimizing digit sweeps over the chosen cells; the most consistent of eight seeded restarts is kept (selection on the grid's conflict-free share only, never on a model outcome). Seeds `[20260926, id, 31, j]`. Legal against the givens by construction (asserted). Achieved per-puzzle conflict-free share, mean 0.556 (quartiles 0.50 / 0.56 / 0.62), against EqR's 0.494 (0.37 / 0.51 / 0.67): the draws are at least as coordinated as the model's errors. Input statistics in `inputs/p1c_sources.json`.

**Receivers, populations, references.** As P1: Attention 128/192/256 through the manuscript's pipeline (bitwise gate re-run per width), MLP 192 (C5) and EqR through the repair-radius pipeline with `eqr` re-run in-process; the 438 eligible pairs; E1_eqr and E1_uniform from the P1 run's report (attention) and port files (C5, EqR).

**Rule.** G_cons = (E1_consistent − E1_uniform) / (E1_eqr − E1_uniform), E1_consistent the mean over the four draws. CONSISTENCY-EXPLAINS if G_cons ≥ 0.67 (coordinated random errors are repaired about as slowly as the model's own: consistency among errors is the operative property); MODEL-DIGITS-SPECIAL if G_cons ≤ 0.33; MIXED otherwise; UNDEFINED under a 5-pp floor. Endpoint exact-at-16 difference consistent − eqr with the stratified paired bootstrap and draw-0 McNemar, reported without a letter. Descriptive: per-puzzle correlation of the conflict-free share with early error.

**Predictions and credences.** Attention receivers: CONSISTENCY-EXPLAINS 0.40; MIXED 0.40; MODEL-DIGITS-SPECIAL 0.20. Expected E1_consistent 22–32 %. The same letter on all three widths: 0.5. C5 and EqR: the same distribution.

**Wording under each outcome.** CONSISTENCY-EXPLAINS: "Random corruptions whose wrong digits are mutually consistent, so that each duplicates nothing correct, are repaired about as slowly as the model's own errors: the operative property is consistency among the errors, not their model origin." This supports the conjecture's premise that coordinated errors resist local correction; it does not identify a hidden representation. MODEL-DIGITS-SPECIAL: "Even random errors as mutually consistent as the model's own are repaired far faster: the model's digit choices resist repair beyond count, legality, location and consistency." The conjecture's combinatorial premise is then insufficient. MIXED: G reported with both readings.

## P2b — continued exchange or clue access? (message freeze and mean ablation)

**Why.** Section 4.1's message removal zeroes the cross-field value projections, which also removes digit-specific clue information and moves activations off their trained range; all four expert reviewers asked for a control that keeps clue access.

**Population.** The study's 256 shared first states (every second repair puzzle), the three attention checkpoints; references the study's `intact` (250 / 250 / 250) and `messages_off` (0 / 0 / 0) branches; initially incorrect 198 / 162 / 177, intact's new discoveries 192 / 156 / 171.

**Modes, applied at iterations 2–16 from the shared first state.** Each outer iteration applies the two-block stack 21 times, so 42 cross-field message tensors per puzzle per iteration. `messages_frozen`: every message tensor is replaced by the same puzzle's tensor recorded at the same (application, block) during iteration 1, so the exchange is the first iteration's and never updates, while the clue embedding still enters every fast update directly. `messages_mean`: every message tensor is replaced by the iteration-1 mean over the batch at the same (application, block, field, cell, channel), a generic message with no puzzle-specific content, replayed at every iteration. Implementation: the release cell's block function is patched in-process for a fresh trace (record and replay through closure capture inside the vmapped step); no released file is edited.

**Gates.** (1) Recording mode reproduces the study's intact logits bitwise at iteration 1 and over the 16-iteration intact continuation on batch 0; (2) replaying the recorded iteration-1 messages at iteration 1 reproduces iteration 1's logits bitwise (the replay path is exact); (3) the shared first state is asserted bitwise per batch.

**Rules.** Exact at 16 of 256. R_m = mode / intact. For `messages_frozen`: CLUE-ACCESS-MOSTLY if R_m ≥ 0.9 (the zero-ablation collapse was lost clue access and off-distribution activations); EXCHANGE-NEEDED if R_m ≤ 0.5 (even with clue-derived, on-distribution messages, no continued exchange loses at least half); MIXED otherwise. Discoveries D = initially incorrect puzzles correct at 16 under `messages_frozen`: NO-DISCOVERY if D ≤ 5; DISCOVERY if D ≥ 0.25 × intact's discoveries; MIXED otherwise. `messages_mean` reported beside: if its R_m is within 0.1 of frozen's, the first iteration's specific content adds nothing beyond a generic message.

**Predictions and credences.** R_m(frozen): MIXED 0.45; EXCHANGE-NEEDED 0.35; CLUE-ACCESS-MOSTLY 0.20. D: MIXED 0.40; DISCOVERY 0.35; NO-DISCOVERY 0.25. mean within 0.1 of frozen: 0.40.

**Wording under each outcome.** EXCHANGE-NEEDED and NO-DISCOVERY: "With the first iteration's messages replayed, so that clue-derived information stays available, no initially incorrect puzzle is solved and most initial solutions are lost: continued exchange between digit fields, not merely access to clue information, is necessary." CLUE-ACCESS-MOSTLY or DISCOVERY: Section 4.1's "continued interaction supports discovery and preservation" is narrowed to "the cross-field branch supplies information the update needs; whether it must update is not established". MIXED: R_m and D reported with both readings. Under every outcome: the frozen messages are the model's own but stale; no message content is identified.

## Labels and plan

Confirmatory: G_cons, R_m and D letters. Exploratory: endpoint intervals, the consistency–error correlation, mean-versus-frozen. Plan: P1c launches first (three widths in parallel, C5 alongside, EqR after C5; about 1.5–2 h); P2b is built while P1c runs, smoked, and launched after its gates pass. No shared tool edited; `lens_repair_radius.T_TOTAL` set to 16 in-process.

---

## P1c Outcome (2026-09-26 06:20Z; `runs/analysis/rebuttal_20260926b/report.{txt,json}`; every gate passed; no shared tool edited)

**Integrity.** Bitwise reproduction of the study's intact first step on all three widths; the port receivers reproduced their P1 references exactly (C5 E1 41.20 %, EqR 37.55 %). Pooled conflict-free share of wrong digits: EqR's own 0.545, consistent draw 0 0.570, legal random 0.064, uniform 0.031.

| receiver | E1 EqR's own | E1 uniform | E1 legal (P1) | E1 consistent (4 draws) | G_cons | letter | exact-16 own / consistent |
|---|---|---|---|---|---|---|---|
| Attention 128 | 36.96 | 5.04 | 16.82 | 36.59 | 0.988 | CONSISTENCY-EXPLAINS | 95.2 / 94.7 % (−0.5 pp [−2.4, +1.5]) |
| Attention 192 | 35.94 | 2.84 | 12.72 | 29.59 | 0.808 | CONSISTENCY-EXPLAINS | 97.3 / 97.4 % (+0.2 [−1.2, +1.7]) |
| Attention 256 | 35.15 | 3.80 | 12.86 | 31.26 | 0.876 | CONSISTENCY-EXPLAINS | 98.2 / 97.5 % (−0.6 [−1.8, +0.5]) |
| MLP 192 (C5) | 41.20 | 5.91 | 18.61 | 36.58 | 0.869 | CONSISTENCY-EXPLAINS | 94.5 / 93.3 % |
| EqR | 37.55 | 2.44 | 13.43 | 29.75 | 0.778 | CONSISTENCY-EXPLAINS | 85.2 / 88.8 % |

Random wrong digits made mutually consistent, so that each duplicates nothing correct in its row, column or box, are repaired about as slowly as the model's own errors: they close 78–99 % of the gap between uniform corruptions and model-produced errors, where legality alone closed 29–37 % (P1). Endpoint differences are within about one point on every attention receiver, intervals through zero; on EqR the consistent random errors end easier than its own (88.8 vs 85.2 %). Draw-to-draw spread is under one point everywhere. Descriptive: within the consistent draws, puzzles whose errors are more mutually consistent are repaired more slowly (correlation +0.26 on Attention 128, +0.29 on MLP 192).

**Predictions scored.** CONSISTENCY-EXPLAINS (0.40) HIT on all three widths; the same letter on all three (0.5) HIT; C5 and EqR the same letter HIT; expected E1_consistent 22–32 %: HIT on 192, 256 and EqR, MISS on 128 and C5 (36.6 %, above the band).

**The registered sentence that applies.** "Random corruptions whose wrong digits are mutually consistent, so that each duplicates nothing correct, are repaired about as slowly as the model's own errors: the operative property is consistency among the errors, not their model origin." This supports the conjecture's premise that coordinated errors resist local correction. It does not identify a hidden representation, and the residual model-specific difficulty is small on the digit-field receivers (G 0.81–0.99) and larger on EqR (0.78).
