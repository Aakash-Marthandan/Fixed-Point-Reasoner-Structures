# Note 2026-10-02 — P18: coordinated revision — are errors held in place by their mutually consistent partners? (registration, before any row)

**Goal (page one).** A MEASUREMENT. No accuracy target; $0; the Mac's CPU; inference only on the manuscript's banked attention checkpoints; nothing trained. Part of the operator program (`Note_2026-10-02_Rebuttal_P15_P16_P17_Registration.md`; plan §7, the behavioural test "coordinated revision"). Tool: `tools/rebuttal_p18.py` (imports the P16 and P15 runners unchanged; outputs `runs/analysis/rebuttal_20261002c/`). Queued behind P16 and the linear layer.

## Why

P15: changing a displayed decision is nearly inert. Interim P16 (first chunks of Attention 128 and 192; not P16's verdict, whose rules are untouched): rewriting a cell's entire state, slow and fast, in the network's own coordinates realizes at most about a quarter of the available gain when correct, and less when wrong; the decision is pulled back by the rest of the grid. P1c: errors that are mutually consistent are repaired as slowly as the model's own; P7: a wrong cell the grid visibly contradicts is not repaired more often. A collective account predicts all of this: each wrong decision is held by the other wrong decisions that are consistent with it, so a correction survives only if its partners change with it.

A fact about the inputs, computed from the saved trajectories before this note (no new model run): for each wrong cell, its **closure** is the smallest set of wrong cells containing it that is closed under "a wrong peer displays the solution digit of a member". Correcting a closure creates no duplicate; correcting a cell without its closure does. At iteration 1, closures are large: median 25 cells against a median of 26 wrong cells per unsolved state (Attention 256; 192 similar); 80 % of wrong cells have closures of seven or more, 10 % of two to six, 9–10 % of one. Over half of unsolved states contain at least one closure of two to six cells. A consistent correction of almost any single error requires correcting nearly all of them, which is also what an abrupt completion looks like.

## Design

**Receivers, states, split.** Attention 256 and 192 (letters), 128 (reported); the study's 256 intervention puzzles, intact fixed-start states after iteration t ∈ {1, 2}, unsolved states only; P15's seeded puzzle-level DISCOVERY/CONFIRMATION split. States through the release step at batch 128, reproducing the study's logits bitwise.

**The edit.** A cell relabel (P16's channel): the cell's slow and fast field vectors for its displayed digit and its solution digit are exchanged, so the cell holds its solution belief in the network's own coordinates; several cells are relabelled jointly by applying this at each.

**Edit sets per state** (seed `[20261004, puzzle ID, t]`): up to two small closures (2–6 cells) chosen uniformly; for each: `closure` (all its cells), `random` and `random2` (size-matched sets of wrong cells drawn uniformly from the wrong cells outside the closure, or from all wrong cells if too few), `single` (one cell of the closure alone), `partial` (the closure minus one cell chosen uniformly, the "omitted" cell); one large closure (≥ 7) with a size-matched random set; `all_wrong` (every wrong cell); `noedit`. Every condition continues four outer iterations through the release step.

**Gates.** States reproduce the study's logits bitwise; every closure creates no duplicate when corrected; every relabel produces the designed display; every no-edit rollout reproduces the study's displayed grids. A smoke on 14 states of Attention 128 passed all four.

## Quantities (k = iterations after the edit; k = 1 primary)

For each edited cell, c = 1 if it displays its solution digit at t + k. e(kind) = [P(c | edit) − P(c | no edit)] / [1 − P(c | no edit)], pooled over the edited cells of that kind (the share of the available gain kept). Move completion: for `partial` edits, the same normalized gain for the omitted cell (the network correcting it itself). Exact completion of the whole grid at t + k by kind, descriptive.

## Rules (confirmatory, CONFIRMATION states, k = 1, letters on Attention 256 and 192)

**A — coordinated revision.** With e_rand the pooled value over `random` and `random2`: COORDINATED-REVISION if e(closure) − e_rand ≥ 0.20; INDEPENDENT if e(closure) − e_rand ≤ 0.05; MIXED otherwise.

**B — the network completes a coordinated move.** COMPLETES-THE-MOVE if the omitted cell's normalized gain ≥ 0.20; DOES-NOT-COMPLETE if ≤ 0.05; MIXED otherwise.

**Predictions and credences (before any row).** A: COORDINATED-REVISION 0.45; MIXED 0.30; INDEPENDENT 0.25. B: COMPLETES-THE-MOVE 0.40; MIXED 0.30; DOES-NOT-COMPLETE 0.30. Same A letter on 256 and 192: 0.6.

**Wording under each outcome.**
- COORDINATED-REVISION: "A set of corrections that together leave no conflict is kept, while the same number of corrections that do not form such a set is undone: errors are held in place by their mutually consistent partners, and the network accepts corrections that move them together."
- INDEPENDENT: "Corrections are kept or undone regardless of whether they form a conflict-free set: the pull-back is not explained by mutual support among the errors."
- COMPLETES-THE-MOVE: "Given all but one of a coordinated correction, the network makes the last one itself."
- DOES-NOT-COMPLETE: "The network does not complete a coordinated correction it has mostly been given."
Under every outcome: the edits are relabels in the network's own coordinates; ground truth defines the closures and is never given to the network.

**Exploratory.** k = 2–4; t = 2; large closures against size-matched random sets; `single` against `closure`; `all_wrong` (the full correction); whole-grid completion by kind.

## Plan

Commit this note and the tool before any row; the queue waits on P16's queue (PID-based), then runs the three widths in parallel at nice 5 and the report; the Outcome appended here and a ledger line written when read.
