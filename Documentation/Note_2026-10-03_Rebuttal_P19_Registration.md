# Note 2026-10-03 — P19: a collective threshold — how much of the error must be corrected at once for the network to accept it and complete the rest? (registration, before any row)

**Goal (page one).** A MEASUREMENT. No accuracy target; $0; the Mac's CPU; inference only on the manuscript's banked attention checkpoints; nothing trained. Part of the operator program (plan §7). Tool: `tools/rebuttal_p19.py` (imports the P18 and P15 runners unchanged; outputs `runs/analysis/rebuttal_20261003/`). Queued behind P18.

## Why

P16 (verdict pending; complete on Attention 192 and 128): a single cell's full state, rewritten in the network's own coordinates, does not hold its decision; the rest of the grid re-derives it. P18 on Attention 192 (complete; its outcome will be written by the registered rule) and the first chunks of 128: corrections of two to six wrong cells at once are kept at 0.14–0.28 of the available gain whether or not they form a conflict-free closure, while corrections of seven or more, closure or random set alike, are kept at 0.96–0.97 and complete the whole grid 94 % of the time; the full correction completes 98 %. On Attention 128 even the full correction completes only 40–53 % of states in the first chunk. The acceptance of a correction appears to depend on how much of the error is corrected at once, not on its structure. This experiment measures that dependence directly: is there a critical share of the error above which the network accepts a correction and completes the rest itself, and how sharp is the boundary? The manuscript's §3 reports that natural completions jump from a median 40 % remaining empty-cell error to zero in one iteration; the threshold measured here is the same quantity seen from an edited state.

## Design

**Receivers, states, split.** Attention 256 and 192 (letters), 128 (reported); the study's 256 intervention puzzles, intact fixed-start states after iteration t ∈ {1, 2}, unsolved states only; P15's seeded puzzle-level DISCOVERY/CONFIRMATION split; states through the release step at batch 128, reproducing the study's logits bitwise.

**The edit.** For each state with wrong-cell set W, a uniformly random subset of size max(1, round(f · |W|)) is corrected at once by cell relabels (P16/P18's channel: each cell's slow and fast field vectors exchanged between its displayed and its solution digit), for f ∈ {0.1, 0.2, …, 1.0}, two draws per f below 1.0 (seed `[20261005, puzzle ID, t]`), plus no edit. Four outer iterations through the release step.

**Gates.** States bitwise; every relabel produces the designed display; every no-edit rollout reproduces the study's displayed grids. A smoke on 14 states of Attention 128 passed all three.

## Quantities (k = 1 primary)

y(f): share of edited states whose whole grid is exact at t + k (pooled over draws and over t = 1, 2). y(0): the same with no edit. Normalized completion z(f) = [y(f) − y(0)] / [y(1.0) − y(0)]. f10, f50, f90: the smallest f (linear interpolation between grid points) at which z reaches 0.1, 0.5, 0.9. Also: the share of edited cells still correct at t + k, and the network's normalized repair of the wrong cells left uncorrected, R(f) = [P(c | edit) − P(c | no edit)] / [1 − P(c | no edit)] over those cells.

## Rule (confirmatory, CONFIRMATION states, k = 1, letters on Attention 256 and 192)

SHARP if f90 − f10 ≤ 0.30 (below a critical share corrections are undone and above it the network completes); GRADUAL if f90 − f10 ≥ 0.60; MIXED otherwise; UNDEFINED if y(1.0) − y(0) < 0.20 (the channel cannot write a stable solution on that receiver).

**Predictions and credences (before any row).** SHARP 0.45; MIXED 0.30; GRADUAL 0.25. f50 between 0.4 and 0.8 on both lettered widths: 0.6. The network repairs uncorrected wrong cells (R ≥ 0.5) at f ≥ f90: 0.7. Attention 128 UNDEFINED: 0.4.

## Wording under each outcome

- SHARP: "Corrections written into the network's state are accepted only collectively: below a critical share of the errors they are undone, and above it the network completes the rest itself within one iteration. In the network's own dynamics the solution's basin has a sharp boundary, and an abrupt completion is the crossing of it."
- GRADUAL: "The more of the error is corrected at once, the more the network completes, without a critical share."
- MIXED: f10, f50 and f90 reported with both readings.
Under every outcome: the edits use ground truth to choose digits and are relabels in the network's own coordinates; the network never receives the solution as input.

**Exploratory.** k = 2–4; t = 1 versus t = 2; f50 against each state's number of wrong cells (is the threshold a share or an absolute count of remaining errors); comparison with the natural pre-completion error of §3.

## Plan

Commit this note and the tool before any row; the queue waits on P18's queue (PID-based), then runs the three widths in parallel at nice 5 and the report; the Outcome appended here and a ledger line written when read.
