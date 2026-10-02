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

---

## P19 Outcome (2026-10-02 21:28Z; `runs/analysis/rebuttal_20261003/report.{txt,json}`; every gate passed; no shared tool edited)

**Integrity.** On all three widths: states bitwise; every relabel produced its designed display; every no-edit rollout reproduced the study's displayed grids. Widths 192 and 128 were started by hand at 19:05Z (same tool and outputs; the queue's later launch found them cached). A reader fix was committed before any outcome was read (c9169be: float32 fractions rounded to the grid keys). An independent recount of the completion curves from the raw arrays agrees with the report at every fraction on every width.

**Whole-grid completion at t + 1 against the share of wrong cells corrected at once (CONFIRMATION, t = 1 and 2 pooled).**

| receiver | 0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 | f10 / f50 / f90 | letter |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Attention 256 | 0.49 | 0.57 | 0.57 | 0.58 | 0.62 | 0.70 | 0.80 | 0.86 | 0.91 | 0.95 | 0.96 | 0.10 / 0.53 / 0.80 | GRADUAL |
| Attention 192 | 0.46 | 0.48 | 0.49 | 0.55 | 0.62 | 0.72 | 0.82 | 0.90 | 0.92 | 0.96 | 0.98 | 0.25 / 0.50 / 0.81 | MIXED |
| Attention 128 (no letter) | 0.31 | 0.34 | 0.32 | 0.34 | 0.35 | 0.47 | 0.43 | 0.48 | 0.51 | 0.56 | 0.60 | 0.34 / 0.49 / 0.94 | (MIXED) |

The network's normalized repair of the wrong cells left uncorrected rises from 0.06–0.12 at f = 0.1 to 0.40–0.52 at f = 0.5, 0.72–0.82 at f = 0.7 and 0.81–0.87 at f = 0.8 on the two wider models (0.03 to 0.39 on 128). The share of edited cells still correct rises from 0.56–0.64 at f = 0.1 to 0.97–0.99 at f = 1.0.

**Letters.** GRADUAL on 256, MIXED on 192 (width-dependent; neither SHARP). On 256 the span is widened by a step of 0.08 at f = 0.1; without it the rise runs from about 0.3 to 0.8 on both widths.

**Predictions scored.** SHARP (0.45) MISS; MIXED (0.30) hit on 192; GRADUAL (0.25) hit on 256. f50 between 0.4 and 0.8 on both lettered widths (0.6) HIT (0.53, 0.50). The network repairs uncorrected cells at R ≥ 0.5 at f ≥ f90 (0.7) HIT (0.81–0.87 at f = 0.8). Attention 128 UNDEFINED (0.4) MISS: its range is 0.29, above the 0.20 floor.

**The registered sentences that apply.** GRADUAL (256): "The more of the error is corrected at once, the more the network completes, without a critical share." MIXED (192): f10, f50 and f90 reported with both readings. Measured on both: completion and the network's own repair of the rest follow a smooth cooperative rise centred at half of the errors corrected (f50 0.50–0.53), steepest between about 0.4 and 0.8; the solution's basin in the network's own dynamics has a graded, not a sharp, boundary.

**Reading.** With P15, P16 and P18: a correction written into the network's state is accepted in proportion to how much of the configuration it changes, and the network finishes the rest itself once roughly half of the errors are corrected. The acceptance is cooperative and majority-like, centred at half, and graded rather than switch-like. This is the quantitative form of the collective account: errors are held by the bulk of the configuration, and the network's correction follows the bulk.
