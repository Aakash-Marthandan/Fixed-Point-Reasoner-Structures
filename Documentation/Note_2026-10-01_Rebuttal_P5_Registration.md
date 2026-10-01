# Note 2026-10-01 — P5: access robustness at the 94k checkpoint (registration, before any row)

**Goal (page one).** A MEASUREMENT. No accuracy target; $0; the Mac's CPU; saved records plus inference only on the banked MLP 192 long-run checkpoints; nothing trained. Plan: `Plan_2026-10-01_Rebuttal_Experiments.md` §2. Tool: `tools/rebuttal_p5.py` (reads `paper/code/evidence/initialization/s*.npz`; runs the 94k checkpoint through the release evaluator with the lens's conventions, `tools/lens_c5l_dynamics.py` unchanged; outputs `runs/analysis/rebuttal_20261001c/`). Registered before the build.

## Why

§5 of the manuscript reports that at the 94k MLP 192 checkpoint the fixed start returns 11 of 128 validation answers while one shared Gaussian start and the independent Gaussian starts each return 124. Outside readers called the checkpoint a post-hoc minimum of one run and asked for (a) the distribution after the 50k benchmark budget rather than one checkpoint and (b) more draws at 94k. Both are available without new training: (a) from the saved series of 38 checkpoints, (b) from sixteen further draws through the same evaluator.

## Design

**P5a — the post-50k distribution (saved records).** For every checkpoint of the series (2k–150k, 38 files) read the endpoint and ever-correct counts of the fixed start (`cold`), the independent Gaussian starts (`ri`) and the shared Gaussian start (`rifix`), plus `sym` and `anchor` where present. Report, over the 25 checkpoints at or beyond 54k: median, quartiles, min and max of each start's endpoint count; the rank of 94k among the fixed-start counts; the number of checkpoints at which the fixed start trails the independent starts by at least 50 answers, by at least 20, and by fewer than 5; the fixed start's terminal-loss counts (ever correct but wrong at 16) per checkpoint; and the manuscript's 94k numbers reproduced exactly as a gate (11 / 124 / 124 / 121 / 128 at 16; 68 ever correct for the fixed start).

**P5b — sixteen further draws at 94k (CPU inference).** On the same 128 validation puzzles, 16 iterations, EMA weights of `ckpt_094000.pkl`: eight new independent Gaussian draws, seeds `[20261001, puzzle index, j, 7]` for j = 1 … 8 through the release's `mi_z0` convention, and eight new shared Gaussian draws, seeds `[20261001, 0, j, 7]` broadcast across puzzles. Also four further draws (two independent, two shared, the same seed convention with j = 1, 2) at the neighbouring checkpoints 90k and 98k and at the 46k benchmark checkpoint (where the fixed start already succeeds), as context. Each start condition is one 128-puzzle, 16-iteration pass of MLP 192 on the CPU; the 94k set is 19 passes and the context 15, so the run is queued behind P3 and P4 on PID waits rather than alongside them. Gate: the fixed start and the original `ri` and `rifix` draws (seed 4242) re-run in-process reproduce the saved `cold_ex`, `ri_ex` and `rifix_ex` flags of `s094000.npz` exactly at every iteration.

## Quantities

Per draw: endpoint count of 128, ever-correct count, number of fixed-start successes preserved (fixed-start endpoint successes also correct under the draw), losses after first success. Summary: the sixteen draws' median, min and max at each checkpoint.

## Rule (confirmatory, P5b at 94k)

ROBUST if every one of the sixteen new draws returns at least 100 of 128 at iteration 16 and preserves all 11 fixed-start successes; FRAGILE if any draw returns at most 68 (the fixed start's ever-correct count); MIXED otherwise; UNDEFINED if the gate fails. P5a carries no letter; its sentence is descriptive.

## Predictions and credences (before any row)

ROBUST 0.80; MIXED 0.17; FRAGILE 0.03. P5a: the post-50k median fixed-start endpoint below 80 and the independent-start median above 120: 0.75 (the manuscript's Table 11 suggests medians near 69 and 124). 94k is the minimum fixed-start count after 50k: 0.6.

## Wording under each outcome

- **ROBUST.** "At 94k, sixteen further Gaussian draws, eight independent and eight shared, each return N₁–N₂ of 128 against the fixed start's 11, preserving every fixed-start success: the access contrast is a property of the start family, not of two draws." With P5a: "Across the 25 checkpoints after the 50k budget, fixed starts return a median of X (range a–b) and Gaussian starts a median of Y (range c–d); 94k is the k-th lowest fixed-start count."
- **MIXED / FRAGILE.** The draw-level spread is reported and the §5 sentence is narrowed to the draws that support it.
Under every outcome: the 94k checkpoint stays a selected example; the series statement replaces "one checkpoint" with the distribution; cross-device validation of the Gaussian starts remains for the pod set (G3).

## Labels and plan

Confirmatory: the P5b letter and the P5a reproduction gate. Exploratory: the neighbouring checkpoints and 46k. Build after this note; selftest and a smoke (the gate on 16 puzzles) before the run; the run is minutes on the Mac and does not wait for P3.

**Gate check before the run (2026-10-01 09:46Z; `runs/analysis/rebuttal_20261001c/gate_pre_run_s094000_n128.json`).** A 16-puzzle smoke reproduced the saved `ri` and `rifix` flags exactly but not the fixed start's. At the registered batch of 128 puzzles the fixed start, the original independent draws and the original shared draw all reproduce the saved flags with zero differing flags at every iteration (11 / 124 / 124 at the endpoint, 68 ever correct for the fixed start). The smoke difference is batch composition acting on a numerically sensitive trajectory, as the manuscript's limitations state; the gate stands as registered, at the full batch, and P5b's rows are read only against it.

---

## P5 Outcome (2026-10-01 17:48Z; `runs/analysis/rebuttal_20261001c/report.{txt,json}`; every gate passed; no shared tool edited)

**Integrity.** At each of the four checkpoints the fixed start and the original seed-4242 independent and shared draws, re-run in-process at the registered batch of 128, reproduce the saved per-iteration flags with zero differing flags (the gate record inside each step file). The series gate reproduced the manuscript's 94k numbers. An independent recount from the step files agrees with the report on every count and on the letter. Deviation from the plan line only: the run was started by hand at 16:56Z once the CPU freed, ahead of the PID queue (same tool and output path; the queue's later pass skips the finished steps).

**P5a, the saved series (descriptive).** Over the 25 checkpoints from 54k to 150k the fixed start returns a median of 69 answers of 128 (quartiles 38 and 114, range 11 to 125), reaches a median of 105 at some iteration, and loses 691 reached answers by iteration 16 in total; the independent Gaussian starts return a median of 124 (quartiles 123 and 125, range 120 to 127) and the shared start 124 (123 and 125, 121 to 126), with no answer lost after being reached at any checkpoint for either family. The independent-minus-fixed gap has median 56; it is at least 50 at 13 checkpoints, at least 20 at 17, and under 5 at 2. The 94k checkpoint is the lowest fixed-start count of the 25.

**P5b, further draws.**

| checkpoint | fixed start (ever correct) | original draws, independent / shared | new draws | new draws min / median / max | all fixed-start successes kept in every new draw | losses after success |
|---|---|---|---|---|---|---|
| 94k | 11 (68) | 124 / 124 | 16 | 122 / 124 / 126 | yes | 0 |
| 90k | 12 (87) | 124 / 126 | 4 | 124 / 124.5 / 126 | yes | 0 |
| 98k | 32 (56) | 125 / 121 | 4 | 126 / 126 / 127 | yes | 0 |
| 46k (benchmark) | 126 (126) | 124 / 127 | 4 | 124 / 125 / 126 | no: at the near-ceiling checkpoint the draws solve a slightly different set | 0 |

**Letter: ROBUST** at 94k (every one of the sixteen new draws returns at least 100 and keeps all eleven fixed-start successes).

**Predictions scored.** ROBUST (0.80) HIT. P5a medians below 80 and above 120 (0.75) HIT (69 and 124). 94k the post-50k minimum (0.6) HIT.

**The registered sentences.** "At 94k, sixteen further Gaussian draws, eight independent and eight shared, each return 122 to 126 of 128 against the fixed start's 11, preserving every fixed-start success: the access contrast is a property of the start family, not of two draws." With P5a: "Across the 25 checkpoints after the 50k budget, fixed starts return a median of 69 (range 11 to 125) and Gaussian starts a median of 124 (range 120 to 127); 94k is the lowest fixed-start count; the fixed start loses reached answers at most checkpoints, the Gaussian starts at none."

**For the manuscript.** The 94k sentence of §5 becomes the series statement. The objection that 94k is a post-hoc minimum is conceded as a fact and answered by the series: after the budget the fixed start's failure is persistent across checkpoints and is mostly a failure to keep what it reaches (ever-correct median 105 against an endpoint median of 69), while either Gaussian start family removes both terms at every checkpoint. The 94k checkpoint stays a selected example. Cross-device validation of the Gaussian starts remains for the pod set.
