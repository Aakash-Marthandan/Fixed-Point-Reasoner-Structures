# Note 2026-10-01 — P6: the common accounting of returned-answer failure (registration, before any row)

**Goal (page one).** An ANALYSIS OF SAVED RECORDS. No model is run. Plan: `Plan_2026-10-01_Rebuttal_Experiments.md` §2. Tool: `tools/rebuttal_p6.py` (reads `paper/code/evidence/selection/*.npz` with their `.protocol.json`, `paper/code/evidence/initialization/s*.npz`, the study's trajectory chunks, and `paper/code/evidence/arc/*.jsonl`; outputs `runs/analysis/rebuttal_20261001d/`). Registered before the build.

## Why

The manuscript's sections answer different questions about one inference process: whether a correct answer is reached, whether it is kept, whether it is available to the return rule, and whether the rule returns it. Outside readers asked for one ledger that places every result in the same frame. For a fixed model, dataset and inference protocol let U be the fraction of problems with a correct prediction anywhere in the observed trajectories, C the fraction with a correct candidate in the set the return rule chooses from, and A the fraction whose returned answer is correct. When the rule chooses from that set, A ≤ C ≤ U, and

1 − A = (1 − U) + (U − C) + (C − A),

read as: not reached in the tested computation; reached but unavailable at selection; available but not selected. This is accounting, not a theorem about reasoning; where a record holds only endpoints, U is marked unmeasured, never inferred.

## Configurations and records

| configuration | record | U | C | A |
|---|---|---|---|---|
| Our TRM, 20,000 test puzzles, depth-64 nested Gaussian banks: baseline seed 0, baseline seed 1, answer anchors only, random starts only | `selection/trm_*.npz` (`mi_exact_k`, `mi_resid_k`), protocols (`vote_at_k`) | unmeasured (endpoints only) | correct candidate among the first k | minimum residual among the first k (ties → earliest), k = 1, 2, …, 128; the fixed start's own endpoint as the k = 0 row |
| Benchmark attention 128 / 192 / 256, 5,000 test puzzles, depth 64, k ≤ 128; the earlier Attention 256 recipe arms, k ≤ 32 | `selection/attention_*.npz`, protocols | unmeasured | as above | as above |
| Benchmark attention, full test, one fixed-start trajectory | the manuscript's depth records (`evidence/depth`, any-loss counts) | ever correct through 64 | = A (one candidate) | endpoint |
| MLP 192 long run, 128 validation puzzles, depth 16, per start family | `initialization/s*.npz` (per-iteration flags) | ever correct through 16 | = A per family (one trajectory per family) | endpoint |
| Released TRM on ARC, 419 queries: fixed start depth 16 and 64; the eight-start Gaussian bank | `arc/cold.jsonl`, `arc/restarts.jsonl` | ever correct along the fixed-start trajectory | fixed endpoint; the bank's correct candidates | endpoint; minimum residual in the bank |

For every bank the protocol's recorded `vote_at_k` is reported beside A_k. Checked in the bank tool before the build: it is **verifier selection**, the fixed start if exact or otherwise any draw among the first k that passes the row, column and box check, not a majority vote; the majority vote cannot be computed because wrong grids were not saved, as the manuscript states.

## Gates (reproduction of the manuscript's counts from the records)

Baseline seed 0 at k = 128: C = 19,941, A = 18,496, missed 1,445; at k = 1: 18,425 / 18,425; anchors only: 19,972 / 7,645; random starts only: 19,663 / 19,659; the attention banks' 8 → 128 gained/lost 2/0, 6/1, 1/2; the recipe arms' missed counts 1, 69, 0, 132; the 94k MLP row 68 ever / 11 endpoint for the fixed start; ARC 138 ever / 128 endpoint at depth 16, bank coverage 134/134/135/136 and selection 134/133/133/131 at k = 1/2/4/8. Any mismatch stops the report.

## Outputs

`accounting.json` and `accounting.md`: one row per configuration and k with U (or "unmeasured"), C, A, vote, and the three gap terms; the overview-figure data (the gap terms per configuration as fractions). No letter; the registered deliverable is the table with every cell traced to a record.

## Predictions (descriptive, before any row)

For the benchmark attention models the C − A term at k = 128 is below 0.1 % and U − C is zero on the full fixed-start test; for the TRM recipes C − A ranges from 0.02 % (random starts only) to over 60 % (anchors only); the vote selector closes most of C − A for the baseline and not for the anchors-only arm; on ARC 1 − U dominates every term.

## Plan

Build after this note; selftest on synthetic banks (the identity, tie handling, the k-prefix minimum); run (seconds); the Outcome appended here; the figure drafted from `accounting.json` for the revision.

---

## Outcome (2026-10-01 09:41Z; `runs/analysis/rebuttal_20261001d/accounting.{md,json}`; all 17 reproduction gates passed; no model run)

**Correction before the build, recorded here.** The protocols' `vote_at_k` is verifier selection (the fixed start if exact, otherwise any draw among the first k that passes the constraint check), not a majority vote; the column is labelled accordingly and the registered sentence about "the vote selector" is read as the verifier. Two evidence file names for the earlier Attention 256 arms are the reverse of their contents; the protocols' checkpoint tags (SA256O at 22k = changed batch and rate; SA256L at 20k = no damping or noise) identify the arms and the gates use them.

**The ledger (fractions of each configuration's problems).**

| configuration | 1 − U (not reached) | U − C (reached, lost) | C − A (available, not selected) | 1 − A |
|---|---|---|---|---|
| Attention 128 / 192 / 256, full test, fixed start, depth 64 | 0.52 % / 0.38 % / 0.38 % | 0 / 0 / 0 (temporary losses 0 / 4 / 7, all recovered) | 0 (one candidate) | 0.52 / 0.38 / 0.38 % |
| MLP 192 at 94k, fixed start, 128 puzzles | 46.88 % | 44.53 % | 0 | 91.41 % |
| MLP 192 at 94k, independent / shared Gaussian start | 3.12 / 3.12 % | 0 / 0 | 0 | 3.12 / 3.12 % |
| Our TRM baseline seed 0, k = 128 | unmeasured | unmeasured | 7.22 % (verifier: 0.29 % total) | 7.52 % (1 − C = 0.29 %) |
| Our TRM baseline seed 1 / anchors only / random starts only, k = 128 | unmeasured | unmeasured | 46.12 % / 61.63 % / 0.02 % | 46.36 / 61.77 / 1.71 % (1 − C = 0.24 / 0.14 / 1.69 %) |
| Attention 128 / 192 / 256 banks, k = 128 | unmeasured | unmeasured | 0.00 / 0.02 / 0.06 % | 0.04 / 0.02 / 0.08 % |
| Earlier Attention 256 arms at k = 32: reference / no starts or anchors / changed batch and rate / no damping or noise | unmeasured | unmeasured | 0.02 / 1.38 / 0.02 / 0.00 % | 0.08 / 1.42 / 2.66 / 0.04 % (1 − C = 0.06 / 0.04 / 2.64 / 0.04 %) |
| Released TRM on ARC, fixed start, depth 16 | 67.06 % | 2.39 % | 0 | 69.45 % |
| Released TRM on ARC, eight-start bank, k = 8 | unmeasured | unmeasured | 1.19 % | 68.74 % (1 − C = 67.54 %) |

**Reading.** For the benchmark attention models every term but "not reached" is zero or within a tenth of a percent: more computation is reliably converted into returned answers. The 94k fixed start fails mostly by losing what it reached (44.5 %) and secondarily by not reaching (46.9 %), while either Gaussian start removes both terms. Across the TRM recipes the selection term runs from 0.02 % to 61.6 % while the reach term stays between 0.14 % and 1.69 %: the residual selector, not candidate generation, decides the returned accuracy; the verifier selector closes the selection term in every bank. Among the earlier Attention 256 arms, removing starts and anchors raises the selection term from 0.02 % to 1.38 %, and the changed batch and rate raises the reach term to 2.64 % while leaving selection intact. On ARC the reach term dominates (67 %) and selection adds about one point.

**Predictions scored.** Benchmark attention C − A below 0.1 % at k = 128: HIT; U − C zero on the full test: HIT; TRM recipes' C − A spanning 0.02 % to over 60 %: HIT; the vote selector closing the baseline's gap but not the anchors-only gap: not resolvable as written (no majority vote in the records); the verifier closes both; ARC dominated by 1 − U: HIT.

**For the manuscript.** The ledger is the overview figure's data; the page-one framework sentence can name the three terms; Table 14 moves into the main text with the selection term as its organizing column; the U column stays unmeasured for every bank until G2 records intermediate readouts.
