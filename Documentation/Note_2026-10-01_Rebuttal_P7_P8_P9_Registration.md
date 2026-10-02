# Note 2026-10-01 — P7, P8, P9: cell-level repair by local evidence, the completing step's intact control, and identical-answer state swaps (registration, before any row)

**Goal (page one).** A MEASUREMENT for all three. No accuracy target; $0; the Mac's CPU; inference only on banked checkpoints (P8, P9) or saved records only (P7); nothing trained.
- Plan: `Plan_2026-10-01_Rebuttal_Experiments.md` §2. These items complement P3–P6 and do not repeat them.
- Tools: `tools/rebuttal_p7.py`, `tools/rebuttal_p8.py`, `tools/rebuttal_p9.py`. Each imports the existing pipelines unchanged and edits no shared tool.
- Outputs: `runs/analysis/rebuttal_20261001e/` (P7), `…f/` (P8), `…g/` (P9).
- P8 and P9 run at `nice -n 10` beside P3 and never wait on or touch P3–P5's queues.
- Written before any tool is built and before any row exists.

---

## P7 — Is correction local refutation? (saved records only)

**Why.** P1 and P1c showed that matched-count corruptions are repaired slowly when the wrong digits are mutually consistent, and quickly when they are refuted by givens or by each other. That is a grid-level statement. It leaves three things open:
- whether the update acts *cell by cell* on contradictions the displayed grid shows;
- whether the same holds in the models' own trajectories, not only from injected grids. A Sep 25 exploratory grid-level check, unregistered and unquotable, found that coupled errors did not persist longer there.
- whether the offsetting revisions of §3 are contradictions resolved on the wrong side.

**Classes.** These are defined on the displayed grid g_t at the start of a transition t → t+1, for empty cells only.

Each wrong digit is exactly one of:
- **given-refuted:** it equals a given in its row, column or box;
- **peer-refuted:** not given-refuted, but it equals the current digit of another non-given cell in a shared unit;
- **unrefuted:** it equals nothing in its units.

Refuted means given-refuted or peer-refuted. A correct empty-cell digit is **conflicted** if a wrong peer digit in a shared unit equals it, and **clear** otherwise.

**Outcomes at t+1.** A wrong cell is *repaired* (now correct), *switched* (another wrong digit) or *kept*. A correct cell is *broken* (now wrong) or kept.

**Populations** (all saved, nothing re-run):

1. *Injected, first transition.* The three attention receivers' P1/P1c/study chunks, transition 0 → 1. The classes are computed on the injected grid itself. Families:
   - `uniform_0…3` (study);
   - `legal_random_0…3` and `legal_at_eqr_0…3` (P1);
   - `consistent_random_0…3` (P1c);
   - `eqr` (study).

   Restricted to the 438 source-error puzzles.
2. *Natural trajectories.*
   - (a) The study's fixed-start runs of Attention 128/192/256 on the 512 stratified repair puzzles, 16 iterations.
   - (b) The inside-completion records of C5, SA128 and EQR on 256 stratified puzzles, 48 slow readouts, used at outer-iteration readouts j = 3, 6, …, 48.

   Transitions are taken from iteration 1 up to each puzzle's first exact iteration (excluded), or to the last iteration if it never completes.

**Rules (confirmatory).** H is the repair probability of refuted wrong cells divided by that of unrefuted wrong cells. It is pooled over cells, and computed only within grids that contain both classes at that transition. Intervals come from 2,000 puzzle-level bootstrap resamples, seed `[20261001, 7]`.

- **R7a** (injected, first transition, per attention receiver, pooled over families):
  - LOCAL-REFUTATION if H ≥ 2 with the interval's lower end above 1 on all three receivers;
  - NO-ASYMMETRY if H ≤ 1.2 on all three;
  - MIXED otherwise.
- **R7b** (natural trajectories, per model, all six models):
  - the same letters, applied to model-produced errors;
  - UNDEFINED for a model with fewer than 200 eligible wrong-cell observations in either class.
- **R7c** (offsetting revisions, natural trajectories, per model). Let s be the share of right-to-wrong events that occur at conflicted correct cells, and q = P(broken | conflicted) / P(repaired | wrong cell that conflicts with it), using the same transitions.
  - CONFLICT-DRIVEN if s ≥ 0.7 and q ≥ 0.5;
  - NOT-CONFLICT-DRIVEN if s ≤ 0.4;
  - MIXED otherwise.
  - The base rate (the share of correct cells that are conflicted) is reported beside s.

**Exploratory (no letter).** H split into given-refuted and peer-refuted; H by family; H by iteration; switches against keeps by class; the uniform 512-puzzle traces where saved grids allow.

**Predictions and credences.**
- R7a: LOCAL-REFUTATION 0.80, MIXED 0.15, NO-ASYMMETRY 0.05.
- R7b: LOCAL-REFUTATION 0.45, MIXED 0.35, NO-ASYMMETRY 0.20.
- R7c: CONFLICT-DRIVEN 0.55, MIXED 0.30, NOT-CONFLICT-DRIVEN 0.15.

**Wording under each outcome.**
- **LOCAL-REFUTATION on both R7a and R7b:** "Within a single grid, wrong digits that the grid itself contradicts are repaired at least twice as often per iteration as wrong digits nothing contradicts, both from injected grids and in the models' own trajectories: the learned update acts where a contradiction is visible, and mutually consistent errors are invisible to it."
- **LOCAL-REFUTATION on R7a only:** "The asymmetry holds for the first update from an injected grid but not inside the models' own trajectories, whose persistent errors are not distinguished by visible contradictions."
- **NO-ASYMMETRY:** reported plainly, with the reading that repair speed is not set by local evidence at the cell level.
- **CONFLICT-DRIVEN:** "Most correct digits that become wrong are ones a wrong peer duplicates, and they break about as often as the duplicating error is repaired: offsetting revisions resolve contradictions on the wrong side about as often as on the right one."

Under every outcome, the classes are combinatorial facts about the displayed grid; no hidden representation is identified.

---

## P8 — The completing step's intact control and a score-preserving slow-state edit (inside-completion lens, pass C)

**Why.** On Sep 25, interventions at the start of the completing cycle κ were measured, but they never entered the manuscript because they lacked an intact control under their replay implementation:
- slow reset blocked completion for 0.93/1.00/1.00 of the cohort;
- message removal blocked 0.97/1.00;
- fast reset blocked 0.26/0.32/0.25.

Two questions remain open: whether completion needs the slow state's readout-invisible content at κ, and whether readout-visible content suffices.

**Design.** Reuse `tools/lens_inside_completion.py` unchanged: `build`, the strat-256 population, and pass C's loop. Cohorts and κ are read from the saved Sep 25 records `runs/analysis/inside_completion_20260925/{C5,SA128,EQR}.npz` (134, 122 and 108 puzzles). Each condition acts at the START of the named cycle, then the trajectory continues unedited to readout 48, at batch 64. Conditions:
- `intact`: no flag set, through the same jitted function (the missing control).
- `slow_reset@0`: re-run; must reproduce the saved Sep 25 flags.
- `perp_sham@0`: each slow-state vector h is split into h∥ (along the shared readout vector) and h⊥, then reassembled unchanged.
- `perp_random@0` and `perp_random@-3`: h⊥ is replaced by a Gaussian vector projected orthogonal to the readout and rescaled to ‖h⊥‖, per field and cell. This keeps every digit score and every vector norm. Seeds are `[20261001, puzzle ID, 81, cycle]`.

C5 and SA128 receive all five conditions. EQR receives `intact` and `slow_reset@0` only, because its readout is a token matrix and P8 does not edit it.

**Gates.**
- (1) `slow_reset@0` reproduces the saved Sep 25 exact-at-κ and exact-at-48 flags exactly, per model.
- (2) For each perp edit, the maximum absolute digit-score change is below 1e-3, recorded per edit.
- (3) `perp_sham@0` exact-at-κ flags equal `intact`'s on at least 99 % of the cohort.

If (1) fails, the run continues and the report says SEP-25 FLAGS NOT REPRODUCED.

**Rules (confirmatory).**
- **Admission:** `intact` exact at κ on at least 98 % of the cohort, per model. If it is lower, the report says REPLAY-DRIFT and uses `intact` as the denominator anyway.
- **R8 (C5 and SA128):** B = 1 − P(exact at κ | `perp_random@0`) / P(exact at κ | `intact`).
  - NEEDS-HIDDEN-STATE if B ≥ 0.5;
  - READOUT-SUFFICES if B ≤ 0.1;
  - MIXED otherwise.
- **Re-scoring (no new letter):** B for the Sep 25 conditions (fast reset, slow reset, message removal, at κ and κ−3) against `intact`. The Sep 25 registered S8 thresholds are restated beside the values.

**Exploratory.** `perp_random@-3`: exact at κ, median delay to the first exact readout, and exact at 48. Delay distributions are given for every condition.

**Predictions and credences.**
- Admission passes: 0.85.
- Gate (1) exact: 0.90.
- Sham gate: 0.90.
- R8 (both models): NEEDS-HIDDEN-STATE 0.50, MIXED 0.30, READOUT-SUFFICES 0.20.
- `perp_random@-3` mostly delays completion rather than preventing it (exact at 48 ≥ 0.8): 0.7.

**Wording under each outcome.**
- **NEEDS-HIDDEN-STATE:** "Replacing only the slow-state features the readout cannot see, once, at the start of the completing cycle, blocks most completions that would have occurred there, with every digit score unchanged: the completing step uses state beyond the displayed answer."
- **READOUT-SUFFICES:** "The completing step proceeds from the displayed scores alone: the readout-invisible features matter before completion, not at it."
- **MIXED:** B reported with both readings.

Under every outcome:
- with admission passed, the Sep 25 interventions become admissible as intact-controlled measurements;
- no feature content is identified;
- the edit uses random directions, not learned ones.

---

## P9 — Identical answers, different futures: state swaps at the 94k checkpoint

**Why.** §5.1 of the manuscript shows one puzzle where the fixed and the Gaussian start both display the correct grid at iteration 3 and only the Gaussian continuation stays correct at 4. Appendix A.5 notes that message and state interventions were not performed at 94k. The test: when two trajectories display the same grid, does the next step follow the grid, the slow state's readout-invisible content, the full slow state, or the fast state?

**Design.**
- MLP 192 seed 0 at 94k, EMA (`runs/_c5l_pull/x/runs/pretrainchamp_C5/ckpt_094000.pkl`).
- The same 128 validation puzzles, at batch 128 (P5's gate shows this batch composition reproduces the saved flags).
- Two trajectories per puzzle:
  - F, the fixed start;
  - G, the independent Gaussian start with the manuscript's seeds `[4242, i, 0, 7]`.
- Loading follows `tools/rebuttal_p5.py`'s `Runner.load` unchanged. The loop replicates `eval_sudoku_extreme.run_batch` step by step, with a per-puzzle pre-step hook.

*Primary cases.* Puzzles where F and G are both exact at some iteration t < 16 and only G is exact at t+1. Each puzzle uses its earliest such t (the manuscript's selection population).

*Controls.* Puzzles where both are exact at some t < 16 and both stay exact at t+1, each at its earliest such t.

**Swaps,** applied to the carry entering iteration t+1, for the hooked puzzles only; every other puzzle in the batch runs unedited:
- `F<-hG`: F's slow state replaced by G's;
- `F<-perpG`: F keeps its own readout-parallel component and takes G's readout-orthogonal component (scores unchanged);
- `F<-lG`: F's fast state replaced by G's;
- `F<-zG`: both states replaced (a mechanics check: it must reproduce G's next flags);
- the four mirrored swaps on G (`G<-hF`, `G<-perpF`, `G<-lF`, `G<-zF`);
- `F<-sham`: F's slow state decomposed and reassembled.

**Outcomes.** Exact at t+1 and at 16 for each hooked puzzle.

**Gates.**
- (1) The replicated loop reproduces the saved 94k flags of F and G exactly at every iteration (`paper/code/evidence/initialization/s094000.npz`, the `cold` and `ri` rows).
- (2) `F<-zG` gives G's exact flag at t+1 on every primary case, and `G<-zF` gives F's.
- (3) `F<-sham` reproduces F's flags at t+1 on at least 99 % of hooked puzzles.
- (4) The perp swaps change no digit score by more than 1e-3.

**Rules (confirmatory, on the primary cases).** Rescue(X) is the share of primary cases where F under swap X is exact at t+1. Transfer(Y) is the share where G under the mirrored swap Y is not exact at t+1.
- STATE-CARRIES if Rescue(`F<-perpG`) ≥ 0.6 and Transfer(`G<-perpF`) ≥ 0.6.
- FULL-SLOW-ONLY if not STATE-CARRIES and Rescue(`F<-hG`) ≥ 0.6 and Transfer(`G<-hF`) ≥ 0.6.
- FAST-CARRIES if neither of the above and Rescue(`F<-lG`) ≥ 0.6 and Transfer(`G<-lF`) ≥ 0.6.
- NONE otherwise.
- UNDEFINED if fewer than 20 primary cases exist or a gate fails.

**Exploratory.** The endpoint (exact at 16) under every swap; the control cases (do swaps break stable answers?); the slow- and fast-state norms and readout margins of F and G at t.

**Predictions and credences.**
- Gates pass: 0.85.
- STATE-CARRIES 0.40, FULL-SLOW-ONLY 0.25, FAST-CARRIES 0.15, NONE 0.20.

**Wording under each outcome.**
- **STATE-CARRIES:** "At identical displayed answers, exchanging only the slow-state features the readout cannot see moves the next step's outcome to that of the donor trajectory, in both directions: the start's effect on preservation is carried by readout-invisible slow state."
- **FULL-SLOW-ONLY:** "The full slow state carries it; its readout-invisible part alone does not."
- **FAST-CARRIES:** "The fast state carries it."
- **NONE:** "No single component carries it; the effect needs both states together."

Under every outcome: one checkpoint and 128 puzzles; no feature content is identified; the 94k checkpoint remains a selected example (P5 reports its distribution).

---

## Labels and order

- **Confirmatory:** R7a, R7b, R7c, R8 (with admission), and P9's letter.
- **Exploratory:** everything else.
- **Order:** P8 builds and launches first (the longest compute, at nice 10, models in sequence C5 → SA128 → EQR). P7 is built and run while P8 computes (saved records, minutes). P9 is built, smoked through its gates, and run after (about 20–40 min).
- **For each experiment:** an Outcome section appended here, and one ledger line.

---

## P7 Outcome (2026-10-01 10:08Z; `runs/analysis/rebuttal_20261001e/report.{txt,json}`; the analyzer's hash was frozen before the run, in `frozen_sha256.txt`)

**Integrity.**
- Selftest: 10 hand-built checks, including a self-peer mutant that the deadly-pattern case catches.
- Independent cross-check after the run: a cell-by-cell loop written separately agrees with the vectorized classifier on all 3,349 cells of 60 random natural-trajectory grids (Attention 128), with 0 mismatches.
- Every chunk's puzzle ids match the input files.

**R7a — injected grids, first transition (pooled over 17 families, 438 puzzles; puzzle-cluster bootstrap).**

| receiver | P(repair), given-refuted | P(repair), peer-refuted | P(repair), refuted (pooled) | P(repair), unrefuted | H [95 %] | letter |
|---|---|---|---|---|---|---|
| Attention 128 | 0.866 | 0.636 | 0.668 | 0.450 | 1.49 [1.45, 1.53] | MIXED |
| Attention 192 | 0.918 | 0.636 | 0.676 | 0.331 | 2.04 [1.96, 2.14] | LOCAL-REFUTATION |
| Attention 256 | 0.892 | 0.634 | 0.671 | 0.349 | 1.92 [1.85, 1.99] | MIXED |

The registered all-receivers letter is **MIXED**.

**R7b — the models' own fixed-start trajectories (transitions starting before first completion).** All six models carry the same letter, so the all-models letter is **NO-ASYMMETRY**.

| model | H [95 %] | refuted vs unrefuted P(repair) | n refuted / unrefuted | H without the completing transition |
|---|---|---|---|---|
| A128 (study) | 0.95 [0.90, 0.99] | 0.329 vs 0.348 | 33,423 / 10,992 | 0.78 |
| A192 (study) | 0.88 [0.82, 0.94] | 0.331 vs 0.375 | 19,885 / 8,754 | 0.87 |
| A256 (study) | 0.94 [0.89, 0.99] | 0.397 vs 0.424 | 18,754 / 9,483 | 0.90 |
| C5 (lens) | 0.98 [0.91, 1.05] | 0.341 vs 0.349 | 14,459 / 8,571 | 0.80 |
| SA128 (lens) | 0.90 [0.84, 0.97] | 0.307 vs 0.340 | 17,802 / 6,079 | 0.77 |
| EQR (lens) | 0.96 [0.88, 1.05] | 0.254 vs 0.264 | 16,561 / 10,114 | 0.92 |

**R7c — offsetting revisions.** All six models read **MIXED**:
- s, the share of right-to-wrong events at conflicted correct cells, is 0.58–0.71, against a base rate of 0.43–0.60;
- q is 0.30–0.50: P(break | conflicted) is 0.12–0.14, while P(repair | wrong digit duplicating a correct one) is 0.26–0.40.

**Predictions scored.**
- R7a LOCAL-REFUTATION (0.80): MISS. Only Attention 192 clears H ≥ 2.
- R7b LOCAL-REFUTATION (0.45): MISS. NO-ASYMMETRY (0.20) holds on all six models.
- R7c CONFLICT-DRIVEN (0.55): MISS. MIXED (0.30) is the letter on all six.

**The registered wording that applies.** R7a is MIXED and R7b is NO-ASYMMETRY. The nearest registered sentence is the R7a-only variant: "The asymmetry holds for the first update from an injected grid but not inside the models' own trajectories, whose persistent errors are not distinguished by visible contradictions." It applies here only in a qualified form, because R7a itself is MIXED: given-refuted digits are repaired best everywhere, while the peer-refuted/unrefuted contrast is moderate.

**Exploratory (no letter; written after the letters).** Per-family repair probabilities, refuted vs unrefuted:

| family | A128 | A192 | A256 |
|---|---|---|---|
| uniform | 0.87 vs 0.85 | 0.92 vs 0.87 | 0.89 vs 0.87 |
| clue-legal | 0.70 vs 0.62 | 0.73 vs 0.63 | 0.73 vs 0.64 |
| legal at EqR's cells | 0.64 vs 0.61 | 0.62 vs 0.56 | 0.63 vs 0.57 |
| mutually consistent | 0.51 vs 0.41 | 0.47 vs 0.29 | 0.46 vs 0.30 |
| EqR's own grid | 0.38 vs 0.35 | 0.32 vs 0.12 | 0.31 vs 0.19 |

Within a family, a contradicted wrong digit is repaired only somewhat more often (1.0–2.6×). Across families the same class of cell moves far more: a contradicted digit in an EqR grid is repaired less often than an uncontradicted digit in a uniform grid. The pooled H in R7a is therefore largely composition: uniform grids are mostly refuted and mostly repaired. In the models' own runs, almost no wrong digit contradicts a given. The contradicted ones conflict with other filled cells and are repaired no more often than uncontradicted ones.

**Reading (descriptive).** A wrong digit's chance of repair is set mainly by the configuration it sits in, not by whether the displayed grid contradicts it locally. Inside the models' own trajectories, local contradiction does not predict repair at all. This weakens a cell-level local-refutation account of the repair results (P1, P1c, §4). It favours a configuration-level or state-level account, which P3, P8 and P9 probe. Offsetting revisions are only mildly concentrated where a wrong peer duplicates a correct digit.

**Limits.**
- One checkpoint per model.
- Classes are combinatorial facts about the displayed grid; repair is a one-step transition; pooled counts weight long trajectories more.
- The natural-trajectory statistic pools all pre-completion transitions; the completing transition is reported both ways.
- The 512 and 256 stratified populations overlap partly with each other but are analysed separately.

---

## P8 Outcome (2026-10-01 10:44Z; `runs/analysis/rebuttal_20261001f/report.{txt,json}`; tool hash unchanged from `frozen_sha256.txt`; run log exit 0)

**Disclosure.** The one-batch smoke on SA128 (64 of 122 cohort puzzles, 09:58Z) printed exact-at-κ rates for every condition before the full run. The rules had been fixed in this note before the smoke. Nothing was changed after it.

**Gates.**
- **Admission PASS on all three.** The intact replay through the same jitted cycle is exact at κ on 134/134, 122/122 and 108/108, and exact at readout 48 on all of them.
- **Gate 1.** The re-run slow reset at κ reproduces the Sep 25 exact-at-κ and exact-at-48 flags exactly on C5, SA128 and EQR.
- **Sham.** Agreement with intact is 1.000 on C5 and SA128.
- **Score drift.** The largest digit-score change from any edit is 8.2e-7, against a limit of 1e-3.

**R8.** The edit replaces each slow-state vector's readout-orthogonal part, once, at the start of the completing cycle κ. It keeps every score and every norm. B is the share of intact completions at κ that the edit blocks.

| model | exact at κ, intact | exact at κ, edit | B | letter | exact at 48, edit | delay quartiles (readouts) | never exact by 48 |
|---|---|---|---|---|---|---|---|
| C5 (MLP 192) | 1.000 | 0.590 | 0.410 | MIXED | 1.000 | 0 / 0 / 1 | 0 / 134 |
| SA128 (Attention 128) | 1.000 | 0.016 | 0.984 | NEEDS-HIDDEN-STATE | 0.877 | 3 / 8 / 13 | 15 / 122 |

**Exploratory: the same edit one outer iteration earlier (κ−3).**
- C5: exact at κ 0.157, at 48 0.896; delay quartiles 1 / 4 / 9 readouts; 14 never exact by 48.
- SA128: exact at κ 0.107, at 48 0.877; delay quartiles 2 / 4 / 10 readouts; 15 never exact by 48.

**The Sep 25 interventions, re-scored against the intact replay.** These are now admissible as intact-controlled measurements. B is the share blocked at κ; the last column is exact at 48.

| model | condition | at κ | at κ−3 | exact at 48 (κ / κ−3) |
|---|---|---|---|---|
| C5 | slow reset | 0.93 | 0.87 | 0.88 / 0.88 |
| C5 | message removal | 0.97 | 0.87 | 0.91 / 0.91 |
| C5 | fast reset | 0.26 | 0.60 | 1.00 / 0.90 |
| SA128 | slow reset | 1.00 | 1.00 | 0.82 / 0.91 |
| SA128 | message removal | 1.00 | 0.98 | 0.80 / 0.84 |
| SA128 | fast reset | 0.32 | 0.66 | 0.99 / 0.91 |
| EQR | slow reset | 1.00 | 0.95 | 0.87 / 0.79 |
| EQR | fast reset | 0.25 | 0.61 | 0.98 / 0.80 |

**Predictions scored.**
- Admission passes (0.85): HIT.
- Gate 1 exact (0.90): HIT.
- Sham (0.90): HIT.
- R8 NEEDS-HIDDEN-STATE (0.50): HIT on SA128. C5 reads MIXED, which had 0.30.
- The edit at κ−3 mostly delays rather than prevents (exact at 48 ≥ 0.8; 0.7): HIT on both models (0.896 and 0.877).

**The registered wording that applies.**
- **SA128 (NEEDS-HIDDEN-STATE):** "Replacing only the slow-state features the readout cannot see, once, at the start of the completing cycle, blocks most completions that would have occurred there, with every digit score unchanged: the completing step uses state beyond the displayed answer." Most blocked puzzles complete later, at a median of 8 readouts, and 12 % do not complete within the horizon.
- **C5 (MIXED):** B = 0.41, reported with both readings. Most completions survive the edit at κ, and all complete later. The same edit one outer iteration earlier blocks 84 % at κ.

Under every outcome, as registered: no feature content is identified, and the edit uses random directions.

**Reading (descriptive).**
- In both models, the readout-invisible part of the slow state in the iteration *before* completion is needed for completion to happen on time.
- In the attention model it is also needed during the completing cycle itself. In the MLP model, the completing cycle can mostly proceed from the scores and the carried fast state.
- With the intact control in place, the Sep 25 picture stands:
  - completion at κ needs the retained slow state and cross-field messages (blocked 87–100 % at κ and κ−3);
  - it mostly does not need the carried fast state at κ (25–32 %);
  - resetting the fast state one iteration earlier blocks 60–66 %, so the carried fast state matters for the approach.

---

## P9 Outcome (2026-10-01 11:11Z; `runs/analysis/rebuttal_20261001g/report.{txt,json}`, `p9.npz`; tool hash unchanged from `frozen_sha256.txt`; run log exit 0; wall 26 min at nice 10)

**Gates, all passed.**
1. The replicated `run_batch` loop reproduces the saved 94k flags of F and G exactly at every iteration: F ends at 11 of 128, G at 124.
2. Full swaps reproduce the donor's flag at t+1 on every primary case.
3. The sham agrees with F's flag at t+1 on all 75 hooked puzzles.
4. The perp swaps change no digit score by more than 8.6e-7.

There are 60 primary cases, the manuscript's selection population, and 15 controls.

**Results on the 60 primary cases** (both trajectories exact at t; only G exact at t+1 intact):

| swap | F with G's component: still exact at t+1 (rescue) | G with F's component: wrong at t+1 (transfer) | at 16: F still exact / G lost |
|---|---|---|---|
| readout-orthogonal slow part (scores kept) | **1.000** (60/60) | **0.000** (0/60) | 1.00 / 0.00 |
| full slow state | 1.000 | 0.117 (7/60) | 1.00 / 0.20 |
| fast state | 0.883 (53/60) | 0.000 | 0.80 / 0.00 |
| both states (mechanics check) | 1.000 | 1.000 | 1.00 / 0.92 |

**Controls.** On the 15 puzzles where both trajectories stay exact intact, no swap in either direction breaks the answer at t+1.

**Registered letter: NONE.** STATE-CARRIES needed rescue and transfer both ≥ 0.6 for the orthogonal-part swap; FULL-SLOW-ONLY and FAST-CARRIES the same for theirs. Transfer is 0.00 / 0.12 / 0.00, so all fail.

**Predictions scored.** Gates pass (0.85): HIT. NONE (0.20): HIT. STATE-CARRIES (0.40): MISS.

**The registered wording.** The registered NONE sentence is "No single component carries it; the effect needs both states together." It holds for the *failure*: F's loss at t+1 occurs only when both its slow-state invisible part and its fast state come from F. It does not describe the *rescue*. The record supports this sentence:

> At identical displayed answers, the fixed start's loss of a solved puzzle at the next step requires both of its hidden components. Replacing only the readout-invisible slow content with the Gaussian trajectory's, every digit score unchanged, prevents the loss in 60 of 60 cases. Replacing only the fast state prevents it in 53 of 60. The Gaussian trajectory is not broken by either of the fixed trajectory's components alone (0 of 60) and rarely by its full slow state (7 of 60).

**Reading (descriptive).** The 94k preservation failure is carried in hidden state, not in the answer. The displayed grid and every digit score can be identical while the next step's outcome differs, and either healthy hidden component restores preservation. This is the causal version of the manuscript's §5.1 example on all 60 cases of its selection population. It fills the gap Appendix A.5 names: "Message and state interventions were not performed at 94k."

**Limits.**
- One checkpoint, 128 validation puzzles, and one Gaussian draw per puzzle.
- The hybrid states are off-trajectory constructions.
- No feature content is identified.
- The asymmetry describes these two start families at this checkpoint only.
