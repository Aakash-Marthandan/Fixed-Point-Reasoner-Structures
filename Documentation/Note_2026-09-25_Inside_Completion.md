# Registration: inside the completing cycle — gradual hidden progress or a collective switch? (2026-09-25 ~15:45Z, before the data)

**Why.** The PI asked whether learned relaxation with memory explains abrupt completion (Ren & Liu's per-sample "grokking"), then asked for local runs that look inside the iterations' fast updates to narrow it down. An exploratory grid-level check the same day (session scratchpad; recorded in memory) failed all four of its predictions: just before completion the decoded grid is conflict-rich, not a set of mutually consistent errors, and Ren & Liu's conflict count rises to a plateau and drops to zero only at completion. Two hidden-state accounts remain, and decoded grids cannot separate them:
- **H, gradual hidden progress:** evidence for the correct digits accumulates below the readout threshold (in the slow state's scores, or in the fast state), and many cells cross the threshold together.
- **C, a collective switch:** the hidden state stays far from the solved state and moves a large distance within one cycle, the kind of coordinated revision the paper's conjecture names (and the "sudden leap" Ren & Liu describe in HRM).

**Goal.** A MEASUREMENT; no accuracy target. $0; the Mac's CPU; banked checkpoints; inference only; nothing trained. Not for the 2026-09-26 submission.

**Known before this registration.** Completion is abrupt at outer-iteration resolution (median about 40 % of empty cells corrected in the completing iteration) and still at inner-cycle resolution (C5: median 25 % in one cycle, i.e. one slow update; saved C_ files). The fast–slow probe (Note_2026-09-19_Fast_Slow_Probe.md): after iteration 1 the fast loop contracts a perturbation of its own state within a cycle; on not-yet-solved puzzles 23–53 % of it reaches the slow state; resetting the fast state before every iteration costs 2–4 points. The paper's 94k example: identical correct readouts can have different continuations. Saved digit logits per OUTER iteration exist for C5, EQR (runs/analysis/commit_validity_20260919/); they have not been analyzed for margins. NOT known: any score, distance or fast-state quantity inside the completing cycle.

**What.** `tools/lens_inside_completion.py`: the model's own segment replicated one fast update at a time (the fast–slow probe's step function and loader). Models: C5 (MLP 192, 46k, the benchmark), SA128 (Attention 128, 42k, the benchmark), EQR (released weights, ported). Puzzles: the 256 rating-stratified test puzzles of the fast–slow and validity lenses; the fixed start; no noise; 16 iterations = 48 slow updates ("readouts" j = 1..48; the evaluator's readouts are j = 3, 6, …, 48).
- **M1 slow-state scores.** After each slow update, per empty cell: the rank of the correct digit among the nine digit scores (1 = decoded) and the margin m = score(correct) − max score(other digits); the relative slow-state change ‖h_j − h_{j−1}‖ / ‖h_{j−1}‖.
- **M2 shadow readouts of the fast state.** After fast update k = 1..6 of each cycle: apply the slow update NOW, h′ = U(h, l_k), and read it out (error on empty cells; margins). At k = 6 this is the actual next slow state (an internal gate).
- **M3 direct fast readout.** The same linear readout applied to l_k (secondary; the readout was trained on h only).
- **M4 distance to the solved state.** ‖h_j − h_48‖ / ‖h_48‖ along completing trajectories (a second plain pass).
- **M5 interventions**, applied to one cycle only, per puzzle: at the start of the completing cycle κ (the cycle whose slow update first yields the exact grid), (a) the fast state reset to its initial buffer, (b) the slow state reset to its initial buffer, (c) cross-field messages removed for that cycle (each block's field-coupling matrix set to zero, as in the paper's message removal); controls: the same at the start of cycle κ − 3 (one outer iteration earlier). Outcomes: exact at readout κ; exact at readout 48. Inference-time interventions outside the training distribution, labeled so.

**Cohort.** Puzzles whose first exact slow readout is κ ≥ 7 (not solved within two outer iterations), exact at 48. W = empty cells wrong at readout κ − 1; P = cells wrong at every readout κ − 4 … κ − 1 (persistently wrong through the last outer iteration).

**Statistics and the accounts' predictions (fixed now; medians over cohort puzzles unless stated).**
- **S1 closure** = median over puzzles of [m(κ−1) − m(κ−4)] / [0 − m(κ−4)], m = the puzzle's median margin over P. **S2 rank-2 share** = the share of W whose correct digit ranks second at κ − 1 (pooled).
  H: S1 ≥ 0.5 AND S2 ≥ 0.5. C: S1 ≤ 0.2 AND S2 < 0.5. Otherwise MIXED.
- **S3 displacement ratio** = median over puzzles of ‖h_κ − h_{κ−1}‖ / median(‖h_j − h_{j−1}‖, j = κ−5 … κ−1), each relative to the earlier state.
  H: S3 ≤ 1.5. C: S3 ≥ 2. Otherwise MIXED.
- **S4 approach** = median of 1 − d(κ−1)/d(κ−6), d = distance to h_48; **S5 jump** = median of 1 − d(κ)/d(κ−1).
  H: S4 ≥ 0.3. C: S4 < 0.1 AND S5 ≥ 0.5.
- **S6 built within the cycle** = share of cohort puzzles whose shadow error after the FIRST fast update of cycle κ is ≥ half the readout error at κ − 1.
  C: S6 ≥ 0.5. H: S6 < 0.5.
- **S7 earlier availability** = share of cohort puzzles with an exact shadow at some fast update of cycle κ − 1 (the fast state reached the solution one cycle before the slow state took it). Descriptive; credence 0.25 that it exceeds 0.1.
- **S8 interventions at κ** (share of cohort puzzles whose completion at κ is blocked):
  slow reset ≥ 0.9 (credence 0.9; predicted by both accounts, not discriminating);
  fast reset ≥ 0.5 would mean the progress or the new configuration sits in the fast state carried across cycles (credence 0.35);
  message removal ≥ 0.9 would mean the completing update itself needs cross-field exchange (credence 0.7).
  Controls at κ − 3: reported as the share still exact at κ and at 48.

**My credences.** H on S1–S2: 0.35; C on S1–S2: 0.35; MIXED: 0.30. S3 ≥ 2: 0.5; S3 ≤ 1.5: 0.3. S6 ≥ 0.5: 0.6.

**What each outcome would mean (written before the data).**
- **H on S1–S4:** abrupt completion is mainly a threshold effect of gradual hidden progress; the conjecture's coordinated revision is not needed to explain the abruptness, though memory and communication may be how the progress accumulates.
- **C on S1–S6:** a collective reorganization of the hidden state within one cycle; consistent with the conjecture's coordinated revision and with Ren & Liu's leap. If message removal and the slow reset at κ both block completion, communication and the retained slow state are necessary for the switch itself (a dependency, not an identified content).
- **MIXED or split across models:** reported per model; no mechanism word.
- Under every outcome: one checkpoint per model, 256 stratified puzzles, distances and linear readouts rather than identified contents; no fixed-point or energy claim; nothing enters the paper without a separate check.

**Gates.** (1) Slow readouts at j = 3t equal the saved validity-lens digits on ≥ 99.9 % of empty cells at t = 1 and on solved puzzles through t = 4 (C5, EQR); SA128 against the evaluator loop (`lens_corpus_normalized.run_iters`) on the same puzzles. (2) The shadow at k = 6 equals the next slow readout on 100 % of cells. (3) Selftest on a toy map with known behavior, and mutants of the margin, rank and masking helpers killed.

---

## Outcome (2026-09-25 ~17:05Z; `runs/analysis/inside_completion_20260925/{report.txt, report.json}`; all three models; every gate passed)

**Gates.** Agreement with the reference at iteration 1 / on solved puzzles through 4: C5 99.993 % / 100 %; SA128 100 % / 100 % (against the evaluator's own loop); EQR 99.986 % / 100 %. Shadow after fast update 6 equals the next slow readout (max |diff| 0) and the distance pass replays pass A exactly (0) on all three. Exact at readout 48, probe vs reference: C5 247 vs 241 (the corpus probe recorded the same pair), SA128 241 vs 242, EQR 229 vs 225: late unsolved trajectories are numerically sensitive; the gates are defined at iteration 1 and on solved puzzles. Cohorts: C5 134, SA128 122, EQR 108.

| Statistic (rule) | C5 (MLP 192) | SA128 (Attention 128) | EQR |
|---|---|---|---|
| S1 closure / S2 correct digit second (H ≥ .5 & ≥ .5; C ≤ .2 & < .5) | .41 / .64 MIXED | .39 / .59 MIXED | .46 / .67 MIXED |
| S3 completing / plateau update (H ≤ 1.5; C ≥ 2) | 1.93 MIXED | 3.23 C | 2.72 C |
| S4 approach / S5 jump (H: S4 ≥ .3; C: S4 < .1 & S5 ≥ .5) | .08 / .37 NEITHER | .02 / .58 C | .09 / .47 NEITHER |
| S6 built within the completing cycle (C ≥ .5) | .72 C | .72 C | .61 C |
| S7 exact shadow already in cycle κ − 1 | 0 | 0 | 0 |
| Blocked at κ: slow reset (predicted ≥ .9) | .93 | 1.00 | 1.00 |
| Blocked at κ: fast reset (≥ .5 = the change sits in the carried fast state) | .26 | .32 | .25 |
| Blocked at κ: messages removed (predicted ≥ .9) | .97 | 1.00 | n/a |
| Still exact at κ after the same at κ − 3: slow / fast / messages | .13 / .40 / .13 | .00 / .34 / .02 | .05 / .39 / — |

First fast update with an exact shadow in the completing cycle: quartiles 2 / 3–3.5 / 5 on all three (half by update 3).

**Credences.** S1–S2 MIXED (credence .30) on all three; S3 ≥ 2 (.5) on two of three, C5 at 1.93; S6 ≥ .5 (.6) on all three; S7 > .1 (.25) on none; slow reset ≥ .9 (.9) on all three; fast reset ≥ .5 (.35) on none; message removal ≥ .9 (.7) on both applicable models.

**By the rule written before the data** (split across models and statistics): reported per model; no mechanism word. Descriptively, on all three models: the slow state does not approach its final value during the plateau while changing by a quarter to two-fifths of its norm every cycle; in the last outer iteration the readout scores prime partially (margins close about 40 % of the gap; the correct digit is runner-up for 59–67 % of the remaining wrong cells); the solution is then assembled within the first few fast updates of one cycle; completion at that cycle needs the retained slow state and (where defined) cross-field messages, and mostly not the carried fast state; perturbing one outer iteration earlier mostly delays completion rather than preventing it (79–91 % exact by 48). One checkpoint per model, 256 stratified puzzles, linear readouts and distances, inference-time interventions outside the training distribution; nothing here enters the 2026-09-26 submission.

---

## Follow-up registration: cascade or simultaneous revision? (2026-09-25 ~17:30Z, before these statistics were computed)

**Why.** The outcome above rules out a steady hidden approach and a slow cascade across iterations, but not a FAST propagation cascade inside the completing cycle: the answer is assembled over fast updates 1–5 (graded), needs cross-field messages and the retained slow state, and survives a fast reset. The PI asked to run the two tests that separate a cascade from a simultaneous revision. Saved records only (`runs/analysis/inside_completion_20260925/{C5,SA128,EQR}.npz`); no inference. Tool: `tools/lens_cascade_tests.py` (selftest first).

**Deduction system (fixed now).** Singles-only propagation: from a partial assignment of CORRECT digits, repeat rounds; in each round assign every naked single (a cell with one remaining candidate) and every hidden single (a digit with one remaining place in a row, column or box), all at once; stop when a round assigns nothing. A cell's DEPTH is the round in which it is assigned (0 = in the seed; undeducible if never). Seeded only with correct digits, every deduction is correct (asserted).

**T1, order inside the completing cycle.** For each cohort puzzle: W = empty cells wrong at readout κ − 1; seed = givens + the empty cells correct at κ − 1; d(c) = depth of c ∈ W; t(c) = the first fast update k (1..6) of cycle κ from which the SHADOW readout of c stays correct through k = 6. Statistic: per puzzle, Spearman ρ(d, t) over the deducible cells of W (≥ 5 cells, both non-constant); median over puzzles; pooled mean t by depth.
- Cascade: median ρ ≥ 0.3 AND the pooled mean t rises from depth 1 to depth 2 to depth ≥ 3.
- Simultaneous revision: |median ρ| < 0.1.
- Otherwise MIXED. Credences: cascade 0.5, simultaneous 0.3, MIXED 0.2.
- Secondary (descriptive): the same with the direct fast readout; cells correct at κ − 1 that turn wrong inside cycle κ (a cascade from correct cells should not break them).

**T2, is a sufficient seed present before completion?** For readouts κ − 1, κ − 2, κ − 4, κ − 6: does singles-only propagation from givens + the empty cells correct at that readout solve the puzzle? p(o) = share of cohort puzzles solved. Also p(givens alone); σ = the first readout whose correct cells suffice; lag = κ − σ (readouts).
- Cascade trigger (completion follows the first sufficient seed): p(κ−1) ≥ 0.7 AND p(κ−4) ≤ 0.3, median lag ≤ 3.
- Seed present early (the wait is not a lack of correct cells): p(κ−4) ≥ 0.5, median lag > 3.
- Otherwise MIXED. Credences: seed early 0.6, trigger 0.25, MIXED 0.15.

**What each outcome would mean (written before the data).**
- Cascade order AND seed early: the completing cycle unfolds in deduction order from the correct cells, but a sufficient set of correct cells existed an iteration or more before; what the stuck phase lacks is not correct cells but the ability to act on them.
- Cascade order AND trigger: the stuck phase ends when the correct cells first become sufficient, and the completing cycle then runs the cascade.
- Simultaneous revision: the completing cycle does not follow deduction order; the cascade alternative is disfavoured for these models.
- Under every outcome: the seed uses ORACLE knowledge of which cells are correct (the model does not have it); singles-only propagation is one deduction system (the network could use stronger or different inferences); the shadow readout is a proxy for the fast state's content; one checkpoint per model; exploratory; no mechanism word enters the paper.

### Follow-up outcome (2026-09-25 ~17:45Z; `runs/analysis/inside_completion_20260925/cascade_tests/{report.txt, report.json}`; selftest 4/4, 36 random seeds agree with an independent single finder, mutants killed 3/3)

| | C5 | SA128 | EQR |
|---|---|---|---|
| T1 median ρ(depth, first stable fast update), shadow (puzzles with ≥ 5 deducible cells and varying depth) | 0.056 (68 puzzles; positive 57 %) | 0.063 (83; 59 %) | 0.028 (56; 52 %) |
| T1 same, direct fast readout (secondary) | 0.065 | 0.153 | 0.091 |
| T1 pooled mean first stable update, depth 1 / 2 / ≥ 3 (n) | 2.75 / 3.32 / 4.29 (1856 / 253 / 21) | 2.97 / 3.63 / 3.47 (1836 / 374 / 34) | 2.49 / 3.10 / 3.00 (1426 / 255 / 18) |
| Cells correct at κ − 1 that turn wrong inside cycle κ (vs wrong cells at κ − 1) | 399 (2130) | 296 (2244) | 185 (1699) |
| **T1 letter** | **SIMULTANEOUS** | **SIMULTANEOUS** | **SIMULTANEOUS** |
| T2 correct cells suffice for singles propagation at κ−1 / κ−2 / κ−4 / κ−6; givens alone | 1.00 / 1.00 / .99 / .99; .00 | 1.00 / 1.00 / .99 / 1.00; .00 | 1.00 / 1.00 / 1.00 / .99; .03 |
| T2 first sufficient readout, lag to completion (median; quartiles) | 11 (8–19.75) | 11 (8–19) | 12 (7–18.25) |
| **T2 letter** | **SEED EARLY** | **SEED EARLY** | **SEED EARLY** |

Credences: T1 cascade .5 (miss), simultaneous .3 (hit); T2 seed early .6 (hit). By the rule written before the data: the completing cycle does not follow singles-deduction order from the cells already correct, and a sufficient correct seed exists about 11–12 readouts (about four outer iterations) before completion on essentially every puzzle; the singles-propagation cascade and the "enough correct cells" trigger are disfavoured for these models. Residual: the pooled means show depth-2 cells fixed about 0.6 fast updates later than depth-1 cells on all three models (a weak ordering component the per-puzzle statistic does not resolve; most wrong cells are depth 1). Limits as registered: oracle seed, singles-only deduction, shadow readout as proxy, one checkpoint per model; not for the paper.
