# Note 2026-10-11 — P28: can transplants between unsuccessful histories complete puzzles, and do they use computation better than the alternatives? (registration, before any row; a local pilot; the external review of the draft incorporated)

**Goal (page one).** A MEASUREMENT of usefulness, before any investment in selection.
- Tool: `tools/rebuttal_p28.py`, over P27's runner and controls and P26's frozen bases.
- Outputs: `runs/analysis/rebuttal_20261014/`.
- Inference only, on the Mac's CPU, through the network's own loop.
- Receivers:
  - **Primary:** Attention 192 with the leading-component transplant.
  - **Replication:** the same intervention on Attention 256.
  - **Prespecified secondary:** the full readout-invisible transplant, on both receivers.
- No operator is chosen per width after the results.

**The question.** P28 removes the oracle correction prompts of P25–P27. Can combining information from two unsuccessful histories of an ordinary unsolved puzzle produce a more productive continuation than spending the same computation on other attempts?
- If the leading transplant beats matched random changes and the equal-work alternatives, the identified state component has practical utility.
- If useful pairs exist but the exhaustive bank is inefficient, that points toward selection.
- If neither holds, the response-transfer mechanism stands, and this way of exploiting it is shown insufficient.

**Changes from the draft (the external review), all adopted:**
1. **Continued computation at equal model work.** Each original history also continues 48 iterations, through iteration 64. Its first 16 iterations remain the paired unchanged reference.
2. **Creating solutions and improving the success rate are separate quantities.** Every letter is defined by an interval's lower bound. Otherwise the result is "advantage unresolved", with its interval.
3. **First-hit success is an executable, solution-free verification.** Final-iteration correctness and later losses are also reported.
4. **The selector gate is relaxed and split:**
   - an *efficient transplant bank*;
   - an *opportunity for selection*, read from budget curves under an outcome-independent ordering, with the hindsight-best candidate as an explicit oracle bound.
   
   Positive evidence on 192 alone justifies the small discovery analysis.
5. **Stalled histories are a new generalization test.** The basis stays frozen, and retained energy and intervention magnitudes are reported.
6. **A frozen primary comparison, screening order and screening budget.** Runtime is estimated from measured concurrent throughput and eligibility yield.

The DEC cell's code shows why the canvas question does not arise for these checkpoints. The network builds an input channel and a previous-answer channel (`paper/code/src/qhrrn2/model.py`, `build_fields_soft`), but the DEC cell reads only `fields[..., 0]`, the puzzle-input channel (`paper/code/src/qhrrn2/dec_cell.py`, `forward_core`). The canvas therefore does not enter the transition, for transplanted states as well. The display agreement check (ledger 2026-10-11 ~10:10Z) is consistent with this; on its own it could not have validated responses to interventions.

## Why

P26 and P27 established that score-preserving transplants transfer history-specific correction responses:
- fully on both widths;
- through the leading component on both (strong on 192, at the statistical boundary on 256).

They also showed no average repair gain on four untouched cells. P22–P24 support an approximate, population- and window-specific continuation model for unsolved states: completion within a 16-iteration window at a roughly constant rate. This is not an established law governing these new hybrids. On these grounds matched random changes, fresh restarts and continued computation are the decisive comparators.

## Design

**Candidates and screening (frozen).**
- Candidates: test puzzles that the released evaluation records as unsolved by the fixed start at 16 iterations (11,365 on 192, 8,747 on 256), minus every earlier pool.
- Screening order: the failures' own rating octiles, a seeded permutation inside each (`[20261014, 28, width, octile]`), interleaved round-robin across octiles.
- Screening runs in chunks of 128 candidates, until 128 eligible puzzles or a maximum of **2,048** screened per receiver.
- If fewer than 128 are found within 2,048, the run proceeds with those found and says so.
- **Sharded screening, with the same answer as a sequential screen.** Chunk c belongs to shard c mod n. Before each chunk, a shard counts the eligible puzzles in the contiguous prefix of finished chunks, and stops once that prefix holds the target. The first 128 eligible puzzles in the frozen order are therefore exactly those a sequential screen finds. Chunks screened beyond that point are counted and reported as extra screening cost.
- **Two stages, with the same eligibility.** The fixed start runs for every candidate. The three Gaussian starts run only for candidates whose fixed start never verifies, since a candidate is ineligible as soon as any history verifies.
- Why the larger budget: on the smoke, the Mac's CPU fixed start solved 7 of 8 (192) and 4 of 8 (256) benchmark cold failures, and the Gaussian starts most of the rest. The benchmark records first_exact = −1 for every one of these failures. CPU and accelerator trajectories differ per puzzle, and these dynamics are chaotic. Eligibility is therefore defined on the CPU, and the yield is low (0–25 % on the smoke).

**The bank** (on the Mac's CPU; it defines eligibility):
- K = 4 histories per candidate: the fixed start, and Gaussian starts (σ = 1; draws j = 1, 2, 3 of seed 20261014).
- Each runs 16 iterations through the network's own loop: release step; carry z ← z + η_z(z_f − z); canvas y ← y + η(softmax − y); η = η_z = 1.
- **Eligible** = no bank readout verifies (see "Success").

**Strategies after the bank** (per eligible puzzle; iteration 16 = the transplant time):

| strategy | rollouts | iterations each | model work |
|---|---|---|---|
| leading transplant h_i + (h_j − h_i) U Uᵀ, all 12 ordered pairs (i ≠ j) | 12 | 16 | 192 |
| its matched control: per row, a random direction in span(U) with the leading change's magnitude and angle to the recipient (P27) | 12 | 16 | 192 |
| full transplant h_i + (perp_j − perp_i) (secondary) | 12 | 16 | 192 |
| its control ξ_L + ξ_M (P27's combined control; secondary) | 12 | 16 | 192 |
| fresh restarts: Gaussian draws j = 101–112, from scratch | 12 | 16 | 192 |
| extended continuation of the four originals, iterations 17–64 | 4 | 48 | 192 |
| *paired reference:* the unchanged continuation, iterations 17–32 (the extended continuation's first 16) | 4 | 16 | (included) |

Further details:
- Transplants change only the slow state's readout-orthogonal part. The fast state and the canvas are the recipient's, and the scores are kept.
- Each control is drawn once per (puzzle, recipient, donor), with seeds `[20261014, puzzle, 8 / 9, i, j]`.
- "Equal model work" counts network iterations. The transplant construction overhead (projections and control draws) and the actual runtime are reported separately.
- **This is a conditional comparison given an existing four-history bank.** Building and screening the bank has a cost, which is reported but not part of the comparison.

**Success** (executable without the solution):
- At every readout, the displayed grid is verified: the givens are kept, and every row, column and box is a permutation of 1–9.
- A rollout's success is its **first verified readout** inside its window.
- The verified grid is cross-checked against the reference answer; mismatches are counted and reported, with zero expected.
- Also reported: correctness at the window's final iteration, and losses (verified, then not verified at the end). This keeps the manuscript's distinction between discovery and preservation.

**Generalization descriptives (stalled states).** For every pair at iteration 16:
- the frozen basis's retained energy ‖L‖² / ‖Δ‖², compared with P27's 0.70 / 0.78 after iteration 1;
- the magnitudes ‖L‖ / ‖h_r‖ and ‖Δ‖ / ‖h_r‖.

**Budget curves (outcome-independent).**
- The 12 ordered pairs of a puzzle are taken in a frozen seeded order (`[20261014, puzzle, 3]`), the same for all four transplant and control arms.
- Restarts are taken in draw order.
- The extended continuation at budget b gives each original 4b iterations, i.e. b × 16 iterations of work.
- Coverage is reported at budgets b ∈ {1, 2, 4, 8, 12}.
- The hindsight-best single candidate, i.e. the full bank's coverage, is reported as an explicit **oracle** upper bound for a budget-1 selector.

## Quantities and letters

The unit is the eligible puzzle. Intervals are 95 % puzzle bootstrap with 4,000 draws.

**Quantities, per arm A:**
- **Additional coverage:** the share of puzzles solved by some rollout of A but by none of the four unchanged reference continuations. Reported with the **reverse losses**: solved by a reference continuation but by no rollout of A.
- **Per-rollout improvement:** the mean paired difference, over the 12 pairs, between each transplant's success and its recipient's own unchanged continuation (same 16-iteration window). Clustered by puzzle.
- **Bank efficiency:** coverage of A minus coverage of each equal-work alternative (the matched random bank, restarts, the extended continuation), with the count of discordant puzzles.
- **Headroom for selection:** the oracle (the full bank's coverage) minus the best budget-1 alternative (one restart, the extended continuation at budget 1, or one transplant in the frozen order).
- **Unique coverage:** solved by the bank but by neither the restarts nor the extended continuation.

**Letters** (each "lower bound > 0"; otherwise "advantage unresolved", with the interval):
- **ADDS-SOLUTIONS:** additional coverage of the leading bank.
- **PER-ROLLOUT-GAIN:** per-rollout improvement of the leading transplant over the paired unchanged continuation.
- **DONOR-SPECIFIC-COVERAGE:** leading bank coverage minus the matched random bank's. This is the claim about useful donor information.
- **EFFICIENT-BANK:** leading bank coverage beats both restarts and the extended continuation.
- **SELECTION-OPPORTUNITY:** headroom for selection.

**How the letters are read.**
- The primary letters are those of the leading transplant on Attention 192.
- 256 replicates; the full transplant is reported with the same quantities, as secondary.
- **SELECTION-OPPORTUNITY on 192 justifies P30**, a small discovery analysis of whether solution-free features identify the useful pairs, even if 256 does not show it.
- A null result is not a hard stop unless its interval excludes effects worth pursuing.

**Power.** A 128-puzzle pilot. Power for coverage differences depends on the paired discordance, which is reported for every comparison; sample size alone does not fix it.

## Predictions and credences (before any row)

| event | credence 192 (primary) | credence 256 (replication) |
|---|---|---|
| ADDS-SOLUTIONS (some new solutions: perturbations re-roll trajectories) | 0.70 | 0.60 |
| PER-ROLLOUT-GAIN (a net paired improvement; re-rolling alone does not predict one) | 0.20 | 0.15 |
| DONOR-SPECIFIC-COVERAGE | 0.15 | 0.10 |
| EFFICIENT-BANK | 0.10 | 0.08 |
| SELECTION-OPPORTUNITY | 0.50 | 0.40 |

## Wording under each outcome

- **EFFICIENT-BANK and DONOR-SPECIFIC-COVERAGE:** "Transplanting the identified component between unsuccessful histories completes more puzzles than matched random changes, fresh restarts or continued computation, at equal model work, given an existing bank."
- **SELECTION-OPPORTUNITY without EFFICIENT-BANK:** "Useful transplant pairs exist with headroom over budget-matched alternatives, but evaluating all of them is not efficient. This motivates a solution-free selector."
- **ADDS-SOLUTIONS only:** "Transplants re-roll stuck trajectories into some new completions, without a net advantage."
- **Otherwise:** "At this pilot scale, the [leading] transplant did not show an advantage [interval]." Scope:
  - it does not reverse P27's early response-transfer finding;
  - the frozen basis was identified on early histories and is applied here to stalled ones;
  - unresolved persistence at iterations 2–4 does not by itself predict this outcome.

Scope under every outcome:
- one checkpoint per receiver; attention receivers;
- a four-history bank;
- 128 eligible puzzles;
- verification-based success;
- a conditional comparison given the bank.

## Gates

- The phase-2 bank reproduces phase 1's eligibility; no readout verifies.
- Restored bank states reproduce the unchanged continuation bitwise (four extra rows per block).
- Every transplant and control keeps scores (≤ 1e-3).
- P27's control matches: norm and inner product.
- The verification versus reference mismatch count is reported, with zero expected.
- The sharded and resumable run equals an unsharded run of the same block bitwise; this was checked on the smoke.

## Smoke and compute

**Two earlier smokes** exposed two faults, both fixed before any row:
- a puzzle-array shape error in verification;
- a startup race when shards of one receiver shared a status file (fixed with per-process metadata folders).

They also showed the low eligibility yield that motivated the two-stage screening and the 2,048 budget. The transplant construction overhead measured 0.08 s per block, negligible against the rollouts.

**The final smoke** ran on 64 screened candidates per receiver:
- two-stage screening on two shards per receiver;
- arms on two shards per receiver;
- an unsharded repeat of Attention 192's blocks in a separate process.

Its results:
- **Screening.** Fixed-start failures were 9 / 32 and 16 / 32 on 192, and 14 / 32 and 12 / 32 on 256. Eligible: 3 + 3 and 4 + 5, so the yield is about 9 % (192) and 14 % (256). The prefix rule cut at chunk 0 and reported the extra chunk.
- **Gates.** Restoration was bitwise in every block. Verification versus reference mismatches: 0. The sharded and unsharded runs of each 192 block were equal on every recorded field.
- The reader ran with its output suppressed: no smoke statistic was seen.

**Measured throughput.**
- One 128-row, 16-iteration batch takes about 134 s (192) and 194 s (256) for a single process.
- Four or five concurrent processes keep the 18 cores about 87 % busy, at about 2–2.8× the single-process time each: an aggregate of about 1.8–2× a single process.
- Each process uses 3–5 GB of memory. Eight processes would exceed the 48 GB with the other open apps.

**The run therefore uses four processes in each phase**, two per receiver. That is all the cores the CPU can keep busy.

**Projection:**
- Screening: about 11 chunks on 192 and 8 on 256 to reach 128 eligible, about 1.2 h.
- Arms: 4 blocks of 32 per receiver, about 21 batches each, about 4 h.
- About 5–5.5 h in total.

## P28 Outcome (2026-10-11 16:42Z; `runs/analysis/rebuttal_20261014/report.{txt,json}`; every gate passed; no tool edited during the run)

**Run.**
- Screening finished at 12:52Z:
  - 192: 128 eligible of 1,024 screened (yield 12.5 %), with one extra chunk screened in parallel;
  - 256: 128 eligible of 896 (14.3 %), likewise one extra chunk.
- Arms: four processes, 3 h 50 min. Summed block time was 6.2 h (192) and 7.7 h (256). Transplant construction overhead was 12 s and 15 s in total.
- Gates:
  - score shift ≤ 8e-14 (measured on the float64 construction, before the float32 cast; the executed float32 tolerance is in the scope addendum);
  - control norm and inner-product matches ≤ 1.3e-13;
  - verification versus reference mismatches 0;
  - restoration bitwise in all 8 blocks.

**The registered letters (95 % puzzle-bootstrap intervals).**

| quantity | 192, leading (primary) | 256, leading (replication) |
|---|---|---|
| additional coverage vs the short unchanged continuation | **+0.312 [+0.234, +0.391]** ADDS-SOLUTIONS (reverse losses 0.047) | **+0.297 [+0.219, +0.375]** ADDS-SOLUTIONS (0.047) |
| per-rollout improvement (paired) | +0.017 [−0.018, +0.052] advantage unresolved | −0.019 [−0.059, +0.020] unresolved |
| bank vs matched random bank | +0.023 [−0.055, +0.102] (25 discordant) unresolved | −0.008 [−0.078, +0.055] (19) unresolved |
| bank vs fresh restarts | +0.000 [−0.094, +0.086] | −0.039 [−0.125, +0.039] |
| bank vs extended continuation | +0.016 [−0.055, +0.094] | +0.039 [−0.047, +0.125] |
| EFFICIENT-BANK | no | no |
| oracle headroom over the best budget-1 alternative | **+0.273 [+0.188, +0.359]** SELECTION-OPPORTUNITY | **+0.312 [+0.227, +0.406]** SELECTION-OPPORTUNITY |
| unique coverage (vs restarts and extended) | +0.055 [+0.016, +0.094] | +0.039 [+0.008, +0.078] |

**Coverage at budgets 1 / 2 / 4 / 8 / 12** (equal model work per budget):

| strategy | 192 | 256 |
|---|---|---|
| leading | 0.180 / 0.289 / 0.414 / 0.633 / 0.695 | 0.148 / 0.250 / 0.430 / 0.578 / 0.641 |
| its control | 0.211 / 0.297 / 0.453 / 0.594 / 0.672 | 0.180 / 0.258 / 0.445 / 0.562 / 0.648 |
| restarts | 0.180 / 0.328 / 0.500 / 0.625 / 0.695 | 0.188 / 0.289 / 0.430 / 0.586 / 0.680 |
| extended | 0.203 / 0.297 / 0.430 / 0.617 / 0.680 | 0.109 / 0.250 / 0.391 / 0.531 / 0.602 |

The four short unchanged continuations cover 0.430 (192) and 0.391 (256).

**Secondary (full transplant).** The letters are the same on both receivers. On 256, restarts beat the full bank: −0.094 [−0.180, −0.016].

**Generalization descriptives (stalled states).**
- The frozen basis captures 0.783 (192) and 0.858 (256) of the iteration-16 donor–recipient difference: more than after iteration 1 in P27 (0.70 / 0.78).
- Transplant magnitudes: ‖L‖ / ‖h‖ = 0.33 / 0.35 and ‖Δ‖ / ‖h‖ = 0.37 / 0.38.

**Discovery and preservation.** Every verified success in every arm was still verified at its window's final iteration: there were no *terminal* losses. The saved fields (first success, final correctness) cannot exclude a temporary loss followed by recovery (corrected in the scope addendum).

**Recount.** Independent code reproduces every registered point estimate exactly. It has its own dense tables from the raw rows, its own budget curves, paired differences and resampler.
- The eligible lists re-derive from the chunk files in the frozen order.
- Every block's restoration flag is set, and the mismatch count is 0.
- Regenerated banks of two eligible puzzles per receiver show no verified readout.
- Own intervals are within about 0.01 of the report's.

**Predictions.**

| event | 192 credence | 192 outcome | 256 credence | 256 outcome |
|---|---|---|---|---|
| ADDS-SOLUTIONS | 0.70 | yes | 0.60 | yes |
| PER-ROLLOUT-GAIN | 0.20 | no | 0.15 | no |
| DONOR-SPECIFIC-COVERAGE | 0.15 | no | 0.10 | no |
| EFFICIENT-BANK | 0.10 | no | 0.08 | no |
| SELECTION-OPPORTUNITY | 0.50 | yes | 0.40 | yes |

Brier score over the 10 events: 0.097.

**Exploratory additions** (after the frozen report; `explore.py`):
- **[Withdrawn wording, scope addendum: similar marginal rates do not establish independence.] The tested strategies have comparable aggregate completion rates over these windows.** Per-rollout success within a 16-iteration window:

  | strategy | 192 | 256 |
  |---|---|---|
  | leading | 0.193 | 0.169 |
  | its control | 0.189 | 0.195 |
  | full | 0.184 | 0.166 |
  | full control | 0.199 | 0.159 |
  | restart | 0.172 | 0.162 |
  | unchanged | 0.176 | 0.188 |

- **The selection opportunity is not transplant-specific.** The oracle headroom over the best budget-1 strategy is similar for every candidate set (0.09–0.19). Fresh restarts have the largest unique coverage (0.062 on both receivers).
- **Union coverage.** All six strategies together cover 0.867 (192) and 0.844 (256); 17 and 20 puzzles are solved by none.
- **Rollout level.** Leading minus its control is +0.004 [−0.019, +0.026] (192) and −0.027 [−0.053, −0.001] (256).

**Wording (registered).**
- ADDS-SOLUTIONS holds without any advantage: "Transplants re-roll stuck trajectories into some new completions, without a net advantage."
- At equal model work, given an existing bank, the leading transplant did not complete more puzzles than matched random changes, fresh restarts or continued computation, on either receiver. The intervals are reported above. The full transplant did no better.
- SELECTION-OPPORTUNITY holds as registered: a perfect selector over the 12 transplants would beat any single budget-1 alternative.
- The exploratory comparison shows this headroom is generic to any set of re-rolls, not specific to transplants.

Scope under every outcome:
- one checkpoint per receiver; attention receivers;
- a four-history bank;
- 128 eligible puzzles per receiver;
- verification-based success;
- a comparison conditional on the bank.

This does not reverse P27's early response-transfer finding. The frozen basis, identified on early histories, captures even more of the late difference, but transplanting it did not help completion here.

**For the program.**
- P26–P27 showed that retained state carries transferable, history-specific correction responses. P28 shows that exploiting this by exhaustive transplanting between unsuccessful histories did not complete more puzzles than matched random changes, fresh restarts or continued computation at equal model work (wording corrected in the scope addendum).
- The tested strategies have comparable aggregate completion rates over these windows.
- **P30** (a solution-free selector of transplant pairs) is justified by the registered rule on 192. The generic headroom means it would have to beat selection among restarts too. On this evidence, the PI may prefer **P29** (task generality on ARC) as the next priority.

## P28 SCOPE ADDENDUM after an external review (2026-10-11 ~19:10Z; every number reproduced; exploratory; `runs/analysis/rebuttal_20261014/review_check.py`, `float32_check.py`)

**What stands.** The registered decisions stand. The reviewer repeated the principal intervals with 20,000 draws. The equal-work comparison is decisive (puzzles solved of 128, after 192 additional iterations of work, conditional on the bank):

| strategy | 192 | 256 |
|---|---|---|
| leading transplants | 89 (69.5 %) | 82 (64.1 %) |
| matched random controls | 86 | 83 |
| full transplants | 88 | 75 |
| full-transplant controls | 87 | 76 |
| fresh restarts | 89 (69.5 %) | 87 (68.0 %) |
| extended originals | 87 | 77 |

- Leading minus its control: +2.3 pp [−5.5, +10.2] (192) and −0.8 pp [−7.8, +5.5] (256).
- Full minus restarts on 256: −9.4 pp [−18.0, −1.6].
- These are failures to show an advantage, not equivalence tests. The 192 pilot still permits a modest benefit, but there is no consistent positive pattern that would support a performance claim.

**"Adds solutions", with its comparison attached.**
- Against the four short unchanged continuations, the leading bank solves 40 more puzzles and misses 6 (192), or 38 and 6 (256). It spends three times their computation.
- Against equal-work restarts:
  - 192: 17 transplant-only and 17 restart-only successes;
  - 256: 12 and 17.
- Transplantation changes *which* puzzles complete without a net benefit.

**Timing** (exploratory, equal work at 48 additional iterations; puzzles solved):

| strategy | 192 | 256 |
|---|---|---|
| 12 leading transplants × 4 iterations | 48 | 42 |
| their random controls | 52 | 53 |
| 12 fresh restarts × 4 iterations | 21 | 33 |
| 4 originals × 12 iterations | 48 | 46 |

- Transplants produce answers sooner than fresh starts. The same advantage appears with matched random changes, and it disappears against continued computation.
- This is consistent with retaining useful computation from the existing histories. It is no evidence that the donor's specific information supplies the benefit.
- Similar final success rates hide these differences in the timing of completion.

**The mechanism distinction.**
- P26–P27 identified coordinates that carry part of a history's correction response. P28 did not identify which values of those coordinates make a stalled computation more productive.
- The frozen subspace captures 78 % (192) and 86 % (256) of the iteration-16 difference, so the negative result is not explained by little captured variation.
- Capturing variation, transferring a response and identifying a beneficial change are separate achievements.
- P28 changes the state population: early histories in P27, against four-history failures at iteration 16. It neither reverses P27 nor establishes that response transfer remains effective in these late states. It tests the usefulness of applying the frozen intervention there.

**"Selection opportunity", narrowed.**
- The reader's "best budget-1 alternative" is a *per-puzzle maximum* over three strategies (restart, extended, one transplant): itself an oracle choice, not the best fixed strategy.
- The registered headroom (+0.273 / +0.312) and the exploratory one (a per-puzzle maximum over six strategies) therefore use different comparators. Against the same six-strategy baseline, headroom is:

  | candidate set | 192 | 256 |
  |---|---|---|
  | transplants | +0.148 | +0.148 |
  | restarts | +0.148 | +0.188 |

- The opportunity is generic to candidate selection. A perfect selector can exploit successes scattered among twelve candidates; that does not show observable features can identify them before running.
- Nothing here makes transplant selection especially promising.

**Wording corrected.**
- **Withdrawn:** "every strategy behaves like an independent re-roll", "a generic re-roll" (Outcome, ledger). Similar marginal success rates do not establish independence, memorylessness or chaos. The supported statement: the tested strategies have comparable aggregate completion rates over these windows.
- **"No losses" means "no terminal losses."** The saved fields (first success, final correctness) cannot exclude a temporary loss followed by recovery.
- **Score preservation.** The ≤ 8e-14 figure was the float64 construction. The executed float32 states, rebuilt exactly as the runner builds them for two eligible puzzles per receiver (12 pairs × 4 arms), change the displayed scores by at most 7.4e-7 (192) and 6.9e-7 (256), with no displayed-digit flips.

**Candidate manuscript wording (the PI decides):** "A subspace identified on separate puzzles supports transfer of history-specific correction responses. Applying the corresponding transplant between unsuccessful histories produces additional completions, but does not establish improved success over matched random perturbations, fresh restarts or extended continuation at equal model work." This shows where the causal finding stops. It is not an inference improvement, and the selection label must not be presented as one.

**Next (the review's priority).**
- **P29**, the cross-task test of response transfer on ARC, directly addresses the generalizable mechanism.
- **P30** only as a bounded exploratory analysis:
  - target the incremental benefit of a choice *within the same puzzle*, using information available before continuation;
  - compete with equally informed selection among controls and restarts;
  - keep all candidates of a puzzle together in evaluation;
  - first establish that a signal exists, before any large selection experiment.
