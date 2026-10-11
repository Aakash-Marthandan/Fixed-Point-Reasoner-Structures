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
