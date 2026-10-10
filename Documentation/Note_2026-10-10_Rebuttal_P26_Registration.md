# Note 2026-10-10 — P26: does a compact readout-invisible component carry a history's response to identical corrections? (registration, before any row; the external review of the draft incorporated)

**Goal (page one).** A MEASUREMENT.
- Tool: `tools/rebuttal_p26.py`, which reuses P25's runner (release states, the donor history loop, variant rollouts, the angle- and norm-matched rotation) and P16's relabel unchanged.
- Outputs: `runs/analysis/rebuttal_20261012/`.
- Frozen bases: `tools/p26_basis_{192,256}.json`, each a deterministic function of its receiver's stage-D file and the procedure registered here, written before that receiver's stage C begins.
- Inference only, on the Mac's CPU. No training.
- Receivers:
  - Attention 192 identifies.
  - Attention 256 replicates with the identification procedure frozen.
  - Attention 128 is not run: P25 showed no transfer beyond random there.
- The choice of 192 for identification is motivated by P25, where the donor-specific evidence was strongest, and is disclosed.

P26 sharpens P25's Part 2 in five ways, following the external review of P25 and then of this note's draft:
1. **Aligned challenges.** Both histories of a pair receive identical corrections and are scored on identical cells.
2. **Within-challenge specificity.** The sharp test (R2) contrasts two equal-sized masks within each (pair, direction, dose). The broad test (R1) keeps the response curve, under a nuisance model fixed now.
3. **Discovery and confirmation separated by puzzle.** A compact component is identified on discovery puzzles. It is then tested on confirmation puzzles with new pairs. The procedure is frozen now and applied unchanged to 256.
4. **Magnitude-matched random controls.** Each compact transfer is compared with an equal-rank random projection scaled to the same magnitude, and retained energy is reported. One primary compact method, the common basis, is fixed now.
5. **"At least half" tested jointly.** The claim is a joint bootstrap of E_compact − ½ E_full on matched contrasts of the sharp test.

## Why

P25's TRANSFER-CARRIES held on Attention 256 and 192 (`Note_2026-10-10_Rebuttal_P25_Registration.md`, Outcome and scope addendum):
- The excess came mostly through the subtracted baseline term: 0.166 of 0.188 on 256, 0.241 of 0.330 on 192.
- A headroom-adjusted excess remained (256 +0.23, 192 +0.29). Donor-specific evidence was strongest on 192 and specification-sensitive on 256.
- Donor and recipient were scored on different cells and masks.

P26 asks two questions:
1. Does a transferred hidden component carry a history's *response to a specific correction*, beyond its effect on the unedited continuation (the baseline) and beyond matched random change?
2. If it does, is that component small and shared across puzzles?

## Design

**Population.**
- 384 fresh test puzzles per receiver: 48 from each of the full test set's eight rating octiles (rank-based bins).
- Excluded: the study's 512, the P12 / P17 / P20 / P22 attention pools, and P25's pools.
- Seeds `[20261012, 26, width, octile]`.

**Pairs.** For each puzzle:
- The recipient-or-donor A is the intact fixed-start state after iteration 1, computed through the release step.
- B is the same puzzle's history from a Gaussian start (seed `[20261012, puzzle, 7]`) after iteration 1, as P25 builds it.
- A pair is kept when the two histories share at least 12 wrong cells: W = W_A ∩ W_B, which implies both are unsolved.
- Expected from P25's states: about 195 eligible pairs on 192 and 156 on 256.

**Aligned challenges.**
- Evaluation cells E: four cells of W, seed `[20261012, puzzle, 1]`. They are never edited, in any arm.
- Correction masks are drawn from R = W \ E at doses 0.25, 0.5 and 0.75 of |R|. Each dose has four masks of equal size, round(f · |R|), seeds `[20261012, puzzle, 2, dose, mask]`.
- Each mask is applied identically to both histories. Its cells are relabelled to their solution digits from that history's own displayed digit: the field vectors of the displayed and solution digits are exchanged in the slow and fast states. This is P16's relabel, i.e. a correction in the network's own coordinates.

**Directions.**
- Each pair is used in both directions: A ← B, and B ← A.
- In each direction, the recipient r receives the donor d's readout-invisible slow part.
- The donor's response to a challenge is the reciprocal direction's *intact* response to the same masks on the same cells.

**Split.**
- Eligible pairs are split by puzzle with a seeded permutation (`[20261012, 26, width, 0]`): the first half is discovery, the rest is confirmation.
- Expected confirmation sizes: about 98 puzzles on 192 and 78 on 256.
- Results on discovery puzzles are reported as exploratory.

**Conditions.** Every condition changes only the readout-orthogonal part of the recipient's slow state. Every displayed score is kept (≤ 1e-3; in practice ≈ 6e-7). Write perp_r and perp_d for the readout-orthogonal parts, and Δ = perp_d − perp_r (per field and cell; dimension w). Each condition gets its own no-edit continuation.

| condition | what it is | stage |
|---|---|---|
| intact | the recipient unchanged | D, C |
| full | recipient + Δ: the donor's whole readout-invisible slow part; the positive control | D, C |
| full_random | the recipient's perp rotated by the same per-row angle and set to the donor's norm, in a random direction (P25's control; seed `[20261012, puzzle, 6, direction]`) | D, C |
| global (**the primary compact method**) | recipient + Δ projected onto the frozen common basis U (k = w/8: 24 on 192, 32 on 256) | C |
| global_rand | recipient + Δ projected onto a random k-dimensional readout-orthogonal subspace (seed `[20261012, puzzle, 9, direction]`), scaled to the global component's Frobenius norm | C |
| adaptive (secondary) | recipient + Δ projected onto the top-k directions of this pair's own Δ over (field, cell): per-pair compression, not reusable | C |
| adaptive_rand (secondary) | the same random subspace as global_rand, scaled to the adaptive component's norm | C |

**Stages per receiver.**
1. **Stage D (discovery puzzles):** intact, full and full_random.
2. **Learn (the identification procedure, frozen now):**
   - Score each discovery pair-direction by the mean, over its six contrasts (defined under R2), of (d_full − d_full_random) · d_donor.
   - Keep the top half (selection fraction 0.5; ties broken by index).
   - U = the top-k right singular vectors of the kept pair-directions' pooled Δ rows (729 rows of dimension w each), Gram–Schmidt orthogonalized to the readout and orthonormalized; k = w/8.
   - The basis file records the stage-D file's sha256, the kept states and their scores. The run asserts the discovery split it was learned on.
3. **Stage C (confirmation puzzles):** all seven conditions.
4. **Order:**
   - 192 and 256 run their stage D concurrently. Each then runs `learn`, and their stage C runs concurrently.
   - The procedure is fully specified here, so nothing at 256 depends on 192's data.
   - An amendment forced by 192 would be disclosed, and 256 would be re-run under it.

**Exact restoration.**
- After every other variant of a stage is built, each recipient's saved tensors (slow and fast) are written back.
- Rolled out on the no-edit arm and on one mask per dose, they must reproduce the intact rollouts *bitwise* at every one of the four iterations.
- Variants are batched in other positions, so this also checks the rollout's independence from batch position and the absence of any in-place change.
- The smokes on 192 and 256 reproduced the rollouts bitwise.

## Primary quantities

All quantities are at k = 1 (one iteration after the change), on the four evaluation cells. Intervals are 95 % puzzle-bootstrap intervals: 2,000 resamples, keeping both directions and every challenge of a puzzle together.

For a pair-direction (state s), condition X, dose i and mask m:
- Y_X(i, m) is the share of evaluation cells displaying their solution digit.
- b_X is X's no-edit share.
- The recipient's intact values are Y_R and b_R. The donor's (the reciprocal state's intact values) are Y_D and b_D.

**BASELINE (P25's baseline transfer, now on aligned cells).**
- Regress (b_X − b_R) on (b_D − b_R) across states.
- Statistic: the slope for full minus the slope for full_random.
- BASELINE-TRANSFER on a receiver if the interval's lower bound is > 0.

**R1: response-curve transfer, beyond the baseline (the broad test).**
- For X ∈ {full, full_random}, across (state, dose, mask), ordinary least squares:

  (Y_X − b_X) = α + **β_D · (Y_D − b_D)** + β_R · (Y_R − b_R) + nuisance

- The nuisance set is registered now: b_X, b_R, b_D, f_R, f_D, f_R², f_D², b_X · f_R, b_R · f_R, b_D · f_D. Here f_R and f_D are the realized fractions of each history's own wrong cells that the mask corrects.
- Statistic: β_D(full) − β_D(full_random), with both regressions inside each bootstrap draw.
- RESPONSE-TRANSFER on a receiver if the lower bound is > 0 on its confirmation stage.

**R2: challenge specificity (the sharp test).**
- Within each (state, dose), the four equal-sized masks give two independent contrasts: d_X = Y_X(m0) − Y_X(m1) and Y_X(m2) − Y_X(m3). Six contrasts per state.
- Baselines cancel inside a contrast, and the dose, the pair and the direction are held fixed.
- Regress (d_full − d_full_random) on d_D and d_R.
- Statistic: the coefficient on d_D, i.e. how much of the donor's response to *that particular* correction the full transfer carries, beyond the recipient's own response.
- Reading the scale: if the full transfer made the recipient respond exactly as the donor does and the random control left it responding as itself, the coefficient would be 1.
- CHALLENGE-SPECIFIC on a receiver if the lower bound is > 0 on its confirmation stage.

**R3: compactness, prespecified on R2.**
- E_full is R2's coefficient for (full − full_random). E_compact is the same coefficient for (global − global_rand), on the same confirmation pair-directions and contrasts.
- The joint bootstrap gives E_compact − ½ E_full.
- Letters:
  - **COMPACT:** E_full resolved > 0, E_compact resolved > 0, and E_compact − ½ E_full resolved > 0.
  - **COMPACT-PARTIAL:** E_full and E_compact resolved > 0, the half-test not resolved.
  - **NOT-COMPACT:** E_full resolved, E_compact not.
  - **NOT-APPLICABLE:** E_full not resolved. The numbers are reported without a letter, because "half" of an unresolved effect is not a claim.
- Reported beside it, as secondary:
  - adaptive − adaptive_rand on R2, and its half-test;
  - the retained energy ‖proj Δ‖² / ‖Δ‖² for the common basis, the adaptive basis and the random subspace (before scaling);
  - R1's statistic for global − global_rand.

**Letters across receivers.**
- A claim in the paper's discussion requires the letter on **both** receivers' confirmation stages.
- A letter on 192 alone is reported as identification-receiver evidence. The discovery stages are exploratory.

**Power** (from P25's variance components, two contrasts per dose, 80 % power at two-sided 5 %):
- R2 detects a coefficient of about 0.13 on 192 (about 98 confirmation puzzles) and 0.14 on 256 (about 78).
- With 256 puzzles per receiver, as in the draft, a single contrast per dose would have detected only about 0.23–0.24. Hence the larger population and the four masks per dose.

## Predictions and credences (before any P26 row)

| event (both receivers' confirmation stages) | credence |
|---|---|
| BASELINE-TRANSFER (P25's baseline transfer replicates on aligned cells) | 0.65 |
| RESPONSE-TRANSFER (R1) | 0.40 |
| CHALLENGE-SPECIFIC (R2) | 0.25 |
| any compact component carrying at least half (conditional on R1) | 0.30, ≈ 0.12 unconditional |
| COMPACT as registered: the shared common basis, on R2, at least half | ≈ 0.06 |

## Wording under each outcome

- **BASELINE-TRANSFER only:** "Transferring a history's readout-invisible slow component shifts the recipient's unedited continuation toward the donor's on the same cells; a transfer of its response to corrections is not resolved at this sample size (the sharp test detects coefficients of about 0.13–0.14)."
- **RESPONSE-TRANSFER:** "Transferring a history's readout-invisible slow component carries that history's response to the same corrections, beyond its effect on the unedited continuation, beyond the registered nuisance terms and beyond matched random change."
- **CHALLENGE-SPECIFIC:** "... including its response to particular corrections, within dose, pair and direction."
- **COMPACT:** "A common k = w/8 subspace learned on other puzzles carries at least half of the challenge-specific transfer, beyond an equal-rank random subspace at the same magnitude."
- **COMPACT-PARTIAL:** "... carries a resolved part of it, but at least half is not resolved."
- **Neither R1 nor R2:** "We did not resolve a transfer of the donor's response beyond matched random change at this sample size. What P25 found to transfer may act through the unedited continuation." This does *not* say the response is not carried: the intervals bound how large a carried response could be, and they are reported.

Scope under every outcome:
- oracle-guided diagnostic corrections (masks use the solution);
- constructed hybrids, not natural histories;
- one checkpoint per receiver; attention receivers only;
- a one-iteration primary horizon (iterations 2–4 recorded);
- random directions for the controls;
- four evaluation cells per state, so outcomes are coarse.

## Gates

- Release states reproduce bitwise.
- Evaluation cells and masks lie within W_A ∩ W_B, and no mask touches E.
- Identical masks for both histories (by construction: one draw per puzzle).
- Every relabel produces its designed display.
- No evaluation cell is edited.
- Every condition keeps the displayed scores (≤ 1e-3).
- Scaled random controls match their compact component's norm (≤ 1e-6).
- The frozen basis is orthonormal, readout-orthogonal, of rank w/8, and was learned on the asserted discovery split.
- Exact restoration reproduces the intact rollouts bitwise.

## Smoke and disclosures

Gates-only smokes ran on the pool's first six puzzles: 192 stage D, `learn` and stage C; 256 stage C with a stand-in random basis. Every gate passed, and restoration was bitwise.
- While the reader was being tested, its printout on those 3–4 smoke pairs per receiver was displayed. At that size it carries no information, and no choice depended on it.
- Those puzzles belong to the pool and are included in the run.
- Two code fixes preceded the passing smokes; neither changes the design:
  - a dictionary-key typo in the random control's guard;
  - the reader's smoke paths.

## Compute

Measured on the smokes, single process: 33.5 s (192) and 48.5 s (256) per chunk of 128 variants.

| receiver | stage D variants | stage C variants |
|---|---|---|
| 192 | ≈ 8,300 | ≈ 18,600 |
| 256 | ≈ 6,700 | ≈ 14,800 |

This is ≈ 4.3 h sequentially, or ≈ 3 h with each stage's two receivers running concurrently. That is at the ~3 h Mac threshold. It stays local because:
- each stage is shorter than the threshold;
- a pod would need a JAX inference port, the PI's spend envelope and setup longer than the overrun;
- CPU keeps P26's states of the same kind as P25's: CPU and accelerator trajectories differ per puzzle.

## P26 Outcome (2026-10-10 14:44Z; `runs/analysis/rebuttal_20261012/report.{txt,json}`; every gate passed on all four runs; no tool edited during the runs)

**Run.**
- Eligible pairs: 208 on 192 and 166 on 256, split 104 / 104 and 83 / 83 into discovery and confirmation.
- Variants: 8,112 and 18,928 on 192; 6,474 and 15,106 on 256 (stage D and stage C).
- Exact restoration reproduced every intact rollout bitwise.
- The bases were frozen before stage C: `tools/p26_basis_192.json` (sha256 d2a5e489…) and `tools/p26_basis_256.json` (sha256 420752a5…). They came from 104 of 208 and 83 of 166 discovery pair-directions.
- The Mac slept twice mid-run (lid closed; once on a drained battery). The processes paused and resumed intact.

**The registered letters (confirmation stages; 95 % puzzle-bootstrap intervals).**

| rule | Attention 192 (104 puzzles) | Attention 256 (83 puzzles) | both receivers |
|---|---|---|---|
| BASELINE: full − random slope | +0.194 [+0.040, +0.368] BASELINE-TRANSFER | +0.070 [−0.079, +0.201] not resolved | no |
| R1: β_D(full) − β_D(random) | +0.202 [+0.132, +0.266] RESPONSE-TRANSFER | +0.103 [+0.042, +0.166] RESPONSE-TRANSFER | **yes** |
| R2: coefficient on d_D | +0.139 [+0.052, +0.227] CHALLENGE-SPECIFIC | +0.102 [+0.030, +0.168] CHALLENGE-SPECIFIC | **yes** |
| R3: E_compact (common basis − scaled random) | −0.017 [−0.112, +0.081] | +0.093 [+0.015, +0.171] | |
| R3: E_compact − ½ E_full | −0.086 [−0.182, −0.002] | +0.042 [−0.035, +0.121] | |
| R3 letter | NOT-COMPACT | COMPACT-PARTIAL | the registered COMPACT event: **no** |

Secondary quantities, as registered:
- Retained energy ‖proj Δ‖² / ‖Δ‖²: the common k = w/8 basis keeps 0.705 (192) and 0.780 (256); the per-pair adaptive basis 0.742 and 0.817; an equal-rank random subspace 0.125 on both (= k/w).
- Adaptive − adaptive_rand on R2: −0.006 (192) and +0.112 (256).
- R1 for the common basis: +0.126 and +0.100.

The discovery stages are exploratory:

| rule | 192 | 256 |
|---|---|---|
| BASELINE | +0.146 [−0.007, +0.314] | +0.139 [+0.020, +0.253] |
| R1 | +0.271 [+0.191, +0.335] | +0.103 [+0.036, +0.170] |
| R2 | +0.164 [+0.081, +0.256] | +0.082 [−0.015, +0.171] |

**Recount.** Independent code reproduces every point estimate exactly. It has its own unit tables, QR least squares, contrasts, selection score and SVD.
- The npz hashes are verified.
- Both bases re-derive from their stage-D files: the selected states are identical and the subspaces differ by about 1e-13.
- The stored retained energies recompute exactly.
- An independent resampler gives intervals within about 0.01 of the report's. The one borderline is 192's half-test upper bound: −0.002 in the registered bootstrap and +0.005 in the recount's.
- 192's NOT-COMPACT letter rests on E_compact's interval, which includes zero in both. No claim is made that the compact part carries resolvably *less* than half.

**Predictions.**

| event | credence | outcome |
|---|---|---|
| BASELINE-TRANSFER on both | 0.65 | no |
| RESPONSE-TRANSFER | 0.40 | yes |
| CHALLENGE-SPECIFIC | 0.25 | yes |
| a compact component carrying at least half, given R1 | 0.30 | no (192 fails) |
| COMPACT as registered | ≈ 0.06 | no |

Brier score: 0.288. The two central letters were under-predicted.

**Wording (registered).**
- RESPONSE-TRANSFER and CHALLENGE-SPECIFIC hold on both receivers: "Transferring a history's readout-invisible slow component carries that history's response to the same corrections, beyond its effect on the unedited continuation, beyond the registered nuisance terms and beyond matched random change, including its response to particular corrections, within dose, pair and direction."
- Magnitude: the full transfer carries about 0.10–0.14 of the donor's challenge-specific response, on a scale where 1 = the recipient responds as the donor does.
- COMPACT is not established. On 256 alone, "a common k = w/8 subspace learned on other puzzles carries a resolved part of it, but at least half is not resolved."
- Scope under every outcome: oracle-guided diagnostic corrections; constructed hybrids; one checkpoint per receiver; attention receivers; four evaluation cells per state.

**What changes against P25.**
- On aligned cells, the response transfer is resolved on both receivers, while the baseline transfer is resolved on 192 only.
- P25's narrowed reading, "the excess comes mostly through the baseline term", came from unaligned cells and masks. It does not describe the aligned measurement.

**Exploratory additions** (after the frozen report; same bootstrap, 2,000 draws; 1,000 for the horizon):
- **The common subspace carries broad response transfer on both receivers** (relative to its own random-subspace control; see the scope addendum on that control). R1 for the common basis is +0.126 [+0.052, +0.198] on 192 and +0.100 [+0.025, +0.166] on 256. That is about 62 % and 97 % of the full transfer's point estimates.
- **[Withdrawn in the scope addendum: this compares two effects with different controls and does not localize the effect.]** ~~On 192 the challenge-specific part lies outside the common subspace.~~ Full minus the common-basis transfer on R2 is +0.155 [+0.056, +0.265]. On 256 it is +0.009 [−0.088, +0.109].
  - On 192 the per-pair top-k subspace carries none of it either: −0.006 [−0.096, +0.075].
  - On 256 the per-pair subspace carries +0.112 [+0.024, +0.201].
  - Neither adaptive half-test is resolved.
  - **[Withdrawn in the scope addendum.]** ~~On 192, then, the specific response sits in the low-energy remainder of the difference~~: about 30 % of its energy, in directions shared by neither the pooled nor the pair's own leading subspace.
- **Horizon** (full − random; k = iterations after the change):

  | receiver | rule | k = 1 | k = 2 | k = 3 | k = 4 |
  |---|---|---|---|---|---|
  | 192 | R1 | +0.202 | +0.130 | +0.101 | +0.096 |
  | 256 | R1 | +0.103 | +0.067 | +0.059 | −0.019 |
  | 192 | R2 | +0.139 | +0.020 | +0.018 | +0.015 |
  | 256 | R2 | +0.102 | +0.040 | +0.088 | +0.070 |

  - On 192, R1 is resolved at every k. On 256 it is resolved at k = 1 only.
  - R2 is resolved at k = 1 only on both receivers. Persistence of the challenge-specific effect beyond one iteration is unresolved (corrected in the scope addendum); the broad response transfer persists on 192.

**For the program.**
- The second link of the target sentence ("an identified hidden component changes that requirement") now has its response form: the readout-invisible slow part carries correction responses, specifically and beyond matched random change.
- "Identified" means the whole readout-invisible difference. Compactness is not established (corrected in the scope addendum: on 192 the control comparison does not localize the effect).
- P27 (control) should therefore manipulate the full readout-invisible part, with the common subspace as a secondary arm on 256. Its readout should be at k = 1 first.

## P26 SCOPE ADDENDUM after an external review (2026-10-10 ~15:40Z; every number reproduced from the saved arrays; exploratory; `runs/analysis/rebuttal_20261012/review_check.py`, `pred_variants.py`)

**What stands.** The registered letters and their wording stand: RESPONSE-TRANSFER and CHALLENGE-SPECIFIC on both receivers; BASELINE-TRANSFER on 192 only; the registered COMPACT event fails. The test on particular, equally sized corrections cancels the no-edit baseline inside each contrast and holds the amount of correction fixed. So the transfer cannot be described as a change of the unedited continuation alone.

**Withdrawn: the localization of the specific transfer on 192.** The Outcome's exploratory reading ("on 192 the specific response sits in the low-energy remainder"; "full minus the common-basis transfer +0.155") compared two effects measured against *different* controls. Each transplant's own donor coefficient (R2 contrasts, relative to intact) shows this:

| condition (confirmation) | 192 | 256 |
|---|---|---|
| full | 0.191 | 0.105 |
| full_random (its control) | 0.053 | 0.003 |
| common-basis component (global) | 0.156 | 0.115 |
| global_rand (its control) | 0.173 | 0.022 |
| adaptive / adaptive_rand | 0.156 / 0.163 | 0.119 / 0.007 |

- **The direct contrast does not resolve.** Full minus the common-basis component is +0.035 [−0.041, +0.117] on 192 and −0.010 [−0.070, +0.047] on 256.
- **The two controls differ.** global_rand minus full_random is +0.120 [+0.046, +0.200] on 192 and +0.019 [−0.058, +0.094] on 256.
- **Why the compact control tracks the donor (the construction).**
  - It is the *same* difference Δ projected onto random channel directions and rescaled: ×2.38 on average on 192 (range 2.01–2.80), ×2.50 on 256.
  - One linear map is applied to every (field, cell) row, so the control keeps the difference's field-by-cell pattern.
  - On 192 it produces donor-associated responses about as strongly as the transplant itself. By contrast, the full transfer's control randomizes each row's direction independently.
- **So the effect has not been localized to the discarded component.** That component might matter alone, through interaction with the retained component, or the contrast may depend on control geometry. A remainder-only intervention with row-independent controls is needed (P27 draft).

**Compactness, revised reading.**
- On 256 the common-basis component reproduces the full transfer (direct difference −0.010 [−0.070, +0.047]) and beats its control (+0.093 [+0.015, +0.171]). "At least half" is unresolved because the registered test differences two noisy contrasts.
- On 192 the registered control comparison is uninformative.
- Leading singular directions preserve the largest variations in state; they do not directly identify the directions that matter for correction. The failure of this extraction does not rule out a compact mechanism.

**Persistence: unresolved, not "a one-iteration effect."** The specific transfer is resolved at k = 1. The later intervals include zero, and some later 256 estimates remain appreciable (+0.088 at k = 3).

**Sensitivity to the mask summary.** Centering all four masks within each state and dose, instead of the two registered contrasts, gives:
- +0.108 [+0.031, +0.179] on 192;
- +0.050 [−0.010, +0.110] on 256.

The registered two-contrast results stand. The sharper 256 effect deserves a fresh confirmation that uses all four masks jointly.

**Predictive value (fit on discovery puzzles, scored on confirmation puzzles; out-of-sample MSE reduction from adding the donor term).**

| target | 192 | 256 |
|---|---|---|
| the sharp contrast target (d_full − d_random on d_R) | +0.99 % [−0.28, +2.47] | +0.58 % [−0.12, +1.27] |
| the transfer-specific differential of the response curves | +1.9 % [−1.3, +4.8] | +1.1 % [+0.1, +2.1] |
| the baseline-conditioned response itself | +11.5 % | +4.2 % |

- The sharp-target figures reproduce the review's.
- The last row includes generic similarity between two histories of the same puzzle, so it is not transfer-specific.
- P26 identifies a reproducible causal influence. It is not yet a useful individual-level predictor.

**No performance gain from indiscriminate transplantation.** Full transplantation minus intact repair on the untouched cells (k = 1):

| arms | 192 | 256 |
|---|---|---|
| corrected arms | −1.9 pp [−4.3, +0.2] | −2.8 pp [−6.1, +0.5] |
| no-edit arms | −2.8 pp [−7.8, +2.5] | −7.2 pp [−15.1, 0.0] |

Transferring correction preferences and selecting beneficial ones are separate achievements.

**Revised mechanism reading: history-dependent configuration of correction responses.**
- The update has fixed weights, but retained state helps set how a change elsewhere affects an untouched decision, and part of that response structure follows the donor state when transplanted.
- "Learned relaxation with memory" now has a concrete meaning: memory configures how corrections propagate through the next update.
- P25 supplied collective, non-additive reconstruction; P26 shows a history-specific component of the response can be moved between histories.
- No claim is made of recovered constraints, the complete algorithm, or a small operator.

**Candidate manuscript wording (the PI decides):** "Score-preserving transplantation of retained state transfers history-specific responses to correction interventions, including differential responses to equally sized corrections, beyond matched random perturbations."
