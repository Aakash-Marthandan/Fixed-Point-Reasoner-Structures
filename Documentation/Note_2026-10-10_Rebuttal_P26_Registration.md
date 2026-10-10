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
