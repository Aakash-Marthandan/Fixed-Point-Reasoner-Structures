# Note 2026-10-11 — P27: where in the readout-invisible difference does the transferred correction response live? (registration, before any row; the PI's changes to the draft incorporated)

**Goal (page one).** A MEASUREMENT.
- Tool: `tools/rebuttal_p27.py`. It reuses P26's runner (release states, donor histories, padded rollouts, exact restoration) and P26's reader helpers unchanged.
- Outputs: `runs/analysis/rebuttal_20261013/`.
- Inference only, on the Mac's CPU. No training.
- Receivers: Attention 192 and Attention 256, **both confirming**.
- There is no discovery stage. The leading subspace is P26's frozen common basis, applied out of sample:
  - `tools/p26_basis_192.json`, sha256 d2a5e489…;
  - `tools/p26_basis_256.json`, sha256 420752a5…;
  - both learned on P26's discovery puzzles.
- **Primary statistic: the all-four-mask centered donor coefficient.**

P27 answers the two open items of P26's scope addendum (`Note_2026-10-10_Rebuttal_P26_Registration.md`):
1. **Where the specific transfer lives.** The arms are the leading component, its orthogonal remainder, and both, each against matched controls in a factorial design.
2. **A fresh confirmation of challenge specificity using all four masks jointly.** P26's sharper result on 256 depended on the mask summary (centered: +0.050 [−0.010, +0.110]).

The PI's changes to the draft, all adopted:
1. **The controls match each row's magnitude *and* its angle to the recipient.**
2. **A combined factorial control replaces the projected-control arm.**
3. **Interaction is tested against the corresponding random interaction.** Half-strength is measured against the total effect relative to the combined control.
4. **The "joint-only" rule is removed.** Component findings are not mutually exclusive.
5. **Geometry localization is kept separate.** It is not tested here.

## Design

**Population.**
- 384 fresh test puzzles per receiver: 48 from each of the full test set's eight rating octiles (rank-based bins).
- Excluded: the study's 512, the P12 / P17 / P20 / P22 attention pools, P25's pools and P26's pools.
- Seeds `[20261013, 27, width, octile]`.

**Pairs and aligned challenges.** P26's construction with P27's seeds:
- A is the intact fixed-start state after iteration 1. B is the same puzzle's Gaussian-start history after iteration 1 (seed `[20261013, puzzle, 7]`).
- A pair is kept when the two histories share at least 12 wrong cells (W = W_A ∩ W_B). Expected from P26's rates: about 208 pairs on 192 and 166 on 256.
- Four never-edited evaluation cells: seed `[20261013, puzzle, 1]`.
- Doses 0.25, 0.5 and 0.75 of the remaining shared wrong cells, with four equal-sized masks per dose (seeds `[20261013, puzzle, 2, dose, mask]`).
- Masks are identical for both histories. Both directions are used.

**Components.** Per pair-direction, with the recipient's slow state h_r and the readout-orthogonal parts perp_r and perp_d:
- Δ = perp_d − perp_r.
- L = U Uᵀ Δ, the projection onto the frozen basis (k = w/8: 24 on 192, 32 on 256).
- M = Δ − L, the orthogonal remainder. It is readout-orthogonal and U-orthogonal.

**Controls, matched in magnitude and angle.** For each (field, cell) row and each component C ∈ {L, M}, ξ_C is drawn inside C's own subspace, with
- ‖ξ_C‖ = ‖C‖ and ⟨h_r, ξ_C⟩ = ⟨h_r, C⟩.

Its component along the recipient's projection onto that subspace equals C's. The rest of its direction is a fresh random unit vector of the subspace, orthogonal to that projection. Properties:
- **Row norms preserved.** The resulting row's norm equals the component's: ‖h_r + ξ_C‖ = ‖h_r + C‖. The combined control has ‖h_r + ξ_L + ξ_M‖ = ‖h_r + Δ‖, because the two subspaces are orthogonal.
- **Degenerate rows (frozen now):**
  - ‖C‖ < 1e-12 gives ξ_C = 0.
  - ‖P_C h_r‖ < 1e-8 ‖h_r‖ drops the inner-product constraint, and ξ_C = ‖C‖ times a random unit vector of the subspace.
- **Drawn once.** Each ξ_C is drawn once per pair-direction (seeds `[20261013, puzzle, 8, direction]` for L and `[…, 9, direction]` for M) and reused across all thirteen arms. The combined control uses the *same* ξ_L and ξ_M.
- **What they do and do not remove.** These controls randomize directional information. They preserve the donor-derived row-magnitude pattern and each row's angle to the recipient. **They do not remove all spatial structure.**

**Conditions.** Eight conditions, each with its own no-edit continuation; 13 arms each (no-edit + 12 masks):

| condition | perturbation of the recipient's slow state |
|---|---|
| intact | none |
| full | L + M (= Δ) |
| full_random | P26's angle- and norm-matched rotation (the original full control; kept for the replication) |
| lead | L |
| lead_ctrl | ξ_L |
| rem | M |
| rem_ctrl | ξ_M |
| comb_ctrl | ξ_L + ξ_M (the same draws): the combined control for the component analysis |

Every perturbation is readout-orthogonal, so the displayed scores are kept. **Exact restoration:** after every other variant is built, the saved tensors are written back and must reproduce the intact rollouts bitwise (no-edit and one mask per dose, all four iterations).

## The primary statistic

All quantities are at k = 1 (one iteration after the change), on the four evaluation cells.
- Within each (state, dose), the four masks' outcomes Y (the share of evaluation cells showing their solution digit) are centered.
- The centered difference (Y_X − Y_ref) is regressed on the centered donor intact outcomes and the centered recipient intact outcomes, across states, doses and masks, by ordinary least squares.
- The donor coefficient is the statistic.
- β(X) denotes it with ref = intact, and β(intact) = 0 by construction. A contrast "X vs ref" equals β(X) − β(ref).
- Intervals: 95 % puzzle bootstrap, 2,000 draws, keeping both directions and every challenge of a puzzle together. Joint quantities are bootstrapped jointly.

## Rules (per receiver; a claim requires both receivers, and a single receiver is reported as such)

**S — CHALLENGE-SPECIFIC-REPLICATES:** β(full) − β(full_random) has a lower bound > 0. This is the fresh, four-mask confirmation of P26's sharp result against P26's control.

**The component analysis (against the combined control).**
- **Total:** E_total = β(full) − β(comb_ctrl). TOTAL-RESOLVED if its lower bound is > 0.
- **Component effects:** E_L = β(lead) − β(lead_ctrl) and E_M = β(rem) − β(rem_ctrl).
  - LEADING-CARRIES or REMAINDER-CARRIES if the lower bound is > 0.
  - **The two are not mutually exclusive.** Both parts could carry substantial effects independently.
- **Half-strength** (only where TOTAL-RESOLVED; otherwise "n/a"): LEADING-HALF if E_L − ½ E_total has a lower bound > 0; REMAINDER-HALF likewise for E_M. Bootstrapped jointly.
- **Direct differences:** β(full) − β(lead) and β(full) − β(rem), reported with intervals, with no letters.
- **Interaction against the corresponding random interaction:**
  - I_signal = β(full) − β(lead) − β(rem);
  - I_control = β(comb_ctrl) − β(lead_ctrl) − β(rem_ctrl);
  - their difference is bootstrapped jointly by puzzle;
  - POSITIVE if the lower bound is > 0, NEGATIVE if the upper bound is < 0, otherwise UNRESOLVED.
  - If the interaction is positive while neither component effect is resolved, the description is exactly: "positive interaction beyond the random interaction; individual component effects unresolved."
- **No "joint-only" rule.** A significant full transfer with non-significant components does not establish that both components are necessary.

**Secondary (reported):**
- P26's two-contrast specificity;
- BASELINE and R1 as registered in P26;
- each condition's donor coefficient;
- retained energies ‖L‖² / ‖Δ‖² and ‖M‖² / ‖Δ‖².

**Not tested here: geometry localization.** Comparing P26's projected control with these controls changes several geometric properties at once, so it cannot isolate why the old control tracked the donor. A future geometry test should scramble the projected control within its own subspace while matching its row geometry.

## Predictions and credences (before any P27 row)

| event | credence |
|---|---|
| S on both receivers | 0.50 |
| TOTAL-RESOLVED on both | 0.50 |
| LEADING-CARRIES on 192 | 0.55 |
| LEADING-CARRIES on 256 | 0.60 |
| LEADING-CARRIES on both | 0.40 |
| REMAINDER-CARRIES on 192 | 0.30 |
| REMAINDER-CARRIES on 256 | 0.20 |
| LEADING-HALF on 192 | 0.35 |
| LEADING-HALF on 256 | 0.45 |
| REMAINDER-HALF on 192 | 0.15 |
| REMAINDER-HALF on 256 | 0.10 |
| POSITIVE interaction, on each receiver | 0.15 |

## Wording under each outcome

- **S:** "On fresh puzzles, using all four equal-sized corrections jointly, the transplant again transfers the donor's differential responses beyond a matched random change."
- **LEADING-CARRIES / REMAINDER-CARRIES:** "The [leading / remainder] component alone transfers part of the donor's differential responses beyond a control matched in each row's magnitude and angle to the recipient."
- **…-HALF:** "… at least half of the total transfer measured against the combined control."
- **Interaction:** report I_signal, I_control and their difference. Use the registered description when it applies.
- **No component resolved:** "Where in the difference the specific response lives is not resolved at this sample size." The intervals bound each part's possible share.

Scope under every outcome:
- oracle-guided diagnostic corrections;
- constructed hybrids;
- one checkpoint per receiver; attention receivers;
- a one-iteration primary horizon (persistence was unresolved in P26);
- four evaluation cells per state.

The controls keep row magnitudes and angles, so they do not remove all spatial structure. Leading singular directions select variance, not correction relevance, so a negative result for L does not rule out a compact mechanism.

## Gates

- P26's gates:
  - release states reproduce;
  - masks lie within the shared wrong cells and avoid E;
  - every relabel produces its designed display;
  - no evaluation cell is edited;
  - displayed scores are kept (≤ 1e-3);
  - exact restoration is bitwise.
- L + M = Δ (≤ 1e-12).
- M, ξ_M ⊥ U and ⊥ the readout; ξ_L ∈ span(U) (≤ 1e-9).
- Per-row norm and inner-product matches (≤ 1e-6 relative).
- Row norms of the resulting states match (≤ 1e-6).
- Degenerate rows are counted.
- The frozen basis hashes are asserted.

## Smoke and disclosures

Gates-only smokes ran concurrently on the pool's first six puzzles at both widths. Every gate passed:
- restoration bitwise;
- split 2e-16;
- subspace memberships ≤ 2e-14;
- norm and inner-product matches ≤ 4e-15;
- no degenerate rows.

The reader ran on the smoke files with its output suppressed: no smoke statistic was seen. Those puzzles belong to the pool and are included.

## Compute

Variants: 8 conditions × 13 arms + 4 restoration variants = 108 per state, giving ≈ 44,900 (192) and ≈ 35,900 (256).

Measured on the smoke with both receivers concurrent: 44–49 s (192) and 61–66 s (256) per 128-variant chunk (47.5 s for 256 alone). That projects to ≈ 5 h on the Mac.
- This is over the ~3 h threshold; the PI chose 384 puzzles per receiver.
- AC power and an open lid are needed for the run: closing the lid sleeps the Mac and pauses the run.

## P27 Outcome (2026-10-11 08:47Z; `runs/analysis/rebuttal_20261013/report.{txt,json}`; every gate passed on both receivers; no tool edited during the run)

**Run.**
- Eligible pairs: 192 on Attention 192 (384 states, 39,936 variants) and 155 on Attention 256 (310 states, 32,240 variants).
- Exact restoration was bitwise.
- L + M = Δ to 4e-16. Subspace memberships ≤ 2.3e-14. Norm and inner-product matches of the controls ≤ 7e-15. Resulting row norms matched to ≤ 4e-15. No degenerate rows.
- Retained energy: L holds 0.703 (192) and 0.775 (256) of ‖Δ‖²; M holds the remaining 0.297 and 0.225.

**Per-condition donor coefficients** (all-four-mask statistic; relative to intact):

| condition | 192 | 256 |
|---|---|---|
| full | 0.234 | 0.110 |
| full_random | 0.057 | 0.025 |
| lead | 0.233 | 0.053 |
| lead_ctrl | 0.042 | −0.011 |
| rem | 0.065 | 0.048 |
| rem_ctrl | 0.020 | 0.023 |
| comb_ctrl | 0.067 | 0.002 |

**The registered rules (95 % puzzle-bootstrap intervals).**

| rule | Attention 192 | Attention 256 | both receivers |
|---|---|---|---|
| S: full − P26's control | +0.177 [+0.119, +0.244] REPLICATES | +0.085 [+0.036, +0.140] REPLICATES | **yes** |
| (two-contrast, secondary) | +0.189 [+0.122, +0.256] | +0.100 [+0.036, +0.166] | |
| E_total: full − combined control | +0.167 [+0.104, +0.237] | +0.108 [+0.050, +0.164] | resolved on both |
| E_L: lead − lead_ctrl | +0.191 [+0.126, +0.254] LEADING-CARRIES | +0.063 [+0.001, +0.127] LEADING-CARRIES | **yes** |
| E_M: rem − rem_ctrl | +0.045 [+0.007, +0.082] REMAINDER-CARRIES | +0.025 [−0.016, +0.068] | 192 only |
| E_L − ½ E_total | +0.108 [+0.054, +0.164] LEADING-HALF | +0.010 [−0.047, +0.066] | 192 only |
| E_M − ½ E_total | −0.039 [−0.088, +0.008] | −0.029 [−0.079, +0.022] | no |
| direct: full − lead | +0.001 [−0.044, +0.046] | +0.057 [+0.013, +0.103] | |
| direct: full − rem | +0.169 [+0.111, +0.228] | +0.062 [+0.009, +0.122] | |
| I_signal | −0.064 [−0.121, −0.007] | +0.009 [−0.053, +0.072] | |
| I_control | +0.005 [−0.060, +0.068] | −0.010 [−0.067, +0.043] | |
| interaction difference | −0.069 [−0.143, +0.007] | +0.019 [−0.058, +0.099] | UNRESOLVED on both |

Secondary, as registered:
- BASELINE (P26's): +0.223 [+0.110, +0.333] on 192 and +0.147 [+0.040, +0.251] on 256.
- R1 (P26's): +0.255 [+0.198, +0.314] and +0.133 [+0.073, +0.192].

**Recount.** Independent code reproduces every registered point estimate exactly. It has its own unit tables, whole-array four-mask centering, QR least squares and donor mapping.
- The npz hashes are verified.
- An independent resampler gives intervals within about 0.01 of the report's.
- Regenerated pairs' leading energies match the stored values exactly.
- Borderlines, resolved in both resamplers:
  - 256's E_L lower bound: +0.001 registered, +0.005 in the recount;
  - 192's E_M lower bound: +0.007 and +0.009.
- 192's interaction difference stays unresolved in both: [−0.143, +0.007] and [−0.145, +0.010].

**Predictions.**

| event | credence | outcome |
|---|---|---|
| S on both | 0.50 | yes |
| TOTAL on both | 0.50 | yes |
| LEADING-CARRIES 192 / 256 / both | 0.55 / 0.60 / 0.40 | yes / yes / yes |
| REMAINDER-CARRIES 192 / 256 | 0.30 / 0.20 | yes / no |
| LEADING-HALF 192 / 256 | 0.35 / 0.45 | yes / no |
| REMAINDER-HALF 192 / 256 | 0.15 / 0.10 | no / no |
| positive interaction, each receiver | 0.15 | no / no |

Brier score over the 13 events: 0.189.

**Wording (registered).**
- **S, both receivers:** "On fresh puzzles, using all four equal-sized corrections jointly, the transplant again transfers the donor's differential responses beyond a matched random change." P26's sharp result is confirmed with the stronger statistic, including on 256, where P26's exploratory check had been sensitive to the mask summary.
- **LEADING-CARRIES, both receivers:** "The leading component alone transfers part of the donor's differential responses beyond a control matched in each row's magnitude and angle to the recipient." The leading component is the projection onto a common subspace of one eighth of the channels, learned on other puzzles in P26.
  - On 192: LEADING-HALF; the direct difference full − leading is +0.001 [−0.044, +0.046].
  - On 256: the lower bound sits at zero, and the full transfer exceeds the leading part alone (+0.057 [+0.013, +0.103]).
- **REMAINDER-CARRIES, 192 only:** a small resolved part (+0.045), not half.
- **Interaction:** unresolved on both receivers against the random interaction. On 192 the signal interaction alone is negative (−0.064 [−0.121, −0.007]): the two parts' effects overlap rather than add. Against the random interaction this is not resolved.
- **On 256, the exact description:**
  - the leading component carries a resolved part (lower bound at zero);
  - the full transfer exceeds each part alone;
  - the remainder's own effect and the interaction are unresolved.

  This does *not* establish that both parts are necessary.

Scope under every outcome:
- oracle-guided diagnostic corrections;
- constructed hybrids;
- one checkpoint per receiver; attention receivers;
- a one-iteration horizon;
- four evaluation cells.

The controls keep row magnitudes and angles; they do not remove all spatial structure.

**What changes against P26.**
- P26's exploratory reading that the specific transfer on 192 sat in the remainder was withdrawn in its addendum. P27 resolves the question the other way: with controls matched per row in magnitude and angle, P26's frozen common subspace carries the specific transfer on both receivers, and on 192 it reproduces the full transfer.
- The row-independent controls track the donor little (0.042 / 0.020 / 0.067 on 192). This is consistent with the addendum's explanation of P26's projected control (0.173). It is not a test of that explanation: geometry localization was registered as not tested.

**For the program.**
- **A reusable compact component.** A common, low-dimensional component, learned on other puzzles, now carries a transferable, challenge-specific correction response on both receivers. On 192 it carries at least half.
  - This is the first compact candidate: 24 / 32 of 192 / 256 channels.
  - It is reusable across puzzles, as the selection experiment (P28) needs.
- **What it is not.** It is not yet a predictor of which corrections become productive, and not a performance gain. P26's addendum showed indiscriminate transplantation does not raise repair.
- **P28 can use the leading component as a candidate intervention** beside the full transplant (on both receivers, not on 256 alone as staged).
