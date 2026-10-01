# Note 2026-10-01 — P3: a one-time, score-preserving edit of the readout-invisible slow state across the corruption ladder (registration, before any row)

**Goal (page one).** A MEASUREMENT. No accuracy target; $0; the Mac's CPU; inference only on the manuscript's banked attention checkpoints; nothing trained. Follows `Note_2026-09-26_Review_Period_P1_P2_Registration.md` (P2's repeated score-preserving replacement) and `Note_2026-09-26_Review_Period_P1c_P2b_Registration.md` (the mutually consistent corruptions). Plan: `Plan_2026-10-01_Rebuttal_Experiments.md` §2. Tool: `tools/rebuttal_p3.py` (imports the P1/P2 pipeline unchanged; outputs `runs/analysis/rebuttal_20261001/`). Registered before the tool is built; the rules below are fixed first.

## Why

The manuscript establishes two dependencies separately: repair depends on the configuration of the wrong digits beyond their number (§4: structured, mutually consistent corruptions are repaired about as slowly as the released model's own grids), and later correction depends on slow-state features that the linear readout cannot see (§4.2: replacing them before every iteration 2–16 lowers endpoints from 250 to 171–230 of 256). Not established: whether repairing difficult configurations draws **more** on the readout-invisible state than repairing easy ones; and whether P2's loss is a one-time setback the trajectory recovers from or the accumulation of fifteen disruptions. Both are the questions an outside reader raised on 2026-10-01 as the missing connection between the paper's strongest findings.

## Design

**Receivers.** Attention 128 / 192 / 256 at their benchmark weights, through the manuscript's own pipeline (`tools/rebuttal_p1p2.R2`, release code), batch 128, 16 iterations, float32 CPU, the bitwise gate against the study's archived intact first batch re-run per width.

**Two populations.**

1. *Repair ladder* — the 512 repair puzzles with their four saved draw-0 corruptions at EqR's wrong-cell counts: `uniform_0` (the study's random draw), `legal_random_0` (clue-compatible, P1), `consistent_random_0` (mutually consistent, P1c), and `eqr` (the released model's own grid). Analysis retains the 438 source-error pairs, as in P1/P1c. Each grid is embedded into the slow state and run from the fixed fast state, exactly as before.
2. *Shared first states* — the 256 intervention puzzles of the study, continued from the shared intact first state (P2's population), to compare the one-time edit with P2's repeated edit.

**Edits.** For each slow-state vector h (one per field and cell) write h = h∥ + h⊥ with h∥ the component along the readout vector, so that every digit score is unchanged by any edit of h⊥. Applied **once**, before iteration t₀ (zero-based step t = t₀ − 1), with the trajectory otherwise intact:

- `random_t2`, `random_t4`: h⊥ replaced by a random vector orthogonal to the readout with the same norm (P2's `null_random` edit, once). Seeds `[20261001, puzzle ID, 41, t]`.
- `rot30_t2`, `rot60_t2`: h⊥ rotated by 30° or 60° toward a random unit direction orthogonal to both the readout and h⊥; the norm is preserved and the displacement is 2‖h⊥‖ sin(θ/2) (full replacement averages about √2‖h⊥‖). Seeds `[20261001, puzzle ID, 42, t]`.
- `sham_t2`: h decomposed and reassembled (float64 arithmetic, cast back to float32) before iteration 2, with nothing changed. The numerical control for the edit path.

Score preservation is asserted at every edit (max digit-score shift below 1e-2 in float64 as in P2; the realized float32 shift is recorded). Reference trajectories without edits exist for every family (the study's `eqr` and `random` draw 0, P1's `legal_random_0`, P1c's `consistent_random_0`, the study's `intact`), all with per-iteration logits; they are read, not re-run.

**Conditions and order.** Repair ladder: 4 families × {`sham_t2`, `random_t2`} first (the primary set), then `random_t4`, then `rot30_t2`, `rot60_t2`. Shared first states: `sham_t2`, `random_t2`, `random_t4`. About 23 runs of 512 or 256 puzzles per width; widths in parallel.

**Gates.** (1) The study's intact first batch reproduced bitwise at all 16 iterations (the P2 gate). (2) Sham neutrality on batch 0 of the `eqr` family: the exact-correctness flags of the sham trajectory equal those of the saved reference at every iteration; the maximum absolute logit difference is recorded. If (2) fails the run continues and the report says SHAM NOT NEUTRAL; every comparison below uses the sham as its reference regardless.

## Quantities (per receiver, per family, per edit)

Let the *pre-edit state* be the prediction after iteration t₀ − 1 of the sham trajectory. Puzzles are stratified by their wrong empty cells there: S1 = 1–5, S2 = 6–15, S3 = 16–30, S4 = 31+; solved puzzles (0 wrong) form the preservation set. Strata with fewer than 20 puzzles in a family are reported but not used in a letter.

- **ΔE** (early sensitivity): mean empty-cell error at iteration t₀ under the edit minus under the sham, over unsolved pre-edit puzzles, per stratum.
- **L** (completion loss): share of puzzles exact at 16 under the sham that are not exact at 16 under the edit, per stratum.
- **Δτ** (delay): median of (first exact iteration under the edit − under the sham) over puzzles exact at 16 under both, per stratum; right-censoring stated.
- **Pres** (preservation): among puzzles solved pre-edit, the share wrong at any iteration ≥ t₀ under the edit, and the share wrong at 16.
- **Shared first states:** exact at 16 under `random_t2` and `random_t4` against intact (250) and P2's repeated `null_random` (171 / 212 / 230); initial solutions lost; new discoveries.

## Rules (confirmatory)

**R1 — configuration dependence (primary; `random_t2`; strata S2–S4 with ≥ 20 puzzles in both families; the structured family = `consistent_random_0`, the easy reference = `legal_random_0`).** For each eligible stratum compute the ratio ρ = ΔE(structured) / ΔE(legal). CONFIGURATION-DEPENDENT if ρ ≥ 1.5 in every eligible stratum on all three widths and L(structured) ≥ L(legal) in every eligible stratum; INDEPENDENT if 0.75 ≤ ρ ≤ 1.33 in every eligible stratum on all three widths; MIXED otherwise; UNDEFINED if ΔE(legal) is below a 1-pp floor in an eligible stratum (then L alone decides with the same thresholds, and the note says so). The `eqr` family is reported beside `consistent_random_0` and must agree in direction for the letter to hold on model-produced grids; `uniform_0` is reported with whatever strata it populates.

**R2 — one-time versus repeated (shared first states; `random_t2`).** R₁ = (exact₁₆ under `random_t2` − exact₁₆ under P2's repeated `null_random`) / (250 − exact₁₆ under repeated). RECOVERS if R₁ ≥ 0.75 on all three widths (the one-time edit loses at most a quarter of what the repeated edit loses); PERSISTS if R₁ ≤ 0.25; MIXED otherwise.

**Exploratory, no letter:** dose–response (ΔE and Δτ monotone in θ across `rot30`, `rot60`, `random`); `random_t4` against `random_t2`; preservation under all edits; the family × stratum table in full; even/odd puzzle-ID halves as a stability check.

## Predictions and credences (before any row)

R1: CONFIGURATION-DEPENDENT 0.35; INDEPENDENT 0.35; MIXED 0.30. R2: RECOVERS 0.55; MIXED 0.30; PERSISTS 0.15. Sham neutral on batch 0 (identical exact flags at every iteration): 0.8. Preservation: solved pre-edit puzzles wrong at 16 under `random_t2` ≤ 2 % per family: 0.7. Dose–response monotone in ΔE on all widths: 0.7.

## Wording under each outcome

- **CONFIGURATION-DEPENDENT.** "A single score-preserving replacement of the slow-state features the readout cannot see, applied once after the first iteration, delays repair more for mutually consistent corruptions than for clue-compatible random ones at matched remaining error: difficult configurations draw more on state beyond the displayed answer." The manuscript's two dependencies become one claim; the hypothesis gains direct support without identifying what the features encode.
- **INDEPENDENT.** "The one-time replacement delays repair by similar amounts across corruption families at matched remaining error: dependence on readout-invisible state is general, not specific to difficult configurations." The two dependencies stay separate findings; the hypothesis is not supported by this test.
- **MIXED.** ρ reported per stratum with both readings.
- **RECOVERS.** "A single replacement is a setback the trajectory recovers from; P2's loss reflects repeated disruption, not the loss of history at one moment." **PERSISTS.** "One replacement costs most of what fifteen cost: the discarded features are not rebuilt within the horizon." Under every outcome: no feature content is identified; the edits are random directions, not learned ones.

## Labels and plan

Confirmatory: R1 and R2 letters. Exploratory: everything else. Build after this note is committed; selftest (projection, rotation norm and orthogonality, the sham's identity, one mutant) and a 16-puzzle smoke with both gates must pass before launch; three widths in parallel under caffeinate; report by the tool; the Outcome section appended here and a ledger line written when read.

---

## P3 Outcome (2026-10-01 18:08Z; `runs/analysis/rebuttal_20261001/report.{txt,json}`; every gate passed; no shared tool edited)

**Integrity.** On all three widths the study's intact first batch was reproduced bitwise, and the sham edit (decompose, reassemble, cast) reproduced the saved reference trajectories bitwise: on batch 0 by the gate, and in the read on all 512 puzzles at every iteration for all four families, twelve of twelve width-by-family comparisons. Every edit preserved the digit scores (largest float64 shift 1.4e-14; largest realized float32 shift 5e-5). An independent recount from the chunk files agrees with the report on every quantity entering R1 and R2. Reader-side fixes made and committed before any ladder result was read (a9bfabb, 7906ec2): the population restricted to the 438 registered source-error pairs, incomplete conditions skipped, and the registered fallback implemented (where the clue-compatible family's early-error change is below the 1-pp floor, completion losses decide with the same thresholds). The rules themselves were not changed.

**R2 — one-time versus repeated (shared first states).**

| receiver | `random_t2` at 16 | P2's repeated edit | intact | R₁ | `random_t4` at 16 | initial solutions lost |
|---|---|---|---|---|---|---|
| Attention 128 | 243 | 171 | 250 | 0.911 | 240 | 0 |
| Attention 192 | 248 | 212 | 250 | 0.947 | 249 | 0 |
| Attention 256 | 251 | 230 | 250 | 1.050 | 251 | 0 |

**Letter: RECOVERS.** The edit dips the count at the iteration it precedes (125 → 78, 173 → 131, 165 → 149 at iteration 2) and the trajectory then repairs the setback within a few iterations; no initial solution is lost under any edit on any width. P2's loss was the accumulation of fifteen replacements, not the loss of history at one moment.

**R1 — configuration dependence (`random_t2`; eligible strata S3 and S4, S2 having fewer than 20 clue-compatible puzzles on every width).**

| receiver | stratum | clue-compatible ΔE / L | mutually consistent ΔE / L | ρ (basis) | EqR's own grids ΔE | L condition |
|---|---|---|---|---|---|---|
| Attention 128 | S3 (16–30 wrong) | +8.07 pp / 2.5 % | +11.47 pp / 3.2 % | 1.42 (ΔE) | +10.35 pp | holds |
| Attention 128 | S4 (31+) | +2.64 pp / 4.7 % | +6.41 pp / 7.1 % | 2.43 (ΔE) | +8.76 pp | holds |
| Attention 192 | S3 | +11.22 pp / 1.9 % | +2.08 pp / 3.4 % | 0.19 (ΔE) | −2.88 pp | holds |
| Attention 192 | S4 | −7.33 pp / 5.4 % | −8.84 pp / 4.8 % | 0.88 (L, fallback) | −14.24 pp | fails |
| Attention 256 | S3 | +14.79 pp / 4.3 % | +1.88 pp / 1.6 % | 0.13 (ΔE) | −2.41 pp | fails |
| Attention 256 | S4 | +0.84 pp / 2.5 % | +0.69 pp / 4.0 % | 1.60 (L, fallback) | −0.86 pp | holds |

**Letter: MIXED.** No width has every eligible stratum in one band: ρ ≥ 1.5 in 128's S4 and in 256's S4 by the fallback, ρ < 0.75 in 192's S3 and 256's S3, and 128's S3 sits at 1.42.

**Both readings, as registered for MIXED.** On Attention 128 the one-time replacement costs the mutually consistent corruptions more early repair than the clue-compatible ones in both strata, and the model's own grids more still: the configuration-dependent reading. On Attention 192 and 256 the sign reverses in the middle stratum: the replacement costs the clue-compatible grids 11 and 15 points of early repair and the consistent grids 2, while EqR's own grids repair slightly faster after it (−2.9 and −2.4 pp); in the largest-error stratum on 192 the replacement lowers error for every family (−7 to −14 pp). On the two wider models the readout-invisible state after one iteration carries work that a clue-compatible repair needs and little that a difficult configuration's repair needs.

**Exploratory.** *Later edit.* Before iteration 4 the replacement costs the consistent and model families 12–17 pp of early repair in S3 on every width (consistent 14.6 / 13.1 / 12.4; EqR 17.0 / 13.8 / 14.7), where before iteration 2 it cost 2–11 pp or helped; for the clue-compatible family the later edit costs about the same or less (9.4 / 9.1 / 5.2 against 8.1 / 11.2 / 14.8). The invisible state becomes load-bearing for difficult configurations once their repair is under way, not at its start. *Dose.* On 128 the early-error cost rises with the edit's size for every family in S3 (30°: 0.1 / 0.4 / 2.3 pp; 60°: 3.7 / 4.0 / 4.9; full: 8.1 / 11.5 / 10.4 for clue-compatible / consistent / EqR); on 192 and 256 rotations of the consistent and model grids often lower early error (−0.7 to −4.2 pp) and only the clue-compatible full replacement is clearly costly, so the registered monotone dose–response holds on 128 only. *Preservation.* Under every edit on every width and family, no puzzle solved before the edit is wrong at iteration 16; at most 1.9 % are wrong at some later iteration. *Endpoints.* Exact-at-16 counts move by at most eight of 438 under any edit; the effects are on the early trajectory, not the endpoint. *Completion losses* stay under 10 % except in the smallest uniform strata; *delays* are 0–1 iterations where measurable.

**Predictions scored.** R1 MIXED (0.30) HIT. R2 RECOVERS (0.55) HIT. Sham neutral (0.8) HIT, exactly. Preservation ≤ 2 % wrong at 16 (0.7) HIT, at 0 %. Dose–response monotone on all widths (0.7) MISS: 128 only.

**Reading, for the manuscript and the hypothesis.** The two findings this experiment was built to connect do not join in the way proposed: difficult configurations do not draw *more* on readout-invisible state at the start of repair; on two of three widths they draw less than easy ones, and their slowness is set by the displayed configuration itself, which is what the structured-corruption result (P1c) found from the other side. What the edits establish instead: (1) §4.2's dependence on state beyond the answer scores is a dependence on repair in progress, strongest for clue-compatible corruptions after one iteration and for every family after three, and a single disruption of it is recovered; the repeated replacement's loss in the manuscript is cumulative and should be described so. (2) The hypothesis clause "errors that reinforce one another internally may resist isolated correction" is not supported as a readout-invisible phenomenon at the first iteration on Attention 192 and 256: replacing the invisible state of a mutually consistent configuration does not release it and slightly helps. (3) The other session's registered reads of this day (uncommitted; `Note_2026-10-01_Rebuttal_P10_P11_Registration.md`, `…P12…`) supply the candidate explanation for the width-dependent sign: a sensitive transient two to three iterations before completion, in which small score-preserving rotations move completion earlier as often as later, and a one-time random replacement of the invisible part that helps puzzles still unsolved at 16. Under every outcome, as registered: no feature content is identified; the edits are random directions, not learned ones.
