# Note 2026-10-03 (revised 2026-10-08) — What the discussion-period experiments established: distributed correction, learned recurrence, and sensitivity without control

**Purpose.** A synthesis for the PI's decision on the rebuttal narrative. Every claim cites a registered outcome; exploratory observations are marked. Nothing here edits the manuscript.

*Revision 2026-10-08.* This revision follows an external review that rechecked the results. Its factual claims were reproduced from the saved arrays before this revision. Four corrections:
- P24's architecture-rate comparison was unpaired (P24 Addendum).
- Amendment 3's "frozen one step before completion" holds on Attention 192 only (P15 Amendment 7).
- "Memoryless" is a 16-iteration-window statement with short-window transients (P22–P24 addenda).
- "Chaotic" is operational sensitivity, not a dynamical measurement.

It also adds SE-RRM attribution round 3 (session d378a5) and the paired shared-failure analysis. The file name is kept so that earlier references resolve.

## 1. The short answer

The experiments materially strengthen the empirical paper, and they change the mechanism we should argue for:
- **Established:** correction is distributed and causally coupled.
- **Established:** the architecture and its regularization determine how much further recursion pays.
- **Established:** hidden-state perturbations strongly change *which* trajectory occurs without changing *whether* it succeeds.
- **Not recovered:** a compact correction operator.
- **Not shown:** a reliable intervention that improves reasoning at inference time.

## 2. Distributed, causally coupled correction

- **Local handles are weak.**
  - P15: helpful display-only flips realize 2–7 % of the available next-step gain.
  - P16: whole-cell edits act more strongly but do not reliably install the intended decision. The commitment dial fails, and its hidden index changes predictive AUC by −0.001 to +0.002.
  - P18: Sudoku dependency closures are no better than size-matched random corrections.
  - P17: perturbing conflict cells equals perturbing random cells.
  - An external review's exploratory fit of the measured score-space operator to a residual-correction factorization (K ≈ BA) leaves a median 59–66 % irreducible squared residual. It is reported by that review and not yet reproduced here. The measured operator should not be presented as a recovered residual-correction rule.
- **Sufficiently distributed correction induces repair elsewhere (P19, the strongest positive result here).**

  | receiver | completion at the next iteration: no intervention | half of the wrong cells corrected | 80 % corrected |
  |---|---|---|---|
  | Attention 256 | 48.6 % | 69.6 % | 90.9 % |
  | Attention 192 | 46.4 % | 72.0 % | 92.0 % |
  | Attention 128 | 31.5 % | 46.9 % | 51.2 % |

  On the two wider models, correcting half the wrong cells also closes about 40–52 % of the remaining repair gap on the cells that were *not* edited. The response is gradual, or mixed, rather than a sharp threshold. Two qualifications:
  - These are oracle-guided diagnostic interventions, not a deployable improvement.
  - "Centred near half" describes the normalized population improvement, not a per-puzzle 50 % threshold.
- **Completion needs information beyond the scores and needs evolving communication.**
  - P8 and session d378a5's P13: a score-preserving replacement at the completing iteration blocks completion on 78/78 single-state cases and 77/94 and 55/79 two-state cases, while 1° controls block 6/78, 5/94 and 0/79.
  - P4: freezing messages from iteration 4, 6 or 8 produces zero new completions.
  - P3: one replacement largely recovers, but repeated replacements do not.
  - P9/P11: transferring a successful trajectory's readout-invisible slow component rescues the next step on every selected primary case. A control-accounting error was found in the P9 runner: 11 of 15 controls were also primary cases intervened at a different time, so the full control sample does not stand. Reported to session d378a5 for correction.

## 3. Learned recurrence and regularization (SE-RRM attribution rounds, session d378a5)

On the identical 5,000 test puzzles:

| recurrent organization | training dropout | depth 16 | depth 64 |
|---|---:|---:|---:|
| two states | none | 98.24 % | 99.60 % |
| one state | none | 89.96 % | 93.08 % |
| two states | 0.2 | 97.98 % | 99.82 % |
| one state | 0.2 | 95.78 % | 98.04 % |

- Dropout shrinks the two-state advantage from 8.28 to 2.20 points at depth 16, and from 6.52 to 1.78 at depth 64.
- It also prevents the single-state run's late validation collapse. Validation goes from a maximum of 92.77 to 71.88 at the end without dropout, against 97.27 to 96.68 with it.
- The registered label is REGULARIZATION-REPLACES; scientifically, "regularization substantially substitutes for the architectural advantage". The remaining differences are clear on these checkpoints, and there is one seed per arm.
- **Early versus late.** The single-state models solve more puzzles at the first iterations, yet the two-state models finish ahead.
- **Paired shared failures** (exploratory, reproduced). Among puzzles both models leave unsolved at iteration 16, the two-state model completes more by 64:
  - 44 against 10 of 61 without dropout;
  - 34 against 20 of 43 with dropout.

  The architecture advantage includes greater success from continued computation on the same failures, even after dropout narrows the gap. P24's cohort comparison is withdrawn in favour of this paired count.

## 4. Sensitivity without control

- **The confirmation sequence.**
  - P12: a pilot gain from one random hidden-state replacement.
  - P17: a 60° rotation looked strong across widths.
  - P20: on fresh puzzles, the rotation matched intact continuation: 71 / 80 / 68 against 75 / 83 / 67.
  - P21: no benefit on three MLP seeds at any horizon to 64.

  P17's repeated-kick policy was effectively untested: unsuccessful trajectories keep changing their displays, so its trigger almost never fired.
- **The distinction to carry forward.** About a third of stalled outcomes change under a perturbation that keeps every score, with gains and losses balanced. *Controlling which trajectory occurs is much easier than controlling whether it succeeds.* A convincing controller must predict a beneficial direction before seeing the outcome.
- **A window-level statistical description of difficult continuations (P22–P24).** For a fixed stalled state, completion over the second sixteen iterations is what the first sixteen predict. The survivors-over-predicted ratio R is near 1:
  - attention: 1.08 / 0.91 / 1.08;
  - MLP: 1.16 / 0.86 / 0.95;
  - SA pair: 1.00 / 1.07, and 0.86 on the stage-matched checkpoint, which is UNRESOLVED.

  This supports a coarse model, P(complete within n windows) ≈ 1 − (1 − p_x)^n, with a per-puzzle window probability p_x. Two limits:
  - Shorter windows show transients after the perturbation. On most two-state receivers R is 1.36–2.62 at two iterations. The single-state model differs at short windows (0.27 / 0.70 / 0.65 at 2 / 4 / 8 iterations).
  - Comparable outcome disagreement at 1° and 60° establishes strong sensitivity, not exponential divergence or mixing.

  So this is *not* evidence for a memoryless internal algorithm or for chaos, and it should not be presented as either.

## 5. Status of the original proposal (plan §7)

| requirement | status |
|---|---|
| Hidden state causally changes behaviour beyond the current scores | Established (P8/P13, P9, P15–P16, P3–P4) |
| A compact identified structure predicts finite correction responses | Not established; the tested commitment index fails (P16) |
| Editing that structure selectively produces the predicted response | Not established (P16–P18) |
| The intervention improves success at matched compute on fresh cases | Not established; the rotation confirmation is negative (P20, P21) |
| The identification-and-control procedure transfers to ARC | Untested |

This is not a reason to retreat to "memory matters". It locates the remaining step: **identify what changes the probability of productive completion, not merely what changes the trajectory.**

## 6. Recommendations for the paper

1. **Centre three additions:**
   - the architecture × dropout factorial with the early-versus-late crossover and the paired shared-failure count;
   - the distributed-correction intervention (P19's induced repair of unedited cells), with P15/P16/P18 as constraints on its interpretation;
   - the confirmation sequence for hidden-state perturbations, followed by the window-level completion model as an emerging explanation for why changed trajectories need not improve aggregate performance.
2. **Use the supporting results** to strengthen existing claims: P3–P5, P8/P13, and P9/P11 once its control accounting is corrected. The detailed campaign belongs in the appendix; the main argument should follow the questions these experiments resolved.
3. **Keep** the distinction between reaching, preserving and selecting answers, and keep ARC as a boundary test. The Sudoku evidence does not justify a universal mechanism across tasks.
4. **Do not claim** a recovered compact operator, control through hidden-state edits, memoryless dynamics or chaos.

## 7. The next decisive test

The target quantity is q_B(z) = P(completion within budget B | a fixed score-preserving perturbation distribution around state z). P22–P24 already estimate it from eight re-rolls per state.

The decisive experiment has three steps:
1. On discovery puzzles, find a compact state feature that predicts q_B.
2. Make a score-preserving intervention along that feature.
3. Show that it raises q_B on held-out puzzles, against random-direction, opposite-direction and restoration controls.

The originally proposed transfer of a correction *response* between states of the same puzzle, left incomplete by the revised P16, remains a direct test of transferring a computational rule rather than rescuing one continuation.

## 8. Open items

- The decisive test above: a registration draft for the PI's review.
- Corrections owed: P9/P11 control accounting (session d378a5).
- Reproduce the K ≈ BA fit before citing it.
- Parallel against sequential compute at benchmark scale. Under the window-level model this needs the transient accounted for. A pod would need the PI's go.
- ARC.
- Manuscript changes only on the PI's instruction.

*Sources:*
- the Outcome sections and addenda of the P15–P17, P18, P19, P20, P21, P22, P23 and P24 registration notes;
- session d378a5's P3–P13 notes and the SE-RRM attribution rounds 1–3 (`runs/analysis/attr3_20261003/attr3_verdict.txt`);
- the ledger lines of 2026-10-02 to 2026-10-08.
