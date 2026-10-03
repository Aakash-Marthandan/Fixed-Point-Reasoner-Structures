# Note 2026-10-03 — What the discussion-period operator program found (P15–P22): collective acceptance, then a memoryless chaotic search

**Purpose.** A synthesis for the PI's decision on the rebuttal narrative. Every claim cites a registered outcome. Exploratory observations are marked. Nothing here edits the manuscript.

## 1. What was asked

Plan §7 (the PI, 2026-10-02) set two commitments:
1. Identify a general principle: retained state adaptively controls how changes propagate between dependent decisions, through a compact set of interactions.
2. Use it to control reasoning predictably.

Four predictions were registered as success criteria:
- retained state changes the correction rule;
- compact structure;
- selective edits reproduce broad interventions and matched random edits do not;
- transfer.

## 2. What the data refute

**Local control of decisions.**
- **P15, display-inert.** Changing a displayed decision is undone within one iteration.
- **P16.** Three letters:
  - NOT-LOCAL: rewriting a cell's entire state keeps at most 9–28 % of the available gain.
  - NO-CONTROL: a conflicting peer's retained commitment does not gate propagation.
  - MIXED compact structure: display features predict responses with AUC of about 0.7, and the hidden commitment index adds nothing.
- **P18, independent.** Correcting a conflict-free set of errors is kept no better than correcting a random set of the same size.
- **P17.** Perturbing the visibly conflicting cells equals perturbing random cells.

**Control by perturbing the hidden state.** One answer-preserving perturbation does not improve stalled trajectories:
- **P17:** an exploratory 60° rotation looked strong.
- **P20:** on fresh attention puzzles, NO-DIFFERENCE on every width, at any angle from 45° to 75°.
- **P21:** NO-DIFFERENCE on all three MLP seeds, at every horizon to iteration 64.

**The four predictions.**
1. *Retained state changes the correction rule:* only partly, and only in exploratory data. In P15, replacing the hidden state shifts which of two conflicting decisions yields.
2. *Compact structure:* not found.
3. *Selective edits:* refuted (P17, P18).
4. *Transfer:* what transfers is the null, and the behaviour behind it.

## 3. What the data support: two regimes

**(a) Decisions are re-derived each iteration from the whole configuration.** No cell holds its own decision (P15, P16).

**(b) The cooperative regime.** Corrections are accepted collectively:
- whole-grid completion rises smoothly with the share of errors corrected at once, centred at half (P19 GRADUAL; f50 0.53 / 0.50);
- what is corrected matters far less than how much (P18).

States about to complete are frozen to small pushes, while states far from completion are sensitive (P15 Amendment 3).

**(c) The stalled regime is a memoryless chaotic search** (P22, confirmed on all three attention widths):
- *Chaotic.* A rotation of the readout-invisible state by 1°, and even 0.1°, which changes no displayed score, decorrelates the outcome sixteen iterations later as fully as 60°.
- *Memoryless.* Across independent re-rolls of the same stalled state, trajectories that survive sixteen iterations complete in the next sixteen at the rate the instance's first window predicts (R = 1.08 / 0.91 / 1.08, intervals inside [0.67, 1.5]).
- *Intact continuation is one more draw.* The law predicts its completions (exploratory).
- *Instance rates span the whole range.* The falling completion rate across puzzles is difficulty heterogeneity, not trajectories getting stuck.
- *Stalled trajectories never freeze.* They keep revising an invalid configuration (P17).

**(d) Consequences, observed before the law was registered and confirmed after it.**
- Perturbations and restarts re-roll a trajectory without changing its rate, so they cannot help on average (P20, P21).
- Parallel re-rolls with a validity check that needs no answer key equal sequential iterations at matched compute: exploratory in P20 and P21, and again in P22.

## 4. The principle, stated for the rebuttal

> In these learned recursive reasoners, test-time iteration runs in two regimes.
>
> In the **cooperative regime** the configuration moves toward the solution as a whole, and a correction sticks once enough of the configuration agrees with it, about half the errors.
>
> If the configuration does not complete, it enters a **stalled regime**: a chaotic search so sensitive to the state the readout cannot see that each stretch of iterations is effectively a fresh attempt, completing at a rate set by the instance.
>
> In the tail, test-time compute buys independent attempts, not accumulated progress.

**Why it is general rather than Sudoku-specific.**
- It is a statement about iterative refinement read out from a high-dimensional state, not about puzzle rules.
- It holds across three independent initializations.
- The re-roll behaviour holds on a second token mixer (P21).
- It falls within classic restart theory for randomized search (Luby, Sinclair and Zuckerman 1993, "Optimal speedup of Las Vegas algorithms"; Gomes, Selman, Crato and Kautz 2000, heavy-tailed run-time distributions in combinatorial search). Restarting cannot shorten waiting times when an instance's run-time distribution is memoryless (a constant hazard). It helps when the hazard falls (heavy tails) and hurts when it rises. These networks' hazard per instance is constant, which predicts, and the data show, that restarts and kicks do not help.

**What it predicts that can be checked.** The effect of any restart, perturbation or parallelization policy, from each instance's rate. Example: k parallel 16-iteration attempts should match 16k sequential iterations.

## 5. Recommendations for the rebuttal

1. Do not claim local operators, compact interaction structure, or control by hidden-state edits. The registered tests reject them, and saying so is a strength.
2. Claim the two-regime account, with its registered evidence:
   - collective, graded acceptance (P19, P18);
   - the memoryless chaotic tail (P22) and its consequences (P20, P21);
   - generality across widths and, for the re-roll behaviour, the MLP mixer.
3. Use it to answer the manuscript's question of what test-time compute buys. In the cooperative regime, iterations buy convergence. In the tail they buy attempts at an instance-specific rate, so the waiting time per stalled instance is geometric and the long tail measures difficulty.

## 6. Open items

- **P22 on the MLP seeds.** Transfer of the law itself. Mac, about 2 h; P21's loader is ready.
- **The parallel–sequential equivalence as a confirmatory, benchmark-scale test.** A uniform test sample with a validity check, at matched compute. A pod would need the PI's go and spend envelope.
- **What sets an instance's rate.** Exploratory correlates such as puzzle rating and violations at iteration 16.
- **ARC, next week.** Do both regimes and the memoryless tail hold there?
- Manuscript changes only on the PI's instruction.

*Sources:* the Outcome sections of `Note_2026-10-02_Rebuttal_P15_P16_P17_Registration.md`, `Note_2026-10-02_Rebuttal_P18_Registration.md`, `Note_2026-10-03_Rebuttal_P19_Registration.md`, `Note_2026-10-03_Rebuttal_P20_Registration.md`, `Note_2026-10-03_Rebuttal_P21_Registration.md` and `Note_2026-10-03_Rebuttal_P22_Registration.md`, and the ledger lines of the same dates.
