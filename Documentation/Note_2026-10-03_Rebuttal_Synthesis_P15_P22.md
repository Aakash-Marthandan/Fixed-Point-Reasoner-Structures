# Note 2026-10-03 — What the discussion-period operator program found (P15–P24): collective acceptance, then a memoryless search

**Purpose.** A synthesis for the PI's decision on the rebuttal narrative. Every claim cites a registered outcome; exploratory observations are marked. Nothing here edits the manuscript.

*Updated 2026-10-03 ~15:00Z with P23 (the memory and chaos rules on the MLP mixer) and P24 (the same on the single-state recurrence and its two-state control). The file name is kept so that earlier references still resolve.*

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
4. *Transfer:* the program's positive result transfers instead: the memoryless law (§3c).

## 3. What the data support: two regimes

**(a) Decisions are re-derived each iteration from the whole configuration.** No cell holds its own decision (P15, P16).

**(b) The cooperative regime.** Corrections are accepted collectively:
- whole-grid completion rises smoothly with the share of errors corrected at once, centred at half (P19 GRADUAL; f50 0.53 / 0.50);
- what is corrected matters far less than how much (P18).

One step from completion, states are frozen to small pushes but need their hidden content:
- the linear sensitivity of about-to-complete states is near zero (P15 Amendment 3);
- in session d378a5's P13 (registered fecbec9; its Outcome reported by message, commit pending), replacing the readout-invisible state at the completing iteration blocks completion (B = 1.00 / 0.82 / 0.70 on SA256U / SA256 28k / SA256 14k), while a 1° rotation blocks 0–6 %. The content matters, not mere sensitivity, on both recurrence structures.

**(c) The stalled regime is a memoryless search, strongly sensitive to the readout-invisible state.** Measured on nine receivers.

*Memoryless.* For a fixed stalled state, re-rolls that survive sixteen iterations complete in the next sixteen at the rate the instance's first window predicts. MEMORYLESS on eight of nine receivers:

| experiment | receivers | R | letter |
|---|---|---|---|
| P22 | three attention widths | 1.08 / 0.91 / 1.08 | confirmed |
| P23 | three MLP 192 seeds | 1.16 / 0.86 / 0.95 | confirmed, transfers |
| P24 | SA256U (single state) | 1.00 | MEMORYLESS |
| P24 | SA256 28k | 1.07 | MEMORYLESS |
| P24 | SA256 14k | 0.86 | UNRESOLVED (interval lower end 0.661 against 0.67) |

SA256U's letter is also at the margin: a separately seeded bootstrap gives a lower end of 0.659.

*Sensitive, chaotic on attention.* A 1° (even 0.1°) rotation that changes no displayed score re-rolls the outcome sixteen iterations later:
- as fully as 60° on the attention models and the SA pair: D(1°)/D(60°) 0.80–0.97, CHAOTIC on all six;
- substantially but not fully on the MLP mixer: 0.74–0.79, INTERMEDIATE on all three (P23).

*Further properties:*
- Intact continuation is one more draw: the law predicts its completions (exploratory; P22–P24).
- Instance rates span the whole range: the falling completion rate across puzzles is difficulty heterogeneity, not trajectories getting stuck.
- Stalled trajectories never freeze. They keep revising an invalid configuration (P17), and in P24 0 % of stalled steps are frozen, with about 12 cells changing per iteration, on both recurrence structures.

**(d) What the two-state design buys in the tail** (P24, exploratory, with SE-RRM attribution round 2). The single-state recurrence's stall is the same kind of search: live, memoryless, sensitive (pattern SAME-TAIL). It completes about a third as often:
- a stalled re-roll completes within 16 iterations 0.116 of the time, against 0.328 and 0.297 for the two-state SA256;
- 44 of 80 single-state puzzles had no re-roll solved by 32.

Round 2 found the two-state design carries the accuracy advantage. P24 locates part of it: in the stalled regime, the slow/fast split raises the per-instance completion rate; it does not change what kind of search runs.

**(e) Consequences, observed before the law was registered and confirmed after it.**
- Perturbations and restarts re-roll a trajectory without changing its rate, so they cannot help on average (P20, P21).
- Parallel re-rolls with a validity check that needs no answer key equal sequential iterations at matched compute (exploratory in P20–P24).

## 4. The principle, stated for the rebuttal

> In these learned recursive reasoners, test-time iteration runs in two regimes.
>
> In the **cooperative regime** the configuration moves toward the solution as a whole, and a correction sticks once enough of the configuration agrees with it, about half the errors. One step from completion the state is frozen to small pushes and carries content the completion needs.
>
> If the configuration does not complete, it enters a **stalled regime**: a search so sensitive to the state the readout cannot see that each stretch of iterations is effectively a fresh attempt, completing at a rate set by the instance and by the architecture.
>
> In the tail, test-time compute buys independent attempts, not accumulated progress. The two-state design buys a higher rate of success per attempt.

**Why it is general rather than Sudoku-specific.**
- It is a statement about iterative refinement read out from a high-dimensional state, not about puzzle rules.
- The memoryless letter holds on eight of nine receivers, spanning three independent initializations per family, two token mixers and both recurrence structures.
- How chaotic the search is at small angles varies by mixer.
- It falls within classic restart theory for randomized search (Luby, Sinclair and Zuckerman 1993, "Optimal speedup of Las Vegas algorithms"; Gomes, Selman, Crato and Kautz 2000, heavy-tailed run-time distributions in combinatorial search). Restarting cannot shorten waiting times when an instance's run-time distribution is memoryless (a constant hazard). It helps when the hazard falls (heavy tails) and hurts when it rises. These networks' hazard per instance is constant, which predicts, and the data show, that restarts and kicks do not help.

**What it predicts that can be checked.** The effect of any restart, perturbation or parallelization policy, from each instance's rate. Example: k parallel 16-iteration attempts should match 16k sequential iterations.

## 5. Recommendations for the rebuttal

1. Do not claim local operators, compact interaction structure, or control by hidden-state edits. The registered tests reject them, and saying so is a strength.
2. Claim the two-regime account, with its registered evidence:
   - collective, graded acceptance (P19, P18);
   - frozen near completion with needed content (Amendment 3; P13, once committed);
   - the memoryless tail on eight of nine receivers (P22, P23, P24) and its consequences (P20, P21);
   - chaos strongest on attention.
3. Use it to answer the manuscript's question of what test-time compute buys. In the cooperative regime, iterations buy convergence. In the tail they buy attempts at an instance-specific rate, so the waiting time per stalled instance is geometric and the long tail measures difficulty.
4. Tie it to the SE-RRM attribution. The two-state advantage is, in the tail, a higher success rate per attempt rather than a different search (P24 with round 2). This gives a mechanistic reading of an accuracy comparison.

## 6. Open items

- **The parallel–sequential equivalence as a confirmatory, benchmark-scale test.** A uniform test sample with a validity check, at matched compute, on a pod. The PI decides after round 3's analysis.
- **What sets an instance's rate.** Exploratory correlates such as puzzle rating and violations at iteration 16. Also, which part of the slow/fast split raises the rate (P24's open question).
- **ARC, next week.** Do both regimes and the memoryless tail hold there?
- Manuscript changes only on the PI's instruction.

*Sources:*
- the Outcome sections of `Note_2026-10-02_Rebuttal_P15_P16_P17_Registration.md`, `Note_2026-10-02_Rebuttal_P18_Registration.md`, `Note_2026-10-03_Rebuttal_P19_Registration.md`, `Note_2026-10-03_Rebuttal_P20_Registration.md`, `Note_2026-10-03_Rebuttal_P21_Registration.md`, `Note_2026-10-03_Rebuttal_P22_Registration.md`, `Note_2026-10-03_Rebuttal_P23_Registration.md` and `Note_2026-10-03_Rebuttal_P24_Registration.md`;
- session d378a5's `Note_2026-10-03_Rebuttal_P13_Registration.md` (Outcome pending) and SE-RRM attribution round 2 (`Note_2026-10-02_Attribution_Round2_Registration.md`);
- the ledger lines of the same dates.
