# Note 2026-10-02 — P15, P16, P17: recovering the correction operator (registration, before any row)

**Goal (page one).** A MEASUREMENT. No accuracy target; $0; the Mac's CPU; inference only on the banked checkpoints; nothing trained. The PI's direction (2026-10-02): recover an operator that predicts corrections, show how hidden state changes it, and use it to improve inference. Three stages, each with its own rule; P15 is registered in full here, P16 and P17 have their criteria fixed here and their frozen objects added by dated amendment before their confirmation data are touched. IDs P15 onward follow the coordination with session d378a5 (P7–P14 theirs). Tool: `tools/rebuttal_p15.py` (imports the P1/P2 runner unchanged; outputs `runs/analysis/rebuttal_20261002/`).

## Why

The discussion-period results establish that later behaviour depends on slow-state content the readout cannot see (P2, P3, P8–P12) and that difficulty is a property of the error configuration (P1c, P7). They do not say what computation that state performs. The leading hypothesis: recurrence performs inference-time credit assignment, deciding which decisions must change together, and the persistent state carries that rule. Its testable core is the response to controlled errors: given a change to the displayed answer, does the network keep it, reject it, or propagate it to the decisions that depend on it; and does the hidden state control that response.

## Validation results obtained before this note (numerical only; no structural statistic computed)

Run 2026-10-02 13:15–13:40Z on Attention 256, batch-128 states of the study's intact first batch (prototype scripts kept in the session scratchpad).

1. **The training-time gradient stops remove every input-state derivative.** `segment` in `paper/code/src/qhrrn2/dec_cell.py` stops gradients after the first two slow cycles of every outer iteration; the release's JVP of next scores with respect to the incoming state is exactly zero. The measurement uses a copy of `segment` without the two stops.
2. **That copy reproduces the release forward bitwise** (next state from the iteration-1 state, max |difference| 0.0).
3. **Its derivative is correct.** In float64 the JVP agrees with central finite differences to a median relative error of 3.9e-8 at step 1e-2 (single-score pushes, one outer iteration).
4. **The displayed scores are saturated, and the network nearly discards small changes to them.** Empty-cell digit scores have median magnitude 52 and a median top-two margin of 84 at iteration 1. A unit push to one score (through the readout direction, kernel and fast state fixed) changes the next iteration's 729 scores by a total norm of 2.0e-3 (median; on-site 3.2e-4). Float32 central differences cannot resolve such responses at small steps, which is why they disagreed with the JVP in the first prototype. At linear order the score-space operator is close to zero: K = I − C ≈ I.

Consequence for the design: the operator that matters for correction acts on decision-scale changes, not infinitesimal ones. P15's primary measurements therefore use finite decision flips; the linear operator is measured descriptively.

## P15 — how the network treats a changed decision, and whether the hidden state controls it

**Receivers.** Attention 256 (primary), Attention 192 (replication), Attention 128 (reported, the bridge to P12). Release code through the P1/P2 runner; float32 CPU; the outer iteration as the release step, and for rollouts the gradient-stop-free copy (forward-identical).

**States.** The study's 256 intervention puzzles, intact fixed-start trajectory, the state after iteration t: t = 1 for every puzzle; t = 2 and t = 3 for puzzles not exact at t. States are computed by the release step at batch 128 and must reproduce the study's logits bitwise.

**Split.** A seeded permutation (`numpy.random.default_rng(20261002)`) of the 512 repair-puzzle IDs, sorted; the first 256 positions are DISCOVERY, the rest CONFIRMATION. Every state of a puzzle belongs to its puzzle's split. The rules below are evaluated on CONFIRMATION states; DISCOVERY states are reported beside them and are the only states used for exploratory analyses and for fitting P16's objects.

**The flip.** At state z = (h, l) and empty cell i displaying a, a flip to digit b swaps the two readout components: the score of b takes a's score and a takes b's, so b is displayed with the old margin. Only the readout-parallel components of two field vectors at cell i change; the readout-invisible slow state, the fast state, every other cell and the puzzle input are unchanged. Gate: after the swap, the displayed grid equals the old grid with i set to b.

**Flip classes** (ground truth used only to classify, never by the network):
- **HA**: i wrong, b = its solution digit, and b is displayed by at least one peer j (necessarily wrong): a helpful flip that creates a visible duplicate with a wrong peer.
- **HN**: i wrong, b = its solution digit, b displayed by no peer.
- **XB**: i correct, b a wrong digit displayed by at least one peer j.
- **XN**: i correct, b a wrong digit displayed by no peer.

Per state, up to three flips per class, chosen uniformly from the eligible (cell, digit) pairs with seed `[20261002, puzzle ID, t, class code]`.

**Conditions** (all within one batch per state, so the no-flip reference shares the batch): `intact_noflip`; `intact_flip` (each selected flip); `kernel_noflip` and `kernel_flip`: at the moment of the flip, every slow vector's readout-orthogonal part is replaced by a random vector orthogonal to the readout with the same norm (P2's edit, applied once, seeds `[20261002, puzzle ID, 51, t]`), scores kept exactly; `reencode_flip` (exploratory): the slow vectors at cell i are replaced by the answer embedding of the flipped grid at i, fast state kept. Each condition continues four outer iterations; the displayed grid is recorded after each.

**Quantities** (k = iterations after the flip; k = 1 primary, k = 2–4 secondary). For a flip at i, c(·) = 1 if i displays its solution digit at t + k.
- Helpful flips (HA ∪ HN): e⁺ = [P(c | flip) − P(c | no flip)] / [1 − P(c | no flip)], pooled as a ratio of means: the share of the available gain the flip realizes.
- Harmful flips (XB ∪ XN): e⁻ = [P(c | no flip) − P(c | flip)] / P(c | no flip): the share of naturally kept correct decisions the flip destroys.
- HA duplicates: for each peer j displaying b at t, e_j = [P(j no longer displays b | flip) − P(same | no flip)] / [1 − P(same | no flip)]: the induced change at the blamed peer. Blame accuracy: among HA flip-partner pairs where exactly one of i and j displays b at t + k, the share in which it is i.

## P15 rules (confirmatory, CONFIRMATION states, k = 1)

**R1 — the treatment of the displayed decision** (per receiver): DISPLAY-INERT if e⁺ ≤ 0.10 and e⁻ ≤ 0.10 (the next answer is recomputed without regard to the displayed decision); FOLLOWER if e⁺ ≥ 0.50 and e⁻ ≥ 0.50 (displayed decisions persist, helpful or not); SELECTIVE if e⁺ ≥ 0.50 and e⁻ ≤ 0.10 (helpful changes are kept and harmful ones rejected); MIXED otherwise.

**R2 — hidden-state control of that treatment** (per receiver): with δ⁺ = e⁺(kernel) − e⁺(intact) and δ⁻ = e⁻(kernel) − e⁻(intact), each `kernel` quantity computed against `kernel_noflip`: KERNEL-CONTROLS if max(|δ⁺|, |δ⁻|) ≥ 0.20; KERNEL-NEUTRAL if both ≤ 0.05; MIXED otherwise.

**R3 — coordinated credit assignment** (HA flips, per receiver): COORDINATES if e_j ≥ 0.30 and blame accuracy ≥ 0.75; LOCAL if e_j ≤ 0.05; MIXED otherwise. UNDEFINED if fewer than 30 HA flip-partner pairs.

Letters are given for Attention 256 and 192; the same letter on both is the stated result, and different letters are reported as width-dependent. Attention 128 is reported without a letter.

**Predictions and credences (before any row).** R1: SELECTIVE 0.30; DISPLAY-INERT 0.25; FOLLOWER 0.20; MIXED 0.25. R2: KERNEL-CONTROLS 0.50; KERNEL-NEUTRAL 0.20; MIXED 0.30. R3: COORDINATES 0.30; LOCAL 0.35; MIXED 0.35. Same R1 letter on 256 and 192: 0.6.

**Wording under each outcome.**
- SELECTIVE: "A change to one displayed decision, with the rest of the state untouched, is kept when it is correct and rejected when it is not: the recurrence evaluates its own displayed answer against the rest of the problem."
- FOLLOWER: "Displayed decisions persist whether correct or not: the network builds on its current answer rather than checking it."
- DISPLAY-INERT: "Changing a displayed decision has no effect on the next answer: the readout-parallel state is an output, and the trajectory is carried by the state the readout cannot see." Together with the validation's near-zero linear operator, this would make the readout a projection without causal role.
- KERNEL-CONTROLS: "Replacing the readout-invisible state, with every score kept, changes how the network treats the same decision change: the hidden state controls the correction rule."
- COORDINATES: "Making one wrong decision correct causes the network to change the wrong decision that now conflicts with it, and it blames the right one: corrections are assigned across dependent decisions."
- LOCAL: "The conflicting peer is not changed more than it would have been anyway: correction is not propagated along the dependency the flip creates."
Under every outcome: flips change only readout-parallel components; no feature content is identified; results hold for the tested states and checkpoints.

**Exploratory (no letters).** k = 2–4; t-dependence; HA versus HN and XB versus XN; the `reencode_flip` channel; induced repairs at cells other than i and j (cascades); the linear operator on 16 DISCOVERY states per receiver (full one-iteration Jacobian on empty-cell scores: on-site, peer and non-peer response mass; correction gain along the singular directions of the occupancy-residual Jacobian A = ∂r/∂s, r the unit-digit occupancy of digit-conditional StableMax probabilities with givens one-hot, and along its null space; softmax as a sensitivity check).

## P16 — transfer of the response rule (criteria fixed now; frozen objects by amendment)

**Question.** The decisive test of the operator hypothesis: does transferring an identified component of the hidden state transfer the response to new errors that were not used to identify it?

**Arm A (attention).** On DISCOVERY states, the readout-invisible state is replaced in one region only: at cell i, at its 20 peers, or everywhere else. The region whose own content best restores the intact e⁺, e⁻ and e_j (when the rest is replaced) is frozen, with its hash, in an amendment. On CONFIRMATION states, with the kernel replaced everywhere except the frozen region (which keeps its intact content), new flips are applied. **TRANSFERS** if each of e⁺, e⁻ and e_j recovers at least 75 % of the gap between the `kernel` and `intact` conditions; **NO-TRANSFER** if at most 25 %; MIXED otherwise; UNDEFINED if the gap is under 0.10.

**Arm B (MLP 192 at 94k, P9's identical-answer pairs).** Both trajectories display the solution at t; harmful flips (XB, XN) at seeded cells measure e⁻ in the fixed-start state F, the Gaussian state G, and F with G's readout-orthogonal slow part (the component P9 identified on natural continuation, not on flips). **RULE-TRANSFERS** if e⁻(F←G) moves at least 75 % of the way from e⁻(F) to e⁻(G); **NO-TRANSFER** if at most 25 %; UNDEFINED if |e⁻(G) − e⁻(F)| < 0.10. Population: P9's 60 primary cases at 94k, with P11's 90k and 114k cases as replication.

## P17 — using the operator (criteria fixed now; the method by amendment)

On trajectories unsolved at iteration 16 (the full-test record's unsolved pool, Attention 128, P12's population definition, new seeded draw), an operator-guided intervention chosen from P15/P16's results, with a trigger that needs no reference answer, is compared at matched outer iterations with intact continuation, P12's random replacement, a fresh Gaussian restart, and a constraint-only baseline (the same number of iterations spent on decisions chosen by visible conflict alone). **GUIDED-HELPS** if it solves more than every comparator by 32 with McNemar p < 0.05 against the strongest; **NO-GAIN** otherwise. The method, its cost accounting and its seed are fixed by amendment before any P17 row.

## Gates (P15)

(1) States: the release step at batch 128 reproduces the study's intact logits bitwise at t = 1, 2, 3. (2) The gradient-stop-free outer iteration reproduces the release step bitwise from the t = 1 state of batch 0. (3) The `intact_noflip` rollout's displayed grids equal the study's at t + 1 … t + 4 for every state (batch composition may change logits in the last bits; displayed grids must match). (4) Every flip produces exactly the designed displayed grid at t. (5) The kernel replacement changes no digit score by more than 1e-2 in float64 (the realized float32 shift recorded). A failed gate stops the run.

## Labels and plan

Confirmatory: R1–R3 (P15), the two P16 letters, the P17 letter. Exploratory: as listed. Build `tools/rebuttal_p15.py`; selftest and a 16-puzzle smoke with gates (1)–(5), printing gate results only; commit this note and the tool; run 256, 192 and 128 (one process each, after checking the other session's load); report by the tool; the Outcome appended here and a ledger line written when read.

---

**Amendment 1 (2026-10-02 ~14:20Z, before any outcome was computed).** At launch (13:29Z) gate 2 stopped Attention 192 and 128: the gradient-stop-free outer iteration was not bitwise equal to the release step. Diagnosis on batch 0 of the intervention population: the release's own `segment`, compiled the same standalone way with its gradient stops kept, differs from the release step by the same amount (largest slow-state difference 5.0e-3 at 192 and 2.1e-3 at 128; largest score difference 0.05 and 0.01 against margins near 84; displayed grids identical), and the stop-free copy equals that stop-kept segment bitwise on both widths. The difference comes from compiling the segment apart from the release step's input path, not from removing the stops; on Attention 256 the two compilations coincide, which is why it passed. Changes: (i) every P15 rollout now runs through the release step itself (`eval_sudoku_extreme._step`, the study's loop and carry update), since flips need no derivative; (ii) gate 2 is restated as "the stop-free copy equals the release segment with its stops, compiled the same way, bitwise", which isolates the effect of removing the stops and protects the derivative measurements; (iii) gate 3 is unchanged and remains the check that the no-flip rollouts reproduce the study's displayed grids. Attention 256 had passed the original gate 2 and had written no chunk; it was stopped so that all three widths run the same code. Re-smokes on all three widths pass gates 1–5. No rule, threshold, population, flip class or seed changed.

---

**Amendment 2 (2026-10-02 ~15:00Z, before any P16 row): P16 revised — belief edits in the network's own coordinates.**

*Why the change.* The first P15 chunks (Attention 128 and 192, batch 0, t = 1, both splits; interim, not P15's verdict, whose rules are untouched) showed display flips nearly inert (effect shares about 0.05), answer-format re-encodes of the cell kept when correct and rejected when wrong (0.25–0.42 against 0.06–0.08), and harmful display flips surviving about three times as often once the hidden state was replaced. The registered P16 Arm A transfers the response to display flips, which carry almost no response; it is withdrawn as uninformative. Arm B (MLP 192 at 94k) is deferred to a later amendment with its criteria unchanged. The PI's framing of 2026-10-02 (plan §7) sets four predictions; P16 now tests the first two and the localisation needed for the third.

*The edits.* Digit fields share every parameter (checked: no parameter is field-specific), so exchanging two digits' vectors at one cell relabels that cell's belief between those digits in the network's own coordinates, with no training format involved. Gate 6 checks the symmetry: exchanging fields 1 and 2 everywhere, slow and fast, and digits 1 and 2 in the givens, permutes the next scores to within 1e-3 of their scale (smoke: 4.7e-4 on a scale of 43). Channels, for the same flips P15 selects (same seeds), at cell i from displayed a to b: `display_swap` (readout-parallel parts of the two fields only, P15's flip), `kernel_swap` (readout-orthogonal slow parts only; every score unchanged), `slow_swap` (whole slow vectors), `cell_relabel` (slow and fast vectors). For each HA flip, one duplicate partner j (seed `[20261003, puzzle ID, t, i, 7]`) and one matched unrelated empty cell k (not a peer of i or j): a dial on retained commitment at j, `dial(β)`, moves the nine fields' readout-orthogonal parts at j toward their mean by the fraction β (β = 1 leaves no hidden digit preference; β < 0 strengthens it), scores unchanged, for β ∈ {−0.5, 0.5, 1.0}, each with and without the `cell_relabel` at i; the same dial with β = 1.0 at k is the control. Every condition continues four outer iterations through the release step from the state after iteration t; states, split, flip selection and receivers as in P15. Gates: P15's gates 1 and 3 (no-edit rollouts reproduce the study's displayed grids), display swaps produce the designed display, kernel swaps and dials change no score by more than 1e-3, slow swaps produce the designed display, and gate 6.

*Rules* (CONFIRMATION states, k = 1; letters on Attention 256 and 192; Attention 128 reported).
- **A — where a decision is held.** Each channel gets P15's R1 letter from its own e⁺ and e⁻ against the no-edit continuation. KERNEL-HOLDS if `display_swap` is DISPLAY-INERT and `kernel_swap` is FOLLOWER or SELECTIVE; SLOW-HOLDS if neither part alone reaches e⁺ ≥ 0.50 but `slow_swap` does (FOLLOWER or SELECTIVE); FAST-NEEDED if only `cell_relabel` does; NOT-LOCAL if no channel does (the cell's decision is re-derived from the rest of the state); MIXED otherwise.
- **B — retained state controls propagation** (HA flips). With resp(β) = P(j no longer displays b | relabel at i, dial(β) at j) − P(same | dial(β) at j alone), resp(0) the same without a dial, and resp_k the same with the dial at k: CONTROLS-PROPAGATION if resp(1.0) − resp(0) ≥ 0.15, resp(0) − resp(−0.5) ≥ 0.05, and |resp_k − resp(0)| ≤ (resp(1.0) − resp(0)) / 3; NO-CONTROL if |resp(1.0) − resp(0)| ≤ 0.05; MIXED otherwise.
- **C — compact structure.** For every `cell_relabel` and every other empty cell j, y = 1 if j's display at t + 1 differs from the no-edit continuation. Logistic models fitted on DISCOVERY pairs, scored by AUC on CONFIRMATION pairs: static (peer, same row, same column, same box), display (static plus j displays b, j displays a, j's and i's display margins), display + hidden (plus the retained-commitment index of j and of i: the distance of the displayed digit's readout-orthogonal slow part from the mean of the other eight, relative to their mean norm). COMPACT-HIDDEN if display + hidden reaches AUC ≥ 0.75 and exceeds display by ≥ 0.05; DISPLAY-SUFFICES if display reaches ≥ 0.75 and the hidden features add < 0.02; WEAK if display + hidden < 0.65; MIXED otherwise.

*Predictions and credences (before any P16 row).* A: KERNEL-HOLDS 0.30; NOT-LOCAL 0.20; SLOW-HOLDS 0.15; FAST-NEEDED 0.10; MIXED 0.25. B: CONTROLS-PROPAGATION 0.30; NO-CONTROL 0.35; MIXED 0.35. C: COMPACT-HIDDEN 0.30; DISPLAY-SUFFICES 0.25; WEAK 0.20; MIXED 0.25.

*Wording under each outcome.* KERNEL-HOLDS: "A cell's decision is held in the slow state the readout cannot see: exchanging only that part between two digits changes what the cell displays next, while exchanging only the displayed part does not." SLOW-HOLDS / FAST-NEEDED: the same with "the whole slow state" or "the slow and fast states together". NOT-LOCAL: "Rewriting a cell's entire state does not change its next decision: decisions are re-derived from the surrounding state, not held by the cell." CONTROLS-PROPAGATION: "How strongly a dependent decision holds its hidden commitment controls whether a change elsewhere propagates into it: weakening the commitment lets the conflicting decision change, strengthening it blocks the change, and the same edit at an unrelated cell does not." NO-CONTROL: "Retained commitment at the dependent decision does not control the propagation." COMPACT-HIDDEN: "A few relation types plus two retained-commitment numbers predict which decisions respond to a change, clearly better than the display alone." DISPLAY-SUFFICES: "The display and the problem's structure predict the responses as well; the hidden commitment index adds nothing." Under every outcome: the dial moves random-free but low-dimensional parts of the state; no feature content beyond the commitment index is identified.

*Plan.* `tools/rebuttal_p16.py` (selftest and reader selftest on synthetic rows; a gates-only smoke on Attention 128 passed gates 1–6). Launch when P15's processes finish; outputs `runs/analysis/rebuttal_20261002b/`.

---

## P15 Outcome (2026-10-02 15:43Z; `runs/analysis/rebuttal_20261002/report.{txt,json}`; every gate passed; no shared tool edited)

**Integrity.** On all three widths: the release-step states reproduced the study's logits bitwise at t = 1–3; the gradient-stop-free outer iteration equalled the release segment compiled the same way, bitwise (Amendment 1's gate 2); every intact no-flip rollout reproduced the study's displayed grids at t + 1 … t + 4; every flip produced its designed display; the hidden-state replacement changed no score by more than 1.4e-14 in float64 (realized float32 ≤ 1.9e-5). An independent recount from the raw chunk arrays agrees with the report on e⁺, e⁻, their kernel-condition values, e_j and blame on every width. States: 1,173 (CONFIRMATION 591, DISCOVERY 582) over the three widths; flips per width on CONFIRMATION: 1,706–2,509.

**Results, CONFIRMATION states, k = 1.**

| receiver | e⁺ helpful | e⁻ harmful | P(i correct): helpful flip / none; harmful flip / none | kernel replaced: e⁺ / e⁻ (δ⁺ / δ⁻) | e_j | blame (intact / kernel) | letters R1 / R2 / R3 |
|---|---|---|---|---|---|---|---|
| Attention 256 | 0.065 | 0.046 | 0.611 / 0.585; 0.891 / 0.934 | 0.129 / 0.101 (+0.065 / +0.055) | 0.070 | 0.68 / 0.57 | DISPLAY-INERT / MIXED / MIXED |
| Attention 192 | 0.022 | 0.080 | 0.537 / 0.526; 0.847 / 0.920 | 0.049 / 0.135 (+0.026 / +0.056) | −0.024 | 0.58 / 0.38 | DISPLAY-INERT / MIXED / LOCAL |
| Attention 128 (no letter) | 0.070 | 0.100 | 0.487 / 0.448; 0.806 / 0.895 | 0.052 / 0.182 (−0.018 / +0.082) | 0.075 | 0.54 / 0.30 | (DISPLAY-INERT / MIXED / MIXED) |

DISCOVERY states give the same picture (e⁺ 0.05–0.10, e⁻ 0.05–0.11). At k = 2–4 every display effect stays within ±0.11 except where the no-flip correctness approaches 1 and the ratio becomes unstable.

**Letters.** R1 **DISPLAY-INERT** on 256 and 192 (the stated result; 128 the same). R2 **MIXED** on both. R3 **MIXED** on 256 and **LOCAL** on 192: width-dependent, and in neither case coordinated.

**Predictions scored.** R1 DISPLAY-INERT (0.25) HIT. Same R1 letter on 256 and 192 (0.6) HIT. R2 MIXED (0.30) HIT. R3: MIXED (0.35) HIT on 256, LOCAL (0.35) HIT on 192; COORDINATES (0.30) MISS.

**The registered sentence that applies, with its measured qualification.** "Changing a displayed decision has no effect on the next answer: the readout-parallel state is an output, and the trajectory is carried by the state the readout cannot see." Measured: helpful display changes realize 2–7 % of the available gain and harmful ones destroy 5–10 % of the naturally kept decisions; within one outer iteration the network restores the decision its hidden state holds. With the validation's near-zero linear score operator, the displayed answer has almost no causal role in the next step.

**Exploratory.** (1) Replacing the hidden state roughly doubles how often a harmful display change survives (e⁻ 0.05→0.10, 0.08→0.14, 0.10→0.18): the retained state helps reject displayed errors, by less than the 0.20 bar. (2) With the hidden state replaced, the resolution of the duplicates a helpful flip creates turns against the correct cell (blame 0.68→0.57, 0.58→0.38, 0.54→0.30): which of two conflicting decisions yields depends on the retained state. (3) Writing the change into the cell's slow vectors in the answer format is kept when correct and rejected when wrong (e⁺ 0.21 / 0.18 / 0.12 against e⁻ 0.05 / 0.05 / 0.07): selective, below the 0.50 bar, and the motivation for P16's belief edits. (4) Credit assignment through display flips is absent at k = 1 (e_j −0.02 to 0.08); blame among resolved duplicates rises to 0.70–0.80 by k = 2–4, mostly through the natural repair of the wrong peer.

**Reading.** The operator the hypothesis needs does not act on the displayed answer. A change to the display is erased, and the network re-derives each decision from state the readout cannot see; that state also biases which of two conflicting decisions gives way. P16 measures the response where the decision is held.

---

## P16 Outcome (Amendment 2's rules; 2026-10-02 19:39Z; `runs/analysis/rebuttal_20261002b/report.{txt,json}`; every gate passed; no shared tool edited)

**Integrity.** On all three widths: states reproduced the study's logits bitwise; the digit-relabel symmetry held (gate 6, deviation ≤ 1e-5 of the score scale); every display and slow swap produced its designed display; kernel swaps and commitment dials changed no score by more than 1e-3; every no-edit rollout reproduced the study's displayed grids. An independent recount from the raw arrays agrees with the report on every channel's e⁺ and e⁻ and on every propagation response.

**A — where a decision is held (CONFIRMATION, k = 1; e⁺ for correct edits / e⁻ for wrong ones).**

| receiver | display swap | kernel swap | slow swap | cell relabel (slow + fast) | letter |
|---|---|---|---|---|---|
| Attention 256 | 0.065 / 0.046 | 0.026 / 0.014 | 0.114 / 0.073 | 0.132 / 0.075 | NOT-LOCAL |
| Attention 192 | 0.022 / 0.080 | −0.039 / 0.009 | 0.081 / 0.122 | 0.090 / 0.129 | NOT-LOCAL |
| Attention 128 (no letter) | 0.070 / 0.100 | 0.123 / 0.037 | 0.300 / 0.169 | 0.283 / 0.196 | (NOT-LOCAL) |

**B — propagation into the conflicting peer** (resp at the dial value against the dial alone; n = 471 / 441 / 621 helpful flips with a duplicate partner): resp(0) 0.017 / −0.020 / −0.005; resp(1.0) −0.008 / 0.016 / −0.011; resp(−0.5) −0.006 / 0.014 / 0.019; the unrelated-cell control −0.008 / 0.036 / 0.027. **NO-CONTROL** on every width.

**C — compact structure** (logistic models fitted on DISCOVERY, AUC on CONFIRMATION; which cells change their display after a cell relabel): static 0.50 on every width; display features 0.689 / 0.735 / 0.692; display plus the retained-commitment index 0.691 / 0.734 / 0.694. **MIXED** on every width: the display and the problem's structure carry what predictability there is, and the hidden index adds nothing.

**Predictions scored.** A: NOT-LOCAL (0.20) HIT on both lettered widths. B: NO-CONTROL (0.35) HIT. C: MIXED (0.25) HIT; the display-only model fell just short of DISPLAY-SUFFICES's 0.75 bar while the hidden gain was zero.

**The registered sentences that apply.** NOT-LOCAL: "Rewriting a cell's entire state does not change its next decision: decisions are re-derived from the surrounding state, not held by the cell." Measured: rewriting the cell's slow and fast state in the network's own coordinates realizes 9–28 % of the available gain when correct and destroys 8–20 % of kept decisions when wrong, at most modestly selective. NO-CONTROL: "Retained commitment at the dependent decision does not control the propagation."

**Reading.** Together with P15, the per-cell, local version of the operator hypothesis fails on these networks: neither a cell's displayed answer, nor its readout-invisible slow state, nor its whole state holds its decision, and the retained commitment of a conflicting decision does not gate whether a change propagates into it. A decision is re-derived each iteration from the configuration of the whole state. The collective version is tested by P18 (coordinated corrections) and P19 (the share of the error that must be corrected at once).
