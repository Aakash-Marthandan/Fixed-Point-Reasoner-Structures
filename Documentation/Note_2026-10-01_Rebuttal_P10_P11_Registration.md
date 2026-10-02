# Note 2026-10-01 — P10 (what the iterations before completion are doing) and P11 (identical-answer swaps at other checkpoints) — registration, before any row

**Goal (page one).** A MEASUREMENT for both. No accuracy target; $0; the Mac's CPU; inference only on banked checkpoints; nothing trained.
- Plan: `Plan_2026-10-01_Rebuttal_Experiments.md` §6. IDs P10–P14 are reserved by this session; the other session uses P15 onward.
- Tools: `tools/rebuttal_p10.py` imports the inside-completion lens and P8's helpers unchanged. `tools/rebuttal_p11.py` runs P9's tool unchanged at other checkpoints, by setting its module constants.
- Outputs: `runs/analysis/rebuttal_20261001h/` (P10) and `…i/` (P11).
- Both run at `nice -n 10` beside the other session's P3 queue and never touch its files.
- Written before either tool is built and before any row exists.

---

## P10 — Do the iterations before completion build hidden progress, sit in a sensitive trap, pay a recovery cost, or idle?

**Why.** §3 of the manuscript reports substantial offsetting revisions with little net progress before an abrupt completion. Two results bear on this:
- P8: replacing the slow state's readout-invisible part once, one outer iteration before the completing cycle, blocks 84–89 % of completions there and mostly delays them, by a median of 4 readouts.
- Lai et al. (2026) attribute such slowdowns to trapping near saddle points, a sensitive transient.

The open question is what the earlier iterations of the plateau contribute. Graded, one-time, score-preserving edits at several distances before completion separate four accounts:

| account | what the edits should do |
|---|---|
| ACCUMULATION | Hidden progress builds up; an edit erases part of it. Delays at every distance, larger the closer to completion, growing with the angle, with no earlier completions. |
| SENSITIVE-TRANSIENT | Small edits shift completion both earlier and later. |
| RECOVERY-COST | Any edit costs about the same fixed delay wherever it lands. |
| IDLE | Edits early in the plateau do not change completion time. |

**Models and population.**
- Models: C5 (MLP 192, 46k) and SA128 (Attention 128, 42k), whose readout is one shared vector.
- Cohorts: the inside-completion cohorts — first exact at slow readout κ ≥ 7 and exact at 48, read from the Sep 25 records (134 and 122 puzzles).
- Run: fixed start; pass C's batching (64); continued to readout 96, i.e. 32 outer iterations, so that delays are observed rather than censored.

**Edit.** Applied once, at the start of cycle c = κ − 3k, for k ∈ {1, 2, 3} outer iterations before the completing cycle. Only puzzles with c ≥ 2 are eligible, so at least one cycle has run.

For every slow-state vector h (per field and cell), write h = h∥ + h⊥ along the shared readout vector v. Then rotate: h⊥′ = cos θ·h⊥ + sin θ·‖h⊥‖·û.
- û is a unit vector orthogonal to both v and h⊥, from a Gaussian draw with seed `[20261001, puzzle ID, 101, c]`.
- Every digit score and every vector norm is kept.
- Computed in float64 outside the jitted cycle and cast to float32, as in P8.

Angles: θ ∈ {0° (the sham, through the same path), 15°, 90°} at every k, plus 45° at k = 1 and k = 3.

**Outcomes.** τ′ is the first exact readout at or after the edit cycle. Δ = τ′ − κ, in readouts, with negative values meaning earlier completion. A puzzle not exact by readout 96 is "never". Exact at 96 is also recorded.

**Gates.**
- (A) The sham's τ′ equals the intact replay's (= κ) on at least 99 % of eligible puzzles at every k.
- (B) At k = 1 and θ = 90°, exact-at-κ lies within ±0.10 of P8's `perp_random@-3` (C5 0.157; SA128 0.107). Outside that band, the report says ROTATION DIFFERS FROM REPLACEMENT; this is not fatal.
- (C) Digit-score drift ≤ 1e-3.
- (D) The intact replay to 96 is exact at κ on every cohort puzzle.

**Rules (confirmatory, per model; at least 30 eligible puzzles at a k, else that k is UNDEFINED).** The shares are of eligible puzzles at the stated (k, θ): delayed (Δ ≥ 1, including never), earlier (Δ ≤ −1), unchanged (Δ = 0). m(k, θ) is the median Δ, with never set to 96 − κ + 1.

1. **SENSITIVE-TRANSIENT** if, at θ = 15° and some eligible k, the share with |Δ| ≥ 3 is at least 0.25 and the share earlier is at least 0.15.
2. **ACCUMULATION** if, at θ = 90°, all of these hold:
   - the share delayed is at least 0.5 at k = 3;
   - m(1, 90°) ≥ m(3, 90°) + 2 (delays shrink with distance from completion);
   - the share earlier is at most 0.10 at every k;
   - at k = 1 the share delayed rises strictly from 15° to 45° to 90°.
3. **RECOVERY-COST** if, at θ = 90°, the share delayed is at least 0.5 at every eligible k and |m(1, 90°) − m(3, 90°)| ≤ 1.
4. **IDLE** if, at θ = 90° and k = 3, the share unchanged is at least 0.7.
5. **MIXED** otherwise.

Precedence runs 1 > 2 > 3 > 4 > 5; every condition's values are printed.

**Exploratory (no letter).**
- Δ distributions per (k, θ).
- Exact at 96 per condition.
- The share that completes and is later lost.
- The relation of Δ to κ (plateau length) at fixed k.

**Predictions and credences.**
- Letters: SENSITIVE-TRANSIENT 0.25, ACCUMULATION 0.20, RECOVERY-COST 0.20, IDLE 0.10, MIXED 0.25.
- Gates pass: 0.85.
- At θ = 15° the share delayed at k = 1 is at most 0.3: 0.5.

**Wording under each outcome.**
- **SENSITIVE-TRANSIENT:** "Small, score-preserving perturbations of the slow state's invisible features before completion shift completion both earlier and later: the pre-completion phase is a sensitive transient, consistent with saddle-type trapping, and the time of completion is not set by the displayed answer."
- **ACCUMULATION:** "Pre-completion iterations build state beyond the displayed answer: erasing it delays completion in proportion to how much has been built, and never brings completion forward."
- **RECOVERY-COST:** "Disrupting the invisible features costs a roughly fixed delay wherever it lands: the state rebuilds them, and the pre-completion iterations do not store progress that a single edit erases."
- **IDLE:** "Edits early in the plateau leave completion time unchanged: what matters is concentrated in the last outer iteration before completion."
- **MIXED:** the conditions reported with every reading they support.

Under every outcome: no feature content is identified, and the edits use random directions.

---

## P11 — Does P9's identical-answer pattern hold at other checkpoints?

**Why.** At 94k, P9 found that the fixed start's loss of a solved puzzle at the next step needs both of its hidden components, and that either healthy component from the Gaussian trajectory prevents it. Outside readers called 94k a selected checkpoint. P5 reports the access contrast across the series; P11 asks whether the mechanism pattern holds at other checkpoints where the fixed start fails.

**Checkpoints.** 90k and 114k, chosen *before any row* as the post-50k checkpoints whose saved fixed-start endpoint is lowest after 94k. Manuscript Table 11 gives fixed-start endpoints of 12 and 14 against Gaussian 124 and 123. A third, 118k (fixed start 16), runs if time allows.

**Design.** P9's tool unchanged (`tools/rebuttal_p9.py`): same 128 validation puzzles, batch 128, the fixed start F, the independent Gaussian start G with seeds `[4242, i, 0, 7]`, the same primary-case and control definitions, the same nine swaps, the same gates. Only the checkpoint, the saved-flag file (`paper/code/evidence/initialization/s{step}.npz`) and the output folder change, set as module constants by the wrapper.

**Rules (per checkpoint; UNDEFINED with fewer than 20 primary cases or a failed gate).**
- **(i)** P9's registered letter, unchanged.
- **(ii) PATTERN-REPLICATES** if all of these hold:
  - rescue by the readout-orthogonal slow part is at least 0.8;
  - rescue by the fast state is at least 0.6;
  - transfer by the orthogonal part is at most 0.2;
  - transfer by the fast state is at most 0.2;
  - transfer by both states is at least 0.8.

  Otherwise PATTERN-DIFFERS.

**Predictions and credences.**
- PATTERN-REPLICATES at 90k: 0.6; at 114k: 0.5.
- The P9 letter is NONE at both: 0.6.
- At least 20 primary cases at each: 0.7.

**Wording under each outcome.**
- **PATTERN-REPLICATES:** "At N of N further checkpoints where the fixed start fails, the loss of a solved puzzle at the next step again needs both of the fixed trajectory's hidden components, and either healthy component prevents it."
- **PATTERN-DIFFERS:** reported per checkpoint, with the 94k pattern scoped to 94k.

---

## Order and labels

- **Order:** P10 builds and runs first, with the two models as two processes at nice 10. P11 follows, one process, 90k then 114k.
- **Confirmatory:** P10's letter per model with its gates; P11's two readings per checkpoint.
- **Exploratory:** everything else.
- **Records:** an Outcome section here and a ledger line for each.

---

## P10 Outcome (2026-10-01 14:12Z; `runs/analysis/rebuttal_20261001h/report.{txt,json}`; tool hash unchanged from `frozen_sha256.txt`; both runs exit 0; walls 95 and 105 min at nice 10)

**Context received after this note was written.** At 12:35Z, after this registration (12:26Z) and before any P10 row, the other session sent an interim P3 read: one random replacement before iteration 2 is recovered (RECOVERS), and on hard strata it sometimes lowers later error. The design, rules and predictions above were not changed.

**Gates, all passed on both models.**
- (A) The sham is unchanged on 100 % of eligible puzzles at every k.
- (B) At k = 1, θ = 90°, exact at κ is 0.142 vs P8's replacement 0.157 (C5), and 0.066 vs 0.107 (SA128); both are within ±0.10.
- (C) The largest score drift is 7.8e-7.
- (D) The intact replay is exact at κ on every cohort puzzle and still exact at readout 96.

**Shares of eligible puzzles** (delayed / earlier / unchanged; median Δ in readouts; 3 readouts = 1 outer iteration):

| model | k | n | θ = 15° | θ = 45° | θ = 90° |
|---|---|---|---|---|---|
| C5 | 1 | 134 | 0.39 / 0.10 / 0.51; m 0 | 0.66 / 0.12 / 0.22; m 1 | 0.86 / 0.05 / 0.09; m 5 |
| C5 | 2 | 119 | 0.61 / 0.24 / 0.15; m 1 | — | 0.61 / 0.31 / 0.08; m 2 |
| C5 | 3 | 88 | 0.64 / 0.33 / 0.03; m 3 | 0.49 / 0.44 / 0.07; m 0 | 0.61 / 0.34 / 0.05; m 3.5 |
| SA128 | 1 | 122 | 0.57 / 0.20 / 0.23; m 1 | 0.78 / 0.15 / 0.07; m 4 | 0.93 / 0.03 / 0.03; m 6 |
| SA128 | 2 | 110 | 0.68 / 0.23 / 0.09; m 4 | — | 0.66 / 0.25 / 0.09; m 3 |
| SA128 | 3 | 70 | 0.50 / 0.43 / 0.07; m 0.5 | 0.59 / 0.37 / 0.04; m 1 | 0.66 / 0.29 / 0.06; m 2 |

- The share with |Δ| ≥ 3 at θ = 15° is 0.26 / 0.54 / 0.75 (C5, k = 1 / 2 / 3) and 0.44 / 0.66 / 0.71 (SA128).
- Exact at 96 is at least 0.94 in every condition; "never" is at most 0.06.

**Registered letters: SENSITIVE-TRANSIENT on both models.**
- The sensitive condition holds at θ = 15° (for example C5 k = 3: |Δ| ≥ 3 in 0.75, earlier in 0.33; SA128 k = 3: 0.71 and 0.43).
- ACCUMULATION fails: the earlier-completion share exceeds 0.10, and delays do not shrink by 2 readouts from k = 1 to k = 3.
- RECOVERY-COST fails: |m(1) − m(3)| > 1 on C5, and > 1 on SA128 (6 vs 2).
- IDLE fails: 5 % / 6 % unchanged at k = 3.

**Predictions scored.**
- SENSITIVE-TRANSIENT (0.25): HIT on both models.
- Gates pass (0.85): HIT.
- "At θ = 15° the share delayed at k = 1 ≤ 0.3" (0.5): MISS (0.39 and 0.57).

**The registered sentence that applies.** "Small, score-preserving perturbations of the slow state's invisible features before completion shift completion both earlier and later: the pre-completion phase is a sensitive transient, consistent with saddle-type trapping, and the time of completion is not set by the displayed answer."

**Exploratory (no letter; computed after the letters).**
- The rotation direction is the same at every angle for a given puzzle and cycle. For puzzles moved by both the 15° and the 90° edit, the sign of the shift agrees in only 0.58 (C5) and 0.66 (SA128) of cases at k = 3, but in 0.84 and 0.70 at k = 1. The response to a push is close to unrelated to its size three iterations out, and more consistent one iteration out.
- At k = 3 and θ = 15°, the 10th–90th percentile of Δ is −4 to +18 readouts (C5) and −5 to +30 (SA128).
- |Δ| barely relates to plateau length: Spearman −0.04 (C5) and 0.24 (SA128).

**Reading (descriptive).** The approach to completion has two phases.
1. Two to three outer iterations before completion, the time of completion depends sensitively on slow-state features the readout cannot see. A 15° rotation of them, with every digit score unchanged, moves completion by an outer iteration or more in most puzzles, earlier or later about as often as a 90° rotation does. The direction of the shift is weakly related to the push.
2. In the last outer iteration the process is directed: the same features are needed, and disturbing them mostly delays completion (P8).

Completion itself is robust: at least 94 % complete within 32 iterations under every edit. This is the causal counterpart, in these models, of the transient-chaos and saddle account (Lai et al., 2026). It also explains why the displayed answer, and even identical answers (P9), do not fix what further iterations will do.

**Limits.**
- One checkpoint per model and the stratified cohorts (κ ≥ 7).
- Random rotation directions within the readout-invisible subspace; no learned directions.
- The 45° rows exist only at k = 1 and k = 3.
- The sign-agreement analysis is post-hoc.

---

## P11 Outcome (2026-10-01 15:08Z; `runs/analysis/rebuttal_20261001i/report.txt` and `s{090000,114000}/report.{txt,json}`; P9's tool unchanged, hash identical to P9's frozen hash; the wrapper's hash unchanged from `frozen_sha256.txt`; run log exit 0)

**Gates, all passed at both checkpoints.**
- (1) The replicated loop reproduces the saved flags exactly: fixed start 12 and 14 at iteration 16; Gaussian 124 and 123.
- (2) The full swaps reproduce the donor.
- (3) The sham agrees 1.000.
- (4) The largest score drift is 9.0e-7.

The run log's gate line says "94k" because the label is hard-coded in P9's tool. The flags were read from `s090000.npz` and `s114000.npz`.

**Results, with 94k (P9) beside them.**

| checkpoint | primary cases | rescue: orthogonal slow / full slow / fast | transfer: orthogonal slow / full slow / fast / both | P9 letter | P11 rule (ii) |
|---|---|---|---|---|---|
| 90k | 78 | 1.00 / 1.00 / 0.29 | 0.01 / 0.71 / 0.00 / 1.00 | FULL-SLOW-ONLY | PATTERN-DIFFERS |
| 94k (P9) | 60 | 1.00 / 1.00 / 0.88 | 0.00 / 0.12 / 0.00 / 1.00 | NONE | (pattern defined here) |
| 114k | 60 | 1.00 / 1.00 / 1.00 | 0.00 / 0.00 / 0.00 / 1.00 | NONE | PATTERN-REPLICATES |

Controls: at 114k no swap breaks a stable answer. At 90k, the fast-state swap into F leaves 0.73 of controls exact, and the full-slow-state swap into G leaves 0.73.

**Predictions scored.**
- PATTERN-REPLICATES at 90k (0.6): MISS.
- PATTERN-REPLICATES at 114k (0.5): HIT.
- P9 letter NONE at both (0.6): HIT at 114k, MISS at 90k.
- At least 20 primary cases at each (0.7): HIT.

**Registered wording.** PATTERN-REPLICATES at one of two further checkpoints (114k); PATTERN-DIFFERS at 90k, reported per checkpoint.

**Reading (descriptive).**
- One element holds at all three checkpoints: at identical displayed answers and identical digit scores, giving the failing fixed-start trajectory the Gaussian trajectory's readout-invisible slow content prevents the loss at the next step in every case (78/78, 60/60, 60/60).
- What carries the failure differs.
  - At 94k and 114k it needs both of the fixed trajectory's hidden components; either healthy one prevents it.
  - At 90k the fixed trajectory's full slow state carries it into the Gaussian trajectory (55/78). Its readout-invisible part alone does not (1/78). The readout-parallel component — the score magnitudes behind the same argmax — is therefore part of what fails at 90k.

**Limits.**
- Three checkpoints of one run, 128 validation puzzles, and one Gaussian draw per puzzle.
- The hybrid states are off-trajectory constructions.
