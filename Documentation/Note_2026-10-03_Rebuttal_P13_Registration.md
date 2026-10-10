# Note 2026-10-03 — P13: does a single recurrent state need its readout-invisible part to complete? (registration, before any row)

## Page one

**Goal.** A MEASUREMENT. No accuracy target; $0; the Mac's CPU; inference only on banked checkpoints; nothing trained.

**Provenance.**
- The PI (2026-10-03): "Let's also queue the mac follow ups on the SA256U."
- The follow-up named in round 2's registration: P8's protocol on SA256U's selected checkpoint.
- Session d378a5's P-slot block (P7–P14). P13 and P14 were earlier proposals that were never registered: P10 on more models, and a pod-scale kick. Both are retired; the second was covered by the other session's P17.
- Reviewed before registration by the other session (3beab6), which suggested five changes, all adopted: oversampling with a minimum cohort, the 1° qualifier, the 14k control, tighter gates, and the explicit B formula.
- Tool: `tools/rebuttal_p13.py` (selftest 9/9).
- **Gated smoke, 2026-10-03 10:19–10:24Z:** 8 puzzles, 6 iterations, every condition at iteration 4, on all three receivers. It printed no outcome.
  - G1 (`run_batch` bitwise) and G2 (determinism) hold on all three.
  - The sham's carry is bitwise unchanged.
  - Every edit's largest digit-score change is ≤ 6.2e-7 and its largest relative norm change ≤ 1.1e-8.
  - 65–95 s per receiver.
- Outputs: `runs/analysis/p13_20261003/`.
- CPU at nice 10, after the other session's P23 finishes (about 11:45Z). About 1.5–2 h.

## Why

**P8 (2026-10-01) on the two-state attention model.** Attention 128 reads NEEDS-HIDDEN-STATE (B 0.984): replacing only the readout-invisible part of the slow state, once, at the start of the completing iteration κ, blocks nearly every completion, with every digit score kept.

**Round 2 (2026-10-03).** Collapsing the two states into one costs 8.3 points at 16 iterations, and the single state completes fast and repairs little.

**The question for the manuscript's account.** Is "completion uses state beyond the displayed answer" a property of the two-state recurrence, or of recurrent reasoning in general? P13 asks it on a matched pair trained with the same recipe:

| receiver | what | checkpoint (EMA) | role |
|---|---|---|---|
| SA256U14 | one state (round 2's arm), selected 14k | `runs/_p13_ckpts/SA256U_ckpt_014000.pkl`, sha256 486f5523… (from `attr2_p0/live/runs`) | the question |
| SA256_28 | two states (the ladder's SA256), selected 28k | `runs/_p13_ckpts/SA256_ckpt_028000.pkl`, 18e9427f… (from `wladder_p1/live/runs`) | the matched control |
| SA256_14 | two states, 14k | `runs/_p13_ckpts/SA256_ckpt_014000.pkl`, b98c802e… | exploratory: SA256U's training stage |

## Design

**Code.** The MAIN repo's `src/qhrrn2`; the release code has no single-state branch. The loader is the evaluator's own rebuild: `E.load_ckpt` and `Config(**{k: type(default)(v)})`, with EMA weights. The loop is `eval_sudoku_extreme.run_batch` mirrored line for line, through its own `_step`, from the fixed start, with one hook: an edit of the carry entering a named iteration.

**Population, per receiver.**
- **Candidates:** from round 2's accelerator depth-64 records on the identical 5,000 test puzzles (`runs/_attr2_pull/stage/runs`). A puzzle qualifies if it is first exact at iterations 7–16 and exact at 64. A seeded draw of ≤ 256 is taken, seed `[20261003, 13, receiver index]`; SA256_14 uses SA256's records.
- **The intact run:** 20 iterations on this CPU, for all candidates.
- **Cohort:** first exact at κ in 7..16 on this run and exact at 20. A seeded draw of ≤ 96 is kept.
- **By construction,** P(exact at κ | intact) = 1.

**Conditions,** each applied once to the carry entering iteration κ + offset, then continued unedited to iteration 20:

| condition | edit | receivers |
|---|---|---|
| `sham` | the reassembly: `rotate_perp` at θ = 0 | all |
| `perp_random` @κ | **P8's edit:** per field and cell, the readout-bearing state's part orthogonal to the shared readout vector is replaced by a Gaussian draw projected orthogonal to it and rescaled to the old part's norm. Every digit score and every vector norm is kept. | all |
| `perp_random_m3` @κ−3 | the same, three iterations earlier | SA256U14, SA256_28 |
| `rot1` @κ | a 1° score- and norm-preserving rotation of that part (`tools/rebuttal_p10.rotate_perp`, imported unchanged) | all |
| `hidden_all` @κ | `perp_random` plus the fast state replaced by a norm-matched Gaussian, so the amount of hidden state edited matches the single state's | SA256_28 only (exploratory) |

The readout-bearing state is the one carry (SA256U) or the slow state (SA256). Edits are made in float64 and cast to float32. Seeds are `[20261003, puzzle ID, 81, iteration, kind]`.

## Gates (per receiver; any failure makes its letters UNDEFINED)

- **G1.** The P13 loop's intact flags equal `run_batch`'s, bitwise, on the first 32 candidates over 20 iterations.
- **G2.** The intact run, repeated, is bitwise identical on those 32.
- **G3.** Every edit changes no digit score by more than 1e-3 (float64).
- **G4.** Every edit keeps each edited vector's norm within a relative 1e-5.
- **G5.** The sham's carry is bitwise unchanged, and its exact flags equal intact's at every iteration on every cohort puzzle.
- **Cohort size** ≥ 40.

## Rules (confirmatory, per receiver)

- **R13a.** B = 1 − P(exact at κ | `perp_random`@κ).
  - NEEDS-HIDDEN-STATE if B ≥ 0.5;
  - READOUT-SUFFICES if B ≤ 0.1;
  - MIXED otherwise.
- **R13b, the qualifier when B ≥ 0.5** (the other session's P22: stalled states re-roll under 1° pushes): S1 = 1 − P(exact at κ | `rot1`@κ).
  - CONTENT if S1 ≤ 0.1: a large replacement breaks completion, a 1° push does not, so the content matters;
  - SENSITIVE if S1 ≥ 0.5: even 1° breaks it, so sensitivity, not content;
  - INTERMEDIATE otherwise.
- **The comparison** of SA256U14 with SA256_28 is reported side by side, descriptively. Different cohorts allow no paired test.

**Descriptive (no label).**
- `perp_random_m3`, `hidden_all` and SA256_14;
- exact at 20 and the median delay to the first exact iteration after each edit;
- the CPU-against-accelerator agreement of the first exact iteration on the pool.

## Predictions and credences

| prediction | credence |
|---|---|
| All gates pass on all three receivers | 0.80 |
| SA256_28 R13a NEEDS-HIDDEN-STATE (P8: Attention 128 B 0.984) | 0.80 |
| SA256U14 R13a NEEDS-HIDDEN-STATE | 0.40 |
| SA256U14 R13a MIXED | 0.35 |
| SA256U14 R13a READOUT-SUFFICES | 0.25 |
| Where NEEDS-HIDDEN-STATE holds: CONTENT | 0.55 |
| Where NEEDS-HIDDEN-STATE holds: INTERMEDIATE | 0.30 |
| Where NEEDS-HIDDEN-STATE holds: SENSITIVE | 0.15 |

The other session's Amendment 3 found states one step from completion frozen to small pushes, which favours CONTENT at κ.

## Wording under each outcome

- **SA256U14 NEEDS-HIDDEN-STATE (CONTENT), as SA256_28:** "A single-state model also completes from state its readout cannot see. Using hidden state at completion is a property of the recurrence, not of the two-state split; the split's advantage lies elsewhere (repair, generalization: round 2)."
- **SA256U14 READOUT-SUFFICES or MIXED while SA256_28 NEEDS-HIDDEN-STATE:** "The single state completes largely from what it displays. Completion through readout-invisible state is a property of the two-state recurrence, the same structure that carries round 2's accuracy."
- **SENSITIVE anywhere:** "At the completing iteration that model's state is sensitive to any push, so the large edit's effect is not evidence that its content is needed."

Under every outcome:
- one checkpoint per receiver;
- this CPU's trajectories, whose first exact iteration can differ from the accelerator's;
- random directions, not learned ones;
- no feature content is identified.

## Outcome (2026-10-03 12:46Z; `runs/analysis/p13_20261003/report.{txt,json}`)

The tool's sha256 equals the registered `fecbec9`'s (066648ed…; `frozen_sha256.txt`). The run ended with exit 0, 11:50–12:46Z at nice 10.

**Gates: all PASS on all three receivers.**
- G1: the loop is bitwise equal to `run_batch`.
- G2: repeated runs are identical.
- G3: every edit's score change is ≤ 1e-3.
- G4: every edit keeps vector norms within 1e-5.
- G5: the sham's carry is bitwise unchanged, and its flags equal intact's.
- Every cohort is ≥ 40.

| receiver | cohort | R13a: B = 1 − P(exact at κ \| `perp_random`@κ) | R13b: S1 = 1 − P(exact at κ \| `rot1`@κ) |
|---|---|---|---|
| SA256U14 (one state, 14k) | 78 | **1.000** (78/78 blocked): **NEEDS-HIDDEN-STATE** | 0.077 (6/78): **CONTENT** |
| SA256_28 (two states, 28k) | 94 | **0.819** (77/94): **NEEDS-HIDDEN-STATE** | 0.053 (5/94): **CONTENT** |
| SA256_14 (two states, 14k; exploratory) | 79 | 0.696 (55/79): NEEDS-HIDDEN-STATE | 0.000 (0/79): CONTENT |

**Descriptive.** Per condition: exact at κ, exact at iteration 20, and the median delay to the first exact iteration.

| receiver | `perp_random`@κ | `perp_random`@κ−3 | `rot1`@κ | `hidden_all`@κ |
|---|---|---|---|---|
| SA256U14 | 0.000 / 0.615 / 4.5 | 0.295 / 0.795 / 1.0 | 0.923 / 0.974 / 0 | n/a |
| SA256_28 | 0.181 / 0.840 / 3.0 | 0.500 / 0.851 / 0 | 0.947 / 0.989 / 0 | 0.032 / 0.862 / 3.0 |
| SA256_14 | 0.304 / 0.785 / 1.5 | n/a | 1.000 / 1.000 / 0 | n/a |

**Hardware.** This CPU's first exact iteration equals the accelerator record's on only 5.5–6.2 % of each pool. That is consistent with the accelerator's default matmul precision against the CPU's float32. Each cohort was defined on the CPU's own trajectories, as registered, so the letters are internally valid. The exact completing iteration, however, is not reproducible across hardware.

**Predictions scored.**

| prediction | credence | result |
|---|---|---|
| All gates pass | 0.80 | yes |
| SA256_28 NEEDS-HIDDEN-STATE | 0.80 | yes |
| SA256U14 NEEDS-HIDDEN-STATE | 0.40 | yes |
| CONTENT where NEEDS-HIDDEN-STATE holds | 0.55 | yes, on all three |

**Reading (the registered wording for SA256U14 NEEDS-HIDDEN-STATE (CONTENT), as SA256_28).** "A single-state model also completes from state its readout cannot see. Using hidden state at completion is a property of the recurrence, not of the two-state split; the split's advantage lies elsewhere (repair, generalization: round 2)."

The 1° qualifier makes it content, not sensitivity, in both structures. A one-degree rotation of the same readout-orthogonal part at κ leaves 92–100 % of completions intact, while replacing it blocks 70–100 %.

**Descriptive observations,** consistent with round 2's repair reading but not registered claims:
- **One state, everything replaced:** in the one-state model the edit replaces all of its hidden state, and it blocks every completion. After it, the model recovers less (61.5 % exact by iteration 20, median delay 4.5) than the two-state model after either edit. That holds for the slow-state-only edit (84.0 %, delay 3.0) and for the edit that also replaces the fast state (86.2 %).
- **One state is not just a younger model:** at the same training stage (14k) the two-state model is blocked less (0.696) and recovers more (78.5 %).

Under every outcome:
- one checkpoint per receiver;
- this CPU's trajectories;
- random directions, not learned ones;
- no feature content is identified.
