# Note 2026-10-03 — SE-RRM attribution, round 3: SE-RRM's dropout on both state structures (registration, before any row)

## Page one

**Goal.** A MEASUREMENT that settles one question, not a better score: does regularization replace the two-state recurrence's advantage?

**What it needs, and who approved it.**
- Pod time on ONE spot TPU v6e-8 in the ARC project (Mumbai, the PI's standing choice there), two arms in sequence.
- Cap: $140. Expected spend: about $95 (each arm about 5.7 h, as SA256U took; one resumable wall recycle in the second arm).
- The PI, after round 2's outcome and my recommendation to defer this round: "let's get round 3 done so we can settle the question once and for all and move on" (2026-10-03, about 15:20 IST).
- Plan: `Plan_2026-10-01_Rebuttal_Experiments.md` §6. It listed dropout among round 2's new cell variants; it runs now as attribution round 3. The plan's own 'round 3' (the GPU cross-check with SE-RRM's code) is a different, unscheduled study. Session d378a5.

**Expected accuracy** on the identical 5,000 test puzzles at 16 / 64 iterations (one seed per arm; seeded floors 2.58 / 2.44 pp):

| arm | expected | for comparison |
|---|---|---|
| SA256UD (one state + dropout 0.2) | 90–97 / 93–99.5 | SA256U (round 2; one state, no dropout): 89.96 / 93.08 |
| SA256D (two states + dropout 0.2) | 96–98.6 / 99.2–99.7 | SA256 (the reference; two states, no dropout): 98.24 / 99.60 |
| | | SE-RRM's published single run (one state with dropout 0.2; full test): 93.73 / 98.22 |

## Why

Round 2 (`Note_2026-10-02_Attribution_Round2_Registration.md`, Outcome) found TWO-STATE-CARRIES:
- SE-RRM's single recurrent state costs our system 8.28 pp at 16 iterations (SA256U 89.96 against SA256 98.24). It lands below SE-RRM's own published 93.73.
- **Descriptively, the single state memorizes the 1,000 training puzzles after 14k steps.** The training rows' exact share rises from 0.41 to 0.95 while validation falls from 92.8 to 64–72; the two-state model never does this.
- **It also generalizes worse before any memorization.** At 8k–12k steps the two models fit the training set equally, yet the single state is 6–11 pp behind on validation.

SE-RRM trains its single state with dropout 0.2. A regularizer against memorization may be what lets their single state reach 93.73. The open question is therefore whether the two-state advantage is something regularization replaces. Round 3 adds SE-RRM's dropout to both structures, completing a 2 × 2 with round 2:

| | no dropout | dropout 0.2 (SE-RRM's) |
|---|---|---|
| two states | SA256 (the ladder's, banked) | **SA256D** (new) |
| one state | SA256U (round 2's, banked) | **SA256UD** (new) |

**SE-RRM's dropout, read at the source today** (`models/recursive_reasoning/trm_equi.py`, github.com/ml-jku/SE-RRM; the released Sudoku command passes `arch.dropout=0.2`):
- It is applied only in the two attention modules, the attention over positions and the attention over symbols.
- It acts as `F.scaled_dot_product_attention(..., dropout_p=self.dropout)`: dropout on the attention weights.
- The MLP has none.
- **Unmoved difference:** the call carries no training-mode check, and `evaluate()` only calls `model.eval()` under `torch.inference_mode()`. In the released code, then, the dropout appears to be active at evaluation too. Whether their reported runs evaluated that way is not stated. We apply dropout in training only.

## The arms

Each arm lives in `tools/chain_champ.sh`, admitted by `tools/chain_sablate.sh`. Budget fixed at 30,000 (explicit `arm_steps` entries), never extended; seed 0.

| arm | flags | differs from SA256 in |
|---|---|---|
| SA256UD | `$(arm_flags SA256U) --dec-dropout 0.2` | `dec_single_state: True`, `dec_dropout: 0.2` |
| SA256D | `$(arm_flags SA256) --dec-dropout 0.2` | `dec_dropout: 0.2` |

What `--dec-dropout 0.2` changes (`Config.dec_dropout`; `dec_cell._drop`, `_attn_tok`, `_block`, `_stack`, `segment`):
- **What is dropped:** inverted dropout on the attention weights of both mixers, the attention over the cells within each field and the attention over the fields at each cell. Each weight is kept with probability 0.8 and rescaled by 1 / 0.8, as torch's `dropout_p` does.
- **Keys:** one per stack application, folded per block and per field.
- **Where it acts:** only in the training forward. `segment` drops only when the trainer passes `drop_rng`, which it does only for the training rows and the anchor rows (`pretrain.drop_kw`). The evaluators, the training-time monitors and any noise-at-evaluation path pass none, so selection and every reported row see the deterministic network.

Everything else is SA256's protocol: the battery, D16 and D64 from the fixed start on the identical 5,000 (both recording the per-iteration bits), plus the 5,000 × k32 restart scan. SA256 and SA256U are READ, never re-run.

## Rules (frozen in `tools/analyze_attr3.py`; selftest 23/23)

- **INTEGRITY.**
  - SA256D's argv and model config differ from SA256's in exactly `dec_dropout: 0.2`; SA256UD's in exactly `dec_single_state: True` and `dec_dropout: 0.2`.
  - Seed 0; budget 30,000 with no extension marker; the last training step equals the budget.
  - Rows n 5,000, EMA and the right depth, on SA256's puzzle ids; all rows and the scan on the selected grid.
  - Round 2's SA256U rows present on the same ids; every D64 record carries the per-iteration bits.
- **R-AB-1, R-AB-2, R-AB-4** as in rounds 1–2: each new arm vs SA256 at 16 and 64 (INSIDE / BELOW-BEYOND / ABOVE-BEYOND, paired exact McNemar on the identical 5,000; a p-value alone never moves a label); vs SE-RRM's published numbers; the k32 selector (CLEAN at ≤ 1 %).
- **R-A3-1, the structure under dropout:** SA256UD vs SA256D.
- **R-A3-2, dropout on the single state:** SA256UD vs SA256U.
- **R-A3-3, dropout on the two states:** SA256D vs SA256.
- **R-A3-4, the regularized single state vs our reference:** SA256UD vs SA256.
- **R-A3-5, the settlement at 16** (the registered question; the same combination is printed at 64 as descriptive):

  | reading | R-A3-1 | R-A3-4 |
  |---|---|---|
  | STRUCTURE-SPECIFIC | BELOW-BEYOND | any |
  | REGULARIZATION-REPLACES | INSIDE | INSIDE |
  | PARTIAL | INSIDE | BELOW-BEYOND |
  | OTHER | any other combination | |

- **DESCRIPTIVE** (no label moves):
  - the interaction: the structure effect without dropout (round 2) against R-A3-1's;
  - round 2's where / persistence / consistency readings for each new arm against SA256 (imported unchanged from `tools/analyze_attr2.py`);
  - MEMORIZATION per arm: validation maximum and end, and the training rows' exact share at the selected step and at the end;
  - STABILITY: the selected grid (EDGE when last), the scan's fixed against one random start, verified accuracy, non-finite losses.

## Predictions and credences (before any row)

| prediction | credence |
|---|---|
| R-A3-5 @16 STRUCTURE-SPECIFIC | 0.65 |
| R-A3-5 @16 REGULARIZATION-REPLACES | 0.15 |
| R-A3-5 @16 PARTIAL | 0.10 |
| R-A3-5 @16 OTHER | 0.10 |
| R-A3-2 @16 (dropout on one state) ABOVE-BEYOND | 0.45 |
| R-A3-2 @16 INSIDE | 0.50 |
| R-A3-3 @16 (dropout on two states) INSIDE | 0.75 |
| R-A3-3 @16 BELOW-BEYOND | 0.25 |
| (descriptive) SA256UD ends with a lower training rows' exact share than SA256U's 0.95 | 0.75 |
| Both new selectors CLEAN | 0.85 |

ABOVE-BEYOND is impossible for SA256D at either depth: the reference plus the floor exceeds 100 %.

## Wording under each outcome

- **STRUCTURE-SPECIFIC.** "With SE-RRM's own regularizer on both, the single recurrent state still costs X points at 16 iterations, beyond the seed floor. The two-state advantage is not a substitute for regularization: the separate slow state carries something dropout does not provide." The recurrence structure stands as the answer to the attribution question.
- **REGULARIZATION-REPLACES.** "With SE-RRM's dropout, the single state matches both two-state models within the floor. In our system, the two-state structure's advantage is equivalent to regularization against memorizing the 1,000 training puzzles, which our two-state model obtains without dropout." The answer becomes "a structure that regularizes", not "a structure that does something regularization cannot".
- **PARTIAL.** "Under equal dropout the two structures read inside the floor, but dropout costs the two-state model. Our unregularized two-state model remains the best configuration."
- **OTHER.** Both letters are reported, with the readings they support.

Under every outcome:
- one seed per arm;
- the 5,000-puzzle subsample;
- SE-RRM's number is their run on their evaluation; their evaluation-time dropout, halting, AdamATan2, precision and layout stay unmoved;
- the descriptive readings describe; a claim built on them needs its own registration.

## Build and checks (before the launch)

- **Code.**
  - `Config.dec_dropout` (default 0.0), and the dropout path through `dec_cell._drop`, `_attn_tok`, `_block`, `_stack` and `segment(drop_rng=)`.
  - In `tools/pretrain.py`: the flag `--dec-dropout` (argparse-suppressed when absent); `drop_kw` at the two training calls; a guard refusing dropout on a cell without attention mixers.
  - `RECORD_OMIT_AT_DEFAULT` now also covers `dec_dropout` at 0.0, matched by type and value.
- **Tests:** `tests/test_dropout.py` 8/8; `tests/test_single_state.py` 6/6, two assertions updated to the new signature and records; `tests/test_final.py` and `tests/test_champ_builds.py` unchanged and passing. They cover:
  - the default is bit-exact against an independent replica whatever `drop_rng` is;
  - with `dec_dropout` 0.2, every path without `drop_rng` (the evaluators, monitors, noise at evaluation) is bit-exact against 0.0;
  - the training forward drops, deterministically per key, on both structures;
  - the mask keeps 80 % and preserves the expectation;
  - a cell without attention is untouched;
  - gradients flow;
  - the records omit the field at its default and round-trip through the evaluators' rebuild.
- **Trainer smokes (CPU).**
  - Width 32, one state with dropout: 20 steps with finite loss.
  - Both arms' exact chain argv at width 256, 2 steps: their records against the banked SA256 differ in exactly the registered keys.
  - At the same seed, dropout changes the step-2 loss (SA256D 4.4079 vs SA256 4.4075; SA256UD 4.4717 vs SA256U 4.4656).
  - SA256U's loss and weights are bit-identical before and after the change.
- **Evaluator smoke.** `tools/eval_sudoku_extreme.py` loads the SA256UD checkpoint and rebuilds `dec_dropout = 0.2` (it never drops).
- **Harness `tools/harness_attr3.sh`: 44/44.** It covers:
  - each arm alone, and both in sequence on one pod (the launch configuration);
  - each arm's trainer argv token for token against the registry;
  - an idempotent rerun, a preflight abort, and refusal of the reference and of an unknown arm;
  - **four mutants, all caught:** SA256UD without dropout, SA256D at rate 0.1, SA256UD out of `fixed_budget`, and SA256D at a 36k budget.
- **Earlier suites, re-run after the edits:** `tools/harness_attr2.sh` 17/17, `tools/harness_attr1.sh` 28/28 and `tools/harness_sablate.sh` 25/25; the analyzer selftests 28/28, 18/18 and 25/25. Earlier rounds' chain lines are kept verbatim, and round 3's arms have their own case patterns.

## Ops

- **The pod.** One spot v6e-8 named `qhrrn2-arc-pod`, launched with the fleet verified at zero. It writes under the fresh prefix `gs://qhrrn2-arc/rescue/attr3_p0`, verified empty before launch. `AB_ARMS="SA256UD SA256D"` runs in sequence: the decisive arm first.
- **Wall.** WALL is 30,600 s, so one resumable wall recycle is expected in the second arm's training. It happens on the same node, from the local checkpoint, with the live bank as the fallback.
- **Safety and monitoring.**
  - Set `runs/tpu_deadline.txt` to launch + 15 h FIRST; check the launchd watchdog.
  - The on-node guard is re-planted by the supervisor.
  - Watch heartbeats every 15 minutes with `tools/ops_watch_pods.sh tools/campaign_attr3_p0.env`.
- **Shared-project rules.** Only `qhrrn2-*` resources are created or deleted; nothing else in the project is touched.
- **The close.** Pull the final manifest; verify fleet zero at the source; stage both arms with the ladder's SA256 and round 2's SA256U dirs; run the analyzer FIRST, untouched; then the report, ledger and Outcome here.
