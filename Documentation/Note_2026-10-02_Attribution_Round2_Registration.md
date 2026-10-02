# Note 2026-10-02 — SE-RRM attribution, round 2: SE-RRM's single recurrent state in our system (registration, before any row)

## Page one

**Goal.** A MEASUREMENT that locates a difference, not a better score.

**What it needs, and who approved it.**
- Pod time on ONE spot TPU v6e-8 in the ARC project (Mumbai, the PI's standing choice there), one arm.
- Cap: $70. Expected spend: about $48 (SA256 trained at 2.0 steps/s on this pod type: 30k steps ≈ 4.2 h, plus about 1.4 h of battery and bring-up).
- The PI: "start building round 2's single-state arm while round 1 runs" (2026-10-02, about 19:00 IST); then, after reading the build summary: "yes, commit and launch round 2 when round 1 closes" (about 19:42 IST). The build follows the plan's round-2 proposal (`paper/documentation/Mac_Results_Digest_2026-10-01.md` §4b, private). **The launch follows round 1's close (one pod at a time).**
- Plan: `Plan_2026-10-01_Rebuttal_Experiments.md` §6 (round 2: a new cell variant). Session d378a5.

**Expected accuracy** on the identical 5,000 test puzzles at 16 / 64 iterations (one seed; seeded floors 2.58 / 2.44 pp):

| arm | expected | for comparison |
|---|---|---|
| SA256U | 92–98 / 97.5–99.6 | the reference SA256 (the width ladder's banked row; 30k, selected 28k): 98.24 / 99.60 |
| | | SE-RRM's published single run (full test): 93.73 / 98.22 |

## Why

Our attention model at hidden 256 holds SE-RRM's two mixers inside our block, loop and recipe, at about the same parameter count, and reads far above SE-RRM's published number. What has been moved so far:
- damping and noise: inside the floor (the recipe ablation, 2026-09-19);
- randomized starts and anchor rows: inside the floor (same);
- SE-RRM's *table* optimizer (batch 272, lr 5e-4): far below;
- SE-RRM's *released* optimizer and budget, and batch 272 at equal rows: round 1, running.

The largest structural difference left is the recurrent state. Read at the source today:
- **SE-RRM** (`models/recursive_reasoning/trm_equi.py`, github.com/ml-jku/SE-RRM): "there is only one hidden variable z, no distinction between higher and lower modules". Each step applies z ← L(z + x) 3 × 6 = 18 times, the first 12 without gradient and the last 6 with it.
- **Ours:** two states. A fast state is updated 6 times from the slow state plus the input; then the slow state is updated once from the fast state. Three cycles = 21 applications, the gradient through the last 7.
- **TRM** (arXiv 2510.04871, §4.2 and its Figure 4) tested exactly this variable. Its single-z variant updates one z = net(x, z) 7 times per cycle, 3 cycles, the gradient through the last cycle. In TRM's MLP model and recipe on Sudoku-Extreme it reads 71.9 against 87.4 for the two-feature model (its Table 2). TRM's explanation: a single state "forces the model to store the solution y within z".

So the single state cost TRM about 15 points in its system, yet SE-RRM reaches 93.73 with one. Round 2 asks which holds in ours. It also ties the accuracy question to the manuscript's measurements of the retained state: those locate repair information in the readout-invisible part of the separate slow state.

## The arm

`SA256U` = `$(arm_flags SA256) --dec-single-state` in `tools/chain_champ.sh`, admitted by `tools/chain_sablate.sh`. Budget fixed at 30,000 (an explicit `arm_steps` entry), never extended; seed 0.

What `--dec-single-state` changes (`dec_cell.segment`; `Config.dec_single_state`):
- **One carry.** It occupies the slow-state slot, which the readout and the halting head read.
- **The update.** 3 × (6 + 1) = 21 applications per segment of z ← 0.05 z + 0.95 B(z + x) + noise 0.01: the input is injected at every application, with the same damping, noise and key schedule as the two-state loop.
- **The gradient** flows through the last 7 applications.
- **The fast-state slot** passes through untouched.

This is TRM's single-z loop, pseudocode for pseudocode. It differs from SE-RRM's released loop only in the count (21 against 18) and the gradient window (7 against 6). We keep our count so that compute per iteration and the gradient depth match the reference: the arm moves the structure alone.

Unchanged from SA256, by construction:
- **The rest of the recipe:** parameters, mixers, width, readout, halting head, randomized starts σ 1 (now drawn for the one carry), anchor rows k 1 (their corrupted answer now starts the one carry), optimizer, data, selection rule and battery.
- **The battery:** D16 and D64 from the fixed start on the identical 5,000, both recording the per-iteration bits, plus the 5,000 × k32 restart scan.

The reference SA256 row is READ, never re-run.

## Rules (frozen in `tools/analyze_attr2.py`; selftest 28/28)

- **INTEGRITY.** SA256U's trainer argv differs from SA256's in exactly `dec_single_state: True`, and its model config in exactly `dec_single_state: True`.
  - Seed 0; budget 30,000 with no extension marker; the last training step equals the budget.
  - Rows n 5,000, EMA and the right depth, on SA256's puzzle ids; all rows and the scan on the selected grid.
  - Both arms' D64 records carry the 64 per-iteration bits.
- **R-AB-1.** SA256U vs SA256 at 16 and 64, with a paired exact McNemar test on the identical 5,000: INSIDE the floor (2.58 / 2.44 pp), BELOW-BEYOND or ABOVE-BEYOND. A p-value alone never moves a label.
- **R-AB-2.** SA256U vs SE-RRM's published 93.73 / 98.22: WITHIN, ABOVE or BELOW.
- **R-AB-3.** LOCATED at 16 / 64 when SA256U reads BELOW-BEYOND, with its share of the SA256-to-SE-RRM gap.
- **R-AB-4.** The selector from the k32 scan: CLEAN at a spurious rate ≤ 1 %, else DIRTY.
- **R-A2-5, the structure reading** (the registered question), at 16 and at 64:

  | reading | SA256U vs SA256 |
  |---|---|
  | TWO-STATE-CARRIES | BELOW-BEYOND |
  | NO-STRUCTURE-EFFECT | INSIDE |
  | SINGLE-STATE-BETTER | ABOVE-BEYOND |

- **R-A2-6, where** (descriptive; no label moves). From each arm's D64 per-iteration bits on the identical 5,000:
  - exact at iterations 1, 2, 4, 8, 16, 32 and 64, with the paired difference and McNemar p at each;
  - the quartiles of the first exact iteration among the puzzles exact at 64;
  - the late share: exact at 64 and not at 16.
- **R-A2-7, persistence** (descriptive). Per arm:
  - LOST: exact at some iteration ≤ 64 and not at 64;
  - DROPPED: any exact → not-exact step within the 64, later recoveries included;
  - the paired McNemar on DROPPED.
- **CONSISTENCY** (descriptive): each arm's D64 bits at iteration 16 against its own D16 row (the same fixed start), as an agreement count.
- **STABILITY** (descriptive): the selected grid, flagged EDGE when it is the last grid; validation maximum and end; the scan's fixed start against one random start; verified accuracy; non-finite losses.

## Predictions and credences (before any row)

| prediction | credence |
|---|---|
| R-A2-5 @16 TWO-STATE-CARRIES | 0.50 |
| R-A2-5 @16 NO-STRUCTURE-EFFECT | 0.50 |
| R-A2-5 @64 NO-STRUCTURE-EFFECT | 0.75 |
| R-A2-5 @64 TWO-STATE-CARRIES | 0.25 |
| R-AB-2 @16 WITHIN SE-RRM's floor band | 0.45 |
| R-AB-2 @16 ABOVE | 0.50 |
| R-AB-2 @16 BELOW | 0.05 |
| Selector CLEAN | 0.85 |
| The selected grid at the budget's edge (EDGE) | 0.40 |
| (descriptive) if SA256U is below at 16, the gap is wider at iterations 4–16 than at 64 | 0.60 |
| (descriptive) SA256U DROPS at least as many puzzles as SA256 | 0.60 |

SINGLE-STATE-BETTER cannot occur against this reference at either depth: 98.24 + 2.58 and 99.60 + 2.44 both exceed 100 %. The label stays so that the rule is complete.

## Wording under each outcome

- **TWO-STATE-CARRIES at 16.** "With every other part of our system unchanged, including the number of network applications per iteration, SE-RRM's single recurrent state costs X points at 16 iterations, beyond the seed floor: the separate slow state carries Y % of the gap between our attention model and SE-RRM's published number. TRM's single-state penalty carries over to our attention system, at a smaller size." Whether the cost runs through the readout-invisible state is the follow-up's question below, not this round's.
- **NO-STRUCTURE-EFFECT at 16.** "In our system, SE-RRM's single recurrent state matches our two states within the seed floor. TRM's 15-point single-state penalty does not appear here, and the state structure does not explain the gap." What remains unmoved: SE-RRM's random 5 % halting, dropout 0.2, AdamATan2, bfloat16, their block and embedding layout, their checkpoint selection and evaluation, and one run on each side. These are round-3 candidates.
- **At 64.** Read separately: the gap to SE-RRM there (98.22) is inside the floor already, so a 64-iteration difference speaks to the recurrence's long-horizon behavior, not to the SE-RRM gap.

Under every outcome:
- one seed;
- the 5,000-puzzle subsample, which reads within ±0.2 pp of the full test on five banked models;
- SE-RRM's number is their run on their evaluation;
- the factors not moved stay unattributed;
- R-A2-6 and R-A2-7 describe; a claim built on them needs its own registration.

**Follow-up, registered separately before it runs (Mac).** Run on SA256U's selected checkpoint: does the readout-invisible part of the one carry hold the information that rescues a stalled trajectory, as the slow state's does in the two-state models (P9's protocol)? This ties the accuracy reading to the manuscript's retained-state measurements under either outcome.

## Build and checks (before the launch)

- **Code.**
  - `Config.dec_single_state` (default False) and the single-state branch in `dec_cell.segment`.
  - The trainer flag `--dec-single-state`, argparse-suppressed when absent.
  - `RECORD_OMIT_AT_DEFAULT` in `tools/pretrain.py`: an unflagged run leaves the field out of config.json and the checkpoints. Its records therefore carry exactly the keys written before the field existed, which the frozen analyzers and `tools/resume_flags_guard.py` compare key by key. The evaluators rebuild the config with absent keys at their defaults.
- **Tests: `tests/test_single_state.py`, 6/6; `tests/test_final.py` and `tests/test_champ_builds.py` unchanged and passing.** They cover:
  - the default segment is BIT-EXACT against an independent replica of the pre-existing loop (MLP and attention mixers, with and without noise);
  - the single-state segment matches its own replica, leaves the fast slot untouched, applies the stack exactly H × (L + 1) = 21 times, and passes gradients through the last H-cycle only;
  - the records omit the field when it is off and carry True when it is on, and both round-trip through the evaluators' rebuild.
- **Trainer smokes (CPU).**
  - Width 32: 20 steps with finite loss and the flag in config.json.
  - SA256U's exact chain argv at width 256: 2 steps.
  - Its records against the banked SA256 config.json: the argv and the model config differ in exactly `dec_single_state` (beyond the smoke's own step, batch and cadence overrides).
  - SA256 run unflagged on the new code: they differ in nothing.
- **Evaluator smoke.** `tools/eval_sudoku_extreme.py` loads the SA256U checkpoint, rebuilds `dec_single_state = True` and records the per-iteration bits.
- **Harness `tools/harness_attr2.sh`: 17/17.** It covers:
  - one pod to completion: the fixed budget although the monitor rises to the end; the trainer argv token for token = `--out`, SA256's registry flags, `--dec-single-state`, `--steps 30000`; the reduced battery; the manifest;
  - an idempotent rerun, a preflight abort, and refusal of the reference and of an unknown arm;
  - **three mutants, all caught:** the arm without its flag (2 checks fail), a wrong explicit budget (3) and the arm out of `fixed_budget` (3).
- **Earlier suites, re-run after the edits:** `tools/harness_sablate.sh` 25/25, `tools/harness_attr1.sh` 28/28, `tools/analyze_sablate.py` 25/25 and `tools/analyze_attr1.py` 18/18.
- **Round 1 is unaffected.** Its pod runs the code archive banked at its supervisor's start; a bring-up that falls back to copying the working tree would still write round 1's records unchanged (above).

## Ops

- **The pod.** One spot v6e-8 named `qhrrn2-arc-pod`, launched only after round 1's pod is deleted and its fleet verified at zero. It writes under the fresh prefix `gs://qhrrn2-arc/rescue/attr2_p0`, which is verified empty before launch. The compile cache is seeded read-only from the ladder's prefix (SA256U's graph differs, so its programs compile fresh). WALL is 30,600 s: one run fits without a recycle; the live bank covers a preemption.
- **Safety and monitoring.**
  - The launchd watchdog stays armed.
  - Set `runs/tpu_deadline.txt` to launch + 10 h.
  - Plant the on-node guard.
  - Watch heartbeats with `tools/ops_watch_pods.sh tools/campaign_attr2_p0.env`.
- **Shared-project rules.** Only `qhrrn2-*` resources are created or deleted; nothing else in the project is touched.
- **The close.** Pull the final manifest; verify fleet zero at the source; stage SA256U with the ladder's SA256 dirs; run the analyzer FIRST, untouched; then the report, ledger and Outcome here.
