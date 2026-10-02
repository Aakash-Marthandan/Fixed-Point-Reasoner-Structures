# Note 2026-10-02 — SE-RRM attribution, round 1: SE-RRM's released optimizer and budget in our system (registration, before any row)

## Page one

**Goal.** A MEASUREMENT that locates a difference, not a better score.

**What it needs, and who approved it.**
- Pod time on ONE spot TPU v6e-8 in the ARC project (Mumbai, the PI's standing choice there), with the two arms run in sequence.
- Cap: $120. Expected spend: about $70.
- The PI: "commit everything and start pod round 1" (2026-10-02, about 16:45 IST). That came after reading the plan's round-1 proposal: `paper/documentation/Mac_Results_Digest_2026-10-01.md` §4a, which is private. The proposal states the arms, the cap and the expected accuracy below.
- Plan: `Plan_2026-10-01_Rebuttal_Experiments.md` §6 (G4 promoted). Session d378a5.

**Expected accuracy** on the identical 5,000 test puzzles at 16 / 64 iterations (one seed per arm; seeded floors 2.58 / 2.44 pp):

| arm | expected | for comparison |
|---|---|---|
| SA256B | 90–96 / 96–99 | the reference SA256 (the width ladder's banked row; 30k, selected 28k): 98.24 / 99.60 |
| SA256BR | 96–98.5 / 99.2–99.6 | SE-RRM's published single run (full test): 93.73 / 98.22 |

The earlier recipe ablation's arm at SE-RRM's *table* setting (batch 272, lr 5e-4, 30k) read 85.42 / 91.50 on the same 5,000.

## Why

Our attention model at hidden 256 contains SE-RRM's two mixers inside our block, loop and recipe, at about the same parameter count. It reads far above SE-RRM's published number. The recipe ablation (2026-09-19) has already tested three of the differences:
- damping and noise: inside the floor;
- randomized starts and anchors: inside the floor;
- SE-RRM's *table* optimizer (batch 272, lr 5e-4): far below.

SE-RRM's *released* Sudoku command, read at the source on 2026-10-01 (github.com/ml-jku/SE-RRM README, `pretrain.py`, `config/cfg_pretrain.yaml`), uses:
- batch 272;
- no `lr=` argument, so the config default 1e-4 applies, which equals ours;
- 10,000 epochs = 10,000 × 1,000 groups / 272 ≈ 36.8k steps ≈ 10M training rows.

Ours: 768 × 30,000 ≈ 23M rows at the reference budget.

The same source shows larger structural differences, kept for round 2:
- one recurrent state with no slow/fast split;
- no damping;
- random 5 % halting;
- dropout 0.2;
- AdamATan2;
- bfloat16.

Round 1 separates SE-RRM's *released optimizer and budget* from *batch size at equal training rows*. No model code changes.

## Arms

Each arm is `$(arm_flags SA256)` plus its override; argparse keeps the last value. The arms live in `tools/chain_champ.sh`, and `tools/chain_sablate.sh` admits them.

| arm | override | budget (fixed, never extended) | selection cadence |
|---|---|---|---|
| SA256B | `--batch 272` (lr 1e-4 and lr-end 1e-4 stay ours) | 36,000 steps ≈ 9.8M rows | monitor/grid every 2,000, as the reference: 18 grids |
| SA256BR | `--batch 272 --monitor-every 5600 --grid-every 5600` | 84,000 steps ≈ 22.8M rows, matched to the reference's 23.0M | every 5,600, which is rows-matched: 15 grids, as the reference |

Everything else is SA256 verbatim:
- seed 0, hidden 256, both attention mixers;
- 3 cycles of (6 fast + 1 slow) updates, two blocks, T 16;
- damping 0.05, noise 0.01, randomized starts σ 1, anchor rows k 1;
- AdamW wd 1.0 / β2 0.95, warm-up 2,000, constant lr, EMA 0.999;
- 1,000 positional copies; the 512-puzzle selection with the earliest tie;
- the reduced battery: D16 and D64 from the fixed start on the identical 5,000 test puzzles, plus the 5,000 × k32 restart scan.

The reference SA256 row is READ, never re-run.

## Rules (frozen in `tools/analyze_attr1.py`; selftest 18/18)

- **INTEGRITY.** Each arm's trainer argv differs from SA256's in exactly its registered keys:
  - SA256B: `batch`, `steps`;
  - SA256BR: `batch`, `steps`, `monitor_every`, `grid_every`.

  The model config differs in none. Seed 0; the arm's budget with no extension marker; its last training step equals its budget; rows n 5,000, EMA, the right depth, on SA256's puzzle ids; all rows and the scan on the selected grid.
- **R-AB-1.** Each arm vs SA256 at 16 and 64, with a paired exact McNemar test on the identical 5,000: INSIDE the floor (2.58 / 2.44 pp), BELOW-BEYOND or ABOVE-BEYOND. A p-value alone never moves a label.
- **R-AB-2.** Each arm vs SE-RRM's published 93.73 / 98.22: WITHIN, ABOVE or BELOW.
- **R-AB-3.** LOCATED at 16 / 64: the arms that read BELOW-BEYOND, with the share of the SA256-to-SE-RRM gap.
- **R-AB-4.** The selector from the k32 scan: CLEAN at a spurious rate ≤ 1 %, else DIRTY.
- **R-A1-5, the pair's reading at 16** (the question is posed at 16, where the gap to SE-RRM exceeds the floor):

  | reading | SA256B | SA256BR |
  |---|---|---|
  | ROWS | BELOW-BEYOND | INSIDE |
  | BATCH | BELOW-BEYOND | BELOW-BEYOND |
  | NEITHER | INSIDE | INSIDE |
  | OTHER | any other combination | |

- **STABILITY** (descriptive): the selected grid, flagged EDGE when it is the last grid; validation maximum and end; the scan's fixed start against one random start; verified accuracy; non-finite losses.

## Predictions and credences (before any row)

| prediction | credence |
|---|---|
| SA256B @16 BELOW-BEYOND | 0.60 |
| SA256B @16 INSIDE | 0.40 |
| SA256BR @16 INSIDE | 0.60 |
| SA256BR @16 BELOW-BEYOND | 0.35 |
| SA256BR @16 ABOVE-BEYOND | 0.05 |
| R-A1-5 ROWS | 0.40 |
| R-A1-5 NEITHER | 0.25 |
| R-A1-5 BATCH | 0.20 |
| R-A1-5 OTHER | 0.15 |
| Both arms INSIDE at 64 | 0.60 |
| Both selectors CLEAN | 0.85 |

## Wording under each outcome

- **ROWS.** "At SE-RRM's released batch size and learning rate, our model trained for SE-RRM's budget falls below our reference by more than the seed floor at 16 iterations. Given the reference's number of training rows, the same batch size recovers it: training length, not batch size, carries the difference in our system."
- **BATCH.** "Even at equal training rows, SE-RRM's batch size costs accuracy beyond the floor: the optimizer setting itself carries part of the difference."
- **NEITHER.** "Neither SE-RRM's released optimizer setting nor its training budget moves our model beyond the floor. The gap lies in what round 1 does not change: the single-state recurrence, halting, dropout, the optimizer variant, precision, or their side's one run." Round 2's single-state arm is then the main suspect.
- **OTHER.** Both letters reported, with the readings they support.

Under every outcome:
- one seed per arm;
- the 5,000-puzzle subsample, which reads within ±0.2 pp of the full test on five banked models;
- SE-RRM's number is their run on their evaluation;
- the factors above that are not ablated stay unattributed.

## Build and checks (before the launch)

- **Chain edits.** Two arm-table entries plus `arm_steps` and `fixed_budget` lines in `tools/chain_champ.sh`, and the admitted arms in `tools/chain_sablate.sh`.
- **The recipe ablation's checks are unchanged:** its harness `tools/harness_sablate.sh` 25/25 and its analyzer `tools/analyze_sablate.py` 25/25, both re-run after the edits.
- **The new harness, `tools/harness_attr1.sh`: 28/28.** It covers each arm alone; both arms in sequence on one pod (the launch configuration); idempotent rerun; preflight abort; and refusal of the reference and of an unknown arm.
- **Mutation checks: both caught.** SA256BR without its cadence override fails 2 checks; SA256B falling back to the default budget fails 4.
- **The new analyzer, `tools/analyze_attr1.py`: 18/18,** with hand-built records for every reading and eight integrity failures.

## Ops

- **The pod.** One spot v6e-8 named `qhrrn2-arc-pod`, under the fresh prefix `gs://qhrrn2-arc/rescue/attr1_p0`, which is verified empty before launch. The compile cache is seeded read-only from the ladder's prefix. `AB_ARMS="SA256B SA256BR"` runs in sequence. WALL is 30,600 s, so one resumable recycle is expected (live bank).
- **Safety and monitoring.**
  - Re-arm the launchd watchdog, paused since 2026-09-21.
  - Set `runs/tpu_deadline.txt` to launch + 14 h.
  - Plant the on-node guard.
  - Watch heartbeats with `tools/ops_watch_pods.sh`.
- **Shared-project rules.** Only `qhrrn2-*` resources are created or deleted; nothing else in the project is touched.
- **The close.** Pull the final manifest; verify fleet zero at the source; run the analyzer FIRST, untouched; then the report, ledger and Outcome here.
