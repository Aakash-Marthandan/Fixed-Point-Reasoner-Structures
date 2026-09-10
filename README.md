# Fixed-Point Reasoner Structures

**Measured laws of recursive reasoning models, on Sudoku-Extreme and ARC.** This repository holds a measurement program: an instrument suite for recurrent solvers of the TRM / EqR class (basin existence, restart coverage, verification-free selection, commitment dynamics, calibration, memorization along training, depth, symmetry, compute), the laws it has measured on our cells and on the field's released weights through one evaluator, and the cell those laws produced — the **Decimating Equilibrium Cell (DEC)**, which leads the thousand-puzzle single-pass column on Sudoku-Extreme at every depth with 0.8–2.8 M parameters. The first phase (QHRRN-2, an information-priced renormalization-group architecture for ARC, 2026-07 → 08) is the lineage of the second paper; its physics background is [`README_PHYSICS.md`](README_PHYSICS.md).

Paper 1 (in preparation; freeze 2026-09-16): *Fixed Points, Basins, and the Price of Information: Measured Laws of Recursive Reasoning Models.* Paper 2: the DEC on ARC. Status below is as of **2026-09-10**; every number traces to an analysis script under `runs/analysis/` and a dated entry in the ledger. Nothing in this file is a claim the ledger does not carry.

**Read in this order:** this file → [`Documentation/README.md`](Documentation/README.md) (the reading order over 50 records) → [`Documentation/Design_Ledger.md`](Documentation/Design_Ledger.md) (the epistemic source of truth: hypotheses, registrations, verdicts, corrections, append-only) → the newest verdict report and the newest plan.

## Results at a glance (Sudoku-Extreme, the thousand-puzzle training convention)

Accuracy in %, the test set beside each number; "full" = all 422,786 test puzzles, 100k / 20k / 5k = uniform subsamples. Ours and the field's released weights are read through **one evaluator on identical puzzles**; the field's own paper numbers are in the comparison tables. D16 … D256 = outer steps at inference. TMAC = 10¹² multiply-accumulates per puzzle, measured by XLA's cost analysis of the evaluator's step map.

| system | params | D16, full | D64 | D128, 20k | D256, 5k | TMAC at D64 |
|---|---|---|---|---|---|---|
| **DEC-w192** (ours, one seed) | 0.79 M | **95.91** | **99.16** full | **99.42** | **99.60** | 0.95 |
| **DEC-w384**, seed triple (ours, mean ± half-spread) | 2.78 M | **95.02 ± 0.65** | **98.18 ± 0.61** 100k | **98.70 ± 0.43** | **98.97 ± 0.46** | 3.20 |
| DEC-w384 + set attention (ours, one seed) | 2.98 M | 96.70 | 98.90 100k | 99.15 | 99.32 | 3.40 |
| EqR, released weights | 5.04 M | 86.52 | 93.15 full | 95.02 | 95.88 | 0.82 |
| CGAR, released weights | 5.04 M | 86.10 | 91.71 full | 93.17 | 94.10 | 0.82 |
| TRM-MLP, released weights | 5.04 M | 79.32 | 83.45 full | 84.97 | 86.26 | 0.82 |

With restarts and **no verifier**, the DEC-w192's residual selector reaches 99.82 at k = 32 and 99.88 at k = 128 on the 5k scan set, equal to what a free verifier collects on the same draws; EqR's paper headline is 99.8 with B = 128 residual selection, PTRM's 98.75 with a learned Q-head over 100 rollouts. Solvers trained on the full 2.7–3.8 M-puzzle split (Sotaku 99.12, the diffusion curriculum 99.90) are a different regime and are not seated in this column. The seed triple is the claim-bearing row; the w192 row is one seed. Full tables with every protocol column: [`Documentation/Comparison_Frontier_Compute_Params_2026-09-09.md`](Documentation/Comparison_Frontier_Compute_Params_2026-09-09.md) (regenerated from artifacts by `tools/comparison_tables.py`).

## The cell

The DEC is the field's two-timescale recursive loop (TRM / EqR: a shared block applied H × L times per segment, deep supervision, adaptive halting, EMA weights, stablemax) placed on **our field-structured state**: nine digit fields over the 81 cells with every parameter shared across the field axis, so a permutation of the digits permutes the state and the logits exactly — the symmetry the field pays 1000× digit augmentation for is exact by construction. Two training objectives from the ARC program ride the loop: random-init rows (RI; the loss sees the model's own recurrent states from random starts) and fixed-point anchor rows (FPA; the loop is started at a corrupted solution and must return to it). The champion recipe: `--cell dec --dec-width 384`, `--fpa-k 1 --fpa-eps 0.2 --fpa-frac 0.25 --trm-ri-sigma 1.0`, no digit augmentation, position augmentation 1000, batch 768, lr 1e-4, weight decay 1.0, EMA .999, 16 segments with ACT, 30k steps, val-selected on a 512-puzzle train-file monitor with the earliest-tie rule. Design: `Documentation/Plan_2026-09-05_FinalPhase.md` §2 and `Plan_2026-09-08_Champion_Night.md`; code: `src/qhrrn2/dec_cell.py`, `src/qhrrn2/trm_cell.py`.

## The laws (each measured on our arms and on the field's released weights; the ledger entry and report are the record)

1. **The loop sets the decoder class.** Every TRM-class loop, ours or the field's, is a *decimating* decoder: 91–98 % of the free cells committed at confidence > .9 after one outer step, 43–74 % of them wrong on the puzzles it will fail; every native equilibrium map is *soft* (3–13 %). The operator and the width set the accuracy, the loop sets the class (50 grids).
2. **The calibrated-commitment gap is two-level.** A decimating decoder's confidence is a decision, not a probability (stalls at confidence 1.000 with half the committed cells wrong). A trained readout on the carried state recovers the *puzzle-level* signal at zero cost (ECE .013 vs the softmax's .501; per-puzzle correctness AUC .995 with no verifier) and no readout recovers the *cell-level* one (AUC .63 for "this committed cell is wrong").
3. **The selector law.** Training that exposes the loss to the model's own recurrent states (RI + anchor rows; the field's exposure-bias / fixed-point-forcing mechanism) makes the residual a verifier: spurious converged-wrong draws ≤ .2 % and residual-selected = verified to ≤ .06 pp at k = 128 on seven arms; without it the residual is worse than chance (AUC .375, 69 % spurious).
4. **The memorization / trajectory law.** Every arm trained on the thousand puzzles peaks and then declines on held-out puzzles while its training loss keeps falling; the clock is set by capacity (w512 ≈ 12k steps, w384 ≈ 28k, w192 > 50k) and not by the augmentation orbit, so selection along training is part of the result and the monitor's plateau end is the onset instrument.
5. **The depth law.** Accuracy rises with inference depth with zero regressions (422,786 puzzles at exact pairing, five arms): +3.3 pp D16 → D64, then +0.2–0.7 per doubling; the operator decides how much is settled per step and the width how much is left to propagate (the revision axis).
6. **Exact symmetry replaces the orbit.** The S9 state buys what 1000× digit augmentation buys the field (−65 pp without either), and a trained map is already position-invariant on unseen puzzles to the numerics floor, so the online position orbit changes neither the peak nor the memorization clock.
7. **Init-invariance is trainable and recipe-dependent.** RI makes a map's answer independent of its start (a random-init draw reads within .25 pp of the cold pass); the field's curriculum-trained cells and HRM are path-dependent; and a trained start can transiently leave the basin the map holds for random starts (the cold-start excursion instrument).

The ARC program's four laws (the information throat declines with capacity and is set by the task; codebook count and basin radius have separate controls; the priced transfer-radius plateau; pricing's dividend is transfer-specific) are in `Documentation/Report_2026-08-21_Reflective_Pass.md` and the rung reports.

## Sudoku and ARC through one suite

The same instruments read the two domains as different regimes (the paper's Sudoku-vs-ARC section; `Documentation/Note_2026-09-10_ARC_Instruments.md`, `Sudoku_vs_ARC_Instrument_Map.md`): on Sudoku the basin of the solution exists for every instance and the question is reaching it (retention 1.00; oracle over restarts ≥ 99.5 %); on ARC basins exist for a minority of task pairs (retention 19–28 %) and where they do not, no restart finds them (×55 conditioning). Depth propagates on Sudoku and does nothing on ARC, where the field's own ARC-1 model loses 8 of 419 inputs it had solved at step 1. The selector law's *mechanism* ports to ARC (RI cleans the residual) but its *value* does not (the same training condenses the restarts onto the cold endpoint); failures are churning stalls on Sudoku and frozen wrong fixed points on ARC, one mechanism with and without a restoring force. The field's ARC identity table carries 26 pp of its accuracy; the Sudoku prefix carries nothing. What is next on ARC is a PI decision: `Documentation/Plan_2026-09-10_DEC-ARC_Build.md`.

## Repository map

```
src/qhrrn2/              the implementation (JAX)
  dec_cell.py            the Decimating Equilibrium Cell (the field loop on the nine-field S9-exact state)
  trm_cell.py            the TRM / EqR loop port (two timescales, RI, segments, ACT, halting head)
  cell.py model.py       the original RG cell, encoder / rule codebook / decoder, the flux ledgers
  objective.py train.py  losses (CE, flux, FPA anchors), the training loop, data-parallel
  sudoku_extreme.py      Sudoku-Extreme data, the native9 layout, position / digit augmentation, the online orbit
  grid.py episodic.py    ARC canvases, D4 × palette transforms, episodes, the pretraining corpus
  rearc.py population.py RE-ARC families and the family holdout; population / vote machinery (ARC)
  config.py              every dial, ledger-annotated
tools/
  pretrain.py            the trainer (every arm of every campaign is one flag registry line)
  eval_sudoku_extreme.py the evaluator: cold rollouts at any depth, k-restart scans with residual / verified /
                         majority selection, screens, calibration, per-step records, the halting and commit heads
  select_ckpt.py         val-selection over the banked grids (earliest tie, second key)
  analyze_<tag>.py       the FROZEN analyzers (registered rules; --selftest on hand-built records)
  analyze_<tag>_physics.py, lens_*.py, suite_*.py   descriptive passes: dynamics, calibration, corpus lens,
                         the joint-readout lens, the orbit-invariance instrument, the ARC readers
  arc_suite.py           the ARC instrument suite (draws, retention, selectors, dynamics) on our ARC substrates
  mac_count.py comparison_tables.py   the measured compute column and the frontier tables
  chain_<tag>.sh harness_<tag>.sh pod.sh live_bank.sh plant_guard.sh   the pod chains, their offline harnesses,
                         the one-pod supervisor, the 5-min live bank, the node-side billing guard
  HANDOFF.md OPS_RUNBOOK.md   the ops record and runbook
tests/                   32 files (pytest; unit tests + the CI gates; exactness, bit-identity and resume scenarios)
Documentation/           the records (index: Documentation/README.md); paper/documentation/ the writing rules
runs/analysis/           every analyzer's output (tracked); runs/* the campaign artifacts (banked in GCS)
data/                    Sudoku-Extreme (CSV + npz), ARC-AGI, ConceptARC, RE-ARC (vendored, git-ignored)
```

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/pytest -q                                                    # unit tests + fast CI gates
```

Sudoku-Extreme: put the released `train.csv` / `test.csv` (the HRM / TRM Sudoku-Extreme split) under `data/sudoku_extreme/`, build the thousand-puzzle training file and the 512-puzzle monitor, train the champion recipe, evaluate.

```bash
.venv/bin/python tools/prep_sudoku_extreme.py --seed 0                 # data/sudoku_extreme/sudoku_extreme_seed0.npz
.venv/bin/python tools/sx_extend_monitor.py                             # the 512-puzzle monitor file (train + test byte-identical)
NPZ=data/sudoku_extreme/sudoku_extreme_seed0_mon512.npz
.venv/bin/python tools/pretrain.py --out runs/dec_w384_s0 --steps 30000 --seed 0 \
  --sudoku-extreme $NPZ --sudoku-layout native9 --equilibrium --sot --act \
  --trm-layers 2 --trm-h-cycles 3 --trm-l-cycles 6 --T 16 --trm-lambda 0.05 --trm-beta 0.01 \
  --sudoku-aug 1000 --loss stablemax --batch 768 --wd 1.0 --warmup 2000 --lr 1e-4 --lr-end 1e-4 --beta2 0.95 --ema 0.999 \
  --cell dec --dec-width 384 --fpa-k 1 --fpa-eps 0.2 --fpa-frac 0.25 --trm-ri-sigma 1.0 --beta-flux-nl 0 \
  --monitor-every 2000 --grid-every 2000 --ckpt-every 500 --val-every 100000 --dp
.venv/bin/python tools/eval_sudoku_extreme.py --ckpt runs/dec_w384_s0/ckpt_016000.pkl --npz $NPZ \
  --out runs/eval_d16 --split test --t-total 16 --ema                                        # the full test at D16
.venv/bin/python tools/eval_sudoku_extreme.py --ckpt runs/dec_w384_s0/ckpt_016000.pkl --npz $NPZ \
  --out runs/scan_k32 --split test --subsample 5000 --t-total 64 --k-init 32 --ema --batch 128  # k = 32 restarts, residual vs verifier
```

The champion night ran seven such arms on one spot v6e-16 for about a day (`tools/chain_champ.sh`; the registration `Documentation/Plan_2026-09-08_Champion_Night.md`); a w384 arm trains in ≈ 10 h on four v6e chips, the full-test D64 row takes ≈ 17 h on four chips. ARC: `git clone --depth 1 https://github.com/fchollet/ARC-AGI.git data/ARC-AGI`, ConceptARC and RE-ARC as in `src/qhrrn2/grid.py`; the ARC substrates and suite are `tools/chain_p13.sh` and `tools/arc_suite.py`.

## Research discipline

Registration before data: every campaign's decision rules are locked verbatim in a frozen analyzer with a selftest before the run, with numeric predictions and credences; the analyzer adjudicates byte-untouched against its registration commit; the physics pass and the lenses are descriptive and labeled; claim-bearing contrasts need three seeds or within-run pairing on identical puzzles; every cross-system number carries its protocol and training-regime columns; kills that fire are reported. The ledger (`Documentation/Design_Ledger.md`) is append-only: corrections are new entries, never edits. Several of this project's results began as kills — the priced ARC cell's collapse at small scale, the graft night whose four failures motivated the DEC, the champion night's orbit null.

## Provenance

The theory documents that seeded the project (Dec 2025 – Jan 2026) are catalogued in the ledger §1 with each claim's current status; the ARC program (2026-07 → 08-21) and its course corrections are the ledger's archived entries. GCP compute for the whole program ≈ $3.3k. Repository: https://github.com/Aakash-Marthandan/Fixed-Point-Reasoner-Structures (renamed from QHRRN; the old URL redirects).
