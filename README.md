# Fixed-Point Reasoner Structures — the research archive

This repository is the research record behind the manuscript (in final revision as of **2026-09-25**). It holds the implementation (JAX), the trainer, the evaluator and the diagnostic lenses, every campaign's registration, frozen analyzer and verdict report, the append-only design ledger, and the analyzers' outputs. The self-contained, anonymized code-and-evidence package that accompanies the paper (six selected checkpoints, per-puzzle records, recount scripts) is prepared separately and is not tracked here: the whole `paper/` tree is local-only by the PI's decision of 2026-09-11.

The repository name, *Fixed-Point-Reasoner-Structures*, dates from an earlier phase of the program and is kept for continuity; the ledger's status log records every registration, verdict and correction along the way. Nothing in this file is a claim the ledger does not carry.

**Read in this order:** this file → [`Documentation/README.md`](Documentation/README.md) (the index of the records, by phase and status) → [`Documentation/Design_Ledger.md`](Documentation/Design_Ledger.md) §5 (the append-only status log: registrations, verdicts, corrections) → the newest verdict report.

## 1. The question and the setting

Recursive reasoning models (HRM, TRM, EqR, SE-RRM) repeatedly apply a small shared network to an evolving state from which an answer is read. The paper asks **what successive iterations accomplish, what makes correction possible, and when additional computation improves the returned answer**. It answers with trajectory measurements and paired interventions on Sudoku-Extreme (1,000 training puzzles; all 422,786 test puzzles), on released models (TRM, CGAR and EqR through one JAX evaluator; a released ARC-AGI-1 TRM natively; released FPRM weights as a qualified diagnostic) and on our compact digit-field solvers. The explanatory account is **correction behaviour measured by interventions on the learned update**: earlier updates change the context for later revisions; retained state supplies history, cross-field communication brings information from other representations into each update, training shapes these responses, initialization affects which trajectories are entered, and selection decides which candidates become returned answers.

## 2. Results at a glance (Sudoku-Extreme, 1,000 base training puzzles, all 422,786 test puzzles)

One fixed-start trajectory per puzzle, EMA weights selected by depth-16 accuracy on a separate 512-puzzle validation monitor, no voting and no early halting. D = outer iterations. Dense MACs count projections, attention products and readouts from the weight shapes (not runtime, not training cost); SE-RRM's count is from its official Sudoku configuration, its accuracy is the published value.

| Model | Params | D16 | D64 | GMACs / iteration | TMACs / puzzle at D64 |
|---|---:|---:|---:|---:|---:|
| Attention 128 (one seed) | 0.62 M | 95.45 | **99.48** | 10.24 | 0.655 |
| Attention 192 (one seed) | 1.06 M | 97.31 | **99.62** | 17.22 | 1.102 |
| Attention 256 (one seed) | 1.97 M | 97.93 | **99.62** | 31.47 | 2.014 |
| MLP 192, three seeds (mean; 95.91 / 94.82 / 95.49 and 99.16 / 98.82 / 99.18) | 0.79 M | 95.41 | 99.05 | 14.67 | 0.939 |
| SE-RRM RoPE2D, published (arXiv 2603.02193, Table 2) | 1.97 M | 93.73 | 98.22 | 34.65 | 2.218 |

- All six runs exceed SE-RRM RoPE2D's published 98.22 % at 64 iterations with fewer dense MACs. At 16 iterations SE-RRM's separate RoPE variant reports 95.4 % (its Table A6), level with Attention 128. This is a named comparison at stated operating points, not a state-of-the-art claim: Flow Reasoning Models report 99.5 % with about 7 M parameters on a fixed 1,000-puzzle test subset under an inference sweep; FPRM reports 94.2 % (about 7 M) and TRM-MLP 87.4 % (about 5 M) under their own protocols.
- Continued depth on a common 50,000-puzzle subset: Attention 128 / 192 / 256 solve 49,713 / 49,789 / 49,805 at D64, 49,877 / 49,894 / 49,882 at D128 and 49,943 / 49,936 / 49,914 at D256; the smallest model leads at 256.
- Through depth 64 the attention models lose correctness temporarily on 0 / 4 / 7 puzzles and every puzzle solved at least once ends correct. This is finite preservation, not a fixed point.
- Released weights through the common evaluator on the same 422,786 puzzles at D64: EqR 393,835 (93.15 %), CGAR 387,718 (91.71 %), alphaXiv TRM 352,809 (83.45 %). Conversion checks disagree with native execution on 3 / 2 / 2 of 64 puzzles, so these are evaluator-specific observations, not published scores.
- Attribution controls: a TRM-cell control without digit fields at MLP 192's parameter count (795,909 vs 789,125) reaches 60.26 % at D16 (full test) and 62.40 % at D64 (100,000 puzzles) after 960k updates under its tested recipe; in the earlier Attention 256 study, changing batch size and learning rate together costs 12.82 / 8.10 points at D16 / D64, the largest tested effect. Neither control isolates the source of the advantage over SE-RRM.

## 3. What the experiments establish

Each item names its model and population; the ledger entry and the report are the record.

1. **Correction before completion.** Among the 287 MLP 192 puzzles first solved at iterations 3–32 (512 uniformly sampled puzzles), mean empty-cell error falls 16.82 points before the completing transition, 13.02 of them in the last transition between incorrect predictions; earlier revisions number 26.94 wrong-to-right and 23.14 right-to-wrong events per 100 empty cells, so substantial revision produces little net progress. Released EqR and TRM and the attention models show the same concentration within their own cohorts.
2. **Repair depends on the error configuration, not only its count.** On 438 puzzle pairs with equal numbers of wrong cells, MLP 192 repairs 414 EqR-produced grids versus 432 random corruptions (+4.11 points, 95 % interval 2.05–6.39); released EqR repairs 373 versus 430 (+13.01, 10.05–16.21). The attention models show the strongest contrast in early repair (35–37 % versus 3–5 % error after one iteration) with small endpoint gaps (four-draw means 4.05 / 2.45 / 1.14 points; three Attention 256 intervals include zero). Only 0.17 % of EqR's wrong digits are contradicted by a given versus 62.93 % of random ones; this accompanies the contrast without isolating its cause.
3. **Continued communication and retained state.** Removing cross-field messages after an identical first state prevents every new discovery in MLP 192, MLP 384 and all three attention models, and loses every initial attention solution by the endpoint (250 intact per model on 256 puzzles). Resetting the slow state before iterations 2–16 leaves 58 / 101 / 126 correct where fast resets leave 246 / 245 / 246; the original four-checkpoint study reads 34.77–80.86 points for slow resets against 2.34–3.52 for fast resets. Both states keep all their within-iteration computation.
4. **Initialization can hide capability.** At the 94k checkpoint of the seed-0 MLP 192 lineage (beyond the 50k benchmark budget), the fixed start solves 11 of 128 validation-pool puzzles, one independent Gaussian start per puzzle solves 124, and one Gaussian start shared across all puzzles also solves 124, with no loss after success; the fixed start reaches 68 and loses 57. At the selected 46k checkpoint the three starts read 126 / 124 / 127; the attention models' fixed starts solve 126 / 126 / 128.
5. **More candidates can mean fewer correct returned answers.** In our TRM implementation (seed 0, 50k, 20,000 puzzles, D64), expanding the pool from 8 to 128 Gaussian starts raises coverage from 19,492 to 19,941 while minimum-residual selection falls from 19,099 to 18,496 (602 gains, 1,205 losses); every loss selects a new wrong candidate with a strictly smaller residual while the earlier correct candidate remains available. Training separates generation from ranking: the randomized-start arm returns 19,659 of 19,663 covered answers, the anchors-only arm 7,645 of 19,972. The benchmark attention models miss 0 / 1 / 3 covered answers at 128 starts on 5,000 puzzles.
6. **ARC-AGI as the boundary.** The released alphaXiv TRM (419 queries, 400 tasks, evaluation-task demonstrations seen in training) reproduces 1,295 of 1,363 demonstrations yet solves 121 of the 367 queries whose demonstrations are all reproduced. No query first becomes correct after iteration 3 through D64 (12 gains, 8 losses, 10 of 138 ever-correct queries end wrong); eight Gaussian restarts raise coverage from 134 to 136 while selection falls from 134 to 131; the union of all observed candidates covers 146 of 419 queries, leaving 273 without a correct candidate. Released FPRM weights, under a reconstructed and uncertified task-ID mapping, recover earlier answers with depth (12 / 18 / 16 correct of 40 at solver steps 16 / 32 / 64, 20 ever correct) and lose demonstration fit from 140 to 13 of 150 when task IDs are reassigned.
7. **Conjecture, kept separate from the findings:** the manuscript's conjecture — mutually supporting errors may require coordinated revision, which retained context and communication make possible. The paper states this as a hypothesis with its discriminating tests, not as an established mechanism.

## 4. The models and the recipe

Digit-field solvers: nine digit fields over the 81 cells, one feature vector per candidate digit and cell, parameters shared across fields so that relabeling the digits permutes the state and the scores exactly. Each block mixes positions within a field (a position MLP, or attention with two-dimensional rotary positions), exchanges information between candidate digits at each cell (a projected mean, or cross-field attention), and transforms channels. The recurrence follows TRM: three cycles of six fast-state updates and one slow-state update per outer iteration, 21 applications of a shared two-block network, both states carried between iterations; a shared linear readout of the slow state gives the digit scores.

| Model | Parameters | Training seeds | Selected update | Internal names |
|---|---:|---|---|---|
| MLP 192 | 789,125 | 0, 1, 2 | 46,000 each | C5, C7, C8 |
| Attention 128 | 623,365 | 0 | 42,000 | SA128 |
| Attention 192 | 1,057,925 | 0 | 40,000 | SA192 |
| Attention 256 | 1,967,621 | 0 | 46,000 | SA256 |

Shared recipe: 1,000 base puzzles with 1,000 positional augmentations each and no digit augmentation; AdamW, batch 768, linear warmup over 2,000 updates to 1e-4 then constant, weight decay 1.0, β₂ = 0.95, gradient-norm clipping 1.0, EMA 0.999; 50,000 optimizer updates (the attention runs were extended from an initial 30,000); StableMax loss on all 81 cells after every outer iteration with gradients truncated between iterations and through the final inner cycle only; damping 0.05 and training noise 0.01; a halting head (weight 0.5) with the 16-iteration cap; randomized Gaussian training starts; and an answer-anchor branch on one quarter of the batch that embeds a randomly corrupted solution (per-cell redraw probability drawn from [0, 0.2)) and supervises one iteration against the solution. Checkpoints are ranked by depth-16 EMA accuracy on the 512-puzzle monitor, then non-EMA accuracy, then the earliest step. Code: `src/qhrrn2/dec_cell.py` (the cell), `src/qhrrn2/trm_cell.py` (the two-timescale loop), `tools/pretrain.py` (the trainer), `tools/eval_sudoku_extreme.py` (the evaluator), `tools/select_ckpt.py` (the selection rule).

Other models in the record: our TRM implementation (X0, width 512, 50k; the nested-pool study), its recipe variants (anchors only, randomized starts only, a second baseline seed), the earlier MLP 384 (C0, selected at 16k within a 30k budget), the earlier Attention 256 recipe study (30k budget, reference at 28k), the seed-0 MLP 192 lineage continued to 150k (the initialization series), the TRM-cell control (X5, 960k), and the released Sudoku ports of TRM, CGAR and EqR.

## 5. Repository map

```
src/qhrrn2/              the implementation (JAX)
  dec_cell.py            the digit-field cell: nine shared fields, MLP or attention mixers, cross-field messages
  trm_cell.py            the TRM / EqR two-timescale loop, randomized starts, segments, the halting head
  decarc_cell.py         the ARC variant of the cell from the 2026-09-15/16 night (excluded from the paper)
  cell.py model.py       the original RG cell of the ARC program (2026-07 → 08; the lineage of a second paper)
  objective.py train.py  the losses (StableMax, halting, answer anchors), the training loop, data-parallel
  sudoku_extreme.py      Sudoku-Extreme data, the native9 layout, positional augmentation
  grid.py episodic.py rearc.py population.py   ARC canvases, transforms, episodes, RE-ARC (the ARC program)
  config.py              every dial, ledger-annotated
tools/
  pretrain.py            the trainer (every arm of every campaign is one flag line; configs saved per run)
  eval_sudoku_extreme.py the evaluator: fixed-start rollouts at any depth, per-iteration records,
                         k-restart banks with residual and verifier selection, the sharded full test
  select_ckpt.py         checkpoint selection over the banked monitor grids (earliest-tie rule)
  sx_*.py prep_*.py      data preparation: the seed-0 split, the 512-puzzle monitor, the 10k validation pool
  analyze_<tag>.py       the FROZEN analyzers (registered rules; --selftest on hand-built records)
  lens_*.py              the descriptive lenses behind the paper: lens_repair_radius (matched-error repair),
                         lens_fast_slow (state resets), lens_corpus_normalized (the trajectory corpus),
                         lens_c5l_dynamics (the initialization series), lens_commit_validity (the
                         normalization check), lens_first_passage, lens_finalA_reselect
  attention_transfer_study.py, analyze_attention_transfer*.py   the final-attention diagnostics
                         (repair, communication and reset interventions, initialization; 2026-09-23)
  stall_calibration.py   confidence under the training normalization (tests/test_stall_calibration.py)
  arc_suite.py probe_e1e3.py   the ConceptARC assays (retention, discovery) on the earlier ARC models
  mac_count.py comparison_tables.py   the 2026-09-10 compute measure and frontier tables (historical)
  chain_<tag>.sh harness_<tag>.sh campaign_<tag>.env pod.sh live_bank.sh plant_guard.sh
                         the pod chains, their offline harnesses, the one-pod supervisor, the guards
  HANDOFF.md OPS_RUNBOOK.md   the ops record and runbook (the watchdogs are paused since 2026-09-21)
tests/                   35 files (pytest: unit tests, CI gates, exactness and resume scenarios)
Documentation/           the records; index in Documentation/README.md
runs/analysis/           every analyzer's output (tracked); runs/* the campaign artifacts (banked in GCS)
data/                    Sudoku-Extreme, ARC-AGI, ConceptARC, RE-ARC (vendored, git-ignored)
paper/                   the manuscript, its frozen account and the code package (local only, git-ignored)
```

## 6. Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/pytest -q                                                    # unit tests + fast CI gates
```

Sudoku-Extreme: put the released `train.csv` / `test.csv` under `data/sudoku_extreme/`, build the seed-0 split and the 512-puzzle monitor, train one benchmark recipe, select and evaluate.

```bash
.venv/bin/python tools/prep_sudoku_extreme.py --seed 0                 # data/sudoku_extreme/sudoku_extreme_seed0.npz
.venv/bin/python tools/sx_extend_monitor.py                             # the 512-puzzle monitor (train + test byte-identical)
NPZ=data/sudoku_extreme/sudoku_extreme_seed0_mon512.npz
# Attention 128 (the paper's recipe; drop --dec-token-mixer attn and set --dec-width 192 for MLP 192)
.venv/bin/python tools/pretrain.py --out runs/attention_128_s0 --steps 50000 --seed 0 \
  --sudoku-extreme $NPZ --sudoku-layout native9 --equilibrium --sot --act \
  --trm-layers 2 --trm-h-cycles 3 --trm-l-cycles 6 --T 16 --trm-lambda 0.05 --trm-beta 0.01 --halt-explore 0.1 \
  --sudoku-aug 1000 --loss stablemax --batch 768 --wd 1.0 --warmup 2000 --lr 1e-4 --lr-end 1e-4 --beta2 0.95 --ema 0.999 \
  --cell dec --dec-width 128 --dec-token-mixer attn --fpa-k 1 --fpa-eps 0.2 --fpa-frac 0.25 --fpa-w 1.0 --trm-ri-sigma 1.0 \
  --beta-flux-nl 0 --monitor-every 2000 --grid-every 2000 --ckpt-every 500 --dp
.venv/bin/python tools/select_ckpt.py runs/attention_128_s0 --key val_t16_ema --second-key val_t16 --tie earliest
.venv/bin/python tools/eval_sudoku_extreme.py --ckpt runs/attention_128_s0/ckpt_042000.pkl --npz $NPZ \
  --out runs/eval_d64 --split test --t-total 64 --ema --record-by-step                        # the full test at D64
.venv/bin/python tools/eval_sudoku_extreme.py --ckpt runs/attention_128_s0/ckpt_042000.pkl --npz $NPZ \
  --out runs/scan_k128 --split test --subsample 5000 --t-total 64 --k-init 128 --ema --batch 128   # 128 restarts, residual vs verifier
```

The attention runs were trained on spot v6e-8 pods: 30k updates in the width-ladder night, then the extension to 50k and the full evaluation battery in about 16 h per pod (`tools/chain_wladder.sh`, `tools/chain_saext.sh`); the chains shard the full test. ARC: `git clone --depth 1 https://github.com/fchollet/ARC-AGI.git data/ARC-AGI`; the released ARC models and the ConceptARC assays are driven by the paper package's adapters and `tools/arc_suite.py`.

## 7. Research discipline

Registration before data: every campaign's decision rules are locked verbatim in a frozen analyzer with a selftest before the run, with numeric predictions and credences; the analyzer adjudicates byte-untouched against its registration commit; lenses and physics passes are descriptive and labeled; claim-bearing contrasts need three seeds or within-run pairing on identical puzzles; every cross-system number carries its protocol and training-regime columns; kills that fire are reported. The ledger (`Documentation/Design_Ledger.md`) is append-only: corrections are new entries, never edits, and a superseded record keeps its text under a dated status banner. Since 2026-09-19 the paper additionally follows an admission rule: a statement enters the manuscript only with its sample, checkpoint, protocol and qualification, after an independent recount of the saved records; "measured on the recorded protocol" is never a proof or a license to generalize.

## 8. Provenance

The theory documents that seeded the project (December 2025 – January 2026) are catalogued in the ledger §1 with each claim's current status; the ARC program (2026-07 → 08-21) and its course corrections are the ledger's archived entries; the Sudoku campaign (2026-08-21 → 09-10), the paper's evidence runs (2026-09-13 → 09-21) and the audit are indexed in `Documentation/README.md`. GCP compute: about $3.3k through 2026-09-10, about $140 for the extensions of 2026-09-14/15, and $969 in the shared ARC project from 2026-09-15 to 09-21 (`tools/HANDOFF.md`), about $4.4k in all. Repository: https://github.com/Aakash-Marthandan/Fixed-Point-Reasoner-Structures (renamed from QHRRN; the old URL redirects).
