# Fixed-Point Reasoner Structures — a research archive on recursive reasoning models

This repository is the research record of a program on recursive reasoning models: small shared networks applied repeatedly to an evolving state from which an answer is read, studied on Sudoku-Extreme and on ARC-AGI. It holds the implementation (JAX), the trainer, the evaluator and the diagnostic lenses, every campaign's registration, frozen analyzer and verdict report, the append-only design ledger, and the analyzers' outputs. The manuscript derived from these records, its frozen account and its code-and-evidence package live in a local `paper/` tree that is not tracked here (the PI's decision of 2026-09-11); this file does not restate its claims or numbers.

The repository name dates from an earlier phase of the program and is kept for continuity; the ledger's status log records every registration, verdict and correction along the way. Nothing in this file is a claim the ledger does not carry.

**Read in this order:** this file → [`Documentation/README.md`](Documentation/README.md) (the index of the records, by phase and status) → [`Documentation/Design_Ledger.md`](Documentation/Design_Ledger.md) §5 (the append-only status log: registrations, verdicts, corrections) → the newest verdict report.

## 1. The program

Two phases share the code and the discipline.

- **The ARC program (2026-07 → 08-21)** built an RG-inspired architecture for ARC-AGI (`src/qhrrn2/cell.py`, `model.py`; the physics background in [`README_PHYSICS.md`](README_PHYSICS.md)), a seeded scale grid, and a ConceptARC assay suite (retention of a supplied answer, discovery from random starts). Its records are the archived ledger entries and the reports indexed in `Documentation/README.md` §5.
- **The Sudoku campaign (2026-08-21 → 09-21)** built a digit-field cell for Sudoku-Extreme (nine digit fields over the 81 cells, one feature vector per candidate digit and cell, parameters shared across fields so that relabeling the digits permutes the state and the scores exactly; position mixing by an MLP or by attention, cross-field mixing by a projected mean or by attention, and a TRM-style two-timescale recurrence), trained it under the thousand-puzzle convention, and measured what the iterations do: full-test accuracy at fixed depths and restart counts, per-iteration trajectory records, matched-error repair of model-produced against random corruptions, communication and state-reset interventions from a shared first state, an initialization series across the checkpoints of one training run, and nested restart banks with residual and verifier selection. The released Sudoku models (TRM, CGAR, EqR) run through one JAX evaluator, a released ARC-AGI-1 TRM natively, and released FPRM weights as a qualified diagnostic.

## 2. Where the results are

Every number traces to an analyzer output under `runs/analysis/` and to a ledger entry. The verdict reports are indexed in [`Documentation/README.md`](Documentation/README.md): §2 for the benchmark and intervention runs of 2026-09-13 → 09-21, §4 for the campaign that built the cell. The ledger's entries of 2026-09-19 and 2026-09-24 are the correction record that scopes the earlier summaries. Numbers are deliberately not restated in this file.

## 3. The models and the recipe

The benchmark digit-field solvers are MLP 192 (three training seeds; internal names C5, C7, C8) and Attention 128, 192 and 256 (one training seed each; SA128, SA192, SA256). Each block mixes positions within a field (a position MLP, or attention with two-dimensional rotary positions), exchanges information between candidate digits at each cell (a projected mean, or cross-field attention), and transforms channels. The recurrence follows TRM: three cycles of six fast-state updates and one slow-state update per outer iteration, 21 applications of a shared two-block network, both states carried between iterations; a shared linear readout of the slow state gives the digit scores.

Shared recipe: 1,000 base puzzles with 1,000 positional augmentations each and no digit augmentation; AdamW, batch 768, linear warmup over 2,000 updates to 1e-4 then constant, weight decay 1.0, β₂ = 0.95, gradient-norm clipping 1.0, EMA 0.999; 50,000 optimizer updates (the attention runs were extended from an initial 30,000); StableMax loss on all 81 cells after every outer iteration with gradients truncated between iterations and through the final inner cycle only; damping 0.05 and training noise 0.01; a halting head (weight 0.5) with the 16-iteration cap; randomized Gaussian training starts; and an answer-anchor branch on one quarter of the batch that embeds a randomly corrupted solution (per-cell redraw probability drawn from [0, 0.2)) and supervises one iteration against the solution. Checkpoints are ranked by depth-16 EMA accuracy on the 512-puzzle monitor, then non-EMA accuracy, then the earliest step. Code: `src/qhrrn2/dec_cell.py` (the cell), `src/qhrrn2/trm_cell.py` (the two-timescale loop), `tools/pretrain.py` (the trainer), `tools/eval_sudoku_extreme.py` (the evaluator), `tools/select_ckpt.py` (the selection rule).

Other models in the record: our TRM implementation (X0, width 512, 50k; the nested-pool study), its recipe variants (anchors only, randomized starts only, a second baseline seed), the earlier MLP 384 (C0, selected at 16k within a 30k budget), the earlier Attention 256 recipe study (30k budget, reference at 28k), the seed-0 MLP 192 lineage continued to 150k (the initialization series), the TRM-cell control (X5, 960k), and the released Sudoku ports of TRM, CGAR and EqR.

## 4. Repository map

```
src/qhrrn2/              the implementation (JAX)
  dec_cell.py            the digit-field cell: nine shared fields, MLP or attention mixers, cross-field messages
  trm_cell.py            the TRM / EqR two-timescale loop, randomized starts, segments, the halting head
  decarc_cell.py         the ARC variant of the cell from the 2026-09-15/16 night (excluded from the manuscript)
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
  lens_*.py              the descriptive lenses: lens_repair_radius (matched-error repair),
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

## 5. Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/pytest -q                                                    # unit tests + fast CI gates
```

Sudoku-Extreme: put the released `train.csv` / `test.csv` under `data/sudoku_extreme/`, build the seed-0 split and the 512-puzzle monitor, train one benchmark recipe, select and evaluate.

```bash
.venv/bin/python tools/prep_sudoku_extreme.py --seed 0                 # data/sudoku_extreme/sudoku_extreme_seed0.npz
.venv/bin/python tools/sx_extend_monitor.py                             # the 512-puzzle monitor (train + test byte-identical)
NPZ=data/sudoku_extreme/sudoku_extreme_seed0_mon512.npz
# Attention 128 (the benchmark recipe; drop --dec-token-mixer attn and set --dec-width 192 for MLP 192)
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

The attention runs were trained on spot v6e-8 pods: 30k updates in the width-ladder night, then the extension to 50k and the full evaluation battery in about 16 h per pod (`tools/chain_wladder.sh`, `tools/chain_saext.sh`); the chains shard the full test. ARC: `git clone --depth 1 https://github.com/fchollet/ARC-AGI.git data/ARC-AGI`; the released ARC models and the ConceptARC assays are driven by the code package's adapters and `tools/arc_suite.py`.

## 6. Research discipline

Registration before data: every campaign's decision rules are locked verbatim in a frozen analyzer with a selftest before the run, with numeric predictions and credences; the analyzer adjudicates byte-untouched against its registration commit; lenses and physics passes are descriptive and labeled; claim-bearing contrasts need three seeds or within-run pairing on identical puzzles; every cross-system number carries its protocol and training-regime columns; kills that fire are reported. The ledger (`Documentation/Design_Ledger.md`) is append-only: corrections are new entries, never edits, and a superseded record keeps its text under a dated status banner. Since 2026-09-19 the manuscript additionally follows an admission rule: a statement enters it only with its sample, checkpoint, protocol and qualification, after an independent recount of the saved records; "measured on the recorded protocol" is never a proof or a license to generalize.

## 7. Provenance

The theory documents that seeded the project (December 2025 – January 2026) are catalogued in the ledger §1 with each claim's current status; the ARC program (2026-07 → 08-21) and its course corrections are the ledger's archived entries; the Sudoku campaign (2026-08-21 → 09-10), the manuscript's evidence runs (2026-09-13 → 09-21) and the audit are indexed in `Documentation/README.md`. GCP compute: about $3.3k through 2026-09-10, about $140 for the extensions of 2026-09-14/15, and $969 in the shared ARC project from 2026-09-15 to 09-21 (`tools/HANDOFF.md`), about $4.4k in all. Repository: https://github.com/Aakash-Marthandan/Fixed-Point-Reasoner-Structures (renamed from QHRRN; the old URL redirects).
