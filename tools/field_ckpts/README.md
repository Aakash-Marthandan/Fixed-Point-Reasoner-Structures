# tools/field_ckpts — the public-checkpoint instrument study (2026-09-05/06)

Registration: `Documentation/Plan_2026-09-05_FieldCheckpoints_Instruments.md`; report: `Documentation/Report_2026-09-06_FieldCheckpoints_Instruments.md`;
knowledge base: `Documentation/Note_2026-09-05_Field_KnowledgeBase.md`. Data, checkpoints, venv and outputs live under `runs/field_ckpts/` (git-ignored):
`runs/field_ckpts/{venv (py3.12 + torch 2.14 + jax 0.10.2), src/{HRM,EqR,TRM_official,TRM_alphaxiv}, ckpts/{hrm_sudoku,eqr,trm_alphaxiv,trm_cgar}, out/<job>/, analysis/}`.
The scripts here are byte-identical copies of `runs/field_ckpts/harness/*` (they resolve paths relative to `runs/field_ckpts/`; run them from a copy there or set the paths).

- `field_models.py` — one loader per public checkpoint (their own model code, a flash-attn shim for CPU/Metal; EqR from the safely extracted npz); `step()` = one outer step.
- `extract_eqr_ckpt.py` — restricted-unpickler extraction of `eqr.pth` (allowlisted globals only; tensors rebuilt from the raw storages).
- `run_field.py` — modes cold | draws | dyn | retain | sym | prefix | jac | train | initrad; records in the evaluator's format.
- `analyze_field.py` — the lens tables (E1/E2/E3/E4/E5/E6, ladder, halting head, retention, Jacobians, prefix, symmetry, memorization, cross-cell overlaps) → `runs/field_ckpts/analysis/field_ckpts_<date>.{txt,json}`.
- `verify_port.py` / `verify_port_ctrl.py` — the JAX `trm_cell` port check on the public TRM weights and its fp64 control.
- `night_*.sh` — the queue drivers (Metal / CPU); every job skips itself when its `done.json` exists.
