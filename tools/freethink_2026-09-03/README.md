# Freethink 2026-09-03 grounding scripts (analysis-time, descriptive, $0 Mac CPU/disk)
Run from the repo root with `PYTHONPATH=src JAX_PLATFORMS=cpu .venv/bin/python tools/freethink_2026-09-03/<script> [n]`. Artifacts land in `runs/analysis/freethink_ground*_20260903.*`.
- ground1_reachability.py — reachability/cold vs givens and source; the unreachable set; draw diversity (1e appended by an inline snippet, see the artifact).
- ground2_trajectories.py — cells-correct and readout entropy per outer step (strat-84), solved vs unsolved.
- ground2b_flips.py — revision dynamics (argmax flips to correct/wrong; step-1 guesses later corrected).
- ground3_fisher.py — curvature concentration from the Adam second moment across regimes.
- ground4_levers.py — inference-time levers: depth to 256, inner latent cycles, readout temperature, kicked restarts.
- ground7_decimation.py — verified decimation with backtracking on stalled puzzles + the confidence-calibration diagnostic.
Grounds 5/6 (what memorization erases; propagation depth vs erasures) were inline record snippets; their artifact is `freethink_ground5_6_20260903.txt`.
