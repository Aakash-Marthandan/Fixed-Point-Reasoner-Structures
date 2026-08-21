# Ledger: SPRINT S2 — materialize the benchmark file that ships to the pod:
# seeded random 1,000 training puzzles (+64 disjoint monitor rows) and the
# FULL test set from sapientinc/sudoku-extreme, as one compressed npz.
"""
  .venv/bin/python tools/prep_sudoku_extreme.py --seed 0 \
      [--train data/sudoku_extreme/train.csv --test data/sudoku_extreme/test.csv]
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qhrrn2 import sudoku_extreme as SX

ap = argparse.ArgumentParser()
ap.add_argument("--train", default="data/sudoku_extreme/train.csv")
ap.add_argument("--test", default="data/sudoku_extreme/test.csv")
ap.add_argument("--k", type=int, default=1000)
ap.add_argument("--n-val", type=int, default=64)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--test-limit", type=int, default=None)
ap.add_argument("--out", default=None)
a = ap.parse_args()
out = a.out or f"data/sudoku_extreme/sudoku_extreme_seed{a.seed}.npz"
meta = SX.prepare(a.train, a.test, out, k=a.k, n_val=a.n_val, seed=a.seed, test_limit=a.test_limit)
print("PREP-OK", out, meta)
