# Ledger: SPRINT S2 wave 2 (2026-08-22) — the GEN row's generator corpus, in
# the prepared-npz format the trainer already reads (--sudoku-extreme), so the
# GEN arm = generator pretrain (this file) -> 1k finetune (--init-from) with
# ZERO trainer changes. Puzzles: uniqueness-preserving hole punch targeting
# --givens (22 -> achieved 23-27, matching the benchmark's 23-28 p5-p95 givens;
# DIFFICULTY is NOT matched — random-punched puzzles are mostly rating-0-class,
# a labeled limitation of the GEN row). Seeded per puzzle (seed, i) so the file
# is reproducible regardless of worker mapping; disjoint from the benchmark
# test set by puzzle string (asserted).
"""
  .venv/bin/python tools/prep_sudoku_gen.py --n 16000 --givens 22 --seed 0 \
      [--bench data/sudoku_extreme/sudoku_extreme_seed0.npz] [--procs 8]
"""
from __future__ import annotations
import argparse, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qhrrn2 import sudoku as SU
from qhrrn2 import sudoku_extreme as SX


def one(args):
    seed, i, givens = args
    rng = np.random.default_rng([seed, i])
    p, s = SU.sample(rng, givens)
    return p.astype(np.int8), s.astype(np.int8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16000)
    ap.add_argument("--n-val", type=int, default=64)
    ap.add_argument("--givens", type=int, default=22)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--bench", default="data/sudoku_extreme/sudoku_extreme_seed0.npz")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"data/sudoku_extreme/sudoku_gen_g{a.givens}_seed{a.seed}.npz"
    t0 = time.time()
    with Pool(a.procs) as pool:
        pairs = pool.map(one, [(a.seed, i, a.givens) for i in range(a.n + a.n_val)], chunksize=64)
    q = np.stack([p for p, _ in pairs]); s = np.stack([s_ for _, s_ in pairs])
    strs = [SX.grid_to_str(g) for g in q]
    assert len(set(strs)) == len(strs), "duplicate generated puzzle"
    if Path(a.bench).exists():
        te = set(SX.grid_to_str(g) for g in SX.load_prepared(a.bench)["test_q"])
        assert not any(x in te for x in strs), "generated puzzle collides with the benchmark test set"
    giv = (q != 0).reshape(len(q), -1).sum(1)
    meta = dict(seed=a.seed, k=a.n, n_val=a.n_val, givens_target=a.givens,
                givens_achieved=[int(giv.min()), float(giv.mean()), int(giv.max())],
                generator="qhrrn2.sudoku punch", n_test=8)
    # tiny placeholder test split (8 generated puzzles) so every reader of the format works
    np.savez_compressed(out,
        train_q=q[:a.n], train_a=s[:a.n], train_rating=np.zeros(a.n, np.int32),
        train_row=np.arange(a.n, dtype=np.int64),
        val_q=q[a.n:], val_a=s[a.n:], val_rating=np.zeros(a.n_val, np.int32),
        test_q=q[:8], test_a=s[:8], test_rating=np.zeros(8, np.int32),
        test_source=np.zeros(8, np.int16), source_names=np.asarray(["gen"]),
        meta=np.asarray([repr(meta)]))
    print("PREP-GEN-OK", out, meta, f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
