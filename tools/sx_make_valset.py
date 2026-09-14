# Ledger: THE C8 EXTENSION (2026-09-14; Documentation/Plan_2026-09-14_C8_Extension.md). A LOW-NOISE SELECTION SET: the 512-puzzle
# monitor resolves ~1 pp and decided C8's extension by one puzzle; this draws VALSET_N further puzzles (seed 20260914) from the TRAIN
# FILE outside the seeded 1k training rows AND all 512 monitor rows, verified string-disjoint from the training subsample, the monitor
# and the full test set. The output npz copies every array of the monitor file byte-identical EXCEPT val_q/val_a/val_rating/val_row,
# which hold ONLY the new set (independent of the monitor that made the registered selection). Never used by the trainer.
"""  .venv/bin/python tools/sx_make_valset.py --npz data/sudoku_extreme/sudoku_extreme_seed0_mon512.npz \
        --train-csv data/sudoku_extreme/train.csv --n 10000 --seed 20260914 --out data/sudoku_extreme/sudoku_extreme_seed0_val10k.npz"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qhrrn2 import sudoku_extreme as SX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True); ap.add_argument("--train-csv", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=10000); ap.add_argument("--seed", type=int, default=20260914)
    a = ap.parse_args()
    t0 = time.time()
    z = np.load(a.npz, allow_pickle=False)
    base = {k: z[k] for k in z.files}
    meta = eval(str(base["meta"][0]))
    n_train = int(meta["n_train_file"])
    used = np.union1d(base["train_row"].astype(np.int64), base["val_row"].astype(np.int64))   # the 1k training rows + all 512 monitor rows
    assert len(used) == len(base["train_row"]) + len(base["val_row"]), "training and monitor rows overlap"
    rng = np.random.default_rng(a.seed)
    pool = np.setdiff1d(np.arange(n_train), used, assume_unique=False)
    new_idx = np.sort(rng.choice(pool, size=a.n, replace=False))
    assert not np.intersect1d(new_idx, used).size
    rows = {}
    for i, _, q, ans, r in SX.read_rows(a.train_csv, indices=set(int(i) for i in new_idx)):
        rows[int(i)] = (q, ans, r)
    assert len(rows) == a.n, (len(rows), a.n)
    q_new = np.stack([rows[int(i)][0] for i in new_idx]).astype(np.int8)
    a_new = np.stack([rows[int(i)][1] for i in new_idx]).astype(np.int8)
    r_new = np.asarray([rows[int(i)][2] for i in new_idx], np.int32)
    s_new = {SX.grid_to_str(g) for g in q_new}
    n_unique = len(s_new)
    for name in ("train_q", "val_q", "test_q"):
        hits = sum(1 for g in base[name] if SX.grid_to_str(g) in s_new)
        assert hits == 0, f"{hits} validation puzzles also appear in {name}"
    # every answer is a valid solution consistent with its givens (the evaluator's exact check relies on it)
    ok = (a_new >= 1).all() and all(((q == 0) | (q == s)).all() for q, s in zip(q_new, a_new))
    assert ok, "a validation answer does not match its givens"
    out = dict(base)
    out["val_q"], out["val_a"], out["val_rating"], out["val_row"] = q_new, a_new, r_new, new_idx.astype(np.int64)
    meta2 = dict(meta, n_val=a.n, valset_seed=a.seed, valset_n=a.n, valset_excludes="train_row + the 512 monitor rows", base_npz=str(a.npz))
    out["meta"] = np.asarray([repr(meta2)])
    np.savez_compressed(a.out, **out)
    print(f"wrote {a.out}: val {a.n} rows (seed {a.seed}; {n_unique} unique puzzle strings; rating mean {r_new.mean():.2f} vs monitor "
          f"{base['val_rating'].mean():.2f} vs test {base['test_rating'].mean():.2f}); train/test byte-identical; {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
