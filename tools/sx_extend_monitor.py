# Ledger: CHAMPION NIGHT build B5 (2026-09-08; the Night A selection-instrument lesson: a 64-puzzle train-file
# monitor resolves 1.6 pp, ties at parity-class accuracy, and the later-tie rule selected the memorization side).
# Extends the prepared benchmark npz with a 512-PUZZLE MONITOR: the original 64 monitor rows first (byte-identical),
# then 448 further rows drawn (seed 20260908) from the TRAIN FILE outside the seeded 1k + 64 subsample — never the
# test set — verified string-disjoint from both the training subsample and the full test set. Every other array
# (train_q/a/rating/row, test_q/a/rating/source, source_names) is copied byte-identical, so the evaluator's test set
# and the trainer's corpus are unchanged; only the trainer's val rows (the monitor) grow. Reproduces the base file's
# own subsample first (subsample_indices(n_train, k + n_val, seed)) and asserts it against the stored train_row.
"""  .venv/bin/python tools/sx_extend_monitor.py --npz data/sudoku_extreme/sudoku_extreme_seed0.npz \
        --train-csv data/sudoku_extreme/train.csv --n-monitor 512 --seed 20260908 --out data/sudoku_extreme/sudoku_extreme_seed0_mon512.npz"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qhrrn2 import sudoku_extreme as SX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True); ap.add_argument("--train-csv", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n-monitor", type=int, default=512); ap.add_argument("--seed", type=int, default=20260908)
    a = ap.parse_args()
    t0 = time.time()
    z = np.load(a.npz, allow_pickle=False)
    base = {k: z[k] for k in z.files}
    meta = eval(str(base["meta"][0]))
    k, n_val, seed, n_train = int(meta["k"]), int(meta["n_val"]), int(meta["seed"]), int(meta["n_train_file"])
    idx = SX.subsample_indices(n_train, k + n_val, seed)                  # the base file's own draw (sorted)
    assert np.array_equal(idx[:k], base["train_row"]), "the base npz's train rows do not reproduce from its meta"
    used = set(int(i) for i in idx)
    n_new = a.n_monitor - n_val
    assert n_new > 0, "n_monitor must exceed the base file's monitor rows"
    rng = np.random.default_rng(a.seed)
    pool = np.setdiff1d(np.arange(n_train), idx, assume_unique=False)
    new_idx = np.sort(rng.choice(pool, size=n_new, replace=False))
    assert not (set(int(i) for i in new_idx) & used)
    rows = {}
    for i, _, q, ans, r in SX.read_rows(a.train_csv, indices=set(int(i) for i in new_idx)):
        rows[int(i)] = (q, ans, r)
    assert len(rows) == n_new, (len(rows), n_new)
    q_new = np.stack([rows[int(i)][0] for i in new_idx]).astype(np.int8)
    a_new = np.stack([rows[int(i)][1] for i in new_idx]).astype(np.int8)
    r_new = np.asarray([rows[int(i)][2] for i in new_idx], np.int32)
    # disjointness by puzzle string: the new monitor rows vs the training subsample, the base monitor rows and the FULL test set
    s_new = {SX.grid_to_str(g) for g in q_new}
    assert len(s_new) == n_new
    for name, arr in (("train_q", base["train_q"]), ("val_q", base["val_q"]), ("test_q", base["test_q"])):
        assert not any(SX.grid_to_str(g) in s_new for g in arr), f"monitor overlap with {name}"
    out = dict(base)
    out["val_q"] = np.concatenate([base["val_q"], q_new]); out["val_a"] = np.concatenate([base["val_a"], a_new])
    out["val_rating"] = np.concatenate([base["val_rating"].astype(np.int32), r_new])
    out["val_row"] = np.concatenate([idx[k:], new_idx]).astype(np.int64)
    meta2 = dict(meta, n_val=a.n_monitor, monitor_ext_seed=a.seed, monitor_ext_n=n_new, base_npz=str(a.npz))
    out["meta"] = np.asarray([repr(meta2)])
    np.savez_compressed(a.out, **out)
    print(f"wrote {a.out}: val {out['val_q'].shape[0]} rows ({n_val} base + {n_new} new, seed {a.seed}); "
          f"train {out['train_q'].shape[0]}, test {out['test_q'].shape[0]} byte-identical; {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
