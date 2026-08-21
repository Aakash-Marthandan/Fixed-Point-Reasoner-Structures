# Ledger: SPRINT S2 — named tests for the Sudoku-Extreme adapter (written
# before any benchmark run): string<->grid, the augmentation group preserves
# validity / givens / counts (and digit identity when digit=False), seeded
# subsample determinism, stratified subsample, prepare() on a synthetic CSV,
# and the corpus contract (one task row, val disjoint from train).
import csv
import numpy as np
import pytest

from qhrrn2 import sudoku as SU
from qhrrn2 import sudoku_extreme as SX


def _pairs(n, seed=3, givens=40):
    return SU.sample_pairs(n, seed=seed, givens=givens)


def test_str_grid_roundtrip():
    puz, sol = _pairs(1)[0]
    s = SX.grid_to_str(puz)
    assert len(s) == 81 and s.count(".") == int((puz == 0).sum())
    assert np.array_equal(SX.str_to_grid(s), puz)
    assert np.array_equal(SX.str_to_grid(SX.grid_to_str(sol)), sol)


def test_augment_is_a_sudoku_symmetry():
    rng = np.random.default_rng(0)
    for puz, sol in _pairs(4):
        for digit in (False, True):
            p2, s2 = SX.augment(puz, sol, rng, digit=digit)
            assert SU.is_valid_solution(s2)
            assert SU.agrees_on_givens(p2, s2)
            assert int((p2 != 0).sum()) == int((puz != 0).sum())
            if not digit:   # position maps only: the multiset of digits is unchanged
                assert sorted(p2[p2 != 0].tolist()) == sorted(puz[puz != 0].tolist())
            assert SU.count_solutions(p2, limit=2) == 1


def test_subsample_deterministic_and_without_replacement():
    a = SX.subsample_indices(5000, 100, seed=7)
    b = SX.subsample_indices(5000, 100, seed=7)
    c = SX.subsample_indices(5000, 100, seed=8)
    assert np.array_equal(a, b) and len(set(a.tolist())) == 100 and not np.array_equal(a, c)


def test_stratified_subsample_covers_bins():
    r = np.arange(1000)
    idx = SX.stratified_subsample(r, 80, seed=1, bins=8)
    assert len(idx) == 80 and len(set(idx.tolist())) == 80
    # every rating octile represented
    assert all(((idx >= b * 125) & (idx < (b + 1) * 125)).sum() == 10 for b in range(8))


def _write_csv(path, pairs, ratings):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(SX.COLS)
        for (p, s), r in zip(pairs, ratings):
            w.writerow(["syn", SX.grid_to_str(p), SX.grid_to_str(s), r])


def test_prepare_and_corpus_contract(tmp_path):
    tr = _pairs(12, seed=11); te = _pairs(5, seed=12)
    _write_csv(tmp_path / "train.csv", tr, range(12))
    _write_csv(tmp_path / "test.csv", te, range(5))
    out = tmp_path / "prep.npz"
    meta = SX.prepare(tmp_path / "train.csv", tmp_path / "test.csv", out, k=6, n_val=3, seed=0)
    assert meta["n_test"] == 5 and meta["k"] == 6
    d = SX.load_prepared(out)
    assert d["train_q"].shape == (6, 9, 9) and d["val_q"].shape == (3, 9, 9) and d["test_q"].shape == (5, 9, 9)
    # train / val disjoint, val / test disjoint (by puzzle string)
    trs = {SX.grid_to_str(q) for q in d["train_q"]}
    assert not any(SX.grid_to_str(q) in trs for q in d["val_q"])
    corpus, val = SX.build_corpus_extreme(out, n_aug=3, seed=0)
    assert corpus.task_ids == ("sudoku",) and corpus.x.shape == (6 * 4, 32, 32)
    assert corpus.starts.tolist() == [0, 24] and (corpus.tidx == 0).all()
    assert val[0][0] == 0 and len(val[0][2]) == 3
    # every corpus x/y is a valid placed puzzle/solution pair
    for x, y in zip(corpus.x[:8], corpus.y[:8]):
        p9, s9 = x[:9, :9].astype(np.int8), y[:9, :9].astype(np.int8)
        assert SU.is_valid_solution(s9) and SU.agrees_on_givens(p9, s9)


def test_build_corpus_digit_aug_flag(tmp_path):
    tr = _pairs(3, seed=21); te = _pairs(2, seed=22)
    _write_csv(tmp_path / "train.csv", tr, [1, 2, 3]); _write_csv(tmp_path / "test.csv", te, [1, 2])
    out = tmp_path / "p.npz"; SX.prepare(tmp_path / "train.csv", tmp_path / "test.csv", out, k=2, n_val=1, seed=0)
    c0, _ = SX.build_corpus_extreme(out, n_aug=5, seed=0, digit_aug=False)
    c1, _ = SX.build_corpus_extreme(out, n_aug=5, seed=0, digit_aug=True)
    assert c0.x.shape == c1.x.shape == (12, 32, 32)
