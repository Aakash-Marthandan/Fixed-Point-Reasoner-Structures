# Ledger: SPRINT S2 (2026-08-21 course correction #3) — the Sudoku-Extreme
# benchmark adapter. Comparability is BY PROTOCOL, registered before any run:
#   dataset  sapientinc/sudoku-extreme (HF): train.csv 3.8M / test.csv 423k,
#            81-char row-major strings ('.' = blank), `rating` = tdoku
#            backtracks, train/test "mathematically inequivalent".
#   benchmark row (HRM/TRM convention, build_sudoku_dataset.py):
#            seeded RANDOM 1,000 training puzzles, no difficulty filter;
#            evaluation = EXACT accuracy on the FULL test set, one
#            deterministic prediction per puzzle, no voting.
#   augmentation group (HRM's shuffle_sudoku, applied to TRAIN only):
#            digit permutation (0 fixed) + transpose + band permutation +
#            row-within-band + stack permutation + col-within-stack.
#            DIGIT PERMUTATION IS OUR EXACT S9 EQUIVARIANCE ([P-C1]) — the
#            network is invariant to it by construction, so it is OFF by
#            default here and exists only as the with/without ABLATION arm;
#            the position maps (transpose/band/row/stack/col) are added.
#   val monitor: 64 extra train.csv rows disjoint from the 1k and from test
#            (same distribution as train) — a training monitor only, never a
#            selection set; the test set is touched by the evaluator alone.
# Digits -> colors 1-9, blank -> 0 (black, the palette fixed point), 9x9 at
# the canvas origin with VOID outside — identical to sudoku.py's adapter, so
# every instrument runs UNCHANGED on benchmark puzzles.
"""Sudoku-Extreme loader, seeded subsample, augmentation group, corpus."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from . import grid as G
from . import sudoku as SU

N = SU.N
BOX = SU.BOX
COLS = ("source", "question", "answer", "rating")


# ── strings <-> grids ─────────────────────────────────────────────────────

def str_to_grid(s: str) -> np.ndarray:
    assert len(s) == N * N, f"expected 81 chars, got {len(s)}"
    return np.array([0 if ch == "." else int(ch) for ch in s],
                    dtype=np.int8).reshape(N, N)


def grid_to_str(g: np.ndarray) -> str:
    return "".join("." if v == 0 else str(int(v)) for v in np.asarray(g).ravel())


# ── CSV access (streaming; train.csv is ~700 MB) ──────────────────────────

def count_rows(path) -> int:
    n = 0
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = next(r)
        assert tuple(header) == COLS, header
        for _ in r:
            n += 1
    return n


def read_rows(path, *, indices=None, limit=None):
    """Yield (row_index, source, puzzle_grid, solution_grid, rating).
    `indices`: a SET of row indices to keep (streaming subsample)."""
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = next(r)
        assert tuple(header) == COLS, header
        for i, row in enumerate(r):
            if limit is not None and i >= limit:
                return
            if indices is not None and i not in indices:
                continue
            src, q, a, rating = row
            yield i, src, str_to_grid(q), str_to_grid(a), int(rating)


def subsample_indices(n_total: int, k: int, seed: int) -> np.ndarray:
    """HRM/TRM convention: uniform random WITHOUT replacement over the whole
    training file (difficulty-agnostic). Our seed is the registered one."""
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_total, size=k, replace=False))


# ── the augmentation group (HRM shuffle_sudoku semantics) ─────────────────

def augment(puz: np.ndarray, sol: np.ndarray, rng: np.random.Generator, *,
            digit: bool = False):
    """One random element of the Sudoku symmetry group used by HRM/TRM:
    transpose (p=.5), band permutation, row permutation within each band,
    stack permutation, column permutation within each stack, and (optionally)
    a digit permutation with blank fixed. Applied identically to puzzle and
    solution. Validity + uniqueness are symmetry-invariant."""
    p, s = np.asarray(puz, np.int8), np.asarray(sol, np.int8)
    if rng.random() < 0.5:
        p, s = p.T.copy(), s.T.copy()
    bands = rng.permutation(BOX)
    rows = np.concatenate([b * BOX + rng.permutation(BOX) for b in bands])
    stacks = rng.permutation(BOX)
    cols = np.concatenate([b * BOX + rng.permutation(BOX) for b in stacks])
    p, s = p[rows][:, cols], s[rows][:, cols]
    if digit:
        dmap = np.concatenate([[0], rng.permutation(N) + 1]).astype(np.int8)
        p, s = dmap[p], dmap[s]
    return np.ascontiguousarray(p), np.ascontiguousarray(s)


# ── prepared benchmark file (ships to the pod as ONE small npz) ───────────

def prepare(train_csv, test_csv, out_npz, *, k: int = 1000, n_val: int = 64,
            seed: int = 0, test_limit: int | None = None) -> dict:
    """Materialize: the seeded k-puzzle training subsample (+ n_val disjoint
    monitor rows from train.csv) and the FULL test set, as int8 arrays."""
    n_train = count_rows(train_csv)
    idx = subsample_indices(n_train, k + n_val, seed)
    keep = set(int(i) for i in idx)
    tr_q, tr_a, tr_r, tr_i = [], [], [], []
    for i, _, q, a, r in read_rows(train_csv, indices=keep):
        tr_q.append(q); tr_a.append(a); tr_r.append(r); tr_i.append(i)
    assert len(tr_q) == k + n_val, (len(tr_q), k + n_val)
    te_q, te_a, te_r, te_s = [], [], [], []
    for _, src, q, a, r in read_rows(test_csv, limit=test_limit):
        te_q.append(q); te_a.append(a); te_r.append(r); te_s.append(src)
    # disjointness train-subsample vs test, by puzzle string (belt and braces
    # over the dataset's own "mathematically inequivalent" guarantee)
    te_set = {grid_to_str(q) for q in te_q}
    assert not any(grid_to_str(q) in te_set for q in tr_q), "train/test overlap"
    srcs = sorted(set(te_s)); s_id = {s: i for i, s in enumerate(srcs)}
    meta = dict(seed=seed, k=k, n_val=n_val, n_train_file=n_train,
                n_test=len(te_q), train_csv=str(train_csv), test_csv=str(test_csv))
    np.savez_compressed(
        out_npz,
        train_q=np.stack(tr_q[:k]), train_a=np.stack(tr_a[:k]),
        train_rating=np.asarray(tr_r[:k], np.int32), train_row=np.asarray(tr_i[:k], np.int64),
        val_q=np.stack(tr_q[k:]), val_a=np.stack(tr_a[k:]),
        val_rating=np.asarray(tr_r[k:], np.int32),
        test_q=np.stack(te_q), test_a=np.stack(te_a),
        test_rating=np.asarray(te_r, np.int32),
        test_source=np.asarray([s_id[s] for s in te_s], np.int16),
        source_names=np.asarray(srcs),
        meta=np.asarray([repr(meta)]))
    return meta


def load_prepared(npz_path):
    z = np.load(npz_path, allow_pickle=False)
    meta = eval(str(z["meta"][0]))  # our own repr(dict) — trusted file
    return {k: z[k] for k in z.files if k != "meta"} | {"meta": meta}


def stratified_subsample(ratings: np.ndarray, n: int, seed: int, bins: int = 8):
    """n indices stratified over RANK-based equal-count rating bins (the
    evaluator's instrument subsample: the physics is read where the difficulty
    is). Rank bins — not quantile edges — because the rating distribution has
    a mass at 0 (quantile edges collapse and bins go empty; smoke 2026-08-21)."""
    rng = np.random.default_rng(seed)
    ratings = np.asarray(ratings)
    order = np.argsort(ratings, kind="stable")
    rank = np.empty(len(ratings), dtype=np.int64); rank[order] = np.arange(len(ratings))
    b_of = (rank * bins) // max(len(ratings), 1)
    per = [n // bins + (1 if i < n % bins else 0) for i in range(bins)]
    out = []
    for b in range(bins):
        pool = np.where(b_of == b)[0]
        out.extend(rng.choice(pool, size=min(per[b], len(pool)), replace=False))
    return np.sort(np.asarray(out, dtype=np.int64))


# ── corpus builder (ONE task row, like build_sudoku_corpus) ───────────────

def build_corpus_extreme(npz_path, *, n_aug: int = 100, seed: int = 0,
                         digit_aug: bool = False, limit_base: int | None = None):
    """Corpus = the k base puzzles x (1 + n_aug) group-augmented copies under
    the single 'sudoku' task row; val = the disjoint monitor rows. Mirrors
    episodic.build_sudoku_corpus's Corpus contract exactly."""
    from .episodic import Corpus
    d = load_prepared(npz_path)
    q, a = d["train_q"], d["train_a"]
    if limit_base is not None:
        q, a = q[:limit_base], a[:limit_base]
    rng = np.random.default_rng(seed + 7_777)
    xs, ys = [], []
    for p_, s_ in zip(q, a):
        xs.append(G.place(p_)); ys.append(G.place(s_))
        for _ in range(n_aug):
            pa, sa = augment(p_, s_, rng, digit=digit_aug)
            xs.append(G.place(pa)); ys.append(G.place(sa))
    P = len(xs)
    corpus = Corpus(
        task_ids=("sudoku",),
        x=np.stack(xs).astype(np.int32), y=np.stack(ys).astype(np.int32),
        tidx=np.zeros(P, dtype=np.int32),
        starts=np.asarray([0, P], dtype=np.int32),
        bound_h=np.full(P, G.CANVAS - N, dtype=np.int32),
        bound_w=np.full(P, G.CANVAS - N, dtype=np.int32))
    val_pairs = [(np.asarray(p_, np.int8), np.asarray(s_, np.int8))
                 for p_, s_ in zip(d["val_q"], d["val_a"])]
    return corpus, [(0, "sudoku", val_pairs)]
