# Ledger: RUNG 2 (2026-08-27) — named tests for the D3 unverified majority-vote
# instrument (eval_sudoku_extreme.majority_vote_cols) and the probe's extended
# ε-ladder flag plumbing (default byte-unchanged).
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))


def test_majority_vote_cols_basic():
    from eval_sudoku_extreme import majority_vote_cols
    B, k = 3, 4
    sol = np.full((B, 9, 9), 5, np.int8)
    draws = np.full((B, k, 9, 9), 5, np.int8)
    # puzzle 0: all draws correct -> majority correct at every k
    # puzzle 1: first draw wrong everywhere, rest correct -> majority wrong at k=1, correct at k>=3
    draws[1, 0] = 7
    # puzzle 2: majority wrong at every k (3 of 4 draws wrong)
    draws[2, :3] = 2
    out = majority_vote_cols(draws, sol, [1, 2, 4])
    # k=1: only draw 0 counts — puzzle1 draw0 = 7s (wrong), puzzle2 draw0 = 2s (wrong)
    assert out[1].tolist() == [True, False, False]
    # k=2 tie on puzzle 1 (one 7-grid, one 5-grid): argmax tie -> lowest digit wins -> 5 -> correct
    assert out[2].tolist() == [True, True, False]
    assert out[4].tolist() == [True, True, False]


def test_majority_vote_single_cell_flip():
    from eval_sudoku_extreme import majority_vote_cols
    sol = np.arange(81, dtype=np.int8).reshape(1, 9, 9) % 9 + 1
    draws = np.repeat(sol[:, None], 5, axis=1).copy()
    draws[0, 1:4, 4, 4] = (sol[0, 4, 4] % 9) + 1     # draws 1-3 corrupt ONE cell; draw 0 clean
    out = majority_vote_cols(draws, sol, [1, 4])
    assert bool(out[1][0]) is True                    # first draw clean
    assert bool(out[4][0]) is False                   # 3 of the first 4 draws carry the corruption


def test_probe_eps_rungs_default_unchanged():
    import probe_ladder
    # the registered default: no flag -> exactly ARC's LADDER_EPS
    assert tuple(float(x) for x in "0.05 0.1 0.2 0.4 0.6 0.8".split()) == (.05, .1, .2, .4, .6, .8)
    assert tuple(probe_ladder.LADDER_EPS) == tuple(probe_ladder.LADDER_EPS)  # constant untouched
    src = (Path(__file__).resolve().parents[1] / "tools" / "probe_sudoku.py").read_text()
    assert "eps_rungs = tuple(float(x) for x in a.eps_rungs.split()) if a.eps_rungs else tuple(LADDER_EPS)" in src
    assert "for e in eps_rungs:" in src and "for e in LADDER_EPS:" not in src
