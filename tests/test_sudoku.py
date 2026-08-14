# Ledger: S-PORT (H-33) named tests — the generator's correctness AND the
# symmetry claim the port is built on (S9 == digit relabeling), tested rather
# than asserted, per discipline rule 1 (no [H] ships without its test).
import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qhrrn2 import grid as G, sudoku as S


def test_generated_solutions_are_valid():
    rng = np.random.default_rng(0)
    for _ in range(3):
        assert S.is_valid_solution(S.full_grid(rng))


def test_puzzles_are_unique_and_agree_with_solution():
    for puz, sol in S.sample_pairs(3, seed=1, givens=32):
        assert S.is_valid_solution(sol)
        assert S.agrees_on_givens(puz, sol)
        assert S.count_solutions(puz, limit=2) == 1        # uniqueness
        assert (puz == S.BLANK).sum() > 0                  # actually a puzzle
        assert np.array_equal(S.solve(puz), sol)           # unique => THE sol


def test_generation_is_seed_deterministic():
    a = S.sample_pairs(2, seed=7, givens=34)
    b = S.sample_pairs(2, seed=7, givens=34)
    for (pa, sa), (pb, sb) in zip(a, b):
        assert np.array_equal(pa, pb) and np.array_equal(sa, sb)


def test_s9_relabeling_is_a_sudoku_symmetry():
    """THE claim the port exploits: our exact S9 color equivariance IS
    Sudoku's digit-relabeling symmetry. A relabeled (puzzle, solution) pair
    is still a valid, still-unique Sudoku instance — so an S9-equivariant
    solver gets that entire orbit for free, with zero augmentation."""
    rng = np.random.default_rng(3)
    puz, sol = S.sample(rng, givens=34)
    lut = G.random_palette(rng)                 # permutes 1..9, fixes 0 and VOID
    rp, rs = G.apply_palette(puz, lut), G.apply_palette(sol, lut)
    assert S.is_valid_solution(rs)
    assert S.agrees_on_givens(rp, rs)
    assert S.count_solutions(rp, limit=2) == 1
    # blanks are INVARIANT: "unknown" must not be relabeled into a digit
    assert np.array_equal(rp == S.BLANK, puz == S.BLANK)


def test_canvas_contract_and_void_separation():
    rng = np.random.default_rng(5)
    puz, sol = S.sample(rng, givens=34)
    cx, cy = S.to_canvas(puz), S.to_canvas(sol)
    assert cx.shape == (G.CANVAS, G.CANVAS) and cy.shape == (G.CANVAS, G.CANVAS)
    assert (cx[:9, :9] == puz).all() and (cy[:9, :9] == sol).all()
    # outside the 9x9 is VOID, never black — blank(0) and off-grid must not
    # be conflated (the C1 padding-conflation hazard, in the Sudoku setting)
    assert (cx[9:, :] == G.VOID).all() and (cx[:, 9:] == G.VOID).all()
    assert G.VOID not in set(np.unique(cy[:9, :9]))
    assert 0 not in set(np.unique(cy[:9, :9]))     # solutions have no blanks


def test_episode_contract():
    puz, sol = S.sample_pairs(1, seed=11, givens=34)[0]
    ep = S.episode(puz, sol)
    assert ep.task_id == "sudoku" and ep.support == ()
    assert ep.query_y is not None and S.is_valid_solution(ep.query_y)


@pytest.mark.parametrize("givens", [30, 36])
def test_difficulty_dial_moves_givens(givens):
    puz, _ = S.sample(np.random.default_rng(givens), givens=givens)
    n = int((puz != S.BLANK).sum())
    # uniqueness can block removals, so givens is a FLOOR, never exceeded by much
    assert givens <= n <= givens + 12
