# Ledger: [H-27] PoE candidate scoring — named tests (phase-plan 2026-08-11).
# The scorer must (a) rank by joint likelihood across members, (b) apply each
# member's VIEW TRANSFORM to the candidate before scoring (the eval-4 lesson:
# views are conjugated frames), (c) leave every existing output untouched
# (additive instrument — the deployed vote path has its own green suite).
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qhrrn2 import population as P
from qhrrn2 import grid as G


class _FakeTr:
    def __init__(self, transpose=False):
        self.transpose = transpose

    def apply(self, grid):
        return grid.T.copy() if self.transpose else grid


def _mk_group(cands_logits, trs):
    """Build a fake 'group' + monkeypatched forward that returns fixed
    log-probs per member. cands_logits: (M, 32, 32, V) log-prob maps."""
    M = cands_logits.shape[0]
    g = {"cfg": None, "tau": 1.0, "model": None,
         "x_query": np.zeros((M, 1, 32, 32)),
         "val": [(None, None, trs[m]) for m in range(M)]}
    lp = cands_logits[:, None]  # (M, Q=1, 32, 32, V)
    lph = np.full((M, 1, 30), np.log(1 / 30))
    lpw = np.full((M, 1, 30), np.log(1 / 30))
    return g, (lp, lph, lpw)


def test_poe_ranks_by_joint_likelihood(monkeypatch):
    V = 11
    a = np.zeros((2, 2), dtype=np.int8)
    b = np.ones((2, 2), dtype=np.int8)
    lp = np.full((2, 32, 32, V), np.log(1e-9))
    # member 0 strongly prefers color 0 everywhere; member 1 mildly prefers 1
    lp[0, :, :, 0] = np.log(0.9)
    lp[0, :, :, 1] = np.log(0.05)
    lp[1, :, :, 1] = np.log(0.4)
    lp[1, :, :, 0] = np.log(0.3)
    g, fixed = _mk_group(lp, [_FakeTr(), _FakeTr()])
    monkeypatch.setattr(P, "_pop_forward_logp",
                        lambda cfg, tau: lambda model, tv, xq: fixed)
    ranked = P.poe_rank([g], [None], [[a, b]])
    (s_best, ci_best), (s_2nd, ci_2nd) = ranked[0][0], ranked[0][1]
    # joint: a gets log.9+log.3 per cell; b gets log.05+log.4 — a wins
    assert ci_best == 0 and s_best > s_2nd


def test_poe_applies_view_transform(monkeypatch):
    V = 11
    cand = np.zeros((2, 3), dtype=np.int8)  # non-square: transpose matters
    lp = np.full((1, 32, 32, V), np.log(1e-9))
    lp[0, :, :, 0] = np.log(0.9)
    lph = np.full((1, 1, 30), np.log(1e-9))
    lpw = np.full((1, 1, 30), np.log(1e-9))
    lph[0, 0, 2] = 0.0   # member expects h=3 (the TRANSPOSED candidate)
    lpw[0, 0, 1] = 0.0   # and w=2
    g = {"cfg": None, "tau": 1.0, "model": None,
         "x_query": np.zeros((1, 1, 32, 32)),
         "val": [(None, None, _FakeTr(transpose=True))]}
    fixed = (lp[:, None], lph, lpw)
    monkeypatch.setattr(P, "_pop_forward_logp",
                        lambda cfg, tau: lambda model, tv, xq: fixed)
    ranked = P.poe_rank([g], [None], [[cand]])
    score = ranked[0][0][0]
    # shape terms contribute ~0 (log 1) ONLY if the transform was applied
    # (h=3,w=2 in member frame); unapplied would add 2*log(1e-9) ~ -41
    assert score > 6 * np.log(0.9) + 2 * np.log(1e-9) + 1  # cells + margin
    assert score > -30
