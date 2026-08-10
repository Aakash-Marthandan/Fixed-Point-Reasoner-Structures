# Ledger: C20a/C20b named tests — vendored RE-ARC adapter + family-holdout law.
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qhrrn2 import rearc
from qhrrn2 import grid as G

pytestmark = pytest.mark.skipif(not rearc.REARC_ROOT.exists(),
                                reason="RE-ARC not vendored")


def test_c20a_families_load():
    fams = rearc.family_ids()
    assert len(fams) == 400


def test_c20b_family_split_deterministic_and_disjoint():
    t1, g1 = rearc.family_split()
    t2, g2 = rearc.family_split()
    assert (t1, g1) == (t2, g2)
    assert len(g1) == 100 and len(t1) == 300
    assert not (set(t1) & set(g1))
    assert set(t1) | set(g1) == set(rearc.family_ids())


def test_c20a_episode_contract():
    rng = np.random.default_rng(3)
    ep = rearc.sample_episode(rearc.family_ids()[10], rng)
    assert ep is not None and isinstance(ep, G.Episode)
    assert len(ep.support) == 3 and ep.query_y is not None
    for x, y in list(ep.support) + [(ep.query_x, ep.query_y)]:
        assert x.dtype == np.int8 and y.dtype == np.int8
        assert max(x.shape + y.shape) <= 30
        assert x.min() >= 0 and x.max() <= 9


def test_c20a_difficulty_dial_moves_complexity():
    rng = np.random.default_rng(5)
    fams = rearc.family_ids()[:25]
    easy, hard = [], []
    for fam in fams:
        e = rearc.sample_instance(fam, rng, diff_lb=0.0, diff_ub=0.25)
        h = rearc.sample_instance(fam, rng, diff_lb=0.75, diff_ub=1.0)
        if e is not None:
            easy.append(e[0].size)
        if h is not None:
            hard.append(h[0].size)
    assert np.mean(hard) > np.mean(easy), (
        f"difficulty dial inert: easy {np.mean(easy):.0f} vs hard {np.mean(hard):.0f}")
