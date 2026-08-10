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


def test_c20c_gate_loader_route():
    eps = G.load_task("rg_00d62c1b")
    assert len(eps) == 3 and len(eps[0].support) == 3
    assert all(e.query_y is not None for e in eps)


def test_c20a_corpus_mixing_and_no_orbit_on_re_rows():
    import qhrrn2.episodic as E
    corpus, _val = E.build_corpus(frozenset(), limit=3, n_val=0, orbit_n=2,
                                  rearc_families=["00d62c1b"],
                                  rearc_per_family=4, rearc_seed=1)
    tids = list(corpus.task_ids)
    assert any(t == "re_00d62c1b" for t in tids), tids
    assert not any(t.startswith("re_") and "@o" in t for t in tids)
    assert any("@o1" in t for t in tids)  # base rows still orbit


def test_c20c_rt_train_set_route_and_disjointness():
    import pathlib
    rt_dir = pathlib.Path(rearc.REARC_ROOT).parent / "re_train48"
    if not rt_dir.exists():
        pytest.skip("rt set not built")
    rt = sorted(p.stem for p in rt_dir.glob("*.json"))
    assert len(rt) == 48
    eps = G.load_task(rt[0])
    assert len(eps) == 3 and eps[0].query_y is not None
    train, gate = rearc.family_split()
    fams = {t[3:] for t in rt}
    assert fams <= set(train) and not (fams & set(gate))
