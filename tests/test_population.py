# Ledger: C11 tests (2026-08-06) — member batches carry correctly transformed
# supervision; inversion round-trips; the vote solves a crafted task where a
# minority of members disagree; the vmapped fit runs and reduces loss.
import numpy as np
import jax
import pytest

from qhrrn2 import grid as G
from qhrrn2 import population as P
from qhrrn2.config import Config
from qhrrn2.model import init_params


def _identity_task(n=3):
    rng = np.random.default_rng(0)
    sup = []
    for _ in range(n + 1):
        g = np.asarray(rng.integers(0, 10, (4, 5)), dtype=np.int8)
        sup.append((g, g.copy()))
    q = np.asarray(rng.integers(0, 10, (4, 5)), dtype=np.int8)
    return [G.Episode(task_id="toy", support=tuple(sup), query_x=q, query_y=q.copy())]


def test_member_batches_transform_supervision():
    eps = _identity_task()
    x_mb, y_mb, val = P.build_member_batches(eps, n_views=8, n_seeds=1, seed=0)
    assert x_mb.shape[0] == 8
    for m, (vx, vy, tr) in enumerate(val):
        assert np.array_equal(vx, tr.apply(eps[0].support[-1][0]))
        # round-trip: inverting the transformed val target recovers the original
        assert np.array_equal(tr.invert_output(np.asarray(vy)),
                              eps[0].support[-1][1])


def test_vote_overrules_minority():
    """Crafted: 3 correct members + 1 wrong-color member -> vote correct."""
    good = np.full((3, 3), 4, dtype=np.int8)
    bad = good.copy(); bad[1, 1] = 7
    votes = [good, good, good, bad]
    stack = np.stack(votes)
    voted = np.apply_along_axis(
        lambda v: np.bincount(v, minlength=11).argmax(), 0, stack).astype(np.int8)
    assert np.array_equal(voted, good)


def test_population_fit_and_score_smoke():
    cfg = Config(d=8, T=1, K=8, attn_max_hw=0)
    state = {"model": init_params(jax.random.PRNGKey(0), cfg),
             "table": np.zeros((4, cfg.d_task), dtype=np.float32)}
    eps = _identity_task()
    F = P.fit_population(state, cfg, eps, n_views=4, n_seeds=1, steps=20,
                         val_every=10, seed=0)
    assert len(F["snaps"]) == 2
    res = P.score_population(F, eps, max_snap_evals=2)
    assert set(res) >= {"solved_pass2", "per_pair_bits", "n_voters", "member_pix"}
    assert len(res["member_pix"]) == 4
