# Ledger: 2026-08-07 tests — cross-bulk agreement-regularized population
# (C11 ⊕ [H-15] ⊕ [H-18]). Named checks: (i) weighted batch_loss keeps
# zero-weight rows loss-inert (arm X ≡ plain population objective);
# (ii) consensus refresh writes each member's VIEW of the consensus into its
# query rows; (iii) heterogeneous bulks (different d, d_task) fit jointly and
# scoring returns per-member query preds + the joint-2-guess bit.
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from qhrrn2 import grid as G
from qhrrn2 import population as P
from qhrrn2.config import Config
from qhrrn2.model import init_params
from qhrrn2.objective import batch_loss


def _identity_task(n=3):
    rng = np.random.default_rng(0)
    sup = []
    for _ in range(n + 1):
        g = np.asarray(rng.integers(0, 10, (4, 5)), dtype=np.int8)
        sup.append((g, g.copy()))
    q = np.asarray(rng.integers(0, 10, (4, 5)), dtype=np.int8)
    return [G.Episode(task_id="toy", support=tuple(sup), query_x=q, query_y=q.copy())]


def _bulk(seed, **kw):
    cfg = Config(d=8, T=1, K=8, attn_max_hw=0, **kw)
    return {"name": f"b{seed}",
            "state": {"model": init_params(jax.random.PRNGKey(seed), cfg),
                      "table": np.zeros((4, cfg.d_task), dtype=np.float32)},
            "cfg": cfg, "tau": 1.0}


def test_weighted_batch_loss_zero_rows_inert():
    """Support-only loss == support+query loss when query rows carry weight 0."""
    cfg = Config(d=8, T=1, K=8, attn_max_hw=0)
    params = init_params(jax.random.PRNGKey(0), cfg)
    eps = _identity_task()
    x_mb, y_mb, val, B, Q = P.build_member_batches_x(eps, 2, 1, seed=0)
    x0, y0 = x_mb[0], y_mb[0]
    w = jnp.concatenate([jnp.ones(B), jnp.zeros(Q)])
    full, _ = batch_loss(params, cfg, x0, y0, tau=1.0, weights=w)
    sup, _ = batch_loss(params, cfg, x0[:B], y0[:B], tau=1.0)
    assert np.isclose(float(full), float(sup), rtol=1e-5)


def test_member_batches_x_query_rows():
    eps = _identity_task()
    x_mb, y_mb, val, B, Q = P.build_member_batches_x(eps, 4, 1, seed=0)
    assert Q == 1 and x_mb.shape[1] == B + 1
    for m, (_, _, tr) in enumerate(val):
        expect = G.place(tr.apply(np.asarray(eps[0].query_x)))
        assert np.array_equal(np.asarray(x_mb[m, B]), expect)
        assert np.all(np.asarray(y_mb[m, B]) == G.VOID)  # pre-consensus


def test_consensus_vote_shape_majority():
    a = np.full((3, 3), 4, dtype=np.int8)
    b = a.copy(); b[0, 0] = 7
    odd = np.full((2, 2), 4, dtype=np.int8)  # minority shape excluded
    voted = P.consensus_vote([a, a, b, odd])
    assert voted.shape == (3, 3) and np.array_equal(voted, a)


def test_cross_bulk_fit_and_score_smoke():
    """Two heterogeneous bulks (d_task differs), agreement ON: consensus rows
    are written, fit runs, scoring emits member preds + joint2."""
    bulks = [_bulk(0), _bulk(1, d_task=16)]
    eps = _identity_task()
    F = P.fit_population_cross(bulks, eps, n_views=2, n_seeds=1, steps=12,
                               val_every=6, agree_lambda=0.5, agree_every=4,
                               agree_warmup=4, seed=0)
    assert len(F["snaps"]) == 2
    # consensus refresh happened: query y-rows are no longer all-VOID
    B = F["groups"][0]["B"]
    yq = np.asarray(F["groups"][0]["y"])[:, B]
    assert not np.all(yq == G.VOID)
    # each member's query row is ITS view of one shared consensus grid
    g0 = F["groups"][0]
    cons0 = None
    for m, (_, _, tr) in enumerate(g0["val"]):
        canvas = np.asarray(g0["y"])[m, B]
        hs = np.any(canvas != G.VOID, axis=1).sum()
        ws = np.any(canvas != G.VOID, axis=0).sum()
        raw = tr.invert_output(canvas[:hs, :ws].astype(np.int8))
        cons0 = raw if cons0 is None else cons0
        assert np.array_equal(raw, cons0)
    res = P.score_population_cross(F, eps)
    assert set(res) >= {"solved_pass2", "solved_joint2", "members",
                        "member_query_preds"}
    assert len(res["members"]) == 4
    assert len(res["member_query_preds"]) == 4
    assert res["members"][0]["bulk"] == "b0"
    assert res["members"][2]["bulk"] == "b1"


def test_arm_x_lambda_zero_never_refreshes():
    bulks = [_bulk(0)]
    eps = _identity_task()
    F = P.fit_population_cross(bulks, eps, n_views=2, n_seeds=1, steps=8,
                               val_every=4, agree_lambda=0.0, seed=0)
    B = F["groups"][0]["B"]
    assert np.all(np.asarray(F["groups"][0]["y"])[:, B:] == G.VOID)
