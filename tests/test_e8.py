# Ledger: E8 named tests ([H-23]) — the yprev_init entry must be inert when
# unused (deployed path bit-identical); corruption behaves; the arm fit runs.
import numpy as np
import jax
import jax.numpy as jnp
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import eval_e8 as E8
from qhrrn2 import grid as G
from qhrrn2.config import Config
from qhrrn2.model import init_params, iterate
from qhrrn2.objective import batch_loss


@pytest.fixture(scope="module")
def setup():
    cfg = Config(d=8, T=2, K=8, attn_max_hw=0)
    params = init_params(jax.random.PRNGKey(0), cfg)
    rng = np.random.default_rng(0)
    x = np.asarray(rng.integers(0, 10, (4, 5)), dtype=np.int8)
    return cfg, params, x


def test_yprev_init_inert(setup):
    cfg, params, x = setup
    xc = jnp.asarray(G.place(x), dtype=jnp.int32)
    a = iterate(params, cfg, xc, tau=1.0)
    void = jnp.full((G.CANVAS, G.CANVAS), G.VOID, dtype=jnp.int32)
    b = iterate(params, cfg, xc, tau=1.0, yprev_init=void)
    assert np.array_equal(np.asarray(a[-1].logits), np.asarray(b[-1].logits))
    # batch path: all-VOID yprev_batch == no yprev_batch
    xb = jnp.stack([xc, xc]); yb = jnp.stack([xc, xc])
    l1, _ = batch_loss(params, cfg, xb, yb, tau=1.0)
    l2, _ = batch_loss(params, cfg, xb, yb, tau=1.0,
                       yprev_batch=jnp.stack([void, void]))
    assert np.isclose(float(l1), float(l2), rtol=1e-6)


def test_corrupt(setup):
    _, _, x = setup
    rng = np.random.default_rng(1)
    assert np.array_equal(E8.corrupt(x, 0.0, rng), x)
    c = E8.corrupt(x, 0.3, rng)
    assert c.shape == x.shape
    frac = float((c != x).mean())
    assert 0.0 < frac <= 0.35


def test_e8_fit_smoke(setup):
    cfg, params, x = setup
    rng = np.random.default_rng(2)
    sup = []
    for _ in range(3):
        g = np.asarray(rng.integers(0, 10, (4, 5)), dtype=np.int8)
        sup.append((g, g.copy()))
    eps_list = [G.Episode(task_id="t", support=tuple(sup), query_x=sup[0][0],
                          query_y=sup[0][1])]
    state = {"model": params, "table": np.zeros((2, cfg.d_task), np.float32)}
    model, snaps = E8.fit_e8(state, cfg, eps_list, steps=12, val_every=6,
                             anchor=True, restarts=True, anneal=True,
                             restart_every=6)
    assert len(snaps) == 2
    assert snaps[-1][2] == E8.TAU_PLATEAUS[-1]  # anneal reached final plateau
    ok = E8.retention(model, cfg, sup[0][0], sup[0][1],
                      jnp.asarray(snaps[-1][1]), snaps[-1][2], k=2)
    assert isinstance(ok, bool)
