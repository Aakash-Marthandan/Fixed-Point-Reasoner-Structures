# Ledger: C17 model-level tests (CI-8 continued, 2026-08-02):
#   - S9 equivariance HOLDS with cluster streams active (partitions are
#     palette-invariant as partitions; aggregation is field-uniform),
#   - the channel is LIVE (gates open, output changes when clusters differ),
#   - flux_obj is finite and populated only when use_obj,
#   - base config (use_obj=False) keeps the pre-C17 graph: flux_obj zeros,
#     no "obj" params, param count unchanged vs the C1-v3 baseline record.
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from qhrrn2 import grid as G
from qhrrn2.config import Config
from qhrrn2.model import count_params, init_params, iterate

CFG_OBJ = Config(d=12, T=2, use_obj=True)
CFG_BASE = Config(d=12, T=2)


@pytest.fixture(scope="module")
def params_obj():
    return init_params(jax.random.PRNGKey(0), CFG_OBJ)


def _random_grid(seed, h=9, w=7):
    return np.asarray(np.random.default_rng(seed).integers(0, 10, (h, w)), dtype=np.int8)


def _last(params, cfg, grid, task_vec=None):
    xc = jnp.asarray(G.place(grid), dtype=jnp.int32)
    return iterate(params, cfg, xc, tau=1.0, task_vec=task_vec)[-1]


def test_c17_equivariance(params_obj):
    x = _random_grid(1)
    lut = G.random_palette(np.random.default_rng(2))
    out = _last(params_obj, CFG_OBJ, x)
    out_p = _last(params_obj, CFG_OBJ, G.apply_palette(x, lut))
    err = float(jnp.max(jnp.abs(out_p.logits[..., jnp.asarray(lut)] - out.logits)))
    assert err < 1e-4, f"C17 broke S9 equivariance: {err}"
    # flux totals sum ~5e5 float32 terms: invariance is exact up to
    # accumulation order — assert RELATIVE error (measured 3e-7)
    rel = jnp.max(jnp.abs(out_p.flux_obj - out.flux_obj)
                  / jnp.maximum(out.flux_obj, 1e-9))
    assert float(rel) < 1e-5, f"flux_obj not palette-invariant: rel {float(rel)}"


def test_c17_channel_live(params_obj):
    """Same cell multiset, different cluster structure -> different output.
    A solid 3x3 block vs the same 9 cells scattered: component_mean differs,
    so logits must differ (the shared-per-object variable is in the graph)."""
    solid = np.zeros((7, 7), dtype=np.int8)
    solid[2:5, 2:5] = 4
    scattered = np.zeros((7, 7), dtype=np.int8)
    scattered[[0, 0, 2, 2, 4, 4, 6, 6, 3], [0, 3, 1, 5, 0, 6, 2, 5, 3]] = 4
    out_a = _last(params_obj, CFG_OBJ, solid)
    out_b = _last(params_obj, CFG_OBJ, scattered)
    assert float(jnp.max(jnp.abs(out_a.logits - out_b.logits))) > 1e-4
    assert bool(jnp.all(jnp.isfinite(out_a.flux_obj)))
    assert float(jnp.sum(out_a.flux_obj)) > 0.0, "cluster VIB emitting zero information"


def test_c17_precomputed_labels_match_ingraph():
    """Speed-pipeline keystone: precomputed + ROLLED labels must produce the
    SAME loss as in-graph segmentation of the rolled canvas (partitions are
    roll-covariant; ids non-canonical but unique)."""
    from qhrrn2 import episodic as E
    from qhrrn2.objective import pair_loss

    corpus, _ = E.build_corpus(frozenset(), n_val=0, seed=0, limit=3)
    dev = E.corpus_to_device(corpus)
    dev["labels"] = E.precompute_labels(corpus)
    params = init_params(jax.random.PRNGKey(0), CFG_OBJ)
    x_b, y_b, t_b, lab_b = E.sample_batch(jax.random.PRNGKey(7), dev,
                                          len(corpus.task_ids), 4)
    for i in range(4):
        l_pre, _ = pair_loss(params, CFG_OBJ, x_b[i], y_b[i], tau=1.0,
                             labels_x=lab_b[i])
        l_ing, _ = pair_loss(params, CFG_OBJ, x_b[i], y_b[i], tau=1.0)
        assert abs(float(l_pre) - float(l_ing)) < 1e-5, (
            f"precomputed labels change the loss: {float(l_pre)} vs {float(l_ing)}")


def test_remat_grad_equality():
    """cfg.remat must change memory, not math: loss and grads equal (2026-08-05)."""
    import dataclasses
    from qhrrn2.objective import pair_loss
    x = _random_grid(9)
    y = _random_grid(10)
    xc = jnp.asarray(G.place(x), dtype=jnp.int32)
    yc = jnp.asarray(G.place(y), dtype=jnp.int32)
    for cfg in (CFG_OBJ, dataclasses.replace(CFG_OBJ, remat=True)):
        p = init_params(jax.random.PRNGKey(0), CFG_OBJ)
        (l, _), g = jax.value_and_grad(pair_loss, has_aux=True)(
            p, cfg, xc, yc, tau=1.0, rng=jax.random.PRNGKey(4))
        if cfg.remat:
            assert abs(float(l) - l0) < 1e-5
            for a, b in zip(jax.tree.leaves(g), g0):
                assert float(jnp.max(jnp.abs(a - b))) < 1e-4
        else:
            l0, g0 = float(l), jax.tree.leaves(g)


def test_c17_off_is_base_graph():
    p_base = init_params(jax.random.PRNGKey(0), CFG_BASE)
    assert "obj" not in p_base
    out = _last(p_base, CFG_BASE, _random_grid(3))
    assert float(jnp.sum(jnp.abs(out.flux_obj))) == 0.0
    # C1-v3 baseline param record at d=12 T=2 must be unchanged by C17-off
    n = count_params(p_base)
    n_obj = count_params(init_params(jax.random.PRNGKey(0), CFG_OBJ))
    assert n_obj > n, "use_obj=True added no params?"
    print(f"params d=12: base {n}, +C17 {n_obj} (+{n_obj - n})")
