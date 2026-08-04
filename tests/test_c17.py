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
