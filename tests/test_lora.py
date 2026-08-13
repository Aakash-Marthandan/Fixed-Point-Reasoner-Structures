# Ledger: [H-12]-on-eq named tests (phase-plan registration 2026-08-11) —
# basin-preserving LoRA TTT: inertness at init, budget bound, trainability.
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from qhrrn2.config import Config
from qhrrn2.model import init_params, forward_fields, build_fields
from qhrrn2 import grid as G
import probe_lora as L

CFG = Config(d=12, T=2)


@pytest.fixture(scope="module")
def params():
    return init_params(jax.random.PRNGKey(0), CFG)


def test_lora_init_inert(params):
    """b=0 init => merged params reproduce the base graph BIT-EXACTLY."""
    lora = L.lora_init(params, jax.random.PRNGKey(1))
    eff = L.merge(params, lora)
    x = jnp.asarray(G.place(np.asarray(
        np.random.default_rng(0).integers(0, 10, (7, 7)), dtype=np.int8)),
        dtype=jnp.int32)
    yprev = jnp.full((G.CANVAS, G.CANVAS), G.VOID, dtype=jnp.int32)
    a = forward_fields(params, CFG, build_fields(x, yprev), t_norm=0.0, tau=1.0)
    b = forward_fields(eff, CFG, build_fields(x, yprev), t_norm=0.0, tau=1.0)
    assert float(jnp.max(jnp.abs(a.logits - b.logits))) == 0.0


def test_lora_budget_under_h12_bound(params):
    lora = L.lora_init(params, jax.random.PRNGKey(1))
    n = L.lora_params_count(lora) + 32  # + tv
    assert n <= 25_000, f"H-12 bound violated: {n}"


def test_lora_nonzero_changes_decode(params):
    lora = L.lora_init(params, jax.random.PRNGKey(1))
    lora = jax.tree.map(lambda x: x + 0.05, lora)
    eff = L.merge(params, lora)
    x = jnp.asarray(G.place(np.asarray(
        np.random.default_rng(2).integers(0, 10, (6, 6)), dtype=np.int8)),
        dtype=jnp.int32)
    yprev = jnp.full((G.CANVAS, G.CANVAS), G.VOID, dtype=jnp.int32)
    a = forward_fields(params, CFG, build_fields(x, yprev), t_norm=0.0, tau=1.0)
    b = forward_fields(eff, CFG, build_fields(x, yprev), t_norm=0.0, tau=1.0)
    assert float(jnp.max(jnp.abs(a.logits - b.logits))) > 1e-6


def test_ewc_penalty_math(params):
    """Cluster V (2026-08-13): zero-b lora -> penalty 0 (inert at init);
    nonzero effective drift -> positive; NU zeros -> 0 regardless."""
    lora = L.lora_init(params, jax.random.PRNGKey(3))
    NU1 = {k: jnp.ones_like(params_at(params, k)) for k in lora}
    NU0 = {k: jnp.zeros_like(params_at(params, k)) for k in lora}
    assert float(L.ewc_penalty(lora, NU1)) == 0.0          # b starts at zero
    lora2 = jax.tree.map(lambda x: x + 0.1, lora)
    assert float(L.ewc_penalty(lora2, NU1)) > 0.0
    assert float(L.ewc_penalty(lora2, NU0)) == 0.0


def params_at(params, key):
    node = params
    for k in key.split("/"):
        node = node[k]
    return node["w"]


def test_ewc_nu_extraction(params):
    """extract_nu finds Adam's second moment in a real optax state and
    returns TARGETS-keyed, shape-matched, nonnegative weights."""
    import optax
    state = {"model": params, "table": jnp.zeros((4, 32))}
    opt = optax.adamw(1e-3, weight_decay=1e-4)
    ost = opt.init(state)
    grads = jax.tree.map(lambda x: jnp.ones_like(x) * 0.1, state)
    _, ost = opt.update(grads, ost, state)
    nu = L.extract_nu(ost)
    assert set(nu) == {"/".join(p) for p in L.TARGETS}
    for k, v in nu.items():
        assert v.shape == params_at(params, k).shape
        assert float(jnp.min(v)) >= 0.0
