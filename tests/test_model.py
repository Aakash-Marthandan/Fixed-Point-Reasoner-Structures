# Ledger: CI-2 (anti-linearity + rank — April E1 inverted into a permanent gate),
#         CI-1 extension (full-model S9 equivariance), C13 (param budget),
#         trainability smoke for CI-3a.
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from qhrrn2 import grid as G
from qhrrn2.config import Config
from qhrrn2.model import (N_FIELDS, build_fields, count_params, forward_fields,
                          init_params, iterate)

CFG = Config(d=12, T=2)


@pytest.fixture(scope="module")
def params():
    return init_params(jax.random.PRNGKey(0), CFG)


def _random_grid(seed, h=9, w=7):
    return np.asarray(np.random.default_rng(seed).integers(0, 10, (h, w)), dtype=np.int8)


# ── Shapes & budget ─────────────────────────────────────────────────────────

def test_forward_shapes(params):
    x = jnp.asarray(G.place(_random_grid(0)), dtype=jnp.int32)
    outs = iterate(params, CFG, x, tau=1.0)
    assert len(outs) == CFG.T
    o = outs[-1]
    assert o.logits.shape == (G.CANVAS, G.CANVAS, N_FIELDS)
    assert o.size_h.shape == (30,) and o.size_w.shape == (30,)
    assert o.flux.shape == (CFG.scales,)
    assert o.rule_q.shape == (CFG.M, CFG.K)
    assert bool(jnp.all(jnp.isfinite(o.logits)))


def test_param_budget(params):
    n = count_params(params)
    print(f"\n  total params at d={CFG.d}: {n:,}")
    assert n < 150_000, "d=12 core must stay well under the toy budget"


# ── CI-1 extension: full-model S9 equivariance (C2/C3 at init) ─────────────

def test_s9_equivariance(params):
    x = _random_grid(1)
    lut = G.random_palette(np.random.default_rng(2))
    xc = jnp.asarray(G.place(x), dtype=jnp.int32)
    xp = jnp.asarray(G.place(G.apply_palette(x, lut)), dtype=jnp.int32)

    yprev = jnp.full((G.CANVAS, G.CANVAS), G.VOID, dtype=jnp.int32)
    out = forward_fields(params, CFG, build_fields(xc, yprev), t_norm=0.0, tau=1.0)
    out_p = forward_fields(params, CFG, build_fields(xp, yprev), t_norm=0.0, tau=1.0)

    # logits_perm[..., lut[c]] must equal logits[..., c]
    gathered = out_p.logits[..., jnp.asarray(lut)]
    err = float(jnp.max(jnp.abs(gathered - out.logits)))
    assert err < 1e-4, f"S9 equivariance violated at init: max err {err}"
    # rule distribution and sizes are invariants
    assert float(jnp.max(jnp.abs(out_p.rule_q - out.rule_q))) < 1e-4
    assert float(jnp.max(jnp.abs(out_p.size_h - out.size_h))) < 1e-4


def test_color_bias_breaks_equivariance(params):
    """Amendment A: nonzero color_bias must be ABLE to break the symmetry."""
    p2 = jax.tree.map(lambda x: x, params)
    p2["color_bias"] = params["color_bias"].at[3].set(1.0)
    x = _random_grid(3)
    lut = G.identity_palette(); lut[3], lut[5] = 5, 3
    xc = jnp.asarray(G.place(x), dtype=jnp.int32)
    xp = jnp.asarray(G.place(G.apply_palette(x, lut)), dtype=jnp.int32)
    yprev = jnp.full((G.CANVAS, G.CANVAS), G.VOID, dtype=jnp.int32)
    out = forward_fields(p2, CFG, build_fields(xc, yprev), t_norm=0.0, tau=1.0)
    out_p = forward_fields(p2, CFG, build_fields(xp, yprev), t_norm=0.0, tau=1.0)
    err = float(jnp.max(jnp.abs(out_p.logits[..., jnp.asarray(lut)] - out.logits)))
    assert err > 1e-3, "color_bias failed to break S9 — color-constant rules would be unrepresentable"


# ── CI-2: anti-linearity + rank (the April E1 probe, inverted) ─────────────

def _f(params, fields):
    return forward_fields(params, CFG, fields, t_norm=0.0, tau=1.0).logits


def test_ci2_superposition_must_fail(params):
    k1, k2 = jax.random.split(jax.random.PRNGKey(7))
    shape = (N_FIELDS, G.CANVAS, G.CANVAS, 2)
    x1, x2 = jax.random.normal(k1, shape), jax.random.normal(k2, shape)
    a, b = 1.7, -0.6
    lhs = _f(params, a * x1 + b * x2)
    rhs = a * _f(params, x1) + b * _f(params, x2)
    rel = float(jnp.max(jnp.abs(lhs - rhs)) / (jnp.max(jnp.abs(lhs)) + 1e-9))
    assert rel > 1e-2, f"model is (near-)linear: superposition holds to {rel:.2e} — April E1 regression"


def test_ci2_output_rank_floor(params):
    n = 40
    xs = jax.random.normal(jax.random.PRNGKey(8), (n, N_FIELDS, G.CANVAS, G.CANVAS, 2))
    outs = jax.vmap(lambda x: _f(params, x))(xs).reshape(n, -1)
    s = np.linalg.svd(np.asarray(outs), compute_uv=False)
    rank = int(np.sum(s > s[0] * 1e-4))
    assert rank >= n - 2, f"output rank {rank} < {n-2}: bottleneck collapse — April E1 regression"


# ── CI-3a smoke: the loss actually descends on a tiny identity episode ─────

def test_fit_smoke_loss_decreases():
    from qhrrn2.train import fit
    cfg = Config(d=8, T=2)
    params = init_params(jax.random.PRNGKey(1), cfg)
    g = _random_grid(4, 4, 4)
    x = jnp.asarray(np.stack([G.place(g)] * 2), dtype=jnp.int32)
    params, losses = fit(params, cfg, x, x, steps=25, lr=3e-3, seed=0)
    assert losses[-1] < losses[0] * 0.8, f"loss did not descend: {losses[0]:.3f} -> {losses[-1]:.3f}"
