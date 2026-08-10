# Ledger: CI-2 (anti-linearity + rank — funnel-collapse probe inverted into a permanent gate),
#         CI-1 extension (full-model S9 equivariance), C13 (param budget),
#         C14 mechanism tests (Amendment D: attention flux measured at all scales,
#         ablation knobs, toll wired into the objective — CI-5's mechanism half),
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
    # rule distribution, sizes, and both flux ledgers are invariants
    assert float(jnp.max(jnp.abs(out_p.rule_q - out.rule_q))) < 1e-4
    assert float(jnp.max(jnp.abs(out_p.size_h - out.size_h))) < 1e-4
    assert float(jnp.max(jnp.abs(out_p.flux - out.flux))) < 1e-3
    assert float(jnp.max(jnp.abs(out_p.flux_attn - out.flux_attn))) < 1e-3


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


# ── CI-2: anti-linearity + rank (the linearity-collapse probe, inverted) ─────────────

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
    assert rel > 1e-2, f"model is (near-)linear: superposition holds to {rel:.2e} — linearity-collapse regression"


def test_ci2_output_rank_floor(params):
    n = 40
    xs = jax.random.normal(jax.random.PRNGKey(8), (n, N_FIELDS, G.CANVAS, G.CANVAS, 2))
    outs = jax.vmap(lambda x: _f(params, x))(xs).reshape(n, -1)
    s = np.linalg.svd(np.asarray(outs), compute_uv=False)
    rank = int(np.sum(s > s[0] * 1e-4))
    assert rank >= n - 2, f"output rank {rank} < {n-2}: bottleneck collapse — linearity-collapse regression"


# ── C14 (Amendment D): priced attention at all scales ──────────────────────

def test_c14_attention_flux_all_scales(params):
    """Every scale's nonlocal channel must report positive, finite flux at
    init (mu is generically nonzero) — the A_s ledger is real, not decorative."""
    x = jnp.asarray(G.place(_random_grid(5)), dtype=jnp.int32)
    yprev = jnp.full((G.CANVAS, G.CANVAS), G.VOID, dtype=jnp.int32)
    out = forward_fields(params, CFG, build_fields(x, yprev), t_norm=0.0, tau=1.0)
    assert out.flux_attn.shape == (CFG.scales,)
    assert bool(jnp.all(jnp.isfinite(out.flux_attn)))
    assert bool(jnp.all(out.flux_attn > 0)), f"dead nonlocal channel: {out.flux_attn}"


def test_c14_ablation_knobs():
    """attn_max_hw=0 -> absent (A_s = 0 everywhere); attn_max_hw=8 -> the old
    Amendment-B coarse-only mode (fine scales 32,16 carry zero nonlocal flux)."""
    x = jnp.asarray(G.place(_random_grid(6)), dtype=jnp.int32)
    yprev = jnp.full((G.CANVAS, G.CANVAS), G.VOID, dtype=jnp.int32)
    fields = build_fields(x, yprev)

    cfg_absent = Config(d=12, T=2, attn_max_hw=0)
    p = init_params(jax.random.PRNGKey(0), cfg_absent)
    out = forward_fields(p, cfg_absent, fields, t_norm=0.0, tau=1.0)
    assert float(jnp.sum(out.flux_attn)) == 0.0

    cfg_coarse = Config(d=12, T=2, attn_max_hw=8)
    p = init_params(jax.random.PRNGKey(0), cfg_coarse)
    out = forward_fields(p, cfg_coarse, fields, t_norm=0.0, tau=1.0)
    assert float(jnp.sum(out.flux_attn[:2])) == 0.0, "fine scales must be silent at attn_max_hw=8"
    assert bool(jnp.all(out.flux_attn[2:] > 0))


def test_c14_toll_enters_loss(params):
    """beta_flux_nl must price A_s in the objective: the loss difference equals
    the deep-supervision mean of the total attention flux, exactly."""
    from qhrrn2.objective import pair_loss
    x = jnp.asarray(G.place(_random_grid(7)), dtype=jnp.int32)
    y = jnp.asarray(G.place(_random_grid(7)), dtype=jnp.int32)
    cfg0 = Config(d=12, T=2, beta_flux_nl=0.0)
    cfg1 = Config(d=12, T=2, beta_flux_nl=1.0)
    loss0, _ = pair_loss(params, cfg0, x, y, tau=1.0)
    loss1, _ = pair_loss(params, cfg1, x, y, tau=1.0)
    outs = iterate(params, cfg0, x, tau=1.0)
    expected = jnp.mean(jnp.stack([jnp.sum(o.flux_attn) for o in outs]))
    diff = float(loss1 - loss0)
    assert diff > 0
    assert abs(diff - float(expected)) < 1e-3 * max(float(expected), 1.0), (
        f"toll mispriced: loss diff {diff} vs mean total A {float(expected)}")


# ── CI-3a smoke: the loss actually descends on a tiny identity episode ─────

def test_fit_smoke_loss_decreases():
    from qhrrn2.train import fit
    cfg = Config(d=8, T=2)
    params = init_params(jax.random.PRNGKey(1), cfg)
    g = _random_grid(4, 4, 4)
    x = jnp.asarray(np.stack([G.place(g)] * 2), dtype=jnp.int32)
    params, losses = fit(params, cfg, x, x, steps=25, lr=3e-3, seed=0)
    assert losses[-1] < losses[0] * 0.8, f"loss did not descend: {losses[0]:.3f} -> {losses[-1]:.3f}"


# ── CI-10: E4 committed-rule boundary condition (transport, [H-6']) ────────

def test_ci10_rule_override_inert_when_none(params):
    """rule_override=None must reproduce the pre-E4 graph bit-exactly."""
    x = jnp.asarray(G.place(_random_grid(11)), dtype=jnp.int32)
    yprev = jnp.full((G.CANVAS, G.CANVAS), G.VOID, dtype=jnp.int32)
    f = build_fields(x, yprev)
    a = forward_fields(params, CFG, f, t_norm=0.0, tau=1.0)
    b = forward_fields(params, CFG, f, t_norm=0.0, tau=1.0, rule_override=None)
    assert float(jnp.max(jnp.abs(a.logits - b.logits))) == 0.0
    assert float(jnp.max(jnp.abs(a.rule_q - b.rule_q))) == 0.0


def test_ci10_rule_override_conditions_decode(params):
    """A clamped rule must (a) surface in rule_q verbatim and (b) actually
    change the decode when it differs from the self-inferred rule."""
    x = jnp.asarray(G.place(_random_grid(12)), dtype=jnp.int32)
    yprev = jnp.full((G.CANVAS, G.CANVAS), G.VOID, dtype=jnp.int32)
    f = build_fields(x, yprev)
    base = forward_fields(params, CFG, f, t_norm=0.0, tau=1.0)
    far = jnp.argmin(base.rule_q, axis=-1)  # least-selected code per slot
    q_hard = jax.nn.one_hot(far, CFG.K)
    out = forward_fields(params, CFG, f, t_norm=0.0, tau=1.0,
                         rule_override=q_hard)
    assert float(jnp.max(jnp.abs(out.rule_q - q_hard))) == 0.0
    assert float(jnp.max(jnp.abs(out.logits - base.logits))) > 1e-6, (
        "override left the decode unchanged — rule path not conditioning")


def test_ci10_rule_override_preserves_equivariance(params):
    """Transport must not break S9: permuted input + SAME committed rule ⇒
    permuted logits (the rule code is color-blind by construction)."""
    x = _random_grid(13)
    lut = G.random_palette(np.random.default_rng(14))
    xc = jnp.asarray(G.place(x), dtype=jnp.int32)
    xp = jnp.asarray(G.place(G.apply_palette(x, lut)), dtype=jnp.int32)
    yprev = jnp.full((G.CANVAS, G.CANVAS), G.VOID, dtype=jnp.int32)
    q = jax.nn.one_hot(jnp.array([3 % CFG.K, 7 % CFG.K]), CFG.K)
    out = forward_fields(params, CFG, build_fields(xc, yprev), t_norm=0.0,
                         tau=1.0, rule_override=q)
    out_p = forward_fields(params, CFG, build_fields(xp, yprev), t_norm=0.0,
                           tau=1.0, rule_override=q)
    gathered = out_p.logits[..., jnp.asarray(lut)]
    err = float(jnp.max(jnp.abs(gathered - out.logits)))
    assert err < 1e-4, f"S9 broken under committed rule: max err {err}"
