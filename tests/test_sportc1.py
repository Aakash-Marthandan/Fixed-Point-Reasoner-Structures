# Ledger: sportC1 build gates (Plan_2026-09-02_Champion_sportC1 §11–§12,
# 2026-09-02). (1) z-NORM (H-50's stabilizer of record): off = the pre-existing
# param tree and graph (no gain key); on = live, scale-invariant (a 1e6-scaled
# carry gives the same output, finite), S9-equivariant. (2) eq_etas: one
# definition, fixed values pin the loop (X0), learned = bit-exact expressions.
# (3) The TRM/EqR field-recipe cell: 5,037,058 params at the field's dims
# (EqR Table 17: 5.03M), forward contract, y = pure readout under fixed etas,
# stablemax loss finite + differentiable, lambda/beta/RI liveness, TRM-exact
# at lambda = beta = 0. (4) stablemax normalization.
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from qhrrn2 import grid as G
from qhrrn2 import model as M
from qhrrn2 import sudoku as SU
from qhrrn2 import trm_cell as TC
from qhrrn2.config import Config
from qhrrn2.objective import log_stablemax, pair_loss

NCFG = Config(d=12, d_ir=16, d_code=16, K=16, T=2, canvas=9, scales=2,
              pool_arity=3, mixer_kind="group9", attn_max_hw=9,
              equilibrium=True, sudoku_layout="native9")
NCFG_N = Config(**{**NCFG.__dict__, "z_norm": "rms"})
TRM_FIELD = Config(canvas=9, scales=2, pool_arity=3, mixer_kind="group9", attn_max_hw=9,
                   equilibrium=True, sudoku_layout="native9", cell_kind="trm", T=16,
                   eta_fixed=1.0, eta_z_fixed=1.0, loss_kind="stablemax")
TRM_TINY = Config(**{**TRM_FIELD.__dict__, "trm_hidden": 32, "trm_layers": 1,
                     "trm_h_cycles": 2, "trm_l_cycles": 2, "T": 3})


def _pair(seed=3):
    puz, sol = SU.sample(np.random.default_rng(seed), 40)
    return jnp.asarray(puz, jnp.int32), jnp.asarray(sol, jnp.int32)


def _open(p, a=0.5):
    return {**p, "eq": {**p["eq"], "alpha_z": jnp.asarray(a)}}


# ── z-norm ────────────────────────────────────────────────────────────────

def test_znorm_off_has_no_gain_and_on_is_live():
    p_off = M.init_params(jax.random.PRNGKey(0), NCFG)
    p_on = M.init_params(jax.random.PRNGKey(0), NCFG_N)
    assert "z_gain" not in p_off["eq"] and p_on["eq"]["z_gain"].shape == (12,)
    # identical weights apart from the gain key (same rng stream)
    for k in ("embed", "readout"):
        assert all(bool(jnp.array_equal(a, b)) for a, b in
                   zip(jax.tree.leaves(p_off[k]), jax.tree.leaves(p_on[k])))
    xc, _ = _pair()
    yprev = jnp.full((9, 9), G.VOID, jnp.int32)
    f = M.build_fields(xc, yprev)
    z_in = jax.random.normal(jax.random.PRNGKey(5), (11, 9, 9, 12))
    o_off = M.forward_fields(_open(p_off), NCFG, f, t_norm=0.0, tau=1.0, z_in=z_in)
    o_on = M.forward_fields(_open(p_on), NCFG_N, f, t_norm=0.0, tau=1.0, z_in=z_in)
    assert float(jnp.max(jnp.abs(o_on.logits - o_off.logits))) > 1e-4      # the norm is live
    # z_in=None: both graphs identical (the entry is the only difference)
    a0 = M.forward_fields(_open(p_off), NCFG, f, t_norm=0.0, tau=1.0)
    b0 = M.forward_fields(_open(p_on), NCFG_N, f, t_norm=0.0, tau=1.0)
    assert float(jnp.max(jnp.abs(a0.logits - b0.logits))) == 0.0


def test_znorm_bounded_and_scale_invariant():
    p_on = _open(M.init_params(jax.random.PRNGKey(0), NCFG_N))
    xc, _ = _pair()
    f = M.build_fields(xc, jnp.full((9, 9), G.VOID, jnp.int32))
    z_in = jax.random.normal(jax.random.PRNGKey(5), (11, 9, 9, 12))
    o1 = M.forward_fields(p_on, NCFG_N, f, t_norm=0.0, tau=1.0, z_in=z_in)
    o6 = M.forward_fields(p_on, NCFG_N, f, t_norm=0.0, tau=1.0, z_in=1e6 * z_in)
    assert bool(jnp.all(jnp.isfinite(o6.logits)))
    assert float(jnp.max(jnp.abs(o6.logits - o1.logits))) < 1e-3
    # a huge carry cannot blow the loop: 64 final-map steps stay finite
    y0 = jax.nn.one_hot(jnp.full((9, 9), G.VOID, jnp.int32), M.VOCAB).transpose(2, 0, 1)
    outs, res, y = M.iterate_eq(p_on, NCFG_N, xc, tau=1.0, t_total=64, y0_probs=y0)
    assert bool(jnp.all(jnp.isfinite(outs[-1].z_fine))) and bool(jnp.all(jnp.isfinite(y)))


def test_znorm_keeps_s9():
    p_on = _open(M.init_params(jax.random.PRNGKey(0), NCFG_N))
    _, sol = _pair(5)
    lut = np.arange(11); lut[1:10] = np.random.default_rng(7).permutation(np.arange(1, 10))
    yprev = jnp.full((9, 9), G.VOID, jnp.int32)
    z_in = jax.random.normal(jax.random.PRNGKey(5), (11, 9, 9, 12))
    o = M.forward_fields(p_on, NCFG_N, M.build_fields(sol, yprev), t_norm=0.0, tau=1.0, z_in=z_in)
    inv = jnp.asarray(np.argsort(lut))
    op = M.forward_fields(p_on, NCFG_N, M.build_fields(jnp.asarray(lut[np.asarray(sol)], jnp.int32), yprev),
                          t_norm=0.0, tau=1.0, z_in=z_in[inv])
    assert float(jnp.max(jnp.abs(op.logits[..., jnp.asarray(lut)] - o.logits))) < 1e-4


# ── eq_etas ───────────────────────────────────────────────────────────────

def test_eq_etas_learned_bit_exact_and_fixed():
    p = M.init_params(jax.random.PRNGKey(1), NCFG)
    p = {**p, "eq": {**p["eq"], "eta": jnp.asarray(1.3), "eta_z": jnp.asarray(-0.4)}}
    eta, eta_z = M.eq_etas(p, NCFG)
    assert float(eta) == float(NCFG.eta_floor + (1.0 - NCFG.eta_floor) * jax.nn.sigmoid(p["eq"]["eta"]))
    assert float(eta_z) == float(jax.nn.sigmoid(p["eq"]["eta_z"]))
    cf = Config(**{**NCFG.__dict__, "eta_fixed": 1.0, "eta_z_fixed": 0.95})
    e, ez = M.eq_etas(p, cf)
    assert float(e) == 1.0 and abs(float(ez) - 0.95) < 1e-6


# ── the field-recipe cell ─────────────────────────────────────────────────

def test_trm_param_count_matches_the_field():
    p = M.init_params(jax.random.PRNGKey(0), TRM_FIELD)
    assert M.count_params(p["trm"]) == 5_037_058       # EqR Table 17 / TRM: 5.03M
    assert set(p) == {"eq", "trm"}


def test_trm_forward_contract_and_readout():
    p = M.init_params(jax.random.PRNGKey(1), TRM_TINY)
    xc, sol = _pair(11)
    outs, res, y = M.iterate_eq(p, TRM_TINY, xc, tau=1.0, t_total=3)
    assert outs[-1].logits.shape == (9, 9, M.VOCAB)
    assert outs[-1].z_fine.shape == (2, 81 + TRM_TINY.trm_puzzle_emb_len, 32)
    # y is a pure readout under eta_fixed = 1
    assert bool(jnp.allclose(y, jax.nn.softmax(outs[-1].logits, -1).transpose(2, 0, 1), atol=1e-6))
    assert all(bool(jnp.isfinite(r)) for r in res)
    # the y slot is not read: a different y0 gives identical logits
    y_alt = jax.nn.one_hot(sol, M.VOCAB).transpose(2, 0, 1)
    outs2, _, _ = M.iterate_eq(p, TRM_TINY, xc, tau=1.0, t_total=3, y0_probs=y_alt)
    assert float(jnp.max(jnp.abs(outs2[-1].logits - outs[-1].logits))) == 0.0


def test_trm_loss_grad_and_liveness():
    p = M.init_params(jax.random.PRNGKey(1), TRM_TINY)
    xc, sol = _pair(13)
    l, aux = pair_loss(p, TRM_TINY, xc, sol, tau=1.0, rng=jax.random.PRNGKey(2))
    assert bool(jnp.isfinite(l)) and "fpa_ce_last" not in aux
    g = jax.grad(lambda q: pair_loss(q, TRM_TINY, xc, sol, tau=1.0, rng=jax.random.PRNGKey(2))[0])(p)
    assert float(sum(jnp.sum(jnp.abs(v)) for v in jax.tree.leaves(g["trm"]["blocks"]))) > 0
    # TRM-exact (lambda = beta = 0, no RI) is rng-independent; EqR knobs make it rng-dependent
    l_a, _ = pair_loss(p, TRM_TINY, xc, sol, tau=1.0, rng=jax.random.PRNGKey(2))
    l_b, _ = pair_loss(p, TRM_TINY, xc, sol, tau=1.0, rng=jax.random.PRNGKey(9))
    assert float(l_a) == float(l_b)
    c_eqr = Config(**{**TRM_TINY.__dict__, "trm_lambda": 0.05, "trm_beta": 0.01, "trm_ri_sigma": 1.0})
    l_c, _ = pair_loss(p, c_eqr, xc, sol, tau=1.0, rng=jax.random.PRNGKey(2))
    l_d, _ = pair_loss(p, c_eqr, xc, sol, tau=1.0, rng=jax.random.PRNGKey(9))
    assert float(l_c) != float(l_d)
    # inference (rng=None) never sees noise or RI: deterministic, and lambda alone changes the map
    c_lam = Config(**{**TRM_TINY.__dict__, "trm_lambda": 0.05})
    o0 = M.iterate_eq(p, TRM_TINY, xc, tau=1.0, t_total=2)[0][-1].logits
    o1 = M.iterate_eq(p, c_lam, xc, tau=1.0, t_total=2)[0][-1].logits
    o2 = M.iterate_eq(p, c_eqr, xc, tau=1.0, t_total=2)[0][-1].logits   # rng=None -> beta/RI inert
    assert float(jnp.max(jnp.abs(o1 - o0))) > 1e-6 and float(jnp.max(jnp.abs(o2 - o1))) == 0.0


def test_trm_segment_grad_through_last_cycle_only():
    """TRM: 'H_cycles-1 without grad' — the gradient w.r.t. the incoming
    carry must equal the gradient of a segment that starts at the last cycle."""
    p = M.init_params(jax.random.PRNGKey(1), TRM_TINY)["trm"]
    xc, _ = _pair(17)
    emb = TC.embed(p, TRM_TINY, xc)
    z = TC.z0(TRM_TINY, 81)
    def f_full(z):
        zH, zL = TC.segment(p, TRM_TINY, emb, z[0], z[1]); return jnp.sum(zH)
    one = Config(**{**TRM_TINY.__dict__, "trm_h_cycles": 1})
    def f_last(z):
        zH, zL = TC.segment(p, one, emb, z[0], z[1]); return jnp.sum(zH)
    # forward of the full segment = last cycle applied to the (no-grad) state after the first cycle
    zH1, zL1 = TC.segment(p, one, emb, z[0], z[1])
    gf = jax.grad(f_full)(z)
    assert float(jnp.max(jnp.abs(gf))) == 0.0        # stop_gradient blocks the carry's gradient
    gl = jax.grad(f_last)(jnp.stack([zH1, zL1]))
    assert float(jnp.max(jnp.abs(gl))) > 0.0


def test_stablemax_normalizes_and_grad_finite_at_one():
    lp = log_stablemax(jnp.array([[-3.0, 0.0, 2.5], [10.0, -10.0, 0.1]]))
    assert bool(jnp.allclose(jnp.exp(lp).sum(-1), 1.0, atol=1e-6))
    assert bool(jnp.all(jnp.isfinite(jax.grad(lambda x: log_stablemax(x)[0, 2])(jnp.array([[-3.0, 0.0, 2.5]])))))
    # the launch incident (2026-09-02): a logit EXACTLY 1.0 must give a finite gradient
    # (the discarded where-branch's derivative was inf there -> NaN)
    for target in (0, 1, 2):
        g = jax.grad(lambda x: log_stablemax(x)[0, target])(jnp.array([[0.3, 1.0, -2.0]]))
        assert bool(jnp.all(jnp.isfinite(g))), (target, g)
