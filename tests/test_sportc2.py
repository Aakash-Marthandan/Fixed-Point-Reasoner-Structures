# Ledger: sportC2 build gates (Freethink 2026-09-03 X-4/X-7; registration
# Plan_2026-09-04_sportC2). The three grafts on our cell are cfg-gated and
# BIT-EXACT at their defaults: (1) inner cycles (cfg.inner_k latent passes per
# outer step; K=1 = the pre-existing graph), (2) hard-decision feedback rows
# (cfg.hard_p; 0 = the registered rng stream and graph; 1 = one-hot feedback with
# a finite straight-through gradient), (3) the carried-(y, z) entry for the SOT
# loop (z0 / return_z / traced per-row schedule selector): a continued rollout
# equals the corresponding slice of one long rollout under the final map.
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from qhrrn2 import grid as G
from qhrrn2 import model as M
from qhrrn2 import sudoku as SU
from qhrrn2.config import Config
from qhrrn2.objective import pair_loss

NCFG = Config(d=12, d_ir=16, d_code=16, K=16, T=3, canvas=9, scales=2,
              pool_arity=3, mixer_kind="group9", attn_max_hw=9,
              equilibrium=True, sudoku_layout="native9", z_norm="rms")


def _cfg(**kw):
    return Config(**{**NCFG.__dict__, **kw})


def _pair(seed=3):
    puz, sol = SU.sample(np.random.default_rng(seed), 40)
    return jnp.asarray(puz, jnp.int32), jnp.asarray(sol, jnp.int32)


def _params(cfg, seed=0):
    return M.init_params(jax.random.PRNGKey(seed), cfg)


def _open(p, a=0.5):
    return {**p, "eq": {**p["eq"], "alpha_z": jnp.asarray(a)}}


def test_inner_k_default_bit_exact_and_k3_live():
    xc, _ = _pair(); p = _params(NCFG)
    o1, r1, y1 = M.iterate_eq(p, NCFG, xc, tau=1.0, t_total=4)
    o1b, r1b, y1b = M.iterate_eq(p, _cfg(inner_k=1), xc, tau=1.0, t_total=4)
    assert np.array_equal(np.asarray(y1), np.asarray(y1b))
    # liveness needs the carried-z gate OPEN (alpha_z = 0 at init makes the latent inert, as in test_sportc1)
    po = _open(p)
    _, _, y1o = M.iterate_eq(po, NCFG, xc, tau=1.0, t_total=4)
    o3, r3, y3 = M.iterate_eq(po, _cfg(inner_k=3), xc, tau=1.0, t_total=4)
    assert len(o3) == 4 and np.isfinite(np.asarray(y3)).all()
    assert not np.allclose(np.asarray(y1o), np.asarray(y3))         # the extra passes change the readout


def test_hard_p_zero_bit_exact_and_hard_one_feeds_one_hot():
    xc, ys = _pair(); p = _params(NCFG); k = jax.random.PRNGKey(7)
    _, _, y0 = M.iterate_eq(p, NCFG, xc, tau=1.0, t_total=3, rng=k)
    _, _, y0b = M.iterate_eq(p, _cfg(hard_p=0.0), xc, tau=1.0, t_total=3, rng=k)
    assert np.array_equal(np.asarray(y0), np.asarray(y0b))          # rng stream untouched at hard_p = 0
    cfg_h = _cfg(hard_p=1.0)
    outs, _, yh = M.iterate_eq(p, cfg_h, xc, tau=1.0, t_total=1, rng=k)
    eta, _ = M.eq_etas(p, cfg_h)
    # after one hard step from VOID: y = void + eta*(onehot - void) -> each cell's mass sits on
    # exactly two vocab entries (VOID and the argmax), the argmax entry at eta
    ph = jax.nn.one_hot(jnp.argmax(outs[0].logits, axis=-1), M.VOCAB).transpose(2, 0, 1)
    vh = jax.nn.one_hot(jnp.full(xc.shape, G.VOID, jnp.int32), M.VOCAB).transpose(2, 0, 1)
    expect = vh + eta * (ph - vh)
    assert np.allclose(np.asarray(yh), np.asarray(expect), atol=1e-6)
    # straight-through: a finite, non-zero gradient reaches the parameters through the hard step
    def loss(pp):
        tot, _ = pair_loss(pp, cfg_h, xc, ys, tau=1.0, rng=k)
        return tot
    g = jax.grad(loss)(p)
    leaves = jax.tree.leaves(g)
    assert all(np.isfinite(np.asarray(l)).all() for l in leaves)
    assert sum(float(jnp.sum(jnp.abs(l))) for l in leaves) > 0


def test_carry_continuation_matches_long_rollout_under_final_map():
    xc, ys = _pair(); p = _open(_params(NCFG))
    # one long rollout on the final map (t_norm_fixed = 1) vs the same split in two carried segments
    y0 = jax.nn.one_hot(jnp.full(xc.shape, G.VOID, jnp.int32), M.VOCAB).transpose(2, 0, 1)
    _, _, y_long, z_long = M.iterate_eq(p, NCFG, xc, tau=1.0, t_total=6, y0_probs=y0, t_norm_fixed=1.0, return_z=True)
    _, _, y_a, z_a = M.iterate_eq(p, NCFG, xc, tau=1.0, t_total=3, y0_probs=y0, t_norm_fixed=1.0, return_z=True)
    _, _, y_b, z_b = M.iterate_eq(p, NCFG, xc, tau=1.0, t_total=3, y0_probs=y_a, z0=z_a, t_norm_fixed=1.0, return_z=True)
    assert np.allclose(np.asarray(y_long), np.asarray(y_b), atol=1e-5)
    assert np.allclose(np.asarray(z_long), np.asarray(z_b), atol=1e-4)
    # traced per-row selector: -1 -> the ramp (equals the default path), 1.0 -> the final map
    _, _, y_ramp = M.iterate_eq(p, NCFG, xc, tau=1.0, t_total=3)
    _, _, y_sel = M.iterate_eq(p, NCFG, xc, tau=1.0, t_total=3, t_norm_fixed=jnp.asarray(-1.0))
    assert np.allclose(np.asarray(y_ramp), np.asarray(y_sel), atol=1e-6)
    _, _, y_fix = M.iterate_eq(p, NCFG, xc, tau=1.0, t_total=3, t_norm_fixed=1.0)
    _, _, y_sel1 = M.iterate_eq(p, NCFG, xc, tau=1.0, t_total=3, t_norm_fixed=jnp.asarray(1.0))
    assert np.allclose(np.asarray(y_fix), np.asarray(y_sel1), atol=1e-6)
    # pair_loss returns the carry when continuing
    tot, aux = pair_loss(p, NCFG, xc, ys, tau=1.0, rng=jax.random.PRNGKey(1), y0_probs=y_a, z0=z_a, t_norm_fixed=jnp.asarray(1.0))
    assert np.isfinite(float(tot)) and aux["carry_y"].shape == y_a.shape and aux["carry_z"].shape == z_a.shape


def test_pair_loss_default_path_untouched():
    xc, ys = _pair(); p = _params(NCFG); k = jax.random.PRNGKey(5)
    t1, a1 = pair_loss(p, NCFG, xc, ys, tau=1.0, rng=k)
    t2, a2 = pair_loss(p, NCFG, xc, ys, tau=1.0, rng=k)
    assert float(t1) == float(t2) and "carry_y" not in a1


def test_zero_latent_with_fresh_flag_equals_fresh_rollout():
    """The SOT carry passes a ZERO z0 for fresh rows with z_fresh=True: the forward is the fresh
    forward (a zero latent is inert at the gate, also under z-norm) and the first pass takes
    z <- z_fine — bit-exact with the plain rollout; a carried row (z_fresh=False) blends."""
    xc, ys = _pair(); p = _open(_params(NCFG))
    y0 = jax.nn.one_hot(jnp.full(xc.shape, G.VOID, jnp.int32), M.VOCAB).transpose(2, 0, 1)
    _, _, y_plain, z_plain = M.iterate_eq(p, NCFG, xc, tau=1.0, t_total=3, return_z=True)
    z_zero = jnp.zeros_like(z_plain)
    _, _, y_fresh, z_fresh = M.iterate_eq(p, NCFG, xc, tau=1.0, t_total=3, y0_probs=y0, z0=z_zero,
                                          t_norm_fixed=jnp.asarray(-1.0), z_fresh=jnp.asarray(True), return_z=True)
    assert np.allclose(np.asarray(y_plain), np.asarray(y_fresh), atol=1e-6)
    assert np.allclose(np.asarray(z_plain), np.asarray(z_fresh), atol=1e-5)
    _, _, y_carr, _ = M.iterate_eq(p, NCFG, xc, tau=1.0, t_total=3, y0_probs=y0, z0=z_zero,
                                   t_norm_fixed=jnp.asarray(-1.0), z_fresh=jnp.asarray(False), return_z=True)
    assert not np.allclose(np.asarray(y_plain), np.asarray(y_carr))


def test_trm_group9_token_mixer_graft():
    """X2: 'group9' replaces the field cell's token MLP by our group mixer on a projected view; 'mlp'
    stays TRM-exact (same param tree); the graft runs, is finite, changes the output, and its param
    count at the field dims is pinned (labeled vs the 5,037,058 of the faithful cell)."""
    from qhrrn2 import trm_cell as TC
    from qhrrn2.config import Config as C
    base = dict(canvas=9, scales=2, pool_arity=3, mixer_kind="group9", attn_max_hw=9, equilibrium=True,
                sudoku_layout="native9", cell_kind="trm", T=3, eta_fixed=1.0, eta_z_fixed=1.0, loss_kind="stablemax",
                trm_hidden=32, trm_layers=1, trm_h_cycles=2, trm_l_cycles=2)
    tiny_mlp = C(**base); tiny_gm = C(**{**base, "trm_token_mixer": "group9", "trm_gm_dim": 8})
    xc, _ = _pair()
    p_mlp = M.init_params(jax.random.PRNGKey(0), tiny_mlp); p_gm = M.init_params(jax.random.PRNGKey(0), tiny_gm)
    assert "mlp_t" in p_mlp["trm"]["blocks"][0] and "gm" in p_gm["trm"]["blocks"][0] and "mlp_t" not in p_gm["trm"]["blocks"][0]
    o_mlp = M.iterate_eq(p_mlp, tiny_mlp, xc, tau=1.0, t_total=2)[0][-1].logits
    o_gm = M.iterate_eq(p_gm, tiny_gm, xc, tau=1.0, t_total=2)[0][-1].logits
    assert o_gm.shape == o_mlp.shape and np.isfinite(np.asarray(o_gm)).all()
    assert not np.allclose(np.asarray(o_mlp), np.asarray(o_gm))
    # the field dims: pinned counts (5,037,058 faithful; the graft's count is the labeled comparator)
    field = C(**{**base, "trm_hidden": 512, "trm_layers": 2, "trm_h_cycles": 3, "trm_l_cycles": 6, "T": 16})
    field_gm = C(**{**field.__dict__, "trm_token_mixer": "group9", "trm_gm_dim": 64})
    n_f = M.count_params(TC.init_params(jax.random.PRNGKey(0), field)["trm"] if False else M.init_params(jax.random.PRNGKey(0), field)["trm"])
    n_g = M.count_params(M.init_params(jax.random.PRNGKey(0), field_gm)["trm"])
    assert n_f == 5_037_058, n_f
    assert 5_500_000 < n_g < 6_500_000, n_g
    print("X2 graft params at field dims:", n_g)
