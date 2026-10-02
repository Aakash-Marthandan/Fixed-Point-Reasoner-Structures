# SE-RRM ATTRIBUTION ROUND 2 (2026-10-02): cfg.dec_single_state = SE-RRM's single-state recurrence inside the DEC.
# (1) the default (two-state) segment is BIT-EXACT against an independent replica of the pre-existing loop, with and
#     without path noise; (2) the single-state segment applies the stack H*(L+1) times to one carry with the input
#     injected each time, returns the fast slot untouched, and matches its own independent replica; (3) gradients
#     reach the blocks through the last H-cycle only; (4) the field round-trips through the saved-config path.
import sys
from dataclasses import asdict
from pathlib import Path as _P

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "src"))
from qhrrn2 import dec_cell as DC
from qhrrn2.config import Config

FIELD = dict(canvas=9, scales=2, pool_arity=3, mixer_kind="group9", attn_max_hw=9,
             equilibrium=True, sudoku_layout="native9", T=16, eta_fixed=1.0, eta_z_fixed=1.0, loss_kind="stablemax")
TINY = dict(**{**FIELD, "T": 3}, cell_kind="dec", dec_width=16, trm_layers=1, trm_h_cycles=2, trm_l_cycles=3)
TINY_ATTN = dict(TINY, dec_width=32, dec_token_mixer="attn", dec_tok_dk=16, dec_coupling_kind="attn", dec_attn_heads=2, dec_attn_dk=16)


def _setup(cfg, seed=0):
    k = jax.random.split(jax.random.PRNGKey(seed), 4)
    p = DC.init_params(k[0], cfg)
    shp = (DC.F, 81, cfg.dec_width)
    emb = jax.random.normal(k[1], shp); zH = jax.random.normal(k[2], shp); zL = jax.random.normal(k[3], shp)
    return p, emb, zH, zL


def _replica_two_state(p, cfg, emb, zH, zL, rng=None):
    """An independent copy of the pre-existing DEC loop (2026-09-18 code), for the bit-exactness check."""
    lam, beta = cfg.trm_lambda, cfg.trm_beta
    n = cfg.trm_h_cycles * (cfg.trm_l_cycles + 1)
    keys = list(jax.random.split(rng, n)) if (rng is not None and beta > 0) else [None] * n
    def step(z, inj, key):
        Fz = DC._stack(p, z + inj, cfg)
        z2 = (z + (1.0 - lam) * (Fz - z)) if lam > 0 else Fz
        return z2 + beta * jax.random.normal(key, z2.shape) if key is not None else z2
    k = 0
    for c in range(cfg.trm_h_cycles):
        for _ in range(cfg.trm_l_cycles):
            zL = step(zL, zH + emb, keys[k]); k += 1
        zH = step(zH, zL, keys[k]); k += 1
        if c < cfg.trm_h_cycles - 1:
            zH, zL = jax.lax.stop_gradient(zH), jax.lax.stop_gradient(zL)
    return zH, zL


def _replica_single_state(p, cfg, emb, zH, zL, rng=None):
    lam, beta = cfg.trm_lambda, cfg.trm_beta
    n = cfg.trm_h_cycles * (cfg.trm_l_cycles + 1)
    keys = list(jax.random.split(rng, n)) if (rng is not None and beta > 0) else [None] * n
    z = zH
    for i in range(n):
        Fz = DC._stack(p, z + emb, cfg)
        z = (z + (1.0 - lam) * (Fz - z)) if lam > 0 else Fz
        if keys[i] is not None: z = z + beta * jax.random.normal(keys[i], z.shape)
    return z, zL


def test_default_is_bit_exact():
    for base in (TINY, TINY_ATTN):
        for lam, beta, rng in ((0.05, 0.0, None), (0.05, 0.01, jax.random.PRNGKey(7)), (0.0, 0.0, None)):
            cfg = Config(**base, trm_lambda=lam, trm_beta=beta)
            assert cfg.dec_single_state is False
            p, emb, zH, zL = _setup(cfg)
            a = DC.segment(p, cfg, emb, zH, zL, rng=rng); b = _replica_two_state(p, cfg, emb, zH, zL, rng=rng)
            assert np.array_equal(np.asarray(a[0]), np.asarray(b[0])) and np.array_equal(np.asarray(a[1]), np.asarray(b[1]))


def test_single_state_matches_replica_and_leaves_fast_slot():
    for base in (TINY, TINY_ATTN):
        for lam, beta, rng in ((0.05, 0.0, None), (0.05, 0.01, jax.random.PRNGKey(7))):
            cfg = Config(**base, trm_lambda=lam, trm_beta=beta, dec_single_state=True)
            p, emb, zH, zL = _setup(cfg)
            z, zL2 = DC.segment(p, cfg, emb, zH, zL, rng=rng); zr, _ = _replica_single_state(p, cfg, emb, zH, zL, rng=rng)
            assert np.array_equal(np.asarray(zL2), np.asarray(zL))                      # the fast slot passes through untouched
            assert np.allclose(np.asarray(z), np.asarray(zr), atol=1e-6) and np.isfinite(np.asarray(z)).all()
            two = DC.segment(p, Config(**base, trm_lambda=lam, trm_beta=beta), emb, zH, zL, rng=rng)[0]
            assert not np.allclose(np.asarray(z), np.asarray(two))                     # a genuinely different recurrence


def test_single_state_stack_count():
    cfg = Config(**TINY, dec_single_state=True); p, emb, zH, zL = _setup(cfg)
    calls = []; orig = DC._stack
    try:
        DC._stack = lambda p_, h_, cfg_=None: (calls.append(1), orig(p_, h_, cfg_))[1]
        DC.segment(p, cfg, emb, zH, zL)
    finally:
        DC._stack = orig
    assert len(calls) == cfg.trm_h_cycles * (cfg.trm_l_cycles + 1)


def test_single_state_gradient_reaches_blocks_through_last_cycle():
    cfg = Config(**TINY, dec_single_state=True); p, emb, zH, zL = _setup(cfg)
    def loss(pp, zh):
        z, _ = DC.segment(pp, cfg, emb, zh, zL)
        return jnp.sum(DC.readout(pp, cfg, z, (9, 9))[0][..., 1:10] ** 2)
    gp, gz = jax.grad(loss, argnums=(0, 1))(p, zH)
    leaves = jax.tree_util.tree_leaves(gp["blocks"])
    assert all(np.isfinite(np.asarray(g)).all() for g in leaves) and any(float(jnp.abs(g).max()) > 0 for g in leaves)
    assert float(jnp.abs(gz).max()) == 0.0                                              # stop_gradient before the last cycle


def test_config_roundtrip():
    cfg = Config(**TINY, dec_single_state=True); d = asdict(cfg); d0 = Config()
    back = Config(**{k: type(getattr(d0, k))(v) if getattr(d0, k) is not None else v for k, v in d.items()})
    assert back.dec_single_state is True and Config().dec_single_state is False


def test_records_omit_the_field_when_off():
    """An unflagged run's records (config.json's argv and config, the checkpoints' config) carry exactly the pre-existing keys:
    the frozen analyzers and tools/resume_flags_guard.py compare them key by key against banked runs. The flagged run records True,
    and the evaluators' rebuild (eval_sudoku_extreme.py: Config(**{k: type(default)(v)})) recovers the field from either record."""
    sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "tools"))
    import pretrain as PT
    d0 = Config()
    rebuild = lambda rec: Config(**{k: type(getattr(d0, k))(v) if getattr(d0, k) is not None else v for k, v in rec.items()})
    off, on = Config(**TINY), Config(**TINY, dec_single_state=True)
    r_off, r_on = PT.cfg_record(off), PT.cfg_record(on)
    assert "dec_single_state" not in r_off and r_off == {k: v for k, v in asdict(off).items() if k != "dec_single_state"}
    assert r_on["dec_single_state"] is True and r_on == asdict(on)
    assert rebuild(r_off) == off and rebuild(r_on) == on
    old = sys.argv
    try:
        sys.argv = ["pretrain.py", "--out", "x"]; v_off = vars(PT.parse_args())
        sys.argv = ["pretrain.py", "--out", "x", "--dec-single-state"]; v_on = vars(PT.parse_args())
    finally:
        sys.argv = old
    assert "dec_single_state" not in v_off and v_on["dec_single_state"] is True
    assert {k: v for k, v in v_on.items() if k != "dec_single_state"} == v_off
