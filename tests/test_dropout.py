# SE-RRM ATTRIBUTION ROUND 3 (2026-10-03): cfg.dec_dropout = SE-RRM's regularizer, dropout on the attention WEIGHTS of both DEC mixers,
# in the TRAINING forward only. (1) dec_dropout 0 (the default) is BIT-EXACT against an independent replica of the pre-existing loop,
# whatever drop_rng is; (2) with dec_dropout > 0, every path that passes no drop_rng (the evaluators, the monitors, noise-at-eval) is
# bit-exact against dec_dropout 0; (3) the training forward drops, deterministically per key, on both state structures; (4) the
# mask keeps 1 - rate of the weights and preserves their expectation; (5) a cell with no attention mixer is untouched; (6) gradients
# flow; (7) the records omit the field at its default and round-trip through the evaluators' rebuild.
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
KEY = jax.random.PRNGKey(11)


def _setup(cfg, seed=0):
    k = jax.random.split(jax.random.PRNGKey(seed), 4)
    p = DC.init_params(k[0], cfg)
    shp = (DC.F, 81, cfg.dec_width)
    return p, jax.random.normal(k[1], shp), jax.random.normal(k[2], shp), jax.random.normal(k[3], shp)


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


def _eq(a, b):
    return all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a, b))


def test_default_is_bit_exact_whatever_drop_rng():
    for base in (TINY, TINY_ATTN):
        for lam, beta, rng in ((0.05, 0.0, None), (0.05, 0.01, jax.random.PRNGKey(7))):
            cfg = Config(**base, trm_lambda=lam, trm_beta=beta)
            assert cfg.dec_dropout == 0.0
            p, emb, zH, zL = _setup(cfg)
            ref = _replica_two_state(p, cfg, emb, zH, zL, rng=rng)
            assert _eq(DC.segment(p, cfg, emb, zH, zL, rng=rng), ref)
            assert _eq(DC.segment(p, cfg, emb, zH, zL, rng=rng, drop_rng=KEY), ref)     # rate 0: the key is ignored


def test_paths_without_drop_rng_never_drop():
    """The evaluators and monitors never pass drop_rng: with dec_dropout 0.2 they equal dec_dropout 0, noise at eval included."""
    for single in (False, True):
        for beta, rng in ((0.0, None), (0.01, jax.random.PRNGKey(7))):
            c0 = Config(**TINY_ATTN, trm_beta=beta, dec_single_state=single)
            c2 = Config(**TINY_ATTN, trm_beta=beta, dec_single_state=single, dec_dropout=0.2)
            p, emb, zH, zL = _setup(c0)
            assert _eq(DC.segment(p, c2, emb, zH, zL, rng=rng), DC.segment(p, c0, emb, zH, zL, rng=rng))
    c0, c2 = Config(**TINY_ATTN), Config(**TINY_ATTN, dec_dropout=0.2)
    p = DC.init_params(jax.random.PRNGKey(3), c0)
    fields = jax.nn.one_hot(jnp.zeros((9, 9), jnp.int32), 12).transpose(2, 0, 1)[..., None].repeat(2, axis=-1)
    a = DC.forward_core(p, c0, fields, rng=jax.random.PRNGKey(5)); b = DC.forward_core(p, c2, fields, rng=jax.random.PRNGKey(5))
    assert all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a, b))


def test_training_forward_drops_deterministically():
    for single in (False, True):
        c0 = Config(**TINY_ATTN, dec_single_state=single); c2 = Config(**TINY_ATTN, dec_single_state=single, dec_dropout=0.2)
        p, emb, zH, zL = _setup(c0)
        off = DC.segment(p, c0, emb, zH, zL)
        on1 = DC.segment(p, c2, emb, zH, zL, drop_rng=KEY); on1b = DC.segment(p, c2, emb, zH, zL, drop_rng=KEY)
        on2 = DC.segment(p, c2, emb, zH, zL, drop_rng=jax.random.PRNGKey(12))
        assert _eq(on1, on1b)                                                         # deterministic per key
        assert not np.allclose(np.asarray(on1[0]), np.asarray(off[0]))                # it drops
        assert not np.allclose(np.asarray(on1[0]), np.asarray(on2[0]))                # a fresh key, a fresh mask
        assert np.isfinite(np.asarray(on1[0])).all()


def test_mask_statistics():
    att = jax.nn.softmax(jax.random.normal(jax.random.PRNGKey(1), (4, 81, 81)), axis=-1)
    d = DC._drop(att, 0.2, jax.random.PRNGKey(2))
    kept = np.asarray(d) > 0
    assert abs(kept.mean() - 0.8) < 0.01                                              # 1 - rate kept
    assert np.allclose(np.asarray(d)[kept], np.asarray(att / 0.8)[kept])              # inverted scaling
    many = jnp.mean(jnp.stack([DC._drop(att, 0.2, jax.random.PRNGKey(i)) for i in range(400)]), axis=0)
    assert float(jnp.max(jnp.abs(many - att))) < 0.25 * float(jnp.max(att)) and abs(float(jnp.mean(many - att))) < 1e-3


def test_no_attention_mixer_is_untouched():
    c0, c2 = Config(**TINY), Config(**TINY, dec_dropout=0.2)                          # the MLP token mixer + the mean coupling
    p, emb, zH, zL = _setup(c0)
    assert _eq(DC.segment(p, c2, emb, zH, zL, drop_rng=KEY), DC.segment(p, c0, emb, zH, zL))


def test_gradients_flow_with_dropout():
    cfg = Config(**TINY_ATTN, dec_dropout=0.2); p, emb, zH, zL = _setup(cfg)
    def loss(pp):
        z, _ = DC.segment(pp, cfg, emb, zH, zL, drop_rng=KEY)
        return jnp.sum(DC.readout(pp, cfg, z, (9, 9))[0][..., 1:10] ** 2)
    leaves = jax.tree_util.tree_leaves(jax.grad(loss)(p)["blocks"])
    assert all(np.isfinite(np.asarray(g)).all() for g in leaves) and any(float(jnp.abs(g).max()) > 0 for g in leaves)


def test_records_and_trainer_hooks():
    sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "tools"))
    import pretrain as PT
    d0 = Config()
    rebuild = lambda rec: Config(**{k: type(getattr(d0, k))(v) if getattr(d0, k) is not None else v for k, v in rec.items()})
    off, on = Config(**TINY_ATTN), Config(**TINY_ATTN, dec_dropout=0.2, dec_single_state=True)
    r_off, r_on = PT.cfg_record(off), PT.cfg_record(on)
    assert "dec_dropout" not in r_off and "dec_single_state" not in r_off and r_off == {k: v for k, v in asdict(off).items() if k not in ("dec_dropout", "dec_single_state")}
    assert r_on["dec_dropout"] == 0.2 and r_on["dec_single_state"] is True and r_on == asdict(on)
    assert rebuild(r_off) == off and rebuild(r_on) == on
    assert PT.cfg_record(Config(**TINY_ATTN, dec_dropout=0.0))["trm_lambda"] == off.trm_lambda and "dec_dropout" not in PT.cfg_record(Config(**TINY_ATTN, dec_dropout=0.0))
    assert PT.drop_kw(off, KEY) == {} and set(PT.drop_kw(on, KEY)) == {"drop_rng"}
    old = sys.argv
    try:
        sys.argv = ["pretrain.py", "--out", "x"]; v_off = vars(PT.parse_args())
        sys.argv = ["pretrain.py", "--out", "x", "--dec-dropout", "0.2"]; v_on = vars(PT.parse_args())
    finally:
        sys.argv = old
    assert "dec_dropout" not in v_off and v_on["dec_dropout"] == 0.2
    assert {k: v for k, v in v_on.items() if k != "dec_dropout"} == v_off
