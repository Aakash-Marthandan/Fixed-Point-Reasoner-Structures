# Ledger: CHAMPION TRACK CI gates (Plan_2026-09-01_Champion_Track §2.7) —
# the 3-adic native9 geometry: S9 bit-exactness on 9x9, box-pool alignment
# (level-1 pooling blocks ARE the Sudoku boxes), group-mixer contracts,
# native end-to-end (iterate_eq / FPA / RI rows), native corpus, and the new
# standing evaluator stats (true B=1; Top-1-residual@k). The canvas32 path's
# inertness guard is the EXISTING suite (defaults unchanged = old graphs).
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from qhrrn2 import cell
from qhrrn2 import grid as G
from qhrrn2 import sudoku as SU
from qhrrn2 import sudoku_extreme as SX
from qhrrn2 import episodic as E
from qhrrn2.config import Config
from qhrrn2.model import (build_fields, forward_fields, init_params, iterate_eq,
                          count_params)
from qhrrn2.objective import pair_loss

NCFG = Config(d=12, d_ir=16, d_code=16, K=16, T=2, canvas=9, scales=2,
              pool_arity=3, mixer_kind="group9", attn_max_hw=9,
              equilibrium=True, sudoku_layout="native9")


@pytest.fixture(scope="module")
def nparams():
    return init_params(jax.random.PRNGKey(0), NCFG)


def _pair(seed=0, givens=40):
    rng = np.random.default_rng(seed)
    return SU.sample(rng, givens)


# ── geometry contracts ─────────────────────────────────────────────────────

def test_layout_native9_roundtrip():
    p, s = _pair()
    assert SU.layout_extent("native9") == 9
    assert SU.layout_canvas("native9") == 9
    assert SU.layout_canvas("origin") == G.CANVAS
    placed = SU.place_layout(s, "native9")
    assert placed.shape == (9, 9) and np.array_equal(placed, s)
    assert np.array_equal(SU.unplace_layout(placed, "native9"), s)


def test_box_pool_alignment():
    """Each level-1 pooling block must gather EXACTLY one Sudoku box."""
    d = 4
    z = np.zeros((2, 9, 9, d), np.float32)
    for bi in range(3):
        for bj in range(3):
            z[:, 3 * bi:3 * bi + 3, 3 * bj:3 * bj + 3, :] = 10 * bi + bj
    # reproduce pool_split's gather (reshape+transpose) and check each pooled
    # site sees 9 copies of its box's constant and nothing else
    a = 3
    u = jnp.asarray(z).reshape(2, 3, a, 3, a, d).transpose(0, 1, 3, 2, 4, 5)
    u = np.asarray(u.reshape(2, 3, 3, a * a * d))
    for bi in range(3):
        for bj in range(3):
            assert np.all(u[:, bi, bj, :] == 10 * bi + bj)
    # and the module runs with matching shapes
    p3 = cell.init_pool_split(jax.random.PRNGKey(1), d, 3, arity=3)
    kept, mu, ls = cell.pool_split(p3, jnp.asarray(z), 3, arity=3)
    assert kept.shape == (2, 3, 3, d) and mu.shape == (2, 3, 3, 3)


def test_group_mixer_box_view_and_residual():
    d = 6
    key = jax.random.PRNGKey(2)
    p = cell.init_group_mixer(key, d)
    z = jax.random.normal(jax.random.PRNGKey(3), (11, 9, 9, d))
    out = cell.group_mixer(p, z)
    assert out.shape == z.shape and bool(jnp.all(jnp.isfinite(out)))
    # zeroed l2 -> the operator is residual-inert (out == z exactly)
    p0 = jax.tree.map(jnp.zeros_like, p)
    assert bool(jnp.all(cell.group_mixer(p0, z) == z))
    # s1 branch: the 3x3 box grid as one group
    z1 = jax.random.normal(jax.random.PRNGKey(4), (11, 3, 3, d))
    out1 = cell.group_mixer(p, z1)
    assert out1.shape == z1.shape and bool(jnp.all(jnp.isfinite(out1)))


def test_native_s9_equivariance(nparams):
    """Digit relabeling (Sudoku's symmetry) must permute logits exactly —
    the [P-C1] guarantee carried into native geometry."""
    _, s = _pair(5)
    lut = np.arange(11)
    lut[1:10] = np.random.default_rng(7).permutation(np.arange(1, 10))
    xc = jnp.asarray(s, jnp.int32)                       # a full grid as input
    xp = jnp.asarray(lut[s], jnp.int32)
    yprev = jnp.full((9, 9), G.VOID, dtype=jnp.int32)
    out = forward_fields(nparams, NCFG, build_fields(xc, yprev), t_norm=0.0, tau=1.0)
    out_p = forward_fields(nparams, NCFG, build_fields(xp, yprev), t_norm=0.0, tau=1.0)
    gathered = out_p.logits[..., jnp.asarray(lut)]
    err = float(jnp.max(jnp.abs(gathered - out.logits)))
    assert err < 1e-4, f"native S9 violated: {err}"
    assert float(jnp.max(jnp.abs(out_p.rule_q - out.rule_q))) < 1e-4
    assert float(jnp.max(jnp.abs(out_p.flux - out.flux))) < 1e-3
    assert float(jnp.max(jnp.abs(out_p.flux_attn - out.flux_attn))) < 1e-3
    # two cuts only, both ledgers present per cut
    assert out.flux.shape == (2,) and out.flux_attn.shape == (2,)


# ── native end-to-end ──────────────────────────────────────────────────────

def test_native_iterate_eq(nparams):
    p, _ = _pair(11)
    outs, res, y = iterate_eq(nparams, NCFG, jnp.asarray(p, jnp.int32),
                              tau=1.0, t_total=3)
    assert len(outs) == 3 and y.shape == (11, 9, 9)
    assert all(np.isfinite(float(r)) for r in res)
    assert outs[-1].logits.shape == (9, 9, 11)


def test_native_fpa_and_ri_rows(nparams):
    p, s = _pair(13)
    cfg = Config(**{**NCFG.__dict__, "fpa_k": 2})
    loss, aux = pair_loss(nparams, cfg, jnp.asarray(p, jnp.int32),
                          jnp.asarray(s, jnp.int32), tau=1.0,
                          rng=jax.random.PRNGKey(5))
    assert np.isfinite(float(loss)) and "fpa_ce_last" in aux
    y_b = jnp.asarray(np.stack([s, s]), jnp.int32)
    ks = jax.random.split(jax.random.PRNGKey(6), 5)
    yp = E.build_y0_rows(ks[0], ks[1], ks[2], y_b, 0.3, 0.15, ri_p=0.5,
                         k_r1=ks[3], k_r2=ks[4])
    assert yp.shape == (2, 9, 9)
    assert int(jnp.min(yp)) >= 0 and int(jnp.max(yp)) <= 10


def test_native_param_count_class():
    """The plan's param-trap guard: group9 at ws6/d96 must stay in the
    ~1.6-1.9M band (a naive 3x3 concat mixer would sit ~3.9M)."""
    ws = 6
    cfg = Config(d=96, d_b=6 * ws, d_a=6 * ws, d_ir=32 * ws, d_code=32 * ws,
                 d_task=32 * ws, K=64, T=2, canvas=9, scales=2, pool_arity=3,
                 mixer_kind="group9", attn_max_hw=9, equilibrium=True,
                 sudoku_layout="native9")
    n = count_params(init_params(jax.random.PRNGKey(0), cfg))
    assert 1_500_000 < n < 2_000_000, n


# ── native corpus ──────────────────────────────────────────────────────────

def _tiny_npz(tmp_path):
    pairs = [SU.sample(np.random.default_rng(i), 40) for i in range(9)]
    q = np.stack([p for p, _ in pairs]); a = np.stack([s for _, s in pairs])
    out = tmp_path / "tiny_sx.npz"
    np.savez_compressed(
        out,
        train_q=q[:3], train_a=a[:3], train_rating=np.zeros(3, np.int32),
        train_row=np.arange(3, dtype=np.int64),
        val_q=q[3:5], val_a=a[3:5], val_rating=np.zeros(2, np.int32),
        test_q=q[5:], test_a=a[5:], test_rating=np.arange(4, dtype=np.int32),
        test_source=np.zeros(4, np.int16), source_names=np.asarray(["t"]),
        meta=np.asarray([repr(dict(seed=0, k=3, n_val=2))]))
    return out


def test_native_corpus(tmp_path):
    npz = _tiny_npz(tmp_path)
    corpus, val = SX.build_corpus_extreme(npz, n_aug=2, seed=0, layout="native9")
    assert corpus.x.shape == (9, 9, 9) and corpus.y.shape == (9, 9, 9)
    assert int(corpus.bound_h.max()) == 0 and int(corpus.bound_w.max()) == 0
    for y in corpus.y:
        assert SU.is_valid_solution(np.asarray(y, np.int8))
    for x, y in zip(corpus.x, corpus.y):
        assert SU.agrees_on_givens(np.asarray(x, np.int8), np.asarray(y, np.int8))
    assert len(val[0][2]) == 2


# ── the new standing evaluator stats ───────────────────────────────────────

def test_summarize_b1_and_t1r(tmp_path):
    import eval_sudoku_extreme as EV
    n, k = 6, 4
    ex = np.zeros((n, k), np.uint8)
    re = np.ones((n, k), np.float16)
    # puzzle 0: draw0 exact & smallest resid -> counted at every k
    ex[0, 0] = 1; re[0, 0] = 0.1
    # puzzle 1: draw2 exact with smallest resid -> counted only at k>=4
    ex[1, 2] = 1; re[1, 2] = 0.05
    # puzzle 2: draw1 exact but draw3 has SMALLER resid -> t1r@4 misses it
    ex[2, 1] = 1; re[2, 1] = 0.5; re[2, 3] = 0.01
    arr = dict(idx=np.arange(n), rating=np.zeros(n, np.int64),
               cold_exact=np.zeros(n, bool), first_exact=np.full(n, -1),
               first_valid=np.full(n, -1), violations=np.ones(n, np.int64),
               cells=np.full(n, 50), givens_kept=np.full(n, 30),
               mi_verified=ex.sum(1), mi_true=ex.sum(1),
               mi_first_hit=np.where(ex.any(1), ex.argmax(1), -1),
               mi_exact_k=ex, mi_resid_k=re)
    Q = np.zeros((n, 9, 9), np.int8); Q[:, 0, 0] = 1
    qs = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8], float)
    s = EV.summarize(arr, Q, np.arange(n), qs, dict(k_init=k))
    assert abs(s["b1_exact"] - 1 / 6) < 1e-9
    assert abs(s["t1r_at_k"]["1"] - 1 / 6) < 1e-9          # only puzzle 0
    # k=2: p2's correct draw1 wins argmin (its mis-director draw3 not yet seen)
    assert abs(s["t1r_at_k"]["2"] - 2 / 6) < 1e-9
    # k=4: p1 arrives BUT p2 is now mis-selected by the smaller-resid wrong
    # draw — Top-1-residual selection is NON-monotone in k (the EqR-statistic
    # property the verified funnel does not share)
    assert abs(s["t1r_at_k"]["4"] - 2 / 6) < 1e-9
    # verified vote sees p2's hit that residual selection missed
    assert s["exact_acc_vote"] == pytest.approx(3 / 6)
