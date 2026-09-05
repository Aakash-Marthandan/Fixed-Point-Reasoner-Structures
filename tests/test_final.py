# Ledger: FINAL PHASE build gates (Plan_2026-09-05_FinalPhase §2 / §6; 2026-09-05). The DEC cell
# (qhrrn2.dec_cell) is EXACTLY S9-equivariant by weight sharing over the digit fields (a digit
# permutation of the puzzle permutes the logits' digit axis, and the carried state's field axis);
# its parameter count follows the field block's formula (w = 512 lands at X0's count class);
# it runs under the model contract (iterate_eq / evaluator / RI carry shape); the field-loop FPA
# anchor rows (pretrain.field_fpa_loss) are finite and differentiable on both field cells and
# absent at fpa_k = 0; the field cell's answer anchor has the contract shape.
from __future__ import annotations

import sys
from pathlib import Path as _P

import numpy as np
import jax
import jax.numpy as jnp

from qhrrn2 import grid as G
from qhrrn2 import model as M
from qhrrn2 import sudoku as SU
from qhrrn2 import trm_cell as TC
from qhrrn2 import dec_cell as DC
from qhrrn2.config import Config

sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "tools"))

FIELD = dict(canvas=9, scales=2, pool_arity=3, mixer_kind="group9", attn_max_hw=9,
             equilibrium=True, sudoku_layout="native9", T=16, eta_fixed=1.0, eta_z_fixed=1.0,
             loss_kind="stablemax")
DEC_A = Config(**FIELD, cell_kind="dec", dec_width=256)                       # the A-night arm
DEC_512 = Config(**FIELD, cell_kind="dec", dec_width=512)                     # X0's parameter class
DEC_TINY = Config(**{**FIELD, "T": 3}, cell_kind="dec", dec_width=16, trm_layers=1,
                  trm_h_cycles=2, trm_l_cycles=2)
TRM_TINY = Config(**{**FIELD, "T": 3}, cell_kind="trm", trm_hidden=32, trm_layers=1,
                  trm_h_cycles=2, trm_l_cycles=2)


def _pair(seed=3, givens=40):
    puz, sol = SU.sample(np.random.default_rng(seed), givens)
    return jnp.asarray(puz, jnp.int32), jnp.asarray(sol, jnp.int32)


def _perm(seed):
    """A random S9 element as a VOCAB lookup: pi[d] = the image of digit d; 0 and VOID fixed."""
    pi = np.arange(M.VOCAB, dtype=np.int32)
    pi[1:10] = np.random.default_rng(seed).permutation(9) + 1
    return pi


def _swiglu_count(n, expansion=4.0):
    inter = TC._find_multiple(round(expansion * n * 2 / 3), 256)
    return n * 2 * inter + inter * n


def test_dec_exact_s9():
    """The lens's warrant for A3: relabeling the digits relabels the output, bit-for-bit up to
    the float reduction order of the field mean (fixed init buffers, no noise)."""
    x, _ = _pair()
    p = M.init_params(jax.random.PRNGKey(0), DEC_TINY)
    outs, _, _ = M.iterate_eq(p, DEC_TINY, x, tau=1.0, t_total=2)
    for seed in (1, 2):
        pi = _perm(seed)
        xp = jnp.asarray(pi)[x]
        outs_p, _, _ = M.iterate_eq(p, DEC_TINY, xp, tau=1.0, t_total=2)
        for o, op in zip(outs, outs_p):
            lg, lgp = np.asarray(o.logits), np.asarray(op.logits)
            assert np.allclose(lgp[..., pi], lg, atol=1e-4), "logits are not S9-equivariant"
            zf, zfp = np.asarray(o.z_fine), np.asarray(op.z_fine)
            # the permuted state's field (pi[d]-1) equals the original's field (d-1)
            for d in range(1, 10):
                assert np.allclose(zfp[:, pi[d] - 1], zf[:, d - 1], atol=1e-4), "state is not S9-equivariant"
    # the padded classes never win
    lg = np.asarray(outs[-1].logits)
    assert np.all(lg[..., 0] <= -1e3) and np.all(lg[..., G.VOID] <= -1e3)


def test_dec_param_counts_follow_the_field_block_formula():
    """A-night arm (w 256, 2 blocks) and the scale arm (w 512 = X0's class): counts from the formula."""
    def expected(w, layers=2):
        per_block = _swiglu_count(81) + _swiglu_count(w) + w * w
        return layers * per_block + 3 * w + w + (2 * w + 2)
    for cfg in (DEC_A, DEC_512):
        p = M.init_params(jax.random.PRNGKey(0), cfg)
        assert set(p) == {"eq", "dec"}
        assert M.count_params(p["dec"]) == expected(cfg.dec_width), cfg.dec_width
    assert 5.0e6 <= M.count_params(M.init_params(jax.random.PRNGKey(0), DEC_512)["dec"]) <= 5.6e6   # X0: 5.04M
    assert 1.3e6 <= M.count_params(M.init_params(jax.random.PRNGKey(0), DEC_A)["dec"]) <= 1.6e6
    # the coupling ablation drops exactly w*w per block
    no_fc = Config(**{**DEC_A.__dict__, "dec_coupling": False})
    assert M.count_params(M.init_params(jax.random.PRNGKey(0), DEC_A)["dec"]) - \
        M.count_params(M.init_params(jax.random.PRNGKey(0), no_fc)["dec"]) == 2 * 256 * 256


def test_dec_contract_and_carry_shape():
    x, y = _pair()
    p = M.init_params(jax.random.PRNGKey(0), DEC_TINY)
    outs, res, yf, zc = M.iterate_eq(p, DEC_TINY, x, tau=1.0, t_total=3, return_z=True)
    assert outs[-1].logits.shape == (9, 9, M.VOCAB) and np.isfinite(np.asarray(outs[-1].logits)).all()
    assert tuple(zc.shape) == (2, DC.F, 81, 16) == tuple(M.carry_shape(DEC_TINY))
    assert tuple(M.carry_shape(TRM_TINY)) == (2, 81 + TRM_TINY.trm_puzzle_emb_len, 32)
    assert M.carry_shape(Config()) is None
    # RI draws: shape, per-key difference, and the fixed-buffer start is field-identical
    c_ri = Config(**{**DEC_TINY.__dict__, "trm_ri_sigma": 1.0})
    z1 = DC.z0(c_ri, 81, rng=jax.random.PRNGKey(1)); z2 = DC.z0(c_ri, 81, rng=jax.random.PRNGKey(2))
    assert z1.shape == (2, 9, 81, 16) and not np.allclose(np.asarray(z1), np.asarray(z2))
    zf = np.asarray(DC.z0(DEC_TINY, 81))
    assert np.array_equal(zf[:, 0], zf[:, 5])
    # the halting logits are S9-invariant
    _, q, _ = DC.forward_core(p["dec"], DEC_TINY, M.build_fields_soft(x, jax.nn.one_hot(jnp.full((9, 9), G.VOID), M.VOCAB).transpose(2, 0, 1)))
    pi = _perm(4); xp = jnp.asarray(pi)[x]
    _, qp, _ = DC.forward_core(p["dec"], DEC_TINY, M.build_fields_soft(xp, jax.nn.one_hot(jnp.full((9, 9), G.VOID), M.VOCAB).transpose(2, 0, 1)))
    assert np.allclose(np.asarray(q), np.asarray(qp), atol=1e-4)


def test_dec_loss_grad_liveness_and_evaluator():
    import eval_sudoku_extreme as EV
    from qhrrn2.objective import pair_loss
    x, y = _pair()
    p = M.init_params(jax.random.PRNGKey(0), DEC_TINY)
    tv = jnp.zeros((DEC_TINY.d_task,), jnp.float32)
    l, aux = pair_loss(p, DEC_TINY, x, y, tau=1.0, rng=jax.random.PRNGKey(3), task_vec=tv)
    assert np.isfinite(float(l))
    g = jax.grad(lambda p_: pair_loss(p_, DEC_TINY, x, y, tau=1.0, rng=jax.random.PRNGKey(3), task_vec=tv)[0])(p)
    assert float(sum(jnp.sum(jnp.abs(v)) for v in jax.tree.leaves(g["dec"]["blocks"]))) > 0
    # the evaluator's batch step on the DEC cell: exact rows, residual on the LATENT, RI draw shape
    puz, sol = SU.sample(np.random.default_rng(5), 40)
    x_can = EV.place_batch(np.stack([puz]), "native9"); sol9 = np.stack([sol]).astype(np.int32); puz9 = np.stack([puz]).astype(np.int32)
    void = jax.nn.one_hot(jnp.full((9, 9), G.VOID, jnp.int32), M.VOCAB).transpose(2, 0, 1)
    y0 = jnp.broadcast_to(void, (1,) + void.shape)
    eta, eta_z = (float(v) for v in M.eq_etas(p, DEC_TINY))
    assert eta == 1.0 and eta_z == 1.0
    z0r = jnp.asarray(np.stack([EV.mi_z0(7, 0, 0, tuple(M.carry_shape(DEC_TINY)), 1.0)]))
    ex, ok, pred, res = EV.run_batch(p, DEC_TINY, tv, x_can, y0, t_total=2, tau=1.0, gamma=1.0, sol9=sol9, puz9=puz9,
                                     eta=eta, eta_z=eta_z, layout="native9", z0=z0r)
    assert ex.shape == (2, 1) and pred.shape == (1, 9, 9) and np.isfinite(res).all()


def test_field_fpa_anchor_rows_on_both_field_cells():
    import pretrain as PT
    B = 4
    pairs = [SU.sample(np.random.default_rng(s), 40) for s in range(B)]
    x = jnp.asarray(np.stack([pz for pz, _ in pairs]), jnp.int32); y = jnp.asarray(np.stack([s_ for _, s_ in pairs]), jnp.int32)
    for base, TCm in ((DEC_TINY, DC), (TRM_TINY, TC)):
        cfg = Config(**{**base.__dict__, "fpa_k": 2, "fpa_eps": 0.3, "fpa_frac": 0.5})
        p = M.init_params(jax.random.PRNGKey(0), cfg)[cfg.cell_kind]
        f = lambda p_: PT.field_fpa_loss(TCm, p_, cfg, x, y, jax.random.PRNGKey(9), 81)
        v = float(f(p)); assert np.isfinite(v) and v > 0
        g = jax.grad(f)(p)
        assert float(sum(jnp.sum(jnp.abs(t)) for t in jax.tree.leaves(g["blocks"]))) > 0
    # the anchor states have the carry's per-field shape
    p_dec = M.init_params(jax.random.PRNGKey(0), DEC_TINY)["dec"]
    assert DC.embed_answer(p_dec, DEC_TINY, y[0]).shape == (9, 81, 16)
    p_trm = M.init_params(jax.random.PRNGKey(0), TRM_TINY)["trm"]
    za = TC.embed_answer(p_trm, TRM_TINY, y[0]); H0, _ = TC.init_states(TRM_TINY)
    assert za.shape == (81 + TRM_TINY.trm_puzzle_emb_len, 32) and np.allclose(np.asarray(za[0]), np.asarray(H0))
