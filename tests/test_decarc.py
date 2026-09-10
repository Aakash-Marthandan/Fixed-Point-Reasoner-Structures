# Ledger: DEC-ARC BUILD gates (Plan_2026-09-10_DEC-ARC_Build §1 / B6; 2026-09-10). The DEC-ARC cell
# (qhrrn2.decarc_cell) is EXACTLY S10-equivariant in the joint action of a colour permutation on the
# input and on the per-colour task code's field axis (the colour logits permute, the VOID logit and
# the halting logits are invariant, the carried state's field axis permutes); its size decode is the
# crop of the non-void region and the rg cells' decode is bit-exact through the shared helper; it runs
# under the model contract (forward_fields / iterate_eq / pair_loss with a (F, d_task) code, finite
# gradients w.r.t. the code and the parameters); the commit head at tau >= 1 leaves the forward
# untouched; the parameter count follows the block formula; the deployed arm-A fit and the probe
# trace run on it unchanged (the fit's code is (F, d_task); trace == predict).
from __future__ import annotations

import sys
from pathlib import Path as _P

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from qhrrn2 import grid as G
from qhrrn2 import model as M
from qhrrn2 import objective as OBJ
from qhrrn2 import train as T
from qhrrn2 import trm_cell as TC
from qhrrn2 import decarc_cell as DAC
from qhrrn2.config import Config

sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "tools"))

ARC = dict(equilibrium=True, T=3, eta_fixed=1.0, eta_z_fixed=1.0, loss_kind="stablemax", d_task=8)
TINY = Config(**ARC, cell_kind="decarc", dec_width=16, decarc_heads=2, trm_layers=1, trm_h_cycles=2, trm_l_cycles=2)
TINY_COMMIT = Config(**ARC, cell_kind="decarc", dec_width=16, decarc_heads=2, trm_layers=1, trm_h_cycles=2, trm_l_cycles=2,
                     dec_commit=True, dec_commit_tau=1.0)
RG_TINY = Config(d=8, d_b=4, d_a=4, d_ir=8, d_code=8, K=4, M=1, T=2, scales=2, equilibrium=True)


def _grid(seed=0, h=6, w=8):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 10, size=(h, w)).astype(np.int8)
    y = rng.integers(0, 10, size=(h + 1, w - 1)).astype(np.int8)
    return jnp.asarray(G.place(x), jnp.int32), jnp.asarray(G.place(y), jnp.int32)


def _code(seed=1):
    return jax.random.normal(jax.random.PRNGKey(seed), (DAC.F, TINY.d_task)) * 0.5


def _perm(seed):
    """A random S10 element as a VOCAB lookup: pi[c] = the image of colour c; VOID fixed."""
    pi = np.arange(M.VOCAB, dtype=np.int32)
    pi[:10] = np.random.default_rng(seed).permutation(10)
    return pi


def test_decarc_exact_s10():
    """A colour permutation of the input together with the same permutation of the code's field axis
    permutes the colour logits and the state's field axis; the VOID and halting logits are invariant."""
    x, _ = _grid(); code = _code()
    p = M.init_params(jax.random.PRNGKey(0), TINY)
    outs, _, _ = M.iterate_eq(p, TINY, x, tau=1.0, t_total=2, task_vec=code)
    for seed in (1, 2):
        pi = _perm(seed); inv = np.argsort(pi[:10])
        xp = jnp.asarray(pi)[x]
        codep = code[jnp.asarray(inv)]                 # codep[pi[c]] = code[c]
        outs_p, _, _ = M.iterate_eq(p, TINY, xp, tau=1.0, t_total=2, task_vec=codep)
        for o, op in zip(outs, outs_p):
            lg, lgp = np.asarray(o.logits), np.asarray(op.logits)
            assert np.allclose(lgp[..., pi], lg, atol=1e-4), "logits are not S10-equivariant"
            assert np.allclose(lgp[..., G.VOID], lg[..., G.VOID], atol=1e-4), "the VOID logit is not invariant"
            zf, zfp = np.asarray(o.z_fine), np.asarray(op.z_fine)
            for c in range(10):
                assert np.allclose(zfp[:, pi[c]], zf[:, c], atol=1e-4), "state is not S10-equivariant"
    # the halting logits are invariant (read from the field-and-cell mean)
    q0 = DAC.readout(p["decarc"], TINY, outs[-1].z_fine[0], (G.CANVAS, G.CANVAS))[1]
    pi = _perm(3); inv = np.argsort(pi[:10])
    outs_p, _, _ = M.iterate_eq(p, TINY, jnp.asarray(pi)[x], tau=1.0, t_total=2, task_vec=code[jnp.asarray(inv)])
    q1 = DAC.readout(p["decarc"], TINY, outs_p[-1].z_fine[0], (G.CANVAS, G.CANVAS))[1]
    assert np.allclose(np.asarray(q0), np.asarray(q1), atol=1e-4)


def test_decode_size_void_crop_and_rg_identity():
    """The DEC-ARC size = the bounding box from the origin of the non-void argmax; the rg cells' decode is
    the pre-existing mixture expression, bit-exact through the shared helper."""
    lg = jnp.full((G.CANVAS, G.CANVAS, M.VOCAB), -1.0).at[..., G.VOID].set(1.0)
    lg = lg.at[:5, :7, 3].set(5.0)                                   # a 5 x 7 non-void block
    p_h, p_w = DAC.size_from_logits(lg)
    assert int(jnp.argmax(p_h)) + 1 == 5 and int(jnp.argmax(p_w)) + 1 == 7
    p_h, p_w = DAC.size_from_logits(jnp.full((G.CANVAS, G.CANVAS, M.VOCAB), -1.0).at[..., G.VOID].set(1.0))
    assert int(jnp.argmax(p_h)) + 1 == 1 and int(jnp.argmax(p_w)) + 1 == 1   # an all-void canvas -> 1 x 1
    x, _ = _grid(seed=4)
    prg = M.init_params(jax.random.PRNGKey(1), RG_TINY)
    out = M.forward_fields(prg, RG_TINY, M.build_fields(x, jnp.full_like(x, G.VOID)), t_norm=1.0, tau=1.0)
    cands = M.size_candidates(x)
    ref_h = M.size_mixture_probs(out.size_sel_h, out.size_h, cands[0]); ref_w = M.size_mixture_probs(out.size_sel_w, out.size_w, cands[1])
    got_h, got_w = M.decode_size(RG_TINY, out, x)
    assert np.array_equal(np.asarray(ref_h), np.asarray(got_h)) and np.array_equal(np.asarray(ref_w), np.asarray(got_w))


def test_decarc_contract_loss_and_gradients():
    """forward_fields / iterate_eq / pair_loss under the model contract with a (F, d_task) code; finite
    gradients w.r.t. the parameters and the code; the carry has the DEC-ARC shape."""
    x, y = _grid(seed=5); code = _code(2)
    p = M.init_params(jax.random.PRNGKey(0), TINY)
    assert set(p) == {"eq", "decarc"}
    out = M.forward_fields(p, TINY, M.build_fields(x, jnp.full_like(x, G.VOID)), t_norm=1.0, tau=1.0, task_vec=code)
    assert out.logits.shape == (G.CANVAS, G.CANVAS, M.VOCAB)
    assert out.z_fine.shape == M.carry_shape(TINY) == (2, DAC.F, G.CANVAS * G.CANVAS, TINY.dec_width)
    assert np.all(np.isfinite(np.asarray(out.logits)))
    outs, res, _ = M.iterate_eq(p, TINY, x, tau=1.0, t_total=2, task_vec=code)
    assert len(outs) == 2 and all(np.isfinite(float(r)) for r in res)

    def loss(params, c):
        return OBJ.pair_loss(params, TINY, x, y, tau=1.0, rng=jax.random.PRNGKey(3), task_vec=c)[0]
    l, (gp, gc) = jax.value_and_grad(loss, argnums=(0, 1))(p, code)
    assert np.isfinite(float(l))
    assert all(np.all(np.isfinite(np.asarray(g))) for g in jax.tree.leaves(gp))
    assert np.all(np.isfinite(np.asarray(gc))) and float(jnp.sum(jnp.abs(gc))) > 0.0, "the code must receive gradient"
    assert float(jnp.sum(jnp.abs(gp["decarc"]["void_head"]["w"]))) > 0.0, "the void readout must train"


def test_decarc_commit_head_inert_at_tau_one():
    """With dec_commit and tau >= 1 the head trains (its logits exist) and the forward is the plain cell's."""
    x, _ = _grid(seed=6); code = _code(3)
    p0 = M.init_params(jax.random.PRNGKey(0), TINY)
    p1 = M.init_params(jax.random.PRNGKey(0), TINY_COMMIT)
    assert "commit_head" in p1["decarc"] and "commit_head" not in p0["decarc"]
    o0 = M.forward_fields(p0, TINY, M.build_fields(x, jnp.full_like(x, G.VOID)), t_norm=1.0, tau=1.0, task_vec=code)
    o1 = M.forward_fields(p1, TINY_COMMIT, M.build_fields(x, jnp.full_like(x, G.VOID)), t_norm=1.0, tau=1.0, task_vec=code)
    assert np.array_equal(np.asarray(o0.logits), np.asarray(o1.logits))
    cl = DAC.commit_logits(p1["decarc"], o1.z_fine[0])
    assert cl.shape == (G.CANVAS * G.CANVAS,) and np.all(np.isfinite(np.asarray(cl)))
    with pytest.raises(AssertionError):
        bad = Config(**{**TINY_COMMIT.__dict__, "dec_commit_tau": 0.9})
        M.forward_fields(p1, bad, M.build_fields(x, jnp.full_like(x, G.VOID)), t_norm=1.0, tau=1.0, task_vec=code)


def test_decarc_param_count_formula():
    def swiglu(n, expansion=4.0):
        inter = TC._find_multiple(round(expansion * n * 2 / 3), 256)
        return n * 2 * inter + inter * n
    for cfg in (TINY, Config(**ARC, cell_kind="decarc", dec_width=160, decarc_heads=4, trm_layers=2)):
        w, hw = cfg.dec_width, G.CANVAS * G.CANVAS
        per_block = 4 * w * w + swiglu(w) + w * w
        expected = cfg.trm_layers * per_block + 3 * w + hw * w + cfg.d_task * w + w + (w + 1) + (2 * w + 2)
        p = M.init_params(jax.random.PRNGKey(0), cfg)
        assert M.count_params(p["decarc"]) == expected, (cfg.dec_width, M.count_params(p["decarc"]), expected)
    w160 = M.count_params(M.init_params(jax.random.PRNGKey(0), Config(**ARC, cell_kind="decarc", dec_width=160, decarc_heads=4, trm_layers=2))["decarc"])
    assert 0.8e6 <= w160 <= 1.2e6, w160     # the plan's ≈ 0.9 M at w160 (the position table adds 0.16 M)


def test_decarc_fpa_anchor_state_and_ri_draw():
    x, y = _grid(seed=7)
    p = M.init_params(jax.random.PRNGKey(0), TINY)
    zH = DAC.embed_answer(p["decarc"], TINY, y)
    assert zH.shape == (DAC.F, G.CANVAS * G.CANVAS, TINY.dec_width) and np.all(np.isfinite(np.asarray(zH)))
    # the anchor at eps = 0 is a permutation-covariant embedding of the answer (field c of y's colour c is 'mine')
    roles = np.asarray(DAC.roles_of(y)); yy = np.asarray(y).reshape(-1)
    for c in range(10):
        assert np.all((roles[c] == DAC.ROLE_MINE) == (yy == c))
    assert np.all((roles[0] == DAC.ROLE_OUT) == (yy == G.VOID))
    ri = Config(**TINY.__dict__ | {"trm_ri_sigma": 1.0})
    z = DAC.z0(ri, G.CANVAS * G.CANVAS, rng=jax.random.PRNGKey(9))
    assert z.shape == (2, DAC.F, G.CANVAS * G.CANVAS, TINY.dec_width) and abs(float(jnp.std(z)) - 1.0) < 0.05
    z_fixed = DAC.z0(TINY, G.CANVAS * G.CANVAS)
    assert np.allclose(np.asarray(z_fixed[0, 0]), np.asarray(z_fixed[0, 7]))     # the fixed start is S10-invariant


def test_decarc_deployed_fit_and_probe_trace():
    """The deployed arm-A fit (eval_dev30._fit) runs on a DEC-ARC checkpoint with a (F, d_task) code table,
    and probe_e1e3.trace reproduces train.predict on the fitted code (the probe's warrant)."""
    import eval_dev30 as ED
    import probe_e1e3 as P
    eps = G.load_task("007bbfb7")          # an ARC training task with 5 support pairs
    p = M.init_params(jax.random.PRNGKey(0), TINY)
    table = np.asarray(jax.random.normal(jax.random.PRNGKey(2), (3, DAC.F, TINY.d_task)) * 0.1)
    Fres = ED._fit("A", TINY, {"model": p, "table": table}, eps, steps=12, val_every=6, wd=1e-4, tau=1.0, seed=0)
    tv = np.asarray(Fres["tv_of"](Fres["final"]))
    assert tv.shape == (DAC.F, TINY.d_task) and np.all(np.isfinite(tv)) and all(np.isfinite(l) for l in Fres["losses"])
    model = Fres["final"]["model"]
    st = P.trace(model, TINY, eps[0].query_x, tau=1.0, task_vec=jnp.asarray(tv), t_total=TINY.T)
    pred, (h, w), canvas = T.predict(model, TINY, eps[0].query_x, tau=1.0, task_vec=jnp.asarray(tv))
    assert st[TINY.T - 1]["hw"] == (h, w)
    # the two paths are separately jitted graphs (a whole-loop jit vs a per-step jit); on an untrained tiny model
    # near-tied logits may flip between them, so the preds are compared where the top-2 margin is resolvable
    x_can = jnp.asarray(G.place(np.asarray(eps[0].query_x)), jnp.int32)
    logits, _, _ = T._predict_core(TINY, 1.0)(model, x_can, jnp.asarray(tv))
    top2 = np.sort(np.asarray(logits), axis=-1)[..., -2:]
    resolvable = (top2[..., 1] - top2[..., 0]) > 1e-3
    tr_canvas = np.where(st[TINY.T - 1]["pred"] == 0, 0, st[TINY.T - 1]["pred"])
    assert np.array_equal(np.where(resolvable[:h, :w], tr_canvas, -1), np.where(resolvable[:h, :w], pred, -1))
    assert resolvable.mean() > 0.5, "the fitted model should resolve most cells"
