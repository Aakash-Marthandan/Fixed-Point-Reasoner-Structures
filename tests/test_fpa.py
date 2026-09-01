# Ledger: SPRINT S2 wave 3a (2026-08-23, H-45) — named tests for the fixed-point
# anchor (FPA) rows, the final-map mode (t_norm_fixed), the evaluator's eq_coupled
# dynamics, and the trajectory monitor: fpa_k=0 leaves the registered loss/rng path
# bit-exact; fpa_k>0 adds a finite, differentiable term; the final-map mode applies
# one map; the monitor returns finite readouts including lambda_max.
import numpy as np
import jax, jax.numpy as jnp
import pytest

from qhrrn2 import sudoku as SU
from qhrrn2 import grid as G
from qhrrn2 import model as M
from qhrrn2 import objective as O
from qhrrn2 import episodic as E
from qhrrn2.config import Config
from qhrrn2.model import init_params


def _setup(**kw):
    cfg = Config(d=8, K=8, T=3, equilibrium=True, **kw)
    params = init_params(jax.random.PRNGKey(0), cfg)
    puz, sol = SU.sample_pairs(1, seed=4, givens=40)[0]
    x = jnp.asarray(G.place(puz), jnp.int32); y = jnp.asarray(G.place(sol), jnp.int32)
    tv = jnp.zeros((cfg.d_task,), jnp.float32)
    return cfg, params, x, y, tv


def test_fpa_off_is_bit_exact_and_on_adds_a_finite_term():
    cfg0, params, x, y, tv = _setup()
    l0, aux0 = O.pair_loss(params, cfg0, x, y, tau=1.0, rng=jax.random.PRNGKey(3), task_vec=tv)
    l0b, _ = O.pair_loss(params, cfg0, x, y, tau=1.0, rng=jax.random.PRNGKey(3), task_vec=tv)
    assert float(l0) == float(l0b) and "fpa_ce_last" not in aux0
    cfg1 = Config(d=8, K=8, T=3, equilibrium=True, fpa_k=2, fpa_eps=0.2)
    l1, aux1 = O.pair_loss(params, cfg1, x, y, tau=1.0, rng=jax.random.PRNGKey(3), task_vec=tv)
    assert np.isfinite(float(l1)) and "fpa_ce_last" in aux1 and np.isfinite(float(aux1["fpa_ce_last"]))
    assert float(l1) != float(l0)            # the term is live
    # differentiable: nonzero gradient through the FPA rollout
    g = jax.grad(lambda p_: O.pair_loss(p_, cfg1, x, y, tau=1.0, rng=jax.random.PRNGKey(3), task_vec=tv)[0])(params)
    assert float(sum(jnp.sum(jnp.abs(v)) for v in jax.tree.leaves(g))) > 0
    # rng=None (inference path) never builds FPA rows
    l2, aux2 = O.pair_loss(params, cfg1, x, y, tau=1.0, rng=None, task_vec=tv)
    assert "fpa_ce_last" not in aux2


def test_t_norm_fixed_applies_one_map():
    cfg, params, x, y, tv = _setup()
    y0 = jax.nn.one_hot(y, M.VOCAB).transpose(2, 0, 1)
    outs_fix, _, _ = M.iterate_eq(params, cfg, x, tau=1.0, rng=None, task_vec=tv, t_total=2, y0_probs=y0, t_norm_fixed=1.0)
    # reference: one manual final-map step from y0
    out_ref = M.forward_fields(params, cfg, M.build_fields_soft(x, y0), t_norm=1.0, tau=1.0, rng=None, task_vec=tv, z_in=None)
    assert np.allclose(np.asarray(outs_fix[0].logits), np.asarray(out_ref.logits), atol=1e-5)
    # the trained ramp differs at step 0 (t_norm=0) unless T==1
    outs_ramp, _, _ = M.iterate_eq(params, cfg, x, tau=1.0, rng=None, task_vec=tv, t_total=2, y0_probs=y0)
    assert not np.allclose(np.asarray(outs_ramp[0].logits), np.asarray(out_ref.logits), atol=1e-5)


def test_evaluator_coupled_and_final_map_modes():
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "tools"))
    import eval_sudoku_extreme as EV
    cfg, params, x, y, tv = _setup(eq_coupled=True)
    ab = EV.coupled_ab(params, cfg); assert ab is not None and 0 < ab[0] < 1 and 0 < ab[1] < 1
    cfg_d, params_d, _, _, _ = _setup(); assert EV.coupled_ab(params_d, cfg_d) is None
    puz, sol = SU.sample_pairs(1, seed=4, givens=40)[0]
    x_can = EV.place_batch(np.stack([puz]), "origin"); sol9 = np.stack([sol]).astype(np.int32); puz9 = np.stack([puz]).astype(np.int32)
    y0 = jax.nn.one_hot(EV.place_batch(sol9, "origin"), M.VOCAB).transpose(0, 3, 1, 2)
    ex, ok, pred, _ = EV.run_batch(params, cfg, tv, x_can, y0, t_total=3, tau=1.0, gamma=1.0, sol9=sol9, puz9=puz9,
                                eta=0.5, eta_z=0.5, ab=ab, t_norm_fixed=1.0)
    assert ex.shape == (3, 1) and pred.shape == (1, 9, 9)


def test_monitor_returns_finite_readouts():
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "tools"))
    import pretrain as PT
    cfg, params, x, y, tv = _setup()
    state = {"model": params, "table": jnp.zeros((1, cfg.d_task), jnp.float32)}
    pairs = [(p_, s_) for p_, s_ in SU.sample_pairs(3, seed=5, givens=40)]
    rec = PT.sudoku_monitor(state, cfg, pairs, t_cold=4, t_ret=2, n_lam=2, lam_iters=3)
    for k in ("val_t64", "ret_sched_t8", "ret_final_t8", "eta", "lam_max_mean", "lam_max_max", "lam_frac_expansive"):
        assert k in rec and np.isfinite(rec[k]), k
    assert rec["lam_max_mean"] > 0
