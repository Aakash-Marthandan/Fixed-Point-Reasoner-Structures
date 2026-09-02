# Ledger: sportC1 §4.5 — the explosion census tool: classification semantics
# (non-finite OR |z| > 1e6 = exploded; first-bad step; finite peak) and an
# end-to-end trajectory read on a tiny native model (shapes, finiteness).
import sys
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import explosion_census as CE
from qhrrn2 import grid as G, model as M, sudoku as SU
from qhrrn2.config import Config


def test_classify_semantics():
    T, B = 5, 4
    z = np.ones((T, B))
    z[3, 1] = np.inf                  # non-finite at step 3
    z[2, 2] = 5e6; z[4, 2] = 1e9      # over the limit from step 2
    z[:, 3] = np.linspace(1, 100, T)  # bounded
    ex, fb, peak = CE.classify(z)
    assert ex.tolist() == [False, True, True, False]
    assert fb.tolist() == [-1, 3, 2, -1]
    assert peak[3] == 100.0 and peak[1] == 1.0 and peak[2] == 1e9
    row, *_ = CE.summarize_row("x", 0.8, 0.7, z, np.ones((T, B)), T, False)
    assert row["n_exploded"] == 2 and row["exploded_frac"] == 0.5 and row["first_bad_median"] == 2.5


def test_trajectories_tiny_native():
    cfg = Config(d=8, K=8, T=2, canvas=9, scales=2, pool_arity=3, mixer_kind="group9",
                 attn_max_hw=9, equilibrium=True, sudoku_layout="native9")
    p = M.init_params(jax.random.PRNGKey(0), cfg)
    p = {**p, "eq": {**p["eq"], "alpha_z": jnp.asarray(0.3)}}
    puzzles = [SU.sample(np.random.default_rng(i), 40)[0] for i in range(3)]
    x_can = jnp.asarray(np.stack(puzzles), jnp.int32)
    void = jax.nn.one_hot(jnp.full((9, 9), G.VOID, jnp.int32), M.VOCAB).transpose(2, 0, 1)
    y0 = jnp.broadcast_to(void, (3,) + void.shape)
    eta, eta_z = (float(v) for v in M.eq_etas(p, cfg))
    zmax, lmax = CE.trajectories(p, cfg, jnp.zeros((cfg.d_task,)), x_can, y0, t_total=4, eta=eta, eta_z=eta_z)
    assert zmax.shape == (4, 3) and lmax.shape == (4, 3) and np.isfinite(zmax).all()
    ex, fb, peak = CE.classify(zmax)
    assert not ex.any() and (fb == -1).all()
