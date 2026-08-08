# Ledger: named tests for the E1/E3 instruments (2026-08-08 program).
# The probe must MEASURE the deployed system, not a variant of it:
# (i) trace() at t_total=cfg.T reproduces train.predict bit-exactly;
# (ii) the H[q] plumbing equals the objective's rule-entropy on the same
# forward; (iii) candidate-stability mode is shape-sane and runs only the
# final map.
import numpy as np
import jax
import jax.numpy as jnp
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import probe_e1e3 as P
from qhrrn2 import grid as G
from qhrrn2 import train as T
from qhrrn2.config import Config
from qhrrn2.model import init_params
from qhrrn2.objective import pair_loss


@pytest.fixture(scope="module")
def setup():
    cfg = Config(d=8, T=3, K=8, attn_max_hw=0)
    params = init_params(jax.random.PRNGKey(0), cfg)
    rng = np.random.default_rng(1)
    x = np.asarray(rng.integers(0, 10, (5, 6)), dtype=np.int8)
    tv = jnp.asarray(np.asarray(rng.normal(size=(cfg.d_task,)), dtype=np.float32))
    return cfg, params, x, tv


def test_trace_matches_predict(setup):
    cfg, params, x, tv = setup
    steps = P.trace(params, cfg, x, tau=1.0, task_vec=tv, t_total=cfg.T)
    pred_ref, (h, w), _ = T.predict(params, cfg, x, tau=1.0, task_vec=tv)
    assert steps[-1]["hw"] == (h, w)
    assert np.array_equal(steps[-1]["pred"], pred_ref)


def test_hq_matches_objective(setup):
    cfg, params, x, tv = setup
    steps = P.trace(params, cfg, x, tau=1.0, task_vec=tv, t_total=cfg.T)
    y = G.place(x)  # any target; aux entropy depends on the forward, not y
    _, aux = pair_loss(params, cfg, jnp.asarray(G.place(x), dtype=jnp.int32),
                       jnp.asarray(y, dtype=jnp.int32), tau=1.0, task_vec=tv)
    # pair_loss aux reports the LAST iterate's mean slot entropy
    assert np.isclose(steps[-1]["H_q"], float(np.mean(np.asarray(
        aux["rule_entropy_last"]))), rtol=1e-4)


def test_candidate_stability_mode(setup):
    cfg, params, x, tv = setup
    init = np.asarray(np.zeros((5, 6)), dtype=np.int8)
    st = P.trace(params, cfg, x, tau=1.0, task_vec=tv, t_total=4,
                 yprev_init=G.place(init), skip_trained=True)
    assert len(st) == 4
    for s in st:
        assert s["pred"].ndim == 2 and "H_q" in s


def test_snapshot_flag_inert(setup):
    """The E1 snapshot flag must not perturb the deployed fit path."""
    import eval_dev30 as ED
    cfg, params, x, tv = setup
    rng = np.random.default_rng(3)
    sup = []
    for _ in range(3):
        g = np.asarray(rng.integers(0, 10, (4, 4)), dtype=np.int8)
        sup.append((g, g.copy()))
    eps = [G.Episode(task_id="t", support=tuple(sup), query_x=sup[0][0],
                     query_y=sup[0][1])]
    state = {"model": params, "table": np.zeros((2, cfg.d_task), np.float32)}
    snaps = []
    F1 = ED._fit("A", cfg, state, eps, steps=8, val_every=4, wd=1e-4,
                 tau=1.0, seed=0, snapshots=snaps)
    F2 = ED._fit("A", cfg, state, eps, steps=8, val_every=4, wd=1e-4,
                 tau=1.0, seed=0)
    assert F1["val_curve"] == F2["val_curve"]
    assert len(snaps) == 2 and snaps[0][0] == 4
    assert np.allclose(np.asarray(F1["tv_of"](F1["final"])),
                       np.asarray(F2["tv_of"](F2["final"])))


def test_probe_selection_matches_deployed(setup):
    """fit_arm_a's selected tv equals the deployed earliest-exact/best rule."""
    import eval_dev30 as ED
    cfg, params, x, tv = setup
    rng = np.random.default_rng(4)
    sup = [(np.asarray(rng.integers(0, 10, (4, 4)), dtype=np.int8),) * 2
           for _ in range(3)]
    eps = [G.Episode(task_id="t", support=tuple(sup), query_x=sup[0][0],
                     query_y=sup[0][1])]
    state = {"model": params, "table": np.zeros((2, cfg.d_task), np.float32)}
    _, snaps, sel, F = P.fit_arm_a(state, cfg, eps, steps=8, val_every=4)
    ref = (F["first_exact"] if F["first_exact"] is not None
           else (F["best"]["trainable"], F["best"]["step"]))
    ref_tv = np.asarray(F["tv_of"](ref[0]))
    assert np.array_equal(sel[1], ref_tv)


def test_extension_repeats_final_map(setup):
    """Steps beyond cfg.T must equal re-application of the t_norm=1 map."""
    cfg, params, x, tv = setup
    ext = P.trace(params, cfg, x, tau=1.0, task_vec=tv, t_total=cfg.T + 2)
    # manually apply the final map twice starting from the T-step canvas
    base = P.trace(params, cfg, x, tau=1.0, task_vec=tv, t_total=cfg.T)
    can = G.place(base[-1]["pred"]) if False else None
    # reconstruct: run skip_trained from the T-1 full canvas
    # (use the raw canvas, not the cropped pred)
    x_can = jnp.asarray(G.place(np.asarray(x)), dtype=jnp.int32)
    yprev = jnp.full((G.CANVAS, G.CANVAS), G.VOID, dtype=jnp.int32)
    for t in range(cfg.T):
        t_norm = t / max(cfg.T - 1, 1)
        out = P._traced_fwd(cfg, 1.0, float(t_norm))(
            params, __import__("qhrrn2.model", fromlist=["build_fields"])
            .build_fields(x_can, yprev), tv)
        yprev = jnp.argmax(out.logits, axis=-1)
    manual = P.trace(params, cfg, x, tau=1.0, task_vec=tv, t_total=2,
                     yprev_init=np.asarray(yprev), skip_trained=True)
    assert np.array_equal(manual[0]["pred"], ext[cfg.T]["pred"])
    assert np.array_equal(manual[1]["pred"], ext[cfg.T + 1]["pred"])
