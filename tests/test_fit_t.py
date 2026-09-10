"""The FIT-ONLY outer-pass knob (2026-09-10, the DEC-ARC pilot's cost finding): eval_dev30._fit(fit_T=...) runs the
arm-A fit STEP through iterate at cfg.T := fit_T while the validation predict keeps the deployed cfg.T.
(i) fit_T=None is the deployed path (the same lru-cached step; losses bit-identical to fit_T=cfg.T);
(ii) fit_T=1 runs, differs from the deployed losses on a T>1 cfg, and the returned trainables keep the full cfg's
validation curve semantics (val entries present); (iii) probe_e1e3.fit_arm_a passes the knob through."""
import sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import probe_e1e3 as P
import eval_dev30 as ED
from qhrrn2 import grid as G
from qhrrn2.config import Config
from qhrrn2.model import init_params


@pytest.fixture(scope="module")
def setup():
    cfg = Config(d=8, T=3, K=8, attn_max_hw=0)
    params = init_params(jax.random.PRNGKey(0), cfg)
    rng = np.random.default_rng(3)
    sup = []
    for _ in range(3):
        g = np.asarray(rng.integers(0, 10, (4, 4)), dtype=np.int8)
        sup.append((g, g.copy()))
    eps = [G.Episode(task_id="t", support=tuple(sup), query_x=sup[0][0], query_y=sup[0][1])]
    state = {"model": params, "table": np.zeros((2, cfg.d_task), np.float32)}
    return cfg, state, eps


def _fit(cfg, state, eps, **kw):
    return ED._fit("A", cfg, state, eps, steps=6, val_every=3, wd=1e-4, tau=1.0, seed=0, **kw)


def test_fit_t_none_is_deployed(setup):
    cfg, state, eps = setup
    F0 = _fit(cfg, state, eps)
    F1 = _fit(cfg, state, eps, fit_T=cfg.T)          # an EQUAL cfg -> the same lru-cached step -> bit-identical
    assert F0["losses"] == F1["losses"]
    assert F0["val_curve"] == F1["val_curve"]


def test_fit_t_one_runs_and_differs(setup):
    cfg, state, eps = setup
    F0 = _fit(cfg, state, eps)
    F1 = _fit(cfg, state, eps, fit_T=1)
    assert len(F1["losses"]) == 6 and all(np.isfinite(F1["losses"]))
    assert F1["losses"] != F0["losses"]                # one map application per step is a different objective at T=3
    assert len(F1["val_curve"]) == 2                   # the validation cadence (the deployed cfg.T predict) is untouched
    assert F1["tv_of"](F1["final"]).shape == (cfg.d_task,)


def test_fit_arm_a_passes_fit_t(setup):
    cfg, state, eps = setup
    m0, _, sel0, F0 = P.fit_arm_a(state, cfg, eps, steps=6, val_every=3, seed=0)
    m1, _, sel1, F1 = P.fit_arm_a(state, cfg, eps, steps=6, val_every=3, seed=0, fit_T=1)
    assert F0["losses"] == _fit(cfg, state, eps)["losses"]
    assert F1["losses"] == _fit(cfg, state, eps, fit_T=1)["losses"]
    assert sel0[1].shape == sel1[1].shape == (cfg.d_task,)
