"""2026-09-15 — the DEC-ARC evaluator's additions before the registration (tools/eval_decarc.py): (i) the FUSED trace's
preds / sizes / canvases are bit-identical to the pilot-proven eager path's and to probe_e1e3.trace's, from the cell's own
deterministic start and from an explicit start, and its descriptive fields agree to float tolerance (res_y read BEFORE the
readout update on both paths); (ii) run_task's start rows (the other deterministic starts of the same fitted code, no
re-fit) and (iii) the mon96 path (a trained code row, no fit) on a real ARC task with a tiny RI-configured model."""
import json
import sys
import types
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import eval_decarc as EA
import probe_e1e3 as P
from qhrrn2 import grid as G
from qhrrn2 import model as M
from qhrrn2 import decarc_cell as DAC
from qhrrn2.config import Config

ARC = dict(equilibrium=True, T=3, eta_fixed=1.0, eta_z_fixed=1.0, loss_kind="stablemax", d_task=8)
CFG = Config(**ARC, cell_kind="decarc", dec_width=16, decarc_heads=2, trm_layers=1, trm_h_cycles=2, trm_l_cycles=2,
             trm_ri_sigma=1.0, dec_commit=True, dec_commit_tau=1.0)
TASK = "007bbfb7"


@pytest.fixture(scope="module")
def setup():
    p = M.init_params(jax.random.PRNGKey(0), CFG)
    table = np.asarray(jax.random.normal(jax.random.PRNGKey(2), (3, DAC.F, CFG.d_task)) * 0.1)
    return {"model": p, "table": table}, G.load_task(TASK)


def _same(A, B):
    return all(a["hw"] == b["hw"] and np.array_equal(a["pred"], b["pred"]) and np.array_equal(a["canvas"], b["canvas"]) for a, b in zip(A, B))


def test_trace_fused_equals_eager_and_probe(setup):
    state, eps = setup
    code = jnp.asarray(state["table"][0]); x = eps[0].query_x
    sf = EA.trace_dec(state["model"], CFG, x, code=code, t_total=5, fused=True)
    se = EA.trace_dec(state["model"], CFG, x, code=code, t_total=5, fused=False)
    sp = P.trace(state["model"], CFG, x, tau=1.0, task_vec=code, t_total=5)
    assert _same(sf, se), "fused vs eager preds differ"
    assert all(a["hw"] == b["hw"] and np.array_equal(a["pred"], b["pred"]) for a, b in zip(sf, sp)), "fused vs the probe trace"
    for a, b in zip(sf, se):
        assert np.allclose(a["conf"], b["conf"], atol=1e-6) and np.allclose(a["c"], b["c"], atol=1e-6) and np.allclose(a["q"], b["q"], atol=1e-5)
        assert abs(a["res_y"] - b["res_y"]) < 1e-6
        assert (a["res_z"] is None and b["res_z"] is None) or abs(a["res_z"] - b["res_z"]) < 1e-6
    assert sf[0]["res_z"] is None and sf[1]["res_z"] is not None and sf[0]["res_y"] > 0.0   # res_y before the update: not ~0
    z = DAC.z0_eval(CFG, G.CANVAS * G.CANVAS, "rifix")
    assert _same(EA.trace_dec(state["model"], CFG, x, code=code, t_total=4, z_init=z, fused=True),
                 EA.trace_dec(state["model"], CFG, x, code=code, t_total=4, z_init=z, fused=False))


def _args(**kw):
    a = types.SimpleNamespace(steps=4, val_every=2, seed=0, fit_t=1, t_total=3, k=2, sigma=1.0, ladder=[0.0, 0.2], ladder_draws=1,
                              ladder_steps=2, views=1, flip_test=False, start_rows=[], set="valhard")
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def test_run_task_start_rows(setup):
    state, eps = setup
    EA.MON96 = None
    rec = EA.run_task(state, CFG, TASK, _args(start_rows=["buffers", "fieldfix", "rifix"]))
    q = rec["queries"][0]
    assert rec["xcheck"] is True and rec["sel_step"] is not None and rec["fit_s"] > 0
    assert set(q["starts"]) == {"buffers", "rifix"}, q["starts"].keys()   # the cfg's own start (fieldfix) IS the cold row: not repeated
    for k in ("buffers", "rifix"):
        assert len(q["starts"][k]["exact_by_step"]) == 3 and isinstance(q["starts"][k]["agree_T"], bool)
    assert "draws" in q and "sel" in q and "ladder" in q and len(q["exact_by_step"]) == 3


def test_run_task_mon96_no_fit(setup, tmp_path):
    state, eps = setup
    EA.MON96 = {TASK: 1}
    try:
        rec = EA.run_task(state, CFG, TASK, _args(k=0, ladder=[0.0]))
    finally:
        EA.MON96 = None
    assert rec["sel_step"] is None and rec["xcheck"] is True and len(rec["queries"]) == 1
    code = jnp.asarray(state["table"][1])   # the code is the trained row, verbatim
    st = EA.trace_dec(state["model"], CFG, eps[0].query_x, code=code, t_total=3)
    assert rec["queries"][0]["exact_by_step"] == [EA.ex(s["pred"], eps[0].query_y) for s in st]
    assert "ladder" in rec["queries"][0] and "draws" not in rec["queries"][0]
    d = tmp_path / "pretraindecarc_D0"; d.mkdir()
    (d / "val_tasks.json").write_text(json.dumps([{"t": 1, "tid": TASK, "n_q": 1}]))
    ck = str(d / "ckpt_000100.pkl")
    assert EA.task_ids_of("mon96", ck) == [TASK] and EA.mon96_rows(ck) == {TASK: 1}
    with pytest.raises(AssertionError):
        EA.task_ids_of("mon96")
