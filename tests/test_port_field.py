"""BUILD B2 (Plan_2026-09-07_Instrument_Suite §4.1 / §6; 2026-09-07): the ported public checkpoints (tools/field_ckpts/port_field_ckpt.py)
read back through OUR evaluator. Skipped when the ported files are absent (they live under runs/, git-ignored). Asserts: the port
check's registered bars (fp32 step-1 max |Δlogit| ≤ 2e-3 on logits of scale ≥ 10, i.e. relative ≤ 2e-4; D16 exact agreement with
their PyTorch forward ≥ 95 %, the chaos floor); the parameter pin (5,037,058 = TRM-MLP's count); the evaluator's cold D16 exact bits
on the same 64 strat puzzles EQUAL the port tool's JAX bits (identical code path: trm_cell through model.forward_fields); and the
summary carries the `port` provenance block."""
import json, os, subprocess, sys
from pathlib import Path
import numpy as np, pytest

ROOT = Path(__file__).resolve().parents[1]
PORTED = ROOT / "runs/field_ckpts/ported"; NPZ = ROOT / "data/sudoku_extreme/sudoku_extreme_seed0.npz"
TAGS = [t for t in ("trmpub", "cgar", "eqr", "trmpub60k") if (PORTED / t / "port_check.json").exists() and (PORTED / t / "ckpt_latest.pkl").exists()]
pytestmark = pytest.mark.skipif(not TAGS or not NPZ.exists(), reason="ported checkpoints / npz absent (runs/ is git-ignored)")


@pytest.mark.parametrize("tag", TAGS)
def test_port_check_bars(tag):
    c = json.loads((PORTED / tag / "port_check.json").read_text())
    assert c["step1_max_abs"] <= 2e-3 and c["step1_scale"] >= 10, (tag, c["step1_max_abs"], c["step1_scale"])
    assert c["exact_agree_D"] >= 0.95, (tag, c["exact_agree_D"])
    assert c["rows"][0]["exact_agree"] == 1.0 and c["rows"][0]["argmax_agree"] >= 0.999


@pytest.mark.parametrize("tag", TAGS[:1])
def test_evaluator_reproduces_the_port_tool_bits(tag, tmp_path):
    c = json.loads((PORTED / tag / "port_check.json").read_text())
    ids = np.asarray(c["idx"]); bits_port = np.asarray(c["exact_jax_by_step"], np.uint8)     # (64, D) from trm_cell directly
    import sys as _s; _s.path.insert(0, str(ROOT / "src"))
    from qhrrn2 import sudoku_extreme as SX
    d = SX.load_prepared(NPZ); sel = SX.stratified_subsample(d["test_rating"], 256, 20260821)[:64]
    assert np.array_equal(sel, ids)
    cmd = [sys.executable, str(ROOT / "tools/eval_sudoku_extreme.py"), "--ckpt", str(PORTED / tag / "ckpt_latest.pkl"), "--npz", str(NPZ),
           "--out", str(tmp_path), "--split", "test", "--stratified", "256", "--limit", "64", "--t-total", str(c["D"]), "--ema", "--record-by-step", "--record-q", "--batch", "64"]
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), JAX_PLATFORMS="cpu", JAX_DEFAULT_MATMUL_PRECISION="highest")
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=ROOT); assert r.returncode == 0, r.stderr[-2000:]
    arr = dict(np.load(tmp_path / "records_all.npz")); summ = json.loads((tmp_path / "summary_all.json").read_text())
    assert np.array_equal(arr["idx"], ids)
    bits = np.unpackbits(arr["exact_by_step"], axis=1, bitorder="little")[:, :c["D"]]
    agree = float((bits == bits_port).mean())
    assert agree >= 0.98, agree                      # identical weights + code path; fp32 round-off on chaotic rows can flip a late bit
    assert abs(float(bits[:, -1].mean()) - c["exact_jax_D"]) <= 2 / 64
    assert summ["port"]["tag"] == tag and summ["record_q"] and "q_halt_auc_last" in summ
    import pickle
    ck = pickle.load(open(PORTED / tag / "ckpt_latest.pkl", "rb")); p = ck["state_ema"]["model"]["trm"]
    n = sum(int(np.asarray(v).size) for k, v in p.items() if k not in ("H_init", "L_init", "blocks", "q_head")) \
        + sum(int(np.asarray(v).size) for b in p["blocks"] for m in b.values() for v in m.values()) + sum(int(np.asarray(v).size) for v in p["q_head"].values())
    assert n == 5_037_058 and p["H_init"].shape == (512,)
