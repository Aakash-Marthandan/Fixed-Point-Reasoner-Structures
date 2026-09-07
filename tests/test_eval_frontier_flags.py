"""FRONTIER SUITE FLAGS (Plan_2026-09-07_Instrument_Suite §6 build B1; 2026-09-07) — named tests on a tiny trm-cell checkpoint
run through the REAL evaluator (subprocess, CPU): --record-by-step (packed bits agree with first_exact / cold), --record-q
(the halting logits per step + the summary's AUC rows), --z0-mode trunc (EqR's reset: std 1, clipped at +-2 comp), perturb at
eps 0 == the cold pass, --z0-device (on-device draws run and differ from the host bitstream), --seg-noise-beta (deterministic per
seed, changes the trajectory), --zero-prefix (changes the readout), and a ported checkpoint's own H_init / L_init buffers
travelling in params (the cold pass changes when the buffers change; the default seeded buffers are unchanged)."""
import json, os, subprocess, sys
from pathlib import Path
import numpy as np, jax, jax.numpy as jnp, pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tools"))
from qhrrn2 import episodic as E, model as M, trm_cell as TC
from qhrrn2.config import Config
import eval_sudoku_extreme as EV

NPZ = ROOT / "data/sudoku_extreme/sudoku_extreme_seed0.npz"
pytestmark = pytest.mark.skipif(not NPZ.exists(), reason="Sudoku-Extreme npz absent")
FIELD = dict(canvas=9, scales=2, pool_arity=3, mixer_kind="group9", attn_max_hw=9, equilibrium=True, sudoku_layout="native9",
             eta_fixed=1.0, eta_z_fixed=1.0, loss_kind="stablemax")
TRM_TINY = Config(**FIELD, T=3, cell_kind="trm", trm_hidden=32, trm_layers=1, trm_h_cycles=2, trm_l_cycles=2, trm_puzzle_emb_len=2,
                  trm_lambda=0.05, trm_beta=0.0)


def _write_ckpt(path, cfg, seed=0, buffers=None, prefix=False):
    params = M.init_params(jax.random.PRNGKey(seed), cfg)
    # TRM zero-inits the Q head (q == (-5, -5) for every state): give it a live weight so q reads the state
    params = {**params, "trm": {**params["trm"], "q_head": {"w": 0.1 * jax.random.normal(jax.random.PRNGKey(9), params["trm"]["q_head"]["w"].shape),
                                                            "b": params["trm"]["q_head"]["b"]}}}
    if prefix:   # TRM zero-inits the puzzle prefix: give it a live value so --zero-prefix has something to remove
        params = {**params, "trm": {**params["trm"], "puzzle_emb": jax.random.normal(jax.random.PRNGKey(5), params["trm"]["puzzle_emb"].shape)}}
    if buffers is not None:
        params = {**params, "trm": {**params["trm"], "H_init": buffers[0], "L_init": buffers[1]}}
    st = {"model": params, "table": jnp.zeros((1, 32))}
    E.save_ckpt(str(path), dict(config=cfg.__dict__, state=st, state_ema=st, step=0, rng=None, opt_state=None,
                                port=dict(tag="tiny", note="test")))


def _run(ckpt, out, *flags, n=6, t=4, k=0):
    cmd = [sys.executable, str(ROOT / "tools/eval_sudoku_extreme.py"), "--ckpt", str(ckpt), "--npz", str(NPZ), "--out", str(out),
           "--split", "test", "--limit", str(n), "--t-total", str(t), "--k-init", str(k), "--batch", "4", *flags]
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), JAX_PLATFORMS="cpu")
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=ROOT)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-3000:]
    arr = dict(np.load(out / "records_all.npz")); summ = json.loads((out / "summary_all.json").read_text())
    return arr, summ


def test_record_by_step_and_q(tmp_path):
    ck = tmp_path / "ckpt_latest.pkl"; _write_ckpt(ck, TRM_TINY)
    arr, summ = _run(ck, tmp_path / "o", "--record-by-step", "--record-q", t=4)
    bits = np.unpackbits(arr["exact_by_step"], axis=1, bitorder="little")[:, :4].astype(bool)
    assert bits.shape == (6, 4) and (bits[:, -1] == arr["cold_exact"]).all()
    fe = np.where(bits.any(1), bits.argmax(1), -1)
    assert (fe == arr["first_exact"]).all()
    assert len(summ["exact_by_step_curve"]) == 4 and "depth_regressions" in summ
    assert arr["q_by_step"].shape == (6, 4, 2) and np.isfinite(arr["q_by_step"].astype(np.float32)).all()
    assert "q_halt_frac_last" in summ and "q_first_halt_step_mean" in summ and "exact_at_first_halt" in summ
    assert summ["record_by_step"] and summ["record_q"] and summ["port"]["tag"] == "tiny"


def test_z0_modes_trunc_perturb_device(tmp_path):
    shape = (2, 83, 32)
    z = EV.mi_z0(7, 3, 0, shape, 1.0, "trunc")
    comp = EV.trunc_normal_comp(1.0)
    assert abs(float(z.std()) - 1.0) < 0.05 and float(np.abs(z).max()) <= 2 * comp + 1e-6 and comp > 1.0
    zg = EV.mi_z0(7, 3, 0, shape, 1.0, "gauss"); zg_ref = EV.mi_z0(7, 3, 0, shape, 1.0)
    assert np.array_equal(zg, zg_ref)                                    # the standing bitstream is unchanged
    zi = np.full(shape, 0.5, np.float32)
    assert np.array_equal(EV.mi_z0(7, 3, 0, shape, 1.0, "perturb", zi, 0.0), zi)
    zd = np.asarray(EV.mi_z0_device(7, [3, 4], 0, shape, 1.0, "trunc"))
    assert zd.shape == (2,) + shape and abs(float(zd.std()) - 1.0) < 0.05 and float(np.abs(zd).max()) <= 2 * comp + 1e-6
    zd2 = np.asarray(EV.mi_z0_device(7, [4, 3], 0, shape, 1.0, "trunc"))
    assert np.array_equal(zd[0], zd2[1]) and not np.array_equal(zd[0], zd[1])   # per-puzzle keys: order-invariant, distinct
    ck = tmp_path / "ckpt_latest.pkl"; _write_ckpt(ck, TRM_TINY)
    arr_p, s_p = _run(ck, tmp_path / "p", "--z0-mode", "perturb", "--z0-eps", "0", k=1)
    assert (arr_p["mi_exact_k"][:, 0].astype(bool) == arr_p["cold_exact"]).all() and s_p["z0_mode"] == "perturb"   # eps 0 = the cold pass
    arr_d, s_d = _run(ck, tmp_path / "d", "--z0-mode", "trunc", "--z0-device", k=2)
    assert s_d["z0_device"] and arr_d["mi_resid_k"].shape == (6, 2)


def test_seg_noise_and_zero_prefix_and_buffers(tmp_path):
    ck = tmp_path / "ckpt_latest.pkl"; _write_ckpt(ck, TRM_TINY)
    a0, s0 = _run(ck, tmp_path / "a0", "--record-q")
    a1, s1 = _run(ck, tmp_path / "a1", "--record-q", "--seg-noise-beta", "0.5")
    a2, s2 = _run(ck, tmp_path / "a2", "--record-q", "--seg-noise-beta", "0.5")
    q0, q1, q2 = (x["q_by_step"].astype(np.float32) for x in (a0, a1, a2))
    assert np.array_equal(q1, q2) and not np.allclose(q0, q1) and s1["seg_noise_beta"] == 0.5   # deterministic, and the noise acts
    ckp = tmp_path / "ckp.pkl"; _write_ckpt(ckp, TRM_TINY, prefix=True)
    ap_, sp_ = _run(ckp, tmp_path / "ap", "--record-q")
    a3, s3 = _run(ckp, tmp_path / "a3", "--record-q", "--zero-prefix")
    assert s3["zero_prefix"] and not np.allclose(ap_["q_by_step"].astype(np.float32), a3["q_by_step"].astype(np.float32))
    assert np.allclose(q0, a3["q_by_step"].astype(np.float32))    # zeroed prefix == the zero-init prefix of the base checkpoint
    # a ported checkpoint's own buffers travel in params and set the cold start
    ck2 = tmp_path / "ck2.pkl"; _write_ckpt(ck2, TRM_TINY, buffers=(3.0 * jnp.ones(32), -2.0 * jnp.ones(32)))
    a4, s4 = _run(ck2, tmp_path / "a4", "--record-q")
    assert not np.allclose(q0, a4["q_by_step"].astype(np.float32))
    ck3 = tmp_path / "ck3.pkl"; _write_ckpt(ck3, TRM_TINY, buffers=TC.init_states(TRM_TINY))    # the seeded buffers, explicit
    a5, s5 = _run(ck3, tmp_path / "a5", "--record-q")
    assert np.allclose(q0, a5["q_by_step"].astype(np.float32))
