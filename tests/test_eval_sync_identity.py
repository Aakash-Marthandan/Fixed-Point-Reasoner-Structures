"""HOST SYNC POLICY (2026-09-07): the evaluator's batch loop keeps its per-step bits and residuals on the device and reads back once per
rollout; --sync-per-step is the old three-readbacks-per-step loop. The records (cold, first_exact, first_valid, violations, cells,
mi_* draw columns, mi_resid_k, exact_by_step, q_by_step) must be IDENTICAL between the two, on a field-class (trm) cell and on a
native (rg) cell, with draws and with the frontier flags."""
import json, os, subprocess, sys
from pathlib import Path
import numpy as np, jax, jax.numpy as jnp, pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tools"))
from qhrrn2 import episodic as E, model as M
from qhrrn2.config import Config

NPZ = ROOT / "data/sudoku_extreme/sudoku_extreme_seed0.npz"
pytestmark = pytest.mark.skipif(not NPZ.exists(), reason="Sudoku-Extreme npz absent")
FIELD = dict(canvas=9, scales=2, pool_arity=3, mixer_kind="group9", attn_max_hw=9, equilibrium=True, sudoku_layout="native9",
             eta_fixed=1.0, eta_z_fixed=1.0, loss_kind="stablemax")
TRM_TINY = Config(**FIELD, T=3, cell_kind="trm", trm_hidden=32, trm_layers=1, trm_h_cycles=2, trm_l_cycles=2, trm_puzzle_emb_len=2, trm_lambda=0.05)
RG_TINY = Config(d=12, d_ir=16, d_code=16, K=16, T=3, canvas=9, scales=2, pool_arity=3, mixer_kind="group9", attn_max_hw=9, equilibrium=True, sudoku_layout="native9")


def _write(path, cfg, seed=0):
    params = M.init_params(jax.random.PRNGKey(seed), cfg)
    if cfg.cell_kind == "trm":
        params = {**params, "trm": {**params["trm"], "q_head": {"w": 0.1 * jax.random.normal(jax.random.PRNGKey(9), params["trm"]["q_head"]["w"].shape), "b": params["trm"]["q_head"]["b"]}}}
    st = {"model": params, "table": jnp.zeros((1, 32))}
    E.save_ckpt(str(path), dict(config=cfg.__dict__, state=st, state_ema=st, step=0, rng=None, opt_state=None))


def _run(ckpt, out, *flags, n=8, t=4, k=2):
    cmd = [sys.executable, str(ROOT / "tools/eval_sudoku_extreme.py"), "--ckpt", str(ckpt), "--npz", str(NPZ), "--out", str(out), "--split", "test",
           "--limit", str(n), "--t-total", str(t), "--k-init", str(k), "--batch", "4", *flags]
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), JAX_PLATFORMS="cpu")
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=ROOT); assert r.returncode == 0, r.stderr[-2000:]
    return dict(np.load(out / "records_all.npz")), json.loads((out / "summary_all.json").read_text())


@pytest.mark.parametrize("cfg,flags", [(TRM_TINY, ("--record-by-step", "--record-q")), (TRM_TINY, ("--seg-noise-beta", "0.3")), (RG_TINY, ())])
def test_records_identical_between_sync_policies(tmp_path, cfg, flags):
    ck = tmp_path / "ck.pkl"; _write(ck, cfg)
    a, sa = _run(ck, tmp_path / "dev", *flags); b, sb = _run(ck, tmp_path / "step", "--sync-per-step", *flags)
    assert set(a) == set(b)
    for key in a:
        assert np.array_equal(a[key], b[key]), key
    for key in ("exact_acc", "exact_acc_vote", "vote_at_k", "b1_exact", "t1r_at_k", "mean_first_exact", "valid_wrong_frac", "mean_violations"):
        assert sa.get(key) == sb.get(key), key
    assert sa["sync_per_step"] is False and sb["sync_per_step"] is True
