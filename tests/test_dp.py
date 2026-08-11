# Ledger: P11-EXT DP-trainer named tests (2026-08-11) — pmean'd shard
# gradients must equal the global-batch gradient (the invariant the DP
# path rides), and --dp --smoke must run end-to-end on 2 fake devices.
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EQUIV_SCRIPT = r'''
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
import sys
sys.path.insert(0, "src")
import functools
import numpy as np
import jax, jax.numpy as jnp
from qhrrn2.config import Config
from qhrrn2.model import init_params
from qhrrn2.objective import batch_loss
from qhrrn2 import grid as G

assert jax.local_device_count() == 2
cfg = Config(d=8, K=8, T=2)
params = init_params(jax.random.PRNGKey(0), cfg)
rng = np.random.default_rng(3)
X = jnp.asarray(np.stack([G.place(rng.integers(0, 10, (6, 6)).astype(np.int8))
                          for _ in range(8)]), dtype=jnp.int32)
Y = jnp.asarray(np.stack([G.place(rng.integers(0, 10, (6, 6)).astype(np.int8))
                          for _ in range(8)]), dtype=jnp.int32)

def loss_fn(p, x, y):
    l, _ = batch_loss(p, cfg, x, y, tau=1.0)
    return l

g_full = jax.grad(loss_fn)(params, X, Y)

Xs = X.reshape(2, 4, *X.shape[1:])
Ys = Y.reshape(2, 4, *Y.shape[1:])
p_rep = jax.tree.map(lambda t: jnp.stack([t] * 2), params)

@functools.partial(jax.pmap, axis_name="dp")
def shard_grad(p, x, y):
    return jax.lax.pmean(jax.grad(loss_fn)(p, x, y), "dp")

g_dp = jax.tree.map(lambda t: t[0], shard_grad(p_rep, Xs, Ys))
diffs = jax.tree.map(lambda a, b: float(jnp.max(jnp.abs(a - b))), g_full, g_dp)
md = max(jax.tree.leaves(diffs))
print(f"MAX_DIFF={md:.2e}")
'''


def test_dp_gradient_equivalence():
    r = subprocess.run([sys.executable, "-c", EQUIV_SCRIPT], cwd=ROOT,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    line = [l for l in r.stdout.splitlines() if l.startswith("MAX_DIFF=")][-1]
    md = float(line.split("=")[1])
    assert md < 1e-5, f"pmean shard grads diverge from global grad: {md}"


def test_dp_smoke_end_to_end(tmp_path):
    env = dict(os.environ,
               XLA_FLAGS="--xla_force_host_platform_device_count=2")
    r = subprocess.run(
        [sys.executable, "tools/pretrain.py", "--smoke", "--dp",
         "--equilibrium", "--anchor-p", "0.3", "--out", str(tmp_path / "dp")],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=1200)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "DP: 2 devices" in r.stdout, r.stdout[-1000:]
    assert "DONE" in r.stdout
    metrics = (tmp_path / "dp" / "metrics.jsonl").read_text().splitlines()
    losses = [json.loads(l)["loss"] for l in metrics if "loss" in json.loads(l)]
    assert len(losses) >= 2 and losses[-1] < losses[0] * 1.2
