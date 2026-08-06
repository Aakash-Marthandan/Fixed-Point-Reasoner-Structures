# TPU population-segfault bisection (2026-08-06). Each mode isolates one
# suspect; run per-mode in its own process. Prints PROBE-OK <mode> on survival.
"""
  python tools/probe_pop_tpu.py --ckpt <bulk> --mode fit_only|m1|closure|full
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import jax
import jax.numpy as jnp
import optax

from qhrrn2 import episodic as E
from qhrrn2 import grid as G
from qhrrn2 import population as P
from qhrrn2.config import Config
from qhrrn2.objective import batch_loss

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--mode", required=True)
ap.add_argument("--tasks", default="ca_AboveBelow5,ca_AboveBelow6,ca_AboveBelow7")
ap.add_argument("--steps", type=int, default=100)
ap.add_argument("--snaps", type=int, default=2)
a = ap.parse_args()

saved = E.load_ckpt(a.ckpt)
defaults = Config()
cfg = Config(**{k: type(getattr(defaults, k))(v) for k, v in saved["config"].items()})
state = saved["state"]

for tid in a.tasks.split(","):
    eps = G.load_task(tid)
    if a.mode == "fit_only":
        F = P.fit_population(state, cfg, eps, n_views=8, n_seeds=2, steps=a.steps,
                             val_every=50)
        print(f"fit ok {tid}", flush=True)
    elif a.mode == "m1":
        F = P.fit_population(state, cfg, eps, n_views=1, n_seeds=1, steps=a.steps,
                             val_every=50)
        _ = P.score_population(F, eps, max_snap_evals=a.snaps)
        print(f"m1 ok {tid}", flush=True)
    elif a.mode == "closure":
        # closed-over model (pre-refactor form), fresh jit per task
        model = jax.tree.map(jnp.asarray, state["model"])
        tv0 = jnp.asarray(np.asarray(state["table"]).mean(0))
        M = 16
        tv = jnp.broadcast_to(tv0, (M,) + tv0.shape) + 0.0
        x_mb, y_mb, val = P.build_member_batches(eps, 8, 2, 0)
        opt = optax.adamw(1e-2, weight_decay=1e-4)
        os_ = jax.vmap(opt.init)(tv)

        @jax.jit
        def step(tv, os_, rng):
            keys = jax.random.split(rng, M + 1)

            def one(tv_m, os_m, x_b, y_b, key):
                def loss_fn(v):
                    tvs = jnp.broadcast_to(v, (x_b.shape[0],) + v.shape)
                    l, _ = batch_loss(model, cfg, x_b, y_b, tau=1.0, rng=key,
                                      task_vecs=tvs)
                    return l
                l, g = jax.value_and_grad(loss_fn)(tv_m)
                u, o2 = opt.update(g, os_m, tv_m)
                return optax.apply_updates(tv_m, u), o2, l
            tv2, os2, ls = jax.vmap(one)(tv, os_, x_mb, y_mb, keys[:M])
            return tv2, os2, ls, keys[M]
        rng = jax.random.PRNGKey(0)
        for i in range(100):
            tv, os_, ls, rng = step(tv, os_, rng)
        jnp.asarray(ls).block_until_ready()
        print(f"closure ok {tid}", flush=True)
    elif a.mode == "full":
        F = P.fit_population(state, cfg, eps, n_views=8, n_seeds=2, steps=a.steps,
                             val_every=50)
        _ = P.score_population(F, eps, max_snap_evals=a.snaps)
        print(f"full ok {tid}", flush=True)
print(f"PROBE-OK {a.mode}", flush=True)
