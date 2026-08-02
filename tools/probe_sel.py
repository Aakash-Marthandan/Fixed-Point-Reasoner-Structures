# Ledger: CC#2 Phase A (2026-08-02) — sel-logit fit-trace probe on d631b094.
# Question: under a B2-style fit (e_t + color_bias + canvas.sel trainable),
# do the selection logits MOVE at all, and does q ever leave candidate 0?
# Distinguishes optimizer-scale stall (logits move too slowly) from
# gradient-structure stall (no signal reaches sel) from moved-but-wrong.
"""
  python tools/probe_sel.py --ckpt runs/pretrain2/ckpt_latest.pkl --steps 2000
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import jax
import jax.numpy as jnp
import optax

from eval_dev30 import _arm_step, ARM_LR
from qhrrn2 import episodic as E
from qhrrn2 import grid as G
from qhrrn2 import train as T
from qhrrn2.config import Config
from qhrrn2.model import build_fields, forward_fields, size_candidates


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--task", default="d631b094")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=100)
    args = p.parse_args()

    saved = E.load_ckpt(args.ckpt)
    defaults = Config()
    cfg = Config(**{k: type(getattr(defaults, k))(v)
                    for k, v in saved["config"].items()})
    model = jax.tree.map(jnp.asarray, saved["state"]["model"])
    tv0 = jnp.asarray(np.asarray(saved["state"]["table"]).mean(0))
    trainable = {"model": model, "tv": tv0}

    eps = G.load_task(args.task)
    support = list(eps[0].support)
    x_b, y_b = T.pairs_to_batch(support[:-1], transforms=None, seed=0)
    val_x, val_y = support[-1]

    step, make_opt = _arm_step(cfg, "B2", ARM_LR["B2"], 1e-4, 1.0, True)
    opt_state = make_opt(trainable).init(trainable)

    def sel_snapshot(tr):
        xc = jnp.asarray(G.place(val_x), dtype=jnp.int32)
        yprev = jnp.full((G.CANVAS, G.CANVAS), G.VOID, dtype=jnp.int32)
        out = forward_fields(tr["model"], cfg, build_fields(xc, yprev),
                             t_norm=1.0, tau=1.0, task_vec=tr["tv"])
        qh = jax.nn.softmax(out.size_sel_h)
        qw = jax.nn.softmax(out.size_sel_w)
        return (np.asarray(out.size_sel_h), np.asarray(qh),
                np.asarray(out.size_sel_w), np.asarray(qw))

    lh0, qh0, lw0, qw0 = sel_snapshot(trainable)
    print(f"step 0: sel_h logits {np.round(lh0, 2)} q_h {np.round(qh0, 3)}")
    print(f"        sel_w logits {np.round(lw0, 2)} q_w {np.round(qw0, 3)}")

    rng = jax.random.PRNGKey(0)
    for i in range(args.steps):
        rng, sub = jax.random.split(rng)
        trainable, opt_state, loss, _ = step(trainable, opt_state, sub, x_b, y_b)
        if (i + 1) % args.log_every == 0:
            lh, qh, lw, qw = sel_snapshot(trainable)
            exact, pix, _ = T.evaluate_pair(trainable["model"], cfg, val_x, val_y,
                                            tau=1.0, task_vec=trainable["tv"])
            pred, shape, _ = T.predict(trainable["model"], cfg, val_x, tau=1.0,
                                       task_vec=trainable["tv"])
            print(f"step {i+1:5d} loss {float(loss):.4f} val_exact {exact} "
                  f"pix {pix:.3f} pred_size {shape} true {val_y.shape}")
            print(f"   dlogit_h {np.round(lh - lh0, 3)}  q_h {np.round(qh, 3)}")
            print(f"   dlogit_w {np.round(lw - lw0, 3)}  q_w {np.round(qw, 3)}",
                  flush=True)
    print("PROBE DONE")


if __name__ == "__main__":
    main()
