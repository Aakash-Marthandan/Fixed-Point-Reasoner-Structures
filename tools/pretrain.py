# Ledger: C16 pretrain-1 (2026-08-01 entry) — joint-episodic pretraining of the
# shared bulk + per-task embedding table on ARC-1 training minus the FROZEN
# dev-30 holdout. Registered config: d=16, K=64, T=4, tau=1.0, B=64
# task-balanced, AdamW warmup(500)->cosine(1e-3 -> 3e-5), wd 1e-4, clip 1.0,
# 20k steps, beta = beta_nl = 0 with BOTH flux ledgers logged (free-attention
# blowup is a registered risk with a named kill condition, not a surprise).
"""Joint-episodic pretraining (C16).

  .venv/bin/python tools/pretrain.py --out runs/pretrain1          # full run
  .venv/bin/python tools/pretrain.py --smoke --out /tmp/p1smoke    # CPU smoke

Resumable: reruns with the same --out continue from ckpt_latest.pkl (spot
preemption tolerance). Metrics stream to <out>/metrics.jsonl; checkpoints to
<out>/ckpt_latest.pkl (+ step-tagged snapshots); config + corpus manifest to
<out>/config.json at start.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import jax
import jax.numpy as jnp
import optax

import dev30
from qhrrn2 import episodic as E
from qhrrn2 import train as T
from qhrrn2.config import Config
from qhrrn2.model import count_params, init_params
from qhrrn2.objective import batch_loss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--d", type=int, default=16)
    p.add_argument("--K", type=int, default=64)
    p.add_argument("--T", type=int, default=4)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-end", type=float, default=3e-5)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-val", type=int, default=20)
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--val-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--limit", type=int, default=None, help="corpus size cap (smoke)")
    p.add_argument("--val-ids-file", default=None,
                   help="json with {'val40': [...]} — explicit val holdout (CC#2)")
    p.add_argument("--obj", action="store_true", help="C17 cluster layers on")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def git_rev():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10,
                              cwd=Path(__file__).resolve().parents[1]).stdout.strip()
    except Exception:
        return "unknown"


def val20_eval(state, cfg, val, tau):
    """Exact-match on held-out QUERY pairs of val tasks, using each task's
    trained embedding — the within-task generalization monitor (R1 signal).
    Also reports CI-8b within-object consistency (C17 gate-collapse early
    warning, monitored DURING pretraining per the 2026-08-02 registration)."""
    from taxonomy import within_object_consistency
    n_exact = n_total = 0
    pix_sum = 0.0
    c_ok = c_tot = 0
    for t, task_id, queries in val:
        tv = state["table"][t]
        for qx, qy in queries:
            exact, pix, _ = T.evaluate_pair(state["model"], cfg, qx, qy,
                                            tau=tau, task_vec=tv)
            pred, shape, _ = T.predict(state["model"], cfg, qx, tau=tau, task_vec=tv)
            ok, tot = within_object_consistency(pred, qy)
            c_ok += ok
            c_tot += tot
            n_exact += int(exact)
            pix_sum += pix
            n_total += 1
    return {"val_exact": n_exact, "val_total": n_total,
            "val_pix_mean": pix_sum / max(n_total, 1),
            "obj_consistency": round(c_ok / max(c_tot, 1), 4),
            "obj_consistency_n": c_tot}


def main():
    a = parse_args()
    if a.smoke:
        a.steps, a.d, a.K, a.T = 40, 8, 8, 2
        a.batch, a.warmup, a.limit = 8, 5, 8
        a.n_val = 2
        a.ckpt_every = a.val_every = 20
        a.log_every = 5

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = Config(d=a.d, K=a.K, T=a.T, use_obj=a.obj)

    exclude = frozenset(dev30.MANIFEST)
    val_ids = None
    if a.val_ids_file:
        val_ids = json.load(open(a.val_ids_file))["val40"]
    corpus, val = E.build_corpus(exclude, n_val=a.n_val, seed=a.seed, limit=a.limit,
                                 val_ids=val_ids)
    dev = E.corpus_to_device(corpus)
    n_tasks = len(corpus.task_ids)
    n_pairs = int(corpus.x.shape[0])

    key = jax.random.PRNGKey(a.seed)
    k_model, k_table, k_run = jax.random.split(key, 3)
    state = {"model": init_params(k_model, cfg),
             "table": E.init_table(k_table, n_tasks, cfg.d_task)}
    n_bulk = count_params(state["model"])
    n_table = count_params(state["table"])

    sched = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=a.lr, warmup_steps=a.warmup,
        decay_steps=a.steps, end_value=a.lr_end)
    opt = optax.chain(optax.clip_by_global_norm(1.0),
                      optax.adamw(sched, weight_decay=a.wd))
    opt_state = opt.init(state)
    start_step = 0
    rng = k_run

    latest = out / "ckpt_latest.pkl"
    if latest.exists():
        saved = E.load_ckpt(latest)
        state, opt_state = saved["state"], saved["opt_state"]
        start_step, rng = int(saved["step"]), jnp.asarray(saved["rng"])
        print(f"RESUMED from {latest} at step {start_step}", flush=True)

    config_rec = {
        "argv": vars(a), "config": dataclasses.asdict(cfg), "git": git_rev(),
        "n_tasks": n_tasks, "n_pairs": n_pairs,
        "n_params_bulk": n_bulk, "n_params_table": n_table,
        "val_tasks": [task_id for _, task_id, _ in val],
        "backend": jax.default_backend(),
        "corpus_task_ids": list(corpus.task_ids),
    }
    (out / "config.json").write_text(json.dumps(config_rec, indent=1))
    print(f"corpus: {n_tasks} tasks / {n_pairs} pairs; params bulk={n_bulk} "
          f"table={n_table}; backend={jax.default_backend()}", flush=True)

    @jax.jit
    def step_fn(state, opt_state, rng):
        rng, k_batch, k_loss = jax.random.split(rng, 3)
        x_b, y_b, t_b = E.sample_batch(k_batch, dev, n_tasks, a.batch)

        def loss_fn(st):
            return batch_loss(st["model"], cfg, x_b, y_b, tau=a.tau, rng=k_loss,
                              task_vecs=st["table"][t_b])
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state)
        updates, opt_state2 = opt.update(grads, opt_state, state)
        return optax.apply_updates(state, updates), opt_state2, loss, aux, rng

    metrics_f = open(out / "metrics.jsonl", "a")
    t_block = time.time()
    for i in range(start_step, a.steps):
        state, opt_state, loss, aux, rng = step_fn(state, opt_state, rng)

        if (i + 1) % a.log_every == 0 or i + 1 == a.steps:
            dt = time.time() - t_block
            sps = a.log_every / dt if dt > 0 else 0.0
            # aux values are means over batch AND trailing axes; × scales
            # recovers per-pair ledger TOTALS (the A-blowup monitor).
            rec = {
                "step": i + 1,
                "loss": float(loss),
                "ce_in": float(aux["ce_in_last"]),
                "I_total": float(aux["flux_last"]) * cfg.scales,
                "A_total": float(aux["flux_attn_last"]) * cfg.scales,
                "rule_H": float(aux["rule_entropy_last"]),
                "lr": float(sched(i + 1)),
                "steps_per_sec": round(sps, 3),
                "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            metrics_f.write(json.dumps(rec) + "\n")
            metrics_f.flush()
            print(f"step {i+1:6d}  loss {rec['loss']:.4f}  ce {rec['ce_in']:.4f}  "
                  f"I {rec['I_total']:.1f}  A {rec['A_total']:.1f}  "
                  f"{sps:.2f} it/s", flush=True)
            t_block = time.time()

        if (i + 1) % a.val_every == 0 or i + 1 == a.steps:
            v = val20_eval(state, cfg, val, a.tau)
            v["step"] = i + 1
            metrics_f.write(json.dumps({"val": v}) + "\n")
            metrics_f.flush()
            print(f"  VAL step {i+1}: exact {v['val_exact']}/{v['val_total']} "
                  f"pix {v['val_pix_mean']:.3f} objcons {v['obj_consistency']:.3f}"
                  f"(n={v['obj_consistency_n']})", flush=True)
            t_block = time.time()

        if (i + 1) % a.ckpt_every == 0 or i + 1 == a.steps:
            payload = {"state": state, "opt_state": opt_state, "step": i + 1,
                       "rng": np.asarray(rng), "config": dataclasses.asdict(cfg)}
            E.save_ckpt(latest, payload)
            if (i + 1) % (5 * a.ckpt_every) == 0 or i + 1 == a.steps:
                E.save_ckpt(out / f"ckpt_{i+1:06d}.pkl", payload)

    metrics_f.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
