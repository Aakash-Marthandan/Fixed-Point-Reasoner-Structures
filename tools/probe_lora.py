# Ledger: [H-12]'s proper test on the equilibrium substrate (phase-plan
# registration 2026-08-11) — BASIN-PRESERVING TTT: LoRA-rank-4 adaptation of
# the compute spine (enc/dec mixers, enc.pool, dec_init, ir_proj) + e_t,
# BULK FROZEN, anchor rows in the fit ([H-23] recipe, eval_e8 lineage).
# The middle path between two measured-dead endpoints: the 64-float keyhole
# (preserves basins, can't express rules — Q2/E8-B1) and full fits (express
# but ERODE — C.2: retention 29%->12.5%). Named tests in tests/test_lora.py:
# B=0 init => merged params BIT-EXACT base (inertness by construction);
# adapted budget <= 25k ([H-12]'s bound); smoke: loss descends, basins hold.
"""
  python tools/probe_lora.py --ckpt runs/pretrain10_a/ckpt_latest.pkl \
      --tasks rt_X,ca_Y --out runs/lora_p10a [--kl 0.0]
"""
from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import jax
import jax.numpy as jnp
import optax

import probe_e1e3 as P
from qhrrn2 import episodic as E
from qhrrn2 import grid as G
from qhrrn2 import train as T
from qhrrn2.config import Config
from qhrrn2.objective import batch_loss

RANK = 4
LORA_ALPHA = 8.0
ANCHOR_EPS = (0.0, 0.1, 0.25)   # E8-B1 registered recipe, incl. idempotence
# v1 target set (registration): the compute spine; attention/canvas/rule
# path excluded (E4: rule transports via tv; canvas is C1-delicate).
TARGETS = [
    ("enc", "mixer", "l1"), ("enc", "mixer", "l2"),
    ("dec", "mixer", "l1"), ("dec", "mixer", "l2"),
    ("enc", "pool", "kept"), ("enc", "pool", "stream"),
    ("dec_init",), ("ir_proj",),
]


def _get(tree, path):
    node = tree
    for k in path:
        node = node[k]
    return node


def lora_init(params, rng):
    """{path_key: {a, b}} with b=0 — merged == base at init, by construction."""
    lora = {}
    for path in TARGETS:
        w = _get(params, path)["w"]
        m, n = w.shape
        rng, sub = jax.random.split(rng)
        lora["/".join(path)] = {
            "a": jax.random.normal(sub, (RANK, n)) / jnp.sqrt(n),
            "b": jnp.zeros((m, RANK)),
        }
    return lora


def lora_params_count(lora):
    return sum(int(np.prod(v["a"].shape)) + int(np.prod(v["b"].shape))
               for v in lora.values())


def merge(params, lora):
    """params with W_eff = W + (alpha/r) * b @ a on each target."""
    out = jax.tree.map(lambda x: x, params)  # shallow-ish copy of the pytree
    scale = LORA_ALPHA / RANK
    for key, ab in lora.items():
        path = tuple(key.split("/"))
        node = _get(out, path)
        node = dict(node)
        node["w"] = node["w"] + scale * (ab["b"] @ ab["a"])
        parent = out
        for k in path[:-1]:
            parent[k] = dict(parent[k])
            parent = parent[k]
        parent[path[-1]] = node
    return out


@functools.lru_cache(maxsize=8)
def _lora_step(cfg: Config, lr: float, wd: float, kl: float):
    opt = optax.adamw(lr, weight_decay=wd)

    def loss_fn(train, base, X, Y, YP, rng):
        eff = merge(base, train["lora"])
        tvs = jnp.broadcast_to(train["tv"], (X.shape[0], train["tv"].shape[0]))
        loss, _ = batch_loss(eff, cfg, X, Y, tau=1.0, rng=rng,
                             task_vecs=tvs, yprev_batch=YP)
        if kl > 0:
            base_tvs = tvs
            bloss, _ = batch_loss(base, cfg, X, Y, tau=1.0, rng=rng,
                                  task_vecs=base_tvs, yprev_batch=YP)
            loss = loss + kl * jnp.square(loss - bloss)
        return loss

    @jax.jit
    def step(train, opt_state, base, X, Y, YP, rng):
        loss, grads = jax.value_and_grad(loss_fn)(train, base, X, Y, YP, rng)
        updates, opt_state = opt.update(grads, opt_state, train)
        return optax.apply_updates(train, updates), opt_state, loss

    return step, opt


def corrupt(y, eps, rng):
    out = y.copy()
    n = int(round(eps * y.size))
    if n:
        idx = rng.choice(y.size, size=n, replace=False)
        out.flat[idx] = rng.integers(0, 10, size=n)
    return out


def fit_lora(state, cfg: Config, eps_list, *, steps, val_every, kl=0.0,
             lr=3e-3, wd=1e-4, seed=0):
    base = jax.tree.map(jnp.asarray, state["model"])
    tv0 = jnp.asarray(np.asarray(state["table"]).mean(0))
    lora0 = lora_init(base, jax.random.PRNGKey(seed + 11))
    train = {"lora": lora0, "tv": tv0}

    train_pairs = list(eps_list[0].support[:-1])
    val_x, val_y = eps_list[0].support[-1]
    x_b, y_b = T.pairs_to_batch(train_pairs, transforms=None, seed=seed)
    B0 = x_b.shape[0]
    rng_np = np.random.default_rng(seed + 7)
    xs = [np.asarray(x_b)]
    ys = [np.asarray(y_b)]
    yps = [np.full((B0, G.CANVAS, G.CANVAS), G.VOID, dtype=np.int32)]
    ax, ay, ap = [], [], []
    for x, y in train_pairs:
        for e in ANCHOR_EPS:
            ax.append(G.place(np.asarray(x)))
            ay.append(G.place(np.asarray(y)))
            ap.append(G.place(corrupt(np.asarray(y), e, rng_np)))
    xs.append(np.stack(ax)); ys.append(np.stack(ay)); yps.append(np.stack(ap))
    X = jnp.asarray(np.concatenate(xs), dtype=jnp.int32)
    Y = jnp.asarray(np.concatenate(ys), dtype=jnp.int32)
    YP = jnp.asarray(np.concatenate(yps), dtype=jnp.int32)

    step, opt = _lora_step(cfg, lr, wd, kl)
    opt_state = opt.init(train)
    rng = jax.random.PRNGKey(seed)
    sel, first_exact = None, None
    for i in range(steps):
        rng, sub = jax.random.split(rng)
        train, opt_state, loss = step(train, opt_state, base, X, Y, YP, sub)
        if (i + 1) % val_every == 0 and first_exact is None:
            eff = merge(base, train["lora"])
            tr = P.trace(eff, cfg, val_x, tau=1.0, task_vec=train["tv"],
                         t_total=cfg.T)
            if tr[-1]["pred"].shape == np.asarray(val_y).shape and \
                    np.array_equal(tr[-1]["pred"], val_y):
                first_exact = (jax.tree.map(lambda x: x, train), i + 1)
    sel = first_exact if first_exact is not None else (train, steps)
    return sel[0], sel[1], base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tasks", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--val-every", type=int, default=100)
    ap.add_argument("--kl", type=float, default=0.0)
    ap.add_argument("--stab-steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    saved = E.load_ckpt(a.ckpt)
    defaults = Config()
    cfg = Config(**{k: type(getattr(defaults, k))(v)
                    for k, v in saved["config"].items()})
    import dev30
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results = out / "results.jsonl"
    done = {json.loads(l)["task"] for l in results.read_text().splitlines()} \
        if results.exists() else set()
    task_ids = a.tasks.split(",") if a.tasks else sorted(dev30.MANIFEST)

    with open(results, "a") as f:
        for tid in task_ids:
            if tid in done:
                print(f"skip {tid}", flush=True)
                continue
            t0 = time.time()
            eps_list = G.load_task(tid)
            train, sel_step, base = fit_lora(
                saved["state"], cfg, eps_list, steps=a.steps,
                val_every=a.val_every, kl=a.kl, seed=a.seed)
            eff = merge(base, train["lora"])
            tvj = jnp.asarray(train["tv"])
            row = {"task": tid, "sel_step": sel_step,
                   "n_adapted": lora_params_count(train["lora"]) + 32,
                   "queries": []}
            for ep in eps_list:
                if ep.query_y is None:
                    continue
                gt = np.asarray(ep.query_y)
                tr = P.trace(eff, cfg, ep.query_x, tau=1.0, task_vec=tvj,
                             t_total=cfg.T)
                st = P.trace(eff, cfg, ep.query_x, tau=1.0, task_vec=tvj,
                             t_total=a.stab_steps, yprev_init=G.place(gt),
                             skip_trained=True)
                row["queries"].append({
                    "exact_T": bool(tr[-1]["pred"].shape == gt.shape
                                    and np.array_equal(tr[-1]["pred"], gt)),
                    "gt_retention": all(
                        bool(s["pred"].shape == gt.shape
                             and np.array_equal(s["pred"], gt)) for s in st)})
            row["wall_s"] = round(time.time() - t0, 1)
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"{tid} sel@{sel_step} "
                  f"ex={[q['exact_T'] for q in row['queries']]} "
                  f"ret={[q['gt_retention'] for q in row['queries']]} "
                  f"({row['wall_s']:.0f}s)", flush=True)
    print("LORA PROBE DONE", flush=True)


if __name__ == "__main__":
    main()
