# Ledger: E8 ([H-23] basin training at TTT) — arms: +anchor (corrupted-target
# rows incl. eps=0 idempotence), +restarts (self-rollout starts), +anneal
# (tau plateaus + beta_flux>0, E2 merged). e_t-only fit, arm-A hyperparams.
# Metric battery: GT-retention + corruption ladder (E9/S4 spectrum), H[q],
# n_distinct, exactness. Baseline rows = the E1/E3 deployed-fit results.
"""
  python tools/eval_e8.py --ckpt <bulk> --tasks a,b --out runs/e8_X \
      [--anchor] [--restarts] [--anneal]
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

TAU_PLATEAUS = (1.0, 0.8, 0.6, 0.45, 0.35, 0.3)
ANCHOR_EPS = (0.0, 0.1, 0.25)
LADDER_EPS = (0.05, 0.1, 0.2, 0.4)


def corrupt(y: np.ndarray, eps: float, rng: np.random.Generator) -> np.ndarray:
    """Resample ~eps of the true-extent cells uniformly in 0..9. eps=0 = y."""
    out = y.copy()
    n = int(round(eps * y.size))
    if n:
        idx = rng.choice(y.size, size=n, replace=False)
        out.flat[idx] = rng.integers(0, 10, size=n)
    return out


@functools.lru_cache(maxsize=32)
def _e8_step(cfg: Config, tau: float, lr: float, wd: float, beta: float,
             full: bool = False):
    """full=False: e_t-only (B1-B3). full=True (B4, ledger 2026-08-08 NIGHT
    amendment): AdamW over {model, tv} jointly at arm-C lr — the
    capacity × basin cell."""
    from dataclasses import replace
    cfg_b = replace(cfg, beta_flux=beta) if beta else cfg
    opt = optax.adamw(lr, weight_decay=wd)

    @jax.jit
    def step(model, tv, opt_state, rng, x_b, y_b, yp_b):
        def loss_fn(tr):
            m, v = (tr["model"], tr["tv"]) if full else (model, tr)
            tvs = jnp.broadcast_to(v, (x_b.shape[0],) + v.shape)
            loss, _ = batch_loss(m, cfg_b, x_b, y_b, tau=tau, rng=rng,
                                 task_vecs=tvs, yprev_batch=yp_b)
            return loss
        tr0 = {"model": model, "tv": tv} if full else tv
        g = jax.grad(loss_fn)(tr0)
        upd, os2 = opt.update(g, opt_state, tr0)
        tr1 = optax.apply_updates(tr0, upd)
        if full:
            return tr1["model"], tr1["tv"], os2
        return model, tr1, os2
    return step, opt


def fit_e8(state, cfg, eps_list, *, steps, val_every, anchor, restarts, anneal,
           full=False, seed=0, lr=1e-2, wd=1e-4, restart_every=25):
    if full:
        lr = 3e-3  # arm-C value (eval_dev30 ARM_LR["C"])
    model = jax.tree.map(jnp.asarray, state["model"])
    tv = jnp.asarray(np.asarray(state["table"]).mean(0))
    train_pairs = list(eps_list[0].support[:-1])
    x_b, y_b = T.pairs_to_batch(train_pairs, transforms=None, seed=seed)
    B0 = x_b.shape[0]
    rng_np = np.random.default_rng(seed + 7)

    xs, ys, yps = [np.asarray(x_b)], [np.asarray(y_b)], \
        [np.full((B0, G.CANVAS, G.CANVAS), G.VOID, dtype=np.int32)]
    if anchor:
        ax, ay, ap = [], [], []
        for x, y in train_pairs:
            for e in ANCHOR_EPS:
                ax.append(G.place(np.asarray(x)))
                ay.append(G.place(np.asarray(y)))
                ap.append(G.place(corrupt(np.asarray(y), e, rng_np)))
        xs.append(np.stack(ax)); ys.append(np.stack(ay)); yps.append(np.stack(ap))
    n_restart = len(train_pairs) if restarts else 0
    if restarts:  # placeholder rows, refreshed in-loop
        rx = [G.place(np.asarray(x)) for x, _ in train_pairs]
        ry = [G.place(np.asarray(y)) for _, y in train_pairs]
        xs.append(np.stack(rx)); ys.append(np.stack(ry))
        yps.append(np.full((n_restart, G.CANVAS, G.CANVAS), G.VOID, np.int32))
    X = jnp.asarray(np.concatenate(xs), dtype=jnp.int32)
    Y = jnp.asarray(np.concatenate(ys), dtype=jnp.int32)
    YP = jnp.asarray(np.concatenate(yps), dtype=jnp.int32)

    beta = 1e-5 if anneal else 0.0
    opt_state = None
    rng = jax.random.PRNGKey(seed)
    snaps = []
    cur_tau = None
    for i in range(steps):
        tau = (TAU_PLATEAUS[min(int(i / (steps / len(TAU_PLATEAUS))),
                                len(TAU_PLATEAUS) - 1)] if anneal else 1.0)
        step_fn, opt = _e8_step(cfg, tau, lr, wd, beta, full)
        if opt_state is None:
            opt_state = opt.init({"model": model, "tv": tv} if full else tv)
            cur_tau = tau
        if restarts and i % restart_every == 0:
            rows = []
            for x, _ in train_pairs:
                k = int(rng_np.integers(1, 9))
                st = P.trace(model, cfg, np.asarray(x), tau=tau,
                             task_vec=tv, t_total=min(k, 16))
                can = G.place(st[-1]["pred"])
                rows.append(can)
            YP = jnp.asarray(np.concatenate(
                [np.asarray(YP[:X.shape[0] - n_restart]), np.stack(rows)]),
                dtype=jnp.int32)
        rng, sub = jax.random.split(rng)
        model, tv, opt_state = step_fn(model, tv, opt_state, sub, X, Y, YP)
        if (i + 1) % val_every == 0 or i + 1 == steps:
            snaps.append((i + 1, (jax.tree.map(np.asarray, model) if full
                                  else None), np.asarray(tv), tau))
    return model, snaps


def retention(model, cfg, x, target, tv, tau, k=8):
    can = G.place(np.asarray(target))
    st = P.trace(model, cfg, np.asarray(x), tau=tau, task_vec=tv,
                 t_total=k, yprev_init=can, skip_trained=True)
    t = np.asarray(target)
    return bool(all(s["pred"].shape == t.shape and np.array_equal(s["pred"], t)
                    for s in st))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tasks", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--val-every", type=int, default=50)
    ap.add_argument("--anchor", action="store_true")
    ap.add_argument("--restarts", action="store_true")
    ap.add_argument("--anneal", action="store_true")
    ap.add_argument("--full", action="store_true",
                    help="B4: fit model+tv jointly at arm-C lr (capacity cell)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    saved = E.load_ckpt(a.ckpt)
    defaults = Config()
    cfg = Config(**{k: type(getattr(defaults, k))(v)
                    for k, v in saved["config"].items()})
    import dev30
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    results = out / "results.jsonl"
    done = {json.loads(l)["task"] for l in results.read_text().splitlines()} \
        if results.exists() else set()
    task_ids = a.tasks.split(",") if a.tasks else sorted(dev30.MANIFEST)

    with open(results, "a") as f:
        for tid in task_ids:
            if tid in done:
                print(f"skip {tid}", flush=True); continue
            t0 = time.time()
            eps_list = G.load_task(tid)
            model, snaps = fit_e8(saved["state"], cfg, eps_list,
                                  steps=a.steps, val_every=a.val_every,
                                  anchor=a.anchor, restarts=a.restarts,
                                  anneal=a.anneal, full=a.full, seed=a.seed)
            vx, vy = eps_list[0].support[-1]
            # selection: earliest LoO-exact at that snapshot's tau, else last;
            # full arm (B4): each snapshot carries its own model
            def snap_model(m_snap):
                return (jax.tree.map(jnp.asarray, m_snap)
                        if m_snap is not None else model)
            sel = None
            for step_i, m_snap, tv, tau in snaps:
                tr = P.trace(snap_model(m_snap), cfg, vx, tau=tau,
                             task_vec=jnp.asarray(tv), t_total=cfg.T)
                if tr[-1]["pred"].shape == vy.shape and \
                        np.array_equal(tr[-1]["pred"], vy):
                    sel = (step_i, m_snap, tv, tau); break
            if sel is None:
                sel = snaps[-1]
            step_i, m_snap, tv, tau = sel
            model = snap_model(m_snap)
            tvj = jnp.asarray(tv)
            rng_np = np.random.default_rng(a.seed + 13)

            row = {"task": tid, "sel_step": step_i, "sel_tau": tau,
                   "arms": {"anchor": a.anchor, "restarts": a.restarts,
                            "anneal": a.anneal},
                   "loo_retention_gt": retention(model, cfg, vx, vy, tvj, tau),
                   "loo_ladder": {}, "queries": []}
            for e in LADDER_EPS:
                cy = corrupt(np.asarray(vy), e, rng_np)
                can = G.place(cy)
                st = P.trace(model, cfg, np.asarray(vx), tau=tau, task_vec=tvj,
                             t_total=8, yprev_init=can, skip_trained=True)
                vyn = np.asarray(vy)
                row["loo_ladder"][str(e)] = bool(
                    st[-1]["pred"].shape == vyn.shape
                    and np.array_equal(st[-1]["pred"], vyn))
            for ep in eps_list:
                q = P.trace(model, cfg, ep.query_x, tau=tau, task_vec=tvj,
                            t_total=16)
                exact = bool(ep.query_y is not None
                             and q[cfg.T - 1]["pred"].shape == ep.query_y.shape
                             and np.array_equal(q[cfg.T - 1]["pred"], ep.query_y))
                qlad = {}
                if ep.query_y is not None:  # C.2: query eps-ladder (basin radius)
                    for e in LADDER_EPS:
                        cy = corrupt(np.asarray(ep.query_y), e, rng_np)
                        st2 = P.trace(model, cfg, np.asarray(ep.query_x),
                                      tau=tau, task_vec=tvj, t_total=8,
                                      yprev_init=G.place(cy), skip_trained=True)
                        gt = np.asarray(ep.query_y)
                        qlad[str(e)] = bool(
                            st2[-1]["pred"].shape == gt.shape
                            and np.array_equal(st2[-1]["pred"], gt))
                qrec = {"exact_T": exact, "q_ladder": qlad,
                        "Hq": q[cfg.T - 1]["H_q"],
                        "n_distinct": len({s["pred"].tobytes() + bytes(s["pred"].shape)
                                           for s in q}),
                        "gt_retention": (retention(model, cfg, ep.query_x,
                                                   ep.query_y, tvj, tau)
                                         if ep.query_y is not None else None)}
                row["queries"].append(qrec)
            f.write(json.dumps(row) + "\n"); f.flush()
            print(f"{tid} sel@{step_i} tau={tau} looRet={row['loo_retention_gt']} "
                  f"qEx={[q['exact_T'] for q in row['queries']]} "
                  f"qRet={[q['gt_retention'] for q in row['queries']]} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    print("E8 DONE", flush=True)


if __name__ == "__main__":
    main()
