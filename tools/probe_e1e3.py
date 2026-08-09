# Ledger: E1 ([H-6] pt 1 — H[q] order parameter vs correctness) and
# E3/E3b ([H-2] named test — fixed-point convergence vs correctness;
# candidate-stability-as-scoring). Instruments ONLY: this file imports
# forward_fields/build_fields and reimplements the arm-A e_t fit locally so
# src/ stays untouched; the fit's outputs are cross-checked against the
# saved sf6 predictions (tests/test_probe_e1e3.py + the smoke protocol).
# GT-initialized stability (E3b oracle arm) is a DIAGNOSTIC — it feeds no
# solve path and is labeled oracle in every output row.
"""
  python tools/probe_e1e3.py --ckpt runs/pretrain6_d24t64/ckpt_latest.pkl \
      --tasks ca_X,ca_Y --out runs/e1e3_d24t64
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import functools

import numpy as np
import jax
import jax.numpy as jnp

from qhrrn2 import episodic as E
from qhrrn2 import grid as G
from qhrrn2 import model as M
from qhrrn2.config import Config


def entropy(q):
    return float(jnp.mean(-jnp.sum(q * jnp.log(q + 1e-9), axis=-1)))


def trace(params, cfg: Config, x_grid, *, tau: float, task_vec, t_total: int,
          yprev_init=None, skip_trained: bool = False):
    """Mirror model.iterate step-for-step, then keep applying the FINAL map
    (t_norm=1.0) beyond cfg.T — [H-2]'s 'apply the transformation again'.

    yprev_init + skip_trained=True = candidate-stability mode (E3b): start
    from a given answer canvas and apply only the final map.
    Returns per-step dicts: cropped pred, (h,w), H[q], canvas hash."""
    assert not cfg.use_obj, "probe v1 covers non-obj configs (registered)"
    x_can = jnp.asarray(G.place(np.asarray(x_grid)), dtype=jnp.int32)
    if cfg.equilibrium:  # E10 Phase C: continuous-state trace (iterate_eq
        # semantics; skip_trained = final-map-only steps from the given state)
        y = (jax.nn.one_hot(jnp.full((G.CANVAS, G.CANVAS), G.VOID, jnp.int32),
                            M.VOCAB).transpose(2, 0, 1)
             if yprev_init is None else
             jax.nn.one_hot(jnp.asarray(yprev_init, dtype=jnp.int32),
                            M.VOCAB).transpose(2, 0, 1))
        eta = jax.nn.sigmoid(params["eq"]["eta"])
        eta_z = jax.nn.sigmoid(params["eq"]["eta_z"])
        z_c = None
        steps = []
        for t in range(t_total):
            t_norm = 1.0 if skip_trained else                 min(t, cfg.T - 1) / max(cfg.T - 1, 1)
            out = _traced_fwd_eq(cfg, tau, float(t_norm))(
                params, x_can, y, task_vec,
                z_c if z_c is not None else jnp.zeros(1))
            if z_c is None:
                z_c = out.z_fine
            else:
                z_c = z_c + eta_z * (out.z_fine - z_c)
            pcan = jax.nn.softmax(out.logits, axis=-1).transpose(2, 0, 1)
            y = y + eta * (pcan - y)
            canvas = np.asarray(jnp.argmax(out.logits, axis=-1))
            cands = M.size_candidates(x_can)
            p_h = M.size_mixture_probs(out.size_sel_h, out.size_h, cands[0])
            p_w = M.size_mixture_probs(out.size_sel_w, out.size_w, cands[1])
            h = int(jnp.argmax(p_h)) + 1
            w = int(jnp.argmax(p_w)) + 1
            pred = np.where(canvas[:h, :w] == G.VOID, 0,
                            canvas[:h, :w]).astype(np.int8)
            steps.append({"pred": pred, "hw": (h, w),
                          "H_q": entropy(out.rule_q)})
        return steps
    yprev = (jnp.full((G.CANVAS, G.CANVAS), G.VOID, dtype=jnp.int32)
             if yprev_init is None else jnp.asarray(yprev_init, dtype=jnp.int32))
    steps = []
    for t in range(t_total):
        if skip_trained:
            t_norm = 1.0
        else:
            t_norm = min(t, cfg.T - 1) / max(cfg.T - 1, 1)
        out = _traced_fwd(cfg, tau, float(t_norm))(
            params, M.build_fields(x_can, yprev), task_vec)
        canvas = jnp.argmax(out.logits, axis=-1)
        cands = M.size_candidates(x_can)
        p_h = M.size_mixture_probs(out.size_sel_h, out.size_h, cands[0])
        p_w = M.size_mixture_probs(out.size_sel_w, out.size_w, cands[1])
        h = int(jnp.argmax(p_h)) + 1
        w = int(jnp.argmax(p_w)) + 1
        can_np = np.asarray(canvas)
        pred = np.where(can_np[:h, :w] == G.VOID, 0, can_np[:h, :w]).astype(np.int8)
        steps.append({"pred": pred, "hw": (h, w), "H_q": entropy(out.rule_q)})
        yprev = canvas
    return steps


@functools.lru_cache(maxsize=64)
def _traced_fwd_eq(cfg: Config, tau: float, t_norm: float):
    @jax.jit
    def fwd(params, x_can, y_probs, task_vec, z_c):
        z_in = None if z_c.ndim == 1 else z_c
        return M.forward_fields(params, cfg,
                                M.build_fields_soft(x_can, y_probs),
                                t_norm=t_norm, tau=tau, rng=None,
                                task_vec=task_vec, z_in=z_in)
    return fwd


@functools.lru_cache(maxsize=64)
def _traced_fwd(cfg: Config, tau: float, t_norm: float):
    @jax.jit
    def fwd(params, fields, task_vec):
        return M.forward_fields(params, cfg, fields, t_norm=t_norm, tau=tau,
                                rng=None, task_vec=task_vec)
    return fwd


def fit_arm_a(state, cfg: Config, episodes, *, steps: int, val_every: int,
              wd: float = 1e-4, tau: float = 1.0, seed: int = 0):
    """THE deployed arm-A fit (eval_dev30._fit imported, not reimplemented —
    the 2026-08-08 cross-check showed a local replication silently diverged;
    the instrument now shares the measured system's code object). Returns
    (model, snapshots [(step, tv)], selected (step, tv), fit dict)."""
    import eval_dev30 as ED
    snaps = []
    F = ED._fit("A", cfg, state, episodes, steps=steps, val_every=val_every,
                wd=wd, tau=tau, seed=seed, snapshots=snaps)
    # deployed attempt-1 semantics: earliest val-exact, else MDL-best
    if F["first_exact"] is not None:
        sel_tr, sel_step = F["first_exact"]
    else:
        sel_tr, sel_step = F["best"]["trainable"], F["best"]["step"]
    model = sel_tr["model"]
    return model, snaps, (sel_step, np.asarray(F["tv_of"](sel_tr))), F


def stability_record(steps, gt):
    """Convergence + correctness record over a trace."""
    preds = [s["pred"] for s in steps]
    conv_at = None
    for i in range(len(preds) - 1):
        if preds[i].shape == preds[i + 1].shape and np.array_equal(preds[i], preds[i + 1]):
            if all(p.shape == preds[i].shape and np.array_equal(p, preds[i])
                   for p in preds[i + 1:]):
                conv_at = i
                break
    def ex(p):
        return bool(gt is not None and p.shape == gt.shape and np.array_equal(p, gt))
    return {
        "converged_at": conv_at,
        "n_distinct": len({p.tobytes() + bytes(p.shape) for p in preds}),
        "exact_per_step": [ex(p) for p in preds],
        "H_q_per_step": [round(s["H_q"], 4) for s in steps],
        "limit_exact": ex(preds[-1]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tasks", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--val-every", type=int, default=50)
    ap.add_argument("--t-total", type=int, default=16)
    ap.add_argument("--stab-steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    saved = E.load_ckpt(a.ckpt)
    defaults = Config()
    cfg = Config(**{k: type(getattr(defaults, k))(v)
                    for k, v in saved["config"].items()})
    state = saved["state"]

    import dev30
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results = out / "results.jsonl"
    done = set()
    if results.exists():
        for line in results.read_text().splitlines():
            try:
                done.add(json.loads(line)["task"])
            except Exception:
                pass
    task_ids = a.tasks.split(",") if a.tasks else sorted(dev30.MANIFEST)

    with open(results, "a") as f:
        for tid in task_ids:
            if tid in done:
                print(f"skip {tid}", flush=True)
                continue
            t0 = time.time()
            eps = G.load_task(tid)
            model, snaps, sel, F = fit_arm_a(
                state, cfg, eps, steps=a.steps, val_every=a.val_every,
                seed=a.seed)
            vx, vy = eps[0].support[-1]

            # E1: H[q] + exactness per snapshot, on LoO pair and every query
            e1 = []
            for step_i, tv in snaps:
                tvj = jnp.asarray(tv)
                loo = trace(model, cfg, vx, tau=1.0, task_vec=tvj,
                            t_total=cfg.T)
                loo_exact = (loo[-1]["pred"].shape == vy.shape
                             and np.array_equal(loo[-1]["pred"], vy))
                row = {"step": step_i, "loo_exact": bool(loo_exact),
                       "loo_Hq": loo[-1]["H_q"], "queries": []}
                for ep in eps:
                    q = trace(model, cfg, ep.query_x, tau=1.0, task_vec=tvj,
                              t_total=cfg.T)
                    row["queries"].append({
                        "exact": bool(ep.query_y is not None
                                      and q[-1]["pred"].shape == ep.query_y.shape
                                      and np.array_equal(q[-1]["pred"], ep.query_y)),
                        "Hq": q[-1]["H_q"]})
                e1.append(row)

            # E3: extended trace at the deployed-selection tv, clean start
            tvj = jnp.asarray(sel[1])
            e3 = []
            for qi, ep in enumerate(eps):
                steps_tr = trace(model, cfg, ep.query_x, tau=1.0, task_vec=tvj,
                                 t_total=a.t_total)
                e3.append(stability_record(steps_tr, ep.query_y))

            # E3b: candidate stability under the final map (own answer vs GT
            # oracle diagnostic)
            e3b = []
            for qi, ep in enumerate(eps):
                own = trace(model, cfg, ep.query_x, tau=1.0, task_vec=tvj,
                            t_total=cfg.T)[-1]["pred"]
                rec = {}
                for name, init in [("own", own), ("gt_oracle", ep.query_y)]:
                    if init is None:
                        continue
                    can = G.place(np.asarray(init))
                    st = trace(model, cfg, ep.query_x, tau=1.0, task_vec=tvj,
                               t_total=a.stab_steps, yprev_init=can,
                               skip_trained=True)
                    kept = [bool(s["pred"].shape == np.asarray(init).shape
                                 and np.array_equal(s["pred"], init))
                            for s in st]
                    rec[name] = {"retained_per_step": kept,
                                 "retained_all": all(kept)}
                e3b.append(rec)

            f.write(json.dumps({
                "task": tid, "sel_step": sel[0], "e1": e1, "e3": e3,
                "e3b": e3b, "wall_s": round(time.time() - t0, 1)}) + "\n")
            f.flush()
            solved_any = any(q["exact"] for r in e1 for q in r["queries"])
            print(f"{tid} sel@{sel[0]} e1_any_exact={solved_any} "
                  f"e3_conv={[r['converged_at'] for r in e3]} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    print("E1E3 PROBE DONE", flush=True)


if __name__ == "__main__":
    main()
