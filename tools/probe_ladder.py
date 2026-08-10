# Ledger: cluster J stage-2 (basin equation of state — S(eps) surface across
# substrates) + cluster L baseline (stationary-flux transport meter) +
# the queued priced-vs-unpriced flux-spectra comparison (pretrain-9
# registration, 2026-08-10). Instrument ONLY — no solve path touched.
#
# Per task: the battery's exact arm-A keyhole fit (probe_e1e3.fit_arm_a —
# same code object as every bat_* number), then per query pair:
#   - eps=0 GT-oracle retention through 8 final-map steps (e3b semantics)
#   - eps-ladder {.05,.1,.2,.4}: corrupt true-extent cells (eval_e8.corrupt
#     semantics), 8 final-map steps, retained?
#   - corruption masks are seeded per (seed, task, query, eps) so all
#     substrates see IDENTICAL corruptions -> paired McNemar per rung
#     (improvement over C.2's sequential rng; noted for the analyzer)
#   - stationary flux spectra: per-scale I_s (stream) and A_s (attention)
#     captured at the LAST trace step (converged regime, z-carry intact) on
#     the query AND on every support input -> cluster L's mismatch
#     |I_s(query) - I_s(supports)| computable downstream; across arms
#     (P9-C priced vs A/B unpriced) -> the registered spectra comparison.
"""
  python tools/probe_ladder.py --ckpt runs/pretrain9_c/ckpt_latest.pkl \
      --tasks ca_X,ca_Y --out runs/lad_p9c
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import jax
import jax.numpy as jnp

import probe_e1e3 as P
from qhrrn2 import episodic as E
from qhrrn2 import grid as G
from qhrrn2 import model as M
from qhrrn2.config import Config

LADDER_EPS = (0.05, 0.1, 0.2, 0.4)  # C.2's rungs, kept for comparability


def corrupt(y: np.ndarray, eps: float, rng: np.random.Generator) -> np.ndarray:
    """Resample ~eps of the true-extent cells uniformly in 0..9 (eval_e8)."""
    out = y.copy()
    n = int(round(eps * y.size))
    if n:
        idx = rng.choice(y.size, size=n, replace=False)
        out.flat[idx] = rng.integers(0, 10, size=n)
    return out


def rung_rng(seed: int, tid: str, qi: int, eps: float) -> np.random.Generator:
    key = f"{tid}|{qi}|{eps}".encode()
    return np.random.default_rng((zlib.crc32(key) ^ (seed * 2654435761)) & 0x7FFFFFFF)


def trace_flux(params, cfg: Config, x_grid, *, tau: float, task_vec,
               t_total: int, yprev_init=None, skip_trained: bool = False):
    """probe_e1e3.trace for equilibrium configs, additionally recording the
    per-scale flux spectra (I_s, A_s) each step. Kept in lockstep with the
    probe's eq branch (same _traced_fwd_eq, same y/z updates)."""
    assert cfg.equilibrium, "flux trace is eq-only (J-2 scope)"
    x_can = jnp.asarray(G.place(np.asarray(x_grid)), dtype=jnp.int32)
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
        t_norm = 1.0 if skip_trained else min(t, cfg.T - 1) / max(cfg.T - 1, 1)
        out = P._traced_fwd_eq(cfg, tau, float(t_norm))(
            params, x_can, y, task_vec,
            z_c if z_c is not None else jnp.zeros(1))
        z_c = out.z_fine if z_c is None else z_c + eta_z * (out.z_fine - z_c)
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
                      "I_s": [round(float(v), 4) for v in np.asarray(out.flux)],
                      "A_s": [round(float(v), 4) for v in np.asarray(out.flux_attn)]})
    return steps


def retained(model, cfg, x, target, tvj, k=8):
    """target stays fixed under k final-map steps? Returns per-step bools."""
    can = G.place(np.asarray(target))
    st = P.trace(model, cfg, x, tau=1.0, task_vec=tvj, t_total=k,
                 yprev_init=can, skip_trained=True)
    tgt = np.asarray(target)
    return [bool(s["pred"].shape == tgt.shape and np.array_equal(s["pred"], tgt))
            for s in st]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tasks", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--val-every", type=int, default=50)
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
            eps_list = G.load_task(tid)
            model, snaps, sel, F = P.fit_arm_a(
                state, cfg, eps_list, steps=a.steps, val_every=a.val_every,
                seed=a.seed)
            tvj = jnp.asarray(sel[1])

            row = {"task": tid, "sel_step": sel[0], "queries": [],
                   "supports": []}
            # support-side stationary spectra (cluster L reference set)
            for sx, sy in eps_list[0].support:
                st = trace_flux(model, cfg, sx, tau=1.0, task_vec=tvj,
                                t_total=cfg.T)
                row["supports"].append({
                    "I_s": st[-1]["I_s"], "A_s": st[-1]["A_s"],
                    "exact": bool(st[-1]["pred"].shape == np.asarray(sy).shape
                                  and np.array_equal(st[-1]["pred"], sy))})
            for qi, ep in enumerate(eps_list):
                if ep.query_y is None:
                    continue
                gt = np.asarray(ep.query_y)
                qtr = trace_flux(model, cfg, ep.query_x, tau=1.0,
                                 task_vec=tvj, t_total=cfg.T)
                r0 = retained(model, cfg, ep.query_x, gt, tvj, a.stab_steps)
                lad = {}
                for e in LADDER_EPS:
                    rng = rung_rng(a.seed, tid, qi, e)
                    cy = corrupt(gt, e, rng)
                    st = P.trace(model, cfg, ep.query_x, tau=1.0,
                                 task_vec=tvj, t_total=a.stab_steps,
                                 yprev_init=G.place(cy), skip_trained=True)
                    lad[str(e)] = bool(
                        st[-1]["pred"].shape == gt.shape
                        and np.array_equal(st[-1]["pred"], gt))
                row["queries"].append({
                    "exact_T": bool(qtr[-1]["pred"].shape == gt.shape
                                    and np.array_equal(qtr[-1]["pred"], gt)),
                    "gt_retention": all(r0), "retained_per_step": r0,
                    "q_ladder": lad,
                    "I_s": qtr[-1]["I_s"], "A_s": qtr[-1]["A_s"]})
            row["wall_s"] = round(time.time() - t0, 1)
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"{tid} sel@{sel[0]} "
                  f"ret={[q['gt_retention'] for q in row['queries']]} "
                  f"lad={[list(q['q_ladder'].values()) for q in row['queries']]} "
                  f"({row['wall_s']:.0f}s)", flush=True)
    print("LADDER PROBE DONE", flush=True)


if __name__ == "__main__":
    main()
