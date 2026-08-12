# Ledger: cluster M (Research_Brainstorm 2026-08-10; [H-5] built at inference)
# — Langevin/temperature candidate sampling from the equilibrium decoder.
# Instrument ONLY: coverage/diversity measurement, explicitly NOT a snap-vote
# protocol (registration 2026-08-10). GT used only to score distances.
#
# Mechanism (amended 2026-08-10 after the smoke reproduced [R-4] — logit
# noise at eta=0.058 is a silently-dead sampler): STATE-SPACE Langevin at
# the canonical scale — y += eta*(p-y) + sqrt(2*T_t*eta)*xi, xi per-cell
# mean-subtracted (simplex-tangent), clip+renormalize; T_t = T0 for the
# anneal phase (12 steps) then 0 (quench, basins capture); t_total 16.
# Cold VOID start; K samples per (query, T0) differ only in noise seed.
"""
  python tools/probe_sample.py --ckpt runs/pretrain8_d16/ckpt_latest.pkl \
      --tasks ca_X,ca_Y --out runs/samp_p8
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

import probe_e1e3 as P
from qhrrn2 import episodic as E
from qhrrn2 import grid as G
from qhrrn2 import model as M
from qhrrn2.config import Config

TEMPS = (0.1, 0.4)    # registered T0 grid (amended design 2026-08-10)
K_SAMPLES = 16        # registered ensemble size
RADIUS = 0.2          # K's pull radius
T_TOTAL = 16          # anneal 12 + quench 4 (horizon data: capture is fast)
T_ANNEAL = 12
RI_SIGMA = 2.0        # --init random: y0 = softmax(RI_SIGMA * N(0,1)) per cell
                      # (cluster S / EqR multi-init variant, 2026-08-12; the
                      # substrate is NOT RI-trained yet — breadth-at-inference)


def dist(pred, gt) -> float:
    p, g = np.asarray(pred), np.asarray(gt)
    if p.shape != g.shape:
        return 1.0
    return float((p != g).mean())


def trace_langevin(params, cfg: Config, x_grid, *, task_vec, T0: float,
                   seed: int, t_total: int = T_TOTAL, t_anneal: int = T_ANNEAL,
                   init: str = "void"):
    """Eq trace with STATE-SPACE Langevin at the canonical sqrt(2*T*eta)
    scale ([R-4]'s lesson: any other scale fails silently). Noise is
    per-cell mean-subtracted (simplex-tangent); T holds at T0 through the
    anneal phase, then quenches to 0 so basins capture. init='random' draws
    y0 = softmax(RI_SIGMA*N(0,1)) per cell — multi-init breadth (T0=0 with
    random init is the pure EqR-style deterministic-breadth variant).
    Returns final pred."""
    assert cfg.equilibrium
    x_can = jnp.asarray(G.place(np.asarray(x_grid)), dtype=jnp.int32)
    if init == "random":
        g = np.random.default_rng(seed ^ 0x5EED).standard_normal(
            (M.VOCAB, G.CANVAS, G.CANVAS)).astype(np.float32)
        y = jax.nn.softmax(jnp.asarray(RI_SIGMA * g), axis=0)
    else:
        y = jax.nn.one_hot(jnp.full((G.CANVAS, G.CANVAS), G.VOID, jnp.int32),
                           M.VOCAB).transpose(2, 0, 1)
    eta = float(jax.nn.sigmoid(params["eq"]["eta"]))
    eta_z = jax.nn.sigmoid(params["eq"]["eta_z"])
    rng = np.random.default_rng(seed)
    z_c = None
    pred = None
    for t in range(t_total):
        t_norm = min(t, cfg.T - 1) / max(cfg.T - 1, 1)
        out = P._traced_fwd_eq(cfg, 1.0, float(t_norm))(
            params, x_can, y, task_vec,
            z_c if z_c is not None else jnp.zeros(1))
        z_c = out.z_fine if z_c is None else z_c + eta_z * (out.z_fine - z_c)
        pcan = jax.nn.softmax(out.logits, axis=-1).transpose(2, 0, 1)
        y = y + eta * (pcan - y)
        T_t = T0 if t < t_anneal else 0.0
        if T_t > 0:
            xi = rng.standard_normal(np.shape(y)).astype(np.float32)
            xi = xi - xi.mean(axis=0, keepdims=True)   # simplex-tangent
            y = y + np.sqrt(2.0 * T_t * eta) * jnp.asarray(xi)
            y = jnp.clip(y, 0.0, None)
            y = y / jnp.maximum(jnp.sum(y, axis=0, keepdims=True), 1e-6)
        canvas = np.asarray(jnp.argmax(out.logits, axis=-1))
        cands = M.size_candidates(x_can)
        p_h = M.size_mixture_probs(out.size_sel_h, out.size_h, cands[0])
        p_w = M.size_mixture_probs(out.size_sel_w, out.size_w, cands[1])
        h = int(jnp.argmax(p_h)) + 1
        w = int(jnp.argmax(p_w)) + 1
        pred = np.where(canvas[:h, :w] == G.VOID, 0,
                        canvas[:h, :w]).astype(np.int8)
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tasks", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--val-every", type=int, default=50)
    ap.add_argument("--k", type=int, default=K_SAMPLES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temps", default=None,
                    help="comma floats; default = registered grid 0.1,0.4")
    ap.add_argument("--init", choices=("void", "random"), default="void")
    ap.add_argument("--save-preds", action="store_true",
                    help="store distinct endpoint grids + visit counts per "
                         "sigma (candidate source for cluster S / PoE)")
    a = ap.parse_args()
    temps = tuple(float(t) for t in a.temps.split(",")) if a.temps else TEMPS

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
            row = {"task": tid, "sel_step": sel[0], "queries": []}
            for qi, ep in enumerate(eps_list):
                if ep.query_y is None:
                    continue
                gt = np.asarray(ep.query_y)
                det = P.trace(model, cfg, ep.query_x, tau=1.0, task_vec=tvj,
                              t_total=cfg.T)[-1]["pred"]
                qrec = {"det_dist": round(dist(det, gt), 4), "sigmas": {}}
                if a.save_preds:
                    qrec["det_pred"] = np.asarray(det).tolist()
                for T0 in temps:
                    ds, seen = [], {}
                    for k in range(a.k):
                        pr = trace_langevin(
                            model, cfg, ep.query_x, task_vec=tvj, T0=T0,
                            seed=(a.seed * 100003 + qi * 1009 + k
                                  + int(T0 * 10) * 31), init=a.init)
                        d = dist(pr, gt)
                        ds.append(d)
                        key = pr.tobytes() + bytes(pr.shape)
                        if key in seen:
                            seen[key]["n"] += 1
                        else:
                            seen[key] = {"n": 1, "dist": round(d, 4),
                                         "grid": pr.tolist()}
                    rec = {
                        "n_distinct": len(seen),
                        "best_dist": round(min(ds), 4),
                        "within_radius": bool(min(ds) <= RADIUS),
                        "dists": [round(d, 4) for d in ds]}
                    if a.save_preds:
                        rec["cands"] = [{"n": v["n"], "dist": v["dist"],
                                         "grid": v["grid"]}
                                        for v in seen.values()]
                    qrec["sigmas"][str(T0)] = rec
                row["queries"].append(qrec)
            row["wall_s"] = round(time.time() - t0, 1)
            f.write(json.dumps(row) + "\n")
            f.flush()
            summ = {s: sum(q["sigmas"][s]["within_radius"]
                           for q in row["queries"]) for s in map(str, temps)}
            print(f"{tid} sel@{sel[0]} within-radius {summ} "
                  f"({row['wall_s']:.0f}s)", flush=True)
    print("SAMPLE PROBE DONE", flush=True)


if __name__ == "__main__":
    main()
