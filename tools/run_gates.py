# Ledger: CI-3a — Phase-1 trainability gate. The triad (identity, color-swap,
# translation) must reach exact match by LoO-validated full-parameter fit at
# toy scale. This is the five-minute test that would have caught April
# (post-mortem E3). CI-3b (frozen-core TTT) re-runs the triad after
# pretraining exists.
#
# AUGMENTATION VALIDITY (ledger 2026-07-20): each task lists its own
# rule-consistent orbit — a transform is admissible only if it commutes with
# the rule. D4 on translate-right is contradictory supervision in full-fit
# mode (measured: train loss stuck at 0.28, val degrading 0.75 -> 0.44).
"""Run: .venv/bin/python tools/run_gates.py [--steps N] [--d D] [--T T] [--only NAME]"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import jax

from qhrrn2 import grid as G
from qhrrn2.config import Config
from qhrrn2.model import count_params, init_params
from qhrrn2.train import evaluate_pair, fit_loo, predict_voted

D4_FULL = [G.Transform(k=k) for k in range(8)]
FLIPUD_ONLY = [G.Transform(k=0), G.Transform(k=6)]  # k=6 == flipud; commutes with x-shifts


def make_task(name: str, rule, n_support: int = 5, hw: int = 8, seed: int = 0) -> G.Episode:
    rng = np.random.default_rng(seed)
    grids = [np.asarray(rng.integers(0, 5, (hw, hw)), dtype=np.int8) for _ in range(n_support + 1)]
    pairs = [(g, rule(g)) for g in grids]
    return G.Episode(task_id=name, support=tuple(pairs[:-1]),
                     query_x=pairs[-1][0], query_y=pairs[-1][1])


TRIAD = {
    # name: (rule, rule-consistent augmentation orbit)
    "identity": (lambda g: g.copy(), D4_FULL),
    "color-swap-1-2": (lambda g: np.where(g == 1, 2, np.where(g == 2, 1, g)).astype(np.int8), D4_FULL),
    "translate-right-1": (lambda g: np.pad(g, ((0, 0), (1, 0)))[:, :-1].astype(np.int8), FLIPUD_ONLY),
}


def run_gate(name: str, ep: G.Episode, cfg: Config, steps: int, transforms, seed: int = 0):
    """LoO-validated fit on the rule-consistent orbit; final query prediction
    uses test-time orbit voting over the same transforms."""
    t0 = time.time()
    params = init_params(jax.random.PRNGKey(seed), cfg)
    params, hist = fit_loo(params, cfg, ep, steps=steps, transforms=transforms,
                           val_every=50, lr=3e-3, tau=1.0, seed=seed,
                           log_every=max(steps // 4, 1))
    # plain single-view prediction (diagnostic) and voted prediction (the gate)
    exact1, pix1, _ = evaluate_pair(params, cfg, ep.query_x, ep.query_y, tau=1.0)
    voted, vshape = predict_voted(params, cfg, ep.query_x, transforms, tau=1.0)
    gt = ep.query_y
    exact = bool(vshape == gt.shape and np.array_equal(voted, gt))
    pix = float((voted == gt).mean()) if vshape == gt.shape else 0.0
    best = hist["best"]
    dt = time.time() - t0
    status = "PASS" if exact else "FAIL"
    print(f"[{status}] {name:<20} voted: exact={exact} pix={pix:.3f} | "
          f"single-view: exact={exact1} pix={pix1:.3f} | "
          f"best@{best['step']} (val_pix {best['val_pix']:.3f}, val_exact {best['val_exact']}) "
          f"({dt:.0f}s)", flush=True)
    if not exact and vshape == gt.shape:
        for (r, c) in np.argwhere(voted != gt):
            print(f"    diff at ({r},{c}): predicted {voted[r, c]}, expected {gt[r, c]} "
                  f"| input cell was {ep.query_x[r, c]}", flush=True)
    return exact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--d", type=int, default=12)
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--only", type=str, default=None)
    args = ap.parse_args()

    cfg = Config(d=args.d, T=args.T)
    print(f"CI-3a sanity triad | d={cfg.d} T={cfg.T} steps={args.steps} "
          f"params={count_params(init_params(jax.random.PRNGKey(0), cfg)):,}", flush=True)

    results = {}
    for name, (rule, transforms) in TRIAD.items():
        if args.only and args.only != name:
            continue
        results[name] = run_gate(name, make_task(name, rule), cfg, args.steps, transforms)

    n_pass = sum(results.values())
    print(f"\nGATE CI-3a: {n_pass}/{len(results)} " + ("PASSED" if n_pass == len(results) else "NOT PASSED"))
    sys.exit(0 if n_pass == len(results) else 1)


if __name__ == "__main__":
    main()
