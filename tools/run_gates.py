# Ledger: CI-3a — Phase-1 trainability gate. The triad (identity, color-swap,
# translation) must reach exact match by full-parameter fit at toy scale.
# This is the five-minute test that would have caught April (post-mortem E3).
# CI-3b (frozen-core TTT) re-runs the triad after pretraining exists.
"""Run: .venv/bin/python tools/run_gates.py [--steps N] [--d D] [--T T]"""
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
from qhrrn2.train import episode_to_batch, fit, predict


def make_task(name: str, rule, n_support: int = 4, hw: int = 8, seed: int = 0) -> G.Episode:
    rng = np.random.default_rng(seed)
    grids = [np.asarray(rng.integers(0, 5, (hw, hw)), dtype=np.int8) for _ in range(n_support + 1)]
    pairs = [(g, rule(g)) for g in grids]
    return G.Episode(task_id=name, support=tuple(pairs[:-1]),
                     query_x=pairs[-1][0], query_y=pairs[-1][1])


TRIAD = {
    "identity": lambda g: g.copy(),
    "color-swap-1-2": lambda g: np.where(g == 1, 2, np.where(g == 2, 1, g)).astype(np.int8),
    "translate-right-1": lambda g: np.pad(g, ((0, 0), (1, 0)))[:, :-1].astype(np.int8),
}


def run_gate(name: str, ep: G.Episode, cfg: Config, steps: int, seed: int = 0):
    t0 = time.time()
    params = init_params(jax.random.PRNGKey(seed), cfg)
    x_b, y_b = episode_to_batch(ep, orbit_n=4, seed=seed, use_palette=False)
    params, losses = fit(params, cfg, x_b, y_b, steps=steps, lr=3e-3,
                         tau=1.0, seed=seed, log_every=max(steps // 4, 1))
    pred, (ph, pw), _ = predict(params, cfg, ep.query_x)
    gt = ep.query_y
    size_ok = (ph, pw) == gt.shape
    exact = size_ok and np.array_equal(pred, gt)
    pix = float((pred[:gt.shape[0], :gt.shape[1]] == gt).mean()) if size_ok else float("nan")
    dt = time.time() - t0
    status = "PASS" if exact else "FAIL"
    print(f"[{status}] {name:<20} exact={exact} size_ok={size_ok} "
          f"pixel_acc={pix:.3f} loss {losses[0]:.3f}->{losses[-1]:.4f} ({dt:.0f}s)")
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
          f"params={count_params(init_params(jax.random.PRNGKey(0), cfg)):,}")

    results = {}
    for name, rule in TRIAD.items():
        if args.only and args.only != name:
            continue
        results[name] = run_gate(name, make_task(name, rule), cfg, args.steps)

    n_pass = sum(results.values())
    print(f"\nGATE CI-3a: {n_pass}/{len(results)} " + ("PASSED" if n_pass == len(results) else "NOT PASSED"))
    sys.exit(0 if n_pass == len(results) else 1)


if __name__ == "__main__":
    main()
