# Ledger: the pre-cloud gate battery (build-discipline rule: 6/6 before spend).
#   CI-3a (this file's original): triad trainability by LoO-validated full fit.
#   CI-4 (C4): seam-boundary task — block-aligned vs offset checkerboard completion.
#   CI-5 (C5/C14/H-4): flux-direction sanity — identity buys UV transport,
#         constant-fill buys ~nothing; pricing compresses vs free. First (I_s, A_s)
#         spectra of the project.
#   CI-6 (C1): canvas-size prediction across varied sizes, no GT size at predict.
# CI-3b (frozen-core TTT) re-runs the triad after pretraining exists.
#
# AUGMENTATION VALIDITY (ledger 2026-07-20): each task lists its own
# rule-consistent orbit — a transform is admissible only if it commutes with
# the rule. D4 on translate-right is contradictory supervision in full-fit
# mode (measured: train loss stuck at 0.28, val degrading 0.75 -> 0.44).
"""Run: .venv/bin/python tools/run_gates.py [--steps N] [--d D] [--T T]
        [--gate triad,seam,flux,canvas|all] [--only NAME] [--seed S]"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import jax
import jax.numpy as jnp

from qhrrn2 import grid as G
from qhrrn2.config import Config
from qhrrn2.model import count_params, init_params, iterate
from qhrrn2.train import evaluate_pair, fit_loo, predict, predict_voted

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
    return exact, pix, params


def run_triad_gate(cfg: Config, steps: int, seed: int = 0, only: str | None = None):
    results = {}
    for name, (rule, transforms) in TRIAD.items():
        if only and only != name:
            continue
        exact, _, _ = run_gate(name, make_task(name, rule), cfg, steps, transforms, seed=seed)
        results[name] = exact
    n = sum(results.values())
    print(f"GATE CI-3a: {n}/{len(results)} "
          + ("PASSED" if n == len(results) else "NOT PASSED"), flush=True)
    return n == len(results)


# ── CI-4 (C4): seam-boundary gate ──────────────────────────────────────────

def make_checkerboard_task(name: str, *, block_offset: int, n_support: int = 5,
                           hw: int = 8, seed: int = 0) -> G.Episode:
    """Checkerboard completion (spec §10.4): two random colors, random phase,
    ~25% of cells deleted to black; output restores the pattern. Content is
    padded to sit at (2,2) [block-aligned] or (1,1) [maximally block-crossing];
    total grid size is identical, so the ONLY difference between variants is
    alignment to the 2x2 pooling lattice. Pattern corners are never deleted,
    keeping the content extent unambiguous per example."""
    rng = np.random.default_rng(seed)
    pairs = []
    for _ in range(n_support + 1):
        a, b = rng.choice(np.arange(1, 10, dtype=np.int8), size=2, replace=False)
        phase = int(rng.integers(2))
        r, c = np.indices((hw, hw))
        full = np.where((r + c + phase) % 2 == 0, a, b).astype(np.int8)
        x = full.copy()
        mask = rng.random((hw, hw)) < 0.25
        mask[0, 0] = mask[0, -1] = mask[-1, 0] = mask[-1, -1] = False
        if not mask.any():
            mask[hw // 2, hw // 2] = True
        x[mask] = 0
        pad = ((2, 0), (2, 0)) if block_offset == 0 else ((1, 1), (1, 1))
        pairs.append((np.pad(x, pad), np.pad(full, pad)))
    return G.Episode(task_id=name, support=tuple(pairs[:-1]),
                     query_x=pairs[-1][0], query_y=pairs[-1][1])


def run_seam_gate(cfg: Config, steps: int, seed: int = 0):
    """PASS = both variants exact. A seam/disentangler failure shows as the
    offset variant degrading relative to aligned (spec §10.4 epsilon clause
    reported as the pix delta)."""
    res = {}
    for variant, off in (("aligned", 0), ("offset", 1)):
        ep = make_checkerboard_task(f"seam-{variant}", block_offset=off, seed=seed)
        exact, pix, _ = run_gate(f"seam-{variant}", ep, cfg, steps, D4_FULL, seed=seed)
        res[variant] = (exact, pix)
    delta = res["aligned"][1] - res["offset"][1]
    ok = res["aligned"][0] and res["offset"][0]
    print(f"GATE CI-4 seam: aligned exact={res['aligned'][0]} "
          f"offset exact={res['offset'][0]} (pix delta aligned-offset {delta:+.3f}) -> "
          + ("PASSED" if ok else "NOT PASSED"), flush=True)
    return ok


# ── CI-5 (C5/C14/H-4): flux-direction sanity ───────────────────────────────

def run_flux_gate(cfg: Config, steps: int, seed: int = 0):
    """Measured at the LoO-selected solution on the query:
    (a) both tasks must still solve under pricing (beta > 0);
    (b) identity's fine-scale stream flux must dwarf constant-fill's total
        stream flux (UV transport is bought only when the rule needs it);
    (c) pricing must compress identity's total (I+A) vs the free (beta=0) fit.
    Prints the project's first per-task (I_s, A_s) spectra."""
    beta = 1e-4
    cfg_b = replace(cfg, beta_flux=beta, beta_flux_nl=beta)

    def spectra(params, c, ep):
        x_can = jnp.asarray(G.place(ep.query_x), dtype=jnp.int32)
        o = iterate(params, c, x_can, tau=1.0)[-1]
        return np.asarray(o.flux), np.asarray(o.flux_attn)

    ep_id = make_task("flux-identity", lambda g: g.copy(), seed=seed)
    ep_cf = make_task("flux-constfill", lambda g: np.full_like(g, 4), seed=seed)

    ex_id, _, p_id = run_gate("flux-identity(b>0)", ep_id, cfg_b, steps, D4_FULL, seed=seed)
    ex_cf, _, p_cf = run_gate("flux-constfill(b>0)", ep_cf, cfg_b, steps, D4_FULL, seed=seed)
    _, _, p_id0 = run_gate("flux-identity(b=0)", ep_id, cfg, steps, D4_FULL, seed=seed)

    I_id, A_id = spectra(p_id, cfg_b, ep_id)
    I_cf, A_cf = spectra(p_cf, cfg_b, ep_cf)
    I_id0, A_id0 = spectra(p_id0, cfg, ep_id)
    print(f"  identity  (b={beta:g}): I_s={np.round(I_id, 1)}  A_s={np.round(A_id, 1)}")
    print(f"  constfill (b={beta:g}): I_s={np.round(I_cf, 1)}  A_s={np.round(A_cf, 1)}")
    print(f"  identity  (b=0):     I_s={np.round(I_id0, 1)}  A_s={np.round(A_id0, 1)}", flush=True)

    fine_id, tot_cf = float(I_id[0] + I_id[1]), float(I_cf.sum())
    tot_b, tot_0 = float(I_id.sum() + A_id.sum()), float(I_id0.sum() + A_id0.sum())
    c_uv = fine_id > 2.0 * tot_cf
    c_dir = tot_b < 0.5 * tot_0
    ok = ex_id and ex_cf and c_uv and c_dir
    print(f"GATE CI-5 flux: solve(id)={ex_id} solve(cf)={ex_cf} | "
          f"fine(id) {fine_id:.1f} vs total(cf) {tot_cf:.1f} [{'OK' if c_uv else 'X'}] | "
          f"priced {tot_b:.1f} vs free {tot_0:.1f} [{'OK' if c_dir else 'X'}] -> "
          + ("PASSED" if ok else "NOT PASSED"), flush=True)
    return ok


# ── CI-6 (C1): canvas-size gate ────────────────────────────────────────────

def run_canvas_gate(cfg: Config, steps: int, seed: int = 0):
    """Identity with VARIED support sizes and an unseen query size: the canvas
    head must track the input size (no memorized constant), and predict()
    derives (H, W) from the head alone — no ground-truth size in the path."""
    sizes = [(5, 7), (8, 8), (6, 4), (7, 5), (4, 6), (9, 6)]  # last = query, size unseen
    rng = np.random.default_rng(seed)
    grids = [np.asarray(rng.integers(0, 5, s), dtype=np.int8) for s in sizes]
    ep = G.Episode(task_id="canvas-sizes",
                   support=tuple((g, g.copy()) for g in grids[:-1]),
                   query_x=grids[-1], query_y=grids[-1].copy())
    exact, _, params = run_gate("canvas-sizes", ep, cfg, steps, D4_FULL, seed=seed)
    all_ok = True
    for x, y in list(ep.support) + [(ep.query_x, ep.query_y)]:
        _, (h, w), _ = predict(params, cfg, x, tau=1.0)
        ok = (h, w) == y.shape
        all_ok &= ok
        print(f"    true {y.shape} -> predicted ({h}, {w}) {'OK' if ok else 'WRONG'}")
    ok = exact and all_ok
    print(f"GATE CI-6 canvas: query exact={exact} sizes "
          f"{'all OK' if all_ok else 'WRONG'} -> "
          + ("PASSED" if ok else "NOT PASSED"), flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--d", type=int, default=12)
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--gate", type=str, default="triad",
                    help="comma list of triad,seam,flux,canvas — or 'all'")
    ap.add_argument("--only", type=str, default=None, help="triad only: run one task")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = Config(d=args.d, T=args.T)
    print(f"gates [{args.gate}] | d={cfg.d} T={cfg.T} steps={args.steps} "
          f"params={count_params(init_params(jax.random.PRNGKey(0), cfg)):,}", flush=True)

    runners = {
        "triad": lambda: run_triad_gate(cfg, args.steps, seed=args.seed, only=args.only),
        "seam": lambda: run_seam_gate(cfg, args.steps, seed=args.seed),
        "flux": lambda: run_flux_gate(cfg, args.steps, seed=args.seed),
        "canvas": lambda: run_canvas_gate(cfg, args.steps, seed=args.seed),
    }
    names = list(runners) if args.gate == "all" else [s.strip() for s in args.gate.split(",")]
    results = {n: runners[n]() for n in names}
    print("\nSUMMARY: " + "  ".join(f"{n}:{'PASS' if ok else 'FAIL'}"
                                    for n, ok in results.items()), flush=True)
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
