# Ledger: the S1/S2 measurement engine (thesis §5: T1/T2 ↔ H-4/H-9).
# Per (task, beta, seed): beta-warmup LoO fit (CI-5 protocol law: free phase
# then priced phase; MDL selection inside fit_loo), then one JSONL row with
# accuracy, the LoO generalization gap (S2's y-axis), and both flux ledgers
# (I_s streams, A_s attention — S3's decomposition) at the selected params.
# Floors are computed OFFLINE from the beta->0+ frontier these rows trace
# (S1's certified upper estimator) plus the per-family analytic lower bounds.
"""Run: .venv/bin/python tools/measure.py --tasks identity,constfill \
         --betas 0,1e-5,1e-4 --seeds 2 --steps 600 [--out runs/measure]"""
from __future__ import annotations

import argparse
import json
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
from qhrrn2.model import init_params, iterate
from qhrrn2.train import evaluate_pair, fit_loo, predict_voted

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_gates import D4_FULL, FLIPUD_ONLY, make_checkerboard_task, make_task

# Constructed families (S1: min-cuts hand-computable on these; the gate tasks
# double as the worked examples — thesis §2 sketch (ii)).
FAMILIES = {
    "identity": (lambda s: make_task("identity", lambda g: g.copy(), seed=s), D4_FULL),
    "constfill": (lambda s: make_task("constfill", lambda g: np.full_like(g, 4), seed=s), D4_FULL),
    "colorswap": (lambda s: make_task(
        "colorswap", lambda g: np.where(g == 1, 2, np.where(g == 2, 1, g)).astype(np.int8),
        seed=s), D4_FULL),
    "translate": (lambda s: make_task(
        "translate", lambda g: np.pad(g, ((0, 0), (1, 0)))[:, :-1].astype(np.int8),
        seed=s), FLIPUD_ONLY),
    "checkerboard": (lambda s: make_checkerboard_task("checkerboard", block_offset=1, seed=s),
                     D4_FULL),
}


def measure_one(family: str, beta: float, seed: int, steps: int, cfg0: Config):
    make, transforms = FAMILIES[family]
    ep = make(seed)
    t0 = time.time()
    params = init_params(jax.random.PRNGKey(seed), cfg0)
    if beta > 0:  # CI-5 protocol law: warmup free, then price
        s1, s2 = steps // 2, steps - steps // 2
        _, hist = fit_loo(params, cfg0, ep, steps=s1, transforms=transforms,
                          val_every=50, tau=1.0, seed=seed)
        cfg = replace(cfg0, beta_flux=beta, beta_flux_nl=beta)
        params, hist = fit_loo(hist["final_params"], cfg, ep, steps=s2,
                               transforms=transforms, val_every=50, tau=1.0,
                               seed=seed + 1)
    else:
        cfg = cfg0
        params, hist = fit_loo(params, cfg, ep, steps=steps, transforms=transforms,
                               val_every=50, tau=1.0, seed=seed)

    # Accuracy: voted query prediction (deployment condition).
    voted, vshape = predict_voted(params, cfg, ep.query_x, transforms, tau=1.0)
    exact = bool(vshape == ep.query_y.shape and np.array_equal(voted, ep.query_y))
    pix = float((voted == ep.query_y).mean()) if vshape == ep.query_y.shape else 0.0

    # S2's gap: mean support-pair accuracy (what the fit could memorize) minus
    # held-out val accuracy (what it generalized) at the SELECTED params.
    sup_pix = float(np.mean([evaluate_pair(params, cfg, x, y, tau=1.0)[1]
                             for x, y in ep.support[:-1]]))
    val_exact, val_pix, _ = evaluate_pair(params, cfg, *ep.support[-1], tau=1.0)

    # Flux ledgers at the selected params on the query (deterministic, b=mu).
    x_can = jnp.asarray(G.place(ep.query_x), dtype=jnp.int32)
    o = iterate(params, cfg, x_can, tau=1.0)[-1]
    return {
        "family": family, "beta": beta, "seed": seed, "steps": steps,
        "d": cfg0.d, "T": cfg0.T,
        "exact": exact, "pix": pix,
        "support_pix": sup_pix, "val_pix": float(val_pix),
        "val_exact": bool(val_exact),
        "loo_gap": sup_pix - float(val_pix),
        "I_s": [float(v) for v in np.asarray(o.flux)],
        "A_s": [float(v) for v in np.asarray(o.flux_attn)],
        "I_total": float(np.sum(np.asarray(o.flux))),
        "A_total": float(np.sum(np.asarray(o.flux_attn))),
        "sel_step": hist["best"]["step"], "wall_s": round(time.time() - t0, 1),
    }


def measure_arc(task_id: str, family: str, beta: float, seed: int, steps: int,
                cfg0: Config):
    """Real-ARC row: identity-only orbit (a real rule's consistent orbit is
    unknown a priori — augmentation-validity law, ledger 2026-07-20); episode
    from the vendored training split."""
    ep = G.load_task(task_id)[0]
    transforms = [G.Transform(k=0)]
    FAMILIES[f"arc:{task_id}"] = (lambda s: ep, transforms)  # reuse the engine
    try:
        row = measure_one(f"arc:{task_id}", beta, seed, steps, cfg0)
    finally:
        del FAMILIES[f"arc:{task_id}"]
    row["family"] = family
    row["task_id"] = task_id
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="identity,constfill",
                    help=f"comma list from {sorted(FAMILIES)} or 'all'")
    ap.add_argument("--arc", action="store_true",
                    help="measure the dev-30 manifest (real tasks) instead of "
                         "constructed families")
    ap.add_argument("--shard", default=None,
                    help="'i/K': run rows with grid-index %% K == i (chip-parallel "
                         "sweeps via tools/shard_run.sh; PI throughput directive "
                         "2026-07-28)")
    ap.add_argument("--betas", default="0,1e-4")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--d", type=int, default=12)
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--attn-max-hw", type=int, default=32,
                    help="0 = attention-absent (the S3 locality-CLASS ablation, "
                         "thesis §6-S3 amendment 2026-07-30)")
    ap.add_argument("--out", default="runs/measure")
    args = ap.parse_args()

    cfg0 = Config(d=args.d, T=args.T, attn_max_hw=args.attn_max_hw)
    betas = [float(b) for b in args.betas.split(",")]
    if args.arc:
        from dev30 import MANIFEST
        units = [(tid, fam) for tid, (fam, _) in sorted(MANIFEST.items())]
    else:
        fams = sorted(FAMILIES) if args.tasks == "all" else args.tasks.split(",")
        units = [(f, f) for f in fams]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = ""
    shard_i, shard_k = 0, 1
    if args.shard:
        shard_i, shard_k = (int(v) for v in args.shard.split("/"))
        suffix = f"-s{shard_i}"
    out = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}{suffix}.jsonl"

    grid = [(u, f, b, s) for u, f in units for b in betas for s in range(args.seeds)]
    grid = [g for i, g in enumerate(grid) if i % shard_k == shard_i]
    print(f"measure: {len(grid)} rows (shard {shard_i}/{shard_k}) -> {out}", flush=True)
    with open(out, "w") as f:
        for unit, fam, beta, seed in grid:
            if args.arc:
                row = measure_arc(unit, fam, beta, seed, args.steps, cfg0)
            else:
                row = measure_one(unit, beta, seed, args.steps, cfg0)
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"  {unit:<14} b={beta:<8g} s={seed} exact={row['exact']} "
                  f"pix={row['pix']:.3f} gap={row['loo_gap']:+.3f} "
                  f"I={row['I_total']:.0f} A={row['A_total']:.0f} "
                  f"({row['wall_s']}s)", flush=True)
    print(f"done -> {out}", flush=True)


if __name__ == "__main__":
    main()
