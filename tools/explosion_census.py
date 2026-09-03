# Ledger: sportC1 §4.5 — the EXPLOSION CENSUS as a standing tool (H-50, report
# §7.1/§7.4; the 2026-09-02 pass ran it from a scratchpad, hence this file).
# Cold (VOID-start) trajectories on the evaluator's rating-stratified test
# puzzles (strat-seed 20260821, the instrument set), inference at the ckpt's
# learned dampings (model.eq_etas), max |z_c| and max |logit| recorded per step
# per puzzle; a puzzle EXPLODED iff any step is non-finite or max |z_c| > 1e6.
# Rows: t=64 on all n puzzles and t=t_long on the first n_long (the depth read).
"""
  .venv/bin/python tools/explosion_census.py --ckpt runs/X/ckpt_latest.pkl \
      --npz data/sudoku_extreme/sudoku_extreme_seed0.npz --out runs/sxcensus_X \
      [--name X@final] [--ema] [--n 512] [--t 64] [--t-long 256] [--n-long 64] [--batch 128]
  -> <out>/census.json  (list of rows: name, eta, eta_z, n, t, exploded_frac, n_exploded,
                         first_bad_median, zmax_median, zmax_p99, logit_max_p99, ema)
     <out>/records.npz  (idx, per-puzzle first_bad + zmax peak for each row)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import jax
import jax.numpy as jnp

from qhrrn2 import episodic as E
from qhrrn2 import grid as G
from qhrrn2 import model as M
from qhrrn2 import sudoku as SU
from qhrrn2 import sudoku_extreme as SX
from qhrrn2.config import Config

Z_LIMIT = 1e6


def trajectories(params, cfg, tvj, x_can, y0, *, t_total, eta, eta_z):
    """(t_total, B) max|z_c| and max|logit| per step (NaN where non-finite);
    the evaluator's own step function (EV._step) so the census reads exactly
    the deployed dynamics (ab-coupled ckpts unsupported here: damped form only)."""
    import eval_sudoku_extreme as EV
    B = x_can.shape[0]
    y, z_c = y0, None
    zmax, lmax = [], []
    trm = getattr(cfg, 'cell_kind', 'rg') == 'trm'
    K = 1 if trm else max(1, int(getattr(cfg, "inner_k", 1)))   # sportC2 R2: mirrors the evaluator
    for t in range(t_total):
        t_norm = min(t, cfg.T - 1) / max(cfg.T - 1, 1)
        if trm:
            t_norm = 0.0   # the field cell ignores t_norm (one compiled step)
        for _k in range(K):
            first = z_c is None
            logits, zf = EV._step(cfg, 1.0, float(t_norm), first)(params, x_can, y, tvj, jnp.zeros(1) if first else z_c)
            z_c = zf if first else z_c + eta_z * (zf - z_c)
        p = jax.nn.softmax(logits, axis=-1).transpose(0, 3, 1, 2)
        y = y + eta * (p - y)
        zmax.append(np.asarray(jnp.max(jnp.abs(z_c).reshape(B, -1), axis=1)))
        lmax.append(np.asarray(jnp.max(jnp.abs(logits).reshape(B, -1), axis=1)))
    return np.stack(zmax), np.stack(lmax)


def classify(zmax):
    """zmax (T, B) -> exploded (B,) bool, first_bad (B,) step or -1, peak finite |z| (B,)."""
    bad = ~np.isfinite(zmax) | (zmax > Z_LIMIT)
    exploded = bad.any(axis=0)
    first_bad = np.where(exploded, bad.argmax(axis=0), -1)
    fin = np.where(np.isfinite(zmax), zmax, np.nan)
    with np.errstate(all="ignore"):
        peak = np.nanmax(fin, axis=0)
    return exploded, first_bad, peak


def summarize_row(name, eta, eta_z, zmax, lmax, t, ema):
    exploded, first_bad, peak = classify(zmax)
    n = zmax.shape[1]
    lfin = np.where(np.isfinite(lmax), lmax, np.nan)
    with np.errstate(all="ignore"):
        lpeak = np.nanmax(lfin, axis=0)
    return dict(name=name, eta=round(float(eta), 4), eta_z=round(float(eta_z), 4), n=int(n), t=int(t),
                exploded_frac=float(exploded.mean()), n_exploded=int(exploded.sum()),
                first_bad_median=(float(np.median(first_bad[exploded])) if exploded.any() else None),
                zmax_median=float(np.nanmedian(peak)), zmax_p99=float(np.nanpercentile(peak, 99)),
                logit_max_p99=float(np.nanpercentile(lpeak, 99)), ema=bool(ema)), exploded, first_bad, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--npz", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--name", default=None); ap.add_argument("--ema", action="store_true")
    ap.add_argument("--n", type=int, default=512); ap.add_argument("--strat-seed", type=int, default=20260821)
    ap.add_argument("--t", type=int, default=64); ap.add_argument("--t-long", type=int, default=256); ap.add_argument("--n-long", type=int, default=64)
    ap.add_argument("--batch", type=int, default=128)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    saved = E.load_ckpt(a.ckpt)
    defaults = Config()
    cfg = Config(**{k: type(getattr(defaults, k))(v) for k, v in saved["config"].items()})
    st = saved["state_ema"] if a.ema else saved["state"]
    assert st is not None, "--ema: no EMA weights in this checkpoint"
    params = st["model"]; tvj = jnp.asarray(st["table"][0])
    eta, eta_z = (float(v) for v in M.eq_etas(params, cfg))
    layout = getattr(cfg, "sudoku_layout", "origin") or "origin"
    cv = SU.layout_canvas(layout)
    d = SX.load_prepared(a.npz)
    Q, R = d["test_q"], d["test_rating"]
    sel = SX.stratified_subsample(R, a.n, a.strat_seed)
    name = a.name or Path(a.ckpt).parent.name
    t0 = time.time()
    rows, recs = [], {}
    for tag, tt, nn in (("t%d" % a.t, a.t, len(sel)), ("t%d" % a.t_long, a.t_long, min(a.n_long, len(sel)))):
        if nn <= 0 or tt <= 0:
            continue
        ids = sel[:nn]
        zm, lm = [], []
        for s in range(0, nn, a.batch):
            b = ids[s:s + a.batch]
            puz9 = Q[b].astype(np.int32)
            x_can = jnp.asarray(np.stack([SU.place_layout(g.astype(np.int8), layout) for g in puz9]), jnp.int32)
            void = jax.nn.one_hot(jnp.full((cv, cv), G.VOID, jnp.int32), M.VOCAB).transpose(2, 0, 1)
            y0 = jnp.broadcast_to(void, (len(b),) + void.shape)
            z, l = trajectories(params, cfg, tvj, x_can, y0, t_total=tt, eta=eta, eta_z=eta_z)
            zm.append(z); lm.append(l)
        zmax, lmax = np.concatenate(zm, axis=1), np.concatenate(lm, axis=1)
        row, exploded, first_bad, peak = summarize_row(f"{name} [{tag}, n={nn}]", eta, eta_z, zmax, lmax, tt, a.ema)
        rows.append(row); recs[f"{tag}_idx"] = ids; recs[f"{tag}_first_bad"] = first_bad; recs[f"{tag}_zpeak"] = peak
        print(f"CENSUS {row['name']}: exploded {row['n_exploded']}/{row['n']} ({100*row['exploded_frac']:.1f}%) "
              f"zmax median {row['zmax_median']:.3g} p99 {row['zmax_p99']:.3g} eta {eta:.3f} ({time.time()-t0:.0f}s)", flush=True)
    (out / "census.json").write_text(json.dumps(dict(ckpt=a.ckpt, ema=bool(a.ema), rows=rows, cell_kind=cfg.cell_kind,
                                                    z_norm=cfg.z_norm, wall_s=round(time.time() - t0, 1)), indent=1))
    np.savez_compressed(out / "records.npz", **recs)
    print("CENSUS-DONE", flush=True)


if __name__ == "__main__":
    main()
