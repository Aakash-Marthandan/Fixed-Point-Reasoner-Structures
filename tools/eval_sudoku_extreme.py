# Ledger: SPRINT S2 — the BATCHED Sudoku-Extreme evaluator (2026-08-21; wave-2
# extensions 2026-08-22). The benchmark number: EXACT accuracy on the FULL 423k
# test set, one deterministic cold-start prediction per puzzle (HRM/TRM/EqR
# convention). Test-time knobs, each a REGISTERED axis of the series and
# reported with the number: inference depth t_total (EqR's D), verify-and-vote
# multi-init (EqR's breadth B — but Sudoku verification is FREE: a valid,
# givens-consistent completion IS the solution by uniqueness, so no majority
# vote), and FPOpt damping decay (FPRM). The per-step y/z updates mirror
# model.iterate_eq exactly (eta floor form, eta_z carry, softmax feedback).
# Per-puzzle records: cold exact, first-exact step, first-valid step, final
# violations / cells correct / givens kept, multi-init verified + true hits.
#
# WAVE 2 (2026-08-22): (i) layout-aware placement (cfg.sudoku_layout: origin |
# box4, the registered box-aligned control); (ii) --subsample N (seeded random
# subsample of the split, for the k=128 breadth number on the best arm);
# (iii) multi-init draws seeded per (puzzle, draw) so the verify-and-vote
# k-curve is NESTED (vote@k for every k <= k_init from ONE run, shard- and
# batch-invariant) — records carry mi_first_hit; (iv) --init solution = the
# batched RETENTION instrument (hand the map the solution, t_total stab
# steps; exact_acc then IS retention), replacing the per-puzzle probe for
# layouts the ARC-shared probe cannot place.
"""
  .venv/bin/python tools/eval_sudoku_extreme.py --ckpt runs/X/ckpt_latest.pkl \
      --npz data/sudoku_extreme/sudoku_extreme_seed0.npz --split test \
      --t-total 64 --k-init 0 --out runs/sxeval_X [--stratified 512] [--shard i/K]
      [--subsample 20000] [--init solution]
  .venv/bin/python tools/eval_sudoku_extreme.py --merge runs/sxeval_X
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import jax
import jax.numpy as jnp

from qhrrn2 import episodic as E
from qhrrn2 import grid as G
from qhrrn2 import model as M
from qhrrn2 import sudoku as SU
from qhrrn2 import sudoku_extreme as SX
from qhrrn2.config import Config

N = SX.N
K_CURVE = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
# 512/1024 added 2026-08-24 (Phase B registration): the wave-3a k=1024 scan's
# summary silently capped at 256; vote_curve already breaks at k > k_init, so
# this only adds entries when the run actually drew that many inits.


# ── device-side checks on the 9x9 window ─────────────────────────────────

def _digit_presence(g9):          # g9: (B, 9, 9) int32 -> (B, 9 rows, 9 digits) bool
    oh = jax.nn.one_hot(g9 - 1, N, dtype=jnp.float32)   # digit d -> index d-1; 0 -> all zero
    return oh


def violations_dev(g9):
    """Missing digits summed over rows, cols, boxes (0 == valid solved grid)."""
    oh = _digit_presence(g9)                               # (B, r, c, d)
    rows = N - jnp.sum(jnp.any(oh > 0, axis=2), axis=-1)   # (B, r)
    cols = N - jnp.sum(jnp.any(oh > 0, axis=1), axis=-1)   # (B, c)
    B = g9.shape[0]
    boxes = oh.reshape(B, 3, 3, 3, 3, N).transpose(0, 1, 3, 2, 4, 5).reshape(B, 9, 9, N)
    box = N - jnp.sum(jnp.any(boxes > 0, axis=2), axis=-1)  # (B, 9)
    return jnp.sum(rows, -1) + jnp.sum(cols, -1) + jnp.sum(box, -1)


def layout_gather(canvas_b, layout: str):
    """(B, CV, CV) canvas argmax -> (B, 9, 9) digits in the layout's cells.
    native9 (2026-09-01): the canvas IS the 9x9 grid."""
    if layout == "native9":
        return canvas_b
    if layout == "origin":
        return canvas_b[:, :N, :N]
    idx = jnp.asarray(SU.box4_index())
    return canvas_b[:, idx][:, :, idx]


def place_batch(grids9, layout: str):
    return jnp.asarray(np.stack([SU.place_layout(g.astype(np.int8), layout) for g in grids9]), jnp.int32)


def make_step(cfg: Config, tau: float, t_norm: float, first: bool):
    """Batched one-step map: (params, x_can[B], y[B], task_vec, z_c[B]) ->
    (logits9[B,9,9,VOCAB], z_new[B,...]). `first` = no carried z yet."""
    def fwd1(params, x_can, y, task_vec, z_c):
        out = M.forward_fields(params, cfg, M.build_fields_soft(x_can, y),
                               t_norm=t_norm, tau=tau, rng=None,
                               task_vec=task_vec, z_in=None if first else z_c)
        return out.logits, out.z_fine
    in_axes = (None, 0, 0, None, None if first else 0)
    return jax.jit(jax.vmap(fwd1, in_axes=in_axes))


@functools.lru_cache(maxsize=None)
def _step(cfg, tau, t_norm, first):
    return make_step(cfg, tau, t_norm, first)


def coupled_ab(params, cfg):
    """FPRM coupled residual (cfg.eq_coupled): y <- a1*y + a2*p; None for the damped form."""
    if not getattr(cfg, "eq_coupled", False):
        return None
    return (float(jax.nn.sigmoid(params["eq"]["alpha1"])), float(jax.nn.sigmoid(params["eq"]["alpha2"])))


def run_batch(params, cfg, tvj, x_can, y0, *, t_total, tau, gamma, sol9, puz9,
              eta, eta_z, layout="origin", t_norm_fixed=None, ab=None, z0=None, hard=False):
    """Returns per-puzzle numpy: exact_t (T,B), valid_ok_t (T,B) [valid & givens-
    consistent], final pred9 (B,9,9), resid3 (B,) — the mean per-step |Delta y|
    over the FINAL 3 steps (EqR's Top-1-Converged selection signal, L=3;
    CHAMPION TRACK 2026-09-01). t_norm_fixed (wave 3a): apply one map at
    every step (1.0 = the FINAL map — the final-map retention instrument);
    ab = (a1, a2) for eq_coupled checkpoints (mirrors model.iterate_eq).
    z0 (sportC1 X0): an explicit initial carry (B, ...) — EqR's RI restart draws
    z ~ N(0, sigma) for the field-recipe cell; None = the cell's own start (the
    fixed buffers for 'trm'; no carry for 'rg'). For cfg.cell_kind == 'trm' the
    selection residual is on the LATENT (EqR D.3: |f(z_t) - z_t| over the last
    L = 3 outer steps) instead of on y.
    sportC2: cfg.inner_k latent passes per outer step (mirrors model.iterate_eq;
    1 = the pre-existing loop); hard=True feeds back the argmax one-hot readout
    at every step (R3's labeled inference mode)."""
    B = x_can.shape[0]
    y = y0
    z_c = None if z0 is None else z0
    trm = getattr(cfg, "cell_kind", "rg") == "trm"
    ex_rows, ok_rows = [], []
    res_tail = []                 # per-step (B,) mean |dy|, last 3 kept
    pred9 = None
    sol9 = jnp.asarray(sol9, jnp.int32); puz9 = jnp.asarray(puz9, jnp.int32)
    mask = puz9 != 0
    K = 1 if trm else max(1, int(getattr(cfg, "inner_k", 1)))
    for t in range(t_total):
        t_norm = (min(t, cfg.T - 1) / max(cfg.T - 1, 1)) if t_norm_fixed is None else float(t_norm_fixed)
        if trm:
            t_norm = 0.0   # the field cell ignores t_norm: one compiled step instead of T (sportC1 pre-mortem)
        for _k in range(K):
            first = z_c is None
            logits, zf = _step(cfg, float(tau), float(t_norm), first)(
                params, x_can, y, tvj, jnp.zeros(1) if first else z_c)
            z_prev = z_c
            z_c = zf if first else z_c + eta_z * (zf - z_c)
        p = jax.nn.softmax(logits, axis=-1).transpose(0, 3, 1, 2)
        if hard:
            p = jax.nn.one_hot(jnp.argmax(logits, axis=-1), M.VOCAB).transpose(0, 3, 1, 2)
        eta_t = eta if (gamma >= 1.0 or t < cfg.T) else eta * (gamma ** (t - cfg.T + 1))
        y_new = (ab[0] * y + ab[1] * p) if ab is not None else (y + eta_t * (p - y))
        if trm:
            zr = z_c - z_prev if z_prev is not None else z_c - z_c
            res_tail.append(np.asarray(jnp.mean(jnp.abs(zr), axis=tuple(range(1, zr.ndim)))))
        else:
            res_tail.append(np.asarray(jnp.mean(jnp.abs(y_new - y), axis=(1, 2, 3))))
        if len(res_tail) > 3:
            res_tail.pop(0)
        y = y_new
        pred9 = layout_gather(jnp.argmax(logits, axis=-1), layout).astype(jnp.int32)
        pred9 = jnp.where(pred9 == G.VOID, 0, pred9)
        exact = jnp.all((pred9 == sol9).reshape(B, -1), axis=1)
        viol = violations_dev(pred9)
        giv_ok = jnp.all(((pred9 == puz9) | ~mask).reshape(B, -1), axis=1)
        ok = (viol == 0) & giv_ok
        ex_rows.append(np.asarray(exact)); ok_rows.append(np.asarray(ok))
    return (np.stack(ex_rows), np.stack(ok_rows), np.asarray(pred9),
            np.mean(np.stack(res_tail), axis=0))


def first_true(rows):             # (T,B) bool -> (B,) first index or -1
    T, B = rows.shape
    any_ = rows.any(axis=0)
    idx = np.where(any_, rows.argmax(axis=0), -1)
    return idx


def mi_canvas(mi_seed: int, idx: int, j: int, layout: str) -> np.ndarray:
    """Draw j of puzzle idx: a uniform random 9x9 canvas in the layout. Seeded
    per (puzzle, draw) => nested k-curves; identical across shards/batches."""
    rng = np.random.default_rng([int(mi_seed), int(idx), int(j)])
    return SU.place_layout(rng.integers(0, 10, size=(N, N)).astype(np.int8), layout)


def mi_z0(mi_seed: int, idx: int, j: int, shape, sigma: float) -> np.ndarray:
    """sportC1 X0: EqR's RI restart for draw j of puzzle idx — z0 ~ N(0, sigma I)
    of the given carry shape, seeded per (puzzle, draw) like mi_canvas (nested
    k-curves, shard/batch invariant)."""
    rng = np.random.default_rng([int(mi_seed), int(idx), int(j), 7])
    return (sigma * rng.standard_normal(shape)).astype(np.float32)


def vote_curve(cold, first_hit, k_init):
    """vote@k = cold-exact OR a verified hit among the first k draws."""
    out = {}
    for k in K_CURVE:
        if k > k_init:
            break
        out[str(k)] = float(np.mean(cold | ((first_hit >= 0) & (first_hit < k))))
    return out


def majority_vote_cols(draws9, sol9, ks):
    """RUNG 2 / D3 demo (2026-08-27): UNVERIFIED cellwise majority vote — the
    EqR-style aggregation, without Sudoku's free verifier. draws9: (B, k, 9, 9)
    int grids; sol9: (B, 9, 9). Returns {k: bool[B]} — majority grid over the
    first k draws equals the solution (ties -> lowest digit, deterministic).
    Pure function (named test tests/test_eval_uv_vote.py)."""
    B = draws9.shape[0]
    out = {}
    for k in ks:
        oh = (draws9[:, :k, :, :, None] == np.arange(10)).sum(1)   # (B,9,9,10)
        maj = oh.argmax(-1)
        out[k] = (maj == sol9).reshape(B, -1).all(1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt"); ap.add_argument("--npz")
    ap.add_argument("--split", default="test", choices=["test", "val", "train"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stratified", type=int, default=None,
                    help="rating-stratified subsample size (instrument set)")
    ap.add_argument("--strat-seed", type=int, default=20260821)
    ap.add_argument("--subsample", type=int, default=None,
                    help="wave-2: seeded uniform random subsample of the split (no replacement)")
    ap.add_argument("--subsample-seed", type=int, default=20260822)
    ap.add_argument("--shard", default=None, help="i/K contiguous split of the selection")
    ap.add_argument("--t-total", type=int, default=None, help="inference depth (default cfg.T)")
    ap.add_argument("--k-init", type=int, default=0, help="verify-and-vote multi-init draws")
    ap.add_argument("--mi-seed", type=int, default=4242)
    ap.add_argument("--init", default="void", choices=["void", "solution"],
                    help="start state: void (cold start, the benchmark) or solution "
                         "(the batched RETENTION instrument: exact_acc = retention)")
    ap.add_argument("--final-map-only", action="store_true",
                    help="wave 3a: apply the FINAL map (t_norm=1) at every step — with "
                         "--init solution this is the final-map retention instrument "
                         "(the probe's e3b semantics, batched, any layout)")
    ap.add_argument("--fpopt-gamma", type=float, default=1.0,
                    help="<1: eta decays by gamma per step beyond cfg.T (FPOpt)")
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--eta-override", type=float, default=None,
                    help="DIAGNOSTIC ONLY (2026-08-23 wave-2 analysis): replace the ckpt's learned "
                         "equilibrium step eta at inference. Never a benchmark number — labeled in the summary.")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--ema", action="store_true",
                    help="sportC1: evaluate the checkpoint's EMA weights (state_ema; the field's "
                         "headline weights on R0/X0) — labeled in the summary")
    ap.add_argument("--z0-sigma", type=float, default=1.0,
                    help="sportC1 X0 (cell_kind trm): RI restart scale for the multi-init draws "
                         "(EqR A.3 default 1); the cold run uses the cell's fixed start")
    ap.add_argument("--hard-feedback", action="store_true",
                    help="sportC2 R3: feed back the argmax one-hot readout at every step (labeled "
                         "inference mode; recorded in the fingerprint and summary)")
    ap.add_argument("--vote-unverified", action="store_true",
                    help="ALSO record EqR-style cellwise MAJORITY vote@k (no verifier) as "
                         "records columns uv_vote_k* — the D3 demo instrument (2026-08-27). "
                         "Strat-scale only (guard: selection <= 5000); merge-safe (columns "
                         "concatenate; summarize recomputes means).")
    ap.add_argument("--bank-every", type=float, default=0,
                    help="PHASE B ops hardening (2026-08-26, registered d06fe39): every N seconds "
                         "write partial_{tag}.npz (atomic, batch-boundary) with a provenance "
                         "fingerprint; on start, a matching partial resumes the shard exactly "
                         "(per-(puzzle,draw) seeding makes resumed == uninterrupted, bit-identical "
                         "— named test). 0 = off, existing behavior byte-for-byte.")
    ap.add_argument("--out", help="output dir")
    ap.add_argument("--merge", default=None, help="merge shard files in DIR and exit")
    a = ap.parse_args()

    if a.merge:
        return merge(Path(a.merge))
    assert a.ckpt and a.npz and a.out
    assert not (a.stratified and a.subsample), "--stratified and --subsample are exclusive"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    saved = E.load_ckpt(a.ckpt)
    defaults = Config()
    cfg = Config(**{k: type(getattr(defaults, k))(v) for k, v in saved["config"].items()})
    assert cfg.equilibrium, "evaluator is eq-only"
    layout = getattr(cfg, "sudoku_layout", "origin") or "origin"
    st = saved["state"]
    if a.ema:
        assert saved.get("state_ema") is not None, "--ema: this checkpoint carries no EMA weights"
        st = saved["state_ema"]
    params = st["model"]
    tvj = jnp.asarray(st["table"][0])
    eta, eta_z = (float(v) for v in M.eq_etas(params, cfg))
    eta_learned = eta
    trm = getattr(cfg, "cell_kind", "rg") == "trm"
    z_shape = (2, cfg.canvas * cfg.canvas + cfg.trm_puzzle_emb_len, cfg.trm_hidden) if trm else None
    if a.eta_override is not None:
        eta = float(a.eta_override)
        print(f"DIAGNOSTIC eta override: learned {eta_learned:.3f} -> {eta:.3f}", flush=True)
    ab = coupled_ab(params, cfg)
    tnf = 1.0 if a.final_map_only else None
    t_total = a.t_total or cfg.T

    d = SX.load_prepared(a.npz)
    Q, A, R = d[f"{a.split}_q"], d[f"{a.split}_a"], d[f"{a.split}_rating"]
    sel = np.arange(len(Q))
    if a.stratified:
        sel = SX.stratified_subsample(R, a.stratified, a.strat_seed)
    if a.subsample:
        rng_s = np.random.default_rng(a.subsample_seed)
        sel = np.sort(rng_s.choice(len(Q), size=min(a.subsample, len(Q)), replace=False))
    if a.limit:
        sel = sel[:a.limit]
    tag = "all"
    if a.shard:
        i, K = map(int, a.shard.split("/")); tag = f"s{i}"
        sel = np.array_split(sel, K)[i]
    # rating bins over the FULL split (shard-invariant)
    qs = np.quantile(R, np.linspace(0, 1, 9)); qs[-1] += 1

    rec = dict(idx=[], rating=[], cold_exact=[], first_exact=[], first_valid=[],
               violations=[], cells=[], givens_kept=[], mi_verified=[], mi_true=[],
               mi_first_hit=[])
    if a.k_init:
        # CHAMPION TRACK standing stats (2026-09-01, program-review §1 riders):
        # per-draw exact bits + per-draw convergence residuals (EqR's L=3
        # signal) -> the protocol-table statistics (true B=1; Top-1-residual@k)
        # come from every scan for free. (B, k) columns; merge concatenates.
        rec["mi_exact_k"] = []
        rec["mi_resid_k"] = []
    uv_ks = []
    if a.vote_unverified:
        if len(sel) > 5000:
            raise SystemExit("--vote-unverified is a strat-scale instrument (selection <= 5000)")
        uv_ks = [k for k in K_CURVE if k <= a.k_init]
        for k in uv_ks:
            rec[f"uv_vote_k{k}"] = []
    t0 = time.time()
    # --bank-every: resume from a provenance-matched partial (batch-boundary counts only)
    fingerprint = json.dumps(dict(ckpt=a.ckpt, npz=a.npz, split=a.split, shard=tag, n=len(sel),
                                  t=t_total, k=a.k_init, init=a.init, layout=layout, batch=a.batch,
                                  mi_seed=a.mi_seed, fmo=bool(a.final_map_only),
                                  uv=bool(a.vote_unverified), ver=2,
                                  **({"ema": True} if a.ema else {}), **({"z0": a.z0_sigma} if trm else {}),
                                  **({"hard": True} if a.hard_feedback else {})),
                     sort_keys=True)
    partial_p = out / f"partial_{tag}.npz"
    start = 0
    if a.bank_every and partial_p.exists():
        try:
            pz = np.load(partial_p, allow_pickle=True)
            if str(pz["_fingerprint"]) == fingerprint and int(pz["_done"]) % a.batch == 0:
                for k in rec:
                    rec[k] = pz[k].tolist()
                start = int(pz["_done"])
                print(f"RESUMED shard {tag} from partial at {start}/{len(sel)}", flush=True)
            else:
                print(f"partial {tag} fingerprint/boundary mismatch — starting fresh", flush=True)
        except Exception as e:
            print(f"partial {tag} unreadable ({e}) — starting fresh", flush=True)
    last_bank = time.time()
    for s in range(start, len(sel), a.batch):
        ids = sel[s:s + a.batch]
        puz9 = Q[ids].astype(np.int32); sol9 = A[ids].astype(np.int32)
        x_can = place_batch(puz9, layout)
        B = len(ids)
        cv = SU.layout_canvas(layout)
        if a.init == "solution":
            y0 = jax.nn.one_hot(place_batch(sol9, layout), M.VOCAB).transpose(0, 3, 1, 2)
        else:
            void = jax.nn.one_hot(jnp.full((cv, cv), G.VOID, jnp.int32), M.VOCAB).transpose(2, 0, 1)
            y0 = jnp.broadcast_to(void, (B,) + void.shape)
        ex, ok, pred9, _ = run_batch(params, cfg, tvj, x_can, y0, t_total=t_total, tau=a.tau,
                                     gamma=a.fpopt_gamma, sol9=sol9, puz9=puz9, eta=eta, eta_z=eta_z,
                                     layout=layout, t_norm_fixed=tnf, ab=ab, hard=a.hard_feedback)
        cold = ex[-1]
        fe, fv = first_true(ex), first_true(ok)
        viol = np.asarray(violations_dev(jnp.asarray(pred9)))
        cells = (pred9 == sol9).reshape(B, -1).sum(1)
        mask = puz9 != 0
        gk = ((pred9 == puz9) & mask).reshape(B, -1).sum(1)
        mi_v = np.zeros(B, int); mi_t = np.zeros(B, int); mi_first = np.full(B, -1, int)
        mi_ex_k = np.zeros((B, a.k_init), np.uint8) if a.k_init else None
        mi_re_k = np.zeros((B, a.k_init), np.float16) if a.k_init else None
        uv_draws = np.zeros((B, len(uv_ks) and a.k_init or 0, 9, 9), np.int8) if uv_ks else None
        for j in range(a.k_init):
            y0r = np.stack([mi_canvas(a.mi_seed, int(ids[b]), j, layout) for b in range(B)])
            y0r = jax.nn.one_hot(jnp.asarray(y0r, jnp.int32), M.VOCAB).transpose(0, 3, 1, 2)
            z0r = (jnp.asarray(np.stack([mi_z0(a.mi_seed, int(ids[b]), j, z_shape, a.z0_sigma) for b in range(B)]))
                   if trm else None)
            exr, okr, predr, resr = run_batch(params, cfg, tvj, x_can, y0r, t_total=t_total, tau=a.tau,
                                              gamma=a.fpopt_gamma, sol9=sol9, puz9=puz9, eta=eta, eta_z=eta_z,
                                              layout=layout, t_norm_fixed=tnf, ab=ab, z0=z0r, hard=a.hard_feedback)
            hit = okr[-1]
            mi_first = np.where((mi_first < 0) & hit, j, mi_first)
            mi_v += hit.astype(int); mi_t += exr[-1].astype(int)
            mi_ex_k[:, j] = exr[-1].astype(np.uint8)
            mi_re_k[:, j] = resr.astype(np.float16)
            if uv_draws is not None:
                uv_draws[:, j] = np.asarray(predr, np.int8)
        if a.k_init:
            rec["mi_exact_k"].extend(mi_ex_k.tolist())
            rec["mi_resid_k"].extend(mi_re_k.tolist())
        if uv_ks:
            uv = majority_vote_cols(uv_draws, sol9, uv_ks)
            for k in uv_ks:
                rec[f"uv_vote_k{k}"].extend(uv[k].tolist())
        rec["idx"].extend(ids.tolist()); rec["rating"].extend(R[ids].tolist())
        rec["cold_exact"].extend(cold.tolist()); rec["first_exact"].extend(fe.tolist())
        rec["first_valid"].extend(fv.tolist()); rec["violations"].extend(viol.tolist())
        rec["cells"].extend(cells.tolist()); rec["givens_kept"].extend(gk.tolist())
        rec["mi_verified"].extend(mi_v.tolist()); rec["mi_true"].extend(mi_t.tolist())
        rec["mi_first_hit"].extend(mi_first.tolist())
        done = s + B
        if done % (a.batch * 8) == 0 or done == len(sel):
            acc = float(np.mean(rec["cold_exact"]))
            print(f"  {done}/{len(sel)} cold-exact {acc:.4f}  {time.time()-t0:.0f}s", flush=True)
        if a.bank_every and done < len(sel) and time.time() - last_bank >= a.bank_every:
            tmp = out / f"partial_{tag}.tmp.npz"   # savez appends .npz if absent — keep the suffix explicit
            np.savez_compressed(tmp, _fingerprint=fingerprint, _done=done,
                                **{k: np.asarray(v) for k, v in rec.items()})
            os.replace(tmp, partial_p)
            last_bank = time.time()
            print(f"  banked partial {tag} @ {done}/{len(sel)}", flush=True)
        if os.environ.get("SX_TEST_ABORT_AFTER") and done >= int(os.environ["SX_TEST_ABORT_AFTER"]):
            print(f"  TEST ABORT at {done} (SX_TEST_ABORT_AFTER)", flush=True)
            sys.exit(3)   # named-test hook only: simulates a preemption mid-shard

    arr = {k: np.asarray(v) for k, v in rec.items()}
    np.savez_compressed(out / f"records_{tag}.npz", **arr)
    partial_p.unlink(missing_ok=True)
    summ = summarize(arr, Q, sel, qs, dict(
        ckpt=a.ckpt, npz=a.npz, split=a.split, t_total=t_total, k_init=a.k_init, init=a.init,
        layout=layout, fpopt_gamma=a.fpopt_gamma, tau=a.tau, shard=tag,
        stratified=a.stratified, subsample=a.subsample, subsample_seed=a.subsample_seed,
        mi_seed=a.mi_seed, eta=eta, eta_z=eta_z, T=cfg.T, d=cfg.d,
        eta_learned=eta_learned, eta_override=a.eta_override,
        final_map_only=bool(a.final_map_only), eq_coupled_ab=ab,
        cell_kind=getattr(cfg, "cell_kind", "rg"), ema=bool(a.ema), z0_sigma=(a.z0_sigma if trm else None),
        inner_k=int(getattr(cfg, "inner_k", 1)), hard_feedback=bool(a.hard_feedback),
        wall_s=round(time.time() - t0, 1)))
    (out / f"summary_{tag}.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({k: summ[k] for k in ("n", "t_total", "k_init", "init", "exact_acc", "exact_acc_vote", "wall_s")}))


def summarize(arr, Q, sel, qs, base):
    n = int(len(arr["idx"]))
    k_init = int(base.get("k_init", 0))
    vote = arr["cold_exact"] | (arr["mi_verified"] > 0) if k_init else arr["cold_exact"]
    fh = arr["mi_first_hit"] if "mi_first_hit" in arr else np.full(n, -1)
    given_tot = np.maximum((Q[arr["idx"]] != 0).reshape(n, -1).sum(1), 1) if n else np.ones(0)
    summ = dict(base)
    summ.update(
        n=n,
        exact_acc=float(arr["cold_exact"].mean()) if n else None,
        exact_acc_vote=float(vote.mean()) if n else None,
        vote_at_k=vote_curve(arr["cold_exact"], fh, k_init) if (n and k_init) else {},
        mean_first_exact=float(np.mean(arr["first_exact"][arr["first_exact"] >= 0])) if (arr["first_exact"] >= 0).any() else None,
        valid_wrong_frac=float(np.mean((arr["violations"] == 0) & ~arr["cold_exact"])) if n else None,
        mean_violations=float(arr["violations"].mean()) if n else None,
        givens_kept_frac=float((arr["givens_kept"] / given_tot).mean()) if n else None,
        mi_hits_mean=float(arr["mi_verified"].mean()) if (n and k_init) else None,
        mi_true_minus_verified=int(np.sum(arr["mi_true"] != arr["mi_verified"])),
        by_rating_bin=[float(arr["cold_exact"][(arr["rating"] >= qs[b]) & (arr["rating"] < qs[b + 1])].mean())
                       if ((arr["rating"] >= qs[b]) & (arr["rating"] < qs[b + 1])).any() else None for b in range(8)],
        by_rating_bin_vote=[float(vote[(arr["rating"] >= qs[b]) & (arr["rating"] < qs[b + 1])].mean())
                            if ((arr["rating"] >= qs[b]) & (arr["rating"] < qs[b + 1])).any() else None for b in range(8)],
        rating_bins=[float(x) for x in qs])
    uv_cols = sorted((k for k in arr if k.startswith("uv_vote_k")), key=lambda s: int(s[9:]))
    if uv_cols and n:
        summ["majority_vote_at_k"] = {c[9:]: float(arr[c].mean()) for c in uv_cols}
    # CHAMPION TRACK standing stats (2026-09-01): true B=1 (single random-init
    # draw, EqR's Table-4 statistic — NOT vote@1, which unions with cold) and
    # Top-1-by-residual@k (EqR's L=3 Top-1-Converged selection, verbatim).
    if n and "mi_exact_k" in arr and arr["mi_exact_k"].ndim == 2 and arr["mi_exact_k"].shape[1]:
        ex_k = arr["mi_exact_k"].astype(bool)
        re_k = arr["mi_resid_k"].astype(np.float32)
        summ["b1_exact"] = float(ex_k[:, 0].mean())
        t1r = {}
        for k in K_CURVE:
            if k > ex_k.shape[1]:
                break
            pick = np.argmin(re_k[:, :k], axis=1)
            t1r[str(k)] = float(ex_k[np.arange(n), pick].mean())
        summ["t1r_at_k"] = t1r
    return summ


def merge(d: Path):
    recs = sorted(d.glob("records_s*.npz"))
    if not recs:
        print("nothing to merge"); return
    arrs = [dict(np.load(p)) for p in recs]
    keys = [k for k in arrs[0] if all(k in x for x in arrs)]
    arr = {k: np.concatenate([x[k] for x in arrs]) for k in keys}
    s0 = json.loads(sorted(d.glob("summary_s*.json"))[0].read_text())
    qs = np.asarray(s0["rating_bins"])
    # per-puzzle givens totals are not in the records: recompute from the npz if reachable,
    # else carry the shard-averaged fraction (labeled)
    npz = None
    for cand in (Path(s0.get("npz", "")),):
        if cand and cand.exists():
            npz = cand
    n = int(len(arr["idx"]))
    summ = dict(s0, shard="merged",
                wall_s=sum(json.loads(p.read_text())["wall_s"] for p in d.glob("summary_s*.json")))
    if npz is not None:
        Q = SX.load_prepared(npz)[f"{s0['split']}_q"]
        summ = summarize(arr, Q, arr["idx"], qs, summ)
    else:
        fake_Q = None
        k_init = int(s0.get("k_init", 0))
        vote = arr["cold_exact"] | (arr["mi_verified"] > 0) if k_init else arr["cold_exact"]
        fh = arr["mi_first_hit"] if "mi_first_hit" in arr else np.full(n, -1)
        summ.update(n=n, exact_acc=float(arr["cold_exact"].mean()), exact_acc_vote=float(vote.mean()),
                    vote_at_k=vote_curve(arr["cold_exact"], fh, k_init) if k_init else {},
                    mean_violations=float(arr["violations"].mean()),
                    valid_wrong_frac=float(np.mean((arr["violations"] == 0) & ~arr["cold_exact"])),
                    mi_hits_mean=float(arr["mi_verified"].mean()) if k_init else None,
                    mi_true_minus_verified=int(np.sum(arr["mi_true"] != arr["mi_verified"])),
                    givens_kept_frac=float(np.mean([json.loads(p.read_text())["givens_kept_frac"] for p in d.glob("summary_s*.json")])),
                    by_rating_bin=[float(arr["cold_exact"][(arr["rating"] >= qs[b]) & (arr["rating"] < qs[b + 1])].mean())
                                   if ((arr["rating"] >= qs[b]) & (arr["rating"] < qs[b + 1])).any() else None for b in range(8)],
                    by_rating_bin_vote=[float(vote[(arr["rating"] >= qs[b]) & (arr["rating"] < qs[b + 1])].mean())
                                        if ((arr["rating"] >= qs[b]) & (arr["rating"] < qs[b + 1])).any() else None for b in range(8)])
    np.savez_compressed(d / "records_all.npz", **arr)
    (d / "summary_all.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({k: summ[k] for k in ("n", "t_total", "k_init", "exact_acc", "exact_acc_vote")}))


if __name__ == "__main__":
    main()
