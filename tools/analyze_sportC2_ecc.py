#!/usr/bin/env python3
"""sportC2 — THE ERROR-CORRECTING-CODE / DECODER LENS (analysis-time, descriptive, NO RULES; 2026-09-05;
PI-directed emphasis: "what we have is really a measurable decoder"). Verdict authority = analyze_sportC2.py.

Frame. A Sudoku puzzle is a codeword of the constraint code (81 cells, 27 all-different checks) transmitted
through an ERASURE channel: 81 - givens cells erased. Every map is a decoder; the test set is the channel at
17-35 givens (erasure fraction .57-.79). The lens reads each map as a decoder, on the identical puzzle sets:
  E0  the channel + two reference decoders on the same 20k: PEELING (naked + hidden singles to a fixpoint; the
      pure propagation decoder, computed here) and tdoku's rating-0 class (its propagation solves without a
      guess); ML = 1.0 everywhere (every puzzle has a unique solution).
  E1  erasure-threshold curves: P(cold | givens), P(reach@128 | givens), b1 | givens; a logistic fit -> g50
      (the 50 % threshold in givens) and the waterfall width; solve rate on propagation-class (rating 0) vs
      search-class (rating > 0) puzzles and by search-demand band; the memorization radius (vsel vs final).
  E2  decoder dynamics on cold trajectories (CPU, strat-N): cells correct / entropy / confident-wrong /
      commitment per step; the flip spectrum (revisions to correct / to wrong); MONOTONICITY (a peeling decoder
      never un-decides a correct cell) vs churn; the SYNDROME trajectory (violations of the argmax grid per
      step) and its late oscillation (trapping-set signature); first_exact distribution.
  E3  decimation quality at stalls: the map's committed cells (p > tau) as hard decisions fed to the PEELING
      decoder -> solved / stuck / contradiction; the fraction of wrong hard decisions (the calibration lens in
      decoder units).
  E4  list decoding: verified@k curves, the per-puzzle per-draw success rate r_i distribution, list size for
      the reachable set, rescue-by-draws of cold failures; E5 the soft-syndrome selector (residual AUC,
      spurious rate); E6 calibration at stalls (from the calib rows); E7 a per-map DECODER SCORECARD.

  PYTHONPATH=src JAX_PLATFORMS=cpu .venv/bin/python tools/analyze_sportC2_ecc.py [--n 256] [--maps W0,R1,...] [--no-dyn]
      -> runs/analysis/sportC2_ecc_20260905.txt (+ .json)
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; RUNS = Path(os.environ.get("QHRRN_RUNS", ROOT / "runs"))
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tools"))
OUT = RUNS / "analysis" / "sportC2_ecc_20260905.txt"; OUTJ = RUNS / "analysis" / "sportC2_ecc_20260905.json"
NPZ = ROOT / "data/sudoku_extreme/sudoku_extreme_seed0.npz"
L = []; J = {}
def say(s=""): L.append(str(s)); print(s, flush=True)
def pp(x): return "  -  " if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{100*x:5.1f}"
def f(x, w=6, p=3): return " " * (w - 1) + "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:{w}.{p}f}"
def jload(p): p = Path(p); return json.loads(p.read_text()) if p.exists() else None
def recs(p):
    p = Path(p); q = p / "records_all.npz"
    if not q.exists(): return None
    z = dict(np.load(q, allow_pickle=True)); o = np.argsort(z["idx"], kind="stable")
    return {k: (v[o] if hasattr(v, "shape") and v.shape and v.shape[0] == len(o) else v) for k, v in z.items()}
# scan sources (20k, k128 per-draw records); fulls (422,786); vsel ckpts for the dynamics lens
SCANS = {"W0": "sxscan_psportC2W0", "R1": "sxscan_psportC2R1", "R2": "sxscan_psportC2R2", "R3": "sxscan_psportC2R3", "R4": "sxscan_psportC2R4", "X1": "sxscan_psportC2X1", "X2": "sxscan_psportC2X2",
         "X0": "sxscan_psportC1X0", "R0": "sxscan_psportC1R0", "B0c": "sxscan_psportC1B0_vselA20k", "B1c": "sxscan_psportC1B1_vselA20k", "P3s1": "sxscan_psportC0P3s1", "D4": "sxscan_psportBr2bD4", "C3X": "sxscan_psportBr2bC3X"}
FULLS = {"W0": ("sxeval_psportC2W0/full_vsel_t64", "sxeval_psportC2W0/full_final_t64"), "R1": ("sxeval_psportC2R1/full_vsel_t64", "sxeval_psportC2R1/full_final_t64"), "R2": ("sxeval_psportC2R2/full_vsel_t64", "sxeval_psportC2R2/full_final_t64"),
         "R3": ("sxeval_psportC2R3/full_vsel_t64", "sxeval_psportC2R3/full_final_t64"), "R4": ("sxeval_psportC2R4/full_vsel_t64", "sxeval_psportC2R4/full_final_t64"), "X1": ("sxeval_psportC2X1/full_vsel_t64", None), "X2": ("sxeval_psportC2X2/full_vsel_t64", None),
         "X0": ("sxeval_psportC1X0/full_vsel_t64", None), "R0": ("sxeval_psportC1R0/full_vsel_t64", None), "B0c": ("sxeval_psportC1B0/full_vsel_t64", "sxeval_psportC1B0/full_final_t64"), "D4": ("sxeval_psportBr2bD4/full_t64", None), "P3s1": ("sxeval_psportC0P3s1/full_t64", None)}
DESC = {"W0": "B0 + wd 1.0 (base)", "R1": "W0 + carry (SOT-rg)", "R2": "W0 + inner cycles K3", "R3": "W0 + hard rows p.5", "R4": "R0 + 50k (field regime)", "X1": "X0 - digit aug", "X2": "X0 + group9 mixer",
        "X0": "FIELD baseline (sportC1)", "R0": "our cell, field regime (sportC1)", "B0c": "B0 A:20k correct grid (sportC1)", "B1c": "B1 A:20k correct grid", "P3s1": "d96 pilot record", "D4": "canvas d96 record", "C3X": "canvas two-phase (wide funnel)"}
GBINS = [(17, 21), (21, 23), (23, 25), (25, 27), (27, 29), (29, 36)]; RBANDS = [(0, 1), (1, 10), (10, 30), (30, 60), (60, 10**6)]

# ---------- the PEELING decoder (naked + hidden singles), vectorized over puzzles ----------
def peel(grid0, max_iter=100):
    """grid0 (N,9,9) int 0=empty -> (grid, status) status: 0 stuck, 1 solved, 2 contradiction."""
    g = grid0.astype(np.int16).copy(); N = len(g); status = np.zeros(N, np.int8); active = np.ones(N, bool)
    bi = (np.arange(9)[:, None] // 3) * 3 + (np.arange(9)[None, :] // 3)   # (9,9) box index
    for _ in range(max_iter):
        oh = (g[..., None] == np.arange(1, 10))                              # (N,9,9,9)
        rowc = oh.sum(2); colc = oh.sum(1)                                     # (N,9,9): row i digit d count / col j
        boxc = np.zeros((N, 9, 9), int)
        for b in range(9): boxc[:, b] = oh[:, bi == b].sum(1)
        dup = (np.maximum(rowc - 1, 0).sum((1, 2)) + np.maximum(colc - 1, 0).sum((1, 2)) + np.maximum(boxc - 1, 0).sum((1, 2))) > 0
        rowhas = rowc > 0; colhas = colc > 0; boxhas = boxc > 0
        cand = ~rowhas[:, :, None, :] & ~colhas[:, None, :, :] & ~boxhas[:, bi, :] & (g == 0)[..., None]   # (N,9,9,9)
        empty = g == 0; ncand = cand.sum(-1)
        contra = dup | ((empty & (ncand == 0)).any((1, 2)))
        newly = contra & active; status[newly] = 2; active &= ~contra
        solved = (~empty.any((1, 2))) & active; status[solved] = 1; active &= ~solved
        if not active.any(): break
        changed = np.zeros(N, bool); fill = np.zeros_like(g)
        # naked singles
        ns = empty & (ncand == 1) & active[:, None, None]
        fill[ns] = cand.argmax(-1)[ns] + 1; changed |= ns.any((1, 2))
        # hidden singles per unit type (rows / cols / boxes), digits not yet placed in the unit
        for unit in ("row", "col", "box"):
            if unit == "row": cnt = cand.sum(2); pos = cand.argmax(2)                 # (N,9i,9d) count over j; pos j
            elif unit == "col": cnt = cand.sum(1); pos = cand.argmax(1)               # (N,9j,9d)
            else:
                cb = np.stack([cand[:, bi == b] for b in range(9)], 1)                # (N,9b,9cell,9d)
                cnt = cb.sum(2); pos = cb.argmax(2)
            hs = (cnt == 1) & active[:, None, None]
            for n_, u_, d_ in zip(*np.where(hs)):
                if unit == "row": i, j = u_, pos[n_, u_, d_]
                elif unit == "col": i, j = pos[n_, u_, d_], u_
                else: cells = np.argwhere(bi == u_); i, j = cells[pos[n_, u_, d_]]
                if fill[n_, i, j] and fill[n_, i, j] != d_ + 1: status[n_] = 2; active[n_] = False; continue
                fill[n_, i, j] = d_ + 1; changed[n_] = True
        g = np.where((fill > 0) & (g == 0) & active[:, None, None], fill, g)
        stuck = active & ~changed; status[stuck] = 0; active &= ~stuck
        if not active.any(): break
    return g, status
def violations(pred):  # (N,9,9) -> duplicate count over rows/cols/boxes (numpy, vectorized)
    oh = (pred[..., None] == np.arange(1, 10)); N = len(pred); bi = (np.arange(9)[:, None] // 3) * 3 + (np.arange(9)[None, :] // 3)
    rowc = oh.sum(2); colc = oh.sum(1); boxc = np.stack([oh[:, bi == b].sum(1) for b in range(9)], 1)
    return np.maximum(rowc - 1, 0).sum((1, 2)) + np.maximum(colc - 1, 0).sum((1, 2)) + np.maximum(boxc - 1, 0).sum((1, 2))
def logistic_g50(g, y):
    """fit P(y=1|g) = sigmoid(a (g - g50)) by binomial NLL; returns (g50, a, width_10_90)."""
    from scipy.optimize import minimize
    g = g.astype(float); y = y.astype(float)
    if y.mean() in (0.0, 1.0): return (None, None, None)
    def nll(th):
        a, g50 = th; zz = np.clip(a * (g - g50), -30, 30); p = 1 / (1 + np.exp(-zz)); return -np.sum(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
    r = minimize(nll, x0=[0.5, 25.0], method="Nelder-Mead"); a, g50 = r.x
    return (float(g50), float(a), float(2 * np.log(9) / abs(a)) if a else None)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=256); ap.add_argument("--maps", default="W0,R1,R2,R3,R4,X1,X2,R0,B0,X0"); ap.add_argument("--no-dyn", action="store_true"); ap.add_argument("--t", type=int, default=64)
    a = ap.parse_args(); t0 = time.time()
    from qhrrn2 import sudoku_extreme as SX
    d = SX.load_prepared(NPZ); Q, A, R, SRC, SN = d["test_q"], d["test_a"], d["test_rating"], d["test_source"], d["source_names"]
    G = (Q != 0).reshape(len(Q), -1).sum(1)
    say("=" * 124); say("sportC2 — THE ERROR-CORRECTING-CODE / DECODER LENS (2026-09-05; analysis-time, descriptive, no rules; verdict authority = analyze_sportC2.py)"); say("=" * 124)
    S = {k: recs(RUNS / p) for k, p in SCANS.items()}; S = {k: v for k, v in S.items() if v is not None}
    base = S["W0"]["idx"] if "W0" in S else next(iter(S.values()))["idx"]
    same = {k: np.array_equal(v["idx"], base) for k, v in S.items()}
    say(f"\nscan sets: {len(base)} puzzles; identical idx to W0's: {[k for k, v in same.items() if v]} | DIFFERENT (paired on intersections only): {[k for k, v in same.items() if not v]}")
    idx = base; g = G[idx]; rat = R[idx]; src = SRC[idx]

    # ---- E0 ----
    say("\n== E0. THE CHANNEL and two reference decoders on the identical 20k (erasures = 81 - givens) ==")
    tp = time.time(); pg, ps = peel(Q[idx]); peel_ok = (ps == 1) & (pg == A[idx]).all((1, 2)); say(f"  peeling decoder (naked + hidden singles): solves {100*peel_ok.mean():.2f} % of the 20k | stuck {100*(ps==0).mean():.2f} % | contradiction {100*(ps==2).mean():.2f} % ({time.time()-tp:.0f}s)")
    r0 = rat == 0; say(f"  tdoku propagation class (rating 0): {100*r0.mean():.2f} % of the 20k; peeling solves {100*peel_ok[r0].mean():.1f} % of them and {100*peel_ok[~r0].mean():.2f} % of rating > 0")
    say("  givens bin | n | erasure frac | rating-0 frac | peeling solves | median rating | ML")
    for lo, hi in GBINS:
        m = (g >= lo) & (g < hi)
        if m.sum(): say(f"  [{lo},{hi}) | {m.sum():5d} | {(81-g[m].mean())/81:.3f} | {pp(r0[m].mean())} | {pp(peel_ok[m].mean())} | {np.median(rat[m]):5.0f} | 100.0")
    J["E0"] = dict(peel_solve=float(peel_ok.mean()), rating0=float(r0.mean()))
    say("  READ: the code's ML decoder solves everything (unique solutions); a pure propagation (peeling) decoder solves only the rating-0 class; the gap between them is the SEARCH the")
    say("        channel demands at this erasure load. Every learned map sits between the two curves; how far above peeling it reaches on rating > 0 puzzles is its decimation yield.")

    # ---- E1 ----
    say("\n== E1. ERASURE-THRESHOLD CURVES (identical 20k): cold / reach@128 / b1 per givens bin; logistic g50 (threshold) and 10-90 width; propagation vs search class ==")
    say("  map | " + " | ".join(f"[{lo},{hi}) cold/reach" for lo, hi in GBINS) + " | g50 cold (width) | g50 reach | P(cold|rating 0) | P(cold|rating>0) | Spearman(cold,g) (reach,g)")
    E1 = {}
    for k in ["W0", "R1", "R2", "R3", "R4", "X1", "X2", "X0", "R0", "B0c", "B1c", "P3s1", "D4", "C3X"]:
        z = S.get(k)
        if z is None: continue
        zi = z["idx"]; gg = G[zi]; rr = R[zi]; cold = z["cold_exact"].astype(bool); reach = cold | (z["mi_first_hit"] >= 0) if "mi_first_hit" in z else cold
        b1 = z["mi_exact_k"][:, 0].astype(bool) if "mi_exact_k" in z else None
        cells = []
        for lo, hi in GBINS:
            m = (gg >= lo) & (gg < hi); cells.append(f"{pp(cold[m].mean()) if m.sum() else '  -  '}/{pp(reach[m].mean()) if m.sum() else '  -  '}")
        g50c, ac, wc = logistic_g50(gg, cold); g50r, ar, wr = logistic_g50(gg, reach)
        sc = float(np.corrcoef(cold.astype(float), gg)[0, 1]); sr = float(np.corrcoef(reach.astype(float), gg)[0, 1])
        E1[k] = dict(g50_cold=g50c, width_cold=wc, g50_reach=g50r, p_cold_r0=float(cold[rr == 0].mean()), p_cold_rpos=float(cold[rr > 0].mean()), p_reach_rpos=float(reach[rr > 0].mean()),
                     bands={f"{lo}-{hi}": float(cold[(rr >= lo) & (rr < hi)].mean()) for lo, hi in RBANDS if ((rr >= lo) & (rr < hi)).any()},
                     bands_reach={f"{lo}-{hi}": float(reach[(rr >= lo) & (rr < hi)].mean()) for lo, hi in RBANDS if ((rr >= lo) & (rr < hi)).any()}, n=int(len(zi)))
        say(f"  {k:4s} | " + " | ".join(cells) + f" | {f(g50c,5,1)} ({f(wc,4,1)}) | {f(g50r,5,1)} | {pp(E1[k]['p_cold_r0'])} | {pp(E1[k]['p_cold_rpos'])} | {sc:+.3f} {sr:+.3f}")
    say("  search-demand spectrum: cold solve rate by tdoku rating band (backtracks demanded): map | 0 | 1-9 | 10-29 | 30-59 | 60+  ||  reach@128 by band")
    for k, e in E1.items():
        say(f"  {k:4s} | " + " | ".join(pp(e["bands"].get(f"{lo}-{hi}")) for lo, hi in RBANDS) + "  ||  " + " | ".join(pp(e["bands_reach"].get(f"{lo}-{hi}")) for lo, hi in RBANDS))
    say("  READ: g50 = the givens count at which the decoder's cold success crosses 50 % (lower = stronger decoder); width = the 10-90 % waterfall in givens. The peeling reference")
    say("        solves ~all of rating 0 and ~none above; a decoder's P(cold | rating > 0) is its measured search/decimation yield; the field's map has no threshold inside 17-35 givens.")
    # memorization / drift radius: vsel vs final by givens on the full test
    say("\n  DECODING RADIUS vs training (full test, vsel grid vs final grid; a memorizing map loses the low-givens capability first): map | bin: vsel -> final (lost / gained)")
    for k in ["W0", "R1", "R2", "R3", "R4", "B0c"]:
        pv, pf = FULLS.get(k, (None, None))
        zv = recs(RUNS / pv) if pv else None; zf = recs(RUNS / pf) if pf else None
        if zv is None or zf is None or not np.array_equal(zv["idx"], zf["idx"]): say(f"  {k:4s}: {'vsel == final or missing' if zv is None or zf is None else 'idx differ'}"); continue
        gg = G[zv["idx"]]; cv = zv["cold_exact"].astype(bool); cf = zf["cold_exact"].astype(bool)
        if np.array_equal(cv, cf): say(f"  {k:4s}: identical grids"); continue
        say(f"  {k:4s} | " + " | ".join(f"[{lo},{hi}) {pp(cv[(gg>=lo)&(gg<hi)].mean())}->{pp(cf[(gg>=lo)&(gg<hi)].mean())} ({pp((cv&~cf)[(gg>=lo)&(gg<hi)].mean())}/{pp((~cv&cf)[(gg>=lo)&(gg<hi)].mean())})" for lo, hi in GBINS))
    J["E1"] = E1

    # ---- E4 ----
    say("\n== E4. LIST DECODING (20k scans, 128 random-init draws + the free verifier as the syndrome check) ==")
    say("  map | verified@1/2/4/8/16/32/64/128 | rho (reachable incl. cold) | per-draw rate r_i on reachable: median / mean | r_i bins: 0 / (0,.05] / (.05,.2] / (.2,.5] / (.5,1] | k for 50 % / 90 % of reachable | P(draw1 | cold fails) | P(any128 | cold fails)")
    E4 = {}
    for k in ["W0", "R1", "R2", "R3", "R4", "X1", "X2", "X0", "R0", "B0c", "B1c", "C3X", "D4"]:
        z = S.get(k)
        if z is None or "mi_first_hit" not in z: continue
        cold = z["cold_exact"].astype(bool); fh = z["mi_first_hit"]
        vk = {kk: float((cold | ((fh >= 0) & (fh < kk))).mean()) for kk in (1, 2, 4, 8, 16, 32, 64, 128)}
        ex = z["mi_exact_k"].astype(bool) if "mi_exact_k" in z else None
        ri = ex.mean(1) if ex is not None else None; reach = cold | (fh >= 0)
        if ri is not None:
            rr_ = ri[reach]; bins = [float((ri == 0).mean()), float(((ri > 0) & (ri <= .05)).mean()), float(((ri > .05) & (ri <= .2)).mean()), float(((ri > .2) & (ri <= .5)).mean()), float((ri > .5).mean())]
            # list size: draws needed to reach 50 % / 90 % of the reachable set (from first hits, cold counted as k=0)
            fh_eff = np.where(cold, -1, fh); hits = np.sort(fh_eff[reach] + 1); n_r = len(hits)
            k50 = int(hits[int(.5 * n_r)]) if n_r else None; k90 = int(hits[int(.9 * n_r)]) if n_r else None
            resc1 = float(ex[~cold, 0].mean()) if (~cold).any() else None; resc = float((fh[~cold] >= 0).mean()) if (~cold).any() else None
        else: rr_ = np.array([]); bins = [None] * 5; k50 = k90 = resc1 = resc = None
        E4[k] = dict(vk=vk, rho=float(reach.mean()), r_med=float(np.median(rr_)) if len(rr_) else None, r_mean=float(rr_.mean()) if len(rr_) else None, bins=bins, k50=k50, k90=k90, resc1=resc1, resc=resc)
        say(f"  {k:4s} | " + "/".join(pp(vk[kk]).strip() for kk in (1, 2, 4, 8, 16, 32, 64, 128)) + f" | {pp(E4[k]['rho'])} | {f(E4[k]['r_med'],5,3)} / {f(E4[k]['r_mean'],5,3)} | " + " / ".join(pp(b) for b in bins) + f" | {k50} / {k90} | {pp(resc1)} | {pp(resc)}")
    say("  READ: rho = the decodable fraction under list decoding with a syndrome check; r_i = each puzzle's per-trial success (its 'channel'); a decoder whose reachable puzzles")
    say("        have r_i ~ 1 needs no list (init-invariant = every trial the same); the field's decoder has both rho ~ 1 and a broad r_i spectrum (draws explore).")
    J["E4"] = E4

    # ---- E5 ----
    say("\n== E5. THE SOFT SYNDROME (convergence residual as a verification-free selector): AUC P(resid correct < resid wrong), spurious rate, t1r/verified ==")
    from scipy import stats
    E5 = {}
    for k in ["W0", "R1", "R2", "R3", "R4", "X1", "X2", "X0", "R0", "B0c", "B1c"]:
        z = S.get(k)
        if z is None or "mi_resid_k" not in z: continue
        ex = z["mi_exact_k"].astype(bool); rs = z["mi_resid_k"].astype(np.float64); fin = np.isfinite(rs); ef = ex & fin; wf = (~ex) & fin
        auc = 1.0 - float(stats.mannwhitneyu(rs[ef], rs[wf], alternative="less").statistic / (ef.sum() * wf.sum())) if ef.sum() and wf.sum() else None
        thr = np.median(rs[ef]) if ef.any() else np.nan; spur = float((rs[wf] <= thr).mean()) if wf.any() else None
        has = ex.any(1); best = np.argmin(np.where(fin, rs, np.inf), 1); pick = ex[np.arange(len(ex)), best]; t1r = float(pick.mean()); ver = float((z["cold_exact"].astype(bool) | has).mean())
        E5[k] = dict(auc=auc, spurious=spur, t1r=t1r, verified=ver, ratio=t1r / ver if ver else None, minres_correct_given_hit=float(pick[has].mean()) if has.any() else None)
        say(f"  {k:4s} | AUC {f(auc,5,3)} | spurious (wrong draws below the correct median) {pp(spur)} | t1r@128 {pp(t1r)} / verified@128 {pp(ver)} = {f(E5[k]['ratio'],5,3)} | min-resid draw correct given >=1 hit {pp(E5[k]['minres_correct_given_hit'])}")
    J["E5"] = E5

    # ---- E6 ----
    say("\n== E6. CALIBRATION AT STALLS (the reliability of the map's hard decisions where a decimating decoder would use them; from the calib rows, strat-512) ==")
    say("  map | mode | cold | n stalled | top-5 correct on stalled | mean conf | overconfidence gap (conf - correct) | entropy step1 | entropy t64 stalled | conf-wrong frac stalled")
    E6 = {}
    for k in ["W0", "R1", "R2", "R3", "R4", "X1", "X2"]:
        for mode in ("vsel", "vsel_hard"):
            c = jload(RUNS / f"sxcalib_psportC2{k}_{mode}" / "calib.json")
            if not c: continue
            gap = (c["mean_conf_stalled"] - c["topk_correct_stalled"]) if c.get("mean_conf_stalled") is not None and c.get("topk_correct_stalled") is not None else None
            E6[f"{k}{'_hard' if mode.endswith('hard') else ''}"] = dict(top5=c.get("topk_correct_stalled"), conf=c.get("mean_conf_stalled"), gap=gap, ent1=c.get("entropy_step1"), ent_st=c.get("entropy_t_stalled"), cw=c.get("conf_wrong_frac_stalled"), cold=c.get("cold"), n_st=c.get("n_stalled"))
            say(f"  {k:4s} | {'HARD' if c.get('hard_feedback') else 'soft'} | {pp(c.get('cold'))} | {c.get('n_stalled'):4d} | {pp(c.get('topk_correct_stalled'))} | {f(c.get('mean_conf_stalled'),5,3)} | {f(gap,6,3)} | {f(c.get('entropy_step1'),5,3)} | {f(c.get('entropy_t_stalled'),5,3)} | {pp(c.get('conf_wrong_frac_stalled'))}")
    say("  sportC1 references (ground 7, strat-256): R0 top-5 72.2 % at conf .96; B0 vsel 70.1 % at .95; solved-set control 100 %.")
    J["E6"] = E6

    # ---- E2 / E3 (dynamics, CPU) ----
    if not a.no_dyn:
        import jax, jax.numpy as jnp
        from qhrrn2 import episodic as E, grid as GR, model as M, sudoku as SU
        from qhrrn2.config import Config
        import eval_sudoku_extreme as EV
        CK = {"W0": None, "R1": None, "R2": None, "R3": None, "R4": None, "X1": None, "X2": None, "R0": (RUNS / "pretrainsportC1_R0/ckpt_latest.pkl", True), "B0": (RUNS / "pretrainsportC1_B0a/ckpt_020000.pkl", False), "X0": (RUNS / "pretrainsportC1_X0/ckpt_latest.pkl", True)}
        for k in ("W0", "R1", "R2", "R3", "R4", "X1", "X2"):
            s = jload(RUNS / f"sxeval_psportC2{k}" / ("full_vsel_t16" if k in ("X1", "X2") else "full_vsel_t64") / "summary_all.json")
            if s: CK[k] = (ROOT / s["ckpt"] if not str(s["ckpt"]).startswith("/") else Path(s["ckpt"]), bool(s.get("ema")))
        maps = [m for m in a.maps.split(",") if CK.get(m)]
        ids = SX.stratified_subsample(R, a.n, 20260821); B = len(ids); puz9 = Q[ids].astype(np.int32); sol9 = A[ids].astype(np.int32); ng = puz9 == 0; giv = ~ng
        say(f"\n== E2. DECODER DYNAMICS on cold trajectories (strat-{B}, seed 20260821, t={a.t}; mirrors the evaluator's step; R2 with its trained K; R3 soft = the registered mode, + a HARD row) ==")
        say("  map | solved | non-given cells correct by step 1/2/4/8/16/32/64: solved | unsolved || readout entropy solved | unsolved || commitment (p>.9) frac step 1/8/64 solved | unsolved || confidently-wrong step 1/8/64 solved | unsolved")
        E2 = {}; E3 = {}
        def run_map(name, ckpt, ema, hard=False):
            saved = E.load_ckpt(str(ckpt)); defaults = Config(); cfg = Config(**{kk: type(getattr(defaults, kk))(v) for kk, v in saved["config"].items()})
            st = saved["state_ema"] if ema else saved["state"]; params = st["model"]; tvj = jnp.asarray(st["table"][0])
            eta, eta_z = (float(v) for v in M.eq_etas(params, cfg)); layout = cfg.sudoku_layout or "origin"; cv = SU.layout_canvas(layout); trm = cfg.cell_kind == "trm"; K = 1 if trm else max(1, int(getattr(cfg, "inner_k", 1)))
            x_can = jnp.asarray(np.stack([SU.place_layout(gq.astype(np.int8), layout) for gq in puz9]), jnp.int32)
            void = jax.nn.one_hot(jnp.full((cv, cv), GR.VOID, jnp.int32), M.VOCAB).transpose(2, 0, 1); y = jnp.broadcast_to(void, (B,) + void.shape); z = None
            preds = np.zeros((a.t, B, 9, 9), np.int16); ent = np.zeros((a.t, B)); cw = np.zeros((a.t, B)); com = np.zeros((a.t, B)); cc = np.zeros((a.t, B)); syn = np.zeros((a.t, B)); p9_last = None; tt = time.time()
            for t in range(a.t):
                tn = 0.0 if trm else min(t, cfg.T - 1) / max(cfg.T - 1, 1)
                for _ in range(K):
                    first = z is None; logits, zf = EV._step(cfg, 1.0, float(tn), first)(params, x_can, y, tvj, jnp.zeros(1) if first else z); z = zf if first else z + eta_z * (zf - z)
                p = jax.nn.softmax(logits, axis=-1); pf = jax.nn.one_hot(jnp.argmax(logits, axis=-1), M.VOCAB) if hard else p; y = y + eta * (pf.transpose(0, 3, 1, 2) - y)
                p9 = np.asarray(EV.layout_gather(p, layout))[..., 1:10]; p9 = p9 / np.maximum(p9.sum(-1, keepdims=True), 1e-9)
                pred = np.asarray(EV.layout_gather(jnp.argmax(logits, axis=-1), layout)); pred = np.where(pred == GR.VOID, 0, pred).astype(np.int16); preds[t] = pred
                e = -(p9 * np.log(p9 + 1e-12)).sum(-1) / np.log(9); conf = p9.max(-1); corr = pred == sol9; nng = np.maximum(ng.sum((1, 2)), 1)
                ent[t] = (e * ng).sum((1, 2)) / nng; cw[t] = ((conf > .9) & ~corr & ng).sum((1, 2)) / nng; com[t] = ((conf > .9) & ng).sum((1, 2)) / nng; cc[t] = (corr & ng).sum((1, 2)) / nng; syn[t] = violations(pred); p9_last = p9
            solved = (preds[-1] == sol9).all((1, 2)); fe = np.array([next((t for t in range(a.t) if (preds[t, b] == sol9[b]).all()), -1) for b in range(B)])
            corr_t = preds == sol9[None]; ever = np.zeros((B, 9, 9), bool); unpeel = np.zeros(B); flips_c = np.zeros((a.t, B)); flips_w = np.zeros((a.t, B))
            for t in range(a.t):
                if t: ch = (preds[t] != preds[t - 1]) & ng; flips_c[t] = (ch & corr_t[t]).sum((1, 2)) / nng; flips_w[t] = (ch & ~corr_t[t]).sum((1, 2)) / nng; unpeel += ((ever & ~corr_t[t]) & ng).sum((1, 2))
                ever |= corr_t[t]
            mono = (unpeel == 0)
            steps = [1, 2, 4, 8, 16, 32, 64]; sidx = [min(s_, a.t) - 1 for s_ in steps]
            def m_(arr, mask, ii): return [float(arr[i, mask].mean()) if mask.any() else None for i in ii]
            o = dict(solved=float(solved.mean()), cells_s=m_(cc, solved, sidx), cells_u=m_(cc, ~solved, sidx), ent_s=m_(ent, solved, sidx), ent_u=m_(ent, ~solved, sidx), com_s=m_(com, solved, [0, 7, a.t - 1]), com_u=m_(com, ~solved, [0, 7, a.t - 1]),
                     cw_s=m_(cw, solved, [0, 7, a.t - 1]), cw_u=m_(cw, ~solved, [0, 7, a.t - 1]), mono_solved=float(mono[solved].mean()) if solved.any() else None, unpeel_per_cell_solved=float((unpeel / nng)[solved].mean()) if solved.any() else None,
                     unpeel_per_cell_unsolved=float((unpeel / nng)[~solved].mean()) if (~solved).any() else None,
                     flips_w_last32_unsolved=float(flips_w[max(a.t - 32, 0):, ~solved].sum(0).mean()) if (~solved).any() else None, flips_c_last32_unsolved=float(flips_c[max(a.t - 32, 0):, ~solved].sum(0).mean()) if (~solved).any() else None,
                     flips_w_last32_solved=float(flips_w[max(a.t - 32, 0):, solved].sum(0).mean()) if solved.any() else None,
                     syn_u=m_(syn, ~solved, sidx), syn_osc_unsolved=float((syn[max(a.t - 32, 0):, ~solved].max(0) - syn[max(a.t - 32, 0):, ~solved].min(0)).mean()) if (~solved).any() else None,
                     syn_mono_frac_unsolved=float(np.mean([bool(np.all(np.diff(syn[:, b]) <= 0)) for b in np.where(~solved)[0]])) if (~solved).any() else None,
                     fe_med=float(np.median(fe[solved])) if solved.any() else None, fe_p90=float(np.percentile(fe[solved], 90)) if solved.any() else None, wall=round(time.time() - tt, 1), K=K, hard=hard)
            # E3: decimation quality at stalls: committed cells (p > tau) as hard decisions -> peeling
            e3 = {}
            for tau in (.9, .99):
                U = ~solved
                if not U.any(): break
                conf = p9_last.max(-1); com_m = (conf > tau) & ng; n_com = com_m.sum((1, 2)) / nng
                wrong_com = ((com_m & (preds[-1] != sol9)).sum((1, 2)) / np.maximum(com_m.sum((1, 2)), 1))
                gg0 = np.where(com_m, preds[-1], puz9)                                      # givens + committed
                pg_, st_ = peel(gg0[U]); ok = (st_ == 1) & (pg_ == sol9[U]).all((1, 2))
                e3[str(tau)] = dict(n_stalled=int(U.sum()), committed_frac=float(n_com[U].mean()), wrong_committed_frac=float(wrong_com[U].mean()), all_committed_correct=float((wrong_com[U] == 0).mean()),
                                    peel_solves=float(ok.mean()), peel_stuck=float((st_ == 0).mean()), peel_contradiction=float((st_ == 2).mean()))
            pg0, st0 = peel(puz9[~solved]) if (~solved).any() else (None, None)
            e3["givens_only"] = dict(peel_solves=float(((st0 == 1) & (pg0 == sol9[~solved]).all((1, 2))).mean())) if st0 is not None else None
            return o, e3
        for name in maps:
            ck, ema = CK[name]
            for hard in ((False, True) if name == "R3" else (False,)):
                try: o, e3 = run_map(name, ck, ema, hard)
                except Exception as ex_: say(f"  {name}: FAILED {type(ex_).__name__}: {ex_}"); continue
                key = name + ("_hard" if hard else ""); E2[key] = o; E3[key] = e3
                r_ = lambda xs: "/".join("-" if v is None else f"{100*v:.0f}" for v in xs); r3 = lambda xs: "/".join("-" if v is None else f"{v:.2f}" for v in xs)
                say(f"  {key:7s} K{o['K']} | {pp(o['solved'])} | {r_(o['cells_s'])} | {r_(o['cells_u'])} || {r3(o['ent_s'])} | {r3(o['ent_u'])} || {r_(o['com_s'])} | {r_(o['com_u'])} || {r_(o['cw_s'])} | {r_(o['cw_u'])} ({o['wall']}s)")
        say("\n  REVISION / MONOTONICITY / SYNDROME (a peeling decoder is monotone: a decided-correct cell never flips back; BP on a trapping set oscillates):")
        say("  map | monotone solved trajectories | un-peel events per cell (solved / unsolved) | flips-to-wrong per cell in the last 32 steps (unsolved / solved) | flips-to-correct last 32 (unsolved) | syndrome (violations) of the argmax grid, unsolved, by step 1/2/4/8/16/32/64 | late oscillation (max-min over the last 32) | syndrome monotone-nonincreasing frac (unsolved) | first_exact median / p90")
        for key, o in E2.items():
            say(f"  {key:7s} | {pp(o['mono_solved'])} | {f(o['unpeel_per_cell_solved'],5,3)} / {f(o['unpeel_per_cell_unsolved'],5,3)} | {f(o['flips_w_last32_unsolved'],5,2)} / {f(o['flips_w_last32_solved'],5,2)} | {f(o['flips_c_last32_unsolved'],5,2)} | " + "/".join("-" if v is None else f"{v:.0f}" for v in o['syn_u']) + f" | {f(o['syn_osc_unsolved'],5,1)} | {pp(o['syn_mono_frac_unsolved'])} | {f(o['fe_med'],3,0)} / {f(o['fe_p90'],3,0)}")
        say("\n== E3. DECIMATION QUALITY AT STALLS: the map's committed cells (p > tau) handed to the PEELING decoder as hard decisions (strat set, stalled puzzles) ==")
        say("  map | tau | n stalled | committed frac of non-given | wrong among committed | P(all committed correct) | peeling from givens+committed: solves / stuck / contradiction | peeling from givens only solves")
        for key, e3 in E3.items():
            for tau in ("0.9", "0.99"):
                q = e3.get(tau)
                if q: say(f"  {key:7s} | {tau:3s} | {q['n_stalled']:4d} | {pp(q['committed_frac'])} | {pp(q['wrong_committed_frac'])} | {pp(q['all_committed_correct'])} | {pp(q['peel_solves'])} / {pp(q['peel_stuck'])} / {pp(q['peel_contradiction'])} | {pp((e3.get('givens_only') or {}).get('peel_solves'))}")
        say("  READ: a calibrated committer hands the peeler only correct cells (contradiction ~0) and enough of them to finish propagation (solves high); our sportC1 maps were 28-30 % wrong at p .96.")
        J["E2"] = E2; J["E3"] = E3
    # ---- E7 scorecard ----
    say("\n== E7. DECODER SCORECARD (one row per map; every column defined above; lower g50 / higher yield = a stronger decoder on this channel) ==")
    say("  map | g50 cold | width | g50 reach | P(cold|r0) | P(cold|r>0) | rho | median r_i | k50 | selector AUC | spurious | top-5 correct at stalls (gap) | monotone solved | churn (flips-to-wrong/cell, last 32, unsolved) | syndrome osc | commit@1 (solved)")
    for k in ["W0", "R1", "R2", "R3", "R4", "X1", "X2", "X0", "R0", "B0c", "B1c", "C3X", "D4"]:
        e1 = E1.get(k, {}); e4 = E4.get(k, {}); e5 = E5.get(k, {}); e6 = E6.get(k, {}); kd = {"B0c": "B0"}.get(k, k); e2 = J.get("E2", {}).get(kd, {})
        say(f"  {k:4s} | {f(e1.get('g50_cold'),5,1)} | {f(e1.get('width_cold'),4,1)} | {f(e1.get('g50_reach'),5,1)} | {pp(e1.get('p_cold_r0'))} | {pp(e1.get('p_cold_rpos'))} | {pp(e4.get('rho'))} | {f(e4.get('r_med'),5,3)} | {e4.get('k50')} | {f(e5.get('auc'),5,3)} | {pp(e5.get('spurious'))} | {pp(e6.get('top5'))} ({f(e6.get('gap'),5,2)}) | {pp(e2.get('mono_solved'))} | {f(e2.get('flips_w_last32_unsolved'),5,2)} | {f(e2.get('syn_osc_unsolved'),5,1)} | {pp((e2.get('com_s') or [None])[0])}")
    say(f"\n({time.time()-t0:.0f}s)")
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text("\n".join(L) + "\n"); OUTJ.write_text(json.dumps(J, indent=1, default=float)); say(f"artifact -> {OUT} (+ .json)")

if __name__ == "__main__":
    main()
