#!/usr/bin/env python3
"""FINAL PHASE Night A — THE ERROR-CORRECTING-CODE / DECODER LENS on the finalA arms (analysis-time, descriptive,
NO RULES; 2026-09-07). Verdict authority = tools/analyze_finalA.py. Derivative of analyze_sportC2_ecc.py (the same
E0-E7 frame and the same reference decoders); the sources differ by class:
  X0-class arms (A0-A2): the 20k k128 t64 scan (cold / reach / b1 / residual) + the D16 full;
  wide DEC arms (A3-A8): NO scan (deferred offline re-run) -> thresholds from the D64 100k row and the D16 full
      (cold only, labeled); list decoding + the selector from the vb SCREEN (strat-512, k256, headline depth D16;
      labeled) — every arm has a vb screen, so E4s/E5s compare all nine arms and X0 at one protocol;
  references: X0 (sportC1 scan + full), X1 (sportC2 scan + full), R3 (sportC2 scan), B0c (sportC1 correct grid).
  E2/E3 dynamics (CPU, strat-N cold trajectories) run on the maps named by --maps (the DEC through the evaluator's
  step; cell_kind 'dec' mirrors 'trm': one compiled step, t_norm ignored, K = 1).

  PYTHONPATH=src JAX_PLATFORMS=cpu .venv/bin/python tools/analyze_finalA_ecc.py [--n 128] [--maps A3,A7,A0,X0] [--no-dyn]
      -> runs/analysis/finalA_ecc_20260907.txt (+ .json)
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; RUNS = Path(os.environ.get("QHRRN_RUNS", ROOT / "runs"))
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tools"))
OUT = RUNS / "analysis" / "finalA_ecc_20260907.txt"; OUTJ = RUNS / "analysis" / "finalA_ecc_20260907.json"
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
TAG = "finalA"; XA = ["A0", "A1", "A2"]; WA = ["A3", "A4", "A5", "A7", "A8"]; ALL = XA + WA
SCANS = {"A0": f"sxscan_p{TAG}A0", "A1": f"sxscan_p{TAG}A1", "A2": f"sxscan_p{TAG}A2", "X0": "sxscan_psportC1X0", "X1": "sxscan_psportC2X1", "R3": "sxscan_psportC2R3", "B0c": "sxscan_psportC1B0_vselA20k"}
D64 = {a: f"sxeval_p{TAG}{a}/full_vsel_t64" for a in ALL}; D64.update({"X0": "sxeval_psportC1X0/full_vsel_t64", "X1": "sxeval_psportC2X1/full_vsel_t64", "R3": "sxeval_psportC2R3/full_vsel_t64"})
D16 = {a: f"sxeval_p{TAG}{a}/full_vsel_t16" for a in ALL}; D16.update({"X0": "sxeval_psportC1X0/full_vsel_t16", "X1": "sxeval_psportC2X1/full_vsel_t16"})
FIN = {a: f"sxeval_p{TAG}{a}/full_final_t16" for a in ALL}
SCREENS = {a: f"sxscreen_p{TAG}{a}_vb" for a in ALL}; SCREENS["X0"] = "sxscreen_psportC1X0_vb"
DESC = {"A0": "X0 seed 1 (floor pair)", "A1": "X0 + FPA k1", "A2": "X0 + RI s1", "A3": "DEC-w384 no aug s0", "A4": "DEC-w384 + digit aug", "A5": "DEC-w384 + FPA + RI", "A7": "DEC-w384 no aug s1", "A8": "DEC-w512 no aug s1",
        "X0": "FIELD baseline (sportC1)", "X1": "X0 - digit aug (sportC2)", "R3": "our champion by rule (sportC2)", "B0c": "B0 correct grid (sportC1)"}
GBINS = [(17, 21), (21, 23), (23, 25), (25, 27), (27, 29), (29, 36)]; RBANDS = [(0, 1), (1, 10), (10, 30), (30, 60), (60, 10**6)]

# ---------- the PEELING decoder (naked + hidden singles), vectorized over puzzles (verbatim from the sportC2 lens) ----------
def peel(grid0, max_iter=100):
    g = grid0.astype(np.int16).copy(); N = len(g); status = np.zeros(N, np.int8); active = np.ones(N, bool)
    bi = (np.arange(9)[:, None] // 3) * 3 + (np.arange(9)[None, :] // 3)
    for _ in range(max_iter):
        oh = (g[..., None] == np.arange(1, 10))
        rowc = oh.sum(2); colc = oh.sum(1)
        boxc = np.zeros((N, 9, 9), int)
        for b in range(9): boxc[:, b] = oh[:, bi == b].sum(1)
        dup = (np.maximum(rowc - 1, 0).sum((1, 2)) + np.maximum(colc - 1, 0).sum((1, 2)) + np.maximum(boxc - 1, 0).sum((1, 2))) > 0
        rowhas = rowc > 0; colhas = colc > 0; boxhas = boxc > 0
        cand = ~rowhas[:, :, None, :] & ~colhas[:, None, :, :] & ~boxhas[:, bi, :] & (g == 0)[..., None]
        empty = g == 0; ncand = cand.sum(-1)
        contra = dup | ((empty & (ncand == 0)).any((1, 2)))
        newly = contra & active; status[newly] = 2; active &= ~contra
        solved = (~empty.any((1, 2))) & active; status[solved] = 1; active &= ~solved
        if not active.any(): break
        changed = np.zeros(N, bool); fill = np.zeros_like(g)
        ns = empty & (ncand == 1) & active[:, None, None]
        fill[ns] = cand.argmax(-1)[ns] + 1; changed |= ns.any((1, 2))
        for unit in ("row", "col", "box"):
            if unit == "row": cnt = cand.sum(2); pos = cand.argmax(2)
            elif unit == "col": cnt = cand.sum(1); pos = cand.argmax(1)
            else:
                cb = np.stack([cand[:, bi == b] for b in range(9)], 1); cnt = cb.sum(2); pos = cb.argmax(2)
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
def violations(pred):
    oh = (pred[..., None] == np.arange(1, 10)); N = len(pred); bi = (np.arange(9)[:, None] // 3) * 3 + (np.arange(9)[None, :] // 3)
    rowc = oh.sum(2); colc = oh.sum(1); boxc = np.stack([oh[:, bi == b].sum(1) for b in range(9)], 1)
    return np.maximum(rowc - 1, 0).sum((1, 2)) + np.maximum(colc - 1, 0).sum((1, 2)) + np.maximum(boxc - 1, 0).sum((1, 2))
def logistic_g50(g, y):
    from scipy.optimize import minimize
    g = g.astype(float); y = y.astype(float)
    if y.mean() in (0.0, 1.0): return (None, None, None)
    def nll(th):
        a, g50 = th; zz = np.clip(a * (g - g50), -30, 30); p = 1 / (1 + np.exp(-zz)); return -np.sum(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
    r = minimize(nll, x0=[0.5, 25.0], method="Nelder-Mead"); a, g50 = r.x
    return (float(g50), float(a), float(2 * np.log(9) / abs(a)) if a else None)

def threshold_row(k, z, G, R, extra=""):
    zi = z["idx"]; gg = G[zi]; rr = R[zi]; cold = z["cold_exact"].astype(bool)
    reach = cold | (z["mi_first_hit"] >= 0) if "mi_first_hit" in z else None; b1 = z["mi_exact_k"][:, 0].astype(bool) if "mi_exact_k" in z else None
    cells = []
    for lo, hi in GBINS:
        m = (gg >= lo) & (gg < hi); cells.append((f"{pp(cold[m].mean()) if m.sum() else '  -  '}" + (f"/{pp(reach[m].mean())}" if reach is not None and m.sum() else "")))
    g50c, ac, wc = logistic_g50(gg, cold); g50r = logistic_g50(gg, reach)[0] if reach is not None else None
    sc = float(np.corrcoef(cold.astype(float), gg)[0, 1])
    row = dict(n=int(len(zi)), g50_cold=g50c, width_cold=wc, g50_reach=g50r, p_cold_r0=float(cold[rr == 0].mean()), p_cold_rpos=float(cold[rr > 0].mean()), p_reach_rpos=(float(reach[rr > 0].mean()) if reach is not None else None),
               bands={f"{lo}-{hi}": float(cold[(rr >= lo) & (rr < hi)].mean()) for lo, hi in RBANDS if ((rr >= lo) & (rr < hi)).any()}, corr_g=sc, b1=(float(b1.mean()) if b1 is not None else None))
    say(f"  {k:4s} {extra:14s} n={row['n']:6d} | " + " | ".join(cells) + f" | g50 {f(g50c,5,1)} (w {f(wc,4,1)}) | g50 reach {f(g50r,5,1)} | P(cold|r0) {pp(row['p_cold_r0'])} | yield {pp(row['p_cold_rpos'])} | corr(cold,g) {sc:+.3f}")
    return row

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=128); ap.add_argument("--maps", default="A3,A7,A5,A8,A0,X0"); ap.add_argument("--no-dyn", action="store_true"); ap.add_argument("--t", type=int, default=64)
    a = ap.parse_args(); t0 = time.time()
    from qhrrn2 import sudoku_extreme as SX
    d = SX.load_prepared(NPZ); Q, A, R = d["test_q"], d["test_a"], d["test_rating"]
    G = (Q != 0).reshape(len(Q), -1).sum(1)
    say("=" * 124); say("FINAL PHASE Night A — THE DECODER LENS (2026-09-07; analysis-time, descriptive, no rules; verdict authority = analyze_finalA.py)"); say("=" * 124)
    S = {k: recs(RUNS / p) for k, p in SCANS.items()}; S = {k: v for k, v in S.items() if v is not None}
    Z64 = {k: recs(RUNS / p) for k, p in D64.items()}; Z64 = {k: v for k, v in Z64.items() if v is not None}
    Z16 = {k: recs(RUNS / p) for k, p in D16.items()}; Z16 = {k: v for k, v in Z16.items() if v is not None}
    SC = {k: recs(RUNS / p) for k, p in SCREENS.items()}; SC = {k: v for k, v in SC.items() if v is not None}
    say(f"\nsources: scans {sorted(S)} | D64 rows {sorted(Z64)} | D16 fulls {sorted(Z16)} | vb screens {sorted(SC)}")
    base = S["X0"]["idx"] if "X0" in S else next(iter(S.values()))["idx"]
    say(f"scan idx identical to X0's 20k: {[k for k, v in S.items() if np.array_equal(v['idx'], base)]}; wide-arm D64 idx identical across the wide arms: {all(np.array_equal(Z64[k]['idx'], Z64['A3']['idx']) for k in WA if k in Z64 and 'A3' in Z64)}")
    # ---- E0 ----
    say("\n== E0. THE CHANNEL and the peeling reference on the 20k scan set (erasures = 81 - givens) ==")
    idx = base; g = G[idx]; rat = R[idx]
    tp = time.time(); pg, ps = peel(Q[idx]); peel_ok = (ps == 1) & (pg == A[idx]).all((1, 2)); r0 = rat == 0
    say(f"  peeling: solves {100*peel_ok.mean():.2f} % of the 20k | rating-0 class {100*r0.mean():.2f} %; peeling solves {100*peel_ok[r0].mean():.1f} % of it and {100*peel_ok[~r0].mean():.2f} % above ({time.time()-tp:.0f}s)")
    J["E0"] = dict(peel_solve=float(peel_ok.mean()), rating0=float(r0.mean()))
    # ---- E1 ----
    say("\n== E1. ERASURE-THRESHOLD CURVES: cold[/reach] per givens bin; logistic g50 + 10-90 width; propagation vs search class ==")
    say("  (a) at D64 — X0-class from the 20k scans (cold/reach@128); wide arms from the D64 100k row (cold only); references X0/X1/R3 scans")
    E1 = {}
    for k in ["A0", "A1", "A2", "X0", "X1", "R3", "B0c"]:
        if k in S: E1[f"{k}@64 scan"] = threshold_row(k, S[k], G, R, "scan 20k k128")
    for k in WA:
        if k in Z64: E1[f"{k}@64 full"] = threshold_row(k, Z64[k], G, R, "D64 100k cold")
    say("  (b) at D16 — the headline depth, every arm on the FULL test (cold only) + X0 / X1")
    for k in ALL + ["X0", "X1"]:
        if k in Z16: E1[f"{k}@16 full"] = threshold_row(k, Z16[k], G, R, "D16 full cold")
    say("  search-demand spectrum (cold by tdoku rating band 0 | 1-9 | 10-29 | 30-59 | 60+):")
    for k, e in E1.items(): say(f"  {k:14s} | " + " | ".join(pp(e["bands"].get(f"{lo}-{hi}")) for lo, hi in RBANDS))
    say("  DECODING RADIUS vs training (D16: vsel grid vs final grid; wide arms on the 50k final subsample -> intersection, labeled): arm | bin: vsel -> final")
    for k in ALL:
        zv, zf = Z16.get(k), recs(RUNS / FIN[k])
        if zv is None or zf is None: continue
        common, ia, ib = np.intersect1d(zv["idx"], zf["idx"], return_indices=True)
        if len(common) == 0: continue
        gg = G[common]; cv = zv["cold_exact"].astype(bool)[ia]; cf = zf["cold_exact"].astype(bool)[ib]
        if np.array_equal(cv, cf): say(f"  {k}: identical"); continue
        say(f"  {k:3s} n={len(common):6d} | " + " | ".join(f"[{lo},{hi}) {pp(cv[(gg>=lo)&(gg<hi)].mean()).strip()}->{pp(cf[(gg>=lo)&(gg<hi)].mean()).strip()}" for lo, hi in GBINS) + f" | all {pp(cv.mean()).strip()}->{pp(cf.mean()).strip()}")
    J["E1"] = E1
    # ---- E4 / E5 on scans (X0-class) and on screens (every arm, labeled) ----
    from scipy import stats
    def list_dec(z, kmax, label_k):
        cold = z["cold_exact"].astype(bool); fh = z["mi_first_hit"]; ks = [kk for kk in (1, 2, 4, 8, 16, 32, 64, 128, 256) if kk <= kmax]
        vk = {kk: float((cold | ((fh >= 0) & (fh < kk))).mean()) for kk in ks}
        ex = z["mi_exact_k"].astype(bool)[:, :kmax]; ri = ex.mean(1); reach = cold | (fh >= 0); rr_ = ri[reach]
        bins = [float((ri == 0).mean()), float(((ri > 0) & (ri <= .05)).mean()), float(((ri > .05) & (ri <= .2)).mean()), float(((ri > .2) & (ri <= .5)).mean()), float((ri > .5).mean())]
        fh_eff = np.where(cold, -1, fh); hits = np.sort(fh_eff[reach] + 1); n_r = len(hits); k50 = int(hits[int(.5 * n_r)]) if n_r else None; k90 = int(hits[int(.9 * n_r)]) if n_r else None
        resc1 = float(ex[~cold, 0].mean()) if (~cold).any() else None; resc = float((fh[~cold] >= 0).mean()) if (~cold).any() else None
        rs = z["mi_resid_k"].astype(np.float64)[:, :kmax]; fin = np.isfinite(rs); ef = ex & fin; wf = (~ex) & fin
        auc = 1.0 - float(stats.mannwhitneyu(rs[ef], rs[wf], alternative="less").statistic / (ef.sum() * wf.sum())) if ef.sum() and wf.sum() else None
        thr = np.median(rs[ef]) if ef.any() else np.nan; spur = float((rs[wf] <= thr).mean()) if wf.any() else None
        best = np.argmin(np.where(fin, rs, np.inf), 1); pick = ex[np.arange(len(ex)), best]; t1r = float(pick.mean()); ver = vk[ks[-1]]
        uv = {kk: float(z[f"uv_vote_k{kk}"].mean()) for kk in ks if f"uv_vote_k{kk}" in z}
        return dict(k=label_k, vk=vk, rho=float(reach.mean()), b1=float(ex[:, 0].mean()), r_med=float(np.median(rr_)) if len(rr_) else None, bins=bins, k50=k50, k90=k90, resc1=resc1, resc=resc, auc=auc, spurious=spur, t1r=t1r, verified=ver, ratio=(t1r / ver if ver else None), uv=uv, cold=float(cold.mean()))
    def show_ld(label, rows):
        say(f"  {label}: map | cold | b1 | verified@1/2/4/8/16/32/64/128[/256] | rho | median r_i | r_i bins 0/(0,.05]/(.05,.2]/(.2,.5]/(.5,1] | k50/k90 | rescue 1 / any | AUC | spurious | t1r@k / verified@k = ratio | majority@k")
        for k, e in rows.items():
            say(f"  {k:4s} | {pp(e['cold'])} | {pp(e['b1'])} | " + "/".join(pp(v).strip() for v in e['vk'].values()) + f" | {pp(e['rho'])} | {f(e['r_med'],5,3)} | " + "/".join(pp(b).strip() for b in e['bins']) + f" | {e['k50']}/{e['k90']} | {pp(e['resc1'])} / {pp(e['resc'])} | {f(e['auc'],5,3)} | {pp(e['spurious'])} | {pp(e['t1r'])} / {pp(e['verified'])} = {f(e['ratio'],5,3)} | {pp(e['uv'].get(e['k'])) if e['uv'] else '  -  '}")
    say("\n== E4/E5. LIST DECODING + THE SOFT-SYNDROME SELECTOR ==")
    E4 = {}
    for k in ["A0", "A1", "A2", "X0", "X1", "R3", "B0c"]:
        if k in S and "mi_resid_k" in S[k]: E4[k] = list_dec(S[k], 128, 128)
    show_ld("(a) 20k scans, k128, t64 (the registered X0-class protocol)", E4)
    E4s = {}
    for k in ALL + ["X0"]:
        if k in SC and "mi_resid_k" in SC[k]: E4s[k] = list_dec(SC[k], 256, 256)
    show_ld("(b) vb SCREENS, strat-512, k256, headline depth D16 (every arm incl. the wide DEC arms; labeled — not the registered scan)", E4s)
    J["E4_scans"] = E4; J["E4_screens"] = E4s
    # ---- E6 ----
    say("\n== E6. CALIBRATION AT STALLS (strat-512, vsel grid): arm | cold | n stalled | top-5 correct on stalled | mean conf | gap | entropy step1 | entropy stalled | conf-wrong frac | committed frac ==")
    E6 = {}
    for k in ALL:
        c = jload(RUNS / f"sxcalib_p{TAG}{k}_vsel" / "calib.json")
        if not c: continue
        gap = (c["mean_conf_stalled"] - c["topk_correct_stalled"]) if c.get("mean_conf_stalled") is not None and c.get("topk_correct_stalled") is not None else None
        E6[k] = dict(top5=c.get("topk_correct_stalled"), conf=c.get("mean_conf_stalled"), gap=gap, ent1=c.get("entropy_step1"), ent_st=c.get("entropy_t_stalled"), cw=c.get("conf_wrong_frac_stalled"), cold=c.get("cold"), n_st=c.get("n_stalled"), com=c.get("committed_frac_stalled"))
        say(f"  {k:3s} | {pp(c.get('cold'))} | {c.get('n_stalled'):4d} | {pp(c.get('topk_correct_stalled'))} | {f(c.get('mean_conf_stalled'),5,3)} | {f(gap,6,3)} | {f(c.get('entropy_step1'),5,3)} | {f(c.get('entropy_t_stalled'),5,3)} | {pp(c.get('conf_wrong_frac_stalled'))} | {pp(c.get('committed_frac_stalled'))}")
    say("  sportC2 references (strat-512): X0 top-5 ~50 % at conf 1.00 (field-class); W0/R3/R4 68-76 % at .96-.98; the public field checkpoints 46-53 % at 1.000.")
    J["E6"] = E6
    # ---- E2 / E3 (dynamics, CPU) ----
    if not a.no_dyn:
        import jax, jax.numpy as jnp
        from qhrrn2 import episodic as E, grid as GR, model as M, sudoku as SU
        from qhrrn2.config import Config
        import eval_sudoku_extreme as EV
        CK = {"X0": (RUNS / "pretrainsportC1_X0/ckpt_latest.pkl", True), "X1": (RUNS / "pretrainsportC2_X1/ckpt_020000.pkl", True)}
        for k in ALL:
            s = jload(RUNS / f"sxeval_p{TAG}{k}" / "full_vsel_t16" / "summary_all.json")
            if s: CK[k] = (ROOT / s["ckpt"] if not str(s["ckpt"]).startswith("/") else Path(s["ckpt"]), bool(s.get("ema")))
        maps = [m for m in a.maps.split(",") if CK.get(m) and Path(CK[m][0]).exists()]
        ids = SX.stratified_subsample(R, a.n, 20260821); B = len(ids); puz9 = Q[ids].astype(np.int32); sol9 = A[ids].astype(np.int32); ng = puz9 == 0
        say(f"\n== E2. DECODER DYNAMICS on cold trajectories (strat-{B}, seed 20260821, t={a.t}; the evaluator's step; headline weights) ==")
        say("  map | solved | non-given cells correct by step 1/2/4/8/16/32/64: solved | unsolved || readout entropy solved | unsolved || commitment (p>.9) step 1/8/64 solved | unsolved || confidently-wrong step 1/8/64 solved | unsolved")
        E2 = {}; E3 = {}
        def run_map(name, ckpt, ema):
            saved = E.load_ckpt(str(ckpt)); defaults = Config(); cfg = Config(**{kk: type(getattr(defaults, kk))(v) for kk, v in saved["config"].items()})
            st = saved["state_ema"] if ema else saved["state"]; params = st["model"]; tvj = jnp.asarray(st["table"][0])
            eta, eta_z = (float(v) for v in M.eq_etas(params, cfg)); layout = cfg.sudoku_layout or "origin"; cv = SU.layout_canvas(layout); trm = cfg.cell_kind in ("trm", "dec"); K = 1 if trm else max(1, int(getattr(cfg, "inner_k", 1)))
            x_can = jnp.asarray(np.stack([SU.place_layout(gq.astype(np.int8), layout) for gq in puz9]), jnp.int32)
            void = jax.nn.one_hot(jnp.full((cv, cv), GR.VOID, jnp.int32), M.VOCAB).transpose(2, 0, 1); y = jnp.broadcast_to(void, (B,) + void.shape); z = None
            preds = np.zeros((a.t, B, 9, 9), np.int16); ent = np.zeros((a.t, B)); cw = np.zeros((a.t, B)); com = np.zeros((a.t, B)); cc = np.zeros((a.t, B)); syn = np.zeros((a.t, B)); p9_last = None; tt = time.time()
            for t in range(a.t):
                tn = 0.0 if trm else min(t, cfg.T - 1) / max(cfg.T - 1, 1)
                for _ in range(K):
                    first = z is None; logits, zf = EV._step(cfg, 1.0, float(tn), first)(params, x_can, y, tvj, jnp.zeros(1) if first else z); z = zf if first else z + eta_z * (zf - z)
                p = jax.nn.softmax(logits, axis=-1); y = y + eta * (p.transpose(0, 3, 1, 2) - y)
                p9 = np.asarray(EV.layout_gather(p, layout))[..., 1:10]; p9 = p9 / np.maximum(p9.sum(-1, keepdims=True), 1e-9)
                pred = np.asarray(EV.layout_gather(jnp.argmax(logits, axis=-1), layout)); pred = np.where(pred == GR.VOID, 0, pred).astype(np.int16); preds[t] = pred
                e = -(p9 * np.log(p9 + 1e-12)).sum(-1) / np.log(9); conf = p9.max(-1); corr = pred == sol9; nng = np.maximum(ng.sum((1, 2)), 1)
                ent[t] = (e * ng).sum((1, 2)) / nng; cw[t] = ((conf > .9) & ~corr & ng).sum((1, 2)) / nng; com[t] = ((conf > .9) & ng).sum((1, 2)) / nng; cc[t] = (corr & ng).sum((1, 2)) / nng; syn[t] = violations(pred); p9_last = p9
                if t == 0: say(f"    [{name}: step 1 done in {time.time()-tt:.0f}s incl. compile]")
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
                     fe_med=float(np.median(fe[solved])) if solved.any() else None, fe_p90=float(np.percentile(fe[solved], 90)) if solved.any() else None, wall=round(time.time() - tt, 1), K=K, n_unsolved=int((~solved).sum()))
            e3 = {}
            for tau in (.9, .99):
                U = ~solved
                if not U.any(): break
                conf = p9_last.max(-1); com_m = (conf > tau) & ng; n_com = com_m.sum((1, 2)) / nng
                wrong_com = ((com_m & (preds[-1] != sol9)).sum((1, 2)) / np.maximum(com_m.sum((1, 2)), 1))
                gg0 = np.where(com_m, preds[-1], puz9)
                pg_, st_ = peel(gg0[U]); ok = (st_ == 1) & (pg_ == sol9[U]).all((1, 2))
                e3[str(tau)] = dict(n_stalled=int(U.sum()), committed_frac=float(n_com[U].mean()), wrong_committed_frac=float(wrong_com[U].mean()), all_committed_correct=float((wrong_com[U] == 0).mean()),
                                    peel_solves=float(ok.mean()), peel_stuck=float((st_ == 0).mean()), peel_contradiction=float((st_ == 2).mean()))
            pg0, st0 = peel(puz9[~solved]) if (~solved).any() else (None, None)
            e3["givens_only"] = dict(peel_solves=float(((st0 == 1) & (pg0 == sol9[~solved]).all((1, 2))).mean())) if st0 is not None else None
            return o, e3
        for name in maps:
            ck, ema = CK[name]
            try: o, e3 = run_map(name, ck, ema)
            except Exception as ex_: say(f"  {name}: FAILED {type(ex_).__name__}: {ex_}"); continue
            E2[name] = o; E3[name] = e3
            r_ = lambda xs: "/".join("-" if v is None else f"{100*v:.0f}" for v in xs); r3 = lambda xs: "/".join("-" if v is None else f"{v:.2f}" for v in xs)
            say(f"  {name:4s} | {pp(o['solved'])} (unsolved n={o['n_unsolved']}) | {r_(o['cells_s'])} | {r_(o['cells_u'])} || {r3(o['ent_s'])} | {r3(o['ent_u'])} || {r_(o['com_s'])} | {r_(o['com_u'])} || {r_(o['cw_s'])} | {r_(o['cw_u'])} ({o['wall']}s)")
        say("\n  REVISION / MONOTONICITY / SYNDROME: map | monotone solved | un-peel per cell (solved / unsolved) | flips-to-wrong per cell last 32 (unsolved / solved) | flips-to-correct last 32 (unsolved) | syndrome unsolved by step 1/2/4/8/16/32/64 | late oscillation | syndrome monotone frac | first_exact med / p90")
        for key, o in E2.items():
            say(f"  {key:4s} | {pp(o['mono_solved'])} | {f(o['unpeel_per_cell_solved'],5,3)} / {f(o['unpeel_per_cell_unsolved'],5,3)} | {f(o['flips_w_last32_unsolved'],5,2)} / {f(o['flips_w_last32_solved'],5,2)} | {f(o['flips_c_last32_unsolved'],5,2)} | " + "/".join("-" if v is None else f"{v:.0f}" for v in o['syn_u']) + f" | {f(o['syn_osc_unsolved'],5,1)} | {pp(o['syn_mono_frac_unsolved'])} | {f(o['fe_med'],3,0)} / {f(o['fe_p90'],3,0)}")
        say("\n== E3. DECIMATION QUALITY AT STALLS (committed cells p > tau handed to the peeling decoder; stalled puzzles of the strat set) ==")
        say("  map | tau | n stalled | committed frac | wrong among committed | P(all committed correct) | peeling from givens+committed: solves / stuck / contradiction | givens-only solves")
        for key, e3 in E3.items():
            for tau in ("0.9", "0.99"):
                q = e3.get(tau)
                if q: say(f"  {key:4s} | {tau:3s} | {q['n_stalled']:4d} | {pp(q['committed_frac'])} | {pp(q['wrong_committed_frac'])} | {pp(q['all_committed_correct'])} | {pp(q['peel_solves'])} / {pp(q['peel_stuck'])} / {pp(q['peel_contradiction'])} | {pp((e3.get('givens_only') or {}).get('peel_solves'))}")
        J["E2"] = E2; J["E3"] = E3
    # ---- E7 ----
    say("\n== E7. DECODER SCORECARD ==")
    say("  map | g50 D64 | width | yield D64 | g50 D16 | yield D16 | rho (screen k256) | median r_i | selector AUC (screen) | spurious (screen) | t1r/ver (screen) | top-5 at stalls (gap) | monotone solved | churn | syndrome osc | commit@1 solved")
    for k in ALL + ["X0", "X1", "R3"]:
        e64 = E1.get(f"{k}@64 scan") or E1.get(f"{k}@64 full") or {}; e16 = E1.get(f"{k}@16 full") or {}; e4 = E4s.get(k) or E4.get(k) or {}; e6 = E6.get(k, {}); e2 = J.get("E2", {}).get(k, {})
        say(f"  {k:4s} | {f(e64.get('g50_cold'),5,1)} | {f(e64.get('width_cold'),4,1)} | {pp(e64.get('p_cold_rpos'))} | {f(e16.get('g50_cold'),5,1)} | {pp(e16.get('p_cold_rpos'))} | {pp(e4.get('rho'))} | {f(e4.get('r_med'),5,3)} | {f(e4.get('auc'),5,3)} | {pp(e4.get('spurious'))} | {f(e4.get('ratio'),5,3)} | {pp(e6.get('top5'))} ({f(e6.get('gap'),5,2)}) | {pp(e2.get('mono_solved'))} | {f(e2.get('flips_w_last32_unsolved'),5,2)} | {f(e2.get('syn_osc_unsolved'),5,1)} | {pp((e2.get('com_s') or [None])[0])}")
    say(f"\n({time.time()-t0:.0f}s)")
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text("\n".join(L) + "\n"); OUTJ.write_text(json.dumps(J, indent=1, default=float)); say(f"artifact -> {OUT} (+ .json)")

if __name__ == "__main__":
    main()
