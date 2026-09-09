#!/usr/bin/env python3
"""THE INSTRUMENT SUITE — record-level backfill over EVERY banked evaluation (Plan_2026-09-07_Instrument_Suite §2.1 / §5.2;
analysis-time, descriptive, no rules). One row per (campaign, arm, eval) from the evaluator's per-puzzle records:
  R1 thresholds + yield (logistic g50 / width on cold vs givens; in-range flag; P(cold | rating 0); search-class yield; rating bands;
     givens bins), R2 first-exact (median / p90 / fraction at step 1 and <= 2), R3 failure texture (violations on failures, cells,
     givens kept, valid-wrong), and for records with draws R5-R7 (rho, b1, the r_i spectrum, k50 / k90, rescue, residual AUC,
     spurious rate, t1r@k vs verified@k, majority@k where recorded). The Mac field-study rows are joined from their records.

  PYTHONPATH=src .venv/bin/python tools/suite_records.py [--only finalA,sportC2] -> runs/analysis/suite_records_<date>.{txt,json,csv}
"""
from __future__ import annotations
import argparse, csv, json, math, os, re, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; RUNS = Path(os.environ.get("QHRRN_RUNS", ROOT / "runs"))
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tools"))
NPZ = ROOT / "data/sudoku_extreme/sudoku_extreme_seed0.npz"
DATE = time.strftime("%Y%m%d")
OUT = RUNS / "analysis" / f"suite_records_{DATE}"
TAGS = ["champ", "finalA", "sportC2", "sportC1", "sportC0", "sportBr2b", "sportBr2", "sportB", "sport3a", "sport2w2", "sport2"]
GBINS = [(17, 21), (21, 23), (23, 25), (25, 27), (27, 29), (29, 36)]; RBANDS = [(0, 1), (1, 10), (10, 30), (30, 60), (60, 10**6)]

def jload(p):
    p = Path(p)
    try: return json.loads(p.read_text()) if p.exists() else None
    except Exception: return None
def parse_name(name):
    """'psportC2W0' -> ('sportC2', 'W0', ''); 'psportC1B0_vselA20k' -> ('sportC1', 'B0', 'vselA20k'); 'pfinalAA3_vb' -> ('finalA','A3','vb')."""
    m = name[1:] if name.startswith("p") else name
    for t in TAGS:
        if m.startswith(t):
            rest = m[len(t):]; arm, _, suf = rest.partition("_"); return t, arm, suf
    arm, _, suf = m.partition("_"); return "", arm, suf
def logistic_g50(g, y, lo=17, hi=35):
    g = np.asarray(g, float); y = np.asarray(y, float)
    if len(g) < 50 or y.mean() <= 0.0 or y.mean() >= 1.0: return None, None, False
    mu, sd = g.mean(), g.std() + 1e-9; x = (g - mu) / sd; a = b = 0.0
    for _ in range(40):
        p = 1.0 / (1.0 + np.exp(-(a + b * x))); w = p * (1 - p) + 1e-6
        G = np.array([np.sum(y - p), np.sum((y - p) * x)]); H = np.array([[np.sum(w), np.sum(w * x)], [np.sum(w * x), np.sum(w * x * x)]]) + 1e-6 * np.eye(2)
        d = np.linalg.solve(H, G); a += d[0]; b += d[1]
        if np.max(np.abs(d)) < 1e-8: break
    if b <= 0.05: return None, None, False
    g50 = (-a / b) * sd + mu; width = 2 * math.log(9) / (b / sd)
    return float(g50), float(width), bool(lo <= g50 <= hi)
def fit_rho_r(fh, k_fit=64):
    fh = np.asarray(fh); hit = (fh >= 0) & (fh < k_fit); t = fh[hit]; n_c = int(np.sum(~hit)); best = (-np.inf, None, None)
    for rho in np.linspace(0.02, 1.0, 50):
        for r in np.geomspace(1e-3, 0.9, 60):
            ll = np.sum(np.log(rho) + t * np.log1p(-r) + np.log(r)) if len(t) else 0.0
            ll += n_c * np.log(max(1 - rho + rho * (1 - r) ** k_fit, 1e-300))
            if ll > best[0]: best = (ll, rho, r)
    return best[1], best[2]

def analyze_records(z, G, R, kmax=None):
    idx = z["idx"].astype(np.int64); o = np.argsort(idx, kind="stable"); idx = idx[o]
    if idx.max() >= len(G): raise ValueError(f"idx beyond the test set (max {idx.max()}): not a test-set record")
    cold = z["cold_exact"].astype(bool)[o]; n = len(idx); gg = G[idx]; rr = R[idx]
    row = dict(n=int(n), cold=float(cold.mean()))
    g50, width, inr = logistic_g50(gg, cold); row.update(g50=g50, width=width, g50_in_range=inr)
    row["p_cold_r0"] = float(cold[rr == 0].mean()) if (rr == 0).any() else None
    row["yield"] = float(cold[rr > 0].mean()) if (rr > 0).any() else None
    row["bands"] = {f"{lo}-{hi}": float(cold[(rr >= lo) & (rr < hi)].mean()) for lo, hi in RBANDS if ((rr >= lo) & (rr < hi)).any()}
    row["gbins"] = {f"{lo}-{hi}": float(cold[(gg >= lo) & (gg < hi)].mean()) for lo, hi in GBINS if ((gg >= lo) & (gg < hi)).any()}
    if "first_exact" in z:
        fe = z["first_exact"][o]; s = fe[fe >= 0]
        row.update(fe_med=float(np.median(s)) if len(s) else None, fe_p90=float(np.percentile(s, 90)) if len(s) else None,
                   fe_step1=float((s == 0).mean()) if len(s) else None, fe_le2=float((s <= 1).mean()) if len(s) else None)
    if "violations" in z:
        v = z["violations"][o]; row["viol_fail"] = float(v[~cold].mean()) if (~cold).any() else None
        row["valid_wrong"] = float(((v == 0) & ~cold).mean())
    if "cells" in z: row["cells_fail"] = float(z["cells"][o][~cold].mean()) if (~cold).any() else None
    if "givens_kept" in z:
        gk = z["givens_kept"][o]; tot = np.maximum(gg, 1); row["givens_kept"] = float((gk / tot).mean())
    if "mi_first_hit" in z:
        fh = z["mi_first_hit"][o]; reach = cold | (fh >= 0); row["rho"] = float(reach.mean())
        K = int(z["mi_exact_k"].shape[1]) if "mi_exact_k" in z else int(max(fh.max() + 1, 1)); K = min(K, kmax) if kmax else K; row["k"] = K
        row["verified"] = {str(k): float((cold | ((fh >= 0) & (fh < k))).mean()) for k in (1, 2, 4, 8, 16, 32, 64, 128, 256) if k <= K}
        fh_eff = np.where(cold, -1, fh); hits = np.sort(fh_eff[reach] + 1); n_r = len(hits)
        row["k50"] = int(hits[int(.5 * n_r)]) if n_r else None; row["k90"] = int(hits[int(.9 * n_r)]) if n_r else None
        row["rescue_any"] = float((fh[~cold] >= 0).mean()) if (~cold).any() else None
        if "mi_exact_k" in z and z["mi_exact_k"].ndim == 2:
            ex = z["mi_exact_k"][o][:, :K].astype(bool); ri = ex.mean(1); rr_ = ri[reach]
            row.update(b1=float(ex[:, 0].mean()), r_med=float(np.median(rr_)) if len(rr_) else None,
                       r_bins=[float((ri == 0).mean()), float(((ri > 0) & (ri <= .05)).mean()), float(((ri > .05) & (ri <= .2)).mean()), float(((ri > .2) & (ri <= .5)).mean()), float((ri > .5).mean())],
                       rescue_1=float(ex[~cold, 0].mean()) if (~cold).any() else None)
            if "mi_resid_k" in z:
                from scipy import stats
                rs = z["mi_resid_k"][o][:, :K].astype(np.float64); fin = np.isfinite(rs); ef = ex & fin; wf = (~ex) & fin
                if ef.sum() and wf.sum():
                    row["auc"] = 1.0 - float(stats.mannwhitneyu(rs[ef], rs[wf], alternative="less").statistic / (ef.sum() * wf.sum()))
                    row["spurious"] = float((rs[wf] <= np.median(rs[ef])).mean())
                best = np.argmin(np.where(fin, rs, np.inf), 1); t1r = float(ex[np.arange(n), best].mean()); ver = row["verified"][str(K)]
                row.update(t1r=t1r, t1r_over_verified=(t1r / ver if ver else None))
            # per-octile (rho, r) on the draws
            qs = np.quantile(rr, np.linspace(0, 1, 5)); qs[-1] += 1; fun = []
            for b in range(4):
                m = (rr >= qs[b]) & (rr < qs[b + 1])
                if m.sum() < 20: continue
                rho, r = fit_rho_r(fh[m]); fun.append(dict(bin=f"[{qs[b]:.0f},{qs[b+1]:.0f})", n=int(m.sum()), cold=float(cold[m].mean()), b1=float(ex[m, 0].mean()), rho=rho, r=r))
            row["funnel"] = fun
        uv = {k[len("uv_vote_k"):]: float(z[k][o].mean()) for k in z if k.startswith("uv_vote_k")}
        if uv: row["majority"] = uv
    return row

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default=None, help="comma list of campaign tags"); ap.add_argument("--no-field", action="store_true"); a = ap.parse_args(); t0 = time.time()
    from qhrrn2 import sudoku_extreme as SX
    d = SX.load_prepared(NPZ); Q, A, R = d["test_q"], d["test_a"], d["test_rating"]; G = (Q != 0).reshape(len(Q), -1).sum(1)
    only = set(a.only.split(",")) if a.only else None
    rows = []
    def add(kind, dirname, sub, path, meta):
        tag, arm, suf = parse_name(dirname)
        if only and tag not in only: return
        q = Path(path) / "records_all.npz"
        if not q.exists(): return
        try: z = dict(np.load(q, allow_pickle=True))
        except Exception as e: print(f"  {q}: unreadable {type(e).__name__}"); return
        if "idx" not in z or "cold_exact" not in z: return
        if z["idx"].max() >= 422786: print(f"  {q}: idx beyond the test set — skipped"); return
        s = jload(Path(path) / "summary_all.json") or {}
        r = dict(kind=kind, campaign=tag, arm=arm, suffix=suf, eval=sub, t=s.get("t_total"), k=s.get("k_init"), set=("full" if s.get("n") == 422786 else ("scan20k" if s.get("subsample") == 20000 else f"strat{s.get('stratified')}" if s.get("stratified") else f"sub{s.get('subsample')}" if s.get("subsample") else str(s.get("n")))),
                 cell=s.get("cell_kind", "rg"), ema=s.get("ema"), init=s.get("init", "void"), ckpt=s.get("ckpt"), fmo=s.get("final_map_only"), hard=s.get("hard_feedback"))
        r.update(analyze_records(z, G, R)); rows.append(r)
        print(f"  {kind:6s} {tag:9s} {arm:7s} {suf:9s} {sub:20s} n={r['n']:6d} cold {100*r['cold']:6.2f} g50 {'-' if r['g50'] is None else f'{r['g50']:.1f}'} yield {'-' if r['yield'] is None else f'{100*r['yield']:.1f}'} rho {'-' if 'rho' not in r else f'{100*r['rho']:.1f}'} spur {'-' if 'spurious' not in r else f'{100*r['spurious']:.1f}'}", flush=True)
    for dd in sorted(RUNS.iterdir()):
        n = dd.name
        if not dd.is_dir(): continue
        if n.startswith("sxeval_"):
            for sub in sorted(dd.iterdir()):
                if sub.is_dir(): add("full", n[7:], sub.name, sub, {})
        elif n.startswith(("sxscan_", "sxscreen_", "sxrider_", "sxd3demo_")):
            add(n.split("_")[0][2:], n.split("_", 1)[1], "", dd, {})
    if not a.no_field:
        F = RUNS / "field_ckpts" / "out"
        for dd in sorted(F.iterdir()) if F.exists() else []:
            q = dd / "records_all.npz"; s = jload(dd / "summary.json")
            if not q.exists() or not s: continue
            pr = s.get("proto", {})
            if pr.get("mode") not in ("cold", "draws"): continue          # "train" records index the TRAIN file, not the test arrays (C12 reads its summary directly)
            z = dict(np.load(q, allow_pickle=True))
            if "cold_exact" not in z and "mi_exact_k" not in z: continue
            zz = {"idx": z["idx"], "cold_exact": z["cold_exact"] if "cold_exact" in z else np.zeros(len(z["idx"]), bool)}
            for k in ("first_exact", "violations", "givens_kept", "mi_first_hit", "mi_exact_k", "mi_resid_k"):
                if k in z: zz[k] = z[k]
            if "givens_kept" in zz and zz["givens_kept"].dtype == bool: zz["givens_kept"] = np.where(zz["givens_kept"], G[z["idx"]], 0)
            r = dict(kind="field", campaign="public", arm=pr.get("model"), suffix=pr.get("mode"), eval=dd.name, t=pr.get("D"), k=pr.get("k"), set=pr.get("set"), cell="public", ema=pr.get("ema"), init=pr.get("init"), ckpt=pr.get("model"), noise=pr.get("noise"), dtype=pr.get("dtype"))
            if pr.get("mode") == "train": r["set"] = "train1k"
            try: r.update(analyze_records(zz, G, R))
            except Exception as e: print(f"  field {dd.name}: {type(e).__name__}: {e}"); continue
            rows.append(r); print(f"  field  {pr.get('model'):5s} {pr.get('mode'):6s} {dd.name:34s} n={r['n']:6d} cold {100*r['cold']:6.2f} g50 {'-' if r['g50'] is None else f'{r['g50']:.1f}'} yield {'-' if r['yield'] is None else f'{100*r['yield']:.1f}'} rho {'-' if 'rho' not in r else f'{100*r['rho']:.1f}'}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    (OUT.with_suffix(".json")).write_text(json.dumps(rows, indent=0, default=float))
    cols = ["kind", "campaign", "arm", "suffix", "eval", "set", "t", "k", "cell", "ema", "n", "cold", "g50", "width", "g50_in_range", "p_cold_r0", "yield", "fe_med", "fe_p90", "fe_step1", "viol_fail", "valid_wrong", "givens_kept", "rho", "b1", "r_med", "k50", "k90", "rescue_1", "rescue_any", "auc", "spurious", "t1r", "t1r_over_verified", "ckpt"]
    with open(OUT.with_suffix(".csv"), "w", newline="") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=cols, extrasaction="ignore"); w.writeheader()
        for r in rows: w.writerow({c: r.get(c) for c in cols})
    lines = [f"THE INSTRUMENT SUITE — record-level backfill over the banked corpus ({DATE}; descriptive; {len(rows)} rows; {time.time()-t0:.0f}s)",
             "kind | campaign | arm | suffix | eval | set | t | k | cell | ema | n | cold | g50 (width) [in 17-35] | P(cold|r0) | yield | fe med/p90 (step1) | viol on fail | valid-wrong | rho | b1 | r_i median | k50/k90 | rescue 1/any | AUC | spurious | t1r@k / verified@k | majority@k"]
    def fmt(x, p=1): return "-" if x is None else f"{100*x:.{p}f}"
    for r in rows:
        ver = r.get("verified", {}); K = r.get("k"); mj = r.get("majority", {})
        lines.append(f"{r['kind']} | {r['campaign']} | {r['arm']} | {r['suffix']} | {r['eval']} | {r['set']} | {r['t']} | {r.get('k')} | {r['cell']} | {r['ema']} | {r['n']} | {fmt(r['cold'],2)} | "
                     f"{'-' if r['g50'] is None else f'{r['g50']:.1f}'} ({'-' if r['width'] is None else f'{r['width']:.1f}'}) [{'yes' if r['g50_in_range'] else 'no'}] | {fmt(r['p_cold_r0'])} | {fmt(r['yield'])} | "
                     f"{'-' if r.get('fe_med') is None else f'{r['fe_med']:.0f}'}/{'-' if r.get('fe_p90') is None else f'{r['fe_p90']:.0f}'} ({fmt(r.get('fe_step1'))}) | {'-' if r.get('viol_fail') is None else f'{r['viol_fail']:.1f}'} | {fmt(r.get('valid_wrong'),3)} | "
                     f"{fmt(r.get('rho'))} | {fmt(r.get('b1'))} | {'-' if r.get('r_med') is None else f'{r['r_med']:.3f}'} | {r.get('k50')}/{r.get('k90')} | {fmt(r.get('rescue_1'))}/{fmt(r.get('rescue_any'))} | "
                     f"{'-' if r.get('auc') is None else f'{r['auc']:.3f}'} | {fmt(r.get('spurious'))} | {fmt(r.get('t1r'))} / {fmt(ver.get(str(K))) if K else '-'} | {fmt(mj.get(str(K))) if mj and K else '-'}")
    (OUT.with_suffix(".txt")).write_text("\n".join(lines) + "\n"); print(f"\n{len(rows)} rows -> {OUT}.{{txt,json,csv}} ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
