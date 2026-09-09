#!/usr/bin/env python3
"""CHAMPION NIGHT — PHYSICS PASS, written AT ANALYSIS TIME (2026-09-09; descriptive only, NO RULES).
The registered verdict authority is tools/analyze_champ.py (byte-untouched since the registration commit 7ada7c7, selftest
59/59). Derivative of analyze_finalA_physics.py (readers), PARAMETERIZED BY TAG (the class-tool lesson of 2026-09-07).

Reads the extracted champion corpus in runs/ (+ the banked Night A A3/A5/A7, the frontier's ported public checkpoints
eqr/cgar/trmpub, and sportC1's X0 as references on disk; every contrast is on IDENTICAL puzzles or a labeled intersection):
  A. ADMISSION (ckpt config + param count + argv), grids, markers, val-selection PROVENANCE (every vsel eval's ckpt path;
     the registered selection rule re-run offline: argmax val_t16_ema over the banked grids, EARLIEST tie, val_t16 second key).
  B. TRAJECTORIES — the 512-puzzle monitor every 2k (EMA / raw), the peak, the plateau (within 1 pp of the peak), the selected
     grid, the end; segment CE / train_exact / halting at the selected step and the end; measured pace; resumes.
  C. STABILITY — explosion census (vsel + final; t64 + t256) and the MID-TRAINING EXCURSIONS read on the monitor and the
     screens (cold vs a random-init draw b1: a cold-start excursion leaves b1 intact).
  D. EVAL TABLE — cold16 vsel (full) / final (50k) / alt raw (50k) / cold64 (100k) / D128 (20k) / D256 (5k) / first-exact /
     failure texture; the screens per grid (10k / 20k / 30k / 40k / vb: cold, b1, t1r, verified, majority); the 5k scans at
     k32 (registered) and k128 (the filler's matched-restart row): b1, t1r@k, verified@k, spurious, AUC; calibration at stalls.
  E. PAIRED McNemar on identical puzzles — the seed triple pairwise; every treatment vs C0 (matched seed 0); C0 vs A5 (the
     same recipe and seed under the old 64-puzzle monitor + 50k budget: the selection-instrument contrast); vs A7 / A3; vs the
     field's released weights (EqR / CGAR / alphaXiv) at D16 (full), D64 (100k ∩ full), D128 (20k), D256 (5k); vs X0; depth
     D16 -> D64 -> D128 per arm; vsel vs final; EMA vs raw; unions.
  F. FUNNEL (rho, r) per rating quartile from the k128 scans and the vb screens.
  G. THE DEPTH LADDER — exact_by_step_curve per row (D16 full / D64 100k / D128 20k / D256 5k) + regressions.
  H. THE HALTING HEAD (R10) on the D16 fulls — AUC(q_halt, exact), precision / recall, the emulated ACT halt.
  I. RATING BANDS at D64 on the identical 100k (C arms, A3/A5/A7, the field's weights by intersection) and at D16 on the full.
  J. THE COMMIT HEAD (C6) — per-step commit fraction and wrong-among-committed on solved vs unsolved puzzles; the head's mean
     as a per-puzzle correctness signal (AUC); reliability bins; against the softmax calibration at stalls (E6).
  K. MEMORIZATION / TRAJECTORY signatures.   L. COMPUTE column (analytic MACs per step, labeled inferred; per-puzzle by depth).

  PYTHONPATH=src .venv/bin/python tools/analyze_champ_physics.py [--tag champ] [--date 20260909] [--no-commit] [--curv]
      -> runs/analysis/<tag>_physics_<date>.txt (+ .json)
"""
from __future__ import annotations
import argparse, datetime as dt, json, math, os, pickle, re, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(os.environ.get("QHRRN_RUNS", ROOT / "runs"))
NPZ = Path(os.environ.get("QHRRN_NPZ", ROOT / "data/sudoku_extreme/sudoku_extreme_seed0.npz"))
TAG = "champ"
SEED_ARMS = ["C0", "C1", "C2"]; TREAT = ["C3", "C4", "C5", "C6"]; ARMS = SEED_ARMS + TREAT
DESC = {"C0": "champion recipe seed 0 (DEC-w384 + FPA k1 + RI s1, no digit aug, 30k)", "C1": "champion recipe seed 1 (EXTENDED to 50k)", "C2": "champion recipe seed 2 (30k)",
        "C3": "C0 + the ONLINE position orbit (50k)", "C4": "C0 + set attention over the fields (4h x dk32)", "C5": "C0 at w192 (EXTENDED to 50k)", "C6": "C0 + the calibrated commit head tau .9 (50k)"}
# references on disk (identical puzzle sets: the 20k and 5k subsamples are shared with the frontier run, the 100k with Night A; the fulls with everything)
REF_FINALA = {"A3": "DEC-w384 plain s0 (Night A, 64-monitor later-tie, 50k)", "A5": "DEC-w384 + FPA + RI s0 (Night A = C0's recipe under the old instrument)", "A7": "DEC-w384 plain s1 (Night A)"}
REF_FRONT = {"eqr": "EqR (EMA, released weights)", "cgar": "CGAR (released)", "trmpub": "alphaXiv TRM-MLP (released)"}
X0 = dict(full16=RUNS / "sxeval_psportC1X0/full_vsel_t16", full64=RUNS / "sxeval_psportC1X0/full_vsel_t64", scan=RUNS / "sxscan_psportC1X0", screen=RUNS / "sxscreen_psportC1X0_vb")
GMAC = {"w384": 70.0, "w192": 20.4, "trm": 15.2}   # analytic MACs per outer step per puzzle (the frontier report: DEC-w384 = 4.6x the TRM-MLP's 15.2; w192 = 1.34x), labeled inferred
RBANDS = [(0, 1), (1, 10), (10, 30), (30, 60), (60, 10**6)]
L = []; J = {}
def say(s=""): L.append(str(s)); print(s, flush=True)
def f(x, w=6, p=2): return " " * (w - 1) + "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:{w}.{p}f}"
def pp(x): return "   -  " if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{100*x:6.2f}"
def jload(p):
    p = Path(p); return json.loads(p.read_text()) if p.exists() else None
def pdir(a): return RUNS / f"pretrain{TAG}_{a}"
def evdir(a): return RUNS / f"sxeval_p{TAG}{a}"
def scan_dir(a): return RUNS / f"sxscan_p{TAG}{a}"
def k128_dir(a): return RUNS / f"filler_sxscan128_p{TAG}{a}"
def screen_dir(a, tag="vb"): return RUNS / f"sxscreen_p{TAG}{a}_{tag}"
def census_json(a, which): return jload(RUNS / f"sxcensus_p{TAG}{a}_{which}" / "census.json")
def calib_json(a): return jload(RUNS / f"sxcalib_p{TAG}{a}_vsel" / "calib.json")
def fdir(a, sub): return RUNS / f"sxeval_pfinalA{a}" / sub
def frdir(t, sub): return RUNS / f"sxeval_pfrontier{t}" / sub

# ---------- readers ----------
def count_params(tree):
    n = 0; st = [tree]
    while st:
        x = st.pop()
        if isinstance(x, dict): st.extend(x.values())
        elif isinstance(x, (list, tuple)): st.extend(x)
        else: n += int(np.prod(np.shape(x)))
    return n
CFG_KEYS = ("cell_kind", "dec_width", "dec_coupling", "dec_attn_heads", "dec_attn_dk", "dec_commit", "dec_commit_tau", "dec_commit_w", "trm_hidden", "trm_layers", "trm_h_cycles", "trm_l_cycles",
            "trm_lambda", "trm_beta", "trm_ri_sigma", "sudoku_digit_aug", "sudoku_orbit_online", "fpa_k", "fpa_eps", "fpa_frac", "seed", "remat", "loss_kind", "T", "sudoku_layout")
ARGV_KEYS = ("seed", "steps", "batch", "lr", "lr_end", "warmup", "wd", "beta2", "ema", "sudoku_aug", "sudoku_orbit_online", "sot", "act", "cell", "dec_width", "dec_coupling", "dec_commit", "fpa_k", "trm_ri_sigma", "remat", "grid_every", "monitor_every", "ckpt_every", "sudoku_extreme")
def admission(arm):
    d = pdir(arm); out = {"dir": d.exists()}
    if not d.exists(): return out
    ck = d / "ckpt_latest.pkl"
    if ck.exists():
        try:
            c = pickle.load(open(ck, "rb")); cfg = c.get("config", {})
            out["step"] = int(c["step"]); out["cfg"] = {k: cfg.get(k) for k in CFG_KEYS if k in cfg}
            out["n_params"] = count_params(c["state"]["model"]); out["has_ema"] = c.get("state_ema") is not None
        except Exception as e: out["ckpt_err"] = f"{type(e).__name__}: {e}"
    cj = jload(d / "config.json") or {}
    a = cj.get("argv", {}); out["argv"] = {k: a.get(k) for k in ARGV_KEYS if k in a} if isinstance(a, dict) else str(a)[:300]; out["git"] = cj.get("git"); out["n_params_bulk"] = cj.get("n_params_bulk")
    for nm in ("STOPPED.txt", "NAN_ABORT.txt", "resumes.txt", "val_best.txt", "RETRY_REMAT.txt", "EXTENDED.txt"):
        p = d / nm; out[nm] = p.read_text().strip().replace("\n", "; ") if p.exists() else None
    out["grid_steps"] = sorted(int(re.search(r"ckpt_(\d+)\.pkl$", p.name).group(1)) for p in d.glob("ckpt_[0-9]*.pkl"))
    return out
def load_metrics(path):
    tr, mon = {}, {}
    if not Path(path).exists(): return tr, mon
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line: continue
        try: r = json.loads(line)
        except Exception: continue
        if "monitor" in r: mon[int(r["monitor"]["step"])] = r["monitor"]
        elif "step" in r and "loss" in r: tr[int(r["step"])] = r
    return dict(sorted(tr.items())), dict(sorted(mon.items()))
def finite(v):
    try: return v is not None and math.isfinite(float(v))
    except Exception: return False
def near(dct, target, window=1500):
    ks = [s for s in dct if finite(dct[s]) and abs(s - target) <= window]
    if not ks: return None
    return dct[min(ks, key=lambda s: abs(s - target))]
def recs(p):
    p = Path(p); q = p / "records_all.npz"
    if not q.exists(): return None
    z = dict(np.load(q, allow_pickle=True)); order = np.argsort(z["idx"], kind="stable")
    return {k: (v[order] if hasattr(v, "shape") and v.shape and v.shape[0] == len(order) else v) for k, v in z.items()}
def mcnemar(a, b):
    oa = int(np.sum(a & ~b)); ob = int(np.sum(~a & b)); n = oa + ob
    if n == 0: return oa, ob, 1.0
    from scipy import stats
    return oa, ob, float(min(1.0, 2 * stats.binom.cdf(min(oa, ob), n, 0.5)))
def paired(za, zb, key="cold_exact"):
    if za is None or zb is None: return None
    ia, ib = za["idx"], zb["idx"]
    if len(ia) == len(ib) and np.array_equal(ia, ib): return za[key].astype(bool), zb[key].astype(bool), len(ia), "identical idx"
    common, ka, kb = np.intersect1d(ia, ib, return_indices=True)
    if len(common) == 0: return None
    return za[key].astype(bool)[ka], zb[key].astype(bool)[kb], len(common), f"INTERSECTION n={len(common)} (labeled)"
def pair_line(name, za, zb, extra=""):
    r = paired(za, zb)
    if r is None: say(f"  {name:44s}: no paired data"); return None
    a, b, n, note = r; oa, ob, p = mcnemar(a, b)
    say(f"  {name:44s}: {pp(a.mean())} vs {pp(b.mean())} (delta {100*(a.mean()-b.mean()):+6.2f} pp) | only-A {oa:7d} / only-B {ob:7d} | p {p:.1e} | n {n} {note} {extra}")
    return dict(a=float(a.mean()), b=float(b.mean()), delta=float(a.mean() - b.mean()), only_a=oa, only_b=ob, p=p, n=n, note=note)
def bits(z, T):
    if z is None or "exact_by_step" not in z: return None
    e = z["exact_by_step"]
    return np.unpackbits(e.astype(np.uint8), axis=1, bitorder="little")[:, :T].astype(bool)
def t1r_bits(z, k):
    ex = z["mi_exact_k"].astype(bool); re_ = z["mi_resid_k"].astype(np.float64); k = min(k, ex.shape[1]); fin = np.isfinite(re_[:, :k])
    best = np.argmin(np.where(fin, re_[:, :k], np.inf), axis=1); return ex[np.arange(len(ex)), best]
def selector(z, k):
    if z is None or "mi_exact_k" not in z or "mi_resid_k" not in z: return None
    from scipy import stats
    k = min(k, z["mi_exact_k"].shape[1]); ex = z["mi_exact_k"][:, :k].astype(bool); rs = z["mi_resid_k"][:, :k].astype(np.float64); fin = np.isfinite(rs); ef = ex & fin; wf = (~ex) & fin
    auc = 1.0 - float(stats.mannwhitneyu(rs[ef], rs[wf], alternative="less").statistic / (ef.sum() * wf.sum())) if ef.sum() and wf.sum() else None
    thr = np.median(rs[ef]) if ef.any() else np.nan; spur = float((rs[wf] <= thr).mean()) if wf.any() else None
    cold = z["cold_exact"].astype(bool); fh = z["mi_first_hit"]; ver = float((cold | ((fh >= 0) & (fh < k))).mean()); t1r = float(t1r_bits(z, k).mean())
    ri = ex.mean(1)
    return dict(k=k, auc=auc, spurious=spur, t1r=t1r, verified=ver, ratio=(t1r / ver if ver else None), b1=float(ex[:, 0].mean()), rho=float((cold | ex.any(1)).mean()), cold=float(cold.mean()),
                r_med=float(np.median(ri[cold | ex.any(1)])) if (cold | ex.any(1)).any() else None, frac_r_gt_half=float((ri > .5).mean()), n_wrong_draws=int(wf.sum()), n_spurious=int((rs[wf] <= thr).sum()) if wf.any() else 0)
def fit_rho_r(fh, k_fit=64):
    fh = np.asarray(fh); hit = (fh >= 0) & (fh < k_fit); t = fh[hit]; n_c = int(np.sum(~hit)); best = (-np.inf, None, None)
    for rho in np.linspace(0.02, 1.0, 50):
        for r in np.geomspace(1e-3, 0.9, 60):
            ll = np.sum(np.log(rho) + t * np.log1p(-r) + np.log(r)) if len(t) else 0.0
            ll += n_c * np.log(max(1 - rho + rho * (1 - r) ** k_fit, 1e-300))
            if ll > best[0]: best = (ll, rho, r)
    return best[1], best[2]
def funnel_rows(z, k, nq=4):
    rat = z["rating"]; fh = z["mi_first_hit"]; cold = z["cold_exact"].astype(bool); b1 = z["mi_exact_k"][:, 0].astype(bool) if "mi_exact_k" in z else None
    qs = np.quantile(rat, np.linspace(0, 1, nq + 1)); qs[-1] += 1; rows = []
    for b in range(nq):
        m = (rat >= qs[b]) & (rat < qs[b + 1])
        if m.sum() == 0: continue
        rho, r = fit_rho_r(fh[m]); act = float(np.mean((fh[m] >= 0) & (fh[m] < k)))
        rows.append(dict(bin=f"[{qs[b]:.0f},{qs[b+1]:.0f})", n=int(m.sum()), cold=float(cold[m].mean()), b1=(float(b1[m].mean()) if b1 is not None else None), hit=act, rho=rho, r=r))
    return rows
def first_exact_stats(z):
    if z is None or "first_exact" not in z: return None
    fe = z["first_exact"]; s = fe[fe >= 0]
    return dict(n_solved=int(len(s)), med=float(np.median(s)) + 1 if len(s) else None, p90=float(np.percentile(s, 90)) + 1 if len(s) else None, frac1=float((s == 0).mean()) if len(s) else None, frac2=float((s <= 1).mean()) if len(s) else None)
def halting(z, T=16):
    if z is None or "q_by_step" not in z or z["q_by_step"].ndim != 3: return None
    from scipy import stats
    q = z["q_by_step"].astype(np.float32); cold = z["cold_exact"].astype(bool); ql, qc = q[:, -1, 0], q[:, -1, 1]; halt = ql > qc; n = len(q)
    def auc(score, y):   # P(score_pos > score_neg): a HIGH q_halt predicts exact (the residual selector's orientation is the opposite)
        if y.all() or (~y).all(): return None
        return float(stats.mannwhitneyu(score[y], score[~y], alternative="greater").statistic / (y.sum() * (~y).sum()))
    prec = float((halt & cold).sum() / max(halt.sum(), 1)); rec = float((halt & cold).sum() / max(cold.sum(), 1))
    ex = bits(z, T); h_any = (q[:, :, 0] > q[:, :, 1]); first = np.where(h_any.any(1), h_any.argmax(1), T - 1)
    exact_at_first = float(ex[np.arange(n), first].mean()) if ex is not None else None
    return dict(n=n, cold=float(cold.mean()), auc=auc(ql, cold), margin_auc=auc(ql - qc, cold), halt_frac=float(halt.mean()), precision=prec, recall=rec, first_halt_mean=float(first.mean() + 1), exact_at_first_halt=exact_at_first,
                delta=(exact_at_first - float(cold.mean())) if exact_at_first is not None else None)

def main():
    global TAG
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="champ"); ap.add_argument("--date", default=time.strftime("%Y%m%d")); ap.add_argument("--no-commit", action="store_true"); ap.add_argument("--curv", action="store_true")
    a = ap.parse_args(); TAG = a.tag; t0 = time.time()
    OUT = RUNS / "analysis" / f"{TAG}_physics_{a.date}.txt"; OUTJ = OUT.with_suffix(".json")
    say("=" * 130); say(f"CHAMPION NIGHT — PHYSICS PASS ({a.date}; analysis-time, descriptive, no rules; verdict authority = tools/analyze_{TAG}.py; tag {TAG})"); say("=" * 130)
    present = [x for x in ARMS if pdir(x).exists()]
    # ---------- A ----------
    say("\n== A. ADMISSION (ckpt_latest config + parameter count; argv from config.json; markers; grids) ==")
    ADM = {}
    for x in ARMS:
        o = admission(x); ADM[x] = o
        if not o["dir"]: say(f"  {x}: ABSENT ({DESC[x]})"); continue
        cfg = o.get("cfg", {}); av = o.get("argv", {})
        say(f"  {x} [{DESC[x]}]: step {o.get('step')} | params {o.get('n_params')} (config.json {o.get('n_params_bulk')}) | ema {o.get('has_ema')} | git {o.get('git')}")
        say(f"     cfg: " + ", ".join(f"{k}={v}" for k, v in cfg.items()))
        if isinstance(av, dict): say(f"     argv: " + ", ".join(f"{k}={v}" for k, v in av.items()))
        gs = o.get("grid_steps", []); say(f"     grids: {len(gs)} banked [{gs[0] if gs else '-'}..{gs[-1] if gs else '-'}] | val_best {o.get('val_best.txt')!r} | EXTENDED {o.get('EXTENDED.txt')!r} | resumes {o.get('resumes.txt')!r} | REMAT {o.get('RETRY_REMAT.txt')!r}" + (f" | STOPPED {o['STOPPED.txt']!r}" if o.get("STOPPED.txt") else ""))
    J["admission"] = {x: {k: v for k, v in o.items() if k != "grid_steps"} for x, o in ADM.items()}
    say("\n  PROVENANCE (one ckpt path per (arm, vsel)) + the registered selection rule re-run offline (argmax val_t16_ema over banked grids; EARLIEST tie; val_t16 second key):")
    PROV = {}
    for x in present:
        paths = {}
        for nm, p in (("full_vsel16", evdir(x) / "full_vsel_t16"), ("alt", evdir(x) / "full_vsel_t16_alt"), ("full_vsel64", evdir(x) / "full_vsel_t64"), ("d128", evdir(x) / "sub20k_t128"), ("d256", evdir(x) / "sub5k_t256"),
                      ("commit", evdir(x) / "sub20k_t16_commit"), ("scan", scan_dir(x)), ("k128", k128_dir(x)), ("screen_vb", screen_dir(x))):
            s = jload(p / "summary_all.json")
            if s and s.get("ckpt"): paths[nm] = s["ckpt"]
        for nm, p in (("census_vsel", RUNS / f"sxcensus_p{TAG}{x}_vsel" / "census.json"), ("calib", RUNS / f"sxcalib_p{TAG}{x}_vsel" / "calib.json")):
            s = jload(p)
            if s and s.get("ckpt"): paths[nm] = s["ckpt"]
        tr, mon = load_metrics(pdir(x) / "metrics.jsonl"); banked = set(ADM[x].get("grid_steps", []))
        cand = [(float(mon[s]["val_t16_ema"]), float(mon[s].get("val_t16", 0.0)), s) for s in mon if s in banked and finite(mon[s].get("val_t16_ema"))]
        best = max(cand, key=lambda c: (c[0], c[1], -c[2]))[2] if cand else None; later = max(cand, key=lambda c: (c[0], c[2]))[2] if cand else None
        top = max(c[0] for c in cand) if cand else None; ties = sorted(c[2] for c in cand if c[0] == top) if cand else []
        uniq = sorted(set(paths.values())); m = re.search(r"ckpt_(\d+)\.pkl", uniq[0]) if uniq else None; vstep = int(m.group(1)) if m else None
        PROV[x] = dict(paths=paths, offline_best=best, later_tie_best=later, ties=ties, top=top, vsel_step=vstep, consistent=len(uniq) == 1, chain_val_best=ADM[x].get("val_best.txt"))
        say(f"  {x}: vsel evals on {len(uniq)} path(s) {uniq} | chain val_best {ADM[x].get('val_best.txt')!r} | offline (earliest tie, 2nd key) = {best} -> {'AGREES' if best == vstep else 'DIFFERS'} | monitor max {pp(top)} tied at {ties} | the OLD later-tie rule would pick {later}")
    J["provenance"] = PROV
    # ---------- B ----------
    say("\n== B. TRAJECTORIES (train rows; the 512-puzzle monitor every 2k: EMA / raw; pace from timestamps) ==")
    TR = {}
    for x in present:
        tr, mon = load_metrics(pdir(x) / "metrics.jsonl"); steps = list(tr)
        val = {s: float(mon[s]["val_t16_ema"]) for s in mon if finite(mon[s].get("val_t16_ema"))}; raw = {s: float(mon[s]["val_t16"]) for s in mon if finite(mon[s].get("val_t16"))}
        CE = {s: tr[s].get("ce_in") for s in steps}; TE = {s: tr[s].get("train_exact") for s in steps}; HF = {s: tr[s].get("halt_frac") for s in steps}; MS = {s: tr[s].get("mean_steps") for s in steps}
        vpk = max(val, key=val.get) if val else None; peak = val[vpk] if val else None
        plateau = [s for s in val if peak is not None and val[s] >= peak - .01]
        ts = []
        for s in steps:
            t = tr[s].get("t")
            if t:
                try: ts.append((s, dt.datetime.fromisoformat(t)))
                except Exception: pass
        pace = None
        if len(ts) >= 4:
            i0 = int(len(ts) * .6); (s0, t0_), (s1, t1_) = ts[i0], ts[-1]; dsec = (t1_ - t0_).total_seconds(); pace = (s1 - s0) / dsec if dsec > 0 else None
        vsel = PROV[x]["vsel_step"]; end = max(val) if val else None
        TR[x] = dict(val=val, raw=raw, peak=(vpk, peak), plateau=(min(plateau), max(plateau)) if plateau else None, vsel=vsel, val_at_vsel=val.get(vsel), raw_at_vsel=raw.get(vsel), end=(end, val.get(end)), decline=(peak - val[end]) if (peak is not None and end in val) else None,
                     ce_at_vsel=near(CE, vsel) if vsel else None, te_at_vsel=near(TE, vsel) if vsel else None, ce_end=float(np.mean([CE[s] for s in steps[-5:] if finite(CE[s])])) if steps else None, te_end=float(np.mean([TE[s] for s in steps[-5:] if finite(TE[s])])) if steps else None,
                     hf_end=near(HF, steps[-1]) if steps else None, ms_end=near(MS, steps[-1]) if steps else None, ce_min=(min((v for v in CE.values() if finite(v)), default=None)), pace=pace, last=steps[-1] if steps else None, n_rows=len(steps),
                     ce_at={k: near(CE, k) for k in (5000, 10000, 20000, 30000, 40000, 50000)}, te_at={k: near(TE, k) for k in (5000, 10000, 20000, 30000, 40000, 50000)}, hf_at={k: near(HF, k) for k in (10000, 30000, 50000)})
    say("  arm | rows | last | monitor peak (step) | plateau (>= peak-1pp) | selected: step, EMA/raw | end: step, EMA | decline peak->end | CE at vsel / end / min | train_exact at vsel / end | halt_frac end | mean_steps end | pace it/s")
    for x in present:
        t = TR[x]; pl = t["plateau"]
        say(f"  {x} | {t['n_rows']} | {t['last']} | {pp(t['peak'][1])} ({t['peak'][0]}) | {pl[0] if pl else '-'}..{pl[1] if pl else '-'} | {t['vsel']}: {pp(t['val_at_vsel'])}/{pp(t['raw_at_vsel'])} | {t['end'][0]}: {pp(t['end'][1])} | {f(t['decline'],6,3)} | {f(t['ce_at_vsel'],5,3)} / {f(t['ce_end'],5,3)} / {f(t['ce_min'],5,3)} | {f(t['te_at_vsel'],5,3)} / {f(t['te_end'],5,3)} | {f(t['hf_end'],5,3)} | {f(t['ms_end'],5,2)} | {f(t['pace'],5,2)}")
    say("  monitor val_t16_ema every 2k (the selection key), '*' = the selected grid:")
    for x in present:
        t = TR[x]; say(f"    {x}: " + " ".join(f"{s//1000}k={100*v:.1f}{'*' if s == t['vsel'] else ''}" for s, v in sorted(t["val"].items())))
    say("  monitor val_t16 RAW every 2k (the second key):")
    for x in present:
        t = TR[x]; say(f"    {x}: " + " ".join(f"{s//1000}k={100*v:.1f}" for s, v in sorted(t["raw"].items())))
    say("  segment CE / train_exact every 10k: " + " | ".join(f"{x}: " + " ".join(f"{k//1000}k={f(TR[x]['ce_at'][k],4,2).strip()}/{f(TR[x]['te_at'][k],4,2).strip()}" for k in (10000, 20000, 30000, 40000, 50000) if TR[x]['ce_at'][k] is not None) for x in present))
    J["traj"] = {x: {k: v for k, v in t.items() if k not in ("val", "raw")} for x, t in TR.items()}; J["traj_val"] = {x: {str(s): v for s, v in t["val"].items()} for x, t in TR.items()}; J["traj_raw"] = {x: {str(s): v for s, v in t["raw"].items()} for x, t in TR.items()}
    # ---------- C ----------
    say("\n== C. STABILITY (explosion census on the vsel and final grids) and the MID-TRAINING EXCURSIONS (monitor vs the screens' cold and random-init b1) ==")
    CEN = {}
    for x in present:
        row = {}
        for which in ("vsel", "final"):
            c = census_json(x, which)
            if c: row[which] = {int(r["t"]): (r.get("exploded_frac"), r.get("n")) for r in c["rows"]}
        CEN[x] = row
        say(f"  {x}: " + " | ".join(f"{which}: " + ", ".join(f"t{t} {pp(v[0])} (n {v[1]})" for t, v in sorted(rr.items())) for which, rr in row.items()))
    J["census"] = {x: {w: {str(t): v for t, v in rr.items()} for w, rr in row.items()} for x, row in CEN.items()}
    say("  excursions: every grid where the monitor (EMA or raw) sits > 10 pp below the arm's running maximum; the screen at that grid (strat-512 k256) when it exists: cold vs b1 (a cold-start excursion leaves b1 intact)")
    EXC = {}
    for x in present:
        t = TR[x]; runmax = -1; rows = []
        for s in sorted(t["val"]):
            runmax = max(runmax, t["val"][s]); v, r = t["val"][s], t["raw"].get(s)
            if (runmax - v > .10) or (r is not None and runmax - r > .10):
                sc = jload(screen_dir(x, f"s{s:06d}") / "summary_all.json")
                rows.append(dict(step=s, ema=v, raw=r, runmax=runmax, screen_cold=(sc or {}).get("exact_acc"), screen_b1=(sc or {}).get("b1_exact"), screen_verified=((sc or {}).get("vote_at_k") or {}).get("256")))
        EXC[x] = rows
        if rows: say(f"  {x}: " + " | ".join(f"{r['step']//1000}k EMA {100*r['ema']:.1f} raw {f(100*r['raw'] if r['raw'] is not None else None,5,1).strip()} (run-max {100*r['runmax']:.1f})" + (f" screen cold {100*r['screen_cold']:.1f} b1 {100*r['screen_b1']:.1f} verified@256 {100*r['screen_verified']:.1f}" if r['screen_cold'] is not None else "") for r in rows))
        else: say(f"  {x}: none")
    J["excursions"] = EXC
    # ---------- D ----------
    say("\n== D. EVAL TABLE (headline = EMA at D16 on the vsel grid, full test; final / alt on 50k; D64 on 100k; D128 on 20k; D256 on 5k; labeled) ==")
    say("  arm | cold16 (n) | final16 (n) | alt raw (n) | cold64 (n) | D128 (n) | D256 (n) | first_exact D16 med/p90 (step-1 frac) | D64 med/p90 | valid-wrong | mean viol | givens kept")
    EV = {}
    def summ(p): return jload(Path(p) / "summary_all.json")
    for x in present:
        S = {nm: summ(evdir(x) / nm) for nm in ("full_vsel_t16", "full_final_t16", "full_vsel_t16_alt", "full_vsel_t64", "sub20k_t128", "sub5k_t256")}
        z16 = recs(evdir(x) / "full_vsel_t16"); z64 = recs(evdir(x) / "full_vsel_t64"); fe16, fe64 = first_exact_stats(z16), first_exact_stats(z64)
        g = lambda nm, k: (S[nm] or {}).get(k)
        EV[x] = dict(cold16=g("full_vsel_t16", "exact_acc"), n16=g("full_vsel_t16", "n"), final=g("full_final_t16", "exact_acc"), nfin=g("full_final_t16", "n"), alt=g("full_vsel_t16_alt", "exact_acc"), nalt=g("full_vsel_t16_alt", "n"),
                     cold64=g("full_vsel_t64", "exact_acc"), n64=g("full_vsel_t64", "n"), d128=g("sub20k_t128", "exact_acc"), n128=g("sub20k_t128", "n"), d256=g("sub5k_t256", "exact_acc"), n256=g("sub5k_t256", "n"),
                     fe16=fe16, fe64=fe64, valid_wrong=g("full_vsel_t16", "valid_wrong_frac"), viol=g("full_vsel_t16", "mean_violations"), gk=g("full_vsel_t16", "givens_kept_frac"), by_bin16=g("full_vsel_t16", "by_rating_bin"), by_bin64=g("full_vsel_t64", "by_rating_bin"),
                     curves={nm: (S[nm] or {}).get("exact_by_step_curve") for nm in ("full_vsel_t16", "full_vsel_t64", "sub20k_t128", "sub5k_t256")}, regr={nm: (S[nm] or {}).get("depth_regressions") for nm in ("full_vsel_t16", "full_vsel_t64", "sub20k_t128", "sub5k_t256")})
        e = EV[x]
        say(f"  {x} | {pp(e['cold16'])} ({e['n16']}) | {pp(e['final'])} ({e['nfin']}) | {pp(e['alt'])} ({e['nalt']}) | {pp(e['cold64'])} ({e['n64']}) | {pp(e['d128'])} ({e['n128']}) | {pp(e['d256'])} ({e['n256']}) | "
            f"{f(fe16['med'],3,0) if fe16 else '  -'}/{f(fe16['p90'],3,0) if fe16 else '  -'} ({pp(fe16['frac1']) if fe16 else '   -  '}) | {f(fe64['med'],3,0) if fe64 else '  -'}/{f(fe64['p90'],3,0) if fe64 else '  -'} | {f(e['valid_wrong'],7,5)} | {f(e['viol'],5,2)} | {f(e['gk'],6,4)}")
    # references
    REF = {}
    for nm, d16, d64, d128, d256 in ([(k, fdir(k, "full_vsel_t16"), fdir(k, "full_vsel_t64"), fdir(k, "sub20k_t128"), None) for k in REF_FINALA] + [(k, frdir(k, "full_vsel_t16"), frdir(k, "full_vsel_t64"), frdir(k, "sub20k_t128"), frdir(k, "sub5k_t256")) for k in REF_FRONT] + [("X0", X0["full16"], X0["full64"], None, None)]):
        s16, s64, s128, s256 = summ(d16), summ(d64), (summ(d128) if d128 else None), (summ(d256) if d256 else None)
        if not s16: continue
        fe16 = first_exact_stats(recs(d16)); fe64 = first_exact_stats(recs(d64))
        REF[nm] = dict(cold16=s16.get("exact_acc"), n16=s16.get("n"), cold64=(s64 or {}).get("exact_acc"), n64=(s64 or {}).get("n"), d128=(s128 or {}).get("exact_acc"), n128=(s128 or {}).get("n"), d256=(s256 or {}).get("exact_acc"), n256=(s256 or {}).get("n"), fe16=fe16, fe64=fe64,
                       curves={"full_vsel_t16": s16.get("exact_by_step_curve"), "full_vsel_t64": (s64 or {}).get("exact_by_step_curve"), "sub20k_t128": (s128 or {}).get("exact_by_step_curve"), "sub5k_t256": (s256 or {}).get("exact_by_step_curve")},
                       regr={"full_vsel_t16": s16.get("depth_regressions"), "full_vsel_t64": (s64 or {}).get("depth_regressions"), "sub20k_t128": (s128 or {}).get("depth_regressions"), "sub5k_t256": (s256 or {}).get("depth_regressions")})
        e = REF[nm]
        say(f"  {nm:6s} ref | {pp(e['cold16'])} ({e['n16']}) | - | - | {pp(e['cold64'])} ({e['n64']}) | {pp(e['d128'])} ({e['n128']}) | {pp(e['d256'])} ({e['n256']}) | {f(fe16['med'],3,0) if fe16 else '  -'}/{f(fe16['p90'],3,0) if fe16 else '  -'} ({pp(fe16['frac1']) if fe16 else '   -  '}) | {f(fe64['med'],3,0) if fe64 else '  -'}/{f(fe64['p90'],3,0) if fe64 else '  -'} | | |")
    J["eval"] = {x: {k: v for k, v in e.items() if k not in ("curves",)} for x, e in EV.items()}; J["refs"] = {x: {k: v for k, v in e.items() if k not in ("curves",)} for x, e in REF.items()}
    say("\n  SCREENS (strat-512, k256, D16, EMA; every banked screen grid): arm@grid | cold | b1 | t1r@256 | verified@256 | majority@256 | AUC | spurious | rho")
    SCR = {}
    for x in present:
        for tag in ("s010000", "s020000", "s030000", "s040000", "vb"):
            p = screen_dir(x, tag); s = summ(p)
            if not s: continue
            z = recs(p); sel = selector(z, 256) if z is not None else None
            SCR[f"{x}@{tag}"] = dict(cold=s.get("exact_acc"), b1=s.get("b1_exact"), t1r=(s.get("t1r_at_k") or {}).get("256"), v=(s.get("vote_at_k") or {}).get("256"), mj=(s.get("majority_vote_at_k") or {}).get("256"), sel=sel, ckpt=s.get("ckpt"))
            r = SCR[f"{x}@{tag}"]
            say(f"  {x}@{tag:7s} ({Path(r['ckpt']).name:16s}) | {pp(r['cold'])} | {pp(r['b1'])} | {pp(r['t1r'])} | {pp(r['v'])} | {pp(r['mj'])} | {f(sel['auc'] if sel else None,5,3)} | {pp(sel['spurious'] if sel else None)} | {pp(sel['rho'] if sel else None)}")
    J["screens"] = SCR
    say("\n  SCANS on the identical 5k subsample (t64, EMA): the registered k32 scan and the filler's k128 scan; the frontier's k128 scans are on the 20k (labeled): arm | k | cold | b1 | t1r@k | verified@k | ratio | AUC | spurious (n wrong draws / n spurious) | rho | median r_i | r_i>.5")
    SCN = {}
    for x in present:
        for lab, p in (("k32", scan_dir(x)), ("k128", k128_dir(x))):
            s = summ(p); z = recs(p)
            if not s or z is None: continue
            k = int(s.get("k_init")); sel = selector(z, k); SCN[f"{x}@{lab}"] = dict(sel=sel, n=s.get("n"), curve_v={kk: v for kk, v in (s.get("vote_at_k") or {}).items()}, curve_t={kk: v for kk, v in (s.get("t1r_at_k") or {}).items()}, wall=s.get("wall_s"))
            say(f"  {x}@{lab:4s} | {k:3d} | {pp(sel['cold'])} | {pp(sel['b1'])} | {pp(sel['t1r'])} | {pp(sel['verified'])} | {f(sel['ratio'],6,4)} | {f(sel['auc'],5,3)} | {pp(sel['spurious'])} ({sel['n_wrong_draws']}/{sel['n_spurious']}) | {pp(sel['rho'])} | {f(sel['r_med'],5,3)} | {pp(sel['frac_r_gt_half'])}")
    for x in REF_FINALA:
        z = recs(RUNS / f"sxscan_pfinalA{x}"); s = summ(RUNS / f"sxscan_pfinalA{x}")
        if z is None or not s: continue
        sel = selector(z, int(s.get("k_init"))); SCN[f"{x}@k32"] = dict(sel=sel, n=s.get("n"))
        say(f"  {x}@k32  ref | {sel['k']:3d} | {pp(sel['cold'])} | {pp(sel['b1'])} | {pp(sel['t1r'])} | {pp(sel['verified'])} | {f(sel['ratio'],6,4)} | {f(sel['auc'],5,3)} | {pp(sel['spurious'])} ({sel['n_wrong_draws']}/{sel['n_spurious']}) | {pp(sel['rho'])} | {f(sel['r_med'],5,3)} | {pp(sel['frac_r_gt_half'])}")
    for x in REF_FRONT:
        z = recs(RUNS / f"sxscan_pfrontier{x}"); s = summ(RUNS / f"sxscan_pfrontier{x}")
        if z is None or not s: continue
        sel = selector(z, int(s.get("k_init"))); SCN[f"{x}@k128(20k)"] = dict(sel=sel, n=s.get("n"))
        say(f"  {x:6s}@k128 20k ref | {sel['k']:3d} | {pp(sel['cold'])} | {pp(sel['b1'])} | {pp(sel['t1r'])} | {pp(sel['verified'])} | {f(sel['ratio'],6,4)} | {f(sel['auc'],5,3)} | {pp(sel['spurious'])} ({sel['n_wrong_draws']}/{sel['n_spurious']}) | {pp(sel['rho'])} | {f(sel['r_med'],5,3)} | {pp(sel['frac_r_gt_half'])}")
    say("  k-curves (5k): verified@k and t1r@k at k = 1..128 where recorded:")
    for key, r in SCN.items():
        if r.get("curve_v"): say(f"    {key:9s} verified: " + " ".join(f"{kk}:{100*v:.2f}" for kk, v in r["curve_v"].items()) + "  | t1r: " + " ".join(f"{kk}:{100*v:.2f}" for kk, v in r["curve_t"].items()))
    J["scans"] = SCN
    say("\n  CALIBRATION AT STALLS (strat-512 t64 on the vsel grid; softmax confidence): arm | cold | n stalled | top-5 correct | mean conf | gap | entropy step1 | entropy stalled | conf-wrong frac stalled")
    CAL = {}
    for x in present:
        c = calib_json(x)
        if not c: continue
        CAL[x] = c; gap = (c["mean_conf_stalled"] - c["topk_correct_stalled"]) if c.get("mean_conf_stalled") is not None and c.get("topk_correct_stalled") is not None else None
        say(f"  {x} | {pp(c.get('cold'))} | {c.get('n_stalled'):3d} | {pp(c.get('topk_correct_stalled'))} | {f(c.get('mean_conf_stalled'),5,3)} | {f(gap,6,3)} | {f(c.get('entropy_step1'),5,3)} | {f(c.get('entropy_t_stalled'),5,3)} | {pp(c.get('conf_wrong_frac_stalled'))}")
    J["calib"] = CAL
    # ---------- E ----------
    say("\n== E. PAIRED McNemar on identical puzzles (cold, EMA; A vs B; exact two-sided binomial) ==")
    PR = {}
    F16 = {x: recs(evdir(x) / "full_vsel_t16") for x in present}; F64 = {x: recs(evdir(x) / "full_vsel_t64") for x in present}; F128 = {x: recs(evdir(x) / "sub20k_t128") for x in present}; F256 = {x: recs(evdir(x) / "sub5k_t256") for x in present}
    for k in REF_FINALA: F16[k] = recs(fdir(k, "full_vsel_t16")); F64[k] = recs(fdir(k, "full_vsel_t64")); F128[k] = recs(fdir(k, "sub20k_t128"))
    for k in REF_FRONT: F16[k] = recs(frdir(k, "full_vsel_t16")); F64[k] = recs(frdir(k, "full_vsel_t64")); F128[k] = recs(frdir(k, "sub20k_t128")); F256[k] = recs(frdir(k, "sub5k_t256"))
    F16["X0"] = recs(X0["full16"]); F64["X0"] = recs(X0["full64"])
    def block(title, F, pairs):
        say(f"  {title}")
        for nm, ka, kb in pairs:
            if F.get(ka) is not None and F.get(kb) is not None: PR[f"{title.split(' ')[0]} {nm}"] = pair_line(nm, F[ka], F[kb])
    triple = [("C1 vs C0 (seeds)", "C1", "C0"), ("C2 vs C0 (seeds)", "C2", "C0"), ("C2 vs C1 (seeds)", "C2", "C1")]
    treat = [("C3 vs C0 (online orbit, matched seed 0)", "C3", "C0"), ("C4 vs C0 (set attention, matched seed 0)", "C4", "C0"), ("C5 vs C0 (w192, matched seed 0)", "C5", "C0"), ("C6 vs C0 (commit head, matched seed 0)", "C6", "C0"),
             ("C5 vs C4 (the two lifts)", "C5", "C4"), ("C5 vs C2 (labeled best vs champion by rule)", "C5", "C2"), ("C4 vs C2", "C4", "C2")]
    nightA = [("C0 vs A5 (same recipe + seed; the 512-monitor/earliest-tie/30k vs the 64-monitor/50k)", "C0", "A5"), ("C2 vs A7 (best plain vs best seeded, labeled)", "C2", "A7"), ("C0 vs A3 (objectives at seed 0, labeled)", "C0", "A3"), ("C1 vs A7 (seed 1: recipe vs plain)", "C1", "A7")]
    field = [(f"{x} vs EqR", x, "eqr") for x in ("C0", "C1", "C2", "C4", "C5")] + [("C5 vs CGAR", "C5", "cgar"), ("C5 vs alphaXiv TRM-MLP", "C5", "trmpub"), ("C2 vs CGAR", "C2", "cgar"), ("C2 vs alphaXiv TRM-MLP", "C2", "trmpub"), ("C5 vs X0", "C5", "X0"), ("C2 vs X0", "C2", "X0"), ("C0 vs X0", "C0", "X0")]
    block("D16 (the headline column; the full 422,786):", F16, triple + treat + nightA + field)
    block("D64 (the 100k subsample; the field's fulls and X0 by intersection, labeled):", F64, triple + treat + nightA + field)
    block("D128 (the 20k scan set, identical across the three campaigns):", F128, triple + treat + [p for p in nightA if p[2] != "A3" or True] + [p for p in field if p[2] in ("eqr", "cgar", "trmpub")])
    block("D256 (the 5k set, identical to the frontier's):", F256, triple + treat + [p for p in field if p[2] in ("eqr", "cgar", "trmpub")])
    say("  depth per arm (regressions = solved at the shallower depth and not at the deeper one): D64 vs D16 on the 100k; D128 vs D64 on the 20k; D256 vs D128 on the 5k ∩ 20k (256 puzzles, labeled)")
    for x in present + list(REF_FRONT):
        if F16.get(x) is not None and F64.get(x) is not None: PR[f"depth16-64 {x}"] = pair_line(f"{x} D64 vs D16", F64[x], F16[x])
        if F64.get(x) is not None and F128.get(x) is not None: PR[f"depth64-128 {x}"] = pair_line(f"{x} D128 vs D64", F128[x], F64[x])
        if F128.get(x) is not None and F256.get(x) is not None: PR[f"depth128-256 {x}"] = pair_line(f"{x} D256 vs D128", F256[x], F128[x])
    say("  vsel vs final (the final grid's 50k ⊂ the full) and EMA (headline) vs raw (alt, 50k):")
    for x in present:
        zf = recs(evdir(x) / "full_final_t16"); za = recs(evdir(x) / "full_vsel_t16_alt")
        if zf is not None and F16.get(x) is not None: PR[f"vsel-final {x}"] = pair_line(f"{x} vsel vs final", F16[x], zf)
        if za is not None and F16.get(x) is not None: PR[f"ema-raw {x}"] = pair_line(f"{x} EMA vs raw", F16[x], za)
    say("  unions (cold16 on the full; portfolio facts, labeled):")
    def union(keys, F):
        zs = [F[k] for k in keys if F.get(k) is not None]
        if len(zs) < 2 or not all(np.array_equal(z["idx"], zs[0]["idx"]) for z in zs): return None
        u = np.zeros(len(zs[0]["idx"]), bool)
        for z in zs: u |= z["cold_exact"].astype(bool)
        return float(u.mean())
    for keys in (["C0", "C1", "C2"], ["C0", "C1", "C2", "C3", "C4", "C5", "C6"], ["C4", "C5"], ["C5", "eqr"]):
        u = union(keys, F16); say(f"    {' u '.join(keys)}: {pp(u)}"); PR[f"union {'+'.join(keys)}"] = u
    say("  overlaps at D16 (Jaccard of the failure sets; identical fulls):")
    def jacc(a, b):
        fa = ~F16[a]["cold_exact"].astype(bool); fb = ~F16[b]["cold_exact"].astype(bool); return float((fa & fb).sum() / max((fa | fb).sum(), 1))
    for a_, b_ in (("C0", "C1"), ("C0", "C2"), ("C1", "C2"), ("C5", "C4"), ("C5", "C2"), ("C2", "eqr"), ("C5", "eqr"), ("C0", "A5")):
        if F16.get(a_) is not None and F16.get(b_) is not None and np.array_equal(F16[a_]["idx"], F16[b_]["idx"]): j = jacc(a_, b_); PR[f"jaccard-fail {a_}|{b_}"] = j; say(f"    failures {a_} | {b_}: Jaccard {j:.3f}")
    J["paired"] = PR
    # ---------- F ----------
    say("\n== F. FUNNEL (rho, r) per rating quartile — the k128 scans (5k) and the vb screens (strat-512, k256); fit on draws <= 64 ==")
    FUN = {}
    for x in present:
        for lab, p, k in (("k128", k128_dir(x), 128), ("screen", screen_dir(x), 256)):
            z = recs(p)
            if z is None or "mi_first_hit" not in z: continue
            rows = funnel_rows(z, k); FUN[f"{lab} {x}"] = rows
            say(f"  {lab:6s} {x}: " + " | ".join(f"{r['bin']} n{r['n']} cold {pp(r['cold']).strip()} b1 {pp(r['b1']).strip()} hit {pp(r['hit']).strip()} rho {f(r['rho'],4,2).strip()} r {f(r['r'],5,3).strip()}" for r in rows))
    J["funnel"] = FUN
    # ---------- G ----------
    say("\n== G. THE DEPTH LADDER (exact_by_step_curve per row; each row on its own set, labeled: D16 full / D64 100k / D128 20k / D256 5k; regressions per row) ==")
    say("  arm | row | exact at step 1 / 2 / 4 / 8 / 16 / 32 / 64 / 128 / 256 | regressions")
    LAD = {}
    for x in present + list(REF_FRONT) + list(REF_FINALA):
        E = EV.get(x) or REF.get(x)
        if not E: continue
        for nm, lab in (("full_vsel_t16", "D16 full"), ("full_vsel_t64", "D64"), ("sub20k_t128", "D128 20k"), ("sub5k_t256", "D256 5k")):
            c = (E.get("curves") or {}).get(nm)
            if not c: continue
            pts = {s: c[s - 1] for s in (1, 2, 4, 8, 16, 32, 64, 128, 256) if s <= len(c)}; LAD[f"{x} {nm}"] = dict(pts=pts, regr=E["regr"].get(nm))
            say(f"  {x:6s} | {lab:9s} | " + " / ".join(f"{100*v:.2f}" for v in pts.values()) + f" | {E['regr'].get(nm)}")
    J["ladder"] = LAD
    # ---------- H ----------
    say("\n== H. THE HALTING HEAD (R10) on the D16 fulls: AUC(q_halt at step 16, exact); halt = q_halt > q_continue; the emulated ACT row = exact at the first halting step ==")
    say("  arm | n | cold | AUC(q_halt) | AUC(margin) | halt frac | precision | recall | mean first-halt step | exact at first halt | delta vs cold")
    HAL = {}
    for x in present + list(REF_FRONT):
        z = F16.get(x); h = halting(z)
        if h is None: say(f"  {x}: no q_by_step column"); continue
        HAL[x] = h; say(f"  {x:6s} | {h['n']} | {pp(h['cold'])} | {f(h['auc'],6,4)} | {f(h['margin_auc'],6,4)} | {pp(h['halt_frac'])} | {pp(h['precision'])} | {pp(h['recall'])} | {f(h['first_halt_mean'],5,2)} | {pp(h['exact_at_first_halt'])} | {f(100*h['delta'] if h['delta'] is not None else None,6,2)} pp")
    J["halting"] = HAL
    # ---------- I ----------
    say("\n== I. RATING BANDS (tdoku rating; cold) — D64 on the identical 100k (C arms, A3/A5/A7; the field's fulls and X0 by intersection) and D16 on the full ==")
    BAND = {}
    for depth, F in (("D64", F64), ("D16", F16)):
        base = F.get("C0")
        if base is None: continue
        idx = base["idx"]; rat = base["rating"]; cols = {}
        for x, z in F.items():
            if z is None: continue
            if np.array_equal(z["idx"], idx): cols[x] = z["cold_exact"].astype(bool)
            else:
                common, ka, kb = np.intersect1d(idx, z["idx"], return_indices=True)
                if len(common) == len(idx): c = np.zeros(len(idx), bool); c[ka] = z["cold_exact"].astype(bool)[kb]; cols[x] = c
        order = [k for k in present + ["A3", "A5", "A7", "eqr", "cgar", "trmpub", "X0"] if k in cols]
        say(f"  {depth}: band | n | " + " | ".join(order))
        BAND[depth] = {}
        for lo, hi in RBANDS:
            m = (rat >= lo) & (rat < hi)
            if not m.any(): continue
            BAND[depth][f"{lo}-{hi}"] = {x: float(cols[x][m].mean()) for x in order}
            say(f"    [{lo},{hi if hi < 10**6 else 'inf'}) | {m.sum():6d} | " + " | ".join(f"{100*cols[x][m].mean():6.2f}" for x in order))
    J["bands"] = BAND
    # ---------- J ----------
    if not a.no_commit:
        say("\n== J. THE COMMIT HEAD (C6; sub20k_t16_commit: the head's probability per free cell per step + correctness bits; tau .9) ==")
        z = recs(evdir("C6") / "sub20k_t16_commit")
        if z is not None and "commit_by_step" in z:
            from scipy import stats
            cb = z["commit_by_step"].astype(np.float32); n, T, _ = cb.shape
            ok = np.unpackbits(z["cellok_by_step"].astype(np.uint8), axis=2, bitorder="little")[:, :, :81].astype(bool)
            free = np.unpackbits(z["free_cells"].astype(np.uint8), axis=1, bitorder="little")[:, :81].astype(bool)
            cold = z["cold_exact"].astype(bool); nfree = np.maximum(free.sum(1), 1)
            say("  step | solved: mean c / committed(c>.9) / wrong-among-committed / cells correct || unsolved: mean c / committed / wrong-among-committed / cells correct")
            CM = {}
            for t in (0, 1, 3, 7, 15):
                c = cb[:, t]; com = (c > .9) & free; row = {}
                for lab, m in (("solved", cold), ("unsolved", ~cold)):
                    if not m.any(): row[lab] = None; continue
                    mc = float((c[m] * free[m]).sum() / free[m].sum()); fc = float(com[m].sum() / free[m].sum()); wc = float((com[m] & ~ok[m, t]).sum() / max(com[m].sum(), 1)); cc = float((ok[m, t] & free[m]).sum() / free[m].sum())
                    row[lab] = (mc, fc, wc, cc)
                CM[t + 1] = row
                say(f"  {t+1:4d} | " + " || ".join(("-" if row[l] is None else f"{row[l][0]:.3f} / {100*row[l][1]:.1f} % / {100*row[l][2]:.1f} % / {100*row[l][3]:.1f} %") for l in ("solved", "unsolved")))
            # per-puzzle signal: the head's mean over free cells at the last step vs exactness
            sig = (cb[:, -1] * free).sum(1) / nfree; mn = np.where(free, cb[:, -1], 1.0).min(1)
            def auc(score, y): return float(stats.mannwhitneyu(score[y], score[~y], alternative="greater").statistic / (y.sum() * (~y).sum()))   # a HIGH c predicts correct
            a_mean, a_min = auc(sig, cold), auc(mn, cold)
            # the head as a verifier-free selector: accept a puzzle if mean c >= thr; precision/recall at thr .9
            thr = .9; acc = sig >= thr; prec = float((acc & cold).sum() / max(acc.sum(), 1)); rec = float((acc & cold).sum() / max(cold.sum(), 1))
            say(f"  per-puzzle correctness signal at step 16: AUC(mean c over free cells, exact) = {a_mean:.4f}; AUC(min c) = {a_min:.4f}; accept-if-mean-c >= .9: precision {100*prec:.2f} % recall {100*rec:.2f} % (n unsolved {int((~cold).sum())})")
            # reliability of c at the last step over free cells (all puzzles; and unsolved only)
            edges = np.array([0, .1, .3, .5, .7, .9, .99, 1.0001]); cl = cb[:, -1][free]; okl = ok[:, -1][free]
            say("  reliability (step 16, free cells): bin | n | mean c | P(correct) || unsolved puzzles only: n | mean c | P(correct)")
            REL = []
            clu = cb[:, -1][~cold][free[~cold]]; oku = ok[:, -1][~cold][free[~cold]]
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (cl >= lo) & (cl < hi); mu = (clu >= lo) & (clu < hi)
                REL.append(dict(lo=float(lo), hi=float(hi), n=int(m.sum()), c=float(cl[m].mean()) if m.any() else None, acc=float(okl[m].mean()) if m.any() else None, n_u=int(mu.sum()), c_u=float(clu[mu].mean()) if mu.any() else None, acc_u=float(oku[mu].mean()) if mu.any() else None))
                r = REL[-1]; say(f"  [{lo:.2f},{min(hi,1):.2f}) | {r['n']:8d} | {f(r['c'],5,3)} | {pp(r['acc'])} || {r['n_u']:7d} | {f(r['c_u'],5,3)} | {pp(r['acc_u'])}")
            ece = float(sum(r["n"] * abs((r["c"] or 0) - (r["acc"] or 0)) for r in REL if r["n"]) / max(sum(r["n"] for r in REL), 1))
            say(f"  ECE (free cells, step 16) = {ece:.4f}; the softmax at stalls (E6, C0-C5): top-5 correct 48-68 % at confidence 1.000")
            J["commit"] = dict(per_step={str(k): v for k, v in CM.items()}, auc_mean=a_mean, auc_min=a_min, precision_at_9=prec, recall_at_9=rec, reliability=REL, ece=ece, n=int(n), cold=float(cold.mean()))
        else: say("  no commit records")
    # ---------- K ----------
    say("\n== K. MEMORIZATION / TRAJECTORY signatures (descriptive; the analyzer's CLEAN split reads these) ==")
    say("  arm | budget | selected | monitor peak (step) | plateau | end EMA | decline | CE at vsel / end | train_exact at vsel / end | halt end | cold16 vsel / final (50k) | drop")
    for x in present:
        t = TR[x]; e = EV[x]; adm = ADM[x]; bud = (adm.get("argv") or {}).get("steps") if isinstance(adm.get("argv"), dict) else None; pl = t["plateau"]
        say(f"  {x} | {bud} | {t['vsel']} | {pp(t['peak'][1])} ({t['peak'][0]}) | {pl[0] if pl else '-'}..{pl[1] if pl else '-'} | {pp(t['end'][1])} | {f(t['decline'],6,3)} | {f(t['ce_at_vsel'],5,3)} / {f(t['ce_end'],5,3)} | {f(t['te_at_vsel'],5,3)} / {f(t['te_end'],5,3)} | {f(t['hf_end'],5,3)} | {pp(e['cold16'])} / {pp(e['final'])} | {f((e['cold16'] or 0) - (e['final'] or 0),6,4) if e['cold16'] is not None and e['final'] is not None else '-'}")
    # ---------- L ----------
    say("\n== L. COMPUTE (analytic MACs per outer step per puzzle from the frontier report: DEC-w384 70.0 GMAC = 4.6x the TRM-MLP's 15.2; w192 20.4 = 1.34x; C4's attention adds its q/k projections, not counted; labeled inferred) ==")
    say("  arm | params | GMAC/step | D16 TMAC (cold16) | D64 TMAC (cold64) | D128 TMAC (D128) | D256 TMAC (D256)")
    COMP = {}
    for x in present + list(REF_FRONT):
        if x in present: w = (ADM[x].get("cfg") or {}).get("dec_width"); g = GMAC["w192"] if w == 192 else GMAC["w384"]; params = ADM[x].get("n_params_bulk"); e = EV[x]
        else: g = GMAC["trm"]; params = 5037058; e = REF[x]
        COMP[x] = dict(params=params, gmac=g, tmac16=16 * g / 1000, tmac64=64 * g / 1000, tmac128=128 * g / 1000, tmac256=256 * g / 1000)
        say(f"  {x:6s} | {params} | {g:5.1f}{' (+attn)' if x == 'C4' else ''} | {16*g/1000:5.2f} ({pp(e['cold16']).strip()}) | {64*g/1000:5.2f} ({pp(e['cold64']).strip()}) | {128*g/1000:5.2f} ({pp(e['d128']).strip()}) | {256*g/1000:5.2f} ({pp(e['d256']).strip()})")
    J["compute"] = COMP
    if a.curv:
        say("\n== M. CURVATURE from Adam-v (ckpt_latest opt_state) — skipped unless --curv ==")
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text("\n".join(L) + "\n")
    OUTJ.write_text(json.dumps(J, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating,)) else (int(o) if isinstance(o, np.integer) else (bool(o) if isinstance(o, np.bool_) else str(o)))))
    say(f"\n({time.time()-t0:.0f}s) artifact -> {OUT} (+ .json)")

if __name__ == "__main__":
    main()
