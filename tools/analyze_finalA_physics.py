#!/usr/bin/env python3
"""FINAL PHASE Night A — physics pass, written AT ANALYSIS TIME (2026-09-07; descriptive only).
The registered verdict authority is tools/analyze_finalA.py (byte-untouched since its registration commit
645f4d1, selftest 32/32); NOTHING here is a decision rule. Derivative of analyze_sportC2_physics.py (readers).

Reads the extracted finalA corpus in runs/ (+ the banked sportC1 X0 and sportC2 X1 / R3 references on disk):
  A. ADMISSION (ckpt config + param count + argv), grid census, val-selection PROVENANCE (every vsel eval's
     ckpt path; the offline re-selection on val_t16_ema vs the chain's val_best.txt).
  B. TRAJECTORIES — train rows (segment CE, train_exact, halt_frac, mean_steps, q_loss, anchor CE where logged),
     2k monitors (val_t16_ema / val_t16 raw), measured wall pace from the row timestamps, resumes.
  C. STABILITY — explosion census rows (vsel + final; t64 + t256), STOPPED markers, monitors.
  D. EVAL TABLE — cold16 vsel / final / alt | cold64 | per-octile cold16 | first_exact (D16, D64) | the vb SCREEN
     rows (strat-512, k256: b1, t1r, verified, majority) on EVERY arm and on X0 | the 20k k128 SCAN rows (A0-A2, X0, X1)
     | census | calibration-at-stalls.
  E. PAIRED McNemar on identical puzzle sets — the field ledger (A1 / A2 vs X0 at matched seed; A0 vs X0 = the seed
     pair), the symmetry row (A3 vs X1, A3 vs X0, A4 vs A3), the objectives (A5 vs A3), the DEC pair (A7 vs A3), the
     width points (A8 vs A3, A8 vs A7 at matched seed), D16 vs D64 per arm (regressions), vsel vs final, EMA vs raw.
  F. FUNNEL (rho, r) per octile from the vb screens (k256, strat-512; labeled) + the scans (A0-A2, k128, 20k).
  G. SELECTOR — residual AUC / spurious rate / t1r vs the free verifier, from screens (all arms) and scans (A0-A2).
  H. EXTRA READINGS the frozen analyzer does not score — one-shot vs propagation (first_exact), the wide arms by
     rating band (A4 - A3, A5 - A3, A7 - A3, A8 - A7), the DEC arms' cold union, the memorization signatures.
  J. CURVATURE CONCENTRATION from Adam-v (PR/n) where the ckpt carries an optimizer state.

  PYTHONPATH=src .venv/bin/python tools/analyze_finalA_physics.py [--grids] [--no-curv]
      -> runs/analysis/finalA_physics_20260907.txt (+ .json)
"""
from __future__ import annotations
import argparse, datetime as dt, json, math, os, pickle, re, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(os.environ.get("QHRRN_RUNS", ROOT / "runs"))
NPZ = Path(os.environ.get("QHRRN_NPZ", ROOT / "data/sudoku_extreme/sudoku_extreme_seed0.npz"))
OUT = RUNS / "analysis" / "finalA_physics_20260907.txt"; OUTJ = RUNS / "analysis" / "finalA_physics_20260907.json"
TAG = "finalA"
X_ARMS = ["A0", "A1", "A2"]; DEC_ARMS = ["A3", "A4", "A5", "A6", "A7", "A8"]; ARMS = X_ARMS + DEC_ARMS
DESC = {"A0": "X0 seed 1 (the noise-floor pair)", "A1": "X0 + FPA k1 eps.2 frac.25 (seed 0)", "A2": "X0 + RI sigma 1 (seed 0)",
        "A3": "DEC-w384, no digit aug (seed 0)", "A4": "DEC-w384 + digit aug (seed 0)", "A5": "DEC-w384 + FPA + RI (seed 0)",
        "A6": "DEC-w512 (seed 0; SKIPPED)", "A7": "DEC-w384 seed 1 (the DEC floor)", "A8": "DEC-w512 seed 1 (remat)"}
X0 = dict(full16=RUNS / "sxeval_psportC1X0/full_vsel_t16", full64=RUNS / "sxeval_psportC1X0/full_vsel_t64", scan=RUNS / "sxscan_psportC1X0", screen=RUNS / "sxscreen_psportC1X0_vb", pre=RUNS / "pretrainsportC1_X0")
X1 = dict(full16=RUNS / "sxeval_psportC2X1/full_vsel_t16", full64=RUNS / "sxeval_psportC2X1/full_vsel_t64", scan=RUNS / "sxscan_psportC2X1")
R3 = dict(full64=RUNS / "sxeval_psportC2R3/full_vsel_t64", scan=RUNS / "sxscan_psportC2R3")
L = []; J = {}
def say(s=""): L.append(str(s)); print(s, flush=True)
def f(x, w=6, p=2): return " " * (w - 1) + "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:{w}.{p}f}"
def pp(x): return "   -  " if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{100*x:6.2f}"
def jload(p):
    p = Path(p); return json.loads(p.read_text()) if p.exists() else None
def pdir(a): return RUNS / f"pretrain{TAG}_{a}"
def evdir(a): return RUNS / f"sxeval_p{TAG}{a}"
def full_dir(a, which, t=16): return evdir(a) / f"full_{which}_t{t}"
def scan_dir(a): return RUNS / f"sxscan_p{TAG}{a}"
def screen_dir(a, tag="vb"): return RUNS / f"sxscreen_p{TAG}{a}_{tag}"
def census_json(a, which): return jload(RUNS / f"sxcensus_p{TAG}{a}_{which}" / "census.json")
def calib_json(a): return jload(RUNS / f"sxcalib_p{TAG}{a}_vsel" / "calib.json")
def is_wide(a): return a in DEC_ARMS

# ---------- readers ----------
def count_params(tree):
    n = 0; st = [tree]
    while st:
        x = st.pop()
        if isinstance(x, dict): st.extend(x.values())
        elif isinstance(x, (list, tuple)): st.extend(x)
        else: n += int(np.prod(np.shape(x)))
    return n
def walk_finite(t):
    st = [t]
    while st:
        x = st.pop()
        if isinstance(x, dict): st.extend(x.values())
        elif isinstance(x, (list, tuple)): st.extend(x)
        elif hasattr(x, "dtype") and np.issubdtype(np.asarray(x).dtype, np.number):
            if not np.isfinite(np.asarray(x)).all(): return False
    return True
CFG_KEYS = ("cell_kind", "dec_width", "dec_coupling", "trm_hidden", "trm_layers", "trm_h_cycles", "trm_l_cycles", "trm_lambda", "trm_beta", "trm_ri_sigma",
            "sudoku_digit_aug", "fpa_k", "fpa_eps", "fpa_frac", "fpa_w", "seed", "remat", "loss_kind", "T", "sudoku_layout")
ARGV_KEYS = ("seed", "steps", "batch", "lr", "lr_end", "warmup", "wd", "beta2", "ema", "sudoku_aug", "sudoku_digit_aug", "sot", "act", "cell", "dec_width", "fpa_k", "fpa_eps", "fpa_frac", "trm_ri_sigma", "remat", "grid_every", "monitor_every", "ckpt_every", "loss")
def admission(arm, grids=False):
    d = pdir(arm); out = {"dir": d.exists()}
    if not d.exists(): return out
    ck = d / "ckpt_latest.pkl"
    if ck.exists():
        try:
            c = pickle.load(open(ck, "rb")); cfg = c.get("config", {})
            out["step"] = int(c["step"]); out["cfg"] = {k: cfg.get(k) for k in CFG_KEYS if k in cfg}
            out["n_params"] = count_params(c["state"]["model"]); out["has_ema"] = c.get("state_ema") is not None
            out["final_finite"] = walk_finite(c["state"]["model"]); out["has_opt"] = c.get("opt_state") is not None
        except Exception as e: out["ckpt_err"] = f"{type(e).__name__}: {e}"
    cj = d / "config.json"
    if cj.exists():
        try:
            j = json.loads(cj.read_text()); a = j.get("argv", {}); out["argv"] = {k: a.get(k) for k in ARGV_KEYS if k in a} if isinstance(a, dict) else str(a)[:300]
            out["git"] = j.get("git"); out["n_params_bulk"] = j.get("n_params_bulk")
        except Exception as e: out["argv_err"] = str(e)
    for nm in ("STOPPED.txt", "NAN_ABORT.txt", "resumes.txt", "val_best.txt", "RETRY_REMAT.txt"):
        p = d / nm; out[nm] = p.read_text().strip() if p.exists() else None
    steps = sorted(int(re.search(r"ckpt_(\d+)\.pkl$", p.name).group(1)) for p in d.glob("ckpt_[0-9]*.pkl"))
    out["grid_steps"] = steps
    if grids:
        bad = []
        for s in steps:
            try:
                if not walk_finite(pickle.load(open(d / f"ckpt_{s:06d}.pkl", "rb"))["state"]["model"]): bad.append(s)
            except Exception: bad.append(s)
        out["grid_nonfinite"] = bad
    return out
def load_metrics(path: Path):
    tr, mon = {}, {}
    if not path.exists(): return tr, mon
    for line in path.read_text().splitlines():
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
def near(dct, target, before=True, window=None):
    ks = [s for s in dct if finite(dct[s]) and (s <= target if before else s >= target)]
    if not ks: return None
    k = max(ks) if before else min(ks)
    if window is not None and abs(k - target) > window: return None
    return dct[k]
def traj_summary(tr, mon, key="val_t16_ema"):
    steps = list(tr.keys())
    if not steps: return {}
    CE = {s: tr[s].get("ce_in") for s in steps}
    def series(k): return {s: tr[s].get(k) for s in steps if finite(tr[s].get(k))}
    ce_f = {s: v for s, v in CE.items() if finite(v)}; ce_min_step = min(ce_f, key=ce_f.get) if ce_f else None
    extra = sorted({k for s in steps[-3:] for k in tr[s].keys()} - {"step", "loss", "ce_in", "lr", "t", "steps_per_sec", "A_total", "I_total", "rule_H", "q_loss", "halt_frac", "mean_steps", "train_exact"})
    ts = []
    for s in steps:
        t = tr[s].get("t")
        if t:
            try: ts.append((s, dt.datetime.fromisoformat(t)))
            except Exception: pass
    wall_h = (ts[-1][1] - ts[0][1]).total_seconds() / 3600 if len(ts) >= 2 else None
    # measured pace from timestamps over the last 40 % of the run (the logged steps_per_sec reads 2x high on the field loop)
    pace_ts = None
    if len(ts) >= 4:
        i0 = int(len(ts) * .6); (s0, t0), (s1, t1) = ts[i0], ts[-1]; dsec = (t1 - t0).total_seconds()
        pace_ts = (s1 - s0) / dsec if dsec > 0 else None
    printed = series("steps_per_sec"); pace_pr = float(np.median(list(printed.values()))) if printed else None
    val = {s: mon[s].get(key) for s in mon if finite(mon[s].get(key))}; raw = {s: mon[s].get(key.replace("_ema", "")) for s in mon if finite(mon[s].get(key.replace("_ema", "")))}
    vpk = max(val, key=val.get) if val else None
    return dict(n_rows=len(steps), last_step=steps[-1], loss_last=tr[steps[-1]].get("loss"), ce_last=float(np.mean([CE[s] for s in steps[-5:] if finite(CE[s])])) if any(finite(CE[s]) for s in steps[-5:]) else None,
                ce_min=ce_f[ce_min_step] if ce_min_step is not None else None, ce_min_step=ce_min_step,
                ce_at={k: near(CE, k, window=1500) for k in (5000, 10000, 20000, 30000, 40000, 50000)},
                train_exact=series("train_exact"), halt=series("halt_frac"), mean_steps=series("mean_steps"), q_loss=series("q_loss"),
                extra_keys=extra, extra_last={k: tr[steps[-1]].get(k) for k in extra}, wall_h=wall_h, pace_ts=pace_ts, pace_printed=pace_pr,
                val=val, val_raw=raw, val_peak=(vpk, val[vpk]) if vpk is not None else None, val_end=(max(val), val[max(val)]) if val else None,
                eta={s: mon[s].get("eta") for s in mon if finite(mon[s].get("eta"))}, n_mon=len(mon))
def recs(p):
    p = Path(p); q = p / "records_all.npz"
    if not q.exists():
        parts = sorted(p.glob("records_s*.npz"))
        if not parts: return None
        arrs = [dict(np.load(x, allow_pickle=True)) for x in parts]; keys = [k for k in arrs[0] if all(k in a for a in arrs)]
        z = {k: np.concatenate([a[k] for a in arrs]) for k in keys}; z["_partial"] = True
    else: z = dict(np.load(q, allow_pickle=True))
    order = np.argsort(z["idx"], kind="stable")
    return {k: (v[order] if hasattr(v, "shape") and v.shape and v.shape[0] == len(order) else v) for k, v in z.items()}
def mcnemar(a, b):
    oa = int(np.sum(a & ~b)); ob = int(np.sum(~a & b)); n = oa + ob
    if n == 0: return oa, ob, 1.0
    from scipy import stats
    return oa, ob, float(min(1.0, 2 * stats.binom.cdf(min(oa, ob), n, 0.5)))
def paired(za, zb):
    """align two record dicts on idx; returns (cold_a, cold_b, n, note)."""
    if za is None or zb is None: return None
    ia, ib = za["idx"], zb["idx"]
    if len(ia) == len(ib) and np.array_equal(ia, ib): return za["cold_exact"].astype(bool), zb["cold_exact"].astype(bool), len(ia), "identical idx"
    common, ka, kb = np.intersect1d(ia, ib, return_indices=True)
    if len(common) == 0: return None
    return za["cold_exact"].astype(bool)[ka], zb["cold_exact"].astype(bool)[kb], len(common), f"INTERSECTION n={len(common)} (labeled)"
def pair_line(name, za, zb, extra=""):
    r = paired(za, zb)
    if r is None: say(f"  {name:36s}: no paired data"); return None
    a, b, n, note = r; oa, ob, p = mcnemar(a, b)
    say(f"  {name:36s}: {pp(a.mean())} vs {pp(b.mean())} (delta {100*(a.mean()-b.mean()):+6.2f}pp) | only-A {oa:7d} / only-B {ob:7d} | p {p:.2e} | n {n} {note} {extra}")
    return dict(a=float(a.mean()), b=float(b.mean()), only_a=oa, only_b=ob, p=p, n=n, note=note)
def b1_bits(z): return z["mi_exact_k"][:, 0].astype(bool) if "mi_exact_k" in z and z["mi_exact_k"].ndim == 2 else None
def t1r_bits(z, k):
    if "mi_exact_k" not in z or "mi_resid_k" not in z: return None
    ex = z["mi_exact_k"].astype(bool); re_ = z["mi_resid_k"].astype(np.float64); k = min(k, ex.shape[1]); fin = np.isfinite(re_[:, :k])
    best = np.argmin(np.where(fin, re_[:, :k], np.inf), axis=1); return ex[np.arange(len(ex)), best]
def selector(z, k):
    if z is None or "mi_exact_k" not in z or "mi_resid_k" not in z: return None
    from scipy import stats
    ex = z["mi_exact_k"][:, :k].astype(bool); rs = z["mi_resid_k"][:, :k].astype(np.float64); fin = np.isfinite(rs); ef = ex & fin; wf = (~ex) & fin
    auc = 1.0 - float(stats.mannwhitneyu(rs[ef], rs[wf], alternative="less").statistic / (ef.sum() * wf.sum())) if ef.sum() and wf.sum() else None
    thr = np.median(rs[ef]) if ef.any() else np.nan; spur = float((rs[wf] <= thr).mean()) if wf.any() else None
    cold = z["cold_exact"].astype(bool); fh = z["mi_first_hit"]; ver = float((cold | ((fh >= 0) & (fh < k))).mean()); t1r = float(t1r_bits(z, k).mean())
    return dict(auc=auc, spurious=spur, t1r=t1r, verified=ver, ratio=(t1r / ver if ver else None), b1=float(ex[:, 0].mean()), rho=float((cold | (ex.any(1))).mean()))
def fit_rho_r(fh, k_fit=64):
    fh = np.asarray(fh); hit = (fh >= 0) & (fh < k_fit); t = fh[hit]; n_c = int(np.sum(~hit)); best = (-np.inf, None, None)
    for rho in np.linspace(0.02, 1.0, 50):
        for r in np.geomspace(1e-3, 0.9, 60):
            ll = np.sum(np.log(rho) + t * np.log1p(-r) + np.log(r)) if len(t) else 0.0
            ll += n_c * np.log(max(1 - rho + rho * (1 - r) ** k_fit, 1e-300))
            if ll > best[0]: best = (ll, rho, r)
    return best[1], best[2]
def funnel_rows(z, k, nq=4):
    """per rating quantile (nq bins): cold, b1, hit@k, rho, r (fit on draws <= 64)."""
    rat = z["rating"]; fh = z["mi_first_hit"]; cold = z["cold_exact"].astype(bool); b1 = b1_bits(z)
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
    return dict(n_solved=int(len(s)), med=float(np.median(s)) if len(s) else None, p90=float(np.percentile(s, 90)) if len(s) else None, frac1=float((s == 0).mean()) if len(s) else None, frac2=float((s <= 1).mean()) if len(s) else None)
def leaves(t, path=""):
    if isinstance(t, dict):
        for k, v in t.items(): yield from leaves(v, path + "/" + k)
    elif isinstance(t, (list, tuple)):
        for i, v in enumerate(t): yield from leaves(v, path + f"[{i}]")
    else: yield path, np.asarray(t)
def find_nu(opt):
    found = []
    def walk(x):
        if hasattr(x, "_fields") and "nu" in x._fields: found.append(x.nu)
        elif isinstance(x, (list, tuple)):
            for y in x: walk(y)
        elif isinstance(x, dict):
            for y in x.values(): walk(y)
    walk(opt); return found
def curvature_row(path):
    c = pickle.load(open(path, "rb")); nus = find_nu(c.get("opt_state"))
    if not nus: return None
    nu = nus[0]; nu = nu["model"] if isinstance(nu, dict) and "model" in nu else nu
    vals = []; blocks = {}
    for p, a in leaves(nu):
        a = a.astype(np.float64).ravel(); vals.append(a); b = p.split("/")[1] if p.count("/") >= 1 else p; b = b.split("[")[0]; blocks[b] = blocks.get(b, 0.0) + float(a.sum())
    v = np.concatenate(vals); n = v.size; tot = v.sum(); pr = tot ** 2 / max((v ** 2).sum(), 1e-300)
    vs = np.sort(v)[::-1]; cum = np.cumsum(vs) / tot; k90 = int(np.searchsorted(cum, 0.9)) + 1
    top = sorted(blocks.items(), key=lambda kv: -kv[1])[:3]
    return dict(n=n, pr_n=pr / n, k90=k90 / n, top=", ".join(f"{b} {100*s/tot:.1f}%" for b, s in top))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--grids", action="store_true", help="walk every banked grid for finiteness (slow)"); ap.add_argument("--no-curv", action="store_true"); a = ap.parse_args(); L.clear()
    say("=" * 120); say("FINAL PHASE Night A — PHYSICS PASS (2026-09-07; analysis-time, descriptive, no rules; verdict authority = tools/analyze_finalA.py)"); say("=" * 120)
    present = [x for x in ARMS if pdir(x).exists()]
    # ---------- A ----------
    say("\n== A. ADMISSION (ckpt_latest config + parameter count; argv from config.json; markers; grids) ==")
    ADM = {}
    for x in ARMS:
        o = admission(x, a.grids); ADM[x] = o
        if not o["dir"]: say(f"  {x}: ABSENT ({DESC[x]})"); continue
        cfg = o.get("cfg", {}); av = o.get("argv", {})
        say(f"  {x} [{DESC[x]}]: step {o.get('step')} | params {o.get('n_params')} (config.json {o.get('n_params_bulk')}) | ema {o.get('has_ema')} | opt_state {o.get('has_opt')} | final finite {o.get('final_finite')} | git {o.get('git')}")
        say(f"     cfg: " + ", ".join(f"{k}={v}" for k, v in cfg.items()))
        if isinstance(av, dict): say(f"     argv: " + ", ".join(f"{k}={v}" for k, v in av.items()))
        gs = o.get("grid_steps", []); say(f"     grids: {len(gs)} banked [{gs[0] if gs else '-'}..{gs[-1] if gs else '-'}]" + (f" NON-FINITE {o['grid_nonfinite']}" if o.get("grid_nonfinite") else "") + f" | val_best {o.get('val_best.txt')!r} | resumes {o.get('resumes.txt')!r}")
        for nm in ("STOPPED.txt", "NAN_ABORT.txt", "RETRY_REMAT.txt"):
            if o.get(nm): say(f"     {nm}: {o[nm]!r}")
    J["admission"] = {x: {k: v for k, v in o.items() if k not in ("grid_steps",)} for x, o in ADM.items()}
    say("\n  PROVENANCE (one ckpt path per (arm, vsel)) + offline re-selection on val_t16_ema among banked grids:")
    PROV = {}
    for x in present:
        paths = {}
        for nm, p in (("full_vsel16", full_dir(x, "vsel") / "summary_all.json"), ("alt", evdir(x) / "full_vsel_t16_alt" / "summary_all.json"), ("full_vsel64", full_dir(x, "vsel", 64) / "summary_all.json"),
                      ("scan", scan_dir(x) / "summary_all.json"), ("screen_vb", screen_dir(x) / "summary_all.json"), ("census_vsel", RUNS / f"sxcensus_p{TAG}{x}_vsel" / "census.json"), ("calib", RUNS / f"sxcalib_p{TAG}{x}_vsel" / "calib.json")):
            s = jload(p)
            if s and s.get("ckpt"): paths[nm] = s["ckpt"]
        fin = jload(full_dir(x, "final") / "summary_all.json"); fpath = fin.get("ckpt") if fin else None
        tr, mon = load_metrics(pdir(x) / "metrics.jsonl"); banked = set(ADM[x].get("grid_steps", []))
        cand = [(mon[s]["val_t16_ema"], s) for s in mon if s in banked and finite(mon[s].get("val_t16_ema"))]
        best = max(cand)[1] if cand else None; chain = ADM[x].get("val_best.txt")
        uniq = sorted(set(paths.values())); m = re.search(r"ckpt_(\d+)\.pkl", uniq[0]) if uniq else None; vstep = int(m.group(1)) if m else None
        PROV[x] = dict(paths=paths, final_path=fpath, offline_best=best, chain_val_best=chain, vsel_step=vstep, consistent=len(uniq) == 1)
        say(f"  {x}: vsel evals on {len(uniq)} path(s) {uniq} | final path {fpath} | chain val_best {chain!r} | offline argmax(val_t16_ema over banked) = {best} -> {'AGREES' if (best is not None and vstep == best) else 'DIFFERS/NA'}")
    J["provenance"] = PROV
    # ---------- B ----------
    say("\n== B. TRAJECTORIES (train rows every 50 steps; monitors every 2k on the 64 train-file puzzles; pace measured from timestamps) ==")
    say("  arm | rows | last | CE 5k/10k/20k/30k/40k/50k | CE end / min (step) | train_exact end | halt_frac end | mean_steps end | q_loss end | wall h | pace ts it/s (printed) | val_t16_ema peak (step) / end | val_t16 raw at peak | extra keys")
    TR = {}
    for x in present:
        tr, mon = load_metrics(pdir(x) / "metrics.jsonl"); t = traj_summary(tr, mon); TR[x] = t
        if not t: say(f"  {x}: no rows"); continue
        ce = t["ce_at"]; te = t["train_exact"]; hf = t["halt"]; ms = t["mean_steps"]; ql = t["q_loss"]
        vp = t["val_peak"]; ve = t["val_end"]; raw_at_peak = t["val_raw"].get(vp[0]) if vp else None
        say(f"  {x} | {t['n_rows']} | {t['last_step']} | " + "/".join(f(ce[k], 4, 2).strip() for k in (5000, 10000, 20000, 30000, 40000, 50000)) + f" | {f(t['ce_last'],5,3)} / {f(t['ce_min'],5,3)} ({t['ce_min_step']}) | {f(te[max(te)] if te else None,5,3)} | {f(hf[max(hf)] if hf else None,5,3)} | {f(ms[max(ms)] if ms else None,5,2)} | {f(ql[max(ql)] if ql else None,5,3)} | {f(t['wall_h'],5,2)} | {f(t['pace_ts'],5,2)} ({f(t['pace_printed'],5,2)}) | {pp(vp[1] if vp else None)} ({vp[0] if vp else '-'}) / {pp(ve[1] if ve else None)} | {pp(raw_at_peak)} | {t['extra_keys']}")
        if t["extra_last"]: say(f"     extra (last row): {t['extra_last']}")
    say("  val_t16_ema every 10k (monitors; the selection key): " + " | ".join(f"{x}: " + " ".join(f"{s//1000}k={100*v:.1f}" for s, v in sorted(TR[x]['val'].items()) if s % 10000 == 0) for x in present if TR.get(x)))
    say("  segment CE every 10k: " + " | ".join(f"{x}: " + " ".join(f"{k//1000}k={f(TR[x]['ce_at'][k],4,2).strip()}" for k in (10000, 20000, 30000, 40000, 50000)) for x in present if TR.get(x)))
    J["traj"] = {x: {k: v for k, v in t.items() if k not in ("val", "val_raw", "train_exact", "halt", "mean_steps", "q_loss", "eta")} for x, t in TR.items()}
    J["traj_val"] = {x: {str(s): v for s, v in t["val"].items()} for x, t in TR.items() if t}
    # ---------- C ----------
    say("\n== C. STABILITY (explosion census on the vsel and final grids; STOPPED markers) ==")
    CEN = {}
    for x in present:
        row = {}
        for which in ("vsel", "final"):
            c = census_json(x, which)
            if c: row[which] = {int(r["t"]): (r.get("exploded_frac"), r.get("n")) for r in c["rows"]}
        CEN[x] = row
        say(f"  {x}: " + " | ".join(f"{which}: " + ", ".join(f"t{t} {pp(v[0])} (n {v[1]})" for t, v in sorted(rr.items())) for which, rr in row.items()) + (f" | STOPPED {ADM[x]['STOPPED.txt']!r}" if ADM[x].get("STOPPED.txt") else ""))
    J["census"] = {x: {w: {str(t): v for t, v in rr.items()} for w, rr in row.items()} for x, row in CEN.items()}
    # ---------- D ----------
    say("\n== D. EVAL TABLE (headline = EMA at D16 on the vsel grid; full test unless labeled; the wide arms' final/alt on 50k and D64 on 100k, labeled) ==")
    say("  arm | cold16 vsel (n) | final (n) | alt raw (n) | cold64 (n) | first_exact D16 med/p90 (frac step1) | first_exact D64 med/p90 | valid-wrong | mean viol | givens kept")
    EV = {}
    def summ(p): return jload(p / "summary_all.json")
    for x in present:
        sv, sf, sa, s64 = summ(full_dir(x, "vsel")), summ(full_dir(x, "final")), summ(evdir(x) / "full_vsel_t16_alt"), summ(full_dir(x, "vsel", 64))
        z16, z64 = recs(full_dir(x, "vsel")), recs(full_dir(x, "vsel", 64)); fe16, fe64 = first_exact_stats(z16), first_exact_stats(z64)
        EV[x] = dict(cold16=(sv or {}).get("exact_acc"), n16=(sv or {}).get("n"), final=(sf or {}).get("exact_acc"), nfin=(sf or {}).get("n"), alt=(sa or {}).get("exact_acc"), nalt=(sa or {}).get("n"), cold64=(s64 or {}).get("exact_acc"), n64=(s64 or {}).get("n"),
                     fe16=fe16, fe64=fe64, valid_wrong=(sv or {}).get("valid_wrong_frac"), viol=(sv or {}).get("mean_violations"), gk=(sv or {}).get("givens_kept_frac"), by_bin16=(sv or {}).get("by_rating_bin"), by_bin64=(s64 or {}).get("by_rating_bin"))
        e = EV[x]
        say(f"  {x} | {pp(e['cold16'])} ({e['n16']}) | {pp(e['final'])} ({e['nfin']}) | {pp(e['alt'])} ({e['nalt']}) | {pp(e['cold64'])} ({e['n64']}) | {f(fe16['med'],3,0) if fe16 else '  -'}/{f(fe16['p90'],3,0) if fe16 else '  -'} ({pp(fe16['frac1']) if fe16 else '   -  '}) | {f(fe64['med'],3,0) if fe64 else '  -'}/{f(fe64['p90'],3,0) if fe64 else '  -'} | {f(e['valid_wrong'],7,5)} | {f(e['viol'],5,1)} | {f(e['gk'],6,4)}")
    # references
    for nm, src in (("X0", X0), ("X1", X1)):
        s16, s64 = summ(src["full16"]), summ(src["full64"]); z16 = recs(src["full16"]); fe16 = first_exact_stats(z16); z64 = recs(src["full64"]); fe64 = first_exact_stats(z64)
        EV[nm] = dict(cold16=(s16 or {}).get("exact_acc"), n16=(s16 or {}).get("n"), cold64=(s64 or {}).get("exact_acc"), n64=(s64 or {}).get("n"), fe16=fe16, fe64=fe64, by_bin16=(s16 or {}).get("by_rating_bin"), by_bin64=(s64 or {}).get("by_rating_bin"))
        say(f"  {nm} ref | {pp(EV[nm]['cold16'])} ({EV[nm]['n16']}) | - | - | {pp(EV[nm]['cold64'])} ({EV[nm]['n64']}) | {f(fe16['med'],3,0) if fe16 else '  -'}/{f(fe16['p90'],3,0) if fe16 else '  -'} ({pp(fe16['frac1']) if fe16 else '   -  '}) | {f(fe64['med'],3,0) if fe64 else '  -'}/{f(fe64['p90'],3,0) if fe64 else '  -'} | | |")
    say("  per-octile cold16 (the summary's 8 rating bins over the full split; wide arms on the full test too):")
    for x in present + ["X0", "X1"]:
        bb = EV[x].get("by_bin16")
        if bb: say(f"    {x}: " + " ".join(pp(v).strip() for v in bb) + "   || cold64 by bin: " + (" ".join(pp(v).strip() for v in EV[x]["by_bin64"]) if EV[x].get("by_bin64") else "-"))
    say("\n  SCREENS (vb grid, strat-512, k256, headline depth/weights; labeled): arm | cold | b1 | t1r@128 | t1r@256 | verified@128 | verified@256 | majority@128 | majority@256 | rho@256 | selector AUC | spurious | mean first_exact")
    SCR = {}
    for x in present + ["X0"]:
        p = screen_dir(x) if x != "X0" else X0["screen"]; s = summ(p); z = recs(p)
        if not s: say(f"  {x}: no screen"); continue
        sel = selector(z, 256) if z is not None else None
        SCR[x] = dict(cold=s.get("exact_acc"), b1=s.get("b1_exact"), t1r128=(s.get("t1r_at_k") or {}).get("128"), t1r256=(s.get("t1r_at_k") or {}).get("256"), v128=(s.get("vote_at_k") or {}).get("128"), v256=(s.get("vote_at_k") or {}).get("256"),
                      mj128=(s.get("majority_vote_at_k") or {}).get("128"), mj256=(s.get("majority_vote_at_k") or {}).get("256"), sel=sel, mfe=s.get("mean_first_exact"), ckpt=s.get("ckpt"))
        r = SCR[x]
        say(f"  {x} | {pp(r['cold'])} | {pp(r['b1'])} | {pp(r['t1r128'])} | {pp(r['t1r256'])} | {pp(r['v128'])} | {pp(r['v256'])} | {pp(r['mj128'])} | {pp(r['mj256'])} | {pp(sel['rho'] if sel else None)} | {f(sel['auc'] if sel else None,5,3)} | {pp(sel['spurious'] if sel else None)} | {f(r['mfe'],5,1)}")
    say("\n  SCANS (20k, k128, t64; the registered X0-class protocol; A3-A8 absent = the deferred offline re-run): arm | cold | b1 | t1r@128 | verified@128 | majority - | rho | AUC | spurious | ratio t1r/verified")
    SCN = {}
    for x, p in [(x, scan_dir(x)) for x in present] + [("X0", X0["scan"]), ("X1", X1["scan"]), ("R3", R3["scan"])]:
        s = summ(p); z = recs(p)
        if not s and z is None: continue
        sel = selector(z, 128) if z is not None else None
        SCN[x] = dict(cold=(s or {}).get("exact_acc"), b1=(s or {}).get("b1_exact"), t1r=((s or {}).get("t1r_at_k") or {}).get("128"), v128=((s or {}).get("vote_at_k") or {}).get("128"), sel=sel, n=(s or {}).get("n"), partial=bool(z.get("_partial")) if z else False)
        r = SCN[x]; say(f"  {x} | {pp(r['cold'])} | {pp(r['b1'])} | {pp(r['t1r'])} | {pp(r['v128'])} | | {pp(sel['rho'] if sel else None)} | {f(sel['auc'] if sel else None,5,3)} | {pp(sel['spurious'] if sel else None)} | {f(sel['ratio'] if sel else None,5,3)}" + (f"  [n={r['n']}{' PARTIAL shards' if r['partial'] else ''}]"))
    say("\n  CALIBRATION AT STALLS (strat-512 on the vsel grid): arm | cold | n stalled | top-5 correct | mean conf | gap | entropy step1 | entropy stalled t64 | conf-wrong frac stalled | committed frac")
    CAL = {}
    for x in present:
        c = calib_json(x)
        if not c: say(f"  {x}: no calib"); continue
        CAL[x] = c; gap = (c.get("mean_conf_stalled") - c.get("topk_correct_stalled")) if (c.get("mean_conf_stalled") is not None and c.get("topk_correct_stalled") is not None) else None
        say(f"  {x} | {pp(c.get('cold'))} | {c.get('n_stalled')} | {pp(c.get('topk_correct_stalled'))} | {f(c.get('mean_conf_stalled'),5,3)} | {f(gap,6,3)} | {f(c.get('entropy_step1'),5,3)} | {f(c.get('entropy_t_stalled'),5,3)} | {pp(c.get('conf_wrong_frac_stalled'))} | {pp(c.get('committed_frac_stalled'))}")
    J["eval"] = EV; J["screens"] = SCR; J["scans"] = SCN; J["calib"] = CAL
    # ---------- E ----------
    say("\n== E. PAIRED McNemar on identical puzzle sets (cold, headline weights; A vs B; p two-sided exact binomial) ==")
    PR = {}
    F16 = {x: recs(full_dir(x, "vsel")) for x in present}; F64 = {x: recs(full_dir(x, "vsel", 64)) for x in present}
    F16["X0"], F64["X0"], F16["X1"], F64["X1"] = recs(X0["full16"]), recs(X0["full64"]), recs(X1["full16"]), recs(X1["full64"]); F64["R3"] = recs(R3["full64"])
    say("  D16 (the headline column; full test):")
    for nm, ka, kb in (("A0 vs X0 (seed pair = the floor)", "A0", "X0"), ("A1 vs X0 (FPA k1, matched seed 0)", "A1", "X0"), ("A2 vs X0 (RI, matched seed 0)", "A2", "X0"), ("A1 vs A0", "A1", "A0"), ("A2 vs A0", "A2", "A0"),
                       ("A3 vs X1 (symmetry vs no-orbit field cell)", "A3", "X1"), ("A3 vs X0 (DEC-w384 vs the field cell)", "A3", "X0"), ("A4 vs A3 (+ digit aug)", "A4", "A3"), ("A5 vs A3 (+ FPA + RI)", "A5", "A3"),
                       ("A7 vs A3 (DEC seed pair)", "A7", "A3"), ("A8 vs A3 (w512 s1 vs w384 s0)", "A8", "A3"), ("A8 vs A7 (w512 vs w384, matched seed 1)", "A8", "A7"), ("A8 vs X0", "A8", "X0"), ("A4 vs X0", "A4", "X0"), ("A5 vs X0", "A5", "X0")):
        if ka in F16 and kb in F16 and F16[ka] is not None and F16[kb] is not None: PR[f"D16 {nm}"] = pair_line(nm, F16[ka], F16[kb])
    say("  D64 (the depth row; X0-class on the full test; wide arms on their 100k subsample -> intersections with the full-test references, labeled):")
    for nm, ka, kb in (("A0 vs X0", "A0", "X0"), ("A1 vs X0", "A1", "X0"), ("A2 vs X0", "A2", "X0"), ("A3 vs X0", "A3", "X0"), ("A3 vs X1", "A3", "X1"), ("A3 vs R3 (our champion by rule, D64)", "A3", "R3"), ("A4 vs A3", "A4", "A3"), ("A5 vs A3", "A5", "A3"), ("A7 vs A3", "A7", "A3"), ("A8 vs A3", "A8", "A3"), ("A8 vs A7", "A8", "A7"), ("A8 vs X0", "A8", "X0")):
        if ka in F64 and kb in F64 and F64[ka] is not None and F64[kb] is not None: PR[f"D64 {nm}"] = pair_line(nm, F64[ka], F64[kb])
    say("  D16 -> D64 per arm (regressions = solved at 16 and not at 64):")
    for x in present + ["X0", "X1"]:
        if F16.get(x) is not None and F64.get(x) is not None: PR[f"depth {x}"] = pair_line(f"{x} D64 vs D16", F64[x], F16[x])
    say("  vsel vs final and headline (EMA) vs alt (raw) per arm (wide arms: 50k intersections, labeled):")
    for x in present:
        zf = recs(full_dir(x, "final")); za = recs(evdir(x) / "full_vsel_t16_alt")
        if zf is not None and F16.get(x) is not None:
            if PROV[x].get("final_path") == PROV[x].get("paths", {}).get("full_vsel16"): say(f"  {x} vsel == final (identical grid)")
            else: PR[f"vsel-final {x}"] = pair_line(f"{x} vsel vs final", F16[x], zf)
        if za is not None and F16.get(x) is not None: PR[f"ema-raw {x}"] = pair_line(f"{x} EMA (headline) vs raw (alt)", F16[x], za)
    say("  unions (cold16, full test; labeled portfolio facts):")
    def union(keys):
        zs = [F16[k] for k in keys if F16.get(k) is not None]
        if len(zs) < 2 or not all(np.array_equal(z["idx"], zs[0]["idx"]) for z in zs): return None
        u = np.zeros(len(zs[0]["idx"]), bool)
        for z in zs: u |= z["cold_exact"].astype(bool)
        return float(u.mean())
    for keys in (["A3", "A4", "A5", "A7", "A8"], ["A3", "A7"], ["A3", "X0"], ["A8", "X0"], ["A0", "A1", "A2", "X0"]):
        u = union(keys); say(f"    {' u '.join(keys)}: {pp(u)}"); PR[f"union {'+'.join(keys)}"] = u
    J["paired"] = PR
    # ---------- F ----------
    say("\n== F. FUNNEL (rho, r) per rating quartile — vb screens (strat-512, k256; fit on draws <= 64; labeled) and the 20k scans (k128) ==")
    FUN = {}
    for x in present + ["X0"]:
        p = screen_dir(x) if x != "X0" else X0["screen"]; z = recs(p)
        if z is None or "mi_first_hit" not in z: continue
        rows = funnel_rows(z, 256); FUN[f"screen {x}"] = rows
        say(f"  screen {x}: " + " | ".join(f"{r['bin']} cold {pp(r['cold']).strip()} b1 {pp(r['b1']).strip()} hit {pp(r['hit']).strip()} rho {f(r['rho'],4,2).strip()} r {f(r['r'],5,3).strip()}" for r in rows))
    for x in present[:3] + ["X0", "X1"]:
        p = scan_dir(x) if x in present else (X0["scan"] if x == "X0" else X1["scan"]); z = recs(p)
        if z is None or "mi_first_hit" not in z: continue
        rows = funnel_rows(z, 128); FUN[f"scan {x}"] = rows
        say(f"  scan   {x}: " + " | ".join(f"{r['bin']} cold {pp(r['cold']).strip()} b1 {pp(r['b1']).strip()} hit {pp(r['hit']).strip()} rho {f(r['rho'],4,2).strip()} r {f(r['r'],5,3).strip()}" for r in rows))
    J["funnel"] = FUN
    # ---------- H ----------
    say("\n== H. EXTRA READINGS (descriptive) ==")
    say("  one-shot vs propagation — first_exact on solved puzzles (D16 / D64): median, p90, fraction exact at outer step 1:")
    for x in present + ["X0", "X1"]:
        e = EV.get(x, {}); a16, a64 = e.get("fe16"), e.get("fe64")
        say(f"    {x}: D16 med {f(a16['med'],3,0) if a16 else '-'} p90 {f(a16['p90'],3,0) if a16 else '-'} step1 {pp(a16['frac1']) if a16 else '-'} step<=2 {pp(a16['frac2']) if a16 else '-'} | D64 med {f(a64['med'],3,0) if a64 else '-'} p90 {f(a64['p90'],3,0) if a64 else '-'} step1 {pp(a64['frac1']) if a64 else '-'}")
    say("  wide arms by rating band at D64 (100k subsample; identical idx across the wide arms -> paired deltas): band | A3 | A4 | A5 | A7 | A8 | X0 (subset) | X1 (subset)")
    RB = [(0, 1), (1, 10), (10, 30), (30, 60), (60, 10**6)]; BAND = {}
    zs = {x: F64.get(x) for x in ("A3", "A4", "A5", "A7", "A8", "X0", "X1") if F64.get(x) is not None}
    base = zs.get("A3")
    if base is not None:
        idx = base["idx"]; rat = base["rating"]
        cols = {}
        for x, z in zs.items():
            if np.array_equal(z["idx"], idx): cols[x] = z["cold_exact"].astype(bool)
            else:
                common, ka, kb = np.intersect1d(idx, z["idx"], return_indices=True)
                if len(common) == len(idx): c = np.zeros(len(idx), bool); c[ka] = z["cold_exact"].astype(bool)[kb]; cols[x] = c
        for lo, hi in RB:
            m = (rat >= lo) & (rat < hi)
            if not m.any(): continue
            BAND[f"{lo}-{hi}"] = {x: float(c[m].mean()) for x, c in cols.items()}
            say(f"    [{lo},{hi}) n={m.sum():6d} | " + " | ".join(f"{x} {pp(cols[x][m].mean()).strip()}" for x in ("A3", "A4", "A5", "A7", "A8", "X0", "X1") if x in cols))
    J["bands64"] = BAND
    say("  memorization signatures: segment CE end / min, train_exact end, vsel-vs-final drop (analyzer's CLEAN split reads these; here the values):")
    for x in present:
        t = TR.get(x, {}); e = EV.get(x, {}); te = t.get("train_exact") or {}
        say(f"    {x}: CE end {f(t.get('ce_last'),5,3)} min {f(t.get('ce_min'),5,3)} @ {t.get('ce_min_step')} | train_exact end {f(te[max(te)] if te else None,5,3)} | cold16 vsel {pp(e.get('cold16'))} final {pp(e.get('final'))} (drop {f((e.get('cold16') or 0) - (e.get('final') or 0),6,4) if e.get('cold16') is not None and e.get('final') is not None else '-'}; wide arms' final on 50k, labeled)")
    # ---------- J ----------
    if not a.no_curv:
        say("\n== J. CURVATURE CONCENTRATION from Adam-v (ckpt_latest opt_state; PR/n = participation ratio per parameter; k90 = fraction of parameters carrying 90 % of the curvature mass) ==")
        CUR = {}
        for x in present + ["X0"]:
            p = (pdir(x) if x != "X0" else X0["pre"]) / "ckpt_latest.pkl"
            if not p.exists(): continue
            try: r = curvature_row(p)
            except Exception as ex: r = None; say(f"  {x}: curvature failed ({type(ex).__name__}: {ex})")
            if r: CUR[x] = r; say(f"  {x}: n {r['n']} | PR/n {r['pr_n']:.2e} | k90 {100*r['k90']:.2f} % | top blocks {r['top']}")
        J["curvature"] = CUR
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text("\n".join(L) + "\n"); OUTJ.write_text(json.dumps(J, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating,)) else (int(o) if isinstance(o, np.integer) else str(o))))
    say(f"\nartifact -> {OUT} (+ .json)")

if __name__ == "__main__":
    main()
