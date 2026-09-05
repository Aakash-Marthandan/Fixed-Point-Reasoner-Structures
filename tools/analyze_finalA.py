# Ledger: FINAL PHASE Night A ANALYZER — written BEFORE any Night A run (2026-09-05; Plan_2026-09-05_FinalPhase
# §3 Night A / §4); the launch registration locks these rules verbatim. Arms (one variable each, the field regime,
# 50k SOT steps, EMA headline at D16 = EqR's base column): A0 (X0 seed 2: the NOISE FLOOR pair with sportC1's X0
# seed 0) · A1 (X0 + FPA anchor rows) · A2 (X0 + RI sigma 1) · A3 (DEC-w256, no digit aug: THE arm) · A4 (DEC +
# digit aug: the redundancy control) · A5 (DEC + FPA + RI) · A6 (DEC-w512 = X0's parameter class, --remat, no digit
# aug: the WIDTH de-confound; runs on the 16's fourth worker only — absent on the 8-shape or after a preflight failure,
# labeled; registration AMENDED pre-data 2026-09-05 on the PI's adversarial pass).
#   REFERENCES (the record, sportC1 verdict 2026-09-03 / sportC2 verdict 2026-09-05): X0 cold@D16 .8603, @D64 .9281,
#     b1 .9212, t1r@128 .9248, verified@128 .9970; X1 (X0 - digit aug) cold@D16 .2120; EqR base 84.8 / D64 93.0;
#     TRM-MLP 87.4. X0's spurious rate is READ from its banked sportC1 scan records when present (else the lens's
#     E5 value .078, labeled).
#   INTEGRITY (a breach WITHHOLDS the verdict): fulls n == 422,786 (vsel/final at D16, vsel at D64); scans n ==
#     20,000, unique idx, identical idx sets, protocol (t64/k128/seed 20260822), vote identity; every vsel-labeled
#     eval of an arm reports ONE checkpoint path (the sportC1 lesson).
#   CLEAN split (the 2026-09-03 registration-gate lesson): STABILITY := not STOPPED AND census(vsel, t64) <= .02;
#     MEMORIZATION (labeled, never disqualifying the val-selected grid) := end segment-CE < .02 OR (vsel - final) > .05.
#   NOISE FLOOR := max(|cold16(A0) - .8603|, .005) — a SEED PAIR, never a treatment (the sportC2 CNC2 lesson);
#     A0 absent -> .01, labeled FLOOR-DEFAULT. A contrast is READ only beyond 2 x FLOOR; else FLAT.
#   SPURIOUS(arm) := the per-draw spurious-attractor rate on the 20k k128 scan (lens G / E5 definition): wrong draws
#     whose latent residual lies below the median residual of the correct draws.
#   R-A-0 ORBIT (A3 vs X1, A4 vs A3): cold16(A3) - .2120 >= .40 AND |cold16(A4) - cold16(A3)| <= max(.02, 2 FLOOR)
#     -> EXACT-S9-REPLACES-AUG; >= .40 AND A4 - A3 > that -> EXACT-S9-PARTIAL; >= .40 AND A4 below A3 by more than
#     that -> EXACT-S9-REPLACES-AUG+AUG-HURTS; < .40 -> EXACT-S9-FAILS. Tags: +PARITY-AT-D16 iff cold16(A3) >= .848;
#     +PARITY iff cold64(A3) >= .920.
#   R-A-1 SELECTOR (A1): SPURIOUS(A1) <= .01 AND t1r@128(A1) >= .97 -> LIFT (-FREE if |cold16(A1) - .8603| <= 2 FLOOR,
#     -AT-COST if below, -AND-GAIN if above); SPURIOUS <= .01 alone -> LIFT-PARTIAL; else LIFT-FLAT.
#   R-A-2 RI (A2): dc = cold16(A2) - .8603, dr = verified@128(A2) - .9970: HELPS iff (dc > 2 FLOOR and dr >= -.01)
#     or (dr > .01 and dc >= -2 FLOOR); HURTS iff dc < -2 FLOOR or dr < -.01; else FLAT.
#   R-A-3 OBJECTIVES (A5 vs A3): d = cold16(A5) - cold16(A3): HELPS (> 2 FLOOR) / HURTS (< -2 FLOOR) / FLAT;
#     +SELECTOR-CLEAN iff SPURIOUS(A5) <= .01.
#   R-A-4 DECODER-CLASS (every arm, from the scan's cold column): g50 = the logistic 50 % crossing of cold exact vs
#     givens (None if no crossing in 17..35 or no rising fit); YIELD = cold solve rate on rating > 0 puzzles.
#     DECIMATING iff g50 is None AND YIELD >= .70; SOFT iff g50 >= 24 AND YIELD <= .45; else MIXED.
#   R-A-6 WIDTH (A6 vs A3): d = cold16(A6) - cold16(A3): WIDTH-HELPS (> 2 FLOOR) / HURTS (< -2 FLOOR) / FLAT;
#     tags +PARITY-AT-D16 iff cold16(A6) >= .848; +PARITY iff cold64(A6) >= .920; A6 absent -> NO-DATA (labeled).
#   R-A-5 CARRY-TO-B (mechanical): CHAMP = argmax cold16 over STABLE non-memorized DEC arms {A3, A4, A5, A6};
#     B0 := A5's objective set iff R-A-3 in {HELPS, FLAT} and SPURIOUS(A5) <= .01 (missing -> A3's) else A3's;
#     W := 512 iff R-A-6 reads WIDTH-HELPS else 256; AUG := carried iff R-A-0 in {FAILS, PARTIAL} else dropped.
#   PREDICTIONS (pre-data): A0 cold16 [.850, .870]; A1 SPURIOUS < .01 and t1r >= .97, cold16 [.845, .875]; A2 cold16
#     [.850, .870]; A3 cold16 [.70, .90]; A4 - A3 [-.02, .02]; A5 - A3 [-.01, .03]; A6 cold16 [.78, .93], A6 - A3 [0, .08].
"""
  .venv/bin/python tools/analyze_finalA.py            # -> runs/analysis/finalA_verdict.txt
  .venv/bin/python tools/analyze_finalA.py --selftest
"""
from __future__ import annotations
import contextlib, io, json, os, sys, tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(os.environ.get("QHRRN_RUNS", ROOT / "runs"))
NPZ = Path(os.environ.get("QHRRN_NPZ", ROOT / "data/sudoku_extreme/sudoku_extreme_seed0.npz"))
OUT = RUNS / "analysis" / "finalA_verdict.txt"
TAG = "finalA"
X_ARMS = ["A0", "A1", "A2"]; DEC_ARMS = ["A3", "A4", "A5", "A6"]; ARMS = X_ARMS + DEC_ARMS
X0_COLD16, X0_COLD64, X0_B1, X0_T1R, X0_V128, X1_COLD16 = 0.8603, 0.9281, 0.9212, 0.9248, 0.9970, 0.2120
EQR_BASE, EQR_D64, TRM_MLP, X0_SPUR_CONST = 0.848, 0.930, 0.874, 0.078
N_FULL, N_SCAN = 422786, 20000
SCAN_PROTO = dict(t_total=64, k_init=128, subsample_seed=20260822)
HEADLINE_T = 16
PRED = {"A0": dict(cold16=(.850, .870)), "A1": dict(cold16=(.845, .875), spur=(0.0, .01), t1r=(.97, 1.0)), "A2": dict(cold16=(.850, .870)),
        "A3": dict(cold16=(.70, .90)), "A4": dict(d_a3=(-.02, .02)), "A5": dict(d_a3=(-.01, .03)), "A6": dict(cold16=(.78, .93), d_a3=(0.0, .08))}
DESC = {"A0": "X0 seed 2 (noise floor pair)", "A1": "X0 + FPA anchor rows", "A2": "X0 + RI sigma 1",
        "A3": "DEC-w256, no digit aug", "A4": "DEC-w256 + digit aug (control)", "A5": "DEC-w256 + FPA + RI", "A6": "DEC-w512 (X0's class), remat"}
LINES = []
def say(s=""): LINES.append(str(s)); print(s)
def jload(p):
    p = Path(p); return json.loads(p.read_text()) if p.exists() else None
def fpp(x): return "  -  " if x is None else f"{100*x:6.2f}"
def hm(x, band): return "n/a" if x is None else ("HIT" if band[0] <= x <= band[1] else ("ABOVE" if x > band[1] else "MISS-BELOW"))

# ---------- readers ----------
def pdir(a): return RUNS / f"pretrain{TAG}_{a}"
def evdir(a): return RUNS / f"sxeval_p{TAG}{a}"
def full(a, which, t=HEADLINE_T): return jload(evdir(a) / f"full_{which}_t{t}" / "summary_all.json")
def cold16(a): s = full(a, "vsel"); return None if not s else s["exact_acc"]
def cold64(a): s = full(a, "vsel", 64); return None if not s else s["exact_acc"]
def fcold(a): s = full(a, "final"); return None if not s else s["exact_acc"]
def scan(a): return jload(RUNS / f"sxscan_p{TAG}{a}" / "summary_all.json")
def scan_recs(a):
    q = RUNS / f"sxscan_p{TAG}{a}" / "records_all.npz"; return dict(np.load(q, allow_pickle=True)) if q.exists() else None
def b1(a): s = scan(a); return None if not s else s.get("b1_exact")
def v128(a): s = scan(a); return None if not s else s.get("vote_at_k", {}).get("128")
def t1r(a): s = scan(a); return None if not s else s.get("t1r_at_k", {}).get("128")
def stopped(a):
    p = pdir(a) / "STOPPED.txt"; return p.read_text().strip().splitlines()[0][:44] if p.exists() else None
def census(a, which="vsel", t=64):
    c = jload(RUNS / f"sxcensus_p{TAG}{a}_{which}" / "census.json")
    if not c: return None
    rows = [r for r in c["rows"] if int(r["t"]) == t]; return None if not rows else rows[0]["exploded_frac"]
def calib(a): return jload(RUNS / f"sxcalib_p{TAG}{a}_vsel" / "calib.json")
def metrics(a):
    p = pdir(a) / "metrics.jsonl"
    if not p.exists(): return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def end_ce(a):
    rows = [r for r in metrics(a) if "loss" in r]
    if not rows: return None
    tail = rows[-5:]; return float(np.mean([r.get("ce_in", np.nan) for r in tail]))
def val_peak_step(a):
    mons = [r["monitor"] for r in metrics(a) if "monitor" in r]
    key = "val_t16_ema"
    mons = [m for m in mons if key in m]
    return None if not mons else int(max(mons, key=lambda m: m[key])["step"])
def spurious(a, recs=None):
    """Lens G / E5: the per-draw spurious-attractor rate — wrong draws whose residual is below the median residual
    of the correct draws (None without the scan's per-draw records)."""
    z = recs if recs is not None else scan_recs(a)
    if z is None or "mi_exact_k" not in z or "mi_resid_k" not in z: return None
    ex = np.asarray(z["mi_exact_k"]).astype(bool); rs = np.asarray(z["mi_resid_k"]).astype(np.float64); fin = np.isfinite(rs)
    ef, wf = ex & fin, (~ex) & fin
    if ef.sum() == 0 or wf.sum() == 0: return None
    return float(np.mean(rs[wf] <= np.median(rs[ef])))
def x0_spurious():
    q = RUNS / "sxscan_psportC1X0" / "records_all.npz"
    if q.exists():
        v = spurious("X0", dict(np.load(q, allow_pickle=True)))
        if v is not None: return v, "read"
    return X0_SPUR_CONST, "lens-E5 constant"
def npz_cols():
    try:
        z = np.load(NPZ, allow_pickle=True); q = z["test_q"]
        return (q != 0).reshape(len(q), -1).sum(1), np.asarray(z["test_rating"])
    except Exception: return None, None
def logistic_g50(g, y, lo=17, hi=35):
    """The logistic 50 % crossing of cold exact vs givens (Newton IRLS on standardized givens); None if no
    rising fit or the crossing lies outside [lo, hi] (= no threshold in the tested range)."""
    g = np.asarray(g, float); y = np.asarray(y, float)
    if len(g) < 50 or y.mean() <= 0.0 or y.mean() >= 1.0: return None
    mu, sd = g.mean(), g.std() + 1e-9; x = (g - mu) / sd; a = b = 0.0
    for _ in range(40):
        p = 1.0 / (1.0 + np.exp(-(a + b * x))); w = p * (1 - p) + 1e-6
        G = np.array([np.sum(y - p), np.sum((y - p) * x)])
        H = np.array([[np.sum(w), np.sum(w * x)], [np.sum(w * x), np.sum(w * x * x)]]) + 1e-6 * np.eye(2)
        d = np.linalg.solve(H, G); a += d[0]; b += d[1]
        if np.max(np.abs(d)) < 1e-8: break
    if b <= 0.05: return None
    g50 = (-a / b) * sd + mu
    return float(g50) if lo <= g50 <= hi else None
def decoder_class(a, giv, rat):
    z = scan_recs(a)
    if z is None or giv is None or "cold_exact" not in z: return "NO-DATA", None, None
    idx = z["idx"]; cold = z["cold_exact"].astype(bool)
    g50 = logistic_g50(giv[idx], cold); m = rat[idx] > 0
    yld = None if m.sum() == 0 else float(cold[m].mean())
    if yld is None: return "NO-DATA", g50, None
    if g50 is None and yld >= .70: return "DECIMATING", g50, yld
    if g50 is not None and g50 >= 24 and yld <= .45: return "SOFT", g50, yld
    return "MIXED", g50, yld
def ckpt_paths(a):
    t = HEADLINE_T; srcs = {}
    for nm, p in (("full_vsel", evdir(a) / f"full_vsel_t{t}" / "summary_all.json"), ("full_alt", evdir(a) / f"full_vsel_t{t}_alt" / "summary_all.json"),
                  ("full_vsel_t64", evdir(a) / "full_vsel_t64" / "summary_all.json"),
                  ("scan", RUNS / f"sxscan_p{TAG}{a}" / "summary_all.json"), ("census_vsel", RUNS / f"sxcensus_p{TAG}{a}_vsel" / "census.json"),
                  ("screen_vb", RUNS / f"sxscreen_p{TAG}{a}_vb" / "summary_all.json"), ("calib", RUNS / f"sxcalib_p{TAG}{a}_vsel" / "calib.json")):
        s = jload(p)
        if s and s.get("ckpt"): srcs[nm] = s["ckpt"]
    return srcs
def stable(a):
    if stopped(a): return False
    c = census(a); return c is not None and c <= .02
def memorized(a):
    ce = end_ce(a); vc, fc = cold16(a), fcold(a)
    tags = []
    if ce is not None and ce < .02: tags.append("END-CE")
    if vc is not None and fc is not None and vc - fc > .05: tags.append("VSEL-FINAL-DROP")
    return tags

# ---------- integrity ----------
def integrity():
    errs = []
    for a in ARMS:
        for which, t in (("vsel", 16), ("final", 16), ("vsel", 64)):
            s = full(a, which, t)
            if s and s["n"] != N_FULL: errs.append(f"{a} full_{which}_t{t} n={s['n']}")
        paths = set(ckpt_paths(a).values())
        if len(paths) > 1: errs.append(f"{a} vsel evals on {len(paths)} different grids: {sorted(paths)}")
    idx_sets = {}
    for a in ARMS:
        s = scan(a)
        if not s: continue
        if s["n"] != N_SCAN: errs.append(f"{a} scan n={s['n']}")
        for k, v in SCAN_PROTO.items():
            if s.get(k) != v: errs.append(f"{a} scan {k}={s.get(k)}!={v}")
        z = scan_recs(a)
        if z is not None:
            if len(np.unique(z["idx"])) != len(z["idx"]): errs.append(f"{a} scan dup idx")
            idx_sets[a] = frozenset(z["idx"].tolist())
            hit = (z["mi_first_hit"] >= 0) | z["cold_exact"].astype(bool)
            if abs(float(hit.mean()) - s["exact_acc_vote"]) > 5e-6: errs.append(f"{a} vote identity breach")
    if len(set(idx_sets.values())) > 1: errs.append("scan idx sets differ across arms")
    return errs

# ---------- verdict ----------
def analyze():
    V = {}
    say(f"== FINAL PHASE Night A VERDICT ({TAG}; rules locked pre-data in this file's header) ==")
    errs = integrity(); V["INTEGRITY"] = "PASS" if not errs else "FAIL"
    say(f"INTEGRITY: {V['INTEGRITY']}" + ("" if not errs else " — " + "; ".join(errs)))
    giv, rat = npz_cols(); xs, xs_src = x0_spurious()
    say(f"\nARM TABLE (headline = EMA at D16; cold64 = the D64 depth row; spurious = lens-E5 per-draw rate; X0 refs: cold16 {X0_COLD16:.4f} cold64 {X0_COLD64:.4f} t1r {X0_T1R:.4f} v128 {X0_V128:.4f} spurious {xs:.3f} [{xs_src}]; X1 cold16 {X1_COLD16:.4f})")
    say("  arm | description                     | cold16 | final16 | cold64 |   b1   |  t1r128 | v128   | spur  | census | end-CE | stable | memorized | class (g50 / yield)")
    rows = {}
    for a in ARMS:
        cls, g50, yld = decoder_class(a, giv, rat)
        rows[a] = dict(cold16=cold16(a), fcold=fcold(a), cold64=cold64(a), b1=b1(a), t1r=t1r(a), v128=v128(a), spur=spurious(a),
                       census=census(a), ce=end_ce(a), stable=stable(a), mem=memorized(a), cls=cls, g50=g50, yld=yld, stopped=stopped(a))
        r = rows[a]
        say(f"  {a:3s} | {DESC[a]:31s} | {fpp(r['cold16'])} | {fpp(r['fcold'])} | {fpp(r['cold64'])} | {fpp(r['b1'])} | {fpp(r['t1r'])} | {fpp(r['v128'])} | "
            f"{'  -  ' if r['spur'] is None else f'{r['spur']:.3f}'} | {'  -  ' if r['census'] is None else f'{r['census']:.3f}'} | {'  -  ' if r['ce'] is None else f'{r['ce']:.3f}'} | "
            f"{'yes' if r['stable'] else ('STOPPED' if r['stopped'] else 'no')} | {'+'.join(r['mem']) or '-'} | {cls} ({'none' if g50 is None else f'{g50:.1f}'} / {'-' if yld is None else f'{100*yld:.1f}'})")
    V["STABILITY"] = "ALL-STABLE" if all(rows[a]["stable"] for a in ARMS if rows[a]["cold16"] is not None) and any(rows[a]["cold16"] is not None for a in ARMS) \
        else "UNSTABLE:" + ",".join(a for a in ARMS if rows[a]["cold16"] is not None and not rows[a]["stable"])
    mem = [a for a in ARMS if rows[a]["mem"]]; V["MEMORIZATION"] = "NONE" if not mem else "MEMORIZED:" + ",".join(mem)
    say(f"\nSTABILITY: {V['STABILITY']}   MEMORIZATION: {V['MEMORIZATION']}")
    if errs:
        say("\nVERDICT WITHHELD: integrity breach (fix the data or the reader, never the rule)."); _write(); return V
    c = {a: rows[a]["cold16"] for a in ARMS}
    if c["A0"] is not None: FLOOR = max(abs(c["A0"] - X0_COLD16), .005); fsrc = "seed pair"
    else: FLOOR = .01; fsrc = "FLOOR-DEFAULT (A0 absent)"
    V["FLOOR"] = FLOOR; say(f"\nNOISE FLOOR = {FLOOR:.4f} ({fsrc}); contrasts read beyond 2 x FLOOR = {2*FLOOR:.4f}")
    # R-A-0 ORBIT
    if c["A3"] is None: V["R-A-0"] = "NO-DATA"
    else:
        d = c["A3"] - X1_COLD16; tol = max(.02, 2 * FLOOR)
        if d < .40: V["R-A-0"] = "EXACT-S9-FAILS"
        elif c["A4"] is None: V["R-A-0"] = "EXACT-S9-REPLACES-AUG?(A4 absent)"
        elif abs(c["A4"] - c["A3"]) <= tol: V["R-A-0"] = "EXACT-S9-REPLACES-AUG"
        elif c["A4"] - c["A3"] > tol: V["R-A-0"] = "EXACT-S9-PARTIAL"
        else: V["R-A-0"] = "EXACT-S9-REPLACES-AUG+AUG-HURTS"
        if c["A3"] >= EQR_BASE: V["R-A-0"] += "+PARITY-AT-D16"
        if rows["A3"]["cold64"] is not None and rows["A3"]["cold64"] >= .920: V["R-A-0"] += "+PARITY"
    say(f"R-A-0 ORBIT (A3 - X1 = {'-' if c['A3'] is None else f'{c['A3'] - X1_COLD16:+.4f}'}; A4 - A3 = {'-' if None in (c['A3'], c['A4']) else f'{c['A4'] - c['A3']:+.4f}'}): {V['R-A-0']}")
    # R-A-1 SELECTOR
    s1, t1 = rows["A1"]["spur"], rows["A1"]["t1r"]
    if s1 is None or t1 is None or c["A1"] is None: V["R-A-1"] = "NO-DATA"
    elif s1 <= .01 and t1 >= .97:
        dc = c["A1"] - X0_COLD16
        V["R-A-1"] = "LIFT-FREE" if abs(dc) <= 2 * FLOOR else ("LIFT-AT-COST" if dc < 0 else "LIFT-AND-GAIN")
    elif s1 <= .01: V["R-A-1"] = "LIFT-PARTIAL"
    else: V["R-A-1"] = "LIFT-FLAT"
    say(f"R-A-1 SELECTOR (A1 spurious {'-' if s1 is None else f'{s1:.3f}'} vs X0 {xs:.3f}; t1r {fpp(t1)} vs X0 {100*X0_T1R:.2f}; cold16 delta {'-' if c['A1'] is None else f'{c['A1'] - X0_COLD16:+.4f}'}): {V['R-A-1']}")
    # R-A-2 RI
    if c["A2"] is None or rows["A2"]["v128"] is None: V["R-A-2"] = "NO-DATA"
    else:
        dc, dr = c["A2"] - X0_COLD16, rows["A2"]["v128"] - X0_V128
        if (dc > 2 * FLOOR and dr >= -.01) or (dr > .01 and dc >= -2 * FLOOR): V["R-A-2"] = "RI-HELPS"
        elif dc < -2 * FLOOR or dr < -.01: V["R-A-2"] = "RI-HURTS"
        else: V["R-A-2"] = "RI-FLAT"
    say(f"R-A-2 RI (A2 cold16 delta {'-' if c['A2'] is None else f'{c['A2'] - X0_COLD16:+.4f}'}; v128 delta {'-' if rows['A2']['v128'] is None else f'{rows['A2']['v128'] - X0_V128:+.4f}'}): {V['R-A-2']}")
    # R-A-3 OBJECTIVES
    if None in (c["A3"], c["A5"]): V["R-A-3"] = "NO-DATA"
    else:
        d = c["A5"] - c["A3"]
        V["R-A-3"] = "OBJ-HELPS" if d > 2 * FLOOR else ("OBJ-HURTS" if d < -2 * FLOOR else "OBJ-FLAT")
        if rows["A5"]["spur"] is not None and rows["A5"]["spur"] <= .01: V["R-A-3"] += "+SELECTOR-CLEAN"
    say(f"R-A-3 OBJECTIVES (A5 - A3 = {'-' if None in (c['A3'], c['A5']) else f'{c['A5'] - c['A3']:+.4f}'}; A5 spurious {'-' if rows['A5']['spur'] is None else f'{rows['A5']['spur']:.3f}'}): {V['R-A-3']}")
    # R-A-4 DECODER-CLASS
    V["R-A-4"] = " ".join(f"{a}:{rows[a]['cls']}" for a in ARMS)
    say(f"R-A-4 DECODER-CLASS: {V['R-A-4']}")
    # R-A-6 WIDTH
    if None in (c["A3"], c["A6"]): V["R-A-6"] = "NO-DATA" + ("" if c["A6"] is not None else " (A6 absent: 8-shape or preflight-skipped, labeled)")
    else:
        d = c["A6"] - c["A3"]
        V["R-A-6"] = "WIDTH-HELPS" if d > 2 * FLOOR else ("WIDTH-HURTS" if d < -2 * FLOOR else "WIDTH-FLAT")
        if c["A6"] >= EQR_BASE: V["R-A-6"] += "+PARITY-AT-D16"
        if rows["A6"]["cold64"] is not None and rows["A6"]["cold64"] >= .920: V["R-A-6"] += "+PARITY"
    say(f"R-A-6 WIDTH (A6 - A3 = {'-' if None in (c['A3'], c['A6']) else f'{c['A6'] - c['A3']:+.4f}'}; A6 cold64 {fpp(rows['A6']['cold64'])}): {V['R-A-6']}")
    # R-A-5 CARRY-TO-B
    cands = [a for a in DEC_ARMS if rows[a]["stable"] and not rows[a]["mem"] and c[a] is not None]
    champ = max(cands, key=lambda a: c[a]) if cands else None
    b0 = "A5" if (V["R-A-3"].startswith(("OBJ-HELPS", "OBJ-FLAT")) and rows["A5"]["spur"] is not None and rows["A5"]["spur"] <= .01) else "A3"
    width = 512 if V["R-A-6"].startswith("WIDTH-HELPS") else 256
    aug = "carried" if V["R-A-0"].startswith(("EXACT-S9-FAILS", "EXACT-S9-PARTIAL")) else "dropped"
    V["R-A-5"] = f"CHAMP:{champ or 'none'}|B0:={b0}|W:={width}|AUG:{aug}"
    say(f"R-A-5 CARRY-TO-B: {V['R-A-5']}" + (f" (champion cold16 {100*c[champ]:.2f}; memorized/unstable DEC arms excluded: {[a for a in DEC_ARMS if a not in cands]})" if champ else ""))
    # predictions
    say("\nPREDICTION SCOREBOARD (bands locked pre-data):")
    sc = {"A0": hm(c["A0"], PRED["A0"]["cold16"]), "A1": f"cold {hm(c['A1'], PRED['A1']['cold16'])} spur {hm(s1, PRED['A1']['spur'])} t1r {hm(t1, PRED['A1']['t1r'])}",
          "A2": hm(c["A2"], PRED["A2"]["cold16"]), "A3": hm(c["A3"], PRED["A3"]["cold16"]),
          "A4": hm(None if None in (c["A3"], c["A4"]) else c["A4"] - c["A3"], PRED["A4"]["d_a3"]),
          "A5": hm(None if None in (c["A3"], c["A5"]) else c["A5"] - c["A3"], PRED["A5"]["d_a3"]),
          "A6": f"cold {hm(c['A6'], PRED['A6']['cold16'])} d_a3 {hm(None if None in (c['A3'], c['A6']) else c['A6'] - c['A3'], PRED['A6']['d_a3'])}"}
    for a in ARMS: say(f"  {a}: {sc[a]}")
    V["PRED"] = sc
    say("\nLETTERS: " + " · ".join(f"{k} {V[k]}" for k in ("INTEGRITY", "STABILITY", "MEMORIZATION", "R-A-0", "R-A-1", "R-A-2", "R-A-3", "R-A-6", "R-A-5")))
    _write(); return V

def _write():
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text("\n".join(LINES) + "\n"); LINES.clear()

# ---------- selftest ----------
def _npz(root, n=N_SCAN, seed=0):
    rng = np.random.default_rng(seed)
    giv = rng.integers(17, 36, size=n)
    q = np.zeros((n, 9, 9), np.int8)
    for i in range(n):
        cells = rng.choice(81, size=int(giv[i]), replace=False); q[i].reshape(-1)[cells] = 1
    np.savez(root / "npz.npz", test_q=q, test_rating=np.where(rng.random(n) < .15, 0, rng.integers(1, 200, size=n)))
    return giv
def _mk(root, a, giv, *, vc=None, fc=None, c64=None, b1v=None, v128v=None, t1rv=None, spur=0.005, ce=0.5, census_=0.0, stopped_=None,
        n_scan=N_SCAN, ckpt="ckpt_040000.pkl", scan_ckpt=None, soft=False, seed=1):
    rng = np.random.default_rng(seed); ev = root / f"sxeval_p{TAG}{a}"; ck = f"runs/pretrain{TAG}_{a}/{ckpt}"
    for which, v, t in (("vsel", vc, 16), ("final", fc, 16), ("vsel", c64, 64)):
        if v is not None:
            d = ev / f"full_{which}_t{t}"; d.mkdir(parents=True, exist_ok=True)
            (d / "summary_all.json").write_text(json.dumps(dict(exact_acc=v, n=N_FULL, ckpt=ck)))
    if b1v is not None:
        d = root / f"sxscan_p{TAG}{a}"; d.mkdir(parents=True, exist_ok=True); n = n_scan; idx = np.arange(n)   # unique: the integrity gate
        if soft: cold = ((giv[idx] >= 26) & (rng.random(n) < .6)) ^ (rng.random(n) < .03)   # a soft threshold decoder (~26 givens, plateau .6)
        else: cold = rng.random(n) < (vc or .3)                              # givens-independent
        hit = np.full(n, -1); nh = int((v128v or .5) * n); hit[:nh] = 3
        vote = float(((hit >= 0) | cold).mean())
        k = 8; ex = np.zeros((n, k), bool); ex[:, 0] = cold; ex[:, 1:] = rng.random((n, k - 1)) < .5
        rs = np.where(ex, rng.random((n, k)) * .01, .05 + rng.random((n, k)) * .05)     # correct draws converge lower
        nw = int(spur * (~ex).sum()); wi = np.argwhere(~ex)[:nw]
        for i, j in wi: rs[i, j] = .002                                                 # spurious: wrong AND converged
        (d / "summary_all.json").write_text(json.dumps(dict(n=n, t_total=64, k_init=128, subsample_seed=20260822, b1_exact=b1v, exact_acc=float(cold.mean()),
                                                            exact_acc_vote=vote, vote_at_k={"128": vote}, t1r_at_k={"128": t1rv if t1rv is not None else vote * .99}, ckpt=(scan_ckpt or ck))))
        np.savez(d / "records_all.npz", idx=idx, cold_exact=cold, mi_first_hit=hit, mi_exact_k=ex, mi_resid_k=rs)
    if census_ is not None:
        d = root / f"sxcensus_p{TAG}{a}_vsel"; d.mkdir(parents=True, exist_ok=True)
        (d / "census.json").write_text(json.dumps(dict(ckpt=ck, rows=[dict(t=64, exploded_frac=census_), dict(t=256, exploded_frac=census_)])))
    d = root / f"sxcalib_p{TAG}{a}_vsel"; d.mkdir(parents=True, exist_ok=True); (d / "calib.json").write_text(json.dumps(dict(ckpt=ck, topk_correct_stalled=.7)))
    pd = root / f"pretrain{TAG}_{a}"; pd.mkdir(parents=True, exist_ok=True)
    if stopped_: (pd / "STOPPED.txt").write_text(stopped_)
    rows = [json.dumps(dict(step=s, loss=.3, ce_in=ce)) for s in (46000, 48000, 50000)]
    rows += [json.dumps({"monitor": {"step": 20000, "val_t16_ema": .70}}), json.dumps({"monitor": {"step": 40000, "val_t16_ema": .80}})]
    (pd / "metrics.jsonl").write_text("\n".join(rows) + "\n")

def selftest():
    checks = []
    def run(build):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); giv = _npz(root); build(root, giv)
            globals()["RUNS"] = root; globals()["OUT"] = root / "a" / "v.txt"; globals()["NPZ"] = root / "npz.npz"
            with contextlib.redirect_stdout(io.StringIO()): return analyze()
    def wA(r, g):  # the hoped-for world: exact S9 replaces aug at parity; the selector lifts free; RI helps; objectives help; decimating class
        _mk(r, "A0", g, vc=.858, fc=.858, c64=.925, b1v=.92, v128v=.997, t1rv=.925, spur=.07)
        _mk(r, "A1", g, vc=.862, fc=.862, c64=.93, b1v=.92, v128v=.997, t1rv=.985, spur=.004)
        _mk(r, "A2", g, vc=.875, fc=.875, c64=.93, b1v=.92, v128v=.999, t1rv=.93, spur=.06)
        _mk(r, "A3", g, vc=.85, fc=.85, c64=.925, b1v=.91, v128v=.996, t1rv=.92, spur=.05)
        _mk(r, "A4", g, vc=.855, fc=.855, c64=.93, b1v=.91, v128v=.996, t1rv=.92, spur=.05)
        _mk(r, "A5", g, vc=.87, fc=.87, c64=.935, b1v=.92, v128v=.998, t1rv=.99, spur=.003)
        _mk(r, "A6", g, vc=.89, fc=.89, c64=.94, b1v=.93, v128v=.998, t1rv=.93, spur=.05)
    v = run(wA)
    checks += [("A integrity", v["INTEGRITY"] == "PASS"), ("A floor = the .005 clamp (|A0 - X0| = .0023 below it)", v["FLOOR"] == .005),
               ("A orbit exact + parity at D16 + parity", v["R-A-0"] == "EXACT-S9-REPLACES-AUG+PARITY-AT-D16+PARITY"),
               ("A selector lift free", v["R-A-1"] == "LIFT-FREE"), ("A RI helps", v["R-A-2"] == "RI-HELPS"),
               ("A objectives help + clean", v["R-A-3"] == "OBJ-HELPS+SELECTOR-CLEAN"), ("A decimating class on every arm", all(x.endswith("DECIMATING") for x in v["R-A-4"].split())),
               ("A width helps + parity on the matched arm", v["R-A-6"] == "WIDTH-HELPS+PARITY-AT-D16+PARITY"),
               ("A carry: champ A6, B0 := A5's objectives at w512, aug dropped", v["R-A-5"] == "CHAMP:A6|B0:=A5|W:=512|AUG:dropped"),
               ("A all stable, none memorized", v["STABILITY"] == "ALL-STABLE" and v["MEMORIZATION"] == "NONE")]
    def wB(r, g):  # kills: symmetry fails (A3 low, soft class, memorized); selector flat; RI hurts; objectives flat with a dirty selector; A2 stopped
        _mk(r, "A0", g, vc=.852, fc=.852, c64=.92, b1v=.92, v128v=.997, t1rv=.92, spur=.07)
        _mk(r, "A1", g, vc=.86, fc=.86, c64=.93, b1v=.92, v128v=.997, t1rv=.92, spur=.06)
        _mk(r, "A2", g, vc=.80, fc=.80, c64=.90, b1v=.85, v128v=.98, t1rv=.90, spur=.07, stopped_="STOPPED final step 30000 (NaN halt)")
        _mk(r, "A3", g, vc=.45, fc=.30, c64=.50, b1v=.44, v128v=.60, t1rv=.55, spur=.05, ce=.001, soft=True)
        _mk(r, "A4", g, vc=.86, fc=.86, c64=.93, b1v=.91, v128v=.996, t1rv=.92, spur=.05)
        _mk(r, "A5", g, vc=.44, fc=.44, c64=.50, b1v=.43, v128v=.60, t1rv=.55, spur=.05, soft=True)
    v = run(wB)
    checks += [("B orbit fails", v["R-A-0"] == "EXACT-S9-FAILS"), ("B selector flat", v["R-A-1"] == "LIFT-FLAT"), ("B RI hurts", v["R-A-2"] == "RI-HURTS"),
               ("B objectives flat, selector dirty", v["R-A-3"] == "OBJ-FLAT"), ("B A3 soft class", "A3:SOFT" in v["R-A-4"] and "A5:SOFT" in v["R-A-4"]),
               ("B A3 memorized (end-CE + drop) labeled", v["MEMORIZATION"] == "MEMORIZED:A3"), ("B A2 unstable (STOPPED)", v["STABILITY"] == "UNSTABLE:A2"),
               ("B width no-data (A6 absent, labeled)", v["R-A-6"].startswith("NO-DATA")),
               ("B carry: champ A4 (A3 memorized excluded), B0 := A3, w256, aug carried", v["R-A-5"] == "CHAMP:A4|B0:=A3|W:=256|AUG:carried")]
    def wC(r, g):  # integrity breach: A1's scan on a different grid than its full
        wA(r, g); _mk(r, "A1", g, vc=.862, fc=.862, c64=.93, b1v=.92, v128v=.997, t1rv=.985, spur=.004, scan_ckpt=f"runs/pretrain{TAG}_A1/ckpt_030000.pkl")
    v = run(wC)
    checks += [("C grid-inconsistency withholds the verdict", v["INTEGRITY"] == "FAIL" and "R-A-0" not in v)]
    def wD(r, g):  # partial: A0 absent -> FLOOR-DEFAULT; A4 present but A3 - X1 short by a hair; A5 absent
        _mk(r, "A3", g, vc=.62, fc=.62, c64=.70, b1v=.60, v128v=.80, t1rv=.75, spur=.02)
        _mk(r, "A4", g, vc=.67, fc=.67, c64=.75, b1v=.65, v128v=.85, t1rv=.80, spur=.02)
    v = run(wD)
    checks += [("D floor default", v["FLOOR"] == .01), ("D orbit partial (aug still adds 5 pp)", v["R-A-0"] == "EXACT-S9-PARTIAL"),
               ("D no-data rules named", v["R-A-1"] == "NO-DATA" and v["R-A-2"] == "NO-DATA" and v["R-A-3"] == "NO-DATA"),
               ("D carry: champ A4, B0 := A3, w256, aug carried", v["R-A-5"] == "CHAMP:A4|B0:=A3|W:=256|AUG:carried")]
    v = run(lambda r, g: None)
    checks += [("E no data at all", v["R-A-0"] == "NO-DATA" and v["R-A-5"] == "CHAMP:none|B0:=A3|W:=256|AUG:dropped")]
    def wF(r, g):  # width flat: the wide arm adds nothing beyond the floor -> W := 256 carried
        wA(r, g); _mk(r, "A6", g, vc=.853, fc=.853, c64=.926, b1v=.92, v128v=.998, t1rv=.93, spur=.05)
    v = run(wF)
    checks += [("F width flat -> w256 carried, parity tags still on the matched arm", v["R-A-6"] == "WIDTH-FLAT+PARITY-AT-D16+PARITY" and "|W:=256|" in v["R-A-5"])]
    # the g50 fitter: a threshold decoder crosses at ~26; a givens-independent decoder has no crossing
    rng = np.random.default_rng(0); g = rng.integers(17, 36, size=5000)
    y_soft = (g >= 26) ^ (rng.random(5000) < .05); y_flat = rng.random(5000) < .9
    gs = logistic_g50(g, y_soft); checks += [("g50 of a 26-givens threshold decoder in [25, 27]", gs is not None and 25 <= gs <= 27), ("no g50 for a flat decoder", logistic_g50(g, y_flat) is None)]
    ok = 0
    for name, passed in checks:
        print(("  PASS  " if passed else "  FAIL  ") + name); ok += bool(passed)
    print(f"selftest: {ok}/{len(checks)}"); return ok == len(checks)


if __name__ == "__main__":
    if "--selftest" in sys.argv: sys.exit(0 if selftest() else 1)
    analyze()
