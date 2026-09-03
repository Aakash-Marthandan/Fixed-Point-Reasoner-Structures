# Ledger: sportC2 ANALYZER — the pre-parity graft night at d128, written BEFORE any sportC2 run
# (2026-09-04); the launch registration locks these rules verbatim (Plan_2026-09-04_sportC2.md).
# Arms: W0 (B0 + wd 1.0: the safe regime at our speed; the BASE of the grafts) · R1 (W0 + the
# persistent-carry loop, verifier-replaced rows) · R2 (W0 + inner cycles K=3, trained) · R3 (W0 +
# calibrated hard-decision rows, p=.5) · R4 (sportC1's R0 continued +50k in the field regime) ·
# X1 (X0 − digit augmentation) · X2 (X0 + our group mixer as token mixer). Every number is read at
# the arm's HEADLINE weights (raw on W0/R1/R2/R3; EMA on R4/X1/X2) from the val-selected grid.
#   INTEGRITY (a breach WITHHOLDS the verdict): full n == 422,786; scans n == 20,000, unique idx,
#     identical idx sets, protocol (t64/k128/seed 20260822), vote identity; and NEW (the sportC1
#     lesson): every vsel-labeled eval of an arm (full_vsel, alt, scan, census_vsel, screen_vb,
#     retfm, calib) reports ONE checkpoint path.
#   STABILITY (admission) := not STOPPED AND retfm >= .9 (natives) AND census(vsel, t64) <= .02.
#   MEMORIZATION (reported separately, never disqualifying the val-selected grid) := end CE < .02.
#   NOISE: CNC2 = max(|vcold(W0) - vcold(sportC1 B0 = .4283)|, .01) — the matched-recipe pair
#     across rounds (same seed, wd is the one variable), floor .01; FNC2 likewise on b1 (.02).
#   REACH(arm) := verified@128 on the 20k scan (= cold ∪ any-hit; the reachability statistic).
#   R-C2-0 REGIME (W0): end CE >= .05 AND val-peak step >= 25,000 (both stages) AND vcold >= .4283 + CNC2
#          -> WD-HOLDS; end CE < .02 -> WD-MEMORIZES; else WD-PARTIAL.
#   R-C2-1 CARRY (R1): REACH >= .70 -> CARRY-WIDENS; <= .55 -> CARRY-FLAT; else CARRY-PARTIAL;
#          + SELECTOR-INTACT iff t1r@128 / verified@128 >= .97 (spurious attractors absent).
#   R-C2-2 DEPTH (R2): median first_exact (solved, full test) <= 0.5 x W0's -> DEPTH-PROPAGATES;
#          reach on 21-25-givens puzzles >= .50 -> DEPTH-REACHES (both may hold); neither -> DEPTH-FLAT.
#   R-C2-3 COMMIT (R3): calib top-5 correct on stalled >= .90 -> COMMIT-CALIBRATED; <= .75 ->
#          COMMIT-FLAT; else COMMIT-PARTIAL; SPURIOUS-APPEAR iff t1r/verified < .97.
#   R-C2-4 CONTINUATION (R4): vcold(R4) - .3733 >= .03 -> REGIME-CLIMBS; >= .01 -> REGIME-SLOW; else REGIME-FLAT.
#   R-C2-5 ORBIT (X1): .8603 - cold@D16(X1) >= .03 -> ORBIT-LOAD-BEARING; <= .01 -> ORBIT-LEARNED; else PARTIAL.
#   R-C2-6 GEOMETRY (X2): cold@D16(X2) - .8603 >= .02 -> GEOMETRY-HELPS; <= -.02 -> GEOMETRY-HURTS; else NEUTRAL.
#   R-C2-7 CHAMPION-RECIPE (mechanical): the grafts {carry, inner cycles, hard rows} whose primary
#          letter is WIDENS / PROPAGATES-or-REACHES / CALIBRATED, on a STABLE arm with vcold >= vcold(W0) - CNC2,
#          are CARRIED to sportC3; wd 1.0 carried iff WD-HOLDS or WD-PARTIAL. Champion-so-far =
#          argmax vcold over STABLE non-memorized {W0,R1,R2,R3,R4} (memorized arms listed, labeled).
#   PREDICTIONS (pre-data): W0 vcold [.43,.50], CE end >= .05 (55 %); R1 REACH [.65,.85] (50 %), vcold
#     [.42,.50]; R2 first_exact median <= 6 (W0 ~10) (55 %), vcold [.44,.52]; R3 top-5 >= .90 (45 %),
#     vcold [.42,.50]; R4 vcold [.39,.44] (60 %); X1 cold@D16 [.80,.85] (a drop, 60 %); X2 [.84,.88] (neutral, 60 %).
"""
  .venv/bin/python tools/analyze_sportC2.py            # -> runs/analysis/sportC2_verdict.txt
  .venv/bin/python tools/analyze_sportC2.py --selftest
"""
from __future__ import annotations
import contextlib, io, json, os, sys, tempfile, re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(os.environ.get("QHRRN_RUNS", ROOT / "runs"))
NPZ = Path(os.environ.get("QHRRN_NPZ", ROOT / "data/sudoku_extreme/sudoku_extreme_seed0.npz"))
OUT = RUNS / "analysis" / "sportC2_verdict.txt"
TAG = "sportC2"
NATIVE = ["W0", "R1", "R2", "R3", "R4"]; FIELD = ["X1", "X2"]; ARMS = NATIVE + FIELD
B0_VCOLD, R0_VCOLD, X0_COLD16 = 0.4283, 0.3733, 0.8603
N_FULL, N_SCAN = 422786, 20000
SCAN_PROTO = dict(t_total=64, k_init=128, subsample_seed=20260822)
HEADLINE_T = {a: 64 for a in NATIVE} | {"X1": 16, "X2": 16}
HEAD_EMA = {"W0": False, "R1": False, "R2": False, "R3": False, "R4": True, "X1": True, "X2": True}
PRED = {"W0": dict(vcold=(.43, .50)), "R1": dict(reach=(.65, .85), vcold=(.42, .50)), "R2": dict(fe_med=(0, 6), vcold=(.44, .52)),
        "R3": dict(top5=(.90, 1.0), vcold=(.42, .50)), "R4": dict(vcold=(.39, .44)), "X1": dict(cold16=(.80, .85)), "X2": dict(cold16=(.84, .88))}
DESC = {"W0": "B0 + wd 1.0 (base)", "R1": "W0 + persistent carry (SOT-rg)", "R2": "W0 + inner cycles K=3", "R3": "W0 + hard rows p=.5",
        "R4": "sportC1 R0 +50k (field regime)", "X1": "X0 - digit aug", "X2": "X0 + our group mixer"}
LINES = []
def say(s=""): LINES.append(str(s)); print(s)
def jload(p):
    p = Path(p); return json.loads(p.read_text()) if p.exists() else None
def fpp(x): return "  -  " if x is None else f"{100*x:6.2f}"
def hm(x, band): return "n/a" if x is None else ("HIT" if band[0] <= x <= band[1] else ("ABOVE" if x > band[1] else "MISS-BELOW"))

# ---------- readers ----------
def pdir(a): return RUNS / f"pretrain{TAG}_{a}"
def evdir(a): return RUNS / f"sxeval_p{TAG}{a}"
def full(a, which, t=None):
    t = HEADLINE_T[a] if t is None else t; return jload(evdir(a) / f"full_{which}_t{t}" / "summary_all.json")
def vcold(a): s = full(a, "vsel"); return None if not s else s["exact_acc"]
def fcold(a): s = full(a, "final"); return None if not s else s["exact_acc"]
def scan(a): return jload(RUNS / f"sxscan_p{TAG}{a}" / "summary_all.json")
def scan_recs(a):
    q = RUNS / f"sxscan_p{TAG}{a}" / "records_all.npz"; return dict(np.load(q, allow_pickle=True)) if q.exists() else None
def full_recs(a):
    q = full_dir(a) / "records_all.npz"; return dict(np.load(q, allow_pickle=True)) if q.exists() else None
def full_dir(a): return evdir(a) / f"full_vsel_t{HEADLINE_T[a]}"
def b1(a): s = scan(a); return None if not s else s.get("b1_exact")
def v128(a): s = scan(a); return None if not s else s.get("vote_at_k", {}).get("128")
def t1r(a): s = scan(a); return None if not s else s.get("t1r_at_k", {}).get("128")
def retfm(a): s = jload(evdir(a) / "retfm_t8" / "summary_all.json"); return None if not s else s["exact_acc"]
def stopped(a):
    p = pdir(a) / "STOPPED.txt"; return p.read_text().strip().splitlines()[0][:44] if p.exists() else None
def census(a, which="vsel", t=64):
    c = jload(RUNS / f"sxcensus_p{TAG}{a}_{which}" / "census.json")
    if not c: return None
    rows = [r for r in c["rows"] if int(r["t"]) == t]; return None if not rows else rows[0]["exploded_frac"]
def calib(a): return jload(RUNS / f"sxcalib_p{TAG}{a}_vsel" / "calib.json")
def metrics(a, stage=""):
    p = RUNS / f"pretrain{TAG}_{a}{stage}" / "metrics.jsonl"; tr, mon = [], []
    if not p.exists(): return tr, mon
    for l in p.read_text().splitlines():
        l = l.strip()
        if not l: continue
        try: r = json.loads(l)
        except Exception: continue
        if "monitor" in r: mon.append(r["monitor"])
        elif "loss" in r: tr.append(r)
    return tr, mon
def end_ce(a):
    tr, _ = metrics(a); rows = [r for r in tr if "ce_in" in r]; return None if not rows else float(rows[-1]["ce_in"])
def val_peak_step(a):
    """argmax of the headline monitor val over BOTH stages (stage A as-is, stage B + 50k); None if no monitors."""
    key = "val_t64_ema" if HEAD_EMA[a] else "val_t64"; best = None
    for stage, off in (("a", 0), ("", 50000 if a in ("W0", "R1", "R2", "R3") else 0)):
        _, mon = metrics(a, stage)
        for m in mon:
            if key in m and "step" in m:
                v = (float(m[key]), int(m["step"]) + off)
                if best is None or v > best: best = v
    return None if best is None else best[1]
def stable(a):
    if stopped(a) is not None: return False
    cz = census(a)
    if cz is None or cz > 0.02: return False
    if a in FIELD: return True
    r = retfm(a); return r is not None and r >= 0.9
def memorized(a):
    ce = end_ce(a); return ce is not None and ce < 0.02
def reach(a): return v128(a)
def givens():
    try:
        z = np.load(NPZ, allow_pickle=True); q = z["test_q"]; return (q != 0).reshape(len(q), -1).sum(1)
    except Exception: return None
def fe_median(a):
    z = full_recs(a)
    if z is None or "first_exact" not in z: return None
    ce = z["cold_exact"].astype(bool); return None if not ce.any() else float(np.median(z["first_exact"][ce]))
def reach_lowgivens(a, g):
    z = scan_recs(a)
    if z is None or g is None: return None
    gg = g[z["idx"]]; m = (gg >= 21) & (gg < 25)
    hit = (z["mi_first_hit"] >= 0) | z["cold_exact"].astype(bool); return None if m.sum() == 0 else float(hit[m].mean())
def ckpt_paths(a):
    t = HEADLINE_T[a]; srcs = {}
    for nm, p in (("full_vsel", evdir(a) / f"full_vsel_t{t}" / "summary_all.json"), ("full_alt", evdir(a) / f"full_vsel_t{t}_alt" / "summary_all.json"),
                  ("scan", RUNS / f"sxscan_p{TAG}{a}" / "summary_all.json"), ("census_vsel", RUNS / f"sxcensus_p{TAG}{a}_vsel" / "census.json"),
                  ("screen_vb", RUNS / f"sxscreen_p{TAG}{a}_vb" / "summary_all.json"), ("retfm", evdir(a) / "retfm_t8" / "summary_all.json"),
                  ("calib", RUNS / f"sxcalib_p{TAG}{a}_vsel" / "calib.json")):
        s = jload(p)
        if s and s.get("ckpt"): srcs[nm] = s["ckpt"]
    return srcs

# ---------- integrity ----------
def integrity():
    errs = []
    for a in ARMS:
        for which in ("vsel", "final"):
            s = full(a, which)
            if s and s["n"] != N_FULL: errs.append(f"{a} full_{which} n={s['n']}")
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
    LINES.clear(); V = {}
    say("=" * 118); say("sportC2 VERDICT — the pre-parity graft night at d128 (rules registered 2026-09-04; analyzer pre-data)"); say("=" * 118)
    errs = integrity(); say("INTEGRITY: " + ("PASS" if not errs else "FAIL: " + " | ".join(errs)))
    if errs: V["INTEGRITY"] = "FAIL"; say("VERDICT WITHHELD — integrity breach"); _write(); return V
    V["INTEGRITY"] = "PASS"; g = givens()
    say("\nSECTION 1 — per arm (headline weights; vsel unless 'final'): vcold | fcold | b1 | reach(v128) | t1r@128 | retfm | endCE | census | calib top5(stalled) | fe_med | status")
    for a in ARMS:
        st = stopped(a); c = calib(a)
        say(f"  {a:3s} {DESC[a]:34s} vcold {fpp(vcold(a))} | fcold {fpp(fcold(a))} | b1 {fpp(b1(a))} | reach {fpp(reach(a))} | t1r {fpp(t1r(a))} | retfm {fpp(retfm(a))}"
            f" | CE {('  -  ' if end_ce(a) is None else f'{end_ce(a):.4f}')} | census {fpp(census(a))} | top5 {fpp(c['topk_correct_stalled'] if c else None)} | fe {('-' if fe_median(a) is None else f'{fe_median(a):.0f}')}"
            f" | {('STOPPED: ' + st) if st else ('STABLE' if stable(a) else 'UNSTABLE')}{' MEMORIZED' if memorized(a) else ''}")
    unst = [a for a in ARMS if not stable(a)]; mem = [a for a in NATIVE if memorized(a)]
    V["STABILITY"] = "ALL-STABLE" if not unst else "UNSTABLE:" + ",".join(unst)
    V["MEMORIZATION"] = "NONE" if not mem else "MEMORIZED:" + ",".join(mem)
    w = vcold("W0"); CNC2 = max(abs(w - B0_VCOLD), 0.01) if w is not None else 0.01
    bw = b1("W0"); FNC2 = 0.02
    V["CNC2"] = f"{100*CNC2:.2f}pp"; say(f"\nNOISE: CNC2 {100*CNC2:.2f}pp (W0 vs sportC1 B0 {100*B0_VCOLD:.2f}) | FNC2 {100*FNC2:.2f}pp")
    # R-C2-0
    ce = end_ce("W0"); pk = val_peak_step("W0")
    if ce is None or w is None: V["R-C2-0"] = "NO-DATA"
    elif ce < 0.02: V["R-C2-0"] = "WD-MEMORIZES"
    elif ce >= 0.05 and pk is not None and pk >= 25000 and w >= B0_VCOLD + CNC2: V["R-C2-0"] = "WD-HOLDS"
    else: V["R-C2-0"] = "WD-PARTIAL"
    say(f"R-C2-0 REGIME (W0): end CE {ce} | val-peak step {pk} | vcold {fpp(w)} vs B0 {fpp(B0_VCOLD)} + CNC2 -> {V['R-C2-0']}")
    # R-C2-1
    r1 = reach("R1")
    if r1 is None: V["R-C2-1"] = "NO-DATA"
    else:
        V["R-C2-1"] = "CARRY-WIDENS" if r1 >= .70 else ("CARRY-FLAT" if r1 <= .55 else "CARRY-PARTIAL")
        sel = (t1r("R1") / v128("R1")) if (t1r("R1") and v128("R1")) else None
        V["R-C2-1"] += "+SELECTOR-INTACT" if (sel is not None and sel >= .97) else ("+SPURIOUS-APPEAR" if sel is not None else "")
    say(f"R-C2-1 CARRY (R1): reach {fpp(r1)} (W0 {fpp(reach('W0'))}) | t1r/verified {('-' if not (t1r('R1') and v128('R1')) else f'{t1r('R1')/v128('R1'):.3f}')} -> {V['R-C2-1']}")
    # R-C2-2
    f2, f0 = fe_median("R2"), fe_median("W0"); rl = reach_lowgivens("R2", g)
    if f2 is None: V["R-C2-2"] = "NO-DATA"
    else:
        tags = []
        if f0 is not None and f2 <= 0.5 * f0: tags.append("DEPTH-PROPAGATES")
        if rl is not None and rl >= .50: tags.append("DEPTH-REACHES")
        V["R-C2-2"] = "+".join(tags) if tags else "DEPTH-FLAT"
    say(f"R-C2-2 DEPTH (R2): first_exact median {f2} (W0 {f0}) | reach@21-25 givens {fpp(rl)} (W0 {fpp(reach_lowgivens('W0', g))}) -> {V['R-C2-2']}")
    # R-C2-3
    c3 = calib("R3"); t5 = c3["topk_correct_stalled"] if c3 else None
    if t5 is None: V["R-C2-3"] = "NO-DATA"
    else:
        V["R-C2-3"] = "COMMIT-CALIBRATED" if t5 >= .90 else ("COMMIT-FLAT" if t5 <= .75 else "COMMIT-PARTIAL")
        sel = (t1r("R3") / v128("R3")) if (t1r("R3") and v128("R3")) else None
        if sel is not None and sel < .97: V["R-C2-3"] += "+SPURIOUS-APPEAR"
    say(f"R-C2-3 COMMIT (R3): calib top-5 on stalled {fpp(t5)} (W0 {fpp(calib('W0')['topk_correct_stalled'] if calib('W0') else None)}) | entropy step1 {(c3 or {}).get('entropy_step1')} -> {V['R-C2-3']}")
    # R-C2-4
    r4 = vcold("R4")
    V["R-C2-4"] = "NO-DATA" if r4 is None else ("REGIME-CLIMBS" if r4 - R0_VCOLD >= .03 else ("REGIME-SLOW" if r4 - R0_VCOLD >= .01 else "REGIME-FLAT"))
    say(f"R-C2-4 CONTINUATION (R4): vcold {fpp(r4)} vs R0 {fpp(R0_VCOLD)} -> {V['R-C2-4']}")
    # R-C2-5 / R-C2-6
    x1, x2 = vcold("X1"), vcold("X2")
    V["R-C2-5"] = "NO-DATA" if x1 is None else ("ORBIT-LOAD-BEARING" if X0_COLD16 - x1 >= .03 else ("ORBIT-LEARNED" if X0_COLD16 - x1 <= .01 else "ORBIT-PARTIAL"))
    V["R-C2-6"] = "NO-DATA" if x2 is None else ("GEOMETRY-HELPS" if x2 - X0_COLD16 >= .02 else ("GEOMETRY-HURTS" if x2 - X0_COLD16 <= -.02 else "GEOMETRY-NEUTRAL"))
    say(f"R-C2-5 ORBIT (X1): cold@D16 {fpp(x1)} vs X0 {fpp(X0_COLD16)} -> {V['R-C2-5']}")
    say(f"R-C2-6 GEOMETRY (X2): cold@D16 {fpp(x2)} vs X0 {fpp(X0_COLD16)} -> {V['R-C2-6']}")
    # R-C2-7
    carried = []
    def ok_arm(a): return stable(a) and vcold(a) is not None and w is not None and vcold(a) >= w - CNC2
    if V["R-C2-1"].startswith("CARRY-WIDENS") and ok_arm("R1"): carried.append("carry")
    if ("DEPTH-PROPAGATES" in V["R-C2-2"] or "DEPTH-REACHES" in V["R-C2-2"]) and ok_arm("R2"): carried.append("inner-cycles")
    if V["R-C2-3"].startswith("COMMIT-CALIBRATED") and ok_arm("R3"): carried.append("hard-rows")
    if V["R-C2-0"] in ("WD-HOLDS", "WD-PARTIAL"): carried.append("wd1.0")
    cands = [(vcold(a), a) for a in NATIVE if stable(a) and not memorized(a) and vcold(a) is not None]
    champ = max(cands)[1] if cands else None
    V["R-C2-7"] = ("CARRY:" + "+".join(carried) if carried else "CARRY:none") + f"|CHAMP:{champ or 'none'}"
    say(f"R-C2-7 CHAMPION-RECIPE: carried levers {carried} | champion-so-far {champ} {fpp(max(cands)[0] if cands else None)} | memorized (labeled) {mem}")
    say("\nSECTION 2 — prediction scoreboard (bands locked pre-data)")
    say(f"  W0 vcold {fpp(w)} {hm(w, PRED['W0']['vcold'])} | CE>=.05 {'HIT' if (ce is not None and ce >= .05) else 'MISS'}")
    say(f"  R1 reach {fpp(r1)} {hm(r1, PRED['R1']['reach'])} | vcold {fpp(vcold('R1'))} {hm(vcold('R1'), PRED['R1']['vcold'])}")
    say(f"  R2 fe_med {f2} {hm(f2, PRED['R2']['fe_med'])} | vcold {fpp(vcold('R2'))} {hm(vcold('R2'), PRED['R2']['vcold'])}")
    say(f"  R3 top5 {fpp(t5)} {hm(t5, PRED['R3']['top5'])} | vcold {fpp(vcold('R3'))} {hm(vcold('R3'), PRED['R3']['vcold'])}")
    say(f"  R4 vcold {fpp(r4)} {hm(r4, PRED['R4']['vcold'])} | X1 cold16 {fpp(x1)} {hm(x1, PRED['X1']['cold16'])} | X2 cold16 {fpp(x2)} {hm(x2, PRED['X2']['cold16'])}")
    say(); say("SUMMARY: " + "  ".join(f"{k}={v}" for k, v in V.items())); _write(); return V

def _write():
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text("\n".join(LINES) + "\n"); say(f"artifact -> {OUT}")

# ---------- selftest ----------
def _mk(root, a, *, vc=None, fc=None, b1v=None, v128v=None, t1rv=None, retfm_=1.0, ce=0.06, census_=0.0, stopped_=None,
        peak=30000, fe=10, top5=None, n_scan=N_SCAN, ckpt="ckpt_020000.pkl", scan_ckpt=None, low_reach=None):
    ev = root / f"sxeval_p{TAG}{a}"; t = HEADLINE_T[a]; ck = f"runs/pretrain{TAG}_{a}/{ckpt}"
    for which, v in (("vsel", vc), ("final", fc)):
        if v is not None:
            d = ev / f"full_{which}_t{t}"; d.mkdir(parents=True, exist_ok=True)
            (d / "summary_all.json").write_text(json.dumps(dict(exact_acc=v, n=N_FULL, ckpt=ck)))
            if which == "vsel":
                n = 4000; cold = np.zeros(n, bool); cold[: int(v * n)] = True
                np.savez(d / "records_all.npz", idx=np.arange(n), cold_exact=cold, first_exact=np.where(cold, fe, -1), rating=np.arange(n) % 60)
    if retfm_ is not None and a in NATIVE:
        d = ev / "retfm_t8"; d.mkdir(parents=True, exist_ok=True); (d / "summary_all.json").write_text(json.dumps(dict(exact_acc=retfm_, n=512, ckpt=ck)))
    if b1v is not None:
        d = root / f"sxscan_p{TAG}{a}"; d.mkdir(parents=True, exist_ok=True); n = n_scan
        cold = np.zeros(n, bool); cold[: int((vc or .3) * n)] = True
        hit = np.full(n, -1); nh = int((v128v or .5) * n); hit[:nh] = 3
        vote = float(((hit >= 0) | cold).mean())
        (d / "summary_all.json").write_text(json.dumps(dict(n=n, t_total=64, k_init=128, subsample_seed=20260822, b1_exact=b1v, exact_acc=float(cold.mean()),
                                                            exact_acc_vote=vote, vote_at_k={"128": vote}, t1r_at_k={"128": t1rv if t1rv is not None else vote * .99}, ckpt=(scan_ckpt or ck))))
        np.savez(d / "records_all.npz", idx=np.arange(n), cold_exact=cold, mi_first_hit=hit)
    if census_ is not None:
        d = root / f"sxcensus_p{TAG}{a}_vsel"; d.mkdir(parents=True, exist_ok=True)
        (d / "census.json").write_text(json.dumps(dict(ckpt=ck, rows=[dict(t=64, exploded_frac=census_), dict(t=256, exploded_frac=census_)])))
    if top5 is not None:
        d = root / f"sxcalib_p{TAG}{a}_vsel"; d.mkdir(parents=True, exist_ok=True)
        (d / "calib.json").write_text(json.dumps(dict(ckpt=ck, topk_correct_stalled=top5, entropy_step1=.3)))
    pd = root / f"pretrain{TAG}_{a}"; pd.mkdir(parents=True, exist_ok=True)
    if stopped_: (pd / "STOPPED.txt").write_text(stopped_)
    key = "val_t64_ema" if HEAD_EMA[a] else "val_t64"
    rows = [json.dumps(dict(step=s, loss=.3, ce_in=ce)) for s in (40000, 50000)]
    rows += [json.dumps({"monitor": {"step": 10000, key: .30}}), json.dumps({"monitor": {"step": 20000, key: (.48 if peak >= 20000 else .30)}})]
    (pd / "metrics.jsonl").write_text("\n".join(rows) + "\n")
    if a in ("W0", "R1", "R2", "R3"):
        pa = root / f"pretrain{TAG}_{a}a"; pa.mkdir(parents=True, exist_ok=True)
        (pa / "metrics.jsonl").write_text("\n".join([json.dumps({"monitor": {"step": 20000, key: .40}}), json.dumps({"monitor": {"step": peak if peak < 50000 else 45000, key: .50}})]) + "\n")

def selftest():
    checks = []
    def run(build):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); build(root)
            globals()["RUNS"] = root; globals()["OUT"] = root / "a" / "v.txt"; globals()["NPZ"] = root / "none.npz"
            with contextlib.redirect_stdout(io.StringIO()): return analyze()
    def wA(r):  # the hoped-for world
        _mk(r, "W0", vc=.46, fc=.45, b1v=.45, v128v=.56, fe=10, top5=.72)
        _mk(r, "R1", vc=.47, fc=.47, b1v=.45, v128v=.74, t1rv=.73, fe=8)
        _mk(r, "R2", vc=.48, fc=.48, b1v=.46, v128v=.60, fe=4)
        _mk(r, "R3", vc=.46, fc=.46, b1v=.45, v128v=.57, t1rv=.56, fe=9, top5=.92)
        _mk(r, "R4", vc=.41, fc=.41, b1v=.40, v128v=.55, fe=12)
        _mk(r, "X1", vc=.82, fc=.82); _mk(r, "X2", vc=.86, fc=.86)
    v = run(wA)
    checks += [("A integrity", v["INTEGRITY"] == "PASS"), ("A wd holds", v["R-C2-0"] == "WD-HOLDS"), ("A carry widens + selector", v["R-C2-1"] == "CARRY-WIDENS+SELECTOR-INTACT"),
               ("A depth propagates", v["R-C2-2"] == "DEPTH-PROPAGATES"), ("A commit calibrated", v["R-C2-3"] == "COMMIT-CALIBRATED"), ("A regime climbs", v["R-C2-4"] == "REGIME-CLIMBS"),
               ("A orbit load-bearing", v["R-C2-5"] == "ORBIT-LOAD-BEARING"), ("A geometry neutral", v["R-C2-6"] == "GEOMETRY-NEUTRAL"),
               ("A recipe carries all", v["R-C2-7"] == "CARRY:carry+inner-cycles+hard-rows+wd1.0|CHAMP:R2"), ("A all stable", v["STABILITY"] == "ALL-STABLE" and v["MEMORIZATION"] == "NONE")]
    def wB(r):  # kills: wd memorizes; carry flat; depth flat; commit flat with spurious; R4 flat; orbit learned; geometry hurts; R2 stopped
        _mk(r, "W0", vc=.44, fc=.25, b1v=.43, v128v=.55, ce=.001, fe=10, top5=.70)
        _mk(r, "R1", vc=.44, fc=.44, b1v=.43, v128v=.52, t1rv=.51, fe=10)
        _mk(r, "R2", vc=.30, fc=.30, b1v=.30, v128v=.40, fe=10, stopped_="STOPPED final step 25000 (NaN halt)")
        _mk(r, "R3", vc=.43, fc=.43, b1v=.42, v128v=.55, t1rv=.50, fe=10, top5=.70)
        _mk(r, "R4", vc=.375, fc=.375, b1v=.37, v128v=.51, fe=12)
        _mk(r, "X1", vc=.855, fc=.855); _mk(r, "X2", vc=.83, fc=.83)
    v = run(wB)
    checks += [("B wd memorizes", v["R-C2-0"] == "WD-MEMORIZES"), ("B carry flat", v["R-C2-1"].startswith("CARRY-FLAT")), ("B R2 stopped -> no data/unstable", "R2" in v["STABILITY"] and v["R-C2-2"] != "NO-DATA"),
               ("B commit flat + spurious", v["R-C2-3"] == "COMMIT-FLAT+SPURIOUS-APPEAR"), ("B regime flat", v["R-C2-4"] == "REGIME-FLAT"), ("B orbit learned", v["R-C2-5"] == "ORBIT-LEARNED"),
               ("B geometry hurts", v["R-C2-6"] == "GEOMETRY-HURTS"), ("B memorized W0 labeled", v["MEMORIZATION"] == "MEMORIZED:W0"), ("B champion excludes memorized", "CHAMP:R1" in v["R-C2-7"])]
    def wC(r):  # integrity breach: R1's scan on a different grid than its full
        wA(r); _mk(r, "R1", vc=.47, fc=.47, b1v=.45, v128v=.74, t1rv=.73, fe=8, scan_ckpt=f"runs/pretrain{TAG}_R1/ckpt_030000.pkl")
    v = run(wC)
    checks += [("C grid-inconsistency withholds", v["INTEGRITY"] == "FAIL" and "R-C2-1" not in v)]
    v = run(lambda r: None)
    checks += [("D no data", v["R-C2-0"] == "NO-DATA" and v["R-C2-3"] == "NO-DATA" and v["R-C2-7"] == "CARRY:none|CHAMP:none")]
    ok = 0
    for name, passed in checks:
        print(("  PASS  " if passed else "  FAIL  ") + name); ok += bool(passed)
    print(f"selftest: {ok}/{len(checks)}"); return ok == len(checks)


if __name__ == "__main__":
    if "--selftest" in sys.argv: sys.exit(0 if selftest() else 1)
    analyze()
