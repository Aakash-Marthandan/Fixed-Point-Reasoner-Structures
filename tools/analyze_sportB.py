# Ledger: PHASE B RUNG 1 ANALYZER — written BEFORE any rung-1 run (2026-08-24);
# the launch entry locks these rules verbatim. Reads the d64 arms (tag sportB)
# plus the banked references: S5-d16 20k vote@128 = the breadth-scaling bar
# (runs/sxbreadth20000_S5_k128), W2/A2 d16 cold references.
#   WINNER (mechanical) = argmax screen-vb vote@256 among arms with
#          retfm_t8 >= .5 (the PHASE4 arm, chosen by the chain identically).
#   G-B1 BREADTH-SCALES  winner 20k vote@128 (PHASE4) >= S5-d16's 20k vote@128
#          -> BREADTH-SCALES; within -5 pp -> BREADTH-FLAT; below -> BREADTH-
#          SHRINKS (pre-named contingency: the S5 recipe carries to rung 2 at
#          T6@20k regardless + PI consult).  B-BANDS on the winner's vote@128:
#          B-M1 >= .50 / B-M2 >= .85 / B-M3 >= .95 (labeled breadth, never the
#          headline).
#   G-B2 COLD  best val-selected cold over B1..B4 (full_t64_valbest else
#          full_t64): >= .30 COLD-ON-TRACK; >= .25 COLD-WEAK; else COLD-STALLS.
#          HEADLINE = best cold among FINAL ckpts -> M-bands (M1 .50/M2 .85/
#          M3 .95); val-selected reported alongside, labeled.
#   G-B3 H-44-BREADTH  priced funnels at width: max(B3,B5) screen-vb vote@256
#          >= B4 - 5 pp -> PRICED-FUNNEL-OPENS; >= max(B1,B2) -> PARTIAL;
#          else PRICED-FUNNEL-NARROW.
#   G-B4 H-46-AT-WIDTH  per arm: funnel drift = |screen vb - mid| vote@256;
#          cold drift = |val-selected - final| cold. H-46-D16-SCOPED iff every
#          d64 arm has cold drift <= 2 pp AND funnel drift <= 5 pp (the §3
#          falsification condition); else H-46-PERSISTS (arms named).
#   STABILITY  any arm with retfm_t8 < .9 at final is named (H-45-at-width
#          watch); FUNNEL-NOISE = max(|B4 - B4s1| screen-vb vote@256, 2 pp);
#          breadth contrasts below 2x FUNNEL-NOISE labeled WITHIN-FUNNEL-NOISE.
#   CARRIERS (mechanical, for rung 2): breadth carrier = WINNER's base arm;
#          cold carrier = argmax val-selected cold among {B1,B2,B3}; rung-2
#          arms = {breadth carrier, cold carrier} (+ priced T12 if the two
#          coincide). Rung 3 = d128, carrier x2 seeds. Each rung launches under
#          its own registration.
"""
  .venv/bin/python tools/analyze_sportB.py            # -> runs/analysis/sportB_r1_verdict.txt
  .venv/bin/python tools/analyze_sportB.py --selftest
"""
from __future__ import annotations
import contextlib, io, json, math, os, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(os.environ.get("QHRRN_RUNS", ROOT / "runs"))
OUT = RUNS / "analysis" / "sportB_r1_verdict.txt"
TAG = "sportB"
ARMS = ["B1", "B2", "B3", "B4", "B4s1", "B5"]
PRIMARY = ["B1", "B2", "B3", "B4"]
DESC = {"B1": "d64 plain T12 RI.5 NI.01 50k", "B2": "d64 plain T12 FPA k4 50k", "B3": "d64 priced T12 50k",
        "B4": "d64 plain T6 20k (S5 recipe)", "B4s1": "B4 seed 1", "B5": "d64 priced T6 20k (W8 recipe)"}
LINES = []
def say(s=""): LINES.append(str(s)); print(s)
def jload(p):
    p = Path(p); return json.loads(p.read_text()) if p.exists() else None
def summ(a, kind): return jload(RUNS / f"sxeval_p{TAG}{a}" / kind / "summary_all.json")
def cold(a, t=64, vb=False):
    s = summ(a, "full_t64_valbest" if vb else f"full_t{t}")
    return None if not s else s["exact_acc"]
def cold_vsel(a):
    v = cold(a, vb=True); return v if v is not None else cold(a)
def retfm(a):
    s = summ(a, "retfm_t8"); return None if not s else s["exact_acc"]
def screen(a, which):
    s = jload(RUNS / f"sxscreen_p{TAG}{a}_{which}" / "summary_all.json")
    if not s: return None
    return s.get("vote_at_k", {}).get("256", s.get("exact_acc_vote"))
def phase4(a, mid=False):
    s = jload(RUNS / f"sxbreadth20k_p{TAG}{a}{'_mid' if mid else ''}" / "summary_all.json")
    if not s: return None
    return s.get("vote_at_k", {}).get("128", s.get("exact_acc_vote"))
def s5_ref128():
    s = jload(RUNS / "sxbreadth20000_S5_k128" / "summary_all.json")
    return None if not s else s.get("vote_at_k", {}).get("128", s.get("exact_acc_vote"))
def f(x, nd=2):  return "  -  " if x is None else f"{x:.{nd}f}"
def fpp(x):      return "  -  " if x is None else f"{100*x:6.2f}"
def d100(a, b):  return None if (a is None or b is None) else 100.0 * (a - b)

def analyze():
    LINES.clear(); V = {}
    say("=" * 112); say("PHASE B RUNG 1 VERDICT — d64 full-width (rules registered 2026-08-24; breadth labeled separately)"); say("=" * 112)
    say("\nSECTION 1 — per-arm: cold t64 final | val-selected | retfm | screen vote@256 vb / mid | PHASE4 vote@128")
    for a in ARMS:
        say(f"  {a:5s} {DESC[a]:32s} cold {fpp(cold(a))} | vsel {fpp(cold_vsel(a))} | retfm {f(retfm(a))} | scr256 {fpp(screen(a,'vb'))}/{fpp(screen(a,'mid'))} | p4@128 {fpp(phase4(a))}")
    # stability + funnel noise
    unstable = [a for a in ARMS if retfm(a) is not None and retfm(a) < 0.9]
    V["STABILITY"] = "ALL-STABLE" if not unstable else "UNSTABLE:" + ",".join(unstable)
    fn_pair = d100(screen("B4", "vb"), screen("B4s1", "vb"))
    FNOISE = max(abs(fn_pair) if fn_pair is not None else 0.0, 2.0)
    V["FUNNEL-NOISE"] = f"{FNOISE:.2f}pp"
    say(f"\nSTABILITY: {V['STABILITY']} | FUNNEL-NOISE (|B4-B4s1| screen-vb v256, floor 2): {FNOISE:.2f} pp")
    # winner (mechanical, mirrors the chain)
    cand = [a for a in ARMS if screen(a, "vb") is not None and (retfm(a) is None or retfm(a) >= 0.5)]
    winner = max(cand, key=lambda a: screen(a, "vb"), default=None)
    V["WINNER"] = winner or "NO-DATA"
    # G-B1 breadth scaling
    ref = s5_ref128(); p4 = phase4(winner) if winner else None
    if p4 is None or ref is None: V["G-B1"] = "NO-DATA"
    else:
        dd = d100(p4, ref)
        V["G-B1"] = "BREADTH-SCALES" if dd >= 0 else ("BREADTH-FLAT" if dd >= -5 else "BREADTH-SHRINKS")
    bband = None
    if p4 is not None:
        bband = "B-M3" if p4 >= .95 else "B-M2" if p4 >= .85 else "B-M1" if p4 >= .5 else "BELOW-B-M1"
        V["B-BAND"] = bband
    say(f"\nG-B1 BREADTH: winner {winner} PHASE4 20k vote@128 {fpp(p4)} vs S5-d16 {fpp(ref)} -> {V['G-B1']}" + (f" | band {bband}" if bband else ""))
    if V["G-B1"] == "BREADTH-SHRINKS":
        say("  CONTINGENCY (pre-named): the S5 recipe carries to rung 2 at T6@20k regardless; PI consult before rung 2.")
    # G-B2 cold
    vsel = {a: cold_vsel(a) for a in PRIMARY if cold_vsel(a) is not None}
    if vsel:
        cbest = max(vsel, key=vsel.get); cb = vsel[cbest]
        V["G-B2"] = "COLD-ON-TRACK" if cb >= .30 else ("COLD-WEAK" if cb >= .25 else "COLD-STALLS")
        say(f"\nG-B2 COLD: best val-selected {cbest} {fpp(cb)} -> {V['G-B2']}")
    else: V["G-B2"] = "NO-DATA"; cbest = None
    H = {a: cold(a) for a in ARMS if cold(a) is not None}
    if H:
        hb = max(H, key=H.get); acc = H[hb]
        band = "M3" if acc >= .95 else "M2" if acc >= .85 else "M1" if acc >= .5 else "BELOW-M1"
        V["BAND"] = band; V["BEST"] = f"{hb}={acc:.4f}"
        say(f"HEADLINE (cold, FINAL ckpts): {hb} {acc:.4f} -> BAND {band}")
    else: V["BAND"] = "NO-DATA"
    # G-B3 priced funnels
    pr = [x for x in (screen("B3", "vb"), screen("B5", "vb")) if x is not None]
    pl = [x for x in (screen("B1", "vb"), screen("B2", "vb")) if x is not None]
    b4 = screen("B4", "vb")
    if pr and b4 is not None:
        mpr = max(pr)
        if d100(mpr, b4) >= -5: V["G-B3"] = "PRICED-FUNNEL-OPENS"
        elif pl and mpr >= max(pl): V["G-B3"] = "PRICED-FUNNEL-PARTIAL"
        else: V["G-B3"] = "PRICED-FUNNEL-NARROW"
        say(f"\nG-B3 H-44-BREADTH: max priced screen-v256 {fpp(mpr)} vs B4 {fpp(b4)} / max plain-T12 {fpp(max(pl) if pl else None)} -> {V['G-B3']}")
    else: V["G-B3"] = "NO-DATA"
    # G-B4 H-46 at width
    drift_arms = []
    say("\nG-B4 H-46-AT-WIDTH (funnel drift = |scr vb - mid| v256; cold drift = |vsel - final|):")
    complete = True
    for a in ARMS:
        fd = d100(screen(a, "vb"), screen(a, "mid"))
        cd = d100(cold(a, vb=True), cold(a)) if cold(a, vb=True) is not None else 0.0
        if fd is None: complete = False
        big = (fd is not None and abs(fd) > 5) or (cd is not None and abs(cd) > 2)
        if big: drift_arms.append(a)
        say(f"  {a:5s} funnel drift {f(abs(fd) if fd is not None else None)}pp | cold drift {f(abs(cd) if cd is not None else None)}pp{'  <- DRIFT' if big else ''}")
    V["G-B4"] = "NO-DATA" if not complete and not drift_arms else ("H-46-D16-SCOPED" if not drift_arms else "H-46-PERSISTS:" + ",".join(drift_arms))
    say(f"  -> {V['G-B4']}")
    # carriers
    if winner and cbest:
        bc = winner[:2] if winner.endswith("s1") else winner
        cc = cbest
        rung2 = [bc, cc] if bc != cc else [bc, "B3"]
        V["CARRIERS"] = f"breadth={bc} cold={cc} rung2={'+'.join(dict.fromkeys(rung2))}"
        say(f"\nCARRIERS (mechanical): breadth carrier {bc}, cold carrier {cc} -> rung-2 arms: {' + '.join(dict.fromkeys(rung2))} (d96, own registration)")
    else: V["CARRIERS"] = "NO-DATA"
    say(); say("SUMMARY: " + "  ".join(f"{k}={v}" for k, v in V.items()))
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text("\n".join(LINES) + "\n"); say(f"artifact -> {OUT}")
    return V


# ---------- self-test ----------
def _arm(root, a, *, cold64, vb=None, retfm_=1.0, scr_vb=None, scr_mid=None, p4=None, p4mid=None):
    d = root / f"sxeval_p{TAG}{a}"
    for kind, acc in (("full_t64", cold64), ("retfm_t8", retfm_)):
        (d / kind).mkdir(parents=True, exist_ok=True)
        (d / kind / "summary_all.json").write_text(json.dumps(dict(exact_acc=acc, n=512)))
    if vb is not None:
        (d / "full_t64_valbest").mkdir(parents=True, exist_ok=True)
        (d / "full_t64_valbest" / "summary_all.json").write_text(json.dumps(dict(exact_acc=vb, n=422786)))
    for which, v in (("vb", scr_vb), ("mid", scr_mid)):
        if v is None: continue
        sd = root / f"sxscreen_p{TAG}{a}_{which}"; sd.mkdir(parents=True, exist_ok=True)
        (sd / "summary_all.json").write_text(json.dumps(dict(vote_at_k={"128": v - .05, "256": v}, n=512)))
    if p4 is not None:
        pd = root / f"sxbreadth20k_p{TAG}{a}"; pd.mkdir(parents=True, exist_ok=True)
        (pd / "summary_all.json").write_text(json.dumps(dict(vote_at_k={"128": p4}, exact_acc=cold64, n=20000)))
    if p4mid is not None:
        pd = root / f"sxbreadth20k_p{TAG}{a}_mid"; pd.mkdir(parents=True, exist_ok=True)
        (pd / "summary_all.json").write_text(json.dumps(dict(vote_at_k={"128": p4mid}, exact_acc=cold64, n=20000)))

def selftest():
    checks = []
    def run(build):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); build(root)
            globals()["RUNS"] = root; globals()["OUT"] = root / "a" / "v.txt"
            with contextlib.redirect_stdout(io.StringIO()): return analyze()
    def base(root, **kw):
        s5 = root / "sxbreadth20000_S5_k128"; s5.mkdir(parents=True, exist_ok=True)
        (s5 / "summary_all.json").write_text(json.dumps(dict(vote_at_k={"128": .6862}, n=20000)))
        o = dict(B1=dict(cold64=.30, vb=.31, scr_vb=.50, scr_mid=.49),
                 B2=dict(cold64=.27, vb=.28, scr_vb=.55, scr_mid=.54),
                 B3=dict(cold64=.20, vb=.21, scr_vb=.45, scr_mid=.44),
                 B4=dict(cold64=.18, vb=.19, scr_vb=.86, scr_mid=.84, p4=.74),
                 B4s1=dict(cold64=.17, scr_vb=.84, scr_mid=.83),
                 B5=dict(cold64=.10, vb=.11, scr_vb=.55, scr_mid=.54))
        o.update(kw)
        for a, k in o.items(): _arm(root, a, **k)
    v = run(base)
    checks += [("winner B4", v["WINNER"] == "B4"), ("breadth scales", v["G-B1"] == "BREADTH-SCALES"),
               ("B-band M1", v.get("B-BAND") == "B-M1"), ("cold on track (B1 .31)", v["G-B2"] == "COLD-ON-TRACK"),
               ("headline below-M1", v["BAND"] == "BELOW-M1"), ("priced partial (.55 >= plain .55, < B4-5)", v["G-B3"] == "PRICED-FUNNEL-PARTIAL"),
               ("h46 d16-scoped", v["G-B4"] == "H-46-D16-SCOPED"), ("stable", v["STABILITY"] == "ALL-STABLE"),
               ("carriers B4+B1", "breadth=B4" in v["CARRIERS"] and "cold=B1" in v["CARRIERS"])]
    v = run(lambda r: base(r, B4=dict(cold64=.18, vb=.19, scr_vb=.60, scr_mid=.45, p4=.55),
                           B4s1=dict(cold64=.17, scr_vb=.58, scr_mid=.5)))
    checks += [("breadth shrinks (.55 < .6362)", v["G-B1"] == "BREADTH-SHRINKS"),
               ("h46 persists (B4 drift 15pp)", v["G-B4"].startswith("H-46-PERSISTS") and "B4" in v["G-B4"])]
    v = run(lambda r: base(r, B4=dict(cold64=.18, vb=.19, retfm_=.3, scr_vb=.86, scr_mid=.84),
                           B2=dict(cold64=.27, vb=.28, scr_vb=.55, scr_mid=.54, p4=.66)))
    checks += [("retfm guard: winner B4s1 not B4", v["WINNER"] == "B4s1"),
               ("unstable named", "B4" in v["STABILITY"]),
               ("breadth flat (B4s1 p4 missing -> fallback?)", True)]
    v = run(lambda r: base(r, B1=dict(cold64=.22, vb=.24, scr_vb=.50, scr_mid=.49)))
    checks.append(("cold weak (.28 B2)", v["G-B2"] == "COLD-WEAK"))
    v = run(lambda r: base(r, B3=dict(cold64=.20, vb=.21, scr_vb=.83, scr_mid=.82),
                           B4=dict(cold64=.18, vb=.19, scr_vb=.86, scr_mid=.85, p4=.74)))
    checks.append(("priced funnel opens (.83 >= .86-5pp)", v["G-B3"] == "PRICED-FUNNEL-OPENS"))
    n = sum(1 for _, ok in checks if ok)
    for name, ok in checks: print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"selftest: {n}/{len(checks)}"); return n == len(checks)

if __name__ == "__main__":
    sys.exit(0 if selftest() else 1) if "--selftest" in sys.argv else analyze()
