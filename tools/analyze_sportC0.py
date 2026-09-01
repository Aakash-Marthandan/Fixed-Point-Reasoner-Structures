# Ledger: CHAMPION TRACK PILOT ANALYZER — written BEFORE any sportC0 run
# (2026-09-01); the launch registration locks these rules verbatim
# (Plan_2026-09-01_Champion_Track §3). Reads the seven native9 d96 arms plus
# the banked CANVAS references (D4 scan/full, constants as fallback).
#   Native noise instruments (measured on the P3/P3s1 champion pair):
#     FNC0    = max(|b1(P3) - b1(P3s1)|, .02)      — per-draw seed noise.
#     CNC0    = max(|cold(P3) - cold(P3s1)|, .01)  — cold seed noise.
#   R-C0-1 ARITY (P1 = the D4-strategy verbatim, native): cold(P1) within
#          FN_ARITY (=.0371, the canvas FN2b) of D4 cold .3353 AND
#          screenvb(P1) within FN_ARITY of D4 vb .8906 -> NATIVE-CARRIES;
#          BOTH below -> ARITY-HURTS (champion falls back per plan §2);
#          else ARITY-MIXED (PI consult). Profile read is descriptive.
#   R-C0-2 RI: b1(P2) >= b1(P1) + .10 AND retfm >= .9 on BOTH P2 seeds ->
#          RI-PAYS; gain < .10 (clean) -> RI-WEAK (champion drops RI);
#          retfm < .9 or STOPPED on either seed -> RI-FRAGILE-AT-WIDTH
#          (champion = no-RI P1-class scaled).
#   R-C0-3 NI: P5 unclean while P2 pair clean -> NI-CONVICTED; P5 clean AND
#          b1(P5) >= b1(P2) + FNC0 -> NI-EARNS; else NI-NEUTRAL (stays out).
#   R-C0-4 AUG: b1(P6) >= b1(P3) + FNC0 AND cold(P6) >= cold(P3) - CNC0 ->
#          AUG1000; else AUG100.
#   R-C0-5 CHAMPION GO: P3 pair clean AND max b1(P3*) >= max(b1(P1), b1(P2))
#          - FNC0 -> GO-P3 (d128 = P3 config verbatim + R-C0-4 aug);
#          P3 pair clean but below -> GO-BEST (config = argmax b1 clean arm);
#          any P3 seed unclean -> GO-FALLBACK (best clean arm);
#          no clean arm -> NO-GO (PI consult).
#   STABILITY: any retfm < .9 or STOPPED named. clean := retfm >= .9 and not
#          STOPPED. Descriptive (no rules): verified@128, t1r@128, majority,
#          rider stats (canvas C3X/D4 EqR-statistic evals), flux profile s1
#          share (arity signature), A_total closure.
"""
  .venv/bin/python tools/analyze_sportC0.py            # -> runs/analysis/sportC0_verdict.txt
  .venv/bin/python tools/analyze_sportC0.py --selftest
"""
from __future__ import annotations
import contextlib, io, json, os, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(os.environ.get("QHRRN_RUNS", ROOT / "runs"))
OUT = RUNS / "analysis" / "sportC0_verdict.txt"
TAG = "sportC0"
ARMS = ["P1", "P2", "P2s1", "P3", "P3s1", "P5", "P6"]
D4_COLD_CONST, D4_VB_CONST = 0.3353, 0.8906   # canvas refs (rung-2b verdict)
FN_ARITY = 0.0371                              # canvas FN2b (matched-pair, 2b)
DESC = {"P1": "native D4-strategy (T12 FPA dose, no RI)",
        "P2": "P1 + RI .5 s0", "P2s1": "P1 + RI .5 s1",
        "P3": "CHAMPION (T16 RI FPA dose 2-phase) s0", "P3s1": "CHAMPION s1",
        "P5": "P2 + NI .01", "P6": "CHAMPION + aug1000"}
LINES = []
def say(s=""): LINES.append(str(s)); print(s)
def jload(p):
    p = Path(p); return json.loads(p.read_text()) if p.exists() else None
def fpp(x): return "  -  " if x is None else f"{100*x:6.2f}"
def cold(a):
    s = jload(RUNS / f"sxeval_p{TAG}{a}" / "full_t64" / "summary_all.json")
    return None if not s else s["exact_acc"]
def scan(a):
    return jload(RUNS / f"sxscan_p{TAG}{a}" / "summary_all.json")
def b1(a):
    s = scan(a); return None if not s else s.get("b1_exact")
def v128(a):
    s = scan(a); return None if not s else s.get("vote_at_k", {}).get("128", s.get("exact_acc_vote"))
def t1r(a):
    s = scan(a); return None if not s else s.get("t1r_at_k", {}).get("128")
def retfm(a):
    s = jload(RUNS / f"sxeval_p{TAG}{a}" / "retfm_t8" / "summary_all.json")
    return None if not s else s["exact_acc"]
def screen_vb(a):
    s = jload(RUNS / f"sxscreen_p{TAG}{a}_vb" / "summary_all.json")
    return None if not s else s.get("vote_at_k", {}).get("256", s.get("exact_acc_vote"))
def stopped(a):
    p = RUNS / f"pretrain{TAG}_{a}" / "STOPPED.txt"
    return p.read_text().strip().splitlines()[0][:44] if p.exists() else None
def clean(a):
    r = retfm(a)
    return (r is not None) and (r >= 0.9) and (stopped(a) is None)

def analyze():
    LINES.clear(); V = {}
    say("=" * 112)
    say("CHAMPION TRACK PILOT VERDICT — sportC0 native9 d96 (rules registered 2026-09-01)")
    say("=" * 112)
    d4s = jload(RUNS / "sxscan_psportBr2bD4" / "summary_all.json")
    D4_COLD = D4_COLD_CONST
    d4f = jload(RUNS / "sxeval_psportBr2bD4" / "full_t64" / "summary_all.json")
    if d4f: D4_COLD = d4f["exact_acc"]
    D4_VB = D4_VB_CONST
    say(f"canvas refs: D4 cold {fpp(D4_COLD)} | D4 screen-vb {fpp(D4_VB)} | FN_ARITY {100*FN_ARITY:.2f}pp")
    say("\nSECTION 1 — per-arm: cold | b1(EqR B=1) | verified@128 | t1r@128 | retfm | screen-vb | stopped?")
    for a in ARMS:
        st = stopped(a)
        say(f"  {a:4s} {DESC[a]:38s} cold {fpp(cold(a))} | b1 {fpp(b1(a))} | v128 {fpp(v128(a))}"
            f" | t1r {fpp(t1r(a))} | retfm {fpp(retfm(a))} | vb {fpp(screen_vb(a))}"
            f" | {('STOPPED: ' + st) if st else 'complete'}")
    unstable = [a for a in ARMS if not clean(a)]
    V["STABILITY"] = "ALL-CLEAN" if not unstable else "UNCLEAN:" + ",".join(unstable)
    bP3, bP3s = b1("P3"), b1("P3s1")
    cP3, cP3s = cold("P3"), cold("P3s1")
    FNC0 = max(abs(bP3 - bP3s), 0.02) if (bP3 is not None and bP3s is not None) else 0.02
    CNC0 = max(abs(cP3 - cP3s), 0.01) if (cP3 is not None and cP3s is not None) else 0.01
    V["FNC0"] = f"{100*FNC0:.2f}pp"; V["CNC0"] = f"{100*CNC0:.2f}pp"
    say(f"\nNOISE (champion pair P3/P3s1): b1 FNC0 {100*FNC0:.2f}pp | cold CNC0 {100*CNC0:.2f}pp")
    # R-C0-1 arity
    c1, s1v = cold("P1"), screen_vb("P1")
    if c1 is None or s1v is None: V["R-C0-1"] = "NO-DATA"
    else:
        cold_ok = c1 >= D4_COLD - FN_ARITY
        vb_ok = s1v >= D4_VB - FN_ARITY
        V["R-C0-1"] = ("NATIVE-CARRIES" if (cold_ok and vb_ok)
                       else ("ARITY-HURTS" if (not cold_ok and not vb_ok) else "ARITY-MIXED"))
    say(f"\nR-C0-1 ARITY: P1 cold {fpp(c1)} vs {fpp(D4_COLD)} | vb {fpp(s1v)} vs {fpp(D4_VB)} -> {V['R-C0-1']}")
    # R-C0-2 RI
    b2, b1p = b1("P2"), b1("P1")
    p2_clean = clean("P2") and clean("P2s1")
    if b2 is None or b1p is None: V["R-C0-2"] = "NO-DATA"
    elif not p2_clean: V["R-C0-2"] = "RI-FRAGILE-AT-WIDTH"
    elif b2 >= b1p + 0.10: V["R-C0-2"] = "RI-PAYS"
    else: V["R-C0-2"] = "RI-WEAK"
    say(f"R-C0-2 RI: b1(P2) {fpp(b2)} vs b1(P1)+10 {fpp(None if b1p is None else b1p + .10)}"
        f" | pair clean {p2_clean} -> {V['R-C0-2']}")
    # R-C0-3 NI
    b5 = b1("P5")
    if b5 is None and not stopped("P5"): V["R-C0-3"] = "NO-DATA"
    elif not clean("P5") and p2_clean: V["R-C0-3"] = "NI-CONVICTED"
    elif clean("P5") and b2 is not None and b5 is not None and b5 >= b2 + FNC0: V["R-C0-3"] = "NI-EARNS"
    else: V["R-C0-3"] = "NI-NEUTRAL"
    say(f"R-C0-3 NI: P5 b1 {fpp(b5)} clean {clean('P5')} -> {V['R-C0-3']}")
    # R-C0-4 aug
    b6, c6 = b1("P6"), cold("P6")
    if b6 is None or bP3 is None: V["R-C0-4"] = "NO-DATA"
    elif clean("P6") and b6 >= bP3 + FNC0 and (c6 is not None and cP3 is not None and c6 >= cP3 - CNC0):
        V["R-C0-4"] = "AUG1000"
    else: V["R-C0-4"] = "AUG100"
    say(f"R-C0-4 AUG: b1(P6) {fpp(b6)} vs b1(P3)+FNC0 {fpp(None if bP3 is None else bP3 + FNC0)} -> {V['R-C0-4']}")
    # R-C0-5 champion GO
    p3_clean = clean("P3") and clean("P3s1")
    cands = [(b1(a), a) for a in ARMS if clean(a) and b1(a) is not None]
    if bP3 is None and not stopped("P3"): V["R-C0-5"] = "NO-DATA"
    elif not cands: V["R-C0-5"] = "NO-GO"
    else:
        best_b, best_a = max(cands)
        bar = max([x for x in (b1("P1"), b1("P2")) if x is not None], default=None)
        if p3_clean and bP3 is not None and bar is not None and max(bP3, bP3s or 0) >= bar - FNC0:
            V["R-C0-5"] = "GO-P3"
        elif p3_clean:
            V["R-C0-5"] = f"GO-BEST:{best_a}"
        else:
            V["R-C0-5"] = f"GO-FALLBACK:{best_a}"
    say(f"R-C0-5 CHAMPION: P3 pair clean {p3_clean} | best clean b1 {cands and fpp(max(cands)[0])} -> {V['R-C0-5']}")
    # descriptive: riders
    say("\nSECTION 2 — canvas riders (EqR-statistic on C3X/D4; program review §1)")
    for src in ("C3X", "D4"):
        s = jload(RUNS / f"sxrider_{src}_sel5k" / "summary_all.json")
        if s:
            say(f"  {src}: b1 {fpp(s.get('b1_exact'))} | t1r@128 {fpp(s.get('t1r_at_k', {}).get('128'))}"
                f" | majority@128 {fpp(s.get('majority_vote_at_k', {}).get('128'))}"
                f" | verified@128 {fpp(s.get('vote_at_k', {}).get('128'))} (n {s.get('n')})")
    say(); say("SUMMARY: " + "  ".join(f"{k}={v}" for k, v in V.items()))
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text("\n".join(LINES) + "\n"); say(f"artifact -> {OUT}")
    return V


# ---------- self-test ----------
def _mk(root, a, *, cold64=None, b1v=None, v128v=None, retfm_=1.0, svb=None, stopped_=None):
    if cold64 is not None:
        d = root / f"sxeval_p{TAG}{a}" / "full_t64"; d.mkdir(parents=True, exist_ok=True)
        (d / "summary_all.json").write_text(json.dumps(dict(exact_acc=cold64, n=422786)))
    if retfm_ is not None:
        d = root / f"sxeval_p{TAG}{a}" / "retfm_t8"; d.mkdir(parents=True, exist_ok=True)
        (d / "summary_all.json").write_text(json.dumps(dict(exact_acc=retfm_, n=512)))
    if b1v is not None:
        d = root / f"sxscan_p{TAG}{a}"; d.mkdir(parents=True, exist_ok=True)
        (d / "summary_all.json").write_text(json.dumps(dict(
            n=20000, b1_exact=b1v, exact_acc_vote=v128v or .8,
            vote_at_k={"128": v128v or .8}, t1r_at_k={"128": (b1v or 0) + .02})))
    if svb is not None:
        d = root / f"sxscreen_p{TAG}{a}_vb"; d.mkdir(parents=True, exist_ok=True)
        (d / "summary_all.json").write_text(json.dumps(dict(vote_at_k={"256": svb}, n=512)))
    pd = root / f"pretrain{TAG}_{a}"; pd.mkdir(parents=True, exist_ok=True)
    if stopped_: (pd / "STOPPED.txt").write_text(stopped_)

def selftest():
    checks = []
    def run(build):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); build(root)
            globals()["RUNS"] = root; globals()["OUT"] = root / "a" / "v.txt"
            with contextlib.redirect_stdout(io.StringIO()): return analyze()
    def wA(r):  # champion world: arity carries, RI pays, P3 wins, aug neutral
        _mk(r, "P1", cold64=.33, b1v=.35, v128v=.84, svb=.88)
        _mk(r, "P2", cold64=.31, b1v=.52, v128v=.90, svb=.90)
        _mk(r, "P2s1", cold64=.30, b1v=.50, v128v=.89, svb=.89)
        _mk(r, "P3", cold64=.32, b1v=.58, v128v=.92, svb=.92)
        _mk(r, "P3s1", cold64=.31, b1v=.56, v128v=.91, svb=.91)
        _mk(r, "P5", cold64=.30, b1v=.53, v128v=.89, svb=.89)
        _mk(r, "P6", cold64=.32, b1v=.59, v128v=.92, svb=.92)
    v = run(wA)
    checks += [("A arity carries", v["R-C0-1"] == "NATIVE-CARRIES"),
               ("A RI pays", v["R-C0-2"] == "RI-PAYS"),
               ("A NI neutral", v["R-C0-3"] == "NI-NEUTRAL"),
               ("A aug100 (within noise)", v["R-C0-4"] == "AUG100"),
               ("A GO-P3", v["R-C0-5"] == "GO-P3"),
               ("A all clean", v["STABILITY"] == "ALL-CLEAN"),
               ("A FNC0 floor", v["FNC0"] == "2.00pp")]
    def wB(r):  # RI fragile: P2s1 stopped; P3 pair unclean too -> fallback P1
        _mk(r, "P1", cold64=.33, b1v=.35, v128v=.84, svb=.88)
        _mk(r, "P2", cold64=.31, b1v=.52, svb=.90)
        _mk(r, "P2s1", cold64=.10, b1v=.05, retfm_=.5, svb=.30,
            stopped_="STOPPED final step 12000 (NaN halt)")
        _mk(r, "P3", cold64=.20, b1v=.30, retfm_=.85, svb=.60)
        _mk(r, "P3s1", cold64=.19, b1v=.28, retfm_=.84, svb=.58)
        _mk(r, "P5", cold64=.28, b1v=.48, svb=.88)
        _mk(r, "P6", cold64=.30, b1v=.50, svb=.90)
    v = run(wB)
    checks += [("B RI fragile", v["R-C0-2"] == "RI-FRAGILE-AT-WIDTH"),
               ("B fallback to best clean", v["R-C0-5"].startswith("GO-FALLBACK")),
               ("B unclean named", "P2s1" in v["STABILITY"] and "P3" in v["STABILITY"])]
    def wC(r):  # arity hurts + NI convicted + aug earns
        _mk(r, "P1", cold64=.22, b1v=.20, v128v=.6, svb=.70)
        _mk(r, "P2", cold64=.24, b1v=.28, svb=.75)
        _mk(r, "P2s1", cold64=.23, b1v=.27, svb=.74)
        _mk(r, "P3", cold64=.25, b1v=.36, svb=.78)
        _mk(r, "P3s1", cold64=.24, b1v=.34, svb=.77)
        _mk(r, "P5", cold64=.05, b1v=.02, retfm_=.4, svb=.20)
        _mk(r, "P6", cold64=.26, b1v=.41, svb=.80)
    v = run(wC)
    checks += [("C arity hurts", v["R-C0-1"] == "ARITY-HURTS"),
               ("C NI convicted", v["R-C0-3"] == "NI-CONVICTED"),
               ("C RI weak (<10pp)", v["R-C0-2"] == "RI-WEAK"),
               ("C aug1000 earns", v["R-C0-4"] == "AUG1000"),
               ("C GO-P3 (within FNC0 of best)", v["R-C0-5"] == "GO-P3")]
    v = run(lambda r: None)
    checks += [("D no data", v["R-C0-1"] == "NO-DATA" and v["R-C0-5"] == "NO-DATA")]
    ok = 0
    for name, passed in checks:
        print(("  PASS  " if passed else "  FAIL  ") + name); ok += bool(passed)
    print(f"selftest: {ok}/{len(checks)}")
    return ok == len(checks)


if __name__ == "__main__":
    if "--selftest" in sys.argv: sys.exit(0 if selftest() else 1)
    analyze()
