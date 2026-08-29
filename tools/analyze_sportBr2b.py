# Ledger: PHASE B RUNG 2B ANALYZER — written BEFORE any rung-2b run (2026-08-29);
# the launch entry locks these rules verbatim. Reads the d96 2b arms (tag
# sportBr2b) plus the banked rung-2 references C3 (screen-vb .8848, cold .2228,
# p4@128 .8057 — read from disk, constants as fallback).
#   FN2b   = max(|C3vb - screenvb(D2)|, .02)   — matched-pair funnel noise.
#   CNOISE = max(|C3cold - cold(D2)|, .01)     — matched-pair cold noise.
#   R2b-1 FUNNEL/B-M2: screenvb(D1) > C3vb + FN2b -> FUNNEL-GROWS (p4 read:
#          >= .85 -> B-M2; >= .8057 + .011 -> B-M1-IMPROVED; else B-M1-FLAT);
#          within +-FN2b -> LENGTH-NEUTRAL (carrier stays 20k);
#          below -> LONGER-TRAINING-HURTS (carrier stays 20k).
#   R2b-2 H-45-WITH-ANCHOR: min monitor retfm(D1) >= .9 -> ANCHOR-HOLDS;
#          else H-45-BREAKS-ANCHOR@step (vb reads the peak; funnel curve stands).
#   R2b-3 H-48: D3 STOPPED -> H-48-FALSIFIED (deep lane CLOSED, regardless of
#          D4); D3 complete & retfm >= .9 -> H-48-SUPPORTED; complete & retfm
#          < .9 -> DOSE-DESTABILIZES (PI consult). D4 secondary: clean ->
#          REGISTERED-LR-RESTORED; STOPPED -> D4-NAN (deep lane runs 5e-4;
#          seed confound named). NO RETRIES this rung; stops are final.
#   R2b-4 (iff H-48-SUPPORTED): vsel-cold(D3) >= max(cold(D1), C3cold) +
#          max(CNOISE, .01) -> DEEP-COLD-PAYS (D3-recipe = d128 cold arm);
#          else DEEP-OPTIONAL (insurance only).
#   CARRIERS (mechanical, d128): breadth = D1-class@50k iff FUNNEL-GROWS else
#          C3-class@20k (the cheaper recipe carries on tie/neutral), x2 seeds;
#          deep lane iff R2b-3 SUPPORTED (billing per R2b-4). Priced lanes: none.
#   STABILITY: any retfm < .9 named. Descriptive: dose efficacy (late-train
#          A_total on D3/D4 vs C1-free 1.1e7; predict <= 1e5), eta, screens curve.
"""
  .venv/bin/python tools/analyze_sportBr2b.py            # -> runs/analysis/sportBr2b_verdict.txt
  .venv/bin/python tools/analyze_sportBr2b.py --selftest
"""
from __future__ import annotations
import contextlib, io, json, os, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(os.environ.get("QHRRN_RUNS", ROOT / "runs"))
OUT = RUNS / "analysis" / "sportBr2b_verdict.txt"
TAG = "sportBr2b"
ARMS = ["D1", "D2", "D3", "D4"]
C3VB_CONST, C3COLD_CONST, C3P4_CONST = 0.8848, 0.2228, 0.8057
DESC = {"D1": "d96 T6 FPA 50k s0 (winner extended)", "D2": "d96 T6 FPA 20k s1 (noise pair)",
        "D3": "d96 T12 FPA 50k s0 lr5e-4 bnl1e-6", "D4": "d96 T12 FPA 50k s1 lr1e-3 bnl1e-6"}
LINES = []
def say(s=""): LINES.append(str(s)); print(s)
def jload(p):
    p = Path(p); return json.loads(p.read_text()) if p.exists() else None
def cold(a):
    s = jload(RUNS / f"sxeval_p{TAG}{a}" / "full_t64" / "summary_all.json")
    return None if not s else s["exact_acc"]
def cold_vsel(a):
    s = jload(RUNS / f"sxeval_p{TAG}{a}" / "full_t64_valbest" / "summary_all.json")
    return s["exact_acc"] if s else cold(a)
def retfm(a):
    s = jload(RUNS / f"sxeval_p{TAG}{a}" / "retfm_t8" / "summary_all.json")
    return None if not s else s["exact_acc"]
def screen(a, which="vb"):
    s = jload(RUNS / f"sxscreen_p{TAG}{a}_{which}" / "summary_all.json")
    if not s: return None
    return s.get("vote_at_k", {}).get("256", s.get("exact_acc_vote"))
def screen_curve(a):
    out = []
    for d in sorted(RUNS.glob(f"sxscreen_p{TAG}{a}_s*")):
        s = jload(d / "summary_all.json")
        if s: out.append((d.name.split("_")[-1], s.get("vote_at_k", {}).get("256")))
    v = screen(a, "vb")
    if v is not None: out.append(("vb", v))
    return out
def stopped(a):
    p = RUNS / f"pretrain{TAG}_{a}" / "STOPPED.txt"
    return p.read_text().strip() if p.exists() else None
def monitors(a):
    p = RUNS / f"pretrain{TAG}_{a}" / "metrics.jsonl"
    if not p.exists(): return {}
    rows = {}
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line: continue
        try: r = json.loads(line)
        except Exception: continue
        if "monitor" in r: rows[r["monitor"]["step"]] = r["monitor"]   # LAST-WINS
    return rows
def late_A(a):
    p = RUNS / f"pretrain{TAG}_{a}" / "metrics.jsonl"
    if not p.exists(): return None
    vals = {}
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line: continue
        try: r = json.loads(line)
        except Exception: continue
        if "step" in r and "A_total" in r: vals[r["step"]] = r["A_total"]   # LAST-WINS
    tail = [vals[k] for k in sorted(vals)[-40:]]
    if not tail: return None
    tail.sort(); return tail[len(tail) // 2]
def p4(a):
    s = jload(RUNS / f"sxbreadth20k_p{TAG}{a}" / "summary_all.json")
    if not s: return None
    return s.get("vote_at_k", {}).get("128", s.get("exact_acc_vote"))
def f(x, nd=2):  return "  -  " if x is None else f"{x:.{nd}f}"
def fpp(x):      return "  -  " if x is None else f"{100*x:6.2f}"

def analyze():
    LINES.clear(); V = {}
    say("=" * 112); say("PHASE B RUNG 2B VERDICT — d96 (rules registered 2026-08-29; breadth labeled separately)"); say("=" * 112)
    c3s = jload(RUNS / "sxscreen_psportBr2C3_vb" / "summary_all.json")
    C3VB = (c3s.get("vote_at_k", {}).get("256") if c3s else None) or C3VB_CONST
    c3c = jload(RUNS / "sxeval_psportBr2C3" / "full_t64" / "summary_all.json")
    C3COLD = (c3c or {}).get("exact_acc") or C3COLD_CONST
    say(f"refs: C3 screen-vb {fpp(C3VB)} | C3 cold {fpp(C3COLD)} | C3 p4@128 {fpp(C3P4_CONST)} (banked rung-2)")
    say("\nSECTION 1 — per-arm: cold | vsel | retfm | screen-vb v256 | stopped? | screens curve")
    for a in ARMS:
        st = stopped(a)
        say(f"  {a:3s} {DESC[a]:38s} cold {fpp(cold(a))} | vsel {fpp(cold_vsel(a))} | retfm {f(retfm(a))}"
            f" | vb {fpp(screen(a))} | {('STOPPED: ' + st.splitlines()[0][:40]) if st else 'complete'}")
        cur = screen_curve(a)
        if cur: say("       curve: " + "  ".join(f"{k}={fpp(v)}" for k, v in cur))
    unstable = [a for a in ARMS if retfm(a) is not None and retfm(a) < 0.9]
    V["STABILITY"] = "ALL-STABLE" if not unstable else "UNSTABLE:" + ",".join(unstable)
    d2 = screen("D2")
    FN2b = max(abs(C3VB - d2), 0.02) if d2 is not None else 0.02
    d2c = cold("D2")
    CNOISE = max(abs(C3COLD - d2c), 0.01) if d2c is not None else 0.01
    V["FN2b"] = f"{100*FN2b:.2f}pp"; V["CNOISE"] = f"{100*CNOISE:.2f}pp"
    say(f"\nNOISE (matched pair D2-vs-C3): funnel FN2b {100*FN2b:.2f} pp | cold {100*CNOISE:.2f} pp")
    # R2b-1
    d1 = screen("D1"); p4v = p4("D1")
    if d1 is None: V["R2b-1"] = "NO-DATA"
    elif d1 > C3VB + FN2b:
        V["R2b-1"] = "FUNNEL-GROWS"
        if p4v is not None:
            V["B-BAND"] = "B-M2" if p4v >= .85 else ("B-M1-IMPROVED" if p4v >= C3P4_CONST + .011 else "B-M1-FLAT")
    elif d1 >= C3VB - FN2b: V["R2b-1"] = "LENGTH-NEUTRAL"
    else: V["R2b-1"] = "LONGER-TRAINING-HURTS"
    say(f"\nR2b-1 FUNNEL: D1 vb {fpp(d1)} vs C3 {fpp(C3VB)} (FN2b {100*FN2b:.2f}) -> {V['R2b-1']}"
        + (f" | p4@128 {fpp(p4v)} -> {V.get('B-BAND')}" if V.get("B-BAND") else (f" | p4@128 {fpp(p4v)}" if p4v is not None else "")))
    # R2b-2
    mons = monitors("D1")
    if not mons: V["R2b-2"] = "NO-DATA"
    else:
        rfs = [(s, m.get("ret_final_t8")) for s, m in sorted(mons.items()) if m.get("ret_final_t8") is not None]
        bad = [(s, r) for s, r in rfs if r < 0.9]
        V["R2b-2"] = "ANCHOR-HOLDS" if not bad else f"H-45-BREAKS-ANCHOR@{bad[0][0]}"
    say(f"R2b-2 H-45-WITH-ANCHOR: D1 monitor retfm min {f(min((r for _, r in rfs), default=None) if mons else None)} -> {V['R2b-2']}")
    # R2b-3
    if stopped("D3"): V["R2b-3"] = "H-48-FALSIFIED"
    elif retfm("D3") is None: V["R2b-3"] = "NO-DATA"
    elif retfm("D3") >= .9: V["R2b-3"] = "H-48-SUPPORTED"
    else: V["R2b-3"] = "DOSE-DESTABILIZES"
    V["D4"] = "D4-NAN" if stopped("D4") else ("REGISTERED-LR-RESTORED" if retfm("D4") is not None else "NO-DATA")
    say(f"R2b-3 H-48: D3 {'STOPPED' if stopped('D3') else 'complete'} retfm {f(retfm('D3'))} -> {V['R2b-3']} | D4 -> {V['D4']}")
    aD3, aD4 = late_A("D3"), late_A("D4")
    say(f"  dose efficacy (descriptive): late-train A_total D3 {aD3 if aD3 is not None else '-'} | D4 {aD4 if aD4 is not None else '-'} (free-C1 ref ~1.1e7; predict <= 1e5)")
    # R2b-4
    if V["R2b-3"] == "H-48-SUPPORTED":
        d3v = cold_vsel("D3"); bar_parts = [x for x in (cold("D1"), C3COLD) if x is not None]
        bar = (max(bar_parts) + max(CNOISE, 0.01)) if bar_parts else None
        if d3v is None or bar is None: V["R2b-4"] = "NO-DATA"
        else: V["R2b-4"] = "DEEP-COLD-PAYS" if d3v >= bar else "DEEP-OPTIONAL"
        say(f"R2b-4 DEEP BILLING: D3 vsel {fpp(cold_vsel('D3'))} vs bar {fpp(bar)} -> {V['R2b-4']}")
    else:
        V["R2b-4"] = "N/A"; say("R2b-4 DEEP BILLING: N/A (H-48 not supported)")
    # carriers
    bc = "D1-class@50k" if V["R2b-1"] == "FUNNEL-GROWS" else "C3-class@20k"
    deep = "D3-lane" if V["R2b-3"] == "H-48-SUPPORTED" else "none"
    V["CARRIERS"] = f"breadth={bc}x2 deep={deep}"
    say(f"\nCARRIERS (mechanical, d128): breadth {bc} x2 seeds | deep lane {deep}"
        + (f" ({V['R2b-4']})" if V["R2b-4"] not in ("N/A", "NO-DATA") else "") + " | priced lanes: none")
    say(f"STABILITY: {V['STABILITY']}")
    say(); say("SUMMARY: " + "  ".join(f"{k}={v}" for k, v in V.items()))
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text("\n".join(LINES) + "\n"); say(f"artifact -> {OUT}")
    return V


# ---------- self-test ----------
def _mk(root, a, *, cold64=None, vb=None, retfm_=1.0, svb=None, scurve=(), p4v=None,
        stopped_=None, mon_retfm=(), A_tail=None):
    d = root / f"sxeval_p{TAG}{a}"
    if cold64 is not None:
        (d / "full_t64").mkdir(parents=True, exist_ok=True)
        (d / "full_t64" / "summary_all.json").write_text(json.dumps(dict(exact_acc=cold64, n=422786)))
    if retfm_ is not None:
        (d / "retfm_t8").mkdir(parents=True, exist_ok=True)
        (d / "retfm_t8" / "summary_all.json").write_text(json.dumps(dict(exact_acc=retfm_, n=512)))
    if vb is not None:
        (d / "full_t64_valbest").mkdir(parents=True, exist_ok=True)
        (d / "full_t64_valbest" / "summary_all.json").write_text(json.dumps(dict(exact_acc=vb, n=422786)))
    if svb is not None:
        sd = root / f"sxscreen_p{TAG}{a}_vb"; sd.mkdir(parents=True, exist_ok=True)
        (sd / "summary_all.json").write_text(json.dumps(dict(vote_at_k={"16": svb*.55, "256": svb}, n=512)))
    for step, v in scurve:
        sd = root / f"sxscreen_p{TAG}{a}_s{step:06d}"; sd.mkdir(parents=True, exist_ok=True)
        (sd / "summary_all.json").write_text(json.dumps(dict(vote_at_k={"256": v}, n=512)))
    if p4v is not None:
        pd = root / f"sxbreadth20k_p{TAG}{a}"; pd.mkdir(parents=True, exist_ok=True)
        (pd / "summary_all.json").write_text(json.dumps(dict(vote_at_k={"128": p4v}, n=20000)))
    pdir = root / f"pretrain{TAG}_{a}"; pdir.mkdir(parents=True, exist_ok=True)
    if stopped_: (pdir / "STOPPED.txt").write_text(stopped_)
    rows = []
    for s, r in mon_retfm: rows.append(json.dumps({"monitor": {"step": s, "ret_final_t8": r}}))
    if A_tail is not None:
        for i, av in enumerate(A_tail): rows.append(json.dumps({"step": 100 * i, "loss": .5, "A_total": av}))
    if rows: (pdir / "metrics.jsonl").write_text("\n".join(rows) + "\n")

def _refs(root):
    sd = root / "sxscreen_psportBr2C3_vb"; sd.mkdir(parents=True, exist_ok=True)
    (sd / "summary_all.json").write_text(json.dumps(dict(vote_at_k={"256": C3VB_CONST}, n=512)))
    ed = root / "sxeval_psportBr2C3" / "full_t64"; ed.mkdir(parents=True, exist_ok=True)
    (ed / "summary_all.json").write_text(json.dumps(dict(exact_acc=C3COLD_CONST, n=422786)))

def selftest():
    checks = []
    def run(build):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); _refs(root); build(root)
            globals()["RUNS"] = root; globals()["OUT"] = root / "a" / "v.txt"
            with contextlib.redirect_stdout(io.StringIO()): return analyze()
    # world A — the registered predictions come true
    def wA(r):
        _mk(r, "D1", cold64=.245, svb=.91, scurve=[(10000, .30), (25000, .80)], p4v=.86,
            mon_retfm=[(2000, 1.0), (26000, .97), (50000, 1.0)])
        _mk(r, "D2", cold64=.218, svb=.87)
        _mk(r, "D3", cold64=.26, svb=.78, mon_retfm=[(50000, 1.0)], A_tail=[5e4] * 45)
        _mk(r, "D4", cold64=.255, svb=.75, mon_retfm=[(50000, 1.0)])
    v = run(wA)
    checks += [("A grows", v["R2b-1"] == "FUNNEL-GROWS"), ("A B-M2", v.get("B-BAND") == "B-M2"),
               ("A anchor holds", v["R2b-2"] == "ANCHOR-HOLDS"),
               ("A H-48 supported", v["R2b-3"] == "H-48-SUPPORTED"),
               ("A lr restored", v["D4"] == "REGISTERED-LR-RESTORED"),
               ("A deep pays (.26 > max(.245,.2228)+.012)", v["R2b-4"] == "DEEP-COLD-PAYS"),
               ("A carrier D1", "breadth=D1-class@50k" in v["CARRIERS"] and "deep=D3-lane" in v["CARRIERS"]),
               ("A stable", v["STABILITY"] == "ALL-STABLE"),
               ("A FN2b floor 2pp (|.8848-.87|=1.48)", v["FN2b"] == "2.00pp")]
    # world B — neutral length; D3 stopped -> falsified regardless of D4
    def wB(r):
        _mk(r, "D1", cold64=.24, svb=.88, mon_retfm=[(50000, 1.0)])
        _mk(r, "D2", cold64=.22, svb=.86)
        _mk(r, "D3", cold64=.21, svb=.60, stopped_="STOPPED final step 20000 (NaN halt)", mon_retfm=[(18000, 1.0)])
        _mk(r, "D4", cold64=.25, svb=.72, mon_retfm=[(50000, 1.0)])
    v = run(wB)
    checks += [("B neutral", v["R2b-1"] == "LENGTH-NEUTRAL"),
               ("B H-48 falsified", v["R2b-3"] == "H-48-FALSIFIED"),
               ("B R2b-4 N/A", v["R2b-4"] == "N/A"),
               ("B carrier C3 no deep", "breadth=C3-class@20k" in v["CARRIERS"] and "deep=none" in v["CARRIERS"])]
    # world C — anchor breaks; hurts; dose destabilizes; D4 NaN
    def wC(r):
        _mk(r, "D1", cold64=.20, svb=.80, mon_retfm=[(2000, 1.0), (30000, .60), (50000, .95)])
        _mk(r, "D2", cold64=.225, svb=.90)
        _mk(r, "D3", cold64=.22, svb=.55, retfm_=.7, mon_retfm=[(50000, .7)])
        _mk(r, "D4", cold64=None, retfm_=None, stopped_="STOPPED final step 10000 (NaN halt)")
    v = run(wC)
    checks += [("C hurts", v["R2b-1"] == "LONGER-TRAINING-HURTS"),
               ("C breaks@30000", v["R2b-2"] == "H-45-BREAKS-ANCHOR@30000"),
               ("C dose destabilizes", v["R2b-3"] == "DOSE-DESTABILIZES"),
               ("C D4 nan", v["D4"] == "D4-NAN"),
               ("C unstable named", "D3" in v["STABILITY"])]
    # world D — no data
    v = run(lambda r: None)
    checks += [("D no data R2b-1", v["R2b-1"] == "NO-DATA"), ("D no data R2b-3", v["R2b-3"] == "NO-DATA")]
    ok = 0
    for name, passed in checks:
        print(("  PASS  " if passed else "  FAIL  ") + name); ok += bool(passed)
    print(f"selftest: {ok}/{len(checks)}")
    return ok == len(checks)


if __name__ == "__main__":
    if "--selftest" in sys.argv: sys.exit(0 if selftest() else 1)
    analyze()
