# Ledger: PHASE B RUNG 2 ANALYZER — written BEFORE any rung-2 run (2026-08-27);
# the launch entry locks these rules verbatim. Reads the d96 arms (tag sportBr2)
# plus the banked references: S5-d16 20k vote@128 = .6862 bar and B2-d64's
# 66.16 / 27.0 / 55.9 reference values named in the registration.
#   WINNER = argmax screen-vb vote@256 among arms with retfm_t8 >= .5.
#   G-R2-1 BREADTH  winner PHASE4 20k vote@128 >= .6862 -> BREADTH-SCALES;
#          >= .6362 -> BREADTH-FLAT; else BREADTH-SHRINKS. B-bands labeled.
#   G-R2-2 COLD  best val-selected over C1/C1s1/C2/C3: >= .30 ON-TRACK /
#          >= .25 WEAK / else STALLS. HEADLINE = best FINAL cold -> M-bands.
#   G-R2-3 PRICED-BREADTH (H-47 + the PI conjecture, registered 2026-08-27):
#          P-A fires iff C2 (std-beta) screen-vb v256 > .270 AND non-flat
#          (v256/v16 >= 1.25); P-B fires iff C4 (beta/3) screen-vb v256 >= .55
#          AND non-flat. Adjudication: BOTH -> CONJECTURE-STRONG; P-A only ->
#          CODE-EFFICIENCY; P-B only -> BUDGET-MEDIATED; neither ->
#          PRICED-BREADTH-DEAD.
#   G-R2-4 INSURANCE  C3 RESCUED iff retfm >= .9 AND screen-vb v256 >= .559
#          (B4-d64's 5k-ckpt value); RESCUED-AND-COMPETITIVE additionally iff
#          >= carrier max - .05 -> joins d128.
#   G-R2-5 H-46-FUNNEL  per arm: max pairwise |delta| across available screens
#          (m1, m2, vb) vote@256; arms > 5pp named with direction (later-vs-
#          earlier screen kinds ordered m1 < m2 < vb).
#   STABILITY any retfm < .9 named. FUNNEL-NOISE = max(|C1 - C1s1| screen-vb
#          v256, 2pp); sub-2x-noise contrasts labeled by the reader.
#   CARRIERS (mechanical, d128): breadth carrier = winner base (s1 stripped)
#          x2 seeds; secondary = argmax screen-vb among gate-passers
#          {C2 iff P-A, C4 iff P-B, C3 iff RESCUED}; cold carrier labeled.
"""
  .venv/bin/python tools/analyze_sportBr2.py            # -> runs/analysis/sportBr2_verdict.txt
  .venv/bin/python tools/analyze_sportBr2.py --selftest
"""
from __future__ import annotations
import contextlib, io, json, os, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(os.environ.get("QHRRN_RUNS", ROOT / "runs"))
OUT = RUNS / "analysis" / "sportBr2_verdict.txt"
TAG = "sportBr2"
ARMS = ["C1", "C1s1", "C2", "C3", "C4"]
COLD_SET = ["C1", "C1s1", "C2", "C3"]
S5_BAR, D64_FLAT_PAD = 0.6862, 0.05
C2_BAR, C4_BAR, NONFLAT, C3_SCR_BAR = 0.270, 0.55, 1.25, 0.559
DESC = {"C1": "d96 plain T12 FPA 50k s0", "C1s1": "d96 plain T12 FPA 50k s1",
        "C2": "d96 priced T12 50k (std beta)", "C3": "d96 plain T6 FPA 20k (insurance)",
        "C4": "d96 priced T12 50k (beta/3)"}
LINES = []
def say(s=""): LINES.append(str(s)); print(s)
def jload(p):
    p = Path(p); return json.loads(p.read_text()) if p.exists() else None
def summ(a, kind): return jload(RUNS / f"sxeval_p{TAG}{a}" / kind / "summary_all.json")
def cold(a, vb=False):
    s = summ(a, "full_t64_valbest" if vb else "full_t64")
    return None if not s else s["exact_acc"]
def cold_vsel(a):
    v = cold(a, vb=True); return v if v is not None else cold(a)
def retfm(a):
    s = summ(a, "retfm_t8"); return None if not s else s["exact_acc"]
def screen(a, which):
    s = jload(RUNS / f"sxscreen_p{TAG}{a}_{which}" / "summary_all.json")
    if not s: return None
    return s.get("vote_at_k", {}).get("256", s.get("exact_acc_vote"))
def screen16(a, which):
    s = jload(RUNS / f"sxscreen_p{TAG}{a}_{which}" / "summary_all.json")
    if not s: return None
    return s.get("vote_at_k", {}).get("16")
def phase4(a):
    s = jload(RUNS / f"sxbreadth20k_p{TAG}{a}" / "summary_all.json")
    if not s: return None
    return s.get("vote_at_k", {}).get("128", s.get("exact_acc_vote"))
def nonflat(a):
    v, v16 = screen(a, "vb"), screen16(a, "vb")
    if v is None or not v16: return None
    return (v / v16) >= NONFLAT
def f(x, nd=2):  return "  -  " if x is None else f"{x:.{nd}f}"
def fpp(x):      return "  -  " if x is None else f"{100*x:6.2f}"

def analyze():
    LINES.clear(); V = {}
    say("=" * 112); say("PHASE B RUNG 2 VERDICT — d96 full-width (rules registered 2026-08-27; breadth labeled separately)"); say("=" * 112)
    say("\nSECTION 1 — per-arm: cold final | vsel | retfm | screens v256 m1/m2/vb | vb-nonflat | p4@128")
    for a in ARMS:
        nf = nonflat(a)
        say(f"  {a:5s} {DESC[a]:34s} cold {fpp(cold(a))} | vsel {fpp(cold_vsel(a))} | retfm {f(retfm(a))}"
            f" | scr {fpp(screen(a,'m1'))}/{fpp(screen(a,'m2'))}/{fpp(screen(a,'vb'))}"
            f" | nf {('Y' if nf else 'n') if nf is not None else '-'} | p4 {fpp(phase4(a))}")
    unstable = [a for a in ARMS if retfm(a) is not None and retfm(a) < 0.9]
    V["STABILITY"] = "ALL-STABLE" if not unstable else "UNSTABLE:" + ",".join(unstable)
    fn = None
    if screen("C1", "vb") is not None and screen("C1s1", "vb") is not None:
        fn = abs(screen("C1", "vb") - screen("C1s1", "vb")) * 100
    FNOISE = max(fn if fn is not None else 0.0, 2.0)
    V["FUNNEL-NOISE"] = f"{FNOISE:.2f}pp"
    say(f"\nSTABILITY: {V['STABILITY']} | FUNNEL-NOISE (|C1-C1s1| scr-vb v256, floor 2): {FNOISE:.2f} pp")
    cand = [a for a in ARMS if screen(a, "vb") is not None and (retfm(a) is None or retfm(a) >= 0.5)]
    winner = max(cand, key=lambda a: screen(a, "vb"), default=None)
    V["WINNER"] = winner or "NO-DATA"
    p4 = phase4(winner) if winner else None
    if p4 is None: V["G-R2-1"] = "NO-DATA"
    else:
        V["G-R2-1"] = "BREADTH-SCALES" if p4 >= S5_BAR else ("BREADTH-FLAT" if p4 >= S5_BAR - D64_FLAT_PAD else "BREADTH-SHRINKS")
        V["B-BAND"] = "B-M3" if p4 >= .95 else "B-M2" if p4 >= .85 else "B-M1" if p4 >= .5 else "BELOW-B-M1"
    say(f"\nG-R2-1 BREADTH: winner {winner} PHASE4 20k vote@128 {fpp(p4)} vs S5-d16 {fpp(S5_BAR)} -> {V['G-R2-1']}"
        + (f" | band {V.get('B-BAND')}" if V.get("B-BAND") else ""))
    vsel = {a: cold_vsel(a) for a in COLD_SET if cold_vsel(a) is not None}
    if vsel:
        cbest = max(vsel, key=vsel.get); cb = vsel[cbest]
        V["G-R2-2"] = "COLD-ON-TRACK" if cb >= .30 else ("COLD-WEAK" if cb >= .25 else "COLD-STALLS")
        say(f"\nG-R2-2 COLD: best val-selected {cbest} {fpp(cb)} -> {V['G-R2-2']}")
    else: V["G-R2-2"] = "NO-DATA"; cbest = None
    H = {a: cold(a) for a in ARMS if cold(a) is not None}
    if H:
        hb = max(H, key=H.get); acc = H[hb]
        V["BAND"] = "M3" if acc >= .95 else "M2" if acc >= .85 else "M1" if acc >= .5 else "BELOW-M1"
        V["BEST"] = f"{hb}={acc:.4f}"
        say(f"HEADLINE (cold, FINAL ckpts): {hb} {acc:.4f} -> BAND {V['BAND']}")
    else: V["BAND"] = "NO-DATA"
    # G-R2-3 priced breadth
    pa = pb = None
    if screen("C2", "vb") is not None and nonflat("C2") is not None:
        pa = screen("C2", "vb") > C2_BAR and nonflat("C2")
    if screen("C4", "vb") is not None and nonflat("C4") is not None:
        pb = screen("C4", "vb") >= C4_BAR and nonflat("C4")
    if pa is None and pb is None: V["G-R2-3"] = "NO-DATA"
    else:
        V["P-A"] = "YES" if pa else "no"; V["P-B"] = "YES" if pb else "no"
        V["G-R2-3"] = ("CONJECTURE-STRONG" if (pa and pb) else "CODE-EFFICIENCY" if pa
                       else "BUDGET-MEDIATED" if pb else "PRICED-BREADTH-DEAD")
    say(f"\nG-R2-3 PRICED-BREADTH: P-A(C2 v256 {fpp(screen('C2','vb'))} > {C2_BAR:.2f} & nonflat)={V.get('P-A','-')}"
        f" | P-B(C4 v256 {fpp(screen('C4','vb'))} >= {C4_BAR:.2f} & nonflat)={V.get('P-B','-')} -> {V['G-R2-3']}")
    # G-R2-4 insurance
    if retfm("C3") is None or screen("C3", "vb") is None: V["G-R2-4"] = "NO-DATA"
    else:
        resc = retfm("C3") >= .9 and screen("C3", "vb") >= C3_SCR_BAR
        carrier_max = max((x for x in (screen("C1", "vb"), screen("C1s1", "vb")) if x is not None), default=None)
        comp = resc and carrier_max is not None and screen("C3", "vb") >= carrier_max - .05
        V["G-R2-4"] = "RESCUED-AND-COMPETITIVE" if comp else ("RESCUED" if resc else "NOT-RESCUED")
    say(f"G-R2-4 INSURANCE: C3 retfm {f(retfm('C3'))} scr-vb {fpp(screen('C3','vb'))} -> {V['G-R2-4']}")
    # G-R2-5 funnel drift across screens
    drift = []
    say("\nG-R2-5 H-46-FUNNEL (max pairwise |delta| v256 across m1/m2/vb; >5pp named with direction):")
    for a in ARMS:
        vals = [(w, screen(a, w)) for w in ("m1", "m2", "vb")]
        have = [(w, v) for w, v in vals if v is not None]
        if len(have) < 2: say(f"  {a:5s} insufficient screens"); continue
        mx = max(abs(v1 - v2) for _, v1 in have for _, v2 in have) * 100
        direction = "GROWTH" if have[-1][1] >= have[0][1] else "COLLAPSE"
        flag = mx > 5
        if flag: drift.append(f"{a}({direction})")
        say(f"  {a:5s} screens " + " ".join(f"{w}={fpp(v)}" for w, v in have) + f" | max-drift {mx:.2f}pp{' <- ' + direction if flag else ''}")
    V["G-R2-5"] = "FUNNEL-STATIONARY" if not drift else "FUNNEL-DRIFT:" + ",".join(drift)
    # carriers
    if winner:
        bc = winner[:-2] if winner.endswith("s1") else winner
        gate_pass = {}
        if V.get("P-A") == "YES": gate_pass["C2"] = screen("C2", "vb")
        if V.get("P-B") == "YES": gate_pass["C4"] = screen("C4", "vb")
        if V.get("G-R2-4", "").startswith("RESCUED"): gate_pass["C3"] = screen("C3", "vb")
        sec = max(gate_pass, key=gate_pass.get) if gate_pass else "-"
        V["CARRIERS"] = f"breadth={bc}x2 secondary={sec} cold={cbest or '-'}"
        say(f"\nCARRIERS (mechanical, d128): breadth {bc} x2 seeds | secondary {sec} | cold carrier {cbest or '-'} (labeled)")
    else: V["CARRIERS"] = "NO-DATA"
    say(); say("SUMMARY: " + "  ".join(f"{k}={v}" for k, v in V.items()))
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text("\n".join(LINES) + "\n"); say(f"artifact -> {OUT}")
    return V


# ---------- self-test ----------
def _arm(root, a, *, cold64=None, vb=None, retfm_=1.0, m1=None, m2=None, svb=None, s16=None, p4=None):
    d = root / f"sxeval_p{TAG}{a}"
    if cold64 is not None:
        (d / "full_t64").mkdir(parents=True, exist_ok=True)
        (d / "full_t64" / "summary_all.json").write_text(json.dumps(dict(exact_acc=cold64, n=422786)))
    (d / "retfm_t8").mkdir(parents=True, exist_ok=True)
    (d / "retfm_t8" / "summary_all.json").write_text(json.dumps(dict(exact_acc=retfm_, n=512)))
    if vb is not None:
        (d / "full_t64_valbest").mkdir(parents=True, exist_ok=True)
        (d / "full_t64_valbest" / "summary_all.json").write_text(json.dumps(dict(exact_acc=vb, n=422786)))
    for which, v in (("m1", m1), ("m2", m2), ("vb", svb)):
        if v is None: continue
        sd = root / f"sxscreen_p{TAG}{a}_{which}"; sd.mkdir(parents=True, exist_ok=True)
        v16 = s16 if (which == "vb" and s16 is not None) else v * 0.55
        (sd / "summary_all.json").write_text(json.dumps(dict(vote_at_k={"16": v16, "128": v - .05, "256": v}, n=512)))
    if p4 is not None:
        pd = root / f"sxbreadth20k_p{TAG}{a}"; pd.mkdir(parents=True, exist_ok=True)
        (pd / "summary_all.json").write_text(json.dumps(dict(vote_at_k={"128": p4}, n=20000)))

def selftest():
    checks = []
    def run(build):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); build(root)
            globals()["RUNS"] = root; globals()["OUT"] = root / "a" / "v.txt"
            with contextlib.redirect_stdout(io.StringIO()): return analyze()
    def world(root, **kw):
        o = dict(C1=dict(cold64=.31, vb=.32, m1=.60, m2=.72, svb=.82, p4=.71),
                 C1s1=dict(cold64=.30, vb=.30, m1=.58, m2=.70, svb=.80),
                 C2=dict(cold64=.22, vb=.22, m1=.65, m2=.55, svb=.45),
                 C3=dict(cold64=.15, vb=.16, m1=.50, m2=.58, svb=.62),
                 C4=dict(cold64=.20, vb=.21, m1=.60, m2=.62, svb=.63))
        o.update(kw)
        for a, k in o.items(): _arm(root, a, **k)
    # world 1: the registered predictions come true
    v = run(world)
    checks += [("winner C1", v["WINNER"] == "C1"), ("scales", v["G-R2-1"] == "BREADTH-SCALES"),
               ("band B-M1", v.get("B-BAND") == "B-M1"), ("cold on-track (.32)", v["G-R2-2"] == "COLD-ON-TRACK"),
               ("headline below-M1", v["BAND"] == "BELOW-M1"),
               ("P-A yes P-B yes -> strong", v["G-R2-3"] == "CONJECTURE-STRONG"),
               ("insurance rescued (not competitive: .62 < .82-.05)", v["G-R2-4"] == "RESCUED"),
               ("C2 drift named COLLAPSE", "C2(COLLAPSE)" in v["G-R2-5"]),
               ("stable", v["STABILITY"] == "ALL-STABLE"),
               ("carriers C1x2 sec C4", "breadth=C1x2" in v["CARRIERS"] and "secondary=C4" in v["CARRIERS"])]
    # world 2: budget-mediated only (C2 flat-k), insurance collapses
    v = run(lambda r: world(r,
        C2=dict(cold64=.22, vb=.22, m1=.30, m2=.28, svb=.26, s16=.24),
        C3=dict(cold64=.08, vb=.12, retfm_=.4, m1=.55, m2=.40, svb=.35),
        C1=dict(cold64=.28, vb=.29, m1=.60, m2=.68, svb=.78, p4=.65),
        C1s1=dict(cold64=.27, vb=.28, m1=.58, m2=.65, svb=.70)))
    checks += [("flat", v["G-R2-1"] == "BREADTH-FLAT"), ("cold weak (.29)", v["G-R2-2"] == "COLD-WEAK"),
               ("budget-mediated", v["G-R2-3"] == "BUDGET-MEDIATED"),
               ("insurance not rescued", v["G-R2-4"] == "NOT-RESCUED"),
               ("unstable C3 named", v["STABILITY"] == "UNSTABLE:C3"),
               ("sec C4", "secondary=C4" in v["CARRIERS"])]
    # world 3: priced breadth dead; shrinks
    v = run(lambda r: world(r,
        C1=dict(cold64=.26, vb=.26, m1=.55, m2=.60, svb=.70, p4=.60),
        C1s1=dict(cold64=.25, vb=.25, m1=.52, m2=.58, svb=.66),
        C2=dict(cold64=.20, vb=.20, m1=.30, m2=.28, svb=.26, s16=.24),
        C4=dict(cold64=.18, vb=.18, m1=.28, m2=.26, svb=.25, s16=.23)))
    checks += [("shrinks", v["G-R2-1"] == "BREADTH-SHRINKS"), ("dead", v["G-R2-3"] == "PRICED-BREADTH-DEAD")]
    # world 4: no data
    v = run(lambda r: None)
    checks += [("no data winner", v["WINNER"] == "NO-DATA"), ("no data g1", v["G-R2-1"] == "NO-DATA")]
    ok = 0
    for name, passed in checks:
        print(("  PASS  " if passed else "  FAIL  ") + name); ok += bool(passed)
    print(f"selftest: {ok}/{len(checks)}")
    return ok == len(checks)


if __name__ == "__main__":
    if "--selftest" in sys.argv: sys.exit(0 if selftest() else 1)
    analyze()
