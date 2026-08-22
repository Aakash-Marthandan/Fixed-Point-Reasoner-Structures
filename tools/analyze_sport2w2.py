# Ledger: SPRINT S2 WAVE 2 ANALYZER — written BEFORE any wave-2 run (2026-08-22),
# pre-registration in code; the launch entry locks these rules verbatim. Reads the
# wave-1 cells (tag sport2: S0 priced T6, S5 plain T6, S4 T24, S7 RI+NI+T12) and
# the wave-2 arms (tag sport2w2) + breadth scans + the 20k-subsample breadth number.
#   M0-W2  W1 (plain T6 @50k) val@t64 exact >= .20 -> STRICT trainable at 50k
#          (STRICT restored as a co-primary labeled row); else GEN stays primary.
#   HEADLINE  best cold full-test exact over ALL arms at t in {6, 64} -> bands
#          M1 >= .50 | M2 >= .85 | M3 >= .95 | else BELOW-M1 (unchanged bands).
#   BREADTH (labeled separately, never the headline): (a) the best arm's 20k-
#          puzzle seeded subsample verify-and-vote at k=128, t=64 (+-95% CI);
#          (b) S5's strat-512 nested k-curve. M1-ON-BREADTH if (a) >= .50
#          (fallback (b) at k=128 when (a) is absent).
#   SEED NOISE  |W1-W1s1| and |W4-W4s1| on cold@t64 (pp); NOISE = max(that, 1 pp);
#          any contrast with |delta| < 2*NOISE is labeled WITHIN-SEED-NOISE.
#   P1 BUDGET  W1-S5 >= +5 pp -> BUDGET-PAYS; <= +2 -> BUDGET-NOT-BINDING; else
#          indeterminate. Budget curve 20k/50k/100k (S5/W1/W13) reported; W13 >= W1
#          within noise = MONOTONE.
#   P2 DEPTH (plain base)  W2-S5 >= +3 pp AND ret(W2) >= .9 -> DEPTH-PAYS; RI+NI:
#          W3 vote@16 - W2 vote@16 >= +3 pp -> RI+NI-RAISES-BREADTH; W3-W2 cold reported.
#   P3 WIDTH (plain)  W4-S5 >= +3 pp -> WIDTH-PAYS.
#   P4 PRICE x SCALE (H-44, PI 2026-08-22): priced cells W8 (d32@20k), W9 (d16@50k)
#          vs S0 (d16@20k): a cell is STABLE if retention >= .9; RECOVERED if stable
#          AND cold@t64 >= 5 pp; HORIZON if retention < .5 AND cold < 2 pp.
#          H-44 SUPPORTED (scale attenuates the price damage) if W8 or W9 is
#          RECOVERED-or-STABLE; FALSIFIED-AT-THESE-SCALES if both are HORIZON;
#          else INDETERMINATE. DOSE (H-43's registered test): W6 (3e-6) STABLE ->
#          'a 10x lower price keeps the equilibrium' with the compression ratio
#          I_total(W6)/I_total(S5) reported; HORIZON -> the dose kills too.
#   P5 GEN  W5-S5 >= +3 pp -> GEN-TRANSFERS; <= 0 -> GEN-NO-TRANSFER; else indet.
#   P6 BOX4 (control)  |W7-S5| <= 2 pp -> GEOMETRY-NULL (predicted); W7-S5 >= +3 ->
#          GEOMETRY-MATTERS; <= -3 -> BOX4-HURTS.
#   P7 BREADTH SCAN  S5 strat vote@128 (t64) >= .50 -> prediction HOLDS; else report
#          saturation (vote@256 - vote@128 < 2 pp = SATURATED-BELOW-M1). RI+NI breadth
#          (H-37 under multi-init deployment): S7 hit-rate > S5 hit-rate -> directional.
"""
  .venv/bin/python tools/analyze_sport2w2.py            # -> runs/analysis/sport2w2_verdict.txt
  .venv/bin/python tools/analyze_sport2w2.py --selftest
"""
from __future__ import annotations
import contextlib, io, json, math, os, sys, tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(os.environ.get("QHRRN_RUNS", ROOT / "runs"))
OUT = RUNS / "analysis" / "sport2w2_verdict.txt"
T1, T2 = "sport2", "sport2w2"
W1_ARMS = ["S0", "S5", "S4", "S7"]
W2_ARMS = ["W1", "W13", "W2", "W3", "W4", "W8", "W9", "W6", "W5", "W7", "W1s1", "W4s1"]
DESC = {"S0": "priced T6 d16 20k (w1)", "S5": "plain T6 d16 20k (w1)", "S4": "priced T24 d16 20k (w1)", "S7": "priced RI+NI T12 20k (w1)",
        "W1": "plain T6 d16 50k", "W13": "plain T6 d16 100k", "W2": "plain T12 d16 30k", "W3": "plain T12+RI+NI 30k",
        "W4": "plain T6 d32 20k", "W8": "priced T6 d32 20k", "W9": "priced T6 d16 50k", "W6": "beta 3e-6 T6 d16 20k",
        "W5": "GEN gen20k->ft20k plain T6", "W7": "plain T6 d16 20k box4", "W1s1": "W1 seed 1", "W4s1": "W4 seed 1"}
LINES = []
def say(s=""): LINES.append(str(s)); print(s)
def jload(p):
    p = Path(p); return json.loads(p.read_text()) if p.exists() else None
def tag_of(arm): return T1 if arm.startswith("S") else T2
def summ(arm, kind):   # kind: full_t6 full_t64 strat_t6 strat_t64 strat_t256 val_t64 ret_t8
    return jload(RUNS / f"sxeval_p{tag_of(arm)}{arm}" / kind / "summary_all.json")
def cold(arm, t=64):
    s = summ(arm, f"full_t{t}"); return None if not s else s["exact_acc"]
def vote16(arm):
    s = summ(arm, "strat_t64"); return None if not s else s.get("exact_acc_vote")
def retention(arm):
    s = summ(arm, "ret_t8")
    if s: return s["exact_acc"]
    p = RUNS / f"sudprobe_p{tag_of(arm)}{arm}" / "results.jsonl"     # wave-1 arms: the probe's e3b retention
    if not p.exists(): return None
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return float(np.mean([r["gt_retention"] for r in rows])) if rows else None
def eta(arm):
    for k in ("full_t64", "strat_t64", "full_t6"):
        s = summ(arm, k)
        if s and s.get("eta") is not None: return s["eta"]
    return None
def itotal(arm):
    p = RUNS / f"pretrain{tag_of(arm)}_{arm}" / "metrics.jsonl"
    if not p.exists(): return None
    last = None
    for l in p.read_text().splitlines():
        try: r = json.loads(l)
        except Exception: continue
        if "I_total" in r: last = r["I_total"]
    return last
def val64(arm):
    s = summ(arm, "val_t64"); return None if not s else s["exact_acc"]
def scan(arm, t=64, k=256):
    return jload(RUNS / f"sxbreadth_{arm}_t{t}_k{k}" / "summary_all.json")
def breadth20k():
    out = {}
    for d in RUNS.glob(f"sxbreadth20k_p{T2}*"):
        s = jload(d / "summary_all.json")
        if s: out[d.name.replace(f"sxbreadth20k_p{T2}", "")] = s
    return out
def pp(x): return None if x is None else 100.0 * x
def f(x, nd=2):  return "  -  " if x is None else f"{x:.{nd}f}"
def fpp(x):     return "  -  " if x is None else f"{100*x:6.2f}"
def delta(a, b): return None if (a is None or b is None) else 100.0 * (a - b)
def ci95(p, n):
    if p is None or not n: return None
    return 1.96 * math.sqrt(max(p * (1 - p), 1e-12) / n)

def analyze():
    LINES.clear(); V = {}
    say("=" * 110); say("SPRINT S2 WAVE 2 VERDICT — Sudoku-Extreme (rules registered 2026-08-22; full 423k test exact; breadth labeled separately)"); say("=" * 110)
    say(); say("SECTION 1 — per-arm table (cold full-test exact @t6/@t64 | strat-512 vote@16 (t64) | retention (t8 solution-init; w1: probe) | val@t64 | eta | I_total)")
    for arm in W1_ARMS + W2_ARMS:
        say(f"  {arm:5s} {DESC[arm]:32s} cold t6 {fpp(cold(arm, 6))} t64 {fpp(cold(arm, 64))} | vote16 {fpp(vote16(arm))} | ret {f(retention(arm))} | val64 {fpp(val64(arm))} | eta {f(eta(arm), 3)} | I {f(itotal(arm), 0)}")
    # seed noise
    dW1 = delta(cold("W1"), cold("W1s1")); dW4 = delta(cold("W4"), cold("W4s1"))
    seeds = [abs(x) for x in (dW1, dW4) if x is not None]
    NOISE = max(max(seeds) if seeds else 0.0, 1.0)
    V["SEED"] = f"dW1={f(dW1)} dW4={f(dW4)} NOISE={NOISE:.2f}pp"
    say(); say(f"SEED NOISE: |W1-W1s1| {f(dW1)} pp, |W4-W4s1| {f(dW4)} pp -> NOISE = {NOISE:.2f} pp (contrasts below {2*NOISE:.2f} pp are WITHIN-SEED-NOISE)")
    def lab(d, thr_hi=None):
        return "" if d is None or abs(d) >= 2 * NOISE else " [WITHIN-SEED-NOISE]"
    # M0
    v = val64("W1")
    if v is None: V["M0"] = "NO-DATA"; say("\nM0-W2: no W1 val@t64")
    else:
        V["M0"] = "STRICT-TRAINABLE" if v >= .20 else "GEN-PRIMARY"
        say(f"\nM0-W2: W1 val@t64 exact {fpp(v)} -> {V['M0']}" + (" (STRICT restored as co-primary labeled row)" if v >= .20 else " (STRICT under-trained at 50k too; GEN stays the primary labeled row)"))
    # headline
    H = {}
    for arm in W1_ARMS + W2_ARMS:
        for t in (6, 64):
            c = cold(arm, t)
            if c is not None and (arm not in H or c > H[arm][0]): H[arm] = (c, t)
    if H:
        best = max(H, key=lambda a: H[a][0]); acc, tb = H[best]
        band = "M3" if acc >= .95 else "M2" if acc >= .85 else "M1" if acc >= .50 else "BELOW-M1"
        V["BAND"] = band; V["BEST"] = f"{best}@t{tb}={acc:.4f}"
        say(f"\nHEADLINE (cold): best arm {best} ({DESC[best]}) at t={tb}: exact {acc:.4f} full test -> BAND {band}")
    else: V["BAND"] = "NO-DATA"; say("\nHEADLINE: no full-test data")
    # breadth
    b20 = breadth20k()
    say("\nBREADTH (labeled separately):")
    bb = None
    for arm, s in sorted(b20.items()):
        va = s.get("vote_at_k", {}).get("128", s.get("exact_acc_vote")); n = s.get("n")
        say(f"  20k-subsample {arm}: cold {fpp(s.get('exact_acc'))} vote@128 {fpp(va)} +-{f(pp(ci95(va, n)))}pp (n={n}) | k-curve {s.get('vote_at_k')}")
        if va is not None and (bb is None or va > bb[1]): bb = (arm, va, n)
    s5 = scan("S5", 64, 256)
    if s5:
        say(f"  S5 strat-512 t64 k-curve: {s5.get('vote_at_k')} | hits/256 mean {f(s5.get('mi_hits_mean'))}")
    for arm in ("S4", "S7"):
        s = scan(arm, 64, 256)
        if s: say(f"  {arm} strat-512 t64 k-curve: {s.get('vote_at_k')} | hits/256 mean {f(s.get('mi_hits_mean'))}")
    s56 = scan("S5", 6, 256)
    if s56: say(f"  S5 strat-512 t6 k-curve: {s56.get('vote_at_k')}")
    if bb: V["BREADTH"] = f"{bb[0]} vote@128={bb[1]:.4f} (20k)"; V["BREADTH-BAND"] = "M1-ON-BREADTH" if bb[1] >= .5 else "BELOW-M1-ON-BREADTH"
    elif s5 and s5.get("vote_at_k", {}).get("128") is not None:
        v128 = s5["vote_at_k"]["128"]; V["BREADTH"] = f"S5 strat vote@128={v128:.4f}"; V["BREADTH-BAND"] = "M1-ON-BREADTH" if v128 >= .5 else "BELOW-M1-ON-BREADTH"
    else: V["BREADTH"] = "NO-DATA"; V["BREADTH-BAND"] = "NO-DATA"
    say(f"  -> {V['BREADTH']} : {V['BREADTH-BAND']}")
    # predictions
    say("\nREGISTERED PREDICTIONS (cold full-test exact @t64 unless noted; pp = percentage points)")
    d = delta(cold("W1"), cold("S5"))
    if d is None: V["P1"] = "NO-DATA"
    else: V["P1"] = "BUDGET-PAYS" if d >= 5 else ("BUDGET-NOT-BINDING" if d <= 2 else "INDETERMINATE")
    d13 = delta(cold("W13"), cold("W1"))
    mono = "" if d13 is None else (" | W13-W1 " + f"{d13:+.2f}pp -> " + ("MONOTONE" if d13 >= -NOISE else "NON-MONOTONE"))
    say(f"  P1 BUDGET: W1-S5 {f(d)}pp{lab(d)} -> {V['P1']}{mono}")
    d2 = delta(cold("W2"), cold("S5")); r2 = retention("W2")
    if d2 is None or r2 is None: V["P2"] = "NO-DATA"
    else: V["P2"] = "DEPTH-PAYS" if (d2 >= 3 and r2 >= .9) else ("DEPTH-NULL" if d2 < 3 else "DEPTH-BUT-NOT-EQUILIBRIUM")
    dv = delta(vote16("W3"), vote16("W2")); d32 = delta(cold("W3"), cold("W2"))
    rini = "NO-DATA" if dv is None else ("RI+NI-RAISES-BREADTH" if dv >= 3 else "RI+NI-BREADTH-FLAT")
    V["P2b"] = rini
    say(f"  P2 DEPTH: W2-S5 {f(d2)}pp{lab(d2)}, ret(W2) {f(r2)} -> {V['P2']} | RI+NI: vote16 W3-W2 {f(dv)}pp, cold W3-W2 {f(d32)}pp -> {rini}")
    d4 = delta(cold("W4"), cold("S5"))
    V["P3"] = "NO-DATA" if d4 is None else ("WIDTH-PAYS" if d4 >= 3 else "WIDTH-NULL")
    say(f"  P3 WIDTH: W4-S5 {f(d4)}pp{lab(d4)} -> {V['P3']}")
    # price x scale
    say("  P4 PRICE x SCALE (H-44): cell | cold@t64 | retention | eta | class")
    def cls(arm):
        c, r = cold(arm), retention(arm)
        if c is None or r is None: return "NO-DATA"
        if r >= .9 and c >= .05: return "RECOVERED"
        if r >= .9: return "STABLE"
        if r < .5 and c < .02: return "HORIZON"
        return "PARTIAL"
    classes = {}
    for arm in ("S0", "W9", "W8", "W6", "S5"):
        classes[arm] = cls(arm)
        say(f"     {arm:4s} {DESC[arm]:28s} cold {fpp(cold(arm))} | ret {f(retention(arm))} | eta {f(eta(arm), 3)} | {classes[arm]}")
    c8, c9 = classes["W8"], classes["W9"]
    if "NO-DATA" in (c8, c9): V["H44"] = "NO-DATA"
    elif c8 in ("RECOVERED", "STABLE") or c9 in ("RECOVERED", "STABLE"): V["H44"] = "SUPPORTED (price damage attenuates with scale: " + ",".join(a for a in ("W8", "W9") if classes[a] in ("RECOVERED", "STABLE")) + ")"
    elif c8 == "HORIZON" and c9 == "HORIZON": V["H44"] = "FALSIFIED-AT-THESE-SCALES (d32@20k and d16@50k stay horizon maps)"
    else: V["H44"] = "INDETERMINATE"
    c6 = classes["W6"]; comp = None
    if itotal("W6") and itotal("S5"): comp = itotal("S5") / itotal("W6")
    V["DOSE"] = "NO-DATA" if c6 == "NO-DATA" else ("DOSE-KEEPS-EQUILIBRIUM" if c6 in ("RECOVERED", "STABLE") else ("DOSE-KILLS" if c6 == "HORIZON" else "DOSE-PARTIAL"))
    say(f"     -> H-44: {V['H44']} | DOSE (H-43 test, beta 3e-6): {V['DOSE']}" + (f" (compression I_total S5/W6 = {comp:.1f}x)" if comp else ""))
    d5 = delta(cold("W5"), cold("S5"))
    V["P5"] = "NO-DATA" if d5 is None else ("GEN-TRANSFERS" if d5 >= 3 else ("GEN-NO-TRANSFER" if d5 <= 0 else "INDETERMINATE"))
    say(f"  P5 GEN: W5-S5 {f(d5)}pp{lab(d5)} -> {V['P5']}")
    d7 = delta(cold("W7"), cold("S5"))
    V["P6"] = "NO-DATA" if d7 is None else ("GEOMETRY-NULL" if abs(d7) <= 2 else ("GEOMETRY-MATTERS" if d7 >= 3 else ("BOX4-HURTS" if d7 <= -3 else "INDETERMINATE")))
    say(f"  P6 BOX4 control: W7-S5 {f(d7)}pp{lab(d7)} -> {V['P6']}")
    if s5 and s5.get("vote_at_k"):
        vk = s5["vote_at_k"]; v128 = vk.get("128"); v256 = vk.get("256")
        if v128 is None: V["P7"] = "NO-DATA"
        elif v128 >= .5: V["P7"] = "HOLDS (S5 vote@128 >= .50)"
        else: V["P7"] = "MISSED" + (" — SATURATED-BELOW-M1" if (v256 is not None and (v256 - v128) * 100 < 2) else " — still rising at k=256")
        h5 = s5.get("mi_hits_mean"); s7 = scan("S7", 64, 256); h7 = s7.get("mi_hits_mean") if s7 else None
        V["P7b"] = "NO-DATA" if (h5 is None or h7 is None) else ("RI+NI-BREADTH (S7 hit-rate > S5, directional)" if h7 > h5 else "NO-RI+NI-BREADTH-GAIN")
        say(f"  P7 BREADTH SCAN: S5 vote@128 {fpp(v128)} vote@256 {fpp(v256)} -> {V['P7']} | hit-rate S5 {f(h5)} vs S7 {f(h7)} -> {V['P7b']}")
    else: V["P7"] = "NO-DATA"; say("  P7 BREADTH SCAN: no S5 scan")
    say(); say("SUMMARY: " + "  ".join(f"{k}={v}" for k, v in V.items()))
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text("\n".join(LINES) + "\n"); say(f"artifact -> {OUT}")
    return V


# ---------- self-test on synthetic layouts ----------
def _mk_arm(root, arm, *, cold6=.02, cold64=.12, vote16=.35, ret=.98, val64=None, eta_=.55, itot=300000., probe_ret=None):
    tag = tag_of(arm)
    for kind, acc, vote, k in (("full_t6", cold6, cold6, 0), ("full_t64", cold64, cold64, 0), ("strat_t64", cold64, vote16, 16)):
        d = root / f"sxeval_p{tag}{arm}" / kind; d.mkdir(parents=True, exist_ok=True)
        (d / "summary_all.json").write_text(json.dumps(dict(exact_acc=acc, exact_acc_vote=vote, k_init=k, n=512, eta=eta_, init="void")))
    if ret is not None and arm.startswith("W"):
        d = root / f"sxeval_p{tag}{arm}" / "ret_t8"; d.mkdir(parents=True, exist_ok=True)
        (d / "summary_all.json").write_text(json.dumps(dict(exact_acc=ret, exact_acc_vote=ret, k_init=0, n=512, init="solution", eta=eta_)))
    if probe_ret is not None:
        d = root / f"sudprobe_p{tag}{arm}"; d.mkdir(parents=True, exist_ok=True)
        rows = [dict(task=f"x{i}", solved=False, gt_retention=(i < int(probe_ret * 100)), multi_init_hits=0, multi_init_k=16, violations=3, cells_correct=70) for i in range(100)]
        (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    if val64 is not None:
        d = root / f"sxeval_p{tag}{arm}" / "val_t64"; d.mkdir(parents=True, exist_ok=True)
        (d / "summary_all.json").write_text(json.dumps(dict(exact_acc=val64, exact_acc_vote=val64, k_init=0, n=64)))
    d = root / f"pretrain{tag}_{arm}"; d.mkdir(parents=True, exist_ok=True)
    (d / "metrics.jsonl").write_text(json.dumps(dict(step=20000, loss=.5, I_total=itot)) + "\n")

def _mk_scan(root, arm, t, k, curve, hits):
    d = root / f"sxbreadth_{arm}_t{t}_k{k}"; d.mkdir(parents=True, exist_ok=True)
    (d / "summary_all.json").write_text(json.dumps(dict(exact_acc=.12, exact_acc_vote=curve[str(k)], vote_at_k=curve, mi_hits_mean=hits, k_init=k, n=512)))

def _mk_b20(root, arm, v128, n=20000):
    d = root / f"sxbreadth20k_p{T2}{arm}"; d.mkdir(parents=True, exist_ok=True)
    (d / "summary_all.json").write_text(json.dumps(dict(exact_acc=.25, exact_acc_vote=v128, vote_at_k={"16": .4, "64": .5, "128": v128}, n=n, k_init=128)))

def selftest():
    checks = []
    def run(build):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); build(root); globals()["RUNS"] = root; globals()["OUT"] = root / "analysis" / "v.txt"
            with contextlib.redirect_stdout(io.StringIO()): return analyze()
    def base(root, **kw):
        o = dict(W1=dict(cold64=.22, val64=.3, ret=.97), W1s1=dict(cold64=.21, ret=.97), W13=dict(cold64=.26, ret=.97),
                 W2=dict(cold64=.18, ret=.95, vote16=.40), W3=dict(cold64=.19, ret=.96, vote16=.46),
                 W4=dict(cold64=.17, ret=.96), W4s1=dict(cold64=.165, ret=.96), W8=dict(cold64=.004, ret=.05, eta_=.93),
                 W9=dict(cold64=.006, ret=.03, eta_=.95), W6=dict(cold64=.11, ret=.95, itot=15000.), W5=dict(cold64=.16, ret=.97), W7=dict(cold64=.125, ret=.97))
        o.update(kw)
        _mk_arm(root, "S0", cold64=.0008, probe_ret=.01, eta_=.90, itot=1800.); _mk_arm(root, "S5", cold64=.12, probe_ret=.98, eta_=.555, itot=301680.)
        _mk_arm(root, "S4", cold64=.109, probe_ret=1.0, eta_=.65, itot=1800.); _mk_arm(root, "S7", cold64=.062, probe_ret=1.0, eta_=.94, itot=2000.)
        for a, kw_ in o.items(): _mk_arm(root, a, **kw_)
        _mk_scan(root, "S5", 64, 256, {"16": .37, "64": .46, "128": .53, "256": .58}, 2.0); _mk_scan(root, "S7", 64, 256, {"16": .10, "64": .2, "128": .3, "256": .4}, 2.5)
        _mk_b20(root, "W13", .56)
    v = run(base)
    checks += [("headline W13 below M1", v["BAND"] == "BELOW-M1" and v["BEST"].startswith("W13@t64")),
               ("M0 strict trainable", v["M0"] == "STRICT-TRAINABLE"),
               ("breadth M1 on 20k", v["BREADTH-BAND"] == "M1-ON-BREADTH" and v["BREADTH"].startswith("W13")),
               ("P1 budget pays", v["P1"] == "BUDGET-PAYS"), ("P2 depth pays", v["P2"] == "DEPTH-PAYS"), ("P2b RI+NI breadth", v["P2b"] == "RI+NI-RAISES-BREADTH"),
               ("P3 width pays", v["P3"] == "WIDTH-PAYS"), ("H44 falsified at scale", v["H44"].startswith("FALSIFIED")),
               ("dose keeps eq", v["DOSE"] == "DOSE-KEEPS-EQUILIBRIUM"), ("P5 gen transfers", v["P5"] == "GEN-TRANSFERS"),
               ("P6 geometry null", v["P6"] == "GEOMETRY-NULL"), ("P7 holds", v["P7"].startswith("HOLDS")), ("P7b RI+NI", v["P7b"].startswith("RI+NI"))]
    v = run(lambda r: base(r, W8=dict(cold64=.08, ret=.95), W1=dict(cold64=.13, val64=.1, ret=.97), W7=dict(cold64=.17, ret=.97), W5=dict(cold64=.11, ret=.9), W6=dict(cold64=.001, ret=.02)))
    checks += [("H44 supported via W8", v["H44"].startswith("SUPPORTED")), ("M0 gen primary", v["M0"] == "GEN-PRIMARY"),
               ("P1 not binding", v["P1"] == "BUDGET-NOT-BINDING"), ("P6 geometry matters", v["P6"] == "GEOMETRY-MATTERS"),
               ("P5 no transfer", v["P5"] == "GEN-NO-TRANSFER"), ("dose kills", v["DOSE"] == "DOSE-KILLS")]
    def no_b20(root):
        base(root)
        for d in root.glob(f"sxbreadth20k_p{T2}*"):
            for f_ in d.iterdir(): f_.unlink()
            d.rmdir()
    v = run(no_b20)
    checks.append(("breadth falls back to S5 scan", v["BREADTH"].startswith("S5 strat") and v["BREADTH-BAND"] == "M1-ON-BREADTH"))
    v = run(lambda r: base(r, W1s1=dict(cold64=.16, ret=.97)))   # big seed spread -> NOISE 6pp
    checks.append(("seed noise computed", "NOISE=6.00pp" in v["SEED"]))
    n = sum(1 for _, o in checks if o)
    for name, o in checks: print(f"  {'PASS' if o else 'FAIL'}  {name}")
    print(f"selftest: {n}/{len(checks)}"); return n == len(checks)

if __name__ == "__main__":
    sys.exit(0 if selftest() else 1) if "--selftest" in sys.argv else analyze()
