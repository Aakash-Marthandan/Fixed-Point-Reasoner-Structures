# Ledger: SPRINT S2 WAVE 3a ANALYZER — written BEFORE any wave-3a run (2026-08-23);
# the launch entry locks these rules verbatim. Reads the wave-2 reference cells
# (tag sport2w2: W2 plain T12 @30k = 19.37 %, W8 priced d32 @20k, W9 priced d16
# @50k; S5 plain T6 @20k from tag sport2) and the wave-3a arms (tag sport3a) +
# the S5 breadth confirmation + the trajectory monitors.
#   BREADTH-CONFIRM  S5 20k-subsample vote@128 @t64 (the EqR-comparable protocol:
#          D=64, B=128) >= .50 -> M1-ON-BREADTH (full-test-grade); report vote@256
#          (20k) and the strat-512 k=1024 saturation.
#   COLLAPSE CONTROL  A7 (plain T12 @50k, W2's recipe continued): COLLAPSED if
#          final-map retention (retfm_t8) < .5 OR cold@t64(final) < W2 − 5 pp; else
#          NO-COLLAPSE (the law does not bite at T12/50k). The monitor trajectory
#          (val@t64, retfm, lam_max every 5k) gives the peak/collapse step.
#   FIX ARMS  A2 (RI .5 + NI .01), A3 (FPA k4), A4 (eq_coupled): each is
#          FIX-HOLDS if retfm_t8 >= .9 at 50k AND cold@t64(final) >= W2 − 2 pp;
#          FIX-IMPROVES additionally if cold(final) >= W2 + 3 pp; FIX-FAILS if
#          retfm < .5 or cold < W2 − 5 pp. (Read against A7's outcome: if A7 does
#          not collapse, "HOLDS" is uninformative and the rule says so.)
#   PRICE CEILING  A5 (priced T12 @50k) vs W9/W2; A6 (priced d32 @50k) vs W8:
#          PRICE-RISES if A6 − W8 >= +3 pp; PRICED-DEPTH-PAYS if A5 − W9 >= +3 pp;
#          class STABLE/RECOVERED/HORIZON as in wave 2. REGULARISER-OF-CHOICE =
#          argmax cold(final) over {best fix arm, best priced arm}, reported with
#          the margin and the seed spread.
#   CONVENTIONAL CONTROL  A8 (aug 500 + wd 1e-3 plain T6 @50k): WORKS if retfm
#          >= .9 AND cold >= S5 + 3 pp.
#   SEED  |A7 − A7s1| cold@t64(final) (and val-selected); NOISE = max(that, 1 pp);
#          contrasts below 2*NOISE are labeled WITHIN-SEED-NOISE.
#   HEADLINE  best cold among FINAL ckpts (the registered protocol) -> bands; the
#          VAL-SELECTED cold numbers are reported alongside, labeled (selection on
#          the train-file monitor rows, test-disjoint). BREADTH (labeled): PHASE4
#          best-arm 20k vote@128 and S5's confirmation.
#   TRAJECTORY LAW  across arms and steps: does lam_max cross 1 where retfm falls
#          and cold collapses? Report Spearman(lam_max, retfm) over all monitor rows
#          and the per-arm first step with lam_frac_expansive > 0.
#   RECIPE DECISION for the ladder (mechanical): among non-COLLAPSED arms, the
#          best val-selected cold; tie-break by strat vote@16. Ties within noise
#          -> prefer the simpler recipe order A3 > A4 > A2 > A5.
"""
  .venv/bin/python tools/analyze_sport3a.py            # -> runs/analysis/sport3a_verdict.txt
  .venv/bin/python tools/analyze_sport3a.py --selftest
"""
from __future__ import annotations
import contextlib, io, json, math, os, sys, tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(os.environ.get("QHRRN_RUNS", ROOT / "runs"))
OUT = RUNS / "analysis" / "sport3a_verdict.txt"
T1, T2, T3 = "sport2", "sport2w2", "sport3a"
REF = {"S5": T1, "W2": T2, "W8": T2, "W9": T2}
ARMS = ["A2", "A3", "A4", "A5", "A6", "A7", "A7s1", "A8", "A2s1", "A3s1", "A4s1", "A5s1", "A6s1", "A9", "A9s1", "A10"]
FIX = ["A2", "A3", "A4", "A9"]
SEED_PAIRS = [("A7", "A7s1"), ("A2", "A2s1"), ("A3", "A3s1"), ("A4", "A4s1"), ("A5", "A5s1"), ("A6", "A6s1"), ("A9", "A9s1")]
DESC = {"S5": "plain T6 d16 20k (w1)", "W2": "plain T12 d16 30k (w2)", "W8": "priced d32 T6 20k (w2)", "W9": "priced d16 T6 50k (w2)",
        "A2": "plain T12 RI.5 NI.01 50k", "A3": "plain T12 FPA k4 50k", "A4": "plain T12 eq_coupled 50k", "A5": "priced T12 50k",
        "A6": "priced d32 T6 50k", "A7": "plain T12 50k (W2 cont.)", "A7s1": "A7 seed 1", "A8": "plain T6 aug500 wd1e-3 50k",
        "A2s1": "A2 seed 1", "A3s1": "A3 seed 1", "A4s1": "A4 seed 1", "A5s1": "A5 seed 1", "A6s1": "A6 seed 1",
        "A9": "plain T12 RI.5 NI.01 + FPA k4 50k", "A9s1": "A9 seed 1", "A10": "priced T12 RI.5 NI.01 50k"}
LINES = []
def say(s=""): LINES.append(str(s)); print(s)
def jload(p):
    p = Path(p); return json.loads(p.read_text()) if p.exists() else None
def tag_of(a): return REF.get(a, T3)
def summ(a, kind): return jload(RUNS / f"sxeval_p{tag_of(a)}{a}" / kind / "summary_all.json")
def cold(a, t=64, vb=False):
    s = summ(a, "full_t64_valbest" if vb else f"full_t{t}"); return None if not s else s["exact_acc"]
def vote16(a):
    s = summ(a, "strat_t64"); return None if not s else s.get("exact_acc_vote")
def retfm(a):
    s = summ(a, "retfm_t8"); return None if not s else s["exact_acc"]
def ret(a):
    s = summ(a, "ret_t8"); return None if not s else s["exact_acc"]
def val64(a):
    s = summ(a, "val_t64"); return None if not s else s["exact_acc"]
def eta(a):
    for k in ("full_t64", "strat_t64"):
        s = summ(a, k)
        if s and s.get("eta") is not None: return s["eta"]
    return None
def monitors(a):
    p = RUNS / f"pretrain{tag_of(a)}_{a}" / "metrics.jsonl"
    if not p.exists(): return []
    out = []
    for l in p.read_text().splitlines():
        try: r = json.loads(l)
        except Exception: continue
        if "monitor" in r: out.append(r["monitor"])
    return sorted(out, key=lambda m: m["step"])
def valbest(a):
    p = RUNS / f"pretrain{tag_of(a)}_{a}" / "val_best.txt"
    if not p.exists(): return None
    parts = p.read_text().split()
    return (int(parts[2]), float(parts[1])) if len(parts) >= 3 else None
def f(x, nd=2):  return "  -  " if x is None else f"{x:.{nd}f}"
def fpp(x):     return "  -  " if x is None else f"{100*x:6.2f}"
def delta(a, b): return None if (a is None or b is None) else 100.0 * (a - b)
def ci95(p, n): return None if (p is None or not n) else 1.96 * math.sqrt(max(p * (1 - p), 1e-12) / n)
def cls(a):
    c, r = cold(a), retfm(a)
    if r is None: r = ret(a)
    if c is None or r is None: return "NO-DATA"
    if r >= .9 and c >= .05: return "RECOVERED"
    if r >= .9: return "STABLE"
    if r < .5 and c < .02: return "HORIZON"
    return "PARTIAL"

def analyze():
    LINES.clear(); V = {}
    say("=" * 112); say("SPRINT S2 WAVE 3a VERDICT — Sudoku-Extreme (rules registered 2026-08-23; full 423k test exact; breadth labeled separately)"); say("=" * 112)
    say("\nSECTION 1 — per-arm table: cold full-test exact @t6/@t64 (FINAL ckpt) | val-selected @t64 [step] | strat vote@16 | ret_t8 sched | retfm_t8 final-map | val@t64 | eta | class")
    for a in ["S5", "W2", "W8", "W9"] + ARMS:
        vb = valbest(a); vbs = f"{fpp(cold(a, vb=True))} [{vb[0]}]" if (vb and cold(a, vb=True) is not None) else ("  (final=best) " if vb else "   -   ")
        say(f"  {a:5s} {DESC[a]:30s} t6 {fpp(cold(a,6))} t64 {fpp(cold(a))} | vb {vbs} | v16 {fpp(vote16(a))} | ret {f(ret(a))} retfm {f(retfm(a))} | val {fpp(val64(a))} | eta {f(eta(a),3)} | {cls(a)}")
    # seed noise
    spreads = {f"{a}-{b}": delta(cold(a), cold(b)) for a, b in SEED_PAIRS}
    dS = spreads["A7-A7s1"]
    vals = [abs(v) for v in spreads.values() if v is not None]
    NOISE = max(max(vals) if vals else 0.0, 1.0)
    V["SEED"] = " ".join(f"|{k}|={f(v)}" for k, v in spreads.items() if v is not None) + f" NOISE={NOISE:.2f}pp"
    say(f"\nSEED NOISE (cold@t64 final, per pair): " + ", ".join(f"{k} {f(v)}pp" for k, v in spreads.items()) + f" -> NOISE {NOISE:.2f} pp (contrasts below {2*NOISE:.2f} pp labeled WITHIN-SEED-NOISE)")
    lab = lambda d: "" if d is None or abs(d) >= 2 * NOISE else " [WITHIN-SEED-NOISE]"
    # breadth confirmation
    b128 = jload(RUNS / "sxbreadth20000_S5_k128" / "summary_all.json"); b256 = jload(RUNS / "sxbreadth20000_S5_k256" / "summary_all.json"); b1k = jload(RUNS / "sxbreadth_S5_t64_k1024" / "summary_all.json")
    say("\nBREADTH CONFIRMATION (S5, the wave-1 plain T6 map; EqR-comparable protocol D=64, B=128):")
    if b128:
        v = b128.get("vote_at_k", {}).get("128", b128.get("exact_acc_vote")); V["BREADTH-CONFIRM"] = "M1-ON-BREADTH (20k, full-test-grade)" if v >= .5 else "BELOW-M1-ON-BREADTH"
        say(f"  20k vote@128 {fpp(v)} +-{f(100*ci95(v,b128['n']) if ci95(v,b128['n']) else None)}pp cold {fpp(b128['exact_acc'])} | curve {b128.get('vote_at_k')} -> {V['BREADTH-CONFIRM']}")
    else: V["BREADTH-CONFIRM"] = "NO-DATA"; say("  20k k128: no data")
    if b256: say(f"  20k vote@256 {fpp(b256.get('vote_at_k',{}).get('256', b256.get('exact_acc_vote')))} | curve {b256.get('vote_at_k')}")
    if b1k:
        vk = b1k.get("vote_at_k", {}); say(f"  strat-512 k=1024 curve {vk} | saturation: vote@1024-vote@512 = {f(100*(vk.get('1024',0)-vk.get('512',0)))}pp")
    # collapse control
    c7, r7 = cold("A7"), retfm("A7"); w2 = cold("W2")
    if c7 is None or r7 is None or w2 is None: V["COLLAPSE"] = "NO-DATA"
    else: V["COLLAPSE"] = "COLLAPSED" if (r7 < .5 or 100*(c7 - w2) < -5) else "NO-COLLAPSE"
    mons7 = monitors("A7"); traj7 = " ".join(f"{m['step']//1000}k:{100*m['val_t64']:.0f}/{m['ret_final_t8']:.2f}/{m.get('lam_joint_mean', m['lam_max_mean']):.2f}" for m in mons7)
    say(f"\nCOLLAPSE CONTROL (A7 = W2's recipe to 50k): cold {fpp(c7)} (W2 {fpp(w2)}) retfm {f(r7)} -> {V['COLLAPSE']} | trajectory (step: val%/retfm/lam): {traj7 or '-'}")
    # fix arms
    say("\nFIX ARMS (vs W2 19.37 % and A7's outcome):")
    for a in FIX:
        c, r = cold(a), retfm(a); d = delta(c, w2)
        if c is None or r is None: V[a] = "NO-DATA"
        elif r < .5 or (d is not None and d < -5): V[a] = "FIX-FAILS"
        elif r >= .9 and d >= -2: V[a] = "FIX-IMPROVES" if d >= 3 else "FIX-HOLDS"
        else: V[a] = "FIX-PARTIAL"
        note = " (uninformative: A7 did not collapse)" if (V["COLLAPSE"] == "NO-COLLAPSE" and V[a] == "FIX-HOLDS") else ""
        mons = monitors(a); first_exp = next((m["step"] for m in mons if m.get("lam_joint_frac_expansive", m.get("lam_frac_expansive", 0)) > 0.5), None)
        say(f"  {a} {DESC[a]:28s} cold {fpp(c)} ({f(d)}pp vs W2{lab(d)}) retfm {f(r)} v16 {fpp(vote16(a))} -> {V[a]}{note} | first expansive-lam step: {first_exp}")
    # price ceiling
    c5, c6 = cold("A5"), cold("A6"); d6 = delta(c6, cold("W8")); d5 = delta(c5, cold("W9"))
    V["PRICE-RISES"] = "NO-DATA" if d6 is None else ("PRICE-RISES" if d6 >= 3 else "PRICE-FLAT")
    V["PRICED-DEPTH"] = "NO-DATA" if d5 is None else ("PRICED-DEPTH-PAYS" if d5 >= 3 else "PRICED-DEPTH-FLAT")
    say(f"\nPRICE CEILING: A6 priced d32 @50k {fpp(c6)} vs W8 @20k {fpp(cold('W8'))} ({f(d6)}pp{lab(d6)}) -> {V['PRICE-RISES']} [{cls('A6')}] | A5 priced T12 @50k {fpp(c5)} vs W9 {fpp(cold('W9'))} ({f(d5)}pp{lab(d5)}) -> {V['PRICED-DEPTH']} [{cls('A5')}]")
    fixbest = max([a for a in FIX if cold(a) is not None], key=lambda a: cold(a), default=None)
    prbest = max([a for a in ("A5", "A6", "A10", "A5s1", "A6s1") if cold(a) is not None], key=lambda a: cold(a), default=None)
    if fixbest and prbest:
        dm = delta(cold(fixbest), cold(prbest)); V["REGULARISER"] = f"{fixbest if dm >= 0 else prbest} (margin {f(abs(dm))}pp{lab(dm)})"
        say(f"  REGULARISER-OF-CHOICE: best fix {fixbest} {fpp(cold(fixbest))} vs best priced {prbest} {fpp(cold(prbest))} -> {V['REGULARISER']}")
    else: V["REGULARISER"] = "NO-DATA"
    # conventional control
    c8, r8 = cold("A8"), retfm("A8"); d8 = delta(c8, cold("S5"))
    V["CONVENTIONAL"] = "NO-DATA" if (c8 is None or r8 is None) else ("CONVENTIONAL-WORKS" if (r8 >= .9 and d8 >= 3) else "CONVENTIONAL-FAILS")
    say(f"\nCONVENTIONAL CONTROL: A8 cold {fpp(c8)} ({f(d8)}pp vs S5{lab(d8)}) retfm {f(r8)} -> {V['CONVENTIONAL']}")
    # headline + breadth
    H = {a: cold(a) for a in ARMS + ["W2"] if cold(a) is not None}
    if H:
        best = max(H, key=H.get); acc = H[best]; band = "M3" if acc >= .95 else "M2" if acc >= .85 else "M1" if acc >= .5 else "BELOW-M1"
        V["BAND"] = band; V["BEST"] = f"{best}={acc:.4f}"
        Hv = {a: cold(a, vb=True) for a in ARMS if cold(a, vb=True) is not None}
        vbbest = (max(Hv, key=Hv.get), max(Hv.values())) if Hv else None
        say(f"\nHEADLINE (cold, FINAL ckpts): {best} {acc:.4f} -> BAND {band}" + (f" | val-selected best: {vbbest[0]} {vbbest[1]:.4f} (labeled)" if vbbest else ""))
    else: V["BAND"] = "NO-DATA"
    b20 = {d.name.replace(f"sxbreadth20k_p{T3}", ""): jload(d / "summary_all.json") for d in RUNS.glob(f"sxbreadth20k_p{T3}*")}
    b20 = {k: v for k, v in b20.items() if v}
    if b20:
        k, s = max(b20.items(), key=lambda kv: kv[1].get("vote_at_k", {}).get("128", 0)); v = s.get("vote_at_k", {}).get("128", s.get("exact_acc_vote"))
        V["BREADTH"] = f"{k} vote@128={v:.4f}"; V["BREADTH-BAND"] = "M1-ON-BREADTH" if v >= .5 else "BELOW-M1-ON-BREADTH"
        say(f"BREADTH (labeled): PHASE4 {k} 20k vote@128 {fpp(v)} cold {fpp(s['exact_acc'])} -> {V['BREADTH-BAND']}")
    else: V["BREADTH"] = "NO-DATA"
    # trajectory law
    rows = [m for a in ARMS for m in monitors(a)]
    if len(rows) >= 6:
        # joint (y,z) lambda_max is the validated contractivity readout (2026-08-23 banked-ckpt check:
        # healthy .73-.85, collapsed 2.3 with 75-88% expansive); y-only lambda kept as fallback
        lam = np.array([m.get("lam_joint_mean", m["lam_max_mean"]) for m in rows]); rf = np.array([m["ret_final_t8"] for m in rows])
        rl = np.argsort(np.argsort(lam)); rr = np.argsort(np.argsort(rf)); rho = float(np.corrcoef(rl, rr)[0, 1])
        V["TRAJ"] = f"spearman(lam_joint,retfm)={rho:.2f} n={len(rows)}"
        say(f"\nTRAJECTORY LAW: Spearman(joint lambda_max, final-map retention) over {len(rows)} monitor rows = {rho:.2f} (H-45 predicts strongly negative; lam>1 <-> retfm->0)")
    else: V["TRAJ"] = "NO-DATA"
    # recipe decision
    cands = [a for a in ARMS if not a.endswith("s1") and cold(a) is not None and not (retfm(a) is not None and retfm(a) < .5)]
    if cands:
        def key(a):
            vb = cold(a, vb=True); return (vb if vb is not None else cold(a), vote16(a) or 0)
        rec = max(cands, key=key); V["RECIPE"] = rec
        say(f"\nRECIPE DECISION for the ladder: {rec} ({DESC[rec]}) — best val-selected cold among non-collapsed arms (tie-break strat vote@16)")
    else: V["RECIPE"] = "NO-DATA"
    say(); say("SUMMARY: " + "  ".join(f"{k}={v}" for k, v in V.items()))
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text("\n".join(LINES) + "\n"); say(f"artifact -> {OUT}")
    return V


# ---------- self-test ----------
def _arm(root, a, *, cold64, cold6=.02, v16=.2, ret=1.0, retfm=1.0, val=.1, eta_=.6, vb=None, mon=None):
    tag = tag_of(a)
    for kind, acc, vote, k in (("full_t6", cold6, cold6, 0), ("full_t64", cold64, cold64, 0), ("strat_t64", cold64, v16, 16)):
        d = root / f"sxeval_p{tag}{a}" / kind; d.mkdir(parents=True, exist_ok=True)
        (d / "summary_all.json").write_text(json.dumps(dict(exact_acc=acc, exact_acc_vote=vote, k_init=k, n=512, eta=eta_)))
    for kind, acc in (("ret_t8", ret), ("retfm_t8", retfm), ("val_t64", val)):
        d = root / f"sxeval_p{tag}{a}" / kind; d.mkdir(parents=True, exist_ok=True)
        (d / "summary_all.json").write_text(json.dumps(dict(exact_acc=acc, exact_acc_vote=acc, k_init=0, n=64)))
    if vb is not None:
        d = root / f"sxeval_p{tag}{a}" / "full_t64_valbest"; d.mkdir(parents=True, exist_ok=True)
        (d / "summary_all.json").write_text(json.dumps(dict(exact_acc=vb[1], exact_acc_vote=vb[1], k_init=0, n=422786)))
        pd = root / f"pretrain{tag}_{a}"; pd.mkdir(parents=True, exist_ok=True); (pd / "val_best.txt").write_text(f"{vb[0]:06d} 0.1 {vb[0]}\n")
    if mon:
        pd = root / f"pretrain{tag}_{a}"; pd.mkdir(parents=True, exist_ok=True)
        (pd / "metrics.jsonl").write_text("".join(json.dumps({"monitor": dict(step=s, val_t64=v, ret_sched_t8=1.0, ret_final_t8=r, lam_max_mean=l, lam_max_max=l, lam_frac_expansive=float(l > 1), eta=.6)}) + "\n" for s, v, r, l in mon))

def _scan(root, name, curve, n):
    d = root / name; d.mkdir(parents=True, exist_ok=True)
    (d / "summary_all.json").write_text(json.dumps(dict(exact_acc=.16, exact_acc_vote=curve[max(curve, key=int)], vote_at_k=curve, n=n, k_init=int(max(curve, key=int)))))

def selftest():
    checks = []
    def run(build):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); build(root); globals()["RUNS"] = root; globals()["OUT"] = root / "a" / "v.txt"
            with contextlib.redirect_stdout(io.StringIO()): return analyze()
    def base(root, **kw):
        _arm(root, "S5", cold64=.1202, v16=.369, ret=.98, retfm=.98, eta_=.555); _arm(root, "W2", cold64=.1937, v16=.232, eta_=.607)
        _arm(root, "W8", cold64=.0935, ret=1.0, retfm=1.0, eta_=.919); _arm(root, "W9", cold64=.0524, ret=1.0, retfm=1.0, eta_=.924)
        o = dict(A7=dict(cold64=.05, retfm=.2, vb=(30000, .19), mon=[(10000,.1,1.0,.8),(30000,.19,.95,.95),(50000,.05,.2,1.3)]),
                 A7s1=dict(cold64=.06, retfm=.3), A2=dict(cold64=.21, retfm=.95, v16=.35, vb=(50000,.21), mon=[(50000,.2,.95,.9)]),
                 A3=dict(cold64=.23, retfm=.99, v16=.25, mon=[(50000,.22,.99,.85)]), A4=dict(cold64=.18, retfm=.92, mon=[(50000,.17,.92,.9)]),
                 A5=dict(cold64=.12, retfm=1.0, eta_=.92), A6=dict(cold64=.14, retfm=1.0, eta_=.92), A8=dict(cold64=.16, retfm=.95))
        o.update(kw)
        for a, k in o.items(): _arm(root, a, **k)
        _scan(root, "sxbreadth20000_S5_k128", {"16": .36, "64": .56, "128": .69}, 20000); _scan(root, "sxbreadth20000_S5_k256", {"128": .69, "256": .79}, 20000)
        _scan(root, "sxbreadth_S5_t64_k1024", {"256": .80, "512": .86, "1024": .89}, 512)
        _scan(root, f"sxbreadth20k_p{T3}A3", {"16": .3, "128": .55}, 20000)
    v = run(base)
    checks += [("breadth confirm M1", v["BREADTH-CONFIRM"].startswith("M1-ON-BREADTH")), ("collapse control collapsed", v["COLLAPSE"] == "COLLAPSED"),
               ("A3 improves", v["A3"] == "FIX-IMPROVES"), ("A2 holds (21 vs 19.4 +1.6)", v["A2"] == "FIX-HOLDS"), ("A4 holds", v["A4"] == "FIX-HOLDS"),
               ("price rises A6", v["PRICE-RISES"] == "PRICE-RISES"), ("priced depth pays A5", v["PRICED-DEPTH"] == "PRICED-DEPTH-PAYS"),
               ("regulariser = fix A3", v["REGULARISER"].startswith("A3")), ("conventional works", v["CONVENTIONAL"] == "CONVENTIONAL-WORKS"),
               ("headline below-M1 best A3", v["BAND"] == "BELOW-M1" and v["BEST"].startswith("A3")), ("breadth band M1 (A3 20k .55)", v["BREADTH-BAND"] == "M1-ON-BREADTH"),
               ("recipe A3", v["RECIPE"] == "A3"), ("traj law computed", v["TRAJ"].startswith("spearman"))]
    v = run(lambda r: base(r, A7=dict(cold64=.20, retfm=.98, vb=(50000,.2)), A3=dict(cold64=.20, retfm=.98), A6=dict(cold64=.10, retfm=1.0), A8=dict(cold64=.08, retfm=.5)))
    checks += [("no collapse", v["COLLAPSE"] == "NO-COLLAPSE"), ("A3 holds uninformative", v["A3"] == "FIX-HOLDS"),
               ("price flat", v["PRICE-RISES"] == "PRICE-FLAT"), ("conventional fails", v["CONVENTIONAL"] == "CONVENTIONAL-FAILS")]
    v = run(lambda r: base(r, A2=dict(cold64=.01, retfm=.1)))
    checks.append(("A2 fails", v["A2"] == "FIX-FAILS"))
    n = sum(1 for _, ok in checks if ok)
    for name, ok in checks: print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"selftest: {n}/{len(checks)}"); return n == len(checks)

if __name__ == "__main__":
    sys.exit(0 if selftest() else 1) if "--selftest" in sys.argv else analyze()
