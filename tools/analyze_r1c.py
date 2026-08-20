# Ledger: RUNG-1c ANALYZER (written 2026-08-20 BEFORE the campaign's data exist —
# pre-registration in code). The WIDTH x BUDGET completion; rules from the
# 2026-08-20 RUNG-1c LAUNCH REGISTRATION exactly:
#   R1c-1 (H-40, the A4-class 2x2): corners a=(d48,40k) n=3 anchor / b=(d48,53k)
#          n=3 (A8) / c=(d64,40k) n=2 (A10) / d=(d64,53k) n=3 (rung-1 A4).
#          On rg-96 (primary; N secondary): BUDGET-DOMINANT if Δbudget|48>=5 AND
#          Δwidth|40k<=3; WIDTH-DOMINANT if Δwidth|40k>=5 AND Δbudget|48<=3;
#          BOTH if each >=4; else INDETERMINATE. |interaction|>=5 reported.
#   R1c-2 (A5 anchors for d96): A11 (d48@40k A5) vs anchor: |Δ|<=3 -> EQUIV
#          (A11 = the d96 baseline); A12 (d64@40k A5) vs A11: >=-3 -> width tax
#          ABSENT at 40k on A5 (d96-GO prior); <=-5 -> width tax PRESENT.
#   R1c-3 (H-41 re-pricing rescue): A13 (d48@53k beta 6e-5) vs A8 (3e-5, n=3):
#          A13>=A8+5 AND A13>=anchor-3 -> RESCUE-FULL (beta scales with budget);
#          A13>=A8+5 else -> RESCUE-PARTIAL; A13<=A8+2 -> RESCUE-FAILS; else IND.
#   R1c-4 (e1e3 mechanism, directional): wrong-stable + nd1 at corners b,c vs
#          the d48@40k (rung-0) and d64@53k (rung-1) e1e3 anchors — reported.
# ADMISSION at artifact level per arm. Loaders mirror analyze_r1b.
"""
  .venv/bin/python tools/analyze_r1c.py            # -> runs/analysis/r1c_verdict.txt
  .venv/bin/python tools/analyze_r1c.py --selftest
"""
from __future__ import annotations
import json, math, os, pickle, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(os.environ.get("QHRRN_RUNS", ROOT / "runs"))
OUT = RUNS / "analysis" / "r1c_verdict.txt"
EPS = ["0.05", "0.1", "0.2", "0.4"]
REG = {"A8": dict(d=48, steps=53333, beta=3e-5, nl=1e-5, floors="350,75,50,15,30", ni=0.01),
       "A10": dict(d=64, steps=40000, beta=3e-5, nl=1e-5, floors="350,75,50,15,30", ni=0.01),
       "A11": dict(d=48, steps=40000, beta=3e-5, nl=1e-5, floors="", ni=0.01),
       "A12": dict(d=64, steps=40000, beta=3e-5, nl=1e-5, floors="", ni=0.01),
       "A13": dict(d=48, steps=53333, beta=6e-5, nl=2e-5, floors="", ni=0.01),
       "A4d48": dict(d=48, steps=40000, beta=3e-5, nl=1e-5, floors="350,75,50,15,30", ni=0.01),
       "A4d64": dict(d=64, steps=53333, beta=3e-5, nl=1e-5, floors="350,75,50,15,30", ni=0.01),
       "A5d64": dict(d=64, steps=53333, beta=3e-5, nl=1e-5, floors="", ni=0.01)}
LINES = []
def say(s=""): LINES.append(s); print(s)

def rg96(rt, cell):
    tot = 0; n = 0
    for pref in ("ladrg", "ladrgb"):
        p = RUNS / f"{pref}_{rt}{cell}" / "results.jsonl"
        if not p.exists(): continue
        for line in p.read_text().splitlines():
            if not line.strip(): continue
            r = json.loads(line)
            for q in r["queries"]: tot += q["gt_retention"]; n += 1
    return tot if n else None

def vhN(rt, cell):
    p = RUNS / f"lad_{rt}{cell}" / "results.jsonl"
    if not p.exists(): return None, None
    qs = [q for l in p.read_text().splitlines() if l.strip() for q in json.loads(l)["queries"]]
    return sum(q["gt_retention"] for q in qs), float(np.median([sum(q["I_s"]) for q in qs]))

def sc(pdir):
    p = RUNS / pdir / "ckpt_latest.pkl"
    if not p.exists(): return None
    ck = pickle.loads(p.read_bytes())
    return dict(eta=float(1/(1+np.exp(-float(np.asarray(ck["state"]["model"]["eq"]["eta"]))))), step=int(ck["step"]), cfg=ck.get("config", {}) or {})

def admit(x, arm):
    if not x: return False, "no ckpt"
    r = REG[arm]; c = x["cfg"]; why = []
    if int(c.get("d", -1)) != r["d"]: why.append(f"d={c.get('d')}")
    if x["step"] != r["steps"]: why.append(f"step={x['step']}")
    for k, kk in (("beta", "beta_flux"), ("nl", "beta_flux_nl")):
        if abs(float(c.get(kk, -1)) - r[k]) > 1e-12: why.append(f"{kk}={c.get(kk)}")
    if str(c.get("flux_floors", "") or "") != r["floors"]: why.append(f"floors={c.get('flux_floors')!r}")
    if abs(float(c.get("ni_sigma", -1)) - r["ni"]) > 1e-12: why.append(f"ni={c.get('ni_sigma')}")
    return (not why), (", ".join(why) or "ok")

def e13(rt, cell):
    p = RUNS / f"e1e3_{rt}{cell}" / "results.jsonl"
    if not p.exists(): return None
    ws = conv = tot = nd1 = 0
    for l in p.read_text().splitlines():
        r = json.loads(l)
        for e3 in r["e3"]:
            tot += 1
            if e3["converged_at"] is not None: conv += 1; ws += (not e3["limit_exact"])
            nd1 += (e3["n_distinct"] == 1)
    return dict(ws=ws, conv=conv, tot=tot, nd1=nd1)

CELLS = {  # label -> (battery tag, pretrain dir, arm-id for admission)
    "a=d48@40k": [(f"p13fA4s{i}", f"pretrain13f_A4s{i}", "A4d48") for i in (0, 1, 2)],
    "b=d48@53k": [("pr1bA8s0", "pretrainr1b_A8s0", "A8"), ("pr1cA8s1", "pretrainr1c_A8s1", "A8"), ("pr1cA8s2", "pretrainr1c_A8s2", "A8")],
    "c=d64@40k": [("pr1cA10s0", "pretrainr1c_A10s0", "A10"), ("pr1cA10s1", "pretrainr1c_A10s1", "A10")],
    "d=d64@53k": [(f"pr1A4s{i}", f"pretrainr1_A4s{i}", "A4d64") for i in (0, 1, 2)],
    "A11 d48@40k A5": [("pr1cA11s0", "pretrainr1c_A11s0", "A11"), ("pr1cA11s1", "pretrainr1c_A11s1", "A11")],
    "A12 d64@40k A5": [("pr1cA12s0", "pretrainr1c_A12s0", "A12"), ("pr1cA12s1", "pretrainr1c_A12s1", "A12")],
    "A13 d48@53k b6e-5": [("pr1cA13s0", "pretrainr1c_A13s0", "A13"), ("pr1cA13s1", "pretrainr1c_A13s1", "A13")],
    "A5 d64@53k (r1)": [("pr1A5s0", "pretrainr1_A5s0", "A5d64"), ("pr1bA5s1", "pretrainr1b_A5s1", "A5d64"), ("pr1bA5s2", "pretrainr1b_A5s2", "A5d64")],
}

def analyze():
    LINES.clear()
    say("=" * 100); say("RUNG-1c VERDICT — the WIDTH x BUDGET completion (H-40 2x2, A5 anchors, H-41 re-pricing rescue)"); say("rules R1c-1..R1c-4 as registered 2026-08-20 (analyzer written pre-data)"); say("=" * 100)
    G = {}
    say(); say("SECTION 1 — admission + the table (rg-96/288, vh N/144, I_med, eta)")
    for grp, cells in CELLS.items():
        vals = []
        for tag, pdir, arm in cells:
            x = sc(pdir); ok, why = admit(x, arm)
            g = rg96("", tag); N, I = vhN("", tag)
            stat = "ADMITTED" if ok else f"EXCLUDED({why})"
            say(f'  {grp:18s} {tag:10s} {stat:22s} rg96 {g if g is not None else "-":>3}  N {N if N is not None else "-":>3}  I {I if I else 0:>6.0f}  eta {x["eta"] if x else 0:.3f}')
            if ok and g is not None and N is not None: vals.append((g, N, I, x["eta"]))
        G[grp] = vals
    def m(grp, i): v = [x[i] for x in G[grp]]; return (float(np.mean(v)) if v else float("nan")), v
    verdicts = {}

    say(); say("R1c-1 (H-40) — the A4-class 2x2 on rg-96 (primary) and N (secondary)")
    (a, av), (b, bv), (c, cv), (d, dv) = m("a=d48@40k", 0), m("b=d48@53k", 0), m("c=d64@40k", 0), m("d=d64@53k", 0)
    if any(math.isnan(x) for x in (a, b, c, d)) or len(bv) < 3 or len(cv) < 2:
        say("  (corners incomplete — no verdict)"); verdicts["R1c-1"] = "NO-DATA"
    else:
        dB48, dW40, dW53, dB64 = a - b, a - c, b - d, c - d
        inter = dW53 - dW40
        say(f'  corners rg96: a {av}={a:.1f}  b {bv}={b:.1f}  c {cv}={c:.1f}  d {dv}={d:.1f}')
        say(f'  Δbudget|d48 {dB48:+.1f}   Δwidth|40k {dW40:+.1f}   Δwidth|53k {dW53:+.1f}   Δbudget|d64 {dB64:+.1f}   interaction {inter:+.1f}')
        aN, bN, cN, dN = m("a=d48@40k", 1)[0], m("b=d48@53k", 1)[0], m("c=d64@40k", 1)[0], m("d=d64@53k", 1)[0]
        say(f'  corners N: a {aN:.1f}  b {bN:.1f}  c {cN:.1f}  d {dN:.1f}   (Δbudget|d48 {aN-bN:+.1f}, Δwidth|40k {aN-cN:+.1f})')
        if dB48 >= 5 and dW40 <= 3: verdicts["R1c-1"] = "BUDGET-DOMINANT"; say("  VERDICT: BUDGET-DOMINANT — H-40 CONFIRMED; the 'width ceiling' was an optimization-length tax at fixed beta; d96 runs at the 40k budget; any steps-law re-prices beta")
        elif dW40 >= 5 and dB48 <= 3: verdicts["R1c-1"] = "WIDTH-DOMINANT"; say("  VERDICT: WIDTH-DOMINANT — the A8s0 cell was noise; H-36 stands as width; depth-lean")
        elif dB48 >= 4 and dW40 >= 4: verdicts["R1c-1"] = "BOTH"; say("  VERDICT: BOTH taxes real (additive); d96 at 40k measures width alone; expectations set with both terms")
        else: verdicts["R1c-1"] = "INDETERMINATE"; say("  VERDICT: INDETERMINATE — effects inside seed noise; report only")
        if abs(inter) >= 5: say(f'  NOTE: interaction {inter:+.1f} >= 5 — the taxes are NOT additive; report prominently')

    say(); say("R1c-2 — A5-class anchors for d96")
    (a11, a11v) = m("A11 d48@40k A5", 0); (a12, a12v) = m("A12 d64@40k A5", 0)
    if math.isnan(a11) or math.isnan(a12) or len(a11v) < 2 or len(a12v) < 2:
        say("  (A11/A12 incomplete — no verdict)"); verdicts["R1c-2"] = "NO-DATA"
    else:
        say(f'  A11 {a11v}={a11:.1f} vs anchor a {a:.1f} (Δ {a11-a:+.1f});  A12 {a12v}={a12:.1f} vs A11 (Δ {a12-a11:+.1f});  A5 d64@53k {m("A5 d64@53k (r1)",0)[1]}={m("A5 d64@53k (r1)",0)[0]:.1f} (the A5 budget axis at d64: {m("A5 d64@53k (r1)",0)[0]-a12:+.1f})')
        eq = "EQUIV" if abs(a11 - a) <= 3 else ("ABOVE" if a11 > a else "BELOW")
        wtax = "ABSENT" if a12 >= a11 - 3 else ("PRESENT" if a12 <= a11 - 5 else "MARGINAL")
        verdicts["R1c-2"] = f"A11-{eq}/WIDTHTAX-{wtax}"
        say(f'  VERDICT: A11 {eq} the floors anchor (floors-inert-at-d48 {"confirmed" if eq=="EQUIV" else "NOT confirmed"} on the A5 substrate; A11 = the d96 baseline); width tax at 40k on A5: {wtax}' + (" — d96-GO prior at the 40k budget" if wtax == "ABSENT" else ""))

    say(); say("R1c-3 (H-41) — the re-pricing rescue: A13 (d48@53k, beta 6e-5) vs A8 (3e-5) and the anchor")
    (a13, a13v) = m("A13 d48@53k b6e-5", 0)
    if math.isnan(a13) or len(a13v) < 2 or math.isnan(b):
        say("  (A13 or A8 incomplete — no verdict)"); verdicts["R1c-3"] = "NO-DATA"
    else:
        i13 = m("A13 d48@53k b6e-5", 2)[0]; i8 = m("b=d48@53k", 2)[0]
        say(f'  A13 {a13v}={a13:.1f} vs A8 {b:.1f} (Δ {a13-b:+.1f}) vs anchor {a:.1f} (Δ {a13-a:+.1f}); I_med A13 {i13:.0f} vs A8 {i8:.0f}')
        if a13 >= b + 5 and a13 >= a - 3: verdicts["R1c-3"] = "RESCUE-FULL"; say("  VERDICT: RESCUE FULL — raising beta with budget cancels the tax; the ladder gains a beta(budget) law; long budgets usable")
        elif a13 >= b + 5: verdicts["R1c-3"] = "RESCUE-PARTIAL"; say("  VERDICT: RESCUE PARTIAL — beta recovers some transfer at long budget; report the gap")
        elif a13 <= b + 2: verdicts["R1c-3"] = "RESCUE-FAILS"; say("  VERDICT: RESCUE FAILS — the budget tax is not priceable at this dose; the ladder holds budgets at 40k")
        else: verdicts["R1c-3"] = "INDETERMINATE"; say("  VERDICT: indeterminate")

    say(); say("R1c-4 (directional) — e1e3 landscape mechanism at the corners")
    rows = [("b=d48@53k A8s1", e13("", "pr1cA8s1")), ("b=d48@53k A8s2", e13("", "pr1cA8s2")), ("c=d64@40k A10s0", e13("", "pr1cA10s0")), ("c=d64@40k A10s1", e13("", "pr1cA10s1"))]
    anch = [(f"a anchor A4s{i}", e13("", f"p13fA4s{i}")) for i in (0, 1, 2)] + [(f"d rung-1 A4s{i}", e13("", f"pr1A4s{i}")) for i in (0, 1, 2)]
    for name, e in anch + rows:
        if e: say(f'  {name:18s} wrong-stable {e["ws"]:>3}/{e["conv"]} ({e["ws"]/max(e["conv"],1):.2f})  nd1 {e["nd1"]:>3}/{e["tot"]}')
    say("  (directional: does spurious-attractor proliferation track budget, width, or both — reported, no rule)")

    say(); say("SUMMARY: " + "  ".join(f"{k}={v}" for k, v in verdicts.items()))
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text("\n".join(LINES) + "\n"); say(); say(f"artifact -> {OUT}")
    return verdicts

def _mk(root, tag, pdir, arm, rg, N, I, eta, ws=None):
    import pickle as pk
    d_ = root / f"lad_{tag}"; d_.mkdir(parents=True, exist_ok=True)
    rows = []
    left = N
    for t in range(48):
        qs = []
        for qi in range(3):
            r = 1 if left > 0 else 0; left -= r
            qs.append(dict(gt_retention=r, q_ladder={e: r for e in EPS}, exact_T=r, I_s=[I*.69, I*.14, I*.085, I*.035, I*.05]))
        rows.append(dict(task=f"t{t}", queries=qs))
    (d_ / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for pref, cnt in (("ladrg", rg // 2), ("ladrgb", rg - rg // 2)):
        d2 = root / f"{pref}_{tag}"; d2.mkdir(parents=True, exist_ok=True)
        rows2 = []; left = cnt
        for t in range(48):
            qs = []
            for qi in range(3):
                r = 1 if left > 0 else 0; left -= r
                qs.append(dict(gt_retention=r, q_ladder={e: r for e in EPS}, exact_T=r, I_s=[1, 1, 1, 1, 1]))
            rows2.append(dict(task=f"g{t}", queries=qs))
        (d2 / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows2))
    if ws is not None:
        d3 = root / f"e1e3_{tag}"; d3.mkdir(parents=True, exist_ok=True)
        rows3 = [dict(task=f"t{t}", e3=[dict(converged_at=1, n_distinct=2, exact_per_step=[], H_q_per_step=[], limit_exact=(i >= ws)) for i in range(3)]) for t in range(48)]
        (d3 / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows3))
    r = REG[arm]; pd_ = root / pdir; pd_.mkdir(parents=True, exist_ok=True)
    cfg = dict(d=r["d"], beta_flux=r["beta"], beta_flux_nl=r["nl"], flux_floors=r["floors"], ni_sigma=r["ni"])
    (pd_ / "ckpt_latest.pkl").write_bytes(pk.dumps(dict(state=dict(model=dict(eq=dict(eta=math.log(eta/(1-eta))))), step=r["steps"], config=cfg)))

def selftest():
    import tempfile, contextlib, io
    checks = []
    def build(root, b=(25, 25, 26), c=(33, 34), a11=(34, 33), a12=(33, 32), a13=(33, 34)):
        for i, g in enumerate((34, 35, 34)): _mk(root, f"p13fA4s{i}", f"pretrain13f_A4s{i}", "A4d48", g, 47, 590, .18, ws=1)
        for i, g in enumerate((27, 23, 24)): _mk(root, f"pr1A4s{i}", f"pretrainr1_A4s{i}", "A4d64", g, 41, 532, .23, ws=2)
        _mk(root, "pr1bA8s0", "pretrainr1b_A8s0", "A8", b[0], 39, 582, .21)
        for i, g in enumerate(b[1:], 1): _mk(root, f"pr1cA8s{i}", f"pretrainr1c_A8s{i}", "A8", g, 40, 580, .21, ws=2)
        for i, g in enumerate(c): _mk(root, f"pr1cA10s{i}", f"pretrainr1c_A10s{i}", "A10", g, 45, 540, .19, ws=1)
        for i, g in enumerate(a11): _mk(root, f"pr1cA11s{i}", f"pretrainr1c_A11s{i}", "A11", g, 46, 520, .18)
        for i, g in enumerate(a12): _mk(root, f"pr1cA12s{i}", f"pretrainr1c_A12s{i}", "A12", g, 44, 470, .19)
        for i, g in enumerate(a13): _mk(root, f"pr1cA13s{i}", f"pretrainr1c_A13s{i}", "A13", g, 42, 430, .21)
        for i, g in enumerate((34, 34, 28)): _mk(root, ["pr1A5s0", "pr1bA5s1", "pr1bA5s2"][i], ["pretrainr1_A5s0", "pretrainr1b_A5s1", "pretrainr1b_A5s2"][i], "A5d64", g, 38, 455, .24)
    def run(**kw):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); build(root, **kw); globals()["RUNS"] = root; globals()["OUT"] = root / "analysis" / "v.txt"
            with contextlib.redirect_stdout(io.StringIO()): return analyze()
    v = run()  # budget-dominant: a 34.3, b 25.3 (Δ9), c 33.5 (Δ0.8); a13 33.5 = rescue-full (Δ vs b +8.2, vs a −0.8)
    checks += [("budget-dominant", v["R1c-1"] == "BUDGET-DOMINANT"), ("A11 equiv + width tax absent", v["R1c-2"] == "A11-EQUIV/WIDTHTAX-ABSENT"), ("rescue full", v["R1c-3"] == "RESCUE-FULL")]
    v = run(b=(33, 34, 33), c=(26, 27))  # width-dominant: Δb48 1.0, Δw40 7.8
    checks.append(("width-dominant", v["R1c-1"] == "WIDTH-DOMINANT"))
    v = run(b=(29, 30, 29), c=(29, 30))  # both ~4.8/4.8
    checks.append(("both", v["R1c-1"] == "BOTH"))
    v = run(b=(32, 33, 32), c=(32, 33))  # deltas ~2: indeterminate
    checks.append(("indeterminate", v["R1c-1"] == "INDETERMINATE"))
    v = run(a13=(26, 27))  # a13 26.5 vs b 25.3: +1.2 -> fails
    checks.append(("rescue fails", v["R1c-3"] == "RESCUE-FAILS"))
    v = run(a13=(31, 30))  # +5.2 vs b, vs a -3.8 -> partial
    checks.append(("rescue partial", v["R1c-3"] == "RESCUE-PARTIAL"))
    v = run(a12=(27, 26))  # A12 26.5 vs A11 33.5: -7 -> width tax present
    checks.append(("A5 width tax present", v["R1c-2"].endswith("WIDTHTAX-PRESENT")))
    n = sum(1 for _, o in checks if o)
    for name, o in checks: print(f"  {'PASS' if o else 'FAIL'}  {name}")
    print(f"selftest: {n}/{len(checks)}"); return n == len(checks)

if __name__ == "__main__":
    sys.exit(0 if selftest() else 1) if "--selftest" in sys.argv else analyze()
