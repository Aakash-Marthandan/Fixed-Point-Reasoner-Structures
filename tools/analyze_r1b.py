# Ledger: RUNG-1b ANALYZER (written 2026-08-19 BEFORE the cells' data exist —
# pre-registration in code). Implements the decision rules of the 2026-08-19
# RUNG-1b LAUNCH REGISTRATION exactly as written there:
#   R1b-1a (H-38, beta-rescale)  A6 (beta 1e-5, n=2) and A7 (3e-6, n=2) vs A5
#            (3e-5, n=3) on rg-96 retention: lower beta buys transfer?
#   R1b-1b (width-ceiling mechanism) best d64 A5-class arm mean rg-96 >= 31 AND
#            vh N >= 44 -> ceiling was SUBSTRATE (price/floors), width back on the
#            table; <= 27 -> GEOMETRIC; between -> reported + (N, rbar) tiebreak
#   R1b-2  (H-39, floors) A5 (n=3) vs A4 (n=3) on rg-96
#   R1b-3  (rt OOD gradient) rt-48 d64 A4 (n=3) vs d48 A4 (n=3: A4s0 + A9 aliases)
#   R1b-4  (A8 = A4-class d48@53,333 s0) eta length-vs-width; budget at fixed width
# ADMISSION at artifact level per arm (d, steps, beta, floors, NI); A9 = rung-0
# A4 ckpts (d48/40k) supplied for rt only. Metric loaders mirror analyze_r1.
"""
  .venv/bin/python tools/analyze_r1b.py            # -> runs/analysis/r1b_verdict.txt
  .venv/bin/python tools/analyze_r1b.py --selftest
"""
from __future__ import annotations
import json, math, os, pickle, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(os.environ.get("QHRRN_RUNS", ROOT / "runs"))
OUT = RUNS / "analysis" / "r1b_verdict.txt"
B = ("pr1b", "pretrainr1b_"); R1 = ("pr1", "pretrainr1_"); R0 = ("p13f", "pretrain13f_")
EPS = ["0.05", "0.1", "0.2", "0.4"]
# registered config per arm id
REG = {"A4": dict(d=64, steps=53333, beta=3e-5, nl=1e-5, floors="350,75,50,15,30", ni=0.01),
       "A5": dict(d=64, steps=53333, beta=3e-5, nl=1e-5, floors="", ni=0.01),
       "A6": dict(d=64, steps=53333, beta=1e-5, nl=3.3e-6, floors="", ni=0.01),
       "A7": dict(d=64, steps=53333, beta=3e-6, nl=1e-6, floors="", ni=0.01),
       "A8": dict(d=48, steps=53333, beta=3e-5, nl=1e-5, floors="350,75,50,15,30", ni=0.01),
       "A9": dict(d=48, steps=40000, beta=3e-5, nl=1e-5, floors="350,75,50,15,30", ni=0.01),
       "A3": dict(d=64, steps=53333, beta=0.0, nl=0.0, floors="", ni=0.0)}
LINES = []
def say(s=""): LINES.append(s); print(s)

def battery(rt, prefix, cell):
    p = RUNS / f"{prefix}_{rt[0]}{cell}" / "results.jsonl"
    if not p.exists(): return None
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    qs = [q for r in rows for q in r["queries"]]
    S = {e: sum(q["q_ladder"][e] for q in qs) for e in EPS}
    radii = [max([float(e) for e in EPS if q["q_ladder"][e]] or [0.0]) for q in qs if q["gt_retention"]]
    return dict(n_tasks=len(rows), N=sum(q["gt_retention"] for q in qs), S=S, ex=sum(q["exact_T"] for q in qs),
                I=float(np.median([sum(q["I_s"]) for q in qs])), rbar=float(np.mean(radii)) if radii else 0.0,
                ret={(r["task"], qi): bool(q["gt_retention"]) for r in rows for qi, q in enumerate(r["queries"])})

def rg96(rt, cell):
    out = dict(ret={}, S0=0, S2=0)
    for pref in ("ladrg", "ladrgb"):
        p = RUNS / f"{pref}_{rt[0]}{cell}" / "results.jsonl"
        if not p.exists(): continue
        for line in p.read_text().splitlines():
            if not line.strip(): continue
            r = json.loads(line)
            for qi, q in enumerate(r["queries"]):
                out["ret"][(pref, r["task"], qi)] = bool(q["gt_retention"]); out["S0"] += q["gt_retention"]; out["S2"] += q["q_ladder"]["0.2"]
    return out if out["ret"] else None

def rt48(rt, cell):
    p = RUNS / f"ladrt_{rt[0]}{cell}" / "results.jsonl"
    if not p.exists(): return None
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return dict(ret=sum(q["gt_retention"] for r in rows for q in r["queries"]), n=sum(len(r["queries"]) for r in rows))

def scalars(rt, cell):
    p = RUNS / f"{rt[1]}{cell}" / "ckpt_latest.pkl"
    if not p.exists(): return None
    ck = pickle.loads(p.read_bytes()); e = ck["state"]["model"]["eq"]["eta"]
    return dict(eta=float(1/(1+np.exp(-float(np.asarray(e))))), step=int(ck["step"]), cfg=ck.get("config", {}) or {})

def admitted(sc, arm):
    if not sc: return False, "no ckpt"
    r = REG[arm]; c = sc["cfg"]; why = []
    if int(c.get("d", -1)) != r["d"]: why.append(f"d={c.get('d')}")
    if sc["step"] != r["steps"]: why.append(f"step={sc['step']}")
    if abs(float(c.get("beta_flux", -1)) - r["beta"]) > 1e-12: why.append(f"beta={c.get('beta_flux')}")
    if abs(float(c.get("beta_flux_nl", -1)) - r["nl"]) > 1e-12: why.append(f"beta_nl={c.get('beta_flux_nl')}")
    if str(c.get("flux_floors", "") or "") != r["floors"]: why.append(f"floors={c.get('flux_floors')!r}")
    if abs(float(c.get("ni_sigma", -1)) - r["ni"]) > 1e-12: why.append(f"ni={c.get('ni_sigma')}")
    return (not why), (", ".join(why) or "ok")

def mcnemar(pa, pb):
    keys = set(pa) & set(pb); b = sum(1 for k in keys if pa[k] and not pb[k]); c = sum(1 for k in keys if pb[k] and not pa[k]); n = b + c
    if n == 0: return b, c, 1.0
    return b, c, min(1.0, sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n * 2)

def mean_spread(xs): return (float(np.mean(xs)), float(max(xs) - min(xs))) if xs else (float("nan"), 0.0)

def analyze():
    LINES.clear()
    say("=" * 100); say("RUNG-1b VERDICT — H-38 beta-rescale / H-39 floors / rt OOD-gradient / A8 eta-budget (d64 unless noted) vs the rung-1 and rung-0 cells"); say("rules R1b-1..R1b-4 as registered 2026-08-19 (analyzer written pre-data)"); say("=" * 100)
    # cells: this campaign (pr1b), rung 1 (pr1), rung 0 (p13f)
    new = {"A5s1": "A5", "A5s2": "A5", "A6s0": "A6", "A6s1": "A6", "A7s0": "A7", "A7s1": "A7", "A8s0": "A8"}
    V, G, SC, OK = {}, {}, {}, {}
    for c, arm in new.items():
        V[c] = battery(B, "lad", c); G[c] = rg96(B, c); SC[c] = scalars(B, c); OK[c] = admitted(SC[c], arm)
    # rung-1 cells (A5s0, A4 x3, A3 x2) from pr1; rt for A4s1/A4s2/A3s1 comes from pr1b (rt-only), for s0 arms from pr1
    r1 = {"A5s0": "A5", "A4s0": "A4", "A4s1": "A4", "A4s2": "A4", "A3s0": "A3", "A3s1": "A3"}
    for c, arm in r1.items():
        V[c] = battery(R1, "lad", c); G[c] = rg96(R1, c); SC[c] = scalars(R1, c); OK[c] = admitted(SC[c], arm)
    RT = {}
    for c in ("A5s1","A5s2","A6s0","A6s1","A7s0","A7s1","A8s0","A4s1","A4s2","A3s1","A9s1","A9s2"): RT[c] = rt48(B, c)
    for c in ("A5s0","A4s0","A3s0"): RT[c] = rt48(R1, c)
    RT["*A4s0"] = rt48(R0, "A4s0")   # rung-0 d48 A4s0 rt
    # d48 anchor val-hard/rg for A8 comparison
    A48 = {s: (battery(R0, "lad", f"A4s{s}"), rg96(R0, f"A4s{s}"), scalars(R0, f"A4s{s}")) for s in (0, 1, 2)}
    # A9 admission (rung-0 ckpts supplied under the r1b tag)
    for c in ("A9s1", "A9s2"):
        SC[c] = scalars(B, c); OK[c] = admitted(SC[c], "A9")

    say(); say("SECTION 1 — presence + ARTIFACT-LEVEL ADMISSION")
    for c in list(new) + ["A9s1", "A9s2"] + list(r1):
        adm, why = OK.get(c, (False, "no ckpt")); say(f'  {c:6s} {"ADMITTED" if adm else "EXCLUDED (" + why + ")":34s} lad {"y" if V.get(c) else "-"} rg96 {"y" if G.get(c) else "-"} rt {"y" if RT.get(c) else "-"}')
        if V.get(c) and V[c]["n_tasks"] != 48: say(f"  !! {c} val-hard {V[c]['n_tasks']}/48")
        if G.get(c) and len(G[c]["ret"]) != 288: say(f"  !! {c} rg-96 {len(G[c]['ret'])}/288")
    for c in list(V):
        if not OK.get(c, (False,))[0]: V[c] = None; G[c] = None; SC[c] = None
    for c in ("A9s1", "A9s2"):
        if not OK.get(c, (False,))[0]: RT[c] = None

    say(); say("SECTION 2 — the table")
    say(f'  {"cell":6s} {"arm":10s} {"N":>3s} {"S.2":>4s} {"S.4":>4s} {"ex":>3s} {"I_med":>6s} {"eta":>5s} {"rbar":>5s} | {"rg96":>4s} {"rgS2":>4s} | {"rt48":>4s}')
    def row(c, arm):
        v, g, sc, rt = V.get(c), G.get(c), SC.get(c), RT.get(c)
        say(f'  {c:6s} {arm:10s} {v["N"] if v else 0:>3d} {v["S"]["0.2"] if v else 0:>4d} {v["S"]["0.4"] if v else 0:>4d} {v["ex"] if v else 0:>3d} {v["I"] if v else 0:>6.0f} {sc["eta"] if sc else 0:>5.3f} {v["rbar"] if v else 0:>5.2f} | {g["S0"] if g else 0:>4d} {g["S2"] if g else 0:>4d} | {rt["ret"] if rt else "-":>4}')
    for c, arm in [("A5s0","A5 b3e-5"),("A5s1","A5 b3e-5"),("A5s2","A5 b3e-5"),("A6s0","A6 b1e-5"),("A6s1","A6 b1e-5"),("A7s0","A7 b3e-6"),("A7s1","A7 b3e-6"),("A4s0","A4 floors"),("A4s1","A4 floors"),("A4s2","A4 floors"),("A3s0","A3 plain"),("A3s1","A3 plain"),("A8s0","A8 d48@53k")]: row(c, arm)
    say(f'  d48 A4 anchor (rung 0): N {[A48[s][0]["N"] for s in (0,1,2) if A48[s][0]]}  rg96 {[A48[s][1]["S0"] for s in (0,1,2) if A48[s][1]]}  eta {[round(A48[s][2]["eta"],3) for s in (0,1,2) if A48[s][2]]}  rt48 {[RT.get("*A4s0",{}).get("ret") if RT.get("*A4s0") else None, RT.get("A9s1",{}).get("ret") if RT.get("A9s1") else None, RT.get("A9s2",{}).get("ret") if RT.get("A9s2") else None]}')
    verdicts = {}
    a5 = [G[c]["S0"] for c in ("A5s0","A5s1","A5s2") if G.get(c)]; a6 = [G[c]["S0"] for c in ("A6s0","A6s1") if G.get(c)]; a7 = [G[c]["S0"] for c in ("A7s0","A7s1") if G.get(c)]; a4 = [G[c]["S0"] for c in ("A4s0","A4s1","A4s2") if G.get(c)]
    I5 = [V[c]["I"] for c in ("A5s0","A5s1","A5s2") if V.get(c)]; I6 = [V[c]["I"] for c in ("A6s0","A6s1") if V.get(c)]; I7 = [V[c]["I"] for c in ("A7s0","A7s1") if V.get(c)]
    N5 = [V[c]["N"] for c in ("A5s0","A5s1","A5s2") if V.get(c)]; N6 = [V[c]["N"] for c in ("A6s0","A6s1") if V.get(c)]; N7 = [V[c]["N"] for c in ("A7s0","A7s1") if V.get(c)]

    # ---------------- R1b-1a ----------------
    say(); say("R1b-1a (H-38) — beta-rescale at d64, A5-class: A6 (1e-5) and A7 (3e-6) vs A5 (3e-5) on rg-96 retention (primary); vh N and I_med reported")
    if len(a5) < 1 or (len(a6) < 2 and len(a7) < 2):
        say("  (A5 or both rescale arms missing — no verdict)"); verdicts["R1b-1a"] = "NO-DATA"
    else:
        m5 = float(np.mean(a5)); say(f'  A5 rg96 {a5} mean {m5:.1f} (n={len(a5)})  N {N5}  I_med {[round(x) for x in I5]}')
        sup = fal = 0; detail = []
        for name, arr, Ia, Na in (("A6 b1e-5", a6, I6, N6), ("A7 b3e-6", a7, I7, N7)):
            if len(arr) < 2: say(f"  {name}: n={len(arr)} — underpowered, reported only {arr}"); continue
            d = float(np.mean(arr)) - m5; both_above = all(x > m5 for x in arr)
            say(f'  {name}: rg96 {arr} mean {np.mean(arr):.1f} (Δ vs A5 {d:+.1f}; both seeds above A5 mean: {both_above})  N {Na}  I_med {[round(x) for x in Ia]}')
            if d >= 5 and both_above: sup += 1; detail.append(name)
            elif d <= 2: fal += 1
        if sup: verdicts["R1b-1a"] = "SUPPORTED"; say(f"  VERDICT: H-38 SUPPORTED — lowering beta raises unseen-family transfer at d64 ({', '.join(detail)}): the knee moved with capacity; re-tune beta per rung")
        elif fal == sum(1 for arr in (a6, a7) if len(arr) >= 2) and fal > 0: verdicts["R1b-1a"] = "FALSIFIED"; say("  VERDICT: H-38 FALSIFIED — lower beta does not buy transfer at d64 (Δ ≤ +2 for every rescale arm); the ceiling is not a fixed-beta artifact")
        else: verdicts["R1b-1a"] = "INDETERMINATE"; say("  VERDICT: INDETERMINATE (Δ between +2 and +5, or seeds straddle A5) — reported; no re-tuning claim")

    # ---------------- R1b-1b ----------------
    say(); say("R1b-1b — the width-ceiling MECHANISM call: best d64 A5-class arm (mean over seeds, n>=2) on rg-96 and vh N vs the d48 A4 anchor (34.3 / 47.3)")
    arms = [("A5", a5, N5, [V[c] for c in ("A5s0","A5s1","A5s2") if V.get(c)]), ("A6", a6, N6, [V[c] for c in ("A6s0","A6s1") if V.get(c)]), ("A7", a7, N7, [V[c] for c in ("A7s0","A7s1") if V.get(c)])]
    arms = [(n, r, N, v) for n, r, N, v in arms if len(r) >= 2]
    if not arms:
        say("  (no A5-class arm at n>=2 — no verdict)"); verdicts["R1b-1b"] = "NO-DATA"
    else:
        best = max(arms, key=lambda t: np.mean(t[1])); bn, br, bN, bv = best
        mr, mN = float(np.mean(br)), float(np.mean(bN)); say(f'  best arm {bn}: rg96 {br} mean {mr:.1f}; N {bN} mean {mN:.1f}; rbar mean {np.mean([x["rbar"] for x in bv]):.3f}')
        if mr >= 31 and mN >= 44: verdicts["R1b-1b"] = "SUBSTRATE"; say(f"  VERDICT: the d64 ceiling was SUBSTRATE (floors/price) — {bn} at d64 matches the d48 anchor on transfer and codebook; WIDTH IS BACK ON THE TABLE: register d96 with this arm's price and NO floors")
        elif mr <= 27: verdicts["R1b-1b"] = "GEOMETRIC"; say("  VERDICT: GEOMETRIC — no d64 substrate reaches the d48 anchor; depth-lean stands")
        else:
            # tiebreak: (N, rbar) vs d48 A4 anchor
            aN = np.mean([A48[s][0]["N"] for s in (0,1,2) if A48[s][0]]); ar = np.mean([A48[s][0]["rbar"] for s in (0,1,2) if A48[s][0]])
            bR = np.mean([x["rbar"] for x in bv])
            say(f'  tiebreak (N, rbar): {bn} ({mN:.1f}, {bR:.3f}) vs d48 A4 ({aN:.1f}, {ar:.3f})')
            if mN >= aN - 2 and bR >= ar - .01: verdicts["R1b-1b"] = "SUBSTRATE(tiebreak)"; say("  VERDICT: PARTIAL recovery; (N, rbar) at the d48 frontier -> treat as substrate (tiebreak) — PI decides whether d96 is worth a pilot")
            else: verdicts["R1b-1b"] = "INDETERMINATE"; say("  VERDICT: INDETERMINATE — transfer partly recovers but codebook does not; reported, PI decides (depth-lean remains the default)")

    # ---------------- R1b-2 ----------------
    say(); say("R1b-2 (H-39) — floors with NI at d64: A5 (n=3) vs A4 (n=3) on rg-96 retention")
    if len(a5) < 3 or len(a4) < 3:
        say(f"  (need 3+3; have A5 {len(a5)}, A4 {len(a4)} — reported only)"); verdicts["R1b-2"] = "NO-DATA" if len(a5) < 2 else "UNDERPOWERED"
        if a5 and a4: say(f"  A5 {a5} vs A4 {a4}")
    else:
        m5, m4 = float(np.mean(a5)), float(np.mean(a4)); above = sum(1 for x in a5 if x > max(a4)); below = sum(1 for x in a5 if x < min(a4))
        I4 = [V[c]["I"] for c in ("A4s0","A4s1","A4s2") if V.get(c)]
        say(f'  A5 {a5} mean {m5:.1f} vs A4 {a4} mean {m4:.1f}; A5 seeds above A4 max: {above}/3, below A4 min: {below}/3; floors cost in nats (A4−A5 I_med): {np.mean(I4)-np.mean(I5):+.0f}')
        if m5 - m4 >= 5 and above >= 2: verdicts["R1b-2"] = "FLOORS-COST"; say("  VERDICT: FLOORS COST TRANSFER at d64 — operating substrate -> A5-class (global+NI); floors dropped from the ladder")
        elif m4 - m5 >= 5 and below >= 2: verdicts["R1b-2"] = "FLOORS-HELP"; say("  VERDICT: floors HELP with NI at d64 (interaction) — keep A4-class")
        else: verdicts["R1b-2"] = "INERT"; say("  VERDICT: floors INERT (rung-0 reading stands) — A5-class adopted by parsimony (one fewer constant), as pre-stated")

    # ---------------- R1b-3 ----------------
    say(); say("R1b-3 — rt-48 OOD gradient: trained-family fresh instances at d64 A4 (n=3) vs d48 A4 (n=3: rung-0 A4s0 + A9 aliases); A5-class and plain reported")
    rt64 = [RT[c]["ret"] for c in ("A4s0","A4s1","A4s2") if RT.get(c)]; rt48l = [RT[c]["ret"] for c in ("*A4s0","A9s1","A9s2") if RT.get(c)]
    say(f'  d64 A4 rt {rt64}  d48 A4 rt {rt48l}  | A5-class: A5 {[RT[c]["ret"] for c in ("A5s0","A5s1","A5s2") if RT.get(c)]} A6 {[RT[c]["ret"] for c in ("A6s0","A6s1") if RT.get(c)]} A7 {[RT[c]["ret"] for c in ("A7s0","A7s1") if RT.get(c)]} | plain d64 {[RT[c]["ret"] for c in ("A3s0","A3s1") if RT.get(c)]}')
    if len(rt64) >= 3 and len(rt48l) >= 3:
        d = float(np.mean(rt64) - np.mean(rt48l)); lower = sum(1 for a, b in zip(sorted(rt64), sorted(rt48l)) if a < b)
        if abs(d) <= 3: verdicts["R1b-3"] = "GRADIENT"; say(f"  VERDICT: rt FLAT across width (Δ {d:+.1f}) while vh (−6.7) and rg-96 (−9.6) fall — the width deficit is OOD-GRADED (in-family generalization preserved, cross-family pays)")
        elif d <= -5 and lower >= 2: verdicts["R1b-3"] = "UNIFORM"; say(f"  VERDICT: rt falls too (Δ {d:+.1f}) — the deficit is UNIFORM, not OOD-specific")
        elif d >= 5: verdicts["R1b-3"] = "SPECIALIZES"; say(f"  VERDICT: rt RISES at d64 (Δ {d:+.1f}) — width specializes to trained families (stronger gradient)")
        else: verdicts["R1b-3"] = "INDETERMINATE"; say(f"  VERDICT: indeterminate (Δ {d:+.1f}) — reported")
    else: verdicts["R1b-3"] = "NO-DATA"; say("  (need 3+3 rt cells — no verdict)")

    # ---------------- R1b-4 ----------------
    say(); say("R1b-4 — A8 = A4-class d48@53,333 (s0): eta length-vs-width; budget at fixed width (directional, n=1)")
    s8 = SC.get("A8s0"); v8 = V.get("A8s0"); g8 = G.get("A8s0")
    if not s8:
        say("  (A8 missing — no verdict)"); verdicts["R1b-4"] = "NO-DATA"
    else:
        e8 = s8["eta"]; e48 = [A48[s][2]["eta"] for s in (0,1,2) if A48[s][2]]; e64 = [SC[c]["eta"] for c in ("A4s0","A4s1","A4s2") if SC.get(c)]
        say(f'  eta A8 {e8:.3f} vs d48/40k A4 {[round(x,3) for x in e48]} vs d64/53k A4 {[round(x,3) for x in e64]}')
        if e8 <= .20: verdicts["R1b-4eta"] = "WIDTH"; say("  eta VERDICT: eta follows WIDTH (more steps at d48 leave it at the d48 value)")
        elif e8 >= .22: verdicts["R1b-4eta"] = "LENGTH"; say("  eta VERDICT: eta follows OPTIMIZATION LENGTH (d48 at 53k reaches the d64/53k value)")
        else: verdicts["R1b-4eta"] = "INDETERMINATE"; say("  eta VERDICT: between the bands — indeterminate")
        if v8 and g8:
            N48 = [A48[s][0]["N"] for s in (0,1,2) if A48[s][0]]; R48 = [A48[s][1]["S0"] for s in (0,1,2) if A48[s][1]]; I48 = [A48[s][0]["I"] for s in (0,1,2) if A48[s][0]]
            say(f'  A8 N {v8["N"]} rg96 {g8["S0"]} I_med {v8["I"]:.0f} vs d48/40k A4 band N {N48} rg96 {R48} I_med {[round(x) for x in I48]}')
            if g8["S0"] <= min(R48) - 5 and v8["I"] < min(I48): verdicts["R1b-4budget"] = "OVERCOMPRESS"; say("  budget VERDICT (directional): more budget at fixed beta and fixed width LOSES transfer while compressing further — H-38's mechanism operates on the budget axis too")
            elif min(R48) - 3 <= g8["S0"]: verdicts["R1b-4budget"] = "NEUTRAL"; say("  budget VERDICT (directional): budget-neutral at d48 — the d64 deficit is width-specific, not budget-induced")
            else: verdicts["R1b-4budget"] = "INDETERMINATE"; say("  budget VERDICT: indeterminate — reported")

    say(); say("SUMMARY: " + "  ".join(f"{k}={v}" for k, v in verdicts.items()))
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text("\n".join(LINES) + "\n"); say(); say(f"artifact -> {OUT}")
    return verdicts

# ---------------- synthetic self-test ----------------
def _cell(root, rt, cell, arm, N, s4, rg, eta, rt_ret=None, I=500.0, step=None, d=None):
    import pickle as pk
    def rows(n_tasks, n_ret, s2n, s4n, name):
        out = []; rc = s2 = s4c = 0
        for t in range(n_tasks):
            qs = []
            for qi in range(3):
                r = 1 if rc < n_ret else 0; rc += r; l2 = 1 if (r and s2 < s2n) else 0; s2 += l2; l4 = 1 if (l2 and s4c < s4n) else 0; s4c += l4
                qs.append(dict(gt_retention=r, q_ladder={"0.05": r, "0.1": r, "0.2": l2, "0.4": l4}, exact_T=r, I_s=[I*.69, I*.14, I*.085, I*.035, I*.05]))
            out.append(dict(task=f"{name}{t}", queries=qs))
        return out
    def dump(dirn, rws): d_ = root / dirn; d_.mkdir(parents=True, exist_ok=True); (d_ / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rws))
    if N is not None:
        dump(f"lad_{rt[0]}{cell}", rows(48, N, N, s4, "vh")); dump(f"ladrg_{rt[0]}{cell}", rows(48, rg // 2, rg // 4, 0, "rg")); dump(f"ladrgb_{rt[0]}{cell}", rows(48, rg - rg // 2, rg // 4, 0, "rb"))
    if rt_ret is not None: dump(f"ladrt_{rt[0]}{cell}", rows(48, rt_ret, rt_ret // 2, 0, "rt"))
    r = REG[arm]; pdir = root / f"{rt[1]}{cell}"; pdir.mkdir(parents=True, exist_ok=True)
    cfg = dict(d=d or r["d"], T=6, beta_flux=r["beta"], beta_flux_nl=r["nl"], flux_floors=r["floors"], ni_sigma=r["ni"])
    (pdir / "ckpt_latest.pkl").write_bytes(pk.dumps(dict(state=dict(model=dict(eq=dict(eta=math.log(eta/(1-eta))))), step=step or r["steps"], config=cfg)))

def selftest():
    import tempfile, contextlib, io
    checks = []
    def build(root, a6=(40, 41), a7=(38, 39), a5=(34, 33, 35), N5=43, N6=45, rt64=(27, 26, 28), rt48=(26, 25, 27), e8=.24, rg8=28, I8=540, a4=(27, 23, 24)):
        # rung 1 cells (pr1)
        for c, r in zip(("A4s0","A4s1","A4s2"), a4): _cell(root, R1, c, "A4", 41, 18, r, .23, rt_ret=None, I=532)
        _cell(root, R1, "A4s0", "A4", 41, 18, a4[0], .23, rt_ret=rt64[0], I=532)
        for c, r in zip(("A3s0","A3s1"), (11, 9)): _cell(root, R1, c, "A3", 22, 11, r, .246, rt_ret=6 if c == "A3s0" else None, I=300000)
        _cell(root, R1, "A5s0", "A5", N5, 22, a5[0], .235, rt_ret=16, I=468)
        # rung 0 anchor (p13f)
        for s, r, N, e in zip((0,1,2), (34,35,34), (48,47,47), (.177,.179,.182)): _cell(root, R0, f"A4s{s}", "A9", N, 21, r, e, rt_ret=rt48[0] if s == 0 else None, I=586)
        # this campaign (pr1b)
        _cell(root, B, "A5s1", "A5", N5, 22, a5[1], .235, rt_ret=16, I=468); _cell(root, B, "A5s2", "A5", N5, 22, a5[2], .235, rt_ret=16, I=468)
        _cell(root, B, "A6s0", "A6", N6, 22, a6[0], .235, rt_ret=18, I=520); _cell(root, B, "A6s1", "A6", N6, 22, a6[1], .235, rt_ret=18, I=520)
        _cell(root, B, "A7s0", "A7", N6, 22, a7[0], .235, rt_ret=18, I=560); _cell(root, B, "A7s1", "A7", N6, 22, a7[1], .235, rt_ret=18, I=560)
        _cell(root, B, "A8s0", "A8", 46, 20, rg8, e8, rt_ret=25, I=I8)
        for c, r in zip(("A4s1","A4s2"), rt64[1:]): _cell(root, B, c, "A4", None, 0, 0, .23, rt_ret=r)
        _cell(root, B, "A3s1", "A3", None, 0, 0, .246, rt_ret=5)
        for c, r in zip(("A9s1","A9s2"), rt48[1:]): _cell(root, B, c, "A9", None, 0, 0, .18, rt_ret=r)
    def run(**kw):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); build(root, **kw); globals()["RUNS"] = root; globals()["OUT"] = root / "analysis" / "v.txt"
            with contextlib.redirect_stdout(io.StringIO()): return analyze()
    v = run(); checks += [("H-38 supported (A6 +6.7 both above)", v["R1b-1a"] == "SUPPORTED"), ("substrate (best arm rg 40.5, N 45)", v["R1b-1b"] == "SUBSTRATE"), ("floors cost (34 vs 24.7, 3 above)", v["R1b-2"] == "FLOORS-COST"), ("rt gradient (27 vs 26)", v["R1b-3"] == "GRADIENT"), ("eta length (.24)", v["R1b-4eta"] == "LENGTH"), ("budget overcompress (rg 28 < 29, I 540 < 560)", v["R1b-4budget"] == "OVERCOMPRESS")]
    v = run(a6=(35, 33), a7=(34, 36), N6=43); checks += [("H-38 falsified (Δ ≤ 2)", v["R1b-1a"] == "FALSIFIED"), ("mechanism: A5-class best 35 -> tiebreak/indeterminate not substrate", v["R1b-1b"] != "SUBSTRATE")]
    v = run(a6=(37, 36), a7=(33, 34)); checks.append(("H-38 indeterminate (Δ +2.5)", v["R1b-1a"] == "INDETERMINATE"))
    v = run(a5=(26, 24, 25), a6=(25, 26), a7=(24, 25), N5=41, N6=41); checks += [("floors inert (25 vs 24.7)", v["R1b-2"] == "INERT"), ("geometric (best ≤ 27)", v["R1b-1b"] == "GEOMETRIC")]
    v = run(a5=(18, 19, 17)); checks.append(("floors help (A4 24.7 vs A5 18)", v["R1b-2"] == "FLOORS-HELP"))
    v = run(rt64=(20, 19, 21)); checks.append(("rt uniform deficit (20 vs 26)", v["R1b-3"] == "UNIFORM"))
    v = run(e8=.18, rg8=34, I8=600); checks += [("eta width (.18)", v["R1b-4eta"] == "WIDTH"), ("budget neutral (rg 34)", v["R1b-4budget"] == "NEUTRAL")]
    n = sum(1 for _, o in checks if o)
    for name, o in checks: print(f"  {'PASS' if o else 'FAIL'}  {name}")
    print(f"selftest: {n}/{len(checks)}"); return n == len(checks)

if __name__ == "__main__":
    sys.exit(0 if selftest() else 1) if "--selftest" in sys.argv else analyze()
