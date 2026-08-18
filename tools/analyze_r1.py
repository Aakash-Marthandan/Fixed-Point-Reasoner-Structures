# Ledger: RUNG-1 ANALYZER (written 2026-08-18 night, BEFORE the campaign's
# batteries exist — pre-registration in code, as for rung 0). Implements the
# decision rules of the 2026-08-15 RUNG-1 LAUNCH REGISTRATION exactly as
# written there:
#   R1-1 (H-36)  WIDTH CEILING: seeded d64 A4 (n=3) vs the seeded d48 A4
#                anchor (rung 0, n=3) on rg-96 retention AND rg-96 S(.2)
#                [val-hard S(.4) reported alongside]; >=2/3 seed-pairs
#                d64>=d48 on BOTH -> ceiling was noise (dense ladder, d72);
#                >=2/3 d64<d48 on both -> CEILING CONFIRMED (depth-lean);
#                mixed -> INDETERMINATE, packing-plane (N, rbar) tiebreak
#   R1-2 (H-30 restated) priced d64/T6 eta ~.24 (.20-.28) AND
#                |plain-priced|/priced <= .15 -> HOLDS; plain off by >25% ->
#                price effect does NOT vanish with width
#   R1-3 (Law-4 at d64) A4 (n=3) vs A3 (n=2) rg-96 retention: priced>plain
#                in >=4/6 seed comparisons -> Law-4 extends to d64
#   R1-4 (NI attribution) A5 global+NI (n=1) vs A4 floors+NI (n=3) rg-96:
#                within the A4 seed band -> NI alone reproduces A4; well
#                below -> floors matter with NI (n=1: directional)
#   R1-5 (A4 seed-invariance) rg-96 spread of A4's three seeds: <=3
#                replicates the stabilizer claim; >8 = luck
#   R1-6 (throat / profile) d64 A4 I_med vs the curve; knee-profile
#                distance <=~.10 for priced, plain on free (H-34, 2nd seeded)
# ADMISSION: a cell enters a verdict only if its checkpoint IS the
# registered config at artifact level (d/T/steps/beta/floors/NI per arm) —
# "only keep good training results" (PI, 2026-08-18). Metric code mirrors
# tools/analyze_r0.py so d64 and d48 are computed identically.
# Statistics: per-seed pairing on identical task sets; McNemar exact
# binomial; the measurement law binds (no sub-10-count cross-run claims).
"""
  .venv/bin/python tools/analyze_r1.py            # -> runs/analysis/r1_verdict.txt
  .venv/bin/python tools/analyze_r1.py --selftest # synthetic ground-truth checks
"""
from __future__ import annotations
import json
import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(os.environ.get("QHRRN_RUNS", ROOT / "runs"))
OUT = RUNS / "analysis" / "r1_verdict.txt"

# rung tags: (battery infix, pretrain dir prefix)
R1 = ("pr1", "pretrainr1_")     # this rung
R0 = ("p13f", "pretrain13f_")   # the d48 anchor (rung 0)
ARM_SEEDS = {"A4": [0, 1, 2], "A3": [0, 1], "A5": [0]}
KNEE = np.array([.69, .14, .085, .035, .048])
FREE = np.array([.76, .18, .045, .012, .003])
EPS = [0.05, 0.1, 0.2, 0.4]
# registered config per arm (artifact-level admission)
REG = {
    "d": 64, "T": 6, "steps": 53333,
    "A4": dict(beta_flux=3e-5, beta_flux_nl=1e-5, flux_floors="350,75,50,15,30", ni_sigma=0.01),
    "A5": dict(beta_flux=3e-5, beta_flux_nl=1e-5, flux_floors="", ni_sigma=0.01),
    "A3": dict(beta_flux=0.0, beta_flux_nl=0.0, flux_floors="", ni_sigma=0.0),
}
LINES: list[str] = []


def say(s=""):
    LINES.append(s)
    print(s)


def tag(arm, seed):
    return f"{arm}s{seed}"


# ---------------- loaders (mirror analyze_r0 line for line) ----------------
def battery(rt, prefix, arm, seed):
    p = RUNS / f"{prefix}_{rt[0]}{tag(arm, seed)}" / "results.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    S = {e: 0 for e in ("0", "0.05", "0.1", "0.2", "0.4")}
    ex = 0; I = []; ret, s4 = {}, {}; radii = []
    for r in rows:
        for qi, q in enumerate(r["queries"]):
            k = (r["task"], qi)
            S["0"] += q["gt_retention"]
            for e in ("0.05", "0.1", "0.2", "0.4"):
                S[e] += q["q_ladder"][e]
            ex += q["exact_T"]
            I.append(sum(q["I_s"]))
            ret[k] = bool(q["gt_retention"]); s4[k] = bool(q["q_ladder"]["0.4"])
            if q["gt_retention"]:   # packing plane: N = retained, rbar = mean max-survived eps
                surv = [e for e in EPS if q["q_ladder"][str(e)]]
                radii.append(max(surv) if surv else 0.0)
    spec = np.median(np.array([q["I_s"] for r in rows for q in r["queries"]]), axis=0)
    return dict(n_tasks=len(rows), S=S, ex=ex, I=float(np.median(I)), ret=ret, s4=s4,
                spec=spec, prof=spec / spec.sum(), N=len(radii),
                rbar=float(np.mean(radii)) if radii else 0.0)


def rg96(rt, arm, seed):
    """G4: rg-96 = frozen rg-48 (ladrg) U rb-48 (ladrgb), keys namespaced."""
    out = dict(ret={}, s2={}, S0=0, S2=0)
    for pref in ("ladrg", "ladrgb"):
        p = RUNS / f"{pref}_{rt[0]}{tag(arm, seed)}" / "results.jsonl"
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            for qi, q in enumerate(r["queries"]):
                k = (pref, r["task"], qi)
                out["ret"][k] = bool(q["gt_retention"]); out["s2"][k] = bool(q["q_ladder"]["0.2"])
                out["S0"] += q["gt_retention"]; out["S2"] += q["q_ladder"]["0.2"]
    return out if out["ret"] else None


def scalars(rt, arm, seed):
    """eta from the ckpt + the ARTIFACT-LEVEL admission record."""
    p = RUNS / f"{rt[1]}{tag(arm, seed)}" / "ckpt_latest.pkl"
    if not p.exists():
        return None
    ck = pickle.loads(p.read_bytes())
    eq = ck["state"]["model"]["eq"]
    e = 1 / (1 + np.exp(-float(np.asarray(eq["eta"]))))
    cfg = ck.get("config", {}) or {}
    return dict(eta=float(e), step=int(ck["step"]), cfg=cfg)


def admitted(sc, arm, want_d):
    """True iff the ckpt is the registered config for this arm (d, T, steps, flags)."""
    if not sc:
        return False, "no ckpt"
    c = sc["cfg"]; why = []
    if int(c.get("d", -1)) != want_d: why.append(f"d={c.get('d')}")
    if int(c.get("T", -1)) != REG["T"]: why.append(f"T={c.get('T')}")
    if want_d == REG["d"] and sc["step"] != REG["steps"]: why.append(f"step={sc['step']}")
    r = REG[arm]
    if abs(float(c.get("beta_flux", -1)) - r["beta_flux"]) > 1e-12: why.append(f"beta={c.get('beta_flux')}")
    if abs(float(c.get("beta_flux_nl", -1)) - r["beta_flux_nl"]) > 1e-12: why.append(f"beta_nl={c.get('beta_flux_nl')}")
    if str(c.get("flux_floors", "") or "") != r["flux_floors"]: why.append(f"floors={c.get('flux_floors')!r}")
    if abs(float(c.get("ni_sigma", -1)) - r["ni_sigma"]) > 1e-12: why.append(f"ni={c.get('ni_sigma')}")
    return (not why), (", ".join(why) or "ok")


def mcnemar(pa, pb):
    keys = set(pa) & set(pb)
    b = sum(1 for k in keys if pa[k] and not pb[k])
    c = sum(1 for k in keys if pb[k] and not pa[k])
    n = b + c
    if n == 0:
        return b, c, 1.0
    p = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n * 2
    return b, c, min(p, 1.0)


# ---------------- the verdict ----------------
def analyze():
    LINES.clear()
    say("=" * 96)
    say("RUNG-1 VERDICT — H-36 width-ceiling decider (d64/T6@53,333: A4 x3, A3 x2, A5 x1) vs the d48 anchor (rung 0)")
    say("decision rules R1-1..R1-6 as registered 2026-08-15 (analyzer written pre-data 2026-08-18)")
    say("=" * 96)

    vh, rg, sc, ok = {}, {}, {}, {}
    for a, seeds in ARM_SEEDS.items():
        for s in seeds:
            vh[(a, s)] = battery(R1, "lad", a, s); rg[(a, s)] = rg96(R1, a, s); sc[(a, s)] = scalars(R1, a, s)
            ok[(a, s)] = admitted(sc[(a, s)], a, REG["d"])
    # d48 anchor (A4 x3 for R1-1; A3 x3 kept for the report)
    vh0, rg0, sc0 = {}, {}, {}
    for a in ("A4", "A3"):
        for s in (0, 1, 2):
            vh0[(a, s)] = battery(R0, "lad", a, s); rg0[(a, s)] = rg96(R0, a, s); sc0[(a, s)] = scalars(R0, a, s)

    say(); say("SECTION 1 — cells present + ARTIFACT-LEVEL ADMISSION (registered config only)")
    n_cells = sum(len(v) for v in ARM_SEEDS.values())
    say(f"  val-hard cells: {sum(1 for k in vh if vh[k])}/{n_cells}   rg-96 cells: {sum(1 for k in rg if rg[k])}/{n_cells}"
        f"   ckpts: {sum(1 for k in sc if sc[k])}/{n_cells}   d48-anchor A4 cells: {sum(1 for s in (0,1,2) if vh0[('A4',s)] and rg0[('A4',s)])}/3")
    for k in vh:
        adm, why = ok[k]
        flag = "ADMITTED" if adm else f"EXCLUDED ({why})"
        say(f"  {tag(*k):6s} ckpt {flag}" + (f" step {sc[k]['step']}" if sc[k] else ""))
        if vh[k] and vh[k]["n_tasks"] != 48: say(f"  !! {tag(*k)} val-hard {vh[k]['n_tasks']}/48 tasks")
        if rg[k] and len(rg[k]["ret"]) != 288: say(f"  !! {tag(*k)} rg-96 {len(rg[k]['ret'])}/288 pairs")
    for s in (0, 1, 2):
        a0, why0 = admitted(sc0[("A4", s)], "A4", 48)
        if not a0: say(f"  !! anchor A4s{s} (d48) ckpt not the registered config: {why0}")
    # only ADMITTED cells feed the rules
    for k in list(vh):
        if not ok[k][0]:
            vh[k] = None; rg[k] = None; sc[k] = None

    say(); say("SECTION 2 — the table (per cell; d48 anchor rows marked *)")
    say(f'  {"cell":8s} {"S0":>4s} {"S.2":>4s} {"S.4":>4s} {"ex":>3s} {"I_med":>6s} {"eta":>5s} {"dknee":>6s} {"dfree":>6s} | {"rg96_ret":>8s} {"rg96_S2":>7s} | {"N":>3s} {"rbar":>5s}')
    def row(lbl, v, g, x):
        if not v: return
        dk = float(np.abs(v["prof"] - KNEE).sum()); df = float(np.abs(v["prof"] - FREE).sum())
        say(f'  {lbl:8s} {v["S"]["0"]:>4d} {v["S"]["0.2"]:>4d} {v["S"]["0.4"]:>4d} {v["ex"]:>3d} {v["I"]:>6.0f} '
            f'{x["eta"] if x else 0:>5.3f} {dk:>6.3f} {df:>6.3f} | {g["S0"] if g else 0:>8d} {g["S2"] if g else 0:>7d} | {v["N"]:>3d} {v["rbar"]:>5.2f}')
    for a, seeds in ARM_SEEDS.items():
        for s in seeds: row(tag(a, s), vh[(a, s)], rg[(a, s)], sc[(a, s)])
    for a in ("A4", "A3"):
        for s in (0, 1, 2): row(f"*{tag(a, s)}", vh0[(a, s)], rg0[(a, s)], sc0[(a, s)])

    verdicts = {}
    # ---------------- R1-1 ----------------
    say(); say("R1-1 (H-36) — WIDTH CEILING: d64 A4 vs d48 A4 anchor, per seed-pair, on rg-96 retention AND rg-96 S(.2) [+ vh S(.4)]")
    up = down = 0; pairs = 0
    for s in (0, 1, 2):
        g1, g0, v1, v0 = rg[("A4", s)], rg0[("A4", s)], vh[("A4", s)], vh0[("A4", s)]
        if not (g1 and g0): continue
        pairs += 1
        both_up = g1["S0"] >= g0["S0"] and g1["S2"] >= g0["S2"]
        both_dn = g1["S0"] < g0["S0"] and g1["S2"] < g0["S2"]
        up += both_up; down += both_dn
        s4 = f'{v1["S"]["0.4"]} vs {v0["S"]["0.4"]}' if (v1 and v0) else "-"
        say(f'  seed {s}: rg96 ret {g1["S0"]} vs {g0["S0"]}   S(.2) {g1["S2"]} vs {g0["S2"]}   vh S(.4) {s4}   -> '
            f'{"d64>=d48 both" if both_up else ("d64<d48 both" if both_dn else "mixed")}')
    if pairs == 0:
        say("  (no seed-pairs — no verdict)"); verdicts["R1-1"] = "NO-DATA"
    else:
        say(f'  seed-pairs: d64>=d48 on both {up}/{pairs}; d64<d48 on both {down}/{pairs}')
        if up >= 2:
            verdicts["R1-1"] = "NOISE"; say("  VERDICT: the d48 ceiling was NOISE — DENSE-WIDTH LADDER proceeds (insert d72)")
        elif down >= 2:
            verdicts["R1-1"] = "CEILING"; say("  VERDICT: WIDTH CEILING CONFIRMED at d48 — ladder goes DEPTH-LEAN (T-scaling at d<=48)")
        else:
            # tiebreak: packing plane (N, rbar) — frontier-interior at d64 = ceiling
            p1 = [(vh[("A4", s)]["N"], vh[("A4", s)]["rbar"]) for s in (0, 1, 2) if vh[("A4", s)]]
            p0 = [(vh0[("A4", s)]["N"], vh0[("A4", s)]["rbar"]) for s in (0, 1, 2) if vh0[("A4", s)]]
            if p1 and p0:
                N1, r1 = np.mean([p[0] for p in p1]), np.mean([p[1] for p in p1])
                N0, r0 = np.mean([p[0] for p in p0]), np.mean([p[1] for p in p0])
                say(f'  tiebreak (N, rbar): d64 ({N1:.1f}, {r1:.3f}) vs d48 ({N0:.1f}, {r0:.3f})')
                if N0 >= N1 and r0 >= r1 and (N0 > N1 or r0 > r1):
                    verdicts["R1-1"] = "CEILING(tiebreak)"; say("  VERDICT: INDETERMINATE on the pairs; d64 is FRONTIER-INTERIOR in (N, rbar) -> ceiling (tiebreak) — report as such")
                elif N1 >= N0 and r1 >= r0 and (N1 > N0 or r1 > r0):
                    verdicts["R1-1"] = "NOISE(tiebreak)"; say("  VERDICT: INDETERMINATE on the pairs; d64 ADVANCES the (N, rbar) frontier -> noise (tiebreak) — report as such")
                else:
                    verdicts["R1-1"] = "INDETERMINATE"; say("  VERDICT: INDETERMINATE — pairs mixed and (N, rbar) non-dominated; reported, no ladder decision from this rung alone")
            else:
                verdicts["R1-1"] = "INDETERMINATE"; say("  VERDICT: INDETERMINATE (tiebreak inputs missing)")

    # ---------------- R1-2 ----------------
    say(); say("R1-2 (H-30 restated) — priced d64/T6 eta ~.24 (.20-.28); plain within 15% of priced")
    e4 = [sc[("A4", s)]["eta"] for s in ARM_SEEDS["A4"] if sc[("A4", s)]]
    e5 = [sc[("A5", s)]["eta"] for s in ARM_SEEDS["A5"] if sc[("A5", s)]]
    e3 = [sc[("A3", s)]["eta"] for s in ARM_SEEDS["A3"] if sc[("A3", s)]]
    for a, seeds in ARM_SEEDS.items():
        for s in seeds:
            if sc[(a, s)]: say(f'  {tag(a,s):6s} eta {sc[(a,s)]["eta"]:.3f}')
    if not e4 or not e3:
        say("  (priced or plain eta missing — no verdict)"); verdicts["R1-2"] = "NO-DATA"
    else:
        pm = float(np.mean(e4)); pl = float(np.mean(e3)); rel = abs(pl - pm) / pm
        say(f'  priced A4 mean {pm:.3f} (n={len(e4)}){f"; A5 {e5[0]:.3f}" if e5 else ""}; plain A3 mean {pl:.3f} (n={len(e3)}); |plain-priced|/priced = {rel:.2f}')
        in_band = .20 <= pm <= .28
        if in_band and rel <= .15:
            verdicts["R1-2"] = "HOLDS"; say("  VERDICT: H-30 restated HOLDS (priced in .20-.28 AND plain within 15%)")
        elif rel > .25:
            verdicts["R1-2"] = "RESTATE"; say("  VERDICT: plain differs from priced by >25% — the price effect does NOT vanish with width; H-30 must be restated again")
        else:
            verdicts["R1-2"] = "STRAIN"; say(f'  VERDICT: STRAIN — priced {"in" if in_band else "OUT of"} band, plain gap {rel:.2f} (between .15 and .25): reported, no clean call')

    # ---------------- R1-3 ----------------
    say(); say("R1-3 (Law-4 at d64) — A4 (n=3) vs A3 (n=2) on rg-96 retention: priced > plain in >=4/6")
    comp = []
    for s4 in ARM_SEEDS["A4"]:
        for s3 in ARM_SEEDS["A3"]:
            g4, g3 = rg[("A4", s4)], rg[("A3", s3)]
            if not (g4 and g3): continue
            b, c, p = mcnemar(g4["ret"], g3["ret"]); comp.append(g4["S0"] > g3["S0"])
            say(f'  A4s{s4} vs A3s{s3}: rg96_ret {g4["S0"]} vs {g3["S0"]} (S.2 {g4["S2"]} vs {g3["S2"]}; flips {b}/{c}, p={p:.3f})')
    if not comp:
        say("  (no comparisons — no verdict)"); verdicts["R1-3"] = "NO-DATA"
    else:
        say(f'  priced > plain in {sum(comp)}/{len(comp)}')
        if len(comp) >= 4 and sum(comp) >= 4:
            verdicts["R1-3"] = "EXTENDS"; say("  VERDICT: Law-4 EXTENDS to d64 (transfer dividend seeded at width)")
        elif len(comp) < 4:
            verdicts["R1-3"] = "UNDERPOWERED"; say("  VERDICT: fewer than 4 comparisons available — underpowered, reported only")
        else:
            verdicts["R1-3"] = "FAILS"; say("  VERDICT: Law-4 does NOT extend to d64 on this evidence (scope it to d<=48)")

    # ---------------- R1-4 ----------------
    say(); say("R1-4 (NI attribution) — A5 global+NI (n=1) vs A4 floors+NI (n=3) on rg-96 retention [directional]")
    r4 = [rg[("A4", s)]["S0"] for s in ARM_SEEDS["A4"] if rg[("A4", s)]]
    r5 = [rg[("A5", s)]["S0"] for s in ARM_SEEDS["A5"] if rg[("A5", s)]]
    if not r4 or not r5:
        say("  (A5 or A4 rg-96 missing — no verdict)"); verdicts["R1-4"] = "NO-DATA"
    else:
        lo, hi = min(r4), max(r4); spread = hi - lo; margin = max(3, spread)
        say(f'  A4 band [{lo}, {hi}] (spread {spread}); A5 {r5[0]}')
        if lo <= r5[0] <= hi or abs(r5[0] - np.mean(r4)) <= margin:
            verdicts["R1-4"] = "NI-ALONE"; say("  VERDICT (directional, n=1): A5 within the A4 band — NI alone reproduces A4; floors dispensable at d64 too")
        elif r5[0] < lo - margin:
            verdicts["R1-4"] = "FLOORS-MATTER"; say("  VERDICT (directional, n=1): A5 well below the A4 band — floors matter WITH NI (interaction)")
        else:
            verdicts["R1-4"] = "A5-ABOVE"; say("  VERDICT (directional, n=1): A5 above the A4 band — floors cost transfer with NI; needs seeds before it carries weight")

    # ---------------- R1-5 ----------------
    say(); say("R1-5 (A4 seed-invariance) — rg-96 retention spread across A4's seeds (rung 0: {34,35,34}, spread 1)")
    if len(r4) >= 3:
        spread = max(r4) - min(r4)
        say(f'  A4 rg96_ret {sorted(r4)} spread {spread}')
        if spread <= 3: verdicts["R1-5"] = "REPLICATES"; say("  VERDICT: <=3 — the stabilizer claim REPLICATES at d64")
        elif spread > 8: verdicts["R1-5"] = "LUCK"; say("  VERDICT: >8 — rung 0's spread-1 was luck; NI's variance-collapse claim is struck")
        else: verdicts["R1-5"] = "INTERMEDIATE"; say("  VERDICT: intermediate (4-8) — weaker than rung 0, not refuted; report the number")
    else:
        say(f"  (need 3 A4 seeds; have {len(r4)} — no verdict)"); verdicts["R1-5"] = "NO-DATA"

    # ---------------- R1-6 ----------------
    say(); say("R1-6 (throat / profile) — d64 A4 I_med vs the curve (d48 A4 anchor; wave-2 d64 459-432); knee distance <=~.10 priced, plain on FREE")
    i4 = [vh[("A4", s)]["I"] for s in ARM_SEEDS["A4"] if vh[("A4", s)]]
    i40 = [vh0[("A4", s)]["I"] for s in (0, 1, 2) if vh0[("A4", s)]]
    if i4 and i40:
        m1, m0 = float(np.mean(i4)), float(np.mean(i40))
        say(f'  I_med: d64 A4 {m1:.0f} (n={len(i4)}: {[round(x) for x in i4]}) vs d48 A4 anchor {m0:.0f}; wave-2 d64 priced band 432-459 (no floors, no NI)')
        say(f'  -> throat {"DECLINES" if m1 < m0 else "does NOT decline"} with width at matched T/budget-per-d ({m1-m0:+.0f} nats)')
    prof_ok = True; n_prof = 0
    for a, seeds in ARM_SEEDS.items():
        for s in seeds:
            v = vh[(a, s)]
            if not v: continue
            n_prof += 1
            dk = float(np.abs(v["prof"] - KNEE).sum()); df = float(np.abs(v["prof"] - FREE).sum())
            cls = "knee" if dk < df else "free"
            expect = "free" if a == "A3" else "knee"
            hit = (cls == expect) and (a == "A3" or dk <= .12)
            prof_ok &= hit
            say(f'  {tag(a,s):6s} d(knee) {dk:.3f}  d(free) {df:.3f}  -> {cls}-class  (expected {expect}) {"OK" if hit else "MISS"}')
    if n_prof:
        verdicts["R1-6"] = "PROFILES-HOLD" if prof_ok else "PROFILE-MISS"
        say(f'  VERDICT: H-34 two-profile structure {"HOLDS at d64 (second seeded test)" if prof_ok else "MISSES on >=1 cell — report which; the profile picture is task-distribution-dependent (S-port) so a miss is informative"}')
    else:
        verdicts["R1-6"] = "NO-DATA"

    say(); say("SUMMARY: " + "  ".join(f"{k}={v}" for k, v in verdicts.items()))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(LINES) + "\n")
    say(); say(f"artifact -> {OUT}")
    return verdicts


# ---------------- synthetic ground-truth self-test ----------------
def _write_cell(root, rt, arm, seed, ret_vh, s4_vh, ret_rg, s2_rg, eta, d, step, cfg_extra, prof):
    """Fabricate one cell: 48 vh tasks x 3 queries, 2x48 rg tasks x 3 queries."""
    rng = np.random.default_rng(hash((rt[0], arm, seed)) % 2**32)
    def rows(n_tasks, n_ret, n_s2, s4n=None, name="t"):
        out = []; k = 0; retc = 0; s2c = 0; s4c = 0
        for t in range(n_tasks):
            qs = []
            for qi in range(3):
                r = 1 if retc < n_ret else 0; retc += r
                s2 = 1 if (r and s2c < n_s2) else 0; s2c += s2
                s4 = 1 if (s2 and s4n is not None and s4c < s4n) else 0; s4c += s4
                I_s = list(np.array(prof) * 500.0)
                qs.append(dict(gt_retention=r, q_ladder={"0.05": r, "0.1": r, "0.2": s2, "0.4": s4}, exact_T=r, I_s=I_s))
            out.append(dict(task=f"{name}{t}", queries=qs))
        return out
    def dump(dirn, rws):
        d_ = root / dirn; d_.mkdir(parents=True, exist_ok=True)
        (d_ / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rws))
    dump(f"lad_{rt[0]}{arm}s{seed}", rows(48, ret_vh, ret_vh, s4_vh, "vh"))
    dump(f"ladrg_{rt[0]}{arm}s{seed}", rows(48, ret_rg // 2, s2_rg // 2, None, "rg"))
    dump(f"ladrgb_{rt[0]}{arm}s{seed}", rows(48, ret_rg - ret_rg // 2, s2_rg - s2_rg // 2, None, "rb"))
    pdir = root / f"{rt[1]}{arm}s{seed}"; pdir.mkdir(parents=True, exist_ok=True)
    logit = math.log(eta / (1 - eta))
    base = dict(d=d, T=6, beta_flux=0.0, beta_flux_nl=0.0, flux_floors="", ni_sigma=0.0); base.update(cfg_extra)
    (pdir / "ckpt_latest.pkl").write_bytes(pickle.dumps(dict(state=dict(model=dict(eq=dict(eta=logit))), step=step, config=base)))


def selftest():
    import tempfile, contextlib, io
    global RUNS, OUT
    checks = []
    A4 = dict(beta_flux=3e-5, beta_flux_nl=1e-5, flux_floors="350,75,50,15,30", ni_sigma=0.01)
    A5 = dict(beta_flux=3e-5, beta_flux_nl=1e-5, flux_floors="", ni_sigma=0.01)
    A3 = {}
    def build(root, d64_up=True, mixed=False, plain_eta=.24, a5=40, a4_rg=(40, 41, 40), excl=False, s4_d64=(25, 25, 25)):
        # anchor d48 A4: rg ret 34,35,34 S2 20; A3 anchor 16
        for s, r in zip((0, 1, 2), (34, 35, 34)):
            _write_cell(root, R0, "A4", s, 47, 21, r, 20, .18, 48, 40000, A4, KNEE)
            _write_cell(root, R0, "A3", s, 28, 14, 16, 8, .17, 48, 40000, A3, FREE)
        for s, r in zip((0, 1, 2), a4_rg):
            rr = r if d64_up else r - 10
            s2 = 24 if d64_up else 15
            if mixed and s == 1: rr, s2 = 30, 25   # ret down, S2 up
            step = 53333 if not (excl and s == 2) else 40000
            _write_cell(root, R1, "A4", s, 50, s4_d64[s], rr, s2, .24, 64, step, A4, KNEE)
        for s in (0, 1):
            _write_cell(root, R1, "A3", s, 30, 14, 20, 9, plain_eta, 64, 53333, A3, FREE)
        _write_cell(root, R1, "A5", 0, 49, 24, a5, 23, .25, 64, 53333, A5, KNEE)
    def run(**kw):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); build(root, **kw)
            RUNS = root; OUT = root / "analysis" / "r1_verdict.txt"
            globals()["RUNS"] = RUNS; globals()["OUT"] = OUT
            with contextlib.redirect_stdout(io.StringIO()):
                return analyze()
    v = run(); checks.append(("noise: d64 up on both -> NOISE", v["R1-1"] == "NOISE"))
    checks.append(("eta holds", v["R1-2"] == "HOLDS")); checks.append(("law-4 extends 6/6", v["R1-3"] == "EXTENDS"))
    checks.append(("NI alone (A5 in band)", v["R1-4"] == "NI-ALONE")); checks.append(("spread 1 replicates", v["R1-5"] == "REPLICATES"))
    checks.append(("profiles hold", v["R1-6"] == "PROFILES-HOLD"))
    v = run(d64_up=False); checks.append(("ceiling: d64 down on both -> CEILING", v["R1-1"] == "CEILING"))
    v = run(mixed=True); checks.append(("mixed 2 up/1 mixed -> still NOISE (>=2/3)", v["R1-1"] == "NOISE"))
    v = run(mixed=True, a4_rg=(40, 41, 40), s4_d64=(25, 25, 25))
    v = run(plain_eta=.15); checks.append(("plain eta 37% off -> RESTATE", v["R1-2"] == "RESTATE"))
    v = run(plain_eta=.20); checks.append(("plain eta 17% off -> STRAIN", v["R1-2"] == "STRAIN"))
    v = run(a5=20); checks.append(("A5 well below band -> FLOORS-MATTER", v["R1-4"] == "FLOORS-MATTER"))
    v = run(a4_rg=(30, 40, 50)); checks.append(("spread 20 -> LUCK", v["R1-5"] == "LUCK"))
    v = run(a4_rg=(36, 40, 42)); checks.append(("spread 6 -> INTERMEDIATE", v["R1-5"] == "INTERMEDIATE"))
    v = run(excl=True); checks.append(("cell with wrong step EXCLUDED -> R1-5 NO-DATA (2 seeds)", v["R1-5"] == "NO-DATA"))
    n_ok = sum(1 for _, o in checks if o)
    for name, o in checks: print(f"  {'PASS' if o else 'FAIL'}  {name}")
    print(f"selftest: {n_ok}/{len(checks)}")
    return n_ok == len(checks)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    analyze()
