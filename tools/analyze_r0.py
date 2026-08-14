# Ledger: RUNG-0 ANALYZER (written 2026-08-14 BEFORE the campaign's data
# existed — pre-registration in its strongest form: the analysis code is
# blind to results). Implements decision rules R1-R6 from the rung-0 launch
# registration EXACTLY as written there:
#   R1 (H-32)  floors A1 vs global A2 on the DEEP TAIL S(.4)
#   R2 (NI/G1) A4 vs A1 retention non-inferiority AND wrong-stable drop
#   R3 (Law-4/G3) priced (A1+A2) vs plain A3 on rg-96
#   R4 (H-30)  trained eta bands (priced .22-.35, plain <.12)
#   R5 (H-34)  knee-profile distance (priced <= ~.10; A3 -> free profile)
#   R6         operating substrate for rung 1
# Statistics: per-seed pairing on identical task sets; McNemar exact
# binomial; the measurement law binds (no sub-10-count cross-run claims).
"""
  .venv/bin/python tools/analyze_r0.py
"""
from __future__ import annotations
import json
import math
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# QHRRN_RUNS lets the synthetic rule-logic test point the analyzer at a
# scratch tree; unset = the real runs/ dir. Never used in production paths.
RUNS = Path(__import__("os").environ.get("QHRRN_RUNS", ROOT / "runs"))
OUT = RUNS / "analysis" / "r0_verdict.txt"

ARMS = ["A1", "A2", "A3", "A4"]
SEEDS = [0, 1, 2]
KNEE = np.array([.69, .14, .085, .035, .048])
FREE = np.array([.76, .18, .045, .012, .003])
LINES: list[str] = []


def say(s=""):
    LINES.append(s)
    print(s)


def tag(arm, seed):
    return f"{arm}s{seed}"


def battery(prefix, arm, seed):
    p = RUNS / f"{prefix}_p13f{tag(arm, seed)}" / "results.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    S = {e: 0 for e in ("0", "0.05", "0.1", "0.2", "0.4")}
    ex = 0
    I = []
    ret_pairs, s4_pairs = {}, {}
    for r in rows:
        for qi, q in enumerate(r["queries"]):
            k = (r["task"], qi)
            S["0"] += q["gt_retention"]
            for e in ("0.05", "0.1", "0.2", "0.4"):
                S[e] += q["q_ladder"][e]
            ex += q["exact_T"]
            I.append(sum(q["I_s"]))
            ret_pairs[k] = bool(q["gt_retention"])
            s4_pairs[k] = bool(q["q_ladder"]["0.4"])
    spec = np.median(np.array([q["I_s"] for r in rows for q in r["queries"]]), axis=0)
    return dict(n_tasks=len(rows), S=S, ex=ex, I=float(np.median(I)),
                ret=ret_pairs, s4=s4_pairs, spec=spec, prof=spec / spec.sum())


def rg96(arm, seed):
    """G4: rg-96 = frozen rg-48 (ladrg) U rb-48 (ladrgb), keys namespaced."""
    out = dict(ret={}, s2={}, S0=0, S2=0)
    for pref in ("ladrg", "ladrgb"):
        p = RUNS / f"{pref}_p13f{tag(arm, seed)}" / "results.jsonl"
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            r = json.loads(line)
            for qi, q in enumerate(r["queries"]):
                k = (pref, r["task"], qi)
                out["ret"][k] = bool(q["gt_retention"])
                out["s2"][k] = bool(q["q_ladder"]["0.2"])
                out["S0"] += q["gt_retention"]; out["S2"] += q["q_ladder"]["0.2"]
    return out if out["ret"] else None


def wrong_stable(arm, seed):
    """G1: converged AND limit not exact (the E3 contingency class)."""
    p = RUNS / f"e1e3_p13f{tag(arm, seed)}" / "results.jsonl"
    if not p.exists():
        return None
    ws = conv = tot = 0
    for line in p.read_text().splitlines():
        r = json.loads(line)
        for e3 in r["e3"]:
            tot += 1
            if e3["converged_at"] is not None:
                conv += 1
                ws += not e3["limit_exact"]
    return dict(ws=ws, conv=conv, tot=tot)


def scalars(arm, seed):
    p = RUNS / f"pretrain13f_{tag(arm, seed)}" / "ckpt_latest.pkl"
    if not p.exists():
        return None
    ck = pickle.loads(p.read_bytes())
    eq = ck["state"]["model"]["eq"]
    e = 1 / (1 + np.exp(-float(np.asarray(eq["eta"]))))
    return dict(eta=float(e), step=int(ck["step"]))


def mcnemar(pa, pb):
    keys = set(pa) & set(pb)
    b = sum(1 for k in keys if pa[k] and not pb[k])
    c = sum(1 for k in keys if pb[k] and not pa[k])
    n = b + c
    if n == 0:
        return b, c, 1.0
    p = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n * 2
    return b, c, min(p, 1.0)


def main():
    say("=" * 96)
    say("RUNG-0 VERDICT — pretrain-13-full v2 (d48/T6@40k, 4 arms x 3 seeds)")
    say("decision rules R1-R6 as registered 2026-08-14 (analyzer written pre-data)")
    say("=" * 96)

    vh, rg, ws, sc = {}, {}, {}, {}
    for a in ARMS:
        for s in SEEDS:
            vh[(a, s)] = battery("lad", a, s)
            rg[(a, s)] = rg96(a, s)
            ws[(a, s)] = wrong_stable(a, s)
            sc[(a, s)] = scalars(a, s)

    say()
    say("SECTION 1 — cells present + integrity")
    have = [(a, s) for a in ARMS for s in SEEDS if vh[(a, s)]]
    say(f"  val-hard cells: {len(have)}/12   rg-96 cells: {sum(1 for k in rg if rg[k])}/12"
        f"   e1e3 cells: {sum(1 for k in ws if ws[k])}/6 (A1+A4)")
    for k in [(a, s) for a in ARMS for s in SEEDS]:
        if vh[k] and vh[k]["n_tasks"] != 48:
            say(f"  !! {tag(*k)} val-hard {vh[k]['n_tasks']}/48 tasks")
        if rg[k] and len(rg[k]["ret"]) != 288:
            say(f"  !! {tag(*k)} rg-96 {len(rg[k]['ret'])}/288 pairs")

    say()
    say("SECTION 2 — the table (per cell)")
    say(f'  {"cell":6s} {"S0":>4s} {"S.2":>4s} {"S.4":>4s} {"ex":>3s} {"I_med":>6s} '
        f'{"eta":>5s} {"dknee":>6s} | {"rg96_ret":>8s} {"rg96_S2":>7s} | {"ws":>5s}')
    for a in ARMS:
        for s in SEEDS:
            v, g, w, x = vh[(a, s)], rg[(a, s)], ws[(a, s)], sc[(a, s)]
            if not v:
                continue
            dk = float(np.abs(v["prof"] - KNEE).sum())
            say(f'  {tag(a,s):6s} {v["S"]["0"]:>4d} {v["S"]["0.2"]:>4d} {v["S"]["0.4"]:>4d} '
                f'{v["ex"]:>3d} {v["I"]:>6.0f} {x["eta"] if x else 0:>5.3f} {dk:>6.3f} | '
                f'{g["S0"] if g else 0:>8d} {g["S2"] if g else 0:>7d} | '
                f'{(str(w["ws"]) + "/" + str(w["conv"])) if w else "-":>5s}')

    # ---------------- R1 ----------------
    say()
    say("R1 (H-32) — floors A1 vs global A2 on the DEEP TAIL S(.4)")
    wins = 0; pooled_a, pooled_b = {}, {}
    for s in SEEDS:
        va, vb = vh[("A1", s)], vh[("A2", s)]
        if not (va and vb):
            continue
        b, c, p = mcnemar(va["s4"], vb["s4"])
        wins += va["S"]["0.4"] > vb["S"]["0.4"]
        pooled_a.update({(s,) + k: v for k, v in va["s4"].items()})
        pooled_b.update({(s,) + k: v for k, v in vb["s4"].items()})
        say(f'  seed {s}: A1 S(.4) {va["S"]["0.4"]} vs A2 {vb["S"]["0.4"]}  '
            f'(flips {b}/{c}, p={p:.3f})')
    if pooled_a:
        b, c, p = mcnemar(pooled_a, pooled_b)
        say(f'  pooled: flips A1-only {b}, A2-only {c}, McNemar p={p:.4f}; '
            f'seed-pairs won by A1: {wins}/3')
        if wins >= 2 and p < .05:
            say("  VERDICT: two-sided S2 CONFIRMED — floors = ladder default")
        elif wins <= 1 and sum(vh[("A1", s)]["S"]["0.4"] < vh[("A2", s)]["S"]["0.4"]
                               for s in SEEDS if vh[("A1", s)] and vh[("A2", s)]) >= 2:
            say("  VERDICT: floors HARMFUL at T6 — investigate floor-vector provenance")
        else:
            say("  VERDICT: H-32 KILL FIRES — sub-floor nats not load-bearing; floors optional")

    # ---------------- R2 ----------------
    say()
    say("R2 (NI/G1) — A4 (floors+NI) vs A1 (floors): retention non-inferiority + wrong-stable")
    dr = []; ws_drop = 0; against = 0
    for s in SEEDS:
        va, v4 = vh[("A1", s)], vh[("A4", s)]
        if va and v4:
            b, c, p = mcnemar(va["ret"], v4["ret"])
            d = v4["S"]["0"] - va["S"]["0"]
            dr.append(d)
            against += (d < 0 and p < .05)
            say(f'  seed {s}: retention A1 {va["S"]["0"]} -> A4 {v4["S"]["0"]} ({d:+d}, p={p:.3f})')
        w1, w4 = ws[("A1", s)], ws[("A4", s)]
        if w1 and w4:
            ws_drop += w4["ws"] < w1["ws"]
            say(f'           wrong-stable A1 {w1["ws"]}/{w1["conv"]} -> A4 {w4["ws"]}/{w4["conv"]}')
    if dr:
        pooled_d = sum(dr)
        say(f'  pooled retention delta {pooled_d:+d}; seed-pairs with wrong-stable drop {ws_drop}/3')
        ni_in = pooled_d >= -5 and against == 0 and ws_drop >= 2
        say(f'  VERDICT: NI {"DEFAULT-IN" if ni_in else "OUT (stays an arm)"} '
            f'— non-inferior={pooled_d >= -5 and against == 0}, ws-drop={ws_drop >= 2}')

    # ---------------- R3 ----------------
    say()
    say("R3 (Law-4/G3) — priced (A1,A2) vs plain (A3) on rg-96 retention")
    per_seed = []
    for s in SEEDS:
        g3 = rg[("A3", s)]
        if not g3:
            continue
        for a in ("A1", "A2"):
            ga = rg[(a, s)]
            if not ga:
                continue
            b, c, p = mcnemar(ga["ret"], g3["ret"])
            per_seed.append(ga["S0"] > g3["S0"])
            say(f'  seed {s}: {a} rg96_ret {ga["S0"]} vs A3 {g3["S0"]} '
                f'(S.2 {ga["S2"]} vs {g3["S2"]}; flips {b}/{c}, p={p:.3f})')
    if per_seed:
        frac = sum(per_seed) / len(per_seed)
        say(f'  priced > plain in {sum(per_seed)}/{len(per_seed)} comparisons ({frac:.0%})')
        say(f'  VERDICT: Law-4 {"SEEDED AT THE FRONTIER — dividend claim stands" if frac >= 2/3 else "SCOPED-OR-RETRACTED (holds <=d32/20k only)"}')

    # ---------------- R4 / R5 ----------------
    say()
    say("R4 (H-30) eta bands — priced .22-.35, plain <.12")
    strain = 0; n_eta = 0
    for a in ARMS:
        for s in SEEDS:
            x = sc[(a, s)]
            if not x:
                continue
            n_eta += 1
            band = (.22, .35) if a != "A3" else (0.0, .12)
            ok = band[0] <= x["eta"] <= band[1]
            strain += not ok
            say(f'  {tag(a,s):6s} eta {x["eta"]:.3f} {"OK" if ok else "OUT-OF-BAND"}')
    if not n_eta:
        say("  (no ckpts read — no verdict)")   # never a verdict from no evidence
    else:
        say(f'  VERDICT: {"H-30 HOLDS" if strain == 0 else f"H-30 STRAIN ({strain}/{n_eta} cells out of band; >=3 = falsified)"}')

    say()
    say("R5 (H-34) knee-profile distance — priced <= ~.10; A3 should sit on the FREE profile")
    for a in ARMS:
        for s in SEEDS:
            v = vh[(a, s)]
            if not v:
                continue
            dk = float(np.abs(v["prof"] - KNEE).sum())
            df = float(np.abs(v["prof"] - FREE).sum())
            say(f'  {tag(a,s):6s} d(knee) {dk:.3f}  d(free) {df:.3f}  -> '
                f'{"knee" if dk < df else "free"}-class')

    # ---------------- R6 ----------------
    say()
    say("R6 — operating substrate for rung 1 (rank: rg96 ret+S2 -> vh S(.4) -> vh count)")
    rank = []
    for a in ARMS:
        rgr = [rg[(a, s)]["S0"] for s in SEEDS if rg[(a, s)]]
        rgs = [rg[(a, s)]["S2"] for s in SEEDS if rg[(a, s)]]
        s4 = [vh[(a, s)]["S"]["0.4"] for s in SEEDS if vh[(a, s)]]
        s0 = [vh[(a, s)]["S"]["0"] for s in SEEDS if vh[(a, s)]]
        if rgr and s4:
            rank.append((np.mean(rgr) + np.mean(rgs), np.mean(s4), np.mean(s0), a))
    for r in sorted(rank, reverse=True):
        say(f'  {r[3]}: rg96(ret+S2) {r[0]:.1f}  vh S(.4) {r[1]:.1f}  vh S0 {r[2]:.1f}')
    if rank:
        say(f'  OPERATING SUBSTRATE -> {sorted(rank, reverse=True)[0][3]} '
            f'(subject to R1/R2 gating above)')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(LINES) + "\n")
    say()
    say(f"artifact -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
