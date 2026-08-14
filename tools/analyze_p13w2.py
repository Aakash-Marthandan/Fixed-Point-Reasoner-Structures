# Ledger: wave-2 verdict analysis (2026-08-14) — the five registered readouts'
# battery half, computed from disk only per the run-execution standard:
#   READOUT 1  steps-law on C80  (rule: rg-radius >= .30 with rg_ret >= 9)
#   READOUT 2  A-decomposition   (Dcoup/Dfloor/Dri single-toggle vs C53)
#   READOUT 3  NI                (B bundle+NI vs pilot-A bundle; wrong-stable
#                                 half UNMEASURED — no e1e3 battery in wave-2,
#                                 absence named per the analysis checklist)
#   + the HV scale-regression flag (rides wave-2 per the 08-13 freethink)
#   + trained eq scalars from the six local ckpts (Dcoup expansive-drift check)
# Comparators: pilot batteries lad/ladrg_p13{A,C} (ckpt files lost to the
# 08-12 preemption; batteries survive), d48 record lad_p1248c40k.
"""
  .venv/bin/python tools/analyze_p13w2.py
"""
from __future__ import annotations
import json
import math
import pickle
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "analysis" / "p13w2_verdict_20260814.txt"

ARMS = ["C53", "C80", "Dcoup", "Dfloor", "Dri", "B"]          # wave-2 (local ckpts)
PILOT = ["C", "A"]                                             # batteries only
D48REF = "p1248c40k"                                           # record substrate

LINES: list[str] = []


def say(s=""):
    LINES.append(s)
    print(s)


def load(prefix, tag):
    p = RUNS / f"{prefix}_p13{tag}" / "results.jsonl" if tag not in (D48REF,) \
        else RUNS / f"{prefix}_{tag}" / "results.jsonl"
    if not p.exists():
        return None
    return [json.loads(l) for l in p.read_text().splitlines()]


def battery(rows):
    ret = ex = 0
    lad = {"0.05": 0, "0.1": 0, "0.2": 0, "0.4": 0}
    I = []
    per_pair = {}   # (task, qi) -> gt_retention  (for paired tests)
    for r in rows:
        for qi, q in enumerate(r["queries"]):
            ret += q["gt_retention"]; ex += q["exact_T"]
            for e in lad:
                lad[e] += q["q_ladder"][e]
            I.append(sum(q["I_s"]))
            per_pair[(r["task"], qi)] = bool(q["gt_retention"])
    return dict(ret=ret, s05=lad["0.05"], s1=lad["0.1"], s2=lad["0.2"],
                s4=lad["0.4"], ex=ex, rad=lad["0.2"] / max(ret, 1),
                I=float(np.median(I)), pairs=per_pair, n=sum(len(r["queries"]) for r in rows))


def mcnemar(pa, pb):
    """Discordant counts + two-sided exact binomial p on retention flips."""
    keys = set(pa) & set(pb)
    b = sum(1 for k in keys if pa[k] and not pb[k])
    c = sum(1 for k in keys if pb[k] and not pa[k])
    n = b + c
    if n == 0:
        return b, c, 1.0
    p = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n * 2
    return b, c, min(p, 1.0)


def fam(task):
    return re.sub(r"\d+$", "", task.replace("ca_", ""))


def scalars(arm):
    p = RUNS / f"pretrain13_{arm}" / "ckpt_latest.pkl"
    if not p.exists():
        return None
    ck = pickle.loads(p.read_bytes())
    eq = {k: float(np.asarray(v)) for k, v in ck["state"]["model"]["eq"].items()}
    sig = lambda x: 1 / (1 + np.exp(-x))
    out = {"step": int(ck["step"]), "eta": round(float(sig(eq["eta"])), 3)}
    if "alpha1" in eq:
        a1, a2 = float(sig(eq["alpha1"])), float(sig(eq["alpha2"]))
        out.update(a1=round(a1, 3), a2=round(a2, 3), a12=round(a1 + a2, 3))
    return out


def main():
    say("=" * 96)
    say("P13 WAVE-2 VERDICT ANALYSIS — 2026-08-14 (registered readouts 1-3 + HV flag)")
    say("=" * 96)

    # ---- integrity: row counts + identical task sets across every battery ----
    data = {}
    tasksets = {}
    for tag in ARMS + PILOT + [D48REF]:
        for pref in ("lad", "ladrg"):
            rows = load(pref, tag)
            if rows is None:
                if tag == D48REF and pref == "ladrg":
                    continue  # rg battery for the d48 ref lives elsewhere; not needed
                say(f"  !! MISSING: {pref}_{tag}")
                continue
            data[(pref, tag)] = battery(rows)
            tasksets[(pref, tag)] = frozenset(r["task"] for r in rows)
    vh_sets = {ts for (p, t), ts in tasksets.items() if p == "lad"}
    rg_sets = {ts for (p, t), ts in tasksets.items() if p == "ladrg"}
    say(f"integrity: {len(data)} batteries loaded; "
        f"val-hard task-set identical across arms: {len(vh_sets) == 1}; "
        f"rg task-set identical: {len(rg_sets) == 1}")
    for (p, t), b in data.items():
        if b["n"] != 144:
            say(f"  !! {p}_{t}: {b['n']} pairs (expected 144)")

    # ---- section 1: the full table ----
    say()
    say("SECTION 1 — battery table (val-hard | rg unseen-family), n=1 seed/cell, 144 pairs")
    say(f'{"arm":8s} {"ret":>4s} {"rad":>5s} {"S.4":>4s} {"ex":>4s} {"I_med":>7s} | '
        f'{"rg_ret":>6s} {"rg_rad":>6s} {"rg_S.2":>6s}')
    for tag in ["C", "A"] + ARMS + [D48REF]:
        v = data.get(("lad", tag)); g = data.get(("ladrg", tag))
        if not v:
            continue
        gs = f'{g["ret"]:>6d} {g["rad"]:>6.3f} {g["s2"]:>6d}' if g else " " * 20
        say(f'{tag:8s} {v["ret"]:>4d} {v["rad"]:>5.2f} {v["s4"]:>4d} {v["ex"]:>4d} '
            f'{v["I"]:>7.0f} | {gs}')

    # ---- section 2: trained scalars ----
    say()
    say("SECTION 2 — trained eq scalars (CPU ckpt reads; pilot A/C from ledger: "
        "C eta=.234, A a1+a2=1.049 a2=.286)")
    for arm in ARMS:
        s = scalars(arm)
        say(f"  {arm:7s} {s}")

    # ---- section 3: READOUT 1, steps-law on C80 ----
    say()
    say("SECTION 3 — READOUT 1: steps-law decider (C80 = d64 @ 80k)")
    g80 = data[("ladrg", "C80")]; g53 = data[("ladrg", "C53")]
    gC = data.get(("ladrg", "C"))
    say(f"  C   (53.3k, pilot): rg_ret {gC['ret']}, rg_rad {gC['rad']:.3f}" if gC else "")
    say(f"  C53 (53.3k, rerun): rg_ret {g53['ret']}, rg_rad {g53['rad']:.3f}")
    say(f"  C80 (80k):          rg_ret {g80['ret']}, rg_rad {g80['rad']:.3f}")
    fired = g80["rad"] >= 0.30 and g80["ret"] >= 9
    say(f"  RULE (registered 08-12): rg-radius >= .30 with rg_ret >= 9  ->  "
        f"{'FIRES: budget-limited confirmed, steps*(d) superlinear (quadratic branch)' if fired else 'DOES NOT FIRE: width-ceiling candidate at d64'}")

    # ---- section 4: READOUT 2, A-decomposition ----
    say()
    say("SECTION 4 — READOUT 2: A-decomposition (single-toggle vs C53; pilot A = full bundle)")
    base = data[("lad", "C53")]
    say(f'  {"cell":8s} {"ret":>4s} {"d_ret":>6s} {"rad":>5s} {"S.4":>4s} {"ex":>4s} | '
        f'McNemar ret vs C53 (b, c, p)')
    for tag in ["C", "Dcoup", "Dfloor", "Dri", "B", "A"]:
        v = data.get(("lad", tag))
        if not v:
            continue
        b, c, p = mcnemar(base["pairs"], v["pairs"])
        say(f'  {tag:8s} {v["ret"]:>4d} {v["ret"] - base["ret"]:>+6d} {v["rad"]:>5.2f} '
            f'{v["s4"]:>4d} {v["ex"]:>4d} | ({b:>2d}, {c:>2d}, p={p:.3f})')
    say("  (pilot-A signature to attribute: ret 18 = halved vs C's 32; rad .94 record)")

    # ---- section 5: READOUT 3, NI ----
    say()
    say("SECTION 5 — READOUT 3: NI (B = bundle+NI vs pilot A = bundle)")
    vA, vB = data.get(("lad", "A")), data[("lad", "B")]
    if vA:
        b, c, p = mcnemar(vA["pairs"], vB["pairs"])
        say(f'  A (bundle):    ret {vA["ret"]}, rad {vA["rad"]:.2f}, S.4 {vA["s4"]}, ex {vA["ex"]}')
        say(f'  B (bundle+NI): ret {vB["ret"]}, rad {vB["rad"]:.2f}, S.4 {vB["s4"]}, ex {vB["ex"]}')
        say(f'  McNemar retention A-vs-B: b={b} c={c} p={p:.3f}')
    say("  NOTE (absence named): the registered wrong-stable readout (probe_e1e3 class)")
    say("  was NOT measured in wave-2 — no e1e3 battery exists on any p13 arm; the")
    say("  spurious-attractor half of the NI claim remains open. Kill condition")
    say("  ('unchanged wrong-stable AND retention drop') is only half-adjudicable.")

    # ---- section 6: HV scale-regression flag ----
    say()
    say("SECTION 6 — HV scale-regression flag (per-family retention, HorizontalVertical)")
    for tag in ["C", "A"] + ARMS + [D48REF]:
        rows = load("lad", tag)
        if rows is None:
            continue
        by = {}
        for r in rows:
            f = fam(r["task"])
            got = sum(q["gt_retention"] for q in r["queries"])
            tot = len(r["queries"])
            a, b = by.get(f, (0, 0)); by[f] = (a + got, b + tot)
        hv = by.get("HorizontalVertical", (0, 0))
        say(f'  {tag:10s} HV retention: {hv[0]}/{hv[1]}')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(LINES) + "\n")
    say()
    say(f"artifact -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
