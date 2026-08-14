# Ledger: S-PORT ANALYZER (H-33) — written BEFORE cell-1's data existed,
# implementing decision rules S-R1..S-R4 exactly as registered 2026-08-14.
#   S-R1  retention ~ solve?   ratio R/S  (CONFIRM <2, KILL >=4)
#         HONESTY GUARD: R < .15 => "substrate too weak, test VOID" — a
#         model that learned nothing yields R=S=0, and that must never be
#         read as the dissociation collapsing.
#   S-R2  RI dividend          sudB vs sudA solve, paired McNemar
#         (CONFIRM p<.05 or >=+5pp, KILL sudB <= sudA)
#   S-R3  basin-existence enrichment on solve-FAILED puzzles
#         (CONFIRM <3x, KILL >=10x; POWER GUARD n>=5 per stratum)
#   S-R4  cross-domain law riders: trained eta (H-30 band .14-.19 at
#         priced-20k) and the spectral profile (H-34) — descriptive, n=1.
"""
  .venv/bin/python tools/analyze_sport.py
"""
from __future__ import annotations
import json
import math
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(__import__("os").environ.get("QHRRN_RUNS", ROOT / "runs"))
OUT = RUNS / "analysis" / "sport_verdict.txt"
ARMS = ["sudA", "sudB"]
KNEE = np.array([.69, .14, .085, .035, .048])
FREE = np.array([.76, .18, .045, .012, .003])
# ARC comparators (measured, ledger): retention 42/144, exact@sel ~8/144
ARC_R, ARC_S = 42 / 144, 8 / 144
LINES: list[str] = []


def say(s=""):
    LINES.append(s)
    print(s)


def load(arm):
    p = RUNS / f"sudprobe_{arm}" / "results.jsonl"
    if not p.exists():
        return None
    return [json.loads(l) for l in p.read_text().splitlines()]


def eta_of(arm):
    p = RUNS / f"sud_{arm}" / "ckpt_latest.pkl"
    if not p.exists():
        return None
    ck = pickle.loads(p.read_bytes())
    eq = ck["state"]["model"].get("eq")
    if not eq:
        return None
    return dict(eta=float(1 / (1 + np.exp(-float(np.asarray(eq["eta"]))))),
                step=int(ck["step"]))


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
    say("=" * 92)
    say("S-PORT CELL-1 VERDICT (H-33, the landscape-class law) — rules S-R1..S-R4")
    say("=" * 92)

    data = {a: load(a) for a in ARMS}
    for a in ARMS:
        if not data[a]:
            say(f"  (missing: sudprobe_{a})")
    if not any(data.values()):
        say("no data yet — nothing to conclude")
        return

    # Rows carry givens_level (the paired difficulty ladder). Pooling levels
    # would mix regimes — an easy level the substrate solves and a hard one it
    # cannot — so every readout is computed PER LEVEL, with the pooled row
    # kept only as a summary.
    def cell(rows):
        n = len(rows)
        return dict(
            n=n,
            solve=sum(r["solved"] for r in rows),
            ret=sum(r["gt_retention"] for r in rows),
            s2=sum(r["q_ladder"]["0.2"] for r in rows),
            viol=float(np.median([r["violations"] for r in rows])),
            cells=float(np.median([r["cells_correct"] for r in rows])),
            gk=sum(r["givens_kept"] for r in rows)
               / max(sum(r["givens_total"] for r in rows), 1),
            mi=sum(r["multi_init_hits"] > 0 for r in rows),
            solve_bits={r["task"]: bool(r["solved"]) for r in rows},
            rows=rows)

    say()
    say(f'{"arm":6s} {"givens":>7s} {"n":>3s} {"solve":>6s} {"reten":>6s} {"S(.2)":>6s} '
        f'{"viol_med":>8s} {"cells_med":>9s} {"givens_kept":>11s} {"mi_hit":>7s}')
    stats = {}          # (arm, level) -> cell ; (arm, "all") -> pooled
    levels = sorted({r.get("givens_level", 0) for a in ARMS if data[a]
                     for r in data[a]}, reverse=True)
    for a in ARMS:
        rows = data[a]
        if not rows:
            continue
        for lv in levels:
            sub = [r for r in rows if r.get("givens_level", 0) == lv]
            if not sub:
                continue
            c = cell(sub)
            stats[(a, lv)] = c
            say(f'{a:6s} {lv:>7d} {c["n"]:>3d} {c["solve"]:>6d} {c["ret"]:>6d} '
                f'{c["s2"]:>6d} {c["viol"]:>8.0f} {c["cells"]:>9.0f} '
                f'{c["gk"]:>10.1%} {c["mi"]:>7d}')
        stats[(a, "all")] = cell(rows)
    say(f'   (ARC comparators: retention 29%, solve ~5.5% of 144 pairs;')
    say(f'    more givens = easier — a solve falloff across levels locates the')
    say(f'    substrate\'s propagation limit, which is the ladder\'s purpose)')

    # ---- S-R1 ----
    say()
    say("S-R1 (H-33-i) — does the retention/solve dissociation COLLAPSE here?")
    for (a, lv) in sorted([k for k in stats if k[1] != "all"],
                          key=lambda k: (k[0], -k[1])):
        st = stats[(a, lv)]
        R, S = st["ret"] / st["n"], st["solve"] / st["n"]
        say(f'  --- {a} @ {lv} givens ---')
        say(f'      retention {R:.1%}, solve {S:.1%}  (ARC: {ARC_R:.1%} vs '
            f'{ARC_S:.1%}, ratio {ARC_R/ARC_S:.1f}x)')
        if R < 0.15:
            say("       VOID — substrate too weak (R<.15): the ratio is meaningless.")
            say("       ACTION per registration: deeper/longer arm (T-scaling first), "
                "NOT a landscape-class conclusion.")
        elif S == 0:
            say(f"       retention without ANY solve — dissociation PERSISTS (ratio infinite)")
        else:
            ratio = R / S
            verdict = ("CONFIRMED (collapses)" if ratio < 2 else
                       "KILL — dissociation is architectural" if ratio >= 4 else
                       "INDETERMINATE (2 <= ratio < 4)")
            say(f'       ratio {ratio:.2f}x -> {verdict}')

    # ---- S-R2 ----
    say()
    say("S-R2 (H-33-ii) — does RI pay on a single-attractor landscape?")
    keys = [lv for lv in levels if ("sudA", lv) in stats and ("sudB", lv) in stats]
    if ("sudA", "all") in stats and ("sudB", "all") in stats:
        keys = keys + ["all"]
    if not keys:
        say("  (needs both arms)")
    for lv in keys:
        A, B = stats[("sudA", lv)], stats[("sudB", lv)]
        b, c, p = mcnemar(A["solve_bits"], B["solve_bits"])
        dA, dB = A["solve"] / A["n"], B["solve"] / B["n"]
        tag = f"@{lv} givens" if lv != "all" else "POOLED"
        say(f'  {tag}: sudA {dA:.1%} vs sudB(+RI) {dB:.1%} '
            f'(flips A-only {b}, B-only {c}, p={p:.4f})')
        if dB <= dA:
            say("  KILL: RI does not pay here either -> the ARC null was OURS "
                "(architecture/protocol), not landscape-class. H-33 loses its mechanism.")
        elif p < .05 or (dB - dA) >= .05:
            say("  CONFIRMED: RI's dividend appears when the landscape has one attractor "
                "per instance — the EqR lever transfers exactly where H-33 says it should.")
        else:
            say("  DIRECTIONAL only — reported, not concluded.")

    # ---- S-R3 ----
    say()
    say("S-R3 (H-33-iii) — basin-existence enrichment on solve-FAILED puzzles")
    for (a, lv) in sorted([k for k in stats if k[1] != "all"],
                          key=lambda k: (k[0], -k[1])):
        failed = [r for r in stats[(a, lv)]["rows"] if not r["solved"]]
        with_b = [r for r in failed if r["gt_retention"]]
        without = [r for r in failed if not r["gt_retention"]]
        hw = sum(r["multi_init_hits"] for r in with_b)
        kw = sum(r["multi_init_k"] for r in with_b)
        ho = sum(r["multi_init_hits"] for r in without)
        ko = sum(r["multi_init_k"] for r in without)
        rw = hw / kw if kw else 0.0
        ro = ho / ko if ko else 0.0
        say(f'  {a} @ {lv}: failed {len(failed)} = {len(with_b)} with-basin / '
            f'{len(without)} without; hit rate {rw:.4f} vs {ro:.4f}')
        if len(with_b) < 5 or len(without) < 5:
            say("       UNDERPOWERED (need >=5 per stratum) — reported, not concluded.")
        elif ro == 0:
            say(f"       enrichment INFINITE (no capture without a basin) — ARC-like (31x+)")
        else:
            e = rw / ro
            say(f'       enrichment {e:.1f}x -> ' +
                ("CONFIRMED (collapses toward 1)" if e < 3 else
                 "KILL — conditioning is architecture-general" if e >= 10 else
                 "INTERMEDIATE"))

    # ---- S-R4 ----
    say()
    say("S-R4 — cross-domain law riders (descriptive, n=1 arm each)")
    for a in ARMS:
        e = eta_of(a)
        if e:
            band = ".14-.19 (priced-20k)"
            inb = 0.14 <= e["eta"] <= 0.19
            say(f'  {a} trained eta {e["eta"]:.3f} @ step {e["step"]} — H-30 band {band}: '
                f'{"IN (law is architectural)" if inb else "OUT (band may be ARC-specific)"}')
    for a in ARMS:
        rows = data.get(a)
        if not rows or not rows[0].get("I_s"):
            continue
        spec = np.median(np.array([r["I_s"] for r in rows if r.get("I_s")]), axis=0)
        prof = spec / spec.sum()
        say(f'  {a} spectrum {np.array2string(spec, precision=0)} '
            f'profile {np.array2string(prof, precision=3)}')
        say(f'       d(knee) {np.abs(prof-KNEE).sum():.3f}  d(free) {np.abs(prof-FREE).sum():.3f}'
            f'  throat {spec.sum():.0f} nats')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(LINES) + "\n")
    say()
    say(f"artifact -> {OUT}")


if __name__ == "__main__":
    main()
