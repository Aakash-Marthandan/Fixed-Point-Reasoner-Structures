# Ledger: T3 scale sweep analysis (phase-plan registration 2026-08-11) —
# predictions P1 (throat scale-invariance under pricing; free flux grows),
# P2 (radius grows with d under pricing), P3 (retention grows with d),
# P4 (family-transfer floor vs d). Scoring discipline (registered): the
# high-n instruments decide; solve counts are reported, never gating.
# Cells: d in {16,24,32} x {plain, priced}; d16 = pretrain10_a/c.
"""
  python tools/analyze_scale.py
"""
from __future__ import annotations

import json
from math import comb
from pathlib import Path

import numpy as np

RUNS = Path(__file__).resolve().parents[1] / "runs"
RUNGS = ["0", "0.05", "0.1", "0.2", "0.4"]

# (label, d, priced, ladder-tag, sample-tag)
CELLS = [
    ("d16 plain", 16, False, "p10a", "p10a"),
    ("d16 priced", 16, True, "p10c", "p10c"),
    ("d24 plain", 24, False, "p1124p", "p1124p"),
    ("d24 priced", 24, True, "p1124c", "p1124c"),
    ("d32 plain", 32, False, "p1132p", "p1132p"),
    ("d32 priced", 32, True, "p1132c", "p1132c"),
]
SETS = [("val-hard", "lad", "samp"), ("rg-48", "ladrg", "samprg"),
        ("rt-48", "ladrt", "samprt")]


def load(path: Path):
    if not path.exists():
        return None
    return [json.loads(l) for l in path.read_text().splitlines()]


def ladder(tag: str, prefix: str):
    rows = load(RUNS / f"{prefix}_{tag}" / "results.jsonl")
    if rows is None:
        return None
    n = 0
    counts = {e: 0 for e in RUNGS}
    exact = 0
    I_tot, A_tot, I_s = [], [], []
    per_pair = {}
    for r in rows:
        for qi, q in enumerate(r["queries"]):
            n += 1
            counts["0"] += q["gt_retention"]
            for e in RUNGS[1:]:
                counts[e] += q["q_ladder"][e]
            exact += q["exact_T"]
            I_tot.append(sum(q["I_s"]))
            A_tot.append(sum(q["A_s"]))
            I_s.append(q["I_s"])
            per_pair[(r["task"], qi)] = {e: bool(q["q_ladder"][e])
                                        for e in RUNGS[1:]}
            per_pair[(r["task"], qi)]["0"] = bool(q["gt_retention"])
    return dict(n=n, counts=counts, exact=exact,
                I_med=float(np.median(I_tot)), A_med=float(np.median(A_tot)),
                I_s_med=np.median(np.array(I_s), axis=0), per_pair=per_pair)


def sampling(tag: str, prefix: str):
    rows = load(RUNS / f"{prefix}_{tag}" / "results.jsonl")
    if rows is None:
        return None
    n = hops = cov = 0
    for r in rows:
        for q in r["queries"]:
            n += 1
            best = min([q["det_dist"]] + [v["best_dist"]
                                          for v in q["sigmas"].values()])
            cov += best <= 0.2
            if q["det_dist"] > 0 and any(v["best_dist"] == 0
                                         for v in q["sigmas"].values()):
                hops += 1
    return dict(n=n, hops=hops, cov=cov)


def mcnemar(a: dict, b: dict, rung: str):
    keys = set(a) & set(b)
    b10 = sum(1 for k in keys if a[k][rung] and not b[k][rung])
    b01 = sum(1 for k in keys if b[k][rung] and not a[k][rung])
    tot = b10 + b01
    if tot == 0:
        return b10, b01, 1.0
    p = sum(comb(tot, i) for i in range(max(b10, b01), tot + 1)) / 2 ** (tot - 1)
    return b10, b01, min(p, 1.0)


def main():
    print("=" * 82)
    print("T3 SCALE SWEEP — d x pricing (registered predictions P1-P4)")
    print("=" * 82)

    store = {}
    for set_name, lp, sp in SETS:
        print(f"\n### {set_name}")
        print(f'{"cell":12s} {"S(0)":>5s} {"S.05":>5s} {"S.1":>5s} {"S.2":>5s} '
              f'{"S.4":>5s} {"S.2/S0":>7s} {"exact":>6s} {"hops":>5s} '
              f'{"cov":>5s} {"I_med":>9s} {"A_med":>10s}')
        for label, d, priced, ltag, stag in CELLS:
            L = ladder(ltag, lp)
            S = sampling(stag, sp)
            if L is None:
                print(f"{label:12s} {'(pending)':>60s}")
                continue
            store[(label, set_name)] = (L, S)
            c = L["counts"]
            ratio = c["0.2"] / max(c["0"], 1)
            hops = f'{S["hops"]}' if S else "-"
            cov = f'{S["cov"]}' if S else "-"
            print(f'{label:12s} {c["0"]:5d} {c["0.05"]:5d} {c["0.1"]:5d} '
                  f'{c["0.2"]:5d} {c["0.4"]:5d} {ratio:7.2f} {L["exact"]:6d} '
                  f'{hops:>5s} {cov:>5s} {L["I_med"]:9.0f} {L["A_med"]:10.1f}')

    print("\n" + "=" * 82)
    print("P1 THROAT SCALE-INVARIANCE (priced I_med ~ d-invariant; free grows?)")
    print("=" * 82)
    for set_name, _, _ in SETS:
        row = []
        for label, d, priced, *_ in CELLS:
            k = (label, set_name)
            if k in store:
                row.append(f'{label}={store[k][0]["I_med"]:.0f}')
        if row:
            print(f"  {set_name:9s} " + "  ".join(row))

    print("\n" + "=" * 82)
    print("P2/P3 RADIUS & RETENTION vs d (val-hard; paired McNemar within pricing)")
    print("=" * 82)
    for priced in (False, True):
        tag = "priced" if priced else "plain"
        cells = [(lab, d) for lab, d, p, *_ in CELLS if p == priced
                 for k in [(lab, "val-hard")] if k in store]
        if len(cells) < 2:
            continue
        print(f"  -- {tag}:")
        for i in range(len(cells) - 1):
            (la, da), (lb, db) = cells[i], cells[i + 1]
            A = store[(la, "val-hard")][0]["per_pair"]
            B = store[(lb, "val-hard")][0]["per_pair"]
            for rung in ("0", "0.2"):
                b10, b01, p = mcnemar(B, A, rung)
                print(f"     d{db} vs d{da} @eps={rung:>4s}: "
                      f"{b10}/{b01} p={p:.3f}")

    print("\n" + "=" * 82)
    print("P4 FAMILY-TRANSFER FLOOR vs d (rt trained-family vs rg unseen-family)")
    print("=" * 82)
    print(f'{"cell":12s} {"rt S(0)":>8s} {"rg S(0)":>8s} {"gap":>6s} '
          f'{"rt hops":>8s} {"rg hops":>8s}')
    for label, *_ in [(c[0],) for c in CELLS]:
        krt, krg = (label, "rt-48"), (label, "rg-48")
        if krt in store and krg in store:
            rt, srt = store[krt]
            rg, srg = store[krg]
            print(f'{label:12s} {rt["counts"]["0"]:8d} {rg["counts"]["0"]:8d} '
                  f'{rt["counts"]["0"] - rg["counts"]["0"]:6d} '
                  f'{(srt or {}).get("hops", "-"):>8} '
                  f'{(srg or {}).get("hops", "-"):>8}')
    print("\n  [decision: rt >> rg => family NOVELTY binds; rt ~ rg => "
          "instance hardness / fit-limit binds]")


if __name__ == "__main__":
    main()
