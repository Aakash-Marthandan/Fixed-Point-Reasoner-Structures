# Ledger: the P11 seeded-grid consolidated analysis (2026-08-12) — the
# campaign's final table: every cell with per-seed rows + seed-means, the
# four claim summaries (throat invariance; transfer dividend; d32 radius
# peak; count-vs-radius across axes), computed from disk only.
"""
  python tools/analyze_grid.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

RUNS = Path(__file__).resolve().parents[1] / "runs"

# (cell label, [seed tags]) — tags name lad_/ladrg_/ladrt_/samp_ dirs
GRID = [
    ("d16-plain",  ["p10a"]),
    ("d16-priced", ["p10c"]),
    ("d24-plain",  ["p1124p"]),
    ("d24-priced", ["p1124c", "p1124cs1", "p1124cs2"]),
    ("d32-plain",  ["p1132p", "p1132ps1"]),
    ("d32-priced", ["p1132c", "p1132cs1", "p1132cs2"]),
    ("d32c-T6",    ["p1132cT6", "p1132cT6s1"]),
    ("d48-plain",  ["p1148p", "p1148ps1"]),
    ("d48-priced", ["p1148c", "p1148cs1"]),
]


def cell(tag, prefix="lad"):
    p = RUNS / f"{prefix}_{tag}" / "results.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    ret = ex = 0
    lad = {"0.05": 0, "0.1": 0, "0.2": 0, "0.4": 0}
    I = []
    for r in rows:
        for q in r["queries"]:
            ret += q["gt_retention"]; ex += q["exact_T"]
            for e in lad: lad[e] += q["q_ladder"][e]
            I.append(sum(q["I_s"]))
    return dict(ret=ret, s2=lad["0.2"], s4=lad["0.4"], ex=ex,
                rad=lad["0.2"] / max(ret, 1), I=float(np.median(I)))


def agg(tags, prefix="lad"):
    cells = [c for c in (cell(t, prefix) for t in tags) if c]
    if not cells:
        return None, 0
    keys = cells[0].keys()
    return {k: float(np.mean([c[k] for c in cells])) for k in keys}, len(cells)


def main():
    print("=" * 88)
    print("P11 SEEDED GRID — seed-means (n = seeds on disk); val-hard | rg unseen | rt trained")
    print("=" * 88)
    print(f'{"cell":11s} {"n":>2s} {"ret":>5s} {"rad":>5s} {"S.4":>5s} {"ex":>5s} '
          f'{"I_med":>8s} | {"rg_ret":>6s} {"rg_rad":>6s} | {"rt_ret":>6s}')
    for label, tags in GRID:
        vh, n = agg(tags)
        rg, _ = agg(tags, "ladrg")
        rt, _ = agg(tags, "ladrt")
        if vh is None:
            print(f"{label:11s}  (pending)")
            continue
        rg_s = f'{rg["ret"]:6.1f} {rg["rad"]:6.2f}' if rg else "   -      -"
        rt_s = f'{rt["ret"]:6.1f}' if rt else "     -"
        print(f'{label:11s} {n:2d} {vh["ret"]:5.1f} {vh["rad"]:5.2f} {vh["s4"]:5.1f} '
              f'{vh["ex"]:5.1f} {vh["I"]:8.0f} | {rg_s} | {rt_s}')
    print()
    print("CLAIM SUMMARIES:")
    pr = {lab: agg(tags)[0] for lab, tags in GRID if "priced" in lab and agg(tags)[0]}
    print("  1. Throat invariance (priced I_med across scale): " +
          "  ".join(f'{k.split("-")[0]}={v["I"]:.0f}' for k, v in pr.items()))
    rg_pr = {lab: agg(tags, "ladrg")[0] for lab, tags in GRID if agg(tags, "ladrg")[0]}
    print("  2. rg radius by cell: " +
          "  ".join(f'{k}={v["rad"]:.2f}' for k, v in rg_pr.items()))


if __name__ == "__main__":
    main()
