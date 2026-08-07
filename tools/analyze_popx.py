# Ledger: 2026-08-07 analysis — gate scoring for arms X/XA + the full [H-18]
# pairwise member-agreement matrix (member_query_preds ride every row).
"""
  python tools/analyze_popx.py runs/popx_X/results.jsonl runs/popx_XA/results.jsonl
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np


def load(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines()]


def outputs_exact(rows):
    s = set()
    for r in rows:
        for i, b in enumerate(r["per_pair_bits"]):
            if b[0] or b[1]:
                s.add((r["task"], i))
    return s


def h18_matrix(rows):
    """Pairwise member agreement on FAILED pairs, split within/cross bulk."""
    within, cross = [], []
    ident_w = ident_c = n_w = n_c = 0
    for r in rows:
        if "member_query_preds" not in r:
            continue
        metas = r["members"]
        preds = r["member_query_preds"]
        for qi, bits in enumerate(r["per_pair_bits"]):
            if bits[0] or bits[1]:
                continue
            grids = [np.asarray(p[qi], dtype=np.int16) for p in preds]
            for a, b in combinations(range(len(grids)), 2):
                ga, gb = grids[a], grids[b]
                same_bulk = metas[a]["bulk"] == metas[b]["bulk"]
                if ga.shape != gb.shape:
                    frac, ident = 0.0, False
                else:
                    frac = float((ga == gb).mean())
                    ident = bool(np.array_equal(ga, gb))
                if same_bulk:
                    within.append(frac); n_w += 1; ident_w += ident
                else:
                    cross.append(frac); n_c += 1; ident_c += ident
    return {"within_mean": float(np.mean(within)) if within else None,
            "cross_mean": float(np.mean(cross)) if cross else None,
            "within_identical": f"{ident_w}/{n_w}",
            "cross_identical": f"{ident_c}/{n_c}"}


def summarize(name, rows):
    pairs = sum(len(r["per_pair_bits"]) for r in rows)
    outs = outputs_exact(rows)
    solved_p2 = sum(r["solved_pass2"] for r in rows)
    solved_j2 = sum(r.get("solved_joint2", False) for r in rows)
    qual = [r["n_qualifiers"] for r in rows]
    walls = [r["wall_s"] for r in rows]
    print(f"== {name}: tasks={len(rows)} pass2={solved_p2} joint2={solved_j2} "
          f"outputs={len(outs)}/{pairs}")
    print(f"   qualifiers: median {int(np.median(qual))} zero {sum(q==0 for q in qual)}; "
          f"wall median {np.median(walls):.0f}s p90 {np.percentile(walls,90):.0f}s "
          f"total {sum(walls)/3600:.1f}h")
    h18 = h18_matrix(rows)
    if h18["within_mean"] is not None:
        print(f"   H-18 failed-pair agreement: within-bulk {h18['within_mean']:.3f} "
              f"(identical {h18['within_identical']}), cross-bulk "
              f"{h18['cross_mean']:.3f} (identical {h18['cross_identical']})")
    return outs


def main():
    base_union = None
    sets = {}
    for path in sys.argv[1:]:
        name = Path(path).parent.name
        rows = load(path)
        sets[name] = summarize(name, rows)
    if len(sets) > 1:
        names = list(sets)
        for a, b in combinations(names, 2):
            print(f"{a} ∩ {b} = {len(sets[a] & sets[b])}; union {len(sets[a] | sets[b])}")
        print("all-arm union:", len(set().union(*sets.values())))
    # vs the 2026-08-06 baselines (ledger): best single 12/144; six-way union 24/144
    print("baselines (08-06): best single protocol 12/144; six-protocol union 24/144")


if __name__ == "__main__":
    main()
