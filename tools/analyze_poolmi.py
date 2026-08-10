# Ledger: cluster K (Research_Brainstorm 2026-08-10) — pool mutual-information
# predictor. From saved member predictions ONLY (no compute): coverage-within-
# eps curves per candidate pool, marginal gain per added bulk/view, and the
# correlation (island) correction = independence-predicted union coverage
# minus actual. Deliverable: a pre-run predictor of snap yield per candidate
# source, to decide portfolio purchases before spending lanes.
# Distance convention (matches the C.3''/C.3''' session analysis): fraction of
# mismatched cells on the GT extent when shapes match, else 1.0.
"""
  python tools/analyze_poolmi.py \
      --pools XA=runs/popx6_XA_results.jsonl,EQ=runs/popeq_merged.jsonl
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from qhrrn2 import grid as G

RADIUS = 0.2  # the measured pull radius (C.2 ladder: strong to 0.2)


def dist(pred, gt) -> float:
    p, g = np.asarray(pred), np.asarray(gt)
    if p.shape != g.shape:
        return 1.0
    return float((p != g).mean())


def load_pool(path: str):
    """-> {task: {"preds": preds[m][q], "meta": member meta list}}"""
    out = {}
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        out[r["task"]] = {"preds": r["member_query_preds"],
                          "meta": r["members"]}
    return out


def pair_table(pool):
    """Per (task, qi): gt, member dists (aligned with meta), member exact."""
    rows = []
    for tid, rec in sorted(pool.items()):
        eps = G.load_task(tid)
        for qi, ep in enumerate(eps):
            if ep.query_y is None:
                continue
            ds = [dist(m[qi], ep.query_y) for m in rec["preds"]]
            rows.append({"task": tid, "qi": qi, "d": np.array(ds),
                         "meta": rec["meta"]})
    return rows


def coverage(rows, idx, eps=RADIUS, subset="all"):
    """Fraction of pairs with min_{m in idx} d <= eps.
    subset='noexact': restrict to pairs where no member in idx is exact."""
    hit = tot = 0
    for r in rows:
        d = r["d"][idx]
        if subset == "noexact" and (d == 0).any():
            continue
        tot += 1
        hit += bool((d <= eps).any())
    return hit / max(tot, 1), tot


def by_bulk(rows):
    meta = rows[0]["meta"]
    bulks = sorted({m["bulk"] for m in meta})
    return {b: [i for i, m in enumerate(meta) if m["bulk"] == b] for b in bulks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", required=True,
                    help="name=path[,name=path...]")
    ap.add_argument("--eps", type=float, default=RADIUS)
    a = ap.parse_args()

    pools = {}
    for spec in a.pools.split(","):
        name, path = spec.split("=", 1)
        pools[name] = pair_table(load_pool(path))

    print("=" * 74)
    print(f"K: POOL-MI PREDICTOR (radius eps={a.eps}; pairs = query pairs w/ GT)")
    print("=" * 74)

    all_rows = {}
    for name, rows in pools.items():
        n_mem = len(rows[0]["d"])
        full = list(range(n_mem))
        best = np.array([r["d"].min() for r in rows])
        cov, n = coverage(rows, full, a.eps)
        cov_ne, n_ne = coverage(rows, full, a.eps, subset="noexact")
        exact_any = float(np.mean([(r["d"] == 0).any() for r in rows]))
        print(f"\nPOOL {name}: {n_mem} members, {n} pairs")
        print(f"  best-member distance: median {np.median(best):.3f} "
              f"mean {best.mean():.3f}")
        print(f"  any-member-exact: {exact_any:.1%}")
        print(f"  coverage within eps: {cov:.1%}")
        print(f"  coverage within eps | no exact member: {cov_ne:.1%} (n={n_ne})")
        all_rows[name] = rows

        # marginal value of bulks
        bb = by_bulk(rows)
        print(f"  -- per-bulk (8 views each):")
        for b, idx in bb.items():
            c1, _ = coverage(rows, idx, a.eps)
            print(f"     {b:18s} cov {c1:.1%}")
        print(f"  -- bulk-count curve (avg over subsets):")
        prev = None
        for k in range(1, len(bb) + 1):
            cs = [coverage(rows, sum((bb[b] for b in combo), []), a.eps)[0]
                  for combo in itertools.combinations(bb, k)]
            cur = np.mean(cs)
            gain = "" if prev is None else f"  (+{cur - prev:.1%})"
            print(f"     {k} bulk(s): {cur:.1%}{gain}")
            prev = cur
        # independence prediction vs actual (the island correction)
        per_bulk_cov = [coverage(rows, idx, a.eps)[0] for idx in bb.values()]
        indep = 1 - np.prod([1 - c for c in per_bulk_cov])
        actual, _ = coverage(rows, full, a.eps)
        print(f"  island correction: independent-union {indep:.1%} "
              f"vs actual {actual:.1%}  (correlation cost "
              f"{indep - actual:+.1%})")
        # view-count curve within one bulk (avg over bulks, nested views)
        print(f"  -- view-count curve (avg over bulks, views 1,2,4,8):")
        for k in (1, 2, 4, 8):
            cs = []
            for b, idx in bb.items():
                cs.append(coverage(rows, idx[:k], a.eps)[0])
            print(f"     {k} view(s): {np.mean(cs):.1%}")

    if len(all_rows) == 2:
        (nA, rA), (nB, rB) = all_rows.items()
        # align pairs by (task, qi)
        key = lambda r: (r["task"], r["qi"])
        mB = {key(r): r for r in rB}
        both = [(r, mB[key(r)]) for r in rA if key(r) in mB]
        hit = 0
        for ra, rb in both:
            if (ra["d"] <= a.eps).any() or (rb["d"] <= a.eps).any():
                hit += 1
        print("\n" + "=" * 74)
        print(f"CROSS-POOL UNION ({nA} u {nB}): coverage "
              f"{hit / max(len(both), 1):.1%} over {len(both)} shared pairs")
        onlyA = sum(1 for ra, rb in both
                    if (ra['d'] <= a.eps).any() and not (rb['d'] <= a.eps).any())
        onlyB = sum(1 for ra, rb in both
                    if (rb['d'] <= a.eps).any() and not (ra['d'] <= a.eps).any())
        print(f"  covered only by {nA}: {onlyA}   only by {nB}: {onlyB}")


if __name__ == "__main__":
    main()
