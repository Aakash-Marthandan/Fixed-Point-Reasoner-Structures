# Ledger: 2026-08-07 pre-registered CompressARC comparison analysis.
# Paired per-task on the seeded N=120 ARC-1 public-eval sample; matched
# scoring = their convention (2 JOINT guesses, solved_joint2) primary;
# Kaggle-style per-output-2-attempts (solved_pass2) + per-output counts
# reported as sensitivity. Exact McNemar (binomial) two-sided.
"""
  python tools/analyze_comparison.py runs/comp_*/results.jsonl
"""
from __future__ import annotations

import json
import sys
from math import comb
from pathlib import Path


def mcnemar_exact_p(b, c):
    """Two-sided exact binomial on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main():
    rows = []
    for path in sys.argv[1:]:
        rows += [json.loads(l) for l in Path(path).read_text().splitlines()]
    seen = {}
    for r in rows:                       # last write wins on retries
        seen[r["task"]] = r
    rows = list(seen.values())

    ca = json.load(open("tools/compressarc_eval_solved.json"))
    ca_solved = set(ca["solved_pass2"])
    sample = json.load(open("tools/evalsample120.json"))["sample"]
    missing = [t for t in sample if t not in seen]
    if missing:
        print(f"WARNING: {len(missing)} sample tasks missing from our results: "
              f"{missing[:6]}{'…' if len(missing) > 6 else ''}")

    for conv, key in [("their convention (2 joint guesses)", "solved_joint2"),
                      ("Kaggle-style (per-output pass@2, all outputs)", "solved_pass2")]:
        ours = {t for t, r in seen.items() if r.get(key)}
        theirs = {t for t in seen if t in ca_solved}
        both = len(ours & theirs)
        only_us = len(ours - theirs)
        only_them = len(theirs - ours)
        neither = len(seen) - both - only_us - only_them
        p = mcnemar_exact_p(only_us, only_them)
        print(f"== {conv}")
        print(f"   us {len(ours)}/{len(seen)} ({len(ours)/max(len(seen),1):.1%})  "
              f"CompressARC {len(theirs)}/{len(seen)} ({len(theirs)/max(len(seen),1):.1%})")
        print(f"   paired: both {both}, only-us {only_us}, only-them {only_them}, "
              f"neither {neither}; exact McNemar two-sided p = {p:.2e}")
        if only_us:
            print(f"   only-us tasks: {sorted(ours - theirs)}")

    pairs = sum(len(r["per_pair_bits"]) for r in seen.values())
    outs = sum(1 for r in seen.values() for b in r["per_pair_bits"] if b[0] or b[1])
    print(f"== per-output (ours, descriptive): {outs}/{pairs} "
          f"({outs/max(pairs,1):.1%})")
    walls = [r["wall_s"] for r in seen.values()]
    if walls:
        import numpy as np
        print(f"== our per-task wall (v5e-1): median {np.median(walls):.0f}s "
              f"p90 {np.percentile(walls,90):.0f}s; total {sum(walls)/3600:.2f}h "
              f"for {len(walls)} tasks")
    print("== CompressARC published cost: ~20.6 min/task RTX 4070, 137.6h/400 tasks; "
          "76K params; no pretraining")


if __name__ == "__main__":
    main()
