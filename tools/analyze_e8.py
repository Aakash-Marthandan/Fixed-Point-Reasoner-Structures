# Ledger: E8 registered analysis (2026-08-08 NIGHT lock + B4 amendment).
# Encodes the decision rules: support-retention floor claim per arm, query
# retention transport claim, exact@T vs the E1/E3 baseline, the B4-vs-B1-3
# interpretation grid; integrity gates per the run-execution standard.
"""
  python tools/analyze_e8.py runs/e8_d16_b1/results.jsonl [...more arms...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

BASELINE = {"q_ret": (17, 288), "exact_T": (18, 288),
            "note": "E1/E3 deployed-fit rows, both bulks pooled "
                    "(d16 8/144+ret 9? use per-bulk when comparing)"}
# Per-bulk baselines from the E1/E3 verdicts (ledger 2026-08-08):
BASE_BULK = {"d16": {"q_ret": (8, 144), "exact_T": (8, 144)},
             "d24t64": {"q_ret": (11, 144), "exact_T": (10, 144)}}


def main():
    expected = set(json.load(open("tools/valhard.json"))["valhard"])
    print("arm                     n    looRet   ladder(.05/.1/.2/.4)      "
          "qRet        exact@T     medHq  n_dist>1")
    for path in sys.argv[1:]:
        name = Path(path).parent.name
        rows = [json.loads(l) for l in Path(path).read_text().splitlines()]
        rows = list({r["task"]: r for r in rows}.values())
        missing = expected - {r["task"] for r in rows}
        if missing:
            print(f"  WARNING {name}: {len(missing)} ABSENT: "
                  f"{sorted(missing)[:5]}{'…' if len(missing) > 5 else ''}")
        bulk = "d24t64" if "d24t64" in name else "d16"
        loo = [r["loo_retention_gt"] for r in rows]
        lad = [np.mean([r["loo_ladder"][e] for r in rows])
               for e in (".05", "0.05") if any(e in r["loo_ladder"] for r in rows)]
        ladder = {e: np.mean([r["loo_ladder"].get(e, r["loo_ladder"].get(str(float(e)), False))
                              for r in rows])
                  for e in ("0.05", "0.1", "0.2", "0.4")}
        qr = [q["gt_retention"] for r in rows for q in r["queries"]
              if q["gt_retention"] is not None]
        qe = [q["exact_T"] for r in rows for q in r["queries"]]
        hq = [q["Hq"] for r in rows for q in r["queries"]]
        nd = [q["n_distinct"] > 1 for r in rows for q in r["queries"]]
        bq, be = BASE_BULK[bulk]["q_ret"], BASE_BULK[bulk]["exact_T"]
        print(f"{name:22s} {len(rows):3d}  {sum(loo):2d}/{len(loo):2d}   "
              f"{'/'.join(f'{ladder[e]:.0%}' for e in ('0.05','0.1','0.2','0.4')):20s}  "
              f"{sum(qr):3d}/{len(qr):3d} (b {bq[0]}/{bq[1]})  "
              f"{sum(qe):3d}/{len(qe):3d} (b {be[0]}/{be[1]})  "
              f"{np.median(hq):.3f}  {sum(nd)}/{len(nd)}")
    print("\ninterpretation grid (ledger): B4 looRet high + B1-3 low ⇒ "
          "mechanisms work, capacity was the block; B4 low too ⇒ [H-23] "
          "implementation suspect — debug before further spend.")


if __name__ == "__main__":
    main()
