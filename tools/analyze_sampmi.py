# Ledger: wave-2 READOUT 4 (2026-08-14) — RI multi-init coverage, the EqR
# core-claim test as registered 2026-08-12 (wave-2 v2):
#   samp --init random, T0=0, k=16 on Dri (RI-trained) vs C53 (matched d64,
#   no RI). Registered: RI-trained multi-init coverage jumps from the
#   thermal-only +2-3pp (cluster-M) toward the single-bulk bar ~65%.
#   KILL: coverage flat (Dri ~ C53) => path-dependence is architectural.
# Comparator constants (cluster-M, 2026-08-10, d16-era substrates, k=16):
#   deterministic-point coverage ~53.5/54.2%; thermal coverage@16 55.6/57.6%;
#   single-bulk 8-view pool bar ~65%; 32-member XA/EQ pools ~72%.
"""
  .venv/bin/python tools/analyze_sampmi.py
"""
from __future__ import annotations
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "analysis" / "sampmi_ri_20260814.txt"

LINES: list[str] = []


def say(s=""):
    LINES.append(s)
    print(s)


def load(tag):
    p = RUNS / f"samp_p13{tag}_mi" / "results.jsonl"
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    per = {}
    for r in rows:
        for qi, q in enumerate(r["queries"]):
            s = q["sigmas"]["0.0"]
            per[(r["task"], qi)] = dict(
                det=q["det_dist"], best=s["best_dist"],
                within=bool(s["within_radius"]), nd=s["n_distinct"],
                det_within=q["det_dist"] <= 0.2)
    return per


def main():
    say("=" * 92)
    say("READOUT 4 — RI MULTI-INIT COVERAGE (samp --init random, T0=0, k=16; eps=.2)")
    say("=" * 92)
    arms = {t: load(t) for t in ("Dri", "C53")}
    keys = set(arms["Dri"]) & set(arms["C53"])
    say(f"pairs: Dri {len(arms['Dri'])}, C53 {len(arms['C53'])}, common {len(keys)}")
    say()
    say(f'{"arm":6s} {"cov@16":>7s} {"det_cov":>8s} {"jump":>6s} {"med_best":>9s} '
        f'{"med_det":>8s} {"nd_med":>7s} {"nd_mean":>8s}')
    stats = {}
    for t, per in arms.items():
        cov = np.mean([v["within"] for v in per.values()])
        dcov = np.mean([v["det_within"] for v in per.values()])
        nd = [v["nd"] for v in per.values()]
        stats[t] = dict(cov=cov, dcov=dcov)
        say(f'{t:6s} {cov:>7.1%} {dcov:>8.1%} {cov - dcov:>+6.1%} '
            f'{np.median([v["best"] for v in per.values()]):>9.3f} '
            f'{np.median([v["det"] for v in per.values()]):>8.3f} '
            f'{np.median(nd):>7.1f} {np.mean(nd):>8.2f}')
    say()
    say("comparators (cluster-M, d16-era, k=16): det ~53.5-54.2% | thermal 55.6-57.6% "
        "(+2-3pp) | single-bulk bar ~65% | 32-member pools ~72%")

    # paired McNemar on within_radius, Dri vs C53
    a, b = arms["Dri"], arms["C53"]
    x = sum(1 for k in keys if a[k]["within"] and not b[k]["within"])
    y = sum(1 for k in keys if b[k]["within"] and not a[k]["within"])
    n = x + y
    p = (sum(math.comb(n, i) for i in range(0, min(x, y) + 1)) / 2 ** n * 2) if n else 1.0
    say(f"paired coverage flips Dri-only={x}, C53-only={y}, McNemar p={min(p,1):.3f}")

    jump_ri = stats["Dri"]["cov"] - stats["Dri"]["dcov"]
    jump_c = stats["C53"]["cov"] - stats["C53"]["dcov"]
    say()
    say(f"RI effect on multi-init jump: Dri {jump_ri:+.1%} vs C53 {jump_c:+.1%} "
        f"(delta {jump_ri - jump_c:+.1%}); absolute coverage delta "
        f"{stats['Dri']['cov'] - stats['C53']['cov']:+.1%}")
    say("REGISTERED KILL: coverage flat (Dri ~ C53) => path-dependence architectural.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(LINES) + "\n")
    say()
    say(f"artifact -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
