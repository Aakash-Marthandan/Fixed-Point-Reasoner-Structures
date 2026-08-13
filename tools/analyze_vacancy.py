# Ledger: family-level basin-vacancy analysis (2026-08-13, consolidation
# pass) — the Q addendum showed walls = ABSENT basins; this asks whether
# vacancy is a FAMILY property or substrate-idiosyncratic: per-family
# GT-retention fraction across every substrate with a val-hard ladder.
# Reads disk only; writes runs/analysis/family_vacancy_<date>.txt.
"""
  python tools/analyze_vacancy.py [tags...]   # default: the standing set
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

RUNS = Path(__file__).resolve().parents[1] / "runs"
OUT = RUNS / "analysis" / "family_vacancy_20260813.txt"
DEFAULT = ["p8", "p9c", "p9d", "p10c", "p1124c", "p1132c", "p1132cT6",
           "p1148c", "p1248c40k", "p13C", "p13A"]


def fam(t: str) -> str:
    return t[3:].rstrip("0123456789") if t.startswith("ca_") else t


def main():
    tags = sys.argv[1:] or DEFAULT
    table = {}
    present = []
    for tag in tags:
        p = RUNS / f"lad_{tag}" / "results.jsonl"
        if not p.exists():
            continue
        present.append(tag)
        for line in p.read_text().splitlines():
            r = json.loads(line)
            f = fam(r["task"])
            for q in r["queries"]:
                d = table.setdefault(f, {}).setdefault(tag, [0, 0])
                d[0] += q["gt_retention"]
                d[1] += 1
    lines = []
    say = lambda s="": (print(s), lines.append(s))
    say("=" * 100)
    say("FAMILY BASIN-VACANCY — GT-retention fraction per (family, substrate); "
        "vacancy = structurally low rows")
    say("=" * 100)
    say(f'{"family":>18s} ' + " ".join(f"{t[-6:]:>6s}" for t in present)
        + "   mean%")
    rows = []
    for f in sorted(table):
        fr = [table[f][t][0] / table[f][t][1] for t in present if t in table[f]]
        cells = [f"{100 * table[f][t][0] / table[f][t][1]:5.0f}%"
                 if t in table[f] else "    -" for t in present]
        rows.append((float(np.mean(fr)), f, cells))
    for m, f, cells in sorted(rows):
        say(f"{f:>18s} " + " ".join(f"{c:>6s}" for c in cells)
            + f"   {100 * m:4.0f}%")
    say()
    say("READING KEYS: rows stable-low across substrates = structural vacancy")
    say("  (basin-creation lever is representational, not scale/price/corpus);")
    say("  single-cell collapses (e.g. a family dying at one scale only) are")
    say("  scale-regression flags for that substrate.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"\n[written] {OUT}")


if __name__ == "__main__":
    main()
