# Ledger: dev-30 stratified development set (Divergence_Analysis §7 roadmap;
# audit §16.2 R1/R2 emphasis). Selection rule: REAL training-split tasks only
# (R5: no generator contact), each classified by eye (rendered grids), family
# labels from the audit's taxonomy. The manifest is data; tools/measure.py
# --arc consumes it. dev-30 gate: Aug 31.
"""Dev-30 manifest + terminal renderer.

  .venv/bin/python tools/dev30.py render <task_id>   # inspect a candidate
  .venv/bin/python tools/dev30.py list               # show current manifest
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qhrrn2 import grid as G

# task_id -> (family, note). Families from audit §16.1. Entries are added ONLY
# after rendering the task and confirming the rule by eye — no memory-sourced
# labels. Target: 30 tasks, >=2 per family, R1/R2 families oversampled.
MANIFEST: dict[str, tuple[str, str]] = {
    # verified 2026-07-28 (rendered):
    "1e0a9b12": ("axial-gravity", "columns compact downward; spec §2.2c constructive example"),
}

_GLYPH = " 123456789▓"  # index = color; VOID -> ▓, 0 -> space


def render(task_id: str):
    eps = G.load_task(task_id)
    ep = eps[0]
    print(f"task {task_id}: {len(ep.support)} support pairs + {len(eps)} test")
    for i, (x, y) in enumerate(list(ep.support) + [(ep.query_x, ep.query_y)]):
        tag = "query" if i == len(ep.support) else f"sup{i}"
        print(f"--- {tag}: {x.shape} -> {None if y is None else y.shape}")
        xs = ["".join(_GLYPH[v] for v in row) for row in x]
        ys = [] if y is None else ["".join(_GLYPH[v] for v in row) for row in y]
        h = max(len(xs), len(ys))
        wx = len(xs[0]) if xs else 0
        for r in range(h):
            left = xs[r] if r < len(xs) else " " * wx
            right = ys[r] if r < len(ys) else ""
            print(f"  |{left}|   |{right}|")


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "render":
        render(sys.argv[2])
    elif len(sys.argv) >= 2 and sys.argv[1] == "list":
        print(f"{len(MANIFEST)} tasks in manifest:")
        for tid, (fam, note) in sorted(MANIFEST.items(), key=lambda kv: kv[1][0]):
            print(f"  {tid}  {fam:<18} {note}")
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
