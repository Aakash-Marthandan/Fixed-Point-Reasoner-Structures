# Ledger: dev-30 stratified development set (Phase-2 roadmap;
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
    # verified 2026-07-30 (rendered):
    "0ca9ddb6": ("object-conditional", "decorate each dot with its color-specific surround (2->4-corners, 1->7-plus); R1 family"),
    "3aa6fb7a": ("object-relational", "place a 1 in the concave corner of every 8-L-shape"),
    "25ff71a9": ("translation", "shift the occupied row down by one; real twin of the constructed translate family"),
    "3631a71a": ("symmetry-completion", "restore 9-occluded patches from the wallpaper pattern's distant mirror content; FIRST NL-CLASS CANDIDATE (R2 family)"),
    "4258a5f9": ("object-decoration", "ring every 5 with 3x3 of 1s; local"),
    "496994bd": ("reflection-copy", "mirror the top color band to the far bottom edge; correspondence/NL candidate"),
    "5521c0d9": ("object-relative-motion", "each block moves up by its own height; per-object measurement drives action (R1-adjacent)"),
    "23581191": ("axial-projection", "each dot casts its color as full row+column; intersections recolored 2 (Amendment C family)"),
    "1cf80156": ("crop-to-content", "output = tight bounding-box crop of the shape; size from content extent"),
    "445eab21": ("object-argmax", "output 2x2 in the LARGER rectangle's color; R1 canonical select-largest"),
    "6150a2bd": ("rot180", "output = 180-degree rotation of the input; D4 geometry"),
    "8be77c9e": ("mirror-extend", "output = input stacked atop its vertical mirror; size x2 transform"),
    "d631b094": ("counting", "output = 1xN bar of the object color, N = number of colored cells"),
    "0d3d703e": ("color-mapping", "fixed color bijection (3<->4, 1<->5, 2<->6, 8<->9) applied per column; color-relational"),
    "6455b5f5": ("region-argmax-fill", "fill LARGEST 2-bounded region with 1, SMALLEST with 8; conditional + argmax + flood fill"),
    "08ed6ac7": ("recolor-by-rank", "recolor each column of 5s by its height rank (tallest=1, ...); canonical color-relational"),
}

_GLYPH = " 123456789▓"  # index = color; VOID -> ▓, 0 -> space


def render(task_id: str, max_pairs: int = 99):
    eps = G.load_task(task_id)
    ep = eps[0]
    print(f"task {task_id}: {len(ep.support)} support pairs + {len(eps)} test")
    pairs = (list(ep.support) + [(ep.query_x, ep.query_y)])[:max_pairs]
    for i, (x, y) in enumerate(pairs):
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
        render(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 99)
    elif len(sys.argv) >= 2 and sys.argv[1] == "list":
        print(f"{len(MANIFEST)} tasks in manifest:")
        for tid, (fam, note) in sorted(MANIFEST.items(), key=lambda kv: kv[1][0]):
            print(f"  {tid}  {fam:<18} {note}")
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
