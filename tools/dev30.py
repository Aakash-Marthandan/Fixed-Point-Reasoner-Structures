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
    # verified 2026-08-01 (rendered) — final 13; manifest FROZEN at 30 as the
    # pretraining holdout (ledger 2026-08-01). Coverage note: 20 families at
    # n=30 makes the original ">=2 per family" unreachable (17 distinct
    # families existed at 17/30); breadth was chosen over strict pairing —
    # 10 families paired, 10 singletons, R1-type tasks 10/30, R2 3/30 (+2
    # correspondence-adjacent). Verified spares (rendered, confirmed, EXCLUDED):
    # ea32f347 (recolor-by-rank; near-duplicate of 08ed6ac7), 74dd1130
    # (transpose; pure-D4 covered by 6150a2bd, cheapest family under orbit aug).
    "a79310a0": ("translation", "shape shifts down one row AND recolors 8->2; translation + constant-recolor compound"),
    "d511f180": ("color-mapping", "swap colors 5<->8 everywhere else identity; color transposition"),
    "4c4377d9": ("mirror-extend", "output = vertical mirror stacked ATOP original (mirror-above variant); size x2"),
    "3906de3d": ("axial-gravity", "2s float UP into/under the gaps of the 1-structure; gravity with obstruction"),
    "6e82a1ae": ("recolor-by-size", "recolor each 5-object by cell count: 4->1, 3->2, 2->3; R1 canonical measurement"),
    "9565186b": ("object-argmax", "majority color kept, ALL minority-color cells -> 5; color-argmax"),
    "b230c067": ("object-relational", "odd-one-out by shape: matching pair -> 1, unique shape -> 2; cross-object comparison"),
    "913fb3ed": ("object-conditional", "decorate each dot with color-specific surround (8->4-ring, 3->6-ring, 2->1-ring); R1 twin of 0ca9ddb6"),
    "23b5c85d": ("crop-to-content", "output = the SMALLEST rectangle, cropped; argmin-select + crop"),
    "05f2a901": ("object-relative-motion", "2-object moves until it touches the fixed 8-block; direction from relative position"),
    "dbc1a6ce": ("line-connection", "connect same-row/column 1-pairs with 8-lines, endpoints stay 1; Amendment-C axial family"),
    "007bbfb7": ("fractal-tile", "input stamped into each occupied cell of itself; 3x3 -> 9x9 self-composition"),
    "29ec7d0e": ("symmetry-completion", "restore occluded regions of a translation-periodic tiling; R2 periodic sibling of 3631a71a"),
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
