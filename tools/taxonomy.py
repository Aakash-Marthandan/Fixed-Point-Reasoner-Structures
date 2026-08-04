# Ledger: CC#2 Phase A (2026-08-02) — error taxonomy over captured predictions.
# Classifies every failed (task, arm, pair, attempt) into the registered
# classes; the class distribution decides the branch clause (transduction vs
# capacity/binding jumps the queue).
"""Error taxonomy for eval runs with --save-preds.

  .venv/bin/python tools/taxonomy.py runs/eval3r/results.jsonl [--render]
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from qhrrn2 import grid as G

_GLYPH = " 123456789▓"


def within_object_consistency(pred: np.ndarray, true: np.ndarray):
    """CI-8b (C17 named metric): over GT-uniform nonblack4 objects (>=3 cells)
    of TRUE, count those whose PRED cells are uniform (any single value).
    Size-mismatched pairs are SKIPPED (returns (0, 0)) — binding coherence,
    not size, is what this measures; the skip count is reported separately.
    Returns (n_pred_uniform, n_objects)."""
    from scipy import ndimage
    if pred.shape != true.shape:
        return 0, 0
    lab, n = ndimage.label((true != 0))
    ok = tot = 0
    for k in range(1, n + 1):
        m = lab == k
        if m.sum() < 3 or len(np.unique(true[m])) != 1:
            continue
        tot += 1
        ok += int(len(np.unique(pred[m])) == 1)
    return ok, tot


def classify(pred: np.ndarray, true: np.ndarray) -> tuple[str, int]:
    """(class, n_wrong_cells). Classes ordered by specificity."""
    if pred.shape != true.shape:
        return ("size", -1)
    diff = int((pred != true).sum())
    if diff == 0:
        return ("exact", 0)
    if diff <= 2:
        return ("near-1-2px", diff)
    if diff <= 5:
        return ("few-3-5px", diff)
    for k in range(1, 8):
        t = G.d4(true, k)
        if t.shape == pred.shape and np.array_equal(pred, t):
            return ("orientation", diff)
    # palette confusion: correct partition, wrong colors — test by relabeling
    # pred's colors via the best bijection onto true
    if pred.max() <= 9 and true.max() <= 9:
        lut = {}
        ok = True
        for c in np.unique(pred):
            vals, counts = np.unique(true[pred == c], return_counts=True)
            m = int(vals[counts.argmax()])
            if m in lut.values():
                ok = False
                break
            lut[int(c)] = m
        if ok:
            relab = np.vectorize(lambda v: lut.get(int(v), int(v)))(pred)
            if np.array_equal(relab, true):
                return ("palette", diff)
    frac = diff / true.size
    return ("structural" if frac > 0.3 else "partial-content", diff)


def render_pair(true, pred):
    rows = []
    h = max(true.shape[0], pred.shape[0])
    wt = true.shape[1]
    for r in range(h):
        left = "".join(_GLYPH[v] for v in true[r]) if r < true.shape[0] else " " * wt
        right = "".join(_GLYPH[v] for v in pred[r]) if r < pred.shape[0] else ""
        rows.append(f"  |{left}|   |{right}|")
    return "\n".join(rows)


def main():
    path = sys.argv[1]
    do_render = "--render" in sys.argv
    rows = [json.loads(l) for l in open(path)]
    counts = Counter()
    per_attempt = {0: Counter(), 1: Counter()}
    details = []
    for r in rows:
        if "preds" not in r:
            continue
        eps = G.load_task(r["task"])
        for i_ep, ep in enumerate(eps):
            if ep.query_y is None:
                continue
            for i_att, pred in enumerate(r["preds"][i_ep]):
                p = np.asarray(pred, dtype=np.int8)
                cls, diff = classify(p, ep.query_y)
                if cls == "exact":
                    continue
                counts[cls] += 1
                per_attempt[i_att][cls] += 1
                details.append((r["task"], r["family"], r["arm"], i_ep, i_att,
                                cls, diff, p, ep.query_y))
    total = sum(counts.values())
    print(f"{total} failed (task, pair, attempt) cells\n")
    for cls, n in counts.most_common():
        print(f"  {cls:16s} {n:4d}  ({100*n/total:.0f}%)")
    print("\nby attempt (0 = selected/voted, 1 = MDL/final):")
    for i in (0, 1):
        t = sum(per_attempt[i].values())
        print(f"  attempt {i}: " + ", ".join(f"{c}:{n}" for c, n in per_attempt[i].most_common()))
    print("\nper-task worst-attempt summary (arm A then B):")
    seen = set()
    for task, fam, arm, i_ep, i_att, cls, diff, p, ty in details:
        key = (task, arm, i_ep)
        if key in seen:
            continue
        seen.add(key)
        print(f"  {task} {fam:<20} {arm:2s} pair{i_ep}: {cls} ({diff} cells)")
    if do_render:
        print("\n===== RENDERS (true | pred), attempt 0 only =====")
        for task, fam, arm, i_ep, i_att, cls, diff, p, ty in details:
            if i_att != 0:
                continue
            print(f"\n--- {task} {fam} arm {arm} pair {i_ep}: {cls} ({diff})")
            print(render_pair(ty, p))


if __name__ == "__main__":
    main()
