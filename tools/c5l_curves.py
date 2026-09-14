#!/usr/bin/env python3
# Ledger: THE WIDTH-192 LONG RUN (2026-09-14; Documentation/Plan_2026-09-14_W192_Long.md = the registration; frozen from its commit).
# The per-grid instrument curves, read ONE way by the pod chain (the held-out pick) and the frozen analyzer (every letter):
#   held-out  <prefix>_val_p<RUN>_s<STEP>/summary_all.json   (10,000 held-out train-file puzzles, split val, D16, EMA)
#   train-1k  <prefix>_tr1k_p<RUN>_s<STEP>/summary_all.json  (the 1,000 training puzzles, split train, D16, EMA)
# A curve = [(step, exact count)] over complete rows (n equal to the instrument's size); several prefixes merge (the first wins on a step).
#   select   the maximum count, the EARLIEST step on a tie
#   plateau  the grids within `tol` counts of the maximum (a set; not necessarily contiguous)
#   onset    the first grid AFTER the peak from which every later grid sits >= `drop` counts under the running maximum, with >= `tail`
#            grids from it to the end (a dip that recovers is not an onset; a lone last grid is not an onset)
#   smooth   W(c) = the mean count over the grid c and its two neighbors at +/- `step`, where all three are measured
"""  .venv/bin/python tools/c5l_curves.py --root runs --kind val --run C5 --prefixes c5l c8x --n 10000 --min-step 10000   -> the pick, six digits
  .venv/bin/python tools/c5l_curves.py --selftest"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

def curve(root, kind, run, prefixes=("c5l",), n=10000, min_step=0, max_step=10**9):
    got = {}
    for pre in prefixes:
        for d in Path(root).glob(f"{pre}_{kind}_p{run}_s*"):
            m = re.search(r"_s(\d+)$", d.name); s = d / "summary_all.json"
            if not m or not s.exists(): continue
            st = int(m.group(1)); j = json.loads(s.read_text())
            if int(j.get("n", -1)) != n or not (min_step <= st <= max_step) or st in got: continue
            got[st] = int(round(float(j["exact_acc"]) * n))
    return sorted(got.items())

def select(c):
    if not c: return None
    best = max(k for _, k in c); return min(s for s, k in c if k == best)

def plateau(c, tol):
    if not c: return []
    best = max(k for _, k in c); return [s for s, k in c if k >= best - tol]

def onset(c, drop, tail=3):
    if len(c) < tail: return None
    pk = select(c); run, mx = [], -1
    for _, k in c: mx = max(mx, k); run.append(mx)
    for i, (s, _) in enumerate(c):
        if s <= pk or len(c) - i < tail: continue
        if all(c[j][1] <= run[j] - drop for j in range(i, len(c))): return s
    return None

def smooth_max(c, lo, hi, step):
    d = dict(c); best = None
    for s, _ in c:
        if lo <= s <= hi and (s - step) in d and (s + step) in d:
            w = (d[s - step] + d[s] + d[s + step]) / 3.0
            if best is None or w > best[1]: best = (s, w)
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root"); ap.add_argument("--kind", default="val"); ap.add_argument("--run", default="C5"); ap.add_argument("--prefixes", nargs="+", default=["c5l"])
    ap.add_argument("--n", type=int, default=10000); ap.add_argument("--min-step", type=int, default=10000); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest: return selftest()
    pick = select(curve(a.root, a.kind, a.run, a.prefixes, a.n, a.min_step))
    if pick is None: sys.exit(1)
    print(f"{pick:06d}")

def selftest():
    import tempfile
    d = Path(tempfile.mkdtemp(prefix="c5lc_"))
    def row(pre, kind, run, st, acc, n):
        p = d / f"{pre}_{kind}_p{run}_s{st:06d}"; p.mkdir(); (p / "summary_all.json").write_text(json.dumps({"n": n, "exact_acc": acc}))
    for st, acc in ((10000, .90), (12000, .95)): row("c8x", "val", "C5", st, acc, 10000)
    row("c5l", "val", "C5", 12000, .10, 10000)                                  # the first prefix wins a shared step
    for st, acc in ((14000, .96), (16000, .96), (18000, .9)): row("c5l", "val", "C5", st, acc, 10000)
    row("c5l", "val", "C5", 20000, .99, 9999)                                   # n != 10,000 is not a grid
    c = curve(d, "val", "C5", ("c5l", "c8x"), 10000)
    assert c == [(10000, 9000), (12000, 1000), (14000, 9600), (16000, 9600), (18000, 9000)], c
    assert curve(d, "val", "C5", ("c8x", "c5l"), 10000)[1] == (12000, 9500)
    assert select(c) == 14000 and plateau(c, 50) == [14000, 16000] and select([]) is None
    assert onset([(2, 10), (4, 20), (6, 16), (8, 16), (10, 16)], 3) == 6                 # sustained >= 3 under the running max from 6
    assert onset([(2, 10), (4, 20), (6, 16), (8, 20), (10, 16)], 3) is None             # a recovery breaks it
    assert onset([(2, 10), (4, 20), (6, 20), (8, 20), (10, 10)], 3) is None             # a lone last grid (tail 1 < 3)
    assert smooth_max([(2, 10), (4, 20), (6, 30), (8, 25), (10, 5)], 0, 100, 2) == (6, 25.0)
    print("selftest OK: 7/7")

if __name__ == "__main__":
    main()
