#!/usr/bin/env python3
# Ledger: THE C8 EXTENSION (2026-09-14; Documentation/Plan_2026-09-14_C8_Extension.md = the registration; frozen from its commit).
# The LOW-NOISE SELECTION on the 10,000-puzzle held-out set (tools/sx_make_valset.py; never the test set, never the monitor): each
# grid's D16 EMA exact count from runs/c8x_val_p<ARM>_s<STEP>/summary_all.json; the selected grid = the maximum exact count over the
# arm's grids at or above --min-step, the EARLIEST step on a tie (the registered selector's tie rule). The chain calls it on the pod to
# decide the extra test rows; the analyzer imports `curve` / `select` so both read the curve one way.
"""  .venv/bin/python tools/c8x_valcurve.py --root runs --arms C8 C5 C7 --select C8 [--n 10000 --min-step 10000 --out runs/analysis/c8x_valcurve.json]
     -> prints the selected step of --select as six digits (exit 1 when that arm has no complete row)
  .venv/bin/python tools/c8x_valcurve.py --selftest"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

def curve(root, arm, n=10000, min_step=10000):
    """[(step, exact_count, acc)] for every complete val row of ARM (n == N), sorted by step."""
    out = []
    for d in Path(root).glob(f"c8x_val_p{arm}_s*"):
        m = re.search(r"_s(\d+)$", d.name); s = d / "summary_all.json"
        if not m or not s.exists(): continue
        j = json.loads(s.read_text()); step = int(m.group(1))
        if int(j.get("n", -1)) != n or step < min_step: continue
        out.append((step, int(round(float(j["exact_acc"]) * n)), float(j["exact_acc"])))
    return sorted(out)

def select(c):
    """The grid with the maximum exact count; the earliest step on a tie. None on an empty curve."""
    if not c: return None
    best = max(k for _, k, _ in c)
    return min(s for s, k, _ in c if k == best)

def plateau(c, tol_count=50):
    """Grids whose exact count is within tol_count of the maximum (50 of 10,000 = 0.5 pp)."""
    if not c: return []
    best = max(k for _, k, _ in c)
    return [s for s, k, _ in c if k >= best - tol_count]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root"); ap.add_argument("--arms", nargs="+", default=["C8"]); ap.add_argument("--select", default="C8")
    ap.add_argument("--n", type=int, default=10000); ap.add_argument("--min-step", type=int, default=10000); ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest: return selftest()
    res = {}
    for arm in a.arms:
        c = curve(a.root, arm, a.n, a.min_step)
        res[arm] = dict(curve=[dict(step=s, exact=k, acc=v) for s, k, v in c], selected=select(c), plateau=plateau(c))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(res, indent=1))
    sel = res.get(a.select, {}).get("selected")
    if sel is None: sys.exit(1)
    print(f"{sel:06d}")

def selftest():
    import tempfile
    d = Path(tempfile.mkdtemp(prefix="c8xvc_"))
    for step, acc, n in ((10000, .90, 10000), (22000, .95, 10000), (46000, .96, 10000), (48000, .96, 10000), (50000, .97, 9999), (8000, .99, 10000)):
        p = d / f"c8x_val_pC8_s{step:06d}"; p.mkdir(); (p / "summary_all.json").write_text(json.dumps({"n": n, "exact_acc": acc}))
    (d / "c8x_val_pC8_s030000").mkdir()   # an incomplete row (no summary) is not a grid
    c = curve(d, "C8")
    assert [s for s, _, _ in c] == [10000, 22000, 46000, 48000], c      # n != 10,000 and steps under --min-step excluded
    assert select(c) == 46000                                           # the tie 46k = 48k -> the earliest
    assert plateau(c) == [46000, 48000] and plateau(c, 150) == [22000, 46000, 48000]
    assert select([]) is None and curve(d, "C5") == []
    print("selftest OK: 4/4")

if __name__ == "__main__":
    main()
