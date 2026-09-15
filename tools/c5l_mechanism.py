#!/usr/bin/env python3
# Ledger: THE WIDTH-192 LONG RUN, plan §9 Note 1 (2026-09-15; committed before any long-run instrument row was pulled to the Mac).
# The mechanism label printed beside every onset letter, from the registered per-grid instruments, read with the frozen curve helpers
# (tools/c5l_curves.py) and the frozen analyzer's curve assembly, onset parameters and has_long gate (tools/analyze_c5l.py, both 57b5656):
#   H       held-out 10k exact counts per grid (C5: the c5l rows, the c8x rows filling 10k-50k); T = train-1k exact counts per grid.
#   dH, dT  the curve's running maximum through the grid minus its value there, in pp of the instrument's size (1 pp = 100 / 10 counts).
#   label   at the onset grid and the next three grids of that run's held-out grid list (fewer only where the run ends first; the
#           onset rule guarantees >= 3): dT >= dH / 2 at all of them -> COLLAPSE-TYPE; dT < dH / 2 at all -> MEMORIZATION-TYPE;
#           else MIXED. No onset, no label. The frozen analyzer's letters are not touched.
#   flags   a clock ratio (rho192: C5 vs the width-384 onsets; rho512: A7 vs A8; R-L4: every run) whose labeled onsets differ, or
#           include MIXED -> mixed-mechanism (reported; not support for either clock hypothesis). < 2 labeled onsets -> n/a.
#   drops   every maximal run of consecutive held-out grids with dH >= 10 pp: first/last grid, the deepest grid (earliest on a tie),
#           dH and dT there, and whether dH came back under 10 pp before the run's last grid (descriptive, onset or not).
"""  .venv/bin/python tools/c5l_mechanism.py --c8x-root runs/_c8x_pull/x/runs --x-root runs/_c5l_pull/x/runs [--out runs/analysis]
  .venv/bin/python tools/c5l_mechanism.py --selftest"""
from __future__ import annotations
import argparse, json, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from c5l_curves import curve, onset                          # frozen at 57b5656, selftested
from analyze_c5l import N_VAL, N_TR1K, EXT_FROM, WIDE, R      # frozen at 57b5656, selftested

EPISODE_PP = 10.0
RUNS = ("C5",) + WIDE

def kfmt(s): return "-" if s is None else f"{s // 1000}k"

def curves(c8x, x):
    X, C8 = Path(x), Path(c8x)
    h5 = dict(curve(X, "val", "C5", ("c5l",), N_VAL)); h5.update({s: k for s, k in curve(C8, "val", "C5", ("c8x",), N_VAL) if s not in h5})
    H, T = {"C5": sorted(h5.items())}, {"C5": curve(X, "tr1k", "C5", ("c5l",), N_TR1K)}
    for r in WIDE: H[r], T[r] = curve(X, "val", r, ("c5l",), N_VAL), curve(X, "tr1k", r, ("c5l",), N_TR1K)
    if not any(s > EXT_FROM for s, _ in H["C5"]): H["C5"], T["C5"] = [], []   # the analyzer's has_long gate
    return H, T

def drops(c, n):
    out, mx = [], -1
    for s, k in c: mx = max(mx, k); out.append((s, 100.0 * (mx - k) / n))
    return out

def label(h, t):
    """(onset, label, rows) for one run; rows = [(step, dH, dT)] at the labeled grids."""
    if not h or not t: return None, None, []
    o = onset(h, R["drop"], R["tail"])
    if o is None: return None, None, []
    dH, dT = dict(drops(h, N_VAL)), dict(drops(t, N_TR1K))
    rows = [(s, dH[s], dT[s]) for s, _ in h if s >= o and s in dT][:4]
    if not rows: return o, None, []
    col = [dt >= dh / 2.0 for _, dh, dt in rows]
    return o, ("COLLAPSE-TYPE" if all(col) else "MEMORIZATION-TYPE" if not any(col) else "MIXED"), rows

def episodes(h, t):
    dT = dict(drops(t, N_TR1K)) if t else {}
    d, out, cur = drops(h, N_VAL), [], []
    for i, (s, dh) in enumerate(d):
        if dh >= EPISODE_PP: cur.append((s, dh))
        if cur and (dh < EPISODE_PP or i == len(d) - 1):
            deep = max(cur, key=lambda r: (r[1], -r[0]))
            out.append(dict(first=cur[0][0], last=cur[-1][0], deepest=deep[0], dH=round(deep[1], 2),
                            dT=(round(dT[deep[0]], 2) if deep[0] in dT else None), recovered=bool(dh < EPISODE_PP)))
            cur = []
    return out

def flag(labels, runs):
    got = [labels.get(r) for r in runs if labels.get(r)]
    if len(got) < 2: return "n/a (fewer than two labeled onsets)"
    return f"same-mechanism {got[0]}" if len(set(got)) == 1 and got[0] != "MIXED" else "mixed-mechanism"

def analyze(c8x, x, out_dir=None):
    H, T = curves(c8x, x); res, lines = {}, ["THE WIDTH-192 LONG RUN — plan §9 Note 1: mechanism labels beside the frozen onset letters (tools/c5l_mechanism.py)"]
    for r in RUNS:
        o, L, rows = label(H[r], T[r])
        res[r] = dict(onset=o, label=L, rows=[dict(step=s, dH=round(a, 2), dT=round(b, 2)) for s, a, b in rows], episodes=episodes(H[r], T[r]) if H[r] else [],
                      grids_H=len(H[r]), grids_T=len(T[r]))
        det = " | ".join(f"{kfmt(s)} dH {a:.2f} dT {b:.2f}" for s, a, b in rows)
        lines.append(f"  {r:<3} onset {kfmt(o):>5}  {L or ('NO-DATA' if not H[r] else 'no onset, no label'):<18} {det}")
    labels = {r: res[r]["label"] for r in RUNS}
    F = {"rho192 (C5 vs C1, A5)": flag(labels, ("C5", "C1", "A5")), "rho512 (A7 vs A8)": flag(labels, ("A7", "A8")), "R-L4 window (all runs)": flag(labels, RUNS)}
    lines += [f"  {k:<24} {v}" for k, v in F.items()]
    lines.append(f"  held-out drops >= {EPISODE_PP:.0f} pp under the running maximum (descriptive):")
    for r in RUNS:
        for e in res[r]["episodes"]:
            lines.append(f"    {r:<3} {kfmt(e['first'])}-{kfmt(e['last'])}  deepest {kfmt(e['deepest'])}: dH {e['dH']:.2f} dT {e['dT'] if e['dT'] is None else format(e['dT'], '.2f')}  "
                         f"{'recovered under 10 pp' if e['recovered'] else 'not recovered by the last grid'}")
    print("\n".join(lines))
    if out_dir:
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        (out / "c5l_mechanism.txt").write_text("\n".join(lines) + "\n"); (out / "c5l_mechanism.json").write_text(json.dumps(dict(runs=res, flags=F), indent=1))
    return res, F

def selftest():
    n = 0
    h = [(2, 9000), (4, 9500), (6, 9600), (8, 9000), (10, 8000), (12, 7000), (14, 7500), (16, 7200)]          # peak 6, onset 8
    tc = [(2, 900), (4, 980), (6, 1000), (8, 960), (10, 900), (12, 850), (14, 880), (16, 870)]              # dT 4/10/15/12 vs dH 6/16/26/21
    o, L, rows = label(h, tc); assert (o, L, [s for s, _, _ in rows]) == (8, "COLLAPSE-TYPE", [8, 10, 12, 14]), (o, L, rows); n += 1
    tm = [(s, 1000 if s >= 6 else k) for s, k in tc]
    assert label(h, tm)[1] == "MEMORIZATION-TYPE"; n += 1
    tx = [(s, 960 if s == 8 else (1000 if s >= 6 else k)) for s, k in tc]
    assert label(h, tx)[1] == "MIXED"; n += 1
    hd = [(2, 9000), (4, 9600), (6, 8000), (8, 8500), (10, 9600), (12, 9550), (14, 9500)]                  # a dip that recovers: no onset
    td = [(2, 900), (4, 1000), (6, 850), (8, 900), (10, 1000), (12, 1000), (14, 1000)]
    assert label(hd, td) == (None, None, []) and episodes(hd, td) == [dict(first=6, last=8, deepest=6, dH=16.0, dT=15.0, recovered=True)]; n += 1
    he = [(2, 9000), (4, 9600), (6, 9400), (8, 9000), (10, 9000)]                                            # onset with exactly 3 grids left
    o, L, rows = label(he, [(s, 1000) for s, _ in he]); assert (o, L, len(rows)) == (6, "MEMORIZATION-TYPE", 3); n += 1
    assert episodes([(2, 9000), (4, 9600), (6, 8000), (8, 8000)], []) == [dict(first=6, last=8, deepest=6, dH=16.0, dT=None, recovered=False)]; n += 1
    assert flag({"C5": "COLLAPSE-TYPE", "C1": "MEMORIZATION-TYPE", "A5": None}, ("C5", "C1", "A5")) == "mixed-mechanism"
    assert flag({"A7": "MEMORIZATION-TYPE", "A8": "MEMORIZATION-TYPE"}, ("A7", "A8")) == "same-mechanism MEMORIZATION-TYPE"
    assert flag({"A7": "MIXED", "A8": "MIXED"}, ("A7", "A8")) == "mixed-mechanism" and flag({"C5": None, "C1": "MIXED"}, ("C5", "C1")).startswith("n/a"); n += 1
    d = Path(tempfile.mkdtemp(prefix="c5lm_")); X, C8 = d / "x", d / "c8"
    def row(root, pre, kind, run, st, acc, size):
        p = root / f"{pre}_{kind}_p{run}_s{st:06d}"; p.mkdir(parents=True); (p / "summary_all.json").write_text(json.dumps({"n": size, "exact_acc": acc}))
    row(X, "c5l", "val", "C5", 52000, .90, N_VAL); row(C8, "c8x", "val", "C5", 10000, .95, N_VAL); row(C8, "c8x", "val", "C5", 52000, .10, N_VAL)
    row(X, "c5l", "tr1k", "C5", 52000, .99, N_TR1K)
    H, T = curves(C8, X); assert H["C5"] == [(10000, 9500), (52000, 9000)] and T["C5"] == [(52000, 990)] and H["A8"] == [], (H, T); n += 1
    d2 = Path(tempfile.mkdtemp(prefix="c5lm_")); row(d2 / "c8", "c8x", "val", "C5", 10000, .95, N_VAL); (d2 / "x").mkdir()
    assert curves(d2 / "c8", d2 / "x")[0]["C5"] == []; n += 1                                                 # the has_long gate
    print(f"selftest OK: {n}/{n}")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--c8x-root"); ap.add_argument("--x-root"); ap.add_argument("--out"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest: return selftest()
    analyze(a.c8x_root, a.x_root, a.out)

if __name__ == "__main__":
    main()
