#!/usr/bin/env python3
# Ledger: THE PAPER'S FINAL RUNS — analysis-time DESCRIPTIVE lens (no rules; labeled exploratory) for the PI's two questions
# (2026-09-14): (Q1) why the width-192 seed pair lowered the single-seed number; (Q2) why the 0.79M width-192 cell matches the
# 2.78M width-384 cell. Reads existing rows only (the staging root of the paper-final analysis):
#   (1) the drop decomposed per row; seed noise per width; C5's place among the champion night's seven arms (selection bias)
#   (2) C8's gap to its width-192 siblings by depth; the extension rule's decisive monitor margins (puzzles of 512)
#   (3) width per seed: paired deltas by depth; the per-step accuracy curve on the full set (from first_exact) and its crossover;
#       accuracy at MATCHED arithmetic per puzzle (the measured GMAC per outer step: w192 14.82, w384 49.95); the depth dividend
#   (4) consistency: step 16 of the D64 records == the separate D16 row (a free end-to-end check of the rows)
#   (5) the memorization clock: training fit vs the held-out monitor by width
#   .venv/bin/python tools/lens_width_seed.py --root runs/_paperfinal_pull/stage   -> <root>/analysis/width_seed.txt
#   .venv/bin/python tools/lens_width_seed.py --selftest
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_paperfinal import paired   # frozen at 7a99d9e, selftested

GMAC = {"w192": 14.816297984, "w384": 49.94912256}   # runs/analysis/mac_count_20260910.json (XLA cost analysis, measured)
PAIRS = [("C5", "C0", 0), ("C7", "C1", 1), ("C8", "C2", 2)]
W192, W384 = ["C5", "C7", "C8"], ["C0", "C1", "C2"]
SEVEN = ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]

def summ(d): q = Path(d) / "summary_all.json"; return json.loads(q.read_text()) if q.exists() else None
def recs(d): q = Path(d) / "records_all.npz"; return dict(np.load(q, allow_pickle=True)) if q.exists() else None

def curve_from_first(first, T):
    """acc(t) for t = 1..T from the evaluator's per-puzzle first_exact, which is 0-BASED (tools/eval_sudoku_extreme.py: 0 = exact
    after the first outer step, -1 = never; its summary's mean_first_exact averages first_exact >= 0)."""
    f = np.asarray(first); f = np.where((f >= 0) & (f < T), f, T)
    counts = np.bincount(f, minlength=T + 1)[:T]
    return np.cumsum(counts) / len(f)

def crossover(ca, cb):
    """First step (1-based) from which curve a stays >= curve b through the end; None if never."""
    ge = np.asarray(ca) >= np.asarray(cb)
    for i in range(len(ge)):
        if ge[i:].all(): return i + 1
    return None

def expected_max_normal(n):
    """E[max of n iid N(0,1)] by numerical integration."""
    x = np.linspace(-10, 10, 200001); phi = np.exp(-x * x / 2) / math.sqrt(2 * math.pi)
    Phi = np.cumsum(phi) * (x[1] - x[0])
    return float(np.sum(x * n * phi * Phi ** (n - 1)) * (x[1] - x[0]))

def monitor(root, a):
    mon = {}
    for l in (Path(root) / f"pretrainchamp_{a}" / "metrics.jsonl").read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            if "monitor" in r: mon[r["monitor"]["step"]] = r["monitor"]
    return mon

def train_rows(root, a):
    return {r["step"]: r for r in (json.loads(l) for l in (Path(root) / f"pretrainchamp_{a}" / "metrics.jsonl").read_text().splitlines() if l.strip()) if "loss" in r}

def lens(root):
    R, L = Path(root), []
    say = lambda s="": (L.append(s), print(s))
    acc = lambda d: (summ(R / d) or {}).get("exact_acc")
    ROW = {"D16 full": "sxeval_pchamp{a}/full_vsel_t16", "D64 full": "filler_sxeval_pchamp{a}_full_t64",
           "D128 50k": "filler_sxeval_pchamp{a}_sub50000_t128", "D256 50k": "filler_sxeval_pchamp{a}_sub50000_t256"}
    say("WIDTH x SEED — descriptive lens (exploratory; no rules)")
    say("(1) Q1: the single-seed number vs the triple, decomposed (pp; contribution = (arm - C5) / 3)")
    for nm, p in ROW.items():
        v = {a: acc(p.format(a=a)) for a in W192}
        m = np.mean(list(v.values()))
        say(f"    {nm:<9} C5 {100*v['C5']:.2f} -> triple {100*m:.2f} ({100*(m-v['C5']):+.2f}): C7 {100*(v['C7']-v['C5'])/3:+.2f}, C8 {100*(v['C8']-v['C5'])/3:+.2f}")
    for w, arms in (("w192", W192), ("w384", W384)):
        for nm in ("D16 full", "D64 full"):
            v = np.array([acc(ROW[nm].format(a=a)) for a in arms]); sd = v.std(ddof=1)
            say(f"    seed SD {w} {nm}: {100*sd:.2f} pp (n 3; values {', '.join(f'{100*x:.2f}' for x in v)})" + (f"; C5 sits {(v[0]-v.mean())/sd:+.2f} SD above its triple" if w == "w192" else ""))
    d64 = {a: acc(f"sxeval_pchamp{a}/full_vsel_t64") for a in SEVEN}
    rank = sorted(d64, key=d64.get, reverse=True)
    say(f"    the champion night's seven arms at D64 (100k): {' > '.join(f'{a} {100*d64[a]:.2f}' for a in rank)}; C5 was labeled best of 7")
    say(f"    selection bias: if seven arms were one recipe, E[best of 7] = {expected_max_normal(7):+.2f} SD above their mean (the champion verdict read every treatment contrast FLAT)")
    say("(2) C8 by depth (the gap closes with depth) and the extension rule's decisive margins on the 512-puzzle monitor")
    for nm, p in ROW.items():
        v = {a: acc(p.format(a=a)) for a in W192}
        say(f"    {nm:<9} C8-C5 {100*(v['C8']-v['C5']):+.2f}  C8-C7 {100*(v['C8']-v['C7']):+.2f}  C7-C5 {100*(v['C7']-v['C5']):+.2f}")
    for a in ("C8", "C2", "C0", "C7", "C1", "C5"):
        mon = monitor(R, a); grid = {s: m for s, m in mon.items() if s <= 30000 and s % 2000 == 0}
        key = lambda s: (round(grid[s]["val_t16_ema"] * 512), round((grid[s].get("val_t16") or 0) * 512))
        best = max(grid, key=lambda s: (key(s), -s)); win = [s for s in grid if s >= 26000]
        wbest = max(win, key=lambda s: (key(s), -s))
        need = "fires" if best >= 26000 else f"the window's best {wbest//1000}k needed EMA > {key(best)[0]}/512 or = {key(best)[0]} with raw > {key(best)[1]}/512; it read {key(wbest)[0]}/512 raw {key(wbest)[1]}/512"
        say(f"    {a}: selected by 30k at {best//1000}k (EMA {key(best)[0]}/512, raw {key(best)[1]}/512) -> {need}")
    say("(3) Q2: width per seed — paired deltas (w192 - w384) by depth on identical puzzles")
    for nm, p in (("D16 full", "sxeval_pchamp{a}/full_vsel_t16"), ("D64 full", "filler_sxeval_pchamp{a}_full_t64"),
                  ("D128 20k", "sxeval_pchamp{a}/sub20k_t128"), ("D256 5k", "sxeval_pchamp{a}/sub5k_t256")):
        parts, ds = [], []
        for x, y, s in PAIRS:
            r = paired(recs(R / p.format(a=x)), recs(R / p.format(a=y))); ds.append(r["diff"])
            parts.append(f"s{s} {100*r['diff']:+.2f} (p {r['p']:.0e})")
        say(f"    {nm:<9} mean {100*np.mean(ds):+.2f}: " + "; ".join(parts))
    say("    per-step accuracy on the full set from the D64 records' first_exact (zero depth regressions make it exact):")
    curves = {a: curve_from_first(recs(R / f"filler_sxeval_pchamp{a}_full_t64")["first_exact"], 64) for a in W192 + W384}
    ratio = GMAC["w384"] / GMAC["w192"]
    for x, y, s in PAIRS:
        cx, cy = curves[x], curves[y]; xo = crossover(cx, cy)
        say(f"    seed {s}: {x} vs {y}  t4 {100*cx[3]:.1f}/{100*cy[3]:.1f}  t8 {100*cx[7]:.1f}/{100*cy[7]:.1f}  t16 {100*cx[15]:.2f}/{100*cy[15]:.2f}"
            f"  t32 {100*cx[31]:.2f}/{100*cy[31]:.2f}  t64 {100*cx[63]:.2f}/{100*cy[63]:.2f}  | w192 ahead from step {xo} on"
            f"  | dividend t16->t64 {100*(cx[63]-cx[15]):+.2f} vs {100*(cy[63]-cy[15]):+.2f}")
    say(f"    matched arithmetic per puzzle (measured GMAC/step w192 {GMAC['w192']:.2f}, w384 {GMAC['w384']:.2f}; ratio {ratio:.2f}):")
    for t384 in (8, 16, 19):
        t192 = min(64, int(round(t384 * ratio)))
        vals = [(curves[x][t192 - 1], curves[y][t384 - 1]) for x, y, _ in PAIRS]
        say(f"    w384 at t{t384} ({t384*GMAC['w384']:.0f} GMAC) {100*np.mean([b for _, b in vals]):.2f} vs w192 at t{t192} ({t192*GMAC['w192']:.0f} GMAC) {100*np.mean([a_ for a_, _ in vals]):.2f}"
            f"  (per seed {', '.join(f'{100*(a_-b):+.2f}' for a_, b in vals)})")
    fe = {}
    for a in W192 + W384:
        z = recs(R / f"filler_sxeval_pchamp{a}_full_t64"); f = np.asarray(z["first_exact"]); ok = f >= 0
        fe[a] = (float(np.mean(f[ok] + 1)), float(np.percentile(f[ok] + 1, 90)), float(np.percentile(f[ok] + 1, 99)))
    say("    first-exact step over solved puzzles (D64 full, 1-based; mean / p90 / p99): " + ", ".join(f"{a} {m:.2f}/{p9:.0f}/{p99:.0f}" for a, (m, p9, p99) in ((a, fe[a]) for a in W384 + W192)))
    say("(4) consistency: the D64 run's step-16 accuracy vs the separate D16 row (full set)")
    for a in W192 + W384:
        d16 = acc(ROW["D16 full"].format(a=a))
        say(f"    {a}: t16 of D64 {100*curves[a][15]:.3f} vs D16 row {100*d16:.3f} ({100*(curves[a][15]-d16):+.3f})")
    say("(5) the memorization clock by width: training fit vs the held-out monitor (EMA, 512 puzzles)")
    for a in W384 + W192 + ["C3", "C6"]:
        tr, mon = train_rows(R, a), monitor(R, a)
        last = max(mon); mx = max(mon.values(), key=lambda m: m["val_t16_ema"])
        pick = lambda s: tr.get(s) or tr[min(tr, key=lambda k: abs(k - s))]
        tx = "  ".join(f"{s//1000}k ce {pick(s)['ce_in']:.3f} tx {pick(s)['train_exact']:.2f}" for s in (20000, 30000, 50000) if s <= max(tr))
        say(f"    {a}: monitor max {100*mx['val_t16_ema']:.1f} at {mx['step']//1000}k, last {100*mon[last]['val_t16_ema']:.1f} at {last//1000}k ({100*(mon[last]['val_t16_ema']-mx['val_t16_ema']):+.1f}) | train {tx}")
    out = R / "analysis"; out.mkdir(parents=True, exist_ok=True); (out / "width_seed.txt").write_text("\n".join(L) + "\n")

def selftest():
    c = curve_from_first(np.array([0, 1, 1, -1, 4, 3, 70]), 4)   # 0-based: exact after step 1, 2, 2, never, step 5 (> T), step 4, never
    assert np.allclose(c, [1/7, 3/7, 3/7, 4/7]), c
    assert crossover([.1, .5, .9, .95], [.2, .6, .8, .9]) == 3
    assert crossover([.1, .9, .5], [.2, .6, .8]) is None
    assert abs(expected_max_normal(7) - 1.3522) < 1e-3 and abs(expected_max_normal(2) - 1 / math.sqrt(math.pi)) < 1e-3
    print("selftest OK: 4/4")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--root"); ap.add_argument("--selftest", action="store_true"); a = ap.parse_args()
    selftest() if a.selftest else lens(a.root)
