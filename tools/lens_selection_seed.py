#!/usr/bin/env python3
# Ledger: THE PAPER'S FINAL RUNS — analysis-time DESCRIPTIVE lens (no rules; labeled exploratory): is checkpoint selection
# seed-dependent? (the PI 2026-09-13: "save this for the analysis pass"; HANDOFF ANALYSIS-PASS ITEM). Reads only rows that
# already exist: the 512-puzzle monitor curves (EMA, every 2k) and the large-n D16 rows at two grids per arm (the selected
# grid on the full set, the final grid on the 50k), paired by idx with the frozen analyzer's selftested `paired`.
#   (A) per arm: argmax (earliest tie), the plateau window (EMA within 1 pp of the arm's max), the plateau end (the first grid
#       >= 2 pp under the running max for the rest of the run), the extension decision the argmax implied.
#   (B) within arm: selected grid vs final grid on the identical 50k (how much the step choice costs at large n).
#   (C) across seeds of one width at a COMMON step vs at their selected steps (does selection make or erase the seed spread?).
#   .venv/bin/python tools/lens_selection_seed.py --root runs/_paperfinal_pull/stage   -> <root>/analysis/selection_seed.txt
#   .venv/bin/python tools/lens_selection_seed.py --selftest
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_paperfinal import paired   # frozen at 7a99d9e, selftested (exact McNemar on idx pairing)

ARMS = ["C0", "C1", "C2", "C3", "C4", "C6", "C5", "C7", "C8"]
SEEDS = {"w384": ["C0", "C1", "C2"], "w192": ["C5", "C7", "C8"]}
BUDGET = {"C0": 30000, "C1": 30000, "C2": 30000, "C3": 50000, "C4": 30000, "C5": 30000, "C6": 50000, "C7": 30000, "C8": 30000}

def curve(root, a):
    mon = {}
    for l in (Path(root) / f"pretrainchamp_{a}" / "metrics.jsonl").read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            if "monitor" in r: mon[r["monitor"]["step"]] = r["monitor"]
    return sorted((s, m.get("val_t16_ema") or 0.0) for s, m in mon.items())

def argmax_earliest(c, upto=None):
    c = [x for x in c if upto is None or x[0] <= upto]
    best = max(v for _, v in c); return next(s for s, v in c if v == best), best

def plateau(c, tol=0.01):
    _, best = argmax_earliest(c); w = [s for s, v in c if v >= best - tol]; return (min(w), max(w)) if w else (None, None)

def plateau_end(c, drop=0.02):
    """First grid >= drop under the running max that never recovers above (running max - drop) afterwards, sustained over >= 2
    grids (a lone last grid is not an end); None if none."""
    run, mx = [], -1.0
    for s, v in c: mx = max(mx, v); run.append(mx)
    for i, (s, v) in enumerate(c[:-1]):
        if all(c[j][1] <= run[j] - drop for j in range(i, len(c))): return s
    return None

def recs(d):
    q = Path(d) / "records_all.npz"
    return dict(np.load(q, allow_pickle=True)) if q.exists() else None

def summ(d):
    q = Path(d) / "summary_all.json"; return json.loads(q.read_text()) if q.exists() else None

def lens(root):
    R, L = Path(root), []
    say = lambda s="": (L.append(s), print(s))
    say("SELECTION x SEED — descriptive lens (exploratory; no rules). Monitor = 512 puzzles, EMA, 2k grids (1 SE ~ 1 pp at .95).")
    say("(A) arm   budget  argmax(earliest tie)  EMA max  plateau(<=1pp)  plateau-end(>=2pp, sustained)  extension implied at the budget")
    for a in ARMS:
        c = curve(R, a); s_b, v_b = argmax_earliest(c, BUDGET[a]); s, v = argmax_earliest(c); p0, p1 = plateau(c); pe = plateau_end(c)
        ext = "fires" if s_b > BUDGET[a] - 4000 else "no"
        say(f"    {a}  {BUDGET[a]//1000:>3}k   {s//1000:>3}k                 {100*v:5.1f}    {p0//1000:>2}k-{p1//1000:>2}k        {('-' if pe is None else str(pe//1000)+'k'):>6}"
            f"                        {ext} (argmax at the budget: {s_b//1000}k, {100*v_b:.1f})")
    say("(B) within arm, D16 on the identical 50k: selected grid (full-set row) vs the final grid (50k row); diff = selected - final")
    for a in ARMS:
        sv, sf = summ(R / f"sxeval_pchamp{a}" / "full_vsel_t16"), summ(R / f"sxeval_pchamp{a}" / "full_final_t16")
        r = paired(recs(R / f"sxeval_pchamp{a}" / "full_vsel_t16"), recs(R / f"sxeval_pchamp{a}" / "full_final_t16"))
        cv = Path(str(sv.get("ckpt"))).name; steps = sorted(s for s, _ in curve(R, a)); fin = steps[-1]
        say(f"    {a}  {cv:<16} vs final {fin//1000}k: {100*r['diff']:+.2f} pp  ({r['only_a']}/{r['only_b']}, n {r['n']}, p {r['p']:.1e})")
    say("(C) across seeds of one width, D16 paired by idx: selected grids (on the full 422,786) vs final grids (on the 50k)")
    def row(a, kind): return R / f"sxeval_pchamp{a}" / ("full_vsel_t16" if kind == "vsel" else "full_final_t16")
    for x, y, note in (("C0", "C2", "w384: selected 16k vs 24k; finals = the common 30k (neither extended)"), ("C5", "C7", "w192, both selected 46k (a common grid) and both finals 50k")):
        for kind in ("vsel", "final"):
            r = paired(recs(row(x, kind)), recs(row(y, kind)))
            say(f"    {x}-{y} {kind:<5} {100*r['diff']:+.2f} pp ({r['only_a']}/{r['only_b']}, n {r['n']}, p {r['p']:.1e})   [{note}]")
    for w, arms in SEEDS.items():
        v = [summ(row(a, 'vsel'))['exact_acc'] for a in arms]
        say(f"    {w} seed spread at the selected grids, D16 full: {100*(max(v)-min(v)):.2f} pp ({', '.join(f'{a} {100*x:.2f}' for a, x in zip(arms, v))})")
    out = R / "analysis"; out.mkdir(parents=True, exist_ok=True); (out / "selection_seed.txt").write_text("\n".join(L) + "\n")

def selftest():
    c = [(2000, .1), (4000, .95), (6000, .96), (8000, .96), (10000, .955), (12000, .93), (14000, .92)]
    assert argmax_earliest(c) == (6000, .96); assert argmax_earliest(c, 4000) == (4000, .95)
    assert plateau(c) == (4000, 10000); assert plateau_end(c) == 12000
    assert plateau_end([(2000, .9), (4000, .87), (6000, .95)]) is None      # a dip that recovers is not an end
    assert plateau_end([(2000, .95), (4000, .95), (6000, .90)]) is None      # a lone last grid is not an end
    print("selftest OK: 6/6")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--root"); ap.add_argument("--selftest", action="store_true"); a = ap.parse_args()
    selftest() if a.selftest else lens(a.root)
