"""Freethink grounding 1 ($0, records + npz): WHAT IS UNREACHABLE? Reachability (any hit in 128 random-init draws) and cold
solve vs the puzzle's givens count (erasures) and test source, per map; the unreachable-by-all-our-RI-maps set and the
field's coverage of it; overlap of the field's cold failures with our unreachable set."""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, "src")
from qhrrn2 import sudoku_extreme as SX
d = SX.load_prepared("data/sudoku_extreme/sudoku_extreme_seed0.npz"); Q, R, SRC, SN = d["test_q"], d["test_rating"], d["test_source"], d["source_names"]
G = (Q != 0).reshape(len(Q), -1).sum(1)
def recs(p):
    z = dict(np.load(Path(p) / "records_all.npz", allow_pickle=True)); o = np.argsort(z["idx"], kind="stable")
    return {k: (v[o] if hasattr(v, "shape") and v.shape and v.shape[0] == len(o) else v) for k, v in z.items()}
L = []
def say(s=""): L.append(s); print(s)
S = {n: recs(f"runs/{p}") for n, p in (("X0", "sxscan_psportC1X0"), ("R0", "sxscan_psportC1R0"), ("A0", "sxscan_psportC1A0"), ("P3s1", "sxscan_psportC0P3s1"), ("D4", "sxscan_psportBr2bD4"), ("C3X", "sxscan_psportBr2bC3X"))}
idx = S["R0"]["idx"]; assert all(np.array_equal(S[k]["idx"], idx) for k in S)
g = G[idx]; src = SRC[idx]; rat = R[idx]
def reach(z): return z["cold_exact"].astype(bool) | (z["mi_first_hit"] >= 0)
say("== GROUND 1: reachability (any hit @128) and cold vs GIVENS count (erasures = 81 - givens) on the identical 20k ==")
bins = [(17, 21), (21, 23), (23, 25), (25, 27), (27, 29), (29, 35)]
say("  givens bin | n | " + " | ".join(f"{k} reach/cold" for k in S))
for lo, hi in bins:
    m = (g >= lo) & (g < hi)
    if m.sum() == 0: continue
    say(f"  [{lo},{hi}) | {m.sum():5d} | " + " | ".join(f"{100*reach(S[k])[m].mean():5.1f}/{100*S[k]['cold_exact'][m].mean():5.1f}" for k in S))
say("  Spearman(reach, givens) per map: " + " ".join(f"{k} {np.corrcoef(reach(S[k]).astype(float), g)[0,1]:+.3f}" for k in S) + " | Spearman(reach, rating): " + " ".join(f"{k} {np.corrcoef(reach(S[k]).astype(float), rat)[0,1]:+.3f}" for k in ("X0","R0","A0")))
say("\n== GROUND 1b: reachability and cold by TEST SOURCE ==")
say("  source | n | median rating | median givens | " + " | ".join(f"{k} reach/cold" for k in S))
for s_ in np.unique(src):
    m = src == s_
    say(f"  {str(SN[s_])[:18]:18s} | {m.sum():5d} | {np.median(rat[m]):5.0f} | {np.median(g[m]):4.0f} | " + " | ".join(f"{100*reach(S[k])[m].mean():5.1f}/{100*S[k]['cold_exact'][m].mean():5.1f}" for k in S))
say("\n== GROUND 1c: the UNREACHABLE set of our RI maps (R0 ∪ A0 ∪ P3s1 at 128 draws each) and the field's coverage of it ==")
ours = reach(S["R0"]) | reach(S["A0"]) | reach(S["P3s1"]); unr = ~ours
say(f"  reachable by any of our three RI maps: {100*ours.mean():.2f}% | unreachable: {100*unr.mean():.2f}% (n={unr.sum()})")
say(f"  of the unreachable: X0 cold {100*S['X0']['cold_exact'][unr].mean():.2f}%, X0 reach {100*reach(S['X0'])[unr].mean():.2f}%; canvas C3X reach {100*reach(S['C3X'])[unr].mean():.2f}%, D4 reach {100*reach(S['D4'])[unr].mean():.2f}%")
say(f"  unreachable-set profile: median rating {np.median(rat[unr]):.0f} (reachable {np.median(rat[ours]):.0f}), median givens {np.median(g[unr]):.0f} (reachable {np.median(g[ours]):.0f}); by source: " + ", ".join(f"{str(SN[s_])[:10]} {100*unr[src==s_].mean():.0f}%" for s_ in np.unique(src)))
xf = ~S["X0"]["cold_exact"].astype(bool); say(f"  X0's cold failures (n={xf.sum()}, {100*xf.mean():.2f}%): inside our unreachable set {100*unr[xf].mean():.1f}% (base rate {100*unr.mean():.1f}%) | X0 unreachable@128 n={(~reach(S['X0'])).sum()}: inside ours {100*unr[~reach(S['X0'])].mean():.1f}%")
say(f"  X0 cold-failure profile: median rating {np.median(rat[xf]):.0f}, median givens {np.median(g[xf]):.0f}")
say("\n== GROUND 1d: per-draw rate r on reachable puzzles vs givens (does the erasure count set the per-draw rate?) ==")
for k in ("X0", "R0", "A0"):
    z = S[k]; rr = reach(z); fh = z["mi_first_hit"]; rows = []
    for lo, hi in bins:
        m = (g >= lo) & (g < hi) & rr & (fh >= 0)
        if m.sum() < 50: rows.append(f"[{lo},{hi}) -"); continue
        rows.append(f"[{lo},{hi}) {1/(1+np.mean(fh[m])):.3f}")   # geometric-rate estimate from mean first-hit draw
    say(f"  {k}: " + " | ".join(rows))
Path("runs/analysis/freethink_ground1_reachability_20260903.txt").write_text("\n".join(L) + "\n")
