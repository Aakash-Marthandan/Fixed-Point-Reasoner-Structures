"""P28 exploratory additions (after the frozen report; labeled exploratory): per-rollout success rates of every strategy, the oracle headroom
and unique coverage for every candidate set (is the selection opportunity transplant-specific?), the union coverage, and the paired
transplant-vs-control comparison at the rollout level."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path("/Users/aakash/Projects/HRRN"); sys.path.insert(0, str(ROOT / "tools")); import rebuttal_p28 as T28
B = ROOT / "runs/analysis/rebuttal_20261014"
for w in (192, 256):
    d = np.load(B / f"attention_{w}/p28.npz", allow_pickle=False); pp = T28.per_puzzle(d); nP = len(pp["pids"]); cov = pp["cov"]; rng = lambda k: [20261014, 28, w, 500 + k]
    ci = lambda v, k: T28.boot(lambda ix: float(np.mean(v[ix])), nP, rng(k))
    rates = {a: pp["S"][a].mean(1) for a in T28.ARMS}; rates["restart"] = pp["R"].mean(1); rates["unchanged_short"] = pp["SH"].mean(1)
    print(f"=== attention_{w} (eligible {nP})")
    print("   per-rollout success: " + " | ".join(f"{a} {np.mean(v):.3f}" for a, v in rates.items()))
    alts = ["lead", "lead_ctrl", "full", "full_ctrl", "restart", "extended"]
    for a in alts:
        others = [b for b in ("restart", "extended", "lead_ctrl" if a != "lead_ctrl" else "lead") if b != a]
        uni = cov[a][12] & ~np.logical_or.reduce([cov[b][12] for b in others]); hd = cov[a][12].astype(float) - np.maximum.reduce([cov[b][1] for b in alts]).astype(float)
        print(f"   {a}: coverage {np.mean(cov[a][12]):.3f} | unique vs {others}: {np.mean(uni):.3f} {np.round(ci(uni.astype(float), 1), 3)} | oracle headroom over the best budget-1 strategy {np.mean(hd):+.3f}")
    union = np.logical_or.reduce([cov[a][12] for a in alts]); print(f"   union coverage of all six strategies: {np.mean(union):.3f}; puzzles solved by none: {int((~union).sum())}")
    dl = rates["lead"] - rates["lead_ctrl"]; print(f"   rollout-level lead - lead_ctrl success rate {np.mean(dl):+.3f} {np.round(ci(dl, 2), 3)}; full - full_ctrl {np.mean(rates['full'] - rates['full_ctrl']):+.3f} {np.round(ci(rates['full'] - rates['full_ctrl'], 3), 3)}")
