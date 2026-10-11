"""Reproduce the external review's P28 checks (exploratory): equal-work coverage counts, the transplant-only / restart-only discordance, the
equal-work comparison at a short horizon (48 iterations of work), and the headroom comparators (the reader's per-puzzle maximum over three
budget-1 strategies vs the same over six)."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path("/Users/aakash/Projects/HRRN"); sys.path.insert(0, str(ROOT / "tools")); import rebuttal_p28 as T28
B = ROOT / "runs/analysis/rebuttal_20261014"
for w in (192, 256):
    d = np.load(B / f"attention_{w}/p28.npz", allow_pickle=False); pp = T28.per_puzzle(d); cov = pp["cov"]; nP = len(pp["pids"])
    print(f"=== attention_{w}: " + " | ".join(f"{a} {int(cov[a][12].sum())}/{nP}" for a in ("lead", "lead_ctrl", "full", "full_ctrl", "restart", "extended")))
    sh = pp["short"]; print(f"   lead vs short: additional {int((cov['lead'][12] & ~sh).sum())}, missed {int((sh & ~cov['lead'][12]).sum())}")
    print(f"   lead vs restart: transplant-only {int((cov['lead'][12] & ~cov['restart'][12]).sum())}, restart-only {int((cov['restart'][12] & ~cov['lead'][12]).sum())}")
    # equal work at 48 iterations: 12 candidates x 4 iterations, or 4 originals x 12 iterations
    kind, p, first = d["kind"], d["p"], d["first"]
    def cov_h(k_, h): return np.array([((kind == k_) & (p == q) & (first >= 0) & (first < h)).any() for q in range(nP)])
    EX = pp["EX"]; ext12 = ((EX >= 0) & (EX < 12)).any(1)
    print(f"   48 iterations of work: lead x4 {int(cov_h('lead', 4).sum())} | lead_ctrl x4 {int(cov_h('lead_ctrl', 4).sum())} | restart x4 {int(cov_h('restart', 4).sum())} | extended x12 {int(ext12.sum())}")
    alts3 = np.maximum.reduce([cov["restart"][1], cov["extended"][1], cov["lead"][1]]); alts6 = np.maximum.reduce([cov[a][1] for a in ("lead", "lead_ctrl", "full", "full_ctrl", "restart", "extended")])
    best_fixed = max(float(np.mean(cov[a][1])) for a in ("lead", "lead_ctrl", "full", "full_ctrl", "restart", "extended"))
    for a in ("lead", "restart"):
        o = cov[a][12].astype(float); print(f"   {a}: oracle {o.mean():.3f} | headroom vs the reader's 3-strategy per-puzzle max {np.mean(o - alts3):+.3f} | vs 6-strategy per-puzzle max {np.mean(o - alts6):+.3f} | vs the best fixed budget-1 strategy ({best_fixed:.3f}) {o.mean() - best_fixed:+.3f}")
