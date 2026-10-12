"""Exploratory: the concentration-aware precision bracket of pilot_concentration.py computed on P27's intact rows (already read; Sudoku),
against P27's realized four-mask bootstrap SE for S (full vs P26's control) and for E_total (full vs the combined control)."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path("/Users/aakash/Projects/HRRN"); sys.path.insert(0, str(ROOT / "tools")); import rebuttal_p26 as T, rebuttal_p27 as P27
for w in (192, 256):
    d = np.load(ROOT / f"runs/analysis/rebuttal_20261013/attention_{w}/p27.npz", allow_pickle=False); conds = [str(x) for x in d["conds"]]
    Y, Bm = T.unit_tables(d); don = T.donor_state(d); pids = d["state_pid"]; r = P27.centered_rows(Y, don, conds, pids, "full", "full_random")
    cD, cR, pid = r["dD"], r["dR"], r["pid"]; n = len(cD); Xr = np.column_stack([cR, np.ones(n)]); rD = cD - Xr @ np.linalg.lstsq(Xr, cD, rcond=None)[0]; S2 = rD @ rD
    v = np.repeat([np.mean((cR[i:i + 4] ** 2 + cD[i:i + 4] ** 2) / 2) for i in range(0, n, 4)], 4)
    lo = np.sqrt(sum((rD[pid == t] ** 2 * 2 * v[pid == t]).sum() for t in np.unique(pid))) / S2
    hi = np.sqrt(sum(((np.abs(rD[pid == t]) * np.sqrt(2 * v[pid == t])).sum()) ** 2 for t in np.unique(pid))) / S2
    rep = json.load(open(ROOT / "runs/analysis/rebuttal_20261013/report.json"))[f"attention_{w}"]
    se = {k: (rep[k]["ci"][1] - rep[k]["ci"][0]) / 3.92 for k in ("S", "E_total")}
    sh = np.array(sorted([(rD[pid == t] ** 2).sum() / S2 for t in np.unique(pid)], reverse=True))
    print(f"{w}: bracket [{lo:.4f}, {hi:.4f}] vs realized SE S {se['S']:.4f}, E_total {se['E_total']:.4f} | puzzles {len(np.unique(pid))}, top share {sh[0]:.3f}, effective {1 / (sh ** 2).sum():.1f}")
