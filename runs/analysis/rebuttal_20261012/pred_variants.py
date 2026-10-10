"""Which predictive specification gives the review's ~1.0 % / 0.6 %? Fit on discovery, score on confirmation; MSE reduction from adding the donor term."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path("/Users/aakash/Projects/HRRN"); sys.path.insert(0, str(ROOT / "tools")); import rebuttal_p26 as T
B = ROOT / "runs/analysis/rebuttal_20261012"; SEED = 20261012
def fitpred(Xtr, ytr, Xte): Xtr, Xte = np.column_stack([Xtr, np.ones(len(Xtr))]), np.column_stack([Xte, np.ones(len(Xte))]); return Xte @ np.linalg.lstsq(Xtr, ytr, rcond=None)[0]
def gain(tr, te, ycol, base, extra):
    g = lambda rows, cols: np.column_stack([np.array([r[c] for r in rows]) for c in cols]); ytr, yte = np.array([r[ycol] for r in tr]), np.array([r[ycol] for r in te])
    e1 = np.mean((yte - fitpred(g(tr, base + extra), ytr, g(te, base + extra))) ** 2); e0 = np.mean((yte - fitpred(g(tr, base), ytr, g(te, base))) ** 2); return 1 - e1 / e0
def rows_r2(d):
    out = []
    for r in T.r2_rows(d, "full", "full_random"): out.append(dict(r))
    a = T.r2_rows(d, "full", "intact")                         # dep = d_full - d_R
    for o, x in zip(out, a): o["dfull"] = x["dep"] + x["dR"]
    return out
def rows_r1(d):
    out = []
    for x, y in zip(T.r1_rows(d, "full", "full_random"), T.r1_rows(d, "full", "full_random")):
        o = dict(x); o["Yfull"] = x["Yx"] + x["bx"]; o["diff"] = x["Yx"] - x["Yref"]; o["fR2"] = x["fR"] ** 2; o["fD2"] = x["fD"] ** 2; o["bxfR"] = x["bx"] * x["fR"]; o["bRfR"] = x["bR"] * x["fR"]; o["bDfD"] = x["bD"] * x["fD"]; out.append(o)
    return out
NUIS = ["YR", "bx", "bR", "bD", "fR", "fD", "fR2", "fD2", "bxfR", "bRfR", "bDfD"]
for w in (192, 256):
    dD, dC = (np.load(B / f"attention_{w}/p26_{s}.npz", allow_pickle=False) for s in ("D", "C"))
    a, b = rows_r2(dD), rows_r2(dC); c, e = rows_r1(dD), rows_r1(dC)
    specs = {"R2 target d_full on d_R (+ d_D)": (a, b, "dfull", ["dR"], ["dD"]), "R2 target d_full - d_random on d_R (+ d_D)": (a, b, "dep", ["dR"], ["dD"]),
             "R1 target Y_full - b_full on nuisance (+ Y_D - b_D)": (c, e, "Yx", NUIS, ["YD"]), "raw Y_full on nuisance (+ Y_D - b_D)": (c, e, "Yfull", NUIS, ["YD"]),
             "R1 differential (full - random responses) on nuisance (+ Y_D - b_D)": (c, e, "diff", NUIS, ["YD"]),
             "raw Y_full on Y_R only (+ Y_D)": (c, e, "Yfull", ["YR"], ["YD"])}
    for k, (tr, te, y, base, ex) in specs.items():
        g0 = gain(tr, te, y, base, ex); ci = T.boot(lambda q: gain(tr, q, y, base, ex), [te], [SEED, 26, w, 50], n=1000)
        print(f"attention_{w}: {k}: {100 * g0:+.2f} % [{100 * ci[0]:+.2f}, {100 * ci[1]:+.2f}]")
