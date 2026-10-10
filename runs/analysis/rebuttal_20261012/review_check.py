"""Reproduce the external review's P26 checks (exploratory): per-condition donor coefficients, the direct full-vs-common contrast, four-mask
centering, out-of-sample predictive value of the donor response, and the mean repair change from full transplantation."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path("/Users/aakash/Projects/HRRN"); sys.path.insert(0, str(ROOT / "tools")); import rebuttal_p26 as T
B = ROOT / "runs/analysis/rebuttal_20261012"; SEED = 20261012
def tables(d):
    Y, Bm = T.unit_tables(d); return Y, Bm
def centered_rows(d, cx, cr):
    Y, Bm = T.unit_tables(d); don = T.donor_state(d); conds = [str(x) for x in d["conds"]]; ix = 0; rows = []
    for s in range(len(d["state_pid"])):
        ds = don[s]
        for i in range(3):
            g = lambda st_, c: np.array([Y[(st_, c, i, m)] for m in range(4)]); cen = lambda v: v - v.mean()
            dep = cen(g(s, conds.index(cx))) - cen(g(s, conds.index(cr))); dD = cen(g(ds, ix)); dR = cen(g(s, ix))
            for m in range(4): rows.append(dict(pid=int(d["state_pid"][s]), dep=dep[m], dD=dD[m], dR=dR[m]))
    return rows
def r1_design(rows, which, with_donor=True):
    col = lambda k_: np.array([r[k_] for r in rows]); b = col("b" + which); fR, fD, bR, bD = col("fR"), col("fD"), col("bR"), col("bD")
    cols = ([col("YD")] if with_donor else []) + [col("YR"), b, bR, bD, fR, fD, fR ** 2, fD ** 2, b * fR, bR * fR, bD * fD]
    return np.column_stack(cols + [np.ones(len(rows))]), col("Y" + which)
for w in (192, 256):
    dC = np.load(B / f"attention_{w}/p26_C.npz", allow_pickle=False); dD = np.load(B / f"attention_{w}/p26_D.npz", allow_pickle=False)
    print(f"=== attention_{w} (confirmation)")
    print("   donor coefficient per condition (R2 contrasts, vs intact): " + ", ".join(f"{c} {T.r2_beta(T.r2_rows(dC, c, 'intact')):+.3f}" for c in T.COND_C[1:]))
    r = T.r2_rows(dC, "full", "global"); print(f"   direct full - common: {T.r2_beta(r):+.3f} {np.round(T.boot(T.r2_beta, [r], [SEED, 26, w, 40]), 3)}")
    r = T.r2_rows(dC, "global_rand", "full_random"); print(f"   control geometry (common-basis random control - full random control): {T.r2_beta(r):+.3f} {np.round(T.boot(T.r2_beta, [r], [SEED, 26, w, 41]), 3)}")
    r = centered_rows(dC, "full", "full_random"); print(f"   four-mask centering, full - random: {T.r2_beta(r):+.3f} {np.round(T.boot(T.r2_beta, [r], [SEED, 26, w, 42]), 3)}")
    en = json.loads(str(dC["energies"])); sc = np.sqrt(np.array([e["energy_global"] / e["energy_random"] for e in en])); sa = np.sqrt(np.array([e["energy_adaptive"] / e["energy_random"] for e in en]))
    print(f"   random-control rescaling factor: common {sc.mean():.2f} (range {sc.min():.2f}-{sc.max():.2f}), adaptive {sa.mean():.2f}")
    # predictive value: the R1 model for `full`, fit on discovery, scored on confirmation
    rD, rC = T.r1_rows(dD, "full", "full_random"), T.r1_rows(dC, "full", "full_random")
    def mse_gain(rows_test, coefs):
        out = []
        for wd, cf in zip((True, False), coefs):
            X, y = r1_design(rows_test, "x", wd); out.append(np.mean((y - X @ cf) ** 2))
        return 1 - out[0] / out[1]
    coefs = [np.linalg.lstsq(*r1_design(rD, "x", wd), rcond=None)[0] for wd in (True, False)]
    print(f"   out-of-sample MSE reduction from the donor term: {100 * mse_gain(rC, coefs):.2f} % {np.round(100 * np.array(T.boot(lambda q: mse_gain(q, coefs), [rC], [SEED, 26, w, 43])), 2)}")
    # mean repair change from indiscriminate full transplantation (untouched cells, k = 1)
    Y, Bm = T.unit_tables(dC); conds = [str(x) for x in dC["conds"]]; rows = []
    for s in range(len(dC["state_pid"])):
        rows.append(dict(pid=int(dC["state_pid"][s]), e=np.mean([Y[(s, 1, i, m)] - Y[(s, 0, i, m)] for i in range(3) for m in range(4)]), n=Bm[(s, 1)] - Bm[(s, 0)]))
    fe = lambda q: float(np.mean([x["e"] for x in q])); fn = lambda q: float(np.mean([x["n"] for x in q]))
    print(f"   full - intact repair (pp): corrected arms {100 * fe(rows):+.1f} {np.round(100 * np.array(T.boot(fe, [rows], [SEED, 26, w, 44])), 1)} | no-edit {100 * fn(rows):+.1f} {np.round(100 * np.array(T.boot(fn, [rows], [SEED, 26, w, 45])), 1)}")
