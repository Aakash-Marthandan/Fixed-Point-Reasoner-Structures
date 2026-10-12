"""P29 independent recount (own code; imports nothing from tools/rebuttal_p29.py). From p29_conf.npz's raw arrays: the evaluation-token margin
(label logit minus the largest other logit, mean over the four E tokens, k = 1), option B's per-query scale (RMS of both histories' centered intact
responses), the four-mask centered rows by whole-array operations, QR least squares, the donor coefficients, and an independent task-clustered
resampler (its own seed; rows gathered by task index lists). Compares point estimates exactly and intervals approximately with the report."""
import json, sys
from pathlib import Path
import numpy as np

B = Path(__file__).resolve().parent
d = np.load(B / "p29_conf.npz", allow_pickle=False); rep = json.loads((B / "report_conf.json").read_text())
conds = [str(x) for x in d["conds"]]; I, F, C = conds.index("intact"), conds.index("full"), conds.index("full_ctrl")
vs, vc, va, vd, vm = d["var_state"], d["var_cond"], d["var_arm"], d["var_dose"], d["var_mask"]
qi, sdir, task = d["state_qi"], d["state_dir"], np.array([str(x) for x in d["state_task"]])
Et = d["state_Etok"][:, :4].astype(int); lab = d["state_lab"].astype(int); S = len(qi)
labE = lab[np.arange(S)[:, None], Et]                                                      # (S, 4)
el = d["elog"][:, 0, :4, :].astype(np.float64)                                             # (V, 4, 12) at k = 1
L = labE[vs]                                                                               # (V, 4)
lablog = np.take_along_axis(el, L[..., None], -1)[..., 0]; oth = el.copy(); np.put_along_axis(oth, L[..., None], -np.inf, -1)
y = (lablog - oth.max(-1)).mean(-1)                                                        # (V,)

# arrays Y[state, cond, dose, mask]
Y = np.full((S, len(conds), 3, 4), np.nan); m1 = va == 1; Y[vs[m1], vc[m1], vd[m1], vm[m1]] = y[m1]
assert not np.isnan(Y).any()
cen = Y - Y.mean(-1, keepdims=True)
partner = {}
for s in range(S):
    o = np.where((qi == qi[s]) & (sdir == 1 - sdir[s]))[0]; partner[s] = int(o[0]) if len(o) else -1
scale = {}
for q in np.unique(qi):
    ss = np.where(qi == q)[0]; scale[int(q)] = float(np.sqrt(np.mean(cen[ss, I] ** 2)))
keep = np.array([partner[s] >= 0 and scale[int(qi[s])] >= 0.01 for s in range(S)])
sc = np.array([scale[int(q)] for q in qi]); Z = cen / sc[:, None, None, None]                 # standardized centered responses
ss = np.where(keep)[0]; dn = np.array([partner[s] for s in ss])


def rows(X):
    dep = (Z[ss, X] - Z[ss, I]).reshape(-1); cD = Z[dn, I].reshape(-1); cR = Z[ss, I].reshape(-1); t = np.repeat(task[ss], 12); return dep, cD, cR, t


def coef(dep, cD, cR):
    A = np.column_stack([cD, cR, np.ones(len(dep))]); Q, R = np.linalg.qr(A)
    if abs(R[0, 0]) < 1e-300 or np.linalg.matrix_rank(A) < 3: return np.nan
    return float(np.linalg.solve(R, Q.T @ dep)[0])


dF, cD, cR, t = rows(F); dC, _, _, _ = rows(C)
bF, bC = coef(dF, cD, cR), coef(dC, cD, cR)
print(f"recount: states {len(ss)} (excluded small-scale queries {sum(1 for v in scale.values() if v < 0.01)}); beta(full) {bF:+.6f} beta(ctrl) {bC:+.6f} diff {bF - bC:+.6f}")
r1 = rep["margin_std|k1"]; print(f"report : beta(full) {r1['beta_full']:+.6f} beta(ctrl) {r1['beta_ctrl']:+.6f} diff {r1['diff']:+.6f}")
assert abs(bF - r1["beta_full"]) < 1e-9 and abs(bC - r1["beta_ctrl"]) < 1e-9, "point estimates differ"

ut = np.unique(t); idx = {u: np.where(t == u)[0] for u in ut}; rng = np.random.default_rng(9029); n = 20000; dF_, dD_ = np.empty(n), np.empty(n)
for b in range(n):
    pick = np.concatenate([idx[u] for u in rng.choice(ut, len(ut))])
    f = coef(dF[pick], cD[pick], cR[pick]); c = coef(dC[pick], cD[pick], cR[pick]); dF_[b], dD_[b] = f, f - c
q = lambda a: (np.nanpercentile(a, 2.5), np.nanpercentile(a, 97.5), int(np.isnan(a).sum()))
print(f"recount intervals: beta(full) [{q(dF_)[0]:+.3f}, {q(dF_)[1]:+.3f}] (unidentified {q(dF_)[2]}), diff [{q(dD_)[0]:+.3f}, {q(dD_)[1]:+.3f}] (unidentified {q(dD_)[2]})")
print(f"report intervals : beta(full) [{r1['ci_full'][0]:+.3f}, {r1['ci_full'][1]:+.3f}], diff [{r1['ci_diff'][0]:+.3f}, {r1['ci_diff'][1]:+.3f}]")
# the task-influence pieces: the top-information task (intact-only residual donor information) and the leave-one-task-out ranges
A = np.column_stack([cR, np.ones(len(cR))]); rD = cD - A @ np.linalg.lstsq(A, cD, rcond=None)[0]; share = {u: float((rD[idx[u]] ** 2).sum() / (rD @ rD)) for u in ut}
top = max(share, key=share.get); m = t != top
print(f"top-information task {top} ({share[top]:.3f}); without it beta(full) {coef(dF[m], cD[m], cR[m]):+.4f}, diff {coef(dF[m], cD[m], cR[m]) - coef(dC[m], cD[m], cR[m]):+.4f}"
      + (f" | report: {rep['omit_top_task']['task']} {rep['omit_top_task']['beta_full']:+.4f} {rep['omit_top_task']['diff']:+.4f}" if "omit_top_task" in rep else ""))
jk = np.array([[coef(dF[t != u], cD[t != u], cR[t != u]), coef(dC[t != u], cD[t != u], cR[t != u])] for u in ut])
print(f"leave-one-task-out: beta(full) [{np.nanmin(jk[:, 0]):+.4f}, {np.nanmax(jk[:, 0]):+.4f}], diff [{np.nanmin(jk[:, 0] - jk[:, 1]):+.4f}, {np.nanmax(jk[:, 0] - jk[:, 1]):+.4f}]"
      + (f" | report: {rep['jackknife']['beta_full']}, {rep['jackknife']['diff']}" if "jackknife" in rep else ""))
