"""EXPLORATORY diagnostic after the frozen Stage-1 read (post hoc; not part of the rule): with the same folds, seeds and ridge choices as the frozen
analysis, for B0 / B1 / M3 at k = 1: per-puzzle SSE of the primary within-dose target for (i) the frozen probe-calibrated predictions, (ii) the
uncalibrated predictions and (iii) predicting zero; puzzle-bootstrap intervals for the reductions; the distribution of the probe gains; and the
Stage-3 prototype without the probe gain."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path("/Users/aakash/Projects/HRRN"); sys.path.insert(0, str(ROOT / "tools")); import rebuttal_p32 as T
names = ["P26D", "P26C", "P27"]; S_of, meta = {}, {}
for nm in names:
    d = np.load(T.OUT / f"p32_features_{nm}_intact.npz", allow_pickle=False); S_of[nm] = len(d["state_pid"]); meta[nm] = d["state_pid"]
puz = np.concatenate([meta[nm] for nm in names]); arch = np.concatenate([[nm] * S_of[nm] for nm in names]); S = len(puz)
Mi = {nm: T.margins_from(T.ARCH[nm], ["intact"]) for nm in names}
Y = np.concatenate([(Mi[nm][:, 0, 1:, 0] - Mi[nm][:, 0, :1, 0]) for nm in names]); Yd = Y.reshape(S, T.ND, T.NM, T.NE).transpose(0, 1, 3, 2)
train_only = arch == "P26D"; scored = ~train_only; up = np.unique(puz[scored]); perm = np.random.default_rng([T.SEED, 1]).permutation(len(up))
ofold = {p: i % T.OUTER for i, p in enumerate(up[perm])}; of = np.array([ofold.get(p, -1) for p in puz])
T_obs = T.unmeasured_centered(Yd); ok = np.sqrt((T_obs ** 2).mean(3)) >= T.RMS_MIN
print(f"primary target: RMS {np.sqrt((T_obs[scored] ** 2).mean()):.2f}; |T| quantiles 50/90/99/max {np.percentile(np.abs(T_obs[scored]), [50, 90, 99, 100]).round(2).tolist()}")
per = {"zero": {p: float(((T_obs ** 2).sum(3) * ok)[puz == p].sum()) for p in up}}; gains = {}
for fam in ("B0", "B1", "M3"):
    X = T.load_block(names, "intact", T.FAM[fam]); p_ = X.shape[1]; Xd = X.reshape(S, T.ND, T.NM, T.NE, p_).transpose(0, 1, 3, 2, 4)
    Tc, Tu = np.zeros_like(T_obs), np.zeros_like(T_obs); gl = []
    for f in range(T.OUTER):
        tr, te = of != f, of == f; lamL, lamS = T.choose_lambda(Xd[tr], Yd[tr], puz[tr], ~train_only[tr], [T.SEED, 2, f, len(fam)])
        a = T.two_part(Xd[tr], Yd[tr]); fitL, fitS = T.ridge_fit(a[0], a[1], lamL, True), T.ridge_fit(a[2], a[3], lamS, False)
        Lt, St = T.predict(fitL, fitS, Xd[tr], p_); tau = float(np.median(np.einsum("sic,sic->sc", St[:, :, :, T.PROBE], St[:, :, :, T.PROBE])))
        Lp, Sp = T.predict(fitL, fitS, Xd[te], p_); gn = T.calib_gain(Lp, Sp, Yd[te], tau); gl.append(gn.ravel())
        Tc[te] = gn[:, None, :, None] * T.unmeasured_centered(Sp); Tu[te] = T.unmeasured_centered(Sp)
    for tag, P in (("cal", Tc), ("unc", Tu)): per[f"{fam}_{tag}"] = {p: float((((T_obs - P) ** 2).sum(3) * ok)[puz == p].sum()) for p in up}
    g = np.concatenate(gl); gains[fam] = np.percentile(g, [5, 25, 50, 75, 95]).round(2).tolist()
rng = np.random.default_rng([T.SEED, 9]); draws = [rng.choice(up, len(up)) for _ in range(2000)]
def red(a, b):
    pt = 1 - sum(per[a].values()) / sum(per[b].values()); bs = [1 - sum(per[a][q] for q in dr) / sum(per[b][q] for q in dr) for dr in draws]
    return f"{100 * pt:+.1f} % [{100 * np.percentile(bs, 2.5):+.1f}, {100 * np.percentile(bs, 97.5):+.1f}]"
for a, b in (("B0_unc", "zero"), ("B1_unc", "zero"), ("M3_unc", "zero"), ("M3_unc", "B1_unc"), ("M3_unc", "B0_unc"), ("B1_cal", "zero"), ("M3_cal", "zero"), ("M3_cal", "B1_cal")):
    print(f"{a} vs {b}: {red(a, b)}")
print("probe gains, 5/25/50/75/95 %:", gains)
