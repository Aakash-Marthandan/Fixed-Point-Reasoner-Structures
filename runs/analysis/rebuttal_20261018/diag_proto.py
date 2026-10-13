"""EXPLORATORY (post hoc): the Stage-3 prototype without the probe gain (g = 1), M3 at k = 1, same folds and ridge choices as the frozen analysis:
the predicted response change of each P27 hybrid relative to its intact recipient on the unmeasured masks, against zero change; slope; correlation."""
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
ofold = {p: i % T.OUTER for i, p in enumerate(up[perm])}; of = np.array([ofold.get(p, -1) for p in puz]); i27 = np.where(arch == "P27")[0]
Mh = T.margins_from(T.ARCH["P27"], ["intact"] + list(T.HYBRIDS)); Yh = (Mh[:, :, 1:, 0] - Mh[:, :, :1, 0]).reshape(len(i27), 1 + len(T.HYBRIDS), T.ND, T.NM, T.NE).transpose(0, 1, 2, 4, 3)
fam = "M3"; X = T.load_block(names, "intact", T.FAM[fam]); p_ = X.shape[1]; Xd = X.reshape(S, T.ND, T.NM, T.NE, p_).transpose(0, 1, 3, 2, 4)
Xh = {c: T.load_block(["P27"], c, T.FAM[fam]).reshape(len(i27), T.ND, T.NM, T.NE, p_).transpose(0, 1, 3, 2, 4) for c in T.HYBRIDS}
pc = {c: np.zeros((len(i27), T.ND, T.NE, T.NM - 1)) for c in T.HYBRIDS}; oc = {c: np.zeros_like(pc[c]) for c in T.HYBRIDS}
for f in range(T.OUTER):
    tr, te = of != f, of == f; lamL, lamS = T.choose_lambda(Xd[tr], Yd[tr], puz[tr], ~train_only[tr], [T.SEED, 2, f, len(fam)])
    a = T.two_part(Xd[tr], Yd[tr]); fitL, fitS = T.ridge_fit(a[0], a[1], lamL, True), T.ridge_fit(a[2], a[3], lamS, False)
    t27 = of[i27] == f; Lp, Sp = T.predict(fitL, fitS, Xd[i27][t27], p_); base = Lp[..., None] + np.delete(Sp, T.PROBE, 3)
    for ci, c in enumerate(T.HYBRIDS):
        Lc, Sc = T.predict(fitL, fitS, Xh[c][t27], p_); pc[c][t27] = (Lc[..., None] + np.delete(Sc, T.PROBE, 3)) - base
        oc[c][t27] = np.delete(Yh[t27, 1 + ci], T.PROBE, 3) - np.delete(Yh[t27, 0], T.PROBE, 3)
for c in T.HYBRIDS:
    pr_, ob = pc[c].ravel(), oc[c].ravel()
    print(f"{c:11s}: reduction vs zero {1 - ((ob - pr_) ** 2).sum() / (ob ** 2).sum():+.3f}, slope {(pr_ @ ob) / (pr_ @ pr_):+.2f}, corr {np.corrcoef(pr_, ob)[0, 1]:+.3f}, pred RMS {np.sqrt((pr_ ** 2).mean()):.2f}, obs RMS {np.sqrt((ob ** 2).mean()):.2f}")
for a_, b_ in (("lead", "lead_ctrl"), ("full", "comb_ctrl")):
    pr_, ob = (pc[a_] - pc[b_]).ravel(), (oc[a_] - oc[b_]).ravel()
    print(f"{a_}-{b_}: reduction vs zero {1 - ((ob - pr_) ** 2).sum() / (ob ** 2).sum():+.3f}, slope {(pr_ @ ob) / (pr_ @ pr_):+.2f}, corr {np.corrcoef(pr_, ob)[0, 1]:+.3f}")
