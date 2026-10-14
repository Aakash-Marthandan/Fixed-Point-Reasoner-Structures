"""EXPLORATORY reproduction of the PI's review of P32 Stage 1 (2026-10-14), own code over the Stage-1 tool's frozen folds and ridge choices:
(a) groups whose unmeasured masks disagree on whether the untouched cell becomes correct, their share of the target's squared magnitude, and the top-10 %
share; (b) hidden features removed by within-dose centering; (c) a post hoc ranking diagnostic: per (history, dose) pick one of the three unmeasured masks
by the uncalibrated predicted mean response over the four untouched cells, and score correctness on those cells; pairwise ranking of discordant masks."""
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
# correctness after each mask at k = 1: the evaluation cell's displayed digit (argmax of its scores) equals the solution
C = []
for nm in names:
    d = np.load(T.ARCH[nm], allow_pickle=False); conds = [str(x) for x in d["conds"]]; ix = conds.index("intact"); Cn = np.zeros((S_of[nm], T.ND * T.NM, T.NE), bool)
    sol, E = d["state_sol"].astype(int), d["state_E"].astype(int)
    for v in np.where((d["var_cond"] == ix) & (d["var_arm"] == 1))[0]:
        s = int(d["var_state"][v]); Cn[s, int(d["var_dose"][v]) * T.NM + int(d["var_mask"][v])] = d["escores"][v][0].argmax(-1) + 1 == sol[s, E[s]]
    C.append(Cn)
Cd = np.concatenate(C).reshape(S, T.ND, T.NM, T.NE).transpose(0, 1, 3, 2)                        # (S, dose, cell, mask)
train_only = arch == "P26D"; scored = ~train_only; up = np.unique(puz[scored]); perm = np.random.default_rng([T.SEED, 1]).permutation(len(up))
ofold = {p: i % T.OUTER for i, p in enumerate(up[perm])}; of = np.array([ofold.get(p, -1) for p in puz])
To = T.unmeasured_centered(Yd); Cu = np.delete(Cd, T.PROBE, 3)
disc = (Cu.min(3) != Cu.max(3)); sq = (To ** 2).sum(3); a = np.abs(To[scored]).ravel(); q90 = np.percentile(a, 90)
print(f"(a) scored groups with unmeasured masks disagreeing on correctness: {disc[scored].mean():.3f}; their share of sum T^2: {sq[scored][disc[scored]].sum() / sq[scored].sum():.3f}; "
      f"top-10 % |T| share of sum T^2: {(a[a >= q90] ** 2).sum() / (a ** 2).sum():.3f}")
names_H = ["ze"] * 24 + ["zbar"] * 24 + ["dzr"] * 24 + ["dzu"] * 24 + ["gdisp"] * 24 + ["gall"] * 24 + ["ze*nrel"] * 24 + ["ip"] * 6 + ["|ze|", "dzn"]
H = T.load_block(names, "intact", ["H"]).reshape(S, T.ND, T.NM, T.NE, -1).transpose(0, 1, 3, 2, 4); within = (H - H.mean(3, keepdims=True))
const = np.abs(within).max((0, 1, 2, 3)) < 1e-12; print(f"(b) hidden features constant within (state, dose, cell): {int(const.sum())} of {H.shape[-1]} ({sorted(set(n for n, c in zip(names_H, const) if c))})")
res = {}
for fam in ("B1", "M3"):
    X = T.load_block(names, "intact", T.FAM[fam]); p_ = X.shape[1]; Xd = X.reshape(S, T.ND, T.NM, T.NE, p_).transpose(0, 1, 3, 2, 4); Sp_all = np.zeros(Yd.shape)
    for f in range(T.OUTER):
        tr, te = of != f, of == f; lamL, lamS = T.choose_lambda(Xd[tr], Yd[tr], puz[tr], ~train_only[tr], [T.SEED, 2, f, len(fam)])
        a2 = T.two_part(Xd[tr], Yd[tr]); fitL, fitS = T.ridge_fit(a2[0], a2[1], lamL, True), T.ridge_fit(a2[2], a2[3], lamS, False); _, Sp = T.predict(fitL, fitS, Xd[te], p_); Sp_all[te] = Sp
    res[fam] = np.delete(Sp_all, T.PROBE, 3)                                                        # (S, dose, cell, 3 unmeasured)
corr_u = Cu.astype(float)                                                                            # (S, dose, cell, 3)
pick = {fam: res[fam].mean(2).argmax(-1) for fam in res}                                            # per (state, dose): the mask with the highest mean predicted response
val = {fam: np.take_along_axis(corr_u.mean(2), pick[fam][..., None], -1)[..., 0] for fam in res}; val["random"] = corr_u.mean(2).mean(-1)
per = {k: {p: float(v[(puz == p)].mean()) for p in up} for k, v in val.items()}
rng = np.random.default_rng(14); draws = [rng.choice(up, len(up)) for _ in range(2000)]
for k in ("random", "B1", "M3"): print(f"(c) selection {k:6s}: untouched-cell correctness {100 * np.mean([per[k][p] for p in up]):.2f} %")
for a_, b_ in (("M3", "random"), ("M3", "B1"), ("B1", "random")):
    pt = np.mean([per[a_][p] - per[b_][p] for p in up]); bs = [np.mean([per[a_][p] - per[b_][p] for p in dr]) for dr in draws]
    print(f"    {a_} - {b_}: {100 * pt:+.2f} pp [{100 * np.percentile(bs, 2.5):+.2f}, {100 * np.percentile(bs, 97.5):+.2f}]")
for fam in ("B1", "M3"):                                                                             # discordant pairs at a cell: the correcting mask ranked higher?
    num = den = 0.0
    for s in np.where(scored)[0]:
        for i in range(T.ND):
            for j in range(T.NE):
                c, pr_ = corr_u[s, i, j], res[fam][s, i, j]
                for u in range(3):
                    for v in range(u + 1, 3):
                        if c[u] != c[v]: den += 1; w = u if c[u] > c[v] else v; l_ = v if w == u else u; num += 1.0 if pr_[w] > pr_[l_] else (0.5 if pr_[w] == pr_[l_] else 0.0)
    print(f"    pairwise: {fam} ranks the correcting mask higher in {100 * num / den:.1f} % of {int(den)} discordant pairs")
