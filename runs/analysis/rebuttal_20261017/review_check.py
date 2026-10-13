"""Exploratory reproduction of the PI's review of P31 (2026-10-13), own specification where the review gave only a description:
(1) P31 without P29's fragile task (b9630600); (2) correctness variation across masks; (3) a response-gain-adjusted contrast: within each
(history, dose) group the condition's centered responses may be any multiple of the recipient's (one free gain per group and condition), the
donor coefficient is estimated beyond that gain (profiled: the donor's responses residualized on the recipient's within each group);
(4) held-out prediction of the full transplant's centered responses: recipient-only (gain) vs recipient + donor, whole tasks held out (8 folds),
relative MSE reduction with a task bootstrap; and coefficients fitted on P29 applied to P31."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path("/Users/aakash/Projects/HRRN"); sys.path.insert(0, str(ROOT / "tools")); import rebuttal_p29 as T, rebuttal_p31 as T31


def setup(base, avg):
    d = T.load("conf", base); conds = [str(x) for x in d["conds"]]; ix, jf = conds.index("intact"), conds.index("full"); don = T.donor_map(d)
    task = np.array([str(x) for x in d["state_task"]]); Y, B, dk, _ = T.tables(d, "margin_std", 4, 1, don)
    if avg: Y, B = T31.with_avg(Y, B, conds); jc = T31.AVG
    else: jc = conds.index("full_ctrl")
    return d, conds, ix, jf, jc, dk, task, Y


def arrays(Y, dk, ix, jx):
    """Per (state, dose): centered 4-vectors of the condition, donor intact, recipient intact; and the state."""
    X, D, R, S = [], [], [], []
    for s in sorted(dk):
        if dk[s] is None: continue
        for i in range(3):
            g = lambda s_, c: np.array([Y[(s_, c, i, m)] for m in range(4)]); cen = lambda v: v - v.mean()
            X.append(cen(g(s, jx))); D.append(cen(g(dk[s], ix))); R.append(cen(g(s, ix))); S.append(s)
    return np.array(X), np.array(D), np.array(R), np.array(S)


def gain_beta(X, D, R, w=None):
    """Donor coefficient with a free recipient gain per group: residualize D and X on R within each group, pool."""
    rr = (R * R).sum(1, keepdims=True); pr = lambda A: A - np.where(rr > 1e-12, (A * R).sum(1, keepdims=True) / np.maximum(rr, 1e-300), 0.0) * R
    rD, rX = pr(D), pr(X); w = np.ones(len(X)) if w is None else w
    return float((w[:, None] * rD * rX).sum() / (w[:, None] * rD * rD).sum())


for name, base, avg in (("P31", T31.OUT, True), ("P29", T.OUT, False)):
    d, conds, ix, jf, jc, dk, task, Y = setup(base, avg)
    XF, D, R, S = arrays(Y, dk, ix, jf); XC, _, _, _ = arrays(Y, dk, ix, jc); tk = task[S]; ut = np.unique(tk); idx = {u: np.where(tk == u)[0] for u in ut}
    g = gain_beta(XF, D, R) - gain_beta(XC, D, R); rng = np.random.default_rng(13); bs = []
    for _ in range(20000):
        cnt = np.bincount(np.searchsorted(ut, rng.choice(ut, len(ut))), minlength=len(ut)).astype(float); w = cnt[np.searchsorted(ut, tk)]
        bs.append(gain_beta(XF, D, R, w) - gain_beta(XC, D, R, w))
    print(f"{name}: gain-adjusted full - control {g:+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]")
    if name == "P31":
        uF, uC = T.centered_units(Y, dk, jf, ix, ix), T.centered_units(Y, dk, jc, ix, ix); tkr = task[uF["state"]]; m = tkr != "b9630600"; cl = np.unique(tkr, return_inverse=True)[1]
        sub = lambda u: {k: v[m] for k, v in u.items()}; dr = T.boot_betas([sub(uF), sub(uC)], [1.0, -1.0], [cl[m], cl[m]], [13, 1], 20000); lo, hi = T.ci(dr)
        print(f"P31 without b9630600 (present: {'b9630600' in set(tkr)}): diff {T.beta(sub(uF)) - T.beta(sub(uC)):+.3f} [{lo:+.3f}, {hi:+.3f}]")
        Yc, _, dkc, _ = T.tables(d, "correct", 4, 1, T.donor_map(d)); n = v = 0
        for s in sorted(dkc):
            if dkc[s] is None: continue
            for i in range(3): n += 1; v += np.ptp([Yc[(s, ix, i, mm)] for mm in range(4)]) > 1e-12
        print(f"P31 correctness: groups varying across masks {v}/{n}")
    # held-out prediction of the full transplant's centered responses (rows = state x dose x mask)
    y = XF.reshape(-1); r = R.reshape(-1); dd = D.reshape(-1); grp = np.repeat(np.arange(len(XF)), 4); tr_ = np.repeat(tk, 4)
    folds = T.grouped_folds(tr_, 8, [13, 2]) if hasattr(T, "grouped_folds") else None
    if folds is None:
        uu = np.unique(tr_); perm = np.random.default_rng([13, 2]).permutation(len(uu)); fm = {u: i % 8 for i, u in enumerate(uu[perm])}; folds = np.array([fm[u] for u in tr_])
    eA, eB = np.zeros(len(y)), np.zeros(len(y))
    for f in range(8):
        a, b = folds != f, folds == f
        cA = np.linalg.lstsq(r[a, None], y[a], rcond=None)[0]; cB = np.linalg.lstsq(np.column_stack([r[a], dd[a]]), y[a], rcond=None)[0]
        eA[b] = (y[b] - r[b] * cA[0]) ** 2; eB[b] = (y[b] - np.column_stack([r[b], dd[b]]) @ cB) ** 2
    red = 1 - eB.sum() / eA.sum(); uu = np.unique(tr_); ii = {u: np.where(tr_ == u)[0] for u in uu}; rng = np.random.default_rng(14); bs = []
    for _ in range(4000):
        p = np.concatenate([ii[u] for u in rng.choice(uu, len(uu))]); bs.append(1 - eB[p].sum() / eA[p].sum())
    print(f"{name}: held-out MSE reduction from adding the donor response {100 * red:+.1f} % [{100 * np.percentile(bs, 2.5):+.1f}, {100 * np.percentile(bs, 97.5):+.1f}]")
    if name == "P31": keep31 = (y, r, dd)
    else:
        cB29 = np.linalg.lstsq(np.column_stack([r, dd]), y, rcond=None)[0]; cA29 = np.linalg.lstsq(r[:, None], y, rcond=None)[0]; y3, r3, d3 = keep31
        print(f"P29 -> P31 transfer: MSE reduction {100 * (1 - ((y3 - np.column_stack([r3, d3]) @ cB29) ** 2).sum() / ((y3 - r3 * cA29[0]) ** 2).sum()):+.1f} % (P29 coefficients: recipient {cB29[0]:+.3f}, donor {cB29[1]:+.3f})")
