#!/usr/bin/env python3
# Ledger: SEQUENCE A of the predictive-mechanism plan (Documentation/Note_2026-10-10_SeqA_Explanation_Comparison_Plan.md, written
# before any fit). EXPLORATORY, read-only analysis of the saved P16 / P18 / P19 banks; no inference, no training.
"""Sequence A: which explanation predicts the network's responses to corrections?
A1 (P19): on the cells neither draw edits, compare M0 state < M1 +amount < M2 +visible configuration < M3 +hidden quantities by
puzzle-grouped held-out log-loss and AUC; the paired identity test (does the configuration difference predict which draw repairs a cell);
the aggregate variance decomposition. A2 (P18): a closure cell's retention alone vs with its partners (and partial vs closure)."""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import rebuttal_p1p2 as P                                   # UNITS: the (81, 81) Sudoku peer matrix, self excluded

OUT = ROOT / "runs/analysis/seqA_20261010"
WIDTHS = (256, 192, 128)
SEED = 20261010
N_FOLDS, N_BOOT, L2 = 5, 1000, 1e-3
TIERS = {"M0": ["y0", "nwrong", "t"], "M1": ["f", "n_ed"], "M2": ["n_ed_peers", "n_conf", "n_unblock", "n_wrong_rem"], "M3": ["margin", "commit", "commit_ed_peers"]}
CONFIG = TIERS["M2"]
P18_KINDS = ("noedit", "closure", "random", "random2", "single", "partial", "large_closure", "large_random", "all_wrong")


# ---------------- small, selftested statistics ----------------
def logit_fit(X, y, l2=L2, iters=50):
    """L2-regularized logistic regression (intercept unpenalized) by Newton's method; X already standardized."""
    Xa = np.hstack([np.ones((len(X), 1)), X]); w = np.zeros(Xa.shape[1]); R = l2 * np.eye(Xa.shape[1]); R[0, 0] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Xa @ w, -30, 30))); g = Xa.T @ (p - y) + R @ w
        H = (Xa * (p * (1 - p))[:, None]).T @ Xa + R + 1e-9 * np.eye(len(w)); step = np.linalg.solve(H, g); w -= step
        if np.abs(step).max() < 1e-8: break
    return w


def logit_pred(X, w): return 1 / (1 + np.exp(-np.clip(np.hstack([np.ones((len(X), 1)), X]) @ w, -30, 30)))


def auc(score, y):
    y = np.asarray(y, bool); pos, neg = int(y.sum()), int((~y).sum())
    if pos == 0 or neg == 0: return float("nan")
    from scipy.stats import rankdata
    r = rankdata(score); return float((r[y].sum() - pos * (pos + 1) / 2) / (pos * neg))


def folds_by_group(groups, k=N_FOLDS, seed=SEED):
    ug = np.unique(groups); perm = np.random.default_rng(seed).permutation(len(ug)); f = {int(g): i % k for i, g in zip(range(len(ug)), ug[perm])}
    return np.array([f[int(g)] for g in groups])


def cv_predict(X, y, groups, k=N_FOLDS):
    """Held-out probabilities from puzzle-grouped k-fold CV (standardization fitted on each training fold)."""
    fold = folds_by_group(groups, k); p = np.zeros(len(y))
    for i in range(k):
        tr, te = fold != i, fold == i
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        w = logit_fit((X[tr] - mu) / sd, y[tr]); p[te] = logit_pred((X[te] - mu) / sd, w)
    return p


def rowloss(p, y): p = np.clip(p, 1e-6, 1 - 1e-6); return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def boot_diff(d, groups, n=N_BOOT, seed=SEED):
    """Puzzle-bootstrap 95 % interval of the mean of per-row values d (resampling puzzles)."""
    ug, inv = np.unique(groups, return_inverse=True); s = np.bincount(inv, weights=d, minlength=len(ug)); c = np.bincount(inv, minlength=len(ug))
    rng = np.random.default_rng(seed); vals = []
    for _ in range(n):
        idx = rng.integers(0, len(ug), len(ug)); vals.append(s[idx].sum() / max(c[idx].sum(), 1))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def mcnemar(b, c):
    from math import comb
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


# ---------------- data ----------------
def p16_state_features(width):
    feat = {}
    for f in sorted(glob.glob(str(ROOT / f"runs/analysis/rebuttal_20261002b/attention_{width}/edits/*.npz"))):
        if "smoke" in f: continue
        d = np.load(f, allow_pickle=False)
        for i, (sid, t) in enumerate(zip(d["state_ids"].tolist(), d["state_t"].tolist())): feat[(sid, t)] = (d["state_margin"][i].astype(np.float64), d["state_commit"][i].astype(np.float64))
    return feat


def a1_rows(width, k):
    """One row per (state, fraction, draw, cell in the common unedited set U)."""
    feat = p16_state_features(width); rows = []; meta = []
    for f in sorted(glob.glob(str(ROOT / f"runs/analysis/rebuttal_20261003/attention_{width}/sweep/*.npz"))):
        if "smoke" in f: continue
        d = np.load(f, allow_pickle=False); fr = np.round(d["var_frac"].astype(np.float64), 1)
        for s in range(len(d["state_ids"])):
            sid, t = int(d["state_ids"][s]), int(d["state_t"][s]); grid, sol, puz = d["state_grid"][s].astype(int), d["state_sol"][s].astype(int), d["state_puz"][s].astype(int)
            W = np.where((puz == 0) & (grid != sol))[0]; vs = np.where(d["var_state"] == s)[0]
            v0 = [v for v in vs if fr[v] == 0.0]; assert len(v0) == 1; pred0 = d["pred"][v0[0], k - 1].astype(int)
            margin, commit = feat[(sid, t)]
            for fv in sorted(set(fr[vs]) - {0.0}):
                dv = {int(d["var_draw"][v]): v for v in vs if fr[v] == fv}
                if set(dv) != {0, 1}: assert fv == 1.0, (fv, set(dv)); continue          # f = 1.0 has one draw (every wrong cell) and no unedited cell
                E = {dr: set(int(c) for c in d["var_cells"][v] if c >= 0) for dr, v in dv.items()}
                U = [int(c) for c in W if c not in E[0] and c not in E[1]]
                for dr, v in dv.items():
                    pred = d["pred"][v, k - 1].astype(int); Ed = np.array(sorted(E[dr]), int); edm = np.zeros(81, bool); edm[Ed] = True
                    remw = np.zeros(81, bool); remw[W] = True; remw[Ed] = False
                    for c in U:
                        pe = P.UNITS[c] & edm
                        n_conf = int(sum(1 for e in np.where(pe)[0] if sol[e] == grid[c]))
                        n_unb = int(sum(1 for e in np.where(pe)[0] if grid[e] == sol[c]))
                        cep = float(commit[pe].mean()) if pe.any() else float(commit[W].mean())
                        rows.append([int(pred0[c] == sol[c]), len(W), t, fv, len(Ed), int(pe.sum()), n_conf, n_unb, int((P.UNITS[c] & remw).sum()), margin[c], commit[c], cep, int(pred[c] == sol[c])])
                        meta.append((sid, t, fv, dr, c))
    cols = ["y0", "nwrong", "t", "f", "n_ed", "n_ed_peers", "n_conf", "n_unblock", "n_wrong_rem", "margin", "commit", "commit_ed_peers", "y"]
    return np.array(rows, np.float64), cols, np.array(meta)


def analyze_a1(width, k):
    X, cols, meta = a1_rows(width, k); ci = {c: i for i, c in enumerate(cols)}; y = X[:, ci["y"]]; groups = meta[:, 0].astype(int)
    res = dict(rows=int(len(y)), puzzles=int(len(np.unique(groups))), states=int(len(set(map(tuple, meta[:, :2].astype(int).tolist())))), base_rate=float(y.mean()))
    feats, P_, L = [], {}, {}
    for tier, add in TIERS.items():
        feats = feats + add; Xt = X[:, [ci[c] for c in feats]]; p = cv_predict(Xt, y, groups); P_[tier] = p; L[tier] = rowloss(p, y)
        res[tier] = dict(features=list(feats), logloss=float(L[tier].mean()), auc=auc(p, y))
    for a, b in (("M0", "M1"), ("M1", "M2"), ("M2", "M3")):
        dd = L[a] - L[b]; lo, hi = boot_diff(dd, groups); res[f"{b}_vs_{a}"] = dict(logloss_gain=float(dd.mean()), ci=(lo, hi), auc_gain=res[b]["auc"] - res[a]["auc"])
    # the paired identity test: cells whose outcome differs between the two draws
    key = {}
    for i, (sid, t, fv, dr, c) in enumerate(meta): key.setdefault((int(sid), int(t), float(fv), int(c)), {})[int(dr)] = i
    pairs = [(v[0], v[1]) for v in key.values() if len(v) == 2 and y[v[0]] != y[v[1]]]
    cfg = [ci[c] for c in CONFIG]
    if pairs:
        i0, i1 = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs]); dX = X[i1][:, cfg] - X[i0][:, cfg]; lab = y[i1]; g = groups[i1]
        sym = np.vstack([dX, -dX]); slab = np.concatenate([lab, 1 - lab]); sg = np.concatenate([g, g])        # symmetrized: no intercept needed
        p = cv_predict(sym, slab, sg); a_ = auc(p, slab)
        rng = np.random.default_rng(SEED); ug = np.unique(sg); bs = []
        for _ in range(N_BOOT):
            pick = rng.choice(ug, len(ug)); m = np.concatenate([np.where(sg == u)[0] for u in pick]); bs.append(auc(p[m], slab[m]))
        res["identity_paired"] = dict(discordant_cells=int(len(pairs)), concordant_share=float(1 - len(pairs) / max(sum(1 for v in key.values() if len(v) == 2), 1)),
                                      auc=a_, ci=(float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))))
    # aggregate variance decomposition of the repaired share r(s, f, draw) on U
    agg = {}
    for i, (sid, t, fv, dr, c) in enumerate(meta): agg.setdefault((int(sid), int(t), float(fv)), {}).setdefault(int(dr), []).append(y[i])
    R = [(s_, f_, np.mean(v[0]), np.mean(v[1])) for (s1, s2, f_), v in agg.items() if 0 in v and 1 in v for s_ in [(s1, s2)]]
    if R:
        r = np.array([[x[2], x[3]] for x in R]); fvals = np.array([x[1] for x in R]); tot = r.var()
        within = float(np.mean((r[:, 0] - r[:, 1]) ** 2 / 2)); mf = {fv: r[fvals == fv].mean() for fv in set(fvals)}
        between_f = float(np.var([mf[fv] for fv in fvals for _ in range(2)]))
        res["aggregate"] = dict(cells_per_set_mean=float(np.mean([len(v[0]) for v in agg.values() if 0 in v])), var_total=float(tot), share_between_draws=within / tot if tot else float("nan"),
                                share_fraction=between_f / tot if tot else float("nan"))
    return res


def analyze_a2(width, k=1):
    """P18: a closure's single cell alone vs within the closure; partial cells vs the same cells in the closure."""
    single_b = single_c = 0; part_b = part_c = 0; n_single = n_part = 0; base_single = []
    for f in sorted(glob.glob(str(ROOT / f"runs/analysis/rebuttal_20261002c/attention_{width}/moves/*.npz"))):
        if "smoke" in f: continue
        d = np.load(f, allow_pickle=False); kind = d["var_kind"]; move = d["var_move"]
        for s in range(len(d["state_ids"])):
            sol = d["state_sol"][s].astype(int); vs = np.where(d["var_state"] == s)[0]
            v0 = [v for v in vs if kind[v] == 0]; pred0 = d["pred"][v0[0], k - 1].astype(int) if v0 else None
            for mv in (0, 1):
                vc = [v for v in vs if kind[v] == 1 and move[v] == mv]; vsn = [v for v in vs if kind[v] == 4 and move[v] == mv]; vp = [v for v in vs if kind[v] == 5 and move[v] == mv]
                if not vc: continue
                pc = d["pred"][vc[0], k - 1].astype(int)
                if vsn:
                    c = int([x for x in d["var_cells"][vsn[0]] if x >= 0][0]); ps = d["pred"][vsn[0], k - 1].astype(int)
                    a_, b_ = ps[c] == sol[c], pc[c] == sol[c]; n_single += 1; single_b += int(b_ and not a_); single_c += int(a_ and not b_)
                    if pred0 is not None: base_single.append((int(a_), int(b_), int(pred0[c] == sol[c])))
                if vp:
                    pp = d["pred"][vp[0], k - 1].astype(int)
                    for c in [int(x) for x in d["var_cells"][vp[0]] if x >= 0]:
                        a_, b_ = pp[c] == sol[c], pc[c] == sol[c]; n_part += 1; part_b += int(b_ and not a_); part_c += int(a_ and not b_)
    bs = np.array(base_single) if base_single else np.zeros((0, 3))
    return dict(single=dict(pairs=n_single, kept_alone=float(bs[:, 0].mean()) if len(bs) else None, kept_with_partners=float(bs[:, 1].mean()) if len(bs) else None,
                            noedit_correct=float(bs[:, 2].mean()) if len(bs) else None, only_with_partners=single_b, only_alone=single_c, p=mcnemar(single_b, single_c)),
                partial=dict(cell_pairs=n_part, only_in_closure=part_b, only_in_partial=part_c, p=mcnemar(part_b, part_c)))


def report(out=OUT, widths=WIDTHS):
    out.mkdir(parents=True, exist_ok=True); js = {}; lines = ["SEQUENCE A (exploratory; plan Note_2026-10-10_SeqA_Explanation_Comparison_Plan.md written before any fit)"]
    for w in widths:
        r = dict(A1={f"k{k}": analyze_a1(w, k) for k in (1, 4)}, A2=analyze_a2(w)); js[f"attention_{w}"] = r
        for kk, a in r["A1"].items():
            lines.append(f"attention_{w} A1 {kk}: rows {a['rows']} ({a['puzzles']} puzzles, {a['states']} states), base rate {a['base_rate']:.3f} | "
                         + " | ".join(f"{t} logloss {a[t]['logloss']:.4f} AUC {a[t]['auc']:.3f}" for t in TIERS))
            lines.append(f"   gains: " + " | ".join(f"{b} over {a_}: {a[f'{b}_vs_{a_}']['logloss_gain']:+.4f} [{a[f'{b}_vs_{a_}']['ci'][0]:+.4f}, {a[f'{b}_vs_{a_}']['ci'][1]:+.4f}] AUC {a[f'{b}_vs_{a_}']['auc_gain']:+.3f}"
                                           for a_, b in (("M0", "M1"), ("M1", "M2"), ("M2", "M3"))))
            if "identity_paired" in a:
                ip = a["identity_paired"]; lines.append(f"   identity (paired, discordant cells {ip['discordant_cells']}, concordant share {ip['concordant_share']:.3f}): AUC {ip['auc']:.3f} [{ip['ci'][0]:.3f}, {ip['ci'][1]:.3f}]")
            if "aggregate" in a:
                ag = a["aggregate"]; lines.append(f"   aggregate r on U (mean {ag['cells_per_set_mean']:.1f} cells per set): share of variance between draws {ag['share_between_draws']:.3f}, by fraction {ag['share_fraction']:.3f}")
        s, p_ = r["A2"]["single"], r["A2"]["partial"]
        lines.append(f"attention_{w} A2: single cell kept alone {s['kept_alone']:.3f} vs with partners {s['kept_with_partners']:.3f} (no edit {s['noedit_correct']:.3f}; pairs {s['pairs']}; "
                     f"only-with-partners {s['only_with_partners']} vs only-alone {s['only_alone']}, p {s['p']:.3g}) | partial cells: only in closure {p_['only_in_closure']} vs only in partial {p_['only_in_partial']} (pairs {p_['cell_pairs']}, p {p_['p']:.3g})")
    (out / "report.txt").write_text("\n".join(lines) + "\n"); (out / "report.json").write_text(json.dumps(js, indent=1, default=float)); print("\n".join(lines))


def selftest():
    rng = np.random.default_rng(0); X = rng.standard_normal((4000, 3)); wt = np.array([0.5, 1.5, -1.0, 0.0])
    y = (rng.random(4000) < 1 / (1 + np.exp(-(wt[0] + X @ wt[1:])))).astype(float); w = logit_fit(X, y)
    assert np.allclose(w, wt, atol=0.15), w
    assert abs(auc(np.array([0.1, 0.4, 0.35, 0.8]), np.array([0, 0, 1, 1])) - 0.75) < 1e-12
    g = np.repeat(np.arange(50), 7); fo = folds_by_group(g); assert all(len(set(fo[g == u])) == 1 for u in range(50)) and len(set(fo)) == N_FOLDS
    assert mcnemar(10, 0) < 0.01 and mcnemar(3, 3) == 1.0
    lo, hi = boot_diff(np.ones(350) * 0.2, g); assert abs(lo - 0.2) < 1e-12 and abs(hi - 0.2) < 1e-12
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--selftest", action="store_true"); a = ap.parse_args()
    if a.selftest: selftest(); return
    report()


if __name__ == "__main__":
    main()
