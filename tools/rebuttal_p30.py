#!/usr/bin/env python3
# Ledger: P30 — a bounded EXPLORATORY check for a within-puzzle selection signal in the P28 bank (plan Documentation/Note_2026-10-12_Rebuttal_P30_Plan.md,
# committed 47c2fb4 before any feature was computed). No new outcome experiment: a deterministic replay of P28's banks and of the restarts' first two
# iterations, for solution-free features only; outcomes are joined afterwards from P28's saved arrays.
"""Phases: `features --width W` (outcome-blind; writes p30_features_W.npz) and `analyze` (joins P28's outcomes; held-out within-puzzle AUC and top-1
selection per candidate family under puzzle-grouped cross-validation; equally informed comparators; compute accounting; the plan's signal rule)."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
P28DIR = ROOT / "runs/analysis/rebuttal_20261014"
OUT = ROOT / "runs/analysis/rebuttal_20261016"
SEED = 20261016
PROBE = 2
FAMILIES = ("lead", "lead_ctrl", "full", "full_ctrl", "restart")
REC_F = ("viol", "marg_mean", "marg_p10", "chg1", "chg3", "stalled", "start_fixed")
PAIR_F = ("d_viol", "d_marg_mean", "d_marg_p10", "disagree", "mag_full", "mag_lead", "energy_lead")
RST_F = ("viol", "marg_mean", "marg_p10", "chg1", "stalled")
CS, OUTER, INNER, NB = (0.01, 0.1, 1.0, 10.0), 8, 4, 2000


# ---------------- pure helpers (selftested) ----------------
def violations(g):
    """Displayed grids (..., 81) of digits 1..9 -> duplicate count over the 27 units (sum over units of entries minus distinct digits)."""
    G = np.asarray(g).reshape(np.shape(g)[:-1] + (9, 9)); oh = (G[..., None] == np.arange(1, 10)).astype(np.int32)
    rows = oh.sum(-2); cols = oh.sum(-3); bx = oh.reshape(oh.shape[:-3] + (3, 3, 3, 3, 9)).sum((-4, -2))
    dup = lambda c: np.maximum(c - 1, 0).sum((-1, -2)); return dup(rows) + dup(cols) + dup(bx.reshape(bx.shape[:-3] + (9, 9)))


def margin_stats(scores):
    """Scores (..., 81, 9) -> (mean, 10th percentile) over cells of top-1 minus top-2."""
    s = np.sort(scores, -1); m = s[..., -1] - s[..., -2]; return m.mean(-1), np.percentile(m, 10, axis=-1)


def logistic_fit(X, y, C):
    """L2 logistic regression (intercept unpenalized), minimizing sum log-loss + ||w||^2 / (2 C), by Newton's method."""
    Xa = np.column_stack([X, np.ones(len(X))]); w = np.zeros(Xa.shape[1]); R = np.eye(Xa.shape[1]) / C; R[-1, -1] = 0.0
    for _ in range(100):
        p = 1 / (1 + np.exp(-np.clip(Xa @ w, -30, 30))); g = Xa.T @ (p - y) + R @ w; H = (Xa * (p * (1 - p))[:, None]).T @ Xa + R + 1e-9 * np.eye(len(w))
        step = np.linalg.solve(H, g); w -= step
        if np.abs(step).max() < 1e-10: break
    return w


def logistic_predict(w, X): return np.column_stack([X, np.ones(len(X))]) @ w


def grouped_folds(groups, k, seed):
    u = np.unique(groups); perm = np.random.default_rng(seed).permutation(len(u)); fold = {g: i % k for i, g in enumerate(u[perm])}
    return np.array([fold[g] for g in groups])


def fit_predict_cv(X, y, groups, seed):
    """Held-out scores: outer puzzle-grouped folds; inside each, standardization and C chosen by inner grouped CV on log-loss."""
    sd0 = [int(x) for x in np.atleast_1d(seed)]; of = grouped_folds(groups, OUTER, sd0 + [0]); out = np.zeros(len(y)); chosen = []
    for f in range(OUTER):
        tr, te = of != f, of == f; Xt, yt, gt = X[tr], y[tr], groups[tr]
        mu, sd = Xt.mean(0), Xt.std(0) + 1e-12; Z = (Xt - mu) / sd; inf = grouped_folds(gt, INNER, sd0 + [1, f]); best = None
        for C in CS:
            ll = 0.0
            for g in range(INNER):
                a, b = inf != g, inf == g; w = logistic_fit(Z[a], yt[a], C); s = logistic_predict(w, Z[b]); p = 1 / (1 + np.exp(-np.clip(s, -30, 30)))
                ll -= np.sum(yt[b] * np.log(p + 1e-12) + (1 - yt[b]) * np.log(1 - p + 1e-12))
            if best is None or ll < best[0]: best = (ll, C)
        w = logistic_fit(Z, yt, best[1]); out[te] = logistic_predict(w, (X[te] - mu) / sd); chosen.append(best[1])
    return out, chosen


def within_auc_parts(score, y, groups):
    """Per puzzle: (concordant + ties/2, discordant pairs) over candidate pairs with different outcomes."""
    res = {}
    for g in np.unique(groups):
        m = groups == g; s, t = score[m], y[m]; pos, neg = s[t == 1], s[t == 0]
        if len(pos) and len(neg):
            c = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum(); res[g] = (float(c), len(pos) * len(neg))
    return res


def top1(score, y, groups, order):
    """Per puzzle: success of the top-scoring candidate (ties broken by the frozen order), the family's mean success (random pick), the oracle."""
    res = {}
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]; k = idx[np.lexsort((order[idx], -score[idx]))[0]]; res[g] = (float(y[k]), float(y[idx].mean()), float(y[idx].max()))
    return res


# ---------------- features (outcome-blind replay; JAX environment) ----------------
def features(width, out=OUT):
    import rebuttal_p28 as T28, rebuttal_p27 as P27
    out.mkdir(parents=True, exist_ok=True); d = np.load(P28DIR / f"attention_{width}/p28.npz", allow_pickle=False); pids = np.array(d["pids"], np.int64)
    r = T28.R22(width, out / "_p28meta", tag=f"p30_{width}"); U = P27.load_basis(width, r.lm); lm = r.lm; t0 = time.time()
    REC = np.zeros((len(pids), 4, len(REC_F))); PAIR = np.zeros((len(pids), 4, 4, len(PAIR_F))); RST = np.zeros((len(pids), len(T28.RESTART_J), len(RST_F)))
    for b0 in range(0, len(pids), 32):
        bk = r.bank(pids[b0:b0 + 32]); D = bk["disp"]
        assert not bk["valid"].any(), "gate: a regenerated bank history verified (P28 screened these puzzles as bank-failing)"; H = bk["z"][:, :, 0].astype(np.float64); sc = np.einsum("pifsw,w->pisf", H, lm)
        mm, mp = margin_stats(sc); vio = violations(D[:, :, -1])
        for p in range(len(D)):
            P_ = b0 + p
            for i in range(4):
                REC[P_, i] = [vio[p, i], mm[p, i], mp[p, i], (D[p, i, -1] != D[p, i, -2]).sum(), (D[p, i, -1] != D[p, i, -4]).sum(), float(np.array_equal(D[p, i, -1], D[p, i, -2])), float(i == 0)]
            for (i, j) in T28.PAIRS:
                _, inf = r.transplant(bk, pids[P_], p, i, j, U)
                PAIR[P_, i, j] = [vio[p, j], mm[p, j], mp[p, j], (D[p, i, -1] != D[p, j, -1]).sum(), inf["mag_full"], inf["mag_lead"], inf["energy_lead"]]
        # the restarts' first PROBE iterations (P28's restart seeds; canvas void), in batches of at most VB rows
        shp = bk["z"].shape[2:]; jobs = [(p, k, jj) for p in range(len(D)) for k, jj in enumerate(T28.RESTART_J)]
        for c0 in range(0, len(jobs), T28.VB):
            jb = jobs[c0:c0 + T28.VB]; zg = np.stack([r.EV.mi_z0(T28.SEED, int(pids[b0 + p]), jj, shp, 1.0, "gauss") for p, _, jj in jb]).astype(np.float32)
            dg, zf, _ = r.loop(zg, r.void_canvas(len(jb)), np.stack([bk["x"][p] for p, _, _ in jb]), PROBE)
            s2 = np.einsum("bfsw,w->bsf", zf[:, 0].astype(np.float64), lm); m2, p2 = margin_stats(s2); v2 = violations(dg[:, -1])
            for q, (p, k, _) in enumerate(jb): RST[b0 + p, k] = [v2[q], m2[q], p2[q], (dg[q, -1] != dg[q, -2]).sum(), float(np.array_equal(dg[q, -1], dg[q, -2]))]
        print(json.dumps(dict(width=width, puzzles_done=min(b0 + 32, len(pids)), seconds=round(time.time() - t0))), flush=True)
    np.savez_compressed(out / f"p30_features_{width}.npz", pids=pids, rec=REC, pair=PAIR, rst=RST, rec_f=np.array(REC_F), pair_f=np.array(PAIR_F), rst_f=np.array(RST_F),
                        meta=json.dumps(dict(created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), seconds=time.time() - t0, probe=PROBE)))


# ---------------- analysis (joins P28's outcomes) ----------------
def family_rows(width, fam, F, d):
    """(X, y, puzzle, frozen order) for one candidate family; outcome = P28's first verified readout within the window."""
    import rebuttal_p28 as T28
    kind, pp, ii, jj, first = d["kind"], d["p"], d["i"], d["j"], d["first"]; m = kind == fam; X, y, g, o = [], [], [], []
    for q in np.where(m)[0]:
        p, i, j = int(pp[q]), int(ii[q]), int(jj[q])
        X.append(F["rst"][p, i] if fam == "restart" else np.r_[F["rec"][p, i], F["pair"][p, i, j]]); y.append(float(first[q] >= 0)); g.append(p)
        o.append(i if fam == "restart" else int(np.where(T28.pair_order(F["pids"][p]) == T28.PAIRS.index((i, j)))[0][0]))   # P28's frozen order breaks ties
    return np.array(X, float), np.array(y), np.array(g), np.array(o)


def boot_mean(vals_by_puzzle, seed, n=NB):
    keys = sorted(vals_by_puzzle); v = np.array([vals_by_puzzle[k] for k in keys]); rng = np.random.default_rng(seed)
    bs = [v[rng.integers(0, len(v), len(v))].mean(0) for _ in range(n)]; return v.mean(0), np.percentile(bs, 2.5, axis=0), np.percentile(bs, 97.5, axis=0)


def analyze(out=OUT, widths=(192, 256)):
    lines = ["P30 (EXPLORATORY; plan 47c2fb4): within-puzzle selection signal in the P28 bank; held-out scores from puzzle-grouped CV"]; js = {}
    for w in widths:
        fz = out / f"p30_features_{w}.npz"
        if not fz.exists(): lines.append(f"attention_{w}: features missing"); continue
        F = dict(np.load(fz, allow_pickle=False)); d = dict(np.load(P28DIR / f"attention_{w}/p28.npz", allow_pickle=False)); R = {}
        for fi, fam in enumerate(FAMILIES):
            X, y, g, o = family_rows(w, fam, F, d); s, ch = fit_predict_cv(X, y, g, [SEED, w, fi])
            pa = within_auc_parts(s, y, g); t1 = top1(s, y, g, o); R[fam] = dict(s=s, y=y, g=g, auc_parts=pa, top1=t1, C=ch, base=float(y.mean()))
            keys = sorted(pa); rng = np.random.default_rng([SEED, w, fi, 9]); num = np.array([pa[k][0] for k in keys]); den = np.array([pa[k][1] for k in keys])
            auc = num.sum() / den.sum(); bs = [(lambda i: num[i].sum() / den[i].sum())(rng.integers(0, len(keys), len(keys))) for _ in range(NB)]
            mean, lo, hi = boot_mean({k: (t1[k][0], t1[k][1], t1[k][2], t1[k][0] - t1[k][1]) for k in t1}, [SEED, w, fi, 10])
            R[fam].update(auc=float(auc), auc_ci=(float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))), puzzles_discordant=len(keys),
                          pick=float(mean[0]), random=float(mean[1]), oracle=float(mean[2]), gain=float(mean[3]), gain_ci=(float(lo[3]), float(hi[3])))
            lines.append(f"attention_{w} {fam:9s}: base {y.mean():.3f}; within-puzzle AUC {auc:.3f} [{R[fam]['auc_ci'][0]:.3f}, {R[fam]['auc_ci'][1]:.3f}] over {len(keys)} puzzles with discordant pairs; "
                         f"top-1 pick {mean[0]:.3f} vs random {mean[1]:.3f} (gain {mean[3]:+.3f} [{lo[3]:+.3f}, {hi[3]:+.3f}]) vs oracle {mean[2]:.3f}; C chosen {sorted(set(ch))}")
        def paired(a, b, seed):
            ks = sorted(set(R[a]["top1"]) & set(R[b]["top1"])); return boot_mean({k: (R[a]["top1"][k][0] - R[a]["top1"][k][1]) - (R[b]["top1"][k][0] - R[b]["top1"][k][1]) for k in ks}, seed)
        gl = paired("lead", "lead_ctrl", [SEED, w, 20]); gr = boot_mean({k: R["lead"]["top1"][k][0] - R["restart"]["top1"][k][0] for k in R["lead"]["top1"]}, [SEED, w, 21])
        rule = dict(auc=R["lead"]["auc_ci"][0] > 0.5, beats_controls=float(gl[1]) > 0, beats_restarts=float(gr[1]) > 0)
        js[f"attention_{w}"] = dict({f: {k: v for k, v in R[f].items() if k not in ("s", "y", "g", "auc_parts", "top1")} for f in FAMILIES},
                                    lead_gain_minus_ctrl_gain=(float(gl[0]), float(gl[1]), float(gl[2])), lead_pick_minus_restart_pick=(float(gr[0]), float(gr[1]), float(gr[2])), rule=rule)
        lines += [f"attention_{w}: leading-transplant gain minus matched-control gain (paired by puzzle) {gl[0]:+.3f} [{gl[1]:+.3f}, {gl[2]:+.3f}]; leading pick minus restart pick {gr[0]:+.3f} [{gr[1]:+.3f}, {gr[2]:+.3f}]",
                  f"   compute per puzzle (model iterations): transplant/control selection = bank 64 + continuation 16 = 80 (bank sunk: 16); restart selection = 12 probes x {PROBE} + {16 - PROBE} = {12 * PROBE + 16 - PROBE}",
                  f"   signal rule (192 decides; 256 replicates): {rule} -> {'SELECTION SIGNAL' if all(rule.values()) else 'no selection signal at this size'}"]
    (out / "p30_report.txt").write_text("\n".join(lines) + "\n"); (out / "p30_report.json").write_text(json.dumps(js, indent=1, default=float)); print("\n".join(lines))


def selftest():
    g = np.tile(np.arange(1, 10), 9).reshape(9, 9); sol = np.array([[(i * 3 + i // 3 + j) % 9 + 1 for j in range(9)] for i in range(9)]).reshape(81)
    assert violations(sol) == 0; bad = sol.copy(); bad[0] = bad[1]; assert violations(bad) == 3                       # one duplicate in a row, a column... and a box
    sc = np.zeros((2, 81, 9)); sc[..., 0] = 3; sc[..., 1] = 1; m, p = margin_stats(sc); assert np.allclose(m, 2) and np.allclose(p, 2)
    rng = np.random.default_rng(0); X = rng.normal(size=(600, 3)); y = (rng.random(600) < 1 / (1 + np.exp(-(1.5 * X[:, 0] - X[:, 1])))).astype(float)
    w = logistic_fit(X, y, 1e6); assert abs(w[0] - 1.5) < 0.4 and abs(w[1] + 1) < 0.4
    grp = np.repeat(np.arange(60), 10); s, ch = fit_predict_cv(X, y, grp, [1, 2]); pa = within_auc_parts(s, y, grp); auc = sum(v[0] for v in pa.values()) / sum(v[1] for v in pa.values()); assert auc > 0.7
    # a puzzle-level signal only (y constant within puzzle given a puzzle feature) gives no within-puzzle pairs
    yp = np.repeat((rng.random(60) < 0.5).astype(float), 10); assert within_auc_parts(rng.normal(size=600), yp, grp) == {}
    t = top1(np.array([0.1, 0.9, 0.9]), np.array([0., 1., 0.]), np.zeros(3, int), np.array([2, 0, 1])); assert t[0] == (1.0, 1 / 3, 1.0)
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("action", nargs="?", choices=["features", "analyze"]); ap.add_argument("--width", type=int, choices=(192, 256))
    ap.add_argument("--selftest", action="store_true"); a = ap.parse_args()
    if a.selftest: selftest(); return
    if a.action == "features": features(a.width)
    elif a.action == "analyze": analyze()
    else: ap.print_help()


if __name__ == "__main__":
    main()
