"""P31 independent recount (own code; imports nothing from tools/rebuttal_p29.py or tools/rebuttal_p31.py). From the merged raw arrays: per-variant
margin (label logit minus the largest other logit, mean over the four E tokens, k = 1) and correctness; option B's per-query scale; each control's
outcome computed from its own rollout, then averaged; four-mask centering by whole arrays; QR least squares; the two contrasts and an independent
task-clustered resampler; the outcome-influence task and its omission; leave-one-task-out ranges; the clipping secondary's counts and estimates."""
import json
from pathlib import Path
import numpy as np

B = Path(__file__).resolve().parent
d = np.load(B / "p29_conf.npz", allow_pickle=False); rep = json.loads((B / "report_conf.json").read_text())
conds = [str(x) for x in d["conds"]]; I, F = conds.index("intact"), conds.index("full"); CJ = [conds.index(f"ctrl_j{j}") for j in range(4)]
vs, vc, va, vd, vm = d["var_state"], d["var_cond"], d["var_arm"], d["var_dose"], d["var_mask"]
qi, sdir, task = d["state_qi"], d["state_dir"], np.array([str(x) for x in d["state_task"]]); S = len(qi)
Et = d["state_Etok"][:, :4].astype(int); lab = d["state_lab"].astype(int); labE = lab[np.arange(S)[:, None], Et]
el = d["elog"][:, 0, :4, :].astype(np.float64); L = labE[vs]
ll = np.take_along_axis(el, L[..., None], -1)[..., 0]; oth = el.copy(); np.put_along_axis(oth, L[..., None], -np.inf, -1); marg = (ll - oth.max(-1)).mean(-1)
corr = (d["disp"][:, 0, :][np.arange(len(vs))[:, None], Et[vs]] == labE[vs]).mean(-1)
partner = {s: int(np.where((qi == qi[s]) & (sdir == 1 - sdir[s]))[0][0]) for s in range(S)}


def table(y):
    Y = np.full((S, len(conds), 3, 4), np.nan); m = va == 1; Y[vs[m], vc[m], vd[m], vm[m]] = y[m]; assert not np.isnan(Y).any(); return Y


def coef(dep, cD, cR):
    A = np.column_stack([cD, cR, np.ones(len(dep))])
    if np.linalg.matrix_rank(A) < 3: return np.nan
    Q, R = np.linalg.qr(A); return float(np.linalg.solve(R, Q.T @ dep)[0])


def analysis(Zc, name, clip=None):
    """Zc: centered responses (S, conds, dose, mask). Returns rows for full and the averaged control (the averaged control = mean of the controls)."""
    if clip is not None:
        n = int((np.abs(Zc) > clip).sum()); qa = len(set(int(qi[s]) for s in np.where((np.abs(Zc) > clip).any((1, 2, 3)))[0])); Zc = np.clip(Zc, -clip, clip)
    avg = Zc[:, CJ].mean(1); Zc = Zc - Zc.mean(-1, keepdims=True); avg = avg - avg.mean(-1, keepdims=True)
    ss = np.arange(S); dn = np.array([partner[s] for s in ss])
    dF = (Zc[ss, F] - Zc[ss, I]).reshape(-1); dC = (avg[ss] - Zc[ss, I]).reshape(-1); cD = Zc[dn, I].reshape(-1); cR = Zc[ss, I].reshape(-1); t = np.repeat(task[ss], 12)
    bF, bC = coef(dF, cD, cR), coef(dC, cD, cR); out = dict(beta_full=bF, beta_ctrl=bC, diff=bF - bC)
    if clip is not None: out.update(clipped=n, queries=qa)
    return out, (dF, dC, cD, cR, t)


def boot(rows, seed, n=20000):
    dF, dC, cD, cR, t = rows; ut = np.unique(t); idx = {u: np.where(t == u)[0] for u in ut}; rng = np.random.default_rng(seed); f_, d_ = np.empty(n), np.empty(n)
    for b in range(n):
        p = np.concatenate([idx[u] for u in rng.choice(ut, len(ut))]); f = coef(dF[p], cD[p], cR[p]); f_[b] = f; d_[b] = f - coef(dC[p], cD[p], cR[p])
    q = lambda a: (round(float(np.nanpercentile(a, 2.5)), 3), round(float(np.nanpercentile(a, 97.5)), 3), int(np.isnan(a).sum())); return q(f_), q(d_)


M = table(marg); C = table(corr)
cenM = M - M.mean(-1, keepdims=True); sc = {int(q): np.sqrt(np.mean(cenM[np.where(qi == q)[0]][:, I] ** 2)) for q in np.unique(qi)}
assert min(sc.values()) >= 0.01
Zstd = cenM / np.array([sc[int(q)] for q in qi])[:, None, None, None]
for name, Z, key in (("margin_std", Zstd, "margin_std|k1"), ("correct", C - C.mean(-1, keepdims=True), "correct|k1")):
    r, rows = analysis(Z, name); f, dd = boot(rows, [3131, len(name)])
    rr = rep[key]; print(f"{name}: recount beta(full) {r['beta_full']:+.6f} beta(avg) {r['beta_ctrl']:+.6f} diff {r['diff']:+.6f} | report {rr['beta_full']:+.6f} {rr['beta_ctrl']:+.6f} {rr['diff']:+.6f}")
    assert abs(r["beta_full"] - rr["beta_full"]) < 1e-9 and abs(r["beta_ctrl"] - rr["beta_ctrl"]) < 1e-9, "point estimates differ"
    print(f"   recount intervals beta(full) {f} diff {dd} | report {[round(x, 3) for x in rr['ci_full']]} {[round(x, 3) for x in rr['ci_diff']]}")
    dF, dC, cD, cR, t = rows; ut = np.unique(t)
    th = np.array([coef(dF[t != u], cD[t != u], cR[t != u]) - coef(dC[t != u], cD[t != u], cR[t != u]) for u in ut]); thF = np.array([coef(dF[t != u], cD[t != u], cR[t != u]) for u in ut])
    dv = (th - np.nanmean(th)) ** 2; sel = ut[int(np.nanargmax(dv))]; inf = rep[f"influence|{name}"]
    m = t != sel; print(f"   selected task {sel} ({np.nanmax(dv) / np.nansum(dv):.3f}) without it: beta(full) {coef(dF[m], cD[m], cR[m]):+.4f}, diff {coef(dF[m], cD[m], cR[m]) - coef(dC[m], cD[m], cR[m]):+.4f}"
                        f" | report {inf['selected_task']} {inf['selected_share']:.3f} {inf['outcome_influence']['beta_full']:+.4f} {inf['outcome_influence']['diff']:+.4f}")
    print(f"   LOTO beta(full) [{np.nanmin(thF):+.4f}, {np.nanmax(thF):+.4f}] diff [{np.nanmin(th):+.4f}, {np.nanmax(th):+.4f}] | report {inf['loto_full']} {inf['loto_diff']}")
r, rows = analysis(Zstd, "clip", clip=10.0); rc = rep["clipped"]
print(f"clipped: recount beta(full) {r['beta_full']:+.6f} diff {r['diff']:+.6f} clipped {r['clipped']} queries {r['queries']} | report {rc['beta_full']:+.6f} {rc['diff']:+.6f} {rc['responses_clipped']} {len(rc['queries_affected'])}")
