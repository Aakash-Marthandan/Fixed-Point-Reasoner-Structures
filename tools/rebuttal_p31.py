#!/usr/bin/env python3
# Ledger: THE DISCUSSION-PERIOD EXPERIMENT P31 — a registered replication of P29's response transfer on ARC, with correctness included
# (registration Documentation/Note_2026-10-12_Rebuttal_P31_Registration.md, written before any P31 row). MEASUREMENT, $0, inference only on the
# Mac's CPU. Reuses P29's runner (tools/rebuttal_p29.py; seed-parametrized, independent controls) and its statistic unchanged.
"""P31: the same model, measurement and 352 confirmation tasks as P29, with fresh history, challenge and control seeds (20261016). The fixed-start
history after one update is deterministic and therefore the same state as in P29; its pairing, challenges and controls are new. Conditions: intact,
full, and M = 4 independent matched controls (each drawn once per pair-direction and reused for the no-edit arm and every mask). Each control's
outcome is computed from its own rollout and the four are averaged; all controls of a query stay in its task-bootstrap cluster.
Letters (fixed sequence): R-MARGIN (standardized margin, k = 1: beta(full) > 0 and beta(full) - beta(avg control) > 0, both 95 % lower bounds); then
R-CORRECT (the same on token correctness), awarded only if R-MARGIN holds. Sensitivity (no letters): per outcome, the omission of the task with the
largest share of the jackknife variance of full - averaged control ("survives the selected task omission" or not), the predictor-information
omission, complete leave-one-task-out ranges; the clipped secondary (standardized responses centered within history-dose, clipped at +-10 per
condition and per control, averaged, recentered)."""
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import rebuttal_p29 as T

OUT = ROOT / "runs/analysis/rebuttal_20261017"
SEED31 = 20261016
M = 4
CONDS = ["intact", "full"] + [f"ctrl_j{j}" for j in range(M)]
AVG = 99                                    # the averaged control's index in the unit tables
CLIP = 10.0
N_BOOT = 20000


# ---------------- the averaged control and the clipped secondary (pure; selftested) ----------------
def with_avg(Y, B, conds):
    """Adds the averaged control under index AVG: the mean of the M controls' outcomes (each computed from its own rollout)."""
    js = [conds.index(f"ctrl_j{j}") for j in range(M)]; Y2, B2 = dict(Y), dict(B)
    for (s, c, i, m) in [k for k in Y if k[1] == js[0]]: Y2[(s, AVG, i, m)] = float(np.mean([Y[(s, j, i, m)] for j in js]))
    for (s, c) in [k for k in B if k[1] == js[0]]: B2[(s, AVG)] = float(np.mean([B[(s, j)] for j in js]))
    return Y2, B2


def clipped(Y, conds, qi_of_state, clip=CLIP):
    """The clipped secondary: within each (history, condition, dose) the four masks' standardized responses are centered, clipped at +-clip
    (each control separately), then the controls are averaged; the statistic recenters. Returns (Y', responses clipped, of total, queries affected)."""
    keys = {}
    for (s, c, i, m), v in Y.items(): keys.setdefault((s, c, i), {})[m] = v
    Y2 = {}; n = tot = 0; q = set()
    for (s, c, i), mv in keys.items():
        a = np.array([mv[m] for m in range(T.MASKS)]); a = a - a.mean(); hit = np.abs(a) > clip; n += int(hit.sum()); tot += len(a)
        if hit.any(): q.add(int(qi_of_state[s]))
        for m in range(T.MASKS): Y2[(s, c, i, m)] = float(np.clip(a[m], -clip, clip))
    js = [conds.index(f"ctrl_j{j}") for j in range(M)]
    for (s, c, i) in [k for k in keys if k[1] == js[0]]:
        for m in range(T.MASKS): Y2[(s, AVG, i, m)] = float(np.mean([Y2[(s, j, i, m)] for j in js]))
    return Y2, n, tot, sorted(q)


# ---------------- the frozen reader ----------------
def contrasts(Y, dk, ix, jf, tcl, seed):
    uF, uC = T.centered_units(Y, dk, jf, ix, ix), T.centered_units(Y, dk, AVG, ix, ix); cl = tcl[uF["state"]]
    bF, bC = T.beta(uF), T.beta(uC); dF = T.boot_betas([uF], [1.0], [cl], seed + [1], N_BOOT); dD = T.boot_betas([uF, uC], [1.0, -1.0], [cl, cl], seed + [2], N_BOOT)
    dC = T.boot_betas([uC], [1.0], [cl], seed + [3], N_BOOT)
    return dict(beta_full=bF, ci_full=T.ci(dF), beta_ctrl=bC, ci_ctrl=T.ci(dC), diff=bF - bC, ci_diff=T.ci(dD), unidentified=dict(full=T.n_unidentified(dF), diff=T.n_unidentified(dD))), uF, uC


def influence(uF, uC, task, tcl, Y, dk, ix, seed):
    """Registered sensitivity (no letters): the outcome-influence omission (largest share of the jackknife variance of full - averaged control),
    the predictor-information omission (P29's), and the complete leave-one-task-out ranges of beta(full) and the difference."""
    sub = lambda u, m: {k: v[m] for k, v in u.items()}; tk = task[uF["state"]]; cl = tcl[uF["state"]]; ut = np.unique(tk)
    th = np.array([T.beta(sub(uF, tk != t)) - T.beta(sub(uC, tk != t)) for t in ut]); thF = np.array([T.beta(sub(uF, tk != t)) for t in ut])
    dv = (th - np.nanmean(th)) ** 2; sel = str(ut[int(np.nanargmax(dv))]); out = dict(selected_task=sel, selected_share=float(np.nanmax(dv) / np.nansum(dv)),
                                                                                       loto_full=(float(np.nanmin(thF)), float(np.nanmax(thF))), loto_diff=(float(np.nanmin(th)), float(np.nanmax(th))))
    for name, t_ in (("outcome_influence", sel), ("predictor_information", (T.design_diagnostics(Y, dk, [str(x) for x in task], ix) or {}).get("top_task"))):
        if t_ is None: continue
        m = tk != t_; vF, vC = sub(uF, m), sub(uC, m); oF = T.ci(T.boot_betas([vF], [1.0], [cl[m]], seed + [len(name)], N_BOOT))
        oD = T.ci(T.boot_betas([vF, vC], [1.0, -1.0], [cl[m], cl[m]], seed + [len(name), 1], N_BOOT))
        out[name] = dict(task=str(t_), beta_full=T.beta(vF), ci_full=oF, diff=T.beta(vF) - T.beta(vC), ci_diff=oD, survives=bool(oF[0] > 0 and oD[0] > 0))
    return out


def report(base=OUT, tag="conf", quiet=False):
    d = T.load(tag, base); conds = [str(x) for x in d["conds"]]; ix, jf = conds.index("intact"), conds.index("full"); don = T.donor_map(d)
    task = np.array([str(x) for x in d["state_task"]]); tcl = np.unique(task, return_inverse=True)[1]; qi = d["state_qi"]; gates = json.loads(str(d["gates"]))
    fmt = lambda e, c: f"{e:+.3f} [{c[0]:+.3f}, {c[1]:+.3f}]"; res = dict(states=int(len(qi)), queries=int(len(np.unique(qi))), tasks=int(len(np.unique(task))),
                                                                        screened=len(json.loads(str(d["screen"]))))
    gm = {k: (min if k == "min_top2_gap" else max)(g.get(k, 0) for g in gates) for k in gates[0] if k not in ("native", "batch_invariant", "batch_maxdev")}
    res["gates"] = dict(max=gm, native=all(n["bitwise"] for g in gates for n in g.get("native", [])), batch=all(g.get("batch_invariant", True) for g in gates))
    p29 = ROOT / "runs/analysis/rebuttal_20261015/p29_conf.npz"
    if p29.exists():
        with np.load(p29, allow_pickle=False) as z: q29 = set(int(x) for x in z["state_qi"])
        q31 = set(int(x) for x in qi); res["overlap_p29"] = dict(p31=len(q31), p29=len(q29), both=len(q31 & q29), p31_only=len(q31 - q29), p29_only=len(q29 - q31))
    L = [f"P31 REPORT (registered before any P31 row; fresh seeds {SEED31}; M = {M} averaged controls; |E| = 4; {N_BOOT} task-clustered draws)",
         f"screened {res['screened']} queries; eligible {res['queries']} ({res['states']} pair-directions, {res['tasks']} tasks); overlap with P29's eligible queries {res.get('overlap_p29')}",
         f"gates: native {res['gates']['native']}, batch {res['gates']['batch']}, max {gm}"]
    for mi, meas in enumerate(("margin_std", "correct", "margin")):
        for k in range(1, T.K_MAX + 1):
            Y, B, dk, exc = T.tables(d, meas, 4, k, don); Y, B = with_avg(Y, B, conds)
            r, uF, uC = contrasts(Y, dk, ix, jf, tcl, [SEED31, 31, mi, k]); res[f"{meas}|k{k}"] = r
            tag_ = "PRIMARY " if (meas == "margin_std" and k == 1) else ("SECOND CONFIRMATORY " if (meas == "correct" and k == 1) else "")
            L.append(f"  {tag_}{meas} k={k}: beta(full) {fmt(r['beta_full'], r['ci_full'])} | beta(avg control) {fmt(r['beta_ctrl'], r['ci_ctrl'])} | full - avg control {fmt(r['diff'], r['ci_diff'])}"
                     + (f" | small-scale excluded {sorted(exc)}" if exc else ""))
            if k == 1 and meas in ("margin_std", "correct"):
                ok = r["ci_full"][0] > 0 and r["ci_diff"][0] > 0; res[f"{meas}_both_bounds"] = bool(ok)
                inf = influence(uF, uC, task, tcl, Y, dk, ix, [SEED31, 31, mi, 50]); res[f"influence|{meas}"] = inf
                for nm in ("outcome_influence", "predictor_information"):
                    if nm in inf: L.append(f"     {nm} omission ({inf[nm]['task']}{', ' + format(inf['selected_share'], '.3f') + ' of the jackknife variance' if nm == 'outcome_influence' else ''}): beta(full) {fmt(inf[nm]['beta_full'], inf[nm]['ci_full'])}, diff {fmt(inf[nm]['diff'], inf[nm]['ci_diff'])} -> {'survives' if inf[nm]['survives'] else 'does not survive'} the selected task omission")
                L.append(f"     leave-one-task-out ranges: beta(full) [{inf['loto_full'][0]:+.3f}, {inf['loto_full'][1]:+.3f}], diff [{inf['loto_diff'][0]:+.3f}, {inf['loto_diff'][1]:+.3f}]")
                if meas == "margin_std":
                    js = [conds.index(f"ctrl_j{j}") for j in range(M)]
                    L.append("     per-control beta (vs intact): " + ", ".join(f"j{j} {T.beta(T.centered_units(Y, dk, c, ix, ix)):+.3f}" for j, c in enumerate(js)))
                    Yc, n, tot, qa = clipped(Y, conds, qi); rc, _, _ = contrasts(Yc, dk, ix, jf, tcl, [SEED31, 31, 9]); res["clipped"] = dict(rc, responses_clipped=n, responses=tot, queries_affected=qa)
                    L.append(f"     CLIPPED SECONDARY (+-{CLIP:g} after centering, per condition and control, then recentered): beta(full) {fmt(rc['beta_full'], rc['ci_full'])}, diff {fmt(rc['diff'], rc['ci_diff'])}; responses clipped {n} of {tot} in {len(qa)} queries")
                    sF, st_ = T.baseline_slope(B, dk, ix, jf); sC, _ = T.baseline_slope(B, dk, ix, AVG)
                    u1 = T.r1_units(d, Y, B, dk, ix, jf, AVG); e1 = T.r1_beta(u1, "x") - T.r1_beta(u1, "ref"); res["BASELINE"] = sF - sC; res["R1"] = e1
                    L.append(f"     secondary BASELINE {sF - sC:+.3f} | R1 {e1:+.3f} (point estimates)")
    rm = res["margin_std_both_bounds"]; res["R_MARGIN"] = "R-MARGIN" if rm else "R-MARGIN-NOT-REPLICATED"
    res["R_CORRECT"] = ("R-CORRECT" if res["correct_both_bounds"] else "R-CORRECT-NOT-RESOLVED") if rm else "CORRECTNESS-REPORTED-WITHOUT-LABEL"
    Yc, Bc, _, _ = T.tables(d, "correct", 4, 1, don); Yc, Bc = with_avg(Yc, Bc, conds)
    rep_ = {c: 100 * float(np.mean([v for (s, j, i, m), v in Yc.items() if j == jj])) for c, jj in (("intact", ix), ("full", jf), ("avg control", AVG))}; res["repair_k1_pct"] = rep_
    L += [f"  challenged-arm evaluation-token correctness at k=1 (%): " + ", ".join(f"{c} {v:.2f}" for c, v in rep_.items()),
          f"LETTERS (fixed sequence; P31 alone): {res['R_MARGIN']}; {res['R_CORRECT']}"]
    (base / f"report_{tag}.txt").write_text("\n".join(L) + "\n"); (base / f"report_{tag}.json").write_text(json.dumps(res, indent=1, default=float))
    print("\n".join(L) if not quiet else f"report written ({len(L)} lines; output suppressed)")


def selftest():
    """Synthetic merged files: planted margin transfer (0.4) with a correctness transfer -> R-MARGIN and R-CORRECT; no margin transfer -> correctness
    reported without a label even when planted; one huge control response -> counted by the clipped secondary."""
    import tempfile
    rng = np.random.default_rng(3); nq, e, S_ = 70, 4, 140; K = T.K_MAX
    for plant_m, want in ((0.4, ("R-MARGIN", "R-CORRECT")), (0.0, ("R-MARGIN-NOT-REPLICATED", "CORRECTNESS-REPORTED-WITHOUT-LABEL"))):
        Yi = rng.normal(size=(S_, 3, 4)); Ci = rng.integers(0, 2, size=(S_, 3, 4, e)); don = lambda s: s ^ 1; V = []
        for s in range(S_):
            for c in range(len(CONDS)):
                for arm, i, m in [(0, -1, -1)] + [(1, i, m) for i in range(3) for m in range(4)]:
                    if arm == 0: y, cor = 0.0, Ci[s, 0, 0]
                    else:
                        y = Yi[s, i, m] + (plant_m * Yi[don(s), i, m] + 0.1 * rng.normal() if c == 1 else (0.1 * rng.normal() if c > 1 else 0.0))
                        cor = Ci[don(s), i, m] if c == 1 else Ci[s, i, m]
                    if (s, c, i, m) == (5, 2, 0, 0): y += 40.0                        # one large response in one control
                    V.append((s, c, arm, i, m, y, cor))
        lab = np.full((S_, T.NPOS), 3, np.uint8); lab[:, 30:] = 0; Et = np.tile(np.arange(e), (S_, 1)).astype(np.int16)
        elog = np.zeros((len(V), K, max(T.E_CAND), T.VOCAB), np.float32); disp = np.zeros((len(V), K, T.NPOS), np.uint8); disp[:, :, :30] = 3
        for q, v in enumerate(V):
            elog[q, :, :e, 3] = v[5]; disp[q, :, :e] = np.where(v[6] == 1, 3, 4)
        with tempfile.TemporaryDirectory() as td:
            np.savez_compressed(Path(td) / "p29_conf.npz", conds=np.array(CONDS), gates=json.dumps([{"x_flips": 0}]), screen=json.dumps([{}] * nq), state_qi=np.repeat(np.arange(nq), 2).astype(np.int32),
                                state_task=np.array([f"t{q // 2}" for q in range(S_)]), state_dir=np.tile([0, 1], nq).astype(np.int8), state_Etok=Et, state_W=np.full(S_, 20, np.int16),
                                state_lab=lab, state_disp_d1=np.where(np.arange(T.NPOS) < 30, 4, 0).astype(np.uint8)[None].repeat(S_, 0), state_masks=json.dumps([[[i, m, [10 + m]] for i in range(3) for m in range(4)]] * S_),
                                state_tinfo=json.dumps([{}] * S_), state_cinfo=json.dumps([[{}]] * S_), var_state=np.array([v[0] for v in V], np.int32), var_cond=np.array([v[1] for v in V], np.int8),
                                var_arm=np.array([v[2] for v in V], np.int8), var_dose=np.array([v[3] for v in V], np.int8), var_mask=np.array([v[4] for v in V], np.int8), disp=disp, elog=elog)
            nb = globals()["N_BOOT"]; globals()["N_BOOT"] = 200
            try: report(Path(td), "conf", quiet=True); res = json.loads((Path(td) / "report_conf.json").read_text())
            finally: globals()["N_BOOT"] = nb
        assert (res["R_MARGIN"], res["R_CORRECT"]) == want, (plant_m, res["R_MARGIN"], res["R_CORRECT"])
        assert res["clipped"]["responses_clipped"] >= 1 and len(res["clipped"]["queries_affected"]) >= 1
    Yt = {(0, 2, 0, m): v for m, v in enumerate([0.0, 0.0, 0.0, 40.0])}; Yt.update({(0, c, 0, m): 0.0 for c in (0, 1, 3, 4, 5) for m in range(4)})
    Yc, n, tot, qa = clipped(Yt, CONDS, np.array([7])); assert n == 1 and qa == [7] and abs(Yc[(0, 2, 0, 3)] - CLIP) < 1e-12 and abs(Yc[(0, AVG, 0, 3)] - CLIP / 4) < 1e-12
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("action", nargs="?", choices=["run", "merge", "report"]); ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--threads", type=int, default=6); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1); ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
    if a.selftest: selftest(); return
    if a.action == "run":
        r = T.Runner(OUT, a.threads, seed=SEED31); _, conf = T.task_split([x["task"] for x in r.rows]); q = T.query_order(r.rows, conf)
        if a.limit: q = q[:a.limit]
        r.run_blocks(q, CONDS, 4, 4, "conf", a.shard, a.nshard, exclude_flips=True, tol=(T.TOL_TRANSPLANT, T.TOL_CORRECTION))
    elif a.action == "merge": T.merge("conf", OUT)
    elif a.action == "report": report(OUT, "conf", a.quiet)
    else: ap.print_help()


if __name__ == "__main__":
    main()
