#!/usr/bin/env python3
"""THE JOINT-READOUT LENS (2026-09-10; analysis-time, descriptive, no rules): the softmax readout's confidence and the trained
commit head's probability on the SAME cells of the SAME cold trajectories (C6 = the DEC-w384 + the calibrated commit head).
Runs the evaluator's step on the rating-stratified set at D16 (the calibration protocol's grid), and at every outer step records
per non-given cell: the softmax confidence max_d p(d), the argmax's correctness, and the head's c = sigmoid(commit_logits(z_H))
on the carried state produced by the step (the evaluator's --record-commit convention). Reports, on the unsolved puzzles at the
last step: the 2x2 of softmax-committed (conf > tau) x head-committed (c > tau) with the wrong rate in each cell; the AUC of c
for "this softmax-committed cell is wrong"; the AUC of 1 - conf for the same; reliability of conf and of c over the free cells;
and per puzzle the AUC of mean c vs mean conf for exactness. On the solved puzzles the same rows as a control.

  PYTHONPATH=src JAX_PLATFORMS=cpu .venv/bin/python tools/lens_commit_joint.py --ckpt runs/pretrainchamp_C6/ckpt_028000.pkl --n 512 --t 16
      -> runs/analysis/champ_commit_joint_<date>.{txt,json}
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; RUNS = ROOT / "runs"
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tools"))
NPZ = ROOT / "data/sudoku_extreme/sudoku_extreme_seed0.npz"
L = []; J = {}
def say(s=""): L.append(str(s)); print(s, flush=True)
def auc(score, y):
    from scipy import stats
    y = np.asarray(y, bool)
    if y.all() or (~y).all() or len(y) == 0: return None
    return float(stats.mannwhitneyu(score[y], score[~y], alternative="greater").statistic / (y.sum() * (~y).sum()))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", default="runs/pretrainchamp_C6/ckpt_028000.pkl"); ap.add_argument("--n", type=int, default=512); ap.add_argument("--t", type=int, default=16)
    ap.add_argument("--tau", type=float, default=0.9); ap.add_argument("--date", default=time.strftime("%Y%m%d")); ap.add_argument("--strat-seed", type=int, default=20260821)
    a = ap.parse_args(); t0 = time.time()
    import jax, jax.numpy as jnp
    from qhrrn2 import episodic as E, grid as GR, model as M, sudoku as SU, sudoku_extreme as SX, dec_cell as DC
    from qhrrn2.config import Config
    import eval_sudoku_extreme as EV
    d = SX.load_prepared(NPZ); Q, A, R = d["test_q"], d["test_a"], d["test_rating"]
    ids = SX.stratified_subsample(R, a.n, a.strat_seed); B = len(ids); puz9 = Q[ids].astype(np.int32); sol9 = A[ids].astype(np.int32); free = puz9 == 0
    saved = E.load_ckpt(str(ROOT / a.ckpt)); defaults = Config(); cfg = Config(**{kk: type(getattr(defaults, kk))(v) for kk, v in saved["config"].items()})
    assert cfg.cell_kind == "dec" and cfg.dec_commit, "a DEC checkpoint trained with --dec-commit"
    st = saved["state_ema"]; params = st["model"]; tvj = jnp.asarray(st["table"][0]); p_dec = params["dec"]
    eta, eta_z = (float(v) for v in M.eq_etas(params, cfg)); layout = cfg.sudoku_layout or "origin"; cv = SU.layout_canvas(layout); ab = EV.coupled_ab(params, cfg)
    x_can = jnp.asarray(np.stack([SU.place_layout(gq.astype(np.int8), layout) for gq in puz9]), jnp.int32)
    void = jax.nn.one_hot(jnp.full((cv, cv), GR.VOID, jnp.int32), M.VOCAB).transpose(2, 0, 1); y = jnp.broadcast_to(void, (B,) + void.shape); z = None
    head = jax.jit(jax.vmap(lambda zz: jax.nn.sigmoid(DC.commit_logits(p_dec, zz[0]))))
    T = a.t; conf = np.zeros((T, B, 81), np.float32); ok = np.zeros((T, B, 81), bool); cc = np.zeros((T, B, 81), np.float32); preds = np.zeros((T, B, 9, 9), np.int16)
    say(f"== THE JOINT-READOUT LENS: {a.ckpt} (tau {cfg.dec_commit_tau}, head w .{int(10*cfg.dec_commit_w)}) on strat-{B} (seed {a.strat_seed}) at D{T}, EMA ==")
    for t in range(T):
        first = z is None; logits, zf = EV._step(cfg, 1.0, 0.0, first)(params, x_can, y, tvj, jnp.zeros(1) if first else z); z = zf if first else z + eta_z * (zf - z)
        p = jax.nn.softmax(logits, axis=-1); pT = p.transpose(0, 3, 1, 2); y = (ab[0] * y + ab[1] * pT) if ab is not None else (y + eta * (pT - y))
        p9 = np.asarray(EV.layout_gather(p, layout))[..., 1:10]; p9 = p9 / np.maximum(p9.sum(-1, keepdims=True), 1e-9)
        pred = np.asarray(EV.layout_gather(jnp.argmax(logits, axis=-1), layout)); pred = np.where(pred == GR.VOID, 0, pred).astype(np.int16); preds[t] = pred
        conf[t] = p9.max(-1).reshape(B, 81); ok[t] = (pred == sol9).reshape(B, 81); cc[t] = np.asarray(head(z)).reshape(B, 81)
        if t == 0: say(f"  step 1 in {time.time()-t0:.0f}s (incl. compile)")
    solved = (preds[-1] == sol9).all((1, 2)); fr = free.reshape(B, 81); tau = a.tau
    say(f"  solved {solved.sum()}/{B} = {100*solved.mean():.2f} % at D{T}; unsolved {int((~solved).sum())}; free cells per puzzle {fr.sum(1).mean():.1f}")
    out = {}
    for lab, m in (("unsolved", ~solved), ("solved", solved)):
        if not m.any(): continue
        c_ = cc[-1][m][fr[m]]; f_ = conf[-1][m][fr[m]]; o_ = ok[-1][m][fr[m]]
        sc = f_ > tau; hc = c_ > tau
        tab = {}
        for sn, sm in (("softmax-committed", sc), ("softmax-uncommitted", ~sc)):
            for hn, hm in (("head-committed", hc), ("head-uncommitted", ~hc)):
                mm = sm & hm; tab[f"{sn} & {hn}"] = dict(n=int(mm.sum()), frac=float(mm.mean()), wrong=(float((~o_[mm]).mean()) if mm.any() else None))
        row = dict(n_puzzles=int(m.sum()), n_cells=int(len(c_)), softmax_committed=float(sc.mean()), head_committed=float(hc.mean()), wrong_among_softmax_committed=(float((~o_[sc]).mean()) if sc.any() else None),
                   wrong_among_head_committed=(float((~o_[hc]).mean()) if hc.any() else None), table=tab,
                   auc_c_wrong_within_softmax_committed=(auc(1 - c_[sc], ~o_[sc]) if sc.any() else None), auc_conf_wrong_within_softmax_committed=(auc(1 - f_[sc], ~o_[sc]) if sc.any() else None),
                   auc_c_wrong_all_free=auc(1 - c_, ~o_), auc_conf_wrong_all_free=auc(1 - f_, ~o_), mean_conf=float(f_.mean()), mean_c=float(c_.mean()), cells_correct=float(o_.mean()))
        out[lab] = row
        say(f"\n  [{lab}: {row['n_puzzles']} puzzles, {row['n_cells']} free cells at step {T}] softmax-committed (conf > {tau}) {100*row['softmax_committed']:.1f} % of cells, {100*(row['wrong_among_softmax_committed'] or 0):.1f} % of them wrong | head-committed (c > {tau}) {100*row['head_committed']:.1f} %, {100*(row['wrong_among_head_committed'] or 0):.1f} % wrong | cells correct {100*row['cells_correct']:.1f} % | mean conf {row['mean_conf']:.3f} mean c {row['mean_c']:.3f}")
        for k_, v_ in tab.items(): say(f"     {k_:44s}: {v_['n']:6d} cells ({100*v_['frac']:5.1f} %), wrong {('-' if v_['wrong'] is None else f'{100*v_['wrong']:.1f} %')}")
        say(f"     AUC for 'this softmax-committed cell is WRONG': the head's 1-c {row['auc_c_wrong_within_softmax_committed'] if row['auc_c_wrong_within_softmax_committed'] is None else round(row['auc_c_wrong_within_softmax_committed'],4)} | the softmax's 1-conf {row['auc_conf_wrong_within_softmax_committed'] if row['auc_conf_wrong_within_softmax_committed'] is None else round(row['auc_conf_wrong_within_softmax_committed'],4)}; over all free cells: 1-c {round(row['auc_c_wrong_all_free'],4) if row['auc_c_wrong_all_free'] is not None else None} | 1-conf {round(row['auc_conf_wrong_all_free'],4) if row['auc_conf_wrong_all_free'] is not None else None}")
    # reliability over the free cells of the unsolved puzzles at the last step: conf bins and c bins
    m = ~solved
    if m.any():
        c_ = cc[-1][m][fr[m]]; f_ = conf[-1][m][fr[m]]; o_ = ok[-1][m][fr[m]]; edges = np.array([0, .3, .5, .7, .9, .99, 1.0001])
        say("\n  reliability on the unsolved puzzles' free cells (step %d): bin | softmax conf: n, mean, P(correct) || head c: n, mean, P(correct)" % T)
        rel = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mf = (f_ >= lo) & (f_ < hi); mc = (c_ >= lo) & (c_ < hi)
            rel.append(dict(lo=float(lo), hi=float(hi), n_conf=int(mf.sum()), conf=(float(f_[mf].mean()) if mf.any() else None), acc_conf=(float(o_[mf].mean()) if mf.any() else None), n_c=int(mc.sum()), c=(float(c_[mc].mean()) if mc.any() else None), acc_c=(float(o_[mc].mean()) if mc.any() else None)))
            r = rel[-1]; say(f"    [{lo:.2f},{min(hi,1):.2f}) | {r['n_conf']:6d} {('-' if r['conf'] is None else f'{r['conf']:.3f}'):>6s} {('-' if r['acc_conf'] is None else f'{100*r['acc_conf']:5.1f}'):>6s} || {r['n_c']:6d} {('-' if r['c'] is None else f'{r['c']:.3f}'):>6s} {('-' if r['acc_c'] is None else f'{100*r['acc_c']:5.1f}'):>6s}")
        ece_conf = float(sum(r["n_conf"] * abs((r["conf"] or 0) - (r["acc_conf"] or 0)) for r in rel) / max(sum(r["n_conf"] for r in rel), 1)); ece_c = float(sum(r["n_c"] * abs((r["c"] or 0) - (r["acc_c"] or 0)) for r in rel) / max(sum(r["n_c"] for r in rel), 1))
        say(f"    ECE on the unsolved puzzles' free cells: softmax conf {ece_conf:.3f} | head c {ece_c:.3f}"); out["reliability_unsolved"] = dict(rows=rel, ece_conf=ece_conf, ece_c=ece_c)
    # per puzzle: mean c vs mean conf vs exactness; and the per-step trajectory of both readouts on unsolved puzzles
    mc_p = (cc[-1] * fr).sum(1) / np.maximum(fr.sum(1), 1); mf_p = (conf[-1] * fr).sum(1) / np.maximum(fr.sum(1), 1)
    out["per_puzzle"] = dict(auc_mean_c=auc(mc_p, solved), auc_mean_conf=auc(mf_p, solved), auc_min_c=auc(np.where(fr, cc[-1], 1).min(1), solved), auc_min_conf=auc(np.where(fr, conf[-1], 1).min(1), solved))
    say(f"\n  per-puzzle exactness signal at step {T}: AUC(mean c) {out['per_puzzle']['auc_mean_c']:.4f} | AUC(mean softmax conf) {out['per_puzzle']['auc_mean_conf']:.4f} | AUC(min c) {out['per_puzzle']['auc_min_c']:.4f} | AUC(min conf) {out['per_puzzle']['auc_min_conf']:.4f}")
    if (~solved).any():
        say("  trajectory on the UNSOLVED puzzles (free cells): step | mean conf | softmax-committed % | wrong among them % | mean c | head-committed % | wrong among them % | cells correct %")
        traj = []
        for t in (0, 1, 3, 7, 11, T - 1):
            f_ = conf[t][~solved][fr[~solved]]; c_ = cc[t][~solved][fr[~solved]]; o_ = ok[t][~solved][fr[~solved]]; sc = f_ > tau; hc = c_ > tau
            traj.append(dict(step=t + 1, conf=float(f_.mean()), sc=float(sc.mean()), sc_wrong=(float((~o_[sc]).mean()) if sc.any() else None), c=float(c_.mean()), hc=float(hc.mean()), hc_wrong=(float((~o_[hc]).mean()) if hc.any() else None), correct=float(o_.mean())))
            r = traj[-1]; say(f"    {r['step']:4d} | {r['conf']:.3f} | {100*r['sc']:5.1f} | {('-' if r['sc_wrong'] is None else f'{100*r['sc_wrong']:5.1f}'):>5s} | {r['c']:.3f} | {100*r['hc']:5.1f} | {('-' if r['hc_wrong'] is None else f'{100*r['hc_wrong']:5.1f}'):>5s} | {100*r['correct']:5.1f}")
        out["trajectory_unsolved"] = traj
    J.update(dict(ckpt=a.ckpt, n=int(B), t=T, tau=tau, solved=float(solved.mean()), rows=out, wall=round(time.time() - t0, 1)))
    OUT = RUNS / "analysis" / f"champ_commit_joint_{a.date}.txt"; OUT.write_text("\n".join(L) + "\n"); OUT.with_suffix(".json").write_text(json.dumps(J, indent=1, default=float))
    say(f"\n({time.time()-t0:.0f}s) artifact -> {OUT} (+ .json)")

if __name__ == "__main__":
    main()
