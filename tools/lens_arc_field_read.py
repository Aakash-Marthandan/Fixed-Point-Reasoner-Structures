"""The field's alphaXiv TRM ARC-1 checkpoint through the harness instruments (runs/field_ckpts/harness/arc_out/*/rows.jsonl): cold, the ID-zero control, retention, draws, the vote."""
import json, numpy as np
from pathlib import Path
from scipy import stats
R = str(Path(__file__).resolve().parents[1] / "runs/field_ckpts/harness/arc_out") + "/"
def rows(d): return [json.loads(l) for l in open(R + d + "/rows.jsonl")]
def auc(score, y):
    score, y = np.asarray(score, float), np.asarray(y, bool)
    if y.all() or (~y).all(): return None
    return float(stats.mannwhitneyu(score[y], score[~y], alternative="greater").statistic / (y.sum() * (~y).sum()))
out = {}
# cold
c = rows("cold_all_fp32"); ex = np.array([r["exact"] for r in c]); ebs = np.array([r["exact_by_step"] for r in c], bool); q = np.array([r["q_final"] for r in c]); fe = np.array([r["first_exact"] for r in c])
resH = np.array([r["resid_H_by_step"][-1] for r in c]); resHrel = np.array([r["resid_H_rel_by_step"][-1] for r in c]); tok = np.array([r["token_acc"] for r in c])
ever = ebs.any(1); lost = (ever & ~ex).sum(); gained = (~ebs[:, 0] & ex).sum(); regress = (ebs[:, 0] & ~ex).sum()
print(f"COLD (fixed init, D16, fp32; n {len(c)}, 400 tasks): exact {100*ex.mean():.2f} % ({ex.sum()}/{len(c)}); exact by step " + " ".join(f"{100*ebs[:, t].mean():.1f}" for t in range(16)))
print(f"  first-exact of the solved: step 1 {int((fe[ex]==0).sum())}, step 2 {int((fe[ex]==1).sum())}, later {int((fe[ex]>1).sum())}; ever-exact {100*ever.mean():.2f} %; solved at step 1 and lost by 16: {int(regress)}; lost at any point {int(lost)}; gained after step 1: {int(gained)}; exact_stable {100*np.mean([r['exact_stable'] for r in c]):.2f} %")
print(f"  token accuracy {tok.mean():.3f} (solved {tok[ex].mean():.3f} / unsolved {tok[~ex].mean():.3f}); the HALTING HEAD: AUC(q_final, exact) {auc(q, ex):.4f}; q >= 0 on {100*(q>=0).mean():.1f} %; precision {100*(ex[q>=0]).mean():.1f} % recall {100*((q>=0)&ex).sum()/ex.sum():.1f} %; agreement (q>=0 <=> exact) {100*((q>=0)==ex).mean():.1f} %")
print(f"  carried-state residual |dz_H| at step 16: solved {resH[ex].mean():.4f} vs unsolved {resH[~ex].mean():.4f}; AUC(-resid, exact) {auc(-resH, ex):.4f}; relative {resHrel[ex].mean():.4f} vs {resHrel[~ex].mean():.4f}")
out["cold"] = dict(n=int(len(c)), exact=float(ex.mean()), by_step=[float(x) for x in ebs.mean(0)], ever=float(ever.mean()), lost=int(lost), regress_from_step1=int(regress), gained=int(gained), q_auc=auc(q, ex), q_agree=float(((q>=0)==ex).mean()), res_auc=auc(-resH, ex))
# ID-zero control
p = rows("cold_all_fp32_pid0"); ex0 = np.array([r["exact"] for r in p]); tok0 = np.array([r["token_acc"] for r in p]); q0 = np.array([r["q_final"] for r in p])
print(f"ID-ZERO CONTROL (the puzzle identifier replaced by the blank row 0): exact {100*ex0.mean():.2f} % ({ex0.sum()}/{len(p)}); token accuracy {tok0.mean():.3f}; q >= 0 on {100*(q0>=0).mean():.1f} %; the identity-conditioned 30.55 -> {100*ex0.mean():.2f}")
out["pid0"] = dict(exact=float(ex0.mean()), token_acc=float(tok0.mean()))
# retention
t = rows("retain_all_fp32"); de = np.concatenate([r["demo_exact"] for r in t]); alld = np.array([r["all_demos_exact"] for r in t]); ce = np.array([r["cold_exact"] for r in t]); ca = np.array([r["carried_exact"] for r in t])
dfe = np.concatenate([r["demo_first_exact"] for r in t])
print(f"RETENTION / the training pairs (n tasks {len(t)}, demo pairs {len(de)}): demo pairs reproduced {100*de.mean():.2f} % (first-exact step 1 on {100*(dfe[de.astype(bool)]==0).mean():.1f} % of them); all demos of a task exact {100*alld.mean():.1f} %; test cold {100*ce.mean():.2f} % vs carried-from-the-last-demo {100*ca.mean():.2f} % (only-cold {int((ce&~ca).sum())} / only-carried {int((~ce&ca).sum())})")
print(f"  P(test exact | all demos reproduced) {100*ce[alld].mean():.1f} % (n {alld.sum()}) vs P(test exact | some demo failed) {100*ce[~alld].mean():.1f} % (n {(~alld).sum()})")
out["retain"] = dict(demo_exact=float(de.mean()), all_demos=float(alld.mean()), test_cold=float(ce.mean()), test_carried=float(ca.mean()), p_exact_given_demos=float(ce[alld].mean()), p_exact_given_not=float(ce[~alld].mean()))
# draws
d = rows("draws_all_fp32_k8"); fx = np.array([r["fixed_exact"] for r in d]); anyx = np.array([r["any_exact"] for r in d]); b1 = np.array([r["b1_exact"] for r in d]); nx = np.array([r["n_exact_draws"] for r in d])
dr = [x for r in d for x in r["draws"]]; k0 = list(dr[0].keys()); print("  draw keys:", k0)
dex = np.array([x["exact"] for x in dr]); 
print(f"DRAWS (k = 8 random inits N(0,1), D16; n {len(d)}): fixed-init exact {100*fx.mean():.2f} %; a random-init draw b1 {100*b1.mean():.2f} %; per-draw {100*dex.mean():.2f} %; oracle@8 {100*anyx.mean():.2f} % (only-oracle {int((anyx&~fx).sum())}, fixed-only {int((fx&~anyx).sum())}); draws exact per input: all 8 on {100*(nx==8).mean():.1f} %, none on {100*(nx==0).mean():.1f} %, 1-7 on {100*((nx>0)&(nx<8)).mean():.1f} %")
rk = [k for k in k0 if "resid" in k]; qk = [k for k in k0 if k.startswith("q")]
if rk:
    rs = np.array([x[rk[0]] for x in dr]); med = np.median(rs[dex]) if dex.any() else None
    print(f"  residual selector on the draws ({rk[0]}): AUC(-resid, exact) {auc(-rs, dex):.4f}; E5 spurious (wrong draws below the correct-draw median) {100*(rs[~dex] <= med).mean():.1f} %")
    # t1r per input: the draw with the smallest residual
    t1r = []
    for r in d:
        xs = r["draws"]; j = int(np.argmin([x[rk[0]] for x in xs])); t1r.append(xs[j]["exact"])
    print(f"  residual-selected top-1 @8: {100*np.mean(t1r):.2f} % vs the fixed init {100*fx.mean():.2f} % vs oracle@8 {100*anyx.mean():.2f} %")
    out["draws"] = dict(fixed=float(fx.mean()), b1=float(b1.mean()), per_draw=float(dex.mean()), oracle8=float(anyx.mean()), res_auc=auc(-rs, dex), spurious_e5=float((rs[~dex] <= med).mean()), t1r8=float(np.mean(t1r)))
if qk:
    qs = np.array([x[qk[0]] for x in dr]); print(f"  the halting head on the draws ({qk[0]}): AUC(q, exact) {auc(qs, dex):.4f}")
# the vote
v = rows("vote40_bf16"); p1 = np.array([r["pass1"] for r in v]); p2 = np.array([r["pass2"] for r in v]); ide = np.array([r["identity_exact"] for r in v]); anyc = np.array([r["any_correct"] for r in v]); cv = np.array([r["correct_votes"]/r["n_votes"] for r in v]); nd = np.array([r["n_distinct"] for r in v])
print(f"THE AUGMENTATION VOTE (40 tasks, {len(v)} test inputs, ~1000 views each, bf16): pass@1 {100*p1.mean():.1f} %, pass@2 {100*p2.mean():.1f} % (the card: 43.00 % pass@2 on all 400); identity-view exact {100*ide.mean():.1f} %; any view correct {100*anyc.mean():.1f} %; correct-view fraction on solved-by-vote {cv[p2].mean():.2f} vs on failed {cv[~p2].mean():.3f}; distinct answers per input median {np.median(nd):.0f} (solved {np.median(nd[p2]):.0f} / failed {np.median(nd[~p2]):.0f})")
out["vote40"] = dict(n=int(len(v)), pass1=float(p1.mean()), pass2=float(p2.mean()), identity=float(ide.mean()), any_view=float(anyc.mean()))
json.dump(out, open(str(Path(__file__).resolve().parents[1] / "runs/analysis/arc_field_trm_20260910.json"), "w"), indent=1)
