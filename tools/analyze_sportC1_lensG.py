"""Ledger: sportC1 LENS G (2026-09-03; analysis-time, descriptive; artifact runs/analysis/sportC1_lensG_records_20260903.txt).
LENS G (analysis-time, descriptive, $0, disk only): the field's reproduced recipe read through our
record-level instruments, side by side with our cells. G1 selector anatomy (residual vs correctness per
draw); G2 time-to-solve (first_exact) and failure texture per rating octile on the FULL test; G5 funnel
(rho, r) evolution across X0/X0n/B0/R0 checkpoints from the k256 screens."""
import json, numpy as np
from pathlib import Path
R = Path("runs"); L = []
def say(s=""): L.append(s); print(s)
def recs(p):
    p = Path(p); q = p / "records_all.npz"
    z = dict(np.load(q, allow_pickle=True)) if q.exists() else None
    if z is None:
        parts = sorted(p.glob("records_s*.npz"))
        if not parts: return None
        arrs = [dict(np.load(x, allow_pickle=True)) for x in parts]; keys = [k for k in arrs[0] if all(k in a for a in arrs)]
        z = {k: np.concatenate([a[k] for a in arrs]) for k in keys}
    o = np.argsort(z["idx"], kind="stable"); return {k: (v[o] if hasattr(v, "shape") and v.shape and v.shape[0] == len(o) else v) for k, v in z.items()}
def octs(r): q = np.quantile(r, np.linspace(0, 1, 9)); q[-1] += 1; return q
def auc(pos, neg):
    """P(score_pos < score_neg) — for residuals, lower = 'more converged'; AUC of 'correct has lower residual'."""
    from scipy import stats
    if len(pos) == 0 or len(neg) == 0: return None
    u = stats.mannwhitneyu(pos, neg, alternative="less").statistic   # U1 = #pairs with pos > neg
    return 1.0 - float(u / (len(pos) * len(neg)))                     # P(resid_correct < resid_wrong)

say("=" * 110); say("LENS G — the field recipe (X0) through our record-level instruments vs our cells (2026-09-03; descriptive)"); say("=" * 110)
# ---- G1 selector anatomy ----
say("\n== G1. SELECTOR ANATOMY (20k scan, 128 draws/puzzle): does the convergence residual separate correct from wrong draws? ==")
say("  arm | draws n | frac correct | median resid correct / wrong | P(resid correct < resid wrong) | puzzles with >=1 correct draw | of those: min-resid draw correct | t1r@128 | b1 | verified@128")
for arm, p in (("X0 (field, no RI)", R/"sxscan_psportC1X0"), ("R0 (ours, field regime, RI)", R/"sxscan_psportC1R0"), ("A0 (ours, RI, 30k)", R/"sxscan_psportC1A0"),
               ("B0 (ours, z-norm, MEMORIZED grid)", R/"sxscan_psportC1B0"), ("P3s1 (d96 pilot, RI)", R/"sxscan_psportC0P3s1"), ("P1 (d96 pilot, no RI, memorized)", R/"sxscan_psportC0P1")):
    z = recs(p)
    if z is None or "mi_resid_k" not in z: say(f"  {arm}: no per-draw residuals"); continue
    ex = z["mi_exact_k"].astype(bool); rs = z["mi_resid_k"].astype(np.float64)
    fin = np.isfinite(rs); ex_f = ex & fin; wr_f = (~ex) & fin
    a = auc(rs[ex_f], rs[wr_f])
    has = ex.any(1); best = np.argmin(np.where(fin, rs, np.inf), axis=1); pick = ex[np.arange(len(ex)), best]
    say(f"  {arm:34s} | {ex.size:8d} | {ex.mean():.3f} | {np.median(rs[ex_f]):.4g} / {np.median(rs[wr_f]):.4g} | {('%.3f' % a) if a is not None else '  -  '} | {has.mean()*100:6.2f}% | {pick[has].mean()*100:6.2f}% | {pick.mean()*100:6.2f} | {ex[:,0].mean()*100:6.2f} | {(z['cold_exact'].astype(bool) | has).mean()*100:6.2f}")
    # per-octile AUC and 'converged-wrong' share: wrong draws whose residual is below the correct-draw median
    rat = z["rating"]; q = octs(rat); cells = []
    for b in range(8):
        m = (rat >= q[b]) & (rat < q[b+1]); e = ex[m]; r_ = rs[m]; f = np.isfinite(r_)
        if (e & f).sum() and ((~e) & f).sum():
            thr = np.median(r_[e & f]); cw = float((r_[(~e) & f] <= thr).mean())
            cells.append(f"o{b+1}: AUC {auc(r_[e & f], r_[(~e) & f]):.2f} wrong-below-median {100*cw:4.1f}%")
    say("      per octile: " + " | ".join(cells))
say("  READ: P=.5 = the residual carries no information about correctness; P->1 = converged implies correct. The selector fails whenever ANY of the")
say("        128 draws is a converged-WRONG state (a spurious attractor): 'wrong-below-median' = the per-draw spurious-attractor rate.")

# ---- G2 time-to-solve + failure texture on the FULL test ----
say("\n== G2. TIME-TO-SOLVE (first_exact along the cold trajectory) and FAILURE TEXTURE per rating octile, FULL test (422,786) ==")
FULL = {"X0 @D64": R/"sxeval_psportC1X0/full_vsel_t64", "X0 @D16": R/"sxeval_psportC1X0/full_vsel_t16", "X0n @D64": R/"sxeval_psportC1X0n/full_vsel_t64",
        "R0 (vsel=50k)": R/"sxeval_psportC1R0/full_vsel_t64", "B0 (vsel A:20k)": R/"sxeval_psportC1B0/full_vsel_t64", "B0 final (memorized)": R/"sxeval_psportC1B0/full_final_t64",
        "A0 (30k)": R/"sxeval_psportC1A0/full_vsel_t64", "P3s1 d96": R/"sxeval_psportC0P3s1/full_t64", "D4 canvas": R/"sxeval_psportBr2bD4/full_t64", "C3X canvas": R/"sxeval_psportBr2bC3X/full_t64"}
say("  arm | cold | solved: median first_exact step (o1..o8) | p90 first_exact (all solved) | first_valid==first_exact frac | FAILED: median violations (o1..o8) | median cells correct/81 | valid_wrong")
for name, p in FULL.items():
    z = recs(p)
    if z is None: say(f"  {name}: missing"); continue
    ce = z["cold_exact"].astype(bool); fe = z["first_exact"]; rat = z["rating"]; q = octs(rat)
    med = []; medv = []
    for b in range(8):
        m = (rat >= q[b]) & (rat < q[b+1])
        med.append(f"{np.median(fe[m & ce]):4.0f}" if (m & ce).any() else "  - ")
        medv.append(f"{np.median(z['violations'][m & ~ce]):4.0f}" if (m & ~ce).any() and "violations" in z else "  - ")
    fv = z["first_valid"] if "first_valid" in z else None
    same = float(np.mean(fv[ce] == fe[ce])) if fv is not None and ce.any() else float("nan")
    p90 = np.percentile(fe[ce], 90) if ce.any() else float("nan")
    cells = np.median(z["cells"][~ce]) if "cells" in z and (~ce).any() else float("nan")
    vw = float(np.mean((z["violations"] == 0) & ~ce)) if "violations" in z else float("nan")
    say(f"  {name:22s} | {100*ce.mean():5.2f} | {' '.join(med)} | {p90:4.0f} | {same:.3f} | {' '.join(medv)} | {cells:5.1f} | {vw:.4f}")
say("  READ: first_exact = the outer step at which the cold trajectory first equals the solution (propagation depth); growth with rating = search/propagation")
say("        cost rising with difficulty; a flat profile = the map's solves are 'one-shot' regardless of rating. Violations on failures = how far the stuck")
say("        state is from valid (D5 texture): near-valid (<10) vs partial propagation (>20).")
# depth gain per octile for X0 D16 -> D64 (paired)
z16 = recs(R/"sxeval_psportC1X0/full_vsel_t16"); z64 = recs(R/"sxeval_psportC1X0/full_vsel_t64")
if z16 is not None and z64 is not None and np.array_equal(z16["idx"], z64["idx"]):
    rat = z64["rating"]; q = octs(rat); c16 = z16["cold_exact"].astype(bool); c64 = z64["cold_exact"].astype(bool)
    say("  X0 depth dividend D16 -> D64 per octile (paired): " + " | ".join(f"o{b+1} {100*c16[(rat>=q[b])&(rat<q[b+1])].mean():5.1f}->{100*c64[(rat>=q[b])&(rat<q[b+1])].mean():5.1f}" for b in range(8)))
    fe64 = z64["first_exact"]; late = (c64 & ~c16)
    say(f"  puzzles solved only at D64: {late.sum()} ({100*late.mean():.2f}%); their median first_exact {np.median(fe64[late]):.0f} (p90 {np.percentile(fe64[late],90):.0f}); regressions D16->D64: {int((c16 & ~c64).sum())}")

# ---- G5 funnel evolution across checkpoints (k256 screens) ----
say("\n== G5. FUNNEL (rho, r) EVOLUTION ACROSS TRAINING (strat-512 screens, k=256 draws; lens-B fit on draws <= 64, checked at 128) ==")
def fit_rho_r(fh, k_fit=64):
    fh = np.asarray(fh); hit = (fh >= 0) & (fh < k_fit); t = fh[hit]; n_c = int(np.sum(~hit)); best = (-np.inf, None, None)
    for rho in np.linspace(0.02, 1.0, 50):
        for r in np.geomspace(1e-3, 0.95, 60):
            ll = np.sum(np.log(rho) + t * np.log1p(-r) + np.log(r)) if len(t) else 0.0
            ll += n_c * np.log(max(1 - rho + rho * (1 - r) ** k_fit, 1e-300))
            if ll > best[0]: best = (ll, rho, r)
    return best[1], best[2]
for arm in ("X0", "X0n", "B0", "B1", "R0", "A0"):
    rows = []
    for p in sorted(R.glob(f"sxscreen_psportC1{arm}_*")):
        s = json.load(open(p/"summary_all.json")); z = recs(p)
        if z is None or "mi_first_hit" not in z: continue
        tag = p.name.split("_")[-1]; ck = s["ckpt"].split("/")[-2][-3:] + "/" + s["ckpt"].split("/")[-1].replace("ckpt_","").replace(".pkl","")
        rho, r = fit_rho_r(z["mi_first_hit"]); hard = z["rating"] >= np.quantile(z["rating"], .75)
        rho_h, r_h = fit_rho_r(z["mi_first_hit"][hard])
        rows.append(f"{tag}@{ck}: cold {100*s['exact_acc']:5.1f} v128 {100*s['vote_at_k'].get('128',0):5.1f} v256 {100*s['vote_at_k'].get('256',0):5.1f} | all rho {rho:.2f} r {r:.3f} | hardest-quartile rho {rho_h:.2f} r {r_h:.3f}")
    say(f"  {arm}: " + " || ".join(rows))
say("  READ: rho = reachable fraction (any of 128 random inits reaches the solution), r = per-draw hit rate on reachable puzzles. The field's map is")
say("        wide-and-fast (rho ~1, r .5-.9) from 15k on; our maps stay at rho .35-.45 at every checkpoint — the binding limit is REACHABILITY, not precision.")
Path("runs/analysis").mkdir(exist_ok=True); Path("runs/analysis/sportC1_lensG_records_20260903.txt").write_text("\n".join(L) + "\n"); print("\nartifact -> runs/analysis/sportC1_lensG_records_20260903.txt")
