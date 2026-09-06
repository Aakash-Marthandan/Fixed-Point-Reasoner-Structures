"""Instrument-portability analysis of the public field checkpoints (registration: Plan_2026-09-05_FieldCheckpoints_Instruments.md).
Reads runs/field_ckpts/out/*, mirrors the decoder lens's statistics (tools/analyze_sportC2_ecc.py), writes analysis/field_ckpts_<date>.{txt,json}.
Descriptive; every row labeled with its set / protocol; NO-DATA where an output is absent."""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path
import numpy as np
H = Path(__file__).resolve().parent; FC = H.parent; ROOT = FC.parents[1]
sys.path.insert(0, str(ROOT / "tools")); sys.path.insert(0, str(ROOT / "src"))
from analyze_sportC2_ecc import peel, violations, logistic_g50, GBINS, RBANDS      # the lens's own functions
from qhrrn2 import sudoku_extreme as SX
OUT = FC / "analysis"; OUT.mkdir(exist_ok=True); DATE = time.strftime("%Y%m%d")
L = []; J = {}
def say(s=""): L.append(str(s)); print(s, flush=True)
def pp(x): return "  -  " if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{100*x:5.1f}"
def f(x, w=6, p=3): return " " * (w - 1) + "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:{w}.{p}f}"
def load(name):
    d = FC / "out" / name
    if not (d / "done.json").exists(): return None
    z = dict(np.load(d / "records_all.npz", allow_pickle=True)); z["_proto"] = json.loads((d / "summary.json").read_text())["proto"]; return z
def auc(pos, neg):
    from scipy import stats
    if len(pos) == 0 or len(neg) == 0: return None
    return float(stats.mannwhitneyu(pos, neg, alternative="greater").statistic / (len(pos) * len(neg)))
MODELS = [("hrm", "HRM-pub 27M (sapientinc)"), ("trm", "TRM-pub 5M (alphaXiv, 79.4)"), ("trmc", "TRM-pub 5M (CGAR, 86.0)"), ("eqr", "EqR-pub 5M (locuslab)")]
def cold_name(t, D=64): return {"eqr": f"eqr_cold_scan20k_D{D}_n05"}.get(t, f"{t}_cold_scan20k_D{D}")
def main():
    d = SX.load_prepared(ROOT / "data/sudoku_extreme/sudoku_extreme_seed0.npz"); Q, A, R = d["test_q"], d["test_a"], d["test_rating"]; G = (Q != 0).reshape(len(Q), -1).sum(1)
    lens = json.loads((ROOT / "runs/analysis/sportC2_ecc_20260905.json").read_text())
    say("=" * 120); say(f"PUBLIC FIELD CHECKPOINTS THROUGH THE INSTRUMENT SUITE (descriptive; {DATE}; registration Plan_2026-09-05_FieldCheckpoints_Instruments.md)"); say("=" * 120)
    say("\n== A. INVENTORY (outputs present) ==")
    inv = sorted(p.name for p in (FC / "out").iterdir() if (p / "done.json").exists()); say("  " + ", ".join(inv)); J["inventory"] = inv
    # ---- B. port ----
    say("\n== B. PORT VERIFICATION (I12 / P14): alphaXiv TRM weights in OUR JAX trm_cell vs their PyTorch fp32, 64 strat puzzles ==")
    vp = FC / "out/verify_port/summary.json"; ctl = FC / "out/verify_port/ctrl_fp64_vs_fp32.json"
    if vp.exists():
        rows = json.loads(vp.read_text())["rows"]; c = json.loads(ctl.read_text()) if ctl.exists() else None
        say("  step | JAX-vs-torch max|dlogit| (scale) | mean | argmax agree | exact agree || torch fp64-vs-fp32 max|d| | argmax agree | exact agree")
        for r in rows[:16]:
            cr = c[r["t"] - 1] if c and len(c) >= r["t"] else None
            say(f"  {r['t']:4d} | {r['max_abs']:9.4f} ({r['scale']:.1f}) | {r['mean_abs']:.5f} | {pp(r['argmax_agree'])} | {pp(r['exact_agree'])} || " + (f"{cr['max_abs']:9.4f} | {pp(cr['argmax_agree'])} | {pp(cr['exact_agree'])}" if cr else "-"))
        J["port"] = dict(rows=rows, ctrl=c)
        say("  READ: an exact port shows a step-1 deviation at fp32 round-off and the SAME geometric growth as the fp64-vs-fp32 control (the recurrence amplifies round-off ~10x per outer step on churning puzzles).")
    else: say("  NO-DATA")
    # ---- C. headline ----
    say("\n== C. HEADLINE (I1 / I7): exact accuracy on the natural 20k subsample (seed 20260822; identical to X0's and every native scan), fp32, their protocol ==")
    say("  model | set | protocol | n | exact@D1/2/4/8/16/32/64 | regressions 16->64 | q_halt acc@16 | valid-wrong | given-violating valid | CI(±pp)@16")
    C = {}
    for t, desc in MODELS:
        for nm, lab in ([(cold_name(t), "trunc RI std1, noise .5" if t == "eqr" else "fixed init")] + ([("eqr_cold_scan20k_D64_n0", "trunc RI std1, noise 0")] if t == "eqr" else []) + ([("trm_cold_scan20k_D16_bf16", "bf16 forward")] if t == "trm" else [])):
            z = load(nm)
            if z is None: say(f"  {t:4s} | {nm}: NO-DATA"); continue
            E = z["exact_by_step"].astype(bool); Dm = E.shape[0]; n = E.shape[1]
            steps = [s for s in (1, 2, 4, 8, 16, 32, 64) if s <= Dm]; accs = {s: float(E[s - 1].mean()) for s in steps}
            reg = float((E[15] & ~E[Dm - 1]).mean()) if Dm >= 64 else None
            e16 = E[min(15, Dm - 1)]; qa = float(((z["q_by_step"][min(15, Dm - 1)].astype(np.float32) >= 0) == e16).mean())
            vw = float(((z["violations"] == 0) & ~E[-1]).mean()); gv = float(((z["violations"] == 0) & ~z["givens_kept"]).mean()); ci = 196 * math.sqrt(accs[min(16, Dm)] * (1 - accs[min(16, Dm)]) / n)
            C[nm] = dict(model=t, protocol=lab, n=int(n), acc=accs, regressions=reg, q_halt_acc16=qa, valid_wrong=vw, given_violating=gv)
            say(f"  {t:4s} | {z['_proto']['set']:8s} | {lab:24s} | {n:5d} | " + "/".join(pp(accs[s]).strip() for s in steps) + f" | {pp(reg)} | {pp(qa)} | {pp(vw)} | {pp(gv)} | ±{ci:.2f}")
    # strat512 rows (CPU, D16)
    for t, _ in MODELS:
        z = load({"eqr": "eqr_cold_strat512_D16_n05"}.get(t, f"{t}_cold_strat512_D16"))
        if z is not None:
            E = z["exact_by_step"].astype(bool); C[f"{t}_strat512"] = dict(acc16=float(E[-1].mean())); say(f"  {t:4s} | strat512 (the calibration set)  | exact@16 {pp(E[-1].mean())}")
    say("  refs (same 20k, t=64): X0 (our reproduction) cold 92.86 / EqR paper 93.0 @D64 B1; our natives R3 43.57-class; TRM paper 87.4; alphaXiv card 79.37; CGAR card 86.02; HRM per TRM's table 55.0")
    J["headline"] = C
    # ---- D. thresholds / yields ----
    say("\n== D. ERASURE THRESHOLDS AND SEARCH-CLASS YIELD (E1; cold at D16 and D64 on the 20k) ==")
    say("  model@D | " + " | ".join(f"[{lo},{hi})" for lo, hi in GBINS) + " | g50 (width) | P(cold|r0) | P(cold|r>0) | bands 0/1-9/10-29/30-59/60+ | class (R-A-4 letters)")
    D1 = {}
    def e1(name, cold, gg, rr):
        g50, _, w = logistic_g50(gg, cold); cells = [pp(cold[(gg >= lo) & (gg < hi)].mean()) if ((gg >= lo) & (gg < hi)).any() else "  -  " for lo, hi in GBINS]
        y = float(cold[rr > 0].mean()); g_in = g50 if (g50 is not None and 17 <= g50 <= 35) else None      # R-A-4: a threshold counts only inside the tested 17..35 givens
        cls = "DECIMATING" if (g_in is None and y >= .70) else ("SOFT" if (g_in is not None and g_in >= 24 and y <= .45) else "MIXED")
        bands = [float(cold[(rr >= lo) & (rr < hi)].mean()) if ((rr >= lo) & (rr < hi)).any() else None for lo, hi in RBANDS]
        say(f"  {name:9s} | " + " | ".join(cells) + f" | {f(g50,5,1)} ({f(w,4,1)}) | {pp(cold[rr == 0].mean())} | {pp(y)} | " + "/".join(pp(b).strip() for b in bands) + f" | {cls}")
        return dict(g50=g50, width=w, p_r0=float(cold[rr == 0].mean()), yield_search=y, bands=bands, cls=cls, cold=float(cold.mean()))
    for t, _ in MODELS:
        for nm in [cold_name(t)] + (["eqr_cold_scan20k_D64_n0"] if t == "eqr" else []):
            z = load(nm)
            if z is None: continue
            E = z["exact_by_step"].astype(bool); gg = G[z["idx"]]; rr = R[z["idx"]]; tag = t + ("(n0)" if nm.endswith("n0") else "")
            for D in (16, 64):
                if E.shape[0] >= D: D1[f"{tag}@{D}"] = e1(f"{tag}@{D}", E[D - 1], gg, rr)
    for k in ("X0", "R3", "W0", "B0c", "X1", "X2"):
        e = lens["E1"].get(k)
        if e: say(f"  {k+'(ref)':9s} | {'':45s} | {f(e['g50_cold'],5,1)} ({f(e['width_cold'],4,1)}) | {pp(e['p_cold_r0'])} | {pp(e['p_cold_rpos'])} | " + "/".join(pp(e['bands'].get(f'{lo}-{hi}')).strip() for lo, hi in RBANDS) + " | (lens 2026-09-05, t=64)")
    J["E1"] = D1
    # ---- D2. the classical ladder on REF500 ----
    ref = ROOT / "runs/analysis/reference_decoders.npz"
    if ref.exists():
        rz = np.load(ref); ridx = rz["idx"]; rrat = rz["rating"]; say("\n  LADDER on the 500-puzzle rating-stratified slice (cold@D64; reference decoders + public checkpoints + our natives + X0):")
        rows = {"peeling": rz["peel"].astype(bool), "BP-40": rz["bp"].astype(bool), "BP+decimation": rz["bpd"].astype(bool)}
        for t, _ in MODELS:
            z = load(cold_name(t))
            if z is None or z["exact_by_step"].shape[0] < 64: continue
            pos = {int(v): i for i, v in enumerate(z["idx"])}; sel = np.array([pos.get(int(v), -1) for v in ridx]); ok = sel >= 0
            rows[t] = np.where(ok, z["exact_by_step"][63][np.maximum(sel, 0)], False)
        for k, p in (("X0", "sxscan_psportC1X0"), ("R3", "sxscan_psportC2R3"), ("B0c", "sxscan_psportC1B0_vselA20k"), ("W0", "sxscan_psportC2W0")):
            q = ROOT / "runs" / p / "records_all.npz"
            if q.exists():
                zz = np.load(q); pos = {int(v): i for i, v in enumerate(zz["idx"])}; sel = np.array([pos.get(int(v), -1) for v in ridx]); rows[k] = zz["cold_exact"].astype(bool)[sel]
        say("  decoder | solved | yield r>0 | bands 0/1-9/10-29/30-59/60+ | Jaccard(stall, BP stall)")
        lad = {}
        for k, y in rows.items():
            bands = [float(y[(rrat >= lo) & (rrat <= hi)].mean()) for lo, hi in ((0, 0), (1, 9), (10, 29), (30, 59), (60, 10**9))]; jac = float((~y & ~rows["BP-40"]).sum() / max(1, (~y | ~rows["BP-40"]).sum()))
            lad[k] = dict(solved=float(y.mean()), yield_search=float(y[rrat > 0].mean()), bands=bands, jac_bp=jac)
            say(f"  {k:14s} | {pp(y.mean())} | {pp(y[rrat > 0].mean())} | " + "/".join(pp(b).strip() for b in bands) + f" | {jac:.3f}")
        J["ladder500"] = lad
    # ---- F. halting head ----
    say("\n== F. THE HALTING HEAD AS A LEARNED VERIFIER (I8): AUC(q_halt, exact), their q_halt accuracy, precision/recall at 0 ==")
    say("  model | D | AUC | q_halt acc | precision@0 | recall@0 | P(q>=0) | exact")
    Fh = {}
    for t, _ in MODELS:
        z = load(cold_name(t))
        if z is None: continue
        for D in (16, 64):
            if z["exact_by_step"].shape[0] < D: continue
            e = z["exact_by_step"][D - 1].astype(bool); q = z["q_by_step"][D - 1].astype(np.float64); a_ = auc(q[e], q[~e]); pos = q >= 0
            Fh[f"{t}@{D}"] = dict(auc=a_, acc=float((pos == e).mean()), prec=float((pos & e).sum() / max(1, pos.sum())), rec=float((pos & e).sum() / max(1, e.sum())), p_pos=float(pos.mean()), exact=float(e.mean()))
            r = Fh[f"{t}@{D}"]; say(f"  {t:4s} | {D:2d} | {f(a_,5,3)} | {pp(r['acc'])} | {pp(r['prec'])} | {pp(r['rec'])} | {pp(r['p_pos'])} | {pp(r['exact'])}")
    J["halting"] = Fh
    # ---- G. draws ----
    say("\n== G. LIST DECODING AND THE SELECTOR (E4/E5) with k = 8 random-init draws N(0,1) on the 5k subsample (labeled: k=8 vs the lens's k=128) ==")
    say("  model | cold@16 | b1 | verified@1/2/4/8 (cold ∪ draws) | rho@8 | P(any8|cold fails) | r_i bins 0/(0,.5]/(.5,1) /1 | AUC resid (ours) / spurious / t1r@8 / verified@8 || AUC logit-score (EqR's) / spurious / t1r@8")
    Gd = {}
    for t0, _ in MODELS:
        for nm in ([f"{t0}_draws_sub5k_D16_k8"] if t0 != "eqr" else ["eqr_draws_sub5k_D16_k8_n05", "eqr_draws_sub5k_D16_k8_n0"]):
            t = t0 + ("(n0)" if nm.endswith("_n0") else "")
            z = load(nm); zc = load(cold_name(t0) if not nm.endswith("_n0") else "eqr_cold_scan20k_D64_n0")
            if z is None: say(f"  {t:4s}: NO-DATA ({nm})"); continue
            ex = z["mi_exact_k"].astype(bool); ok = z["mi_verified_k"].astype(bool); n, K = ex.shape
            cold = np.zeros(n, bool)
            if zc is not None:
                pos = {int(v): i for i, v in enumerate(zc["idx"])}; sel = np.array([pos[int(v)] for v in z["idx"]]); cold = zc["exact_by_step"][15][sel].astype(bool)
            fh = z["mi_first_hit"]; vk = {k: float((cold | ((fh >= 0) & (fh < k))).mean()) for k in (1, 2, 4, 8)}; reach = cold | (fh >= 0); ri = ex.mean(1)
            bins = [float((ri == 0).mean()), float(((ri > 0) & (ri <= .5)).mean()), float(((ri > .5) & (ri < 1)).mean()), float((ri == 1).mean())]
            resc = float((fh[~cold] >= 0).mean()) if (~cold).any() else None
            def sel_stats(rs):
                rs = rs.astype(np.float64); fin = np.isfinite(rs); ef, wf = ex & fin, (~ex) & fin
                a_ = auc(-rs[ef], -rs[wf]); thr = np.median(rs[ef]) if ef.any() else np.nan; sp = float((rs[wf] <= thr).mean()) if wf.any() else None
                best = np.argmin(np.where(fin, rs, np.inf), 1); pick = ex[np.arange(n), best]; return dict(auc=a_, spurious=sp, t1r=float(pick.mean()), verified=float(ex.any(1).mean()))
            s1 = sel_stats(z["mi_resid_k"]); s2 = sel_stats(z["mi_resid_logit_k"])
            order = np.argsort(z["mi_resid_logit_k"].astype(np.float64), 1)[:, :4]; top4 = ex[np.arange(n)[:, None], order]   # EqR's convergence_top_k=4 at B=8
            s2["top4_mean"] = float(top4.mean()); s2["top4_any"] = float(top4.any(1).mean())
            Gd[t] = dict(cold16=float(cold.mean()), b1=float(ex[:, 0].mean()), vk=vk, rho=float(reach.mean()), rescue=resc, bins=bins, sel_ours=s1, sel_logit=s2, n=int(n))
            say(f"  {t:7s} | {pp(cold.mean())} | {pp(ex[:, 0].mean())} | " + "/".join(pp(vk[k]).strip() for k in (1, 2, 4, 8)) + f" | {pp(reach.mean())} | {pp(resc)} | " + "/".join(pp(b).strip() for b in bins) + f" | {f(s1['auc'],5,3)} / {pp(s1['spurious'])} / {pp(s1['t1r'])} / {pp(s1['verified'])} || {f(s2['auc'],5,3)} / {pp(s2['spurious'])} / {pp(s2['t1r'])} | EqR top-4-of-8 by score: mean {pp(s2['top4_mean'])} any {pp(s2['top4_any'])}")
    e5 = lens["E5"].get("X0"); e4 = lens["E4"].get("X0")
    if e5 and e4: say(f"  X0(ref, k128 20k) | rho {pp(e4['rho'])} | AUC {f(e5['auc'],5,3)} / spurious {pp(e5['spurious'])} / t1r {pp(e5['t1r'])} / verified {pp(e5['verified'])} (lens)")
    J["draws"] = Gd
    # ---- H. dynamics (E2/E3/E6) ----
    say("\n== H. DECODER DYNAMICS (E2), DECIMATION QUALITY AT STALLS (E3), CALIBRATION AT STALLS (E6) — strat-256 cold trajectories, t = 64 ==")
    say("  model | solved | cells correct step 1/2/4/8/16/32/64 solved | unsolved || entropy solved | unsolved || commit(p>.9) 1/8/64 solved | unsolved || conf-wrong 1/8/64 solved | unsolved")
    Hd = {}; H3 = {}; H6 = {}
    for t, _ in MODELS:
        for nm in ([f"{t}_dyn_strat256_D64"] if t != "eqr" else ["eqr_dyn_strat256_D64_n05", "eqr_dyn_strat256_D64_n0"]):
            z = load(nm)
            if z is None: continue
            key = t + ("(n0)" if nm.endswith("n0") else ""); P = z["preds"].astype(np.int16); T, n, _ = P.shape; sol = z["sol"].astype(np.int16); puz = z["puz"].astype(np.int16); ng = puz == 0; nng = np.maximum(ng.sum(1), 1)
            corr = P == sol[None]; mp = z["maxp"].astype(np.float32); en = z["ent"].astype(np.float32)
            cc = (corr & ng[None]).sum(2) / nng; ent = (en * ng[None]).sum(2) / nng; com = ((mp > .9) & ng[None]).sum(2) / nng; cw = ((mp > .9) & ~corr & ng[None]).sum(2) / nng
            solved = corr[-1].all(1); fe = np.where(corr.all(2).any(0), corr.all(2).argmax(0), -1)
            syn = np.stack([violations(P[i].reshape(n, 9, 9)) for i in range(T)])
            ever = np.zeros((n, 81), bool); unpeel = np.zeros(n); fw = np.zeros((T, n)); fc = np.zeros((T, n))
            for i in range(T):
                if i: ch = (P[i] != P[i - 1]) & ng; fc[i] = (ch & corr[i]).sum(1) / nng; fw[i] = (ch & ~corr[i]).sum(1) / nng; unpeel += ((ever & ~corr[i]) & ng).sum(1)
                ever |= corr[i]
            sidx = [0, 1, 3, 7, 15, 31, 63]; m_ = lambda arr, mask, ii: [float(arr[i, mask].mean()) if mask.any() else None for i in ii]
            o = dict(solved=float(solved.mean()), cells_s=m_(cc, solved, sidx), cells_u=m_(cc, ~solved, sidx), ent_s=m_(ent, solved, sidx), ent_u=m_(ent, ~solved, sidx), com_s=m_(com, solved, [0, 7, 63]), com_u=m_(com, ~solved, [0, 7, 63]),
                     cw_s=m_(cw, solved, [0, 7, 63]), cw_u=m_(cw, ~solved, [0, 7, 63]), mono_solved=float((unpeel == 0)[solved].mean()) if solved.any() else None,
                     unpeel_s=float((unpeel / nng)[solved].mean()) if solved.any() else None, unpeel_u=float((unpeel / nng)[~solved].mean()) if (~solved).any() else None,
                     flips_w_u=float(fw[32:, ~solved].sum(0).mean()) if (~solved).any() else None, flips_w_s=float(fw[32:, solved].sum(0).mean()) if solved.any() else None, flips_c_u=float(fc[32:, ~solved].sum(0).mean()) if (~solved).any() else None,
                     syn_u=m_(syn, ~solved, sidx), syn_osc=float((syn[32:, ~solved].max(0) - syn[32:, ~solved].min(0)).mean()) if (~solved).any() else None,
                     syn_mono=float(np.mean([bool(np.all(np.diff(syn[:, b]) <= 0)) for b in np.where(~solved)[0]])) if (~solved).any() else None,
                     fe_med=float(np.median(fe[solved])) if solved.any() else None, fe_p90=float(np.percentile(fe[solved], 90)) if solved.any() else None)
            Hd[key] = o; r_ = lambda xs: "/".join("-" if v is None else f"{100*v:.0f}" for v in xs); r3 = lambda xs: "/".join("-" if v is None else f"{v:.2f}" for v in xs)
            say(f"  {key:8s} | {pp(o['solved'])} | {r_(o['cells_s'])} | {r_(o['cells_u'])} || {r3(o['ent_s'])} | {r3(o['ent_u'])} || {r_(o['com_s'])} | {r_(o['com_u'])} || {r_(o['cw_s'])} | {r_(o['cw_u'])}")
            # E3 at stalls
            e3 = {}; U = ~solved; p9last = mp[-1]
            for tau in (.9, .99):
                if not U.any(): break
                com_m = (p9last > tau) & ng; ncom = com_m.sum(1) / nng; wrong = (com_m & ~corr[-1]).sum(1) / np.maximum(com_m.sum(1), 1)
                gg0 = np.where(com_m, P[-1], puz).reshape(n, 9, 9)[U]; pg_, st_ = peel(gg0.astype(np.int64)); okp = (st_ == 1) & (pg_ == sol.reshape(n, 9, 9)[U]).all((1, 2))
                e3[str(tau)] = dict(n_stalled=int(U.sum()), committed=float(ncom[U].mean()), wrong=float(wrong[U].mean()), all_correct=float((wrong[U] == 0).mean()), solves=float(okp.mean()), stuck=float((st_ == 0).mean()), contra=float((st_ == 2).mean()))
            H3[key] = e3
            # E6 calibration at stalls: top-5 confident non-given cells at t=64 on stalled puzzles
            if U.any():
                conf = np.where(ng, p9last, -1)[U]; top = np.argsort(-conf, 1)[:, :5]; rowsU = np.arange(U.sum())[:, None]
                c5 = corr[-1][U][rowsU, top].mean(); mc = conf[rowsU, top].mean(); e1s = float(ent[0, U].mean()); cwf = float(cw[-1, U].mean())
                H6[key] = dict(top5=float(c5), conf=float(mc), gap=float(mc - c5), ent1=e1s, cw=cwf, n=int(U.sum()))
    say("\n  REVISION / MONOTONICITY / SYNDROME: model | monotone solved | un-peel per cell (solved/unsolved) | flips-to-wrong last 32 (unsolved/solved) | flips-to-correct (unsolved) | syndrome unsolved by step 1/2/4/8/16/32/64 | late osc | syndrome monotone frac | first_exact med/p90")
    for key, o in Hd.items():
        say(f"  {key:8s} | {pp(o['mono_solved'])} | {f(o['unpeel_s'],5,3)}/{f(o['unpeel_u'],5,3)} | {f(o['flips_w_u'],5,2)}/{f(o['flips_w_s'],5,2)} | {f(o['flips_c_u'],5,2)} | " + "/".join("-" if v is None else f"{v:.0f}" for v in o['syn_u']) + f" | {f(o['syn_osc'],5,1)} | {pp(o['syn_mono'])} | {f(o['fe_med'],3,0)}/{f(o['fe_p90'],3,0)}")
    for k in ("X0", "R3", "W0", "B0"):
        e = lens.get("E2", {}).get(k)
        if e: say(f"  {k+'(ref)':8s} | {pp(e['mono_solved'])} | {f(e['unpeel_per_cell_solved'],5,3)}/{f(e['unpeel_per_cell_unsolved'],5,3)} | {f(e['flips_w_last32_unsolved'],5,2)}/{f(e['flips_w_last32_solved'],5,2)} | {f(e['flips_c_last32_unsolved'],5,2)} | " + "/".join("-" if v is None else f"{v:.0f}" for v in e['syn_u']) + f" | {f(e['syn_osc_unsolved'],5,1)} | {pp(e['syn_mono_frac_unsolved'])} | {f(e['fe_med'],3,0)}/{f(e['fe_p90'],3,0)}  (lens)")
    say("\n  E3 DECIMATION QUALITY AT STALLS: model | tau | n stalled | committed frac | wrong among committed | P(all committed correct) | peeling from givens+committed: solves/stuck/contradiction")
    for key, e3 in H3.items():
        for tau in ("0.9", "0.99"):
            q = e3.get(tau)
            if q: say(f"  {key:8s} | {tau:4s} | {q['n_stalled']:4d} | {pp(q['committed'])} | {pp(q['wrong'])} | {pp(q['all_correct'])} | {pp(q['solves'])}/{pp(q['stuck'])}/{pp(q['contra'])}")
    for k in ("X0", "R3", "W0"):
        e = lens.get("E3", {}).get(k, {}).get("0.9")
        if e: say(f"  {k+'(ref)':8s} | 0.9  | {e['n_stalled']:4d} | {pp(e['committed_frac'])} | {pp(e['wrong_committed_frac'])} | {pp(e['all_committed_correct'])} | {pp(e['peel_solves'])}/{pp(e['peel_stuck'])}/{pp(e['peel_contradiction'])}  (lens)")
    say("\n  E6 CALIBRATION AT STALLS (top-5 most confident non-given cells at t=64): model | n stalled | top-5 correct | mean conf | gap | entropy step 1 | conf-wrong frac")
    for key, e in H6.items(): say(f"  {key:8s} | {e['n']:4d} | {pp(e['top5'])} | {f(e['conf'],5,3)} | {f(e['gap'],6,3)} | {f(e['ent1'],5,3)} | {pp(e['cw'])}")
    J["E2"] = Hd; J["E3"] = H3; J["E6"] = H6
    # ---- I. retention / jac / prefix / sym / train ----
    say("\n== I. RETENTION (I6), FIXED POINTS (I5), PREFIX (I11), SYMMETRY (I9), MEMORIZATION (I10) ==")
    I = {}
    for t, _ in MODELS:
        for nm in ([f"{t}_retain_strat512_D8"] if t != "eqr" else ["eqr_retain_strat512_D8_n0", "eqr_retain_strat512_D8_n05"]):
            z = load(nm)
            if z is None: continue
            rb = z["retained_by_step"].astype(bool); I[nm] = dict(by_step=[float(x) for x in rb.mean(1)], all=float(rb.all(0).mean()), final=float(rb[-1].mean()))
            say(f"  RETENTION {nm:28s} | retained at step 1..8: " + "/".join(pp(x).strip() for x in rb.mean(1)) + f" | all 8: {pp(rb.all(0).mean())} | final: {pp(rb[-1].mean())}")
        for nm in ([f"{t}_jac_strat84_D64"] if t != "eqr" else ["eqr_jac_strat84_D64_n0"]):
            z = load(nm)
            if z is None: continue
            ex = z["exact"].astype(bool); rad = z["radius_iters"][-1]; rr = z["resid_rel"]
            I[nm] = dict(rad_s=float(np.median(rad[ex])) if ex.any() else None, rad_u=float(np.median(rad[~ex])) if (~ex).any() else None, rad_s_p10=float(np.percentile(rad[ex], 10)) if ex.any() else None, frac_contractive_s=float((rad[ex] < 1).mean()) if ex.any() else None,
                         res_s=float(np.median(rr[ex])) if ex.any() else None, res_u=float(np.median(rr[~ex])) if (~ex).any() else None, exact=float(ex.mean()))
            r = I[nm]; say(f"  FIXED POINTS {nm:24s} | exact {pp(r['exact'])} | FD Jacobian radius median solved {f(r['rad_s'],5,2)} (p10 {f(r['rad_s_p10'],4,2)}; frac<1 {pp(r['frac_contractive_s'])}) / unsolved {f(r['rad_u'],5,2)} | relative latent residual solved {f(r['res_s'],7,4)} / unsolved {f(r['res_u'],7,4)}")
        zp = load({"eqr": "eqr_prefix_strat512_D16_n05"}.get(t, f"{t}_prefix_strat512_D16")); zc = load({"eqr": "eqr_cold_strat512_D16_n05"}.get(t, f"{t}_cold_strat512_D16"))
        if zp is not None and zc is not None:
            a0, a1 = float(zc["exact_by_step"][-1].mean()), float(zp["exact_by_step"][-1].mean()); I[f"{t}_prefix"] = dict(with_=a0, zeroed=a1, delta=a1 - a0)
            say(f"  PREFIX {t:4s} | exact@16 with the trained prefix {pp(a0)} | prefix zeroed {pp(a1)} | delta {100*(a1-a0):+.2f} pp")
        zs = load({"eqr": "eqr_sym_strat512_D16_n05"}.get(t, f"{t}_sym_strat512_D16"))
        if zs is not None:
            E = zs["exact_by_variant"].astype(bool); v = zs["orbit_vote_exact"].astype(bool)
            I[f"{t}_sym"] = dict(identity=float(E[0].mean()), orbit_mean=float(E.mean()), consistent=float((E.all(0) | ~E.any(0)).mean()), any=float(E.any(0).mean()), vote=float(v.mean()), gain=float(v.mean() - E[0].mean()))
            r = I[f"{t}_sym"]; say(f"  SYMMETRY {t:4s} | exact identity {pp(r['identity'])} | mean over 9 relabelings {pp(r['orbit_mean'])} | orbit-consistent {pp(r['consistent'])} | any-of-9 {pp(r['any'])} | orbit vote {pp(r['vote'])} (gain {100*r['gain']:+.2f} pp)")
    zt = load("eqr_train1k_D16_n05")
    if zt is not None:
        E = zt["exact_by_step"].astype(bool); I["eqr_train"] = dict(exact16=float(E[-1].mean()), exact1=float(E[0].mean()), given_viol=float(((zt["violations"] == 0) & ~zt["givens_kept"]).mean()))
        say(f"  MEMORIZATION EqR on its OWN seed-42 training 1k | exact@16 {pp(E[-1].mean())} | exact@1 {pp(E[0].mean())} | (test@16 from C)")
    say("\n  EXPLORATORY (post-hoc, plan §8): INIT-BASIN RADIUS — exact@16 on strat512 vs the size eps of a Gaussian perturbation of the trained init (eps 0 = cold; eps 3 ~ a random draw)")
    for t, _ in MODELS:
        z = load({"eqr": "eqr_initrad_strat512_D16_n0"}.get(t, f"{t}_initrad_strat512_D16"))
        if z is None: continue
        I[f"{t}_initrad"] = dict(eps=[float(x) for x in z["eps"]], exact=[float(x) for x in z["exact_vs_eps"]])
        say(f"  {t:4s} | " + " | ".join(f"eps {e:g}: {pp(x)}" for e, x in zip(z["eps"], z["exact_vs_eps"])))
    z = load("hrm_draws_sub1k_D64_k2")
    if z is not None:
        ex = z["mi_exact_k"].astype(bool); I["hrm_draws_D64"] = dict(b1=float(ex[:, 0].mean()), any2=float(ex.any(1).mean()), n=int(len(ex)))
        say(f"  HRM random-init draws at D=64 (1k puzzles, k=2): draw-1 exact {pp(ex[:, 0].mean())} | any of 2 {pp(ex.any(1).mean())}  (at D=16 on 5k: 3.2 %)")
    J["I"] = I
    # ---- J. cross-cell overlaps at D64 on the 20k ----
    say("\n== J. CROSS-CELL OVERLAP on the identical 20k at t/D = 64: solved-set Jaccard, McNemar discordants, and where each map's cold failures fall ==")
    sets = {}
    for t, _ in MODELS:
        z = load(cold_name(t))
        if z is not None and z["exact_by_step"].shape[0] >= 64: sets[t] = (z["idx"], z["exact_by_step"][63].astype(bool))
    for k, p in (("X0", "sxscan_psportC1X0"), ("R3", "sxscan_psportC2R3"), ("W0", "sxscan_psportC2W0"), ("B0c", "sxscan_psportC1B0_vselA20k")):
        q = ROOT / "runs" / p / "records_all.npz"
        if q.exists(): zz = np.load(q); sets[k] = (zz["idx"], zz["cold_exact"].astype(bool))
    keys = list(sets); base = sets[keys[0]][0]; al = {}
    for k, (ix, e) in sets.items():
        pos = {int(v): i for i, v in enumerate(ix)}; sel = np.array([pos.get(int(v), -1) for v in base]); al[k] = np.where(sel >= 0, e[np.maximum(sel, 0)], False)
    say("  pair | Jaccard(solved) | only-A | only-B | P(A fails | B fails) | P(B fails | A fails)")
    Jx = {}
    for i, a_ in enumerate(keys):
        for b_ in keys[i + 1:]:
            A_, B_ = al[a_], al[b_]; jac = float((A_ & B_).sum() / max(1, (A_ | B_).sum())); oa, ob = int((A_ & ~B_).sum()), int((~A_ & B_).sum())
            Jx[f"{a_}|{b_}"] = dict(jaccard=jac, only_a=oa, only_b=ob, pA_fail_given_B_fail=float((~A_[~B_]).mean()) if (~B_).any() else None, pB_fail_given_A_fail=float((~B_[~A_]).mean()) if (~A_).any() else None)
            r = Jx[f"{a_}|{b_}"]; say(f"  {a_:4s}|{b_:4s} | {jac:.3f} | {oa:6d} | {ob:6d} | {pp(r['pA_fail_given_B_fail'])} | {pp(r['pB_fail_given_A_fail'])}")
    J["overlap"] = Jx
    say("\n== K. EXPLORATORY (post-hoc, plan §8): init-basin radius (exact@16 on strat512 vs perturbation eps of the trained init) and HRM random-init draws at D64 ==")
    K = {}
    for nm in ("hrm_initrad_strat512_D16", "trm_initrad_strat512_D16", "trmc_initrad_strat512_D16", "eqr_initrad_strat512_D16_n0"):
        z = load(nm)
        if z is None: continue
        K[nm] = dict(eps=[float(x) for x in z["eps"]], exact=[float(x) for x in z["exact_vs_eps"]])
        say(f"  {nm:28s} | eps " + " / ".join(f"{e:g}: {100*x:.1f}" for e, x in zip(K[nm]["eps"], K[nm]["exact"])))
    z = load("hrm_draws_sub1k_D64_k2")
    if z is not None:
        ex = z["mi_exact_k"].astype(bool); K["hrm_draws_D64"] = dict(b1=float(ex[:, 0].mean()), any2=float(ex.any(1).mean()), n=int(ex.shape[0]))
        say(f"  HRM random-init draws at D64 (1k puzzles, k=2): draw-1 exact {pp(ex[:, 0].mean())} | any of 2 {pp(ex.any(1).mean())}   (at D16, k=8 on 5k: b1 3.2)")
    J["K"] = K
    say(f"\n(artifact {DATE})"); (OUT / f"field_ckpts_{DATE}.txt").write_text("\n".join(L) + "\n"); (OUT / f"field_ckpts_{DATE}.json").write_text(json.dumps(J, indent=1, default=float)); print("artifact ->", OUT / f"field_ckpts_{DATE}.txt")

if __name__ == "__main__":
    main()
