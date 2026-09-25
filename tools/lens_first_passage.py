#!/usr/bin/env python3
# Ledger: SOLVING AS FIRST PASSAGE (2026-09-19; registration Documentation/Note_2026-09-19_First_Passage_Tests.md, written before any
# curve was read). Banked per-puzzle records only; no model is run. T1 depth scaling from the waiting-time tail; T2 restart scaling from
# 16 restarts; T3 depth against restarts at equal compute; T4 hardness as an escape rate.
"""  .venv/bin/python tools/lens_first_passage.py --selftest
  .venv/bin/python tools/lens_first_passage.py --run"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln, gammaln, logsumexp
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]; R = ROOT / "runs"; OUT = R / "analysis/first_passage_20260919"
T0, T1FIT = 8, 64                                   # the registered fit window: left-truncated at 8, right-censored at 64
BANDS = ((0, 0), (1, 9), (10, 29), (30, 59), (60, 10 ** 9))
PP, SP, CX = "_paperfinal_pull/x/runs/", "_sudokupend_pull/", "_c8x_pull/x/runs/"
MODELS = {  # key: (depth-256 on the 5,000, k128 scan on the same 5,000, full-set 64 with per-step bits, label)
    "C5": ("sxeval_pchampC5/sub5k_t256", SP + "stage/runs/filler_sxscan128_pchampC5", None, "ours, width 192, seed 0"),
    "C7": (PP + "sxeval_pchampC7/sub5k_t256", SP + "x_final/runs/filler_sxscan128_pchampC7", PP + "filler_sxeval_pchampC7_full_t64", "ours, width 192, seed 1"),
    "C8": (CX + "sxeval_pchampC8/sub5k_t256", SP + "x_final/runs/filler_sxscan128_pchampC8", CX + "filler_sxeval_pchampC8_full_t64", "ours, width 192, seed 2"),
    "C0": ("sxeval_pchampC0/sub5k_t256", "filler_sxscan128_pchampC0", None, "ours, width 384, seed 0"),
    "C2": ("sxeval_pchampC2/sub5k_t256", "filler_sxscan128_pchampC2", "_champ_pull/filler/x_d64full_C2/runs/filler_sxeval_pchampC2_full_t64", "ours, width 384, seed 2"),
    "C4": ("sxeval_pchampC4/sub5k_t256", "filler_sxscan128_pchampC4", PP + "filler_sxeval_pchampC4_full_t64", "ours, width 384, field attention"),
    "EQR": ("sxeval_pfrontiereqr/sub5k_t256", SP + "stage/runs/filler_sxscan128_pport_eqr", "sxeval_pfrontiereqr/full_vsel_t64", "EqR released"),
    "CGAR": ("sxeval_pfrontiercgar/sub5k_t256", None, "sxeval_pfrontiercgar/full_vsel_t64", "CGAR released"),
    "TRMPUB": ("sxeval_pfrontiertrmpub/sub5k_t256", None, "sxeval_pfrontiertrmpub/full_vsel_t64", "TRM released (alphaXiv)"),
}

# ---------------- pure helpers (selftested) ----------------
def unpack(bits_packed, T):
    """(N, ceil(T/8)) uint8, LITTLE-endian within a byte (the evaluator's np.packbits(..., bitorder='little')) -> (N, T) bool."""
    return np.unpackbits(np.asarray(bits_packed, np.uint8), axis=1, bitorder="little")[:, :T].astype(bool)

def waiting_time(bits):
    """(N, T) exactness by step -> (N,) the 1-based first fully-right step; 0 = never inside the horizon."""
    return np.where(bits.any(1), bits.argmax(1) + 1, 0)

def surv(family, th, t):
    t = np.asarray(t, float)
    if family == "lomax": a, tau = np.exp(th); return (1.0 + t / tau) ** (-a)
    if family == "geometric": return np.exp(-np.exp(th[0]) * t)
    if family == "weibull": lam, k = np.exp(th); return np.exp(-((t / lam) ** k))
    raise ValueError(family)

def fit_tail(T, family, t0=T0, t1=T1FIT):
    """Maximum likelihood on the discrete waiting times in (t0, t1], LEFT-TRUNCATED at t0 (puzzles with T <= t0 are not in the sample),
    RIGHT-CENSORED at t1 (T > t1 or never: known only to exceed t1). Returns (theta, n_at_risk, n_events, n_censored, nll)."""
    T = np.asarray(T); risk = (T == 0) | (T > t0); ev = T[risk & (T > t0) & (T <= t1)]; ncen = int((risk & ((T == 0) | (T > t1))).sum())
    cnt = np.bincount(ev, minlength=t1 + 1)[t0 + 1:t1 + 1].astype(float); ts = np.arange(t0 + 1, t1 + 1)
    def nll(th):
        S0 = surv(family, th, t0); pm = (surv(family, th, ts - 1) - surv(family, th, ts)) / S0; Sc = surv(family, th, t1) / S0
        if not np.all(np.isfinite(pm)) or np.any(pm <= 0) or Sc <= 0: return 1e18
        return -(cnt @ np.log(pm) + ncen * np.log(Sc))
    x0 = {"lomax": [0.0, np.log(10.0)], "geometric": [np.log(0.05)], "weibull": [np.log(20.0), np.log(0.7)]}[family]
    best = min((minimize(nll, np.asarray(x0) + d, method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-9, maxiter=4000)) for d in (0.0, 1.0, -1.0)), key=lambda r: r.fun)
    return best.x, int(risk.sum()), int(len(ev)), ncen, float(best.fun)

def predict_unsolved(family, th, n_at_t0, t, t0=T0): return float(n_at_t0 * surv(family, th, t) / surv(family, th, t0))

def unsolved_at(T, t):
    """Puzzles whose waiting time exceeds t (never counts)."""
    T = np.asarray(T); return int(((T == 0) | (T > t)).sum())

def bb_ll(th, s, n, zi):
    a, b = np.exp(th[:2]); base = gammaln(n + 1) - gammaln(s + 1) - gammaln(n - s + 1) + betaln(a + s, b + n - s) - betaln(a, b)
    if not zi: return base
    pi = 1.0 / (1.0 + np.exp(-th[2])); return np.where(s == 0, np.logaddexp(np.log(pi), np.log1p(-pi) + base), np.log1p(-pi) + base)

def fit_bb(s, n, zi):
    x0 = [np.log(2.0), np.log(0.5)] + ([-3.0] if zi else [])
    r = min((minimize(lambda th: -bb_ll(th, s, n, zi).sum(), np.asarray(x0) + d, method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-9, maxiter=6000)) for d in (0.0, 0.7, -0.7)), key=lambda q: q.fun)
    return r.x

def bb_predict_uncovered(th, s, n, K, zi):
    """Expected number of puzzles with no success in all K draws, given s successes in the first n (only s == 0 puzzles can be uncovered)."""
    a, b = np.exp(th[:2]); n0 = int((s == 0).sum()); cont = np.exp(betaln(a, b + K) - betaln(a, b + n))          # P(next K-n all fail | s=0, not the zero class)
    if not zi: return n0 * cont
    pi = 1.0 / (1.0 + np.exp(-th[2])); pz = pi / (pi + (1 - pi) * np.exp(betaln(a, b + n) - betaln(a, b))); return n0 * (pz + (1 - pz) * cont)

def bb_heldout_ll(th, s, n, s2, m, zi):
    """log P(s2 successes in the NEXT m draws | s in the first n), summed over puzzles."""
    a, b = np.exp(th[:2]); post = gammaln(m + 1) - gammaln(s2 + 1) - gammaln(m - s2 + 1) + betaln(a + s + s2, b + n - s + m - s2) - betaln(a + s, b + n - s)
    if not zi: return float(post.sum())
    pi = 1.0 / (1.0 + np.exp(-th[2])); pz = np.where(s == 0, pi / (pi + (1 - pi) * np.exp(betaln(a, b + n) - betaln(a, b))), 0.0)
    with np.errstate(divide="ignore"): ll = np.where(s2 == 0, np.logaddexp(np.log(pz), np.log1p(-pz) + post), np.log1p(-pz) + post)
    return float(ll.sum())

def equal_compute(bits256, exk, resk, ks=(1, 2, 4)):
    """One pass of 64k iterations against the FIRST k restarts of 64 iterations: terminal exactness at step 64k; some restart right; the smallest-residual restart right."""
    out = {}
    for k in ks:
        sel = exk[np.arange(len(exk)), resk[:, :k].argmin(1)]; out[k] = dict(depth=float(bits256[:, 64 * k - 1].mean()), verified=float(exk[:, :k].any(1).mean()), selected=float(sel.mean()))
    return out

def band_of(rating):
    r = np.asarray(rating); b = np.full(len(r), -1)
    for i, (lo, hi) in enumerate(BANDS): b[(r >= lo) & (r <= hi)] = i
    return b

def factor(pred, obs): return float(max(pred, 1e-9) / obs) if obs > 0 else float("inf")
def within(pred, obs, f): return (abs(np.log(max(pred, 1e-9) / obs)) <= np.log(f)) if obs > 1 else (abs(pred - obs) <= 2)

# ---------------- the run ----------------
def load(d):
    z = np.load(R / d / "records_all.npz"); s = json.load(open(R / d / "summary_all.json")); o = np.argsort(z["idx"]); return {k: z[k][o] for k in z.files}, s

def run():
    OUT.mkdir(parents=True, exist_ok=True); Ls, J = [], {}; say = lambda s_="": (Ls.append(s_), print(s_, flush=True))
    say("SOLVING AS FIRST PASSAGE (tools/lens_first_passage.py; registration Note_2026-09-19_First_Passage_Tests.md). Banked records; the fixed start; all 81 cells.")
    ids5k = None; W = {}
    say(); say(f"== T1: depth scaling from the waiting-time tail (fit on iterations {T0 + 1}-{T1FIT}, predict 128 and 256; the same 5,000 puzzles)")
    for k, (d256, scan, full, label) in MODELS.items():
        z, s = load(d256); assert s["t_total"] == 256 and len(z["idx"]) == 5000; ids5k = z["idx"] if ids5k is None else ids5k; assert (z["idx"] == ids5k).all(), k
        bits = unpack(z["exact_by_step"], 256); assert (bits[:, -1] == z["cold_exact"]).all(), f"{k}: bit order"; T = waiting_time(bits); W[k] = dict(T=T, bits=bits, rating=z["rating"])
        fe = z["first_exact"]; conv = "0-based" if ((fe[T > 0] + 1) == T[T > 0]).all() else ("1-based" if (fe[T > 0] == T[T > 0]).all() else "DIFFERS"); assert conv != "DIFFERS", f"{k}: the record's first_exact disagrees with the bits"
        leave = int((bits[:, :-1] & ~bits[:, 1:]).any(1).sum()); J[k] = dict(label=label, n=5000, first_exact_field=conv, leave_256=leave, unsolved={t: unsolved_at(T, t) for t in (8, 16, 64, 128, 256)}, fits={})
        say(); say(f"  {k} ({label}): unsolved at 8 / 16 / 64 / 128 / 256: " + " / ".join(str(J[k]['unsolved'][t]) for t in (8, 16, 64, 128, 256)) + f" of 5,000 | solved-then-unsolved transitions to 256: {leave} | terminal unsolved at 256: {int((~bits[:, -1]).sum())}")
        for fam in ("lomax", "geometric", "weibull"):
            th, nrisk, nev, ncen, nll = fit_tail(T, fam); p128, p256 = (predict_unsolved(fam, th, J[k]["unsolved"][8], t) for t in (128, 256)); o128, o256 = J[k]["unsolved"][128], J[k]["unsolved"][256]
            J[k]["fits"][fam] = dict(theta=np.exp(th).tolist(), nll=nll, at_risk=nrisk, events=nev, censored=ncen, pred128=p128, pred256=p256, f128=factor(p128, o128), f256=factor(p256, o256))
            say(f"    {fam:9s} params {np.round(np.exp(th), 4).tolist()} | at risk {nrisk}, events {nev}, censored {ncen} | predicted unsolved at 128: {p128:7.1f} (observed {o128}; x{factor(p128, o128):.2f}) | at 256: {p256:7.1f} (observed {o256}; x{factor(p256, o256):.2f})")
    a = sum(within(J[k]["fits"]["lomax"]["pred256"], J[k]["unsolved"][256], 1.5) for k in MODELS); b = sum(within(J[k]["fits"]["lomax"]["pred128"], J[k]["unsolved"][128], 1.3) for k in MODELS)
    c = sum(abs(np.log(J[k]["fits"]["lomax"]["f256"])) < abs(np.log(J[k]["fits"]["geometric"]["f256"])) for k in MODELS if J[k]["unsolved"][256] > 0)
    say(); say(f"  T1 PASS-a (Lomax within x1.5 at 256 on >= 7 of 9): {'PASS' if a >= 7 else 'FAIL'} ({a} of 9) | PASS-b (within x1.3 at 128 on >= 7 of 9): {'PASS' if b >= 7 else 'FAIL'} ({b} of 9) | PASS-c (Lomax closer than geometric at 256 on all 9): {'PASS' if c == 9 else 'FAIL'} ({c} of 9)")
    say("  by rating band (pooled over the three width-192 seeds; Lomax fitted per band on 9-64; descriptive):")
    for i, (lo, hi) in enumerate(BANDS):
        Tb = np.concatenate([W[k]["T"][band_of(W[k]["rating"]) == i] for k in ("C5", "C7", "C8")])
        if unsolved_at(Tb, 8) >= 30 and ((Tb > T0) & (Tb <= T1FIT)).sum() >= 10:
            th, *_ = fit_tail(Tb, "lomax"); say(f"    rating {lo}-{hi if hi < 10**8 else 'up'}: n {len(Tb)}, unsolved at 8 / 64 / 256: {unsolved_at(Tb, 8)} / {unsolved_at(Tb, 64)} / {unsolved_at(Tb, 256)}; predicted at 256: {predict_unsolved('lomax', th, unsolved_at(Tb, 8), 256):.1f}")
        else: say(f"    rating {lo}-{hi if hi < 10**8 else 'up'}: n {len(Tb)}, unsolved at 8: {unsolved_at(Tb, 8)} (too few for a fit)")

    say(); say("== T2: restart scaling from the first 16 restarts (predict the puzzles with NO successful start among 128; 64 iterations; the same 5,000)"); SC = {}
    for k, (d256, scan, full, label) in MODELS.items():
        if scan is None: continue
        z, s = load(scan); assert s["k_init"] == 128 and s["t_total"] == 64 and (z["idx"] == ids5k).all(), k; ex = z["mi_exact_k"].astype(bool); SC[k] = (ex, z["mi_resid_k"])
        s16, s112 = ex[:, :16].sum(1), ex[:, 16:].sum(1); obs = int((~ex.any(1)).sum()); res = {}
        for zi in (False, True):
            th = fit_bb(s16, 16, zi); res[zi] = dict(theta=(np.exp(th[:2]).tolist() + ([float(1 / (1 + np.exp(-th[2])))] if zi else [])), pred=float(bb_predict_uncovered(th, s16, 16, 128, zi)), held=bb_heldout_ll(th, s16, 16, s112, 112, zi),
                                                     pred32=float(bb_predict_uncovered(th, s16, 16, 32, zi)), pred64=float(bb_predict_uncovered(th, s16, 16, 64, zi)))
        best = max(res, key=lambda q: res[q]["held"]); J[k]["restarts"] = dict(observed_uncovered=obs, none_in_16=int((s16 == 0).sum()), obs32=int((~ex[:, :32].any(1)).sum()), obs64=int((~ex[:, :64].any(1)).sum()), plain=res[False], zero_inflated=res[True], better="zero_inflated" if best else "plain")
        say(f"  {k}: no success in 16 / 32 / 64 / 128 restarts: {int((s16 == 0).sum())} / {J[k]['restarts']['obs32']} / {J[k]['restarts']['obs64']} / {obs} | beta-binomial predicts {res[False]['pred']:.1f} at 128 (held-out log-lik {res[False]['held']:.1f}) | zero-inflated predicts {res[True]['pred']:.1f} (zero mass {res[True]['theta'][2]:.4f}; held-out {res[True]['held']:.1f}) | better: {J[k]['restarts']['better']}")
    a = sum(within(J[k]["restarts"][J[k]["restarts"]["better"]]["pred"], J[k]["restarts"]["observed_uncovered"], 2.0) for k in SC); b = sum(J[k]["restarts"]["better"] == "zero_inflated" for k in SC)
    say(f"  T2 PASS-a (the better model within x2, or 2 puzzles, on >= 5 of 7): {'PASS' if a >= 5 else 'FAIL'} ({a} of {len(SC)}) | PASS-b (zero-inflated better on >= 5 of 7): {'PASS' if b >= 5 else 'FAIL'} ({b} of {len(SC)})")

    say(); say("== T3: depth against restarts at equal compute (one pass of 64k iterations against the first k restarts of 64; the same 5,000)")
    for k, (ex, rs) in SC.items():
        e = equal_compute(W[k]["bits"], ex, rs); J[k]["equal_compute"] = e
        say(f"  {k}: " + " | ".join(f"k={kk}: depth {100 * v['depth']:.2f}, some restart right {100 * v['verified']:.2f}, smallest residual {100 * v['selected']:.2f}" for kk, v in e.items()))
    a = sum(J[k]["equal_compute"][4]["verified"] > J[k]["equal_compute"][4]["depth"] for k in SC); hs = [k for k in ("C5", "C7", "C8", "EQR") if k in SC]; b = sum(J[k]["equal_compute"][4]["selected"] >= J[k]["equal_compute"][4]["depth"] for k in hs)
    say(f"  T3 PASS-a (verified restarts beat depth at k = 4 on all 7): {'PASS' if a == len(SC) else 'FAIL'} ({a} of {len(SC)}) | PASS-b (residual-selected >= depth at k = 4 on the three seeds and EqR): {'PASS' if b == len(hs) else 'FAIL'} ({b} of {len(hs)})")

    say(); say("== T4: hardness as an escape rate (all 422,786 puzzles; 64 iterations)"); FT = {}
    for k, (d256, scan, full, label) in MODELS.items():
        if full is None: continue
        z, s = load(full); assert s["t_total"] == 64 and len(z["idx"]) == 422786, k; bits = unpack(z["exact_by_step"], 64); assert (bits[:, -1] == z["cold_exact"]).all(), f"{k}: bit order"
        T = waiting_time(bits); Tc = np.where(T == 0, 65, T); FT[k] = Tc; rho = float(spearmanr(z["rating"], Tc).statistic); bd = band_of(z["rating"]); kap = []
        for i in range(len(BANDS)):
            Tb = T[bd == i]; th, nrisk, nev, ncen, _ = fit_tail(Tb, "geometric") if (((Tb > T0) & (Tb <= T1FIT)).sum() >= 20) else (None, 0, 0, 0, 0); kap.append(None if th is None else float(np.exp(th[0])))
        J[k]["hardness"] = dict(n=len(T), rho_rating=rho, kappa_by_band=kap, eta_by_band=[None if q is None else float(-np.log10(q)) for q in kap], leave_64=int((bits[:, :-1] & ~bits[:, 1:]).any(1).sum()), unsolved_by_band=[int(((T == 0))[bd == i].sum()) for i in range(len(BANDS))], n_by_band=[int((bd == i).sum()) for i in range(len(BANDS))])
        say(f"  {k} ({label}): rank correlation rating ~ waiting time {rho:+.3f} | tail escape rate per band (rating 0 | 1-9 | 10-29 | 30-59 | 60 up): " + " | ".join("  -  " if q is None else f"{q:.4f}" for q in kap) + f" | never solved per band: {J[k]['hardness']['unsolved_by_band']} of {J[k]['hardness']['n_by_band']}")
    a = sum(J[k]["hardness"]["rho_rating"] >= 0.30 for k in FT); mono = lambda q: all(x is not None for x in q) and all(q[i] > q[i + 1] for i in range(len(q) - 1)); b = sum(mono(J[k]["hardness"]["kappa_by_band"]) for k in FT)
    r_seed = float(spearmanr(FT["C7"], FT["C8"]).statistic); r_eqr = float(spearmanr(FT["C7"], FT["EQR"]).statistic); J["_agreement"] = dict(C7_C8=r_seed, C7_EQR=r_eqr)
    say(f"  T4 PASS-a (rank correlation >= 0.30 on all 7): {'PASS' if a == len(FT) else 'FAIL'} ({a} of {len(FT)}) | PASS-b (escape rate falls band to band on all 7): {'PASS' if b == len(FT) else 'FAIL'} ({b} of {len(FT)}) | PASS-c (waiting times agree: two seeds {r_seed:+.3f} >= 0.50; ours against EqR {r_eqr:+.3f} >= 0.30): {'PASS' if (r_seed >= 0.5 and r_eqr >= 0.3) else 'FAIL'}")
    (OUT / "report.txt").write_text("\n".join(Ls) + "\n"); (OUT / "report.json").write_text(json.dumps(J, indent=1, default=float))

# ---------------- selftest ----------------
def selftest():
    n = 0; rng = np.random.default_rng(0)
    b = np.zeros((2, 16), bool); b[0, 9] = True; b[1, 0] = True; packed = np.packbits(b.astype(np.uint8), axis=1, bitorder="little")
    assert (unpack(packed, 16) == b).all() and waiting_time(unpack(packed, 16)).tolist() == [10, 1] and waiting_time(np.zeros((1, 8), bool)).tolist() == [0]; n += 1      # little-endian; 1-based; never = 0
    assert not (np.unpackbits(packed, axis=1, bitorder="big")[:, :16].astype(bool) == b).all(); n += 1                                                                # the big-endian read is a different array
    # a Lomax population with known parameters: the truncated, censored fit must recover the tail and beat one escape rate
    a0, tau0, N = 0.9, 6.0, 60000; U = rng.random(N); Tc = tau0 * (U ** (-1 / a0) - 1); T = np.ceil(Tc).astype(int); T = np.where(T > 256, 0, np.maximum(T, 1))
    thL, nrisk, nev, ncen, _ = fit_tail(T, "lomax"); thG, *_ = fit_tail(T, "geometric"); o256 = unsolved_at(T, 256); pL = predict_unsolved("lomax", thL, unsolved_at(T, 8), 256); pG = predict_unsolved("geometric", thG, unsolved_at(T, 8), 256)
    assert nrisk == unsolved_at(T, 8) and nev + ncen == nrisk and abs(np.log(pL / o256)) < np.log(1.1) and abs(np.log(pG / o256)) > np.log(3); n += 1
    Tbad = np.where((T == 0) | (T > 64), 64, T); thB, *_ = fit_tail(Tbad, "lomax"); assert abs(np.log(predict_unsolved("lomax", thB, unsolved_at(T, 8), 256) / o256)) > np.log(1.5); n += 1   # treating the censored as EVENTS at 64 must NOT recover the tail
    # a geometric population: the geometric fit recovers kappa
    q = 0.07; Tg = rng.geometric(1 - np.exp(-q), 40000); thg, *_ = fit_tail(Tg, "geometric"); assert abs(np.exp(thg[0]) - q) < 0.005; n += 1
    # beta-binomial and its zero-inflated form
    Np = 20000; p = rng.beta(0.6, 0.15, Np); p[:400] = 0.0; X = rng.random((Np, 128)) < p[:, None]; s16, s112 = X[:, :16].sum(1), X[:, 16:].sum(1); obs = int((~X.any(1)).sum())
    th0, th1 = fit_bb(s16, 16, False), fit_bb(s16, 16, True); p0, p1 = bb_predict_uncovered(th0, s16, 16, 128, False), bb_predict_uncovered(th1, s16, 16, 128, True)
    assert abs(np.log(p1 / obs)) < np.log(1.25) and bb_heldout_ll(th1, s16, 16, s112, 112, True) > bb_heldout_ll(th0, s16, 16, s112, 112, False) and abs(1 / (1 + np.exp(-th1[2])) - 0.02) < 0.01; n += 1
    assert abs(bb_predict_uncovered(np.log([1.0, 1.0]), np.array([0, 3]), 16, 16, False) - 1.0) < 1e-12; n += 1                                                       # K = n: every s = 0 puzzle is uncovered; s > 0 never
    bits = np.zeros((3, 256), bool); bits[0, 60:] = True; bits[1, 200:] = True; exk = np.array([[0, 1, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]], bool); resk = np.array([[.1, .5, .9, .9], [.2, .1, .3, .4], [.9, .1, .5, .5]])
    e = equal_compute(bits, exk, resk); assert e[1] == dict(depth=1 / 3, verified=1 / 3, selected=1 / 3) and abs(e[4]["depth"] - 2 / 3) < 1e-12 and abs(e[2]["verified"] - 2 / 3) < 1e-12 and e[2]["selected"] == 0.0; n += 1
    assert band_of([0, 5, 29, 30, 500]).tolist() == [0, 1, 2, 3, 4] and within(10, 14, 1.5) and not within(10, 16, 1.5) and within(2.5, 1, 2.0) and not within(4, 1, 2.0); n += 1
    assert all((R / d / "records_all.npz").exists() for m in MODELS.values() for d in m[:3] if d); n += 1
    print(f"selftest OK: {n}/{n}")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true"); ap.add_argument("--run", action="store_true"); a = ap.parse_args()
    if a.selftest: return selftest()
    if a.run: run()

if __name__ == "__main__":
    main()
