#!/usr/bin/env python3
# Ledger: THE PAPER'S FINAL RUNS — the verification pass (the PI, 2026-09-14: "make sure these results look good and are made with no
# errors before updating the records and the paper"). Independent code (never imports the frozen analyzer), over the staging root:
#   V1  record integrity per row: n, unique idx, recomputed accuracy == summary, t, EMA (raw-weights rows EMA off), selected ckpt,
#       the full-set rows == idx 0..422,785, shard-log errors; V1b identical puzzle sets wherever rows are paired
#   V2  every letter-bearing number recomputed (means, spreads, paired contrasts with a scipy binomial test); V2b restart selection
#       bounded under worst-/best-case tie-breaking of equal minimal residuals
#   V3  selection + extension replayed from metrics.jsonl (EMA key, raw second key, earliest tie, banked 2k grids, the 4k window)
#   V4  training health (non-finite values, step continuity)
#   V5  TPU determinism: the D16 row vs step 16 of the D64 row (same checkpoint, separate pod runs), per puzzle
#   V6  checkpoint identity (with --xcheck): a CPU re-evaluation of each banked grid vs the pods' D16 records, per puzzle; exact bits
#       carry the CPU/TPU round-off floor (Note_2026-09-10_Champion_Mechanisms §3: 4.3-4.7 %), so identity is read on the first-exact
#       step of the puzzles the pod solves at step 1-2, against the other seeds' models as the baseline
#   CPU re-evaluation (the Mac, ~3 min per w192 grid):  JAX_PLATFORMS=cpu PYTHONPATH=src .venv/bin/python tools/eval_sudoku_extreme.py \
#       --ckpt <grid> --npz data/sudoku_extreme/sudoku_extreme_seed0_mon512.npz --out <xcheck>/<arm> --split test --limit 384 --t-total 16 --ema --batch 128
#   .venv/bin/python tools/verify_paperfinal.py --root runs/_paperfinal_pull/stage [--xcheck DIR]   -> <root>/analysis/verify.txt
#   .venv/bin/python tools/verify_paperfinal.py --selftest
from __future__ import annotations
import argparse, glob, json, math, re
from pathlib import Path
import numpy as np
from scipy.stats import binomtest

N_FULL = 422786
VSEL = {"C0": 16000, "C1": 28000, "C2": 24000, "C3": 18000, "C4": 20000, "C5": 46000, "C6": 28000, "C7": 46000, "C8": 22000}
EXPECT_T = {"full_vsel_t16": 16, "full_final_t16": 16, "full_vsel_t16_alt": 16, "full_vsel_t64": 64, "sub20k_t128": 128, "sub5k_t256": 256,
            "d64full": 64, "d128sub": 128, "d256sub": 256, "k32": 64, "k128": 64}

def pair_bits(ia, ea, ib, eb):
    """Exact pairing by idx (own implementation): (diff pp, only_a, only_b, n, two-sided binomial p)."""
    oa, ob = np.argsort(ia), np.argsort(ib); ia, ea, ib, eb = ia[oa], ea[oa], ib[ob], eb[ob]
    c = np.intersect1d(ia, ib); ea = ea[np.searchsorted(ia, c)]; eb = eb[np.searchsorted(ib, c)]
    b, d = int((ea & ~eb).sum()), int((~ea & eb).sum())
    return 100 * (ea.mean() - eb.mean()), b, d, len(c), (1.0 if b + d == 0 else binomtest(min(b, d), b + d, 0.5).pvalue)

def t1r_bounds(ex, rs):
    rs = np.where(np.isfinite(rs), rs, np.inf); tied = rs == rs.min(axis=1, keepdims=True)
    return ex[np.arange(len(ex)), rs.argmin(axis=1)], np.all(~tied | ex, axis=1), np.any(tied & ex, axis=1)

def verify(root, xcheck=None):
    ST, L, FAIL = Path(root), [], []
    say = lambda s="": (L.append(s), print(s))
    def bad(m): FAIL.append(m); say("  !! " + m)
    S = lambda d: json.loads((ST / d / "summary_all.json").read_text()) if (ST / d / "summary_all.json").exists() else None
    Z = lambda d: np.load(ST / d / "records_all.npz", allow_pickle=True) if (ST / d / "records_all.npz").exists() else None
    rows = {}
    for a in ["C0", "C1", "C2", "C4", "C5", "C7", "C8"]:
        for nm, n in [("full_vsel_t16", N_FULL), ("full_final_t16", 50000), ("full_vsel_t16_alt", 50000), ("full_vsel_t64", 100000), ("sub20k_t128", 20000), ("sub5k_t256", 5000)]:
            rows[f"{a}/{nm}"] = (f"sxeval_pchamp{a}/{nm}", n)
        rows[f"{a}/d64full"] = (f"filler_sxeval_pchamp{a}_full_t64", N_FULL)
    for a in ["C5", "C7", "C8"]:
        rows[f"{a}/d128sub"] = (f"filler_sxeval_pchamp{a}_sub50000_t128", 50000); rows[f"{a}/d256sub"] = (f"filler_sxeval_pchamp{a}_sub50000_t256", 50000)
        rows[f"{a}/k32"] = (f"sxscan_pchamp{a}", 5000)
    for a in ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]: rows[f"{a}/k128"] = (f"filler_sxscan128_pchamp{a}", 5000)
    rows["EqR/k128"] = ("filler_sxscan128_pport_eqr", 5000)
    say("V1 record integrity"); ids, acc, R = {}, {}, {}
    for key, (d, n) in rows.items():
        s, z = S(d), Z(d); arm, kind = key.split("/")
        if s is None or z is None: bad(f"{key}: missing"); continue
        idx = np.asarray(z["idx"]); ce = np.asarray(z["cold_exact"]).astype(bool)
        if s.get("n") != n or len(idx) != n or len(np.unique(idx)) != n: bad(f"{key}: n {s.get('n')}/{len(idx)}/{len(np.unique(idx))} != {n}")
        if abs(float(ce.mean()) - s["exact_acc"]) > 1e-9: bad(f"{key}: summary accuracy != records")
        if s.get("t_total") != EXPECT_T[kind]: bad(f"{key}: t_total {s.get('t_total')}")
        if arm != "EqR":
            if s.get("ema") is not (kind != "full_vsel_t16_alt"): bad(f"{key}: ema {s.get('ema')}")
            want = "ckpt_latest.pkl" if kind == "full_final_t16" else f"ckpt_{VSEL[arm]:06d}.pkl"
            if Path(str(s.get("ckpt"))).name != want: bad(f"{key}: ckpt {s.get('ckpt')} != {want}")
        elif "eqr" not in str(s.get("ckpt", "")).lower(): bad(f"{key}: ckpt {s.get('ckpt')}")
        if kind in ("full_vsel_t16", "d64full") and not np.array_equal(np.sort(idx), np.arange(N_FULL)): bad(f"{key}: full-set idx")
        if [f for f in glob.glob(str(ST / d / "shard_*.log")) if re.search(r"Traceback|FloatingPointError|Error:", Path(f).read_text(errors="ignore"))]: bad(f"{key}: shard-log errors")
        ids[key] = frozenset(np.unique(idx).tolist()); acc[key] = float(ce.mean()); R[key] = z
    say(f"  {len(acc)} rows checked")
    for label, keys in [("the 50k", [f"{a}/{k}" for a in ["C0", "C1", "C2", "C4", "C5", "C7", "C8"] for k in ("full_final_t16", "full_vsel_t16_alt")] + [f"{a}/{k}" for a in ["C5", "C7", "C8"] for k in ("d128sub", "d256sub")]),
                        ("the 100k", [f"{a}/full_vsel_t64" for a in ["C0", "C1", "C2", "C4", "C5", "C7", "C8"]]),
                        ("the 20k", [f"{a}/sub20k_t128" for a in ["C0", "C1", "C2", "C4", "C5", "C7", "C8"]]),
                        ("the 5k", [f"{a}/sub5k_t256" for a in ["C0", "C1", "C2", "C4", "C5", "C7", "C8"]] + [f"{a}/k32" for a in ["C5", "C7", "C8"]] + [k for k in rows if k.endswith("/k128")])]:
        ok = all(k in ids for k in keys) and len({ids[k] for k in keys}) == 1
        say(f"  V1b {label}: {len(keys)} rows on one puzzle set = {ok}") if ok else bad(f"V1b {label}: puzzle sets differ")
    say("V2 recomputation")
    for row in ("full_vsel_t16", "d64full", "d128sub", "d256sub"):
        v = [acc[f"{a}/{row}"] for a in ("C5", "C7", "C8")]
        say(f"  w192 {row}: {' / '.join(f'{100*x:.3f}' for x in v)}  mean {100*np.mean(v):.3f} +/- {50*(max(v)-min(v)):.3f}")
    for row in ("full_vsel_t16", "d64full"):
        v = [acc[f"{a}/{row}"] for a in ("C0", "C1", "C2")]; say(f"  w384 {row}: mean {100*np.mean(v):.3f} +/- {50*(max(v)-min(v)):.3f}")
        ds = []
        for x, y in (("C5", "C0"), ("C7", "C1"), ("C8", "C2")):
            d, b, c, n, p = pair_bits(R[f"{x}/{row}"]["idx"], R[f"{x}/{row}"]["cold_exact"].astype(bool), R[f"{y}/{row}"]["idx"], R[f"{y}/{row}"]["cold_exact"].astype(bool)); ds.append(d)
            say(f"  {row} {x}-{y} {d:+.3f} ({b}/{c}, n {n}, p {p:.1e})")
        say(f"  {row} mean {np.mean(ds):+.3f}")
    ze = R["EqR/k128"]; fe_, we_, be_ = t1r_bounds(ze["mi_exact_k"].astype(bool), ze["mi_resid_k"].astype(np.float64))
    say(f"  EqR k128: t1r {100*fe_.mean():.2f} (tie bounds {100*we_.mean():.2f}..{100*be_.mean():.2f})")
    for a in ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]:
        z = R[f"{a}/k128"]; f, w, b = t1r_bounds(z["mi_exact_k"].astype(bool), z["mi_resid_k"].astype(np.float64))
        d, b1, c1, n, p = pair_bits(z["idx"], w, ze["idx"], be_)
        say(f"  {a}-EqR t1r {100*(f.mean()-fe_.mean()):+.2f}; adversarial ties (arm worst, EqR best) {d:+.2f} ({b1}/{c1}, p {p:.1e})")
    say("V3 selection + extension replay")
    for a in VSEL:
        mon = {}
        for l in (ST / f"pretrainchamp_{a}" / "metrics.jsonl").read_text().splitlines():
            if l.strip() and "monitor" in (r := json.loads(l)): mon[r["monitor"]["step"]] = r["monitor"]
        banked = {int(re.search(r"ckpt_(\d+)", p.name).group(1)) for p in (ST / f"pretrainchamp_{a}").glob("ckpt_[0-9]*.pkl")}
        def sel(upto):
            c = [((m["val_t16_ema"], m.get("val_t16") or 0.0), s) for s, m in mon.items() if s in banked and s <= upto]
            best = max(k for k, _ in c); return min(s for k, s in c if k == best)
        b0 = 50000 if a in ("C3", "C6") else 30000; s0 = sel(b0); ext = s0 >= b0 - 4000; s1 = sel(max(mon))
        ok = s1 == VSEL[a] and ext == (ST / f"pretrainchamp_{a}" / "EXTENDED.txt").exists()
        say(f"  {a}: by the budget {s0//1000}k -> extend {ext}; final {s1//1000}k = registered {ok}") if ok else bad(f"V3 {a}: replay {s0}/{ext}/{s1}")
    say("V4 training health")
    for a in ("C5", "C7", "C8"):
        tr = [json.loads(l) for l in (ST / f"pretrainchamp_{a}" / "metrics.jsonl").read_text().splitlines() if l.strip()]
        tr = [r for r in tr if "loss" in r]; nf = sum(1 for r in tr for k in ("loss", "ce_in") if not math.isfinite(r.get(k, 0.0)))
        say(f"  {a}: {len(tr)} rows, non-finite {nf}") if nf == 0 else bad(f"V4 {a}: non-finite values")
    say("V5 TPU determinism: the D16 row vs step 16 of the D64 row (same checkpoint, separate pod runs)")
    for a in ("C0", "C1", "C2", "C5", "C7", "C8"):
        d16, d64 = R[f"{a}/full_vsel_t16"], R[f"{a}/d64full"]; o16, o64 = np.argsort(d16["idx"]), np.argsort(d64["idx"])
        f64 = np.asarray(d64["first_exact"])[o64]; e64 = (f64 >= 0) & (f64 < 16)   # first_exact is 0-based
        nd = int(np.sum(np.asarray(d16["cold_exact"]).astype(bool)[o16] != e64))
        say(f"  {a}: {nd} of {N_FULL} puzzles differ") if nd <= 50 else bad(f"V5 {a}: {nd} puzzles differ between two TPU runs")
    if xcheck:
        say("V6 checkpoint identity: CPU re-evaluation vs the pods' D16 records (first-exact step on the pod's step-1/2 puzzles; exact bits carry the route floor)")
        X = Path(xcheck); pod = {a: R[f"{a}/full_vsel_t16"] for a in ("C5", "C7", "C8")}
        for m in ("C5", "C7", "C8"):
            q = X / m / "records_all.npz"
            if not q.exists(): bad(f"V6 {m}: no CPU records"); continue
            zm = np.load(q); pos = {int(i): k for k, i in enumerate(np.asarray(pod[m]["idx"]))}
            parts, own = [], None
            for p_ in ("C5", "C7", "C8"):
                sel_ = np.array([{int(i): k for k, i in enumerate(np.asarray(pod[p_]["idx"]))}[int(i)] for i in zm["idx"]])
                pf = np.asarray(pod[p_]["first_exact"])[sel_]; easy = (pf >= 0) & (pf <= 1)
                agree = float(np.mean(np.asarray(zm["first_exact"])[easy] == pf[easy]))
                bits = float(np.mean(np.asarray(zm["cold_exact"]).astype(bool) == np.asarray(pod[p_]["cold_exact"]).astype(bool)[sel_]))
                parts.append(f"vs {p_} {100*agree:.1f} % (exact bits {100*bits:.1f} %)")
                if p_ == m: own = agree
                elif agree >= own if own is not None else False: pass
            others = [float(x.split()[2]) for x in parts if not x.startswith(f"vs {m} ")]
            ok = own is not None and own >= 0.90 and all(o < 100 * own - 30 for o in others)
            say(f"  CPU {m} ({len(zm['idx'])} puzzles): " + " | ".join(parts)) if ok else bad(f"V6 {m}: identity not separated: " + " | ".join(parts))
    say("\nVERIFY: " + ("ALL CHECKS PASS" if not FAIL else f"{len(FAIL)} FAILURES: " + " | ".join(FAIL)))
    out = ST / "analysis"; out.mkdir(parents=True, exist_ok=True); (out / "verify.txt").write_text("\n".join(L) + "\n")
    return FAIL

def selftest():
    d, b, c, n, p = pair_bits(np.array([3, 1, 2, 9]), np.array([1, 1, 0, 1], bool), np.array([2, 3, 1]), np.array([1, 0, 1], bool))
    assert (b, c, n) == (1, 1, 3) and abs(d) < 1e-9 and abs(p - 1.0) < 1e-9
    ex = np.array([[1, 0, 0], [0, 1, 0]], bool); rs = np.array([[0.0, 0.0, 1.0], [0.5, 0.5, np.nan]])
    f, w, b_ = t1r_bounds(ex, rs)
    assert f.tolist() == [True, False] and w.tolist() == [False, False] and b_.tolist() == [True, True]
    print("selftest OK: 2/2")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--root"); ap.add_argument("--xcheck"); ap.add_argument("--selftest", action="store_true"); a = ap.parse_args()
    selftest() if a.selftest else verify(a.root, a.xcheck)
