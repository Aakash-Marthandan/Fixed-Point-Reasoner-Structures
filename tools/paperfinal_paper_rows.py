#!/usr/bin/env python3
# Ledger: THE PAPER'S ROWS after the paper-final runs and the C8 extension (2026-09-14). The PI's reporting decision (B), recorded
# before any extension data (Plan_2026-09-14_C8_Extension.md §8, commit 63a3917): the width-192 row is the BUDGET-MATCHED triple (every
# seed to 50k, the registered monitor pick: C5 46k, C7 46k, C8 46k); the registered 30k-budget triple (C8 at 22k) goes to the
# appendix, labeled. This script turns the crc-verified pulls into ONE analysis artifact the paper's number pipeline reads
# (tools/paper_numbers.py); every row is n-gated and checkpoint-checked here, the paired contrasts use the frozen analyzer's `paired`.
#   .venv/bin/python tools/paperfinal_paper_rows.py [--pf runs/_paperfinal_pull/stage --x runs/_c8x_pull/x/runs --out runs/analysis/paperfinal_rows_20260914.json]
#   .venv/bin/python tools/paperfinal_paper_rows.py --selftest
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_paperfinal import paired   # frozen at 7a99d9e, selftested

N_FULL, N_50K, N_100K, N_20K, N_5K = 422786, 50000, 100000, 20000, 5000

def summ(d): q = Path(d) / "summary_all.json"; return json.loads(q.read_text()) if q.exists() else None
def recs(d): q = Path(d) / "records_all.npz"; return dict(np.load(q, allow_pickle=True)) if q.exists() else None
def step_of(ck): m = re.search(r"ckpt_(\d+)", str(ck)); return int(m.group(1)) if m else None

def row(root, rel, n, step=None, ema=True):
    d = Path(root) / rel; s = summ(d)
    assert s is not None, f"missing {d}"
    assert s["n"] == n, f"{d}: n {s['n']} != {n}"
    assert bool(s.get("ema")) == ema, f"{d}: ema {s.get('ema')}"
    if step is not None: assert step_of(s["ckpt"]) == step, f"{d}: ckpt {s['ckpt']} != step {step}"
    return s

def scan32(root, arm, step):
    d = Path(root) / f"sxscan_pchamp{arm}"; s = row(root, f"sxscan_pchamp{arm}", N_5K, step)
    assert s["k_init"] == 32 and s["t_total"] == 64
    z = recs(d); ex = np.asarray(z["mi_exact_k"]).astype(bool); rs = np.asarray(z["mi_resid_k"]).astype(np.float64)
    rs = np.where(np.isfinite(rs), rs, np.inf); fin = np.isfinite(rs)
    t1r = float(ex[np.arange(len(ex)), rs.argmin(axis=1)].mean()); ver = float(ex.any(axis=1).mean())
    spur = float(np.mean(rs[(~ex) & fin] <= np.median(rs[ex & fin])))
    return dict(cold=s["exact_acc"], t1r32=t1r, v32=ver, spur32=spur)

def arm_rows(root, arm, step):
    """The registered battery + filler rows of one model at its selected grid (EMA single pass unless noted)."""
    d16 = row(root, f"sxeval_pchamp{arm}/full_vsel_t16", N_FULL, step)
    out = dict(step=step, d16=d16["exact_acc"], halt_auc=d16.get("q_halt_auc_last"),
               final16=row(root, f"sxeval_pchamp{arm}/full_final_t16", N_50K)["exact_acc"],
               raw16=row(root, f"sxeval_pchamp{arm}/full_vsel_t16_alt", N_50K, step, ema=False)["exact_acc"],
               d64_100k=row(root, f"sxeval_pchamp{arm}/full_vsel_t64", N_100K, step)["exact_acc"],
               d128_20k=row(root, f"sxeval_pchamp{arm}/sub20k_t128", N_20K, step)["exact_acc"],
               d256_5k=row(root, f"sxeval_pchamp{arm}/sub5k_t256", N_5K, step)["exact_acc"],
               d64_full=row(root, f"filler_sxeval_pchamp{arm}_full_t64", N_FULL, step)["exact_acc"],
               d128_50k=row(root, f"filler_sxeval_pchamp{arm}_sub50000_t128", N_50K, step)["exact_acc"],
               d256_50k=row(root, f"filler_sxeval_pchamp{arm}_sub50000_t256", N_50K, step)["exact_acc"])
    out.update(scan32(root, arm, step))
    return out

NS = {"d16": N_FULL, "d64_full": N_FULL, "d128_50k": N_50K, "d256_50k": N_50K}
def triple(rows, keys=("d16", "d64_full", "d128_50k", "d256_50k")):
    """Mean and half-spread; `counts` = the exact solved counts (accuracy x n), so the paper can round from exact fractions."""
    t = {}
    for k in keys:
        v = [r[k] for r in rows]; n = NS[k]; c = [int(round(x * n)) for x in v]
        assert all(abs(ci / n - x) < 1e-12 for ci, x in zip(c, v)), f"{k}: accuracy is not an exact count over n {n}"
        t[k] = dict(values=v, mean=float(np.mean(v)), half=float((max(v) - min(v)) / 2), counts=c, n=n)
    return t

def build(pf, x):
    PF, X = Path(pf), Path(x)
    A = {"C5": arm_rows(PF, "C5", 46000), "C7": arm_rows(PF, "C7", 46000), "C8@22k": arm_rows(PF, "C8", 22000), "C8@46k": arm_rows(X, "C8", 46000)}
    W384 = {a: row(PF, f"filler_sxeval_pchamp{a}_full_t64", N_FULL)["exact_acc"] for a in ("C0", "C1", "C2")}
    res = dict(arms=A, w384_d64_full=W384,
               triple_budget_matched=triple([A["C5"], A["C7"], A["C8@46k"]]),
               triple_registered_30k=triple([A["C5"], A["C7"], A["C8@22k"]]))
    width = {}
    for depth, rel in (("d16", "sxeval_pchamp{a}/full_vsel_t16"), ("d64_full", "filler_sxeval_pchamp{a}_full_t64"), ("d64_100k", "sxeval_pchamp{a}/full_vsel_t64")):
        per = {}
        for tag, xa, xroot, ya in (("C5-C0", "C5", PF, "C0"), ("C7-C1", "C7", PF, "C1"), ("C8@46k-C2", "C8", X, "C2"), ("C8@22k-C2", "C8", PF, "C2")):
            r = paired(recs(Path(xroot) / rel.format(a=xa)), recs(PF / rel.format(a=ya)))
            per[tag] = dict(diff=r["diff"], only_a=r["only_a"], only_b=r["only_b"], n=r["n"], p=r["p"])
        width[depth] = per
    res["width_paired"] = width
    c4 = row(PF, "filler_sxeval_pchampC4_full_t64", N_FULL, 20000); res["C4_d64_full"] = c4["exact_acc"]
    ze = recs(PF / "filler_sxscan128_pport_eqr"); se = row(PF, "filler_sxscan128_pport_eqr", N_5K, ema=True)
    ex = np.asarray(ze["mi_exact_k"]).astype(bool); rs = np.asarray(ze["mi_resid_k"]).astype(np.float64); rs = np.where(np.isfinite(rs), rs, np.inf); fin = np.isfinite(rs)
    res["eqr_k128_5k"] = dict(t1r=float(ex[np.arange(len(ex)), rs.argmin(axis=1)].mean()), verified=float(ex.any(axis=1).mean()),
                              spurious=float(np.mean(rs[(~ex) & fin] <= np.median(rs[ex & fin]))), cold=se["exact_acc"])
    for name, p in (("paperfinal_verdict", PF / "analysis" / "paperfinal_verdict.json"), ("c8x_verdict", X.parent / "analysis_prelim" / "c8x_verdict.json")):
        res[name] = json.loads(p.read_text()) if p.exists() else None
    return res

def selftest():
    v = triple([dict(d16=.95, d64_full=.99, d128_50k=.994, d256_50k=.996), dict(d16=.94, d64_full=.98, d128_50k=.992, d256_50k=.994), dict(d16=.96, d64_full=.985, d128_50k=.993, d256_50k=.995)])
    assert abs(v["d16"]["mean"] - .95) < 1e-12 and abs(v["d16"]["half"] - .01) < 1e-12 and abs(v["d64_full"]["half"] - .005) < 1e-12
    assert step_of("runs/pretrainchamp_C8/ckpt_046000.pkl") == 46000 and step_of("ckpt_latest.pkl") is None
    print("selftest OK: 2/2")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--pf", default="runs/_paperfinal_pull/stage"); ap.add_argument("--x", default="runs/_c8x_pull/x/runs")
    ap.add_argument("--out", default="runs/analysis/paperfinal_rows_20260914.json"); ap.add_argument("--selftest", action="store_true"); a = ap.parse_args()
    if a.selftest: selftest(); sys.exit(0)
    res = build(a.pf, a.x); Path(a.out).write_text(json.dumps(res, indent=1))
    bm, rg = res["triple_budget_matched"], res["triple_registered_30k"]
    for lab, t in (("budget-matched (B)", bm), ("registered 30k", rg)):
        print(lab + ": " + " · ".join(f"{k} {100*t[k]['mean']:.3f} +/- {100*t[k]['half']:.3f} ({', '.join(f'{100*x:.2f}' for x in t[k]['values'])})" for k in t))
    for dp, per in res["width_paired"].items():
        print(f"width w192-w384 {dp}: " + "; ".join(f"{k} {100*v['diff']:+.2f} (p {v['p']:.1e})" for k, v in per.items()))
    print(f"C4 D64 full {100*res['C4_d64_full']:.2f}; EqR k128 on the identical 5k: t1r {100*res['eqr_k128_5k']['t1r']:.2f} verified {100*res['eqr_k128_5k']['verified']:.2f}")
    for k, r in res["arms"].items():
        print(f"{k}: step {r['step']} d16 {100*r['d16']:.2f} final {100*r['final16']:.2f} raw {100*r['raw16']:.2f} d64(100k) {100*r['d64_100k']:.2f} d64 full {100*r['d64_full']:.2f} d128 20k/50k {100*r['d128_20k']:.2f}/{100*r['d128_50k']:.2f} d256 5k/50k {100*r['d256_5k']:.2f}/{100*r['d256_50k']:.2f} k32 {100*r['t1r32']:.2f}/{100*r['v32']:.2f} spur {100*r['spur32']:.2f} halt {r['halt_auc']}")
    print(f"-> {a.out}")
