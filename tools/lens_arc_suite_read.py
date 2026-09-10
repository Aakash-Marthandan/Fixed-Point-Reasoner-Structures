"""Paired reader for the pod's ARC suite (tools/arc_suite.py): p13Dri (RI-trained) vs p13C53 (plain twin) on identical (task, query) pairs."""
import json, numpy as np
from pathlib import Path
R = str(Path(__file__).resolve().parents[1] / "runs" / "filler_arcsuite") + "/"
def load(run):
    rows = [json.loads(l) for l in open(R + run + "/results.jsonl")]
    Q = {}
    for r in rows:
        for q in r["queries"]: Q[(r["task"], q["q"])] = dict(q, task=r["task"], sel_step=r["sel_step"])
    return Q
def mean(xs): xs = [x for x in xs if x is not None]; return float(np.mean(xs)) if xs else None
for st in ("valhard", "dev30"):
    A, B = load(f"p13Dri_{st}"), load(f"p13C53_{st}"); keys = sorted(set(A) & set(B)); n = len(keys)
    print(f"\n===== {st}: {n} paired queries ({len(set(k[0] for k in keys))} tasks) =====")
    def col(Q, f): return np.array([f(Q[k]) for k in keys], float)
    rows = {
     "clean exact (limit)": (lambda q: q["dyn"]["limit_exact"]), "clean exact at T": (lambda q: q["exact_T"]),
     "oracle@32 (any random-init draw exact)": (lambda q: q["sel"]["oracle"]), "per-draw exact rate": (lambda q: q["sel"]["draw_exact_rate"]),
     "residual-selected (y) exact": (lambda q: q["sel"]["t1r_y"]), "residual-selected (z) exact": (lambda q: q["sel"]["t1r_z"]), "majority exact": (lambda q: q["sel"]["majority"]),
     "draws converged": (lambda q: q["sel"]["converged_rate"]), "spurious (converged & wrong) per draw": (lambda q: q["sel"]["spurious_rate"]), "n distinct endpoints / 32": (lambda q: q["sel"]["n_distinct"]),
     "retention of the handed truth": (lambda q: q["retain"]["gt"]), "retention of its own limit": (lambda q: q["retain"]["own"]),
     "clean run converged": (lambda q: q["dyn"]["converged_at"] is not None), "size right at step 1": (lambda q: q["dyn"]["size1_ok"]), "size right at the end": (lambda q: q["dyn"]["size_last_ok"]),
     "commit@1 (conf>.9)": (lambda q: q["dyn"]["commit1"]), "commit@16": (lambda q: q["dyn"]["commit_last"]), "flips per cell": (lambda q: q["dyn"]["flips_per_cell"]),
     "H_q first / last": (lambda q: q["dyn"]["H_q_first"]),
    }
    print(f"{'row':44s} | {'RI (Dri)':>9s} | {'plain (C53)':>11s} | only-RI / only-plain (binary rows)")
    for name, f in rows.items():
        a, b = col(A, f), col(B, f)
        binary = set(np.unique(np.concatenate([a, b]))) <= {0.0, 1.0}
        extra = f" | {int(((a==1)&(b==0)).sum())} / {int(((a==0)&(b==1)).sum())}" if binary else ""
        print(f"{name:44s} | {100*a.mean() if binary or a.max()<=1 else a.mean():9.2f} | {100*b.mean() if binary or b.max()<=1 else b.mean():11.2f}{extra}")
    # basin conditioning: P(oracle | retain_gt) vs P(oracle | not retain_gt); and clean exact by retention
    for lab, Q in (("RI", A), ("plain", B)):
        ret = col(Q, lambda q: q["retain"]["gt"]).astype(bool); orc = col(Q, lambda q: q["sel"]["oracle"]).astype(bool); cl = col(Q, lambda q: q["dyn"]["limit_exact"]).astype(bool)
        p1 = orc[ret].mean() if ret.any() else float("nan"); p0 = orc[~ret].mean() if (~ret).any() else float("nan")
        print(f"  [{lab}] basin conditioning: P(oracle@32 | retains truth) {100*p1:.1f} % (n {ret.sum()}) vs P(oracle | does not) {100*p0:.1f} % (n {(~ret).sum()}) -> enrichment x{(p1/p0 if p0>0 else float('inf')):.1f}; clean exact | retains {100*cl[ret].mean():.1f} vs {100*cl[~ret].mean():.1f}")
        fails = [Q[k]["dyn"] for k in keys if not Q[k]["dyn"]["limit_exact"]]
        print(f"  [{lab}] failures (n {len(fails)}): cell acc at end {100*mean([d.get('cell_acc_last') for d in fails]):.1f} %; conf on WRONG cells at end {mean([d.get('wrong_conf_last') for d in fails]):.3f}; of committed cells wrong {100*mean([d.get('committed_wrong_last') for d in fails]):.1f} %; conf on right cells {mean([d.get('right_conf_last') for d in fails]):.3f}; converged-wrong of converged {100*mean([not d['limit_exact'] for d in fails if d['converged_at'] is not None]) if any(d['converged_at'] is not None for d in fails) else float('nan'):.1f} %")
        dr = [d for k in keys for d in Q[k]["draws"]]
        ex = np.array([d["exact"] for d in dr]); ry = np.array([d["res_y"] for d in dr]); conv = np.array([d["converged"] for d in dr])
        from scipy import stats
        auc_y = 1 - stats.mannwhitneyu(ry[ex], ry[~ex], alternative="less").statistic / (ex.sum() * (~ex).sum()) if ex.any() and (~ex).any() else None
        med = np.median(ry[ex]) if ex.any() else None; spur_e5 = float((ry[~ex] <= med).mean()) if med is not None else None
        print(f"  [{lab}] draws n {len(dr)}: exact {100*ex.mean():.2f} %; converged {100*conv.mean():.1f} %; converged & wrong {100*(conv & ~ex).mean():.1f} %; residual(y) AUC for exact {auc_y if auc_y is None else round(auc_y,3)}; E5 spurious (wrong draws below the correct median residual) {'-' if spur_e5 is None else f'{100*spur_e5:.1f} %'}")
