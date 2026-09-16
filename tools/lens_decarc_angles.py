#!/usr/bin/env python3
# The DEC-ARC evaluator's per-query records from other angles (2026-09-17; descriptive, exploratory, no rules): the records behind the
# summaries answer questions the registered letters do not ask — WHEN the cell stops changing its answer (the decision-time
# census), how many endpoints an input has (the attractor census), whether two seeds produce the SAME wrong answer (is the
# wrong fixed point the input's or the seed's), what a failure looks like at the cell level (near-miss or far-miss; a size
# error or a content error), what the halting head and the residual say on solved vs failed queries, and which ConceptARC
# families any cell solves. `--selftest` runs every helper on hand-built records (the descriptive-tool lesson).
"""
  .venv/bin/python tools/lens_decarc_angles.py --root runs/_decarc_stage/runs [--out runs/analysis/decarc_angles_20260917]
  .venv/bin/python tools/lens_decarc_angles.py --selftest
"""
from __future__ import annotations
import argparse, collections, json, re, statistics, sys
from pathlib import Path

ARMS = ("D0", "D1", "D2", "N0"); DEC = ("D0", "D1", "D2")


def records(root: Path, arm: str, st: str):
    """[(task, q, query-dict)] from results*.jsonl (set level, else the shard dirs)."""
    d = root / f"decarceval_{arm}" / st
    files = sorted(d.glob("results*.jsonl")) or sorted(d.glob("s*/results*.jsonl"))
    out = []
    for f in files:
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                for q in r.get("queries", []): out.append((r["task"], q.get("q"), q))
    return out


def exact(q):
    d = q.get("dyn") or {}
    return bool(d["limit_exact"]) if d.get("limit_exact") is not None else bool(q.get("exact_T"))


def answer_hash(q):
    """the arm's answer for a query = the most common endpoint hash among its draws (a single attractor on nearly every query)."""
    hs = [dr.get("hash") for dr in (q.get("draws") or []) if dr.get("hash")]
    return collections.Counter(hs).most_common(1)[0][0] if hs else None


def family(task: str):
    """ConceptARC ids are ca_<Family><n> where the family may end in 2D/3D: strip the trailing task number only."""
    return re.sub(r"\d+$", "", task[3:]) if task.startswith("ca_") else task.split("_", 1)[0]


def fmt_hist(c: collections.Counter, keys):
    return " ".join(f"{k}:{c.get(k, 0)}" for k in keys)


# ---------- the angles ----------
def decision_time(rs):
    """converged_at = the step after which the canvas never changes (0 = decided at the first iteration); 'never' = not converged."""
    c = collections.Counter(); ex_c = collections.Counter()
    for _, _, q in rs:
        d = q["dyn"]; k = d.get("converged_at"); k = "never" if k is None or k < 0 else min(int(k), 8)
        c[k] += 1
        if exact(q): ex_c[k] += 1
    keys = list(range(0, 9)) + ["never"]
    return f"all: {fmt_hist(c, keys)} | solved: {fmt_hist(ex_c, keys)}  (8 = 8 or later)"


def attractors(rs):
    """distinct endpoints among the k draws per query; how many inputs have ONE; are the multi-endpoint inputs the failures."""
    nd = [q["sel"]["n_distinct"] for _, _, q in rs if q.get("sel") and q["sel"].get("n_distinct") is not None]
    if not nd: return "no draw records"
    one = sum(1 for x in nd if x == 1); multi_solved = sum(1 for _, _, q in rs if q.get("sel") and q["sel"].get("n_distinct", 1) > 1 and exact(q))
    conv = [q["sel"].get("converged_rate") for _, _, q in rs if q.get("sel") and q["sel"].get("converged_rate") is not None]
    return (f"queries with one endpoint {one}/{len(nd)}; two {sum(1 for x in nd if x == 2)}; three+ {sum(1 for x in nd if x >= 3)}; "
            f"max {max(nd)}; multi-endpoint queries that are solved {multi_solved}; mean draw convergence {statistics.mean(conv):.3f}" if conv else "")


def failure_anatomy(rs):
    """on failed queries: cell accuracy at the end, size right, committed-wrong, confidence on wrong cells; the near-miss share
    (rows whose records carry a reduced dyn are skipped per field)."""
    F = [q["dyn"] for _, _, q in rs if not exact(q)]
    if not F: return "no failures"
    def vals(k): return [d[k] for d in F if d.get(k) is not None]
    def med(k, nd=3): v = vals(k); return f"{statistics.median(v):.{nd}f}" if v else "–"
    acc = vals("cell_acc_last"); size = vals("size_last_ok"); fl = vals("flips_per_cell")
    near = sum(1 for a in acc if a >= 0.9); far = sum(1 for a in acc if a < 0.5)
    return (f"failed {len(F)}: cell accuracy median {med('cell_acc_last')} (≥ .9 on {near}, < .5 on {far}, of {len(acc)}); size right at the end on "
            f"{sum(1 for s in size if s)}/{len(size)}; committed-wrong median {med('committed_wrong_last')}; "
            f"wrong-cell confidence median {med('wrong_conf_last')}; right-cell {med('right_conf_last')}; "
            f"flips per cell median {med('flips_per_cell', 4)}, max {max(fl):.3f}" if fl else f"failed {len(F)}: reduced records")


def readouts(rs):
    """the halting logit at the last step and the residual on z, solved vs failed (the AUCs' raw material)."""
    S = [q for _, _, q in rs if exact(q)]; F = [q for _, _, q in rs if not exact(q)]
    def med(qs, f):
        v = [f(q) for q in qs if f(q) is not None]; return f"{statistics.median(v):.3f}" if v else "–"
    qlast = lambda q: (q.get("q_by_step") or [None])[-1]; rz = lambda q: q["dyn"].get("res_z_last"); hq = lambda q: q["dyn"].get("H_q_last")
    return (f"halting logit at 16: solved {med(S, qlast)} vs failed {med(F, qlast)} | residual on z at 16: solved {med(S, rz)} vs failed {med(F, rz)} | "
            f"rule entropy H_q at 16: solved {med(S, hq)} vs failed {med(F, hq)}")


def cross_arm_wrong(R, arms):
    """on the queries every arm fails: do two arms produce the same wrong answer (the same endpoint hash)?"""
    keys = set.intersection(*(set((t, i) for t, i, _ in R[a]) for a in arms))
    H = {a: {(t, i): (answer_hash(q), exact(q)) for t, i, q in R[a]} for a in arms}
    lines = []
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            both_fail = [k for k in keys if not H[a][k][1] and not H[b][k][1] and H[a][k][0] and H[b][k][0]]
            same = sum(1 for k in both_fail if H[a][k][0] == H[b][k][0])
            lines.append(f"    {a} vs {b}: both fail on {len(both_fail)} queries; the SAME wrong answer on {same} ({100 * same / len(both_fail):.0f} %)" if both_fail else f"    {a} vs {b}: no shared failures with hashes")
    return lines


def families(R, arms):
    """per ConceptARC family (3 tasks × 3 queries on val-hard): solved queries per arm."""
    fams = sorted({family(t) for t, _, _ in R[arms[0]]})
    L = ["| family | " + " | ".join(arms) + " | queries |", "|---|" + "---|" * (len(arms) + 1)]
    for f in fams:
        cells = []
        for a in arms:
            qs = [q for t, _, q in R[a] if family(t) == f]
            cells.append(str(sum(1 for q in qs if exact(q))))
        n = sum(1 for t, _, _ in R[arms[0]] if family(t) == f)
        L.append(f"| {f} | " + " | ".join(cells) + f" | {n} |")
    return L


def main(root: Path, out: Path | None):
    S = ["DEC-ARC evaluator records — other angles (tools/lens_decarc_angles.py; descriptive, exploratory, no rules)", f"root {root}"]
    for st in ("valhard", "mon96", "arc1eval"):
        S.append(f"\n## {st}")
        R = {a: records(root, a, st) for a in ARMS}
        R = {a: r for a, r in R.items() if r}
        for a, rs in R.items():
            S.append(f"\n### {a} ({len(rs)} queries; solved {sum(1 for _, _, q in rs if exact(q))})")
            S.append("  decision time (converged_at): " + decision_time(rs))
            S.append("  attractors: " + attractors(rs))
            S.append("  failures: " + failure_anatomy(rs))
            S.append("  readouts: " + readouts(rs))
        dec_here = [a for a in DEC if a in R]
        if len(dec_here) >= 2:
            S.append("\n  the same wrong answer across arms (endpoint hashes on shared failures):")
            S += cross_arm_wrong(R, dec_here)
        if st == "valhard" and len(R) >= 2:
            S.append("\n  ConceptARC families on val-hard (solved queries of 9 per family):")
            S += families(R, [a for a in ARMS if a in R])
    text = "\n".join(S); print(text)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True); out.with_suffix(".txt").write_text(text + "\n"); print(f"\nwritten: {out.with_suffix('.txt')}")


def selftest():
    ok = 0; bad = []
    def chk(name, cond):
        nonlocal ok
        if cond: ok += 1
        else: bad.append(name)
    mk = lambda ex, ca, hashes, nd=1, acc=0.95, size=True, cw=0.1, wc=0.98, rc=0.99, fl=0.01, qb=0.5, rz=0.05: {
        "dyn": {"limit_exact": ex, "converged_at": ca, "cell_acc_last": acc, "size_last_ok": size, "committed_wrong_last": cw, "wrong_conf_last": wc,
                "right_conf_last": rc, "flips_per_cell": fl, "res_z_last": rz, "H_q_last": 0.0},
        "sel": {"n_distinct": nd, "converged_rate": 1.0}, "draws": [{"hash": h} for h in hashes], "q_by_step": [0.0] * 15 + [qb]}
    A = [("ca_CleanUp2", 0, mk(True, 0, ["x"])), ("ca_CleanUp2", 1, mk(False, 1, ["w1", "w1", "w2"], nd=2, acc=0.92)), ("ca_Count4", 0, mk(False, -1, ["w3"], acc=0.4, size=False))]
    B = [("ca_CleanUp2", 0, mk(True, 0, ["x"])), ("ca_CleanUp2", 1, mk(False, 2, ["w1"])), ("ca_Count4", 0, mk(False, 3, ["w9"], acc=0.4))]
    chk("exact", exact(A[0][2]) and not exact(A[1][2]))
    chk("answer hash = majority", answer_hash(A[1][2]) == "w1")
    chk("family", family("ca_TopBottom2D9") == "TopBottom2D" and family("ca_AboveBelow5") == "AboveBelow" and family("rb_c8f0f002") == "rb")
    dt = decision_time(A); chk("decision time", "all: 0:1 1:1 2:0 3:0 4:0 5:0 6:0 7:0 8:0 never:1" in dt and "solved: 0:1" in dt)
    at = attractors(A); chk("attractors", at.startswith("queries with one endpoint 2/3; two 1; three+ 0; max 2; multi-endpoint queries that are solved 0"))
    fa = failure_anatomy(A); chk("failures", fa.startswith("failed 2: cell accuracy median 0.660 (≥ .9 on 1, < .5 on 1, of 2); size right at the end on 1/2"))
    cw = cross_arm_wrong({"D0": A, "D1": B}, ("D0", "D1")); chk("cross-arm wrong", cw == ["    D0 vs D1: both fail on 2 queries; the SAME wrong answer on 1 (50 %)"])
    fm = families({"D0": A, "D1": B}, ("D0", "D1")); chk("families", "| CleanUp | 1 | 1 | 2 |" in fm and "| Count | 0 | 0 | 1 |" in fm)
    ro = readouts(A); chk("readouts", ro.startswith("halting logit at 16: solved 0.500 vs failed 0.500"))
    print(f"selftest {'OK' if not bad else 'FAILED'}: {ok}/{ok + len(bad)} checks" + (f"; failed: {bad}" if bad else ""))
    return 0 if not bad else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="runs/_decarc_stage/runs"); ap.add_argument("--out", default=None); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest: sys.exit(selftest())
    main(Path(a.root), Path(a.out) if a.out else None)
