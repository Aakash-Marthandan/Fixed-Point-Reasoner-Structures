#!/usr/bin/env python3
# A DESCRIPTIVE reader for the DEC-ARC evaluator's artifacts (2026-09-16; analysis-time, no rules, labeled exploratory). It reads the same
# artifacts the frozen analyzer reads (runs/decarceval_<arm>/<set>/summary.json + results*.jsonl; runs/pretraindecarc_<arm>/
# metrics.jsonl) and prints what the registered letters sit on: every row's counts beside its fractions (the count floor
# is the reading at these accuracies), the depth curves, the PAIRED per-query reading on the identical val-hard queries
# with an exact McNemar test per arm pair (the measurement law's upgrade path), the training-side monitors, and the
# Sudoku-vs-ARC instrument column for the report. `--selftest` runs the pairing and the McNemar on hand-built records
# (the 2026-09-09 descriptive-tool lesson: a fresh tool prints nothing until its selftest passes).
"""
  .venv/bin/python tools/lens_decarc_read.py --root runs/_decarc_stage/runs [--out runs/analysis/decarc_lens_20260916]
  .venv/bin/python tools/lens_decarc_read.py --selftest
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path

ARMS = ("D0", "D1", "D2", "N0")
DEC = ("D0", "D1", "D2")
SETS = ("valhard", "dev30", "mon96", "rg96", "rt48", "arc1eval", "valhard_final", "valhard_bridge")
N_Q = {"valhard": 144, "dev30": 31, "mon96": 96, "rg96": 288, "rt48": 144, "arc1eval": 419, "valhard_final": 144, "valhard_bridge": 144}


def rj(p: Path):
    return json.load(open(p)) if p.exists() else None


def pct(x, n=None):
    """'a/n = p %' when the count is known, else 'p %'; '–' when missing."""
    if x is None: return "–"
    if n: return f"{round(x * n)}/{n} = {100 * x:.1f} %"
    return f"{100 * x:.1f} %"


def f3(x):
    return "–" if x is None else f"{x:.3f}"


def mcnemar_exact(b: int, c: int):
    """two-sided exact binomial McNemar on the discordant pairs (b = A solved, B not; c = B solved, A not)."""
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) * 0.5 ** n
    return min(1.0, p)


# ---------- per-query records ----------
def query_outcomes(root: Path, arm: str, st: str):
    """{(task, q): exact_limit} from the shard results (DEC: results_*.jsonl; N0: results.jsonl); exact = dyn.limit_exact."""
    d = root / f"decarceval_{arm}" / st
    out = {}
    files = sorted(d.glob("results*.jsonl")) or sorted(d.glob("s*/results*.jsonl"))   # the native public row keeps its per-query records in the shard dirs only
    for f in files:
        for line in open(f):
            if not line.strip(): continue
            r = json.loads(line)
            for q in r.get("queries", []):
                dyn = q.get("dyn") or {}
                ex = dyn.get("limit_exact")
                if ex is None: ex = q.get("exact_T")
                out[(r["task"], q.get("q"))] = bool(ex)
    return out


def paired(root: Path, st: str, arms):
    O = {a: query_outcomes(root, a, st) for a in arms}
    keys = set.intersection(*(set(o) for o in O.values() if o)) if any(O.values()) else set()
    lines = [f"paired on {len(keys)} identical ({st}) queries"]
    solved = {a: {k for k in keys if O[a].get(k)} for a in arms}
    for a in arms:
        tasks = sorted({k[0] for k in solved[a]})
        lines.append(f"  {a}: solved {len(solved[a])}  tasks: {', '.join(tasks) if tasks else '–'}")
    lines.append("  pairwise (b = first solved & second not; c = the reverse; exact two-sided McNemar):")
    for i, a in enumerate(arms):
        for bb in arms[i + 1:]:
            b = len(solved[a] - solved[bb]); c = len(solved[bb] - solved[a]); both = len(solved[a] & solved[bb])
            lines.append(f"    {a} vs {bb}: both {both}, only-{a} {b}, only-{bb} {c}, neither {len(keys) - both - b - c}; McNemar p = {mcnemar_exact(b, c):.3f}")
    return lines, solved


# ---------- summaries ----------
def load(root: Path):
    rec = {}
    for a in ARMS:
        rec[a] = {"evals": {}, "monitor": [], "train": [], "vsel": rj(root / f"pretraindecarc_{a}" / "vsel.json")}
        for st in SETS:
            s = rj(root / f"decarceval_{a}" / st / "summary.json")
            if s: rec[a]["evals"][st] = s
        mf = root / f"pretraindecarc_{a}" / "metrics.jsonl"
        if mf.exists():
            for line in open(mf):
                try: j = json.loads(line)
                except Exception: continue
                if "monitor" in j: rec[a]["monitor"].append(j["monitor"])
                elif "val" in j: rec[a]["monitor"].append(j["val"])
                else: rec[a]["train"].append(j)
    return rec


def table(rec):
    """the per-arm × set table of the registered fields, counts beside fractions."""
    L = ["| arm | set | n_q | cold exact (limit) | at T | ever | lost | first-exact med | converged | conv-wrong | commit@1 | conf@1 | flips/cell | size@1 ok | fail cell acc | fail wrong conf | oracle | majority | n_distinct | AUC res_z | spurious | retention (all) | ret. solved / failed | ladder ε0/.2/.4/.6/.8 | vote p1/p2/any | flip colour / floor | halt AUC | fit s/task | wall s/task |",
         "|" + "---|" * 30]
    for a in ARMS:
        for st, s in rec[a]["evals"].items():
            n = s.get("n_queries") or N_Q.get(st)
            g = s.get
            ret_all = g("retention_gt", g("retain_gt")); rs = g("retention_gt_solved"); rf = g("retention_gt_failed", g("retain_gt_on_failures"))
            lad = g("ladder"); lad_s = "/".join(f"{k}:{100 * lad[k]:.1f}" for k in sorted(lad, key=float)) if lad else "–"
            v = g("vote"); vote_s = f"{100 * v['pass1']:.1f}/{100 * v['pass2']:.1f}/{100 * v['any_view']:.1f}" if v else "–"
            fl = g("flip"); flip_s = f"{100 * fl['flip_rate_colour']:.1f} / {100 * fl['flip_rate_floor']:.1f}" if fl else "–"
            nd = g("n_distinct", g("n_distinct_mean")); fe = g("first_exact_med", g("first_exact_median_solved"))
            L.append("| " + " | ".join([a, st, str(n), pct(g("clean_exact_limit"), n), pct(g("clean_exact_T"), n), pct(g("ever_exact"), n), str(g("lost", "–")),
                                      "–" if fe is None else f"{fe:g}", pct(g("converged_frac")), f3(g("converged_wrong_of_converged")), f3(g("commit1")), f3(g("conf1")),
                                      "–" if g("flips_per_cell") is None else f"{g('flips_per_cell'):.3f}", f3(g("size1_ok")), f3(g("fail_cell_acc_last")), f3(g("fail_wrong_conf_last")),
                                      pct(g("oracle"), n), pct(g("majority"), n), "–" if nd is None else f"{nd:.2f}", f3(g("auc_res_z")), f3(g("spurious_rate")),
                                      pct(ret_all, n), f"{f3(rs)} / {f3(rf)}", lad_s, vote_s, flip_s, f3(g("halt_auc")),
                                      "–" if g("fit_s_mean") is None else f"{g('fit_s_mean'):.0f}", "–" if g("wall_s_mean") is None else f"{g('wall_s_mean'):.0f}"]) + " |")
    return L


def depth(rec):
    L = ["| arm | set | exact by step 1 → 16 (counts) |", "|---|---|---|"]
    for a in ARMS:
        for st, s in rec[a]["evals"].items():
            e = s.get("exact_by_step")
            if not e: continue
            n = s.get("n_queries") or N_Q.get(st)
            L.append(f"| {a} | {st} | " + " ".join(str(round(x * n)) for x in e) + " |")
    return L


def training(rec):
    L = ["| arm | monitor rows | max EMA-t16 (count/96) @ step | selected | final EMA-t16 | max pix_ema | final pix_ema | train loss (last 50 rows) | train_exact (last 5k) | fpa_ce (last 5k) |", "|" + "---|" * 10]
    for a in ARMS:
        mon = rec[a]["monitor"]; tr = rec[a]["train"]; vs = rec[a]["vsel"] or {}
        if not mon: L.append(f"| {a} | 0 | – | {vs.get('step', '–')} | – | – | – | – | – | – |"); continue
        if "val_t16_ema" in mon[0]:
            vals = [(m["val_t16_ema"] * 96, m["step"]) for m in mon]; pix = [m["val_pix_ema"] for m in mon]
            mx = max(vals, key=lambda t: (t[0], -t[1]))
            fin = vals[-1][0]; pmx = max(pix); pfin = pix[-1]
        else:   # N0: val_exact counts
            vals = [(m["val_exact"], m["step"]) for m in mon]; pix = [m["val_pix_mean"] for m in mon]
            mx = max(vals, key=lambda t: (t[0], -t[1])); fin = vals[-1][0]; pmx = max(pix); pfin = pix[-1]
        last = tr[-50:]; last5k = [r for r in tr if r["step"] > tr[-1]["step"] - 5000]
        loss = sum(r.get("loss", 0) for r in last) / max(1, len(last))
        te = [r["train_exact"] for r in last5k if "train_exact" in r]; fc = [r["fpa_ce"] for r in last5k if "fpa_ce" in r]
        L.append(f"| {a} | {len(mon)} | {mx[0]:.0f}/96 @ {mx[1]} | {vs.get('step', '–')} | {fin:.0f}/96 | {pmx:.3f} | {pfin:.3f} | {loss:.3f} | "
                 + (f"{100 * sum(te) / len(te):.1f} %" if te else "–") + " | " + (f"{sum(fc) / len(fc):.3f}" if fc else "–") + " |")
    return L


def starts(rec):
    L = ["| arm | set | cold start | exact | buffers exact / agree / lost | rifix | symfix |", "|---|---|---|---|---|---|---|"]
    for a in DEC:
        for st in ("valhard", "mon96", "dev30"):
            s = rec[a]["evals"].get(st)
            if not s or not s.get("starts"): continue
            n = s.get("n_queries"); S = s["starts"]
            cell = lambda k: f"{pct(S[k]['exact_T'], n)} / {100 * S[k]['agree_T']:.0f} % / {S[k]['lost']}" if k in S else "–"
            L.append(f"| {a} | {st} | {s.get('eval_start')} | {pct(s.get('clean_exact_T'), n)} | {cell('buffers')} | {cell('rifix')} | {cell('symfix')} |")
    return L


def column(rec, solved_vh):
    """the campaign's column for the Sudoku-vs-ARC instrument table, one line per instrument row (Note_2026-09-10 §0)."""
    class D(dict):
        def __getitem__(self, k):
            x = dict.get(self, k)
            return D(x) if isinstance(x, dict) else (x if x is not None else float("nan"))
        def get(self, k, d=None):
            x = dict.get(self, k, d); return D(x) if isinstance(x, dict) else x
    v = {a: D(rec[a]["evals"].get("valhard", {})) for a in ARMS}; m = {a: D(rec[a]["evals"].get("mon96", {})) for a in DEC}
    p = {a: D(rec[a]["evals"].get("arc1eval", {})) for a in ARMS}
    n0 = v["N0"]
    L = ["| instrument | DEC-ARC w160 RI+FPA (D0 / D1) | the plain twin D2 | the native d96 control N0 |", "|---|---|---|---|"]
    L.append(f"| retention: handed the answer, does the model keep it (val-hard) | {pct(v['D0'].get('retention_gt'), 144)} / {pct(v['D1'].get('retention_gt'), 144)}; on solved {f3(v['D0'].get('retention_gt_solved'))} / {f3(v['D1'].get('retention_gt_solved'))}, on failed {f3(v['D0'].get('retention_gt_failed'))} / {f3(v['D1'].get('retention_gt_failed'))} | {pct(v['D2'].get('retention_gt'), 144)}; on failed {f3(v['D2'].get('retention_gt_failed'))} | {pct(n0.get('retain_gt'), 144)}; on failures {f3(n0.get('retain_gt_on_failures'))}; own endpoint {f3(n0.get('retain_own'))} |")
    L.append(f"| retention on TRAINED tasks' held-out queries with their trained codes, no fit (mon96) | {pct(m['D0'].get('retention_gt'), 96)} / {pct(m['D1'].get('retention_gt'), 96)} (cold {pct(m['D0'].get('clean_exact_limit'), 96)} / {pct(m['D1'].get('clean_exact_limit'), 96)}) | {pct(m['D2'].get('retention_gt'), 96)} (cold {pct(m['D2'].get('clean_exact_limit'), 96)}) | – |")
    L.append(f"| some restart solves it, given the answer is kept vs not | {f3(v['D0'].get('oracle_given_retains'))} vs {f3(v['D0'].get('oracle_given_not'))} / {f3(v['D1'].get('oracle_given_retains'))} vs {f3(v['D1'].get('oracle_given_not'))} | {f3(v['D2'].get('oracle_given_retains'))} vs {f3(v['D2'].get('oracle_given_not'))} | spurious on basin vs no-basin queries {f3(n0.get('spurious_on_basin_queries'))} vs {f3(n0.get('spurious_on_nobasin_queries'))} |")
    L.append(f"| single pass (val-hard, T1×150 fit) and first-exact step | {pct(v['D0'].get('clean_exact_limit'), 144)} / {pct(v['D1'].get('clean_exact_limit'), 144)}; first-exact median {v['D0'].get('first_exact_med')} / {v['D1'].get('first_exact_med')} | {pct(v['D2'].get('clean_exact_limit'), 144)} | {pct(n0.get('clean_exact_limit'), 144)} (deployed 6-pass fit {pct(rec['N0']['evals'].get('valhard_bridge', {}).get('clean_exact_limit'), 144)}) |")
    L.append(f"| depth: exact at step 16 − step 2; solved-then-lost | {100 * (v['D0']['exact_by_step'][-1] - v['D0']['exact_by_step'][1]):+.1f} / {100 * (v['D1']['exact_by_step'][-1] - v['D1']['exact_by_step'][1]):+.1f} pp; lost {v['D0'].get('lost')} / {v['D1'].get('lost')} of 144 | {100 * (v['D2']['exact_by_step'][-1] - v['D2']['exact_by_step'][1]):+.1f} pp; lost {v['D2'].get('lost')} | at T {pct(n0.get('clean_exact_T'), 144)} vs limit {pct(n0.get('clean_exact_limit'), 144)} |")
    L.append(f"| commitment after one iteration (cells at conf > .9); confidence | {f3(v['D0'].get('commit1'))} / {f3(v['D1'].get('commit1'))}; conf {f3(v['D0'].get('conf1'))} / {f3(v['D1'].get('conf1'))} → {f3(v['D0'].get('commit_last'))} / {f3(v['D1'].get('commit_last'))} by 16 | {f3(v['D2'].get('commit1'))} → {f3(v['D2'].get('commit_last'))} | {f3(n0.get('commit1'))} → {f3(n0.get('commit_last'))}; conf {f3(n0.get('conf1'))} |")
    L.append(f"| failure texture: converged; converged-wrong; flips per cell; conf on wrong cells | {pct(v['D0'].get('converged_frac'))} / {pct(v['D1'].get('converged_frac'))}; {f3(v['D0'].get('converged_wrong_of_converged'))} / {f3(v['D1'].get('converged_wrong_of_converged'))}; {v['D0'].get('flips_per_cell'):.3f} / {v['D1'].get('flips_per_cell'):.3f}; {f3(v['D0'].get('fail_wrong_conf_last'))} / {f3(v['D1'].get('fail_wrong_conf_last'))} | {pct(v['D2'].get('converged_frac'))}; {f3(v['D2'].get('converged_wrong_of_converged'))}; {v['D2'].get('flips_per_cell'):.3f}; {f3(v['D2'].get('fail_wrong_conf_last'))} | {pct(n0.get('converged_frac'))}; {f3(n0.get('converged_wrong_of_converged'))}; {n0.get('flips_per_cell'):.3f}; {f3(n0.get('fail_wrong_conf_last'))} |")
    L.append(f"| the selector: residual-on-z AUC; spurious rate; distinct endpoints of k; oracle@k − cold | {f3(v['D0'].get('auc_res_z'))} / {f3(v['D1'].get('auc_res_z'))}; {f3(v['D0'].get('spurious_rate'))} / {f3(v['D1'].get('spurious_rate'))}; {v['D0'].get('n_distinct'):.2f} / {v['D1'].get('n_distinct'):.2f} of 16; {100 * (v['D0']['oracle'] - v['D0']['clean_exact_limit']):+.1f} / {100 * (v['D1']['oracle'] - v['D1']['clean_exact_limit']):+.1f} pp | {f3(v['D2'].get('auc_res_z'))}; {f3(v['D2'].get('spurious_rate'))}; {v['D2'].get('n_distinct'):.2f}; {100 * (v['D2']['oracle'] - v['D2']['clean_exact_limit']):+.1f} pp | {f3(n0.get('auc_res_z'))}; {f3(n0.get('spurious_rate'))}; {n0.get('n_distinct_mean'):.2f} of 16; oracle {pct(n0.get('oracle'), 144)} − cold {100 * (n0['oracle'] - n0['clean_exact_limit']):+.1f} pp |")
    L.append(f"| init-invariance: the other starts of the same fitted code (exact; per-query agreement with the cold row) | buffers {pct(v['D0']['starts']['buffers']['exact_T'], 144)} / {pct(v['D1']['starts']['buffers']['exact_T'], 144)}, agree {100 * v['D0']['starts']['buffers']['agree_T']:.0f} / {100 * v['D1']['starts']['buffers']['agree_T']:.0f} %; rifix agree {100 * v['D0']['starts']['rifix']['agree_T']:.0f} / {100 * v['D1']['starts']['rifix']['agree_T']:.0f} % | buffers agree {100 * v['D2']['starts']['buffers']['agree_T']:.0f} %; rifix {100 * v['D2']['starts']['rifix']['agree_T']:.0f} % | draws converged {f3(n0.get('draw_converged_rate'))}; per-draw exact {pct(n0.get('draw_exact_rate'), 144)} |")
    L.append(f"| the dihedral vote (2 views + identity): pass@1 / pass@2 / any view; distinct answers solved vs failed | {100 * v['D0']['vote']['pass1']:.1f} / {100 * v['D0']['vote']['pass2']:.1f} / {100 * v['D0']['vote']['any_view']:.1f}; {v['D0']['vote']['n_distinct_solved']:.1f} vs {v['D0']['vote']['n_distinct_failed']:.1f} | {100 * v['D2']['vote']['pass1']:.1f} / {100 * v['D2']['vote']['pass2']:.1f} / {100 * v['D2']['vote']['any_view']:.1f} | – |")
    L.append(f"| exact colour symmetry: flip rate under an S10 palette of the whole task vs the re-fit floor | {100 * v['D0']['flip']['flip_rate_colour']:.1f} vs {100 * v['D0']['flip']['flip_rate_floor']:.1f} % / {100 * v['D1']['flip']['flip_rate_colour']:.1f} vs {100 * v['D1']['flip']['flip_rate_floor']:.1f} % | {100 * v['D2']['flip']['flip_rate_colour']:.1f} vs {100 * v['D2']['flip']['flip_rate_floor']:.1f} % | (a learned colour embedding; not measured) |")
    L.append(f"| the halting head as a verifier (AUC) | {f3(v['D0'].get('halt_auc'))} / {f3(v['D1'].get('halt_auc'))} | {f3(v['D2'].get('halt_auc'))} | H_q AUC {f3(n0.get('auc_Hq'))} |")
    L.append(f"| the output-size readout right at step 1 | {f3(v['D0'].get('size1_ok'))} / {f3(v['D1'].get('size1_ok'))} | {f3(v['D2'].get('size1_ok'))} | {f3(n0.get('size1_ok'))} |")
    L.append(f"| ARC-AGI-1 evaluation (400 tasks / 419 inputs), pass@1, 1 view, k 4 | {pct(p['D0'].get('clean_exact_limit'), 419)} / {pct(p['D1'].get('clean_exact_limit'), 419)} | {pct(p['D2'].get('clean_exact_limit'), 419)} | {pct(p['N0'].get('clean_exact_limit'), 419)}; retention {pct(p['N0'].get('retain_gt'), 419)} |")
    return L


def main(root: Path, out: Path | None):
    rec = load(root)
    S = []
    S.append("DEC-ARC NIGHT — the descriptive reader (tools/lens_decarc_read.py; exploratory, no rules; every number from the artifacts)")
    S.append(f"root {root}")
    S.append("\n## A. The per-arm × set table (counts beside fractions; the count floor IS the reading at these accuracies)")
    S += table(rec)
    S.append("\n## B. Depth: exact by step (counts)")
    S += depth(rec)
    S.append("\n## C. The paired reading on the identical queries (exact McNemar on the discordant pairs)")
    for st, arms in (("valhard", ARMS), ("dev30", ARMS), ("rg96", ARMS), ("rt48", ARMS), ("arc1eval", ARMS), ("mon96", DEC)):
        try:
            lines, solved = paired(root, st, arms)
            S += lines
        except Exception as e:
            S.append(f"paired {st}: unreadable ({e})")
    S.append("\n## D. The other starts of the same fitted code (the width-192 lens's ARC twin)")
    S += starts(rec)
    S.append("\n## E. The training side (the 2k monitors; the trainer's own rows)")
    S += training(rec)
    S.append("\n## F. The campaign's column for the Sudoku-vs-ARC instrument table")
    _, solved_vh = paired(root, "valhard", ARMS)
    S += column(rec, solved_vh)
    text = "\n".join(S)
    print(text)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        (out.with_suffix(".txt")).write_text(text + "\n")
        json.dump({a: {"evals": rec[a]["evals"], "vsel": rec[a]["vsel"], "monitor": rec[a]["monitor"]} for a in ARMS}, open(out.with_suffix(".json"), "w"), indent=1)
        print(f"\nwritten: {out.with_suffix('.txt')} and .json")


def selftest():
    import tempfile
    ok = 0; bad = []
    def chk(name, cond):
        nonlocal ok
        if cond: ok += 1
        else: bad.append(name)
    # McNemar: b=3, c=0 -> p = 2 * 0.5^3 = 0.25; b=c=0 -> 1.0; b=5, c=0 -> 0.0625; b=1, c=1 -> 1.0
    chk("mcnemar 3/0", abs(mcnemar_exact(3, 0) - 0.25) < 1e-12)
    chk("mcnemar 0/0", mcnemar_exact(0, 0) == 1.0)
    chk("mcnemar 5/0", abs(mcnemar_exact(5, 0) - 0.0625) < 1e-12)
    chk("mcnemar 1/1", mcnemar_exact(1, 1) == 1.0)
    chk("mcnemar 10/2", abs(mcnemar_exact(10, 2) - 2 * (1 + 12 + 66) * 0.5 ** 12) < 1e-12)
    # hand-built records: two arms on 3 tasks x 2 queries; A solves (t1,0),(t2,1); B solves (t2,1),(t3,0),(t3,1)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        def write(arm, solved, fname):
            d = root / f"decarceval_{arm}" / "valhard"; d.mkdir(parents=True)
            with open(d / fname, "w") as f:
                for t in ("t1", "t2", "t3"):
                    f.write(json.dumps({"task": t, "queries": [{"q": q, "dyn": {"limit_exact": (t, q) in solved}} for q in (0, 1)]}) + "\n")
        write("D0", {("t1", 0), ("t2", 1)}, "results_0.jsonl")
        write("N0", {("t2", 1), ("t3", 0), ("t3", 1)}, "results.jsonl")
        lines, solved = paired(root, "valhard", ("D0", "N0"))
        chk("paired count", lines[0].startswith("paired on 6 identical"))
        chk("solved D0", solved["D0"] == {("t1", 0), ("t2", 1)})
        chk("solved N0", solved["N0"] == {("t2", 1), ("t3", 0), ("t3", 1)})
        chk("discordant line", any("both 1, only-D0 1, only-N0 2, neither 2; McNemar p = 1.000" in l for l in lines))
        # the table reader tolerates a missing set and prints counts beside fractions
        (root / "decarceval_D0" / "valhard" / "summary.json").write_text(json.dumps({"n_queries": 144, "clean_exact_limit": 5 / 144, "clean_exact_T": 5 / 144, "flips_per_cell": 0.01, "retention_gt": 5 / 144, "ladder": {"0.0": 5 / 144, "0.2": 5 / 144, "0.4": 5 / 144, "0.6": 5 / 144, "0.8": 5 / 144}}))
        rec = load(root)
        rows = table(rec)
        chk("table row", any("| D0 | valhard | 144 | 5/144 = 3.5 % |" in r for r in rows))
        chk("ladder cell", any("0.0:3.5/0.2:3.5/0.4:3.5/0.6:3.5/0.8:3.5" in r for r in rows))
        chk("missing sets tolerated", len([r for r in rows if r.startswith("| D")]) == 1)
    print(f"selftest {'OK' if not bad else 'FAILED'}: {ok}/{ok + len(bad)} checks" + (f"; failed: {bad}" if bad else ""))
    return 0 if not bad else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/_decarc_stage/runs"); ap.add_argument("--out", default=None); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest: sys.exit(selftest())
    main(Path(a.root), Path(a.out) if a.out else None)
