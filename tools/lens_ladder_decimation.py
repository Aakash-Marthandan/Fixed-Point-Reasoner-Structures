#!/usr/bin/env python3
"""DOES THE ATTENTION-MIXED SYMMETRIC DEC STILL DECIMATE?  (2026-09-19; the PI: "does decimation still happen with the attention mixer since
that is our central claim").  DESCRIPTIVE, zero cloud.  A DRIVER around the paper session's tools/lens_repair_radius.py (imported, never
edited): it registers the width ladder's selected checkpoints as extra models and runs that lens's Panel A — the fixed start, 64 iterations,
no halting, no noise, EMA weights, the same 512 rating-stratified test puzzles as the width-192 model's banked panel — then prints the
decimation signature beside the banked MLP-mixer panels (C5 = width 192 at 46k; C0 / A7 = width 384), which it only READS.

The signature, as the paper states it (memory `commit-then-repair-record`): after ITERATION 1 nearly every empty cell is committed (top-digit
confidence > 0.9); on puzzles that end up SOLVED a large share of those commitments is WRONG and is repaired within ~8 iterations; on FAILED
puzzles the wrong share stays flat.

EXPECTATIONS, written before any row (credences): E1 committed after iteration 1 >= 90 % on every attention arm (0.80) · E2 committed-and-wrong
on solved rows >= 10 % of the empty cells, i.e. there IS something to repair (0.75), and smaller than the width-192 MLP model's (0.85) ·
E3 median first-exact iteration on solved rows <= the MLP model's (0.85) · E4 failed rows flat: wrong share at 64 within 5 points of its
iteration-1 value (0.65).

  JAX_PLATFORMS=cpu .venv/bin/python tools/lens_ladder_decimation.py --run SA256 [SA192 W256 ...]   (resume-safe per model)
  .venv/bin/python tools/lens_ladder_decimation.py --report
  .venv/bin/python tools/lens_ladder_decimation.py --selftest
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
OUT = ROOT / "runs/analysis/ladder_decimation_20260919"
REF = ROOT / "runs/analysis/repair_radius_20260917"          # the paper session's banked panels: READ ONLY
PULL = ROOT / "runs/_wladder_pull"
LADDER = {   # name -> (the arm's SELECTED checkpoint, label)
    "SA256": (PULL / "grids/runs/pretrainchamp_SA256/ckpt_028000.pkl", "attention mixers, hidden 256, selected 28k"),
    "SA192": (PULL / "tgz/p1/SA192_ckpt.pkl", "attention mixers, hidden 192, selected 30k (= the final state)"),
    "SA128": (PULL / "tgz/p0/SA128_ckpt.pkl", "attention mixers, hidden 128, selected 30k (= the final state)"),
    "W256": (PULL / "grids/runs/pretrainchamp_W256/ckpt_018000.pkl", "the 81-cell SwiGLU, hidden 256, selected 18k"),
    "W128": (PULL / "tgz/p0/W128_ckpt.pkl", "the 81-cell SwiGLU, hidden 128, selected 30k (= the final state)"),
}
REFS = {"C5": "the 81-cell SwiGLU, hidden 192, 46k (the paper's model)", "C0": "the 81-cell SwiGLU, hidden 384", "A7": "the 81-cell SwiGLU, hidden 384, plain recipe"}


def signature(D, st="cold"):
    """the decimation signature of one Panel A record (arrays per puzzle; e = wrong share of the empty cells by iteration)."""
    g = lambda k: np.asarray(D[f"{st}_{k}"]); S = g("ex64").astype(bool); F = ~S
    m = lambda a, w: float(a[w].mean()) if w.any() else None
    e = g("e"); fe = g("fe")
    return dict(n=int(len(S)), solved64=float(S.mean()), solved16=float(g("ex16").astype(bool).mean()), c1=float(g("c1").mean()),
                e1_solved=m(g("e1"), S), e1_failed=m(g("e1"), F), cwE_solved=m(g("cwE"), S), cwE_failed=m(g("cwE"), F), w1_solved=m(g("w1"), S),
                e_solved=[m(e[:, i], S) for i in range(e.shape[1])], e_failed=[m(e[:, i], F) for i in range(e.shape[1])],
                fe_med=float(np.median(fe[S])) if S.any() else None, fe_p90=float(np.percentile(fe[S], 90)) if S.any() else None,
                e_end_failed=m(g("e_end"), F), c2w_solved=m(g("c2w"), S), n_failed=int(F.sum()))


def verdicts(s, ref):
    """the four expectations read on one arm's signature against the reference (the width-192 MLP model)."""
    flat = None if (s["e1_failed"] is None or s["e_end_failed"] is None) else abs(s["e_end_failed"] - s["e1_failed"]) <= 0.05
    return dict(E1=s["c1"] >= 0.90, E2a=(s["cwE_solved"] or 0) >= 0.10, E2b=(s["cwE_solved"] or 0) < (ref["cwE_solved"] or 0),
                E3=(s["fe_med"] is not None and ref["fe_med"] is not None and s["fe_med"] <= ref["fe_med"]), E4=flat)


def pct(x): return "  n/a" if x is None else f"{100 * x:5.1f}"


def report():
    import lens_repair_radius as L
    steps = list(L.STEPS); rows = {}
    for name in list(LADDER) + list(REFS):
        p = (OUT if name in LADDER else REF) / f"{name}_A.npz"
        if p.exists(): rows[name] = (signature(dict(np.load(p, allow_pickle=True))), (LADDER.get(name, (None, REFS.get(name)))[1]))
    lines = ["DECIMATION ON THE LADDER'S ARMS (EXPLORATORY; the fixed start, 64 iterations, the identical 512 stratified test puzzles; Panel A of tools/lens_repair_radius.py)",
             f"{'model':6s} {'solved@16':>9s} {'@64':>6s} | after iteration 1: {'committed':>9s} {'wrong(solved)':>13s} {'committed&wrong(solved)':>23s} {'wrong among committed':>21s} | first-exact median / p90 | failed rows: wrong at 1 -> at 64 (n)"]
    for name, (s, lab) in rows.items():
        lines.append(f"{name:6s} {pct(s['solved16']):>9s} {pct(s['solved64']):>6s} |                    {pct(s['c1']):>9s} {pct(s['e1_solved']):>13s} {pct(s['cwE_solved']):>23s} {pct(s['w1_solved']):>21s} | {s['fe_med']} / {s['fe_p90']}"
                     f" | {pct(s['e1_failed'])} -> {pct(s['e_end_failed'])} ({s['n_failed']})   [{lab}]")
    lines.append("wrong share of the empty cells by iteration, SOLVED rows (iterations " + " ".join(str(t) for t in steps) + ")")
    for name, (s, _) in rows.items(): lines.append(f"  {name:6s} " + " ".join(pct(x) for x in s["e_solved"]))
    if "C5" in rows:
        lines.append("the four expectations against the width-192 MLP model (C5):")
        for name in LADDER:
            if name in rows: v = verdicts(rows[name][0], rows["C5"][0]); lines.append(f"  {name:6s} " + "  ".join(f"{k} {'HOLDS' if x else ('n/a' if x is None else 'FAILS')}" for k, x in v.items()))
    text = "\n".join(lines); print(text); OUT.mkdir(parents=True, exist_ok=True); (OUT / "report.txt").write_text(text + "\n")
    (OUT / "report.json").write_text(json.dumps({k: v[0] for k, v in rows.items()}, indent=1, default=float))


def run(models, n):
    import lens_repair_radius as L
    for name in models:
        ck, lab = LADDER[name]; assert ck.exists(), f"{name}: {ck} absent"
        L.MODELS[name] = (str(ck), lab)                      # an absolute path: ROOT / <absolute> is that path
        if (OUT / f"{name}_A.npz").exists(): print(f"{name}: banked, skipped"); continue
        sys.argv = ["lens_repair_radius.py", "--run", "--model", name, "--panel", "A", "--starts", "cold", "--n", str(n), "--out", str(OUT)]
        print(f"=== {name}: Panel A, the fixed start, n {n} ===", flush=True); L.main()


def selftest():
    rng = np.random.default_rng(0); n, T = 40, 12; S = np.arange(n) < 30
    e = np.where(S[:, None], np.linspace(0.3, 0.0, T)[None, :], 0.4); D = {"cold_e": e, "cold_e1": e[:, 0], "cold_fe": np.where(S, 5, -1), "cold_e_end": e[:, -1],
         "cold_ex16": S, "cold_ex64": S, "cold_c1": np.full(n, 0.95), "cold_w1": np.full(n, 0.2), "cold_cwE": np.where(S, 0.25, 0.35), "cold_c2w": np.full(n, 0.01)}
    s = signature(D); ref = dict(s); ref["cwE_solved"] = 0.30; ref["fe_med"] = 6.0
    ok = [abs(s["solved64"] - 0.75) < 1e-12, abs(s["c1"] - 0.95) < 1e-12, abs(s["cwE_solved"] - 0.25) < 1e-12, abs(s["cwE_failed"] - 0.35) < 1e-12, s["fe_med"] == 5.0, s["n_failed"] == 10,
          abs(s["e_solved"][0] - 0.3) < 1e-12 and abs(s["e_solved"][-1]) < 1e-12, abs(s["e_end_failed"] - 0.4) < 1e-12]
    v = verdicts(s, ref); ok += [v == dict(E1=True, E2a=True, E2b=True, E3=True, E4=True)]
    s2 = dict(s); s2["c1"] = 0.5; s2["cwE_solved"] = 0.05; s2["fe_med"] = 9.0; s2["e_end_failed"] = 0.1
    ok += [verdicts(s2, ref) == dict(E1=False, E2a=False, E2b=True, E3=False, E4=False), pct(None).strip() == "n/a" and pct(0.256).strip() == "25.6"]
    print(f"selftest {'OK' if all(ok) else 'FAILED'}: {sum(ok)}/{len(ok)} checks"); return 0 if all(ok) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--run", nargs="*"); ap.add_argument("--n", type=int, default=512); ap.add_argument("--report", action="store_true"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest: sys.exit(selftest())
    if a.run is not None: run(a.run or list(LADDER), a.n)
    if a.report: report()
