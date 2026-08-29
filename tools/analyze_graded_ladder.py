# Ledger: GRADED-LADDER RETRO-ANALYSIS ($0, disk-only; 2026-08-29, run DURING the
# 2b ride at the PI's direction — reads ONLY banked past campaigns, never sportBr2b).
# Purpose (twofold):
#  (1) CORRECTION: the rung-1/rung-2 physics passes' S(eps) reader iterated the
#      Sudoku probes' q_ladder DICT as a list -> bool(key-string) == True always
#      -> the "ladder saturates (S=1.00)" instrument notes were READER ARTIFACTS.
#      This tool reads the dict correctly and recomputes every S(eps) curve.
#  (2) The instrument-redesign option-1 payoff, available retroactively: the
#      multi-rung booleans already yield a GRADED radius distribution (first-
#      failure rung per row) once read correctly — no new probe data needed.
# Outputs per arm: retention | TRUE S(eps) of retained | first-failure-rung
# histogram | median cold-failure distance (violations, cells_correct) |
# multi_init_best_wrong median | any retained_per_step leak rows.
"""
  .venv/bin/python tools/analyze_graded_ladder.py  # -> runs/analysis/graded_ladder_20260829.txt
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "analysis" / "graded_ladder_20260829.txt"
L = []
def say(s=""): L.append(str(s)); print(s)

GROUPS = [
    ("d16 wave-1 (sport2, 20k)",   "sudprobe_psport2S*"),
    ("d16/d32 wave-2 (sport2w2)",  "sudprobe_psport2w2*"),
    ("d16 wave-3a (sport3a, 50k)", "sudprobe_psport3a*"),
    ("d64 rung-1 (sportB)",        "sudprobe_psportBB*"),
    ("d96 rung-2 (sportBr2)",      "sudprobe_psportBr2*"),
    ("generator S-port (d16)",     "sudprobe_sud*"),
]

def rungs_of(q):
    if isinstance(q, dict):
        return sorted(((float(k), bool(v)) for k, v in q.items()), key=lambda x: x[0])
    return [(e, bool(v)) for e, v in zip((.05, .1, .2, .4, .6, .8), q)]

def analyze_dir(d: Path):
    rows = [json.loads(l) for l in (d / "results.jsonl").read_text().splitlines() if l.strip()]
    ret_rows = [r for r in rows if r.get("gt_retention")]
    n, nret = len(rows), len(ret_rows)
    if not nret: return dict(n=n, ret=0.0)
    eps = [e for e, _ in rungs_of(ret_rows[0]["q_ladder"])]
    surv = {e: 0 for e in eps}
    ff = {e: 0 for e in eps}; ff["none"] = 0     # first-failure rung histogram
    for r in ret_rows:
        rr = rungs_of(r["q_ladder"])
        for e, v in rr:
            if v: surv[e] += 1
        fail = next((e for e, v in rr if not v), None)
        ff[fail if fail is not None else "none"] += 1
    fails = [r for r in rows if not r.get("solved")]
    viol = float(np.median([r["violations"] for r in fails])) if fails else None
    cc = float(np.median([r["cells_correct"] for r in fails])) if fails else None
    bw = float(np.median([r.get("multi_init_best_wrong", -1) for r in fails])) if fails else None
    leak = sum(1 for r in rows if r.get("retained_per_step") and not all(r["retained_per_step"]))
    return dict(n=n, ret=nret / n, eps=eps,
                S={e: surv[e] / nret for e in eps}, ff=ff,
                viol=viol, cells=cc, bw=bw, leak=leak)

say("=" * 116)
say("GRADED LADDER RETRO-ANALYSIS (corrected dict reader; 2026-08-29) — all banked Sudoku probes; sportBr2b EXCLUDED (live)")
say("=" * 116)
say("per arm: retention | TRUE S(eps) of retained | first-failure histogram (rung: count | none=survived all) | med viol/cells@fail | leak")
for label, pat in GROUPS:
    say(f"\n== {label} ==")
    for d in sorted(RUNS.glob(pat)):
        if "sportBr2b" in d.name: continue
        if not (d / "results.jsonl").exists(): continue
        a = analyze_dir(d)
        arm = d.name.replace("sudprobe_p", "").replace("sudprobe_", "")
        if a.get("ret", 0) == 0:
            say(f"  {arm:14s} n={a['n']} retention 0.000 (no retained set)"); continue
        scurve = "/".join(f"{a['S'][e]:.2f}" for e in a["eps"])
        ffs = " ".join(f"{e}:{a['ff'][e]}" for e in a["eps"] if a["ff"][e]) + f" none:{a['ff']['none']}"
        say(f"  {arm:14s} ret {a['ret']:.3f} | S({'/'.join(str(e) for e in a['eps'])}) {scurve}"
            f" | ff {ffs} | viol {a['viol']:.0f} cells {a['cells']:.0f} bw {a['bw']:.0f} | leak {a['leak']}")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(L) + "\n")
print(f"\nartifact -> {OUT}")
