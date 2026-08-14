# Instrument verification for tools/analyze_r0.py: synthesize cells with
# KNOWN ground truth and check each registered rule fires as written.
# Scenario built in: floors WIN the deep tail (R1 should confirm two-sided
# S2), NI is non-inferior + drops wrong-stable (R2 -> default-in), priced
# beats plain on rg-96 (R3 -> Law-4 stands). If the analyzer reports
# anything else, the rule logic is wrong — not the data.
import json, os, pathlib, random, subprocess, sys

SCRATCH = pathlib.Path(sys.argv[1])
random.seed(7)
VH = [f"ca_T{i}" for i in range(48)]
RG = [f"rg_{i}" for i in range(48)]
RB = [f"rb_{i}" for i in range(48)]

def write(dirname, rows):
    d = SCRATCH / dirname
    d.mkdir(parents=True, exist_ok=True)
    (d / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

def qrow(ret, s4, s2=None):
    s2 = ret if s2 is None else s2
    return {"exact_T": False, "gt_retention": ret,
            "retained_per_step": [ret] * 8,
            "q_ladder": {"0.05": ret, "0.1": ret, "0.2": s2, "0.4": s4},
            "I_s": [340.0, 70.0, 45.0, 17.0, 24.0], "A_s": [0.0] * 5}

# rates chosen so the intended verdicts are unambiguous
RATE = {  # (vh_ret, vh_s4, rg_ret)
    "A1": (.30, .22, .14), "A2": (.30, .11, .13),   # floors win deep tail
    "A3": (.28, .10, .06),                           # plain: worst transfer
    "A4": (.30, .21, .14),                           # NI ~ A1
}
for arm, (rr, r4, rg) in RATE.items():
    for s in (0, 1, 2):
        tag = f"{arm}s{s}"
        write(f"lad_p13f{tag}", [{"task": t, "sel_step": 50, "wall_s": 1.0,
              "supports": [], "queries": [qrow(random.random() < rr,
                                               random.random() < r4)
                                          for _ in range(3)]} for t in VH])
        for pref, ids in (("ladrg", RG), ("ladrgb", RB)):
            write(f"{pref}_p13f{tag}", [{"task": t, "sel_step": 50, "wall_s": 1.0,
                  "supports": [], "queries": [qrow(random.random() < rg,
                                                   False)
                                              for _ in range(3)]} for t in ids])
        if arm in ("A1", "A4"):
            ws_rate = .70 if arm == "A1" else .55      # NI drops wrong-stable
            write(f"e1e3_p13f{tag}", [{"task": t, "sel_step": 50, "wall_s": 1.0,
                  "e1": [], "e3": [{"converged_at": 3, "n_distinct": 2,
                                    "limit_exact": random.random() > ws_rate,
                                    "exact_per_step": [], "H_q_per_step": []}
                                   for _ in range(3)], "e3b": []} for t in VH])

env = dict(os.environ, QHRRN_RUNS=str(SCRATCH))
out = subprocess.run([".venv/bin/python", "tools/analyze_r0.py"], env=env,
                     capture_output=True, text=True).stdout
print(out)
checks = {
    "R1 confirms floors": "two-sided S2 CONFIRMED" in out,
    "R2 NI default-in": "NI DEFAULT-IN" in out,
    "R3 Law-4 stands": "SEEDED AT THE FRONTIER" in out,
    "12/12 vh cells": "val-hard cells: 12/12" in out,
    "rg-96 288 pairs (no integrity warning)": "!!" not in out,
}
print("=" * 60)
for k, v in checks.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}")
sys.exit(0 if all(checks.values()) else 1)
