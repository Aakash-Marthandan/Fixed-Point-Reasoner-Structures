# Instrument verification for tools/analyze_sport.py (same discipline as
# test_analyze_r0_rules.py): synthesize S-port rows whose verdicts are known
# by construction and check each registered rule fires as written.
#
# The scenario deliberately encodes the outcome the difficulty ladder exists
# to expose: the SAME substrate collapses the retention/solve dissociation at
# an EASY level (50 givens) while the dissociation PERSISTS at a HARD one
# (30 givens). If the analyzer pooled levels it would report one muddled
# ratio and miss that entirely — this test would then fail.
import json, os, pathlib, random, subprocess, sys

SCRATCH = pathlib.Path(sys.argv[1])
random.seed(11)

# (level, solve_rate, retention_rate, mi_hit_rate_with_basin)
SPEC = {50: (.60, .70, .30), 40: (.25, .65, .10), 30: (.02, .60, .02)}
RI_BONUS = .18          # sudB solves more — the H-33-ii dividend


def rows(arm):
    out = []
    for lv, (sr, rr, mi) in SPEC.items():
        for i in range(40):
            solved = random.random() < (sr + (RI_BONUS if arm == "sudB" and sr > .01 else 0))
            ret = solved or random.random() < rr
            hits = 0
            if not solved:
                # capture only where a basin exists -> enrichment is infinite
                hits = sum(random.random() < mi for _ in range(16)) if ret else 0
            out.append({
                "task": f"g{lv}_{i:04d}", "givens_level": lv,
                "solved": solved, "gt_retention": ret,
                "retained_per_step": [ret] * 8,
                "q_ladder": {"0.05": ret, "0.1": ret, "0.2": ret, "0.4": solved},
                "violations": 0 if solved else random.randint(5, 60),
                "cells_correct": 81 if solved else random.randint(30, 70),
                "givens_kept": lv, "givens_total": lv,
                "multi_init_hits": hits, "multi_init_k": 16,
                "multi_init_best_wrong": 0 if hits else 9,
                "n_givens": lv, "I_s": [340.0, 70.0, 45.0, 17.0, 24.0],
                "wall_s": 1.0})
    return out


for arm in ("sudA", "sudB"):
    d = SCRATCH / f"sudprobe_{arm}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "results.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows(arm)) + "\n")

env = dict(os.environ, QHRRN_RUNS=str(SCRATCH))
out = subprocess.run([".venv/bin/python", "tools/analyze_sport.py"], env=env,
                     capture_output=True, text=True).stdout
print(out)
checks = {
    "per-level table (3 levels x 2 arms)": out.count("sudA ") >= 3 and out.count("sudB ") >= 3,
    "S-R1 collapses at the EASY level": "CONFIRMED (collapses)" in out,
    "S-R1 dissociation persists at a HARD level":
        ("KILL — dissociation is architectural" in out or "ratio infinite" in out),
    "S-R1 never VOIDs (retention is high here)": "VOID" not in out,
    "S-R2 finds the RI dividend": "CONFIRMED: RI's dividend" in out,
    "S-R2 has no dangling 'needs both arms'": "(needs both arms)" not in out,
    "S-R3 reports enrichment": ("enrichment" in out),
}
print("=" * 64)
for k, v in checks.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}")
sys.exit(0 if all(checks.values()) else 1)
