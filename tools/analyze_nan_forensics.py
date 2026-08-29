# Ledger: NaN EARLY-WARNING CASE-CONTROL STUDY ($0, disk-only; 2026-08-29, lens C
# of the retro-measurement program, PI go "A and C tonight"). Question: does ANY
# logged quantity carry a precursor of the free-arm state explosions at the
# 50-step (train-row) or 2k (monitor) resolution — or is the cliff genuinely
# unheralded (H-48's operating assumption: prevention must be structural)?
# CASES (pre-death windows; labeled provenance):
#   E3 C1@5e-4   — GCS live tail to 22,100 incl. the first explosion row (death ~22.1k)
#   E4 C1s1@5e-4 — GCS live tail to 11,950 (death ~12.2k; last ~250 steps unsynced)
#   E5 D1@1e-3   — GCS live tail (2b OPS-FORENSICS material, already in the ops
#                  record; no 2b EVAL data touched) to ~10,750 (death ~10.8k)
#   E1/E2 (first NaNs at 1e-3) — windows LOST to the registered rollback
#   truncations; steps known (11.5k / 8.3k), trajectories unavailable. Labeled.
# CONTROLS: every completed free/priced arm at d64/d96 (B1,B2,B3,C2,C3,C4) —
#   same features on every window of every control trajectory.
# METHOD: trailing-window features (W=1000 steps, stride 250): A_total log-slope,
#   I_total log-slope, loss std, loss max-jump; the case value at the LAST
#   pre-death window ranked against the pooled control-window distribution AND
#   the case run's own earlier windows (case-crossover). Monitor-side: lam_joint
#   max at the last pre-death monitor vs excursion base rates (frac of monitors
#   with lamJ > 2) on cases vs controls.
"""
  .venv/bin/python tools/analyze_nan_forensics.py  # -> runs/analysis/nan_forensics_20260829.txt
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
SP = Path("/private/tmp/claude-501/-Users-aakash-Projects-HRRN/6d10cb0e-61cd-4b13-9fe1-5c0a08f49979/scratchpad/forensics")
OUT = RUNS / "analysis" / "nan_forensics_20260829.txt"
L = []
def say(s=""): L.append(str(s)); print(s)

def load(path):
    tr, mon = {}, {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line: continue
        try: r = json.loads(line)
        except Exception: continue
        if "monitor" in r: mon[r["monitor"]["step"]] = r["monitor"]
        elif "step" in r and "loss" in r: tr[r["step"]] = r
    steps = sorted(tr)
    return steps, tr, dict(sorted(mon.items()))

def windows(steps, tr, W=1000, stride=250, drop_nonfinite=True):
    """Trailing-window features; exclude any window containing a non-finite row."""
    out = []
    arr = [(s, tr[s]["loss"], tr[s].get("A_total"), tr[s].get("I_total")) for s in steps]
    for end_i in range(len(arr)):
        e = arr[end_i][0]
        w = [x for x in arr if e - W < x[0] <= e]
        if len(w) < W // 100: continue
        if (arr[end_i][0] - arr[0][0]) < W: continue
        if e % stride: continue
        loss = np.array([x[1] for x in w], float)
        A = np.array([x[2] for x in w], float)
        I = np.array([x[3] for x in w], float)
        if drop_nonfinite and not (np.isfinite(loss).all() and np.isfinite(A).all()): continue
        xs = np.arange(len(w))
        def lslope(v):
            v = np.maximum(v, 1e-9)
            return float(np.polyfit(xs, np.log(v), 1)[0]) * len(w)   # log-change per window
        out.append(dict(end=e, a_sl=lslope(A), i_sl=lslope(I),
                        l_std=float(loss.std()), l_jump=float(np.max(np.abs(np.diff(loss))))))
    return out

CASES = [
    ("E3 C1@5e-4  (death ~22.1k)", SP / "C1_metrics_live.jsonl", 22100),
    ("E4 C1s1@5e-4 (death ~12.2k)", SP / "C1s1_metrics_live.jsonl", 11950),
    ("E5 D1@1e-3  (death ~10.8k; 2b ops-forensics, labeled)", SP / "D1_metrics_live.jsonl", 10750),
]
CONTROLS = [
    ("B1-d64", RUNS / "pretrainsportB_B1" / "metrics.jsonl"),
    ("B2-d64", RUNS / "pretrainsportB_B2" / "metrics.jsonl"),
    ("B3-d64", RUNS / "pretrainsportB_B3" / "metrics.jsonl"),
    ("C2-d96", RUNS / "pretrainsportBr2_C2" / "metrics.jsonl"),
    ("C3-d96", RUNS / "pretrainsportBr2_C3" / "metrics.jsonl"),
    ("C4-d96", RUNS / "pretrainsportBr2_C4" / "metrics.jsonl"),
]
FEATS = ("a_sl", "i_sl", "l_std", "l_jump")

say("=" * 112)
say("NaN EARLY-WARNING CASE-CONTROL (2026-08-29) — 3 usable death windows vs 6 survivor trajectories")
say("=" * 112)
ctrl = []
for name, p in CONTROLS:
    if not p.exists(): say(f"  control {name}: missing"); continue
    steps, tr, mon = load(p)
    ws = windows(steps, tr)
    ctrl.extend(ws)
    lamx = [m.get("lam_joint_max") for m in mon.values() if m.get("lam_joint_max") is not None]
    frac2 = np.mean([x > 2 for x in lamx]) if lamx else float("nan")
    say(f"  control {name}: {len(ws)} windows | monitors {len(lamx)} | frac(lamJ>2) {frac2:.2f} | maxLamJ {max(lamx) if lamx else float('nan'):.2f}")
say(f"  pooled control windows: {len(ctrl)}")
cd = {f: np.array([w[f] for w in ctrl]) for f in FEATS}

say("\nCASES — the last PRE-DEATH window vs (i) pooled controls, (ii) the run's own earlier windows:")
for name, p, death in CASES:
    if not p.exists(): say(f"  {name}: not staged"); continue
    steps, tr, mon = load(p)
    ws = windows(steps, tr)                      # non-finite rows excluded -> last window = pre-death
    if not ws: say(f"  {name}: no windows"); continue
    last = ws[-1]; own = ws[:-1]
    say(f"  {name}: last clean window ends @{last['end']} ({death - last['end']} steps before death); {len(own)} own earlier windows")
    for f in FEATS:
        pool_pct = 100 * np.mean(cd[f] < last[f])
        own_pct = 100 * np.mean([w[f] < last[f] for w in own]) if own else float("nan")
        say(f"    {f:6s} = {last[f]:+.4g} | percentile vs controls {pool_pct:5.1f} | vs own history {own_pct:5.1f}")
    lam_items = [(s, m.get("lam_joint_max")) for s, m in mon.items() if m.get("lam_joint_max") is not None]
    if lam_items:
        s_last, lam_last = lam_items[-1]
        frac2 = np.mean([x > 2 for _, x in lam_items])
        say(f"    lamJ @ last monitor (step {s_last}): {lam_last:.2f} | own frac(lamJ>2) {frac2:.2f}")

say("\nMONITOR-SIDE BASE RATES (specificity check): does lamJ>2 discriminate death runs from survivors?")
say("  (survivor rates above; note the d64 z-mode caveat — B2's late lamJ frac-expansive 1.0 at retfm 1.00 was already ruled benign)")
say("\nE1/E2 (first NaNs at lr 1e-3, steps ~11.5k / ~8.3k): trajectories LOST to the registered rollback truncations — labeled unavailable.")
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(L) + "\n")
print(f"\nartifact -> {OUT}")
