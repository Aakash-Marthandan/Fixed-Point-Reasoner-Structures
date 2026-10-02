#!/usr/bin/env python3
# Ledger: DISCUSSION-PERIOD EXPERIMENT P11 (registration Documentation/Note_2026-10-01_Rebuttal_P10_P11_Registration.md, written before
# any row). MEASUREMENT, $0, the Mac's CPU, inference only. P9's identical-answer state swaps (tools/rebuttal_p9.py, run UNCHANGED) at
# further post-50k MLP 192 checkpoints, chosen before any row as the lowest saved fixed-start endpoints after 94k (90k, 114k; 118k if
# time). The wrapper only sets P9's module constants (STEP, SAVED, OUT) and applies P11's registered pattern rule to P9's report.
"""  JAX_PLATFORMS=cpu .venv/bin/python tools/rebuttal_p11.py --selftest
  JAX_PLATFORMS=cpu nice -n 10 .venv/bin/python tools/rebuttal_p11.py --run 90000 114000 [118000]
  .venv/bin/python tools/rebuttal_p11.py --report 90000 114000 [118000]"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tools"))
BASE = ROOT / "runs/analysis/rebuttal_20261001i"

def pattern(rep):
    """P11 rule (ii) on a P9 report.json: AND-failure / OR-rescue pattern of 94k."""
    if not rep.get("gates", {}).get("ok", False) or len(rep["meta"]["primary"]) < 20: return "UNDEFINED"
    r, t = rep["rescue"], rep["transfer"]
    ok = r["perp"] >= 0.8 and r["l"] >= 0.6 and t["perp"] <= 0.2 and t["l"] <= 0.2 and t["z"] >= 0.8
    return "PATTERN-REPLICATES" if ok else "PATTERN-DIFFERS"

def configure(step):
    import rebuttal_p9 as P9
    P9.STEP = int(step); P9.SAVED = ROOT / f"paper/code/evidence/initialization/s{int(step):06d}.npz"; P9.OUT = BASE / f"s{int(step):06d}"
    return P9

def selftest():
    good = dict(gates=dict(ok=True), meta=dict(primary={str(i): 1 for i in range(25)}),
                rescue=dict(perp=1.0, h=1.0, l=0.88, z=1.0), transfer=dict(perp=0.0, h=0.12, l=0.0, z=1.0))
    assert pattern(good) == "PATTERN-REPLICATES"
    assert pattern(dict(good, rescue=dict(good["rescue"], perp=0.79))) == "PATTERN-DIFFERS"
    assert pattern(dict(good, transfer=dict(good["transfer"], l=0.21))) == "PATTERN-DIFFERS"
    assert pattern(dict(good, meta=dict(primary={str(i): 1 for i in range(19)}))) == "UNDEFINED"
    assert pattern(dict(good, gates=dict(ok=False))) == "UNDEFINED"
    P9 = configure(90000); assert P9.STEP == 90000 and P9.SAVED.name == "s090000.npz" and P9.OUT.name == "s090000"
    print("selftest OK (6 checks)")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true"); ap.add_argument("--run", nargs="*"); ap.add_argument("--report", nargs="*")
    a = ap.parse_args()
    if a.selftest: selftest()
    for step in (a.run or []):
        P9 = configure(step)
        if (P9.OUT / "p9.npz").exists(): print(f"SKIP {step} (done)", flush=True); continue
        P9.run(); P9.report(out=P9.OUT)
    if a.report:
        lines = []
        for step in a.report:
            p = BASE / f"s{int(step):06d}" / "report.json"
            if not p.exists(): lines.append(f"{step}: not run"); continue
            rep = json.loads(p.read_text())
            lines.append(f"{step}: primary {len(rep['meta']['primary'])}, controls {len(rep['meta']['controls'])}; gates {rep['gates']}; "
                         f"rescue {rep['rescue']}; transfer {rep['transfer']}; P9 letter {rep['letter']}; P11 rule (ii): {pattern(rep)}")
        BASE.mkdir(parents=True, exist_ok=True); (BASE / "report.txt").write_text("\n".join(lines) + "\n"); print("\n".join(lines))

if __name__ == "__main__":
    main()
