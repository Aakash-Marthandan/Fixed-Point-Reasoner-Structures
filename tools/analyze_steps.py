# Ledger: the steps(d) calibration (registered 2026-08-12, discriminating-cell
# verdict entry) — fit the training-budget law for basin radius from the
# priced scale line (d16..d48 @ 20k) + the compute-matched pair (d48 @ 40k).
# Output: the 5-75M allocation rule + the d64 pilot step count, with the
# one-sided sufficiency logic stated explicitly. Reads disk only.
"""
  python tools/analyze_steps.py          # table + law -> runs/analysis/
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

RUNS = Path(__file__).resolve().parents[1] / "runs"
OUT = RUNS / "analysis" / "steps_calibration_20260812.txt"

# priced scale line: (d, steps, [ladder tags])
CELLS = [
    (16, 20_000, ["p10c"]),
    (24, 20_000, ["p1124c", "p1124cs1", "p1124cs2"]),
    (32, 20_000, ["p1132c", "p1132cs1", "p1132cs2"]),
    (48, 20_000, ["p1148c", "p1148cs1"]),
    (48, 40_000, ["p1248c40k"]),
]
BAND = 0.30  # pre-registered plateau-band threshold (2026-08-12 rule)


def rg_radius(tag: str):
    p = RUNS / f"ladrg_{tag}" / "results.jsonl"
    if not p.exists():
        return None
    ret = s2 = 0
    for line in p.read_text().splitlines():
        for q in json.loads(line)["queries"]:
            ret += q["gt_retention"]
            s2 += q["q_ladder"]["0.2"]
    return dict(ret=ret, rad=s2 / max(ret, 1))


def main():
    lines = []
    say = lambda s="": (print(s), lines.append(s))
    say("=" * 78)
    say("steps(d) CALIBRATION — priced scale line, rg-48 transfer radius S(.2)/S(0)")
    say("=" * 78)
    say(f"{'d':>4s} {'steps':>7s} {'seeds':>6s} {'rg_rad (per-seed)':>24s} "
        f"{'mean':>6s}  {'>= band .30?':>12s}")
    rows = []
    for d, steps, tags in CELLS:
        cells = [(t, rg_radius(t)) for t in tags]
        cells = [(t, c) for t, c in cells if c]
        rads = [c["rad"] for _, c in cells]
        m = float(np.mean(rads)) if rads else float("nan")
        rows.append((d, steps, m, len(rads)))
        per = " ".join(f"{r:.2f}" for r in rads)
        say(f"{d:4d} {steps:7d} {len(rads):6d} {per:>24s} {m:6.2f}  "
            f"{'YES' if m >= BAND else 'no':>12s}")
    say()
    say("SUFFICIENCY CONSTRAINTS on steps*(d) = min steps to reach the band:")
    say("  steps*(24) <= 20k, steps*(32) <= 20k (band reached at 20k, n=3 each)")
    say("  20k < steps*(48) <= 40k (the discriminating pair: .18 mean -> .33)")
    say()
    say("LINEAR LAW steps* = c*d:   consistent c in (417, 625] steps per unit d")
    say("  (c>417 from d48@20k insufficient; c<=625 from d32@20k sufficient)")
    say("  tightest-consistent allocation: steps(d) = 625*d")
    say("    d48 -> 30k (untested; 40k is the measured-sufficient point)")
    say("    d64 -> 40k   d96 -> 60k   d128 -> 80k")
    say("QUADRATIC LAW steps* = c2*d^2:  consistent c2 in (8.7, 19.5]")
    say("  NOT excluded by the bracket; at d64 -> 36-80k, d96 -> 80-180k")
    say()
    say("PILOT ALLOCATION (d64), decision:")
    say("  d48-proportional (anchor the measured-sufficient point 40k@48):")
    say("    steps(64) = 40k * 64/48 = 53.3k  <- RECOMMENDED (safe under the")
    say("    linear branch, mid-bracket under the quadratic branch; ~2x pod")
    say("    margin costs ~minutes)")
    say("  REGISTERED READOUT: pilot rg-radius >= .30 => 53k sufficient at d64")
    say("    (law bracket holds); < .30 => linear branch DEAD at d64, the")
    say("    quadratic branch promoted and the 5-75M budget re-derived.")
    say()
    say("Caveats (honest): n=1 at (48, 40k); band from d24/d32 seed-means")
    say("  .34/.37; steps* is bracketed one-sidedly (sufficiency observed only")
    say("  at 20k/40k grid points — no cell between). Slope datum at d48:")
    say(f"  +{rows[4][2] - rows[3][2]:.2f} radius per step-doubling (below saturation).")
    OUT.write_text("\n".join(lines) + "\n")
    print(f"\n[written] {OUT}")


if __name__ == "__main__":
    main()
