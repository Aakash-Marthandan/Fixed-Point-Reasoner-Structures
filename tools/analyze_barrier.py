# Ledger: cluster Q analyzer (barrier spectroscopy, registration 2026-08-12)
# — merges the 8 pod shards, computes per-(pair, T0) hop rates (exact capture
# on det-failed pairs; within-radius as soft variant), fits Arrhenius
# ln(rate) = ln(A) - dF/T0 per task family where >=3 temps give nonzero
# rates, and runs the registered prediction check (barrier height vs
# snap-convertibility) + the kill check (rate flat in T0). Reads disk only.
"""
  python tools/analyze_barrier.py [--base runs/barrier_p1248c40k]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

RUNS = Path(__file__).resolve().parents[1] / "runs"
# C.3' task-level conversions + att1-exact tasks (ledger 2026-08-09/10) —
# the registered snap-convertibility set for the prediction check
CONVERTED = {"ca_Center2", "ca_InsideOutside4"}


def family(tid: str) -> str:
    """ca_AboveBelow5 -> AboveBelow (ConceptARC concept-group prefix)."""
    stem = tid[3:] if tid.startswith("ca_") else tid
    return stem.rstrip("0123456789")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(RUNS / "barrier_p1248c40k"))
    ap.add_argument("--out", default=str(RUNS / "analysis" /
                                         "barrier_spectroscopy_20260812.txt"))
    a = ap.parse_args()
    rows = []
    for shard in sorted(Path(a.base).parent.glob(Path(a.base).name + "_s*")):
        p = shard / "results.jsonl"
        if p.exists():
            rows += [json.loads(l) for l in p.read_text().splitlines()]
    lines = []
    say = lambda s="": (print(s), lines.append(s))
    say("=" * 80)
    say(f"CLUSTER Q — BARRIER SPECTROSCOPY ({len(rows)} tasks merged)")
    say("=" * 80)
    if not rows:
        say("no rows"); return
    temps = sorted(float(t) for t in rows[0]["queries"][0]["sigmas"])
    say(f"T0 grid: {temps}")
    # per-family hop-rate tables over det-failed pairs
    fam_counts = {}   # fam -> T0 -> [hops, trials]
    per_task = {}
    for r in rows:
        fam = family(r["task"])
        for q in r["queries"]:
            if q["det_dist"] == 0.0:
                continue  # det-solved: no barrier to hop
            for t in temps:
                rec = q["sigmas"][str(t)]
                k = len(rec["dists"])
                hops = sum(1 for d in rec["dists"] if d == 0.0)
                c = fam_counts.setdefault(fam, {}).setdefault(t, [0, 0])
                c[0] += hops; c[1] += k
                pt = per_task.setdefault(r["task"], {}).setdefault(t, [0, 0])
                pt[0] += hops; pt[1] += k
    say()
    say(f'{"family":>16s} ' + " ".join(f"T{t:<5g}" for t in temps)
        + "   dF (Arrhenius)   n_pairs")
    fits = {}
    for fam in sorted(fam_counts):
        rates = []
        for t in temps:
            h, n = fam_counts[fam].get(t, (0, 0))
            rates.append(h / n if n else 0.0)
        nz = [(t, r) for t, r in zip(temps, rates) if r > 0]
        dF = ""
        if len(nz) >= 3:
            x = np.array([1.0 / t for t, _ in nz])
            y = np.log(np.array([r for _, r in nz]))
            slope, _ = np.polyfit(x, y, 1)
            fits[fam] = -slope
            dF = f"{-slope:8.3f}"
        n_pairs = fam_counts[fam][temps[0]][1] // max(
            len(rows[0]["queries"][0]["sigmas"][str(temps[0])]["dists"]), 1)
        say(f'{fam:>16s} ' + " ".join(f"{r:6.3f}" for r in rates)
            + f"   {dF:>14s}   {n_pairs:7d}")
    say()
    # aggregate rate vs T (kill check: flat?)
    agg = []
    for t in temps:
        h = sum(c.get(t, (0, 0))[0] for c in fam_counts.values())
        n = sum(c.get(t, (0, 0))[1] for c in fam_counts.values())
        agg.append(h / n if n else 0.0)
    say("AGGREGATE hop rate by T0: " +
        "  ".join(f"{t}:{r:.4f}" for t, r in zip(temps, agg)))
    span = (max(agg) / max(min([r for r in agg if r > 0], default=1e-9), 1e-9)
            if any(agg) else 0)
    say(f"  dynamic range (max/min-nonzero): {span:.1f}x "
        f"{'— KILL FIRES (flat)' if span < 2 and any(agg) else ''}")
    if not any(agg):
        say("  ZERO hops anywhere — barriers above the accessible window "
            "(kill branch: sampler-design datum)")
    say()
    # prediction: converted tasks' families sit low in dF
    if fits:
        conv_fams = {family(t) for t in CONVERTED}
        say("PREDICTION CHECK (converted families low-barrier):")
        ranked = sorted(fits.items(), key=lambda kv: kv[1])
        for fam, dF in ranked:
            mark = " <-- converted" if fam in conv_fams else ""
            say(f"  {fam:>16s} dF={dF:.3f}{mark}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(lines) + "\n")
    print(f"\n[written] {a.out}")


if __name__ == "__main__":
    main()
