# Ledger: cluster P analyzer (registration 2026-08-12) — aggregates
# probe_aniso rows into the retention-anisotropy spectrum: per (subspace,
# rho), the retention-survival fraction relative to baseline; the >=10x
# separation test (mixers-class vs boundary-class at matched rho) and the
# near-isotropy kill check. Reads disk only.
"""
  python tools/analyze_aniso.py runs/aniso_p1248c40k [more dirs...]
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

RUNS = Path(__file__).resolve().parents[1] / "runs"
SPINE = ("mixers", "attn", "film")          # the [H-12] erosion class
FLAT_PRED = ("boundary", "gates", "e_t")    # predicted low-sensitivity


def load(d: Path):
    rows = [json.loads(l) for l in (d / "results.jsonl").read_text().splitlines()]
    base = 0
    cells = {}  # (sub, rho) -> [kept, base_total]
    for r in rows:
        nb = sum(r["base_ret"])
        base += nb
        for c in r["cells"]:
            k = (c["sub"], c["rho"])
            kept = sum(a and b for a, b in zip(c["ret"], r["base_ret"]))
            cc = cells.setdefault(k, [0, 0])
            cc[0] += kept
            cc[1] += nb * 1  # per-dir normalization handled below
    return rows, base, cells


def main():
    dirs = [Path(a) for a in sys.argv[1:]] or [RUNS / "aniso_p1248c40k"]
    for d in dirs:
        if not (d / "results.jsonl").exists():
            print(f"missing {d}")
            continue
        rows, base, cells = load(d)
        subs = sorted({s for s, _ in cells}, key=lambda s: s)
        rhos = sorted({r for _, r in cells})
        print("=" * 74)
        print(f"CLUSTER P — RETENTION ANISOTROPY: {d.name} "
              f"({len(rows)} tasks, baseline retained {base})")
        print("=" * 74)
        print(f'{"subspace":>10s} ' + " ".join(f"rho{r:<5g}" for r in rhos)
              + "   (survival fraction of baseline-retained pairs)")
        surv = {}
        for s in subs:
            vals = []
            for r in rhos:
                kept, tot = cells.get((s, r), (0, 0))
                v = kept / tot if tot else float("nan")
                surv[(s, r)] = v
                vals.append(v)
            print(f'{s:>10s} ' + " ".join(f"{v:8.3f}" for v in vals))
        print()
        # separation at each rho: destruction(spine) / destruction(flat)
        for r in rhos:
            spine_d = np.mean([1 - surv[(s, r)] for s in SPINE if (s, r) in surv])
            flat_d = np.mean([1 - surv[(s, r)] for s in FLAT_PRED if (s, r) in surv])
            sep = spine_d / max(flat_d, 1e-9) if flat_d or spine_d else float("nan")
            print(f"rho {r}: spine destruction {spine_d:.3f} vs "
                  f"flat-class {flat_d:.3f} -> separation "
                  f"{'inf' if flat_d == 0 and spine_d > 0 else f'{sep:.1f}x'}")
        # kill check: near-isotropy = all subspaces within 2x at every rho
        iso = all(
            (mx := max(1 - surv[(s, r)] for s in subs if (s, r) in surv)) <=
            2 * max(min(1 - surv[(s, r)] for s in subs if (s, r) in surv), 1e-9)
            or mx == 0
            for r in rhos)
        print(f"KILL CHECK (near-isotropy): {'FIRES' if iso else 'does not fire'}")
        print()


if __name__ == "__main__":
    main()
