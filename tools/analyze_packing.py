# Ledger: cluster N (packing frontier, freethink 2026-08-12) — every substrate
# with a val-hard ladder placed in the (codebook size N, code distance r) plane;
# empirical Pareto frontier; packing-exchange-rate fit n_eff; the registered
# ordering test (priced nearer the frontier than free at matched N; count
# movers move ALONG it) and the kill check (frontier proximity must not
# anti-correlate with rg transfer). Reads disk only.
"""
  python tools/analyze_packing.py        # -> runs/analysis/packing_frontier_*.txt
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

RUNS = Path(__file__).resolve().parents[1] / "runs"
OUT = RUNS / "analysis" / "packing_frontier_20260812.txt"
EPS = [0.05, 0.1, 0.2, 0.4]

# tag -> (d, group, note); group: priced | free | dials | depth (T6, priced)
INV = {
    "p8":        (16, "free",  "p8 orbit-corpus"),
    "p9a":       (16, "free",  "p9 seed1"),
    "p9b":       (16, "free",  "p9 seed2"),
    "p9c":       (16, "priced", "p9 knee"),
    "p9d":       (16, "dials", "eta-floor+z"),
    "p10a":      (16, "free",  "C20 seed0"),
    "p10b":      (16, "free",  "C20 seed1"),
    "p10c":      (16, "priced", "C20 knee"),
    "p10d":      (16, "priced", "C20 DOSE 1e-4"),
    "p1124p":    (24, "free",  ""),
    "p1124c":    (24, "priced", "s0"), "p1124cs1": (24, "priced", "s1"),
    "p1124cs2":  (24, "priced", "s2"),
    "p1132p":    (32, "free",  "s0"), "p1132ps1": (32, "free", "s1"),
    "p1132c":    (32, "priced", "s0"), "p1132cs1": (32, "priced", "s1"),
    "p1132cs2":  (32, "priced", "s2"),
    "p1132cT6":  (32, "depth", "T6 s0"), "p1132cT6s1": (32, "depth", "T6 s1"),
    "p1148p":    (48, "free",  "s0"), "p1148ps1": (48, "free", "s1"),
    "p1148c":    (48, "priced", "s0 20k"), "p1148cs1": (48, "priced", "s1 20k"),
    "p1248c40k": (48, "priced", "40k DP"),
}


def h_nats(rho: float, q: int = 10) -> float:
    """Per-cell log-volume of a Hamming ball at flip fraction rho (q-ary)."""
    if rho <= 0:
        return 0.0
    rho = min(rho, 1 - 1e-9)
    return float(-rho * np.log(rho) - (1 - rho) * np.log(1 - rho)
                 + rho * np.log(q - 1))


def load(tag: str, prefix: str = "lad"):
    p = RUNS / f"{prefix}_{tag}" / "results.jsonl"
    if not p.exists():
        return None
    N = s2 = 0
    radii = []
    for line in p.read_text().splitlines():
        for q in json.loads(line)["queries"]:
            if not q["gt_retention"]:
                continue
            N += 1
            s2 += q["q_ladder"]["0.2"]
            surv = [e for e in EPS if q["q_ladder"][str(e)]]
            radii.append(max(surv) if surv else 0.0)
    return dict(N=N, rbar=float(np.mean(radii)) if radii else 0.0,
                rad=s2 / max(N, 1))


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    if len(x) < 3:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    lines = []
    say = lambda s="": (print(s), lines.append(s))
    rows = []
    for tag, (d, grp, note) in INV.items():
        c = load(tag)
        if c is None:
            continue
        rg = load(tag, "ladrg") or {}
        rows.append(dict(tag=tag, d=d, grp=grp, note=note, **c,
                         rg_N=rg.get("N"), rg_rad=rg.get("rad")))
    # Pareto frontier in (N, rbar), maximizing both
    for r in rows:
        r["front"] = not any(o["N"] >= r["N"] and o["rbar"] >= r["rbar"]
                             and (o["N"] > r["N"] or o["rbar"] > r["rbar"])
                             for o in rows)
    # normalized distance to nearest frontier point
    Ns = np.array([r["N"] for r in rows], float)
    Rs = np.array([r["rbar"] for r in rows], float)
    nN, nR = Ns.max() - Ns.min() or 1, Rs.max() - Rs.min() or 1
    fr = [r for r in rows if r["front"]]
    for r in rows:
        r["dist"] = 0.0 if r["front"] else min(
            float(np.hypot((f["N"] - r["N"]) / nN, (f["rbar"] - r["rbar"]) / nR))
            for f in fr if f["N"] >= r["N"] or f["rbar"] >= r["rbar"])
    say("=" * 86)
    say("CLUSTER N — PACKING PLANE: N = val-hard retention count /144, "
        "rbar = mean max-survived eps")
    say("=" * 86)
    say(f'{"tag":>11s} {"d":>3s} {"group":>7s} {"N":>4s} {"rbar":>5s} '
        f'{"S.2/S0":>6s} {"front":>5s} {"dist":>5s} {"rg_N":>5s} {"rg_rad":>6s}')
    for r in sorted(rows, key=lambda r: (-r["front"], r["dist"])):
        say(f'{r["tag"]:>11s} {r["d"]:3d} {r["grp"]:>7s} {r["N"]:4d} '
            f'{r["rbar"]:5.2f} {r["rad"]:6.2f} {"*" if r["front"] else "":>5s} '
            f'{r["dist"]:5.2f} '
            f'{r["rg_N"] if r["rg_N"] is not None else "-":>5} '
            f'{f"{r0:.2f}" if (r0 := r["rg_rad"]) is not None else "-":>6s}')
    say()
    # exchange-rate fit along the frontier: log N = C - n_eff * h(0.9*rbar)
    fN = np.log([r["N"] for r in fr])
    fh = np.array([h_nats(0.9 * r["rbar"]) for r in fr])
    if len(fr) >= 3 and np.ptp(fh) > 0:
        A = np.vstack([np.ones_like(fh), -fh]).T
        (C, n_eff), *_ = np.linalg.lstsq(A, fN, rcond=None)
        say(f"FRONTIER EXCHANGE RATE: log N = {C:.2f} - n_eff*h(rho), "
            f"n_eff = {n_eff:.1f} cells")
        say("  (h = per-cell q-ary ball-volume exponent, rho = 0.9*rbar; "
            "n_eff ~ cells of Hamming")
        say("   volume the codebook trades per basin — compare to true-extent "
            "~100s of cells)")
    say()
    for grp in ("priced", "free", "dials", "depth"):
        ds = [r["dist"] for r in rows if r["grp"] == grp]
        ns = [r["N"] for r in rows if r["grp"] == grp]
        if ds:
            say(f"  {grp:>7s}: median frontier-dist {np.median(ds):.3f}  "
                f"(n={len(ds)}, median N {np.median(ns):.0f})")
    say()
    # kill check: frontier proximity vs rg transfer (lower dist should NOT
    # mean worse transfer)
    have = [r for r in rows if r["rg_rad"] is not None and r["rg_N"] is not None]
    sp_rad = spearman([-r["dist"] for r in have], [r["rg_rad"] for r in have])
    sp_ret = spearman([-r["dist"] for r in have], [r["rg_N"] for r in have])
    say(f"KILL CHECK (n={len(have)}): Spearman(frontier-proximity, rg_rad) = "
        f"{sp_rad:+.2f}; (proximity, rg_N) = {sp_ret:+.2f}")
    say("  kill fires iff proximity anti-correlates with transfer (both < 0)")
    OUT.write_text("\n".join(lines) + "\n")
    print(f"\n[written] {OUT}")


if __name__ == "__main__":
    main()
