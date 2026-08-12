# Ledger: cluster O (spectral shape universality, freethink 2026-08-12) —
# normalized stationary spectra I_hat(s) = med I_s / sum over the priced-knee
# substrates: is the information distribution across RG cuts a d-INVARIANT
# scaling function? Free arms + the dose arm (beta 1e-4) as contrasts; the
# collapse metric is per-scale CV across substrates within group + max
# pairwise L1 between normalized profiles. Reads disk only.
"""
  python tools/analyze_spectra.py        # -> runs/analysis/spectral_collapse_*.txt
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

RUNS = Path(__file__).resolve().parents[1] / "runs"
OUT = RUNS / "analysis" / "spectral_collapse_20260812.txt"

# tag -> (d, group, note); groups: knee (beta 3e-5/1e-5), dose, free
INV = [
    ("p9c",       16, "knee", "orbit corpus"),
    ("p10c",      16, "knee", "C20"),
    ("p1124c",    24, "knee", "s0"), ("p1124cs1", 24, "knee", "s1"),
    ("p1124cs2",  24, "knee", "s2"),
    ("p1132c",    32, "knee", "s0"), ("p1132cs1", 32, "knee", "s1"),
    ("p1132cs2",  32, "knee", "s2"),
    ("p1132cT6",  32, "knee", "T6 s0"), ("p1132cT6s1", 32, "knee", "T6 s1"),
    ("p1148c",    48, "knee", "s0 20k"), ("p1148cs1", 48, "knee", "s1 20k"),
    ("p1248c40k", 48, "knee", "40k DP"),
    ("p10d",      16, "dose", "beta 1e-4"),
    ("p8",        16, "free", ""), ("p9a", 16, "free", ""),
    ("p9b",       16, "free", ""), ("p9d", 16, "free", "dials"),
    ("p10a",      16, "free", ""), ("p10b", 16, "free", ""),
    ("p1124p",    24, "free", ""),
    ("p1132p",    32, "free", "s0"), ("p1132ps1", 32, "free", "s1"),
    ("p1148p",    48, "free", "s0"), ("p1148ps1", 48, "free", "s1"),
]


def spectrum(tag: str):
    p = RUNS / f"lad_{tag}" / "results.jsonl"
    if not p.exists():
        return None
    I = []
    for line in p.read_text().splitlines():
        for q in json.loads(line)["queries"]:
            I.append(q["I_s"])
    med = np.median(np.array(I), axis=0)
    return med


def main():
    lines = []
    say = lambda s="": (print(s), lines.append(s))
    say("=" * 84)
    say("CLUSTER O — NORMALIZED STATIONARY SPECTRA I_hat(s) (componentwise "
        "median over 144 val-hard queries)")
    say("=" * 84)
    say(f'{"tag":>11s} {"d":>3s} {"grp":>5s} {"I_tot":>8s}  '
        f'{"I_hat(0..4)":>44s}')
    prof = {}
    for tag, d, grp, note in INV:
        med = spectrum(tag)
        if med is None:
            continue
        tot = med.sum()
        ih = med / tot
        prof[tag] = (d, grp, ih, tot)
        say(f'{tag:>11s} {d:3d} {grp:>5s} {tot:8.0f}  '
            + " ".join(f"{v:7.3f}" for v in ih) + f"   {note}")
    say()
    for grp in ("knee", "free"):
        mats = np.array([v[2] for v in prof.values() if v[1] == grp])
        if len(mats) < 2:
            continue
        cv = mats.std(axis=0) / np.maximum(mats.mean(axis=0), 1e-9)
        l1 = max(float(np.abs(a - b).sum())
                 for i, a in enumerate(mats) for b in mats[i + 1:])
        say(f"{grp.upper()} group (n={len(mats)}): per-scale CV = "
            + " ".join(f"{v:.2f}" for v in cv)
            + f" | mean CV {cv.mean():.3f} | max pairwise L1 {l1:.3f}")
    say()
    # d-trend of the UV share within each group (free steepening check)
    say("UV share I_hat(0) by d (group means):")
    for grp in ("knee", "free"):
        by_d = {}
        for tag, (d, g, ih, tot) in prof.items():
            if g == grp:
                by_d.setdefault(d, []).append(ih[0])
        say(f"  {grp:>5s}: " + "  ".join(
            f"d{d}={np.mean(v):.3f}" for d, v in sorted(by_d.items())))
    say()
    # throat context: I_tot by d, knee group (the 40k datum labeled)
    say("Throat I_tot (median) context, knee group:")
    for tag, (d, g, ih, tot) in prof.items():
        if g == "knee":
            say(f"  {tag:>11s} d{d:<3d} {tot:7.0f}")
    say()
    say("READING KEYS: collapse = knee-group mean CV small and max L1 small")
    say("  vs the free group's; kill = no collapse at matched beta.")
    OUT.write_text("\n".join(lines) + "\n")
    print(f"\n[written] {OUT}")


if __name__ == "__main__":
    main()
