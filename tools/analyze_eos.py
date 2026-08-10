# Ledger: cluster J (Research_Brainstorm 2026-08-10) — basin equation-of-state,
# stage 1: everything recomputable from saved batteries. S(eps) = retained-pair
# count at corruption eps (eps=0 == GT-oracle retention, the e3b instrument).
# dS/dbeta at eps=0 from P9-C vs P9-A/B; dS/deta from P9-D vs P9-A/B; per-step
# decay curves; arm-pair overlap (same-pairs vs different-pairs decomposition
# of the priced/dials gains). Stage 2 (ladder battery) extends eps>0 to all
# substrates; this script consumes its output when present.
"""
  python tools/analyze_eos.py --bat-root <dir with runs/bat_*, runs/pc_*, runs/c2_*>
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path

ARMS = {  # label -> (path fragment, fit protocol note)
    "old_b4@2000": ("pc_b4_2000", "old pretrain6_d16, full-fit@2000"),
    "p8_keyhole": ("pc_e1e3_p8d16", "pretrain8_d16, e_t-only@600 (Phase-C battery)"),
    "p8_full@2000(C.2)": ("c2_p8d16", "pretrain8_d16, full-fit@2000 (eroded)"),
    "p9a_s1": ("bat_p9a", "pretrain9_a seed1 plain, e_t-only@600"),
    "p9b_s2": ("bat_p9b", "pretrain9_b seed2 plain, e_t-only@600"),
    "p9c_priced": ("bat_p9c", "pretrain9_c beta_flux 3e-5 + nl 1e-5, e_t-only@600"),
    "p9d_dials": ("bat_p9d", "pretrain9_d eta_floor .2 + z_gate .3, e_t-only@600"),
}


def load(root: Path, frag: str):
    p = root / "runs" / frag / "results.jsonl"
    if not p.exists():
        return None
    return [json.loads(l) for l in p.read_text().splitlines()]


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def battery_retention(rows):
    """e3b-format batteries: per-pair gt_oracle retained_all + per-step curve."""
    pairs, per_step = [], None
    for r in rows:
        for qi, q in enumerate(r.get("e3b", [])):
            g = q.get("gt_oracle", {})
            steps = g.get("retained_per_step", [])
            pairs.append(((r["task"], qi), bool(g.get("retained_all", False)), steps))
            if per_step is None and steps:
                per_step = [0] * len(steps)
    if per_step is not None:
        for _, _, steps in pairs:
            for i, s in enumerate(steps):
                if i < len(per_step) and s:
                    per_step[i] += 1
    return pairs, per_step


def c2_ladder(rows, side="queries"):
    """c2-format: q_ladder / loo_ladder at eps in {.05,.1,.2,.4}; gt_retention = eps=0."""
    eps_keys = ["0.05", "0.1", "0.2", "0.4"]
    counts = {"0.0": 0}
    n = 0
    for r in rows:
        if side == "queries":
            for q in r.get("queries", []):
                n += 1
                if q.get("gt_retention"):
                    counts["0.0"] += 1
                lad = q.get("q_ladder")
                if lad:
                    for k in eps_keys:
                        counts[k] = counts.get(k, 0) + (1 if lad.get(k) else 0)
        else:
            n += 1
            if r.get("loo_retention_gt"):
                counts["0.0"] += 1
            lad = r.get("loo_ladder") or {}
            for k in eps_keys:
                counts[k] = counts.get(k, 0) + (1 if lad.get(k) else 0)
    return counts, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bat-root", default=".")
    a = ap.parse_args()
    root = Path(a.bat_root)

    print("=" * 72)
    print("J-1: BASIN EQUATION OF STATE — stage-1 (from saved batteries)")
    print("=" * 72)

    table = {}
    for label, (frag, note) in ARMS.items():
        rows = load(root, frag)
        if rows is None:
            print(f"  [missing] {label} ({frag})")
            continue
        if frag.startswith(("bat_", "pc_e1e3")):
            pairs, per_step = battery_retention(rows)
            k = sum(1 for _, r, _ in pairs if r)
            n = len(pairs)
            lo, hi = wilson(k, n)
            table[label] = {"pairs": {p for p, r, _ in pairs if r}, "k": k, "n": n}
            decay = " ".join(str(c) for c in (per_step or []))
            print(f"\n{label:22s} {note}")
            print(f"  S(0) retention: {k}/{n} = {k/n:.1%}  CI95 [{lo:.1%},{hi:.1%}]")
            print(f"  per-step retained counts (k=1..{len(per_step or [])}): {decay}")
        else:
            counts, n = c2_ladder(rows, side="queries")
            print(f"\n{label:22s} {note}")
            print(f"  query ladder (n={n}): " + "  ".join(
                f"eps={e}:{c}" for e, c in sorted(counts.items(), key=lambda t: float(t[0]))))
            lcounts, ln = c2_ladder(rows, side="loo")
            print(f"  LoO ladder   (n={ln}): " + "  ".join(
                f"eps={e}:{c}" for e, c in sorted(lcounts.items(), key=lambda t: float(t[0]))))
            table[label] = {"ladder": counts, "n": n}

    # old-arch loo ladder
    rows = load(root, "pc_b4_2000")
    if rows:
        lcounts, ln = c2_ladder(rows, side="loo")
        print(f"\n{'old_b4@2000':22s} LoO-side ladder (n={ln}): " + "  ".join(
            f"eps={e}:{c}" for e, c in sorted(lcounts.items(), key=lambda t: float(t[0]))))

    # pairwise decomposition among batteries
    bats = {k: v for k, v in table.items() if "pairs" in v}
    print("\n" + "=" * 72)
    print("ARM-PAIR DECOMPOSITION at eps=0 (same-pairs vs portfolio effect)")
    print("=" * 72)
    labels = list(bats)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            A, B = bats[labels[i]], bats[labels[j]]
            inter = len(A["pairs"] & B["pairs"])
            onlyA = len(A["pairs"] - B["pairs"])
            onlyB = len(B["pairs"] - A["pairs"])
            union = len(A["pairs"] | B["pairs"])
            print(f"  {labels[i]:14s} vs {labels[j]:14s}: "
                  f"both {inter:3d}  onlyL {onlyA:3d}  onlyR {onlyB:3d}  union {union:3d}")

    # derivative estimates at eps=0
    if all(k in bats for k in ("p9a_s1", "p9b_s2", "p9c_priced", "p9d_dials")):
        base = (bats["p9a_s1"]["k"] + bats["p9b_s2"]["k"]) / 2
        n = bats["p9a_s1"]["n"]
        seed_spread = abs(bats["p9a_s1"]["k"] - bats["p9b_s2"]["k"])
        print("\n" + "=" * 72)
        print("EOS DERIVATIVES at eps=0 (units: retained pairs /144; seed spread = noise floor)")
        print("=" * 72)
        print(f"  seed noise floor: |P9A-P9B| = {seed_spread} pairs")
        print(f"  dS/dbeta  (P9-C {bats['p9c_priced']['k']} vs seed-mean {base:.1f}): "
              f"{bats['p9c_priced']['k'] - base:+.1f} pairs at beta_flux=3e-5")
        print(f"  dS/d(eta,z) (P9-D {bats['p9d_dials']['k']} vs seed-mean {base:.1f}): "
              f"{bats['p9d_dials']['k'] - base:+.1f} pairs at eta_floor=.2, z_gate=.3")
        if "p8_keyhole" in bats:
            print(f"  reference p8 keyhole: {bats['p8_keyhole']['k']} "
                  f"(P9 plain-arm delta vs p8 = arch/data drift, same protocol)")


if __name__ == "__main__":
    main()
