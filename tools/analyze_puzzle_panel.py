# Ledger: PER-PUZZLE PAIRED PANEL ($0, disk-only; 2026-08-29, lens A of the
# retro-measurement program, PI go "A and C tonight"; banked campaigns ONLY —
# sportBr2b excluded). Every full-test eval banked per-puzzle records on the
# IDENTICAL 422,786-puzzle set; every 20k scan on the identical seeded subsample.
# This panel joins them by puzzle idx and delivers:
#  1. PAIRED McNemar for the decision-relevant cross-arm/cross-scale contrasts —
#     the measurement-law upgrade (within-set pairing) applied retroactively.
#  2. THE HARD CORE: puzzles no arm ever solves cold (the vacancy-floor analog),
#     by rating octile; solve-multiplicity distribution.
#  3. FRONTIER/REGRESSION sets: per-puzzle solvability flips across scale.
#  4. PORTFOLIO DECORRELATION: same-scale overlap/Jaccard + union-vs-best.
#  5. Rating-vs-model-difficulty consistency (does tdoku rating order OUR hard?).
#  6. Breadth-paired: vote@128 hit bits on the shared 20k subsample (C3-d96 vs
#     B2-d64 vs S5-d16) — G-R2-1's contrast as a paired test.
"""
  .venv/bin/python tools/analyze_puzzle_panel.py  # -> runs/analysis/puzzle_panel_20260830.txt
"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "analysis" / "puzzle_panel_20260830.txt"
L = []
def say(s=""): L.append(str(s)); print(s)

def load_records(d):
    """Concatenate records shards (or records_all), sort by idx; return dict of arrays or None."""
    p = RUNS / d
    parts = sorted(p.glob("records_s*.npz")) or ([p / "records_all.npz"] if (p / "records_all.npz").exists() else [])
    if not parts: return None
    cols = {}
    for q in parts:
        z = np.load(q, allow_pickle=True)
        if "idx" not in z.files or "cold_exact" not in z.files: return None
        for k in ("idx", "rating", "cold_exact", "mi_first_hit"):
            if k in z.files: cols.setdefault(k, []).append(np.asarray(z[k]))
    out = {k: np.concatenate(v) for k, v in cols.items()}
    order = np.argsort(out["idx"], kind="stable")
    return {k: v[order] for k, v in out.items()}

FULLS = [  # (label, dir, scale-group)
    ("S5-d16-20k",    "sxeval_psport2S5/full_t64",    "d16"),
    ("S4-d16-T24",    "sxeval_psport2S4/full_t64",    "d16"),
    ("W2-d16-30k",    "sxeval_psport2w2W2/full_t64",  "d16"),
    ("A2-d16",        "sxeval_psport3aA2/full_t64",   "d16"),
    ("A3-d16-FPA",    "sxeval_psport3aA3/full_t64",   "d16"),
    ("A4s1-d16-rec",  "sxeval_psport3aA4s1/full_t64", "d16"),
    ("A5-d16-priced", "sxeval_psport3aA5/full_t64",   "d16"),
    ("A7s1-d16",      "sxeval_psport3aA7s1/full_t64", "d16"),
    ("B1-d64",        "sxeval_psportBB1/full_t64",    "d64"),
    ("B2-d64-FPA",    "sxeval_psportBB2/full_t64",    "d64"),
    ("B3-d64-priced", "sxeval_psportBB3/full_t64",    "d64"),
    ("B4-d64-broken", "sxeval_psportBB4/full_t64",    "d64"),
    ("C1-d96-FPA20k", "sxeval_psportBr2C1/full_t64",  "d96"),
    ("C1s1-d96-10k",  "sxeval_psportBr2C1s1/full_t64","d96"),
    ("C2-d96-priced", "sxeval_psportBr2C2/full_t64",  "d96"),
    ("C3-d96-T6FPA",  "sxeval_psportBr2C3/full_t64",  "d96"),
    ("C4-d96-beta3",  "sxeval_psportBr2C4/full_t64",  "d96"),
    ("D1-d96-stop10k","sxeval_psportBr2bD1/full_t64", "d96"),
    ("D2-d96-T6s1",   "sxeval_psportBr2bD2/full_t64", "d96"),
    ("D3-d96-dose5e4","sxeval_psportBr2bD3/full_t64", "d96"),
    ("D4-d96-dose1e3","sxeval_psportBr2bD4/full_t64", "d96"),
    ("C3X-d96-cont",  "sxeval_psportBr2bC3X/full_t64","d96"),
]

say("=" * 118)
say("PER-PUZZLE PAIRED PANEL (2026-08-29) — banked full-test records, idx-aligned; sportBr2b EXCLUDED (live)")
say("=" * 118)

panel, ratings, ref_idx = {}, None, None
for label, d, grp in FULLS:
    r = load_records(d)
    if r is None: say(f"  [census] {label}: no per-puzzle records (pre-panel evaluator) — skipped"); continue
    if ref_idx is None:
        ref_idx = r["idx"]; ratings = r["rating"]
    if len(r["idx"]) != len(ref_idx) or not np.array_equal(r["idx"], ref_idx):
        say(f"  [census] {label}: idx mismatch (n={len(r['idx'])}) — skipped"); continue
    panel[label] = (np.asarray(r["cold_exact"], dtype=bool), grp)
say(f"\npanel: {len(panel)} arms x {len(ref_idx)} puzzles (idx-aligned, identical set)")

def mcnemar(a, b):
    x, y = panel[a][0], panel[b][0]
    b01 = int(np.sum(x & ~y)); b10 = int(np.sum(~x & y))
    n = b01 + b10
    if n == 0: return b01, b10, 1.0
    z = (abs(b01 - b10) - 1) / math.sqrt(n)
    pz = math.erfc(z / math.sqrt(2))
    return b01, b10, pz

say("\n1. PAIRED McNEMAR (cold, full test; a-only vs b-only discordant counts; the measurement-law upgrade)")
CONTRASTS = [
    ("C1-d96-FPA20k", "C3-d96-T6FPA",  "d96 deep-partial vs shallow-winner"),
    ("C1-d96-FPA20k", "C2-d96-priced", "d96 free-deep-partial vs priced"),
    ("C3-d96-T6FPA",  "C4-d96-beta3",  "d96 winner vs beta/3"),
    ("C1-d96-FPA20k", "C1s1-d96-10k",  "d96 stopped pair (20k vs 10k)"),
    ("B2-d64-FPA",    "C1-d96-FPA20k", "d64 record vs d96 censored carrier"),
    ("B2-d64-FPA",    "C3-d96-T6FPA",  "d64 record vs d96 winner"),
    ("B3-d64-priced", "C2-d96-priced", "priced across scale d64->d96"),
    ("B2-d64-FPA",    "B1-d64",        "d64 FPA vs RI/NI(half-lr)"),
    ("B2-d64-FPA",    "B3-d64-priced", "d64 free vs priced"),
    ("A4s1-d16-rec",  "B2-d64-FPA",    "d16 record vs d64 record"),
    ("A3-d16-FPA",    "A5-d16-priced", "d16 FPA vs priced"),
    ("S5-d16-20k",    "C3-d96-T6FPA",  "the T6 lineage d16->d96"),
]
for a, b, why in CONTRASTS:
    if a not in panel or b not in panel: continue
    b01, b10, p = mcnemar(a, b)
    ca, cb = panel[a][0].mean(), panel[b][0].mean()
    say(f"  {a:15s} {100*ca:5.2f} vs {b:15s} {100*cb:5.2f} | only-a {b01:6d} only-b {b10:6d} | p {'<1e-300' if p == 0 else f'{p:.2e}'} | {why}")

say("\n2. THE HARD CORE (cold; across all panel arms) + solve multiplicity")
mat = np.stack([panel[k][0] for k in panel])
mult = mat.sum(axis=0)
hard = mult == 0
say(f"  never-solved-cold by ANY of {len(panel)} arms: {hard.sum()} / {len(ref_idx)} ({100*hard.mean():.2f} %)")
qs = np.quantile(ratings, np.linspace(0, 1, 9))
say("  hard-core fraction by rating octile: " + " ".join(
    f"[{qs[i]:.0f}-{qs[i+1]:.0f}]:{100*hard[(ratings>=qs[i])&((ratings<=qs[i+1]) if i==7 else (ratings<qs[i+1]))].mean():.1f}%"
    for i in range(8)))
say("  solve-multiplicity histogram (arms solving each puzzle): " + " ".join(
    f"{k}:{int((mult==k).sum())}" for k in range(0, len(panel)+1) if (mult==k).sum()))

say("\n3. FRONTIER / REGRESSION across scale (union per scale group)")
grp_union = {}
for g in ("d16", "d64", "d96"):
    arms = [k for k in panel if panel[k][1] == g]
    grp_union[g] = np.any(np.stack([panel[k][0] for k in arms]), axis=0)
    say(f"  {g}: union-cold over {len(arms)} arms = {100*grp_union[g].mean():5.2f} % (best single {100*max(panel[k][0].mean() for k in arms):5.2f})")
new96 = grp_union["d96"] & ~grp_union["d64"] & ~grp_union["d16"]
lost96 = (grp_union["d16"] | grp_union["d64"]) & ~grp_union["d96"]
new64 = grp_union["d64"] & ~grp_union["d16"]
say(f"  FRONTIER: solved first at d64: {new64.sum()} | solved first at d96: {new96.sum()} | solved at d16/d64 but NOT d96: {lost96.sum()}")
say("  new-at-d96 by octile: " + " ".join(
    f"{100*new96[(ratings>=qs[i])&((ratings<=qs[i+1]) if i==7 else (ratings<qs[i+1]))].mean():.1f}%" for i in range(8)))

say("\n4. PORTFOLIO DECORRELATION (same-scale pairwise Jaccard of cold solve sets)")
for g in ("d16", "d64", "d96"):
    arms = [k for k in panel if panel[k][1] == g]
    for i in range(len(arms)):
        for j in range(i + 1, len(arms)):
            x, y = panel[arms[i]][0], panel[arms[j]][0]
            jac = np.sum(x & y) / max(np.sum(x | y), 1)
            say(f"  {g} {arms[i]:15s} ~ {arms[j]:15s} J={jac:.3f}")

say("\n5. RATING-vs-MODEL-DIFFICULTY consistency")
from numpy import corrcoef
rk = np.argsort(np.argsort(ratings))
mk = np.argsort(np.argsort(-mult.astype(float)))
rho = corrcoef(rk, mk)[0, 1]
say(f"  Spearman(tdoku rating, model-difficulty[-multiplicity]) = {rho:.3f} (n={len(ref_idx)})")
easy_for_us = (ratings >= qs[6]) & (mult >= max(1, int(0.8 * len(panel))))
hard_for_us = (ratings <= qs[1]) & (mult == 0)
say(f"  anomalies: high-rating (top quartile) yet solved by >=80% of arms: {easy_for_us.sum()} | rating<=octile-1 yet NEVER solved: {hard_for_us.sum()}")

say("\n6. BREADTH-PAIRED (vote@128 hit bits, shared 20k subsample)")
b20 = {}
for label, d in [("C3-d96", "sxbreadth20k_psportBr2C3"), ("B2-d64", "sxbreadth20k_psportBB2"), ("S5-d16", "sxbreadth20000_S5_k128")]:
    r = load_records(d)
    if r is None: say(f"  {label}: no records"); continue
    b20[label] = (r["idx"], np.asarray(r["mi_first_hit"]) >= 0)
if len(b20) >= 2:
    ks = list(b20)
    ref = b20[ks[0]][0]
    if all(np.array_equal(b20[k][0], ref) for k in ks[1:]):
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                x, y = b20[ks[i]][1], b20[ks[j]][1]
                b01 = int(np.sum(x & ~y)); b10 = int(np.sum(~x & y)); n = b01 + b10
                z = (abs(b01 - b10) - 1) / math.sqrt(max(n, 1))
                p = math.erfc(z / math.sqrt(2))
                say(f"  vote@128 {ks[i]} {100*x.mean():5.2f} vs {ks[j]} {100*y.mean():5.2f} | only-a {b01} only-b {b10} | p {'<1e-300' if p == 0 else f'{p:.2e}'}")
    else:
        say("  subsample idx mismatch — paired test refused (report only)")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(L) + "\n")
print(f"\nartifact -> {OUT}")
