#!/usr/bin/env python3
"""Rung-2b physics pass (written AT ANALYSIS TIME, 2026-08-30; descriptive only).

Reads the extracted sportBr2b corpus in runs/: monitor trajectories (last-wins
dedup by step), train-row A/I flux (the dose-efficacy story), screen k-curves,
and the paired per-puzzle McNemar contrasts the registration's questions need.
No decision rules live here — the analyzer (untouched) is the verdict authority.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "analysis" / "sportBr2b_physics_20260830.txt"
L = []
def say(s=""): L.append(str(s)); print(s)

ARMS = ["D1", "D2", "D3", "D4", "C3X"]

def monitor_rows(arm):
    mp = RUNS / f"pretrainsportBr2b_{arm}" / "metrics.jsonl"
    if not mp.exists(): return {}
    mons, train = {}, {}
    for line in mp.read_text().splitlines():
        try: j = json.loads(line)
        except Exception: continue
        if "monitor" in j:
            row = j["monitor"]
            mons[int(row["step"])] = row                 # last-wins dedup (step lives in the monitor dict)
        elif "step" in j:
            train[int(j["step"])] = j                    # last-wins dedup
    return {"mon": dict(sorted(mons.items())), "train": dict(sorted(train.items()))}

say("=" * 112)
say("RUNG-2B PHYSICS PASS (2026-08-30, analysis-time, descriptive) — dose efficacy, trajectories, paired contrasts")
say("=" * 112)

say("\n== A. monitor trajectories (last-wins dedup) ==")
say("arm | monitors | retfm min/final | ret_sched min/final | eta final | lamJ max/final | lam_y max/final")
traj = {}
for a in ARMS:
    d = monitor_rows(a); traj[a] = d
    m = d.get("mon", {})
    if not m: say(f"  {a}: no monitor rows"); continue
    def col(k):
        v = [row.get(k) for row in m.values() if row.get(k) is not None]
        return v
    retfm, rsch = col("ret_final_t8"), col("ret_sched_t8")
    eta, lj, ly = col("eta"), col("lam_joint_max"), col("lam_max_max")
    fm = lambda v, f: (f % v if v is not None else "-")
    say(f"  {a:4s} n={len(m):3d} | retfm {min(retfm):.2f}/{retfm[-1]:.2f} | "
        f"ret_sched {min(rsch):.2f}/{rsch[-1]:.2f} | eta {eta[-1]:.3f} | "
        f"lamJ {max(lj):.2f}/{lj[-1]:.2f} | lam_y {max(ly):.2f}/{ly[-1]:.2f}"
        if retfm and eta and lj and ly else f"  {a}: partial keys {list(next(iter(m.values())).keys())[:8]}")
    if a == "D1" and retfm:
        bad = [(s, r.get("ret_final_t8")) for s, r in m.items() if r.get("ret_final_t8") is not None and r.get("ret_final_t8") < 0.9]
        say(f"       D1 sub-.9 retfm monitors: {bad}")

say("\n== B. dose efficacy — train-row flux trajectories (A_total = the priced channel; I_total = free streams) ==")
say("arm | A: max / at-2k / at-10k / final | I: final | (C1-free ref: A ~1.1e7)")
for a in ARMS:
    t = traj[a].get("train", {})
    if not t: continue
    steps = sorted(t)
    def at(kstep, key):
        cand = [s for s in steps if s <= kstep]
        return t[cand[-1]].get(key) if cand else None
    A = [(s, t[s].get("A_total")) for s in steps if t[s].get("A_total") is not None]
    I = [(s, t[s].get("I_total")) for s in steps if t[s].get("I_total") is not None]
    if not A: say(f"  {a}: no A rows (keys: {list(t[steps[0]].keys())[:10]})"); continue
    Av = [v for _, v in A]
    say(f"  {a:4s} A max {max(Av):9.1f} | @2k {at(2000,'A_total')} | @10k {at(10000,'A_total')} | final {Av[-1]:8.2f} | I final {I[-1][1]:.3e}")

say("\n== C. screen k-curves (vote@16 vs @256 — funnel flatness class) ==")
for a in ARMS:
    for kind in ("vb", "s040000", "s020000", "s015000", "s010000", "s025000"):
        sa = RUNS / f"sxscreen_psportBr2b{a}_{kind}" / "summary_all.json"
        if not sa.exists(): continue
        j = json.loads(sa.read_text())
        v = j.get("vote_at_k", {})
        v16, v256 = v.get("16"), v.get("256")
        if v16 and v256:
            say(f"  {a:4s} {kind:8s} v16 {100*float(v16):5.2f} v256 {100*float(v256):5.2f} ratio {float(v256)/float(v16):.2f}")

say("\n== D. paired per-puzzle McNemar (full-test cold; idx-aligned) ==")
def cold(dirname):
    p = RUNS / dirname / "full_t64"
    parts = sorted(p.glob("records_s*.npz")) or [p / "records_all.npz"]
    idx, ce = [], []
    for q in parts:
        if not q.exists(): return None
        z = np.load(q, allow_pickle=True)
        idx.append(np.asarray(z["idx"])); ce.append(np.asarray(z["cold_exact"]).astype(bool))
    idx, ce = np.concatenate(idx), np.concatenate(ce)
    o = np.argsort(idx, kind="stable")
    return idx[o], ce[o]

PAIRS = [
    ("D4 (33.53, dosed lr1e-3)", "sxeval_psportBr2bD4", "B2-d64 record (25.27)", "sxeval_psportBB2"),
    ("D4", "sxeval_psportBr2bD4", "D3 (dosed lr5e-4; seed confound NAMED)", "sxeval_psportBr2bD3"),
    ("D4", "sxeval_psportBr2bD4", "C3X (continuation)", "sxeval_psportBr2bC3X"),
    ("D2 (seed1 of C3 recipe)", "sxeval_psportBr2bD2", "C3-d96 (seed0, rung-2)", "sxeval_psportBr2C3"),
    ("C3X (continuation +30k)", "sxeval_psportBr2bC3X", "C3 (its own 20k init)", "sxeval_psportBr2C3"),
    ("D3", "sxeval_psportBr2bD3", "C1-d96@20k (censored free twin)", "sxeval_psportBr2C1"),
]
for la, da, lb, db in PAIRS:
    A, B = cold(da), cold(db)
    if A is None or B is None: say(f"  {la} vs {lb}: MISSING"); continue
    if not np.array_equal(A[0], B[0]): say(f"  {la} vs {lb}: idx mismatch"); continue
    a, b = A[1], B[1]
    oa, ob = int((a & ~b).sum()), int((~a & b).sum())
    n = min(oa, ob); m = oa + ob
    p = stats.binomtest(n, m, 0.5).pvalue if m else 1.0
    say(f"  {la:34s} vs {lb:34s} | only-a {oa:6d} only-b {ob:6d} | p {p:.3g}")

say("\n== E. eta / lambda registered-prediction checks (from A-section values) ==")
say("  D3 predictions: lamJ max <= 2.5; eta -> [.80,.90]; A<=1e5 by 10k  (read rows above)")
say("  (adjudication lives in the verdict/report; this pass is descriptive)")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(L) + "\n")
print(f"\nartifact -> {OUT}")
