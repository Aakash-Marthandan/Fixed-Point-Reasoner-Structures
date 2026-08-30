# Ledger: IGNITION DYNAMICS LENS ($0, disk-only; 2026-08-29, lens D+E of the
# retro-measurement program — the training-strategy question the PI prioritized
# after lens B: WHAT CHANGES INSIDE THE NETWORK WHEN A FUNNEL IGNITES (rho
# expands) OR CONDENSES? Banked campaigns ONLY — sportBr2b excluded.)
# Known event windows from multi-ckpt screens (rung-2 verdict artifacts):
#   C3-d96  IGNITION between 10k and 15k (screens 29.7 -> 71.7 -> 88.5)
#   C1-d96  OSCILLATION 10k wide -> 15k condensed -> 20k re-opened (x-screens)
#   B3-d64  CONDENSATION between 25k and 50k (71.3 -> 27.0)
#   C2-d96  condensed-STATIONARY at every screen (30.5/33.4/33.0)
#   C4-d96  monotone GROWTH (27.2 -> 33.6 -> 43.0)
# Per 5k-grid window and per arm: (a) per-block relative parameter displacement
# ||dTheta||/||Theta||; (b) Fisher top-set ROTATION (Jaccard of the top-1% of
# Adam-v between consecutive grids — does the important-parameter set drift?);
# (c) trajectory signals in the window (eta delta, rule_H mean, fp_drift).
# Question: does any cheap signal single out the ignition/condensation windows?
"""
  .venv/bin/python tools/analyze_ignition_dynamics.py  # -> runs/analysis/ignition_dynamics_20260830.txt
"""
from __future__ import annotations
import json, pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "analysis" / "ignition_dynamics_20260830.txt"
L = []
def say(s=""): L.append(str(s)); print(s)

def flatd(t, pre=""):
    out = {}
    if isinstance(t, dict):
        for k, v in t.items(): out.update(flatd(v, f"{pre}/{k}"))
    elif hasattr(t, "shape"): out[pre] = np.asarray(t)
    return out
def find_nu(t):
    if hasattr(t, "_fields") and "nu" in getattr(t, "_fields", ()): return t.nu
    if isinstance(t, (list, tuple)):
        for x in t:
            r = find_nu(x)
            if r is not None: return r
    return None
def block_of(k):
    parts = [x for x in k.split("/") if x]
    return parts[1] if len(parts) > 1 else parts[0]

def load_ck(p):
    with open(p, "rb") as fh: c = pickle.load(fh)
    return c

def arm_windows(d, events=""):
    cks = sorted(Path(d).glob("ckpt_0*.pkl"))
    if len(cks) < 2: return None
    say(f"\n== {Path(d).name} {('[' + events + ']') if events else ''} ==")
    say("   window      | rel-disp total | top blocks of displacement      | Fisher-rot J(top1%) | d_eta   | rule_H  | fp_drift")
    # trajectory signals
    mp = Path(d) / "metrics.jsonl"
    tr, mons = {}, {}
    if mp.exists():
        for line in mp.read_text().splitlines():
            line = line.strip()
            if not line: continue
            try: r = json.loads(line)
            except Exception: continue
            if "monitor" in r: mons[r["monitor"]["step"]] = r["monitor"]
            elif "step" in r: tr[r["step"]] = r
    prev = load_ck(cks[0]); prev_step = prev["step"]
    pf = flatd(prev.get("state", {}).get("model", prev.get("state", {})))
    pnu = flatd(find_nu(prev.get("opt_state")) or {})
    ptop = None
    if pnu:
        allv = np.concatenate([v.ravel() for v in pnu.values()])
        thr = np.quantile(allv, 0.99)
        ptop = {k: (v > thr) for k, v in pnu.items()}
    for ck in cks[1:]:
        cur = load_ck(ck); step = cur["step"]
        cf = flatd(cur.get("state", {}).get("model", cur.get("state", {})))
        num = sum(float(((cf[k] - pf[k]) ** 2).sum()) for k in cf if k in pf)
        den = sum(float((pf[k] ** 2).sum()) for k in pf)
        rel = (num / max(den, 1e-12)) ** 0.5
        bl = {}
        for k in cf:
            if k not in pf: continue
            b = block_of(k)
            bl[b] = bl.get(b, 0.0) + float(((cf[k] - pf[k]) ** 2).sum())
        tot = sum(bl.values()) or 1.0
        top = sorted(bl.items(), key=lambda x: -x[1])[:2]
        tops = ", ".join(f"{b}={v/tot:.0%}" for b, v in top)
        # Fisher rotation
        rot = None
        cnu = flatd(find_nu(cur.get("opt_state")) or {})
        if cnu and ptop:
            allv = np.concatenate([v.ravel() for v in cnu.values()])
            thr = np.quantile(allv, 0.99)
            ctop = {k: (v > thr) for k, v in cnu.items()}
            inter = sum(int(np.sum(ptop[k] & ctop[k])) for k in ctop if k in ptop)
            union = sum(int(np.sum(ptop[k] | ctop[k])) for k in ctop if k in ptop)
            rot = inter / max(union, 1)
            ptop = ctop
        # trajectory signals in (prev_step, step]
        et0 = [m.get("eta") for s, m in sorted(mons.items()) if s <= prev_step and m.get("eta") is not None]
        et1 = [m.get("eta") for s, m in sorted(mons.items()) if s <= step and m.get("eta") is not None]
        deta = (et1[-1] - et0[-1]) if (et0 and et1) else None
        rh = [r.get("rule_H") for s, r in tr.items() if prev_step < s <= step and r.get("rule_H") is not None]
        rhm = float(np.mean(rh)) if rh else None
        fpd = [m.get("fp_drift_max") for s, m in mons.items() if prev_step < s <= step and m.get("fp_drift_max") is not None]
        fpm = float(np.max(fpd)) if fpd else None
        say(f"  {prev_step:>6}->{step:<6} | {rel:14.4f} | {tops:31s} | {('  -  ' if rot is None else f'{rot:.3f}'):>19s}"
            f" | {('  -  ' if deta is None else f'{deta:+.3f}'):>7s} | {('  -  ' if rhm is None else f'{rhm:.3f}'):>7s} | {('  -  ' if fpm is None else f'{fpm:.2e}')}")
        prev, prev_step, pf = cur, step, cf
    return True

say("=" * 118)
say("IGNITION DYNAMICS (2026-08-29) — per-5k-window parameter displacement, Fisher rotation, trajectory signals; sportBr2b EXCLUDED")
say("=" * 118)
ARMS = [
    ("runs/pretrainsportBr2_C3",   "IGNITION 10k->15k (screens 29.7->71.7->88.5)"),
    ("runs/pretrainsportBr2_C1",   "OSCILLATION 10k/15k/20k (50.4->27.2->71.3); STOPPED"),
    ("runs/pretrainsportBr2_C1s1", "ignited by 10k (21.5@5k->76.2@10k); STOPPED"),
    ("runs/pretrainsportBr2_C2",   "condensed-STATIONARY at all screens"),
    ("runs/pretrainsportBr2_C4",   "monotone GROWTH 27->34->43"),
    ("runs/pretrainsportB_B2",     "d64 carrier; mid-25k rate-limited -> final"),
    ("runs/pretrainsportB_B3",     "d64 priced; CONDENSATION 25k->50k (71->27)"),
    ("runs/pretrainsportB_B4",     "d64 broken (retfm .34)"),
    ("runs/pretrainsportBr2b_D2",  "2b seed pair; EARLY ignition (73.2@10k->84.8@vb)"),
    ("runs/pretrainsportBr2b_D3",  "2b dosed T12 5e-4; LATE ignition 25k->40k (51->93)"),
    ("runs/pretrainsportBr2b_D4",  "2b dosed T12 1e-3; ignition 68->79->89, rising@50k"),
    ("runs/pretrainsportBr2b_C3X", "2b continuation; 81.5@+10k->91@+20k->94.5@+30k"),
    ("runs/pretrainsportBr2b_D1",  "2b T6@50k STOPPED@10k (retfm .88 break; screen 29.5)"),
]
for d, ev in ARMS:
    if Path(d).exists(): arm_windows(d, ev)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(L) + "\n")
print(f"\nartifact -> {OUT}")
