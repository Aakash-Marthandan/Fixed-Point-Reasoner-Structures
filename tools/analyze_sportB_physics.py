# Ledger: PHASE B RUNG 1 PHYSICS PASS — written AT ANALYSIS TIME (2026-08-27),
# descriptive only (no registered rules; the verdict is analyze_sportB.py's).
# A. trajectories (metrics.jsonl, LAST-WINS dedup): eta / both lambda families /
#    retfm / val@t64 — endpoint stratification at d64, B4/B4s1 collapse ordering,
#    B3's late reorganization.
# B. funnel spectroscopy vs d (per-octile censored-geometric hit-rate, log-slope).
# C. probes: basin ladders S(eps), retention, multi-init, per-scale I_s at d64.
# D. Fisher-from-Adam-v: nu tree from ScaleByAdamState, per-block shares + PR.
"""
  .venv/bin/python tools/analyze_sportB_physics.py   # -> runs/analysis/sportB_r1_physics_20260827.txt
"""
from __future__ import annotations
import json, math, pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "analysis" / "sportB_r1_physics_20260827.txt"
ARMS = ["B1", "B2", "B3", "B4", "B4s1", "B5"]
L = []
def say(s=""): L.append(str(s)); print(s)
def fmt(x, nd=3): return "  -  " if x is None else f"{x:.{nd}f}"

# ---------- A. trajectories ----------
def load_rows(arm):
    p = RUNS / f"pretrainsportB_{arm}" / "metrics.jsonl"
    if not p.exists(): return {}
    rows = {}
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line: continue
        try: r = json.loads(line)
        except json.JSONDecodeError: continue
        s = r.get("step", r.get("monitor", {}).get("step") if isinstance(r.get("monitor"), dict)
                  else r.get("val", {}).get("step") if isinstance(r.get("val"), dict) else None)
        if s is None: continue
        key = ("mon" if "monitor" in r else "val" if "val" in r else "train", s)
        rows[key] = r                                   # LAST-WINS per (kind, step)
    return rows

say("=" * 104)
say("PHASE B RUNG 1 PHYSICS PASS (descriptive; 2026-08-27) — d64 full-width, Sudoku-Extreme")
say("=" * 104)
say("\nA. TRAJECTORIES (monitor rows every 5k; last-wins dedup)")
say("   step -> val@t64 | ret_sched | retfm | eta | lam_joint_max (frac_exp) | lam_max_max")
traj = {}
for a in ARMS:
    rows = load_rows(a)
    mons = dict(sorted((s, r["monitor"]) for (k, s), r in rows.items() if k == "mon"))
    traj[a] = mons
    say(f"  {a} ({sum(1 for k in rows if k[0]=='train')} train rows, {len(mons)} monitors):")
    for s, m in mons.items():
        say(f"    {s:>6}: val {fmt(m.get('val_t64'))} | ret_s {fmt(m.get('ret_sched_t8'))} | retfm {fmt(m.get('ret_final_t8'))}"
            f" | eta {fmt(m.get('eta'))} | lamJ {fmt(m.get('lam_joint_max'))} ({fmt(m.get('lam_joint_frac_expansive'),2)})"
            f" | lamY {fmt(m.get('lam_max_max'))}")
say("\n  A.1 collapse ordering (B4/B4s1): first monitor lam_joint_max>1 vs first retfm<0.9")
for a in ["B4", "B4s1"]:
    mons = traj.get(a, {})
    lam_cross = next((s for s, m in mons.items() if (m.get("lam_joint_max") or 0) > 1.0), None)
    lamy_cross = next((s for s, m in mons.items() if (m.get("lam_max_max") or 0) > 1.0), None)
    rf_cross = next((s for s, m in mons.items() if (m.get("ret_final_t8") if m.get("ret_final_t8") is not None else 1) < 0.9), None)
    say(f"    {a}: lamJ>1 @ {lam_cross} | lamY>1 @ {lamy_cross} | retfm<.9 @ {rf_cross}")

# ---------- B. funnel spectroscopy ----------
say("\nB. FUNNEL SPECTROSCOPY vs d (censored-geometric per-draw hit-rate by rating octile; log-slope over octiles 2-8)")
def spec(recdir, k):
    p = RUNS / recdir / "records_all.npz"
    if p.exists():
        z = np.load(p, allow_pickle=True)
        rat = z["rating"]; fh = z["first_exact"] if "first_exact" in z.files else z["mi_first_hit"]
    else:
        parts = sorted((RUNS / recdir).glob("records_s*.npz"))
        if not parts: return None
        rat, fh = [], []
        for q in parts:
            z = np.load(q, allow_pickle=True)
            rat.append(z["rating"]); fh.append(z["first_exact"] if "first_exact" in z.files else z["mi_first_hit"])
        rat = np.concatenate(rat); fh = np.concatenate(fh)
    qs = np.quantile(rat, np.linspace(0, 1, 9))
    rates = []
    for i in range(8):
        lo, hi = qs[i], qs[i + 1]
        m = (rat >= lo) & ((rat <= hi) if i == 7 else (rat < hi))
        if m.sum() == 0: rates.append(float("nan")); continue
        hit = fh[m] >= 0
        draws = np.where(hit, fh[m] + 1, k)
        rates.append(float(hit.sum()) / max(float(draws.sum()), 1.0))
    return rates
def logslope(rates):
    xs = [i for i, r in enumerate(rates) if r and r > 0 and not math.isnan(r)]
    ys = [math.log(rates[i]) for i in xs]
    if len(xs) < 3: return None
    n = len(xs); sx = sum(xs); sy = sum(ys)
    sxx = sum(x * x for x in xs); sxy = sum(x * y for x, y in zip(xs, ys))
    return (n * sxy - sx * sy) / (n * sxx - sx * sx)
for name, d, k in [("B2-d64 final(50k)", "sxbreadth20k_psportBB2", 128),
                   ("B2-d64 mid(25k)  ", "sxbreadth20k_psportBB2_mid", 128),
                   ("S5-d16 T6 (ref)  ", "sxbreadth20000_S5_k128", 128),
                   ("A4s1-d16 T12eq   ", "sxbreadth20k_psport3aA4s1", 128),
                   ("A2-d16 T12 RI/NI ", "sxbreadth20k_psport3aA2", 128),
                   ("A7s1-d16 T12plain", "sxbreadth20k_psport3aA7s1", 128)]:
    r = spec(d, k)
    if r is None: say(f"  {name}: no records"); continue
    sl = logslope(r)
    say(f"  {name}: %/draw " + " ".join("  -  " if (x != x) else f"{100*x:5.2f}" for x in r)
        + (f" | slope {sl:+.3f}" if sl is not None else ""))

# ---------- C. probes: basins at width ----------
say("\nC. PROBES AT d64 (512 rows/arm): retention | ladder S(eps) survival of retained | multi-init hit rate | median I_s")
def probe(a):
    p = RUNS / f"sudprobe_psportB{a}" / "results.jsonl"
    if not p.exists(): return None
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    ret = np.mean([r["gt_retention"] for r in rows])
    lad = [r["q_ladder"] for r in rows if r.get("q_ladder")]
    S = None
    if lad:
        # CORRECTED 2026-08-29 (graded-ladder retro-analysis): q_ladder is a DICT
        # keyed by eps-string; the earlier list-iteration read bool(key) == True
        # always -> false "saturation". Read values in eps order.
        def _vals(q):
            if isinstance(q, dict):
                return [bool(v) for _, v in sorted(((float(k), v) for k, v in q.items()))]
            return [bool(x) for x in q]
        arr = np.array([_vals(row) for row in lad])
        base = np.array([r["gt_retention"] for r in rows if r.get("q_ladder")], dtype=bool)
        if base.sum(): S = arr[base].mean(axis=0)
    mih = np.array([r.get("multi_init_hits", 0) for r in rows]); mik = np.array([r.get("multi_init_k", 1) for r in rows])
    mi = float(mih.sum()) / max(float(mik.sum()), 1)
    Is = np.median(np.array([r["I_s"] for r in rows if r.get("I_s")]), axis=0) if any(r.get("I_s") for r in rows) else None
    return ret, S, mi, Is
for a in ["B1", "B2", "B3", "B4"]:
    pr = probe(a)
    if pr is None: say(f"  {a}: no probe"); continue
    ret, S, mi, Is = pr
    say(f"  {a}: retention {ret:.3f} | S(eps) " + ("  -  " if S is None else "/".join(f"{x:.2f}" for x in S))
        + f" | mi rate/draw {100*mi:.2f}% | I_s " + ("  -  " if Is is None else "[" + ", ".join(f"{x:.0f}" for x in Is) + "]"
        + f" tot {Is.sum():.0f}"))

# ---------- D. Fisher-from-Adam-v ----------
say("\nD. FISHER-FROM-ADAM-v (nu from ScaleByAdamState; per-block share of v-trace + participation ratio)")
def find_nu(t):
    if hasattr(t, "_fields") and "nu" in getattr(t, "_fields", ()): return t.nu
    if isinstance(t, (list, tuple)):
        for x in t:
            r = find_nu(x)
            if r is not None: return r
    return None
def flatd(t, pre=""):
    out = {}
    if isinstance(t, dict):
        for k, v in t.items(): out.update(flatd(v, f"{pre}/{k}"))
    elif hasattr(t, "shape"): out[pre] = np.asarray(t)
    return out
def vstats(ck):
    try:
        with open(ck, "rb") as fh: c = pickle.load(fh)
    except Exception: return None
    nu = find_nu(c.get("opt_state"))
    if nu is None: return None
    fl = flatd(nu)
    if not fl: return None
    tot = sum(float(v.sum()) for v in fl.values())
    sq = sum(float((v.astype(np.float64) ** 2).sum()) for v in fl.values())
    n = sum(v.size for v in fl.values())
    pr = (tot * tot) / (n * sq) if sq > 0 else None
    blocks = {}
    for k, v in fl.items():
        parts = [x for x in k.split("/") if x]
        base = parts[1] if len(parts) > 1 else parts[0]
        blocks[base] = blocks.get(base, 0.0) + float(v.sum())
    top = sorted(blocks.items(), key=lambda x: -x[1])[:4]
    return tot, pr, [(b, v / tot if tot else 0) for b, v in top]
for a in ["B2", "B4", "B3"]:
    d = RUNS / f"pretrainsportB_{a}"
    cks = sorted(d.glob("ckpt_0*.pkl"))
    say(f"  {a}:")
    step_sel = cks[:: max(1, len(cks) // 4)][:5]
    for ck in step_sel:
        st = vstats(ck)
        step = ck.stem.replace("ckpt_", "")
        if st is None: say(f"    {step}: nu not found"); continue
        tot, pr, top = st
        say(f"    {step}: v-trace {tot:.3e} | PR {fmt(pr,4)} | " + ", ".join(f"{b}={s:.0%}" for b, s in top))

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(L) + "\n")
print(f"\nartifact -> {OUT}")
