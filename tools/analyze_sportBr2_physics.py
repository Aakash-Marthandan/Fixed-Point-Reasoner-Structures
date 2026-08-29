# Ledger: PHASE B RUNG 2 PHYSICS PASS — written AT ANALYSIS TIME (2026-08-29),
# descriptive only (no registered rules; the verdict is analyze_sportBr2.py's).
# A. trajectories (metrics.jsonl, LAST-WINS dedup): eta/eta_z, lam_y vs lam_joint,
#    retfm/ret_sched, val@t64 at 2k cadence; A.1 = NaN forensics on the stopped
#    carrier pair from the GCS live-metrics tails (labeled: pre-halt rows synced
#    <=5 min before each halt; scratchpad copies).
# B. funnel spectroscopy vs scale (censored-geometric per-draw hit-rate by rating
#    octile, log-slope): C3-d96 vs B2-d64 vs S5-d16 (identical 20k subsample) +
#    the registered d96 slope prediction [-0.50,-0.35].
# C. screen k-curves vs training step (H-47/H-46-funnel instrument), including
#    the LABELED supplementary x_ screens of the stopped pair.
# D. probes (512 rows/arm): retention | EXTENDED eps-ladder S(eps) | multi-init
#    rate | median I_s + throat prediction checks; D5 = the registered $0
#    spurious-attractor analysis (violations at stuck states, best-wrong).
# E. Fisher-from-Adam-v (nu from opt_state): per-block shares + PR across grids.
# F. depth rider t256-vs-t64 by rating bin (the inference-depth lever at d96).
# G. D3 demo read (verified vs unverified vote; banked b2d64 + s5d16 cells).
"""
  .venv/bin/python tools/analyze_sportBr2_physics.py  # -> runs/analysis/sportBr2_physics_20260829.txt
"""
from __future__ import annotations
import json, math, pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "analysis" / "sportBr2_physics_20260829.txt"
SP = Path("/private/tmp/claude-501/-Users-aakash-Projects-HRRN/6d10cb0e-61cd-4b13-9fe1-5c0a08f49979/scratchpad/forensics")
ARMS = ["C1", "C1s1", "C2", "C3", "C4"]
L = []
def say(s=""): L.append(str(s)); print(s)
def fmt(x, nd=3): return "  -  " if x is None else f"{x:.{nd}f}"

def load_rows(path):
    p = Path(path)
    if not p.exists(): return {}
    rows = {}
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line: continue
        try: r = json.loads(line)
        except json.JSONDecodeError: continue
        if "monitor" in r: key = ("mon", r["monitor"]["step"])
        elif "step" in r:  key = ("train", r["step"])
        else: continue
        rows[key] = r                                    # LAST-WINS per (kind, step)
    return rows

say("=" * 108)
say("PHASE B RUNG 2 PHYSICS PASS (descriptive; 2026-08-29) — d96 full-width, Sudoku-Extreme")
say("=" * 108)

# ---------- A. trajectories ----------
say("\nA. TRAJECTORIES (monitor rows every 2k; last-wins dedup; stopped arms = labeled partials)")
say("   step -> val@t64 | ret_sched | retfm | eta | eta_z | lamY (max) | lamJ (max, frac_exp)")
for a in ARMS:
    rows = load_rows(RUNS / f"pretrainsportBr2_{a}" / "metrics.jsonl")
    mons = dict(sorted((s, r["monitor"]) for (k, s), r in rows.items() if k == "mon"))
    ntr = sum(1 for k in rows if k[0] == "train")
    say(f"  {a} ({ntr} train rows, {len(mons)} monitors):")
    for s, m in mons.items():
        say(f"    {s:>6}: val {fmt(m.get('val_t64'))} | ret_s {fmt(m.get('ret_sched_t8'))} | retfm {fmt(m.get('ret_final_t8'))}"
            f" | eta {fmt(m.get('eta'))} | eta_z {fmt(m.get('eta_z'))} | lamY {fmt(m.get('lam_max_max'))}"
            f" | lamJ {fmt(m.get('lam_joint_max'))} ({fmt(m.get('lam_joint_frac_expansive'), 2)})")

say("\n  A.1 NaN FORENSICS on the stopped carrier pair (GCS live-metrics tails; LABELED pre-halt rows)")
for a, halt in [("C1", "~22.1k (half-lr; first NaN ~11.5k at lr 1e-3)"),
                ("C1s1", "~12.2k (half-lr; first NaN ~8.3k at lr 1e-3)")]:
    p = SP / f"{a}_metrics_live.jsonl"
    if not p.exists(): say(f"    {a}: live metrics not staged"); continue
    rows = load_rows(p)
    tr = dict(sorted((s, r) for (k, s), r in rows.items() if k == "train"))
    mons = dict(sorted((s, r["monitor"]) for (k, s), r in rows.items() if k == "mon"))
    steps = list(tr)
    say(f"    {a} (halt {halt}): live train rows to {steps[-1]}")
    for s in steps[-4:]:
        r = tr[s]
        say(f"      {s:>6}: loss {r['loss']:.4g} | I {r.get('I_total'):.4g} | A {r.get('A_total'):.4g} | fpa_ce {r.get('fpa_ce'):.3g}")
    lastm = list(mons)[-3:]
    for s in lastm:
        m = mons[s]
        say(f"      mon {s:>6}: val {fmt(m.get('val_t64'))} | retfm {fmt(m.get('ret_final_t8'))} | eta {fmt(m.get('eta'))}"
            f" | lamY {fmt(m.get('lam_max_max'))} | lamJ {fmt(m.get('lam_joint_max'))}")

say("\n  A.2 free-vs-priced flux along training (train-row medians of I_total / A_total, last-wins)")
for a in ARMS:
    rows = load_rows(RUNS / f"pretrainsportBr2_{a}" / "metrics.jsonl")
    tr = [r for (k, s), r in sorted(rows.items()) if k == "train"]
    if not tr: say(f"    {a}: no train rows"); continue
    I = np.median([r["I_total"] for r in tr[-40:]]); A = np.median([r["A_total"] for r in tr[-40:]])
    say(f"    {a}: late-train median I_total {I:.4g} | A_total {A:.4g}")

# ---------- B. funnel spectroscopy ----------
say("\nB. FUNNEL SPECTROSCOPY vs SCALE (per-octile censored-geometric %/draw; log-slope over nonzero octiles)")
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
for name, d, k in [("C3-d96 T6+FPA 20k ", "sxbreadth20k_psportBr2C3", 128),
                   ("B2-d64 T12+FPA 50k", "sxbreadth20k_psportBB2", 128),
                   ("B2-d64 mid(25k)   ", "sxbreadth20k_psportBB2_mid", 128),
                   ("S5-d16 T6 plain   ", "sxbreadth20000_S5_k128", 128)]:
    r = spec(d, k)
    if r is None: say(f"  {name}: no records"); continue
    sl = logslope(r)
    say(f"  {name}: %/draw " + " ".join("  -  " if (x != x) else f"{100*x:5.2f}" for x in r)
        + (f" | slope {sl:+.3f}" if sl is not None else ""))
say("  registered d96 slope prediction: flattens into [-0.50, -0.35] (from -0.529 at d64)")
p4 = json.loads((RUNS / "sxbreadth20k_psportBr2C3" / "summary_all.json").read_text())
bv, rb = p4.get("by_rating_bin_vote"), p4.get("rating_bins", [])
if bv:
    labels = [f"{int(rb[i])}-{int(rb[i+1])}" if i + 1 < len(rb) else str(i) for i in range(len(bv))]
    say("  C3-d96 per-octile vote@128 (20k): " + " ".join(f"[{l}]={100*v:.1f}" for l, v in zip(labels, bv) if v is not None))
s5 = json.loads((RUNS / "sxbreadth20000_S5_k128" / "summary_all.json").read_text())
bv5 = s5.get("by_rating_bin_vote")
if bv5: say("  S5-d16 per-octile vote@128 (20k): " + " ".join(f"[{l}]={100*v:.1f}" for l, v in zip(labels, bv5) if v is not None))
b2 = json.loads((RUNS / "sxbreadth20k_psportBB2" / "summary_all.json").read_text())
bv2 = b2.get("by_rating_bin_vote")
if bv2: say("  B2-d64 per-octile vote@128 (20k): " + " ".join(f"[{l}]={100*v:.1f}" for l, v in zip(labels, bv2) if v is not None))

# ---------- C. screen k-curves vs training ----------
say("\nC. FUNNEL vs TRAINING (screens vote@k on strat-512; x_ rows = LABELED supplementary screens of stopped arms)")
def kcurve(d):
    p = RUNS / d / "summary_all.json"
    if not p.exists(): return None
    s = json.loads(p.read_text())
    v = s.get("vote_at_k", {})
    return {int(kk): vv for kk, vv in v.items()}
rows = [("C1  ", [("x 10000", "sxscreen_x_psportBr2C1_s010000"), ("x 15000", "sxscreen_x_psportBr2C1_s015000"),
                  ("vb 20000", "sxscreen_psportBr2C1_vb")]),
        ("C1s1", [("x 5000", "sxscreen_x_psportBr2C1s1_s005000"), ("vb 10000", "sxscreen_psportBr2C1s1_vb")]),
        ("C2  ", [("m1 25000", "sxscreen_psportBr2C2_m1"), ("m2 40000", "sxscreen_psportBr2C2_m2"), ("vb 50000", "sxscreen_psportBr2C2_vb")]),
        ("C3  ", [("m1 10000", "sxscreen_psportBr2C3_m1"), ("m2 15000", "sxscreen_psportBr2C3_m2"), ("vb 20000", "sxscreen_psportBr2C3_vb")]),
        ("C4  ", [("m1 25000", "sxscreen_psportBr2C4_m1"), ("m2 40000", "sxscreen_psportBr2C4_m2"), ("vb 50000", "sxscreen_psportBr2C4_vb")])]
for arm, lst in rows:
    say(f"  {arm}:")
    for tag, d in lst:
        kv = kcurve(d)
        if not kv: say(f"    {tag:9s}: no screen"); continue
        ks = sorted(kv)
        say(f"    {tag:9s}: " + " ".join(f"v{k}={100*kv[k]:5.2f}" for k in ks if k in (1, 16, 64, 128, 256))
            + f" | v256/v16 {kv[256]/max(kv[16],1e-9):.2f}")

# ---------- D. probes ----------
say("\nD. PROBES (512 rows/arm; EXTENDED eps-ladder .05/.1/.2/.4/.6/.8)")
say("   arm: retention | S(eps) of retained | mi rate/draw | median I_s [per-scale] tot")
pred = {"C2": (850, 1100), "C4": (1500, 4000)}
for a in ["C1", "C1s1", "C2", "C3"]:
    p = RUNS / f"sudprobe_psportBr2{a}" / "results.jsonl"
    if not p.exists(): say(f"  {a}: no probe"); continue
    rows_ = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    ret = np.mean([r["gt_retention"] for r in rows_])
    lad = [r["q_ladder"] for r in rows_ if r.get("q_ladder")]
    S = None
    if lad:
        arr = np.array([[bool(x) for x in row] for row in lad])
        base = np.array([r["gt_retention"] for r in rows_ if r.get("q_ladder")], dtype=bool)
        if base.sum(): S = arr[base].mean(axis=0)
    mih = np.array([r.get("multi_init_hits", 0) for r in rows_]); mik = np.array([r.get("multi_init_k", 1) for r in rows_])
    mi = float(mih.sum()) / max(float(mik.sum()), 1)
    Is = np.median(np.array([r["I_s"] for r in rows_ if r.get("I_s")]), axis=0) if any(r.get("I_s") for r in rows_) else None
    line = (f"  {a}: ret {ret:.3f} | S(eps) " + ("  -  " if S is None else "/".join(f"{x:.2f}" for x in S))
            + f" | mi {100*mi:.2f}%")
    if Is is not None:
        line += f" | I_s [" + ", ".join(f"{x:.0f}" for x in Is) + f"] tot {Is.sum():.0f}"
        if a in pred: line += f" (pred {pred[a][0]}-{pred[a][1]})"
    say(line)
say("\n  D5. SPURIOUS-ATTRACTOR ANALYSIS (registered $0 rider: stuck states on det-failed rows)")
say("      arm: failed rows | med violations@fail | frac givens intact | frac mi-best-wrong (draws end wrong-valid?)")
for a in ["C1", "C1s1", "C2", "C3"]:
    p = RUNS / f"sudprobe_psportBr2{a}" / "results.jsonl"
    if not p.exists(): continue
    rows_ = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    fail = [r for r in rows_ if not r.get("solved")]
    if not fail: say(f"  {a}: no failed rows"); continue
    viol = np.median([r.get("violations", -1) for r in fail])
    gk = np.mean([1.0 if r.get("givens_kept", 0) >= r.get("givens_total", 1) else 0.0 for r in fail])
    bw = np.mean([1.0 if r.get("multi_init_best_wrong") else 0.0 for r in fail])
    say(f"  {a}: {len(fail)} failed | med viol {viol:.0f} | givens-intact {gk:.2f} | mi-best-wrong {bw:.2f}")

# ---------- E. Fisher-from-Adam-v ----------
say("\nE. FISHER-FROM-ADAM-v (nu from opt_state; per-block share of v-trace + participation ratio)")
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
for a in ARMS:
    d = RUNS / f"pretrainsportBr2_{a}"
    cks = sorted(d.glob("ckpt_0*.pkl"))
    say(f"  {a}:")
    step_sel = cks[:: max(1, len(cks) // 4)][:5]
    for ck in step_sel:
        st = vstats(ck)
        step = ck.stem.replace("ckpt_", "")
        if st is None: say(f"    {step}: nu not found"); continue
        tot, pr, top = st
        say(f"    {step}: v-trace {tot:.3e} | PR {fmt(pr, 4)} | " + ", ".join(f"{b}={s:.0%}" for b, s in top))

# ---------- F. depth rider ----------
say("\nF. INFERENCE-DEPTH RIDER (C3: t=256 vs t=64, full test n=422,786; labeled row, never the headline)")
dep = json.loads((RUNS / "sxdepth_psportBr2C3_t256" / "summary_all.json").read_text())
t64 = json.loads((RUNS / "sxeval_psportBr2C3" / "full_t64" / "summary_all.json").read_text())
say(f"  cold: t64 {100*t64['exact_acc']:.2f} -> t256 {100*dep['exact_acc']:.2f} (+{100*(dep['exact_acc']-t64['exact_acc']):.2f} pp)")
b64, b256, rb = t64.get("by_rating_bin"), dep.get("by_rating_bin"), t64.get("rating_bins", [])
if b64 and b256:
    labels = [f"{int(rb[i])}-{int(rb[i+1])}" if i + 1 < len(rb) else str(i) for i in range(len(b64))]
    say("  by rating octile (t64 -> t256): " + " | ".join(
        f"[{l}] {100*x:.1f}->{100*y:.1f}" for l, x, y in zip(labels, b64, b256) if x is not None and y is not None))
say("  all-arm depth response (strat-512 cheap evals, exact t6 -> t64 -> t256; labeled instrument set):")
for a in ARMS:
    try:
        vals = [json.loads((RUNS / f"sxeval_psportBr2{a}" / f"strat_t{t}" / "summary_all.json").read_text())["exact_acc"]
                for t in (6, 64, 256)]
        say(f"    {a}: " + " -> ".join(f"{100*v:.2f}" for v in vals))
    except Exception: say(f"    {a}: incomplete")
say("  failure class (full_t64, n=422,786): valid_wrong_frac | mean violations | givens kept")
for a in ARMS:
    s = json.loads((RUNS / f"sxeval_psportBr2{a}" / "full_t64" / "summary_all.json").read_text())
    say(f"    {a}: valid_wrong {s.get('valid_wrong_frac'):.4f} | viol {s.get('mean_violations'):.2f} | givens {s.get('givens_kept_frac'):.4f}")

# ---------- G. D3 demo ----------
say("\nG. D3 DEMO (informational; verified vote vs unverified majority on banked ckpts)")
for name, d in [("B2-d64", "sxd3demo_b2d64"), ("S5-d16", "sxd3demo_s5d16")]:
    p = RUNS / d / "summary_all.json"
    if not p.exists(): say(f"  {name}: absent"); continue
    s = json.loads(p.read_text())
    v = s.get("vote_at_k", {}); u = s.get("majority_vote_at_k", {})
    ks = sorted(set(v) & set(u), key=int) if u else sorted(v, key=int)
    if u:
        say(f"  {name}: " + "  ".join(f"k={k}: ver {100*v[k]:.1f} / unver-maj {100*u[k]:.1f}" for k in ks if int(k) in (16, 64, 128)))
    else:
        say(f"  {name}: verified-only " + " ".join(f"v{k}={100*vv:.1f}" for k, vv in sorted(v.items(), key=lambda x: int(x[0]))))

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(L) + "\n")
print(f"\nartifact -> {OUT}")
