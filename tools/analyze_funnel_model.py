# Ledger: TWO-PARAMETER FUNNEL MODEL ($0, disk-only; 2026-08-29, lens B of the
# retro-measurement program; banked campaigns ONLY — sportBr2b excluded).
# MOTIVATION: the rung-2 instrument dissociation — the censored-geometric mean
# rate steepened with scale while vote-coverage dominated: mean-rate and
# fraction-reachable are different objects. MODEL per rating octile:
#     P(first hit at draw t) = rho * (1-r)^t * r          (t = 0,1,...)
#     P(no hit ever)         = (1-rho)
# i.e. a puzzle is REACHABLE with prob rho; given reachable, draws hit i.i.d.
# at rate r. FIT: MLE on draws <= K_FIT=64 (hits later than 64 count as
# censored-at-64 for the fit). VALIDATION (out-of-sample in the draw axis):
# predict vote@128 = rho*(1-(1-r)^128) and compare to the ACTUAL held-out
# vote@128 from the same records; baseline = the single-rate censored-geometric
# (rho fixed 1). Deliverables: (a) the (rho, r) decomposition per octile per
# scale — WHERE width pays; (b) the validation table; (c) labeled k-extrapolation
# for C3-d96 (vote@256/512/1024 forecasts); (d) records-vs-summary hygiene
# (dedup by idx; the B2 convention flag adjudicated).
"""
  .venv/bin/python tools/analyze_funnel_model.py  # -> runs/analysis/funnel_model_20260829.txt
"""
from __future__ import annotations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "analysis" / "funnel_model_20260830.txt"
K_FIT, K_VAL = 64, 128
L = []
def say(s=""): L.append(str(s)); print(s)

def load(d):
    p = RUNS / d
    parts = sorted(p.glob("records_s*.npz")) or ([p / "records_all.npz"] if (p / "records_all.npz").exists() else [])
    if not parts: return None
    idx, rat, fh = [], [], []
    for q in parts:
        z = np.load(q, allow_pickle=True)
        if "mi_first_hit" not in z.files: return None
        idx.append(np.asarray(z["idx"])); rat.append(np.asarray(z["rating"])); fh.append(np.asarray(z["mi_first_hit"]))
    idx, rat, fh = map(np.concatenate, (idx, rat, fh))
    _, first = np.unique(idx, return_index=True)          # DEDUP by idx (stale-shard guard)
    dup = len(idx) - len(first)
    return dict(idx=idx[first], rating=rat[first], fh=fh[first], dup=dup)

def fit_octile(fh, k_fit=K_FIT):
    """Grid MLE for (rho, r) on draws <= k_fit."""
    hit = (fh >= 0) & (fh < k_fit)
    t = fh[hit].astype(float)
    m = int((~hit).sum())                                  # censored at k_fit
    rhos = np.linspace(0.01, 1.0, 100)
    rs = np.exp(np.linspace(np.log(1e-5), np.log(0.9), 120))
    R, P = np.meshgrid(rs, rhos)
    with np.errstate(divide="ignore"):
        ll_hit = (np.log(P) + np.log(R))[..., None] + np.log1p(-R)[..., None] * t[None, None, :]
        ll = ll_hit.sum(axis=-1) if len(t) else 0.0
        ll = ll + m * np.log((1 - P) + P * (1 - R) ** k_fit)
    i = np.unravel_index(np.argmax(ll), ll.shape)
    rho, r = float(P[i]), float(R[i])
    # single-rate baseline (rho == 1)
    j = np.argmax(ll[-1, :] if rhos[-1] == 1.0 else ll[-1, :])
    r1 = float(rs[j])
    return rho, r, r1

def analyze(name, d, octiles=8):
    rec = load(d)
    if rec is None: say(f"  {name}: no per-draw records"); return
    fh, rat = rec["fh"], rec["rating"]
    n = len(fh)
    act128 = float(np.mean((fh >= 0) & (fh < K_VAL)))
    say(f"\n  {name} (n={n}, dup-dropped {rec['dup']}): actual vote@64 {100*np.mean((fh>=0)&(fh<K_FIT)):.2f} | vote@128 {100*act128:.2f}")
    qs = np.quantile(rat, np.linspace(0, 1, octiles + 1))
    pred2, pred1, w = 0.0, 0.0, 0
    say("    octile   rho     r/draw    | pred@128(2p)  pred@128(1p)  actual@128")
    for i in range(octiles):
        m = (rat >= qs[i]) & ((rat <= qs[i + 1]) if i == octiles - 1 else (rat < qs[i + 1]))
        if m.sum() < 50: continue
        rho, r, r1 = fit_octile(fh[m])
        p2 = rho * (1 - (1 - r) ** K_VAL)
        p1 = 1 - (1 - r1) ** K_VAL
        a = float(np.mean((fh[m] >= 0) & (fh[m] < K_VAL)))
        pred2 += p2 * m.sum(); pred1 += p1 * m.sum(); w += m.sum()
        say(f"    [{qs[i]:3.0f}-{qs[i+1]:3.0f}] rho {rho:.3f}  r {r:8.5f} | {100*p2:6.2f}        {100*p1:6.2f}        {100*a:6.2f}")
    say(f"    AGGREGATE pred@128: two-param {100*pred2/w:.2f} | single-rate {100*pred1/w:.2f} | ACTUAL {100*act128:.2f}"
        f"  -> 2p err {100*(pred2/w-act128):+.2f}pp, 1p err {100*(pred1/w-act128):+.2f}pp")
    return rec

say("=" * 114)
say(f"TWO-PARAMETER FUNNEL MODEL (2026-08-29) — fit on draws<= {K_FIT}, validated out-of-sample on vote@{K_VAL}; sportBr2b EXCLUDED")
say("=" * 114)

DATASETS = [
    ("S5-d16 (20k, k128)",      "sxbreadth20000_S5_k128"),
    ("W2-d16-w2 (20k, k128)",   "sxbreadth20k_psport2w2W2"),
    ("A4s1-d16 (20k, k128)",    "sxbreadth20k_psport3aA4s1"),
    ("A2-d16 (20k, k128)",      "sxbreadth20k_psport3aA2"),
    ("B2-d64 (20k, k128)",      "sxbreadth20k_psportBB2"),
    ("B2-d64-mid25k (20k)",     "sxbreadth20k_psportBB2_mid"),
    ("C3-d96 (20k, k128)",      "sxbreadth20k_psportBr2C3"),
    # rung-2b screens (strat-512, k256; n=512 => noisier fits, labeled)
    ("D1-d96-vb@10k (strat512)",  "sxscreen_psportBr2bD1_vb"),
    ("D2-d96-vb@20k (strat512)",  "sxscreen_psportBr2bD2_vb"),
    ("D3-d96-vb@40k (strat512)",  "sxscreen_psportBr2bD3_vb"),
    ("D4-d96-vb@50k (strat512)",  "sxscreen_psportBr2bD4_vb"),
    ("C3X-d96-vb@30k (strat512)", "sxscreen_psportBr2bC3X_vb"),
]
recs = {}
for name, d in DATASETS:
    r = analyze(name, d)
    if r is not None: recs[name] = r

say("\nK-EXTRAPOLATION (labeled MODEL-BASED forecast, not a measurement): C3-d96 per-octile (rho, r) -> vote@k")
rec = recs.get("C3-d96 (20k, k128)")
if rec:
    fh, rat = rec["fh"], rec["rating"]
    qs = np.quantile(rat, np.linspace(0, 1, 9))
    for K in (256, 512, 1024):
        tot, w = 0.0, 0
        for i in range(8):
            m = (rat >= qs[i]) & ((rat <= qs[i + 1]) if i == 7 else (rat < qs[i + 1]))
            if m.sum() < 50: continue
            rho, r, _ = fit_octile(fh[m], k_fit=K_VAL)   # forecast uses ALL k<=128 data
            tot += rho * (1 - (1 - r) ** K) * m.sum(); w += m.sum()
        say(f"  predicted vote@{K}: {100*tot/w:.2f} %  (B-M2 bar 85; B-M3 bar 95)")

say("\nHYGIENE — records-vs-summary reconciliation (the B2 flag):")
import json
for name, d, sk in [("B2-d64", "sxbreadth20k_psportBB2", "128"), ("C3-d96", "sxbreadth20k_psportBr2C3", "128"), ("S5-d16", "sxbreadth20000_S5_k128", "128")]:
    rec = load(d)
    s = json.loads((RUNS / d / "summary_all.json").read_text())
    sv = s.get("vote_at_k", {}).get(sk)
    rv = float(np.mean((rec["fh"] >= 0) & (rec["fh"] < int(sk))))
    say(f"  {name}: records n={len(rec['fh'])} (dup-dropped {rec['dup']}) vote@{sk} {100*rv:.2f} | summary {100*sv:.2f} | delta {100*(rv-sv):+.2f}pp")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(L) + "\n")
print(f"\nartifact -> {OUT}")
