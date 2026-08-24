# Ledger: SPRINT S2 WAVE 3a PHYSICS PASS (2026-08-24) — descriptive/exploratory,
# written AT analysis time (the registered rules live in analyze_sport3a.py,
# untouched). Reads banked artifacts only; no decision rules here. Sections:
#   A. RG lens        — monitor trajectories (val@t64, final-map retention,
#                       joint lambda_max, eta) per arm; the contractivity flow.
#   B. QG lens        — lambda/retfm event ordering on the collapsing arms;
#                       eq_coupled (a1,a2); schedule-vs-final retention split.
#   C. Info lens      — throats (I_total/A_total at end of training), fpa_ce,
#                       compression vs seed reproducibility.
#   D. Holography     — vote@k curves incl. recomputed vote@512/@1024 from the
#                       k=1024 records; per-draw hit-rate vs rating octile and
#                       its log-linear decay slope per substrate (funnel width).
#   E. Spectra/basins — probe I_s per-scale profiles (UV share), corruption
#                       ladders S(eps), multi-init hit rates.
"""  .venv/bin/python tools/analyze_sport3a_physics.py  -> runs/analysis/sport3a_physics_20260824.txt  """
from __future__ import annotations
import json, math, os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(os.environ.get("QHRRN_RUNS", ROOT / "runs"))
OUT = RUNS / "analysis" / "sport3a_physics_20260824.txt"
ARMS = ["A2", "A3", "A4", "A5", "A6", "A7", "A7s1", "A8", "A2s1", "A3s1", "A4s1", "A5s1", "A6s1", "A9", "A9s1", "A10"]
TAG = {"S5": "sport2", "W2": "sport2w2", "W3": "sport2w2", "W8": "sport2w2", "W9": "sport2w2"}
L = []
def say(s=""): L.append(str(s)); print(s)
def jload(p):
    p = Path(p); return json.loads(p.read_text()) if p.exists() else None
def tag(a): return TAG.get(a, "sport3a")
def summ(a, kind): return jload(RUNS / f"sxeval_p{tag(a)}{a}" / kind / "summary_all.json")
def monitors(a):
    p = RUNS / f"pretrain{tag(a)}_{a}" / "metrics.jsonl"
    out = []
    if p.exists():
        for l in p.read_text().splitlines():
            try: r = json.loads(l)
            except Exception: continue
            if "monitor" in r: out.append(r["monitor"])
    return sorted(out, key=lambda m: m["step"])
def metrics_rows(a):
    p = RUNS / f"pretrain{tag(a)}_{a}" / "metrics.jsonl"
    out = []
    if p.exists():
        for l in p.read_text().splitlines():
            try: r = json.loads(l)
            except Exception: continue
            if "monitor" not in r: out.append(r)
    return out
def probe_rows(a):
    p = RUNS / f"sudprobe_p{tag(a)}{a}" / "results.jsonl"
    if not p.exists(): return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def spear(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0: return None
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])

def main():
    say("=" * 110); say("SPRINT S2 WAVE 3a PHYSICS PASS — 2026-08-24 (descriptive; registered verdicts in sport3a_verdict.txt)"); say("=" * 110)

    # ---------------- A. RG lens: trajectories ----------------
    say("\nA. RG LENS — monitor trajectories (per 5k: val@t64% / ret_final / lam_joint_mean; last row also eta)")
    for a in ARMS:
        M = monitors(a)
        cells = " ".join(f"{m['step']//1000:2d}k:{100*m['val_t64']:4.1f}/{m['ret_final_t8']:.2f}/{m.get('lam_joint_mean', m.get('lam_max_mean', float('nan'))):.2f}" for m in M)
        eta_end = M[-1].get("eta") if M else None
        say(f"  {a:5s} eta_end={eta_end if eta_end is None else f'{eta_end:.3f}'}")
        say(f"        {cells}")
    say("\n  per-arm summary: lam_mean end | lam max over traj | first step lam_mean>1 | val range last 25k (checkpoint variance)")
    for a in ARMS:
        M = monitors(a)
        lam = [m.get("lam_joint_mean", m.get("lam_max_mean")) for m in M]
        first = next((m["step"] for m, l in zip(M, lam) if l is not None and l > 1), None)
        tail = [100 * m["val_t64"] for m in M if m["step"] >= 25000]
        rng = (max(tail) - min(tail)) if tail else None
        say(f"  {a:5s} lam_end={lam[-1]:.2f} lam_max={max(lam):.2f} first>1={first} val25k+_range={rng:.1f}pp" if lam else f"  {a:5s} no monitors")

    # ---------------- B. QG lens ----------------
    say("\nB. QUANTUM-GEOMETRY LENS")
    say("  B1. event ordering on arms that lost final-map retention (retfm<0.9 at any monitor):")
    for a in ARMS:
        M = monitors(a)
        drop = next((m["step"] for m in M if m["ret_final_t8"] < 0.9), None)
        if drop is None: continue
        lam = {m["step"]: m.get("lam_joint_mean", m.get("lam_max_mean")) for m in M}
        cross = next((m["step"] for m in M if (m.get("lam_joint_mean", m.get("lam_max_mean")) or 0) > 1), None)
        say(f"    {a:5s} first retfm<0.9 at {drop}; first lam_mean>1 at {cross}; lam at drop = {lam.get(drop):.2f}")
    say("  B2. eq_coupled trained (a1,a2) [from eval summaries]:")
    for a in ["A4", "A4s1"]:
        s = summ(a, "strat_t64") or {}
        say(f"    {a:5s} eq_coupled_ab = {s.get('eq_coupled_ab')}")
    say("  B3. schedule-vs-final retention split (ret_t8 sched | retfm_t8 final) — the ramp-path geometry:")
    for a in ARMS:
        r1 = (summ(a, "ret_t8") or {}).get("exact_acc"); r2 = (summ(a, "retfm_t8") or {}).get("exact_acc")
        if r1 is not None and r2 is not None and abs(r1 - r2) > 0.1:
            say(f"    {a:5s} sched {r1:.2f} vs final {r2:.2f}  (delta {r2-r1:+.2f})")

    # ---------------- C. Info lens: throats ----------------
    say("\nC. INFORMATION LENS — end-of-training channel usage (last metrics row): I_total (streams) | A_total (attention) | fpa_ce | train loss")
    for a in ARMS:
        R = [r for r in metrics_rows(a) if r.get("I_total") is not None]
        if not R: say(f"  {a:5s} no metrics"); continue
        r = R[-1]
        fpa = r.get("fpa_ce"); ce = r.get("ce_in", r.get("ce"))
        say(f"  {a:5s} step={r.get('step', -1):6d} I={r['I_total']:12.1f} A={r.get('A_total', float('nan')):14.1f} fpa_ce={f'{fpa:.3f}' if fpa is not None else '  -  '} loss={r.get('loss', float('nan')):.3f} ce={f'{ce:.3f}' if ce is not None else '-'}")
    say("\n  seed-pair reproducibility vs compression (|cold(final) s0-s1| pp, I_total class):")
    pairs = [("A7","A7s1"),("A2","A2s1"),("A3","A3s1"),("A4","A4s1"),("A5","A5s1"),("A6","A6s1"),("A9","A9s1")]
    for a, b in pairs:
        ca = (summ(a, "full_t64") or {}).get("exact_acc"); cb = (summ(b, "full_t64") or {}).get("exact_acc")
        RI = [r for r in metrics_rows(a) if r.get("I_total") is not None]
        Ia = RI[-1].get("I_total") if RI else None
        if ca is None or cb is None: continue
        say(f"    {a}-{b}: |d|={100*abs(ca-cb):5.2f}pp  I_end={f'{Ia:12.1f}' if Ia is not None else '     -      '}")

    # ---------------- D. Holography: breadth ----------------
    say("\nD. HOLOGRAPHY LENS — verify-and-vote breadth (funnel width)")
    z = np.load(RUNS / "sxbreadth_S5_t64_k1024" / "records_all.npz", allow_pickle=True)
    fh, ver = z["mi_first_hit"], z["mi_verified"]
    hit = ver > 0
    def vote_at(k): return float(np.mean(hit & (fh < k)))
    say("  D1. S5 strat-512 k=1024 recomputed from records (mi_first_hit):")
    for k in [128, 256, 512, 1024]:
        say(f"      vote@{k:4d} = {100*vote_at(k):5.1f}%")
    s = jload(RUNS / "sxbreadth_S5_t64_k1024" / "summary_all.json") or {}
    say(f"      cross-check vs summary vote_at_k[128]={100*s.get('vote_at_k',{}).get('128',float('nan')):.1f} [256]={100*s.get('vote_at_k',{}).get('256',float('nan')):.1f}")
    say(f"      saturation: @1024-@512 = {100*(vote_at(1024)-vote_at(512)):+.1f}pp; @512-@256 = {100*(vote_at(512)-vote_at(256)):+.1f}pp")
    say("  D1b. S5 per-octile vote@k (records; octiles by rating rank, 64 each, easiest->hardest):")
    rat0 = z["rating"]
    order = np.argsort(rat0, kind="stable"); ob0 = np.empty(len(rat0), int); ob0[order] = np.arange(len(rat0)) // 64
    for k in [128, 1024]:
        per = [100 * float(np.mean((hit & (fh < k))[ob0 == i])) for i in range(8)]
        say(f"      vote@{k:4d} by octile: " + " ".join(f"{p:5.1f}" for p in per))
    say("\n  D2. per-draw hit-rate by rating octile + log-linear decay slope (funnel-width spectroscopy)")
    rat = z["rating"]
    def octiles(rr):
        # strat-512 is 64-per-rating-octile by construction: bin by rank into 8 groups of n/8
        order = np.argsort(np.asarray(rr), kind="stable"); ob = np.empty(len(rr), int)
        ob[order] = np.arange(len(rr)) // (len(rr) // 8)
        return np.clip(ob, 0, 7)
    def hitrate_slope(name, ratings, hits_count, k):
        ob = octiles(ratings); med = [float(np.median(ratings[ob == i])) for i in range(8)]
        hr = [float(np.mean(hits_count[ob == i]) / k) for i in range(8)]
        pos = [(m, h) for m, h in zip(med, hr) if h > 0]
        slope = None
        if len(pos) >= 4:
            xs = np.array([p[0] for p in pos]); ys = np.log(np.array([p[1] for p in pos]))
            slope = float(np.polyfit(xs, ys, 1)[0])
        say(f"      {name:28s} k={k:4d} hit/draw by octile: " + " ".join(f"{h:.3f}" for h in hr) + f"  | log-slope per rating unit: {f'{slope:.4f}' if slope is not None else '  -  '}")
        return hr, slope
    hitrate_slope("S5 plain T6 20k (strat,1024)", rat, ver.astype(float), 1024)
    for nm, lbl in [("W2", "W2 plain T12 30k"), ("W3", "W3 RI+NI T12 30k"), ("W8", "W8 priced d32 20k"), ("W9", "W9 priced d16 50k")]:
        p = RUNS / f"sxbreadth_{nm}_t64_k256" / "records_all.npz"
        if p.exists():
            zz = np.load(p, allow_pickle=True)
            hitrate_slope(f"{lbl} (strat,256)", zz["rating"], zz["mi_verified"].astype(float), 256)
    say("\n  D3. vote@k curves (labeled; breadth is never the headline):")
    def curve(name, s):
        if not s: say(f"      {name:34s} no data"); return
        v = s.get("vote_at_k", {}); ks = sorted(v.keys(), key=int)
        say(f"      {name:34s} n={s.get('n'):6d} " + " ".join(f"@{k}:{100*v[k]:.1f}" for k in ks))
    curve("S5 20k k128", jload(RUNS / "sxbreadth20000_S5_k128" / "summary_all.json"))
    curve("S5 20k k256", jload(RUNS / "sxbreadth20000_S5_k256" / "summary_all.json"))
    for nm in ["A2", "A4s1", "A7s1"]:
        curve(f"{nm} 20k k128 (PHASE4)", jload(RUNS / f"sxbreadth20k_psport3a{nm}" / "summary_all.json"))
    for nm in ["W2", "W3", "W8", "W9"]:
        curve(f"{nm} strat k256 (filler)", jload(RUNS / f"sxbreadth_{nm}_t64_k256" / "summary_all.json"))
    say("\n  D4. wave-3a arms strat vote@16 + per-draw mi hit rate (k=16, noisy):")
    for a in ARMS:
        s = summ(a, "strat_t64") or {}
        v = s.get("vote_at_k", {}).get("16"); mih = s.get("mi_hits_mean")
        say(f"      {a:5s} vote@16={100*v:5.1f}%  mi_hits/16={mih:.2f}  ({mih/16:.3f}/draw)" if v is not None and mih is not None else f"      {a:5s} -")

    # ---------------- E. Spectra & basins ----------------
    say("\nE. SPECTRA / BASIN GEOMETRY (probe suite; 512 rows/arm; A4/A4s1 skipped by design)")
    say("    I_s median per scale [s0..s4] | I_med total | UV share s0 | gt_retention | q_ladder S(eps) counts eps=.05/.1/.2/.4 | mi hits/k")
    for a in ARMS:
        P = probe_rows(a)
        if not P: say(f"  {a:5s} (no probe)"); continue
        Is = np.array([r["I_s"] for r in P if r.get("I_s") is not None])
        med = np.median(Is, axis=0) if len(Is) else None
        itot = float(np.median(Is.sum(axis=1))) if len(Is) else None
        uv = med[0] / med.sum() if med is not None else None
        ret = np.mean([r["gt_retention"] for r in P])
        lad = None
        if isinstance(P[0].get("q_ladder"), dict):
            eps_keys = sorted(P[0]["q_ladder"].keys(), key=float)
            lad = [np.mean([bool(r["q_ladder"].get(e)) for r in P]) for e in eps_keys]
        mih = np.mean([r["multi_init_hits"] for r in P]); mik = P[0].get("multi_init_k", 16)
        say(f"  {a:5s} I_s=" + (" ".join(f"{v:8.1f}" for v in med) if med is not None else "-") +
            f" | I_med={itot:9.1f} UV={uv:.2f} | ret={ret:.2f} | S(eps)=" + (" ".join(f"{v:.2f}" for v in lad) if lad is not None else "-") + f" | mi={mih:.1f}/{mik}")

    # within-arm lambda/retfm relationship (restriction-of-range check on the -0.42)
    say("\nF. TRAJECTORY-LAW DECOMPOSITION (context for the registered Spearman −0.42):")
    alllam, allrf = [], []
    for a in ARMS:
        M = monitors(a)
        lam = [m.get("lam_joint_mean", m.get("lam_max_mean")) for m in M]; rf = [m["ret_final_t8"] for m in M]
        alllam += lam; allrf += rf
        s = spear(lam, rf)
        say(f"    {a:5s} within-arm spearman(lam, retfm) = {f'{s:+.2f}' if s is not None else '  n/a (no variance)'}   retfm range [{min(rf):.2f},{max(rf):.2f}]")
    n_flat = sum(1 for a in ARMS if len(set(round(m['ret_final_t8'],2) for m in monitors(a))) <= 2)
    say(f"    arms with essentially flat retfm (no collapse to correlate): {n_flat}/16 — the global −0.42 is carried by the few collapsing arms")

    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text("\n".join(L) + "\n")
    say(f"\nartifact -> {OUT}")

if __name__ == "__main__":
    main()
