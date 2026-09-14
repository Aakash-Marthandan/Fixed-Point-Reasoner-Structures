#!/usr/bin/env python3
# Ledger: THE C8 EXTENSION — analysis-time DESCRIPTIVE lens (no rules; labeled exploratory; the registered letters are tools/analyze_c8x.py's).
# The PI (2026-09-14): "do proper analyses on these runs to see what we can learn further". Reads the crc-verified pulls only:
#   (A) SELECTION PRECISION — per width-192 seed: the 10k held-out curve's peak and exact plateau membership, the monitor pick's regret,
#       the monitor-vs-held-out rank agreement (Spearman) over all grids and over the late region; the extension decision at 30k replayed
#       under the held-out instrument (did monitor NOISE stop C8, or the rule?); the held-out pick vs the monitor pick on TEST, paired.
#   (B) THE TRAJECTORY — per seed on the held-out set: the early peak, the mid-training collapse (depth, step), the recovery, the late peak.
#   (C) WHAT THE EXTRA TRAINING BOUGHT PER ITERATION — C8 at 22k / 46k / 48k: exact accuracy by step t on the full set (first_exact).
#   (D) ERROR STRUCTURE — at D64 on the full set, per width triple: failures per seed, shared by all three, any seed solves (union), the
#       ratings of shared vs seed-specific failures (puzzle-intrinsic hardness vs seed-specific misses).
#   (E) MEMORIZATION AT 50k (C8) — the 50k grid vs 46k on the identical 50k test subsample (paired), held-out 50k vs its peak, training fit.
#   .venv/bin/python tools/lens_c8x_learn.py [--pf runs/_paperfinal_pull/stage --x runs/_c8x_pull/x/runs --out runs/_c8x_pull/analysis/learn.txt]
#   .venv/bin/python tools/lens_c8x_learn.py --selftest
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_paperfinal import paired                 # frozen at 7a99d9e, selftested (exact McNemar on idx pairing)
from c8x_valcurve import curve, select, plateau       # frozen at 543e7bf, selftested
from lens_width_seed import curve_from_first          # selftested (0-based first_exact)

def recs(d): q = Path(d) / "records_all.npz"; return dict(np.load(q, allow_pickle=True)) if q.exists() else None
def summ(d): q = Path(d) / "summary_all.json"; return json.loads(q.read_text()) if q.exists() else None

def spearman(a, b):
    """Spearman rank correlation with average ranks for ties."""
    def ranks(x):
        x = np.asarray(x, float); order = np.argsort(x, kind="mergesort"); r = np.empty(len(x)); i = 0
        while i < len(x):
            j = i
            while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]: j += 1
            r[order[i:j + 1]] = (i + j) / 2.0; i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    return float(np.corrcoef(ra, rb)[0, 1])

def monitor(pdir):
    mon = {}
    for l in (Path(pdir) / "metrics.jsonl").read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            if "monitor" in r: mon[int(r["monitor"]["step"])] = r["monitor"]
    return mon

def train_fit(pdir, step):
    rows = [json.loads(l) for l in (Path(pdir) / "metrics.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if "loss" in r and "train_exact" in r and abs(r["step"] - step) <= 500]
    return float(np.mean([r["train_exact"] for r in rows])) if rows else None

def two_stage(c, split=30000):
    """The mid-training collapse as the LARGEST DRAWDOWN before `split` (a drop from the running maximum, not the untrained start):
    early peak = the running-max grid preceding the deepest drawdown; late peak = the best grid after the collapse."""
    early = [(s, k) for s, k, _ in c if s <= split]
    if len(early) < 2: return None
    best_dd, pk, lo, run = -1, early[0], early[0], early[0]
    for s, k in early[1:]:
        if k > run[1]: run = (s, k)
        dd = run[1] - k
        if dd > best_dd: best_dd, pk, lo = dd, run, (s, k)
    late = [(s, k) for s, k, _ in c if s > lo[0]]
    lp = max(late, key=lambda t: (t[1], -t[0])) if late else lo
    return dict(early_peak=pk, collapse=lo, late_peak=lp, drawdown=best_dd)

def overlap(fail_sets):
    """fail_sets: list of boolean failure arrays on identical puzzles -> (per-seed failures, all fail, any fail, union solved frac)."""
    F = np.stack(fail_sets); n = F.shape[1]
    return dict(per_seed=[int(f.sum()) for f in F], all_fail=int(F.all(axis=0).sum()), any_fail=int(F.any(axis=0).sum()),
                union_solved=float(1 - F.all(axis=0).mean()), n=n)

def aligned(z, idx_ref):
    o = np.argsort(np.asarray(z["idx"])); zi = np.asarray(z["idx"])[o]
    pos = np.searchsorted(zi, idx_ref); assert np.array_equal(zi[pos], idx_ref)
    return {k: np.asarray(v)[o][pos] for k, v in z.items() if np.asarray(v).shape[:1] == np.asarray(z["idx"]).shape}

def lens(pf, x, out):
    PF, X, L = Path(pf), Path(x), []
    say = lambda s="": (L.append(s), print(s))
    say("THE C8 EXTENSION — what we learn (descriptive lens; exploratory; the registered letters are analyze_c8x.py's)")
    picks = {"C5": 46000, "C7": 46000, "C8": 46000}
    mdirs = {"C5": PF / "pretrainchamp_C5", "C7": PF / "pretrainchamp_C7", "C8": X / "pretrainchamp_C8"}
    say("(A) SELECTION PRECISION on the width-192 recipe (held-out = 10,000 train-file puzzles, D16 EMA; monitor = 512 puzzles)")
    for a in ("C5", "C7", "C8"):
        c = curve(X, a); cm = {s: k for s, k, _ in c}; mx = max(cm.values()); pl = plateau(c)
        mon = monitor(mdirs[a]); steps = [s for s, _, _ in c]
        me = [mon[s]["val_t16_ema"] for s in steps]; ho = [cm[s] for s in steps]
        late = [i for i, s in enumerate(steps) if s >= 32000]
        rho_all, rho_late = spearman(me, ho), spearman([me[i] for i in late], [ho[i] for i in late])
        reg = mx - cm[picks[a]]
        by30 = [(s, k) for s, k, _ in c if s <= 30000]; b30 = max(by30, key=lambda t: (t[1], -t[0]))[0]
        say(f"  {a}: held-out peak {select(c)//1000}k ({100*mx/1e4:.2f}); plateau (<= 50 puzzles below) = {[s//1000 for s in pl]}k; monitor pick 46k = {100*cm[46000]/1e4:.2f} "
            f"(regret {reg} puzzles = {reg/100:.2f} pp); Spearman monitor~held-out: all grids {rho_all:+.2f}, 32k-50k {rho_late:+.2f}; "
            f"extension at 30k under the held-out instrument: argmax by 30k = {b30//1000}k -> {'fires' if b30 >= 26000 else 'does NOT fire'}")
    r16 = paired(recs(X / "c8x_xrow_pchampC8_d16full_s048000"), recs(X / "sxeval_pchampC8" / "full_vsel_t16"))
    r64 = paired(recs(X / "c8x_xrow_pchampC8_d64full_s048000"), recs(X / "filler_sxeval_pchampC8_full_t64"))
    say(f"  C8 on TEST, the held-out pick (48k) minus the monitor pick (46k): D16 full {100*r16['diff']:+.2f} pp ({r16['only_a']}/{r16['only_b']}, p {r16['p']:.1e}); "
        f"D64 full {100*r64['diff']:+.2f} pp ({r64['only_a']}/{r64['only_b']}, p {r64['p']:.1e})")
    say("(B) THE TRAJECTORY on the held-out set (exact of 10,000)")
    for a in ("C5", "C7", "C8"):
        t = two_stage(curve(X, a)); cm = {s: k for s, k, _ in curve(X, a)}
        say(f"  {a}: early peak {t['early_peak'][0]//1000}k {t['early_peak'][1]} -> collapse {t['collapse'][0]//1000}k {t['collapse'][1]} "
            f"(drawdown {t['drawdown']}) -> 30k {cm[30000]} -> late peak {t['late_peak'][0]//1000}k {t['late_peak'][1]} (30k -> peak {t['late_peak'][1]-cm[30000]:+d})")
    say("(C) WHAT THE EXTRA TRAINING BOUGHT PER ITERATION — C8, exact % by step t on all 422,786 (from the D64 records' first_exact)")
    rows = {"22k": PF / "filler_sxeval_pchampC8_full_t64", "46k": X / "filler_sxeval_pchampC8_full_t64", "48k": X / "c8x_xrow_pchampC8_d64full_s048000"}
    cur = {g: curve_from_first(recs(d)["first_exact"], 64) for g, d in rows.items()}
    ts = (1, 2, 4, 8, 16, 32, 64)
    for g in rows:
        f = np.asarray(recs(rows[g])["first_exact"]); ok = f >= 0
        say(f"  C8@{g}: " + "  ".join(f"t{t} {100*cur[g][t-1]:.2f}" for t in ts) + f"  | mean first-exact step {np.mean(f[ok] + 1):.2f}")
    say("  gain 46k - 22k by step: " + "  ".join(f"t{t} {100*(cur['46k'][t-1]-cur['22k'][t-1]):+.2f}" for t in ts))
    say("(D) ERROR STRUCTURE at D64 on the full set (identical 422,786 puzzles)")
    ref = np.sort(np.asarray(recs(PF / "filler_sxeval_pchampC0_full_t64")["idx"]))
    rating = aligned(recs(PF / "filler_sxeval_pchampC0_full_t64"), ref)["rating"]
    for label, dirs in (("width 192 (budget-matched: C5, C7, C8@46k)", [PF / "filler_sxeval_pchampC5_full_t64", PF / "filler_sxeval_pchampC7_full_t64", X / "filler_sxeval_pchampC8_full_t64"]),
                        ("width 192 (registered: C5, C7, C8@22k)", [PF / "filler_sxeval_pchampC5_full_t64", PF / "filler_sxeval_pchampC7_full_t64", PF / "filler_sxeval_pchampC8_full_t64"]),
                        ("width 384 (C0, C1, C2)", [PF / f"filler_sxeval_pchampC{i}_full_t64" for i in (0, 1, 2)])):
        fails = [~aligned(recs(d), ref)["cold_exact"].astype(bool) for d in dirs]
        o = overlap(fails); shared = np.stack(fails).all(axis=0); some = np.stack(fails).any(axis=0) & ~shared
        say(f"  {label}: failures per seed {o['per_seed']}; failed by ALL three {o['all_fail']} ({100*o['all_fail']/o['n']:.3f} %); by at least one {o['any_fail']}; "
            f"any-seed-solves {100*o['union_solved']:.2f} %; share of a seed's failures that all three share {100*o['all_fail']/np.mean(o['per_seed']):.0f} %; "
            f"mean rating: shared failures {rating[shared].mean():.1f}, seed-specific {rating[some].mean():.1f}, all puzzles {rating.mean():.1f}")
    say("(E) MEMORIZATION AT 50k (C8)")
    rf = paired(recs(X / "sxeval_pchampC8" / "full_final_t16"), recs(X / "sxeval_pchampC8" / "full_vsel_t16"))
    c8 = {s: k for s, k, _ in curve(X, "C8")}
    say(f"  the 50k grid minus 46k on the identical 50k test subsample at D16: {100*rf['diff']:+.2f} pp ({rf['only_a']}/{rf['only_b']}, p {rf['p']:.1e}); "
        f"held-out 50k {c8[50000]} vs peak {max(c8.values())} ({c8[50000]-max(c8.values()):+d}); training-batch exact near 30k / 46k / 50k: "
        + " / ".join(f"{train_fit(X / 'pretrainchamp_C8', s):.3f}" for s in (30000, 46000, 50000)))
    Path(out).parent.mkdir(parents=True, exist_ok=True); Path(out).write_text("\n".join(L) + "\n")

def selftest():
    assert abs(spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1) < 1e-12 and abs(spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1) < 1e-12
    assert abs(spearman([1, 1, 2], [1, 1, 2]) - 1) < 1e-12
    c = [(10000, 90, .9), (12000, 95, .95), (14000, 40, .4), (16000, 80, .8), (30000, 92, .92), (40000, 97, .97), (50000, 96, .96)]
    t = two_stage(c)
    assert t["early_peak"] == (12000, 95) and t["collapse"] == (14000, 40) and t["late_peak"] == (40000, 97) and t["drawdown"] == 55, t
    t = two_stage([(10000, 30, .3), (12000, 90, .9), (14000, 70, .7), (16000, 95, .95), (30000, 94, .94), (40000, 99, .99)])   # a low start is not a collapse
    assert t["early_peak"] == (12000, 90) and t["collapse"] == (14000, 70) and t["drawdown"] == 20 and t["late_peak"] == (40000, 99), t
    o = overlap([np.array([1, 1, 0, 0], bool), np.array([1, 0, 1, 0], bool), np.array([1, 0, 0, 0], bool)])
    assert o["per_seed"] == [2, 2, 1] and o["all_fail"] == 1 and o["any_fail"] == 3 and abs(o["union_solved"] - .75) < 1e-12, o
    print("selftest OK: 5/5")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--pf", default="runs/_paperfinal_pull/stage"); ap.add_argument("--x", default="runs/_c8x_pull/x/runs")
    ap.add_argument("--out", default="runs/_c8x_pull/analysis/learn.txt"); ap.add_argument("--selftest", action="store_true"); a = ap.parse_args()
    selftest() if a.selftest else lens(a.pf, a.x, a.out)
