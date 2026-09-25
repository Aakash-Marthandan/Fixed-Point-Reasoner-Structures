#!/usr/bin/env python3
# Ledger: CASCADE OR SIMULTANEOUS REVISION? (2026-09-25; follow-up registration in Documentation/Note_2026-09-25_Inside_Completion.md,
# written before these statistics were computed). DESCRIPTIVE; saved records of lens_inside_completion only; no inference.
# T1: does the order in which cells become correct inside the completing cycle follow singles-propagation depth from the cells
# already correct at readout kappa-1? T2: is a sufficient correct seed present before completion?
"""  .venv/bin/python tools/lens_cascade_tests.py --selftest
  .venv/bin/python tools/lens_cascade_tests.py --report"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "runs/analysis/inside_completion_20260925"
OUT = SRC / "cascade_tests"
MODELS = ("C5", "SA128", "EQR")
OFFSETS = (-1, -2, -4, -6)

UNITS = [[r * 9 + c for c in range(9)] for r in range(9)] + [[r * 9 + c for r in range(9)] for c in range(9)] \
      + [[(br * 3 + i) * 9 + bc * 3 + j for i in range(3) for j in range(3)] for br in range(3) for bc in range(3)]
PEERS = [sorted({p for u in UNITS if c in u for p in u} - {c}) for c in range(81)]
CELL_UNITS = [[k for k, u in enumerate(UNITS) if c in u] for c in range(81)]

# ---------------- pure helpers (selftested) ----------------
def round_singles(known):
    """known (81,) ints, 0 = unknown -> dict cell -> digit for every naked or hidden single in this state."""
    cand = {}
    for c in range(81):
        if known[c] == 0:
            cand[c] = set(range(1, 10)) - {int(known[p]) for p in PEERS[c] if known[p]}
    found = {c: next(iter(s)) for c, s in cand.items() if len(s) == 1}                      # naked singles
    for u in UNITS:
        present = {int(known[c]) for c in u if known[c]}
        for d in range(1, 10):
            if d in present: continue
            places = [c for c in u if known[c] == 0 and d in cand[c]]
            if len(places) == 1: found.setdefault(places[0], d)                                # hidden singles
    return found

def propagate(seed, sol):
    """seed (81,) ints (correct digits or 0) -> depth (81,): 0 seeded, r >= 1 assigned in round r, -1 never. Asserts soundness."""
    known = np.array(seed, dtype=np.int64); depth = np.where(known > 0, 0, -1); r = 0
    while True:
        found = round_singles(known)
        if not found: break
        r += 1
        for c, d in found.items():
            assert d == sol[c], f"unsound deduction at cell {c}: {d} vs {sol[c]}"
            known[c] = d; depth[c] = r
    return depth

def spearman(x, y):
    """Spearman rank correlation with average ranks for ties; NaN if either is constant."""
    def ranks(a):
        a = np.asarray(a, float); o = np.argsort(a, kind="stable"); r = np.empty(len(a)); r[o] = np.arange(len(a))
        for v in np.unique(a):
            m = a == v; r[m] = r[m].mean()
        return r
    rx, ry = ranks(x), ranks(y)
    if rx.std() == 0 or ry.std() == 0: return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])

def first_stable(correct_by_k):
    """correct_by_k (K, 81) bool -> t (81,): the first 1-based k from which the cell stays correct through K; 0 if not correct at K."""
    K = correct_by_k.shape[0]; t = np.zeros(correct_by_k.shape[1], int)
    for c in range(correct_by_k.shape[1]):
        if not correct_by_k[-1, c]: continue
        k = K
        while k > 1 and correct_by_k[k - 2, c]: k -= 1
        t[c] = k
    return t

# ---------------- the analysis ----------------
def analyze(key):
    D = np.load(SRC / f"{key}.npz", allow_pickle=True)
    sol = D["sol"].reshape(-1, 81).astype(np.int64); puz = D["puz"].reshape(-1, 81).astype(np.int64); ng = puz == 0
    idx, kappa = D["cohort"], D["kappa"]; slow = D["slow_digit"].astype(np.int64); sh = D["shadow_digit"].astype(np.int64); fa = D["fast_digit"].astype(np.int64)
    res = dict(key=key, cohort=int(len(idx)))
    # ---- T1
    rho_sh, rho_fa, by_depth, broken, n_W, undeduc = [], [], {}, 0, 0, 0
    for i in idx:
        a = kappa[i] - 1                                                                     # 0-based index of readout kappa
        prev = slow[a - 1, i]; W = ng[i] & (prev != sol[i])
        seed = np.where(~ng[i] | (prev == sol[i]), sol[i], 0)
        depth = propagate(seed, sol[i])
        for arr, store in ((sh, rho_sh), (fa, rho_fa)):
            corr = arr[a, :, i] == sol[i][None]                                              # (6, 81)
            t = first_stable(corr)
            cells = np.flatnonzero(W & (depth > 0) & (t > 0))
            if len(cells) >= 5:
                rr = spearman(depth[cells], t[cells])
                if np.isfinite(rr): store.append(rr)
            if arr is sh:
                for c in np.flatnonzero(W):
                    b = "undeducible" if depth[c] < 0 else (str(depth[c]) if depth[c] < 3 else ">=3")
                    by_depth.setdefault(b, []).append(t[c])
                broken += int((((prev == sol[i]) & ng[i])[None] & ~corr).any(0).sum())
        n_W += int(W.sum()); undeduc += int((W & (depth < 0)).sum())
    mean_t = {b: float(np.mean(v)) for b, v in by_depth.items()}; n_t = {b: len(v) for b, v in by_depth.items()}
    med = float(np.median(rho_sh)) if rho_sh else float("nan")
    rising = all(b in mean_t for b in ("1", "2", ">=3")) and mean_t["1"] < mean_t["2"] < mean_t[">=3"]
    res["T1"] = dict(median_rho_shadow=med, n_puzzles_rho=len(rho_sh), share_rho_positive=float(np.mean(np.asarray(rho_sh) > 0)) if rho_sh else None,
                     median_rho_fast=float(np.median(rho_fa)) if rho_fa else None, mean_t_by_depth=mean_t, n_by_depth=n_t, rising=rising,
                     wrong_cells=n_W, undeducible_cells=undeduc, correct_cells_broken_in_cycle=broken,
                     letter="CASCADE" if (med >= 0.3 and rising) else ("SIMULTANEOUS" if abs(med) < 0.1 else "MIXED"))
    # ---- T2
    p = {}
    for o in OFFSETS:
        ok = []
        for i in idx:
            j = kappa[i] - 1 + o
            if j < 0: continue
            rd = slow[j, i]; seed = np.where(~ng[i] | (rd == sol[i]), sol[i], 0); ok.append(bool((propagate(seed, sol[i]) >= 0).all()))
        p[o] = float(np.mean(ok))
    giv = float(np.mean([bool((propagate(np.where(~ng[i], sol[i], 0), sol[i]) >= 0).all()) for i in idx]))
    lags = []
    for i in idx:
        sig = None
        for j in range(0, kappa[i] - 1):                                                     # readouts 1 .. kappa-1
            rd = slow[j, i]; seed = np.where(~ng[i] | (rd == sol[i]), sol[i], 0)
            if (propagate(seed, sol[i]) >= 0).all(): sig = j + 1; break
        lags.append(kappa[i] - sig if sig is not None else np.nan)
    lags = np.asarray(lags, float); med_lag = float(np.nanmedian(lags)) if np.isfinite(lags).any() else float("nan")
    trig = p[-1] >= 0.7 and p[-4] <= 0.3 and med_lag <= 3
    early = p[-4] >= 0.5 and med_lag > 3
    res["T2"] = dict(p_by_offset={str(k): v for k, v in p.items()}, p_givens=giv, median_lag=med_lag, share_never_sufficient_before=float(np.mean(~np.isfinite(lags))),
                     lag_quartiles=[float(x) for x in np.nanpercentile(lags, [25, 50, 75])] if np.isfinite(lags).any() else None,
                     letter="TRIGGER" if trig else ("SEED EARLY" if early else "MIXED"))
    return res

def report():
    OUT.mkdir(parents=True, exist_ok=True); Ls, J = [], {}; say = lambda s_="": (Ls.append(s_), print(s_))
    say("CASCADE OR SIMULTANEOUS REVISION? (tools/lens_cascade_tests.py; follow-up registration in Note_2026-09-25_Inside_Completion.md). Saved records; no inference.")
    for key in MODELS:
        if not (SRC / f"{key}.npz").exists(): continue
        r = analyze(key); J[key] = r; t1, t2 = r["T1"], r["T2"]
        say(); say(f"== {key} (cohort {r['cohort']})")
        say(f"   T1 order: median Spearman rho(depth, first stable fast update) over {t1['n_puzzles_rho']} puzzles = {t1['median_rho_shadow']:.3f} "
            f"(positive in {100 * t1['share_rho_positive']:.0f} %); direct fast readout {t1['median_rho_fast']:.3f}")
        say("      mean first stable fast update by depth: " + "  ".join(f"{b}: {t1['mean_t_by_depth'][b]:.2f} (n {t1['n_by_depth'][b]})" for b in ("1", "2", ">=3", "undeducible") if b in t1["mean_t_by_depth"]))
        say(f"      wrong cells at kappa-1: {t1['wrong_cells']}; undeducible from the correct cells: {t1['undeducible_cells']}; cells correct at kappa-1 broken inside the cycle: {t1['correct_cells_broken_in_cycle']}")
        say(f"      -> {t1['letter']}")
        say(f"   T2 seed: singles propagation from givens + correct cells solves the puzzle at readout "
            + "  ".join(f"kappa{o:+d}: {t2['p_by_offset'][str(o)]:.2f}" for o in OFFSETS) + f"  | givens alone: {t2['p_givens']:.2f}")
        say(f"      first sufficient readout before completion: median lag {t2['median_lag']:.1f} readouts (quartiles {t2['lag_quartiles']}); never sufficient before: {100 * t2['share_never_sufficient_before']:.0f} %")
        say(f"      -> {t2['letter']}")
    (OUT / "report.txt").write_text("\n".join(Ls) + "\n"); (OUT / "report.json").write_text(json.dumps(J, indent=1, default=float))

# ---------------- selftest ----------------
def _brute_singles(known):
    """An independent, slow single finder (different code path) for the property test."""
    out = {}
    for c in range(81):
        if known[c]: continue
        ok = [d for d in range(1, 10) if all(known[p] != d for p in range(81) if p != c and (p // 9 == c // 9 or p % 9 == c % 9 or (p // 27 == c // 27 and (p % 9) // 3 == (c % 9) // 3)))]
        if len(ok) == 1: out[c] = ok[0]
    for u in UNITS:
        for d in range(1, 10):
            if any(known[c] == d for c in u): continue
            places = [c for c in u if not known[c] and all(known[p] != d for p in PEERS[c])]
            if len(places) == 1: out.setdefault(places[0], d)
    return out

def _checks(prop, spear, fstab, sol):
    full = sol.copy(); assert (prop(full, sol) == 0).all()
    one = sol.copy(); one[40] = 0; d = prop(one, sol); assert d[40] == 1 and (d[np.arange(81) != 40] == 0).all()
    row = sol.copy(); row[9:18] = 0; d = prop(row, sol); assert (d[9:18] == 1).all()
    assert abs(spear([1, 2, 3, 4, 5], [2, 3, 4, 5, 6]) - 1.0) < 1e-12 and abs(spear([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) + 1.0) < 1e-12
    ck = np.array([[0, 1, 0, 1, 1], [1, 1, 0, 1, 0], [1, 1, 1, 0, 1]], bool); assert fstab(ck).tolist() == [2, 1, 3, 0, 3]   # last cell flickers

def selftest():
    D = np.load(SRC / "C5.npz", allow_pickle=True); sols = D["sol"].reshape(-1, 81).astype(np.int64); puz = D["puz"].reshape(-1, 81).astype(np.int64)
    n = 0; sol = sols[0]
    _checks(propagate, spearman, first_stable, sol); n += 1
    rng = np.random.default_rng(7); checked = deeper = 0
    for i in range(12):                                                                      # property test on random correct seeds
        s = sols[i]
        for size in (30, 38, 46):
            seed = np.where(rng.random(81) < size / 81, s, 0); depth = propagate(seed, s)
            known = np.where(depth == 0, s, 0); r = 1
            while True:
                found = _brute_singles(known)
                if not found: break
                assert set(found) == set(np.flatnonzero(depth == r).tolist()), "depth disagrees with the independent single finder"
                for c, d in found.items(): known[c] = d
                r += 1
            assert set(np.flatnonzero(depth < 0).tolist()) == set(np.flatnonzero(known == 0).tolist()); checked += 1; deeper += int((depth >= 2).any())
    assert deeper > 0, "no seed exercised depth >= 2"; n += 1
    d_giv = propagate(np.where(puz[0] > 0, sols[0], 0), sols[0]); assert (d_giv[puz[0] > 0] == 0).all(); n += 1           # givens alone: sound
    mutants = [
        (lambda seed, s: (lambda d: np.where(d > 0, d + 1, d))(propagate(seed, s)), spearman, first_stable),                   # depth off by one
        (propagate, lambda x, y: spearman(x, y[::-1]), first_stable),                                                           # ranks misaligned
        (propagate, spearman, lambda ck: np.array([int(np.argmax(ck[:, c])) + 1 if ck[-1, c] else 0 for c in range(ck.shape[1])])),   # first-ever, not first-stable
    ]
    killed = 0
    for mu in mutants:
        try: _checks(*mu, sol)
        except AssertionError: killed += 1
    assert killed == len(mutants), f"only {killed}/{len(mutants)} mutants killed"; n += 1
    print(f"selftest OK: {n}/{n} ({checked} random seeds checked against the independent single finder; mutants killed {killed}/{len(mutants)})")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true"); ap.add_argument("--report", action="store_true"); a = ap.parse_args()
    if a.selftest: selftest()
    if a.report: report()

if __name__ == "__main__":
    main()
