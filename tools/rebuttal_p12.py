#!/usr/bin/env python3
# Ledger: DISCUSSION-PERIOD EXPERIMENT P12 (registration Documentation/Note_2026-10-01_Rebuttal_P12_Registration.md, written before any
# row). MEASUREMENT, $0, the Mac's CPU, inference only. Hidden-state nudges as test-time exploration: on 256 test puzzles Attention 128 had
# not solved at 16 iterations (the manuscript's full-test record), continue to 32 iterations intact, with a sham, with ONE 15-degree
# score-preserving rotation of the readout-invisible slow state before iteration 17, or with ONE norm-matched random replacement there;
# beside them a fresh Gaussian restart run for 16 iterations. Uses the other session's P3 runner (tools/rebuttal_p3.R5) unchanged.
"""  JAX_PLATFORMS=cpu .venv/bin/python tools/rebuttal_p12.py --selftest
  JAX_PLATFORMS=cpu nice -n 10 .venv/bin/python tools/rebuttal_p12.py --run
  .venv/bin/python tools/rebuttal_p12.py --report"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))            # NOT the repo src: the P3/P1-P2 runner puts the RELEASE src first (its loader reads .npz checkpoints)
OUT = ROOT / "runs/analysis/rebuttal_20261001j"
REC16 = ROOT / "paper/code/evidence/benchmark/attention_128_d16.npz"
REC64 = ROOT / "paper/code/evidence/benchmark/attention_128_d64.npz"
DATA = ROOT / "data/sudoku_extreme/sudoku_extreme_seed0_mon512.npz"
N_POOL, POOL_SEED, WIDTH, BATCH, STEPS = 256, [20261001, 12], 128, 128, 32
ARMS = ("intact", "sham_t17", "rot15_t17", "random_t17")
D_MIN, P_MAX = 0.05, 0.01

def utc(): return datetime.now(timezone.utc).isoformat()

# ---------------- pure helpers (selftested) ----------------
def pool_ids(cold16, idx, n=N_POOL, seed=POOL_SEED):
    """Uniform sample (sorted) of test indices unsolved at 16 in the saved full-test record."""
    cand = idx[~cold16]
    return np.sort(np.random.default_rng(seed).choice(cand, size=n, replace=False))

def mcnemar_exact(b, c):
    """Two-sided exact McNemar p for discordant counts b (arm-only successes) and c (intact-only successes)."""
    n = b + c
    if n == 0: return 1.0
    k = min(b, c); p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)

def letter(delta, p):
    if delta >= D_MIN and p < P_MAX: return "NUDGE-HELPS"
    if delta <= -D_MIN and p < P_MAX: return "NUDGE-HURTS"
    return "NEUTRAL"

# ---------------- the run ----------------
def run(log=print):
    import rebuttal_p3 as P3
    OUT.mkdir(parents=True, exist_ok=True)
    r16, r64 = np.load(REC16, allow_pickle=False), np.load(REC64, allow_pickle=False)
    assert np.array_equal(r16["idx"], r64["idx"])
    ids = pool_ids(r16["cold_exact"], r16["idx"])
    pos = np.searchsorted(r16["idx"], ids); trapped = ~r64["cold_exact"][pos]
    R = P3.R5(WIDTH, OUT); t0 = time.time()                              # construct first: it sets the release import path
    with np.load(DATA, allow_pickle=False) as d: puz, sol = d["test_q"][ids].astype(np.int32), d["test_a"][ids].astype(np.int32)
    shp = tuple(np.asarray(R.z0).shape)
    res = {}; extras = {}
    for arm in ARMS + ("restart",):
        dst = OUT / f"{arm}.npz"
        if dst.exists(): log(f"SKIP {arm}"); continue
        exact = []; ex_extra = []
        for b in range(0, N_POOL, BATCH):
            sl = slice(b, b + BATCH); ii, pp, ss = ids[sl], puz[sl], sol[sl]
            if arm == "restart":
                z = np.stack([R.EV.mi_z0(4242, int(i), 1, shp, 1.0, "gauss") for i in ii]).astype(np.float32)
                vals, _, ext = R.trajectory_edit(pp, ii, initial=R.jnp.asarray(z), edit=None, steps=16)
            else:
                vals, _, ext = R.trajectory_edit(pp, ii, edit=None if arm == "intact" else arm, steps=STEPS)
            exact.append((vals["pred"].reshape(vals["pred"].shape[0], len(ii), 81) == ss.reshape(len(ii), 81)[None]).all(-1))
            ex_extra.append(ext)
            log(f"  {arm} batch {b // BATCH + 1}/{N_POOL // BATCH} ({time.time() - t0:.0f}s)")
        ex = np.concatenate(exact, axis=1)
        np.savez_compressed(OUT / f"{arm}.tmp.npz", exact=ex, ids=ids, trapped=trapped, extra=json.dumps(ex_extra, default=float))
        os.replace(OUT / f"{arm}.tmp.npz", dst)
    (OUT / "meta.json").write_text(json.dumps(dict(created=utc(), wall=round(time.time() - t0, 1), n=N_POOL, width=WIDTH, steps=STEPS,
                                                  registration="Documentation/Note_2026-10-01_Rebuttal_P12_Registration.md"), indent=1))
    log(f"DONE {round(time.time() - t0)}s")

# ---------------- the report ----------------
def report():
    L = {a: np.load(OUT / f"{a}.npz", allow_pickle=True) for a in ARMS + ("restart",) if (OUT / f"{a}.npz").exists()}
    Ls = []; say = lambda s_="": (Ls.append(s_), print(s_)); JS = dict(created=utc())
    I = L["intact"]["exact"]; trapped = L["intact"]["trapped"]; n = I.shape[1]
    U = ~I[15]                                                                 # unsolved at iteration 16 on THIS run
    say(f"P12 report ({JS['created']}); pool {n}; unsolved at 16 on this run {int(U.sum())} (record said all {n}); trapped labels {int(trapped.sum())}")
    if "sham_t17" in L:
        g1 = bool(np.array_equal(L["sham_t17"]["exact"], I)); JS["gate1_sham"] = g1; say(f"gate 1 (sham reproduces intact flags at every iteration): {g1}")
    shifts = []
    for a in ("rot15_t17", "random_t17"):
        if a in L:
            for e in json.loads(str(L[a]["extra"])): shifts.append(float(e.get("realized_float32_score_shift", e.get("max_score_shift", 0.0)) or 0.0))
    JS["gate2_max_shift"] = max(shifts) if shifts else None; say(f"gate 2 (max realized score shift <= 1e-3): {JS['gate2_max_shift']}")
    s32_int = I[31] & U; JS["intact_gain_16_32"] = int(s32_int.sum())
    say(f"intact: solved by 32 among U = {int(s32_int.sum())} / {int(U.sum())} ({s32_int.sum() / max(U.sum(), 1):.3f})")
    JS["arms"] = {}
    for a in ("rot15_t17", "random_t17"):
        if a not in L: continue
        A = L[a]["exact"]; sa = A[31] & U
        b = int((sa & ~s32_int).sum()); c = int((~sa & s32_int).sum()); delta = (int(sa.sum()) - int(s32_int.sum())) / max(int(U.sum()), 1)
        p = mcnemar_exact(b, c); Lt = letter(delta, p)
        tr = trapped & U; tn = ~trapped & U
        JS["arms"][a] = dict(solved32=int(sa.sum()), arm_only=b, intact_only=c, delta=delta, p=p, letter=Lt,
                             trapped=dict(n=int(tr.sum()), arm=int((sa & tr).sum()), intact=int((s32_int & tr).sum())),
                             transient=dict(n=int(tn.sum()), arm=int((sa & tn).sum()), intact=int((s32_int & tn).sum())))
        say(f"{a}: solved by 32 among U {int(sa.sum())} vs intact {int(s32_int.sum())}; arm-only {b}, intact-only {c}; delta {delta:+.3f}; McNemar p {p:.3g} -> {Lt}")
        say(f"   trapped (unsolved at 64 in the record): arm {int((sa & tr).sum())} vs intact {int((s32_int & tr).sum())} of {int(tr.sum())}; "
            f"transient: arm {int((sa & tn).sum())} vs intact {int((s32_int & tn).sum())} of {int(tn.sum())}")
    if "restart" in L:
        Rr = L["restart"]["exact"]; sr = Rr[15] & U
        JS["restart"] = dict(solved16=int(sr.sum()), union_with_intact32=int((sr | s32_int).sum()), restart_only=int((sr & ~s32_int).sum()),
                             trapped=int((sr & trapped & U).sum()))
        say(f"restart (fresh Gaussian start, 16 iterations): solves {int(sr.sum())} of U (intact's extra 16 iterations solve {int(s32_int.sum())}); "
            f"union {int((sr | s32_int).sum())}; restart-only {int((sr & ~s32_int).sum())}; trapped solved by restart {int((sr & trapped & U).sum())}")
    (OUT / "report.txt").write_text("\n".join(Ls) + "\n"); (OUT / "report.json").write_text(json.dumps(JS, indent=1, default=float))

def selftest():
    n = 0
    cold = np.array([True, False, False, True, False, False]); idx = np.array([10, 11, 12, 13, 14, 15])
    p = pool_ids(cold, idx, n=3, seed=[1, 2]); assert set(p) <= {11, 12, 14, 15} and len(p) == 3 and np.all(np.diff(p) > 0); n += 1
    assert abs(mcnemar_exact(0, 0) - 1.0) < 1e-12 and mcnemar_exact(10, 0) < 0.01 and abs(mcnemar_exact(5, 5) - 1.0) < 1e-12; n += 1
    assert letter(0.05, 0.009) == "NUDGE-HELPS" and letter(0.05, 0.01) == "NEUTRAL" and letter(-0.06, 0.001) == "NUDGE-HURTS" and letter(0.04, 1e-6) == "NEUTRAL"; n += 1
    print(f"selftest OK ({n} checks)")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true"); ap.add_argument("--run", action="store_true"); ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.selftest: selftest()
    if a.run: run()
    if a.report: report()

if __name__ == "__main__":
    main()
