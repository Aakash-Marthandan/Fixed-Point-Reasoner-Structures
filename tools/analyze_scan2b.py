# Ledger: RUNG 2B RIDER SCANS VERDICT reader (2026-08-31) — standing-rule
# application of the 2026-08-30 pre-data registration; NO new rules here.
#   Registration (ledger "RUNG 2B RIDER SCANS — LAUNCH REGISTRATION"):
#     C3X scan = THE B-M2 ADJUDICATION on the frozen B-bands (analyze_sportBr2b.py
#       R2b-1 p4 read, applied verbatim): p4x@128 >= .85 -> B-M2;
#       >= .8057 + .011 -> B-M1-IMPROVED; else B-M1-FLAT.
#     D4 scan = labeled row, NO rule (informational/carrier-facing).
#     PREDICTIONS locked pre-data: C3X in [.84,.89] @128; D4 in [.76,.84].
#     BOTH named statistics reported: vote@k-incl-cold (summary convention,
#       = cold ∪ draws) AND draw-funnel (records convention, mi_first_hit
#       draws-only) — the records-vs-summary closure (2b verdict) names both.
#   Integrity gates (run BEFORE any band read; hard-fail the verdict on breach):
#     n == 20000 per scan; idx unique == 20000; C3X/D4 idx sets IDENTICAL
#     (subsample_seed 20260822 both -> paired-comparable); protocol fields ==
#     registration (t 64, k 128, split test, seed); vote-convention identity
#     exact_acc_vote == mean(cold ∪ draw-hit) exact on records.
"""
  .venv/bin/python tools/analyze_scan2b.py            # -> runs/analysis/sportBr2b_scan_verdict.txt
  .venv/bin/python tools/analyze_scan2b.py --selftest
"""
from __future__ import annotations
import contextlib, io, json, os, sys, tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(os.environ.get("QHRRN_RUNS", ROOT / "runs"))
OUT = RUNS / "analysis" / "sportBr2b_scan_verdict.txt"
TAG = "sportBr2b"
C3P4_CONST = 0.8057          # rung-2 C3 p4@128 (banked reference, analyzer constant)
C3VB20K_FORECAST = 0.825     # lens-B saturation forecast for C3@20k-ckpt (labeled lower-bound)
BANDS = {"C3X": (0.84, 0.89), "D4": (0.76, 0.84)}   # locked pre-data 2026-08-30
PROTO = dict(t_total=64, k_init=128, subsample=20000, subsample_seed=20260822, split="test")
LINES = []
def say(s=""): LINES.append(str(s)); print(s)
def fpp(x): return "  -  " if x is None else f"{100*x:6.2f}"

def load(arm):
    d = RUNS / f"sxscan_p{TAG}{arm}"
    s = json.loads((d / "summary_all.json").read_text())
    z = np.load(d / "records_all.npz")
    nsh = len(list(d.glob("summary_s[0-9].json")))
    return s, z, nsh

def integrity(arm, s, z, nsh):
    errs = []
    if s["n"] != 20000: errs.append(f"n={s['n']} != 20000")
    for k, v in PROTO.items():
        if s.get(k) != v: errs.append(f"{k}={s.get(k)} != {v}")
    idx = z["idx"]
    if len(np.unique(idx)) != 20000: errs.append(f"idx unique {len(np.unique(idx))} != 20000")
    if nsh != 8: errs.append(f"shard summaries {nsh} != NSH 8")
    # vote-convention identity: summary vote == cold ∪ draw-hit on records
    hit = (z["mi_first_hit"] >= 0) | z["cold_exact"]
    if abs(hit.mean() - s["exact_acc_vote"]) > 5e-6:
        errs.append(f"vote identity breach: records {hit.mean():.5f} vs summary {s['exact_acc_vote']:.5f}")
    if abs(z["cold_exact"].mean() - s["exact_acc"]) > 5e-6:
        errs.append(f"cold identity breach: records {z['cold_exact'].mean():.5f} vs summary {s['exact_acc']:.5f}")
    if s.get("valid_wrong_frac", 0) != 0.0: errs.append(f"valid_wrong {s['valid_wrong_frac']} != 0")
    return errs

def analyze():
    LINES.clear(); V = {}
    say("=" * 112)
    say("RUNG 2B RIDER SCANS VERDICT — C3X + D4 one-shot 20k breadth scans (registration 2026-08-30, pre-data)")
    say("=" * 112)
    data, bad = {}, []
    for arm in ("C3X", "D4"):
        s, z, nsh = load(arm)
        errs = integrity(arm, s, z, nsh)
        data[arm] = (s, z)
        say(f"INTEGRITY {arm}: n {s['n']} | idx unique {len(np.unique(z['idx']))} | shards {nsh}/8 | "
            f"seed {s['subsample_seed']} | t {s['t_total']} k {s['k_init']} | valid_wrong {s['valid_wrong_frac']:.4f} | "
            + ("PASS" if not errs else "FAIL: " + "; ".join(errs)))
        bad += [f"{arm}:{e}" for e in errs]
    sC, zC = data["C3X"]; sD, zD = data["D4"]
    if set(zC["idx"].tolist()) != set(zD["idx"].tolist()):
        bad.append("C3X/D4 idx sets differ (not paired-comparable)")
    say(f"INTEGRITY pair: C3X/D4 idx sets identical -> {'PASS' if set(zC['idx'].tolist()) == set(zD['idx'].tolist()) else 'FAIL'}")
    if bad:
        V["INTEGRITY"] = "FAIL"; say("\nVERDICT WITHHELD — integrity breach: " + " | ".join(bad))
        OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text("\n".join(LINES) + "\n")
        return V
    V["INTEGRITY"] = "PASS"

    say("\nSECTION 1 — the two named statistics per arm (both conventions of record)")
    stats = {}
    for arm in ("C3X", "D4"):
        s, z = data[arm]
        vote = s["exact_acc_vote"]                      # vote@k-incl-cold (cold ∪ draws)
        draw = float((z["mi_first_hit"] >= 0).mean())   # draw-funnel (draws-only)
        cold = s["exact_acc"]
        cold_only = int((z["cold_exact"] & (z["mi_first_hit"] < 0)).sum())
        stats[arm] = dict(vote=vote, draw=draw, cold=cold)
        say(f"  {arm:3s} vote@128-incl-cold {fpp(vote)} | draw-funnel {fpp(draw)} | cold {fpp(cold)} | "
            f"cold-only extras {cold_only} puzzles ({fpp(cold_only/20000)}pp of the gap)")
        say(f"       vote curve k=1..128: " + "  ".join(f"{k}={fpp(v)}" for k, v in sorted(s["vote_at_k"].items(), key=lambda kv: int(kv[0]))))
        lo, hi = BANDS[arm]
        V[f"{arm}-PRED"] = "HIT" if lo <= vote <= hi else ("ABOVE" if vote > hi else "MISS-BELOW")
        say(f"       prediction [{100*lo:.0f},{100*hi:.0f}] -> {fpp(vote)} = {V[f'{arm}-PRED']}")

    # ---- THE B-M2 ADJUDICATION (C3X only; frozen R2b-1 p4 bands, verbatim) ----
    p4v = stats["C3X"]["vote"]
    V["B-BAND"] = "B-M2" if p4v >= .85 else ("B-M1-IMPROVED" if p4v >= C3P4_CONST + .011 else "B-M1-FLAT")
    say(f"\nB-M2 ADJUDICATION (C3X, the registered goal-metric read): p4x@128 {fpp(p4v)} vs bands"
        f" [B-M2 >= 85.00 | B-M1-IMPROVED >= {100*(C3P4_CONST+.011):.2f}] -> {V['B-BAND']}")
    say(f"  draw-funnel convention: {fpp(stats['C3X']['draw'])} -> {'also >= .85' if stats['C3X']['draw'] >= .85 else 'below .85 (convention-sensitivity NAMED)'}")
    say(f"  vs rung-2 carrier C3 p4@128 {fpp(C3P4_CONST)}: {'+' if p4v > C3P4_CONST else ''}{100*(p4v-C3P4_CONST):.2f}pp"
        f" | vs lens-B attempts-only forecast {fpp(C3VB20K_FORECAST)} (C3@20k plateau): +{100*(p4v-C3VB20K_FORECAST):.2f}pp -> the rho-via-training route, measured")
    V["D4-ROW"] = f"labeled {fpp(stats['D4']['vote']).strip()}"

    say("\nSECTION 2 — descriptive (no rules): paired D4-vs-C3X on the identical 20k set")
    order_C = np.argsort(zC["idx"]); order_D = np.argsort(zD["idx"])
    cC, cD = zC["cold_exact"][order_C], zD["cold_exact"][order_D]
    vC = (zC["mi_first_hit"][order_C] >= 0) | cC
    vD = (zD["mi_first_hit"][order_D] >= 0) | cD
    say(f"  cold : D4 {fpp(cD.mean())} vs C3X {fpp(cC.mean())} (+{100*(cD.mean()-cC.mean()):.2f}pp) | McNemar D4-only {int((cD & ~cC).sum())} / C3X-only {int((cC & ~cD).sum())}")
    say(f"  vote : C3X {fpp(vC.mean())} vs D4 {fpp(vD.mean())} (+{100*(vC.mean()-vD.mean()):.2f}pp) | McNemar C3X-only {int((vC & ~vD).sum())} / D4-only {int((vD & ~vC).sum())}")
    say(f"  union: cold {fpp((cC | cD).mean())} | vote {fpp((vC | vD).mean())} | vote-core unsolved-by-both {fpp((~vC & ~vD).mean())}")
    for arm, s in (("C3X", sC), ("D4", sD)):
        say(f"  {arm} hardest rating bin (48-386): vote {fpp(s['by_rating_bin_vote'][7])} cold {fpp(s['by_rating_bin'][7])}"
            f" | mean_first_exact {s['mean_first_exact']:.2f}")

    say(); say("SUMMARY: " + "  ".join(f"{k}={v}" for k, v in V.items()))
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text("\n".join(LINES) + "\n"); say(f"artifact -> {OUT}")
    return V


# ---------- self-test ----------
def _mk(root, arm, *, vote_target, cold_frac, n=20000, seed=20260822, break_=None):
    d = root / f"sxscan_p{TAG}{arm}"; d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7 if arm == "C3X" else 8)
    idx = np.arange(n, dtype=np.int64)
    cold = np.zeros(n, bool); cold[: int(cold_frac * n)] = True
    hit = np.full(n, -1, np.int64)
    n_hit = int(vote_target * n) - 12          # 12 cold-only extras by construction
    hit[:n_hit] = rng.integers(0, 128, n_hit)
    cold[n_hit : n_hit + 12] = True            # cold-only (no draw hit)
    vote = float(((hit >= 0) | cold).mean())
    if break_ == "n": n_sum = n - 1
    else: n_sum = n
    if break_ == "dupidx": idx[0] = idx[1]
    s = dict(n=n_sum, t_total=64, k_init=128, subsample=20000, subsample_seed=seed, split="test",
             exact_acc=float(cold.mean()), exact_acc_vote=(vote if break_ != "identity" else vote - 0.01),
             vote_at_k={str(k): vote * (0.4 + 0.6 * i / 7) for i, k in enumerate([1, 2, 4, 8, 16, 32, 64, 128])},
             valid_wrong_frac=0.0, mean_first_exact=10.0,
             by_rating_bin=[None] + [cold_frac] * 7, by_rating_bin_vote=[None] + [vote] * 7)
    s["vote_at_k"]["128"] = vote
    (d / "summary_all.json").write_text(json.dumps(s))
    np.savez(d / "records_all.npz", idx=idx, cold_exact=cold, mi_first_hit=hit,
             rating=np.ones(n, np.int64), first_exact=hit, first_valid=hit,
             violations=np.zeros(n, np.int64), cells=np.full(n, 81, np.int64),
             givens_kept=np.full(n, 81, np.int64), mi_verified=hit, mi_true=hit)
    for k in range(8): (d / f"summary_s{k}.json").write_text("{}")

def selftest():
    checks = []
    def run(build):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); build(root)
            globals()["RUNS"] = root; globals()["OUT"] = root / "a" / "v.txt"
            with contextlib.redirect_stdout(io.StringIO()): return analyze()
    def wA(r):  # B-M2 world: C3X .888, D4 .812
        _mk(r, "C3X", vote_target=.888, cold_frac=.246); _mk(r, "D4", vote_target=.812, cold_frac=.333)
    v = run(wA)
    checks += [("A integrity PASS", v["INTEGRITY"] == "PASS"),
               ("A B-M2", v["B-BAND"] == "B-M2"),
               ("A C3X pred HIT", v["C3X-PRED"] == "HIT"),
               ("A D4 pred HIT", v["D4-PRED"] == "HIT")]
    def wB(r):  # improved-not-B-M2 world + D4 below band
        _mk(r, "C3X", vote_target=.83, cold_frac=.24); _mk(r, "D4", vote_target=.74, cold_frac=.30)
    v = run(wB)
    checks += [("B band improved", v["B-BAND"] == "B-M1-IMPROVED"),
               ("B C3X pred MISS-BELOW", v["C3X-PRED"] == "MISS-BELOW"),
               ("B D4 pred MISS-BELOW", v["D4-PRED"] == "MISS-BELOW")]
    def wC(r):  # flat world, above-band D4
        _mk(r, "C3X", vote_target=.808, cold_frac=.24); _mk(r, "D4", vote_target=.85, cold_frac=.33)
    v = run(wC)
    checks += [("C band flat", v["B-BAND"] == "B-M1-FLAT"), ("C D4 ABOVE", v["D4-PRED"] == "ABOVE")]
    for br, name in (("n", "n-gate"), ("dupidx", "dup-idx"), ("identity", "vote-identity")):
        def wX(r, br=br):
            _mk(r, "C3X", vote_target=.888, cold_frac=.246, break_=br); _mk(r, "D4", vote_target=.812, cold_frac=.333)
        v = run(wX)
        checks.append((f"X {name} breach -> WITHHELD", v["INTEGRITY"] == "FAIL" and "B-BAND" not in v))
    ok = 0
    for name, passed in checks:
        print(("  PASS  " if passed else "  FAIL  ") + name); ok += bool(passed)
    print(f"selftest: {ok}/{len(checks)}")
    return ok == len(checks)


if __name__ == "__main__":
    if "--selftest" in sys.argv: sys.exit(0 if selftest() else 1)
    analyze()
