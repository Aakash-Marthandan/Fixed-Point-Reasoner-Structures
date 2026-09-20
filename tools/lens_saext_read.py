#!/usr/bin/env python3
"""THE ATTENTION ARMS AT THE PAPER'S BUDGET — a descriptive read (2026-09-20; EXPLORATORY: written AFTER the frozen letters, registers nothing,
moves no letter). It reads what the letters do not: the per-iteration curve of each extended arm on the SAME full test set SE-RRM reports on,
beside their published curve and our width-192 model's; and the selected-vs-final-vs-raw spread.

  T1 ACCURACY BY ITERATION on all 422,786 (the D64 full row's packed exact bit per step; the bit order verified against the row's own cold_exact).
  T2 THE SELECTED GRID against the final grid and the raw (non-EMA) weights, and where the selection sits in the run.
  T3 RESTARTS on the identical 5,000 (the k32 and k128 scans, 64 iterations, EMA): the fixed start, one random start, the residual-SELECTED
     draw as restarts are added (the reader's t1r_at_k), and VERIFIED (any draw exact) — the paper's restart column, beside its k128 triple.

  .venv/bin/python tools/lens_saext_read.py --root runs/_saext_pull/stage/runs --out runs/_saext_pull/analysis
  .venv/bin/python tools/lens_saext_read.py --selftest
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lens_wladder_read import unpack_steps          # the packed per-step bits (order verified against the row's own exact bit)
ARMS = ("SA128", "SA192", "SA256")
ITS = (1, 2, 4, 8, 16, 32, 64)
SERRM = {1: .1605, 2: .6206, 4: .7731, 8: .8738, 16: .9373, 32: .9682, 64: .9822}   # their Table 2 (main text, RoPE2D; one run, the full test set)
W192 = {1: .1705, 2: .4571, 4: .7265, 8: .8931, 16: .9591, 32: .9824, 64: .9910}    # our width-192 seed 0 (champ_physics §G; the same full set)
LINES = []
def say(s=""): LINES.append(str(s)); print(s)
def pct(x): return "  n/a" if x is None else f"{100 * x:6.2f}"


def by_iteration(root, arm):
    p = Path(root) / f"filler_sxeval_pchamp{arm}_full_t64" / "records_all.npz"
    if not p.exists(): return None, None
    z = dict(np.load(p, allow_pickle=True)); b, order, base = unpack_steps(z, 64)
    return (None, None) if b is None else ({t: float(b[:, t - 1].mean()) for t in ITS}, len(z["idx"]))


K128_TRIPLE_SEL, K128_TRIPLE_VER, EQR_K128 = 0.9985, 0.9989, 0.9884   # the paper's width-192 k128 triple (selected / verified) and EqR's k128 (the pending runs' verdict)


def scan_row(root, d):
    """(fixed start, one random start, {k: residual-selected}, verified) of a banked scan summary."""
    p = Path(root) / d / "summary_all.json"
    if not p.exists(): return None
    s = json.loads(p.read_text()); t1 = s.get("t1r_at_k") or {}
    return dict(fixed=s.get("exact_acc"), b1=s.get("b1_exact"), sel={int(k): v for k, v in t1.items()}, ver=s.get("exact_acc_vote"), n=s.get("n"), k=s.get("k_init"))


def report(root, out):
    say("THE ATTENTION ARMS AT THE PAPER'S BUDGET — a descriptive read (EXPLORATORY; after the frozen letters; nothing here is a letter)")
    say("T1 accuracy by iteration on ALL 422,786 test puzzles (the fixed start, EMA weights, the selected grid)")
    say(f"  {'model':26s}" + " ".join(f"{t:>7d}" for t in ITS))
    for a in ARMS:
        c, n = by_iteration(root, a)
        say(f"  {a + ' (50k budget)':26s}" + (" ".join(pct(c[t]) + " " for t in ITS) if c else "  per-step bits absent or inconsistent") + (f"   n {n}" if c else ""))
    say(f"  {'our width-192, seed 0':26s}" + " ".join(pct(W192[t]) + " " for t in ITS) + "   n 422786")
    say(f"  {'SE-RRM published':26s}" + " ".join(pct(SERRM[t]) + " " for t in ITS) + "   n 422786 (their one run, main text)")
    say("T2 the selected grid against the final grid and the raw (non-EMA) weights")
    for a in ARMS:
        g = lambda d: (json.loads((Path(root) / d / "summary_all.json").read_text()) if (Path(root) / d / "summary_all.json").exists() else {})
        vb = (Path(root) / f"pretrainchamp_{a}" / "val_best.txt").read_text().split()[0]
        res = (Path(root) / f"pretrainchamp_{a}" / "resumes.txt").read_text().split()
        say(f"  {a:6s} selected {vb} (resumes at {', '.join(res) or 'none'}) | selected {pct(g(f'sxeval_pchamp{a}/full_vsel_t16').get('exact_acc'))}"
            f"  final grid {pct(g(f'sxeval_pchamp{a}/full_final_t16').get('exact_acc'))} (50k sub)  raw weights {pct(g(f'sxeval_pchamp{a}/full_vsel_t16_alt').get('exact_acc'))} (50k sub)")
    say("T3 restarts on the identical 5,000 (64 iterations, EMA weights, the selected grid): the fixed start, ONE random start, the residual-selected draw at k, and verified (any draw exact)")
    say(f"  {'model / scan':22s} {'fixed':>7s} {'1 start':>8s} {'sel k1':>7s} {'sel k4':>7s} {'sel k8':>7s} {'sel k32':>8s} {'sel k128':>9s} {'verified':>9s}")
    for a_ in ARMS:
        for tag, d in ((f"{a_} k32", f"sxscan_pchamp{a_}"), (f"{a_} k128", f"filler_sxscan128_pchamp{a_}")):
            r = scan_row(root, d)
            if not r: say(f"  {tag:22s} absent"); continue
            g = lambda k: pct(r["sel"].get(k))
            say(f"  {tag:22s} {pct(r['fixed'])} {pct(r['b1'])} {g(1)} {g(4)} {g(8)} {g(32)} {g(128)} {pct(r['ver'])}")
    say(f"  {'width-192 k128 triple':22s} {'':>7s} {'':>8s} {'':>7s} {'':>7s} {'':>7s} {'':>8s} {pct(K128_TRIPLE_SEL):>9s} {pct(K128_TRIPLE_VER):>9s}   (the paper's restart column; EqR k128 selected {pct(EQR_K128)})")
    Path(out).mkdir(parents=True, exist_ok=True); (Path(out) / "descriptive_read.txt").write_text("\n".join(LINES) + "\n")


def selftest():
    n = 0
    bits = np.zeros((4, 64), bool); bits[0, 3:] = True; bits[1, 63] = True; bits[2, :] = True
    z = dict(exact_by_step=np.packbits(bits, axis=1, bitorder="little"), cold_exact=bits[:, -1], first_exact=np.array([3, 63, 0, -1]), idx=np.arange(4))
    b, order, base = unpack_steps(z, 64)
    assert order == "little" and base == 0 and abs(b[:, 3].mean() - .5) < 1e-12 and abs(b[:, 0].mean() - .25) < 1e-12 and abs(b[:, 63].mean() - .75) < 1e-12; n += 1
    assert pct(None).strip() == "n/a" and pct(0.95452).strip() == "95.45"; n += 1
    assert SERRM[16] == .9373 and SERRM[64] == .9822 and W192[16] == .9591 and set(ITS) <= set(SERRM); n += 1
    import tempfile, os
    with tempfile.TemporaryDirectory() as t_:
        d = Path(t_) / "sxscan_pchampSAx"; d.mkdir()
        (d / "summary_all.json").write_text(json.dumps(dict(exact_acc=.99, b1_exact=.98, t1r_at_k={"1": .98, "32": .995}, exact_acc_vote=.999, n=5000, k_init=32)))
        r = scan_row(Path(t_), "sxscan_pchampSAx")
        assert r["fixed"] == .99 and r["b1"] == .98 and r["sel"][32] == .995 and r["ver"] == .999 and r["n"] == 5000 and scan_row(Path(t_), "nope") is None; n += 1
    assert (K128_TRIPLE_SEL, K128_TRIPLE_VER, EQR_K128) == (0.9985, 0.9989, 0.9884); n += 1
    print(f"selftest OK: {n}/{n}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--root"); ap.add_argument("--out"); ap.add_argument("--selftest", action="store_true"); a = ap.parse_args()
    selftest() if a.selftest else report(a.root, a.out)
