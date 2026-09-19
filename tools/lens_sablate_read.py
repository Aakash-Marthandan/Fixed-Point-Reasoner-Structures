#!/usr/bin/env python3
"""THE RECIPE ABLATION — a descriptive read (2026-09-19; EXPLORATORY: written AFTER the frozen analyzer's letters, registers nothing, moves no
letter).  It asks what the letters cannot: WHY the optimizer arm falls, and how the three arms look next to SE-RRM's published curve.

  T1 THE VALIDATION TRACE BY STEPS AND BY TRAINING ROWS  SA256O holds the steps at 30,000 with a third of the batch, so it also sees 2.8x fewer
     rows (registered as a confound).  This table reads each arm's 512-puzzle validation (EMA weights, 16 iterations) against the reference's at
     MATCHED ROWS (rows = steps x batch; the reference linearly interpolated between its 2,000-step grids).  If the optimizer arm is below the
     reference at the same number of rows, fewer rows alone do not explain the fall.
  T2 THE TRAIN SIDE at the end of training (the last 20 logged rows averaged): loss, train exact, the raw-weights validation beside the EMA's.
  T3 ACCURACY BY ITERATION on the identical 5,000 (the rows' packed exact bit per step), beside SE-RRM's published curve (full test set, one run).
  T4 THE SCAN: fixed start, one random start, the residual-selected accuracy as restarts are added (k 1 / 8 / 32), verified, the spurious share.

  .venv/bin/python tools/lens_sablate_read.py --root runs/_sablate_pull/stage/runs --out runs/_sablate_pull/analysis
  .venv/bin/python tools/lens_sablate_read.py --selftest
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_paperfinal as PF            # spurious
from lens_wladder_read import unpack_steps   # the packed per-step bits, order verified against the row's own exact bit
ARMS = ("SA256L", "SA256S", "SA256O"); REF = "SA256"
ITS = (1, 2, 4, 8, 16, 32, 64)
SERRM = {1: .1605, 2: .6206, 4: .7731, 8: .8738, 16: .9373, 32: .9682, 64: .9822}     # their Table 2 (memory serrm-comparison; read at the source 2026-09-18)
LINES = []
def say(s=""): LINES.append(str(s)); print(s)
def pct(x): return "  n/a" if x is None else f"{100 * x:6.2f}"


def metrics(root, arm):
    p = root / f"pretrainchamp_{arm}" / "metrics.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def trace(M, key="val_t16_ema"):
    """{step: value} of the monitor rows (the LAST row wins on a repeated step)."""
    return {int(r["monitor"]["step"]): float(r["monitor"][key]) for r in M if isinstance(r.get("monitor"), dict) and key in r["monitor"]}


def at_rows(tr, batch, rows):
    """the trace's value at a number of training rows, linear between its grids; None outside them."""
    xs = sorted(tr); X = np.array([s * batch for s in xs], float); Y = np.array([tr[s] for s in xs], float)
    return None if (not len(xs) or rows < X[0] or rows > X[-1]) else float(np.interp(rows, X, Y))


def batch_of(root, arm):
    c = json.loads((root / f"pretrainchamp_{arm}" / "config.json").read_text()); return int(c["argv"]["batch"])


def by_iteration(root, arm):
    p = root / f"sxeval_pchamp{arm}" / "sub5k_vsel_t64" / "records_all.npz"
    if not p.exists(): return None
    z = dict(np.load(p, allow_pickle=True)); b, order, base = unpack_steps(z, 64)
    return None if b is None else {t: float(b[:, t - 1].mean()) for t in ITS}


def report(root, out):
    say("THE RECIPE ABLATION — a descriptive read (EXPLORATORY; after the frozen letters; nothing here is a letter)")
    T = {a: trace(metrics(root, a)) for a in (REF,) + ARMS}; B = {a: batch_of(root, a) for a in (REF,) + ARMS}
    say(f"T1 validation (512 puzzles, EMA weights, 16 iterations) by step; batch: " + ", ".join(f"{a} {B[a]}" for a in T))
    steps = sorted(set().union(*[set(t) for t in T.values()]))
    say(f"  {'step':>6s} " + " ".join(f"{a:>8s}" for a in T) + f" | {'SA256O rows (M)':>15s} {'SA256 at those rows':>20s} {'difference':>10s}")
    for s in steps:
        rows = s * B["SA256O"]; ref = at_rows(T[REF], B[REF], rows); o = T["SA256O"].get(s)
        say(f"  {s:6d} " + " ".join(pct(T[a].get(s)) + "  " for a in T) + f" | {rows / 1e6:15.2f} {pct(ref):>20s} {'' if (ref is None or o is None) else f'{100 * (o - ref):+10.2f}'}")
    for a in T:
        if T[a]: m = max(T[a].values()); sm = min(s for s, v in T[a].items() if v == m); say(f"  {a}: maximum {pct(m)} first at step {sm} ({sm * B[a] / 1e6:.2f}M rows); end {pct(T[a][max(T[a])])}")
    say("T2 the train side, the last 20 logged rows averaged; the raw weights' validation beside the EMA's at the last grid")
    for a in (REF,) + ARMS:
        M = metrics(root, a); L = [r for r in M if "loss" in r][-20:]; raw = trace(M, "val_t16"); ema = T[a]
        g = lambda k: float(np.mean([r[k] for r in L if k in r])) if any(k in r for r in L) else None
        say(f"  {a:7s} loss {g('loss'):.4f}  train exact {pct(g('train_exact'))}  halted {pct(g('halt_frac'))}  lr {L[-1].get('lr'):.1e} | last grid: raw {pct(raw.get(max(raw)) if raw else None)}  EMA {pct(ema.get(max(ema)) if ema else None)}")
    say("T3 accuracy by iteration on the identical 5,000 (fixed start, EMA, the selected grid); SE-RRM = their published curve (full test set, one run)")
    say(f"  {'':8s}" + " ".join(f"{t:>7d}" for t in ITS))
    for a in (REF,) + ARMS:
        c = by_iteration(root, a); say(f"  {a:8s}" + (" ".join(pct(c[t]) + " " for t in ITS) if c else " per-step bits absent or inconsistent"))
    say(f"  {'SE-RRM':8s}" + " ".join(pct(SERRM[t]) + " " for t in ITS))
    say("T4 the 32-restart scan on the 5,000 (64 iterations)")
    for a in (REF,) + ARMS:
        d = root / f"sxscan_pchamp{a}"; s = json.loads((d / "summary_all.json").read_text()); z = dict(np.load(d / "records_all.npz", allow_pickle=True)); sp = PF.spurious(z)
        ex = np.asarray(z["mi_exact_k"]).astype(bool)
        t1 = s.get("t1r_at_k") or {}
        say(f"  {a:7s} fixed start {pct(s.get('exact_acc'))}  one random start {pct(s.get('b1_exact'))}  selected by the smallest residual at k 1 / 8 / 32: {pct(t1.get('1'))} {pct(t1.get('8'))} {pct(t1.get('32'))}  "
            f"verified {pct(s.get('exact_acc_vote'))}  spurious {pct(sp)} %  | random draws exact {pct(float(ex.mean()))}, puzzles with no exact draw {int((~ex.any(1)).sum())}")
    Path(out).mkdir(parents=True, exist_ok=True); (Path(out) / "descriptive_read.txt").write_text("\n".join(LINES) + "\n")


def selftest():
    n = 0
    tr = {2000: 0.5, 4000: 0.7, 6000: 0.9}
    assert abs(at_rows(tr, 100, 300000) - 0.6) < 1e-12 and at_rows(tr, 100, 100000) is None and at_rows(tr, 100, 700000) is None and abs(at_rows(tr, 100, 600000) - 0.9) < 1e-12; n += 1
    assert at_rows({}, 100, 5) is None; n += 1
    M = [dict(monitor=dict(step=2000, val_t16_ema=0.4, val_t16=0.3)), dict(step=2000, loss=1.0), dict(monitor=dict(step=2000, val_t16_ema=0.45, val_t16=0.35)), dict(monitor=dict(step=4000, val_t16_ema=0.6))]
    assert trace(M) == {2000: 0.45, 4000: 0.6} and trace(M, "val_t16") == {2000: 0.35}; n += 1
    assert pct(None).strip() == "n/a" and pct(0.98765).strip() == "98.77"; n += 1
    bits = np.zeros((4, 64), bool); bits[0, 3:] = True; bits[1, 63] = True; bits[2, :] = True     # exact from iteration 4 / only at 64 / always / never
    z = dict(exact_by_step=np.packbits(bits, axis=1, bitorder="little"), cold_exact=bits[:, -1], first_exact=np.array([3, 63, 0, -1]))
    b, order, base = unpack_steps(z, 64); assert order == "little" and base == 0 and abs(b[:, 3].mean() - 0.5) < 1e-12 and abs(b[:, 0].mean() - 0.25) < 1e-12; n += 1
    print(f"selftest OK: {n}/{n}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--root"); ap.add_argument("--out"); ap.add_argument("--selftest", action="store_true"); a = ap.parse_args()
    selftest() if a.selftest else report(Path(a.root), a.out)
