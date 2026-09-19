#!/usr/bin/env python3
"""THE LADDER'S ADVERSARIAL PASS — the checks behind "why does SA256 read above SE-RRM's published number, and is any of it an artifact?"
(2026-09-19; the PI: "run an adversarial pass to tear this apart").  EXPLORATORY and DESCRIPTIVE: written AFTER the ladder's verdict, zero cloud,
reads banked records only, registers nothing.  Every number the pass quoted in conversation is reproduced here by a script.

  T1 SUBSET BIAS   is the 5,000-puzzle subsample easier than the full test set?  For each model with a banked full-set row: its accuracy on the
                   full set against its accuracy on the ladder's 5,000 ids (16 iterations: 422,786 puzzles; 64: the 100,000 row's overlap, n printed).
  T2 LEAKAGE       do the selector's 512 validation puzzles, or the 1,000 training puzzles, appear in the test set?  Exact 81-cell match, and a
                   match up to a relabeling of the digits (canonical form: digits renamed in order of first appearance).  Position symmetries
                   (rotations, band / stack permutations) are NOT canonicalized: stated, not tested.
  T3 PARAMETERS    where each model's parameters sit: the mixer ACROSS THE CELLS, the coupling ACROSS THE FIELDS, the per-cell MLP, the rest.
  T4 FIELD ATTENTION ALONE  C4 (attention across the fields, hidden 384, the MLP cell mixer kept) against C0, paired on identical ids.

  .venv/bin/python tools/lens_ladder_adversarial.py            ->  runs/analysis/ladder_adversarial_20260919/report.txt
  .venv/bin/python tools/lens_ladder_adversarial.py --selftest
"""
import argparse, pickle, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import analyze_paperfinal as PF   # paired (exact McNemar on identical ids)
OUT = ROOT / "runs/analysis/ladder_adversarial_20260919"
IDS_ROW = ROOT / "runs/_wladder_pull/x_final/runs/sxeval_pchampSA256/sub5k_vsel_t16/records_all.npz"
FULL = {"C5": ROOT / "runs/_sudokupend_pull/stage/runs", "C7": ROOT / "runs/_sudokupend_pull/stage/runs", "C8": ROOT / "runs/_sudokupend_pull/stage/runs", "C0": ROOT / "runs", "C4": ROOT / "runs"}
NPZ = ROOT / "data/sudoku_extreme/sudoku_extreme_seed0_mon512.npz"
CKPTS = {"W128 (MLP mixer, hidden 128)": "runs/_wladder_pull/tgz/p0/W128_ckpt.pkl", "W256 (MLP mixer, hidden 256)": "runs/_wladder_pull/grids/runs/pretrainchamp_W256/ckpt_018000.pkl",
         "SA128 (attention, hidden 128)": "runs/_wladder_pull/tgz/p0/SA128_ckpt.pkl", "SA192 (attention, hidden 192)": "runs/_wladder_pull/tgz/p1/SA192_ckpt.pkl",
         "SA256 (attention, hidden 256)": "runs/_wladder_pull/grids/runs/pretrainchamp_SA256/ckpt_028000.pkl"}
LINES = []
def say(s=""): LINES.append(str(s)); print(s)


def subset_bias(recs, ids):
    """(accuracy on all of recs, accuracy on recs restricted to ids, n found)."""
    idx = np.asarray(recs["idx"]); ex = np.asarray(recs["cold_exact"]).astype(bool); _, pa, _ = np.intersect1d(idx, ids, return_indices=True)
    return float(ex.mean()), (float(ex[pa].mean()) if len(pa) else None), int(len(pa))


def canon(q):
    """(N, 81) int grid -> the same grids with the digits renamed in order of first appearance (0 = empty stays 0)."""
    q = np.asarray(q).reshape(len(q), -1); out = np.zeros_like(q)
    for i, row in enumerate(q):
        m = {0: 0}
        for j, v in enumerate(row.tolist()):
            if v not in m: m[v] = len(m)
            out[i, j] = m[v]
    return out


def overlap(a, b):
    """how many rows of a appear among the rows of b (exact)."""
    sb = {r.tobytes() for r in np.ascontiguousarray(np.asarray(b).reshape(len(b), -1).astype(np.int8))}
    return sum(1 for r in np.ascontiguousarray(np.asarray(a).reshape(len(a), -1).astype(np.int8)) if r.tobytes() in sb)


def leaves(t, pre=""):
    if isinstance(t, dict):
        for k, v in t.items(): yield from leaves(v, f"{pre}/{k}")
    elif isinstance(t, (list, tuple)):
        for i, v in enumerate(t): yield from leaves(v, f"{pre}/{i}")
    elif hasattr(t, "shape"): yield pre, int(np.prod(t.shape)) if len(t.shape) else 1


def param_split(tree):
    """{across the cells, across the fields, per-cell MLP, rest, total} by the parameter's name."""
    s = dict(cells=0, fields=0, mlp=0, rest=0)
    for name, n in leaves(tree):
        leaf = name.split("/blocks/")[-1] if "/blocks/" in name else ""
        k = "cells" if ("/mlp_t/" in name or leaf.split("/")[-1] in ("att_qkv", "att_o")) else "fields" if leaf.split("/")[-1] in ("attn_q", "attn_k", "attn_v", "attn_o") else "mlp" if "/mlp/" in name else "rest"
        s[k] += n
    s["total"] = sum(s.values()); return s


def main():
    ids = np.asarray(np.load(IDS_ROW, allow_pickle=True)["idx"]); say(f"THE LADDER'S ADVERSARIAL PASS (EXPLORATORY; banked records only). The ladder's subsample: {len(ids)} ids.")
    say("T1 SUBSET BIAS — accuracy on the full banked row against the same row restricted to the 5,000 ids (positive = the subsample is EASIER)")
    d16 = []
    for a, r in FULL.items():
        for t, name in ((16, "full_vsel_t16"), (64, "full_vsel_t64")):
            p = r / f"sxeval_pchamp{a}" / name / "records_all.npz"
            if not p.exists(): continue
            full, sub, n = subset_bias(dict(np.load(p, allow_pickle=True)), ids)
            if sub is None: continue
            say(f"  {a:3s} @{t:<3d} full {100*full:6.2f} (n {len(np.load(p, allow_pickle=True)['idx'])})   on the 5,000: {100*sub:6.2f} (n {n})   difference {100*(sub-full):+.2f} pp")
            if t == 16 and n == len(ids): d16.append(sub - full)
    if d16: say(f"  mean difference at 16 iterations over {len(d16)} models with all 5,000 ids present: {100*float(np.mean(d16)):+.2f} pp (range {100*min(d16):+.2f} to {100*max(d16):+.2f})")
    z = np.load(NPZ, allow_pickle=True); say("T2 LEAKAGE — the selector's 512 validation puzzles and the 1,000 training puzzles against the 422,786 test puzzles")
    for name in ("val", "train"):
        q = z[f"{name}_q"]; say(f"  {name:5s} ({len(q)}): exact matches in the test set {overlap(q, z['test_q'])}; matches up to a relabeling of the digits {overlap(canon(q), canon(z['test_q']))}")
    say(f"  val against train: exact {overlap(z['val_q'], z['train_q'])}, up to relabeling {overlap(canon(z['val_q']), canon(z['train_q']))}   (position symmetries are not canonicalized)")
    say("T3 PARAMETERS — the EMA weights of each selected checkpoint, by role")
    say(f"  {'model':32s} {'across the cells':>16s} {'across the fields':>17s} {'per-cell MLP':>12s} {'rest':>8s} {'total':>10s}")
    for lab, p in CKPTS.items():
        if not (ROOT / p).exists(): say(f"  {lab:32s} checkpoint absent"); continue
        s = param_split(pickle.load(open(ROOT / p, "rb"))["state_ema"]); say(f"  {lab:32s} {s['cells']:16,d} {s['fields']:17,d} {s['mlp']:12,d} {s['rest']:8,d} {s['total']:10,d}")
    say("T4 FIELD ATTENTION ALONE — C4 (attention across the fields, hidden 384, the MLP cell mixer kept) against C0, paired on identical ids")
    for t, name in ((16, "full_vsel_t16"), (64, "full_vsel_t64")):
        pa, pb = (FULL["C4"] / "sxeval_pchampC4" / name / "records_all.npz"), (FULL["C0"] / "sxeval_pchampC0" / name / "records_all.npz")
        if not (pa.exists() and pb.exists()): say(f"  @{t}: rows absent"); continue
        q = PF.paired(dict(np.load(pa, allow_pickle=True)), dict(np.load(pb, allow_pickle=True)))
        say(f"  @{t:<3d} C4 - C0 = {100*q['diff']:+.2f} pp (only-C4 {q['only_a']}, only-C0 {q['only_b']}, n {q['n']}, p {q['p']:.2g})")
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "report.txt").write_text("\n".join(LINES) + "\n")


def selftest():
    n = 0
    recs = dict(idx=np.arange(10), cold_exact=np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])); f, s, k = subset_bias(recs, np.array([0, 1, 4, 99]))
    assert abs(f - 0.4) < 1e-12 and abs(s - 2 / 3) < 1e-12 and k == 3 and subset_bias(recs, np.array([77]))[1:] == (None, 0); n += 1
    a = np.array([[0, 5, 5, 3, 0, 9]]); b = np.array([[0, 2, 2, 7, 0, 4]]); c = np.array([[0, 2, 7, 7, 0, 4]])
    assert (canon(a) == np.array([[0, 1, 1, 2, 0, 3]])).all() and (canon(a) == canon(b)).all() and not (canon(a) == canon(c)).all(); n += 1
    assert overlap(a, np.vstack([b, a])) == 1 and overlap(a, b) == 0 and overlap(canon(a), canon(np.vstack([b, c]))) == 1; n += 1
    Z = lambda *s: np.zeros(s)
    mlp = {"model": {"dec": {"blocks": [{"fc": Z(4, 4), "mlp": {"down": Z(12, 4), "gate_up": Z(4, 24)}, "mlp_t": {"down": Z(256, 81), "gate_up": Z(81, 512)}}], "lm_head": Z(4)}}, "table": Z(1, 32)}
    att = {"model": {"dec": {"blocks": [{"fc": Z(4, 4), "mlp": {"down": Z(12, 4), "gate_up": Z(4, 24)}, "att_qkv": Z(4, 12), "att_o": Z(4, 4), "attn_q": Z(4, 8), "attn_k": Z(4, 8)}], "lm_head": Z(4)}}}
    assert param_split(mlp) == dict(cells=62208, fields=0, mlp=144, rest=16 + 4 + 32, total=62208 + 144 + 52); n += 1
    assert param_split(att) == dict(cells=64, fields=64, mlp=144, rest=20, total=292); n += 1
    print(f"selftest OK: {n}/{n}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true"); a = ap.parse_args()
    selftest() if a.selftest else main()
