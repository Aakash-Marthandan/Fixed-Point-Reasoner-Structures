#!/usr/bin/env python3
"""DESCRIPTIVE reader for the width ladder (EXPLORATORY; no registered letter is read or changed here; tools/analyze_wladder.py is the verdict).

Five tables from the banked artifacts:
  1. the training side per arm: parameters, the trainer's pace and its per-step cost against the width-192 DEC's measured 8-chip pace, the EMA
     monitor's maximum and where it sits in the fixed budget, the registered excursion check (EMA under 60 % after first passing 85 %);
  2. exact accuracy by iteration on the identical puzzles, from the 64-iteration rows' packed per-step bits (the unpacking is VERIFIED per row:
     the last bit equals the row's exact bit on every puzzle and the first set bit equals first_exact);
  3. width-384 and width-192 seed references restricted to the identical ids (rows given as name=dir);
  4. the k32 restart scan's selector anatomy per arm (tools/lens_sudokupend_read.selector_anatomy);
  5. exploratory pairs on the identical puzzles (exact McNemar; NOT registered pairs).

usage: .venv/bin/python tools/lens_wladder_read.py --root <stage>/runs [--refs name=dir ...] [--out file.txt]
       .venv/bin/python tools/lens_wladder_read.py --selftest
"""
import argparse, json, sys, tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_paperfinal as PF          # paired, exact_mcnemar
import lens_sudokupend_read as LS        # selector_anatomy, train_side (selftested there)

ARMS = ("W128", "W256", "SA128", "SA192", "SA256")
REF = "W192R"
DEC_PACE = 14.2                          # the width-192 DEC's measured 8-chip trainer pace (C7, C8), steps/s
ITS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
PAIRS = (("SA256", REF), ("SA192", REF), ("SA128", REF), ("SA128", "W256"))


def unpack_steps(z, T):
    """(bits (n, T) bool, bit order, first_exact base) of a row's packed per-step exact bits, or (None, None, None) when no order is
    consistent with the row's own exact bit. The order is accepted only if the LAST bit equals cold_exact on every puzzle."""
    for order in ("big", "little"):
        b = np.unpackbits(np.asarray(z["exact_by_step"]), axis=1, bitorder=order)[:, :T].astype(bool)
        if b.shape[1] == T and (b[:, T - 1] == np.asarray(z["cold_exact"]).astype(bool)).all():
            base = None
            if "first_exact" in z:
                fe = np.asarray(z["first_exact"]); ok = fe >= 0; first = np.where(b.any(1), b.argmax(1), -1)
                base = 0 if (first[ok] == fe[ok]).all() else (1 if (first[ok] + 1 == fe[ok]).all() else None)
            return b, order, base
    return None, None, None


def excursion(ema, hi=0.85, lo=0.60):
    """the registered check: grids whose EMA falls under lo after the curve first passed hi. ema = [(step, value)]."""
    first = next((i for i, (_, v) in enumerate(ema) if v >= hi), None)
    return [] if first is None else [s for s, v in ema[first + 1:] if v < lo]


def on_ids(d: Path, ids):
    """(accuracy, n found, grid, row n) of a row restricted to ids; None when the row is absent."""
    if not (d / "records_all.npz").exists(): return None
    z = np.load(d / "records_all.npz"); s = json.loads((d / "summary_all.json").read_text()) if (d / "summary_all.json").exists() else {}
    _, pa, _ = np.intersect1d(np.asarray(z["idx"]), ids, return_indices=True)
    return (float(np.asarray(z["cold_exact"]).astype(bool)[pa].mean()) if len(pa) else None), int(len(pa)), str(s.get("ckpt")), s.get("n")


def recs(root: Path, arm: str, t: int): return dict(np.load(root / f"sxeval_pchamp{arm}" / f"sub5k_vsel_t{t}" / "records_all.npz"))
def pct(x): return "n/a" if x is None else f"{100 * x:.2f}"


def report(root: Path, refs, out):
    L = ["THE WIDTH LADDER — DESCRIPTIVE READ (EXPLORATORY; tools/lens_wladder_read.py)"]
    arms = [a for a in ARMS if (root / f"pretrainchamp_{a}" / "metrics.jsonl").exists()]
    L.append("\n1. the training side (fixed 30,000 steps; the EMA monitor every 2,000 on 512 validation puzzles)")
    for a in arms:
        c = json.loads((root / f"pretrainchamp_{a}" / "config.json").read_text()); n = (c.get("n_params_bulk") or 0) + (c.get("n_params_table") or 0)
        t = LS.train_side(root / f"pretrainchamp_{a}"); ema = [(m["step"], m.get("val_t16_ema") or 0.0) for m in t["mon"]]
        rs = root / f"pretrainchamp_{a}" / "resumes.txt"; res = rs.read_text().split() if rs.exists() else []
        L.append(f"   {a:6s} params {n:>9,} | trainer {t['sps']:5.1f} steps/s = x{DEC_PACE / t['sps']:.1f} of the width-192 DEC's per-step cost | EMA max {pct(t['best'][1])} at {t['at']}"
                 f" -> {'ONLY AT THE BUDGET EDGE' if t['at_edge'] else ('reaches the edge (tie)' if t['at'][-1] == t['last'][0] else 'inside the budget')}; last three {[pct(v) for _, v in ema[-3:]]}; excursion grids {excursion(ema)}; resumes {res}")
    L.append("\n2. exact accuracy by iteration on the identical puzzles (the 64-iteration rows)")
    L.append("   arm      " + " ".join(f"{i:>6d}" for i in ITS))
    for a in [x for x in ("W128", REF, "W256", "SA128", "SA192", "SA256") if (root / f"sxeval_pchamp{x}" / "sub5k_vsel_t64" / "records_all.npz").exists()]:
        z = recs(root, a, 64); b, order, base = unpack_steps(z, 64)
        if b is None: L.append(f"   {a:8s} UNPACK-FAILED (no bit order matches the row's exact bit)"); continue
        z16 = recs(root, a, 16); same = abs(float(b[:, 15].mean()) - float(np.asarray(z16["cold_exact"]).mean())) < 1e-12 if np.array_equal(np.sort(z["idx"]), np.sort(z16["idx"])) else None
        L.append(f"   {a:8s} " + " ".join(f"{100 * b[:, i - 1].mean():6.2f}" for i in ITS) + f"   [{order}-endian, first_exact {base}-based, iteration 16 == the 16-row: {same}]")
    ids = np.sort(recs(root, REF, 64)["idx"]) if (root / f"sxeval_pchamp{REF}" / "sub5k_vsel_t64" / "records_all.npz").exists() else None
    if refs and ids is not None:
        L.append(f"\n3. references restricted to the identical {len(ids)} ids")
        for name, d in refs:
            r = on_ids(Path(d), ids); L.append(f"   {name:24s} " + ("absent" if r is None else f"{pct(r[0])} on {r[1]} of {len(ids)} | grid {r[2][-24:]} | row n {r[3]}"))
    L.append("\n4. the k32 restart scan (the selector = the minimum-residual draw; spurious = wrong draws at or under the exact draws' median residual)")
    L.append(f"   {'arm':6s} {'cold':>6s} {'selected':>9s} {'verified':>9s} {'spur%':>6s} {'per-draw':>9s} {'no-basin':>9s} {'rescued':>8s} | residual p50 exact | wrong")
    for a in arms:
        d = root / f"sxscan_pchamp{a}"
        if not (d / "records_all.npz").exists(): continue
        t = LS.selector_anatomy(d)
        L.append(f"   {a:6s} {pct(t['cold']):>6s} {pct(t['sel']):>9s} {pct(t['ver']):>9s} {100 * (t['spur'] or 0):6.2f} {pct(t['per_draw']):>9s} {t['no_basin']:>9d} {t['rescued']:>8d} | {t['q_exact'][1]:.4f} | {t['q_wrong'][1]:.4f}" + ("  INVERTED" if t["inverted"] else ""))
    L.append("\n5. exploratory pairs on the identical puzzles (exact McNemar; NOT registered pairs)")
    for a, c in PAIRS:
        for t in (16, 64):
            try: q = PF.paired(recs(root, a, t), recs(root, c, t))
            except FileNotFoundError: q = None
            L.append(f"   {a} - {c} @{t}: " + ("n/a" if not q else f"{100 * q['diff']:+.2f} pp (only-{a} {q['only_a']}, only-{c} {q['only_b']}, n {q['n']}, p {q['p']:.1e})"))
    text = "\n".join(L); print(text)
    if out: Path(out).parent.mkdir(parents=True, exist_ok=True); Path(out).write_text(text + "\n")


def selftest():
    ok, bad = 0, []
    def chk(n, c):
        nonlocal ok
        if c: ok += 1
        else: bad.append(n)
    rng = np.random.default_rng(0); n, T = 40, 16
    bits = np.zeros((n, T), bool); fe = np.full(n, -1)
    for i in range(n):
        if rng.random() < 0.7:
            k = int(rng.integers(0, T)); bits[i, k:] = True; fe[i] = k
    for order in ("little", "big"):
        z = dict(exact_by_step=np.packbits(bits, axis=1, bitorder=order), cold_exact=bits[:, -1], first_exact=fe)
        b, o, base = unpack_steps(z, T); chk(f"unpack {order}", b is not None and o == order and base == 0 and (b == bits).all())
    z1 = dict(exact_by_step=np.packbits(bits, axis=1, bitorder="little"), cold_exact=bits[:, -1], first_exact=np.where(fe >= 0, fe + 1, -1))
    chk("first_exact 1-based detected", unpack_steps(z1, T)[2] == 1)
    zb = dict(exact_by_step=np.packbits(bits, axis=1, bitorder="little"), cold_exact=~bits[:, -1])
    chk("inconsistent row refused", unpack_steps(zb, T) == (None, None, None))
    chk("excursion", excursion([(2, .1), (4, .9), (6, .5), (8, .95), (10, .59)]) == [6, 10] and excursion([(2, .5), (4, .7)]) == [] and excursion([(2, .9), (4, .88)]) == [])
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "row"; d.mkdir()
        np.savez(d / "records_all.npz", idx=np.arange(10), cold_exact=np.array([1, 1, 0, 0, 1, 1, 1, 0, 1, 1], bool)); (d / "summary_all.json").write_text(json.dumps({"n": 10, "ckpt": "runs/x/ckpt_016000.pkl"}))
        r = on_ids(d, np.array([0, 2, 4, 99])); chk("on_ids", r[1] == 3 and abs(r[0] - 2 / 3) < 1e-12 and r[2].endswith("ckpt_016000.pkl") and r[3] == 10)
        chk("on_ids absent", on_ids(Path(td) / "nope", np.arange(3)) is None)
    chk("pct", pct(None) == "n/a" and pct(0.98236) == "98.24")
    print(f"selftest {'OK' if not bad else 'FAILED'}: {ok}/{ok + len(bad)} checks" + (f"; failed: {bad}" if bad else ""))
    return 0 if not bad else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="runs"); ap.add_argument("--refs", nargs="*", default=[]); ap.add_argument("--out", default=None); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest: sys.exit(selftest())
    report(Path(a.root), [tuple(r.split("=", 1)) for r in a.refs], a.out)
