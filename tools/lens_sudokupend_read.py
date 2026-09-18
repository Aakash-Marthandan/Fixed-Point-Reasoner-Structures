#!/usr/bin/env python3
"""DESCRIPTIVE reader for the pending Sudoku runs (EXPLORATORY; no registered letter is read or changed here).

Five tables from the banked artifacts, beside tools/analyze_sudokupend.py's registered verdict:
  1. the k128 rows paired against EqR on the identical puzzles: discordant counts, the exact McNemar p, the triple's shared failures, the restart gain;
  2. the X arm's battery rows as the evaluator wrote them;
  3. the X arm's training side against a DEC seed: the monitor curve, windowed training means, the measured pace, where the selected grid sits in the budget;
  4. the residual selector's anatomy (residual quantiles of exact vs wrong draws, the verifier's rescue count, cold vs selected);
  5. the final grid against the selected grid and the raw weights against the EMA, paired on the identical puzzles.

usage: python3 tools/lens_sudokupend_read.py --root <stage>/runs [--arm X5] [--ref C5] [--out file.txt]
       python3 tools/lens_sudokupend_read.py --selftest
"""
import argparse, glob, json, sys, tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_paperfinal as PF   # selected_exact, exact_mcnemar, spurious

TRIPLE = ("C5", "C7", "C8")


def scan_row(d: Path):
    """one k-restart row, sorted by puzzle index: idx, selected, verified, cold, the per-draw exact matrix."""
    z = np.load(d / "records_all.npz"); sel, ver = PF.selected_exact(z); o = np.argsort(z["idx"])
    return dict(idx=np.asarray(z["idx"])[o], sel=np.asarray(sel)[o], ver=np.asarray(ver)[o], cold=np.asarray(z["cold_exact"])[o], ex=np.asarray(z["mi_exact_k"])[o], spur=PF.spurious(z))


def pair_counts(a, e):
    """(only-a, only-e, both-fail, p) on the selected bit; the rows must cover the identical puzzles."""
    if not np.array_equal(a["idx"], e["idx"]): return None
    oa, oe = int((a["sel"] & ~e["sel"]).sum()), int((~a["sel"] & e["sel"]).sum())
    return oa, oe, int((~a["sel"] & ~e["sel"]).sum()), PF.exact_mcnemar(oa, oe)


def shared_failures(rows, e):
    F = {k: set(r["idx"][~r["sel"]].tolist()) for k, r in rows.items()}
    u = set().union(*F.values()); i = set.intersection(*F.values()); ef = set(e["idx"][~e["sel"]].tolist())
    return dict(per={k: len(v) for k, v in F.items()}, union=len(u), all=len(i), union_and_eqr=len(u & ef), eqr=len(ef))


def selector_anatomy(d: Path):
    """the residual selector's anatomy on one k-restart row: residual quantiles of exact vs wrong draws (the selector picks the minimum
    residual), the puzzles a verifier would rescue (verified-but-not-selected), cold vs selected, and the no-basin count."""
    z = np.load(d / "records_all.npz"); ex = np.asarray(z["mi_exact_k"]).astype(bool); rs = np.asarray(z["mi_resid_k"]).astype(float)
    sel, ver = PF.selected_exact(z); fin = np.isfinite(rs); r, w = ex & fin, (~ex) & fin
    q = lambda a: (float(np.quantile(a, .1)), float(np.quantile(a, .5)), float(np.quantile(a, .9))) if a.size else (None, None, None)
    return dict(n=int(ex.shape[0]), k=int(ex.shape[1]), cold=float(np.asarray(z["cold_exact"]).mean()), sel=float(sel.mean()), ver=float(ver.mean()),
                spur=PF.spurious(z), per_draw=float(ex.mean()), no_basin=int((ex.sum(1) == 0).sum()), rescued=int((ver & ~sel).sum()),
                q_exact=q(rs[r]), q_wrong=q(rs[w]), inverted=bool(r.any() and w.any() and np.median(rs[w]) < np.median(rs[r])))


def paired_rows(a: Path, b: Path):
    """two D16 rows paired on their common puzzles (records_all.npz with idx + cold_exact): diff = a - b, the discordant counts, p."""
    if not (a / "records_all.npz").exists() or not (b / "records_all.npz").exists(): return None
    return PF.paired(dict(np.load(a / "records_all.npz")), dict(np.load(b / "records_all.npz")))


def windows(tr, width=10000, keys=("loss", "ce_in", "train_exact")):
    out, last = [], max(r["step"] for r in tr)
    for lo in range(0, last, width):
        w = [r for r in tr if lo < r["step"] <= lo + width]
        if w: out.append((lo + width, {k: float(np.mean([r[k] for r in w if k in r])) for k in keys}))
    return out


def train_side(p: Path):
    rows = [json.loads(l) for l in open(p / "metrics.jsonl") if l.strip()]
    mon = sorted({r["monitor"]["step"]: r["monitor"] for r in rows if "monitor" in r}.values(), key=lambda m: m["step"])
    tr = [r for r in rows if "step" in r and "loss" in r]
    ema = [(m["step"], m.get("val_t16_ema") or 0.0) for m in mon]
    top = max(v for _, v in ema); at = [s for s, v in ema if v == top]; best = (at[0], top)   # every grid at the maximum is listed; the edge flag needs the maximum ONLY at the last grid
    sps = [r["steps_per_sec"] for r in tr if r.get("steps_per_sec")]
    return dict(mon=mon, win=windows(tr), best=best, at=at, last=ema[-1], at_edge=at == [ema[-1][0]], sps=float(np.median(sps)) if sps else None,
                late_gain=ema[-1][1] - next((v for s, v in ema if s >= ema[-1][0] - 10000), ema[0][1]))


def fmt(x): return "n/a" if x is None else f"{100 * x:.2f}"


def report(root: Path, arm: str, ref: str, out: Path | None):
    L = ["PENDING SUDOKU RUNS — DESCRIPTIVE READ (EXPLORATORY; tools/lens_sudokupend_read.py)"]
    e = scan_row(root / "filler_sxscan128_pport_eqr"); rows = {a: scan_row(root / f"filler_sxscan128_pchamp{a}") for a in TRIPLE + (arm,) if (root / f"filler_sxscan128_pchamp{a}" / "records_all.npz").exists()}
    L.append(f"\n1. k128 on the identical {len(e['idx'])} puzzles at 64 iterations, against EqR (selected {fmt(e['sel'].mean())}, cold {fmt(e['cold'].mean())}, {int((e['ex'].sum(1) == 0).sum())} puzzles with no exact draw)")
    L.append(f"   {'row':4s} {'cold':>6s} {'selected':>9s} {'verified':>9s} {'spurious%':>10s} {'no-exact':>9s} | only-row only-EqR both-fail  p")
    for a, r in rows.items():
        pc = pair_counts(r, e)
        L.append(f"   {a:4s} {fmt(r['cold'].mean()):>6s} {fmt(r['sel'].mean()):>9s} {fmt(r['ver'].mean()):>9s} {100 * r['spur']:>10.3f} {int((r['ex'].sum(1) == 0).sum()):>9d} | " + ("idx MISMATCH" if pc is None else f"{pc[0]:>8d} {pc[1]:>8d} {pc[2]:>9d}  {pc[3]:.1e}"))
    tri = {a: rows[a] for a in TRIPLE if a in rows}
    if len(tri) == 3:
        for k, name in (("sel", "selected"), ("ver", "verified"), ("cold", "cold")):
            v = [float(r[k].mean()) for r in tri.values()]; L.append(f"   the triple's {name}: mean {fmt(float(np.mean(v)))}, half-spread {fmt((max(v) - min(v)) / 2)}")
        sf = shared_failures(tri, e)
        L.append(f"   the triple's unsolved (selected): {sf['per']}; union {sf['union']}, all three {sf['all']}; of the union EqR also fails {sf['union_and_eqr']} (EqR fails {sf['eqr']}); some seed solves {len(e['idx']) - sf['all']} of {len(e['idx'])}")
    L.append(f"\n2. {arm}'s battery rows (the evaluator's summaries)")
    for f in sorted(glob.glob(str(root / f"sxeval_pchamp{arm}" / "*" / "summary_all.json"))) + sorted(glob.glob(str(root / f"sxscreen_pchamp{arm}_*" / "summary_all.json"))):
        s = json.load(open(f)); name = str(Path(f).parent.relative_to(root))
        L.append(f"   {name:36s} n {s.get('n'):>6} grid {str(s.get('ckpt'))[-11:-4]:>7s} t {s.get('t_total'):>3} k {s.get('k_init'):>3} exact {fmt(s.get('exact_acc'))}")
    L.append(f"\n3. the training side: {arm} against {ref} (the same loop, regime and budget)")
    for a in (arm, ref):
        if not (root / f"pretrainchamp_{a}" / "metrics.jsonl").exists(): continue
        t = train_side(root / f"pretrainchamp_{a}")
        L.append(f"   {a}: the EMA monitor's maximum {fmt(t['best'][1])} at grid(s) {t['at']}, the budget's last grid {t['last'][0]} -> {'ONLY AT THE BUDGET EDGE' if t['at_edge'] else 'first reached inside the budget'}; EMA gain over the last 10k steps {100 * t['late_gain']:+.1f} pp; median pace {t['sps']:.1f} steps/s")
        ema = [(m["step"], m.get("val_t16_ema") or 0) for m in t["mon"]]; top = t["best"][1]
        L.append(f"      within 1 pp of the maximum: {[s for s, v in ema if top - v < 0.01]}; EMA at the last grid {fmt(ema[-1][1])}")
        L.append("      monitor EMA: " + " ".join(f"{m['step'] // 1000}k:{100 * (m.get('val_t16_ema') or 0):.1f}" for m in t["mon"]))
        L.append("      training:    " + " | ".join(f"{s // 1000}k loss {w['loss']:.3f} exact {100 * w['train_exact']:.1f}" for s, w in t["win"]))
    L.append(f"\n4. the residual selector's anatomy (k-restart rows; the selector = the minimum-residual draw; spurious = wrong draws at or under the exact draws' median residual)")
    L.append(f"   {'row':14s} {'n':>5s} {'k':>4s} {'cold':>6s} {'selected':>9s} {'verified':>9s} {'spur%':>6s} {'per-draw':>8s} {'no-basin':>8s} {'rescued':>7s} | residual p10/p50/p90  exact | wrong")
    for a, d in ((f"{arm}", root / f"sxscan_pchamp{arm}"), (f"{arm} k128", root / f"filler_sxscan128_pchamp{arm}"), (f"{ref} k128", root / f"filler_sxscan128_pchamp{ref}")):
        if not (d / "records_all.npz").exists(): continue
        t = selector_anatomy(d); qe, qw = t["q_exact"], t["q_wrong"]
        L.append(f"   {a:14s} {t['n']:>5d} {t['k']:>4d} {fmt(t['cold']):>6s} {fmt(t['sel']):>9s} {fmt(t['ver']):>9s} {100*(t['spur'] or 0):>6.2f} {fmt(t['per_draw']):>8s} {t['no_basin']:>8d} {t['rescued']:>7d} | "
                 + (f"{qe[0]:.4f}/{qe[1]:.4f}/{qe[2]:.4f} | {qw[0]:.4f}/{qw[1]:.4f}/{qw[2]:.4f}" if qe[0] is not None and qw[0] is not None else "n/a") + ("  INVERTED (wrong draws more converged than exact ones)" if t["inverted"] else ""))
    L.append(f"\n5. {arm}'s final grid against its selected grid, and its raw weights against its EMA, paired on the identical puzzles")
    for name, a, b in (("final − selected", root / f"sxeval_pchamp{arm}" / "full_final_t16", root / f"sxeval_pchamp{arm}" / "full_vsel_t16"),
                       ("raw − EMA (selected)", root / f"sxeval_pchamp{arm}" / "full_vsel_t16_alt", root / f"sxeval_pchamp{arm}" / "full_vsel_t16")):
        pr = paired_rows(a, b)
        L.append(f"   {name:22s} " + ("n/a" if pr is None else f"d {100*pr['diff']:+.2f} pp on {pr['n']:,}; only-first {pr['only_a']:,}, only-second {pr['only_b']:,}; p {pr['p']:.1e}"))
    text = "\n".join(L); print(text)
    if out: out.parent.mkdir(parents=True, exist_ok=True); out.write_text(text + "\n")


def selftest():
    ok, bad = 0, []
    def chk(n, c):
        nonlocal ok
        if c: ok += 1
        else: bad.append(n)
    def mk(sel_bits):
        n = len(sel_bits); ex = np.zeros((n, 4), bool); res = np.full((n, 4), 0.9)
        for i, b in enumerate(sel_bits):
            if b: ex[i, 1] = True; res[i, 1] = 0.01
        return dict(idx=np.arange(n)[::-1].copy(), cold_exact=np.zeros(n, bool), mi_exact_k=ex[::-1].copy(), mi_resid_k=res[::-1].copy())
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for name, bits in (("a", [1, 1, 1, 0, 0, 1]), ("b", [1, 1, 0, 0, 1, 1]), ("c", [1, 0, 1, 0, 1, 1]), ("e", [1, 0, 0, 0, 1, 0])):
            (td / name).mkdir(); np.savez(td / name / "records_all.npz", **mk(bits))
        a, b, c, e = (scan_row(td / k) for k in "abce")
        chk("sorted by idx", a["idx"].tolist() == list(range(6)) and a["sel"].tolist() == [True, True, True, False, False, True])
        pc = pair_counts(a, e); chk("pair counts", pc[:3] == (3, 1, 1) and abs(pc[3] - PF.exact_mcnemar(3, 1)) < 1e-12)
        e2 = dict(e); e2["idx"] = e["idx"] + 1; chk("idx mismatch refused", pair_counts(a, e2) is None)
        sf = shared_failures(dict(a=a, b=b, c=c), e)
        chk("shared failures", sf["per"] == dict(a=2, b=2, c=2) and sf["union"] == 4 and sf["all"] == 1 and sf["eqr"] == 4 and sf["union_and_eqr"] == 3)
        chk("no-exact count", int((a["ex"].sum(1) == 0).sum()) == 2)
        p = td / "pretrainchamp_X"; p.mkdir()
        with open(p / "metrics.jsonl", "w") as f:
            for s in range(1000, 20001, 1000): f.write(json.dumps({"step": s, "loss": 1.0 if s <= 10000 else 0.5, "ce_in": 0.4, "train_exact": 0.1 if s <= 10000 else 0.3, "steps_per_sec": 80.0}) + "\n")
            for s, v in ((5000, 0.2), (10000, 0.3), (15000, 0.3), (20000, 0.45)): f.write(json.dumps({"monitor": {"step": s, "val_t16": v, "val_t16_ema": v}}) + "\n")
        t = train_side(p)
        chk("edge", t["at_edge"] and t["best"] == (20000, 0.45)); chk("late gain", abs(t["late_gain"] - 0.15) < 1e-9); chk("pace", t["sps"] == 80.0)
        chk("windows", [s for s, _ in t["win"]] == [10000, 20000] and abs(t["win"][0][1]["loss"] - 1.0) < 1e-9 and abs(t["win"][1][1]["train_exact"] - 0.3) < 1e-9)
        with open(p / "metrics.jsonl", "a") as f: f.write(json.dumps({"monitor": {"step": 22000, "val_t16": 0.40, "val_t16_ema": 0.40}}) + "\n")
        chk("inside the budget", not train_side(p)["at_edge"])
        with open(p / "metrics.jsonl", "a") as f: f.write(json.dumps({"monitor": {"step": 24000, "val_t16": 0.45, "val_t16_ema": 0.45}}) + "\n")
        t = train_side(p); chk("a tie with an earlier grid is not the edge", t["at"] == [20000, 24000] and not t["at_edge"] and t["best"][0] == 20000)
        chk("formatter", fmt(None) == "n/a" and fmt(0.4504) == "45.04")
        # selector anatomy: a clean row (exact draws tiny residual) and an INVERTED row (wrong draws more converged); a verifier rescue
        def mkrow(dd, ex_res, wr_res, bits):
            dd.mkdir(parents=True, exist_ok=True); n = len(bits); ex = np.zeros((n, 3), bool); rs = np.full((n, 3), wr_res)
            for i, b in enumerate(bits):
                if b: ex[i, 2] = True; rs[i, 2] = ex_res
            np.savez(dd / "records_all.npz", idx=np.arange(n), cold_exact=np.asarray(bits, bool), mi_exact_k=ex, mi_resid_k=rs)
        mkrow(td / "clean", 0.002, 0.03, [1, 1, 0, 1]); mkrow(td / "inv", 0.15, 0.10, [1, 1, 0, 1])
        c = selector_anatomy(td / "clean"); i = selector_anatomy(td / "inv")
        chk("anatomy clean", not c["inverted"] and c["spur"] == 0.0 and c["rescued"] == 0 and abs(c["sel"] - 0.75) < 1e-9 and c["no_basin"] == 1)
        chk("anatomy inverted", i["inverted"] and i["spur"] == 1.0 and i["rescued"] == 3 and i["sel"] == 0.0 and abs(i["ver"] - 0.75) < 1e-9)
        # paired rows: diff = a - b on the common idx
        (td / "pa").mkdir(); (td / "pb").mkdir()
        np.savez(td / "pa" / "records_all.npz", idx=np.arange(6), cold_exact=np.array([1, 1, 1, 0, 0, 0], bool))
        np.savez(td / "pb" / "records_all.npz", idx=np.arange(2, 8), cold_exact=np.array([0, 0, 1, 1, 1, 1], bool))   # common idx 2..5: a 1,0,0,0 vs b 0,0,1,1
        pr = paired_rows(td / "pa", td / "pb"); chk("paired rows", pr["n"] == 4 and pr["only_a"] == 1 and pr["only_b"] == 2 and abs(pr["diff"] + 0.25) < 1e-9)
        chk("paired rows absent", paired_rows(td / "pa", td / "nope") is None)
    print(f"selftest {'OK' if not bad else 'FAILED'}: {ok}/{ok + len(bad)} checks" + (f"; failed: {bad}" if bad else ""))
    return 0 if not bad else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="runs"); ap.add_argument("--arm", default="X5"); ap.add_argument("--ref", default="C5"); ap.add_argument("--out", default=None); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest: sys.exit(selftest())
    report(Path(a.root), a.arm, a.ref, Path(a.out) if a.out else None)
