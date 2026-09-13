#!/usr/bin/env python3
# Ledger: THE PAPER'S FINAL RUNS — the frozen analyzer (registration: Documentation/Plan_2026-09-13_Paper_Final_Runs.md; the rules
# below are locked verbatim at the registration commit and adjudicate byte-untouched; `--selftest` exercises every letter both ways).
# The runs: C7 / C8 = C5's recipe (the symmetric DEC at width 192) at seeds 1 / 2 through the champion battery + the full-set / 50k
# rows; EqR's released weights at k = 128 on the champion arms' identical 5k; the set-attention arm C4 at D64 on the full set.
#
# REGISTRY (R) — the decision rules:
#   INTEGRITY  C7/C8 trained with C5's argv except --seed (--out / --steps / --remat ignored, --remat labeled); every vsel row of an
#              arm (chain rows, its k32 scan, its filler rows) on ONE checkpoint path; n-gates (D16 full 422,786; final 50,000 or
#              422,786; alt 50,000; D64 100,000 chain / 422,786 filler; D128 20,000 chain / 50,000 filler; D256 5,000 / 50,000; k32 scan
#              n 5,000 t 64; k128 scans n 5,000 k 128 t 64); EqR's 5k idx set == every champion arm's k128 idx set (identical puzzles).
#   R-PF-1 SEEDS-W192       spread of {C5, C7, C8} cold exact at D16 on the full set <= 1.5 pp -> TIGHT, else WIDE (the D64-full
#                           spread reported beside it).
#   R-PF-2 HEADLINE-W192    the width-192 triple's D64 full-set mean: >= 99.0 % HOLDS · [98.5, 99.0) SOFTENS · < 98.5 FALLS; the
#                           paper's number for the 0.8M cell := the triple mean +/- half-spread, whatever the letter.
#   R-PF-3 WIDTH-PAIRED     per seed s: D(s) = w192(s) - w384(s) at D64 on the full set (C5-C0, C7-C1, C8-C2), exact pairing by idx:
#                           mean D >= +0.5 pp AND all three D > 0 -> WIDTH-DOWN-PAYS; mean D <= -0.5 pp AND all three < 0 -> COSTS;
#                           |mean D| < 0.5 pp -> PARITY; else MIXED. The same letter is reported at D16 (descriptive).
#   R-PF-4 SELECTOR-W192    per arm (C7, C8): spurious <= .005 AND t1r@32 >= verified@32 - .001 -> CLEAN; both CLEAN -> SELECTOR-CLEAN.
#   R-PF-5 DEPTH-W192       per arm (C7, C8): every depth row's depth_regressions <= .0005 x n -> MONOTONE; puzzles solved at D16 full
#                           and unsolved at D64 full (exact pairing) <= 2 -> +ZERO-REGRESSION.
#   R-PF-6 MEMORIZATION     per arm: cold16 (full) - final16 (50k) > .05 -> VSEL-FINAL-DROP; mean ce_in over the last 5 logged
#                           rows < .02 -> END-CE; neither -> NOT-MEMORIZED (the champion analyzer's definitions).
#   R-PF-7 EXTENSION        EXTENDED.txt present -> EXTENDED; the selected step > the original budget -> +PAID.
#   R-PF-8 EXCURSION        after the first 2k monitor grid with val_t16_ema >= .90, >= 2 CONSECUTIVE grids with raw val_t16 < .50
#                           -> EXCURSION (steps listed), else NONE; the detector is valid only if C5 reads EXCURSION (else
#                           DETECTOR-INVALID); calibrated before registration on C0-C6: C5 fires (16k-28k), C4's single 12k dip does not.
#   R-PF-9 RESTART-PAIRED   EqR on the identical 5k at k 128: t1r and verified recomputed from the per-draw records (cross-checked to
#                           the summary); CONSISTENT with its 20k reading (t1r 98.85) if |diff| <= 3 x the combined binomial SE, else
#                           SET-SHIFT. Per champion arm with a k128 row: D = t1r(arm) - t1r(EqR) on the identical 5k, exact McNemar
#                           on the selected draw's exact bit: p < .01 and D > 0 -> AHEAD; p < .01 and D < 0 -> BEHIND; else PARITY.
#   R-PF-10 C4-FULL         C4 at D64 on the full set vs its 100k row: |diff| <= .0025 -> CONSISTENT, else SUBSET-SHIFT; paired vs
#                           C0 on the full set reported (D, McNemar p).
#   PREDICTIONS (bands locked; the scoreboard reads HIT / MISS-ABOVE / MISS-BELOW): C7, C8 cold16 in [.949, .969]; C7, C8 D64 full in
#     [.986, .995]; SEEDS-W192 TIGHT; HEADLINE-W192 HOLDS; WIDTH-PAIRED PAYS; SELECTOR-CLEAN; MONOTONE+ZERO-REGRESSION on both;
#     NOT-MEMORIZED on both; >= 1 EXTENDED; >= 1 EXCURSION of C7/C8; EqR 5k t1r in [.983, .994]; EqR CONSISTENT; EqR spurious <= .005;
#     C5 AHEAD of EqR; C4 D64 full in [.9885, .9910] and CONSISTENT.
"""
  .venv/bin/python tools/analyze_paperfinal.py --root <dir laid out like runs/>   -> <root>/analysis/paperfinal_verdict.{txt,json}
  .venv/bin/python tools/analyze_paperfinal.py --selftest
"""
from __future__ import annotations
import argparse, json, math, re, shutil, sys, tempfile
from pathlib import Path
import numpy as np

TAG = "champ"
N_FULL, N_50K, N_100K, N_20K, N_5K = 422786, 50000, 100000, 20000, 5000
R = dict(seeds_spread=0.015, headline_holds=0.990, headline_softens=0.985, width_pp=0.005, spur_max=0.005, t1r_tol=0.001,
         regress_frac=0.0005, regress_d16_d64_max=2, mem_drop=0.05, end_ce=0.02, exc_ema=0.90, exc_raw=0.50, exc_consec=2,
         mcnemar_alpha=0.01, eqr_t1r_20k=0.9885, eqr_n_20k=20000, c4_subset_tol=0.0025)
PRED = dict(cold16=(.949, .969), d64full=(.986, .995), eqr_t1r=(.983, .994), c4_full=(.9885, .9910))
W384 = {"C5": "C0", "C7": "C1", "C8": "C2"}          # the seed-matched width pairs
TRIPLE = ("C5", "C7", "C8"); NEW = ("C7", "C8")
LINES: list[str] = []
def say(s=""): LINES.append(str(s)); print(s)
def jload(p):
    p = Path(p); return json.loads(p.read_text()) if p.exists() else None
def pp(x): return "  -  " if x is None else f"{100*x:.2f}"

class Root:
    def __init__(self, root): self.r = Path(root)
    def chain(self, a, name): return self.r / f"sxeval_p{TAG}{a}" / name
    def filler(self, a, kind): return self.r / {"d64": f"filler_sxeval_p{TAG}{a}_full_t64", "d128": f"filler_sxeval_p{TAG}{a}_sub50000_t128",
                                                  "d256": f"filler_sxeval_p{TAG}{a}_sub50000_t256"}[kind]
    def scan32(self, a): return self.r / f"sxscan_p{TAG}{a}"
    def scan128(self, a): return self.r / f"filler_sxscan128_p{TAG}{a}"
    def eqr128(self): return self.r / "filler_sxscan128_pport_eqr"
    def pdir(self, a): return self.r / f"pretrain{TAG}_{a}"
    def summ(self, d): return jload(Path(d) / "summary_all.json")
    def recs(self, d):
        q = Path(d) / "records_all.npz"
        return dict(np.load(q, allow_pickle=True)) if q.exists() else None
    def acc(self, d): s = self.summ(d); return None if not s else s.get("exact_acc")

def exact_mcnemar(b, c):
    """Two-sided exact McNemar p on the discordant counts (b, c)."""
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    lp = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2) for i in range(k + 1)]
    m = max(lp); tail = m + math.log(sum(math.exp(x - m) for x in lp))
    return min(1.0, 2 * math.exp(tail))

def paired(ra, rb, key="cold_exact"):
    """Exact pairing by idx -> (diff = mean(a) - mean(b), b_only_a, c_only_b, n, p)."""
    if ra is None or rb is None: return None
    ia, ib = np.asarray(ra["idx"]), np.asarray(rb["idx"])
    common, pa, pb = np.intersect1d(ia, ib, return_indices=True)
    if len(common) == 0: return None
    ea = np.asarray(ra[key]).astype(bool)[pa]; eb = np.asarray(rb[key]).astype(bool)[pb]
    b = int(np.sum(ea & ~eb)); c = int(np.sum(~ea & eb))
    return dict(diff=float(ea.mean() - eb.mean()), only_a=b, only_b=c, n=int(len(common)), p=exact_mcnemar(b, c))

def selected_exact(z):
    """Per puzzle: the exact bit of the draw with the smallest finite residual (t1r), and any-draw exact (verified)."""
    ex = np.asarray(z["mi_exact_k"]).astype(bool); rs = np.asarray(z["mi_resid_k"]).astype(np.float64)
    rs = np.where(np.isfinite(rs), rs, np.inf); sel = np.argmin(rs, axis=1)
    return ex[np.arange(len(sel)), sel], ex.any(axis=1)

def spurious(z):
    ex = np.asarray(z["mi_exact_k"]).astype(bool); rs = np.asarray(z["mi_resid_k"]).astype(np.float64); fin = np.isfinite(rs)
    ef, wf = ex & fin, (~ex) & fin
    if ef.sum() == 0 or wf.sum() == 0: return None
    return float(np.mean(rs[wf] <= np.median(rs[ef])))

def argv_of(R_, a):
    cfg = jload(R_.pdir(a) / "config.json"); av = (cfg or {}).get("argv")
    return av if isinstance(av, list) else None

def strip_argv(av):
    out, i, remat = [], 0, False
    while i < len(av):
        t = av[i]
        if t in ("--out", "--steps", "--seed"): i += 2; continue
        if t == "--remat": remat = True; i += 1; continue
        out.append(t); i += 1
    return out, remat

def metrics(R_, a):
    p = R_.pdir(a) / "metrics.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []

def excursion(R_, a):
    mon = sorted({m["step"]: m for m in (r["monitor"] for r in metrics(R_, a) if "monitor" in r)}.values(), key=lambda m: m["step"])
    if not mon: return None
    first = next((i for i, m in enumerate(mon) if (m.get("val_t16_ema") or 0) >= R["exc_ema"]), None)
    if first is None: return dict(letter="NONE", steps=[], note="EMA never >= .90")
    run, best, cur = [], [], []
    for m in mon[first + 1:]:
        if (m.get("val_t16") if m.get("val_t16") is not None else 1.0) < R["exc_raw"]: cur.append(m["step"])
        else:
            if len(cur) > len(best): best = cur
            cur = []
    if len(cur) > len(best): best = cur
    return dict(letter="EXCURSION" if len(best) >= R["exc_consec"] else "NONE", steps=best)

def band(x, b): return "n/a" if x is None else ("HIT" if b[0] <= x <= b[1] else ("MISS-ABOVE" if x > b[1] else "MISS-BELOW"))

def analyze(root):
    LINES.clear(); R_ = Root(root); V: dict = {}
    say("THE PAPER'S FINAL RUNS — REGISTERED VERDICT (tools/analyze_paperfinal.py; registry at the top of the file)")
    # ---------- INTEGRITY ----------
    errs = []
    base, _ = strip_argv(argv_of(R_, "C5") or [])
    for a, s in (("C7", "1"), ("C8", "2")):
        av = argv_of(R_, a)
        if av is None: errs.append(f"{a}: no config argv"); continue
        body, remat = strip_argv(av)
        if body != base: errs.append(f"{a}: argv differs from C5's beyond --seed/--out/--steps/--remat")
        if "--seed" not in av or av[av.index("--seed") + 1] != s: errs.append(f"{a}: --seed != {s}")
        if remat: say(f"  (labeled) {a} ran with --remat (the registered numerics-equivalent OOM retry)")
    gates = []
    for a in ("C0", "C1", "C2", "C5", "C7", "C8"):
        gates += [(R_.chain(a, "full_vsel_t16"), {N_FULL}), (R_.filler(a, "d64"), {N_FULL})]
    for a in NEW:
        gates += [(R_.chain(a, "full_final_t16"), {N_FULL, N_50K}), (R_.chain(a, "full_vsel_t16_alt"), {N_50K}), (R_.chain(a, "full_vsel_t64"), {N_100K}),
                  (R_.chain(a, "sub20k_t128"), {N_20K}), (R_.chain(a, "sub5k_t256"), {N_5K}), (R_.scan32(a), {N_5K})]
    for a in TRIPLE: gates += [(R_.filler(a, "d128"), {N_50K}), (R_.filler(a, "d256"), {N_50K})]
    gates += [(R_.eqr128(), {N_5K}), (R_.filler("C4", "d64"), {N_FULL})]
    for d, ns in gates:
        s = R_.summ(d)
        if s is None: errs.append(f"missing {Path(d).relative_to(R_.r)}"); continue
        if s.get("n") not in ns: errs.append(f"{Path(d).relative_to(R_.r)} n={s.get('n')} not in {sorted(ns)}")
    for a in NEW:
        paths = set()
        for d in [R_.chain(a, n) for n in ("full_vsel_t16", "full_vsel_t16_alt", "full_vsel_t64", "sub20k_t128", "sub5k_t256")] + [R_.scan32(a)] + [R_.filler(a, k) for k in ("d64", "d128", "d256")]:
            s = R_.summ(d)
            if s and s.get("ckpt"): paths.add(s["ckpt"])
        if len(paths) > 1: errs.append(f"{a}: vsel rows on {len(paths)} checkpoints {sorted(paths)}")
    for d, kk in [(R_.scan32(a), 32) for a in NEW] + [(R_.eqr128(), 128)]:
        s = R_.summ(d)
        if s and (s.get("k_init") != kk or s.get("t_total") != 64): errs.append(f"{Path(d).name}: k_init/t_total {s.get('k_init')}/{s.get('t_total')} != {kk}/64")
    ze = R_.recs(R_.eqr128()); idx_e = None if ze is None else frozenset(np.asarray(ze["idx"]).tolist())
    for a in ("C0", "C1", "C2", "C3", "C4", "C5", "C6"):
        z = R_.recs(R_.scan128(a))
        if z is not None and idx_e is not None and frozenset(np.asarray(z["idx"]).tolist()) != idx_e: errs.append(f"EqR 5k idx set != {a}'s k128 idx set")
    V["INTEGRITY"] = "PASS" if not errs else "FAIL: " + "; ".join(errs)
    say(f"  INTEGRITY            {V['INTEGRITY']}")
    # ---------- R-PF-1 / R-PF-2 ----------
    c16 = {a: R_.acc(R_.chain(a, "full_vsel_t16")) for a in TRIPLE}
    c64 = {a: R_.acc(R_.filler(a, "d64")) for a in TRIPLE}
    d128 = {a: R_.acc(R_.filler(a, "d128")) for a in TRIPLE}; d256 = {a: R_.acc(R_.filler(a, "d256")) for a in TRIPLE}
    def ms(d):
        v = [x for x in d.values() if x is not None]
        return (None, None) if len(v) < 3 else (float(np.mean(v)), (max(v) - min(v)) / 2)
    m16, h16 = ms(c16); m64, h64 = ms(c64)
    if m16 is None: V["R-PF-1 SEEDS-W192"] = "NO-DATA"
    else: V["R-PF-1 SEEDS-W192"] = ("TIGHT" if 2 * h16 <= R["seeds_spread"] else "WIDE") + f" (D16 spread {100*2*h16:.2f} pp; D64-full spread {pp(2*h64) if h64 is not None else '-'} pp)"
    if m64 is None: V["R-PF-2 HEADLINE-W192"] = "NO-DATA"
    else:
        L = "HOLDS" if m64 >= R["headline_holds"] else ("SOFTENS" if m64 >= R["headline_softens"] else "FALLS")
        m1, h1 = ms(d128); m2, h2 = ms(d256)
        V["R-PF-2 HEADLINE-W192"] = f"{L} (D16 full {pp(m16)} +/- {pp(h16)} · D64 full {pp(m64)} +/- {pp(h64)} · D128 50k {pp(m1)} +/- {pp(h1)} · D256 50k {pp(m2)} +/- {pp(h2)})"
    say(f"  R-PF-1 SEEDS-W192    {V['R-PF-1 SEEDS-W192']}"); say(f"  R-PF-2 HEADLINE-W192 {V['R-PF-2 HEADLINE-W192']}")
    # ---------- R-PF-3 WIDTH-PAIRED ----------
    def width_letter(rowfn):
        ds, parts = [], []
        for a, b in W384.items():
            r = paired(R_.recs(rowfn(a)), R_.recs(rowfn(b)))
            if r is None: return "NO-DATA", []
            ds.append(r["diff"]); parts.append(f"{a}-{b} {100*r['diff']:+.2f} ({r['only_a']}/{r['only_b']}, p {r['p']:.1e})")
        md = float(np.mean(ds))
        if md >= R["width_pp"] and all(d > 0 for d in ds): L = "WIDTH-DOWN-PAYS"
        elif md <= -R["width_pp"] and all(d < 0 for d in ds): L = "WIDTH-DOWN-COSTS"
        elif abs(md) < R["width_pp"]: L = "PARITY"
        else: L = "MIXED"
        return f"{L} (mean {100*md:+.2f} pp; " + "; ".join(parts) + ")", ds
    V["R-PF-3 WIDTH-PAIRED"], _ = width_letter(lambda a: R_.filler(a, "d64"))
    V["R-PF-3 (D16, descriptive)"], _ = width_letter(lambda a: R_.chain(a, "full_vsel_t16"))
    say(f"  R-PF-3 WIDTH-PAIRED  {V['R-PF-3 WIDTH-PAIRED']}"); say(f"         at D16         {V['R-PF-3 (D16, descriptive)']}")
    # ---------- R-PF-4 SELECTOR ----------
    outs, allc = [], True
    for a in NEW:
        z, s = R_.recs(R_.scan32(a)), R_.summ(R_.scan32(a))
        if z is None or s is None: outs.append(f"{a} NO-DATA"); allc = False; continue
        t1r, ver = selected_exact(z); sp = spurious(z)
        clean = sp is not None and sp <= R["spur_max"] and t1r.mean() >= ver.mean() - R["t1r_tol"]
        allc = allc and clean
        outs.append(f"{a} {'CLEAN' if clean else 'DIRTY'} (spurious {pp(sp)} %, t1r@32 {pp(t1r.mean())}, verified@32 {pp(ver.mean())})")
    V["R-PF-4 SELECTOR-W192"] = ("SELECTOR-CLEAN | " if allc else "") + " | ".join(outs)
    say(f"  R-PF-4 SELECTOR      {V['R-PF-4 SELECTOR-W192']}")
    # ---------- R-PF-5 DEPTH ----------
    outs = []
    for a in NEW:
        rows = [R_.chain(a, n) for n in ("full_vsel_t16", "full_vsel_t64", "sub20k_t128", "sub5k_t256")] + [R_.filler(a, k) for k in ("d64", "d128", "d256")]
        mono, seen = True, 0
        for d in rows:
            s = R_.summ(d)
            if s and s.get("depth_regressions") is not None and s.get("n"):
                seen += 1; mono = mono and int(s["depth_regressions"]) <= R["regress_frac"] * int(s["n"])
        ra, rb = R_.recs(R_.chain(a, "full_vsel_t16")), R_.recs(R_.filler(a, "d64"))
        reg = None
        if ra is not None and rb is not None:
            r = paired(ra, rb); reg = r["only_a"] if r else None
        if seen == 0: outs.append(f"{a} NO-DATA"); continue
        outs.append(f"{a} {'MONOTONE' if mono else 'REGRESSES'}" + ("+ZERO-REGRESSION" if reg is not None and reg <= R["regress_d16_d64_max"] else "")
                    + f" (D16->D64 full: {reg} lost)")
    V["R-PF-5 DEPTH-W192"] = " | ".join(outs); say(f"  R-PF-5 DEPTH         {V['R-PF-5 DEPTH-W192']}")
    # ---------- R-PF-6 MEMORIZATION / R-PF-7 EXTENSION / R-PF-8 EXCURSION ----------
    outs6, outs7, outs8 = [], [], []
    for a in TRIPLE:
        rows = [r for r in metrics(R_, a) if "loss" in r]
        ce = float(np.mean([r.get("ce_in", np.nan) for r in rows[-5:]])) if rows else None
        vc, fc = R_.acc(R_.chain(a, "full_vsel_t16")), R_.acc(R_.chain(a, "full_final_t16"))
        tags = (["END-CE"] if ce is not None and ce < R["end_ce"] else []) + (["VSEL-FINAL-DROP"] if vc is not None and fc is not None and vc - fc > R["mem_drop"] else [])
        outs6.append(f"{a} {'+'.join(tags) if tags else 'NOT-MEMORIZED'} (vsel-final {pp(None if vc is None or fc is None else vc - fc)} pp, end ce {'-' if ce is None else f'{ce:.3f}'})")
        ext = (R_.pdir(a) / "EXTENDED.txt"); vb = (R_.pdir(a) / "val_best.txt")
        vstep = None
        if vb.exists():
            tok = vb.read_text().strip().split()[0] if vb.read_text().strip() else ""
            vstep = int(tok) if tok.isdigit() else None
        if ext.exists():
            m = re.search(r"from\s+(\d+)\s+to\s+(\d+)", ext.read_text()); frm = int(m.group(1)) if m else None
            outs7.append(f"{a} EXTENDED" + ("+PAID" if vstep is not None and frm is not None and vstep > frm else "") + f" (selected {vstep})")
        else: outs7.append(f"{a} NOT-EXTENDED (selected {vstep})")
        e = excursion(R_, a); outs8.append(f"{a} {'NO-DATA' if e is None else e['letter']}" + ("" if not e or not e["steps"] else f" {e['steps'][0]}-{e['steps'][-1]}"))
    c5e = excursion(R_, "C5")
    if c5e is None or c5e["letter"] != "EXCURSION": outs8.insert(0, "DETECTOR-INVALID (C5 does not read EXCURSION)")
    V["R-PF-6 MEMORIZATION"] = " | ".join(outs6); V["R-PF-7 EXTENSION"] = " | ".join(outs7); V["R-PF-8 EXCURSION"] = " | ".join(outs8)
    say(f"  R-PF-6 MEMORIZATION  {V['R-PF-6 MEMORIZATION']}"); say(f"  R-PF-7 EXTENSION     {V['R-PF-7 EXTENSION']}"); say(f"  R-PF-8 EXCURSION     {V['R-PF-8 EXCURSION']}")
    # ---------- R-PF-9 RESTART-PAIRED ----------
    eqr = None
    if ze is not None:
        t1e, vee = selected_exact(ze); p = float(t1e.mean()); s = R_.summ(R_.eqr128()) or {}
        summ_t1r = (s.get("t1r_at_k") or {}).get("128")
        se = math.sqrt(p * (1 - p) / len(t1e) + R["eqr_t1r_20k"] * (1 - R["eqr_t1r_20k"]) / R["eqr_n_20k"])
        cons = "CONSISTENT" if abs(p - R["eqr_t1r_20k"]) <= 3 * se else "SET-SHIFT"
        eqr = dict(t1r=p, verified=float(vee.mean()), spurious=spurious(ze), consistent=cons, summary_t1r=summ_t1r)
        parts = []
        for a in ("C0", "C1", "C2", "C3", "C4", "C5", "C6"):
            za = R_.recs(R_.scan128(a))
            if za is None: continue
            t1a, _v = selected_exact(za)
            r = paired(dict(idx=za["idx"], sel=t1a), dict(idx=ze["idx"], sel=t1e), key="sel")
            L = "AHEAD" if (r["p"] < R["mcnemar_alpha"] and r["diff"] > 0) else ("BEHIND" if (r["p"] < R["mcnemar_alpha"] and r["diff"] < 0) else "PARITY")
            parts.append(f"{a} {L} ({100*r['diff']:+.2f} pp, {r['only_a']}/{r['only_b']}, p {r['p']:.1e})"); eqr[f"vs_{a}"] = L
        xchk = "" if summ_t1r is None or abs(summ_t1r - p) < 1e-9 else f" [RECOMPUTE-MISMATCH summary {summ_t1r}]"
        V["R-PF-9 RESTART-PAIRED"] = f"EqR 5k t1r@128 {pp(p)} verified {pp(eqr['verified'])} spurious {pp(eqr['spurious'])} % {cons} vs its 20k 98.85{xchk} | " + " | ".join(parts)
    else: V["R-PF-9 RESTART-PAIRED"] = "NO-DATA"
    say(f"  R-PF-9 RESTART       {V['R-PF-9 RESTART-PAIRED']}")
    # ---------- R-PF-10 C4-FULL ----------
    c4f, c4s = R_.acc(R_.filler("C4", "d64")), R_.acc(R_.chain("C4", "full_vsel_t64"))
    if c4f is None: V["R-PF-10 C4-FULL"] = "NO-DATA"
    else:
        r = paired(R_.recs(R_.filler("C4", "d64")), R_.recs(R_.filler("C0", "d64")))
        L = "n/a" if c4s is None else ("CONSISTENT" if abs(c4f - c4s) <= R["c4_subset_tol"] else "SUBSET-SHIFT")
        V["R-PF-10 C4-FULL"] = f"{L} (full {pp(c4f)} vs 100k {pp(c4s)}" + ("" if r is None else f"; vs C0 full {100*r['diff']:+.2f} pp, {r['only_a']}/{r['only_b']}, p {r['p']:.1e}") + ")"
    say(f"  R-PF-10 C4-FULL      {V['R-PF-10 C4-FULL']}")
    # ---------- PREDICTIONS ----------
    sc = []
    for a in NEW:
        sc.append(f"{a} cold16 {band(c16.get(a), PRED['cold16'])}"); sc.append(f"{a} D64full {band(c64.get(a), PRED['d64full'])}")
    sc.append("SEEDS TIGHT " + ("HIT" if V["R-PF-1 SEEDS-W192"].startswith("TIGHT") else "MISS"))
    sc.append("HEADLINE HOLDS " + ("HIT" if V["R-PF-2 HEADLINE-W192"].startswith("HOLDS") else "MISS"))
    sc.append("WIDTH PAYS " + ("HIT" if V["R-PF-3 WIDTH-PAIRED"].startswith("WIDTH-DOWN-PAYS") else "MISS"))
    sc.append("SELECTOR-CLEAN " + ("HIT" if V["R-PF-4 SELECTOR-W192"].startswith("SELECTOR-CLEAN") else "MISS"))
    sc.append("DEPTH " + ("HIT" if V["R-PF-5 DEPTH-W192"].count("MONOTONE+ZERO-REGRESSION") == 2 else "MISS"))
    sc.append("NOT-MEMORIZED " + ("HIT" if all(f"{a} NOT-MEMORIZED" in V["R-PF-6 MEMORIZATION"] for a in NEW) else "MISS"))
    sc.append(">=1 EXTENDED " + ("HIT" if any(f"{a} EXTENDED" in V["R-PF-7 EXTENSION"] for a in NEW) else "MISS"))
    sc.append(">=1 EXCURSION " + ("HIT" if any(f"{a} EXCURSION" in V["R-PF-8 EXCURSION"] for a in NEW) else "MISS"))
    if eqr:
        sc.append(f"EqR t1r {band(eqr['t1r'], PRED['eqr_t1r'])}"); sc.append("EqR CONSISTENT " + ("HIT" if eqr["consistent"] == "CONSISTENT" else "MISS"))
        sc.append("EqR spurious<=.005 " + ("HIT" if eqr["spurious"] is not None and eqr["spurious"] <= .005 else "MISS"))
        sc.append("C5 AHEAD " + ("HIT" if eqr.get("vs_C5") == "AHEAD" else "MISS"))
    sc.append(f"C4 full {band(c4f, PRED['c4_full'])}" + (" CONSISTENT " + ("HIT" if V["R-PF-10 C4-FULL"].startswith("CONSISTENT") else "MISS")))
    V["PREDICTIONS"] = " · ".join(sc); say(f"  PREDICTIONS          {V['PREDICTIONS']}")
    out = Path(root) / "analysis"; out.mkdir(parents=True, exist_ok=True)
    (out / "paperfinal_verdict.txt").write_text("\n".join(LINES) + "\n"); (out / "paperfinal_verdict.json").write_text(json.dumps(V, indent=1))
    return V

# ---------- selftest (synthetic rows; every letter both ways) ----------
def _write_row(d, n, acc, rng, idx=None, extra=None, recs_extra=None, regress=0):
    d.mkdir(parents=True, exist_ok=True)
    idx = np.arange(n) if idx is None else idx
    ex = np.zeros(len(idx), bool); ex[: int(round(acc * len(idx)))] = True; rng.shuffle(ex)
    rec = dict(idx=idx, cold_exact=ex); rec.update(recs_extra or {})
    np.savez(d / "records_all.npz", **rec)
    s = dict(n=int(len(idx)), exact_acc=float(ex.mean()), depth_regressions=regress, ckpt=str(d.parent), t_total=64, k_init=0); s.update(extra or {})
    (d / "summary_all.json").write_text(json.dumps(s))
    return rec

def _mk(root, acc16, acc64, rng, width_shift=0.0, spur_bad=False, c5_exc=True, c7_exc=False, eqr_t1r=0.989, c4_full=0.9895,
        seed_mismatch=False, idx_mismatch=False, n_bad=False):
    R_ = Root(root)
    C5_ARGV = ["tools/pretrain.py", "--out", "X", "--cell", "dec", "--dec-width", "384", "--seed", "0", "--dec-width", "192", "--steps", "30000"]
    for a, s in (("C5", "0"), ("C7", "1"), ("C8", "2"), ("C0", "0"), ("C1", "1"), ("C2", "2"), ("C4", "0")):
        p = R_.pdir(a); p.mkdir(parents=True, exist_ok=True)
        av = list(C5_ARGV); av[av.index("--seed") + 1] = ("9" if (seed_mismatch and a == "C8") else s)
        (p / "config.json").write_text(json.dumps({"argv": av})); (p / "val_best.txt").write_text("024000 0.97 24000\n")
        mon = []
        for st in range(2000, 50001, 2000):
            raw = 0.95 if st >= 10000 else 0.5
            if (a == "C5" and c5_exc and 16000 <= st <= 26000) or (a == "C7" and c7_exc and 20000 <= st <= 24000): raw = 0.05
            if a == "C4" and st == 12000: raw = 0.39
            mon.append({"monitor": {"step": st, "val_t16": raw, "val_t16_ema": 0.95 if st >= 8000 else 0.3}})
        rows = [{"step": st, "loss": .5, "ce_in": .4} for st in range(46000, 50001, 1000)]
        (p / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in mon + rows) + "\n")
    ck = lambda a: str(R_.pdir(a) / "ckpt_024000.pkl")
    for a in ("C0", "C1", "C2"): _write_row(R_.chain(a, "full_vsel_t16"), 2000, acc16 - 0.01, rng, extra={"ckpt": ck(a), "n": 422786}); _write_row(R_.filler(a, "d64"), 2000, acc64 - width_shift, rng, extra={"ckpt": ck(a), "n": 422786})
    for i, a in enumerate(TRIPLE):
        _write_row(R_.chain(a, "full_vsel_t16"), 2000, acc16 + 0.002 * i, rng, extra={"ckpt": ck(a), "n": 422786 if not (n_bad and a == "C7") else 7})
        _write_row(R_.filler(a, "d64"), 2000, acc64 + 0.001 * i, rng, extra={"ckpt": ck(a), "n": 422786})
        _write_row(R_.filler(a, "d128"), 2000, .994, rng, extra={"ckpt": ck(a), "n": 50000}); _write_row(R_.filler(a, "d256"), 2000, .996, rng, extra={"ckpt": ck(a), "n": 50000})
        _write_row(R_.chain(a, "full_final_t16"), 2000, acc16 - 0.001, rng, extra={"ckpt": str(R_.pdir(a) / "ckpt_latest.pkl"), "n": 50000})
    for a in NEW:
        for nm, n in (("full_vsel_t16_alt", 50000), ("full_vsel_t64", 100000), ("sub20k_t128", 20000), ("sub5k_t256", 5000)):
            _write_row(R_.chain(a, nm), 2000, .99, rng, extra={"ckpt": ck(a), "n": n})
        k = 32; exk = (rng.random((500, k)) < .9).astype(np.int64); exk[:, 0] = 1; rs = rng.random((500, k))
        rs[exk.astype(bool)] *= (1.0 if spur_bad else 1e-5)   # clean: correct draws converge far below the wrong ones
        _write_row(R_.scan32(a), 500, .99, rng, extra={"ckpt": ck(a), "n": 5000, "k_init": 32, "t_total": 64}, recs_extra=dict(mi_exact_k=exk, mi_resid_k=rs))
    idx5 = np.arange(1000, 4000)   # 3,000 puzzles: a 1 pp paired gap = 30 discordant pairs (p << .01), like the real 5k rows
    def scan128(d, t1r_target, idx):
        exk = np.zeros((len(idx), 128), np.int8); rs = np.ones((len(idx), 128), np.float32)
        good = int(round(t1r_target * len(idx)))
        exk[:good, 0] = 1; rs[:, 0] = 0.001; exk[good:, 5] = 1   # the selected draw (0) exact on `good`; draw 5 exact elsewhere (coverage)
        return _write_row(d, len(idx), .99, rng, idx=idx, extra={"n": 5000, "k_init": 128, "t_total": 64, "t1r_at_k": {"128": good / len(idx)}}, recs_extra=dict(mi_exact_k=exk, mi_resid_k=rs))
    for a in ("C0", "C1", "C2", "C3", "C4", "C5", "C6"): scan128(R_.scan128(a), .998, idx5 if not (idx_mismatch and a == "C3") else idx5 + 1)
    scan128(R_.eqr128(), eqr_t1r, idx5)
    _write_row(R_.filler("C4", "d64"), 2000, c4_full, rng, extra={"ckpt": ck("C4"), "n": 422786}); _write_row(R_.chain("C4", "full_vsel_t64"), 2000, .9890, rng, extra={"ckpt": ck("C4"), "n": 100000})
    (R_.pdir("C7") / "EXTENDED.txt").write_text("EXTENDED from 30000 to 50000 (peak at 28000)\n"); (R_.pdir("C7") / "val_best.txt").write_text("046000 0.97 46000\n")

def selftest():
    ok = 0
    def run(**kw):
        d = Path(tempfile.mkdtemp(prefix="apf_")); rng = np.random.default_rng(7)
        _mk(d, kw.pop("acc16", .959), kw.pop("acc64", .992), rng, **kw)
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()): V = analyze(d)
        shutil.rmtree(d); return V
    V = run()
    assert V["INTEGRITY"] == "PASS", V["INTEGRITY"]; ok += 1
    assert V["R-PF-1 SEEDS-W192"].startswith("TIGHT"), V["R-PF-1 SEEDS-W192"]; ok += 1
    assert V["R-PF-2 HEADLINE-W192"].startswith("HOLDS"), V["R-PF-2 HEADLINE-W192"]; ok += 1
    assert V["R-PF-3 WIDTH-PAIRED"].startswith("PARITY"), V["R-PF-3 WIDTH-PAIRED"]; ok += 1
    assert V["R-PF-4 SELECTOR-W192"].startswith("SELECTOR-CLEAN"), V["R-PF-4 SELECTOR-W192"]; ok += 1
    assert V["R-PF-5 DEPTH-W192"].count("MONOTONE") == 2, V["R-PF-5 DEPTH-W192"]; ok += 1
    assert "C7 NOT-MEMORIZED" in V["R-PF-6 MEMORIZATION"], V["R-PF-6 MEMORIZATION"]; ok += 1
    assert "C7 EXTENDED+PAID" in V["R-PF-7 EXTENSION"] and "C8 NOT-EXTENDED" in V["R-PF-7 EXTENSION"], V["R-PF-7 EXTENSION"]; ok += 1
    assert "C5 EXCURSION" in V["R-PF-8 EXCURSION"] and "C7 NONE" in V["R-PF-8 EXCURSION"] and "DETECTOR-INVALID" not in V["R-PF-8 EXCURSION"], V["R-PF-8 EXCURSION"]; ok += 1
    assert "CONSISTENT" in V["R-PF-9 RESTART-PAIRED"] and "C5 AHEAD" in V["R-PF-9 RESTART-PAIRED"], V["R-PF-9 RESTART-PAIRED"]; ok += 1
    assert V["R-PF-10 C4-FULL"].startswith("CONSISTENT"), V["R-PF-10 C4-FULL"]; ok += 1
    V = run(width_shift=0.02)
    assert V["R-PF-3 WIDTH-PAIRED"].startswith("WIDTH-DOWN-PAYS"), V["R-PF-3 WIDTH-PAIRED"]; ok += 1
    V = run(width_shift=-0.02)
    assert V["R-PF-3 WIDTH-PAIRED"].startswith("WIDTH-DOWN-COSTS"), V["R-PF-3 WIDTH-PAIRED"]; ok += 1
    V = run(acc64=.987)
    assert V["R-PF-2 HEADLINE-W192"].startswith("SOFTENS"), V["R-PF-2 HEADLINE-W192"]; ok += 1
    V = run(acc64=.980)
    assert V["R-PF-2 HEADLINE-W192"].startswith("FALLS"), V["R-PF-2 HEADLINE-W192"]; ok += 1
    V = run(spur_bad=True)
    assert "DIRTY" in V["R-PF-4 SELECTOR-W192"] and not V["R-PF-4 SELECTOR-W192"].startswith("SELECTOR-CLEAN"), V["R-PF-4 SELECTOR-W192"]; ok += 1
    V = run(c5_exc=False)
    assert "DETECTOR-INVALID" in V["R-PF-8 EXCURSION"], V["R-PF-8 EXCURSION"]; ok += 1
    V = run(c7_exc=True)
    assert "C7 EXCURSION" in V["R-PF-8 EXCURSION"], V["R-PF-8 EXCURSION"]; ok += 1
    V = run(eqr_t1r=0.940)
    assert "SET-SHIFT" in V["R-PF-9 RESTART-PAIRED"] and "C5 AHEAD" in V["R-PF-9 RESTART-PAIRED"], V["R-PF-9 RESTART-PAIRED"]; ok += 1
    V = run(eqr_t1r=0.998)
    assert "C5 PARITY" in V["R-PF-9 RESTART-PAIRED"], V["R-PF-9 RESTART-PAIRED"]; ok += 1
    V = run(c4_full=0.9800)
    assert V["R-PF-10 C4-FULL"].startswith("SUBSET-SHIFT"), V["R-PF-10 C4-FULL"]; ok += 1
    V = run(seed_mismatch=True)
    assert V["INTEGRITY"].startswith("FAIL") and "C8: --seed != 2" in V["INTEGRITY"], V["INTEGRITY"]; ok += 1
    V = run(idx_mismatch=True)
    assert "EqR 5k idx set != C3" in V["INTEGRITY"], V["INTEGRITY"]; ok += 1
    V = run(n_bad=True)
    assert "n=7" in V["INTEGRITY"], V["INTEGRITY"]; ok += 1
    assert abs(exact_mcnemar(10, 10) - 1.0) < 1e-9 and exact_mcnemar(0, 20) < 1e-5 and abs(exact_mcnemar(3, 1) - 0.625) < 1e-9; ok += 1
    print(f"selftest OK: {ok}/{ok} checks")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1] / "runs")); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    selftest() if a.selftest else analyze(a.root)
