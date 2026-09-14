#!/usr/bin/env python3
# Ledger: THE WIDTH-192 LONG RUN — the frozen analyzer (registration: Documentation/Plan_2026-09-14_W192_Long.md; the rules below are locked
# verbatim at the registration commit and adjudicate byte-untouched; `--selftest` exercises every letter both ways). The questions (the PI,
# 2026-09-14): trained past 50k, does the width-192 cell find a better point before it memorizes; where does its memorization begin; how do
# the training clocks scale with width (widths 192 / 384 / 512 on identical instruments) — for the ARC DEC port's width choice.
#
# INSTRUMENTS (per grid, EMA weights, 16 iterations, single pass): H = exact count on the 10,000 held-out train-file puzzles (split val);
# T = exact count on the 1,000 training puzzles (split train); the memorization gap = T/1,000 - H/10,000. Test rows: the 50,000-puzzle test
# subsample (seed 20260822) at D16 and D64 — never the full test set (the PI).
# REGISTRY (R) — the decision rules:
#   INTEGRITY  I1 the long run's argv == the champion C5's except out/steps/remat (seed 0, width 192). I2 every monitor row at step <= 50,000
#              equals the champion C5's; monitor rows at every 2k from 52k to 150k; resumes.txt holds 50000. I3 ckpt_046000 and ckpt_050000
#              sha256-identical to the champion C5's. I4 every row: n (10,000 held-out / 1,000 train / 50,000 test), split, EMA, t_total, the
#              grid in the checkpoint path equals the row's grid, the run in the path equals the row's run; test rows subsample 50,000 at
#              seed 20260822. I5 coverage: C5 held-out at 4k-8k (new) + 10k-50k (the C8 extension's rows) + 52k-150k (new), C5 train-1k at
#              4k-150k, every wide run (C1, A5, A7, A8) at WIDE_GRIDS on both instruments. I6 all test rows on one puzzle set, contained in
#              the paper's 46k reference rows.
#   R-L1 LATER-BETTER   the smoothed held-out curve W(c) = mean H over {c-2k, c, c+2k}: early = max W over centers 12k-48k, late = max W
#                       over centers 54k-148k; D = (late - early) / 100 pp. D >= +0.5 LATER-BETTER · D <= -0.5 EARLIER-BEST · else SAME-PLATEAU.
#   R-L2 ONSET-W192     the onset on C5's held-out curve (4k-150k): the first grid after the peak from which H stays >= 150 counts (1.5 pp)
#                       under the running maximum at every later grid, >= 3 grids to the end. <= 64k EARLY · 66k-90k LR-CLOCK-BAND ·
#                       92k-146k LATE-BAND · none NONE-BY-150K. The fit onset (the gap >= its value at the peak + .03 at every later grid,
#                       >= 3 grids) reported beside it.
#   R-L3 CLOCK-SCALING  held-out onsets (the R-L2 definition) of the wide runs. rho192 = onset(C5) / median(onset(C1), onset(A5)):
#                       < 2.65 WIDTH-LR-CLOCK · >= 2.65 PARAM-CLOCK; C5 without an onset: the bound 146k / median >= 2.65 ->
#                       PARAM-CLOCK-OR-SLOWER, else UNREADABLE. rho512 = onset(A7) / onset(A8) (seed 1, plain recipe): < 1.60
#                       WIDTH-LR-CLOCK · >= 1.60 PARAM-CLOCK. A missing onset on a denominator run -> NO-DATA.
#   R-L4 WINDOW         Wn = onset / peak on the held-out curve (C5 without an onset: the bound 146k / peak). TWO-RATES-SUPPORTED iff
#                       Wn(A8) < median Wn(C1, A5, A7) < Wn(C5); else NOT-SUPPORTED; NO-DATA if a run lacks an onset (C5 excepted).
#   R-L5 GAP-ORDER      the memorization gap at 50k: gap(A8) > median gap(C1, A5, A7) > gap(C5) -> GAP-WIDTH-ORDERED, else NOT-ORDERED.
#   R-L6 TEST-50K       candidates g_val (the held-out argmax over >= 10k, earliest tie; the letter of record), g_mon (the registered monitor
#                       pick over 2k-150k, cross-checked by a replay) and the final 150k grid, each paired by idx with the paper's 46k rows on
#                       the 50k subsample at D16 and D64: per depth p < .01 and D > 0 UP · p < .01 and D < 0 DOWN · else FLAT; both UP
#                       BETTER · both DOWN WORSE · both FLAT SAME · else MIXED; a candidate at 46k -> NO-MOVE.
#   PREDICTIONS (credences in the plan): R-L1 SAME-PLATEAU; R-L2 LATE-BAND; R-L3 rho192 PARAM-CLOCK(-OR-SLOWER), rho512 PARAM-CLOCK;
#     R-L4 TWO-RATES-SUPPORTED; R-L5 GAP-WIDTH-ORDERED; R-L6 g_val SAME; the final 150k grid WORSE.
"""
  .venv/bin/python tools/analyze_c5l.py --champ-root runs/_paperfinal_pull/stage --c8x-root runs/_c8x_pull/x/runs --x-root runs/_c5l_pull/x/runs [--out DIR]
  .venv/bin/python tools/analyze_c5l.py --selftest
"""
from __future__ import annotations
import argparse, hashlib, json, re, shutil, sys, tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_paperfinal import paired              # frozen at 7a99d9e, selftested
from c5l_curves import curve, select, onset, smooth_max   # frozen with this registration, selftested

N_VAL, N_TR1K, N_TEST, SUB_SEED = 10000, 1000, 50000, 20260822
EXT_FROM, EXT_TO, STEP, PAPER = 50000, 150000, 2000, 46000
WIDE = ("C1", "A5", "A7", "A8")
WIDE_GRIDS = tuple(list(range(4000, 30001, 2000)) + [34000, 38000, 42000, 46000, 50000])
PATHNAME = {"C5": "pretrainchamp_C5", "C1": "pretrainchamp_C1", "A5": "pretrainfinalA_A5", "A7": "pretrainfinalA_A7", "A8": "pretrainfinalA_A8"}
R = dict(later=50, drop=150, tail=3, early_hi=64000, lr_lo=66000, lr_hi=90000, late_lo=92000, late_hi=146000, rho192=2.65, rho512=1.60,
         gap_fit=0.03, alpha=0.01, pick_min=10000)
LINES: list[str] = []
def say(s=""): LINES.append(str(s)); print(s)
def jload(p): p = Path(p); return json.loads(p.read_text()) if p.exists() else None
def recs(d): q = Path(d) / "records_all.npz"; return dict(np.load(q, allow_pickle=True)) if q.exists() else None
def sha(p): p = Path(p); return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
def step_in(s): m = re.search(r"ckpt_(\d+)", str(s or "")); return int(m.group(1)) if m else None
def kfmt(s): return "-" if s is None else f"{s // 1000}k"

def monitor(pdir):
    mon = {}
    p = Path(pdir) / "metrics.jsonl"
    if p.exists():
        for l in p.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                if "monitor" in r: mon[int(r["monitor"]["step"])] = r["monitor"]
    return mon

def replay_pick(pdir):
    mon = monitor(pdir); banked = {step_in(p.name) for p in Path(pdir).glob("ckpt_[0-9]*.pkl")}
    c = [((m.get("val_t16_ema") or 0.0, m.get("val_t16") or 0.0), s) for s, m in mon.items() if s in banked]
    if not c: return None
    best = max(k for k, _ in c); return min(s for k, s in c if k == best)

def argv_body(av):
    if isinstance(av, dict): return {k: v for k, v in av.items() if k not in ("out", "steps", "remat")}, av.get("seed")
    return None, None

def fit_onset(h, t):
    """The first grid after the held-out peak from which gap >= gap(peak) + .03 at every later grid (>= 3 grids)."""
    hd, td = dict(h), dict(t); grids = [s for s in sorted(hd) if s in td]
    if not grids: return None
    pk = select([(s, hd[s]) for s in grids]); g0 = td[pk] / N_TR1K - hd[pk] / N_VAL
    gap = [(s, td[s] / N_TR1K - hd[s] / N_VAL) for s in grids]
    for i, (s, _) in enumerate(gap):
        if s <= pk or len(gap) - i < R["tail"]: continue
        if all(g >= g0 + R["gap_fit"] for _, g in gap[i:]): return s
    return None

def depth_letter(r):
    if r is None: return None
    return "UP" if (r["p"] < R["alpha"] and r["diff"] > 0) else ("DOWN" if (r["p"] < R["alpha"] and r["diff"] < 0) else "FLAT")

def analyze(champ, c8x, x, out_dir=None):
    LINES.clear(); CH, C8, X = Path(champ), Path(c8x), Path(x); V: dict = {}
    say("THE WIDTH-192 LONG RUN — REGISTERED READING (tools/analyze_c5l.py; registry at the top of the file)")
    errs = []
    cC5, xC5 = CH / "pretrainchamp_C5", X / "pretrainchamp_C5"
    a_c, _ = argv_body((jload(cC5 / "config.json") or {}).get("argv")); a_x, s_x = argv_body((jload(xC5 / "config.json") or {}).get("argv"))
    if a_c is None or a_x is None: errs.append("I1 no argv")
    elif a_c != a_x or str(s_x) != "0": errs.append("I1 argv differs from the champion C5's beyond out/steps/remat (or seed != 0)")
    mc, mx = monitor(cC5), monitor(xC5)
    for s in [s for s in mc if s <= EXT_FROM]:
        if s not in mx or (mx[s].get("val_t16"), mx[s].get("val_t16_ema")) != (mc[s].get("val_t16"), mc[s].get("val_t16_ema")): errs.append(f"I2 monitor row {s} differs"); break
    if any(s not in mx for s in range(EXT_FROM + STEP, EXT_TO + 1, STEP)): errs.append("I2 the long run's monitor rows incomplete")
    if "50000" not in ((xC5 / "resumes.txt").read_text() if (xC5 / "resumes.txt").exists() else ""): errs.append("I2 resumes.txt lacks 50000")
    for g in (46000, 50000):
        if sha(cC5 / f"ckpt_{g:06d}.pkl") is None or sha(cC5 / f"ckpt_{g:06d}.pkl") != sha(xC5 / f"ckpt_{g:06d}.pkl"): errs.append(f"I3 ckpt_{g:06d} not byte-identical")
    def check_rows(root, prefix, kind, run, grids, n, split):
        for g in grids:
            s = jload(Path(root) / f"{prefix}_{kind}_p{run}_s{g:06d}" / "summary_all.json")
            if s is None: errs.append(f"I5 missing {prefix}_{kind} {run} {g}"); return
            if s.get("n") != n or s.get("split") != split or not s.get("ema") or s.get("t_total") != 16 or step_in(s.get("ckpt")) != g or PATHNAME[run] not in str(s.get("ckpt")):
                errs.append(f"I4 {prefix}_{kind} {run} {g}"); return
    check_rows(X, "c5l", "val", "C5", [4000, 6000, 8000] + list(range(EXT_FROM + STEP, EXT_TO + 1, STEP)), N_VAL, "val")
    check_rows(C8, "c8x", "val", "C5", range(10000, 50001, 2000), N_VAL, "val")
    check_rows(X, "c5l", "tr1k", "C5", range(4000, EXT_TO + 1, STEP), N_TR1K, "train")
    for run in WIDE:
        check_rows(X, "c5l", "val", run, WIDE_GRIDS, N_VAL, "val"); check_rows(X, "c5l", "tr1k", run, WIDE_GRIDS, N_TR1K, "train")
    tests = {}
    for d in sorted(X.glob("c5l_test_pC5_*")):
        m = re.match(r"c5l_test_pC5_(d16|d64)_s(\d+)$", d.name); s = jload(d / "summary_all.json")
        if not m or s is None: continue
        dep, g = m.group(1), int(m.group(2))
        if s.get("n") != N_TEST or s.get("subsample") != N_TEST or int(s.get("subsample_seed") or -1) != SUB_SEED or s.get("split") != "test" or s.get("t_total") != int(dep[1:]) or not s.get("ema") or step_in(s.get("ckpt")) != g:
            errs.append(f"I4 test {d.name}"); continue
        tests[(dep, g)] = d
    ref = {"d16": CH / "sxeval_pchampC5" / "full_vsel_t16", "d64": CH / "filler_sxeval_pchampC5_full_t64"}
    idsets = {frozenset(np.asarray(recs(d)["idx"]).tolist()) for d in tests.values() if recs(d) is not None}
    if len(idsets) > 1: errs.append("I6 test rows on different puzzle sets")
    for dep, d in ref.items():
        rr = recs(d)
        if tests and (rr is None or not all(i <= frozenset(np.asarray(rr["idx"]).tolist()) for i in idsets)): errs.append(f"I6 test puzzles not inside the {dep} reference"); break
    V["INTEGRITY"] = "PASS" if not errs else "FAIL: " + "; ".join(errs[:8]) + (f" (+{len(errs) - 8} more)" if len(errs) > 8 else "")
    say(f"  INTEGRITY            {V['INTEGRITY']}")
    # ---------- the curves ----------
    H = {"C5": curve(X, "val", "C5", ("c5l",), N_VAL) + []}
    h5 = dict(curve(X, "val", "C5", ("c5l",), N_VAL)); h5.update({s: k for s, k in curve(C8, "val", "C5", ("c8x",), N_VAL) if s not in h5})
    H["C5"] = sorted(h5.items())
    T = {"C5": curve(X, "tr1k", "C5", ("c5l",), N_TR1K)}
    for run in WIDE: H[run] = curve(X, "val", run, ("c5l",), N_VAL); T[run] = curve(X, "tr1k", run, ("c5l",), N_TR1K)
    has_long = any(st > EXT_FROM for st, _ in H["C5"])   # the width-192 readings need the long run's own held-out rows, not the reused 10k-50k alone
    if not has_long: H["C5"], T["C5"] = [], []
    ons = {r: (onset(H[r], R["drop"], R["tail"]) if H[r] else None) for r in H}
    pks = {r: select(H[r]) for r in H}
    # ---------- R-L1 ----------
    e = smooth_max(H["C5"], 12000, 48000, STEP); l = smooth_max(H["C5"], 54000, 148000, STEP)
    if e is None or l is None: V["R-L1 LATER-BETTER"] = "NO-DATA"
    else:
        D = (l[1] - e[1]) / 100.0
        L = "LATER-BETTER" if l[1] - e[1] >= R["later"] else ("EARLIER-BEST" if e[1] - l[1] >= R["later"] else "SAME-PLATEAU")
        h = dict(H["C5"])
        V["R-L1 LATER-BETTER"] = (f"{L} (smoothed late max at {kfmt(l[0])} {l[1]/100:.2f} % vs early max at {kfmt(e[0])} {e[1]/100:.2f} %: {D:+.2f} pp; raw held-out peak "
                                  f"{kfmt(pks['C5'])} {h.get(pks['C5'], 0)/100:.2f} %; 150k {h.get(EXT_TO, 0)/100:.2f} %)")
    say(f"  R-L1 LATER-BETTER    {V['R-L1 LATER-BETTER']}")
    # ---------- R-L2 ----------
    o5 = ons["C5"]; fo5 = fit_onset(H["C5"], T["C5"]) if H["C5"] and T["C5"] else None
    if not H["C5"]: V["R-L2 ONSET-W192"] = "NO-DATA"
    else:
        L = ("NONE-BY-150K" if o5 is None else "EARLY" if o5 <= R["early_hi"] else "LR-CLOCK-BAND" if R["lr_lo"] <= o5 <= R["lr_hi"] else "LATE-BAND" if R["late_lo"] <= o5 <= R["late_hi"] else "LATE-BAND")
        V["R-L2 ONSET-W192"] = f"{L} (held-out onset {kfmt(o5)}; peak {kfmt(pks['C5'])}; fit onset {kfmt(fo5)})"
    say(f"  R-L2 ONSET-W192      {V['R-L2 ONSET-W192']}")
    # ---------- R-L3 ----------
    w384 = [ons[r] for r in ("C1", "A5") if ons.get(r) is not None]
    if not w384 or not H["C5"]: L3a = "NO-DATA"
    else:
        med = float(np.median(w384))
        if o5 is not None:
            rho = o5 / med; L3a = ("WIDTH-LR-CLOCK" if rho < R["rho192"] else "PARAM-CLOCK") + f" (rho192 {rho:.2f} = {kfmt(o5)} / {med/1000:.0f}k)"
        else:
            lb = R["late_hi"] / med; L3a = ("PARAM-CLOCK-OR-SLOWER" if lb >= R["rho192"] else "UNREADABLE") + f" (rho192 > {lb:.2f} = 146k / {med/1000:.0f}k)"
    if ons.get("A7") is None or ons.get("A8") is None: L3b = "NO-DATA"
    else:
        rho = ons["A7"] / ons["A8"]; L3b = ("WIDTH-LR-CLOCK" if rho < R["rho512"] else "PARAM-CLOCK") + f" (rho512 {rho:.2f} = {kfmt(ons['A7'])} / {kfmt(ons['A8'])})"
    V["R-L3 CLOCK-SCALING"] = f"w192 vs w384: {L3a} | w384 vs w512: {L3b} | onsets " + ", ".join(f"{r} {kfmt(ons.get(r))}" for r in ("C5",) + WIDE) + " | peaks " + ", ".join(f"{r} {kfmt(pks.get(r))}" for r in ("C5",) + WIDE)
    say(f"  R-L3 CLOCK-SCALING   {V['R-L3 CLOCK-SCALING']}")
    # ---------- R-L4 ----------
    def wn(r):
        if not H.get(r) or pks.get(r) in (None, 0): return None
        if ons.get(r) is not None: return ons[r] / pks[r]
        return (R["late_hi"] / pks[r]) if r == "C5" else None
    W4 = {r: wn(r) for r in ("C5",) + WIDE}
    if any(W4[r] is None for r in W4): V["R-L4 WINDOW"] = "NO-DATA (" + ", ".join(f"{r} {'-' if W4[r] is None else f'{W4[r]:.2f}'}" for r in W4) + ")"
    else:
        m384 = float(np.median([W4["C1"], W4["A5"], W4["A7"]]))
        L = "TWO-RATES-SUPPORTED" if W4["A8"] < m384 < W4["C5"] else "NOT-SUPPORTED"
        V["R-L4 WINDOW"] = f"{L} (onset/peak: w512 A8 {W4['A8']:.2f} · w384 median {m384:.2f} (C1 {W4['C1']:.2f}, A5 {W4['A5']:.2f}, A7 {W4['A7']:.2f}) · w192 C5 {W4['C5']:.2f}{'' if ons['C5'] is not None else ' (bound)'})"
    say(f"  R-L4 WINDOW          {V['R-L4 WINDOW']}")
    # ---------- R-L5 ----------
    def gap50(r):
        h, t = dict(H.get(r, [])), dict(T.get(r, []))
        return None if 50000 not in h or 50000 not in t else t[50000] / N_TR1K - h[50000] / N_VAL
    G5 = {r: gap50(r) for r in ("C5",) + WIDE}
    if any(v is None for v in G5.values()): V["R-L5 GAP-ORDER"] = "NO-DATA"
    else:
        m384 = float(np.median([G5["C1"], G5["A5"], G5["A7"]]))
        L = "GAP-WIDTH-ORDERED" if G5["A8"] > m384 > G5["C5"] else "NOT-ORDERED"
        V["R-L5 GAP-ORDER"] = f"{L} (train-1k minus held-out at 50k: w512 {100*G5['A8']:+.2f} · w384 median {100*m384:+.2f} · w192 {100*G5['C5']:+.2f} pp)"
    say(f"  R-L5 GAP-ORDER       {V['R-L5 GAP-ORDER']}")
    # ---------- R-L6 ----------
    gval = select([(s, k) for s, k in H["C5"] if s >= R["pick_min"]]) if H["C5"] else None
    vb = (xC5 / "val_best.txt"); gmon = int(vb.read_text().split()[0]) if vb.exists() and vb.read_text().split() and vb.read_text().split()[0].isdigit() else None
    rp = replay_pick(xC5)
    def cand(g):
        if g is None: return "NO-DATA"
        if g == PAPER: return "NO-MOVE"
        ls, det = [], []
        for dep in ("d16", "d64"):
            if (dep, g) not in tests: return "NO-DATA"
            r = paired(recs(tests[(dep, g)]), recs(ref[dep])); lt = depth_letter(r); ls.append(lt)
            det.append(f"{dep.upper()} {100*r['diff']:+.2f} pp ({r['only_a']}/{r['only_b']}, n {r['n']}, p {r['p']:.1e}) {lt}")
        L = "BETTER" if ls == ["UP", "UP"] else ("WORSE" if ls == ["DOWN", "DOWN"] else ("SAME" if ls == ["FLAT", "FLAT"] else "MIXED"))
        return f"{L} [{'; '.join(det)}]"
    V["R-L6 TEST-50K"] = f"g_val={kfmt(gval)} {cand(gval)} | g_mon={kfmt(gmon)} {cand(gmon)} (replay {kfmt(rp)}{'' if rp == gmon else ' MISMATCH'}) | final={kfmt(EXT_TO)} {cand(EXT_TO)}"
    say(f"  R-L6 TEST-50K        {V['R-L6 TEST-50K']}")
    # ---------- PREDICTIONS ----------
    hit = lambda cond: "HIT" if cond else "MISS"
    sc = [f"R-L1 SAME-PLATEAU {hit(V['R-L1 LATER-BETTER'].startswith('SAME-PLATEAU'))}", f"R-L2 LATE-BAND {hit(V['R-L2 ONSET-W192'].startswith('LATE-BAND'))}",
          f"R-L3 rho192 PARAM {hit('w192 vs w384: PARAM-CLOCK' in V['R-L3 CLOCK-SCALING'])}", f"R-L3 rho512 PARAM {hit('w384 vs w512: PARAM-CLOCK' in V['R-L3 CLOCK-SCALING'])}",
          f"R-L4 TWO-RATES {hit(V['R-L4 WINDOW'].startswith('TWO-RATES-SUPPORTED'))}", f"R-L5 ORDERED {hit(V['R-L5 GAP-ORDER'].startswith('GAP-WIDTH-ORDERED'))}",
          f"R-L6 g_val SAME {hit(V['R-L6 TEST-50K'].startswith(f'g_val={kfmt(gval)} SAME') or V['R-L6 TEST-50K'].startswith(f'g_val={kfmt(gval)} NO-MOVE'))}",
          f"R-L6 final WORSE {hit(f'final={kfmt(EXT_TO)} WORSE' in V['R-L6 TEST-50K'])}"]
    V["PREDICTIONS"] = " · ".join(sc); say(f"  PREDICTIONS          {V['PREDICTIONS']}")
    V["curves"] = {r: dict(held_out=H[r], train1k=T.get(r, []), onset=ons.get(r), peak=pks.get(r)) for r in H}
    out = Path(out_dir) if out_dir else X.parent / "analysis"; out.mkdir(parents=True, exist_ok=True)
    (out / "c5l_verdict.txt").write_text("\n".join(LINES) + "\n"); (out / "c5l_verdict.json").write_text(json.dumps(V, indent=1))
    return V

# ---------- selftest (synthetic roots; every letter both ways) ----------
def _curve_counts(grids, peak, top, onset_at=None, rise=0.004, fall=0.012, base=0.80):
    out = {}
    for g in grids:
        v = top - rise * max(0, (peak - g) / 2000) if g <= peak else top
        if onset_at is not None and g >= onset_at: v = top - 0.02 - fall * ((g - onset_at) / 2000)
        out[g] = int(round(max(v, base - 0.3) * N_VAL))
    return out

def _row(d, n, acc, split, ckpt, extra=None, idx=None, bits=None):
    d.mkdir(parents=True, exist_ok=True)
    s = dict(n=n, exact_acc=acc, split=split, ema=True, t_total=16, ckpt=ckpt); s.update(extra or {})
    (d / "summary_all.json").write_text(json.dumps(s))
    if idx is not None: np.savez(d / "records_all.npz", idx=idx, cold_exact=bits)

def _mk(root, rng, c5_peak=60000, c5_onset=110000, c5_top=0.965, early_top=0.955, early_level=None, wide=None, gap_c5=0.01, gap_w=(0.08, 0.05, 0.05, 0.05),
        test_gain=(150, 150), final_gain=(-150, -150), gval_at_paper=False, bad=None):
    CH, C8, X = root / "ch", root / "c8x", root / "x"
    wide = wide or {"C1": (24000, 36000), "A5": (20000, 40000), "A7": (22000, 30000), "A8": (12000, 16000)}
    argv = {"out": "x", "steps": 50000, "seed": 0, "dec_width": 192, "lr": 1e-4, "remat": True}
    cC5 = CH / "pretrainchamp_C5"; cC5.mkdir(parents=True)
    mon = [json.dumps({"monitor": {"step": s, "val_t16_ema": 0.95 if s == 46000 else 0.94, "val_t16": 0.93}}) for s in range(2000, 50001, 2000)]
    (cC5 / "metrics.jsonl").write_text("\n".join(mon) + "\n"); (cC5 / "config.json").write_text(json.dumps({"argv": argv}))
    for g in (46000, 50000): (cC5 / f"ckpt_{g:06d}.pkl").write_bytes(f"c5-{g}".encode())
    xC5 = X / "pretrainchamp_C5"; xC5.mkdir(parents=True)
    ax = dict(argv, steps=150000, remat=False, seed=(9 if bad == "argv" else 0))
    (xC5 / "config.json").write_text(json.dumps({"argv": ax}))
    monx = mon + [json.dumps({"monitor": {"step": s, "val_t16_ema": 0.96 if s == 60000 else 0.93, "val_t16": 0.92}}) for s in range(52000, 150001, 2000)]
    if bad == "monitor": monx[0] = monx[0].replace("0.94", "0.941")
    (xC5 / "metrics.jsonl").write_text("\n".join(monx) + "\n"); (xC5 / "resumes.txt").write_text("30000\n37500\n50000\n")
    for g in range(2000, 150001, 2000): (xC5 / f"ckpt_{g:06d}.pkl").write_bytes(f"c5-{g}".encode() if not (bad == "sha" and g == 46000) else b"x")
    gmon = 60000; (xC5 / "val_best.txt").write_text(f"{gmon:06d} 0.96 {gmon}\n")
    grids5 = [4000, 6000, 8000] + list(range(10000, 150001, 2000))
    h = _curve_counts(grids5, c5_peak, c5_top, c5_onset)
    for g in grids5:
        if 10000 <= g <= 46000: h[g] = min(h[g], int(early_top * N_VAL))
        if early_level is not None and 20000 <= g <= 48000: h[g] = int(early_level * N_VAL)   # an early plateau above the later curve
    if gval_at_paper: h = {g: (int(0.97 * N_VAL) if g == 46000 else min(v, int(0.95 * N_VAL))) for g, v in h.items()}
    for g in grids5:
        pre, root_ = ("c8x", C8) if 10000 <= g <= 50000 else ("c5l", X)
        _row(root_ / f"{pre}_val_pC5_s{g:06d}", N_VAL if bad != "n" or g != 60000 else 9999, h[g] / N_VAL, "val", f"runs/pretrainchamp_C5/ckpt_{g:06d}.pkl")
    for g in range(4000, 150001, 2000):
        gap = gap_c5 + (0.06 * (g - c5_onset) / 40000 if (c5_onset is not None and g >= c5_onset) else 0.0)
        _row(X / f"c5l_tr1k_pC5_s{g:06d}", N_TR1K, min(1.0, h[g] / N_VAL + gap), "train", f"runs/pretrainchamp_C5/ckpt_{g:06d}.pkl")
    for (run, (pk, on)), gw in zip(wide.items(), gap_w):
        hw = _curve_counts(WIDE_GRIDS, pk, 0.95, on, fall=0.02)
        for g in WIDE_GRIDS:
            if bad == "missing" and run == "A7" and g == 50000: continue
            _row(X / f"c5l_val_p{run}_s{g:06d}", N_VAL, hw[g] / N_VAL, "val", f"runs/{PATHNAME[run]}/ckpt_{g:06d}.pkl")
            extra_gap = gw + (0.3 * (g - on) / 40000 if g >= on else 0.0)
            _row(X / f"c5l_tr1k_p{run}_s{g:06d}", N_TR1K, min(1.0, hw[g] / N_VAL + extra_gap), "train", f"runs/{PATHNAME[run]}/ckpt_{g:06d}.pkl")
    nref = 60000; idx_ref = np.arange(nref)
    base16 = rng.random(nref) < 0.955; base64 = rng.random(nref) < 0.99
    _row(CH / "sxeval_pchampC5" / "full_vsel_t16", 422786, float(base16.mean()), "test", "runs/pretrainchamp_C5/ckpt_046000.pkl", idx=idx_ref, bits=base16)
    _row(CH / "filler_sxeval_pchampC5_full_t64", 422786, float(base64.mean()), "test", "runs/pretrainchamp_C5/ckpt_046000.pkl", idx=idx_ref, bits=base64)
    sub = np.sort(rng.choice(nref, N_TEST, replace=False))
    gval = 46000 if gval_at_paper else select([(g, h[g]) for g in grids5 if g >= 10000])
    def flip(bits, k):
        b = bits.copy()
        if k > 0: w = np.where(~b)[0][:k]; b[w] = True
        elif k < 0: w = np.where(b)[0][:(-k)]; b[w] = False
        return b
    for g, (k16, k64) in {gval: test_gain, gmon: test_gain, 150000: final_gain}.items():
        if g == PAPER: continue
        for dep, bits, k in (("d16", base16, k16), ("d64", base64, k64)):
            bb = flip(bits[sub], k)
            _row(X / f"c5l_test_pC5_{dep}_s{g:06d}", N_TEST if bad != "fulltest" else 422786, float(bb.mean()), "test", f"runs/pretrainchamp_C5/ckpt_{g:06d}.pkl",
                 extra={"t_total": int(dep[1:]), "subsample": N_TEST if bad != "fulltest" else None, "subsample_seed": SUB_SEED}, idx=idx_ref[sub], bits=bb)
    return CH, C8, X

def selftest():
    import io, contextlib
    ok = 0
    def run(**kw):
        d = Path(tempfile.mkdtemp(prefix="ac5l_")); rng = np.random.default_rng(5)
        CH, C8, X = _mk(d, rng, **kw)
        with contextlib.redirect_stdout(io.StringIO()): V = analyze(CH, C8, X, d / "out")
        shutil.rmtree(d); return V
    V = run()
    assert V["INTEGRITY"] == "PASS", V["INTEGRITY"]; ok += 1
    assert V["R-L1 LATER-BETTER"].startswith("LATER-BETTER"), V["R-L1 LATER-BETTER"]; ok += 1
    assert V["R-L2 ONSET-W192"].startswith("LATE-BAND"), V["R-L2 ONSET-W192"]; ok += 1
    assert "w192 vs w384: PARAM-CLOCK (rho192" in V["R-L3 CLOCK-SCALING"] and "w384 vs w512: PARAM-CLOCK (rho512 1.88" in V["R-L3 CLOCK-SCALING"], V["R-L3 CLOCK-SCALING"]; ok += 1
    assert V["R-L4 WINDOW"].startswith("TWO-RATES-SUPPORTED"), V["R-L4 WINDOW"]; ok += 1
    assert V["R-L5 GAP-ORDER"].startswith("GAP-WIDTH-ORDERED"), V["R-L5 GAP-ORDER"]; ok += 1
    assert " BETTER [" in V["R-L6 TEST-50K"].split("|")[0] and "final=150k WORSE" in V["R-L6 TEST-50K"], V["R-L6 TEST-50K"]; ok += 1
    V = run(c5_onset=70000, c5_peak=50000, c5_top=0.955, early_level=0.965)
    assert V["R-L2 ONSET-W192"].startswith("LR-CLOCK-BAND") and "w192 vs w384: WIDTH-LR-CLOCK" in V["R-L3 CLOCK-SCALING"] and V["R-L1 LATER-BETTER"].startswith("EARLIER-BEST"), (V["R-L2 ONSET-W192"], V["R-L3 CLOCK-SCALING"], V["R-L1 LATER-BETTER"]); ok += 1
    V = run(c5_onset=None, c5_peak=46000, c5_top=0.958, early_top=0.958)
    assert V["R-L2 ONSET-W192"].startswith("NONE-BY-150K") and "PARAM-CLOCK-OR-SLOWER" in V["R-L3 CLOCK-SCALING"] and V["R-L1 LATER-BETTER"].startswith("SAME-PLATEAU"), (V["R-L2 ONSET-W192"], V["R-L3 CLOCK-SCALING"], V["R-L1 LATER-BETTER"]); ok += 1
    V = run(c5_onset=58000, c5_peak=52000)
    assert V["R-L2 ONSET-W192"].startswith("EARLY"), V["R-L2 ONSET-W192"]; ok += 1
    V = run(c5_peak=60000, c5_onset=70000, wide={"C1": (24000, 36000), "A5": (20000, 40000), "A7": (22000, 26000), "A8": (18000, 20000)})   # w192 window 1.17 < the w384 median
    assert "w384 vs w512: WIDTH-LR-CLOCK (rho512 1.30" in V["R-L3 CLOCK-SCALING"] and V["R-L4 WINDOW"].startswith("NOT-SUPPORTED"), (V["R-L3 CLOCK-SCALING"], V["R-L4 WINDOW"]); ok += 1
    V = run(gap_c5=0.05, gap_w=(0.0, 0.0, 0.0, 0.0), wide={"C1": (24000, 60000), "A5": (20000, 60000), "A7": (22000, 60000), "A8": (12000, 60000)})   # no wide memorization by 50k
    assert V["R-L5 GAP-ORDER"].startswith("NOT-ORDERED"), V["R-L5 GAP-ORDER"]; ok += 1
    V = run(test_gain=(-200, -200), final_gain=(0, 0))
    assert " WORSE [" in V["R-L6 TEST-50K"].split("|")[0] and "final=150k SAME" in V["R-L6 TEST-50K"], V["R-L6 TEST-50K"]; ok += 1
    V = run(test_gain=(250, -250))
    assert " MIXED [" in V["R-L6 TEST-50K"].split("|")[0], V["R-L6 TEST-50K"]; ok += 1
    V = run(gval_at_paper=True)
    assert V["R-L6 TEST-50K"].startswith("g_val=46k NO-MOVE"), V["R-L6 TEST-50K"]; ok += 1
    for bad, frag in (("argv", "I1"), ("monitor", "I2 monitor row"), ("sha", "I3 ckpt_046000"), ("n", "I4 c5l_val C5 60000"), ("fulltest", "I4 test"), ("missing", "I5 missing c5l_val A7 50000")):
        V = run(bad=bad); assert V["INTEGRITY"].startswith("FAIL") and frag in V["INTEGRITY"], (bad, V["INTEGRITY"]); ok += 1
    d = Path(tempfile.mkdtemp(prefix="ac5l_empty_")); [(d / s).mkdir() for s in ("ch", "c8x", "x")]
    for g in range(10000, 50001, 2000):   # only the reused rows present: no width-192 reading may come from them alone
        _row(d / "c8x" / f"c8x_val_pC5_s{g:06d}", N_VAL, 0.95 if g == 46000 else 0.94, "val", f"runs/pretrainchamp_C5/ckpt_{g:06d}.pkl")
    with contextlib.redirect_stdout(io.StringIO()): V = analyze(d / "ch", d / "c8x", d / "x", d / "out")
    shutil.rmtree(d)
    assert V["INTEGRITY"].startswith("FAIL") and V["R-L1 LATER-BETTER"] == "NO-DATA" and V["R-L2 ONSET-W192"] == "NO-DATA" and V["R-L6 TEST-50K"].startswith("g_val=- NO-DATA") and V["R-L4 WINDOW"].startswith("NO-DATA"), V; ok += 1
    print(f"selftest OK: {ok}/{ok} checks")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--champ-root"); ap.add_argument("--c8x-root"); ap.add_argument("--x-root"); ap.add_argument("--out"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    selftest() if a.selftest else analyze(a.champ_root, a.c8x_root, a.x_root, a.out)
