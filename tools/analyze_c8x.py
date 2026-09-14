#!/usr/bin/env python3
# Ledger: THE C8 EXTENSION — the frozen analyzer (registration: Documentation/Plan_2026-09-14_C8_Extension.md; the rules below are locked
# verbatim at the registration commit and adjudicate byte-untouched; `--selftest` exercises every letter both ways). The question (the PI,
# 2026-09-14): C8, the width-192 seed the registered extension rule stopped at 30k by one monitor puzzle, trained on to 50k — is there a
# better point than its selected 22k grid, and was the rule's noise the reason (a best point near 46k suspected)?
#
# REGISTRY (R) — the decision rules:
#   INTEGRITY  I1 the extension's argv == the paper-final C8's except --out/--steps/--remat (seed 2). I2 the resume: every monitor row at
#              step <= 30000 equals the paper-final C8's; monitor rows at every 2k from 32000 to 50000; resumes.txt holds 30000; EXTENDED.txt.
#              I3 ckpt_022000.pkl and ckpt_030000.pkl byte-identical (sha256) to the paper-final's. I4 n-gates: the battery (D16 full 422,786;
#              final 50,000 or 422,786; alt 50,000; D64 100,000; D128 20,000; D256 5,000; the k32 scan 5,000), the filler (D64 422,786; D128
#              and D256 50,000), every val row 10,000 on split val at D16 EMA, the extra rows (d16full / d64full 422,786; d64sub100k 100,000).
#              I5 one checkpoint path over the g_mon rows; each val / extra row's grid equals its name. I6 val grids 10k..50k every 2k for C5,
#              C7, C8 (21 each). I7 paired sets: the extension's D16 full and D64 100k idx == the paper-final C8's.
#   R-X1 MONITOR-PICK   g_mon = the registered selector's grid over every banked grid (the full_vsel_t16 provenance, cross-checked by a
#                       replay: EMA key, raw second key, earliest tie). 22000 -> SAME; else MOVED(g_mon).
#   R-X2 VAL-PEAK       g_val = the maximum exact count on the 10k held-out set over C8's grids >= 10k, earliest tie. >= 40000 PEAK-LATE ·
#                       (30000, 40000) PEAK-MID · <= 30000 PEAK-EARLY. Plateau = grids within 50 puzzles (0.5 pp) of the maximum. 22k vs
#                       g_val paired on the 10k (exact McNemar): p < .01 and g_val ahead -> 22K-BELOW, else 22K-IN-PLATEAU. The PI's 46k:
#                       in the plateau -> HYP-IN-PLATEAU else HYP-OUT; paired vs 22k: p < .01 and ahead -> HYP-ABOVE-22K else HYP-NOT-ABOVE.
#   R-X3 BETTER-POINT   per candidate g in {g_val (the letter of record), g_mon, 46k}, g != 22k, on the test set paired with the paper-final
#                       22k rows: D16 full; D64 full where the candidate has it (g_mon: the filler; g_val: the extra row), else D64 on the
#                       100k vs the 22k's 100k row. Per depth: p < .01 and D > 0 UP · p < .01 and D < 0 DOWN · else FLAT. Both UP ->
#                       BETTER · both DOWN -> WORSE · both FLAT -> SAME · else MIXED; g == 22k -> NO-MOVE.
#   R-X4 SEED-OR-BUDGET R = (D64full(g_val) - D64full(22k)) / (D64full(C7) - D64full(22k)): R >= .75 BUDGET-MOSTLY · .25 < R < .75 BOTH ·
#                       R <= .25 SEED-MOSTLY (g_val == 22k -> R = 0). The same ratio with g_mon reported.
#   R-X5 TRIPLE-SENS    the width-192 D64 full mean with C8 := g_mon's row: >= .990 HOLDS · [.985, .990) SOFTENS · < .985 FALLS (descriptive
#                       sensitivity; the paper-final letters are unchanged). The same with g_val reported.
#   R-X6 W192-PLATEAU   g_val per arm (C5, C7, C8) on the 10k: span <= 8000 SHARED-PEAK else SEED-PEAKS; MONITOR-IN-PLATEAU k/3 = arms whose
#                       registered monitor pick (C5 46k, C7 46k: the paper-final provenance; C8 g_mon) lies in their own val plateau.
#   PREDICTIONS (credences in the plan): MOVED; PEAK-LATE; 22K-BELOW; HYP-IN-PLATEAU; g_val BETTER; BOTH; SOFTENS; SHARED-PEAK;
#     MONITOR-IN-PLATEAU >= 2/3; C8 at g_val D16 full in [.930, .950] and D64 full in [.980, .989].
"""
  .venv/bin/python tools/analyze_c8x.py --pf-root runs/_paperfinal_pull/stage --x-root runs/_c8x_pull/x/runs [--out DIR]
  .venv/bin/python tools/analyze_c8x.py --selftest
"""
from __future__ import annotations
import argparse, hashlib, json, re, shutil, sys, tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_paperfinal import exact_mcnemar, paired      # frozen at 7a99d9e, selftested
from c8x_valcurve import curve, select, plateau            # frozen with this registration, selftested

N_FULL, N_50K, N_100K, N_20K, N_5K, N_VAL = 422786, 50000, 100000, 20000, 5000, 10000
R = dict(alpha=0.01, late=40000, early=30000, plateau=50, budget_hi=0.75, budget_lo=0.25, holds=0.990, softens=0.985, span=8000,
         ext_from=30000, ext_to=50000, pf_grid=22000, hyp=46000, min_step=10000)
PRED = dict(d16=(.930, .950), d64=(.980, .989))
LINES: list[str] = []
def say(s=""): LINES.append(str(s)); print(s)
def jload(p): p = Path(p); return json.loads(p.read_text()) if p.exists() else None
def recs(d): q = Path(d) / "records_all.npz"; return dict(np.load(q, allow_pickle=True)) if q.exists() else None
def summ(d): return jload(Path(d) / "summary_all.json")
def acc(d): s = summ(d); return None if s is None else s.get("exact_acc")
def pp(x): return "  -  " if x is None else f"{100*x:.2f}"
def sha(p): p = Path(p); return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
def step_in(path): m = re.search(r"ckpt_(\d+)", str(path or "")); return int(m.group(1)) if m else None

def argv_body(av):
    if isinstance(av, dict): return {k: v for k, v in av.items() if k not in ("out", "steps", "remat")}, av.get("seed")
    if isinstance(av, list):
        out, i, seed = [], 0, None
        while i < len(av):
            t = av[i]
            if t in ("--out", "--steps"): i += 2; continue
            if t == "--remat": i += 1; continue
            if t == "--seed": seed = av[i + 1]
            out.append(t); i += 1
        return out, seed
    return None, None

def monitor(pdir):
    p = Path(pdir) / "metrics.jsonl"; mon = {}
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

def depth_letter(r):
    if r is None: return None
    return "UP" if (r["p"] < R["alpha"] and r["diff"] > 0) else ("DOWN" if (r["p"] < R["alpha"] and r["diff"] < 0) else "FLAT")

def analyze(pf_root, x_root, out_dir=None):
    LINES.clear(); PF, X = Path(pf_root), Path(x_root); V: dict = {}
    say("THE C8 EXTENSION — REGISTERED READING (tools/analyze_c8x.py; registry at the top of the file)")
    pfC8, xC8 = PF / "pretrainchamp_C8", X / "pretrainchamp_C8"
    errs = []
    # ---------- INTEGRITY ----------
    a_pf, s_pf = argv_body((jload(pfC8 / "config.json") or {}).get("argv")); a_x, s_x = argv_body((jload(xC8 / "config.json") or {}).get("argv"))
    if a_x is None or a_pf is None: errs.append("I1 no argv")
    elif a_x != a_pf or str(s_x) != "2": errs.append("I1 argv differs from the paper-final C8's beyond --out/--steps/--remat (or seed != 2)")
    mp, mx = monitor(pfC8), monitor(xC8)
    for s in [s for s in mp if s <= R["ext_from"]]:
        if s not in mx or (mx[s].get("val_t16"), mx[s].get("val_t16_ema")) != (mp[s].get("val_t16"), mp[s].get("val_t16_ema")): errs.append(f"I2 monitor row {s} differs"); break
    if any(s not in mx for s in range(R["ext_from"] + 2000, R["ext_to"] + 1, 2000)): errs.append("I2 extension monitor rows incomplete")
    if "30000" not in ((xC8 / "resumes.txt").read_text() if (xC8 / "resumes.txt").exists() else ""): errs.append("I2 resumes.txt lacks 30000")
    if not (xC8 / "EXTENDED.txt").exists(): errs.append("I2 no EXTENDED.txt")
    for g in (22000, 30000):
        a, b = sha(pfC8 / f"ckpt_{g:06d}.pkl"), sha(xC8 / f"ckpt_{g:06d}.pkl")
        if a is None or a != b: errs.append(f"I3 ckpt_{g:06d} not byte-identical")
    xch = lambda n: X / "sxeval_pchampC8" / n
    gates = [(xch("full_vsel_t16"), {N_FULL}), (xch("full_final_t16"), {N_50K, N_FULL}), (xch("full_vsel_t16_alt"), {N_50K}), (xch("full_vsel_t64"), {N_100K}),
             (xch("sub20k_t128"), {N_20K}), (xch("sub5k_t256"), {N_5K}), (X / "sxscan_pchampC8", {N_5K}), (X / "filler_sxeval_pchampC8_full_t64", {N_FULL}),
             (X / "filler_sxeval_pchampC8_sub50000_t128", {N_50K}), (X / "filler_sxeval_pchampC8_sub50000_t256", {N_50K})]
    for d, ns in gates:
        s = summ(d)
        if s is None: errs.append(f"I4 missing {Path(d).name if Path(d).parent == X else Path(d).parent.name + '/' + Path(d).name}")
        elif s.get("n") not in ns: errs.append(f"I4 {Path(d).name} n={s.get('n')}")
    gmon = step_in((summ(xch("full_vsel_t16")) or {}).get("ckpt"))
    paths = {(summ(d) or {}).get("ckpt") for d in [xch(n) for n in ("full_vsel_t16", "full_vsel_t16_alt", "full_vsel_t64", "sub20k_t128", "sub5k_t256")]
             + [X / "sxscan_pchampC8"] + [X / f"filler_sxeval_pchampC8_{k}" for k in ("full_t64", "sub50000_t128", "sub50000_t256")]} - {None}
    if len(paths) > 1: errs.append(f"I5 g_mon rows on {len(paths)} checkpoints")
    val = {}
    for arm in ("C5", "C7", "C8"):
        c = curve(X, arm, N_VAL, R["min_step"]); val[arm] = c
        if [s for s, _, _ in c] != list(range(R["min_step"], R["ext_to"] + 1, 2000)): errs.append(f"I6 {arm} val grids {[s for s, _, _ in c][:3]}.. ({len(c)})")
        for s, _, _ in c:
            j = summ(X / f"c8x_val_p{arm}_s{s:06d}") or {}
            if j.get("split") != "val" or not j.get("ema") or j.get("t_total") != 16 or step_in(j.get("ckpt")) != s: errs.append(f"I4/I5 val row {arm} {s}"); break
    xr = {}
    for d in sorted(X.glob("c8x_xrow_pchampC8_*")):
        name = d.name.replace("c8x_xrow_pchampC8_", ""); s = summ(d) or {}; kind, st = name.rsplit("_s", 1)
        need = {"d16full": N_FULL, "d64full": N_FULL, "d64sub100k": N_100K}.get(kind)
        if s.get("n") != need or step_in(s.get("ckpt")) != int(st): errs.append(f"I4/I5 extra row {name}")
        xr[(kind, int(st))] = d
    for a_, b_ in ((xch("full_vsel_t16"), PF / "sxeval_pchampC8" / "full_vsel_t16"), (xch("full_vsel_t64"), PF / "sxeval_pchampC8" / "full_vsel_t64")):
        ra, rb = recs(a_), recs(b_)
        if ra is None or rb is None or not np.array_equal(np.sort(ra["idx"]), np.sort(rb["idx"])): errs.append(f"I7 paired set {Path(a_).name}")
    V["INTEGRITY"] = "PASS" if not errs else "FAIL: " + "; ".join(errs)
    say(f"  INTEGRITY            {V['INTEGRITY']}")
    # ---------- R-X1 ----------
    rp = replay_pick(xC8)
    V["R-X1 MONITOR-PICK"] = ("NO-DATA" if gmon is None else ("SAME" if gmon == R["pf_grid"] else f"MOVED({gmon})")) + f" (provenance {gmon}, replay {rp}{'' if rp == gmon else ' MISMATCH'})"
    say(f"  R-X1 MONITOR-PICK    {V['R-X1 MONITOR-PICK']}")
    # ---------- R-X2 ----------
    c8 = val["C8"]; gval = select(c8); pl = plateau(c8, R["plateau"])
    def val_pair(a, b):
        ra, rb = recs(X / f"c8x_val_pC8_s{a:06d}"), recs(X / f"c8x_val_pC8_s{b:06d}")
        return paired(ra, rb) if ra is not None and rb is not None else None
    if gval is None: V["R-X2 VAL-PEAK"] = "NO-DATA"
    else:
        L = "PEAK-LATE" if gval >= R["late"] else ("PEAK-EARLY" if gval <= R["early"] else "PEAK-MID")
        r22 = val_pair(gval, R["pf_grid"]); l22 = "22K-BELOW" if (r22 and r22["p"] < R["alpha"] and r22["diff"] > 0) else "22K-IN-PLATEAU"
        rh = val_pair(R["hyp"], R["pf_grid"]); lh = ("HYP-IN-PLATEAU" if R["hyp"] in pl else "HYP-OUT") + " · " + ("HYP-ABOVE-22K" if (rh and rh["p"] < R["alpha"] and rh["diff"] > 0) else "HYP-NOT-ABOVE")
        top = sorted(c8, key=lambda t: (-t[1], t[0]))[:3]
        V["R-X2 VAL-PEAK"] = (f"{L}({gval}) · {l22} · {lh} (plateau {pl[0] if pl else '-'}..{pl[-1] if pl else '-'} = {len(pl)} grids; top {', '.join(f'{s//1000}k {k}' for s, k, _ in top)} of {N_VAL}"
                              + (f"; g_val-22k {100*r22['diff']:+.2f} pp ({r22['only_a']}/{r22['only_b']}, p {r22['p']:.1e})" if r22 else "")
                              + (f"; 46k-22k {100*rh['diff']:+.2f} pp (p {rh['p']:.1e})" if rh else "") + ")")
    say(f"  R-X2 VAL-PEAK        {V['R-X2 VAL-PEAK']}")
    # ---------- R-X3 ----------
    pf16, pf64f, pf64s = PF / "sxeval_pchampC8" / "full_vsel_t16", PF / "filler_sxeval_pchampC8_full_t64", PF / "sxeval_pchampC8" / "full_vsel_t64"
    def rows_for(g):
        if g == gmon: return xch("full_vsel_t16"), X / "filler_sxeval_pchampC8_full_t64", "full"
        d16 = xr.get(("d16full", g))
        if ("d64full", g) in xr: return d16, xr[("d64full", g)], "full"
        return d16, xr.get(("d64sub100k", g)), "100k"
    def cand(g):
        if g is None: return "NO-DATA", {}
        if g == R["pf_grid"]: return "NO-MOVE", {}
        d16, d64, kind = rows_for(g)
        r16 = paired(recs(d16), recs(pf16)) if d16 else None
        r64 = paired(recs(d64), recs(pf64f if kind == "full" else pf64s)) if d64 else None
        l16, l64 = depth_letter(r16), depth_letter(r64)
        if l16 is None or l64 is None: return "NO-DATA", {}
        L = "BETTER" if l16 == l64 == "UP" else ("WORSE" if l16 == l64 == "DOWN" else ("SAME" if l16 == l64 == "FLAT" else "MIXED"))
        det = f"D16 full {pp(acc(d16))} vs {pp(acc(pf16))} {100*r16['diff']:+.2f} ({r16['only_a']}/{r16['only_b']}, p {r16['p']:.1e}) {l16}; D64 {kind} {pp(acc(d64))} vs {pp(acc(pf64f if kind == 'full' else pf64s))} {100*r64['diff']:+.2f} ({r64['only_a']}/{r64['only_b']}, p {r64['p']:.1e}) {l64}"
        return L, dict(det=det, d16=acc(d16), d64=acc(d64), kind=kind)
    L3, parts, info = {}, [], {}
    for tag, g in (("g_val", gval), ("g_mon", gmon), ("46k", R["hyp"])):
        L, inf = cand(g); L3[tag] = L; info[tag] = inf
        parts.append(f"{tag}={g} {L}" + (f" [{inf['det']}]" if inf else ""))
    V["R-X3 BETTER-POINT"] = " | ".join(parts); say(f"  R-X3 BETTER-POINT    {V['R-X3 BETTER-POINT']}")
    # ---------- R-X4 / R-X5 ----------
    d64_22, d64_c7, d64_c5 = acc(pf64f), acc(PF / "filler_sxeval_pchampC7_full_t64"), acc(PF / "filler_sxeval_pchampC5_full_t64")
    def d64full_of(g):
        if g is None: return None
        if g == R["pf_grid"]: return d64_22
        if g == gmon: return acc(X / "filler_sxeval_pchampC8_full_t64")
        return acc(xr[("d64full", g)]) if ("d64full", g) in xr else None
    def ratio(g):
        v = d64full_of(g)
        if v is None or d64_22 is None or d64_c7 is None or d64_c7 == d64_22: return None
        return (v - d64_22) / (d64_c7 - d64_22)
    rv, rm = ratio(gval), ratio(gmon)
    L4 = "NO-DATA" if rv is None else ("BUDGET-MOSTLY" if rv >= R["budget_hi"] else ("SEED-MOSTLY" if rv <= R["budget_lo"] else "BOTH"))
    V["R-X4 SEED-OR-BUDGET"] = f"{L4} (R with g_val {'-' if rv is None else f'{rv:+.2f}'}; with g_mon {'-' if rm is None else f'{rm:+.2f}'}; C7 D64 full {pp(d64_c7)}, C8@22k {pp(d64_22)})"
    say(f"  R-X4 SEED-OR-BUDGET  {V['R-X4 SEED-OR-BUDGET']}")
    def triple(g):
        v = d64full_of(g)
        if v is None or d64_c5 is None or d64_c7 is None: return None, None
        m = (v + d64_c5 + d64_c7) / 3; return m, (max(v, d64_c5, d64_c7) - min(v, d64_c5, d64_c7)) / 2
    tm, th = triple(gmon); tv, tvh = triple(gval)
    L5 = "NO-DATA" if tm is None else ("HOLDS" if tm >= R["holds"] else ("SOFTENS" if tm >= R["softens"] else "FALLS"))
    V["R-X5 TRIPLE-SENS"] = f"{L5} (with C8 := g_mon {pp(tm)} +/- {pp(th)}; with C8 := g_val {pp(tv)} +/- {pp(tvh)}; the registered triple 98.65 +/- 0.59 unchanged)"
    say(f"  R-X5 TRIPLE-SENS     {V['R-X5 TRIPLE-SENS']}")
    # ---------- R-X6 ----------
    picks = {"C5": step_in((summ(PF / "sxeval_pchampC5" / "full_vsel_t16") or {}).get("ckpt")), "C7": step_in((summ(PF / "sxeval_pchampC7" / "full_vsel_t16") or {}).get("ckpt")), "C8": gmon}
    gv = {a: select(val[a]) for a in ("C5", "C7", "C8")}; pls = {a: plateau(val[a], R["plateau"]) for a in ("C5", "C7", "C8")}
    if any(v is None for v in gv.values()): V["R-X6 W192-PLATEAU"] = "NO-DATA"
    else:
        span = max(gv.values()) - min(gv.values()); inside = [a for a in ("C5", "C7", "C8") if picks[a] in pls[a]]
        V["R-X6 W192-PLATEAU"] = (("SHARED-PEAK" if span <= R["span"] else "SEED-PEAKS") + f" (g_val C5 {gv['C5']}, C7 {gv['C7']}, C8 {gv['C8']}; span {span}) · MONITOR-IN-PLATEAU {len(inside)}/3 ["
                                  + "; ".join(f"{a} pick {picks[a]} plateau {pls[a][0]}..{pls[a][-1]}" for a in ("C5", "C7", "C8")) + "]")
    say(f"  R-X6 W192-PLATEAU    {V['R-X6 W192-PLATEAU']}")
    # ---------- PREDICTIONS ----------
    band = lambda x, b: "n/a" if x is None else ("HIT" if b[0] <= x <= b[1] else ("MISS-ABOVE" if x > b[1] else "MISS-BELOW"))
    sc = ["MOVED " + ("HIT" if V["R-X1 MONITOR-PICK"].startswith("MOVED") else "MISS"),
          "PEAK-LATE " + ("HIT" if V["R-X2 VAL-PEAK"].startswith("PEAK-LATE") else "MISS"),
          "22K-BELOW " + ("HIT" if "22K-BELOW" in V["R-X2 VAL-PEAK"] else "MISS"),
          "HYP-IN-PLATEAU " + ("HIT" if "HYP-IN-PLATEAU" in V["R-X2 VAL-PEAK"] else "MISS"),
          "g_val BETTER " + ("HIT" if L3.get("g_val") == "BETTER" else "MISS"),
          "BOTH " + ("HIT" if L4 == "BOTH" else "MISS"), "SOFTENS " + ("HIT" if L5 == "SOFTENS" else "MISS"),
          "SHARED-PEAK " + ("HIT" if V["R-X6 W192-PLATEAU"].startswith("SHARED-PEAK") else "MISS"),
          "MONITOR-IN-PLATEAU>=2 " + ("HIT" if re.search(r"MONITOR-IN-PLATEAU [23]/3", V["R-X6 W192-PLATEAU"]) else "MISS")]
    gi = info.get("g_val") or {}
    d16v = gi.get("d16") if gi else (acc(pf16) if gval == R["pf_grid"] else None); d64v = d64full_of(gval)
    sc += [f"g_val D16 full {band(d16v, PRED['d16'])}", f"g_val D64 full {band(d64v, PRED['d64'])}"]
    V["PREDICTIONS"] = " · ".join(sc); say(f"  PREDICTIONS          {V['PREDICTIONS']}")
    out = Path(out_dir) if out_dir else X.parent / "analysis"; out.mkdir(parents=True, exist_ok=True)
    (out / "c8x_verdict.txt").write_text("\n".join(LINES) + "\n"); (out / "c8x_verdict.json").write_text(json.dumps(V, indent=1))
    return V

# ---------- selftest (synthetic roots; every letter both ways) ----------
def _row(d, idx, bits, n=None, extra=None):
    d.mkdir(parents=True, exist_ok=True); bits = np.asarray(bits, bool)
    np.savez(d / "records_all.npz", idx=np.asarray(idx), cold_exact=bits)
    s = dict(n=int(n if n is not None else len(idx)), exact_acc=float(bits.mean())); s.update(extra or {})
    (d / "summary_all.json").write_text(json.dumps(s))

def _bits(n, acc_, rng, base=None, flips=0):
    if base is None:
        b = np.zeros(n, bool); b[: int(round(acc_ * n))] = True; rng.shuffle(b); return b
    b = base.copy(); idx = np.where(~b)[0][:flips] if flips > 0 else np.where(b)[0][: -flips]; b[idx] = flips > 0; return b

def _mk(root, rng, gmon=46000, val_peak=46000, x_gain16=300, x_gain64=120, argv_bad=False, resume_bad=False, sha_bad=False, c8_peak_22=False,
        c5_peak=46000, c7_peak=46000):
    PF, X = root / "pf", root / "x"
    n = 4000
    def pdir(base, arm, steps, mon_fn, seed="2"):
        d = base / f"pretrainchamp_{arm}"; d.mkdir(parents=True, exist_ok=True)
        for s in steps: (d / f"ckpt_{s:06d}.pkl").write_bytes(f"{arm}-{s}".encode())
        rows = [json.dumps({"monitor": {"step": s, "val_t16_ema": mon_fn(s), "val_t16": mon_fn(s) - .01}}) for s in steps]
        (d / "metrics.jsonl").write_text("\n".join(rows) + "\n")
        (d / "config.json").write_text(json.dumps({"argv": {"out": str(d), "steps": steps[-1], "seed": int(seed), "dec_width": 192, "remat": False}}))
        return d
    mon22 = lambda s: 0.95 if s == 22000 else 0.94
    pdir(PF, "C8", list(range(2000, 30001, 2000)), mon22)
    monx = lambda s: (0.96 if s == gmon else 0.94) if s > 30000 else mon22(s)
    dx = pdir(X, "C8", list(range(2000, 50001, 2000)), monx, seed="3" if argv_bad else "2")
    if resume_bad:
        t = (dx / "metrics.jsonl").read_text().replace('"val_t16_ema": 0.95', '"val_t16_ema": 0.951', 1); (dx / "metrics.jsonl").write_text(t)
    (dx / "resumes.txt").write_text("30000\n"); (dx / "EXTENDED.txt").write_text("EXTENDED from 30000 to 50000\n")
    if sha_bad: (dx / "ckpt_022000.pkl").write_bytes(b"different")
    full = np.arange(n); b16 = _bits(n, .93, rng); b64 = _bits(n, .98, rng)
    ck = lambda s: f"runs/pretrainchamp_C8/ckpt_{s:06d}.pkl"
    _row(PF / "sxeval_pchampC8" / "full_vsel_t16", full, b16, N_FULL, {"ckpt": ck(22000)})
    _row(PF / "sxeval_pchampC8" / "full_vsel_t64", full, b64, N_100K, {"ckpt": ck(22000)})
    _row(PF / "filler_sxeval_pchampC8_full_t64", full, b64, N_FULL, {"ckpt": ck(22000)})
    _row(PF / "filler_sxeval_pchampC7_full_t64", full, _bits(n, .988, rng), N_FULL); _row(PF / "filler_sxeval_pchampC5_full_t64", full, _bits(n, .9916, rng), N_FULL)
    for arm in ("C5", "C7"): _row(PF / f"sxeval_pchamp{arm}" / "full_vsel_t16", full, _bits(n, .95, rng), N_FULL, {"ckpt": f"runs/pretrainchamp_{arm}/ckpt_046000.pkl"})
    same = gmon == 22000
    g16 = b16 if same else _bits(n, 0, rng, b16, x_gain16); g64 = b64 if same else _bits(n, 0, rng, b64, x_gain64)
    for nm, nn, bb in (("full_vsel_t16", N_FULL, g16), ("full_vsel_t16_alt", N_50K, g16), ("full_vsel_t64", N_100K, g64), ("sub20k_t128", N_20K, g64), ("sub5k_t256", N_5K, g64)):
        _row(X / "sxeval_pchampC8" / nm, full, bb, nn, {"ckpt": ck(gmon)})
    _row(X / "sxeval_pchampC8" / "full_final_t16", full, g16, N_50K, {"ckpt": "runs/pretrainchamp_C8/ckpt_latest.pkl"})
    _row(X / "sxscan_pchampC8", full, g16, N_5K, {"ckpt": ck(gmon)})
    _row(X / "filler_sxeval_pchampC8_full_t64", full, g64, N_FULL, {"ckpt": ck(gmon)})
    for k in ("sub50000_t128", "sub50000_t256"): _row(X / f"filler_sxeval_pchampC8_{k}", full, g64, N_50K, {"ckpt": ck(gmon)})
    vb = _bits(N_VAL, .93, rng)
    for arm, peak in (("C8", 22000 if c8_peak_22 else val_peak), ("C5", c5_peak), ("C7", c7_peak)):
        for s in range(10000, 50001, 2000):
            bits = _bits(N_VAL, 0, rng, vb, 400) if s == peak else (_bits(N_VAL, 0, rng, vb, 380) if abs(s - peak) == 2000 else vb)
            _row(X / f"c8x_val_p{arm}_s{s:06d}", np.arange(N_VAL), bits, N_VAL, {"split": "val", "ema": True, "t_total": 16, "ckpt": f"/tmp/p/runs/pretrainchamp_{arm}/ckpt_{s:06d}.pkl"})
    gval = 22000 if c8_peak_22 else val_peak
    if gval not in (gmon, 22000):
        _row(X / f"c8x_xrow_pchampC8_d16full_s{gval:06d}", full, _bits(n, 0, rng, b16, x_gain16), N_FULL, {"ckpt": ck(gval)})
        _row(X / f"c8x_xrow_pchampC8_d64full_s{gval:06d}", full, _bits(n, 0, rng, b64, x_gain64), N_FULL, {"ckpt": ck(gval)})
    if 46000 not in (gmon, gval, 22000):
        _row(X / "c8x_xrow_pchampC8_d16full_s046000", full, _bits(n, 0, rng, b16, 10), N_FULL, {"ckpt": ck(46000)})
        _row(X / "c8x_xrow_pchampC8_d64sub100k_s046000", full, _bits(n, 0, rng, b64, 5), N_100K, {"ckpt": ck(46000)})
    for g in (22000, 30000): shutil.copy(PF / "pretrainchamp_C8" / f"ckpt_{g:06d}.pkl", X / "pretrainchamp_C8" / f"ckpt_{g:06d}.pkl") if not (sha_bad and g == 22000) else None
    return PF, X

def selftest():
    import io, contextlib
    ok = 0
    def run(**kw):
        d = Path(tempfile.mkdtemp(prefix="ac8x_")); rng = np.random.default_rng(11)
        PF, X = _mk(d, rng, **kw)
        with contextlib.redirect_stdout(io.StringIO()): V = analyze(PF, X, d / "out")
        shutil.rmtree(d); return V
    V = run()
    assert V["INTEGRITY"] == "PASS", V["INTEGRITY"]; ok += 1
    assert V["R-X1 MONITOR-PICK"].startswith("MOVED(46000)") and "MISMATCH" not in V["R-X1 MONITOR-PICK"], V["R-X1 MONITOR-PICK"]; ok += 1
    assert V["R-X2 VAL-PEAK"].startswith("PEAK-LATE(46000)") and "22K-BELOW" in V["R-X2 VAL-PEAK"] and "HYP-IN-PLATEAU" in V["R-X2 VAL-PEAK"], V["R-X2 VAL-PEAK"]; ok += 1
    assert "g_val=46000 BETTER" in V["R-X3 BETTER-POINT"] and "g_mon=46000 BETTER" in V["R-X3 BETTER-POINT"], V["R-X3 BETTER-POINT"]; ok += 1
    assert V["R-X6 W192-PLATEAU"].startswith("SHARED-PEAK") and "MONITOR-IN-PLATEAU 3/3" in V["R-X6 W192-PLATEAU"], V["R-X6 W192-PLATEAU"]; ok += 1
    V = run(gmon=22000, val_peak=40000)
    assert V["R-X1 MONITOR-PICK"].startswith("SAME") and "g_mon=22000 NO-MOVE" in V["R-X3 BETTER-POINT"], V; ok += 1
    assert "g_val=40000 BETTER" in V["R-X3 BETTER-POINT"] and "46k=46000" in V["R-X3 BETTER-POINT"], V["R-X3 BETTER-POINT"]; ok += 1
    V = run(x_gain16=0, x_gain64=0)
    assert "g_val=46000 SAME" in V["R-X3 BETTER-POINT"] and V["R-X4 SEED-OR-BUDGET"].startswith("SEED-MOSTLY"), (V["R-X3 BETTER-POINT"], V["R-X4 SEED-OR-BUDGET"]); ok += 1
    V = run(x_gain16=-300, x_gain64=-120)
    assert "g_val=46000 WORSE" in V["R-X3 BETTER-POINT"] and V["R-X5 TRIPLE-SENS"].startswith("FALLS"), (V["R-X3 BETTER-POINT"], V["R-X5 TRIPLE-SENS"]); ok += 1
    V = run(x_gain16=300, x_gain64=0)
    assert "g_val=46000 MIXED" in V["R-X3 BETTER-POINT"], V["R-X3 BETTER-POINT"]; ok += 1
    V = run(x_gain64=16)   # the fixture's C7 - C8@22k gap at D64 = 32 of 4,000 puzzles: a gain of 16 -> R = .50
    assert V["R-X4 SEED-OR-BUDGET"].startswith("BOTH"), V["R-X4 SEED-OR-BUDGET"]; ok += 1
    V = run(x_gain64=32)   # a gain of 32 -> R = 1.00
    assert V["R-X4 SEED-OR-BUDGET"].startswith("BUDGET-MOSTLY"), V["R-X4 SEED-OR-BUDGET"]; ok += 1
    V = run(c8_peak_22=True, gmon=22000)
    assert V["R-X2 VAL-PEAK"].startswith("PEAK-EARLY(22000)") and "22K-IN-PLATEAU" in V["R-X2 VAL-PEAK"] and "g_val=22000 NO-MOVE" in V["R-X3 BETTER-POINT"], V["R-X2 VAL-PEAK"]; ok += 1
    V = run(val_peak=34000, gmon=34000)
    assert V["R-X2 VAL-PEAK"].startswith("PEAK-MID(34000)") and "HYP-OUT" in V["R-X2 VAL-PEAK"], V["R-X2 VAL-PEAK"]; ok += 1
    V = run(c5_peak=12000, c7_peak=40000)   # C5 and C7 peak away from their 46k monitor picks (plateaus 10-14k and 38-42k); C8 in
    assert V["R-X6 W192-PLATEAU"].startswith("SEED-PEAKS") and "MONITOR-IN-PLATEAU 1/3" in V["R-X6 W192-PLATEAU"], V["R-X6 W192-PLATEAU"]; ok += 1
    V = run(argv_bad=True); assert V["INTEGRITY"].startswith("FAIL") and "I1" in V["INTEGRITY"], V["INTEGRITY"]; ok += 1
    import io as _io, contextlib as _cl
    _d = Path(tempfile.mkdtemp(prefix="ac8x_empty_")); (_d / "pf").mkdir(); (_d / "x").mkdir()
    with _cl.redirect_stdout(_io.StringIO()): V = analyze(_d / "pf", _d / "x", _d / "out")
    shutil.rmtree(_d)
    assert V["R-X1 MONITOR-PICK"].startswith("NO-DATA") and "MOVED MISS" in V["PREDICTIONS"] and V["INTEGRITY"].startswith("FAIL"), V; ok += 1
    V = run(resume_bad=True); assert "I2 monitor row" in V["INTEGRITY"], V["INTEGRITY"]; ok += 1
    V = run(sha_bad=True); assert "I3 ckpt_022000" in V["INTEGRITY"], V["INTEGRITY"]; ok += 1
    print(f"selftest OK: {ok}/{ok} checks")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--pf-root"); ap.add_argument("--x-root"); ap.add_argument("--out"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    selftest() if a.selftest else analyze(a.pf_root, a.x_root, a.out)
