#!/usr/bin/env python3
"""THE X5 LONG RUN — the registered verdict (Documentation/Plan_2026-09-17_X5_Long.md; FROZEN at the registration commit).

X5 = TRM's cell at hidden 160 under the full recipe, parameter-matched to the width-192 symmetric DEC. Its 50k row was read at the edge of
its budget. The long run resumes that state to the COMPUTE-MATCHED budget (960,000 steps) and is read here against the same triple.

The rules R-SP-1..5 are tools/analyze_sudokupend.py's, imported and unchanged. New here:
  INTEGRITY+  the budget reached; the long run IS X5's continuation (its monitor rows up to 50k equal the 50k run's; the first resume at
              50,000); the long cadence; the 50k reference rows on the identical puzzles
  R-XL-1 PLATEAU   on the EMA monitor: the running maximum's gain over the last plateau_window steps < plateau_pp -> PLATEAUED, else STILL-RISING
  R-XL-3 GAIN-16   the long run against the 50k row at 16 iterations on the identical 422,786: exact McNemar p < mcnemar_p AND |d| >= gain_floor
                   -> GAINED / LOST, else FLAT (a relative rule with its absolute floor)
  R-XL-6 DEC-VS-X  each width-192 seed against the long run at k = 128 on the identical 5,000 (the selected bit): AHEAD / BEHIND / PARITY
  COMPUTE (descriptive)  the long run's measured pace and its training compute as a fraction of one DEC seed's (dec_pace, 50k steps)

usage: python3 tools/analyze_x5long.py --root <stage>/runs --ref-root <the 50k stage>/runs [--out file]
       python3 tools/analyze_x5long.py --selftest
"""
import argparse, json, sys, tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_paperfinal as PF
import analyze_sudokupend as SP

ARM = "X5"
R = {
    "ext_from": 50_000, "ext_to": 960_000,   # the registered budget: 50k x 272.7 / 14.2 (the measured same-pod paces), rounded down to the cadence
    "mon": 10_000,                           # the long cadence: a monitor row + a grid every 10k steps past ext_from
    "plateau_window": 300_000, "plateau_pp": 0.02,   # R-XL-1: the running maximum gained < 2 pp (~ 10 of the 512 monitor puzzles) over the last 300k steps
    "gain_floor": 0.01, "mcnemar_p": 0.01,   # R-XL-3 / R-XL-6
    "dec_pace": 14.2, "dec_steps": 50_000,   # COMPUTE: the width-192 DEC's measured 8-chip pace (C7, C8) and budget
}


def monitor(root: PF.Root):
    mon = sorted({m["step"]: m for m in (r["monitor"] for r in PF.metrics(root, ARM) if "monitor" in r)}.values(), key=lambda m: m["step"])
    return [(m["step"], m.get("val_t16_ema")) for m in mon if m.get("val_t16_ema") is not None]


def plateau(ema):
    """(letter, gain, runmax_end, runmax_before) on the EMA monitor curve [(step, value)]."""
    end = ema[-1][0]; before = [v for s, v in ema if s <= end - R["plateau_window"]]
    if not before: return "NO-WINDOW", None, max(v for _, v in ema), None
    top, prev = max(v for _, v in ema), max(before); g = top - prev
    return ("PLATEAUED" if g < R["plateau_pp"] else "STILL-RISING"), g, top, prev


def letters(root: PF.Root, ref: PF.Root):
    base = SP.letters(root, ARM); L, probs = {}, []
    if base["INTEGRITY"] != "PASS": probs.append(base["INTEGRITY"].replace("FAIL: ", ""))
    ema = monitor(root); ema_ref = monitor(ref)
    if not ema: probs.append("no monitor rows")
    else:
        if ema[-1][0] != R["ext_to"]: probs.append(f"budget not reached: last monitor row {ema[-1][0]} != {R['ext_to']}")
        head = {s: v for s, v in ema if s <= R["ext_from"]}; href = {s: v for s, v in ema_ref if s <= R["ext_from"]}
        if not href: probs.append("the 50k reference curve is absent")
        elif head != href: probs.append(f"not X5's continuation: {sum(1 for s in href if head.get(s) != href[s])} of {len(href)} monitor rows up to {R['ext_from']} differ from the 50k run's")
        off = [s for s, _ in ema if s > R["ext_from"] and s % R["mon"]]
        if off: probs.append(f"{len(off)} monitor rows past {R['ext_from']} off the {R['mon']}-step cadence")
        missing = [s for s in range(R["ext_from"] + R["mon"], R["ext_to"] + 1, R["mon"]) if s not in {t for t, _ in ema}]
        if missing: probs.append(f"{len(missing)} cadence rows missing (first {missing[0]})")
    rs = (root.pdir(ARM) / "resumes.txt"); res = [int(x) for x in rs.read_text().split()] if rs.exists() else []
    if not res or min(res) < R["ext_from"]: probs.append(f"resumes {res[:4]}: the first resume is not at or after {R['ext_from']}")
    z16, r16 = root.recs(root.chain(ARM, "full_vsel_t16")), ref.recs(ref.chain(ARM, "full_vsel_t16"))
    if r16 is None: probs.append("the 50k D16 reference row is absent")
    elif z16 is not None and not np.array_equal(np.sort(z16["idx"]), np.sort(r16["idx"])): probs.append("the long and the 50k D16 rows cover different puzzles")
    L["INTEGRITY"] = "PASS" if not probs else "FAIL: " + "; ".join(probs)
    for k in ("R-SP-1 PARAMS",): L[k] = base[k]
    # R-XL-1
    if ema:
        let, g, top, prev = plateau(ema); at = next(s for s, v in ema if v == top)
        L["R-XL-1 PLATEAU"] = let if g is None else f"{let} (the EMA monitor's running maximum {100*top:.2f} first at {at}; {100*prev:.2f} by {ema[-1][0] - R['plateau_window']}; gain {100*g:+.2f} pp over the last {R['plateau_window']:,} steps, the line {100*R['plateau_pp']:.0f} pp)"
    else: L["R-XL-1 PLATEAU"] = "NO-DATA"
    L["R-XL-2 PARITY-16"], L["R-XL-2 PARITY-64"] = base["R-SP-2 PARITY-16"], base["R-SP-2 PARITY-64"]
    # R-XL-3
    pr = PF.paired(z16, r16) if z16 is not None and r16 is not None else None
    if pr is None: L["R-XL-3 GAIN-16"] = "NO-DATA"
    else:
        sig = pr["p"] < R["mcnemar_p"] and abs(pr["diff"]) >= R["gain_floor"]
        L["R-XL-3 GAIN-16"] = ("GAINED" if sig and pr["diff"] > 0 else "LOST" if sig else "FLAT") + f" (d {100*pr['diff']:+.2f} pp on {pr['n']:,}; only-long {pr['only_a']:,}, only-50k {pr['only_b']:,}; p {pr['p']:.1e}; the floor {100*R['gain_floor']:.0f} pp)"
    L["R-XL-4 SELECTOR"], L["R-XL-5 ONSET"] = base["R-SP-3 SELECTOR"], base["R-SP-5 ONSET"]
    # R-XL-6
    zx = root.recs(root.scan128(ARM)); out, ahead = [], 0
    if zx is not None:
        sx, vx = PF.selected_exact(zx); X = dict(idx=zx["idx"], sel=sx)
        for a in SP.TRIPLE:
            z = root.recs(root.scan128(a))
            if z is None: out.append(f"{a} NO-DATA"); continue
            s, _ = PF.selected_exact(z); q = PF.paired(dict(idx=z["idx"], sel=s), X, key="sel")
            let = "PARITY" if q is None or q["p"] >= R["mcnemar_p"] else ("AHEAD" if q["diff"] > 0 else "BEHIND"); ahead += let == "AHEAD"
            out.append(f"{a} {let} ({100*s.mean():.2f} vs X {100*sx.mean():.2f}; only-DEC {q['only_a']}, only-X {q['only_b']}; p {q['p']:.1e})" if q else f"{a} {let}")
        out.append("TRIPLE-AHEAD" if ahead == 3 else f"TRIPLE-AHEAD {ahead}/3")
        L["X k128 (descriptive)"] = f"selected {100*sx.mean():.2f}, verified {100*vx.mean():.2f}"
    L["R-XL-6 DEC-VS-X"] = " | ".join(out) if out else "NO-DATA"
    # COMPUTE (descriptive)
    tr = [r for r in PF.metrics(root, ARM) if "loss" in r and r.get("step", 0) > R["ext_from"] and r.get("steps_per_sec")]
    if tr and ema:
        pace = float(np.median([r["steps_per_sec"] for r in tr])); frac = (ema[-1][0] / pace) / (R["dec_steps"] / R["dec_pace"])
        L["COMPUTE (descriptive)"] = f"median pace {pace:.1f} steps/s past {R['ext_from']:,}; {ema[-1][0]:,} steps = x{frac:.2f} of one DEC seed's training compute ({R['dec_steps']:,} steps at {R['dec_pace']} steps/s, the same pod type)"
    s16 = root.summ(root.chain(ARM, "full_vsel_t16")) or {}
    L["SELECTED GRID (descriptive)"] = str(s16.get("ckpt"))
    return L


def report(root: Path, ref: Path, out: Path | None):
    L = letters(PF.Root(root), PF.Root(ref))
    lines = ["THE X5 LONG RUN — REGISTERED VERDICT (tools/analyze_x5long.py; the registry at the top of this file; R-SP rules from tools/analyze_sudokupend.py)"]
    lines += [f"  {k:28s} {v}" for k, v in L.items()]
    text = "\n".join(lines); print(text)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True); out.with_suffix(".txt").write_text(text + "\n"); json.dump(L, open(out.with_suffix(".json"), "w"), indent=1)
    return L


# ---------- selftest: hand-built records on analyze_sudokupend's fixture ----------
def _long_root(tmp: Path, curve, x16=0.70, x64=0.72, x_sel=0.80, resumes=(50000,), d16=None, argv=None, spur_x=0.004):
    """a long-run root: SP's fixture (the triple, EqR, X5's rows) + the long monitor curve, resumes.txt and per-puzzle D16 records."""
    r = SP._root(tmp, x16, x64, (0.954, 0.9595, 0.9487), (0.9905, 0.9923, 0.9887), x_sel=x_sel, spur_x=spur_x, argv_dict=argv or {"cell": "trm", "sudoku_digit_aug": True, "fpa_k": 1, "trm_ri_sigma": 1.0})
    p = r / "pretrainchamp_X5"
    with open(p / "metrics.jsonl", "w") as f:
        for s, v in curve:
            f.write(json.dumps({"monitor": {"step": s, "val_t16": v, "val_t16_ema": v}}) + "\n")
            f.write(json.dumps({"step": s, "loss": 0.7, "steps_per_sec": 284.0}) + "\n")
    (p / "resumes.txt").write_text("".join(f"{x}\n" for x in resumes))
    if d16 is not None:
        np.savez(r / "sxeval_pchampX5" / "full_vsel_t16" / "records_all.npz", idx=np.arange(len(d16)), cold_exact=np.asarray(d16, bool))
    return r

def _curve(tail):
    head = [(s, min(0.47, 0.01 * s / 1000)) for s in range(2000, 50001, 2000)]
    return head + [(s, tail(s)) for s in range(60000, R["ext_to"] + 1, 10000)]


def selftest():
    ok, bad = 0, []
    def chk(n, c):
        nonlocal ok
        if c: ok += 1
        else: bad.append(n)
    rng = np.random.default_rng(1); n = 4000
    ref_bits = rng.random(n) < 0.45
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        head = [(s, v) for s, v in _curve(lambda s: 0.0) if s <= 50000]
        ref = _long_root(td / "ref", head, x16=0.45, x64=0.47, d16=ref_bits, resumes=())
        # A: rises to 0.70 by 400k, flat after -> PLATEAUED; GAINED; BELOW; the DEC ahead; continuation intact
        long_bits = ref_bits | (rng.random(n) < 0.45)
        A = _long_root(td / "a", _curve(lambda s: min(0.70, 0.47 + 0.23 * (s - 50000) / 350000)), d16=long_bits)
        L = letters(PF.Root(A), PF.Root(ref))
        chk("A integrity", L["INTEGRITY"] == "PASS")
        chk("A plateaued", L["R-XL-1 PLATEAU"].startswith("PLATEAUED") and "gain +0.00 pp" in L["R-XL-1 PLATEAU"] and "first at 400000" in L["R-XL-1 PLATEAU"])
        chk("A parity BELOW (SP's rule, unchanged)", L["R-XL-2 PARITY-16"].startswith("BELOW") and L["R-XL-2 PARITY-64"].startswith("BELOW"))
        chk("A gained", L["R-XL-3 GAIN-16"].startswith("GAINED") and "only-50k 0;" in L["R-XL-3 GAIN-16"])
        chk("A dec ahead", L["R-XL-6 DEC-VS-X"].endswith("TRIPLE-AHEAD") and L["R-XL-6 DEC-VS-X"].count("AHEAD (") == 3)
        chk("A compute", "x0.96 of one DEC seed" in L["COMPUTE (descriptive)"])        # (960000 / 284) / (50000 / 14.2) = 0.96
        # B: still rising at the cap (a 5 pp gain over the last 300k); a gain under the 1 pp floor is FLAT however small its p
        tiny = ref_bits.copy(); flip = np.flatnonzero(~ref_bits)[:30]; tiny[flip] = True          # +0.75 pp, 30:0 discordant (p ~ 1e-9)
        B = _long_root(td / "b", _curve(lambda s: 0.47 + 0.15 * (s - 50000) / 910000), d16=tiny, x_sel=0.9985, spur_x=0.0)   # no spurious draws: the selector keeps X at the triple's level
        L = letters(PF.Root(B), PF.Root(ref))
        chk("B still rising", L["R-XL-1 PLATEAU"].startswith("STILL-RISING") and "gain +4.95 pp" in L["R-XL-1 PLATEAU"])
        chk("B flat under the floor", L["R-XL-3 GAIN-16"].startswith("FLAT") and "only-long 30" in L["R-XL-3 GAIN-16"])
        chk("B dec parity at k128", "PARITY" in L["R-XL-6 DEC-VS-X"] and "TRIPLE-AHEAD" in L["R-XL-6 DEC-VS-X"] and not L["R-XL-6 DEC-VS-X"].endswith("| TRIPLE-AHEAD"))
        # C: integrity failures — a different head (not X5's continuation), the budget short, an off-cadence row, a resume before 50k, other puzzles
        c = _curve(lambda s: 0.6); c[3] = (c[3][0], 0.99); c = [t for t in c if t[0] <= 900000] + [(905000, 0.6)]
        C = _long_root(td / "c", c, d16=np.r_[ref_bits, True], resumes=(30000, 50000))
        I = letters(PF.Root(C), PF.Root(ref))["INTEGRITY"]
        chk("C not the continuation", "not X5's continuation: 1 of 25" in I)
        chk("C budget short", "budget not reached: last monitor row 905000" in I)
        chk("C off cadence + missing", "1 monitor rows past 50000 off the 10000-step cadence" in I and "cadence rows missing (first 910000)" in I)
        chk("C early resume", "resumes [30000, 50000]" in I)
        chk("C other puzzles", "cover different puzzles" in I)
        # D: a decline after a peak -> ONSET fires (SP's rule) and the plateau letter still reads the running maximum
        D = _long_root(td / "d", _curve(lambda s: 0.75 if s <= 300000 else 0.60), d16=long_bits)
        L = letters(PF.Root(D), PF.Root(ref))
        chk("D onset + plateau", L["R-XL-5 ONSET"] == "MEMORIZES-BY 310000" and L["R-XL-1 PLATEAU"].startswith("PLATEAUED"))
        # E: SP's integrity failure passes through (a wrong lever)
        E = _long_root(td / "e", _curve(lambda s: 0.6), d16=long_bits, argv={"cell": "dec", "sudoku_digit_aug": True, "fpa_k": 1, "trm_ri_sigma": 1.0})
        chk("E lever failure passes through", "argv lacks ['--cell', 'trm']" in letters(PF.Root(E), PF.Root(ref))["INTEGRITY"])
        chk("plateau no window", plateau([(10000, 0.1), (20000, 0.2)])[0] == "NO-WINDOW")
    print(f"selftest {'OK' if not bad else 'FAILED'}: {ok}/{ok + len(bad)} checks" + (f"; failed: {bad}" if bad else ""))
    return 0 if not bad else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="runs"); ap.add_argument("--ref-root", default=None); ap.add_argument("--out", default=None); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest: sys.exit(selftest())
    if not a.ref_root: sys.exit("--ref-root (the 50k run's staging root) is required")
    report(Path(a.root), Path(a.ref_root), Path(a.out) if a.out else None)
