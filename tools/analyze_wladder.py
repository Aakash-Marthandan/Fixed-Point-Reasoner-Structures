#!/usr/bin/env python3
# Ledger: THE WIDTH LADDER analyzer (2026-09-18; Documentation/Plan_2026-09-18_Width_Ladder.md §3 = the rules, FROZEN at the registration commit).
# A MEASUREMENT for paper 1, one seed per new arm: the rules below label what one seed can resolve and say UNRESOLVED otherwise.
#   Arms (tools/chain_wladder.sh, the reduced battery on the identical 5,000 test puzzles, a fixed 30,000-step budget, seed 0):
#     W128 W256          the symmetric DEC's hidden size, one variable
#     SA128 SA192 SA256  OUR reimplementation of SE-RRM's mixers on this loop and recipe (attention over the cells with 2D rotary positions,
#                        attention across the nine fields)
#   References on the same ids when present under the root: W192R (the width-192 seed-0 grid selected inside 30,000 steps, read on the Mac),
#   C0 / C2 (width 384 at the 30k budget; their full-set and 100k records are intersected with the 5,000 ids).
#   FLOORS (twice the largest seeded spread of this recipe): 2.58 pp at 16 iterations (width 384's 1.29), 2.44 pp at 64 (width 384's 1.22).
#   INTEGRITY   per arm: the config's cell / width / mixers / seed / fixed 30,000 budget, no extension marker, the three rows on ONE grid = the
#               512-puzzle selection, n = 5,000 at 16 and 64 iterations with EMA, identical puzzle ids across every arm.
#   R-WL-1      each new width against W192R, paired on identical puzzles at 16 and 64: INSIDE / BEYOND the floor, with the exact McNemar p.
#   R-WL-2      the shape of accuracy at 64 over the hidden sizes present (128, 192, 256, 384): INTERIOR-192 / NARROWER-BETTER / WIDER-BETTER /
#               MIXED, and RESOLVED only when the range exceeds the floor (else UNRESOLVED-AT-ONE-SEED).
#   R-WL-3      the port's fidelity: SA256 against SE-RRM's published 93.73 / 98.22 (arXiv 2603.02193, Table 2): FAITHFUL inside the floors at
#               both depths, else ABOVE / BELOW per depth.
#   R-WL-4      does narrowing transfer to the attention-mixed model: SA192 - SA256 and SA128 - SA256 at 64, paired: the sign, INSIDE / BEYOND.
#   R-WL-5      the mixer at a matched hidden size: SA128 - W128, SA256 - W256, SA192 - W192R at 16 and 64, paired.
#   STABILITY   (descriptive) the fixed start against one random start from the scan (b1 - cold at 64), the selected step, the validation
#               maximum and its drop to the end of the budget.
"""  .venv/bin/python tools/analyze_wladder.py --selftest
  .venv/bin/python tools/analyze_wladder.py --root runs/_wladder_pull/stage/runs [--out runs/_wladder_pull/analysis]"""
from __future__ import annotations
import argparse, json, sys, tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_paperfinal as PF   # Root, jload, exact_mcnemar, paired, metrics

FLOOR16, FLOOR64 = 0.0258, 0.0244
SERRM16, SERRM64 = 0.9373, 0.9822
NEW = {"W128": (128, "mlp", "mean"), "W256": (256, "mlp", "mean"), "SA128": (128, "attn", "attn"), "SA192": (192, "attn", "attn"), "SA256": (256, "attn", "attn")}
WIDTH_REF = {"W192R": 192, "C0": 384}
BUDGET = 30000

def row(root, arm, t):
    """(summary, records) of the arm's row at t iterations: the ladder's 5k row, else (the references) the champion rows."""
    for name in ((f"sub5k_vsel_t{t}",) if arm in NEW or arm == "W192R" else (f"sub5k_vsel_t{t}", "full_vsel_t16" if t == 16 else "full_vsel_t64")):
        d = root.chain(arm, name); s = root.summ(d)
        if s: return s, root.recs(d)
    return None, None

def acc_on(recs, ids):
    """Exact accuracy restricted to ids (the 5,000): (accuracy, n found)."""
    if recs is None: return None, 0
    _, pa, _ = np.intersect1d(np.asarray(recs["idx"]), ids, return_indices=True)
    return (float(np.asarray(recs["cold_exact"]).astype(bool)[pa].mean()), int(len(pa))) if len(pa) else (None, 0)

def restrict(recs, ids):
    if recs is None: return None
    _, pa, _ = np.intersect1d(np.asarray(recs["idx"]), ids, return_indices=True)
    return dict(idx=np.asarray(recs["idx"])[pa], cold_exact=np.asarray(recs["cold_exact"])[pa])

def floor_label(d, floor): return "n/a" if d is None else ("BEYOND" if abs(d) > floor else "INSIDE")

def shape(accs):
    """accs {hidden size: accuracy at 64}: the registered shape letter and whether one seed resolves it."""
    ws = sorted(w for w, a in accs.items() if a is not None)
    if len(ws) < 3: return "NO-DATA", "n/a"
    v = [accs[w] for w in ws]; rng = max(v) - min(v); res = "RESOLVED" if rng > FLOOR64 else "UNRESOLVED-AT-ONE-SEED"
    if all(v[i] > v[i + 1] for i in range(len(v) - 1)): return "NARROWER-BETTER", res
    if all(v[i] < v[i + 1] for i in range(len(v) - 1)): return "WIDER-BETTER", res
    if 192 in accs and accs[192] is not None and accs[192] == max(v): return "INTERIOR-192", res
    return "MIXED", res

def integrity(root, arms):
    bad = []; ids0 = None
    for a in arms:
        w, mix, cpl = NEW[a]; cfg = PF.jload(root.pdir(a) / "config.json") or {}; av = cfg.get("argv") if isinstance(cfg.get("argv"), dict) else {}
        c = cfg.get("config", cfg)                                    # the trainer's config.json: {"argv": vars(a), "config": {...}, ...}
        got = (c.get("cell_kind"), c.get("dec_width"), (c.get("dec_token_mixer") or "mlp"), (c.get("dec_coupling_kind") or "mean"))
        if got != ("dec", w, mix, cpl): bad.append(f"{a} config {got} != ('dec', {w}, '{mix}', '{cpl}')")
        if av and (av.get("seed") != 0 or av.get("steps") != BUDGET): bad.append(f"{a} argv seed/steps {av.get('seed')}/{av.get('steps')}")
        if (root.pdir(a) / "EXTENDED.txt").exists(): bad.append(f"{a} carries an extension marker (the budget is fixed)")
        m = [r for r in PF.metrics(root, a) if "loss" in r]
        if m and int(m[-1].get("step", -1)) not in (BUDGET, BUDGET - 1): bad.append(f"{a} last training step {m[-1].get('step')} != {BUDGET}")
        vb = (root.pdir(a) / "val_best.txt").read_text().split()[0] if (root.pdir(a) / "val_best.txt").exists() else None
        grids = set()
        for t in (16, 64):
            s, r = row(root, a, t)
            if not s: bad.append(f"{a} row t{t} absent"); continue
            if s.get("n") != 5000 or s.get("t_total") != t or not s.get("ema") or s.get("subsample") != 5000: bad.append(f"{a} t{t}: n/t/ema/subsample {s.get('n')}/{s.get('t_total')}/{s.get('ema')}/{s.get('subsample')}")
            grids.add(str(s.get("ckpt")))
            if r is not None:
                ids = np.sort(np.asarray(r["idx"]))
                if ids0 is None: ids0 = ids
                elif len(ids) != len(ids0) or (ids != ids0).any(): bad.append(f"{a} t{t}: puzzle ids differ from the first arm's")
        sc = root.summ(root.scan32(a))
        if sc: grids.add(str(sc.get("ckpt")))
        if len(grids) > 1: bad.append(f"{a} rows on {len(grids)} grids {sorted(grids)}")
        if vb and grids and vb not in next(iter(grids)): bad.append(f"{a} rows not on the selected grid {vb}")
    return bad, ids0

def analyze(root_path):
    root = PF.Root(root_path); J = {}; arms = [a for a in NEW if root.summ(root.chain(a, "sub5k_vsel_t64")) or root.summ(root.chain(a, "sub5k_vsel_t16"))]
    PF.say(f"THE WIDTH LADDER — analyzer (frozen rules; one seed per new arm; floors {100*FLOOR16:.2f} pp at 16 and {100*FLOOR64:.2f} pp at 64)")
    PF.say(f"arms present: {arms or 'none'}")
    if not arms: PF.say("NO-DATA"); return {"INTEGRITY": "NO-DATA"}
    bad, ids = integrity(root, arms); J["INTEGRITY"] = "PASS" if not bad else "FAIL: " + "; ".join(bad); PF.say(f"INTEGRITY                {J['INTEGRITY']}")
    A = {}
    for a in arms + [r for r in WIDTH_REF if row(root, r, 64)[0] or row(root, r, 16)[0]]:
        A[a] = {}
        for t in (16, 64):
            s, r = row(root, a, t); A[a][t] = (restrict(r, ids) if (r is not None and ids is not None) else r); acc, n = acc_on(r, ids) if (r is not None and ids is not None) else ((s or {}).get("exact_acc"), (s or {}).get("n", 0))
            A[a][f"acc{t}"], A[a][f"n{t}"] = acc, n
        PF.say(f"  {a:6s} 16: {PF.pp(A[a]['acc16'])} (n {A[a]['n16']})   64: {PF.pp(A[a]['acc64'])} (n {A[a]['n64']})")
    def pr(a, b, t):
        if a not in A or b not in A: return None
        return PF.paired(A[a][t], A[b][t])
    for a in ("W128", "W256"):
        for t, fl in ((16, FLOOR16), (64, FLOOR64)):
            q = pr(a, "W192R", t); J[f"R-WL-1 {a} vs W192R @{t}"] = "NO-DATA" if not q else f"{floor_label(q['diff'], fl)} ({100*q['diff']:+.2f} pp, only-{a} {q['only_a']}, only-W192R {q['only_b']}, n {q['n']}, p {q['p']:.2g})"
            PF.say(f"R-WL-1 {a} vs W192R @{t:<3d} {J[f'R-WL-1 {a} vs W192R @{t}']}")
    accs = {NEW[a][0]: A[a]["acc64"] for a in ("W128", "W256") if a in A}; accs.update({w: A[r]["acc64"] for r, w in WIDTH_REF.items() if r in A})
    sh, res = shape(accs); J["R-WL-2 SHAPE@64"] = f"{sh} / {res} ({', '.join(f'{w}: {PF.pp(accs[w])}' for w in sorted(accs))})"; PF.say(f"R-WL-2 SHAPE@64          {J['R-WL-2 SHAPE@64']}")
    if "SA256" in A and A["SA256"]["acc16"] is not None and A["SA256"]["acc64"] is not None:
        d16, d64 = A["SA256"]["acc16"] - SERRM16, A["SA256"]["acc64"] - SERRM64
        lab = "FAITHFUL" if abs(d16) <= FLOOR16 and abs(d64) <= FLOOR64 else " / ".join(f"{'ABOVE' if d > 0 else 'BELOW'}@{t}" for d, t, fl in ((d16, 16, FLOOR16), (d64, 64, FLOOR64)) if abs(d) > fl)
        J["R-WL-3 FIDELITY"] = f"{lab} (SA256 {PF.pp(A['SA256']['acc16'])} / {PF.pp(A['SA256']['acc64'])} vs SE-RRM's published 93.73 / 98.22; {100*d16:+.2f} / {100*d64:+.2f} pp; different evaluation sets)"
    else: J["R-WL-3 FIDELITY"] = "NO-DATA"
    PF.say(f"R-WL-3 FIDELITY          {J['R-WL-3 FIDELITY']}")
    for a in ("SA192", "SA128"):
        q = pr(a, "SA256", 64); J[f"R-WL-4 {a} vs SA256 @64"] = "NO-DATA" if not q else f"{'NARROWER-AHEAD' if q['diff'] > 0 else 'NARROWER-BEHIND'} / {floor_label(q['diff'], FLOOR64)} ({100*q['diff']:+.2f} pp, p {q['p']:.2g}, n {q['n']})"
        PF.say(f"R-WL-4 {a} vs SA256 @64  {J[f'R-WL-4 {a} vs SA256 @64']}")
    for a, b in (("SA128", "W128"), ("SA256", "W256"), ("SA192", "W192R")):
        for t, fl in ((16, FLOOR16), (64, FLOOR64)):
            q = pr(a, b, t); J[f"R-WL-5 {a} vs {b} @{t}"] = "NO-DATA" if not q else f"{floor_label(q['diff'], fl)} ({100*q['diff']:+.2f} pp, p {q['p']:.2g}, n {q['n']})"
            PF.say(f"R-WL-5 {a} vs {b} @{t:<3d} {J[f'R-WL-5 {a} vs {b} @{t}']}")
    for a in arms:
        sc = root.summ(root.scan32(a)) or {}; m = [r["monitor"] for r in PF.metrics(root, a) if isinstance(r.get("monitor"), dict) and "val_t16_ema" in r["monitor"]]; vm = max((r["val_t16_ema"] for r in m), default=None)
        vb = (root.pdir(a) / "val_best.txt").read_text().split() if (root.pdir(a) / "val_best.txt").exists() else ["-"]
        gap = (sc.get("b1_exact") - sc.get("exact_acc")) if (sc.get("b1_exact") is not None and sc.get("exact_acc") is not None) else None
        J[f"STABILITY {a}"] = f"selected {vb[0]}; validation max {PF.pp(vm)}, end {PF.pp(m[-1]['val_t16_ema'] if m else None)}; scan cold {PF.pp(sc.get('exact_acc'))}, one random start {PF.pp(sc.get('b1_exact'))} (gap {'-' if gap is None else f'{100*gap:+.2f}'} pp), verified {PF.pp(sc.get('exact_acc_vote'))}"
        PF.say(f"STABILITY {a:6s}         {J[f'STABILITY {a}']}")
    return J

# ---------------- selftest (hand-built records; the printed labels asserted) ----------------
def _mk(tmp, accs, cfg_over=None, ext=None, grid_split=None, ids_shift=None):
    rng = np.random.default_rng(0); ids = np.arange(5000) * 3
    for a, (a16, a64) in accs.items():
        for t, acc in ((16, a16), (64, a64)):
            if a in ("C0",):
                d = tmp / f"sxeval_pchamp{a}" / ("full_vsel_t16" if t == 16 else "full_vsel_t64"); PF._write_row(d, 0, acc, rng, idx=np.arange(20000), extra=dict(t_total=t, ema=True, n=20000, ckpt="runs/pretrainchamp_C0/ckpt_016000.pkl"))
                continue
            d = tmp / f"sxeval_pchamp{a}" / f"sub5k_vsel_t{t}"; ii = ids + (ids_shift if (ids_shift and a == ids_shift[0] and t == 64) else 0) if not isinstance(ids_shift, tuple) else (ids + ids_shift[1] if (a == ids_shift[0] and t == 64) else ids)
            ck = f"runs/pretrainchamp_{a}/ckpt_022000.pkl" if not (grid_split == a and t == 64) else f"runs/pretrainchamp_{a}/ckpt_030000.pkl"
            PF._write_row(d, 0, acc, rng, idx=ii, extra=dict(t_total=t, ema=True, subsample=5000, ckpt=ck))
        if a in NEW:
            w, mix, cpl = NEW[a]; p = tmp / f"pretrainchamp_{a}"; p.mkdir(parents=True, exist_ok=True); c = dict(cell_kind="dec", dec_width=w, dec_token_mixer=mix, dec_coupling_kind=cpl); c.update((cfg_over or {}).get(a, {}))
            (p / "config.json").write_text(json.dumps(dict(config=c, argv=dict(seed=0, steps=BUDGET)))); (p / "val_best.txt").write_text("022000 0.9600 22000")
            (p / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in [dict(monitor=dict(step=22000, val_t16_ema=0.96)), dict(step=BUDGET, loss=0.5), dict(monitor=dict(step=BUDGET, val_t16_ema=0.95))]))
            if ext == a: (p / "EXTENDED.txt").write_text("x")
            s = tmp / f"sxscan_pchamp{a}"; s.mkdir(parents=True, exist_ok=True); (s / "summary_all.json").write_text(json.dumps(dict(ckpt=f"runs/pretrainchamp_{a}/ckpt_022000.pkl", exact_acc=a64, b1_exact=a64 - 0.002, exact_acc_vote=0.998, n=5000)))

def selftest():
    n = 0
    assert shape({128: .99, 192: .985, 256: .98, 384: .95}) == ("NARROWER-BETTER", "RESOLVED") and shape({128: .980, 192: .990, 256: .985, 384: .982}) == ("INTERIOR-192", "UNRESOLVED-AT-ONE-SEED"); n += 1
    assert shape({128: .95, 192: .97, 256: .99})[0] == "WIDER-BETTER" and shape({128: .99, 256: .97})[0] == "NO-DATA" and shape({128: .99, 192: .97, 256: .98, 384: .96})[0] == "MIXED"; n += 1
    assert floor_label(0.03, FLOOR16) == "BEYOND" and floor_label(-0.02, FLOOR16) == "INSIDE" and floor_label(None, FLOOR16) == "n/a"; n += 1
    base = {"W128": (.955, .991), "W256": (.950, .984), "SA128": (.93, .975), "SA192": (.94, .985), "SA256": (.935, .981), "W192R": (.958, .990), "C0": (.951, .984)}
    with tempfile.TemporaryDirectory() as t_:
        t = Path(t_); _mk(t, base); PF.LINES.clear(); J = analyze(t)
        assert J["INTEGRITY"] == "PASS", J["INTEGRITY"]; n += 1
        assert J["R-WL-1 W128 vs W192R @64"].startswith("INSIDE (+0.10 pp") and J["R-WL-1 W256 vs W192R @16"].startswith("INSIDE (-0.80 pp"), (J["R-WL-1 W128 vs W192R @64"], J["R-WL-1 W256 vs W192R @16"]); n += 1
        assert J["R-WL-2 SHAPE@64"].startswith("NARROWER-BETTER / UNRESOLVED-AT-ONE-SEED") or J["R-WL-2 SHAPE@64"].startswith("MIXED"), J["R-WL-2 SHAPE@64"]; n += 1
        assert J["R-WL-3 FIDELITY"].startswith("FAITHFUL (SA256 93.50 / 98.10"), J["R-WL-3 FIDELITY"]; n += 1
        assert J["R-WL-4 SA192 vs SA256 @64"].startswith("NARROWER-AHEAD / INSIDE (+0.40 pp") and J["R-WL-4 SA128 vs SA256 @64"].startswith("NARROWER-BEHIND / INSIDE (-0.60 pp"); n += 1
        assert J["R-WL-5 SA128 vs W128 @16"].startswith("INSIDE (-2.50 pp") and J["R-WL-5 SA192 vs W192R @64"].startswith("INSIDE (-0.50 pp"); n += 1
        assert "selected 022000" in J["STABILITY W128"] and "gap -0.20 pp" in J["STABILITY W128"]; n += 1
    with tempfile.TemporaryDirectory() as t_:   # a collapse at 128 is BEYOND and the fidelity miss is labeled per depth
        t = Path(t_); _mk(t, {**base, "W128": (.90, .95), "SA256": (.90, .985)}); PF.LINES.clear(); J = analyze(t)
        assert J["R-WL-1 W128 vs W192R @64"].startswith("BEYOND (-4.00 pp") and J["R-WL-1 W128 vs W192R @16"].startswith("BEYOND (-5.80 pp") and J["R-WL-3 FIDELITY"].startswith("BELOW@16 (SA256 90.00"), (J["R-WL-1 W128 vs W192R @64"], J["R-WL-3 FIDELITY"]); n += 1
    for kw, frag in ((dict(cfg_over={"SA128": dict(dec_token_mixer="mlp")}), "SA128 config"), (dict(ext="W256"), "W256 carries an extension marker"), (dict(grid_split="W128"), "W128 rows on 2 grids"), (dict(ids_shift=("SA192", 1)), "SA192 t64: puzzle ids differ")):
        with tempfile.TemporaryDirectory() as t_:
            t = Path(t_); _mk(t, base, **kw); PF.LINES.clear(); J = analyze(t); assert J["INTEGRITY"].startswith("FAIL") and frag in J["INTEGRITY"], (frag, J["INTEGRITY"]); n += 1
    with tempfile.TemporaryDirectory() as t_:
        PF.LINES.clear(); assert analyze(Path(t_)) == {"INTEGRITY": "NO-DATA"}; n += 1
    print(f"selftest OK: {n}/{n}")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true"); ap.add_argument("--root"); ap.add_argument("--out"); a = ap.parse_args()
    if a.selftest: return selftest()
    J = analyze(a.root)
    if a.out:
        o = Path(a.out); o.mkdir(parents=True, exist_ok=True); (o / "wladder_verdict.txt").write_text("\n".join(PF.LINES) + "\n"); (o / "wladder_verdict.json").write_text(json.dumps(J, indent=1))

if __name__ == "__main__":
    main()
