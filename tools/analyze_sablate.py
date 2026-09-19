#!/usr/bin/env python3
"""THE RECIPE ABLATION — analyzer (Plan_2026-09-19_Recipe_Ablation.md; FROZEN at the registration commit, before any row exists).

The question (the PI, 2026-09-19): "What are we doing that's different from everyone else that enables us to get such high scores?  Change the
three things to locate the advantage we have."  The ladder's SA256 arm — SE-RRM's two mixers inside OUR recipe — reads 98.24 / 99.60 where
SE-RRM's paper reports 93.73 / 98.22, so the advantage is not the mixer.  Each arm here is SA256 with ONE recipe item moved to SE-RRM's side:
  SA256L  the LOOP        EqR's damping and path noise off             (--trm-lambda 0 --trm-beta 0)
  SA256S  the START-UP    randomized initialization and anchor rows off (--trm-ri-sigma 0 --fpa-k 0)
  SA256O  the OPTIMIZER   SE-RRM's batch and learning rate              (--batch 272 --lr 5e-4 --lr-end 5e-4)
Same seed (0), same fixed 30,000 steps, same selection rule, the same 5,000 test puzzles at 16 and 64 iterations.  The reference is the ladder's
banked SA256 row; it is READ, never re-run.  One seed per arm, so every difference is read against the seeded floors.

RULES (frozen):
  INTEGRITY   every arm's trainer argv and model config differ from SA256's in EXACTLY the registered keys, at the registered values (`out` and
              `resume` ignored; `remat` reported, math-neutral); seed 0; 30,000 steps, no extension marker; rows n 5,000 / EMA / the right depth,
              on the identical puzzle ids as SA256's rows; all of an arm's rows and its scan on one grid = its selected grid.
  R-AB-1      <arm> vs SA256 at 16 and at 64, paired on the identical 5,000 (exact McNemar): INSIDE the floor (2.58 pp at 16, 2.44 pp at 64),
              or BELOW-BEYOND / ABOVE-BEYOND.  The floor is absolute; a p-value alone never moves a label.
  R-AB-2      <arm> vs SE-RRM's published 93.73 / 98.22: WITHIN the floor of it, ABOVE, or BELOW (different evaluation sets; their one run).
  R-AB-3      LOCATED at 16 / at 64 = the arms that read BELOW-BEYOND; NOT-LOCATED when none does.  The share of the SA256-to-SE-RRM gap an arm
              accounts for is printed ONLY for an arm that is BELOW-BEYOND (a relative number needs the absolute floor first).
  R-AB-4      SELECTOR per arm from its k32 scan: CLEAN when the spurious rate <= 1 %, else DIRTY (the reader's definition, analyze_paperfinal).
  STABILITY   descriptive: the selected grid (EDGE when it is the budget's last grid: the arm is then a lower bound), the validation maximum and
              end, the scan's fixed start against one random start, the verified accuracy; NaN / abort markers.

  .venv/bin/python tools/analyze_sablate.py --root <stage>/runs --out <dir>      (the stage holds the three arms AND the ladder's SA256 dirs)
  .venv/bin/python tools/analyze_sablate.py --selftest
"""
from __future__ import annotations
import argparse, json, sys, tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_paperfinal as PF   # Root, jload, paired, spurious, metrics, say, pp
import analyze_wladder as WL      # row, acc_on, restrict (the ladder's readers; the reference row is the ladder's)

FLOOR16, FLOOR64 = 0.0258, 0.0244
SERRM16, SERRM64 = 0.9373, 0.9822
REF, BUDGET, SPUR_CLEAN = "SA256", 30000, 0.01
IGNORE, NEUTRAL = {"out", "resume"}, {"remat"}
ARMS = {   # arm -> (what it moves, the registered argv differences from SA256, the registered model-config differences from SA256)
    "SA256L": ("the LOOP: damping and path noise off", dict(trm_lambda=0.0, trm_beta=0.0), dict(trm_lambda=0.0, trm_beta=0.0)),
    "SA256S": ("the START-UP levers: randomized initialization and anchor rows off", dict(trm_ri_sigma=0.0, fpa_k=0), dict(trm_ri_sigma=0.0, fpa_k=0)),
    "SA256O": ("the OPTIMIZER: batch 272, learning rate 5e-4", dict(batch=272, lr=5e-4, lr_end=5e-4), {}),
}

def same(x, y):
    if isinstance(x, bool) or isinstance(y, bool) or x is None or y is None or isinstance(x, str) or isinstance(y, str): return x == y
    try: return abs(float(x) - float(y)) <= 1e-12 * max(1.0, abs(float(y)))
    except (TypeError, ValueError): return x == y

def dict_diff(a, b, skip=()):
    """keys whose values differ between two flat dicts (a key absent on one side counts), minus skip."""
    return sorted(k for k in set(a) | set(b) if k not in skip and not same(a.get(k, "<absent>"), b.get(k, "<absent>")))

def cfg_of(root, arm):
    c = PF.jload(root.pdir(arm) / "config.json") or {}
    return (c.get("argv") if isinstance(c.get("argv"), dict) else {}), (c.get("config") if isinstance(c.get("config"), dict) else {})

def label1(d, floor): return "n/a" if d is None else ("INSIDE" if abs(d) <= floor else ("BELOW-BEYOND" if d < 0 else "ABOVE-BEYOND"))
def label2(acc, ref, floor): return "n/a" if acc is None else ("WITHIN" if abs(acc - ref) <= floor else ("ABOVE" if acc > ref else "BELOW"))

def integrity(root, arms):
    bad, notes = [], []; rav, rcf = cfg_of(root, REF); ids0 = None
    if not rav: bad.append(f"{REF} (the reference) config.json absent or without an argv dict")
    for t in (16, 64):
        s, r = WL.row(root, REF, t)
        if not s or r is None: bad.append(f"{REF} row t{t} absent"); continue
        if s.get("n") != 5000 or s.get("t_total") != t or not s.get("ema"): bad.append(f"{REF} t{t}: n/t/ema {s.get('n')}/{s.get('t_total')}/{s.get('ema')}")
        ids = np.sort(np.asarray(r["idx"]))
        if ids0 is None: ids0 = ids
        elif len(ids) != len(ids0) or (ids != ids0).any(): bad.append(f"{REF} t{t}: puzzle ids differ from its t16 row's")
    for a in arms:
        _, want_av, want_cf = ARMS[a]; av, cf = cfg_of(root, a)
        if not av: bad.append(f"{a} config.json absent or without an argv dict"); continue
        d_av = dict_diff(av, rav, skip=IGNORE); neutral = [k for k in d_av if k in NEUTRAL]; d_av = [k for k in d_av if k not in NEUTRAL]
        if neutral: notes.append(f"{a} differs in the math-neutral {neutral}")
        if d_av != sorted(want_av): bad.append(f"{a} argv differs from {REF}'s in {d_av}, registered {sorted(want_av)}")
        for k, v in want_av.items():
            if not same(av.get(k), v): bad.append(f"{a} argv {k} = {av.get(k)} != the registered {v}")
        d_cf = [k for k in dict_diff(cf, rcf) if k not in NEUTRAL]
        if d_cf != sorted(want_cf): bad.append(f"{a} model config differs from {REF}'s in {d_cf}, registered {sorted(want_cf)}")
        for k, v in want_cf.items():
            if not same(cf.get(k), v): bad.append(f"{a} config {k} = {cf.get(k)} != the registered {v}")
        if av.get("seed") != 0 or av.get("steps") != BUDGET: bad.append(f"{a} argv seed/steps {av.get('seed')}/{av.get('steps')}")
        if (root.pdir(a) / "EXTENDED.txt").exists(): bad.append(f"{a} carries an extension marker (the budget is fixed)")
        m = [r for r in PF.metrics(root, a) if "loss" in r]
        if m and int(m[-1].get("step", -1)) not in (BUDGET, BUDGET - 1): bad.append(f"{a} last training step {m[-1].get('step')} != {BUDGET}")
        vb = (root.pdir(a) / "val_best.txt").read_text().split()[0] if (root.pdir(a) / "val_best.txt").exists() else None
        grids = set()
        for t in (16, 64):
            s, r = WL.row(root, a, t)
            if not s: bad.append(f"{a} row t{t} absent"); continue
            if s.get("n") != 5000 or s.get("t_total") != t or not s.get("ema") or s.get("subsample") != 5000: bad.append(f"{a} t{t}: n/t/ema/subsample {s.get('n')}/{s.get('t_total')}/{s.get('ema')}/{s.get('subsample')}")
            grids.add(str(s.get("ckpt")))
            if r is not None and ids0 is not None:
                ids = np.sort(np.asarray(r["idx"]))
                if len(ids) != len(ids0) or (ids != ids0).any(): bad.append(f"{a} t{t}: puzzle ids differ from {REF}'s")
        sc = root.summ(root.scan32(a))
        if sc: grids.add(str(sc.get("ckpt")))
        if len(grids) > 1: bad.append(f"{a} rows on {len(grids)} grids {sorted(grids)}")
        if vb and grids and vb not in next(iter(grids)): bad.append(f"{a} rows not on the selected grid {vb}")
    return bad, notes, ids0

def analyze(root_path):
    root = PF.Root(root_path); J = {}; arms = [a for a in ARMS if root.summ(root.chain(a, "sub5k_vsel_t64")) or root.summ(root.chain(a, "sub5k_vsel_t16")) or (root.pdir(a) / "config.json").exists()]
    PF.say(f"THE RECIPE ABLATION — analyzer (frozen rules; one seed per arm; floors {100*FLOOR16:.2f} pp at 16 and {100*FLOOR64:.2f} pp at 64; the reference = the ladder's {REF})")
    PF.say(f"arms present: {arms or 'none'}")
    if not arms: PF.say("NO-DATA"); return {"INTEGRITY": "NO-DATA"}
    bad, notes, ids = integrity(root, arms); J["INTEGRITY"] = ("PASS" if not bad else "FAIL: " + "; ".join(bad)) + (f" [note: {'; '.join(notes)}]" if notes else ""); PF.say(f"INTEGRITY                {J['INTEGRITY']}")
    A = {}
    for a in [REF] + arms:
        A[a] = {}
        for t in (16, 64):
            s, r = WL.row(root, a, t); A[a][t] = WL.restrict(r, ids) if (r is not None and ids is not None) else r
            A[a][f"acc{t}"], A[a][f"n{t}"] = WL.acc_on(r, ids) if (r is not None and ids is not None) else ((s or {}).get("exact_acc"), (s or {}).get("n", 0))
        PF.say(f"  {a:6s} 16: {PF.pp(A[a]['acc16'])} (n {A[a]['n16']})   64: {PF.pp(A[a]['acc64'])} (n {A[a]['n64']})" + ("" if a == REF else f"   [{ARMS[a][0]}]"))
    located = {16: [], 64: []}
    for a in arms:
        for t, fl, se in ((16, FLOOR16, SERRM16), (64, FLOOR64, SERRM64)):
            q = PF.paired(A[a][t], A[REF][t]); k = f"R-AB-1 {a} vs {REF} @{t}"
            J[k] = "NO-DATA" if not q else f"{label1(q['diff'], fl)} ({100*q['diff']:+.2f} pp, only-{a} {q['only_a']}, only-{REF} {q['only_b']}, n {q['n']}, p {q['p']:.2g})"; PF.say(f"{k:30s} {J[k]}")
            if q and label1(q["diff"], fl) == "BELOW-BEYOND":
                gap = (A[REF][f"acc{t}"] or 0) - se; located[t].append(f"{a} ({100*q['diff']:+.2f} pp" + (f" = {100 * -q['diff'] / gap:.0f} % of the {100*gap:.2f} pp to SE-RRM" if gap > fl else "") + ")")
            acc = A[a][f"acc{t}"]; k2 = f"R-AB-2 {a} vs SE-RRM @{t}"
            J[k2] = "NO-DATA" if acc is None else f"{label2(acc, se, fl)} ({PF.pp(acc)} vs the published {100*se:.2f}; {100*(acc - se):+.2f} pp; different evaluation sets, their one run)"; PF.say(f"{k2:30s} {J[k2]}")
    for t in (16, 64):
        have = [a for a in arms if A[a][f"acc{t}"] is not None]; k = f"R-AB-3 LOCATED @{t}"
        J[k] = "NO-DATA" if not have else (("LOCATED: " + "; ".join(located[t])) if located[t] else f"NOT-LOCATED (no arm of {have} reads below {REF} beyond the floor; what remains: the items not ablated, an interaction, or their side's one run)")
        if have and len(have) < len(ARMS): J[k] += f" [only {len(have)} of {len(ARMS)} arms present]"
        PF.say(f"{k:30s} {J[k]}")
    for a in [REF] + arms:
        z = root.recs(root.scan32(a)); sp = PF.spurious(z) if z is not None else None; k = f"R-AB-4 SELECTOR {a}"
        J[k] = "NO-DATA" if sp is None else f"{'CLEAN' if sp <= SPUR_CLEAN else 'DIRTY'} (spurious k32 {100*sp:.2f} %)"; PF.say(f"{k:30s} {J[k]}")
    for a in arms:
        sc = root.summ(root.scan32(a)) or {}; M = PF.metrics(root, a); m = [r["monitor"] for r in M if isinstance(r.get("monitor"), dict) and "val_t16_ema" in r["monitor"]]; vm = max((r["val_t16_ema"] for r in m), default=None)
        vb = (root.pdir(a) / "val_best.txt").read_text().split() if (root.pdir(a) / "val_best.txt").exists() else ["-"]
        edge = vb[0].isdigit() and int(vb[0]) == BUDGET; nan = sum(1 for r in M if "loss" in r and not np.isfinite(r["loss"]))
        gap = (sc.get("b1_exact") - sc.get("exact_acc")) if (sc.get("b1_exact") is not None and sc.get("exact_acc") is not None) else None
        J[f"STABILITY {a}"] = (f"selected {vb[0]}{' EDGE (a lower bound: the budget ended on the maximum)' if edge else ''}; validation max {PF.pp(vm)}, end {PF.pp(m[-1]['val_t16_ema'] if m else None)}; "
                               f"scan fixed start {PF.pp(sc.get('exact_acc'))}, one random start {PF.pp(sc.get('b1_exact'))} (gap {'-' if gap is None else f'{100*gap:+.2f}'} pp), verified {PF.pp(sc.get('exact_acc_vote'))}; non-finite loss rows {nan}")
        PF.say(f"STABILITY {a:6s}         {J[f'STABILITY {a}']}")
    return J

# ---------------- selftest (hand-built records; the printed labels asserted) ----------------
BASE_AV = dict(seed=0, steps=BUDGET, batch=768, lr=1e-4, lr_end=1e-4, trm_lambda=0.05, trm_beta=0.01, trm_ri_sigma=1.0, fpa_k=1, fpa_eps=0.2, wd=1.0, remat=False, resume=None)
BASE_CF = dict(cell_kind="dec", dec_width=256, dec_token_mixer="attn", trm_lambda=0.05, trm_beta=0.01, trm_ri_sigma=1.0, fpa_k=1, remat=False)

def _mk(tmp, accs, av_over=None, cf_over=None, ext=None, grid_split=None, ids_shift=None, sel=None, spur=None, steps_end=None):
    rng = np.random.default_rng(0); ids = np.arange(5000) * 3
    for a, (a16, a64) in accs.items():
        vb = (sel or {}).get(a, "022000")
        for t, acc in ((16, a16), (64, a64)):
            d = tmp / f"sxeval_pchamp{a}" / f"sub5k_vsel_t{t}"; ii = ids + 1 if (ids_shift == a and t == 64) else ids
            ck = f"runs/pretrainchamp_{a}/ckpt_{vb}.pkl" if not (grid_split == a and t == 64) else f"runs/pretrainchamp_{a}/ckpt_010000.pkl"
            PF._write_row(d, 0, acc, rng, idx=ii, extra=dict(t_total=t, ema=True, subsample=5000, ckpt=ck))
        av, cf = dict(BASE_AV, out=f"runs/pretrainchamp_{a}"), dict(BASE_CF)
        if a in ARMS: av.update(ARMS[a][1]); cf.update(ARMS[a][2])
        av.update((av_over or {}).get(a, {})); cf.update((cf_over or {}).get(a, {}))
        p = tmp / f"pretrainchamp_{a}"; p.mkdir(parents=True, exist_ok=True); (p / "config.json").write_text(json.dumps(dict(config=cf, argv=av))); (p / "val_best.txt").write_text(f"{vb} 0.9600 {int(vb)}")
        (p / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in [dict(monitor=dict(step=22000, val_t16_ema=0.96)), dict(step=(steps_end or {}).get(a, BUDGET), loss=0.5), dict(monitor=dict(step=BUDGET, val_t16_ema=0.95))]))
        if ext == a: (p / "EXTENDED.txt").write_text("x")
        s = tmp / f"sxscan_pchamp{a}"; s.mkdir(parents=True, exist_ok=True); (s / "summary_all.json").write_text(json.dumps(dict(ckpt=f"runs/pretrainchamp_{a}/ckpt_{vb}.pkl", exact_acc=a64, b1_exact=a64 - 0.002, exact_acc_vote=0.998, n=5000)))
        n, k = 400, 8; ex = rng.random((n, k)) < 0.9; res = np.where(ex, 0.001, 0.5); f = (spur or {}).get(a, 0.0); res[(~ex) & (rng.random((n, k)) < f)] = 0.0005   # a fraction f of the WRONG draws look converged
        np.savez(s / "records_all.npz", idx=np.arange(n), mi_exact_k=ex, mi_resid_k=res)

def selftest():
    n = 0
    assert label1(-0.03, FLOOR16) == "BELOW-BEYOND" and label1(0.03, FLOOR16) == "ABOVE-BEYOND" and label1(-0.0258, FLOOR16) == "INSIDE" and label1(None, FLOOR16) == "n/a"; n += 1
    assert label2(0.95, SERRM16, FLOOR16) == "WITHIN" and label2(0.98, SERRM16, FLOOR16) == "ABOVE" and label2(0.90, SERRM16, FLOOR16) == "BELOW"; n += 1
    assert same(0, 0.0) and same(5e-4, 0.0005) and not same(1e-4, 5e-4) and same(None, None) and not same(None, 0) and not same(True, 1.5) and dict_diff(dict(a=1, b=2, out="x"), dict(a=1, b=3, c=0, out="y"), skip=IGNORE) == ["b", "c"]; n += 1
    base = {"SA256": (.982, .996), "SA256L": (.975, .994), "SA256S": (.980, .995), "SA256O": (.940, .984)}
    with tempfile.TemporaryDirectory() as t_:   # the optimizer arm falls beyond the floor at 16 and lands within SE-RRM's number; the others stay inside
        t = Path(t_); _mk(t, base, sel={"SA256O": "030000"}, spur={"SA256S": 0.6}); PF.LINES.clear(); J = analyze(t)
        assert J["INTEGRITY"] == "PASS", J["INTEGRITY"]; n += 1
        assert J["R-AB-1 SA256L vs SA256 @16"].startswith("INSIDE (-0.70 pp") and J["R-AB-1 SA256S vs SA256 @64"].startswith("INSIDE (-0.10 pp") and J["R-AB-1 SA256O vs SA256 @16"].startswith("BELOW-BEYOND (-4.20 pp"), J["R-AB-1 SA256O vs SA256 @16"]; n += 1
        assert J["R-AB-1 SA256O vs SA256 @64"].startswith("INSIDE (-1.20 pp"), J["R-AB-1 SA256O vs SA256 @64"]; n += 1
        assert J["R-AB-2 SA256O vs SE-RRM @16"].startswith("WITHIN (94.00 vs the published 93.73; +0.27 pp") and J["R-AB-2 SA256L vs SE-RRM @16"].startswith("ABOVE (97.50"), J["R-AB-2 SA256O vs SE-RRM @16"]; n += 1
        assert J["R-AB-3 LOCATED @16"] == "LOCATED: SA256O (-4.20 pp = 94 % of the 4.47 pp to SE-RRM)", J["R-AB-3 LOCATED @16"]; n += 1
        assert J["R-AB-3 LOCATED @64"].startswith("NOT-LOCATED"), J["R-AB-3 LOCATED @64"]; n += 1
        assert J["R-AB-4 SELECTOR SA256S"].startswith("DIRTY") and J["R-AB-4 SELECTOR SA256L"].startswith("CLEAN") and J["R-AB-4 SELECTOR SA256"].startswith("CLEAN"), J["R-AB-4 SELECTOR SA256S"]; n += 1
        assert "selected 030000 EDGE" in J["STABILITY SA256O"] and "EDGE" not in J["STABILITY SA256L"] and "gap -0.20 pp" in J["STABILITY SA256L"] and "non-finite loss rows 0" in J["STABILITY SA256L"]; n += 1
    with tempfile.TemporaryDirectory() as t_:   # an arm ABOVE the reference beyond the floor is labeled, never "located"; a lone arm is marked partial
        t = Path(t_); _mk(t, {"SA256": (.94, .97), "SA256L": (.975, .996)}); PF.LINES.clear(); J = analyze(t)
        assert J["R-AB-1 SA256L vs SA256 @16"].startswith("ABOVE-BEYOND (+3.50 pp") and J["R-AB-3 LOCATED @16"].startswith("NOT-LOCATED") and J["R-AB-3 LOCATED @16"].endswith("[only 1 of 3 arms present]"), J["R-AB-3 LOCATED @16"]; n += 1
    for kw, frag in ((dict(av_over={"SA256L": dict(trm_beta=0.01)}), "SA256L argv differs from SA256's in ['trm_lambda'], registered ['trm_beta', 'trm_lambda']"),
                     (dict(av_over={"SA256S": dict(wd=0.1)}), "SA256S argv differs from SA256's in ['fpa_k', 'trm_ri_sigma', 'wd']"),
                     (dict(av_over={"SA256O": dict(lr=3e-4)}), "SA256O argv lr = 0.0003 != the registered 0.0005"),
                     (dict(cf_over={"SA256O": dict(trm_lambda=0.0)}), "SA256O model config differs from SA256's in ['trm_lambda'], registered []"),
                     (dict(cf_over={"SA256S": dict(fpa_k=1)}), "SA256S model config differs from SA256's in ['trm_ri_sigma']"),
                     (dict(ext="SA256L"), "SA256L carries an extension marker"), (dict(grid_split="SA256S"), "SA256S rows on 2 grids"), (dict(ids_shift="SA256O"), "SA256O t64: puzzle ids differ from SA256's"),
                     (dict(steps_end={"SA256O": 18000}), "SA256O last training step 18000"), (dict(av_over={"SA256L": dict(seed=1)}), "SA256L argv seed/steps 1/30000")):
        with tempfile.TemporaryDirectory() as t_:
            t = Path(t_); _mk(t, base, **kw); PF.LINES.clear(); J = analyze(t); assert J["INTEGRITY"].startswith("FAIL") and frag in J["INTEGRITY"], (frag, J["INTEGRITY"]); n += 1
    with tempfile.TemporaryDirectory() as t_:   # the math-neutral --remat retry and a resume are NOT failures; remat is reported
        t = Path(t_); _mk(t, base, av_over={"SA256O": dict(remat=True, resume="runs/pretrainchamp_SA256O/ckpt_latest.pkl")}, cf_over={"SA256O": dict(remat=True)}); PF.LINES.clear(); J = analyze(t)
        assert J["INTEGRITY"].startswith("PASS [note: SA256O differs in the math-neutral ['remat']"), J["INTEGRITY"]; n += 1
    with tempfile.TemporaryDirectory() as t_:   # the reference absent -> a loud failure, never a silent unpaired read
        t = Path(t_); _mk(t, {k: v for k, v in base.items() if k != "SA256"}); PF.LINES.clear(); J = analyze(t)
        assert J["INTEGRITY"].startswith("FAIL") and "SA256 (the reference) config.json absent" in J["INTEGRITY"] and J["R-AB-1 SA256L vs SA256 @16"] == "NO-DATA"; n += 1
    with tempfile.TemporaryDirectory() as t_:
        PF.LINES.clear(); assert analyze(Path(t_)) == {"INTEGRITY": "NO-DATA"}; n += 1
    print(f"selftest OK: {n}/{n}")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true"); ap.add_argument("--root"); ap.add_argument("--out"); a = ap.parse_args()
    if a.selftest: return selftest()
    J = analyze(a.root)
    if a.out:
        o = Path(a.out); o.mkdir(parents=True, exist_ok=True); (o / "sablate_verdict.txt").write_text("\n".join(PF.LINES) + "\n"); (o / "sablate_verdict.json").write_text(json.dumps(J, indent=1))

if __name__ == "__main__":
    main()
