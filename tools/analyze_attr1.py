#!/usr/bin/env python3
"""SE-RRM ATTRIBUTION, ROUND 1 — analyzer (Note_2026-10-02_Attribution_Round1_Registration.md; FROZEN at the registration commit, before any row).

The question (the PI, 2026-10-01): isolate the mechanism of our accuracy gain over SE-RRM. SE-RRM's mixers inside our block, loop and recipe
(the ladder's SA256) read 98.24 / 99.60 on the 5,000 where SE-RRM's paper reports 93.73 / 98.22. The recipe ablation (2026-09-19) found
damping/noise and start-up levers inside the floor and SE-RRM's TABLE optimizer (batch 272, lr 5e-4) far below (85.42 / 91.50). SE-RRM's
RELEASED Sudoku command uses batch 272 at the config default lr 1e-4 (= ours) for 10,000 epochs = about 36.8k steps. Two arms, SA256 otherwise:
  SA256B   batch 272 (lr 1e-4 as ours), 36,000 steps          = SE-RRM's released optimizer AND its training budget (about 9.8M rows)
  SA256BR  batch 272 (lr 1e-4 as ours), 84,000 steps, monitor/grid every 5,600 = batch 272 at the reference's training rows (about 22.8M)
Same seed (0), the same selection rule, the same 5,000 test puzzles at 16 and 64 iterations. The reference is the ladder's banked SA256 row
(30k, selected 28k); it is READ, never re-run. One seed per arm: every difference is read against the seeded floors (2.58 / 2.44 pp).

RULES (frozen; R-AB-1..4 are the recipe ablation's, verbatim in meaning):
  INTEGRITY  each arm's trainer argv differs from SA256's in EXACTLY the registered keys at the registered values (`out`, `resume` ignored;
             `remat` math-neutral), the model config in none; seed 0; its own budget, no extension marker; last training step = its budget;
             rows n 5,000 / EMA / the right depth on SA256's puzzle ids; all of an arm's rows and its scan on its selected grid.
  R-AB-1     <arm> vs SA256 at 16 and 64, paired exact McNemar on the identical 5,000: INSIDE the floor, BELOW-BEYOND or ABOVE-BEYOND.
  R-AB-2     <arm> vs SE-RRM's published 93.73 / 98.22: WITHIN / ABOVE / BELOW (different evaluation sets; their one run).
  R-AB-3     LOCATED at 16 / 64 = the arms that read BELOW-BEYOND (share of the SA256-to-SE-RRM gap printed only then).
  R-AB-4     SELECTOR per arm from its k32 scan: CLEAN at a spurious rate <= 1 %, else DIRTY.
  R-A1-5     THE PAIR'S READING at 16 (the registered question is posed at 16, where the SE-RRM gap exceeds the floor):
               ROWS     SA256B BELOW-BEYOND and SA256BR INSIDE   (SE-RRM's shorter training carries the gap; batch size does not)
               BATCH    both BELOW-BEYOND                         (the smaller batch itself costs accuracy at equal rows)
               NEITHER  both INSIDE                               (neither the released optimizer nor the budget explains the gap)
               OTHER    any other combination (reported with both letters)
  STABILITY  descriptive: the selected grid (EDGE when it is the arm's last grid), validation max/end, scan fixed vs one random start, verified.

  .venv/bin/python tools/analyze_attr1.py --root <stage>/runs --out <dir>      (the stage holds both arms AND the ladder's SA256 dirs)
  .venv/bin/python tools/analyze_attr1.py --selftest
"""
from __future__ import annotations
import argparse, json, sys, tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_paperfinal as PF   # Root, jload, paired, spurious, metrics, say, pp, _write_row
import analyze_wladder as WL      # row, acc_on, restrict
import analyze_sablate as AB      # same, dict_diff, cfg_of, label1, label2 (the recipe ablation's frozen helpers, imported unchanged)

FLOOR16, FLOOR64 = AB.FLOOR16, AB.FLOOR64
SERRM16, SERRM64 = AB.SERRM16, AB.SERRM64
REF, SPUR_CLEAN = "SA256", 0.01
IGNORE, NEUTRAL = {"out", "resume"}, {"remat"}
ARMS = {   # arm -> (what it moves, its budget, the registered argv differences from SA256, the registered model-config differences)
    "SA256B": ("SE-RRM's released optimizer and budget: batch 272, lr 1e-4, 36,000 steps", 36000, dict(batch=272, steps=36000), {}),
    "SA256BR": ("batch 272 at the reference's training rows: 84,000 steps, monitor/grid every 5,600", 84000,
                dict(batch=272, steps=84000, monitor_every=5600, grid_every=5600), {}),
}

def integrity(root, arms):
    bad, notes = [], []; rav, rcf = AB.cfg_of(root, REF); ids0 = None
    if not rav: bad.append(f"{REF} (the reference) config.json absent or without an argv dict")
    for t in (16, 64):
        s, r = WL.row(root, REF, t)
        if not s or r is None: bad.append(f"{REF} row t{t} absent"); continue
        if s.get("n") != 5000 or s.get("t_total") != t or not s.get("ema"): bad.append(f"{REF} t{t}: n/t/ema {s.get('n')}/{s.get('t_total')}/{s.get('ema')}")
        ids = np.sort(np.asarray(r["idx"]))
        if ids0 is None: ids0 = ids
        elif len(ids) != len(ids0) or (ids != ids0).any(): bad.append(f"{REF} t{t}: puzzle ids differ from its t16 row's")
    for a in arms:
        _, budget, want_av, want_cf = ARMS[a]; av, cf = AB.cfg_of(root, a)
        if not av: bad.append(f"{a} config.json absent or without an argv dict"); continue
        d_av = AB.dict_diff(av, rav, skip=IGNORE); neutral = [k for k in d_av if k in NEUTRAL]; d_av = [k for k in d_av if k not in NEUTRAL]
        if neutral: notes.append(f"{a} differs in the math-neutral {neutral}")
        if d_av != sorted(want_av): bad.append(f"{a} argv differs from {REF}'s in {d_av}, registered {sorted(want_av)}")
        for k, v in want_av.items():
            if not AB.same(av.get(k), v): bad.append(f"{a} argv {k} = {av.get(k)} != the registered {v}")
        d_cf = [k for k in AB.dict_diff(cf, rcf) if k not in NEUTRAL]
        if d_cf != sorted(want_cf): bad.append(f"{a} model config differs from {REF}'s in {d_cf}, registered {sorted(want_cf)}")
        if av.get("seed") != 0 or av.get("steps") != budget: bad.append(f"{a} argv seed/steps {av.get('seed')}/{av.get('steps')}")
        if (root.pdir(a) / "EXTENDED.txt").exists(): bad.append(f"{a} carries an extension marker (the budget is fixed)")
        m = [r for r in PF.metrics(root, a) if "loss" in r]
        if m and int(m[-1].get("step", -1)) not in (budget, budget - 1): bad.append(f"{a} last training step {m[-1].get('step')} != {budget}")
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

def pair_reading(l16):
    """l16: {arm: label at 16} for SA256B and SA256BR -> the registered R-A1-5 reading."""
    b, br = l16.get("SA256B"), l16.get("SA256BR")
    if b is None or br is None or "n/a" in (b, br): return "NO-DATA"
    if b == "BELOW-BEYOND" and br == "INSIDE": return "ROWS"
    if b == "BELOW-BEYOND" and br == "BELOW-BEYOND": return "BATCH"
    if b == "INSIDE" and br == "INSIDE": return "NEITHER"
    return f"OTHER (SA256B {b}, SA256BR {br})"

def analyze(root_path):
    root = PF.Root(root_path); J = {}
    arms = [a for a in ARMS if root.summ(root.chain(a, "sub5k_vsel_t64")) or root.summ(root.chain(a, "sub5k_vsel_t16")) or (root.pdir(a) / "config.json").exists()]
    PF.say(f"SE-RRM ATTRIBUTION ROUND 1 — analyzer (frozen rules; one seed per arm; floors {100*FLOOR16:.2f} pp at 16 and {100*FLOOR64:.2f} pp at 64; the reference = the ladder's {REF})")
    PF.say(f"arms present: {arms or 'none'}")
    if not arms: PF.say("NO-DATA"); return {"INTEGRITY": "NO-DATA"}
    bad, notes, ids = integrity(root, arms)
    J["INTEGRITY"] = ("PASS" if not bad else "FAIL: " + "; ".join(bad)) + (f" [note: {'; '.join(notes)}]" if notes else ""); PF.say(f"INTEGRITY                {J['INTEGRITY']}")
    A = {}
    for a in [REF] + arms:
        A[a] = {}
        for t in (16, 64):
            s, r = WL.row(root, a, t); A[a][t] = WL.restrict(r, ids) if (r is not None and ids is not None) else r
            A[a][f"acc{t}"], A[a][f"n{t}"] = WL.acc_on(r, ids) if (r is not None and ids is not None) else ((s or {}).get("exact_acc"), (s or {}).get("n", 0))
        PF.say(f"  {a:7s} 16: {PF.pp(A[a]['acc16'])} (n {A[a]['n16']})   64: {PF.pp(A[a]['acc64'])} (n {A[a]['n64']})" + ("" if a == REF else f"   [{ARMS[a][0]}]"))
    located = {16: [], 64: []}; l16 = {}
    for a in arms:
        for t, fl, se in ((16, FLOOR16, SERRM16), (64, FLOOR64, SERRM64)):
            q = PF.paired(A[a][t], A[REF][t]); k = f"R-AB-1 {a} vs {REF} @{t}"
            lab = AB.label1(q["diff"], fl) if q else "n/a"
            if t == 16: l16[a] = lab
            J[k] = "NO-DATA" if not q else f"{lab} ({100*q['diff']:+.2f} pp, only-{a} {q['only_a']}, only-{REF} {q['only_b']}, n {q['n']}, p {q['p']:.2g})"; PF.say(f"{k:31s} {J[k]}")
            if q and lab == "BELOW-BEYOND":
                gap = (A[REF][f"acc{t}"] or 0) - se; located[t].append(f"{a} ({100*q['diff']:+.2f} pp" + (f" = {100 * -q['diff'] / gap:.0f} % of the {100*gap:.2f} pp to SE-RRM" if gap > fl else "") + ")")
            acc = A[a][f"acc{t}"]; k2 = f"R-AB-2 {a} vs SE-RRM @{t}"
            J[k2] = "NO-DATA" if acc is None else f"{AB.label2(acc, se, fl)} ({PF.pp(acc)} vs the published {100*se:.2f}; {100*(acc - se):+.2f} pp; different evaluation sets, their one run)"; PF.say(f"{k2:31s} {J[k2]}")
    for t in (16, 64):
        have = [a for a in arms if A[a][f"acc{t}"] is not None]; k = f"R-AB-3 LOCATED @{t}"
        J[k] = "NO-DATA" if not have else (("LOCATED: " + "; ".join(located[t])) if located[t] else f"NOT-LOCATED (no arm of {have} reads below {REF} beyond the floor)")
        if have and len(have) < len(ARMS): J[k] += f" [only {len(have)} of {len(ARMS)} arms present]"
        PF.say(f"{k:31s} {J[k]}")
    J["R-A1-5 PAIR @16"] = pair_reading(l16); PF.say(f"{'R-A1-5 PAIR @16':31s} {J['R-A1-5 PAIR @16']}")
    for a in [REF] + arms:
        z = root.recs(root.scan32(a)); sp = PF.spurious(z) if z is not None else None; k = f"R-AB-4 SELECTOR {a}"
        J[k] = "NO-DATA" if sp is None else f"{'CLEAN' if sp <= SPUR_CLEAN else 'DIRTY'} (spurious k32 {100*sp:.2f} %)"; PF.say(f"{k:31s} {J[k]}")
    for a in arms:
        budget = ARMS[a][1]; sc = root.summ(root.scan32(a)) or {}; M = PF.metrics(root, a)
        m = [r["monitor"] for r in M if isinstance(r.get("monitor"), dict) and "val_t16_ema" in r["monitor"]]; vm = max((r["val_t16_ema"] for r in m), default=None)
        vb = (root.pdir(a) / "val_best.txt").read_text().split() if (root.pdir(a) / "val_best.txt").exists() else ["-"]
        edge = vb[0].isdigit() and int(vb[0]) == budget; nan = sum(1 for r in M if "loss" in r and not np.isfinite(r["loss"]))
        gap = (sc.get("b1_exact") - sc.get("exact_acc")) if (sc.get("b1_exact") is not None and sc.get("exact_acc") is not None) else None
        J[f"STABILITY {a}"] = (f"selected {vb[0]}{' EDGE (a lower bound: the budget ended on the maximum)' if edge else ''}; validation max {PF.pp(vm)}, end {PF.pp(m[-1]['val_t16_ema'] if m else None)}; "
                               f"scan fixed start {PF.pp(sc.get('exact_acc'))}, one random start {PF.pp(sc.get('b1_exact'))} (gap {'-' if gap is None else f'{100*gap:+.2f}'} pp), verified {PF.pp(sc.get('exact_acc_vote'))}; non-finite loss rows {nan}")
        PF.say(f"STABILITY {a:7s}         {J[f'STABILITY {a}']}")
    return J

# ---------------- selftest (hand-built records; printed labels asserted) ----------------
BASE_AV = dict(seed=0, steps=30000, batch=768, lr=1e-4, lr_end=1e-4, trm_lambda=0.05, trm_beta=0.01, trm_ri_sigma=1.0, fpa_k=1, fpa_eps=0.2, wd=1.0,
               monitor_every=2000, grid_every=2000, remat=False, resume=None)
BASE_CF = dict(cell_kind="dec", dec_width=256, dec_token_mixer="attn", trm_lambda=0.05, trm_beta=0.01, trm_ri_sigma=1.0, fpa_k=1, remat=False)

def _mk(tmp, accs, av_over=None, cf_over=None, ext=None, sel=None, steps_end=None, ids_shift=None):
    rng = np.random.default_rng(0); ids = np.arange(5000) * 3
    for a, (a16, a64) in accs.items():
        budget = ARMS[a][1] if a in ARMS else 30000
        vb = (sel or {}).get(a, "022400" if a == "SA256BR" else "022000")
        for t, acc in ((16, a16), (64, a64)):
            d = tmp / f"sxeval_pchamp{a}" / f"sub5k_vsel_t{t}"; ii = ids + 1 if (ids_shift == a and t == 64) else ids
            PF._write_row(d, 0, acc, rng, idx=ii, extra=dict(t_total=t, ema=True, subsample=5000, ckpt=f"runs/pretrainchamp_{a}/ckpt_{vb}.pkl"))
        av, cf = dict(BASE_AV, out=f"runs/pretrainchamp_{a}"), dict(BASE_CF)
        if a in ARMS: av.update(ARMS[a][2]); cf.update(ARMS[a][3])
        av.update((av_over or {}).get(a, {})); cf.update((cf_over or {}).get(a, {}))
        p = tmp / f"pretrainchamp_{a}"; p.mkdir(parents=True, exist_ok=True); (p / "config.json").write_text(json.dumps(dict(config=cf, argv=av))); (p / "val_best.txt").write_text(f"{vb} 0.9600 {int(vb)}")
        (p / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in [dict(monitor=dict(step=22000, val_t16_ema=0.96)), dict(step=(steps_end or {}).get(a, budget), loss=0.5), dict(monitor=dict(step=budget, val_t16_ema=0.95))]))
        if ext == a: (p / "EXTENDED.txt").write_text("x")
        s = tmp / f"sxscan_pchamp{a}"; s.mkdir(parents=True, exist_ok=True); (s / "summary_all.json").write_text(json.dumps(dict(ckpt=f"runs/pretrainchamp_{a}/ckpt_{vb}.pkl", exact_acc=a64, b1_exact=a64 - 0.002, exact_acc_vote=0.998, n=5000)))
        n, k = 400, 8; ex = rng.random((n, k)) < 0.9; res = np.where(ex, 0.001, 0.5)
        np.savez(s / "records_all.npz", idx=np.arange(n), mi_exact_k=ex, mi_resid_k=res)

def selftest():
    n = 0
    assert pair_reading({"SA256B": "BELOW-BEYOND", "SA256BR": "INSIDE"}) == "ROWS" and pair_reading({"SA256B": "BELOW-BEYOND", "SA256BR": "BELOW-BEYOND"}) == "BATCH"; n += 1
    assert pair_reading({"SA256B": "INSIDE", "SA256BR": "INSIDE"}) == "NEITHER" and pair_reading({"SA256B": "INSIDE", "SA256BR": "BELOW-BEYOND"}).startswith("OTHER"); n += 1
    assert pair_reading({"SA256B": "INSIDE"}) == "NO-DATA" and pair_reading({"SA256B": "ABOVE-BEYOND", "SA256BR": "INSIDE"}).startswith("OTHER"); n += 1
    base = {"SA256": (.982, .996), "SA256B": (.945, .985), "SA256BR": (.978, .995)}
    with tempfile.TemporaryDirectory() as t_:   # the budget arm falls beyond the floor at 16; the rows-matched arm stays inside -> ROWS
        t = Path(t_); _mk(t, base, sel={"SA256B": "036000"}); PF.LINES.clear(); J = analyze(t)
        assert J["INTEGRITY"] == "PASS", J["INTEGRITY"]; n += 1
        assert J["R-AB-1 SA256B vs SA256 @16"].startswith("BELOW-BEYOND (-3.70 pp") and J["R-AB-1 SA256BR vs SA256 @16"].startswith("INSIDE (-0.40 pp"), J["R-AB-1 SA256B vs SA256 @16"]; n += 1
        assert J["R-A1-5 PAIR @16"] == "ROWS" and J["R-AB-3 LOCATED @16"].startswith("LOCATED: SA256B (-3.70 pp = 83 % of the 4.47 pp to SE-RRM)"), (J["R-A1-5 PAIR @16"], J["R-AB-3 LOCATED @16"]); n += 1
        assert J["R-AB-2 SA256B vs SE-RRM @16"].startswith("WITHIN (94.50 vs the published 93.73") and "selected 036000 EDGE" in J["STABILITY SA256B"] and "EDGE" not in J["STABILITY SA256BR"]; n += 1
    with tempfile.TemporaryDirectory() as t_:   # both inside -> NEITHER
        t = Path(t_); _mk(t, {"SA256": (.982, .996), "SA256B": (.975, .994), "SA256BR": (.980, .995)}); PF.LINES.clear(); J = analyze(t)
        assert J["R-A1-5 PAIR @16"] == "NEITHER" and J["R-AB-3 LOCATED @16"].startswith("NOT-LOCATED"); n += 1
    for kw, frag in ((dict(av_over={"SA256BR": dict(monitor_every=2000)}), "SA256BR argv differs from SA256's in ['batch', 'grid_every', 'steps'], registered ['batch', 'grid_every', 'monitor_every', 'steps']"),
                     (dict(av_over={"SA256B": dict(lr=5e-4)}), "SA256B argv differs from SA256's in ['batch', 'lr', 'steps']"),
                     (dict(av_over={"SA256B": dict(steps=30000)}), "SA256B argv differs from SA256's in ['batch'], registered ['batch', 'steps']"),
                     (dict(cf_over={"SA256BR": dict(trm_lambda=0.0)}), "SA256BR model config differs from SA256's in ['trm_lambda'], registered []"),
                     (dict(ext="SA256B"), "SA256B carries an extension marker"), (dict(steps_end={"SA256BR": 56000}), "SA256BR last training step 56000 != 84000"),
                     (dict(ids_shift="SA256B"), "SA256B t64: puzzle ids differ from SA256's"), (dict(av_over={"SA256B": dict(seed=1)}), "SA256B argv seed/steps 1/36000")):
        with tempfile.TemporaryDirectory() as t_:
            t = Path(t_); _mk(t, base, **kw); PF.LINES.clear(); J = analyze(t); assert J["INTEGRITY"].startswith("FAIL") and frag in J["INTEGRITY"], (frag, J["INTEGRITY"]); n += 1
    with tempfile.TemporaryDirectory() as t_:   # the reference absent -> a loud failure
        t = Path(t_); _mk(t, {k: v for k, v in base.items() if k != "SA256"}); PF.LINES.clear(); J = analyze(t)
        assert J["INTEGRITY"].startswith("FAIL") and "SA256 (the reference) config.json absent" in J["INTEGRITY"] and J["R-AB-1 SA256B vs SA256 @16"] == "NO-DATA" and J["R-A1-5 PAIR @16"] == "NO-DATA"; n += 1
    with tempfile.TemporaryDirectory() as t_:
        PF.LINES.clear(); assert analyze(Path(t_)) == {"INTEGRITY": "NO-DATA"}; n += 1
    print(f"selftest OK: {n}/{n}")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true"); ap.add_argument("--root"); ap.add_argument("--out"); a = ap.parse_args()
    if a.selftest: return selftest()
    J = analyze(a.root)
    if a.out:
        o = Path(a.out); o.mkdir(parents=True, exist_ok=True); (o / "attr1_verdict.txt").write_text("\n".join(PF.LINES) + "\n"); (o / "attr1_verdict.json").write_text(json.dumps(J, indent=1))

if __name__ == "__main__":
    main()
