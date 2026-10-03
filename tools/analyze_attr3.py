#!/usr/bin/env python3
"""SE-RRM ATTRIBUTION, ROUND 3 — analyzer (Note_2026-10-03_Attribution_Round3_Registration.md; FROZEN at the registration commit, before any row).

The question (the PI, 2026-10-03: "settle the question once and for all"): round 2 found that SE-RRM's single recurrent state costs our system
8.28 pp at 16 iterations (SA256U 89.96 vs SA256 98.24), below SE-RRM's own published 93.73, and that the single state memorizes the training
puzzles after 14k steps while the two-state model never does. SE-RRM trains its single state with dropout 0.2 on the attention weights. Is the
two-state advantage something that regularization replaces? Round 3 adds SE-RRM's dropout (rate 0.2, the attention weights of both mixers, the
training forward only) to both state structures, completing a 2 x 2 with round 2:
                       no dropout                      dropout 0.2
  two states           SA256  (the ladder's, banked)   SA256D   (new)
  one state            SA256U (round 2's, banked)      SA256UD  (new)
Same seed (0), budget (30,000, fixed), selection rule, and the same 5,000 test puzzles at 16 and 64 iterations. SA256 and SA256U are READ,
never re-run. One seed: every difference is read against the seeded floors (2.58 / 2.44 pp).

RULES (frozen; R-AB-1/2/4 are the recipe ablation's, verbatim in meaning):
  INTEGRITY  SA256D's trainer argv and model config differ from SA256's in EXACTLY {dec_dropout: 0.2}; SA256UD's in EXACTLY {dec_single_state: True,
             dec_dropout: 0.2} (`out`, `resume` ignored; `remat` math-neutral); seed 0; budget 30,000, no extension marker; last training step =
             30,000; rows n 5,000 / EMA / the right depth on SA256's puzzle ids; all rows and the scan on the selected grid; SA256U's rows (round 2)
             present on the same ids; every D64 record carries the 64 per-iteration bits.
  R-AB-1     each new arm vs SA256 at 16 and 64, paired exact McNemar on the identical 5,000: INSIDE the floor, BELOW-BEYOND or ABOVE-BEYOND.
  R-AB-2     each new arm vs SE-RRM's published 93.73 / 98.22: WITHIN / ABOVE / BELOW (different evaluation sets; their one run).
  R-AB-4     SELECTOR per arm from its k32 scan: CLEAN at a spurious rate <= 1 %, else DIRTY.
  R-A3-1     THE STRUCTURE UNDER DROPOUT: SA256UD vs SA256D at 16 and 64 (the same label rule).
  R-A3-2     DROPOUT ON THE SINGLE STATE: SA256UD vs SA256U at 16 and 64.
  R-A3-3     DROPOUT ON THE TWO STATES: SA256D vs SA256 at 16 and 64.
  R-A3-4     THE REGULARIZED SINGLE STATE vs OUR REFERENCE: SA256UD vs SA256 at 16 and 64 (= R-AB-1 for SA256UD).
  R-A3-5     THE SETTLEMENT at 16 (the registered question; the same combination is printed at 64, descriptive):
               STRUCTURE-SPECIFIC       R-A3-1 BELOW-BEYOND (with SE-RRM's regularizer on both, the single state still loses beyond the floor)
               REGULARIZATION-REPLACES  R-A3-1 INSIDE and R-A3-4 INSIDE (with dropout, the single state matches both two-state models)
               PARTIAL                  R-A3-1 INSIDE and R-A3-4 BELOW-BEYOND (no structure effect under equal dropout; dropout costs the two states)
               OTHER                    any other combination (reported with its letters)
  DESCRIPTIVE (no label moves): the interaction (the structure effect without dropout, from round 2, against R-A3-1's); round 2's R-A2-6 / R-A2-7
             / CONSISTENCY readings (imported unchanged from tools/analyze_attr2.py) for each new arm against SA256; and MEMORIZATION per arm from
             its own metrics: the validation maximum and end, and the training rows' exact share at the selected step and at the end.
  STABILITY  descriptive: the selected grid (EDGE when it is the arm's last grid), validation max/end, scan fixed vs one random start, verified.

  .venv/bin/python tools/analyze_attr3.py --root <stage>/runs --out <dir>   (the stage holds SA256D, SA256UD, round 2's SA256U and the ladder's SA256)
  .venv/bin/python tools/analyze_attr3.py --selftest
"""
from __future__ import annotations
import argparse, json, sys, tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_paperfinal as PF   # Root, jload, paired, spurious, metrics, say, pp
import analyze_wladder as WL      # row, acc_on, restrict
import analyze_sablate as AB      # same, dict_diff, cfg_of, label1, label2 (the recipe ablation's frozen helpers, imported unchanged)
import analyze_attr2 as A2        # bits64, where_and_persistence (round 2's frozen readings, imported unchanged)

FLOOR16, FLOOR64 = AB.FLOOR16, AB.FLOOR64
SERRM16, SERRM64 = AB.SERRM16, AB.SERRM64
REF, U, SPUR_CLEAN = "SA256", "SA256U", 0.01
IGNORE, NEUTRAL = {"out", "resume"}, {"remat"}
ARMS = {   # arm -> (what it moves, its budget, the registered argv differences from SA256, the registered model-config differences)
    "SA256UD": ("one state + SE-RRM's dropout 0.2 on the attention weights (training only)", 30000,
                dict(dec_single_state=True, dec_dropout=0.2), dict(dec_single_state=True, dec_dropout=0.2)),
    "SA256D": ("two states + SE-RRM's dropout 0.2 on the attention weights (training only)", 30000,
               dict(dec_dropout=0.2), dict(dec_dropout=0.2)),
}
CONTRASTS = (("R-A3-1 STRUCTURE UNDER DROPOUT", "SA256UD", "SA256D"), ("R-A3-2 DROPOUT ON ONE STATE", "SA256UD", U),
             ("R-A3-3 DROPOUT ON TWO STATES", "SA256D", REF), ("R-A3-4 REGULARIZED ONE STATE vs REF", "SA256UD", REF))

def settle(l1, l4):
    """l1 = R-A3-1's label, l4 = R-A3-4's label at one depth -> the registered R-A3-5 reading."""
    if l1 is None or l4 is None or "n/a" in (l1, l4): return "NO-DATA"
    if l1 == "BELOW-BEYOND": return "STRUCTURE-SPECIFIC"
    if l1 == "INSIDE" and l4 == "INSIDE": return "REGULARIZATION-REPLACES"
    if l1 == "INSIDE" and l4 == "BELOW-BEYOND": return "PARTIAL"
    return f"OTHER (R-A3-1 {l1}, R-A3-4 {l4})"

def integrity(root, arms):
    bad, notes = [], []; rav, rcf = AB.cfg_of(root, REF); ids0 = None
    if not rav: bad.append(f"{REF} (the reference) config.json absent or without an argv dict")
    for a0 in (REF, U):
        for t in (16, 64):
            s, r = WL.row(root, a0, t)
            if not s or r is None: bad.append(f"{a0} row t{t} absent"); continue
            if s.get("n") != 5000 or s.get("t_total") != t or not s.get("ema"): bad.append(f"{a0} t{t}: n/t/ema {s.get('n')}/{s.get('t_total')}/{s.get('ema')}")
            ids = np.sort(np.asarray(r["idx"]))
            if ids0 is None: ids0 = ids
            elif len(ids) != len(ids0) or (ids != ids0).any(): bad.append(f"{a0} t{t}: puzzle ids differ from {REF}'s t16 row's")
            if t == 64 and A2.bits64(r) is None: bad.append(f"{a0} t64: no per-iteration bits")
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
        for k, v in want_cf.items():
            if not AB.same(cf.get(k), v): bad.append(f"{a} model config {k} = {cf.get(k)} != the registered {v}")
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
            if t == 64 and A2.bits64(r) is None: bad.append(f"{a} t64: no per-iteration bits")
        sc = root.summ(root.scan32(a))
        if sc: grids.add(str(sc.get("ckpt")))
        if len(grids) > 1: bad.append(f"{a} rows on {len(grids)} grids {sorted(grids)}")
        if vb and grids and vb not in next(iter(grids)): bad.append(f"{a} rows not on the selected grid {vb}")
    return bad, notes, ids0

def memorization(root, a):
    """Descriptive: validation max / end, and the training rows' exact share at the selected step and at the end, from the arm's metrics."""
    M = PF.metrics(root, a)
    mon = [r["monitor"] for r in M if isinstance(r.get("monitor"), dict) and "val_t16_ema" in r["monitor"]]
    tr = {int(r["step"]): r.get("train_exact") for r in M if "loss" in r and "step" in r and r.get("train_exact") is not None}
    vb = (root.pdir(a) / "val_best.txt").read_text().split() if (root.pdir(a) / "val_best.txt").exists() else ["-"]
    near = lambda s: tr[min(tr, key=lambda x: abs(x - s))] if tr else None
    sel = int(vb[0]) if vb[0].isdigit() else None
    if not mon: return "NO-DATA"
    vm = max(m["val_t16_ema"] for m in mon); ve = mon[-1]["val_t16_ema"]
    te_sel = near(sel) if sel is not None else None; te_end = near(max(tr)) if tr else None
    f = lambda x: "-" if x is None else f"{x:.3f}"
    return (f"validation max {PF.pp(vm)}, end {PF.pp(ve)} (end - max {100*(ve - vm):+.2f} pp); training rows' exact share at the selected step "
            f"{f(te_sel)}, at the end {f(te_end)}")

def analyze(root_path):
    root = PF.Root(root_path); J = {}
    arms = [a for a in ARMS if root.summ(root.chain(a, "sub5k_vsel_t64")) or root.summ(root.chain(a, "sub5k_vsel_t16")) or (root.pdir(a) / "config.json").exists()]
    PF.say(f"SE-RRM ATTRIBUTION ROUND 3 — analyzer (frozen rules; one seed per arm; floors {100*FLOOR16:.2f} pp at 16 and {100*FLOOR64:.2f} pp at 64; "
           f"the references = the ladder's {REF} and round 2's {U})")
    PF.say(f"arms present: {arms or 'none'}")
    if not arms: PF.say("NO-DATA"); return {"INTEGRITY": "NO-DATA"}
    bad, notes, ids = integrity(root, arms)
    J["INTEGRITY"] = ("PASS" if not bad else "FAIL: " + "; ".join(bad)) + (f" [note: {'; '.join(notes)}]" if notes else ""); PF.say(f"INTEGRITY                {J['INTEGRITY']}")
    A = {}
    for a in [REF, U] + arms:
        A[a] = {}
        for t in (16, 64):
            s, r = WL.row(root, a, t); A[a][t] = WL.restrict(r, ids) if (r is not None and ids is not None) else r
            A[a][f"acc{t}"], A[a][f"n{t}"] = WL.acc_on(r, ids) if (r is not None and ids is not None) else ((s or {}).get("exact_acc"), (s or {}).get("n", 0))
        PF.say(f"  {a:7s} 16: {PF.pp(A[a]['acc16'])} (n {A[a]['n16']})   64: {PF.pp(A[a]['acc64'])} (n {A[a]['n64']})" + (f"   [{ARMS[a][0]}]" if a in ARMS else ""))
    lab = {}
    for name, a, b in [("R-AB-1 " + x + " vs " + REF, x, REF) for x in arms] + list(CONTRASTS):
        for t, fl in ((16, FLOOR16), (64, FLOOR64)):
            q = PF.paired(A.get(a, {}).get(t), A.get(b, {}).get(t)) if (a in A and b in A) else None; k = f"{name} @{t}"
            lab[(name, t)] = AB.label1(q["diff"], fl) if q else "n/a"
            J[k] = "NO-DATA" if not q else f"{lab[(name, t)]} ({100*q['diff']:+.2f} pp, only-{a} {q['only_a']}, only-{b} {q['only_b']}, n {q['n']}, p {q['p']:.2g})"
            PF.say(f"{k:42s} {J[k]}")
    for a in arms:
        for t, se in ((16, SERRM16), (64, SERRM64)):
            acc = A[a][f"acc{t}"]; k2 = f"R-AB-2 {a} vs SE-RRM @{t}"
            J[k2] = "NO-DATA" if acc is None else f"{AB.label2(acc, se, FLOOR16 if t == 16 else FLOOR64)} ({PF.pp(acc)} vs the published {100*se:.2f}; {100*(acc - se):+.2f} pp; different evaluation sets, their one run)"
            PF.say(f"{k2:42s} {J[k2]}")
    for t in (16, 64):
        k = f"R-A3-5 SETTLEMENT @{t}" + ("" if t == 16 else " (descriptive)")
        J[k] = settle(lab.get(("R-A3-1 STRUCTURE UNDER DROPOUT", t)), lab.get(("R-A3-4 REGULARIZED ONE STATE vs REF", t))); PF.say(f"{k:42s} {J[k]}")
        q0 = PF.paired(A[U][t], A[REF][t]); q1 = PF.paired(A["SA256UD"][t], A["SA256D"][t]) if ("SA256UD" in A and "SA256D" in A) else None
        J[f"INTERACTION @{t}"] = ("NO-DATA" if not (q0 and q1) else f"the structure effect {100*q0['diff']:+.2f} pp without dropout, {100*q1['diff']:+.2f} pp with it "
                                  f"(difference {100*(q1['diff'] - q0['diff']):+.2f} pp; descriptive)"); PF.say(f"{'INTERACTION @' + str(t) + ' (descriptive)':42s} {J[f'INTERACTION @{t}']}")
    for a in [REF, U] + arms:
        z = root.recs(root.scan32(a)); sp = PF.spurious(z) if z is not None else None; k = f"R-AB-4 SELECTOR {a}"
        J[k] = "NO-DATA" if sp is None else f"{'CLEAN' if sp <= SPUR_CLEAN else 'DIRTY'} (spurious k32 {100*sp:.2f} %)"; PF.say(f"{k:42s} {J[k]}")
    PF.say("DESCRIPTIVE (round 2's readings, imported unchanged; each new arm against SA256):")
    A2.where_and_persistence(root, arms, ids, J)
    for a in [REF, U] + arms:
        J[f"MEMORIZATION {a}"] = memorization(root, a); PF.say(f"MEMORIZATION {a:7s} (descriptive)   {J[f'MEMORIZATION {a}']}")
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
EXTRA = {"SA256U": (dict(dec_single_state=True), dict(dec_single_state=True))}

def _mk(tmp, accs, av_over=None, cf_over=None, ext=None, sel=None, steps_end=None, ids_shift=None, nobits=None):
    rng = np.random.default_rng(0); ids = np.arange(5000) * 3
    for a, (a16, a64) in accs.items():
        budget = ARMS[a][1] if a in ARMS else 30000; vb = (sel or {}).get(a, "028000")
        b = A2._traj(len(ids), a16, a64, rng); fe = np.where(b.any(axis=1), b.argmax(axis=1), -1)
        for t, ex in ((16, b[:, 15]), (64, b[:, -1])):
            d = tmp / f"sxeval_pchamp{a}" / f"sub5k_vsel_t{t}"; d.mkdir(parents=True, exist_ok=True); ii = ids + 1 if (ids_shift == a and t == 64) else ids
            rec = dict(idx=ii, cold_exact=ex, first_exact=fe)
            if t == 64 and nobits != a: rec["exact_by_step"] = np.packbits(b.astype(np.uint8), axis=1, bitorder="little")
            np.savez(d / "records_all.npz", **rec)
            (d / "summary_all.json").write_text(json.dumps(dict(n=len(ii), exact_acc=float(ex.mean()), t_total=t, ema=True, subsample=5000, ckpt=f"runs/pretrainchamp_{a}/ckpt_{vb}.pkl")))
        av, cf = dict(BASE_AV, out=f"runs/pretrainchamp_{a}"), dict(BASE_CF)
        if a in ARMS: av.update(ARMS[a][2]); cf.update(ARMS[a][3])
        if a in EXTRA: av.update(EXTRA[a][0]); cf.update(EXTRA[a][1])
        for k, v in (av_over or {}).get(a, {}).items(): av.pop(k) if v is None else av.__setitem__(k, v)
        for k, v in (cf_over or {}).get(a, {}).items(): cf.pop(k) if v is None else cf.__setitem__(k, v)
        p = tmp / f"pretrainchamp_{a}"; p.mkdir(parents=True, exist_ok=True); (p / "config.json").write_text(json.dumps(dict(config=cf, argv=av))); (p / "val_best.txt").write_text(f"{vb} 0.9600 {int(vb)}")
        (p / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in [dict(monitor=dict(step=22000, val_t16_ema=0.96)), dict(step=22000, loss=0.6, train_exact=0.3),
                                                                            dict(step=28000, loss=0.55, train_exact=0.32), dict(step=(steps_end or {}).get(a, budget), loss=0.5, train_exact=0.35), dict(monitor=dict(step=budget, val_t16_ema=0.95))]))
        if ext == a: (p / "EXTENDED.txt").write_text("x")
        s = tmp / f"sxscan_pchamp{a}"; s.mkdir(parents=True, exist_ok=True); (s / "summary_all.json").write_text(json.dumps(dict(ckpt=f"runs/pretrainchamp_{a}/ckpt_{vb}.pkl", exact_acc=a64, b1_exact=a64 - 0.002, exact_acc_vote=0.998, n=5000)))
        n, k = 400, 8; ex = rng.random((n, k)) < 0.9; res = np.where(ex, 0.001, 0.5)
        np.savez(s / "records_all.npz", idx=np.arange(n), mi_exact_k=ex, mi_resid_k=res)

def selftest():
    n = 0
    assert settle("BELOW-BEYOND", "BELOW-BEYOND") == "STRUCTURE-SPECIFIC" and settle("BELOW-BEYOND", "INSIDE") == "STRUCTURE-SPECIFIC"; n += 1
    assert settle("INSIDE", "INSIDE") == "REGULARIZATION-REPLACES" and settle("INSIDE", "BELOW-BEYOND") == "PARTIAL"; n += 1
    assert settle("ABOVE-BEYOND", "INSIDE").startswith("OTHER") and settle("INSIDE", "ABOVE-BEYOND").startswith("OTHER") and settle(None, "INSIDE") == "NO-DATA"; n += 1
    base = {"SA256": (.982, .996), "SA256U": (.900, .931), "SA256UD": (.930, .965), "SA256D": (.980, .995)}
    with tempfile.TemporaryDirectory() as t_:   # dropout lifts the single state but it stays beyond the floor below the regularized two states -> STRUCTURE-SPECIFIC
        t = Path(t_); _mk(t, base); PF.LINES.clear(); J = analyze(t)
        assert J["INTEGRITY"] == "PASS", J["INTEGRITY"]; n += 1
        assert J["R-A3-1 STRUCTURE UNDER DROPOUT @16"].startswith("BELOW-BEYOND (-5.00 pp") and J["R-A3-2 DROPOUT ON ONE STATE @16"].startswith("ABOVE-BEYOND (+3.00 pp"), (J["R-A3-1 STRUCTURE UNDER DROPOUT @16"], J["R-A3-2 DROPOUT ON ONE STATE @16"]); n += 1
        assert J["R-A3-3 DROPOUT ON TWO STATES @16"].startswith("INSIDE (-0.20 pp") and J["R-A3-4 REGULARIZED ONE STATE vs REF @16"].startswith("BELOW-BEYOND (-5.20 pp"); n += 1
        assert J["R-A3-5 SETTLEMENT @16"] == "STRUCTURE-SPECIFIC" and J["INTERACTION @16"].startswith("the structure effect -8.20 pp without dropout, -5.00 pp with it (difference +3.20 pp"), (J["R-A3-5 SETTLEMENT @16"], J["INTERACTION @16"]); n += 1
        assert J["R-AB-2 SA256UD vs SE-RRM @16"].startswith("WITHIN (93.00 vs the published 93.73") and J["R-AB-4 SELECTOR SA256UD"].startswith("CLEAN") and "R-A2-6 WHERE SA256UD" in J; n += 1
        assert J["MEMORIZATION SA256UD"].startswith("validation max 96.00, end 95.00 (end - max -1.00 pp); training rows' exact share at the selected step 0.320, at the end 0.350"), J["MEMORIZATION SA256UD"]; n += 1
    with tempfile.TemporaryDirectory() as t_:   # dropout closes the gap to both two-state models -> REGULARIZATION-REPLACES
        t = Path(t_); _mk(t, dict(base, SA256UD=(.975, .994))); PF.LINES.clear(); J = analyze(t)
        assert J["R-A3-5 SETTLEMENT @16"] == "REGULARIZATION-REPLACES" and J["INTEGRITY"] == "PASS", J["R-A3-5 SETTLEMENT @16"]; n += 1
    with tempfile.TemporaryDirectory() as t_:   # dropout costs the two states as much as it gains the one -> PARTIAL
        t = Path(t_); _mk(t, dict(base, SA256UD=(.950, .985), SA256D=(.955, .987))); PF.LINES.clear(); J = analyze(t)
        assert J["R-A3-5 SETTLEMENT @16"] == "PARTIAL" and J["R-A3-3 DROPOUT ON TWO STATES @16"].startswith("BELOW-BEYOND"), J["R-A3-5 SETTLEMENT @16"]; n += 1
    for kw, frag in ((dict(av_over={"SA256UD": dict(dec_dropout=None)}, cf_over={"SA256UD": dict(dec_dropout=None)}), "SA256UD argv differs from SA256's in ['dec_single_state'], registered ['dec_dropout', 'dec_single_state']"),
                     (dict(av_over={"SA256D": dict(dec_dropout=0.1)}, cf_over={"SA256D": dict(dec_dropout=0.1)}), "SA256D argv dec_dropout = 0.1 != the registered 0.2"),
                     (dict(cf_over={"SA256D": dict(dec_single_state=True)}), "SA256D model config differs from SA256's in ['dec_dropout', 'dec_single_state'], registered ['dec_dropout']"),
                     (dict(av_over={"SA256D": dict(steps=36000)}), "SA256D argv differs from SA256's in ['dec_dropout', 'steps']"),
                     (dict(ext="SA256UD"), "SA256UD carries an extension marker"), (dict(steps_end={"SA256D": 22000}), "SA256D last training step 22000 != 30000"),
                     (dict(ids_shift="SA256UD"), "SA256UD t64: puzzle ids differ from SA256's"), (dict(nobits="SA256D"), "SA256D t64: no per-iteration bits"),
                     (dict(ids_shift="SA256U"), "SA256U t64: puzzle ids differ from SA256's t16 row's")):
        with tempfile.TemporaryDirectory() as t_:
            t = Path(t_); _mk(t, base, **kw); PF.LINES.clear(); J = analyze(t); assert J["INTEGRITY"].startswith("FAIL") and frag in J["INTEGRITY"], (frag, J["INTEGRITY"]); n += 1
    with tempfile.TemporaryDirectory() as t_:   # round 2's arm absent -> a loud failure and no settlement
        t = Path(t_); _mk(t, {k: v for k, v in base.items() if k != "SA256U"}); PF.LINES.clear(); J = analyze(t)
        assert J["INTEGRITY"].startswith("FAIL") and "SA256U row t16 absent" in J["INTEGRITY"] and J["R-A3-2 DROPOUT ON ONE STATE @16"] == "NO-DATA"; n += 1
    with tempfile.TemporaryDirectory() as t_:   # only one new arm present -> the settlement says NO-DATA
        t = Path(t_); _mk(t, {k: v for k, v in base.items() if k != "SA256D"}); PF.LINES.clear(); J = analyze(t)
        assert J["R-A3-5 SETTLEMENT @16"] == "NO-DATA" and J["R-A3-2 DROPOUT ON ONE STATE @16"].startswith("ABOVE-BEYOND"), J["R-A3-5 SETTLEMENT @16"]; n += 1
    with tempfile.TemporaryDirectory() as t_:
        PF.LINES.clear(); assert analyze(Path(t_)) == {"INTEGRITY": "NO-DATA"}; n += 1
    print(f"selftest OK: {n}/{n}")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true"); ap.add_argument("--root"); ap.add_argument("--out"); a = ap.parse_args()
    if a.selftest: return selftest()
    J = analyze(a.root)
    if a.out:
        o = Path(a.out); o.mkdir(parents=True, exist_ok=True); (o / "attr3_verdict.txt").write_text("\n".join(PF.LINES) + "\n"); (o / "attr3_verdict.json").write_text(json.dumps(J, indent=1))

if __name__ == "__main__":
    main()
