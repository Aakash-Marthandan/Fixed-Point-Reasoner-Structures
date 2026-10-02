#!/usr/bin/env python3
"""SE-RRM ATTRIBUTION, ROUND 2 — analyzer (Note_2026-10-02_Attribution_Round2_Registration.md; FROZEN at the registration commit, before any row).

The question (the PI, 2026-10-01): isolate the mechanism of our accuracy gain over SE-RRM. SE-RRM's mixers inside our block, loop and recipe
(the ladder's SA256) read 98.24 / 99.60 on the 5,000 where SE-RRM's paper reports 93.73 / 98.22. Already moved: damping and noise, randomized
starts and anchor rows (the recipe ablation, 2026-09-19: inside the floor); SE-RRM's released optimizer and budget (round 1, 2026-10-02).
Round 2 moves the STATE STRUCTURE. SE-RRM's released model carries ONE recurrent state ("there is only one hidden variable z, no distinction
between higher and lower modules": z <- L(z + x), 3 x 6 = 18 times per step, the gradient through the last 6). Ours carries two: the fast state
updated 6 times from the slow state plus the input, then the slow state once from the fast state; three cycles = 21 applications, the gradient
through the last 7. One arm, SA256 otherwise:
  SA256U  --dec-single-state: ONE carry (the slow slot, which the readout and the halting head read) updated 3 x (6 + 1) = 21 times per segment as
          z <- 0.05 z + 0.95 B(z + x) + noise, the gradient through the last 7 applications. This is TRM's own single-z variant (its Figure 4; its
          Table 2: 71.9 against 87.4 for two features, in TRM's MLP model and recipe). The application count and the gradient window are ours,
          so the arm moves the structure alone.
Same seed (0), the same fixed 30,000-step budget and selection rule, the same 5,000 test puzzles at 16 and 64 iterations. The reference is the
ladder's banked SA256 row (30k, selected 28k); it is READ, never re-run. One seed: every difference is read against the seeded floors (2.58 / 2.44 pp).

RULES (frozen; R-AB-1..4 are the recipe ablation's, verbatim in meaning):
  INTEGRITY   SA256U's trainer argv differs from SA256's in EXACTLY {dec_single_state: True} (`out`, `resume` ignored; `remat` math-neutral), its
              model config in EXACTLY {dec_single_state: True}; seed 0; budget 30,000, no extension marker; last training step = 30,000; rows
              n 5,000 / EMA / the right depth on SA256's puzzle ids; all its rows and its scan on its selected grid; both arms' D64 records carry
              the 64 per-iteration bits (R-A2-6 and R-A2-7 read them).
  R-AB-1      SA256U vs SA256 at 16 and 64, paired exact McNemar on the identical 5,000: INSIDE the floor, BELOW-BEYOND or ABOVE-BEYOND.
  R-AB-2      SA256U vs SE-RRM's published 93.73 / 98.22: WITHIN / ABOVE / BELOW (different evaluation sets; their one run).
  R-AB-3      LOCATED at 16 / 64 = SA256U when it reads BELOW-BEYOND, with its share of the SA256-to-SE-RRM gap.
  R-AB-4      SELECTOR per arm from its k32 scan: CLEAN at a spurious rate <= 1 %, else DIRTY.
  R-A2-5      THE STRUCTURE READING at 16 and at 64 (the registered question): TWO-STATE-CARRIES (SA256U BELOW-BEYOND), NO-STRUCTURE-EFFECT
              (INSIDE), SINGLE-STATE-BETTER (ABOVE-BEYOND).
  R-A2-6      WHERE (descriptive; no label moves): from each arm's D64 per-iteration bits on the identical 5,000, exact at iterations 1, 2, 4, 8,
              16, 32, 64 with the paired difference and McNemar p at each; the quartiles of the first exact iteration among the puzzles exact at
              64; the late share = exact at 64 and not at 16.
  R-A2-7      PERSISTENCE (descriptive): per arm, LOST = exact at some iteration <= 64 and not at 64; DROPPED = any exact -> not-exact step within
              the 64 (a later recovery included); the paired McNemar on DROPPED.
  CONSISTENCY (descriptive): each arm's D64 bits at iteration 16 against its own D16 row (the same fixed start): the agreement count.
  STABILITY   descriptive: the selected grid (EDGE when it is the arm's last grid), validation max/end, scan fixed vs one random start, verified.

  .venv/bin/python tools/analyze_attr2.py --root <stage>/runs --out <dir>      (the stage holds SA256U AND the ladder's SA256 dirs)
  .venv/bin/python tools/analyze_attr2.py --selftest
"""
from __future__ import annotations
import argparse, json, sys, tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_paperfinal as PF   # Root, jload, paired, spurious, metrics, say, pp
import analyze_wladder as WL      # row, acc_on, restrict
import analyze_sablate as AB      # same, dict_diff, cfg_of, label1, label2 (the recipe ablation's frozen helpers, imported unchanged)

FLOOR16, FLOOR64 = AB.FLOOR16, AB.FLOOR64
SERRM16, SERRM64 = AB.SERRM16, AB.SERRM64
REF, SPUR_CLEAN, T64 = "SA256", 0.01, 64
IGNORE, NEUTRAL = {"out", "resume"}, {"remat"}
CURVE = (1, 2, 4, 8, 16, 32, 64)
ARMS = {   # arm -> (what it moves, its budget, the registered argv differences from SA256, the registered model-config differences)
    "SA256U": ("SE-RRM's single-state recurrence (= TRM's single-z): one carry, 21 applications per segment, the input injected at each", 30000,
               dict(dec_single_state=True), dict(dec_single_state=True)),
}
STRUCT = {"BELOW-BEYOND": "TWO-STATE-CARRIES", "INSIDE": "NO-STRUCTURE-EFFECT", "ABOVE-BEYOND": "SINGLE-STATE-BETTER"}

def bits64(recs):
    """(idx, (n, 64) bool with column k-1 = exact after iteration k) from a D64 record; None without the per-iteration bits."""
    if recs is None or "exact_by_step" not in recs: return None
    eb = np.asarray(recs["exact_by_step"]).astype(np.uint8)
    if eb.ndim != 2 or eb.shape[1] * 8 < T64: return None
    return np.asarray(recs["idx"]), np.unpackbits(eb, axis=1, bitorder="little")[:, :T64].astype(bool)

def on_ids(idx, arr, ids):
    _, pa, _ = np.intersect1d(idx, ids, return_indices=True)
    return idx[pa], arr[pa]

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
        if t == 64 and bits64(r) is None: bad.append(f"{REF} t64: no per-iteration bits")
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
            if t == 64 and bits64(r) is None: bad.append(f"{a} t64: no per-iteration bits")
        sc = root.summ(root.scan32(a))
        if sc: grids.add(str(sc.get("ckpt")))
        if len(grids) > 1: bad.append(f"{a} rows on {len(grids)} grids {sorted(grids)}")
        if vb and grids and vb not in next(iter(grids)): bad.append(f"{a} rows not on the selected grid {vb}")
    return bad, notes, ids0

def where_and_persistence(root, arms, ids, J):
    """R-A2-6, R-A2-7 and CONSISTENCY (descriptive) from the D64 per-iteration bits, all on the identical ids."""
    B = {}
    for a in [REF] + arms:
        b = bits64(WL.row(root, a, 64)[1])
        if b is not None and ids is not None: B[a] = on_ids(b[0], b[1], ids)
    for a in arms:
        if a not in B or REF not in B or len(np.intersect1d(B[a][0], B[REF][0])) == 0: J[f"R-A2-6 WHERE {a}"] = J[f"R-A2-7 PERSISTENCE {a}"] = "NO-DATA"; continue
        parts = []
        for k in CURVE:
            q = PF.paired(dict(idx=B[a][0], e=B[a][1][:, k - 1]), dict(idx=B[REF][0], e=B[REF][1][:, k - 1]), key="e")
            parts.append(f"@{k} {100 * B[a][1][:, k - 1].mean():.2f} vs {100 * B[REF][1][:, k - 1].mean():.2f} ({100 * q['diff']:+.2f} pp, p {q['p']:.2g})")
        J[f"R-A2-6 WHERE {a}"] = "; ".join(parts); PF.say(f"R-A2-6 WHERE {a} (exact {a} vs {REF}): {J[f'R-A2-6 WHERE {a}']}")
        for x in (a, REF):
            b = B[x][1]; fe = np.where(b.any(axis=1), b.argmax(axis=1) + 1, -1); s64 = b[:, -1]
            qs = np.percentile(fe[s64], [25, 50, 75]) if s64.any() else None
            J[f"R-A2-6 FIRST-EXACT {x}"] = ("n/a" if qs is None else f"quartiles {qs[0]:.0f} / {qs[1]:.0f} / {qs[2]:.0f} among {int(s64.sum())} exact at 64; "
                                           f"late share (exact at 64, not at 16) {100 * (s64 & ~b[:, 15]).mean():.2f} %")
            PF.say(f"R-A2-6 FIRST-EXACT {x:7s} {J[f'R-A2-6 FIRST-EXACT {x}']}")
        drop = {}
        for x in (a, REF):
            b = B[x][1]; lost = b.any(axis=1) & ~b[:, -1]; drop[x] = (b[:, :-1] & ~b[:, 1:]).any(axis=1)
            J[f"R-A2-7 {x}"] = f"LOST {int(lost.sum())}, DROPPED {int(drop[x].sum())} (recovered by 64: {int((drop[x] & b[:, -1]).sum())}) of {len(b)}"
            PF.say(f"R-A2-7 PERSISTENCE {x:7s} {J[f'R-A2-7 {x}']}")
        q = PF.paired(dict(idx=B[a][0], e=drop[a]), dict(idx=B[REF][0], e=drop[REF]), key="e")
        J[f"R-A2-7 PERSISTENCE {a}"] = f"DROPPED only-{a} {q['only_a']}, only-{REF} {q['only_b']} (p {q['p']:.2g})"; PF.say(f"R-A2-7 PERSISTENCE {a} paired: {J[f'R-A2-7 PERSISTENCE {a}']}")
    for x in [REF] + arms:
        r16 = WL.row(root, x, 16)[1]
        if x not in B or r16 is None or len(B[x][0]) == 0: J[f"CONSISTENCY {x}"] = "NO-DATA"; continue
        i16, e16 = on_ids(np.asarray(r16["idx"]), np.asarray(r16["cold_exact"]).astype(bool), ids)
        common, pa, pb = np.intersect1d(B[x][0], i16, return_indices=True)
        agree = int((B[x][1][pa, 15] == e16[pb]).sum())
        J[f"CONSISTENCY {x}"] = f"D64 bits at 16 agree with the D16 row on {agree} of {len(common)}"; PF.say(f"CONSISTENCY {x:7s}      {J[f'CONSISTENCY {x}']}")

def analyze(root_path):
    root = PF.Root(root_path); J = {}
    arms = [a for a in ARMS if root.summ(root.chain(a, "sub5k_vsel_t64")) or root.summ(root.chain(a, "sub5k_vsel_t16")) or (root.pdir(a) / "config.json").exists()]
    PF.say(f"SE-RRM ATTRIBUTION ROUND 2 — analyzer (frozen rules; one seed; floors {100*FLOOR16:.2f} pp at 16 and {100*FLOOR64:.2f} pp at 64; the reference = the ladder's {REF})")
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
    located = {16: [], 64: []}
    for a in arms:
        for t, fl, se in ((16, FLOOR16, SERRM16), (64, FLOOR64, SERRM64)):
            q = PF.paired(A[a][t], A[REF][t]); k = f"R-AB-1 {a} vs {REF} @{t}"
            lab = AB.label1(q["diff"], fl) if q else "n/a"
            J[k] = "NO-DATA" if not q else f"{lab} ({100*q['diff']:+.2f} pp, only-{a} {q['only_a']}, only-{REF} {q['only_b']}, n {q['n']}, p {q['p']:.2g})"; PF.say(f"{k:31s} {J[k]}")
            k5 = f"R-A2-5 STRUCTURE @{t}"; J[k5] = "NO-DATA" if not q else f"{STRUCT[lab]} ({lab}, {100*q['diff']:+.2f} pp)"; PF.say(f"{k5:31s} {J[k5]}")
            if q and lab == "BELOW-BEYOND":
                gap = (A[REF][f"acc{t}"] or 0) - se; located[t].append(f"{a} ({100*q['diff']:+.2f} pp" + (f" = {100 * -q['diff'] / gap:.0f} % of the {100*gap:.2f} pp to SE-RRM" if gap > fl else "") + ")")
            acc = A[a][f"acc{t}"]; k2 = f"R-AB-2 {a} vs SE-RRM @{t}"
            J[k2] = "NO-DATA" if acc is None else f"{AB.label2(acc, se, fl)} ({PF.pp(acc)} vs the published {100*se:.2f}; {100*(acc - se):+.2f} pp; different evaluation sets, their one run)"; PF.say(f"{k2:31s} {J[k2]}")
    for t in (16, 64):
        have = [a for a in arms if A[a][f"acc{t}"] is not None]; k = f"R-AB-3 LOCATED @{t}"
        J[k] = "NO-DATA" if not have else (("LOCATED: " + "; ".join(located[t])) if located[t] else f"NOT-LOCATED (no arm of {have} reads below {REF} beyond the floor)")
        PF.say(f"{k:31s} {J[k]}")
    for a in [REF] + arms:
        z = root.recs(root.scan32(a)); sp = PF.spurious(z) if z is not None else None; k = f"R-AB-4 SELECTOR {a}"
        J[k] = "NO-DATA" if sp is None else f"{'CLEAN' if sp <= SPUR_CLEAN else 'DIRTY'} (spurious k32 {100*sp:.2f} %)"; PF.say(f"{k:31s} {J[k]}")
    where_and_persistence(root, arms, ids, J)
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

def _traj(n, a16, a64, rng, lost=0):
    """(n, 64) trajectories, exact from a first iteration onward: a16 of the puzzles first exact at 1..16, a64 by 64; then `lost` puzzles
    never exact at 64 made exact at iterations 20..40 only (so the 16- and 64-iteration rows stay as built)."""
    b = np.zeros((n, T64), bool); o = rng.permutation(n); n16, n64 = int(round(a16 * n)), int(round(a64 * n))
    first = np.full(n, -1); first[o[:n16]] = rng.integers(1, 17, n16); first[o[n16:n64]] = rng.integers(17, 65, n64 - n16)
    for i in np.flatnonzero(first > 0): b[i, first[i] - 1:] = True
    b[o[n64:n64 + lost], 19:40] = True
    return b

def _mk(tmp, accs, av_over=None, cf_over=None, ext=None, sel=None, steps_end=None, ids_shift=None, lost=None, nobits=None, d16_flip=None):
    rng = np.random.default_rng(0); ids = np.arange(5000) * 3
    for a, (a16, a64) in accs.items():
        budget = ARMS[a][1] if a in ARMS else 30000; vb = (sel or {}).get(a, "028000")
        b = _traj(len(ids), a16, a64, rng, lost=(lost or {}).get(a, 0)); fe = np.where(b.any(axis=1), b.argmax(axis=1), -1)
        e16 = b[:, 15].copy()
        if d16_flip == a: e16[:7] = ~e16[:7]
        for t, ex in ((16, e16), (64, b[:, -1])):
            d = tmp / f"sxeval_pchamp{a}" / f"sub5k_vsel_t{t}"; d.mkdir(parents=True, exist_ok=True); ii = ids + 1 if (ids_shift == a and t == 64) else ids
            rec = dict(idx=ii, cold_exact=ex, first_exact=fe if t == 64 else np.where(b[:, :16].any(axis=1), b[:, :16].argmax(axis=1), -1))
            if t == 64 and nobits != a: rec["exact_by_step"] = np.packbits(b.astype(np.uint8), axis=1, bitorder="little")
            np.savez(d / "records_all.npz", **rec)
            (d / "summary_all.json").write_text(json.dumps(dict(n=len(ii), exact_acc=float(ex.mean()), t_total=t, ema=True, subsample=5000, ckpt=f"runs/pretrainchamp_{a}/ckpt_{vb}.pkl")))
        av, cf = dict(BASE_AV, out=f"runs/pretrainchamp_{a}"), dict(BASE_CF)
        if a in ARMS: av.update(ARMS[a][2]); cf.update(ARMS[a][3])
        for k, v in (av_over or {}).get(a, {}).items(): av.pop(k) if v is None else av.__setitem__(k, v)
        for k, v in (cf_over or {}).get(a, {}).items(): cf.pop(k) if v is None else cf.__setitem__(k, v)
        p = tmp / f"pretrainchamp_{a}"; p.mkdir(parents=True, exist_ok=True); (p / "config.json").write_text(json.dumps(dict(config=cf, argv=av))); (p / "val_best.txt").write_text(f"{vb} 0.9600 {int(vb)}")
        (p / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in [dict(monitor=dict(step=22000, val_t16_ema=0.96)), dict(step=(steps_end or {}).get(a, budget), loss=0.5), dict(monitor=dict(step=budget, val_t16_ema=0.95))]))
        if ext == a: (p / "EXTENDED.txt").write_text("x")
        s = tmp / f"sxscan_pchamp{a}"; s.mkdir(parents=True, exist_ok=True); (s / "summary_all.json").write_text(json.dumps(dict(ckpt=f"runs/pretrainchamp_{a}/ckpt_{vb}.pkl", exact_acc=a64, b1_exact=a64 - 0.002, exact_acc_vote=0.998, n=5000)))
        n, k = 400, 8; ex = rng.random((n, k)) < 0.9; res = np.where(ex, 0.001, 0.5)
        np.savez(s / "records_all.npz", idx=np.arange(n), mi_exact_k=ex, mi_resid_k=res)

def selftest():
    n = 0
    base = {"SA256": (.982, .996), "SA256U": (.940, .989)}
    with tempfile.TemporaryDirectory() as t_:   # the single state falls beyond the floor at 16, inside at 64; 30 puzzles lost in the arm
        t = Path(t_); _mk(t, base, lost={"SA256U": 30}); PF.LINES.clear(); J = analyze(t)
        assert J["INTEGRITY"] == "PASS", J["INTEGRITY"]; n += 1
        assert J["R-AB-1 SA256U vs SA256 @16"].startswith("BELOW-BEYOND (-4.20 pp") and J["R-AB-1 SA256U vs SA256 @64"].startswith("INSIDE (-0.70 pp"), J["R-AB-1 SA256U vs SA256 @16"]; n += 1
        assert J["R-A2-5 STRUCTURE @16"].startswith("TWO-STATE-CARRIES (BELOW-BEYOND, -4.20 pp)") and J["R-A2-5 STRUCTURE @64"].startswith("NO-STRUCTURE-EFFECT (INSIDE"), (J["R-A2-5 STRUCTURE @16"], J["R-A2-5 STRUCTURE @64"]); n += 1
        assert J["R-AB-3 LOCATED @16"].startswith("LOCATED: SA256U (-4.20 pp = 94 % of the 4.47 pp to SE-RRM)") and J["R-AB-3 LOCATED @64"].startswith("NOT-LOCATED"), J["R-AB-3 LOCATED @16"]; n += 1
        assert J["R-AB-2 SA256U vs SE-RRM @16"].startswith("WITHIN (94.00 vs the published 93.73") and J["R-AB-4 SELECTOR SA256U"].startswith("CLEAN"); n += 1
        assert "@16 94.00 vs 98.20 (-4.20 pp" in J["R-A2-6 WHERE SA256U"] and "@64 98.90 vs 99.60 (-0.70 pp" in J["R-A2-6 WHERE SA256U"], J["R-A2-6 WHERE SA256U"]; n += 1
        assert J["R-A2-7 SA256U"].startswith("LOST 30, DROPPED 30 (recovered by 64: 0) of 5000") and J["R-A2-7 SA256"].startswith("LOST 0, DROPPED 0"), (J["R-A2-7 SA256U"], J["R-A2-7 SA256"]); n += 1
        assert J["R-A2-7 PERSISTENCE SA256U"].startswith("DROPPED only-SA256U 30, only-SA256 0") and J["CONSISTENCY SA256U"] == "D64 bits at 16 agree with the D16 row on 5000 of 5000"; n += 1
        assert "late share (exact at 64, not at 16) 4.90 %" in J["R-A2-6 FIRST-EXACT SA256U"] and "among 4945 exact at 64" in J["R-A2-6 FIRST-EXACT SA256U"], J["R-A2-6 FIRST-EXACT SA256U"]; n += 1
    with tempfile.TemporaryDirectory() as t_:   # inside at both depths -> NO-STRUCTURE-EFFECT, NOT-LOCATED; a D16 row that disagrees on 7 is reported
        t = Path(t_); _mk(t, {"SA256": (.982, .996), "SA256U": (.975, .995)}, d16_flip="SA256U"); PF.LINES.clear(); J = analyze(t)
        assert J["R-A2-5 STRUCTURE @16"].startswith("NO-STRUCTURE-EFFECT") and J["R-AB-3 LOCATED @16"].startswith("NOT-LOCATED") and J["INTEGRITY"] == "PASS"; n += 1
        assert J["CONSISTENCY SA256U"] == "D64 bits at 16 agree with the D16 row on 4993 of 5000" and J["CONSISTENCY SA256"].endswith("5000 of 5000"), J["CONSISTENCY SA256U"]; n += 1
    with tempfile.TemporaryDirectory() as t_:   # the label's other side (a low reference, so a rise can clear the floor)
        t = Path(t_); _mk(t, {"SA256": (.930, .985), "SA256U": (.970, .995)}); PF.LINES.clear(); J = analyze(t)
        assert J["R-A2-5 STRUCTURE @16"].startswith("SINGLE-STATE-BETTER (ABOVE-BEYOND, +4.00 pp)") and J["R-A2-5 STRUCTURE @64"].startswith("NO-STRUCTURE-EFFECT"), J["R-A2-5 STRUCTURE @16"]; n += 1
    for kw, frag in ((dict(av_over={"SA256U": dict(dec_single_state=None)}, cf_over={"SA256U": dict(dec_single_state=None)}), "SA256U argv differs from SA256's in [], registered ['dec_single_state']"),
                     (dict(cf_over={"SA256U": dict(dec_single_state=None)}), "SA256U model config differs from SA256's in [], registered ['dec_single_state']"),
                     (dict(cf_over={"SA256U": dict(dec_single_state=False)}), "SA256U model config dec_single_state = False != the registered True"),
                     (dict(av_over={"SA256U": dict(lr=5e-4)}), "SA256U argv differs from SA256's in ['dec_single_state', 'lr']"),
                     (dict(av_over={"SA256U": dict(steps=36000)}), "SA256U argv differs from SA256's in ['dec_single_state', 'steps']"),
                     (dict(cf_over={"SA256U": dict(trm_lambda=0.0)}), "SA256U model config differs from SA256's in ['dec_single_state', 'trm_lambda']"),
                     (dict(ext="SA256U"), "SA256U carries an extension marker"), (dict(steps_end={"SA256U": 22000}), "SA256U last training step 22000 != 30000"),
                     (dict(ids_shift="SA256U"), "SA256U t64: puzzle ids differ from SA256's"), (dict(av_over={"SA256U": dict(seed=1)}), "SA256U argv differs from SA256's in ['dec_single_state', 'seed']"),
                     (dict(nobits="SA256U"), "SA256U t64: no per-iteration bits"), (dict(nobits="SA256"), "SA256 t64: no per-iteration bits"),
                     (dict(sel={"SA256U": "030000"}, steps_end={"SA256U": 30000}), None)):
        with tempfile.TemporaryDirectory() as t_:
            t = Path(t_); _mk(t, base, **kw); PF.LINES.clear(); J = analyze(t)
            if frag is None: assert J["INTEGRITY"] == "PASS" and "selected 030000 EDGE" in J["STABILITY SA256U"], J["STABILITY SA256U"]; n += 1; continue
            assert J["INTEGRITY"].startswith("FAIL") and frag in J["INTEGRITY"], (frag, J["INTEGRITY"]); n += 1
    with tempfile.TemporaryDirectory() as t_:   # missing per-iteration bits: the descriptive readings say NO-DATA, the labels still print
        t = Path(t_); _mk(t, base, nobits="SA256U"); PF.LINES.clear(); J = analyze(t)
        assert J["R-A2-6 WHERE SA256U"] == "NO-DATA" and J["CONSISTENCY SA256U"] == "NO-DATA" and J["R-A2-5 STRUCTURE @16"].startswith("TWO-STATE-CARRIES"); n += 1
    with tempfile.TemporaryDirectory() as t_:   # the reference absent -> a loud failure
        t = Path(t_); _mk(t, {"SA256U": base["SA256U"]}); PF.LINES.clear(); J = analyze(t)
        assert J["INTEGRITY"].startswith("FAIL") and "SA256 (the reference) config.json absent" in J["INTEGRITY"] and J["R-AB-1 SA256U vs SA256 @16"] == "NO-DATA" and J["R-A2-5 STRUCTURE @16"] == "NO-DATA"; n += 1
    with tempfile.TemporaryDirectory() as t_:
        PF.LINES.clear(); assert analyze(Path(t_)) == {"INTEGRITY": "NO-DATA"}; n += 1
    print(f"selftest OK: {n}/{n}")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true"); ap.add_argument("--root"); ap.add_argument("--out"); a = ap.parse_args()
    if a.selftest: return selftest()
    J = analyze(a.root)
    if a.out:
        o = Path(a.out); o.mkdir(parents=True, exist_ok=True); (o / "attr2_verdict.txt").write_text("\n".join(PF.LINES) + "\n"); (o / "attr2_verdict.json").write_text(json.dumps(J, indent=1))

if __name__ == "__main__":
    main()
