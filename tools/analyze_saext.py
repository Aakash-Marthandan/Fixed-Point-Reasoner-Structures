#!/usr/bin/env python3
"""THE ATTENTION ARMS TO THE PAPER'S BUDGET — analyzer (Plan_2026-09-19_SA_Extension.md; FROZEN at the registration commit, before any row exists).

SA128 / SA192 / SA256 (SE-RRM's two mixers inside our block, loop and recipe; the width ladder's arms) RESUMED from their banked 30k states to
50,000 steps (an exact continuation: the constant learning rate), selected over all 25 grids with the registered rule, and read with the champion
battery (part A) and the four filler rows the paper's width-192 seeds carry (part B, when present). One seed per arm; floors 2.58 / 2.44 pp.

RULES (frozen; the plan's §3):
  INTEGRITY  per arm: the run's argv = the banked argv in every key but steps (30,000 -> 50,000) and the banked config = the ladder's own; the
             EXTENDED record 30,000 -> 50,000; 30,000 in the resume record; the monitor rows = the 25 grids 2,000..50,000 (a repeated step only
             with identical values) and the first 15 = the ladder's rows unchanged; the last training step 50,000; the selection re-derived with
             the registered rule (EMA val, then raw val, then the earliest step, among banked grids) = val_best.txt; every row on the selected
             grid (the final row on the last grid; the raw row without EMA) with its registered size / depth / restarts / EMA flag; the rows
             on 5,000 puzzles on the ladder's 5,000 ids; the D16 full row on all 422,786 distinct ids.
  R-SE-1     the extension's effect, paired on the ladder's 5,000: the selected 50k-budget grid vs the ladder's selected 30k grid at 16 (the
             full D16 row restricted) and at 64 (the full D64 filler row restricted; without it, the 100k row's overlap): INSIDE /
             ABOVE-BEYOND / BELOW-BEYOND; SAME-GRID when the selection is the ladder's own grid.
  R-SE-2     where the selection lands: LADDER-GRID (<= 30,000) / INTERIOR (32,000-48,000) / EDGE (50,000: a lower bound).
  R-SE-3     against the paper's width-192 triple on all 422,786 (95.406 / 99.050; C5 / C7 / C8 at 46k): ABOVE / INSIDE / BELOW by the floor
             (at 64 on the full row; without part B the 100k row, labelled).
  R-SE-4     against SE-RRM's published numbers on their set: 16 vs 93.73 (main text) and 95.4 (appendix Table A6); 64 vs 98.22:
             ABOVE / WITHIN / BELOW by the floor.
  R-SE-5     SELECTOR: the k32 scan (and the k128 with part B): CLEAN at a spurious rate <= 1 %, else DIRTY.
  R-SE-6     the width order at the paper's budget, paired: SA192 vs SA256 and SA128 vs SA192 at 16 (all 422,786) and 64 (the full row, or
             the 100k row): NARROWER-AHEAD / NARROWER-BEHIND, INSIDE / BEYOND.
  DESCRIPTIVE the depth rows, the final and raw rows, the census, the calibration, the screens; notes (extra resumes, non-finite loss rows).

  .venv/bin/python tools/analyze_saext.py --root <stage>/runs --ladder runs/_wladder_pull/x_final/runs --out <dir>
  .venv/bin/python tools/analyze_saext.py --selftest
"""
from __future__ import annotations
import argparse, json, math, sys, tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_paperfinal as PF   # paired (exact McNemar on identical ids), spurious, say, pp, LINES

FLOOR16, FLOOR64 = 0.0258, 0.0244
W192 = {"C5": (0.95906, 0.99155), "C7": (0.94823, 0.98817), "C8": (0.95487, 0.99177)}   # SURVIVING_CLAIMS_2026-09-19 §1: all 422,786, step 46,000
W192_16 = sum(v[0] for v in W192.values()) / 3; W192_64 = sum(v[1] for v in W192.values()) / 3
SERRM16, SERRM16_A6, SERRM64 = 0.9373, 0.954, 0.9822
ARMS = ("SA128", "SA192", "SA256")
FROM, TO, GRID = 30000, 50000, 2000
N_FULL, N_FIN, N_D64, N_SUB, N_5K, N_50K = 422786, 50000, 100000, 20000, 5000, 50000
LADDER_RESUMES = {"SA192": [25500]}
PART_B = {"d64full": ("filler_sxeval_pchamp{a}_full_t64", N_FULL, 64, 0), "d128sub": ("filler_sxeval_pchamp{a}_sub50000_t128", N_50K, 128, 0),
          "d256sub": ("filler_sxeval_pchamp{a}_sub50000_t256", N_50K, 256, 0), "k128": ("filler_sxscan128_pchamp{a}", N_5K, 64, 128)}


def jload(p):
    p = Path(p); return json.loads(p.read_text()) if p.exists() else None


def recs(d):
    q = Path(d) / "records_all.npz"
    return dict(np.load(q, allow_pickle=True)) if q.exists() else None


def monitor_rows(path):
    """[(step, val_t16_ema, val_t16)] in file order (every row, repeats kept)."""
    out = []
    if not Path(path).exists(): return out
    for l in Path(path).read_text().splitlines():
        try: r = json.loads(l)
        except Exception: continue
        m = r.get("monitor")
        if isinstance(m, dict) and "val_t16_ema" in m:
            out.append((int(m["step"]), float(m["val_t16_ema"]), float(m.get("val_t16", 0.0))))
    return out


def loss_rows(path):
    out = []
    if not Path(path).exists(): return out
    for l in Path(path).read_text().splitlines():
        try: r = json.loads(l)
        except Exception: continue
        if "loss" in r and "step" in r: out.append((int(r["step"]), float(r["loss"])))
    return out


def select(rows, banked):
    """the registered rule (tools/select_ckpt.py --key val_t16_ema --tie earliest --second-key val_t16), re-derived independently."""
    cand = [(v, v2, -s, s) for s, v, v2 in rows if s in banked]
    return max(cand)[3] if cand else None


def label_pair(d, floor): return "n/a" if d is None else ("INSIDE" if abs(d) <= floor else ("ABOVE-BEYOND" if d > 0 else "BELOW-BEYOND"))
def label_ext(acc, ref, floor): return "n/a" if acc is None else ("WITHIN" if abs(acc - ref) <= floor else ("ABOVE" if acc > ref else "BELOW"))
def label_tri(acc, ref, floor): return "n/a" if acc is None else ("INSIDE" if abs(acc - ref) <= floor else ("ABOVE" if acc > ref else "BELOW"))


def position(step):
    if step is None: return "n/a"
    return "LADDER-GRID" if step <= FROM else ("EDGE" if step >= TO else "INTERIOR")


def restrict(r, ids):
    if r is None or ids is None: return None
    _, pa, _ = np.intersect1d(np.asarray(r["idx"]), ids, return_indices=True)
    return dict(idx=np.asarray(r["idx"])[pa], cold_exact=np.asarray(r["cold_exact"])[pa])


def acc(r): return None if r is None or not len(r["idx"]) else float(np.asarray(r["cold_exact"]).astype(bool).mean())


def rows_spec(a, vb, part_b):
    """(label, dir, n, t_total, k_init, ema, ckpt) of every registered row of arm a on the selected grid vb."""
    g = f"runs/pretrainchamp_{a}/ckpt_{vb:06d}.pkl"; last = f"runs/pretrainchamp_{a}/ckpt_latest.pkl"
    # the final row: ckpt_latest on the 50k subsample ALWAYS (chain_champ.sh copies the vsel row only when the selection fell back to
    # ckpt_latest, i.e. no grid had a monitor row; a selected 50,000 grid resolves to ckpt_050000.pkl, so the final row still runs)
    S = [("D16 full", f"sxeval_pchamp{a}/full_vsel_t16", N_FULL, 16, 0, True, g),
         ("final", f"sxeval_pchamp{a}/full_final_t16", N_FIN, 16, 0, True, last),
         ("raw", f"sxeval_pchamp{a}/full_vsel_t16_alt", N_FIN, 16, 0, False, g),
         ("D64 100k", f"sxeval_pchamp{a}/full_vsel_t64", N_D64, 64, 0, True, g),
         ("D128 20k", f"sxeval_pchamp{a}/sub20k_t128", N_SUB, 128, 0, True, g),
         ("D256 5k", f"sxeval_pchamp{a}/sub5k_t256", N_5K, 256, 0, True, g),
         ("k32 scan", f"sxscan_pchamp{a}", N_5K, 64, 32, True, g)]
    if part_b:
        S += [(j, d.format(a=a), n, t, k, True, g) for j, (d, n, t, k) in PART_B.items()]
    return S


def integrity(root, ladder, a, part_b):
    bad, notes = [], []; P = root / f"pretrainchamp_{a}"
    cfg, cb, lc = jload(P / "config.json"), jload(P / "config_banked.json"), jload(ladder / f"pretrainchamp_{a}" / "config.json")
    av = (cfg or {}).get("argv"); bv = (cb or {}).get("argv"); lv = (lc or {}).get("argv")
    if not (isinstance(av, dict) and isinstance(bv, dict) and isinstance(lv, dict)):
        bad.append(f"{a} config.json / config_banked.json / the ladder's config.json absent or without an argv dict"); return bad, notes, None
    diff = sorted(k for k in set(av) | set(bv) if av.get(k, "<absent>") != bv.get(k, "<absent>"))
    if diff != ["steps"]: bad.append(f"{a} the run's argv differs from the banked argv in {diff} (registered: ['steps'])")
    if av.get("steps") != TO or bv.get("steps") != FROM: bad.append(f"{a} steps: run {av.get('steps')} banked {bv.get('steps')} (registered {TO} / {FROM})")
    if bv != lv: bad.append(f"{a} the banked config is not the ladder's own (differs in {sorted(k for k in set(bv) | set(lv) if bv.get(k) != lv.get(k))})")
    ext = (P / "EXTENDED.txt").read_text() if (P / "EXTENDED.txt").exists() else ""
    if f"from {FROM} to {TO}" not in ext: bad.append(f"{a} EXTENDED record absent or not {FROM} -> {TO}")
    res = [int(x) for x in (P / "resumes.txt").read_text().split()] if (P / "resumes.txt").exists() else []
    if FROM not in res: bad.append(f"{a} the resume record lacks {FROM}")
    extra = [r for r in res if r != FROM and r not in LADDER_RESUMES.get(a, [])]
    if extra: notes.append(f"{a} resumes besides the join: {extra} (chain recycles; the SOT carry reset there, labelled)")
    M = monitor_rows(P / "metrics.jsonl"); steps = sorted({s for s, _, _ in M})
    if steps != list(range(GRID, TO + 1, GRID)): bad.append(f"{a} monitor steps {steps[:2]}..{steps[-2:]} n={len(steps)} (registered: the 25 grids {GRID}..{TO})")
    byst = {}
    for s, v, v2 in M: byst.setdefault(s, set()).add((v, v2))
    rep = [s for s, vs in byst.items() if len(vs) > 1]
    if rep: bad.append(f"{a} a repeated monitor step with DIFFERENT values: {rep[:4]}")
    if len(M) != len(steps): notes.append(f"{a} {len(M) - len(steps)} repeated monitor row(s) with identical values (a recycle re-logged them)")
    LM = {s: (v, v2) for s, v, v2 in monitor_rows(ladder / f"pretrainchamp_{a}" / "metrics.jsonl")}
    chg = [s for s in range(GRID, FROM + 1, GRID) if s not in LM or {LM[s]} != byst.get(s)]
    if chg: bad.append(f"{a} the first {FROM // GRID} monitor rows are not the ladder's unchanged (steps {chg[:4]})")
    L = loss_rows(P / "metrics.jsonl")
    if not L or max(s for s, _ in L) != TO: bad.append(f"{a} last training step {max((s for s, _ in L), default=None)} != {TO}")
    nf = sum(1 for _, x in L if not math.isfinite(x))
    if nf: notes.append(f"{a} {nf} non-finite loss row(s)")
    vbt = (P / "val_best.txt").read_text().split() if (P / "val_best.txt").exists() else []
    vb = int(vbt[0]) if vbt and vbt[0].isdigit() else None
    mine = select(M, set(range(GRID, TO + 1, GRID)))
    if vb is None or mine != vb: bad.append(f"{a} the selection re-derived ({mine}) != val_best.txt ({vbt[:1]})")
    if vb is None: return bad, notes, None
    lid = None; l5 = recs(ladder / f"sxeval_pchamp{a}" / "sub5k_vsel_t16")
    if l5 is not None: lid = np.sort(np.asarray(l5["idx"]))
    else: bad.append(f"{a} the ladder's 5k row absent")
    for lab, d, n, t, k, ema, ck in rows_spec(a, vb, part_b):
        s = jload(root / d / "summary_all.json")
        if not s: bad.append(f"{a} {lab} row absent ({d})"); continue
        got = (s.get("n"), s.get("t_total"), int(s.get("k_init") or 0), bool(s.get("ema")), s.get("ckpt"))
        if got != (n, t, k, ema, ck): bad.append(f"{a} {lab}: n/t/k/ema/ckpt {got} != {(n, t, k, ema, ck)}")
    sc = recs(root / f"sxscan_pchamp{a}")
    if sc is not None and lid is not None and (len(sc["idx"]) != len(lid) or (np.sort(np.asarray(sc["idx"])) != lid).any()): bad.append(f"{a} the k32 scan's ids are not the ladder's 5,000")
    f16 = recs(root / f"sxeval_pchamp{a}" / "full_vsel_t16")
    if f16 is not None and len(np.unique(np.asarray(f16["idx"]))) != N_FULL: bad.append(f"{a} the D16 full row covers {len(np.unique(np.asarray(f16['idx'])))} distinct ids, not {N_FULL}")
    return bad, notes, vb


def analyze(root_path, ladder_path):
    root, ladder = Path(root_path), Path(ladder_path); J = {}
    arms = [a for a in ARMS if (root / f"pretrainchamp_{a}" / "config.json").exists()]
    PF.say(f"THE ATTENTION ARMS TO THE PAPER'S BUDGET — analyzer (frozen rules; one seed per arm; floors {100*FLOOR16:.2f} pp at 16, {100*FLOOR64:.2f} pp at 64)")
    PF.say(f"arms present: {arms or 'none'}")
    if not arms: PF.say("NO-DATA"); return {"INTEGRITY": "NO-DATA"}
    part_b = {a: all((root / d.format(a=a) / "summary_all.json").exists() for d, *_ in PART_B.values()) for a in arms}
    PF.say("part B (the four filler rows): " + ", ".join(f"{a} {'present' if part_b[a] else 'ABSENT'}" for a in arms))
    bad, notes, VB = [], [], {}
    for a in arms:
        b, n, vb = integrity(root, ladder, a, part_b[a]); bad += b; notes += n; VB[a] = vb
    J["INTEGRITY"] = ("PASS" if not bad else "FAIL: " + "; ".join(bad)) + (f" [notes: {'; '.join(notes)}]" if notes else "")
    PF.say(f"INTEGRITY                {J['INTEGRITY']}")
    A = {}
    for a in arms:
        f16 = recs(root / f"sxeval_pchamp{a}" / "full_vsel_t16")
        d64 = recs(root / f"filler_sxeval_pchamp{a}_full_t64") if part_b[a] else recs(root / f"sxeval_pchamp{a}" / "full_vsel_t64")
        A[a] = dict(r16=f16, r64=d64, acc16=acc(f16), acc64=acc(d64), n64=None if d64 is None else len(d64["idx"]), full64=part_b[a])
        PF.say(f"  {a:6s} selected {VB[a]}  16: {PF.pp(A[a]['acc16'])} (n {0 if f16 is None else len(f16['idx'])})   64: {PF.pp(A[a]['acc64'])} (n {A[a]['n64']}{'' if part_b[a] else ', the 100k row'})")
    for a in arms:
        l16, l64 = recs(ladder / f"sxeval_pchamp{a}" / "sub5k_vsel_t16"), recs(ladder / f"sxeval_pchamp{a}" / "sub5k_vsel_t64")
        lvb = (ladder / f"pretrainchamp_{a}" / "val_best.txt"); lvb = int(lvb.read_text().split()[0]) if lvb.exists() else None
        for t, lr, nr, fl in ((16, l16, A[a]["r16"], FLOOR16), (64, l64, A[a]["r64"], FLOOR64)):
            ids = None if lr is None else np.asarray(lr["idx"]); q = PF.paired(restrict(nr, ids), lr) if (nr is not None and lr is not None) else None
            same = VB[a] is not None and lvb is not None and VB[a] == lvb
            k = f"R-SE-1 {a} @{t}"
            J[k] = "NO-DATA" if not q else (f"{'SAME-GRID ' if same else ''}{label_pair(q['diff'], fl)} ({100*q['diff']:+.2f} pp, only-new {q['only_a']}, only-ladder {q['only_b']}, n {q['n']}, p {q['p']:.2g}; ladder grid {lvb}, new grid {VB[a]})")
            PF.say(f"{k:26s} {J[k]}")
        J[f"R-SE-2 {a}"] = f"{position(VB[a])} (selected {VB[a]})"; PF.say(f"{'R-SE-2 ' + a:26s} {J[f'R-SE-2 {a}']}")
    for a in arms:
        for t, x, ref, fl in ((16, A[a]["acc16"], W192_16, FLOOR16), (64, A[a]["acc64"], W192_64, FLOOR64)):
            k = f"R-SE-3 {a} @{t}"; lab = "" if (t == 16 or A[a]["full64"]) else " [the 100k row, not the full set]"
            J[k] = "NO-DATA" if x is None else f"{label_tri(x, ref, fl)} ({PF.pp(x)} vs the width-192 triple {100*ref:.2f}; {100*(x - ref):+.2f} pp){lab}"; PF.say(f"{k:26s} {J[k]}")
        for t, x, ref, fl, nm in ((16, A[a]["acc16"], SERRM16, FLOOR16, "93.73 (main text)"), (16, A[a]["acc16"], SERRM16_A6, FLOOR16, "95.4 (appendix Table A6)"), (64, A[a]["acc64"], SERRM64, FLOOR64, "98.22")):
            k = f"R-SE-4 {a} @{t} vs {nm.split()[0]}"; lab = "" if (t == 16 or A[a]["full64"]) else " [the 100k row, not their set]"
            J[k] = "NO-DATA" if x is None else f"{label_ext(x, ref, fl)} ({PF.pp(x)} vs SE-RRM's published {nm}; {100*(x - ref):+.2f} pp; their one run){lab}"; PF.say(f"{k:26s} {J[k]}")
    for a in arms:
        for tag, d in (("k32", f"sxscan_pchamp{a}"), ("k128", f"filler_sxscan128_pchamp{a}")):
            z = recs(root / d); sp = PF.spurious(z) if z is not None else None; k = f"R-SE-5 {a} {tag}"
            if tag == "k128" and not part_b[a]: continue
            J[k] = "NO-DATA" if sp is None else f"{'CLEAN' if sp <= 0.01 else 'DIRTY'} (spurious {100*sp:.2f} %)"; PF.say(f"{k:26s} {J[k]}")
    for na, wi in (("SA192", "SA256"), ("SA128", "SA192")):
        if na not in A or wi not in A: continue
        for t, fl in ((16, FLOOR16), (64, FLOOR64)):
            q = PF.paired(A[na][f"r{t}"], A[wi][f"r{t}"]); k = f"R-SE-6 {na} vs {wi} @{t}"
            J[k] = "NO-DATA" if not q else f"{'NARROWER-AHEAD' if q['diff'] > 0 else 'NARROWER-BEHIND'} / {'INSIDE' if abs(q['diff']) <= fl else 'BEYOND'} ({100*q['diff']:+.2f} pp, p {q['p']:.2g}, n {q['n']})"
            PF.say(f"{k:26s} {J[k]}")
    for a in arms:   # DESCRIPTIVE
        g = lambda d: jload(root / d / "summary_all.json") or {}
        ce = jload(root / f"sxcalib_pchamp{a}_vsel" / "calib.json") or {}
        J[f"DESCRIPTIVE {a}"] = (f"D128 20k {PF.pp(g(f'sxeval_pchamp{a}/sub20k_t128').get('exact_acc'))}, D256 5k {PF.pp(g(f'sxeval_pchamp{a}/sub5k_t256').get('exact_acc'))}"
                                 + (f", D128 50k {PF.pp(g(f'filler_sxeval_pchamp{a}_sub50000_t128').get('exact_acc'))}, D256 50k {PF.pp(g(f'filler_sxeval_pchamp{a}_sub50000_t256').get('exact_acc'))}" if part_b[a] else "")
                                 + f"; final {PF.pp(g(f'sxeval_pchamp{a}/full_final_t16').get('exact_acc'))}, raw {PF.pp(g(f'sxeval_pchamp{a}/full_vsel_t16_alt').get('exact_acc'))}; calibration top-k on stalled {ce.get('topk_correct_stalled')}")
        PF.say(f"DESCRIPTIVE {a:6s}       {J[f'DESCRIPTIVE {a}']}")
    return J


# ---------------- selftest (hand-built stages; the printed labels asserted; mutants must fail) ----------------
def _row(d, idx, ex, n=None, t=16, k=0, ema=True, ck="", extra=None):
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "records_all.npz", idx=idx, cold_exact=ex, **(extra or {}))
    (d / "summary_all.json").write_text(json.dumps(dict(n=int(len(idx) if n is None else n), t_total=t, k_init=k, ema=ema, ckpt=ck, exact_acc=float(np.mean(ex)))))


def _mk(tmp, sel=None, acc16=None, acc64=None, part_b=True, argv_over=None, banked_over=None, ladder_over=None, ext_txt=None, resumes=None,
        mon_change=None, mon_dup_diff=False, last_step=TO, vb_file=None, ck_over=None, scan_ids_shift=False, spur=None, trunc_loss=None, ema_over=None):
    root, lad = tmp / "stage", tmp / "ladder"; rng = np.random.default_rng(1)
    ids5 = np.arange(N_5K) * 7; idsF = np.arange(N_FULL); ids100 = np.arange(N_D64) * 4
    for a in ARMS:
        base = dict(seed=0, steps=FROM, batch=768, lr=1e-4, dec_width=int(a[2:]), dp=True, out=f"runs/pretrainchamp_{a}")
        lv = dict(base); lv.update((ladder_over or {}).get(a, {}))
        bv = dict(lv); bv.update((banked_over or {}).get(a, {}))
        av = dict(bv, steps=TO); av.update((argv_over or {}).get(a, {}))
        LP = lad / f"pretrainchamp_{a}"; LP.mkdir(parents=True, exist_ok=True); (LP / "config.json").write_text(json.dumps({"argv": lv}))
        vals = {s: 0.90 + 0.0001 * (s // GRID) for s in range(GRID, TO + 1, GRID)}
        top = (sel or {}).get(a, 40000); vals[top] = 0.99
        lsel = max((s for s in range(GRID, FROM + 1, GRID)), key=lambda s: (vals[s], -s))
        (LP / "metrics.jsonl").write_text("\n".join(json.dumps({"monitor": {"step": s, "val_t16_ema": vals[s], "val_t16": vals[s] - .01}}) for s in range(GRID, FROM + 1, GRID)))
        (LP / "val_best.txt").write_text(f"{lsel:06d} {vals[lsel]:.4f} {lsel}")
        P = root / f"pretrainchamp_{a}"; P.mkdir(parents=True, exist_ok=True)
        (P / "config.json").write_text(json.dumps({"argv": av})); (P / "config_banked.json").write_text(json.dumps({"argv": bv}))
        (P / "EXTENDED.txt").write_text(ext_txt or f"EXTENDED from {FROM} to {TO} (registered)")
        (P / "resumes.txt").write_text("\n".join(str(x) for x in (resumes or {}).get(a, LADDER_RESUMES.get(a, []) + [FROM])) + "\n")
        mrows = []
        for s in range(GRID, TO + 1, GRID):
            v = vals[s] + ((mon_change or {}).get(a, {}).get(s, 0.0))
            mrows.append(json.dumps({"monitor": {"step": s, "val_t16_ema": v, "val_t16": v - .01}}))
            if not (trunc_loss and a == "SA192" and s > trunc_loss): mrows.append(json.dumps({"step": s, "loss": 0.5}))
        if mon_dup_diff and a == "SA256": mrows.append(json.dumps({"monitor": {"step": 36000, "val_t16_ema": 0.5, "val_t16": 0.4}}))
        if not (trunc_loss and a == "SA192"): mrows.append(json.dumps({"step": last_step, "loss": 0.5}))
        (P / "metrics.jsonl").write_text("\n".join(mrows))
        (P / "val_best.txt").write_text(vb_file.get(a) if (vb_file and a in vb_file) else f"{top:06d} {vals[top]:.4f} {top}")
        G_, L_ = f"runs/pretrainchamp_{a}/ckpt_{top:06d}.pkl", f"runs/pretrainchamp_{a}/ckpt_latest.pkl"
        TABLE = [("D16 full", f"sxeval_pchamp{a}/full_vsel_t16", 422786, 16, 0, True, G_), ("final", f"sxeval_pchamp{a}/full_final_t16", 50000, 16, 0, True, L_),
                 ("raw", f"sxeval_pchamp{a}/full_vsel_t16_alt", 50000, 16, 0, False, G_), ("D64 100k", f"sxeval_pchamp{a}/full_vsel_t64", 100000, 64, 0, True, G_),
                 ("D128 20k", f"sxeval_pchamp{a}/sub20k_t128", 20000, 128, 0, True, G_), ("D256 5k", f"sxeval_pchamp{a}/sub5k_t256", 5000, 256, 0, True, G_),
                 ("k32 scan", f"sxscan_pchamp{a}", 5000, 64, 32, True, G_)]
        if part_b:
            TABLE += [("d64full", f"filler_sxeval_pchamp{a}_full_t64", 422786, 64, 0, True, G_), ("d128sub", f"filler_sxeval_pchamp{a}_sub50000_t128", 50000, 128, 0, True, G_),
                      ("d256sub", f"filler_sxeval_pchamp{a}_sub50000_t256", 50000, 256, 0, True, G_), ("k128", f"filler_sxscan128_pchamp{a}", 5000, 64, 128, True, G_)]
        for lab, d, n, t, k, ema, ck in TABLE:
            ck = (ck_over or {}).get((a, lab), ck); ema = (ema_over or {}).get((a, lab), ema)
            p16 = (acc16 or {}).get(a, .97); p64 = (acc64 or {}).get(a, .996)
            if lab == "D16 full": _row(root / d, idsF, rng.random(N_FULL) < p16, n, t, k, ema, ck)
            elif lab == "d64full": _row(root / d, idsF, rng.random(N_FULL) < p64, n, t, k, ema, ck)
            elif lab == "D64 100k": _row(root / d, ids100, rng.random(N_D64) < p64, n, t, k, ema, ck)
            elif lab in ("k32 scan", "k128"):
                kk = k; ex = rng.random((N_5K, kk)) < .97; res = np.where(ex, 1e-3, .5); f = (spur or {}).get(a, 0.0); res[(~ex) & (rng.random((N_5K, kk)) < f)] = 1e-4
                _row(root / d, ids5 + (1 if (scan_ids_shift and a == "SA128" and lab == "k32 scan") else 0), ex[:, 0], n, t, k, ema, ck, extra=dict(mi_exact_k=ex, mi_resid_k=res))
            else: _row(root / d, np.arange(n), rng.random(n) < p16, n, t, k, ema, ck)
        # the ladder's 5k rows: on a same-grid selection the very bits of the new rows (the same checkpoint); otherwise fresh draws at the same rate
        p16 = (acc16 or {}).get(a, .97); p64 = (acc64 or {}).get(a, .996)
        n16 = recs(root / f"sxeval_pchamp{a}" / "full_vsel_t16"); n64 = recs(root / (f"filler_sxeval_pchamp{a}_full_t64" if part_b else f"sxeval_pchamp{a}/full_vsel_t64"))
        same = top == lsel
        b16 = np.asarray(n16["cold_exact"])[ids5] if same else rng.random(N_5K) < p16
        b64 = np.asarray(n64["cold_exact"])[ids5] if (same and part_b) else rng.random(N_5K) < p64
        _row(lad / f"sxeval_pchamp{a}" / "sub5k_vsel_t16", ids5, b16, t=16); _row(lad / f"sxeval_pchamp{a}" / "sub5k_vsel_t64", ids5, b64, t=64)
    return root, lad


def selftest():
    n = 0
    assert select([(2000, .5, .4), (4000, .6, .5), (6000, .6, .5), (8000, .6, .55)], {2000, 4000, 6000, 8000}) == 8000; n += 1       # the second key breaks the tie
    assert select([(2000, .5, .4), (4000, .6, .5), (6000, .6, .5)], {2000, 4000, 6000}) == 4000 and select([(2000, .9, .9)], {4000}) is None; n += 1   # earliest tie; banked only
    assert position(28000) == "LADDER-GRID" and position(30000) == "LADDER-GRID" and position(32000) == "INTERIOR" and position(50000) == "EDGE"; n += 1
    assert label_pair(-0.03, FLOOR16) == "BELOW-BEYOND" and label_pair(0.02, FLOOR16) == "INSIDE" and label_ext(0.99, SERRM16, FLOOR16) == "ABOVE" and label_tri(0.95, W192_16, FLOOR16) == "INSIDE"; n += 1
    assert abs(W192_16 - 0.954053) < 1e-6 and abs(W192_64 - 0.990497) < 1e-6; n += 1
    with tempfile.TemporaryDirectory() as t_:   # the clean case: every letter as built
        r, l = _mk(Path(t_), sel={"SA128": 50000, "SA192": 44000, "SA256": 28000}, acc16={"SA128": .955, "SA192": .975, "SA256": .985}, spur={"SA192": .6})
        PF.LINES.clear(); J = analyze(r, l)
        assert J["INTEGRITY"] == "PASS", J["INTEGRITY"]; n += 1
        assert J["R-SE-2 SA128"].startswith("EDGE") and J["R-SE-2 SA192"].startswith("INTERIOR") and J["R-SE-2 SA256"].startswith("LADDER-GRID"); n += 1
        assert J["R-SE-1 SA256 @16"].startswith("SAME-GRID INSIDE (+0.00 pp, only-new 0, only-ladder 0") and J["R-SE-1 SA192 @16"].startswith("INSIDE") and not J["R-SE-1 SA192 @16"].startswith("SAME-GRID"), (J["R-SE-1 SA256 @16"], J["R-SE-1 SA192 @16"]); n += 1
        assert J["R-SE-3 SA256 @16"].startswith("ABOVE") and J["R-SE-3 SA128 @16"].startswith("INSIDE") and J["R-SE-3 SA192 @64"].startswith("INSIDE"), (J["R-SE-3 SA256 @16"], J["R-SE-3 SA128 @16"]); n += 1
        assert J["R-SE-4 SA256 @16 vs 93.73"].startswith("ABOVE") and J["R-SE-4 SA128 @16 vs 95.4"].startswith("WITHIN") and J["R-SE-4 SA128 @64 vs 98.22"].startswith("WITHIN"), J["R-SE-4 SA128 @16 vs 95.4"]; n += 1
        assert J["R-SE-5 SA192 k32"].startswith("DIRTY") and J["R-SE-5 SA256 k32"].startswith("CLEAN") and "R-SE-5 SA256 k128" in J; n += 1
        assert J["R-SE-6 SA192 vs SA256 @16"].startswith("NARROWER-BEHIND / INSIDE") and J["R-SE-6 SA128 vs SA192 @16"].startswith("NARROWER-BEHIND / INSIDE"), J["R-SE-6 SA192 vs SA256 @16"]; n += 1
    with tempfile.TemporaryDirectory() as t_:   # without part B: 64 on the 100k row, labelled; no k128 letter
        r, l = _mk(Path(t_), part_b=False); PF.LINES.clear(); J = analyze(r, l)
        assert J["INTEGRITY"] == "PASS" and "the 100k row" in J["R-SE-3 SA256 @64"] and "R-SE-5 SA256 k128" not in J, J["INTEGRITY"]; n += 1
    for kw, frag in ((dict(argv_over={"SA192": dict(lr=2e-4)}), "SA192 the run's argv differs from the banked argv in ['lr', 'steps']"),
                     (dict(banked_over={"SA128": dict(dp=False)}), "SA128 the banked config is not the ladder's own"),
                     (dict(ladder_over={"SA256": dict(batch=272)}, banked_over={"SA256": dict(batch=768)}), "SA256 the banked config is not the ladder's own"),
                     (dict(ext_txt="EXTENDED from 30000 to 60000"), "EXTENDED record absent or not 30000 -> 50000"),
                     (dict(resumes={"SA256": [25000]}), "SA256 the resume record lacks 30000"),
                     (dict(mon_change={"SA128": {12000: 0.01}}), "SA128 the first 15 monitor rows are not the ladder's unchanged (steps [12000]"),
                     (dict(mon_dup_diff=True), "SA256 a repeated monitor step with DIFFERENT values: [36000]"),
                     (dict(last_step=48000), "last training step 50000"),     # a row at 50000 exists with the monitor: the max stays 50000 -> must NOT fire
                     (dict(trunc_loss=48000), "SA192 last training step 48000 != 50000"),
                     (dict(vb_file={"SA192": "030000 0.9 30000"}), "SA192 the selection re-derived (40000) != val_best.txt (['030000'])"),
                     (dict(ck_over={("SA256", "D64 100k"): "runs/pretrainchamp_SA256/ckpt_038000.pkl"}), "SA256 D64 100k: n/t/k/ema/ckpt"),
                     (dict(ck_over={("SA128", "raw"): "runs/pretrainchamp_SA128/ckpt_latest.pkl"}), "SA128 raw: n/t/k/ema/ckpt"),
                     (dict(scan_ids_shift=True), "SA128 the k32 scan's ids are not the ladder's 5,000"),
                     (dict(ema_over={("SA192", "raw"): True}), "SA192 raw: n/t/k/ema/ckpt"),
                     (dict(sel={"SA128": 50000}, ck_over={("SA128", "final"): "runs/pretrainchamp_SA128/ckpt_050000.pkl"}), "SA128 final: n/t/k/ema/ckpt")):
        with tempfile.TemporaryDirectory() as t_:
            r, l = _mk(Path(t_), **kw); PF.LINES.clear(); J = analyze(r, l)
            if frag == "last training step 50000": assert J["INTEGRITY"] == "PASS", J["INTEGRITY"]; n += 1; continue
            assert J["INTEGRITY"].startswith("FAIL") and frag in J["INTEGRITY"], (frag, J["INTEGRITY"]); n += 1
    with tempfile.TemporaryDirectory() as t_:   # a repeated monitor row with IDENTICAL values (a recycle) is a note, not a failure; an extra resume is a note
        r, l = _mk(Path(t_), resumes={"SA256": [FROM, 41500]}); P = r / "pretrainchamp_SA256"
        m = (P / "metrics.jsonl").read_text() + "\n" + json.dumps({"monitor": {"step": 42000, "val_t16_ema": 0.90 + 0.0001 * 21, "val_t16": 0.90 + 0.0001 * 21 - .01}}); (P / "metrics.jsonl").write_text(m)
        PF.LINES.clear(); J = analyze(r, l)
        assert J["INTEGRITY"].startswith("PASS [notes:") and "resumes besides the join: [41500]" in J["INTEGRITY"] and "1 repeated monitor row(s) with identical values" in J["INTEGRITY"], J["INTEGRITY"]; n += 1
    with tempfile.TemporaryDirectory() as t_:
        PF.LINES.clear(); assert analyze(Path(t_), Path(t_)) == {"INTEGRITY": "NO-DATA"}; n += 1
    print(f"selftest OK: {n}/{n}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true"); ap.add_argument("--root"); ap.add_argument("--ladder"); ap.add_argument("--out"); a = ap.parse_args()
    if a.selftest: return selftest()
    J = analyze(a.root, a.ladder)
    if a.out:
        o = Path(a.out); o.mkdir(parents=True, exist_ok=True); (o / "saext_verdict.txt").write_text("\n".join(PF.LINES) + "\n"); (o / "saext_verdict.json").write_text(json.dumps(J, indent=1))


if __name__ == "__main__":
    main()
