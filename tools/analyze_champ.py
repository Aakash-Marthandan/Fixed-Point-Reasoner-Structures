# Ledger: CHAMPION NIGHT ANALYZER — written BEFORE any champion-night run (2026-09-08; Plan_2026-09-08_Champion_Night.md
# §2 arms / §3 rules / §4 predictions); the launch registration locks these rules verbatim. TAG champ. ARMS (one variable
# each; every arm the DEC at w384 unless noted, FPA k1 + RI sigma 1, no digit aug, 30k steps unless noted): C0/C1/C2 = the
# champion recipe at seeds 0/1/2 (the claim-bearing triple). C3 = C0 + the ONLINE position orbit (a fresh group element per
# row per step; 50k steps). C4 = C0 with the DEC coupling "attn" (set attention over the other eight fields per cell, 4 heads
# x dk 32). C5 = C0 at w192 (the accuracy-per-MAC point, 1.34x the TRM-MLP's compute). C6 = C0 + the calibrated commit head
# (tau .9, BCE weight .1; 50k steps).
#   FILES (under runs/, env QHRRN_RUNS overrides): pretrainchamp_<arm>/{metrics.jsonl, val_best.txt ("NNNNNN val step" or
#     "FALLBACK-FINAL"), STOPPED.txt?, EXTENDED.txt? ("EXTENDED from 30000 to 50000 (peak at NNNNN)"), config.json (argv,
#     n_params_bulk)}; sxeval_pchamp<arm>/<row>/summary_all.json for full_vsel_t16 (n 422786) · full_final_t16 (422786 when
#     vsel == final and copied, else 50000) · full_vsel_t16_alt (50000) · full_vsel_t64 (100000) · sub20k_t128 (20000) ·
#     sub5k_t256 (5000) · C6 only sub20k_t16_commit (20000; commit_frac_last, commit_wrong_among_committed_last,
#     commit_wrong_among_committed_unsolved_last, commit_auc_last, commit_tau); every row: n, exact_acc, ckpt, t_total, ema,
#     exact_by_step_curve, depth_regressions. sxscan_pchamp<arm>/{summary_all.json (n 5000, k_init 32, t_total 64,
#     subsample_seed 20260822, ckpt, b1_exact, vote_at_k, t1r_at_k, exact_acc), records_all.npz (idx, mi_exact_k, mi_resid_k,
#     cold_exact)}. sxscreen_pchamp<arm>_<tag> for tag in {s010000, s020000, s030000, s040000, vb} (n 512, exact_acc,
#     exact_acc_vote, uv_vote_k256?). sxcensus_pchamp<arm>_{vsel,final}/census.json; sxcalib_pchamp<arm>_vsel/calib.json.
#     The sync-policy A/B rider runs/analysis/champ_sync_ab.json {wall_default_s, wall_per_step_s, rows, k, zone} (optional).
#   RULES (letters only; every number from the files; a missing arm's rows read NO-DATA and never crash):
#   INTEGRITY (a breach on a PRESENT arm WITHHOLDS the verdict, printed as "INTEGRITY: FAIL (reasons)"; missing rows are
#     NO-DATA, not a breach): full_vsel_t16 n == 422786; full_final_t16 n in {422786, 50000}; full_vsel_t16_alt n == 50000;
#     full_vsel_t64 n == 100000; scan n == 5000, k_init 32, t_total 64, subsample_seed 20260822, unique idx, identical idx
#     sets across every present scan; sub20k_t128 n == 20000; sub5k_t256 n == 5000; sub20k_t16_commit n == 20000; every vsel-labeled eval of an arm
#     (full_vsel_t16, full_vsel_t16_alt, full_vsel_t64, sub20k_t128, sub5k_t256, the scan, the vb screen, sub20k_t16_commit)
#     reports ONE checkpoint path (ckpt), and that path's step equals the step in val_best.txt.
#   CLEAN split: STABILITY := not STOPPED AND census(vsel, t64) <= .02; MEMORIZATION (labeled, never disqualifying) :=
#     end segment-CE < .02 (mean ce_in over the last 5 loss rows) OR (cold16 vsel - cold16 final) > .05.
#   NOISE FLOOR: FLOOR_C := max(max - min of cold16 over the STABLE members of {C0, C1, C2}, .005); fewer than 2 stable
#     members -> FLOOR_C := .015 labeled FLOOR-DEFAULT. A contrast is READ only beyond 2 x FLOOR_C; else FLAT. Descriptive
#     beside it: FLOOR_SCREEN := max - min of the vb screens' exact_acc over the same members.
#   R-CH-0 SEEDS: SPREAD-C := max - min of cold16 over the stable seed arms: <= .015 -> TIGHT; <= .03 -> LOOSE; else WIDE.
#     Fewer than 2 -> NO-DATA.
#   R-CH-1 ORBIT (C3 vs the seed mean M := mean cold16 of the stable seed arms): d = cold16(C3) - M: > 2 FLOOR_C ->
#     ORBIT-LIFTS; < -2 FLOOR_C -> ORBIT-HURTS; else ORBIT-FLAT. Tag +DELAYS iff C3's memorization onset (the first logged
#     step with ce_in < .05; "never" if none) is >= 10000 steps later than the mean onset of the stable seed arms, or is
#     "never" while theirs exist; else +SAME-ONSET.
#   R-CH-2 OPERATOR (C4 vs M): d > 2 FLOOR_C -> ATTENTION-HELPS; d < -2 FLOOR_C -> ATTENTION-HURTS; else MEAN-SUFFICES.
#   R-CH-3 WIDTH-DOWN (C5): cold64(C5) >= .920 -> PARITY-AT-1.3x; >= .874 -> TRM-CLASS-AT-1.3x; else BELOW-AT-1.3x.
#   R-CH-4 COMMIT (C6): w := commit_wrong_among_committed_unsolved_last from sub20k_t16_commit; d := cold16(C6) - M.
#     w < .30 AND d > 2 FLOOR_C -> CALIBRATED-AND-LIFTS; w < .30 AND |d| <= 2 FLOOR_C -> CALIBRATED-INERT; w < .30 AND
#     d < -2 FLOOR_C -> CALIBRATED-AT-COST; w >= .30 -> UNCALIBRATED (+LIFTS / +FLAT / +COST by d). The commit AUC and the
#     committed fraction are reported.
#   R-CH-5 SELECTOR: every present arm's SPURIOUS <= .01 -> SELECTOR-CLEAN; else SELECTOR-DIRTY: <list of arms>. Report per
#     arm spurious, t1r@32, verified@32, t1r/verified. SPURIOUS(arm) exactly as analyze_finalA.py computes it from the scan
#     records (wrong draws whose residual lies below the median residual of the correct draws), at k = 32 for every arm
#     (every champion arm is a wide DEC-class arm).
#   R-CH-6 DEPTH: per arm the ladder cold16 (full) / cold64 (100k, labeled) / D128 (20k, labeled) / D256 (5k, labeled) and
#     the per-row depth_regressions; DEPTH-MONOTONE iff every present row has depth_regressions <= .0005 x n, else
#     DEPTH-REGRESSES: <arms>.
#   R-CH-7 PARITY: cold64 >= .920 on every stable seed arm -> PARITY-x<k> where k = the number of stable seed arms (all
#     pass); else PARITY-PARTIAL <passing>/<k>. HEADLINE := mean and (max - min) of cold64 and of D128 over the stable seed arms.
#   R-CH-EXT (descriptive, the registered extension rule's audit): per arm whether EXTENDED.txt exists, the selected step
#     (val_best), the arm's step budget (from config.json argv "steps"), and whether the selected step lies inside the last
#     4000 steps of the FINAL budget (a peak still rising at the end).
#   CHAMPION BY RULE: CHAMP := the stable seed arm with the highest cold64 (memorized allowed, labeled); LABELED-BEST := the
#     highest cold64 over every present arm. Print "R-CH-CHAMP: <arm> cold64 <value> (labeled best: <arm> <value>)".
#   SYNC A/B (descriptive): if champ_sync_ab.json exists print the two walls and the ratio wall_per_step / wall_default;
#     predicted >= 1.5.
#   PREDICTIONS scoreboard (HIT / MISS-ABOVE / MISS-BELOW / n/a): C0..C2 cold16 in [.935, .965] each; SPREAD-C in [0, .015];
#     cold64 in [.970, .985]; D128 in [.975, .990]; D256 in [.978, .992]; spurious in [0, .001] each seed arm; verified@32
#     >= .995; C3 cold16 in [.950, .975]; C4 d in [-.015, .015]; C5 cold16 in [.86, .93] and cold64 in [.91, .96]; C6 w in
#     [0, .30] and d in [-.01, .01]; per-arm depth gain D64->D128 in [.003, .008]; sync ratio >= 1.5; number of extended arms
#     in [0, 2].
#   READER CONVENTIONS (not rules): ties in CHAMP / LABELED-BEST break by arm order (the lower seed first); "inside the last
#     4000 steps" = selected step > final budget - 4000, the final budget = EXTENDED.txt's "to" when it exists, else argv
#     steps; FALLBACK-FINAL in val_best.txt means vsel := the final step; a val_best.txt absent while evals exist skips the
#     step check (NO-DATA, labeled in the audit); a C3 without metrics.jsonl reads +ONSET-NO-DATA rather than "never".
"""
  .venv/bin/python tools/analyze_champ.py            # -> runs/analysis/champ_verdict.{txt,json}
  .venv/bin/python tools/analyze_champ.py --selftest
"""
from __future__ import annotations
import contextlib, io, json, os, re, shutil, sys, tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(os.environ.get("QHRRN_RUNS", ROOT / "runs"))
TAG = "champ"
SEED_ARMS = ["C0", "C1", "C2"]; TREAT_ARMS = ["C3", "C4", "C5", "C6"]; ARMS = SEED_ARMS + TREAT_ARMS
N_FULL, N_FIN, N_ALT, N_D64, N_D128, N_D256, N_COMMIT = 422786, {422786, 50000}, 50000, 100000, 20000, 5000, 20000
N_SCAN, K_SCAN = 5000, 32
SCAN_PROTO = dict(t_total=64, k_init=K_SCAN, subsample_seed=20260822)
PARITY_D64, TRM_MLP, FLOOR_DEFAULT, FLOOR_MIN, END_WINDOW, ONSET_CE, DELAY_STEPS = 0.920, 0.874, 0.015, 0.005, 4000, 0.05, 10000
VSEL_ROWS = ("full_vsel_t16", "full_vsel_t16_alt", "full_vsel_t64", "sub20k_t128", "sub5k_t256", "sub20k_t16_commit")
DEPTH_ROWS = (("full_vsel_t16", 16), ("full_vsel_t64", 64), ("sub20k_t128", 128), ("sub5k_t256", 256))
INF = float("inf")
PRED = dict(cold16=(.935, .965), spread=(0.0, .015), cold64=(.970, .985), d128=(.975, .990), d256=(.978, .992), spur=(0.0, .001), v32=(.995, 1.0),
            c3_cold16=(.950, .975), c4_d=(-.015, .015), c5_cold16=(.86, .93), c5_cold64=(.91, .96), c6_w=(0.0, .30), c6_d=(-.01, .01),
            gain=(.003, .008), sync=(1.5, INF), n_ext=(0, 2))
DESC = {"C0": "champion recipe seed 0 (DEC-w384 FPA k1 RI s1)", "C1": "champion recipe seed 1", "C2": "champion recipe seed 2",
        "C3": "C0 + online position orbit (50k)", "C4": "C0 + attn coupling (4h x dk32)", "C5": "C0 at w192 (1.34x TRM-MLP MACs)",
        "C6": "C0 + calibrated commit head tau .9 (50k)"}
LINES = []
def say(s=""): LINES.append(str(s)); print(s)
def jload(p):
    p = Path(p); return json.loads(p.read_text()) if p.exists() else None
def fpp(x): return "  -   " if x is None else f"{100*x:6.2f}"
def f3(x): return "  -  " if x is None else f"{x:.3f}"
def f4(x): return "-" if x is None else f"{x:.4f}"
def fd(x): return "-" if x is None else f"{x:+.4f}"
def fi(x): return "-" if x is None else str(int(x))
def hm(x, band): return "n/a" if x is None else ("HIT" if band[0] <= x <= band[1] else ("MISS-ABOVE" if x > band[1] else "MISS-BELOW"))

# ---------- readers (one per row; every reader returns None on a missing file) ----------
def pdir(a): return RUNS / f"pretrain{TAG}_{a}"
def evdir(a): return RUNS / f"sxeval_p{TAG}{a}"
def row(a, name): return jload(evdir(a) / name / "summary_all.json")
def acc(a, name): s = row(a, name); return None if not s else s.get("exact_acc")
def cold16(a): return acc(a, "full_vsel_t16")
def fcold(a): return acc(a, "full_final_t16")
def cold64(a): return acc(a, "full_vsel_t64")
def d128(a): return acc(a, "sub20k_t128")
def d256(a): return acc(a, "sub5k_t256")
def commit(a): return row(a, "sub20k_t16_commit")
def scan(a): return jload(RUNS / f"sxscan_p{TAG}{a}" / "summary_all.json")
def scan_recs(a):
    q = RUNS / f"sxscan_p{TAG}{a}" / "records_all.npz"; return dict(np.load(q, allow_pickle=True)) if q.exists() else None
def b1(a): s = scan(a); return None if not s else s.get("b1_exact")
def verified32(a): s = scan(a); return None if not s else (s.get("vote_at_k") or {}).get(str(K_SCAN))
def t1r32(a): s = scan(a); return None if not s else (s.get("t1r_at_k") or {}).get(str(K_SCAN))
def screen(a, tag): return jload(RUNS / f"sxscreen_p{TAG}{a}_{tag}" / "summary_all.json")
def vb_screen(a): s = screen(a, "vb"); return None if not s else s.get("exact_acc")
def census(a, which="vsel", t=64):
    c = jload(RUNS / f"sxcensus_p{TAG}{a}_{which}" / "census.json")
    if not c: return None
    rows = [r for r in c.get("rows", []) if int(r["t"]) == t]; return None if not rows else rows[0]["exploded_frac"]
def calib(a): return jload(RUNS / f"sxcalib_p{TAG}{a}_vsel" / "calib.json")
def metrics(a):
    p = pdir(a) / "metrics.jsonl"
    if not p.exists(): return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def end_ce(a):
    rows = [r for r in metrics(a) if "loss" in r]
    if not rows: return None
    return float(np.mean([r.get("ce_in", np.nan) for r in rows[-5:]]))
def onset(a):
    """The memorization onset: the first logged step with ce_in < .05 (None = never)."""
    for r in metrics(a):
        if "loss" in r and r.get("ce_in") is not None and float(r["ce_in"]) < ONSET_CE: return int(r["step"])
    return None
def stopped(a):
    p = pdir(a) / "STOPPED.txt"; return p.read_text().strip().splitlines()[0][:44] if p.exists() else None
def budget(a):
    cfg = jload(pdir(a) / "config.json"); argv = (cfg or {}).get("argv")
    if isinstance(argv, dict) and argv.get("steps") is not None: return int(argv["steps"])
    if isinstance(argv, list) and "--steps" in argv: return int(argv[argv.index("--steps") + 1])
    return None
def extended(a):
    p = pdir(a) / "EXTENDED.txt"
    if not p.exists(): return None
    m = re.search(r"from\s+(\d+)\s+to\s+(\d+)(?:.*?peak at\s+(\d+))?", p.read_text())
    return dict(frm=int(m.group(1)), to=int(m.group(2)), peak=int(m.group(3)) if m.group(3) else None) if m else dict(frm=None, to=None, peak=None)
def final_budget(a):
    e = extended(a); return e["to"] if e and e.get("to") else budget(a)
def val_best(a):
    p = pdir(a) / "val_best.txt"
    if not p.exists(): return None
    txt = p.read_text().strip(); line = txt.splitlines()[0].strip() if txt else ""
    if line.startswith("FALLBACK-FINAL"): return dict(step=final_budget(a), label="FALLBACK-FINAL")
    tok = line.split()[0].split(":")[-1] if line else ""
    return dict(step=int(tok) if tok.isdigit() else None, label="val")
def ckpt_step(path):
    m = re.search(r"ckpt_(\d+)\.pkl", str(path)); return int(m.group(1)) if m else None
def ckpt_paths(a):
    srcs = {}
    for nm in VSEL_ROWS:
        s = row(a, nm)
        if s and s.get("ckpt"): srcs[nm] = s["ckpt"]
    for nm, s in (("scan", scan(a)), ("screen_vb", screen(a, "vb"))):
        if s and s.get("ckpt"): srcs[nm] = s["ckpt"]
    return srcs
def spurious(a, recs=None):
    """Lens G / E5 (analyze_finalA.py, verbatim): the per-draw spurious-attractor rate — wrong draws whose residual is
    below the median residual of the correct draws (None without the scan's per-draw records)."""
    z = recs if recs is not None else scan_recs(a)
    if z is None or "mi_exact_k" not in z or "mi_resid_k" not in z: return None
    ex = np.asarray(z["mi_exact_k"]).astype(bool); rs = np.asarray(z["mi_resid_k"]).astype(np.float64); fin = np.isfinite(rs)
    ef, wf = ex & fin, (~ex) & fin
    if ef.sum() == 0 or wf.sum() == 0: return None
    return float(np.mean(rs[wf] <= np.median(rs[ef])))
def regressions(a):
    out = {}
    for name, _t in DEPTH_ROWS:
        s = row(a, name)
        if s and s.get("depth_regressions") is not None and s.get("n"): out[name] = (int(s["depth_regressions"]), int(s["n"]))
    return out
def stable(a):
    if stopped(a): return False
    c = census(a); return c is not None and c <= .02
def memorized(a):
    ce = end_ce(a); vc, fc = cold16(a), fcold(a); tags = []
    if ce is not None and ce < .02: tags.append("END-CE")
    if vc is not None and fc is not None and vc - fc > .05: tags.append("VSEL-FINAL-DROP")
    return tags

# ---------- integrity ----------
def integrity():
    errs = []; idx_sets = {}
    for a in ARMS:
        for name, ns in (("full_vsel_t16", {N_FULL}), ("full_final_t16", N_FIN), ("full_vsel_t16_alt", {N_ALT}), ("full_vsel_t64", {N_D64}),
                         ("sub20k_t128", {N_D128}), ("sub5k_t256", {N_D256}), ("sub20k_t16_commit", {N_COMMIT})):
            s = row(a, name)
            if s and s.get("n") not in ns: errs.append(f"{a} {name} n={s.get('n')} not in {sorted(ns)}")
        s = scan(a)
        if s:
            if s.get("n") != N_SCAN: errs.append(f"{a} scan n={s.get('n')}!={N_SCAN}")
            for k, v in SCAN_PROTO.items():
                if s.get(k) != v: errs.append(f"{a} scan {k}={s.get(k)}!={v}")
            z = scan_recs(a)
            if z is not None and "idx" in z:
                if len(np.unique(z["idx"])) != len(z["idx"]): errs.append(f"{a} scan dup idx")
                idx_sets[a] = frozenset(np.asarray(z["idx"]).tolist())
        paths = set(ckpt_paths(a).values())
        if len(paths) > 1: errs.append(f"{a} vsel evals on {len(paths)} different grids: {sorted(paths)}")
        elif paths:
            vb = val_best(a); st = ckpt_step(next(iter(paths)))
            if vb and vb["step"] is not None and st != vb["step"]: errs.append(f"{a} vsel evals on step {st} but val_best.txt selects {vb['step']}")
    if len(set(idx_sets.values())) > 1: errs.append(f"scan idx sets differ across arms {sorted(idx_sets)}")
    return errs

# ---------- verdict ----------
def analyze():
    V = {}
    say(f"== CHAMPION NIGHT VERDICT ({TAG}; rules locked pre-data in this file's header) ==")
    errs = integrity(); V["INTEGRITY"] = "PASS" if not errs else "FAIL"; V["INTEGRITY_REASONS"] = errs
    say(f"INTEGRITY: {V['INTEGRITY']}" + ("" if not errs else " (" + "; ".join(errs) + ")"))
    say(f"\nARM TABLE (headline = EMA single pass, cold16 on all 422,786; cold64 on 100k, D128 on 20k, D256 on 5k, labeled; scan 5k x k32 t64; "
        f"spurious = lens-E5 per-draw rate; onset = first step with ce_in < {ONSET_CE}; refs: parity D64 {PARITY_D64:.3f}, TRM-MLP {TRM_MLP:.3f})")
    say("  arm | description                                  | cold16 | final16 | cold64 |  D128  |  D256  |   b1   | t1r@32 |  v@32  | spur  | census | end-CE | onset  | stable  | memorized      | vb-scr")
    rows = {}
    for a in ARMS:
        ms = metrics(a)
        r = rows[a] = dict(cold16=cold16(a), fcold=fcold(a), cold64=cold64(a), d128=d128(a), d256=d256(a), b1=b1(a), t1r=t1r32(a), v32=verified32(a),
                           spur=spurious(a), census=census(a), ce=end_ce(a), onset=onset(a), has_metrics=bool(ms), stable=stable(a), mem=memorized(a),
                           stopped=stopped(a), vb=vb_screen(a))
        r["present"] = r["cold16"] is not None
        ons = "  n/a " if not r["has_metrics"] else ("never " if r["onset"] is None else f"{r['onset']:6d}")
        say(f"  {a:3s} | {DESC[a]:44s} | {fpp(r['cold16'])} | {fpp(r['fcold'])}  | {fpp(r['cold64'])} | {fpp(r['d128'])} | {fpp(r['d256'])} | {fpp(r['b1'])} | "
            f"{fpp(r['t1r'])} | {fpp(r['v32'])} | {f3(r['spur'])} | {f3(r['census'])}  | {f3(r['ce'])}  | {ons} | {('yes' if r['stable'] else ('STOPPED' if r['stopped'] else 'no')):7s} | "
            f"{('+'.join(r['mem']) or '-'):14s} | {fpp(r['vb'])}")
    V["arms"] = rows
    present = [a for a in ARMS if rows[a]["present"]]
    V["STABILITY"] = "NO-DATA" if not present else ("ALL-STABLE" if all(rows[a]["stable"] for a in present) else "UNSTABLE:" + ",".join(a for a in present if not rows[a]["stable"]))
    mem = [a for a in ARMS if rows[a]["mem"]]; V["MEMORIZATION"] = "NONE" if not mem else "MEMORIZED:" + ",".join(mem)
    say(f"\nSTABILITY: {V['STABILITY']}   MEMORIZATION: {V['MEMORIZATION']}")
    if errs:
        say("\nVERDICT WITHHELD: integrity breach (fix the data or the reader, never the rule)."); _write(V); return V
    c = {a: rows[a]["cold16"] for a in ARMS}
    sseed = [a for a in SEED_ARMS if rows[a]["stable"] and c[a] is not None]; V["STABLE_SEEDS"] = sseed
    if len(sseed) >= 2: spread = max(c[a] for a in sseed) - min(c[a] for a in sseed); FLOOR = max(spread, FLOOR_MIN); fsrc = f"seed triple {sseed}"
    else: spread = None; FLOOR = FLOOR_DEFAULT; fsrc = "FLOOR-DEFAULT (fewer than 2 stable seed arms)"
    vbs = [rows[a]["vb"] for a in sseed if rows[a]["vb"] is not None]; FLOOR_SCREEN = (max(vbs) - min(vbs)) if len(vbs) >= 2 else None
    M = float(np.mean([c[a] for a in sseed])) if sseed else None
    V.update({"FLOOR_C": FLOOR, "FLOOR_SRC": fsrc, "FLOOR_SCREEN": FLOOR_SCREEN, "SPREAD-C": spread, "M": M})
    say(f"\nNOISE FLOOR_C = {FLOOR:.4f} ({fsrc}); contrasts read beyond 2 x FLOOR_C = {2*FLOOR:.4f}; FLOOR_SCREEN (vb strat-512, descriptive) = {f4(FLOOR_SCREEN)}")
    def contrast(a): return None if None in (c[a], M) else c[a] - M
    # R-CH-0 SEEDS
    V["R-CH-0"] = "NO-DATA" if spread is None else ("TIGHT" if spread <= .015 else ("LOOSE" if spread <= .03 else "WIDE"))
    say(f"R-CH-0 SEEDS (SPREAD-C {f4(spread)} over {sseed}; seed mean M = {f4(M)}): {V['R-CH-0']}")
    # R-CH-1 ORBIT
    d3 = contrast("C3"); V["ONSETS"] = {a: rows[a]["onset"] for a in ARMS}
    so = [rows[a]["onset"] for a in sseed if rows[a]["onset"] is not None]; mso = float(np.mean(so)) if so else None; o3 = rows["C3"]["onset"]
    if d3 is None: V["R-CH-1"] = "NO-DATA"
    else:
        V["R-CH-1"] = "ORBIT-LIFTS" if d3 > 2 * FLOOR else ("ORBIT-HURTS" if d3 < -2 * FLOOR else "ORBIT-FLAT")
        if not rows["C3"]["has_metrics"]: V["R-CH-1"] += "+ONSET-NO-DATA"
        elif so and (o3 is None or o3 - mso >= DELAY_STEPS): V["R-CH-1"] += "+DELAYS"
        else: V["R-CH-1"] += "+SAME-ONSET"
    say(f"R-CH-1 ORBIT (C3 - M = {fd(d3)}; onset C3 {'n/a' if not rows['C3']['has_metrics'] else ('never' if o3 is None else o3)} vs seed mean {fi(mso)} over {[a for a in sseed if rows[a]['onset'] is not None]}): {V['R-CH-1']}")
    # R-CH-2 OPERATOR
    d4 = contrast("C4")
    V["R-CH-2"] = "NO-DATA" if d4 is None else ("ATTENTION-HELPS" if d4 > 2 * FLOOR else ("ATTENTION-HURTS" if d4 < -2 * FLOOR else "MEAN-SUFFICES"))
    say(f"R-CH-2 OPERATOR (C4 - M = {fd(d4)}): {V['R-CH-2']}")
    # R-CH-3 WIDTH-DOWN
    c5 = rows["C5"]["cold64"]
    V["R-CH-3"] = "NO-DATA" if c5 is None else ("PARITY-AT-1.3x" if c5 >= PARITY_D64 else ("TRM-CLASS-AT-1.3x" if c5 >= TRM_MLP else "BELOW-AT-1.3x"))
    say(f"R-CH-3 WIDTH-DOWN (C5 cold16 {fpp(c['C5'])} cold64 {fpp(c5)} vs parity {100*PARITY_D64:.1f} / TRM-MLP {100*TRM_MLP:.1f}): {V['R-CH-3']}")
    # R-CH-4 COMMIT
    cm = commit("C6") or {}; w6 = cm.get("commit_wrong_among_committed_unsolved_last"); d6 = contrast("C6")
    V["COMMIT"] = dict(w=w6, wrong_all=cm.get("commit_wrong_among_committed_last"), frac=cm.get("commit_frac_last"), auc=cm.get("commit_auc_last"), tau=cm.get("commit_tau"), n=cm.get("n"), d=d6)
    if w6 is None or d6 is None: V["R-CH-4"] = "NO-DATA"
    elif w6 < .30: V["R-CH-4"] = "CALIBRATED-AND-LIFTS" if d6 > 2 * FLOOR else ("CALIBRATED-AT-COST" if d6 < -2 * FLOOR else "CALIBRATED-INERT")
    else: V["R-CH-4"] = "UNCALIBRATED" + ("+LIFTS" if d6 > 2 * FLOOR else ("+COST" if d6 < -2 * FLOOR else "+FLAT"))
    say(f"R-CH-4 COMMIT (C6 sub20k_t16_commit n {fi(cm.get('n'))}, tau {cm.get('commit_tau')}: wrong-among-committed at unsolved w = {f4(w6)}, over all committed {f4(V['COMMIT']['wrong_all'])}, "
        f"committed fraction {f4(V['COMMIT']['frac'])}, commit AUC {f4(V['COMMIT']['auc'])}; C6 - M = {fd(d6)}): {V['R-CH-4']}")
    # R-CH-5 SELECTOR
    say("\nSELECTOR TABLE (5k x k32 scan at t64; spurious = lens-E5 per-draw rate, clean <= .010; verified@32 = vote_at_k[32]; t1r@32 = residual-selected top-1)")
    say("  arm | spurious | t1r@32 | verified@32 | t1r/verified")
    sel = {}
    for a in ARMS:
        r = rows[a]; ratio = None if (r["t1r"] is None or not r["v32"]) else r["t1r"] / r["v32"]
        sel[a] = dict(spurious=r["spur"], t1r32=r["t1r"], verified32=r["v32"], ratio=ratio)
        say(f"  {a:3s} | {f3(r['spur'])}    | {fpp(r['t1r'])} | {fpp(r['v32'])}      | {f4(ratio)}")
    V["SELECTOR"] = sel
    have = [a for a in ARMS if rows[a]["spur"] is not None]; dirty = [a for a in have if rows[a]["spur"] > .01]
    V["R-CH-5"] = "NO-DATA" if not have else ("SELECTOR-CLEAN" if not dirty else "SELECTOR-DIRTY:" + ",".join(dirty))
    say(f"R-CH-5 SELECTOR (present {have}): {V['R-CH-5']}")
    # R-CH-6 DEPTH
    say("\nDEPTH TABLE (ladder cold16 / cold64 / D128 / D256; regressions = depth_regressions/n per row [D16 full, D64 100k, D128 20k, D256 5k]; bound .0005 x n)")
    say("  arm | cold16 | cold64 |  D128  |  D256  | gain 64->128 | gain 128->256 | regressions D16 / D64 / D128 / D256 | flag")
    dep = {}; bad = []; any_row = False
    for a in ARMS:
        r = rows[a]; reg = regressions(a); flags = []
        for name, t in DEPTH_ROWS:
            if name in reg:
                any_row = True
                if reg[name][0] > .0005 * reg[name][1]: flags.append(f"D{t}")
        if flags: bad.append(a)
        g1 = None if None in (r["cold64"], r["d128"]) else r["d128"] - r["cold64"]; g2 = None if None in (r["d128"], r["d256"]) else r["d256"] - r["d128"]
        dep[a] = dict(cold16=r["cold16"], cold64=r["cold64"], d128=r["d128"], d256=r["d256"], gain_64_128=g1, gain_128_256=g2,
                      regressions={name: reg.get(name) for name, _t in DEPTH_ROWS}, regress_rows=flags)
        rtxt = " / ".join("-" if name not in reg else f"{reg[name][0]}/{reg[name][1]}" for name, _t in DEPTH_ROWS)
        say(f"  {a:3s} | {fpp(r['cold16'])} | {fpp(r['cold64'])} | {fpp(r['d128'])} | {fpp(r['d256'])} | {fd(g1):>12s} | {fd(g2):>13s} | {rtxt:35s} | {'REGRESSES ' + ','.join(flags) if flags else ('ok' if reg else '-')}")
    V["DEPTH"] = dep
    V["R-CH-6"] = "NO-DATA" if not any_row else ("DEPTH-MONOTONE" if not bad else "DEPTH-REGRESSES:" + ",".join(bad))
    say(f"R-CH-6 DEPTH: {V['R-CH-6']}")
    # R-CH-7 PARITY + HEADLINE
    passing = [a for a in sseed if rows[a]["cold64"] is not None and rows[a]["cold64"] >= PARITY_D64]
    if not any(rows[a]["cold64"] is not None for a in sseed): V["R-CH-7"] = "NO-DATA"
    else: V["R-CH-7"] = f"PARITY-x{len(sseed)}" if len(passing) == len(sseed) else f"PARITY-PARTIAL {len(passing)}/{len(sseed)}"
    head = {}
    for key in ("cold64", "d128"):
        vals = [rows[a][key] for a in sseed if rows[a][key] is not None]
        head[key] = dict(mean=float(np.mean(vals)) if vals else None, spread=(max(vals) - min(vals)) if vals else None, n=len(vals))
    V["HEADLINE"] = head
    say(f"R-CH-7 PARITY (cold64 >= {PARITY_D64:.3f} on {passing} of the stable seed arms {sseed}): {V['R-CH-7']}")
    say(f"HEADLINE (stable seed arms, single pass EMA): cold64 mean {fpp(head['cold64']['mean'])} spread {f4(head['cold64']['spread'])} (n {head['cold64']['n']}); "
        f"D128 mean {fpp(head['d128']['mean'])} spread {f4(head['d128']['spread'])} (n {head['d128']['n']})")
    # R-CH-EXT
    say("\nEXTENSION AUDIT (R-CH-EXT; END-PEAK = the selected step inside the last 4000 steps of the FINAL budget = a peak still rising at the end)")
    say("  arm | EXTENDED | from -> to (peak)     | selected        | budget(argv) | final budget | END-PEAK")
    ext = {}
    for a in ARMS:
        e = extended(a); vb = val_best(a); bud = budget(a); fin = e["to"] if e and e.get("to") else bud
        step = vb["step"] if vb else None; endpk = None if None in (step, fin) else bool(step > fin - END_WINDOW)
        ext[a] = dict(extended=bool(e), frm=e["frm"] if e else None, to=e["to"] if e else None, peak=e["peak"] if e else None, selected=step,
                      label=vb["label"] if vb else None, budget=bud, final_budget=fin, end_peak=endpk)
        say(f"  {a:3s} | {'yes' if e else 'no':8s} | {(f'{fi(e['frm'])} -> {fi(e['to'])} (peak {fi(e['peak'])})' if e else '-'):21s} | "
            f"{(fi(step) + (' [' + vb['label'] + ']' if vb and vb['label'] != 'val' else '')) if vb else '-':15s} | {fi(bud):12s} | {fi(fin):12s} | {'-' if endpk is None else ('YES' if endpk else 'no')}")
    V["EXT"] = ext; V["N_EXTENDED"] = sum(1 for a in ARMS if ext[a]["extended"]); V["END_PEAK"] = [a for a in ARMS if ext[a]["end_peak"]]
    V["R-CH-EXT"] = f"EXTENDED:{','.join(a for a in ARMS if ext[a]['extended']) or 'none'}|END-PEAK:{','.join(V['END_PEAK']) or 'none'}"
    say(f"R-CH-EXT: {V['R-CH-EXT']} ({V['N_EXTENDED']} extended arm(s))")
    # CHAMPION BY RULE (ties break by arm order: max() returns the first maximum)
    cands = [a for a in sseed if rows[a]["cold64"] is not None]; champ = max(cands, key=lambda a: rows[a]["cold64"]) if cands else None
    allc = [a for a in ARMS if rows[a]["cold64"] is not None]; best = max(allc, key=lambda a: rows[a]["cold64"]) if allc else None
    V.update({"CHAMP": champ, "CHAMP_COLD64": rows[champ]["cold64"] if champ else None, "CHAMP_MEMORIZED": rows[champ]["mem"] if champ else None,
              "LABELED_BEST": best, "LABELED_BEST_COLD64": rows[best]["cold64"] if best else None})
    V["R-CH-CHAMP"] = f"{champ or 'none'}|cold64:{f4(V['CHAMP_COLD64'])}|LABELED-BEST:{best or 'none'}|cold64:{f4(V['LABELED_BEST_COLD64'])}"
    say(f"R-CH-CHAMP: {champ or 'none'} cold64 {f4(V['CHAMP_COLD64'])}{(' [MEMORIZED ' + '+'.join(rows[champ]['mem']) + ']') if champ and rows[champ]['mem'] else ''} "
        f"(labeled best: {best or 'none'} {f4(V['LABELED_BEST_COLD64'])})")
    # SYNC A/B rider
    sy = jload(RUNS / "analysis" / "champ_sync_ab.json"); ratio = None
    if sy and sy.get("wall_default_s") and sy.get("wall_per_step_s") is not None:
        ratio = float(sy["wall_per_step_s"]) / float(sy["wall_default_s"])
        V["SYNC"] = dict(wall_default_s=sy["wall_default_s"], wall_per_step_s=sy["wall_per_step_s"], ratio=ratio, rows=sy.get("rows"), k=sy.get("k"), zone=sy.get("zone"))
        say(f"SYNC A/B (descriptive; {sy.get('rows')} rows x k{sy.get('k')} in {sy.get('zone')}): default {sy['wall_default_s']:.1f} s, per-step {sy['wall_per_step_s']:.1f} s, ratio per-step/default = {ratio:.2f} (predicted >= 1.5)")
    else: V["SYNC"] = None; say("SYNC A/B: NO-DATA (runs/analysis/champ_sync_ab.json absent)")
    # predictions
    say("\nPREDICTION SCOREBOARD (bands locked pre-data):")
    sc = {}
    for a in SEED_ARMS:
        r = rows[a]
        sc[a] = (f"cold16 {hm(c[a], PRED['cold16'])} cold64 {hm(r['cold64'], PRED['cold64'])} D128 {hm(r['d128'], PRED['d128'])} D256 {hm(r['d256'], PRED['d256'])} "
                 f"spurious {hm(r['spur'], PRED['spur'])} verified@32 {hm(r['v32'], PRED['v32'])}")
    sc["SPREAD-C"] = hm(spread, PRED["spread"]); sc["C3"] = f"cold16 {hm(c['C3'], PRED['c3_cold16'])}"; sc["C4"] = f"d {hm(d4, PRED['c4_d'])}"
    sc["C5"] = f"cold16 {hm(c['C5'], PRED['c5_cold16'])} cold64 {hm(c5, PRED['c5_cold64'])}"; sc["C6"] = f"w {hm(w6, PRED['c6_w'])} d {hm(d6, PRED['c6_d'])}"
    sc["DEPTH-GAIN 64->128"] = " ".join(f"{a}:{hm(dep[a]['gain_64_128'], PRED['gain'])}" for a in ARMS)
    any_pre = any(ext[a]["budget"] is not None or ext[a]["extended"] for a in ARMS)      # no pretrain dir anywhere -> n/a, not a vacuous HIT
    sc["SYNC"] = hm(ratio, PRED["sync"]); sc["N-EXTENDED"] = hm(V["N_EXTENDED"] if any_pre else None, PRED["n_ext"])
    for k_, v_ in sc.items(): say(f"  {k_}: {v_}")
    V["PRED"] = sc
    say("\nLETTERS: " + " · ".join(f"{k} {V[k]}" for k in ("INTEGRITY", "STABILITY", "MEMORIZATION", "R-CH-0", "R-CH-1", "R-CH-2", "R-CH-3", "R-CH-4", "R-CH-5",
                                                          "R-CH-6", "R-CH-7", "R-CH-EXT", "R-CH-CHAMP")) + f" · FLOOR_C {FLOOR:.4f}")
    _write(V); return V

def _san(x):
    if isinstance(x, dict): return {str(k): _san(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set, frozenset)): return [_san(v) for v in x]
    if isinstance(x, np.integer): return int(x)
    if isinstance(x, np.floating): return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.bool_): return bool(x)
    if isinstance(x, float) and not np.isfinite(x): return None
    return x
def _write(V):
    d = RUNS / "analysis"; d.mkdir(parents=True, exist_ok=True)
    (d / "champ_verdict.txt").write_text("\n".join(LINES) + "\n"); LINES.clear()
    (d / "champ_verdict.json").write_text(json.dumps(_san(V), indent=1))

# ---------- selftest ----------
def _mk(root, a, *, vc, fc=None, c64=None, d128v=None, d256v=None, b1v=None, v32=.997, t1rv=.99, spur=.0005, ce_end=None, onset_=20000, census_=0.0, stopped_=None,
        n_scan_override=None, idx_offset=0, vb=20000, vb_txt=None, scan_ckpt=None, budget_=30000, ext=None, commit_=None, regress=None, fallback=False, metrics_=True,
        fin_copied=False, n_fin=None, seed=1):
    """One synthetic arm: every file the analyzer reads, from the given numbers (an override rewrites the arm from scratch)."""
    rng = np.random.default_rng(seed)
    for p in list(root.glob(f"*{TAG}{a}_*")) + list(root.glob(f"*{TAG}{a}")) + [root / f"pretrain{TAG}_{a}"]:
        if p.exists(): shutil.rmtree(p)
    fin = ext[1] if ext else budget_
    if fallback: vb = fin
    ck = f"runs/pretrain{TAG}_{a}/ckpt_{vb:06d}.pkl"; ev = root / f"sxeval_p{TAG}{a}"
    fc = vc - .03 if fc is None else fc; c64 = min(vc + .03, .999) if c64 is None else c64
    d128v = min(c64 + .005, .999) if d128v is None else d128v; d256v = min(d128v + .003, .999) if d256v is None else d256v
    b1v = min(vc + .01, .999) if b1v is None else b1v; regress = regress or {}
    def wrow(name, v, n, t, **extra):
        d = ev / name; d.mkdir(parents=True, exist_ok=True)
        (d / "summary_all.json").write_text(json.dumps(dict(n=n, exact_acc=v, ckpt=ck, t_total=t, ema=True, **extra)))
    def depth(name, t): return dict(exact_by_step_curve=[0.0] * (t - 1) + [1.0], depth_regressions=int(regress.get(name, 0)))
    wrow("full_vsel_t16", vc, N_FULL, 16, **depth("full_vsel_t16", 16))
    wrow("full_final_t16", fc, n_fin if n_fin is not None else (N_FULL if fin_copied else 50000), 16)
    wrow("full_vsel_t16_alt", vc + .001, N_ALT, 16)
    wrow("full_vsel_t64", c64, N_D64, 64, **depth("full_vsel_t64", 64))
    wrow("sub20k_t128", d128v, N_D128, 128, **depth("sub20k_t128", 128)); wrow("sub5k_t256", d256v, N_D256, 256, **depth("sub5k_t256", 256))
    if commit_: wrow("sub20k_t16_commit", vc, N_COMMIT, 16, commit_frac_last=commit_.get("frac", .6), commit_wrong_among_committed_last=commit_.get("wrong", .05),
                     commit_wrong_among_committed_unsolved_last=commit_["w"], commit_auc_last=commit_.get("auc", .9), commit_tau=.9)
    # the 5k x k32 scan with an injected spurious rate (correct draws converge low; spurious = wrong AND converged)
    d = root / f"sxscan_p{TAG}{a}"; d.mkdir(parents=True, exist_ok=True); n = N_SCAN if n_scan_override is None else n_scan_override; idx = np.arange(n) + idx_offset
    cold = rng.random(n) < vc; hit = np.full(n, -1); hit[: int(v32 * n)] = 3
    ex = np.zeros((n, K_SCAN), bool); ex[:, 0] = cold; ex[:, 1:] = rng.random((n, K_SCAN - 1)) < .5
    rs = np.where(ex, rng.random((n, K_SCAN)) * .01, .05 + rng.random((n, K_SCAN)) * .05)
    for i, j in np.argwhere(~ex)[: int(round(spur * (~ex).sum()))]: rs[i, j] = .002
    (d / "summary_all.json").write_text(json.dumps(dict(n=n, b1_exact=b1v, exact_acc=float(cold.mean()), exact_acc_vote=v32, vote_at_k={"32": v32}, t1r_at_k={"32": t1rv},
                                                        ckpt=scan_ckpt or ck, **SCAN_PROTO)))
    np.savez(d / "records_all.npz", idx=idx, cold_exact=cold, mi_first_hit=hit, mi_exact_k=ex.astype(np.int64), mi_resid_k=rs)
    for tg in ["s010000", "s020000", "vb"] + (["s030000", "s040000"] if fin >= 50000 else []):
        d = root / f"sxscreen_p{TAG}{a}_{tg}"; d.mkdir(parents=True, exist_ok=True)
        (d / "summary_all.json").write_text(json.dumps(dict(n=512, exact_acc=vc - .01, exact_acc_vote=1.0, uv_vote_k256=min(vc + .02, 1.0),
                                                            ckpt=ck if tg == "vb" else f"runs/pretrain{TAG}_{a}/ckpt_{tg[1:]}.pkl")))
    for which in ("vsel", "final"):
        d = root / f"sxcensus_p{TAG}{a}_{which}"; d.mkdir(parents=True, exist_ok=True)
        (d / "census.json").write_text(json.dumps(dict(ckpt=ck, rows=[dict(t=64, exploded_frac=census_), dict(t=256, exploded_frac=census_)])))
    d = root / f"sxcalib_p{TAG}{a}_vsel"; d.mkdir(parents=True, exist_ok=True); (d / "calib.json").write_text(json.dumps(dict(ckpt=ck, topk_correct_stalled=.6)))
    pd = root / f"pretrain{TAG}_{a}"; pd.mkdir(parents=True, exist_ok=True)
    (pd / "config.json").write_text(json.dumps(dict(argv=dict(steps=budget_, seed=int(a[1]), dec_width=384, out=f"runs/pretrain{TAG}_{a}"), n_params_bulk=2780933)))
    vt = vb if vb_txt is None else vb_txt
    (pd / "val_best.txt").write_text("FALLBACK-FINAL\n" if fallback else f"{vt:06d} 0.9531 {vt}\n")
    if stopped_: (pd / "STOPPED.txt").write_text(stopped_ + "\n")
    if ext: (pd / "EXTENDED.txt").write_text(f"EXTENDED from {ext[0]} to {ext[1]} (peak at {ext[2]})\n")
    if metrics_:
        rows = [dict(step=s, loss=(.5 if (onset_ is None or s < onset_) else .03) + .01, ce_in=(.5 if (onset_ is None or s < onset_) else .03)) for s in range(1000, fin + 1, 1000)]
        if ce_end is not None:
            for r in rows[-5:]: r["ce_in"] = ce_end
        lines = [json.dumps(r) for r in rows] + [json.dumps({"monitor": {"step": s, "val_t16": .9, "val_t16_ema": .93, "n_val": 512}}) for s in range(2000, fin + 1, 2000)]
        (pd / "metrics.jsonl").write_text("\n".join(lines) + "\n")

def selftest():
    checks = []
    def run(build):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); globals()["RUNS"] = root; build(root)
            with contextlib.redirect_stdout(io.StringIO()): V = analyze()
            return V, (root / "analysis" / "champ_verdict.txt").read_text(), json.loads((root / "analysis" / "champ_verdict.json").read_text())
    def wA(r):  # the hoped-for world: a tight triple at parity, the orbit lifts and delays, the mean suffices, w192 at parity, the head calibrated and inert
        _mk(r, "C0", vc=.950, c64=.978, d128v=.983, d256v=.986, onset_=20000)
        _mk(r, "C1", vc=.955, c64=.981, d128v=.986, d256v=.989, onset_=22000, vb=22000)
        _mk(r, "C2", vc=.946, c64=.975, d128v=.980, d256v=.983, onset_=18000, vb=18000)
        _mk(r, "C3", vc=.972, c64=.990, d128v=.995, d256v=.997, onset_=35000, budget_=30000, ext=(30000, 50000, 28000), vb=48000)
        _mk(r, "C4", vc=.952, c64=.979)
        _mk(r, "C5", vc=.900, c64=.935)
        _mk(r, "C6", vc=.953, c64=.980, budget_=50000, vb=30000, commit_=dict(w=.20, auc=.93, frac=.55, wrong=.03))
        (r / "analysis").mkdir(exist_ok=True)
        (r / "analysis" / "champ_sync_ab.json").write_text(json.dumps(dict(wall_default_s=100.0, wall_per_step_s=180.0, rows=512, k=32, zone="asia-south1-b")))
    v, txt, js = run(wA)
    checks += [("A integrity PASS", v["INTEGRITY"] == "PASS"), ("A all stable, none memorized", v["STABILITY"] == "ALL-STABLE" and v["MEMORIZATION"] == "NONE"),
               ("A FLOOR_C = the triple's spread .009", abs(v["FLOOR_C"] - .009) < 1e-9 and v["STABLE_SEEDS"] == ["C0", "C1", "C2"]),
               ("A FLOOR_SCREEN read over the triple", v["FLOOR_SCREEN"] is not None and abs(v["FLOOR_SCREEN"] - .009) < 1e-9),
               ("A seeds TIGHT", v["R-CH-0"] == "TIGHT"), ("A orbit lifts + delays (35k vs mean 20k)", v["R-CH-1"] == "ORBIT-LIFTS+DELAYS"),
               ("A operator: the mean suffices", v["R-CH-2"] == "MEAN-SUFFICES"), ("A width-down parity at 1.3x", v["R-CH-3"] == "PARITY-AT-1.3x"),
               ("A commit calibrated + inert, AUC/frac reported", v["R-CH-4"] == "CALIBRATED-INERT" and v["COMMIT"]["auc"] == .93 and v["COMMIT"]["frac"] == .55),
               ("A selector clean at the injected .0005", v["R-CH-5"] == "SELECTOR-CLEAN" and all(abs(v["SELECTOR"][a]["spurious"] - .0005) < 2e-4 for a in ARMS)),
               ("A depth monotone", v["R-CH-6"] == "DEPTH-MONOTONE"), ("A parity x3", v["R-CH-7"] == "PARITY-x3"),
               ("A headline = the triple's mean/spread at D64 and D128", abs(v["HEADLINE"]["cold64"]["mean"] - .978) < 1e-9 and abs(v["HEADLINE"]["d128"]["spread"] - .006) < 1e-9),
               ("A champion C1 by cold64; labeled best C3", v["CHAMP"] == "C1" and v["LABELED_BEST"] == "C3" and "R-CH-CHAMP: C1 cold64 0.9810 (labeled best: C3 0.9900)" in txt),
               ("A extension audit: C3 extended to 50k, selected 48k = END-PEAK; C6 not", v["EXT"]["C3"]["extended"] and v["EXT"]["C3"]["final_budget"] == 50000 and v["EXT"]["C3"]["end_peak"]
                and v["N_EXTENDED"] == 1 and v["END_PEAK"] == ["C3"] and v["EXT"]["C6"]["end_peak"] is False),
               ("A sync ratio 1.8", abs(v["SYNC"]["ratio"] - 1.8) < 1e-9), ("A every prediction HIT", all("MISS" not in s and "n/a" not in s for s in v["PRED"].values())),
               ("A JSON round-trips the letters", js["R-CH-1"] == v["R-CH-1"] and js["arms"]["C0"]["cold16"] == .950 and js["INTEGRITY"] == "PASS")]
    def wB1(r): wA(r); _mk(r, "C3", vc=.972, c64=.990, n_scan_override=20000, budget_=30000, ext=(30000, 50000, 28000), vb=48000)   # a wide arm scanned at 20k
    v, txt, js = run(wB1); checks += [("B1 scan at n 20000 = protocol breach, verdict withheld", v["INTEGRITY"] == "FAIL" and "R-CH-0" not in v and "VERDICT WITHHELD" in txt)]
    def wB2(r): wA(r); _mk(r, "C1", vc=.955, c64=.981, vb=22000, scan_ckpt=f"runs/pretrain{TAG}_C1/ckpt_020000.pkl")                 # two ckpt paths for one arm
    v, txt, js = run(wB2); checks += [("B2 two checkpoint paths on one arm withhold", v["INTEGRITY"] == "FAIL" and any("different grids" in e for e in v["INTEGRITY_REASONS"]))]
    def wB3(r): wA(r); _mk(r, "C0", vc=.950, c64=.978, vb=22000, vb_txt=20000)                                                     # evals on 22k, val_best says 20k
    v, txt, js = run(wB3); checks += [("B3 eval step != val_best step withholds", v["INTEGRITY"] == "FAIL" and any("val_best.txt selects 20000" in e for e in v["INTEGRITY_REASONS"]))]
    def wB4(r): wA(r); _mk(r, "C2", vc=.946, c64=.975, vb=18000, idx_offset=1, n_fin=30000)                                       # idx set differs + a final n off-protocol
    v, txt, js = run(wB4)
    checks += [("B4 idx sets differ + final n 30000: both reasons named", v["INTEGRITY"] == "FAIL" and any("idx sets differ" in e for e in v["INTEGRITY_REASONS"])
                and any("full_final_t16 n=30000" in e for e in v["INTEGRITY_REASONS"])),
               ("B4 INTEGRITY: FAIL (reasons) printed", "INTEGRITY: FAIL (" in txt)]
    def wC(r):  # memorized C0 (counted, labeled), STOPPED C1, dirty C2 below parity, C3 missing, C4 helps but regresses, C5 TRM-class, C6 uncalibrated + lifts
        _mk(r, "C0", vc=.950, fc=.80, c64=.930, ce_end=.01)
        _mk(r, "C1", vc=.940, c64=.970, stopped_="STOPPED final step 18000 (NaN halt; one-shot amputation)")
        _mk(r, "C2", vc=.930, c64=.910, spur=.03)
        _mk(r, "C4", vc=.990, c64=.985, regress={"sub20k_t128": 100})
        _mk(r, "C5", vc=.85, c64=.900)
        _mk(r, "C6", vc=.990, c64=.984, budget_=50000, commit_=dict(w=.45))
    v, txt, js = run(wC)
    checks += [("C memorized C0 labeled (END-CE + drop), never disqualifying: still a stable seed arm", v["MEMORIZATION"] == "MEMORIZED:C0" and v["arms"]["C0"]["mem"] == ["END-CE", "VSEL-FINAL-DROP"]
                and v["STABLE_SEEDS"] == ["C0", "C2"]),
               ("C STOPPED C1 unstable", v["STABILITY"] == "UNSTABLE:C1" and v["arms"]["C1"]["stable"] is False),
               ("C floor from the two stable seeds (.02), LOOSE", abs(v["FLOOR_C"] - .02) < 1e-9 and v["R-CH-0"] == "LOOSE"),
               ("C missing C3: NO-DATA everywhere it appears", v["R-CH-1"] == "NO-DATA" and v["arms"]["C3"]["cold16"] is None and v["SELECTOR"]["C3"]["spurious"] is None
                and v["DEPTH"]["C3"]["cold64"] is None and v["EXT"]["C3"]["selected"] is None and v["PRED"]["C3"] == "cold16 n/a"),
               ("C attention helps (+5 pp beyond 2 x .02)", v["R-CH-2"] == "ATTENTION-HELPS"), ("C w192 TRM-class", v["R-CH-3"] == "TRM-CLASS-AT-1.3x"),
               ("C commit uncalibrated + lifts", v["R-CH-4"] == "UNCALIBRATED+LIFTS"), ("C selector dirty: C2", v["R-CH-5"] == "SELECTOR-DIRTY:C2"),
               ("C depth regresses: C4 (100 > .0005 x 20000 on D128)", v["R-CH-6"] == "DEPTH-REGRESSES:C4" and v["DEPTH"]["C4"]["regress_rows"] == ["D128"]),
               ("C parity partial 1/2", v["R-CH-7"] == "PARITY-PARTIAL 1/2"),
               ("C champion C0 (memorized allowed, labeled); labeled best C4", v["CHAMP"] == "C0" and v["CHAMP_MEMORIZED"] and "[MEMORIZED END-CE+VSEL-FINAL-DROP]" in txt and v["LABELED_BEST"] == "C4"),
               ("C sync absent -> NO-DATA; no extended arm", v["SYNC"] is None and v["PRED"]["SYNC"] == "n/a" and v["N_EXTENDED"] == 0)]
    def wD(r):  # one stable seed arm -> FLOOR-DEFAULT; the orbit hurts at the same onset; attention hurts; w192 below; the head calibrated at cost; parity x1
        _mk(r, "C0", vc=.950, c64=.978)
        _mk(r, "C3", vc=.900, c64=.940, onset_=20000, budget_=50000)
        _mk(r, "C4", vc=.900)
        _mk(r, "C5", vc=.800, c64=.850)
        _mk(r, "C6", vc=.900, budget_=50000, commit_=dict(w=.2))
    v, txt, js = run(wD)
    checks += [("D FLOOR-DEFAULT .015 with one stable seed arm; seeds NO-DATA", v["FLOOR_C"] == .015 and "FLOOR-DEFAULT" in v["FLOOR_SRC"] and v["R-CH-0"] == "NO-DATA" and v["FLOOR_SCREEN"] is None),
               ("D orbit hurts + same onset", v["R-CH-1"] == "ORBIT-HURTS+SAME-ONSET"), ("D attention hurts", v["R-CH-2"] == "ATTENTION-HURTS"),
               ("D w192 below", v["R-CH-3"] == "BELOW-AT-1.3x"), ("D commit calibrated at cost", v["R-CH-4"] == "CALIBRATED-AT-COST"),
               ("D parity x1, champion C0", v["R-CH-7"] == "PARITY-x1" and v["CHAMP"] == "C0")]
    v, txt, js = run(lambda r: None)
    checks += [("E no data at all: every letter NO-DATA, nothing crashes", v["INTEGRITY"] == "PASS" and v["STABILITY"] == "NO-DATA" and all(v[k] == "NO-DATA" for k in ("R-CH-0", "R-CH-1", "R-CH-2", "R-CH-3", "R-CH-4", "R-CH-5", "R-CH-6", "R-CH-7"))
                and v["R-CH-CHAMP"].startswith("none|") and v["R-CH-EXT"] == "EXTENDED:none|END-PEAK:none" and v["FLOOR_C"] == .015)]
    def wF1(r): wA(r); _mk(r, "C6", vc=.953, c64=.980, budget_=50000, vb=30000, commit_=dict(w=.45))
    v, txt, js = run(wF1); checks += [("F1 uncalibrated + flat", v["R-CH-4"] == "UNCALIBRATED+FLAT")]
    def wF2(r): wA(r); _mk(r, "C6", vc=.900, c64=.950, budget_=50000, vb=30000, commit_=dict(w=.45))
    v, txt, js = run(wF2); checks += [("F2 uncalibrated + cost", v["R-CH-4"] == "UNCALIBRATED+COST")]
    def wF3(r): wA(r); _mk(r, "C6", vc=.990, c64=.992, budget_=50000, vb=30000, commit_=dict(w=.20))
    v, txt, js = run(wF3); checks += [("F3 calibrated and lifts; the labeled best moves to C6", v["R-CH-4"] == "CALIBRATED-AND-LIFTS" and v["LABELED_BEST"] == "C6" and v["CHAMP"] == "C1")]
    def wF4(r): wA(r); _mk(r, "C3", vc=.972, c64=.990, onset_=None, budget_=30000, ext=(30000, 50000, 28000), vb=48000)
    v, txt, js = run(wF4); checks += [("F4 C3 never memorizes while the seeds do -> +DELAYS", v["R-CH-1"] == "ORBIT-LIFTS+DELAYS" and v["ONSETS"]["C3"] is None)]
    def wF5(r): wA(r); _mk(r, "C3", vc=.955, c64=.982, onset_=24000, budget_=50000, vb=24000)
    v, txt, js = run(wF5); checks += [("F5 orbit flat (+.5 pp inside 2 x .009) + same onset (24k vs 20k < 10k)", v["R-CH-1"] == "ORBIT-FLAT+SAME-ONSET")]
    def wF6(r): wA(r); _mk(r, "C3", vc=.972, c64=.990, metrics_=False, budget_=50000, vb=20000)
    v, txt, js = run(wF6); checks += [("F6 C3 without metrics.jsonl reads +ONSET-NO-DATA (never 'never')", v["R-CH-1"] == "ORBIT-LIFTS+ONSET-NO-DATA")]
    def wF7(r): wA(r); _mk(r, "C2", vc=.930, c64=.975, vb=18000)   # spread .025 -> LOOSE, FLOOR .025: C3's +2.7 pp reads FLAT
    v, txt, js = run(wF7); checks += [("F7 LOOSE spread .025 widens the floor: the orbit's +2.7 pp reads FLAT", v["R-CH-0"] == "LOOSE" and abs(v["FLOOR_C"] - .025) < 1e-9 and v["R-CH-1"].startswith("ORBIT-FLAT"))]
    def wF8(r): wA(r); _mk(r, "C2", vc=.900, c64=.975, vb=18000)
    v, txt, js = run(wF8); checks += [("F8 WIDE spread .055", v["R-CH-0"] == "WIDE" and v["PRED"]["SPREAD-C"] == "MISS-ABOVE")]
    def wF9(r): wA(r); _mk(r, "C0", vc=.950, c64=.981, d128v=.983); _mk(r, "C3", vc=.972, c64=.981, onset_=35000, budget_=50000, vb=20000)
    v, txt, js = run(wF9); checks += [("F9 ties: C0 = C1 = C3 at cold64 .981 -> champion C0, labeled best C0 (arm order)", v["CHAMP"] == "C0" and v["LABELED_BEST"] == "C0")]
    def wF10(r): wA(r); _mk(r, "C4", vc=.952, c64=.979, fallback=True, fin_copied=True)   # vsel := final (30000), the final full copied at 422,786
    v, txt, js = run(wF10)
    checks += [("F10 FALLBACK-FINAL: vsel = the final step, integrity holds, audit labels it and reads END-PEAK", v["INTEGRITY"] == "PASS" and v["EXT"]["C4"]["label"] == "FALLBACK-FINAL"
                and v["EXT"]["C4"]["selected"] == 30000 and v["EXT"]["C4"]["end_peak"] is True and "[FALLBACK-FINAL]" in txt)]
    def wF11(r):
        wA(r); _mk(r, "C3", vc=.972, c64=.990, onset_=35000, budget_=30000, ext=(30000, 50000, 28000), vb=40000)
        _mk(r, "C0", vc=.950, c64=.978, d128v=.983, ext=(30000, 50000, 28000), vb=20000); _mk(r, "C6", vc=.953, c64=.980, budget_=50000, vb=30000, ext=(50000, 70000, 48000), commit_=dict(w=.2))
    v, txt, js = run(wF11)
    checks += [("F11 extension audit: 3 extended arms (MISS-ABOVE), C3 selected 40k of 50k is not END-PEAK", v["N_EXTENDED"] == 3 and v["PRED"]["N-EXTENDED"] == "MISS-ABOVE"
                and v["EXT"]["C3"]["end_peak"] is False and v["EXT"]["C6"]["final_budget"] == 70000 and v["R-CH-EXT"] == "EXTENDED:C0,C3,C6|END-PEAK:none")]
    def wF12(r): wA(r); _mk(r, "C4", vc=.952, c64=.979, regress={"full_vsel_t64": 50}); _mk(r, "C5", vc=.900, c64=.935, regress={"full_vsel_t64": 51})
    v, txt, js = run(wF12); checks += [("F12 the regression bound is inclusive: 50/100k ok, 51/100k regresses", v["R-CH-6"] == "DEPTH-REGRESSES:C5" and v["DEPTH"]["C4"]["regress_rows"] == [])]
    def wF13(r): wA(r); _mk(r, "C2", vc=.946, c64=.910, vb=18000)
    v, txt, js = run(wF13); checks += [("F13 parity partial 2/3; champion still C1", v["R-CH-7"] == "PARITY-PARTIAL 2/3" and v["CHAMP"] == "C1")]
    def wF14(r): wA(r); _mk(r, "C1", vc=.955, c64=.981, vb=22000, census_=.05)   # census-unstable seed arm drops out of the floor
    v, txt, js = run(wF14)
    checks += [("F14 census > .02 -> unstable C1; floor from C0/C2 clamps to .005; parity x2", v["STABILITY"] == "UNSTABLE:C1" and v["STABLE_SEEDS"] == ["C0", "C2"] and v["FLOOR_C"] == .005
                and v["R-CH-7"] == "PARITY-x2" and v["R-CH-0"] == "TIGHT")]
    # the spurious reader on hand-built records: 10 of 1000 wrong draws converged -> .010 (clean at the bound); no correct draws -> None
    ex = np.zeros((50, 21), bool); ex[:, 0] = True; rs = np.where(ex, .001, .08); rs[:10, 1] = .0001
    checks += [("spurious = 10/1000 = .010 on hand-built records", abs(spurious("x", dict(mi_exact_k=ex.astype(np.int64), mi_resid_k=rs)) - .010) < 1e-12),
               ("spurious None without correct draws", spurious("x", dict(mi_exact_k=np.zeros((5, 4), np.int64), mi_resid_k=np.ones((5, 4)))) is None),
               ("hm bands", hm(None, (0, 1)) == "n/a" and hm(.5, (0, .3)) == "MISS-ABOVE" and hm(.1, (.2, .3)) == "MISS-BELOW" and hm(1.8, (1.5, INF)) == "HIT" and hm(.975, PRED["c3_cold16"]) == "HIT")]
    ok = 0
    for name, passed in checks:
        print(("  PASS  " if passed else "  FAIL  ") + name); ok += bool(passed)
    print(f"selftest: {ok}/{len(checks)}"); return ok == len(checks)


if __name__ == "__main__":
    if "--selftest" in sys.argv: sys.exit(0 if selftest() else 1)
    analyze()
