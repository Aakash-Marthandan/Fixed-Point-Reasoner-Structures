#!/usr/bin/env python3
# Ledger: THE COMMITMENT-VALIDITY LENS (2026-09-19; registration Documentation/Note_2026-09-19_Commitment_Validity.md, written
# before any row). DESCRIPTIVE, zero cloud: banked checkpoints on the Mac. Its own loop (mirrors suite_ckpt.run_dyn, so the rg
# controls load); the digit logits of every iteration are saved, so any later transform is an offline read of these trajectories.
"""  .venv/bin/python tools/lens_commit_validity.py --selftest
  JAX_PLATFORMS=cpu .venv/bin/python tools/lens_commit_validity.py --run C5 C0 X0 EQR SA256 B0 W0 R1     (resume-safe per model)
  .venv/bin/python tools/lens_commit_validity.py --report"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tools"))
OUT = ROOT / "runs/analysis/commit_validity_20260919"
NPZ = ROOT / "data/sudoku_extreme/sudoku_extreme_seed0.npz"
# key: (checkpoint, EMA?, the loss it was trained under, the class, description)  — EMA / raw as the corpus rows used
MODELS = {
    "C5":    ("runs/pretrainchamp_C5/ckpt_046000.pkl", True, "stablemax", "loop", "the symmetric model, width 192, seed 0 (46k)"),
    "C0":    ("runs/pretrainchamp_C0/ckpt_016000.pkl", True, "stablemax", "loop", "the symmetric model, width 384, seed 0 (16k)"),
    "X0":    ("runs/pretrainsportC1_X0/ckpt_050000.pkl", True, "stablemax", "loop", "TRM's network reproduced, 5M (50k)"),
    "EQR":   ("runs/field_ckpts/ported/eqr/ckpt_latest.pkl", True, "stablemax", "loop", "EqR's released weights, ported (loss per their code)"),
    "SA256": ("runs/_wladder_pull/grids/runs/pretrainchamp_SA256/ckpt_028000.pkl", True, "stablemax", "loop", "SE-RRM's mixers in our recipe, hidden 256 (28k)"),
    "B0":    ("runs/pretrainsportC1_B0a/ckpt_020000.pkl", False, "softmax", "control", "control without the nested loop (the best one; 45 % at 64)"),
    "W0":    ("runs/pretrainsportC2_W0a/ckpt_050000.pkl", False, "softmax", "control", "control without the nested loop"),
    "R1":    ("runs/pretrainsportC2_R1/ckpt_010000.pkl", False, "softmax", "control", "control without the nested loop, trained with deep supervision"),
}
N, T, BS, STRAT_SEED = 256, 16, 128, 20260821
TAUS = (0.5, 0.7, 0.9, 0.99); ITERS = (1, 2, 4, 8, 16)

# ---------------- pure helpers (selftested) ----------------
def softmax9(lg):
    e = np.exp(lg - lg.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)

def stablemax9(lg):
    """HRM/TRM's stablemax over the given logits: s(x) = x + 1 (x >= 0), 1 / (1 - x) (x < 0); s / sum s."""
    lg = np.asarray(lg, np.float64); s = np.where(lg >= 0, lg + 1.0, 1.0 / (1.0 - np.minimum(lg, 0.0))); return s / s.sum(-1, keepdims=True)

def share(mask, ng):
    """Per puzzle: the share of the EMPTY cells inside mask."""
    return (mask & ng).sum((1, 2)) / np.maximum(ng.sum((1, 2)), 1)

def wrong_among(com, wrong, ng):
    """Per puzzle: wrong among the committed (denominator = the committed empty cells); nan where none is committed."""
    c = (com & ng).sum((1, 2)); return np.where(c > 0, (com & wrong & ng).sum((1, 2)) / np.maximum(c, 1), np.nan)

def hold_stay(preds, ng):
    """preds (T, B, 9, 9). hold = the iteration-1 digit is the final digit; stay = the digit never changes after iteration 1."""
    hold = preds[0] == preds[-1]; stay = (preds[1:] == preds[0][None]).all(0) if len(preds) > 1 else np.ones_like(hold)
    return share(hold, ng), share(stay, ng)

def auc(pos, neg):
    """P(score of a positive > score of a negative), ties 1/2; None without both classes."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0: return None
    allv = np.concatenate([pos, neg]); order = allv.argsort(kind="mergesort"); ranks = np.empty(len(allv)); sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]: j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

def pct(v, d=1): return "  -  " if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{100 * v:.{d}f}"

def com_line(tag, tau, by_iter): return f"    {tag:9s} top probability > {tau:<4}: " + "  ".join(f"it {it:>2} {pct(v):>5} %" for it, v in by_iter)

def summarize(lg, sol, ng):
    """lg (T, B, 9, 9, 9) digit logits; sol, ng (B, 9, 9). Every rate is over the EMPTY cells, per puzzle, then the mean."""
    lg = np.asarray(lg, np.float64); preds = lg.argmax(-1) + 1; wrong = preds != sol[None]; solved = (~(wrong[-1] & ng)).all((1, 2)); J = {}
    conf = {"softmax": softmax9(lg).max(-1), "stablemax": stablemax9(lg).max(-1)}
    for k, c in conf.items():
        J[k] = {str(tau): [(it, float(share(c[it - 1] > tau, ng).mean())) for it in ITERS if it <= len(lg)] for tau in TAUS}
        J[k]["wrong_among_committed_it1"] = float(np.nanmean(wrong_among(c[0] > 0.9, wrong[0], ng)))
        J[k]["wrong_among_committed_it1_solved"] = float(np.nanmean(wrong_among(c[0] > 0.9, wrong[0], ng)[solved])) if solved.any() else None
        J[k]["committed_it1_solved"] = float(share(c[0] > 0.9, ng)[solved].mean()) if solved.any() else None
        J[k]["committed_it1_unsolved"] = float(share(c[0] > 0.9, ng)[~solved].mean()) if (~solved).any() else None
        mc = (c[-1] * ng).sum((1, 2)) / np.maximum(ng.sum((1, 2)), 1)
        J[k]["conf_end_solved"] = float(mc[solved].mean()) if solved.any() else None; J[k]["conf_end_unsolved"] = float(mc[~solved].mean()) if (~solved).any() else None
        J[k]["auc_wrong_it1"] = auc(-c[0][wrong[0] & ng], -c[0][(~wrong[0]) & ng])      # low confidence should mark the wrong cells
    hold, stay = hold_stay(preds, ng); srt = np.sort(lg[0], -1); top, margin = srt[..., -1], srt[..., -1] - srt[..., -2]
    J["free"] = dict(n=int(len(sol)), n_unsolved=int((~solved).sum()), exact_end=float(solved.mean()), first_guess_wrong=float(share(wrong[0], ng).mean()),
                     first_guess_wrong_solved=float(share(wrong[0], ng)[solved].mean()) if solved.any() else None, hold=float(hold.mean()), stay=float(stay.mean()),
                     top_logit_median=float(np.median(top[ng])), margin_median=float(np.median(margin[ng])), margin_p10=float(np.percentile(margin[ng], 10)),
                     margin_median_wrong=float(np.median(margin[wrong[0] & ng])) if (wrong[0] & ng).any() else None)
    return J

# ---------------- the run (mirrors suite_ckpt.run_dyn; the fixed start, no halting, no noise) ----------------
def run_model(key):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax, jax.numpy as jnp
    from qhrrn2 import episodic as E, grid as GR, model as M, sudoku as SU, sudoku_extreme as SX
    from qhrrn2.config import Config
    import eval_sudoku_extreme as EV
    path, ema, loss, cls, _ = MODELS[key]; d = SX.load_prepared(NPZ); ids = np.asarray(SX.stratified_subsample(d["test_rating"], N, STRAT_SEED))
    puz = d["test_q"][ids].astype(np.int32); sol = d["test_a"][ids].astype(np.int32)
    saved = E.load_ckpt(str(ROOT / path)); d0 = Config(); cfg = Config(**{k: type(getattr(d0, k))(v) for k, v in saved["config"].items()})
    assert cfg.loss_kind == loss or key == "EQR", f"{key}: config says loss_kind={cfg.loss_kind}, the registry says {loss}"
    st = saved["state_ema"] if ema else saved["state"]; assert st is not None; params = st["model"]; tvj = jnp.asarray(st["table"][0])
    eta, eta_z = (float(v) for v in M.eq_etas(params, cfg)); lay = cfg.sudoku_layout or "origin"; cv = SU.layout_canvas(lay)
    trm = cfg.cell_kind in ("trm", "dec"); K = 1 if trm else max(1, int(getattr(cfg, "inner_k", 1))); ab = EV.coupled_ab(params, cfg)
    void = jax.nn.one_hot(jnp.full((cv, cv), GR.VOID, jnp.int32), M.VOCAB).transpose(2, 0, 1); out = np.zeros((T, N, 9, 9, 9), np.float16); t_a = time.time()
    for b0 in range(0, N, BS):
        x_can = EV.place_batch(puz[b0:b0 + BS], lay); B = x_can.shape[0]; y = jnp.broadcast_to(void, (B,) + void.shape); z = None
        for t in range(T):
            tn = 0.0 if trm else min(t, cfg.T - 1) / max(cfg.T - 1, 1)
            for _ in range(K):
                first = z is None; logits, zf = EV._step(cfg, 1.0, float(tn), first)(params, x_can, y, tvj, jnp.zeros(1) if first else z); z = zf if first else z + eta_z * (zf - z)
            p = jax.nn.softmax(logits, axis=-1); pT = p.transpose(0, 3, 1, 2); y = (ab[0] * y + ab[1] * pT) if ab is not None else (y + eta * (pT - y))
            full = np.asarray(EV.layout_gather(logits, lay)); assert full.shape[1:3] == (9, 9), full.shape
            pred_ev = np.asarray(EV.layout_gather(jnp.argmax(logits, axis=-1), lay)); dig = full[..., 1:10]
            agree = ((dig.argmax(-1) + 1 == pred_ev) | (puz[b0:b0 + BS] != 0)).mean(); assert agree > 0.995, f"digit argmax differs from the evaluator's on the empty cells ({agree:.4f})"
            out[t, b0:b0 + B] = np.clip(dig, -6e4, 6e4).astype(np.float16)
        print(f"  {key}: rows {b0 + B}/{N} ({time.time() - t_a:.0f}s)", flush=True)
    return out, puz, sol, ids, dict(cell=cfg.cell_kind, loss_kind=cfg.loss_kind, ema=ema, K=K, step=int(saved.get("step", -1)))

def run(keys):
    OUT.mkdir(parents=True, exist_ok=True)
    for k in keys:
        dst = OUT / f"{k}.npz"
        if dst.exists(): print(f"SKIP {dst.name}"); continue
        lg, puz, sol, ids, meta = run_model(k)
        np.savez_compressed(OUT / f"{k}.tmp.npz", logits=lg, puz=puz, sol=sol, ids=ids, meta=json.dumps(meta), ckpt=MODELS[k][0]); os.replace(OUT / f"{k}.tmp.npz", dst); print(f"DONE {dst.name} {meta}", flush=True)

# ---------------- report ----------------
def report():
    Ls, J = [], {}; say = lambda s_="": (Ls.append(s_), print(s_))
    say("THE COMMITMENT-VALIDITY LENS (descriptive; tools/lens_commit_validity.py; registration Note_2026-09-19_Commitment_Validity.md).")
    say(f"The fixed start, no halting, no noise, {T} iterations, {N} rating-stratified test puzzles; every rate is over the EMPTY cells. OWN = the normalization the model was trained under.")
    for k, (path, ema, loss, cls, desc) in MODELS.items():
        f = OUT / f"{k}.npz"
        if not f.exists(): continue
        D = np.load(f, allow_pickle=True); s = summarize(D["logits"], D["sol"], D["puz"] == 0); s["meta"] = dict(json.loads(str(D["meta"])), cls=cls, own=loss, ckpt=path); J[k] = s; fr = s["free"]
        say(); say(f"  {k} [{cls}; trained under {loss}; {'EMA' if ema else 'raw'}]: {desc}")
        say(f"    exact at {T}: {pct(fr['exact_end'])} % (unsolved {fr['n_unsolved']} of {fr['n']}) | first guess wrong {pct(fr['first_guess_wrong'])} % (on solved puzzles {pct(fr['first_guess_wrong_solved'])} %) | iteration-1 digit = final digit {pct(fr['hold'])} %, never changes {pct(fr['stay'])} %")
        say(f"    iteration-1 logits: median top {fr['top_logit_median']:.1f}, median top-minus-second {fr['margin_median']:.1f} (10th percentile {fr['margin_p10']:.1f}; on wrong cells {fr['margin_median_wrong'] if fr['margin_median_wrong'] is None else round(fr['margin_median_wrong'], 1)})")
        for nm in ("softmax", "stablemax"):
            tag = nm + (" OWN" if nm == loss else ""); say(com_line(tag[:9] if len(tag) <= 9 else nm[:5] + " OWN", 0.9, s[nm]["0.9"]))
        for nm in ("softmax", "stablemax"):
            r = s[nm]; say(f"    {nm:9s} at iteration 1: > 0.5 {pct(r['0.5'][0][1])} | > 0.7 {pct(r['0.7'][0][1])} | > 0.9 {pct(r['0.9'][0][1])} | > 0.99 {pct(r['0.99'][0][1])} % || wrong among the committed (0.9) {pct(r['wrong_among_committed_it1'])} % (solved puzzles {pct(r['wrong_among_committed_it1_solved'])} %) | wrong-cell AUC {r['auc_wrong_it1'] if r['auc_wrong_it1'] is None else round(r['auc_wrong_it1'], 3)} | mean top probability at {T}: solved {pct(r['conf_end_solved'])}, unsolved {pct(r['conf_end_unsolved'])}")
    if J:
        say(); say("== the registered predictions ==")
        own = lambda k: J[k][J[k]["meta"]["own"]]["0.9"][0][1]; loops = [k for k in J if J[k]["meta"]["cls"] == "loop"]; ctrls = [k for k in J if J[k]["meta"]["cls"] == "control"]
        if loops:
            lo = min(own(k) for k in loops); say(f"  P1 own-normalization commitment >= 80 % on every loop model: {'HELD' if lo >= 0.8 else 'FAILED'} (minimum {pct(lo)} %: " + ", ".join(f"{k} {pct(own(k))}" for k in loops) + ")")
            say(f"  P2 >= 50 % on every loop model: {'HELD' if lo >= 0.5 else 'FAILED'}")
        if ctrls:
            hi_both = max(max(J[k]["softmax"]["0.9"][0][1], J[k]["stablemax"]["0.9"][0][1]) for k in ctrls); say(f"  P3 the controls under 15 % under both transforms: {'HELD' if hi_both < 0.15 else 'FAILED'} (maximum {pct(hi_both)} %)")
        if loops and ctrls:
            hi = max(own(k) for k in ctrls); say(f"  P4 the lowest loop model >= 3 x the highest control, own normalization: {'HELD' if lo >= 3 * hi else 'FAILED'} ({pct(lo)} % against {pct(hi)} %)")
        if "C5" in J:
            dlt = J["C5"]["softmax"]["wrong_among_committed_it1"] - J["C5"]["stablemax"]["wrong_among_committed_it1"]; say(f"  P5 C5: wrong among the committed at least 5 points lower under stablemax: {'HELD' if dlt >= 0.05 else 'FAILED'} ({100 * dlt:+.1f} points)")
        elig = [k for k in loops if J[k]["free"]["n_unsolved"] >= 5]
        if elig:
            ok = all((J[k]["stablemax"]["conf_end_solved"] - J[k]["stablemax"]["conf_end_unsolved"]) * (J[k]["softmax"]["conf_end_solved"] - J[k]["softmax"]["conf_end_unsolved"]) > 0 for k in elig)
            say(f"  P6 the confidence gap keeps its sign under stablemax (loop models with >= 5 unsolved: {', '.join(elig)}): {'HELD' if ok else 'FAILED'}")
        else: say("  P6 not scorable: no loop model has 5 unsolved puzzles here")
        if loops:
            ds = {k: abs(J[k]["softmax"]["auc_wrong_it1"] - J[k]["stablemax"]["auc_wrong_it1"]) for k in loops}; say(f"  P7 the wrong-cell AUC differs by < 0.03 between the confidences on every loop model: {'HELD' if max(ds.values()) < 0.03 else 'FAILED'} (" + ", ".join(f"{k} {v:.3f}" for k, v in ds.items()) + ")")
        if loops and ctrls:
            ok8 = all(J[k]["free"]["margin_median"] > 10 for k in loops) and all(J[k]["free"]["margin_median"] < 5 for k in ctrls)
            say(f"  P8 median margin > 10 on every loop model and < 5 on every control: {'HELD' if ok8 else 'FAILED'} (" + ", ".join(f"{k} {J[k]['free']['margin_median']:.1f}" for k in loops + ctrls) + ")")
    (OUT / "report.txt").write_text("\n".join(Ls) + "\n"); (OUT / "report.json").write_text(json.dumps(J, indent=1, default=float))

# ---------------- POST HOC (written AFTER the first rows were read; not registered; every line labeled) ----------------
def ece(conf, correct, bins=10):
    """Expected calibration error of the top probability against correctness, equal-width bins."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    if len(conf) == 0: return None
    idx = np.minimum((conf * bins).astype(int), bins - 1); tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any(): tot += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(tot)

def sure_rows(lg, sol, ng, own):
    """POST HOC helper. Per iteration t: the cells whose OWN-normalized top probability exceeds 0.9 (pooled over puzzles): their share of the
    empty cells, the share wrong at t, the share whose digit at the last iteration differs (revised); the same two for the other empty cells."""
    lg = np.asarray(lg, np.float64); preds = lg.argmax(-1) + 1; rows = []
    for t in (1, 2, 4, 8):
        if t > len(lg): break
        c = (own(lg[t - 1]).max(-1) > 0.9) & ng; u = (~c) & ng; n, m = max(int(c.sum()), 1), max(int(u.sum()), 1); w = preds[t - 1] != sol; r = preds[t - 1] != preds[-1]
        rows.append(dict(t=t, sure=float(c.sum() / ng.sum()), sure_wrong=float((w & c).sum() / n), sure_revised=float((r & c).sum() / n), unsure_wrong=float((w & u).sum() / m), unsure_revised=float((r & u).sum() / m)))
    return rows

def posthoc():
    Ls = []; say = lambda s_="": (Ls.append(s_), print(s_)); J = {}
    say("POST HOC READ of the saved logits (NOT registered; written after the first registered rows were seen; exploratory).")
    say("The second claim unit's quantities under both normalizations: on the puzzles a model FAILS at 16 iterations, the empty cells at iteration 16.")
    for k, (path, ema, loss, cls, desc) in MODELS.items():
        f = OUT / f"{k}.npz"
        if not f.exists(): continue
        D = np.load(f, allow_pickle=True); lg = D["logits"].astype(np.float64); sol = D["sol"]; ng = D["puz"] == 0
        preds = lg.argmax(-1) + 1; wrong = preds != sol[None]; solved = (~(wrong[-1] & ng)).all((1, 2)); U = ~solved; J[k] = {}
        say(); say(f"  {k} [{cls}; trained under {loss}]: unsolved {int(U.sum())} of {len(U)} puzzles; wrong cells on them at 16: {pct(float(share(wrong[-1], ng)[U].mean()) if U.any() else None)} % of the empty cells")
        for nm, fn in (("softmax", softmax9), ("stablemax", stablemax9)):
            c1, cT = fn(lg[0]).max(-1), fn(lg[-1]).max(-1); r = {}
            r["ece_it1_all"] = ece(c1[ng], ~wrong[0][ng]); r["ece_end_unsolved"] = ece(cT[U][ng[U]], ~wrong[-1][U][ng[U]]) if U.any() else None
            r["com_end_unsolved"] = float(share(cT > 0.9, ng)[U].mean()) if U.any() else None
            wa = wrong_among(cT > 0.9, wrong[-1], ng)[U] if U.any() else np.array([np.nan]); r["wrong_among_committed_end_unsolved"] = float(np.nanmean(wa)) if np.isfinite(wa).any() else None
            r["auc_wrong_end_unsolved"] = auc(-cT[U][wrong[-1][U] & ng[U]], -cT[U][(~wrong[-1][U]) & ng[U]]) if U.any() else None
            J[k][nm] = r
            say(f"    {nm:9s}{' OWN' if nm == loss else '    '}: calibration error at iteration 1 (all puzzles) {r['ece_it1_all']:.3f} | on the unsolved puzzles at 16: calibration error {r['ece_end_unsolved'] if r['ece_end_unsolved'] is None else round(r['ece_end_unsolved'], 3)}, committed {pct(r['com_end_unsolved'])} %, wrong among the committed {pct(r['wrong_among_committed_end_unsolved'])} %, wrong-cell AUC {r['auc_wrong_end_unsolved'] if r['auc_wrong_end_unsolved'] is None else round(r['auc_wrong_end_unsolved'], 3)}")
    say(); say("POST HOC, second read: what the model is SURE of under its OWN normalization (top probability > 0.9), pooled over the empty cells of all puzzles: is it right, and is it ever revised by iteration 16?")
    for k, (path, ema, loss, cls, desc) in MODELS.items():
        f = OUT / f"{k}.npz"
        if not f.exists(): continue
        D = np.load(f, allow_pickle=True); rows = sure_rows(D["logits"], D["sol"], D["puz"] == 0, stablemax9 if loss == "stablemax" else softmax9); J[k]["sure"] = rows
        say(f"  {k} [{cls}; own = {loss}]")
        for r in rows: say(f"     iteration {r['t']}: sure of {pct(r['sure'])} % of the empty cells; of those, wrong {pct(r['sure_wrong'])} %, revised later {pct(r['sure_revised'])} % | the other cells: wrong {pct(r['unsure_wrong'])} %, revised later {pct(r['unsure_revised'])} %")
    (OUT / "posthoc.txt").write_text("\n".join(Ls) + "\n"); (OUT / "posthoc.json").write_text(json.dumps(J, indent=1, default=float))

# ---------------- selftest ----------------
def selftest():
    n = 0
    lg = np.array([5.0] + [0.0] * 8); assert abs(softmax9(lg).max() - 0.94884) < 1e-4 and abs(stablemax9(lg).max() - 6 / 14) < 1e-12; n += 1          # the note's example
    lg = np.array([9.0] + [-9.0] * 8); assert abs(stablemax9(lg).max() - 10 / 10.8) < 1e-12; n += 1
    import jax.numpy as jnp
    from qhrrn2.objective import log_stablemax
    rng = np.random.default_rng(0); x = rng.normal(0, 6, (4, 11)).astype(np.float32); full = np.exp(np.asarray(log_stablemax(jnp.asarray(x)), np.float64))
    ren = full[:, 1:10] / full[:, 1:10].sum(-1, keepdims=True); assert np.abs(ren - stablemax9(x[:, 1:10])).max() < 1e-5; n += 1                      # = the trainer's, renormalized over the digits
    ng = np.zeros((1, 9, 9), bool); ng[0, 0, :4] = True; com = np.zeros((1, 9, 9), bool); com[0, 0, :3] = True; com[0, 5, 5] = True; wrong = np.zeros((1, 9, 9), bool); wrong[0, 0, 0] = True; wrong[0, 0, 3] = True
    assert share(com, ng).tolist() == [0.75] and abs(wrong_among(com, wrong, ng)[0] - 1 / 3) < 1e-12 and np.isnan(wrong_among(np.zeros_like(com), wrong, ng)[0]); n += 1
    preds = np.ones((3, 1, 9, 9), int); preds[1, 0, 0, 0] = 2; preds[2, 0, 0, 1] = 3; h, s = hold_stay(preds, ng); assert h.tolist() == [0.75] and s.tolist() == [0.5]; n += 1
    assert auc([3, 2], [1, 0]) == 1.0 and auc([1], [1]) == 0.5 and auc([], [1]) is None and abs(auc([0, 2], [1, 1]) - 0.5) < 1e-12; n += 1
    assert com_line("softmax", 0.9, [(1, 0.5), (16, 1.0)]) == "    softmax   top probability > 0.9 : it  1  50.0 %  it 16 100.0 %"; n += 1
    T_, B_ = 2, 2; lgs = np.full((T_, B_, 9, 9, 9), -20.0); sol = np.ones((B_, 9, 9), int); ngs = np.zeros((B_, 9, 9), bool); ngs[:, 0, :2] = True
    lgs[..., 0] = 20.0; lgs[0, 1, 0, 0, 0] = 1.0; lgs[0, 1, 0, 0, 1] = 5.0; lgs[0, 1, 0, 0, 2:] = 0.0     # puzzle 1, cell (0,0), iteration 1: digit 2 by a 4-logit margin, wrong; repaired at iteration 2
    S = summarize(lgs, sol, ngs); assert S["free"]["exact_end"] == 1.0 and abs(S["free"]["first_guess_wrong"] - 0.25) < 1e-12 and abs(S["free"]["hold"] - 0.75) < 1e-12; n += 1
    assert S["softmax"]["0.9"][0] == (1, 1.0) and abs(S["stablemax"]["0.9"][0][1] - 0.75) < 1e-12; n += 1   # margin 4 to the second, 5 to the rest: softmax .936 commits, stablemax 6/15 does not
    assert abs(S["softmax"]["wrong_among_committed_it1"] - 0.25) < 1e-12 and S["stablemax"]["wrong_among_committed_it1"] == 0.0; n += 1
    assert all((ROOT / v[0]).exists() for v in MODELS.values()); n += 1
    _lg = np.full((2, 1, 9, 9, 9), -30.0); _lg[..., 0] = 30.0; _lg[0, 0, 0, 0] = 0.0; _lg[0, 0, 0, 0, 1] = 2.0; _sol = np.ones((1, 9, 9), int); _ng = np.zeros((1, 9, 9), bool); _ng[0, 0, :2] = True
    _r = sure_rows(_lg, _sol, _ng, stablemax9)[0]; assert _r["sure"] == 0.5 and _r["sure_wrong"] == 0.0 and _r["sure_revised"] == 0.0 and _r["unsure_wrong"] == 1.0 and _r["unsure_revised"] == 1.0; n += 1   # post hoc helper
    assert abs(ece([0.95, 0.95, 0.55, 0.55], [1, 1, 1, 0]) - (0.5 * 0.05 + 0.5 * 0.05)) < 1e-12 and ece([], []) is None; n += 1   # post hoc helper
    print(f"selftest OK: {n}/{n}")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true"); ap.add_argument("--run", nargs="*"); ap.add_argument("--report", action="store_true"); ap.add_argument("--posthoc", action="store_true"); a = ap.parse_args()
    if a.selftest: return selftest()
    if a.run is not None: run(a.run or list(MODELS))
    if a.report: report()
    if a.posthoc: posthoc()

if __name__ == "__main__":
    main()
