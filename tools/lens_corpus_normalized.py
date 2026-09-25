#!/usr/bin/env python3
# Ledger: THE CORPUS UNDER THE TRAINING NORMALIZATION (2026-09-19; registration Documentation/Note_2026-09-19_Corpus_Normalized.md,
# written before any row). DESCRIPTIVE, zero cloud: banked checkpoints on the Mac. Imports the validity lens's selftested helpers; its own
# loop (mirrors suite_ckpt.run_dyn). Saves the evaluator's argmax grid (int8) and the digit logits (float16) of every iteration.
#   Part C  inside the iteration: the readout after each OUTER CYCLE, 16 iterations, the validity lens's 256 puzzles, five models.
#   Part B  a REPRESENTATIVE sample: 512 uniformly drawn test puzzles, 32 iterations, nine models.
#   Part A  the corpus: the banked rows' 128 stratified puzzles, 64 iterations (argmax rows must reproduce the banked ones) + extras.
"""  JAX_PLATFORMS=cpu .venv/bin/python tools/lens_corpus_normalized.py --selftest
  JAX_PLATFORMS=cpu .venv/bin/python tools/lens_corpus_normalized.py --run C|B|A|all        (resume-safe per checkpoint)
  .venv/bin/python tools/lens_corpus_normalized.py --report"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tools"))
import lens_commit_validity as V     # softmax9, stablemax9, share, wrong_among, auc, ece, pct (selftested there)

OUT = ROOT / "runs/analysis/corpus_normalized_20260919"
NPZ = ROOT / "data/sudoku_extreme/sudoku_extreme_seed0.npz"
STRAT_SEED, REP_SEED, BS = 20260821, 20260919, 128
ITERS = (1, 2, 4, 8, 16, 32, 64); OFFS = (-3, -2, -1, 0, 1)
W = "runs/_wladder_pull/"
PART_C = {k: V.MODELS[k][:2] for k in ("C5", "C0", "SA256", "X0", "EQR")}
PART_B = {"C5": ("runs/pretrainchamp_C5/ckpt_046000.pkl", True), "C7": ("runs/_paperfinal_pull/x/runs/pretrainchamp_C7/ckpt_046000.pkl", True),
          "C8": ("runs/_c8x_pull/x/runs/pretrainchamp_C8/ckpt_046000.pkl", True), "C0": ("runs/pretrainchamp_C0/ckpt_016000.pkl", True),
          "SA256": (W + "grids/runs/pretrainchamp_SA256/ckpt_028000.pkl", True), "X0": ("runs/pretrainsportC1_X0/ckpt_050000.pkl", True),
          "EQR": ("runs/field_ckpts/ported/eqr/ckpt_latest.pkl", True), "CGAR": ("runs/field_ckpts/ported/cgar/ckpt_latest.pkl", True),
          "TRMPUB": ("runs/field_ckpts/ported/trmpub/ckpt_latest.pkl", True)}
EXTRA_A = {"champ/C7@46k-vsel": PART_B["C7"], "champ/C8@46k-vsel": PART_B["C8"], "ladder/SA256@28k": PART_B["SA256"],
           "ladder/SA192@30k": (W + "tgz/p1/SA192_ckpt.pkl", True), "ladder/SA128@30k": (W + "tgz/p0/SA128_ckpt.pkl", True),
           "ladder/W256@18k": (W + "grids/runs/pretrainchamp_W256/ckpt_018000.pkl", True), "ladder/W128@30k": (W + "tgz/p0/W128_ckpt.pkl", True),
           "ablate/SA256S": ("runs/_sablate_pull/tgz/p1/SA256S_ckpt.pkl", True), "ablate/SA256L": ("runs/_sablate_pull/tgz/p0/SA256L_ckpt.pkl", True),
           "ablate/SA256O": ("runs/_sablate_pull/tgz/p2/SA256O_ckpt.pkl", True), "finalA/A3@42k": ("runs/pretrainfinalA_A3/ckpt_042000.pkl", True),
           "finalA/A7@22k": ("runs/pretrainfinalA_A7/ckpt_022000.pkl", True), "x5long/X5@660k": ("runs/_x5long_pull/x660/runs/pretrainchamp_X5/ckpt_660000.pkl", True)}
FIRST_A = ("frontier/", "champ/C5@46k", "champ/C7", "champ/C8", "champ/C0@16k", "sportC1/X0", "finalA/A0", "sportC2/X1", "sportC2/X2", "x5long/", "ladder/", "ablate/")

def banked_rows():
    rows = []
    for f in sorted(glob.glob(str(ROOT / "runs/analysis/suite_ckpt_dyn_202609*.jsonl"))):
        rows += [json.loads(l) for l in open(f)]
    return {r["name"]: r for r in rows}

def targets_a():
    B = banked_rows(); T = {n: (r["ckpt"], bool(r["ema"])) for n, r in B.items()}; T.update(EXTRA_A)
    order = sorted(T, key=lambda n: (0 if n.startswith(FIRST_A) else (1 if not n.startswith("champ/") else 2), n)); return [(n, *T[n]) for n in order]

# ---------------- pure helpers (selftested) ----------------
def first_exact(exact):
    """exact (T, B) bool -> (B,) the 1-based first fully-right step, 0 = never."""
    return np.where(exact.any(0), exact.argmax(0) + 1, 0)

def aligned(right_share, sure_share, fe, lo, hi, offs=OFFS):
    """right_share, sure_share (T, B) per-puzzle shares of the empty cells; fe (B,) 1-based. Puzzles with lo <= fe <= hi.
    Returns {off: (mean right, median WRONG, mean sure, n)}."""
    T = right_share.shape[0]; sel = np.where((fe >= lo) & (fe <= hi))[0]; out = {}
    for o in offs:
        idx = fe[sel] + o - 1; ok = (idx >= 0) & (idx < T)
        if not ok.any(): out[o] = (None, None, None, 0); continue
        r = right_share[idx[ok], sel[ok]]; s = sure_share[idx[ok], sel[ok]]; out[o] = (float(r.mean()), float(np.median(1 - r)), float(s.mean()), int(ok.sum()))
    return out

def by_state(right, sure, exact, ng, t):
    """At step t (1-based): pooled over the empty cells of not-yet-solved / already-solved puzzles: (right, sure, n puzzles)."""
    out = {}
    for nm, m in (("unsolved", ~exact[t - 1]), ("solved", exact[t - 1])):
        d = ng[m].sum(); out[nm] = (float((right[t - 1][m] & ng[m]).sum() / d), float((sure[t - 1][m] & ng[m]).sum() / d), int(m.sum())) if m.any() else (None, None, 0)
    return out

def reliability(conf, correct, edges=(0, .3, .5, .7, .9, .99, 1.0001)):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float); rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf >= a) & (conf < b)
        if m.any(): rows.append((a, min(b, 1.0), int(m.sum()), float(conf[m].mean()), float(correct[m].mean())))
    return rows

def summarize(preds, lg, sol, ng, own_name, lo=3):
    """preds (T, B, 9, 9) the evaluator's argmax; lg (T, B, 9, 9, 9) digit logits. Every share is over the EMPTY cells."""
    lg = np.asarray(lg, np.float64); T = len(preds); right = preds == sol[None]; exact = right.all((2, 3)); fe = first_exact(exact)   # ALL 81 cells, as the evaluator scores (audit F06/F20, 2026-09-19)
    own = V.stablemax9 if own_name == "stablemax" else V.softmax9; p_own = own(lg).max(-1); p_soft = V.softmax9(lg).max(-1); nng = np.maximum(ng.sum((1, 2)), 1)
    rs = (right & ng[None]).sum((2, 3)) / nng[None]; J = dict(own=own_name, n=int(len(sol)), T=int(T), solved_end=float(exact[-1].mean()), n_never=int((fe == 0).sum()))
    J["solved_by"] = {str(t): float(((fe > 0) & (fe <= t)).mean()) for t in ITERS if t <= T}
    later, never = fe > 0, fe == 0
    for nm, p in (("own", p_own), ("softmax", p_soft)):
        for tau in (0.5, 0.9):
            sure = (p > tau) & ng[None]; ss = sure.sum((2, 3)) / nng[None]
            J[f"{nm}.{tau}"] = {str(t): dict(all=float(ss[t - 1].mean()), later=float(ss[t - 1][later].mean()) if later.any() else None, never=float(ss[t - 1][never].mean()) if never.any() else None,
                                           wrong_among=float(np.nanmean(V.wrong_among(p[t - 1] > tau, ~right[t - 1], ng)))) for t in ITERS if t <= T}
    sure = (p_own > 0.9) & ng[None]; ss = sure.sum((2, 3)) / nng[None]
    J["aligned"] = {str(o): v for o, v in aligned(rs, ss, fe, lo, T).items()}; J["aligned_lo"] = lo
    J["state"] = {str(t): by_state(right, sure, exact, ng, t) for t in ITERS if t <= T}
    J["ece_it1"] = V.ece(p_own[0][ng], right[0][ng]); J["ece_pooled"] = V.ece(np.concatenate([p_own[t][ng] for t in range(T)]), np.concatenate([right[t][ng] for t in range(T)]))
    J["ece_soft_it1"] = V.ece(p_soft[0][ng], right[0][ng]); J["reliability_pooled"] = reliability(np.concatenate([p_own[t][ng] for t in range(T)]), np.concatenate([right[t][ng] for t in range(T)]))
    if never.any():
        m = ng[never]; J["never"] = dict(n=int(never.sum()), wrong_end=float(((~right[-1][never]) & m).sum() / m.sum()), sure_end=float((sure[-1][never]).sum() / m.sum()),
                                         wrong_among_sure_end=float(((~right[-1][never]) & sure[-1][never]).sum() / max(sure[-1][never].sum(), 1)), ece_end=V.ece(p_own[-1][never][m], right[-1][never][m]),
                                         mean_top_end=float(p_own[-1][never][m].mean()), auc_own=V.auc(-p_own[-1][never][(~right[-1][never]) & m], -p_own[-1][never][right[-1][never] & m]),
                                         auc_soft=V.auc(-p_soft[-1][never][(~right[-1][never]) & m], -p_soft[-1][never][right[-1][never] & m]))
    if later.any(): J["mean_top_end_solved"] = float(p_own[-1][later][ng[later]].mean())
    w1 = 1 - rs[0]; far = w1 > 0.5
    J["first_guess"] = dict(wrong_all=float(w1.mean()), wrong_later=float(w1[later].mean()) if later.any() else None, wrong_never=float(w1[never].mean()) if never.any() else None,
                            far_share=float(far.mean()), far_solved=float(later[far].mean()) if far.any() else None)
    return J

def line_aligned(tag, A):
    g = lambda o, i: A[str(o)][i] if str(o) in A else A[o][i]
    return f"    {tag}: aligned to the first fully-right step f: " + " | ".join(f"f{o:+d} (n {g(o, 3)}): right {V.pct(g(o, 0))} sure {V.pct(g(o, 2))}" for o in OFFS) + f" || median WRONG at f-1: {V.pct(g(-1, 1))} %"

# ---------------- the loops ----------------
def _load(path, ema):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax, jax.numpy as jnp
    from qhrrn2 import episodic as E
    from qhrrn2.config import Config
    saved = E.load_ckpt(str(ROOT / path) if not str(path).startswith("/") else str(path)); d0 = Config(); cfg = Config(**{k: type(getattr(d0, k))(v) for k, v in saved["config"].items()})
    st = saved["state_ema"] if ema else saved["state"]; assert st is not None, f"{path}: no {'EMA' if ema else 'raw'} weights"
    return jax, jnp, cfg, st, int(saved.get("step", -1))

def own_of(cfg, path): return "stablemax" if "field_ckpts/ported" in str(path) else str(cfg.loss_kind)   # the released weights: stablemax per their code

def puzzles(kind, n):
    from qhrrn2 import sudoku_extreme as SX
    d = SX.load_prepared(NPZ); R = d["test_rating"]
    ids = np.asarray(SX.stratified_subsample(R, n, STRAT_SEED)) if kind == "strat" else np.sort(np.random.default_rng(REP_SEED).choice(len(R), n, replace=False))
    return ids, d["test_q"][ids].astype(np.int32), d["test_a"][ids].astype(np.int32)

def run_iters(path, ema, puz, T, tag):
    """The evaluator's loop (mirrors suite_ckpt.run_dyn): returns preds (T, N, 9, 9) int8, digit logits (T, N, 9, 9, 9) float16, meta."""
    jax, jnp, cfg, st, step = _load(path, ema)
    from qhrrn2 import grid as GR, model as M, sudoku as SU
    import eval_sudoku_extreme as EV
    params = st["model"]; tvj = jnp.asarray(st["table"][0]); eta, eta_z = (float(v) for v in M.eq_etas(params, cfg)); lay = cfg.sudoku_layout or "origin"; cv = SU.layout_canvas(lay)
    trm = cfg.cell_kind in ("trm", "dec"); K = 1 if trm else max(1, int(getattr(cfg, "inner_k", 1))); ab = EV.coupled_ab(params, cfg); N = len(puz)
    void = jax.nn.one_hot(jnp.full((cv, cv), GR.VOID, jnp.int32), M.VOCAB).transpose(2, 0, 1); P = np.zeros((T, N, 9, 9), np.int8); L = np.zeros((T, N, 9, 9, 9), np.float16); t_a = time.time()
    for b0 in range(0, N, BS):
        x_can = EV.place_batch(puz[b0:b0 + BS], lay); B = x_can.shape[0]; y = jnp.broadcast_to(void, (B,) + void.shape); z = None
        for t in range(T):
            tn = 0.0 if trm else min(t, cfg.T - 1) / max(cfg.T - 1, 1)
            for _ in range(K):
                first = z is None; logits, zf = EV._step(cfg, 1.0, float(tn), first)(params, x_can, y, tvj, jnp.zeros(1) if first else z); z = zf if first else z + eta_z * (zf - z)
            p = jax.nn.softmax(logits, axis=-1); pT = p.transpose(0, 3, 1, 2); y = (ab[0] * y + ab[1] * pT) if ab is not None else (y + eta * (pT - y))
            pred = np.asarray(EV.layout_gather(jnp.argmax(logits, axis=-1), lay)); P[t, b0:b0 + B] = np.where(pred == GR.VOID, 0, pred).astype(np.int8)
            L[t, b0:b0 + B] = np.clip(np.asarray(EV.layout_gather(logits, lay))[..., 1:10], -6e4, 6e4).astype(np.float16)
        print(f"  {tag}: rows {b0 + B}/{N} ({time.time() - t_a:.0f}s)", flush=True)
    return P, L, dict(cell=cfg.cell_kind, own=own_of(cfg, path), ema=bool(ema), K=K, step=step, ckpt=str(path), wall=round(time.time() - t_a, 1))

def run_cycles(path, ema, puz, n_iter, tag):
    """The readout after EVERY outer cycle (trm / dec only; the cell's own segment, replicated one iteration at a time; eval mode: no noise).
    Returns preds (n_iter * H, N, 9, 9) int8, logits float16, H, meta."""
    jax, jnp, cfg, st, step = _load(path, ema); assert cfg.cell_kind in ("trm", "dec"), cfg.cell_kind; dec = cfg.cell_kind == "dec"
    from qhrrn2 import dec_cell as DC, trm_cell as TC
    C = DC if dec else TC; p = jax.tree_util.tree_map(jnp.asarray, st["model"]["dec" if dec else "trm"]); hw = cfg.canvas * cfg.canvas; lam = cfg.trm_lambda; H = cfg.trm_h_cycles
    stack = (lambda h: DC._stack(p, h, cfg)) if dec else (lambda h: TC._stack(p, h, cfg.trm_puzzle_emb_len))
    stp = lambda z, inj: (lambda Fz: (z + (1.0 - lam) * (Fz - z)) if lam > 0 else Fz)(stack(z + inj))
    def one(x, zH, zL):
        emb = C.embed(p, cfg, x); outs = []
        for _c in range(H):
            for _l in range(cfg.trm_l_cycles): zL = stp(zL, zH + emb)
            zH = stp(zH, zL); outs.append(C.readout(p, cfg, zH, x.shape)[0])
        return jnp.stack(outs), zH, zL
    f = jax.jit(jax.vmap(one)); z0 = C.z0(cfg, hw) if dec else TC.z0(cfg, hw, p=p); N = len(puz)
    P = np.zeros((n_iter * H, N, 9, 9), np.int8); L = np.zeros((n_iter * H, N, 9, 9, 9), np.float16); t_a = time.time()
    for b0 in range(0, N, BS):
        x = puz[b0:b0 + BS]; keep = len(x); xp = jnp.asarray(np.concatenate([x, np.repeat(x[-1:], BS - keep, 0)]) if keep < BS else x)
        zH = jnp.broadcast_to(z0[0], (BS,) + z0[0].shape); zL = jnp.broadcast_to(z0[1], (BS,) + z0[1].shape)
        for t in range(n_iter):
            lg, zH, zL = f(xp, zH, zL); lg = np.asarray(lg)[:keep, :, :, :, 1:10]                       # (B, H, 9, 9, 9)
            for c in range(H): L[t * H + c, b0:b0 + keep] = np.clip(lg[:, c], -6e4, 6e4).astype(np.float16); P[t * H + c, b0:b0 + keep] = (lg[:, c].argmax(-1) + 1).astype(np.int8)
        print(f"  {tag}: rows {b0 + keep}/{N} ({time.time() - t_a:.0f}s)", flush=True)
    return P, L, H, dict(cell=cfg.cell_kind, own=own_of(cfg, path), ema=bool(ema), step=step, ckpt=str(path), wall=round(time.time() - t_a, 1))

def _save(dst, **kw):
    tmp = dst.with_suffix(".tmp.npz"); np.savez_compressed(tmp, **kw); os.replace(tmp, dst)

def run(part):
    OUT.mkdir(parents=True, exist_ok=True)
    if part in ("C", "all"):
        ids, puz, sol = puzzles("strat", 256)
        for k, (path, ema) in PART_C.items():
            dst = OUT / f"C_{k}.npz"
            if dst.exists(): print(f"SKIP {dst.name}"); continue
            P, L, H, meta = run_cycles(path, ema, puz, 16, f"C/{k}"); ref = ROOT / f"runs/analysis/commit_validity_20260919/{k}.npz"; agree = None
            if ref.exists():
                R = np.load(ref, allow_pickle=True); assert (R["ids"] == ids).all(); rp = R["logits"].astype(np.float64).argmax(-1) + 1; ng = puz == 0
                # The gate (2026-09-19, after the first launch): a different compiled kernel is a different compute ROUTE, and unsolved puzzles amplify
                # round-off (measured on C5: 100 % agreement at iteration 1, >= 99.95 % through 4, 64 % on the 9 puzzles unsolved at 16). So the probe
                # must equal the evaluator at iteration 1 and on the puzzles the evaluator has SOLVED; elsewhere the agreement is recorded, not asserted.
                ev_exact = ((rp == sol[None]) | ~ng[None]).all((2, 3)); agree = [float(((P[t * H + H - 1] == rp[t]) & ng).sum() / ng.sum()) for t in range(16)]
                agree_solved = [float(((P[t * H + H - 1] == rp[t]) & ng)[ev_exact[t]].sum() / max(ng[ev_exact[t]].sum(), 1)) for t in range(16)]
                pb_exact = ((P[H - 1::H].astype(np.int64) == sol[None]) | ~ng[None]).all((2, 3)); meta["agree_by_iter"] = agree; meta["agree_on_solved_by_iter"] = agree_solved
                meta["solved16_evaluator_probe"] = [int(ev_exact[-1].sum()), int(pb_exact[-1].sum())]
                if agree[0] <= 0.999 or min(agree_solved[:4]) <= 0.999: print(f"FAILED-GATE C/{k}: iteration-1 agreement {agree[0]:.4f}, on solved puzzles through iteration 4 {min(agree_solved[:4]):.4f}; NOT saved", flush=True); continue
            meta["agree_min"] = None if agree is None else min(agree); _save(dst, preds=P, logits=L, puz=puz, sol=sol, ids=ids, H=H, meta=json.dumps(meta)); print(f"DONE {dst.name} {meta}", flush=True)
    if part in ("B", "all"):
        ids, puz, sol = puzzles("rep", 512)
        for k, (path, ema) in PART_B.items():
            dst = OUT / f"B_{k}.npz"
            if dst.exists(): print(f"SKIP {dst.name}"); continue
            P, L, meta = run_iters(path, ema, puz, 32, f"B/{k}"); _save(dst, preds=P, logits=L, puz=puz, sol=sol, ids=ids, meta=json.dumps(meta)); print(f"DONE {dst.name} {meta}", flush=True)
    if part in ("A", "all"):
        ids, puz, sol = puzzles("strat", 128)
        for name, path, ema in targets_a():
            dst = OUT / ("A_" + name.replace("/", "__").replace("@", "_at_") + ".npz")
            if dst.exists(): print(f"SKIP {dst.name}"); continue
            if not (ROOT / path).exists(): print(f"MISSING {name} {path}", flush=True); continue
            P, L, meta = run_iters(path, ema, puz, 64, f"A/{name}"); meta["name"] = name; _save(dst, preds=P, logits=L, puz=puz, sol=sol, ids=ids, meta=json.dumps(meta)); print(f"DONE {dst.name} {meta}", flush=True)

# ---------------- report ----------------
def report():
    Ls, J = [], {"A": {}, "B": {}, "C": {}}; say = lambda s_="": (Ls.append(s_), print(s_)); B = banked_rows()
    say("THE CORPUS UNDER THE TRAINING NORMALIZATION (descriptive; tools/lens_corpus_normalized.py; registration Note_2026-09-19_Corpus_Normalized.md).")
    say("The fixed start, no halting, no noise. Every share is over the EMPTY cells. OWN = the normalization of the checkpoint's training loss; SURE = OWN top probability > 0.9.")
    for part, title in (("C", "PART C: inside the iteration, at outer-cycle resolution (256 stratified puzzles; steps are OUTER CYCLES, three per iteration)"),
                        ("B", "PART B: a REPRESENTATIVE sample (512 uniformly drawn test puzzles, 32 iterations)"), ("A", "PART A: the corpus (the banked rows' 128 stratified puzzles, 64 iterations)")):
        files = sorted(OUT.glob(f"{part}_*.npz"))
        if not files: continue
        say(); say("== " + title)
        for f in files:
            D = np.load(f, allow_pickle=True); meta = json.loads(str(D["meta"])); ng = D["puz"] == 0; s = summarize(D["preds"].astype(np.int64), D["logits"], D["sol"], ng, meta["own"], lo=4 if part == "C" else 3); s["meta"] = meta
            key = meta.get("name", f.stem[2:]); J[part][key] = s; o9 = s["own.0.9"]; fg = s["first_guess"]
            say(); say(f"  {key} [{meta['cell']}; OWN = {meta['own']}; {'EMA' if meta['ema'] else 'raw'}; step {meta['step']}]" + (f" (the probe against the evaluator: {100 * meta['agree_by_iter'][0]:.2f} % of the empty cells at iteration 1, {100 * min(meta['agree_on_solved_by_iter']):.2f} % at worst on puzzles the evaluator has solved, {100 * meta['agree_min']:.2f} % at worst overall; solved at 16: {meta['solved16_evaluator_probe'][0]} / {meta['solved16_evaluator_probe'][1]})" if meta.get("agree_by_iter") else ""))
            say(f"    solved at the end {V.pct(s['solved_end'])} % (never solved {s['n_never']} of {s['n']}) | solved by step " + ", ".join(f"{t}: {V.pct(v)}" for t, v in s["solved_by"].items()))
            say(f"    first guess wrong: all {V.pct(fg['wrong_all'])} %, later-solved {V.pct(fg['wrong_later'])} %, never-solved {V.pct(fg['wrong_never'])} % | puzzles more than half wrong at step 1: {V.pct(fg['far_share'])} %, of which later solved {V.pct(fg['far_solved'])} %")
            say("    SURE share (OWN 0.9) by step: " + ", ".join(f"{t}: {V.pct(v['all'])}" for t, v in o9.items()) + f" | softmax 0.9 at step 1: {V.pct(s['softmax.0.9']['1']['all'])} % | OWN 0.5 at step 1: {V.pct(s['own.0.5']['1']['all'])} % | wrong among the SURE at step 1: {V.pct(o9['1']['wrong_among'])} %")
            say(line_aligned("SNAP", s["aligned"]))
            say("    not-yet-solved / already-solved puzzles, SURE share by step: " + ", ".join(f"{t}: {V.pct(v['unsolved'][1])} / {V.pct(v['solved'][1])}" for t, v in s["state"].items()))
            say(f"    calibration error OWN: step 1 {s['ece_it1']:.3f}, pooled {s['ece_pooled']:.3f} (softmax at step 1: {s['ece_soft_it1']:.3f}) | reliability pooled (says -> right): " + ", ".join(f"{100 * c:.0f}->{100 * r:.0f}" for _, _, _, c, r in s["reliability_pooled"]))
            if "never" in s:
                nv = s["never"]; say(f"    never-solved puzzles (n {nv['n']}) at the end: wrong {V.pct(nv['wrong_end'])} %, SURE {V.pct(nv['sure_end'])} % (of those wrong {V.pct(nv['wrong_among_sure_end'])} %), mean top probability {V.pct(nv['mean_top_end'])} % (solved: {V.pct(s.get('mean_top_end_solved'))} %), wrong-cell AUC own {nv['auc_own'] if nv['auc_own'] is None else round(nv['auc_own'], 3)} / softmax {nv['auc_soft'] if nv['auc_soft'] is None else round(nv['auc_soft'], 3)}")
            if part == "A" and key in B: say(f"    determinism against the banked row: solved {V.pct(s['solved_end'])} vs {V.pct(B[key]['solved'])} | cells right at step 1 on later-solved: banked {V.pct(B[key]['cells_s'][0])}")
    # ---- the registered predictions
    A = J["A"]; exp_a = [n for n, _, _ in targets_a()]; miss = dict(A=[n for n in exp_a if n not in A], B=[k for k in PART_B if k not in J["B"]], C=[k for k in PART_C if k not in J["C"]])
    say(); say(f"== completeness (audit F20): Part A {len(exp_a) - len(miss['A'])} of {len(exp_a)}, Part B {len(PART_B) - len(miss['B'])} of {len(PART_B)}, Part C {len(PART_C) - len(miss['C'])} of {len(PART_C)}. A universal prediction is scored ONLY when its part is complete; until then the lines below are PARTIAL READS, not letters.")
    tag = lambda part, ok: ("HELD" if ok else "FAILED") if not miss[part] else f"INCOMPLETE ({len(miss[part])} missing; partial read: {'holds so far' if ok else 'ALREADY FAILS'})"
    if A:
        loop = {k: v for k, v in A.items() if v["meta"]["cell"] in ("trm", "dec")}; ctrl = {k: v for k, v in A.items() if v["meta"]["cell"] == "rg"}; mature = {k: v for k, v in loop.items() if v["solved_end"] >= 0.8}
        say(); say(f"== the registered predictions (Part A: {len(loop)} loop checkpoints, {len(mature)} mature, {len(ctrl)} controls)")
        q1 = {k: v["own.0.9"]["1"]["all"] for k, v in loop.items()}; bad = {k: round(100 * x, 1) for k, x in q1.items() if x >= 0.5}
        say(f"  Q1 OWN commitment at step 1 under 50 % on every loop checkpoint: {tag('A', not bad)} (range {V.pct(min(q1.values()))}-{V.pct(max(q1.values()))} %; at or above 50 %: {bad})")
        el = {k: v for k, v in mature.items() if v["aligned"]["0"][3] >= 20}
        if el:
            q2 = {k: v["aligned"]["-1"][1] for k, v in el.items()}; bad = {k: round(100 * x, 1) for k, x in q2.items() if x < 0.2}
            say(f"  Q2 median WRONG at f-1 at least 20 % on every mature loop checkpoint with n >= 20 ({len(el)}): {tag('A', not bad)} (range {V.pct(min(q2.values()))}-{V.pct(max(q2.values()))} %; under 20 %: {bad})")
            bad = {k: (round(100 * v["aligned"]["-1"][2], 1), round(100 * v["aligned"]["0"][2], 1)) for k, v in el.items() if not (v["aligned"]["-1"][2] < 0.15 and v["aligned"]["0"][2] > 0.9)}
            say(f"  Q3 SURE under 15 % at f-1 and over 90 % at f on the same: {tag('A', not bad)} (f-1 range {V.pct(min(v['aligned']['-1'][2] for v in el.values()))}-{V.pct(max(v['aligned']['-1'][2] for v in el.values()))} %, f range {V.pct(min(v['aligned']['0'][2] for v in el.values()))}-{V.pct(max(v['aligned']['0'][2] for v in el.values()))} %; failing: {bad})")
        if mature:
            q4 = {k: v["ece_pooled"] for k, v in mature.items()}; bad = {k: round(x, 3) for k, x in q4.items() if x > 0.06}; say(f"  Q4 pooled OWN calibration error at most 0.06 on every mature loop checkpoint: {tag('A', not bad)} (range {min(q4.values()):.3f}-{max(q4.values()):.3f}; above: {bad})")
        nv = {k: v["never"] for k, v in loop.items() if "never" in v and v["never"]["n"] >= 5}
        if nv:
            bad = {k: round(100 * x["sure_end"], 1) for k, x in nv.items() if x["sure_end"] >= 0.15}; say(f"  Q5 SURE under 15 % on never-solved puzzles at the end ({len(nv)} checkpoints with n >= 5): {tag('A', not bad)} (range {V.pct(min(x['sure_end'] for x in nv.values()))}-{V.pct(max(x['sure_end'] for x in nv.values()))} %; at or above: {bad})")
            bad = {k: round(x["auc_own"], 3) for k, x in nv.items() if x["auc_own"] is not None and x["auc_own"] >= 0.75}; say(f"  Q6 OWN wrong-cell AUC under 0.75 on the same: {tag('A', not bad)} (range {min(x['auc_own'] for x in nv.values() if x['auc_own'] is not None):.3f}-{max(x['auc_own'] for x in nv.values() if x['auc_own'] is not None):.3f}; at or above: {bad})")
        ce = {k: v for k, v in ctrl.items() if v["aligned"]["0"][3] >= 10}
        if ce:
            q9 = {k: v["aligned"]["-1"][1] for k, v in ce.items()}; bad = {k: round(100 * x, 1) for k, x in q9.items() if x >= 0.1}; say(f"  Q9 the controls complete GRADUALLY (median WRONG at f-1 under 10 %; {len(ce)} controls with n >= 10): {tag('A', not bad)} (range {V.pct(min(q9.values()))}-{V.pct(max(q9.values()))} %; at or above 10 %: {bad})")
        both = [k for k in A if k in B]; same = [k for k in both if abs(A[k]["solved_end"] - B[k]["solved"]) < 1e-9]
        say(f"  Q10 determinism: solved at 64 reproduces the banked row exactly on at least 45 of 48: {tag('A', len(same) >= 45)} ({len(same)} of {len(both)}; differing: {[(k, round(100 * A[k]['solved_end'], 2), round(100 * B[k]['solved'], 2)) for k in both if k not in same]})")
    if J["B"]:
        q7a = {k: v["first_guess"]["wrong_later"] for k, v in J["B"].items()}; q7b = {k: v["solved_by"]["2"] for k, v in J["B"].items()}
        say(f"  Q7 Part B: first guess wrong on 20-40 % of later-solved puzzles' empty cells on all: {tag('B', all(0.2 <= x <= 0.4 for x in q7a.values()))} ({', '.join(f'{k} {V.pct(x)}' for k, x in q7a.items())}); at least 40 % solved by step 2 on all: {tag('B', all(x >= 0.4 for x in q7b.values()))} ({', '.join(f'{k} {V.pct(x)}' for k, x in q7b.items())})")
    if J["C"]:
        el = {k: v for k, v in J["C"].items() if v["aligned"]["0"][3] >= 5}
        say(f"  Q8 Part C: at outer-cycle resolution, median WRONG at c-1 at least 10 % on all five: {tag('C', bool(el) and all(v['aligned']['-1'][1] >= 0.1 for v in el.values()))} ({', '.join(f'{k} {V.pct(v['aligned']['-1'][1])}' for k, v in el.items())}); SURE at c-1 under 30 % on all five: {tag('C', bool(el) and all(v['aligned']['-1'][2] < 0.3 for v in el.values()))} ({', '.join(f'{k} {V.pct(v['aligned']['-1'][2])}' for k, v in el.items())})")
    (OUT / "report.txt").write_text("\n".join(Ls) + "\n"); (OUT / "report.json").write_text(json.dumps(J, indent=1, default=float))

# ---------------- selftest ----------------
def selftest():
    n = 0
    ex = np.zeros((4, 3), bool); ex[2:, 0] = True; ex[0, 1] = True; assert first_exact(ex).tolist() == [3, 1, 0]; n += 1
    rs = np.array([[.5, 1, .4], [.6, 1, .4], [1., 1, .5], [1., 1, .5]]); ss = np.array([[0, 1, 0], [.1, 1, 0], [1., 1, 0], [1., 1, 0]]); A = aligned(rs, ss, first_exact(ex), 3, 4)
    assert A[0] == (1.0, 0.0, 1.0, 1) and abs(A[-1][0] - .6) < 1e-12 and abs(A[-1][1] - .4) < 1e-12 and abs(A[-1][2] - .1) < 1e-12 and A[-3][3] == 0 and A[1][3] == 1; n += 1   # only puzzle 0 has f >= 3; f-3 is before step 1
    # two puzzles, three steps. Puzzle 0 (2 empty cells): cell (0,0) wrong and UNSURE at step 1, right from step 2 -> solved at step 2.
    # Puzzle 1 (3 empty cells), never solved: cell (0,1) wrong at every step, UNSURE at steps 1 and 3 and SURE-AND-WRONG at step 2;
    # cell (0,2) right, but at step 2 its logits are (5, 0, ..., 0): softmax 0.949 (sure) and stablemax 0.429 (NOT sure): the two reads must differ.
    T, B = 3, 2; sol = np.ones((B, 9, 9), int); ng = np.zeros((B, 9, 9), bool); ng[0, 0, :2] = True; ng[1, 0, :3] = True; preds = np.ones((T, B, 9, 9), int); preds[0, 0, 0, 0] = 2; preds[:, 1, 0, 1] = 3
    lg = np.full((T, B, 9, 9, 9), -30.0); lg[..., 0] = 30.0; lg[0, 0, 0, 0] = 0.0; lg[0, 0, 0, 0, 1] = 1.0; lg[:, 1, 0, 1] = 0.0; lg[:, 1, 0, 1, 2] = 1.0
    lg[1, 1, 0, 1] = -30.0; lg[1, 1, 0, 1, 2] = 30.0; lg[1, 1, 0, 2] = 0.0; lg[1, 1, 0, 2, 0] = 5.0
    S = summarize(preds, lg, sol, ng, "stablemax"); assert S["solved_end"] == 0.5 and S["n_never"] == 1 and S["solved_by"] == {"1": 0.0, "2": 0.5}; n += 1
    assert abs(S["first_guess"]["wrong_all"] - (1 / 2 + 1 / 3) / 2) < 1e-12 and abs(S["first_guess"]["wrong_later"] - 0.5) < 1e-12 and S["first_guess"]["far_share"] == 0.0; n += 1
    assert abs(S["own.0.9"]["1"]["all"] - (1 / 2 + 2 / 3) / 2) < 1e-12 and S["own.0.9"]["1"]["wrong_among"] == 0.0; n += 1
    assert abs(S["own.0.9"]["2"]["all"] - (1 + 2 / 3) / 2) < 1e-12 and S["softmax.0.9"]["2"]["all"] == 1.0; n += 1                      # kills "own = softmax"
    assert abs(S["own.0.9"]["2"]["wrong_among"] - (0 + 1 / 2) / 2) < 1e-12 and abs(S["softmax.0.9"]["2"]["wrong_among"] - (0 + 1 / 3) / 2) < 1e-12; n += 1   # kills "divide by the empty cells" (own: 1 wrong of 2 sure; 3 empty)
    assert abs(S["never"]["sure_end"] - 2 / 3) < 1e-12 and abs(S["never"]["wrong_end"] - 1 / 3) < 1e-12 and S["never"]["wrong_among_sure_end"] == 0.0; n += 1
    pg = preds.copy(); pg[:, 0, 5, 5] = 9; assert summarize(pg, lg, sol, ng, "stablemax")["solved_end"] == 0.0; n += 1            # a wrong GIVEN is not a solved puzzle (audit F06/F20)
    S2 = summarize(preds, lg, sol, ng, "softmax"); assert S2["own.0.9"]["2"]["all"] == 1.0 and S2["own"] == "softmax"; n += 1                  # a softmax-trained checkpoint's OWN read IS softmax
    st = by_state(preds == sol[None], (V.stablemax9(lg).max(-1) > .9) & ng[None], ((preds == sol[None]) | ~ng[None]).all((2, 3)), ng, 2); assert st["solved"] == (1.0, 1.0, 1) and abs(st["unsolved"][0] - 2 / 3) < 1e-12 and abs(st["unsolved"][1] - 2 / 3) < 1e-12 and st["unsolved"][2] == 1; n += 1
    assert reliability([.95, .95, .2], [1, 0, 0]) == [(0, .3, 1, .2, 0.0), (.9, .99, 2, .95, .5)]; n += 1
    assert line_aligned("SNAP", {str(o): (0.5, 0.25, 0.1, 7) for o in OFFS}).endswith("|| median WRONG at f-1: 25.0 %") and "f+0 (n 7): right 50.0 sure 10.0" in line_aligned("SNAP", {str(o): (0.5, 0.25, 0.1, 7) for o in OFFS}); n += 1
    ta = targets_a(); assert len(ta) == len(banked_rows()) + len(EXTRA_A) and ta[0][0].startswith(FIRST_A); miss = [t for t in ta if not (ROOT / t[1]).exists()]; assert not miss, miss; n += 1
    assert all((ROOT / v[0]).exists() for v in list(PART_B.values()) + list(PART_C.values())); n += 1
    print(f"selftest OK: {n}/{n}")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true"); ap.add_argument("--run", choices=["C", "B", "A", "all"]); ap.add_argument("--report", action="store_true"); a = ap.parse_args()
    if a.selftest: return selftest()
    if a.run: run(a.run)
    if a.report: report()

if __name__ == "__main__":
    main()
