#!/usr/bin/env python3
# Ledger: THE REPAIR-RADIUS LENS (2026-09-17; registration Documentation/Note_2026-09-17_Repair_Radius_Lens.md, written before any row).
# DESCRIPTIVE, no rules; zero cloud: the Mac reads two banked checkpoints (the width-192 symmetric DEC's selected grid and EqR's released
# weights in our evaluator) on the evaluator's 512 rating-stratified test puzzles, exactly as the paper evaluates (no halting, no noise).
#   For a run, E = the empty cells, e_t = the share of E whose top digit is wrong after iteration t (1-based); committed = the top digit's
#   confidence (softmax over the nine digits, as tools/suite_ckpt.py) above 0.9; in conflict = the cell's digit repeats in its row, column
#   or box (givens at their given values).
#   Panel A  across puzzles: the fixed start and one random start z ~ N(0, 1) (draw 0), all 64 iterations on every row (the regression count).
#   Panel B  within puzzle: k random starts per puzzle; rows exact at iteration EXIT_AT are NOT continued (labeled `assumed`; Panel A checks it).
#   Panel C  the corrupted-solution start: z_H = embed_answer(solution with each empty cell resampled over 1..9 w.p. eps), z_L = the fixed
#            buffer (pretrain.field_fpa_loss's construction at a fixed eps); every second puzzle of the 512; the same early exit.
"""  .venv/bin/python tools/lens_repair_radius.py --selftest
  JAX_PLATFORMS=cpu .venv/bin/python tools/lens_repair_radius.py --run --model C5|EQR --panel A|B|C [--n 512] [--k 8] [--out DIR]   (resume-safe per file)
  .venv/bin/python tools/lens_repair_radius.py --report [--out DIR]"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/analysis/repair_radius_20260917"
NPZ = ROOT / "data/sudoku_extreme/sudoku_extreme_seed0.npz"
MODELS = {"C5": ("runs/pretrainchamp_C5/ckpt_046000.pkl", "the width-192 symmetric DEC, seed 0, selected checkpoint (46,000 steps), EMA"),
          "EQR": ("runs/field_ckpts/ported/eqr/ckpt_latest.pkl", "EqR's released weights in our evaluator, EMA"),
          # the 2026-09-18 extension (registered in the note's Extension section before its data): Panel A from the fixed start only
          "A7": ("runs/pretrainfinalA_A7/ckpt_022000.pkl", "the symmetric state ALONE on TRM's loop and recipe (no digit augmentation, no randomized-init training, no anchor examples), width 384, seed 1, re-selected at 22,000, EMA"),
          "X0": ("runs/pretrainsportC1_X0/ckpt_050000.pkl", "TRM's network reproduced on our code under EqR's recipe (5M), EMA"),
          "X5L": ("runs/_x5long_pull/x660/runs/pretrainchamp_X5/ckpt_660000.pkl", "TRM's network at 0.8M under our full recipe at matched training compute, its selected grid (660,000 steps), EMA"),
          "C0": ("runs/pretrainchamp_C0/ckpt_016000.pkl", "the width-384 symmetric DEC, seed 0, selected checkpoint (16,000 steps), EMA")}
PAIRS = (("A7", "X0"), ("C5", "C0"), ("A7", "C0"), ("X5L", "EQR"), ("X0", "EQR"), ("A7", "EQR"))
SOURCES = ("C5", "EQR", "RAND")
STRAT_SEED, MI_SEED, C_SEED = 20260821, 4242, 20260917
T_TOTAL, EXIT_AT, BS, TAU = 64, 16, 128, 0.9
STEPS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
EPS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)
EDGES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0000001)

# ---------------- pure helpers (selftested) ----------------
def err_frac(pred, sol, ng):
    nng = np.maximum(ng.sum((1, 2)), 1)
    return ((pred != sol) & ng).sum((1, 2)) / nng

def conflict_mask(grid):
    """(B, 9, 9) digits (0 = none) -> (B, 9, 9) bool: the cell's digit repeats in its row, column or box."""
    g = np.asarray(grid); B = g.shape[0]; oh = (g[..., None] == np.arange(1, 10)).astype(np.int16)           # (B, 9, 9, 9)
    row = oh.sum(2, keepdims=True); col = oh.sum(1, keepdims=True)
    box = np.repeat(np.repeat(oh.reshape(B, 3, 3, 3, 3, 9).sum((2, 4)), 3, axis=1), 3, axis=2)                  # (B, 3, 3, 9) box counts -> per cell
    peers = np.maximum(np.maximum(np.broadcast_to(row, oh.shape), np.broadcast_to(col, oh.shape)), box)      # max count of the digit over the three units
    return (oh * peers).sum(-1) > 1

def violated_units(grid):
    """(B,) the number of the 27 units holding a repeated digit."""
    g = np.asarray(grid); B = g.shape[0]; oh = (g[..., None] == np.arange(1, 10)).astype(np.int16)
    row = (oh.sum(2) > 1).any(-1).sum(1); col = (oh.sum(1) > 1).any(-1).sum(1)
    box = (oh.reshape(B, 3, 3, 3, 3, 9).sum((2, 4)) > 1).any(-1).sum((1, 2))
    return row + col + box

def auc(pos, neg):
    """P(a positive scores above a negative), ties one half; None when either side is empty."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg): return None
    from scipy.stats import rankdata
    r = rankdata(np.concatenate([pos, neg])); return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

def sign_p(n_pos, n):
    """Two-sided exact binomial (p = 1/2) for n_pos of n."""
    if n == 0: return None
    k = min(n_pos, n - n_pos); return float(min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n))

def dose(e1, ok, edges=EDGES):
    e1, ok = np.asarray(e1, float), np.asarray(ok, bool); rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (e1 >= lo) & (e1 < hi); rows.append(dict(lo=lo, hi=min(hi, 1.0), n=int(m.sum()), ok=(float(ok[m].mean()) if m.any() else None)))
    return rows

def within(e1, ok):
    """e1, ok (P, k): over the puzzles whose restarts disagree, d_p = mean e1 of the failing minus mean e1 of the succeeding restarts."""
    e1, ok = np.asarray(e1, float), np.asarray(ok, bool); mixed = ok.any(1) & ~ok.all(1); d, a = [], []
    for p in np.where(mixed)[0]:
        d.append(e1[p][~ok[p]].mean() - e1[p][ok[p]].mean()); a.append(auc(e1[p][~ok[p]], e1[p][ok[p]]))
    d = np.asarray(d); n_pos, n_neg = int((d > 0).sum()), int((d < 0).sum())
    return dict(n_puzzles=int(len(e1)), n_mixed=int(mixed.sum()), n_all_fail=int((~ok.any(1)).sum()), n_pos=n_pos, n_neg=n_neg,
                share_pos=(n_pos / max(n_pos + n_neg, 1) if len(d) else None), sign_p=sign_p(n_pos, n_pos + n_neg),
                mean_d=(float(d.mean()) if len(d) else None), mean_auc=(float(np.mean(a)) if a else None))

def between_share(x):
    """One-way variance split of x (P, k): the between-puzzle share, with the within-puzzle noise of the puzzle means removed."""
    x = np.asarray(x, float); P, k = x.shape; w = float(x.var(axis=1, ddof=1).mean()); b = max(float(x.mean(axis=1).var(ddof=1)) - w / k, 0.0)
    return b / max(b + w, 1e-30)

def corrupt(sol, ng, eps, rng):
    flip = (rng.random(sol.shape) < eps) & ng; return np.where(flip, rng.integers(1, 10, size=sol.shape), sol).astype(sol.dtype)

def wrong_matched(sol, ng, n_wrong, rng):
    """The solution with exactly n_wrong of its empty cells made wrong (random cells, a random digit other than the solution's)."""
    g = sol.copy(); cells = np.argwhere(ng); pick = cells[rng.choice(len(cells), size=int(min(n_wrong, len(cells))), replace=False)] if n_wrong else np.zeros((0, 2), int)
    for r, c in pick: g[r, c] = (sol[r, c] - 1 + rng.integers(1, 9)) % 9 + 1
    return g

E_CONDS = (("a0_all", 0.0, "all"), ("a0_first", 0.0, "first"), ("a0_after", 0.0, "after"), ("a05_all", 0.5, "all"))

def scale_coupling(tree, alpha):
    """A copy of the DEC's parameter tree with every block's cross-field coupling matrix `fc` scaled by alpha (the sub-layer's norm stays)."""
    if isinstance(tree, dict): return {k: (v * alpha if k == "fc" else scale_coupling(v, alpha)) for k, v in tree.items()}
    if isinstance(tree, (list, tuple)): return type(tree)(scale_coupling(v, alpha) for v in tree)
    return tree

def phase_on(phase, t):
    """Is the ablation active at 0-based iteration t?"""
    return phase == "all" or (phase == "first" and t == 0) or (phase == "after" and t > 0)

def spearman(a, b):
    from scipy.stats import rankdata
    a, b = np.asarray(a, float), np.asarray(b, float); m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3: return None
    ra, rb = rankdata(a[m]), rankdata(b[m]); return float(np.corrcoef(ra, rb)[0, 1])

BANDS = ((0.0, 0.3), (0.3, 0.5), (0.5, 1.0000001))

def bands(e1, ok16, ok64, fe, c2w, visw, visr):
    """The near / middle / far first-guess bands: counts, outcomes, the iterations a solved row needed, the collateral flips, the constraint signal."""
    e1 = np.asarray(e1, float); ok16, ok64 = np.asarray(ok16, bool), np.asarray(ok64, bool); out = []
    for lo, hi in BANDS:
        m = (e1 >= lo) & (e1 < hi); ms = m & ok64; f = lambda a_, mm: (float(np.nanmean(np.asarray(a_, float)[mm])) if mm.any() else None)
        out.append(dict(lo=lo, hi=min(hi, 1.0), n=int(m.sum()), share=float(m.mean()), exact16=f(ok16, m), exact64=f(ok64, m), fe_med=(float(np.median(np.asarray(fe)[ms])) if ms.any() else None),
                        fe_p90=(float(np.percentile(np.asarray(fe)[ms], 90)) if ms.any() else None), c2w_solved=f(c2w, ms), visw=f(visw, m), visr=f(visr, m)))
    return out

def paired_models(ea, eb, oka, okb):
    """Two models on identical puzzles: is the first guess different (paired), and what does each solve where BOTH first guesses are more than half wrong?"""
    from scipy.stats import wilcoxon
    ea, eb = np.asarray(ea, float), np.asarray(eb, float); oka, okb = np.asarray(oka, bool), np.asarray(okb, bool); far = (ea > 0.5) & (eb > 0.5)
    return dict(n=int(len(ea)), mean_a=float(ea.mean()), mean_b=float(eb.mean()), a_lower=int((ea < eb).sum()), a_higher=int((ea > eb).sum()), equal=int((ea == eb).sum()),
                wilcoxon_p=(float(wilcoxon(ea, eb).pvalue) if (ea != eb).any() else None), n_both_far=int(far.sum()),
                a_solves_both_far=(float(oka[far].mean()) if far.any() else None), b_solves_both_far=(float(okb[far].mean()) if far.any() else None))

def band_line(b): return f"    first guess wrong on [{b['lo']:.1f}, {b['hi']:.1f}) of the empty cells: {pct(b['share'])} % of rows (n {b['n']}) | exact at 16 {pct(b['exact16'])} %, at 64 {pct(b['exact64'])} % | median first-exact iteration {b['fe_med']}, 90th percentile {b['fe_p90']}"

def pct(v, d=1): return "  -  " if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{100 * v:.{d}f}"

def dose_line(r): return f"    e1 in [{r['lo']:.1f}, {r['hi']:.1f})  n {r['n']:5d}  exact {pct(r['ok'])} %"

# ---------------- the recorder ----------------
class Rec:
    def __init__(s, puz9, sol9):
        R = len(puz9); s.puz, s.sol, s.ng = puz9, sol9, puz9 == 0; z = lambda dt=float: np.zeros(R, dt)
        s.e = np.full((R, len(STEPS)), np.nan); s.fe = z(np.int32); s.emin = np.full(R, np.inf); s.t_emin = z(np.int32); s.e_end = z(); s.ran = z(np.int32)
        s.ex16 = z(bool); s.ex_end = z(bool); s.ever = z(bool); s.c2w = z(); s.prev = np.zeros((R, 9, 9), bool); s.assumed = z(bool)
        s.c1 = z(); s.w1 = z(); s.cwE = z(); s.visw = np.full(R, np.nan); s.visr = np.full(R, np.nan); s.viol1 = z(np.int32)
    def update(s, rows, t, pred, conf=None):
        ng, sol = s.ng[rows], s.sol[rows]; nng = np.maximum(ng.sum((1, 2)), 1); corr = pred == sol; e = ((~corr) & ng).sum((1, 2)) / nng
        exact = corr.all((1, 2)); s.e_end[rows] = e; s.ex_end[rows] = exact; s.ran[rows] = t + 1
        if (t + 1) in STEPS: s.e[rows, STEPS.index(t + 1)] = e
        s.fe[rows] = np.where((s.fe[rows] == 0) & exact, t + 1, s.fe[rows]); s.ever[rows] |= exact
        better = e < s.emin[rows]; s.t_emin[rows] = np.where(better, t + 1, s.t_emin[rows]); s.emin[rows] = np.where(better, e, s.emin[rows])
        if t: s.c2w[rows] += (s.prev[rows] & ~corr & ng).sum((1, 2)) / nng
        s.prev[rows] = corr
        if t + 1 == EXIT_AT: s.ex16[rows] = exact
        if t == 0:
            com = (conf > TAU) & ng; s.c1[rows] = com.sum((1, 2)) / nng; s.cwE[rows] = (com & ~corr).sum((1, 2)) / nng
            s.w1[rows] = (com & ~corr).sum((1, 2)) / np.maximum(com.sum((1, 2)), 1)
            grid = np.where(ng, pred, s.puz[rows]); cm = conflict_mask(grid); wr, ri = (~corr) & ng, corr & ng; s.viol1[rows] = violated_units(grid)
            s.visw[rows] = np.where(wr.sum((1, 2)) > 0, (cm & wr).sum((1, 2)) / np.maximum(wr.sum((1, 2)), 1), np.nan)
            s.visr[rows] = np.where(ri.sum((1, 2)) > 0, (cm & ri).sum((1, 2)) / np.maximum(ri.sum((1, 2)), 1), np.nan)
    def close_assumed(s, rows):
        """Rows exact at EXIT_AT and not continued: exact at the end BY ASSUMPTION (labeled), e_t = 0 past EXIT_AT."""
        s.assumed[rows] = True
        for i, st in enumerate(STEPS):
            if st > EXIT_AT: s.e[rows, i] = 0.0
    def arrays(s):
        return dict(e=s.e, e1=s.e[:, 0], fe=s.fe, emin=s.emin, t_emin=s.t_emin, e_end=s.e_end, ran=s.ran, ex16=s.ex16, ex64=s.ex_end, ever=s.ever,
                    c2w=s.c2w, assumed=s.assumed, c1=s.c1, w1=s.w1, cwE=s.cwE, visw=s.visw, visr=s.visr, viol1=s.viol1)

class Cap:
    """Captures the readout after each iteration (Panel G: the first guess)."""
    def __init__(s, R): s.pred = np.zeros((R, 9, 9), np.int16)
    def update(s, rows, t, pred, conf=None): s.pred[rows] = pred

# ---------------- the model and the loop (mirrors eval_sudoku_extreme.run_batch / suite_ckpt.run_dyn) ----------------
def _jax():
    os.environ.setdefault("JAX_PLATFORMS", "cpu"); sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tools"))
    import jax, jax.numpy as jnp
    import eval_sudoku_extreme as EV
    from qhrrn2 import episodic as E, model as M, grid as G, sudoku as SU, sudoku_extreme as SX
    from qhrrn2.config import Config
    return jax, jnp, EV, E, M, G, SU, SX, Config

class Model:
    def __init__(s, key):
        jax, jnp, EV, E, M, G, SU, SX, Config = _jax(); s.jax, s.jnp, s.EV, s.M, s.G = jax, jnp, EV, M, G
        saved = E.load_ckpt(str(ROOT / MODELS[key][0])); d0 = Config(); s.cfg = cfg = Config(**{k: type(getattr(d0, k))(v) for k, v in saved["config"].items()})
        assert cfg.cell_kind in ("trm", "dec"), cfg.cell_kind
        st = saved["state_ema"]; assert st is not None; s.params = st["model"]; s.tvj = jnp.asarray(st["table"][0])
        s.eta, s.eta_z = (float(v) for v in M.eq_etas(s.params, cfg)); s.ab = EV.coupled_ab(s.params, cfg); s.lay = getattr(cfg, "sudoku_layout", "origin") or "origin"
        s.shp = tuple(M.carry_shape(cfg)); cv = SU.layout_canvas(s.lay); s.void = jax.nn.one_hot(jnp.full((cv, cv), G.VOID, jnp.int32), M.VOCAB).transpose(2, 0, 1)
        s.key, s.step = key, int(saved.get("step", -1)); s.params_at = None   # Panel E: t -> the parameter tree of iteration t
    def iterate(s, x_can, y, z, t0, t1, rec, rows, keep):
        """Iterations t0..t1-1 on one padded batch; rec is updated for the first `keep` rows (global ids `rows`). Returns (y, z)."""
        jnp, EV = s.jnp, s.EV
        for t in range(t0, t1):
            first = z is None
            logits, zf = EV._step(s.cfg, 1.0, 0.0, first)(s.params if s.params_at is None else s.params_at(t), x_can, y, s.tvj, jnp.zeros(1) if first else z)
            z = zf if first else z + s.eta_z * (zf - z)
            p = s.jax.nn.softmax(logits, axis=-1); pT = p.transpose(0, 3, 1, 2); y = (s.ab[0] * y + s.ab[1] * pT) if s.ab is not None else (y + s.eta * (pT - y))
            pred = np.asarray(EV.layout_gather(jnp.argmax(logits, axis=-1), s.lay)); pred = np.where(pred == s.G.VOID, 0, pred).astype(np.int16)[:keep]
            conf = None
            if t == 0:
                p9 = np.asarray(EV.layout_gather(p, s.lay))[..., 1:10]; conf = (p9 / np.maximum(p9.sum(-1, keepdims=True), 1e-9)).max(-1)[:keep]
            rec.update(rows, t, pred, conf)
        return y, z

def embedder(m):
    """grids (B, 9, 9) -> the start carry (B, 2, ...): z_H = the model's embedding of the grid (embed_answer), z_L = the fixed buffer (Panel C's construction)."""
    jnp = m.jnp; cell = "dec" if m.cfg.cell_kind == "dec" else "trm"
    from qhrrn2 import dec_cell as DC, trm_cell as TC
    pc = m.jax.tree_util.tree_map(jnp.asarray, m.params[cell]); hw = m.cfg.canvas * m.cfg.canvas
    zL0 = np.asarray(DC.z0(m.cfg, hw)[1] if cell == "dec" else TC.z0(m.cfg, hw, p=pc)[1])
    emb = m.jax.jit(m.jax.vmap(lambda g: (DC if cell == "dec" else TC).embed_answer(pc, m.cfg, g)))
    def f(grids):
        zH = np.asarray(emb(jnp.asarray(grids))); return np.stack([zH, np.broadcast_to(zL0, zH.shape)], axis=1)
    return f

def run_rows(m, puz9, sol9, z0_fn, exit_at, tag):
    """All rows through T_TOTAL iterations (exit_at None) or through exit_at, then only the rows not exact there to T_TOTAL."""
    jnp, EV = m.jnp, m.EV; R = len(puz9); rec = Rec(puz9, sol9); left = []; t_a = time.time()
    pad = lambda a, n: np.concatenate([a, np.repeat(a[-1:], n - len(a), axis=0)]) if len(a) < n else a
    for b0 in range(0, R, BS):
        rows = np.arange(b0, min(b0 + BS, R)); keep = len(rows); x_can = EV.place_batch(pad(puz9[rows], BS), m.lay)
        y = jnp.broadcast_to(m.void, (BS,) + m.void.shape); z0 = z0_fn(rows); z = None if z0 is None else jnp.asarray(pad(z0, BS))
        y, z = m.iterate(x_can, y, z, 0, exit_at or T_TOTAL, rec, rows, keep)
        if exit_at:
            un = ~rec.ex16[rows]; rec.close_assumed(rows[~un])
            if un.any(): left.append((rows[un], np.asarray(y)[:keep][un], np.asarray(z)[:keep][un]))
        print(f"  {tag}: rows {b0 + keep}/{R} ({time.time() - t_a:.0f}s)", flush=True)
    if left:
        rows = np.concatenate([l[0] for l in left]); ys = np.concatenate([l[1] for l in left]); zs = np.concatenate([l[2] for l in left])
        for b0 in range(0, len(rows), BS):
            rr = rows[b0:b0 + BS]; keep = len(rr); x_can = EV.place_batch(pad(puz9[rr], BS), m.lay)
            m.iterate(x_can, jnp.asarray(pad(ys[b0:b0 + BS], BS)), jnp.asarray(pad(zs[b0:b0 + BS], BS)), exit_at, T_TOTAL, rec, rr, keep)
            print(f"  {tag}: continued {b0 + keep}/{len(rows)} ({time.time() - t_a:.0f}s)", flush=True)
    return rec.arrays()

def puzzles(n):
    sys.path.insert(0, str(ROOT / "src")); from qhrrn2 import sudoku_extreme as SX
    d = SX.load_prepared(NPZ); ids = np.asarray(SX.stratified_subsample(d["test_rating"], n, STRAT_SEED))
    return ids, d["test_q"][ids].astype(np.int32), d["test_a"][ids].astype(np.int32), np.asarray(d["test_rating"])[ids]

def run(model, panel, n, k, out, starts="both"):
    dst = out / f"{model}_{panel}.npz"
    if dst.exists(): print(f"SKIP {dst.name} (done)"); return
    ids, puz9, sol9, rating = puzzles(n); m = Model(model); EV = m.EV; t0 = time.time(); meta = dict(model=model, ckpt=MODELS[model][0], step=m.step, n=n, t_total=T_TOTAL, exit_at=EXIT_AT)
    ri = lambda P, j: np.stack([EV.mi_z0(MI_SEED, int(ids[p]), int(j), m.shp, 1.0, "gauss") for p in P])
    if panel == "A":
        res = {}
        for name, fn in (("cold", lambda rows: None), ("ri", lambda rows: ri(rows, 0))):
            if starts == "cold" and name != "cold": continue
            a = run_rows(m, puz9, sol9, fn, None, f"{model} A/{name}"); res.update({f"{name}_{kk}": v for kk, v in a.items()})
        np.savez(out / f"{model}_{panel}.tmp.npz", idx=ids, rating=rating, meta=json.dumps(meta), wall=time.time() - t0, **res)
    elif panel == "B":
        P = np.repeat(np.arange(n), k); J = np.tile(np.arange(k), n)
        a = run_rows(m, puz9[P], sol9[P], lambda rows: np.stack([EV.mi_z0(MI_SEED, int(ids[P[r]]), int(J[r]), m.shp, 1.0, "gauss") for r in rows]), EXIT_AT, f"{model} B")
        np.savez(out / f"{model}_{panel}.tmp.npz", idx=ids, rating=rating, puzzle=P, draw=J, k=k, meta=json.dumps(meta), wall=time.time() - t0, **a)
    elif panel == "E":   # cutting the cross-field channel at inference (the note's Side panel E); the DEC only
        assert m.cfg.cell_kind == "dec" and m.cfg.dec_coupling, "Panel E ablates the DEC's field coupling"
        sel = np.arange(0, n, 2); res = {}
        for name, alpha, phase in E_CONDS:
            pa = scale_coupling(m.params, alpha); m.params_at = (lambda t, pa=pa, phase=phase: pa if phase_on(phase, t) else m.params)
            a = run_rows(m, puz9[sel], sol9[sel], lambda rows: None, EXIT_AT, f"{model} E/{name}"); res.update({f"{name}_{kk}": v for kk, v in a.items()})
        m.params_at = None
        np.savez(out / f"{model}_{panel}.tmp.npz", idx=ids, sel=sel, conds=np.asarray([c[0] for c in E_CONDS]), meta=json.dumps(meta), wall=time.time() - t0, **res)
    elif panel == "G":   # the first guess: the fixed-start readout after iteration 1, givens at their given values
        cap = Cap(n); pad = lambda a, nn: np.concatenate([a, np.repeat(a[-1:], nn - len(a), axis=0)]) if len(a) < nn else a
        for b0 in range(0, n, BS):
            rows = np.arange(b0, min(b0 + BS, n)); x_can = EV.place_batch(pad(puz9[rows], BS), m.lay)
            m.iterate(x_can, m.jnp.broadcast_to(m.void, (BS,) + m.void.shape), None, 0, 1, cap, rows, len(rows))
        grid = np.where(puz9 == 0, cap.pred, puz9).astype(np.int32)
        np.savez(out / f"{model}_{panel}.tmp.npz", idx=ids, grid=grid, e1=err_frac(grid, sol9, puz9 == 0), meta=json.dumps(meta), wall=time.time() - t0)
    elif panel == "D":   # handed first guesses: own, the other model's, and a random control matched to EqR's wrong counts
        ng = puz9 == 0; G = {}
        for s_ in ("C5", "EQR"):
            g_ = np.load(out / f"{s_}_G.npz", allow_pickle=True); assert (np.asarray(g_["idx"]) == ids).all(); G[s_] = np.asarray(g_["grid"])
        nw = ((G["EQR"] != sol9) & ng).sum((1, 2))
        G["RAND"] = np.stack([wrong_matched(sol9[p], ng[p], int(nw[p]), np.random.default_rng([C_SEED, int(ids[p]), 77])) for p in range(n)]).astype(np.int32)
        assert (((G["RAND"] != sol9) & ng).sum((1, 2)) == nw).all() and (G["RAND"][~ng] == sol9[~ng]).all()
        P = np.tile(np.arange(n), len(SOURCES)); SRC = np.repeat(np.arange(len(SOURCES)), n); start = np.concatenate([G[s_] for s_ in SOURCES]); e0 = err_frac(start, sol9[P], ng[P]); f = embedder(m)
        a = run_rows(m, puz9[P], sol9[P], lambda rows: f(start[rows]), EXIT_AT, f"{model} D")
        np.savez(out / f"{model}_{panel}.tmp.npz", idx=ids, rating=rating, puzzle=P, source=SRC, sources=np.asarray(SOURCES), e0=e0, meta=json.dumps(meta), wall=time.time() - t0, **a)
    else:
        sel = np.arange(0, n, 2); jnp = m.jnp; cell = "dec" if m.cfg.cell_kind == "dec" else "trm"
        from qhrrn2 import dec_cell as DC, trm_cell as TC
        pc = m.jax.tree_util.tree_map(jnp.asarray, m.params[cell]); hw = m.cfg.canvas * m.cfg.canvas
        zL0 = np.asarray(DC.z0(m.cfg, hw)[1] if cell == "dec" else TC.z0(m.cfg, hw, p=pc)[1])
        emb = m.jax.jit(m.jax.vmap(lambda g: (DC if cell == "dec" else TC).embed_answer(pc, m.cfg, g)))
        P = np.tile(sel, len(EPS)); L = np.repeat(np.arange(len(EPS)), len(sel)); ng = puz9 == 0
        start = np.stack([corrupt(sol9[p], ng[p], EPS[l], np.random.default_rng([C_SEED, int(ids[p]), int(l)])) for p, l in zip(P, L)])
        e0 = err_frac(start, sol9[P], ng[P])
        def z0_fn(rows):
            zH = np.asarray(emb(jnp.asarray(start[rows]))); return np.stack([zH, np.broadcast_to(zL0, zH.shape)], axis=1)
        a = run_rows(m, puz9[P], sol9[P], z0_fn, EXIT_AT, f"{model} C")
        np.savez(out / f"{model}_{panel}.tmp.npz", idx=ids, rating=rating, puzzle=P, level=L, eps=np.asarray(EPS), e0=e0, meta=json.dumps(meta), wall=time.time() - t0, **a)
    os.replace(out / f"{model}_{panel}.tmp.npz", dst); print(f"DONE {dst.name} ({time.time() - t0:.0f}s)", flush=True)

# ---------------- report ----------------
def report(out):
    L, J = [], {}; say = lambda s_="": (L.append(s_), print(s_))
    say("THE REPAIR-RADIUS LENS (descriptive, exploratory; tools/lens_repair_radius.py; registration Note_2026-09-17_Repair_Radius_Lens.md)")
    say(f"  64 iterations, no halting, no noise, EMA weights; the 512 rating-stratified test puzzles (seed {STRAT_SEED}); e_t = share of the empty cells wrong after iteration t")
    for mk, (ck, desc) in MODELS.items():
        say(); say(f"== {mk}: {desc} ({ck}) =="); Jm = J.setdefault(mk, {})
        pa = out / f"{mk}_A.npz"
        if pa.exists():
            D = np.load(pa, allow_pickle=True)
            for stt in ("cold", "ri"):
                if f"{stt}_e1" not in D.files: continue
                g = lambda kk: np.asarray(D[f"{stt}_{kk}"]); e1, ok16, ok64, fe, ever = g("e1"), g("ex16").astype(bool), g("ex64").astype(bool), g("fe"), g("ever").astype(bool)
                S, F = ok64, ~ok64; regress = int((ever & ~ok64).sum()); e = g("e"); q = lambda a_, m_: (float(np.nanmean(a_[m_])) if m_.any() else None)
                r = dict(n=int(len(e1)), exact16=float(ok16.mean()), exact64=float(ok64.mean()), n_fail16=int((~ok16).sum()), n_fail64=int(F.sum()), regress=regress,
                         c1=float(g("c1").mean()), e1_solved=q(e1, S), e1_failed=q(e1, F), w1_solved=q(g("w1"), S), w1_failed=q(g("w1"), F), cwE_solved=q(g("cwE"), S), cwE_failed=q(g("cwE"), F),
                         e_solved={st: q(e[:, i], S) for i, st in enumerate(STEPS)}, e_failed={st: q(e[:, i], F) for i, st in enumerate(STEPS)},
                         fe1=float((fe[S] == 1).mean()) if S.any() else None, fe_med=float(np.median(fe[S])) if S.any() else None, fe_p90=float(np.percentile(fe[S], 90)) if S.any() else None,
                         emin_failed_med=(float(np.median(g("emin")[F])) if F.any() else None), emin_failed_min=(float(g("emin")[F].min()) if F.any() else None),
                         e_end_failed=q(g("e_end"), F), c2w_solved=q(g("c2w"), S),
                         auc16=auc(e1[~ok16], e1[ok16]), auc64=auc(e1[F], e1[S]), dose16=dose(e1, ok16), dose64=dose(e1, ok64),
                         visw_solved=q(g("visw"), S), visr_solved=q(g("visr"), S), visw_failed=q(g("visw"), F), visr_failed=q(g("visr"), F),
                         rho_informed=spearman(e1, g("visw") - g("visr")), viol1_solved=q(g("viol1").astype(float), S), viol1_failed=q(g("viol1").astype(float), F))
                r["far_share"] = float((e1 > 0.5).mean()); r["rho_fe"] = spearman(e1[S], fe[S].astype(float)); r["bands"] = bands(e1, ok16, ok64, fe, g("c2w"), g("visw"), g("visr"))
                Jm[f"A_{stt}"] = r
                say(f"  PANEL A, start = {'the fixed start' if stt == 'cold' else 'one random start'}: n {r['n']} | exact at 16 {pct(r['exact16'])} %, at 64 {pct(r['exact64'])} % ({r['n_fail64']} failed) | depth regressions {regress}")
                say(f"    after iteration 1: committed {pct(r['c1'])} % of the empty cells | wrong share of the empty cells: solved rows {pct(r['e1_solved'])}, failed rows {pct(r['e1_failed'])} | wrong AMONG THE COMMITTED: solved {pct(r['w1_solved'])}, failed {pct(r['w1_failed'])} (committed-and-wrong over the empty cells: {pct(r['cwE_solved'])} / {pct(r['cwE_failed'])})")
                say("    wrong share by iteration, solved rows: " + " ".join(f"{st}:{pct(r['e_solved'][st])}" for st in STEPS))
                say("    wrong share by iteration, failed rows: " + " ".join(f"{st}:{pct(r['e_failed'][st])}" for st in STEPS))
                say(f"    solved rows: exact after iteration 1 {pct(r['fe1'])} %, median first-exact iteration {r['fe_med']}, 90th percentile {r['fe_p90']}; correct cells flipped to wrong along the way {pct(r['c2w_solved'])} % of the empty cells")
                say(f"    failed rows: closest approach min_t e_t median {pct(r['emin_failed_med'])} %, best {pct(r['emin_failed_min'])} %; e at iteration 64 {pct(r['e_end_failed'])} %")
                say(f"    DOSE-RESPONSE on e_1: AUC for failure at 16 {'-' if r['auc16'] is None else f'{r['auc16']:.3f}'} ({r['n_fail16']} failed), at 64 {'-' if r['auc64'] is None else f'{r['auc64']:.3f}'} ({r['n_fail64']} failed)")
                say(f"    first guess more than half wrong on {pct(r['far_share'])} % of rows | Spearman(e_1, first-exact iteration) on solved rows {'-' if r['rho_fe'] is None else f'{r['rho_fe']:.2f}'}"); [say(band_line(x)) for x in r["bands"]]
                say("    in those bands, solved rows' correct-to-wrong flips (share of the empty cells) " + " / ".join(pct(x["c2w_solved"]) for x in r["bands"]) + " %; wrong cells in conflict " + " / ".join(pct(x["visw"]) for x in r["bands"]) + " %, right cells in conflict " + " / ".join(pct(x["visr"]) for x in r["bands"]) + " %")
                say("    exact at 16 by e_1:"); [say(dose_line(x)) for x in r["dose16"]]
                say("    exact at 64 by e_1:"); [say(dose_line(x)) for x in r["dose64"]]
                say(f"    THE CONSTRAINT SIGNAL at iteration 1: wrong cells in conflict {pct(r['visw_solved'])} % (solved rows) / {pct(r['visw_failed'])} % (failed); correct cells in conflict {pct(r['visr_solved'])} / {pct(r['visr_failed'])} %; "
                    f"units with a repeat {r['viol1_solved']:.1f} / {('-' if r['viol1_failed'] is None else f'{r['viol1_failed']:.1f}')} of 27; Spearman(e_1, informedness) {'-' if r['rho_informed'] is None else f'{r['rho_informed']:.2f}'}")
        pb = out / f"{mk}_B.npz"
        if pb.exists():
            D = np.load(pb, allow_pickle=True); k = int(D["k"]); n = len(D["idx"]); sh = lambda a_: np.asarray(a_).reshape(n, k)
            e1, ok16, ok64 = sh(D["e1"]), sh(D["ex16"]).astype(bool), sh(D["ex64"]).astype(bool)
            r = dict(k=k, n=n, exact16=float(ok16.mean()), exact64=float(ok64.mean()), cover16=float(ok16.any(1).mean()), cover64=float(ok64.any(1).mean()), assumed=float(np.asarray(D["assumed"]).mean()),
                     between_share=between_share(e1), w16=within(e1, ok16), w64=within(e1, ok64), auc16=auc(e1[~ok16], e1[ok16]), auc64=auc(e1[~ok64], e1[ok64]))
            Jm["B"] = r
            say(f"  PANEL B, {k} random starts per puzzle ({n} puzzles; rows exact at {EXIT_AT} not continued = {pct(r['assumed'])} % of rows, exact at 64 by assumption): per-restart exact at 16 {pct(r['exact16'])} %, at 64 {pct(r['exact64'])} %; some restart exact at 16 {pct(r['cover16'])} %, at 64 {pct(r['cover64'])} %")
            say(f"    between-puzzle share of the variance of e_1: {r['between_share']:.3f} | pooled AUC of e_1 for failure at 16 {'-' if r['auc16'] is None else f'{r['auc16']:.3f}'}, at 64 {'-' if r['auc64'] is None else f'{r['auc64']:.3f}'}")
            for dd, w in ((16, r["w16"]), (64, r["w64"])):
                say(f"    WITHIN PUZZLE at {dd}: {w['n_mixed']} puzzles whose restarts disagree ({w['n_all_fail']} where every restart fails): failing restarts have the larger e_1 on {w['n_pos']}, the smaller on {w['n_neg']} "
                    f"(share {pct(w['share_pos'])} %, sign test p {'-' if w['sign_p'] is None else f'{w['sign_p']:.3g}'}); mean difference {pct(w['mean_d'])} points; mean per-puzzle AUC {'-' if w['mean_auc'] is None else f'{w['mean_auc']:.3f}'}")
        pc_ = out / f"{mk}_C.npz"
        if pc_.exists():
            D = np.load(pc_, allow_pickle=True); lv = np.asarray(D["level"]); rows = []
            say(f"  PANEL C, the corrupted-solution start ({int((lv == 0).sum())} puzzles per level; same early exit){' [EqR never trained on this start family: off-distribution]' if mk == 'EQR' else ''}:")
            for l, eps in enumerate(np.asarray(D["eps"])):
                m_ = lv == l; rr = dict(eps=float(eps), e0=float(np.asarray(D["e0"])[m_].mean()), e1=float(np.asarray(D["e1"])[m_].mean()), exact16=float(np.asarray(D["ex16"])[m_].mean()), exact64=float(np.asarray(D["ex64"])[m_].mean()), exact1=float((np.asarray(D["fe"])[m_] == 1).mean()),
                                        fe_med=(float(np.median(np.asarray(D["fe"])[m_][np.asarray(D["ex64"])[m_].astype(bool)])) if np.asarray(D["ex64"])[m_].any() else None)); rows.append(rr)
                say(f"    eps {eps:.1f}: wrong share of the handed grid {pct(rr['e0'])} % -> after iteration 1 {pct(rr['e1'])} % | exact after ONE iteration {pct(rr['exact1'])} %, at 16 {pct(rr['exact16'])} %, at 64 {pct(rr['exact64'])} % | median first-exact iteration {rr['fe_med']}")
            Jm["C"] = rows
        pe_ = out / f"{mk}_E.npz"
        if pe_.exists() and pa.exists():
            D = np.load(pe_, allow_pickle=True); A_ = np.load(pa, allow_pickle=True); sel = np.asarray(D["sel"]); Jm["E"] = {}
            ctl = dict(c1=float(np.asarray(A_["cold_c1"])[sel].mean()), e1=float(np.asarray(A_["cold_e1"])[sel].mean()), w1=float(np.asarray(A_["cold_w1"])[sel].mean()), exact16=float(np.asarray(A_["cold_ex16"])[sel].mean()), exact64=float(np.asarray(A_["cold_ex64"])[sel].mean()))
            say(f"  PANEL E, the cross-field coupling scaled at INFERENCE (the trained weights never saw it; {len(sel)} puzzles, the fixed start; same early exit):")
            say(f"    control (Panel A, the same puzzles): committed after iteration 1 {pct(ctl['c1'])} %, first guess wrong {pct(ctl['e1'])} %, wrong among the committed {pct(ctl['w1'])} % | exact at 16 {pct(ctl['exact16'])} %, at 64 {pct(ctl['exact64'])} %"); Jm["E"]["control"] = ctl
            for name, alpha, phase in E_CONDS:
                if f"{name}_e1" not in D.files: continue
                g = lambda kk: np.asarray(D[f"{name}_{kk}"]); ok = g("ex64").astype(bool)
                rr = dict(alpha=alpha, phase=phase, c1=float(g("c1").mean()), e1=float(g("e1").mean()), w1=float(g("w1").mean()), exact16=float(g("ex16").mean()), exact64=float(ok.mean()), e_end_failed=(float(g("e_end")[~ok].mean()) if (~ok).any() else None),
                          visw=float(np.nanmean(g("visw"))), visr=float(np.nanmean(g("visr"))), viol1=float(g("viol1").mean())); Jm["E"][name] = rr
                say(f"    coupling x {alpha:.1f} in {'every iteration' if phase == 'all' else ('iteration 1 only' if phase == 'first' else 'every iteration but the first'):29s}: committed {pct(rr['c1'])} %, first guess wrong {pct(rr['e1'])} %, wrong among the committed {pct(rr['w1'])} %, units with a repeat {rr['viol1']:.1f} | exact at 16 {pct(rr['exact16'])} %, at 64 {pct(rr['exact64'])} % | failed rows end {pct(rr['e_end_failed'])} % wrong")
        pd_ = out / f"{mk}_D.npz"
        if pd_.exists():
            D = np.load(pd_, allow_pickle=True); src = np.asarray(D["source"]); names = [str(x) for x in D["sources"]]; e0 = np.asarray(D["e0"]); ok16, ok64 = np.asarray(D["ex16"]).astype(bool), np.asarray(D["ex64"]).astype(bool); Jm["D"] = {}
            say(f"  PANEL D, {mk} started from a HANDED complete grid (z_H = its embedding of the grid, z_L = the fixed buffer; same early exit); bands by the handed grid's wrong share:")
            for si, sn in enumerate(names):
                m_ = src == si; nat = None
                if sn in MODELS and (out / f"{sn}_A.npz").exists(): nat = np.asarray(np.load(out / f"{sn}_A.npz", allow_pickle=True)["cold_ex64"]).astype(bool)
                rows_ = []
                for lo, hi in BANDS:
                    b_ = (e0[m_] >= lo) & (e0[m_] < hi); rr = dict(lo=lo, hi=min(hi, 1.0), n=int(b_.sum()), e0=(float(e0[m_][b_].mean()) if b_.any() else None), e1=(float(np.asarray(D["e1"])[m_][b_].mean()) if b_.any() else None),
                                                                   exact16=(float(ok16[m_][b_].mean()) if b_.any() else None), exact64=(float(ok64[m_][b_].mean()) if b_.any() else None), source_native64=(float(nat[b_].mean()) if (nat is not None and b_.any()) else None)); rows_.append(rr)
                    say(f"    handed {sn:4s} [{lo:.1f}, {min(hi, 1.0):.1f}): n {rr['n']:3d} | wrong share handed {pct(rr['e0'])} % -> after one iteration {pct(rr['e1'])} % | exact at 16 {pct(rr['exact16'])} %, at 64 {pct(rr['exact64'])} % | the source model's own run on these puzzles, exact at 64: {pct(rr['source_native64'])} %")
                Jm["D"][sn] = dict(bands=rows_, exact64=float(ok64[m_].mean()), exact16=float(ok16[m_].mean()))
    pcs = [out / f"{mk}_C.npz" for mk in MODELS if (out / f"{mk}_C.npz").exists()]
    if pcs:   # the constraint signal of the HANDED grids (no model: the corruption is recomputed from its seeds) beside the models' own first guesses at a matched error
        D = np.load(pcs[0], allow_pickle=True); ids, puz9, sol9, _ = puzzles(len(D["idx"])); assert (ids == np.asarray(D["idx"])).all(); ng = puz9 == 0; rows = []
        say(); say("== RANDOM ERRORS AGAINST THE MODELS' OWN: the constraint signal of the handed grids of Panel C (no model involved) ==")
        for l, eps in enumerate(np.asarray(D["eps"])):
            if eps == 0: continue
            P = np.asarray(D["puzzle"])[np.asarray(D["level"]) == l]
            g = np.stack([corrupt(sol9[p], ng[p], float(eps), np.random.default_rng([C_SEED, int(ids[p]), int(l)])) for p in P]); assert np.allclose(err_frac(g, sol9[P], ng[P]), np.asarray(D["e0"])[np.asarray(D["level"]) == l])
            cm = conflict_mask(np.where(ng[P], g, puz9[P])); wr, ri = (g != sol9[P]) & ng[P], (g == sol9[P]) & ng[P]
            rr = dict(eps=float(eps), e0=float(err_frac(g, sol9[P], ng[P]).mean()), visw=float((cm & wr).sum() / max(wr.sum(), 1)), visr=float((cm & ri).sum() / max(ri.sum(), 1))); rows.append(rr)
            say(f"    eps {eps:.1f}: wrong share {pct(rr['e0'])} % | wrong cells in conflict {pct(rr['visw'])} %, right cells in conflict {pct(rr['visr'])} %")
        J["handed_signal"] = rows
    pa, pb = out / "C5_A.npz", out / "EQR_A.npz"
    if pa.exists() and pb.exists():
        A, B = np.load(pa, allow_pickle=True), np.load(pb, allow_pickle=True); assert (np.asarray(A["idx"]) == np.asarray(B["idx"])).all()
        say(); say("== THE TWO MODELS ON IDENTICAL PUZZLES (a = C5, b = EQR; Panel A) =="); J["paired"] = {}
        for stt in ("cold", "ri"):
            for dd in (16, 64):
                q = paired_models(A[f"{stt}_e1"], B[f"{stt}_e1"], A[f"{stt}_ex{dd}"], B[f"{stt}_ex{dd}"]); J["paired"][f"{stt}_{dd}"] = q
            q = J["paired"][f"{stt}_64"]; q16 = J["paired"][f"{stt}_16"]
            say(f"  start = {'the fixed start' if stt == 'cold' else 'one random start'}: mean e_1 {pct(q['mean_a'])} vs {pct(q['mean_b'])} %; a lower on {q['a_lower']}, higher on {q['a_higher']}, equal on {q['equal']} (Wilcoxon p {q['wilcoxon_p']:.3g}) | "
                f"where BOTH first guesses are more than half wrong (n {q['n_both_far']}): a solves {pct(q['a_solves_both_far'])} %, b {pct(q['b_solves_both_far'])} % at 64 ({pct(q16['a_solves_both_far'])} / {pct(q16['b_solves_both_far'])} % at 16)")
    J["pairs"] = {}
    for a_, b_ in PAIRS:
        fa, fb = out / f"{a_}_A.npz", out / f"{b_}_A.npz"
        if not (fa.exists() and fb.exists()): continue
        A, B = np.load(fa, allow_pickle=True), np.load(fb, allow_pickle=True); assert (np.asarray(A["idx"]) == np.asarray(B["idx"])).all()
        if not J["pairs"]: say(); say("== MORE PAIRS ON IDENTICAL PUZZLES (the fixed start; Panel A) ==")
        q = paired_models(A["cold_e1"], B["cold_e1"], A["cold_ex64"], B["cold_ex64"]); q16 = paired_models(A["cold_e1"], B["cold_e1"], A["cold_ex16"], B["cold_ex16"]); J["pairs"][f"{a_}_{b_}"] = dict(d64=q, d16=q16)
        say(f"  a = {a_}, b = {b_}: mean e_1 {pct(q['mean_a'])} vs {pct(q['mean_b'])} %; a lower on {q['a_lower']}, higher on {q['a_higher']}, equal on {q['equal']} (Wilcoxon p {q['wilcoxon_p']:.3g}) | where BOTH first guesses are more than half wrong (n {q['n_both_far']}): a solves {pct(q['a_solves_both_far'])} %, b {pct(q['b_solves_both_far'])} % at 64 ({pct(q16['a_solves_both_far'])} / {pct(q16['b_solves_both_far'])} % at 16)")
    (out / "report.txt").write_text("\n".join(L) + "\n"); (out / "report.json").write_text(json.dumps(J, indent=1, default=float))

# ---------------- selftest ----------------
def selftest():
    n = 0
    sol = np.arange(81).reshape(1, 9, 9) % 9 + 1; ng = np.zeros((1, 9, 9), bool); ng[0, 0, :4] = True; pred = sol.copy(); pred[0, 0, 0] = 9 if sol[0, 0, 0] != 9 else 1
    assert err_frac(pred, sol, ng).tolist() == [0.25]; n += 1
    g = np.zeros((1, 9, 9), int); g[0, 0, 0] = 5; g[0, 0, 8] = 5; g[0, 4, 4] = 5; g[0, 5, 5] = 5; g[0, 8, 0] = 7     # a row repeat (5,5) and a box repeat (5 at (4,4),(5,5)); 7 alone
    cm = conflict_mask(g); assert cm[0, 0, 0] and cm[0, 0, 8] and cm[0, 4, 4] and cm[0, 5, 5] and not cm[0, 8, 0] and cm.sum() == 4; n += 1
    assert violated_units(g).tolist() == [2]; n += 1
    g2 = np.zeros((1, 9, 9), int); g2[0, 1, 3] = 2; g2[0, 7, 3] = 2; assert conflict_mask(g2).sum() == 2 and violated_units(g2).tolist() == [1]; n += 1          # a column repeat
    assert auc([3, 4], [1, 2]) == 1.0 and auc([1], [1]) == 0.5 and auc([], [1]) is None; n += 1
    assert sign_p(8, 8) == 2 / 256 and sign_p(4, 8) == 1.0 and sign_p(0, 0) is None; n += 1
    d = dose([0.05, 0.15, 0.15, 0.95], [1, 1, 0, 0]); assert d[0]["n"] == 1 and d[0]["ok"] == 1.0 and d[1]["n"] == 2 and d[1]["ok"] == 0.5 and d[-1]["n"] == 1 and d[-1]["ok"] == 0.0 and d[2]["ok"] is None; n += 1
    assert dose_line(d[1]) == "    e1 in [0.1, 0.2)  n     2  exact 50.0 %" and dose_line(d[2]) == "    e1 in [0.2, 0.3)  n     0  exact   -   %"; n += 1      # the printed strings
    w = within([[.2, .6], [.5, .1], [.3, .3], [.9, .9]], [[1, 0], [1, 0], [1, 1], [0, 0]])
    assert w["n_mixed"] == 2 and w["n_all_fail"] == 1 and w["n_pos"] == 1 and w["n_neg"] == 1 and w["share_pos"] == 0.5 and abs(w["mean_d"]) < 1e-12 and w["mean_auc"] == 0.5; n += 1
    assert between_share([[1.0, 1.0], [3.0, 3.0]]) == 1.0 and between_share([[1.0, 3.0], [3.0, 1.0]]) == 0.0; n += 1
    r = np.random.default_rng(0); c0 = corrupt(sol[0], ng[0], 0.0, r); c1 = corrupt(sol[0], ng[0], 1.0, r)
    assert (c0 == sol[0]).all() and (c1[~ng[0]] == sol[0][~ng[0]]).all() and c1.min() >= 1 and c1.max() <= 9; n += 1
    assert abs(spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1) < 1e-12 and spearman([1, np.nan], [1, 2]) is None; n += 1
    assert pct(0.12345) == "12.3" and pct(None) == "  -  " and pct(float("nan")) == "  -  "; n += 1
    rec = Rec(np.where(ng, 0, sol)[[0, 0]], sol[[0, 0]]); rows = np.arange(2); p_bad = np.stack([pred[0], sol[0]]); conf = np.ones((2, 9, 9))
    rec.update(rows, 0, p_bad, conf); rec.update(rows, 1, np.stack([sol[0], sol[0]]))
    a = rec.arrays(); assert a["e1"].tolist() == [0.25, 0.0] and a["fe"].tolist() == [2, 1] and a["w1"].tolist() == [0.25, 0.0] and a["c1"].tolist() == [1.0, 1.0] and a["ex64"].all() and a["emin"].tolist() == [0.0, 0.0]; n += 1
    bb = bands([0.1, 0.4, 0.6, 0.9], [1, 1, 0, 0], [1, 1, 1, 0], [1, 3, 20, 0], [0.0, 0.1, 0.5, 0.2], [1.0, 0.5, 0.4, 0.3], [0.0, 0.2, 0.4, 0.5])
    assert [b["n"] for b in bb] == [1, 1, 2] and bb[2]["exact16"] == 0.0 and bb[2]["exact64"] == 0.5 and bb[2]["fe_med"] == 20.0 and bb[2]["c2w_solved"] == 0.5 and bb[0]["visw"] == 1.0; n += 1
    assert band_line(bb[2]) == "    first guess wrong on [0.5, 1.0) of the empty cells: 50.0 % of rows (n 2) | exact at 16 0.0 %, at 64 50.0 % | median first-exact iteration 20.0, 90th percentile 20.0"; n += 1
    pm = paired_models([0.6, 0.7, 0.2, 0.1], [0.7, 0.6, 0.2, 0.3], [1, 1, 1, 1], [0, 1, 1, 1])
    assert pm["a_lower"] == 2 and pm["a_higher"] == 1 and pm["equal"] == 1 and pm["n_both_far"] == 2 and pm["a_solves_both_far"] == 1.0 and pm["b_solves_both_far"] == 0.5; n += 1
    wm = wrong_matched(sol[0], ng[0], 3, np.random.default_rng(1)); assert int(((wm != sol[0]) & ng[0]).sum()) == 3 and (wm[~ng[0]] == sol[0][~ng[0]]).all() and wm.min() >= 1 and wm.max() <= 9; n += 1
    assert (wrong_matched(sol[0], ng[0], 0, np.random.default_rng(1)) == sol[0]).all() and int(((wrong_matched(sol[0], ng[0], 99, np.random.default_rng(1)) != sol[0]) & ng[0]).sum()) == 4; n += 1
    cp = Cap(2); cp.update(np.array([1]), 0, pred[:1]); assert (cp.pred[1] == pred[0]).all() and (cp.pred[0] == 0).all(); n += 1
    tr = {"dec": {"blocks": [{"fc": np.ones((2, 2)), "mlp": {"down": np.ones(2)}}, {"fc": 2 * np.ones((2, 2))}], "lm_head": np.ones(2)}}
    sc = scale_coupling(tr, 0.5); assert sc["dec"]["blocks"][0]["fc"].tolist() == [[.5, .5], [.5, .5]] and sc["dec"]["blocks"][1]["fc"][0, 0] == 1.0 and sc["dec"]["blocks"][0]["mlp"]["down"].tolist() == [1, 1] and tr["dec"]["blocks"][0]["fc"][0, 0] == 1.0; n += 1
    assert [phase_on("first", t) for t in (0, 1)] == [True, False] and [phase_on("after", t) for t in (0, 1)] == [False, True] and phase_on("all", 5); n += 1
    assert all(a_ in MODELS and b_ in MODELS for a_, b_ in PAIRS) and SOURCES == ("C5", "EQR", "RAND"); n += 1
    assert EXIT_AT in STEPS and T_TOTAL in STEPS and STEPS[0] == 1 and EPS[0] == 0.0 and EPS[-1] == 1.0; n += 1
    print(f"selftest OK: {n}/{n}")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true"); ap.add_argument("--run", action="store_true"); ap.add_argument("--report", action="store_true")
    ap.add_argument("--model", choices=list(MODELS)); ap.add_argument("--panel", choices=["A", "B", "C", "G", "D", "E"]); ap.add_argument("--starts", choices=["both", "cold"], default="both"); ap.add_argument("--n", type=int, default=512); ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out", default=None, help="output dir (default runs/analysis/repair_radius_20260917); smokes write elsewhere so the resume-skip never keeps a smoke file")
    a = ap.parse_args(); out = Path(a.out) if a.out else OUT
    if a.selftest: return selftest()
    out.mkdir(parents=True, exist_ok=True)
    if a.run: run(a.model, a.panel, a.n, a.k, out, a.starts)
    if a.report: report(out)

if __name__ == "__main__":
    main()
