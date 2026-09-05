# Ledger: FINAL PHASE — REFERENCE DECODERS on the Sudoku constraint code (Plan_2026-09-05_FinalPhase §5 / §6.5;
# 2026-09-05; the decoder lens's ladder). Three classical decoders on the same erasure channel as the learned
# cells (a puzzle = a codeword with 81 - givens cells erased): (1) PEELING = naked + hidden singles to a fixpoint
# (the propagation decoder; the lens's 11.1 % reference); (2) SUM-PRODUCT BP on the factor graph (81 cell
# variables x 9 values; 27 all-different factors) with the EXACT all-different factor-to-variable message
# (the permanent of the 8x8 minor of the incoming-message matrix; the non-negative subset DP, vectorized), NO
# damping (damping keeps stale mass and leaves BP confidently wrong even on singles-solvable puzzles: measured
# 2026-09-05; undamped, exact zeros propagate = generalized arc consistency inside sum-product), up to --iters
# iterations, decoded by the argmax of the beliefs; (3) BP + DECIMATION (BPD) = guided decimation
# without backtracking: run BP; if the argmax grid is not a valid completion, COMMIT the most polarized undecided
# cell to its argmax and re-run; stop on a valid grid or a contradiction (an empty candidate set / a zero belief).
# Maximum-likelihood decoding solves 100 % (unique solutions). Output: per-puzzle solved flags for each decoder
# on a rating-stratified slice of the 20k scan set (identical puzzles to the learned cells' scans: the stall-set
# overlaps are exact on the slice), solve rates by rating band, and the logistic g50 of each decoder — the
# decoder-class LADDER peeling < BP < (learned cells) < BPD < field < ML. Descriptive (analysis-time, no rules).
"""
  .venv/bin/python tools/reference_decoders.py --n 2000 [--idx-from runs/sxscan_psportC2W0/records_all.npz]
  -> runs/analysis/reference_decoders.{txt,json,npz}
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qhrrn2 import sudoku_extreme as SX   # noqa: E402

UNITS = []   # 27 units x 9 cell indices
for r in range(9): UNITS.append([r * 9 + c for c in range(9)])
for c in range(9): UNITS.append([r * 9 + c for r in range(9)])
for br in range(3):
    for bc in range(3): UNITS.append([(3 * br + i) * 9 + (3 * bc + j) for i in range(3) for j in range(3)])
UNITS = np.asarray(UNITS)                                    # (27, 9)
CELL_UNITS = [[u for u in range(27) if i in UNITS[u]] for i in range(81)]

# minor index arrays: for cell k and value v, the 8 other rows / 8 other columns
_RIDX = np.array([[r for r in range(9) if r != k] for k in range(9)])      # (9, 8)
_CIDX = np.array([[c for c in range(9) if c != v] for v in range(9)])      # (9, 8)
# each cell's 3 (unit, position) slots, and for each (unit, position) the other two slots of that cell
CU = np.zeros((81, 3), int); CK = np.zeros((81, 3), int)
for i in range(81):
    slots = [(u, int(np.where(UNITS[u] == i)[0][0])) for u in CELL_UNITS[i]]
    CU[i] = [u for u, _ in slots]; CK[i] = [k for _, k in slots]
OU = np.zeros((27, 9, 2), int); OK_ = np.zeros((27, 9, 2), int)
for u in range(27):
    for k in range(9):
        i = UNITS[u, k]; others = [(uu, kk) for uu, kk in zip(CU[i], CK[i]) if uu != u]
        OU[u, k] = [o[0] for o in others]; OK_[u, k] = [o[1] for o in others]


# subset-DP permanent tables: for j in 0..7, the subsets containing j and their complements without j
_WITH = [np.array([S for S in range(256) if S & (1 << j)]) for j in range(8)]
_WITHOUT = [w ^ (1 << j) for j, w in enumerate(_WITH)]


def perm8(M):
    """Permanents of a batch of 8x8 NON-NEGATIVE matrices M (N, 8, 8) by the subset dynamic program
    f_k(S) = sum_{j in S} M[k, j] f_{k-1}(S \\ {j}) — only additions and products of non-negative
    numbers, so exact zeros stay exact (Ryser's alternating sum leaves +-1e-15 residues that leak
    eliminated values back into the messages: found 2026-09-05 on the reference-decoder build)."""
    N = M.shape[0]
    f = np.zeros((N, 256)); f[:, 0] = 1.0
    for k in range(8):
        g = np.zeros((N, 256))
        for j in range(8):
            g[:, _WITH[j]] += M[:, k, j:j + 1] * f[:, _WITHOUT[j]]
        f = g
    return f[:, 255]


def valid_complete(g):
    if (g == 0).any(): return False
    for u in UNITS:
        if len(set(g[u].tolist())) != 9: return False
    return True


def peel(puz):
    """Naked + hidden singles to a fixpoint. Returns (grid, contradiction)."""
    g = puz.astype(np.int64).copy()
    cand = np.ones((81, 9), bool)
    for i in range(81):
        if g[i]: cand[i] = False; cand[i, g[i] - 1] = True
    def eliminate():
        for u in UNITS:
            for i in u:
                if g[i]:
                    for j in u:
                        if j != i: cand[j, g[i] - 1] = False
    changed = True
    while changed:
        changed = False; eliminate()
        for i in range(81):
            if g[i] == 0:
                nc = cand[i].sum()
                if nc == 0: return g, True
                if nc == 1: g[i] = int(np.argmax(cand[i])) + 1; changed = True
        eliminate()
        for u in UNITS:
            for v in range(9):
                cells = [i for i in u if g[i] == 0 and cand[i, v]]
                if not cells and not any(g[i] == v + 1 for i in u): return g, True
                if len(cells) == 1 and not any(g[i] == v + 1 for i in u):
                    g[cells[0]] = v + 1; cand[cells[0]] = False; cand[cells[0], v] = True; changed = True
    return g, False


def _prior(puz, fixed=None):
    given = puz.astype(np.int64).copy()
    if fixed is not None: given = np.where(fixed > 0, fixed, given)
    prior = np.full((81, 9), 1.0 / 9)
    for i in np.nonzero(given)[0]: prior[i] = 0.0; prior[i, given[i] - 1] = 1.0
    return prior, given


FLOOR = 1e-30


def _floor(m):
    """Keep EXACT zeros (sound eliminations from one-hot messages) but stop positive entries from
    underflowing through the 8-fold products of the permanents (a positive message at 1e-88 ×
    8 → 0.0 in float64 = a false elimination; found 2026-09-05)."""
    m = np.where(m > 0, np.maximum(m, FLOOR), 0.0)
    return m / m.sum(-1, keepdims=True)


def bp_iterate(prior, mf, iters, damp=0.0, tol=1e-5):
    """Sum-product with exact all-different factors, fully vectorized. mf (27, 9, 9) factor->variable messages
    (warm start). Returns (mf, converged, contradiction)."""
    conv = False
    for _ in range(iters):
        # variable -> factor: prior x the OTHER two units' messages to that cell (no division)
        mv = prior[UNITS] * mf[OU[..., 0], OK_[..., 0]] * mf[OU[..., 1], OK_[..., 1]]        # (27, 9, 9)
        ssum = mv.sum(-1, keepdims=True)
        if (ssum <= 0).any(): return mf, False, True
        mv = _floor(mv / ssum)
        # factor -> variable: permanents of the (other cells x other values) minors, all units at once
        minors = mv[:, _RIDX[:, None, :, None], _CIDX[None, :, None, :]]                    # (27, 9 k, 9 v, 8, 8)
        p = perm8(minors.reshape(-1, 8, 8)).reshape(27, 9, 9)
        psum = p.sum(-1, keepdims=True)
        if (psum <= 0).any(): return mf, False, True
        new = _floor(p / psum)
        delta = float(np.abs(new - mf).max())
        mf = damp * mf + (1 - damp) * new
        if delta < tol: conv = True; break
    return mf, conv, False


def beliefs(prior, mf):
    bel = prior * mf[CU[:, 0], CK[:, 0]] * mf[CU[:, 1], CK[:, 1]] * mf[CU[:, 2], CK[:, 2]]
    s = bel.sum(1, keepdims=True)
    return bel / np.maximum(s, 1e-300), bool((s <= 0).any())


def bp_decode(puz, iters=40):
    prior, given = _prior(puz)
    mf, conv, contra = bp_iterate(prior, np.full((27, 9, 9), 1.0 / 9), iters)
    if contra: return False, conv, True
    bel, contra = beliefs(prior, mf)
    g = np.where(given > 0, given, bel.argmax(1) + 1)
    return valid_complete(g), conv, contra


def bpd_decode(puz, iters=40, iters_round=5, max_commits=81):
    """Guided decimation, no backtracking: BP; commit the most polarized undecided cell; re-run BP warm-started
    for iters_round iterations; stop on a valid grid or a contradiction."""
    fixed = np.zeros(81, np.int64); commits = 0
    mf = np.full((27, 9, 9), 1.0 / 9); its = iters
    while commits <= max_commits:
        prior, given = _prior(puz, fixed)
        mf, conv, contra = bp_iterate(prior, mf, its); its = iters_round
        if contra: return False, commits, "contradiction"
        bel, contra = beliefs(prior, mf)
        if contra: return False, commits, "contradiction"
        g = np.where(given > 0, given, bel.argmax(1) + 1)
        if valid_complete(g): return True, commits, "solved"
        free = given == 0
        if not free.any(): return False, commits, "stuck"
        pol = np.where(free, bel.max(1), -1.0); i = int(np.argmax(pol))
        fixed[i] = int(bel[i].argmax()) + 1; commits += 1
    return False, commits, "stuck"


def logistic_g50(g, y, lo=17, hi=35):
    g = np.asarray(g, float); y = np.asarray(y, float)
    if len(g) < 50 or y.mean() <= 0 or y.mean() >= 1: return None
    mu, sd = g.mean(), g.std() + 1e-9; x = (g - mu) / sd; a = b = 0.0
    for _ in range(40):
        p = 1 / (1 + np.exp(-(a + b * x))); w = p * (1 - p) + 1e-6
        G = np.array([np.sum(y - p), np.sum((y - p) * x)]); H = np.array([[np.sum(w), np.sum(w * x)], [np.sum(w * x), np.sum(w * x * x)]]) + 1e-6 * np.eye(2)
        d = np.linalg.solve(H, G); a += d[0]; b += d[1]
        if np.max(np.abs(d)) < 1e-8: break
    if b <= 0.05: return None
    g50 = (-a / b) * sd + mu; return float(g50) if lo <= g50 <= hi else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=str(ROOT / "data/sudoku_extreme/sudoku_extreme_seed0.npz"))
    ap.add_argument("--idx-from", default=str(ROOT / "runs/sxscan_psportC2W0/records_all.npz"), help="the scan's idx set (identical puzzles)")
    ap.add_argument("--n", type=int, default=500); ap.add_argument("--seed", type=int, default=20260905)
    ap.add_argument("--iters", type=int, default=40); ap.add_argument("--out", default=str(ROOT / "runs/analysis/reference_decoders"))
    a = ap.parse_args(); t0 = time.time()
    d = SX.load_prepared(a.npz); Q, A, R = d["test_q"], d["test_a"], d["test_rating"]
    pool = np.load(a.idx_from, allow_pickle=True)["idx"] if Path(a.idx_from).exists() else np.arange(len(Q))
    sub = SX.stratified_subsample(R[pool], a.n, a.seed); idx = np.sort(pool[sub])
    giv = (Q[idx] != 0).reshape(len(idx), -1).sum(1); rat = R[idx]
    res = dict(peel=np.zeros(len(idx), bool), bp=np.zeros(len(idx), bool), bpd=np.zeros(len(idx), bool),
               bp_conv=np.zeros(len(idx), bool), bpd_commits=np.zeros(len(idx), np.int32), bpd_end=np.empty(len(idx), object))
    for n, i in enumerate(idx):
        puz = Q[i].reshape(-1).astype(np.int64); sol = A[i].reshape(-1).astype(np.int64)
        g, contra = peel(puz); res["peel"][n] = (not contra) and np.array_equal(g, sol)
        ok, conv, _ = bp_decode(puz, a.iters); res["bp"][n] = ok; res["bp_conv"][n] = conv
        ok, k, end = bpd_decode(puz, a.iters); res["bpd"][n] = ok; res["bpd_commits"][n] = k; res["bpd_end"][n] = end
        if (n + 1) % 100 == 0:
            print(f"  {n+1}/{len(idx)}  peel {100*res['peel'][:n+1].mean():.1f}  bp {100*res['bp'][:n+1].mean():.1f}  bpd {100*res['bpd'][:n+1].mean():.1f}  ({time.time()-t0:.0f}s)", flush=True)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    bands = [(0, 0), (1, 9), (10, 29), (30, 59), (60, 10**9)]
    L = [f"== REFERENCE DECODERS on {len(idx)} rating-stratified scan puzzles (seed {a.seed}; BP iters {a.iters}, undamped, exact all-different factors) ==",
         "decoder | solved | by rating band 0 / 1-9 / 10-29 / 30-59 / 60+ | search-class yield (rating > 0) | g50 (givens)"]
    summ = {}
    for k in ("peel", "bp", "bpd"):
        y = res[k]; by = [float(y[(rat >= lo) & (rat <= hi)].mean()) if ((rat >= lo) & (rat <= hi)).any() else float("nan") for lo, hi in bands]
        yld = float(y[rat > 0].mean()); g50 = logistic_g50(giv, y)
        summ[k] = dict(solved=float(y.mean()), bands=by, yield_search=yld, g50=g50)
        L.append(f"  {k:5s} | {100*y.mean():5.1f} | " + " / ".join(f"{100*b:4.1f}" for b in by) + f" | {100*yld:5.1f} | {'none' if g50 is None else f'{g50:.1f}'}")
    L.append(f"  BP converged on {100*res['bp_conv'].mean():.1f} % of puzzles; BPD ends: " + ", ".join(f"{e} {int((res['bpd_end'] == e).sum())}" for e in ("solved", "contradiction", "stuck")) +
             f"; median commits to solve {int(np.median(res['bpd_commits'][res['bpd']])) if res['bpd'].any() else '-'}")
    L.append(f"  ({time.time()-t0:.0f}s)")
    Path(str(out) + ".txt").write_text("\n".join(L) + "\n"); print("\n".join(L))
    Path(str(out) + ".json").write_text(json.dumps(dict(n=len(idx), seed=a.seed, iters=a.iters, summary=summ), indent=1))
    np.savez(str(out) + ".npz", idx=idx, givens=giv, rating=rat, peel=res["peel"], bp=res["bp"], bpd=res["bpd"], bp_conv=res["bp_conv"], bpd_commits=res["bpd_commits"])


if __name__ == "__main__":
    main()
