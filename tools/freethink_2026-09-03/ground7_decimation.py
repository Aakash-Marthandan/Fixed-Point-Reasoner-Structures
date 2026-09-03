"""Freethink grounding 7 (CPU, $0): VERIFIED DECIMATION at inference on our trained maps (no retraining). When the cold run
stalls (unsolved at t=64), FIX the M most confident non-given cells as givens (decimate), propagate 32 more steps from the
current state; if the grid is exact -> solved; if a hard contradiction appears (duplicate digits) -> backtrack (undo the
round, fix only the single top cell at its SECOND-best digit); up to R rounds. Diagnostic: how often are the top-M
confident cells correct on the stalled state (safety of decimation)."""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "tools")
import jax, jax.numpy as jnp
from qhrrn2 import episodic as E, grid as G, model as M, sudoku as SU, sudoku_extreme as SX
from qhrrn2.config import Config
import eval_sudoku_extreme as EV
N = int(sys.argv[1]) if len(sys.argv) > 1 else 256; MFIX = 5; ROUNDS = 8; STEPS = 32
d = SX.load_prepared("data/sudoku_extreme/sudoku_extreme_seed0.npz"); Q, A, R = d["test_q"], d["test_a"], d["test_rating"]
ids = SX.stratified_subsample(R, 512, 20260821)[:N]; B = len(ids); out = {}
def violations(pred9):  # duplicate-digit count over rows/cols/boxes (numpy)
    v = np.zeros(len(pred9), int)
    for b in range(len(pred9)):
        g = pred9[b]
        for i in range(9):
            for grp in (g[i, :], g[:, i], g[(i//3)*3:(i//3)*3+3, (i%3)*3:(i%3)*3+3].ravel()):
                vals = grp[grp > 0]; v[b] += len(vals) - len(np.unique(vals))
    return v
def run(name, ckpt, ema):
    saved = E.load_ckpt(ckpt); defaults = Config(); cfg = Config(**{k: type(getattr(defaults, k))(v) for k, v in saved["config"].items()})
    st = saved["state_ema"] if ema else saved["state"]; params = st["model"]; tvj = jnp.asarray(st["table"][0])
    eta, eta_z = (float(v) for v in M.eq_etas(params, cfg)); layout = cfg.sudoku_layout; cv = SU.layout_canvas(layout)
    puz9 = Q[ids].astype(np.int32).copy(); sol9 = A[ids].astype(np.int32); given0 = puz9 != 0
    void = jax.nn.one_hot(jnp.full((cv, cv), G.VOID, jnp.int32), M.VOCAB).transpose(2, 0, 1)
    def canvas(p9): return jnp.asarray(np.stack([SU.place_layout(g.astype(np.int8), layout) for g in p9]), jnp.int32)
    def roll(x_can, y, z, T, t_off):
        for t in range(T):
            first = z is None; tn = min(t + t_off, cfg.T - 1) / max(cfg.T - 1, 1)
            logits, zf = EV._step(cfg, 1.0, float(tn), first)(params, x_can, y, tvj, jnp.zeros(1) if first else z)
            z = zf if first else z + eta_z * (zf - z); p = jax.nn.softmax(logits, axis=-1); y = y + eta * (p.transpose(0, 3, 1, 2) - y)
        pred9 = np.asarray(EV.layout_gather(jnp.argmax(logits, axis=-1), layout)); pred9 = np.where(pred9 == G.VOID, 0, pred9)
        p9 = np.asarray(EV.layout_gather(p, layout))[..., 1:10]; p9 = p9 / np.maximum(p9.sum(-1, keepdims=True), 1e-9)
        return y, z, pred9, p9
    t0 = time.time(); y, z, pred, p9 = roll(canvas(puz9), jnp.broadcast_to(void, (B,) + void.shape), None, 64, 0)
    solved = np.all((pred == sol9).reshape(B, -1), 1); U = ~solved; print(f"{name}: cold @64 {100*solved.mean():.1f}% | stalled {U.sum()}", flush=True)
    # safety diagnostic on the stalled state: are the top-M confident non-given cells correct?
    conf = p9.max(-1); conf[given0] = -1; top = np.argsort(conf.reshape(B, -1), 1)[:, ::-1][:, :MFIX]
    corr_top = np.array([np.mean([(pred[b].ravel()[c] == sol9[b].ravel()[c]) for c in top[b]]) for b in range(B)])
    print(f"   stalled: top-{MFIX} confident cells correct {100*corr_top[U].mean():.1f}% (mean confidence {np.array([conf.reshape(B,-1)[b, top[b]].mean() for b in range(B)])[U].mean():.2f}); solved-set control {100*corr_top[solved].mean():.1f}%", flush=True)
    # verified decimation on the stalled puzzles
    fixed = puz9.copy(); rescued = np.zeros(B, bool); rounds_used = np.full(B, -1); backtracks = np.zeros(B, int); wrong_fix_ever = np.zeros(B, bool)
    yc, zc = y, z
    for r in range(ROUNDS):
        act = U & ~rescued
        if not act.any(): break
        conf = p9.max(-1).copy(); conf[fixed != 0] = -1
        top = np.argsort(conf.reshape(B, -1), 1)[:, ::-1][:, :MFIX]
        trial = fixed.copy(); fixed_cells = {}
        for b in np.where(act)[0]:
            for c in top[b]:
                if conf.reshape(B, -1)[b, c] < 0: continue
                i, j = divmod(int(c), 9); trial[b, i, j] = int(pred[b, i, j]); fixed_cells.setdefault(b, []).append((i, j, int(pred[b, i, j])))
        yc, zc, pred_t, p9_t = roll(canvas(trial), yc, zc, STEPS, cfg.T - 1)
        ex = np.all((pred_t == sol9).reshape(B, -1), 1); viol = violations(pred_t)
        for b in np.where(act)[0]:
            if ex[b]: rescued[b] = True; rounds_used[b] = r + 1; fixed = trial if False else fixed; continue
        # contradiction -> backtrack: undo this round's fixes for that puzzle, fix only the top cell at its second-best digit
        bt = act & (viol > 0) & ~ex
        for b in np.where(bt)[0]:
            backtracks[b] += 1
            if b in fixed_cells:
                i, j, dgt = fixed_cells[b][0]; second = int(np.argsort(p9[b, i, j])[::-1][1]) + 1
                for (ii, jj, _) in fixed_cells[b]: trial[b, ii, jj] = fixed[b, ii, jj]
                trial[b, i, j] = second
        ok = act & ~bt & ~ex
        for b in np.where(act)[0]:
            if b in fixed_cells: wrong_fix_ever[b] |= any(sol9[b, i, j] != dgt for (i, j, dgt) in fixed_cells[b]) and not bt[b]
        fixed = np.where((act[:, None, None]) & (trial != 0), trial, fixed); pred = np.where(act[:, None, None], pred_t, pred); p9 = np.where(act[:, None, None, None], p9_t, p9)
        print(f"   round {r+1}: rescued so far {rescued[U].sum()}/{U.sum()} ({100*rescued[U].mean():.1f}% of stalled); contradictions this round {int(bt.sum())} ({time.time()-t0:.0f}s)", flush=True)
    tot = solved | rescued
    print(f"{name} VERIFIED DECIMATION: cold {100*solved.mean():.1f}% -> {100*tot.mean():.1f}% (rescued {100*rescued[U].mean():.1f}% of stalled; median rounds {np.median(rounds_used[rescued]) if rescued.any() else '-'}; puzzles with >=1 backtrack {int((backtracks>0)[U].sum())}; rescued despite a wrong fix at some round {int((rescued & wrong_fix_ever).sum())}) ({time.time()-t0:.0f}s)", flush=True)
    out[name] = dict(cold=float(solved.mean()), total=float(tot.mean()), rescued_frac_of_stalled=float(rescued[U].mean()), n_stalled=int(U.sum()), top_correct_stalled=float(corr_top[U].mean()), backtracks=int((backtracks > 0)[U].sum()))
run("R0", "runs/pretrainsportC1_R0/ckpt_latest.pkl", True); run("B0vsel", "runs/pretrainsportC1_B0a/ckpt_020000.pkl", False)
Path("runs/analysis/freethink_ground7_decimation_20260903.json").write_text(json.dumps(out)); print("GROUND7-DONE", flush=True)
