"""Freethink grounding 2 (CPU, $0): PROPAGATION VELOCITY and READOUT ENTROPY along cold trajectories (strat-84):
cells correct per outer step and the mean per-cell entropy of the softmax readout, split solved/unsolved — the
'measurement collapse' question (does our map commit early on the puzzles it fails?) and the per-step propagation rate."""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "tools")
import jax, jax.numpy as jnp
from qhrrn2 import episodic as E, grid as G, model as M, sudoku as SU, sudoku_extreme as SX
from qhrrn2.config import Config
import eval_sudoku_extreme as EV
d = SX.load_prepared("data/sudoku_extreme/sudoku_extreme_seed0.npz"); Q, A, R = d["test_q"], d["test_a"], d["test_rating"]
sel = SX.stratified_subsample(R, 512, 20260821); qs = np.quantile(R[sel], np.linspace(0, 1, 9)); qs[-1] += 1
ids = np.concatenate([sel[(R[sel] >= qs[b]) & (R[sel] < qs[b+1])][:12] for b in range(8)]); B = len(ids)
out = {}
def run(name, ckpt, ema):
    saved = E.load_ckpt(ckpt); defaults = Config(); cfg = Config(**{k: type(getattr(defaults, k))(v) for k, v in saved["config"].items()})
    st = saved["state_ema"] if ema else saved["state"]; params = st["model"]; tvj = jnp.asarray(st["table"][0])
    trm = cfg.cell_kind == "trm"; eta, eta_z = (float(v) for v in M.eq_etas(params, cfg)); layout = cfg.sudoku_layout; cv = SU.layout_canvas(layout)
    puz9 = Q[ids].astype(np.int32); sol9 = jnp.asarray(A[ids].astype(np.int32)); mask_given = jnp.asarray(puz9 != 0)
    x_can = jnp.asarray(np.stack([SU.place_layout(g.astype(np.int8), layout) for g in puz9]), jnp.int32)
    void = jax.nn.one_hot(jnp.full((cv, cv), G.VOID, jnp.int32), M.VOCAB).transpose(2, 0, 1); y = jnp.broadcast_to(void, (B,) + void.shape); z = None
    cells_t, ent_t, ex_t, conf_wrong_t = [], [], [], []; t0 = time.time()
    for t in range(64):
        first = z is None; t_norm = 0.0 if trm else min(t, cfg.T - 1) / max(cfg.T - 1, 1)
        logits, zf = EV._step(cfg, 1.0, float(t_norm), first)(params, x_can, y, tvj, jnp.zeros(1) if first else z)
        z = zf if first else z + eta_z * (zf - z); p = jax.nn.softmax(logits, axis=-1); y = y + eta * (p.transpose(0, 3, 1, 2) - y)
        pred9 = EV.layout_gather(jnp.argmax(logits, axis=-1), layout).astype(jnp.int32); pred9 = jnp.where(pred9 == G.VOID, 0, pred9)
        corr = (pred9 == sol9); ex = jnp.all(corr.reshape(B, -1), axis=1)
        # per-cell entropy of the readout over the 9 digit classes (digits 1..9 = vocab ids 1..9 in the 9x9 gather), on non-given cells
        p9 = EV.layout_gather(p, layout) if p.ndim == 4 else p   # (B,9,9,V)
        pd = p9[..., 1:10]; pd = pd / jnp.maximum(pd.sum(-1, keepdims=True), 1e-9); ent = -(pd * jnp.log(pd + 1e-12)).sum(-1) / np.log(9)
        ng = ~mask_given; ent_ng = (ent * ng).sum((1, 2)) / jnp.maximum(ng.sum((1, 2)), 1)
        conf = jnp.max(pd, -1); wrong_conf = ((conf > 0.9) & ~corr & ng).sum((1, 2)) / jnp.maximum(ng.sum((1, 2)), 1)
        cells_t.append(np.asarray((corr & ng).sum((1, 2)) / jnp.maximum(ng.sum((1, 2)), 1))); ent_t.append(np.asarray(ent_ng)); ex_t.append(np.asarray(ex)); conf_wrong_t.append(np.asarray(wrong_conf))
    cells_t = np.stack(cells_t); ent_t = np.stack(ent_t); ex_t = np.stack(ex_t); cw = np.stack(conf_wrong_t); s = ex_t[-1]
    steps = [1, 2, 4, 8, 16, 32, 64]
    o = dict(solved64=float(s.mean()), cells_solved=[float(cells_t[i-1, s].mean()) for i in steps], cells_unsolved=[float(cells_t[i-1, ~s].mean()) if (~s).any() else None for i in steps],
             ent_solved=[float(ent_t[i-1, s].mean()) for i in steps], ent_unsolved=[float(ent_t[i-1, ~s].mean()) if (~s).any() else None for i in steps],
             confwrong_solved=[float(cw[i-1, s].mean()) for i in steps], confwrong_unsolved=[float(cw[i-1, ~s].mean()) if (~s).any() else None for i in steps], wall=round(time.time()-t0, 1))
    print(f"{name}: solved {100*o['solved64']:.1f}% | non-given cells correct by step {steps}: solved {[round(100*v) for v in o['cells_solved']]} unsolved {[round(100*v) if v is not None else None for v in o['cells_unsolved']]}", flush=True)
    print(f"    readout entropy (per non-given cell, /log9): solved {[round(v,2) for v in o['ent_solved']]} unsolved {[round(v,2) if v is not None else None for v in o['ent_unsolved']]}", flush=True)
    print(f"    confidently-WRONG non-given cells (p>.9, wrong): solved {[round(100*v,1) for v in o['confwrong_solved']]} unsolved {[round(100*v,1) if v is not None else None for v in o['confwrong_unsolved']]} ({o['wall']}s)", flush=True)
    out[name] = o
run("R0", "runs/pretrainsportC1_R0/ckpt_latest.pkl", True); run("B0vsel", "runs/pretrainsportC1_B0a/ckpt_020000.pkl", False); run("X0", "runs/pretrainsportC1_X0/ckpt_latest.pkl", True)
Path("runs/analysis/freethink_ground2_trajectories_20260903.json").write_text(json.dumps(out)); print("GROUND2-DONE")
