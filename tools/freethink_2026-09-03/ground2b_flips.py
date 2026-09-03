"""Freethink grounding 2b (CPU, $0): REVISION DYNAMICS — per outer step, the fraction of non-given cells whose argmax
changes (flip rate), split into flips that land on the correct digit vs on a wrong digit; and 'guess' cells (confident,
wrong at step 1) that are later corrected. Solved vs unsolved. strat-84."""
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
ids = np.concatenate([sel[(R[sel] >= qs[b]) & (R[sel] < qs[b+1])][:12] for b in range(8)]); B = len(ids); out = {}
def run(name, ckpt, ema):
    saved = E.load_ckpt(ckpt); defaults = Config(); cfg = Config(**{k: type(getattr(defaults, k))(v) for k, v in saved["config"].items()})
    st = saved["state_ema"] if ema else saved["state"]; params = st["model"]; tvj = jnp.asarray(st["table"][0])
    trm = cfg.cell_kind == "trm"; eta, eta_z = (float(v) for v in M.eq_etas(params, cfg)); layout = cfg.sudoku_layout; cv = SU.layout_canvas(layout)
    puz9 = Q[ids].astype(np.int32); sol9 = np.asarray(A[ids].astype(np.int32)); ng = puz9 == 0
    x_can = jnp.asarray(np.stack([SU.place_layout(g.astype(np.int8), layout) for g in puz9]), jnp.int32)
    void = jax.nn.one_hot(jnp.full((cv, cv), G.VOID, jnp.int32), M.VOCAB).transpose(2, 0, 1); y = jnp.broadcast_to(void, (B,) + void.shape); z = None
    prev = None; flips_c, flips_w, ex_t = [], [], []; t0 = time.time(); first_pred = None
    for t in range(64):
        first = z is None; t_norm = 0.0 if trm else min(t, cfg.T - 1) / max(cfg.T - 1, 1)
        logits, zf = EV._step(cfg, 1.0, float(t_norm), first)(params, x_can, y, tvj, jnp.zeros(1) if first else z)
        z = zf if first else z + eta_z * (zf - z); p = jax.nn.softmax(logits, axis=-1); y = y + eta * (p.transpose(0, 3, 1, 2) - y)
        pred9 = np.asarray(EV.layout_gather(jnp.argmax(logits, axis=-1), layout)); pred9 = np.where(pred9 == G.VOID, 0, pred9)
        ex_t.append(np.all((pred9 == sol9).reshape(B, -1), axis=1))
        if first_pred is None: first_pred = pred9.copy()
        if prev is not None:
            ch = (pred9 != prev) & ng
            flips_c.append((ch & (pred9 == sol9)).sum((1, 2)) / np.maximum(ng.sum((1, 2)), 1)); flips_w.append((ch & (pred9 != sol9)).sum((1, 2)) / np.maximum(ng.sum((1, 2)), 1))
        prev = pred9
    s = ex_t[-1]; fc = np.stack(flips_c); fw = np.stack(flips_w)
    wrong1 = (first_pred != sol9) & ng & (first_pred != 0); corrected = wrong1 & (prev == sol9)
    windows = [(1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64)]
    def w(arr, mask): return [float(arr[a-1:b-1, mask].sum(0).mean()) if mask.any() else None for a, b in windows]
    o = dict(solved=float(s.mean()), flips_correct_solved=w(fc, s), flips_wrong_solved=w(fc * 0 + fw, s), flips_correct_unsolved=w(fc, ~s), flips_wrong_unsolved=w(fw, ~s),
             step1_wrong_filled_frac_solved=float((wrong1.sum((1, 2)) / np.maximum(ng.sum((1, 2)), 1))[s].mean()), step1_wrong_later_corrected_frac_solved=float((corrected.sum((1, 2)) / np.maximum(wrong1.sum((1, 2)), 1))[s].mean()),
             step1_wrong_filled_frac_unsolved=float((wrong1.sum((1, 2)) / np.maximum(ng.sum((1, 2)), 1))[~s].mean()) if (~s).any() else None,
             step1_wrong_later_corrected_frac_unsolved=float((corrected.sum((1, 2)) / np.maximum(wrong1.sum((1, 2)), 1))[~s].mean()) if (~s).any() else None, wall=round(time.time() - t0, 1))
    print(f"{name}: solved {100*o['solved']:.1f}% | cumulative flips per non-given cell in windows {windows}:", flush=True)
    print(f"   solved:   to-correct {[round(100*v,1) for v in o['flips_correct_solved']]}  to-wrong {[round(100*v,1) for v in o['flips_wrong_solved']]}", flush=True)
    print(f"   unsolved: to-correct {[round(100*v,1) if v is not None else None for v in o['flips_correct_unsolved']]}  to-wrong {[round(100*v,1) if v is not None else None for v in o['flips_wrong_unsolved']]}", flush=True)
    print(f"   step-1 WRONG non-given digits (guesses): solved {100*o['step1_wrong_filled_frac_solved']:.1f}% of cells, of which later corrected {100*o['step1_wrong_later_corrected_frac_solved']:.1f}% | unsolved {o['step1_wrong_filled_frac_unsolved'] and round(100*o['step1_wrong_filled_frac_unsolved'],1)}%, corrected {o['step1_wrong_later_corrected_frac_unsolved'] and round(100*o['step1_wrong_later_corrected_frac_unsolved'],1)}% ({o['wall']}s)", flush=True)
    out[name] = o
run("R0", "runs/pretrainsportC1_R0/ckpt_latest.pkl", True); run("B0vsel", "runs/pretrainsportC1_B0a/ckpt_020000.pkl", False); run("X0", "runs/pretrainsportC1_X0/ckpt_latest.pkl", True)
Path("runs/analysis/freethink_ground2b_flips_20260903.json").write_text(json.dumps(out)); print("GROUND2B-DONE", flush=True)
