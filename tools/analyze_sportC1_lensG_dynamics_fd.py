"""LENS G3 fallback for the field cell: Jacobian spectral radius of the outer-step map by FINITE-DIFFERENCE power
iteration (autodiff through the trm cell returned NaN), at the t=16 and t=64 cold endpoints; plus per-step residuals."""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "tools")
import jax, jax.numpy as jnp
from qhrrn2 import episodic as E, grid as G, model as M, sudoku as SU, sudoku_extreme as SX
from qhrrn2.config import Config
import eval_sudoku_extreme as EV
N_PER_OCT = int(sys.argv[1]) if len(sys.argv) > 1 else 12; ITERS = 8; EPS = 1e-3
d = SX.load_prepared("data/sudoku_extreme/sudoku_extreme_seed0.npz"); Q, A, R = d["test_q"], d["test_a"], d["test_rating"]
sel = SX.stratified_subsample(R, 512, 20260821); qs = np.quantile(R[sel], np.linspace(0, 1, 9)); qs[-1] += 1
ids = np.concatenate([sel[(R[sel] >= qs[b]) & (R[sel] < qs[b+1])][:N_PER_OCT] for b in range(8)])
saved = E.load_ckpt("runs/pretrainsportC1_X0/ckpt_latest.pkl"); defaults = Config()
cfg = Config(**{k: type(getattr(defaults, k))(v) for k, v in saved["config"].items()}); st = saved["state_ema"]
params = st["model"]; tvj = jnp.asarray(st["table"][0]); layout = cfg.sudoku_layout; cv = SU.layout_canvas(layout)
puz9 = Q[ids].astype(np.int32); sol9 = jnp.asarray(A[ids].astype(np.int32)); B = len(ids)
x_can = jnp.asarray(np.stack([SU.place_layout(g.astype(np.int8), layout) for g in puz9]), jnp.int32)
void = jax.nn.one_hot(jnp.full((cv, cv), G.VOID, jnp.int32), M.VOCAB).transpose(2, 0, 1); y = jnp.broadcast_to(void, (B,) + void.shape)
step_first = EV._step(cfg, 1.0, 0.0, True); step = EV._step(cfg, 1.0, 0.0, False)
def g(z): return step(params, x_can, y, tvj, z)[1]
def nrm(v): return jnp.sqrt(jnp.sum(v.reshape(B, -1) ** 2, 1))
def bcast(s, like): return s.reshape(-1, *[1] * (like.ndim - 1))
z = None; ex_t, res_t, lam = [], [], {}; t0 = time.time()
for t in range(64):
    first = z is None
    logits, zf = (step_first if first else step)(params, x_can, y, tvj, jnp.zeros(1) if first else z)
    res = jnp.mean(jnp.abs(zf - (z if z is not None else zf)), axis=tuple(range(1, zf.ndim))); z = zf
    pred9 = EV.layout_gather(jnp.argmax(logits, axis=-1), layout).astype(jnp.int32); pred9 = jnp.where(pred9 == G.VOID, 0, pred9)
    ex_t.append(np.asarray(jnp.all((pred9 == sol9).reshape(B, -1), axis=1))); res_t.append(np.asarray(res))
    if (t + 1) in (16, 64):
        gz = g(z); v = jax.random.normal(jax.random.PRNGKey(0), z.shape); v = v / bcast(nrm(v), v)
        zs = bcast(EPS * jnp.sqrt(jnp.mean(z.reshape(B, -1) ** 2, 1)) + 1e-8, z)
        hist = []
        for it in range(ITERS):
            jv = (g(z + zs * v) - gz) / zs; n_ = nrm(jv); hist.append(np.asarray(n_)); v = jv / bcast(n_ + 1e-30, jv)
        lam[t + 1] = hist[-1]
        print(f"  X0 FD t={t+1}: lambda median {np.median(hist[-1]):.3f} p90 {np.percentile(hist[-1],90):.3f} (iter drift {np.median(np.abs(hist[-1]-hist[-2])):.3g}) ({time.time()-t0:.0f}s)", flush=True)
ex_t = np.stack(ex_t); res_t = np.stack(res_t); solved = ex_t[-1]; rat = R[ids]
fe = np.where(ex_t.any(0), ex_t.argmax(0), -1)
o = dict(solved16=float(ex_t[15].mean()), solved64=float(solved.mean()), exact_vs_step=ex_t.mean(1).tolist(),
         first_exact_median_solved=float(np.median(fe[solved])), resid_solved_median_by_step=np.median(res_t[:, solved], 1).tolist(),
         resid_unsolved_median_by_step=(np.median(res_t[:, ~solved], 1).tolist() if (~solved).any() else None),
         resid_t64_solved=res_t[-1, solved].tolist(), resid_t64_unsolved=res_t[-1, ~solved].tolist(),
         lam16=lam[16].tolist(), lam64=lam[64].tolist(), solved64_mask=solved.tolist(), rating=rat.tolist(),
         lam64_solved_median=float(np.median(lam[64][solved])), lam64_unsolved_median=(float(np.median(lam[64][~solved])) if (~solved).any() else None),
         unsolved_converged=int(np.sum(res_t[-1, ~solved] <= np.median(res_t[-1, solved]))), n_unsolved=int((~solved).sum()))
print(f"X0 FD: solved@16 {100*o['solved16']:.1f} @64 {100*o['solved64']:.1f} | first_exact median {o['first_exact_median_solved']} | resid t64 solved {np.median(res_t[-1, solved]):.3g} unsolved {np.median(res_t[-1, ~solved]) if (~solved).any() else float('nan'):.3g} | lambda64 solved {o['lam64_solved_median']:.3f} unsolved {o['lam64_unsolved_median']} | converged-wrong {o['unsolved_converged']}/{o['n_unsolved']}", flush=True)
Path("runs/analysis/sportC1_lensG_dynamics_X0fd_20260903.json").write_text(json.dumps(o)); print("X0-FD-DONE")
