"""LENS G3 (analysis-time, descriptive, CPU, $0): the field cell's (X0) and our cell's (R0) COLD dynamics read step by
step on a rating-stratified test subsample: exactness vs outer step (propagation vs one-shot), the convergence
residual per step for solved vs unsolved puzzles, the Jacobian spectral radius (power iteration) of the outer-step
map at the t=64 endpoint (contractive fixed point? spurious attractor?) and at t=16."""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "tools")
import jax, jax.numpy as jnp
from qhrrn2 import episodic as E, grid as G, model as M, sudoku as SU, sudoku_extreme as SX
from qhrrn2.config import Config
import eval_sudoku_extreme as EV

N_PER_OCT = int(sys.argv[1]) if len(sys.argv) > 1 else 12
T_TOTAL = 64; POWER_ITERS = 10
d = SX.load_prepared("data/sudoku_extreme/sudoku_extreme_seed0.npz"); Q, A, R = d["test_q"], d["test_a"], d["test_rating"]
sel = SX.stratified_subsample(R, 512, 20260821)
qs = np.quantile(R[sel], np.linspace(0, 1, 9)); qs[-1] += 1
ids = np.concatenate([sel[(R[sel] >= qs[b]) & (R[sel] < qs[b+1])][:N_PER_OCT] for b in range(8)])
print(f"puzzles {len(ids)} ratings octile edges {np.round(qs[:-1]).astype(int).tolist()}", flush=True)
out = {"ids": ids.tolist(), "rating": R[ids].tolist(), "arms": {}}

def load(ckpt, ema):
    saved = E.load_ckpt(ckpt); defaults = Config()
    cfg = Config(**{k: type(getattr(defaults, k))(v) for k, v in saved["config"].items()})
    st = saved["state_ema"] if ema else saved["state"]
    return cfg, st["model"], jnp.asarray(st["table"][0])

def run(name, ckpt, ema):
    cfg, params, tvj = load(ckpt, ema); trm = cfg.cell_kind == "trm"
    eta, eta_z = (float(v) for v in M.eq_etas(params, cfg)); layout = cfg.sudoku_layout or "origin"; cv = SU.layout_canvas(layout)
    puz9 = Q[ids].astype(np.int32); sol9 = jnp.asarray(A[ids].astype(np.int32))
    x_can = jnp.asarray(np.stack([SU.place_layout(g.astype(np.int8), layout) for g in puz9]), jnp.int32)
    void = jax.nn.one_hot(jnp.full((cv, cv), G.VOID, jnp.int32), M.VOCAB).transpose(2, 0, 1); B = len(ids)
    y = jnp.broadcast_to(void, (B,) + void.shape); z = None
    ex_t, res_t, lam = [], [], {}
    t0 = time.time()
    def step_fn(t_norm, first):
        f = EV._step(cfg, 1.0, float(t_norm), first)
        return f
    for t in range(T_TOTAL):
        t_norm = 0.0 if trm else min(t, cfg.T - 1) / max(cfg.T - 1, 1); first = z is None
        logits, zf = step_fn(t_norm, first)(params, x_can, y, tvj, jnp.zeros(1) if first else z)
        z_new = zf if first else z + eta_z * (zf - z)
        p = jax.nn.softmax(logits, axis=-1).transpose(0, 3, 1, 2); y_new = y + eta * (p - y)
        if trm: res = jnp.mean(jnp.abs(z_new - (z if z is not None else z_new)), axis=tuple(range(1, z_new.ndim)))
        else: res = jnp.mean(jnp.abs(y_new - y), axis=(1, 2, 3))
        pred9 = EV.layout_gather(jnp.argmax(logits, axis=-1), layout).astype(jnp.int32); pred9 = jnp.where(pred9 == G.VOID, 0, pred9)
        ex = jnp.all((pred9 == sol9).reshape(B, -1), axis=1)
        ex_t.append(np.asarray(ex)); res_t.append(np.asarray(res)); y, z = y_new, z_new
        if (t + 1) in (16, 64):
            # power iteration on the outer-step map at the current state (per sample; samples are independent)
            tn = 0.0 if trm else 1.0
            f = step_fn(tn, False)
            def gmap(yz):
                yy, zz = yz
                lg, zf2 = f(params, x_can, yy, tvj, zz)
                z2 = zz + eta_z * (zf2 - zz); p2 = jax.nn.softmax(lg, axis=-1).transpose(0, 3, 1, 2); y2 = yy + eta * (p2 - yy)
                return (y2, z2)
            key = jax.random.PRNGKey(0); vy = jax.random.normal(key, y.shape); vz = jax.random.normal(jax.random.PRNGKey(1), z.shape)
            if trm: vy = jnp.zeros_like(vy)   # y is a pure readout of z on the field cell: the state is z
            def nrm(v): return jnp.sqrt(jnp.sum(v[0].reshape(B, -1) ** 2, 1) + jnp.sum(v[1].reshape(B, -1) ** 2, 1))
            n0 = nrm((vy, vz)); v = (vy / n0.reshape(-1, *[1] * (vy.ndim - 1)), vz / n0.reshape(-1, *[1] * (vz.ndim - 1)))
            lam_hist = []
            for it in range(POWER_ITERS):
                _, jv = jax.jvp(gmap, ((y, z),), (v,))
                if trm: jv = (jnp.zeros_like(jv[0]), jv[1])
                nn_ = nrm(jv); lam_hist.append(np.asarray(nn_)); v = (jv[0] / nn_.reshape(-1, *[1] * (jv[0].ndim - 1)), jv[1] / nn_.reshape(-1, *[1] * (jv[1].ndim - 1)))
            lam[t + 1] = np.asarray(lam_hist[-1])
            print(f"  {name} t={t+1}: lambda_max median {np.median(lam[t+1]):.3f} p90 {np.percentile(lam[t+1],90):.3f} ({time.time()-t0:.0f}s)", flush=True)
    ex_t = np.stack(ex_t); res_t = np.stack(res_t)
    solved64 = ex_t[-1]; solved16 = ex_t[15]
    first_ex = np.where(ex_t.any(0), ex_t.argmax(0), -1)
    rat = R[ids]
    o = {"eta": eta, "eta_z": eta_z, "solved16": float(solved16.mean()), "solved64": float(solved64.mean()),
         "first_exact_median_solved": float(np.median(first_ex[solved64])) if solved64.any() else None,
         "first_exact_by_octile": [float(np.median(first_ex[(rat >= qs[b]) & (rat < qs[b+1]) & solved64])) if ((rat >= qs[b]) & (rat < qs[b+1]) & solved64).any() else None for b in range(8)],
         "exact_vs_step": ex_t.mean(1).tolist(),
         "resid_solved_median_by_step": np.median(res_t[:, solved64], axis=1).tolist() if solved64.any() else None,
         "resid_unsolved_median_by_step": np.median(res_t[:, ~solved64], axis=1).tolist() if (~solved64).any() else None,
         "resid_t64_solved": res_t[-1, solved64].tolist(), "resid_t64_unsolved": res_t[-1, ~solved64].tolist(),
         "lam16": lam[16].tolist(), "lam64": lam[64].tolist(), "solved64_mask": solved64.tolist(), "solved16_mask": solved16.tolist(),
         "unsolved_converged": int(np.sum(res_t[-1, ~solved64] <= (np.median(res_t[-1, solved64]) if solved64.any() else 0))),
         "n_unsolved": int((~solved64).sum()), "wall_s": round(time.time() - t0, 1)}
    print(f"{name}: solved@16 {100*o['solved16']:.1f} @64 {100*o['solved64']:.1f} | first_exact median {o['first_exact_median_solved']} by octile {o['first_exact_by_octile']} | "
          f"resid t64 solved median {np.median(res_t[-1, solved64]) if solved64.any() else float('nan'):.3g} unsolved median {np.median(res_t[-1, ~solved64]) if (~solved64).any() else float('nan'):.3g} | "
          f"lambda t64 solved median {np.median(lam[64][solved64]) if solved64.any() else float('nan'):.3f} unsolved median {np.median(lam[64][~solved64]) if (~solved64).any() else float('nan'):.3f} | converged-wrong {o['unsolved_converged']}/{o['n_unsolved']} | {o['wall_s']}s", flush=True)
    out["arms"][name] = o

run("R0 (ours, field regime, z-norm, EMA)", "runs/pretrainsportC1_R0/ckpt_latest.pkl", True)
run("B0 (ours, z-norm, vsel A:20k, raw)", "runs/pretrainsportC1_B0a/ckpt_020000.pkl", False)
run("X0 (field cell, EMA)", "runs/pretrainsportC1_X0/ckpt_latest.pkl", True)
Path("runs/analysis/sportC1_lensG_dynamics_20260903.json").write_text(json.dumps(out))
print("LENSG-DYNAMICS-DONE", flush=True)
