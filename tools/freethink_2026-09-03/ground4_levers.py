"""Freethink grounding 4 (CPU, $0): INFERENCE-TIME LEVERS on our trained d128 maps (strat-512, cold), no retraining:
 (A) inner cycles: K latent passes per outer step before the readout update (per-step depth without training);
 (B) depth: exact at t = 64 / 128 / 256 (slow propagation vs stuck);
 (I) softer readout feedback: logits / tau_r before the softmax that feeds y back (measurement-collapse test);
 (D) kicked restarts from the t=64 STUCK state: reset a fraction eps of non-given cells to VOID (keep z or reset z),
     iterate 64 more steps, 8 kicks; compare with fresh random-init draws (records) on the same failures."""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "tools")
import jax, jax.numpy as jnp
from qhrrn2 import episodic as E, grid as G, model as M, sudoku as SU, sudoku_extreme as SX
from qhrrn2.config import Config
import eval_sudoku_extreme as EV
d = SX.load_prepared("data/sudoku_extreme/sudoku_extreme_seed0.npz"); Q, A, R = d["test_q"], d["test_a"], d["test_rating"]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 512
ids = SX.stratified_subsample(R, 512, 20260821)[:N]; B = len(ids); rat = R[ids]; oct_edges = np.quantile(R[SX.stratified_subsample(R, 512, 20260821)], np.linspace(0, 1, 9)); oct_edges[-1] += 1
out = {}
def load(ckpt, ema):
    saved = E.load_ckpt(ckpt); defaults = Config(); cfg = Config(**{k: type(getattr(defaults, k))(v) for k, v in saved["config"].items()})
    st = saved["state_ema"] if ema else saved["state"]; return cfg, st["model"], jnp.asarray(st["table"][0])
def setup(cfg):
    layout = cfg.sudoku_layout; cv = SU.layout_canvas(layout); puz9 = Q[ids].astype(np.int32)
    x_can = jnp.asarray(np.stack([SU.place_layout(g.astype(np.int8), layout) for g in puz9]), jnp.int32)
    void = jax.nn.one_hot(jnp.full((cv, cv), G.VOID, jnp.int32), M.VOCAB).transpose(2, 0, 1)
    return layout, x_can, jnp.broadcast_to(void, (B,) + void.shape), jnp.asarray(A[ids].astype(np.int32)), jnp.asarray(puz9 != 0), void
def rollout(cfg, params, tvj, x_can, y, z, sol9, layout, eta, eta_z, T, K=1, tau_r=1.0, t_offset=0, record_at=()):
    ex = {}; y_c, z_c = y, z
    for t in range(T):
        tn = min(t + t_offset, cfg.T - 1) / max(cfg.T - 1, 1)
        for k in range(K):
            first = z_c is None
            logits, zf = EV._step(cfg, 1.0, float(tn), first)(params, x_can, y_c, tvj, jnp.zeros(1) if first else z_c)
            z_c = zf if first else z_c + eta_z * (zf - z_c)
        p = jax.nn.softmax(logits / tau_r, axis=-1).transpose(0, 3, 1, 2); y_c = y_c + eta * (p - y_c)
        pred9 = EV.layout_gather(jnp.argmax(logits, axis=-1), layout).astype(jnp.int32); pred9 = jnp.where(pred9 == G.VOID, 0, pred9)
        if (t + 1) in record_at or t + 1 == T: ex[t + 1] = np.asarray(jnp.all((pred9 == sol9).reshape(B, -1), axis=1))
    return ex, y_c, z_c, np.asarray(pred9)
for name, ckpt, ema in (("R0", "runs/pretrainsportC1_R0/ckpt_latest.pkl", True), ("B0vsel", "runs/pretrainsportC1_B0a/ckpt_020000.pkl", False)):
    cfg, params, tvj = load(ckpt, ema); eta, eta_z = (float(v) for v in M.eq_etas(params, cfg)); layout, x_can, y0, sol9, given, void = setup(cfg)
    o = {}; t0 = time.time()
    # (B) depth + baseline
    ex, y64, z64, pred64 = rollout(cfg, params, tvj, x_can, y0, None, sol9, layout, eta, eta_z, 256, record_at=(16, 64, 128, 256))
    o["depth"] = {str(k): float(v.mean()) for k, v in ex.items()}; base64 = ex[64]
    print(f"{name} (B) cold exact @16/64/128/256: " + " ".join(f"{k}:{100*v.mean():.1f}" for k, v in ex.items()) + f" ({time.time()-t0:.0f}s)", flush=True)
    # (A) inner cycles
    for K in (2, 3):
        exK, *_ = rollout(cfg, params, tvj, x_can, y0, None, sol9, layout, eta, eta_z, 64, K=K, record_at=(16, 64))
        o[f"innerK{K}"] = {str(k): float(v.mean()) for k, v in exK.items()}
        print(f"{name} (A) inner cycles K={K}: exact @16 {100*exK[16].mean():.1f} @64 {100*exK[64].mean():.1f} (base @64 {100*base64.mean():.1f}; gained {int((exK[64] & ~base64).sum())} lost {int((~exK[64] & base64).sum())}) ({time.time()-t0:.0f}s)", flush=True)
    # (I) softer readout feedback
    for tr in (1.5, 2.0, 0.7):
        exT, *_ = rollout(cfg, params, tvj, x_can, y0, None, sol9, layout, eta, eta_z, 64, tau_r=tr, record_at=(64,))
        o[f"tau{tr}"] = float(exT[64].mean())
        print(f"{name} (I) readout temperature {tr}: exact @64 {100*exT[64].mean():.1f} (gained {int((exT[64] & ~base64).sum())} lost {int((~exT[64] & base64).sum())})", flush=True)
    # (D) kicked restarts from the t=64 stuck state (unsolved puzzles only; y64/z64 from the base rollout)
    # a fresh-draw baseline from the records is printed in the summary (any-of-8 draws rescue rate)
    unsolved = ~base64; rng = np.random.default_rng(0); nk = 8
    ng = np.asarray(~given)   # non-given cells (B,9,9)
    for eps, keep_z, label in ((0.2, True, "eps.2 keep-z"), (0.5, True, "eps.5 keep-z"), (0.2, False, "eps.2 reset-z"), (1.0, False, "full VOID reset-z (= cold restart control)")):
        any_hit = np.zeros(B, bool); per_kick = []
        for k in range(nk):
            mask = ng & (rng.random(ng.shape) < eps)     # cells to reset (in the 9x9 frame)
            # build the kicked y: VOID one-hot on reset cells, y64 elsewhere (map 9x9 mask into the canvas frame)
            m_can = np.stack([SU.place_layout(mk.astype(np.int8), layout) for mk in mask]).astype(bool)   # nonzero where reset
            m_can = jnp.asarray(m_can)[:, None, :, :]
            y_k = jnp.where(m_can, void[None], y64)
            exk, *_ = rollout(cfg, params, tvj, x_can, y_k, (z64 if keep_z else None), sol9, layout, eta, eta_z, 64, t_offset=(cfg.T - 1 if keep_z else 0), record_at=(64,))
            any_hit |= np.asarray(exk[64]); per_kick.append(float(np.asarray(exk[64])[unsolved].mean()))
        o[f"kick {label}"] = dict(per_kick_rescue=float(np.mean(per_kick)), any8_rescue=float(any_hit[unsolved].mean()), n_unsolved=int(unsolved.sum()))
        print(f"{name} (D) kick {label}: rescue of t64 failures per kick {100*np.mean(per_kick):.1f}%, any of 8 kicks {100*any_hit[unsolved].mean():.1f}% (n_unsolved {unsolved.sum()}) ({time.time()-t0:.0f}s)", flush=True)
    out[name] = o
# fresh-draw rescue baselines from the banked records (same puzzles: strat-512 is a subset of the 20k? no — different sets; print the 20k rates for reference)
def recs(p):
    z = dict(np.load(Path(p)/"records_all.npz", allow_pickle=True)); return z
for name, p in (("R0", "runs/sxscan_psportC1R0"),):
    z = recs(p); cold = z["cold_exact"].astype(bool); fh = z["mi_first_hit"]
    print(f"{name} fresh random-init draws on the 20k (reference): rescue of cold failures by any of 8 draws {100*np.mean((fh[~cold] >= 0) & (fh[~cold] < 8)):.1f}%, by any of 128 {100*np.mean(fh[~cold] >= 0):.1f}%", flush=True)
Path("runs/analysis/freethink_ground4_levers_20260903.json").write_text(json.dumps(out)); print("GROUND4-DONE", flush=True)
