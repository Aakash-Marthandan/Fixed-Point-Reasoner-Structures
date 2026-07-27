# Ledger: Phase-1 trainability fitting (CI-3a) with LoO-validated selection —
# the C10 protocol mechanism pulled forward after the first triad run showed
# fit-to-zero memorization (ledger log 2026-07-20). The frozen-core TTT
# protocol (C10, CI-3b) arrives with pretraining.
from __future__ import annotations

import functools
import os

import numpy as np
import jax
import jax.numpy as jnp
import optax

# Persistent XLA compilation cache: first triad run spent ~920s compiling.
_CACHE = os.path.join(os.path.dirname(__file__), "..", "..", ".jax_cache")
jax.config.update("jax_compilation_cache_dir", os.path.abspath(_CACHE))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 5.0)

from qhrrn2 import grid as G
from qhrrn2.config import Config
from qhrrn2.model import iterate
from qhrrn2.objective import batch_loss


def pairs_to_batch(pairs, *, transforms=None, n_offsets: int = 2, seed: int = 0):
    """pairs × transform orbit × placement offsets -> (B,32,32) int32 canvases.

    VALIDITY RULE (ledger 2026-07-20): in full-fit mode an augmentation
    transform is admissible ONLY if it commutes with the task rule — D4 copies
    of a direction-dependent rule (e.g. translate-right) are contradictory
    supervision, and palette copies of a color-constant rule likewise.
    Callers pass the rule-consistent orbit explicitly; default is identity
    only. Placement offset (0,0) is always included so prediction-time
    placement is in-distribution.
    """
    rng = np.random.default_rng(seed)
    if transforms is None:
        transforms = [G.Transform(k=0)]
    xs, ys = [], []
    for t in transforms:
        for x, y in pairs:
            tx, ty = t.apply(x), t.apply(y)
            offsets = [(0, 0)]
            for _ in range(n_offsets):
                oy = int(rng.integers(0, G.CANVAS - max(tx.shape[0], ty.shape[0]) + 1))
                ox = int(rng.integers(0, G.CANVAS - max(tx.shape[1], ty.shape[1]) + 1))
                offsets.append((oy, ox))
            for oy, ox in offsets:
                xs.append(G.place_at(tx, oy, ox))
                ys.append(G.place_at(ty, oy, ox))
    return (jnp.asarray(np.stack(xs), dtype=jnp.int32),
            jnp.asarray(np.stack(ys), dtype=jnp.int32))


def episode_to_batch(ep: G.Episode, **kw):
    return pairs_to_batch(list(ep.support), **kw)


def fit(params, cfg: Config, x_batch, y_batch, *, steps: int, lr: float = 3e-3,
        weight_decay: float = 1e-4, tau: float = 1.0, seed: int = 0, log_every: int = 0):
    """Full-parameter AdamW fit; returns (params, losses)."""
    step, opt = _step_and_opt(cfg, lr, weight_decay, tau)
    opt_state = opt.init(params)

    rng = jax.random.PRNGKey(seed)
    losses = []
    for i in range(steps):
        rng, sub = jax.random.split(rng)
        params, opt_state, loss, aux = step(params, opt_state, sub, x_batch, y_batch)
        losses.append(float(loss))
        if log_every and (i + 1) % log_every == 0:
            print(f"  step {i+1:4d}  loss {losses[-1]:.4f}  ce_in {float(aux['ce_in_last']):.4f}",
                  flush=True)
    return params, losses


def evaluate_pair(params, cfg: Config, x_grid, y_grid, *, tau: float):
    """Exact-match + pixel accuracy of the GT-size-free prediction on one pair."""
    pred, (ph, pw), _ = predict(params, cfg, x_grid, tau=tau)
    size_ok = (ph, pw) == tuple(y_grid.shape)
    exact = bool(size_ok and np.array_equal(pred, y_grid))
    if size_ok:
        pix = float((pred == y_grid).mean())
    else:  # size wrong: score overlap region, penalized implicitly by exact=False
        h, w = min(ph, y_grid.shape[0]), min(pw, y_grid.shape[1])
        pix = float((pred[:h, :w] == y_grid[:h, :w]).mean()) if h and w else 0.0
    return exact, pix, size_ok


def fit_loo(params, cfg: Config, ep: G.Episode, *, steps: int, transforms=None,
            val_every: int = 50, lr: float = 3e-3, weight_decay: float = 1e-4,
            tau: float = 1.0, seed: int = 0, log_every: int = 0):
    """Leave-one-out fit: train on support[:-1], select best params by held-out
    validation on support[-1]. Returns (best_params, history dict)."""
    train_pairs = list(ep.support[:-1])
    val_x, val_y = ep.support[-1]
    x_b, y_b = pairs_to_batch(train_pairs, transforms=transforms, seed=seed)

    step, opt = _step_and_opt(cfg, lr, weight_decay, tau)
    opt_state = opt.init(params)

    rng = jax.random.PRNGKey(seed)
    best = {"params": params, "val_pix": -1.0, "val_exact": False, "step": 0}
    losses, val_curve = [], []
    for i in range(steps):
        rng, sub = jax.random.split(rng)
        params, opt_state, loss, _ = step(params, opt_state, sub, x_b, y_b)
        losses.append(float(loss))
        if (i + 1) % val_every == 0 or i + 1 == steps:
            exact, pix, _ = evaluate_pair(params, cfg, val_x, val_y, tau=tau)
            val_curve.append((i + 1, pix, exact))
            if (exact, pix) > (best["val_exact"], best["val_pix"]):
                best = {"params": params, "val_pix": pix, "val_exact": exact, "step": i + 1}
            if log_every and (i + 1) % log_every == 0:
                print(f"  step {i+1:4d}  loss {losses[-1]:.4f}  val_pix {pix:.3f}"
                      f"  val_exact {exact}", flush=True)
    return best["params"], {"losses": losses, "val_curve": val_curve, "best": best,
                            "final_params": params}


@functools.lru_cache(maxsize=8)
def _step_and_opt(cfg: Config, lr: float, weight_decay: float, tau: float):
    """Jitted train step, cached by hyperparameters. Batches are ARGUMENTS so
    the compiled program is batch-independent (closing over them baked
    constants into the HLO and forced a full recompile per task — measured
    ~20 min each, 2026-07-20); caching the step object also skips re-tracing
    across tasks in the same process."""
    opt = optax.adamw(lr, weight_decay=weight_decay)

    @jax.jit
    def step(params, opt_state, rng, x_b, y_b):
        (loss, aux), grads = jax.value_and_grad(batch_loss, has_aux=True)(
            params, cfg, x_b, y_b, tau=tau, rng=rng)
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss, aux
    return step, opt


@functools.lru_cache(maxsize=8)
def _predict_core(cfg: Config, tau: float):
    @jax.jit
    def core(params, x_can):
        outs = iterate(params, cfg, x_can, tau=tau, rng=None)
        last = outs[-1]
        return last.logits, last.size_h, last.size_w
    return core


def predict_voted(params, cfg: Config, x_grid: np.ndarray, transforms, *, tau: float = 1.0):
    """Test-time orbit voting (spec §, symmetrization): predict under each
    rule-consistent transform, invert, majority-vote cells among predictions
    that agree with the majority shape."""
    preds = []
    for t in transforms:
        pred, _, _ = predict(params, cfg, t.apply(np.asarray(x_grid)), tau=tau)
        preds.append(t.invert_output(pred))
    shapes = [p.shape for p in preds]
    maj_shape = max(set(shapes), key=shapes.count)
    stack = np.stack([p for p in preds if p.shape == maj_shape])
    voted = np.apply_along_axis(lambda v: np.bincount(v, minlength=11).argmax(), 0, stack)
    return voted.astype(np.int8), maj_shape


def predict(params, cfg: Config, x_grid: np.ndarray, *, tau: float = 1.0):
    """Deterministic prediction at the SAME tau used in fitting (the first triad
    run predicted at tau=0.05 after fitting at 1.0 — a distribution shift).
    Returns (grid cropped to the PREDICTED canvas, predicted (H, W), full canvas
    argmax) — no ground-truth size anywhere (C1)."""
    x_can = jnp.asarray(G.place(np.asarray(x_grid)), dtype=jnp.int32)
    logits, size_h, size_w = _predict_core(cfg, tau)(params, x_can)
    canvas = np.asarray(jnp.argmax(logits, axis=-1))
    h = int(jnp.argmax(size_h)) + 1
    w = int(jnp.argmax(size_w)) + 1
    pred = np.where(canvas[:h, :w] == G.VOID, 0, canvas[:h, :w]).astype(np.int8)
    return pred, (h, w), canvas
