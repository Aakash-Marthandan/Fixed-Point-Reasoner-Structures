# Ledger: Phase-1 trainability fitting (CI-3a). The frozen-core TTT protocol
# (C10, CI-3b) comes after pretraining exists; this module is the full-fit loop
# used by gates and, later, by pretraining.
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import optax

from qhrrn2 import grid as G
from qhrrn2.config import Config
from qhrrn2.model import iterate
from qhrrn2.objective import batch_loss


def episode_to_batch(ep: G.Episode, orbit_n: int = 8, seed: int = 0,
                     use_palette: bool = False):
    """Support pairs × D4 orbit -> stacked canvases (B,32,32) int32.

    Palette augmentation is OFF by default here: in full-fit mode the rule is
    memorized in the parameters, so palette-permuted copies would be
    *different-rule* training data for color-specific tasks. (In frozen-core
    TTT with per-task params this trade-off is revisited; see ledger C10.)
    """
    rng = np.random.default_rng(seed)
    transforms = G.sample_orbit(rng, orbit_n, use_d4=True, use_palette=use_palette)
    xs, ys = [], []
    for t in transforms:
        for x, y in ep.support:
            xs.append(G.place(t.apply(x)))
            ys.append(G.place(t.apply(y)))
    return (jnp.asarray(np.stack(xs), dtype=jnp.int32),
            jnp.asarray(np.stack(ys), dtype=jnp.int32))


def fit(params, cfg: Config, x_batch, y_batch, *, steps: int, lr: float = 3e-3,
        tau: float = 1.0, seed: int = 0, log_every: int = 0):
    """Full-parameter Adam fit; returns (params, losses)."""
    opt = optax.adam(lr)
    opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state, rng):
        (loss, aux), grads = jax.value_and_grad(batch_loss, has_aux=True)(
            params, cfg, x_batch, y_batch, tau=tau, rng=rng)
        updates, opt_state = opt.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss, aux

    rng = jax.random.PRNGKey(seed)
    losses = []
    for i in range(steps):
        rng, sub = jax.random.split(rng)
        params, opt_state, loss, aux = step(params, opt_state, sub)
        losses.append(float(loss))
        if log_every and (i + 1) % log_every == 0:
            print(f"  step {i+1:4d}  loss {losses[-1]:.4f}  ce_in {float(aux['ce_in_last']):.4f}")
    return params, losses


def predict(params, cfg: Config, x_grid: np.ndarray, *, tau: float = 0.05):
    """Deterministic prediction. Returns (grid cropped to the PREDICTED canvas,
    predicted (H, W), full canvas argmax) — no ground-truth size anywhere (C1)."""
    x_can = jnp.asarray(G.place(np.asarray(x_grid)), dtype=jnp.int32)
    outs = iterate(params, cfg, x_can, tau=tau, rng=None)
    last = outs[-1]
    canvas = np.asarray(jnp.argmax(last.logits, axis=-1))
    h = int(jnp.argmax(last.size_h)) + 1
    w = int(jnp.argmax(last.size_w)) + 1
    pred = np.where(canvas[:h, :w] == G.VOID, 0, canvas[:h, :w]).astype(np.int8)
    return pred, (h, w), canvas
