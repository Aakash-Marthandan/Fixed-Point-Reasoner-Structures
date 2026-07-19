# Ledger: C1 (masked CE + weighted VOID region + canvas-size loss),
#         C5/H-4 (flux ledger, beta-priced), C9 (deep supervision over iterates).
from __future__ import annotations

import jax
import jax.numpy as jnp

from qhrrn2.config import Config
from qhrrn2.grid import VOID
from qhrrn2.model import iterate


def _step_loss(out, y_canvas, mask, cfg: Config):
    logp = jax.nn.log_softmax(out.logits, axis=-1)
    ce_map = -jnp.take_along_axis(logp, y_canvas[..., None], axis=-1)[..., 0]
    n_in = jnp.maximum(mask.sum(), 1)
    n_out = jnp.maximum((~mask).sum(), 1)
    ce_in = jnp.sum(ce_map * mask) / n_in
    ce_out = jnp.sum(ce_map * ~mask) / n_out

    h_true = jnp.sum(jnp.any(mask, axis=1))          # rows occupied (top-left placement)
    w_true = jnp.sum(jnp.any(mask, axis=0))
    size_ce = (-jax.nn.log_softmax(out.size_h)[h_true - 1]
               - jax.nn.log_softmax(out.size_w)[w_true - 1])

    total = (ce_in + cfg.w_void * ce_out
             + cfg.lambda_size * size_ce
             + cfg.beta_flux * jnp.sum(out.flux))
    return total, ce_in


def pair_loss(params, cfg: Config, x_canvas, y_canvas, *, tau: float, rng=None):
    """Deep-supervised loss for one (input, output) pair; mask = true output canvas."""
    mask = y_canvas != VOID
    outs = iterate(params, cfg, x_canvas, tau=tau, rng=rng)
    losses, ces = zip(*(_step_loss(o, y_canvas, mask, cfg) for o in outs))
    return jnp.mean(jnp.stack(losses)), {
        "ce_in_last": ces[-1],
        "flux_last": outs[-1].flux,
        "rule_entropy_last": -jnp.sum(outs[-1].rule_q * jnp.log(outs[-1].rule_q + 1e-9), axis=-1),
    }


def batch_loss(params, cfg: Config, x_batch, y_batch, *, tau: float, rng=None):
    """Mean pair_loss over a batch of canvases (B, 32, 32)."""
    if rng is None:
        keys = None
        f = lambda x, y: pair_loss(params, cfg, x, y, tau=tau)
        losses, aux = jax.vmap(f)(x_batch, y_batch)
    else:
        keys = jax.random.split(rng, x_batch.shape[0])
        f = lambda x, y, k: pair_loss(params, cfg, x, y, tau=tau, rng=k)
        losses, aux = jax.vmap(f)(x_batch, y_batch, keys)
    return jnp.mean(losses), jax.tree.map(jnp.mean, aux)
