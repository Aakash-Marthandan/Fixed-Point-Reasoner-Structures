# Ledger: C1 (masked CE + weighted VOID region + canvas-size loss),
#         C5/H-4 (flux ledger, beta-priced), C14 (attention flux, beta_nl-priced),
#         C9 (deep supervision over iterates).
from __future__ import annotations

import jax
import jax.numpy as jnp

from qhrrn2.config import Config
from qhrrn2.grid import VOID
from qhrrn2.model import iterate, size_candidates, size_mixture_probs


def _step_loss(out, x_canvas, y_canvas, mask, cfg: Config):
    logp = jax.nn.log_softmax(out.logits, axis=-1)
    ce_map = -jnp.take_along_axis(logp, y_canvas[..., None], axis=-1)[..., 0]
    n_in = jnp.maximum(mask.sum(), 1)
    n_out = jnp.maximum((~mask).sum(), 1)
    ce_in = jnp.sum(ce_map * mask) / n_in
    ce_out = jnp.sum(ce_map * ~mask) / n_out

    # C1 v3 (ledger 2026-08-02): size = mixture over MEASURED candidates,
    # offsets applied relative to the selected candidate. v2's relative frame
    # is the candidate-0 slice; content-derived sizes (counting bars, x2/x3
    # tiling) get their own candidates — selection extrapolates by
    # construction where absolute classes provably could not.
    cands = size_candidates(x_canvas)
    h_true = jnp.sum(jnp.any(mask, axis=1))
    w_true = jnp.sum(jnp.any(mask, axis=0))
    p_h = size_mixture_probs(out.size_sel_h, out.size_h, cands[0])
    p_w = size_mixture_probs(out.size_sel_w, out.size_w, cands[1])
    size_ce = (-jnp.log(p_h[jnp.clip(h_true - 1, 0, 29)] + 1e-9)
               - jnp.log(p_w[jnp.clip(w_true - 1, 0, 29)] + 1e-9))

    # B1-full free-bits form (pretrain-13, ledger 2026-08-12; the Q1 dose-kill
    # consequence): with flux_floors set, only per-scale EXCESS above the
    # floor is tolled — price the cuts that carry excess, floor the cuts that
    # carry structure. () = the global toll, expression untouched.
    if cfg.flux_floors:
        floors = [float(x) for x in cfg.flux_floors.split(",")]
        assert len(floors) == cfg.scales, "one floor per scale"
        flux_toll = jnp.sum(jax.nn.relu(out.flux - jnp.asarray(floors)))
    else:
        flux_toll = jnp.sum(out.flux)
    total = (ce_in + cfg.w_void * ce_out
             + cfg.lambda_size * size_ce
             + cfg.beta_flux * flux_toll
             + cfg.beta_flux_nl * jnp.sum(out.flux_attn)
             + cfg.beta_flux_obj * jnp.sum(out.flux_obj))
    return total, ce_in


def pair_loss(params, cfg: Config, x_canvas, y_canvas, *, tau: float, rng=None,
              task_vec=None, labels_x=None, yprev_init=None):
    """Deep-supervised loss for one (input, output) pair; mask = true output canvas.

    yprev_init: optional initial feedback canvas ([H-23] basin rows)."""
    mask = y_canvas != VOID
    outs = iterate(params, cfg, x_canvas, tau=tau, rng=rng, task_vec=task_vec,
                   labels_x=labels_x, yprev_init=yprev_init)
    losses, ces = zip(*(_step_loss(o, x_canvas, y_canvas, mask, cfg) for o in outs))
    return jnp.mean(jnp.stack(losses)), {
        "ce_in_last": ces[-1],
        "flux_last": outs[-1].flux,
        "flux_attn_last": outs[-1].flux_attn,
        "flux_obj_last": outs[-1].flux_obj,
        "rule_entropy_last": -jnp.sum(outs[-1].rule_q * jnp.log(outs[-1].rule_q + 1e-9), axis=-1),
    }


def batch_loss(params, cfg: Config, x_batch, y_batch, *, tau: float, rng=None,
               task_vecs=None, labels_x=None, weights=None, yprev_batch=None):
    """Mean pair_loss over a batch of canvases (B, 32, 32).

    task_vecs: optional (B, d_task) — per-example program embeddings (C16),
    e.g. table rows gathered for a mixed-task joint batch.
    labels_x: optional (B, 3, 32, 32) precomputed input segmentations (C17
    speed pipeline) in OBJ_ENC_MODES order.
    weights: optional (B,) per-row loss weights (agreement-regularized
    population, ledger 2026-08-07: support rows 1.0, consensus-pseudo-labeled
    query rows λ). Normalized by sum(weights), so weights=None is the plain
    mean; zero-weight rows contribute nothing (but still cost a forward)."""
    keys = None if rng is None else jax.random.split(rng, x_batch.shape[0])
    args, axes = [x_batch, y_batch], [0, 0]
    if keys is not None:
        args.append(keys); axes.append(0)
    if task_vecs is not None:
        args.append(task_vecs); axes.append(0)
    if labels_x is not None:
        args.append(labels_x); axes.append(0)
    if yprev_batch is not None:
        args.append(yprev_batch); axes.append(0)

    def f(x, y, *rest):
        i = 0
        k = tv = lx = yp = None
        if keys is not None:
            k = rest[i]; i += 1
        if task_vecs is not None:
            tv = rest[i]; i += 1
        if labels_x is not None:
            lx = rest[i]; i += 1
        if yprev_batch is not None:
            yp = rest[i]
        return pair_loss(params, cfg, x, y, tau=tau, rng=k, task_vec=tv,
                         labels_x=lx, yprev_init=yp)

    losses, aux = jax.vmap(f, in_axes=tuple(axes))(*args)
    if weights is None:
        return jnp.mean(losses), jax.tree.map(jnp.mean, aux)
    w = weights / jnp.maximum(jnp.sum(weights), 1e-9)
    return jnp.sum(losses * w), jax.tree.map(jnp.mean, aux)
