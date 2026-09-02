# Ledger: C1 (masked CE + weighted VOID region + canvas-size loss),
#         C5/H-4 (flux ledger, beta-priced), C14 (attention flux, beta_nl-priced),
#         C9 (deep supervision over iterates).
from __future__ import annotations

import jax
import jax.numpy as jnp

from qhrrn2.config import Config
from qhrrn2.grid import VOID, NUM_COLORS, CANVAS
from qhrrn2.model import iterate, iterate_eq, size_candidates, size_mixture_probs, VOCAB


def log_stablemax(x):
    """HRM/TRM's stablemax log-probabilities (their losses.py, read 2026-09-02):
    s(x) = x + 1 for x >= 0, 1 / (1 - x) for x < 0; log(s / sum s). Used by the
    X0 field-recipe arm (cfg.loss_kind == "stablemax", labeled); f32 here
    (they compute it in f64)."""
    s = jnp.where(x < 0, 1.0 / (1.0 - x + 1e-30), x + 1.0)
    return jnp.log(s / jnp.sum(s, axis=-1, keepdims=True))


def _step_loss(out, x_canvas, y_canvas, mask, cfg: Config):
    if cfg.loss_kind == "stablemax":
        logp = log_stablemax(out.logits)
    else:
        logp = jax.nn.log_softmax(out.logits, axis=-1)
    ce_map = -jnp.take_along_axis(logp, y_canvas[..., None], axis=-1)[..., 0]
    n_in = jnp.maximum(mask.sum(), 1)
    n_out = jnp.maximum((~mask).sum(), 1)
    ce_in = jnp.sum(ce_map * mask) / n_in
    ce_out = jnp.sum(ce_map * ~mask) / n_out
    if cfg.cell_kind == "trm":
        # X0: the field-recipe cell has no size / flux / rule channels — its
        # loss is the (stablemax or softmax) cross-entropy alone (TRM lm_loss).
        return ce_in + cfg.w_void * ce_out, ce_in

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
    k_fpa = None
    if cfg.equilibrium and cfg.fpa_k > 0 and rng is not None:
        # wave 3a FPA: one extra split ONLY on this branch (fpa_k=0 leaves the
        # registered rng stream bit-exact — tests/test_fpa.py)
        rng, k_fpa = jax.random.split(rng)
    outs = iterate(params, cfg, x_canvas, tau=tau, rng=rng, task_vec=task_vec,
                   labels_x=labels_x, yprev_init=yprev_init)
    losses, ces = zip(*(_step_loss(o, x_canvas, y_canvas, mask, cfg) for o in outs))
    total = jnp.mean(jnp.stack(losses))
    aux = {
        "ce_in_last": ces[-1],
        "flux_last": outs[-1].flux,
        "flux_attn_last": outs[-1].flux_attn,
        "flux_obj_last": outs[-1].flux_obj,
        "rule_entropy_last": -jnp.sum(outs[-1].rule_q * jnp.log(outs[-1].rule_q + 1e-9), axis=-1),
    }
    if k_fpa is not None:
        # FIXED-POINT ANCHOR rows (H-45): corrupt eps~U[0,fpa_eps] of the true-extent
        # cells of the SOLUTION uniformly in 0..9, then apply the FINAL map (t_norm=1)
        # fpa_k times from there and supervise every step toward the solution —
        # local contraction of the final map at its fixed point, trained.
        k_e, k_m, k_c, k_roll = jax.random.split(k_fpa, 4)
        eps = jax.random.uniform(k_e, (), minval=0.0, maxval=cfg.fpa_eps)
        # canvas shape derived from the pair (2026-09-01 native9; canvas32
        # sees identical values — the old (CANVAS, CANVAS) constant)
        flip = jax.random.bernoulli(k_m, eps, y_canvas.shape) & mask
        rand = jax.random.randint(k_c, y_canvas.shape, 0, NUM_COLORS).astype(jnp.int32)
        y_corr = jnp.where(flip, rand, y_canvas.astype(jnp.int32))
        y0 = jax.nn.one_hot(y_corr, VOCAB).transpose(2, 0, 1)
        outs_fp, _, _ = iterate_eq(params, cfg, x_canvas, tau=tau, rng=k_roll,
                                   task_vec=task_vec, t_total=cfg.fpa_k, y0_probs=y0,
                                   t_norm_fixed=1.0)
        l_fp, c_fp = zip(*(_step_loss(o, x_canvas, y_canvas, mask, cfg) for o in outs_fp))
        total = total + cfg.fpa_w * jnp.mean(jnp.stack(l_fp))
        aux["fpa_ce_last"] = c_fp[-1]
    return total, aux


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
