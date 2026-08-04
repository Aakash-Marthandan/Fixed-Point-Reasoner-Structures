# Ledger: C17 (cluster-update layers, registered 2026-08-02). Objecthood is
# task-dependent, so segmentations are MEASURED CANDIDATES (three modes) and
# selection is learned via rule-conditioned gates at the model level. The
# partitions are S9-INVARIANT as partitions (palette permutation relabels
# values but preserves same-color groupings), so aggregating every field
# channel with the same spatial partition preserves equivariance (CI-1 ext).
"""Connected components on device + component-mean aggregation.

Labels are computed by min-label propagation to a fixed point
(lax.while_loop — convergence in O(component diameter), no giant unroll).
Out-of-mask cells keep their own index → singleton components → aggregation
is the identity there (a structural no-op, not a special case).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from qhrrn2.grid import CANVAS, VOID

MODES = ("color4", "color8", "nonblack4")
_BIG = CANVAS * CANVAS + 7


def _shifts(conn8: bool):
    s4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    return s4 + ([(-1, -1), (-1, 1), (1, -1), (1, 1)] if conn8 else [])


def _shift(a, dy, dx, fill):
    """Non-wrapping shift: pad with `fill`, slice back to (H, W)."""
    H, W = a.shape
    p = jnp.pad(a, 1, constant_values=fill)
    return jax.lax.dynamic_slice(p, (1 - dy, 1 - dx), (H, W))


def connected_components(canvas, mode: str):
    """(H, W) int canvas -> (H, W) int32 labels in [0, H*W).

    color4/color8: same-VALUE components over non-VOID cells (black included —
    region tasks segment black areas). nonblack4: components of the mask
    (value != 0 and != VOID), colors mixed.
    """
    H, W = canvas.shape
    idx = jnp.arange(H * W, dtype=jnp.int32).reshape(H, W)
    if mode == "nonblack4":
        mask = (canvas != 0) & (canvas != VOID)
        conn8 = False
    elif mode in ("color4", "color8"):
        mask = canvas != VOID
        conn8 = mode == "color8"
    else:
        raise ValueError(mode)

    def connected(dy, dx):
        n_mask = _shift(mask, dy, dx, False)
        if mode == "nonblack4":
            return mask & n_mask
        n_val = _shift(canvas, dy, dx, VOID)
        return mask & n_mask & (canvas == n_val)

    conn = [(dy, dx, connected(dy, dx)) for dy, dx in _shifts(conn8)]

    def body(labels):
        new = labels
        for dy, dx, c in conn:
            n_lab = _shift(labels, dy, dx, _BIG)
            new = jnp.where(c, jnp.minimum(new, n_lab), new)
        return new

    def cond(state):
        labels, changed = state
        return changed

    def step(state):
        labels, _ = state
        new = body(labels)
        return new, jnp.any(new != labels)

    labels0 = idx  # every cell starts as its own component (incl. out-of-mask)
    labels, _ = jax.lax.while_loop(cond, step, (labels0, jnp.array(True)))
    return labels


def component_mean(z, labels):
    """Aggregate features over components and broadcast back.

    z: (C, H, W, d); labels: (H, W) int32 in [0, H*W).
    Returns (C, H, W, d) where each cell carries its component's mean —
    the shared per-object variable the binding analysis found missing.
    """
    C, H, W, d = z.shape
    seg = labels.reshape(-1)
    zf = z.reshape(C, H * W, d)
    sums = jax.vmap(lambda x: jax.ops.segment_sum(x, seg, num_segments=H * W))(zf)
    counts = jax.ops.segment_sum(jnp.ones((H * W,)), seg, num_segments=H * W)
    means = sums / jnp.maximum(counts, 1.0)[None, :, None]
    return means[:, seg, :].reshape(C, H, W, d)
