# Ledger: CI-8a (C17, 2026-08-02) — device connected-components must match
# scipy.ndimage.label EXACTLY (as partitions) on all three modes, including
# the serpentine worst case; component_mean must be a per-component constant,
# an exact mean, and the identity on singletons. Substrate tests come before
# model wiring (the AppleDouble/checkpoint lessons).
import numpy as np
import jax.numpy as jnp
import pytest
from scipy import ndimage

from qhrrn2 import grid as G
from qhrrn2.objects import MODES, component_mean, connected_components


def _partition_equal(labels_a: np.ndarray, labels_b: np.ndarray, mask: np.ndarray):
    """Same partition of `mask` cells, label names ignored."""
    fa = labels_a[mask]
    fb = labels_b[mask]
    map_ab, map_ba = {}, {}
    for a, b in zip(fa.tolist(), fb.tolist()):
        if map_ab.setdefault(a, b) != b or map_ba.setdefault(b, a) != a:
            return False
    return True


def _scipy_labels(canvas: np.ndarray, mode: str):
    H, W = canvas.shape
    out = np.full((H, W), -1, dtype=np.int64)
    if mode == "nonblack4":
        mask = (canvas != 0) & (canvas != G.VOID)
        lab, _ = ndimage.label(mask)
        out[mask] = lab[mask]
        return out, mask
    struct = np.ones((3, 3)) if mode == "color8" else None
    mask = canvas != G.VOID
    nxt = 1
    for v in np.unique(canvas[mask]):
        m = canvas == v
        lab, n = ndimage.label(m, structure=struct)
        out[m] = lab[m] + nxt
        nxt += n + 1
    return out, mask


def _check(canvas):
    for mode in MODES:
        ours = np.asarray(connected_components(jnp.asarray(canvas, jnp.int32), mode))
        ref, mask = _scipy_labels(canvas, mode)
        assert _partition_equal(ours, ref, mask), f"partition mismatch: {mode}"
        # out-of-mask cells are singletons (their own index)
        idx = np.arange(canvas.size).reshape(canvas.shape)
        assert (ours[~mask] == idx[~mask]).all(), f"non-singleton out-of-mask: {mode}"


def test_ci8a_random_grids():
    rng = np.random.default_rng(0)
    for i in range(12):
        density = rng.uniform(0.15, 0.9)
        g = np.where(rng.random((rng.integers(3, 30), rng.integers(3, 30))) < density,
                     rng.integers(0, 10, (30, 30))[: 30, : 30][0, 0], 0)
        g = rng.integers(0, 10, g.shape) * (rng.random(g.shape) < density)
        _check(G.place(g.astype(np.int8)))


def test_ci8a_serpentine_worst_case():
    """Single snake component of path length ~450 — the propagation-depth
    stressor for the while_loop fixed point."""
    g = np.zeros((29, 29), dtype=np.int8)
    for r in range(29):
        if r % 2 == 0:
            g[r, :] = 3
        else:
            g[r, 0 if (r // 2) % 2 else 28] = 3
    _check(G.place(g))
    labels = np.asarray(connected_components(jnp.asarray(G.place(g), jnp.int32), "color4"))
    snake = np.asarray(G.place(g)) == 3
    assert len(np.unique(labels[snake])) == 1, "snake split into pieces"


def test_ci8a_diagonal_modes_differ():
    g = np.zeros((4, 4), dtype=np.int8)
    g[0, 0] = g[1, 1] = 5  # touching only diagonally
    c = jnp.asarray(G.place(g), jnp.int32)
    l4 = np.asarray(connected_components(c, "color4"))
    l8 = np.asarray(connected_components(c, "color8"))
    assert l4[0, 0] != l4[1, 1], "color4 must NOT join diagonals"
    assert l8[0, 0] == l8[1, 1], "color8 must join diagonals"


def test_component_mean_exact_and_identity():
    g = np.zeros((6, 6), dtype=np.int8)
    g[1:3, 1:4] = 2   # a 2x3 object
    g[4, 5] = 7       # a singleton
    canvas = jnp.asarray(G.place(g), jnp.int32)
    labels = connected_components(canvas, "nonblack4")
    rng = np.random.default_rng(1)
    z = jnp.asarray(rng.normal(size=(3, G.CANVAS, G.CANVAS, 5)).astype(np.float32))
    out = np.asarray(component_mean(z, labels))
    zn = np.asarray(z)
    obj = [(r, c) for r in (1, 2) for c in (1, 2, 3)]
    want = np.mean([zn[:, r, c, :] for r, c in obj], axis=0)
    for r, c in obj:
        assert np.allclose(out[:, r, c, :], want, atol=1e-5), "component mean wrong"
    assert np.allclose(out[:, 4, 5, :], zn[:, 4, 5, :], atol=1e-6), "singleton not identity"
    assert np.allclose(out[:, 20, 20, :], zn[:, 20, 20, :], atol=1e-6), "VOID cell not identity"
