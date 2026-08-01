# Ledger: C1-v3 named tests (registered 2026-08-02 BEFORE the build ran):
#   t1 construction-expressibility — same-size, x2, x3, transpose, and
#      count-valued sizes each reachable exactly by some (selection, offset),
#   t2 count extrapolation — out_w = colored-cell count fitted on small
#      counts must decode UNSEEN larger counts (the trap absolute classes
#      failed, measured 2026-07-27; and v2's offset frame cannot express),
#   t3 v2-regression — extent-relative offsets still exact via candidate 0.
import numpy as np
import jax
import jax.numpy as jnp
import optax
import pytest

from qhrrn2 import grid as G
from qhrrn2.config import Config
from qhrrn2.model import (N_SIZE_CANDS, init_params, size_candidates,
                          size_mixture_probs)
from qhrrn2.objective import batch_loss
from qhrrn2.train import predict


def _one_hot_logits(i, n):
    return jnp.where(jnp.arange(n) == i, 50.0, -50.0)


def _decode(sel_idx, off_delta, cands_axis):
    p = size_mixture_probs(_one_hot_logits(sel_idx, N_SIZE_CANDS),
                           _one_hot_logits(off_delta + 15, 30), cands_axis)
    return int(jnp.argmax(p)) + 1


# ── t1: expressibility by construction ──────────────────────────────────────

def test_t1_candidates_and_expressibility():
    g = np.zeros((7, 5), dtype=np.int8)
    g[0, :3] = 4          # 3 colored cells of color 4
    g[2, 1] = 7           # 1 of color 7
    x = jnp.asarray(G.place(g), dtype=jnp.int32)
    cands = np.asarray(size_candidates(x))
    # axis h: [h, w, 2h, 3h, ceil(h/2), ceil(h/3), occ_rows, top1]
    assert list(cands[0]) == [7, 5, 14, 21, 4, 3, 2, 3]
    assert list(cands[1]) == [5, 7, 10, 15, 3, 2, 3, 3]

    ch = jnp.asarray(cands[0])
    assert _decode(0, 0, ch) == 7          # same-size
    assert _decode(1, 0, ch) == 5          # transpose
    assert _decode(2, 0, ch) == 14         # x2
    assert _decode(3, 0, ch) == 21         # x3
    assert _decode(7, 0, ch) == 3          # count-valued (top color = 3 cells)
    assert _decode(7, 1, ch) == 4          # count + offset


# ── t3: v2 regression — extent-relative offsets via candidate 0 ────────────

def test_t3_v2_frame_is_candidate_zero():
    for h, dlt in [(9, 0), (9, -3), (4, 11), (30, 0)]:
        g = np.ones((h, 3), dtype=np.int8)
        x = jnp.asarray(G.place(g), dtype=jnp.int32)
        cands = size_candidates(x)[0]
        assert _decode(0, dlt, cands) == int(np.clip(h + dlt, 1, 30))


# ── t2: count extrapolation through a real fit ──────────────────────────────

def test_t2_count_extrapolation():
    """Fit 1xN-bar episodes (N = colored cells) on N in {2,3,4,6}; the decoded
    WIDTH must be right on unseen N in {8, 11}. Content is not asserted —
    this is the size head's gate, not a solve gate."""
    cfg = Config(d=8, T=1, K=8, attn_max_hw=0)
    rng = np.random.default_rng(0)

    def episode(n):
        g = np.zeros((6, 12), dtype=np.int8)
        cells = rng.choice(6 * 12, size=n, replace=False)
        g[np.unravel_index(cells, g.shape)] = 3
        y = np.full((1, n), 3, dtype=np.int8)
        return g, y

    xs, ys = zip(*(episode(n) for n in (2, 3, 4, 6) for _ in range(3)))
    x_b = jnp.asarray(np.stack([G.place(x) for x in xs]), dtype=jnp.int32)
    y_b = jnp.asarray(np.stack([G.place(y) for y in ys]), dtype=jnp.int32)

    params = init_params(jax.random.PRNGKey(0), cfg)
    opt = optax.adamw(1e-2, weight_decay=1e-4)
    opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state, key):
        (loss, _), grads = jax.value_and_grad(batch_loss, has_aux=True)(
            params, cfg, x_b, y_b, tau=1.0, rng=key)
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    key = jax.random.PRNGKey(1)
    for _ in range(150):
        key, sub = jax.random.split(key)
        params, opt_state, _ = step(params, opt_state, sub)

    for n_unseen in (8, 11):
        xq, _ = episode(n_unseen)
        _, (ph, pw), _ = predict(params, cfg, xq, tau=1.0)
        assert (ph, pw) == (1, n_unseen), (
            f"count extrapolation failed: N={n_unseen} -> predicted {(ph, pw)}")
