# Ledger: C4 (seam mixer, offset blocks), C5 (pool/split -> priced streams),
#         C6+C14 (priced attention at all scales, Amendments B+D),
#         C7 (axial summaries, Amendment C),
#         C13 (nonlinearity everywhere; guarded by CI-2 in tests/test_model.py).
"""The RG cell's building blocks, as pure functions over explicit param pytrees.

State convention: Z has shape (C, H, W, d) — leading axis is the field axis
(9 symmetric colors + black + void). Every op treats fields identically
(weights shared over C), so S9 equivariance holds by construction; black/void
specificity enters only through role embeddings added at the model level.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp


def _linear_init(key, n_in, n_out, scale=None):
    scale = scale if scale is not None else 1.0 / math.sqrt(n_in)
    return {
        "w": jax.random.normal(key, (n_in, n_out)) * scale,
        "b": jnp.zeros((n_out,)),
    }


def _linear(p, x):
    return x @ p["w"] + p["b"]


# ── Seam mixer (C4 + C7): offset 2x2 blocks, axial context, GELU residual ──

def init_mixer(key, d):
    k1, k2 = jax.random.split(key)
    n_in = 16 * d          # 4 sites x [z, field-mean, row-mean, col-mean]
    hidden = 4 * d
    return {"l1": _linear_init(k1, n_in, hidden), "l2": _linear_init(k2, hidden, 4 * d)}


def mixer(p, z):
    """z: (C, H, W, d) -> (C, H, W, d). Blocks offset by (1,1) via pad+crop."""
    C, H, W, d = z.shape
    zbar = jnp.mean(z, axis=0, keepdims=True)                    # (1,H,W,d) color context
    row = jnp.mean(z, axis=2, keepdims=True)                     # (C,H,1,d) axial (C7)
    col = jnp.mean(z, axis=1, keepdims=True)                     # (C,1,W,d)
    aug = jnp.concatenate([
        z,
        jnp.broadcast_to(zbar, z.shape),
        jnp.broadcast_to(row, z.shape),
        jnp.broadcast_to(col, z.shape),
    ], axis=-1)                                                  # (C,H,W,4d)

    padded = jnp.pad(aug, ((0, 0), (1, 1), (1, 1), (0, 0)))      # (C,H+2,W+2,4d)
    Hb, Wb = (H + 2) // 2, (W + 2) // 2
    blocks = padded.reshape(C, Hb, 2, Wb, 2, 4 * d).transpose(0, 1, 3, 2, 4, 5)
    flat = blocks.reshape(C, Hb, Wb, 16 * d)

    h = jax.nn.gelu(_linear(p["l1"], flat))
    delta = _linear(p["l2"], h).reshape(C, Hb, Wb, 2, 2, d)

    delta = delta.transpose(0, 1, 3, 2, 4, 5).reshape(C, Hb * 2, Wb * 2, d)
    return z + delta[:, 1:H + 1, 1:W + 1, :]                     # crop back; residual


# ── Pool & split (C5): aligned axa -> kept d  ⊕  stream (mu, log_sigma) ────
# arity 2 = the dyadic pyramid (every checkpoint before 2026-09-01, bit-exact);
# arity 3 = the CHAMPION TRACK's 3-adic pyramid (9 -> 3 -> 1) where each
# level-1 pooling block IS a Sudoku box (Plan_2026-09-01 §1).

def init_pool_split(key, d, d_b, arity: int = 2):
    k1, k2 = jax.random.split(key)
    n_in = arity * arity * d
    return {"kept": _linear_init(k1, n_in, d), "stream": _linear_init(k2, n_in, 2 * d_b)}


def pool_split(p, z, d_b, arity: int = 2):
    """(C,H,W,d) -> kept (C,H/a,W/a,d), mu/log_sigma (C,H/a,W/a,d_b) each."""
    C, H, W, d = z.shape
    a = arity
    u = z.reshape(C, H // a, a, W // a, a, d).transpose(0, 1, 3, 2, 4, 5)
    u = u.reshape(C, H // a, W // a, a * a * d)
    kept = jax.nn.gelu(_linear(p["kept"], u))
    stats = _linear(p["stream"], u)
    mu, log_sigma = stats[..., :d_b], stats[..., d_b:]
    log_sigma = jnp.clip(log_sigma, -6.0, 2.0)
    return kept, mu, log_sigma


def stream_kl(mu, log_sigma):
    """KL(N(mu, sigma) || N(0,1)) summed over everything — nats crossing this cut."""
    return 0.5 * jnp.sum(mu ** 2 + jnp.exp(2 * log_sigma) - 2 * log_sigma - 1.0)


# ── Priced attention (C6+C14): the nonlocal channel, tolled at the mouth ───

def init_attention(key, d, d_a):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    return {
        "q": _linear_init(k1, d, d), "k": _linear_init(k2, d, d),
        "vib": _linear_init(k3, d, 2 * d_a),
        "o": _linear_init(k4, d_a, d, scale=1e-2),
    }


def attention(p, z, rng=None):
    """Residual attention as a *priced* channel (Amendment D). Each site emits
    a variational message (mu, log_sigma) of width d_a; the toll
    A = KL(N(mu, sigma) || N(0,1)) is the nats this scale spends on
    nonlocality. Messages are mixed by the attention pattern — a convex
    combination of the sampled codes, so by data processing A upper-bounds
    what actually crosses. S9-safe: the pattern comes from the field-mean and
    all projections are shared across fields. Returns (z', A)."""
    C, H, W, d = z.shape
    zbar = jnp.mean(z, axis=0).reshape(H * W, d)
    q, k = _linear(p["q"], zbar), _linear(p["k"], zbar)
    a = jax.nn.softmax(q @ k.T / math.sqrt(d), axis=-1)          # (HW, HW)
    stats = _linear(p["vib"], z.reshape(C, H * W, d))
    d_a = stats.shape[-1] // 2
    mu, log_sigma = stats[..., :d_a], jnp.clip(stats[..., d_a:], -6.0, 2.0)
    if rng is not None:
        m = mu + jnp.exp(log_sigma) * jax.random.normal(rng, mu.shape)
    else:
        m = mu
    flux = stream_kl(mu, log_sigma)
    out = jnp.einsum("ts,csa->cta", a, m)
    return z + _linear(p["o"], out).reshape(C, H, W, d), flux


# ── FiLM modulation by (scale, iteration) (C13/H-8) ────────────────────────

def init_film(key, d, n_points=2):
    k1, k2 = jax.random.split(key)
    return {
        "l1": _linear_init(k1, 2, 16),
        "l2": _linear_init(k2, 16, n_points * 2 * d, scale=1e-3),
    }


def film_params(p, s_norm, t_norm, d):
    x = jnp.array([s_norm, t_norm])
    out = _linear(p["l2"], jax.nn.gelu(_linear(p["l1"], x)))
    out = out.reshape(-1, 2, d)
    return out[:, 0, :], out[:, 1, :]                            # gammas, betas


def film(z, gamma, beta):
    return z * (1.0 + gamma) + beta


# ── Stream injection at decode (C5): gated, per-field-shared ───────────────

def init_inject(key, d_b, d):
    return {"proj": _linear_init(key, d_b, d)}


def inject(p, z, b, gate):
    """z: (C,H,W,d); b: (C,H,W,d_b); gate: (d_b,) in [0,1]."""
    return z + _linear(p["proj"], b * gate)


def upsample(z, arity: int = 2):
    return jnp.repeat(jnp.repeat(z, arity, axis=1), arity, axis=2)


# ── The GROUP mixer (CHAMPION TRACK, Plan_2026-09-01 §2) ───────────────────
# ONE shared "all-different" operator applied to every Sudoku constraint
# group: at s0 (9x9) the partitions are rows / cols / boxes (27 groups of 9
# cells — Sudoku's exact constraint basis); at s1 (3x3) the whole box grid is
# one group (partition type 3). Weight-tied across partitions because the
# all-different constraint is IDENTICAL for every unit; per-partition-type and
# per-slot embeddings restore the (cheap) distinctions. Param-safe: ~0.5M at
# d=96 vs ~6M for a naive 3x3-window concat mixer (the named param trap).
# S9-safe: every op is shared over the field axis C; context = field-mean.

N_GROUP = 9
PART_TYPES = 4          # 0 = rows, 1 = cols, 2 = boxes, 3 = whole-grid (s1)


def init_group_mixer(key, d):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    return {
        "l1": _linear_init(k1, 2 * N_GROUP * d, 2 * d),
        "l2": _linear_init(k2, 2 * d, N_GROUP * d),
        "type_emb": jax.random.normal(k3, (PART_TYPES, d)) * 0.1,
        "slot_emb": jax.random.normal(k4, (N_GROUP, d)) * 0.1,
    }


def _group_apply(p, zg, zbg, ptype):
    """zg, zbg: (C, G, 9, d) group views of z and the field-mean context.
    Returns (C, G, 9, d) deltas."""
    C, Gn, S, d = zg.shape
    m = zg + p["type_emb"][ptype] + p["slot_emb"][None, None, :, :]
    aug = jnp.concatenate([m, zbg], axis=-1).reshape(C, Gn, 2 * S * d)
    h = jax.nn.gelu(_linear(p["l1"], aug))
    return _linear(p["l2"], h).reshape(C, Gn, S, d)


def group_mixer(p, z):
    """z: (C, H, W, d) with (H, W) = (9, 9) [s0] or (3, 3) [s1]."""
    C, H, W, d = z.shape
    zb = jnp.broadcast_to(jnp.mean(z, axis=0, keepdims=True), z.shape)
    if H == 3 and W == 3:   # s1: the box grid as ONE group of 9 tokens
        zg = z.reshape(C, 1, N_GROUP, d)
        zbg = zb.reshape(C, 1, N_GROUP, d)
        delta = _group_apply(p, zg, zbg, 3).reshape(C, H, W, d)
        return z + delta
    assert H == 9 and W == 9, (H, W)

    def box_view(t):        # (C,9,9,d) -> (C, box g, slot, d), g=(br,bc), slot=(r%3,c%3)
        return (t.reshape(C, 3, 3, 3, 3, d).transpose(0, 1, 3, 2, 4, 5)
                .reshape(C, N_GROUP, N_GROUP, d))

    def box_unview(t):
        return (t.reshape(C, 3, 3, 3, 3, d).transpose(0, 1, 3, 2, 4, 5)
                .reshape(C, 9, 9, d))

    d_rows = _group_apply(p, z, zb, 0)                                 # groups = rows
    d_cols = _group_apply(p, z.transpose(0, 2, 1, 3),
                          zb.transpose(0, 2, 1, 3), 1).transpose(0, 2, 1, 3)
    d_box = box_unview(_group_apply(p, box_view(z), box_view(zb), 2))
    return z + (d_rows + d_cols + d_box) / 3.0
