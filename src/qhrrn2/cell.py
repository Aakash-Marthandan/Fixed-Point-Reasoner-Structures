# Ledger: C4 (seam mixer, offset blocks), C5 (pool/split -> priced streams),
#         C6 (coarse attention, Amendment B), C7 (axial summaries, Amendment C),
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


# ── Pool & split (C5): aligned 2x2 -> kept d  ⊕  stream (mu, log_sigma) ────

def init_pool_split(key, d, d_b):
    k1, k2 = jax.random.split(key)
    return {"kept": _linear_init(k1, 4 * d, d), "stream": _linear_init(k2, 4 * d, 2 * d_b)}


def pool_split(p, z, d_b):
    """(C,H,W,d) -> kept (C,H/2,W/2,d), mu/log_sigma (C,H/2,W/2,d_b) each."""
    C, H, W, d = z.shape
    u = z.reshape(C, H // 2, 2, W // 2, 2, d).transpose(0, 1, 3, 2, 4, 5)
    u = u.reshape(C, H // 2, W // 2, 4 * d)
    kept = jax.nn.gelu(_linear(p["kept"], u))
    stats = _linear(p["stream"], u)
    mu, log_sigma = stats[..., :d_b], stats[..., d_b:]
    log_sigma = jnp.clip(log_sigma, -6.0, 2.0)
    return kept, mu, log_sigma


def stream_kl(mu, log_sigma):
    """KL(N(mu, sigma) || N(0,1)) summed over everything — nats crossing this cut."""
    return 0.5 * jnp.sum(mu ** 2 + jnp.exp(2 * log_sigma) - 2 * log_sigma - 1.0)


# ── Coarse attention (C6): pattern from color-mean, applied to every field ─

def init_attention(key, d):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    return {
        "q": _linear_init(k1, d, d), "k": _linear_init(k2, d, d),
        "v": _linear_init(k3, d, d), "o": _linear_init(k4, d, d, scale=1e-2),
    }


def attention(p, z):
    """Residual single-head attention over all sites; S9-safe: the attention
    pattern is computed from the field-mean (equivariant scalar function) and
    shared across fields."""
    C, H, W, d = z.shape
    zbar = jnp.mean(z, axis=0).reshape(H * W, d)
    q, k = _linear(p["q"], zbar), _linear(p["k"], zbar)
    a = jax.nn.softmax(q @ k.T / math.sqrt(d), axis=-1)          # (HW, HW)
    v = _linear(p["v"], z.reshape(C, H * W, d))
    out = jnp.einsum("ts,csd->ctd", a, v)
    return z + _linear(p["o"], out).reshape(C, H, W, d)


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


def upsample(z):
    return jnp.repeat(jnp.repeat(z, 2, axis=1), 2, axis=2)
