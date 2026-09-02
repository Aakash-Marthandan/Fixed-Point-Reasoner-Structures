# Ledger: sportC1 X0 — the FIELD-RECIPE cell (Plan_2026-09-02_Champion_sportC1
# §11.1 / §11.2, 2026-09-02). A port of the public TRM code (Jolicoeur-Martineau
# 2025; models/recursive_reasoning/trm.py + models/layers.py, read 2026-09-02)
# which EqR (Huang, Geng, Kolter 2026, appendix D.1) states it uses verbatim
# for Sudoku: L_layers 2, H_cycles 3, L_cycles 6, hidden 512, 5.03M params,
# MLP-mixer token mixing, POST-norm parameter-free RMSNorm, SwiGLU (expansion 4,
# inter rounded up to a multiple of 256), sqrt(hidden)-scaled embeddings, a
# zero-init 16-token puzzle prefix (one identifier on Sudoku), fixed
# trunc-normal(std 1) initial states z_H / z_L, the gradient through the LAST
# H-cycle only, and EqR's per-pass damping/noise z <- z + (1-lambda)(F(z) - z)
# + beta*eps (Eq. 2; lambda .05, beta .01) with RI z0 ~ N(0, sigma) (A.3).
# It is NOT S9-equivariant (their shuffle includes digit permutation: train
# X0 with --sudoku-digit-aug). No flux / rule / size channels exist here.
"""The TRM/EqR block stack as pure functions over explicit param pytrees.

  params = init_params(key, cfg, hw)          # cfg.trm_* dims
  logits, q, z_fine = forward_core(params, cfg, fields, z_in, rng)
  segment(params, cfg, emb, zH, zL, rng)      # one outer segment (the SOT unit)
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from qhrrn2.config import Config
from qhrrn2.grid import VOCAB

RMS_EPS = 1e-5            # TRM rms_norm_eps
INIT_SEED = 20260902      # the fixed z_H / z_L buffers (nn.Buffer in TRM; never trained)


def _find_multiple(a: int, b: int) -> int:
    return (-(a // -b)) * b


def _trunc_normal(key, shape, std):
    """TRM's trunc_normal_init_: normal truncated at +-2 std."""
    return std * jax.random.truncated_normal(key, -2.0, 2.0, shape)


def _linear_init(key, n_in, n_out):
    return _trunc_normal(key, (n_in, n_out), 1.0 / math.sqrt(n_in))   # LeCun


def _swiglu_init(key, n, expansion):
    inter = _find_multiple(round(expansion * n * 2 / 3), 256)
    k1, k2 = jax.random.split(key)
    return {"gate_up": _linear_init(k1, n, 2 * inter), "down": _linear_init(k2, inter, n)}


def _swiglu(p, x):
    gate, up = jnp.split(x @ p["gate_up"], 2, axis=-1)
    return (jax.nn.silu(gate) * up) @ p["down"]


def _rms_norm(x):
    return x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + RMS_EPS)


def seq_len(cfg: Config, hw: int) -> int:
    return hw + cfg.trm_puzzle_emb_len


def init_params(key, cfg: Config, hw: int = 81):
    S, hid = seq_len(cfg, hw), cfg.trm_hidden
    ks = jax.random.split(key, 2 + 2 * cfg.trm_layers)
    blocks = [{"mlp_t": _swiglu_init(ks[2 + 2 * i], S, cfg.trm_expansion),
               "mlp": _swiglu_init(ks[3 + 2 * i], hid, cfg.trm_expansion)}
              for i in range(cfg.trm_layers)]
    return {"tok_emb": _trunc_normal(ks[0], (VOCAB, hid), 1.0 / math.sqrt(hid)),
            "puzzle_emb": jnp.zeros((cfg.trm_puzzle_emb_len, hid)),   # zero-init (TRM)
            "lm_head": _linear_init(ks[1], hid, VOCAB),                # bias-free
            "q_head": {"w": jnp.zeros((hid, 2)), "b": jnp.full((2,), -5.0)},  # TRM's Q init
            "blocks": blocks}


def init_states(cfg: Config):
    """H_init / L_init: fixed trunc-normal(std 1) vectors (TRM nn.Buffer,
    persistent, never trained) — regenerated deterministically so they are
    neither optimized nor weight-decayed."""
    kH, kL = jax.random.split(jax.random.PRNGKey(INIT_SEED))
    return _trunc_normal(kH, (cfg.trm_hidden,), 1.0), _trunc_normal(kL, (cfg.trm_hidden,), 1.0)


def z0(cfg: Config, hw: int, rng=None):
    """(2, S, hid) initial carry: the fixed buffers, or EqR's RI draw
    z ~ N(0, sigma I) when cfg.trm_ri_sigma > 0 and an rng is threaded."""
    S, hid = seq_len(cfg, hw), cfg.trm_hidden
    if cfg.trm_ri_sigma > 0 and rng is not None:
        return cfg.trm_ri_sigma * jax.random.normal(rng, (2, S, hid))
    H0, L0 = init_states(cfg)
    return jnp.stack([jnp.broadcast_to(H0, (S, hid)), jnp.broadcast_to(L0, (S, hid))])


def _block(p, h):
    """POST-norm block (TRM comment: 'Post Norm'): token-mixing SwiGLU over
    the sequence axis, then the channel SwiGLU, each residual + RMSNorm."""
    ht = h.T                                            # (hid, S)
    ht = _rms_norm(ht + _swiglu(p["mlp_t"], ht))
    h = ht.T                                            # (S, hid)
    return _rms_norm(h + _swiglu(p["mlp"], h))


def _stack(p, h):
    for b in p["blocks"]:
        h = _block(b, h)
    return h


def embed(p, cfg: Config, x_tokens):
    """x_tokens (H, W) ints in 0..VOCAB-1 -> (S, hid): [puzzle prefix ; token
    embeddings], scaled by sqrt(hid) (TRM _input_embeddings, no pos-enc for MLP-T)."""
    e = jnp.concatenate([p["puzzle_emb"], p["tok_emb"][x_tokens.reshape(-1)]], axis=0)
    return math.sqrt(cfg.trm_hidden) * e


def segment(p, cfg: Config, emb, zH, zL, rng=None):
    """ONE outer segment = H_cycles x (L_cycles + 1) block-stack passes:
    z_L <- F(z_L + z_H + emb) L_cycles times, then z_H <- F(z_H + z_L); the
    gradient flows through the LAST H-cycle only (TRM: 'H_cycles-1 without
    grad'). EqR Eq. 2 per pass: z <- z + (1 - lambda)(F(z) - z) + beta*eps
    (lambda = beta = 0 recovers TRM exactly; noise only with an rng)."""
    lam, beta = cfg.trm_lambda, cfg.trm_beta
    n_keys = cfg.trm_h_cycles * (cfg.trm_l_cycles + 1)
    keys = (list(jax.random.split(rng, n_keys)) if (rng is not None and beta > 0)
            else [None] * n_keys)

    def step(z, inj, key):
        F = _stack(p, z + inj)
        z2 = (z + (1.0 - lam) * (F - z)) if lam > 0 else F
        if key is not None:
            z2 = z2 + beta * jax.random.normal(key, z2.shape)
        return z2

    k = 0
    for c in range(cfg.trm_h_cycles):
        for _ in range(cfg.trm_l_cycles):
            zL = step(zL, zH + emb, keys[k]); k += 1
        zH = step(zH, zL, keys[k]); k += 1
        if c < cfg.trm_h_cycles - 1:
            zH, zL = jax.lax.stop_gradient(zH), jax.lax.stop_gradient(zL)
    return zH, zL


def readout(p, cfg: Config, zH, hw_shape):
    """logits (H, W, VOCAB) from the 81 cell tokens; q = (q_halt, q_continue)
    logits from the first prefix token (TRM q_head)."""
    P = cfg.trm_puzzle_emb_len
    logits = (zH[P:] @ p["lm_head"]).reshape(tuple(hw_shape) + (VOCAB,))
    q = zH[0] @ p["q_head"]["w"] + p["q_head"]["b"]
    return logits, q


def forward_core(p, cfg: Config, fields, *, z_in=None, rng=None):
    """model.forward_fields contract: fields (VOCAB, H, W, 2) [x one-hot at
    [..., 0]; the y slot is NOT read], carried z_in (2, S, hid) or None ->
    (logits (H, W, VOCAB), q (2,), z_fine (2, S, hid))."""
    H, W = fields.shape[1], fields.shape[2]
    x_tokens = jnp.argmax(fields[..., 0], axis=0)            # exact on one-hot input
    emb = embed(p, cfg, x_tokens)
    k_ri, k_seg = (None, None) if rng is None else tuple(jax.random.split(rng))
    z = z0(cfg, H * W, rng=k_ri) if z_in is None else z_in
    zH, zL = segment(p, cfg, emb, z[0], z[1], rng=k_seg)
    logits, q = readout(p, cfg, zH, (H, W))
    return logits, q, jnp.stack([zH, zL])
