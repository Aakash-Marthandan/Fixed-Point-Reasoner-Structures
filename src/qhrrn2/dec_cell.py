# Ledger: FINAL PHASE — the DECIMATING EQUILIBRIUM CELL (DEC; Plan_2026-09-05_FinalPhase §2,
# 2026-09-05). The field's block and two-timescale loop (qhrrn2.trm_cell: the TRM/EqR port)
# on OUR field-structured state (C2): z_H / z_L of shape (F = 9 digit fields, S = 81 cells, w),
# every parameter shared over the field axis, so a permutation of the digits permutes the
# state and the logits exactly (tests/test_final.py::test_dec_exact_s9) — the symmetry the
# field pays 1000x digit augmentation for (sportC2 X1: -65 pp without it). Per block: the
# token-mixing SwiGLU over the 81 cells per field (TRM's mlp_t, vmapped over the shared-weight
# field axis), an equivariant FIELD COUPLING (each field's cell token reads the mean of the
# OTHER eight fields' tokens at that cell — the all-different message in DeepSets form; a
# residual + RMSNorm sub-layer, cfg.dec_coupling), then the channel SwiGLU (TRM's mlp).
# Givens enter as a per-(field, cell) ROLE embedding (empty / given-as-this-digit /
# given-as-another-digit); no puzzle prefix (zero-init, one identifier on Sudoku: inert);
# the halting logits read the field-and-cell MEAN of z_H (S9-invariant); the readout is one
# shared vector per field token -> 9 logits per cell (classes 0 and VOID padded at -1e4:
# never a target on Sudoku). embed_answer = the FPA anchor state (the embedded corrupted
# solution as z_H; plan §2 "our objectives on the loop", tools/pretrain.py::field_fpa_loss).
# The loop dials are the field's (cfg.trm_layers / h_cycles / l_cycles / lambda / beta /
# ri_sigma / expansion); the width dial is cfg.dec_width (256 = the A-night arm; 512 = X0's
# parameter count). cfg.remat checkpoints each stack pass (the chain's OOM retry).
"""The DEC block stack as pure functions over explicit param pytrees.

  params = init_params(key, cfg, hw)          # cfg.dec_width, cfg.trm_* loop dials
  logits, q, z_fine = forward_core(params, cfg, fields, z_in, rng)
  segment(params, cfg, emb, zH, zL, rng)      # one outer segment (the SOT unit)
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from qhrrn2.config import Config
from qhrrn2.grid import VOCAB
from qhrrn2 import trm_cell as TC

F = 9                     # digit fields 1..9 (the S9-symmetric set)
PAD_LOGIT = -1e4          # classes 0 (blank) and VOID: never a Sudoku target (finite: no NaN gradient)


def carry_shape(cfg: Config, hw: int = 81):
    return (2, F, hw, cfg.dec_width)


def init_params(key, cfg: Config, hw: int = 81):
    w, S = cfg.dec_width, hw
    ks = jax.random.split(key, 4 + 3 * cfg.trm_layers)
    blocks = []
    for i in range(cfg.trm_layers):
        b = {"mlp_t": TC._swiglu_init(ks[4 + 3 * i], S, cfg.trm_expansion),    # token mixing over the 81 cells
             "mlp": TC._swiglu_init(ks[5 + 3 * i], w, cfg.trm_expansion)}      # channel SwiGLU
        if cfg.dec_coupling:
            b["fc"] = TC._linear_init(ks[6 + 3 * i], w, w)                       # the field coupling (w, w)
        blocks.append(b)
    return {"role_emb": TC._trunc_normal(ks[0], (3, w), 1.0 / math.sqrt(w)),   # empty / mine / other
            "lm_head": TC._trunc_normal(ks[2], (w,), 1.0 / math.sqrt(w)),      # shared readout vector
            "q_head": {"w": jnp.zeros((w, 2)), "b": jnp.full((2,), -5.0)},     # TRM's Q init
            "blocks": blocks}


def init_states(cfg: Config):
    """H_init / L_init: fixed trunc-normal(std 1) vectors (TRM nn.Buffer; never trained),
    regenerated deterministically from the same seed as the field cell's."""
    kH, kL = jax.random.split(jax.random.PRNGKey(TC.INIT_SEED))
    return TC._trunc_normal(kH, (cfg.dec_width,), 1.0), TC._trunc_normal(kL, (cfg.dec_width,), 1.0)


def z0(cfg: Config, hw: int, rng=None):
    """(2, F, S, w) initial carry: the fixed buffers broadcast over fields and cells (identical
    across fields: exactly S9-invariant), or EqR's RI draw z ~ N(0, sigma I) when
    cfg.trm_ri_sigma > 0 and an rng is threaded (exchangeable across fields)."""
    w = cfg.dec_width
    if cfg.trm_ri_sigma > 0 and rng is not None:
        return cfg.trm_ri_sigma * jax.random.normal(rng, (2, F, hw, w))
    H0, L0 = init_states(cfg)
    return jnp.stack([jnp.broadcast_to(H0, (F, hw, w)), jnp.broadcast_to(L0, (F, hw, w))])


def _block(p, h):
    """h (F, S, w) -> (F, S, w). POST-norm as TRM: per-field token mixing over the S cells (the
    same weights for every field), the equivariant field coupling, the channel SwiGLU."""
    def tok(hf):                                   # (S, w): TRM's token-mixing sub-layer
        ht = hf.T
        ht = TC._rms_norm(ht + TC._swiglu(p["mlp_t"], ht))
        return ht.T
    h = jax.vmap(tok)(h)
    if "fc" in p:
        others = (jnp.sum(h, axis=0, keepdims=True) - h) / (F - 1)   # the mean of the OTHER fields, per cell
        h = TC._rms_norm(h + others @ p["fc"])
    return TC._rms_norm(h + TC._swiglu(p["mlp"], h))


def _stack(p, h):
    for b in p["blocks"]:
        h = _block(b, h)
    return h


def embed(p, cfg: Config, x_tokens):
    """x_tokens (H, W) ints in 0..VOCAB-1 -> (F, S, w): the ROLE of each (field f = digit f+1, cell):
    0 = empty cell, 1 = given AS this digit, 2 = given as another digit; scaled by sqrt(w)
    (TRM's embedding scale). A digit permutation permutes the field axis of this tensor."""
    x = x_tokens.reshape(-1)
    digit = jnp.arange(1, F + 1)[:, None]
    role = jnp.where(x[None, :] == 0, 0, jnp.where(x[None, :] == digit, 1, 2))
    return math.sqrt(cfg.dec_width) * p["role_emb"][role]


def embed_answer(p, cfg: Config, y_grid):
    """The FPA anchor state (Plan §2): z_H := the embedded (corrupted) solution read through the
    INPUT's own role table — field f sees 'given AS my digit' (role 1) where the answer is its digit
    and 'given as another digit' (role 2) elsewhere — as the field cell anchors through tok_emb.
    (The loop detaches every H-cycle but the last, so a separate anchor table would never train;
    the role table trains through the input path.) (F, S, w)."""
    y = y_grid.reshape(-1)
    mine = y[None, :] == jnp.arange(1, F + 1)[:, None]
    return math.sqrt(cfg.dec_width) * p["role_emb"][jnp.where(mine, 1, 2)]


def segment(p, cfg: Config, emb, zH, zL, rng=None):
    """ONE outer segment = H_cycles x (L_cycles + 1) stack passes, exactly the field cell's
    (trm_cell.segment): z_L <- F(z_L + z_H + emb) L_cycles times, then z_H <- F(z_H + z_L);
    the gradient through the LAST H-cycle only; EqR Eq. 2 damping / noise per pass."""
    lam, beta = cfg.trm_lambda, cfg.trm_beta
    n_keys = cfg.trm_h_cycles * (cfg.trm_l_cycles + 1)
    keys = (list(jax.random.split(rng, n_keys)) if (rng is not None and beta > 0)
            else [None] * n_keys)
    stack = jax.checkpoint(_stack) if cfg.remat else _stack

    def step(z, inj, key):
        Fz = stack(p, z + inj)
        z2 = (z + (1.0 - lam) * (Fz - z)) if lam > 0 else Fz
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
    """logits (H, W, VOCAB): digit d's logit at a cell = that field's cell token . lm_head (one
    shared vector), classes 0 and VOID padded at PAD_LOGIT; q (2,) = (q_halt, q_continue) from the
    field-and-cell mean of z_H (S9-invariant)."""
    lg = jnp.einsum("fsw,w->sf", zH, p["lm_head"])                       # (S, F)
    S = lg.shape[0]
    left = jnp.full((S, 1), PAD_LOGIT, lg.dtype)
    right = jnp.full((S, VOCAB - F - 1), PAD_LOGIT, lg.dtype)
    logits = jnp.concatenate([left, lg, right], axis=1).reshape(tuple(hw_shape) + (VOCAB,))
    q = jnp.mean(zH, axis=(0, 1)) @ p["q_head"]["w"] + p["q_head"]["b"]
    return logits, q


def forward_core(p, cfg: Config, fields, *, z_in=None, rng=None):
    """model.forward_fields contract: fields (VOCAB, H, W, 2) [x one-hot at [..., 0]; the y slot
    is NOT read], carried z_in (2, F, S, w) or None -> (logits (H, W, VOCAB), q (2,), z_fine)."""
    H, W = fields.shape[1], fields.shape[2]
    x_tokens = jnp.argmax(fields[..., 0], axis=0)            # exact on one-hot input
    emb = embed(p, cfg, x_tokens)
    k_ri, k_seg = (None, None) if rng is None else tuple(jax.random.split(rng))
    z = z0(cfg, H * W, rng=k_ri) if z_in is None else z_in
    zH, zL = segment(p, cfg, emb, z[0], z[1], rng=k_seg)
    logits, q = readout(p, cfg, zH, (H, W))
    return logits, q, jnp.stack([zH, zL])
