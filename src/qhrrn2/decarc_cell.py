# Ledger: DEC-ARC BUILD (Plan_2026-09-10_DEC-ARC_Build §1; 2026-09-10). The Decimating Equilibrium
# Cell generalized to ARC: the field's block + two-timescale loop (qhrrn2.trm_cell / dec_cell) on a
# TEN-FIELD COLOUR STATE — z_H / z_L of shape (F = 10 colour fields, S = canvas cells, w), every
# parameter shared over the field axis, so a permutation of the colours permutes the state and the
# colour logits EXACTLY (tests/test_decarc.py::test_decarc_exact_s10). Per block: multi-head
# self-attention over the S cells per field (the token mixer; the DEC's token-mixing SwiGLU over 81
# cells does not scale to 1,024 — this is why the field's ARC model is TRM-Attention), the
# equivariant FIELD COUPLING (each field reads the mean of the other nine fields' cell tokens; the
# DEC's, cfg.dec_coupling), then the channel SwiGLU; POST-norm as TRM. The INPUT enters as a
# per-(field, cell) ROLE embedding (the input's colour here IS this field / is another colour / the
# cell lies outside the input) plus a learned position table shared over the fields (S10-safe).
# The TASK enters as a per-colour code (F, d_task) — the natives' task-vector protocol (a learned
# row per training task; the deployed arm-A fit of the code on a task's support pairs at test
# time) made colour-indexed, so that the joint action of a colour permutation on the input AND on
# the code's field axis is an exact symmetry of the map: colour-specific rules survive exactness
# because the code carries them field by field. NO puzzle identity at test time (the field's
# memorization channel: 26 pp on its ARC-1 model), no support encoder (paper 2's object).
# The readout per cell: ten colour logits (one shared vector per field token) plus a VOID logit
# from the S10-INVARIANT cell mean — the output size is the crop of the non-void region
# (model.decode_size), the field's crop rule; the halting logits read the field-and-cell mean
# (as the DEC); the commit head (C6's) rides as the abstention instrument (dec_commit_tau >= 1:
# the head trains, the forward is untouched — asserted, no hardening on ARC).
# embed_answer = the FPA anchor state (the embedded corrupted OUTPUT, void cells included, through
# the input's own role table so it trains through the input path — dec_cell's note) = the DEC's
# handed-truth RETENTION instrument on ARC (the FPA start at eps = 0) and its basin-radius ladder.
# THE DETERMINISTIC EVALUATION START (2026-09-15; the width-192 long run's CPU lens, Report_2026-09-15_W192_Long_Verdict
# §8): an RI-trained cell trains every fresh row from z ~ N(0, sigma) and never from its fixed buffers, so the buffers are an
# UNCONTROLLED start for it (on Sudoku the fixed start left and re-entered the basin as the weights drifted while every
# random start solved). The cold pass of the monitor, the evaluator, predict and the fit's validation therefore starts
# from z0_eval: a SEEDED draw from the training family, shared by every input (deterministic), TIED OVER THE FIELD AXIS so
# that it is exactly S10-invariant and the cell's colour exactness survives the start (cfg.decarc_eval_start:
# "fieldfix" = one draw per (stream, cell, channel) broadcast over the ten fields [the default]; "symfix" = one vector per
# stream broadcast over fields and cells; "rifix" = an untied draw, NOT S10-invariant — a descriptive row only, never
# the selection or the cold row; "buffers" = the pre-2026-09-15 start). A plain cell (sigma 0) keeps its buffers (they
# are its training start). Training (an rng threaded) is untouched: the RI draw, or the buffers when sigma is 0.
"""The DEC-ARC block stack as pure functions over explicit param pytrees.

  params = init_params(key, cfg, hw)                      # cfg.dec_width, cfg.decarc_heads, the trm_* loop dials
  logits, q, z_fine = forward_core(params, cfg, fields, task_vec=code, z_in=..., rng=...)
  segment(params, cfg, emb, zH, zL, rng)                  # one outer segment (the SOT unit)
  z0_eval(cfg, hw[, kind])                                # the deterministic evaluation start (S10-invariant kinds)
"""
from __future__ import annotations

import functools
import math

import numpy as np
import jax
import jax.numpy as jnp

from qhrrn2.config import Config
from qhrrn2.grid import VOCAB, VOID, NUM_COLORS
from qhrrn2 import trm_cell as TC

F = NUM_COLORS            # the ten colour fields 0..9 (S10 by weight sharing; the background is a colour)
ROLE_MINE, ROLE_OTHER, ROLE_OUT = 0, 1, 2   # the input's colour here is this field / another colour / outside the input


def carry_shape(cfg: Config, hw: int):
    return (2, F, hw, cfg.dec_width)


def init_params(key, cfg: Config, hw: int):
    w, nh = cfg.dec_width, cfg.decarc_heads
    assert w % nh == 0, "dec_width must divide by decarc_heads"
    ks = jax.random.split(key, 8 + 4 * cfg.trm_layers)
    blocks = []
    for i in range(cfg.trm_layers):
        ka, kb, kc, kd = jax.random.split(ks[8 + 4 * i], 4)
        b = {"attn": {"q": TC._linear_init(ka, w, w), "k": TC._linear_init(kb, w, w),
                      "v": TC._linear_init(kc, w, w), "o": TC._linear_init(kd, w, w)},
             "mlp": TC._swiglu_init(ks[8 + 4 * i + 1], w, cfg.trm_expansion)}
        if cfg.dec_coupling:
            b["fc"] = TC._linear_init(ks[8 + 4 * i + 2], w, w)
        blocks.append(b)
    out = {"role_emb": TC._trunc_normal(ks[0], (3, w), 1.0 / math.sqrt(w)),      # mine / other / outside
           "pos_emb": TC._trunc_normal(ks[1], (hw, w), 1.0 / math.sqrt(w)),      # shared over the fields
           "task_proj": TC._linear_init(ks[2], cfg.d_task, w),                    # the per-colour code -> per-field token
           "lm_head": TC._trunc_normal(ks[3], (w,), 1.0 / math.sqrt(w)),          # one shared colour readout vector
           "void_head": {"w": TC._trunc_normal(ks[4], (w,), 1.0 / math.sqrt(w)), "b": jnp.zeros(())},
           "q_head": {"w": jnp.zeros((w, 2)), "b": jnp.full((2,), -5.0)},         # TRM's Q init
           "blocks": blocks}
    if cfg.dec_commit:
        out["commit_head"] = {"w": TC._trunc_normal(ks[5], (w,), 1.0 / math.sqrt(w)), "b": jnp.full((), -2.0)}
    return out


def init_states(cfg: Config):
    kH, kL = jax.random.split(jax.random.PRNGKey(TC.INIT_SEED))
    return TC._trunc_normal(kH, (cfg.dec_width,), 1.0), TC._trunc_normal(kL, (cfg.dec_width,), 1.0)


def z0(cfg: Config, hw: int, rng=None):
    """(2, F, S, w): the fixed buffers broadcast over fields and cells (exactly S10-invariant), or the
    RI draw z ~ N(0, sigma I) when cfg.trm_ri_sigma > 0 and an rng is threaded. The FPA anchor's z_L and the
    plain cell's start read this; the deterministic evaluation start is z0_eval."""
    w = cfg.dec_width
    if cfg.trm_ri_sigma > 0 and rng is not None:
        return cfg.trm_ri_sigma * jax.random.normal(rng, (2, F, hw, w))
    H0, L0 = init_states(cfg)
    return jnp.stack([jnp.broadcast_to(H0, (F, hw, w)), jnp.broadcast_to(L0, (F, hw, w))])


EVAL_START_SEED = 20260915                                   # the seeded evaluation draw (the date the start was defined)
EVAL_STARTS = ("buffers", "symfix", "fieldfix", "rifix")      # the S10-invariant kinds: buffers, symfix, fieldfix
EVAL_START_INVARIANT = ("buffers", "symfix", "fieldfix")


@functools.lru_cache(maxsize=8)
def _eval_start_draw(kind: str, hw: int, w: int, sigma: float):
    """NumPy draws (platform-independent bits: the same start on the CPU and the TPU), cached per shape."""
    g = np.random.default_rng([EVAL_START_SEED, {"symfix": 1, "fieldfix": 2, "rifix": 3}[kind]])
    shape = {"symfix": (2, 1, 1, w), "fieldfix": (2, 1, hw, w), "rifix": (2, F, hw, w)}[kind]
    return (sigma * g.standard_normal(shape)).astype(np.float32)


def z0_eval(cfg: Config, hw: int, kind: str | None = None):
    """The DETERMINISTIC EVALUATION START (2, F, S, w) of an RI-trained DEC-ARC (see the module note): kind = cfg.decarc_eval_start
    unless given. The buffers when the cell trains from them (sigma 0) or kind = "buffers"; otherwise the seeded draw
    sigma * N(0, 1) tied over the fields ("fieldfix"), over fields and cells ("symfix"), or untied ("rifix", not invariant)."""
    kind = cfg.decarc_eval_start if kind is None else kind
    assert kind in EVAL_STARTS, kind
    if kind == "buffers" or cfg.trm_ri_sigma <= 0:
        return z0(cfg, hw)
    g = _eval_start_draw(kind, hw, cfg.dec_width, float(cfg.trm_ri_sigma))
    return jnp.broadcast_to(jnp.asarray(g), (2, F, hw, cfg.dec_width))


def _attention(pa, h, nh: int):
    """h (F, S, w) -> (F, S, w): multi-head self-attention over the S cells, per field, shared weights."""
    Fd, S, w = h.shape
    dk = w // nh
    q = (h @ pa["q"]).reshape(Fd, S, nh, dk)
    k = (h @ pa["k"]).reshape(Fd, S, nh, dk)
    v = (h @ pa["v"]).reshape(Fd, S, nh, dk)
    e = jnp.einsum("fshd,fthd->fhst", q, k) / math.sqrt(dk)
    att = jax.nn.softmax(e, axis=-1)
    o = jnp.einsum("fhst,fthd->fshd", att, v).reshape(Fd, S, w)
    return o @ pa["o"]


def _block(p, h, cfg: Config):
    """POST-norm block: attention over the cells (per field), the field coupling, the channel SwiGLU."""
    h = TC._rms_norm(h + _attention(p["attn"], h, cfg.decarc_heads))
    if "fc" in p:
        others = (jnp.sum(h, axis=0, keepdims=True) - h) / (F - 1)        # the mean of the OTHER nine fields, per cell
        h = TC._rms_norm(h + others @ p["fc"])
    return TC._rms_norm(h + TC._swiglu(p["mlp"], h))


def _stack(p, h, cfg: Config):
    for b in p["blocks"]:
        h = _block(b, h, cfg)
    return h


def roles_of(x_tokens):
    """x (H, W) ints in 0..VOCAB-1 -> (F, S) roles: this field's colour here / another colour / outside."""
    x = x_tokens.reshape(-1)
    colour = jnp.arange(F)[:, None]
    return jnp.where(x[None, :] == VOID, ROLE_OUT, jnp.where(x[None, :] == colour, ROLE_MINE, ROLE_OTHER))


def task_tokens(p, code):
    """code (F, d_task) -> (F, w): the shared projection of the per-colour code (permutes with the colours)."""
    return code @ p["task_proj"]


def embed(p, cfg: Config, x_tokens, code=None):
    """(F, S, w): sqrt(w) x (role + position) [+ the per-field task token]. A colour permutation of x
    together with the same permutation of the code's field axis permutes the field axis of this tensor."""
    w = cfg.dec_width
    emb = math.sqrt(w) * (p["role_emb"][roles_of(x_tokens)] + p["pos_emb"][None, :, :])
    if code is not None:
        emb = emb + task_tokens(p, code)[:, None, :]
    return emb


def embed_answer(p, cfg: Config, y_grid):
    """The FPA anchor state: z_H := the embedded (corrupted) OUTPUT grid, void cells included, through the
    input's role table (mine / other / void = outside) + position, no task term (the task enters through
    emb at every inner step). (F, S, w)."""
    return math.sqrt(cfg.dec_width) * (p["role_emb"][roles_of(y_grid)] + p["pos_emb"][None, :, :])


def segment(p, cfg: Config, emb, zH, zL, rng=None):
    """ONE outer segment = H_cycles x (L_cycles + 1) stack passes, exactly the DEC's (dec_cell.segment):
    z_L <- F(z_L + z_H + emb) L_cycles times, then z_H <- F(z_H + z_L); the gradient through the LAST
    H-cycle only; EqR damping / noise per pass."""
    lam, beta = cfg.trm_lambda, cfg.trm_beta
    n_keys = cfg.trm_h_cycles * (cfg.trm_l_cycles + 1)
    keys = (list(jax.random.split(rng, n_keys)) if (rng is not None and beta > 0) else [None] * n_keys)
    stack_c = (lambda p_, h_: _stack(p_, h_, cfg))
    stack = jax.checkpoint(stack_c) if cfg.remat else stack_c

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


def commit_logits(p, zH):
    """C6: the commit logit per cell (S,) = the field-and-cell-shared readout of z_H (S10-invariant)."""
    return jnp.mean(jnp.einsum("fsw,w->sf", zH, p["commit_head"]["w"]), axis=-1) + p["commit_head"]["b"]


def readout(p, cfg: Config, zH, hw_shape):
    """logits (H, W, VOCAB): colour c's logit at a cell = field c's cell token . lm_head; the VOID logit =
    the cell's field MEAN . void_head (invariant); q (2,) from the field-and-cell mean of z_H."""
    lg = jnp.einsum("fsw,w->sf", zH, p["lm_head"])                                  # (S, F) colours 0..9
    void = jnp.mean(zH, axis=0) @ p["void_head"]["w"] + p["void_head"]["b"]          # (S,)
    logits = jnp.concatenate([lg, void[:, None]], axis=1)                            # VOID = class F = 10
    assert logits.shape[1] == VOCAB and VOID == F
    logits = logits.reshape(tuple(hw_shape) + (VOCAB,))
    q = jnp.mean(zH, axis=(0, 1)) @ p["q_head"]["w"] + p["q_head"]["b"]
    return logits, q


def forward_core(p, cfg: Config, fields, *, task_vec=None, z_in=None, rng=None):
    """model.forward_fields contract: fields (VOCAB, H, W, 2) [x one-hot at [..., 0]; the y slot is NOT
    read], task_vec = the per-colour code (F, d_task) or None, carried z_in (2, F, S, w) or None ->
    (logits (H, W, VOCAB), q (2,), z_fine)."""
    assert (not cfg.dec_commit) or cfg.dec_commit_tau >= 1.0, "DEC-ARC: the commit head trains only (no hardening on ARC)"
    H, W = fields.shape[1], fields.shape[2]
    x_tokens = jnp.argmax(fields[..., 0], axis=0)            # exact on one-hot input
    emb = embed(p, cfg, x_tokens, task_vec)
    k_ri, k_seg = (None, None) if rng is None else tuple(jax.random.split(rng))
    if z_in is not None:
        z = z_in
    elif rng is None:
        z = z0_eval(cfg, H * W)          # the deterministic evaluation start (2026-09-15; the buffers for a plain cell)
    else:
        z = z0(cfg, H * W, rng=k_ri)     # training: the RI draw (sigma > 0) or the buffers
    zH, zL = segment(p, cfg, emb, z[0], z[1], rng=k_seg)
    logits, q = readout(p, cfg, zH, (H, W))
    return logits, q, jnp.stack([zH, zL])


def size_from_logits(logits):
    """The crop rule: (h, w) = the bounding box from the origin of the non-VOID argmax cells (at least 1 x 1).
    Returns one-hot size distributions (30,) x 2 in the size heads' convention (index = size - 1)."""
    non_void = jnp.argmax(logits, axis=-1) != VOID                     # (H, W)
    H, W = non_void.shape
    rows = jnp.any(non_void, axis=1); cols = jnp.any(non_void, axis=0)
    h = jnp.max(jnp.where(rows, jnp.arange(H), -1)) + 1
    w = jnp.max(jnp.where(cols, jnp.arange(W), -1)) + 1
    h = jnp.clip(h, 1, 30); w = jnp.clip(w, 1, 30)
    return jax.nn.one_hot(h - 1, 30), jax.nn.one_hot(w - 1, 30)
