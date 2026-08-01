# Ledger: C2 (S9 field axis + roles), C3 (color_bias — Amendment A), C5 (streams+gates),
#         C6+C14 (priced attention at all scales — Amendment D, flux_attn ledger),
#         C8 (rule codebook, tau-annealed), C9 (recursion, deep supervision feeds),
#         C1 (canvas head). Equivariance guarded by tests/test_model.py::test_s9_equivariance.
"""QHRRN-2 model: equivariant recursive coarse-graining with priced streams.

Field order convention (axis 0 of the state): index c is ARC color c for
c in 0..9, index 10 is VOID. Role classes: black (0) and VOID (10) are
distinguished; colors 1..9 are the S9-symmetric set.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from qhrrn2 import cell
from qhrrn2.config import Config
from qhrrn2.grid import CANVAS, NUM_COLORS, VOCAB, VOID

N_FIELDS = VOCAB  # 11
# role index per field: 0 = symmetric color (1..9), 1 = black, 2 = void
ROLE_OF_FIELD = jnp.array([1] + [0] * 9 + [2], dtype=jnp.int32)
SYMMETRIC_FIELDS = tuple(range(1, NUM_COLORS))  # 1..9


class StepOutput(NamedTuple):
    logits: jax.Array        # (H, W, VOCAB)
    size_h: jax.Array        # (30,) logits for H_out-1
    size_w: jax.Array        # (30,)
    flux: jax.Array          # (scales,) nats per RG cut — I_local (C5)
    flux_attn: jax.Array     # (scales,) nats through nonlocal channels — A_nonlocal (C14);
    #                          index s = enc + dec attention at resolution 32/2^s
    rule_q: jax.Array        # (M, K) slot distributions
    h_ir: jax.Array          # (d_ir,)


# ── Init ───────────────────────────────────────────────────────────────────

def init_params(key, cfg: Config):
    ks = list(jax.random.split(key, 22))
    d, db = cfg.d, cfg.d_b
    r_dim = cfg.M * cfg.d_code

    def lin(k, i, o, scale=None):
        return cell._linear_init(k, i, o, scale)

    return {
        "embed": {  # shared 3x3 conv over each field's 2 channels (x, y_prev)
            "w": jax.random.normal(ks[0], (d, 2, 3, 3)) * 0.3,
            "b": jnp.zeros((d,)),
        },
        "role_emb": jax.random.normal(ks[1], (3, d)) * 0.3,
        "color_bias": jnp.zeros((N_FIELDS, d)),  # Amendment A: evidence breaks S9 (C3)
        "enc": {
            "mixer": cell.init_mixer(ks[2], d),
            "pool": cell.init_pool_split(ks[3], d, db),
            "attn": cell.init_attention(ks[4], d, cfg.d_a),
            "film": cell.init_film(ks[5], d),
        },
        "ir_proj": lin(ks[6], 3 * d, cfg.d_ir),
        "codebook": jax.random.normal(ks[7], (cfg.K, cfg.d_code)) / jnp.sqrt(cfg.d_code),
        "rule_query": jax.random.normal(ks[8], (cfg.M, cfg.d_ir, cfg.d_code)) / jnp.sqrt(cfg.d_ir),
        "dec_init": lin(ks[9], cfg.d_ir + r_dim, d),
        "dec": {
            "mixer": cell.init_mixer(ks[10], d),
            "attn": cell.init_attention(ks[11], d, cfg.d_a),
            "film": cell.init_film(ks[12], d),
            "inject": cell.init_inject(ks[13], db, d),
        },
        "gate": {"l1": lin(ks[14], r_dim + cfg.scales, 16), "l2": lin(ks[15], 16, db, scale=1e-2)},
        "readout": {"w": jax.random.normal(ks[16], (d,)) / jnp.sqrt(d), "role_b": jnp.zeros((3,))},
        # C1 v2 (ledger 2026-07-27): input-extent one-hots as head FEATURES,
        # and the heads classify size OFFSETS relative to that extent (see
        # objective._step_loss / train.predict) — extent is observable at
        # predict time, no GT leak; the relative frame is what extrapolates.
        "canvas": {"l1": lin(ks[17], cfg.d_ir + r_dim + 60, 64), "h": lin(ks[18], 64, 30), "w": lin(ks[19], 64, 30)},
        # C16: task-vector entry points. e_ir biases the rule path (S9-safe);
        # e_cb is the per-task Amendment-A color bias — h_ir is S9-invariant by
        # construction, so WITHOUT this path a shared bulk cannot represent
        # task-specific color constants at all. task_vec=None skips both.
        "task_proj": {"e_ir": lin(ks[20], cfg.d_task, cfg.d_ir),
                      "e_cb": lin(ks[21], cfg.d_task, N_FIELDS * d, scale=1e-2)},
    }


def count_params(tree) -> int:
    return sum(x.size for x in jax.tree.leaves(tree))


# ── Forward ────────────────────────────────────────────────────────────────

def _embed(params, fields):
    """fields: (N_FIELDS, H, W, 2) -> (N_FIELDS, H, W, d)."""
    x = fields.transpose(0, 3, 1, 2)  # (C, 2, H, W): fields as batch, shared conv
    z = jax.lax.conv_general_dilated(
        x, params["embed"]["w"], window_strides=(1, 1), padding="SAME",
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
    ).transpose(0, 2, 3, 1) + params["embed"]["b"]
    z = z + params["role_emb"][ROLE_OF_FIELD][:, None, None, :]
    z = z + params["color_bias"][:, None, None, :]
    return z


def forward_fields(params, cfg: Config, fields, *, t_norm: float, tau: float,
                   rng=None, task_vec=None) -> StepOutput:
    """One encode→rule→decode pass on continuous occupancy fields.

    fields: (N_FIELDS, CANVAS, CANVAS, 2) float32. Exposed at this level so the
    anti-linearity CI gate can probe the map on arbitrary continuous inputs.
    task_vec: optional (d_task,) program embedding (C16). None skips the task
    paths entirely — the pre-C16 compute graph, exactly.
    """
    d, db, S = cfg.d, cfg.d_b, cfg.scales
    z = _embed(params, fields)
    cb_t = None
    if task_vec is not None:
        cb_t = cell._linear(params["task_proj"]["e_cb"], task_vec).reshape(N_FIELDS, d)
        z = z + cb_t[:, None, None, :]

    def split(r):
        return jax.random.split(r) if r is not None else (None, None)

    streams, flux = [], []
    flux_nl = [jnp.zeros(())] * S           # A_s ledger (C14): enc + dec per scale
    for s in range(S):
        s_norm = s / max(S - 1, 1)
        gammas, betas = cell.film_params(params["enc"]["film"], s_norm, t_norm, d)
        z = cell.film(cell.mixer(params["enc"]["mixer"], z), gammas[0], betas[0])
        if z.shape[1] <= cfg.attn_max_hw:
            rng, sub = split(rng)
            z, a_s = cell.attention(params["enc"]["attn"], z, rng=sub)
            flux_nl[s] = flux_nl[s] + a_s
        kept, mu, log_sigma = cell.pool_split(params["enc"]["pool"], z, db)
        if rng is not None:
            rng, sub = jax.random.split(rng)
            b = mu + jnp.exp(log_sigma) * jax.random.normal(sub, mu.shape)
        else:
            b = mu
        streams.append(b)
        flux.append(cell.stream_kl(mu, log_sigma))
        z = cell.film(kept, gammas[1], betas[1])

    # IR summary: equivariant pooling — symmetric-set mean, black, void.
    top = z[:, 0, 0, :]                                   # (N_FIELDS, d)
    sym = jnp.mean(top[jnp.array(SYMMETRIC_FIELDS)], axis=0)
    h_ir = jax.nn.gelu(cell._linear(params["ir_proj"],
                                    jnp.concatenate([sym, top[0], top[VOID]])))
    if task_vec is not None:  # C16 entry (i): rule-path bias, color-blind
        h_ir = h_ir + cell._linear(params["task_proj"]["e_ir"], task_vec)

    # Rule slots (C8): tau-annealed categorical attention over the codebook.
    E = params["codebook"]
    rule_q, rule_vecs = [], []
    for m in range(cfg.M):
        logits_k = (h_ir @ params["rule_query"][m]) @ E.T / tau
        q = jax.nn.softmax(logits_k)
        rule_q.append(q)
        rule_vecs.append(q @ E)
    r = jnp.concatenate(rule_vecs)                        # (M * d_code,)

    # Decoder: rule-conditioned init, inject-at-coarse → upsample → mix.
    zd = jax.nn.gelu(cell._linear(params["dec_init"], jnp.concatenate([h_ir, r])))
    zd = jnp.broadcast_to(zd, (N_FIELDS, 1, 1, d))
    zd = zd + params["role_emb"][ROLE_OF_FIELD][:, None, None, :]
    zd = zd + params["color_bias"][:, None, None, :]
    if cb_t is not None:  # C16 entry (ii), decoder side
        zd = zd + cb_t[:, None, None, :]
    for s in reversed(range(S)):
        s_norm = s / max(S - 1, 1)
        s_onehot = jax.nn.one_hot(s, S)
        g = jax.nn.sigmoid(cell._linear(params["gate"]["l2"], jax.nn.gelu(
            cell._linear(params["gate"]["l1"], jnp.concatenate([r, s_onehot])))))
        zd = cell.inject(params["dec"]["inject"], zd, streams[s], g)
        zd = cell.upsample(zd)
        gammas, betas = cell.film_params(params["dec"]["film"], s_norm, t_norm, d)
        zd = cell.film(cell.mixer(params["dec"]["mixer"], zd), gammas[0], betas[0])
        if zd.shape[1] <= cfg.attn_max_hw:
            rng, sub = split(rng)
            zd, a_s = cell.attention(params["dec"]["attn"], zd, rng=sub)
            flux_nl[s] = flux_nl[s] + a_s
        zd = cell.film(zd, gammas[1], betas[1])

    # Equivariant readout: shared vector + role bias -> (H, W, N_FIELDS) logits.
    logits = jnp.einsum("chwd,d->chw", zd, params["readout"]["w"])
    logits = logits + params["readout"]["role_b"][ROLE_OF_FIELD][:, None, None]
    logits = logits.transpose(1, 2, 0)

    # Input extent from the x occupancy (1 - VOID channel): palette-invariant,
    # observable at predict time — no ground-truth size anywhere (C1).
    x_occ = 1.0 - fields[VOID, :, :, 0]
    h_in = jnp.clip(jnp.round(jnp.sum(jnp.max(x_occ, axis=1))).astype(jnp.int32) - 1, 0, 29)
    w_in = jnp.clip(jnp.round(jnp.sum(jnp.max(x_occ, axis=0))).astype(jnp.int32) - 1, 0, 29)
    extent = jnp.concatenate([jax.nn.one_hot(h_in, 30), jax.nn.one_hot(w_in, 30)])

    hc = jax.nn.gelu(cell._linear(params["canvas"]["l1"],
                                  jnp.concatenate([h_ir, r, extent])))
    return StepOutput(
        logits=logits,
        size_h=cell._linear(params["canvas"]["h"], hc),
        size_w=cell._linear(params["canvas"]["w"], hc),
        flux=jnp.stack(flux),
        flux_attn=jnp.stack(flux_nl),
        rule_q=jnp.stack(rule_q),
        h_ir=h_ir,
    )


def build_fields(x_canvas, yprev_canvas):
    """Two int canvases -> (N_FIELDS, H, W, 2) occupancy fields."""
    fx = jax.nn.one_hot(x_canvas, VOCAB).transpose(2, 0, 1)
    fy = jax.nn.one_hot(yprev_canvas, VOCAB).transpose(2, 0, 1)
    return jnp.stack([fx, fy], axis=-1)


def iterate(params, cfg: Config, x_canvas, *, tau: float, rng=None,
            task_vec=None) -> list[StepOutput]:
    """T recursion passes (C9). Feedback is the argmax canvas (detached by
    construction); deep supervision trains every pass."""
    yprev = jnp.full((CANVAS, CANVAS), VOID, dtype=jnp.int32)
    outs = []
    for t in range(cfg.T):
        t_norm = t / max(cfg.T - 1, 1)
        step_rng = None
        if rng is not None:
            rng, step_rng = jax.random.split(rng)
        out = forward_fields(params, cfg, build_fields(x_canvas, yprev),
                             t_norm=t_norm, tau=tau, rng=step_rng,
                             task_vec=task_vec)
        outs.append(out)
        yprev = jnp.argmax(out.logits, axis=-1)
    return outs
