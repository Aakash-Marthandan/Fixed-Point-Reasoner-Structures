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
from qhrrn2 import objects as OBJ
from qhrrn2.config import Config
from qhrrn2.grid import CANVAS, NUM_COLORS, VOCAB, VOID

N_FIELDS = VOCAB  # 11
# role index per field: 0 = symmetric color (1..9), 1 = black, 2 = void
ROLE_OF_FIELD = jnp.array([1] + [0] * 9 + [2], dtype=jnp.int32)
SYMMETRIC_FIELDS = tuple(range(1, NUM_COLORS))  # 1..9


N_SIZE_CANDS = 8  # C1-v3: measured size candidates per axis
# C17 stream layout: encoder aggregates over the 3 input segmentations;
# decoder over the same 3 plus the model's own yprev (nonblack4) — the
# shared-per-object variable on the OUTPUT side, per recursion step.
OBJ_ENC_MODES = ("color4", "color8", "nonblack4")
OBJ_DEC_MODES = ("color4", "color8", "nonblack4", "yprev")
N_OBJ_STREAMS = len(OBJ_ENC_MODES) + len(OBJ_DEC_MODES)
D_OBJ = 6  # VIB message width per cluster stream (C14 convention)


class StepOutput(NamedTuple):
    logits: jax.Array        # (H, W, VOCAB)
    size_h: jax.Array        # (30,) OFFSET logits relative to the selected candidate (C1 v3)
    size_w: jax.Array        # (30,)
    flux: jax.Array          # (scales,) nats per RG cut — I_local (C5)
    flux_attn: jax.Array     # (scales,) nats through nonlocal channels — A_nonlocal (C14);
    #                          index s = enc + dec attention at resolution 32/2^s
    rule_q: jax.Array        # (M, K) slot distributions
    h_ir: jax.Array          # (d_ir,)
    size_sel_h: jax.Array    # (N_SIZE_CANDS,) candidate-selection logits (C1 v3)
    size_sel_w: jax.Array    # (N_SIZE_CANDS,)
    flux_obj: jax.Array      # (N_OBJ_STREAMS,) nats through cluster channels (C17)
    z_fine: jax.Array        # (N_FIELDS, H, W, d) pre-readout decoder state (E10 A.2 carry)


# ── Init ───────────────────────────────────────────────────────────────────

def init_params(key, cfg: Config):
    ks = list(jax.random.split(key, 24))
    d, db = cfg.d, cfg.d_b
    r_dim = cfg.M * cfg.d_code

    def lin(k, i, o, scale=None):
        return cell._linear_init(k, i, o, scale)

    eq = ({"eq": {"eta": jnp.zeros(()), "eta_z": jnp.zeros(()),
                  "alpha_z": jnp.asarray(cfg.z_gate_init),  # 0-init = A.2
                  # carry inert at start; z_gate_init>0 warm-opens it (pretrain-9)
                  **({"alpha1": jnp.asarray(1.0986123),   # sigmoid -> .75
                      "alpha2": jnp.asarray(-1.0986123)}  # sigmoid -> .25
                     # FPRM coupled residual scaling (pretrain-13): learnable,
                     # initialized contractive per their Thm-1 recipe; keys
                     # exist only when eq_coupled (old ckpts stay loadable)
                     if cfg.eq_coupled else {})}}
          if cfg.equilibrium else {})
    return {**eq,
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
        # C1 v3 (ledger 2026-08-02): "h"/"w" are OFFSET heads relative to a
        # SELECTED measured candidate; "sel" picks the candidate per axis.
        # Selection over measurements extrapolates by construction — v2 is
        # the special case sel = δ(candidate 0 = input extent).
        "canvas": {"l1": lin(ks[17], cfg.d_ir + r_dim + 60, 64), "h": lin(ks[18], 64, 30), "w": lin(ks[19], 64, 30),
                   "sel": lin(ks[22], 64, 2 * N_SIZE_CANDS)},
        # C16: task-vector entry points. e_ir biases the rule path (S9-safe);
        # e_cb is the per-task Amendment-A color bias — h_ir is S9-invariant by
        # construction, so WITHOUT this path a shared bulk cannot represent
        # task-specific color constants at all. task_vec=None skips both.
        "task_proj": {"e_ir": lin(ks[20], cfg.d_task, cfg.d_ir),
                      "e_cb": lin(ks[21], cfg.d_task, N_FIELDS * d, scale=1e-2)},
    } | _init_obj(ks[23], cfg, d, r_dim, lin)


def _init_obj(key, cfg: Config, d, r_dim, lin):
    """C17 cluster streams. VIB emission (d -> 2*D_OBJ) + small return
    (D_OBJ -> d, scale 1e-2) per stream; gates INIT OPEN (bias +2 — the
    sel-saturation lesson: closed priors never reopen under TTT)."""
    if not cfg.use_obj:
        return {}
    kk = list(jax.random.split(key, 2 * N_OBJ_STREAMS + 2))
    streams = []
    for i in range(N_OBJ_STREAMS):
        streams.append({
            "vib": lin(kk[2 * i], d, 2 * D_OBJ),
            "out": lin(kk[2 * i + 1], D_OBJ, d, scale=1e-2),
        })
    return {"obj": {
        "streams": streams,
        # encoder gates see the task vector (rule code not yet computed);
        # decoder gates see the rule vector r
        "gate_enc": {"w": jax.random.normal(kk[-2], (len(OBJ_ENC_MODES), cfg.d_task)) * 0.1,
                     "b": jnp.full((len(OBJ_ENC_MODES),), 2.0)},
        "gate_dec": {"w": jax.random.normal(kk[-1], (len(OBJ_DEC_MODES), r_dim)) * 0.1,
                     "b": jnp.full((len(OBJ_DEC_MODES),), 2.0)},
    }}


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


def _obj_stream(p_s, z, labels, rng):
    """C17: component-mean -> VIB emission -> small return. The KL is the
    stream's I_object contribution (same variational status as I_s/A_s).
    log_sigma clipped to the house bounds (cell.py C5/C14 convention) —
    unbounded logs overflowed exp() at decoder magnitudes (found at smoke)."""
    agg = OBJ.component_mean(z, labels)
    mu_ls = cell._linear(p_s["vib"], agg)
    mu = mu_ls[..., :D_OBJ]
    log_sigma = jnp.clip(mu_ls[..., D_OBJ:], -6.0, 2.0)
    if rng is not None:
        b = mu + jnp.exp(log_sigma) * jax.random.normal(rng, mu.shape)
    else:
        b = mu
    return cell._linear(p_s["out"], b), cell.stream_kl(mu, log_sigma)


def forward_fields(params, cfg: Config, fields, *, t_norm: float, tau: float,
                   rng=None, task_vec=None, labels_obj=None,
                   z_in=None, rule_override=None) -> StepOutput:
    """One encode→rule→decode pass on continuous occupancy fields.

    fields: (N_FIELDS, CANVAS, CANVAS, 2) float32. Exposed at this level so the
    anti-linearity CI gate can probe the map on arbitrary continuous inputs.
    task_vec: optional (d_task,) program embedding (C16). None skips the task
    paths entirely — the pre-C16 compute graph, exactly.
    labels_obj: C17 label maps {mode: (H, W) int32} incl. "yprev"; required
    when cfg.use_obj (iterate supplies them), ignored otherwise.
    rule_override: optional (M, K) distributions — E4 committed-rule boundary
    condition ([H-6']/CI-10): the rule slots are CLAMPED to the given q's
    (everything downstream — decoder init, gates, canvas head — conditions on
    the committed rule) instead of re-inferred from this input. None = the
    pre-E4 graph, exactly.
    """
    d, db, S = cfg.d, cfg.d_b, cfg.scales
    z = _embed(params, fields)
    if z_in is not None:  # E10 A.2: carried latent enters through a 0-init gate
        z = z + params["eq"]["alpha_z"] * z_in
    cb_t = None
    if task_vec is not None:
        cb_t = cell._linear(params["task_proj"]["e_cb"], task_vec).reshape(N_FIELDS, d)
        z = z + cb_t[:, None, None, :]

    flux_obj = [jnp.zeros(())] * N_OBJ_STREAMS
    use_obj = cfg.use_obj and labels_obj is not None
    if use_obj:  # C17 encoder side: shared per-object variables on the input
        ge = params["obj"]["gate_enc"]
        g_enc = jax.nn.sigmoid(ge["b"] + (ge["w"] @ task_vec if task_vec is not None
                                          else jnp.zeros((len(OBJ_ENC_MODES),))))
        for i, mode in enumerate(OBJ_ENC_MODES):
            sub = None
            if rng is not None:
                rng, sub = jax.random.split(rng)
            upd, kl = _obj_stream(params["obj"]["streams"][i], z, labels_obj[mode], sub)
            z = z + g_enc[i] * upd
            flux_obj[i] = kl

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
    # E4 (CI-10): rule_override clamps q — the committed rule transports from
    # the supports; the query no longer re-infers it.
    E = params["codebook"]
    rule_q, rule_vecs = [], []
    for m in range(cfg.M):
        if rule_override is not None:
            q = rule_override[m]
        else:
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

    if use_obj:  # C17 decoder side: one-decision-per-object where output lands
        gd = params["obj"]["gate_dec"]
        g_dec = jax.nn.sigmoid(gd["b"] + gd["w"] @ r)
        n_e = len(OBJ_ENC_MODES)
        for j, mode in enumerate(OBJ_DEC_MODES):
            sub = None
            if rng is not None:
                rng, sub = jax.random.split(rng)
            upd, kl = _obj_stream(params["obj"]["streams"][n_e + j], zd,
                                  labels_obj[mode], sub)
            zd = zd + g_dec[j] * upd
            flux_obj[n_e + j] = kl

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
    sel = cell._linear(params["canvas"]["sel"], hc)
    return StepOutput(
        logits=logits,
        size_h=cell._linear(params["canvas"]["h"], hc),
        size_w=cell._linear(params["canvas"]["w"], hc),
        flux=jnp.stack(flux),
        flux_attn=jnp.stack(flux_nl),
        rule_q=jnp.stack(rule_q),
        h_ir=h_ir,
        size_sel_h=sel[:N_SIZE_CANDS],
        size_sel_w=sel[N_SIZE_CANDS:],
        flux_obj=jnp.stack(flux_obj),
        z_fine=zd,
    )


def size_candidates(x_canvas):
    """C1 v3: the (2, N_SIZE_CANDS) measured size candidates, all observable
    from the INPUT canvas at predict time (no GT anywhere). Per axis:
    [own extent, other extent, 2x, 3x, ceil(/2), ceil(/3),
     occupied lines along the axis, top-color cell count]."""
    mask = x_canvas != VOID
    h_in = jnp.sum(jnp.any(mask, axis=1)).astype(jnp.int32)
    w_in = jnp.sum(jnp.any(mask, axis=0)).astype(jnp.int32)
    colored = mask & (x_canvas != 0)
    occ_r = jnp.sum(jnp.any(colored, axis=1)).astype(jnp.int32)
    occ_c = jnp.sum(jnp.any(colored, axis=0)).astype(jnp.int32)
    counts = jnp.stack([jnp.sum(x_canvas == c) for c in range(1, NUM_COLORS)])
    top1 = jnp.max(counts).astype(jnp.int32)

    def axis(a, b, occ):
        return jnp.stack([a, b, 2 * a, 3 * a, (a + 1) // 2, (a + 2) // 3, occ, top1])
    cands = jnp.stack([axis(h_in, w_in, occ_r), axis(w_in, h_in, occ_c)])
    return jnp.clip(cands, 1, 30)


def size_mixture_probs(sel_logits, off_logits, cands):
    """p(out_size = s), s in 1..30: sum_c q_c * p_off(s - cand_c + 15).

    sel_logits (N_SIZE_CANDS,), off_logits (30,), cands (N_SIZE_CANDS,) ->
    (30,) probabilities over sizes 1..30 (index s-1). Pure function shared by
    the objective, the decoder, and the C1-v3 expressibility tests."""
    q = jax.nn.softmax(sel_logits)
    poff = jax.nn.softmax(off_logits)
    sizes = jnp.arange(1, 31)                       # (30,)
    idx = sizes[None, :] - cands[:, None] + 15      # (C, 30) offset index
    valid = (idx >= 0) & (idx < 30)
    contrib = jnp.where(valid, jnp.take(poff, jnp.clip(idx, 0, 29)), 0.0)
    return q @ contrib                              # (30,)


def build_fields(x_canvas, yprev_canvas):
    """Two int canvases -> (N_FIELDS, H, W, 2) occupancy fields."""
    fx = jax.nn.one_hot(x_canvas, VOCAB).transpose(2, 0, 1)
    fy = jax.nn.one_hot(yprev_canvas, VOCAB).transpose(2, 0, 1)
    return jnp.stack([fx, fy], axis=-1)


def build_fields_soft(x_canvas, y_probs):
    """E10: int input canvas + CONTINUOUS carried answer (N_FIELDS, H, W)
    probability canvas -> fields. The y slot never sees an argmax."""
    fx = jax.nn.one_hot(x_canvas, VOCAB).transpose(2, 0, 1)
    return jnp.stack([fx, y_probs], axis=-1)


def iterate_eq(params, cfg: Config, x_canvas, *, tau: float, rng=None,
               task_vec=None, t_total=None, y0_probs=None):
    """E10 equilibrium loop ([H-2'], ledger 2026-08-09): continuous carried
    answer register with a damped update y <- y + eta*(softmax(logits) - y);
    eta = sigmoid(params['eq']['eta']) (FPRM-style learnable step). Steps
    beyond cfg.T repeat the final map (t_norm frozen at 1). Returns
    (outs, residuals, y_final); residuals are mean |Delta y| per step —
    the inference halt (res < cfg.res_tau) is applied by callers/probes."""
    T = t_total if t_total is not None else cfg.T
    void_can = jnp.full((CANVAS, CANVAS), VOID, dtype=jnp.int32)
    y = (jax.nn.one_hot(void_can, VOCAB).transpose(2, 0, 1)
         if y0_probs is None else y0_probs)
    eta = cfg.eta_floor + (1.0 - cfg.eta_floor) * jax.nn.sigmoid(params["eq"]["eta"])
    eta_z = jax.nn.sigmoid(params["eq"]["eta_z"])
    if cfg.eq_coupled:  # pretrain-13: y <- a1*y + a2*p (FPRM two-scalar form;
        #               the damped update is the a1=1-eta, a2=eta special case)
        a1 = jax.nn.sigmoid(params["eq"]["alpha1"])
        a2 = jax.nn.sigmoid(params["eq"]["alpha2"])
    outs, residuals = [], []
    z_c = None
    for t in range(T):
        t_norm = min(t, cfg.T - 1) / max(cfg.T - 1, 1)
        step_rng = None
        if rng is not None:
            rng, step_rng = jax.random.split(rng)
        if cfg.remat:
            # Rematerialize per eq step (P11-EXT 2026-08-11: the original
            # remat covered only the projective iterate — d48/B64 OOM'd at
            # an unchanged 17.72G because this loop never checkpointed).
            # z sentinel keeps the closure signature fixed across t=0/t>0.
            def _fwd(p, yp, r, tv, zc, _t=t_norm):
                return forward_fields(p, cfg, build_fields_soft(x_canvas, yp),
                                      t_norm=_t, tau=tau, rng=r, task_vec=tv,
                                      z_in=None if zc.ndim == 1 else zc)
            out = jax.checkpoint(_fwd)(
                params, y, step_rng, task_vec,
                z_c if z_c is not None else jnp.zeros(1))
        else:
            out = forward_fields(params, cfg, build_fields_soft(x_canvas, y),
                                 t_norm=t_norm, tau=tau, rng=step_rng,
                                 task_vec=task_vec, z_in=z_c)
        z_c = (out.z_fine if z_c is None
               else z_c + eta_z * (out.z_fine - z_c))
        p = jax.nn.softmax(out.logits, axis=-1).transpose(2, 0, 1)
        y_new = (a1 * y + a2 * p) if cfg.eq_coupled else (y + eta * (p - y))
        if cfg.ni_sigma > 0 and rng is not None:
            # pretrain-13 NI (EqR per-step training noise, raw std ni_sigma):
            # simplex-tangent draw + clip/renorm ([R-4]: state-space, never
            # logits). rng is threaded only by training callers, so probes
            # and deployment (rng=None) never see noise. The extra split
            # exists only on this branch — ni_sigma=0 leaves the registered
            # rng stream untouched.
            rng, k_ni = jax.random.split(rng)
            xi = jax.random.normal(k_ni, y_new.shape)
            xi = xi - xi.mean(axis=0, keepdims=True)
            y_new = jnp.clip(y_new + cfg.ni_sigma * xi, 0.0, None)
            y_new = y_new / jnp.maximum(
                jnp.sum(y_new, axis=0, keepdims=True), 1e-6)
        residuals.append(jnp.mean(jnp.abs(y_new - y)))
        y = y_new
        outs.append(out)
    return outs, residuals, y


def iterate(params, cfg: Config, x_canvas, *, tau: float, rng=None,
            task_vec=None, labels_x=None, yprev_init=None) -> list[StepOutput]:
    """T recursion passes (C9). Feedback is the argmax canvas (detached by
    construction); deep supervision trains every pass.

    yprev_init (ledger 2026-08-08, [H-23] basin training): optional initial
    feedback canvas — corrupted targets / self-rollout states enter here so
    the map is trained to restore/progress from them. None = the deployed
    all-VOID start, bit-identical (tests/test_e8.py::test_yprev_init_inert).

    cfg.equilibrium (E10): dispatch to the continuous-state loop; the outs
    contract (list of StepOutput, deep supervision per step) is preserved."""
    if cfg.equilibrium:
        y0 = (None if yprev_init is None else
              jax.nn.one_hot(jnp.asarray(yprev_init, dtype=jnp.int32),
                             VOCAB).transpose(2, 0, 1))
        outs, _, _ = iterate_eq(params, cfg, x_canvas, tau=tau, rng=rng,
                                task_vec=task_vec, y0_probs=y0)
        return outs
    yprev = (jnp.full((CANVAS, CANVAS), VOID, dtype=jnp.int32)
             if yprev_init is None else jnp.asarray(yprev_init, dtype=jnp.int32))
    labs_x = None
    if cfg.use_obj:
        if labels_x is not None:
            # Precomputed input segmentations (speed pipeline, 2026-08-02):
            # (3, H, W) stacked in OBJ_ENC_MODES order. Rolled labels are valid
            # partitions — segment ids need uniqueness, not canonicality.
            labs_x = {m: labels_x[i] for i, m in enumerate(OBJ_ENC_MODES)}
        else:  # eval/TTT path: segment in-graph, iteration-invariant
            labs_x = {m: OBJ.connected_components(x_canvas, m) for m in OBJ_ENC_MODES}
    # t=0 feedback is all-VOID: every cell a singleton — aggregation is the
    # identity by construction, so skip the (expensive) in-graph CC there.
    identity_labels = jnp.arange(CANVAS * CANVAS, dtype=jnp.int32).reshape(CANVAS, CANVAS)
    outs = []
    for t in range(cfg.T):
        t_norm = t / max(cfg.T - 1, 1)
        step_rng = None
        if rng is not None:
            rng, step_rng = jax.random.split(rng)
        labels_obj = None
        if cfg.use_obj:  # yprev re-segmented each step: cohere what was painted
            lab_y = identity_labels if t == 0 else OBJ.connected_components(yprev, "nonblack4")
            labels_obj = labs_x | {"yprev": lab_y}
        if cfg.remat:
            # Rematerialize each recursion step's activations on the backward
            # pass: memory drops ~T-fold for ~30% extra compute. Positional
            # closure per t (t_norm/tau are Python constants of this step).
            def _fwd(p, f, r, tv, lo, _t=t_norm):
                return forward_fields(p, cfg, f, t_norm=_t, tau=tau, rng=r,
                                      task_vec=tv, labels_obj=lo)
            out = jax.checkpoint(_fwd)(params, build_fields(x_canvas, yprev),
                                       step_rng, task_vec, labels_obj)
        else:
            out = forward_fields(params, cfg, build_fields(x_canvas, yprev),
                                 t_norm=t_norm, tau=tau, rng=step_rng,
                                 task_vec=task_vec, labels_obj=labels_obj)
        outs.append(out)
        yprev = jnp.argmax(out.logits, axis=-1)
    return outs
