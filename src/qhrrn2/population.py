# Ledger: C11 (population TTT, spec §; promoted to prerequisite by eval-4's
# per-view finding and the assembly doctrine 2026-08-06). Members are
# (D4 view k, seed s): each fits e_t ONLY against the frozen bulk on ITS
# transformed copy of the supports — per-view supervision is self-consistent
# (the conjugated task), so the fit-time validity law holds. Prediction:
# qualifying members predict on their view, INVERT, and vote.
#
# 2026-08-07 (C11 ⊕ [H-15] ⊕ [H-18]): cross-bulk agreement-regularized
# populations. Members are (bulk g, view k, seed s); groups vmap within a
# bulk (heterogeneous param trees can't share one vmap). The query inputs
# ride in every member's batch as extra rows whose y-canvases are CONSENSUS
# pseudo-labels — the cross-member majority prediction, refreshed every
# agree_every steps and weighted agree_lambda after agree_warmup. Support
# rows stay ground-truth-anchored (weight 1); the LoO pair stays held out;
# queries contribute only their INPUTS (transduction validity law).
"""Vectorized population TTT: M members = views × seeds, one vmapped fit."""
from __future__ import annotations

import functools

import numpy as np
import jax
import jax.numpy as jnp
import optax

from qhrrn2 import grid as G
from qhrrn2 import train as T
from qhrrn2.config import Config
from qhrrn2.objective import batch_loss


def member_specs(n_views: int, n_seeds: int):
    return [(k, s) for k in range(n_views) for s in range(n_seeds)]


def build_member_batches(episodes, n_views: int, n_seeds: int, seed: int):
    """Stack per-member transformed support batches + val pairs.

    Returns (x_mb, y_mb) as (M, B, 32, 32) plus per-member
    [(val_x, val_y, Transform)] in member_specs order."""
    support = list(episodes[0].support)
    train_pairs, (val_x, val_y) = support[:-1], support[-1]
    xs, ys, val = [], [], []
    for k, s in member_specs(n_views, n_seeds):
        tr = G.Transform(k=k)
        tp = [(tr.apply(x), tr.apply(y)) for x, y in train_pairs]
        x_b, y_b = T.pairs_to_batch(tp, transforms=None, seed=seed + 97 * s + k)
        xs.append(x_b)
        ys.append(y_b)
        val.append((tr.apply(val_x), tr.apply(val_y), tr))
    return jnp.stack(xs), jnp.stack(ys), val


def fit_population(ckpt_state, cfg: Config, episodes, *, n_views: int = 8,
                   n_seeds: int = 2, steps: int = 600, val_every: int = 50,
                   lr: float = 1e-2, wd: float = 1e-4, tau: float = 1.0,
                   seed: int = 0):
    """Fit M = n_views*n_seeds e_t members against the frozen bulk.

    Returns dict with tv snapshots (n_snap, M, d_task), member val infos, and
    the model reference — scoring/voting happens host-side in score_population.
    """
    model = jax.tree.map(jnp.asarray, ckpt_state["model"])
    tv0 = jnp.asarray(np.asarray(ckpt_state["table"]).mean(0))
    M = n_views * n_seeds
    tv = jnp.broadcast_to(tv0, (M,) + tv0.shape) + 0.0

    x_mb, y_mb, val = build_member_batches(episodes, n_views, n_seeds, seed)

    # Module-level cached step (the _step_and_opt house pattern): a jitted
    # closure defined per task RETAINS its graph — after ~1 task of distinct
    # shapes the accumulation segfaulted libtpu (2026-08-06). Model rides as
    # an ARGUMENT so the cache key is shapes/hparams only.
    step, opt = _pop_step(cfg, tau, lr, wd, M)
    opt_state = jax.vmap(opt.init)(tv)

    rng = jax.random.PRNGKey(seed)
    snaps = []
    for i in range(steps):
        tv, opt_state, losses, rng = step(model, tv, opt_state, rng, x_mb, y_mb)
        if (i + 1) % val_every == 0 or i + 1 == steps:
            snaps.append((i + 1, np.asarray(tv)))
    return {"model": model, "snaps": snaps, "val": val, "cfg": cfg, "tau": tau,
            "n_views": n_views, "n_seeds": n_seeds}


@functools.lru_cache(maxsize=8)
def _pop_step(cfg: Config, tau: float, lr: float, wd: float, M: int):
    opt = optax.adamw(lr, weight_decay=wd)

    @jax.jit
    def step(model, tv, opt_state, rng, x_mb, y_mb):
        keys = jax.random.split(rng, M + 1)

        def one(tv_m, os_m, x_b, y_b, key):
            def loss_fn(v):
                tvs = jnp.broadcast_to(v, (x_b.shape[0],) + v.shape)
                loss, _ = batch_loss(model, cfg, x_b, y_b, tau=tau, rng=key,
                                     task_vecs=tvs)
                return loss
            loss, g = jax.value_and_grad(loss_fn)(tv_m)
            upd, os2 = opt.update(g, os_m, tv_m)
            return optax.apply_updates(tv_m, upd), os2, loss
        tv2, os2, losses = jax.vmap(one)(tv, opt_state, x_mb, y_mb, keys[:M])
        return tv2, os2, losses, keys[M]
    return step, opt


# ── Cross-bulk agreement-regularized populations (2026-08-07) ─────────────────

def build_member_batches_x(episodes, n_views: int, n_seeds: int, seed: int):
    """build_member_batches + per-member query rows appended after the
    support rows. Returns (x_mb, y_mb, val, n_support_rows, queries) where
    queries[m][q] is member m's Transform-applied query canvas row index
    n_support_rows + q. y rows for queries start all-VOID (weight 0 until
    the first consensus refresh)."""
    x_mb, y_mb, val = build_member_batches(episodes, n_views, n_seeds, seed)
    Q = len(episodes)
    B = x_mb.shape[1]
    xq, yq = [], []
    for m, (k, s) in enumerate(member_specs(n_views, n_seeds)):
        tr = val[m][2]
        rows = [G.place(tr.apply(np.asarray(ep.query_x))) for ep in episodes]
        xq.append(np.stack(rows))
        yq.append(np.full((Q, G.CANVAS, G.CANVAS), G.VOID, dtype=np.int32))
    x_mb = jnp.concatenate([x_mb, jnp.asarray(np.stack(xq), dtype=jnp.int32)], axis=1)
    y_mb = jnp.concatenate([y_mb, jnp.asarray(np.stack(yq), dtype=jnp.int32)], axis=1)
    return x_mb, y_mb, val, B, Q


@functools.lru_cache(maxsize=8)
def _pop_step_w(cfg: Config, tau: float, lr: float, wd: float, M: int):
    """_pop_step with per-row loss weights (agreement rows carry λ)."""
    opt = optax.adamw(lr, weight_decay=wd)

    @jax.jit
    def step(model, tv, opt_state, rng, x_mb, y_mb, w_rows):
        keys = jax.random.split(rng, M + 1)

        def one(tv_m, os_m, x_b, y_b, key):
            def loss_fn(v):
                tvs = jnp.broadcast_to(v, (x_b.shape[0],) + v.shape)
                loss, _ = batch_loss(model, cfg, x_b, y_b, tau=tau, rng=key,
                                     task_vecs=tvs, weights=w_rows)
                return loss
            loss, g = jax.value_and_grad(loss_fn)(tv_m)
            upd, os2 = opt.update(g, os_m, tv_m)
            return optax.apply_updates(tv_m, upd), os2, loss
        tv2, os2, losses = jax.vmap(one)(tv, opt_state, x_mb, y_mb, keys[:M])
        return tv2, os2, losses, keys[M]
    return step, opt


@functools.lru_cache(maxsize=8)
def _pop_forward(cfg: Config, tau: float):
    """Vmapped member forward on query rows: (M,Q,32,32) → argmax canvases +
    size mixture probs. Used for consensus refresh and final member preds."""
    from qhrrn2 import model as Mod
    from qhrrn2.model import iterate

    @jax.jit
    def fwd(model, tv, xq):
        def one_member(tv_m, xq_m):
            def one_query(x_can):
                outs = iterate(model, cfg, x_can, tau=tau, rng=None,
                               task_vec=tv_m)
                last = outs[-1]
                cands = Mod.size_candidates(x_can)
                p_h = Mod.size_mixture_probs(last.size_sel_h, last.size_h, cands[0])
                p_w = Mod.size_mixture_probs(last.size_sel_w, last.size_w, cands[1])
                return jnp.argmax(last.logits, axis=-1), p_h, p_w
            return jax.vmap(one_query)(xq_m)
        return jax.vmap(one_member)(tv, xq)
    return fwd


def _member_raw_preds(groups, tvs_per_group):
    """All members' current query predictions, inverted to canonical frame.

    groups: list of dicts with model/cfg/tau/x_query (M_g,Q,32,32)/val.
    Returns preds[g][m][q] = np.int8 raw grid."""
    out = []
    for g, tv_g in zip(groups, tvs_per_group):
        canv, p_h, p_w = _pop_forward(g["cfg"], g["tau"])(g["model"], tv_g,
                                                          g["x_query"])
        canv, p_h, p_w = np.asarray(canv), np.asarray(p_h), np.asarray(p_w)
        g_preds = []
        for m in range(canv.shape[0]):
            tr = g["val"][m][2]
            row = []
            for q in range(canv.shape[1]):
                h = int(p_h[m, q].argmax()) + 1
                w = int(p_w[m, q].argmax()) + 1
                grid = canv[m, q, :h, :w]
                grid = np.where(grid == G.VOID, 0, grid).astype(np.int8)
                row.append(tr.invert_output(grid))
            g_preds.append(row)
        out.append(g_preds)
    return out


def consensus_vote(preds_flat):
    """Majority-shape + cellwise-majority over a list of raw grids."""
    shapes = [p.shape for p in preds_flat]
    maj = max(sorted(set(shapes)), key=shapes.count)
    stack = np.stack([p for p in preds_flat if p.shape == maj])
    voted = np.apply_along_axis(
        lambda v: np.bincount(v, minlength=11).argmax(), 0, stack)
    return voted.astype(np.int8)


# ── [H-27] PoE candidate scoring (first build, phase-plan 2026-08-11) ──────
# Product-of-experts over the population: score each CANONICAL candidate grid
# by the summed log-likelihood of its cells (+ its shape) under every
# member's final-step distributions, each member seeing the candidate in ITS
# OWN view frame (tr.apply before placement). Additive instrument: callers
# record poe_att1/att2 ALONGSIDE the cellwise vote; no deployed selection
# changes until the rg-gate comparison adjudicates (steering law).

@functools.lru_cache(maxsize=8)
def _pop_forward_logp(cfg: Config, tau: float):
    from qhrrn2.model import iterate
    from qhrrn2 import model as Mod

    @jax.jit
    def fwd(model, tv, xq):
        def one_member(tv_m, xq_m):
            def one_query(x_can):
                outs = iterate(model, cfg, x_can, tau=tau, rng=None,
                               task_vec=tv_m)
                last = outs[-1]
                cands = Mod.size_candidates(x_can)
                p_h = Mod.size_mixture_probs(last.size_sel_h, last.size_h,
                                             cands[0])
                p_w = Mod.size_mixture_probs(last.size_sel_w, last.size_w,
                                             cands[1])
                return (jax.nn.log_softmax(last.logits, axis=-1),
                        jnp.log(p_h + 1e-9), jnp.log(p_w + 1e-9))
            return jax.vmap(one_query)(xq_m)
        return jax.vmap(one_member)(tv, xq)
    return fwd


def poe_rank(groups, tvs_per_group, candidates_per_q):
    """candidates_per_q: [q] -> list of canonical np.int8 grids.
    Returns [q] -> list of (total_logp, cand_idx), best first."""
    logps = []  # per group: (M, Q, 32, 32, V), (M, Q, 30), (M, Q, 30)
    for g, tv_g in zip(groups, tvs_per_group):
        lp, lph, lpw = _pop_forward_logp(g["cfg"], g["tau"])(
            g["model"], tv_g, g["x_query"])
        logps.append((np.asarray(lp), np.asarray(lph), np.asarray(lpw)))
    n_q = len(candidates_per_q)
    ranked = []
    for q in range(n_q):
        scores = []
        for ci, cand in enumerate(candidates_per_q[q]):
            total = 0.0
            for g, (lp, lph, lpw) in zip(groups, logps):
                for m in range(lp.shape[0]):
                    tr = g["val"][m][2]
                    tc = tr.apply(np.asarray(cand, dtype=np.int8))
                    h, w = tc.shape
                    if h > G.CANVAS or w > G.CANVAS or h < 1 or w < 1:
                        total += -1e9
                        continue
                    placed = G.place(tc)
                    cell_lp = np.take_along_axis(
                        lp[m, q, :h, :w], placed[:h, :w, None].astype(np.int64),
                        axis=2)[..., 0].sum()
                    total += float(cell_lp) + float(lph[m, q, h - 1]) \
                        + float(lpw[m, q, w - 1])
            scores.append((total, ci))
        ranked.append(sorted(scores, reverse=True))
    return ranked


def fit_population_cross(bulks, episodes, *, n_views: int = 8, n_seeds: int = 1,
                         steps: int = 600, val_every: int = 50,
                         agree_lambda: float = 0.0, agree_every: int = 25,
                         agree_warmup: int = 150, lr: float = 1e-2,
                         wd: float = 1e-4, seed: int = 0):
    """Fit per-bulk member groups jointly with cross-bulk consensus agreement.

    bulks: list of {"name", "state" (ckpt state), "cfg", "tau"}. agree_lambda=0
    is the pure cross-bulk-population ablation (arm X): query rows ride at
    weight 0 (loss-inert) and no consensus is ever computed."""
    groups = []
    for b in bulks:
        model = jax.tree.map(jnp.asarray, b["state"]["model"])
        tv0 = jnp.asarray(np.asarray(b["state"]["table"]).mean(0))
        M = n_views * n_seeds
        tv = jnp.broadcast_to(tv0, (M,) + tv0.shape) + 0.0
        x_mb, y_mb, val, B, Q = build_member_batches_x(
            episodes, n_views, n_seeds, seed)
        step, opt = _pop_step_w(b["cfg"], b["tau"], lr, wd, M)
        groups.append({
            "name": b["name"], "model": model, "cfg": b["cfg"],
            "tau": b["tau"], "tv": tv, "opt_state": jax.vmap(opt.init)(tv),
            "x": x_mb, "y": y_mb, "val": val, "B": B, "Q": Q, "step": step,
            "x_query": x_mb[:, B:], "rng": jax.random.PRNGKey(seed),
        })
    B, Q = groups[0]["B"], groups[0]["Q"]

    def weights(active: bool):
        w = np.ones(B + Q, dtype=np.float32)
        w[B:] = agree_lambda if active else 0.0
        return jnp.asarray(w)

    w_off, w_on = weights(False), weights(True)
    snaps = []  # (step, [tv per group])
    refreshed = False
    for i in range(steps):
        if agree_lambda > 0 and (i + 1) % agree_every == 0 \
                and (i + 1) >= agree_warmup:
            refreshed = True
            all_preds = _member_raw_preds(groups, [g["tv"] for g in groups])
            for q in range(Q):
                cons = consensus_vote([mp[q] for gp in all_preds for mp in gp])
                for g in groups:
                    rows = [G.place(g["val"][m][2].apply(cons))
                            for m in range(len(g["val"]))]
                    y = np.array(g["y"])  # np.asarray of a jax array is read-only
                    y[:, B + q] = np.stack(rows)
                    g["y"] = jnp.asarray(y)
        active = agree_lambda > 0 and (i + 1) > agree_warmup and refreshed
        for g in groups:
            g["tv"], g["opt_state"], _, g["rng"] = g["step"](
                g["model"], g["tv"], g["opt_state"], g["rng"], g["x"], g["y"],
                w_on if active else w_off)
        if (i + 1) % val_every == 0 or i + 1 == steps:
            snaps.append((i + 1, [np.asarray(g["tv"]) for g in groups]))
    return {"groups": groups, "snaps": snaps, "n_views": n_views,
            "n_seeds": n_seeds}


def score_population_cross(F, episodes):
    """Earliest-LoO-exact selection per member across groups; qualifier vote +
    best-single attempts; per-member query predictions SAVED (the [H-18]
    pairwise matrix comes free with every gate run)."""
    groups, snaps = F["groups"], F["snaps"]
    chosen = []  # flat member list: (g, m, tv, step, exact, pix)
    for gi, g in enumerate(groups):
        M = len(g["val"])
        for m in range(M):
            best = None
            for step_i, tvs in snaps:
                vx, vy, _ = g["val"][m]
                exact, pix, _ = T.evaluate_pair(
                    g["model"], g["cfg"], vx, vy, tau=g["tau"],
                    task_vec=jnp.asarray(tvs[gi][m]))
                if exact:
                    best = (gi, m, tvs[gi][m], step_i, True, pix)
                    break
                if best is None or pix > best[5]:
                    best = (gi, m, tvs[gi][m], step_i, False, pix)
            chosen.append(best)

    qual = [c for c in chosen if c[4]]
    rank = sorted(chosen, key=lambda c: (c[4], c[5]), reverse=True)
    voters = qual if qual else rank[: max(2, len(chosen) // 4)]
    best_single = rank[0]

    # one forward per group at each member's chosen tv, per query
    per_pair, preds, member_preds = [], [], []
    pred_cache = {}
    for c in chosen:
        gi, m, tv, _, _, _ = c
        g = groups[gi]
        tr = g["val"][m][2]
        row = []
        for qi, ep in enumerate(episodes):
            pred, _, _ = T.predict(g["model"], g["cfg"],
                                   tr.apply(np.asarray(ep.query_x)),
                                   tau=g["tau"], task_vec=jnp.asarray(tv))
            row.append(tr.invert_output(pred))
        pred_cache[(gi, m)] = row
        member_preds.append(row)

    for qi, ep in enumerate(episodes):
        votes = [pred_cache[(c[0], c[1])][qi] for c in voters]
        voted = consensus_vote(votes)
        single = pred_cache[(best_single[0], best_single[1])][qi]
        bits = []
        for att in (voted, single):
            ok = bool(ep.query_y is not None and att.shape == ep.query_y.shape
                      and np.array_equal(att, ep.query_y))
            bits.append(ok)
        per_pair.append(bits)
        preds.append([voted.tolist(), single.tolist()])

    member_meta = [{"bulk": groups[c[0]]["name"], "view": groups[c[0]]["val"][c[1]][2].k,
                    "member": c[1], "sel_step": c[3], "loo_exact": c[4],
                    "loo_pix": round(float(c[5]), 4)} for c in chosen]
    return {
        "solved_pass2": all(b[0] or b[1] for b in per_pair),
        "solved_at1": all(b[0] for b in per_pair),
        "solved_joint2": (all(b[0] for b in per_pair)
                          or all(b[1] for b in per_pair)),
        "per_pair_bits": per_pair, "preds": preds,
        "n_qualifiers": len(qual), "n_voters": len(voters),
        "members": member_meta,
        "member_query_preds": [[p.tolist() for p in row]
                               for row in member_preds],
    }


def score_population(F, episodes, *, max_snap_evals: int = 0):
    """Host-side: per member, find the earliest LoO-exact snapshot (subsampled);
    qualifying members predict their view of each query, invert, vote.

    Attempts: [vote-of-qualifiers, best-single-member]. Falls back to
    all-members vote / best-by-pix when no member reaches exactness."""
    model, cfg, tau = F["model"], F["cfg"], F["tau"]
    snaps = F["snaps"]
    # max_snap_evals=0 -> scan ALL snapshots (2026-08-06: subsampling missed
    # exactness — members at 0.99+ pix never qualified and the vote degraded
    # to disagreeing near-missers)
    if max_snap_evals and max_snap_evals < len(snaps):
        idxs = sorted({int(round(i)) for i in
                       np.linspace(0, len(snaps) - 1, max_snap_evals)})
    else:
        idxs = list(range(len(snaps)))
    M = snaps[0][1].shape[0]

    chosen = []  # per member: (tv, step, exact, pix)
    for m in range(M):
        best = None
        for si in idxs:
            step_i, tv_all = snaps[si]
            vx, vy, _ = F["val"][m]
            exact, pix, _ = T.evaluate_pair(model, cfg, vx, vy, tau=tau,
                                            task_vec=jnp.asarray(tv_all[m]))
            if exact:
                best = (tv_all[m], step_i, True, pix)
                break  # earliest-exact (eval-2/3 selection lesson)
            if best is None or pix > best[3]:
                best = (tv_all[m], step_i, False, pix)
        chosen.append(best)

    qual = [m for m in range(M) if chosen[m][2]]
    rank = sorted(range(M), key=lambda m: (chosen[m][2], chosen[m][3]), reverse=True)
    voters = qual if qual else rank[: max(2, M // 4)]
    best_single = rank[0]

    per_pair, preds = [], []
    for ep in episodes:
        votes = []
        for m in voters:
            _, _, tr = F["val"][m]
            pred, _, _ = T.predict(model, cfg, tr.apply(ep.query_x), tau=tau,
                                   task_vec=jnp.asarray(chosen[m][0]))
            votes.append(tr.invert_output(pred))
        shapes = [v.shape for v in votes]
        maj = max(set(shapes), key=shapes.count)
        stack = np.stack([v for v in votes if v.shape == maj])
        voted = np.apply_along_axis(
            lambda v: np.bincount(v, minlength=11).argmax(), 0, stack).astype(np.int8)

        _, _, tr_b = F["val"][best_single]
        pred_b, _, _ = T.predict(model, cfg, tr_b.apply(ep.query_x), tau=tau,
                                 task_vec=jnp.asarray(chosen[best_single][0]))
        single = tr_b.invert_output(pred_b)

        bits = []
        for att in (voted, single):
            ok = bool(ep.query_y is not None and att.shape == ep.query_y.shape
                      and np.array_equal(att, ep.query_y))
            bits.append(ok)
        per_pair.append(bits)
        preds.append([voted.tolist(), single.tolist()])

    return {
        "solved_pass2": all(b[0] or b[1] for b in per_pair),
        "solved_at1": all(b[0] for b in per_pair),
        "per_pair_bits": per_pair, "preds": preds,
        "n_qualifiers": len(qual), "n_voters": len(voters),
        "member_exact": [c[2] for c in chosen],
        "member_pix": [round(float(c[3]), 4) for c in chosen],
        "best_member_view": F["val"][best_single][2].k,
    }
