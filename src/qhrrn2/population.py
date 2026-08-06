# Ledger: C11 (population TTT, spec §; promoted to prerequisite by eval-4's
# per-view finding and the assembly doctrine 2026-08-06). Members are
# (D4 view k, seed s): each fits e_t ONLY against the frozen bulk on ITS
# transformed copy of the supports — per-view supervision is self-consistent
# (the conjugated task), so the fit-time validity law holds. Prediction:
# qualifying members predict on their view, INVERT, and vote.
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
