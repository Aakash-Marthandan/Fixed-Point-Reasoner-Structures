# Ledger: C16 (joint-episodic pretraining) — shared bulk + per-task program
# embeddings over the public training split minus the frozen dev-30 holdout
# (ledger 2026-08-01; dev-30 frozen at commit 10c3ac9). The augmentation
# validity law holds at corpus scale: placement offsets only — a D4/palette
# copy of a task may NEVER share its embedding row (contradictory-supervision
# trap, ledger 2026-07-20); orbit expansion, if ever used, means NEW rows.
"""Joint-episodic corpus: build, sample, and the per-task embedding table.

The corpus is flattened to fixed-shape arrays once on the host (all pairs
placed at the canvas origin); per-step placement augmentation is a jnp.roll
by a bounded random offset, which is exactly `place_at(oy, ox)` because the
wrapped-in cells are all VOID. Sampling is task-balanced: uniform over tasks,
then uniform over that task's pairs — a 2-pair task and a 10-pair task get
equal expected gradient weight.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp

from qhrrn2 import grid as G


@dataclass(frozen=True)
class Corpus:
    """Flattened pair arrays, sorted by task; index i in [starts[t], starts[t+1])
    belongs to task_ids[t]. Bounds are the MAX valid placement offsets."""
    task_ids: tuple[str, ...]
    x: np.ndarray        # (P, 32, 32) int32, placed at origin
    y: np.ndarray        # (P, 32, 32) int32
    tidx: np.ndarray     # (P,) int32 task index
    starts: np.ndarray   # (n_tasks + 1,) int32 prefix offsets into the pair axis
    bound_h: np.ndarray  # (P,) int32 max oy (inclusive)
    bound_w: np.ndarray  # (P,) int32 max ox (inclusive)


def _task_pairs(task_id: str, include_queries: bool):
    eps = G.load_task(task_id)
    pairs = list(eps[0].support)
    if include_queries:
        pairs += [(ep.query_x, ep.query_y) for ep in eps if ep.query_y is not None]
    return pairs


def build_corpus(exclude: frozenset[str], *, n_val: int = 20, seed: int = 0,
                 split: str = "training", limit: int | None = None):
    """Corpus over `split` minus `exclude`, plus the val-20 slice.

    Returns (corpus, val) where val is a list of (task_index, task_id,
    [(query_x, query_y), ...]) whose QUERY pairs were withheld from the pair
    pool (their supports remain) — the within-task generalization monitor.
    """
    ids = [t for t in G.list_task_ids(split) if t not in exclude]
    if limit is not None:
        ids = ids[:limit]
    rng = np.random.default_rng(seed)
    n_val = min(n_val, len(ids))
    val_set = set(rng.choice(len(ids), size=n_val, replace=False).tolist())

    xs, ys, tidx, starts, bh, bw, val = [], [], [], [0], [], [], []
    for t, task_id in enumerate(ids):
        pairs = _task_pairs(task_id, include_queries=t not in val_set)
        if t in val_set:
            eps = G.load_task(task_id)
            val.append((t, task_id,
                        [(ep.query_x, ep.query_y) for ep in eps if ep.query_y is not None]))
        for x, y in pairs:
            xs.append(G.place(x))
            ys.append(G.place(y))
            tidx.append(t)
            bh.append(G.CANVAS - max(x.shape[0], y.shape[0]))
            bw.append(G.CANVAS - max(x.shape[1], y.shape[1]))
        starts.append(len(xs))
    corpus = Corpus(
        task_ids=tuple(ids),
        x=np.stack(xs).astype(np.int32),
        y=np.stack(ys).astype(np.int32),
        tidx=np.asarray(tidx, dtype=np.int32),
        starts=np.asarray(starts, dtype=np.int32),
        bound_h=np.asarray(bh, dtype=np.int32),
        bound_w=np.asarray(bw, dtype=np.int32),
    )
    return corpus, val


def init_table(key, n_tasks: int, d_task: int):
    """Per-task program embeddings; small init = near-neutral programs (the
    e=0 point is the exact pre-C16 model)."""
    return jax.random.normal(key, (n_tasks, d_task)) * 0.1


def sample_batch(rng, corpus_dev: dict, n_tasks: int, batch: int):
    """Task-balanced batch with placement augmentation, fully on device.

    corpus_dev: the Corpus arrays as jnp (x, y, starts, bound_h, bound_w).
    Returns (x_b, y_b, t_b) — canvases rolled to a valid random offset.
    """
    k_task, k_pair, k_oy, k_ox = jax.random.split(rng, 4)
    t_b = jax.random.randint(k_task, (batch,), 0, n_tasks)
    lo = corpus_dev["starts"][t_b]
    hi = corpus_dev["starts"][t_b + 1]
    u = jax.random.uniform(k_pair, (batch,))
    p_b = lo + jnp.floor(u * (hi - lo)).astype(jnp.int32)

    x_b = corpus_dev["x"][p_b]
    y_b = corpus_dev["y"][p_b]
    u_oy = jax.random.uniform(k_oy, (batch,))
    u_ox = jax.random.uniform(k_ox, (batch,))
    oy = jnp.floor(u_oy * (corpus_dev["bound_h"][p_b] + 1)).astype(jnp.int32)
    ox = jnp.floor(u_ox * (corpus_dev["bound_w"][p_b] + 1)).astype(jnp.int32)

    def roll(x, y, dy, dx):
        return (jnp.roll(x, (dy, dx), axis=(0, 1)),
                jnp.roll(y, (dy, dx), axis=(0, 1)))

    x_b, y_b = jax.vmap(roll)(x_b, y_b, oy, ox)
    return x_b, y_b, t_b


def corpus_to_device(corpus: Corpus) -> dict:
    return {
        "x": jnp.asarray(corpus.x),
        "y": jnp.asarray(corpus.y),
        "starts": jnp.asarray(corpus.starts),
        "bound_h": jnp.asarray(corpus.bound_h),
        "bound_w": jnp.asarray(corpus.bound_w),
    }


# ── Checkpointing (host-side pickle of pure array pytrees) ──────────────────

def save_ckpt(path, tree):
    """Pickle a pytree with DEVICE arrays pulled to host numpy. Non-array
    leaves (ints, floats, strings in metadata like the config dict) pass
    through untouched — 2026-08-01: np.asarray(16) is an unhashable 0-d
    array, and a Config rebuilt from such leaves broke lru_cache hashing."""
    import pickle
    host = jax.tree.map(lambda a: np.asarray(a) if isinstance(a, jax.Array) else a,
                        tree)
    with open(path, "wb") as f:
        pickle.dump(host, f)


def load_ckpt(path):
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)
