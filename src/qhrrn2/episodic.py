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
    off_stride: int = 1  # placement offsets are multiples of this (wave-2 box4
    #                      layout needs 4 so boxes stay on 4x4 pooling blocks);
    #                      1 = every offset, the exact pre-existing sampler.


def build_sudoku_corpus(n_train: int, *, n_val: int = 64, seed: int = 0,
                        givens: int = 30, givens_hi: int | None = None):
    """S-PORT corpus (H-33): ONE task row, many instances — Sudoku's rule is
    universal, so this is the single-attractor contrast to ARC's per-task
    rule inference. Train/val puzzles come from ONE sequential rng stream, so
    they are distinct draws by construction (disjointness asserted below).

    No orbit expansion: the generator is the diversity source (the C20a
    precedent for re_ rows). Placement offsets ARE kept — Sudoku constraints
    are position-independent, so translation stays a valid augmentation under
    our convolutional equivariance, exactly as for ARC.
    """
    from . import sudoku as SU
    rng = np.random.default_rng(seed)
    # MIXED DIFFICULTY (2026-08-14 design review, before cell-1 ran): a single
    # hard setting risks a substrate that solves NOTHING, which would VOID all
    # three H-33 readouts rather than answer them. Training across a givens
    # range and evaluating on a difficulty LADDER instead measures where the
    # substrate's propagation depth runs out — an answer either way.
    def draw():
        g = givens if givens_hi is None else int(rng.integers(givens, givens_hi + 1))
        return SU.sample(rng, g)
    pairs = [draw() for _ in range(n_train + n_val)]
    train_pairs, val_pairs = pairs[:n_train], pairs[n_train:]
    seen = {p.tobytes() for p, _ in train_pairs}
    assert not any(p.tobytes() in seen for p, _ in val_pairs), \
        "sudoku val puzzle collided with train — rng stream reuse"

    xs, ys, tidx, bh, bw = [], [], [], [], []
    for x, y in train_pairs:
        xs.append(G.place(x)); ys.append(G.place(y))
        tidx.append(0)
        bh.append(G.CANVAS - SU.N); bw.append(G.CANVAS - SU.N)
    corpus = Corpus(
        task_ids=("sudoku",),
        x=np.stack(xs).astype(np.int32),
        y=np.stack(ys).astype(np.int32),
        tidx=np.asarray(tidx, dtype=np.int32),
        starts=np.asarray([0, len(xs)], dtype=np.int32),
        bound_h=np.asarray(bh, dtype=np.int32),
        bound_w=np.asarray(bw, dtype=np.int32),
    )
    # val slice in build_corpus's format: (task_index, task_id, query pairs)
    return corpus, [(0, "sudoku", val_pairs)]


def _task_pairs(task_id: str, include_queries: bool):
    eps = G.load_task(task_id)
    pairs = list(eps[0].support)
    if include_queries:
        pairs += [(ep.query_x, ep.query_y) for ep in eps if ep.query_y is not None]
    return pairs


def build_corpus(exclude: frozenset[str], *, n_val: int = 20, seed: int = 0,
                 split: str = "training", limit: int | None = None,
                 val_ids: list[str] | None = None, orbit_n: int = 1,
                 conceptarc: bool = False,
                 exclude_ca: frozenset[str] = frozenset(),
                 rearc_families: list[str] | None = None,
                 rearc_per_family: int = 20, rearc_seed: int = 0):
    """Corpus over `split` minus `exclude` (+ optionally ConceptARC minus
    `exclude_ca`), orbit-expanded, plus the val slice.

    orbit_n (assembly doctrine 2026-08-06): each base task additionally
    contributes orbit_n-1 VIRTUAL tasks — a seeded joint D4×palette transform
    of all its pairs, with its OWN task row/embedding (the augmentation
    validity law: a transformed copy may never share the base embedding).
    Val tasks' queries are excluded from ALL their copies (a transformed query
    is trivially recoverable). val is the base-copy monitor list, as before.

    rearc_families (C20a, 2026-08-10): generator families contributing
    rearc_per_family sampled verified instances each, as `re_<fam>` rows —
    SEPARATE rows from any base task of the same id (RE-ARC generators
    occasionally broaden the original rule; sharing a row would blur two
    nearby programs into one embedding). Family-holdout law (C20b) and the
    dev-30/gate-original exclusions are the CALLER's contract via `exclude`
    and the family list; this function only mixes what it is given. RE-ARC
    rows are never val and take no orbit expansion (the generator's own
    sampling is the diversity source).
    """
    ids = [t for t in G.list_task_ids(split) if t not in exclude]
    if limit is not None:
        ids = ids[:limit]
    if val_ids is not None:
        wanted = set(val_ids)
        val_flags = [t in wanted for t in ids]
        missing = wanted - {t for t in ids if t in wanted}
        if missing:
            raise ValueError(f"val_ids not in corpus: {sorted(missing)}")
    else:
        rng = np.random.default_rng(seed)
        n_val = min(n_val, len(ids))
        vs = set(rng.choice(len(ids), size=n_val, replace=False).tolist())
        val_flags = [i in vs for i in range(len(ids))]

    # base tasks: (tid, pairs, is_val, queries_for_monitor)
    base = []
    for task_id, is_val in zip(ids, val_flags):
        pairs = _task_pairs(task_id, include_queries=not is_val)
        qs = []
        if is_val:
            eps = G.load_task(task_id)
            qs = [(ep.query_x, ep.query_y) for ep in eps if ep.query_y is not None]
        base.append((task_id, pairs, is_val, qs))
    if conceptarc:
        for tid, path, concept in G.list_conceptarc():
            if tid in exclude_ca:
                continue
            eps = G.load_task_file(path, tid)
            pairs = list(eps[0].support) + [(e.query_x, e.query_y) for e in eps
                                            if e.query_y is not None]
            base.append((tid, pairs, False, []))
    if rearc_families:
        from qhrrn2 import rearc as R
        rrng = np.random.default_rng(rearc_seed)
        for fam in rearc_families:
            got = []
            for _ in range(rearc_per_family):
                p = R.sample_instance(fam, rrng)
                if p is not None:
                    got.append(p)
            if got:
                base.append((f"re_{fam}", got, False, []))

    # orbit expansion: virtual tasks with their own rows
    expanded = []
    for tid, pairs, is_val, qs in base:
        expanded.append((tid, pairs, is_val, qs))
        if tid.startswith("re_"):   # C20a: generator sampling is the
            continue                # diversity source; no orbit rows
        for k in range(1, orbit_n):
            # zlib.crc32, NOT hash(): Python's hash is salted per process —
            # a resumed pretrain would silently re-derive different transforms
            # for the same virtual rows (caught at build, 2026-08-06)
            import zlib
            trng = np.random.default_rng(zlib.crc32(f"{tid}|{k}".encode()))
            tr = G.sample_orbit(trng, 2)[1]  # one non-identity joint transform
            vpairs = [(tr.apply(x), tr.apply(y)) for x, y in pairs]
            expanded.append((f"{tid}@o{k}", vpairs, False, []))

    xs, ys, tidx, starts, bh, bw, val = [], [], [], [0], [], [], []
    out_ids = []
    for t, (tid, pairs, is_val, qs) in enumerate(expanded):
        out_ids.append(tid)
        if is_val:
            val.append((t, tid, qs))
        for x, y in pairs:
            xs.append(G.place(x))
            ys.append(G.place(y))
            tidx.append(t)
            bh.append(G.CANVAS - max(x.shape[0], y.shape[0]))
            bw.append(G.CANVAS - max(x.shape[1], y.shape[1]))
        starts.append(len(xs))
    corpus = Corpus(
        task_ids=tuple(out_ids),
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


def precompute_labels(corpus: Corpus) -> jnp.ndarray:
    """(P, 3, 32, 32) input segmentations in OBJ_ENC_MODES order, computed
    once at startup (speed pipeline 2026-08-02: labels depend only on the
    sample; recomputing them per training step was the C17 5x tax)."""
    from qhrrn2.model import OBJ_ENC_MODES
    from qhrrn2.objects import connected_components

    @jax.jit
    def one(x):
        return jnp.stack([connected_components(x, m) for m in OBJ_ENC_MODES])
    return jax.lax.map(one, jnp.asarray(corpus.x))


def sample_batch(rng, corpus_dev: dict, n_tasks: int, batch: int):
    """Task-balanced batch with placement augmentation, fully on device.

    corpus_dev: the Corpus arrays as jnp (x, y, starts, bound_h, bound_w),
    plus optional "labels" (P, 3, 32, 32) from precompute_labels.
    Returns (x_b, y_b, t_b, labels_b_or_None) — canvases AND labels rolled by
    the same offset: rolled labels remain valid partitions (segment ids need
    uniqueness, not canonicality; all pre-roll ids are distinct).
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
    # off_stride (wave-2 box4 layout): offsets are multiples of the stride;
    # stride 1 is the exact pre-existing arithmetic (integer floor-divide by 1).
    stride = int(corpus_dev.get("off_stride", 1))
    oy = jnp.floor(u_oy * (corpus_dev["bound_h"][p_b] // stride + 1)).astype(jnp.int32) * stride
    ox = jnp.floor(u_ox * (corpus_dev["bound_w"][p_b] // stride + 1)).astype(jnp.int32) * stride

    def roll(x, y, dy, dx):
        return (jnp.roll(x, (dy, dx), axis=(0, 1)),
                jnp.roll(y, (dy, dx), axis=(0, 1)))

    x_b, y_b = jax.vmap(roll)(x_b, y_b, oy, ox)
    labels_b = None
    if "labels" in corpus_dev:
        lab = corpus_dev["labels"][p_b]
        labels_b = jax.vmap(lambda l, dy, dx: jnp.roll(l, (dy, dx), axis=(1, 2)))(
            lab, oy, ox)
    return x_b, y_b, t_b, labels_b


def build_y0_rows(k_a1, k_a2, k_a3, y_b, anchor_p: float, anchor_eps: float,
                  ri_p: float = 0.0, k_r1=None, k_r2=None):
    """y0 row-type assembly for the eq trainer: [H-23] anchor rows +
    pretrain-13 RI rows (ledger 2026-08-12, EqR deep-read).

    Anchor block (2026-08-09, key usage bit-preserved): with prob anchor_p a
    row starts from corrupt(y) — anchor_eps of its cells resampled — else
    VOID. RI block: with prob ri_p a row starts from a FULL uniform random
    color canvas (the eps=1 limit of the same corruption family) — training
    path-independence from far starts, the init-distribution sibling of the
    anchor rows. RI overrides an anchor draw on the same row, so marginal
    anchor probability is anchor_p*(1-ri_p). ri_p=0 uses no extra keys and
    reproduces the pre-13 formula bit-exactly (tests/test_p13.py).
    Returns (B, 32, 32) int canvases, or None when both probs are 0."""
    yp_b = None
    if anchor_p > 0:
        row = jax.random.bernoulli(k_a1, anchor_p, (y_b.shape[0], 1, 1))
        cell_m = jax.random.bernoulli(k_a2, anchor_eps, y_b.shape)
        rand = jax.random.randint(k_a3, y_b.shape, 0, 10)
        ycor = jnp.where(cell_m, rand, y_b)
        void = jnp.full_like(y_b, 10)
        yp_b = jnp.where(row, ycor, void)
    if ri_p > 0:
        row_ri = jax.random.bernoulli(k_r1, ri_p, (y_b.shape[0], 1, 1))
        rand_ri = jax.random.randint(k_r2, y_b.shape, 0, 10)
        base = yp_b if yp_b is not None else jnp.full_like(y_b, 10)
        yp_b = jnp.where(row_ri, rand_ri, base)
    return yp_b


def corpus_to_device(corpus: Corpus) -> dict:
    return {
        "x": jnp.asarray(corpus.x),
        "y": jnp.asarray(corpus.y),
        "starts": jnp.asarray(corpus.starts),
        "bound_h": jnp.asarray(corpus.bound_h),
        "bound_w": jnp.asarray(corpus.bound_w),
        "off_stride": int(getattr(corpus, "off_stride", 1)),
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
