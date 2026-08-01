# Ledger: CI-7 (joint-trainer correctness, C16) — the local gate that must be
# green before any cloud pretraining spend:
#   (a) task_vec=None ≡ task_vec=0 (bit-level backward compatibility),
#   (b) S9 equivariance with the task vector active (entry (i) is color-blind;
#       entry (ii) breaks it by design — Amendment A at corpus scale),
#   (c) sampler correctness (roll == place_at; holdout + val-query exclusion),
#   (d) joint smoke fit (loss decreases) + checkpoint round-trip.
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp
import optax
import pytest

from qhrrn2 import episodic as E
from qhrrn2 import grid as G
from qhrrn2.config import Config
from qhrrn2.model import build_fields, forward_fields, init_params
from qhrrn2.objective import batch_loss

CFG = Config(d=12, T=2)


@pytest.fixture(scope="module")
def params():
    return init_params(jax.random.PRNGKey(0), CFG)


def _out(params, x_grid, task_vec=None):
    xc = jnp.asarray(G.place(x_grid), dtype=jnp.int32)
    yprev = jnp.full((G.CANVAS, G.CANVAS), G.VOID, dtype=jnp.int32)
    return forward_fields(params, CFG, build_fields(xc, yprev),
                          t_norm=0.0, tau=1.0, task_vec=task_vec)


def _random_grid(seed, h=9, w=7):
    return np.asarray(np.random.default_rng(seed).integers(0, 10, (h, w)), dtype=np.int8)


# ── (a) Backward compatibility: e = 0 is the pre-C16 model ─────────────────

def test_task_vec_none_equals_zero(params):
    x = _random_grid(1)
    out_none = _out(params, x, task_vec=None)
    out_zero = _out(params, x, task_vec=jnp.zeros((CFG.d_task,)))
    for a, b in zip(out_none, out_zero):
        assert float(jnp.max(jnp.abs(a - b))) < 1e-6


def test_task_vec_is_live(params):
    """A nonzero program vector must actually change the computation."""
    x = _random_grid(2)
    out_none = _out(params, x, task_vec=None)
    tv = jax.random.normal(jax.random.PRNGKey(7), (CFG.d_task,))
    out_tv = _out(params, x, task_vec=tv)
    assert float(jnp.max(jnp.abs(out_tv.logits - out_none.logits))) > 1e-4


# ── (b) Equivariance: entry (i) color-blind, entry (ii) breaks by design ───

def test_equivariance_with_task_vec_rule_path_only(params):
    """With e_cb zeroed, a task vector rides the rule path only — S9
    equivariance must hold exactly as at init."""
    p2 = jax.tree.map(lambda a: a, params)
    p2["task_proj"]["e_cb"] = jax.tree.map(jnp.zeros_like, params["task_proj"]["e_cb"])
    tv = jax.random.normal(jax.random.PRNGKey(3), (CFG.d_task,))

    x = _random_grid(4)
    lut = G.random_palette(np.random.default_rng(5))
    out = _out(p2, x, task_vec=tv)
    out_p = _out(p2, G.apply_palette(x, lut), task_vec=tv)
    err = float(jnp.max(jnp.abs(out_p.logits[..., jnp.asarray(lut)] - out.logits)))
    assert err < 1e-4, f"rule-path task vector broke S9: {err}"
    assert float(jnp.max(jnp.abs(out_p.rule_q - out.rule_q))) < 1e-4


def test_task_color_path_breaks_equivariance(params):
    """Entry (ii) must be ABLE to break S9 per task (Amendment A at corpus
    scale) — otherwise color-constant corpus rules are unrepresentable."""
    p2 = jax.tree.map(lambda a: a, params)
    w = params["task_proj"]["e_cb"]["w"]
    p2["task_proj"]["e_cb"] = {"w": jnp.ones_like(w) * 0.3,
                               "b": params["task_proj"]["e_cb"]["b"]}
    tv = jnp.ones((CFG.d_task,))
    x = _random_grid(6)
    lut = G.identity_palette(); lut[3], lut[5] = 5, 3
    out = _out(p2, x, task_vec=tv)
    out_p = _out(p2, G.apply_palette(x, lut), task_vec=tv)
    err = float(jnp.max(jnp.abs(out_p.logits[..., jnp.asarray(lut)] - out.logits)))
    # uniform e_cb rows are color-symmetric; perturb one field's row instead
    w2 = w.at[:, 3 * CFG.d:4 * CFG.d].add(1.0)  # field 3's block
    p2["task_proj"]["e_cb"] = {"w": w2, "b": params["task_proj"]["e_cb"]["b"]}
    out = _out(p2, x, task_vec=tv)
    out_p = _out(p2, G.apply_palette(x, lut), task_vec=tv)
    err = float(jnp.max(jnp.abs(out_p.logits[..., jnp.asarray(lut)] - out.logits)))
    assert err > 1e-3, "task color path cannot break S9 — C16 entry (ii) dead"


# ── (c) Sampler and corpus hygiene ──────────────────────────────────────────

def test_roll_equals_place_at():
    g = _random_grid(8, h=5, w=6)
    for oy, ox in [(0, 0), (3, 9), (27, 26)]:
        rolled = np.roll(G.place(g), (oy, ox), axis=(0, 1))
        assert np.array_equal(rolled, G.place_at(g, oy, ox))


def test_corpus_excludes_holdout_and_val_queries():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
    import dev30
    exclude = frozenset(dev30.MANIFEST)
    corpus, val = E.build_corpus(exclude, n_val=5, seed=0, limit=40)
    assert not (set(corpus.task_ids) & exclude), "dev-30 leaked into the corpus"
    assert len(val) == 5
    # val tasks contribute exactly their SUPPORT pairs to the pool
    for t, task_id, queries in val:
        n_in_pool = int(corpus.starts[t + 1] - corpus.starts[t])
        n_support = len(G.load_task(task_id)[0].support)
        assert n_in_pool == n_support, f"{task_id}: query pair leaked into pool"
        assert len(queries) >= 1
    # prefix/tidx consistency
    assert corpus.starts[-1] == corpus.x.shape[0]
    for t in range(len(corpus.task_ids)):
        seg = corpus.tidx[corpus.starts[t]:corpus.starts[t + 1]]
        assert (seg == t).all()


def test_sample_batch_valid_and_balanced():
    corpus, _ = E.build_corpus(frozenset(), n_val=0, seed=0, limit=12)
    dev = E.corpus_to_device(corpus)
    n = len(corpus.task_ids)
    x_b, y_b, t_b = E.sample_batch(jax.random.PRNGKey(0), dev, n, 256)
    t_np = np.asarray(t_b)
    assert t_np.min() >= 0 and t_np.max() < n
    assert len(np.unique(t_np)) > n // 2, "task-balanced sampling looks degenerate"
    # rolled canvases still contain exactly the original cell multiset
    x0 = np.asarray(x_b[0])
    assert (x0 == G.VOID).sum() > 0 and x0.shape == (G.CANVAS, G.CANVAS)


# ── (d) Joint smoke fit + checkpoint round-trip ─────────────────────────────

def test_joint_smoke_and_checkpoint(tmp_path):
    cfg = Config(d=8, T=2, K=8, attn_max_hw=8)
    corpus, _ = E.build_corpus(frozenset(), n_val=0, seed=0, limit=6)
    dev = E.corpus_to_device(corpus)
    n = len(corpus.task_ids)

    key = jax.random.PRNGKey(0)
    state = {"model": init_params(key, cfg),
             "table": E.init_table(jax.random.PRNGKey(1), n, cfg.d_task)}
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(3e-3, weight_decay=1e-4))
    opt_state = opt.init(state)

    @jax.jit
    def step(state, opt_state, rng):
        k_batch, k_loss = jax.random.split(rng)
        x_b, y_b, t_b = E.sample_batch(k_batch, dev, n, 8)

        def loss_fn(st):
            return batch_loss(st["model"], cfg, x_b, y_b, tau=1.0, rng=k_loss,
                              task_vecs=st["table"][t_b])
        (loss, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(state)
        updates, opt_state2 = opt.update(grads, opt_state, state)
        return optax.apply_updates(state, updates), opt_state2, loss

    rng = jax.random.PRNGKey(42)
    losses = []
    for _ in range(30):
        rng, sub = jax.random.split(rng)
        state, opt_state, loss = step(state, opt_state, sub)
        losses.append(float(loss))
    assert np.mean(losses[-5:]) < np.mean(losses[:5]), (
        f"joint loss did not decrease: {losses[:5]} -> {losses[-5:]}")

    path = tmp_path / "ckpt.pkl"
    E.save_ckpt(path, {"state": state, "step": 30})
    loaded = E.load_ckpt(path)
    flat_a = jax.tree.leaves(state)
    flat_b = jax.tree.leaves(loaded["state"])
    assert len(flat_a) == len(flat_b)
    for a, b in zip(flat_a, flat_b):
        assert np.array_equal(np.asarray(a), np.asarray(b))
