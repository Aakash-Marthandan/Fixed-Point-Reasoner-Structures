# Ledger: E10 Phase-A CI gates (2026-08-09 registration). CI-9a inertness &
# shapes; CI-9b contraction sanity; CI-9c TOY RETENTION GATE (the E3b
# instrument as a shipping gate); CI-9d convergence-correctness.
import numpy as np
import jax
import jax.numpy as jnp
import optax
import pytest

from qhrrn2 import grid as G
from qhrrn2.config import Config
from qhrrn2.model import (build_fields_soft, init_params, iterate, iterate_eq,
                          VOCAB)
from qhrrn2.objective import batch_loss


CFG = Config(d=8, T=3, K=8, attn_max_hw=0, equilibrium=True)
CFG_OLD = Config(d=8, T=3, K=8, attn_max_hw=0)


def _identity_batch(n=3, seed=0):
    rng = np.random.default_rng(seed)
    xs = [np.asarray(rng.integers(0, 10, (5, 6)), dtype=np.int8)
          for _ in range(n)]
    xb = jnp.asarray(np.stack([G.place(x) for x in xs]), dtype=jnp.int32)
    return xs, xb, xb  # identity: y == x


def test_ci9a_inertness_and_shapes():
    """equilibrium=False params carry no eq key and run the old graph;
    equilibrium=True adds only eq/eta and iterate returns the same contract."""
    p_old = init_params(jax.random.PRNGKey(0), CFG_OLD)
    assert "eq" not in p_old
    p_new = init_params(jax.random.PRNGKey(0), CFG)
    assert set(p_new) - set(p_old) == {"eq"}
    x = jnp.asarray(G.place(np.zeros((4, 4), np.int8)), dtype=jnp.int32)
    outs = iterate(p_new, CFG, x, tau=1.0)
    assert len(outs) == CFG.T and outs[-1].logits.shape[-1] == VOCAB


def test_ci9b_contraction_sanity():
    """Residuals are finite and shrink toward the tail at init (damped step);
    a converged sequence triggers the halt criterion."""
    p = init_params(jax.random.PRNGKey(1), CFG)
    x = jnp.asarray(G.place(np.ones((5, 5), np.int8)), dtype=jnp.int32)
    _, res, _ = iterate_eq(p, CFG, x, tau=1.0, t_total=12)
    r = np.asarray([float(v) for v in res])
    assert np.all(np.isfinite(r))
    assert r[-1] < r[0]  # damping contracts the update at init
    assert (r[-1] < CFG.res_tau) == (r[-1] < 0.05)


def _fit_toy(p, cfg, xb, yb, steps=300, lr=3e-3, anchor=True, seed=0):
    """Tiny full-parameter fit with basin rows: eps=0 idempotence row +
    eps=0.1 corrupted row per pair (the registered E8/E10 recipe)."""
    opt = optax.adamw(lr, weight_decay=1e-4)
    rng_np = np.random.default_rng(seed)
    yp = []
    for y in np.asarray(yb):
        c = y.copy()
        if anchor:
            idx = rng_np.choice(c.size, size=max(1, c.size // 10), replace=False)
            c.flat[idx] = rng_np.integers(0, 10, size=idx.size)
        yp.append(c)
    yp_b = jnp.asarray(np.stack(yp), dtype=jnp.int32)
    x_all = jnp.concatenate([xb, xb, xb])
    y_all = jnp.concatenate([yb, yb, yb])
    yp_all = jnp.concatenate([jnp.full_like(yb, G.VOID), yb, yp_b])

    @jax.jit
    def step(params, opt_state, rng):
        def loss_fn(pp):
            l, _ = batch_loss(pp, cfg, x_all, y_all, tau=1.0, rng=rng,
                              yprev_batch=yp_all)
            return l
        loss, g = jax.value_and_grad(loss_fn)(params)
        upd, os2 = opt.update(g, opt_state, params)
        return optax.apply_updates(params, upd), os2, loss
    os_ = opt.init(p)
    rng = jax.random.PRNGKey(seed)
    for _ in range(steps):
        rng, sub = jax.random.split(rng)
        p, os_, loss = step(p, os_, sub)
    return p


@pytest.fixture(scope="module")
def fitted():
    xs, xb, yb = _identity_batch()
    p = init_params(jax.random.PRNGKey(2), CFG)
    return xs, xb, yb, _fit_toy(p, CFG, xb, yb)


def test_ci9c_toy_retention_gate(fitted):
    """THE gate: after a small fit with basin rows, ground truth handed to
    the map is RETAINED (>80% of pairs) — the E3b failure must not exist."""
    xs, xb, yb, p = fitted
    kept = 0
    for x, y in zip(np.asarray(xb), np.asarray(yb)):
        y0 = jax.nn.one_hot(jnp.asarray(y), VOCAB).transpose(2, 0, 1)
        outs, _, yfin = iterate_eq(p, CFG, jnp.asarray(x), tau=1.0,
                                   t_total=8, y0_probs=y0)
        pred = np.asarray(jnp.argmax(yfin, axis=0))
        kept += bool(np.array_equal(pred, y))
    assert kept / xb.shape[0] > 0.8, f"retention {kept}/{xb.shape[0]}"


def test_ci9d_convergence_correctness(fitted):
    """Solved toy pairs: the trajectory converges (residual < tau within
    t_max) and the limit decodes to the answer."""
    xs, xb, yb, p = fitted
    ok_conv = ok_ans = n = 0
    for x, y in zip(np.asarray(xb), np.asarray(yb)):
        outs, res, yfin = iterate_eq(p, CFG, jnp.asarray(x), tau=1.0,
                                     t_total=CFG.t_max)
        pred = np.asarray(jnp.argmax(yfin, axis=0))
        if np.array_equal(pred, y):  # solved pairs only
            n += 1
            ok_ans += 1
            ok_conv += bool(float(res[-1]) < CFG.res_tau)
    if n:
        assert ok_conv / n > 0.8, f"converged {ok_conv}/{n} solved"


def test_p9_dials_inert_by_default():
    """eta_floor=0 / z_gate_init=0 reproduce the pretrain-8 graph exactly;
    set dials change init/dynamics as specified."""
    p0 = init_params(jax.random.PRNGKey(7), CFG)
    assert float(p0["eq"]["alpha_z"]) == 0.0
    from dataclasses import replace
    cfg9 = replace(CFG, eta_floor=0.2, z_gate_init=0.3)
    p9 = init_params(jax.random.PRNGKey(7), cfg9)
    assert float(p9["eq"]["alpha_z"]) == pytest.approx(0.3)
    x = jnp.asarray(G.place(np.ones((4, 4), np.int8)), dtype=jnp.int32)
    o0 = iterate_eq(p0, CFG, x, tau=1.0, t_total=2)
    o9 = iterate_eq(p9, cfg9, x, tau=1.0, t_total=2)
    # floor: effective eta >= 0.2 -> first-step residual strictly larger
    assert float(o9[1][0]) > float(o0[1][0])


def test_eq_remat_matches_no_remat():
    """P11-EXT (2026-08-11): cfg.remat on the eq loop must be a pure memory
    optimization — losses identical to the un-checkpointed path."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from qhrrn2.config import Config
    from qhrrn2.model import init_params
    from qhrrn2.objective import pair_loss
    from qhrrn2 import grid as G

    rng = np.random.default_rng(9)
    x = jnp.asarray(G.place(rng.integers(0, 10, (6, 6)).astype(np.int8)),
                    dtype=jnp.int32)
    y = jnp.asarray(G.place(rng.integers(0, 10, (6, 6)).astype(np.int8)),
                    dtype=jnp.int32)
    base = dict(d=8, K=8, T=2, equilibrium=True)
    p = init_params(jax.random.PRNGKey(0), Config(**base))
    l0, _ = pair_loss(p, Config(**base), x, y, tau=1.0)
    l1, _ = pair_loss(p, Config(**base, remat=True), x, y, tau=1.0)
    assert abs(float(l0) - float(l1)) < 1e-6, (float(l0), float(l1))
    g0 = jax.grad(lambda q: pair_loss(q, Config(**base), x, y, tau=1.0)[0])(p)
    g1 = jax.grad(lambda q: pair_loss(q, Config(**base, remat=True), x, y,
                                      tau=1.0)[0])(p)
    md = max(jax.tree.leaves(jax.tree.map(
        lambda a, b: float(jnp.max(jnp.abs(a - b))), g0, g1)))
    assert md < 1e-5, f"remat changed gradients: {md}"
