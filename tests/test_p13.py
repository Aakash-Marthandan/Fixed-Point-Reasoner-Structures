# Ledger: pretrain-13 mechanism gates (2026-08-12 registration; EqR/FPRM
# deep-read consequences). Every mechanism is branch-inert at its config
# default (pretrain-12 reproduced bit-exactly) and demonstrably LIVE when
# enabled: RI rows (episodic.build_y0_rows), NI per-step training noise,
# FPRM coupled residual scaling a1/a2, B1-full free-bits flux floors.
import numpy as np
import jax
import jax.numpy as jnp

from qhrrn2 import episodic as E
from qhrrn2 import grid as G
from qhrrn2.config import Config
from qhrrn2.model import init_params, iterate_eq
from qhrrn2.objective import pair_loss

CFG = Config(d=8, T=3, K=8, attn_max_hw=0, equilibrium=True)


def _xy(seed=0, hw=(5, 6)):
    rng = np.random.default_rng(seed)
    x = np.asarray(rng.integers(0, 10, hw), dtype=np.int8)
    xc = jnp.asarray(G.place(x), dtype=jnp.int32)
    return xc, xc  # identity pair


# ---------- RI rows ----------

def test_ri_rows_inert_at_zero():
    """ri_p=0 reproduces the pre-13 anchor formula bit-exactly for the same
    keys (the [H-23] block was lifted verbatim into build_y0_rows)."""
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(7), 3)
    y_b = jnp.asarray(np.random.default_rng(0).integers(0, 10, (6, 32, 32)),
                      dtype=jnp.int32)
    got = E.build_y0_rows(k1, k2, k3, y_b, 0.4, 0.15, ri_p=0.0)
    # the pre-13 inline formula, verbatim:
    row = jax.random.bernoulli(k1, 0.4, (y_b.shape[0], 1, 1))
    cell_m = jax.random.bernoulli(k2, 0.15, y_b.shape)
    rand = jax.random.randint(k3, y_b.shape, 0, 10)
    ref = jnp.where(row, jnp.where(cell_m, rand, y_b), jnp.full_like(y_b, 10))
    assert jnp.array_equal(got, ref)
    assert E.build_y0_rows(k1, k2, k3, y_b, 0.0, 0.15) is None


def test_ri_rows_full_random():
    """ri_p=1: every row is a full uniform color canvas — no VOID anywhere,
    independent of the anchor draw (the eps=1 limit)."""
    ks = jax.random.split(jax.random.PRNGKey(3), 5)
    y_b = jnp.full((4, 32, 32), 5, dtype=jnp.int32)
    got = E.build_y0_rows(ks[0], ks[1], ks[2], y_b, 0.4, 0.15,
                          ri_p=1.0, k_r1=ks[3], k_r2=ks[4])
    assert int(jnp.sum(got == 10)) == 0          # no VOID rows survive
    assert int(jnp.max(got)) <= 9 and int(jnp.min(got)) >= 0
    assert len(np.unique(np.asarray(got))) > 3   # actually random, not y_b


# ---------- coupled residual scaling ----------

def test_eq_coupled_params_and_init():
    """eq_coupled=False adds no keys (old checkpoints load); True adds
    alpha1/alpha2 initialized contractive at sigmoid -> .75/.25."""
    p_off = init_params(jax.random.PRNGKey(0), CFG)
    assert "alpha1" not in p_off["eq"] and "alpha2" not in p_off["eq"]
    cfgc = Config(**{**CFG.__dict__, "eq_coupled": True})
    p_on = init_params(jax.random.PRNGKey(0), cfgc)
    a1 = float(jax.nn.sigmoid(p_on["eq"]["alpha1"]))
    a2 = float(jax.nn.sigmoid(p_on["eq"]["alpha2"]))
    assert abs(a1 - 0.75) < 1e-4 and abs(a2 - 0.25) < 1e-4
    assert a1 + a2 <= 1.0 + 1e-4                  # contractive at init


def test_eq_coupled_live_and_grad():
    """Coupled update changes the trajectory (mechanism live) and gradients
    reach alpha1/alpha2."""
    cfgc = Config(**{**CFG.__dict__, "eq_coupled": True})
    p_on = init_params(jax.random.PRNGKey(1), cfgc)
    x, y = _xy(1)
    outs_off, _, yf_off = iterate_eq(
        {**p_on, "eq": {k: v for k, v in p_on["eq"].items()
                        if not k.startswith("alpha1") and not k.startswith("alpha2")}},
        CFG, x, tau=1.0)
    outs_on, _, yf_on = iterate_eq(p_on, cfgc, x, tau=1.0)
    assert not jnp.allclose(yf_off, yf_on)        # .75/.25 != 1-eta/eta path

    def loss(p):
        return pair_loss(p, cfgc, x, y, tau=1.0)[0]
    g = jax.grad(loss)(p_on)
    assert float(jnp.abs(g["eq"]["alpha1"])) > 0
    assert float(jnp.abs(g["eq"]["alpha2"])) > 0


# ---------- NI training noise ----------

def test_ni_scoping():
    """rng=None never sees noise (inference safety, bit-exact vs sigma=0);
    with rng, sigma>0 changes the loss and different rngs differ; the state
    stays on-simplex after the noise step."""
    x, y = _xy(2)
    p = init_params(jax.random.PRNGKey(2), CFG)
    cfgn = Config(**{**CFG.__dict__, "ni_sigma": 0.05})
    l_base = pair_loss(p, CFG, x, y, tau=1.0)[0]
    l_none = pair_loss(p, cfgn, x, y, tau=1.0)[0]          # rng=None
    assert jnp.array_equal(l_base, l_none)
    k = jax.random.PRNGKey(5)
    l_noise1 = pair_loss(p, cfgn, x, y, tau=1.0, rng=k)[0]
    l_noise2 = pair_loss(p, cfgn, x, y, tau=1.0,
                         rng=jax.random.PRNGKey(6))[0]
    assert not jnp.allclose(l_noise1, l_noise2)
    assert jnp.isfinite(l_noise1) and jnp.isfinite(l_noise2)
    _, _, yf = iterate_eq(p, cfgn, x, tau=1.0, rng=k)
    sums = jnp.sum(yf, axis=0)
    assert float(jnp.max(jnp.abs(sums - 1.0))) < 1e-3      # renormalized


# ---------- flux floors ----------

def test_flux_floors():
    """() = the global toll exactly; floors above every measured I_s zero the
    toll (loss == beta 0); zero floors == () bit-exact."""
    x, y = _xy(3)
    base = Config(**{**CFG.__dict__, "beta_flux": 1e-3})
    p = init_params(jax.random.PRNGKey(4), base)
    l_glob = pair_loss(p, base, x, y, tau=1.0)[0]
    z5 = Config(**{**base.__dict__, "flux_floors": "0,0,0,0,0"})
    assert jnp.array_equal(pair_loss(p, z5, x, y, tau=1.0)[0], l_glob)
    hi = Config(**{**base.__dict__, "flux_floors": "1e9,1e9,1e9,1e9,1e9"})
    free = Config(**{**CFG.__dict__, "beta_flux": 0.0})
    assert jnp.allclose(pair_loss(p, hi, x, y, tau=1.0)[0],
                        pair_loss(p, free, x, y, tau=1.0)[0])


# ---------- the bundle trains ----------

def test_p13_bundle_descends():
    """All mechanisms on together (RI enters via y0; NI + coupled + floors in
    cfg): a short fit on the identity pair descends and stays finite."""
    import optax
    cfgb = Config(**{**CFG.__dict__, "eq_coupled": True, "ni_sigma": 0.02,
                     "beta_flux": 1e-4, "flux_floors": "50,20,10,5,5"})
    p = init_params(jax.random.PRNGKey(8), cfgb)
    x, y = _xy(4)
    opt = optax.adam(3e-3)
    st = opt.init(p)
    ks = jax.random.split(jax.random.PRNGKey(9), 5)
    y0 = E.build_y0_rows(ks[0], ks[1], ks[2],
                         y[None].astype(jnp.int32), 0.0, 0.15,
                         ri_p=1.0, k_r1=ks[3], k_r2=ks[4])[0]

    def loss(pp, k):
        return pair_loss(pp, cfgb, x, y, tau=1.0, rng=k, yprev_init=y0)[0]

    l0 = None
    k = jax.random.PRNGKey(10)
    for i in range(30):
        k, ki = jax.random.split(k)
        l, g = jax.value_and_grad(loss)(p, ki)
        if l0 is None:
            l0 = float(l)
        up, st = opt.update(g, st)
        p = optax.apply_updates(p, up)
    assert np.isfinite(float(l)) and float(l) < l0
