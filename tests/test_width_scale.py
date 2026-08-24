# Ledger: PHASE B ladder (2026-08-24) — --width-scale full-width scaling.
# Named tests: (1) ws=1.0 is bit-exact (identical param trees) vs the default
# construction; (2) side widths scale as registered and params grow ~d^2
# (measured points: d16 77,980 / d64 959,482 / d128 3,709,090).
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qhrrn2.config import Config          # noqa: E402
from qhrrn2 import model                  # noqa: E402


def _side(ws):
    return dict(d_b=int(round(6 * ws)), d_a=int(round(6 * ws)),
                d_ir=int(round(32 * ws)), d_code=int(round(32 * ws)),
                d_task=int(round(32 * ws)))


def _count(cfg):
    return sum(x.size for x in jax.tree.leaves(model.init_params(jax.random.PRNGKey(0), cfg)))


def test_ws1_bit_exact():
    c1 = Config(d=16, T=12, equilibrium=True)
    c2 = Config(d=16, T=12, equilibrium=True, **_side(1.0))
    p1 = model.init_params(jax.random.PRNGKey(7), c1)
    p2 = model.init_params(jax.random.PRNGKey(7), c2)
    l1, l2 = jax.tree.leaves(p1), jax.tree.leaves(p2)
    assert len(l1) == len(l2)
    assert all(bool(jnp.array_equal(a, b)) for a, b in zip(l1, l2))


def test_ladder_param_points():
    assert _count(Config(d=16, T=12, equilibrium=True)) == 77_980
    assert _count(Config(d=64, T=12, equilibrium=True, **_side(4.0))) == 959_482
    assert _count(Config(d=128, T=12, equilibrium=True, **_side(8.0))) == 3_709_090
