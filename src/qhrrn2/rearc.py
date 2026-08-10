# Ledger: C20a (RE-ARC vendor adapter, registered 2026-08-10) + C20b
# (family-holdout law: a seeded fraction of generator FAMILIES is reserved
# for the family-transfer gate and never enters pretraining).
#
# RE-ARC (Hodel, MIT, pinned e5b7f1d, vendored at data/re_arc) provides one
# parameterized example generator per ARC-1 training task ("family"). This
# adapter samples verified instances into our Episode format. matplotlib is
# stubbed at import (utils.py wants it only for plotting we never call).
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

from qhrrn2 import grid as G

REARC_ROOT = Path(__file__).resolve().parents[2] / "data" / "re_arc"
_MODS = None


def _load_rearc():
    """Import the vendored generators/verifiers with matplotlib stubbed."""
    global _MODS
    if _MODS is not None:
        return _MODS
    if "matplotlib" not in sys.modules:
        mpl = types.ModuleType("matplotlib")
        mpl.__path__ = []  # mark as package so submodule imports resolve
        for sub in ("pyplot", "colors"):
            m = types.ModuleType(f"matplotlib.{sub}")
            if sub == "colors":
                m.ListedColormap = object
                m.Normalize = object
            setattr(mpl, sub, m)
            sys.modules[f"matplotlib.{sub}"] = m
        sys.modules["matplotlib"] = mpl
    sys.path.insert(0, str(REARC_ROOT))
    try:
        import generators
        import verifiers
    finally:
        sys.path.pop(0)
    _MODS = (generators, verifiers)
    return _MODS


def family_ids() -> list[str]:
    """All generator family keys (ARC-1 training task ids), sorted."""
    generators, _ = _load_rearc()
    return sorted(n[len("generate_"):] for n in dir(generators)
                  if n.startswith("generate_"))


def family_split(*, holdout_frac: float = 0.25, seed: int = 20260810):
    """C20b: seeded family-level split -> (train_families, gate_families).
    The gate families NEVER enter pretraining in any form."""
    fams = family_ids()
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(fams))
    n_gate = int(round(holdout_frac * len(fams)))
    gate = sorted(fams[i] for i in idx[:n_gate])
    train = sorted(fams[i] for i in idx[n_gate:])
    return train, gate


def _to_grid(t) -> np.ndarray:
    return np.asarray(t, dtype=np.int8)


def sample_instance(fam: str, rng: np.random.Generator, *,
                    diff_lb: float = 0.0, diff_ub: float = 1.0,
                    max_tries: int = 12):
    """One verified (input, output) pair from a family's generator.
    Returns None if the generator refuses within max_tries (some samplers
    raise on unlucky draws — RE-ARC's own harness retries the same way)."""
    generators, verifiers = _load_rearc()
    gen = getattr(generators, f"generate_{fam}")
    ver = getattr(verifiers, f"verify_{fam}", None)
    import random as _random
    for _ in range(max_tries):
        _random.seed(int(rng.integers(0, 2**31)))
        try:
            ex = gen(diff_lb, diff_ub)
            gi, go = _to_grid(ex["input"]), _to_grid(ex["output"])
        except Exception:
            continue
        if gi.size == 0 or go.size == 0 or max(gi.shape + go.shape) > 30:
            continue
        if ver is not None:
            try:
                if _to_grid(ver(ex["input"])).tolist() != go.tolist():
                    continue
            except Exception:
                continue
        return gi, go
    return None


def sample_episode(fam: str, rng: np.random.Generator, *, n_support: int = 3,
                   **kw) -> G.Episode | None:
    """One Episode: n_support demonstration pairs + one query, all fresh
    instances of the same family (the generator IS the rule)."""
    pairs = []
    for _ in range(n_support + 1):
        p = sample_instance(fam, rng, **kw)
        if p is None:
            return None
        pairs.append(p)
    return G.Episode(
        task_id=f"re_{fam}",
        support=tuple(pairs[:-1]),
        query_x=pairs[-1][0],
        query_y=pairs[-1][1])


def sample_corpus_pairs(families: list[str], *, per_family: int, seed: int):
    """Pretraining pair pool: {family: [(x, y), ...]} — flat pairs, one
    virtual task row per family (the C16 embedding-table contract)."""
    rng = np.random.default_rng(seed)
    out = {}
    for fam in families:
        got = []
        for _ in range(per_family):
            p = sample_instance(fam, rng)
            if p is not None:
                got.append(p)
        if got:
            out[fam] = got
    return out
