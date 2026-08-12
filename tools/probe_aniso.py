# Ledger: cluster P (retention-metric anisotropy, freethink 2026-08-12;
# registration in §5) — the [H-12] hard-negative turned into a design
# principle: measure WHICH parameter subspaces basin structure is sensitive
# to. Per task: one keyhole fit (battery protocol), baseline GT-retention,
# then random unit perturbations per (subspace, relative norm rho, dir) and
# the retention under each. The anisotropy spectrum gates convert-phase TTT
# ("fit where the code isn't stored") and any 5-75M TTT commitment.
# Instrument only; GT used only to score retention (e3b semantics).
"""
  python tools/probe_aniso.py --ckpt runs/pretrain12_48c_40k/ckpt_latest.pkl \
      --tasks ca_X,ca_Y --out runs/aniso_p1248c40k
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import jax
import jax.numpy as jnp

import probe_e1e3 as P
import probe_ladder as L
from qhrrn2 import episodic as E
from qhrrn2 import grid as G
from qhrrn2.config import Config

RHOS = (0.01, 0.03, 0.1)   # relative perturbation norms (x subspace L2)
N_DIRS = 2                 # random directions per (subspace, rho)

# subspace -> (top-level key, optional sub-key filter); "e_t" is special
# (perturbs the fitted task vector, params untouched)
SUBSPACES = {
    "mixers":   [("enc", "mixer"), ("dec", "mixer")],   # the compute spine
    "attn":     [("enc", "attn"), ("dec", "attn")],
    "film":     [("enc", "film"), ("dec", "film")],
    "gates":    [("gate", None)],
    "heads":    [("canvas", None), ("readout", None)],
    "codebook": [("codebook", None), ("rule_query", None)],
    "embed":    [("embed", None), ("role_emb", None), ("dec_init", None),
                 ("ir_proj", None)],
    "boundary": [("color_bias", None), ("task_proj", None)],  # C10 philosophy
    "eq":       [("eq", None)],
    "e_t":      None,
}


def _walk(node, prefix=()):
    """(path, leaf) pairs via plain dict recursion — the param tree is
    nested dicts of arrays, nothing fancier."""
    if isinstance(node, dict):
        out = []
        for k, v in node.items():
            out += _walk(v, prefix + (k,))
        return out
    return [(prefix, node)]


def _leaves(tree, spec):
    out = []
    for top, sub in spec:
        node = tree[top] if sub is None else tree[top][sub]
        out += _walk(node, (top,) if sub is None else (top, sub))
    return out


def perturb(model, tvj, name, rho, seed):
    """(model', tvj') with a random unit direction in the named subspace
    scaled to rho * ||subspace||_2 added; everything else untouched."""
    rng = np.random.default_rng(seed)
    if name == "e_t":
        v = np.asarray(tvj)
        u = rng.standard_normal(v.shape)
        u /= max(np.linalg.norm(u), 1e-12)
        return model, jnp.asarray(v + rho * np.linalg.norm(v) * u,
                                  dtype=v.dtype)
    leaves = _leaves(model, SUBSPACES[name])
    norm = float(np.sqrt(sum(float(jnp.sum(a * a)) for _, a in leaves)))
    us = {path: rng.standard_normal(np.shape(a)) for path, a in leaves}
    u_norm = float(np.sqrt(sum(float((u * u).sum()) for u in us.values())))
    scale = rho * norm / max(u_norm, 1e-12)

    def rebuild(node, prefix=()):
        if isinstance(node, dict):
            return {k: rebuild(v, prefix + (k,)) for k, v in node.items()}
        u = us.get(prefix)
        return node if u is None else node + jnp.asarray(
            scale * u, dtype=node.dtype)

    return rebuild(model), tvj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tasks", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--val-every", type=int, default=50)
    ap.add_argument("--stab-steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    saved = E.load_ckpt(a.ckpt)
    defaults = Config()
    cfg = Config(**{k: type(getattr(defaults, k))(v)
                    for k, v in saved["config"].items()})
    state = saved["state"]

    import dev30
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results = out / "results.jsonl"
    done = set()
    if results.exists():
        for line in results.read_text().splitlines():
            try:
                done.add(json.loads(line)["task"])
            except Exception:
                pass
    task_ids = a.tasks.split(",") if a.tasks else sorted(dev30.MANIFEST)

    with open(results, "a") as f:
        for tid in task_ids:
            if tid in done:
                print(f"skip {tid}", flush=True)
                continue
            t0 = time.time()
            eps_list = G.load_task(tid)
            model, snaps, sel, F = P.fit_arm_a(
                state, cfg, eps_list, steps=a.steps, val_every=a.val_every,
                seed=a.seed)
            tvj = jnp.asarray(sel[1])
            queries = [(qi, ep) for qi, ep in enumerate(eps_list)
                       if ep.query_y is not None]
            base = [all(L.retained(model, cfg, ep.query_x,
                                   np.asarray(ep.query_y), tvj, a.stab_steps))
                    for qi, ep in queries]
            row = {"task": tid, "sel_step": sel[0],
                   "base_ret": [bool(b) for b in base], "cells": []}
            for name in SUBSPACES:
                for rho in RHOS:
                    for d in range(N_DIRS):
                        # stable across processes (hash() is salted per run)
                        seed = (a.seed * 7919
                                + list(SUBSPACES).index(name) * 101
                                + int(rho * 1000) * 13 + d)
                        m2, tv2 = perturb(model, tvj, name, rho, seed)
                        ret = [all(L.retained(m2, cfg, ep.query_x,
                                              np.asarray(ep.query_y), tv2,
                                              a.stab_steps))
                               for qi, ep in queries]
                        row["cells"].append(
                            {"sub": name, "rho": rho, "dir": d,
                             "ret": [bool(r) for r in ret]})
            row["wall_s"] = round(time.time() - t0, 1)
            f.write(json.dumps(row) + "\n")
            f.flush()
            nb = sum(base)
            drops = {}
            for c in row["cells"]:
                k = c["sub"]
                drops[k] = drops.get(k, 0) + (nb - sum(c["ret"]))
            print(f"{tid} base {nb}/{len(base)} drops {drops} "
                  f"({row['wall_s']:.0f}s)", flush=True)
    print("ANISO PROBE DONE", flush=True)


if __name__ == "__main__":
    main()
