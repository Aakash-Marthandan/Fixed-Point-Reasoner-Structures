"""Checkpoint / run-dir provenance inspector (2026-08-18).

Prints, for every ckpt path or run dir given, what the artifact actually
IS: step, model config (d, T, scales...), bulk param count, and — when a
config.json sits beside it — the training argv (steps, seed, arm flags,
git rev). Science-integrity tool: a banked ckpt is admitted to a verdict
only after this says it is the registered config, never on its filename.

  .venv/bin/python tools/inspect_ckpt.py PATH [PATH ...]
PATH may be a .pkl or a run dir (uses ckpt_latest.pkl + config.json).
"""
from __future__ import annotations
import json
import pickle
import sys
from pathlib import Path


def _n_leaves(tree) -> int:
    import numpy as np
    n = 0
    stack = [tree]
    while stack:
        t = stack.pop()
        if isinstance(t, dict):
            stack.extend(t.values())
        elif isinstance(t, (list, tuple)):
            stack.extend(t)
        else:
            try:
                n += int(np.prod(np.shape(t)))
            except Exception:
                pass
    return n


def show_ckpt(p: Path) -> None:
    try:
        with open(p, "rb") as f:
            d = pickle.load(f)
    except Exception as e:  # truncated / foreign
        print(f"{p}: LOAD-FAIL {type(e).__name__}: {e}")
        return
    cfg = d.get("config", {}) or {}
    st = d.get("state", {}) or {}
    n_bulk = _n_leaves(st.get("model", {})) if isinstance(st, dict) else -1
    want = ("d", "T", "scales", "K", "d_task", "attn", "equilibrium",
            "beta", "floor", "ni", "sigma", "noise", "eta", "chi")
    keys = [k for k in cfg if any(w in k for w in want)]
    kv = ", ".join(f"{k}={cfg[k]}" for k in sorted(keys))
    print(f"{p}: step={d.get('step')} n_bulk={n_bulk} cfg{{ {kv} }}")


def show_dir(p: Path) -> None:
    ck = p / "ckpt_latest.pkl"
    if ck.exists():
        show_ckpt(ck)
    else:
        print(f"{p}: (no ckpt_latest.pkl)")
    cj = p / "config.json"
    if cj.exists():
        c = json.loads(cj.read_text())
        a = c.get("argv", {})
        flds = ("d", "T", "steps", "seed", "beta_flux", "beta_flux_nl",
                "flux_floors", "ni_sigma", "ri_p", "anchor_p", "anchor_eps",
                "dp", "orbit", "rearc", "conceptarc", "batch", "lr", "tau")
        kv = " ".join(f"{k}={a.get(k)}" for k in flds if k in a)
        print(f"  argv: {kv}")
        print(f"  git={c.get('git')} n_params_bulk={c.get('n_params_bulk')} "
              f"n_tasks={c.get('n_tasks')} n_pairs={c.get('n_pairs')} "
              f"backend={c.get('backend')}")
    done = (p / ".done").exists()
    print(f"  .done={'yes' if done else 'no'}")


def main() -> int:
    for arg in sys.argv[1:]:
        p = Path(arg)
        if p.is_dir():
            show_dir(p)
        else:
            show_ckpt(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
