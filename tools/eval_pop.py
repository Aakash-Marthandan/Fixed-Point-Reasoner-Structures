# Ledger: C11 harness (2026-08-06) — population TTT eval: M = views × seeds
# e_t members vs a frozen bulk, earliest-exact member selection, invert+vote.
"""
  python tools/eval_pop.py --ckpt <bulk> --tasks a,b,c --out runs/pop1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import dev30
from qhrrn2 import episodic as E
from qhrrn2 import grid as G
from qhrrn2 import population as P
from qhrrn2.config import Config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tasks", default=None, help="comma list; default dev-30")
    ap.add_argument("--out", required=True)
    ap.add_argument("--views", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--val-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    saved = E.load_ckpt(a.ckpt)
    defaults = Config()
    cfg = Config(**{k: type(getattr(defaults, k))(v)
                    for k, v in saved["config"].items()})
    state = saved["state"]

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
            eps = G.load_task(tid)
            F = P.fit_population(state, cfg, eps, n_views=a.views,
                                 n_seeds=a.seeds, steps=a.steps,
                                 val_every=a.val_every, seed=a.seed)
            res = P.score_population(F, eps)
            res.update({"task": tid,
                        "family": dev30.MANIFEST.get(tid, ("?", ""))[0],
                        "views": a.views, "seeds": a.seeds, "steps": a.steps,
                        "wall_s": round(time.time() - t0, 1)})
            f.write(json.dumps(res) + "\n")
            f.flush()
            print(f"{tid} {res['family']:<20} pass2={res['solved_pass2']} "
                  f"at1={res['solved_at1']} qual={res['n_qualifiers']}/{a.views*a.seeds} "
                  f"({res['wall_s']}s)", flush=True)
    print("POP EVAL DONE", flush=True)


if __name__ == "__main__":
    main()
