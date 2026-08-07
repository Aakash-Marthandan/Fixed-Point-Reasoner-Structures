# Ledger: 2026-08-07 harness — cross-bulk agreement-regularized population
# (C11 ⊕ [H-15] ⊕ [H-18]). Members = (bulk, view, seed); consensus
# pseudo-labels on query rows after warmup when --agree-lambda > 0.
# Saves per-member query predictions (the H-18 matrix rides every run).
"""
  python tools/eval_popx.py --ckpts a.pkl,b.pkl --tasks t1,t2 --out runs/popx
  arm X  : --agree-lambda 0
  arm XA : --agree-lambda 0.5 --agree-every 25 --agree-warmup 150
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


def load_bulk(path: str):
    saved = E.load_ckpt(path)
    defaults = Config()
    cfg = Config(**{k: type(getattr(defaults, k))(v)
                    for k, v in saved["config"].items()})
    name = Path(path).parent.name
    return {"name": name, "state": saved["state"], "cfg": cfg, "tau": 1.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", required=True, help="comma list of bulk ckpts")
    ap.add_argument("--tasks", default=None, help="comma list; default dev-30")
    ap.add_argument("--out", required=True)
    ap.add_argument("--views", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--val-every", type=int, default=50)
    ap.add_argument("--agree-lambda", type=float, default=0.0)
    ap.add_argument("--agree-every", type=int, default=25)
    ap.add_argument("--agree-warmup", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-member-preds", action="store_true",
                    help="drop member_query_preds from rows (size)")
    a = ap.parse_args()

    bulks = [load_bulk(p) for p in a.ckpts.split(",")]

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
            F = P.fit_population_cross(
                bulks, eps, n_views=a.views, n_seeds=a.seeds, steps=a.steps,
                val_every=a.val_every, agree_lambda=a.agree_lambda,
                agree_every=a.agree_every, agree_warmup=a.agree_warmup,
                seed=a.seed)
            res = P.score_population_cross(F, eps)
            if a.no_member_preds:
                res.pop("member_query_preds", None)
            res.update({"task": tid,
                        "family": dev30.MANIFEST.get(tid, ("?", ""))[0],
                        "bulks": [b["name"] for b in bulks],
                        "views": a.views, "seeds": a.seeds, "steps": a.steps,
                        "agree_lambda": a.agree_lambda,
                        "agree_every": a.agree_every,
                        "agree_warmup": a.agree_warmup,
                        "wall_s": round(time.time() - t0, 1)})
            f.write(json.dumps(res) + "\n")
            f.flush()
            n_mem = len(res["members"])
            print(f"{tid} pass2={res['solved_pass2']} joint2={res['solved_joint2']} "
                  f"qual={res['n_qualifiers']}/{n_mem} "
                  f"({res['wall_s']}s)", flush=True)
    print("POPX EVAL DONE", flush=True)


if __name__ == "__main__":
    main()
