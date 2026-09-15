"""COST PROBE (2026-09-10, ops rule 13a): time ONE task's evaluator rows on ONE chip before any battery starts, so the
chain projects the battery wall from a MEASURED cost instead of an assumed one (the DEC-ARC pilot's 100x miss).
Rows timed (the deployed code objects, not replicas): the arm-A fit step through eval_dev30._fit's own jitted step
(compile = the first step; s_per_step = the median of the rest), evaluate_pair (the validation predict; first call =
its compile), and probe_e1e3.trace at t_total (cold + one latent draw start for the equilibrium cells).
Usage: cost_probe.py --ckpt C --set S --out runs/cost_probe_<arm>.json [--task-idx 0] [--steps 8] [--fit-t N] [--t-total 16] [--ema]"""
import argparse, json, sys, time, dataclasses
from pathlib import Path
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import eval_dev30 as ED, probe_e1e3 as P
from qhrrn2 import grid as G
from qhrrn2.config import Config
from qhrrn2 import train as T
from qhrrn2 import episodic as E   # the checkpoint loader shared with eval_decarc (E.load_ckpt)


def task_ids_of(name):
    import eval_decarc as EA
    return EA.task_ids_of(name)


def load_task(tid):
    import eval_decarc as EA
    return EA.load_task(tid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--set", default="valhard"); ap.add_argument("--out", required=True)
    ap.add_argument("--task-idx", type=int, default=0); ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--fit-t", type=int, default=None); ap.add_argument("--t-total", type=int, default=16); ap.add_argument("--ema", action="store_true")
    a = ap.parse_args()
    saved = E.load_ckpt(a.ckpt); defaults = Config()
    cfg = Config(**{k: type(getattr(defaults, k))(v) for k, v in saved["config"].items() if hasattr(defaults, k)})
    state = saved["state_ema"] if (a.ema and saved.get("state_ema") is not None) else saved["state"]
    tid = task_ids_of(a.set)[a.task_idx]; eps = load_task(tid)
    support = list(eps[0].support); train_pairs, (val_x, val_y) = support[:-1], support[-1]
    x_b, y_b = T.pairs_to_batch(train_pairs, transforms=None, seed=0)
    cfg_fit = cfg if a.fit_t is None else dataclasses.replace(cfg, T=int(a.fit_t))
    model = jax.tree.map(jnp.asarray, state["model"]); tv0 = jnp.asarray(np.asarray(state["table"]).mean(0))
    trainable = {"model": model, "tv": tv0}
    step, make_opt = ED._arm_step(cfg_fit, "A", ED.ARM_LR["A"], 1e-4, 1.0, True)
    opt_state = make_opt(trainable).init(trainable)
    rng = jax.random.PRNGKey(0); walls = []
    for i in range(a.steps):
        rng, sub = jax.random.split(rng); ts = time.time()
        trainable, opt_state, loss, _ = step(trainable, opt_state, sub, x_b, y_b); float(loss)
        walls.append(time.time() - ts)
    ts = time.time(); T.evaluate_pair(trainable["model"], cfg, val_x, val_y, tau=1.0, task_vec=trainable["tv"]); val_first = time.time() - ts
    ts = time.time(); T.evaluate_pair(trainable["model"], cfg, val_x, val_y, tau=1.0, task_vec=trainable["tv"]); val_s = time.time() - ts
    tv = jnp.asarray(trainable["tv"]); qx = eps[0].query_x
    ts = time.time(); P.trace(trainable["model"], cfg, qx, tau=1.0, task_vec=tv, t_total=a.t_total); trace_first = time.time() - ts
    ts = time.time(); P.trace(trainable["model"], cfg, qx, tau=1.0, task_vec=tv, t_total=a.t_total); trace_s = time.time() - ts
    rec = {"ckpt": a.ckpt, "set": a.set, "task": tid, "cell_kind": cfg.cell_kind, "B": int(x_b.shape[0]), "n_support": len(support),
           "fit_T": cfg_fit.T, "cfg_T": cfg.T, "t_total": a.t_total, "steps": a.steps, "compile_s": round(walls[0], 2),
           "s_per_step": round(float(np.median(walls[1:])), 3) if len(walls) > 1 else round(walls[0], 3),
           "val_first_s": round(val_first, 2), "val_s": round(val_s, 3), "trace_first_s": round(trace_first, 2), "trace_s": round(trace_s, 3),
           "device": str(jax.devices()[0].platform)}
    if cfg.cell_kind == "decarc":
        # 2026-09-15: the battery's OWN trace (tools/eval_decarc.trace_dec, fused stats) timed on the chip — the pilot priced the
        # trace term from the probe trace and missed 8.5x — and cross-checked here, before any battery, against the eager path
        # and the probe trace (identical preds); the chain falls back to --trace-fused 0 when xcheck_fused_eager is false.
        import eval_decarc as EA
        same = lambda A, B: all(x["hw"] == y_["hw"] and np.array_equal(x["pred"], y_["pred"]) for x, y_ in zip(A, B))
        ts = time.time(); sf = EA.trace_dec(trainable["model"], cfg, qx, code=tv, t_total=a.t_total, fused=True); td_first = time.time() - ts
        ts = time.time(); sf = EA.trace_dec(trainable["model"], cfg, qx, code=tv, t_total=a.t_total, fused=True); td_s = time.time() - ts
        ts = time.time(); se = EA.trace_dec(trainable["model"], cfg, qx, code=tv, t_total=a.t_total, fused=False); te_s = time.time() - ts
        sp = P.trace(trainable["model"], cfg, qx, tau=1.0, task_vec=tv, t_total=a.t_total)
        rec.update({"trace_dec_first_s": round(td_first, 2), "trace_dec_s": round(td_s, 3), "trace_dec_eager_s": round(te_s, 3),
                    "xcheck_fused_eager": bool(same(sf, se)), "xcheck_fused_probe": bool(same(sf, sp)), "eval_start": cfg.decarc_eval_start})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(rec, indent=1)); print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
