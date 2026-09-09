# Ledger: ARC INSTRUMENTS on our JAX substrates (2026-09-08; the PI: "start the two Mac items") — the selector law
# (D3/D9: random-start draws, residual selection, converged-wrong rate), the C1/C2 dynamics + C3 calibration rows,
# and retention (E3b) — measured in the SAME units as the Sudoku suite, on the pretrain-13 substrates (RI vs plain).
# Instruments ONLY. The deployed arm-A fit and the trace are IMPORTED from probe_e1e3 (the 2026-08-08 lesson: a local
# replication silently diverged); the confidence-carrying trace below mirrors probe_e1e3.trace's equilibrium path and
# is CROSS-CHECKED against it (identical preds on every task's first query; the count is in summary.json).
# Random starts = the training-time RI distribution itself (episodic.build_y0_rows: a FULL uniform random color canvas).
"""
  .venv/bin/python tools/arc_suite.py --ckpt runs/pretrain13_Dri/ckpt_053333.pkl --set valhard --k 32 \
      --out runs/arcsuite_p13Dri
Appends one JSON line per task to <out>/results.jsonl (resume-safe), writes <out>/summary.json at the end."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import jax, jax.numpy as jnp
from qhrrn2 import episodic as E
from qhrrn2 import grid as G
from qhrrn2 import model as M
from qhrrn2.config import Config
import probe_e1e3 as P

CONF_T = 0.9   # the suite's commitment threshold (C1/C2 rows use conf > .9)


def trace_conf(params, cfg: Config, x_grid, *, task_vec, t_total: int, y0_canvas=None, tau: float = 1.0):
    """probe_e1e3.trace's equilibrium path + per-step confidence and residuals.
    y0_canvas: an int (32,32) start canvas (None = VOID; a random canvas = an RI draw); the trained ramp."""
    assert cfg.equilibrium and not cfg.use_obj
    x_can = jnp.asarray(G.place(np.asarray(x_grid)), dtype=jnp.int32)
    y0 = jnp.full((G.CANVAS, G.CANVAS), G.VOID, jnp.int32) if y0_canvas is None else jnp.asarray(y0_canvas, jnp.int32)
    y = jax.nn.one_hot(y0, M.VOCAB).transpose(2, 0, 1)
    eta = jax.nn.sigmoid(params["eq"]["eta"]); eta_z = jax.nn.sigmoid(params["eq"]["eta_z"])
    z_c = None; steps = []
    for t in range(t_total):
        t_norm = min(t, cfg.T - 1) / max(cfg.T - 1, 1)
        out = P._traced_fwd_eq(cfg, tau, float(t_norm))(params, x_can, y, task_vec, z_c if z_c is not None else jnp.zeros(1))
        res_z = None if z_c is None else float(jnp.mean(jnp.abs(out.z_fine - z_c)))
        z_c = out.z_fine if z_c is None else z_c + eta_z * (out.z_fine - z_c)
        probs = jax.nn.softmax(out.logits, axis=-1)
        pcan = probs.transpose(2, 0, 1)
        res_y = float(jnp.mean(jnp.abs(pcan - y)))
        y = y + eta * (pcan - y)
        canvas = np.asarray(jnp.argmax(out.logits, axis=-1)); conf = np.asarray(jnp.max(probs, axis=-1))
        cands = M.size_candidates(x_can)
        h = int(jnp.argmax(M.size_mixture_probs(out.size_sel_h, out.size_h, cands[0]))) + 1
        w = int(jnp.argmax(M.size_mixture_probs(out.size_sel_w, out.size_w, cands[1]))) + 1
        pred = np.where(canvas[:h, :w] == G.VOID, 0, canvas[:h, :w]).astype(np.int8)
        steps.append({"pred": pred, "hw": (h, w), "H_q": P.entropy(out.rule_q), "conf": conf[:h, :w].astype(np.float32),
                      "canvas": canvas.astype(np.int8), "res_y": res_y, "res_z": res_z})
    return steps


def ex(p, gt):
    return bool(gt is not None and p.shape == gt.shape and np.array_equal(p, gt))


def dyn_record(steps, gt):
    """C1/C2 dynamics + C3 calibration from one clean-start trace (cells compared only where the size is right)."""
    st = P.stability_record(steps, gt)
    s0, sl = steps[0], steps[-1]
    rec = {"converged_at": st["converged_at"], "n_distinct": st["n_distinct"], "exact_by_step": st["exact_per_step"],
           "limit_exact": st["limit_exact"], "first_exact": next((i for i, e in enumerate(st["exact_per_step"]) if e), None),
           "H_q_first": round(s0["H_q"], 4), "H_q_last": round(sl["H_q"], 4),
           "conf1": float(s0["conf"].mean()), "commit1": float((s0["conf"] > CONF_T).mean()),
           "conf_last": float(sl["conf"].mean()), "commit_last": float((sl["conf"] > CONF_T).mean()),
           "res_y_last": sl["res_y"], "res_z_last": sl["res_z"], "size1_ok": bool(s0["pred"].shape == gt.shape),
           "size_last_ok": bool(sl["pred"].shape == gt.shape)}
    cv = np.stack([s["canvas"] for s in steps]); rec["flips_per_cell"] = float((cv[1:] != cv[:-1]).sum(0).mean())
    for tag, s in (("1", s0), ("last", sl)):
        if s["pred"].shape == gt.shape:
            wrong = s["pred"] != gt; comm = s["conf"] > CONF_T
            rec[f"cell_acc_{tag}"] = float(1 - wrong.mean())
            rec[f"wrong_conf_{tag}"] = float(s["conf"][wrong].mean()) if wrong.any() else None
            rec[f"right_conf_{tag}"] = float(s["conf"][~wrong].mean()) if (~wrong).any() else None
            rec[f"committed_wrong_{tag}"] = float(wrong[comm].mean()) if comm.any() else None   # of committed cells, wrong
    return rec


def draw_records(params, cfg, ep, tv, *, k: int, t_total: int, seed: int, gt):
    """k random-start draws (the RI distribution: full uniform random color canvases)."""
    rng = np.random.default_rng(seed)
    rows = []
    for j in range(k):
        y0 = rng.integers(0, 10, size=(G.CANVAS, G.CANVAS), dtype=np.int32)
        steps = trace_conf(params, cfg, ep.query_x, task_vec=tv, t_total=t_total, y0_canvas=y0)
        preds = [s["pred"] for s in steps]
        conv = all(p.shape == preds[-1].shape and np.array_equal(p, preds[-1]) for p in preds[-3:])
        rz = [s["res_z"] for s in steps[-3:] if s["res_z"] is not None]
        rows.append({"exact": ex(preds[-1], gt), "converged": bool(conv), "res_y": float(np.mean([s["res_y"] for s in steps[-3:]])),
                     "res_z": float(np.mean(rz)) if rz else None, "H_q": round(steps[-1]["H_q"], 4),
                     "conf": float(steps[-1]["conf"].mean()), "hash": __import__("hashlib").md5(preds[-1].tobytes() + bytes(preds[-1].shape)).hexdigest()[:12]})
    return rows


def query_selectors(rows):
    from collections import Counter
    n = len(rows); exact = [r["exact"] for r in rows]
    maj = Counter(r["hash"] for r in rows).most_common(1)[0][0]
    maj_exact = next(r["exact"] for r in rows if r["hash"] == maj)
    t1y = min(range(n), key=lambda i: rows[i]["res_y"]); t1z = min(range(n), key=lambda i: (rows[i]["res_z"] is None, rows[i]["res_z"]))
    return {"oracle": any(exact), "majority": bool(maj_exact), "t1r_y": rows[t1y]["exact"], "t1r_z": rows[t1z]["exact"],
            "draw_exact_rate": float(np.mean(exact)), "converged_rate": float(np.mean([r["converged"] for r in rows])),
            "spurious_rate": float(np.mean([r["converged"] and not r["exact"] for r in rows])),
            "n_distinct": len({r["hash"] for r in rows})}


def auc(scores, labels):
    """AUC of (−score) as a detector of label=True: P(score_pos < score_neg) with ties at .5."""
    pos = [s for s, l in zip(scores, labels) if l and s is not None]; neg = [s for s, l in zip(scores, labels) if not l and s is not None]
    if not pos or not neg: return None
    pos = np.asarray(pos); neg = np.asarray(neg)
    return float(np.mean((pos[:, None] < neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--set", default="valhard", choices=["valhard", "dev30"]); ap.add_argument("--tasks", default=None)
    ap.add_argument("--steps", type=int, default=600); ap.add_argument("--val-every", type=int, default=50)
    ap.add_argument("--t-total", type=int, default=16); ap.add_argument("--stab-steps", type=int, default=8)
    ap.add_argument("--k", type=int, default=32); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    saved = E.load_ckpt(a.ckpt); defaults = Config()
    cfg = Config(**{k: type(getattr(defaults, k))(v) for k, v in saved["config"].items()}); state = saved["state"]
    if a.tasks: task_ids = a.tasks.split(",")
    elif a.set == "valhard": task_ids = json.load(open(Path(__file__).parent / "valhard.json"))["valhard"]
    else:
        import dev30; task_ids = sorted(dev30.MANIFEST)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); results = out / "results.jsonl"
    done = set()
    if results.exists():
        for line in results.read_text().splitlines():
            try: done.add(json.loads(line)["task"])
            except Exception: pass
    with open(results, "a") as f:
        for ti, tid in enumerate(task_ids):
            if tid in done: print(f"skip {tid}", flush=True); continue
            t0 = time.time(); eps = G.load_task(tid)
            model, snaps, sel, F = P.fit_arm_a(state, cfg, eps, steps=a.steps, val_every=a.val_every, seed=a.seed)
            tv = jnp.asarray(sel[1]); qrecs = []; xcheck = None
            for qi, ep in enumerate(eps):
                gt = ep.query_y
                if gt is None: continue
                steps = trace_conf(model, cfg, ep.query_x, task_vec=tv, t_total=a.t_total)
                if qi == 0:   # the cross-check against the shared trace (bit-identical preds)
                    ref = P.trace(model, cfg, ep.query_x, tau=1.0, task_vec=tv, t_total=a.t_total)
                    xcheck = all(r["pred"].shape == s["pred"].shape and np.array_equal(r["pred"], s["pred"]) for r, s in zip(ref, steps))
                rec = {"q": qi, "dyn": dyn_record(steps, gt), "exact_T": ex(steps[cfg.T - 1]["pred"], gt)}
                own = steps[cfg.T - 1]["pred"]; ret = {}
                for name, init in (("own", own), ("gt", gt)):
                    st = P.trace(model, cfg, ep.query_x, tau=1.0, task_vec=tv, t_total=a.stab_steps,
                                 yprev_init=G.place(np.asarray(init)), skip_trained=True)
                    ret[name] = all(ex(s["pred"], np.asarray(init)) for s in st)
                rec["retain"] = ret
                if a.k > 0:
                    rows = draw_records(model, cfg, ep, tv, k=a.k, t_total=a.t_total, seed=a.seed * 100003 + ti * 101 + qi, gt=gt)
                    rec["draws"] = rows; rec["sel"] = query_selectors(rows)
                qrecs.append(rec)
            f.write(json.dumps({"task": tid, "sel_step": sel[0], "xcheck": xcheck, "queries": qrecs,
                                "wall_s": round(time.time() - t0, 1)}) + "\n"); f.flush()
            print(f"{tid} sel@{sel[0]} q={len(qrecs)} clean_exact={sum(r['dyn']['limit_exact'] for r in qrecs)} "
                  f"oracle={sum(r.get('sel', {}).get('oracle', 0) for r in qrecs)} xcheck={xcheck} {time.time()-t0:.0f}s", flush=True)
    summarize(out, a)


def summarize(out: Path, a):
    rows = [json.loads(l) for l in (out / "results.jsonl").read_text().splitlines()]
    Q = [q for r in rows for q in r["queries"]]; D = [q["dyn"] for q in Q]
    def mean(xs):
        xs = [x for x in xs if x is not None]; return float(np.mean(xs)) if xs else None
    fails = [d for d in D if not d["limit_exact"]]
    S = {"ckpt": a.ckpt, "set": a.set, "n_tasks": len(rows), "n_queries": len(Q), "t_total": a.t_total, "k": a.k,
         "xcheck_ok": sum(1 for r in rows if r["xcheck"]), "xcheck_n": sum(1 for r in rows if r["xcheck"] is not None),
         "clean_exact_T": mean([q["exact_T"] for q in Q]), "clean_exact_limit": mean([d["limit_exact"] for d in D]),
         "converged_frac": mean([d["converged_at"] is not None for d in D]),
         "converged_wrong_of_converged": mean([not d["limit_exact"] for d in D if d["converged_at"] is not None]),
         "first_exact_median_solved": (float(np.median([d["first_exact"] for d in D if d["first_exact"] is not None]))
                                       if any(d["first_exact"] is not None for d in D) else None),
         "commit1": mean([d["commit1"] for d in D]), "conf1": mean([d["conf1"] for d in D]),
         "commit_last": mean([d["commit_last"] for d in D]), "flips_per_cell": mean([d["flips_per_cell"] for d in D]),
         "size1_ok": mean([d["size1_ok"] for d in D]), "size_last_ok": mean([d["size_last_ok"] for d in D]),
         "fail_wrong_conf_1": mean([d.get("wrong_conf_1") for d in fails]), "fail_committed_wrong_1": mean([d.get("committed_wrong_1") for d in fails]),
         "fail_wrong_conf_last": mean([d.get("wrong_conf_last") for d in fails]), "fail_committed_wrong_last": mean([d.get("committed_wrong_last") for d in fails]),
         "fail_right_conf_last": mean([d.get("right_conf_last") for d in fails]), "fail_cell_acc_last": mean([d.get("cell_acc_last") for d in fails]),
         "H_q_first": mean([d["H_q_first"] for d in D]), "H_q_last": mean([d["H_q_last"] for d in D]),
         "retain_gt": mean([q["retain"]["gt"] for q in Q]), "retain_own": mean([q["retain"]["own"] for q in Q]),
         "retain_gt_on_failures": mean([q["retain"]["gt"] for q in Q if not q["dyn"]["limit_exact"]])}
    if a.k > 0 and all("sel" in q for q in Q):
        sel = [q["sel"] for q in Q]; dr = [d for q in Q for d in q["draws"]]
        S.update({"draw_exact_rate": mean([s["draw_exact_rate"] for s in sel]), "oracle": mean([s["oracle"] for s in sel]),
                  "majority": mean([s["majority"] for s in sel]), "t1r_y": mean([s["t1r_y"] for s in sel]), "t1r_z": mean([s["t1r_z"] for s in sel]),
                  "spurious_rate": mean([s["spurious_rate"] for s in sel]), "draw_converged_rate": mean([s["converged_rate"] for s in sel]),
                  "n_distinct_mean": mean([s["n_distinct"] for s in sel]),
                  "auc_res_y": auc([d["res_y"] for d in dr], [d["exact"] for d in dr]), "auc_res_z": auc([d["res_z"] for d in dr], [d["exact"] for d in dr]),
                  "auc_Hq": auc([d["H_q"] for d in dr], [d["exact"] for d in dr]),
                  "oracle_only_queries": sum(1 for q in Q if q["sel"]["oracle"] and not q["dyn"]["limit_exact"]),
                  "spurious_on_basin_queries": mean([s["spurious_rate"] for s, q in zip(sel, Q) if s["oracle"]]),
                  "spurious_on_nobasin_queries": mean([s["spurious_rate"] for s, q in zip(sel, Q) if not s["oracle"]])})
    (out / "summary.json").write_text(json.dumps(S, indent=1))
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
