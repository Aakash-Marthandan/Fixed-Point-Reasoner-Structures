# Ledger: DEC-ARC BUILD B3 (Plan_2026-09-10_DEC-ARC_Build §1-§2; 2026-09-10). The DEC-ARC battery on one checkpoint:
# per task the deployed arm-A fit of the per-colour code (probe_e1e3.fit_arm_a -> eval_dev30._fit, the natives'
# protocol, never re-implemented), then per query: the COLD trace with per-step exactness (the depth row), the C1/C2/C3
# rows (arc_suite.dyn_record), the halting and commit heads on the carried state, k LATENT RESTARTS z0 ~ N(0, sigma)
# (the RI distribution of a cell with no answer register; selectors oracle / majority / residual-on-z, E5 spurious,
# AUCs; arc_suite.query_selectors), the FPA-START RETENTION LADDER (z_H := the embedded corrupted truth at eps rungs,
# eps = 0 = the handed-truth retention), the DIHEDRAL VOTE (the whole task re-fitted per D4 view, the inverted
# answers counted; pass@1 / pass@2 / any-view / distinct answers) and the COLOUR-FLIP row (the whole task under an S10
# palette, re-fitted; the deployed protocol's invariance against the fit-seed floor). Sets: val-hard, dev-30, rg-96,
# rt-48, the ARC-AGI-1 public evaluation split. Resume-safe per task; shardable; summary.json from every shard.
"""
  .venv/bin/python tools/eval_decarc.py --ckpt runs/decarc_D0/ckpt_020000.pkl --ema --set valhard --k 32 \
      --ladder 0,0.2,0.4,0.6,0.8 --views 8 --flip-test --out runs/decarc_D0/eval_valhard [--shard 0/8]
  .venv/bin/python tools/eval_decarc.py --out runs/decarc_D0/eval_valhard --summarize
"""
from __future__ import annotations
import argparse, json, sys, time, hashlib, zlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import jax, jax.numpy as jnp
from qhrrn2 import episodic as E
from qhrrn2 import grid as G
from qhrrn2 import model as M
from qhrrn2 import decarc_cell as DAC
from qhrrn2.config import Config
import probe_e1e3 as P
import arc_suite as AS

ROOT = Path(__file__).resolve().parents[1]


# ---------- sets ----------
def task_ids_of(name: str) -> list[str]:
    if name == "valhard":
        return json.load(open(Path(__file__).parent / "valhard.json"))["valhard"]
    if name == "dev30":
        import dev30; return sorted(dev30.MANIFEST)
    if name == "rg96":
        return sorted(p.stem for d in ("re_gate48", "re_gateb48") for p in (ROOT / "data" / d).glob("*.json"))
    if name == "rt48":
        return sorted(p.stem for p in (ROOT / "data" / "re_train48").glob("*.json"))
    if name == "arc1eval":
        return sorted(p.stem for p in (G.ARC_DATA_ROOT / "evaluation").glob("*.json"))
    raise ValueError(name)


def load_task(tid: str):
    try:
        return G.load_task(tid)
    except FileNotFoundError:
        p = G.ARC_DATA_ROOT / "evaluation" / f"{tid}.json"
        return G.load_task_file(p, tid)


# ---------- the trace with an explicit start (the DEC-ARC has no answer register: restarts and anchors live in z) ----------
def trace_dec(params, cfg: Config, x_grid, *, code, t_total: int, z_init=None, tau: float = 1.0):
    """probe_e1e3.trace's equilibrium path (P._traced_fwd_eq, M.decode_size) from an EXPLICIT initial carry z_init
    (2, F, S, w) or the cell's own start (None); per step: pred, hw, canvas, conf, res_z, q (halting logits), c (the
    commit probabilities when the head exists). Preds equal probe_e1e3.trace's for z_init = None (the same jitted map)."""
    assert cfg.cell_kind == "decarc"
    x_can = jnp.asarray(G.place(np.asarray(x_grid)), dtype=jnp.int32)
    y = jax.nn.one_hot(jnp.full((G.CANVAS, G.CANVAS), G.VOID, jnp.int32), M.VOCAB).transpose(2, 0, 1)
    eta, eta_z = M.eq_etas(params, cfg)
    p = params["decarc"]; has_c = "commit_head" in p
    z_c = None if z_init is None else jnp.asarray(z_init)
    steps = []
    fwd = P._traced_fwd_eq(cfg, tau, 1.0)     # the DEC-ARC ignores t_norm: one compiled map
    for t in range(t_total):
        out = fwd(params, x_can, y, code, z_c if z_c is not None else jnp.zeros(1))
        res_z = None if z_c is None else float(jnp.mean(jnp.abs(out.z_fine - z_c)))
        z_c = out.z_fine if z_c is None else z_c + eta_z * (out.z_fine - z_c)
        probs = jax.nn.softmax(out.logits, axis=-1); y = y + eta * (probs.transpose(2, 0, 1) - y)
        canvas = np.asarray(jnp.argmax(out.logits, axis=-1)); conf = np.asarray(jnp.max(probs, axis=-1))
        p_h, p_w = M.decode_size(cfg, out, x_can)
        h = int(jnp.argmax(p_h)) + 1; w = int(jnp.argmax(p_w)) + 1
        pred = np.where(canvas[:h, :w] == G.VOID, 0, canvas[:h, :w]).astype(np.int8)
        q = np.asarray(DAC.readout(p, cfg, z_c[0], (G.CANVAS, G.CANVAS))[1])
        rec = {"pred": pred, "hw": (h, w), "H_q": 0.0, "conf": conf[:h, :w].astype(np.float32), "canvas": canvas.astype(np.int8),
               "res_y": float(jnp.mean(jnp.abs(probs.transpose(2, 0, 1) - y))), "res_z": res_z, "q": (float(q[0]), float(q[1]))}
        if has_c:
            rec["c"] = np.asarray(jax.nn.sigmoid(DAC.commit_logits(p, z_c[0]))).reshape(G.CANVAS, G.CANVAS)[:h, :w].astype(np.float32)
        steps.append(rec)
    return steps


def z_draw(cfg: Config, key, sigma: float):
    return sigma * jax.random.normal(key, M.carry_shape(cfg))


def anchor_start(params, cfg: Config, y_grid, eps: float, key):
    """The FPA start: z_H := the embedded truth with eps of the OUTPUT cells resampled uniformly over the colours; z_L :=
    the fixed buffer. eps = 0 is the handed truth (retention); the rungs are the basin-radius ladder."""
    y_can = jnp.asarray(G.place(np.asarray(y_grid)), jnp.int32)
    if eps > 0:
        k1, k2 = jax.random.split(key)
        flip = (jax.random.uniform(k1, y_can.shape) < eps) & (y_can != G.VOID)
        y_can = jnp.where(flip, jax.random.randint(k2, y_can.shape, 0, 10), y_can)
    zH = DAC.embed_answer(params["decarc"], cfg, y_can)
    zL = DAC.z0(cfg, G.CANVAS * G.CANVAS)[1]
    return jnp.stack([zH, zL])


def ex(p, gt):
    return AS.ex(p, gt)


# ---------- one task ----------
def run_task(state, cfg, tid, a):
    eps = load_task(tid)
    t0 = time.time()
    model, _snaps, sel, _F = P.fit_arm_a(state, cfg, eps, steps=a.steps, val_every=a.val_every, seed=a.seed)
    code = jnp.asarray(sel[1]); assert code.shape == (DAC.F, cfg.d_task), code.shape
    rec = {"task": tid, "sel_step": sel[0], "fit_s": round(time.time() - t0, 1), "queries": []}
    for qi, ep in enumerate(eps):
        gt = ep.query_y
        if gt is None: continue
        st = trace_dec(model, cfg, ep.query_x, code=code, t_total=a.t_total)
        if qi == 0:   # the cross-check against the shared probe trace (identical preds from the cell's own start)
            ref = P.trace(model, cfg, ep.query_x, tau=1.0, task_vec=code, t_total=a.t_total)
            rec["xcheck"] = all(r["pred"].shape == s["pred"].shape and np.array_equal(r["pred"], s["pred"]) for r, s in zip(ref, st))
        q = {"q": qi, "dyn": AS.dyn_record(st, gt), "exact_T": ex(st[cfg.T - 1]["pred"], gt),
             "q_by_step": [s["q"][0] - s["q"][1] for s in st],            # q_halt - q_continue per step
             "exact_by_step": [ex(s["pred"], gt) for s in st]}
        if "c" in st[-1]:
            cl = st[-1]["c"]; pred = st[-1]["pred"]
            if pred.shape == gt.shape:
                wrong = pred != gt
                q["commit"] = {"mean_c": float(cl.mean()), "c_wrong": float(cl[wrong].mean()) if wrong.any() else None,
                               "c_right": float(cl[~wrong].mean()) if (~wrong).any() else None,
                               "auc_wrong": AS.auc([-float(v) for v in cl.reshape(-1)], [bool(v) for v in wrong.reshape(-1)]) if wrong.any() and (~wrong).any() else None}
        # k latent restarts
        if a.k > 0:
            rows = []
            key = jax.random.PRNGKey((a.seed * 100003 + qi * 101 + zlib.crc32(tid.encode())) % (2 ** 31))   # per (task, query, seed): nested k-curves
            for j in range(a.k):
                zj = z_draw(cfg, jax.random.fold_in(key, j), a.sigma)
                sj = trace_dec(model, cfg, ep.query_x, code=code, t_total=a.t_total, z_init=zj)
                preds = [s["pred"] for s in sj]
                conv = all(p_.shape == preds[-1].shape and np.array_equal(p_, preds[-1]) for p_ in preds[-3:])
                rz = [s["res_z"] for s in sj[-3:] if s["res_z"] is not None]
                rows.append({"exact": ex(preds[-1], gt), "converged": bool(conv), "res_y": float(np.mean([s["res_y"] for s in sj[-3:]])),
                             "res_z": float(np.mean(rz)) if rz else None, "H_q": 0.0, "conf": float(sj[-1]["conf"].mean()),
                             "hash": hashlib.md5(preds[-1].tobytes() + bytes(preds[-1].shape)).hexdigest()[:12]})
            q["draws"] = rows; q["sel"] = AS.query_selectors(rows)
        # the FPA-start retention ladder
        if a.ladder:
            lad = {}
            for e_ in a.ladder:
                hits = []
                for r_ in range(a.ladder_draws if e_ > 0 else 1):
                    z_a = anchor_start(model, cfg, gt, e_, jax.random.PRNGKey(a.seed * 7 + qi * 131 + r_ + int(e_ * 1000)))
                    sa = trace_dec(model, cfg, ep.query_x, code=code, t_total=a.ladder_steps, z_init=z_a)
                    hits.append(ex(sa[-1]["pred"], gt))
                lad[str(e_)] = float(np.mean(hits))
            q["ladder"] = lad
        rec["queries"].append(q)
    # the dihedral vote: the whole task re-fitted per D4 view, answers inverted and counted
    if a.views > 1:
        per_q = {}
        for k in range(a.views):
            t = G.Transform(k=k)
            teps = [G.transform_episode(e, t) for e in eps]
            mk, _s, sk, _ = P.fit_arm_a(state, cfg, teps, steps=a.steps, val_every=a.val_every, seed=a.seed) if k > 0 else (model, None, sel, None)
            ck = jnp.asarray(sk[1])
            for qi, tep in enumerate(teps):
                if tep.query_y is None: continue
                sv = trace_dec(mk, cfg, tep.query_x, code=ck, t_total=a.t_total)
                inv = t.invert_output(sv[cfg.T - 1]["pred"])
                per_q.setdefault(qi, []).append(inv)
        for q in rec["queries"]:
            gt = eps[q["q"]].query_y; cands = per_q[q["q"]]
            keys_ = [hashlib.md5(c.tobytes() + bytes(c.shape)).hexdigest() for c in cands]
            order = []
            for kk in keys_:
                if kk not in order: order.append(kk)
            counts = sorted(((keys_.count(kk), -order.index(kk), kk) for kk in order), reverse=True)
            top = [cands[keys_.index(kk)] for _, _, kk in counts[:2]]
            q["vote"] = {"n_views": len(cands), "any_view": any(ex(c, gt) for c in cands), "pass1": ex(top[0], gt),
                         "pass2": any(ex(c, gt) for c in top), "n_distinct": len(order), "top_count": counts[0][0],
                         "correct_views": float(np.mean([ex(c, gt) for c in cands]))}
    # the colour-flip row: the whole task under one S10 palette, re-fitted; the floor = the identity re-fitted with seed + 1
    if a.flip_test:
        rng = np.random.default_rng(int(hashlib.md5(tid.encode()).hexdigest()[:8], 16) + a.seed)
        lut = np.arange(M.VOCAB, dtype=np.int8); lut[:10] = rng.permutation(10).astype(np.int8)
        tp = G.Transform(k=0, lut=lut); teps = [G.transform_episode(e, tp) for e in eps]
        mp, _s, sp, _ = P.fit_arm_a(state, cfg, teps, steps=a.steps, val_every=a.val_every, seed=a.seed)
        mf, _s, sf, _ = P.fit_arm_a(state, cfg, eps, steps=a.steps, val_every=a.val_every, seed=a.seed + 1)
        for q in rec["queries"]:
            qi = q["q"]
            sv = trace_dec(mp, cfg, teps[qi].query_x, code=jnp.asarray(sp[1]), t_total=a.t_total)
            sfl = trace_dec(mf, cfg, eps[qi].query_x, code=jnp.asarray(sf[1]), t_total=a.t_total)
            q["flip"] = {"exact_perm": ex(sv[cfg.T - 1]["pred"], teps[qi].query_y), "exact_refit": ex(sfl[cfg.T - 1]["pred"], eps[qi].query_y)}
    rec["wall_s"] = round(time.time() - t0, 1)
    return rec


# ---------- summary ----------
def summarize(out: Path, a):
    R = []
    for f in sorted(out.glob("results*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip(): R.append(json.loads(line))
    Q = [q for r in R for q in r["queries"]]; D = [q["dyn"] for q in Q]
    mean = lambda xs: float(np.mean([x for x in xs if x is not None])) if any(x is not None for x in xs) else None
    T = len(Q[0]["exact_by_step"]) if Q else 0
    S = {"ckpt": getattr(a, "ckpt", None), "ema": bool(getattr(a, "ema", False)), "set": getattr(a, "set", None), "shards": len(list(out.glob("results*.jsonl"))),
         "n_tasks": len(R), "n_queries": len(Q), "xcheck_all": all(r.get("xcheck", True) for r in R),
         "clean_exact_limit": mean([d["limit_exact"] for d in D]), "clean_exact_T": mean([q["exact_T"] for q in Q]),
         "exact_by_step": [mean([q["exact_by_step"][t] for q in Q]) for t in range(T)],
         "ever_exact": mean([any(q["exact_by_step"]) for q in Q]),
         "lost": int(sum(any(q["exact_by_step"]) and not q["exact_by_step"][-1] for q in Q)),
         "first_exact_med": float(np.median([d["first_exact"] for d in D if d["first_exact"] is not None])) if any(d["first_exact"] is not None for d in D) else None,
         "converged_frac": mean([d["converged_at"] is not None for d in D]),
         "converged_wrong_of_converged": mean([not d["limit_exact"] for d in D if d["converged_at"] is not None]),
         "size1_ok": mean([d["size1_ok"] for d in D]), "size_last_ok": mean([d["size_last_ok"] for d in D]),
         "commit1": mean([d["commit1"] for d in D]), "commit_last": mean([d["commit_last"] for d in D]), "conf1": mean([d["conf1"] for d in D]),
         "flips_per_cell": mean([d["flips_per_cell"] for d in D]),
         "fail_cell_acc_last": mean([d.get("cell_acc_last") for d in D if not d["limit_exact"]]),
         "fail_wrong_conf_last": mean([d.get("wrong_conf_last") for d in D if not d["limit_exact"]]),
         "fail_right_conf_last": mean([d.get("right_conf_last") for d in D if not d["limit_exact"]]),
         "fail_committed_wrong_last": mean([d.get("committed_wrong_last") for d in D if not d["limit_exact"]]),
         "halt_auc": AS.auc([q["q_by_step"][-1] for q in Q], [q["exact_by_step"][-1] for q in Q]) if Q else None,
         "halt_frac_last": mean([q["q_by_step"][-1] > 0 for q in Q]),
         "fit_s_mean": mean([r["fit_s"] for r in R]), "wall_s_mean": mean([r["wall_s"] for r in R])}
    if Q and "commit" in Q[0]:
        C = [q["commit"] for q in Q if "commit" in q]
        S["commit_head"] = {"mean_c_solved": mean([c["mean_c"] for c, q in zip(C, Q) if q["exact_T"]]),
                            "mean_c_failed": mean([c["mean_c"] for c, q in zip(C, Q) if not q["exact_T"]]),
                            "c_wrong": mean([c["c_wrong"] for c in C]), "c_right": mean([c["c_right"] for c in C]),
                            "auc_wrong_cell": mean([c["auc_wrong"] for c in C]),
                            "auc_instance": AS.auc([c["mean_c"] for c in C], [q["exact_T"] for q in Q if "commit" in q])}
    if Q and "sel" in Q[0]:
        sel = [q["sel"] for q in Q]; dr = [d for q in Q for d in q["draws"]]
        S.update({"draw_exact_rate": mean([s["draw_exact_rate"] for s in sel]), "oracle": mean([s["oracle"] for s in sel]),
                  "majority": mean([s["majority"] for s in sel]), "t1r_y": mean([s["t1r_y"] for s in sel]), "t1r_z": mean([s["t1r_z"] for s in sel]),
                  "spurious_rate": mean([s["spurious_rate"] for s in sel]), "draw_converged_rate": mean([s["converged_rate"] for s in sel]),
                  "n_distinct": mean([len({d["hash"] for d in q["draws"]}) for q in Q]),
                  "auc_res_z": AS.auc([-(d["res_z"] if d["res_z"] is not None else 1e9) for d in dr], [d["exact"] for d in dr]),
                  "auc_res_y": AS.auc([-d["res_y"] for d in dr], [d["exact"] for d in dr]),
                  "oracle_only_queries": int(sum(1 for q in Q if q["sel"]["oracle"] and not q["dyn"]["limit_exact"]))})
        # E5 spurious: wrong draws whose z-residual is below the correct draws' median
        cor = [d["res_z"] for d in dr if d["exact"] and d["res_z"] is not None]
        if cor:
            med = float(np.median(cor)); wrong = [d for d in dr if not d["exact"] and d["res_z"] is not None]
            S["spurious_e5_z"] = mean([d["res_z"] < med for d in wrong]) if wrong else None
    if Q and "ladder" in Q[0]:
        rungs = sorted({e for q in Q for e in q["ladder"]}, key=float)
        S["ladder"] = {e: mean([q["ladder"].get(e) for q in Q]) for e in rungs}
        S["retention_gt"] = S["ladder"].get("0.0", S["ladder"].get("0"))
        sol = [q for q in Q if q["exact_T"]]; uns = [q for q in Q if not q["exact_T"]]
        S["retention_gt_solved"] = mean([q["ladder"].get("0.0", q["ladder"].get("0")) for q in sol]) if sol else None
        S["retention_gt_failed"] = mean([q["ladder"].get("0.0", q["ladder"].get("0")) for q in uns]) if uns else None
        if "sel" in Q[0]:
            ret = lambda q: q["ladder"].get("0.0", q["ladder"].get("0"))
            S["oracle_given_retains"] = mean([q["sel"]["oracle"] for q in Q if ret(q)])
            S["oracle_given_not"] = mean([q["sel"]["oracle"] for q in Q if not ret(q)])
    if Q and "vote" in Q[0]:
        V = [q["vote"] for q in Q]
        S["vote"] = {"pass1": mean([v["pass1"] for v in V]), "pass2": mean([v["pass2"] for v in V]), "any_view": mean([v["any_view"] for v in V]),
                     "n_distinct_solved": float(np.median([v["n_distinct"] for v in V if v["pass1"]])) if any(v["pass1"] for v in V) else None,
                     "n_distinct_failed": float(np.median([v["n_distinct"] for v in V if not v["pass1"]])) if any(not v["pass1"] for v in V) else None,
                     "correct_views_solved": mean([v["correct_views"] for v in V if v["pass1"]]), "correct_views_failed": mean([v["correct_views"] for v in V if not v["pass1"]])}
    if Q and "flip" in Q[0]:
        fl = [(q["exact_T"], q["flip"]["exact_perm"], q["flip"]["exact_refit"]) for q in Q]
        S["flip"] = {"exact_identity": mean([e for e, _, _ in fl]), "exact_perm": mean([p for _, p, _ in fl]), "exact_refit": mean([r for _, _, r in fl]),
                     "flip_rate_colour": mean([e != p for e, p, _ in fl]), "flip_rate_floor": mean([e != r for e, _, r in fl])}
    (out / "summary.json").write_text(json.dumps(S, indent=1))
    print(json.dumps(S, indent=1))
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt"); ap.add_argument("--out", required=True); ap.add_argument("--ema", action="store_true")
    ap.add_argument("--set", default="valhard", choices=["valhard", "dev30", "rg96", "rt48", "arc1eval"]); ap.add_argument("--tasks", default=None)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--shard", default=None, help="i/n: this process's task slice")
    ap.add_argument("--steps", type=int, default=600); ap.add_argument("--val-every", type=int, default=50)
    ap.add_argument("--t-total", type=int, default=16); ap.add_argument("--k", type=int, default=32); ap.add_argument("--sigma", type=float, default=None)
    ap.add_argument("--ladder", default="0,0.2,0.4,0.6,0.8"); ap.add_argument("--ladder-draws", type=int, default=2); ap.add_argument("--ladder-steps", type=int, default=8)
    ap.add_argument("--views", type=int, default=8); ap.add_argument("--flip-test", action="store_true")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    a.ladder = [float(v) for v in a.ladder.split(",")] if a.ladder else []
    if a.summarize:
        provs = sorted(out.glob("provenance*.json"))
        if provs:
            pv = json.load(open(provs[0])); a.ckpt = pv.get("ckpt"); a.ema = pv.get("ema"); a.set = pv.get("set")
            assert all(json.load(open(q)).get("ckpt") == a.ckpt for q in provs), "shards evaluated different checkpoints"
        return summarize(out, a)
    saved = E.load_ckpt(a.ckpt); defaults = Config()
    cfg = Config(**{k: type(getattr(defaults, k))(v) for k, v in saved["config"].items() if hasattr(defaults, k)})
    assert cfg.cell_kind == "decarc", cfg.cell_kind
    state = saved["state_ema"] if (a.ema and saved.get("state_ema") is not None) else saved["state"]
    if a.sigma is None: a.sigma = cfg.trm_ri_sigma if cfg.trm_ri_sigma > 0 else 1.0
    ids = a.tasks.split(",") if a.tasks else task_ids_of(a.set)
    if a.limit: ids = ids[:a.limit]
    tag = ""
    if a.shard:
        i, n = (int(v) for v in a.shard.split("/")); ids = ids[i::n]; tag = f"_{i}"
    results = out / f"results{tag}.jsonl"; done = set()
    if results.exists():
        for line in results.read_text().splitlines():
            try: done.add(json.loads(line)["task"])
            except Exception: pass
    (out / f"provenance{tag}.json").write_text(json.dumps({"ckpt": a.ckpt, "ema": a.ema, "set": a.set, "k": a.k, "views": a.views, "ladder": a.ladder, "steps": a.steps, "t_total": a.t_total, "seed": a.seed, "sigma": a.sigma}))
    with open(results, "a") as f:
        for tid in ids:
            if tid in done: print(f"skip {tid}", flush=True); continue
            rec = run_task(state, cfg, tid, a)
            f.write(json.dumps(rec) + "\n"); f.flush()
            nq = len(rec["queries"])
            print(f"{tid} sel@{rec['sel_step']} q={nq} exact={sum(q['exact_T'] for q in rec['queries'])} "
                  f"oracle={sum(q.get('sel', {}).get('oracle', 0) for q in rec['queries'])} xcheck={rec.get('xcheck')} {rec['wall_s']}s", flush=True)
    if not a.shard:
        summarize(out, a)


if __name__ == "__main__":
    main()
