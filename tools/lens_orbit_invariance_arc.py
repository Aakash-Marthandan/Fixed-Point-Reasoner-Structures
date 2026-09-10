"""G1 (Plan_2026-09-10_DEC-ARC_Build §3): the orbit-invariance instrument's ARC twin on a native substrate.
For every task, the WHOLE task (supports + queries) is transformed by a joint symmetry g (an S9 colour permutation
fixing black, a D4 element, or both), the deployed arm-A fit is re-run on the transformed supports, the transformed
query is traced, and exactness is read against the transformed truth. Invariance = the solve status does not
depend on g; the floor = the identity task re-fitted with a second fit seed (the deployed protocol's own variance).
Descriptive; no rules. Mirrors tools/arc_suite.py (fit_arm_a / trace imported from probe_e1e3, never re-implemented).

  .venv/bin/python tools/lens_orbit_invariance_arc.py --ckpt runs/pretrain13_Dri/ckpt_053333.pkl --set valhard \
      --out runs/orbit_arc_p13Dri_valhard [--transforms g0,g0b,g1,g2,g3,g4] [--limit N]
Appends one JSON line per task to <out>/results.jsonl (resume-safe); summary.txt at the end (or via --summarize)."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import jax.numpy as jnp
from qhrrn2 import episodic as E
from qhrrn2 import grid as G
from qhrrn2.config import Config
import probe_e1e3 as P
from qhrrn2 import model as M
import jax


def trace_z(params, cfg, x_grid, *, task_vec, t_total: int, tau: float = 1.0):
    """probe_e1e3.trace's equilibrium path (mirrored as in tools/arc_suite.trace_conf) + the carried latent z_c per step
    and the per-cell confidence — the features of the readout fit (G2). Cross-checked against P.trace on every task."""
    assert cfg.equilibrium and not cfg.use_obj
    x_can = jnp.asarray(G.place(np.asarray(x_grid)), dtype=jnp.int32)
    y = jax.nn.one_hot(jnp.full((G.CANVAS, G.CANVAS), G.VOID, jnp.int32), M.VOCAB).transpose(2, 0, 1)
    eta = jax.nn.sigmoid(params["eq"]["eta"]); eta_z = jax.nn.sigmoid(params["eq"]["eta_z"])
    z_c = None; steps = []
    for t in range(t_total):
        t_norm = min(t, cfg.T - 1) / max(cfg.T - 1, 1)
        out = P._traced_fwd_eq(cfg, tau, float(t_norm))(params, x_can, y, task_vec, z_c if z_c is not None else jnp.zeros(1))
        z_c = out.z_fine if z_c is None else z_c + eta_z * (out.z_fine - z_c)
        probs = jax.nn.softmax(out.logits, axis=-1); y = y + eta * (probs.transpose(2, 0, 1) - y)
        canvas = np.asarray(jnp.argmax(out.logits, axis=-1)); conf = np.asarray(jnp.max(probs, axis=-1))
        cands = M.size_candidates(x_can)
        h = int(jnp.argmax(M.size_mixture_probs(out.size_sel_h, out.size_h, cands[0]))) + 1
        w = int(jnp.argmax(M.size_mixture_probs(out.size_sel_w, out.size_w, cands[1]))) + 1
        pred = np.where(canvas[:h, :w] == G.VOID, 0, canvas[:h, :w]).astype(np.int8)
        steps.append({"pred": pred, "hw": (h, w), "canvas": canvas.astype(np.int8), "conf": conf.astype(np.float32), "z": np.asarray(z_c)})
    return steps

TRANSFORMS = ["g0", "g0b", "g1", "g2", "g3", "g4"]   # identity | identity + fit seed 1 | palette | rot90 | fliplr | palette x (rot90+flip)


def make_transform(name: str, rng: np.random.Generator) -> tuple[G.Transform, int]:
    """(transform, fit seed)."""
    if name == "g0": return G.Transform(k=0), 0
    if name == "g0b": return G.Transform(k=0), 1
    if name == "g1": return G.Transform(k=0, lut=G.random_palette(rng)), 0
    if name == "g2": return G.Transform(k=1), 0
    if name == "g3": return G.Transform(k=4), 0
    if name == "g4": return G.Transform(k=5, lut=G.random_palette(rng)), 0
    raise ValueError(name)


def ex(p, gt):
    return bool(gt is not None and p.shape == gt.shape and np.array_equal(p, gt))


def run_task(state, cfg, tid, names, a):
    eps = G.load_task(tid)
    rng = np.random.default_rng(hash((tid, a.seed)) % (2**32))
    rec = {"task": tid, "n_queries": sum(e.query_y is not None for e in eps), "g": {}}
    for name in names:
        t, fit_seed = make_transform(name, rng)
        teps = [G.transform_episode(e, t) for e in eps]
        t0 = time.time()
        model, _snaps, sel, _F = P.fit_arm_a(state, cfg, teps, steps=a.steps, val_every=a.val_every, seed=a.seed + fit_seed)
        tv = jnp.asarray(sel[1]); rows = []
        for qi, ep in enumerate(teps):
            if ep.query_y is None: continue
            st = P.trace(model, cfg, ep.query_x, tau=1.0, task_vec=tv, t_total=a.t_total)
            exs = [ex(s["pred"], ep.query_y) for s in st]
            if a.dump_z and name == "g0":   # G2: the readout-fit features on the identity task (steps 1 and T, float16)
                sz = trace_z(model, cfg, ep.query_x, task_vec=tv, t_total=a.t_total)
                assert all(r["pred"].shape == q["pred"].shape and np.array_equal(r["pred"], q["pred"]) for r, q in zip(st, sz)), "trace_z != P.trace"
                gt_can = np.full((G.CANVAS, G.CANVAS), -1, np.int8); gy = np.asarray(ep.query_y, np.int8); gt_can[:gy.shape[0], :gy.shape[1]] = gy
                np.savez_compressed(Path(a.out) / f"z_{tid}_q{qi}.npz", gt=gt_can, hw_gt=np.array(gy.shape),
                                    **{f"{k}{i}": v for i in (0, cfg.T - 1, a.t_total - 1) for k, v in (("z", sz[i]["z"].astype(np.float16)), ("conf", sz[i]["conf"]), ("canvas", sz[i]["canvas"]), ("hw", np.array(sz[i]["hw"])))})
            rows.append({"exact_T": exs[cfg.T - 1], "exact_limit": exs[-1], "first_exact": next((i for i, e in enumerate(exs) if e), None),
                         "size_ok": bool(st[-1]["pred"].shape == ep.query_y.shape)})
        rec["g"][name] = {"k": t.k, "lut": [int(v) for v in t.lut[:10]], "fit_seed": fit_seed, "sel_step": sel[0], "wall_s": round(time.time() - t0, 1), "q": rows}
    return rec


def summarize(out: Path, names):
    R = [json.loads(l) for l in (out / "results.jsonl").read_text().splitlines() if l.strip()]
    names = [n for n in names if all(n in r["g"] for r in R)]
    lines = [f"ORBIT INVARIANCE, ARC twin: {out.name}; tasks {len(R)}, queries {sum(r['n_queries'] for r in R)}; transforms {names}"]
    ref = {(r["task"], i): q for r in R for i, q in enumerate(r["g"]["g0"]["q"])} if "g0" in names else {}
    for key in ("exact_limit", "exact_T"):
        lines.append(f"  [{key}] g | solved % | flips vs g0 % (n) | only-g / only-g0 | wall/fit s")
        for n in names:
            qs = [(r["task"], i, q) for r in R for i, q in enumerate(r["g"][n]["q"])]
            solved = np.mean([q[key] for _, _, q in qs]) * 100
            flips = [q[key] != ref[(t, i)][key] for t, i, q in qs if (t, i) in ref]
            og = sum(q[key] and not ref[(t, i)][key] for t, i, q in qs if (t, i) in ref)
            o0 = sum(ref[(t, i)][key] and not q[key] for t, i, q in qs if (t, i) in ref)
            wall = np.mean([r["g"][n]["wall_s"] for r in R])
            lines.append(f"    {n:4s} | {solved:6.2f} | {np.mean(flips)*100 if flips else float('nan'):6.2f} ({len(flips)}) | {og} / {o0} | {wall:.0f}")
    size = {n: np.mean([q["size_ok"] for r in R for q in r["g"][n]["q"]]) * 100 for n in names}
    lines.append("  size right at the end: " + ", ".join(f"{n} {v:.1f} %" for n, v in size.items()))
    txt = "\n".join(lines); print(txt); (out / "summary.txt").write_text(txt + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--set", default="valhard", choices=["valhard", "dev30"]); ap.add_argument("--tasks", default=None)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--transforms", default=",".join(TRANSFORMS))
    ap.add_argument("--steps", type=int, default=600); ap.add_argument("--val-every", type=int, default=50)
    ap.add_argument("--t-total", type=int, default=16); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--dump-z", action="store_true", help="on g0: save the carried latent + confidence at steps 1 / T / t_total per query (the G2 readout-fit features)")
    a = ap.parse_args(); names = a.transforms.split(","); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.summarize: return summarize(out, names)
    saved = E.load_ckpt(a.ckpt); defaults = Config()
    cfg = Config(**{k: type(getattr(defaults, k))(v) for k, v in saved["config"].items()}); state = saved["state"]
    if a.tasks: task_ids = a.tasks.split(",")
    elif a.set == "valhard": task_ids = json.load(open(Path(__file__).parent / "valhard.json"))["valhard"]
    else:
        import dev30; task_ids = sorted(dev30.MANIFEST)
    if a.limit: task_ids = task_ids[:a.limit]
    results = out / "results.jsonl"; done = set()
    if results.exists():
        for line in results.read_text().splitlines():
            try: done.add(json.loads(line)["task"])
            except Exception: pass
    with open(results, "a") as f:
        for tid in task_ids:
            if tid in done: print(f"skip {tid}", flush=True); continue
            t0 = time.time(); rec = run_task(state, cfg, tid, names, a)
            f.write(json.dumps(rec) + "\n"); f.flush()
            print(f"{tid} " + " ".join(f"{n}:{sum(q['exact_limit'] for q in rec['g'][n]['q'])}/{rec['n_queries']}" for n in names) + f" {time.time()-t0:.0f}s", flush=True)
    summarize(out, names)


if __name__ == "__main__":
    main()
