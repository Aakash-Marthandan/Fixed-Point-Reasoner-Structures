# Ledger: C16 pretrain-1 (2026-08-01 entry) — joint-episodic pretraining of the
# shared bulk + per-task embedding table on ARC-1 training minus the FROZEN
# dev-30 holdout. Registered config: d=16, K=64, T=4, tau=1.0, B=64
# task-balanced, AdamW warmup(500)->cosine(1e-3 -> 3e-5), wd 1e-4, clip 1.0,
# 20k steps, beta = beta_nl = 0 with BOTH flux ledgers logged (free-attention
# blowup is a registered risk with a named kill condition, not a surprise).
"""Joint-episodic pretraining (C16).

  .venv/bin/python tools/pretrain.py --out runs/pretrain1          # full run
  .venv/bin/python tools/pretrain.py --smoke --out /tmp/p1smoke    # CPU smoke

Resumable: reruns with the same --out continue from ckpt_latest.pkl (spot
preemption tolerance). Metrics stream to <out>/metrics.jsonl; checkpoints to
<out>/ckpt_latest.pkl (+ step-tagged snapshots); config + corpus manifest to
<out>/config.json at start.
"""
from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import jax
import jax.numpy as jnp
import optax

import dev30
from qhrrn2 import episodic as E
from qhrrn2 import grid as G
from qhrrn2 import sudoku_extreme as SX
from qhrrn2 import train as T
from qhrrn2.config import Config
from qhrrn2.model import count_params, init_params
from qhrrn2.objective import batch_loss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--d", type=int, default=16)
    p.add_argument("--equilibrium", action="store_true",
                   help="E10: continuous-state equilibrium core")
    p.add_argument("--anchor-p", type=float, default=0.0,
                   help="E10 Phase B: fraction of batch rows given corrupted-"
                        "target yprev init (basin training at corpus scale)")
    p.add_argument("--anchor-eps", type=float, default=0.15)
    p.add_argument("--ri-p", type=float, default=0.0,
                   help="pretrain-13 RI rows (EqR): fraction of rows starting "
                        "from a full uniform random color canvas — trains "
                        "path-independence; 0 = pre-13 stream bit-exact")
    p.add_argument("--ni-sigma", type=float, default=0.0,
                   help="pretrain-13 NI (EqR): per-step training noise std "
                        "in state space (simplex-tangent); 0 = off")
    p.add_argument("--eq-coupled", action="store_true",
                   help="pretrain-13: FPRM coupled residual scaling a1/a2 "
                        "(learnable, init contractive .75/.25) instead of "
                        "the damped y+eta*(p-y) update")
    p.add_argument("--flux-floors", default=None,
                   help="pretrain-13 B1-full: comma nats per scale, e.g. "
                        "'350,75,50,15,30' — free-bits floors; only excess "
                        "above the floor is beta-priced")
    p.add_argument("--beta-flux", type=float, default=0.0,
                   help="P9-C: the RT toll shaping the landscape (S1/S2)")
    p.add_argument("--beta-flux-nl", type=float, default=0.0)
    p.add_argument("--eta-floor", type=float, default=0.0)
    p.add_argument("--z-gate-init", type=float, default=0.0)
    p.add_argument("--K", type=int, default=64)
    p.add_argument("--T", type=int, default=4)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-end", type=float, default=3e-5)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-val", type=int, default=20)
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--val-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--limit", type=int, default=None, help="corpus size cap (smoke)")
    p.add_argument("--val-ids-file", default=None,
                   help="json with {'val40': [...]} — explicit val holdout (CC#2)")
    p.add_argument("--obj", action="store_true", help="C17 cluster layers on")
    p.add_argument("--sudoku", type=int, default=0,
                   help="S-port (H-33): train on N generated Sudoku puzzles "
                        "instead of the ARC corpus (one task row)")
    p.add_argument("--sudoku-givens", type=int, default=30,
                   help="S-port difficulty dial: target givens per puzzle")
    p.add_argument("--sudoku-givens-hi", type=int, default=None,
                   help="if set, givens ~ U[givens, givens_hi] per puzzle "
                        "(mixed-difficulty training; see the cell-1 design "
                        "review — a single hard setting can VOID H-33)")
    p.add_argument("--sudoku-extreme", default=None,
                   help="SPRINT S2: prepared Sudoku-Extreme npz "
                        "(tools/prep_sudoku_extreme.py) — train on its seeded "
                        "1k subsample x --sudoku-aug group-augmented copies "
                        "(one task row); val = its disjoint monitor rows")
    p.add_argument("--sudoku-aug", type=int, default=100,
                   help="group-augmented copies per base puzzle (HRM: 1000)")
    p.add_argument("--sudoku-digit-aug", action="store_true",
                   help="ablation: explicit digit permutation too (exact S9 "
                        "covers it by construction; prediction = no change)")
    # wave 3a (ledger 2026-08-23, H-45): fixed-point anchor rows + trajectory monitor
    p.add_argument("--fpa-k", type=int, default=0,
                   help="fixed-point anchor: final-map steps supervised from a corrupted solution (0=off)")
    p.add_argument("--fpa-eps", type=float, default=0.2, help="max corruption fraction for FPA starts")
    p.add_argument("--fpa-w", type=float, default=1.0, help="FPA loss weight")
    p.add_argument("--monitor-every", type=int, default=0,
                   help="every N steps: val@t64 cold, schedule + FINAL-MAP retention (t8, solution "
                        "init), eta, and lambda_max of the final map at the solution (power iteration) "
                        "on the monitor puzzles -> metrics.jsonl {'monitor':...}; 0 = off")
    p.add_argument("--sudoku-layout", default="origin", choices=["origin", "box4"],
                   help="wave-2 (2026-08-22): Sudoku canvas layout; box4 = the "
                        "registered box-aligned control (carried in the ckpt cfg)")
    p.add_argument("--init-from", default=None,
                   help="warm-start params+table from this ckpt at step 0 with "
                        "a fresh optimizer (the GEN arm's 1k finetune)")
    p.add_argument("--d-task", type=int, default=32,
                   help="boundary program width (H-17 co-scaling)")
    p.add_argument("--width-scale", type=float, default=1.0,
                   help="PHASE B ladder (ledger 2026-08-24): multiply the side widths "
                        "(d_b, d_a, d_ir, d_code, d_task) by this factor from their d16-reference "
                        "defaults (6,6,32,32,32) so params scale ~d^2 (full-width scaling; the "
                        "canvas-head hidden 64 stays fixed — documented exception). The concrete "
                        "widths are stored in the ckpt Config; downstream tools need no changes. "
                        "1.0 = bit-exact pre-existing behavior. Incompatible with an explicit --d-task.")
    p.add_argument("--orbit", type=int, default=1,
                   help="orbit expansion factor: virtual tasks per base task")
    p.add_argument("--rearc", action="store_true",
                   help="C20a: mix RE-ARC train-family instances into the "
                        "corpus; enforces C20b family-holdout + dev-30 "
                        "exclusion + gate-original exclusion (pretrain-10 law)")
    p.add_argument("--rearc-per-family", type=int, default=20)
    p.add_argument("--rearc-seed", type=int, default=0)
    p.add_argument("--conceptarc", action="store_true",
                   help="merge vendored ConceptARC (minus val-hard) into the corpus")
    p.add_argument("--remat", action="store_true",
                   help="gradient-checkpoint recursion steps (HBM relief)")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--dp", action="store_true",
                   help="data-parallel pmap over local devices (P11-EXT "
                        "2026-08-11); global batch preserved, grads pmean'd")
    return p.parse_args()


def git_rev():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10,
                              cwd=Path(__file__).resolve().parents[1]).stdout.strip()
    except Exception:
        return "unknown"


def val20_eval(state, cfg, val, tau):
    """Exact-match on held-out QUERY pairs of val tasks, using each task's
    trained embedding — the within-task generalization monitor (R1 signal).
    Also reports CI-8b within-object consistency (C17 gate-collapse early
    warning, monitored DURING pretraining per the 2026-08-02 registration)."""
    from taxonomy import within_object_consistency
    n_exact = n_total = 0
    pix_sum = 0.0
    c_ok = c_tot = 0
    for t, task_id, queries in val:
        tv = state["table"][t]
        for qx, qy in queries:
            exact, pix, _ = T.evaluate_pair(state["model"], cfg, qx, qy,
                                            tau=tau, task_vec=tv)
            pred, shape, _ = T.predict(state["model"], cfg, qx, tau=tau, task_vec=tv)
            ok, tot = within_object_consistency(pred, qy)
            c_ok += ok
            c_tot += tot
            n_exact += int(exact)
            pix_sum += pix
            n_total += 1
    return {"val_exact": n_exact, "val_total": n_total,
            "val_pix_mean": pix_sum / max(n_total, 1),
            "obj_consistency": round(c_ok / max(c_tot, 1), 4),
            "obj_consistency_n": c_tot}


def sudoku_monitor(state, cfg, val_pairs, *, t_cold=64, t_ret=8, n_lam=16, lam_iters=12):
    """Wave 3a TRAJECTORY MONITOR (H-45): on the held-out monitor puzzles —
    val@t_cold cold exact; retention from the solution under the trained
    schedule and under the FINAL map only (t_norm=1); eta/eta_z (or a1/a2);
    lambda_max of the final-map update Jacobian at the solution (power
    iteration, no z-carry, first n_lam puzzles) — the RG-eigenvalue readout:
    contractive iff |lambda|max < 1. Uses the batched evaluator's own step
    functions so the monitor and the registered evaluation agree."""
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    import eval_sudoku_extreme as EV
    from qhrrn2 import model as M
    params = state["model"]; tvj = jnp.asarray(state["table"][0])
    layout = getattr(cfg, "sudoku_layout", "origin") or "origin"
    eta = float(cfg.eta_floor + (1.0 - cfg.eta_floor) * jax.nn.sigmoid(params["eq"]["eta"]))
    eta_z = float(jax.nn.sigmoid(params["eq"]["eta_z"]))
    ab = EV.coupled_ab(params, cfg)
    puz9 = np.stack([np.asarray(p_, np.int32) for p_, _ in val_pairs]); sol9 = np.stack([np.asarray(s_, np.int32) for _, s_ in val_pairs])
    x_can = EV.place_batch(puz9, layout); B = x_can.shape[0]
    void = jax.nn.one_hot(jnp.full((G.CANVAS, G.CANVAS), G.VOID, jnp.int32), M.VOCAB).transpose(2, 0, 1)
    y0v = jnp.broadcast_to(void, (B,) + void.shape)
    ex, _, _ = EV.run_batch(params, cfg, tvj, x_can, y0v, t_total=t_cold, tau=1.0, gamma=1.0,
                            sol9=sol9, puz9=puz9, eta=eta, eta_z=eta_z, layout=layout, ab=ab)
    val_cold = float(ex[-1].mean())
    y0s = jax.nn.one_hot(EV.place_batch(sol9, layout), M.VOCAB).transpose(0, 3, 1, 2)
    exr, _, _ = EV.run_batch(params, cfg, tvj, x_can, y0s, t_total=t_ret, tau=1.0, gamma=1.0,
                             sol9=sol9, puz9=puz9, eta=eta, eta_z=eta_z, layout=layout, ab=ab)
    exf, _, _ = EV.run_batch(params, cfg, tvj, x_can, y0s, t_total=t_ret, tau=1.0, gamma=1.0,
                             sol9=sol9, puz9=puz9, eta=eta, eta_z=eta_z, layout=layout, ab=ab,
                             t_norm_fixed=1.0)
    # Final-map fixed-point readouts on the first n_lam puzzles (2026-08-23, refined after the
    # banked-ckpt check: the y-only/z-less spectrum does NOT separate collapsed from healthy maps;
    # the collapse reads as the FINAL map's fixed point DRIFTING off the solution). Three numbers:
    #   fp_drift  = mean |F(sol, z*) - sol| per cell after settling z* (4 final-map steps from the
    #               solution): "is the solution still a fixed point of the final map?" (0 = yes)
    #   lam_joint = |lambda|max of the JOINT (y, z) final-map Jacobian at (sol, z*) by power iteration
    #   lam_yonly = the y-only, z=None spectrum (kept for continuity; informational)
    eta_z_v = eta_z
    def Fy(yy, xx, zz):
        out = M.forward_fields(params, cfg, M.build_fields_soft(xx, yy), t_norm=1.0, tau=1.0,
                               rng=None, task_vec=tvj, z_in=zz)
        pp = jax.nn.softmax(out.logits, axis=-1).transpose(2, 0, 1)
        y2 = (ab[0] * yy + ab[1] * pp) if ab is not None else (yy + eta * (pp - yy))
        return y2, out.z_fine
    def F_joint(yz, xx):
        yy, zz = yz
        y2, zf = Fy(yy, xx, zz)
        return (y2, zz + eta_z_v * (zf - zz))
    def settle(xx, ys):          # z* by 4 final-map steps from the solution (first step has no z)
        y2, zf = Fy(ys, xx, None); z = zf
        def body(_, carry):
            yy, zz = carry; return F_joint((yy, zz), xx)
        yy, zz = jax.lax.fori_loop(0, 3, body, (y2, z))
        return zz
    def readouts(xx, ys, key):
        zs = settle(xx, ys)
        y1, _ = F_joint((ys, zs), xx)
        drift = jnp.mean(jnp.abs(y1 - ys))
        vy = jax.random.normal(key, ys.shape); vz = jax.random.normal(jax.random.fold_in(key, 1), zs.shape)
        nrm = jnp.sqrt(jnp.sum(vy**2) + jnp.sum(vz**2)) + 1e-9; vy, vz = vy / nrm, vz / nrm
        def body(_, v):
            _, jv = jax.jvp(lambda yz: F_joint(yz, xx), ((ys, zs),), (v,))
            n = jnp.sqrt(jnp.sum(jv[0]**2) + jnp.sum(jv[1]**2)) + 1e-12
            return (jv[0] / n, jv[1] / n)
        v = jax.lax.fori_loop(0, lam_iters, body, (vy, vz))
        _, jv = jax.jvp(lambda yz: F_joint(yz, xx), ((ys, zs),), (v,))
        lam_j = jnp.sqrt(jnp.sum(jv[0]**2) + jnp.sum(jv[1]**2))
        # y-only, z=None (the original readout)
        def Fy0(yy): return Fy(yy, xx, None)[0]
        u = jax.random.normal(jax.random.fold_in(key, 2), ys.shape); u = u / (jnp.linalg.norm(u) + 1e-9)
        def body0(_, u):
            _, ju = jax.jvp(Fy0, (ys,), (u,)); return ju / (jnp.linalg.norm(ju) + 1e-12)
        u = jax.lax.fori_loop(0, lam_iters, body0, u)
        _, ju = jax.jvp(Fy0, (ys,), (u,))
        return drift, lam_j, jnp.linalg.norm(ju)
    n = min(n_lam, B)
    keys = jax.random.split(jax.random.PRNGKey(0), n)
    drift, lam_j, lam = (np.asarray(t) for t in jax.vmap(readouts)(x_can[:n], y0s[:n], keys))
    rec = dict(val_t64=val_cold, ret_sched_t8=float(exr[-1].mean()), ret_final_t8=float(exf[-1].mean()),
               eta=eta, eta_z=eta_z,
               fp_drift_mean=float(drift.mean()), fp_drift_max=float(drift.max()),
               lam_joint_mean=float(lam_j.mean()), lam_joint_max=float(lam_j.max()), lam_joint_frac_expansive=float((lam_j > 1.0).mean()),
               lam_max_mean=float(lam.mean()), lam_max_max=float(lam.max()),
               lam_frac_expansive=float((lam > 1.0).mean()), n_val=int(B), n_lam=int(n))
    if ab is not None:
        rec["a1"], rec["a2"] = float(ab[0]), float(ab[1])
    return rec


def main():
    a = parse_args()
    if a.smoke:
        a.steps, a.d, a.K, a.T = 40, 8, 8, 2
        a.batch, a.warmup, a.limit = 8, 5, 8
        a.n_val = 2
        a.ckpt_every = a.val_every = 20
        a.log_every = 5

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.width_scale != 1.0:
        if a.d_task != 32:
            raise SystemExit("--width-scale is incompatible with an explicit --d-task")
        ws = a.width_scale
        a.d_task = int(round(32 * ws))
        side = dict(d_b=int(round(6 * ws)), d_a=int(round(6 * ws)),
                    d_ir=int(round(32 * ws)), d_code=int(round(32 * ws)))
    else:
        side = {}
    cfg = Config(d=a.d, K=a.K, T=a.T, use_obj=a.obj, remat=a.remat, **side,
                 d_task=a.d_task, equilibrium=a.equilibrium,
                 beta_flux=a.beta_flux, beta_flux_nl=a.beta_flux_nl,
                 eta_floor=a.eta_floor, z_gate_init=a.z_gate_init,
                 eq_coupled=a.eq_coupled, ni_sigma=a.ni_sigma,
                 flux_floors=a.flux_floors or "",
                 sudoku_layout=a.sudoku_layout,
                 fpa_k=a.fpa_k, fpa_eps=a.fpa_eps, fpa_w=a.fpa_w)

    exclude = frozenset(dev30.MANIFEST)
    rearc_families = None
    if a.rearc:
        # C20b contamination laws (ledger 2026-08-10): gate families never
        # pretrain in ANY form (originals excluded too); dev-30 families
        # never enter as generator instances.
        from qhrrn2 import rearc as R
        train_fams, gate_fams = R.family_split()
        exclude = exclude | frozenset(gate_fams)
        rearc_families = sorted(set(train_fams) - set(dev30.MANIFEST))
        print(f"C20 corpus law: {len(rearc_families)} RE-ARC train families "
              f"({len(set(train_fams) & set(dev30.MANIFEST))} dev-30 removed); "
              f"{len(gate_fams)} gate-family originals excluded", flush=True)
    val_ids = None
    if a.val_ids_file:
        val_ids = json.load(open(a.val_ids_file))["val40"]
    exclude_ca = frozenset()
    if a.conceptarc:
        from qhrrn2.grid import list_conceptarc
        if not list_conceptarc():
            sys.exit("--conceptarc: data/ConceptARC not vendored on this host")
        vh_path = Path(__file__).resolve().parent / "valhard.json"
        exclude_ca = frozenset(json.load(open(vh_path))["valhard"])
    if a.sudoku_extreme:
        # SPRINT S2 (2026-08-21): the BENCHMARK corpus — seeded 1k Sudoku-
        # Extreme puzzles x group-augmented copies, one task row; val = the
        # file's disjoint monitor rows (never the test set).
        corpus, val = SX.build_corpus_extreme(
            a.sudoku_extreme, n_aug=a.sudoku_aug, seed=a.seed,
            digit_aug=a.sudoku_digit_aug, layout=a.sudoku_layout)
        print(f"SPRINT-S2 corpus: Sudoku-Extreme {a.sudoku_extreme} x "
              f"(1+{a.sudoku_aug}) group copies (digit_aug={a.sudoku_digit_aug}, "
              f"layout={a.sudoku_layout}) = {corpus.x.shape[0]} pairs, one task row; "
              f"val = monitor rows", flush=True)
        if a.sudoku_layout != "origin":
            # the trainer's val monitor (val20_eval -> T.predict) places grids
            # at the origin; under another layout it would read a wrong-layout
            # zero. The batched evaluator (layout-aware) measures val@t for
            # these arms (wave-2 M0 reads the evaluator, not this monitor).
            val = []
            print("val monitor DISABLED for non-origin layout (evaluator measures val)", flush=True)
    elif a.sudoku:
        # S-PORT (H-33): the single-attractor domain. ONE task row, generated
        # instances, no ARC corpus involved — the contamination laws below
        # are vacuous here and deliberately skipped.
        corpus, val = E.build_sudoku_corpus(
            a.sudoku, n_val=a.n_val, seed=a.seed, givens=a.sudoku_givens,
            givens_hi=a.sudoku_givens_hi)
        rng_txt = (f"{a.sudoku_givens}-{a.sudoku_givens_hi}"
                   if a.sudoku_givens_hi else str(a.sudoku_givens))
        print(f"S-port corpus: {a.sudoku} generated puzzles @ {rng_txt} "
              f"givens (+{a.n_val} held-out), one task row", flush=True)
    else:
        corpus, val = E.build_corpus(exclude, n_val=a.n_val, seed=a.seed, limit=a.limit,
                                     val_ids=val_ids, orbit_n=a.orbit,
                                     conceptarc=a.conceptarc, exclude_ca=exclude_ca,
                                     rearc_families=rearc_families,
                                     rearc_per_family=a.rearc_per_family,
                                     rearc_seed=a.rearc_seed)
    dev = E.corpus_to_device(corpus)
    n_tasks = len(corpus.task_ids)
    n_pairs = int(corpus.x.shape[0])

    key = jax.random.PRNGKey(a.seed)
    k_model, k_table, k_run = jax.random.split(key, 3)
    state = {"model": init_params(k_model, cfg),
             "table": E.init_table(k_table, n_tasks, cfg.d_task)}
    n_bulk = count_params(state["model"])
    n_table = count_params(state["table"])

    sched = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=a.lr, warmup_steps=a.warmup,
        decay_steps=a.steps, end_value=a.lr_end)
    opt = optax.chain(optax.clip_by_global_norm(1.0),
                      optax.adamw(sched, weight_decay=a.wd))
    opt_state = opt.init(state)
    start_step = 0
    rng = k_run

    latest = out / "ckpt_latest.pkl"
    if latest.exists():
        saved = E.load_ckpt(latest)
        state, opt_state = saved["state"], saved["opt_state"]
        start_step, rng = int(saved["step"]), jnp.asarray(saved["rng"])
        print(f"RESUMED from {latest} at step {start_step}", flush=True)
    elif a.init_from:
        saved = E.load_ckpt(a.init_from)
        src = saved["state"]
        assert jax.tree.structure(src["model"]) == jax.tree.structure(state["model"]), \
            "--init-from: model tree mismatch (config differs)"
        state = dict(state, model=src["model"])
        if np.asarray(src["table"]).shape == np.asarray(state["table"]).shape:
            state["table"] = src["table"]
        opt_state = opt.init(state)
        print(f"INIT-FROM {a.init_from} (step 0, fresh optimizer)", flush=True)

    config_rec = {
        "argv": vars(a), "config": dataclasses.asdict(cfg), "git": git_rev(),
        "n_tasks": n_tasks, "n_pairs": n_pairs,
        "n_params_bulk": n_bulk, "n_params_table": n_table,
        "val_tasks": [task_id for _, task_id, _ in val],
        "backend": jax.default_backend(),
        "corpus_task_ids": list(corpus.task_ids),
    }
    (out / "config.json").write_text(json.dumps(config_rec, indent=1))
    print(f"corpus: {n_tasks} tasks / {n_pairs} pairs; params bulk={n_bulk} "
          f"table={n_table}; backend={jax.default_backend()}", flush=True)

    if cfg.use_obj:
        t0 = time.time()
        dev["labels"] = E.precompute_labels(corpus)
        dev["labels"].block_until_ready()
        print(f"labels precomputed for {n_pairs} pairs in {time.time()-t0:.1f}s",
              flush=True)

    def _sample_and_loss(st, rng_step, batch_sz):
        """Shared batch-construction + loss core (single-device and DP paths
        MUST stay in lockstep — the DP gradient-equivalence test in
        tests/test_dp.py guards it)."""
        import jax.numpy as jnp
        # pretrain-13: the ri_p>0 branch draws 2 extra keys; at ri_p=0 the
        # split count and every downstream key match pretrain-12 bit-exactly.
        nk = 7 if a.ri_p > 0 else 5
        keys = jax.random.split(rng_step, nk)
        k_batch, k_loss, k_a1, k_a2, k_a3 = keys[:5]
        x_b, y_b, t_b, lab_b = E.sample_batch(k_batch, dev, n_tasks, batch_sz)
        # [H-23] anchor rows + pretrain-13 RI rows (builder + named tests in
        # episodic.build_y0_rows / tests/test_p13.py).
        yp_b = E.build_y0_rows(
            k_a1, k_a2, k_a3, y_b, a.anchor_p, a.anchor_eps, ri_p=a.ri_p,
            k_r1=keys[5] if a.ri_p > 0 else None,
            k_r2=keys[6] if a.ri_p > 0 else None)

        def loss_fn(s):
            return batch_loss(s["model"], cfg, x_b, y_b, tau=a.tau, rng=k_loss,
                              task_vecs=s["table"][t_b], labels_x=lab_b,
                              yprev_batch=yp_b)
        return jax.value_and_grad(loss_fn, has_aux=True)(st)

    @jax.jit
    def step_fn(state, opt_state, rng):
        rng, rng_step = jax.random.split(rng)
        (loss, aux), grads = _sample_and_loss(state, rng_step, a.batch)
        updates, opt_state2 = opt.update(grads, opt_state, state)
        return optax.apply_updates(state, updates), opt_state2, loss, aux, rng

    # DP path (P11-EXT registration 2026-08-11): pmap over local devices,
    # per-device batch shard (global batch preserved: a.batch // n_dev rows
    # each), gradients pmean'd, replicas updated in lockstep. Enabler for
    # litepod-8/16 single-model pretrains (the 5-75M track).
    n_dev = jax.local_device_count() if a.dp else 1
    if a.dp:
        assert a.batch % n_dev == 0, f"batch {a.batch} % devices {n_dev} != 0"

        @functools.partial(jax.pmap, axis_name="dp")
        def step_fn_dp(state, opt_state, rng):
            rng, rng_step = jax.random.split(rng)
            rng_step = jax.random.fold_in(rng_step, jax.lax.axis_index("dp"))
            (loss, aux), grads = _sample_and_loss(
                state, rng_step, a.batch // n_dev)
            grads = jax.lax.pmean(grads, "dp")
            loss = jax.lax.pmean(loss, "dp")
            aux = jax.lax.pmean(aux, "dp")
            updates, opt_state2 = opt.update(grads, opt_state, state)
            return optax.apply_updates(state, updates), opt_state2, loss, aux, rng

    if a.dp:
        rep = lambda tree: jax.tree.map(
            lambda x: jnp.stack([x] * n_dev), tree)
        state_r, opt_r = rep(state), rep(opt_state)
        rng_r = jax.random.split(rng, n_dev)
        print(f"DP: {n_dev} devices, {a.batch // n_dev} rows/device", flush=True)
    metrics_f = open(out / "metrics.jsonl", "a")
    t_block = time.time()
    for i in range(start_step, a.steps):
        if a.dp:
            state_r, opt_r, loss_r, aux_r, rng_r = step_fn_dp(state_r, opt_r, rng_r)
            loss = jax.tree.map(lambda x: x[0], loss_r)
            aux = jax.tree.map(lambda x: x[0], aux_r)
            if (i + 1) % a.val_every == 0 or (i + 1) % a.ckpt_every == 0 \
                    or i + 1 == a.steps:
                state = jax.tree.map(lambda x: x[0], state_r)
                opt_state = jax.tree.map(lambda x: x[0], opt_r)
                rng = rng_r[0]
        else:
            state, opt_state, loss, aux, rng = step_fn(state, opt_state, rng)

        if (i + 1) % a.log_every == 0 or i + 1 == a.steps:
            dt = time.time() - t_block
            sps = a.log_every / dt if dt > 0 else 0.0
            # aux values are means over batch AND trailing axes; × scales
            # recovers per-pair ledger TOTALS (the A-blowup monitor).
            rec = {
                "step": i + 1,
                "loss": float(loss),
                **({"fpa_ce": float(aux["fpa_ce_last"])} if "fpa_ce_last" in aux else {}),
                "ce_in": float(aux["ce_in_last"]),
                "I_total": float(aux["flux_last"]) * cfg.scales,
                "A_total": float(aux["flux_attn_last"]) * cfg.scales,
                "rule_H": float(aux["rule_entropy_last"]),
                "lr": float(sched(i + 1)),
                "steps_per_sec": round(sps, 3),
                "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            metrics_f.write(json.dumps(rec) + "\n")
            metrics_f.flush()
            print(f"step {i+1:6d}  loss {rec['loss']:.4f}  ce {rec['ce_in']:.4f}  "
                  f"I {rec['I_total']:.1f}  A {rec['A_total']:.1f}  "
                  f"{sps:.2f} it/s", flush=True)
            t_block = time.time()

        if (i + 1) % a.val_every == 0 or i + 1 == a.steps:
            v = val20_eval(state, cfg, val, a.tau)
            v["step"] = i + 1
            metrics_f.write(json.dumps({"val": v}) + "\n")
            metrics_f.flush()
            print(f"  VAL step {i+1}: exact {v['val_exact']}/{v['val_total']} "
                  f"pix {v['val_pix_mean']:.3f} objcons {v['obj_consistency']:.3f}"
                  f"(n={v['obj_consistency_n']})", flush=True)
            t_block = time.time()

        if a.monitor_every and cfg.equilibrium and val and ((i + 1) % a.monitor_every == 0 or i + 1 == a.steps):
            if a.dp:
                state = jax.tree.map(lambda x: x[0], state_r)
            t_m = time.time()
            mon = sudoku_monitor(state, cfg, val[0][2])
            mon["step"] = i + 1; mon["wall_s"] = round(time.time() - t_m, 1)
            metrics_f.write(json.dumps({"monitor": mon}) + "\n"); metrics_f.flush()
            print(f"  MONITOR step {i+1}: val@t64 {mon['val_t64']:.3f} ret_sched {mon['ret_sched_t8']:.2f} "
                  f"ret_final {mon['ret_final_t8']:.2f} eta {mon['eta']:.3f} fp_drift {mon['fp_drift_mean']:.4f} "
                  f"lam_joint {mon['lam_joint_mean']:.3f}/{mon['lam_joint_max']:.3f} lam_y {mon['lam_max_mean']:.3f} ({mon['wall_s']}s)", flush=True)
            t_block = time.time()

        if (i + 1) % a.ckpt_every == 0 or i + 1 == a.steps:
            payload = {"state": state, "opt_state": opt_state, "step": i + 1,
                       "rng": np.asarray(rng), "config": dataclasses.asdict(cfg)}
            E.save_ckpt(latest, payload)
            if (i + 1) % (5 * a.ckpt_every) == 0 or i + 1 == a.steps:
                E.save_ckpt(out / f"ckpt_{i+1:06d}.pkl", payload)

    metrics_f.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
