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
from qhrrn2 import trm_cell as TC
from qhrrn2.config import Config
from qhrrn2.model import count_params, init_params
from qhrrn2.objective import batch_loss, log_stablemax
import math


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
    p.add_argument("--sudoku-layout", default="origin", choices=["origin", "box4", "native9"],
                   help="wave-2 (2026-08-22): Sudoku canvas layout; box4 = the "
                        "registered box-aligned control (carried in the ckpt cfg); "
                        "native9 (CHAMPION TRACK 2026-09-01) = 3-adic native 9x9 — "
                        "sets canvas=9, scales=2, pool_arity=3, mixer group9, "
                        "attn_max_hw=9 in the Config automatically")
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
    # sportC1 (2026-09-02; Plan_2026-09-02_Champion_sportC1 §11–§12)
    p.add_argument("--z-norm", default="", choices=["", "rms"],
                   help="H-50 stabilizer of record: RMSNorm the carried latent at its entry (arms B0/B1/R0)")
    p.add_argument("--cell", default="rg", choices=["rg", "trm"],
                   help="X0: 'trm' = the TRM/EqR field-recipe cell (qhrrn2.trm_cell); 'rg' = ours")
    p.add_argument("--trm-hidden", type=int, default=512)
    p.add_argument("--trm-layers", type=int, default=2)
    p.add_argument("--trm-h-cycles", type=int, default=3)
    p.add_argument("--trm-l-cycles", type=int, default=6)
    p.add_argument("--trm-lambda", type=float, default=0.0, help="EqR Eq. 2 damping per inner pass (.05)")
    p.add_argument("--trm-beta", type=float, default=0.0, help="EqR path noise per inner pass (.01)")
    p.add_argument("--trm-ri-sigma", type=float, default=0.0, help="EqR RI: z0 ~ N(0, sigma) (1)")
    p.add_argument("--trm-token-mixer", default="mlp", choices=["mlp", "group9"],
                   help="sportC2 X2: 'group9' = our factorized group mixer as the field cell's token mixer "
                        "on a --trm-gm-dim projection (labeled param count); 'mlp' = TRM exact")
    p.add_argument("--trm-gm-dim", type=int, default=64)
    p.add_argument("--loss", default="softmax", choices=["softmax", "stablemax"],
                   help="stablemax = HRM/TRM's cross-entropy (X0, labeled)")
    p.add_argument("--beta2", type=float, default=0.999, help="AdamW beta2 (the field: .95)")
    p.add_argument("--ema", type=float, default=0.0,
                   help="EMA of the weights (rate; 0 = off) kept eval-side as state_ema in the ckpt "
                        "(the field's .999); the monitor logs val on both raw and EMA weights")
    p.add_argument("--sot", action="store_true",
                   help="X0: SEGMENTED ONLINE TRAINING (HRM/TRM/EqR loop) — one outer segment per "
                        "optimizer step with a persistent detached carry; rows that finish cfg.T "
                        "segments (or halt under --act) are replaced by fresh samples")
    p.add_argument("--act", action="store_true",
                   help="X0: adaptive halting during --sot (TRM no_ACT_continue variant: q_halt head, "
                        "BCE vs sequence-correct, halt at sigmoid(q) > .5, exploration prob --halt-explore)")
    p.add_argument("--halt-explore", type=float, default=0.1)
    p.add_argument("--inner-k", type=int, default=1,
                   help="sportC2 R2: latent passes per outer step before the readout update (1 = bit-exact)")
    p.add_argument("--hard-p", type=float, default=0.0,
                   help="sportC2 R3: probability per outer step of HARD (argmax, straight-through) feedback "
                        "during training (0 = bit-exact)")
    p.add_argument("--sot-segments", type=int, default=4,
                   help="sportC2 R1 (--sot on our cell): max T-step segments a row is carried before it is "
                        "replaced (4 x T16 = the t=64 evaluation horizon); rows are also replaced when EXACT")
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
    eta, eta_z = (float(v) for v in M.eq_etas(params, cfg))
    ab = EV.coupled_ab(params, cfg)
    if cfg.cell_kind == "trm":
        # X0: y is a readout and there is no answer register — retention / final-map
        # readouts do not exist for this cell; val at D = cfg.T outer segments (EqR's
        # base budget, the number their 84.8 is read at), key val_t{T}.
        t_cold = cfg.T
    puz9 = np.stack([np.asarray(p_, np.int32) for p_, _ in val_pairs]); sol9 = np.stack([np.asarray(s_, np.int32) for _, s_ in val_pairs])
    x_can = EV.place_batch(puz9, layout); B = x_can.shape[0]
    from qhrrn2 import sudoku as SUD
    cv = SUD.layout_canvas(layout)
    void = jax.nn.one_hot(jnp.full((cv, cv), G.VOID, jnp.int32), M.VOCAB).transpose(2, 0, 1)
    y0v = jnp.broadcast_to(void, (B,) + void.shape)
    ex, _, _, _ = EV.run_batch(params, cfg, tvj, x_can, y0v, t_total=t_cold, tau=1.0, gamma=1.0,
                               sol9=sol9, puz9=puz9, eta=eta, eta_z=eta_z, layout=layout, ab=ab)
    val_cold = float(ex[-1].mean())
    if cfg.cell_kind == "trm":
        return {f"val_t{t_cold}": val_cold, "eta": eta, "eta_z": eta_z, "n_val": int(B)}
    y0s = jax.nn.one_hot(EV.place_batch(sol9, layout), M.VOCAB).transpose(0, 3, 1, 2)
    exr, _, _, _ = EV.run_batch(params, cfg, tvj, x_can, y0s, t_total=t_ret, tau=1.0, gamma=1.0,
                                sol9=sol9, puz9=puz9, eta=eta, eta_z=eta_z, layout=layout, ab=ab)
    exf, _, _, _ = EV.run_batch(params, cfg, tvj, x_can, y0s, t_total=t_ret, tau=1.0, gamma=1.0,
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
    K_in = max(1, int(getattr(cfg, "inner_k", 1)))   # sportC2 R2: the final map = K latent passes per readout
    def Fy(yy, xx, zz):
        """One outer step of the FINAL map from (y, z): returns (y2, z_next) with z_next the carried
        latent AFTER the step (K inner passes; K = 1 reproduces the pre-existing map bit-exactly)."""
        z = zz
        for _k in range(K_in):
            out = M.forward_fields(params, cfg, M.build_fields_soft(xx, yy), t_norm=1.0, tau=1.0,
                                   rng=None, task_vec=tvj, z_in=z)
            z = out.z_fine if z is None else z + eta_z_v * (out.z_fine - z)
        pp = jax.nn.softmax(out.logits, axis=-1).transpose(2, 0, 1)
        y2 = (ab[0] * yy + ab[1] * pp) if ab is not None else (yy + eta * (pp - yy))
        return y2, z
    def F_joint(yz, xx):
        yy, zz = yz
        return Fy(yy, xx, zz)
    def settle(xx, ys):          # z* by 4 final-map steps from the solution (first step has no z)
        y2, z = Fy(ys, xx, None)
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
    # CHAMPION TRACK (2026-09-01): native9 fixes the 3-adic geometry dials —
    # canvas IS the 9x9 grid (9 -> 3 -> 1 box-aligned), factorized group mixer.
    geo = (dict(canvas=9, scales=2, pool_arity=3, mixer_kind="group9",
                attn_max_hw=9)
           if a.sudoku_layout == "native9" else {})
    trm = (dict(cell_kind="trm", trm_hidden=a.trm_hidden, trm_layers=a.trm_layers,
                trm_h_cycles=a.trm_h_cycles, trm_l_cycles=a.trm_l_cycles,
                trm_lambda=a.trm_lambda, trm_beta=a.trm_beta, trm_ri_sigma=a.trm_ri_sigma, trm_token_mixer=a.trm_token_mixer, trm_gm_dim=a.trm_gm_dim,
                eta_fixed=1.0, eta_z_fixed=1.0)     # y = readout; the latent carries undamped
           if a.cell == "trm" else {})
    if a.cell == "trm":
        # the field-recipe cell has no answer register: our y-side mechanisms do not apply
        assert a.equilibrium and a.sudoku_layout == "native9", "--cell trm needs --equilibrium + native9"
        assert a.fpa_k == 0 and a.ri_p == 0 and a.anchor_p == 0 and a.ni_sigma == 0 and not a.eq_coupled, \
            "--cell trm: FPA / RI rows / anchors / NI / eq_coupled are y-register mechanisms (use --trm-* dials)"
    assert not a.act or a.sot, "--act needs --sot"
    # --sot on the trm cell = the field's online segment loop (X0); on our cell = the sportC2
    # persistent (y, z) carry with verifier-replaced rows (R1; run_sot_rg)
    cfg = Config(d=a.d, K=a.K, T=a.T, use_obj=a.obj, remat=a.remat, **side, **geo, **trm,
                 d_task=a.d_task, equilibrium=a.equilibrium,
                 beta_flux=a.beta_flux, beta_flux_nl=a.beta_flux_nl,
                 eta_floor=a.eta_floor, z_gate_init=a.z_gate_init,
                 eq_coupled=a.eq_coupled, ni_sigma=a.ni_sigma,
                 flux_floors=a.flux_floors or "",
                 sudoku_layout=a.sudoku_layout,
                 fpa_k=a.fpa_k, fpa_eps=a.fpa_eps, fpa_w=a.fpa_w,
                 z_norm=a.z_norm, loss_kind=a.loss,
                 inner_k=a.inner_k, hard_p=a.hard_p)

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
            # these arms. CHAMPION TRACK fix (2026-09-01): keep the val PAIRS —
            # the trajectory monitor (sudoku_monitor -> EV.run_batch) is
            # layout-aware and NEEDS them; only val20_eval is skipped (below).
            print("val20 monitor DISABLED for non-origin layout "
                  "(trajectory monitor + evaluator measure val)", flush=True)
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
    # val20_eval places at the ARC origin — valid only for origin-layout runs;
    # non-origin Sudoku arms use the layout-aware trajectory monitor instead.
    val20_ok = not (a.sudoku_extreme and a.sudoku_layout != "origin")

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
                      optax.adamw(sched, b2=a.beta2, weight_decay=a.wd))
    opt_state = opt.init(state)
    start_step = 0
    rng = k_run

    latest = out / "ckpt_latest.pkl"
    ema = jax.tree.map(lambda x: x, state) if a.ema > 0 else None   # eval-side EMA copy (the field's .999)
    if latest.exists():
        saved = E.load_ckpt(latest)
        state, opt_state = saved["state"], saved["opt_state"]
        start_step, rng = int(saved["step"]), jnp.asarray(saved["rng"])
        if a.ema > 0:
            ema = saved.get("state_ema") or jax.tree.map(lambda x: x, state)
        print(f"RESUMED from {latest} at step {start_step}", flush=True)
        # sportC1 §4.7: resume steps on disk -> resume-adjacent deaths are labeled automatically
        with open(out / "resumes.txt", "a") as rf:
            rf.write(f"{start_step}\n")
    elif a.init_from:
        saved = E.load_ckpt(a.init_from)
        src = saved["state"]
        assert jax.tree.structure(src["model"]) == jax.tree.structure(state["model"]), \
            "--init-from: model tree mismatch (config differs)"
        state = dict(state, model=src["model"])
        if np.asarray(src["table"]).shape == np.asarray(state["table"]).shape:
            state["table"] = src["table"]
        opt_state = opt.init(state)
        if a.ema > 0:
            ema = jax.tree.map(lambda x: x, state)
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

    def _ema(ema_tree, new_state):
        if ema_tree is None:
            return None
        mu = a.ema
        return jax.tree.map(lambda e, s: e * mu + s * (1.0 - mu), ema_tree, new_state)

    @jax.jit
    def step_fn(state, opt_state, rng, ema_tree):
        rng, rng_step = jax.random.split(rng)
        (loss, aux), grads = _sample_and_loss(state, rng_step, a.batch)
        updates, opt_state2 = opt.update(grads, opt_state, state)
        new_state = optax.apply_updates(state, updates)
        return new_state, opt_state2, loss, aux, rng, _ema(ema_tree, new_state)

    # DP path (P11-EXT registration 2026-08-11): pmap over local devices,
    # per-device batch shard (global batch preserved: a.batch // n_dev rows
    # each), gradients pmean'd, replicas updated in lockstep. Enabler for
    # litepod-8/16 single-model pretrains (the 5-75M track).
    n_dev = jax.local_device_count() if a.dp else 1
    if a.dp:
        assert a.batch % n_dev == 0, f"batch {a.batch} % devices {n_dev} != 0"

        @functools.partial(jax.pmap, axis_name="dp")
        def step_fn_dp(state, opt_state, rng, ema_tree):
            rng, rng_step = jax.random.split(rng)
            rng_step = jax.random.fold_in(rng_step, jax.lax.axis_index("dp"))
            (loss, aux), grads = _sample_and_loss(
                state, rng_step, a.batch // n_dev)
            grads = jax.lax.pmean(grads, "dp")
            loss = jax.lax.pmean(loss, "dp")
            aux = jax.lax.pmean(aux, "dp")
            updates, opt_state2 = opt.update(grads, opt_state, state)
            new_state = optax.apply_updates(state, updates)
            return new_state, opt_state2, loss, aux, rng, _ema(ema_tree, new_state)

    if a.sot and cfg.cell_kind != "trm":
        return run_sot_rg(a, cfg, state, opt, opt_state, sched, start_step, rng, dev, n_tasks,
                          val, out, latest, ema, n_dev)
    if a.sot:
        return run_sot(a, cfg, state, opt, opt_state, sched, start_step, rng, dev, n_tasks,
                       val, out, latest, ema, n_dev)

    if a.dp:
        rep = lambda tree: jax.tree.map(
            lambda x: jnp.stack([x] * n_dev), tree)
        state_r, opt_r = rep(state), rep(opt_state)
        ema_r = rep(ema) if ema is not None else None
        rng_r = jax.random.split(rng, n_dev)
        print(f"DP: {n_dev} devices, {a.batch // n_dev} rows/device", flush=True)
    metrics_f = open(out / "metrics.jsonl", "a")
    t_block = time.time()
    for i in range(start_step, a.steps):
        if a.dp:
            state_r, opt_r, loss_r, aux_r, rng_r, ema_r = step_fn_dp(state_r, opt_r, rng_r, ema_r)
            loss = jax.tree.map(lambda x: x[0], loss_r)
            aux = jax.tree.map(lambda x: x[0], aux_r)
            if (i + 1) % a.val_every == 0 or (i + 1) % a.ckpt_every == 0 \
                    or i + 1 == a.steps or (a.monitor_every and (i + 1) % a.monitor_every == 0):
                state = jax.tree.map(lambda x: x[0], state_r)
                opt_state = jax.tree.map(lambda x: x[0], opt_r)
                rng = rng_r[0]
                ema = jax.tree.map(lambda x: x[0], ema_r) if ema_r is not None else None
        else:
            state, opt_state, loss, aux, rng, ema = step_fn(state, opt_state, rng, ema)

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
            if not math.isfinite(rec["loss"]):
                # sportC1 §4.1 EARLY-NaN ABORT (the pilot trained 25-30k steps on
                # non-finite parameters per death): stop here; the chain's one-shot
                # amputation banks the last FINITE grid and labels the arm STOPPED.
                nan_abort(out, i + 1, metrics_f)

        if val20_ok and ((i + 1) % a.val_every == 0 or i + 1 == a.steps):
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
            if ema is not None:   # the EMA weights' val (the field's headline weights; select_ckpt --key)
                vk = [k for k in mon if k.startswith("val_t")][0]
                mon[vk + "_ema"] = sudoku_monitor(ema, cfg, val[0][2])[vk]
            mon["step"] = i + 1; mon["wall_s"] = round(time.time() - t_m, 1)
            metrics_f.write(json.dumps({"monitor": mon}) + "\n"); metrics_f.flush()
            print("  MONITOR step %d: %s (%ss)" % (i + 1, " ".join(f"{k} {v:.3f}" for k, v in mon.items()
                  if isinstance(v, float) and k not in ("wall_s",)), mon["wall_s"]), flush=True)
            t_block = time.time()

        if (i + 1) % a.ckpt_every == 0 or i + 1 == a.steps:
            payload = {"state": state, "opt_state": opt_state, "step": i + 1,
                       "rng": np.asarray(rng), "config": dataclasses.asdict(cfg),
                       **({"state_ema": ema} if ema is not None else {})}
            E.save_ckpt(latest, payload)
            if (i + 1) % (5 * a.ckpt_every) == 0 or i + 1 == a.steps:
                E.save_ckpt(out / f"ckpt_{i+1:06d}.pkl", payload)

    metrics_f.close()
    print("DONE", flush=True)


def nan_abort(out: Path, step: int, metrics_f):
    (out / "NAN_ABORT.txt").write_text(f"NAN-ABORT at step {step} (non-finite loss at a log row; early-NaN abort, sportC1 §4.1)\n")
    metrics_f.close()
    print(f"NAN-ABORT step {step}: non-finite loss — stopping (the chain amputates to the last finite grid)", flush=True)
    sys.exit(3)


def run_sot_rg(a, cfg, state, opt, opt_state, sched, start_step, rng, dev, n_tasks, val, out, latest, ema, n_dev):
    """sportC2 R1 — the PERSISTENT-CARRY loop on OUR cell (Freethink 2026-09-03 X-4 graft (c)):
    every optimizer step advances every row of a persistent carry (x, y-probs, z, task, steps,
    halted) by ONE T-step segment of the equilibrium loop with deep supervision at every step
    (+ the registered FPA rows from the solution), takes the step, and replaces rows that are
    EXACT at the segment's end (the verifier's halt; on Sudoku exact == valid by uniqueness) or
    have run --sot-segments segments (4 x T = the t=64 horizon) with fresh samples. Fresh rows
    start as the offline loop's rows (VOID / anchor / RI y0 per build_y0_rows, no latent) on the
    trained ramp; carried rows continue their own (y, z) under the FINAL map (t_norm 1). Carried
    y/z are detached. The carry is NOT checkpointed (a resume restarts every row, labeled)."""
    from qhrrn2.objective import pair_loss as _pair_loss
    from qhrrn2 import model as M
    from qhrrn2 import grid as G
    T = cfg.T
    B = a.batch // n_dev
    use_ema = ema is not None
    # the carried latent's shape from one dummy forward (no latent in)
    x_d, y_d, t_d, _ = E.sample_batch(jax.random.PRNGKey(0), dev, n_tasks, 1)
    void_p = jax.nn.one_hot(jnp.full(x_d[0].shape, G.VOID, jnp.int32), M.VOCAB).transpose(2, 0, 1)
    z_shape = M.forward_fields(state["model"], cfg, M.build_fields_soft(x_d[0], void_p), t_norm=0.0, tau=a.tau,
                               rng=None, task_vec=state["table"][t_d[0]], z_in=None).z_fine.shape
    H, W = x_d[0].shape

    def init_carry():
        return dict(x=jnp.zeros((B, H, W), jnp.int32), ysol=jnp.zeros((B, H, W), jnp.int32),
                    y=jnp.zeros((B, M.VOCAB, H, W), jnp.float32),
                    z=jnp.zeros((B,) + tuple(z_shape), jnp.float32), t=jnp.zeros((B,), jnp.int32),
                    steps=jnp.zeros((B,), jnp.int32), halted=jnp.ones((B,), bool))

    def sot_step(state, opt_state, carry, rng, ema_tree):
        nk = 7 if a.ri_p > 0 else 5
        rng, rng_step = jax.random.split(rng)
        keys = jax.random.split(rng_step, nk)
        k_batch, k_loss, k_a1, k_a2, k_a3 = keys[:5]
        x_f, y_f, t_f, _ = E.sample_batch(k_batch, dev, n_tasks, B)
        yp_f = E.build_y0_rows(k_a1, k_a2, k_a3, y_f, a.anchor_p, a.anchor_eps, ri_p=a.ri_p,
                               k_r1=keys[5] if a.ri_p > 0 else None, k_r2=keys[6] if a.ri_p > 0 else None)
        y0_f = (jnp.broadcast_to(void_p, (B,) + void_p.shape) if yp_f is None
                else jax.nn.one_hot(jnp.asarray(yp_f, jnp.int32), M.VOCAB).transpose(0, 3, 1, 2))
        h = carry["halted"]
        x = jnp.where(h[:, None, None], x_f, carry["x"])
        yb = jnp.where(h[:, None, None], y_f, carry["ysol"])   # the loss target travels with the row
        y0 = jnp.where(h[:, None, None, None], y0_f, carry["y"])
        z0 = jnp.where(h.reshape((B,) + (1,) * len(z_shape)), jnp.zeros_like(carry["z"]), carry["z"])
        tb = jnp.where(h, t_f, carry["t"])
        steps = jnp.where(h, 0, carry["steps"])
        tsel = jnp.where(h, -1.0, 1.0)          # fresh -> the trained ramp; carried -> the final map
        row_keys = jax.random.split(k_loss, B)

        def loss_fn(st):
            def one(xx, yy, kk, tt, yy0, zz0, ts, fr):
                tot, aux = _pair_loss(st["model"], cfg, xx, yy, tau=a.tau, rng=kk, task_vec=st["table"][tt],
                                      y0_probs=yy0, z0=zz0, t_norm_fixed=ts, z_fresh=fr)
                return tot, aux
            losses, aux = jax.vmap(one)(x, yb, row_keys, tb, y0, z0, tsel, h)
            scal = {k: jnp.mean(v) for k, v in aux.items() if k not in ("carry_y", "carry_z", "exact_last")}
            return jnp.mean(losses), (scal, aux["carry_y"], aux["carry_z"], aux["exact_last"])
        (loss, (scal, cy, cz, ex)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state)
        if n_dev > 1:
            grads = jax.lax.pmean(grads, "dp"); loss = jax.lax.pmean(loss, "dp"); scal = jax.lax.pmean(scal, "dp")
        updates, opt_state = opt.update(grads, opt_state, state)
        state = optax.apply_updates(state, updates)
        if use_ema:
            ema_tree = jax.tree.map(lambda e, s_: e * a.ema + s_ * (1.0 - a.ema), ema_tree, state)
        steps = steps + 1
        halted = ex | (steps >= a.sot_segments)
        carry = dict(x=x, ysol=yb, y=jax.lax.stop_gradient(cy), z=jax.lax.stop_gradient(cz), t=tb, steps=steps, halted=halted)
        scal = dict(scal, halt_frac=jnp.mean(halted.astype(jnp.float32)), mean_steps=jnp.mean(steps.astype(jnp.float32)),
                    train_exact=jnp.mean(ex.astype(jnp.float32)), fresh_frac=jnp.mean(h.astype(jnp.float32)))
        return state, opt_state, carry, loss, scal, rng, ema_tree

    if n_dev > 1:
        step = jax.pmap(lambda st, os_, c, r, e: sot_step(st, os_, c, jax.random.fold_in(r, jax.lax.axis_index("dp")), e),
                        axis_name="dp")
        rep = lambda tree: jax.tree.map(lambda v: jnp.stack([v] * n_dev), tree)
        state, opt_state = rep(state), rep(opt_state)
        ema = rep(ema) if use_ema else None
        carry = rep(init_carry()); rng = jax.random.split(rng, n_dev)
        first = lambda tree: jax.tree.map(lambda v: v[0], tree)
        print(f"SOT-RG DP: {n_dev} devices, {B} rows/device; segments <= {a.sot_segments}; carry {'reset (resume)' if start_step else 'fresh'}", flush=True)
    else:
        step = jax.jit(sot_step)
        carry = init_carry(); first = lambda tree: tree
        print(f"SOT-RG: {B} rows; segments <= {a.sot_segments}; carry {'reset (resume)' if start_step else 'fresh'}", flush=True)
    metrics_f = open(out / "metrics.jsonl", "a")
    t_block = time.time()
    for i in range(start_step, a.steps):
        state, opt_state, carry, loss, scal, rng, ema = step(state, opt_state, carry, rng, ema)
        if (i + 1) % a.log_every == 0 or i + 1 == a.steps:
            dt = time.time() - t_block; sps = a.log_every / dt if dt > 0 else 0.0
            sc = first(scal); lv = float(first(loss))
            rec = {"step": i + 1, "loss": lv, "ce_in": float(sc["ce_in_last"]),
                   **({"fpa_ce": float(sc["fpa_ce_last"])} if "fpa_ce_last" in sc else {}),
                   "I_total": float(sc["flux_last"]) * cfg.scales, "A_total": float(sc["flux_attn_last"]) * cfg.scales,
                   "rule_H": float(sc["rule_entropy_last"]), "train_exact": float(sc["train_exact"]),
                   "halt_frac": float(sc["halt_frac"]), "mean_steps": float(sc["mean_steps"]), "fresh_frac": float(sc["fresh_frac"]),
                   "lr": float(sched(i + 1)), "steps_per_sec": round(sps, 3), "t": time.strftime("%Y-%m-%dT%H:%M:%S")}
            metrics_f.write(json.dumps(rec) + "\n"); metrics_f.flush()
            print(f"step {i+1:6d}  loss {lv:.4f}  ce {rec['ce_in']:.4f}  I {rec['I_total']:.1f}  A {rec['A_total']:.1f}  "
                  f"exact {rec['train_exact']:.3f}  halt {rec['halt_frac']:.2f}  segs {rec['mean_steps']:.2f}  {sps:.2f} it/s", flush=True)
            t_block = time.time()
            if not math.isfinite(lv):
                nan_abort(out, i + 1, metrics_f)
        do_mon = a.monitor_every and val and ((i + 1) % a.monitor_every == 0 or i + 1 == a.steps)
        do_ck = (i + 1) % a.ckpt_every == 0 or i + 1 == a.steps
        if do_mon or do_ck:
            st1, os1, rng1 = first(state), first(opt_state), first(rng)
            ema1 = first(ema) if use_ema else None
        if do_mon:
            t_m = time.time()
            mon = sudoku_monitor(st1, cfg, val[0][2])
            if use_ema:
                vk = [k for k in mon if k.startswith("val_t")][0]
                mon[vk + "_ema"] = sudoku_monitor(ema1, cfg, val[0][2])[vk]
            mon["step"] = i + 1; mon["wall_s"] = round(time.time() - t_m, 1)
            metrics_f.write(json.dumps({"monitor": mon}) + "\n"); metrics_f.flush()
            print("  MONITOR step %d: %s (%ss)" % (i + 1, " ".join(f"{k} {v:.3f}" for k, v in mon.items()
                  if isinstance(v, float) and k != "wall_s"), mon["wall_s"]), flush=True)
            t_block = time.time()
        if do_ck:
            payload = {"state": st1, "opt_state": os1, "step": i + 1, "rng": np.asarray(rng1),
                       "config": dataclasses.asdict(cfg), **({"state_ema": ema1} if use_ema else {})}
            E.save_ckpt(latest, payload)
            if (i + 1) % (5 * a.ckpt_every) == 0 or i + 1 == a.steps:
                E.save_ckpt(out / f"ckpt_{i+1:06d}.pkl", payload)
    metrics_f.close()
    print("DONE", flush=True)


def run_sot(a, cfg, state, opt, opt_state, sched, start_step, rng, dev, n_tasks, val, out, latest, ema, n_dev):
    """X0 — SEGMENTED ONLINE TRAINING (HRM/TRM/EqR's training loop, ported from
    trm.py + losses.py + pretrain.py, read 2026-09-02): a persistent carry of B
    rows (x, y, z_H, z_L, steps, halted); every optimizer step advances EVERY row
    by ONE outer segment (H_cycles x (L_cycles+1) stack passes, grad through the
    last H-cycle), supervises that segment's logits (+ 0.5 x the halting BCE under
    --act), takes the step, and replaces halted rows with fresh samples (fresh
    z0 = the fixed buffers, or RI N(0, sigma)). Halting: steps >= T, or under
    --act sigmoid(q_halt) > .5 subject to TRM's exploration rule (with prob
    --halt-explore a row may not halt before a uniform 2..T threshold). EMA as
    the offline loop. The carry is NOT checkpointed: a resume restarts every row
    (labeled in the log). Eval never halts early (D outer steps)."""
    T = cfg.T
    H = W = cfg.canvas
    hw = H * W
    S = TC.seq_len(cfg, hw); hid = cfg.trm_hidden
    B = a.batch // n_dev
    use_ema = ema is not None

    def init_carry():
        return dict(x=jnp.zeros((B, H, W), jnp.int32), y=jnp.zeros((B, H, W), jnp.int32),
                    z=jnp.zeros((B, 2, S, hid), jnp.float32), steps=jnp.zeros((B,), jnp.int32),
                    halted=jnp.ones((B,), bool))

    def sot_step(state, opt_state, carry, rng, ema_tree):
        rng, k_s, k_r, k_n, k_e1, k_e2 = jax.random.split(rng, 6)
        x_f, y_f, _, _ = E.sample_batch(k_s, dev, n_tasks, B)
        h = carry["halted"]
        x = jnp.where(h[:, None, None], x_f, carry["x"])
        y = jnp.where(h[:, None, None], y_f, carry["y"])
        z_fresh = jax.vmap(lambda k: TC.z0(cfg, hw, rng=k))(jax.random.split(k_r, B))
        z = jnp.where(h[:, None, None, None], z_fresh, carry["z"])
        steps = jnp.where(h, 0, carry["steps"])
        keys = jax.random.split(k_n, B)

        def loss_fn(st):
            p = st["model"]["trm"]

            def one(xx, yy, zz, kk):
                emb = TC.embed(p, cfg, xx)
                zH, zL = TC.segment(p, cfg, emb, zz[0], zz[1], rng=kk if cfg.trm_beta > 0 else None)
                logits, q = TC.readout(p, cfg, zH, xx.shape)
                logp = log_stablemax(logits) if cfg.loss_kind == "stablemax" else jax.nn.log_softmax(logits, axis=-1)
                ce = -jnp.mean(jnp.take_along_axis(logp, yy[..., None], axis=-1)[..., 0])
                correct = jnp.all(jnp.argmax(logits, axis=-1) == yy)
                return ce, q, correct, jnp.stack([zH, zL])
            ce, q, corr, z_new = jax.vmap(one)(x, y, z, keys)
            lm = jnp.mean(ce)
            q_halt = q[:, 0]
            q_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(q_halt, corr.astype(jnp.float32)))
            total = lm + (0.5 * q_loss if a.act else 0.0)
            return total, dict(ce=lm, q_loss=q_loss, train_exact=jnp.mean(corr.astype(jnp.float32)),
                               q_halt=q_halt, z_new=z_new)
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state)
        scal = dict(ce=aux["ce"], q_loss=aux["q_loss"], train_exact=aux["train_exact"])
        if n_dev > 1:
            grads = jax.lax.pmean(grads, "dp"); loss = jax.lax.pmean(loss, "dp"); scal = jax.lax.pmean(scal, "dp")
        updates, opt_state = opt.update(grads, opt_state, state)
        state = optax.apply_updates(state, updates)
        if use_ema:
            ema_tree = jax.tree.map(lambda e, s_: e * a.ema + s_ * (1.0 - a.ema), ema_tree, state)
        steps = steps + 1
        is_last = steps >= T
        halted = is_last
        if a.act:
            explore = jax.random.uniform(k_e1, (B,)) < a.halt_explore
            min_halt = jnp.where(explore, jax.random.randint(k_e2, (B,), 2, T + 1), 0)
            halted = (is_last | (aux["q_halt"] > 0)) & (steps >= min_halt)
        carry = dict(x=x, y=y, z=jax.lax.stop_gradient(aux["z_new"]), steps=steps, halted=halted)
        scal["halt_frac"] = jnp.mean(halted.astype(jnp.float32))
        scal["mean_steps"] = jnp.mean(steps.astype(jnp.float32))
        return state, opt_state, carry, loss, scal, rng, ema_tree

    if n_dev > 1:
        step = jax.pmap(lambda st, os_, c, r, e: sot_step(st, os_, c, jax.random.fold_in(r, jax.lax.axis_index("dp")), e),
                        axis_name="dp")
        rep = lambda tree: jax.tree.map(lambda v: jnp.stack([v] * n_dev), tree)
        state, opt_state = rep(state), rep(opt_state)
        ema = rep(ema) if use_ema else None
        carry = rep(init_carry()); rng = jax.random.split(rng, n_dev)
        first = lambda tree: jax.tree.map(lambda v: v[0], tree)
        print(f"SOT DP: {n_dev} devices, {B} rows/device; carry {'reset (resume)' if start_step else 'fresh'}", flush=True)
    else:
        step = jax.jit(sot_step)
        carry = init_carry(); first = lambda tree: tree
        print(f"SOT: {B} rows; carry {'reset (resume)' if start_step else 'fresh'}", flush=True)
    metrics_f = open(out / "metrics.jsonl", "a")
    t_block = time.time()
    for i in range(start_step, a.steps):
        state, opt_state, carry, loss, scal, rng, ema = step(state, opt_state, carry, rng, ema)
        if (i + 1) % a.log_every == 0 or i + 1 == a.steps:
            dt = time.time() - t_block; sps = a.log_every / dt if dt > 0 else 0.0
            sc = first(scal); lv = float(first(loss))
            rec = {"step": i + 1, "loss": lv, "ce_in": float(sc["ce"]), "q_loss": float(sc["q_loss"]),
                   "train_exact": float(sc["train_exact"]), "halt_frac": float(sc["halt_frac"]),
                   "mean_steps": float(sc["mean_steps"]), "I_total": 0.0, "A_total": 0.0, "rule_H": 0.0,
                   "lr": float(sched(i + 1)), "steps_per_sec": round(sps, 3), "t": time.strftime("%Y-%m-%dT%H:%M:%S")}
            metrics_f.write(json.dumps(rec) + "\n"); metrics_f.flush()
            print(f"step {i+1:6d}  loss {lv:.4f}  ce {rec['ce_in']:.4f}  q {rec['q_loss']:.4f}  "
                  f"train_exact {rec['train_exact']:.3f}  halt {rec['halt_frac']:.2f}  {sps:.2f} it/s", flush=True)
            t_block = time.time()
            if not math.isfinite(lv):
                nan_abort(out, i + 1, metrics_f)
        do_mon = a.monitor_every and val and ((i + 1) % a.monitor_every == 0 or i + 1 == a.steps)
        do_ck = (i + 1) % a.ckpt_every == 0 or i + 1 == a.steps
        if do_mon or do_ck:
            st1, os1, rng1 = first(state), first(opt_state), first(rng)
            ema1 = first(ema) if use_ema else None
        if do_mon:
            t_m = time.time()
            mon = sudoku_monitor(st1, cfg, val[0][2])
            if use_ema:
                vk = [k for k in mon if k.startswith("val_t")][0]
                mon[vk + "_ema"] = sudoku_monitor(ema1, cfg, val[0][2])[vk]
            mon["step"] = i + 1; mon["wall_s"] = round(time.time() - t_m, 1)
            metrics_f.write(json.dumps({"monitor": mon}) + "\n"); metrics_f.flush()
            print("  MONITOR step %d: %s (%ss)" % (i + 1, " ".join(f"{k} {v:.3f}" for k, v in mon.items()
                  if isinstance(v, float) and k != "wall_s"), mon["wall_s"]), flush=True)
            t_block = time.time()
        if do_ck:
            payload = {"state": st1, "opt_state": os1, "step": i + 1, "rng": np.asarray(rng1),
                       "config": dataclasses.asdict(cfg), **({"state_ema": ema1} if use_ema else {})}
            E.save_ckpt(latest, payload)
            if (i + 1) % (5 * a.ckpt_every) == 0 or i + 1 == a.steps:
                E.save_ckpt(out / f"ckpt_{i+1:06d}.pkl", payload)
    metrics_f.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
