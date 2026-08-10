# Ledger: E4 committed-rule transport ([H-6'] / CI-10) + cluster L
# stationary-flux transport meter (Research_Brainstorm 2026-08-10).
# Instrument ONLY — no solve path touched; GT is used only to score and for
# the oracle-retention diagnostic (labeled), exactly as in the batteries.
#
# Per task (keyhole arm-A fit, the battery protocol):
#   1. COMMIT: trace each SUPPORT input under the fitted map; collect the
#      stationary rule_q two ways: cold-start ("cold") and truth-anchored
#      ("truth": yprev_init = y_s, final-map steps). Aggregate over supports
#      (mean -> soft; per-slot argmax one-hot -> hard). Records cross-support
#      rule agreement (do supports even agree on the rule?).
#   2. DECODE each query cold-start x {baseline, soft-override, hard-override}
#      -> exact@T (reachability under the committed rule).
#   3. RETAIN: GT-oracle retention (8 final-map steps) x the same three arms
#      (does the committed rule enlarge/hold the truth basin?).
#   4. L-METER: stationary spectra I_s/A_s on supports and query; mismatch
#      m = mean_s |I_s(q) - mean_supports I_s| per arm. Registered question:
#      does the committed rule REDUCE m, and does reduction co-occur with
#      conversion? (E4's named test + the L instrument's first dataset.)
"""
  python tools/probe_e4.py --ckpt runs/pretrain9_d/ckpt_latest.pkl \
      --tasks ca_X,ca_Y --out runs/e4_p9d
"""
from __future__ import annotations

import argparse
import functools
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
from qhrrn2 import episodic as E
from qhrrn2 import grid as G
from qhrrn2 import model as M
from qhrrn2.config import Config


@functools.lru_cache(maxsize=64)
def _fwd_eq_ov(cfg: Config, tau: float, t_norm: float, use_ov: bool):
    @jax.jit
    def fwd(params, x_can, y_probs, task_vec, z_c, q_ov):
        z_in = None if z_c.ndim == 1 else z_c
        return M.forward_fields(params, cfg,
                                M.build_fields_soft(x_can, y_probs),
                                t_norm=t_norm, tau=tau, rng=None,
                                task_vec=task_vec, z_in=z_in,
                                rule_override=q_ov if use_ov else None)
    return fwd


def trace_ov(params, cfg: Config, x_grid, *, tau: float, task_vec,
             t_total: int, rule_override=None, yprev_init=None,
             skip_trained: bool = False):
    """probe_e1e3.trace (eq branch) + rule_override + flux capture."""
    assert cfg.equilibrium, "E4 probe is eq-only"
    x_can = jnp.asarray(G.place(np.asarray(x_grid)), dtype=jnp.int32)
    y = (jax.nn.one_hot(jnp.full((G.CANVAS, G.CANVAS), G.VOID, jnp.int32),
                        M.VOCAB).transpose(2, 0, 1)
         if yprev_init is None else
         jax.nn.one_hot(jnp.asarray(yprev_init, dtype=jnp.int32),
                        M.VOCAB).transpose(2, 0, 1))
    eta = jax.nn.sigmoid(params["eq"]["eta"])
    eta_z = jax.nn.sigmoid(params["eq"]["eta_z"])
    use_ov = rule_override is not None
    q_ov = (jnp.asarray(rule_override) if use_ov
            else jnp.zeros((cfg.M, cfg.K)))
    z_c = None
    steps = []
    for t in range(t_total):
        t_norm = 1.0 if skip_trained else min(t, cfg.T - 1) / max(cfg.T - 1, 1)
        out = _fwd_eq_ov(cfg, tau, float(t_norm), use_ov)(
            params, x_can, y, task_vec,
            z_c if z_c is not None else jnp.zeros(1), q_ov)
        z_c = out.z_fine if z_c is None else z_c + eta_z * (out.z_fine - z_c)
        pcan = jax.nn.softmax(out.logits, axis=-1).transpose(2, 0, 1)
        y = y + eta * (pcan - y)
        canvas = np.asarray(jnp.argmax(out.logits, axis=-1))
        cands = M.size_candidates(x_can)
        p_h = M.size_mixture_probs(out.size_sel_h, out.size_h, cands[0])
        p_w = M.size_mixture_probs(out.size_sel_w, out.size_w, cands[1])
        h = int(jnp.argmax(p_h)) + 1
        w = int(jnp.argmax(p_w)) + 1
        pred = np.where(canvas[:h, :w] == G.VOID, 0,
                        canvas[:h, :w]).astype(np.int8)
        steps.append({"pred": pred,
                      "rule_q": np.asarray(out.rule_q),
                      "I_s": np.asarray(out.flux, dtype=np.float64),
                      "A_s": np.asarray(out.flux_attn, dtype=np.float64)})
    return steps


def exact(p, gt):
    g = np.asarray(gt)
    return bool(p.shape == g.shape and np.array_equal(p, g))


def retention(model, cfg, x, gt, tvj, ov, k=8):
    st = trace_ov(model, cfg, x, tau=1.0, task_vec=tvj, t_total=k,
                  rule_override=ov, yprev_init=G.place(np.asarray(gt)),
                  skip_trained=True)
    return all(exact(s["pred"], gt) for s in st)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tasks", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--val-every", type=int, default=50)
    ap.add_argument("--commit-k", type=int, default=4,
                    help="final-map steps per support for the stationary rule")
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

            # 1. COMMIT from supports
            sup_specs = []
            rules_truth, rules_cold = [], []
            for sx, sy in eps_list[0].support:
                cold = trace_ov(model, cfg, sx, tau=1.0, task_vec=tvj,
                                t_total=cfg.T)
                anch = trace_ov(model, cfg, sx, tau=1.0, task_vec=tvj,
                                t_total=a.commit_k,
                                yprev_init=G.place(np.asarray(sy)),
                                skip_trained=True)
                rules_cold.append(cold[-1]["rule_q"])
                rules_truth.append(anch[-1]["rule_q"])
                sup_specs.append({"I_s": anch[-1]["I_s"].round(4).tolist(),
                                  "A_s": anch[-1]["A_s"].round(4).tolist(),
                                  "cold_exact": exact(cold[-1]["pred"], sy)})
            q_soft = np.mean(rules_truth, axis=0)             # (M, K)
            hard_codes = np.argmax(q_soft, axis=-1)
            q_hard = np.eye(q_soft.shape[-1])[hard_codes]
            per_sup_codes = [np.argmax(r, axis=-1).tolist() for r in rules_truth]
            agree = float(np.mean([c == hard_codes.tolist()
                                   for c in per_sup_codes]))
            cold_codes = [np.argmax(r, axis=-1).tolist() for r in rules_cold]
            sup_I = np.mean([s["I_s"] for s in sup_specs], axis=0) \
                if sup_specs else None

            row = {"task": tid, "sel_step": sel[0],
                   "committed": {"hard_codes": hard_codes.tolist(),
                                 "per_support_codes": per_sup_codes,
                                 "cold_codes": cold_codes,
                                 "support_agreement": agree},
                   "supports": sup_specs, "queries": []}

            # 2-4. queries x {baseline, soft, hard}
            for qi, ep in enumerate(eps_list):
                if ep.query_y is None:
                    continue
                gt = np.asarray(ep.query_y)
                qrec = {}
                for arm, ov in (("baseline", None), ("soft", q_soft),
                                ("hard", q_hard)):
                    tr = trace_ov(model, cfg, ep.query_x, tau=1.0,
                                  task_vec=tvj, t_total=cfg.T,
                                  rule_override=ov)
                    I_q = tr[-1]["I_s"]
                    mism = (float(np.mean(np.abs(I_q - sup_I)))
                            if sup_I is not None else None)
                    qrec[arm] = {
                        "exact_T": exact(tr[-1]["pred"], gt),
                        "gt_retention": retention(model, cfg, ep.query_x,
                                                  gt, tvj, ov),
                        "I_s": I_q.round(4).tolist(),
                        "A_s": tr[-1]["A_s"].round(4).tolist(),
                        "flux_mismatch": mism}
                    if arm == "baseline":
                        qrec[arm]["self_codes"] = np.argmax(
                            tr[-1]["rule_q"], axis=-1).tolist()
                row["queries"].append(qrec)

            row["wall_s"] = round(time.time() - t0, 1)
            f.write(json.dumps(row) + "\n")
            f.flush()
            b = [q["baseline"]["exact_T"] for q in row["queries"]]
            h = [q["hard"]["exact_T"] for q in row["queries"]]
            rb = [q["baseline"]["gt_retention"] for q in row["queries"]]
            rh = [q["hard"]["gt_retention"] for q in row["queries"]]
            print(f"{tid} sel@{sel[0]} agree={agree:.2f} "
                  f"exact b{b}->h{h} ret b{rb}->h{rh} "
                  f"({row['wall_s']:.0f}s)", flush=True)
    print("E4 PROBE DONE", flush=True)


if __name__ == "__main__":
    main()
