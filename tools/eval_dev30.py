# Ledger: eval-1 (pre-registered 2026-08-01, BEFORE pretrain-1 results existed).
# Arms per dev-30 task, all with identity transform + placement offsets only
# (admissible orbit UNKNOWN on real tasks), LoO/MDL selection as in train.fit_loo,
# 600 steps, pass@2 = {LoO-selected params, final params}, solve = exact match
# on ALL test pairs (ARC rule):
#   A  fit e_t only (32 params; init = pretrained table mean)     lr 1e-2
#   B  e_t + color_bias (~208 params at d=16)                     lr 1e-2
#   C  full fine-tune from the pretrained checkpoint (ceiling)    lr 3e-3
#   D  full fit from RANDOM init (the null: pretraining adds 0)   lr 3e-3
# Registered comparisons: (1) C vs D — transfer at all (H-14 kill test);
# (2) A,B vs C — boundary-sufficiency (C10/C12 bet); (3) family stratification.
"""Dev-30 evaluation (eval-1).

  .venv/bin/python tools/eval_dev30.py --ckpt runs/pretrain1/ckpt_latest.pkl \
      --arms A,B,C,D --out runs/eval1
  .venv/bin/python tools/eval_dev30.py --arms D --out runs/eval1   # null only

Appends one JSON line per (task, arm) to <out>/results.jsonl — safe to re-run;
finished (task, arm) pairs are skipped.
"""
from __future__ import annotations

import argparse
import dataclasses
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
import optax

import dev30
from qhrrn2 import episodic as E
from qhrrn2 import grid as G
from qhrrn2 import train as T
from qhrrn2.config import Config
from qhrrn2.model import init_params
from qhrrn2.objective import batch_loss, pair_loss

ARM_LR = {"A": 1e-2, "B": 1e-2, "B2": 1e-2, "C": 3e-3, "D": 3e-3}
# B2 (eval-3, ledger 2026-08-02): B + canvas.sel unfrozen — the sel-saturation
# probe showed the pretrained selection prior never flips through the e_t
# path; B2 optimizes the ~1k selection logits directly.


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None, help="pretrain checkpoint (arms A/B/C)")
    p.add_argument("--ckpt2", default=None,
                   help="secondary bulk for the layered two-prior protocol "
                        "(eval-5); implies arm L replacing --arms")
    p.add_argument("--arms", default="A,B,C,D")
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--val-every", type=int, default=50)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    # TTT-time prices (ledger 2026-08-02 val-20 ablation; None = checkpoint cfg)
    p.add_argument("--beta", type=float, default=None)
    p.add_argument("--beta-nl", type=float, default=None)
    p.add_argument("--vote", action="store_true",
                   help="eval-4: D4 orbit voting on attempt 1 (inference only)")
    p.add_argument("--alt-size", action="store_true",
                   help="attempt 2 = runner-up size at the attempt-1 checkpoint "
                        "(aliasing mitigation; measured: old attempt-2 added 0 "
                        "solves in 240 cells)")
    p.add_argument("--save-preds", action="store_true",
                   help="record predicted grids per attempt (error taxonomy, CC#2 D1)")
    p.add_argument("--d", type=int, default=16, help="model width for arm D / no-ckpt")
    p.add_argument("--tasks", default=None, help="comma list; default = dev-30")
    return p.parse_args()


def _label_tree(tree, arm: str):
    """'train'/'freeze' labels per leaf for optax.multi_transform."""
    def lab(path, _):
        keys = [getattr(k, "key", getattr(k, "idx", None)) for k in path]
        if arm in ("C", "D"):
            return "train"
        if keys and keys[0] == "tv":
            return "train"
        if arm in ("B", "B2") and len(keys) >= 2 and keys[1] == "color_bias":
            return "train"
        if arm == "B2" and len(keys) >= 3 and keys[1] == "canvas" and keys[2] == "sel":
            return "train"
        return "freeze"
    return jax.tree_util.tree_map_with_path(lab, tree)


@functools.lru_cache(maxsize=16)
def _arm_step(cfg: Config, arm: str, lr: float, wd: float, tau: float, has_tv: bool):
    """Jitted masked-AdamW step over {'model':…, 'tv':…}; batches are args."""
    def make_opt(trainable):
        return optax.multi_transform(
            {"train": optax.adamw(lr, weight_decay=wd), "freeze": optax.set_to_zero()},
            _label_tree(trainable, arm))

    @jax.jit
    def step(trainable, opt_state, rng, x_b, y_b):
        def loss_fn(tr):
            tv = tr.get("tv") if has_tv else None
            tvs = None if tv is None else jnp.broadcast_to(tv, (x_b.shape[0],) + tv.shape)
            return batch_loss(tr["model"], cfg, x_b, y_b, tau=tau, rng=rng,
                              task_vecs=tvs)
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(trainable)
        opt = make_opt(trainable)
        updates, opt_state2 = opt.update(grads, opt_state, trainable)
        return optax.apply_updates(trainable, updates), opt_state2, loss, aux
    return step, make_opt


def measure_flux(params, cfg, x_b, y_b, tau, tv):
    """(S,) mean last-step flux spectra over the support batch, deterministic."""
    tvs = None if tv is None else jnp.broadcast_to(tv, (x_b.shape[0],) + tv.shape)

    def one(x, y, t):
        _, aux = pair_loss(params, cfg, x, y, tau=tau, task_vec=t)
        return aux["flux_last"], aux["flux_attn_last"]
    if tvs is None:
        I, A = jax.vmap(lambda x, y: one(x, y, None))(x_b, y_b)
    else:
        I, A = jax.vmap(one)(x_b, y_b, tvs)
    return np.asarray(I.mean(0)).tolist(), np.asarray(A.mean(0)).tolist()


def _fit(arm, cfg, ckpt_state, episodes, *, steps, val_every, wd, tau, seed,
         snapshots=None):
    """The fitting core: LoO-validated arm fit. Returns a dict with the
    earliest-exact/best/final trainables, curves, and tv accessor.

    snapshots: optional list — when given, (step, tv_numpy) is appended at
    every validation point (E1 instrument, ledger 2026-08-08; default None
    leaves the deployed path byte-identical — equivalence test
    tests/test_probe_e1e3.py::test_snapshot_flag_inert)."""
    support = list(episodes[0].support)
    train_pairs, (val_x, val_y) = support[:-1], support[-1]
    x_b, y_b = T.pairs_to_batch(train_pairs, transforms=None, seed=seed)

    if arm == "D":
        model = init_params(jax.random.PRNGKey(seed), cfg)
        trainable, has_tv = {"model": model}, False
    else:
        model = jax.tree.map(jnp.asarray, ckpt_state["model"])
        tv0 = jnp.asarray(np.asarray(ckpt_state["table"]).mean(0))
        trainable, has_tv = {"model": model, "tv": tv0}, True

    step, make_opt = _arm_step(cfg, arm, ARM_LR[arm], wd, tau, has_tv)
    opt_state = make_opt(trainable).init(trainable)

    def tv_of(tr):
        return tr.get("tv") if has_tv else None

    rng = jax.random.PRNGKey(seed)
    best = {"trainable": trainable, "val_pix": -1.0, "val_exact": False,
            "step": 0, "loss": float("inf")}
    first_exact = None  # (trainable, step) at the EARLIEST val-exact checkpoint
    losses, val_curve = [], []
    for i in range(steps):
        rng, sub = jax.random.split(rng)
        trainable, opt_state, loss, _ = step(trainable, opt_state, sub, x_b, y_b)
        losses.append(float(loss))
        if (i + 1) % val_every == 0 or i + 1 == steps:
            exact, pix, _ = T.evaluate_pair(trainable["model"], cfg, val_x, val_y,
                                            tau=tau, task_vec=tv_of(trainable))
            val_curve.append((i + 1, round(pix, 4), bool(exact)))
            if snapshots is not None and tv_of(trainable) is not None:
                snapshots.append((i + 1, np.asarray(tv_of(trainable))))
            if exact and first_exact is None:
                first_exact = (trainable, i + 1)
            if (exact, pix, -losses[-1]) > (best["val_exact"], best["val_pix"],
                                            -best["loss"]):
                best = {"trainable": trainable, "val_pix": pix, "val_exact": exact,
                        "step": i + 1, "loss": losses[-1]}
    return {"best": best, "first_exact": first_exact, "final": trainable,
            "losses": losses, "val_curve": val_curve, "tv_of": tv_of,
            "x_b": x_b, "y_b": y_b}


def fit_arm(arm, cfg, ckpt_state, episodes, *, steps, val_every, wd, tau, seed,
            vote=False, save_preds=False, alt_size=False):
    """LoO/MDL fit of one arm on one task; returns (result dict, per-attempt preds)."""
    F = _fit(arm, cfg, ckpt_state, episodes, steps=steps, val_every=val_every,
             wd=wd, tau=tau, seed=seed)
    best, first_exact, trainable = F["best"], F["first_exact"], F["final"]
    losses, val_curve, tv_of = F["losses"], F["val_curve"], F["tv_of"]
    x_b, y_b = F["x_b"], F["y_b"]

    # Pass@2 (eval-3 rule, ledger 2026-08-02): attempt 1 = EARLIEST-val-exact
    # checkpoint (eval-2 measured MDL walking past the generalizing solution:
    # exact@50 -> selected@2000 -> query fail), attempt 2 = MDL-best; fallbacks
    # keep the eval-1/2 pair when no val-exact checkpoint exists.
    if first_exact is not None and first_exact[1] != best["step"]:
        attempts, attempt_rule = [first_exact[0], best["trainable"]], "earliest+mdl"
    elif first_exact is not None:
        attempts, attempt_rule = [first_exact[0], trainable], "earliest+final"
    else:
        attempts, attempt_rule = [best["trainable"], trainable], "mdl+final"
    per_pair, preds_rec = [], []
    for ep in episodes:
        bits = []
        pair_preds = []
        if alt_size:
            # both attempts from the attempt-1 checkpoint: primary size, then
            # the runner-up-size crop (falls back to primary when identical)
            att0 = attempts[0]
            p1, s1, p2, s2 = T.predict_alt_size(att0["model"], cfg, ep.query_x,
                                                tau=tau, task_vec=tv_of(att0))
            for pred, shape in ((p1, s1), (p2 if p2 is not None else p1,
                                           s2 if s2 is not None else s1)):
                ok = bool(ep.query_y is not None and shape == ep.query_y.shape
                          and np.array_equal(pred, ep.query_y))
                bits.append(ok)
                if save_preds:
                    pair_preds.append(np.asarray(pred).tolist())
            per_pair.append(bits)
            if save_preds:
                preds_rec.append(pair_preds)
            continue
        for i_att, att in enumerate(attempts):
            if vote and i_att == 0:
                # eval-4 (ledger 2026-08-02): inference-side D4 orbit voting at
                # the attempt-1 checkpoint — invert-then-vote is rule-consistent
                # under the joint transform; fit-time supervision untouched.
                d4 = [G.Transform(k=k) for k in range(8)]
                pred, shape = T.predict_voted(att["model"], cfg, ep.query_x,
                                              d4, tau=tau, task_vec=tv_of(att))
            else:
                pred, shape, _ = T.predict(att["model"], cfg, ep.query_x, tau=tau,
                                           task_vec=tv_of(att))
            ok = bool(ep.query_y is not None and shape == ep.query_y.shape
                      and np.array_equal(pred, ep.query_y))
            bits.append(ok)
            if save_preds:
                pair_preds.append(np.asarray(pred).tolist())
        per_pair.append(bits)
        if save_preds:
            preds_rec.append(pair_preds)
    solved_at1 = all(b[0] for b in per_pair)
    solved_pass2 = all(b[0] or b[1] for b in per_pair)

    I_sel, A_sel = measure_flux(best["trainable"]["model"], cfg, x_b, y_b, tau,
                                tv_of(best["trainable"]))
    return {
        "solved_pass2": solved_pass2, "solved_at1": solved_at1,
        "per_pair_bits": per_pair, "best_step": best["step"],
        **({"preds": preds_rec} if save_preds else {}),
        "best_val_pix": round(best["val_pix"], 4), "best_val_exact": best["val_exact"],
        "train_loss_at_best": round(best["loss"], 5),
        "attempt_rule": attempt_rule,
        "first_exact_step": None if first_exact is None else first_exact[1],
        "steps_to_val_exact": next((s for s, _, e in val_curve if e), None),
        "val_curve": val_curve,
        "I_s_selected": I_sel, "A_s_selected": A_sel,
    }


def fit_task_layered(cfg1, state1, cfg2, state2, episodes, *, steps, val_every,
                     wd, tau, seed, save_preds=False):
    """Eval-5 layered two-prior pass@2 (ledger 2026-08-05): attempt 1 = the
    PRIMARY bulk's earliest-exact/MDL checkpoint at primary size; attempt 2 =
    the SECONDARY bulk's prediction IF its fit reached LoO-exactness, ELSE the
    primary's runner-up-size variant, ELSE the primary MDL/final prediction."""
    F1 = _fit("A", cfg1, state1, episodes, steps=steps, val_every=val_every,
              wd=wd, tau=tau, seed=seed)
    F2 = _fit("A", cfg2, state2, episodes, steps=steps, val_every=val_every,
              wd=wd, tau=tau, seed=seed)
    tr1 = F1["first_exact"][0] if F1["first_exact"] else F1["best"]["trainable"]
    tr1_rule = "earliest" if F1["first_exact"] else "mdl"

    per_pair, preds_rec, att2_rules = [], [], []
    for ep in episodes:
        p1, s1, p1a, s1a = T.predict_alt_size(tr1["model"], cfg1, ep.query_x,
                                              tau=tau, task_vec=F1["tv_of"](tr1))
        if F2["first_exact"] is not None:
            tr2 = F2["first_exact"][0]
            p2, s2, _, _ = T.predict_alt_size(tr2["model"], cfg2, ep.query_x,
                                              tau=tau, task_vec=F2["tv_of"](tr2))
            rule2 = "bulk2-earliest"
        elif p1a is not None:
            p2, s2, rule2 = p1a, s1a, "alt-size"
        else:
            b = F1["best"]["trainable"]
            p2, s2, _ = T.predict(b["model"], cfg1, ep.query_x, tau=tau,
                                  task_vec=F1["tv_of"](b))
            s2 = tuple(p2.shape)
            rule2 = "mdl-fallback"
        att2_rules.append(rule2)
        bits = []
        for pred, shape in ((p1, s1), (p2, s2)):
            ok = bool(ep.query_y is not None and tuple(shape) == ep.query_y.shape
                      and np.array_equal(pred, ep.query_y))
            bits.append(ok)
        per_pair.append(bits)
        if save_preds:
            preds_rec.append([np.asarray(p1).tolist(), np.asarray(p2).tolist()])

    I_sel, A_sel = measure_flux(tr1["model"], cfg1, F1["x_b"], F1["y_b"], tau,
                                F1["tv_of"](tr1))
    return {
        "solved_pass2": all(b[0] or b[1] for b in per_pair),
        "solved_at1": all(b[0] for b in per_pair),
        "per_pair_bits": per_pair,
        **({"preds": preds_rec} if save_preds else {}),
        "attempt_rule": f"{tr1_rule}+{'/'.join(att2_rules)}",
        "first_exact_step": None if F1["first_exact"] is None else F1["first_exact"][1],
        "bulk2_first_exact_step": None if F2["first_exact"] is None else F2["first_exact"][1],
        "best_step": F1["best"]["step"],
        "best_val_pix": round(F1["best"]["val_pix"], 4),
        "best_val_exact": F1["best"]["val_exact"],
        "bulk2_best_val_pix": round(F2["best"]["val_pix"], 4),
        "train_loss_at_best": round(F1["best"]["loss"], 5),
        "steps_to_val_exact": next((s for s, _, e in F1["val_curve"] if e), None),
        "val_curve": F1["val_curve"],
        "I_s_selected": I_sel, "A_s_selected": A_sel,
    }


def main():
    a = parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "results.jsonl"

    done = set()
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add((r["task"], r["arm"]))
            except Exception:
                pass

    arms = [s.strip().upper() for s in a.arms.split(",") if s.strip()]

    def _load(path):
        saved = E.load_ckpt(path)
        defaults = Config()
        # Coerce to the dataclass's scalar types: checkpoints written before
        # the save_ckpt scalar fix carry 0-d ndarrays here (unhashable Config).
        c = Config(**{k: type(getattr(defaults, k))(v)
                      for k, v in saved["config"].items()})
        return saved["state"], c

    ckpt_state = cfg = state2 = cfg2 = None
    if a.ckpt2:
        arms = ["L"]  # layered two-prior protocol (eval-5)
        if not a.ckpt:
            sys.exit("--ckpt2 needs --ckpt (primary bulk)")
        state2, cfg2 = _load(a.ckpt2)
    if any(arm != "D" for arm in arms):
        if not a.ckpt:
            sys.exit("all arms except D need --ckpt")
        ckpt_state, cfg = _load(a.ckpt)
    cfg_d = cfg if cfg is not None else Config(d=a.d)

    if a.beta is not None or a.beta_nl is not None:
        repl = {}
        if a.beta is not None:
            repl["beta_flux"] = a.beta
        if a.beta_nl is not None:
            repl["beta_flux_nl"] = a.beta_nl
        if cfg is not None:
            cfg = dataclasses.replace(cfg, **repl)
        cfg_d = dataclasses.replace(cfg_d, **repl)

    task_ids = (a.tasks.split(",") if a.tasks else sorted(dev30.MANIFEST))
    fam = {t: dev30.MANIFEST.get(t, ("?", ""))[0] for t in task_ids}

    with open(results_path, "a") as f:
        for task_id in task_ids:
            episodes = G.load_task(task_id)
            for arm in arms:
                if (task_id, arm) in done:
                    print(f"skip {task_id} {arm} (done)", flush=True)
                    continue
                use_cfg = cfg_d if arm == "D" else cfg
                t0 = time.time()
                if arm == "L":
                    res = fit_task_layered(cfg, ckpt_state, cfg2, state2,
                                           episodes, steps=a.steps,
                                           val_every=a.val_every, wd=a.wd,
                                           tau=a.tau, seed=a.seed,
                                           save_preds=a.save_preds)
                else:
                    res = fit_arm(arm, use_cfg, ckpt_state, episodes, steps=a.steps,
                                  val_every=a.val_every, wd=a.wd, tau=a.tau,
                                  seed=a.seed, vote=a.vote,
                                  save_preds=a.save_preds, alt_size=a.alt_size)
                res.update({"task": task_id, "arm": arm, "family": fam[task_id],
                            "wall_s": round(time.time() - t0, 1),
                            "steps": a.steps, "seed": a.seed,
                            "beta": a.beta, "beta_nl": a.beta_nl, "vote": a.vote})
                f.write(json.dumps(res) + "\n")
                f.flush()
                print(f"{task_id} {fam[task_id]:<22} {arm}: "
                      f"pass2={res['solved_pass2']} at1={res['solved_at1']} "
                      f"best@{res['best_step']} val_pix {res['best_val_pix']} "
                      f"({res['wall_s']}s)", flush=True)
    print("EVAL DONE", flush=True)


if __name__ == "__main__":
    main()
