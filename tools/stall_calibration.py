# Ledger: sportC2 R3's instrument of record — CALIBRATION AT STALLS (Freethink 2026-09-03 §0.10,
# ground 7): on the evaluator's rating-stratified test puzzles, run the cold trajectory to t=64,
# split solved/stalled, and on the STALLED puzzles read (i) the correctness of the top-k most
# confident non-given cells (k=5; sportC1 maps: 70-72 % at mean confidence .96), (ii) the readout
# entropy at step 1 and at t=64, (iii) the confidently-wrong cell fraction; plus the solved-set
# control. Top-k accuracy alone is not a calibration statistic. Descriptive on any arm; a RULE only where a
# registration names it. Mirrors the evaluator's step (EV._step; inner_k honoured; --ema).
# Confidence uses the checkpoint's training normalization over digits 1..9.
# This measurement does not change the evaluator's softmax/one-hot feedback.
"""
  .venv/bin/python tools/stall_calibration.py --ckpt runs/X/ckpt.pkl --npz data/sudoku_extreme/sudoku_extreme_seed0.npz \
      --out runs/sxcalib_X [--ema] [--n 512] [--t 64] [--topk 5] [--hard-feedback]
  -> <out>/calib.json  (n, n_solved, n_stalled, topk_correct_stalled, topk_correct_solved, mean_conf_stalled,
                        entropy_step1, entropy_t_stalled, entropy_t_solved, conf_wrong_frac_stalled, cold)
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src")); sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, jax, jax.numpy as jnp
from qhrrn2 import episodic as E, grid as G, model as M, sudoku as SU, sudoku_extreme as SX
from qhrrn2.config import Config
from qhrrn2.objective import log_stablemax
import eval_sudoku_extreme as EV


def digit_probabilities(logits, loss_kind):
    """Training-normalized readout conditional on Sudoku digits, not VOID/black.

    Normalize the digit logits directly: renormalizing full-vocabulary softmax
    can underflow when a special-token logit dominates. StableMax must retain
    the original logit origin, because it is not translation invariant.
    """
    digits = jnp.asarray(logits[..., 1:10], dtype=jnp.float32)
    if loss_kind == "stablemax":
        return jnp.exp(log_stablemax(digits))
    if loss_kind == "softmax":
        return jax.nn.softmax(digits, axis=-1)
    raise ValueError(f"Unsupported checkpoint loss_kind: {loss_kind!r}")


def topk_empty_metrics(confidence, pred, solution, empty, topk):
    """Per-puzzle accuracy/confidence of up to topk initially empty cells.

    Preserve the existing argsort tie convention. Rows without empty cells have
    no local statistic and return NaN, excluded by the summary mean.
    """
    if topk < 1:
        raise ValueError("topk must be positive")
    conf = np.asarray(confidence).reshape(len(pred), -1)
    mask = np.asarray(empty, dtype=bool).reshape(len(pred), -1)
    masked = np.where(mask, conf, -1)
    correct = (np.asarray(pred) == np.asarray(solution)).reshape(len(pred), -1)
    accuracy, mean_conf = [], []
    for i, ranked in enumerate(np.argsort(masked, axis=1)[:, ::-1]):
        selected = ranked[mask[i, ranked]][:topk]
        accuracy.append(float(correct[i, selected].mean()) if len(selected) else np.nan)
        mean_conf.append(float(conf[i, selected].mean()) if len(selected) else np.nan)
    return np.asarray(accuracy), np.asarray(mean_conf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--npz", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--ema", action="store_true"); ap.add_argument("--n", type=int, default=512); ap.add_argument("--strat-seed", type=int, default=20260821)
    ap.add_argument("--t", type=int, default=64); ap.add_argument("--topk", type=int, default=5); ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--hard-feedback", action="store_true")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    saved = E.load_ckpt(a.ckpt); defaults = Config(); cfg = Config(**{k: type(getattr(defaults, k))(v) for k, v in saved["config"].items()})
    st = saved["state_ema"] if a.ema else saved["state"]; params = st["model"]; tvj = jnp.asarray(st["table"][0])
    eta, eta_z = (float(v) for v in M.eq_etas(params, cfg)); layout = cfg.sudoku_layout or "origin"; cv = SU.layout_canvas(layout)
    trm = cfg.cell_kind in ("trm", "dec"); K = 1 if trm else max(1, int(getattr(cfg, "inner_k", 1)))
    d = SX.load_prepared(a.npz); Q, A, R = d["test_q"], d["test_a"], d["test_rating"]
    ids = SX.stratified_subsample(R, a.n, a.strat_seed); t0 = time.time()
    tops_st, tops_sv, conf_st, ent1, ent_st, ent_sv, cw_st, solved_all = [], [], [], [], [], [], [], []
    for s0 in range(0, len(ids), a.batch):
        b = ids[s0:s0 + a.batch]; puz9 = Q[b].astype(np.int32); sol9 = A[b].astype(np.int32); ng = puz9 == 0; B = len(b)
        x_can = jnp.asarray(np.stack([SU.place_layout(g.astype(np.int8), layout) for g in puz9]), jnp.int32)
        void = jax.nn.one_hot(jnp.full((cv, cv), G.VOID, jnp.int32), M.VOCAB).transpose(2, 0, 1); y = jnp.broadcast_to(void, (B,) + void.shape); z = None
        for t in range(a.t):
            tn = 0.0 if trm else min(t, cfg.T - 1) / max(cfg.T - 1, 1)
            for _k in range(K):
                first = z is None
                logits, zf = EV._step(cfg, 1.0, float(tn), first)(params, x_can, y, tvj, jnp.zeros(1) if first else z)
                z = zf if first else z + eta_z * (zf - z)
            p = jax.nn.softmax(logits, axis=-1)
            pf = jax.nn.one_hot(jnp.argmax(logits, axis=-1), M.VOCAB) if a.hard_feedback else p
            y = y + eta * (pf.transpose(0, 3, 1, 2) - y)
            p9 = np.asarray(digit_probabilities(EV.layout_gather(logits, layout), cfg.loss_kind))
            ent = -(p9 * np.log(p9 + 1e-12)).sum(-1) / np.log(9)
            if t == 0: e1 = (ent * ng).sum((1, 2)) / np.maximum(ng.sum((1, 2)), 1)
        pred = np.asarray(EV.layout_gather(jnp.argmax(logits, axis=-1), layout)); pred = np.where(pred == G.VOID, 0, pred)
        solved = np.all((pred == sol9).reshape(B, -1), 1); conf = p9.max(-1)
        topc, mconf = topk_empty_metrics(conf, pred, sol9, ng, a.topk)
        e_t = (ent * ng).sum((1, 2)) / np.maximum(ng.sum((1, 2)), 1)
        cw = (((conf > 0.9) & (pred != sol9) & ng).sum((1, 2))) / np.maximum(ng.sum((1, 2)), 1)
        tops_st += topc[~solved].tolist(); tops_sv += topc[solved].tolist(); conf_st += mconf[~solved].tolist()
        ent1 += e1.tolist(); ent_st += e_t[~solved].tolist(); ent_sv += e_t[solved].tolist(); cw_st += cw[~solved].tolist(); solved_all += solved.tolist()
    f = lambda v: (float(np.mean(np.asarray(v)[np.isfinite(v)])) if np.isfinite(v).any() else None)
    res = dict(ckpt=a.ckpt, ema=bool(a.ema), hard_feedback=bool(a.hard_feedback), inner_k=K, n=int(len(ids)), t=a.t, topk=a.topk,
               confidence_normalization=cfg.loss_kind, confidence_domain="digit-conditional (1..9)",
               normalization_source=("checkpoint.config.loss_kind" if "loss_kind" in saved["config"] else "Config default for legacy checkpoint"),
               confidence_dtype="float32", confidence_threshold=0.9, cell_mask="initially empty",
               feedback_normalization=("one_hot_argmax" if a.hard_feedback else "softmax"),
               topk_scope="up to topk initially empty cells; exclude zero-empty rows from local means",
               cold=f(solved_all), n_solved=int(np.sum(solved_all)), n_stalled=int(len(solved_all) - np.sum(solved_all)),
               topk_correct_stalled=f(tops_st), topk_correct_solved=f(tops_sv), mean_conf_stalled=f(conf_st),
               entropy_step1=f(ent1), entropy_t_stalled=f(ent_st), entropy_t_solved=f(ent_sv), conf_wrong_frac_stalled=f(cw_st),
               wall_s=round(time.time() - t0, 1))
    (out / "calib.json").write_text(json.dumps(res, indent=1))
    print(f"CALIB n={res['n']} cold {100*res['cold']:.1f}% stalled {res['n_stalled']} | top{a.topk} correct on stalled {res['topk_correct_stalled']} (conf {res['mean_conf_stalled']}) solved {res['topk_correct_solved']} | entropy step1 {res['entropy_step1']} t{a.t} stalled {res['entropy_t_stalled']} | conf-wrong stalled {res['conf_wrong_frac_stalled']} ({res['wall_s']}s) | normalization {cfg.loss_kind} digit-conditional", flush=True)
    print("CALIB-DONE", flush=True)


if __name__ == "__main__":
    main()
