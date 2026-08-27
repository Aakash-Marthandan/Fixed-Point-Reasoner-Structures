# Ledger: S-PORT INSTRUMENT SUITE (H-33, the landscape-class law).
#
# THE POINT: run the SAME basin instruments that produced ARC's laws on a
# SINGLE-ATTRACTOR domain, and see which measured regularities are properties
# of our architecture and which are properties of ARC's landscape class.
# H-33 predicts three things flip on Sudoku: (i) retention ~ solve (ARC's
# retention-vs-reachability dissociation collapses when every instance has
# exactly one valid completion), (ii) RI/multi-init buys real accuracy (EqR's
# +24pp lever, which bought us nothing on ARC), (iii) basin-existence
# enrichment falls toward 1 (ARC: 31x — where pretraining built no basin, no
# search helps).
#
# COMPARABILITY IS BY CONSTRUCTION, NOT BY ASSERTION: this probe imports
# ARC's own instrument functions — probe_ladder.corrupt (uniform 0..9
# resample), probe_ladder.rung_rng (seeded per task/query/eps), the same
# LADDER_EPS rungs, the same 8 final-map stab steps, probe_e1e3.trace. The
# ONLY protocol difference is the one the domain forces: NO keyhole TTT fit.
# Sudoku's rule is universal, so the trained map is used directly with the
# single trained task row — which is also the EqR/FPRM no-TTT convention,
# making the comparison to their numbers the honest one.
#
# DOMAIN-NATIVE EXTRAS (measurements ARC cannot give us): constraint
# violations make correctness GRADED, so "basin exists but unreachable" and
# "no basin" become separable by degree rather than by a binary; givens
# preservation says whether the dynamics respect the boundary condition it
# was handed, or overwrite its own evidence.
"""
  .venv/bin/python tools/probe_sudoku.py --ckpt runs/sud_d16/ckpt_latest.pkl \
      --n 64 --out runs/sud_probe
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import jax.numpy as jnp

import probe_e1e3 as P
from probe_ladder import LADDER_EPS, corrupt, rung_rng, retained, trace_flux
from qhrrn2 import episodic as E
from qhrrn2 import grid as G
from qhrrn2 import sudoku as SU
from qhrrn2 import sudoku_extreme as SX
from qhrrn2.config import Config


def violations(g: np.ndarray) -> int:
    """Sudoku constraint violations: missing digits across rows/cols/boxes.
    0 == a valid solved grid. The graded correctness ARC cannot offer."""
    if g.shape != (SU.N, SU.N):
        return 3 * SU.N * SU.N
    want = set(range(1, SU.N + 1))
    v = 0
    for i in range(SU.N):
        v += len(want - set(g[i, :].tolist()))
        v += len(want - set(g[:, i].tolist()))
    for br in range(0, SU.N, SU.BOX):
        for bc in range(0, SU.N, SU.BOX):
            v += len(want - set(g[br:br + SU.BOX, bc:bc + SU.BOX].ravel().tolist()))
    return v


def givens_kept(pred: np.ndarray, puz: np.ndarray) -> tuple[int, int]:
    """(kept, total) — does the map respect the clues it was given?"""
    m = puz != SU.BLANK
    if pred.shape != puz.shape:
        return 0, int(m.sum())
    return int((pred[m] == puz[m]).sum()), int(m.sum())


def cells_correct(pred: np.ndarray, sol: np.ndarray) -> int:
    if pred.shape != sol.shape:
        return 0
    return int((pred == sol).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=64, help="held-out puzzles")
    ap.add_argument("--givens", type=int, default=30)
    ap.add_argument("--givens-list", default=None,
                    help="difficulty LADDER, e.g. '50,40,30' — the same "
                         "solution grids punched at each level (paired "
                         "within-grid contrast); overrides --givens")
    ap.add_argument("--eval-seed", type=int, default=99999,
                    help="MUST differ from the pretrain seed — these puzzles "
                         "are the held-out test set")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stab-steps", type=int, default=8)
    ap.add_argument("--k-init", type=int, default=16, help="multi-init draws")
    ap.add_argument("--seed", type=int, default=0)
    # SPRINT S2 (2026-08-21): benchmark puzzles + inference depth
    ap.add_argument("--t-total", type=int, default=None,
                    help="inference depth for the SOLVE step (default cfg.T)")
    ap.add_argument("--pairs-file", default=None,
                    help="prepared Sudoku-Extreme npz (overrides the generator)")
    ap.add_argument("--split", default="test", choices=["test", "val", "train"])
    ap.add_argument("--stratified", type=int, default=None,
                    help="rating-stratified subsample size from --pairs-file")
    ap.add_argument("--strat-seed", type=int, default=20260821)
    # RUNG 2 (2026-08-27): the d64 ladder SATURATED (S(eps)=1.00 through .4 on
    # every arm) — width out-ranged the instrument. Extended rungs via flag so
    # the registered default (ARC's LADDER_EPS) stays byte-identical; the rung-2
    # chain passes ".05 .1 .2 .4 .6 .8".
    ap.add_argument("--eps-rungs", default=None,
                    help="space-separated ladder rungs (default: ARC LADDER_EPS, unchanged)")
    a = ap.parse_args()
    eps_rungs = tuple(float(x) for x in a.eps_rungs.split()) if a.eps_rungs else tuple(LADDER_EPS)

    saved = E.load_ckpt(a.ckpt)
    defaults = Config()
    cfg = Config(**{k: type(getattr(defaults, k))(v)
                    for k, v in saved["config"].items()})
    state = saved["state"]
    model = state["model"]
    tvj = jnp.asarray(state["table"][0])      # the single "sudoku" task row
    if getattr(cfg, "sudoku_layout", "origin") not in ("", "origin"):
        # wave 2 (2026-08-22): the ARC-shared instruments (trace/retained) place
        # grids at the canvas origin; non-origin layouts are measured by the
        # batched evaluator (cold/retention/multi-init) — refuse, never mis-place.
        sys.exit(f"probe_sudoku: layout {cfg.sudoku_layout!r} is not origin — use "
                 "tools/eval_sudoku_extreme.py (--init solution for retention)")

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

    if a.pairs_file:
        d = SX.load_prepared(a.pairs_file)
        Q, A, R = d[f"{a.split}_q"], d[f"{a.split}_a"], d[f"{a.split}_rating"]
        sel = (SX.stratified_subsample(R, a.stratified, a.strat_seed)
               if a.stratified else np.arange(min(a.n, len(Q))))
        # tid carries the row index; 'level' carries the tdoku rating
        work = [(f"x{int(i):06d}", int(R[i]), Q[i].astype(np.int8), A[i].astype(np.int8))
                for i in sel]
    elif a.givens_list:
        levels = [int(g) for g in a.givens_list.split(",")]
        ladder = SU.sample_ladder(a.n, a.eval_seed, levels)
        # flatten to (tid, givens_level, puz, sol); tid carries the level so
        # resume-by-task stays correct across levels
        work = [(f"g{g}_{i:04d}", g, p_, s_)
                for g in levels for i, (p_, s_) in enumerate(ladder[g])]
    else:
        work = [(f"sud{i:04d}", a.givens, p_, s_)
                for i, (p_, s_) in enumerate(
                    SU.sample_pairs(a.n, seed=a.eval_seed, givens=a.givens))]
    rng_init = np.random.default_rng(a.seed + 4242)

    with open(results, "a") as f:
        for tid, level, puz, sol in work:
            if tid in done:
                continue
            t0 = time.time()

            # (1) SOLVE — cold start, the EqR/FPRM comparison metric.
            # trace_flux (not P.trace) so the per-scale spectra ride along:
            # S-R4's cross-domain H-34 test needs I_s on a NON-generative
            # domain, and P.trace records only pred/hw/H_q.
            tr = trace_flux(model, cfg, puz, tau=1.0, task_vec=tvj,
                            t_total=(a.t_total or cfg.T))
            pred = tr[-1]["pred"]
            solved = bool(pred.shape == sol.shape and np.array_equal(pred, sol))
            kept, ntot = givens_kept(pred, puz)

            # (2) RETENTION — hand it the solution (ARC's e3b semantics)
            r0 = retained(model, cfg, puz, sol, tvj, a.stab_steps)

            # (3) LADDER — identical rungs, corruption, seeding as ARC
            lad = {}
            for e in eps_rungs:
                rng = rung_rng(a.seed, tid, 0, e)
                cy = corrupt(sol, e, rng)
                st = P.trace(model, cfg, puz, tau=1.0, task_vec=tvj,
                             t_total=a.stab_steps, yprev_init=G.place(cy),
                             skip_trained=True)
                lad[str(e)] = bool(
                    st[-1]["pred"].shape == sol.shape
                    and np.array_equal(st[-1]["pred"], sol))

            # (4) MULTI-INIT — the RI dividend test (EqR's core lever)
            mi_hits, mi_best = 0, SU.N * SU.N
            for _ in range(a.k_init):
                y0 = rng_init.integers(0, 10, size=(SU.N, SU.N)).astype(np.int8)
                st = P.trace(model, cfg, puz, tau=1.0, task_vec=tvj,
                             t_total=a.stab_steps, yprev_init=G.place(y0),
                             skip_trained=True)
                p = st[-1]["pred"]
                if p.shape == sol.shape:
                    mi_hits += int(np.array_equal(p, sol))
                    mi_best = min(mi_best, int((p != sol).sum()))

            row = {
                "task": tid,
                "givens_level": level,
                "solved": solved,
                "gt_retention": bool(r0[-1]),
                "retained_per_step": r0,
                "q_ladder": lad,
                "violations": violations(pred),
                "cells_correct": cells_correct(pred, sol),
                "givens_kept": kept, "givens_total": ntot,
                "multi_init_hits": mi_hits, "multi_init_k": a.k_init,
                "multi_init_best_wrong": mi_best,
                "n_givens": int((puz != SU.BLANK).sum()),
                "I_s": [round(float(v), 4) for v in np.asarray(tr[-1].get("I_s", []))]
                       if "I_s" in tr[-1] else [],
                "wall_s": round(time.time() - t0, 1),
            }
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"{tid} solved={solved} ret={row['gt_retention']} "
                  f"viol={row['violations']} cells={row['cells_correct']}/81 "
                  f"mi={mi_hits}/{a.k_init} ({row['wall_s']}s)", flush=True)


if __name__ == "__main__":
    main()
