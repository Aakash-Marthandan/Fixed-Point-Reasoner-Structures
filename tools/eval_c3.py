# Ledger: C.3' basin-snapped population voting (2026-08-09 registration).
# Per task: keyhole e_t fit on the equilibrium bulk (the Phase-C protocol —
# PRESERVES basins per the C.2 erosion finding); candidates = saved gate
# member predictions + own cold-start answer; SNAP each candidate through
# 8 final-map steps (basins collapse near-misses onto attractors); vote on
# the snapped limits. att1 = snapped plurality, att2 = second plurality.
"""
  python tools/eval_c3.py --ckpt runs/pretrain8_d16/ckpt_latest.pkl \
      --members runs/popx6_XA_results.jsonl --tasks a,b --out runs/c3
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import jax.numpy as jnp
import probe_e1e3 as P
from qhrrn2 import episodic as E
from qhrrn2 import grid as G
from qhrrn2.config import Config

def snap(model, cfg, x, cand, tv, k=8):
    st = P.trace(model, cfg, np.asarray(x), tau=1.0, task_vec=tv,
                 t_total=k, yprev_init=G.place(np.asarray(cand)),
                 skip_trained=True)
    return st[-1]["pred"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--members", required=True)
    ap.add_argument("--tasks", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--snap-k", type=int, default=8)
    ap.add_argument("--alt-rank", choices=("none", "residual"), default="none",
                    help="cluster S rider (2026-08-12): ALSO rank snapped "
                         "limits by fixed-point residual (EqR mechanism-6 "
                         "retest, post-shaping) — recorded alongside the "
                         "vote, protocol fields untouched")
    a = ap.parse_args()
    saved = E.load_ckpt(a.ckpt)
    defaults = Config()
    cfg = Config(**{k: type(getattr(defaults, k))(v)
                    for k, v in saved["config"].items()})
    mtxt = Path(a.members).read_text()
    if mtxt.lstrip().startswith("{"):
        mem = json.loads(mtxt)   # samp_to_members dict form (cluster S)
    else:
        mem = {json.loads(l)["task"]: json.loads(l)
               for l in mtxt.splitlines()}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    results = out / "results.jsonl"
    done = {json.loads(l)["task"] for l in results.read_text().splitlines()} \
        if results.exists() else set()
    import dev30
    task_ids = a.tasks.split(",") if a.tasks else sorted(dev30.MANIFEST)
    with open(results, "a") as f:
        for tid in task_ids:
            if tid in done:
                print(f"skip {tid}", flush=True); continue
            t0 = time.time()
            eps = G.load_task(tid)
            model, snaps_fit, sel, F = P.fit_arm_a(
                saved["state"], cfg, eps, steps=a.steps, val_every=50)
            tvj = jnp.asarray(sel[1])
            per_pair, atts, per_pair_alt = [], [], []
            for qi, ep in enumerate(eps):
                cands = []
                own = P.trace(model, cfg, ep.query_x, tau=1.0, task_vec=tvj,
                              t_total=cfg.T)[-1]["pred"]
                cands.append(own)
                for m in mem.get(tid, {}).get("member_query_preds", []):
                    cands.append(np.asarray(m[qi], dtype=np.int8))
                seen, uniq = set(), []
                for c in cands:
                    k = c.tobytes() + bytes(c.shape)
                    if k not in seen:
                        seen.add(k); uniq.append(c)
                snapped = [snap(model, cfg, ep.query_x, c, tvj, a.snap_k)
                           for c in uniq]
                counts = {}
                for s_ in snapped:
                    k = s_.tobytes() + bytes(s_.shape)
                    counts[k] = (counts.get(k, (0, s_))[0] + 1, s_)
                rank = sorted(counts.values(), key=lambda t: -t[0])
                att1 = rank[0][1]
                att2 = rank[1][1] if len(rank) > 1 else rank[0][1]
                gt = ep.query_y
                bits = [bool(gt is not None and att.shape == gt.shape
                             and np.array_equal(att, np.asarray(gt)))
                        for att in (att1, att2)]
                per_pair.append(bits)
                atts.append([att1.tolist(), att2.tolist()])
                if a.alt_rank == "residual":
                    # EqR mechanism-6 retest: unique snapped limits ranked by
                    # self-drift under snap_k more map steps (stability
                    # first, visit count as tiebreak). The 08-08 rejection of
                    # stability-scoring predates basin shaping — this records
                    # whether it discriminates on shaped substrates.
                    scored = []
                    for cnt, s_ in counts.values():
                        r2 = snap(model, cfg, ep.query_x, s_, tvj, a.snap_k)
                        drift = (1.0 if r2.shape != s_.shape
                                 else float((r2 != s_).mean()))
                        scored.append((drift, -cnt, s_))
                    scored.sort(key=lambda t: (t[0], t[1]))
                    a1r = scored[0][2]
                    a2r = scored[1][2] if len(scored) > 1 else scored[0][2]
                    per_pair_alt.append(
                        [bool(gt is not None and att.shape == gt.shape
                              and np.array_equal(att, np.asarray(gt)))
                         for att in (a1r, a2r)])
            row = {"task": tid, "sel_step": sel[0],
                   "solved_pass2": all(b[0] or b[1] for b in per_pair),
                   "per_pair_bits": per_pair, "preds": atts,
                   "n_members": len(mem.get(tid, {}).get("member_query_preds", [])),
                   "wall_s": round(time.time() - t0, 1)}
            if a.alt_rank != "none":
                row["alt_rank"] = a.alt_rank
                row["per_pair_bits_alt"] = per_pair_alt
                row["solved_pass2_alt"] = all(b[0] or b[1]
                                              for b in per_pair_alt)
            f.write(json.dumps(row) + "\n"); f.flush()
            print(f"{tid} pass2={row['solved_pass2']} "
                  f"bits={per_pair} ({row['wall_s']}s)", flush=True)
    print("C3 DONE", flush=True)

if __name__ == "__main__":
    main()
