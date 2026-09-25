#!/usr/bin/env python3
# Ledger: THE ACCEPTANCE PLAN, Tier 1(a) (the PI 2026-09-15: analysis on the data we have, no pod runs). DESCRIPTIVE: a re-selection
# of the Night-A symmetric-cell arms on the champion night's 512-puzzle monitor, from their banked checkpoints, on the Mac's CPU.
#   Why: Finding 6 (digit augmentation on the symmetric cell, A4 - A3 = +3.17 pp) and the width-512 contrast (A8 - A7) were read against
#   the seed-pair spread of A3 / A7 (+5.08 pp), which the paper attributes to the 64-puzzle monitor those runs selected on. Re-selecting
#   the same checkpoints on the 512-puzzle monitor gives the spread and the contrast under the paper's own instrument.
#   What: for each arm in A3 (the symmetric cell, no digit augmentation), A4 (+ digit augmentation), A7 (the symmetric cell, seed 1):
#     SWEEP    every banked checkpoint (2k .. 50k), EMA weights, the cold D16 pass on the 512 monitor puzzles -> exact at 16;
#     SELECT   the monitor's maximum, ties to the earliest (the champion rule; the raw-weight second key is not applied here);
#     READ     the selected checkpoint, cold D16 (EMA) on the first N puzzles of the paper's seeded 20,000-puzzle test subsample
#              (subsample_seed 20260822, the evaluator's own draw), N = --read (default 5000; the D256 set's size).
#   The 512 monitor puzzles (data/sudoku_extreme/sudoku_extreme_seed0_mon512.npz, val_q/val_a) are training-file puzzles outside the
#   1,000-puzzle training subsample, which every arm shares (asserted at load).
"""  .venv/bin/python tools/lens_finalA_reselect.py --pull            (gsutil: the banked checkpoints of A3 A4 A7 -> runs/_finalA_pull/x/)
  JAX_PLATFORMS=cpu .venv/bin/python tools/lens_finalA_reselect.py --sweep  [--arms A3,A4,A7]   (resume-safe per checkpoint)
  JAX_PLATFORMS=cpu .venv/bin/python tools/lens_finalA_reselect.py --read   [--read 5000]         (after the sweep)
  .venv/bin/python tools/lens_finalA_reselect.py --report"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUCKET = "gs://qhrrn2-rescue/finalA/live/runs"
PULL = ROOT / "runs/_finalA_pull/x/runs"
MON_NPZ = ROOT / "data/sudoku_extreme/sudoku_extreme_seed0_mon512.npz"
TEST_NPZ = ROOT / "data/sudoku_extreme/sudoku_extreme_seed0_val10k.npz"   # carries the full 422,786 test split
OUT = ROOT / "runs/analysis/finalA_reselect_20260915"
ARMS_ALL = ["A3", "A4", "A7"]
T_TOTAL, SUBSAMPLE_SEED = 16, 20260822
LABEL = {"A3": "the symmetric cell, no digit augmentation", "A4": "the symmetric cell + digit augmentation", "A7": "the symmetric cell, seed 1"}

def _jax():
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tools"))
    import jax, jax.numpy as jnp
    import eval_sudoku_extreme as EV
    from qhrrn2 import episodic as E, model as M, grid as G, sudoku as SU, sudoku_extreme as SX
    from qhrrn2.config import Config
    return jax, jnp, EV, E, M, G, SU, SX, Config

def ck_dir(arm): return PULL / f"pretrainfinalA_{arm}"
def ck_steps(arm): return sorted(int(p.name[5:11]) for p in ck_dir(arm).glob("ckpt_*.pkl") if p.name[5:11].isdigit())

def pull(arms):
    for arm in arms:
        d = ck_dir(arm); d.mkdir(parents=True, exist_ok=True)
        cmd = ["gcloud", "storage", "cp", "--no-clobber", f"{BUCKET}/pretrainfinalA_{arm}/ckpt_*.pkl", str(d)]
        print("PULL", arm, " ".join(cmd), flush=True); subprocess.run(cmd, check=True)
        print(f"PULLED {arm}: {len(ck_steps(arm))} checkpoints", flush=True)

def _load_sets():
    _, _, _, _, _, _, _, SX, _ = _jax()
    mon = SX.load_prepared(str(MON_NPZ)); test = SX.load_prepared(str(TEST_NPZ))
    tq = {bytes(q.astype(np.int8).tobytes()) for q in mon["train_q"]}
    assert all(bytes(q.astype(np.int8).tobytes()) not in tq for q in mon["val_q"]), "monitor puzzles overlap the training subsample"
    assert np.array_equal(mon["train_q"], test["train_q"]), "the two prepared files carry different 1,000-puzzle training sets"
    return mon, test

def _cold16(saved, puz9, sol9, batch=512):
    jax, jnp, EV, E, M, G, SU, SX, Config = _jax()
    defaults = Config(); cfg = Config(**{k: type(getattr(defaults, k))(v) for k, v in saved["config"].items()})
    assert cfg.cell_kind == "dec", cfg.cell_kind
    st = saved["state_ema"]; params = st["model"]; tvj = jnp.asarray(st["table"][0])
    eta, eta_z = (float(v) for v in M.eq_etas(params, cfg)); ab = EV.coupled_ab(params, cfg); lay = getattr(cfg, "sudoku_layout", "origin") or "origin"
    cv = SU.layout_canvas(lay)
    void = jax.nn.one_hot(jnp.full((cv, cv), G.VOID, jnp.int32), M.VOCAB).transpose(2, 0, 1)
    ex_all = []
    for i in range(0, len(puz9), batch):
        p, s = puz9[i:i + batch], sol9[i:i + batch]
        x_can = EV.place_batch(p, lay); y0 = jnp.broadcast_to(void, (len(p),) + void.shape)
        ex, _, _, _ = EV.run_batch(params, cfg, tvj, x_can, y0, z0=None, t_total=T_TOTAL, tau=1.0, gamma=1.0, sol9=s, puz9=p, eta=eta, eta_z=eta_z, layout=lay, ab=ab)
        ex_all.append(np.asarray(ex)[-1])
    return np.concatenate(ex_all).astype(bool)

def sweep(arms):
    jax, jnp, EV, E, M, G, SU, SX, Config = _jax()
    mon, _ = _load_sets(); puz9 = mon["val_q"].astype(np.int32); sol9 = mon["val_a"].astype(np.int32); assert len(puz9) == 512
    (OUT / "sweep").mkdir(parents=True, exist_ok=True)
    for arm in arms:
        for step in ck_steps(arm):
            dst = OUT / "sweep" / f"{arm}_s{step:06d}.npz"
            if dst.exists(): continue
            t0 = time.time(); saved = E.load_ckpt(str(ck_dir(arm) / f"ckpt_{step:06d}.pkl")); assert int(saved["step"]) == step
            ex = _cold16(saved, puz9, sol9)
            tmp = dst.with_suffix(".tmp.npz"); np.savez(tmp, exact=ex, step=step, arm=arm, wall=time.time() - t0, n=len(ex), t_total=T_TOTAL, ema=True); os.replace(tmp, dst)
            print(f"SWEEP {arm} {step:6d}: mon512 exact@16 {100 * ex.mean():6.2f} %  ({time.time() - t0:.0f}s)", flush=True)

def selection():
    sel = {}
    for arm in ARMS_ALL:
        rows = sorted((int(np.load(p)["step"]), float(np.load(p)["exact"].mean())) for p in (OUT / "sweep").glob(f"{arm}_s*.npz"))
        if not rows: continue
        best = max(v for _, v in rows); step = min(s for s, v in rows if v == best)
        sel[arm] = {"step": step, "mon512": best, "curve": rows, "old_step_64mon": None}
    return sel

def read(arms, n):
    jax, jnp, EV, E, M, G, SU, SX, Config = _jax()
    _, test = _load_sets(); Q, A = test["test_q"], test["test_a"]
    rng = np.random.default_rng(SUBSAMPLE_SEED); idx = np.sort(rng.choice(len(Q), size=20000, replace=False))[:n]   # the evaluator's draw, its first n
    puz9 = Q[idx].astype(np.int32); sol9 = A[idx].astype(np.int32)
    (OUT / "read").mkdir(parents=True, exist_ok=True); sel = selection()
    for arm in arms:
        if arm not in sel: print(f"READ {arm}: no sweep yet", flush=True); continue
        step = sel[arm]["step"]; dst = OUT / "read" / f"{arm}_s{step:06d}_n{n}.npz"
        if dst.exists(): continue
        t0 = time.time(); saved = E.load_ckpt(str(ck_dir(arm) / f"ckpt_{step:06d}.pkl")); ex = _cold16(saved, puz9, sol9)
        np.savez(dst, exact=ex, idx=idx, step=step, arm=arm, n=n, subsample_seed=SUBSAMPLE_SEED, t_total=T_TOTAL, ema=True, wall=time.time() - t0)
        print(f"READ {arm} @ {step}: test-subsample({n}) exact@16 {100 * ex.mean():6.2f} %  ({time.time() - t0:.0f}s)", flush=True)

def report():
    sel = selection(); lines = ["DESCRIPTIVE — the Night-A symmetric-cell arms re-selected on the champion night's 512-puzzle monitor (EMA, cold D16), from the banked checkpoints; no rule adjudicates this reading.", ""]
    reads = {}
    for p in sorted((OUT / "read").glob("*.npz")) if (OUT / "read").exists() else []:
        z = np.load(p); reads[str(z["arm"])] = (int(z["step"]), float(z["exact"].mean()), int(z["n"]), np.asarray(z["exact"]))
    for arm in ARMS_ALL:
        if arm not in sel: continue
        s = sel[arm]; curve = " ".join(f"{st // 1000}k:{100 * v:.1f}" for st, v in s["curve"])
        lines.append(f"{arm} ({LABEL[arm]}): selected {s['step']} (mon512 {100 * s['mon512']:.2f} %); curve {curve}")
        if arm in reads: st, v, n, _ = reads[arm]; lines.append(f"    read at {st}: cold D16 on the first {n} of the seeded 20,000 test subsample = {100 * v:.2f} %  (binomial SE {100 * np.sqrt(v * (1 - v) / n):.2f} pp)")
    if all(a in reads for a in ("A3", "A7")):
        spread = 100 * abs(reads["A3"][1] - reads["A7"][1]); lines.append(f"seed spread A3 vs A7 under the 512 monitor: {spread:.2f} pp (the 64-monitor selection read 5.08)")
        if "A4" in reads:
            d = 100 * (reads["A4"][1] - reads["A3"][1]); n = min(reads[a][2] for a in ("A3", "A4"))
            e3, e4 = reads["A3"][3][:n], reads["A4"][3][:n]; b = int((e4 & ~e3).sum()); c = int((e3 & ~e4).sum())
            lines.append(f"digit augmentation A4 - A3 under the 512 monitor: {d:+.2f} pp (paired on {n}: only-A4 {b}, only-A3 {c}); twice the re-measured spread = {2 * spread:.2f} pp -> {'FLAT' if abs(d) <= 2 * spread else 'BEYOND'} by the paper's rule")
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "report.txt").write_text("\n".join(lines) + "\n"); print("\n".join(lines))
    json.dump({"selection": {a: {"step": s["step"], "mon512": s["mon512"], "curve": s["curve"]} for a, s in sel.items()}, "reads": {a: {"step": r[0], "exact": r[1], "n": r[2]} for a, r in reads.items()}}, open(OUT / "reselect.json", "w"), indent=1)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--pull", action="store_true"); ap.add_argument("--sweep", action="store_true"); ap.add_argument("--read", type=int, nargs="?", const=5000)
    ap.add_argument("--report", action="store_true"); ap.add_argument("--arms", default=",".join(ARMS_ALL)); a = ap.parse_args(); arms = a.arms.split(",")
    if a.pull: pull(arms)
    if a.sweep: sweep(arms)
    if a.read: read(arms, a.read)
    if a.report: report()
