#!/usr/bin/env python3
"""THE INSTRUMENT SUITE — checkpoint-level instruments with a generic interface (Plan_2026-09-07_Instrument_Suite §2.2 / §5.3;
analysis-time, descriptive, no rules). Mode `dyn` = C1 decoder dynamics + C2 decimation quality at stalls on the evaluator's
rating-stratified cold trajectories (the finalA lens's E2/E3, lifted out of the campaign-specific tool); any checkpoint of any
cell class (rg natives incl. inner_k, trm, dec). One JSON row per run is appended to --out (idempotent on (name, n, t)).

  PYTHONPATH=src JAX_PLATFORMS=cpu .venv/bin/python tools/suite_ckpt.py --mode dyn --ckpt runs/pretrainfinalA_A3/ckpt_042000.pkl --ema \
      --name finalA/A3@42k --n 128 --t 64 --out runs/analysis/suite_ckpt_dyn_<date>.jsonl
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; RUNS = Path(os.environ.get("QHRRN_RUNS", ROOT / "runs"))
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tools"))
NPZ = ROOT / "data/sudoku_extreme/sudoku_extreme_seed0.npz"
from analyze_finalA_ecc import peel, violations   # the peeling decoder + the syndrome (the lens's own functions)

def run_dyn(ckpt, ema, n, t_total, strat_seed=20260821):
    import jax, jax.numpy as jnp
    from qhrrn2 import episodic as E, grid as GR, model as M, sudoku as SU, sudoku_extreme as SX
    from qhrrn2.config import Config
    import eval_sudoku_extreme as EV
    d = SX.load_prepared(NPZ); Q, A, R = d["test_q"], d["test_a"], d["test_rating"]
    ids = SX.stratified_subsample(R, n, strat_seed); B = len(ids); puz9 = Q[ids].astype(np.int32); sol9 = A[ids].astype(np.int32); ng = puz9 == 0
    saved = E.load_ckpt(str(ckpt)); defaults = Config(); cfg = Config(**{kk: type(getattr(defaults, kk))(v) for kk, v in saved["config"].items()})
    st = saved["state_ema"] if ema else saved["state"]
    if st is None: raise SystemExit("no EMA weights in this checkpoint")
    params = st["model"]; tvj = jnp.asarray(st["table"][0])
    eta, eta_z = (float(v) for v in M.eq_etas(params, cfg)); layout = cfg.sudoku_layout or "origin"; cv = SU.layout_canvas(layout)
    trm = cfg.cell_kind in ("trm", "dec"); K = 1 if trm else max(1, int(getattr(cfg, "inner_k", 1))); ab = EV.coupled_ab(params, cfg)
    x_can = jnp.asarray(np.stack([SU.place_layout(gq.astype(np.int8), layout) for gq in puz9]), jnp.int32)
    void = jax.nn.one_hot(jnp.full((cv, cv), GR.VOID, jnp.int32), M.VOCAB).transpose(2, 0, 1); y = jnp.broadcast_to(void, (B,) + void.shape); z = None
    T = t_total; preds = np.zeros((T, B, 9, 9), np.int16); ent = np.zeros((T, B)); cw = np.zeros((T, B)); com = np.zeros((T, B)); cc = np.zeros((T, B)); syn = np.zeros((T, B)); p9_last = None; tt = time.time()
    for t in range(T):
        tn = 0.0 if trm else min(t, cfg.T - 1) / max(cfg.T - 1, 1)
        for _ in range(K):
            first = z is None; logits, zf = EV._step(cfg, 1.0, float(tn), first)(params, x_can, y, tvj, jnp.zeros(1) if first else z); z = zf if first else z + eta_z * (zf - z)
        p = jax.nn.softmax(logits, axis=-1); pT = p.transpose(0, 3, 1, 2); y = (ab[0] * y + ab[1] * pT) if ab is not None else (y + eta * (pT - y))
        p9 = np.asarray(EV.layout_gather(p, layout))[..., 1:10]; p9 = p9 / np.maximum(p9.sum(-1, keepdims=True), 1e-9)
        pred = np.asarray(EV.layout_gather(jnp.argmax(logits, axis=-1), layout)); pred = np.where(pred == GR.VOID, 0, pred).astype(np.int16); preds[t] = pred
        e = -(p9 * np.log(p9 + 1e-12)).sum(-1) / np.log(9); conf = p9.max(-1); corr = pred == sol9; nng = np.maximum(ng.sum((1, 2)), 1)
        ent[t] = (e * ng).sum((1, 2)) / nng; cw[t] = ((conf > .9) & ~corr & ng).sum((1, 2)) / nng; com[t] = ((conf > .9) & ng).sum((1, 2)) / nng; cc[t] = (corr & ng).sum((1, 2)) / nng; syn[t] = violations(pred); p9_last = p9
        if t == 0: print(f"    step 1 in {time.time()-tt:.0f}s (incl. compile)", flush=True)
    solved = (preds[-1] == sol9).all((1, 2)); fe = np.array([next((s for s in range(T) if (preds[s, b] == sol9[b]).all()), -1) for b in range(B)])
    corr_t = preds == sol9[None]; ever = np.zeros((B, 9, 9), bool); unpeel = np.zeros(B); flips_c = np.zeros((T, B)); flips_w = np.zeros((T, B))
    for t in range(T):
        if t: ch = (preds[t] != preds[t - 1]) & ng; flips_c[t] = (ch & corr_t[t]).sum((1, 2)) / nng; flips_w[t] = (ch & ~corr_t[t]).sum((1, 2)) / nng; unpeel += ((ever & ~corr_t[t]) & ng).sum((1, 2))
        ever |= corr_t[t]
    mono = (unpeel == 0); steps = [1, 2, 4, 8, 16, 32, 64]; sidx = [min(s_, T) - 1 for s_ in steps]
    def m_(arr, mask, ii): return [float(arr[i, mask].mean()) if mask.any() else None for i in ii]
    o = dict(solved=float(solved.mean()), n=int(B), t=int(T), n_unsolved=int((~solved).sum()), K=K, cell=cfg.cell_kind,
             cells_s=m_(cc, solved, sidx), cells_u=m_(cc, ~solved, sidx), ent_s=m_(ent, solved, sidx), ent_u=m_(ent, ~solved, sidx), com_s=m_(com, solved, [0, 7, T - 1]), com_u=m_(com, ~solved, [0, 7, T - 1]),
             cw_s=m_(cw, solved, [0, 7, T - 1]), cw_u=m_(cw, ~solved, [0, 7, T - 1]), mono_solved=float(mono[solved].mean()) if solved.any() else None,
             unpeel_solved=float((unpeel / nng)[solved].mean()) if solved.any() else None, unpeel_unsolved=float((unpeel / nng)[~solved].mean()) if (~solved).any() else None,
             flips_w_last32_unsolved=float(flips_w[max(T - 32, 0):, ~solved].sum(0).mean()) if (~solved).any() else None, flips_c_last32_unsolved=float(flips_c[max(T - 32, 0):, ~solved].sum(0).mean()) if (~solved).any() else None,
             syn_u=m_(syn, ~solved, sidx), syn_osc_unsolved=float((syn[max(T - 32, 0):, ~solved].max(0) - syn[max(T - 32, 0):, ~solved].min(0)).mean()) if (~solved).any() else None,
             syn_mono_frac_unsolved=float(np.mean([bool(np.all(np.diff(syn[:, b]) <= 0)) for b in np.where(~solved)[0]])) if (~solved).any() else None,
             fe_med=float(np.median(fe[solved])) if solved.any() else None, fe_p90=float(np.percentile(fe[solved], 90)) if solved.any() else None, fe_step1=float((fe[solved] == 0).mean()) if solved.any() else None, wall=round(time.time() - tt, 1))
    e3 = {}
    for tau in (.9, .99):
        U = ~solved
        if not U.any(): break
        conf = p9_last.max(-1); com_m = (conf > tau) & ng; n_com = com_m.sum((1, 2)) / nng
        wrong_com = ((com_m & (preds[-1] != sol9)).sum((1, 2)) / np.maximum(com_m.sum((1, 2)), 1))
        gg0 = np.where(com_m, preds[-1], puz9); pg_, st_ = peel(gg0[U]); ok = (st_ == 1) & (pg_ == sol9[U]).all((1, 2))
        e3[str(tau)] = dict(n_stalled=int(U.sum()), committed_frac=float(n_com[U].mean()), wrong_committed_frac=float(wrong_com[U].mean()), all_committed_correct=float((wrong_com[U] == 0).mean()),
                            peel_solves=float(ok.mean()), peel_stuck=float((st_ == 0).mean()), peel_contradiction=float((st_ == 2).mean()))
    o["e3"] = e3
    return o

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", default="dyn", choices=["dyn"]); ap.add_argument("--ckpt", required=True); ap.add_argument("--ema", action="store_true")
    ap.add_argument("--name", required=True); ap.add_argument("--n", type=int, default=128); ap.add_argument("--t", type=int, default=64); ap.add_argument("--out", required=True); a = ap.parse_args()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        for l in out.read_text().splitlines():
            try: r = json.loads(l)
            except Exception: continue
            if r.get("name") == a.name and r.get("n") == a.n and r.get("t") == a.t: print(f"SKIP (done): {a.name}"); return
    print(f"== suite_ckpt {a.mode}: {a.name} ({a.ckpt}, ema={a.ema}, n={a.n}, t={a.t}) ==", flush=True)
    o = run_dyn(a.ckpt, a.ema, a.n, a.t); o.update(name=a.name, ckpt=str(a.ckpt), ema=bool(a.ema), mode=a.mode)
    with open(out, "a") as fh: fh.write(json.dumps(o, default=float) + "\n")
    r_ = lambda xs: "/".join("-" if v is None else f"{100*v:.0f}" for v in xs)
    print(f"  {a.name}: solved {100*o['solved']:.1f} | cells s {r_(o['cells_s'])} u {r_(o['cells_u'])} | commit s {r_(o['com_s'])} u {r_(o['com_u'])} | conf-wrong s {r_(o['cw_s'])} u {r_(o['cw_u'])} | mono {'-' if o['mono_solved'] is None else f'{100*o['mono_solved']:.1f}'} | fe {o['fe_med']}/{o['fe_p90']} step1 {'-' if o['fe_step1'] is None else f'{100*o['fe_step1']:.0f}'} | E3@.9 {o['e3'].get('0.9')} ({o['wall']}s)")

if __name__ == "__main__":
    main()
