#!/usr/bin/env python3
# Ledger: WHY THE LOOP COMMITS (2026-09-19; registration Documentation/Note_2026-09-19_Why_Decimation.md, written before any row).
# DESCRIPTIVE, zero cloud: banked checkpoints on the Mac. Reuses the repair-radius lens's model loader, loop and recorder
# (tools/lens_repair_radius.py, imported, never edited here); its own report files.
#   F1  train against test puzzles at iteration 1 (16 iterations run): committed share, first-guess wrong share, entropy, exact at 16.
#   F2  inside the first two iterations: the readout after EACH outer cycle (z_H changes only there); the probe's full-iteration grid
#       is asserted equal to the evaluator's.
#   F3  along training: iteration-1 committed share / wrong share and exact at 16 on 128 train and 128 test puzzles per checkpoint.
"""  .venv/bin/python tools/lens_why_decimation.py --selftest
  JAX_PLATFORMS=cpu .venv/bin/python tools/lens_why_decimation.py --run F1|F2|F3
  .venv/bin/python tools/lens_why_decimation.py --report"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import lens_repair_radius as L   # Model, Rec, run_rows, puzzles, pct, spearman, TAU

OUT = ROOT / "runs/analysis/why_decimation_20260919"
F1_MODELS = {"C5": ("runs/pretrainchamp_C5/ckpt_046000.pkl", "the symmetric model, width 192, seed 0 (46k)"),
             "C0": ("runs/pretrainchamp_C0/ckpt_016000.pkl", "the symmetric model, width 384, seed 0 (16k)"),
             "X0": ("runs/pretrainsportC1_X0/ckpt_050000.pkl", "TRM's network reproduced, 5M (50k)"),
             "X1": ("runs/pretrainsportC2_X1/ckpt_020000.pkl", "TRM's loop WITHOUT digit augmentation (20k)")}
F2_MODELS = {"C5": F1_MODELS["C5"], "EQR": ("runs/field_ckpts/ported/eqr/ckpt_latest.pkl", "EqR's released weights")}
F3_RUNS = {"C5": ("runs/pretrainchamp_C5", (2, 4, 6, 8, 10, 14, 18, 22, 26, 30, 38, 46, 50)), "C0": ("runs/pretrainchamp_C0", (2, 4, 6, 8, 10, 16, 30))}
T1 = 16

# ---------------- pure helpers (selftested) ----------------
def entropy9(p9):
    """(.., 9) probabilities -> the entropy in units of log 9."""
    p = np.clip(np.asarray(p9, float), 1e-12, 1.0); return -(p * np.log(p)).sum(-1) / np.log(9)

def cell_stats(pred, conf, sol, ng):
    """Per puzzle: committed share, wrong share, wrong among the committed (all over the EMPTY cells)."""
    nng = np.maximum(ng.sum((1, 2)), 1); com = (conf > L.TAU) & ng; wr = (pred != sol) & ng
    return com.sum((1, 2)) / nng, wr.sum((1, 2)) / nng, (com & wr).sum((1, 2)) / np.maximum(com.sum((1, 2)), 1)

def summarize(a):
    ok = np.asarray(a["ex_end"]).astype(bool)
    return dict(n=int(len(ok)), c1=float(np.mean(a["c1"])), e1=float(np.mean(a["e1"])), w1=float(np.mean(a["w1"])), exact1=float(np.mean(np.asarray(a["fe"]) == 1)), exact16=float(ok.mean()))

def f1_line(tag, s): return f"    {tag:5s} committed {L.pct(s['c1'])} % | first guess wrong {L.pct(s['e1'])} % | wrong among the committed {L.pct(s['w1'])} % | exact after 1: {L.pct(s['exact1'])} %, at 16: {L.pct(s['exact16'])} %"

# ---------------- the runs ----------------
def sets(n_train, n_test):
    sys.path.insert(0, str(ROOT / "src")); from qhrrn2 import sudoku_extreme as SX
    d = SX.load_prepared(L.NPZ); ids, puz, sol, _ = L.puzzles(512)
    return (d["train_q"][:n_train].astype(np.int32), d["train_a"][:n_train].astype(np.int32)), (puz[:: 512 // n_test][:n_test], sol[:: 512 // n_test][:n_test])

def model_at(path, key="C5"):
    L.MODELS["_tmp"] = (path, path); return L.Model("_tmp")

def rows16(m, puz, sol, tag):
    """16 iterations from the fixed start on every row (no early exit): the recorder's arrays."""
    old = L.T_TOTAL; L.T_TOTAL = T1
    try: a = L.run_rows(m, puz, sol, lambda rows: None, None, tag)
    finally: L.T_TOTAL = old
    a["ex_end"] = a["ex64"]; return a

def run_f1():
    OUT.mkdir(parents=True, exist_ok=True); (tr, te) = sets(512, 512)
    for k, (path, _) in F1_MODELS.items():
        dst = OUT / f"F1_{k}.npz"
        if dst.exists(): print(f"SKIP {dst.name}"); continue
        m = model_at(path); res = {}
        for name, (puz, sol) in (("train", tr), ("test", te)):
            a = rows16(m, puz, sol, f"F1 {k}/{name}"); res.update({f"{name}_{kk}": v for kk, v in a.items()})
        np.savez(OUT / f"F1_{k}.tmp.npz", ckpt=path, **res); os.replace(OUT / f"F1_{k}.tmp.npz", dst); print(f"DONE {dst.name}", flush=True)

def probe_cycles(m, puz, sol, n_iter=2):
    """The readout after every outer cycle of the first n_iter iterations (the cell's own segment, replicated; eval mode: no noise).
    Returns com, wrong (cycles, B) and the grid after each full iteration (n_iter, B, 9, 9)."""
    jax, jnp = m.jax, m.jnp; cfg = m.cfg; dec = cfg.cell_kind == "dec"
    from qhrrn2 import dec_cell as DC, trm_cell as TC
    C = DC if dec else TC; p = jax.tree_util.tree_map(jnp.asarray, m.params["dec" if dec else "trm"]); hw = cfg.canvas * cfg.canvas; lam = cfg.trm_lambda
    stack = (lambda h: DC._stack(p, h, cfg)) if dec else (lambda h: TC._stack(p, h, cfg.trm_puzzle_emb_len))
    step = lambda z, inj: (lambda Fz: (z + (1.0 - lam) * (Fz - z)) if lam > 0 else Fz)(stack(z + inj))
    def one(x):
        emb = C.embed(p, cfg, x); z = C.z0(cfg, hw) if dec else TC.z0(cfg, hw, p=p); zH, zL = z[0], z[1]; outs = []
        for _ in range(n_iter):
            for _c in range(cfg.trm_h_cycles):
                for _l in range(cfg.trm_l_cycles): zL = step(zL, zH + emb)
                zH = step(zH, zL); outs.append(C.readout(p, cfg, zH, x.shape)[0])
        return jnp.stack(outs)                                                      # (cycles, 9, 9, VOCAB)
    f = jax.jit(jax.vmap(one)); ng = puz == 0; com, wrong, ent, grids = [], [], [], []
    for b0 in range(0, len(puz), L.BS):
        x = puz[b0:b0 + L.BS]; keep = len(x); xp = np.concatenate([x, np.repeat(x[-1:], L.BS - keep, 0)]) if keep < L.BS else x
        lg = np.asarray(f(jnp.asarray(xp)))[:keep]                                   # (B, cycles, 9, 9, VOCAB)
        p9 = np.exp(lg[..., 1:10] - lg[..., 1:10].max(-1, keepdims=True)); p9 = p9 / p9.sum(-1, keepdims=True)
        pred = lg[..., 1:10].argmax(-1) + 1; conf = p9.max(-1); e9 = entropy9(p9)
        for c in range(lg.shape[1]):
            a, w, _ = cell_stats(pred[:, c], conf[:, c], sol[b0:b0 + keep], ng[b0:b0 + keep])
            if b0 == 0: com.append([]); wrong.append([]); ent.append([])
            com[c].extend(a); wrong[c].extend(w); ent[c].extend((e9[:, c] * ng[b0:b0 + keep]).sum((1, 2)) / np.maximum(ng[b0:b0 + keep].sum((1, 2)), 1))
        grids.append(pred[:, cfg.trm_h_cycles - 1::cfg.trm_h_cycles])
        print(f"  F2 rows {b0 + keep}/{len(puz)}", flush=True)
    return np.asarray(com), np.asarray(wrong), np.asarray(ent), np.concatenate(grids)

def run_f2():
    OUT.mkdir(parents=True, exist_ok=True); _, (puz, sol) = sets(8, 256)
    for k, (path, _) in F2_MODELS.items():
        dst = OUT / f"F2_{k}.npz"
        if dst.exists(): print(f"SKIP {dst.name}"); continue
        m = model_at(path); com, wrong, ent, grids = probe_cycles(m, puz, sol)
        cap = L.Cap(len(puz)); pad = lambda a: np.concatenate([a, np.repeat(a[-1:], L.BS - len(a), 0)]) if len(a) < L.BS else a   # the evaluator's own first iteration
        for b0 in range(0, len(puz), L.BS):
            rows = np.arange(b0, min(b0 + L.BS, len(puz))); x_can = m.EV.place_batch(pad(puz[rows]), m.lay)
            m.iterate(x_can, m.jnp.broadcast_to(m.void, (L.BS,) + m.void.shape), None, 0, 1, cap, rows, len(rows))
        ng = puz == 0; agree = float(((grids[:, 0] == cap.pred) & ng).sum() / ng.sum()); assert agree > 0.999, f"the probe's first iteration differs from the evaluator's on the empty cells ({agree:.4f})"
        np.savez(OUT / f"F2_{k}.tmp.npz", ckpt=path, com=com, wrong=wrong, ent=ent, agree=agree, h_cycles=m.cfg.trm_h_cycles); os.replace(OUT / f"F2_{k}.tmp.npz", dst); print(f"DONE {dst.name} (probe = evaluator on {100*agree:.2f} % of the empty cells)", flush=True)

def run_f3():
    OUT.mkdir(parents=True, exist_ok=True); (tr, te) = sets(128, 128)
    for k, (d, steps) in F3_RUNS.items():
        for st in steps:
            dst = OUT / f"F3_{k}_{st:03d}k.npz"
            if dst.exists(): print(f"SKIP {dst.name}"); continue
            m = model_at(f"{d}/ckpt_{st * 1000:06d}.pkl"); res = {}
            for name, (puz, sol) in (("train", tr), ("test", te)):
                a = rows16(m, puz, sol, f"F3 {k}@{st}k/{name}"); res.update({f"{name}_{kk}": v for kk, v in a.items()})
            np.savez(OUT / f"F3_{k}_{st:03d}k.tmp.npz", step=st * 1000, **res); os.replace(OUT / f"F3_{k}_{st:03d}k.tmp.npz", dst); print(f"DONE {dst.name}", flush=True)

# ---------------- report ----------------
def report():
    Ls, J = [], {}; say = lambda s_="": (Ls.append(s_), print(s_))
    say("WHY THE LOOP COMMITS (descriptive; tools/lens_why_decimation.py; registration Note_2026-09-19_Why_Decimation.md). EMA, the fixed start, no halting, no noise.")
    say(); say("== F1: TRAIN against TEST puzzles at iteration 1 (512 each; 16 iterations run) =="); J["F1"] = {}
    for k, (path, desc) in F1_MODELS.items():
        f = OUT / f"F1_{k}.npz"
        if not f.exists(): continue
        D = np.load(f, allow_pickle=True); say(f"  {k}: {desc}"); J["F1"][k] = {}
        for name in ("train", "test"):
            s = summarize({kk: D[f"{name}_{kk}"] for kk in ("c1", "e1", "w1", "fe", "ex_end")}); J["F1"][k][name] = s; say(f1_line(name, s))
    say(); say("== F2: the readout after each OUTER CYCLE of iterations 1 and 2 (256 test puzzles) =="); J["F2"] = {}
    for k, (path, desc) in F2_MODELS.items():
        f = OUT / f"F2_{k}.npz"
        if not f.exists(): continue
        D = np.load(f, allow_pickle=True); com, wr, en = D["com"].mean(1), D["wrong"].mean(1), D["ent"].mean(1); J["F2"][k] = dict(com=com.tolist(), wrong=wr.tolist(), ent=en.tolist(), agree=float(D["agree"]))
        say(f"  {k}: {desc} (the probe's first iteration = the evaluator's on {100*float(D['agree']):.2f} % of the empty cells)")
        say("    cycle:          " + "  ".join(f"{c + 1:5d}" for c in range(len(com)))); say("    committed %:    " + "  ".join(f"{100*v:5.1f}" for v in com))
        say("    wrong %:        " + "  ".join(f"{100*v:5.1f}" for v in wr)); say("    entropy (/ln9): " + "  ".join(f"{v:5.3f}" for v in en))
    say(); say("== F3: along training (128 train and 128 test puzzles per checkpoint) =="); J["F3"] = {}
    for k, (d, steps) in F3_RUNS.items():
        rows = []
        for st in steps:
            f = OUT / f"F3_{k}_{st:03d}k.npz"
            if not f.exists(): continue
            D = np.load(f, allow_pickle=True); r = dict(step=st * 1000)
            for name in ("train", "test"): r[name] = summarize({kk: D[f"{name}_{kk}"] for kk in ("c1", "e1", "w1", "fe", "ex_end")})
            rows.append(r)
        if not rows: continue
        say(f"  {k}: step | committed train / test | first guess wrong train / test | exact at 16 train / test")
        for r in rows: say(f"    {r['step']:6d} | {L.pct(r['train']['c1'])} / {L.pct(r['test']['c1'])} | {L.pct(r['train']['e1'])} / {L.pct(r['test']['e1'])} | {L.pct(r['train']['exact16'])} / {L.pct(r['test']['exact16'])}")
        c = [r["test"]["c1"] for r in rows]; rho_tr = L.spearman(c, [1 - r["train"]["e1"] for r in rows]); rho_te = L.spearman(c, [r["test"]["exact16"] for r in rows])
        first = next((r for r in rows if r["test"]["c1"] > 0.8), None)
        say(f"    Spearman over checkpoints: test commitment with TRAIN first-guess accuracy {rho_tr if rho_tr is None else round(rho_tr, 2)}, with TEST exact at 16 {rho_te if rho_te is None else round(rho_te, 2)}"
            + (f" | first checkpoint past 80 % commitment: {first['step']} (first guess wrong: train {L.pct(first['train']['e1'])} %, test {L.pct(first['test']['e1'])} %)" if first else ""))
        J["F3"][k] = dict(rows=rows, rho_commit_trainacc=rho_tr, rho_commit_testexact=rho_te)
    (OUT / "report.txt").write_text("\n".join(Ls) + "\n"); (OUT / "report.json").write_text(json.dumps(J, indent=1, default=float))

# ---------------- selftest ----------------
def selftest():
    n = 0
    assert abs(entropy9(np.full(9, 1 / 9)) - 1.0) < 1e-9 and entropy9(np.eye(9)[0]) < 1e-9; n += 1
    sol = np.arange(81).reshape(1, 9, 9) % 9 + 1; ng = np.zeros((1, 9, 9), bool); ng[0, 0, :4] = True; pred = sol.copy(); pred[0, 0, 0] = 9 if sol[0, 0, 0] != 9 else 1
    conf = np.ones((1, 9, 9)); conf[0, 0, 1] = 0.5
    c, w, wc = cell_stats(pred, conf, sol, ng); assert c.tolist() == [0.75] and w.tolist() == [0.25] and abs(wc[0] - 1 / 3) < 1e-12; n += 1
    s = summarize(dict(c1=[1.0, 0.5], e1=[0.2, 0.4], w1=[0.1, 0.3], fe=[1, 3], ex_end=[1, 0])); assert s["exact1"] == 0.5 and s["exact16"] == 0.5 and abs(s["e1"] - 0.3) < 1e-12; n += 1
    assert f1_line("train", s) == "    train committed 75.0 % | first guess wrong 30.0 % | wrong among the committed 20.0 % | exact after 1: 50.0 %, at 16: 50.0 %"; n += 1
    assert all((ROOT / p).exists() for p, _ in F1_MODELS.values()) and all((ROOT / p).exists() for p, _ in F2_MODELS.values()); n += 1
    assert all((ROOT / d / f"ckpt_{st * 1000:06d}.pkl").exists() for d, steps in F3_RUNS.values() for st in steps); n += 1
    print(f"selftest OK: {n}/{n}")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true"); ap.add_argument("--run", choices=["F1", "F2", "F3"]); ap.add_argument("--report", action="store_true"); a = ap.parse_args()
    if a.selftest: return selftest()
    if a.run: {"F1": run_f1, "F2": run_f2, "F3": run_f3}[a.run]()
    if a.report: report()

if __name__ == "__main__":
    main()
