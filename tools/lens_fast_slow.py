#!/usr/bin/env python3
# Ledger: IS THE FAST STATE SLAVED TO THE SLOW ONE? (2026-09-19; registration Documentation/Note_2026-09-19_Fast_Slow_Probe.md, written
# before any row). DESCRIPTIVE, zero cloud. The model's own segment, replicated one iteration at a time (as lens_corpus_normalized's
# Part C, same gate). Part 1: perturb ONLY z_L at the start of iteration t and follow the difference through the six inner updates at
# fixed z_H, then into z_H and the output. Part 2: carry only z_H (z_L reset every iteration) or only z_L (z_H reset).
"""  JAX_PLATFORMS=cpu .venv/bin/python tools/lens_fast_slow.py --selftest
  JAX_PLATFORMS=cpu .venv/bin/python tools/lens_fast_slow.py --run C5 X0 EQR C0      (resume-safe per model)
  .venv/bin/python tools/lens_fast_slow.py --report"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tools"))
import lens_commit_validity as V
import lens_corpus_normalized as CN

OUT = ROOT / "runs/analysis/fast_slow_20260919"
MODELS = {k: (V.MODELS[k][0], V.MODELS[k][1], V.MODELS[k][4]) for k in ("C5", "C0", "X0", "EQR")}
BRANCH, EPS, N_IT, BS, SEED = (1, 2, 4, 8), 0.1, 16, 128, 20260919
GROUPS = ("already solved", "not yet, solved by 16", "never by 16")

# ---------------- pure helpers (selftested) ----------------
def make_pair_iter(jnp, stp, embed, readout, H, L):
    """One iteration on two copies that share x and z_H and differ in z_L. Returns (delta (L+1,), Delta (H,), logits_a, logits_b, zH_a, zL_a, zH_b, zL_b).
    delta_l = ||zL_b - zL_a|| / ||zL_a|| after l inner updates of the FIRST outer cycle; Delta_c = ||zH_b - zH_a|| / ||zH_a|| after outer cycle c."""
    rel = lambda a, b: jnp.sqrt(jnp.sum((b - a) ** 2)) / jnp.maximum(jnp.sqrt(jnp.sum(a ** 2)), 1e-30)
    def f(x, zH, zLa, zLb):
        emb = embed(x); zHa, zHb = zH, zH; delta = [rel(zLa, zLb)]; Delta = []
        for c in range(H):
            for _ in range(L):
                zLa = stp(zLa, zHa + emb); zLb = stp(zLb, zHb + emb)
                if c == 0: delta.append(rel(zLa, zLb))
            zHa = stp(zHa, zLa); zHb = stp(zHb, zLb); Delta.append(rel(zHa, zHb))
        return jnp.stack(delta), jnp.stack(Delta), readout(zHa), readout(zHb), zHa, zLa, zHb, zLb
    return f

def make_one_iter(jnp, stp, embed, readout, H, L):
    """One iteration of ONE copy (the unperturbed run and the reset runs): (logits, zH, zL). Same order of updates as make_pair_iter."""
    def f(x, zH, zL):
        emb = embed(x)
        for _ in range(H):
            for _ in range(L): zL = stp(zL, zH + emb)
            zH = stp(zH, zL)
        return readout(zH), zH, zL
    return f

def group_of(exact_by_it, t):
    """exact_by_it (T, B) bool along the UNPERTURBED run; t 1-based branch iteration -> (B,) index into GROUPS."""
    solved_before = exact_by_it[t - 2] if t > 1 else np.zeros(exact_by_it.shape[1], bool); ever = exact_by_it.any(0)
    return np.where(solved_before, 0, np.where(ever, 1, 2))

def summarize_branch(delta, Delta, disagree, grp):
    """delta (B, L+1), Delta (B, H), disagree (B,), grp (B,) -> per group: n, median delta_L/delta_0, the median per-update factor, median Delta_1/delta_0, Delta_H/delta_0, mean output disagreement."""
    out = {}
    for g, name in enumerate(GROUPS):
        m = grp == g
        if not m.any(): out[name] = dict(n=0); continue
        r = delta[m, -1] / delta[m, 0]
        out[name] = dict(n=int(m.sum()), contraction=float(np.median(r)), per_update=float(np.median(r) ** (1.0 / (delta.shape[1] - 1))), expanding_share=float((r >= 1).mean()),
                         to_slow_first=float(np.median(Delta[m, 0] / delta[m, 0])), to_slow_end=float(np.median(Delta[m, -1] / delta[m, 0])), output_disagree=float(disagree[m].mean()))
    return out

def line(tag, s):
    return f"      {tag:24s} n {s['n']:3d}" + ("" if not s["n"] else f" | fast difference after six inner updates x{s['contraction']:.4f} (x{s['per_update']:.3f} per update; expanding on {100 * s['expanding_share']:.0f} % of puzzles) | reaches the slow state: x{s['to_slow_first']:.4f} after one outer cycle, x{s['to_slow_end']:.4f} at the iteration's end | output cells changed {100 * s['output_disagree']:.2f} %")

# ---------------- the run ----------------
def run_model(key):
    path, ema, _ = MODELS[key]; jax, jnp, cfg, st, step = CN._load(path, ema); assert cfg.cell_kind in ("trm", "dec"); dec = cfg.cell_kind == "dec"
    from qhrrn2 import dec_cell as DC, trm_cell as TC
    C = DC if dec else TC; p = jax.tree_util.tree_map(jnp.asarray, st["model"]["dec" if dec else "trm"]); hw = cfg.canvas * cfg.canvas; lam = cfg.trm_lambda; H, L = cfg.trm_h_cycles, cfg.trm_l_cycles
    stack = (lambda h: DC._stack(p, h, cfg)) if dec else (lambda h: TC._stack(p, h, cfg.trm_puzzle_emb_len))
    stp = lambda z, inj: (lambda Fz: (z + (1.0 - lam) * (Fz - z)) if lam > 0 else Fz)(stack(z + inj))
    ids, puz, sol = CN.puzzles("strat", 256); ng = puz == 0; N = len(puz)
    emb_f, read_f = (lambda x: C.embed(p, cfg, x)), (lambda zH: C.readout(p, cfg, zH, (9, 9))[0])
    pair = jax.jit(jax.vmap(make_pair_iter(jnp, stp, emb_f, read_f, H, L))); one = jax.jit(jax.vmap(make_one_iter(jnp, stp, emb_f, read_f, H, L)))
    z0 = C.z0(cfg, hw) if dec else TC.z0(cfg, hw, p=p); zH0, zL0 = np.asarray(z0[0]), np.asarray(z0[1]); rng = np.random.default_rng(SEED); t_a = time.time()
    dig = lambda lg: (np.asarray(lg)[..., 1:10].argmax(-1) + 1).astype(np.int8)
    base = np.zeros((N_IT, N, 9, 9), np.int8); P1 = {}; reset = {"carry only the slow state": np.zeros((N, 9, 9), np.int8), "carry only the fast state": np.zeros((N, 9, 9), np.int8)}
    for b0 in range(0, N, BS):
        x = jnp.asarray(puz[b0:b0 + BS]); B = x.shape[0]; zH = jnp.broadcast_to(z0[0], (B,) + zH0.shape); zL = jnp.broadcast_to(z0[1], (B,) + zL0.shape)
        for t in range(1, N_IT + 1):
            if t in BRANCH:
                for variant in ("noise", "reinit"):
                    if variant == "reinit" and t == 1: continue
                    if variant == "noise":
                        rms = np.sqrt(np.mean(np.asarray(zL) ** 2, axis=tuple(range(1, zL.ndim)), keepdims=True)); zLb = zL + EPS * rms * rng.standard_normal(zL.shape).astype(np.float32)
                    else: zLb = jnp.broadcast_to(z0[1], zL.shape)
                    d, D, la, lb, *_ = pair(x, zH, zL, jnp.asarray(zLb)); P1.setdefault((t, variant), []).append((np.asarray(d), np.asarray(D), ((dig(la) != dig(lb)) & ng[b0:b0 + B]).sum((1, 2)) / ng[b0:b0 + B].sum((1, 2))))
            la, zH, zL = one(x, zH, zL); base[t - 1, b0:b0 + B] = dig(la)
        for name in reset:                                                     # Part 2: reset one state at the start of every iteration
            zH2 = jnp.broadcast_to(z0[0], (B,) + zH0.shape); zL2 = jnp.broadcast_to(z0[1], (B,) + zL0.shape)
            for t in range(1, N_IT + 1):
                if t > 1 and name == "carry only the slow state": zL2 = jnp.broadcast_to(z0[1], zL2.shape)
                if t > 1 and name == "carry only the fast state": zH2 = jnp.broadcast_to(z0[0], zH2.shape)
                la, zH2, zL2 = one(x, zH2, zL2)
            reset[name][b0:b0 + B] = dig(la)
        print(f"  {key}: rows {b0 + B}/{N} ({time.time() - t_a:.0f}s)", flush=True)
    ref = ROOT / f"runs/analysis/commit_validity_20260919/{key}.npz"; R_ = np.load(ref, allow_pickle=True); assert (R_["ids"] == ids).all(); rp = R_["logits"].astype(np.float64).argmax(-1) + 1
    ev_exact = ((rp == sol[None]) | ~ng[None]).all((2, 3)); agree1 = float(((base[0] == rp[0]) & ng).sum() / ng.sum()); agree_solved = min(float(((base[t] == rp[t]) & ng)[ev_exact[t]].sum() / max(ng[ev_exact[t]].sum(), 1)) for t in range(4))
    exact = (base.astype(np.int64) == sol[None]).all((2, 3)); meta = dict(key=key, step=step, H=H, L=L, lam=float(lam), agree_iter1=agree1, agree_on_solved_through_4=agree_solved, wall=round(time.time() - t_a, 1))
    if agree1 <= 0.999 or agree_solved <= 0.999: print(f"FAILED-GATE {key}: {meta}; NOT saved", flush=True); return
    save = dict(base=base, exact=exact, sol=sol, puz=puz, ids=ids, meta=json.dumps(meta))
    for (t, variant), chunks in P1.items():
        save[f"delta_{variant}_{t}"] = np.concatenate([c[0] for c in chunks]); save[f"Delta_{variant}_{t}"] = np.concatenate([c[1] for c in chunks]); save[f"dis_{variant}_{t}"] = np.concatenate([c[2] for c in chunks])
    for name, pr in reset.items(): save["reset_" + name.replace(" ", "_")] = (pr.astype(np.int64) == sol).all((1, 2))
    OUT.mkdir(parents=True, exist_ok=True); tmp = OUT / f"{key}.tmp.npz"; np.savez_compressed(tmp, **save); os.replace(tmp, OUT / f"{key}.npz"); print(f"DONE {key}.npz {meta}", flush=True)

def report():
    Ls, J = [], {}; say = lambda s_="": (Ls.append(s_), print(s_)); have = [k for k in MODELS if (OUT / f"{k}.npz").exists()]
    say("IS THE FAST STATE SLAVED TO THE SLOW ONE? (tools/lens_fast_slow.py; registration Note_2026-09-19_Fast_Slow_Probe.md). 256 stratified puzzles, the fixed start, 16 iterations; one checkpoint per model.")
    say(f"completeness: {len(have)} of {len(MODELS)} models present ({', '.join(have)}). Registered predictions are scored only when all are present.")
    for k in have:
        D = np.load(OUT / f"{k}.npz", allow_pickle=True); meta = json.loads(str(D["meta"])); ex = D["exact"]; J[k] = dict(meta=meta, branches={}, reset={})
        say(); say(f"  {k} ({MODELS[k][2]}; H {meta['H']}, L {meta['L']}, damping {meta['lam']}; the probe = the evaluator on {100 * meta['agree_iter1']:.2f} % of the empty cells at iteration 1 and {100 * meta['agree_on_solved_through_4']:.2f} % on solved puzzles through 4) | solved at 16: {int(ex[-1].sum())} of {ex.shape[1]}")
        for variant, label in (("noise", f"a fast-state perturbation of {EPS} rms"), ("reinit", "the fast state replaced by its initial buffer")):
            for t in BRANCH:
                if f"delta_{variant}_{t}" not in D.files: continue
                s = summarize_branch(D[f"delta_{variant}_{t}"], D[f"Delta_{variant}_{t}"], D[f"dis_{variant}_{t}"], group_of(ex, t)); J[k]["branches"][f"{variant}@{t}"] = s
                say(f"    at the start of iteration {t}, {label}:"); [say(line(g, s[g])) for g in GROUPS if s[g]["n"]]
        b = ex[-1]
        for name in ("carry only the slow state", "carry only the fast state"):
            r = D["reset_" + name.replace(" ", "_")]; J[k]["reset"][name] = dict(exact=float(r.mean()), base=float(b.mean()), base_only=int((b & ~r).sum()), reset_only=int((r & ~b).sum()))
            say(f"    {name} (the other reset at every iteration): exact at 16 {100 * r.mean():.1f} % against {100 * b.mean():.1f} % unperturbed (unperturbed-only {int((b & ~r).sum())}, reset-only {int((r & ~b).sum())} of {len(b)})")
    if len(have) == len(MODELS):
        say(); say("== the registered predictions")
        g = lambda k, t, grp, f: J[k]["branches"].get(f"noise@{t}", {}).get(grp, {}).get(f)
        f1 = {k: [g(k, t, "already solved", "contraction") for t in (2, 4, 8) if g(k, t, "already solved", "contraction") is not None] for k in have}
        say(f"  F1 the fast loop contracts on already-solved puzzles (median delta_6/delta_0 < 1 at every branch) on all four: {'HELD' if all(v and max(v) < 1 for v in f1.values()) else 'FAILED'}; strongly (<= 0.1): {'HELD' if all(v and max(v) <= 0.1 for v in f1.values()) else 'FAILED'} ({ {k: [round(x, 4) for x in v] for k, v in f1.items()} })")
        f2 = {k: [(g(k, t, "not yet, solved by 16", "contraction"), g(k, t, "already solved", "contraction")) for t in (4, 8)] for k in have}
        say(f"  F2 weaker contraction on not-yet-solved than on already-solved puzzles at t = 4 and 8, all four: {'HELD' if all(a is not None and b is not None and a > b for v in f2.values() for a, b in v) else 'FAILED'} ({ {k: [(None if a is None else round(a, 4), None if b is None else round(b, 4)) for a, b in v] for k, v in f2.items()} })")
        f3 = {k: max([x for t in BRANCH for grp in GROUPS[1:] for x in [g(k, t, grp, "contraction")] if x is not None] or [0]) for k in have}
        say(f"  F3 the fast loop EXPANDS (median >= 1) on not-yet-solved puzzles on at least one model: {'HELD' if max(f3.values()) >= 1 else 'FAILED'} (largest median per model { {k: round(v, 4) for k, v in f3.items()} })")
        a = {k: J[k]["reset"]["carry only the slow state"] for k in have}; b_ = {k: J[k]["reset"]["carry only the fast state"] for k in have}
        say(f"  F4 carrying only the slow state changes accuracy by < 2 points on all four: {'HELD' if all(abs(v['exact'] - v['base']) < 0.02 for v in a.values()) else 'FAILED'} ({ {k: round(100 * (v['exact'] - v['base']), 1) for k, v in a.items()} } points); carrying only the fast state leaves < half on all four: {'HELD' if all(v['exact'] < 0.5 * v['base'] for v in b_.values()) else 'FAILED'} ({ {k: round(100 * v['exact'], 1) for k, v in b_.items()} } %)")
        f5 = {k: [g(k, t, "already solved", "to_slow_end") for t in (2, 4, 8) if g(k, t, "already solved", "to_slow_end") is not None] for k in have}
        say(f"  F5 on already-solved puzzles the perturbation barely reaches the slow state (median Delta_end/delta_0 < 0.1 at every branch) on all four: {'HELD' if all(v and max(v) < 0.1 for v in f5.values()) else 'FAILED'} ({ {k: [round(x, 4) for x in v] for k, v in f5.items()} })")
    (OUT / "report.txt").write_text("\n".join(Ls) + "\n"); (OUT / "report.json").write_text(json.dumps(J, indent=1, default=float))

# ---------------- selftest ----------------
def selftest():
    os.environ.setdefault("JAX_PLATFORMS", "cpu"); import jax, jax.numpy as jnp
    n = 0; a = 0.5                                                           # a toy fast map with a KNOWN contraction: z <- a z + inj; the slow update copies the fast state
    stp = lambda z, inj: a * z + inj; f = make_pair_iter(jnp, stp, lambda x: x, lambda zH: zH, 3, 6)
    x = jnp.ones(4); zH = jnp.zeros(4); zLa = jnp.ones(4); zLb = zLa + 0.1; d, D, la, lb, zHa, zLa2, zHb, zLb2 = f(x, zH, zLa, zLb); d = np.asarray(d); D = np.asarray(D)
    zs = [np.ones(4)]
    for _ in range(6): zs.append(a * zs[-1] + 1.0)                          # the unperturbed fast state in the first outer cycle (inj = z_H + emb = 1)
    assert len(d) == 7 and len(D) == 3 and np.allclose(d * np.asarray([np.linalg.norm(z) for z in zs]), np.linalg.norm(np.full(4, 0.1)) * a ** np.arange(7), rtol=1e-4); n += 1   # the ABSOLUTE difference shrinks by exactly a per inner update
    assert abs(d[6] / d[0] - (a ** 6) * np.linalg.norm(zs[0]) / np.linalg.norm(zs[6])) < 1e-5 and D[0] > 0 and float(jnp.abs(zHa - zLa2 * 0 - zHa).max()) == 0.0; n += 1            # the relative ratio is the absolute one rescaled by the state's norm
    g1 = make_one_iter(jnp, stp, lambda x: x, lambda zH: zH, 3, 6); lo, zHo, zLo = g1(x, zH, zLa); assert np.allclose(np.asarray(zHo), np.asarray(zHa)) and np.allclose(np.asarray(zLo), np.asarray(zLa2)) and np.allclose(np.asarray(lo), np.asarray(la)); n += 1   # one copy = copy a of the pair
    ex = np.zeros((4, 3), bool); ex[1:, 0] = True; ex[3, 1] = True; assert group_of(ex, 1).tolist() == [1, 1, 2] and group_of(ex, 3).tolist() == [0, 1, 2] and group_of(ex, 2).tolist() == [1, 1, 2]; n += 1
    delta = np.array([[.1, .05, .025, .0125, .00625, .003125, .0015625], [.1, .1, .1, .1, .1, .1, .2]]); Delta = np.array([[.01, .02, .001], [.1, .2, .3]]); s = summarize_branch(delta, Delta, np.array([0., .5]), np.array([0, 1]))
    assert abs(s["already solved"]["contraction"] - 0.015625) < 1e-12 and abs(s["already solved"]["per_update"] - 0.5) < 1e-9 and s["not yet, solved by 16"]["expanding_share"] == 1.0 and abs(s["already solved"]["to_slow_end"] - 0.01) < 1e-12 and s["never by 16"]["n"] == 0; n += 1
    assert line("already solved", s["already solved"]).startswith("      already solved           n   1 | fast difference after six inner updates x0.0156 (x0.500 per update; expanding on 0 % of puzzles)"); n += 1
    assert all((ROOT / v[0]).exists() for v in MODELS.values()) and all((ROOT / f"runs/analysis/commit_validity_20260919/{k}.npz").exists() for k in MODELS); n += 1
    print(f"selftest OK: {n}/{n}")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true"); ap.add_argument("--run", nargs="*"); ap.add_argument("--report", action="store_true"); a = ap.parse_args()
    if a.selftest: return selftest()
    if a.run is not None:
        for k in (a.run or list(MODELS)):
            if (OUT / f"{k}.npz").exists(): print(f"SKIP {k}.npz"); continue
            run_model(k)
    if a.report: report()

if __name__ == "__main__":
    main()
