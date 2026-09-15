#!/usr/bin/env python3
# Ledger: THE WIDTH-192 LONG RUN — the CPU lens (the PI 2026-09-15: "The integrity fail is trivial, go ahead with the CPU lens, nothing on
# the pod. We'll close sudoku with this"). DESCRIPTIVE, no rules; exploratory labels on every line. Zero cloud: the Mac reads the 75 banked
# C5 checkpoints (runs/_c5l_pull/x/runs/pretrainchamp_C5, 2k-150k) and the TPU instrument records already pulled.
#   Panel T (TPU records, every held-out grid): the cold D16 EMA pass on all 10,000 held-out and 1,000 train puzzles — exact at 16,
#            ever exact (first_exact >= 0; 0-based, -1 = never) and SOLVED-THEN-LOST (ever exact, not exact at 16).
#   Panel D (CPU, one warm process, the evaluator's own run_batch; D16, EMA weights; the first N held-out puzzles, idx 0..N-1).
#            The DEC reads NO canvas (dec_cell.forward_core: "the y slot is NOT read"), so the instruments differ only in the START
#            CARRY z = (z_H, z_L) (2, F, S, w). Training starts every fresh row from z ~ N(0, 1) i.i.d. (pretrain.py: TCm.z0(cfg, hw, rng=k)
#            with trm_ri_sigma 1); the benchmark's cold pass starts from the fixed buffers (one trunc-normal vector per stream, identical
#            in every cell and field) — a start the recipe never trains from.
#            COLD    the benchmark pass (the fixed buffers), cross-checked against Panel T on the same puzzles;
#            RI      the evaluator's multi-init draw j = 0 (mi_z0: z ~ N(0, 1) per puzzle, mi_seed 4242) — the training start family;
#            RIFIX   ONE seeded N(0, 1) carry shared by every puzzle (deterministic like COLD, from the training family);
#            SYM     one seeded N(0, 1) vector per stream broadcast over every cell and field (COLD's symmetric structure, another
#                    vector) — separates "all cells identical" from "that buffer vector"            [the SUBSET grids only]
#            ANCHOR  FPA's anchor: z_H = embed_answer(solution), z_L = the fixed buffer (the state FPA trains the map to hold):
#                    is the answer a fixed point? exact per step, the first step it is lost      [the SUBSET grids only]
#            each with exact at every step (lost = exact at some step, not at 16) and EqR's L = 3 residual (mean |dz|, last 3 steps).
#   Panel W (weights, all 75 checkpoints; the width-clock lens's groups and definitions, tools/lens_width_clocks.py): RMS per group,
#            relative Frobenius update per 2k steps (raw), the spectral relative update of blocks/0/mlp/gate_up (raw), raw-to-EMA distance
#            per group, top singular values of the channel matrices (EMA), Adam's gradient RMS sqrt(mean nu) and update coherence
#            ||mu|| / sqrt(sum nu) per group (the clipped gradient's first / second moments at that step).
"""  .venv/bin/python tools/lens_c5l_dynamics.py --selftest
  JAX_PLATFORMS=cpu .venv/bin/python tools/lens_c5l_dynamics.py --dynamics [--n 128] [--grids 2000,6000,...]   (resume-safe per grid)
  .venv/bin/python tools/lens_c5l_dynamics.py --weights
  .venv/bin/python tools/lens_c5l_dynamics.py --report"""
from __future__ import annotations
import argparse, json, os, sys, tempfile, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CK_DIR = ROOT / "runs/_c5l_pull/x/runs/pretrainchamp_C5"
X_ROWS, C8_ROWS = ROOT / "runs/_c5l_pull/x/runs", ROOT / "runs/_c8x_pull/x/runs"
VAL_NPZ = ROOT / "data/sudoku_extreme/sudoku_extreme_seed0_val10k.npz"
OUT = ROOT / "runs/analysis/c5l_lens_20260915"
GRIDS = tuple(range(2000, 150001, 4000))            # every 4k from 2k: 2k ... 150k (38 grids; contains 22k, 46k, 58k, 94k, 150k)
SUBSET = tuple(range(14000, 150001, 16000)) + (22000, 150000)   # every 16k from 14k (14k ... 142k) + the report's named 22k and the final 150k
PASSES, SUB_PASSES = ("cold", "ri", "rifix"), ("sym", "anchor")
MI_SEED, T_TOTAL = 4242, 16
GROUPS = ("channel", "token", "io")

# ---------------- pure helpers (selftested) ----------------
def group_of(path):
    if "/mlp_t/" in path: return "token"
    if path.endswith("/fc") or "/mlp/" in path: return "channel"
    if any(k in path for k in ("role_emb", "lm_head", "q_head")): return "io"
    return None

def leaves(t, p=""):
    if hasattr(t, "_fields"):
        for k in t._fields: yield from leaves(getattr(t, k), f"{p}/{k}")
    elif isinstance(t, dict):
        for k, v in t.items(): yield from leaves(v, f"{p}/{k}")
    elif isinstance(t, (list, tuple)):
        for i, v in enumerate(t): yield from leaves(v, f"{p}/{i}")
    else:
        yield p, np.asarray(t, dtype=np.float64)

def grouped(tree):
    g = {k: [] for k in GROUPS}; mats = {}
    for p, a in leaves(tree):
        k = group_of(p)
        if k: g[k].append(a.ravel())
        if k == "channel" and a.ndim == 2: mats[p] = a
    return {k: np.concatenate(v) for k, v in g.items() if v}, mats

def rel(a, b): return float(np.linalg.norm(b - a) / max(np.linalg.norm(a), 1e-12))

def adam_moments(opt_state):
    """The (mu, nu) model trees of the first ScaleByAdamState-like node (attributes mu and nu) in the optimizer state."""
    stack = [opt_state]
    while stack:
        t = stack.pop(0)
        if hasattr(t, "mu") and hasattr(t, "nu"):
            mu, nu = t.mu, t.nu
            return (mu.get("model", mu) if isinstance(mu, dict) else mu), (nu.get("model", nu) if isinstance(nu, dict) else nu)
        if isinstance(t, dict): stack.extend(t.values())
        elif isinstance(t, (list, tuple)): stack.extend(t)
    return None, None

def adam_stats(mu, nu):
    gm, _ = grouped(mu); gn, _ = grouped(nu)
    return {k: dict(grad_rms=float(np.sqrt(np.mean(gn[k]))), coherence=float(np.linalg.norm(gm[k]) / max(np.sqrt(np.sum(gn[k])), 1e-30)))
            for k in gm if k in gn}

def lost_stats(first_exact, cold_exact):
    fe, ce = np.asarray(first_exact), np.asarray(cold_exact).astype(bool)
    ever = fe >= 0
    return dict(n=int(len(fe)), cold=float(ce.mean()), ever=float(ever.mean()), lost=float((ever & ~ce).mean()))

def first_lost(ex_tb):
    """(T, B) exact rows of a retention pass -> (B,) first step (0-based) not exact, -1 = held at every step."""
    ex = np.asarray(ex_tb).astype(bool); miss = ~ex
    return np.where(miss.any(axis=0), miss.argmax(axis=0), -1)

def paired(a, b):
    a, b = np.asarray(a).astype(bool), np.asarray(b).astype(bool)
    return dict(both=int((a & b).sum()), a_only=int((a & ~b).sum()), b_only=int((~a & b).sum()), neither=int((~a & ~b).sum()))

# ---------------- Panel T ----------------
def tpu_rows(kind, step):
    if kind == "val":
        d = X_ROWS / f"c5l_val_pC5_s{step:06d}"
        if not (d / "records_all.npz").exists(): d = C8_ROWS / f"c8x_val_pC5_s{step:06d}"
    else:
        d = X_ROWS / f"c5l_tr1k_pC5_s{step:06d}"
    p = d / "records_all.npz"
    if not p.exists(): return None
    r = np.load(p, allow_pickle=True); s = json.loads((d / "summary_all.json").read_text())
    assert int(s["n"]) == len(r["idx"]) and int(s["t_total"]) == T_TOTAL and s.get("ema") and f"ckpt_{step:06d}" in str(s["ckpt"]), (d, s.get("ckpt"))
    return r

# ---------------- Panel D ----------------
def _jax():
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tools"))
    import jax, jax.numpy as jnp
    import eval_sudoku_extreme as EV
    from qhrrn2 import episodic as E, model as M, grid as G, sudoku as SU, sudoku_extreme as SX
    from qhrrn2.config import Config
    return jax, jnp, EV, E, M, G, SU, SX, Config

def dynamics(grids, n):
    jax, jnp, EV, E, M, G, SU, SX, Config = _jax()
    from qhrrn2 import dec_cell as DC
    out = OUT / "dyn"; out.mkdir(parents=True, exist_ok=True)
    d = SX.load_prepared(str(VAL_NPZ)); Q, A = d["val_q"], d["val_a"]
    ids = np.arange(n); puz9 = Q[ids].astype(np.int32); sol9 = A[ids].astype(np.int32)
    for step in grids:
        dst = out / f"s{step:06d}.npz"
        if dst.exists(): print(f"SKIP {step} (done)", flush=True); continue
        t0 = time.time(); saved = E.load_ckpt(str(CK_DIR / f"ckpt_{step:06d}.pkl")); assert int(saved["step"]) == step, (step, saved["step"])
        defaults = Config(); cfg = Config(**{k: type(getattr(defaults, k))(v) for k, v in saved["config"].items()})
        assert cfg.cell_kind == "dec" and cfg.trm_ri_sigma == 1.0, (cfg.cell_kind, cfg.trm_ri_sigma)
        st = saved["state_ema"]; params = st["model"]; tvj = jnp.asarray(st["table"][0])
        eta, eta_z = (float(v) for v in M.eq_etas(params, cfg)); ab = EV.coupled_ab(params, cfg); lay = getattr(cfg, "sudoku_layout", "origin") or "origin"
        x_can = EV.place_batch(puz9, lay); cv = SU.layout_canvas(lay); shp = tuple(M.carry_shape(cfg))
        kw = dict(t_total=T_TOTAL, tau=1.0, gamma=1.0, sol9=sol9, puz9=puz9, eta=eta, eta_z=eta_z, layout=lay, ab=ab)
        void = jax.nn.one_hot(jnp.full((cv, cv), G.VOID, jnp.int32), M.VOCAB).transpose(2, 0, 1)
        y0 = jnp.broadcast_to(void, (n,) + void.shape)          # the canvas is not read by the DEC; every pass hands the same void canvas
        starts = {"cold": None,
                  "ri": jnp.asarray(np.stack([EV.mi_z0(MI_SEED, int(i), 0, shp, 1.0, "gauss") for i in ids])),
                  "rifix": jnp.asarray(np.broadcast_to(EV.mi_z0(MI_SEED, 0, 0, shp, 1.0, "gauss"), (n,) + shp).copy())}
        if step in SUBSET:
            g = np.random.default_rng([MI_SEED, 0, 0, 11]).standard_normal((2, shp[-1])).astype(np.float32)
            starts["sym"] = jnp.asarray(np.broadcast_to(g[:, None, None, :], (n,) + shp).copy())
            zL0 = DC.z0(cfg, shp[2])[1]
            pdec = jax.tree_util.tree_map(jnp.asarray, params["dec"])            # checkpoint leaves are numpy; traced indexing needs jax arrays
            zH = jax.vmap(lambda y_: DC.embed_answer(pdec, cfg, y_))(jnp.asarray(sol9))
            starts["anchor"] = jnp.stack([zH, jnp.broadcast_to(zL0, zH.shape)], axis=1)
            assert starts["anchor"].shape == (n,) + shp, starts["anchor"].shape
        res = {}
        for name, z0 in starts.items():
            ex, _, _, rs = EV.run_batch(params, cfg, tvj, x_can, y0, z0=z0, **kw)
            res[f"{name}_ex"], res[f"{name}_res"] = ex, rs
        tmp = out / f"s{step:06d}.tmp.npz"
        np.savez(tmp, idx=ids, wall=time.time() - t0, ckpt=str(CK_DIR / f"ckpt_{step:06d}.pkl"), n=n, t_total=T_TOTAL, mi_seed=MI_SEED, **res)
        os.replace(tmp, dst)
        print(f"DONE {step}: " + " ".join(f"{k} {res[k + '_ex'][-1].mean():.3f}" for k in starts) + f" ({time.time()-t0:.0f}s)", flush=True)

# ---------------- Panel W ----------------
def weights_panel():
    sys.path.insert(0, str(ROOT / "src")); from qhrrn2 import episodic as E
    cks = sorted(CK_DIR.glob("ckpt_[0-9]*.pkl")); rows, prev = {}, None
    for ck in cks:
        s = E.load_ckpt(str(ck)); step = int(s["step"]); assert f"{step:06d}" in ck.name
        gr, mr = grouped(s["state"]["model"]); ge, me = grouped(s["state_ema"]["model"])
        r = {f"rms_{k}": float(np.sqrt(np.mean(ge[k] ** 2))) for k in ge}
        r.update({f"raw_ema_{k}": rel(ge[k], gr[k]) for k in ge})
        r.update({f"sig_{p.split('/dec/')[-1]}": float(np.linalg.norm(a, 2)) for p, a in me.items()})
        mu, nu = adam_moments(s["opt_state"])
        if mu is not None: r.update({f"{k2}_{k}": v2 for k, v in adam_stats(mu, nu).items() for k2, v2 in v.items()})
        gu = next(a for p, a in mr.items() if p.endswith("blocks/0/mlp/gate_up"))
        if prev is not None and step - prev[0] == 2000:
            r.update({f"du_{k}": rel(prev[1][k], gr[k]) for k in gr})
            r["du_spec"] = float(np.linalg.norm(gu - prev[2], 2) / np.linalg.norm(prev[2], 2))
        rows[step] = r; prev = (step, gr, gu)
        print(f"W {step}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "weights.json").write_text(json.dumps({str(k): v for k, v in rows.items()}, indent=1))

# ---------------- report ----------------
def report():
    W = {int(k): v for k, v in json.loads((OUT / "weights.json").read_text()).items()} if (OUT / "weights.json").exists() else {}
    L, J = [], {}
    say = lambda s="": (L.append(s), print(s))
    say("THE WIDTH-192 LONG RUN — CPU LENS (descriptive, exploratory; tools/lens_c5l_dynamics.py). D16, EMA weights unless named.")
    say("  T = TPU records (all puzzles): held-out cold % / solved-then-lost % | train-1k cold %   ·   D = CPU on the first N held-out puzzles, by START CARRY:")
    say("  cold % (agreement with T) · RI % · RIFIX % [· SYM % · ANCHOR % (median first-lost step)] · lost % cold/RI · paired cold-vs-RI [both, cold only, RI only, neither]")
    say("  W = weights: du channel (rel. update per 2k) · spec (gate_up0) · raw-EMA channel · sigma_max gate_up0/gate_up1 (EMA) · grad RMS channel · coherence channel")
    for step in sorted(set(GRIDS) | {s for s in W}):
        h, t = tpu_rows("val", step), tpu_rows("train", step)
        th = lost_stats(h["first_exact"], h["cold_exact"]) if h is not None else None
        tt = lost_stats(t["first_exact"], t["cold_exact"]) if t is not None else None
        dp = OUT / "dyn" / f"s{step:06d}.npz"; D = np.load(dp, allow_pickle=True) if dp.exists() else None
        row = dict(tpu_val=th, tpu_train=tt)
        seg = []
        if th: seg.append(f"T {100*th['cold']:5.1f}/{100*th['lost']:4.1f} | {100*tt['cold']:5.1f}" if tt else f"T {100*th['cold']:5.1f}/{100*th['lost']:4.1f} |   -  ")
        if D is not None:
            names = [k[:-3] for k in D.files if k.endswith("_ex")]
            fin = {k: np.asarray(D[f"{k}_ex"])[-1].astype(bool) for k in names}
            c = fin["cold"]; agree = None
            if h is not None:
                m = dict(zip(np.asarray(h["idx"]).tolist(), np.asarray(h["cold_exact"]).astype(bool).tolist()))
                agree = float(np.mean([bool(c[b]) == m[int(i)] for b, i in enumerate(np.asarray(D["idx"]))]))
            cpu = dict(n=int(D["n"]), agree_cold_vs_tpu=agree, cold_vs_ri=paired(c, fin["ri"]), cold_vs_rifix=paired(c, fin["rifix"]))
            for k in names:
                ex = np.asarray(D[f"{k}_ex"]).astype(bool)
                cpu[k] = dict(exact=float(fin[k].mean()), lost=float((ex.any(axis=0) & ~fin[k]).mean()), res_median=float(np.median(D[f"{k}_res"])))
            if "anchor" in names:
                fl = first_lost(D["anchor_ex"]); cpu["anchor"]["first_lost_median"] = float(np.median(fl[fl >= 0])) if (fl >= 0).any() else None
            row["cpu"] = cpu; pr = cpu["cold_vs_ri"]
            sub = ""
            if "sym" in names:
                flm = cpu["anchor"]["first_lost_median"]
                sub = f" · SYM {100*cpu['sym']['exact']:5.1f} · ANCHOR {100*cpu['anchor']['exact']:5.1f}({'-' if flm is None else f'{flm:.0f}'})"
            seg.append(f"D {100*cpu['cold']['exact']:5.1f}({'-' if agree is None else f'{100*agree:.0f}'}) · RI {100*cpu['ri']['exact']:5.1f} · RIFIX {100*cpu['rifix']['exact']:5.1f}{sub} "
                       f"· lost {100*cpu['cold']['lost']:.0f}/{100*cpu['ri']['lost']:.0f} [{pr['both']},{pr['a_only']},{pr['b_only']},{pr['neither']}]")
        w = W.get(step)
        if w:
            row["weights"] = w
            seg.append(f"W {w.get('du_channel', float('nan')):.4f} {w.get('du_spec', float('nan')):.4f} {w['raw_ema_channel']:.4f} "
                       f"{w.get('sig_blocks/0/mlp/gate_up', float('nan')):.2f}/{w.get('sig_blocks/1/mlp/gate_up', float('nan')):.2f} "
                       f"{w.get('grad_rms_channel', float('nan')):.2e} {w.get('coherence_channel', float('nan')):.3f}")
        if seg: say(f"  {step//1000:>3}k  " + "  ·  ".join(seg)); J[str(step)] = row
    (OUT / "lens.txt").write_text("\n".join(L) + "\n"); (OUT / "lens.json").write_text(json.dumps(J, indent=1))

# ---------------- selftest ----------------
def selftest():
    n = 0
    assert [group_of(p) for p in ("/dec/blocks/0/fc", "/dec/blocks/1/mlp/down", "/dec/blocks/0/mlp_t/gate_up", "/dec/q_head/w", "/eq/eta")] == \
        ["channel", "channel", "token", "io", None]; n += 1
    tree = {"dec": {"blocks": [{"fc": np.ones((2, 2)), "mlp": {"down": 2 * np.ones((2, 1))}, "mlp_t": {"down": np.zeros((1, 1))}}], "q_head": {"w": np.ones(3)}},
            "eq": {"eta": np.float32(1.0)}}
    g, m = grouped(tree)
    assert sorted(g) == ["channel", "io", "token"] and g["channel"].tolist() == [1, 1, 1, 1, 2, 2] and len(m) == 2; n += 1
    assert abs(rel(np.array([3.0, 4.0]), np.array([3.0, 4.0 + 5.0])) - 1.0) < 1e-12; n += 1
    class S:                                                                              # a ScaleByAdamState stand-in
        _fields = ("count", "mu", "nu")
        def __init__(s, mu, nu): s.count, s.mu, s.nu = 1, mu, nu
    mu = {"model": {"dec": {"blocks": [{"fc": np.array([[1.0, 1.0], [1.0, 1.0]])}]}}}; nu = {"model": {"dec": {"blocks": [{"fc": np.full((2, 2), 4.0)}]}}}
    a_mu, a_nu = adam_moments((object(), (S(mu, nu), object())))
    st = adam_stats(a_mu, a_nu); assert abs(st["channel"]["grad_rms"] - 2.0) < 1e-12 and abs(st["channel"]["coherence"] - 0.5) < 1e-12; n += 1
    assert lost_stats([0, 3, -1, 5], [1, 0, 0, 1]) == dict(n=4, cold=0.5, ever=0.75, lost=0.25); n += 1
    ex = np.array([[1, 1, 1], [1, 0, 1], [0, 0, 1]], bool)                                   # (T=3, B=3)
    assert first_lost(ex).tolist() == [2, 1, -1]; n += 1
    assert paired([1, 1, 0, 0], [1, 0, 1, 0]) == dict(both=1, a_only=1, b_only=1, neither=1); n += 1
    assert GRIDS[0] == 2000 and GRIDS[-1] == 150000 and len(GRIDS) == 38 and all(s in GRIDS for s in (22000, 46000, 58000, 94000)); n += 1
    print(f"selftest OK: {n}/{n}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true"); ap.add_argument("--dynamics", action="store_true"); ap.add_argument("--weights", action="store_true")
    ap.add_argument("--report", action="store_true"); ap.add_argument("--n", type=int, default=128); ap.add_argument("--grids", default=None)
    ap.add_argument("--out", default=None, help="output dir (default runs/analysis/c5l_lens_20260915); smokes write elsewhere so the resume-skip never keeps a smoke file")
    a = ap.parse_args()
    if a.selftest: return selftest()
    global OUT
    if a.out: OUT = Path(a.out)
    if a.weights: weights_panel()
    if a.dynamics: dynamics([int(x) for x in a.grids.split(",")] if a.grids else list(GRIDS), a.n)
    if a.report: report()

if __name__ == "__main__":
    main()
