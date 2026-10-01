#!/usr/bin/env python3
# Ledger: THE DISCUSSION-PERIOD EXPERIMENT P5 — access robustness at the 94k checkpoint (registration
# Documentation/Note_2026-10-01_Rebuttal_P5_Registration.md, written before this build). MEASUREMENT, $0; saved records
# (paper/code/evidence/initialization) plus CPU inference on the banked MLP 192 long-run checkpoints through the release evaluator,
# with the lens's conventions (tools/lens_c5l_dynamics.py, imported unchanged).
"""P5a: the post-50k distribution of fixed-start vs Gaussian-start successes over the 38-checkpoint series (saved records).
P5b: sixteen further Gaussian draws (eight independent, eight shared) at 94k, four at 90k / 98k / 46k, through the same evaluator;
gate: the fixed start and the original seed-4242 draws reproduce the saved flags exactly."""
import argparse, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import lens_c5l_dynamics as LC                               # constants and the JAX import helper, unchanged

EVID = ROOT / "paper/code/evidence/initialization"
OUT = ROOT / "runs/analysis/rebuttal_20261001c"
SEED, N, T_TOTAL = 20261001, 128, 16
FULL_STEP, CONTEXT_STEPS = 94000, (90000, 98000, 46000)
NEW_FULL, NEW_CONTEXT = 8, 2                                  # per start kind
STARTS = ("cold", "ri", "rifix", "sym", "anchor")
ROBUST_MIN, FRAGILE_MAX, FIXED_SUCCESSES_94K = 100, 68, 11
POST50K = 54000


def utc(): return datetime.now(timezone.utc).isoformat()


# ---------------- P5a: the saved series ----------------
def series():
    rows = {}
    for f in sorted(EVID.glob("s*.npz")):
        step = int(f.stem[1:])
        with np.load(f, allow_pickle=False) as d:
            r = dict(n=int(d["n"]), mi_seed=int(d["mi_seed"]), t_total=int(d["t_total"]))
            for s in STARTS:
                if f"{s}_ex" in d.files:
                    ex = d[f"{s}_ex"]; r[s] = dict(endpoint=int(ex[-1].sum()), ever=int(ex.any(0).sum()), terminal_losses=int((ex.any(0) & ~ex[-1]).sum()))
        rows[step] = r
    return rows


def post50k_summary(rows):
    steps = sorted(s for s in rows if s >= POST50K); out = dict(steps=steps, n_checkpoints=len(steps))
    for s in ("cold", "ri", "rifix"):
        v = np.array([rows[k][s]["endpoint"] for k in steps]); ev = np.array([rows[k][s]["ever"] for k in steps])
        out[s] = dict(median=float(np.median(v)), q1=float(np.percentile(v, 25)), q3=float(np.percentile(v, 75)), min=int(v.min()), max=int(v.max()),
                      ever_median=float(np.median(ev)), terminal_losses_total=int(sum(rows[k][s]["terminal_losses"] for k in steps)))
    cold = np.array([rows[k]["cold"]["endpoint"] for k in steps]); ri = np.array([rows[k]["ri"]["endpoint"] for k in steps])
    gap = ri - cold
    out["gap_ri_minus_cold"] = dict(at_least_50=int((gap >= 50).sum()), at_least_20=int((gap >= 20).sum()), under_5=int((gap < 5).sum()), median=float(np.median(gap)))
    rank = int((cold < rows[FULL_STEP]["cold"]["endpoint"]).sum()) + 1
    out["rank_of_94k_fixed_start_ascending"] = rank; out["is_minimum"] = bool(rank == 1)
    return out


def gate_series(rows):
    r = rows[FULL_STEP]
    expect = dict(cold=(11, 68), ri=(124, 124), rifix=(124, 124), sym=(121, 125), anchor=(128, 128))
    for s, (end, ever) in expect.items():
        assert r[s]["endpoint"] == end and r[s]["ever"] == ever, (s, r[s], end, ever)
    return True


# ---------------- P5b: the draws through the release evaluator ----------------
class Runner:
    def __init__(self, n=N):
        jax, jnp, EV, E, M, G, SU, SX, Config = LC._jax()
        from qhrrn2 import dec_cell as DC
        self.jax, self.jnp, self.EV, self.E, self.M, self.G, self.SU, self.SX, self.Config, self.DC = jax, jnp, EV, E, M, G, SU, SX, Config, DC
        d = SX.load_prepared(str(LC.VAL_NPZ)); self.n = n
        self.ids = np.arange(n); self.puz9 = d["val_q"][self.ids].astype(np.int32); self.sol9 = d["val_a"][self.ids].astype(np.int32)

    def load(self, step):
        jax, jnp, EV, E, M, G, SU, SX, Config = self.jax, self.jnp, self.EV, self.E, self.M, self.G, self.SU, self.SX, self.Config
        ck = LC.CK_DIR / f"ckpt_{step:06d}.pkl"; saved = E.load_ckpt(str(ck)); assert int(saved["step"]) == step
        defaults = Config(); cfg = Config(**{k: type(getattr(defaults, k))(v) for k, v in saved["config"].items()})
        assert cfg.cell_kind == "dec" and cfg.trm_ri_sigma == 1.0
        st = saved["state_ema"]; params = st["model"]; tvj = jnp.asarray(st["table"][0])
        eta, eta_z = (float(v) for v in M.eq_etas(params, cfg)); ab = EV.coupled_ab(params, cfg); lay = getattr(cfg, "sudoku_layout", "origin") or "origin"
        x_can = EV.place_batch(self.puz9, lay); cv = SU.layout_canvas(lay); shp = tuple(M.carry_shape(cfg))
        kw = dict(t_total=T_TOTAL, tau=1.0, gamma=1.0, sol9=self.sol9, puz9=self.puz9, eta=eta, eta_z=eta_z, layout=lay, ab=ab)
        void = jax.nn.one_hot(jnp.full((cv, cv), G.VOID, jnp.int32), M.VOCAB).transpose(2, 0, 1)
        y0 = jnp.broadcast_to(void, (self.n,) + void.shape)
        return dict(step=step, ckpt=str(ck), params=params, cfg=cfg, tvj=tvj, x_can=x_can, y0=y0, shp=shp, kw=kw)

    def starts(self, m, new_per_kind):
        jnp, EV, shp, n = self.jnp, self.EV, m["shp"], self.n
        s = {"cold": None,
             "ri_orig": jnp.asarray(np.stack([EV.mi_z0(LC.MI_SEED, int(i), 0, shp, 1.0, "gauss") for i in self.ids])),
             "rifix_orig": jnp.asarray(np.broadcast_to(EV.mi_z0(LC.MI_SEED, 0, 0, shp, 1.0, "gauss"), (n,) + shp).copy())}
        for j in range(1, new_per_kind + 1):
            s[f"ri_new_{j}"] = jnp.asarray(np.stack([EV.mi_z0(SEED, int(i), j, shp, 1.0, "gauss") for i in self.ids]))
            s[f"rifix_new_{j}"] = jnp.asarray(np.broadcast_to(EV.mi_z0(SEED, 0, j, shp, 1.0, "gauss"), (n,) + shp).copy())
        return s

    def run_step(self, step, new_per_kind, out, gate_saved=None, log=print, gate_only=False):
        dst = out / f"s{step:06d}.npz"
        if dst.exists() and not gate_only: log(f"SKIP {step} (done)"); return np.load(dst, allow_pickle=False)
        m = self.load(step); res = {}; t0 = time.time()
        for name, z0 in self.starts(m, 0 if gate_only else new_per_kind).items():
            ex, _, _, rs = self.EV.run_batch(m["params"], m["cfg"], m["tvj"], m["x_can"], m["y0"], z0=z0, **m["kw"])
            res[f"{name}_ex"] = np.asarray(ex, bool); res[f"{name}_res"] = np.asarray(rs, np.float32)
            log(f"{step} {name}: endpoint {int(res[name + '_ex'][-1].sum())} ever {int(res[name + '_ex'].any(0).sum())} ({time.time() - t0:.0f}s)")
        gate = None
        if gate_saved is not None:
            with np.load(gate_saved, allow_pickle=False) as d:
                k = self.n
                gate = dict(cold=bool(np.array_equal(res["cold_ex"], d["cold_ex"][:, :k])), ri=bool(np.array_equal(res["ri_orig_ex"], d["ri_ex"][:, :k])),
                            rifix=bool(np.array_equal(res["rifix_orig_ex"], d["rifix_ex"][:, :k])))
                detail = {s: dict(differing_flags=int((res[f"{a}_ex"] != d[f"{s}_ex"][:, :k]).sum()), differing_puzzles=int((res[f"{a}_ex"] != d[f"{s}_ex"][:, :k]).any(0).sum()),
                                  endpoint_ours=int(res[f"{a}_ex"][-1].sum()), endpoint_saved=int(d[f"{s}_ex"][-1, :k].sum()), n=k)
                          for s, a in (("cold", "cold"), ("ri", "ri_orig"), ("rifix", "rifix_orig"))}
                gate["detail"] = detail
            out.mkdir(parents=True, exist_ok=True); (out / f"gate_s{step:06d}_n{self.n}.json").write_text(json.dumps(gate, indent=1))
            assert all(v for k_, v in gate.items() if k_ != "detail"), f"gate failed at {step}: {gate}"
        if gate_only: log(f"GATE-ONLY {step}: {gate}"); return None
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / f"s{step:06d}.tmp.npz", idx=self.ids, n=self.n, t_total=T_TOTAL, seed=SEED, mi_seed=LC.MI_SEED, ckpt=m["ckpt"], wall=time.time() - t0,
                 gate=json.dumps(gate), **res)
        os.replace(out / f"s{step:06d}.tmp.npz", dst); log(f"DONE {step} gate {gate}")
        return np.load(dst, allow_pickle=False)


def letter(draw_endpoints, preserved_all):
    """draw_endpoints: endpoint counts of the sixteen new draws at 94k; preserved_all: all fixed-start successes kept in every draw."""
    if not draw_endpoints: return "UNDEFINED"
    if min(draw_endpoints) <= FRAGILE_MAX: return "FRAGILE"
    if min(draw_endpoints) >= ROBUST_MIN and preserved_all: return "ROBUST"
    return "MIXED"


def report(out=OUT):
    rows = series(); gate_series(rows); summ = post50k_summary(rows)
    lines = ["P5 REPORT (rules registered 2026-10-01): access robustness at the 94k MLP 192 checkpoint"]
    lines.append(f"P5a series gate at 94k reproduced (11/68, 124, 124, 121, 128). Post-50k checkpoints (>= {POST50K}): n {summ['n_checkpoints']}")
    for s in ("cold", "ri", "rifix"):
        v = summ[s]; lines.append(f"  {s}: endpoint median {v['median']:.0f} [q1 {v['q1']:.0f}, q3 {v['q3']:.0f}] range {v['min']}-{v['max']} | ever median {v['ever_median']:.0f} | terminal losses total {v['terminal_losses_total']}")
    g = summ["gap_ri_minus_cold"]; lines.append(f"  independent minus fixed: median {g['median']:.0f}; >= 50 at {g['at_least_50']} checkpoints, >= 20 at {g['at_least_20']}, < 5 at {g['under_5']}; 94k fixed-start rank (ascending) {summ['rank_of_94k_fixed_start_ascending']} of {summ['n_checkpoints']} (minimum: {summ['is_minimum']})")
    res = dict(series=rows, post50k=summ, draws={})
    for step in (FULL_STEP,) + CONTEXT_STEPS:
        f = out / f"s{step:06d}.npz"
        if not f.exists(): lines.append(f"  {step}: not run"); continue
        with np.load(f, allow_pickle=False) as d:
            names = sorted(k[:-3] for k in d.files if k.endswith("_ex")); cold = d["cold_ex"][-1]
            r = {}
            for nm in names:
                ex = d[nm + "_ex"]; r[nm] = dict(endpoint=int(ex[-1].sum()), ever=int(ex.any(0).sum()), preserved_fixed=int((cold & ex[-1]).sum()), losses_after_success=int((ex.any(0) & ~ex[-1]).sum()))
            gate = json.loads(str(d["gate"])) if "gate" in d.files else None
        new = [nm for nm in names if "_new_" in nm]; ends = [r[nm]["endpoint"] for nm in new]
        res["draws"][step] = dict(gate=gate, results=r)
        lines.append(f"  {step}: gate {gate} | fixed {r['cold']['endpoint']} (ever {r['cold']['ever']}) | ri_orig {r['ri_orig']['endpoint']} rifix_orig {r['rifix_orig']['endpoint']} | new draws n {len(new)}: median {np.median(ends) if ends else None} min {min(ends) if ends else None} max {max(ends) if ends else None} | preserved all fixed successes in every new draw: {all(r[nm]['preserved_fixed'] == r['cold']['endpoint'] for nm in new) if new else None}")
        if step == FULL_STEP:
            L = letter(ends, all(r[nm]["preserved_fixed"] == r["cold"]["endpoint"] for nm in new)) if new else "NOT RUN"
            res["letter"] = L; lines.append(f"P5b letter at 94k: {L}")
    (out / "report.txt").write_text("\n".join(lines) + "\n"); (out / "report.json").write_text(json.dumps(res, indent=1, default=float))
    print("\n".join(lines))


def selftest():
    assert letter([124, 120, 110, 100], True) == "ROBUST" and letter([124, 99], True) == "MIXED" and letter([124, 60], True) == "FRAGILE"
    assert letter([124, 120], False) == "MIXED" and letter([], True) == "UNDEFINED"
    rows = series(); assert len(rows) == 38 and gate_series(rows)
    s = post50k_summary(rows); assert s["n_checkpoints"] == 25 and 1 <= s["rank_of_94k_fixed_start_ascending"] <= 25
    # the draw convention: deterministic, puzzle-specific for ri, shared for rifix, and distinct from the original seed
    jax, jnp, EV, *_ = LC._jax(); shp = (2, 9, 81, 4)
    a = EV.mi_z0(SEED, 3, 1, shp, 1.0, "gauss"); b = EV.mi_z0(SEED, 3, 1, shp, 1.0, "gauss"); c = EV.mi_z0(SEED, 4, 1, shp, 1.0, "gauss"); o = EV.mi_z0(LC.MI_SEED, 3, 0, shp, 1.0, "gauss")
    assert np.array_equal(a, b) and not np.array_equal(a, c) and not np.array_equal(a, o) and a.dtype == np.float32
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", nargs="?", choices=["run", "report"]); ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--gate-only", action="store_true", help="full batch, the fixed start and the original draws only; writes gate_s*.json")
    ap.add_argument("--out", default=None); ap.add_argument("--steps", default=None, help="comma-separated checkpoint steps (default: 94k then 90k, 98k, 46k)")
    a = ap.parse_args()
    if a.selftest: selftest(); return
    out = Path(a.out) if a.out else OUT
    if a.action == "run":
        out.mkdir(parents=True, exist_ok=True); log = lambda s: (print(s, flush=True), (out / "run.log").open("a").write(f"{utc()} {s}\n"))
        n = 16 if a.smoke else N; r = Runner(n)
        steps = [int(x) for x in a.steps.split(",")] if a.steps else [FULL_STEP] + list(CONTEXT_STEPS)
        for step in steps:
            new = (NEW_FULL if step == FULL_STEP else NEW_CONTEXT); new = 1 if a.smoke else new
            r.run_step(step, new, out, gate_saved=(EVID / f"s{step:06d}.npz"), log=log, gate_only=a.gate_only)
        if not a.smoke and not a.gate_only: report(out)
    elif a.action == "report": report(out)
    else: ap.print_help()


if __name__ == "__main__":
    main()
