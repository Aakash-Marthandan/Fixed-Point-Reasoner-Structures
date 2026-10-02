#!/usr/bin/env python3
# Ledger: DISCUSSION-PERIOD EXPERIMENT P9 (registration Documentation/Note_2026-10-01_Rebuttal_P7_P8_P9_Registration.md, written before
# any row). MEASUREMENT, $0, the Mac's CPU, inference only. Identical answers, different futures: at the 94k MLP 192 checkpoint, where the
# fixed start F and the independent Gaussian start G both display the correct grid at iteration t and only G is correct at t+1, swap state
# components between the two trajectories and read which component carries the next step. Loading = tools/rebuttal_p5.Runner.load
# (unchanged); the loop replicates tools/eval_sudoku_extreme.run_batch step for step with a per-puzzle pre-step hook.
"""  JAX_PLATFORMS=cpu .venv/bin/python tools/rebuttal_p9.py --selftest
  JAX_PLATFORMS=cpu nice -n 10 .venv/bin/python tools/rebuttal_p9.py --run
  .venv/bin/python tools/rebuttal_p9.py --report"""
from __future__ import annotations
import argparse, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tools"))
OUT = ROOT / "runs/analysis/rebuttal_20261001g"
SAVED = ROOT / "paper/code/evidence/initialization/s094000.npz"
STEP, N, T = 94000, 128, 16
CARRY_MIN, MIN_CASES, SHAM_MIN, SHIFT_MAX = 0.6, 20, 0.99, 1e-3
SWAPS = ("F<-hG", "F<-perpG", "F<-lG", "F<-zG", "G<-hF", "G<-perpF", "G<-lF", "G<-zF", "F<-sham")

def utc(): return datetime.now(timezone.utc).isoformat()

# ---------------- pure helpers (selftested) ----------------
def split(h, v):
    """h (..., w) float64, v (w,) -> (h_par, h_perp) along the shared readout vector."""
    u = v / np.linalg.norm(v); par = (h @ u)[..., None] * u
    return par, h - par

def hybrid(name, zF, zG, v):
    """zF, zG (2, F, S, w) float64 carries (slow, fast) of the two trajectories at the same moment -> the receiver's new carry."""
    hF, lF, hG, lG = zF[0], zF[1], zG[0], zG[1]
    pF, oF = split(hF, v); pG, oG = split(hG, v)
    table = {"F<-hG": (hG, lF), "F<-perpG": (pF + oG, lF), "F<-lG": (hF, lG), "F<-zG": (hG, lG),
             "G<-hF": (hF, lG), "G<-perpF": (pG + oF, lG), "G<-lF": (hG, lF), "G<-zF": (hF, lF), "F<-sham": (pF + oF, lF)}
    h, l = table[name]
    return np.stack([h, l])

def cases(exF, exG):
    """exF, exG (T, N) exact flags at iterations 1..T -> primary {puzzle: t} (both exact at t, only G exact at t+1; earliest t) and
    controls {puzzle: t} (both exact at t and at t+1; earliest t). t is 1-based."""
    prim, ctrl = {}, {}
    for n in range(exF.shape[1]):
        for t in range(1, exF.shape[0]):                                  # t = 1..T-1 (needs t+1 <= T)
            both = exF[t - 1, n] and exG[t - 1, n]
            if not both: continue
            if n not in prim and (not exF[t, n]) and exG[t, n]: prim[n] = t
            if n not in ctrl and exF[t, n] and exG[t, n]: ctrl[n] = t
    return prim, ctrl

def letter_p9(rescue, transfer, n_cases):
    """rescue/transfer: dicts over 'perp', 'h', 'l' of shares on the primary cases."""
    if n_cases < MIN_CASES: return "UNDEFINED"
    if rescue["perp"] >= CARRY_MIN and transfer["perp"] >= CARRY_MIN: return "STATE-CARRIES"
    if rescue["h"] >= CARRY_MIN and transfer["h"] >= CARRY_MIN: return "FULL-SLOW-ONLY"
    if rescue["l"] >= CARRY_MIN and transfer["l"] >= CARRY_MIN: return "FAST-CARRIES"
    return "NONE"

# ---------------- the run ----------------
class Loop:
    def __init__(self):
        import rebuttal_p5 as P5
        self.R = P5.Runner(n=N); self.m = self.R.load(STEP)
        self.jax, self.jnp, self.EV = self.R.jax, self.R.jnp, self.R.EV
        self.v = np.asarray(self.m["params"]["dec"]["lm_head"], np.float64)
        import lens_c5l_dynamics as LC
        self.zG0 = self.jnp.asarray(np.stack([self.EV.mi_z0(LC.MI_SEED, int(i), 0, self.m["shp"], 1.0, "gauss") for i in self.R.ids]))

    def run(self, start, hook=None, record=None):
        """start 'F' (the cell's fixed start) or 'G'; hook(s, z_c) -> z_c before step s (0-based) or None; record: {s: rows} -> states
        after step s for those rows. Mirrors eval_sudoku_extreme.run_batch for the dec cell (y is carried but not read by the cell)."""
        jax, jnp, EV, m = self.jax, self.jnp, self.EV, self.m; kw = m["kw"]; cfg = m["cfg"]
        y = m["y0"]; z_c = None if start == "F" else self.zG0; ex = []; rec = {}
        sol9 = jnp.asarray(kw["sol9"], jnp.int32)
        for s in range(T):
            if hook is not None and z_c is not None:
                z_new = hook(s, z_c)
                if z_new is not None: z_c = z_new
            first = z_c is None
            logits, zf = EV._step(cfg, float(kw["tau"]), 0.0, first)(m["params"], m["x_can"], y, m["tvj"], jnp.zeros(1) if first else z_c)
            z_c = zf if first else z_c + kw["eta_z"] * (zf - z_c)
            p = jax.nn.softmax(logits, axis=-1).transpose(0, 3, 1, 2)
            y = y + kw["eta"] * (p - y)
            pred9 = EV.layout_gather(jnp.argmax(logits, axis=-1), kw["layout"]).astype(jnp.int32)
            pred9 = jnp.where(pred9 == self.R.G.VOID, 0, pred9)
            ex.append(np.asarray(jnp.all((pred9 == sol9).reshape(N, -1), axis=1)))
            if record is not None and s in record:
                rec[s] = {int(r): np.asarray(z_c[r], np.float32).astype(np.float64) for r in record[s]}
        return np.stack(ex), rec

def run(log=print):
    OUT.mkdir(parents=True, exist_ok=True); t0 = time.time(); L = Loop()
    exF, _ = L.run("F"); exG, _ = L.run("G")
    S = np.load(SAVED, allow_pickle=False)
    gate1 = dict(F=bool(np.array_equal(exF, S["cold_ex"])), G=bool(np.array_equal(exG, S["ri_ex"])))
    log(f"gate 1 (saved 94k flags reproduced): {gate1}; F endpoint {int(exF[-1].sum())}, G endpoint {int(exG[-1].sum())} ({time.time() - t0:.0f}s)")
    prim, ctrl = cases(exF, exG); log(f"primary cases {len(prim)}, controls {len(ctrl)}")
    hooked = {**{n: t for n, t in ctrl.items()}, **{n: t for n, t in prim.items()}}             # one moment per puzzle (primary wins)
    need = {}
    for n, t in hooked.items(): need.setdefault(t - 1, []).append(n)                              # state after step t-1 (iteration t)
    _, recF = L.run("F", record=need); _, recG = L.run("G", record=need)
    res = dict(exF=exF, exG=exG); shift = {}
    for name in SWAPS:
        recv = name[0]; shift[name] = 0.0
        def hook(s, z_c, name=name):
            rows = [n for n, t in hooked.items() if t == s]                                         # before step s = iteration t+1
            if not rows: return None
            z = np.asarray(z_c, np.float32).astype(np.float64).copy()
            for n in rows:
                zF, zG = recF[s - 1][n], recG[s - 1][n]
                new = hybrid(name, zF, zG, L.v)
                if "perp" in name or "sham" in name:
                    base = zF[0] if recv == "F" else zG[0]
                    shift[name] = max(shift[name], float(np.abs(new[0].astype(np.float32).astype(np.float64) @ L.v - base @ L.v).max()))
                z[n] = new
            return L.jnp.asarray(z.astype(np.float32))
        ex, _ = L.run(recv, hook=hook); res["ex_" + name.replace("<-", "_from_")] = ex
        log(f"  {name}: done ({time.time() - t0:.0f}s)")
    meta = dict(step=STEP, n=N, created=utc(), wall=round(time.time() - t0, 1), gate1=gate1, primary={str(k): v for k, v in prim.items()},
                controls={str(k): v for k, v in ctrl.items()}, score_shift_max=shift)
    np.savez_compressed(OUT / "p9.tmp.npz", meta=json.dumps(meta), **res); os.replace(OUT / "p9.tmp.npz", OUT / "p9.npz")
    log(f"DONE {OUT / 'p9.npz'} {round(time.time() - t0)}s")

# ---------------- the report ----------------
def report(out=OUT):
    D = np.load(out / "p9.npz", allow_pickle=True); meta = json.loads(str(D["meta"])); Ls = []; say = lambda s_="": (Ls.append(s_), print(s_))
    prim = {int(k): v for k, v in meta["primary"].items()}; ctrl = {int(k): v for k, v in meta["controls"].items()}
    exF, exG = D["exF"], D["exG"]; JS = dict(meta=meta)
    def at(ex, n, t): return bool(ex[t, n])                                                          # iteration t+1 is row index t
    say(f"P9 report ({utc()}); gate 1 {meta['gate1']}; primary {len(prim)}, controls {len(ctrl)}; max score shift {meta['score_shift_max']}")
    g2 = all(at(D["ex_F_from_zG"], n, t) == at(exG, n, t) for n, t in prim.items()) and all(at(D["ex_G_from_zF"], n, t) == at(exF, n, t) for n, t in prim.items())
    hk = {**ctrl, **prim}; sham = np.mean([at(D["ex_F_from_sham"], n, t) == at(exF, n, t) for n, t in hk.items()]) if hk else float("nan")
    g4 = all(v <= SHIFT_MAX for k, v in meta["score_shift_max"].items())
    gates_ok = all(meta["gate1"].values()) and g2 and sham >= SHAM_MIN and g4
    say(f"gate 2 (full swaps reproduce the donor at t+1): {g2}; gate 3 (sham agreement) {sham:.3f}; gate 4 (score shift <= 1e-3): {g4}")
    rescue = {k: float(np.mean([at(D[f'ex_F_from_{k}G'], n, t) for n, t in prim.items()])) if prim else float("nan") for k in ("perp", "h", "l", "z")}
    transfer = {k: float(np.mean([not at(D[f'ex_G_from_{k}F'], n, t) for n, t in prim.items()])) if prim else float("nan") for k in ("perp", "h", "l", "z")}
    end_r = {k: float(np.mean([bool(D[f'ex_F_from_{k}G'][-1, n]) for n in prim])) if prim else float("nan") for k in ("perp", "h", "l", "z")}
    end_t = {k: float(np.mean([not bool(D[f'ex_G_from_{k}F'][-1, n]) for n in prim])) if prim else float("nan") for k in ("perp", "h", "l", "z")}
    L_ = letter_p9(rescue, transfer, len(prim)) if gates_ok else "UNDEFINED"
    say(f"rescue (F with G's component, exact at t+1): {rescue}")
    say(f"transfer (G with F's component, wrong at t+1): {transfer}")
    say(f"endpoint 16: F rescued {end_r}; G lost {end_t}")
    say(f"P9 letter: {L_}")
    cb = {k: float(np.mean([at(D[f'ex_F_from_{k}G'], n, t) for n, t in ctrl.items()])) if ctrl else float("nan") for k in ("perp", "h", "l")}
    cg = {k: float(np.mean([at(D[f'ex_G_from_{k}F'], n, t) for n, t in ctrl.items()])) if ctrl else float("nan") for k in ("perp", "h", "l")}
    say(f"controls (both stay correct intact): F after swap still exact {cb}; G after swap still exact {cg}")
    JS.update(gates=dict(g2=g2, sham=sham, g4=g4, ok=gates_ok), rescue=rescue, transfer=transfer, endpoint_rescue=end_r, endpoint_transfer=end_t,
              letter=L_, controls=dict(F=cb, G=cg))
    (out / "report.txt").write_text("\n".join(Ls) + "\n"); (out / "report.json").write_text(json.dumps(JS, indent=1, default=float))

# ---------------- selftest ----------------
def selftest():
    rng = np.random.default_rng(0); n = 0; v = rng.standard_normal(16)
    zF = rng.standard_normal((2, 9, 81, 16)); zG = rng.standard_normal((2, 9, 81, 16))
    h = hybrid("F<-perpG", zF, zG, v)
    assert np.abs(h[0] @ v - zF[0] @ v).max() < 1e-9 and np.array_equal(h[1], zF[1]); n += 1                     # F's scores and fast state kept
    _, oG = split(zG[0], v); _, oh = split(h[0], v); assert np.abs(oh - oG).max() < 1e-9; n += 1               # G's orthogonal part taken
    assert np.array_equal(hybrid("F<-zG", zF, zG, v), zG) and np.array_equal(hybrid("G<-zF", zF, zG, v), zF); n += 1
    assert np.abs(hybrid("F<-sham", zF, zG, v) - zF).max() < 1e-12; n += 1
    assert np.array_equal(hybrid("G<-lF", zF, zG, v)[1], zF[1]) and np.array_equal(hybrid("G<-lF", zF, zG, v)[0], zG[0]); n += 1
    exF = np.zeros((4, 3), bool); exG = np.zeros((4, 3), bool)
    exF[1, 0] = exG[1, 0] = True; exG[2, 0] = True                     # puzzle 0: both exact at t=2, only G at 3 -> primary t=2
    exF[0:3, 1] = exG[0:3, 1] = True                                   # puzzle 1: both exact at 1 and 2 -> control t=1
    p, c = cases(exF, exG); assert p == {0: 2} and c == {1: 1}; n += 1
    r = dict(perp=0.6, h=0.9, l=0.1); tr = dict(perp=0.6, h=0.9, l=0.1)
    assert letter_p9(r, tr, 20) == "STATE-CARRIES" and letter_p9(dict(r, perp=0.59), tr, 20) == "FULL-SLOW-ONLY" and letter_p9(r, tr, 19) == "UNDEFINED"; n += 1
    assert letter_p9(dict(perp=0.1, h=0.1, l=0.7), dict(perp=0.1, h=0.1, l=0.7), 30) == "FAST-CARRIES"; n += 1
    assert letter_p9(dict(perp=0.1, h=0.1, l=0.1), dict(perp=0.9, h=0.9, l=0.9), 30) == "NONE"; n += 1
    print(f"selftest OK ({n} checks)")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true"); ap.add_argument("--run", action="store_true")
    ap.add_argument("--report", action="store_true"); a = ap.parse_args()
    if a.selftest: selftest()
    if a.run: run()
    if a.report: report()

if __name__ == "__main__":
    main()
