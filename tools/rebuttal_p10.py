#!/usr/bin/env python3
# Ledger: DISCUSSION-PERIOD EXPERIMENT P10 (registration Documentation/Note_2026-10-01_Rebuttal_P10_P11_Registration.md, written before
# any row). MEASUREMENT, $0, the Mac's CPU, inference only. On the inside-completion cohorts (C5, SA128), the slow state's readout-
# orthogonal part is ROTATED by theta (score- and norm-preserving) once, k = 1, 2, 3 outer iterations before the completing cycle, and the
# trajectory is continued to readout 96. Accounts: SENSITIVE-TRANSIENT / ACCUMULATION / RECOVERY-COST / IDLE / MIXED. The lens's jitted
# cycle (tools/lens_inside_completion.make_cycle_intervene, every flag off) is reused unchanged; the edit is float64 outside the jit.
"""  JAX_PLATFORMS=cpu .venv/bin/python tools/rebuttal_p10.py --selftest
  JAX_PLATFORMS=cpu nice -n 10 .venv/bin/python tools/rebuttal_p10.py --run C5     (resume-safe per model; run C5 and SA128 as two processes)
  .venv/bin/python tools/rebuttal_p10.py --report"""
from __future__ import annotations
import argparse, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tools"))
SEP25 = ROOT / "runs/analysis/inside_completion_20260925"
P8OUT = ROOT / "runs/analysis/rebuttal_20261001f"
OUT = ROOT / "runs/analysis/rebuttal_20261001h"
SEED, BS, J_EXT = 20261001, 64, 96
CONDS = [(k, th) for k in (1, 2, 3) for th in (0, 15, 90)] + [(1, 45), (3, 45)]
MIN_ELIG, SHAM_MIN, SHIFT_MAX, REPL_TOL = 30, 0.99, 1e-3, 0.10
P8_REF = {"C5": 0.157, "SA128": 0.107}                         # P8's perp_random@-3 exact-at-kappa (gate B), from its report

def utc(): return datetime.now(timezone.utc).isoformat()

# ---------------- pure helpers (selftested) ----------------
def rotate_perp(h, v, theta_deg, r):
    """h (..., w), v (w,), r (..., w) Gaussian draws -> h' = h_par + cos(t) h_perp + sin(t) |h_perp| u, with u a unit vector orthogonal to
    v and to h_perp (Gram-Schmidt of r). Keeps every score h.v and every norm |h|; theta = 0 reassembles h (the sham). float64."""
    h = np.asarray(h, np.float64); v = np.asarray(v, np.float64); r = np.asarray(r, np.float64)
    e = v / np.linalg.norm(v)
    par = (h @ e)[..., None] * e; perp = h - par
    pn = np.linalg.norm(perp, axis=-1, keepdims=True); pu = perp / np.maximum(pn, 1e-300)
    u = r - (r @ e)[..., None] * e; u = u - np.sum(u * pu, axis=-1, keepdims=True) * pu
    u = u / np.maximum(np.linalg.norm(u, axis=-1, keepdims=True), 1e-300)
    t = np.deg2rad(theta_deg)
    return par + np.cos(t) * perp + np.sin(t) * pn * u

def first_exact_from(ex, start):
    """ex (J, N) exact flags at readouts 1..J; start (N,) 1-based readout at which to begin looking -> (N,) first exact readout >= start, 0 if none."""
    J, N = ex.shape; out = np.zeros(N, np.int64)
    for n in range(N):
        later = np.flatnonzero(ex[start[n] - 1:, n])
        out[n] = start[n] + later[0] if len(later) else 0
    return out

def shares(delta, never):
    """delta (n,) readouts (tau' - kappa), never (n,) bool -> dict of shares; never counts as delayed and as |delta| >= 3."""
    n = max(len(delta), 1)
    d = np.where(never, 10 ** 6, delta)
    return dict(delayed=float((d >= 1).sum() / n), earlier=float((d <= -1).sum() / n), unchanged=float((d == 0).sum() / n),
                big=float((np.abs(d) >= 3).sum() / n), never=float(never.sum() / n))

def med(delta, never, kappa, horizon=J_EXT):
    """median delta with 'never' set to horizon - kappa + 1."""
    d = np.where(never, horizon - kappa + 1, delta)
    return float(np.median(d)) if len(d) else float("nan")

def letter_p10(S):
    """S: {(k, theta): dict(n, shares..., m)} for one model (only eligible (k, theta) present). Returns (letter, flags)."""
    ok = lambda k, th: (k, th) in S and S[(k, th)]["n"] >= MIN_ELIG
    f = {}
    f["sensitive"] = any(ok(k, 15) and S[(k, 15)]["big"] >= 0.25 and S[(k, 15)]["earlier"] >= 0.15 for k in (1, 2, 3))
    f["accumulation"] = (ok(3, 90) and ok(1, 90) and S[(3, 90)]["delayed"] >= 0.5 and S[(1, 90)]["m"] >= S[(3, 90)]["m"] + 2
                         and all(S[(k, 90)]["earlier"] <= 0.10 for k in (1, 2, 3) if ok(k, 90))
                         and ok(1, 15) and ok(1, 45) and S[(1, 15)]["delayed"] < S[(1, 45)]["delayed"] < S[(1, 90)]["delayed"])
    elig90 = [k for k in (1, 2, 3) if ok(k, 90)]
    f["recovery"] = (len(elig90) > 0 and all(S[(k, 90)]["delayed"] >= 0.5 for k in elig90) and ok(1, 90) and ok(3, 90)
                     and abs(S[(1, 90)]["m"] - S[(3, 90)]["m"]) <= 1)
    f["idle"] = ok(3, 90) and S[(3, 90)]["unchanged"] >= 0.7
    for name, key in (("SENSITIVE-TRANSIENT", "sensitive"), ("ACCUMULATION", "accumulation"), ("RECOVERY-COST", "recovery"), ("IDLE", "idle")):
        if f[key]: return name, f
    return "MIXED", f

# ---------------- the run ----------------
def run_model(key, log=print):
    import lens_corpus_normalized as CN, lens_inside_completion as LIC, rebuttal_p8 as P8
    OUT.mkdir(parents=True, exist_ok=True); dst = OUT / f"{key}.npz"
    if dst.exists(): log(f"SKIP {key} (done)"); return
    S = np.load(SEP25 / f"{key}.npz", allow_pickle=True)
    M = LIC.build(key); jax, jnp = M["jax"], M["jnp"]; L = M["L"]
    ids, puz, sol = CN.puzzles("strat", 256)
    assert np.array_equal(ids, S["ids"]) and np.array_equal(puz, S["puz"]), "population differs from Sep 25"
    solf = sol.reshape(len(puz), 81); idx = S["cohort"].astype(np.int64); kappa = S["kappa"].astype(np.int64)
    v = P8.readout_vector(key); zH0, zL0 = M["zH0"], M["zL0"]
    f = jax.jit(jax.vmap(LIC.make_cycle_intervene(jnp, M["stp"], None, M["emb"], M["scores"], L), in_axes=(0, 0, 0, None, None, 0, 0, 0)))
    nf = jnp.zeros(BS, bool); t0 = time.time(); save = dict(cohort=idx, kappa=kappa, ids=ids); shift = {}
    for cond in [None] + CONDS:
        name = "intact" if cond is None else f"k{cond[0]}_t{cond[1]}"; smax = 0.0
        ex_all = np.zeros((J_EXT, len(idx)), bool)
        for c0 in range(0, len(idx), BS):
            sel = idx[c0:c0 + BS]; keep = len(sel); selp = np.concatenate([sel, np.repeat(sel[-1:], BS - keep)]) if keep < BS else sel
            x = jnp.asarray(puz[selp]); zH = jnp.broadcast_to(zH0, (BS,) + zH0.shape); zL = jnp.broadcast_to(zL0, (BS,) + zL0.shape)
            target = (kappa[selp] - 3 * cond[0]) if cond is not None else np.full(BS, -1)
            elig = target >= 2; ex = np.zeros((J_EXT, BS), bool)
            for j in range(1, J_EXT + 1):
                hit = elig & (target == j)
                if cond is not None and hit.any():
                    h = np.asarray(zH, np.float32).astype(np.float64); new = h.copy(); rows = np.flatnonzero(hit)
                    for b in rows:
                        r = np.random.default_rng([SEED, int(ids[selp[b]]), 101, int(j)]).standard_normal(h[b].shape)
                        new[b] = rotate_perp(h[b], v, cond[1], r)
                    new32 = new.astype(np.float32); smax = max(smax, float(np.abs(new32[rows].astype(np.float64) @ v - h[rows] @ v).max()))
                    zH = jnp.asarray(new32)
                zH, zL, dg = f(x, zH, zL, zH0, zL0, nf, nf, nf); ex[j - 1] = (np.asarray(dg) == solf[selp]).all(-1)
            ex_all[:, c0:c0 + keep] = ex[:, :keep]
        save[f"{name}_ex"] = ex_all; shift[name] = smax
        log(f"  {key}: {name} done; max score shift {smax:.2e} ({time.time() - t0:.0f}s)")
    meta = dict(key=key, J=J_EXT, batch=BS, seed=SEED, conds=CONDS, created=utc(), wall=round(time.time() - t0, 1), score_shift_max=shift,
                registration="Documentation/Note_2026-10-01_Rebuttal_P10_P11_Registration.md")
    tmp = OUT / f"{key}.tmp.npz"; np.savez_compressed(tmp, meta=json.dumps(meta), **save); os.replace(tmp, dst); log(f"DONE {key} wall {meta['wall']}s")

# ---------------- the report ----------------
def report():
    Ls = []; JS = dict(created=utc()); say = lambda s_="": (Ls.append(s_), print(s_))
    say(f"P10 report ({JS['created']}); registration Documentation/Note_2026-10-01_Rebuttal_P10_P11_Registration.md")
    for key in ("C5", "SA128"):
        p = OUT / f"{key}.npz"
        if not p.exists(): say(f"\n== {key}: not run"); continue
        D = np.load(p, allow_pickle=True); meta = json.loads(str(D["meta"])); idx = D["cohort"]; kap = D["kappa"][idx]
        exI = D["intact_ex"]; n = len(idx)
        gD = bool(all(exI[kap[i] - 1, i] for i in range(n))); persist = float(exI[-1].mean())
        say(f"\n== {key}: cohort {n}; gate D (intact exact at kappa on every puzzle): {gD}; intact exact at {J_EXT}: {persist:.3f}; "
            f"max score shift {max(meta['score_shift_max'].values()):.2e} (gate C {'PASS' if max(meta['score_shift_max'].values()) <= SHIFT_MAX else 'FAIL'})")
        St = {}; JK = dict(gateD=gD, intact_exact_end=persist, conds={})
        for (k, th) in CONDS:
            ex = D[f"k{k}_t{th}_ex"]; start = kap - 3 * k; elig = start >= 2
            tp = first_exact_from(ex[:, elig], start[elig]); never = tp == 0; delta = np.where(never, 0, tp - kap[elig])
            s = shares(delta, never); s["n"] = int(elig.sum()); s["m"] = med(delta, never, kap[elig])
            s["exact_at_kappa"] = float(np.mean([ex[kap[i] - 1, i] for i in np.flatnonzero(elig)])); s["exact_end"] = float(ex[-1, elig].mean())
            St[(k, th)] = s; JK["conds"][f"k{k}_t{th}"] = s
            say(f"   k={k} theta={th:>2}: n {s['n']:3d}; delayed {s['delayed']:.3f} earlier {s['earlier']:.3f} unchanged {s['unchanged']:.3f} "
                f"|d|>=3 {s['big']:.3f} never {s['never']:.3f}; median delta {s['m']:.1f}; exact at kappa {s['exact_at_kappa']:.3f}; exact at {J_EXT} {s['exact_end']:.3f}")
        gA = all(St[(k, 0)]["unchanged"] >= SHAM_MIN for k in (1, 2, 3)); JK["gateA_sham"] = gA
        gB = abs(St[(1, 90)]["exact_at_kappa"] - P8_REF[key]) <= REPL_TOL; JK["gateB_replication"] = gB
        L_, flags = letter_p10({kk: vv for kk, vv in St.items() if kk[1] != 0}) if gA and gD else ("UNDEFINED", {})
        JK.update(letter=L_, flags=flags)
        say(f"   gate A (sham unchanged >= {SHAM_MIN} at every k): {gA}; gate B (k=1, 90 deg vs P8's replacement at kappa-3: "
            f"{St[(1, 90)]['exact_at_kappa']:.3f} vs {P8_REF[key]:.3f}): {'PASS' if gB else 'ROTATION DIFFERS FROM REPLACEMENT'}")
        say(f"   P10 letter: {L_}  conditions {flags}")
        JS[key] = JK
    (OUT / "report.txt").write_text("\n".join(Ls) + "\n"); (OUT / "report.json").write_text(json.dumps(JS, indent=1, default=float))

# ---------------- selftest ----------------
def selftest():
    rng = np.random.default_rng(0); n = 0; v = rng.standard_normal(24); h = rng.standard_normal((9, 81, 24)); r = rng.standard_normal((9, 81, 24))
    e = v / np.linalg.norm(v); perp = lambda x: x - (x @ e)[..., None] * e
    for th in (15, 45, 90):
        h2 = rotate_perp(h, v, th, r)
        assert np.abs(h2 @ v - h @ v).max() < 1e-9 and np.abs(np.linalg.norm(h2, axis=-1) - np.linalg.norm(h, axis=-1)).max() < 1e-9
        p1, p2 = perp(h), perp(h2); cosang = np.sum(p1 * p2, -1) / (np.linalg.norm(p1, axis=-1) * np.linalg.norm(p2, axis=-1))
        assert np.abs(cosang - np.cos(np.deg2rad(th))).max() < 1e-9
    n += 3
    assert np.abs(rotate_perp(h, v, 0, r) - h).max() < 1e-12; n += 1
    def mutant(h, v, th, r):                                    # forgets to remove the readout direction from r: scores must move
        e_ = v / np.linalg.norm(v); par = (h @ e_)[..., None] * e_; pp = h - par; pn = np.linalg.norm(pp, axis=-1, keepdims=True)
        u = r / np.linalg.norm(r, axis=-1, keepdims=True); t = np.deg2rad(th); return par + np.cos(t) * pp + np.sin(t) * pn * u
    assert np.abs(mutant(h, v, 90, r) @ v - h @ v).max() > 1e-3; n += 1
    ex = np.zeros((6, 3), bool); ex[3, 0] = True; ex[1, 1] = True; ex[5, 1] = True
    assert list(first_exact_from(ex, np.array([2, 3, 1]))) == [4, 6, 0]; n += 1
    base = dict(n=50, delayed=0.0, earlier=0.0, unchanged=1.0, big=0.0, never=0.0, m=0.0)
    S = {(k, th): dict(base) for k in (1, 2, 3) for th in (15, 90)}; S[(1, 45)] = dict(base); S[(3, 45)] = dict(base)
    assert letter_p10(S)[0] == "IDLE"; n += 1
    S2 = {kk: dict(vv) for kk, vv in S.items()}; S2[(2, 15)].update(big=0.3, earlier=0.2); assert letter_p10(S2)[0] == "SENSITIVE-TRANSIENT"; n += 1
    S3 = {kk: dict(vv) for kk, vv in S.items()}
    for k in (1, 2, 3): S3[(k, 90)].update(delayed=0.8, unchanged=0.2, m=4.0)
    assert letter_p10(S3)[0] == "RECOVERY-COST"; n += 1
    S4 = {kk: dict(vv) for kk, vv in S3.items()}; S4[(1, 90)].update(m=7.0); S4[(1, 15)].update(delayed=0.2); S4[(1, 45)].update(delayed=0.5)
    assert letter_p10(S4)[0] == "ACCUMULATION"; n += 1
    S5 = {kk: dict(vv) for kk, vv in S4.items()}; S5[(1, 45)].update(delayed=0.1); assert letter_p10(S5)[0] == "MIXED"; n += 1     # dose not monotone, |m1-m3| = 3
    S6 = {kk: dict(vv) for kk, vv in S.items()}; S6[(3, 90)].update(n=29); assert letter_p10(S6)[0] == "MIXED"; n += 1             # k=3 undefined -> no IDLE
    print(f"selftest OK ({n} checks)")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true"); ap.add_argument("--run", nargs="*"); ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.selftest: selftest()
    if a.run:
        for k in a.run: run_model(k)
    if a.report: report()

if __name__ == "__main__":
    main()
