#!/usr/bin/env python3
# Ledger: DISCUSSION-PERIOD EXPERIMENT P8 (registration Documentation/Note_2026-10-01_Rebuttal_P7_P8_P9_Registration.md, written
# before any row). MEASUREMENT, $0, the Mac's CPU, inference only. The inside-completion lens's pass C (tools/lens_inside_completion.py,
# imported unchanged) re-run with (i) the missing INTACT control through the same jitted cycle, (ii) the Sep 25 slow reset at kappa as a
# reproduction gate, (iii) a score-preserving replacement of the slow state's readout-orthogonal component at the start of the completing
# cycle (and one outer iteration earlier), with a reassembly sham. The edit is applied in float64 outside the jitted cycle (P2/P3's
# convention) and cast back to float32; the jitted path is the lens's own make_cycle_intervene with every flag off.
"""  JAX_PLATFORMS=cpu .venv/bin/python tools/rebuttal_p8.py --selftest
  JAX_PLATFORMS=cpu nice -n 10 .venv/bin/python tools/rebuttal_p8.py --run C5 SA128 EQR [--smoke]     (resume-safe per model)
  .venv/bin/python tools/rebuttal_p8.py --report"""
from __future__ import annotations
import argparse, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tools"))
SEP25 = ROOT / "runs/analysis/inside_completion_20260925"
OUT = ROOT / "runs/analysis/rebuttal_20261001f"
SEED = 20261001
BS = 64
B_HIGH, B_LOW, ADMIT, SHAM_MIN, SHIFT_MAX = 0.5, 0.1, 0.98, 0.99, 1e-3
PERP_MODELS = ("C5", "SA128")                      # DEC readout = one shared vector; EQR gets intact + the reproduction gate only


def utc(): return datetime.now(timezone.utc).isoformat()


# ---------------- pure helpers (selftested) ----------------
def perp_replace(h, v, r):
    """h (..., w) slow-state vectors, v (w,) the shared readout vector, r (..., w) Gaussian draws -> h' with h'.v == h.v (every digit
    score kept) and ||h'|| == ||h|| (each vector's norm kept): h' = h_par + r_perp * ||h_perp|| / ||r_perp||. float64 throughout."""
    h = np.asarray(h, np.float64); v = np.asarray(v, np.float64); r = np.asarray(r, np.float64)
    u = v / np.linalg.norm(v)
    par = (h @ u)[..., None] * u; perp = h - par
    rp = r - (r @ u)[..., None] * u
    scale = np.linalg.norm(perp, axis=-1, keepdims=True) / np.maximum(np.linalg.norm(rp, axis=-1, keepdims=True), 1e-300)
    return par + rp * scale

def perp_sham(h, v):
    """The same decomposition reassembled with nothing changed (the numerical control for the edit path)."""
    h = np.asarray(h, np.float64); u = np.asarray(v, np.float64); u = u / np.linalg.norm(u)
    par = (h @ u)[..., None] * u
    return par + (h - par)

def blocked(p_cond, p_intact):
    """B = 1 - P(exact at kappa | condition) / P(exact at kappa | intact)."""
    return float("nan") if p_intact <= 0 else 1.0 - p_cond / p_intact

def letter_r8(B):
    if not np.isfinite(B): return "UNDEFINED"
    return "NEEDS-HIDDEN-STATE" if B >= B_HIGH else ("READOUT-SUFFICES" if B <= B_LOW else "MIXED")


# ---------------- the run ----------------
def readout_vector(key):
    import lens_corpus_normalized as CN, lens_inside_completion as LIC
    path, ema, _, _ = LIC.MODELS[key]
    _jax, _jnp, cfg, st, _step = CN._load(path, ema)
    assert cfg.cell_kind == "dec", key
    return np.asarray(st["model"]["dec"]["lm_head"], np.float64)

def run_model(key, smoke=False, log=print):
    import lens_corpus_normalized as CN, lens_inside_completion as LIC
    out = OUT / "smoke" if smoke else OUT; out.mkdir(parents=True, exist_ok=True)
    dst = out / f"{key}.npz"
    if dst.exists(): log(f"SKIP {key} (done)"); return
    S = np.load(SEP25 / f"{key}.npz", allow_pickle=True)
    M = LIC.build(key); jax, jnp = M["jax"], M["jnp"]; H, L = M["H"], M["L"]; J = LIC.N_IT * H
    ids, puz, sol = CN.puzzles("strat", 256)
    assert np.array_equal(ids, S["ids"]) and np.array_equal(puz, S["puz"]) and np.array_equal(sol, S["sol"]), "population differs from Sep 25"
    solf = sol.reshape(len(puz), 81); idx = S["cohort"].astype(np.int64); kappa = S["kappa"].astype(np.int64)
    if smoke: idx = idx[:BS]
    v = readout_vector(key) if key in PERP_MODELS else None
    zH0, zL0 = M["zH0"], M["zL0"]
    f = jax.jit(jax.vmap(LIC.make_cycle_intervene(jnp, M["stp"], None, M["emb"], M["scores"], L), in_axes=(0, 0, 0, None, None, 0, 0, 0)))
    conds = [("intact", 0), ("slow_reset", 0)] + ([("perp_sham", 0), ("perp_random", 0), ("perp_random", -3)] if v is not None else [])
    R = {}; t0 = time.time(); shift_max = {}
    for cond, off in conds:
        at_k = np.zeros(len(idx), bool); at_end = np.zeros(len(idx), bool); first_after = np.zeros(len(idx), np.int32); smax = 0.0
        for c0 in range(0, len(idx), BS):                                  # pass C's batching, padding by repeating the last puzzle
            sel = idx[c0:c0 + BS]; keep = len(sel); selp = np.concatenate([sel, np.repeat(sel[-1:], BS - keep)]) if keep < BS else sel
            x = jnp.asarray(puz[selp]); target = kappa[selp] + off
            zH = jnp.broadcast_to(zH0, (BS,) + zH0.shape); zL = jnp.broadcast_to(zL0, (BS,) + zL0.shape); ex = np.zeros((J, BS), bool)
            for j in range(1, J + 1):
                hit = target == j
                sr = jnp.asarray(hit & (cond == "slow_reset")); fr = jnp.zeros(BS, bool); nm = jnp.zeros(BS, bool)
                if cond in ("perp_sham", "perp_random") and hit.any():
                    h = np.asarray(zH, np.float32).astype(np.float64)              # (BS, F, S, w)
                    rows = np.flatnonzero(hit); new = h.copy()
                    for b in rows:
                        if cond == "perp_sham":
                            new[b] = perp_sham(h[b], v)
                        else:
                            r = np.random.default_rng([SEED, int(ids[selp[b]]), 81, int(j)]).standard_normal(h[b].shape)
                            new[b] = perp_replace(h[b], v, r)
                    new32 = new.astype(np.float32)
                    smax = max(smax, float(np.abs(new32[rows].astype(np.float64) @ v - h[rows] @ v).max()))
                    zH = jnp.asarray(new32)
                zH, zL, dg = f(x, zH, zL, zH0, zL0, fr, sr, nm); ex[j - 1] = (np.asarray(dg) == solf[selp]).all(-1)
            for n in range(keep):
                k1 = kappa[sel[n]] - 1; at_k[c0 + n] = ex[k1, n]; at_end[c0 + n] = ex[-1, n]
                ts = kappa[sel[n]] + off - 1; later = np.flatnonzero(ex[ts:, n]); first_after[c0 + n] = (ts + later[0] + 1) if len(later) else 0
        name = f"{cond}@{off}"; R[name] = dict(exact_at_kappa=at_k, exact_at_end=at_end, first_exact_after=first_after); shift_max[name] = smax
        log(f"  {key}: {name}: exact at kappa {at_k.mean():.3f}, at {J} {at_end.mean():.3f}, max score shift {smax:.2e} ({time.time() - t0:.0f}s)")
    meta = dict(key=key, n_cohort=int(len(idx)), J=J, batch=BS, seed=SEED, created=utc(), wall=round(time.time() - t0, 1), smoke=smoke,
                score_shift_max=shift_max, registration="Documentation/Note_2026-10-01_Rebuttal_P7_P8_P9_Registration.md")
    save = dict(cohort=idx, kappa=kappa, ids=ids, meta=json.dumps(meta))
    for k_, d_ in R.items():
        for f_, arr in d_.items(): save[f"{k_}_{f_}"] = arr
    tmp = out / f"{key}.tmp.npz"; np.savez_compressed(tmp, **save); os.replace(tmp, dst); log(f"DONE {key} {meta}")


# ---------------- the report ----------------
def report(out=OUT):
    Ls = []; JS = {}; say = lambda s_="": (Ls.append(s_), print(s_))
    say(f"P8 report ({utc()}); registration Documentation/Note_2026-10-01_Rebuttal_P7_P8_P9_Registration.md")
    for key in ("C5", "SA128", "EQR"):
        p = out / f"{key}.npz"
        if not p.exists(): say(f"\n== {key}: not run"); continue
        D = np.load(p, allow_pickle=True); S = np.load(SEP25 / f"{key}.npz", allow_pickle=True); meta = json.loads(str(D["meta"]))
        n = len(D["cohort"]); full = n == len(S["cohort"])
        ik = D["intact@0_exact_at_kappa"]; p_int = float(ik.mean()); admit = p_int >= ADMIT
        g1 = full and bool(np.array_equal(D["slow_reset@0_exact_at_kappa"], S["int_slow_reset@0_exact_at_kappa"])) \
            and bool(np.array_equal(D["slow_reset@0_exact_at_end"], S["int_slow_reset@0_exact_at_end"]))
        J = dict(n=n, intact_exact_at_kappa=p_int, intact_exact_at_end=float(D["intact@0_exact_at_end"].mean()),
                 admission="PASS" if admit else "REPLAY-DRIFT", gate1_sep25_reproduced=g1, score_shift_max=meta["score_shift_max"])
        say(f"\n== {key}: cohort {n}; intact replay exact at kappa {p_int:.3f} ({'ADMISSION PASS' if admit else 'REPLAY-DRIFT'}), at 48 {J['intact_exact_at_end']:.3f}; "
            f"Sep 25 slow-reset flags reproduced: {g1}")
        if "perp_sham@0_exact_at_kappa" in D.files:
            sham_agree = float((D["perp_sham@0_exact_at_kappa"] == ik).mean()); J["sham_agreement"] = sham_agree
            pr = D["perp_random@0_exact_at_kappa"]; Bp = blocked(float(pr.mean()), p_int); J["B_perp_random@0"] = Bp; J["R8"] = letter_r8(Bp)
            say(f"   sham agreement with intact at kappa {sham_agree:.3f} ({'PASS' if sham_agree >= SHAM_MIN else 'SHAM NOT NEUTRAL'}); "
                f"max score shift {max(meta['score_shift_max'].get(k_, 0) for k_ in meta['score_shift_max']):.2e}")
            say(f"   R8: perp_random at kappa: exact at kappa {pr.mean():.3f} -> B = {Bp:.3f} -> {J['R8']}")
            for nm_ in ("perp_random@0", "perp_random@-3"):
                fa = D[f"{nm_}_first_exact_after"]; kk = D["kappa"][D["cohort"]]; delay = np.where(fa > 0, fa - kk, np.nan)
                J[nm_] = dict(exact_at_kappa=float(D[f"{nm_}_exact_at_kappa"].mean()), exact_at_end=float(D[f"{nm_}_exact_at_end"].mean()),
                              median_delay_readouts=float(np.nanmedian(delay)) if np.isfinite(delay).any() else None, never=int((fa == 0).sum()))
                say(f"   {nm_}: exact at kappa {J[nm_]['exact_at_kappa']:.3f}, exact at 48 {J[nm_]['exact_at_end']:.3f}, "
                    f"median delay {J[nm_]['median_delay_readouts']} readouts, never exact again {J[nm_]['never']}")
        resc = {}
        for v_ in ("fast_reset", "slow_reset", "no_messages"):
            for off in (0, -3):
                k_ = f"int_{v_}@{off}_exact_at_kappa"
                if k_ in S.files and full:
                    resc[f"{v_}@{off}"] = dict(exact_at_kappa=float(S[k_].mean()), B=blocked(float(S[k_].mean()), p_int),
                                                exact_at_end=float(S[f"int_{v_}@{off}_exact_at_end"].mean()))
        J["sep25_rescored"] = resc
        for k_, d_ in resc.items():
            say(f"   Sep 25 {k_:16s}: exact at kappa {d_['exact_at_kappa']:.3f} -> B against intact {d_['B']:.3f}; exact at 48 {d_['exact_at_end']:.3f}")
        JS[key] = J
    OUT.mkdir(parents=True, exist_ok=True)
    (out / "report.txt").write_text("\n".join(Ls) + "\n"); (out / "report.json").write_text(json.dumps(JS, indent=1, default=float))


# ---------------- selftest ----------------
def selftest():
    rng = np.random.default_rng(0); n = 0
    v = rng.standard_normal(32); h = rng.standard_normal((9, 81, 32)); r = rng.standard_normal((9, 81, 32))
    h2 = perp_replace(h, v, r)
    assert np.abs(h2 @ v - h @ v).max() < 1e-9; n += 1                                                       # scores kept
    assert np.abs(np.linalg.norm(h2, axis=-1) - np.linalg.norm(h, axis=-1)).max() < 1e-9; n += 1             # norms kept
    assert np.abs(h2 - h).max() > 0.1; n += 1                                                                 # something changed
    assert np.abs(perp_sham(h, v) - h).max() < 1e-12; n += 1                                                  # sham is an identity
    def mutant(h, v, r):                                                                                      # forgets to project r: must break scores
        u = v / np.linalg.norm(v); par = (h @ u)[..., None] * u; perp = h - par
        return par + r * (np.linalg.norm(perp, axis=-1, keepdims=True) / np.linalg.norm(r, axis=-1, keepdims=True))
    assert np.abs(mutant(h, v, r) @ v - h @ v).max() > 1e-3; n += 1                                          # the score check catches it
    assert letter_r8(blocked(0.5, 1.0)) == "NEEDS-HIDDEN-STATE" and letter_r8(blocked(0.51, 1.0)) == "MIXED"; n += 1
    assert letter_r8(blocked(0.9, 1.0)) == "READOUT-SUFFICES" and letter_r8(blocked(0.89, 1.0)) == "MIXED"; n += 1
    assert letter_r8(blocked(0.4, 0.8)) == "NEEDS-HIDDEN-STATE" and letter_r8(blocked(0.2, 0.0)) == "UNDEFINED"; n += 1
    print(f"selftest OK ({n} checks)")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true"); ap.add_argument("--run", nargs="*")
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--report", action="store_true"); a = ap.parse_args()
    if a.selftest: selftest()
    if a.run:
        for k in a.run: run_model(k, smoke=a.smoke)
    if a.report: report()


if __name__ == "__main__":
    main()
