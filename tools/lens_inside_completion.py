#!/usr/bin/env python3
# Ledger: INSIDE THE COMPLETING CYCLE (2026-09-25; registration Documentation/Note_2026-09-25_Inside_Completion.md, written before any
# row). DESCRIPTIVE, zero cloud, inference only. The model's own segment replicated one FAST update at a time (lens_fast_slow's loader
# and step). Pass A: slow-state scores after every slow update (M1), shadow readouts of the fast state (M2: the slow update applied
# after fast update k), the linear readout applied to the fast state (M3). Pass B: distance of the slow state to its final value (M4).
# Pass C: one-cycle interventions at the completing cycle and one outer iteration earlier (M5).
"""  JAX_PLATFORMS=cpu .venv/bin/python tools/lens_inside_completion.py --selftest
  JAX_PLATFORMS=cpu nice -n 10 .venv/bin/python tools/lens_inside_completion.py --run C5 SA128 EQR      (resume-safe per model)
  .venv/bin/python tools/lens_inside_completion.py --report"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tools"))

OUT = ROOT / "runs/analysis/inside_completion_20260925"
MODELS = {   # key: (checkpoint, EMA, label, gate reference)
    "C5":    ("runs/pretrainchamp_C5/ckpt_046000.pkl", True, "MLP 192, seed 0, 46k (the benchmark)", "validity"),
    "SA128": ("runs/analysis/paper_update_20260920/new_runs/selected_checkpoints/SA128_ckpt_042000.pkl", True, "Attention 128, 42k (the benchmark)", "evaluator"),
    "EQR":   ("runs/field_ckpts/ported/eqr/ckpt_latest.pkl", True, "EqR's released weights, ported", "validity"),
}
N_IT, BS, COHORT_MIN = 16, 64, 7
SMOKE = bool(os.environ.get("LENS_SMOKE"))
if SMOKE: OUT = OUT / "smoke"
VARIANTS = ("fast_reset", "slow_reset", "no_messages")
OFFSETS = (0, -3)

# ---------------- pure helpers (selftested) ----------------
def margin_rank(xp, scores, sol):
    """scores (..., 81, 9) digit scores, sol (..., 81) digits 1..9 -> (margin, rank, digit), each (..., 81).
    margin = score(correct) - max score over the other eight digits; rank = 1 + number of digits scoring STRICTLY higher."""
    c = (sol - 1)[..., None]
    sc = xp.take_along_axis(scores, c, axis=-1)[..., 0]
    other = xp.where(xp.arange(9) == c, -xp.inf, scores).max(-1)
    rank = 1 + (scores > sc[..., None]).sum(-1)
    return sc - other, rank, scores.argmax(-1) + 1

def first_exact(exact):
    """exact (J, N) bool over readouts j = 1..J -> kappa (N,) 1-based first exact readout, 0 if never."""
    return np.where(exact.any(0), exact.argmax(0) + 1, 0)

def cohort_of(exact, kmin=COHORT_MIN):
    """Cohort: first exact at kappa >= kmin and exact at the last readout."""
    k = first_exact(exact)
    return np.flatnonzero((k >= kmin) & exact[-1]), k

def closure(margins, wrong, kappa, idx):
    """margins (J, N, 81), wrong (J, N, 81) bool (empty cells only), kappa (N,) -> per cohort puzzle the closure
    [m(k-1) - m(k-4)] / [0 - m(k-4)], m = median margin over cells wrong at every readout k-4..k-1; NaN if undefined."""
    out = np.full(len(idx), np.nan)
    for n, i in enumerate(idx):
        a = kappa[i] - 1                                      # 0-based index of readout kappa
        P = wrong[a - 4:a, i].all(0)
        if not P.any(): continue
        m4, m1 = np.median(margins[a - 4, i][P]), np.median(margins[a - 1, i][P])
        if m4 < 0: out[n] = (m1 - m4) / (0 - m4)
    return out

def rank2_share(rank, wrong, kappa, idx):
    """Pooled share of cells wrong at readout kappa-1 whose correct digit ranks second there."""
    num = den = 0
    for i in idx:
        a = kappa[i] - 1; W = wrong[a - 1, i]
        num += int((rank[a - 1, i][W] == 2).sum()); den += int(W.sum())
    return num / den if den else float("nan")

def displacement_ratio(rel, kappa, idx):
    """rel (J, N): ||h_j - h_{j-1}|| / ||h_{j-1}|| recorded after slow update j -> per puzzle rel(kappa) / median rel(kappa-5..kappa-1)."""
    out = []
    for i in idx:
        a = kappa[i] - 1
        out.append(rel[a, i] / np.median(rel[a - 5:a, i]))
    return np.asarray(out)

def approach_jump(dist, kappa, idx):
    """dist (J, N) distance to the final slow state -> (1 - d(k-1)/d(k-6), 1 - d(k)/d(k-1)) per puzzle."""
    ap, jp = [], []
    for i in idx:
        a = kappa[i] - 1
        ap.append(1 - dist[a - 1, i] / dist[a - 6, i]); jp.append(1 - dist[a, i] / dist[a - 1, i])
    return np.asarray(ap), np.asarray(jp)

def within_cycle(shadow_err, err, kappa, idx):
    """shadow_err (J, K, N): error of the shadow readout after fast update k of cycle j; err (J, N) readout error.
    -> (S6 flags: shadow error after the FIRST fast update of cycle kappa >= half the readout error at kappa-1;
        S7 flags: some shadow of cycle kappa-1 exact)."""
    s6, s7 = [], []
    for i in idx:
        a = kappa[i] - 1
        s6.append(shadow_err[a, 0, i] >= 0.5 * err[a - 1, i]); s7.append(bool((shadow_err[a - 1, :, i] == 0).any()))
    return np.asarray(s6), np.asarray(s7)

def letter(value, h_rule, c_rule):
    return "H" if h_rule(value) else ("C" if c_rule(value) else "MIXED")

# ---------------- the model ----------------
def build(key):
    import lens_corpus_normalized as CN
    path, ema, _, _ = MODELS[key]; jax, jnp, cfg, st, step = CN._load(path, ema); assert cfg.cell_kind in ("trm", "dec"); dec = cfg.cell_kind == "dec"
    from qhrrn2 import dec_cell as DC, trm_cell as TC
    C = DC if dec else TC; p = jax.tree_util.tree_map(jnp.asarray, st["model"]["dec" if dec else "trm"]); hw = cfg.canvas * cfg.canvas; lam = cfg.trm_lambda
    def make_stp(pp):
        stack = (lambda h: DC._stack(pp, h, cfg)) if dec else (lambda h: TC._stack(pp, h, cfg.trm_puzzle_emb_len))
        return lambda z, inj: (lambda Fz: (z + (1.0 - lam) * (Fz - z)) if lam > 0 else Fz)(stack(z + inj))
    stp = make_stp(p)
    has_fc = dec and all("fc" in b for b in p["blocks"])
    stp_nm = None
    if has_fc:                                                                   # message removal: every block's field coupling set to zero
        p_nm = dict(p); p_nm["blocks"] = [dict(b, fc=jnp.zeros_like(b["fc"])) for b in p["blocks"]]; stp_nm = make_stp(p_nm)
    emb_f = lambda x: C.embed(p, cfg, x)
    scores_f = lambda z: C.readout(p, cfg, z, (9, 9))[0][..., 1:10].reshape(81, 9)
    z0 = C.z0(cfg, hw) if dec else TC.z0(cfg, hw, p=p)
    return dict(jax=jax, jnp=jnp, cfg=cfg, step=step, stp=stp, stp_nm=stp_nm, emb=emb_f, scores=scores_f, zH0=z0[0], zL0=z0[1],
                H=cfg.trm_h_cycles, L=cfg.trm_l_cycles, lam=float(lam), path=path)

def make_cycle_record(jnp, stp, emb_f, scores_f, L):
    """One cycle with records: L fast updates (each followed by a SHADOW slow update and the readout of both the shadow and the fast
    state), then the actual slow update (the segment's own order). Returns the new states and compact records."""
    nrm = lambda a: jnp.sqrt(jnp.sum(a * a))
    def f(x, sol, zH, zL):
        emb = emb_f(x); shm, shd, fam, fad = [], [], [], []
        hs = None
        for _ in range(L):
            zL = stp(zL, zH + emb)
            hs = stp(zH, zL)                                                     # the shadow: the slow update applied now
            m, _r, d = margin_rank(jnp, scores_f(hs), sol); shm.append(m); shd.append(d)
            m2, _r2, d2 = margin_rank(jnp, scores_f(zL), sol); fam.append(m2); fad.append(d2)
        zH2 = stp(zH, zL)                                                        # the actual slow update
        sc = scores_f(zH2); m, r, d = margin_rank(jnp, sc, sol)
        same6 = jnp.max(jnp.abs(scores_f(hs) - sc))
        return zH2, zL, jnp.stack(shm), jnp.stack(shd), jnp.stack(fam), jnp.stack(fad), m, r, d, nrm(zH2 - zH) / jnp.maximum(nrm(zH), 1e-30), same6
    return f

def make_cycle_dist(jnp, stp, emb_f, L):
    nrm = lambda a: jnp.sqrt(jnp.sum(a * a))
    def f(x, zH, zL, zref):
        emb = emb_f(x)
        for _ in range(L): zL = stp(zL, zH + emb)
        zH = stp(zH, zL)
        return zH, zL, nrm(zH - zref) / jnp.maximum(nrm(zref), 1e-30)
    return f

def make_cycle_intervene(jnp, stp, stp_alt, emb_f, scores_f, L):
    """One cycle; per puzzle flags applied at the START of this cycle: fr (fast state -> its buffer), sr (slow state -> its buffer),
    nm (this cycle's updates use stp_alt, e.g. messages removed). Returns the new states and the decoded digits."""
    def f(x, zH, zL, zH0, zL0, fr, sr, nm):
        zL = jnp.where(fr, zL0, zL); zH = jnp.where(sr, zH0, zH); emb = emb_f(x)
        sel = (lambda z, inj: jnp.where(nm, stp_alt(z, inj), stp(z, inj))) if stp_alt is not None else stp
        for _ in range(L): zL = sel(zL, zH + emb)
        zH = sel(zH, zL)
        return zH, zL, scores_f(zH).argmax(-1) + 1
    return f

# ---------------- the run ----------------
def gate_reference(key, ids, puz):
    import lens_corpus_normalized as CN
    path, ema, _, kind = MODELS[key]
    if kind == "validity":
        R_ = np.load(ROOT / f"runs/analysis/commit_validity_20260919/{key}.npz", allow_pickle=True); n = len(ids); assert (R_["ids"][:n] == ids).all()
        return R_["logits"][:, :n].astype(np.float64).argmax(-1) + 1                             # (T, N, 9, 9)
    P, _L, _m = CN.run_iters(path, ema, puz, N_IT, f"{key} evaluator gate"); return P.astype(np.int64)

def run_model(key):
    import lens_corpus_normalized as CN
    M = build(key); jax, jnp = M["jax"], M["jnp"]; H, L = M["H"], M["L"]; J = N_IT * H
    ids, puz, sol = CN.puzzles("strat", 256)
    if SMOKE: ids, puz, sol = ids[:BS], puz[:BS], sol[:BS]                                        # one batch; outputs under OUT/smoke
    N = len(puz); ng = (puz == 0).reshape(N, 81); solf = sol.reshape(N, 81)
    rec = jax.jit(jax.vmap(make_cycle_record(jnp, M["stp"], M["emb"], M["scores"], L)))
    dst = jax.jit(jax.vmap(make_cycle_dist(jnp, M["stp"], M["emb"], L)))
    zH0, zL0 = M["zH0"], M["zL0"]; t_a = time.time()
    A = dict(slow_margin=np.zeros((J, N, 81), np.float16), slow_rank=np.zeros((J, N, 81), np.int8), slow_digit=np.zeros((J, N, 81), np.int8),
             rel=np.zeros((J, N), np.float32), shadow_margin=np.zeros((J, L, N, 81), np.float16), shadow_digit=np.zeros((J, L, N, 81), np.int8),
             fast_margin=np.zeros((J, L, N, 81), np.float16), fast_digit=np.zeros((J, L, N, 81), np.int8), dist=np.zeros((J, N), np.float32))
    same6 = 0.0
    def padded(a, b0, keep):
        a = a[b0:b0 + keep]; return np.concatenate([a, np.repeat(a[-1:], BS - keep, 0)]) if keep < BS else a
    for b0 in range(0, N, BS):
        keep = min(BS, N - b0); x = jnp.asarray(padded(puz, b0, keep)); s = jnp.asarray(padded(solf, b0, keep))
        zH = jnp.broadcast_to(zH0, (BS,) + zH0.shape); zL = jnp.broadcast_to(zL0, (BS,) + zL0.shape)
        for j in range(J):                                                                          # pass A
            zH, zL, shm, shd, fam, fad, m, r, d, rl, s6 = rec(x, s, zH, zL)
            A["shadow_margin"][j, :, b0:b0 + keep] = np.asarray(shm, np.float32).transpose(1, 0, 2)[:, :keep].astype(np.float16)
            A["shadow_digit"][j, :, b0:b0 + keep] = np.asarray(shd).transpose(1, 0, 2)[:, :keep].astype(np.int8)
            A["fast_margin"][j, :, b0:b0 + keep] = np.asarray(fam, np.float32).transpose(1, 0, 2)[:, :keep].astype(np.float16)
            A["fast_digit"][j, :, b0:b0 + keep] = np.asarray(fad).transpose(1, 0, 2)[:, :keep].astype(np.int8)
            A["slow_margin"][j, b0:b0 + keep] = np.asarray(m, np.float32)[:keep].astype(np.float16); A["slow_rank"][j, b0:b0 + keep] = np.asarray(r)[:keep]
            A["slow_digit"][j, b0:b0 + keep] = np.asarray(d)[:keep]; A["rel"][j, b0:b0 + keep] = np.asarray(rl)[:keep]; same6 = max(same6, float(np.asarray(s6)[:keep].max()))
        zref = zH; zH = jnp.broadcast_to(zH0, (BS,) + zH0.shape); zL = jnp.broadcast_to(zL0, (BS,) + zL0.shape)
        for j in range(J):                                                                          # pass B
            zH, zL, dd = dst(x, zH, zL, zref); A["dist"][j, b0:b0 + keep] = np.asarray(dd)[:keep]
        print(f"  {key}: passes A+B rows {b0 + keep}/{N} ({time.time() - t_a:.0f}s)", flush=True)
    exact = (A["slow_digit"].astype(np.int64) == solf[None]).all(-1)                                 # (J, N) all 81 cells
    ref = gate_reference(key, ids, puz); ev_exact = (ref.reshape(N_IT, N, 81) == solf[None]).all(-1)
    outer = A["slow_digit"][H - 1::H].astype(np.int64)                                               # readouts j = 3t
    agree1 = float(((outer[0] == ref.reshape(N_IT, N, 81)[0]) & ng).sum() / ng.sum())
    agree_solved = min(float(((outer[t] == ref.reshape(N_IT, N, 81)[t]) & ng)[ev_exact[t]].sum() / max(ng[ev_exact[t]].sum(), 1)) for t in range(4))
    dist_end = float(A["dist"][-1].max())                                                            # pass B replays pass A: must be 0
    meta = dict(key=key, step=M["step"], H=H, L=L, lam=M["lam"], agree_iter1=agree1, agree_on_solved_through_4=agree_solved, same6=same6,
                dist_end_max=dist_end, exact_end=int(exact[-1].sum()), ref_exact_end=int(ev_exact[-1].sum()), has_messages=M["stp_nm"] is not None)
    if agree1 <= 0.999 or agree_solved <= 0.999 or same6 > 1e-4 or dist_end > 1e-5:
        print(f"FAILED-GATE {key}: {meta}; NOT saved", flush=True); return
    idx, kappa = cohort_of(exact)
    print(f"  {key}: gate passed {meta}; cohort {len(idx)} (first exact at readout >= {COHORT_MIN} and exact at {J})", flush=True)
    C = {}                                                                                              # pass C
    for variant in VARIANTS:
        if variant == "no_messages" and M["stp_nm"] is None: continue
        f = jax.jit(jax.vmap(make_cycle_intervene(jnp, M["stp"], M["stp_nm"] if variant == "no_messages" else None, M["emb"], M["scores"], L),
                             in_axes=(0, 0, 0, None, None, 0, 0, 0)))
        for off in OFFSETS:
            at_k = np.zeros(len(idx), bool); at_end = np.zeros(len(idx), bool); first_after = np.zeros(len(idx), np.int32)
            for c0 in range(0, len(idx), BS):
                sel = idx[c0:c0 + BS]; keep = len(sel); selp = np.concatenate([sel, np.repeat(sel[-1:], BS - keep)]) if keep < BS else sel
                x = jnp.asarray(puz[selp]); target = kappa[selp] + off                              # the cycle at whose START the intervention acts
                zH = jnp.broadcast_to(zH0, (BS,) + zH0.shape); zL = jnp.broadcast_to(zL0, (BS,) + zL0.shape); ex = np.zeros((J, BS), bool)
                for j in range(1, J + 1):
                    hit = jnp.asarray(target == j)
                    fr = hit & (variant == "fast_reset"); sr = hit & (variant == "slow_reset"); nm = hit & (variant == "no_messages")
                    zH, zL, dg = f(x, zH, zL, zH0, zL0, fr, sr, nm); ex[j - 1] = (np.asarray(dg) == solf[selp]).all(-1)
                for n in range(keep):
                    k1 = kappa[sel[n]] - 1; at_k[c0 + n] = ex[k1, n]; at_end[c0 + n] = ex[-1, n]
                    t0 = kappa[sel[n]] + off - 1; later = np.flatnonzero(ex[t0:, n]); first_after[c0 + n] = (t0 + later[0] + 1) if len(later) else 0
            C[f"{variant}@{off}"] = dict(exact_at_kappa=at_k, exact_at_end=at_end, first_exact_after=first_after)
            print(f"  {key}: {variant} at kappa{off:+d}: exact at kappa {at_k.mean():.3f}, at {J} {at_end.mean():.3f} ({time.time() - t_a:.0f}s)", flush=True)
    meta["wall"] = round(time.time() - t_a, 1)
    save = dict(A, exact=exact, kappa=kappa, cohort=idx, puz=puz, sol=sol, ids=ids, meta=json.dumps(meta))
    for k_, v in C.items():
        for f_, arr in v.items(): save[f"int_{k_}_{f_}"] = arr
    OUT.mkdir(parents=True, exist_ok=True); tmp = OUT / f"{key}.tmp.npz"; np.savez_compressed(tmp, **save); os.replace(tmp, OUT / f"{key}.npz")
    print(f"DONE {key}.npz {meta}", flush=True)

# ---------------- the report ----------------
def report():
    Ls, JS = [], {}; say = lambda s_="": (Ls.append(s_), print(s_)); have = [k for k in MODELS if (OUT / f"{k}.npz").exists()]
    say("INSIDE THE COMPLETING CYCLE (tools/lens_inside_completion.py; registration Documentation/Note_2026-09-25_Inside_Completion.md).")
    say(f"256 stratified test puzzles, the fixed start, 16 iterations = 48 slow readouts; one checkpoint per model. Present: {', '.join(have)}.")
    for key in have:
        D = np.load(OUT / f"{key}.npz", allow_pickle=True); meta = json.loads(str(D["meta"])); sol = D["sol"].reshape(-1, 81); ng = (D["puz"].reshape(-1, 81) == 0)
        idx, kappa = D["cohort"], D["kappa"]; dig = D["slow_digit"].astype(np.int64); wrong = (dig != sol[None]) & ng[None]
        err = wrong.sum(-1) / ng.sum(-1)[None]; sh_wrong = (D["shadow_digit"].astype(np.int64) != sol[None, None]) & ng[None, None]
        sh_err = sh_wrong.sum(-1) / ng.sum(-1)[None, None]
        margins = D["slow_margin"].astype(np.float64)
        S1 = closure(margins, wrong, kappa, idx); S2 = rank2_share(D["slow_rank"], wrong, kappa, idx)
        S3 = displacement_ratio(D["rel"].astype(np.float64), kappa, idx); S4, S5 = approach_jump(D["dist"].astype(np.float64), kappa, idx)
        S6, S7 = within_cycle(sh_err, err, kappa, idx)
        s1, s3, s4, s5 = (float(np.nanmedian(v)) for v in (S1, S3, S4, S5))
        l12 = "H" if (s1 >= 0.5 and S2 >= 0.5) else ("C" if (s1 <= 0.2 and S2 < 0.5) else "MIXED")
        l3 = "H" if s3 <= 1.5 else ("C" if s3 >= 2 else "MIXED")
        l45 = "H" if s4 >= 0.3 else ("C" if (s4 < 0.1 and s5 >= 0.5) else "NEITHER")
        l6 = "C" if S6.mean() >= 0.5 else "H"
        JS[key] = dict(meta=meta, cohort=int(len(idx)), S1=s1, S1_n=int(np.isfinite(S1).sum()), S2=S2, S3=s3, S4=s4, S5=s5, S6=float(S6.mean()), S7=float(S7.mean()),
                       letters=dict(S1_S2=l12, S3=l3, S4_S5=l45, S6=l6), interventions={})
        say(); say(f"== {key}: {MODELS[key][2]} | gate: {100 * meta['agree_iter1']:.2f} % of empty cells at iteration 1, {100 * meta['agree_on_solved_through_4']:.2f} % on solved puzzles through 4; shadow-6 max |diff| {meta['same6']:.2e}")
        say(f"   exact at readout 48: {meta['exact_end']} of {len(sol)} (reference {meta['ref_exact_end']}); cohort (first exact at readout >= {COHORT_MIN}, exact at 48): {len(idx)}")
        say(f"   S1 closure of the margin gap over the last outer iteration (median; n {int(np.isfinite(S1).sum())}): {s1:.3f}   S2 correct digit ranked second at kappa-1: {S2:.3f}   -> {l12}")
        say(f"   S3 completing slow update / plateau (median ratio of relative changes): {s3:.2f}   -> {l3}")
        say(f"   S4 approach to the final slow state over kappa-6..kappa-1: {s4:.3f}   S5 jump at kappa: {s5:.3f}   -> {l45}")
        say(f"   S6 built within the completing cycle (shadow error after fast update 1 >= half the readout error at kappa-1): {S6.mean():.3f}   -> {l6}")
        say(f"   S7 an exact shadow already in cycle kappa-1: {S7.mean():.3f}")
        # descriptive curves aligned to kappa
        offs = range(-6, 1); med_m = []
        for o in offs:
            vals = [np.median(margins[kappa[i] - 1 + o, i][wrong[kappa[i] - 2, i]]) for i in idx if wrong[kappa[i] - 2, i].any()]
            med_m.append(float(np.median(vals)))
        say("   median margin of the cells wrong at kappa-1, by readout kappa+o: " + "  ".join(f"{o:+d}:{v:.2f}" for o, v in zip(offs, med_m)))
        say("   median readout error by kappa+o: " + "  ".join(f"{o:+d}:{np.median([err[kappa[i] - 1 + o, i] for i in idx]):.3f}" for o in offs))
        say("   median distance to the final slow state by kappa+o: " + "  ".join(f"{o:+d}:{np.median([D['dist'][kappa[i] - 1 + o, i] for i in idx]):.3f}" for o in offs))
        say("   median relative slow-state change by kappa+o: " + "  ".join(f"{o:+d}:{np.median([D['rel'][kappa[i] - 1 + o, i] for i in idx]):.4f}" for o in offs))
        for cyc, lab in ((-1, "cycle kappa-1"), (0, "cycle kappa (completing)")):
            say(f"   median shadow error after fast update k in {lab}: " + "  ".join(f"k{k + 1}:{np.median([sh_err[kappa[i] - 1 + cyc, k, i] for i in idx]):.3f}" for k in range(sh_err.shape[1])))
        fa_wrong = (D["fast_digit"].astype(np.int64) != sol[None, None]) & ng[None, None]; fa_err = fa_wrong.sum(-1) / ng.sum(-1)[None, None]
        say("   median error of the readout applied to the fast state, cycle kappa: " + "  ".join(f"k{k + 1}:{np.median([fa_err[kappa[i] - 1, k, i] for i in idx]):.3f}" for k in range(fa_err.shape[1])))
        for v in VARIANTS:
            for off in OFFSETS:
                name = f"int_{v}@{off}_exact_at_kappa"
                if name not in D.files: continue
                ak = D[name]; ae = D[f"int_{v}@{off}_exact_at_end"]; JS[key]["interventions"][f"{v}@{off}"] = dict(exact_at_kappa=float(ak.mean()), exact_at_end=float(ae.mean()))
                say(f"   intervention {v:12s} at kappa{off:+d}: still exact at kappa {ak.mean():.3f} (blocked {1 - ak.mean():.3f}); exact at 48 {ae.mean():.3f}")
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "report.txt").write_text("\n".join(Ls) + "\n"); (OUT / "report.json").write_text(json.dumps(JS, indent=1, default=float))

# ---------------- selftest ----------------
def _checks(mr, fe, cl):
    """The assertions the selftest applies to (margin_rank, first_exact, closure); mutants must fail at least one."""
    sc = np.array([[3., 1., 2., 0, 0, 0, 0, 0, 0], [2., 2., 0, 0, 0, 0, 0, 0, 0]]); sol = np.array([3, 2])
    m, r, d = mr(np, sc, sol)
    assert np.allclose(m, [-1., 0.]) and r.tolist() == [2, 1] and d.tolist() == [1, 1]
    ex = np.zeros((6, 3), bool); ex[2:, 0] = True; ex[5, 1] = True
    assert fe(ex).tolist() == [3, 6, 0]
    J, N = 10, 1; mg = np.zeros((J, N, 81)); wr = np.zeros((J, N, 81), bool); wr[:, 0, :2] = True
    mg[4, 0, :2] = -4.0; mg[7, 0, :2] = -1.0; kap = np.array([9])                       # readout kappa-4 is index 4, kappa-1 is index 7
    assert np.allclose(cl(mg, wr, kap, np.array([0])), [0.75])

def selftest():
    os.environ.setdefault("JAX_PLATFORMS", "cpu"); import jax, jax.numpy as jnp
    n = 0
    _checks(margin_rank, first_exact, closure); n += 1
    mutants = [
        (lambda xp, s, y: (lambda m, r, d: (-m, r, d))(*margin_rank(xp, s, y)), first_exact, closure),                     # margin sign flipped
        (lambda xp, s, y: (lambda m, r, d: (m, r + (s == xp.take_along_axis(s, (y - 1)[..., None], -1)).sum(-1) - 1, d))(*margin_rank(xp, s, y)), first_exact, closure),   # ties counted as higher
        (margin_rank, lambda e: np.where(e.any(0), e.argmax(0), 0), closure),                                                  # 0-based kappa
        (margin_rank, first_exact, lambda mg, wr, k, i: closure(mg, wr, k + 1, i)),                                           # offset shifted by one readout
    ]
    killed = 0
    for mu in mutants:
        try: _checks(*mu)
        except AssertionError: killed += 1
    assert killed == len(mutants), f"only {killed}/{len(mutants)} mutants killed"; n += 1
    rng = np.random.default_rng(0); sc = rng.normal(size=(5, 81, 9)).astype(np.float32); y = rng.integers(1, 10, size=(5, 81))
    a = margin_rank(np, sc, y); b = margin_rank(jnp, jnp.asarray(sc), jnp.asarray(y))
    assert all(np.allclose(np.asarray(u), np.asarray(v)) for u, v in zip(a, b)); n += 1                                   # numpy and jax agree
    a_ = 0.5; stp = lambda z, inj: a_ * z + inj; emb = lambda x: x; scores = lambda z: z[:81 * 9].reshape(81, 9)       # a toy linear map
    f = make_cycle_record(jnp, stp, emb, scores, 6); x = jnp.ones(81 * 9); zH = jnp.zeros(81 * 9); zL = jnp.ones(81 * 9); s = jnp.ones(81, jnp.int32)
    zH2, zL2, shm, shd, fam, fad, m, r, d, rl, s6 = f(x, s, zH, zL)
    zl = np.ones(81 * 9)
    for _ in range(6): zl = a_ * zl + (0 + 1.0)
    assert np.allclose(np.asarray(zL2), zl) and np.allclose(np.asarray(zH2), a_ * 0 + zl) and float(s6) == 0.0 and shm.shape == (6, 81); n += 1   # order of updates; shadow at k=6 = actual
    g = make_cycle_intervene(jnp, stp, None, emb, scores, 6)
    zHb, zLb, dg = g(x, zH + 5.0, zL + 5.0, zH, zL, jnp.bool_(False), jnp.bool_(True), jnp.bool_(False))
    zHc, zLc, dgc = g(x, zH, zL + 5.0, zH, zL, jnp.bool_(False), jnp.bool_(False), jnp.bool_(False))
    assert np.allclose(np.asarray(zHb), np.asarray(zHc)) and np.allclose(np.asarray(zLb), np.asarray(zLc)); n += 1          # a slow reset makes the slow state its buffer, nothing else
    ex = np.zeros((10, 3), bool); ex[7:, 0] = True; ex[3:, 1] = True; ex[8, 2] = True
    idx, k = cohort_of(ex); assert idx.tolist() == [0] and k.tolist() == [8, 4, 9]; n += 1                                 # cohort: kappa >= 7 AND exact at the end
    rel = np.ones((12, 1)); rel[9, 0] = 4.0; assert np.allclose(displacement_ratio(rel, np.array([10]), np.array([0])), [4.0]); n += 1
    dist = np.linspace(1, 0.5, 12)[:, None]; dist[9, 0] = 0.1
    ap, jp = approach_jump(dist, np.array([10]), np.array([0])); assert np.allclose(ap, [1 - dist[8, 0] / dist[3, 0]]) and np.allclose(jp, [1 - 0.1 / dist[8, 0]]); n += 1
    she = np.ones((12, 6, 1)); she[8, 2, 0] = 0.0; er = np.full((12, 1), 0.4)
    s6_, s7_ = within_cycle(she, er, np.array([10]), np.array([0])); assert s6_.tolist() == [True] and s7_.tolist() == [True]; n += 1
    assert all((ROOT / v[0]).exists() for v in MODELS.values()) and all((ROOT / f"runs/analysis/commit_validity_20260919/{k}.npz").exists() for k, v in MODELS.items() if v[3] == "validity"); n += 1
    print(f"selftest OK: {n}/{n} (mutants killed {killed}/{len(mutants)})")

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
