#!/usr/bin/env python3
# Ledger: DISCUSSION-PERIOD EXPERIMENT P13 (registration Documentation/Note_2026-10-03_Rebuttal_P13_Registration.md, written before any row).
# MEASUREMENT, $0, the Mac's CPU, inference only. P8's completion edit on a matched pair of state structures: SA256U (ONE recurrent state, round 2's
# arm at its selected 14k) and SA256 (two states, the ladder's reference at its selected 28k), plus SA256 at 14k (exploratory: the same training
# stage). On each receiver's own cohort (puzzles first exact at iteration kappa in 7..16 on this CPU run), the carry entering iteration kappa is
# edited once: the readout-orthogonal part of the readout-bearing state (the one carry; the slow state) is replaced by a norm-matched Gaussian
# orthogonal to the readout (P8's perp_random), or rotated by 1 degree (rebuttal_p10.rotate_perp, imported unchanged), every digit score and
# every vector norm kept. The loop is the evaluator's own step (eval_sudoku_extreme._step), mirrored line for line from run_batch with an edit
# hook, and gated bitwise against run_batch itself. MAIN-repo code (src/qhrrn2): the release code has no single-state branch.
"""  JAX_PLATFORMS=cpu .venv/bin/python tools/rebuttal_p13.py --selftest
  JAX_PLATFORMS=cpu .venv/bin/python tools/rebuttal_p13.py --smoke          (gates only; prints no outcome)
  JAX_PLATFORMS=cpu nice -n 10 .venv/bin/python tools/rebuttal_p13.py --run SA256U14 SA256_28 SA256_14     (resume-safe per receiver)
  .venv/bin/python tools/rebuttal_p13.py --report"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tools"))
OUT = ROOT / "runs/analysis/p13_20261003"
REC = ROOT / "runs/_attr2_pull/stage/runs"
NPZ = ROOT / "data/sudoku_extreme/sudoku_extreme_seed0_mon512.npz"
SEED = 20261003
POOL, COHORT_CAP, COHORT_MIN, T_RUN, KMIN, KMAX, GATE_N, BS = 256, 96, 40, 20, 7, 16, 32, 64
SCORE_TOL, NORM_TOL = 1e-3, 1e-5
B_NEEDS, B_SUFF, S_CONTENT, S_SENS = 0.5, 0.1, 0.1, 0.5
FULL = ("sham", "perp_random", "perp_random_m3", "rot1")
RECEIVERS = {   # key: (checkpoint, sha256 prefix, the round-2 records its candidates come from, label, conditions)
    "SA256U14": ("runs/_p13_ckpts/SA256U_ckpt_014000.pkl", "486f5523", "sxeval_pchampSA256U", "one state (SA256U), selected 14k", FULL),
    "SA256_28": ("runs/_p13_ckpts/SA256_ckpt_028000.pkl", "18e9427f", "sxeval_pchampSA256", "two states (SA256), selected 28k", FULL + ("hidden_all",)),
    "SA256_14": ("runs/_p13_ckpts/SA256_ckpt_014000.pkl", "b98c802e", "sxeval_pchampSA256", "two states (SA256), 14k (exploratory: SA256U's stage)", ("sham", "perp_random", "rot1")),
}
COND = {"sham": (0, "sham"), "perp_random": (0, "perp"), "perp_random_m3": (-3, "perp"), "rot1": (0, "rot1"), "hidden_all": (0, "hidden")}
KCODE = {"sham": 0, "perp": 1, "rot1": 2, "hidden": 3}

def utc(): return datetime.now(timezone.utc).isoformat()

# ---------------- pure helpers (selftested) ----------------
def perp_random(h, v, rng):
    """P8's edit. h (..., w), v (w,): the part of h orthogonal to v replaced by a Gaussian draw projected orthogonal to v and rescaled to the
    old part's norm, per vector. Keeps every score h.v and every norm |h|. float64."""
    h = np.asarray(h, np.float64); e = np.asarray(v, np.float64); e = e / np.linalg.norm(e)
    par = (h @ e)[..., None] * e; pn = np.linalg.norm(h - par, axis=-1, keepdims=True)
    g = rng.standard_normal(h.shape); g = g - (g @ e)[..., None] * e
    return par + g / np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-300) * pn

def norm_matched_random(h, rng):
    """Each vector replaced by a Gaussian draw rescaled to its norm (the fast state in hidden_all). float64."""
    h = np.asarray(h, np.float64); g = rng.standard_normal(h.shape)
    return g / np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-300) * np.linalg.norm(h, axis=-1, keepdims=True)

def edit_carry(zc, v, kind, rng):
    """zc (2, F, S, w) float32 carry of one puzzle (slot 0 = the readout-bearing state) -> the edited float32 carry."""
    import rebuttal_p10 as P10                                     # rotate_perp, unchanged (main-repo paths only)
    z = np.asarray(zc, np.float64).copy(); h = z[0]
    if kind == "sham":     z[0] = P10.rotate_perp(h, v, 0.0, rng.standard_normal(h.shape))   # the reassembly, theta 0
    elif kind == "perp":   z[0] = perp_random(h, v, rng)
    elif kind == "rot1":   z[0] = P10.rotate_perp(h, v, 1.0, rng.standard_normal(h.shape))
    elif kind == "hidden": z[0] = perp_random(h, v, rng); z[1] = norm_matched_random(z[1], rng)
    else: raise ValueError(kind)
    return z.astype(np.float32)

def edit_checks(old, new, v, kind):
    """-> (max |score change| over slot 0 in float64, max relative norm change over the edited slots, slot-0 bitwise equal)."""
    o, n = np.asarray(old, np.float64), np.asarray(new, np.float64); v = np.asarray(v, np.float64)
    ds = float(np.abs(n[0] @ v - o[0] @ v).max())
    slots = (0, 1) if kind == "hidden" else (0,)
    dn = max(float(np.abs(np.linalg.norm(n[s], axis=-1) / np.maximum(np.linalg.norm(o[s], axis=-1), 1e-30) - 1).max()) for s in slots)
    return ds, dn, bool(np.array_equal(np.asarray(old), np.asarray(new)))

def kappa_of(ex):
    """ex (T, N) exact flags at iterations 1..T -> (N,) 1-based first exact iteration, 0 if never."""
    return np.where(ex.any(0), ex.argmax(0) + 1, 0)

def cohort_rule(ex, kmin=KMIN, kmax=KMAX):
    """Cohort: first exact at kappa in [kmin, kmax] and exact at the last iteration -> (mask (N,), kappa (N,))."""
    k = kappa_of(ex); return (k >= kmin) & (k <= kmax) & ex[-1], k

def letters(p_perp, p_rot):
    """P(exact at kappa | perp_random@kappa), P(exact at kappa | rot1@kappa) -> (B, R13a, S1, R13b). P(exact at kappa | intact) = 1 by construction."""
    B = 1.0 - p_perp; a = "NEEDS-HIDDEN-STATE" if B >= B_NEEDS else ("READOUT-SUFFICES" if B <= B_SUFF else "MIXED")
    S = None if p_rot is None else 1.0 - p_rot
    if B < B_NEEDS or S is None: b = "n/a (B < 0.5)" if B < B_NEEDS else "NO-DATA"
    else: b = "CONTENT" if S <= S_CONTENT else ("SENSITIVE" if S >= S_SENS else "INTERMEDIATE")
    return B, a, S, b

# ---------------- the receiver and the loop ----------------
class Receiver:
    def __init__(self, key):
        import jax, jax.numpy as jnp
        from qhrrn2 import episodic as E, model as M, sudoku_extreme as SX
        from qhrrn2.config import Config
        path, sha, recdir, self.label, self.conds = RECEIVERS[key]; self.key = key
        h = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        assert h.startswith(sha), f"{key}: checkpoint sha256 {h[:16]} != the registered {sha}"
        saved = E.load_ckpt(str(ROOT / path)); d0 = Config()
        self.cfg = Config(**{k: type(getattr(d0, k))(v) for k, v in saved["config"].items()})   # the evaluator's own rebuild
        st = saved["state_ema"]; self.params = st["model"]; self.tvj = jnp.asarray(st["table"][0])
        self.eta, self.eta_z = (float(x) for x in M.eq_etas(self.params, self.cfg))
        self.lm = np.asarray(self.params["dec"]["lm_head"], np.float64)
        d = SX.load_prepared(str(NPZ)); self.Q, self.A = d["test_q"], d["test_a"]
        r = dict(np.load(REC / recdir / "sub5k_vsel_t64" / "records_all.npz", allow_pickle=True))
        self.rec_idx, self.rec_fe, self.rec_cold = np.asarray(r["idx"]), np.asarray(r["first_exact"]), np.asarray(r["cold_exact"]).astype(bool)
        self.step_tag = int(saved.get("step", -1)); self.jax, self.jnp = jax, jnp

    def candidates(self):
        """Record candidates: first exact (0-based) in [KMIN-1, KMAX-1] and exact at 64 on the accelerator; a seeded draw of <= POOL."""
        c = self.rec_idx[(self.rec_fe >= KMIN - 1) & (self.rec_fe <= KMAX - 1) & self.rec_cold]
        rng = np.random.default_rng([SEED, 13, list(RECEIVERS).index(self.key)])
        return np.sort(rng.permutation(np.sort(c))[:POOL])

    def run(self, ids, T, edit_iter=None, kind=None, diag=None):
        """The evaluator's run_batch loop (fixed start, layout native9), mirrored, with ONE optional edit per puzzle on the carry entering
        iteration edit_iter[n] (1-based). -> ex (T, N) exact flags."""
        import eval_sudoku_extreme as EV
        from qhrrn2 import grid as G, model as M
        jax, jnp = self.jax, self.jnp; out = []
        for s in range(0, len(ids), BS):
            bid = np.asarray(ids[s:s + BS]); B = len(bid)
            puz9 = self.Q[bid].astype(np.int32); sol9 = jnp.asarray(self.A[bid].astype(np.int32), jnp.int32)
            x_can = EV.place_batch(puz9, "native9")
            void = jax.nn.one_hot(jnp.full((9, 9), G.VOID, jnp.int32), M.VOCAB).transpose(2, 0, 1)
            y = jnp.broadcast_to(void, (B,) + void.shape); z_c = None; rows = []
            for t in range(T):
                if edit_iter is not None and z_c is not None:
                    sel = np.flatnonzero(np.asarray(edit_iter[s:s + B]) == t + 1)
                    if len(sel):
                        zn = np.array(z_c)
                        for b in sel:
                            rng = np.random.default_rng([SEED, int(bid[b]), 81, t + 1, KCODE[kind]])
                            new = edit_carry(zn[b], self.lm, kind, rng)
                            if diag is not None: diag.append(edit_checks(zn[b], new, self.lm, kind))
                            zn[b] = new
                        z_c = jnp.asarray(zn)
                first = z_c is None
                logits, zf = EV._step(self.cfg, 1.0, 0.0, first)(self.params, x_can, y, self.tvj, jnp.zeros(1) if first else z_c)
                z_c = zf if first else z_c + self.eta_z * (zf - z_c)
                p = jax.nn.softmax(logits, axis=-1).transpose(0, 3, 1, 2)
                y = y + self.eta * (p - y)
                pred9 = EV.layout_gather(jnp.argmax(logits, axis=-1), "native9").astype(jnp.int32)
                pred9 = jnp.where(pred9 == G.VOID, 0, pred9)
                rows.append(np.asarray(jnp.all((pred9 == sol9).reshape(B, -1), axis=1)))
            out.append(np.stack(rows))
        return np.concatenate(out, axis=1)

    def run_batch_ref(self, ids, T):
        """The evaluator's own run_batch on the same puzzles (gate G1)."""
        import eval_sudoku_extreme as EV
        from qhrrn2 import grid as G, model as M
        jax, jnp = self.jax, self.jnp; out = []
        for s in range(0, len(ids), BS):
            bid = np.asarray(ids[s:s + BS]); B = len(bid); puz9 = self.Q[bid].astype(np.int32); sol9 = self.A[bid].astype(np.int32)
            void = jax.nn.one_hot(jnp.full((9, 9), G.VOID, jnp.int32), M.VOCAB).transpose(2, 0, 1)
            ex, _, _, _ = EV.run_batch(self.params, self.cfg, self.tvj, EV.place_batch(puz9, "native9"), jnp.broadcast_to(void, (B,) + void.shape),
                                       t_total=T, tau=1.0, gamma=1.0, sol9=sol9, puz9=puz9, eta=self.eta, eta_z=self.eta_z, layout="native9")
            out.append(np.asarray(ex))
        return np.concatenate(out, axis=1)

def run_receiver(key, smoke=False):
    OUT.mkdir(parents=True, exist_ok=True); f = OUT / f"{key}{'_smoke' if smoke else ''}.npz"
    if f.exists() and not smoke: print(f"{key}: done ({f.name})"); return
    t0 = time.time(); R = Receiver(key); pool = R.candidates()
    if smoke: pool, T, gn = pool[:8], 6, 8
    else: T, gn = T_RUN, GATE_N
    print(f"{utc()} {key}: {R.label}; pool {len(pool)}; T {T}", flush=True)
    ex_pool = R.run(pool, T); ex_pool2 = R.run(pool[:gn], T); ex_ref = R.run_batch_ref(pool[:gn], T)
    g1 = bool(np.array_equal(ex_pool[:, :gn], ex_ref)); g2 = bool(np.array_equal(ex_pool[:, :gn], ex_pool2))
    res = dict(key=key, label=R.label, ckpt_step=R.step_tag, pool=pool, ex_pool=ex_pool, g1_run_batch_bitwise=g1, g2_determinism=g2, T=T)
    if smoke:   # gates only: one edit of each kind at iteration 4 on the 8 puzzles; nothing about outcomes is printed
        for c in R.conds:
            diag = []; R.run(pool, T, edit_iter=np.full(len(pool), 4), kind=COND[c][1], diag=diag)
            ds = max(d[0] for d in diag); dn = max(d[1] for d in diag); bw = all(d[2] for d in diag)
            print(f"  SMOKE {key} {c}: edits {len(diag)}, max |score change| {ds:.2e}, max rel norm change {dn:.2e}, slot-0 bitwise unchanged {bw}", flush=True)
        print(f"  SMOKE {key}: G1 run_batch bitwise {g1}, G2 determinism {g2}; {time.time() - t0:.0f} s", flush=True); return
    mask, kap = cohort_rule(ex_pool)
    order = np.random.default_rng([SEED, 13, 7, list(RECEIVERS).index(key)]).permutation(np.flatnonzero(mask))[:COHORT_CAP]
    coh = np.sort(order); res.update(cohort=pool[coh], kappa=kap[coh], n_eligible=int(mask.sum()),
                                     rec_fe_pool=np.array([R.rec_fe[np.flatnonzero(R.rec_idx == i)[0]] for i in pool]))
    print(f"{utc()} {key}: eligible {int(mask.sum())}, cohort {len(coh)}; G1 {g1}, G2 {g2}", flush=True)
    for c in R.conds:
        off, kind = COND[c]; diag = []
        ex = R.run(pool[coh], T, edit_iter=kap[coh] + off, kind=kind, diag=diag)
        res[f"ex_{c}"] = ex; res[f"diag_{c}"] = np.array(diag, dtype=np.float64) if diag else np.zeros((0, 3))
        print(f"{utc()} {key}: {c} done ({len(diag)} edits)", flush=True)
    res["wall_s"] = time.time() - t0
    np.savez(f, **{k: (np.asarray(v) if not isinstance(v, (str, bool, int, float)) else np.asarray(v)) for k, v in res.items()})
    np.savez(OUT / f"intact_flags_{key}.npz", pool=pool, ex_pool=ex_pool, kappa=kappa_of(ex_pool))
    print(f"{utc()} {key}: saved {f.name} ({res['wall_s']:.0f} s)", flush=True)

def report():
    L, J = [], {}
    say = lambda s: (L.append(s), print(s))
    say("P13 — does a single recurrent state need its readout-invisible part to complete? (P8's completion edit; registered rules)")
    for key in RECEIVERS:
        f = OUT / f"{key}.npz"
        if not f.exists(): J[key] = "NO-DATA"; say(f"{key}: NO-DATA"); continue
        z = dict(np.load(f, allow_pickle=True)); kap = z["kappa"]; n = len(kap); T = int(z["T"])
        at = lambda c: z[f"ex_{c}"][kap - 1, np.arange(n)] if f"ex_{c}" in z else None
        sham = at("sham"); perp = at("perp_random"); rot = at("rot1")
        gates = {"G1 run_batch bitwise": bool(z["g1_run_batch_bitwise"]), "G2 determinism": bool(z["g2_determinism"])}
        for c in RECEIVERS[key][4]:
            d = z[f"diag_{c}"]
            if c == "sham": gates["G5 sham bitwise (carry)"] = bool(d[:, 2].all()) if len(d) else False
            elif len(d): gates[f"G3 {c} score <= {SCORE_TOL}"] = bool(d[:, 0].max() <= SCORE_TOL); gates[f"G4 {c} norm <= {NORM_TOL}"] = bool(d[:, 1].max() <= NORM_TOL)
        gates["G5 sham flags = intact"] = bool(sham is not None and sham.all() and np.array_equal(z["ex_sham"], z["ex_pool"][:, np.searchsorted(z["pool"], z["cohort"])]))
        ok = all(gates.values()) and n >= COHORT_MIN
        say(f"{key} [{z['label']}; ckpt step {int(z['ckpt_step'])}]: eligible {int(z['n_eligible'])}, cohort {n}; gates " + ", ".join(f"{k} {'PASS' if v else 'FAIL'}" for k, v in gates.items()))
        if not ok:
            J[key] = "UNDEFINED (" + ("cohort < 40" if n < COHORT_MIN else "a gate failed") + ")"; say(f"  {J[key]}"); continue
        B, a, S, b = letters(float(perp.mean()), None if rot is None else float(rot.mean()))
        J[key] = dict(B=B, R13a=a, S1=S, R13b=b, n=n)
        say(f"  R13a: B = 1 - P(exact at kappa | perp_random@kappa) = {B:.3f} ({int((~perp).sum())}/{n} blocked) -> {a}")
        if S is not None: say(f"  R13b: S1 = 1 - P(exact at kappa | rot1@kappa) = {S:.3f} ({int((~rot).sum())}/{n}) -> {b}")
        for c in RECEIVERS[key][4]:                                    # descriptive: exact at kappa, the delay, exact at T
            if c == "sham": continue
            off = COND[c][0]; ex = z[f"ex_{c}"]; fe = np.array([np.flatnonzero(ex[kap[i] - 1:, i])[0] + kap[i] if ex[kap[i] - 1:, i].any() else 0 for i in range(n)])
            d = fe[fe > 0] - kap[fe > 0]
            say(f"  (descriptive) {c}@{'kappa' if off == 0 else 'kappa' + str(off)}: exact at kappa {ex[kap - 1, np.arange(n)].mean():.3f}, exact at {T} {ex[-1].mean():.3f}, "
                f"median delay {np.median(d) if len(d) else float('nan'):.1f} (n {len(d)})")
        agree = float(np.mean((kappa_of(z["ex_pool"]) - 1) == z["rec_fe_pool"]))
        say(f"  (descriptive) the CPU first-exact iteration equals the accelerator record's on {100*agree:.1f} % of the pool ({len(z['pool'])})")
    (OUT / "report.txt").write_text("\n".join(L) + "\n"); (OUT / "report.json").write_text(json.dumps(J, indent=1, default=float))

def selftest():
    n = 0; rng = np.random.default_rng(0); v = rng.standard_normal(32); h = rng.standard_normal((9, 81, 32))
    p = perp_random(h, v, np.random.default_rng(1))
    assert np.allclose(p @ v, h @ v, atol=1e-10) and np.allclose(np.linalg.norm(p, axis=-1), np.linalg.norm(h, axis=-1), rtol=1e-12); n += 1
    e = v / np.linalg.norm(v); op = h - (h @ e)[..., None] * e; npp = p - (p @ e)[..., None] * e
    cs = np.sum(op * npp, -1) / (np.linalg.norm(op, axis=-1) * np.linalg.norm(npp, axis=-1)); assert abs(cs.mean()) < 0.05; n += 1   # a fresh direction
    z = rng.standard_normal((2, 9, 81, 32)).astype(np.float32)
    s = edit_carry(z, v, "sham", np.random.default_rng(2)); ds, dn, bw = edit_checks(z, s, v, "sham"); assert ds < 1e-4 and dn < 1e-6; n += 1
    r = edit_carry(z, v, "rot1", np.random.default_rng(3)); o = z[0].astype(np.float64); q = r[0].astype(np.float64)
    o_p, q_p = o - (o @ e)[..., None] * e, q - (q @ e)[..., None] * e
    ang = np.degrees(np.arccos(np.clip(np.sum(o_p * q_p, -1) / (np.linalg.norm(o_p, axis=-1) * np.linalg.norm(q_p, axis=-1)), -1, 1)))
    assert abs(np.median(ang) - 1.0) < 0.05 and edit_checks(z, r, v, "rot1")[0] < 1e-3; n += 1
    hd = edit_carry(z, v, "hidden", np.random.default_rng(4)); ds, dn, _ = edit_checks(z, hd, v, "hidden")
    assert ds < 1e-3 and dn < 1e-5 and not np.allclose(hd[1], z[1]) and np.allclose(edit_carry(z, v, "perp", np.random.default_rng(4))[1], z[1]); n += 1
    ex = np.zeros((20, 5), bool); ex[6:, 0] = True; ex[15:, 1] = True; ex[3:, 2] = True; ex[16:, 3] = True; ex[8:15, 4] = True
    m, k = cohort_rule(ex); assert list(k) == [7, 16, 4, 17, 9] and list(m) == [True, True, False, False, False]; n += 1
    assert letters(0.1, 0.95)[1] == "NEEDS-HIDDEN-STATE" and letters(0.1, 0.95)[3] == "CONTENT" and abs(letters(0.1, 0.95)[2] - 0.05) < 1e-12; n += 1
    assert letters(0.1, 0.2)[3] == "SENSITIVE" and letters(0.1, 0.7)[3] == "INTERMEDIATE" and letters(0.95, 0.99)[1] == "READOUT-SUFFICES" and letters(0.7, 0.9)[1] == "MIXED"; n += 1
    assert letters(0.7, 0.9)[3].startswith("n/a") and letters(0.2, None)[3] == "NO-DATA"; n += 1
    print(f"selftest OK: {n}/{n}")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true"); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--run", nargs="*"); ap.add_argument("--report", action="store_true"); a = ap.parse_args()
    if a.selftest: return selftest()
    if a.smoke:
        for k in RECEIVERS: run_receiver(k, smoke=True)
        return
    if a.run is not None:
        for k in (a.run or list(RECEIVERS)): run_receiver(k)
    if a.report: report()

if __name__ == "__main__":
    main()
