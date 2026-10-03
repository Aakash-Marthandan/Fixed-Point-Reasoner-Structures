#!/usr/bin/env python3
# Ledger: THE DISCUSSION-PERIOD EXPERIMENT P22 — is stalled correction memoryless? Repeated answer-preserving re-rolls of the same
# stalled state (registration Documentation/Note_2026-10-03_Rebuttal_P22_Registration.md, written before any row and before P21's
# results). MEASUREMENT, $0, inference only on the banked attention checkpoints; imports P17's runner (tools/rebuttal_p17.py: loop,
# rotation, trigger) and the P17 / P20 pools unchanged.
"""P22: per receiver, a fresh draw of test puzzles unsolved at 16 in the saved full-test record (excluding P12's, P17's and P20's
pools); the trigger is an invalid displayed grid at iteration 16 on the CPU; from that state: A0 intact (32 iterations), eight
independent 60-degree score-preserving re-rolls (32 iterations each) and four independent 1-degree re-rolls (16 iterations each).
Rules: A (memory) from R = completions in iterations 17-32 after the kick among re-rolls that survived 1-16, divided by the
count memorylessness predicts from each puzzle's own first-window rate (an unbiased plug-in): AGING / ACCUMULATING / MEMORYLESS /
UNRESOLVED; B (chaos) from the ratio of within-puzzle outcome disagreement at +16 for
1-degree vs 60-degree re-rolls: CHAOTIC / INTERMEDIATE / REGULAR."""
import argparse, json, math, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import rebuttal_p17 as R17
import rebuttal_p20 as R20
import rebuttal_p15 as Q

OUT = ROOT / "runs/analysis/rebuttal_20261003e"
SEED = 20261004
N_DRAW = 160
STEPS1, STEPS_LONG, STEPS_SHORT = 16, 32, 16          # trigger at 16; A0 and 60-degree re-rolls to 48; 1-degree re-rolls to 32
K_BIG, K_SMALL, BIG, SMALL, TINY = 8, 4, 60.0, 1.0, 0.1               # TINY: an exploratory 0.1-degree probe (K_SMALL re-rolls, 16 iterations)
E_WIN, L_WIN = (1, 16), (17, 32)                      # iterations after the kick
N_BOOT = 2000
HR_LO, HR_HI = 0.67, 1.5                              # equivalence margins for MEMORYLESS
CHAOS_HI, CHAOS_LO, MIN_RECEIVERS = 0.8, 0.5, 2


def utc(): return datetime.now(timezone.utc).isoformat()


def pool(width):
    """A fresh seeded draw from the record's unsolved-at-16 pool, excluding P12's pool and this width's P17 and P20 pools."""
    import rebuttal_p12 as P12
    rec = np.load(ROOT / f"paper/code/evidence/benchmark/attention_{width}_d16.npz", allow_pickle=False)
    r128 = np.load(ROOT / "paper/code/evidence/benchmark/attention_128_d16.npz", allow_pickle=False)
    excl = set(int(i) for i in P12.pool_ids(r128["cold_exact"], r128["idx"])) | set(int(i) for i in R17.pool(width)) | set(int(i) for i in R20.pool(width))
    cand = np.array([i for i in rec["idx"][~rec["cold_exact"]] if int(i) not in excl])
    return np.sort(np.random.default_rng([SEED, 22, width]).choice(cand, size=N_DRAW, replace=False))


# ---------------- pure helpers (selftested) ----------------
def first_hits(X):
    """X: (steps, n) bool exact flags (row t = iteration t+1 after the kick) -> (n,) first hit in 1..steps, or steps+1 if never."""
    return np.where(X.any(0), X.argmax(0) + 1, X.shape[0] + 1)


def window_counts(first, lo, hi):
    """Events and at-risk iterations inside [lo, hi] for trajectories with first-hit times `first` (any shape)."""
    ev = ((first >= lo) & (first <= hi)).sum()
    risk = np.clip(np.minimum(first, hi) - lo + 1, 0, None).sum()          # at risk from lo through min(first, hi)
    return int(ev), int(risk)


def memory_ratio(firsts):
    """firsts: list over puzzles of (K,) first-hit arrays (equal windows E = 1-16, L = 17-32). R = sum of completions in L among
    E-survivors / sum of (K - dE) dE / (K - 1). Under memorylessness, E[dL | dE] = (K - dE) p and E[(K - dE) dE / (K - 1)] = K p (1 - p)
    = E[dL] for dE ~ Binomial(K, p), so numerator and denominator estimate the same quantity; R < 1: survivors complete less often
    than fresh re-rolls (aging); R > 1: more often (accumulating)."""
    O = Ex = 0.0
    for f in firsts:
        K = len(f); dE = int(((f >= E_WIN[0]) & (f <= E_WIN[1])).sum()); dL = int(((f >= L_WIN[0]) & (f <= L_WIN[1])).sum())
        O += dL; Ex += (K - dE) * dE / (K - 1)
    return O / Ex if Ex > 0 else float("nan")


def boot_ci(firsts, seed):
    rng = np.random.default_rng(seed); n = len(firsts); vals = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n); v = memory_ratio([firsts[i] for i in idx])
        if np.isfinite(v): vals.append(v)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"), float("nan"))


def disagreement(O):
    """O: (K, n) bool outcomes of K re-rolls per puzzle -> mean over puzzles of the within-puzzle pairwise disagreement rate."""
    K = O.shape[0]; pairs = [(a, b) for a in range(K) for b in range(a + 1, K)]
    return float(np.mean([np.mean([O[a, i] != O[b, i] for a, b in pairs]) for i in range(O.shape[1])]))


def letter_a(hr, lo, hi):
    if hi < 1: return "AGING"
    if lo > 1: return "ACCUMULATING"
    if lo >= HR_LO and hi <= HR_HI: return "MEMORYLESS"
    return "UNRESOLVED"


def letter_b(ratio):
    return "CHAOTIC" if ratio >= CHAOS_HI else ("REGULAR" if ratio <= CHAOS_LO else "INTERMEDIATE")


class R14(R17.R11):
    def __init__(self, width, out=OUT):
        super().__init__(width, out)
        self.meta.update(study="rebuttal_20261003e", pipeline="tools/rebuttal_p22.py over tools/rebuttal_p17.py over tools/rebuttal_p15.py")
        self.ATS.write_json(self.dir / "runtime.json", self.meta)

    def rerolls(self, z16, x, sol81, ids, T, angle, K, salt, steps, tag):
        """K independent score-preserving rotations of each triggered state; returns exact flags (K, steps, nT) and gate statistics."""
        pairs = [(k, j) for k in range(K) for j in range(len(T))]; out = np.zeros((K, steps, len(T)), bool); shift = 0.0; ang_dev = 0.0
        for c0 in range(0, len(pairs), Q.VBATCH):
            chunk = pairs[c0:c0 + Q.VBATCH]; pad = Q.VBATCH - len(chunk)
            zz = np.stack([z16[T[j]] for _, j in chunk]).copy(); xb = np.stack([x[T[j]] for _, j in chunk])
            for q, (k, j) in enumerate(chunk):
                h = zz[q, 0]; h2 = self.rotate(h, angle, np.random.default_rng([SEED, int(ids[T[j]]), salt, k]))
                s_old = np.einsum("fsw,w->sf", h.astype(np.float64), self.lm); s_new = np.einsum("fsw,w->sf", h2.astype(np.float64), self.lm)
                shift = max(shift, float(np.abs(s_new - s_old).max()))
                po = h.astype(np.float64) - (h.astype(np.float64) @ self.lm / (self.lm @ self.lm))[..., None] * self.lm
                pn = h2.astype(np.float64) - (h2.astype(np.float64) @ self.lm / (self.lm @ self.lm))[..., None] * self.lm
                c = (po * pn).sum(-1) / np.maximum(np.linalg.norm(po, axis=-1) * np.linalg.norm(pn, axis=-1), 1e-30)
                realized = np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))                                    # the realized angle per (field, cell) vector
                ang_dev = max(ang_dev, float(np.abs(realized - angle).max() / angle))
                zz[q, 0] = h2
            zp = np.concatenate([zz, np.repeat(zz[:1], pad, 0)]) if pad else zz; xp = np.concatenate([xb, np.repeat(xb[:1], pad, 0)]) if pad else xb
            d, _ = self.loop(zp, xp, steps)
            for q, (k, j) in enumerate(chunk): out[k, :, j] = (d[:, q] == sol81[T[j]][None]).all(-1)
            self.status("running", arm=tag, done=c0 + len(chunk), of=len(pairs))
        return out, shift, ang_dev

    def run(self, smoke=False):
        try:
            ids = pool(self.width)
            if smoke: ids = ids[:16]
            with np.load(R17.DATA, allow_pickle=False) as d: puz, sol = d["test_q"][ids].astype(np.int32), d["test_a"][ids].astype(np.int32)
            x = np.asarray(self.EV.place_batch(puz, self.layout)); sol81 = sol.reshape(-1, 81); puz81 = puz.reshape(-1, 81)
            disp1, z16 = [], []
            for b0 in range(0, len(ids), Q.VBATCH):
                xb = x[b0:b0 + Q.VBATCH]; pad = Q.VBATCH - len(xb); xp = np.concatenate([xb, np.repeat(xb[:1], pad, 0)]) if pad else xb
                d, z = self.loop(None, xp, STEPS1, first=True); disp1.append(d[:, :len(xb)]); z16.append(z[:len(xb)])
            disp1 = np.concatenate(disp1, 1); z16 = np.concatenate(z16, 0)
            trig = R17.invalid(disp1[-1], puz81); exact16 = (disp1[-1] == sol81).all(1)
            assert not (trig & exact16).any() and np.all(trig | exact16), "trigger is not equivalent to CPU-unsolved"
            T = np.where(trig)[0]; ex0 = []
            for c0 in range(0, len(T), Q.VBATCH):
                tb = T[c0:c0 + Q.VBATCH]; pad = Q.VBATCH - len(tb); zz = z16[tb].copy(); xb = x[tb]
                zp = np.concatenate([zz, np.repeat(zz[:1], pad, 0)]) if pad else zz; xp = np.concatenate([xb, np.repeat(xb[:1], pad, 0)]) if pad else xb
                d, _ = self.loop(zp, xp, STEPS_LONG); ex0.append((d[:, :len(tb)] == sol81[tb][None]).all(-1)); self.status("running", arm="A0_intact", done=c0 + len(tb), of=len(T))
            exA0 = np.concatenate(ex0, 1) if ex0 else np.zeros((STEPS_LONG, 0), bool); self.ATS.log("arm_complete", width=self.width, arm="A0_intact", of=len(T))
            exBig, sB, cB = self.rerolls(z16, x, sol81, ids, T, BIG, K_BIG, 101, STEPS_LONG, "R60_rerolls"); self.ATS.log("arm_complete", width=self.width, arm="R60_rerolls", of=len(T))
            exSmall, sS, cS = self.rerolls(z16, x, sol81, ids, T, SMALL, K_SMALL, 102, STEPS_SHORT, "R1_rerolls"); self.ATS.log("arm_complete", width=self.width, arm="R1_rerolls", of=len(T))
            exTiny, sT, cT = self.rerolls(z16, x, sol81, ids, T, TINY, K_SMALL, 103, STEPS_SHORT, "R01_rerolls"); self.ATS.log("arm_complete", width=self.width, arm="R01_rerolls", of=len(T))
            assert max(sB, sS, sT) < 1e-2, f"gate: a rotation changed a score by {max(sB, sS, sT)}"
            assert max(cB, cS, cT) < 0.01, f"gate: a realized rotation angle differs from its nominal angle by {max(cB, cS, cT):.2%}"
            dst = self.dir / ("p22_smoke.npz" if smoke else "p22.npz")
            np.savez_compressed(dst, ids=ids, triggered=trig, exact16=exact16, T=T, exact_A0=exA0, exact_R60=exBig, exact_R1=exSmall, exact_R01=exTiny,
                                shift_max=max(sB, sS, sT), angle_dev_max=max(cB, cS, cT),
                                meta=json.dumps(dict(self.meta, created=utc(), n=len(ids), triggered=int(trig.sum()), smoke=smoke)))
            if not smoke: self.ATS.write_json(dst.with_suffix(".sha256.json"), dict(sha256=self.ATS.sha(dst)))
            print(json.dumps(dict(width=self.width, n=len(ids), triggered=int(trig.sum()), shift_max=max(sB, sS, sT), angle_dev_max=max(cB, cS, cT))))
            self.ATS.verify_sources(); self.status("inference_complete", completed=utc()); self.ATS.log("model_complete", width=self.width, group="p22")
        except BaseException as exc:
            self.status("failed", error=repr(exc)); raise


def report(base=OUT, widths=Q.WIDTHS):
    lines = ["P22 REPORT (rules registered 2026-10-03, before any P22 row and before P21's results): is stalled correction memoryless? re-rolls of the same stalled state"]
    out = {}; la, lb = [], []
    for w in widths:
        f = base / f"attention_{w}/p22.npz"
        if not f.exists(): lines.append(f"attention_{w}: not run"); continue
        d = np.load(f, allow_pickle=False); nT = len(d["T"]); A0 = d["exact_A0"].astype(bool); B = d["exact_R60"].astype(bool); S = d["exact_R1"].astype(bool)
        Tn = d["exact_R01"].astype(bool); unsolve = int((A0[:-1] & ~A0[1:]).sum() + (B[:, :-1] & ~B[:, 1:]).sum() + (S[:, :-1] & ~S[:, 1:]).sum() + (Tn[:, :-1] & ~Tn[:, 1:]).sum())
        firsts = [first_hits(B[:, :, j].T) for j in range(nT)]                   # per puzzle: (K,) first hits after the kick
        hr = memory_ratio(firsts); lo, hi = boot_ci(firsts, [SEED, 22, w, 9])
        dE = sum(window_counts(f_, *E_WIN)[0] for f_ in firsts); dL = sum(window_counts(f_, *L_WIN)[0] for f_ in firsts)
        Db = disagreement(B[:K_SMALL, 15, :]); Ds = disagreement(S[:, 15, :]); ratio = Ds / Db if Db > 0 else float("nan")
        Dt = disagreement(d["exact_R01"].astype(bool)[:, 15, :]); ratio_tiny = Dt / Db if Db > 0 else float("nan")       # exploratory
        rng = np.random.default_rng([SEED, 22, w, 10]); bs = []
        for _ in range(N_BOOT):
            idx = rng.integers(0, nT, nT); db = disagreement(B[:K_SMALL, 15, idx]); bs.append(disagreement(S[:, 15, idx]) / db if db > 0 else np.nan)
        bs = np.array(bs); bs = bs[np.isfinite(bs)]; rci = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))) if len(bs) else (float("nan"),) * 2
        res = dict(triggered=nT, unsolve_transitions=unsolve, A=dict(hr=hr, ci=(lo, hi), events_early=dE, events_late=dL, letter=letter_a(hr, lo, hi)),
                   B=dict(D_small=Ds, D_big=Db, ratio=ratio, ci=rci, letter=letter_b(ratio)), B_tiny_exploratory=dict(D_tiny=Dt, ratio=ratio_tiny))
        # exploratory: exchangeability of the intact continuation, parallel vs sequential at matched compute, the hazard by window
        q = B[:, 15, :].mean(0); a0 = A0[15]
        par2 = B[0, 15] | B[1, 15]; seqA0 = A0[31]; seqR = B[0, 31]
        mc = lambda x_, y_: (int((x_ & ~y_).sum()), int((~x_ & y_).sum()), R17.mcnemar_exact(int((x_ & ~y_).sum()), int((~x_ & y_).sum())))
        res["exploratory"] = dict(A0_solved_16=int(a0.sum()), mean_reroll_q=float(q.mean() * nT), q_mean_A0_solved=float(q[a0].mean()) if a0.any() else None,
                                  q_mean_A0_unsolved=float(q[~a0].mean()) if (~a0).any() else None,
                                  parallel2_at16_vs_A0_at32=dict(parallel=int(par2.sum()), sequential=int(seqA0.sum()), mcnemar=mc(par2, seqA0)),
                                  parallel2_at16_vs_reroll_at32=dict(parallel=int(par2.sum()), sequential=int(seqR.sum()), mcnemar=mc(par2, seqR)),
                                  A0_solved_by=dict(zip(("+16", "+32"), (int(A0[15].sum()), int(A0[31].sum())))))
        la.append(res["A"]["letter"]); lb.append(res["B"]["letter"]); out[f"attention_{w}"] = res
        lines.append(f"attention_{w}: triggered {nT} | A: memory ratio R (survivors' late completions / predicted) {hr:.3f} [{lo:.3f}, {hi:.3f}] (events {dE} early, {dL} late) -> {res['A']['letter']} | "
                     f"B: disagreement 1deg {Ds:.3f} vs 60deg {Db:.3f}, ratio {ratio:.3f} [{rci[0]:.3f}, {rci[1]:.3f}] -> {res['B']['letter']} (0.1deg exploratory {Dt:.3f}, ratio {ratio_tiny:.3f}) | solved->unsolved {unsolve} | "
                     f"exploratory: A0 solved by +16 {int(a0.sum())} vs the re-rolls' expected {q.mean() * nT:.1f}; parallel 2x16 {int(par2.sum())} vs A0 32 {int(seqA0.sum())} {mc(par2, seqA0)} vs one re-roll 32 {int(seqR.sum())} {mc(par2, seqR)}")
    agg = {}
    for name, ls in (("A", la), ("B", lb)):
        counts = {v: ls.count(v) for v in set(ls)}; top = [v for v, c in counts.items() if c >= MIN_RECEIVERS]
        agg[name] = top[0] if top else "NO-CONSENSUS"
    out["aggregate"] = agg; lines.append(f"aggregate (the same letter on at least {MIN_RECEIVERS} of 3 receivers): A {agg['A']} | B {agg['B']}")
    (base / "report.txt").write_text("\n".join(lines) + "\n"); (base / "report.json").write_text(json.dumps(out, indent=1, default=float)); print("\n".join(lines))


def selftest():
    X = np.zeros((4, 3), bool); X[2, 0] = X[3, 0] = True; X[0, 1] = X[1, 1] = X[2, 1] = X[3, 1] = True
    assert list(first_hits(X)) == [3, 1, 5]
    f = np.array([3, 20, 40]); assert window_counts(f, 1, 16) == (1, 3 + 16 + 16) and window_counts(f, 17, 32) == (1, 4 + 16)
    rng = np.random.default_rng(0)                                               # memoryless synthetic data: constant per-puzzle hazards -> R near 1
    firsts = [np.minimum(rng.geometric(h, size=8), 33) for h in rng.uniform(0.005, 0.12, 2000)]
    hr = memory_ratio(firsts); assert 0.93 < hr < 1.07, hr
    aging = [np.minimum(np.where(rng.random(8) < 0.5, rng.geometric(0.15, 8), 99), 33) for _ in range(2000)]   # a half that never finishes
    assert memory_ratio(aging) < 0.7, memory_ratio(aging)
    acc = []
    for h in rng.uniform(0.005, 0.12, 2000):                                     # the hazard doubles after 16
        f = rng.geometric(h, 8); late = f > 16; f[late] = 16 + rng.geometric(min(2 * h, 0.9), int(late.sum())); acc.append(np.minimum(f, 33))
    assert memory_ratio(acc) > 1.3, memory_ratio(acc)
    assert abs(memory_ratio([np.array([1, 1, 20, 40])]) - 1 / (2 * 2 / 3)) < 1e-12               # dE 2, dL 1, K 4: R = 1 / (2*2/3)
    O = np.array([[1, 0], [1, 1], [0, 1], [1, 0]], bool); assert abs(disagreement(O) - np.mean([3 / 6, 4 / 6])) < 1e-12   # puzzle 0: 1,1,0,1; puzzle 1: 0,1,1,0
    assert letter_a(1.0, 0.8, 1.2) == "MEMORYLESS" and letter_a(0.6, 0.5, 0.9) == "AGING" and letter_a(1.0, 0.5, 1.9) == "UNRESOLVED" and letter_a(1.6, 1.2, 2.0) == "ACCUMULATING"
    assert letter_b(0.9) == "CHAOTIC" and letter_b(0.4) == "REGULAR" and letter_b(0.6) == "INTERMEDIATE"
    for w in (128,):
        p = pool(w); assert len(p) == N_DRAW and np.all(np.diff(p) > 0) and np.array_equal(p, pool(w))
        assert not (set(int(i) for i in p) & (set(int(i) for i in R17.pool(w)) | set(int(i) for i in R20.pool(w))))
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", nargs="?", choices=["run", "report"]); ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--width", type=int, choices=Q.WIDTHS); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.selftest: selftest(); return
    if a.action == "report": report(Path(a.out) if a.out else OUT); return
    if a.action == "run":
        if a.width is None: ap.error("--width required")
        R14(a.width, Path(a.out) if a.out else OUT).run(a.smoke)
    else: ap.print_help()


if __name__ == "__main__":
    main()
