#!/usr/bin/env python3
# Ledger: THE DISCUSSION-PERIOD EXPERIMENT P23 — does the memoryless chaotic tail transfer to another token mixer? P22's protocol and
# rules on the MLP 192 checkpoints (seeds 0-2) (registration Documentation/Note_2026-10-03_Rebuttal_P23_Registration.md, written before
# any row). MEASUREMENT, $0, inference only. Imports P22's statistics and reader helpers and P21's MLP loader (installed in this
# process only over P17's runner) unchanged; the run loop is P22's with the MLP pool and its own seed.
"""P23: per MLP seed, a fresh draw of test puzzles unsolved at 16 in that model's saved full-test record (excluding P21's pool); the
trigger is an invalid displayed grid at iteration 16 on the CPU; from that state: A0 intact (32 iterations), eight independent
60-degree score-preserving re-rolls (32 iterations each), four 1-degree re-rolls and four 0.1-degree re-rolls (16 iterations each).
Rules: P22's A (memory ratio R: AGING / ACCUMULATING / MEMORYLESS / UNRESOLVED) and B (chaos ratio: CHAOTIC / INTERMEDIATE / REGULAR),
confirmed on at least two of three seeds."""
import argparse, json, math, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import rebuttal_p1p2 as P
import rebuttal_p15 as Q
import rebuttal_p17 as R17
import rebuttal_p21 as R21
import rebuttal_p22 as R22

OUT = ROOT / "runs/analysis/rebuttal_20261003f"
SEED = 20261005
MODELS = R21.MODELS
N_DRAW = R22.N_DRAW
STEPS1, STEPS_LONG, STEPS_SHORT = R22.STEPS1, R22.STEPS_LONG, R22.STEPS_SHORT
K_BIG, K_SMALL, BIG, SMALL, TINY = R22.K_BIG, R22.K_SMALL, R22.BIG, R22.SMALL, R22.TINY
MIN_RECEIVERS = R22.MIN_RECEIVERS


def utc(): return datetime.now(timezone.utc).isoformat()


def pool(name):
    """A fresh seeded draw from the model's own record's unsolved-at-16 pool, excluding P21's pool for that model."""
    rec = np.load(ROOT / f"paper/code/evidence/benchmark/{name}_d16.npz", allow_pickle=False)
    excl = set(int(i) for i in R21.pool(name))
    cand = np.array([i for i in rec["idx"][~rec["cold_exact"]] if int(i) not in excl])
    return np.sort(np.random.default_rng([SEED, 23, R21.seed_of(name)]).choice(cand, size=N_DRAW, replace=False))


class R15(R17.R11):
    def __init__(self, name, out=OUT):
        assert name in MODELS
        R21.NAME["value"] = name; P.R2.__init__ = R21.mlp_base_init
        super().__init__(R21.WIDTH, out)
        self.name = name
        self.meta.update(study="rebuttal_20261003f", pipeline="tools/rebuttal_p23.py over tools/rebuttal_p22.py helpers, tools/rebuttal_p21.py loader, tools/rebuttal_p17.py runner")
        self.ATS.write_json(self.dir / "runtime.json", self.meta)

    def rerolls(self, z16, x, sol81, ids, T, angle, K, salt, steps, tag):
        """P22's re-roll loop with P23's seed: K independent score-preserving rotations per triggered state."""
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
                ang_dev = max(ang_dev, float(np.abs(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))) - angle).max() / angle))
                zz[q, 0] = h2
            zp = np.concatenate([zz, np.repeat(zz[:1], pad, 0)]) if pad else zz; xp = np.concatenate([xb, np.repeat(xb[:1], pad, 0)]) if pad else xb
            d, _ = self.loop(zp, xp, steps)
            for q, (k, j) in enumerate(chunk): out[k, :, j] = (d[:, q] == sol81[T[j]][None]).all(-1)
            self.status("running", model=self.name, arm=tag, done=c0 + len(chunk), of=len(pairs))
        return out, shift, ang_dev

    def run(self, smoke=False):
        try:
            ids = pool(self.name)
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
                d, _ = self.loop(zp, xp, STEPS_LONG); ex0.append((d[:, :len(tb)] == sol81[tb][None]).all(-1)); self.status("running", model=self.name, arm="A0_intact", done=c0 + len(tb), of=len(T))
            exA0 = np.concatenate(ex0, 1) if ex0 else np.zeros((STEPS_LONG, 0), bool); self.ATS.log("arm_complete", model=self.name, arm="A0_intact", of=len(T))
            exBig, sB, cB = self.rerolls(z16, x, sol81, ids, T, BIG, K_BIG, 101, STEPS_LONG, "R60_rerolls"); self.ATS.log("arm_complete", model=self.name, arm="R60_rerolls", of=len(T))
            exSmall, sS, cS = self.rerolls(z16, x, sol81, ids, T, SMALL, K_SMALL, 102, STEPS_SHORT, "R1_rerolls"); self.ATS.log("arm_complete", model=self.name, arm="R1_rerolls", of=len(T))
            exTiny, sT, cT = self.rerolls(z16, x, sol81, ids, T, TINY, K_SMALL, 103, STEPS_SHORT, "R01_rerolls"); self.ATS.log("arm_complete", model=self.name, arm="R01_rerolls", of=len(T))
            assert max(sB, sS, sT) < 1e-2, f"gate: a rotation changed a score by {max(sB, sS, sT)}"
            assert max(cB, cS, cT) < 0.01, f"gate: a realized rotation angle differs from its nominal angle by {max(cB, cS, cT):.2%}"
            dst = self.dir / ("p23_smoke.npz" if smoke else "p23.npz")
            np.savez_compressed(dst, ids=ids, triggered=trig, exact16=exact16, T=T, exact_A0=exA0, exact_R60=exBig, exact_R1=exSmall, exact_R01=exTiny,
                                shift_max=max(sB, sS, sT), angle_dev_max=max(cB, cS, cT),
                                meta=json.dumps(dict(self.meta, created=utc(), n=len(ids), triggered=int(trig.sum()), smoke=smoke)))
            if not smoke: self.ATS.write_json(dst.with_suffix(".sha256.json"), dict(sha256=self.ATS.sha(dst)))
            print(json.dumps(dict(model=self.name, n=len(ids), triggered=int(trig.sum()), shift_max=max(sB, sS, sT), angle_dev_max=max(cB, cS, cT))))
            self.ATS.verify_sources(); self.status("inference_complete", completed=utc()); self.ATS.log("model_complete", model=self.name, group="p23")
        except BaseException as exc:
            self.status("failed", error=repr(exc)); raise


def report(base=OUT, models=MODELS):
    """P22's reader applied to the MLP seeds (the same statistics, intervals and letters; bootstrap seeds from P23's SEED)."""
    lines = ["P23 REPORT (rules registered 2026-10-03, before any P23 row): P22's memory and chaos rules on MLP 192 (three seeds)"]
    out = {}; la, lb = [], []
    for name in models:
        f = base / f"{name}/p23.npz"
        if not f.exists(): lines.append(f"{name}: not run"); continue
        d = np.load(f, allow_pickle=False); nT = len(d["T"]); A0 = d["exact_A0"].astype(bool); B = d["exact_R60"].astype(bool); S = d["exact_R1"].astype(bool); Tn = d["exact_R01"].astype(bool)
        unsolve = int((A0[:-1] & ~A0[1:]).sum() + (B[:, :-1] & ~B[:, 1:]).sum() + (S[:, :-1] & ~S[:, 1:]).sum() + (Tn[:, :-1] & ~Tn[:, 1:]).sum())
        firsts = [R22.first_hits(B[:, :, j].T) for j in range(nT)]
        hr = R22.memory_ratio(firsts); lo, hi = R22.boot_ci(firsts, [SEED, 23, R21.seed_of(name), 9])
        dE = sum(R22.window_counts(f_, *R22.E_WIN)[0] for f_ in firsts); dL = sum(R22.window_counts(f_, *R22.L_WIN)[0] for f_ in firsts)
        Db = R22.disagreement(B[:K_SMALL, 15, :]); Ds = R22.disagreement(S[:, 15, :]); ratio = Ds / Db if Db > 0 else float("nan")
        Dt = R22.disagreement(Tn[:, 15, :]); ratio_tiny = Dt / Db if Db > 0 else float("nan")
        rng = np.random.default_rng([SEED, 23, R21.seed_of(name), 10]); bs = []
        for _ in range(R22.N_BOOT):
            idx = rng.integers(0, nT, nT); db = R22.disagreement(B[:K_SMALL, 15, idx]); bs.append(R22.disagreement(S[:, 15, idx]) / db if db > 0 else np.nan)
        bs = np.array(bs); bs = bs[np.isfinite(bs)]; rci = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))) if len(bs) else (float("nan"),) * 2
        q = B[:, 15, :].mean(0); a0 = A0[15]; par2 = B[0, 15] | B[1, 15]
        mc = lambda x_, y_: (int((x_ & ~y_).sum()), int((~x_ & y_).sum()), R17.mcnemar_exact(int((x_ & ~y_).sum()), int((~x_ & y_).sum())))
        res = dict(triggered=nT, unsolve_transitions=unsolve, A=dict(hr=hr, ci=(lo, hi), events_early=dE, events_late=dL, letter=R22.letter_a(hr, lo, hi)),
                   B=dict(D_small=Ds, D_big=Db, ratio=ratio, ci=rci, letter=R22.letter_b(ratio)), B_tiny_exploratory=dict(D_tiny=Dt, ratio=ratio_tiny),
                   exploratory=dict(A0_solved_16=int(a0.sum()), mean_reroll_q=float(q.mean() * nT), A0_solved_32=int(A0[31].sum()),
                                    memoryless_prediction_A0_32=float((1 - (1 - q) ** 2).sum()),
                                    parallel2_at16_vs_A0_at32=dict(parallel=int(par2.sum()), sequential=int(A0[31].sum()), mcnemar=mc(par2, A0[31]))))
        la.append(res["A"]["letter"]); lb.append(res["B"]["letter"]); out[name] = res
        lines.append(f"{name}: triggered {nT} | A: memory ratio R {hr:.3f} [{lo:.3f}, {hi:.3f}] (events {dE} early, {dL} late) -> {res['A']['letter']} | "
                     f"B: disagreement 1deg {Ds:.3f} vs 60deg {Db:.3f}, ratio {ratio:.3f} [{rci[0]:.3f}, {rci[1]:.3f}] -> {res['B']['letter']} (0.1deg exploratory {Dt:.3f}, ratio {ratio_tiny:.3f}) | solved->unsolved {unsolve} | "
                     f"exploratory: A0 solved by +16 {int(a0.sum())} vs the re-rolls' expected {q.mean() * nT:.1f}; A0 by +32 {int(A0[31].sum())} vs memoryless prediction {(1 - (1 - q) ** 2).sum():.1f}; parallel 2x16 {int(par2.sum())} {mc(par2, A0[31])}")
    agg = {}
    for nm, ls in (("A", la), ("B", lb)):
        top = [v for v in set(ls) if ls.count(v) >= MIN_RECEIVERS]; agg[nm] = top[0] if top else "NO-CONSENSUS"
    out["aggregate"] = agg; lines.append(f"aggregate (the same letter on at least {MIN_RECEIVERS} of 3 seeds): A {agg['A']} | B {agg['B']}")
    (base / "report.txt").write_text("\n".join(lines) + "\n"); (base / "report.json").write_text(json.dumps(out, indent=1, default=float)); print("\n".join(lines))


def selftest():
    for name in MODELS:
        p = pool(name); assert len(p) == N_DRAW and np.all(np.diff(p) > 0) and np.array_equal(p, pool(name))
        assert not (set(int(i) for i in p) & set(int(i) for i in R21.pool(name)))
        rec = np.load(ROOT / f"paper/code/evidence/benchmark/{name}_d16.npz", allow_pickle=False); assert not np.isin(p, rec["idx"][rec["cold_exact"]]).any()
    assert SEED != R22.SEED and (STEPS1, STEPS_LONG, STEPS_SHORT, K_BIG, K_SMALL) == (16, 32, 16, 8, 4)
    R22.selftest()                                                               # the shared statistics, letters and pools
    print("selftest OK (P23)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", nargs="?", choices=["run", "report"]); ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model", choices=MODELS); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.selftest: selftest(); return
    if a.action == "report": report(Path(a.out) if a.out else OUT); return
    if a.action == "run":
        if a.model is None: ap.error("--model required")
        R15(a.model, Path(a.out) if a.out else OUT).run(a.smoke)
    else: ap.print_help()


if __name__ == "__main__":
    main()
