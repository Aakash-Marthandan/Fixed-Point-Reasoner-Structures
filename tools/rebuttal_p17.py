#!/usr/bin/env python3
# Ledger: THE DISCUSSION-PERIOD EXPERIMENT P17 — using the collective account to restart stalled correction (registration:
# Documentation/Note_2026-10-02_Rebuttal_P15_P16_P17_Registration.md, Amendment 4, written before any row). MEASUREMENT, $0, inference
# only on the banked attention checkpoints; imports tools/rebuttal_p15.py (R7: model, release step), tools/rebuttal_p3.py (rotation) and
# tools/rebuttal_p12.py (its population, excluded) unchanged.
"""P17: on test puzzles unsolved at 16 iterations in the saved full-test record, run 16 fixed-start iterations on the CPU; the trigger
(no answer key) is a visible duplicate in the displayed grid at iteration 16. From the iteration-16 state, all arms continue to iteration
32: A0 intact; A1 one norm-matched random replacement of the readout-invisible slow state at every cell (P12's nudge); A2 rotations of
that part by 30 or 60 degrees; A3 the replacement only at cells in visible conflicts; A4 GUIDED: A1, then another full replacement at
any later iteration where the display is unchanged and still violates a constraint (at most every second iteration); A5 a fresh
Gaussian restart for 16 iterations. Rules: GUIDED-HELPS, GLOBAL-BEATS-LOCAL."""
import argparse, json, math, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import rebuttal_p15 as Q
import rebuttal_p1p2 as P

OUT = ROOT / "runs/analysis/rebuttal_20261003b"
SEED = 20261002
N_DRAW, STEPS1, STEPS2, STEPS_LABEL = 384, 16, 16, 48          # A0 continues to iteration 64 to label transient vs trapped
DATA = ROOT / "data/sudoku_extreme/sudoku_extreme_seed0_mon512.npz"
ARMS = ("A0_intact", "A1_replace", "A2_rot15", "A2_rot30", "A2_rot60", "A3_local", "A3b_local_random", "A4_guided", "A5_restart")
GUIDED, P_MAX, GL_DIFF, RESCUE_MIN, MIN_RECEIVERS = "A4_guided", 0.05, 0.10, 0.10, 2


def utc(): return datetime.now(timezone.utc).isoformat()


def violations(grid81):
    """Per cell: True if its displayed digit duplicates a peer's displayed digit (any row, column or box)."""
    g = np.asarray(grid81).reshape(-1, 81)
    return np.stack([np.array([bool((g[b][P.UNITS[i]] == g[b][i]).any()) for i in range(81)]) for b in range(len(g))])


def invalid(grid81, puz81):
    """The evaluator's ok flag negated: a visible duplicate, or a given altered."""
    g = np.asarray(grid81).reshape(-1, 81); pz = np.asarray(puz81).reshape(-1, 81)
    return violations(g).any(1) | ((pz != 0) & (g != pz)).any(1)


def mcnemar_exact(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c); return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def pool(width):
    """New seeded draw from the record's unsolved-at-16 pool, excluding P12's pool."""
    import rebuttal_p12 as P12
    rec = np.load(ROOT / f"paper/code/evidence/benchmark/attention_{width}_d16.npz", allow_pickle=False)
    r128 = np.load(ROOT / "paper/code/evidence/benchmark/attention_128_d16.npz", allow_pickle=False)
    excl = set(int(i) for i in P12.pool_ids(r128["cold_exact"], r128["idx"]))
    cand = np.array([i for i in rec["idx"][~rec["cold_exact"]] if int(i) not in excl])
    return np.sort(np.random.default_rng([SEED, 17, width]).choice(cand, size=N_DRAW, replace=False))


class R11(Q.R7):
    def __init__(self, width, out=OUT):
        super().__init__(width, out)
        self.meta.update(study="rebuttal_20261003b", pipeline="tools/rebuttal_p17.py over tools/rebuttal_p15.py")
        self.ATS.write_json(self.dir / "runtime.json", self.meta)
        self.step_first = self.EV._step(self.cfg, 1., 0., True)

    def loop(self, z, x, steps, first=False, hook=None):
        """The study's loop through the release step; returns displays (steps, B, 81) and the final carried state. hook(t, z, prev, disp) -> z."""
        jnp = self.jnp; x = jnp.asarray(x); y = jnp.broadcast_to(self.void, (x.shape[0],) + self.void.shape); disp = []; prev = None
        for t in range(steps):
            f = first and t == 0
            lg_dev, zf = (self.step_first if f else self.step_rel)(self.params, x, y, self.tv, jnp.zeros(1) if f else jnp.asarray(z))
            z = np.array(zf) if f else np.array(jnp.asarray(z) + self.eta_z * (zf - jnp.asarray(z)))     # writable copies: the guided arm edits z in place
            d = (np.asarray(self.EV.layout_gather(lg_dev, self.layout), np.float32)[..., 1:10].reshape(x.shape[0], 81, 9).argmax(-1) + 1).astype(np.int8)
            disp.append(d)
            if hook is not None: z = hook(t, z, prev, d)
            prev = d
        return np.stack(disp), z

    def replace(self, h, cells, rng):
        """Norm-matched random readout-orthogonal parts at the given cells (all nine fields), scores kept."""
        h = h.astype(np.float64).copy(); par, perp = P.project(h[:, cells], self.lm)
        h[:, cells] = par + P.norm_matched_random(perp, self.lm, rng)
        return h.astype(np.float32)

    def rotate(self, h, theta, rng):
        import rebuttal_p3 as P3
        h = h.astype(np.float64); par, perp = P.project(h, self.lm)
        return (par + P3.rotate_perp(perp, self.lm, theta, rng)).astype(np.float32)

    def run(self, smoke=False):
        try:
            ids = pool(self.width)
            if smoke: ids = ids[:16]
            with np.load(DATA, allow_pickle=False) as d: puz, sol = d["test_q"][ids].astype(np.int32), d["test_a"][ids].astype(np.int32)
            x = np.asarray(self.EV.place_batch(puz, self.layout)); sol81 = sol.reshape(-1, 81)
            disp1, z16 = [], []
            for b0 in range(0, len(ids), Q.VBATCH):
                sl = slice(b0, b0 + Q.VBATCH); xb = x[sl]; pad = Q.VBATCH - len(xb)
                xp = np.concatenate([xb, np.repeat(xb[:1], pad, 0)]) if pad else xb
                d, z = self.loop(None, xp, STEPS1, first=True); disp1.append(d[:, :len(xb)]); z16.append(z[:len(xb)])
            disp1 = np.concatenate(disp1, 1); z16 = np.concatenate(z16, 0)
            trig = invalid(disp1[-1], puz.reshape(-1, 81)); exact16 = (disp1[-1] == sol81).all(1)
            assert not (trig & exact16).any(), "a triggered grid is exact"
            assert np.all(trig | exact16), "an untriggered grid is not exact (a full grid without violations must be the solution)"
            T = np.where(trig)[0]; shift_max = 0.0; results = {}; nudges = np.zeros(len(T), int)
            for arm in ARMS:
                rows, ex = [], []
                for c0 in range(0, len(T), Q.VBATCH):
                    tb = T[c0:c0 + Q.VBATCH]; pad = Q.VBATCH - len(tb); zz = z16[tb].copy(); xb = x[tb]
                    if arm == "A5_restart":
                        shp = zz.shape[1:]; zz = np.stack([self.EV.mi_z0(SEED, int(ids[b]), 72, shp, 1.0, "gauss") for b in tb]).astype(np.float32)
                    else:
                        for q, b in enumerate(tb):
                            pid = int(ids[b]); h = zz[q, 0]
                            if arm in ("A1_replace", "A4_guided"): h2 = self.replace(h, list(range(81)), np.random.default_rng([SEED, pid, 71]))
                            elif arm.startswith("A2_rot"): h2 = self.rotate(h, float(arm[-2:]), np.random.default_rng([SEED, pid, 73, int(arm[-2:])]))
                            elif arm == "A3_local":
                                vc = list(np.where(violations(disp1[-1, b])[0])[0]); h2 = self.replace(h, vc, np.random.default_rng([SEED, pid, 74]))
                            elif arm == "A3b_local_random":                    # size-matched random non-conflict cells (the A3 control)
                                vmask = violations(disp1[-1, b])[0]; nonc = np.where(~vmask)[0]; rr = np.random.default_rng([SEED, pid, 76])
                                k = int(vmask.sum()); cells = sorted(int(c) for c in rr.choice(nonc, size=min(k, len(nonc)), replace=False))
                                h2 = self.replace(h, cells, np.random.default_rng([SEED, pid, 77]))
                            else: h2 = h
                            shift_max = max(shift_max, float(np.abs(np.einsum("fsw,w->sf", h2.astype(np.float64), self.lm) - np.einsum("fsw,w->sf", h.astype(np.float64), self.lm)).max()))
                            zz[q, 0] = h2
                    last = {q: 0 for q in range(len(tb))}
                    def hook(t, z, prev, d, tb=tb, last=last):
                        if arm != "A4_guided" or prev is None: return z
                        v = invalid(d[:len(tb)], puz.reshape(-1, 81)[tb])
                        for q in range(len(tb)):
                            if v[q] and np.array_equal(prev[q], d[q]) and (t - last[q]) >= 2:
                                pid = int(ids[tb[q]]); z[q, 0] = self.replace(z[q, 0], list(range(81)), np.random.default_rng([SEED, pid, 75, t])); last[q] = t
                                nudges[c0 + q] += 1
                        return z
                    zp = np.concatenate([zz, np.repeat(zz[:1], pad, 0)]) if pad else zz; xp = np.concatenate([xb, np.repeat(xb[:1], pad, 0)]) if pad else xb
                    d, _ = self.loop(zp, xp, STEPS2 + (STEPS_LABEL if arm == "A0_intact" else 0), hook=hook); ex.append((d[:, :len(tb)] == sol81[tb][None]).all(-1))
                    self.status("running", arm=arm, done=c0 + len(tb), of=len(T))
                results[arm] = np.concatenate(ex, 1)                                             # (16, nT)
                self.ATS.log("arm_complete", width=self.width, arm=arm, of=len(T))       # no outcome counts in the log (read by the registered reader only)
            assert shift_max < 1e-2, f"gate: a nudge changed a score by {shift_max}"
            dst = self.dir / ("p17_smoke.npz" if smoke else "p17.npz")
            np.savez_compressed(dst, ids=ids, triggered=trig, exact16=exact16, T=T, nudges_guided=nudges, shift_max=shift_max,
                                **{f"exact_{a}": results[a] for a in ARMS}, meta=json.dumps(dict(self.meta, created=utc(), n=len(ids), triggered=int(trig.sum()), smoke=smoke)))
            if not smoke: self.ATS.write_json(dst.with_suffix(".sha256.json"), dict(sha256=self.ATS.sha(dst)))
            print(json.dumps(dict(width=self.width, n=len(ids), triggered=int(trig.sum()), exact16_cpu=int(exact16.sum()), shift_max=shift_max, guided_nudges_mean=float(nudges.mean()) if len(nudges) else None)))
            self.ATS.verify_sources(); self.status("inference_complete", completed=utc()); self.ATS.log("model_complete", width=self.width, group="p17")
        except BaseException as exc:
            self.status("failed", error=repr(exc)); raise


def report(base=OUT, widths=Q.WIDTHS):
    lines = ["P17 REPORT (rules registered 2026-10-02, Amendment 4): restarting stalled correction; solved by iteration 32 among triggered puzzles, by stratum"]; out = {}; guided_letters = []
    for w in widths:
        f = base / f"attention_{w}/p17.npz"
        if not f.exists(): lines.append(f"attention_{w}: not run"); continue
        d = np.load(f, allow_pickle=False); nT = len(d["T"])
        A0 = d["exact_A0_intact"]; trapped = ~A0[-1]; transient = ~trapped                     # A0 to iteration 64 (index 47); solved by 32 = index 15
        S = {a: d[f"exact_{a}"][15] for a in ARMS}
        res = dict(n=int(len(d["ids"])), triggered=nT, transient=int(transient.sum()), trapped=int(trapped.sum()), strata={})
        for name, m in (("all", np.ones(nT, bool)), ("transient", transient), ("trapped", trapped)):
            rows = {a: int(S[a][m].sum()) for a in ARMS}; losses = {a: int((S["A0_intact"][m] & ~S[a][m]).sum()) for a in ARMS if a != "A0_intact"}
            others = [a for a in ARMS if a != GUIDED]; strongest = max(others, key=lambda a: rows[a])
            b = int((S[GUIDED][m] & ~S[strongest][m]).sum()); c = int((~S[GUIDED][m] & S[strongest][m]).sum()); pg = mcnemar_exact(b, c)
            con = {}
            for x_, y_ in (("A4_guided", "A1_replace"), ("A4_guided", "A5_restart"), ("A1_replace", "A3_local"), ("A3_local", "A3b_local_random"), ("A1_replace", "A0_intact")):
                bb = int((S[x_][m] & ~S[y_][m]).sum()); cc = int((~S[x_][m] & S[y_][m]).sum()); con[f"{x_}_vs_{y_}"] = (bb, cc, mcnemar_exact(bb, cc))
            res["strata"][name] = dict(solved=rows, losses=losses, strongest_other=strongest, guided_vs_strongest=(b, c, pg), contrasts=con, n=int(m.sum()),
                                       guided_helps=bool(all(rows[GUIDED] > rows[a] for a in others) and pg < P_MAX))
            lines.append(f"attention_{w} {name} (n {int(m.sum())}): solved by 32 " + " ".join(f"{a} {rows[a]}" for a in ARMS) + " | losses vs A0 " + " ".join(f"{a} {v}" for a, v in losses.items())
                         + f" | guided vs strongest ({strongest}) {b}/{c} p {pg:.3g} | A4-A1 {con['A4_guided_vs_A1_replace']} A4-A5 {con['A4_guided_vs_A5_restart']} A1-A3 {con['A1_replace_vs_A3_local']} A3-A3b {con['A3_local_vs_A3b_local_random']}")
        st = res["strata"]; gl = st["all"]["contrasts"]["A1_replace_vs_A3_local"]
        res["letter_guided"] = "GUIDED-HELPS" if st["all"]["guided_helps"] else "NO-GAIN"
        res["letter_guided_stratum"] = ("rescue" if st["trapped"]["guided_helps"] else ("speed-up" if st["transient"]["guided_helps"] else "neither"))
        tr = st["trapped"]; resc = tr["solved"][GUIDED] / max(tr["n"], 1)
        tr_others = [a for a in ARMS if a not in (GUIDED, "A0_intact")]; tr_strong = max(tr_others, key=lambda a: tr["solved"][a])
        bt = int((S[GUIDED][trapped] & ~S[tr_strong][trapped]).sum()); ct = int((~S[GUIDED][trapped] & S[tr_strong][trapped]).sum()); pt = mcnemar_exact(bt, ct)
        res["letter_rescue"] = "TRAPPED-RESCUE" if (resc >= RESCUE_MIN and all(tr["solved"][GUIDED] > tr["solved"][a] for a in tr_others) and pt < P_MAX) else "NO-RESCUE"
        res["letter_global"] = "GLOBAL-BEATS-LOCAL" if ((st["all"]["solved"]["A1_replace"] - st["all"]["solved"]["A3_local"]) / max(nT, 1) >= GL_DIFF and gl[2] < P_MAX) else "NOT-SHOWN"
        res["guided_nudges_mean"] = float(d["nudges_guided"].mean()) if nT else None
        guided_letters.append(res["letter_guided"]); out[f"attention_{w}"] = res
        lines.append(f"attention_{w}: letters GUIDED {res['letter_guided']} ({res['letter_guided_stratum']}) | RESCUE {res['letter_rescue']} | GLOBAL {res['letter_global']} | guided kicks per puzzle {res['guided_nudges_mean']:.2f}")
    overall = "GUIDED-HELPS" if sum(1 for g in guided_letters if g == "GUIDED-HELPS") >= MIN_RECEIVERS else "NO-GAIN"
    lines.append(f"aggregate (GUIDED-HELPS on at least {MIN_RECEIVERS} of 3 receivers): {overall}"); out["aggregate_guided"] = overall
    (base / "report.txt").write_text("\n".join(lines) + "\n"); (base / "report.json").write_text(json.dumps(out, indent=1, default=float)); print("\n".join(lines))


def selftest():
    base = np.array([[(3 * (r % 3) + r // 3 + c) % 9 + 1 for c in range(9)] for r in range(9)]).reshape(81)
    assert not violations(base).any()
    g = base.copy(); g[0] = g[1]; v = violations(g)[0]; assert v[0] and v[1] and v.sum() >= 2
    puz = base.copy(); puz[10:] = 0; g2 = np.where(base == 1, 2, np.where(base == 2, 1, base))   # another valid grid (digits 1 and 2 relabelled): no duplicate, givens altered
    assert not violations(g2).any() and invalid(g2, puz)[0] and not invalid(base, puz)[0]
    assert mcnemar_exact(10, 0) < 0.01 and mcnemar_exact(5, 5) == 1.0
    p = pool(128); assert len(p) == N_DRAW and np.all(np.diff(p) > 0) and np.array_equal(p, pool(128))
    import rebuttal_p12 as P12
    r128 = np.load(ROOT / "paper/code/evidence/benchmark/attention_128_d16.npz", allow_pickle=False)
    assert not (set(int(i) for i in p) & set(int(i) for i in P12.pool_ids(r128["cold_exact"], r128["idx"])))
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
        R11(a.width, Path(a.out) if a.out else OUT).run(a.smoke)
    else: ap.print_help()


if __name__ == "__main__":
    main()
