#!/usr/bin/env python3
# Ledger: THE DISCUSSION-PERIOD EXPERIMENT P20 — a confirmatory test of the moderate hidden-state rotation found in P17's exploratory dose
# arms (registration Documentation/Note_2026-10-03_Rebuttal_P20_Registration.md, written before any row). MEASUREMENT, $0, inference only
# on the banked attention checkpoints; imports tools/rebuttal_p17.py (R11: loop, kicks, trigger, pool) unchanged.
"""P20: on a fresh draw of test puzzles unsolved at 16 in the saved full-test record (excluding P12's and P17's pools), the trigger is an
invalid displayed grid at iteration 16 on the CPU; from that state: A0 intact (to 64 for the strata), R60 (pre-specified: the readout-
invisible slow part rotated by 60 degrees toward a random orthogonal direction, scores and norms kept), R45 and R75 (the operating range,
exploratory), and A5 a fresh Gaussian restart for 16 iterations. Rules: ROTATION-HELPS, BEATS-RESTART, TRAPPED-RESCUE."""
import argparse, json, math, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import rebuttal_p17 as R17
import rebuttal_p15 as Q

OUT = ROOT / "runs/analysis/rebuttal_20261003c"
SEED = 20261003
N_DRAW = 384
ARMS = ("A0_intact", "R45", "R60", "R75", "A5_restart")
P_MAX, RESCUE_MIN, MIN_RECEIVERS = 0.05, 0.10, 2


def utc(): return datetime.now(timezone.utc).isoformat()


def pool(width):
    """A fresh seeded draw from the record's unsolved-at-16 pool, excluding P12's pool and this width's P17 pool."""
    import rebuttal_p12 as P12
    rec = np.load(ROOT / f"paper/code/evidence/benchmark/attention_{width}_d16.npz", allow_pickle=False)
    r128 = np.load(ROOT / "paper/code/evidence/benchmark/attention_128_d16.npz", allow_pickle=False)
    excl = set(int(i) for i in P12.pool_ids(r128["cold_exact"], r128["idx"])) | set(int(i) for i in R17.pool(width))
    cand = np.array([i for i in rec["idx"][~rec["cold_exact"]] if int(i) not in excl])
    return np.sort(np.random.default_rng([SEED, 20, width]).choice(cand, size=N_DRAW, replace=False))


class R12(R17.R11):
    def __init__(self, width, out=OUT):
        super().__init__(width, out)
        self.meta.update(study="rebuttal_20261003c", pipeline="tools/rebuttal_p20.py over tools/rebuttal_p17.py over tools/rebuttal_p15.py")
        self.ATS.write_json(self.dir / "runtime.json", self.meta)

    def run(self, smoke=False):
        try:
            ids = pool(self.width)
            if smoke: ids = ids[:16]
            with np.load(R17.DATA, allow_pickle=False) as d: puz, sol = d["test_q"][ids].astype(np.int32), d["test_a"][ids].astype(np.int32)
            x = np.asarray(self.EV.place_batch(puz, self.layout)); sol81 = sol.reshape(-1, 81); puz81 = puz.reshape(-1, 81)
            disp1, z16 = [], []
            for b0 in range(0, len(ids), Q.VBATCH):
                xb = x[b0:b0 + Q.VBATCH]; pad = Q.VBATCH - len(xb); xp = np.concatenate([xb, np.repeat(xb[:1], pad, 0)]) if pad else xb
                d, z = self.loop(None, xp, R17.STEPS1, first=True); disp1.append(d[:, :len(xb)]); z16.append(z[:len(xb)])
            disp1 = np.concatenate(disp1, 1); z16 = np.concatenate(z16, 0)
            trig = R17.invalid(disp1[-1], puz81); exact16 = (disp1[-1] == sol81).all(1)
            assert not (trig & exact16).any() and np.all(trig | exact16), "trigger is not equivalent to CPU-unsolved"
            T = np.where(trig)[0]; shift_max = 0.0; results = {}
            for arm in ARMS:
                ex = []
                for c0 in range(0, len(T), Q.VBATCH):
                    tb = T[c0:c0 + Q.VBATCH]; pad = Q.VBATCH - len(tb); zz = z16[tb].copy(); xb = x[tb]
                    if arm == "A5_restart":
                        zz = np.stack([self.EV.mi_z0(SEED, int(ids[b]), 82, zz.shape[1:], 1.0, "gauss") for b in tb]).astype(np.float32)
                    elif arm.startswith("R"):
                        ang = float(arm[1:])
                        for q, b in enumerate(tb):
                            h = zz[q, 0]; h2 = self.rotate(h, ang, np.random.default_rng([SEED, int(ids[b]), 81, int(ang)]))
                            shift_max = max(shift_max, float(np.abs(np.einsum("fsw,w->sf", h2.astype(np.float64), self.lm) - np.einsum("fsw,w->sf", h.astype(np.float64), self.lm)).max()))
                            zz[q, 0] = h2
                    zp = np.concatenate([zz, np.repeat(zz[:1], pad, 0)]) if pad else zz; xp = np.concatenate([xb, np.repeat(xb[:1], pad, 0)]) if pad else xb
                    d, _ = self.loop(zp, xp, R17.STEPS2 + (R17.STEPS_LABEL if arm == "A0_intact" else 0)); ex.append((d[:, :len(tb)] == sol81[tb][None]).all(-1))
                    self.status("running", arm=arm, done=c0 + len(tb), of=len(T))
                results[arm] = np.concatenate(ex, 1); self.ATS.log("arm_complete", width=self.width, arm=arm, of=len(T))
            assert shift_max < 1e-2, f"gate: a rotation changed a score by {shift_max}"
            dst = self.dir / ("p20_smoke.npz" if smoke else "p20.npz")
            np.savez_compressed(dst, ids=ids, triggered=trig, exact16=exact16, T=T, shift_max=shift_max, **{f"exact_{a}": results[a] for a in ARMS},
                                meta=json.dumps(dict(self.meta, created=utc(), n=len(ids), triggered=int(trig.sum()), smoke=smoke)))
            if not smoke: self.ATS.write_json(dst.with_suffix(".sha256.json"), dict(sha256=self.ATS.sha(dst)))
            print(json.dumps(dict(width=self.width, n=len(ids), triggered=int(trig.sum()), shift_max=shift_max)))
            self.ATS.verify_sources(); self.status("inference_complete", completed=utc()); self.ATS.log("model_complete", width=self.width, group="p20")
        except BaseException as exc:
            self.status("failed", error=repr(exc)); raise


def report(base=OUT, widths=Q.WIDTHS):
    lines = ["P20 REPORT (rules registered 2026-10-03): the confirmatory 60-degree hidden-state rotation; solved by iteration 32 among triggered puzzles"]; out = {}; helps, beats, rescues = [], [], []
    for w in widths:
        f = base / f"attention_{w}/p20.npz"
        if not f.exists(): lines.append(f"attention_{w}: not run"); continue
        d = np.load(f, allow_pickle=False); nT = len(d["T"]); trapped = ~d["exact_A0_intact"][R17.STEPS_LABEL - 1].astype(bool)   # registered label: iteration 64 (A0 runs on to 80; Amendment 1)
        S = {a: d[f"exact_{a}"][15].astype(bool) for a in ARMS}; res = dict(triggered=nT, trapped=int(trapped.sum()), trapped_by_80_exploratory=int((~d["exact_A0_intact"][-1].astype(bool)).sum()), strata={})
        for name, m in (("all", np.ones(nT, bool)), ("transient", ~trapped), ("trapped", trapped)):
            res["strata"][name] = dict(solved={a: int(S[a][m].sum()) for a in ARMS}, losses={a: int((S["A0_intact"][m] & ~S[a][m]).sum()) for a in ARMS[1:]}, n=int(m.sum()))
        con = {}
        for x_, y_ in (("R60", "A0_intact"), ("R60", "A5_restart"), ("R45", "A0_intact"), ("R75", "A0_intact")):
            b = int((S[x_] & ~S[y_]).sum()); c = int((~S[x_] & S[y_]).sum()); con[f"{x_}_vs_{y_}"] = (b, c, R17.mcnemar_exact(b, c))
        h = con["R60_vs_A0_intact"]; r5 = con["R60_vs_A5_restart"]
        res["contrasts"] = con; res["letter_helps"] = "ROTATION-HELPS" if (h[0] > h[1] and h[2] < P_MAX) else ("ROTATION-HURTS" if (h[1] > h[0] and h[2] < P_MAX) else "NO-DIFFERENCE")
        res["letter_restart"] = "BEATS-RESTART" if (r5[0] > r5[1] and r5[2] < P_MAX) else "NOT-SHOWN"
        rr = res["strata"]["trapped"]["solved"]["R60"] / max(res["strata"]["trapped"]["n"], 1); res["rescue_rate"] = rr
        helps.append(res["letter_helps"]); beats.append(res["letter_restart"]); rescues.append(rr >= RESCUE_MIN); out[f"attention_{w}"] = res
        lines.append(f"attention_{w}: triggered {nT}, trapped {int(trapped.sum())} | solved by 32 " + " ".join(f"{a} {res['strata']['all']['solved'][a]}" for a in ARMS)
                     + f" | R60 vs intact {h[0]}/{h[1]} p {h[2]:.3g} -> {res['letter_helps']} | R60 vs restart {r5[0]}/{r5[1]} p {r5[2]:.3g} -> {res['letter_restart']} | trapped rescued by R60 {rr:.2f}"
                     + " | transient " + " ".join(f"{a} {res['strata']['transient']['solved'][a]}" for a in ARMS) + " | trapped " + " ".join(f"{a} {res['strata']['trapped']['solved'][a]}" for a in ARMS))
    agg = dict(ROTATION_HELPS=sum(1 for v in helps if v == "ROTATION-HELPS") >= MIN_RECEIVERS, BEATS_RESTART=sum(1 for v in beats if v == "BEATS-RESTART") >= MIN_RECEIVERS,
               TRAPPED_RESCUE=sum(rescues) >= MIN_RECEIVERS)
    out["aggregate"] = agg; lines.append(f"aggregate (at least {MIN_RECEIVERS} of 3 receivers): " + " ".join(f"{k} {v}" for k, v in agg.items()))
    (base / "report.txt").write_text("\n".join(lines) + "\n"); (base / "report.json").write_text(json.dumps(out, indent=1, default=float)); print("\n".join(lines))


def selftest():
    for w in (128,):
        p = pool(w); assert len(p) == N_DRAW and np.all(np.diff(p) > 0) and np.array_equal(p, pool(w))
        assert not (set(int(i) for i in p) & set(int(i) for i in R17.pool(w)))
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
        R12(a.width, Path(a.out) if a.out else OUT).run(a.smoke)
    else: ap.print_help()


if __name__ == "__main__":
    main()
