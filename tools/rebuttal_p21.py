#!/usr/bin/env python3
# Ledger: THE DISCUSSION-PERIOD EXPERIMENT P21 — transfer of the answer-preserving hidden-state rotation to another token mixer and to
# independent initializations (MLP 192, seeds 0-2) (registration Documentation/Note_2026-10-03_Rebuttal_P21_Registration.md, written
# before any row and before any P20 result). MEASUREMENT, $0, inference only on the banked MLP checkpoints; imports P17's runner
# (tools/rebuttal_p17.py: loop, rotation, trigger) unchanged and replaces, in this process only, the base loader
# (tools/rebuttal_p1p2.py R2.__init__) with one for the named MLP checkpoint; every downstream line is unchanged.
"""P21: per MLP seed, a fresh draw of test puzzles unsolved at 16 in that model's saved full-test record; the trigger is an invalid
displayed grid at iteration 16 on the CPU; from that state: A0 intact (to 64, the strata), R60 (pre-specified, P20's operation: the
readout-invisible slow part rotated by 60 degrees toward a random orthogonal direction, scores and norms kept; to 64), R45 and R75
(the operating range, exploratory; to 32) and A5 a fresh Gaussian restart for 16 iterations. Rules: TRANSFERS (ROTATION-HELPS on at
least two of three seeds), BEATS-RESTART, TRAPPED-RESCUE; exploratory: the long horizon (solved by 48 and 64; waiting saved)."""
import argparse, json, os, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import rebuttal_p1p2 as P
import rebuttal_p15 as Q
import rebuttal_p17 as R17

OUT = ROOT / "runs/analysis/rebuttal_20261003d"
SEED = 20261003
MODELS = ("mlp_192_seed0", "mlp_192_seed1", "mlp_192_seed2")
WIDTH = 192
N_DRAW = 384
STEPS1, STEPS2, STEPS_LONG = 16, 16, 48            # trigger at 16; short arms to 32; A0 and R60 to 64 (index 47); solved by 32 = index 15
ARMS = ("A0_intact", "R45", "R60", "R75", "A5_restart")
LONG = ("A0_intact", "R60")
P_MAX, RESCUE_MIN, MIN_RECEIVERS = 0.05, 0.10, 2
NAME = {"value": None}


def utc(): return datetime.now(timezone.utc).isoformat()


def seed_of(name): return int(name.rsplit("seed", 1)[1])


def mlp_base_init(self, width, out=P.OUT):
    """R2.__init__ for a named MLP release checkpoint (installed in this process only); every line not about the mixer is R2's."""
    import attention_transfer_study as ATS, platform
    name = NAME["value"]
    self.ATS = ATS; self.STUDY = ATS.OUT; self.RELEASE = ATS.RELEASE
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    sys.path[:0] = [str(self.RELEASE / "src"), str(self.RELEASE / "tools")]
    import jax, jax.numpy as jnp
    from qhrrn2 import episodic as E, model as M, grid as G, dec_cell as DC
    from qhrrn2.config import Config
    import eval_sudoku_extreme as EV
    self.jax, self.jnp, self.EV, self.DC, self.G = jax, jnp, EV, DC, G
    self.width = width
    self.source_hash = ATS.verify_sources()
    self.path = self.RELEASE / f"checkpoints/{name}.npz"
    manifest = json.loads((self.RELEASE / "MANIFEST.json").read_text())["files"][f"checkpoints/{name}.npz"]
    assert ATS.sha(self.path) == manifest["sha256"], "gate: the checkpoint's hash differs from the release manifest"
    saved = E.load_ckpt(self.path)
    self.cfg = cfg = Config(**saved["config"])
    inventory = json.loads((self.RELEASE / "evidence/models.json").read_text())[name]
    assert saved["step"] == inventory["selected_step"]
    assert cfg.dec_width == width and cfg.dec_coupling_kind == "mean" and cfg.dec_token_mixer == "mlp"
    assert cfg.cell_kind == "dec" and not cfg.dec_commit and cfg.loss_kind == "stablemax" and jax.default_backend() == "cpu"
    self.params = jax.tree.map(jnp.asarray, saved["state_ema"]["model"])
    self.tv = jnp.asarray(saved["state_ema"]["table"][0])
    self.eta, self.eta_z = map(float, M.eq_etas(self.params, cfg))
    assert self.eta_z == 1.0 and EV.coupled_ab(self.params, cfg) is None
    self.layout = cfg.sudoku_layout or "origin"
    self.z0 = DC.z0(cfg, 81)
    self.lm = np.asarray(self.params["dec"]["lm_head"], np.float64); assert self.lm.shape == (width,)
    self.embed = jax.jit(jax.vmap(lambda g: DC.embed_answer(self.params["dec"], cfg, g)))
    self.meta = dict(model=name, width=width, checkpoint_step=saved["step"], ema=True, checkpoint_sha256=ATS.sha(self.path),
                     source_checkpoint_sha256=inventory["source_checkpoint_sha256"], source_manifest_sha256=self.source_hash,
                     backend=jax.default_backend(), jax=jax.__version__, numpy=np.__version__, python=platform.python_version(),
                     operating_system=platform.platform(), process_id=os.getpid())
    self.out = out; self.dir = out / name; self.dir.mkdir(parents=True, exist_ok=True)
    ATS.write_json(self.dir / "runtime.json", self.meta)


def pool(name):
    """A fresh seeded uniform draw from the model's own record's unsolved-at-16 pool."""
    rec = np.load(ROOT / f"paper/code/evidence/benchmark/{name}_d16.npz", allow_pickle=False)
    cand = rec["idx"][~rec["cold_exact"]]
    return np.sort(np.random.default_rng([SEED, 21, seed_of(name)]).choice(cand, size=N_DRAW, replace=False))


class R13(R17.R11):
    def __init__(self, name, out=OUT):
        assert name in MODELS
        NAME["value"] = name; P.R2.__init__ = mlp_base_init
        super().__init__(WIDTH, out)
        self.name = name
        self.meta.update(study="rebuttal_20261003d", pipeline="tools/rebuttal_p21.py over tools/rebuttal_p17.py over tools/rebuttal_p15.py (base loader replaced in-process for the MLP checkpoint)")
        self.ATS.write_json(self.dir / "runtime.json", self.meta)

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
            T = np.where(trig)[0]; shift_max = 0.0; results = {}
            for arm in ARMS:
                ex = []; steps = STEPS_LONG if arm in LONG else STEPS2
                for c0 in range(0, len(T), Q.VBATCH):
                    tb = T[c0:c0 + Q.VBATCH]; pad = Q.VBATCH - len(tb); zz = z16[tb].copy(); xb = x[tb]
                    if arm == "A5_restart":
                        zz = np.stack([self.EV.mi_z0(SEED, int(ids[b]), 92, zz.shape[1:], 1.0, "gauss") for b in tb]).astype(np.float32)
                    elif arm.startswith("R"):
                        ang = float(arm[1:])
                        for q, b in enumerate(tb):
                            h = zz[q, 0]; h2 = self.rotate(h, ang, np.random.default_rng([SEED, int(ids[b]), 91, int(ang)]))
                            shift_max = max(shift_max, float(np.abs(np.einsum("fsw,w->sf", h2.astype(np.float64), self.lm) - np.einsum("fsw,w->sf", h.astype(np.float64), self.lm)).max()))
                            zz[q, 0] = h2
                    zp = np.concatenate([zz, np.repeat(zz[:1], pad, 0)]) if pad else zz; xp = np.concatenate([xb, np.repeat(xb[:1], pad, 0)]) if pad else xb
                    d, _ = self.loop(zp, xp, steps); ex.append((d[:, :len(tb)] == sol81[tb][None]).all(-1))
                    self.status("running", model=self.name, arm=arm, done=c0 + len(tb), of=len(T))
                results[arm] = np.concatenate(ex, 1) if ex else np.zeros((steps, 0), bool); self.ATS.log("arm_complete", model=self.name, arm=arm, of=len(T))
            assert shift_max < 1e-2, f"gate: a rotation changed a score by {shift_max}"
            dst = self.dir / ("p21_smoke.npz" if smoke else "p21.npz")
            np.savez_compressed(dst, ids=ids, triggered=trig, exact16=exact16, T=T, shift_max=shift_max, **{f"exact_{a}": results[a] for a in ARMS},
                                meta=json.dumps(dict(self.meta, created=utc(), n=len(ids), triggered=int(trig.sum()), smoke=smoke)))
            if not smoke: self.ATS.write_json(dst.with_suffix(".sha256.json"), dict(sha256=self.ATS.sha(dst)))
            print(json.dumps(dict(model=self.name, n=len(ids), triggered=int(trig.sum()), exact16_cpu=int(exact16.sum()), shift_max=shift_max)))
            self.ATS.verify_sources(); self.status("inference_complete", completed=utc()); self.ATS.log("model_complete", model=self.name, group="p21")
        except BaseException as exc:
            self.status("failed", error=repr(exc)); raise


def report(base=OUT, models=MODELS):
    lines = ["P21 REPORT (rules registered 2026-10-03, before any P20 result): transfer of the 60-degree hidden-state rotation to MLP 192 (three seeds); solved by iteration 32 among triggered puzzles"]
    out = {}; helps, beats, rescues, ahead64 = [], [], [], []
    for name in models:
        f = base / f"{name}/p21.npz"
        if not f.exists(): lines.append(f"{name}: not run"); continue
        d = np.load(f, allow_pickle=False); nT = len(d["T"]); A0 = d["exact_A0_intact"].astype(bool); R60 = d["exact_R60"].astype(bool)
        assert A0.shape[0] == STEPS_LONG and R60.shape[0] == STEPS_LONG
        trapped = ~A0[STEPS_LONG - 1]                                                   # not solved by iteration 64
        S = {a: d[f"exact_{a}"][15].astype(bool) for a in ARMS}; res = dict(triggered=nT, trapped=int(trapped.sum()), strata={})
        for sname, m in (("all", np.ones(nT, bool)), ("transient", ~trapped), ("trapped", trapped)):
            res["strata"][sname] = dict(solved={a: int(S[a][m].sum()) for a in ARMS}, losses={a: int((S["A0_intact"][m] & ~S[a][m]).sum()) for a in ARMS[1:]}, n=int(m.sum()))
        con = {}
        for x_, y_ in (("R60", "A0_intact"), ("R60", "A5_restart"), ("R45", "A0_intact"), ("R75", "A0_intact")):
            b = int((S[x_] & ~S[y_]).sum()); c = int((~S[x_] & S[y_]).sum()); con[f"{x_}_vs_{y_}"] = (b, c, R17.mcnemar_exact(b, c))
        h = con["R60_vs_A0_intact"]; r5 = con["R60_vs_A5_restart"]
        res["contrasts"] = con
        res["letter_helps"] = "ROTATION-HELPS" if (h[0] > h[1] and h[2] < P_MAX) else ("ROTATION-HURTS" if (h[1] > h[0] and h[2] < P_MAX) else "NO-DIFFERENCE")
        res["letter_restart"] = "BEATS-RESTART" if (r5[0] > r5[1] and r5[2] < P_MAX) else "NOT-SHOWN"
        rr = res["strata"]["trapped"]["solved"]["R60"] / max(res["strata"]["trapped"]["n"], 1); res["rescue_rate"] = rr
        long = {}                                                                        # exploratory: the long horizon
        for it in (32, 48, 64):
            k = it - STEPS1 - 1; b = int((R60[k] & ~A0[k]).sum()); c = int((~R60[k] & A0[k]).sum())
            long[str(it)] = dict(R60=int(R60[k].sum()), A0=int(A0[k].sum()), mcnemar=(b, c, R17.mcnemar_exact(b, c)))
        reach = np.where(A0.sum(1) >= int(R60[15].sum()))[0]
        long["A0_reaches_R60_at_32_by_iteration"] = int(reach[0] + STEPS1 + 1) if len(reach) else None
        res["long_horizon_exploratory"] = long; ahead64.append(long["64"]["R60"] > long["64"]["A0"])
        helps.append(res["letter_helps"]); beats.append(res["letter_restart"]); rescues.append(rr >= RESCUE_MIN); out[name] = res
        lines.append(f"{name}: triggered {nT}, trapped {int(trapped.sum())} | solved by 32 " + " ".join(f"{a} {res['strata']['all']['solved'][a]}" for a in ARMS)
                     + f" | R60 vs intact {h[0]}/{h[1]} p {h[2]:.3g} -> {res['letter_helps']} | R60 vs restart {r5[0]}/{r5[1]} p {r5[2]:.3g} -> {res['letter_restart']} | trapped rescued by R60 {rr:.2f}"
                     + " | transient " + " ".join(f"{a} {res['strata']['transient']['solved'][a]}" for a in ARMS) + " | trapped " + " ".join(f"{a} {res['strata']['trapped']['solved'][a]}" for a in ARMS)
                     + " | losses " + " ".join(f"{a} {v}" for a, v in res["strata"]["all"]["losses"].items())
                     + " | long horizon (exploratory) " + " ".join(f"@{it} R60 {long[str(it)]['R60']} A0 {long[str(it)]['A0']}" for it in (32, 48, 64))
                     + f"; A0 reaches R60's count at 32 by iteration {long['A0_reaches_R60_at_32_by_iteration']}")
    agg = dict(TRANSFERS=sum(1 for v in helps if v == "ROTATION-HELPS") >= MIN_RECEIVERS, BEATS_RESTART=sum(1 for v in beats if v == "BEATS-RESTART") >= MIN_RECEIVERS,
               TRAPPED_RESCUE=sum(rescues) >= MIN_RECEIVERS, R60_AHEAD_AT_64_exploratory=int(sum(ahead64)))
    out["aggregate"] = agg; lines.append(f"aggregate (at least {MIN_RECEIVERS} of 3 seeds): " + " ".join(f"{k} {v}" for k, v in agg.items()))
    (base / "report.txt").write_text("\n".join(lines) + "\n"); (base / "report.json").write_text(json.dumps(out, indent=1, default=float)); print("\n".join(lines))


def selftest():
    for name in MODELS:
        p = pool(name); assert len(p) == N_DRAW and np.all(np.diff(p) > 0) and np.array_equal(p, pool(name))
        rec = np.load(ROOT / f"paper/code/evidence/benchmark/{name}_d16.npz", allow_pickle=False)
        assert not np.isin(p, rec["idx"][rec["cold_exact"]]).any()
    assert seed_of("mlp_192_seed2") == 2 and STEPS1 + STEPS_LONG == 64 and STEPS1 + STEPS2 == 32
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", nargs="?", choices=["run", "report"]); ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model", choices=MODELS); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.selftest: selftest(); return
    if a.action == "report": report(Path(a.out) if a.out else OUT); return
    if a.action == "run":
        if a.model is None: ap.error("--model required")
        R13(a.model, Path(a.out) if a.out else OUT).run(a.smoke)
    else: ap.print_help()


if __name__ == "__main__":
    main()
