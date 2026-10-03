#!/usr/bin/env python3
# Ledger: THE DISCUSSION-PERIOD EXPERIMENT P24 — is the single-state model's stall equally memoryless and chaotic? P22's protocol on
# SA256U (the one-carry recurrence, selected 14k) and SA256 (two states, its selected 28k and the matched 14k) (registration
# Documentation/Note_2026-10-03_Rebuttal_P24_Registration.md, written before any row; receivers offered by session d378a5, the PI's go
# 2026-10-03). MEASUREMENT, $0, inference only. Uses the MAIN repo's src/qhrrn2 and tools/eval_sudoku_extreme.py (the release code has no
# single-state branch) with the evaluator's own loader recipe; imports P22's statistics, letters and reader helpers unchanged.
"""P24: per receiver, a seeded draw of test puzzles unsolved at 16 in a saved record (SA256U: its own 5,000-puzzle record; both SA256
checkpoints: one shared draw from SA256's full-test record at 46k, the manuscript's attention-256 run, excluding P17/P20/P22's pools);
the trigger is an invalid displayed grid at iteration 16 on the CPU (the first 80 triggered kept); from that state: A0 intact (32
iterations, with per-iteration display changes recorded), eight 60-degree score-preserving re-rolls (32 iterations), four 1-degree and
four 0.1-degree re-rolls (16 iterations). Rules: P22's A (memory) and B (chaos) per receiver; the pattern across receivers decides the
registered sentence."""
import argparse, hashlib, json, math, os, platform, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]              # the MAIN repo first: its qhrrn2 carries the single-state branch
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import rebuttal_p1p2 as P                                             # project() only (no loader is instantiated)
import rebuttal_p17 as R17                                            # invalid(), mcnemar_exact(), DATA
import rebuttal_p20 as R20
import rebuttal_p22 as R22                                            # statistics, letters, pools

OUT = ROOT / "runs/analysis/rebuttal_20261003g"
SEED = 20261006
CKPT = {"SA256U_14k": ROOT / "runs/_p13_ckpts/SA256U_ckpt_014000.pkl", "SA256_28k": ROOT / "runs/_p13_ckpts/SA256_ckpt_028000.pkl",
        "SA256_14k": ROOT / "runs/_p13_ckpts/SA256_ckpt_014000.pkl"}
SHA_PREFIX = {"SA256U_14k": "486f5523", "SA256_28k": "18e9427f", "SA256_14k": "b98c802e"}       # as reported by session d378a5
REC_U = ROOT / "runs/_attr2_pull/stage/runs/sxeval_pchampSA256U/sub5k_vsel_t16/records_all.npz"
REC_S46 = ROOT / "runs/_saext_pull/stage/runs/filler_sxeval_pchampSA256_full_t64/records_all.npz"
RECEIVERS = tuple(CKPT)
N_DRAW, T_CAP, VBATCH = 160, 80, 128
STEPS1, STEPS_LONG, STEPS_SHORT = 16, 32, 16
K_BIG, K_SMALL, BIG, SMALL, TINY = 8, 4, 60.0, 1.0, 0.1


def utc(): return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(2 ** 20), b""): h.update(b)
    return h.hexdigest()


def pool(receiver):
    """SA256U: a seeded draw from its own 5,000-puzzle record's unsolved-at-16 puzzles. SA256 (both checkpoints): one shared seeded draw
    from the full-test record at 46k (first exact never or after iteration 16), excluding P17/P20/P22's attention-256 pools."""
    if receiver == "SA256U_14k":
        r = np.load(REC_U, allow_pickle=False); cand = np.sort(r["idx"][~r["cold_exact"].astype(bool)])
        return np.sort(np.random.default_rng([SEED, 24, 0]).choice(cand, size=N_DRAW, replace=False))
    r = np.load(REC_S46, allow_pickle=False); fe = r["first_exact"]
    excl = set(int(i) for i in R17.pool(256)) | set(int(i) for i in R20.pool(256)) | set(int(i) for i in R22.pool(256))
    cand = np.sort(np.array([int(i) for i in r["idx"][(fe < 0) | (fe > 15)] if int(i) not in excl]))
    return np.sort(np.random.default_rng([SEED, 24, 1]).choice(cand, size=N_DRAW, replace=False))


class R16:
    """A standalone runner on the MAIN repo: the evaluator's loader recipe, its step, and a loop that mirrors run_batch."""
    def __init__(self, receiver, out=OUT):
        import jax, jax.numpy as jnp
        import qhrrn2, eval_sudoku_extreme as EV
        from qhrrn2 import episodic as E, model as M, grid as G
        from qhrrn2.config import Config
        assert Path(qhrrn2.__file__).resolve().is_relative_to(ROOT / "src") and Path(EV.__file__).resolve().is_relative_to(ROOT / "tools"), "not the main repo"
        assert jax.default_backend() == "cpu"
        self.jax, self.jnp, self.EV, self.M, self.G = jax, jnp, EV, M, G
        self.receiver = receiver; path = CKPT[receiver]; digest = sha(path)
        assert digest.startswith(SHA_PREFIX[receiver]), f"gate: {path.name} sha256 {digest[:8]} is not the one reported"
        saved = E.load_ckpt(path); d0 = Config()
        self.cfg = cfg = Config(**{k: type(getattr(d0, k))(v) for k, v in saved["config"].items()})          # eval_sudoku_extreme.py's recipe
        assert cfg.cell_kind == "dec" and cfg.dec_width == 256 and not cfg.dec_commit and cfg.loss_kind == "stablemax"
        assert bool(getattr(cfg, "dec_single_state", False)) == (receiver == "SA256U_14k")
        self.params = jax.tree.map(jnp.asarray, saved["state_ema"]["model"]); self.tv = jnp.asarray(saved["state_ema"]["table"][0])
        self.eta, self.eta_z = map(float, M.eq_etas(self.params, cfg)); assert self.eta == 1.0 and self.eta_z == 1.0
        assert EV.coupled_ab(self.params, cfg) is None
        self.layout = cfg.sudoku_layout or "origin"
        self.lm = np.asarray(self.params["dec"]["lm_head"], np.float64); assert self.lm.shape == (256,)
        self.step_first = EV._step(cfg, 1.0, 0.0, True); self.step_rel = EV._step(cfg, 1.0, 0.0, False)
        self.void = jax.nn.one_hot(jnp.full((9, 9), G.VOID, jnp.int32), 11).transpose(2, 0, 1)
        self.meta = dict(receiver=receiver, checkpoint=str(path.relative_to(ROOT)), checkpoint_step=int(saved["step"]), ema=True, checkpoint_sha256=digest,
                         single_state=bool(getattr(cfg, "dec_single_state", False)), code="main repo src/qhrrn2 + tools/eval_sudoku_extreme.py",
                         backend="cpu", jax=jax.__version__, numpy=np.__version__, python=platform.python_version(), operating_system=platform.platform(),
                         process_id=os.getpid(), study="rebuttal_20261003g", pipeline="tools/rebuttal_p24.py (P22's helpers)")
        self.dir = out / receiver; self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "runtime.json").write_text(json.dumps(self.meta, indent=2) + "\n")

    def status(self, state, **values):
        (self.dir / "status.json").write_text(json.dumps(dict(state=state, updated=utc(), pid=os.getpid(), receiver=self.receiver, **values)) + "\n")

    def log(self, event, **values): print(json.dumps(dict(time=utc(), event=event, **values)), flush=True)

    def loop(self, z, x, steps, first=False):
        """run_batch's loop (carry z <- z + eta_z (zf - z); canvas y <- y + eta (p - y)); displays (steps, B, 81) by the digit argmax."""
        jnp, jax = self.jnp, self.jax; x = jnp.asarray(x); y = jnp.broadcast_to(self.void, (x.shape[0],) + self.void.shape); disp = []
        zc = None if first else jnp.asarray(z)
        for t in range(steps):
            f = zc is None
            logits, zf = (self.step_first if f else self.step_rel)(self.params, x, y, self.tv, jnp.zeros(1) if f else zc)
            zc = zf if f else zc + self.eta_z * (zf - zc)
            p = jax.nn.softmax(logits, axis=-1).transpose(0, 3, 1, 2); y = y + self.eta * (p - y)
            d = (np.asarray(self.EV.layout_gather(logits, self.layout), np.float32)[..., 1:10].reshape(x.shape[0], 81, 9).argmax(-1) + 1).astype(np.int8)
            disp.append(d)
        return np.stack(disp), np.array(zc)

    def rotate(self, h, theta, rng):
        import rebuttal_p3 as P3
        h = h.astype(np.float64); par, perp = P.project(h, self.lm)
        return (par + P3.rotate_perp(perp, self.lm, theta, rng)).astype(np.float32)

    def gate_run_batch(self, x, puz, sol):
        """G1: the loop's exact flags equal the main evaluator's run_batch flags bitwise (16 iterations, fixed start, one batch)."""
        y0 = self.jnp.broadcast_to(self.void, (x.shape[0],) + self.void.shape)
        ex_rb, *_ = self.EV.run_batch(self.params, self.cfg, self.tv, self.jnp.asarray(x), y0, t_total=STEPS1, tau=1.0, gamma=1.0, sol9=sol, puz9=puz,
                                      eta=self.eta, eta_z=self.eta_z, layout=self.layout)
        d, _ = self.loop(None, x, STEPS1, first=True); ex = (d == sol.reshape(-1, 81)[None]).all(-1)
        assert np.array_equal(np.asarray(ex_rb).astype(bool), ex), "gate G1: the loop does not reproduce run_batch's exact flags"
        return int(ex[-1].sum())

    def rerolls(self, z16, x, sol81, ids, T, angle, K, salt, steps, tag):
        pairs = [(k, j) for k in range(K) for j in range(len(T))]; out = np.zeros((K, steps, len(T)), bool); shift = 0.0; ang_dev = 0.0
        for c0 in range(0, len(pairs), VBATCH):
            chunk = pairs[c0:c0 + VBATCH]; pad = VBATCH - len(chunk)
            zz = np.stack([z16[T[j]] for _, j in chunk]).copy(); xb = np.stack([x[T[j]] for _, j in chunk])
            for q, (k, j) in enumerate(chunk):
                h = zz[q, 0]; h2 = self.rotate(h, angle, np.random.default_rng([SEED, int(ids[T[j]]), salt, k]))
                shift = max(shift, float(np.abs(np.einsum("fsw,w->sf", h2.astype(np.float64), self.lm) - np.einsum("fsw,w->sf", h.astype(np.float64), self.lm)).max()))
                po = P.project(h.astype(np.float64), self.lm)[1]; pn = P.project(h2.astype(np.float64), self.lm)[1]
                c = (po * pn).sum(-1) / np.maximum(np.linalg.norm(po, axis=-1) * np.linalg.norm(pn, axis=-1), 1e-30)
                ang_dev = max(ang_dev, float(np.abs(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))) - angle).max() / angle))
                zz[q, 0] = h2
            zp = np.concatenate([zz, np.repeat(zz[:1], pad, 0)]) if pad else zz; xp = np.concatenate([xb, np.repeat(xb[:1], pad, 0)]) if pad else xb
            d, _ = self.loop(zp, xp, steps)
            for q, (k, j) in enumerate(chunk): out[k, :, j] = (d[:, q] == sol81[T[j]][None]).all(-1)
            self.status("running", arm=tag, done=c0 + len(chunk), of=len(pairs))
        return out, shift, ang_dev

    def run(self, smoke=False):
        try:
            ids = pool(self.receiver)
            if smoke: ids = ids[:16]
            with np.load(R17.DATA, allow_pickle=False) as d: puz, sol = d["test_q"][ids].astype(np.int32), d["test_a"][ids].astype(np.int32)
            x = np.asarray(self.EV.place_batch(puz, self.layout)); sol81 = sol.reshape(-1, 81); puz81 = puz.reshape(-1, 81)
            pad0 = VBATCH - min(VBATCH, len(ids)); xg = np.concatenate([x[:VBATCH], np.repeat(x[:1], pad0, 0)]) if pad0 else x[:VBATCH]
            pg = np.concatenate([puz[:VBATCH], np.repeat(puz[:1], pad0, 0)]) if pad0 else puz[:VBATCH]; sg = np.concatenate([sol[:VBATCH], np.repeat(sol[:1], pad0, 0)]) if pad0 else sol[:VBATCH]
            g1 = self.gate_run_batch(xg, pg, sg); self.log("gate_G1_passed", receiver=self.receiver)
            disp1, z16 = [], []
            for b0 in range(0, len(ids), VBATCH):
                xb = x[b0:b0 + VBATCH]; pad = VBATCH - len(xb); xp = np.concatenate([xb, np.repeat(xb[:1], pad, 0)]) if pad else xb
                d, z = self.loop(None, xp, STEPS1, first=True); disp1.append(d[:, :len(xb)]); z16.append(z[:len(xb)])
            disp1 = np.concatenate(disp1, 1); z16 = np.concatenate(z16, 0)
            trig = R17.invalid(disp1[-1], puz81); exact16 = (disp1[-1] == sol81).all(1)
            assert not (trig & exact16).any() and np.all(trig | exact16), "trigger is not equivalent to CPU-unsolved"
            T = np.where(trig)[0][:T_CAP]
            ex0, ch0 = [], []
            for c0 in range(0, len(T), VBATCH):
                tb = T[c0:c0 + VBATCH]; pad = VBATCH - len(tb); zz = z16[tb].copy(); xb = x[tb]
                zp = np.concatenate([zz, np.repeat(zz[:1], pad, 0)]) if pad else zz; xp = np.concatenate([xb, np.repeat(xb[:1], pad, 0)]) if pad else xb
                d, _ = self.loop(zp, xp, STEPS_LONG); d = d[:, :len(tb)]
                ex0.append((d == sol81[tb][None]).all(-1)); prev = np.concatenate([disp1[-1][tb][None], d[:-1]], 0); ch0.append((d != prev).sum(-1).astype(np.int16))
                self.status("running", arm="A0_intact", done=c0 + len(tb), of=len(T))
            exA0 = np.concatenate(ex0, 1) if ex0 else np.zeros((STEPS_LONG, 0), bool); chA0 = np.concatenate(ch0, 1) if ch0 else np.zeros((STEPS_LONG, 0), np.int16)
            self.log("arm_complete", receiver=self.receiver, arm="A0_intact", of=len(T))
            exBig, sB, cB = self.rerolls(z16, x, sol81, ids, T, BIG, K_BIG, 101, STEPS_LONG, "R60_rerolls"); self.log("arm_complete", receiver=self.receiver, arm="R60_rerolls", of=len(T))
            exSmall, sS, cS = self.rerolls(z16, x, sol81, ids, T, SMALL, K_SMALL, 102, STEPS_SHORT, "R1_rerolls"); self.log("arm_complete", receiver=self.receiver, arm="R1_rerolls", of=len(T))
            exTiny, sT, cT = self.rerolls(z16, x, sol81, ids, T, TINY, K_SMALL, 103, STEPS_SHORT, "R01_rerolls"); self.log("arm_complete", receiver=self.receiver, arm="R01_rerolls", of=len(T))
            assert max(sB, sS, sT) < 1e-2, f"gate: a rotation changed a score by {max(sB, sS, sT)}"
            assert max(cB, cS, cT) < 0.01, f"gate: a realized rotation angle differs from its nominal angle by {max(cB, cS, cT):.2%}"
            dst = self.dir / ("p24_smoke.npz" if smoke else "p24.npz")
            np.savez_compressed(dst, ids=ids, triggered=trig, exact16=exact16, T=T, exact_A0=exA0, changes_A0=chA0, exact_R60=exBig, exact_R1=exSmall, exact_R01=exTiny,
                                shift_max=max(sB, sS, sT), angle_dev_max=max(cB, cS, cT), meta=json.dumps(dict(self.meta, created=utc(), n=len(ids), triggered=int(trig.sum()), kept=int(len(T)), smoke=smoke)))
            if not smoke: (dst.with_suffix(".sha256.json")).write_text(json.dumps(dict(sha256=sha(dst))) + "\n")
            print(json.dumps(dict(receiver=self.receiver, n=len(ids), triggered=int(trig.sum()), kept=int(len(T)), shift_max=max(sB, sS, sT), angle_dev_max=max(cB, cS, cT))))
            self.status("inference_complete", completed=utc()); self.log("model_complete", receiver=self.receiver, group="p24")
        except BaseException as exc:
            self.status("failed", error=repr(exc)); raise


def report(base=OUT, receivers=RECEIVERS):
    lines = ["P24 REPORT (rules registered 2026-10-03, before any P24 row): P22's memory and chaos rules on the single-state model and its two-state control"]
    out = {}
    for rc in receivers:
        f = base / f"{rc}/p24.npz"
        if not f.exists(): lines.append(f"{rc}: not run"); continue
        d = np.load(f, allow_pickle=False); nT = len(d["T"]); A0 = d["exact_A0"].astype(bool); B = d["exact_R60"].astype(bool); S = d["exact_R1"].astype(bool); Tn = d["exact_R01"].astype(bool)
        unsolve = int((A0[:-1] & ~A0[1:]).sum() + (B[:, :-1] & ~B[:, 1:]).sum() + (S[:, :-1] & ~S[:, 1:]).sum() + (Tn[:, :-1] & ~Tn[:, 1:]).sum())
        firsts = [R22.first_hits(B[:, :, j].T) for j in range(nT)]
        hr = R22.memory_ratio(firsts); lo, hi = R22.boot_ci(firsts, [SEED, 24, RECEIVERS.index(rc), 9])
        dE = sum(R22.window_counts(f_, *R22.E_WIN)[0] for f_ in firsts); dL = sum(R22.window_counts(f_, *R22.L_WIN)[0] for f_ in firsts)
        Db = R22.disagreement(B[:K_SMALL, 15, :]); Ds = R22.disagreement(S[:, 15, :]); ratio = Ds / Db if Db > 0 else float("nan")
        Dt = R22.disagreement(Tn[:, 15, :]); ratio_tiny = Dt / Db if Db > 0 else float("nan")
        rng = np.random.default_rng([SEED, 24, RECEIVERS.index(rc), 10]); bs = []
        for _ in range(R22.N_BOOT):
            idx = rng.integers(0, nT, nT); db = R22.disagreement(B[:K_SMALL, 15, idx]); bs.append(R22.disagreement(S[:, 15, idx]) / db if db > 0 else np.nan)
        bs = np.array(bs); bs = bs[np.isfinite(bs)]; rci = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))) if len(bs) else (float("nan"),) * 2
        ch = d["changes_A0"].astype(int); stalled = ~A0                                                           # display changes on not-yet-solved intact steps
        frozen_frac = float((ch[stalled] == 0).mean()) if stalled.any() else float("nan"); med_change = float(np.median(ch[stalled])) if stalled.any() else float("nan")
        q = B[:, 15, :].mean(0)
        res = dict(triggered=nT, unsolve_transitions=unsolve, A=dict(hr=hr, ci=(lo, hi), events_early=dE, events_late=dL, letter=R22.letter_a(hr, lo, hi)),
                   B=dict(D_small=Ds, D_big=Db, ratio=ratio, ci=rci, letter=R22.letter_b(ratio)), B_tiny_exploratory=dict(D_tiny=Dt, ratio=ratio_tiny),
                   exploratory=dict(A0_solved_16=int(A0[15].sum()), mean_reroll_q=float(q.mean() * nT), A0_solved_32=int(A0[31].sum()), memoryless_prediction_A0_32=float((1 - (1 - q) ** 2).sum()),
                                    stalled_steps_frozen_fraction=frozen_frac, stalled_steps_median_cells_changed=med_change))
        out[rc] = res
        lines.append(f"{rc}: kept {nT} | A: R {hr:.3f} [{lo:.3f}, {hi:.3f}] (events {dE} early, {dL} late) -> {res['A']['letter']} | B: D 1deg {Ds:.3f} vs 60deg {Db:.3f}, ratio {ratio:.3f} [{rci[0]:.3f}, {rci[1]:.3f}] -> {res['B']['letter']} "
                     f"(0.1deg {Dt:.3f}, ratio {ratio_tiny:.3f}) | solved->unsolved {unsolve} | exploratory: stalled intact steps frozen {frozen_frac:.3f}, median cells changed {med_change:.0f}; A0 by +16 {int(A0[15].sum())} vs expected {q.mean() * nT:.1f}; A0 by +32 {int(A0[31].sum())} vs memoryless prediction {(1 - (1 - q) ** 2).sum():.1f}")
    if all(rc in out for rc in receivers):
        L = {rc: (out[rc]["A"]["letter"], out[rc]["B"]["letter"]) for rc in receivers}
        if L["SA256U_14k"] == ("MEMORYLESS", "CHAOTIC") and L["SA256_28k"] == ("MEMORYLESS", "CHAOTIC"): pattern = "SAME-TAIL"
        elif L["SA256U_14k"] != L["SA256_28k"] and L["SA256U_14k"] != L["SA256_14k"]: pattern = "SINGLE-STATE-DIFFERS"
        else: pattern = "MIXED"
        out["pattern"] = pattern; lines.append(f"pattern across receivers: {pattern} ({'; '.join(f'{rc} {a}/{b}' for rc, (a, b) in L.items())})")
    (base / "report.txt").write_text("\n".join(lines) + "\n"); (base / "report.json").write_text(json.dumps(out, indent=1, default=float)); print("\n".join(lines))


def selftest():
    for rc in RECEIVERS:
        p = pool(rc); assert len(p) == N_DRAW and np.all(np.diff(p) > 0) and np.array_equal(p, pool(rc))
    assert np.array_equal(pool("SA256_28k"), pool("SA256_14k"))                      # one shared draw for the two SA256 checkpoints
    s = set(int(i) for i in pool("SA256_28k")); assert not (s & (set(int(i) for i in R17.pool(256)) | set(int(i) for i in R20.pool(256)) | set(int(i) for i in R22.pool(256))))
    r = np.load(REC_U, allow_pickle=False); assert not np.isin(pool("SA256U_14k"), r["idx"][r["cold_exact"].astype(bool)]).any()
    for rc, path in CKPT.items(): assert path.exists(), path
    R22.selftest()
    print("selftest OK (P24)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", nargs="?", choices=["run", "report"]); ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--receiver", choices=RECEIVERS); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.selftest: selftest(); return
    if a.action == "report": report(Path(a.out) if a.out else OUT); return
    if a.action == "run":
        if a.receiver is None: ap.error("--receiver required")
        R16(a.receiver, Path(a.out) if a.out else OUT).run(a.smoke)
    else: ap.print_help()


if __name__ == "__main__":
    main()
