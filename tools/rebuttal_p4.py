#!/usr/bin/env python3
# Ledger: THE DISCUSSION-PERIOD EXPERIMENT P4 — the freeze-time ladder (registration Documentation/Note_2026-10-01_Rebuttal_P4_Registration.md,
# written before this build). MEASUREMENT, $0, inference only on the banked attention checkpoints; subclass of the P2b runner (tools/rebuttal_p2b.py,
# unchanged); the release files are untouched (the block is patched in-process by the P2b runner).
"""P4: messages frozen FROM iteration t0 (the iteration-(t0-1) messages replayed at t0..16) for t0 in {2, 4, 6, 8} on the 256 shared first states.

t0 = 2 is P2b's messages_frozen and must reproduce it bitwise (gate). Rule: F(t0) = late discoveries completed under the freeze / late
discoveries under intact; EXCHANGE-UNTIL-COMPLETION if F <= .25 for t0 in {4, 6, 8} on all widths; LATE-EXCHANGE-DISPENSABLE if F(8) >= .75; MIXED.
"""
import argparse, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import rebuttal_p2b as Q                                    # the P2b runner (hook, step functions), unchanged
import rebuttal_p1p2 as P

OUT = ROOT / "runs/analysis/rebuttal_20261001b"
P2B_OUT = ROOT / "runs/analysis/rebuttal_20260926c"
T0S = (2, 4, 6, 8)
CONDS = tuple(f"frozen_from_t{t}" for t in T0S)
WIDTHS, BATCH, STEPS = P.WIDTHS, Q.BATCH, Q.STEPS
F_LOW, F_HIGH, L_MIN, LETTER_T0 = 0.25, 0.75, 10, (4, 6, 8)


def utc(): return datetime.now(timezone.utc).isoformat()


def t0_of(cond): return int(cond.rsplit("t", 1)[1])


def letter(F):
    """F: {width: {t0: F or None}} (None = UNDEFINED there). Letter over t0 in LETTER_T0 on all widths; excluded cells stated by the caller."""
    vals = [F[w][t] for w in F for t in LETTER_T0 if F[w].get(t) is not None]
    if not vals: return "UNDEFINED"
    if all(v <= F_LOW for v in vals): return "EXCHANGE-UNTIL-COMPLETION"
    f8 = [F[w][8] for w in F if F[w].get(8) is not None]
    if f8 and len(f8) == len(F) and all(v >= F_HIGH for v in f8): return "LATE-EXCHANGE-DISPENSABLE"
    return "MIXED"


class R6(Q.R4):
    def __init__(self, width, out=OUT):
        super().__init__(width, out)
        self.meta.update(study="rebuttal_20261001b", pipeline="tools/rebuttal_p4.py over tools/rebuttal_p2b.py over tools/rebuttal_p1p2.py")
        self.ATS.write_json(self.dir / "runtime.json", self.meta)

    def run_freeze(self, puz, t0, steps=STEPS, progress=None, record_check=False):
        """Iterations 1..t0-1 intact (hook off), the messages of iteration t0-1 recorded, then replayed at t0..steps.
        Returns (values, bank, record_check_max_abs)."""
        jax, jnp, EV = self.jax, self.jnp, self.EV
        x = EV.place_batch(puz, self.layout)
        void = jax.nn.one_hot(jnp.full((9, 9), self.G.VOID, jnp.int32), 11).transpose(2, 0, 1); y = jnp.broadcast_to(void, (len(puz),) + void.shape)
        z = None; logits, preds = [], []; bank = None; check = None
        for t in range(steps):                               # t is zero-based; iteration t+1
            it = t + 1; first = (t == 0)
            if it < t0 - 1 or (t0 <= 1):
                mode, b, bb = "off", jnp.zeros(1), False
            elif it == t0 - 1:
                mode, b, bb = "record", jnp.zeros(1), False
            else:
                mode, b, bb = "replay", bank, True
            y_in, z_in = y, z
            y, z, lg, pr, extra = self.one_step(x, y, z, first, mode, b, bb)
            if mode == "record":
                bank = extra
                if record_check:                              # replaying the fresh record at the same step reproduces it (gate 3)
                    _, _, lg2, _, _ = self.one_step(x, y_in, z_in, first, "replay", bank, True)
                    check = float(np.abs(lg2 - lg).max())
            logits.append(lg); preds.append(pr)
            if progress and it % 4 == 0: self.status("running", **progress, iteration=it)
        return dict(logits=np.stack(logits), pred=np.stack(preds)), bank, check

    def gate(self, smoke):
        n = 16 if smoke else BATCH; steps = 9 if smoke else STEPS
        with np.load(self.STUDY / f"attention_{self.width}/interventions/intact/batch_0000.npz", allow_pickle=False) as d:
            ids, puz, ref = d["ids"][:n], d["puz"][:n], d["logits"][:steps, :n]
        # (1) hook off reproduces the study's intact continuation bitwise
        off, _, _ = self.run_freeze(puz, t0=99, steps=steps)
        g1 = bool(np.array_equal(off["logits"], ref)); d1 = float(np.abs(off["logits"] - ref).max())
        assert g1 or d1 < 1e-4, f"hook-off path differs from the study by {d1}"
        # (2) t0 = 2 reproduces P2b's messages_frozen chunk 0 bitwise
        g2 = None; d2 = None
        p2b = P2B_OUT / f"attention_{self.width}/state/messages_frozen/batch_0000.npz"
        if p2b.exists():
            with np.load(p2b, allow_pickle=False) as d: refb, idsb = d["logits"][:steps, :n], d["ids"][:n]
            assert np.array_equal(idsb, ids)
            fr2, _, _ = self.run_freeze(puz, t0=2, steps=steps)
            g2 = bool(np.array_equal(fr2["logits"], refb)); d2 = float(np.abs(fr2["logits"] - refb).max())
            assert g2 or d2 < 1e-4, f"t0=2 differs from P2b's frozen chunk by {d2}"
        # (3) the replay path is exact at a later step (t0 = 4: the iteration-3 record replayed at iteration 3)
        _, _, d3 = self.run_freeze(puz, t0=4, steps=4, record_check=True)
        g3 = (d3 == 0.0)
        assert g3 or d3 < 1e-4, f"later-step replay differs from its record by {d3}"
        info = dict(passed=True, created=utc(), n=n, steps=steps, off_vs_study_bitwise=g1, off_vs_study_max_abs=d1,
                    t2_vs_p2b_bitwise=g2, t2_vs_p2b_max_abs=d2, later_replay_vs_record_bitwise=g3, later_replay_vs_record_max_abs=d3,
                    messages_per_iteration=self.n_msgs)
        self.ATS.write_json(self.dir / ("gate_smoke.json" if smoke else "gate.json"), info); self.ATS.log("gate_passed", width=self.width, **info)

    def run(self, smoke):
        try:
            self.gate(smoke)
            with np.load(self.STUDY / "inputs/interventions.npz", allow_pickle=False) as d: ids, puz, sol = d["ids"], d["puz"], d["sol"]
            if smoke: ids, puz, sol = ids[:16], puz[:16], sol[:16]
            steps = 9 if smoke else STEPS; group = "state"
            for b in range(0, len(ids), BATCH):
                sl = slice(b, b + BATCH); ii, pp, ss = ids[sl], puz[sl], sol[sl]
                for c in CONDS:
                    path = self.dir / group / c / f"batch_{b:04d}.npz"
                    if self.cached(path, ii, group, c): continue
                    started, cpu = time.monotonic(), time.process_time()
                    values, _, _ = self.run_freeze(pp, t0_of(c), steps=steps, progress=dict(group=group, condition=c, batch_start=b))
                    if not smoke:
                        with np.load(self.STUDY / f"attention_{self.width}/interventions/intact/batch_{(b // 128) * 128:04d}.npz", allow_pickle=False) as d:
                            off = b % 128; k = t0_of(c) - 1
                            assert np.array_equal(d["logits"][:k, off:off + len(ii)], values["logits"][:k]), "pre-freeze iterations differ from the study"
                    self.save(path, values, pp, ss, ii, group, c, started, cpu, dict(t0=t0_of(c), messages_per_iteration=self.n_msgs, batch_size=BATCH))
            self.ATS.verify_sources(); self.status("inference_complete", completed=utc()); self.ATS.log("model_complete", width=self.width, group="p4")
        except BaseException as exc:
            self.status("failed", error=repr(exc)); raise


def load_exact(d):
    parts = sorted(Path(d).glob("batch_*.npz")); ex, pred, ids, sol = [], [], [], []
    for pth in parts:
        with np.load(pth, allow_pickle=False) as z: ex.append(z["exact"]); pred.append(z["pred"]); ids.append(z["ids"]); sol.append(z["sol"])
    if not parts: return None
    return dict(exact=np.concatenate(ex, 1), pred=np.concatenate(pred, 1), ids=np.concatenate(ids), sol=np.concatenate(sol))


def report(base=OUT):
    import attention_transfer_study as ATS
    lines = ["P4 REPORT (rules registered 2026-10-01): messages frozen from iteration t0 on the 256 shared first states"]; out = {}; F = {}
    for w in WIDTHS:
        intact = load_exact(ATS.OUT / f"attention_{w}/interventions/intact")
        if intact is None: continue
        exI = intact["exact"]; tauI = np.where(exI.any(0), exI.argmax(0) + 1, 0); sol = intact["sol"]
        res = {}; F[w] = {}
        for c in CONDS:
            d = load_exact(base / f"attention_{w}/state/{c}")
            if d is None: continue
            assert np.array_equal(d["ids"], intact["ids"]); t0 = t0_of(c); ex = d["exact"]
            assert np.array_equal(ex[:t0 - 1], exI[:t0 - 1])
            before = exI[t0 - 2] if t0 >= 2 else np.zeros(ex.shape[1], bool)
            late = (~before) & exI[-1]; L = int(late.sum()); D = int((late & ex[-1]).sum())
            f = (D / L) if L >= L_MIN else None; F[w][t0] = f
            lost = int((before & ~ex[-1]).sum()); kept_n = int(before.sum())
            wrong_before = (intact["pred"][t0 - 2] != sol).reshape(ex.shape[1], -1).sum(1) if t0 >= 2 else None
            disc = late & ex[-1]
            tau_rel = (tauI[disc] - t0).tolist() if disc.any() else []
            wb = np.sort(wrong_before[disc]).tolist() if disc.any() else []
            p = d["pred"]; cons = [float((p[t] != p[t - 1]).reshape(ex.shape[1], -1).sum(1).mean()) for t in range(t0 - 1, 16)]
            res[c] = dict(t0=t0, exact16=int(ex[-1].sum()), intact16=int(exI[-1].sum()), late=L, discovered=D, F=f, before_exact=kept_n, lost=lost,
                          tau_minus_t0_of_discoveries=tau_rel, wrong_cells_before_of_discoveries=wb[:40], settling_cells_changed=cons)
            lines.append(f"attention_{w} {c}: exact16 {int(ex[-1].sum())} (intact {int(exI[-1].sum())}) | late L {L} discovered D {D} F {('%.3f' % f) if f is not None else 'UNDEFINED'}"
                         f" | solved before t0 {kept_n}, lost by 16 {lost} | tau_intact - t0 of discoveries: {sorted(tau_rel)[:30]} | wrong cells before t0 of discoveries: {wb[:30]}")
        out[f"attention_{w}"] = res
    Lt = letter(F) if F else "NOT RUN"
    excluded = [(w, t) for w in F for t in LETTER_T0 if F[w].get(t) is None]
    lines.append(f"letter: {Lt}" + (f" (UNDEFINED cells excluded: {excluded})" if excluded else ""))
    out["letter"] = Lt; out["excluded"] = excluded
    (base / "report.txt").write_text("\n".join(lines) + "\n"); (base / "report.json").write_text(json.dumps(out, indent=1, default=float))
    print("\n".join(lines))


def selftest():
    assert [t0_of(c) for c in CONDS] == list(T0S)
    assert letter({128: {4: .1, 6: .2, 8: .25}, 192: {4: .0, 6: .1, 8: .2}}) == "EXCHANGE-UNTIL-COMPLETION"
    assert letter({128: {4: .5, 6: .8, 8: .9}, 192: {4: .6, 6: .8, 8: .8}}) == "LATE-EXCHANGE-DISPENSABLE"
    assert letter({128: {4: .1, 6: .2, 8: .9}, 192: {4: .0, 6: .1, 8: .2}}) == "MIXED"
    assert letter({128: {4: .1, 6: .2, 8: None}, 192: {4: .0, 6: .1, 8: .2}}) == "EXCHANGE-UNTIL-COMPLETION"
    assert letter({128: {4: .9, 6: .9, 8: None}, 192: {4: .9, 6: .9, 8: .9}}) == "MIXED"       # F(8) missing on a width cannot give DISPENSABLE
    assert letter({128: {}}) == "UNDEFINED"
    # the late set / F arithmetic on a toy record
    exI = np.array([[0, 0, 0, 1], [0, 1, 0, 1], [1, 1, 0, 1]], bool); ex = np.array([[0, 0, 0, 1], [0, 1, 0, 1], [0, 1, 0, 1]], bool); t0 = 3
    before = exI[t0 - 2]; late = (~before) & exI[-1]; assert late.tolist() == [True, False, False, False]
    assert int((late & ex[-1]).sum()) == 0 and int((before & ~ex[-1]).sum()) == 0
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", nargs="?", choices=["run", "report"])
    ap.add_argument("--selftest", action="store_true"); ap.add_argument("--width", type=int, choices=WIDTHS)
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.selftest: selftest(); return
    out = Path(a.out) if a.out else OUT
    if a.action == "run":
        if a.width is None: ap.error("--width required")
        R6(a.width, out).run(a.smoke)
    elif a.action == "report": report(out)
    else: ap.print_help()


if __name__ == "__main__":
    main()
