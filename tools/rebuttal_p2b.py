#!/usr/bin/env python3
# Ledger: THE REVIEW-PERIOD EXPERIMENT P2b — message controls that keep clue access (registration
# Documentation/Note_2026-09-26_Review_Period_P1c_P2b_Registration.md, written before this build). MEASUREMENT, $0, inference only.
#   On the paper's 256 shared first states, iterations 2-16 run with every cross-field message tensor (42 per iteration: 21 stack
#   applications x 2 blocks) replaced by (a) the same puzzle's tensor recorded at iteration 1 (`messages_frozen`), or (b) the iteration-1
#   mean over the 256 puzzles (`messages_mean`). The release cell's block is patched in-process for a fresh trace; no released file is
#   edited; the record path is gated bitwise against the study's intact first step and the replay path against the record.
"""  .venv/bin/python tools/rebuttal_p2b.py --selftest
  JAX_PLATFORMS=cpu .venv/bin/python tools/rebuttal_p2b.py run --width 128|192|256 [--smoke]
  .venv/bin/python tools/rebuttal_p2b.py report"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import rebuttal_p1p2 as P
from rebuttal_p1p2 import utc

OUT = ROOT / "runs/analysis/rebuttal_20260926c"
BATCH, STEPS = 64, 16
MODES = ("messages_frozen", "messages_mean")
RM_HIGH, RM_LOW, D_LOW, D_FRAC = 0.9, 0.5, 5, 0.25


class Hook:
    """Trace-time interception of the cross-field message inside the release cell's block."""
    mode = "off"; captured = None; bank = None; k = 0
    @classmethod
    def intercept(cls, msg):
        if cls.mode == "record":
            cls.captured.append(msg); return msg
        if cls.mode == "replay":
            m = cls.bank[cls.k]; cls.k += 1; return m
        return msg


def letter_rm(R):
    if R is None: return "UNDEFINED"
    if R >= RM_HIGH: return "CLUE-ACCESS-MOSTLY"
    if R <= RM_LOW: return "EXCHANGE-NEEDED"
    return "MIXED"

def letter_d(D, intact_new):
    if D is None: return "UNDEFINED"
    if D <= D_LOW: return "NO-DISCOVERY"
    if D >= D_FRAC * intact_new: return "DISCOVERY"
    return "MIXED"


class R4(P.R2):
    def __init__(self, width, out=OUT):
        super().__init__(width, out=out)
        import jax, jax.numpy as jnp
        from qhrrn2 import model as M
        self.M = M; DC = self.DC; TC = DC.TC; F = DC.F
        assert not getattr(self.cfg, "remat", False), "remat would re-trace the hooked block"
        self.n_msgs = self.cfg.trm_h_cycles * (self.cfg.trm_l_cycles + 1) * self.cfg.trm_layers
        assert self.n_msgs == 42

        def hooked_block(p, h, cfg=None):                       # the release's _block with one interception line
            def tok(hf):
                if "att_qkv" in p:
                    return DC._attn_tok(p, hf, cfg.dec_tok_dk)
                ht = hf.T
                ht = TC._rms_norm(ht + TC._swiglu(p["mlp_t"], ht))
                return ht.T
            h = jax.vmap(tok)(h)
            if "fc" in p:
                if "attn_q" in p:
                    nh = cfg.dec_attn_heads; dk = cfg.dec_attn_dk; w = h.shape[-1]
                    q = (h @ p["attn_q"]).reshape(F, -1, nh, dk)
                    k = (h @ p["attn_k"]).reshape(F, -1, nh, dk)
                    v = (h @ p["fc"]).reshape(F, -1, nh, w // nh)
                    e = jnp.einsum("fshd,gshd->fgsh", q, k) / math.sqrt(dk)
                    e = jnp.where(jnp.eye(F, dtype=bool)[:, :, None, None], -1000000000.0, e)
                    att = jax.nn.softmax(e, axis=1)
                    msg = jnp.einsum("fgsh,gshe->fshe", att, v).reshape(h.shape)
                    msg = Hook.intercept(msg)
                    h = TC._rms_norm(h + msg)
                else:
                    others = (jnp.sum(h, axis=0, keepdims=True) - h) / (F - 1)
                    h = TC._rms_norm(h + Hook.intercept(others @ p["fc"]))
            return TC._rms_norm(h + TC._swiglu(p["mlp"], h))
        self.original_block = DC._block
        DC._block = hooked_block                                 # in-process only; the release file is untouched
        self._steps = {}

    def step_fn(self, first, mode, bank_batched):
        key = (first, mode, bank_batched)
        if key in self._steps: return self._steps[key]
        jax, jnp, M, cfg = self.jax, self.jnp, self.M, self.cfg
        n_msgs = self.n_msgs
        def fwd(params, x_can, y, tv, z, bank):
            Hook.mode = mode; Hook.captured = []; Hook.bank = bank; Hook.k = 0
            out = M.forward_fields(params, cfg, M.build_fields_soft(x_can, y), t_norm=0.0, tau=1.0, rng=None, task_vec=tv, z_in=None if first else z)
            if mode == "record":
                assert len(Hook.captured) == n_msgs, len(Hook.captured); extra = jnp.stack(Hook.captured)
            elif mode == "replay":
                assert Hook.k == n_msgs, Hook.k; extra = jnp.zeros(())
            else:
                extra = jnp.zeros(())
            Hook.mode = "off"
            return out.logits, out.z_fine, extra
        in_axes = (None, 0, 0, None, None if first else 0, 0 if bank_batched else None)
        f = jax.jit(jax.vmap(fwd, in_axes=in_axes)); self._steps[key] = f; return f

    def one_step(self, x, y, z, first, mode, bank, bank_batched):
        jax, jnp, EV = self.jax, self.jnp, self.EV
        f = self.step_fn(first, mode, bank_batched)
        lg_dev, zf, extra = f(self.params, x, y, self.tv, jnp.zeros(1) if first else z, bank)
        z = zf if first else z + self.eta_z * (zf - z)
        p = jax.nn.softmax(lg_dev, axis=-1).transpose(0, 3, 1, 2); y = y + self.eta * (p - y)
        lg = np.asarray(EV.layout_gather(lg_dev, self.layout), np.float32)
        raw = np.asarray(EV.layout_gather(jnp.argmax(lg_dev, axis=-1), self.layout)); pr = np.where(raw == self.G.VOID, 0, raw).astype(np.int8)
        return y, z, lg, pr, extra

    def first_step(self, puz):
        """The shared first state with the recorded iteration-1 messages (B, 42, F, S, w)."""
        jax, jnp, EV = self.jax, self.jnp, self.EV
        x = EV.place_batch(puz, self.layout)
        void = jax.nn.one_hot(jnp.full((9, 9), self.G.VOID, jnp.int32), 11).transpose(2, 0, 1); y = jnp.broadcast_to(void, (len(puz),) + void.shape)
        y, z, lg, pr, bank = self.one_step(x, y, None, True, "record", jnp.zeros(1), False)
        return x, y, z, lg, pr, bank

    def continue_with(self, x, y, z, lg0, pr0, bank, bank_batched, steps, progress=None):
        logits, preds = [lg0], [pr0]
        for t in range(1, steps):
            y, z, lg, pr, _ = self.one_step(x, y, z, False, "replay", bank, bank_batched)
            logits.append(lg); preds.append(pr)
            if progress and (t + 1) % 4 == 0: self.status("running", **progress, iteration=t + 1)
        return dict(logits=np.stack(logits), pred=np.stack(preds))

    def gate(self, smoke):
        with np.load(self.STUDY / f"attention_{self.width}/interventions/intact/batch_0000.npz", allow_pickle=False) as d:
            ids, puz, ref = d["ids"], d["puz"], d["logits"]
        n = 16 if smoke else BATCH
        x, y, z, lg, pr, bank = self.first_step(puz[:n])
        d1 = float(np.abs(lg - ref[0, :n]).max())
        assert np.array_equal(lg, ref[0, :n]) or d1 < 1e-4, f"record-mode first step differs from the study by {d1}"
        jnp = self.jnp
        y2, z2, lg2, pr2, _ = self.one_step(x, jnp.broadcast_to(y[:1] * 0, y.shape) if False else y * 0 + (self.jax.nn.one_hot(jnp.full((9, 9), self.G.VOID, jnp.int32), 11).transpose(2, 0, 1))[None], None, True, "replay", bank, True)
        d2 = float(np.abs(lg2 - lg).max())
        assert np.array_equal(lg2, lg) or d2 < 1e-4, f"replay of the recorded messages differs from the record by {d2}"
        info = dict(passed=True, created=utc(), n=n, record_vs_study_bitwise=bool(np.array_equal(lg, ref[0, :n])), record_vs_study_max_abs=d1,
                    replay_vs_record_bitwise=bool(np.array_equal(lg2, lg)), replay_vs_record_max_abs=d2, messages_per_iteration=self.n_msgs)
        self.ATS.write_json(self.dir / ("gate_smoke.json" if smoke else "gate.json"), info); self.ATS.log("gate_passed", width=self.width, **info)

    def run(self, smoke):
        try:
            self.gate(smoke)
            with np.load(self.STUDY / "inputs/interventions.npz", allow_pickle=False) as d: ids, puz, sol = d["ids"], d["puz"], d["sol"]
            if smoke: ids, puz, sol = ids[:16], puz[:16], sol[:16]
            steps = 2 if smoke else STEPS; group = "state"
            # phase A: the iteration-1 population mean of every message tensor
            mean_path = self.dir / "mean_bank.npz"
            if mean_path.exists():
                with np.load(mean_path) as d: mean_bank = d["mean"]
            else:
                acc = None
                for b in range(0, len(ids), BATCH):
                    _, _, _, _, _, bank = self.first_step(puz[b:b + BATCH]); s = np.asarray(bank, np.float64).sum(0); acc = s if acc is None else acc + s
                mean_bank = (acc / len(ids)).astype(np.float32); np.savez_compressed(mean_path, mean=mean_bank, n=len(ids))
            mean_bank_j = self.jnp.asarray(mean_bank)
            # phase B: the two conditions, branching from the shared first state
            for b in range(0, len(ids), BATCH):
                sl = slice(b, b + BATCH); ii, pp, ss = ids[sl], puz[sl], sol[sl]
                paths = {m: self.dir / group / m / f"batch_{b:04d}.npz" for m in MODES}
                done = {m: self.cached(paths[m], ii, group, m) for m in MODES}
                if all(done.values()): continue
                x, y, z, lg, pr, bank = self.first_step(pp)
                if not smoke:
                    with np.load(self.STUDY / f"attention_{self.width}/interventions/intact/batch_{(b // 128) * 128:04d}.npz", allow_pickle=False) as d:
                        off = b % 128; assert np.abs(d["logits"][0, off:off + len(ii)] - lg).max() < 1e-4, "shared first state differs from the study"
                for m in MODES:
                    if done[m]: continue
                    started, cpu = time.monotonic(), time.process_time()
                    values = self.continue_with(x, y, z, lg, pr, bank if m == "messages_frozen" else mean_bank_j, m == "messages_frozen", steps, progress=dict(group=group, condition=m, batch_start=b))
                    self.save(paths[m], values, pp, ss, ii, group, m, started, cpu, dict(messages_per_iteration=self.n_msgs, batch_size=BATCH))
            self.ATS.verify_sources(); self.status("inference_complete", completed=utc()); self.ATS.log("model_complete", width=self.width, group="p2b")
        except BaseException as exc:
            self.status("failed", error=repr(exc)); raise


def report(base=OUT):
    import attention_transfer_study as ATS
    lines = ["P2b REPORT (rules registered 2026-09-26): message freeze and mean ablation on the 256 shared first states"]; out = {}
    for w in P.WIDTHS:
        ref = P.load_group(w, "interventions", ("intact", "messages_off"), ATS.OUT); new = P.load_group(w, "state", MODES, base)
        if not ref or len(new) < len(MODES): lines.append(f"attention_{w}: not complete"); continue
        # initial correctness from the study's intact chunks (iteration 1 is shared)
        init = []
        for pth in sorted((ATS.OUT / f"attention_{w}/interventions/intact").glob("batch_*.npz")):
            with np.load(pth, allow_pickle=False) as d: init.append(d["exact"][0])
        init = np.concatenate(init); I = int(ref["intact"]["exact"].sum()); OFF = int(ref["messages_off"]["exact"].sum()); new_intact = int((ref["intact"]["exact"] & ~init).sum())
        res = dict(intact=I, messages_off=OFF, initially_correct=int(init.sum()), intact_new=new_intact)
        for m in MODES:
            ex = new[m]["exact"]; assert len(ex) == len(init); M_ = int(ex.sum()); R = M_ / I if I else None; D = int((ex & ~init).sum()); L = int((init & ~ex).sum())
            res[m] = dict(exact=M_, R=R, letter_R=letter_rm(R), new=D, lost=L, letter_D=letter_d(D, new_intact))
            lines.append(f"attention_{w} {m}: exact16 {M_} (intact {I}, messages off {OFF}) | R_m {R:.3f} -> {letter_rm(R)} | new discoveries {D} of {int((~init).sum())} initially incorrect (intact {new_intact}) -> {letter_d(D, new_intact)} | initial solutions lost {L} of {int(init.sum())}")
        out[f"attention_{w}"] = res
    (base / "report.txt").write_text("\n".join(lines) + "\n"); (base / "report.json").write_text(json.dumps(out, indent=1, default=float)); print("\n".join(lines))


def selftest():
    import numpy as np
    Hook.mode = "record"; Hook.captured = []; a, b = np.ones(3), 2 * np.ones(3)
    assert Hook.intercept(a) is a and Hook.intercept(b) is b and len(Hook.captured) == 2
    Hook.mode = "replay"; Hook.bank = np.stack([b, a]); Hook.k = 0
    assert np.array_equal(Hook.intercept(a), b) and np.array_equal(Hook.intercept(a), a) and Hook.k == 2
    Hook.mode = "off"; assert Hook.intercept(a) is a
    assert letter_rm(0.95) == "CLUE-ACCESS-MOSTLY" and letter_rm(0.9) == "CLUE-ACCESS-MOSTLY" and letter_rm(0.5) == "EXCHANGE-NEEDED" and letter_rm(0.7) == "MIXED" and letter_rm(None) == "UNDEFINED"
    assert letter_d(0, 192) == "NO-DISCOVERY" and letter_d(5, 192) == "NO-DISCOVERY" and letter_d(48, 192) == "DISCOVERY" and letter_d(47, 192) == "MIXED" and letter_d(6, 192) == "MIXED"
    # a mutant replay that ignores the bank would return the live message
    Hook.mode = "replay"; Hook.bank = np.stack([b]); Hook.k = 0; assert not np.array_equal(Hook.intercept(a), a)
    print("selftest OK: hook record/replay/off semantics, replay mutant, R_m and D letters at their boundaries")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", nargs="?", choices=["run", "report"]); ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--width", type=int, choices=P.WIDTHS); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.selftest: selftest(); return
    if a.action == "run":
        if a.width is None: ap.error("--width required")
        out = Path(os.environ["P2B_SMOKE_OUT"]) if a.smoke else OUT
        R4(a.width, out).run(a.smoke)
    elif a.action == "report": report()
    else: ap.print_help()


if __name__ == "__main__":
    main()
