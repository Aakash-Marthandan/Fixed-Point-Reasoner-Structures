#!/usr/bin/env python3
# Ledger: THE DISCUSSION-PERIOD EXPERIMENT P3 — a one-time, score-preserving edit of the readout-invisible slow state across the
# corruption ladder (registration Documentation/Note_2026-10-01_Rebuttal_P3_Registration.md, written before this build). MEASUREMENT,
# $0, inference only on the banked attention checkpoints; imports the P1/P2 pipeline (tools/rebuttal_p1p2.py) unchanged.
"""P3: edit the slow state ONCE, before iteration t0, keeping every digit score, and watch the repair of corrupted grids.

Edits (h = h_par + h_perp per field/cell vector; h_par is the readout component, h_perp is invisible to the readout):
  random_t2 / random_t4  h_perp -> random vector orthogonal to the readout, same norm (P2's null_random edit, applied once)
  rot30_t2 / rot60_t2    h_perp rotated by 30 / 60 degrees toward a random direction orthogonal to the readout and to h_perp
  sham_t2                h_perp reassembled unchanged (the numerical control for the edit path)
Populations: the 512 repair puzzles under four draw-0 corruption families (uniform_0, legal_random_0, consistent_random_0, eqr),
and the 256 shared first states of the study (P2's population). Rules R1 / R2, strata and letters are in the registration note.
"""
import argparse, json, math, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import rebuttal_p1p2 as P                                   # the P1/P2 pipeline, unchanged

OUT = ROOT / "runs/analysis/rebuttal_20261001"
SEED = 20261001
WIDTHS, BATCH, STEPS = P.WIDTHS, P.BATCH, P.STEPS
FAMILIES = ("uniform_0", "legal_random_0", "consistent_random_0", "eqr")
EDITS_PRIMARY, EDITS_SECOND, EDITS_DOSE = ("sham_t2", "random_t2"), ("random_t4",), ("rot30_t2", "rot60_t2")
REPAIR_EDITS = EDITS_PRIMARY + EDITS_SECOND + EDITS_DOSE
STATE_EDITS = ("sham_t2", "random_t2", "random_t4")
STRATA = (("S1", 1, 5), ("S2", 6, 15), ("S3", 16, 30), ("S4", 31, 81))
MIN_STRATUM, RHO_HIGH, RHO_LO, RHO_HI, E_FLOOR = 20, 1.5, 0.75, 1.33, 0.01
R1_HIGH, R1_LOW = 0.75, 0.25
P1_OUT = P.OUT                                             # runs/analysis/rebuttal_20260926 (P1/P2)
P1C_OUT = ROOT / "runs/analysis/rebuttal_20260926b"        # P1c


def utc(): return datetime.now(timezone.utc).isoformat()


# ---------------- the edits (float64 numpy; every edit preserves the digit scores exactly up to rounding) ----------------
def parse_edit(name):
    """'random_t2' -> ('random', None, 2); 'rot30_t2' -> ('rot', 30.0, 2); 'sham_t2' -> ('sham', None, 2)."""
    kind, t = name.split("_t"); t0 = int(t)
    if kind.startswith("rot"): return "rot", float(kind[3:]), t0
    assert kind in ("random", "sham"), name
    return kind, None, t0


def rotate_perp(perp, v, theta_deg, rng):
    """Rotate each vector of `perp` (..., w), all orthogonal to v, by theta toward a random unit direction orthogonal to v and to
    the vector itself. Norm preserved; displacement = 2 |perp| sin(theta/2); readout component stays zero."""
    v = np.asarray(v, np.float64); vhat = v / np.linalg.norm(v)
    g = rng.standard_normal(perp.shape)
    g = g - (g @ vhat)[..., None] * vhat                                  # orthogonal to the readout
    n2 = np.maximum((perp * perp).sum(-1, keepdims=True), 1e-30)
    g = g - (g * perp).sum(-1, keepdims=True) / n2 * perp                  # orthogonal to the vector itself
    g = g / np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-30)
    th = math.radians(theta_deg)
    return math.cos(th) * perp + math.sin(th) * np.sqrt(n2) * g


def apply_edit(h, lm, kind, theta, ids, t, record=None):
    """h: (B, F, S, w) float64 slow state. Returns the edited state (float64) and a dict of realized statistics."""
    par, perp = P.project(h, lm)
    if kind == "sham":
        new = perp
    elif kind == "random":
        new = np.stack([P.norm_matched_random(perp[b], lm, np.random.default_rng([SEED, int(ids[b]), 41, t])) for b in range(len(ids))])
    else:
        new = np.stack([rotate_perp(perp[b], lm, theta, np.random.default_rng([SEED, int(ids[b]), 42, t])) for b in range(len(ids))])
    h2 = par + new
    shift = float(np.abs(h2 @ lm - h @ lm).max())
    pn = np.linalg.norm(perp, axis=-1); disp = np.linalg.norm(new - perp, axis=-1)
    stats = dict(max_score_shift=shift, relative_displacement_mean=float((disp / np.maximum(pn, 1e-30)).mean()),
                 perp_norm_mean=float(pn.mean()), norm_ratio_mean=float((np.linalg.norm(new, axis=-1) / np.maximum(pn, 1e-30)).mean()))
    assert shift < 1e-2, f"score shift {shift}"
    return h2, stats


# ---------------- inputs ----------------
def family_sources():
    """The 512 repair puzzles with the four draw-0 corruption families; ids asserted identical across the three input files."""
    ATS, STUDY = P.study_paths()
    with np.load(STUDY / "inputs/repair.npz", allow_pickle=False) as d:
        ids, puz, sol, eqr, uni, counts, bins = (d[k] for k in ("ids", "puz", "sol", "eqr", "random", "wrong_count", "rating_bin"))
    with np.load(P1_OUT / "inputs/p1_sources.npz", allow_pickle=False) as d:
        assert np.array_equal(d["ids"], ids); legal = d["legal_random_0"]; assert np.array_equal(d["uniform_0"], uni[0])
    with np.load(P1C_OUT / "inputs/p1c_sources.npz", allow_pickle=False) as d:
        assert np.array_equal(d["ids"], ids); cons = d["consistent_random_0"]
    fam = dict(uniform_0=uni[0], legal_random_0=legal, consistent_random_0=cons, eqr=eqr)
    for name, g in fam.items():
        assert g.shape == sol.shape and np.array_equal(g[puz != 0], puz[puz != 0]), name
    return dict(ids=ids, puz=puz, sol=sol, wrong_count=counts, rating_bin=bins, families=fam)


def reference_dir(width, family):
    """The saved unedited trajectory of each family (per-iteration logits), read rather than re-run."""
    ATS, STUDY = P.study_paths()
    return {"eqr": STUDY / f"attention_{width}/repair/eqr", "uniform_0": STUDY / f"attention_{width}/repair/random_0",
            "legal_random_0": P1_OUT / f"attention_{width}/repair/legal_random_0",
            "consistent_random_0": P1C_OUT / f"attention_{width}/repair/consistent_random_0"}[family]


def load_chunks(d, keys=("ids", "pred", "exact", "sol", "puz")):
    parts = sorted(Path(d).glob("batch_*.npz"))
    if not parts: return None
    out = {}
    for pth in parts:
        side = pth.with_suffix("").with_suffix(".sha256.json")
        if side.exists():
            import hashlib; assert hashlib.sha256(pth.read_bytes()).hexdigest() == json.loads(side.read_text())["sha256"], f"digest {pth}"
        with np.load(pth, allow_pickle=False) as z:
            for k in keys: out.setdefault(k, []).append(z[k])
    return {k: np.concatenate(v, axis=1 if k in ("pred", "exact", "logits") else 0) for k, v in out.items()}


# ---------------- the runner ----------------
class R5(P.R2):
    def __init__(self, width, out=OUT):
        super().__init__(width, out)
        self.meta.update(study="rebuttal_20261001", pipeline="tools/rebuttal_p3.py over tools/rebuttal_p1p2.py over tools/attention_transfer_study.py")
        self.ATS.write_json(self.dir / "runtime.json", self.meta)

    def trajectory_edit(self, puz, ids, initial=None, branch=None, edit=None, steps=STEPS, progress=None):
        """The study's loop (P.R2.trajectory, mode='intact') with ONE edit of the slow state before iteration t0."""
        jax, jnp, EV = self.jax, self.jnp, self.EV
        kind, theta, t0 = parse_edit(edit) if edit else (None, None, None)
        x = EV.place_batch(puz, self.layout)
        void = jax.nn.one_hot(jnp.full((9, 9), self.G.VOID, jnp.int32), 11).transpose(2, 0, 1)
        y = jnp.broadcast_to(void, (len(puz),) + void.shape)
        z = initial; pr = None
        logits, preds, residuals, norms = [], [], [], []
        first_state = None; edit_stats = None
        for t in range(steps):
            if t == 0 and branch is not None:
                y, z, lg, pr, rr, nn = branch
            else:
                if edit and t == t0 - 1:
                    assert t > 0 and z is not None
                    h = np.asarray(z[:, 0], np.float64)
                    h2, edit_stats = apply_edit(h, self.lm, kind, theta, ids, t)
                    h2 = jnp.asarray(h2, jnp.float32)
                    edit_stats["realized_float32_score_shift"] = float(jnp.abs(jnp.einsum("bfsw,w->bsf", h2, jnp.asarray(self.lm, jnp.float32))
                                                                                 - jnp.einsum("bfsw,w->bsf", z[:, 0], jnp.asarray(self.lm, jnp.float32))).max())
                    edit_stats["applied_before_iteration"] = t + 1
                    z = z.at[:, 0].set(h2)
                active = self.params
                first = z is None
                old = z
                lg_dev, zf = EV._step(self.cfg, 1., 0., first)(active, x, y, self.tv, jnp.zeros(1) if first else z)
                z = zf if first else z + self.eta_z * (zf - z)
                p = jax.nn.softmax(lg_dev, axis=-1).transpose(0, 3, 1, 2)
                y = y + self.eta * (p - y)
                lg = np.asarray(EV.layout_gather(lg_dev, self.layout), np.float32)
                raw_pred = np.asarray(EV.layout_gather(jnp.argmax(lg_dev, axis=-1), self.layout))
                pr = np.where(raw_pred == self.G.VOID, 0, raw_pred).astype(np.int8)
                delta = z - (jnp.broadcast_to(self.z0, z.shape) if old is None else old)
                rr = np.asarray(jnp.mean(jnp.abs(delta), axis=tuple(range(2, z.ndim))), np.float32)
                nn = np.asarray(jnp.sqrt(jnp.mean(z*z, axis=tuple(range(2, z.ndim)))), np.float32)
                assert np.isfinite(lg).all() and np.isfinite(rr).all() and np.isfinite(nn).all()
                assert np.array_equal(pr, np.where(lg.argmax(-1) == self.G.VOID, 0, lg.argmax(-1)))
            if t == 0:
                first_state = (y, z, lg, pr, rr, nn)
            logits.append(lg); preds.append(pr); residuals.append(rr); norms.append(nn)
            if progress and (t + 1) % 4 == 0:
                self.status("running", **progress, iteration=t + 1)
        if edit and steps < t0: edit_stats = dict(skipped=True, applied_before_iteration=None)
        extra = dict(edit=edit, **(edit_stats or {}))
        return dict(logits=np.stack(logits), pred=np.stack(preds), carry_residual=np.stack(residuals), carry_rms=np.stack(norms)), first_state, extra

    def gate(self, smoke):
        """(1) P2's gate: the study's intact first batch bitwise. (2) Sham neutrality on batch 0 of the eqr family."""
        super().gate(smoke)
        n = 16 if smoke else BATCH; steps = 5 if smoke else STEPS
        src = family_sources(); ids, puz, sol = src["ids"][:n], src["puz"][:n], src["sol"][:n]; grid = src["families"]["eqr"][:n]
        ref = load_chunks(reference_dir(self.width, "eqr"), keys=("ids", "exact", "logits"))
        assert np.array_equal(ref["ids"][:n], ids)
        values, _, extra = self.trajectory_edit(puz, ids, initial=self.starting_state(grid), edit="sham_t2", steps=steps)
        exact = (values["pred"] == sol[None]).all((2, 3))
        same = bool(np.array_equal(exact, ref["exact"][:steps, :n]))
        maxdiff = float(np.abs(values["logits"] - ref["logits"][:steps, :n]).max())
        self.sham_neutral = same
        self.ATS.write_json(self.dir / ("gate_sham_smoke.json" if smoke else "gate_sham.json"),
                            dict(passed_exact_flags=same, max_abs_logit_diff=maxdiff, n=n, steps=steps, created=utc(), **extra))
        self.ATS.log("gate_sham", width=self.width, n=n, exact_flags_identical=same, max_abs_logit_diff=maxdiff, smoke=smoke)

    def state_group(self, smoke, edits=STATE_EDITS):
        group = "state"
        with np.load(self.STUDY / "inputs/interventions.npz", allow_pickle=False) as d: ids, puz, sol = d["ids"], d["puz"], d["sol"]
        if smoke: ids, puz, sol = ids[:16], puz[:16], sol[:16]
        steps = 5 if smoke else STEPS
        for b in range(0, len(ids), BATCH):
            sl = slice(b, b + BATCH); ii, pp, ss = ids[sl], puz[sl], sol[sl]
            paths = {c: self.dir / group / c / f"batch_{b:04d}.npz" for c in edits}
            done = {c: self.cached(paths[c], ii, group, c) for c in edits}
            if all(done.values()): continue
            _, branch, _ = self.trajectory(pp, ii, steps=1)
            if not smoke:
                with np.load(self.STUDY / f"attention_{self.width}/interventions/intact/batch_{b:04d}.npz", allow_pickle=False) as d:
                    assert np.array_equal(d["logits"][0], branch[2]), "shared first state differs from the study"
            for c in edits:
                if done[c]: continue
                started, cpu = time.monotonic(), time.process_time()
                values, own, extra = self.trajectory_edit(pp, ii, branch=branch, edit=c, steps=steps, progress=dict(group=group, condition=c, batch_start=b))
                assert np.array_equal(values["logits"][0], branch[2])
                self.save(paths[c], values, pp, ss, ii, group, c, started, cpu, extra)

    def repair_group(self, smoke, edits=REPAIR_EDITS, families=FAMILIES):
        src = family_sources(); ids, puz, sol = src["ids"], src["puz"], src["sol"]
        n = 16 if smoke else len(ids); steps = 5 if smoke else STEPS
        for c in edits:                                   # edit-major order: the primary set finishes first on every family
            for fam in families:
                group = f"repair/{fam}"
                for b in range(0, n, BATCH):
                    sl = slice(b, min(b + BATCH, n)); ii, pp, ss, gg = ids[sl], puz[sl], sol[sl], src["families"][fam][sl]
                    path = self.dir / group / c / f"batch_{b:04d}.npz"
                    if self.cached(path, ii, group, c): continue
                    started, cpu = time.monotonic(), time.process_time()
                    values, _, extra = self.trajectory_edit(pp, ii, initial=self.starting_state(gg), edit=c, steps=steps,
                                                            progress=dict(group=group, condition=c, batch_start=b))
                    self.save(path, values, pp, ss, ii, group, c, started, cpu, extra, supplied=gg)

    def run(self, smoke, stage="all"):
        try:
            self.gate(smoke)
            if stage in ("state", "all"): self.state_group(smoke)
            if stage in ("primary", "all"): self.repair_group(smoke, EDITS_PRIMARY)
            if stage in ("second", "all"): self.repair_group(smoke, EDITS_SECOND)
            if stage in ("dose", "all"): self.repair_group(smoke, EDITS_DOSE)
            self.ATS.verify_sources(); self.status("inference_complete", completed=utc()); self.ATS.log("model_complete", width=self.width, group=f"p3_{stage}")
        except BaseException as exc:
            self.status("failed", error=repr(exc)); raise


# ---------------- the read (rules R1 / R2 and the exploratory tables) ----------------
def empty_error(pred, sol, puz):
    ng = puz == 0
    return ((pred != sol[None]) & ng[None]).sum((2, 3)) / np.maximum(ng.sum((1, 2)), 1)[None]


def first_exact(exact):
    """First exact iteration (1-based) per puzzle, 0 if never."""
    T, n = exact.shape; any_ = exact.any(0); first = exact.argmax(0) + 1
    return np.where(any_, first, 0)


def stratum_of(wrong):
    out = np.full(wrong.shape, "", dtype=object)
    for name, lo, hi in STRATA: out[(wrong >= lo) & (wrong <= hi)] = name
    out[wrong == 0] = "solved"
    return out


def letter_r1(rho_by_stratum, L_ok_by_stratum, undefined):
    """rho_by_stratum: {width: {stratum: rho}} over eligible strata; L_ok: {width: {stratum: bool}}; undefined: {width: {stratum: bool}}."""
    rhos = [r for w in rho_by_stratum for r in rho_by_stratum[w].values()]
    if not rhos: return "UNDEFINED"
    if any(u for w in undefined for u in undefined[w].values()): return "UNDEFINED"
    if all(r >= RHO_HIGH for r in rhos) and all(ok for w in L_ok_by_stratum for ok in L_ok_by_stratum[w].values()): return "CONFIGURATION-DEPENDENT"
    if all(RHO_LO <= r <= RHO_HI for r in rhos): return "INDEPENDENT"
    return "MIXED"


def letter_r2(R1s):
    if not R1s or any(r is None for r in R1s): return "UNDEFINED"
    if all(r >= R1_HIGH for r in R1s): return "RECOVERS"
    if all(r <= R1_LOW for r in R1s): return "PERSISTS"
    return "MIXED"


def analyse_family(width, fam, edits, base):
    """Per-puzzle quantities for one family: sham reference and each edit."""
    sham = load_chunks(base / f"attention_{width}/repair/{fam}/sham_t2")
    if sham is None: return None
    ref = load_chunks(reference_dir(width, fam), keys=("ids", "exact"))
    assert np.array_equal(ref["ids"], sham["ids"])
    sol, puz = sham["sol"], sham["puz"]; n = len(sham["ids"])
    eligible = ((sham["pred"][0] != sol) | True)                   # placeholder shape
    e_sham = empty_error(sham["pred"], sol, puz); ex_sham = sham["exact"]; tau_sham = first_exact(ex_sham)
    out = dict(n=n, ids=sham["ids"], sham_vs_reference_exact_identical=bool(np.array_equal(ex_sham, ref["exact"])), edits={})
    wrong_pre = {}
    for c in edits:
        d = load_chunks(base / f"attention_{width}/repair/{fam}/{c}")
        if d is None: continue
        assert np.array_equal(d["ids"], sham["ids"])
        kind, theta, t0 = parse_edit(c)
        pre = ((sham["pred"][t0 - 2] != sol) & (puz == 0)).sum((1, 2)) if t0 >= 2 else None   # wrong empty cells after iteration t0-1 (sham)
        strata = stratum_of(pre)
        e = empty_error(d["pred"], sol, puz); ex = d["exact"]; tau = first_exact(ex)
        res = dict(t0=t0, strata={})
        for name in [s[0] for s in STRATA]:
            m = strata == name; k = int(m.sum())
            if k == 0: res["strata"][name] = dict(n=0); continue
            dE = float((e[t0 - 1][m] - e_sham[t0 - 1][m]).mean())                     # error at iteration t0 (index t0-1), edit minus sham
            both = m & (ex_sham[-1]) ; L = float((both & ~ex[-1]).sum() / max(int(both.sum()), 1))
            both_done = m & ex_sham[-1] & ex[-1]
            dtau = float(np.median((tau[both_done] - tau_sham[both_done]))) if both_done.any() else None
            res["strata"][name] = dict(n=k, dE=dE, L=L, n_exact_sham=int(both.sum()), dtau_median=dtau, n_both_exact=int(both_done.sum()),
                                       censored_edit=int((both & ~ex[-1]).sum()))
        solved = strata == "solved"
        if solved.any():
            later = ~ex[t0 - 1:, solved].all(0)
            res["preservation"] = dict(n=int(solved.sum()), wrong_any_later=float(later.mean()), wrong_at_16=float((~ex[-1][solved]).mean()))
        res["exact16"] = int(ex[-1].sum()); res["exact16_sham"] = int(ex_sham[-1].sum())
        res["halves"] = {h: dict(exact16=int(ex[-1][sham["ids"] % 2 == h].sum()), exact16_sham=int(ex_sham[-1][sham["ids"] % 2 == h].sum())) for h in (0, 1)}
        out["edits"][c] = res
    return out


def report(base=OUT):
    lines = ["P3 REPORT (rules registered 2026-10-01): one-time score-preserving edits of the readout-invisible slow state"]; out = {}
    p2 = json.loads((P1_OUT / "report.json").read_text()) if (P1_OUT / "report.json").exists() else {}
    rho, Lok, undef, R1s = {}, {}, {}, []
    for w in WIDTHS:
        res = dict(repair={}, state={})
        for fam in FAMILIES:
            a = analyse_family(w, fam, REPAIR_EDITS, base)
            if a is None: lines.append(f"attention_{w} {fam}: not run"); continue
            res["repair"][fam] = dict(n=a["n"], sham_neutral=a["sham_vs_reference_exact_identical"], edits=a["edits"])
            for c, r in a["edits"].items():
                st = " | ".join(f"{s}: n {v['n']}" + (f" dE {100*v['dE']:+.2f}pp L {100*v['L']:.1f}% dtau {v['dtau_median']}" if v["n"] else "") for s, v in r["strata"].items())
                pres = r.get("preservation"); pres_s = f" | solved pre-edit n {pres['n']} wrong later {100*pres['wrong_any_later']:.1f}% at16 {100*pres['wrong_at_16']:.1f}%" if pres else ""
                lines.append(f"attention_{w} {fam} {c}: exact16 {r['exact16']} (sham {r['exact16_sham']}) | {st}{pres_s}")
        # R1 on random_t2: consistent_random_0 vs legal_random_0, strata S2-S4 with >= 20 in both
        rr = res["repair"]
        if "consistent_random_0" in rr and "legal_random_0" in rr and "random_t2" in rr["consistent_random_0"]["edits"] and "random_t2" in rr["legal_random_0"]["edits"]:
            cs, ls = rr["consistent_random_0"]["edits"]["random_t2"]["strata"], rr["legal_random_0"]["edits"]["random_t2"]["strata"]
            rho[w], Lok[w], undef[w] = {}, {}, {}
            for s in ("S2", "S3", "S4"):
                if cs[s]["n"] >= MIN_STRATUM and ls[s]["n"] >= MIN_STRATUM:
                    if ls[s]["dE"] < E_FLOOR:
                        undef[w][s] = True; rho[w][s] = float("nan")
                    else:
                        undef[w][s] = False; rho[w][s] = cs[s]["dE"] / ls[s]["dE"]
                    Lok[w][s] = cs[s]["L"] >= ls[s]["L"]
            lines.append(f"attention_{w} R1 strata: " + ", ".join(f"{s} rho {rho[w][s]:.2f} L_ok {Lok[w][s]}" for s in rho[w]) + (" (none eligible)" if not rho[w] else ""))
            if "eqr" in rr and "random_t2" in rr["eqr"]["edits"]:
                es = rr["eqr"]["edits"]["random_t2"]["strata"]
                lines.append(f"attention_{w} R1 model grids beside: " + ", ".join(f"{s} rho_eqr {es[s]['dE']/ls[s]['dE']:.2f}" for s in rho[w] if ls[s]['dE'] >= E_FLOOR))
            res["R1"] = dict(rho=rho[w], L_ok=Lok[w], undefined=undef[w])
        # R2 on the shared first states
        st = {}
        for c in STATE_EDITS:
            d = load_chunks(base / f"attention_{w}/state/{c}")
            if d is None: continue
            init = d["exact"][0]; ex = d["exact"][-1]
            st[c] = dict(exact16=int(ex.sum()), initial_lost=int((init & ~ex).sum()), new=int((~init & ex).sum()))
        res["state"] = st
        rep = p2.get(f"attention_{w}", {}).get("state", {}).get("null_random", {}).get("exact")
        if "random_t2" in st and rep is not None:
            R1 = (st["random_t2"]["exact16"] - rep) / (250 - rep); R1s.append(R1); res["R2"] = dict(R1=R1, repeated=rep)
            lines.append(f"attention_{w} R2: random_t2 exact16 {st['random_t2']['exact16']} vs repeated {rep} vs intact 250 -> R1 {R1:.3f}" +
                         (f"; random_t4 {st['random_t4']['exact16']}" if "random_t4" in st else "") + (f"; sham {st['sham_t2']['exact16']}" if "sham_t2" in st else ""))
        out[f"attention_{w}"] = res
    L1 = letter_r1(rho, Lok, undef) if rho else "NOT RUN"; L2 = letter_r2(R1s) if R1s else "NOT RUN"
    lines.append(f"R1 letter: {L1} | R2 letter: {L2}")
    out["letters"] = dict(R1=L1, R2=L2)
    (base / "report.txt").write_text("\n".join(lines) + "\n"); (base / "report.json").write_text(json.dumps(out, indent=1, default=float))
    print("\n".join(lines))


# ---------------- selftest ----------------
def selftest():
    rng = np.random.default_rng(0); w = 16; v = rng.standard_normal(w); h = rng.standard_normal((2, 3, 4, w))
    par, perp = P.project(h, v)
    assert np.allclose(perp @ v, 0, atol=1e-9) and np.allclose(par + perp, h)
    # rotation: orthogonal to v, norm preserved, angle theta, displacement 2|p| sin(theta/2)
    for th in (30.0, 60.0):
        new = rotate_perp(perp, v, th, np.random.default_rng(1))
        assert np.allclose(new @ v, 0, atol=1e-9)
        assert np.allclose(np.linalg.norm(new, axis=-1), np.linalg.norm(perp, axis=-1))
        cosang = (new * perp).sum(-1) / (np.linalg.norm(new, axis=-1) * np.linalg.norm(perp, axis=-1))
        assert np.allclose(cosang, math.cos(math.radians(th)), atol=1e-9)
        assert np.allclose(np.linalg.norm(new - perp, axis=-1), 2 * np.linalg.norm(perp, axis=-1) * math.sin(math.radians(th) / 2))
    # apply_edit: sham identity, random/rot preserve scores; mutant: a rotation that forgets to re-normalize is caught
    ids = np.array([5, 6])
    h2, s = apply_edit(h, v, "sham", None, ids, 1); assert np.allclose(h2, h) and s["max_score_shift"] < 1e-9 and abs(s["norm_ratio_mean"] - 1) < 1e-9
    h3, s3 = apply_edit(h, v, "random", None, ids, 1); assert s3["max_score_shift"] < 1e-9 and abs(s3["norm_ratio_mean"] - 1) < 1e-9 and s3["relative_displacement_mean"] > 1.0
    h4, s4 = apply_edit(h, v, "rot", 60.0, ids, 1); assert s4["max_score_shift"] < 1e-9 and abs(s4["relative_displacement_mean"] - 1.0) < 1e-9
    bad = perp + 0.3 * np.ones_like(perp); assert not np.allclose(np.linalg.norm(bad, axis=-1), np.linalg.norm(perp, axis=-1)), "mutant must differ"
    # determinism of the seeded edits
    assert np.array_equal(apply_edit(h, v, "random", None, ids, 1)[0], h3)
    # parse / strata / letters
    assert parse_edit("random_t2") == ("random", None, 2) and parse_edit("rot30_t2") == ("rot", 30.0, 2) and parse_edit("sham_t2") == ("sham", None, 2)
    assert list(stratum_of(np.array([0, 1, 5, 6, 15, 16, 30, 31, 81]))) == ["solved", "S1", "S1", "S2", "S2", "S3", "S3", "S4", "S4"]
    assert first_exact(np.array([[0, 0], [1, 0], [1, 0]], bool)).tolist() == [2, 0]
    assert letter_r1({128: {"S2": 1.6, "S3": 2.0}}, {128: {"S2": True, "S3": True}}, {128: {"S2": False, "S3": False}}) == "CONFIGURATION-DEPENDENT"
    assert letter_r1({128: {"S2": 1.0, "S3": 1.2}}, {128: {"S2": True, "S3": True}}, {128: {"S2": False, "S3": False}}) == "INDEPENDENT"
    assert letter_r1({128: {"S2": 1.6, "S3": 1.0}}, {128: {"S2": True, "S3": True}}, {128: {"S2": False, "S3": False}}) == "MIXED"
    assert letter_r1({128: {"S2": 1.6}}, {128: {"S2": False}}, {128: {"S2": False}}) == "MIXED"
    assert letter_r1({128: {"S2": float("nan")}}, {128: {"S2": True}}, {128: {"S2": True}}) == "UNDEFINED"
    assert letter_r2([0.8, 0.9, 0.76]) == "RECOVERS" and letter_r2([0.1, 0.2, 0.0]) == "PERSISTS" and letter_r2([0.8, 0.4, 0.9]) == "MIXED"
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", nargs="?", choices=["run", "report"])
    ap.add_argument("--selftest", action="store_true"); ap.add_argument("--width", type=int, choices=WIDTHS)
    ap.add_argument("--stage", default="all", choices=["state", "primary", "second", "dose", "all"])
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.selftest: selftest(); return
    out = Path(a.out) if a.out else OUT
    if a.action == "run":
        if a.width is None: ap.error("--width required")
        R5(a.width, out).run(a.smoke, a.stage)
    elif a.action == "report": report(out)
    else: ap.print_help()


if __name__ == "__main__":
    main()
