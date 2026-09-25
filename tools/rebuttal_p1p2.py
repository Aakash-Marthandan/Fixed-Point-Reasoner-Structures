#!/usr/bin/env python3
# Ledger: THE REVIEW-PERIOD EXPERIMENTS P1 + P2 (registration Documentation/Note_2026-09-26_Review_Period_P1_P2_Registration.md,
# written before any row). MEASUREMENT, $0, the Mac's CPU, inference only, nothing trained.
#   P1 (repair): does direct clue refutability explain the matched-error repair gap? Model-produced grids (EqR's first guess; the
#       attention models' own first guesses) versus random corruptions that are LEGAL against the givens, at matched wrong-cell counts
#       and, in one condition, at EqR's own error cells. Receivers: the three benchmark attention checkpoints (the paper's own pipeline,
#       tools/attention_transfer_study.py, release code, pinned sources) and, through tools/lens_repair_radius.py, MLP 192 (C5) and EqR.
#   P2 (state): what does the slow carry contribute beyond its readout? On the paper's 256 shared first states: (a) re-encode the
#       decoded grid into the slow state before every iteration 2-16; (b) keep every digit score and replace the readout-invisible
#       component by the initial buffer's; (c) the same with a norm-matched random component. Compared with the study's intact and
#       slow-reset branches.
"""  .venv/bin/python tools/rebuttal_p1p2.py --selftest
  .venv/bin/python tools/rebuttal_p1p2.py prepare
  JAX_PLATFORMS=cpu .venv/bin/python tools/rebuttal_p1p2.py run --width 128|192|256 [--group state|repair|all] [--smoke]
  JAX_PLATFORMS=cpu .venv/bin/python tools/rebuttal_p1p2.py port --receiver C5|EQR [--smoke]
  .venv/bin/python tools/rebuttal_p1p2.py report"""
from __future__ import annotations
import argparse, json, math, os, platform, resource, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
OUT = ROOT / "runs/analysis/rebuttal_20260926"
SEED_LEGAL, DRAWS, STEPS, BATCH = 20260926, 4, 16, 128
WIDTHS = (128, 192, 256)
REPAIR_CONDS = tuple([f"legal_random_{j}" for j in range(DRAWS)] + [f"legal_at_eqr_{j}" for j in range(DRAWS)]
                     + ["guess_sa128", "guess_sa256", "legal_match_sa128", "legal_match_sa256"])
STATE_CONDS = ("reencode_slow", "null_h0", "null_random")
PORT_SOURCES = ("eqr", "uniform_0") + REPAIR_CONDS
G_HIGH, G_LOW, R_HIGH, R_LOW = 0.67, 0.33, 0.9, 0.5


def utc(): return datetime.now(timezone.utc).isoformat()


# ---------------- pure helpers (selftested) ----------------
def unit_masks():
    """(81, 81) bool: cells sharing a row, column or box with each cell (self excluded)."""
    r, c = np.divmod(np.arange(81), 9)
    same = (r[:, None] == r[None]) | (c[:, None] == c[None]) | ((r[:, None] // 3 == r[None] // 3) & (c[:, None] // 3 == c[None] // 3))
    np.fill_diagonal(same, False)
    return same

UNITS = unit_masks()

def given_digits_in_units(puz):
    """puz (9, 9) -> (81, 9) bool: digit d (1..9) is a GIVEN in a unit of cell i."""
    p = np.asarray(puz).reshape(81)
    oh = (p[:, None] == np.arange(1, 10)[None]) & (p[:, None] != 0)           # (81, 9) given digit one-hot
    return (UNITS.astype(np.int32) @ oh.astype(np.int32)) > 0

def legal_wrong_digits(puz, sol, cell):
    """The digits (1..9) at flat cell index `cell` that are wrong and do not duplicate a given in the cell's row, column or box."""
    s = np.asarray(sol).reshape(81)[cell]
    bad = given_digits_in_units(puz)[cell]
    return [d for d in range(1, 10) if d != s and not bad[d - 1]]

def sample_legal_random(puz, sol, k, rng):
    """The solution with k empty cells made wrong by LEGAL digits at random feasible cells; returns (grid, assigned, deficit)."""
    puz, sol = np.asarray(puz), np.asarray(sol); g = sol.copy().reshape(81)
    empty = np.where(puz.reshape(81) == 0)[0]
    feasible = [c for c in empty if legal_wrong_digits(puz, sol, c)]
    take = int(min(k, len(feasible)))
    chosen = rng.choice(feasible, size=take, replace=False) if take else []
    for c in chosen:
        opts = legal_wrong_digits(puz, sol, int(c)); g[c] = opts[rng.integers(len(opts))]
    return g.reshape(9, 9), take, int(k - take)

def sample_legal_at_cells(puz, sol, cells, fallback, rng):
    """Legal wrong digits OTHER than the fallback grid's at the given flat cells; the fallback digit is kept only when it is the sole legal wrong digit. Returns (grid, fallbacks, fallback-and-wrong)."""
    puz, sol = np.asarray(puz), np.asarray(sol); g = sol.copy().reshape(81); fb = np.asarray(fallback).reshape(81)
    n_fb = n_co = 0
    for c in cells:
        opts = [d for d in legal_wrong_digits(puz, sol, int(c)) if d != fb[c]]     # EqR's own digit only when it is the sole legal wrong digit
        if opts:
            g[c] = opts[rng.integers(len(opts))]
        else:
            g[c] = fb[c]; n_fb += 1; n_co += int(fb[c] != sol.reshape(81)[c])
    return g.reshape(9, 9), n_fb, n_co

def clue_conflict_share(grid, puz, sol):
    """Share of wrong empty-cell digits that duplicate a GIVEN in their row, column or box (the paper's refutability measure)."""
    grid, puz, sol = (np.asarray(a).reshape(-1, 81) for a in (grid, puz, sol))
    wrong = (grid != sol) & (puz == 0); n = int(wrong.sum())
    if n == 0: return float("nan"), 0
    bad = np.stack([given_digits_in_units(p.reshape(9, 9)) for p in puz])           # (B, 81, 9)
    conflict = np.take_along_axis(bad, np.clip(grid, 1, 9)[..., None] - 1, axis=-1)[..., 0] & wrong
    return float(conflict.sum() / n), n

def project(h, v):
    """h (..., w), v (w,) -> (parallel, perpendicular) with parallel = (h.v / v.v) v; scores h.v are carried by `parallel` alone."""
    v = np.asarray(v, np.float64); h = np.asarray(h, np.float64)
    coef = (h @ v) / (v @ v)
    par = coef[..., None] * v
    return par, h - par

def norm_matched_random(perp, v, rng):
    """A random vector per (..., w) row, orthogonal to v, with the same L2 norm as `perp`'s row."""
    v = np.asarray(v, np.float64); r = rng.standard_normal(perp.shape)
    r = r - ((r @ v) / (v @ v))[..., None] * v
    scale = np.linalg.norm(perp, axis=-1, keepdims=True) / np.maximum(np.linalg.norm(r, axis=-1, keepdims=True), 1e-30)
    return r * scale

def gap_closure(e1_model, e1_uniform, e1_legal, floor=0.05):
    """G = (E1 legal - E1 uniform) / (E1 model - E1 uniform); None when the model-uniform gap is under `floor`."""
    den = e1_model - e1_uniform
    if not np.isfinite(den) or den < floor: return None
    return float((e1_legal - e1_uniform) / den)

def letter_p1(G):
    if G is None: return "UNDEFINED"
    if G >= G_HIGH: return "REFUTABILITY-MOSTLY"
    if G <= G_LOW: return "CONFIGURATION-BEYOND"
    return "MIXED"

def recovery(intact, mode, reset_slow, floor=20):
    """R = 1 - (intact - mode) / (intact - reset_slow); None when the slow-reset loss is under `floor` counts."""
    den = intact - reset_slow
    if den < floor: return None
    return float(1 - (intact - mode) / den)

def letter_p2(R):
    if R is None: return "UNDEFINED"
    if R >= R_HIGH: return "ANSWER-ONLY"
    if R <= R_LOW: return "BEYOND-READOUT"
    return "MIXED"

def mcnemar_exact(b, c):
    """Two-sided exact McNemar on discordant counts b, c (p = 1 when b + c = 0)."""
    n = b + c
    if n == 0: return 1.0
    k = min(b, c); return float(min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n))

def paired_bootstrap(a, b, strata, reps=10000, seed=20260926):
    """95 % interval of mean(a) - mean(b) under paired resampling within strata (bool arrays)."""
    a, b, strata = np.asarray(a, float), np.asarray(b, float), np.asarray(strata)
    rng = np.random.default_rng(seed); d = a - b; out = np.empty(reps)
    groups = [np.where(strata == s)[0] for s in np.unique(strata)]
    for i in range(reps):
        idx = np.concatenate([g[rng.integers(len(g), size=len(g))] for g in groups]); out[i] = d[idx].mean()
    return float(d.mean()), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))

# ---------------- inputs ----------------
def study_paths():
    import attention_transfer_study as ATS
    return ATS, ATS.OUT

def first_guess(width):
    """The fixed-start first-iteration grid of a benchmark attention model on the 512 repair puzzles, clues restored."""
    ATS, STUDY = study_paths()
    grids = []
    for b in range(0, 512, BATCH):
        with np.load(STUDY / f"attention_{width}/repair/fixed/batch_{b:04d}.npz", allow_pickle=False) as d:
            pr, puz = d["pred"][0].astype(np.int32), d["puz"]; grids.append(np.where(puz == 0, pr, puz))
    return np.concatenate(grids)

def prepare():
    ATS, STUDY = study_paths()
    dst = OUT / "inputs/p1_sources.npz"
    if dst.exists(): print("prepared inputs already present"); return
    with np.load(STUDY / "inputs/repair.npz", allow_pickle=False) as d:
        ids, puz, sol, eqr, uni, counts, bins = (d[k] for k in ("ids", "puz", "sol", "eqr", "random", "wrong_count", "rating_bin"))
    n = len(ids); src = {}; stats = {}
    for j in range(DRAWS):
        g = np.zeros_like(sol); deficit = np.zeros(n, int)
        for i in range(n):
            g[i], _, deficit[i] = sample_legal_random(puz[i], sol[i], int(counts[i]), np.random.default_rng([SEED_LEGAL, int(ids[i]), 11, j]))
        src[f"legal_random_{j}"] = g; stats[f"legal_random_{j}_deficit_puzzles"] = int((deficit > 0).sum()); stats[f"legal_random_{j}_deficit_cells"] = int(deficit.sum())
        g2 = np.zeros_like(sol); fb = np.zeros(n, int); co = np.zeros(n, int)
        for i in range(n):
            cells = np.where(((eqr[i] != sol[i]) & (puz[i] == 0)).reshape(81))[0]
            g2[i], fb[i], co[i] = sample_legal_at_cells(puz[i], sol[i], cells, eqr[i], np.random.default_rng([SEED_LEGAL, int(ids[i]), 12, j]))
        src[f"legal_at_eqr_{j}"] = g2; stats[f"legal_at_eqr_{j}_fallback_cells"] = int(fb.sum()); stats[f"legal_at_eqr_{j}_fallback_wrong_cells"] = int(co.sum())
    for w in (128, 256):
        g = first_guess(w); src[f"guess_sa{w}"] = g; cw = ((g != sol) & (puz == 0)).sum((1, 2)); stats[f"guess_sa{w}_wrong_total"] = int(cw.sum())
        gm = np.zeros_like(sol); deficit = np.zeros(n, int)
        for i in range(n):
            gm[i], _, deficit[i] = sample_legal_random(puz[i], sol[i], int(cw[i]), np.random.default_rng([SEED_LEGAL, int(ids[i]), 13, w]))
        src[f"legal_match_sa{w}"] = gm; stats[f"legal_match_sa{w}_deficit_puzzles"] = int((deficit > 0).sum())
    for name, g in src.items():
        assert g.shape == sol.shape and np.isin(g, np.arange(1, 10)).all(), name
        assert np.array_equal(g[puz != 0], puz[puz != 0]), name + ": clues altered"
        share, nw = clue_conflict_share(g, puz, sol); stats[f"{name}_clue_conflict_share"] = share; stats[f"{name}_wrong_total"] = nw
        if name.startswith("legal_random") or name.startswith("legal_match"):
            assert share == 0.0 or np.isnan(share), name
    for j in range(DRAWS):
        cw = ((src[f"legal_random_{j}"] != sol) & (puz == 0)).sum((1, 2))
        assert np.all(cw == np.minimum(counts, counts)) or stats[f"legal_random_{j}_deficit_puzzles"] > 0
        assert np.all(cw <= counts) and np.all(cw + 0 >= 0)
        cells_eq = ((src[f"legal_at_eqr_{j}"] != sol) & (puz == 0)); cells_ref = ((eqr != sol) & (puz == 0))
        assert np.array_equal(cells_eq, cells_ref), "location match broken"
    stats["eqr_clue_conflict_share"] = clue_conflict_share(eqr, puz, sol)[0]
    stats["uniform_0_clue_conflict_share"] = clue_conflict_share(uni[0], puz, sol)[0]
    assert abs(stats["eqr_clue_conflict_share"] - 17 / 10185) < 1e-6 and 0.55 < stats["uniform_0_clue_conflict_share"] < 0.72
    ATS.write_npz(dst, ids=ids, puz=puz, sol=sol, eqr=eqr, uniform_0=uni[0], wrong_count=counts, rating_bin=bins, **src)
    ATS.write_json(OUT / "inputs/p1_sources.json", dict(created=utc(), seed=SEED_LEGAL, draws=DRAWS, conditions=list(REPAIR_CONDS), stats=stats,
                                                          sha256_repair_inputs=ATS.sha(STUDY / "inputs/repair.npz")))
    print(json.dumps(stats, indent=1))


# ---------------- the attention receivers (the paper's own pipeline, release code) ----------------
class R2:
    def __init__(self, width, out=OUT):
        import attention_transfer_study as ATS
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
        self.path = self.RELEASE / f"checkpoints/attention_{width}.npz"
        saved = E.load_ckpt(self.path)
        self.cfg = cfg = Config(**saved["config"])
        inventory = json.loads((self.RELEASE / "evidence/models.json").read_text())[f"attention_{width}"]
        assert saved["step"] == inventory["selected_step"]
        assert cfg.dec_width == width and cfg.dec_token_mixer == cfg.dec_coupling_kind == "attn"
        assert cfg.cell_kind == "dec" and not cfg.dec_commit and cfg.loss_kind == "stablemax" and jax.default_backend() == "cpu"
        self.params = jax.tree.map(jnp.asarray, saved["state_ema"]["model"])
        self.tv = jnp.asarray(saved["state_ema"]["table"][0])
        self.eta, self.eta_z = map(float, M.eq_etas(self.params, cfg))
        assert self.eta_z == 1.0 and EV.coupled_ab(self.params, cfg) is None
        self.layout = cfg.sudoku_layout or "origin"
        self.z0 = DC.z0(cfg, 81)
        self.lm = np.asarray(self.params["dec"]["lm_head"], np.float64); assert self.lm.shape == (width,)
        self.embed = jax.jit(jax.vmap(lambda g: DC.embed_answer(self.params["dec"], cfg, g)))
        self.meta = dict(width=width, checkpoint_step=saved["step"], ema=True, checkpoint_sha256=ATS.sha(self.path),
                         source_checkpoint_sha256=inventory["source_checkpoint_sha256"], source_manifest_sha256=self.source_hash,
                         batch=BATCH, steps=STEPS, backend=jax.default_backend(), jax=jax.__version__, numpy=np.__version__,
                         python=platform.python_version(), operating_system=platform.platform(), process_id=os.getpid(),
                         study="rebuttal_20260926", pipeline="tools/rebuttal_p1p2.py over tools/attention_transfer_study.py")
        self.out = out; self.dir = out / f"attention_{width}"; self.dir.mkdir(parents=True, exist_ok=True)
        ATS.write_json(self.dir / "runtime.json", self.meta)

    # --- the study's loop, with the three new modes; every other line is the study's ---
    def trajectory(self, puz, ids, initial=None, branch=None, mode="intact", steps=STEPS, progress=None):
        jax, jnp, EV = self.jax, self.jnp, self.EV
        x = EV.place_batch(puz, self.layout)
        void = jax.nn.one_hot(jnp.full((9, 9), self.G.VOID, jnp.int32), 11).transpose(2, 0, 1)
        y = jnp.broadcast_to(void, (len(puz),) + void.shape)
        z = initial; pr = None
        logits, preds, residuals, norms = [], [], [], []
        first_state = None; max_shift = 0.0; readout_agree = []
        z0h = np.asarray(self.z0[0], np.float64)                      # (F, S, w) broadcast buffer
        z0h_perp = project(z0h, self.lm)[1]
        for t in range(steps):
            if t == 0 and branch is not None:
                y, z, lg, pr, rr, nn = branch
            else:
                if t > 0 and mode == "reset_fast": z = z.at[:, 1].set(self.z0[1])
                if t > 0 and mode == "reset_slow": z = z.at[:, 0].set(self.z0[0])
                if t > 0 and mode == "reencode_slow":
                    grid = np.where(np.asarray(puz) == 0, pr.astype(np.int32), np.asarray(puz)).astype(np.int32)
                    h_new = self.embed(jnp.asarray(grid, jnp.int32))
                    z = z.at[:, 0].set(h_new)
                    own = np.asarray(jnp.einsum("bfsw,w->bsf", h_new, jnp.asarray(self.lm, jnp.float32)))     # immediate readout of the re-encoded state
                    dec = (own.argmax(-1) + 1).reshape(len(puz), 9, 9)
                    readout_agree.append(float(((dec == grid) | (np.asarray(puz) != 0)).mean()))
                if t > 0 and mode in ("null_h0", "null_random"):
                    h = np.asarray(z[:, 0], np.float64)                  # (B, F, S, w)
                    par, perp = project(h, self.lm)
                    if mode == "null_h0":
                        new = np.broadcast_to(z0h_perp, perp.shape)
                    else:
                        new = np.stack([norm_matched_random(perp[b], self.lm, np.random.default_rng([SEED_LEGAL, int(ids[b]), 21, t])) for b in range(len(ids))])
                    h2 = par + new
                    shift = float(np.abs(h2 @ self.lm - h @ self.lm).max()); max_shift = max(max_shift, shift)
                    assert shift < 1e-2, f"score shift {shift}"
                    z = z.at[:, 0].set(jnp.asarray(h2, jnp.float32))
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
                self.status("running", **progress, iteration=t+1)
        extra = dict(max_score_shift=max_shift, reencode_readout_agreement=(float(np.mean(readout_agree)) if readout_agree else None))
        return dict(logits=np.stack(logits), pred=np.stack(preds), carry_residual=np.stack(residuals), carry_rms=np.stack(norms)), first_state, extra

    def starting_state(self, supplied):
        jnp = self.jnp
        high = self.embed(jnp.asarray(supplied, jnp.int32))
        return jnp.stack([high, jnp.broadcast_to(self.z0[1], high.shape)], axis=1)

    def status(self, state, **values):
        self.ATS.write_json(self.dir / "status.json", dict(state=state, updated=utc(), pid=os.getpid(), width=self.width, **values))

    def save(self, path, values, puz, sol, ids, group, condition, started, cpu_started, extra, supplied=None):
        elapsed = time.monotonic() - started; cpu = time.process_time() - cpu_started
        metadata = dict(self.meta, group=group, condition=condition, n=len(ids), created=utc(), seconds=elapsed, cpu_seconds=cpu,
                        average_cpu_cores=cpu / max(elapsed, 1e-9), peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, **extra)
        exact = (values["pred"] == sol[None]).all((2, 3))
        if supplied is not None: values["supplied_grid"] = supplied
        self.ATS.write_npz(path, **values, ids=ids, puz=puz, sol=sol, exact=exact, metadata=np.array(json.dumps(metadata)))
        self.ATS.write_json(path.with_suffix(".sha256.json"), dict(sha256=self.ATS.sha(path)))
        self.ATS.log("chunk_complete", width=self.width, group=group, condition=condition, n=len(ids), final=int(exact[-1].sum()), seconds=round(elapsed, 1), **extra)

    def cached(self, path, ids, group, condition):
        if not path.exists(): return False
        side = path.with_suffix(".sha256.json")
        if not side.exists() or json.loads(side.read_text())["sha256"] != self.ATS.sha(path): raise RuntimeError(f"Incomplete or altered output: {path}")
        with np.load(path, allow_pickle=False) as d:
            meta = json.loads(str(d["metadata"]))
            assert meta["source_manifest_sha256"] == self.source_hash and meta["checkpoint_sha256"] == self.meta["checkpoint_sha256"]
            assert meta["group"] == group and meta["condition"] == condition and np.array_equal(d["ids"], ids)
        return True

    def gate(self, smoke):
        """The study's intact interventions chunk 0 reproduced bitwise by this loop (the smoke checks the shared first step only)."""
        with np.load(self.STUDY / f"attention_{self.width}/interventions/intact/batch_0000.npz", allow_pickle=False) as d:
            ids, puz, sol, ref = d["ids"], d["puz"], d["sol"], d["logits"]
        n = 16 if smoke else len(ids)
        _, branch, _ = self.trajectory(puz[:n], ids[:n], steps=1)
        assert np.array_equal(branch[2], ref[0, :n]), "first-step logits differ from the study"
        if not smoke:
            values, _, _ = self.trajectory(puz, ids, branch=branch, mode="intact")
            assert np.array_equal(values["logits"], ref), "intact continuation differs from the study"
        self.ATS.write_json(self.dir / ("gate_smoke.json" if smoke else "gate.json"), dict(passed=True, created=utc(), n=n, bitwise_logits=True))
        self.ATS.log("gate_passed", width=self.width, n=n, smoke=smoke)

    def state_group(self, smoke):
        group = "state"
        with np.load(self.STUDY / "inputs/interventions.npz", allow_pickle=False) as d: ids, puz, sol = d["ids"], d["puz"], d["sol"]
        if smoke: ids, puz, sol = ids[:16], puz[:16], sol[:16]
        steps = 2 if smoke else STEPS
        for b in range(0, len(ids), BATCH):
            sl = slice(b, b + BATCH); ii, pp, ss = ids[sl], puz[sl], sol[sl]
            paths = {c: self.dir / group / c / f"batch_{b:04d}.npz" for c in STATE_CONDS}
            done = {c: self.cached(paths[c], ii, group, c) for c in STATE_CONDS}
            if all(done.values()): continue
            _, branch, _ = self.trajectory(pp, ii, steps=1)
            if not smoke:
                with np.load(self.STUDY / f"attention_{self.width}/interventions/intact/batch_{b:04d}.npz", allow_pickle=False) as d:
                    assert np.array_equal(d["logits"][0], branch[2]), "shared first state differs from the study"
            for c in STATE_CONDS:
                if done[c]: continue
                started, cpu = time.monotonic(), time.process_time()
                values, own, extra = self.trajectory(pp, ii, branch=branch, mode=c, steps=steps, progress=dict(group=group, condition=c, batch_start=b))
                assert own[1] is branch[1] and np.array_equal(values["logits"][0], branch[2])
                self.save(paths[c], values, pp, ss, ii, group, c, started, cpu, extra)

    def repair_group(self, smoke):
        group = "repair"
        with np.load(OUT / "inputs/p1_sources.npz", allow_pickle=False) as d: data = {k: d[k] for k in d.files}
        ids, puz, sol = data["ids"], data["puz"], data["sol"]
        n = 16 if smoke else len(ids); steps = 2 if smoke else STEPS
        for c in REPAIR_CONDS:
            for b in range(0, n, BATCH):
                sl = slice(b, min(b + BATCH, n)); ii, pp, ss, gg = ids[sl], puz[sl], sol[sl], data[c][sl]
                path = self.dir / group / c / f"batch_{b:04d}.npz"
                if self.cached(path, ii, group, c): continue
                started, cpu = time.monotonic(), time.process_time()
                values, _, extra = self.trajectory(pp, ii, initial=self.starting_state(gg), steps=steps, progress=dict(group=group, condition=c, batch_start=b))
                self.save(path, values, pp, ss, ii, group, c, started, cpu, extra, supplied=gg)

    def run(self, group, smoke):
        try:
            self.gate(smoke)
            if group in ("state", "all"): self.state_group(smoke)
            if group in ("repair", "all"): self.repair_group(smoke)
            self.ATS.verify_sources(); self.status("inference_complete", group=group, completed=utc()); self.ATS.log("model_complete", width=self.width, group=group)
        except BaseException as exc:
            self.status("failed", error=repr(exc)); raise


# ---------------- the MLP 192 and EqR receivers (the paper's original pipeline) ----------------
def port(receiver, smoke, out=OUT):
    import lens_repair_radius as RR
    RR.T_TOTAL = 2 if smoke else STEPS
    dst = out / f"port_{receiver}.npz"
    if dst.exists(): print(f"SKIP {dst.name} (done)"); return
    with np.load(OUT / "inputs/p1_sources.npz", allow_pickle=False) as d: data = {k: d[k] for k in d.files}
    ids, puz, sol = data["ids"], data["puz"], data["sol"]
    n = 16 if smoke else len(ids)
    m = RR.Model(receiver); f = RR.embedder(m); t0 = time.time()
    P = np.tile(np.arange(n), len(PORT_SOURCES)); SRC = np.repeat(np.arange(len(PORT_SOURCES)), n)
    start = np.concatenate([data[s][:n] for s in PORT_SOURCES]).astype(np.int32)
    e0 = RR.err_frac(start, sol[P], puz[P] == 0)
    a = RR.run_rows(m, puz[P].astype(np.int32), sol[P].astype(np.int32), lambda rows: f(start[rows]), None, f"{receiver} P1")
    meta = dict(receiver=receiver, ckpt=RR.MODELS[receiver][0], step=m.step, n=n, t_total=RR.T_TOTAL, sources=list(PORT_SOURCES), created=utc(), smoke=smoke)
    np.savez(out / f"port_{receiver}.tmp.npz", idx=ids[:n], puzzle=P, source=SRC, sources=np.asarray(PORT_SOURCES), e0=e0, meta=json.dumps(meta), wall=time.time() - t0, **a)
    os.replace(out / f"port_{receiver}.tmp.npz", dst); print(f"DONE {dst.name} ({time.time() - t0:.0f}s)", flush=True)


# ---------------- report (the registered rules) ----------------
def load_group(width, group, conds, base):
    rows = {}
    for c in conds:
        parts = sorted((base / f"attention_{width}/{group}/{c}").glob("batch_*.npz"))
        if not parts: continue
        ids, e1, ex, shift = [], [], [], []
        for p in parts:
            with np.load(p, allow_pickle=False) as d:
                pred, sol, puz = d["pred"], d["sol"], d["puz"]; ng = puz == 0
                ids.append(d["ids"]); e1.append(((pred[0] != sol) & ng).sum((1, 2)) / np.maximum(ng.sum((1, 2)), 1)); ex.append(d["exact"][-1])
                shift.append(json.loads(str(d["metadata"])).get("max_score_shift", 0.0))
        rows[c] = dict(ids=np.concatenate(ids), e1=np.concatenate(e1), exact=np.concatenate(ex), max_shift=max(shift))
    return rows

def report(base=OUT):
    import attention_transfer_study as ATS
    STUDY = ATS.OUT; out = {}
    with np.load(OUT / "inputs/p1_sources.npz", allow_pickle=False) as d: eligible = d["wrong_count"] > 0; bins = d["rating_bin"]; ids_all = d["ids"]
    lines = ["P1 / P2 REPORT (rules registered 2026-09-26; see the note)"]
    for w in WIDTHS:
        ref = load_group(w, "repair", ("eqr", "fixed") + tuple(f"random_{j}" for j in range(DRAWS)), STUDY)
        new = load_group(w, "repair", REPAIR_CONDS, base)
        if not ref or "eqr" not in ref or not new: lines.append(f"attention_{w}: repair not complete"); continue
        el = eligible; e1 = lambda r: float(r["e1"][el].mean())
        e_eqr, e_uni = e1(ref["eqr"]), float(np.mean([e1(ref[f"random_{j}"]) for j in range(DRAWS) if f"random_{j}" in ref]))
        res = dict(e1_eqr=e_eqr, e1_uniform=e_uni)
        for fam in ("legal_random", "legal_at_eqr"):
            ks = [f"{fam}_{j}" for j in range(DRAWS) if f"{fam}_{j}" in new]
            if not ks: continue
            e_leg = float(np.mean([e1(new[k]) for k in ks])); G = gap_closure(e_eqr, e_uni, e_leg)
            ex_leg = np.mean([new[k]["exact"][el] for k in ks], axis=0); ex_eqr = ref["eqr"]["exact"][el].astype(float)
            mean, lo, hi = paired_bootstrap(ex_leg, ex_eqr, bins[el])
            b = int(((new[ks[0]]["exact"][el]) & ~ref["eqr"]["exact"][el]).sum()); c = int((~new[ks[0]]["exact"][el] & ref["eqr"]["exact"][el]).sum())
            res[fam] = dict(e1=e_leg, G=G, letter=letter_p1(G), endpoint_advantage_pp=100 * mean, ci=[100 * lo, 100 * hi], draw0_mcnemar_p=mcnemar_exact(b, c), draws=len(ks))
            lines.append(f"attention_{w} {fam}: E1 eqr {100*e_eqr:.2f} uniform {100*e_uni:.2f} legal {100*e_leg:.2f} | G {G if G is None else round(G,3)} -> {letter_p1(G)} | endpoint legal-eqr {100*mean:+.2f} pp [{100*lo:+.2f}, {100*hi:+.2f}] draw0 McNemar p {mcnemar_exact(b, c):.3f}")
        for src in ("sa128", "sa256"):
            g, mkey = f"guess_{src}", f"legal_match_{src}"
            if g in new and mkey in new:
                with np.load(OUT / "inputs/p1_sources.npz", allow_pickle=False) as d: cw = ((d[g] != d["sol"]) & (d["puz"] == 0)).sum((1, 2)) > 0
                eg, em = float(new[g]["e1"][cw].mean()), float(new[mkey]["e1"][cw].mean())
                res[g] = dict(e1_guess=eg, e1_legal_match=em, exact_guess=int(new[g]["exact"][cw].sum()), exact_match=int(new[mkey]["exact"][cw].sum()), n=int(cw.sum()))
                lines.append(f"attention_{w} {g} (exploratory, n {int(cw.sum())}): E1 guess {100*eg:.2f} vs legal-matched {100*em:.2f}; exact16 {int(new[g]['exact'][cw].sum())} vs {int(new[mkey]['exact'][cw].sum())}")
        st = load_group(w, "state", STATE_CONDS, base); ref_s = load_group(w, "interventions", ("intact", "reset_slow", "reset_fast"), STUDY)
        if st and ref_s:
            I, S = int(ref_s["intact"]["exact"].sum()), int(ref_s["reset_slow"]["exact"].sum()); res["state"] = dict(intact=I, reset_slow=S)
            for c in STATE_CONDS:
                if c not in st: continue
                M = int(st[c]["exact"].sum()); R = recovery(I, M, S); res["state"][c] = dict(exact=M, R=R, letter=letter_p2(R), max_score_shift=st[c]["max_shift"])
                lines.append(f"attention_{w} state {c}: exact16 {M} (intact {I}, slow reset {S}) | R {R if R is None else round(R,3)} -> {letter_p2(R)} | max score shift {st[c]['max_shift']:.2e}")
        out[f"attention_{w}"] = res
    for rec in ("C5", "EQR"):
        p = base / f"port_{rec}.npz"
        if not p.exists(): lines.append(f"port {rec}: not complete"); continue
        d = np.load(p, allow_pickle=True); srcs = list(d["sources"]); P, S = d["puzzle"], d["source"]; el = eligible[d["idx"]]
        e1 = d["e"][:, 0]; ex = d["ex16"]
        def m(name, arr): k = srcs.index(name); sel = (S == k) & el[P]; return float(np.nanmean(arr[sel]))
        e_eqr, e_uni = m("eqr", e1), m("uniform_0", e1); res = dict(e1_eqr=e_eqr, e1_uniform=e_uni)
        for fam in ("legal_random", "legal_at_eqr"):
            e_leg = float(np.mean([m(f"{fam}_{j}", e1) for j in range(DRAWS)])); G = gap_closure(e_eqr, e_uni, e_leg)
            res[fam] = dict(e1=e_leg, G=G, letter=letter_p1(G), exact_eqr=m("eqr", ex.astype(float)), exact_legal=float(np.mean([m(f"{fam}_{j}", ex.astype(float)) for j in range(DRAWS)])))
            lines.append(f"port {rec} {fam}: E1 eqr {100*e_eqr:.2f} uniform {100*e_uni:.2f} legal {100*e_leg:.2f} | G {G if G is None else round(G,3)} -> {letter_p1(G)} | exact16 eqr {100*res[fam]['exact_eqr']:.1f} % legal {100*res[fam]['exact_legal']:.1f} %")
        out[f"port_{rec}"] = res
    (base / "report.txt").write_text("\n".join(lines) + "\n"); (base / "report.json").write_text(json.dumps(out, indent=1, default=float))
    print("\n".join(lines))


# ---------------- selftest ----------------
def selftest():
    sol = np.array([[(3 * (r % 3) + r // 3 + c) % 9 + 1 for c in range(9)] for r in range(9)])
    assert all(sorted(sol[r]) == list(range(1, 10)) for r in range(9)) and all(sorted(sol[:, c]) == list(range(1, 10)) for c in range(9))
    assert all(sorted(sol[3*br:3*br+3, 3*bc:3*bc+3].reshape(-1)) == list(range(1, 10)) for br in range(3) for bc in range(3))
    rng = np.random.default_rng(0); puz = np.where(rng.random((9, 9)) < 0.4, sol, 0)
    # legal digits exclude the solution and every given in the cell's units
    for cell in np.where(puz.reshape(81) == 0)[0][:20]:
        opts = legal_wrong_digits(puz, sol, int(cell)); r, c = divmod(int(cell), 9)
        givens = set(puz[r]) | set(puz[:, c]) | set(puz[3*(r//3):3*(r//3)+3, 3*(c//3):3*(c//3)+3].reshape(-1)); givens.discard(0)
        assert sol[r, c] not in opts and not (set(opts) & givens) and set(opts) <= set(range(1, 10))
    g, take, deficit = sample_legal_random(puz, sol, 10, np.random.default_rng(1))
    assert take == 10 and deficit == 0 and ((g != sol) & (puz == 0)).sum() == 10 and np.array_equal(g[puz != 0], puz[puz != 0])
    assert clue_conflict_share(g, puz, sol)[0] == 0.0
    # deficit path: a puzzle whose only empty cells have no legal wrong digit
    full = sol.copy(); full[0, 0] = 0; full[0, 1] = 0            # two empties; every other digit of their units is given -> no legal wrong digit
    g2, take2, def2 = sample_legal_random(full, sol, 2, np.random.default_rng(2)); assert take2 == 0 and def2 == 2 and np.array_equal(g2, sol)
    # location-matched sampling and the fallback
    eqr = sol.copy(); cells = np.where(puz.reshape(81) == 0)[0][:5]
    for c in cells: eqr.reshape(81)[c] = legal_wrong_digits(puz, sol, int(c))[0]
    g3, fb, co = sample_legal_at_cells(puz, sol, cells, eqr, np.random.default_rng(3))
    singles = sum(len(legal_wrong_digits(puz, sol, int(c))) == 1 for c in cells)
    assert fb == singles and np.array_equal((g3 != sol), (eqr != sol)) and clue_conflict_share(g3, puz, sol)[0] == 0.0
    assert all(g3.reshape(81)[c] != eqr.reshape(81)[c] or len(legal_wrong_digits(puz, sol, int(c))) == 1 for c in cells)
    g4, fb4, _ = sample_legal_at_cells(full, sol, [0, 1], np.where(full == 0, 9, sol), np.random.default_rng(4)); assert fb4 == 2
    # uniform corruptions do conflict with givens
    uni = sol.copy().reshape(81); emp = np.where(puz.reshape(81) == 0)[0]
    for c in emp[:15]: uni[c] = (sol.reshape(81)[c] % 9) + 1
    assert clue_conflict_share(uni.reshape(9, 9), puz, sol)[0] > 0.2
    # projection preserves scores and reconstructs
    v = np.random.default_rng(5).standard_normal(16); h = np.random.default_rng(6).standard_normal((3, 4, 16))
    par, perp = project(h, v); assert np.allclose(par + perp, h) and np.allclose(perp @ v, 0, atol=1e-9) and np.allclose(par @ v, h @ v)
    rnd = norm_matched_random(perp, v, np.random.default_rng(7)); assert np.allclose(rnd @ v, 0, atol=1e-9) and np.allclose(np.linalg.norm(rnd, axis=-1), np.linalg.norm(perp, axis=-1))
    assert not np.allclose((par + rnd) @ v - h @ v, 1.0) and np.allclose((par + rnd) @ v, h @ v)
    bad = h - par                                                    # a mutant edit that drops the parallel part changes the scores
    assert not np.allclose(bad @ v, h @ v)
    # rules and letters, including boundaries and the floors
    assert letter_p1(gap_closure(0.36, 0.04, 0.30)) == "REFUTABILITY-MOSTLY" and letter_p1(gap_closure(0.36, 0.04, 0.10)) == "CONFIGURATION-BEYOND"
    assert letter_p1(gap_closure(0.36, 0.04, 0.20)) == "MIXED" and letter_p1(gap_closure(0.06, 0.04, 0.05)) == "UNDEFINED"
    assert abs(gap_closure(0.36, 0.04, 0.20) - 0.5) < 1e-12 and letter_p1(gap_closure(0.36, 0.04, 0.36 * 0 + 0.04 + 0.67 * 0.32)) == "REFUTABILITY-MOSTLY"
    assert letter_p2(recovery(250, 245, 58)) == "ANSWER-ONLY" and letter_p2(recovery(250, 100, 58)) == "BEYOND-READOUT" and letter_p2(recovery(250, 180, 58)) == "MIXED"
    assert recovery(250, 240, 240) is None and letter_p2(None) == "UNDEFINED" and abs(recovery(250, 154, 58) - 0.5) < 1e-12
    assert mcnemar_exact(0, 0) == 1.0 and abs(mcnemar_exact(5, 0) - 2 / 32) < 1e-12 and mcnemar_exact(3, 3) == 1.0
    m, lo, hi = paired_bootstrap(np.array([1, 1, 1, 0, 1, 1]), np.array([0, 1, 0, 0, 1, 0]), np.array([0, 0, 0, 1, 1, 1]), reps=500); assert lo <= m <= hi and m == 0.5
    print("selftest OK: 40 assertions (legal digits, feasibility and fallback rules, conflict measure, score-preserving projection with a mutant, letters at their boundaries and floors, McNemar, bootstrap)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", nargs="?", choices=["prepare", "run", "port", "report"])
    ap.add_argument("--selftest", action="store_true"); ap.add_argument("--width", type=int, choices=WIDTHS)
    ap.add_argument("--group", default="all", choices=["state", "repair", "all"]); ap.add_argument("--receiver", choices=["C5", "EQR"])
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.selftest: selftest(); return
    out = Path(a.out) if a.out else OUT
    if a.action == "prepare": prepare()
    elif a.action == "run":
        if a.width is None: ap.error("--width required")
        R2(a.width, out).run(a.group, a.smoke)
    elif a.action == "port":
        if a.receiver is None: ap.error("--receiver required")
        port(a.receiver, a.smoke, out)
    elif a.action == "report": report(out)
    else: ap.print_help()


if __name__ == "__main__":
    main()
