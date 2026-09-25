#!/usr/bin/env python3
"""Resumable, source-pinned inference on the three final attention checkpoints.

Uses the frozen release read-only. New inputs and outputs live in a separate
research directory. See its PROTOCOL.md for hypotheses, samples and limitations.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "paper/code"
OUT = ROOT / "runs/analysis/attention_transfer_20260923"
OLD = ROOT / "runs/analysis/paper_update_20260920/corpus/selected_stablemax"
COUPLING = ROOT / "runs/analysis/coupling_matched_error_20260920/cross_model_coupling/SA128"
WIDTHS = (128, 192, 256)
STEPS, BATCH, DRAWS = 16, 128, 4


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(2**20), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def write_npz(path, **values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp.npz")
    np.savez_compressed(temp, **values)
    temp.replace(path)


def log(event, **values):
    print(json.dumps(dict(time=utc(), event=event, **values), allow_nan=False), flush=True)


def corrupt(solution, puzzle, ids, counts, draw):
    out = solution.copy()
    for i, pid in enumerate(ids):
        seed = [20260917, int(pid), 77] if draw == 0 else [20260923, int(pid), 77, draw]
        rng = np.random.default_rng(seed)
        cells = np.argwhere(puzzle[i] == 0)
        k = int(counts[i])
        chosen = cells[rng.choice(len(cells), size=k, replace=False)] if k else []
        for row, col in chosen:
            out[i, row, col] = (solution[i, row, col] - 1 + rng.integers(1, 9)) % 9 + 1
    return out


def source_files():
    files = [Path(__file__), ROOT / "tools/analyze_attention_transfer.py", OUT / "PROTOCOL.md",
             RELEASE / "evidence/models.json", RELEASE / "evidence/hardening/repair_context.npz",
             RELEASE / "evidence/interventions/matched_repair.npz",
             RELEASE / "evidence/interventions/coupling_extension.npz",
             ROOT / "data/sudoku_extreme/sudoku_extreme_seed0_mon512.npz",
             ROOT / "data/sudoku_extreme/sudoku_extreme_seed0_val10k.npz",
             RELEASE / "tools/eval_sudoku_extreme.py", RELEASE / "tools/probe.py",
             OLD / "sources.json", OLD / "environment.json",
             COUPLING / "control.npz", COUPLING / "after_first_off.npz", COUPLING / "metadata.json"]
    files += list((RELEASE / "src").rglob("*.py"))
    for w in WIDTHS:
        files += [RELEASE / f"checkpoints/attention_{w}.npz", RELEASE / f"configs/attention_{w}.json",
                  RELEASE / f"evidence/selection/attention_{w}_k128.npz", OLD / f"SA{w}_selected.npz"]
    return sorted(set(files))


def prepare():
    if (OUT / "source_manifest.json").exists():
        verify_sources()
        log("prepared_inputs_already_present")
        return
    with np.load(RELEASE / "evidence/hardening/repair_context.npz", allow_pickle=False) as z:
        ids, puz, sol = z["ids"], z["puzzle"].astype(np.int32), z["solution"].astype(np.int32)
        eqr = z["eqr_first_grid"].copy()
    assert len(ids) == 512 and np.array_equal(np.sort(ids), ids) and len(set(ids)) == 512
    eqr[puz != 0] = puz[puz != 0]
    counts = ((eqr != sol) & (puz == 0)).sum((1, 2))
    assert np.isin(eqr, np.arange(1, 10)).all() and int((counts > 0).sum()) == 438
    with np.load(RELEASE / "evidence/interventions/matched_repair.npz", allow_pickle=False) as z:
        assert np.array_equal(ids, z["ids"]) and np.array_equal(counts, z["EQR_wrong_count"])
        bins = z["rating_bin"].copy()
    with np.load(RELEASE / "evidence/interventions/coupling_extension.npz", allow_pickle=False) as z:
        assert np.array_equal(ids[::2], z["ids"])
    with np.load(ROOT / "data/sudoku_extreme/sudoku_extreme_seed0_mon512.npz", allow_pickle=False) as z:
        assert np.array_equal(puz, z["test_q"][ids]) and np.array_equal(sol, z["test_a"][ids])
    random = np.stack([corrupt(sol, puz, ids, counts, d) for d in range(DRAWS)])
    assert np.all(((random != sol[None]) & (puz[None] == 0)).sum((2, 3)) == counts[None])
    assert all(np.array_equal(x[puz != 0], puz[puz != 0]) for x in random)
    write_npz(OUT / "inputs/repair.npz", ids=ids, puz=puz, sol=sol, eqr=eqr,
              random=random, wrong_count=counts, rating_bin=bins)
    write_npz(OUT / "inputs/interventions.npz", ids=ids[::2], puz=puz[::2], sol=sol[::2])
    with np.load(ROOT / "data/sudoku_extreme/sudoku_extreme_seed0_val10k.npz", allow_pickle=False) as z:
        vp, vs = z["val_q"][:128].astype(np.int32), z["val_a"][:128].astype(np.int32)
    write_npz(OUT / "inputs/initialization.npz", ids=np.arange(128), puz=vp, sol=vs)
    files = source_files() + list((OUT / "inputs").glob("*.npz"))
    manifest = dict(created=utc(), sha256={str(p.relative_to(ROOT)): sha(p) for p in files})
    write_json(OUT / "source_manifest.json", manifest)
    protected = [ROOT / "paper/final draft/FULL_ACCOUNT.md", ROOT / "paper/code/MANIFEST.json"]
    protected += list((ROOT / "paper/final draft").rglob("*.tex"))
    write_json(OUT / "protected_files.json", {str(p.relative_to(ROOT)): sha(p) for p in protected})
    log("prepared", repair_n=512, eligible_n=438, intervention_n=256, initialization_n=128,
        corruption_draws=DRAWS, sources=len(files))


def verify_sources():
    record = json.loads((OUT / "source_manifest.json").read_text())
    for name, digest in record["sha256"].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError(f"Pinned source changed: {name}")
    return sha(OUT / "source_manifest.json")


def zero_messages(tree):
    if isinstance(tree, dict):
        return {k: np.zeros_like(v) if k == "fc" else zero_messages(v) for k, v in tree.items()}
    if isinstance(tree, list):
        return [zero_messages(v) for v in tree]
    if isinstance(tree, tuple):
        return tuple(zero_messages(v) for v in tree)
    return tree


class Runner:
    def __init__(self, width):
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        sys.path[:0] = [str(RELEASE / "src"), str(RELEASE / "tools")]
        import jax
        import jax.numpy as jnp
        from qhrrn2 import episodic as E, model as M, grid as G, dec_cell as DC
        from qhrrn2.config import Config
        import eval_sudoku_extreme as EV
        self.jax, self.jnp, self.EV, self.DC, self.G = jax, jnp, EV, DC, G
        self.width = width
        self.source_hash = verify_sources()
        self.path = RELEASE / f"checkpoints/attention_{width}.npz"
        saved = E.load_ckpt(self.path)
        self.cfg = cfg = Config(**saved["config"])
        inventory = json.loads((RELEASE / "evidence/models.json").read_text())[f"attention_{width}"]
        assert saved["step"] == inventory["selected_step"]
        assert cfg.dec_width == width and cfg.dec_token_mixer == cfg.dec_coupling_kind == "attn"
        assert cfg.cell_kind == "dec" and not cfg.dec_commit and cfg.loss_kind == "stablemax"
        assert cfg.canvas == 9 and jax.default_backend() == "cpu"
        self.params = jax.tree.map(jnp.asarray, saved["state_ema"]["model"])
        self.tv = jnp.asarray(saved["state_ema"]["table"][0])
        self.eta, self.eta_z = map(float, M.eq_etas(self.params, cfg))
        self.ab = EV.coupled_ab(self.params, cfg)
        assert self.eta_z == 1.0 and self.ab is None
        self.layout = cfg.sudoku_layout or "origin"
        self.z0 = DC.z0(cfg, 81)
        self.off = zero_messages(self.params)
        pairs1 = jax.tree_util.tree_flatten_with_path(self.params)[0]
        pairs2 = jax.tree_util.tree_flatten_with_path(self.off)[0]
        changed = []
        for (pa, a), (pb, b) in zip(pairs1, pairs2, strict=True):
            assert pa == pb
            if not np.array_equal(a, b):
                assert getattr(pa[-1], "key", None) == "fc" and not np.asarray(b).any()
                changed.append(str(pa))
        assert len(changed) == cfg.trm_layers == 2
        self.embed = jax.jit(jax.vmap(lambda g: DC.embed_answer(self.params["dec"], cfg, g)))
        self.meta = dict(width=width, checkpoint_step=saved["step"], ema=True,
                         checkpoint_sha256=sha(self.path), source_checkpoint_sha256=inventory["source_checkpoint_sha256"],
                         source_manifest_sha256=self.source_hash, batch=BATCH, steps=STEPS,
                         backend=jax.default_backend(), devices=[str(d) for d in jax.devices()],
                         jax=jax.__version__, numpy=np.__version__, python=platform.python_version(),
                         matmul_precision=str(jax.config.jax_default_matmul_precision),
                         changed_message_leaves=changed, full_vocabulary=True,
                         operating_system=platform.platform(), process_id=os.getpid())
        self.dir = OUT / f"attention_{width}"
        self.dir.mkdir(exist_ok=True)
        write_json(self.dir / "runtime.json", self.meta)

    def starting_state(self, supplied, ids, mode, draw=0):
        jnp = self.jnp
        n = len(ids)
        if mode == "fixed":
            return None
        if mode in ("independent", "shared"):
            tensors = [self.EV.mi_z0(4242, 0 if mode == "shared" else int(i), draw,
                                   tuple(self.z0.shape), 1.0) for i in ids]
            return jnp.asarray(np.stack(tensors))
        high = self.embed(jnp.asarray(supplied, jnp.int32))
        return jnp.stack([high, jnp.broadcast_to(self.z0[1], high.shape)], axis=1)

    def trajectory(self, puz, ids, initial=None, branch=None, mode="intact", steps=STEPS, progress=None):
        jax, jnp, EV = self.jax, self.jnp, self.EV
        x = EV.place_batch(puz, self.layout)
        void = jax.nn.one_hot(jnp.full((9, 9), self.G.VOID, jnp.int32), 11).transpose(2, 0, 1)
        y = jnp.broadcast_to(void, (len(puz),) + void.shape)
        z = initial
        logits, preds, residuals, norms = [], [], [], []
        first_state = None
        for t in range(steps):
            if t == 0 and branch is not None:
                y, z, lg, pr, rr, nn = branch
            else:
                if t > 0 and mode == "reset_fast":
                    z = z.at[:, 1].set(self.z0[1])
                if t > 0 and mode == "reset_slow":
                    z = z.at[:, 0].set(self.z0[0])
                active = self.off if mode == "messages_off" and t > 0 else self.params
                first = z is None
                old = z
                lg_dev, zf = EV._step(self.cfg, 1., 0., first)(active, x, y, self.tv,
                                                            jnp.zeros(1) if first else z)
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
        return dict(logits=np.stack(logits), pred=np.stack(preds),
                    carry_residual=np.stack(residuals), carry_rms=np.stack(norms)), first_state

    def status(self, state, **values):
        write_json(self.dir / "status.json", dict(state=state, updated=utc(), pid=os.getpid(), width=self.width, **values))

    def save(self, path, values, puz, sol, ids, group, condition, started, cpu_started, supplied=None):
        elapsed = time.monotonic() - started
        cpu = time.process_time() - cpu_started
        metadata = dict(self.meta, group=group, condition=condition, n=len(ids),
                        created=utc(), seconds=elapsed, cpu_seconds=cpu,
                        average_cpu_cores=cpu/max(elapsed, 1e-9),
                        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                        residual_definition="Mean absolute change per carry; reset changes measured after reset; first fixed update measured against fixed buffers")
        exact = (values["pred"] == sol[None]).all((2, 3))
        if supplied is not None:
            values["supplied_grid"] = supplied
        write_npz(path, **values, ids=ids, puz=puz, sol=sol, exact=exact,
                  metadata=np.array(json.dumps(metadata)))
        write_json(path.with_suffix(".sha256.json"), dict(sha256=sha(path)))
        log("chunk_complete", width=self.width, group=group, condition=condition,
            n=len(ids), final=int(exact[-1].sum()), seconds=round(elapsed, 2),
            average_cpu_cores=round(metadata["average_cpu_cores"], 2))

    def cached(self, path, ids, group, condition):
        if not path.exists():
            return False
        side = path.with_suffix(".sha256.json")
        if not side.exists() or json.loads(side.read_text())["sha256"] != sha(path):
            raise RuntimeError(f"Incomplete or altered output: {path}")
        with np.load(path, allow_pickle=False) as d:
            meta = json.loads(str(d["metadata"]))
            assert meta["source_manifest_sha256"] == self.source_hash
            assert meta["checkpoint_sha256"] == self.meta["checkpoint_sha256"]
            assert meta["group"] == group and meta["condition"] == condition
            assert meta["batch"] == BATCH and meta["steps"] == STEPS and np.array_equal(d["ids"], ids)
        return True

    def validate(self):
        path = self.dir / "validation.npz"
        with np.load(OLD / f"SA{self.width}_selected.npz", allow_pickle=False) as d:
            ids, puz, sol = d["ids"], d["puz"], d["sol"]
            ref_pr, ref_lg = d["preds"], d["logits"]
            original = json.loads(str(d["meta"]))
        assert original["checkpoint_sha256"] == self.meta["source_checkpoint_sha256"]
        if not self.cached(path, ids, "validation", "fixed"):
            started, cpu = time.monotonic(), time.process_time()
            values, _ = self.trajectory(puz, ids, progress=dict(group="validation", condition="fixed"))
            assert np.array_equal(values["pred"], ref_pr), "Archived prediction mismatch"
            assert np.array_equal(values["logits"][..., 1:10], ref_lg), "Archived float32 logit mismatch"
            self.save(path, values, puz, sol, ids, "validation", "fixed", started, cpu)
        with np.load(path, allow_pickle=False) as d:
            assert np.array_equal(d["pred"], ref_pr)
            assert np.array_equal(d["logits"][..., 1:10], ref_lg)
        write_json(self.dir / "validation.json", dict(passed=True, created=utc(),
                   bitwise_all_predictions=True, bitwise_all_digit_logits=True, n=128, steps=16,
                   checkpoint_identity=True, input_sha256=sha(OLD / f"SA{self.width}_selected.npz"),
                   output_sha256=sha(path), source_manifest_sha256=self.source_hash))
        log("validation_passed", width=self.width, n=128, all_digit_logits_bitwise_equal=True)

    def geometry(self):
        with np.load(OUT / "inputs/repair.npz", allow_pickle=False) as d:
            sol, eqr, grids, count = d["sol"], d["eqr"], d["random"], d["wrong_count"]
        p = self.params["dec"]
        roles = np.asarray(p["role_emb"], np.float64)
        expected = 2*self.width*count*np.sum((roles[1]-roles[2])**2)
        max_relative = 0.
        for row in (0, 1, 7, 19, 63, 127, 255, 511):
            target = np.asarray(self.DC.embed_answer(p, self.cfg, self.jnp.asarray(sol[row])), np.float64)
            norm = np.square(target).sum()
            for grid in [eqr[row], *grids[:, row]]:
                embedded = np.asarray(self.DC.embed_answer(p, self.cfg, self.jnp.asarray(grid)), np.float64)
                actual = np.square(embedded-target).sum()
                assert np.isclose(actual, expected[row], rtol=2e-6, atol=1e-6)
                assert np.isclose(np.square(embedded).sum(), norm, rtol=2e-6)
                max_relative = max(max_relative, abs(actual-expected[row])/max(expected[row], 1))
        write_json(self.dir / "embedding_geometry.json", dict(passed=True,
                   all_grids_have_verified_matched_counts=True, checked_rows=[0,1,7,19,63,127,255,511],
                   max_relative_error=max_relative, formula="2 * width * wrong_count * ||role_match-role_other||^2",
                   source_manifest_sha256=self.source_hash))

    def interventions(self):
        group = "interventions"
        with np.load(OUT / "inputs/interventions.npz", allow_pickle=False) as d:
            ids, puz, sol = d["ids"], d["puz"], d["sol"]
        conditions = ("intact", "reset_fast", "reset_slow", "messages_off")
        for b in range(0, len(ids), BATCH):
            batch = slice(b, b+BATCH); ii, pp, ss = ids[batch], puz[batch], sol[batch]
            paths = {c: self.dir / group / c / f"batch_{b:04d}.npz" for c in conditions}
            done = {c: self.cached(paths[c], ii, group, c) for c in conditions}
            if all(done.values()):
                continue
            started, cpu = time.monotonic(), time.process_time()
            _, branch = self.trajectory(pp, ii, steps=1)
            for condition in conditions:
                if done[condition]:
                    with np.load(paths[condition]) as d:
                        assert np.array_equal(d["logits"][0], branch[2])
                    continue
                if condition != "intact":
                    started, cpu = time.monotonic(), time.process_time()
                values, own_branch = self.trajectory(pp, ii, branch=branch, mode=condition,
                                  progress=dict(group=group, condition=condition, batch_start=b))
                assert own_branch[1] is branch[1] and np.array_equal(values["logits"][0], branch[2])
                if self.width == 128 and condition in ("intact", "messages_off"):
                    name = "control" if condition == "intact" else "after_first_off"
                    with np.load(COUPLING / f"{name}.npz") as d:
                        assert np.array_equal(ii, d["ids"][batch])
                        assert np.array_equal(values["pred"], d["pred"][:, batch]), "A128 communication prediction mismatch"
                        assert np.array_equal(values["logits"], d["logits"][:, batch]), "A128 communication logit mismatch"
                self.save(paths[condition], values, pp, ss, ii, group, condition, started, cpu)
        write_json(self.dir / "intervention_checks.json", dict(passed=True, shared_first_carry_by_identity=True,
                   shared_first_logits_bitwise=True, original_A128_all_logits_matched=self.width==128,
                   source_manifest_sha256=self.source_hash))

    def ordinary_group(self, group):
        with np.load(OUT / f"inputs/{group}.npz", allow_pickle=False) as d:
            data = {k: d[k] for k in d.files}
        ids, puz, sol = data["ids"], data["puz"], data["sol"]
        if group == "repair":
            arms = [("fixed", "fixed", 0, None), ("answer", "answer", 0, sol),
                    ("eqr", "answer", 0, data["eqr"])]
            arms += [(f"random_{j}", "answer", j, data["random"][j]) for j in range(DRAWS)]
        else:
            arms = [("fixed", "fixed", 0, None), ("answer", "answer", 0, sol)]
            arms += [(f"{family}_{j}", family, j, None) for j in range(DRAWS) for family in ("independent", "shared")]
        for condition, mode, draw, grid in arms:
            for b in range(0, len(ids), BATCH):
                batch = slice(b, b+BATCH); ii, pp, ss = ids[batch], puz[batch], sol[batch]
                path = self.dir / group / condition / f"batch_{b:04d}.npz"
                if self.cached(path, ii, group, condition):
                    continue
                started, cpu = time.monotonic(), time.process_time()
                supplied = None if grid is None else grid[batch]
                initial = self.starting_state(supplied, ii, mode, draw)
                values, _ = self.trajectory(pp, ii, initial=initial,
                               progress=dict(group=group, condition=condition, batch_start=b))
                self.save(path, values, pp, ss, ii, group, condition, started, cpu, supplied)

    def run(self):
        try:
            self.validate()
            self.geometry()
            self.interventions()
            self.ordinary_group("repair")
            self.ordinary_group("initialization")
            verify_sources()
            self.status("inference_complete", completed=utc())
            log("model_complete", width=self.width)
        except BaseException as exc:
            self.status("failed", error=repr(exc))
            raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["prepare", "validate", "run", "verify"])
    ap.add_argument("--width", type=int, choices=WIDTHS)
    args = ap.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "verify":
        log("sources_verified", manifest=verify_sources())
    else:
        if args.width is None:
            ap.error("--width is required for inference")
        runner = Runner(args.width)
        if args.action == "validate":
            runner.validate()
        else:
            runner.run()


if __name__ == "__main__":
    main()
