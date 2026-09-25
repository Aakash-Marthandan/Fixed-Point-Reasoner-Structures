#!/usr/bin/env python3
"""Independent NumPy recount of the final attention transfer experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/analysis/attention_transfer_20260923"
RELEASE = ROOT / "paper/code"
OLD = ROOT / "runs/analysis/paper_update_20260920/corpus/selected_stablemax"
WIDTHS = (128, 192, 256)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(2**20), b""):
            h.update(b)
    return h.hexdigest()


def write_json(p, obj):
    p = Path(p)
    t = p.with_suffix(p.suffix + ".tmp")
    t.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    t.replace(p)


def events(ex):
    ever = ex.any(0)
    return dict(n=ex.shape[1], steps=ex.shape[0], correct_by_step=ex.sum(1).tolist(),
                first=int(ex[0].sum()), final=int(ex[-1].sum()), ever=int(ever.sum()),
                any_loss=int((ex[:-1] & ~ex[1:]).any(0).sum()),
                terminal_loss=int((ever & ~ex[-1]).sum()), retained_every_step=int(ex.all(0).sum()))


def paired(after, before):
    after, before = np.asarray(after, bool), np.asarray(before, bool)
    assert after.shape == before.shape
    return dict(n=len(before), before=int(before.sum()), after=int(after.sum()),
                gained=int((after & ~before).sum()), lost=int((before & ~after).sum()),
                both=int((after & before).sum()), neither=int((~after & ~before).sum()),
                difference_pp=float(100*(after.mean()-before.mean())))


def interval(difference, strata=None):
    difference = np.asarray(difference, float)
    strata = np.zeros(len(difference), int) if strata is None else np.asarray(strata)
    rng = np.random.default_rng(20260923)
    bootstrap = np.zeros(10000)
    for label in np.unique(strata):
        values = difference[strata == label]
        sampled = rng.integers(0, len(values), (10000, len(values)))
        bootstrap += values[sampled].sum(1)/len(difference)
    return dict(difference_pp=float(100*difference.mean()),
                paired_puzzle_95_percentile_pp=(100*np.quantile(bootstrap, [.025, .975])).tolist(),
                resamples=10000, seed=20260923, strata=int(len(np.unique(strata))))


def progress(pred, puz, sol, horizon=None):
    if horizon:
        pred = pred[:horizon]
    ex = (pred == sol[None]).all((2, 3))
    empty = puz == 0
    errors = ((pred != sol[None]) & empty[None]).sum((2, 3))/empty.sum((1, 2))[None]
    first = np.where(ex.any(0), ex.argmax(0)+1, 0)
    rows = np.flatnonzero(first >= 3)
    last = errors[first[rows]-2, rows]
    delta = errors[0, rows]-last
    return dict(n=len(puz), horizon=len(pred), events=events(ex),
                unconditional_mean_error_percent=(100*errors.mean(1)).tolist(),
                first_1_2_later_censored=[int((first == 1).sum()), int((first == 2).sum()),
                                        len(rows), int((first == 0).sum())],
                completion_cohort=dict(n=len(rows), mean_earlier_reduction_pp=float(100*delta.mean()) if len(rows) else None,
                  median_remaining_error_percent=float(100*np.median(last)) if len(rows) else None,
                  improved_tied_worsened=[int((delta>0).sum()),int((delta==0).sum()),int((delta<0).sum())]))


def read_group(width, group, condition, inputs, manifest, verification):
    folder = OUT / f"attention_{width}" / group / condition
    pred, lg0, exact, error, empty_exact, rms, residual = [], [], [], [], [], [], []
    for b in range(0, len(inputs["ids"]), 128):
        path = folder / f"batch_{b:04d}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        digest = sha(path)
        assert digest == json.loads(path.with_suffix(".sha256.json").read_text())["sha256"]
        manifest[str(path.relative_to(OUT))] = digest
        with np.load(path, allow_pickle=False) as d:
            meta = json.loads(str(d["metadata"]))
            assert meta["source_manifest_sha256"] == sha(OUT / "source_manifest.json")
            assert meta["condition"] == condition and meta["group"] == group
            for k in ("ids", "puz", "sol"):
                assert np.array_equal(d[k], inputs[k][b:b+128])
            logits, pr = d["logits"], d["pred"]
            assert logits.dtype == np.float32 and logits.shape == (16, len(d["ids"]), 9, 9, 11)
            assert np.isfinite(logits).all() and np.isfinite(d["carry_residual"]).all() and np.isfinite(d["carry_rms"]).all()
            decoded = logits.argmax(-1)
            decoded[decoded == 10] = 0
            assert np.array_equal(decoded, pr)
            eq = (pr == d["sol"][None]).all((2, 3))
            assert np.array_equal(eq, d["exact"])
            empty = d["puz"] == 0
            er = ((pr != d["sol"][None]) & empty[None]).sum((2, 3))/empty.sum((1, 2))[None]
            ex_empty = ((pr == d["sol"][None]) | ~empty[None]).all((2, 3))
            if group == "repair" and condition != "fixed":
                grid = inputs["sol"] if condition == "answer" else inputs["eqr"] if condition == "eqr" else inputs["random"][int(condition[-1])]
                assert np.array_equal(d["supplied_grid"], grid[b:b+128])
            if group == "initialization" and condition == "answer":
                assert np.array_equal(d["supplied_grid"], d["sol"])
            pred.append(pr); exact.append(eq); error.append(er); empty_exact.append(ex_empty)
            lg0.append(logits[0]); rms.append(d["carry_rms"]); residual.append(d["carry_residual"])
            verification["decoded_outputs"] += int(np.prod(eq.shape))
            verification["chunks"] += 1
    return dict(pred=np.concatenate(pred, axis=1), exact=np.concatenate(exact, axis=1),
                error=np.concatenate(error, axis=1), empty_exact=np.concatenate(empty_exact, axis=1),
                first_logits=np.concatenate(lg0, axis=0), rms=np.concatenate(rms, axis=1),
                residual=np.concatenate(residual, axis=1))


def condition_summary(rec, keep=None):
    keep = np.ones(rec["exact"].shape[1], bool) if keep is None else keep
    return dict(events=events(rec["exact"][:, keep]),
                mean_empty_error_percent=(100*rec["error"][:, keep].mean(1)).tolist(),
                empty_cell_exactness_equal_all_cell=bool(np.array_equal(rec["exact"][:, keep], rec["empty_exact"][:, keep])),
                mean_final_residual_by_carry=rec["residual"][-1, keep].mean(0).tolist(),
                mean_final_rms_by_carry=rec["rms"][-1, keep].mean(0).tolist())


def existing_evidence():
    result = {}
    models = json.loads((RELEASE / "evidence/models.json").read_text())
    for w in WIDTHS:
        path = OLD / f"SA{w}_selected.npz"
        with np.load(path, allow_pickle=False) as d:
            meta = json.loads(str(d["meta"]))
            assert meta["checkpoint_sha256"] == models[f"attention_{w}"]["source_checkpoint_sha256"]
            assert np.array_equal(d["preds"], d["logits"].argmax(-1)+1)
            trajectory = progress(d["preds"], d["puz"], d["sol"])
            trajectory.update(scope="Existing 128 rating-stratified test IDs, D16, CPU; completion cohorts differ across checkpoints", sha256=sha(path))
        bank = RELEASE / f"evidence/selection/attention_{w}_k128.npz"
        with np.load(bank, allow_pickle=False) as d:
            ex, score = d["mi_exact_k"].astype(bool), d["mi_resid_k"]
            assert np.isfinite(score).all() and np.array_equal(score, score.astype(np.float16).astype(score.dtype))
            assert np.array_equal(d["mi_true"], d["mi_verified"])
            rows = np.arange(len(ex)); prefixes = {}
            for k in (1,2,4,8,16,32,64,128):
                index = score[:, :k].argmin(1)
                prefixes[k] = dict(covered=int(ex[:, :k].any(1).sum()), selected=int(ex[rows, index].sum()))
            a, b = score[:, :8].argmin(1), score.argmin(1)
            before, after = ex[rows,a], ex[rows,b]
            lost = before & ~after
            strict = bool(np.all(b[lost]>=8) and np.all(score[rows[lost], b[lost]] < score[rows[lost], a[lost]]))
            assert strict
            selection = dict(n=len(ex), depth=64, prefixes=prefixes, k8_to_k128=paired(after,before),
                             every_lost_success_displaced_by_new_strictly_smaller_wrong_score=strict,
                             scope="Existing single D64 Gaussian bank, float16 stored scores and earliest ties; no new inference", sha256=sha(bank))
        result[str(w)] = dict(trajectories=trajectory, selection=selection)
    return result


def analyze():
    sources = json.loads((OUT / "source_manifest.json").read_text())["sha256"]
    for name, digest in sources.items():
        assert sha(ROOT / name) == digest, name
    protected = json.loads((OUT / "protected_files.json").read_text())
    changed_protected = [name for name,digest in protected.items() if sha(ROOT/name)!=digest]
    inputs = {}
    for group in ("repair", "interventions", "initialization"):
        with np.load(OUT / f"inputs/{group}.npz", allow_pickle=False) as d:
            inputs[group] = {k:d[k] for k in d.files}
    repair = inputs["repair"]
    count = ((repair["eqr"]!=repair["sol"]) & (repair["puz"]==0)).sum((1,2))
    assert np.array_equal(count, repair["wrong_count"])
    assert np.all(((repair["random"]!=repair["sol"][None]) & (repair["puz"][None]==0)).sum((2,3)) == count[None])
    # Reconstruct each saved random grid independently of the producer function.
    for draw in range(4):
        reconstruction = repair["sol"].copy()
        for row,pid in enumerate(repair["ids"]):
            rng = np.random.default_rng([20260917,int(pid),77] if draw==0 else [20260923,int(pid),77,draw])
            empty_indices = np.flatnonzero(repair["puz"][row].reshape(-1)==0)
            chosen = empty_indices[rng.choice(len(empty_indices),int(count[row]),replace=False)] if count[row] else []
            for pos in chosen:
                reconstruction[row].reshape(-1)[pos] = (repair["sol"][row].reshape(-1)[pos]-1+int(rng.integers(1,9)))%9+1
        assert np.array_equal(reconstruction,repair["random"][draw])
    result = dict(status="incomplete", protocol_sha256=sha(OUT/"PROTOCOL.md"),
                  source_manifest_sha256=sha(OUT/"source_manifest.json"), models={},
                  existing=existing_evidence(), scope="Conditional inference diagnostics; no retraining or manuscript amendment",
                  verification=dict(source_files=len(sources), protected_files=len(protected),
                                    changed_protected_files=changed_protected, chunks=0, decoded_outputs=0,
                                    corruption_reconstruction=True), missing=[])
    manifest = {}
    for w in WIDTHS:
        try:
            verification = result["verification"]
            gate = json.loads((OUT/f"attention_{w}/validation.json").read_text())
            assert gate["passed"] and gate["source_manifest_sha256"] == result["source_manifest_sha256"]
            assert sha(OUT/f"attention_{w}/validation.npz") == gate["output_sha256"]
            geometry = json.loads((OUT/f"attention_{w}/embedding_geometry.json").read_text())
            assert geometry["passed"]
            rec = {c:read_group(w,"interventions",c,inputs["interventions"],manifest,verification)
                   for c in ("intact","reset_fast","reset_slow","messages_off")}
            base = rec["intact"]; initial = base["exact"][0]
            interventions = {}
            for c,r in rec.items():
                assert np.array_equal(r["first_logits"], base["first_logits"])
                interventions[c] = condition_summary(r)
                interventions[c]["versus_intact"] = paired(r["exact"][-1],base["exact"][-1])
                interventions[c]["interval"] = interval(r["exact"][-1].astype(float)-base["exact"][-1])
                interventions[c]["initially_incorrect"] = events(r["exact"][:,~initial])
                interventions[c]["initially_correct"] = events(r["exact"][:,initial])
            arms = ["fixed","answer","eqr",*[f"random_{j}" for j in range(4)]]
            rr = {c:read_group(w,"repair",c,repair,manifest,verification) for c in arms}
            use = count>0; reference = rr["eqr"]["exact"][-1,use]
            rep = dict(n_source=512,n_eligible=int(use.sum()), conditions={}, comparisons={})
            for c,r in rr.items():
                rep["conditions"][c] = condition_summary(r,use)
                rep["conditions"][c]["zero_error_cohort"] = condition_summary(r,~use)
                rep["conditions"][c]["versus_ordinary"] = paired(r["exact"][-1,use],rr["fixed"]["exact"][-1,use])
                if c.startswith("random_"):
                    outcome = r["exact"][-1,use]
                    rep["comparisons"][c] = dict(paired=paired(outcome,reference),
                            interval=interval(outcome.astype(float)-reference,repair["rating_bin"][use]))
            averaged = np.stack([rr[f"random_{j}"]["exact"][-1,use] for j in range(4)]).mean(0)
            rep["four_draw_mean_vs_eqr"] = interval(averaged-reference,repair["rating_bin"][use])
            rep["answer_retention_all_512"] = events(rr["answer"]["exact"])
            rep["ordinary_correction_before_completion"] = progress(rr["fixed"]["pred"],repair["puz"],repair["sol"])
            init_arms = ["fixed","answer",*[f"{f}_{j}" for j in range(4) for f in ("independent","shared")]]
            ir = {c:read_group(w,"initialization",c,inputs["initialization"],manifest,verification) for c in init_arms}
            initialization = {}
            for c,r in ir.items():
                initialization[c] = condition_summary(r)
                initialization[c]["versus_fixed"] = paired(r["exact"][-1],ir["fixed"]["exact"][-1])
                initialization[c]["interval"] = interval(r["exact"][-1].astype(float)-ir["fixed"]["exact"][-1])
            result["models"][str(w)] = dict(validation=gate,geometry=geometry,interventions=interventions,
                                             repair=rep,initialization=initialization)
        except FileNotFoundError as exc:
            result["missing"].append(str(exc))
    result["status"] = "complete" if len(result["models"])==3 and not changed_protected else "incomplete"
    write_json(OUT/"analysis.json",result)
    write_json(OUT/"output_manifest.json",manifest)
    write_report(result)
    print(json.dumps(dict(status=result["status"],complete_models=list(result["models"]),
                         verification=result["verification"],missing=result["missing"])))
    return result


def write_report(r):
    lines = ["# Final attention transfer study", "", f"Status: **{r['status']}**.", "",
             "Fresh inference uses the exact selected EMA weights, CPU float32, batch 128 and 16 outer iterations. "
             "Comparisons condition on checkpoints, populations and draws. No manuscript or frozen evidence is amended.", ""]
    for w,m in r["models"].items():
        lines += [f"## Attention {w}", "", "### Communication and retained state", "",
                  "All four conditions share the complete first state on the same 256 puzzles.", "",
                  "| Condition | Correct at 16 | Lost / gained versus intact | New discoveries after the shared first update |",
                  "|---|---:|---:|---:|"]
        for c,a in m["interventions"].items():
            p=a["versus_intact"]
            lines.append(f"| {c.replace('_',' ')} | {a['events']['final']}/256 | {p['lost']} / {p['gained']} | {a['initially_incorrect']['ever']} |")
        lines += ["", "### Matched-error repair", "",
                  "Primary repair population: 438 incorrect EqR-source grids. Random grids match each puzzle's wrong-cell count.", "",
                  "| Start | Correct at 16 | First-update empty-cell error | Lost / gained versus ordinary |",
                  "|---|---:|---:|---:|"]
        for c,a in m["repair"]["conditions"].items():
            p=a["versus_ordinary"]
            lines.append(f"| {c.replace('_',' ')} | {a['events']['final']}/438 | {a['mean_empty_error_percent'][0]:.2f}% | {p['lost']} / {p['gained']} |")
        lines += ["", "| Corruption draw | Random minus EqR repair (pp) | Paired 95% interval |", "|---|---:|---:|"]
        for c,a in m["repair"]["comparisons"].items():
            v=a["interval"];ci=v["paired_puzzle_95_percentile_pp"]
            lines.append(f"| {c} | {v['difference_pp']:.2f} | [{ci[0]:.2f}, {ci[1]:.2f}] |")
        avg=m["repair"]["four_draw_mean_vs_eqr"];ci=avg["paired_puzzle_95_percentile_pp"]
        lines += ["",f"Averaging the four corruption draws per puzzle gives {avg['difference_pp']:.2f} pp "
                  f"[{ci[0]:.2f}, {ci[1]:.2f}]. Intervals resample paired puzzles within the original rating strata; "
                  "they do not measure training-seed variation or uncertainty over all possible corruptions.", "",
                  "### Initialization and preservation", "",
                  "Each row uses the same 128 validation puzzles. Shared draws are used across all puzzles; no best draw is selected.", "",
                  "| Start | Ever correct | Correct at 16 | Any loss | Terminal loss |", "|---|---:|---:|---:|---:|"]
        for c,a in m["initialization"].items():
            e=a["events"]
            lines.append(f"| {c.replace('_',' ')} | {e['ever']} | {e['final']} | {e['any_loss']} | {e['terminal_loss']} |")
        lines += ["", f"Injected answers retained at every step: {m['initialization']['answer']['events']['retained_every_step']}/128.", ""]
    lines += ["## Existing final-checkpoint evidence", "",
              "These are recounts of archived observations, not additional inference. The progress sample is 128 "
              "stratified test puzzles at D16; the selection sample is a distinct 5,000-puzzle D64/k128 bank.", "",
              "| Model | Completion cohort | Earlier correction (pp) | Median precompletion error | Selected / covered at k128 |",
              "|---|---:|---:|---:|---:|"]
    for w,a in r["existing"].items():
        p=a["trajectories"]["completion_cohort"];s=a["selection"]["prefixes"][128]
        lines.append(f"| Attention {w} | {p['n']} | {p['mean_earlier_reduction_pp']:.2f} | "
                     f"{p['median_remaining_error_percent']:.2f}% | {s['selected']} / {s['covered']} |")
    lines += ["", "## Verification and interpretation", "",
              f"Independently decoded {r['verification']['decoded_outputs']:,} saved predictions in "
              f"{r['verification']['chunks']} inference chunks. All loaded predictions agree with their "
              "float32-logit argmax and stored correctness flags. Inputs, checkpoint identities, unchanged "
              "sources, corruption counts and random reconstruction are checked. Each model must reproduce "
              "its archived intact predictions and digit logits bit-for-bit before new inference.", "",
              "Null and reversed contrasts remain in the tables. The observations test whether the specified "
              "functional dependencies hold in the benchmark checkpoints; they do not identify memory content, "
              "isolate why a configuration is difficult, establish fixed points, or explain the full accuracy advantage.", ""]
    if r["missing"]:
        lines += ["Missing outputs: " + "; ".join(r["missing"]), ""]
    if r["verification"]["changed_protected_files"]:
        lines += ["Protected files changed during execution: " + "; ".join(r["verification"]["changed_protected_files"]), ""]
    path=OUT/"REPORT.md";temp=path.with_suffix(".md.tmp");temp.write_text("\n".join(lines));temp.replace(path)


def selftest():
    ex=np.array([[0,1,1,0],[1,0,1,0],[0,1,1,0]],bool)
    e=events(ex)
    assert (e['ever'],e['final'],e['any_loss'],e['terminal_loss'],e['retained_every_step']) == (3,2,2,1,1)
    p=paired(np.array([1,0,1,0]),np.array([0,1,1,0]))
    assert (p['gained'],p['lost'],p['both'],p['neither'],p['difference_pp'])==(1,1,1,1,0)
    ci=interval(np.ones(8),np.repeat([0,1],4))
    assert ci['paired_puzzle_95_percentile_pp']==[100.,100.]
    print('Analysis synthetic event/pairing/resampling checks passed.')


if __name__ == "__main__":
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--selftest',action='store_true');a=ap.parse_args()
    if a.selftest:selftest()
    else:analyze()
