#!/usr/bin/env python3
"""Recount completed attention diagnostics; separately label exploratory checks.

No model inference or amendment of the declared protocol/frozen paper occurs.
The natural-first-grid clue test and >=50%-wrong subgroup are post-run analyses.
"""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

import analyze_attention_transfer as A

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/analysis/attention_transfer_20260923"


def load(width, group, condition):
    paths = sorted((OUT / f"attention_{width}" / group / condition).glob("batch_*.npz"))
    records = []
    for path in paths:
        with np.load(path, allow_pickle=False) as d:
            records.append({k: d[k] for k in ("pred", "puz", "sol", "ids")})
    return {k: np.concatenate([r[k] for r in records], axis=1 if k == "pred" else 0)
            for k in records[0]}


def clue_conflicts(grid, puzzle, solution):
    """Check wrong digits against givens, using two independent formulations."""
    wrong = (grid != solution) & (puzzle == 0)
    assert np.isin(grid, np.arange(1, 10)).all()
    loop = np.zeros_like(wrong)
    for i, r, c in np.argwhere(wrong):
        digit = grid[i, r, c]
        loop[i, r, c] = ((puzzle[i, r, :] == digit).any()
                        or (puzzle[i, :, c] == digit).any()
                        or (puzzle[i, r//3*3:r//3*3+3, c//3*3:c//3*3+3] == digit).any())
    row, col = np.indices((9, 9)).reshape(2, 81)
    box = row//3*3 + col//3
    peers = ((row[:, None] == row[None]) | (col[:, None] == col[None])
             | (box[:, None] == box[None]))
    p, g = puzzle.reshape(-1, 81), grid.reshape(-1, 81)
    matrix = (((g[:, :, None] == p[:, None, :]) & peers[None]
               & (p[:, None, :] != 0)).any(2).reshape(wrong.shape) & wrong)
    assert np.array_equal(loop, matrix)
    return dict(wrong_empty_cells=int(wrong.sum()), clue_refuted=int(loop.sum()),
                clue_refuted_percent=100*float(loop.sum()/wrong.sum()),
                wrong_clues=int(((grid != solution) & (puzzle != 0)).sum()))


def main():
    summary = json.loads((OUT / "analysis.json").read_text())
    assert summary["status"] == "complete"
    sources = json.loads((OUT / "source_manifest.json").read_text())["sha256"]
    outputs = json.loads((OUT / "output_manifest.json").read_text())
    for name, digest in sources.items():
        assert A.sha(ROOT / name) == digest
    for name, digest in outputs.items():
        assert A.sha(OUT / name) == digest
    for name, digest in json.loads((OUT / "protected_files.json").read_text()).items():
        assert A.sha(ROOT / name) == digest
    with np.load(OUT / "inputs/repair.npz", allow_pickle=False) as d:
        repair = {k: d[k] for k in d.files}
    use = repair["wrong_count"] > 0
    fraction = repair["wrong_count"] / (repair["puz"] == 0).sum((1, 2))
    hard = use & (fraction >= .5)
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  scope="Saved-data recount only; exploratory checks explicitly separated; no manuscript amendment",
                  source_sha256=A.sha(Path(__file__)),
                  primary_recount_sha256=A.sha(ROOT / "tools/analyze_attention_transfer.py"),
                  source_manifest_sha256=A.sha(OUT / "source_manifest.json"),
                  output_manifest_sha256=A.sha(OUT / "output_manifest.json"),
                  primary_analysis_sha256=A.sha(OUT / "analysis.json"),
                  original_primary=dict(n=int(use.sum()), mean_initial_wrong_percent=100*float(fraction[use].mean())),
                  exploratory=dict(note="Designed after run completion; descriptive, not causal or preregistered",
                                   high_error_subgroup_n=int(hard.sum()), source_clue_conflicts={}),
                  models={})
    for name, grids in [("eqr", repair["eqr"]), *[(f"random_{i}", g) for i, g in enumerate(repair["random"])]]:
        result["exploratory"]["source_clue_conflicts"][name] = clue_conflicts(
            grids[use], repair["puz"][use], repair["sol"][use])
    assert result["exploratory"]["source_clue_conflicts"]["eqr"]["clue_refuted"] == 17
    assert result["exploratory"]["source_clue_conflicts"]["random_0"]["clue_refuted"] == 6409
    common_bank_ids = None
    for w in (128, 192, 256):
        with np.load(OUT / f"attention_{w}/validation.npz") as new, np.load(
                ROOT / f"runs/analysis/paper_update_20260920/corpus/selected_stablemax/SA{w}_selected.npz") as old:
            assert np.array_equal(new["pred"], old["preds"])
            assert np.array_equal(new["logits"][..., 1:10], old["logits"])
        rec = {c: load(w, "repair", c) for c in ["fixed", "answer", "eqr", *[f"random_{i}" for i in range(4)]]}
        for r in rec.values():
            assert all(np.array_equal(r[k], repair[k]) for k in ("ids", "puz", "sol"))
        exact = {c: (r["pred"] == r["sol"][None]).all((2, 3)) for c, r in rec.items()}
        errors = {c: ((r["pred"] != r["sol"][None]) & (r["puz"][None] == 0)).sum((2, 3))
                  / (r["puz"] == 0).sum((1, 2))[None] for c, r in rec.items()}
        averaged = np.stack([exact[f"random_{i}"][:, use] for i in range(4)]).mean(0)
        first = A.interval(averaged[0]-exact["eqr"][0, use], repair["rating_bin"][use])
        off = load(w, "interventions", "messages_off")
        whole = (off["pred"] == off["sol"][None]).all((2, 3))
        empty = ((off["pred"] == off["sol"][None]) | (off["puz"][None] != 0)).all((2, 3))
        scoring = dict(whole_correct=whole.sum(1).tolist(), empty_correct=empty.sum(1).tolist(),
                       differing_observations=[dict(iteration=int(t+1), test_id=int(off["ids"][i]))
                                               for t, i in np.argwhere(whole != empty)],
                       empty_new_discoveries=int((empty[1:] & ~empty[0]).any(0).sum()))
        with np.load(ROOT / f"paper/code/evidence/selection/attention_{w}_k128.npz") as d:
            if common_bank_ids is None:
                common_bank_ids = d["idx"].copy()
            assert np.array_equal(common_bank_ids, d["idx"])
            ex, score = d["mi_exact_k"].astype(bool), d["mi_resid_k"]
            rows = np.arange(len(ex)); a = score[:, :8].argmin(1); b = score.argmin(1)
            losses = rows[ex[rows, a] & ~ex[rows, b]]
            displacement = []
            for i in losses:
                assert b[i] >= 8 and score[i, b[i]] < score[i, a[i]]
                displacement.append(dict(test_id=int(d["idx"][i]), old_index=int(a[i]), new_index=int(b[i]),
                                         old_score=float(score[i, a[i]]), new_score=float(score[i, b[i]])))
        result["models"][str(w)] = dict(
            archived_predictions_and_digit_logits_bitwise_rechecked=True,
            first_update_random_mean_minus_eqr=first,
            repair_correct_at_1_2_4_8_16={c:e[[0, 1, 3, 7, 15]][:, use].sum(1).tolist() for c,e in exact.items()},
            mean_first_update_error_percent={c:100*float(e[0,use].mean()) for c,e in errors.items()},
            message_scoring=scoring, selection_displacements_8_to_128=displacement,
            injected_answers_all_step_preservation=int(exact["answer"].all(0).sum()),
            exploratory=dict(high_error_endpoints={c:int(e[-1,hard].sum()) for c,e in exact.items()},
                             natural_first_grid_clue_conflicts=clue_conflicts(rec["fixed"]["pred"][0], repair["puz"], repair["sol"])))
    result["verification"] = dict(source_files=len(sources), output_chunks=len(outputs),
                                   unchanged_protected_files=True, selection_bank_ids_identical=True,
                                   two_clue_conflict_implementations_agree=True)
    A.write_json(OUT / "followup_analysis.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
