# Ledger: PHASE B ops hardening (2026-08-26, registered d06fe39) — evaluator
# incremental banking. Named test: an eval shard interrupted mid-run (test hook
# SX_TEST_ABORT_AFTER) and resumed from its banked partial produces records and
# summary BIT-IDENTICAL to an uninterrupted run (per-(puzzle,draw) seeding makes
# the batch stream deterministic across the resume boundary).
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "runs" / "pretrainsport2_S5" / "ckpt_latest.pkl"
NPZ = ROOT / "data" / "sudoku_extreme" / "sudoku_extreme_seed0.npz"


@pytest.mark.skipif(not (CKPT.exists() and NPZ.exists()), reason="needs banked S5 ckpt + benchmark npz")
def test_bank_resume_bit_identical(tmp_path):
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    base = [sys.executable, str(ROOT / "tools" / "eval_sudoku_extreme.py"),
            "--ckpt", str(CKPT), "--npz", str(NPZ), "--split", "test",
            "--subsample", "24", "--batch", "8", "--t-total", "8", "--k-init", "2"]

    ref = tmp_path / "ref"
    subprocess.run(base + ["--out", str(ref)], env=env, check=True, capture_output=True)

    res = tmp_path / "res"
    # bank after every batch, abort after 8 rows -> partial holds one batch
    p1 = subprocess.run(base + ["--out", str(res), "--bank-every", "0.0001"],
                        env=dict(env, SX_TEST_ABORT_AFTER="8"), capture_output=True)
    assert p1.returncode == 3
    assert (res / "partial_all.npz").exists()
    subprocess.run(base + ["--out", str(res), "--bank-every", "0.0001"], env=env, check=True,
                   capture_output=True)
    assert not (res / "partial_all.npz").exists()  # cleaned on completion

    a = np.load(ref / "records_all.npz")
    b = np.load(res / "records_all.npz")
    assert sorted(a.files) == sorted(b.files)
    for k in a.files:
        assert np.array_equal(a[k], b[k]), f"records differ on {k}"
    sa = json.loads((ref / "summary_all.json").read_text())
    sb = json.loads((res / "summary_all.json").read_text())
    for k in set(sa) - {"wall_s"}:
        assert sa[k] == sb[k], f"summary differs on {k}"


@pytest.mark.skipif(not (CKPT.exists() and NPZ.exists()), reason="needs banked S5 ckpt + benchmark npz")
def test_batch_size_invariance(tmp_path):
    # RUNG 2 O3 gate (2026-08-27): the chain moves to --batch 128 to shrink the
    # bank quantum; per-(puzzle,draw) seeding makes results batch-invariant BY
    # DESIGN — this test pins it empirically (records identical across sizes,
    # incl. the new uv_vote columns).
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    base = [sys.executable, str(ROOT / "tools" / "eval_sudoku_extreme.py"),
            "--ckpt", str(CKPT), "--npz", str(NPZ), "--split", "test",
            "--subsample", "24", "--t-total", "8", "--k-init", "2", "--vote-unverified"]
    outs = {}
    for bs in ("8", "24"):
        o = tmp_path / f"b{bs}"
        subprocess.run(base + ["--batch", bs, "--out", str(o)], env=env, check=True, capture_output=True)
        outs[bs] = o
    a = np.load(outs["8"] / "records_all.npz")
    b = np.load(outs["24"] / "records_all.npz")
    assert sorted(a.files) == sorted(b.files)
    assert any(k.startswith("uv_vote_k") for k in a.files)
    for k in a.files:
        assert np.array_equal(a[k], b[k]), f"records differ on {k} across batch sizes"
