"""CHAMPION NIGHT builds (Plan_2026-09-08_Champion_Night §5; 2026-09-08) — named tests:
(1) the DEC's set-attention coupling (C4) is exactly S9-equivariant, its parameter count is pinned (+2 w heads dk per block),
    and with zero query/key projections (uniform attention) it reproduces the mean coupling; (2) the commit head (C6):
    tau >= 1 leaves the forward BIT-IDENTICAL to the plain DEC, tau < 1 with a hot head hardens free cells (the logits
    change, given cells never), the head's logit is S9-invariant and the straight-through gradient reaches the head;
(3) the online position orbit (C3) keeps every pair a valid Sudoku (rows/cols/boxes permutations, givens consistent and
    counted), is deterministic per key, and moves most rows; (4) select_ckpt's earliest-tie and second-key rules (the
    default call unchanged); (5) the 512-puzzle monitor npz: train + test arrays byte-identical to the base, 512 disjoint
    monitor rows; (6) the trainer on CPU with every new flag at once (attn + commit + orbit + the 512 monitor) writes finite
    losses, a commit_bce column and a 512-row monitor, and the evaluator's --record-commit rows read back from its grid."""
import json, os, subprocess, sys
from pathlib import Path
import numpy as np, jax, jax.numpy as jnp, pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tools"))
from qhrrn2 import grid as G, model as M, sudoku as SU, dec_cell as DC, sudoku_extreme as SX
from qhrrn2.config import Config

NPZ = ROOT / "data/sudoku_extreme/sudoku_extreme_seed0.npz"
NPZ512 = ROOT / "data/sudoku_extreme/sudoku_extreme_seed0_mon512.npz"
FIELD = dict(canvas=9, scales=2, pool_arity=3, mixer_kind="group9", attn_max_hw=9, equilibrium=True, sudoku_layout="native9",
             T=3, eta_fixed=1.0, eta_z_fixed=1.0, loss_kind="stablemax")
TINY = dict(cell_kind="dec", dec_width=16, trm_layers=1, trm_h_cycles=2, trm_l_cycles=2)
DEC_TINY = Config(**FIELD, **TINY)
DEC_ATTN = Config(**FIELD, **TINY, dec_coupling_kind="attn", dec_attn_heads=4, dec_attn_dk=4)
DEC_COMMIT1 = Config(**FIELD, **TINY, dec_commit=True, dec_commit_tau=1.0)
DEC_COMMIT = Config(**FIELD, **TINY, dec_commit=True, dec_commit_tau=0.5)


def _pair(seed=3, givens=40):
    puz, sol = SU.sample(np.random.default_rng(seed), givens)
    return jnp.asarray(puz, jnp.int32), jnp.asarray(sol, jnp.int32)


def _perm(seed):
    pi = np.arange(M.VOCAB, dtype=np.int32); pi[1:10] = np.random.default_rng(seed).permutation(9) + 1
    return pi


def _logits(p, cfg, x, t=2):
    outs, _, _ = M.iterate_eq(p, cfg, x, tau=1.0, t_total=t)
    return [np.asarray(o.logits) for o in outs]


def test_attn_coupling_exact_s9_pinned_count_and_uniform_limit():
    x, _ = _pair()
    p = M.init_params(jax.random.PRNGKey(0), DEC_ATTN)
    b = p["dec"]["blocks"][0]
    assert b["attn_q"].shape == (16, 16) and b["attn_k"].shape == (16, 16)
    assert M.count_params(p["dec"]) - M.count_params(M.init_params(jax.random.PRNGKey(0), DEC_TINY)["dec"]) == 2 * 16 * 4 * 4
    outs = _logits(p, DEC_ATTN, x)
    for seed in (1, 2):
        pi = _perm(seed); outs_p = _logits(p, DEC_ATTN, jnp.asarray(pi)[x])
        for lg, lgp in zip(outs, outs_p):
            assert np.allclose(lgp[..., pi], lg, atol=1e-4), "attention coupling is not S9-equivariant"
    # zero projections -> uniform attention over the other eight fields == the mean coupling (same fc)
    p0 = M.init_params(jax.random.PRNGKey(0), DEC_TINY)
    pz = jax.tree.map(lambda a: a, p)
    for blk, blk0 in zip(pz["dec"]["blocks"], p0["dec"]["blocks"]):
        blk["attn_q"] = jnp.zeros_like(blk["attn_q"]); blk["attn_k"] = jnp.zeros_like(blk["attn_k"])
        for k_ in ("mlp_t", "mlp", "fc"):
            blk[k_] = blk0[k_]
    for k_ in ("role_emb", "lm_head", "q_head"):
        pz["dec"][k_] = p0["dec"][k_]
    a_, b_ = _logits(pz, DEC_ATTN, x), _logits(p0, DEC_TINY, x)
    for la, lb in zip(a_, b_):
        assert np.allclose(la, lb, atol=1e-4), "uniform attention does not reproduce the mean coupling"


def test_commit_head_identity_at_tau_1_hardening_below_and_invariance():
    x, y = _pair()
    p_plain = M.init_params(jax.random.PRNGKey(0), DEC_TINY)
    p1 = M.init_params(jax.random.PRNGKey(0), DEC_COMMIT1)
    assert "commit_head" in p1["dec"] and p1["dec"]["commit_head"]["w"].shape == (16,)
    # the head adds w + 1 params and changes nothing else: bit-identical logits at tau 1
    assert M.count_params(p1["dec"]) - M.count_params(p_plain["dec"]) == 17
    for la, lb in zip(_logits(p1, DEC_COMMIT1, x, t=3), _logits(p_plain, DEC_TINY, x, t=3)):
        assert np.array_equal(la, lb)
    # tau .5 with a hot head (bias +6): free cells harden from step 2 on -> the logits change; step 1 (no carry) identical
    p_hot = jax.tree.map(lambda a: a, p1)
    p_hot["dec"]["commit_head"] = {"w": p1["dec"]["commit_head"]["w"], "b": jnp.full((), 6.0)}
    lh = _logits(p_hot, DEC_COMMIT, x, t=3); lp = _logits(p_plain, DEC_TINY, x, t=3)
    assert np.array_equal(lh[0], lp[0]) and not np.allclose(lh[1], lp[1], atol=1e-3)
    # the gate: every free cell fires with the hot head, no given cell ever does
    outs, _, _, zc = M.iterate_eq(p_hot, DEC_COMMIT, x, tau=1.0, t_total=1, return_z=True)
    emb, gate = DC.embed_effective(p_hot["dec"], DEC_COMMIT, x, zc[0])
    gate = np.asarray(gate); free = np.asarray(x).reshape(-1) == 0
    assert gate[free].all() and not gate[~free].any()
    # S9-invariance of the commit logit
    cl = np.asarray(DC.commit_logits(p_hot["dec"], zc[0]))
    pi = _perm(5); _, _, _, zcp = M.iterate_eq(p_hot, DEC_COMMIT, jnp.asarray(pi)[x], tau=1.0, t_total=1, return_z=True)
    assert np.allclose(cl, np.asarray(DC.commit_logits(p_hot["dec"], zcp[0])), atol=1e-4)
    # the straight-through path: a loss on the hardened segment's logits has a nonzero gradient at the head's weights
    def loss(pp):
        e, _ = DC.embed_effective(pp["dec"], DEC_COMMIT, x, zc[0])
        zH, _ = DC.segment(pp["dec"], DEC_COMMIT, e, zc[0], zc[1])
        lg, _ = DC.readout(pp["dec"], DEC_COMMIT, zH, (9, 9))
        return -jnp.mean(jnp.take_along_axis(jax.nn.log_softmax(lg, -1), y[..., None], -1))
    g = jax.grad(loss)(p_hot)
    assert float(jnp.sum(jnp.abs(g["dec"]["commit_head"]["w"]))) > 0


def test_orbit_batch_valid_deterministic_and_moving():
    rng = np.random.default_rng(11)
    pairs = [SU.sample(rng, int(g)) for g in rng.integers(25, 45, size=256)]
    x = jnp.asarray(np.stack([p for p, _ in pairs]), jnp.int32); y = jnp.asarray(np.stack([s for _, s in pairs]), jnp.int32)
    xo, yo = SX.orbit_batch(jax.random.PRNGKey(7), x, y)
    xo2, yo2 = SX.orbit_batch(jax.random.PRNGKey(7), x, y)
    assert np.array_equal(np.asarray(xo), np.asarray(xo2)) and np.array_equal(np.asarray(yo), np.asarray(yo2))
    xo, yo = np.asarray(xo), np.asarray(yo)
    full = set(range(1, 10))
    for b in range(xo.shape[0]):
        s = yo[b]
        for i in range(9):
            assert set(s[i].tolist()) == full and set(s[:, i].tolist()) == full
        for bi in range(3):
            for bj in range(3):
                assert set(s[3 * bi:3 * bi + 3, 3 * bj:3 * bj + 3].reshape(-1).tolist()) == full
        m = xo[b] != 0
        assert (xo[b][m] == s[m]).all() and m.sum() == (np.asarray(x[b]) != 0).sum()
    moved = np.mean([not np.array_equal(xo[b], np.asarray(x[b])) for b in range(xo.shape[0])])
    assert moved > 0.95


def test_select_ckpt_tie_rules(tmp_path):
    d = tmp_path / "pretrainchamp_C0"; d.mkdir()
    rows = []
    for st, v, raw in ((10000, .90, .88), (16000, .95, .93), (30000, .95, .90), (42000, .95, .95)):
        rows.append(json.dumps({"monitor": {"step": st, "val_t16_ema": v, "val_t16": raw}}))
        (d / f"ckpt_{st:06d}.pkl").write_bytes(b"x")
    (d / "metrics.jsonl").write_text("\n".join(rows) + "\n")
    def run(*flags):
        r = subprocess.run([sys.executable, str(ROOT / "tools/select_ckpt.py"), str(d), "--key", "val_t16_ema", *flags], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr; return r.stdout.split()[0]
    assert run() == "042000"                                           # the pre-existing rule: ties -> later
    assert run("--tie", "earliest") == "016000"                         # earliest tie
    assert run("--tie", "earliest", "--second-key", "val_t16") == "042000"   # the raw monitor breaks the tie first (.95 > .93)
    assert run("--second-key", "val_t16") == "042000"


@pytest.mark.skipif(not (NPZ.exists() and NPZ512.exists()), reason="npz files absent")
def test_monitor_npz_extension_identity_and_disjointness():
    a = np.load(NPZ); b = np.load(NPZ512)
    for k in ("train_q", "train_a", "train_rating", "train_row", "test_q", "test_a", "test_rating", "test_source", "source_names"):
        assert np.array_equal(a[k], b[k]), k
    assert b["val_q"].shape == (512, 9, 9) and np.array_equal(a["val_q"], b["val_q"][:64]) and np.array_equal(a["val_a"], b["val_a"][:64])
    sv = {SX.grid_to_str(g) for g in b["val_q"]}
    assert len(sv) == 512
    assert not any(SX.grid_to_str(g) in sv for g in b["train_q"])
    te = {SX.grid_to_str(g) for g in b["test_q"][:50000]}
    assert not (sv & te)
    meta = eval(str(b["meta"][0])); assert meta["n_val"] == 512 and meta["monitor_ext_seed"] == 20260908


@pytest.mark.skipif(not (NPZ.exists() and NPZ512.exists()), reason="npz files absent")
def test_trainer_all_flags_cpu_and_record_commit(tmp_path):
    out = tmp_path / "pt"
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), JAX_PLATFORMS="cpu")
    cmd = [sys.executable, str(ROOT / "tools/pretrain.py"), "--out", str(out), "--sudoku-extreme", str(NPZ512), "--sudoku-layout", "native9",
           "--equilibrium", "--sot", "--act", "--cell", "dec", "--dec-width", "16", "--trm-layers", "1", "--trm-h-cycles", "1", "--trm-l-cycles", "1",
           "--T", "2", "--sudoku-aug", "0", "--sudoku-orbit-online", "--dec-coupling", "attn", "--dec-attn-heads", "4", "--dec-attn-dk", "4",
           "--dec-commit", "--dec-commit-tau", "0.5", "--dec-commit-w", "0.1", "--fpa-k", "1", "--fpa-frac", "0.5", "--trm-ri-sigma", "1.0",
           "--batch", "4", "--steps", "3", "--warmup", "1", "--lr", "1e-3", "--lr-end", "1e-3", "--ema", "0.9", "--loss", "stablemax",
           "--monitor-every", "3", "--monitor-chunk", "256", "--ckpt-every", "3", "--grid-every", "3", "--log-every", "1", "--val-every", "100000"]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=ROOT)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    rows = [json.loads(l) for l in (out / "metrics.jsonl").read_text().splitlines() if l.strip()]
    loss = [x for x in rows if "loss" in x]; mon = [x["monitor"] for x in rows if "monitor" in x]
    assert loss and all(np.isfinite(x["loss"]) for x in loss) and all("commit_bce" in x for x in loss) and all("fpa_ce" in x for x in loss)
    assert mon and mon[-1]["n_val"] == 512 and "val_t16_ema" not in mon[-1] and "val_t2_ema" in mon[-1]   # T = 2 -> val_t2
    assert (out / "ckpt_000003.pkl").exists()
    cfg = json.loads((out / "config.json").read_text())["config"]
    assert cfg["dec_coupling_kind"] == "attn" and cfg["dec_commit"] and cfg["dec_commit_tau"] == 0.5
    # the evaluator's --record-commit on the grid
    eo = tmp_path / "ev"
    cmd = [sys.executable, str(ROOT / "tools/eval_sudoku_extreme.py"), "--ckpt", str(out / "ckpt_000003.pkl"), "--npz", str(NPZ512), "--out", str(eo),
           "--split", "test", "--limit", "6", "--t-total", "3", "--k-init", "0", "--batch", "4", "--ema", "--record-commit", "--record-by-step"]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=ROOT)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    arr = dict(np.load(eo / "records_all.npz")); summ = json.loads((eo / "summary_all.json").read_text())
    assert arr["commit_by_step"].shape == (6, 3, 81) and arr["cellok_by_step"].shape == (6, 3, 11) and arr["free_cells"].shape == (6, 11)
    assert summ["record_commit"] and summ["commit_tau"] == 0.5 and "commit_frac_last" in summ and "commit_auc_last" in summ
    c = arr["commit_by_step"].astype(np.float32); assert (c >= 0).all() and (c <= 1).all()
