# Ledger: verifies C1 (canvas/VOID) and the group machinery under CI-1 (equivariance).
import numpy as np
import pytest

from qhrrn2 import grid as G


def _probe(h=4, w=5, seed=0):
    return np.asarray(np.random.default_rng(seed).integers(0, 10, size=(h, w)), dtype=np.int8)


# ── C1: canvas, VOID, mask ──────────────────────────────────────────────────

def test_place_pads_with_void_not_black():
    g = _probe(3, 4)
    c = G.place(g)
    assert c.shape == (G.CANVAS, G.CANVAS)
    assert np.array_equal(c[:3, :4], g)
    assert np.all(c[~G.canvas_mask((3, 4))] == G.VOID)
    assert G.VOID != 0, "VOID must never be black (background/padding conflation hazard)"


def test_crop_inverts_place():
    g = _probe(7, 2)
    assert np.array_equal(G.crop(G.place(g), g.shape), g)


def test_one_hot_fields_partition():
    c = G.place(_probe())
    f = G.one_hot_fields(c)
    assert f.shape == (G.CANVAS, G.CANVAS, G.VOCAB)
    assert np.all(f.sum(-1) == 1.0)
    assert np.array_equal(f.argmax(-1), c)


def test_place_rejects_oversize_and_bad_values():
    with pytest.raises(ValueError):
        G.place(np.zeros((33, 4), dtype=np.int8))
    with pytest.raises(ValueError):
        G.place(np.full((2, 2), 10, dtype=np.int8))  # VOID not a legal input color


# ── D4 group ────────────────────────────────────────────────────────────────

def test_d4_inverse_table_round_trips():
    g = _probe()
    for k in range(8):
        assert np.array_equal(G.d4(G.d4(g, k), G.d4_inverse(k)), g), f"k={k}"


def test_d4_elements_distinct():
    g = _probe(4, 4, seed=1)  # square, so all 8 images comparable
    images = [G.d4(g, k).tobytes() for k in range(8)]
    assert len(set(images)) == 8, "the 8 D4 images of a generic grid must be distinct"


def test_d4_closure():
    g = _probe(4, 4, seed=2)
    all_images = {G.d4(g, k).tobytes() for k in range(8)}
    for a in range(8):
        for b in range(8):
            assert G.d4(G.d4(g, a), b).tobytes() in all_images


# ── S9 palette permutations ─────────────────────────────────────────────────

def test_palette_fixes_black_and_void():
    rng = np.random.default_rng(0)
    for _ in range(20):
        lut = G.random_palette(rng)
        assert lut[0] == 0, "black is a fixed point (Amendment A)"
        assert lut[G.VOID] == G.VOID, "VOID is outside the color group"
        assert sorted(lut[1:10].tolist()) == list(range(1, 10))


def test_palette_inverse_round_trips():
    rng = np.random.default_rng(1)
    g = _probe()
    for _ in range(10):
        lut = G.random_palette(rng)
        assert np.array_equal(G.apply_palette(G.apply_palette(g, lut), G.palette_inverse(lut)), g)


# ── Episode transforms ──────────────────────────────────────────────────────

def test_transform_invert_output_round_trips():
    rng = np.random.default_rng(2)
    y = _probe(6, 6, seed=3)
    for k in range(8):
        t = G.Transform(k=k, lut=G.random_palette(rng))
        assert np.array_equal(t.invert_output(t.apply(y)), y), f"k={k}"


def test_transform_episode_is_joint():
    eps = G.load_task("007bbfb7")
    t = G.Transform(k=3, lut=G.random_palette(np.random.default_rng(4)))
    te = G.transform_episode(eps[0], t)
    assert len(te.support) == len(eps[0].support)
    x0, _ = eps[0].support[0]
    assert np.array_equal(te.support[0][0], t.apply(x0))


def test_sample_orbit_starts_with_identity():
    orbit = G.sample_orbit(np.random.default_rng(5), 8)
    assert orbit[0].k == 0 and np.array_equal(orbit[0].lut, G.identity_palette())
    assert len(orbit) == 8


# ── Vendored data loading (no network — environment-drift rule) ──────────────────────────

def test_load_real_training_task():
    eps = G.load_task("007bbfb7")
    assert len(eps) >= 1
    ep = eps[0]
    assert len(ep.support) >= 2
    for x, y in ep.support:
        assert x.dtype == np.int8 and y.dtype == np.int8
        assert 0 <= x.min() and x.max() < G.NUM_COLORS
    assert ep.query_y is not None


def test_load_real_evaluation_task():
    ids = G.list_task_ids("evaluation")
    assert len(ids) == 400
    eps = G.load_task(ids[0])
    assert eps[0].support


def test_missing_task_raises():
    with pytest.raises(FileNotFoundError):
        G.load_task("ffffffff")
