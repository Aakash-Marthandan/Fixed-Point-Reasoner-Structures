# Ledger: C1 (canvas/VOID/mask), substrate for C2 (S9 color axis) and CI-1 (equivariance).
# Data layer is pure numpy; JAX enters at the model boundary.
"""Grids, the canvas, the D4 group, color permutations, and ARC episodes.

Representation decisions (Design_Ledger C1, QHRRN2_Architecture §1):
- Cell alphabet: 0..9 ARC colors, VOID = 10. VOID is a real state, never
  conflated with black (April failure E4).
- Canvas: fixed 32x32; a grid is placed top-left; the rest is VOID.
- Color symmetry group: S9 over colors 1..9. Black (0) and VOID are fixed
  points of every palette permutation (Amendment A).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

NUM_COLORS = 10          # ARC colors 0..9
VOID = 10                # canvas cells outside the true grid
VOCAB = NUM_COLORS + 1   # categorical states incl. VOID
CANVAS = 32              # covers ARC's 30x30 maximum; one JIT shape

ARC_DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "ARC-AGI" / "data"

# ── Canvas ─────────────────────────────────────────────────────────────────

def place(grid: np.ndarray) -> np.ndarray:
    """Place an (H, W) grid of colors 0..9 top-left on a VOID-filled canvas."""
    return place_at(grid, 0, 0)


def place_at(grid: np.ndarray, oy: int, ox: int) -> np.ndarray:
    """Place a grid at offset (oy, ox) — translation augmentation (ledger C10)."""
    grid = np.asarray(grid, dtype=np.int8)
    h, w = grid.shape
    if oy < 0 or ox < 0 or oy + h > CANVAS or ox + w > CANVAS:
        raise ValueError(f"grid {h}x{w} at ({oy},{ox}) exceeds canvas {CANVAS}")
    if grid.min() < 0 or grid.max() >= NUM_COLORS:
        raise ValueError("grid values must be ARC colors 0..9")
    canvas = np.full((CANVAS, CANVAS), VOID, dtype=np.int8)
    canvas[oy:oy + h, ox:ox + w] = grid
    return canvas


def canvas_mask(shape: tuple[int, int]) -> np.ndarray:
    """Boolean (CANVAS, CANVAS) mask that is True on the true-grid region."""
    h, w = shape
    mask = np.zeros((CANVAS, CANVAS), dtype=bool)
    mask[:h, :w] = True
    return mask


def one_hot_fields(canvas: np.ndarray) -> np.ndarray:
    """(CANVAS, CANVAS) int -> (CANVAS, CANVAS, VOCAB) float32 occupancy fields."""
    return np.eye(VOCAB, dtype=np.float32)[canvas]


def crop(canvas: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Inverse of place: recover the (H, W) grid from the canvas top-left."""
    h, w = shape
    return canvas[:h, :w]

# ── D4: the 8 rigid symmetries of the square ───────────────────────────────
# Convention: k = 4*flip + rot; apply rot90 `rot` times, then fliplr if flip.

def d4(grid: np.ndarray, k: int) -> np.ndarray:
    if not 0 <= k < 8:
        raise ValueError(f"D4 index must be 0..7, got {k}")
    out = np.rot90(grid, k % 4)
    if k >= 4:
        out = np.fliplr(out)
    return np.ascontiguousarray(out)


def _compute_d4_inverses() -> tuple[int, ...]:
    """Derive the inverse table by probing, so no hand-derivation can be wrong."""
    probe = np.arange(6).reshape(2, 3)  # no self-symmetry
    inv = []
    for k in range(8):
        gk = d4(probe, k)
        matches = [j for j in range(8) if np.array_equal(d4(gk, j), probe)]
        assert len(matches) == 1, f"D4 inverse of {k} not unique: {matches}"
        inv.append(matches[0])
    return tuple(inv)


D4_INVERSE: tuple[int, ...] = _compute_d4_inverses()


def d4_inverse(k: int) -> int:
    return D4_INVERSE[k]

# ── Color permutations: S9 over colors 1..9; black and VOID fixed ──────────

def identity_palette() -> np.ndarray:
    """Length-VOCAB lookup table mapping every state to itself."""
    return np.arange(VOCAB, dtype=np.int8)


def random_palette(rng: np.random.Generator) -> np.ndarray:
    """Random S9 element as a lookup table; fixes black (0) and VOID."""
    lut = identity_palette()
    lut[1:NUM_COLORS] = rng.permutation(np.arange(1, NUM_COLORS)).astype(np.int8)
    return lut


def palette_inverse(lut: np.ndarray) -> np.ndarray:
    inv = np.empty_like(lut)
    inv[lut] = np.arange(len(lut), dtype=lut.dtype)
    return inv


def apply_palette(grid: np.ndarray, lut: np.ndarray) -> np.ndarray:
    return lut[np.asarray(grid, dtype=np.int8)]

# ── Episodes and the symmetry orbit ────────────────────────────────────────

@dataclass(frozen=True)
class Episode:
    """One ARC task instance: demonstration pairs plus one query."""
    task_id: str
    support: tuple[tuple[np.ndarray, np.ndarray], ...]
    query_x: np.ndarray
    query_y: np.ndarray | None = None  # None for hidden test outputs


@dataclass(frozen=True)
class Transform:
    """A joint symmetry applied identically to every grid in an episode."""
    k: int                      # D4 element
    lut: np.ndarray = field(default_factory=identity_palette)

    def apply(self, grid: np.ndarray) -> np.ndarray:
        return apply_palette(d4(grid, self.k), self.lut)

    def invert_output(self, grid: np.ndarray) -> np.ndarray:
        """Undo this transform on a predicted output grid."""
        return d4(apply_palette(grid, palette_inverse(self.lut)), d4_inverse(self.k))


def transform_episode(ep: Episode, t: Transform) -> Episode:
    return Episode(
        task_id=ep.task_id,
        support=tuple((t.apply(x), t.apply(y)) for x, y in ep.support),
        query_x=t.apply(ep.query_x),
        query_y=None if ep.query_y is None else t.apply(ep.query_y),
    )


def sample_orbit(rng: np.random.Generator, n: int, use_d4: bool = True,
                 use_palette: bool = True) -> list[Transform]:
    """n distinct-ish joint transforms; always includes the identity first."""
    out = [Transform(k=0)]
    while len(out) < n:
        k = int(rng.integers(8)) if use_d4 else 0
        lut = random_palette(rng) if use_palette else identity_palette()
        out.append(Transform(k=k, lut=lut))
    return out

# ── Local ARC loading (vendored data only — no network, April E8) ──────────

def task_path(task_id: str, root: Path = ARC_DATA_ROOT) -> Path:
    for split in ("training", "evaluation"):
        p = root / split / f"{task_id}.json"
        if p.exists():
            return p
    raise FileNotFoundError(f"task {task_id} not found under {root}")


def load_task(task_id: str, root: Path = ARC_DATA_ROOT) -> list[Episode]:
    """One Episode per test pair (most tasks have exactly one)."""
    with open(task_path(task_id, root)) as f:
        raw = json.load(f)
    support = tuple(
        (np.asarray(p["input"], dtype=np.int8), np.asarray(p["output"], dtype=np.int8))
        for p in raw["train"]
    )
    episodes = []
    for p in raw["test"]:
        episodes.append(Episode(
            task_id=task_id,
            support=support,
            query_x=np.asarray(p["input"], dtype=np.int8),
            query_y=np.asarray(p["output"], dtype=np.int8) if "output" in p else None,
        ))
    return episodes


def list_task_ids(split: str, root: Path = ARC_DATA_ROOT) -> list[str]:
    return sorted(p.stem for p in (root / split).glob("*.json"))
