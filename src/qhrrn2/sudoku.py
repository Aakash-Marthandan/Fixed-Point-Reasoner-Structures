# Ledger: S-PORT (H-33, the landscape-class law — registered 2026-08-14
# paper-track entry). Sudoku as the SINGLE-ATTRACTOR comparison domain for
# the ARC instrument suite: EqR reports 99.8% @ 5.03M on Sudoku-Extreme with
# randomized-init training as their biggest lever, while our RI arm bought
# convergence discipline and NO accuracy on ARC. H-33 says the difference is
# the LANDSCAPE CLASS (one valid completion per instance vs an inventory of
# rule-basins), and this module is the substrate that tests it.
#
# REPRESENTATION DECISIONS (each one a claim about symmetry, not convenience):
#   digits 1-9 -> colors 1-9.  Our S9 equivariance (C2/[P-C1]) IS Sudoku's
#     digit-relabeling symmetry, exactly and by construction — an inductive
#     bias neither EqR nor FPRM carries. Relabeling a solved grid yields a
#     solved grid; our network's output relabels identically, bit-exact.
#   blank -> color 0 (black).  Black is a FIXED POINT of every palette
#     permutation (Amendment A), so "unknown" is invariant under relabeling
#     — the only correct choice; VOID is reserved for off-canvas.
#   9x9 placed top-left on the 32x32 canvas, remainder VOID (one JIT shape,
#     the ARC model runs UNCHANGED — no architecture edits for the port).
#
# REGISTERED STRUCTURAL RISK (the 3-adic/2-adic mismatch): Sudoku's
# constraint geometry is 3-adic (3x3 boxes; 9 = 3^2) while our RG pyramid is
# dyadic (2x2 pooling; 32 = 2^5). Box boundaries therefore NEVER align with
# pooling blocks at any scale. If the port underperforms, this misalignment
# is a named candidate cause (testable: box-aligned placement/3-adic pooling
# variant), not a mystery — and it is itself a finding about whether RG
# coarse-graining must match a task's constraint arity.
#
# HONESTY NOTE (binding on any comparison): puzzles here are GENERATED with a
# uniqueness-preserving hole punch at a difficulty dial. They are NOT
# Sudoku-Extreme. No number from this module may be compared to EqR's 99.8%
# unless their benchmark is vendored and run; our numbers are labeled
# "generator Sudoku @ <givens> givens".
"""Sudoku puzzle generation + the ARC-canvas episode adapter (S-port)."""
from __future__ import annotations

import numpy as np

from . import grid as G

N = 9
BOX = 3
FULL = 0x1FF          # bitmask of digits 1..9 (bit d-1 set = digit d allowed)
BLANK = 0             # ARC black: the palette-permutation fixed point


# ── solver (bitmask backtracking; counts solutions with early exit) ────────

def _masks(board: np.ndarray):
    rows = [FULL] * N
    cols = [FULL] * N
    boxes = [FULL] * N
    for r in range(N):
        for c in range(N):
            d = int(board[r, c])
            if d:
                bit = 1 << (d - 1)
                rows[r] &= ~bit
                cols[c] &= ~bit
                boxes[(r // BOX) * BOX + c // BOX] &= ~bit
    return rows, cols, boxes


def count_solutions(board: np.ndarray, limit: int = 2, rng=None) -> int:
    """Number of completions, capped at `limit` (2 suffices for uniqueness)."""
    b = board.copy()
    rows, cols, boxes = _masks(b)
    found = 0

    def rec() -> bool:
        """True = stop (limit reached)."""
        nonlocal found
        # most-constrained cell first (this is what makes it fast)
        best, best_mask, best_n = None, 0, 10
        for r in range(N):
            for c in range(N):
                if b[r, c]:
                    continue
                m = rows[r] & cols[c] & boxes[(r // BOX) * BOX + c // BOX]
                n = bin(m).count("1")
                if n == 0:
                    return False          # dead end
                if n < best_n:
                    best, best_mask, best_n = (r, c), m, n
                    if n == 1:
                        break
            if best_n == 1:
                break
        if best is None:                  # no empty cell: a solution
            found += 1
            return found >= limit
        r, c = best
        bi = (r // BOX) * BOX + c // BOX
        digits = [d for d in range(1, N + 1) if best_mask & (1 << (d - 1))]
        if rng is not None:
            rng.shuffle(digits)
        for d in digits:
            bit = 1 << (d - 1)
            b[r, c] = d
            rows[r] &= ~bit; cols[c] &= ~bit; boxes[bi] &= ~bit
            stop = rec()
            b[r, c] = 0
            rows[r] |= bit; cols[c] |= bit; boxes[bi] |= bit
            if stop:
                return True
        return False

    rec()
    return found


def solve(board: np.ndarray, rng=None) -> np.ndarray | None:
    """Return one completion (or None). Used for generation and scoring."""
    b = board.copy()
    rows, cols, boxes = _masks(b)

    def rec() -> bool:
        best, best_mask, best_n = None, 0, 10
        for r in range(N):
            for c in range(N):
                if b[r, c]:
                    continue
                m = rows[r] & cols[c] & boxes[(r // BOX) * BOX + c // BOX]
                n = bin(m).count("1")
                if n == 0:
                    return False
                if n < best_n:
                    best, best_mask, best_n = (r, c), m, n
                    if n == 1:
                        break
            if best_n == 1:
                break
        if best is None:
            return True
        r, c = best
        bi = (r // BOX) * BOX + c // BOX
        digits = [d for d in range(1, N + 1) if best_mask & (1 << (d - 1))]
        if rng is not None:
            rng.shuffle(digits)
        for d in digits:
            bit = 1 << (d - 1)
            b[r, c] = d
            rows[r] &= ~bit; cols[c] &= ~bit; boxes[bi] &= ~bit
            if rec():
                return True
            b[r, c] = 0
            rows[r] |= bit; cols[c] |= bit; boxes[bi] |= bit
        return False

    return b if rec() else None


# ── generation ─────────────────────────────────────────────────────────────

def full_grid(rng: np.random.Generator) -> np.ndarray:
    """A uniformly-shuffled complete valid grid."""
    sol = solve(np.zeros((N, N), dtype=np.int8), rng=rng)
    assert sol is not None
    return sol.astype(np.int8)


def punch(sol: np.ndarray, rng: np.random.Generator, givens: int) -> np.ndarray:
    """Remove cells while the solution stays UNIQUE, down to `givens`."""
    puz = sol.copy()
    order = rng.permutation(N * N)
    for idx in order:
        if int((puz != BLANK).sum()) <= givens:
            break
        r, c = divmod(int(idx), N)
        if puz[r, c] == BLANK:
            continue
        keep = puz[r, c]
        puz[r, c] = BLANK
        if count_solutions(puz, limit=2) != 1:
            puz[r, c] = keep          # removal broke uniqueness — put it back
    return puz


def sample(rng: np.random.Generator, givens: int = 30):
    """One (puzzle, solution) pair, both 9x9 int8; puzzle blanks = 0."""
    sol = full_grid(rng)
    return punch(sol, rng, givens), sol


def sample_pairs(n: int, seed: int = 0, givens: int = 30):
    """n (x, y) pairs as 9x9 int8 arrays — the corpus's raw material."""
    rng = np.random.default_rng(seed)
    return [sample(rng, givens) for _ in range(n)]


def sample_ladder(n: int, seed: int, givens_list):
    """The DIFFICULTY LADDER instrument: n solution grids, each punched at
    every difficulty, so 'how many givens' becomes a WITHIN-GRID contrast
    (the same solution seen at 50/40/30 clues) instead of a between-sample
    one. That pairing is what lets a solve-rate falloff be attributed to
    required propagation depth rather than to grid luck.
    -> {givens: [(puzzle, solution), ...]} with solutions shared across keys.
    """
    rng = np.random.default_rng(seed)
    sols = [full_grid(rng) for _ in range(n)]
    out = {}
    for g in givens_list:
        prng = np.random.default_rng(seed + 1_000 + g)   # per-level, seeded
        out[g] = [(punch(s, prng, g), s) for s in sols]
    return out


# ── canvas LAYOUTS (SPRINT S2 wave 2, ledger 2026-08-22) ───────────────────
# "origin": the 9x9 grid at the canvas origin (every cell-1 / wave-1 arm).
# "box4":   the REGISTERED box-aligned control — each 3x3 box sits in the
#           top-left 3x3 of a 4x4 canvas block (a 12x12 window, the 4th row/col
#           of every block VOID), so box boundaries coincide with scale-2 (4x4)
#           pooling blocks: the 3-adic/dyadic mismatch named at the S-port
#           launch becomes a testable arm. Placement offsets for box4 must be
#           multiples of 4 (Corpus.off_stride) or the alignment is lost.
# "native9" (CHAMPION TRACK, 2026-09-01): the canvas IS the 9x9 grid — no
#           placement, no padding, no offsets; pairs with the 3-adic model
#           geometry (cfg.canvas=9, pool_arity=3, scales=2, mixer "group9").
LAYOUTS = ("origin", "box4", "native9")
BOX4 = 4
BOX4_EXTENT = BOX * BOX4        # 12


def layout_extent(layout: str = "origin") -> int:
    if layout in ("origin", "native9"):
        return N
    if layout == "box4":
        return BOX4_EXTENT
    raise ValueError(f"unknown layout {layout!r}")


def layout_canvas(layout: str = "origin") -> int:
    """Side length of the training/eval canvas for this layout."""
    return N if layout == "native9" else G.CANVAS


def box4_index() -> np.ndarray:
    """Window row (= col) index of grid row r: 4*(r//3) + r%3."""
    rr = np.arange(N)
    return (rr // BOX) * BOX4 + rr % BOX


def place_layout(g: np.ndarray, layout: str = "origin", oy: int = 0, ox: int = 0) -> np.ndarray:
    """9x9 digits/blanks -> canvas in the given layout (VOID elsewhere).
    native9: the identity — the 9x9 grid IS the canvas."""
    g = np.asarray(g, dtype=np.int8)
    assert g.shape == (N, N), g.shape
    if layout == "native9":
        assert oy == 0 and ox == 0, "native9 has no placement offsets"
        return g.copy()
    if layout == "origin":
        return G.place_at(g, oy, ox)
    if layout == "box4":
        win = np.full((BOX4_EXTENT, BOX4_EXTENT), G.VOID, dtype=np.int8)
        idx = box4_index()
        win[np.ix_(idx, idx)] = g
        canvas = np.full((G.CANVAS, G.CANVAS), G.VOID, dtype=np.int8)
        if oy < 0 or ox < 0 or oy + BOX4_EXTENT > G.CANVAS or ox + BOX4_EXTENT > G.CANVAS:
            raise ValueError("box4 window exceeds canvas")
        canvas[oy:oy + BOX4_EXTENT, ox:ox + BOX4_EXTENT] = win
        return canvas
    raise ValueError(f"unknown layout {layout!r}")


def unplace_layout(arr: np.ndarray, layout: str = "origin"):
    """Canvas (32x32) or the cropped window (placed at the origin) -> 9x9 digits,
    or None when the array is too small to decode. Gap cells are never read."""
    a = np.asarray(arr)
    e = layout_extent(layout)
    if a.ndim != 2 or a.shape[0] < e or a.shape[1] < e:
        return None
    if layout in ("origin", "native9"):
        return a[:N, :N]
    idx = box4_index()
    return a[np.ix_(idx, idx)]


# ── validity + the canvas adapter ─────────────────────────────────────────

def is_valid_solution(g: np.ndarray) -> bool:
    want = set(range(1, N + 1))
    if g.shape != (N, N) or set(np.unique(g)) != want:
        return False
    for i in range(N):
        if set(g[i, :]) != want or set(g[:, i]) != want:
            return False
    for br in range(0, N, BOX):
        for bc in range(0, N, BOX):
            if set(g[br:br + BOX, bc:bc + BOX].ravel()) != want:
                return False
    return True


def agrees_on_givens(puz: np.ndarray, sol: np.ndarray) -> bool:
    m = puz != BLANK
    return bool(np.array_equal(puz[m], sol[m]))


def to_canvas(g: np.ndarray) -> np.ndarray:
    """9x9 digits/blanks -> 32x32 ARC canvas (VOID outside)."""
    return G.place(np.asarray(g, dtype=np.int8))


def episode(puz: np.ndarray, sol: np.ndarray, task_id: str = "sudoku") -> G.Episode:
    """One instance as an Episode. Sudoku's rule is UNIVERSAL, so there are no
    support pairs to infer it from: the instance IS the query (the EqR/FPRM
    direct-completion convention). Training uses (x, y) pairs under ONE task
    row — one rule, many instances — which is the structural contrast with
    ARC's per-task rule inference that H-33 is about."""
    return G.Episode(task_id=task_id, support=(), query_x=np.asarray(puz, np.int8),
                     query_y=np.asarray(sol, np.int8))
