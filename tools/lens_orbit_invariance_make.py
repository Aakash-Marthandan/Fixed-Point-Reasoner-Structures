"""Build position-group-transformed copies of the test arrays (one fixed group element per file; the puzzle ORDER is kept, so
--stratified 512 selects the same puzzle indices and every comparison is paired per puzzle). g0 = identity (the CPU-route
baseline), g1 / g2 = two position elements (transpose + band/row + stack/col permutations, no digit map), g3 = a DIGIT
permutation only (exact S9 equivariance -> the DEC must be invariant up to float associativity)."""
import numpy as np, sys
sys.path.insert(0, "/Users/aakash/Projects/HRRN/src")
from qhrrn2 import sudoku_extreme as SX
src = "/Users/aakash/Projects/HRRN/data/sudoku_extreme/sudoku_extreme_seed0.npz"
z = np.load(src, allow_pickle=False); keys = list(z.files)
Q, A = z["test_q"], z["test_a"]; print("test arrays", Q.shape, A.shape, Q.dtype, "keys", keys)
BOX, N = 3, 9
def element(seed, digit=False, position=True):
    rng = np.random.default_rng(seed)
    T = bool(rng.random() < 0.5) if position else False
    bands = rng.permutation(BOX); rows = np.concatenate([b * BOX + rng.permutation(BOX) for b in bands])
    stacks = rng.permutation(BOX); cols = np.concatenate([b * BOX + rng.permutation(BOX) for b in stacks])
    dmap = np.concatenate([[0], rng.permutation(N) + 1]).astype(np.int8) if digit else np.arange(10, dtype=np.int8)
    return dict(T=T, rows=rows if position else np.arange(9), cols=cols if position else np.arange(9), dmap=dmap)
def apply(arr, g):
    a = arr.reshape(len(arr), 9, 9)
    if g["T"]: a = np.transpose(a, (0, 2, 1))
    a = a[:, g["rows"]][:, :, g["cols"]]; a = g["dmap"][a]
    return np.ascontiguousarray(a.reshape(arr.shape)).astype(arr.dtype)
for name, g in (("g0", element(0, position=False)), ("g1", element(1)), ("g2", element(2)), ("g3", element(3, digit=True, position=False))):
    out = {k: z[k] for k in keys}
    out["test_q"] = apply(Q, g); out["test_a"] = apply(A, g)
    # validity: the transformed solution must still solve the transformed puzzle (givens agree) and be a valid Sudoku
    q9, a9 = out["test_q"].reshape(-1, 9, 9), out["test_a"].reshape(-1, 9, 9)
    assert ((q9 == 0) | (q9 == a9)).all()
    ok = all(sorted(a9[0][i]) == list(range(1, 10)) for i in range(9)) and all(sorted(a9[0][:, j]) == list(range(1, 10)) for j in range(9))
    np.savez(f"/Users/aakash/Projects/HRRN/runs/_orbit_{name}.npz", **out)
    print(name, "T" if g["T"] else "-", "rows", g["rows"].tolist(), "cols", g["cols"].tolist(), "dmap", g["dmap"].tolist(), "row/col valid on #0:", ok)
