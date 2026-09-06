import sys, time, numpy as np, torch
sys.path.insert(0, "/Users/aakash/Projects/HRRN/runs/field_ckpts/harness"); sys.path.insert(0, "/Users/aakash/Projects/HRRN/src")
import field_models as FM
from qhrrn2 import sudoku_extreme as SX
tag, dev = sys.argv[1], sys.argv[2]; B = int(sys.argv[3]); D = 16
d = SX.load_prepared("/Users/aakash/Projects/HRRN/data/sudoku_extreme/sudoku_extreme_seed0.npz")
ids = SX.stratified_subsample(d["test_rating"], 256, 20260821)[:B]
puz, sol = d["test_q"][ids].astype(np.int64), d["test_a"][ids].astype(np.int64)
kw = dict(noise_scale=0.0) if tag == "eqr" else {}
m = FM.load(tag, dev, torch.float32, **kw); print(f"{tag} on {dev}: params {FM.count_params(m):,}")
batch = m.tokens(puz); torch.manual_seed(0)
st = m.init_state(B, "trunc" if tag == "eqr" else "fixed", gen=torch.Generator().manual_seed(0))
t0 = time.time()
for t in range(D):
    st, lg, q = m.step(st, batch)
    if dev != "cpu": torch.mps.synchronize()
dt = time.time() - t0
pred = m.logits9(lg).argmax(-1).cpu().numpy() + 1; ex = (pred == sol.reshape(-1, 81)).all(1)
print(f"  exact@{D} on {B} strat puzzles: {ex.mean():.3f} | argmax-not-digit cells: {(~m.full_argmax_is_digit(lg)).float().mean():.4f} | q_halt mean {q.mean():.2f} | {dt/D/B*1000:.1f} ms/step/puzzle ({dt:.1f}s total)")
