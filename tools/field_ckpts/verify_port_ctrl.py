"""Control for I12: PyTorch fp64 vs fp32 on the SAME model/puzzles (is the JAX-vs-torch gap numerical chaos of the recurrence?)."""
import sys, json, numpy as np, torch
from pathlib import Path
H = Path(__file__).resolve().parent; FC = H.parent; ROOT = FC.parents[1]; sys.path.insert(0, str(H)); sys.path.insert(0, str(ROOT / "src"))
import field_models as FM
from qhrrn2 import sudoku_extreme as SX
d = SX.load_prepared(ROOT / "data/sudoku_extreme/sudoku_extreme_seed0.npz"); ids = SX.stratified_subsample(d["test_rating"], 256, 20260821)[:64]
puz = d["test_q"][ids].astype(np.int64); sol = d["test_a"][ids].reshape(64, 81).astype(np.int64)
m32 = FM.load("trm", "cpu", torch.float32); m64 = FM.load("trm", "cpu", torch.float32); m64.inner.double(); m64.dtype = torch.float64
b32, b64 = m32.tokens(puz), m64.tokens(puz); s32, s64 = m32.init_state(64, "fixed"), m64.init_state(64, "fixed"); rows = []
for t in range(16):
    s32, l32, q32 = m32.step(s32, b32); s64, l64, q64 = m64.step(s64, b64)
    a, b = m32.logits9(l32).numpy(), m64.logits9(l64).numpy(); ea = (a.argmax(-1) + 1 == sol).all(1); eb = (b.argmax(-1) + 1 == sol).all(1)
    rows.append(dict(t=t + 1, max_abs=float(np.abs(a - b).max()), mean_abs=float(np.abs(a - b).mean()), argmax_agree=float((a.argmax(-1) == b.argmax(-1)).mean()), exact32=float(ea.mean()), exact64=float(eb.mean()), exact_agree=float((ea == eb).mean())))
    print(json.dumps(rows[-1]), flush=True)
json.dump(rows, open(FC / "out" / "verify_port" / "ctrl_fp64_vs_fp32.json", "w"), indent=1)
