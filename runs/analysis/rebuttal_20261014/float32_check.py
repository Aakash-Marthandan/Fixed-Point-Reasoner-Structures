"""The executed readout tolerance of P28's transplants: rebuild two eligible puzzles' banks per receiver, construct every arm exactly as the
runner does (float64 construction, float32 cast), and measure the displayed-score change of the float32 state against the recipient's."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path("/Users/aakash/Projects/HRRN"); sys.path.insert(0, str(ROOT / "tools")); import rebuttal_p28 as T28, rebuttal_p27 as P27
B = ROOT / "runs/analysis/rebuttal_20261014"
for w in (192, 256):
    d = np.load(B / f"attention_{w}/p28.npz", allow_pickle=False); pids = np.array(d["pids"][:2], np.int64)
    r = T28.R22(w, Path("/private/tmp/claude-501/-Users-aakash-Projects-HRRN/29fa6425-c970-40e0-b70e-5aa7c3c811d3/scratchpad/p28_f32"), tag="f32"); U = P27.load_basis(w, r.lm)
    bk = r.bank(pids); worst = {a: 0.0 for a in T28.ARMS}; flips = 0
    for p in range(len(pids)):
        for (i, j) in T28.PAIRS:
            st, _ = r.transplant(bk, pids[p], p, i, j, U); hr = bk["z"][p, i][0].astype(np.float64); s0 = np.einsum("fsw,w->sf", hr, r.lm)
            for a in T28.ARMS:
                s1 = np.einsum("fsw,w->sf", st[a][0].astype(np.float64), r.lm); worst[a] = max(worst[a], float(np.abs(s1 - s0).max())); flips += int((s1.argmax(1) != s0.argmax(1)).sum())
    print(f"attention_{w}: max |score change| of the float32 states: " + ", ".join(f"{a} {v:.2e}" for a, v in worst.items()) + f" | displayed-digit flips {flips}")
