"""Extension of float32_check.py (exploratory; requested in the PI's review of 2026-10-12): the executed readout tolerance of P28's transplants
over EVERY eligible puzzle per receiver (128 x 12 pairs x 4 arms), not two. Rebuild the banks, construct every arm exactly as the runner does
(float64 construction, float32 cast) and measure the displayed-score change of the float32 state against the recipient's."""
import json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path("/Users/aakash/Projects/HRRN"); sys.path.insert(0, str(ROOT / "tools")); import rebuttal_p28 as T28, rebuttal_p27 as P27
B = ROOT / "runs/analysis/rebuttal_20261014"; res = {}
for w in (192, 256):
    t0 = time.time(); d = np.load(B / f"attention_{w}/p28.npz", allow_pickle=False); pids = np.array(d["pids"], np.int64)
    r = T28.R22(w, Path(sys.argv[1]), tag="f32all"); U = P27.load_basis(w, r.lm); worst = {a: 0.0 for a in T28.ARMS}; flips = 0; n = 0
    for b0 in range(0, len(pids), 32):
        bk = r.bank(pids[b0:b0 + 32])
        for p in range(len(bk["puz"])):
            for (i, j) in T28.PAIRS:
                st, _ = r.transplant(bk, pids[b0 + p], p, i, j, U); hr = bk["z"][p, i][0].astype(np.float64); s0 = np.einsum("fsw,w->sf", hr, r.lm); n += 1
                for a in T28.ARMS:
                    s1 = np.einsum("fsw,w->sf", st[a][0].astype(np.float64), r.lm); worst[a] = max(worst[a], float(np.abs(s1 - s0).max())); flips += int((s1.argmax(1) != s0.argmax(1)).sum())
    res[w] = dict(puzzles=len(pids), pair_directions=n, max_score_change=worst, flips=flips, wall_s=round(time.time() - t0))
    print(f"attention_{w}: {len(pids)} puzzles, {n} ordered pairs: max |score change| of the float32 states: " + ", ".join(f"{a} {v:.2e}" for a, v in worst.items()) + f" | displayed-digit flips {flips} | {time.time() - t0:.0f} s", flush=True)
(B / "float32_check_all.json").write_text(json.dumps(res, indent=1) + "\n")
