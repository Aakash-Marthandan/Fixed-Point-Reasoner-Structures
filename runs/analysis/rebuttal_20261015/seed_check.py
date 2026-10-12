"""Exploratory: Monte Carlo stability of the primary 'beyond the control' lower bound (20,000-draw task-clustered bootstrap, other seeds)."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path("/Users/aakash/Projects/HRRN"); sys.path.insert(0, str(ROOT / "tools")); import rebuttal_p29 as T
d = T.load("conf"); conds = [str(x) for x in d["conds"]]; ix, jf, jc = conds.index("intact"), conds.index("full"), conds.index("full_ctrl")
don = T.donor_map(d); tcl = np.unique(d["state_task"], return_inverse=True)[1]; Y, B, dk, exc = T.tables(d, "margin_std", 4, 1, don)
uF, uC = T.centered_units(Y, dk, jf, ix, ix), T.centered_units(Y, dk, jc, ix, ix); cl = tcl[uF["state"]]
for seed in (1, 2, 3, 4):
    dr = T.boot_betas([uF, uC], [1.0, -1.0], [cl, cl], [seed, 29], 20000); lo, hi = T.ci(dr); print(f"seed {seed}: full - ctrl [{lo:+.3f}, {hi:+.3f}], P(draw <= 0) {np.mean(dr <= 0):.4f}")
