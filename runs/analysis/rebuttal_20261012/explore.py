"""P26 exploratory additions (after the frozen report; labeled exploratory): intervals for the secondary quantities the reader reports without
intervals (R1 for the common basis, R2 and its half-test for the adaptive basis), and the four-iteration horizon of R1/R2 (full vs matched random)."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path("/Users/aakash/Projects/HRRN"); sys.path.insert(0, str(ROOT / "tools")); import rebuttal_p26 as T
B = ROOT / "runs/analysis/rebuttal_20261012"; SEED = 20261012
def at_k(d, k):
    """A view of the stage file whose k=1 display is replaced by iteration k's (the reader's functions read disp_e[:, 0])."""
    dd = {key: d[key] for key in d.files}; dd["disp_e"] = d["disp_e"][:, k - 1:k]; return type("D", (), {"__getitem__": lambda s, key: dd[key], "files": list(dd)})()
for w in (192, 256):
    d = np.load(B / f"attention_{w}/p26_C.npz", allow_pickle=False)
    r1c = T.r1_rows(d, "global", "global_rand"); e = lambda r: T.r1_beta(r, "x") - T.r1_beta(r, "ref")
    print(f"attention_{w} C: R1 common basis {e(r1c):+.3f} {np.round(T.boot(e, [r1c], [SEED, 26, w, 11]), 3)}")
    ra, r2 = T.r2_rows(d, "adaptive", "adaptive_rand"), T.r2_rows(d, "full", "full_random"); half = lambda a, b: T.r2_beta(a) - 0.5 * T.r2_beta(b)
    print(f"   R2 adaptive {T.r2_beta(ra):+.3f} {np.round(T.boot(T.r2_beta, [ra], [SEED, 26, w, 12]), 3)} | adaptive - E_full/2 {half(ra, r2):+.3f} {np.round(T.boot(half, [ra, r2], [SEED, 26, w, 13]), 3)}")
    rg = T.r2_rows(d, "global", "global_rand"); diff = lambda a, b: T.r2_beta(b) - T.r2_beta(a)
    print(f"   R2 full minus common-basis transfer (the residual's share) {diff(rg, r2):+.3f} {np.round(T.boot(diff, [rg, r2], [SEED, 26, w, 14]), 3)}")
    for k in (1, 2, 3, 4):
        dk = at_k(d, k); rr = T.r1_rows(dk, "full", "full_random"); q2 = T.r2_rows(dk, "full", "full_random"); qg = T.r2_rows(dk, "global", "global_rand")
        print(f"   k={k}: R1 full-random {e(rr):+.3f} {np.round(T.boot(e, [rr], [SEED, 26, w, 20 + k], n=1000), 3)} | R2 {T.r2_beta(q2):+.3f} {np.round(T.boot(T.r2_beta, [q2], [SEED, 26, w, 30 + k], n=1000), 3)} | R2 common basis {T.r2_beta(qg):+.3f}")
