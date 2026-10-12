"""Exploratory sanity check after the frozen P30 analysis: every feature's spread, and its univariate within-puzzle AUC (both signs reported via
|AUC - 0.5|), for the leading transplants and restarts on both receivers; a puzzle-bootstrap interval for the best single feature."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path("/Users/aakash/Projects/HRRN"); sys.path.insert(0, str(ROOT / "tools")); import rebuttal_p30 as T
for w in (192, 256):
    F = dict(np.load(T.OUT / f"p30_features_{w}.npz", allow_pickle=False)); d = dict(np.load(T.P28DIR / f"attention_{w}/p28.npz", allow_pickle=False))
    for fam, names in (("lead", T.REC_F + T.PAIR_F), ("restart", T.RST_F)):
        X, y, g, o = T.family_rows(w, fam, F, d); best = None; row = []
        for c, nm in enumerate(names):
            pa = T.within_auc_parts(X[:, c], y, g); a = sum(v[0] for v in pa.values()) / sum(v[1] for v in pa.values()); row.append(f"{nm} {a:.3f} (sd {X[:, c].std():.3g})")
            if best is None or abs(a - 0.5) > abs(best[1] - 0.5): best = (c, a, nm, pa)
        keys = sorted(best[3]); num = np.array([best[3][k][0] for k in keys]); den = np.array([best[3][k][1] for k in keys]); rng = np.random.default_rng(1)
        bs = [(lambda i: num[i].sum() / den[i].sum())(rng.integers(0, len(keys), len(keys))) for _ in range(2000)]
        print(f"{w} {fam}: " + "; ".join(row)); print(f"   best single feature {best[2]}: within-puzzle AUC {best[1]:.3f} [{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}] (chosen as the best of {len(names)}: not corrected)")
