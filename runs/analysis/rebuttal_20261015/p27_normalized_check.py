"""Exploratory reproduction of the PI's third-review check: P27's saved evaluation scores read as the correct-digit margin (mean over the four
evaluation cells, k = 1), standardized per puzzle by the pooled RMS of both directions' centered intact responses (P29's option-B normalization);
the full - P26-control four-mask donor contrast with a 20,000-draw puzzle bootstrap; the optimistic noise scenario's predicted SE beside it."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path("/Users/aakash/Projects/HRRN"); sys.path.insert(0, str(ROOT / "tools")); import rebuttal_p29 as T
for w in (192, 256):
    d = np.load(ROOT / f"runs/analysis/rebuttal_20261013/attention_{w}/p27.npz", allow_pickle=False); conds = [str(x) for x in d["conds"]]
    st, cond, arm, dose, mask = d["var_state"], d["var_cond"], d["var_arm"], d["var_dose"], d["var_mask"]; sol = d["state_sol"].astype(int); E = d["state_E"].astype(int)
    esc = d["escores"][:, 0].astype(np.float64)                                   # (V, 4 cells, 9 digits) at k = 1
    lab = np.stack([sol[s_, E[s_]] - 1 for s_ in st])                            # digits 1..9 -> index 0..8
    mg = T.margins(esc, lab) if False else np.array([T.margins(esc[i], lab[i]).mean() for i in range(len(st))])
    Y = {}; pid = d["state_pid"]; dirs = d["state_dir"]
    for i in range(len(st)):
        if arm[i] == 1: Y[(int(st[i]), int(cond[i]), int(dose[i]), int(mask[i]))] = mg[i]
    key = {(int(p), int(x)): s_ for s_, (p, x) in enumerate(zip(pid, dirs))}; don = {s_: key.get((int(p), 1 - int(x))) for s_, (p, x) in enumerate(zip(pid, dirs))}
    ix, jf, jr = conds.index("intact"), conds.index("full"), conds.index("full_random")
    sc = {}
    for p in np.unique(pid):
        v = []
        for s_ in [key.get((int(p), 0)), key.get((int(p), 1))]:
            if s_ is None: continue
            for i in range(3):
                a = np.array([Y[(s_, ix, i, m)] for m in range(4)]); v += list(a - a.mean())
        sc[int(p)] = np.sqrt(np.mean(np.square(v)))
    Yn = {k_: v / sc[int(pid[k_[0]])] for k_, v in Y.items()}
    for name, YY in (("raw", Y), ("normalized", Yn)):
        uF, uR = T.centered_units(YY, don, jf, ix, ix), T.centered_units(YY, don, jr, ix, ix); cl = pid[uF["state"]]
        dr = T.boot_betas([uF, uR], [1.0, -1.0], [cl, cl], [20261015, 27, w], 20000); lo, hi = T.ci(dr)
        u0 = T.centered_units(YY, don, ix, ix, ix); cD, cR = u0["cD"], u0["cR"]; n = len(cD); Xr = np.column_stack([cR, np.ones(n)]); rD = cD - Xr @ np.linalg.lstsq(Xr, cD, rcond=None)[0]
        vv = np.repeat([np.mean((cR[i:i + 4] ** 2 + cD[i:i + 4] ** 2) / 2) for i in range(0, n, 4)], 4); c0 = pid[u0["state"]]
        pred = np.sqrt(sum((rD[c0 == t] ** 2 * 2 * vv[c0 == t]).sum() for t in np.unique(c0))) / (rD @ rD)
        print(f"{w} {name:10s}: full - control {T.beta(uF) - T.beta(uR):+.3f} [{lo:+.3f}, {hi:+.3f}] | measured SE {(hi - lo) / 3.92:.3f} vs the noise model's predicted {pred:.3f} (ratio {(hi - lo) / 3.92 / pred:.1f})")
