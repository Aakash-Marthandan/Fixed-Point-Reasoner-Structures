"""Exploratory reproduction of the PI's review of the P29 outcome (2026-10-12): outcome influence vs predictor information, the influential task's
omission, the control response scale on it, correctness variation, and the challenged-arm repair means. No registered quantity is replaced."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path("/Users/aakash/Projects/HRRN"); sys.path.insert(0, str(ROOT / "tools")); import rebuttal_p29 as T
d = T.load("conf"); conds = [str(x) for x in d["conds"]]; ix, jf, jc = conds.index("intact"), conds.index("full"), conds.index("full_ctrl")
don = T.donor_map(d); task = np.array([str(x) for x in d["state_task"]]); tcl = np.unique(task, return_inverse=True)[1]
Y, B, dk, exc = T.tables(d, "margin_std", 4, 1, don); uF, uC = T.centered_units(Y, dk, jf, ix, ix), T.centered_units(Y, dk, jc, ix, ix)
tk = task[uF["state"]]; cl = tcl[uF["state"]]; sub = lambda u, m: {k: v[m] for k, v in u.items()}; ut = np.unique(tk)
diff = lambda m: T.beta(sub(uF, m)) - T.beta(sub(uC, m)); full = diff(np.ones(len(tk), bool))
th = np.array([diff(tk != t) for t in ut]); dev2 = (th - th.mean()) ** 2; top = ut[int(np.argmax(dev2))]
print(f"diff {full:+.4f}; jackknife: top task {top} holds {dev2.max() / dev2.sum():.3f} of the jackknife variance; min/max single-task omission {th.min():+.4f}/{th.max():+.4f}; all positive {bool((th > 0).all())}")
m = tk != top; dr = T.boot_betas([sub(uF, m), sub(uC, m)], [1.0, -1.0], [cl[m], cl[m]], [20261015, 29, 40], 20000); lo, hi = T.ci(dr)
print(f"without {top}: diff {diff(m):+.4f} [{lo:+.3f}, {hi:+.3f}]")
ss = [s for s in sorted(dk) if dk[s] is not None and task[s] == top]
for s in ss:
    for c, j in (("intact", ix), ("full", jf), ("full_ctrl", jc)):
        v = np.array([[Y[(s, j, i, mm)] for mm in range(4)] for i in range(3)]); r = np.sqrt(np.mean((v - v.mean(1, keepdims=True)) ** 2))
        print(f"   {top} state {s} dir {int(d['state_dir'][s])}: {c} centered response RMS / query scale {r:.2f}")
Yc, Bc, _, _ = T.tables(d, "correct", 4, 1, don); n = vary = 0
for s in sorted(don):
    if don[s] is None: continue
    for i in range(3):
        n += 1; vary += np.ptp([Yc[(s, ix, i, mm)] for mm in range(4)]) > 1e-12
print(f"correctness: intact groups varying across masks {vary}/{n} = {vary / n:.3f}")
for c, j in (("intact", ix), ("full", jf), ("full_ctrl", jc)):
    print(f"challenged arms, mean evaluation-token correctness at k=1, {c}: {100 * np.mean([v for (s, jj, i, mm), v in Yc.items() if jj == j]):.2f} %")
