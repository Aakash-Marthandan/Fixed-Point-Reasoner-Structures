"""Reproduce the external review's P27 checks (exploratory): the four-iteration horizon of the registered contrasts, the mean repair change from
leading and full transplantation (untouched cells, corrected arms), and the 256 leading effect with 20,000 bootstrap draws."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path("/Users/aakash/Projects/HRRN"); sys.path.insert(0, str(ROOT / "tools")); import rebuttal_p27 as P27, rebuttal_p26 as T
B = ROOT / "runs/analysis/rebuttal_20261013"
def at_k(d, k):
    dd = {key: d[key] for key in d.files}; dd["disp_e"] = d["disp_e"][:, k - 1:k]; return type("D", (), {"__getitem__": lambda s, key: dd[key], "files": list(dd)})()
for w in (192, 256):
    d = np.load(B / f"attention_{w}/p27.npz", allow_pickle=False); conds = [str(x) for x in d["conds"]]; pids = d["state_pid"]
    print(f"=== attention_{w}")
    for k in (1, 2, 3, 4):
        dk = at_k(d, k); Y, Bm = T.unit_tables(dk); don = T.donor_state(dk); rw = lambda a, b: P27.centered_rows(Y, don, conds, pids, a, b)
        out = []
        for name, (a, b) in (("S", ("full", "full_random")), ("E_L", ("lead", "lead_ctrl")), ("E_total", ("full", "comb_ctrl"))):
            r = rw(a, b); lo, hi = P27.boot_arr(P27.beta, [r], [20261013, 27, w, 100 + k], n=2000); out.append(f"{name} {P27.beta(r):+.3f} [{lo:+.3f}, {hi:+.3f}]")
        print(f"   k={k}: " + " | ".join(out))
    Y, Bm = T.unit_tables(d); rows = []
    for s in range(len(pids)):
        rows.append(dict(pid=int(pids[s]), lead=np.mean([Y[(s, conds.index("lead"), i, m)] - Y[(s, 0, i, m)] for i in range(3) for m in range(4)]),
                         full=np.mean([Y[(s, conds.index("full"), i, m)] - Y[(s, 0, i, m)] for i in range(3) for m in range(4)])))
    for c in ("lead", "full"):
        f = lambda q, c=c: float(np.mean([x[c] for x in q])); lo, hi = T.boot(f, [rows], [20261013, 27, w, 200]); print(f"   {c} - intact repair, corrected arms (pp): {100 * f(rows):+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]")
    don = T.donor_state(d); r = P27.centered_rows(Y, don, conds, pids, "lead", "lead_ctrl"); lo, hi = P27.boot_arr(P27.beta, [r], [20261013, 27, w, 300], n=20000)
    print(f"   E_L with 20,000 draws: {P27.beta(r):+.3f} [{lo:+.4f}, {hi:+.4f}]")
