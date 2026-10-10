#!/usr/bin/env python3
# Ledger: CORRECTION 2026-10-08 (found by an external review, relayed and confirmed read-only by session 3beab6; confirmed and extended to P11 by
# session d378a5). tools/rebuttal_p9.py builds ONE hook moment per puzzle with the primary time winning (`hooked = {**ctrl, **prim}`, line 100),
# while its report reads every control at the CONTROL time (lines 147-148). A puzzle that is both a primary case and a control is therefore
# intervened at its primary time and scored at its control time: that control reading is not a swap at the control time. The registered letters
# use primary cases only and are unaffected. This tool recomputes the exploratory control readings from the saved arrays on the CLEAN controls
# (controls that are not primary cases, whose hook fired at their own time), beside the readings as reported. Inference-free; no rerun.
"""  .venv/bin/python tools/verify_p9_controls.py            (P9 at 94k; P11 at 90k and 114k)
  .venv/bin/python tools/verify_p9_controls.py --selftest"""
import argparse, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FILES = {"P9 94k": "runs/analysis/rebuttal_20261001g/p9.npz", "P11 90k": "runs/analysis/rebuttal_20261001i/s090000/p9.npz",
         "P11 114k": "runs/analysis/rebuttal_20261001i/s114000/p9.npz"}

def readings(D, prim, ctrl):
    """-> {label: (overlap, clean, {swap: (exact at t+1 as reported over all controls, n), ...}, {swap: (exact at t+1 on clean controls, n)})}."""
    at = lambda ex, n, t: bool(ex[t, n])                                    # iteration t+1 is row index t (rebuttal_p9.report)
    clean = {n: t for n, t in ctrl.items() if n not in prim}
    out = {}
    for C, lab in ((ctrl, "reported"), (clean, "clean")):
        out[lab] = {f"F<-{k}G": (sum(at(D[f"ex_F_from_{k}G"], n, t) for n, t in C.items()), len(C)) for k in ("perp", "h", "l")}
        out[lab].update({f"G<-{k}F": (sum(at(D[f"ex_G_from_{k}F"], n, t) for n, t in C.items()), len(C)) for k in ("perp", "h", "l")})
    out["overlap_different_time"] = sum(1 for n in ctrl if n in prim and prim[n] != ctrl[n]); out["overlap"] = sum(1 for n in ctrl if n in prim)
    return out

def main_run():
    for name, f in FILES.items():
        D = np.load(ROOT / f, allow_pickle=True); m = json.loads(str(D["meta"]))
        prim = {int(k): v for k, v in m["primary"].items()}; ctrl = {int(k): v for k, v in m["controls"].items()}
        r = readings(D, prim, ctrl)
        print(f"{name} ({f}): primary {len(prim)}, controls {len(ctrl)}; controls that are also primary cases {r['overlap']} "
              f"(at a different time {r['overlap_different_time']}); clean controls {len(ctrl) - r['overlap']}")
        for lab in ("reported", "clean"):
            print(f"  {lab:8s} exact at t+1: " + ", ".join(f"{k} {a}/{n}" for k, (a, n) in r[lab].items()))

def selftest():
    T, N = 6, 4; exF = np.ones((T, N), bool); exG = np.ones((T, N), bool)
    D = {f"ex_F_from_{k}G": exF.copy() for k in ("perp", "h", "l")}; D.update({f"ex_G_from_{k}F": exG.copy() for k in ("perp", "h", "l")})
    D["ex_F_from_lG"][2, 0] = False                                         # puzzle 0: broken at row 2 = its control time t = 2
    prim, ctrl = {0: 1}, {0: 2, 1: 3}                                       # puzzle 0 overlaps (primary t 1, control t 2); puzzle 1 is clean
    r = readings(D, prim, ctrl)
    assert r["overlap"] == 1 and r["overlap_different_time"] == 1
    assert r["reported"]["F<-lG"] == (1, 2) and r["clean"]["F<-lG"] == (1, 1) and r["clean"]["G<-hF"] == (1, 1)
    print("selftest OK: 3/3")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true"); a = ap.parse_args()
    selftest() if a.selftest else main_run()
