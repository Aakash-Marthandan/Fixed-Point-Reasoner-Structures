"""Read the orbit-invariance instrument: per grid, the strat-512 cold exactness at D16 under g0 (identity, CPU), g1/g2 (position elements),
g3 (a digit permutation), aligned on idx; the flip rate of each element against g0 and against the TPU vb screen (the route floor)."""
import json, numpy as np, os, sys
R = "/Users/aakash/Projects/HRRN/runs/"
def cold(p):
    q = os.path.join(p, "records_all.npz")
    if not os.path.exists(q): return None
    z = np.load(q, allow_pickle=True); o = np.argsort(z["idx"]); return z["idx"][o], z["cold_exact"][o].astype(bool)
out = {}
for arm, grids in (("C0", "g0 g1 g2 g3"), ("C3", "g0 g1 g2"), ("C5", "g0 g1 g2")):
    res = {g: cold(R + f"orbit_{arm}_{g}") for g in grids.split()}; tpu = cold(R + f"sxscreen_pchamp{arm}_vb")
    if res.get("g0") is None: print(arm, "g0 absent"); continue
    idx0, c0 = res["g0"]; line = [f"{arm}: g0 (identity, CPU) {100*c0.mean():.2f}"]
    if tpu is not None and np.array_equal(tpu[0], idx0):
        fl = float((tpu[1] != c0).mean()); line.append(f"| TPU vb screen {100*tpu[1].mean():.2f}, route flip rate {100*fl:.2f} % ({int((tpu[1] != c0).sum())}/512)"); out[f"{arm} route"] = fl
    for g in grids.split()[1:]:
        if res.get(g) is None: line.append(f"| {g} absent"); continue
        idx, c = res[g]; assert np.array_equal(idx, idx0)
        fl = float((c != c0).mean()); only_g = int((c & ~c0).sum()); only_0 = int((~c & c0).sum())
        line.append(f"| {g} {100*c.mean():.2f}, flips vs g0 {100*fl:.2f} % (+{only_g}/-{only_0})"); out[f"{arm} {g}"] = dict(acc=float(c.mean()), flip=fl, only_g=only_g, only_g0=only_0)
    print(" ".join(line))
json.dump(out, open(R + "analysis/champ_orbit_invariance_20260910.json", "w"), indent=1)

# the flipped puzzles' ratings (the round-off floor lives on the churning, hard trajectories) and their first-exact on the identity run
z = np.load("/Users/aakash/Projects/HRRN/data/sudoku_extreme/sudoku_extreme_seed0.npz", allow_pickle=False); RAT = z["test_rating"]
for arm, grids in (("C0", "g1 g2 g3"), ("C3", "g1 g2"), ("C5", "g1 g2")):
    r0 = cold(R + f"orbit_{arm}_g0")
    if r0 is None: continue
    idx0, c0 = r0; q0 = np.load(R + f"orbit_{arm}_g0/records_all.npz", allow_pickle=True); o0 = np.argsort(q0["idx"]); fe0 = q0["first_exact"][o0]
    parts = []
    for g in grids.split():
        r = cold(R + f"orbit_{arm}_{g}")
        if r is None: continue
        fl = r[1] != c0; parts.append(f"{g}: flipped {int(fl.sum())} at median rating {np.median(RAT[idx0[fl]]):.0f} (all 512: {np.median(RAT[idx0]):.0f}); identity first-exact median on the flipped {np.median(fe0[fl]) + 1:.0f} vs {np.median(fe0[c0]) + 1:.0f} on the solved")
    print(f"{arm} flips by rating | " + " | ".join(parts))
