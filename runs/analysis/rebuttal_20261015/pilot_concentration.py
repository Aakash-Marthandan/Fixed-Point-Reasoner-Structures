"""Exploratory, intact-only (no transplant outcome exists in the pilot): (1) per-task residual donor information and response scale; (2) a
concentration-aware precision bracket replacing the scale-free proxy: Var(beta) = sum_t (sum_{i in t} rD_i e_i)^2 / (sum rD^2)^2 with a noise
e_i of variance 2 v_u (v_u = the unit's mean of (cR^2 + cD^2) / 2, the pair's response scale: a perturbed response as noisy as a fresh one; the recipient's own variance alone is zero exactly where the donor varies), independent within a task (lower) or perfectly
correlated within a task (upper); projected to the confirmation by the task-count ratio; (3) the margin's logit scale per task."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path("/Users/aakash/Projects/HRRN"); sys.path.insert(0, str(ROOT / "tools")); import rebuttal_p29 as T
B = ROOT / "runs/analysis/rebuttal_20261015"; out = {}; pj = json.loads((B / "pilot_report.json").read_text())["eligibility"]["projection"]
for e in T.E_CAND:
    d = T.load(f"pilot_E{e}"); don = T.donor_map(d); task = np.array([str(x) for x in d["state_task"]])
    for meas in ("correct", "margin"):
        Y, Bm = T.unit_table(d, meas, e, 1); u = T.centered_units(Y, don, 0, 0, 0); cD, cR = u["cD"], u["cR"]; n = len(cD); tk = task[u["state"]]
        Xr = np.column_stack([cR, np.ones(n)]); rD = cD - Xr @ np.linalg.lstsq(Xr, cD, rcond=None)[0]
        v = np.repeat([np.mean((cR[i:i + 4] ** 2 + cD[i:i + 4] ** 2) / 2) for i in range(0, n, 4)], 4); S2 = rD @ rD
        lo = np.sqrt(sum((rD[tk == t] ** 2 * 2 * v[tk == t]).sum() for t in np.unique(tk))) / S2
        hi = np.sqrt(sum(((np.abs(rD[tk == t]) * np.sqrt(2 * v[tk == t])).sum()) ** 2 for t in np.unique(tk))) / S2
        ntask = len(np.unique(tk)); f = np.sqrt(ntask / (pj[str(e)]["projected"] * ntask / max(len(np.unique(d["state_qi"])), 1)))
        sh = {t: float((rD[tk == t] ** 2).sum() / S2) for t in np.unique(tk)}; top = sorted(sh.items(), key=lambda x: -x[1])[:3]
        scale = {t: float(np.sqrt(np.mean(cR[tk == t] ** 2))) for t in np.unique(tk)}
        r = dict(se_lo=float(lo), se_hi=float(hi), proj_se_lo=float(lo * f), proj_se_hi=float(hi * f), mde_lo=float(2.8 * lo * f), mde_hi=float(2.8 * hi * f),
                 top=[(t, round(s_, 3), round(scale[t], 3)) for t, s_ in top], median_task_scale=float(np.median(list(scale.values()))))
        out[f"{meas}|E{e}"] = r
        print(f"{meas:7s} |E|={e:2d}: concentration-aware SE (pilot) [{lo:.3f}, {hi:.3f}] -> projected [{lo * f:.3f}, {hi * f:.3f}], MDE [{2.8 * lo * f:.3f}, {2.8 * hi * f:.3f}] | top tasks (share, response RMS) {r['top']} vs median task RMS {r['median_task_scale']:.3f}")
    Ym, Bmm = T.unit_table(d, "margin", e, 1); nm = {t: np.mean([v_ for (s_, c), v_ in Bmm.items() if task[s_] == t]) for t in np.unique(task)}
    print(f"   |E|={e}: no-edit E-margin by task: median {np.median(list(nm.values())):.1f}, range [{min(nm.values()):.1f}, {max(nm.values()):.1f}]")
(B / "pilot_concentration.json").write_text(json.dumps(out, indent=1) + "\n")


# ---- option (not adopted; for the PI's decision): the margin standardized per query (both histories, all conditions divided by the pair's
# pooled RMS of centered intact responses), which weights tasks equally; same calibrated bracket (lower end) and concentration
print("OPTION per-query standardized margin (intact-only):")
for e in T.E_CAND:
    d = T.load(f"pilot_E{e}"); don = T.donor_map(d); task = np.array([str(x) for x in d["state_task"]]); Y, Bm = T.unit_table(d, "margin", e, 1)
    u = T.centered_units(Y, don, 0, 0, 0); qi = d["state_qi"][u["state"]]; sc = {q: np.sqrt(np.mean(np.r_[u["cR"][qi == q], u["cD"][qi == q]] ** 2)) for q in np.unique(qi)}
    k_ = np.array([sc[q] for q in qi]); cD, cR = u["cD"] / k_, u["cR"] / k_; n = len(cD); tk = task[u["state"]]
    Xr = np.column_stack([cR, np.ones(n)]); rD = cD - Xr @ np.linalg.lstsq(Xr, cD, rcond=None)[0]; S2 = rD @ rD
    v = np.repeat([np.mean((cR[i:i + 4] ** 2 + cD[i:i + 4] ** 2) / 2) for i in range(0, n, 4)], 4)
    lo = np.sqrt(sum((rD[tk == t] ** 2 * 2 * v[tk == t]).sum() for t in np.unique(tk))) / S2; sh = np.array(sorted([(rD[tk == t] ** 2).sum() / S2 for t in np.unique(tk)], reverse=True))
    f = np.sqrt(len(np.unique(qi)) / pj[str(e)]["projected"])
    print(f"   |E|={e:2d}: top task share {sh[0]:.3f}, effective tasks {1 / (sh ** 2).sum():.1f} of {len(sh)}; calibrated SE {lo:.3f} -> projected {lo * f:.3f}, MDE {2.8 * lo * f:.3f}")
