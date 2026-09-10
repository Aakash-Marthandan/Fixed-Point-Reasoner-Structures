"""G3 (Plan_2026-09-10_DEC-ARC_Build §3): the C1/C2 dynamics columns that the field's ARC-1 TRM traces admit
(runs/field_ckpts/harness/arc_out/cold_all_fp32/rows.jsonl carries exact_by_step, token_acc_by_step, q_halt_by_step per
input; no per-cell confidences, so commitment is not readable here). Descriptive; no rules."""
import json, numpy as np
from pathlib import Path
R = Path(__file__).resolve().parents[1] / "runs" / "field_ckpts" / "harness" / "arc_out" / "cold_all_fp32" / "rows.jsonl"
rows = [json.loads(l) for l in open(R)]
ex = np.array([r["exact"] for r in rows]); acc = np.array([r["token_acc_by_step"] for r in rows], float)
exs = np.array([r["exact_by_step"] for r in rows], bool); q = np.array([r["q_halt_by_step"] for r in rows], float)
n, T = acc.shape
out = []
out.append(f"THE FIELD'S ARC-1 TRM, per-step dynamics on the 419 canonical inputs (fp32, D16): n {n}, exact {ex.mean()*100:.2f} %")
for lab, m in (("solved", ex), ("unsolved", ~ex)):
    a = acc[m]; steps = [0, 1, 3, 7, 15]
    traj = " | ".join(f"s{s+1} {a[:, s].mean()*100:.1f}" for s in steps)
    mono = np.mean([np.all(np.diff(x) >= -1e-9) for x in a]) * 100
    fall = np.mean([x[-1] < x[0] - 1e-9 for x in a]) * 100
    out.append(f"  [{lab} n {m.sum()}] cells right by step: {traj} | monotone trajectories {mono:.1f} % | cells right FALL from step 1 to 16 on {fall:.1f} %")
    d = a[:, -1] - a[:, 0]
    out.append(f"      delta cells-right (16 - 1): mean {d.mean()*100:+.2f} pp, p10 {np.percentile(d,10)*100:+.2f}, p90 {np.percentile(d,90)*100:+.2f}")
fe = np.array([r["first_exact"] for r in rows if r["exact"]] , dtype=object)
fe_i = np.array([int(x) for x in fe if x is not None])
out.append(f"  first-exact of the solved (0-based step): step 0 {np.mean(fe_i==0)*100:.1f} %, step 1 {np.mean(fe_i==1)*100:.1f} %, later {np.mean(fe_i>1)*100:.1f} % (median {np.median(fe_i):.0f})")
# the frozen / churning proxy the rows admit: how many distinct exactness states along the rollout and the halting trajectory
flips = np.array([np.sum(np.diff(x.astype(int)) != 0) for x in exs])
out.append(f"  exactness flips along the rollout: solved mean {flips[ex].mean():.2f}, unsolved {flips[~ex].mean():.2f}; inputs with >= 1 flip: {np.mean(flips>0)*100:.1f} %")
qs = q[:, [0, 1, 3, 7, 15]]
out.append("  halting logit q by step (s1 s2 s4 s8 s16): solved " + " ".join(f"{v:+.2f}" for v in qs[ex].mean(0)) + " | unsolved " + " ".join(f"{v:+.2f}" for v in qs[~ex].mean(0)))
out.append(f"  q >= 0 at step 1: solved {np.mean(q[ex,0]>=0)*100:.1f} % / unsolved {np.mean(q[~ex,0]>=0)*100:.1f} %; at step 16: {np.mean(q[ex,-1]>=0)*100:.1f} / {np.mean(q[~ex,-1]>=0)*100:.1f}")
# the size decision: the token accuracy is on the padded 30x30; an input whose token_exact differs from exact is a size/crop case
te = np.array([r["token_exact"] for r in rows]); out.append(f"  token-exact but not exact (crop/size cases): {np.sum(te & ~ex)}; exact but not token-exact: {np.sum(ex & ~te)}")
txt = "\n".join(out); print(txt)
open(Path(__file__).resolve().parents[1] / "runs" / "analysis" / "arc_field_dyn_20260910.txt", "w").write(txt + "\n")
