"""Freethink grounding 3 ($0): curvature concentration from the checkpoints' Adam second moment (the Fisher-from-Adam-v lens):
participation ratio PR = (sum v)^2 / sum v^2 over model params (1/PR = effective number of curvature-carrying directions
fraction), per-block share of curvature mass, and the fraction of parameters carrying 90% of the mass — regime comparison."""
import pickle, numpy as np
from pathlib import Path
def leaves(t, path=""):
    if isinstance(t, dict):
        for k, v in t.items(): yield from leaves(v, path + "/" + k)
    elif isinstance(t, (list, tuple)):
        for i, v in enumerate(t): yield from leaves(v, path + f"[{i}]")
    else: yield path, np.asarray(t)
def find_nu(opt):
    # optax chain: find the ScaleByAdamState-like node holding 'nu' (a tree of arrays matching the model)
    found = []
    def walk(x):
        if hasattr(x, "_fields") and "nu" in x._fields: found.append(x.nu)
        elif isinstance(x, (list, tuple)):
            for y in x: walk(y)
        elif isinstance(x, dict):
            for y in x.values(): walk(y)
    walk(opt); return found
L = []
def say(s=""): L.append(s); print(s)
say("== GROUND 3: curvature concentration (Adam second moment ~ diagonal Fisher) per checkpoint ==")
say("  ckpt | regime | n params | PR/n (effective fraction of curvature directions) | params carrying 90% of mass | top-3 blocks by mass share")
for name, path, regime in (("A0 @30k", "runs/pretrainsportC1_A0/ckpt_latest.pkl", "ours, no norm, wd 1e-4"), ("B0a @20k (vsel)", "runs/pretrainsportC1_B0a/ckpt_020000.pkl", "ours, z-norm, wd 1e-4"),
                           ("B0 @80k (memorized)", "runs/pretrainsportC1_B0/ckpt_latest.pkl", "ours, z-norm, wd 1e-4, floor"), ("R0 @50k", "runs/pretrainsportC1_R0/ckpt_latest.pkl", "field regime wd 1.0 lr 1e-4"),
                           ("X0 @50k", "runs/pretrainsportC1_X0/ckpt_latest.pkl", "field cell, field regime"), ("X0n @50k (memorized)", "runs/pretrainsportC1_X0n/ckpt_latest.pkl", "field cell, no ACT")):
    c = pickle.load(open(path, "rb")); nus = find_nu(c["opt_state"])
    if not nus: say(f"  {name}: no Adam nu found (opt keys: {type(c['opt_state'])})"); continue
    nu = nus[0]; nu = nu["model"] if isinstance(nu, dict) and "model" in nu else nu
    vals = []; blocks = {}
    for p, a in leaves(nu):
        a = a.astype(np.float64).ravel(); vals.append(a)
        b = p.split("/")[1] if p.count("/") >= 1 else p; b = b.split("[")[0]
        blocks[b] = blocks.get(b, 0.0) + float(a.sum())
    v = np.concatenate(vals); n = v.size; tot = v.sum(); pr = tot ** 2 / max((v ** 2).sum(), 1e-300)
    vs = np.sort(v)[::-1]; cum = np.cumsum(vs) / tot; k90 = int(np.searchsorted(cum, 0.9)) + 1
    top = sorted(blocks.items(), key=lambda kv: -kv[1])[:3]
    say(f"  {name:22s} | {regime:28s} | {n:9d} | {pr/n:.2e} | {100*k90/n:6.3f}% | " + ", ".join(f"{b} {100*s/tot:.1f}%" for b, s in top))
Path("runs/analysis/freethink_ground3_fisher_20260903.txt").write_text("\n".join(L) + "\n")
