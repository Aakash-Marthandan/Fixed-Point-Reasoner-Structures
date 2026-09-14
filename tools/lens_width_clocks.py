#!/usr/bin/env python3
# Ledger: WIDTH x TRAINING-TIME DYNAMICS — analysis-time DESCRIPTIVE lens (no rules; exploratory). The PI (2026-09-14): "understand the
# dynamics in terms of parameter scale and training step time to formalize their mechanisms"; "since w384 memorizes but w192 doesn't,
# there must be an optimum parameter scale". Reads only existing artifacts: every run's metrics.jsonl and its banked 2k checkpoints.
# The width ladder (same loop, batch 768, AdamW lr 1e-4 constant after warmup, wd 1.0, EMA .999, standard init, ONE global lr):
#   champion recipe (FPA k1 + RI sigma 1): w192 C5 C7 C8 (to 50k) · w384 C0 C2 (30k), C1 (50k), A5 (seed 0, 50k, 64-puzzle monitor)
#   plain recipe: w384 A3 (s0), A7 (s1) · w512 A8 (s1), all to 50k on the 64-puzzle monitor
# Per run: (1) CLOCKS on the held-out monitor (EMA), in the monitor's own noise units (1 SE = sqrt(p(1-p)/n) at the peak): the peak
# (earliest max), the plateau (within 1 SE), the onset of decline (the first grid from which the curve stays >= 2 SE under its running
# max through the end, over >= 2 grids); (2) FIT: training-batch exact and CE by 2k block; (3) WEIGHTS from the raw state of every grid:
# RMS per group (channel = fc + channel SwiGLU, width^2 params; token = the 81-cell token-mixing SwiGLU, width-independent; io = role
# embedding, readout, halting head), the relative Frobenius update per 2k steps per group, and the spectral relative update of the
# block-0 channel matrix (||dW||_2 / ||W||_2, the width-scaling quantity for Adam hidden weights).
#   .venv/bin/python tools/lens_width_clocks.py [--out runs/analysis/width_clocks_20260914]
#   .venv/bin/python tools/lens_width_clocks.py --selftest
from __future__ import annotations
import argparse, json, math, pickle, sys
from pathlib import Path
import numpy as np

RUNS = [  # (name, width, recipe, dir, monitor n)
    ("C5", 192, "champ", "runs/pretrainchamp_C5", 512), ("C7", 192, "champ", "runs/_paperfinal_pull/x/runs/pretrainchamp_C7", 512),
    ("C8", 192, "champ", "runs/_c8x_pull/x/runs/pretrainchamp_C8", 512),
    ("C0", 384, "champ", "runs/pretrainchamp_C0", 512), ("C1", 384, "champ", "runs/pretrainchamp_C1", 512), ("C2", 384, "champ", "runs/pretrainchamp_C2", 512),
    ("A5", 384, "champ", "runs/pretrainfinalA_A5", 64),
    ("A3", 384, "plain", "runs/pretrainfinalA_A3", 64), ("A7", 384, "plain", "runs/pretrainfinalA_A7", 64), ("A8", 512, "plain", "runs/pretrainfinalA_A8", 64),
]

def monitor_curve(pdir):
    mon = {}
    for l in (Path(pdir) / "metrics.jsonl").read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            if "monitor" in r: mon[int(r["monitor"]["step"])] = float(r["monitor"]["val_t16_ema"])
    return sorted(mon.items())

def clocks(c, n):
    """c = [(step, acc)] -> peak, plateau (within 1 SE of the max), onset (sustained >= 2 SE under the running max to the end, >= 2 grids)."""
    best = max(v for _, v in c); peak = min(s for s, v in c if v == best)
    se = math.sqrt(max(best * (1 - best), 1e-9) / n)
    plateau = [s for s, v in c if v >= best - se]
    run, mx = [], -1.0
    for _, v in c: mx = max(mx, v); run.append(mx)
    onset = None
    for i in range(len(c) - 1):
        if all(c[j][1] <= run[j] - 2 * se for j in range(i, len(c))): onset = c[i][0]; break
    return dict(peak=peak, best=best, se=se, plateau=(min(plateau), max(plateau)), onset=onset, end=c[-1][0], end_drop=best - c[-1][1])

def fit_blocks(pdir):
    rows = [json.loads(l) for l in (Path(pdir) / "metrics.jsonl").read_text().splitlines() if l.strip()]
    rows = {r["step"]: r for r in rows if "loss" in r and "train_exact" in r}
    out = {}
    for b in range(2000, max(rows) + 1, 2000):
        rs = [rows[s] for s in rows if b - 2000 < s <= b]
        if rs: out[b] = (float(np.mean([r["train_exact"] for r in rs])), float(np.mean([r["ce_in"] for r in rs])))
    return out

GROUPS = {"channel": ("fc", "mlp/"), "token": ("mlp_t/",), "io": ("role_emb", "lm_head", "q_head")}
def leaves(t, p=""):
    if isinstance(t, dict):
        for k, v in t.items(): yield from leaves(v, f"{p}/{k}")
    elif isinstance(t, (list, tuple)):
        for i, v in enumerate(t): yield from leaves(v, f"{p}/{i}")
    else: yield p, np.asarray(t, dtype=np.float64)

def group_of(path):
    if "/mlp_t/" in path: return "token"
    if path.endswith("/fc") or "/mlp/" in path: return "channel"
    if any(k in path for k in GROUPS["io"]): return "io"
    return None

def weights(ck):
    d = pickle.load(open(ck, "rb")); g = {"channel": [], "token": [], "io": []}; W0 = None
    for p, a in leaves(d["state"]["model"]):
        k = group_of(p)
        if k: g[k].append(a.ravel())
        if p.endswith("blocks/0/mlp/gate_up"): W0 = a
    return {k: np.concatenate(v) for k, v in g.items()}, W0

def rel(a, b): return float(np.linalg.norm(b - a) / max(np.linalg.norm(a), 1e-12))

def lens(out):
    L, J = [], {}
    say = lambda s="": (L.append(s), print(s))
    say("WIDTH x TRAINING-TIME DYNAMICS (descriptive; exploratory). Monitor EMA; SE units at each run's peak; weights = the raw trained state.")
    say("(1) CLOCKS + (2) FIT")
    say("  run  width recipe  mon   peak(best)        plateau(1SE)   onset(2SE,sustained)  end-drop  | train-batch exact @10k/20k/30k/40k/50k")
    for name, w, rec, d, n in RUNS:
        c = monitor_curve(d); k = clocks(c, n); f = fit_blocks(d)
        tx = " / ".join(f"{f[s][0]:.2f}" if s in f else "  - " for s in (10000, 20000, 30000, 40000, 50000))
        say(f"  {name:<3}  {w:>4}  {rec:<6} {n:>3}   {k['peak']//1000:>2}k ({100*k['best']:.1f})     {k['plateau'][0]//1000:>2}-{k['plateau'][1]//1000:<2}k        "
            f"{('none by ' + str(k['end']//1000) + 'k') if k['onset'] is None else str(k['onset']//1000) + 'k':<18}  {100*k['end_drop']:+5.1f}pp  | {tx}")
        J[name] = dict(width=w, recipe=rec, monitor_n=n, clocks=k, fit={str(s): v for s, v in f.items()})
    say("(3) WEIGHTS: RMS per group at 10k/30k/50k · relative update per 2k steps (median over 10-30k | 30-50k) · spectral rel. update of the block-0 channel matrix")
    for name, w, rec, d, n in RUNS:
        cks = sorted(Path(d).glob("ckpt_0*.pkl")); steps = [int(p.stem.split("_")[1]) for p in cks]
        prev = None; rows = {}
        for s, ck in zip(steps, cks):
            g, W0 = weights(ck); r = {k: float(np.sqrt(np.mean(v ** 2))) for k, v in g.items()}
            if prev is not None and s - prev[0] == 2000:
                pg, pW = prev[1], prev[2]
                r.update({f"du_{k}": rel(pg[k], g[k]) for k in g})
                r["du_spec"] = float(np.linalg.norm(W0 - pW, 2) / np.linalg.norm(pW, 2))
            rows[s] = r; prev = (s, g, W0)
        def med(key, lo, hi):
            v = [rows[s][key] for s in rows if lo < s <= hi and key in rows[s]]
            return float(np.median(v)) if v else float("nan")
        rms = " ".join(f"{g}:{'/'.join(f'{rows[s][g]:.4f}' if s in rows else '  -   ' for s in (10000, 30000, 50000))}" for g in ("channel", "token", "io"))
        du = " ".join(f"{g}:{med('du_'+g, 10000, 30000):.4f}|{med('du_'+g, 30000, 50000):.4f}" for g in ("channel", "token", "io"))
        say(f"  {name:<3} w{w:<3} {rms}   du {du}   spec {med('du_spec', 10000, 30000):.4f}|{med('du_spec', 30000, 50000):.4f}")
        J[name]["weights"] = {str(s): r for s, r in rows.items()}
    Path(out + ".txt").parent.mkdir(parents=True, exist_ok=True)
    Path(out + ".txt").write_text("\n".join(L) + "\n"); Path(out + ".json").write_text(json.dumps(J, indent=1))

def selftest():
    c = [(2000, .50), (4000, .90), (6000, .96), (8000, .958), (10000, .95), (12000, .90), (14000, .89), (16000, .88)]
    k = clocks(c, 512)
    assert k["peak"] == 6000 and abs(k["se"] - math.sqrt(.96 * .04 / 512)) < 1e-12
    assert k["plateau"] == (6000, 8000), k           # .95 is 1.1 SE under .96 (SE .0087): outside
    assert k["onset"] == 12000, k                    # sustained >= 2 SE (1.7 pp) under the running max from 12k to the end
    assert clocks([(2000, .9), (4000, .95), (6000, .90), (8000, .95)], 512)["onset"] is None   # a dip that recovers is not an onset
    a = np.ones(4); assert abs(rel(a, a * 1.1) - 0.1) < 1e-12
    assert group_of("/model/dec/blocks/0/mlp_t/down") == "token" and group_of("/model/dec/blocks/1/fc") == "channel" and group_of("/model/dec/lm_head") == "io"
    tree = {"dec": {"blocks": [{"fc": np.ones((2, 2)), "mlp_t": {"down": np.ones(3)}}, {"fc": np.zeros((2, 2))}], "lm_head": np.ones(2)}}
    assert [p for p, _ in leaves(tree)] == ["/dec/blocks/0/fc", "/dec/blocks/0/mlp_t/down", "/dec/blocks/1/fc", "/dec/lm_head"]   # lists of blocks walked
    print("selftest OK: 7/7")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="runs/analysis/width_clocks_20260914"); ap.add_argument("--selftest", action="store_true"); a = ap.parse_args()
    selftest() if a.selftest else lens(a.out)
