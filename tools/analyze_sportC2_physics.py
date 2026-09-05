#!/usr/bin/env python3
"""sportC2 (the pre-parity graft night at d128) — physics pass, written AT ANALYSIS TIME
(2026-09-05; descriptive only). The registered verdict authority is tools/analyze_sportC2.py
(byte-untouched since its creation commit eea3c32 / the registration 15735c3, selftest 21/21);
NOTHING here is a decision rule. Derivative of analyze_sportC1_physics.py (same readers).

Reads the extracted sportC2 corpus in runs/ (+ the banked sportC1 / sportC0 / rung-2b references
already on disk) and answers the mechanism questions the registration posed but its rules do not read:
  A. ADMISSION + grid census + val-selection provenance (every vsel eval's ckpt path; the one-ckpt
     gate read again here) + an OFFLINE re-selection over both stages (the PM-1 check).
  B. TRAJECTORIES — train rows + 2k monitors, two-stage arms stitched (stage B offset +50k), R4's
     continuation (step counter from 0 = effective +50k on sportC1 R0), R1's SOT-rg rows.
  C. STABILITY CENSUS + THE wd TWIN BRIDGE — no deaths expected; retfm / census / eta / lambda_J per
     arm; the seed-paired B0 (wd 1e-4, sportC1) vs W0 (wd 1.0) trajectories at matched steps.
  D. EVAL TABLE — vsel / final / alt-weights cold | b1 | verified@k | t1r@k | majority@k | retfm |
     census rows | calibration-at-stalls rows (+ R3's hard-feedback rows) | X1/X2 D16 and D64 rows.
  E. PAIRED McNemar on identical puzzle sets — the wd bridge (W0 vs B0 on BOTH B0 grids: the memorized
     sportC1 scan and the rider's correct A:20k scan), the three grafts vs W0, R4 vs R0, X1/X2 vs X0,
     the riders (B0/B1 correct-grid scans, B0 EMA full), the canvas EqR-statistic riders, unions,
     vsel-vs-final, headline-vs-alt weights, D16-vs-D64, per-octile tables.
  F. FUNNEL (rho, r) per octile on every scan.   G. SELECTOR — t1r vs the free verifier.
  H. EXTRA PREDICTIONS (plan §5 / freethink X-7 items the frozen analyzer does not score).
  J. CURVATURE CONCENTRATION from Adam-v (the freethink ground-3 lens as a standing output).

  .venv/bin/python tools/analyze_sportC2_physics.py [--no-grids]
      -> runs/analysis/sportC2_physics_20260905.txt
"""
from __future__ import annotations
import argparse, json, math, os, pickle, re, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(os.environ.get("QHRRN_RUNS", ROOT / "runs"))
OUT = RUNS / "analysis" / "sportC2_physics_20260905.txt"
TAG = "sportC2"
NATIVE = ["W0", "R1", "R2", "R3", "R4"]; FIELD = ["X1", "X2"]; ARMS = NATIVE + FIELD
S_A = 50000
META = {
    "W0": dict(two=True, cell="rg", key="val_t64", head_ema=False, head_t=64, pin=3004658, offset=0, desc="B0 + wd 1.0 (the base)"),
    "R1": dict(two=True, cell="rg", key="val_t64", head_ema=False, head_t=64, pin=3004658, offset=0, desc="W0 + persistent carry (SOT-rg, 4 segments)"),
    "R2": dict(two=True, cell="rg", key="val_t64", head_ema=False, head_t=64, pin=3004658, offset=0, desc="W0 + inner cycles K=3 (trained)"),
    "R3": dict(two=True, cell="rg", key="val_t64", head_ema=False, head_t=64, pin=3004658, offset=0, desc="W0 + hard rows p=.5 (STE)"),
    "R4": dict(two=False, cell="rg", key="val_t64_ema", head_ema=True, head_t=64, pin=3004658, offset=50000, desc="sportC1 R0 +50k, field regime (EMA)"),
    "X1": dict(two=False, cell="trm", key="val_t16_ema", head_ema=True, head_t=16, pin=5037058, offset=0, desc="X0 - digit aug (EMA)"),
    "X2": dict(two=False, cell="trm", key="val_t16_ema", head_ema=True, head_t=16, pin=5839874, offset=0, desc="X0 + group9 token mixer (EMA)"),
}
L = []
def say(s=""): L.append(str(s)); print(s)
def f(x, w=6, p=2):
    return " " * (w - 1) + "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:{w}.{p}f}"
def pp(x): return "   -  " if x is None else f"{100*x:6.2f}"
def jload(p):
    p = Path(p); return json.loads(p.read_text()) if p.exists() else None
def pdir(a): return RUNS / f"pretrain{TAG}_{a}"
def pdirA(a): return RUNS / f"pretrain{TAG}_{a}a"
def evdir(a): return RUNS / f"sxeval_p{TAG}{a}"
def full_dir(a, which, t=None):
    t = META[a]["head_t"] if t is None else t
    return evdir(a) / f"full_{which}_t{t}"
def scan_dir(a): return RUNS / f"sxscan_p{TAG}{a}"
def screen_dir(a, tag): return RUNS / f"sxscreen_p{TAG}{a}_{tag}"
def census_json(a, which): return jload(RUNS / f"sxcensus_p{TAG}{a}_{which}" / "census.json")
def calib_json(a, which="vsel"): return jload(RUNS / f"sxcalib_p{TAG}{a}_{which}" / "calib.json")

# ---------- A. admission + grid census ----------
def walk_finite(t):
    st = [t]
    while st:
        x = st.pop()
        if isinstance(x, dict): st.extend(x.values())
        elif isinstance(x, (list, tuple)): st.extend(x)
        elif hasattr(x, "dtype") and np.issubdtype(np.asarray(x).dtype, np.number):
            if not np.isfinite(np.asarray(x)).all(): return False
    return True
def count_params(tree):
    n = 0; st = [tree]
    while st:
        x = st.pop()
        if isinstance(x, dict): st.extend(x.values())
        elif isinstance(x, (list, tuple)): st.extend(x)
        else: n += int(np.prod(np.shape(x)))
    return n
def grid_census(d: Path, walk=True):
    rows = []
    for p in sorted(d.glob("ckpt_0*.pkl")):
        try:
            c = pickle.load(open(p, "rb")); rows.append((int(c["step"]), walk_finite(c["state"]["model"]) if walk else None))
        except Exception: rows.append((-1, False))
    return rows
CFG_KEYS = ("d", "T", "scales", "canvas", "pool_arity", "mixer_kind", "sudoku_layout", "attn_max_hw", "beta_flux", "beta_flux_nl", "fpa_k", "fpa_eps",
            "ni_sigma", "z_norm", "cell_kind", "loss_kind", "inner_k", "hard_p", "trm_token_mixer", "trm_gm_dim", "trm_hidden", "trm_layers", "trm_h_cycles", "trm_l_cycles", "trm_lambda", "trm_beta")
ARGV_KEYS = ("T", "seed", "ri_p", "sudoku_aug", "sudoku_digit_aug", "steps", "lr", "lr_end", "warmup", "wd", "batch", "beta2", "ema", "init_from", "width_scale", "fpa_k",
             "beta_flux_nl", "z_norm", "cell", "sot", "sot_segments", "act", "loss", "remat", "inner_k", "hard_p", "trm_token_mixer")
def admission(arm, walk=True):
    d = pdir(arm); out = {"dir": d.exists()}
    if not d.exists(): return out
    ck = d / "ckpt_latest.pkl"
    if ck.exists():
        c = pickle.load(open(ck, "rb")); cfg = c.get("config", {})
        out["step"] = int(c["step"]); out["cfg"] = {k: cfg.get(k) for k in CFG_KEYS}
        out["n_bulk"] = count_params(c["state"]["model"]); out["has_ema"] = c.get("state_ema") is not None
        out["final_finite"] = walk_finite(c["state"]["model"]) if walk else None
    cj = d / "config.json"
    if cj.exists():
        a = json.loads(cj.read_text()).get("argv", {}); out["argv"] = {k: a.get(k) for k in ARGV_KEYS}
    for nm in ("STOPPED.txt", "NAN_ABORT.txt", "resumes.txt", "val_best.txt", "RETRY_REMAT.txt"):
        p = d / nm; out[nm] = p.read_text().strip() if p.exists() else None
    da = pdirA(arm)
    if da.exists():
        for nm in ("NAN_ABORT.txt", "resumes.txt", "RETRY_REMAT.txt"):
            p = da / nm
            if p.exists(): out[nm] = (out.get(nm) or "") + " (stage A) " + p.read_text().strip()
    out["grids"] = grid_census(d, walk); out["gridsA"] = grid_census(da, walk) if da.exists() else None
    return out
def ckpt_step_from_path(path: str, arm: str):
    if not path: return ("?", None)
    name = path.split("/")[-1]; dd = path.split("/")[-2] if "/" in path else ""
    off = META[arm]["offset"]
    if name == "ckpt_latest.pkl": return ("final", None)
    m = re.search(r"ckpt_(\d+)\.pkl", name)
    if not m: return ("?", None)
    s = int(m.group(1))
    if META[arm]["two"] and dd.endswith("a"): return (f"A:{s}", s)
    if META[arm]["two"]: return (f"B:{s}", S_A + s)
    return (f"{s}", s + off)

# ---------- B. trajectories ----------
def load_metrics(path: Path):
    tr, mon = {}, {}
    if not path.exists(): return tr, mon
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line: continue
        try: r = json.loads(line)
        except Exception: continue
        if "monitor" in r: mon[int(r["monitor"]["step"])] = r["monitor"]
        elif "step" in r and "loss" in r: tr[int(r["step"])] = r
    return dict(sorted(tr.items())), dict(sorted(mon.items()))
def arm_traj(arm):
    d = pdir(arm); tr, mon = load_metrics(d / "metrics.jsonl"); note = ""
    if META[arm]["two"]:
        da = pdirA(arm)
        if (d / "STOPPED.txt").exists(): note = "died in stage A"
        else:
            trB = {k + S_A: dict(v, step=k + S_A) for k, v in tr.items()}; monB = {k + S_A: dict(v, step=k + S_A) for k, v in mon.items()}
            trA, monA = load_metrics(da / "metrics.jsonl") if da.exists() else ({}, {})
            note = "stage A present; stage B offset +50k" if trA else "stage A NOT shipped; stage B offset +50k"
            tr = dict(sorted({**trA, **trB}.items())); mon = dict(sorted({**monA, **monB}.items()))
    elif META[arm]["offset"]: note = f"continuation: steps as logged (0..50k); effective = +{META[arm]['offset']} on sportC1 R0"
    return tr, mon, note
def finite(v):
    try: return v is not None and math.isfinite(float(v))
    except Exception: return False
def near(dct, target, before=True, window=None):
    ks = [s for s in dct if finite(dct[s]) and (s <= target if before else s >= target)]
    if not ks: return None
    k = max(ks) if before else min(ks)
    if window is not None and abs(k - target) > window: return None
    return dct[k]
def traj_summary(arm, tr, mon):
    steps = list(tr.keys()); key = META[arm]["key"]
    first_bad = next((s for s in steps if not finite(tr[s].get("loss"))), None)
    last_good = max([s for s in steps if finite(tr[s].get("loss"))], default=None)
    A = {s: tr[s].get("A_total") for s in steps}; I = {s: tr[s].get("I_total") for s in steps}; CE = {s: tr[s].get("ce_in") for s in steps}
    A_peak = max([A[s] for s in steps if finite(A[s])], default=None)
    pace = float(np.median([tr[s]["steps_per_sec"] for s in steps if finite(tr[s].get("steps_per_sec")) and tr[s]["steps_per_sec"] > 0])) if steps else None
    ms = [s for s in mon if s <= (last_good or 10**9)]
    def mseries(k): return [(s, mon[s].get(k)) for s in ms if finite(mon[s].get(k))]
    val = mseries(key); val_raw = mseries(key.replace("_ema", "")) if key.endswith("_ema") else mseries(key + "_ema")
    vpk = max(val, key=lambda x: x[1]) if val else None
    rise = None
    if len(val) >= 2:
        s_end, v_end = val[-1]; prev = [v for s, v in val if s <= s_end - 10000]; rise = (v_end - prev[-1]) if prev else None
    ce_min = min([CE[s] for s in steps if finite(CE[s])], default=None)
    def tail(k): xs = [tr[s].get(k) for s in steps if finite(tr[s].get(k))]; return (xs[-1] if xs else None, max(xs) if xs else None)
    tx = tail("train_exact"); hf = tail("halt_frac"); ff = tail("fresh_frac"); mst = tail("mean_steps")
    ts = [tr[s].get("t") for s in steps if tr[s].get("t")]
    return dict(n_train_rows=len(steps), first_bad=first_bad, last_good=last_good, last_step=max(steps) if steps else None,
                loss_last=tr[last_good]["loss"] if last_good else None, ce_last=near(CE, last_good or 0), ce_min=ce_min,
                ce_10k=near(CE, 10000, window=2000), ce_20k=near(CE, 20000, window=2000), ce_30k=near(CE, 30000, window=2000), ce_50k=near(CE, 50000, window=2000),
                A_peak=A_peak, A_2k=near(A, 2000, False, window=2000), A_10k=near(A, 10000, window=2000), A_last=near(A, last_good or 0),
                I_10k=near(I, 10000, window=2000), I_last=near(I, last_good or 0), ruleH_last=near({s: tr[s].get("rule_H") for s in steps}, last_good or 0), pace=pace,
                val=val, val_other=val_raw, val_peak=vpk, val_end=val[-1] if val else None, val_rise10k=rise,
                eta=mseries("eta"), eta_z=mseries("eta_z"), lamj=mseries("lam_joint_max"), retfm=mseries("ret_final_t8"), rets=mseries("ret_sched_t8"),
                train_exact_last=tx[0], train_exact_max=tx[1], halt_last=hf[0], fresh_last=ff[0], mean_steps_last=mst[0], t_first=(ts[0] if ts else None), t_last=(ts[-1] if ts else None))

# ---------- D/E. evals ----------
def recs(p):
    p = Path(p); q = p / "records_all.npz"
    if not q.exists():
        parts = sorted(p.glob("records_s*.npz"))
        if not parts: return None
        arrs = [dict(np.load(x, allow_pickle=True)) for x in parts]; keys = [k for k in arrs[0] if all(k in a for a in arrs)]
        z = {k: np.concatenate([a[k] for a in arrs]) for k in keys}
    else: z = dict(np.load(q, allow_pickle=True))
    order = np.argsort(z["idx"], kind="stable")
    return {k: (v[order] if hasattr(v, "shape") and v.shape and v.shape[0] == len(order) else v) for k, v in z.items()}
def mcnemar(a, b):
    oa = int(np.sum(a & ~b)); ob = int(np.sum(~a & b)); n = oa + ob
    if n == 0: return oa, ob, 1.0
    from scipy import stats
    return oa, ob, float(min(1.0, 2 * stats.binom.cdf(min(oa, ob), n, 0.5)))
def vote_bits(z, k):
    fh = z["mi_first_hit"]; return z["cold_exact"].astype(bool) | ((fh >= 0) & (fh < k))
def b1_bits(z): return z["mi_exact_k"][:, 0].astype(bool) if "mi_exact_k" in z and z["mi_exact_k"].ndim == 2 else None
def t1r_bits(z, k=128):
    if "mi_exact_k" not in z or "mi_resid_k" not in z: return None
    ex = z["mi_exact_k"].astype(bool); re_ = z["mi_resid_k"].astype(np.float32); k = min(k, ex.shape[1])
    best = np.argmin(re_[:, :k], axis=1); return ex[np.arange(len(ex)), best]
def fit_rho_r(fh, k_fit=64):
    fh = np.asarray(fh); hit = (fh >= 0) & (fh < k_fit); t = fh[hit]; n_c = int(np.sum(~hit)); best = (-np.inf, None, None)
    for rho in np.linspace(0.02, 1.0, 50):
        for r in np.geomspace(1e-3, 0.9, 60):
            ll = np.sum(np.log(rho) + t * np.log1p(-r) + np.log(r)) if len(t) else 0.0
            ll += n_c * np.log(max(1 - rho + rho * (1 - r) ** k_fit, 1e-300))
            if ll > best[0]: best = (ll, rho, r)
    return best[1], best[2]
def octiles(rat):
    qs = np.quantile(rat, np.linspace(0, 1, 9)); qs[-1] += 1; return qs
def funnel_table(z, label):
    rat = z["rating"]; fh = z["mi_first_hit"]; cold = z["cold_exact"].astype(bool); qs = octiles(rat); b1 = b1_bits(z)
    say(f"  {label}: octile [rating) | n | cold | b1 | draw-hit@128 | rho | r | pred@128")
    for b in range(8):
        m = (rat >= qs[b]) & (rat < qs[b + 1])
        if m.sum() == 0: continue
        rho, r = fit_rho_r(fh[m]); pred = rho * (1 - (1 - r) ** 128) if rho is not None else None
        act = float(np.mean((fh[m] >= 0) & (fh[m] < 128)))
        say(f"    {b} [{qs[b]:.0f},{qs[b+1]:.0f}) n={m.sum():5d} cold {pp(cold[m].mean())} b1 {pp(b1[m].mean() if b1 is not None else None)} hit {pp(act)} rho {f(rho,5,2)} r {f(r,6,3)} pred {pp(pred)}")

# ---------- J. curvature from Adam-v ----------
def leaves(t, path=""):
    if isinstance(t, dict):
        for k, v in t.items(): yield from leaves(v, path + "/" + k)
    elif isinstance(t, (list, tuple)):
        for i, v in enumerate(t): yield from leaves(v, path + f"[{i}]")
    else: yield path, np.asarray(t)
def find_nu(opt):
    found = []
    def walk(x):
        if hasattr(x, "_fields") and "nu" in x._fields: found.append(x.nu)
        elif isinstance(x, (list, tuple)):
            for y in x: walk(y)
        elif isinstance(x, dict):
            for y in x.values(): walk(y)
    walk(opt); return found
def curvature_row(path):
    c = pickle.load(open(path, "rb")); nus = find_nu(c.get("opt_state"))
    if not nus: return None
    nu = nus[0]; nu = nu["model"] if isinstance(nu, dict) and "model" in nu else nu
    vals = []; blocks = {}
    for p, a in leaves(nu):
        a = a.astype(np.float64).ravel(); vals.append(a); b = p.split("/")[1] if p.count("/") >= 1 else p; b = b.split("[")[0]; blocks[b] = blocks.get(b, 0.0) + float(a.sum())
    v = np.concatenate(vals); n = v.size; tot = v.sum(); pr = tot ** 2 / max((v ** 2).sum(), 1e-300)
    vs = np.sort(v)[::-1]; cum = np.cumsum(vs) / tot; k90 = int(np.searchsorted(cum, 0.9)) + 1
    top = sorted(blocks.items(), key=lambda kv: -kv[1])[:3]
    return dict(n=n, pr_n=pr / n, k90=k90 / n, top=", ".join(f"{b} {100*s/tot:.1f}%" for b, s in top))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--no-grids", action="store_true"); a = ap.parse_args(); L.clear()
    say("=" * 120); say("sportC2 (the pre-parity graft night at d128) — PHYSICS PASS 2026-09-05 (analysis-time, descriptive; verdict authority = analyze_sportC2.py)"); say("=" * 120)

    # ---- A ----
    say("\n== A. ADMISSION + BANKED-GRID CENSUS + VAL-SELECTION PROVENANCE (artifact level) ==")
    adm = {x: admission(x, walk=not a.no_grids) for x in ARMS}
    for x in ARMS:
        o = adm[x]
        if not o.get("dir"): say(f"  {x}: run dir MISSING"); continue
        c = o.get("cfg", {}); v = o.get("argv", {}); pin = META[x]["pin"]; nb = o.get("n_bulk")
        pin_ok = nb in (pin, pin + 3) if META[x]["cell"] == "trm" else nb == pin
        say(f"  {x:3s} {META[x]['desc']:44s} step {o.get('step')} n_bulk {nb} (pin {pin}{' +3 loop scalars' if META[x]['cell']=='trm' else ''} {'OK' if pin_ok else 'MISMATCH'}) ema_in_ckpt {o.get('has_ema')} final_finite {o.get('final_finite')}")
        say(f"        cfg: d{c.get('d')} T{c.get('T')} cell {c.get('cell_kind')} z_norm '{c.get('z_norm')}' layout {c.get('sudoku_layout')} arity {c.get('pool_arity')} mixer {c.get('mixer_kind')} b_nl {c.get('beta_flux_nl')} fpa_k {c.get('fpa_k')} ni {c.get('ni_sigma')} "
            f"inner_k {c.get('inner_k')} hard_p {c.get('hard_p')} loss {c.get('loss_kind')}" + (f" trm h{c.get('trm_hidden')} L{c.get('trm_layers')} H{c.get('trm_h_cycles')} Lc{c.get('trm_l_cycles')} lam {c.get('trm_lambda')} beta {c.get('trm_beta')} mixer {c.get('trm_token_mixer')} gm_dim {c.get('trm_gm_dim')}" if c.get("cell_kind") == "trm" else ""))
        say(f"        argv: seed {v.get('seed')} ri_p {v.get('ri_p')} aug {v.get('sudoku_aug')} digit_aug {v.get('sudoku_digit_aug')} steps {v.get('steps')} batch {v.get('batch')} lr {v.get('lr')}->{v.get('lr_end')} warmup {v.get('warmup')} wd {v.get('wd')} beta2 {v.get('beta2')} ema {v.get('ema')} "
            f"sot {v.get('sot')} segs {v.get('sot_segments')} act {v.get('act')} remat {v.get('remat')} inner_k {v.get('inner_k')} hard_p {v.get('hard_p')} init_from {v.get('init_from')}")
        say(f"        STOPPED: {o.get('STOPPED.txt')} | NAN_ABORT: {o.get('NAN_ABORT.txt')} | RETRY_REMAT: {o.get('RETRY_REMAT.txt')} | resumes: {o.get('resumes.txt')} | val_best.txt: {o.get('val_best.txt')}")
        g = o.get("grids", []); fin = [s for s, ok in g if ok is not False]; bad = [s for s, ok in g if ok is False]; gA = o.get("gridsA")
        say(f"        banked grids (D): {fin} | NON-FINITE: {bad}" + (f" | stage-A grids: {[s for s, _ in gA]}" if gA else " | no stage-A dir"))
    say("\n  val-selection provenance (every vsel-labeled eval's ckpt path; the analyzer's one-ckpt gate) + OFFLINE re-selection on the headline key over both stages:")
    VSEL = {}
    for x in ARMS:
        t = META[x]["head_t"]; srcs = {}
        for nm, p in (("full_vsel", full_dir(x, "vsel") / "summary_all.json"), ("full_alt", evdir(x) / f"full_vsel_t{t}_alt" / "summary_all.json"), ("full64", evdir(x) / "full_vsel_t64" / "summary_all.json"),
                      ("scan", scan_dir(x) / "summary_all.json"), ("census_vsel", RUNS / f"sxcensus_p{TAG}{x}_vsel" / "census.json"), ("screen_vb", screen_dir(x, "vb") / "summary_all.json"),
                      ("retfm", evdir(x) / "retfm_t8" / "summary_all.json"), ("calib", RUNS / f"sxcalib_p{TAG}{x}_vsel" / "calib.json"),
                      ("screen_vb_hard", screen_dir(x, "vb_hard") / "summary_all.json"), ("calib_hard", RUNS / f"sxcalib_p{TAG}{x}_vsel_hard" / "calib.json")):
            s = jload(p)
            if s: srcs[nm] = ckpt_step_from_path(s.get("ckpt", ""), x)
        labels = {k: v[0] for k, v in srcs.items()}; uniq = set(labels.values()); eff = next((v[1] for v in srcs.values()), None)
        VSEL[x] = dict(label=(labels.get("full_vsel") or next(iter(uniq), "?")), eff=eff, consistent=(len(uniq) <= 1))
        key = META[x]["key"]; cand = []
        for dd, off in ((pdir(x), S_A if META[x]["two"] else 0), (pdirA(x), 0)):
            if not dd.exists() or (dd == pdirA(x) and not META[x]["two"]): continue
            banked = {int(re.search(r"ckpt_(\d+)\.pkl$", p.name).group(1)) for p in dd.glob("ckpt_[0-9]*.pkl")}
            _, mon = load_metrics(dd / "metrics.jsonl")
            cand += [(float(m[key]), s + off, ("A" if dd == pdirA(x) else ("B" if META[x]["two"] else "")) + f":{s}" if META[x]["two"] else f"{s}") for s, m in mon.items() if key in m and s in banked]
        best = max(cand, key=lambda q: (q[0], q[1])) if cand else None
        match = best and (VSEL[x]["label"] == best[2])
        say(f"  {x:3s} chain vsel = {VSEL[x]['label']:>10s} (eff step {VSEL[x]['eff']}) sources {labels} {'CONSISTENT' if VSEL[x]['consistent'] else 'INCONSISTENT'} | offline argmax {key}: {best[2] if best else '-'} val {f(best[0] if best else None,5,3)} {'MATCH' if match else 'DIFFERS'}")

    # ---- B ----
    say("\n== B. TRAJECTORIES (last-wins dedup; two-stage arms = stage A + stage B offset +50k; R4 = continuation steps 0..50k; headline key per arm) ==")
    TS = {}
    for x in ARMS:
        tr, mon, note = arm_traj(x); TS[x] = traj_summary(x, tr, mon); s = TS[x]
        say(f"  {x:3s} rows {s['n_train_rows']:4d} last step {s['last_step']} | first NON-FINITE row {s['first_bad']} | pace {f(s['pace'],5,2)} it/s (printed) | wall {s['t_first']} -> {s['t_last']} | {note}")
        say(f"        loss last {f(s['loss_last'],7,4)} | CE @10k {f(s['ce_10k'],6,4)} @20k {f(s['ce_20k'],6,4)} @30k {f(s['ce_30k'],6,4)} @50k {f(s['ce_50k'],6,4)} last {f(s['ce_last'],6,4)} min {f(s['ce_min'],6,4)} | A_total peak {f(s['A_peak'],10,1)} @10k {f(s['A_10k'],8,1)} last {f(s['A_last'],8,1)} | I_total @10k {f(s['I_10k'],9,1)} last {f(s['I_last'],9,1)} | rule_H last {f(s['ruleH_last'],5,3)}"
            + (f" | SOT rows: train_exact last {f(s['train_exact_last'],5,3)} max {f(s['train_exact_max'],5,3)} halt {f(s['halt_last'],5,3)} fresh {f(s['fresh_last'],5,3)} segs {f(s['mean_steps_last'],5,2)}" if s['train_exact_last'] is not None else ""))
        def ser(name, xs, p=3, every=1):
            if not xs: return f"{name}: -"
            xs = xs[::every] if len(xs) > 24 else xs
            return f"{name}: " + " ".join(f"{st//1000}k:{v:.{p}f}" for st, v in xs)
        say("        " + ser(META[x]["key"], s["val"], 3, 2) + f"  | peak {s['val_peak']} | end {s['val_end']} | rise over last 10k {('-' if s['val_rise10k'] is None else f'{100*s['val_rise10k']:+.2f}pp')}")
        say("        " + ser(("raw val" if META[x]["key"].endswith("_ema") else "EMA val"), s["val_other"], 3, 2))
        say("        " + ser("eta", s["eta"], 3, 2) + " | " + ser("eta_z", s["eta_z"][-3:], 3))
        if META[x]["cell"] == "rg": say("        " + ser("lamJ_max", s["lamj"], 2, 2)); say("        " + ser("retfm", s["retfm"], 2, 2) + " | " + ser("ret_sched", s["rets"], 2, 2))

    # ---- C ----
    say("\n== C. STABILITY CENSUS + THE wd TWIN BRIDGE (B0 sportC1 wd 1e-4 vs W0 wd 1.0; same seed/recipe) ==")
    say("  arm | STOPPED | min retfm (monitors) | eta end | lamJ max | A_total last | I_total last | census vsel t64/t256 | census final t64/t256")
    for x in ARMS:
        s = TS[x]; o = adm[x]; cz = {w: {int(r["t"]): r["exploded_frac"] for r in (census_json(x, w) or {}).get("rows", [])} for w in ("vsel", "final")}
        rf = min([v for _, v in s["retfm"]], default=None); et = s["eta"][-1][1] if s["eta"] else None; lj = max([v for _, v in s["lamj"]], default=None)
        say(f"  {x:3s} | {bool(o.get('STOPPED.txt'))!s:5s} | {f(rf,4,2)} | {f(et,5,3)} | {f(lj,5,2)} | {f(s['A_last'],9,1)} | {f(s['I_last'],9,1)} | {pp(cz['vsel'].get(64))}/{pp(cz['vsel'].get(256))} | {pp(cz['final'].get(64))}/{pp(cz['final'].get(256))}")
    say("\n  seed-paired wd twins at matched steps (B0 = sportC1 wd 1e-4 [memorized by 40k]; W0 = wd 1.0): step | CE B0/W0 | val B0/W0 | I_total B0/W0 | eta B0/W0 | rule_H B0/W0 | lamJ B0/W0 | retfm B0/W0")
    def c1_traj(arm):
        d = RUNS / f"pretrainsportC1_{arm}"; da = RUNS / f"pretrainsportC1_{arm}a"; tr, mon = load_metrics(d / "metrics.jsonl")
        trB = {k + S_A: v for k, v in tr.items()}; monB = {k + S_A: v for k, v in mon.items()}; trA, monA = load_metrics(da / "metrics.jsonl")
        return dict(sorted({**trA, **trB}.items())), dict(sorted({**monA, **monB}.items()))
    tb, mb = c1_traj("B0"); tw, mw, _ = arm_traj("W0")
    for st in sorted(set(mb) | set(mw)):
        if st % 4000 and st not in (2000, 6000): continue
        B_ = mb.get(st, {}); W_ = mw.get(st, {})
        if not B_ and not W_: continue
        g = lambda tr_, k: near({s: v.get(k) for s, v in tr_.items()}, st, window=1000)
        say(f"    {st:6d} | CE {f(g(tb,'ce_in'),6,4)}/{f(g(tw,'ce_in'),6,4)} | val {f(B_.get('val_t64'),5,3)}/{f(W_.get('val_t64'),5,3)} | I {f(g(tb,'I_total'),9,0)}/{f(g(tw,'I_total'),9,0)} | eta {f(B_.get('eta'),5,3)}/{f(W_.get('eta'),5,3)} | ruleH {f(g(tb,'rule_H'),5,3)}/{f(g(tw,'rule_H'),5,3)} | lamJ {f(B_.get('lam_joint_max'),5,2)}/{f(W_.get('lam_joint_max'),5,2)} | retfm {f(B_.get('ret_final_t8'),4,2)}/{f(W_.get('ret_final_t8'),4,2)}")
    say("  grafts vs the base at matched steps (W0 / R1 / R2 / R3; val_t64 monitor): step | val W0 R1 R2 R3 | CE W0 R1 R2 R3 | eta W0 R1 R2 R3")
    TR = {x: arm_traj(x)[:2] for x in ("W0", "R1", "R2", "R3")}
    for st in sorted(set().union(*[set(TR[x][1]) for x in TR])):
        if st % 8000 and st not in (2000, 4000): continue
        row = [TR[x][1].get(st, {}) for x in ("W0", "R1", "R2", "R3")]
        say(f"    {st:6d} | val " + " ".join(f(r.get('val_t64'),5,3) for r in row) + " | CE " + " ".join(f(near({s: v.get('ce_in') for s, v in TR[x][0].items()}, st, window=1000),6,4) for x in ("W0","R1","R2","R3")) + " | eta " + " ".join(f(r.get('eta'),5,3) for r in row))

    # ---- D ----
    say("\n== D. EVAL TABLE (vsel grid, HEADLINE weights unless named) ==")
    say("  arm | vcold | fcold | alt-weights vcold | b1 (B=1) | verified@1/8/32/128 | t1r@1/8/32/128 | majority@16/64/128/256 (vb screen) | retfm | census vsel t64/t256 | census final t64/t256")
    EV = {}
    for x in ARMS:
        t = META[x]["head_t"]; fv = jload(full_dir(x, "vsel") / "summary_all.json"); ff = jload(full_dir(x, "final") / "summary_all.json"); fa = jload(evdir(x) / f"full_vsel_t{t}_alt" / "summary_all.json")
        scan = jload(scan_dir(x) / "summary_all.json"); scr = jload(screen_dir(x, "vb") / "summary_all.json"); rf = jload(evdir(x) / "retfm_t8" / "summary_all.json")
        cz = {w: {int(r["t"]): r["exploded_frac"] for r in (census_json(x, w) or {}).get("rows", [])} for w in ("vsel", "final")}
        vk = scan.get("vote_at_k", {}) if scan else {}; t1 = scan.get("t1r_at_k", {}) if scan else {}; mj = scr.get("majority_vote_at_k", {}) if scr else {}; sv = scr.get("vote_at_k", {}) if scr else {}
        EV[x] = dict(vcold=fv["exact_acc"] if fv else None, fcold=ff["exact_acc"] if ff else None, acold=fa["exact_acc"] if fa else None, b1=scan.get("b1_exact") if scan else None, vk=vk, t1=t1, mj=mj, sv=sv,
                     retfm=rf["exact_acc"] if rf else None, cz=cz, scan_cold=scan.get("exact_acc") if scan else None, val_wrong=fv.get("valid_wrong_frac") if fv else None, viol=fv.get("mean_violations") if fv else None,
                     givens=fv.get("givens_kept_frac") if fv else None, mfe=fv.get("mean_first_exact") if fv else None, scr_cold=scr.get("exact_acc") if scr else None)
        e = EV[x]
        say(f"  {x:3s} | {pp(e['vcold'])} | {pp(e['fcold'])} | {pp(e['acold'])} ({'EMA' if not META[x]['head_ema'] else 'raw'}) | {pp(e['b1'])} | {pp(vk.get('1'))}/{pp(vk.get('8'))}/{pp(vk.get('32'))}/{pp(vk.get('128'))} | "
            f"{pp(t1.get('1'))}/{pp(t1.get('8'))}/{pp(t1.get('32'))}/{pp(t1.get('128'))} | {pp(mj.get('16'))}/{pp(mj.get('64'))}/{pp(mj.get('128'))}/{pp(mj.get('256'))} | {pp(e['retfm'])} | {pp(cz['vsel'].get(64))}/{pp(cz['vsel'].get(256))} | {pp(cz['final'].get(64))}/{pp(cz['final'].get(256))}")
    say("  full-test texture (vsel grid): arm | valid_wrong | mean violations | givens kept | mean first_exact (solved)")
    for x in ARMS:
        e = EV[x]; say(f"  {x:3s} | {f(e['val_wrong'],7,4)} | {f(e['viol'],6,2)} | {f(e['givens'],6,4)} | {f(e['mfe'],6,2)}")
    say("\n  CALIBRATION AT STALLS (strat-512 cold, vsel grid; tools/stall_calibration.py): arm | mode | cold | n stalled | top-5 correct on stalled | mean conf | top-5 correct solved | entropy step1 | entropy t64 stalled/solved | conf-wrong frac stalled")
    for x in ARMS:
        for mode in ("vsel", "vsel_hard"):
            c = calib_json(x, mode)
            if not c: continue
            say(f"  {x:3s} | {'HARD feedback' if c.get('hard_feedback') else 'soft':13s} K={c.get('inner_k')} | {pp(c.get('cold'))} | {c.get('n_stalled'):4d} | {pp(c.get('topk_correct_stalled'))} | {f(c.get('mean_conf_stalled'),5,3)} | {pp(c.get('topk_correct_solved'))} | {f(c.get('entropy_step1'),5,3)} | {f(c.get('entropy_t_stalled'),5,3)}/{f(c.get('entropy_t_solved'),5,3)} | {pp(c.get('conf_wrong_frac_stalled'))}")
    sh = jload(screen_dir("R3", "vb_hard") / "summary_all.json"); ss = jload(screen_dir("R3", "vb") / "summary_all.json")
    if sh: say(f"  R3 vsel grid in HARD-feedback inference (strat-512 cold, labeled): cold {pp(sh.get('exact_acc'))} vs soft {pp(ss.get('exact_acc') if ss else None)} | mean violations {f(sh.get('mean_violations'),6,2)} vs {f(ss.get('mean_violations') if ss else None,6,2)} | givens kept {f(sh.get('givens_kept_frac'),6,4)}")
    say("\n  X1 / X2 depth rows (EMA headline): D16 vsel | D16 final | D16 alt(raw) | D64 vsel  [sportC1 X0: 86.03 | 86.03 | 80.81 | 92.81]")
    for x in FIELD:
        d64 = jload(evdir(x) / "full_vsel_t64" / "summary_all.json"); say(f"  {x:3s} | {pp(EV[x]['vcold'])} | {pp(EV[x]['fcold'])} | {pp(EV[x]['acold'])} | {pp(d64['exact_acc'] if d64 else None)}")
    say("\n  screen curves (strat-512 k256 at the headline depth/weights; sAend = stage-A final, sB005000/sB015000 = stage-B +5k/+15k; R4/X: s015000/s035000 = continuation steps):")
    for x in ARMS:
        rows = []
        for p in sorted(RUNS.glob(f"sxscreen_p{TAG}{x}_*")):
            s = jload(p / "summary_all.json")
            if not s: continue
            tag = "_".join(p.name.split("_")[2:]); lab, eff = ckpt_step_from_path(s.get("ckpt", ""), x)
            rows.append((tag, lab, s["vote_at_k"].get("256"), s["vote_at_k"].get("128"), s["exact_acc"], s.get("majority_vote_at_k", {}).get("256"), s.get("givens_kept_frac")))
        say(f"  {x:3s}: " + " | ".join(f"{t}@{lab}: cold {pp(c)} v128 {pp(v1)} v256 {pp(v)} maj256 {pp(m)}{' [givens 0: GARBAGE]' if g == 0.0 else ''}" for t, lab, v, v1, c, m, g in rows))
    say("\n  RIDERS (sportC1 provenance closure + canvas EqR statistics):")
    for nm, p in (("B0 A:20k correct-grid scan", RUNS / "sxscan_psportC1B0_vselA20k"), ("B1 A:20k correct-grid scan", RUNS / "sxscan_psportC1B1_vselA20k"),
                  ("B0 memorized-grid scan (sportC1)", RUNS / "sxscan_psportC1B0"), ("B1 memorized-grid scan (sportC1)", RUNS / "sxscan_psportC1B1"),
                  ("C3X canvas sel-5k k128", RUNS / "sxrider_C3X_sel5k"), ("D4 canvas sel-5k k128", RUNS / "sxrider_D4_sel5k"), ("R0 sportC1 scan", RUNS / "sxscan_psportC1R0"), ("X0 sportC1 scan @D64", RUNS / "sxscan_psportC1X0")):
        s = jload(p / "summary_all.json")
        if not s: say(f"  {nm}: MISSING"); continue
        vk = s.get("vote_at_k", {}); t1 = s.get("t1r_at_k", {}); mj = s.get("majority_vote_at_k", {})
        say(f"  {nm:34s} n={s.get('n')} ckpt {s.get('ckpt','?').split('/')[-2:] } | cold {pp(s.get('exact_acc'))} b1 {pp(s.get('b1_exact'))} | verified@8/32/128 {pp(vk.get('8'))}/{pp(vk.get('32'))}/{pp(vk.get('128'))} | t1r@8/32/128 {pp(t1.get('8'))}/{pp(t1.get('32'))}/{pp(t1.get('128'))} | majority@16/64/128 {pp(mj.get('16'))}/{pp(mj.get('64'))}/{pp(mj.get('128'))}")
    for nm, p in (("B0 EMA full on A:20k (rider)", RUNS / "sxeval_psportC1B0/full_vselA20k_t64_ema"), ("B0 raw full on A:20k (sportC1)", RUNS / "sxeval_psportC1B0/full_vsel_t64"), ("B1 EMA full on A:20k (sportC1 alt)", RUNS / "sxeval_psportC1B1/full_vsel_t64_alt")):
        s = jload(p / "summary_all.json"); say(f"  {nm:34s} n={s.get('n') if s else '-'} | cold {pp(s.get('exact_acc') if s else None)} | ckpt {s.get('ckpt') if s else '-'}")

    # ---- E ----
    say("\n== E. PAIRED CONTRASTS (McNemar exact; identical puzzle sets by idx; intersections labeled) ==")
    Z = {}
    for x in ARMS:
        t = META[x]["head_t"]
        Z[x] = dict(scan=recs(scan_dir(x)), full=recs(full_dir(x, "vsel")), full_final=recs(full_dir(x, "final")), full_alt=recs(evdir(x) / f"full_vsel_t{t}_alt"), full64=recs(evdir(x) / "full_vsel_t64"))
    REFS = {"B0m": ("sxscan_psportC1B0", "sxeval_psportC1B0/full_vsel_t64"), "B0c": ("sxscan_psportC1B0_vselA20k", "sxeval_psportC1B0/full_vsel_t64"), "B0f": (None, "sxeval_psportC1B0/full_final_t64"),
            "B0e": (None, "sxeval_psportC1B0/full_vselA20k_t64_ema"), "B1m": ("sxscan_psportC1B1", "sxeval_psportC1B1/full_vsel_t64"), "B1c": ("sxscan_psportC1B1_vselA20k", "sxeval_psportC1B1/full_vsel_t64"),
            "B1e": (None, "sxeval_psportC1B1/full_vsel_t64_alt"), "R0": ("sxscan_psportC1R0", "sxeval_psportC1R0/full_vsel_t64"), "X0": ("sxscan_psportC1X0", "sxeval_psportC1X0/full_vsel_t16"), "X0d64": (None, "sxeval_psportC1X0/full_vsel_t64"),
            "D4": ("sxscan_psportBr2bD4", "sxeval_psportBr2bD4/full_t64"), "C3X": ("sxscan_psportBr2bC3X", "sxeval_psportBr2bC3X/full_t64"), "P3s1": ("sxscan_psportC0P3s1", "sxeval_psportC0P3s1/full_t64"),
            "C3Xr": ("sxrider_C3X_sel5k", None), "D4r": ("sxrider_D4_sel5k", None)}
    for k, (sc, fu) in REFS.items():
        Z[k] = dict(scan=recs(RUNS / sc) if sc else None, full=recs(RUNS / fu) if fu else None, full_final=None, full_alt=None, full64=None)
    say("  ref keys: B0m/B1m = sportC1 scans on the MEMORIZED stage-B grids; B0c/B1c = the riders' CORRECT A:20k grids; B0f = B0 memorized final full; B0e = B0 EMA full on A:20k (rider); B1e = B1 EMA on A:20k; X0 = D16 full / D64 scan; X0d64 = D64 full; C3Xr/D4r = canvas sel-5k riders")
    def getbits(x, kind, stat):
        z = Z.get(x, {}).get(kind)
        if z is None: return None, None
        if stat == "cold": return z["cold_exact"].astype(bool), z["idx"]
        if stat == "vote128": return (vote_bits(z, 128), z["idx"]) if "mi_first_hit" in z else (None, None)
        if stat == "b1": b = b1_bits(z); return (None, None) if b is None else (b, z["idx"])
        if stat == "t1r128": b = t1r_bits(z); return (None, None) if b is None else (b, z["idx"])
        return None, None
    def pair(x, y, kx, ky, stat, label=""):
        ax, ix = getbits(x, kx, stat); ay, iy = getbits(y, ky, stat)
        if ax is None or ay is None: say(f"  {x}[{kx}] vs {y}[{ky}] {stat}: data missing"); return
        if len(ix) != len(iy) or not np.array_equal(ix, iy):
            com, ia, ib = np.intersect1d(ix, iy, return_indices=True)
            if len(com) == 0: say(f"  {x}[{kx}] vs {y}[{ky}] {stat}: idx sets DISJOINT — not paired"); return
            ax, ay = ax[ia], ay[ib]; label = label + f" [INTERSECTION n={len(com)} of {len(ix)}/{len(iy)}]"
        oa, ob, p = mcnemar(ax, ay)
        say(f"  {x:5s}[{kx:10s}] vs {y:5s}[{ky:10s}] {stat:7s} n={len(ax):6d}: {pp(ax.mean())} vs {pp(ay.mean())} ({100*(ax.mean()-ay.mean()):+.2f}pp) | only-{x} {oa:6d} only-{y} {ob:6d} | p={p:.2e} | union {pp((ax|ay).mean())} {label}")
    say("  -- THE wd BRIDGE (W0 = B0 + wd 1.0; 20k scan set): W0 vs B0's correct grid (B0c) and vs its memorized grid (B0m) --")
    for st in ("cold", "b1", "vote128", "t1r128"):
        pair("W0", "B0c", "scan", "scan", st); pair("W0", "B0m", "scan", "scan", st)
    say("  -- THE GRAFTS vs the base (20k scan set) --")
    for x in ("R1", "R2", "R3"):
        for st in ("cold", "b1", "vote128", "t1r128"): pair(x, "W0", "scan", "scan", st)
    pair("R1", "R2", "scan", "scan", "vote128"); pair("R1", "R3", "scan", "scan", "vote128"); pair("R2", "R3", "scan", "scan", "vote128")
    say("  -- CONTINUATION (R4 = sportC1 R0 +50k in the field regime) --")
    for st in ("cold", "b1", "vote128", "t1r128"): pair("R4", "R0", "scan", "scan", st)
    pair("R4", "W0", "scan", "scan", "cold"); pair("R4", "W0", "scan", "scan", "vote128"); pair("R4", "B0c", "scan", "scan", "cold")
    say("  -- THE FIELD CELL (X1 = X0 - digit aug; X2 = X0 + group9 mixer; scans at D64, fulls at D16 and D64) --")
    for x in ("X1", "X2"):
        for st in ("cold", "b1", "vote128", "t1r128"): pair(x, "X0", "scan", "scan", st)
        pair(x, "X0", "full", "full", "cold", "(D16 full)"); pair(x, "X0d64", "full64", "full", "cold", "(D64 full)")
    pair("X1", "X2", "full", "full", "cold", "(D16)"); pair("X1", "X2", "scan", "scan", "vote128")
    say("  -- RIDERS: B0/B1 correct grids vs the memorized grids the sportC1 pass read (labeled there) --")
    for st in ("cold", "b1", "vote128", "t1r128"): pair("B0c", "B0m", "scan", "scan", st); pair("B1c", "B1m", "scan", "scan", st)
    pair("B0c", "B1c", "scan", "scan", "cold", "(z-norm seed pair, correct grids)"); pair("B0c", "B1c", "scan", "scan", "b1", "(seed pair)"); pair("B0c", "B1c", "scan", "scan", "vote128", "(seed pair)")
    pair("B0e", "B0c", "full", "full", "cold", "(B0 EMA vs raw on A:20k, full test)"); pair("B0e", "B1e", "full", "full", "cold", "(EMA rows on the correct grids, seed pair)")
    pair("W0", "B0e", "full", "full", "cold", "(W0 headline vs B0's EMA row)")
    say("  -- canvas EqR-statistic riders (sel-5k; paired on the intersection with the 20k scans) --")
    for r_ in ("C3Xr", "D4r"):
        pair(r_, "W0", "scan", "scan", "vote128"); pair(r_, "X0", "scan", "scan", "vote128"); pair(r_, r_.rstrip("r"), "scan", "scan", "vote128", "(rider vs the rung-2b 20k scan, same ckpt)")
    say("  -- FULL 422,786 pairings --")
    say("   vsel vs final per arm (drift / memorization read):")
    for x in ARMS:
        if Z[x]["full_final"] is not None and EV[x]["vcold"] != EV[x]["fcold"]: pair(x, x, "full", "full_final", "cold")
        else: say(f"  {x}: vsel == final grid (identical)")
    say("   headline vs alternate weights per arm (EMA lens; vsel grid):")
    for x in ARMS: pair(x, x, "full", "full_alt", "cold", f"(headline {'EMA' if META[x]['head_ema'] else 'raw'} vs alt {'raw' if META[x]['head_ema'] else 'EMA'})")
    say("   X1/X2 depth (D16 headline vs D64):")
    for x in FIELD: pair(x, x, "full64", "full", "cold", "(D64 vs D16)")
    say("   cross-arm (vsel grids, t64):")
    for x in ("R1", "R2", "R3", "R4"): pair(x, "W0", "full", "full", "cold")
    pair("W0", "B0c", "full", "full", "cold", "(wd bridge, full test, correct A:20k grid)"); pair("W0", "B0f", "full", "full", "cold", "(vs B0's memorized final)"); pair("R4", "R0", "full", "full", "cold"); pair("R4", "B0c", "full", "full", "cold")
    for x in ("W0", "R1", "R2", "R3", "R4"): pair(x, "R0", "full", "full", "cold"); pair(x, "D4", "full", "full", "cold"); pair(x, "P3s1", "full", "full", "cold")
    pair("X0d64", "W0", "full", "full", "cold", "(X0 @D64 vs W0)")
    say("  -- portfolio (labeled facts): unions on the 20k set, verified vote@128 / cold --")
    for combo in (("W0", "R1"), ("W0", "R2"), ("W0", "R3"), ("W0", "R1", "R2", "R3"), ("W0", "R1", "R2", "R3", "R4"), ("W0", "R1", "R2", "R3", "R4", "B0c", "B1c"), ("W0", "R1", "R2", "R3", "R4", "C3X", "D4"), ("W0", "R1", "R2", "R3", "R4", "X0"), ("R1", "R3")):
        zs = [Z[c]["scan"] for c in combo]
        if any(z is None for z in zs): say(f"  union {'+'.join(combo)}: data missing"); continue
        if any(not np.array_equal(z["idx"], zs[0]["idx"]) for z in zs): say(f"  union {'+'.join(combo)}: idx differ"); continue
        v = np.zeros(len(zs[0]["idx"]), bool); c_ = np.zeros_like(v)
        for z in zs: v |= vote_bits(z, 128); c_ |= z["cold_exact"].astype(bool)
        say(f"  union {'+'.join(combo):30s}: vote@128 {pp(v.mean())} | cold {pp(c_.mean())}")
    say("\n  per-octile (20k scan; rating octiles of the subsample): cold / b1 / verified@128 / t1r@128")
    for x in ["W0", "R1", "R2", "R3", "R4", "X1", "X2", "B0c", "B1c", "R0", "X0", "P3s1", "D4", "C3X"]:
        z = Z[x]["scan"]
        if z is None: continue
        rat = z["rating"]; qs = octiles(rat); cells = []; b1 = b1_bits(z); t1 = t1r_bits(z)
        for b in range(8):
            m = (rat >= qs[b]) & (rat < qs[b + 1])
            cells.append(f"{100*z['cold_exact'][m].mean():4.1f}/{('%4.1f' % (100*b1[m].mean())) if b1 is not None else ' -  '}/{100*vote_bits(z,128)[m].mean():4.1f}/{('%4.1f' % (100*t1[m].mean())) if t1 is not None else ' -  '}")
        say(f"  {x:5s} " + " | ".join(cells))
    say("\n  per-octile FULL-test cold (422,786; octiles of the full rating distribution): arm | o1..o8")
    for x in ["W0", "R1", "R2", "R3", "R4", "X1", "X2", "B0c", "B0e", "R0", "X0", "X0d64", "P3s1", "D4"]:
        z = Z[x]["full64"] if x in FIELD else Z[x]["full"]
        if z is None: continue
        rat = z["rating"]; qs = octiles(rat)
        say(f"  {x:5s}{' (D64)' if x in FIELD else '':6s} " + " | ".join(f"{100*z['cold_exact'][(rat >= qs[b]) & (rat < qs[b+1])].mean():5.1f}" for b in range(8)))

    # ---- F ----
    say("\n== F. FUNNEL MODEL (rho, r) per octile — lens-B form, fit on draws <= 64, checked at 128 ==")
    for x in ["W0", "R1", "R2", "R3", "R4", "X1", "X2", "B0c", "R0", "X0"]:
        z = Z[x]["scan"]
        if z is not None and "mi_first_hit" in z: funnel_table(z, x)

    # ---- G ----
    say("\n== G. SELECTOR — Top-1-by-residual (verification-free) vs the free verifier, per arm (20k scan) ==")
    say("  arm | b1 | t1r@1/8/32/128 | verified@1/8/32/128 | t1r@128 / verified@128 | P(resid correct < resid wrong) | spurious rate per wrong draw (wrong below the correct median) | majority@128 (screen)")
    for x in ARMS + ["B0c", "B1c"]:
        z = Z[x]["scan"]
        if z is None or "mi_resid_k" not in z: say(f"  {x}: no per-draw residuals"); continue
        ex = z["mi_exact_k"].astype(bool); rs = z["mi_resid_k"].astype(np.float64); fin = np.isfinite(rs); ex_f = ex & fin; wr_f = (~ex) & fin
        from scipy import stats
        u = stats.mannwhitneyu(rs[ex_f], rs[wr_f], alternative="less").statistic if ex_f.sum() and wr_f.sum() else None
        auc = None if u is None else 1.0 - float(u / (ex_f.sum() * wr_f.sum()))
        thr = np.median(rs[ex_f]) if ex_f.any() else np.nan; spur = float((rs[wr_f] <= thr).mean()) if wr_f.any() else None
        s = jload(scan_dir(x) / "summary_all.json") if x in ARMS else jload(RUNS / REFS[x][0] / "summary_all.json"); t1 = s.get("t1r_at_k", {}); vk = s.get("vote_at_k", {})
        ratio = (t1.get("128") / vk.get("128")) if t1.get("128") is not None and vk.get("128") else None
        mj = EV[x]["mj"].get("128") if x in EV else None
        say(f"  {x:4s} | {pp(s.get('b1_exact'))} | {pp(t1.get('1'))}/{pp(t1.get('8'))}/{pp(t1.get('32'))}/{pp(t1.get('128'))} | {pp(vk.get('1'))}/{pp(vk.get('8'))}/{pp(vk.get('32'))}/{pp(vk.get('128'))} | {f(ratio,5,3)} | {f(auc,5,3)} | {pp(spur)} | {pp(mj)}")

    # ---- H ----
    say("\n== H. EXTRA PREDICTIONS (plan §5 / freethink X-7 items the frozen analyzer does not score; descriptive) ==")
    s0 = TS["W0"]; say(f"  W0 val peak past 20k and vsel == final? peak {s0['val_peak']} | vsel label {VSEL['W0']['label']} | vcold {pp(EV['W0']['vcold'])} fcold {pp(EV['W0']['fcold'])} -> {'HIT' if s0['val_peak'] and s0['val_peak'][0] > 20000 else 'MISS'} (peak > 20k)")
    for x in ("R1", "R2", "R3", "R4"):
        rows = []
        for p in sorted(RUNS.glob(f"sxscreen_p{TAG}{x}_*")):
            s = jload(p / "summary_all.json"); z = recs(p)
            if not s or z is None or "mi_first_hit" not in z or "hard" in p.name: continue
            rho, r = fit_rho_r(z["mi_first_hit"]); rows.append(f"{'_'.join(p.name.split('_')[2:])}: rho {f(rho,4,2)} r {f(r,5,3)} v128 {pp(s['vote_at_k'].get('128'))}")
        say(f"  {x} rho per checkpoint (strat-512 screens): " + " | ".join(rows))
    c3 = calib_json("R3"); c0 = calib_json("W0")
    say(f"  R3 entropy at step 1 < .3: {f(c3.get('entropy_step1') if c3 else None,5,3)} (W0 {f(c0.get('entropy_step1') if c0 else None,5,3)}) -> {'HIT' if c3 and c3.get('entropy_step1') is not None and c3['entropy_step1'] < .3 else 'MISS'}")
    s4 = TS["R4"]; say(f"  R4 cold >= 42 (freethink) / [39,44] (registered): vcold {pp(EV['R4']['vcold'])} | val rise over last 10k {('-' if s4['val_rise10k'] is None else f'{100*s4['val_rise10k']:+.2f}pp')} (still rising?)")
    for x in ("R1", "R2", "R3"):
        say(f"  {x} cold vs W0 (+5-10pp predicted for R2): {pp(EV[x]['vcold'])} vs {pp(EV['W0']['vcold'])} = {('-' if EV[x]['vcold'] is None or EV['W0']['vcold'] is None else f'{100*(EV[x]['vcold']-EV['W0']['vcold']):+.2f}pp')}")
    say(f"  X1 drop vs X0 @D16 >= 3pp (freethink) / >= 1pp (registered): {pp(EV['X1']['vcold'])} vs 86.03 = {('-' if EV['X1']['vcold'] is None else f'{100*(EV['X1']['vcold']-.8603):+.2f}pp')}")
    say(f"  X2 within +-1pp of X0 @D16: {pp(EV['X2']['vcold'])} vs 86.03 = {('-' if EV['X2']['vcold'] is None else f'{100*(EV['X2']['vcold']-.8603):+.2f}pp')} | X2 params {adm['X2'].get('n_bulk')} (registered 5,839,874 + 3 loop scalars)")
    for x in NATIVE:
        s = TS[x]; say(f"  {x} A_total closed <= 500 nats by 10k: @10k {f(s['A_10k'],8,1)} -> {'HIT' if s['A_10k'] is not None and s['A_10k'] <= 500 else 'MISS'} | last {f(s['A_last'],8,1)} | retfm min {f(min([v for _, v in s['retfm']], default=None),4,2)}")
    say("  epochs seen (samples / 1,001,000 distinct pairs): W0-R3 80k x 64 = 5.1 | R4 +50k x 384 = +19.2 (total 38.4 on R0's weights) | X1/X2 50k x 768 segments = 38.4 segment-passes")

    # ---- J ----
    say("\n== J. CURVATURE CONCENTRATION from Adam-v (diagonal-Fisher proxy; the freethink ground-3 lens) ==")
    say("  ckpt | regime | n | PR/n | params carrying 90% of mass | top-3 blocks")
    for name, path, regime in ([(f"{x} final", pdir(x) / "ckpt_latest.pkl", META[x]["desc"]) for x in ARMS] +
                               [("sportC1 B0a @20k (vsel)", RUNS / "pretrainsportC1_B0a/ckpt_020000.pkl", "wd 1e-4, our lr"), ("sportC1 B0 @80k (memorized)", RUNS / "pretrainsportC1_B0/ckpt_latest.pkl", "wd 1e-4 floor"),
                                ("sportC1 R0 @50k", RUNS / "pretrainsportC1_R0/ckpt_latest.pkl", "field regime"), ("sportC1 X0 @50k", RUNS / "pretrainsportC1_X0/ckpt_latest.pkl", "field cell")]):
        try:
            r = curvature_row(path) if Path(path).exists() else None
        except Exception as e:
            r = None; say(f"  {name}: {type(e).__name__}: {e}")
        if r is None: say(f"  {name:28s} | {regime:40s} | no Adam nu"); continue
        say(f"  {name:28s} | {regime:40s} | {r['n']:8d} | {r['pr_n']:.2e} | {100*r['k90']:6.3f}% | {r['top']}")
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text("\n".join(L) + "\n"); say(f"\nartifact -> {OUT}")

if __name__ == "__main__":
    main()
