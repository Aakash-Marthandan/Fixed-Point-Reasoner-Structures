#!/usr/bin/env python3
"""CHAMPION sportC1 (d128 native round) — physics pass, written AT ANALYSIS TIME
(2026-09-03; descriptive only). The registered verdict authority is
tools/analyze_sportC1.py (byte-untouched since the registration commit 5317955,
selftest 33/33); NOTHING here is a decision rule.

Reads the extracted sportC1 corpus in runs/ and answers the mechanism questions
the registration posed but its rules do not read:
  A. ADMISSION + grid census — every arm's ckpt config + config.json argv at
     artifact level (param pins 3,004,530 / 3,004,658 / 5,037,058), STOPPED /
     NAN_ABORT / resumes, which banked 5k grids are FINITE, val-selection
     provenance (the grid every vsel eval actually ran on) + an OFFLINE
     re-selection over BOTH stages (the PM-1 check).
  B. TRAJECTORIES — train rows + 2k monitors (raw AND EMA val rows; the field
     cell's SOT rows: train_exact / halt_frac / q_loss), two-stage arms stitched
     (stage A as-is, stage B offset +50k), last-wins dedup.
  C. DEATH CENSUS + THE z-NORM TWIN BRIDGE — A0/A1 death intervals, the last
     monitor before death, the attention channel at death (H-48 discriminator),
     and the seed-paired A-vs-B trajectories at matched steps (eta, val, CE, I).
  D. EVAL TABLE — vsel cold (headline weights) | final cold | alt-weights cold |
     b1 | verified@k | Top-1-residual@k | majority@k | retfm | screen curve |
     census rows (vsel+final, t64/t256) | X0's D16/D64 rows.
  E. PAIRED McNemar on identical puzzle sets — the 20k scan set (z-norm bridge,
     regime, architecture-at-matched-regime, ladder d96->d128, canvas refs,
     portfolio unions) and the full 422,786 set (vsel vs final, EMA vs raw,
     X0 depth D16->D64, X0 ACT gain, cross-arm) + per-octile tables.
  F. FUNNEL (rho, r) per octile on every scan (lens-B model, reused).
  G. SELECTOR — Top-1-residual vs the free verifier per arm incl. the field cell.
  H. EXTRA PREDICTIONS (plan §12.8 items the analyzer does not score).
  I. FLUX PROFILE (optional; --flux-n N > 0): per-cut I_s/A_s on the native arms.

  .venv/bin/python tools/analyze_sportC1_physics.py [--flux-n 0] [--no-grids]
      -> runs/analysis/sportC1_physics_20260903.txt
"""
from __future__ import annotations
import argparse, json, math, os, pickle, re, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(os.environ.get("QHRRN_RUNS", ROOT / "runs"))
OUT = RUNS / "analysis" / "sportC1_physics_20260903.txt"
TAG = "sportC1"
NATIVE = ["A0", "A1", "B0", "B1", "R0"]
FIELD = ["X0", "X0n"]
ARMS = NATIVE + FIELD
S_A = 50000  # two-stage split (50k cosine + 30k floor)
META = {  # arm -> dict
    "A0": dict(norm=False, two=True, seed=0, regime="ours", cell="rg", key="val_t64", head_ema=False, head_t=64, desc="champion recipe s0 (no norm)"),
    "A1": dict(norm=False, two=True, seed=1, regime="ours", cell="rg", key="val_t64", head_ema=False, head_t=64, desc="champion recipe s1 (no norm)"),
    "B0": dict(norm=True, two=True, seed=0, regime="ours", cell="rg", key="val_t64", head_ema=False, head_t=64, desc="A0 + z-norm"),
    "B1": dict(norm=True, two=True, seed=1, regime="ours", cell="rg", key="val_t64", head_ema=False, head_t=64, desc="A1 + z-norm"),
    "R0": dict(norm=True, two=False, seed=0, regime="field", cell="rg", key="val_t64_ema", head_ema=True, head_t=64, desc="our cell, FIELD regime (+z-norm, EMA)"),
    "X0": dict(norm=None, two=False, seed=0, regime="field", cell="trm", key="val_t16_ema", head_ema=True, head_t=16, desc="FIELD baseline: TRM cell + SOT + ACT (EMA)"),
    "X0n": dict(norm=None, two=False, seed=0, regime="field", cell="trm", key="val_t16_ema", head_ema=True, head_t=16, desc="X0 without ACT"),
}
PINS = {"rg_nonorm": 3004530, "rg_norm": 3004658, "trm": 5037058}
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
            c = pickle.load(open(p, "rb"))
            rows.append((int(c["step"]), walk_finite(c["state"]["model"]) if walk else None))
        except Exception:
            rows.append((-1, False))
    return rows

def admission(arm, walk=True):
    d = pdir(arm); out = {"dir": d.exists()}
    if not d.exists(): return out
    ck = d / "ckpt_latest.pkl"
    if ck.exists():
        c = pickle.load(open(ck, "rb")); cfg = c.get("config", {})
        out["step"] = int(c["step"])
        out["cfg"] = {k: cfg.get(k) for k in ("d", "T", "scales", "canvas", "pool_arity", "mixer_kind", "sudoku_layout",
                                                "attn_max_hw", "d_task", "d_code", "d_b", "beta_flux", "beta_flux_nl",
                                                "fpa_k", "fpa_eps", "ni_sigma", "z_norm", "cell_kind", "loss_kind",
                                                "trm_hidden", "trm_layers", "trm_h_cycles", "trm_l_cycles", "trm_lambda", "trm_beta")}
        out["n_bulk"] = count_params(c["state"]["model"])
        out["has_ema"] = c.get("state_ema") is not None
        out["final_finite"] = walk_finite(c["state"]["model"]) if walk else None
    cj = d / "config.json"
    if cj.exists():
        a = json.loads(cj.read_text()).get("argv", {})
        out["argv"] = {k: a.get(k) for k in ("T", "seed", "ri_p", "ni_sigma", "sudoku_aug", "sudoku_digit_aug", "steps", "lr", "lr_end",
                                              "warmup", "wd", "batch", "beta2", "ema", "init_from", "width_scale", "fpa_k",
                                              "beta_flux_nl", "z_norm", "cell", "sot", "act", "loss", "remat")}
    for nm in ("STOPPED.txt", "NAN_ABORT.txt", "resumes.txt", "val_best.txt"):
        p = d / nm; out[nm] = p.read_text().strip() if p.exists() else None
    da = pdirA(arm)
    if da.exists():
        for nm in ("NAN_ABORT.txt", "resumes.txt"):
            p = da / nm
            if p.exists() and not out.get(nm): out[nm] = "(stage A) " + p.read_text().strip()
    out["grids"] = grid_census(d, walk)
    out["gridsA"] = grid_census(da, walk) if da.exists() else None
    out["stageA_dir"] = da.exists()
    return out

def ckpt_step_from_path(path: str, arm: str):
    """-> (label, effective_step) from an eval summary's ckpt path."""
    if not path: return ("?", None)
    name = path.split("/")[-1]; dd = path.split("/")[-2] if "/" in path else ""
    if name == "ckpt_latest.pkl": return ("final", None)
    m = re.search(r"ckpt_(\d+)\.pkl", name)
    if not m: return ("?", None)
    s = int(m.group(1))
    if META[arm]["two"] and dd.endswith("a"): return (f"A:{s}", s)
    if META[arm]["two"]: return (f"B:{s}", S_A + s)
    return (f"{s}", s)


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
        da = pdirA(arm); stopped = (d / "STOPPED.txt").exists()
        if stopped:
            note = "died in stage A: D holds the stage-A trajectory (steps as-is)"
        else:
            trB = {k + S_A: dict(v, step=k + S_A) for k, v in tr.items()}
            monB = {k + S_A: dict(v, step=k + S_A) for k, v in mon.items()}
            trA, monA = load_metrics(da / "metrics.jsonl") if da.exists() else ({}, {})
            note = "stage A trajectory present; stage B offset +50k" if trA else "stage A trajectory NOT shipped; stage B offset +50k"
            tr = dict(sorted({**trA, **trB}.items())); mon = dict(sorted({**monA, **monB}.items()))
    return tr, mon, note

def finite(v):
    try: return v is not None and math.isfinite(float(v))
    except Exception: return False

def near(dct, target, before=True, window=None):
    """value at the last finite step <= target (or first >= target); with window, only if that step is within `window` of target."""
    ks = [s for s in dct if finite(dct[s]) and (s <= target if before else s >= target)]
    if not ks: return None
    k = max(ks) if before else min(ks)
    if window is not None and abs(k - target) > window: return None
    return dct[k]

def traj_summary(arm, tr, mon):
    steps = list(tr.keys()); key = META[arm]["key"]
    first_bad = next((s for s in steps if not finite(tr[s].get("loss"))), None)
    last_good = max([s for s in steps if finite(tr[s].get("loss"))], default=None)
    A = {s: tr[s].get("A_total") for s in steps}; I = {s: tr[s].get("I_total") for s in steps}
    CE = {s: tr[s].get("ce_in") for s in steps}
    A_peak = max([A[s] for s in steps if finite(A[s])], default=None)
    pace = float(np.median([tr[s]["steps_per_sec"] for s in steps if finite(tr[s].get("steps_per_sec")) and tr[s]["steps_per_sec"] > 0])) if steps else None
    ms = [s for s in mon if s <= (last_good or 10**9)]
    def mseries(k): return [(s, mon[s].get(k)) for s in ms if finite(mon[s].get(k))]
    val = mseries(key); val_raw = mseries(key.replace("_ema", "")) if key.endswith("_ema") else mseries(key + "_ema")
    vpk = max(val, key=lambda x: x[1]) if val else None
    rise = None
    if len(val) >= 2:
        s_end, v_end = val[-1]; prev = [v for s, v in val if s <= s_end - 10000]
        rise = (v_end - prev[-1]) if prev else None
    ce_min = min([CE[s] for s in steps if finite(CE[s])], default=None)
    tx = [tr[s].get("train_exact") for s in steps if finite(tr[s].get("train_exact"))]
    hf = [tr[s].get("halt_frac") for s in steps if finite(tr[s].get("halt_frac"))]
    ts = [tr[s].get("t") for s in steps if tr[s].get("t")]
    return dict(n_train_rows=len(steps), first_bad=first_bad, last_good=last_good, last_step=max(steps) if steps else None,
                loss_last=tr[last_good]["loss"] if last_good else None, ce_last=near(CE, last_good or 0), ce_min=ce_min,
                ce_10k=near(CE, 10000, window=2000), ce_30k=near(CE, 30000, window=2000), ce_50k=near(CE, 50000, window=2000),
                A_peak=A_peak, A_2k=near(A, 2000, False, window=2000), A_10k=near(A, 10000, window=2000), A_last=near(A, last_good or 0),
                I_2k=near(I, 2000, False, window=2000), I_10k=near(I, 10000, window=2000), I_last=near(I, last_good or 0),
                ruleH_last=near({s: tr[s].get("rule_H") for s in steps}, last_good or 0), pace=pace,
                val=val, val_other=val_raw, val_peak=vpk, val_end=val[-1] if val else None, val_rise10k=rise,
                eta=mseries("eta"), eta_z=mseries("eta_z"), lamj=mseries("lam_joint_max"), retfm=mseries("ret_final_t8"),
                rets=mseries("ret_sched_t8"), drift=mseries("fp_drift_mean"),
                train_exact_last=(tx[-1] if tx else None), train_exact_max=(max(tx) if tx else None),
                halt_last=(hf[-1] if hf else None), t_first=(ts[0] if ts else None), t_last=(ts[-1] if ts else None))


# ---------- D/E. evals ----------
def recs(p):
    p = Path(p); q = p / "records_all.npz"
    if not q.exists():
        parts = sorted(p.glob("records_s*.npz"))
        if not parts: return None
        arrs = [dict(np.load(x, allow_pickle=True)) for x in parts]
        keys = [k for k in arrs[0] if all(k in a for a in arrs)]
        z = {k: np.concatenate([a[k] for a in arrs]) for k in keys}
    else:
        z = dict(np.load(q, allow_pickle=True))
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

def fit_rho_r(fh, k_fit=64):
    fh = np.asarray(fh); hit = (fh >= 0) & (fh < k_fit)
    t = fh[hit]; n_c = int(np.sum(~hit)); best = (-np.inf, None, None)
    for rho in np.linspace(0.02, 1.0, 50):
        for r in np.geomspace(1e-3, 0.9, 60):
            ll = np.sum(np.log(rho) + t * np.log1p(-r) + np.log(r)) if len(t) else 0.0
            ll += n_c * np.log(max(1 - rho + rho * (1 - r) ** k_fit, 1e-300))
            if ll > best[0]: best = (ll, rho, r)
    return best[1], best[2]

def octiles(rat):
    qs = np.quantile(rat, np.linspace(0, 1, 9)); qs[-1] += 1; return qs

def funnel_table(z, label):
    rat = z["rating"]; fh = z["mi_first_hit"]; cold = z["cold_exact"].astype(bool); qs = octiles(rat)
    say(f"  {label}: octile [rating) | n | cold | b1 | draw-hit@128 | rho | r | pred@128")
    b1 = b1_bits(z)
    for b in range(8):
        m = (rat >= qs[b]) & (rat < qs[b + 1])
        if m.sum() == 0: continue
        rho, r = fit_rho_r(fh[m]); pred = rho * (1 - (1 - r) ** 128) if rho is not None else None
        act = float(np.mean((fh[m] >= 0) & (fh[m] < 128)))
        say(f"    {b} [{qs[b]:.0f},{qs[b+1]:.0f}) n={m.sum():5d} cold {pp(cold[m].mean())} b1 {pp(b1[m].mean() if b1 is not None else None)} hit {pp(act)} rho {f(rho,5,2)} r {f(r,6,3)} pred {pp(pred)}")


# ---------- I. flux (optional) ----------
def flux_profile(arm, n_puz, sel_idx, Q):
    import jax, jax.numpy as jnp
    sys.path.insert(0, str(ROOT / "src"))
    from qhrrn2 import episodic as E, model as M
    from qhrrn2.config import Config
    saved = E.load_ckpt(pdir(arm) / "ckpt_latest.pkl")
    defaults = Config(); cfg = Config(**{k: type(getattr(defaults, k))(v) for k, v in saved["config"].items()})
    params = saved["state"]["model"]; tv = jnp.asarray(saved["state"]["table"][0])
    eta, eta_z = (float(v) for v in M.eq_etas(params, cfg))
    x = jnp.asarray(Q[sel_idx[:n_puz]].astype(np.int32)); T = cfg.T
    def one(xc):
        y = jax.nn.one_hot(jnp.full(xc.shape, 10, jnp.int32), M.VOCAB).transpose(2, 0, 1)
        z = None; fl = fa = None
        for t in range(64):
            tn = min(t, T - 1) / max(T - 1, 1)
            out = M.forward_fields(params, cfg, M.build_fields_soft(xc, y), t_norm=tn, tau=1.0, rng=None, task_vec=tv, z_in=z)
            z = out.z_fine if z is None else z + eta_z * (out.z_fine - z)
            p = jax.nn.softmax(out.logits, axis=-1).transpose(2, 0, 1)
            y = y + eta * (p - y); fl, fa = out.flux, out.flux_attn
        return fl, fa
    fl, fa = jax.jit(jax.vmap(one))(x)
    return np.asarray(fl), np.asarray(fa), eta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flux-n", type=int, default=0)
    ap.add_argument("--no-grids", action="store_true")
    a = ap.parse_args(); L.clear()
    say("=" * 120)
    say("CHAMPION sportC1 (d128 native round) — PHYSICS PASS 2026-09-03 (analysis-time, descriptive; verdict authority = analyze_sportC1.py)")
    say("=" * 120)

    # ---- A ----
    say("\n== A. ADMISSION + BANKED-GRID CENSUS + VAL-SELECTION PROVENANCE (artifact level) ==")
    adm = {x: admission(x, walk=not a.no_grids) for x in ARMS}
    for x in ARMS:
        o = adm[x]
        if not o.get("dir"): say(f"  {x}: run dir MISSING"); continue
        c = o.get("cfg", {}); v = o.get("argv", {})
        pin = PINS["trm"] if c.get("cell_kind") == "trm" else (PINS["rg_norm"] if c.get("z_norm") else PINS["rg_nonorm"])
        say(f"  {x:3s} {META[x]['desc']:44s} step {o.get('step')} n_bulk {o.get('n_bulk')} (pin {pin} {'OK' if o.get('n_bulk') == pin else 'MISMATCH'}) ema_in_ckpt {o.get('has_ema')} final_finite {o.get('final_finite')}")
        say(f"        cfg: d{c.get('d')} T{c.get('T')} cell {c.get('cell_kind')} z_norm '{c.get('z_norm')}' layout {c.get('sudoku_layout')} arity {c.get('pool_arity')} mixer {c.get('mixer_kind')} "
            f"b_nl {c.get('beta_flux_nl')} b_flux {c.get('beta_flux')} fpa_k {c.get('fpa_k')} ni {c.get('ni_sigma')} loss {c.get('loss_kind')}"
            + (f" trm h{c.get('trm_hidden')} L{c.get('trm_layers')} H{c.get('trm_h_cycles')} Lc{c.get('trm_l_cycles')} lam {c.get('trm_lambda')} beta {c.get('trm_beta')}" if c.get("cell_kind") == "trm" else ""))
        say(f"        argv: seed {v.get('seed')} ri_p {v.get('ri_p')} aug {v.get('sudoku_aug')} digit_aug {v.get('sudoku_digit_aug')} steps {v.get('steps')} batch {v.get('batch')} lr {v.get('lr')}->{v.get('lr_end')} warmup {v.get('warmup')} "
            f"wd {v.get('wd')} beta2 {v.get('beta2')} ema {v.get('ema')} sot {v.get('sot')} act {v.get('act')} remat {v.get('remat')} init_from {bool(v.get('init_from'))}")
        say(f"        STOPPED: {o.get('STOPPED.txt')} | NAN_ABORT: {o.get('NAN_ABORT.txt')} | resumes: {o.get('resumes.txt')} | val_best.txt: {o.get('val_best.txt')}")
        g = o.get("grids", []); fin = [s for s, ok in g if ok is not False]; bad = [s for s, ok in g if ok is False]
        gA = o.get("gridsA")
        say(f"        banked grids (D): {fin} | NON-FINITE: {bad}" + (f" | stage-A dir grids: {[s for s, _ in gA]}" if gA else " | no stage-A dir"))
    # vsel provenance from the eval summaries + offline re-selection
    say("\n  val-selection provenance (the grid each vsel eval ran on; from summary ckpt fields) + OFFLINE re-selection on the headline key over both stages:")
    VSEL = {}
    for x in ARMS:
        t = META[x]["head_t"]; srcs = {}
        for nm, p in (("full_vsel", full_dir(x, "vsel") / "summary_all.json"), ("full_alt", full_dir(x, "vsel").parent / f"full_vsel_t{t}_alt" / "summary_all.json"),
                      ("scan", scan_dir(x) / "summary_all.json"), ("census_vsel", RUNS / f"sxcensus_p{TAG}{x}_vsel" / "census.json"),
                      ("screen_vb", screen_dir(x, "vb") / "summary_all.json"), ("retfm", evdir(x) / "retfm_t8" / "summary_all.json")):
            s = jload(p)
            if s: srcs[nm] = ckpt_step_from_path(s.get("ckpt", ""), x)
        labels = {k: v[0] for k, v in srcs.items()}; uniq = set(labels.values())
        eff = next((v[1] for v in srcs.values()), None)
        VSEL[x] = dict(label=(labels.get("full_vsel") or next(iter(uniq), "?")), eff=eff, consistent=(len(uniq) <= 1))
        # offline re-selection
        key = META[x]["key"]; cand = []
        for dd, off in ((pdir(x), 0 if not META[x]["two"] or adm[x].get("STOPPED.txt") else S_A), (pdirA(x), 0)):
            if not dd.exists() or (dd == pdirA(x) and (adm[x].get("STOPPED.txt") or not META[x]["two"])): continue
            banked = {int(re.search(r"ckpt_(\d+)\.pkl$", p.name).group(1)) for p in dd.glob("ckpt_[0-9]*.pkl")}
            _, mon = load_metrics(dd / "metrics.jsonl")
            cand += [(float(m[key]), s + off, ("A" if dd == pdirA(x) else ("B" if META[x]["two"] and off else "")) + f":{s}") for s, m in mon.items() if key in m and s in banked]
        best = max(cand, key=lambda q: (q[0], q[1])) if cand else None
        say(f"  {x:3s} chain vsel = {VSEL[x]['label']:>10s} (eff step {VSEL[x]['eff']}) sources {labels} {'CONSISTENT' if VSEL[x]['consistent'] else 'INCONSISTENT'} | offline argmax {key}: {best[2] if best else '-'} val {f(best[0] if best else None,5,3)} {'MATCH' if best and (VSEL[x]['label'] == best[2] or VSEL[x]['label'].lstrip('AB:') == best[2].lstrip('AB:')) else 'DIFFERS'}")

    # ---- B ----
    say("\n== B. TRAJECTORIES (last-wins dedup; two-stage arms = stage A + stage B offset +50k; headline key per arm) ==")
    TS = {}
    for x in ARMS:
        tr, mon, note = arm_traj(x); TS[x] = traj_summary(x, tr, mon); s = TS[x]
        say(f"  {x:3s} rows {s['n_train_rows']:4d} last step {s['last_step']} | first NON-FINITE row {s['first_bad']} | last finite {s['last_good']} | pace {f(s['pace'],5,2)} it/s | wall {s['t_first']} -> {s['t_last']} | {note}")
        say(f"        loss last {f(s['loss_last'],7,4)} | CE @10k {f(s['ce_10k'],6,4)} @30k {f(s['ce_30k'],6,4)} @50k {f(s['ce_50k'],6,4)} last {f(s['ce_last'],6,4)} min {f(s['ce_min'],6,4)} | A_total peak {f(s['A_peak'],10,1)} @2k {f(s['A_2k'],9,1)} @10k {f(s['A_10k'],8,1)} last {f(s['A_last'],8,1)} | I_total @10k {f(s['I_10k'],9,1)} last {f(s['I_last'],9,1)} | rule_H last {f(s['ruleH_last'],5,3)}"
            + (f" | train_exact last {f(s['train_exact_last'],5,3)} max {f(s['train_exact_max'],5,3)} halt_frac last {f(s['halt_last'],5,3)}" if META[x]['cell'] == 'trm' else ""))
        def ser(name, xs, p=3, every=1):
            if not xs: return f"{name}: -"
            xs = xs[::every] if len(xs) > 24 else xs
            return f"{name}: " + " ".join(f"{st//1000}k:{v:.{p}f}" for st, v in xs)
        say("        " + ser(f"{META[x]['key']}", s["val"], 3, 2) + f"  | peak {s['val_peak']} | end {s['val_end']} | rise over last 10k {('-' if s['val_rise10k'] is None else f'{100*s['val_rise10k']:+.2f}pp')}")
        say("        " + ser(("raw val" if META[x]["key"].endswith("_ema") else "EMA val"), s["val_other"], 3, 2))
        say("        " + ser("eta", s["eta"], 3, 2) + " | " + ser("eta_z", s["eta_z"][-3:], 3))
        if META[x]["cell"] == "rg":
            say("        " + ser("lamJ_max", s["lamj"], 2, 2))
            say("        " + ser("retfm", s["retfm"], 2, 2) + " | " + ser("ret_sched", s["rets"], 2, 2))

    # ---- C ----
    say("\n== C. DEATH CENSUS + THE z-NORM TWIN BRIDGE (H-50's test) ==")
    say("  arm | norm | death (NAN_ABORT step / grid interval) | last monitor before death: eta / lamJ / retfm / ret_sched / val | A_total last-finite | I_total last-finite | CE last-finite")
    for x in NATIVE:
        s = TS[x]; o = adm[x]; died = s["first_bad"]
        nab = o.get("NAN_ABORT.txt") or ""; m = re.search(r"step (\d+)", nab); nstep = int(m.group(1)) if m else None
        g = o.get("grids", []); fin = [st for st, ok in g if ok is not False]; bad = [st for st, ok in g if ok is False]
        if o.get("STOPPED.txt"):
            lo = max(fin) if fin else None; interval = f"({lo}, {nstep or '?'}]"; died = died or nstep
        else: interval = "clean"
        def last(xs):
            xs = [q for q in xs if died is None or q[0] < died]; return xs[-1][1] if xs else None
        say(f"  {x:3s} | {'z-norm' if META[x]['norm'] else 'none  '} | {interval:>16s} | eta {f(last(s['eta']),5,3)} lamJ {f(last(s['lamj']),5,2)} retfm {f(last(s['retfm']),4,2)} ret_s {f(last(s['rets']),4,2)} val {f(last(s['val']),5,3)} | A {f(s['A_last'],9,1)} | I {f(s['I_last'],9,1)} | CE {f(s['ce_last'],6,4)}")
    say("  READ: A_total at every death = the dose's CLOSED value (H-48 discriminator) if <= ~500 nats; the free stream I_total is the inflating channel.")
    say("\n  seed-paired twin trajectories at matched steps (A = no norm, B = z-norm): step | eta A/B | val A/B | CE A/B | I_total A/B | lamJ A/B | retfm A/B")
    for pair in (("A0", "B0"), ("A1", "B1")):
        xa, xb = pair; ta, ma, _ = arm_traj(xa); tb, mb, _ = arm_traj(xb)
        steps = sorted(set(ma) | set(mb)); steps = [st for st in steps if st <= 80000]
        say(f"  pair {xa}/{xb}:")
        for st in steps:
            if st % 4000 and st not in (2000, 6000): continue
            A_ = ma.get(st, {}); B_ = mb.get(st, {}); tA = near({k: v.get("ce_in") for k, v in ta.items()}, st, window=1000); tB = near({k: v.get("ce_in") for k, v in tb.items()}, st, window=1000)
            iA = near({k: v.get("I_total") for k, v in ta.items()}, st, window=1000); iB = near({k: v.get("I_total") for k, v in tb.items()}, st, window=1000)
            if not A_ and not B_: continue
            say(f"    {st:6d} | eta {f(A_.get('eta'),5,3)}/{f(B_.get('eta'),5,3)} | val {f(A_.get('val_t64'),5,3)}/{f(B_.get('val_t64'),5,3)} | CE {f(tA,6,4)}/{f(tB,6,4)} | I {f(iA,9,0)}/{f(iB,9,0)} | lamJ {f(A_.get('lam_joint_max'),5,2)}/{f(B_.get('lam_joint_max'),5,2)} | retfm {f(A_.get('ret_final_t8'),4,2)}/{f(B_.get('ret_final_t8'),4,2)}")

    # ---- D ----
    say("\n== D. EVAL TABLE (vsel grid, HEADLINE weights unless named; STOPPED arms = amputated, labeled) ==")
    say("  arm | vcold | fcold | alt-weights vcold | b1 (B=1) | verified@1/8/32/128 | t1r@1/8/32/128 | majority@16/64/128/256 (vb screen) | retfm | census vsel t64/t256 | census final t64/t256")
    EV = {}
    for x in ARMS:
        t = META[x]["head_t"]
        fv = jload(full_dir(x, "vsel") / "summary_all.json"); ff = jload(full_dir(x, "final") / "summary_all.json")
        fa = jload(evdir(x) / f"full_vsel_t{t}_alt" / "summary_all.json")
        scan = jload(scan_dir(x) / "summary_all.json"); scr = jload(screen_dir(x, "vb") / "summary_all.json")
        rf = jload(evdir(x) / "retfm_t8" / "summary_all.json")
        cz = {w: {int(r["t"]): r["exploded_frac"] for r in (census_json(x, w) or {}).get("rows", [])} for w in ("vsel", "final")}
        vk = scan.get("vote_at_k", {}) if scan else {}; t1 = scan.get("t1r_at_k", {}) if scan else {}
        mj = scr.get("majority_vote_at_k", {}) if scr else {}; sv = scr.get("vote_at_k", {}) if scr else {}
        EV[x] = dict(vcold=fv["exact_acc"] if fv else None, fcold=ff["exact_acc"] if ff else None, acold=fa["exact_acc"] if fa else None,
                     b1=scan.get("b1_exact") if scan else None, vk=vk, t1=t1, mj=mj, sv=sv, retfm=rf["exact_acc"] if rf else None, cz=cz,
                     scan_cold=scan.get("exact_acc") if scan else None, scan_ema=scan.get("ema") if scan else None,
                     val_wrong=fv.get("valid_wrong_frac") if fv else None, viol=fv.get("mean_violations") if fv else None, givens=fv.get("givens_kept_frac") if fv else None)
        e = EV[x]
        say(f"  {x:3s} | {pp(e['vcold'])} | {pp(e['fcold'])} | {pp(e['acold'])} ({'EMA' if not META[x]['head_ema'] else 'raw'}) | {pp(e['b1'])} | {pp(vk.get('1'))}/{pp(vk.get('8'))}/{pp(vk.get('32'))}/{pp(vk.get('128'))} | "
            f"{pp(t1.get('1'))}/{pp(t1.get('8'))}/{pp(t1.get('32'))}/{pp(t1.get('128'))} | {pp(mj.get('16'))}/{pp(mj.get('64'))}/{pp(mj.get('128'))}/{pp(mj.get('256'))} | {pp(e['retfm'])} | "
            f"{pp(cz['vsel'].get(64))}/{pp(cz['vsel'].get(256))} | {pp(cz['final'].get(64))}/{pp(cz['final'].get(256))}")
    say("  full-test failure texture (vsel grid): arm | valid_wrong frac | mean violations | givens kept")
    for x in ARMS:
        e = EV[x]; say(f"  {x:3s} | {f(e['val_wrong'],7,4)} | {f(e['viol'],6,2)} | {f(e['givens'],6,4)}")
    say("\n  X0 / X0n depth rows (EMA headline): D16 vsel | D16 final | D16 alt(raw) | D64 vsel")
    for x in FIELD:
        d64 = jload(evdir(x) / "full_vsel_t64" / "summary_all.json")
        say(f"  {x:3s} | {pp(EV[x]['vcold'])} | {pp(EV[x]['fcold'])} | {pp(EV[x]['acold'])} | {pp(d64['exact_acc'] if d64 else None)}")
    say("\n  screen curves (strat-512 k256 at the headline depth/weights; two-stage tags: sAend = stage-A final, sB005000/sB015000 = stage-B +5k/+15k):")
    for x in ARMS:
        rows = []
        for p in sorted(RUNS.glob(f"sxscreen_p{TAG}{x}_*")):
            s = jload(p / "summary_all.json")
            if not s: continue
            tag = p.name.split("_")[-1]; lab, eff = ckpt_step_from_path(s.get("ckpt", ""), x)
            rows.append((tag, lab, s["vote_at_k"].get("256"), s["exact_acc"], s.get("majority_vote_at_k", {}).get("256"), s.get("givens_kept_frac")))
        say(f"  {x:3s}: " + " | ".join(f"{t}@{lab}: v256 {pp(v)} cold {pp(c)} maj256 {pp(m)}{' [givens 0: GARBAGE]' if g == 0.0 else ''}" for t, lab, v, c, m, g in rows))

    # ---- E ----
    say("\n== E. PAIRED CONTRASTS (McNemar exact; identical puzzle sets by idx) ==")
    Z = {}
    for x in ARMS:
        t = META[x]["head_t"]
        Z[x] = dict(scan=recs(scan_dir(x)), full=recs(full_dir(x, "vsel")), full_final=recs(full_dir(x, "final")),
                    full_alt=recs(evdir(x) / f"full_vsel_t{t}_alt"), full64=recs(evdir(x) / "full_vsel_t64"))
    for ref, sc, fu in (("D4", "sxscan_psportBr2bD4", "sxeval_psportBr2bD4/full_t64"), ("C3X", "sxscan_psportBr2bC3X", "sxeval_psportBr2bC3X/full_t64"),
                        ("P3s1", "sxscan_psportC0P3s1", "sxeval_psportC0P3s1/full_t64"), ("P2", "sxscan_psportC0P2", "sxeval_psportC0P2/full_t64"), ("P1", "sxscan_psportC0P1", "sxeval_psportC0P1/full_t64")):
        Z[ref] = dict(scan=recs(RUNS / sc), full=recs(RUNS / fu), full_final=None, full_alt=None, full64=None)
    def getbits(x, kind, stat):
        z = Z.get(x, {}).get(kind)
        if z is None: return None, None
        if stat == "cold": return z["cold_exact"].astype(bool), z["idx"]
        if stat == "vote128": return vote_bits(z, 128), z["idx"]
        if stat == "b1": b = b1_bits(z); return (None, None) if b is None else (b, z["idx"])
        if stat == "t1r128" and "mi_exact_k" in z and "mi_resid_k" in z:
            ex = z["mi_exact_k"].astype(bool); re_ = z["mi_resid_k"].astype(np.float32)
            best = np.argmin(re_[:, :128], axis=1); return ex[np.arange(len(ex)), best], z["idx"]
        return None, None
    def pair(x, y, kx, ky, stat, label=""):
        ax, ix = getbits(x, kx, stat); ay, iy = getbits(y, ky, stat)
        if ax is None or ay is None: say(f"  {x}[{kx}] vs {y}[{ky}] {stat}: data missing"); return
        if len(ix) != len(iy) or not np.array_equal(ix, iy): say(f"  {x}[{kx}] vs {y}[{ky}] {stat}: idx sets DIFFER (n {len(ix)} vs {len(iy)}) — not paired"); return
        oa, ob, p = mcnemar(ax, ay)
        say(f"  {x:4s}[{kx:10s}] vs {y:4s}[{ky:10s}] {stat:7s} n={len(ax):6d}: {pp(ax.mean())} vs {pp(ay.mean())} ({100*(ax.mean()-ay.mean()):+.2f}pp) | only-{x} {oa:6d} only-{y} {ob:6d} | p={p:.2e} | union {pp((ax|ay).mean())} {label}")
    say("  -- the z-norm twin bridge (seed-paired; 20k scan set) --")
    for st in ("cold", "b1", "vote128", "t1r128"):
        pair("B0", "A0", "scan", "scan", st); 
    pair("B1", "A1", "scan", "scan", "cold", "(A1 = 5k amputee: NOT a matched-length pair)"); pair("B1", "A1", "scan", "scan", "b1", "(labeled)")
    pair("B0", "B1", "scan", "scan", "cold", "(z-norm seed pair)"); pair("B0", "B1", "scan", "scan", "b1", "(z-norm seed pair)"); pair("B0", "B1", "scan", "scan", "vote128", "(z-norm seed pair)")
    say("  -- REGIME (same cell + z-norm; ours vs the field's optimizer regime) --")
    for st in ("cold", "b1", "vote128", "t1r128"):
        pair("R0", "B0", "scan", "scan", st); pair("R0", "B1", "scan", "scan", st)
    say("  -- ARCHITECTURE at matched regime (field cell X0 @D64 vs our cell R0; both EMA, field regime) --")
    for st in ("cold", "b1", "vote128", "t1r128"):
        pair("X0", "R0", "scan", "scan", st)
    for st in ("cold", "b1", "vote128"):
        pair("X0", "B0", "scan", "scan", st)
    say("  -- LADDER d96 -> d128 (pilot P3s1 = the d96 record arm; identical 20k set if seeds match) and canvas refs --")
    for x in ("B0", "B1", "R0", "A0"):
        pair(x, "P3s1", "scan", "scan", "cold"); pair(x, "P3s1", "scan", "scan", "b1"); pair(x, "P3s1", "scan", "scan", "vote128")
    for x in ("B0", "R0", "X0"):
        pair(x, "D4", "scan", "scan", "cold"); pair(x, "D4", "scan", "scan", "vote128"); pair(x, "C3X", "scan", "scan", "vote128")
    say("  -- FULL 422,786 pairings (vsel grids) --")
    say("   vsel vs final per arm (drift / memorization read):")
    for x in ARMS:
        if Z[x]["full_final"] is not None and EV[x]["vcold"] != EV[x]["fcold"]: pair(x, x, "full", "full_final", "cold")
        else: say(f"  {x}: vsel == final grid (identical)")
    say("   headline vs alternate weights per arm (EMA lens; vsel grid):")
    for x in ARMS: pair(x, x, "full", "full_alt", "cold", f"(headline {'EMA' if META[x]['head_ema'] else 'raw'} vs alt {'raw' if META[x]['head_ema'] else 'EMA'})")
    say("   X0 depth (D16 headline vs D64) and ACT gain (X0 vs X0n at D16), paired:")
    pair("X0", "X0", "full64", "full", "cold", "(D64 vs D16)"); pair("X0n", "X0n", "full64", "full", "cold", "(D64 vs D16)"); pair("X0", "X0n", "full", "full", "cold", "(ACT gain @D16)"); pair("X0", "X0n", "full64", "full64", "cold", "(ACT gain @D64)")
    say("   cross-arm (vsel grids, t64):")
    pair("B0", "A0", "full", "full", "cold"); pair("B1", "B0", "full", "full", "cold"); pair("R0", "B0", "full", "full", "cold"); pair("R0", "B1", "full", "full", "cold"); pair("R0", "A0", "full", "full", "cold")
    pair("X0", "R0", "full64", "full", "cold", "(X0 @D64 vs R0)"); pair("X0", "B0", "full64", "full", "cold", "(X0 @D64 vs B0)")
    for x in ("B0", "B1", "R0"):
        pair(x, "D4", "full", "full", "cold"); pair(x, "P3s1", "full", "full", "cold")
    pair("X0", "D4", "full64", "full", "cold"); pair("X0", "P3s1", "full64", "full", "cold")
    say("  -- portfolio (labeled facts): unions on the 20k set, verified vote@128 / cold --")
    for combo in (("B0", "B1"), ("B0", "R0"), ("B0", "B1", "R0"), ("B0", "B1", "R0", "A0"), ("B0", "C3X"), ("R0", "C3X"), ("B0", "B1", "R0", "C3X", "D4"), ("B0", "B1", "R0", "X0"), ("B0", "B1", "R0", "C3X", "D4", "X0")):
        zs = [Z[c]["scan"] for c in combo]
        if any(z is None for z in zs): say(f"  union {'+'.join(combo)}: data missing"); continue
        if any(not np.array_equal(z["idx"], zs[0]["idx"]) for z in zs): say(f"  union {'+'.join(combo)}: idx differ"); continue
        v = np.zeros(len(zs[0]["idx"]), bool); c_ = np.zeros_like(v)
        for z in zs: v |= vote_bits(z, 128); c_ |= z["cold_exact"].astype(bool)
        say(f"  union {'+'.join(combo):22s}: vote@128 {pp(v.mean())} | cold {pp(c_.mean())}")
    say("\n  per-octile (20k scan; rating octiles of the subsample): cold / b1 / verified@128 / t1r@128")
    for x in ["A0", "A1", "B0", "B1", "R0", "X0", "P3s1", "D4", "C3X"]:
        z = Z[x]["scan"]
        if z is None: continue
        rat = z["rating"]; qs = octiles(rat); cells = []; b1 = b1_bits(z); t1, _ = getbits(x, "scan", "t1r128")
        for b in range(8):
            m = (rat >= qs[b]) & (rat < qs[b + 1])
            cells.append(f"{100*z['cold_exact'][m].mean():4.1f}/{('%4.1f' % (100*b1[m].mean())) if b1 is not None else ' -  '}/{100*vote_bits(z,128)[m].mean():4.1f}/{('%4.1f' % (100*t1[m].mean())) if t1 is not None else ' -  '}")
        say(f"  {x:4s} " + " | ".join(cells))
    say("\n  per-octile FULL-test cold (422,786; octiles of the full rating distribution): arm | o1..o8")
    for x in ["A0", "A1", "B0", "B1", "R0", "X0", "X0n", "P3s1", "D4"]:
        z = Z[x]["full64"] if x in FIELD else Z[x]["full"]
        if z is None: continue
        rat = z["rating"]; qs = octiles(rat)
        say(f"  {x:4s}{' (D64)' if x in FIELD else '':6s} " + " | ".join(f"{100*z['cold_exact'][(rat >= qs[b]) & (rat < qs[b+1])].mean():5.1f}" for b in range(8)))

    # ---- F ----
    say("\n== F. FUNNEL MODEL (rho, r) per octile — lens-B form, fit on draws <= 64, checked at 128 ==")
    for x in ["A0", "B0", "B1", "R0", "X0", "P3s1", "C3X"]:
        z = Z[x]["scan"]
        if z is not None: funnel_table(z, x)

    # ---- G ----
    say("\n== G. SELECTOR — Top-1-by-residual (verification-free) vs the free verifier, per arm (20k scan) ==")
    say("  arm | b1 | t1r@1/8/32/128 | verified@1/8/32/128 | t1r@128 / verified@128 | majority@128 (screen)")
    for x in ["A0", "A1", "B0", "B1", "R0", "X0"]:
        e = EV[x]; t1 = e["t1"]; vk = e["vk"]
        ratio = (t1.get("128") / vk.get("128")) if t1.get("128") is not None and vk.get("128") else None
        say(f"  {x:3s} | {pp(e['b1'])} | {pp(t1.get('1'))}/{pp(t1.get('8'))}/{pp(t1.get('32'))}/{pp(t1.get('128'))} | {pp(vk.get('1'))}/{pp(vk.get('8'))}/{pp(vk.get('32'))}/{pp(vk.get('128'))} | {f(ratio,5,3)} | {pp(e['mj'].get('128'))}")

    # ---- H ----
    say("\n== H. EXTRA PREDICTIONS (plan §12.8 items the frozen analyzer does not score; descriptive) ==")
    def cz_of(x, w, t): return EV[x]["cz"].get(w, {}).get(t)
    bcz = [(x, w, t, cz_of(x, w, t)) for x in ("B0", "B1") for w in ("vsel", "final") for t in (64, 256)]
    say(f"  B census 0 % at t=64 and t=256 on both grids: " + ", ".join(f"{x}/{w}/t{t}={pp(v)}" for x, w, t, v in bcz) + f" -> {'HIT' if all(v == 0.0 for *_, v in bcz if v is not None) else 'MISS'}")
    acz = [(x, cz_of(x, "final", 64)) for x in ("A0", "A1")]
    say(f"  >= 1 A final grid > 2 % (t=64): " + ", ".join(f"{x}={pp(v)}" for x, v in acz) + f" -> {'HIT' if any(v is not None and v > .02 for _, v in acz) else 'MISS'}")
    for x in NATIVE:
        s = TS[x]; say(f"  {x} A_total closed <= 500 nats by 10k: @10k {f(s['A_10k'],8,1)} -> {'HIT' if s['A_10k'] is not None and s['A_10k'] <= 500 else 'MISS'} | last {f(s['A_last'],8,1)}")
    for x in ("A0", "A1"):
        s = TS[x]; died = s["first_bad"] or (int(re.search(r'step (\d+)', adm[x].get('NAN_ABORT.txt') or 'step 0').group(1)) or None)
        e = [v for st, v in s["eta"] if died is None or st < died]
        say(f"  {x} death carries eta >= .92 at the last monitor: eta {f(e[-1] if e else None,5,3)} -> {'HIT' if e and e[-1] >= .92 else 'MISS'}")
    x16 = EV["X0"]["vcold"]; xn = EV["X0n"]["vcold"]
    say(f"  X0 - X0n >= +4pp at D16: {pp(x16)} - {pp(xn)} = {('-' if x16 is None or xn is None else f'{100*(x16-xn):+.2f}pp')} -> {'HIT' if x16 is not None and xn is not None and x16 - xn >= .04 else 'MISS'}")
    r = TS["R0"]["val_rise10k"]; say(f"  R0 val still rising at 50k (> +1pp/10k): {('-' if r is None else f'{100*r:+.2f}pp')} -> {'HIT' if r is not None and r > .01 else 'MISS'}")
    say("  epochs seen (samples / 1,001,000 distinct pairs): A/B 80k x 64 = 5.1 | R0 50k x 384 = 19.2 | X0 50k x 768 segments = 38.4 segment-passes (~2.4 full-trajectory passes at 16 segments/row)")

    # ---- I ----
    if a.flux_n > 0:
        say("\n== I. FLUX PROFILE (offline CPU forward, cold trajectory, t=64; native arms) ==")
        try:
            sys.path.insert(0, str(ROOT / "src"))
            from qhrrn2 import sudoku_extreme as SX
            dnpz = SX.load_prepared(ROOT / "data/sudoku_extreme/sudoku_extreme_seed0.npz")
            Q = dnpz["test_q"]; R = dnpz["test_rating"]; sel = SX.stratified_subsample(R, 512, 20260821)
            for x in ("B0", "R0", "A0"):
                fl, fa, eta = flux_profile(x, a.flux_n, sel, Q); tot = fl.sum(1); sh = (fl / np.maximum(tot[:, None], 1e-9)).mean(0)
                say(f"  {x} (n={a.flux_n}): mean I_s {np.round(fl.mean(0),1).tolist()} | total {tot.mean():.0f} nats | shares {np.round(sh,3).tolist()} | A_s {np.round(fa.mean(0),2).tolist()} (total {fa.sum(1).mean():.2f}) | eta {eta:.3f}")
        except Exception as e:
            say(f"  flux profile skipped: {type(e).__name__}: {e}")

    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text("\n".join(L) + "\n")
    say(f"\nartifact -> {OUT}")


if __name__ == "__main__":
    main()
