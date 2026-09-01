#!/usr/bin/env python3
"""CHAMPION TRACK PILOT (sportC0) — physics pass, written AT ANALYSIS TIME
(2026-09-02; descriptive only). The registered verdict authority is
tools/analyze_sportC0.py (untouched, 16/16); nothing here is a decision rule.

Reads the extracted sportC0 corpus in runs/ and answers the mechanism
questions the registration posed but its rules do not read:
  A. ADMISSION + grid census — every arm's ckpt config at artifact level; which
     banked 5k grids are FINITE (the STOPPED-arm screens that ran on post-death
     grids are garbage and are filtered everywhere below).
  B. TRAJECTORIES — train rows (loss/A/I/rule_H/pace) + 2k monitors (val@t64,
     retfm, ret_sched, eta, lambda_J, fp_drift), last-wins dedup; two-stage arms
     read stage A (if present) + stage B with a step offset.
  C. DEATH CENSUS — RI vs no-RI: step of the first non-finite row, what the
     monitors read just before, whether the attention channel was closed (the
     H-48 discriminator: dose-held A_total => RI-intrinsic blowup).
  D. EVAL TABLE — cold (final grid) | retfm | b1 (EqR B=1) | verified@k |
     Top-1-residual@k | unverified majority@k (vb screen) | screen curve.
  E. PAIRED McNemar on the identical 20k subsample (native vs canvas D4/C3X,
     RI vs no-RI) + the full 422,786 pairing for P1 vs D4; per-octile tables.
  F. FUNNEL (rho, r) per octile (lens-B model, reused) on every native scan.
  G. FLUX PROFILE (the registered arity signature): offline CPU forward of
     each clean native ckpt on the probe's strat-512 test set -> per-cut I_s
     shares vs the canvas D4 probe rows' I_s shares.

  .venv/bin/python tools/analyze_sportC0_physics.py [--flux-n 128] [--no-flux]
      -> runs/analysis/sportC0_physics_20260902.txt
"""
from __future__ import annotations
import argparse, json, math, pickle, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "analysis" / "sportC0_physics_20260902.txt"
TAG = "sportC0"
ARMS = ["P1", "P2", "P2s1", "P3", "P3s1", "P5", "P6"]
META = {  # arm -> (RI, T, NI, aug, two_stage, seed)
    "P1":   (False, 12, False, 100, False, 0),
    "P2":   (True, 12, False, 100, False, 0),
    "P2s1": (True, 12, False, 100, False, 1),
    "P3":   (True, 16, False, 100, True, 0),
    "P3s1": (True, 16, False, 100, True, 1),
    "P5":   (True, 12, True, 100, False, 0),
    "P6":   (True, 16, False, 1000, True, 0),
}
S_A = 35000  # two-stage split (35k cosine + 15k floor)
L = []
def say(s=""): L.append(str(s)); print(s)
def f(x, w=6, p=2):
    return " " * (w - 1) + "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:{w}.{p}f}"
def pp(x): return "   -  " if x is None else f"{100*x:6.2f}"


# ---------- A. admission + grid census ----------
def walk_finite(t):
    """True iff every array leaf is finite (numpy walk; no jax needed)."""
    st = [t]
    while st:
        x = st.pop()
        if isinstance(x, dict): st.extend(x.values())
        elif isinstance(x, (list, tuple)): st.extend(x)
        elif hasattr(x, "dtype") and np.issubdtype(np.asarray(x).dtype, np.number):
            if not np.isfinite(np.asarray(x)).all(): return False
    return True

def grid_census(d: Path):
    rows = []
    for p in sorted(d.glob("ckpt_0*.pkl")):
        try:
            c = pickle.load(open(p, "rb"))
            rows.append((int(c["step"]), walk_finite(c["state"]["model"])))
        except Exception as e:
            rows.append((-1, False))
    return rows

def admission(arm):
    d = RUNS / f"pretrain{TAG}_{arm}"
    out = {"dir": d.exists()}
    if not d.exists(): return out
    ck = d / "ckpt_latest.pkl"
    if ck.exists():
        c = pickle.load(open(ck, "rb"))
        cfg = c.get("config", {})
        out["step"] = int(c["step"])
        out["cfg"] = {k: cfg.get(k) for k in ("d", "T", "scales", "canvas", "pool_arity", "mixer_kind",
                                                "sudoku_layout", "attn_max_hw", "d_task", "d_code", "d_b",
                                                "beta_flux", "beta_flux_nl", "fpa_k", "fpa_eps", "ni_sigma", "eq_coupled")}
        n = 0
        st = [c["state"]["model"]]
        while st:
            x = st.pop()
            if isinstance(x, dict): st.extend(x.values())
            elif isinstance(x, (list, tuple)): st.extend(x)
            else: n += int(np.prod(np.shape(x)))
        out["n_bulk"] = n
        out["final_finite"] = walk_finite(c["state"]["model"])
    cj = d / "config.json"
    if cj.exists():
        a = json.loads(cj.read_text()).get("argv", {})
        out["argv"] = {k: a.get(k) for k in ("T", "seed", "ri_p", "ni_sigma", "sudoku_aug", "steps", "lr", "lr_end",
                                              "warmup", "init_from", "width_scale", "fpa_k", "beta_flux_nl", "sudoku_layout")}
    stp = d / "STOPPED.txt"
    out["stopped"] = stp.read_text().strip() if stp.exists() else None
    out["grids"] = grid_census(d)
    da = RUNS / f"pretrain{TAG}_{arm}a"
    out["stageA_dir"] = da.exists()
    return out


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
    """Concatenate stage A (if present, steps as-is) and stage B (offset S_A) for two-stage arms."""
    two = META[arm][4]
    d = RUNS / f"pretrain{TAG}_{arm}"
    tr, mon = load_metrics(d / "metrics.jsonl")
    note = ""
    if two:
        da = RUNS / f"pretrain{TAG}_{arm}a"
        stopped = (d / "STOPPED.txt").exists()
        if stopped:
            note = "died in stage A: D holds the stage-A trajectory (steps as-is)"
        else:
            # D = stage B (steps from 0); stage A trajectory only if the A dir shipped
            trB = {k + S_A: dict(v, step=k + S_A) for k, v in tr.items()}
            monB = {k + S_A: dict(v, step=k + S_A) for k, v in mon.items()}
            trA, monA = load_metrics(da / "metrics.jsonl") if da.exists() else ({}, {})
            note = ("stage A trajectory present" if trA else
                    "stage A trajectory NOT shipped (only its final ckpt banked in GCS); stage B offset +35k")
            tr = dict(sorted({**trA, **trB}.items())); mon = dict(sorted({**monA, **monB}.items()))
    return tr, mon, note

def finite(v):
    try: return v is not None and math.isfinite(float(v))
    except Exception: return False

def traj_summary(arm, tr, mon):
    steps = list(tr.keys())
    first_bad = next((s for s in steps if not finite(tr[s].get("loss"))), None)
    last_good = max([s for s in steps if finite(tr[s].get("loss"))], default=None)
    A = {s: tr[s].get("A_total") for s in steps}
    I = {s: tr[s].get("I_total") for s in steps}
    def near(dct, target, before=True):
        ks = [s for s in dct if finite(dct[s]) and (s <= target if before else s >= target)]
        return dct[max(ks)] if ks else None
    A_peak = max([A[s] for s in steps if finite(A[s])], default=None)
    pace = np.median([tr[s]["steps_per_sec"] for s in steps if finite(tr[s].get("steps_per_sec")) and tr[s]["steps_per_sec"] > 0]) if steps else None
    ms = [s for s in mon if s <= (last_good or 10**9)]
    def mseries(key):
        return [(s, mon[s].get(key)) for s in ms if finite(mon[s].get(key))]
    eta = mseries("eta"); lamj = mseries("lam_joint_max"); lamf = mseries("lam_joint_frac_expansive")
    retfm = mseries("ret_final_t8"); rets = mseries("ret_sched_t8"); val = mseries("val_t64"); drift = mseries("fp_drift_mean")
    return dict(
        n_train_rows=len(steps), first_bad=first_bad, last_good=last_good, last_step=max(steps) if steps else None,
        wasted=(max(steps) - first_bad) if first_bad is not None else 0,
        loss_last_good=tr[last_good]["loss"] if last_good else None,
        loss_m2k=near({s: tr[s]["loss"] for s in steps}, (last_good or 0) - 2000),
        A_peak=A_peak, A_2k=near(A, 2000, False), A_10k=near(A, 10000), A_last=near(A, last_good or 0),
        I_2k=near(I, 2000, False), I_10k=near(I, 10000), I_last=near(I, last_good or 0),
        ruleH_last=near({s: tr[s].get("rule_H") for s in steps}, last_good or 0),
        pace=pace, eta=eta, lamj=lamj, lamf=lamf, retfm=retfm, rets=rets, val=val, drift=drift)


# ---------- D/E. evals ----------
def jload(p):
    p = Path(p); return json.loads(p.read_text()) if p.exists() else None
def recs(p):
    p = Path(p)
    q = p / "records_all.npz"
    if not q.exists():
        parts = sorted(p.glob("records_s*.npz"))
        if not parts: return None
        arrs = [dict(np.load(x, allow_pickle=True)) for x in parts]
        keys = [k for k in arrs[0] if all(k in a for a in arrs)]
        z = {k: np.concatenate([a[k] for a in arrs]) for k in keys}
    else:
        z = dict(np.load(q, allow_pickle=True))
    order = np.argsort(z["idx"], kind="stable")
    return {k: v[order] for k, v in z.items()}

def mcnemar(a, b):
    """a, b: bool arrays aligned. Returns (only_a, only_b, two-sided exact-binomial p)."""
    oa = int(np.sum(a & ~b)); ob = int(np.sum(~a & b))
    n = oa + ob
    if n == 0: return oa, ob, 1.0
    from scipy import stats
    p = float(min(1.0, 2 * stats.binom.cdf(min(oa, ob), n, 0.5)))
    return oa, ob, p

def vote_bits(z, k):
    fh = z["mi_first_hit"]
    return z["cold_exact"].astype(bool) | ((fh >= 0) & (fh < k))


# ---------- F. funnel (rho, r) — the lens-B model, reused ----------
def fit_rho_r(fh, k_fit=64):
    """MLE of rho (reachable fraction) x r (per-draw rate) on first-hit draws <= k_fit
    (later hits censored at k_fit). Grid search (the lens-B implementation's form)."""
    fh = np.asarray(fh)
    hit = (fh >= 0) & (fh < k_fit)
    t = fh[hit]; n_c = int(np.sum(~hit)); n = len(fh)
    best = (-np.inf, None, None)
    for rho in np.linspace(0.02, 1.0, 50):
        for r in np.geomspace(1e-3, 0.9, 60):
            ll = np.sum(np.log(rho) + t * np.log1p(-r) + np.log(r)) if len(t) else 0.0
            ll += n_c * np.log(max(1 - rho + rho * (1 - r) ** k_fit, 1e-300))
            if ll > best[0]: best = (ll, rho, r)
    return best[1], best[2]

def funnel_table(z, label):
    rat = z["rating"]; fh = z["mi_first_hit"]; cold = z["cold_exact"].astype(bool)
    qs = np.quantile(rat, np.linspace(0, 1, 9)); qs[-1] += 1
    say(f"  {label}: octile | n | cold | draw-hit@128 | rho | r | pred@128 (rho*(1-(1-r)^128)) | actual draw@128")
    for b in range(8):
        m = (rat >= qs[b]) & (rat < qs[b + 1])
        if m.sum() == 0: continue
        rho, r = fit_rho_r(fh[m])
        pred = rho * (1 - (1 - r) ** 128) if rho is not None else None
        act = float(np.mean((fh[m] >= 0) & (fh[m] < 128)))
        say(f"    {b} [{qs[b]:.0f},{qs[b+1]:.0f}) n={m.sum():5d} cold {pp(cold[m].mean())} hit {pp(act)} rho {f(rho,5,2)} r {f(r,6,3)} pred {pp(pred)} act {pp(act)}")


# ---------- G. flux profile (offline forward) ----------
def flux_profile(arm, n_puz, sel_idx, Q):
    """Per-cut I_s and A_s at the final cold step (t=64) on n_puz puzzles of the
    probe's strat-512 set. CPU jax; d96 native is small."""
    import jax, jax.numpy as jnp
    sys.path.insert(0, str(ROOT / "src"))
    from qhrrn2 import episodic as E, model as M
    from qhrrn2.config import Config
    d = RUNS / f"pretrain{TAG}_{arm}"
    saved = E.load_ckpt(d / "ckpt_latest.pkl")
    defaults = Config()
    cfg = Config(**{k: type(getattr(defaults, k))(v) for k, v in saved["config"].items()})
    params = saved["state"]["model"]; tv = jnp.asarray(saved["state"]["table"][0])
    eta = float(cfg.eta_floor + (1 - cfg.eta_floor) * jax.nn.sigmoid(params["eq"]["eta"]))
    eta_z = float(jax.nn.sigmoid(params["eq"]["eta_z"]))
    x = jnp.asarray(Q[sel_idx[:n_puz]].astype(np.int32))
    T = cfg.T

    def one(xc):
        y = jax.nn.one_hot(jnp.full(xc.shape, 10, jnp.int32), M.VOCAB).transpose(2, 0, 1)
        z = None; fl = fa = None
        for t in range(64):
            tn = min(t, T - 1) / max(T - 1, 1)
            out = M.forward_fields(params, cfg, M.build_fields_soft(xc, y), t_norm=tn, tau=1.0, rng=None,
                                   task_vec=tv, z_in=z)
            z = out.z_fine if z is None else z + eta_z * (out.z_fine - z)
            p = jax.nn.softmax(out.logits, axis=-1).transpose(2, 0, 1)
            y = y + eta * (p - y)
            fl, fa = out.flux, out.flux_attn
        return fl, fa
    fl, fa = jax.jit(jax.vmap(one))(x)
    fl = np.asarray(fl); fa = np.asarray(fa)
    return fl, fa, eta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flux-n", type=int, default=128)
    ap.add_argument("--no-flux", action="store_true")
    a = ap.parse_args()
    L.clear()
    say("=" * 120)
    say("CHAMPION TRACK PILOT (sportC0) — PHYSICS PASS 2026-09-02 (analysis-time, descriptive; verdict authority = analyze_sportC0.py)")
    say("=" * 120)

    # ---- A ----
    say("\n== A. ADMISSION + BANKED-GRID CENSUS (artifact level) ==")
    adm = {a_: admission(a_) for a_ in ARMS}
    finite_steps = {}
    for a_ in ARMS:
        o = adm[a_]
        if not o.get("dir"): say(f"  {a_}: run dir MISSING"); continue
        c = o.get("cfg", {}); v = o.get("argv", {})
        say(f"  {a_:4s} step {o.get('step')} n_bulk {o.get('n_bulk')} final_finite {o.get('final_finite')} | cfg d{c.get('d')} T{c.get('T')} scales{c.get('scales')} canvas{c.get('canvas')} "
            f"arity{c.get('pool_arity')} mixer {c.get('mixer_kind')} layout {c.get('sudoku_layout')} attn_hw {c.get('attn_max_hw')} d_task {c.get('d_task')} "
            f"b_nl {c.get('beta_flux_nl')} b_flux {c.get('beta_flux')} fpa_k {c.get('fpa_k')} ni {c.get('ni_sigma')}")
        say(f"        argv: T {v.get('T')} seed {v.get('seed')} ri_p {v.get('ri_p')} ni {v.get('ni_sigma')} aug {v.get('sudoku_aug')} steps {v.get('steps')} lr {v.get('lr')}->{v.get('lr_end')} warmup {v.get('warmup')} ws {v.get('width_scale')} init_from {bool(v.get('init_from'))}")
        say(f"        STOPPED: {o.get('stopped')}")
        g = o.get("grids", [])
        fin = [s for s, ok in g if ok]; bad = [s for s, ok in g if not ok]
        finite_steps[a_] = set(fin)
        say(f"        banked grids finite: {fin} | NON-FINITE: {bad} | stageA dir shipped: {o.get('stageA_dir')}")

    # ---- B/C ----
    say("\n== B. TRAJECTORIES (last-wins dedup; two-stage arms = stage A [if shipped] + stage B offset +35k) ==")
    TS = {}
    for a_ in ARMS:
        tr, mon, note = arm_traj(a_)
        TS[a_] = traj_summary(a_, tr, mon)
        s = TS[a_]
        say(f"  {a_:4s} rows {s['n_train_rows']:4d} last {s['last_step']} | first NON-FINITE {s['first_bad']} | last finite {s['last_good']} | steps trained past death {s['wasted']} | pace {f(s['pace'],5,2)} it/s | {note}")
        say(f"        loss last-finite {f(s['loss_last_good'],7,4)} (2k before: {f(s['loss_m2k'],7,4)}) | A_total peak {f(s['A_peak'],10,1)} @2k {f(s['A_2k'],9,1)} @10k {f(s['A_10k'],8,1)} last {f(s['A_last'],8,1)} | I_total @2k {f(s['I_2k'],9,1)} @10k {f(s['I_10k'],8,1)} last {f(s['I_last'],8,1)} | rule_H last {f(s['ruleH_last'],6,3)}")
        def ser(name, xs, p=3):
            if not xs: return f"{name}: -"
            return f"{name}: " + " ".join(f"{st//1000}k:{v:.{p}f}" for st, v in xs)
        say("        " + ser("eta", s["eta"]))
        say("        " + ser("lamJ_max", s["lamj"], 2))
        say("        " + ser("retfm", s["retfm"], 2))
        say("        " + ser("ret_sched", s["rets"], 2))
        say("        " + ser("val@t64", s["val"], 3))
        say("        " + ser("fp_drift", s["drift"], 4))

    say("\n== C. DEATH CENSUS — RI vs no-RI (the registration's RI x dose question) ==")
    say("  NOTE: the one-shot amputation TRUNCATED each STOPPED arm's metrics.jsonl to <= its last FINITE banked grid, so the exact")
    say("  NaN step is not on disk; the death is bounded to (last finite grid, first non-finite grid]. The trainer kept running on")
    say("  non-finite parameters until nan_check caught it at the end (the banked non-finite grids prove how far).")
    say("  arm | RI | T | NI | aug | death interval | last monitor before death: eta / lamJ_max / retfm / ret_sched | A_total last-finite | I_total last-finite | loss last-finite")
    for a_ in ARMS:
        s = TS[a_]; ri, T, ni, aug, two, seed = META[a_]
        died = s["first_bad"]
        g = adm[a_].get("grids", [])
        fin = [st for st, ok in g if ok]; bad = [st for st, ok in g if not ok]
        if adm[a_].get("stopped"):
            lo = max(fin) if fin else None; hi = min([b for b in bad if lo is None or b > lo], default=None)
            interval = f"({lo},{hi}]" if hi else f"(>{lo})"
            ran_to = max(bad) if bad else None
            interval += f" ran>={ran_to}" if ran_to else ""
            died = died or (hi if hi else None)
        else:
            interval = "clean"
        def last(xs):
            xs = [x for x in xs if died is None or x[0] < died]
            return xs[-1][1] if xs else None
        say(f"  {a_:4s} | {'RI ' if ri else 'no '} | T{T} | {'NI ' if ni else '   '} | {aug:4d} | {interval:>24s} | "
            f"eta {f(last(s['eta']),5,3)} lamJ {f(last(s['lamj']),5,2)} retfm {f(last(s['retfm']),4,2)} ret_s {f(last(s['rets']),4,2)} | "
            f"A {f(s['A_last'],9,1)} | I {f(s["I_last"],9,1)} | loss {f(s['loss_last_good'],6,3)}")
    ri_eta = [TS[a_]["eta"][-1][1] for a_ in ARMS if META[a_][0] and TS[a_]["eta"]]
    no_eta = [TS[a_]["eta"][-1][1] for a_ in ARMS if not META[a_][0] and TS[a_]["eta"]]
    say(f"  eta at last finite monitor — RI arms: {[round(x,3) for x in ri_eta]} | no-RI: {[round(x,3) for x in no_eta]}")
    say("  READ: A_total at every death is the dose's CLOSED value (see B) => the deaths are not free-channel (H-48) events; RI-intrinsic dynamics named in the report.")

    # ---- D ----
    say("\n== D. EVAL TABLE (final grid unless noted; STOPPED arms = amputated finals, labeled) ==")
    say("  arm  | cold(full) | retfm | vb-step | b1 (B=1) | verified@1/8/32/128 | t1r@1/8/32/128 | majority@16/64/128/256 (vb screen) | screen-vb v128/v256")
    refs = {"D4": ("sxeval_psportBr2bD4/full_t64", "sxscan_psportBr2bD4", "sxscreen_psportBr2bD4_vb"),
            "C3X": ("sxeval_psportBr2bC3X/full_t64", "sxscan_psportBr2bC3X", "sxscreen_psportBr2bC3X_vb")}
    EV = {}
    for a_ in ARMS + ["D4", "C3X"]:
        if a_ in refs:
            fe, sc, vb = refs[a_]
            full = jload(RUNS / fe / "summary_all.json"); scan = jload(RUNS / sc / "summary_all.json"); scr = jload(RUNS / vb / "summary_all.json")
            rf = None
        else:
            full = jload(RUNS / f"sxeval_p{TAG}{a_}" / "full_t64" / "summary_all.json")
            scan = jload(RUNS / f"sxscan_p{TAG}{a_}" / "summary_all.json")
            scr = jload(RUNS / f"sxscreen_p{TAG}{a_}_vb" / "summary_all.json")
            rfj = jload(RUNS / f"sxeval_p{TAG}{a_}" / "retfm_t8" / "summary_all.json"); rf = rfj["exact_acc"] if rfj else None
        cold = full["exact_acc"] if full else None
        vbstep = scr["ckpt"].split("/")[-1].replace("ckpt_", "").replace(".pkl", "") if scr else "-"
        b1 = scan.get("b1_exact") if scan else None
        vk = scan.get("vote_at_k", {}) if scan else {}
        t1 = scan.get("t1r_at_k", {}) if scan else {}
        mj = scr.get("majority_vote_at_k", {}) if scr else {}
        sv = scr.get("vote_at_k", {}) if scr else {}
        EV[a_] = dict(cold=cold, retfm=rf, b1=b1, vk=vk, t1=t1, mj=mj, sv=sv, vb=vbstep)
        say(f"  {a_:4s} | {pp(cold)} | {pp(rf)} | {vbstep:>7s} | {pp(b1)} | {pp(vk.get('1'))}/{pp(vk.get('8'))}/{pp(vk.get('32'))}/{pp(vk.get('128'))} | "
            f"{pp(t1.get('1'))}/{pp(t1.get('8'))}/{pp(t1.get('32'))}/{pp(t1.get('128'))} | {pp(mj.get('16'))}/{pp(mj.get('64'))}/{pp(mj.get('128'))}/{pp(mj.get('256'))} | {pp(sv.get('128'))}/{pp(sv.get('256'))}")
    say("  NOTE b1 = single random-init draw (EqR Table-4 statistic); vote@1 = cold UNION draw 1 (a different object); t1r = Top-1-by-L=3-residual over k draws, unverified.")

    say("\n  screen curves across banked ckpts (strat-512 k256 v256; garbage screens on post-death grids FILTERED):")
    for a_ in ARMS:
        rows = []
        for p in sorted(RUNS.glob(f"sxscreen_p{TAG}{a_}_*")):
            s = jload(p / "summary_all.json")
            if not s: continue
            tag = p.name.split("_")[-1]
            ck = s["ckpt"].split("/")[-1]
            step = None
            if ck.startswith("ckpt_0"): step = int(ck[5:11])
            two = META[a_][4]
            died = TS[a_]["first_bad"]
            # filter: any screen whose ckpt step lies beyond the arm's first non-finite row is garbage
            eff = step
            if two and tag in ("sB05", "vb") and step is not None and not adm[a_].get("stopped"):
                eff = step + S_A
            garbage = (died is not None and eff is not None and eff >= died) or (s.get("givens_kept_frac") == 0.0)
            rows.append((tag, ck, s["vote_at_k"].get("256"), s["exact_acc"], s.get("majority_vote_at_k", {}).get("256"), garbage))
        say(f"  {a_:4s}: " + " | ".join(f"{t}@{ck[5:11] if ck.startswith('ckpt_0') else ck}: v256 {pp(v)} cold {pp(c)} maj256 {pp(m)}{' [GARBAGE post-death grid, IGNORED]' if g else ''}" for t, ck, v, c, m, g in rows))

    # ---- E ----
    say("\n== E. PAIRED CONTRASTS (McNemar exact; identical puzzle sets by idx) ==")
    Z = {}
    for a_ in ARMS:
        Z[a_] = dict(scan=recs(RUNS / f"sxscan_p{TAG}{a_}"), full=recs(RUNS / f"sxeval_p{TAG}{a_}" / "full_t64"))
    Z["D4"] = dict(scan=recs(RUNS / "sxscan_psportBr2bD4"), full=recs(RUNS / "sxeval_psportBr2bD4" / "full_t64"))
    Z["C3X"] = dict(scan=recs(RUNS / "sxscan_psportBr2bC3X"), full=recs(RUNS / "sxeval_psportBr2bC3X" / "full_t64"))
    def pair(x, y, kind, stat):
        zx, zy = Z[x][kind], Z[y][kind]
        if zx is None or zy is None: say(f"  {x} vs {y} [{kind}/{stat}]: data missing"); return
        if not np.array_equal(zx["idx"], zy["idx"]): say(f"  {x} vs {y} [{kind}]: idx sets DIFFER — not paired"); return
        if stat == "cold": ax, ay = zx["cold_exact"].astype(bool), zy["cold_exact"].astype(bool)
        elif stat == "vote128": ax, ay = vote_bits(zx, 128), vote_bits(zy, 128)
        elif stat == "b1":
            if "mi_exact_k" not in zx or "mi_exact_k" not in zy: say(f"  {x} vs {y} [b1]: per-draw bits missing on one side"); return
            ax, ay = zx["mi_exact_k"][:, 0].astype(bool), zy["mi_exact_k"][:, 0].astype(bool)
        else: return
        oa, ob, p = mcnemar(ax, ay)
        say(f"  {x:4s} vs {y:4s} [{kind:4s} n={len(ax):6d} {stat:7s}] {x} {pp(ax.mean())} vs {y} {pp(ay.mean())} | only-{x} {oa:6d} only-{y} {ob:6d} | p={p:.2e} | union {pp((ax|ay).mean())}")
    say("  -- arity: native P1 vs canvas D4 (the D4-strategy verbatim) and vs the breadth owner C3X --")
    pair("P1", "D4", "full", "cold"); pair("P1", "D4", "scan", "cold"); pair("P1", "D4", "scan", "vote128")
    pair("P1", "C3X", "scan", "vote128"); pair("P1", "C3X", "scan", "cold")
    say("  -- RI lever: P2 (RI s0) vs P1; P3s1 (T16 RI two-phase, clean) vs P1; P2 vs P3s1 --")
    pair("P2", "P1", "scan", "cold"); pair("P2", "P1", "scan", "b1"); pair("P2", "P1", "scan", "vote128"); pair("P2", "P1", "full", "cold")
    pair("P3s1", "P1", "scan", "cold"); pair("P3s1", "P1", "scan", "b1"); pair("P3s1", "P1", "scan", "vote128"); pair("P3s1", "P1", "full", "cold")
    pair("P2", "P3s1", "scan", "b1"); pair("P2", "P3s1", "scan", "vote128")
    pair("P2", "D4", "scan", "cold"); pair("P2", "D4", "scan", "vote128"); pair("P2", "C3X", "scan", "vote128")
    say("  -- stopped arms (amputated finals; labeled) vs P1 --")
    for a_ in ("P2s1", "P3", "P5", "P6"):
        pair(a_, "P1", "full", "cold")
        if Z[a_]["scan"] is not None: pair(a_, "P1", "scan", "vote128")
    say("  -- portfolio (labeled facts): unions on the 20k set --")
    for combo in (("P1", "P2"), ("P1", "P3s1"), ("P2", "P3s1"), ("P1", "D4"), ("P2", "C3X"), ("P1", "P2", "P3s1")):
        zs = [Z[c]["scan"] for c in combo]
        if any(z is None for z in zs): continue
        v = np.zeros(len(zs[0]["idx"]), bool); c_ = np.zeros_like(v)
        for z in zs: v |= vote_bits(z, 128); c_ |= z["cold_exact"].astype(bool)
        say(f"  union {'+'.join(combo):14s}: vote@128 {pp(v.mean())} | cold {pp(c_.mean())}")

    say("\n  per-octile (20k scan; rating octiles of the subsample): cold / verified@128 / b1")
    for a_ in ["P1", "P2", "P3s1", "P2s1", "P5", "D4", "C3X"]:
        z = Z[a_]["scan"]
        if z is None: continue
        rat = z["rating"]; qs = np.quantile(rat, np.linspace(0, 1, 9)); qs[-1] += 1
        cells = []
        for b in range(8):
            m = (rat >= qs[b]) & (rat < qs[b + 1])
            b1 = z["mi_exact_k"][m, 0].mean() if "mi_exact_k" in z else None
            cells.append(f"{100*z['cold_exact'][m].mean():4.1f}/{100*vote_bits(z,128)[m].mean():4.1f}/{('%4.1f' % (100*b1)) if b1 is not None else '  - '}")
        say(f"  {a_:4s} " + " | ".join(cells))

    # ---- F ----
    say("\n== F. FUNNEL MODEL (rho, r) per octile — lens-B form, fit on draws <= 64, held-out check at 128 ==")
    for a_ in ["P1", "P2", "P3s1", "D4", "C3X"]:
        z = Z[a_]["scan"]
        if z is not None: funnel_table(z, a_)

    # ---- G ----
    if not a.no_flux:
        say("\n== G. FLUX PROFILE — the registered arity signature (offline CPU forward, cold trajectory, t=64) ==")
        try:
            sys.path.insert(0, str(ROOT / "src"))
            from qhrrn2 import sudoku_extreme as SX
            dnpz = SX.load_prepared(ROOT / "data/sudoku_extreme/sudoku_extreme_seed0.npz")
            Q = dnpz["test_q"]; R = dnpz["test_rating"]
            sel = SX.stratified_subsample(R, 512, 20260821)
            # canvas reference: D4 probe rows (I_s lists over the 5 dyadic cuts)
            for ref in ("D4", "D3"):
                pth = RUNS / f"sudprobe_psportBr2b{ref}" / "results.jsonl"
                if pth.exists():
                    Is = np.array([json.loads(l)["I_s"] for l in pth.read_text().splitlines() if l.strip()])
                    tot = Is.sum(1, keepdims=True); sh = (Is / np.maximum(tot, 1e-9)).mean(0)
                    say(f"  canvas {ref} probe (n={len(Is)}, 5 dyadic cuts 32->16->8->4->2->1): mean I_s {np.round(Is.mean(0),1).tolist()} | total {Is.sum(1).mean():.0f} nats | shares {np.round(sh,3).tolist()} | UV(s0) share {sh[0]:.3f}, s1 share {sh[1]:.3f}")
            for a_ in ["P1", "P2", "P3s1"]:
                fl, fa, eta = flux_profile(a_, a.flux_n, sel, Q)
                tot = fl.sum(1); sh = (fl / np.maximum(tot[:, None], 1e-9)).mean(0)
                say(f"  native {a_} (n={a.flux_n}, cuts 9->3 [s0, box-forming] and 3->1 [s1]): mean I_s {np.round(fl.mean(0),1).tolist()} | total {tot.mean():.0f} nats | shares {np.round(sh,3).tolist()} | A_s {np.round(fa.mean(0),2).tolist()} (total {fa.sum(1).mean():.2f}) | eta {eta:.3f}")
            say("  READ: the registration's signature = 'flux mass migrates s0 -> s1 (box scale) under alignment; s1 share >= 2x the canvas s1 share'. Native has TWO cuts; the comparable object is the share above the finest cut.")
        except Exception as e:
            say(f"  flux profile skipped: {type(e).__name__}: {e}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n")
    say(f"\nartifact -> {OUT}")


if __name__ == "__main__":
    main()
