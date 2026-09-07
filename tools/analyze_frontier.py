#!/usr/bin/env python3
"""FRONTIER POD RUN — the DESCRIPTIVE analyzer (registration: Documentation/Plan_2026-09-07_Frontier_Registration.md — §1 models,
§3 the job list / output directories of tools/chain_frontier.sh, §4 predictions P1-P27). DESCRIPTIVE, NO RULES: it reads the
evaluator's per-puzzle records (tools/eval_sudoku_extreme.py: records_all.npz + summary_all.json) for the ported public checkpoints
(trmpub, eqr, cgar, trmpub60k) and the DEC pass-two rows (A3 A7 A5 A4 A8), prints the field study's tables at the full protocol
and the prediction scoreboard; nothing is decided, no champion rule, no letters.
  A reproduction (cold D16/D64 FULL + the trmpub60k / EqR raw / noise / train-1k rows; the depth ladder from the per-step bits)
  B decoder lens R1/R2/R3 via tools/suite_records.analyze_records on the FULL records (class per analyze_finalA's R-A-4 rule)
  C selector law on the k-draw scans   D screens (STRAT512 x k256 + unverified majority)   E the halting head (R10, D16 fulls)
  F init radius   G prefix zeroed   H numerics (fp32 vs bf16) + the cross-route row vs the Mac torch rows (runs/field_ckpts/out)
  I census + calibration   J the DEC pass two (5k x k32 scans, D128 rows, RECIPE-DEC)   K pairwise overlaps (McNemar, Jaccard) vs X0
  L the prediction scoreboard: predicted band | measured | HIT / MISS-ABOVE / MISS-BELOW / NO-DATA per clause.
Every row names the model, the set (FULL / SCAN20k / SUB5k / STRAT512 / TRAIN-1k), the protocol (cold|k / D / z0 mode / noise /
precision / EMA|raw) and n. Absent artifacts are ABSENT rows (the pod may still be producing files, a DEC scan may have deadlocked);
a directory with shard records but no merge (records_s*.npz / partial_s*.npz) is read as a PARTIAL row. Old records without the
exact_by_step / q_by_step columns get "-" in the depth and halting cells.

  PYTHONPATH=src JAX_PLATFORMS=cpu .venv/bin/python tools/analyze_frontier.py [--date YYYYMMDD] [--runs runs]
      [--tags trmpub,eqr,cgar,trmpub60k] [--dec-arms A3,A7,A5,A4,A8] [--x0 X0:psportC1]
  --tags entries are tag[:prefix[:mac[:alias]]]: prefix = the directory prefix (default pfrontier: runs/sxeval_pfrontier<tag>/...,
  runs/sxscan_pfrontier<tag>, ...); mac = the Mac field-study cold run(s) for the cross-route row ('+'-separated; default by tag);
  alias = the registration model the rows are SCORED as (TEST USE ONLY: A0:pfinalA:trm:trmpub reads Night A's arm A0 as a model).
  -> runs/analysis/frontier_<date>.{txt,json}
"""
from __future__ import annotations
import argparse, json, math, re, sys, time
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tools"))
from suite_records import analyze_records, jload, RBANDS          # the suite's own record functions (R1-R7), reused by import

N_TEST = 422786; K_CURVE = (1, 2, 4, 8, 16, 32, 64, 128, 256); LADDER = (1, 2, 4, 8, 16, 32, 64, 128, 256); EPS = ("0.03", "0.1", "0.3", "1", "3")
MAC_DEFAULT = {"trmpub": "trm_cold_scan20k_D64+trm_cold_scan20k_D16_bf16", "cgar": "trmc_cold_scan20k_D64", "eqr": "eqr_cold_scan20k_D64_n0+eqr_cold_scan20k_D64_n05"}
RUNS = ROOT / "runs"; L = []; J = {}; M = {}; DEC = {}; CACHE = {}; SB = []; G = R = None

class Spec:
    def __init__(self, s):
        f = (s.split(":") + ["", "", ""])[:4]; self.tag = f[0]; self.prefix = f[1] or "pfrontier"
        self.mac = [m for m in (f[2] or MAC_DEFAULT.get(self.tag, "")).split("+") if m]; self.alias = f[3] or self.tag
        self.riders = self.tag != "trmpub60k"                      # trmpub60k runs only the D16 full + the 20k D64 (§3 item 6)
    def ev(self, sub): return RUNS / f"sxeval_{self.prefix}{self.tag}" / sub
    def d(self, kind, suffix=""): return RUNS / f"sx{kind}_{self.prefix}{self.tag}{suffix}"
    def label(self): return self.tag + (f">{self.alias}" if self.alias != self.tag else "")

def say(s=""): L.append(str(s)); print(s, flush=True)
def _bad(x): return x is None or (isinstance(x, float) and not math.isfinite(x))
def pc(x, p=2): return "-" if _bad(x) else f"{100 * x:.{p}f}"
def fl(x, p=3): return "-" if _bad(x) else f"{x:.{p}f}"
def sg(x, p=2): return "-" if _bad(x) else f"{100 * x:+.{p}f}"
def fe1(x): return "-" if _bad(x) else f"{x + 1:.0f}"           # first_exact is 0-based in the records; printed 1-based

# ---------------------------------------------------------------- loading (records_all | shards | absent) ----------------------------------------------------------------
def fingerprint_summary(fp):
    try: f = json.loads(fp)
    except Exception: return {}
    return dict(t_total=f.get("t"), k_init=f.get("k"), ema=f.get("ema", False), z0_mode=f.get("z0_mode"), z0_eps=f.get("z0_eps"), z0_device=f.get("z0_device", False),
                seg_noise_beta=f.get("seg_noise_beta"), zero_prefix=f.get("zero_prefix", False), from_fingerprint=True)
def load_run(d):
    """One evaluator output directory -> dict(status OK|PARTIAL|ABSENT, z=records, s=summary, n, T, K, note). PARTIAL = no records_all.npz yet:
    the finished shards' records_s*.npz plus the in-flight partial_s*.npz (300 s banks) concatenated; the protocol then comes from the fingerprint."""
    d = Path(d); key = str(d)
    if key in CACHE: return CACHE[key]
    r = dict(status="ABSENT", z=None, s={}, n=0, T=0, K=0, note="", dir=d.name, path=key)
    if d.exists():
        s = jload(d / "summary_all.json") or {}; note = ""
        if not s:                                                   # no merge yet: a finished shard's summary carries the protocol
            for p in sorted(d.glob("summary_s*.json")):
                s = jload(p) or {}
                if s: s = dict(s, shard="unmerged", n=None); break
        if (d / "records_all.npz").exists():
            try: r.update(status="OK", z=dict(np.load(d / "records_all.npz", allow_pickle=True)))
            except Exception as e: note = f"records_all unreadable ({type(e).__name__})"
        if r["z"] is None:
            files = {re.search(r"_s(\d+)", p.stem).group(1): p for p in d.glob("records_s*.npz")}
            for p in sorted(d.glob("partial_s*.npz")): files.setdefault(re.search(r"_s(\d+)", p.stem).group(1), p)
            parts = []
            for p in files.values():
                try:
                    pz = dict(np.load(p, allow_pickle=True))
                    if not s and "_fingerprint" in pz: s = fingerprint_summary(str(pz["_fingerprint"]))
                    parts.append({k: v for k, v in pz.items() if not k.startswith("_")})
                except Exception as e: note += f" {p.name} unreadable ({type(e).__name__})"
            if parts:
                keys = [k for k in parts[0] if all(k in q for q in parts)]
                r.update(status="PARTIAL", z={k: np.concatenate([q[k] for q in parts]) for k in keys}, note=(f"{len(parts)} shard files" + note).strip())
            elif not note: note = "no records yet" + (f" ({len(list(d.glob('shard_*.log')))} shard logs)" if list(d.glob("shard_*.log")) else "")
        r["s"] = s; r["note"] = r["note"] or note
    else: r["note"] = "no directory"
    if r["z"] is not None:
        z = r["z"]; r["n"] = int(len(z["idx"])); m = re.search(r"_t(\d+)", d.name)
        r["T"] = int(r["s"].get("t_total") or (m.group(1) if m else 0) or 0)
        r["K"] = int(z["mi_exact_k"].shape[1]) if "mi_exact_k" in z and z["mi_exact_k"].ndim == 2 else int(r["s"].get("k_init") or 0)
    CACHE[key] = r; return r
def load_mac(name):
    """A Mac field-study cold run (tools/field_ckpts): records exact_by_step (T, n) bool, q_by_step (T, n) q_halt, givens_kept bool."""
    d = RUNS / "field_ckpts" / "out" / name; key = "mac:" + name
    if key in CACHE: return CACHE[key]
    r = dict(status="ABSENT", z=None, s={}, n=0, T=0, K=0, note="no directory" if not d.exists() else "no records", dir=name, path=str(d))
    if (d / "records_all.npz").exists() and (d / "summary.json").exists():
        try:
            z = dict(np.load(d / "records_all.npz", allow_pickle=True)); pr = (jload(d / "summary.json") or {}).get("proto", {})
            r.update(status="OK", z=z, n=int(len(z["idx"])), T=int(pr.get("D") or z["exact_by_step"].shape[0]), s=dict(t_total=pr.get("D"), ema=pr.get("ema"), mac=pr), note="")
        except Exception as e: r["note"] = f"unreadable ({type(e).__name__})"
    CACHE[key] = r; return r
def mac_proto(r):
    pr = r["s"].get("mac", {}); return f"torch {pr.get('dtype')} {pr.get('device')} cold D{pr.get('D')} init:{pr.get('init')}" + (f" noise{pr.get('noise'):g}" if pr.get("noise") is not None else "") + (" EMA" if pr.get("ema") else " raw")
def set_label(r):
    s, name = r["s"], r["dir"]
    if "train1k" in name: return "TRAIN-1k"
    if s.get("stratified"): return f"STRAT{s['stratified']}"
    sub = s.get("subsample")
    if sub: return {20000: "SCAN20k", 5000: "SUB5k"}.get(sub, f"SUB{sub // 1000}k" if sub % 1000 == 0 else f"SUB{sub}")
    if s.get("n") == N_TEST or (r["status"] == "OK" and r["n"] == N_TEST): return "FULL"
    if r["status"] == "PARTIAL":
        guess = {"full": "FULL", "sub20k": "SCAN20k", "sub5k": "SUB5k", "initrad": "STRAT512"}.get(name.split("_")[0]) or ("STRAT512" if "sxscreen" in r["path"] else "SCAN20k" if "sxscan" in r["path"] else "?")
        return guess + "(part)"
    return f"n{r['n']}" if r["n"] else "?"
def proto(r):
    """cold|k / D / z0 mode / noise / precision / EMA|raw. Precision is inferred from the directory name (the chain sets JAX_DEFAULT_MATMUL_PRECISION; the summary does not record it)."""
    s, k = r["s"], r["K"] or 0; p = ["cold" if not k else f"k{k}", f"D{r['T'] or '?'}"]
    if k or s.get("z0_mode"): p.append("z0:" + (s.get("z0_mode") or "gauss") + (f"(eps {s['z0_eps']:g})" if s.get("z0_eps") is not None else "") + ("/dev" if s.get("z0_device") else ""))
    if s.get("seg_noise_beta") is not None: p.append(f"noise{s['seg_noise_beta']:g}")
    if s.get("zero_prefix"): p.append("prefix0")
    p.append("fp32" if r["dir"].endswith("fp32") else "bf16"); p.append("EMA" if s.get("ema") else "raw")
    return " ".join(p)
def head(sp, r, set_=None): return f"{sp.label():10s} | {set_ or set_label(r):13s} | {proto(r):36s} | n={r['n']:6d}" + (f" [{r['status']}]" if r["status"] != "OK" else "")
def absent(sp, d, set_="?"): r = load_run(d); say(f"  {sp.label():10s} | {set_:13s} | ABSENT {Path(d).name}" + (f" ({r['note']})" if r["note"] else ""))

# ---------------------------------------------------------------- record helpers ----------------------------------------------------------------
def bits_of(r):
    """exact-by-step as (n, T) bool: ours = packed uint8 (n, ceil(T/8)), little-endian bit t = step t; the Mac's = (T, n) bool. None if absent."""
    z = r["z"]
    if z is None or "exact_by_step" not in z: return None
    e = z["exact_by_step"]
    if e.dtype == bool: return e.T if e.shape[0] != r["n"] else e
    T = r["T"] or 8 * e.shape[1]
    return np.unpackbits(e.astype(np.uint8), axis=1, bitorder="little")[:, :T].astype(bool)
def ladder(r):
    b = bits_of(r)
    if b is None: return None
    out = {str(s): float(b[:, s - 1].mean()) for s in LADDER if s <= b.shape[1]}
    out.update(T=int(b.shape[1]), n=int(len(b)), last=float(b[:, -1].mean()), regress=int(np.sum(b.any(1) & ~b[:, -1]))); out["regress_frac"] = out["regress"] / max(1, len(b)); return out
def rank_auc(score, label):
    label = np.asarray(label, bool); n1, n0 = int(label.sum()), int((~label).sum())
    if not n1 or not n0: return None
    rk = stats.rankdata(np.asarray(score, np.float64)); return float((rk[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
def halting(r):
    """R10 from q_by_step (n, T, 2) = (q_halt, q_continue) per step, the evaluator's own formulas; the summary's values when the column is absent."""
    z, s = r["z"], r["s"]
    if z is None or "q_by_step" not in z or z["q_by_step"].ndim != 3:
        return None if s.get("q_halt_auc_last") is None else dict(auc=s.get("q_halt_auc_last"), margin_auc=s.get("q_halt_margin_auc_last"), halt_frac=s.get("q_halt_frac_last"), precision=s.get("q_halt_precision_last"),
                                                                 recall=s.get("q_halt_recall_last"), first_halt_mean=s.get("q_first_halt_step_mean"), exact_at_first_halt=s.get("exact_at_first_halt"), cold=s.get("exact_acc"), source="summary")
    q = z["q_by_step"].astype(np.float32); cold = z["cold_exact"].astype(bool); ql, qc = q[:, -1, 0], q[:, -1, 1]; halt = ql > qc; n = len(q)
    hs = q[:, :, 0] > q[:, :, 1]; first = np.where(hs.any(1), hs.argmax(1), q.shape[1] - 1)
    out = dict(auc=rank_auc(ql, cold), margin_auc=rank_auc(ql - qc, cold), halt_frac=float(halt.mean()), precision=float(cold[halt].mean()) if halt.any() else None,
               recall=float(halt[cold].mean()) if cold.any() else None, first_halt_mean=float(first.mean() + 1), cold=float(cold.mean()), source="records", n=int(n))
    b = bits_of(r); out["exact_at_first_halt"] = None if b is None else float(b[np.arange(n), np.minimum(first, b.shape[1] - 1)].mean()); return out
def align(za, zb):
    """the common idx of two record sets -> (common, positions in a, positions in b)."""
    return np.intersect1d(za["idx"].astype(np.int64), zb["idx"].astype(np.int64), return_indices=True)
def agree(ea, eb):
    ea, eb = np.asarray(ea, bool), np.asarray(eb, bool)
    return dict(n=int(len(ea)), a=float(ea.mean()), b=float(eb.mean()), delta=float(eb.mean() - ea.mean()), agree=float((ea == eb).mean()), only_a=int((ea & ~eb).sum()), only_b=int((~ea & eb).sum()))
def mcnemar(ea, eb):
    b, c = int((ea & ~eb).sum()), int((~ea & eb).sum()); n = b + c; chi = (abs(b - c) - 1) ** 2 / n if n else 0.0
    return dict(only_a=b, only_b=c, chi2=chi, p=float(stats.chi2.sf(chi, 1)) if n else 1.0, jaccard=float((ea & eb).sum() / max(1, (ea | eb).sum())),
                pA_fail_given_B_fail=float((~ea[~eb]).mean()) if (~eb).any() else None, pB_fail_given_A_fail=float((~eb[~ea]).mean()) if (~ea).any() else None)
def lens(r):
    """suite_records.analyze_records (R1/R2/R3 + R5-R7 when draws exist) + the decoder class (analyze_finalA R-A-4: DECIMATING iff no g50 in 17-35 and yield >= .70;
    SOFT iff g50 >= 24 and yield <= .45; else MIXED). None when the records index another file (the train-1k)."""
    z = r["z"]
    if int(z["idx"].max()) >= N_TEST: return None
    row = analyze_records(z, G, R); g_in = row["g50"] if row["g50_in_range"] else None; y = row["yield"]
    row["cls"] = "NO-DATA" if y is None else "DECIMATING" if (g_in is None and y >= .70) else "SOFT" if (g_in is not None and g_in >= 24 and y <= .45) else "MIXED"
    if r["K"] and "mi_exact_k" in z and "mi_resid_k" in z:
        ex = z["mi_exact_k"][:, :r["K"]].astype(bool); rs = z["mi_resid_k"][:, :r["K"]].astype(np.float64); rs = np.where(np.isfinite(rs), rs, np.inf); ar = np.arange(len(ex))
        row["t1r_curve"] = {str(k): float(ex[ar, np.argmin(rs[:, :k], 1)].mean()) for k in K_CURVE if k <= r["K"]}
    return row

# ---------------------------------------------------------------- the tables ----------------------------------------------------------------
def table_0(specs, arms):
    say("\n== 0. INVENTORY (status of every expected artifact; OK = merged records, PARTIAL = shards in flight, ABSENT) ==")
    inv = {}
    for sp in specs:
        subs = ["full_vsel_t16", "full_vsel_t64"] + (["sub20k_t64"] if not sp.riders else ["sub20k_t128", "sub5k_t256", "sub20k_t16_prefix0", "sub20k_t16_fp32"] + [f"initrad_e{e}" for e in EPS])
        if sp.tag == "eqr" or sp.alias == "eqr": subs += ["full_raw_t16", "sub20k_t16_noise05", "sub20k_t64_noise05", "sub20k_t16_noise001", "train1k_t16"]
        st = {s: load_run(sp.ev(s))["status"] for s in subs}
        if sp.riders:
            st["scan"] = load_run(sp.d("scan"))["status"]; st["screen"] = load_run(sp.d("screen", "_vb"))["status"]
            st["census"] = "OK" if (sp.d("census", "_vsel") / "census.json").exists() else "ABSENT"; st["calib"] = "OK" if (sp.d("calib", "_vsel") / "calib.json").exists() else "ABSENT"
            if sp.tag == "eqr" or sp.alias == "eqr": st["scan_trunc"] = load_run(sp.d("scan", "_trunc"))["status"]; st["scan_noise05"] = load_run(sp.d("scan", "_noise05"))["status"]
        inv[sp.label()] = st; say(f"  {sp.label():10s} [{sp.prefix}] " + " ".join(f"{k}:{v[:2] if v != 'OK' else 'OK'}" for k, v in st.items()))
    for arm in arms:
        st = dict(scan=load_run(RUNS / f"sxscan_pfinalA{arm}")["status"], d128=load_run(RUNS / f"sxeval_pfinalA{arm}" / "sub20k_t128")["status"]); inv["DEC " + arm] = st
        say(f"  DEC {arm:6s} [pfinalA]  " + " ".join(f"{k}:{v[:2] if v != 'OK' else 'OK'}" for k, v in st.items()))
    J["inventory"] = inv

def table_A(specs):
    say("\n== A. REPRODUCTION — cold exact, as released, through OUR evaluator on the ported weights (+ trmpub60k, EqR raw / noise / train-1k rows) ==")
    say("  model | set | protocol | n | cold exact % | first-exact mean (1-based) | valid-wrong % | depth regressions (solved at some step, unsolved at the last)")
    for sp in specs:
        subs = ["full_vsel_t16", "full_vsel_t64"] + (["sub20k_t64"] if not sp.riders else []) + (["full_raw_t16", "sub20k_t16_noise05", "sub20k_t64_noise05", "sub20k_t16_noise001", "train1k_t16"] if sp.tag == "eqr" or sp.alias == "eqr" else [])
        for sub in subs:
            r = load_run(sp.ev(sub))
            if r["z"] is None: absent(sp, sp.ev(sub), "FULL" if sub.startswith("full") else "TRAIN-1k" if "train" in sub else "SCAN20k"); continue
            z = r["z"]; cold = z["cold_exact"].astype(bool); fe = z["first_exact"] if "first_exact" in z else np.full(r["n"], -1); lad = ladder(r)
            row = dict(set=set_label(r), proto=proto(r), n=r["n"], status=r["status"], cold=float(cold.mean()), fe_mean=(float(fe[fe >= 0].mean()) + 1) if (fe >= 0).any() else None,
                       valid_wrong=float(((z["violations"] == 0) & ~cold).mean()) if "violations" in z else None, regress=None if lad is None else lad["regress"], regress_frac=None if lad is None else lad["regress_frac"], port=r["s"].get("port"))
            M[sp.alias][sub] = row
            say(f"  {head(sp, r)} | {pc(row['cold'])} | {fl(row['fe_mean'], 2)} | {pc(row['valid_wrong'], 3)} | " + ("-" if lad is None else f"{lad['regress']} ({pc(lad['regress_frac'], 3)} %)"))
    say("  PROVENANCE (the summary's port block): " + "; ".join(f"{sp.label()}: {json.dumps(M[sp.alias]['full_vsel_t16'].get('port'))[:140]}" for sp in specs if "full_vsel_t16" in M[sp.alias]))
    say("\n  DEPTH LADDER — exact at step D from the per-step bits of ONE cold pass (each row its own run): model | set | protocol | n | " + " | ".join(f"D{s}" for s in LADDER) + " | regressions")
    for sp in specs:
        for sub in ("full_vsel_t16", "full_vsel_t64", "sub20k_t64", "sub20k_t128", "sub5k_t256"):
            r = load_run(sp.ev(sub))
            if r["z"] is None:
                if sub == "full_vsel_t64" or (sp.riders and sub in ("sub20k_t128", "sub5k_t256")): absent(sp, sp.ev(sub), "FULL" if sub.startswith("full") else "SCAN20k" if "20k" in sub else "SUB5k")
                continue
            lad = ladder(r)
            if lad is None: say(f"  {head(sp, r)} | " + " | ".join("-" for _ in LADDER) + " | - (no exact_by_step column: older records)"); continue
            M[sp.alias]["ladder_" + sub] = lad; say(f"  {head(sp, r)} | " + " | ".join(pc(lad.get(str(s))) for s in LADDER) + f" | {lad['regress']} ({pc(lad['regress_frac'], 3)} %)")
    say("\n  PROTOCOL DELTAS on the identical puzzles (row minus its comparator, restricted by idx): model | row | comparator | n common | row % | comparator % | delta pp | agreement %")
    for sp in specs:
        for sub, base in (("sub20k_t16_noise05", "full_vsel_t16"), ("sub20k_t16_noise001", "full_vsel_t16"), ("sub20k_t64_noise05", "full_vsel_t64"), ("full_raw_t16", "full_vsel_t16")):
            a, b = load_run(sp.ev(base)), load_run(sp.ev(sub))
            if a["z"] is None or b["z"] is None: continue
            _, ia, ib = align(a["z"], b["z"])
            if not len(ia): continue
            g = agree(a["z"]["cold_exact"][ia], b["z"]["cold_exact"][ib]); M[sp.alias]["delta_" + sub] = g
            say(f"  {sp.label():10s} | {sub:20s} | {base:14s} | {g['n']:6d} | {pc(g['b'])} | {pc(g['a'])} | {sg(g['delta'])} | {pc(g['agree'])}")

def table_B(specs):
    say("\n== B. DECODER LENS R1/R2/R3 — tools/suite_records.analyze_records on the FULL cold records (class per analyze_finalA R-A-4: DECIMATING iff no g50 in 17-35 and yield >= .70; SOFT iff g50 >= 24 and yield <= .45; else MIXED) ==")
    say("  model | set | protocol | n | cold % | g50 (width) [in 17-35] | P(cold|r0) % | yield P(cold|r>0) % | class | rating bands 0/1-9/10-29/30-59/60+ % | fe med/p90 (1-based; step-1 %) | viol on fail | cells on fail | valid-wrong % | givens kept %")
    for sp in specs:
        for sub in ("full_vsel_t64", "full_vsel_t16"):
            r = load_run(sp.ev(sub))
            if r["z"] is None: absent(sp, sp.ev(sub), "FULL"); continue
            row = lens(r)
            if row is None: say(f"  {head(sp, r)} | records index beyond the test set: lens skipped"); continue
            M[sp.alias]["lens" + sub[-2:]] = row
            say(f"  {head(sp, r)} | {pc(row['cold'])} | {fl(row['g50'], 1)} ({fl(row['width'], 1)}) [{'yes' if row['g50_in_range'] else 'no'}] | {pc(row['p_cold_r0'], 1)} | {pc(row['yield'], 1)} | {row['cls']:10s} | "
                + "/".join(pc(row["bands"].get(f"{lo}-{hi}"), 1) for lo, hi in RBANDS) + f" | {fe1(row.get('fe_med'))}/{fe1(row.get('fe_p90'))} ({pc(row.get('fe_step1'), 1)}) | {fl(row.get('viol_fail'), 2)} | {fl(row.get('cells_fail'), 1)} | {pc(row.get('valid_wrong'), 3)} | {pc(row.get('givens_kept'), 3)}")

def scan_line(sp, r, row):
    K = r["K"]; ver = row.get("verified", {})
    say(f"  {head(sp, r)} | {pc(row['cold'])} | {pc(row.get('b1'))} | " + "/".join(pc(ver.get(str(k)), 1) for k in K_CURVE if k <= K) + f" | {pc(row.get('t1r'))} ({fl(row.get('t1r_over_verified'), 3)}) | {pc(row.get('spurious'))} | {fl(row.get('auc'))} | {pc(row.get('rho'))} | {row.get('k50')}/{row.get('k90')} | {pc(row.get('rescue_1'), 1)}/{pc(row.get('rescue_any'), 1)} | {fl(row.get('r_med'))} | " + "/".join(pc(x, 1) for x in row.get("r_bins", [])))
    if row.get("t1r_curve"): say(f"  {'':10s}   t1r@k %: " + " ".join(f"{k}:{pc(v, 1)}" for k, v in row["t1r_curve"].items()) + "   verified@k %: " + " ".join(f"{k}:{pc(v, 1)}" for k, v in ver.items()) + ("   funnel (rating quartile: n cold b1 rho r): " + "; ".join(f"{f['bin']} {f['n']} {pc(f['cold'], 0)} {pc(f['b1'], 0)} {fl(f['rho'], 2)} {fl(f['r'], 3)}" for f in row.get("funnel", [])) if row.get("funnel") else ""))
def table_C(specs):
    say("\n== C. SELECTOR LAW — the k-draw scans (b1 = ONE random-init draw; verified@k = cold OR a verified hit among k; t1r@k = top-1 by residual; spurious = wrong draws under the median residual of exact draws; AUC = residual separates exact from wrong; rho = reach) ==")
    say("  model | set | protocol | n | cold % | b1 % | verified@1/2/4/../K % | t1r@K % (t1r/verified) | spurious % | AUC | rho % | k50/k90 | rescue 1/any % | r_i median | r_i bins 0/(0,.05]/(.05,.2]/(.2,.5]/(.5,1] %")
    for sp in specs:
        if not sp.riders: continue
        for suffix, key in (("", "scan"), ("_trunc", "scan_trunc"), ("_noise05", "scan_noise05")):
            if suffix and sp.tag != "eqr" and sp.alias != "eqr": continue
            r = load_run(sp.d("scan", suffix))
            if r["z"] is None or "mi_exact_k" not in r["z"]: absent(sp, sp.d("scan", suffix), "SCAN20k" if suffix != "_noise05" else "SUB5k"); continue
            row = lens(r); M[sp.alias][key] = row; scan_line(sp, r, row)

def table_D(specs):
    say("\n== D. SCREENS — STRAT512 x k256 at D16 with the unverified majority (uv_vote_k = the majority over k draws without a verifier) ==")
    say("  model | set | protocol | n | cold % | b1 % | verified@256 % | t1r@256 % | spurious % | AUC | majority@1/2/4/8/16/32/64/128/256 %")
    for sp in specs:
        if not sp.riders: continue
        r = load_run(sp.d("screen", "_vb"))
        if r["z"] is None or "mi_exact_k" not in r["z"]: absent(sp, sp.d("screen", "_vb"), "STRAT512"); continue
        row = lens(r); M[sp.alias]["screen"] = row; K = str(r["K"]); mj = row.get("majority", {})
        say(f"  {head(sp, r)} | {pc(row['cold'])} | {pc(row.get('b1'))} | {pc(row.get('verified', {}).get(K))} | {pc(row.get('t1r'))} | {pc(row.get('spurious'))} | {fl(row.get('auc'))} | " + "/".join(pc(mj.get(str(k)), 1) for k in K_CURVE if str(k) in mj))

def table_E(specs):
    say("\n== E. THE HALTING HEAD (R10) on the D16 fulls — (q_halt, q_continue) per step; AUC(q_halt at step 16, exact); halt = q_halt > q_continue at step 16; the emulated ACT row = exact at the first halting step ==")
    say("  model | set | protocol | n | cold % | AUC(q_halt) | AUC(q_halt - q_cont) | halt frac % | precision % | recall % | mean first-halt step (1-based) | exact at first halt % | delta vs cold pp | source")
    for sp in specs:
        r = load_run(sp.ev("full_vsel_t16"))
        if r["z"] is None: absent(sp, sp.ev("full_vsel_t16"), "FULL"); continue
        h = halting(r)
        if h is None: say(f"  {head(sp, r)} | {pc(float(r['z']['cold_exact'].mean()))} | - (no q_by_step column: older records)"); continue
        M[sp.alias]["halt"] = h; d = None if _bad(h.get("exact_at_first_halt")) else h["exact_at_first_halt"] - h["cold"]
        say(f"  {head(sp, r)} | {pc(h['cold'])} | {fl(h['auc'], 4)} | {fl(h['margin_auc'], 4)} | {pc(h['halt_frac'])} | {pc(h['precision'])} | {pc(h['recall'])} | {fl(h['first_halt_mean'], 2)} | {pc(h.get('exact_at_first_halt'))} | {sg(d)} | {h['source']}")

def table_F(specs):
    say("\n== F. INIT RADIUS — STRAT512 at D16, one perturbed draw per puzzle (z0 = the carried init + eps * N(0,1)); eps 0 = the same run's cold pass; the ratio = exact(eps) / cold ==")
    say("  model | set | protocol | n | cold (eps 0) % | " + " | ".join(f"eps {e}: exact % (ratio)" for e in EPS))
    for sp in specs:
        if not sp.riders: continue
        cells = {}; r0 = None
        for e in EPS:
            r = load_run(sp.ev(f"initrad_e{e}"))
            if r["z"] is None or "mi_exact_k" not in r["z"]: cells[e] = None; continue
            z = r["z"]; c = float(z["cold_exact"].mean()); p = float(z["mi_exact_k"][:, 0].mean()); cells[e] = dict(cold=c, exact=p, ratio=(p / c if c else None), n=r["n"], status=r["status"]); r0 = r0 or r
        if r0 is None: absent(sp, sp.ev("initrad_e*"), "STRAT512"); continue
        M[sp.alias]["initrad"] = cells; colds = {round(c["cold"], 6) for c in cells.values() if c}
        say(f"  {head(sp, r0)} | {pc(min(colds))}{'' if len(colds) == 1 else ' (cold differs across eps runs: ' + ','.join(pc(c) for c in sorted(colds)) + ')'} | "
            + " | ".join("ABSENT" if cells[e] is None else f"{pc(cells[e]['exact'])} ({fl(cells[e]['ratio'])})" + ("" if cells[e]["status"] == "OK" else f" [{cells[e]['status']} n={cells[e]['n']}]") for e in EPS))

def table_G(specs):
    say("\n== G. PREFIX ZEROED — SCAN20k at D16 with the trained puzzle prefix zeroed vs the same puzzles with it (bf16 comparator = the FULL D16 restricted by idx; the fp32 control as the second comparator) ==")
    say("  model | set | protocol | n common | prefix0 % | with prefix % (comparator) | delta pp | agreement %")
    for sp in specs:
        if not sp.riders: continue
        r = load_run(sp.ev("sub20k_t16_prefix0"))
        if r["z"] is None: absent(sp, sp.ev("sub20k_t16_prefix0"), "SCAN20k"); continue
        out = {}
        for base, lab in (("full_vsel_t16", "bf16 FULL D16 restricted"), ("sub20k_t16_fp32", "fp32 control")):
            a = load_run(sp.ev(base))
            if a["z"] is None: say(f"  {head(sp, r)} | comparator {base}: ABSENT"); continue
            _, ia, ib = align(a["z"], r["z"])
            if not len(ia): continue
            g = agree(a["z"]["cold_exact"][ia], r["z"]["cold_exact"][ib]); out[base] = g
            say(f"  {head(sp, r)} | {g['n']:6d} | {pc(g['b'])} | {pc(g['a'])} ({lab}) | {sg(g['delta'])} | {pc(g['agree'])}")
        M[sp.alias]["prefix0"] = out

def table_H(specs):
    say("\n== H. NUMERICS — the fp32 control vs the field's bf16 on the identical SCAN20k puzzles at D16; THE CROSS-ROUTE ROW: our pod rows (JAX, bf16) vs the Mac torch rows (runs/field_ckpts/out) on the identical idx ==")
    say("  model | set | A (comparator) | B (ours) | n common | A % | B % | B - A pp | agreement % | only-A | only-B | idx identical to our SCAN20k")
    for sp in specs:
        if not sp.riders: continue
        a, b = load_run(sp.ev("full_vsel_t16")), load_run(sp.ev("sub20k_t16_fp32"))
        if a["z"] is None or b["z"] is None: absent(sp, sp.ev("sub20k_t16_fp32"), "SCAN20k")
        else:
            _, ia, ib = align(a["z"], b["z"]); g = agree(a["z"]["cold_exact"][ia], b["z"]["cold_exact"][ib]); M[sp.alias]["fp32"] = g
            say(f"  {sp.label():10s} | {'SCAN20k':13s} | {'FULL D16 bf16 restricted':30s} | {proto(b):30s} | {g['n']:6d} | {pc(g['a'])} | {pc(g['b'])} | {sg(g['delta'])} | {pc(g['agree'])} | {g['only_a']:5d} | {g['only_b']:5d} | -")
        scan = load_run(sp.d("scan"))
        for i, name in enumerate(sp.mac):
            mz = load_mac(name)
            if mz["z"] is None: say(f"  {sp.label():10s} | {'SCAN20k':13s} | Mac {name}: ABSENT ({mz['note']})"); continue
            mb = bits_of(mz); ident = None if scan["z"] is None else bool(np.array_equal(np.sort(mz["z"]["idx"].astype(np.int64)), np.sort(scan["z"]["idx"].astype(np.int64)))); out = dict(idx_identical_to_scan=ident, mac_proto=mac_proto(mz), mac_n=mz["n"])
            for D, ours, olab, key in ((16, load_run(sp.ev("full_vsel_t16")), "FULL D16 bf16 restricted", "16"), (64, load_run(sp.ev("full_vsel_t64")), "FULL D64 bf16 restricted", "64"), (64, scan, "SCAN20k k-scan cold D64 bf16", "64scan")):
                if ours["z"] is None or mb is None or mb.shape[1] < D: continue
                _, ia, ib = align(ours["z"], mz["z"])
                if not len(ia): continue
                g = agree(mb[ib, D - 1], ours["z"]["cold_exact"][ia]); out[key] = g
                say(f"  {sp.label():10s} | {'SCAN20k':13s} | {('Mac @D' + str(D) + ' ' + name)[:34]:34s} | {olab:30s} | {g['n']:6d} | {pc(g['a'])} | {pc(g['b'])} | {sg(g['delta'])} | {pc(g['agree'])} | {g['only_a']:5d} | {g['only_b']:5d} | {ident}")
            say(f"  {'':10s}   Mac protocol: {mac_proto(mz)} (n={mz['n']})" + ("  [EqR: the Mac's trunc reset is a different draw from the pod's seed-20260907 draw — the agreement is a two-draw statistic, not a numerics floor]" if mz["s"].get("mac", {}).get("init") == "trunc" else ""))
            M[sp.alias].setdefault("cross", {})[name] = out
            if i == 0: M[sp.alias]["cross_primary"] = out

def table_I(specs):
    say("\n== I. EXPLOSION CENSUS + STALL CALIBRATION (tools/explosion_census.py census.json; tools/stall_calibration.py calib.json; both on STRAT512 seed 20260821) ==")
    say("  CENSUS: model | set | protocol | n | exploded % | n exploded | first-bad median | zmax median / p99 | logit max p99")
    for sp in specs:
        if not sp.riders: continue
        c = jload(sp.d("census", "_vsel") / "census.json")
        if not c: say(f"  {sp.label():10s} | {'STRAT512':13s} | ABSENT {sp.d('census', '_vsel').name}"); continue
        M[sp.alias]["census"] = c.get("rows", [])
        for row in c.get("rows", []):
            say(f"  {sp.label():10s} | {'STRAT512':13s} | {'cold D' + str(row.get('t')) + (' EMA' if row.get('ema') else ' raw'):36s} | n={row.get('n', 0):6d} | {pc(row.get('exploded_frac'))} | {row.get('n_exploded')} | {row.get('first_bad_median')} | {fl(row.get('zmax_median'), 2)} / {fl(row.get('zmax_p99'), 2)} | {fl(row.get('logit_max_p99'), 1)}")
    say("  CALIBRATION: model | set | protocol | n | cold % | n stalled | top-k correct at stalls | mean conf at stalls | gap (conf - top-k) | entropy step 1 / at t (stalled) | conf-wrong frac at stalls")
    for sp in specs:
        if not sp.riders: continue
        c = jload(sp.d("calib", "_vsel") / "calib.json")
        if not c: say(f"  {sp.label():10s} | {'STRAT512':13s} | ABSENT {sp.d('calib', '_vsel').name}"); continue
        c["gap"] = None if _bad(c.get("mean_conf_stalled")) or _bad(c.get("topk_correct_stalled")) else c["mean_conf_stalled"] - c["topk_correct_stalled"]; M[sp.alias]["calib"] = c
        say(f"  {sp.label():10s} | {'STRAT512':13s} | {'cold D' + str(c.get('t')) + (' EMA' if c.get('ema') else ' raw') + ' top-' + str(c.get('topk')):36s} | n={c.get('n', 0):6d} | {pc(c.get('cold'))} | {c.get('n_stalled')} | {fl(c.get('topk_correct_stalled'))} | {fl(c.get('mean_conf_stalled'))} | {fl(c.get('gap'))} | {fl(c.get('entropy_step1'))} / {fl(c.get('entropy_t_stalled'))} | {fl(c.get('conf_wrong_frac_stalled'))}")

def table_J(arms):
    say("\n== J. THE DEC PASS TWO — the wide arms' scans (registered 5k x k32 t64 under the canary recipe) and the D128 rows on the 20k (Night A's val-selected DEC grids, EMA) ==")
    rp = RUNS / "frontier_recipe.txt"; recipe = rp.read_text().strip() if rp.exists() else None; DEC["recipe"] = recipe
    say(f"  RECIPE-DEC (runs/frontier_recipe.txt): {recipe if recipe else 'ABSENT'}")
    say("  SCAN: arm | set | protocol | n | cold % | b1 % | verified@1/2/../K % | t1r@K % (t1r/verified) | spurious % | AUC | rho % | k50/k90 | rescue 1/any % | r_i median | r_i bins %   [stall evidence: py-spy dumps / partials / shard logs in the directory]")
    for arm in arms:
        d = RUNS / f"sxscan_pfinalA{arm}"; r = load_run(d); sp = Spec(f"{arm}:pfinalA"); ev = dict(pyspy=len(list(d.glob("pyspy_*.txt"))), partials=len(list(d.glob("partial_s*.npz"))), logs=len(list(d.glob("shard_*.log")))) if d.exists() else {}
        DEC[arm] = dict(stall_evidence=ev)
        if r["z"] is None or "mi_exact_k" not in r["z"]: say(f"  {arm:10s} | {'SUB5k':13s} | ABSENT {d.name} ({r['note']}; py-spy dumps {ev.get('pyspy', 0)}, partials {ev.get('partials', 0)}, shard logs {ev.get('logs', 0)})"); continue
        row = lens(r); DEC[arm]["scan"] = row; DEC[arm]["scan_meta"] = dict(set=set_label(r), proto=proto(r), n=r["n"], status=r["status"], z0_device=r["s"].get("z0_device")); scan_line(sp, r, row)
        say(f"  {'':10s}   stall evidence: py-spy dumps {ev['pyspy']}, partials {ev['partials']}, shard logs {ev['logs']}")
    say("  D128: arm | set | protocol | n | D16 | D32 | D64 | D128 % (one pass) | D128 - D64 pp | regressions | Night A D64 (full_vsel_t64 restricted by idx): n common / D64 % / agreement % / D128 - D64 pp")
    for arm in arms:
        d = RUNS / f"sxeval_pfinalA{arm}" / "sub20k_t128"; r = load_run(d); sp = Spec(f"{arm}:pfinalA")
        if r["z"] is None: say(f"  {arm:10s} | {'SCAN20k':13s} | ABSENT {d.parent.name}/{d.name} ({r['note']})"); continue
        lad = ladder(r); DEC[arm]["d128"] = lad
        if lad is None: say(f"  {head(sp, r)} | no exact_by_step column"); continue
        a = load_run(RUNS / f"sxeval_pfinalA{arm}" / "full_vsel_t64"); cmp_ = None
        if a["z"] is not None:
            _, ia, ib = align(a["z"], r["z"])
            if len(ia): cmp_ = agree(a["z"]["cold_exact"][ia], r["z"]["cold_exact"][ib]); DEC[arm]["d128_vs_nightA_d64"] = cmp_
        say(f"  {head(sp, r)} | {pc(lad.get('16'))} | {pc(lad.get('32'))} | {pc(lad.get('64'))} | {pc(lad.get('128'))} | {sg(None if lad.get('64') is None else lad.get('128', lad['last']) - lad['64'])} | {lad['regress']} ({pc(lad['regress_frac'], 3)} %) | "
            + ("-" if cmp_ is None else f"{cmp_['n']} / {pc(cmp_['a'])} / {pc(cmp_['agree'])} / {sg(cmp_['delta'])}"))

def table_K(specs, x0):
    say("\n== K. PAIRWISE OVERLAP of the solved sets on the identical idx (FULL D16 and FULL D64 cold): Jaccard, discordants, McNemar (continuity-corrected chi2, df 1), conditional failure ==")
    say("  pair A|B | set | D | n common | A % | B % | Jaccard(solved) | only-A | only-B | McNemar chi2 | p | P(A fails|B fails) % | P(B fails|A fails) %")
    out = {}
    for sub, D in (("full_vsel_t16", 16), ("full_vsel_t64", 64)):
        rows = [(sp.label(), load_run(sp.ev(sub))) for sp in list(specs) + [x0]]; rows = [(lab, r) for lab, r in rows if r["z"] is not None]
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                (la, ra), (lb, rb) = rows[i], rows[j]; _, ia, ib = align(ra["z"], rb["z"])
                if not len(ia): continue
                ea, eb = ra["z"]["cold_exact"].astype(bool)[ia], rb["z"]["cold_exact"].astype(bool)[ib]; m = mcnemar(ea, eb); m.update(n=int(len(ia)), a=float(ea.mean()), b=float(eb.mean()), set=set_label(ra) + "&" + set_label(rb), D=D); out[f"{la}|{lb}@D{D}"] = m
                say(f"  {la:10s}|{lb:10s} | {m['set']:13s} | {D:3d} | {m['n']:6d} | {pc(m['a'])} | {pc(m['b'])} | {fl(m['jaccard'], 4)} | {m['only_a']:6d} | {m['only_b']:6d} | {m['chi2']:9.1f} | {m['p']:.2e} | {pc(m['pA_fail_given_B_fail'], 1)} | {pc(m['pB_fail_given_A_fail'], 1)}")
    if not out: say("  (no pair with two record sets present)")
    J["K"] = out

# ---------------------------------------------------------------- L. the prediction scoreboard ----------------------------------------------------------------
def verdict(v, lo, hi):
    if v is None or (isinstance(v, float) and not math.isfinite(v)): return "NO-DATA"
    if isinstance(v, (bool, np.bool_)): return "HIT" if v else "MISS"
    if lo is not None and v < lo - 1e-9: return "MISS-BELOW"
    if hi is not None and v > hi + 1e-9: return "MISS-ABOVE"
    return "HIT"
def P(pid, clause, v, lo=None, hi=None, unit=" pp", fmt="{:.2f}"):
    isb = isinstance(v, (bool, np.bool_)) or (lo is None and hi is None); vd = verdict(v, lo, hi)     # no band = a true/false clause
    band = "true" if isb else (f"[{lo:g}, {hi:g}]{unit}" if lo is not None and hi is not None else f">= {lo:g}{unit}" if lo is not None else f"<= {hi:g}{unit}")
    SB.append(dict(p=pid, clause=clause, band=band, value=(None if v is None else bool(v) if isb else float(v)), verdict=vd))
    say(f"  {pid:4s} | {clause:78s} | {band:18s} | {('-' if v is None else str(bool(v)) if isb else fmt.format(v)):>10s} | {vd}")
def mv(alias, *ks, pct=True):
    v = M.get(alias, {})
    for k in ks:
        if not isinstance(v, dict) or v.get(k) is None: return None
        v = v[k]
    return None if _bad(v) else (100 * v if pct else v)
def dv(alias, *ks, pct=True):
    v = DEC.get(alias, {})
    for k in ks:
        if not isinstance(v, dict) or v.get(k) is None: return None
        v = v[k]
    return None if _bad(v) else (100 * v if pct else v)
def sub(a, b): return None if a is None or b is None else a - b
def scoreboard(specs, arms):
    say("\n== L. THE PREDICTION SCOREBOARD (registration §4, P1-P27; the bands as worded; one line per clause; nothing is decided by them) ==")
    if any(sp.alias != sp.tag for sp in specs): say("  !! TEST ALIASES IN FORCE: " + ", ".join(f"{sp.tag} scored as {sp.alias}" for sp in specs if sp.alias != sp.tag) + " — these verdicts are a dry run of the scoreboard, not the frontier's numbers")
    say("  P | clause | predicted band | measured | verdict")
    for t, lo16, hi16, lo64, hi64, pid in (("trmpub", 78, 80, 82.5, 84.5, "P1"), ("cgar", 85.5, 86.7, 91, 92.5, "P2"), ("eqr", 86, 87.5, 92.5, 93.7, "P3")):
        P(pid, f"{t} cold D16 (FULL) in [{lo16}, {hi16}]", mv(t, "full_vsel_t16", "cold"), lo16, hi16, " %"); P(pid, f"{t} cold D64 (FULL) in [{lo64}, {hi64}]", mv(t, "full_vsel_t64", "cold"), lo64, hi64, " %")
    P("P4", "trmpub60k D16 minus trmpub D16 within +-1.5 pp", sub(mv("trmpub60k", "full_vsel_t16", "cold"), mv("trmpub", "full_vsel_t16", "cold")), -1.5, 1.5)
    for D in ("16", "64"):
        P("P5", f"trmpub pod bf16 minus Mac fp32 torch on the identical 20k at D{D} within +-.5 pp", mv("trmpub", "cross_primary", D, "delta"), -.5, .5); P("P5", f"trmpub per-puzzle agreement pod bf16 vs Mac fp32 at D{D} >= 94 %", mv("trmpub", "cross_primary", D, "agree"), 94, None, " %")
    for t in ("trmpub", "eqr", "cgar"):
        P("P5", f"{t} fp32 control minus bf16 (SCAN20k D16) within +-.3 pp", mv(t, "fp32", "delta"), -.3, .3); P("P5", f"{t} fp32 vs bf16 per-puzzle agreement in 94-97 %", mv(t, "fp32", "agree"), 94, 97, " %")
    for t, lo, hi in (("trmpub", 1, 2), ("cgar", .5, 1.5), ("eqr", .5, 1.5)):
        P("P6", f"{t} D128 minus D64 (one 20k pass) in [{lo}, {hi}] pp", sub(mv(t, "ladder_sub20k_t128", "128"), mv(t, "ladder_sub20k_t128", "64")), lo, hi)
        P("P6", f"{t} D256 minus D128 (one 5k pass) <= +.5 pp", sub(mv(t, "ladder_sub5k_t256", "256"), mv(t, "ladder_sub5k_t256", "128")), None, .5)
    for t in ("trmpub", "cgar", "eqr"):
        vals = [mv(t, k, "regress_frac") for k in ("ladder_full_vsel_t16", "ladder_full_vsel_t64", "ladder_sub20k_t128", "ladder_sub5k_t256")]; vals = [v for v in vals if v is not None]
        P("P7", f"{t} depth regressions <= .1 % of puzzles at every depth (max over {len(vals)} depth rows)", max(vals) if vals else None, None, .1, " %", "{:.3f}")
    for t in ("cgar", "eqr"):
        P("P8", f"{t} class DECIMATING on the FULL D64 records (no g50 in 17-35)", None if mv(t, "lens64", "cls", pct=False) is None else mv(t, "lens64", "cls", pct=False) == "DECIMATING"); P("P8", f"{t} yield P(cold | rating > 0) >= 88 %", mv(t, "lens64", "yield"), 88, None, " %")
    P("P8", "trmpub class MIXED on the FULL D64 records", None if mv("trmpub", "lens64", "cls", pct=False) is None else mv("trmpub", "lens64", "cls", pct=False) == "MIXED"); P("P8", "trmpub g50 in [17, 21] givens", mv("trmpub", "lens64", "g50", pct=False), 17, 21, ""); P("P8", "trmpub yield in 78-83 %", mv("trmpub", "lens64", "yield"), 78, 83, " %")
    for t in ("trmpub", "cgar", "eqr"):
        fm = mv(t, "lens64", "fe_med", pct=False); P("P9", f"{t} first-exact median (1-based, FULL D64 records) == 1", None if fm is None else fm + 1, 1, 1, "", "{:.0f}"); fp = mv(t, "lens64", "fe_p90", pct=False); P("P9", f"{t} first-exact p90 (1-based) <= 12", None if fp is None else fp + 1, None, 12, "", "{:.0f}")
    for t, lo, hi in (("eqr", None, 1), ("trmpub", 2, 5), ("cgar", 20, 30)): P("P10", f"{t} spurious rate (SCAN20k x k128)" + (f" in {lo}-{hi} %" if lo is not None else f" <= {hi} %"), mv(t, "scan", "spurious"), lo, hi, " %")
    for t in ("trmpub", "eqr"): P("P11", f"{t} b1 minus cold (SCAN20k scan) within +-2 pp", sub(mv(t, "scan", "b1"), mv(t, "scan", "cold")), -2, 2)
    P("P11", "cgar cold minus b1 in [15, 25] pp", sub(mv("cgar", "scan", "cold"), mv("cgar", "scan", "b1")), 15, 25)
    for t, lo in (("trmpub", 97), ("eqr", 99), ("cgar", 99)): P("P12", f"{t} verified@128 >= {lo} %" + (" (>= 97 on all three)" if lo == 97 else ""), mv(t, "scan", "verified", "128"), lo, None, " %")
    for t, lo, hi in (("eqr", .98, None), ("trmpub", .95, .99), ("cgar", None, .85)): P("P13", f"{t} t1r@128 / verified@128" + (f" in [{lo}, {hi}]" if lo is not None and hi is not None else f" >= {lo}" if lo is not None else f" <= {hi}"), mv(t, "scan", "t1r_over_verified", pct=False), lo, hi, "", "{:.4f}")
    for t in ("trmpub", "cgar", "eqr"): P("P14", f"{t} AUC(q_halt at step 16, exact) >= .995 (D16 full)", mv(t, "halt", "auc", pct=False), .995, None, "", "{:.4f}")
    for t in ("trmpub", "cgar", "eqr"): P("P15", f"{t} exact at the first halting step minus D16 cold within +-1 pp", sub(mv(t, "halt", "exact_at_first_halt"), mv(t, "halt", "cold")), -1, 1)
    P("P16", "eqr noise .5 cost at D16 on the 20k (noise 0 minus noise .5) in [8, 11] pp", None if mv("eqr", "delta_sub20k_t16_noise05", "delta") is None else -mv("eqr", "delta_sub20k_t16_noise05", "delta"), 8, 11)
    P("P16", "eqr noise .5 cost at D64 on the 20k in [2.5, 4] pp", None if mv("eqr", "delta_sub20k_t64_noise05", "delta") is None else -mv("eqr", "delta_sub20k_t64_noise05", "delta"), 2.5, 4)
    P("P16", "eqr noise .01 minus noise 0 at D16 within +-.5 pp", mv("eqr", "delta_sub20k_t16_noise001", "delta"), -.5, .5)
    P("P17", "eqr trunc-reset scan b1 minus gauss scan b1 within +-1.5 pp", sub(mv("eqr", "scan_trunc", "b1"), mv("eqr", "scan", "b1")), -1.5, 1.5); P("P17", "eqr trunc-reset scan spurious rate <= 1 %", mv("eqr", "scan_trunc", "spurious"), None, 1, " %")
    P("P18", "eqr EMA minus raw weights at D16 (FULL) in [1, 8] pp", None if mv("eqr", "delta_full_raw_t16", "delta") is None else -mv("eqr", "delta_full_raw_t16", "delta"), 1, 8)
    P("P19", "eqr own train-1k cold D16 in [79, 85] %", mv("eqr", "train1k_t16", "cold"), 79, 85, " %"); P("P19", "eqr train-1k minus test D16 within +-3 pp (no memorization)", sub(mv("eqr", "train1k_t16", "cold"), mv("eqr", "full_vsel_t16", "cold")), -3, 3)
    for t, lo, hi in (("eqr", 95, None), ("trmpub", 90, None), ("cgar", None, 88)): P("P20", f"{t} exact at eps 3 as % of cold (D16, STRAT512)" + (f" >= {lo}" if lo else f" <= {hi}"), mv(t, "initrad", "3", "ratio"), lo, hi, " %")
    P("P20", "cgar gain at eps .3 (exact minus cold) >= +1 pp", sub(mv("cgar", "initrad", "0.3", "exact"), mv("cgar", "initrad", "0.3", "cold")), 1, None)
    for t, lo, hi in (("trmpub", -1.5, 1.5), ("cgar", -1.5, 1.5), ("eqr", -7, -4)): P("P21", f"{t} prefix zeroed minus with prefix (SCAN20k D16, bf16 comparator) in [{lo}, {hi}] pp", mv(t, "prefix0", "full_vsel_t16", "delta"), lo, hi)
    for t in ("trmpub", "cgar", "eqr"):
        rows = M.get(t, {}).get("census"); P("P22", f"{t} explosion census exploded == 0.00 % (max over census rows)", None if not rows else max(100 * (x.get("exploded_frac") or 0) for x in rows), 0, 0, " %"); P("P22", f"{t} top-5 correct at stalls (calibration) in [.45, .60]", mv(t, "calib", "topk_correct_stalled", pct=False), .45, .60, "", "{:.3f}")
    for t, lo, hi in (("eqr", 90, None), ("cgar", 85, 90), ("trmpub", 78, 85)): P("P23", f"{t} unverified majority@256 (STRAT512 D16)" + (f" in {lo}-{hi} %" if hi else f" >= {lo} %"), mv(t, "screen", "majority", "256"), lo, hi, " %")
    rec = DEC.get("recipe"); a3 = DEC.get("A3", {}); z0dev = (a3.get("scan_meta") or {}).get("z0_device"); pyspy = (a3.get("stall_evidence") or {}).get("pyspy", 0)
    canary = None if rec is None and not a3.get("scan") else (rec.strip() in ("BATCHONLY", "") if rec is not None else (not z0dev and pyspy == 0))
    P("P24", f"the A3 canary completes at --batch 128 without a stall (recipe {rec!r}; z0_device {z0dev}; py-spy dumps {pyspy})", canary)
    P("P24", "if it stalled: --z0-device completes (the 35 % branch; n/a when the canary passed)", None if canary is None or canary else (rec.strip() == "--z0-device" if rec else bool(z0dev)))
    for arm, lo, hi in (("A3", 2, 6), ("A7", 30, None), ("A5", None, .5), ("A4", 1, 4), ("A8", None, .5)): P("P25", f"DEC {arm} spurious rate (5k x k32 scan)" + (f" in {lo}-{hi} %" if lo is not None and hi is not None else f" >= {lo} %" if lo is not None else f" <= {hi} %"), dv(arm, "scan", "spurious"), lo, hi, " %")
    for arm in arms: P("P26", f"DEC {arm} verified@32 >= 99 %", dv(arm, "scan", "verified", "32"), 99, None, " %"); P("P26", f"DEC {arm} b1 minus cold within +-2 pp", sub(dv(arm, "scan", "b1"), dv(arm, "scan", "cold")), -2, 2)
    for arm, lo, hi in (("A3", .3, 1.0), ("A7", .2, .6), ("A5", .2, .6)): P("P27", f"DEC {arm} D128 minus D64 (one 20k pass) in [{lo}, {hi}] pp", sub(dv(arm, "d128", "128"), dv(arm, "d128", "64")), lo, hi)
    for arm in ("A3", "A7", "A5"): P("P27", f"DEC {arm} D128 depth regressions <= .05 %", dv(arm, "d128", "regress_frac"), None, .05, " %", "{:.3f}")
    tally = {k: sum(1 for s in SB if s["verdict"] == k) for k in ("HIT", "MISS-ABOVE", "MISS-BELOW", "MISS", "NO-DATA")}; J["scoreboard"] = SB; J["tally"] = tally
    say("  TALLY over clauses: " + ", ".join(f"{k} {v}" for k, v in tally.items()) + "   (a prediction is HIT only when every one of its clauses is; credences: P1 75, P2 70, P3 70, P4 60, P5 70, P6 55, P7 70, P8 60, P9 75, P10 60, P11 65, P12 65, P13 60, P14 75, P15 55, P16 70, P17 65, P18 60, P19 70, P20 60, P21 65, P22 70, P23 55, P24 50/35, P25 55, P26 70, P27 55)")
    per = {}
    for s in SB: per.setdefault(s["p"], []).append(s["verdict"])
    J["per_prediction"] = {p: ("NO-DATA" if all(v == "NO-DATA" for v in vs) else "HIT" if all(v == "HIT" for v in vs) else "MISS" if any(v.startswith("MISS") for v in vs) else "PARTIAL(no-data clauses)") for p, vs in per.items()}
    say("  PER PREDICTION: " + " ".join(f"{p}:{v}" for p, v in J["per_prediction"].items()))

def main():
    global RUNS, G, R
    ap = argparse.ArgumentParser(description="FRONTIER POD RUN descriptive analyzer (no rules)")
    ap.add_argument("--date", default=time.strftime("%Y%m%d")); ap.add_argument("--runs", default="runs")
    ap.add_argument("--tags", default="trmpub,eqr,cgar,trmpub60k", help="tag[:prefix[:mac[:alias]]] entries (see the docstring)")
    ap.add_argument("--dec-arms", default="A3,A7,A5,A4,A8"); ap.add_argument("--x0", default="X0:psportC1", help="the comparator's tag:prefix for table K")
    a = ap.parse_args(); t0 = time.time(); RUNS = Path(a.runs) if Path(a.runs).is_absolute() else ROOT / a.runs
    from qhrrn2 import sudoku_extreme as SX
    d = SX.load_prepared(ROOT / "data/sudoku_extreme/sudoku_extreme_seed0.npz"); Q = d["test_q"]; R = d["test_rating"].astype(np.int64); G = (Q != 0).reshape(len(Q), -1).sum(1)
    specs = [Spec(s) for s in a.tags.split(",") if s]; arms = [x for x in a.dec_arms.split(",") if x]; x0 = Spec(a.x0)
    for sp in specs: M.setdefault(sp.alias, {})
    say("=" * 150); say(f"FRONTIER POD RUN — DESCRIPTIVE ANALYSIS ({a.date}; registration Plan_2026-09-07_Frontier_Registration.md; chain tools/chain_frontier.sh; NO RULES, nothing decided)"); say("=" * 150)
    say("  models: " + "; ".join(f"{sp.label()} -> runs/sx*_{sp.prefix}{sp.tag}* (Mac cross-route: {'+'.join(sp.mac) or 'none'})" for sp in specs) + f"; DEC arms: {' '.join(arms)} -> runs/sxscan_pfinalA<arm>, runs/sxeval_pfinalA<arm>/sub20k_t128; X0: runs/sxeval_{x0.prefix}{x0.tag}")
    say("  precision column: inferred from the directory name (the chain's ARM_PREC: bf16 matmul on every field-class eval, fp32 = 'highest' on sub20k_t16_fp32); first-exact steps printed 1-based (records are 0-based)")
    table_0(specs, arms); table_A(specs); table_B(specs); table_C(specs); table_D(specs); table_E(specs); table_F(specs); table_G(specs); table_H(specs); table_I(specs); table_J(arms); table_K(specs, x0); scoreboard(specs, arms)
    J.update(date=a.date, models={sp.alias: dict(tag=sp.tag, prefix=sp.prefix, mac=sp.mac, alias=sp.alias) for sp in specs}, M=M, DEC=DEC, seconds=round(time.time() - t0, 1))
    out = RUNS / "analysis" / f"frontier_{a.date}"; out.parent.mkdir(parents=True, exist_ok=True)
    say(f"\n(artifact frontier_{a.date}; {time.time() - t0:.0f}s)"); out.with_suffix(".txt").write_text("\n".join(L) + "\n"); out.with_suffix(".json").write_text(json.dumps(J, indent=0, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print(f"-> {out}.txt / .json")

if __name__ == "__main__":
    main()
