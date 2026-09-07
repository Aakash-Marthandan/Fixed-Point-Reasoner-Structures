#!/usr/bin/env python3
"""THE CORPUS THROUGH ONE LENS (Plan_2026-09-07_Instrument_Suite §5; 2026-09-07; descriptive, no rules): one row per banked grid
joining (R1) the decoder-class reading of its D64 record set (g50 / in-range / search yield -> the class letter as
analyze_finalA's R-A-4 words it: DECIMATING = no g50 in 17-35 and yield >= .70; SOFT = g50 in range and yield < .50; else MIXED),
(R2) first-exact, (C1) the dynamics row of the strat-128 t64 backfill (commit at step 1 solved|unsolved, confidently-wrong at
step 1 -> 64 on failures, monotone solves, flips-to-wrong per cell over the last 32 steps, syndrome oscillation), (C2) decimation
quality at stalls (committed / wrong-among-committed / peeling contradiction at tau .9) and (C3) calibration at stalls (top-5
correct vs mean confidence, entropy at step 1). Sources: runs/analysis/suite_records_<date>.csv, suite_ckpt_dyn_<date>.jsonl,
runs/analysis/finalA_ecc_<date>.json (the Night A lens rows), runs/sxcalib_*/ (calibration). Output: runs/analysis/corpus_lens_<date>.{csv,txt}.
  PYTHONPATH=src .venv/bin/python tools/suite_corpus_table.py --date 20260907"""
import argparse, csv, glob, json, os, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; RUNS = ROOT / "runs"; AN = RUNS / "analysis"
LADDER = ["sport2", "sport3a", "sportB", "sportBr2", "sportBr2b", "sportC0", "sportC1", "sportC2", "finalA", "frontier", "field"]


def fnum(x):
    try: return float(x)
    except Exception: return None


def class_letter(g50_in_range, yld):
    if yld is None: return "-"
    if not g50_in_range and yld >= .70: return "DECIMATING"
    if g50_in_range and yld < .50: return "SOFT"
    return "MIXED"


def load_records(date):
    rows = list(csv.DictReader(open(AN / f"suite_records_{date}.csv")))
    by = {}
    for r in rows:
        key = (r["campaign"] or r["kind"], r["arm"])
        if not r["campaign"] and r["arm"].startswith("frontier"):   # the pod's ported-checkpoint evals parse as arm 'frontier<tag>'
            key = ("frontier", r["arm"][len("frontier"):])
        by.setdefault(key, []).append(r)
    return by


def pick_d64(recs):
    """the D64 record set of a grid: the val-selected full at t64, else the final full at t64, else the 20k scan (t64), else any t64 row."""
    def score(r):
        ev = r.get("eval", ""); k = r.get("kind", ""); t = r.get("t", "")
        if t != "64": return -1
        if k == "full" and "vsel" in ev and "alt" not in ev: return 5
        if k == "full" and "final" in ev: return 4
        if k == "scan": return 3
        if k == "full": return 2
        return 1
    best = max(recs, key=score, default=None)
    return best if best is not None and score(best) >= 1 else None


def load_dyn(date):
    out = {}
    p = AN / f"suite_ckpt_dyn_{date}.jsonl"
    if p.exists():
        for l in open(p):
            try: r = json.loads(l)
            except Exception: continue
            out[r["name"]] = r
    # the Night A lens rows (A3, A7, A5, A8, A0, X0 at n 128 t 64) carry the same fields
    q = AN / f"finalA_ecc_{date}.json"
    if q.exists():
        J = json.load(open(q))
        for k, o in J.get("E2", {}).items():
            name = f"{'finalA' if k.startswith('A') else 'sportC1'}/{k}@vsel"
            out.setdefault(name, dict(o, name=name, e3=J.get("E3", {}).get(k, {}), cell=("dec" if k in ("A3", "A5", "A7", "A8") else "trm")))
    return out


def load_calib():
    out = {}
    for d in glob.glob(str(RUNS / "sxcalib_*")):
        fs = glob.glob(d + "/*.json")
        if not fs: continue
        try: c = json.load(open(fs[0]))
        except Exception: continue
        b = os.path.basename(d)
        out[b] = c
    return out


def calib_for(campaign, arm, grid):
    cands = [f"sxcalib_p{campaign}{arm}_{grid}", f"sxcalib_suite_{campaign}_{arm}_{grid}", f"sxcalib_p{campaign}{arm}_vsel", f"sxcalib_suite_{campaign}_{arm}_final"]
    return cands


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--date", default=time.strftime("%Y%m%d")); a = ap.parse_args()
    recs = load_records(a.date); dyn = load_dyn(a.date); cal = load_calib()
    rows = []
    names = sorted(dyn, key=lambda n: (next((i for i, c in enumerate(LADDER) if n.startswith(c)), 99), n))
    for name in names:
        d = dyn[name]; camp, rest = name.split("/", 1); arm, grid = rest.split("@")
        rr = recs.get((camp, arm), [])
        r64 = pick_d64(rr)
        g50 = fnum(r64["g50"]) if r64 else None; inr = (r64["g50_in_range"] in ("True", "1", "true")) if r64 else False; yld = fnum(r64["yield"]) if r64 else None
        cold64 = fnum(r64["cold"]) if r64 else None
        c = next((cal[k] for k in calib_for(camp, arm, grid) if k in cal), None)
        e3 = (d.get("e3") or {}).get("0.9") or {}
        row = dict(grid=name, cell=d.get("cell"), record_set=(f"{r64['kind']}:{r64['eval'] or r64['set']} n{r64['n']}" if r64 else "-"),
                   cold64=cold64, g50=g50, g50_in_range=inr, yield_=yld, cls=class_letter(inr, yld) if r64 else "-",
                   fe_med=fnum(r64["fe_med"]) if r64 else None, fe_p90=fnum(r64["fe_p90"]) if r64 else None,
                   dyn_solved=d.get("solved"), commit1_s=(d.get("com_s") or [None])[0], commit1_u=(d.get("com_u") or [None])[0],
                   cw1_u=(d.get("cw_u") or [None])[0], cw64_u=(d.get("cw_u") or [None, None, None])[-1], cells_u1=(d.get("cells_u") or [None])[0],
                   ent1_s=(d.get("ent_s") or [None])[0], mono=d.get("mono_solved"), flips_w=d.get("flips_w_last32_unsolved"), syn_osc=d.get("syn_osc_unsolved"),
                   dyn_fe_med=d.get("fe_med"), dyn_fe_p90=d.get("fe_p90"),
                   e3_committed=e3.get("committed_frac"), e3_wrong=e3.get("wrong_committed_frac"), e3_contra=e3.get("peel_contradiction"),
                   cal_cold=(c or {}).get("cold"), cal_stalled=(c or {}).get("n_stalled"), cal_top5=(c or {}).get("topk_correct_stalled"),
                   cal_conf=(c or {}).get("mean_conf_stalled"), cal_ent1=(c or {}).get("entropy_step1"), cal_cw=(c or {}).get("conf_wrong_frac_stalled"))
        rows.append(row)
    # the Mac field study's rows (their own torch code; R1 on SCAN20k D64, dynamics on STRAT256 t64, calibration E6) — labeled by protocol
    fj = RUNS / "field_ckpts/analysis/field_ckpts_20260906.json"
    if fj.exists():
        J = json.load(open(fj)); label = {"hrm": "HRM-pub", "trm": "TRM-pub(alphaXiv)", "trmc": "TRM-CGAR", "eqr": "EqR-pub(noise .5)", "eqr(n0)": "EqR-pub(noise 0)"}
        for k in ("hrm", "trm", "trmc", "eqr(n0)", "eqr"):
            e1 = J.get("E1", {}).get(f"{k}@64", {}); e2 = J.get("E2", {}).get(k, {}); e3 = (J.get("E3", {}).get(k, {}) or {}).get("0.9", {}) or {}; e6 = J.get("E6", {}).get(k, {})
            if not e2: continue
            g50 = e1.get("g50"); inr = g50 is not None and 17 <= g50 <= 35; yld = e1.get("yield_search")
            def pick(d, *names):
                for n_ in names:
                    if n_ in d: return d[n_]
                for kk in d:
                    if any(n_.split("_")[0] in kk for n_ in names): return d[kk]
                return None
            rows.append(dict(grid=f"field/{label[k]}@Mac-torch", cell="trm" if k != "hrm" else "hrm", record_set="field:cold SCAN20k D64 (torch fp32); dyn STRAT256 t64; calib E6 (labeled protocol)",
                             cold64=e1.get("cold"), g50=g50, g50_in_range=inr, yield_=(yld / 100 if (yld is not None and yld > 1) else yld), cls=(e1.get("cls") or class_letter(inr, yld / 100 if (yld and yld > 1) else yld)),
                             fe_med=e2.get("fe_med"), fe_p90=e2.get("fe_p90"), dyn_solved=e2.get("solved"), commit1_s=(e2.get("com_s") or [None])[0], commit1_u=(e2.get("com_u") or [None])[0],
                             cw1_u=(e2.get("cw_u") or [None])[0], cw64_u=(e2.get("cw_u") or [None, None, None])[-1], cells_u1=(e2.get("cells_u") or [None])[0], ent1_s=(e2.get("ent_s") or [None])[0],
                             mono=e2.get("mono_solved"), flips_w=e2.get("flips_w_u"), syn_osc=e2.get("syn_osc"), dyn_fe_med=e2.get("fe_med"), dyn_fe_p90=e2.get("fe_p90"),
                             e3_committed=e3.get("committed"), e3_wrong=e3.get("wrong"), e3_contra=e3.get("contra"),
                             cal_cold=None, cal_stalled=e6.get("n"), cal_top5=e6.get("top5"), cal_conf=e6.get("conf"), cal_ent1=e6.get("ent1"), cal_cw=e6.get("cw")))
    out_csv = AN / f"corpus_lens_{a.date}.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    P = lambda x, p=0: "-" if x is None else f"{100 * x:.{p}f}"
    F = lambda x, p=1: "-" if x is None else f"{x:.{p}f}"
    lines = ["THE CORPUS THROUGH ONE LENS — one row per banked grid (R1 class from its D64 record set; C1/C2 from the strat-128 t64 dynamics backfill; C3 from the strat-512 calibration; descriptive)",
             f"{'grid':26s} {'cell':4s} {'class':10s} {'cold64':>6s} {'g50':>5s} {'yield':>5s} {'fe':>5s} | {'dyn':>5s} {'commit@1 s|u':>12s} {'cw u 1->64':>10s} {'ent1':>5s} {'mono':>4s} {'flipW':>5s} {'synosc':>6s} | {'E3 com/wrong/contra':>19s} | {'C3 top5/conf':>12s} {'ent1':>5s}"]
    for r in rows:
        lines.append(f"{r['grid']:26s} {str(r['cell']):4s} {r['cls']:10s} {P(r['cold64'], 1):>6s} {F(r['g50']):>5s} {P(r['yield_']):>5s} {F(r['fe_med'], 0)+'/'+F(r['fe_p90'], 0):>5s} | "
                     f"{P(r['dyn_solved'], 1):>5s} {P(r['commit1_s']):>5s}|{P(r['commit1_u']):<6s} {P(r['cw1_u']):>4s}->{P(r['cw64_u']):<4s} {F(r['ent1_s'], 2):>5s} {P(r['mono']):>4s} {F(r['flips_w'], 2):>5s} {F(r['syn_osc']):>6s} | "
                     f"{P(r['e3_committed']):>5s}/{P(r['e3_wrong']):>5s}/{P(r['e3_contra']):>5s} | {P(r['cal_top5']):>5s}/{P(r['cal_conf']):<6s} {F(r['cal_ent1'], 2):>5s}")
    txt = "\n".join(lines); (AN / f"corpus_lens_{a.date}.txt").write_text(txt + "\n"); print(txt); print(f"\n-> {out_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
