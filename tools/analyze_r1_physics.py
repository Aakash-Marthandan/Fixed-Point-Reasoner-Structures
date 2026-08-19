# Ledger: RUNG-1 PHYSICS PASS (2026-08-19) — the rung-1 data (d64/T6@53,333,
# A4 x3 / A3 x2 / A5 x1) re-read through the RG / holography / code-geometry /
# information lenses against the seeded d48 anchor (rung 0) and the wave-2 d64
# n=1 cells. Companion to tools/analyze_r1.py (the registered verdict R1-1..R1-6).
# Reads disk only; every number here is computed, none hand-typed.
"""
  .venv/bin/python tools/analyze_r1_physics.py   # -> runs/analysis/r1_physics_20260819.txt
"""
from __future__ import annotations
import json, pickle, re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; RUNS = ROOT / "runs"
OUT = RUNS / "analysis" / "r1_physics_20260819.txt"
KNEE = np.array([.69, .14, .085, .035, .048]); FREE = np.array([.76, .18, .045, .012, .003])
FLOORS = np.array([350, 75, 50, 15, 30.0])
EPS = ["0.05", "0.1", "0.2", "0.4"]
LINES = []
def say(s=""): LINES.append(s); print(s)

# cells: label -> (lad dir tag, rg tags, pretrain dir, group, d, steps)
R1 = {f"A4s{s}": (f"pr1A4s{s}", [f"pr1A4s{s}"], f"pretrainr1_A4s{s}", "floors+NI", 64, 53333) for s in (0,1,2)}
R1.update({f"A3s{s}": (f"pr1A3s{s}", [f"pr1A3s{s}"], f"pretrainr1_A3s{s}", "plain", 64, 53333) for s in (0,1)})
R1["A5s0"] = ("pr1A5s0", ["pr1A5s0"], "pretrainr1_A5s0", "global+NI", 64, 53333)
R0 = {f"*A4s{s}": (f"p13fA4s{s}", [f"p13fA4s{s}"], f"pretrain13f_A4s{s}", "floors+NI", 48, 40000) for s in (0,1,2)}
R0.update({f"*A3s{s}": (f"p13fA3s{s}", [f"p13fA3s{s}"], f"pretrain13f_A3s{s}", "plain", 48, 40000) for s in (0,1,2)})
R0.update({f"*A2s{s}": (f"p13fA2s{s}", [f"p13fA2s{s}"], f"pretrain13f_A2s{s}", "global", 48, 40000) for s in (0,1,2)})
R0.update({f"*A1s{s}": (f"p13fA1s{s}", [f"p13fA1s{s}"], f"pretrain13f_A1s{s}", "floors", 48, 40000) for s in (0,1,2)})
W2 = {"C53": ("p13C53", ["p13C53"], "pretrain13_C53", "global(w2)", 64, 53333),
      "C80": ("p13C80", ["p13C80"], "pretrain13_C80", "global(w2)", 64, 80000),
      "Dfloor": ("p13Dfloor", ["p13Dfloor"], "pretrain13_Dfloor", "floors(w2)", 64, 53333),
      "Dri": ("p13Dri", ["p13Dri"], "pretrain13_Dri", "RI(w2)", 64, 53333),
      "B": ("p13B", ["p13B"], "pretrain13_B", "B(w2)", 64, 53333)}

def rows(prefix, tag):
    p = RUNS / f"{prefix}_{tag}" / "results.jsonl"
    if not p.exists(): return None
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

def fam(task): return re.sub(r"\d+$", "", task.replace("ca_", ""))

def vh_stats(tag):
    rs = rows("lad", tag)
    if not rs: return None
    N = S = {}; 
    q_all = [(r["task"], qi, q) for r in rs for qi, q in enumerate(r["queries"])]
    ret = np.array([q["gt_retention"] for _,_,q in q_all]); ex = np.array([q["exact_T"] for _,_,q in q_all])
    lad = {e: np.array([q["q_ladder"][e] for _,_,q in q_all]) for e in EPS}
    I = np.array([q["I_s"] for _,_,q in q_all]); spec = np.median(I, axis=0)
    radii = []
    for _,_,q in q_all:
        if q["gt_retention"]:
            surv = [float(e) for e in EPS if q["q_ladder"][e]]; radii.append(max(surv) if surv else 0.0)
    fuzz = sum(1 for _,_,q in q_all if any(q["q_ladder"][EPS[i]] < q["q_ladder"][EPS[i+1]] for i in range(3)))
    fams = {}
    for t, qi, q in q_all:
        f = fam(t); fams.setdefault(f, [0,0,0]); fams[f][0] += q["gt_retention"]; fams[f][1] += q["exact_T"]; fams[f][2] += 1
    return dict(n=len(q_all), N=int(ret.sum()), ex=int(ex.sum()), S={e:int(lad[e].sum()) for e in EPS},
                ex_and_ret=int((ex & ret).sum()), ex_not_ret=int((ex & (1-ret)).sum()),
                I_med=float(np.median(I.sum(1))), spec=spec, prof=spec/spec.sum(), rbar=float(np.mean(radii)) if radii else 0.0,
                fuzz=fuzz, fams=fams, ret_keys={(t,qi): bool(q["gt_retention"]) for t,qi,q in q_all})

def rg_stats(tags):
    out = dict(ret=0, S2=0, n=0, parts={})
    for tag in tags:
        for pref in ("ladrg", "ladrgb"):
            rs = rows(pref, tag)
            if not rs: continue
            r_ = sum(q["gt_retention"] for r in rs for q in r["queries"]); s2 = sum(q["q_ladder"]["0.2"] for r in rs for q in r["queries"]); n = sum(len(r["queries"]) for r in rs)
            out["ret"] += r_; out["S2"] += s2; out["n"] += n; out["parts"][pref] = (r_, s2, n)
    return out if out["n"] else None

def rt_stats(tag):
    rs = rows("ladrt", tag)
    if not rs: return None
    return dict(ret=sum(q["gt_retention"] for r in rs for q in r["queries"]), S2=sum(q["q_ladder"]["0.2"] for r in rs for q in r["queries"]), n=sum(len(r["queries"]) for r in rs))

def eta_of(pdir):
    p = RUNS / pdir / "ckpt_latest.pkl"
    if not p.exists(): return None
    ck = pickle.loads(p.read_bytes()); e = ck["state"]["model"]["eq"]["eta"]
    return float(1/(1+np.exp(-float(np.asarray(e)))))

def e1e3_stats(tag):
    rs = rows("e1e3", tag)
    if not rs: return None
    conv = ws = tot = 0; nd = []; le = 0
    for r in rs:
        for e3 in r["e3"]:
            tot += 1
            if e3["converged_at"] is not None:
                conv += 1; ws += (not e3["limit_exact"])
            le += bool(e3["limit_exact"]); nd.append(e3["n_distinct"])
    nd = np.array(nd)
    return dict(tot=tot, conv=conv, ws=ws, limit_exact=le, nd1=int((nd==1).sum()), nd_ge13=int((nd>=13).sum()), nd_med=float(np.median(nd)))

def main():
    say("=" * 100); say("RUNG-1 PHYSICS PASS — d64/T6@53,333 (A4 x3, A3 x2, A5 x1) vs the seeded d48 anchor (rung 0) and the wave-2 d64 n=1 cells"); say("(disk-only; seed means where n>=2; * = d48 anchor; w2 = 2026-08-13/14 wave-2 cells)"); say("=" * 100)
    cells = {}; 
    for D in (R1, R0, W2):
        for lbl, (lt, rgt, pdir, grp, d, steps) in D.items():
            v = vh_stats(lt); 
            if not v: continue
            cells[lbl] = dict(v=v, rg=rg_stats(rgt), rt=rt_stats(lt), eta=eta_of(pdir), e13=e1e3_stats(lt), grp=grp, d=d, steps=steps)

    # ---------------- A. code geometry: the (N, rbar) plane + survival ----------------
    say(); say("A. CODE GEOMETRY — codebook size N=S(0), survival S(eps) on val-hard (144 pairs), rbar = mean max-survived eps, exact@T (quench reachability), rg-96")
    say(f'  {"cell":8s} {"grp":11s} {"d":>3s} {"steps":>6s} | {"N":>3s} {"S.05":>4s} {"S.1":>4s} {"S.2":>4s} {"S.4":>4s} {"rbar":>5s} {"S.2/N":>5s} {"S.4/N":>5s} | {"ex":>3s} {"ex&ret":>6s} {"ex/N":>5s} | {"rg96":>4s} {"rgS2":>4s} {"rg48":>4s} {"rb48":>4s} | {"rt48":>4s}')
    for lbl, c in cells.items():
        v, g, rt = c["v"], c["rg"], c["rt"]
        rg48 = g["parts"].get("ladrg", (0,0,0))[0] if g else 0; rb48 = g["parts"].get("ladrgb", (0,0,0))[0] if g else 0
        say(f'  {lbl:8s} {c["grp"]:11s} {c["d"]:>3d} {c["steps"]:>6d} | {v["N"]:>3d} {v["S"]["0.05"]:>4d} {v["S"]["0.1"]:>4d} {v["S"]["0.2"]:>4d} {v["S"]["0.4"]:>4d} {v["rbar"]:>5.2f} {v["S"]["0.2"]/max(v["N"],1):>5.2f} {v["S"]["0.4"]/max(v["N"],1):>5.2f} | {v["ex"]:>3d} {v["ex_and_ret"]:>6d} {v["ex"]/max(v["N"],1):>5.2f} | {g["ret"] if g else 0:>4d} {g["S2"] if g else 0:>4d} {rg48:>4d} {rb48:>4d} | {rt["ret"] if rt else "-":>4}')
    def mean_of(keys, f):
        xs = [f(cells[k]) for k in keys if k in cells]; return (float(np.mean(xs)), float(np.max(xs)-np.min(xs)), len(xs)) if xs else (float("nan"),0,0)
    say(); say("  seed means (n, spread) — the width-ceiling statement in the (N, rbar) plane:")
    for name, keys in (("d64 A4 floors+NI", ["A4s0","A4s1","A4s2"]), ("d48 A4 floors+NI", ["*A4s0","*A4s1","*A4s2"]), ("d64 A3 plain", ["A3s0","A3s1"]), ("d48 A3 plain", ["*A3s0","*A3s1","*A3s2"]), ("d48 A2 global", ["*A2s0","*A2s1","*A2s2"]), ("d48 A1 floors", ["*A1s0","*A1s1","*A1s2"])):
        N = mean_of(keys, lambda c: c["v"]["N"]); rb = mean_of(keys, lambda c: c["v"]["rbar"]); s4 = mean_of(keys, lambda c: c["v"]["S"]["0.4"]); ex = mean_of(keys, lambda c: c["v"]["ex"]); rg = mean_of(keys, lambda c: c["rg"]["ret"] if c["rg"] else 0); I = mean_of(keys, lambda c: c["v"]["I_med"])
        say(f'   {name:18s} N {N[0]:5.1f}±{N[1]:<4.0f} rbar {rb[0]:.3f}±{rb[1]:<5.3f} S(.4) {s4[0]:5.1f}±{s4[1]:<3.0f} exact {ex[0]:5.1f}±{ex[1]:<3.0f} rg96 {rg[0]:5.1f}±{rg[1]:<3.0f} I_med {I[0]:8.0f} (n={N[2]})')
    say("  Wave-2 d64 n=1 comparators (rg = rg-48 only there): " + "; ".join(f'{k}: N {cells[k]["v"]["N"]} rbar {cells[k]["v"]["rbar"]:.2f} S.4 {cells[k]["v"]["S"]["0.4"]} ex {cells[k]["v"]["ex"]} rg48 {cells[k]["rg"]["ret"] if cells[k]["rg"] else "-"} I {cells[k]["v"]["I_med"]:.0f}' for k in ("C53","C80","Dfloor","Dri","B") if k in cells))

    # ---------------- B. families ----------------
    say(); say("B. FAMILIES — val-hard retention per family (9 pairs each): vacancy floor (Copy/ExtractObjects/Count), Center width-casualty (H-35), and WHERE the d64 deficit lives")
    fams = sorted(cells["A4s0"]["v"]["fams"].keys())
    def fam_mean(keys, f): return [np.mean([cells[k]["v"]["fams"][f][0] for k in keys if k in cells]) for _ in [0]][0]
    say(f'  {"family":18s} {"d48A4":>6s} {"d64A4":>6s} {"Δ":>5s} | {"d64A5":>6s} {"d48A3":>6s} {"d64A3":>6s} | {"C53":>4s} {"C80":>4s} {"Dfl":>4s}   (d48A4/d64A4 = seed means of 3; exact counts in brackets for A4)')
    losses = []
    for f in fams:
        a48 = fam_mean(["*A4s0","*A4s1","*A4s2"], f); a64 = fam_mean(["A4s0","A4s1","A4s2"], f)
        ex48 = np.mean([cells[k]["v"]["fams"][f][1] for k in ["*A4s0","*A4s1","*A4s2"]]); ex64 = np.mean([cells[k]["v"]["fams"][f][1] for k in ["A4s0","A4s1","A4s2"]])
        a5 = cells["A5s0"]["v"]["fams"][f][0]; p48 = fam_mean(["*A3s0","*A3s1","*A3s2"], f); p64 = fam_mean(["A3s0","A3s1"], f)
        w = lambda k: cells[k]["v"]["fams"][f][0] if k in cells else -1
        losses.append((a64-a48, f))
        say(f'  {f:18s} {a48:6.1f} {a64:6.1f} {a64-a48:+5.1f} | {a5:6d} {p48:6.1f} {p64:6.1f} | {w("C53"):>4d} {w("C80"):>4d} {w("Dfloor"):>4d}   [ex {ex48:.1f}->{ex64:.1f}]')
    losses.sort()
    say("  largest d64 losses (A4, seed-mean): " + ", ".join(f"{f} {d:+.1f}" for d, f in losses[:5]) + "  | gains: " + ", ".join(f"{f} {d:+.1f}" for d, f in losses[-3:] if d > 0))
    vac = {f: [cells[k]["v"]["fams"][f][0] for k in ("A4s0","A4s1","A4s2","A3s0","A3s1","A5s0")] for f in ("Copy","ExtractObjects","Count")}
    say("  VACANCY FLOOR at seeded d64 (per-arm retention/9): " + "; ".join(f"{f} {v}" for f, v in vac.items()))
    say("  CENTER (H-35): d48 A4 " + str([cells[k]["v"]["fams"]["Center"][0] for k in ("*A4s0","*A4s1","*A4s2")]) + " -> d64 A4 " + str([cells[k]["v"]["fams"]["Center"][0] for k in ("A4s0","A4s1","A4s2")]) + f"; d64 A5 {cells['A5s0']['v']['fams']['Center'][0]}; d64 plain {[cells[k]['v']['fams']['Center'][0] for k in ('A3s0','A3s1')]}")

    # ---------------- C. spectra / RG profile ----------------
    say(); say("C. RG PROFILE — per-cut median flux I_s (nats) and normalized profile; knee/free distances; UV share; floors as a fixed point")
    say(f'  {"cell":8s} {"I_med":>7s}  {"I_s medians (s0..s4)":40s} {"profile":40s} {"dknee":>6s} {"dfree":>6s} {"UV":>5s} {"IR(s3+s4)":>9s}')
    say(f'  {"knee":8s} {"":>7s}  {"":40s} {np.array2string(KNEE, precision=3):40s} {"-":>6s} {"-":>6s} {KNEE[0]:>5.2f} {KNEE[3:].sum():>9.3f}')
    say(f'  {"free":8s} {"":>7s}  {"":40s} {np.array2string(FREE, precision=3):40s} {"-":>6s} {"-":>6s} {FREE[0]:>5.2f} {FREE[3:].sum():>9.3f}')
    fl = FLOORS/FLOORS.sum(); say(f'  {"floors":8s} {FLOORS.sum():>7.0f}  {np.array2string(FLOORS, precision=0):40s} {np.array2string(fl, precision=3):40s} {np.abs(fl-KNEE).sum():>6.3f} {np.abs(fl-FREE).sum():>6.3f} {fl[0]:>5.2f} {fl[3:].sum():>9.3f}')
    for lbl in list(R1) + ["*A4s0","*A4s1","*A4s2","*A2s0","*A2s1","*A2s2","*A1s0","*A3s0","C53","C80","Dfloor","Dri"]:
        if lbl not in cells: continue
        v = cells[lbl]["v"]; p = v["prof"]
        say(f'  {lbl:8s} {v["I_med"]:>7.0f}  {np.array2string(v["spec"], precision=0, max_line_width=80):40s} {np.array2string(p, precision=3):40s} {np.abs(p-KNEE).sum():>6.3f} {np.abs(p-FREE).sum():>6.3f} {p[0]:>5.2f} {p[3:].sum():>9.3f}')
    def spec_mean(keys): return np.mean([cells[k]["v"]["spec"] for k in keys if k in cells], axis=0)
    a4_64 = spec_mean(["A4s0","A4s1","A4s2"]); a4_48 = spec_mean(["*A4s0","*A4s1","*A4s2"]); a5 = cells["A5s0"]["v"]["spec"]; a2_48 = spec_mean(["*A2s0","*A2s1","*A2s2"]); a1_48 = spec_mean(["*A1s0","*A1s1","*A1s2"])
    say(); say("  per-scale deltas (nats; + = more flux):")
    say(f'   width at fixed arm, A4 d64−d48 (n=3 each):        {np.array2string(a4_64-a4_48, precision=1)}  total {a4_64.sum()-a4_48.sum():+.0f}  — the throat decline is {"UV-led" if abs((a4_64-a4_48)[0]) > abs((a4_64-a4_48)[1:]).max() else "not UV-led"}')
    say(f'   floors at d64 with NI, A4−A5 (n=3 vs 1):             {np.array2string(a4_64-a5, precision=1)}  total {a4_64.sum()-a5.sum():+.0f}  vs rung-0 floors cost A1−A2 (d48): {np.array2string(a1_48-a2_48, precision=1)} total {a1_48.sum()-a2_48.sum():+.0f}')
    say(f'   A4 d64 spectrum vs floor vector:                      {np.array2string(a4_64, precision=0)} vs {np.array2string(FLOORS, precision=0)} -> at/above floors on {int((a4_64 >= FLOORS-1).sum())}/5 cuts')
    say(f'   A5 d64 (global+NI) vs floor vector:                   {np.array2string(a5, precision=0)} -> below floors on cuts {[i for i in range(5) if a5[i] < FLOORS[i]-1]} (sub-floor compression, as H-32 mechanism predicts)')
    if "C53" in cells:
        c53 = cells["C53"]["v"]["spec"]; say(f'   NI tax at d64, A5−C53 (global+NI vs global; n=1 each):    {np.array2string(a5-c53, precision=1)}  total {a5.sum()-c53.sum():+.0f}   (rung-0 clean NI tax A4−A1 at d48: {np.array2string(a4_48-a1_48, precision=1)} total {a4_48.sum()-a1_48.sum():+.0f})')
    a3_64 = spec_mean(["A3s0","A3s1"]); a3_48 = spec_mean(["*A3s0","*A3s1","*A3s2"])
    say(f'   plain inflation with width, A3 d64−d48:               total {a3_64.sum():.0f} vs {a3_48.sum():.0f} = {a3_64.sum()/a3_48.sum():.2f}x (free models inflate with width; priced compress)')

    # ---------------- D. flow constant ----------------
    say(); say("D. FLOW CONSTANT eta (learned damping) — the RG velocity across cells")
    for lbl in list(cells):
        if cells[lbl]["eta"] is not None:
            say(f'  {lbl:8s} {cells[lbl]["grp"]:11s} d{cells[lbl]["d"]} steps {cells[lbl]["steps"]:>6d}  eta {cells[lbl]["eta"]:.3f}   N {cells[lbl]["v"]["N"]:>3d}  S.4 {cells[lbl]["v"]["S"]["0.4"]:>3d}')
    pairs = [(cells[k]["eta"], cells[k]["v"]["N"]) for k in cells if cells[k]["eta"] is not None and cells[k]["grp"] not in ("B(w2)",)]
    if len(pairs) > 4:
        e, n = np.array(pairs).T; rho = np.corrcoef(np.argsort(np.argsort(e)), np.argsort(np.argsort(n)))[0,1]
        say(f'  Spearman(eta, N) over {len(pairs)} cells (all arms, both widths, w2): {rho:+.2f}   [curiosity: does faster flow shed codewords? confounded with budget/width/arm — correlational only]')

    # ---------------- E. E1/E3 ----------------
    say(); say("E. E1/E3 on the A4 arms — convergence, wrong-stable limits, endpoint condensation (n_distinct), limit-exact")
    for lbl in ("A4s0","A4s1","A4s2","*A4s0","*A4s1","*A4s2","*A1s0","*A1s1","*A1s2"):
        c = cells.get(lbl); 
        if c and c["e13"]:
            e = c["e13"]; say(f'  {lbl:8s} conv {e["conv"]:>3d}/{e["tot"]}  wrong-stable {e["ws"]:>3d} ({e["ws"]/max(e["conv"],1):.2f} of converged)  limit_exact {e["limit_exact"]:>3d}  nd=1 {e["nd1"]:>3d}  nd>=13 {e["nd_ge13"]:>3d}  nd_med {e["nd_med"]:.0f}')

    # ---------------- F. dissociation ----------------
    say(); say("F. EXISTENCE vs REACHABILITY — exact@T / N (the quench reaches what fraction of the codebook) and exact-but-not-retained (decode-without-hold)")
    for lbl in list(R1) + ["*A4s0","*A4s1","*A4s2","*A3s0","*A3s1","*A3s2","C53","C80"]:
        if lbl not in cells: continue
        v = cells[lbl]["v"]; say(f'  {lbl:8s} N {v["N"]:>3d} exact {v["ex"]:>3d} ratio {v["ex"]/max(v["N"],1):.2f}  exact&retained {v["ex_and_ret"]:>3d}  exact&NOT-retained {v["ex_not_ret"]:>2d}  fuzz(non-monotone ladder rows) {v["fuzz"]:>2d}')

    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text("\n".join(LINES) + "\n"); say(); say(f"artifact -> {OUT.relative_to(ROOT)}")

if __name__ == "__main__":
    main()
