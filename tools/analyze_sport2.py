# Ledger: SPRINT S2 ANALYZER — written BEFORE any benchmark run (2026-08-21),
# pre-registration in code. Rules exactly as the course-correction entry
# registers them (the launch entry locks them verbatim):
#   M0  fit gate: S0's val MONITOR exact (disjoint train.csv rows) >= .20 at
#       the end of training; else "STRICT regime under-fits at this protocol"
#       -> GEN (generator-pretrained + 1k finetune) becomes the primary LABELED
#       row; STRICT is still reported as-is (never hidden).
#   HEADLINE per arm = best single deterministic prediction exact accuracy on
#       the FULL test set over the registered inference depths t in T_LIST
#       (test-time depth = EqR's D axis, reported with the number).
#       verify-and-vote (k>0) is a SEPARATE labeled number (EqR's B axis).
#   BANDS on the best arm's headline: M1 >= .50 (mechanisms work; gates
#       scale-up) | M2 >= .85 (TRM-class at ~10-100x fewer params: paper 1's
#       frontier claim) | M3 >= .95 (EqR-class, stretch).
#   REGISTERED PREDICTIONS (from the prior cell's instruments — H-37's data
#       and the cell-1 depth sweep — locked before these arms run):
#       P1 RI (S1) raises verified multi-init hits vs S0 (H-37: RI pays under
#          multi-init deployment); P2 inference depth raises exact on EVERY
#          arm (t=T -> 64 -> 256 non-decreasing) and training depth (S3/S4)
#          raises cold exact at matched t (depth-limitation); P3 S6 == S0
#          within 1pp (exact S9 already covers digit permutation);
#       P4 priced (S0) vs plain (S5) within 2pp on exact (Law 4: the dividend
#          is TRANSFER-specific; CSP has no unseen families) while the throat
#          differs ~100x — priced > plain by >2pp would be a NEW finding;
#       P5 the combined recipe S7 >= max(S1, S2, S3).
#   BRANCH: best arm < M1 after depth + RI + breadth => the 3-adic/dyadic
#       risk is live => box-aligned representation before further scale.
"""
  .venv/bin/python tools/analyze_sport2.py            # -> runs/analysis/sport2_verdict.txt
  .venv/bin/python tools/analyze_sport2.py --selftest
"""
from __future__ import annotations
import json, os, sys, tempfile, contextlib, io
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(os.environ.get("QHRRN_RUNS", ROOT / "runs"))
OUT = RUNS / "analysis" / "sport2_verdict.txt"
TAG = os.environ.get("R_TAG", "sport2")
ARMS = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"]
T_LIST = [6, 64, 256]
LINES = []
def say(s=""): LINES.append(str(s)); print(s)

def jload(p):
    p = Path(p); return json.loads(p.read_text()) if p.exists() else None

def full(arm, t):   return jload(RUNS / f"sxeval_p{TAG}{arm}" / f"full_t{t}" / "summary_all.json")
def strat(arm, t):  return jload(RUNS / f"sxeval_p{TAG}{arm}" / f"strat_t{t}" / "summary_all.json")

def val_monitor(arm):
    p = RUNS / f"pretrain{TAG}_{arm}" / "metrics.jsonl"
    if not p.exists(): return None
    last = None
    for l in p.read_text().splitlines():
        try: r = json.loads(l)
        except Exception: continue
        if "val" in r: last = r["val"]
    return None if not last else last["val_exact"] / max(last["val_total"], 1)

def probe(arm):
    p = RUNS / f"sudprobe_p{TAG}{arm}" / "results.jsonl"
    if not p.exists(): return None
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    if not rows: return None
    k = max(r.get("multi_init_k", 1) for r in rows)
    return dict(n=len(rows), solve=np.mean([r["solved"] for r in rows]),
                ret=np.mean([r["gt_retention"] for r in rows]),
                mi=np.mean([r["multi_init_hits"] / k for r in rows]),
                viol=float(np.median([r["violations"] for r in rows])),
                cells=float(np.median([r["cells_correct"] for r in rows])))

def analyze(arms=ARMS):
    LINES.clear(); V = {}
    say("=" * 100); say("SPRINT S2 VERDICT — Sudoku-Extreme (full 423k test, exact accuracy; rules registered 2026-08-21)"); say("=" * 100)
    say(); say("SECTION 1 — per-arm table: full-test exact at each inference depth | strat-512 vote (k) | val monitor | instruments")
    H = {}
    for arm in arms:
        fs = {t: full(arm, t) for t in T_LIST}
        ss = {t: strat(arm, t) for t in T_LIST}
        vm = val_monitor(arm); pr = probe(arm)
        acc = {t: (fs[t]["exact_acc"] if fs[t] else None) for t in T_LIST}
        have = {t: a for t, a in acc.items() if a is not None}
        if have:
            tb = max(have, key=have.get); H[arm] = (have[tb], tb)
        vote = {t: (ss[t]["exact_acc_vote"] if ss[t] else None) for t in T_LIST}
        kk = next((ss[t]["k_init"] for t in T_LIST if ss[t]), None)
        say(f'  {arm:4s} full ' + " ".join(f't{t}:{(f"{a:.4f}" if a is not None else "  -   "):>6s}' for t, a in acc.items())
            + f' | vote(k={kk}) ' + " ".join(f't{t}:{(f"{v:.3f}" if v is not None else " -  "):>5s}' for t, v in vote.items())
            + f' | val {(f"{vm:.2f}" if vm is not None else "-"):>4s}'
            + (f' | probe solve {pr["solve"]:.2f} ret {pr["ret"]:.2f} mi {pr["mi"]:.2f} viol {pr["viol"]:.0f} cells {pr["cells"]:.0f}' if pr else ' | probe -'))
    # M0
    say(); vm0 = val_monitor("S0")
    if vm0 is None: say("M0: (no S0 monitor) "); V["M0"] = "NO-DATA"
    else:
        V["M0"] = "PASS" if vm0 >= .20 else "FAIL"
        say(f'M0 fit gate: S0 val monitor {vm0:.2f} -> {V["M0"]}' + ("" if vm0 >= .20 else " — STRICT under-fits at this protocol: GEN becomes the primary labeled row; STRICT still reported"))
    # headline + bands
    say()
    if not H: say("HEADLINE: no full-test results"); V["BAND"] = "NO-DATA"
    else:
        best = max(H, key=lambda a: H[a][0]); acc, tb = H[best]
        band = "M3" if acc >= .95 else "M2" if acc >= .85 else "M1" if acc >= .50 else "BELOW-M1"
        V["BAND"] = band; V["BEST"] = f"{best}@t{tb}={acc:.4f}"
        say(f'HEADLINE: best arm {best} at inference depth t={tb}: exact {acc:.4f} on the full test set -> BAND {band}'
            + {"M3": " (EqR-class)", "M2": " (TRM-class at ~10-100x fewer params — paper-1 frontier claim)", "M1": " (mechanisms work; scale-up gated open)", "BELOW-M1": " (below M1 — branch rule below)"}[band])
        if band == "BELOW-M1": say("  BRANCH: depth + RI + breadth did not reach M1 -> the 3-adic/dyadic pooling risk is LIVE -> box-aligned representation before further scale")
    # predictions
    say(); say("REGISTERED PREDICTIONS (Δ in pp of full-test exact unless noted)")
    def acc_at(arm, t): f = full(arm, t); return None if not f else f["exact_acc"]
    def pr_mi(arm): p = probe(arm); return None if not p else p["mi"]
    # P1 RI raises verified multi-init hits (probe mi rate) vs S0
    m0, m1 = pr_mi("S0"), pr_mi("S1")
    if m0 is not None and m1 is not None:
        V["P1"] = "HOLDS" if m1 > m0 + .02 else ("FLAT" if abs(m1 - m0) <= .02 else "REVERSED")
        say(f'  P1 (H-37: RI raises multi-init hits): S0 {m0:.3f} -> S1 {m1:.3f}  {V["P1"]}')
    else: V["P1"] = "NO-DATA"; say("  P1: no probe data")
    # P2 depth: inference depth non-decreasing on every arm; training depth helps at matched t
    mono = [];
    for arm in arms:   # over the depths that HAVE full-test results (>=2 points)
        a_ = [acc_at(arm, t) for t in T_LIST]; a_ = [x for x in a_ if x is not None]
        if len(a_) >= 2: mono.append(all(a_[i] <= a_[i + 1] + .005 for i in range(len(a_) - 1)))
    tdep = [(acc_at(a, 64), acc_at("S0", 64)) for a in ("S3", "S4")]
    tdep = [(x, y) for x, y in tdep if x is not None and y is not None]
    if mono or tdep:
        V["P2"] = f"inference-depth monotone {sum(mono)}/{len(mono)} arms; training depth Δ@t64 " + ", ".join(f"{(x-y)*100:+.1f}" for x, y in tdep)
        say(f'  P2 (depth-limitation): {V["P2"]}')
    else: V["P2"] = "NO-DATA"; say("  P2: no data")
    # P3 S6 == S0
    a0, a6 = acc_at("S0", 64), acc_at("S6", 64)
    if a0 is not None and a6 is not None:
        V["P3"] = "HOLDS" if abs(a6 - a0) <= .01 else "VIOLATED"; say(f'  P3 (S9 covers digit aug): S6−S0 @t64 {(a6-a0)*100:+.2f}pp  {V["P3"]}')
    else: V["P3"] = "NO-DATA"; say("  P3: no data")
    # P4 priced vs plain
    a5 = acc_at("S5", 64)
    if a0 is not None and a5 is not None:
        d = (a0 - a5) * 100
        V["P4"] = "WITHIN-2pp" if abs(d) <= 2 else ("PRICED>PLAIN (NEW)" if d > 2 else "PLAIN>PRICED")
        say(f'  P4 (Law 4 on CSP): priced−plain @t64 {d:+.2f}pp  {V["P4"]}')
    else: V["P4"] = "NO-DATA"; say("  P4: no data")
    # P5 combined
    a7 = acc_at("S7", 64); parts = [acc_at(a, 64) for a in ("S1", "S2", "S3")]
    if a7 is not None and all(x is not None for x in parts):
        V["P5"] = "HOLDS" if a7 >= max(parts) - .005 else "VIOLATED"; say(f'  P5 (combined recipe): S7 {a7:.4f} vs max(S1,S2,S3) {max(parts):.4f}  {V["P5"]}')
    else: V["P5"] = "NO-DATA"; say("  P5: no data")
    say(); say("SUMMARY: " + "  ".join(f"{k}={v}" for k, v in V.items()))
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text("\n".join(LINES) + "\n"); say(f"artifact -> {OUT}")
    return V

def _mk(root, arm, full_acc, strat_vote, val, mi):
    for t, a in full_acc.items():
        d = root / f"sxeval_p{TAG}{arm}" / f"full_t{t}"; d.mkdir(parents=True, exist_ok=True)
        (d / "summary_all.json").write_text(json.dumps(dict(exact_acc=a, exact_acc_vote=a, k_init=0, n=423000)))
        d2 = root / f"sxeval_p{TAG}{arm}" / f"strat_t{t}"; d2.mkdir(parents=True, exist_ok=True)
        (d2 / "summary_all.json").write_text(json.dumps(dict(exact_acc=a, exact_acc_vote=strat_vote, k_init=16, n=512)))
    d = root / f"pretrain{TAG}_{arm}"; d.mkdir(parents=True, exist_ok=True)
    (d / "metrics.jsonl").write_text(json.dumps(dict(step=50, loss=1.0)) + "\n" + json.dumps(dict(val=dict(val_exact=int(val * 64), val_total=64, step=20000))) + "\n")
    d = root / f"sudprobe_p{TAG}{arm}"; d.mkdir(parents=True, exist_ok=True)
    rows = [dict(task=f"x{i}", solved=(i < 256), gt_retention=True, multi_init_hits=int(mi * 16), multi_init_k=16, violations=0 if i < 256 else 3, cells_correct=81 if i < 256 else 70) for i in range(512)]
    (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

def selftest():
    checks = []
    def run(build):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); build(root); globals()["RUNS"] = root; globals()["OUT"] = root / "analysis" / "v.txt"
            with contextlib.redirect_stdout(io.StringIO()): return analyze()
    def base(root, s0=(.30, .55, .60), s1=(.32, .58, .63), s3=(.40, .70, .72), s5=(.30, .54, .60), s6=(.30, .55, .60), s7=(.45, .88, .90), val=.5, mi0=.10, mi1=.30):
        _mk(root, "S0", dict(zip(T_LIST, s0)), .7, val, mi0); _mk(root, "S1", dict(zip(T_LIST, s1)), .8, val, mi1)
        _mk(root, "S2", dict(zip(T_LIST, s0)), .7, val, mi0); _mk(root, "S3", dict(zip(T_LIST, s3)), .8, val, mi0)
        _mk(root, "S4", dict(zip(T_LIST, s3)), .8, val, mi0); _mk(root, "S5", dict(zip(T_LIST, s5)), .7, val, mi0)
        _mk(root, "S6", dict(zip(T_LIST, s6)), .7, val, mi0); _mk(root, "S7", dict(zip(T_LIST, s7)), .9, val, mi1)
    v = run(base)
    checks += [("M2 band", v["BAND"] == "M2" and v["BEST"].startswith("S7@t256")), ("M0 pass", v["M0"] == "PASS"),
               ("P1 holds", v["P1"] == "HOLDS"), ("P3 holds", v["P3"] == "HOLDS"), ("P4 within", v["P4"] == "WITHIN-2pp"), ("P5 holds", v["P5"] == "HOLDS"),
               ("P2 monotone 8/8", v["P2"].startswith("inference-depth monotone 8/8"))]
    v = run(lambda r: base(r, s0=(.20, .35, .40), s1=(.22, .38, .42), s3=(.20, .40, .45), s5=(.20, .34, .40), s6=(.20, .35, .40), s7=(.20, .40, .45), val=.1))
    checks += [("below M1 + branch", v["BAND"] == "BELOW-M1"), ("M0 fail", v["M0"] == "FAIL")]
    v = run(lambda r: base(r, s7=(.90, .96, .97)))
    checks.append(("M3 band", v["BAND"] == "M3"))
    v = run(lambda r: base(r, s5=(.30, .50, .55)))
    checks.append(("P4 priced>plain new", v["P4"] == "PRICED>PLAIN (NEW)"))
    v = run(lambda r: base(r, s6=(.30, .60, .65)))
    checks.append(("P3 violated", v["P3"] == "VIOLATED"))
    n = sum(1 for _, o in checks if o)
    for name, o in checks: print(f"  {'PASS' if o else 'FAIL'}  {name}")
    print(f"selftest: {n}/{len(checks)}"); return n == len(checks)

if __name__ == "__main__":
    sys.exit(0 if selftest() else 1) if "--selftest" in sys.argv else analyze()
