# Ledger: CHAMPION sportC1 ANALYZER — written BEFORE any sportC1 run (2026-09-02);
# the launch registration locks these rules verbatim (Plan_2026-09-02_Champion_sportC1
# §5/§9/§10/§12). Arms: A0 A1 (d128 champion recipe, controls) · B0 B1 (+ z-norm,
# the H-50 twin-arm bridge) · R0 (our cell in the FIELD's optimizer regime) · X0
# (the field-recipe baseline, TRM/EqR cell + SOT + ACT) · X0n (X0 without ACT;
# descriptive only). Every number is read at the arm's HEADLINE weights (raw on
# A/B; EMA on R0/X0/X0n) from the val-selected checkpoint unless named "final".
#   INTEGRITY (before any value is read; a breach WITHHOLDS the verdict): every
#     full-test summary n == 422,786; every 20k scan n == 20,000 with unique idx,
#     identical idx sets across the arms that have one (paired-comparable),
#     protocol fields (t 64, k 128, subsample seed 20260822), vote identity
#     exact_acc_vote == mean(cold | draw-hit) on the records.
#   NOISE: FNC1 = max(|b1(A0) - b1(A1)|, .02) over a clean A pair (else .02);
#          CNC1 = max(|vcold(A0) - vcold(A1)|, .01) likewise (vcold = vsel cold).
#   CLEAN (native arms) := not STOPPED AND retfm >= .9 AND end-CE >= .02 AND
#          census(vsel, t=64) exploded_frac <= .02; X0/X0n: not STOPPED AND census <= .02.
#   R-C1-0 REGIME: per native arm — end train ce_in >= .02 AND |vcold - fcold| <= CNC1
#          -> ok; CE < .02 -> MEMORIZED:<arm>; else DRIFT:<arm>. Letter NO-MEMORIZATION
#          iff no arm is MEMORIZED (DRIFT arms named).
#   R-C1-1 SURVIVAL: clean A-arms 2/2 -> RI-CARRIES; 1/2 -> RI-LOTTERY; 0/2 -> RI-DEAD-AT-d128.
#   R-C1-2 STABILIZER (z-norm bridge): both B clean AND mean b1(B) >= mean b1(clean A) - FNC1
#          -> NORM-WORKS; both clean but lower by > FNC1 -> NORM-COSTS; any B unclean ->
#          NORM-FAILS; both clean with no clean A reference -> NORM-WORKS-UNREFERENCED.
#   R-C1-3 CHAMPION: argmax vcold over clean {A0, A1, B0, B1, R0} (ties: higher b1, then
#          name); >= .55 HRM-BEATEN / >= .45 ON-TRACK / < .45 PLATEAU; none clean -> NO-GO.
#   R-C1-4 NATIVE-vs-CANVAS (ladder read): best native vcold vs canvas D4 .3353:
#          >= D4 + CNC1 NATIVE-CARRIES-AT-d128 / within +-CNC1 NATIVE-FLAT / below NATIVE-BELOW-CANVAS-d96.
#   R-C1-5 BREADTH: champion verified@128 >= .8889 -> BREADTH-SCALES-AT-d128 else NARROW-FUNNEL.
#   R-C1-6 PORTFOLIO (labeled): union verified@128 over clean native scans (+ banked canvas
#          C3X/D4 scans when idx-identical) >= .95 -> B-M3-BY-PORTFOLIO else BELOW.
#   R-C1-7 REPRODUCTION: X0 vsel cold @D16 >= .80 REPRODUCED / [.60,.80) PARTIAL / < .60
#          NOT-REPRODUCED; X0 STOPPED -> X0-DEAD. X0@D64, b1/t1r/majority, X0n: descriptive.
#   R-C1-8 REGIME-vs-ARCHITECTURE: d = vcold(R0) - max vcold(clean A/B): >= CNC1 REGIME-EXPLAINS;
#          <= -CNC1 -> REGIME-UNDERTRAINED-AT-50k if R0's headline monitor val rose > .01 over
#          its last 10k steps, else REGIME-HURTS-OUR-CELL; else REGIME-NEUTRAL; R0 unclean -> NO-DATA.
#   PREDICTIONS (plan §12.8) scored HIT/MISS per arm; STABILITY names unclean arms.
"""
  .venv/bin/python tools/analyze_sportC1.py            # -> runs/analysis/sportC1_verdict.txt
  .venv/bin/python tools/analyze_sportC1.py --selftest
"""
from __future__ import annotations
import contextlib, io, json, os, sys, tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(os.environ.get("QHRRN_RUNS", ROOT / "runs"))
OUT = RUNS / "analysis" / "sportC1_verdict.txt"
TAG = "sportC1"
NATIVE = ["A0", "A1", "B0", "B1", "R0"]
FIELD = ["X0", "X0n"]
ARMS = NATIVE + FIELD
D4_COLD = 0.3353                  # canvas D4 full-test cold (rung-2b verdict; the d96 record it replaced)
C3X_V128 = 0.8889                 # canvas C3X 20k verified@128 (B-M2)
N_FULL, N_SCAN = 422786, 20000
SCAN_PROTO = dict(t_total=64, k_init=128, subsample_seed=20260822)
HEADLINE_T = {a: 64 for a in NATIVE} | {"X0": 16, "X0n": 16}
PRED = {  # plan §12.8 (pre-data)
    "A": dict(vcold=(.42, .52), b1=(.40, .55), t1r=(.55, .70), v128=(.55, .72), eta=(.85, .95)),
    "R0": dict(vcold=(.35, .55)),
    "X0": dict(cold16=(.78, .87), cold64=(.85, .94), b1=(.80, .92), t1r=(.90, .99)),
    "X0n": dict(cold16=(.70, .80)),
}
DESC = {"A0": "champion recipe s0 (no norm)", "A1": "champion recipe s1 (no norm)",
        "B0": "A0 + z-norm", "B1": "A1 + z-norm", "R0": "our cell, FIELD regime (+z-norm, EMA)",
        "X0": "FIELD baseline: TRM/EqR cell + SOT + ACT (EMA)", "X0n": "X0 without ACT (descriptive)"}
LINES = []
def say(s=""): LINES.append(str(s)); print(s)
def jload(p):
    p = Path(p); return json.loads(p.read_text()) if p.exists() else None
def fpp(x): return "  -  " if x is None else f"{100*x:6.2f}"
def in_band(x, band): return None if x is None else (band[0] <= x <= band[1])
def hm(x, band): return "n/a" if x is None else ("HIT" if in_band(x, band) else ("ABOVE" if x > band[1] else "MISS-BELOW"))

# ---------- readers (paths = the chain's contract) ----------
def pdir(a): return RUNS / f"pretrain{TAG}_{a}"
def full(a, which, t=None):
    t = HEADLINE_T[a] if t is None else t
    return jload(RUNS / f"sxeval_p{TAG}{a}" / f"full_{which}_t{t}" / "summary_all.json")
def vcold(a):
    s = full(a, "vsel"); return None if not s else s["exact_acc"]
def fcold(a):
    s = full(a, "final"); return None if not s else s["exact_acc"]
def scan(a): return jload(RUNS / f"sxscan_p{TAG}{a}" / "summary_all.json")
def scan_recs(a):
    q = RUNS / f"sxscan_p{TAG}{a}" / "records_all.npz"
    return dict(np.load(q, allow_pickle=True)) if q.exists() else None
def b1(a):
    s = scan(a); return None if not s else s.get("b1_exact")
def v128(a):
    s = scan(a); return None if not s else s.get("vote_at_k", {}).get("128")
def t1r(a):
    s = scan(a); return None if not s else s.get("t1r_at_k", {}).get("128")
def maj(a):
    s = jload(RUNS / f"sxscreen_p{TAG}{a}_vb" / "summary_all.json")
    return None if not s else s.get("majority_vote_at_k", {}).get("128")
def retfm(a):
    s = jload(RUNS / f"sxeval_p{TAG}{a}" / "retfm_t8" / "summary_all.json")
    return None if not s else s["exact_acc"]
def stopped(a):
    p = pdir(a) / "STOPPED.txt"
    return p.read_text().strip().splitlines()[0][:44] if p.exists() else None
def census(a, which="vsel", t=64):
    c = jload(RUNS / f"sxcensus_p{TAG}{a}_{which}" / "census.json")
    if not c: return None
    rows = [r for r in c["rows"] if int(r["t"]) == t]
    return None if not rows else rows[0]["exploded_frac"]
def metrics(a):
    p = pdir(a) / "metrics.jsonl"
    tr, mon = [], []
    if not p.exists(): return tr, mon
    for l in p.read_text().splitlines():
        l = l.strip()
        if not l: continue
        try: r = json.loads(l)
        except Exception: continue
        if "monitor" in r: mon.append(r["monitor"])
        elif "loss" in r: tr.append(r)
    return tr, mon
def end_ce(a):
    tr, _ = metrics(a)
    rows = [r for r in tr if "ce_in" in r]
    return None if not rows else float(rows[-1]["ce_in"])
def last_eta(a):
    _, mon = metrics(a)
    return None if not mon else float(mon[-1].get("eta", float("nan")))
def val_rise_last10k(a):
    """headline monitor val: v(last) - v(last - 10k steps); None when unreadable."""
    _, mon = metrics(a)
    key = "val_t64_ema" if a == "R0" else "val_t64"
    rows = [(int(m["step"]), float(m[key])) for m in mon if key in m and "step" in m]
    if len(rows) < 2: return None
    s_end, v_end = rows[-1]
    prev = [v for s, v in rows if s <= s_end - 10000]
    return None if not prev else v_end - prev[-1]
def clean(a):
    if stopped(a) is not None: return False
    cz = census(a)
    if cz is None or cz > 0.02: return False
    if a in FIELD: return True
    r = retfm(a); ce = end_ce(a)
    return (r is not None and r >= 0.9) and (ce is not None and ce >= 0.02)

# ---------- integrity ----------
def integrity():
    errs = []
    for a in ARMS:
        for which in ("vsel", "final"):
            s = full(a, which)
            if s and s["n"] != N_FULL: errs.append(f"{a} full_{which} n={s['n']}")
    idx_sets = {}
    for a in ARMS:
        s = scan(a)
        if not s: continue
        if s["n"] != N_SCAN: errs.append(f"{a} scan n={s['n']}")
        for k, v in SCAN_PROTO.items():
            if s.get(k) != v: errs.append(f"{a} scan {k}={s.get(k)}!={v}")
        z = scan_recs(a)
        if z is not None:
            if len(np.unique(z["idx"])) != len(z["idx"]): errs.append(f"{a} scan dup idx")
            idx_sets[a] = frozenset(z["idx"].tolist())
            hit = (z["mi_first_hit"] >= 0) | z["cold_exact"].astype(bool)
            if abs(float(hit.mean()) - s["exact_acc_vote"]) > 5e-6: errs.append(f"{a} vote identity breach")
    if len(set(idx_sets.values())) > 1: errs.append("scan idx sets differ across arms")
    return errs

# ---------- the verdict ----------
def analyze():
    LINES.clear(); V = {}
    say("=" * 118); say("CHAMPION sportC1 VERDICT — d128 native round (rules registered 2026-09-02; analyzer pre-data)"); say("=" * 118)
    errs = integrity()
    say("INTEGRITY: " + ("PASS" if not errs else "FAIL: " + " | ".join(errs)))
    if errs:
        V["INTEGRITY"] = "FAIL"; say("VERDICT WITHHELD — integrity breach"); _write(); return V
    V["INTEGRITY"] = "PASS"
    say("\nSECTION 1 — per arm (headline weights; vsel unless 'final'): vcold | fcold | b1 | verified@128 | t1r@128 | majority@128 | retfm | endCE | eta | census(vsel,t64) | status")
    for a in ARMS:
        st = stopped(a); cz = census(a)
        say(f"  {a:3s} {DESC[a]:44s} vcold {fpp(vcold(a))} | fcold {fpp(fcold(a))} | b1 {fpp(b1(a))} | v128 {fpp(v128(a))} | t1r {fpp(t1r(a))}"
            f" | maj {fpp(maj(a))} | retfm {fpp(retfm(a))} | CE {('  -  ' if end_ce(a) is None else f'{end_ce(a):.4f}')} | eta {('-' if last_eta(a) is None else f'{last_eta(a):.3f}')}"
            f" | census {fpp(cz)} | {('STOPPED: ' + st) if st else ('clean' if clean(a) else 'UNCLEAN')}")
    unclean = [a for a in ARMS if not clean(a)]
    V["STABILITY"] = "ALL-CLEAN" if not unclean else "UNCLEAN:" + ",".join(unclean)
    # noise
    cA = [a for a in ("A0", "A1") if clean(a)]
    if len(cA) == 2 and b1("A0") is not None and b1("A1") is not None: FNC1 = max(abs(b1("A0") - b1("A1")), 0.02)
    else: FNC1 = 0.02
    if len(cA) == 2 and vcold("A0") is not None and vcold("A1") is not None: CNC1 = max(abs(vcold("A0") - vcold("A1")), 0.01)
    else: CNC1 = 0.01
    V["FNC1"] = f"{100*FNC1:.2f}pp"; V["CNC1"] = f"{100*CNC1:.2f}pp"
    say(f"\nNOISE (A pair, clean={cA}): FNC1 {100*FNC1:.2f}pp | CNC1 {100*CNC1:.2f}pp")
    # R-C1-0
    mem, drift, ok0 = [], [], []
    for a in NATIVE:
        ce, v, f = end_ce(a), vcold(a), fcold(a)
        if ce is None or v is None or f is None: continue
        if ce < 0.02: mem.append(a)
        elif abs(v - f) > CNC1: drift.append(a)
        else: ok0.append(a)
    V["R-C1-0"] = ("NO-DATA" if not (mem or drift or ok0) else
                   ("NO-MEMORIZATION" if not mem else "MEMORIZED:" + ",".join(mem)) + (" DRIFT:" + ",".join(drift) if drift else ""))
    say(f"R-C1-0 REGIME: ok {ok0} | memorized {mem} | drift {drift} -> {V['R-C1-0']}")
    # R-C1-1
    V["R-C1-1"] = {2: "RI-CARRIES", 1: "RI-LOTTERY", 0: "RI-DEAD-AT-d128"}[len(cA)] if any(pdir(a).exists() for a in ("A0", "A1")) else "NO-DATA"
    say(f"R-C1-1 SURVIVAL: clean A {cA} -> {V['R-C1-1']}")
    # R-C1-2
    cB = [a for a in ("B0", "B1") if clean(a)]
    bB = [b1(a) for a in cB if b1(a) is not None]; bA = [b1(a) for a in cA if b1(a) is not None]
    if not any(pdir(a).exists() for a in ("B0", "B1")): V["R-C1-2"] = "NO-DATA"
    elif len(cB) < 2: V["R-C1-2"] = "NORM-FAILS"
    elif not bA or not bB: V["R-C1-2"] = "NORM-WORKS-UNREFERENCED"
    elif np.mean(bB) >= np.mean(bA) - FNC1: V["R-C1-2"] = "NORM-WORKS"
    else: V["R-C1-2"] = "NORM-COSTS"
    say(f"R-C1-2 STABILIZER (z-norm bridge): clean B {cB} | mean b1 B {fpp(np.mean(bB) if bB else None)} vs A {fpp(np.mean(bA) if bA else None)} - FNC1 -> {V['R-C1-2']}")
    # R-C1-3
    cands = [(vcold(a), a) for a in NATIVE if clean(a) and vcold(a) is not None]
    if not cands: V["R-C1-3"] = "NO-GO" if any(pdir(a).exists() for a in NATIVE) else "NO-DATA"; champ = None
    else:
        cv, champ = max(cands, key=lambda t: (t[0], b1(t[1]) or 0.0, t[1]))   # ties: higher b1, then name
        V["R-C1-3"] = f"{'HRM-BEATEN' if cv >= .55 else ('ON-TRACK' if cv >= .45 else 'PLATEAU')}:{champ}"
    say(f"R-C1-3 CHAMPION: candidates {[(a, round(100*v,2)) for v, a in sorted(cands, reverse=True)]} -> {V['R-C1-3']}")
    # R-C1-4
    if cands:
        best = max(cands)[0]
        V["R-C1-4"] = ("NATIVE-CARRIES-AT-d128" if best >= D4_COLD + CNC1 else ("NATIVE-FLAT" if best >= D4_COLD - CNC1 else "NATIVE-BELOW-CANVAS-d96"))
    else: V["R-C1-4"] = "NO-DATA"
    say(f"R-C1-4 NATIVE-vs-CANVAS: best native vcold {fpp(max(cands)[0] if cands else None)} vs D4 {fpp(D4_COLD)} -> {V['R-C1-4']}")
    # R-C1-5
    vc = v128(champ) if champ else None
    V["R-C1-5"] = "NO-DATA" if vc is None else ("BREADTH-SCALES-AT-d128" if vc >= C3X_V128 else "NARROW-FUNNEL")
    say(f"R-C1-5 BREADTH: champion {champ} verified@128 {fpp(vc)} vs {fpp(C3X_V128)} -> {V['R-C1-5']}")
    # R-C1-6
    uni = None; members = []
    for a in NATIVE:
        z = scan_recs(a)
        if z is None or not clean(a): continue
        bits = (z["mi_first_hit"] >= 0) | z["cold_exact"].astype(bool)
        order = np.argsort(z["idx"]); bits = bits[order]
        uni = bits if uni is None else (uni | bits); members.append(a)
    for src in ("C3X", "D4"):
        q = RUNS / f"sxscan_psportBr2b{src}" / "records_all.npz"
        if q.exists() and uni is not None:
            z = dict(np.load(q, allow_pickle=True)); ref = scan_recs(members[0])
            if set(z["idx"].tolist()) == set(ref["idx"].tolist()):
                bits = ((z["mi_first_hit"] >= 0) | z["cold_exact"].astype(bool))[np.argsort(z["idx"])]
                uni = uni | bits; members.append(src)
    V["R-C1-6"] = "NO-DATA" if uni is None else ("B-M3-BY-PORTFOLIO" if uni.mean() >= .95 else "PORTFOLIO-BELOW-B-M3")
    say(f"R-C1-6 PORTFOLIO (labeled): union verified@128 over {members} = {fpp(None if uni is None else float(uni.mean()))} -> {V['R-C1-6']}")
    # R-C1-7
    x16 = vcold("X0")
    if stopped("X0"): V["R-C1-7"] = "X0-DEAD"
    elif x16 is None: V["R-C1-7"] = "NO-DATA"
    else: V["R-C1-7"] = "REPRODUCED" if x16 >= .80 else ("PARTIAL" if x16 >= .60 else "NOT-REPRODUCED")
    x64 = full("X0", "vsel", 64); x64 = None if not x64 else x64["exact_acc"]
    xn = vcold("X0n")
    say(f"R-C1-7 REPRODUCTION: X0 vcold@D16 {fpp(x16)} (EqR baseline 84.8) -> {V['R-C1-7']} | X0@D64 {fpp(x64)} (93.0) | b1@D64 {fpp(b1('X0'))} | t1r@128 {fpp(t1r('X0'))} (99.8)"
        f" | maj@128 {fpp(maj('X0'))} | X0n@D16 {fpp(xn)} (76.5) | ACT gain {fpp(None if (x16 is None or xn is None) else x16 - xn)}")
    # R-C1-8
    ref = [vcold(a) for a in ("A0", "A1", "B0", "B1") if clean(a) and vcold(a) is not None]
    r0 = vcold("R0")
    if not clean("R0") or r0 is None or not ref: V["R-C1-8"] = "NO-DATA"
    else:
        dlt = r0 - max(ref); rise = val_rise_last10k("R0")
        if dlt >= CNC1: V["R-C1-8"] = "REGIME-EXPLAINS"
        elif dlt <= -CNC1: V["R-C1-8"] = "REGIME-UNDERTRAINED-AT-50k" if (rise is not None and rise > .01) else "REGIME-HURTS-OUR-CELL"
        else: V["R-C1-8"] = "REGIME-NEUTRAL"
        say(f"R-C1-8 REGIME-vs-ARCHITECTURE: vcold(R0) {fpp(r0)} - best A/B {fpp(max(ref))} = {100*dlt:+.2f}pp | R0 val rise last10k {('-' if rise is None else f'{100*rise:+.2f}pp')} -> {V['R-C1-8']}")
    if V["R-C1-8"] == "NO-DATA": say(f"R-C1-8 REGIME-vs-ARCHITECTURE: -> NO-DATA (R0 clean={clean('R0')}, refs={len(ref)})")
    # predictions
    say("\nSECTION 2 — prediction scoreboard (plan §12.8; bands locked pre-data)")
    for a in ("A0", "A1", "B0", "B1"):
        say(f"  {a}: vcold {fpp(vcold(a))} {hm(vcold(a), PRED['A']['vcold'])} | b1 {fpp(b1(a))} {hm(b1(a), PRED['A']['b1'])} | t1r {fpp(t1r(a))} {hm(t1r(a), PRED['A']['t1r'])}"
            f" | v128 {fpp(v128(a))} {hm(v128(a), PRED['A']['v128'])} | eta {('-' if last_eta(a) is None else f'{last_eta(a):.3f}')} {hm(last_eta(a), PRED['A']['eta'])}")
    say(f"  R0: vcold {fpp(vcold('R0'))} {hm(vcold('R0'), PRED['R0']['vcold'])}")
    say(f"  X0: cold16 {fpp(x16)} {hm(x16, PRED['X0']['cold16'])} | cold64 {fpp(x64)} {hm(x64, PRED['X0']['cold64'])} | b1 {fpp(b1('X0'))} {hm(b1('X0'), PRED['X0']['b1'])} | t1r {fpp(t1r('X0'))} {hm(t1r('X0'), PRED['X0']['t1r'])}")
    say(f"  X0n: cold16 {fpp(xn)} {hm(xn, PRED['X0n']['cold16'])}")
    say(); say("SUMMARY: " + "  ".join(f"{k}={v}" for k, v in V.items()))
    _write(); return V

def _write():
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text("\n".join(LINES) + "\n"); say(f"artifact -> {OUT}")

# ---------- self-test ----------
def _mk(root, a, *, vc=None, fc=None, b1v=None, v128v=None, t1rv=None, retfm_=1.0, ce=0.05, census_=0.0,
        stopped_=None, eta=0.9, val_rise=None, n_full=N_FULL, n_scan=N_SCAN, seed=20260822, break_vote=False,
        decor=False):
    ev = root / f"sxeval_p{TAG}{a}"; t = HEADLINE_T[a]
    for which, v in (("vsel", vc), ("final", fc)):
        if v is not None:
            d = ev / f"full_{which}_t{t}"; d.mkdir(parents=True, exist_ok=True)
            (d / "summary_all.json").write_text(json.dumps(dict(exact_acc=v, n=n_full)))
    if a in FIELD and vc is not None:
        d = ev / "full_vsel_t64"; d.mkdir(parents=True, exist_ok=True)
        (d / "summary_all.json").write_text(json.dumps(dict(exact_acc=min(vc + .07, .99), n=n_full)))
    if retfm_ is not None and a in NATIVE:
        d = ev / "retfm_t8"; d.mkdir(parents=True, exist_ok=True)
        (d / "summary_all.json").write_text(json.dumps(dict(exact_acc=retfm_, n=512)))
    if b1v is not None:
        d = root / f"sxscan_p{TAG}{a}"; d.mkdir(parents=True, exist_ok=True)
        n = n_scan; rng = np.random.default_rng(abs(hash(a)) % 1000)
        cold = np.zeros(n, bool); cold[: int((vc or .3) * n)] = True
        hit = np.full(n, -1); nh = int((v128v or .5) * n)
        pos = rng.choice(n, nh, replace=False) if decor else np.arange(nh)   # decor: per-arm hit sets decorrelate
        hit[pos] = rng.integers(0, 128, nh)
        vote = float(((hit >= 0) | cold).mean())
        (d / "summary_all.json").write_text(json.dumps(dict(
            n=n, t_total=64, k_init=128, subsample_seed=seed, b1_exact=b1v, exact_acc=float(cold.mean()),
            exact_acc_vote=vote + (0.01 if break_vote else 0.0),
            vote_at_k={"128": vote}, t1r_at_k={"128": t1rv if t1rv is not None else b1v})))
        np.savez(d / "records_all.npz", idx=np.arange(n), cold_exact=cold, mi_first_hit=hit)
    if v128v is not None:
        d = root / f"sxscreen_p{TAG}{a}_vb"; d.mkdir(parents=True, exist_ok=True)
        (d / "summary_all.json").write_text(json.dumps(dict(majority_vote_at_k={"128": (b1v or 0) + .05}, n=512)))
    if census_ is not None:
        d = root / f"sxcensus_p{TAG}{a}_vsel"; d.mkdir(parents=True, exist_ok=True)
        (d / "census.json").write_text(json.dumps(dict(rows=[dict(t=64, exploded_frac=census_), dict(t=256, exploded_frac=census_)])))
    pd = root / f"pretrain{TAG}_{a}"; pd.mkdir(parents=True, exist_ok=True)
    if stopped_: (pd / "STOPPED.txt").write_text(stopped_)
    rows = [json.dumps(dict(step=s, loss=.3, ce_in=ce)) for s in (40000, 50000)]
    key = "val_t64_ema" if a == "R0" else ("val_t16_ema" if a in FIELD else "val_t64")
    v_end = (vc or .3); v_prev = v_end - (val_rise if val_rise is not None else 0.0)
    rows += [json.dumps({"monitor": {"step": 40000, key: v_prev, "eta": eta}}), json.dumps({"monitor": {"step": 50000, key: v_end, "eta": eta}})]
    (pd / "metrics.jsonl").write_text("\n".join(rows) + "\n")

def selftest():
    checks = []
    def run(build):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); build(root)
            globals()["RUNS"] = root; globals()["OUT"] = root / "a" / "v.txt"
            with contextlib.redirect_stdout(io.StringIO()): return analyze()
    def wA(r):  # the hoped-for world
        _mk(r, "A0", vc=.47, fc=.46, b1v=.46, v128v=.60, t1rv=.58); _mk(r, "A1", vc=.45, fc=.45, b1v=.44, v128v=.58, t1rv=.56)
        _mk(r, "B0", vc=.48, fc=.48, b1v=.47, v128v=.61, t1rv=.59); _mk(r, "B1", vc=.46, fc=.46, b1v=.45, v128v=.59, t1rv=.57)
        _mk(r, "R0", vc=.47, fc=.47, b1v=.45, v128v=.60, t1rv=.58, val_rise=.002)
        _mk(r, "X0", vc=.84, fc=.84, b1v=.80, v128v=.97, t1rv=.95); _mk(r, "X0n", vc=.76, fc=.76)
    v = run(wA)
    checks += [("A integrity", v["INTEGRITY"] == "PASS"), ("A no memorization", v["R-C1-0"] == "NO-MEMORIZATION"),
               ("A RI carries", v["R-C1-1"] == "RI-CARRIES"), ("A norm works", v["R-C1-2"] == "NORM-WORKS"),
               ("A champion B0 on-track", v["R-C1-3"] == "ON-TRACK:B0"), ("A native carries", v["R-C1-4"] == "NATIVE-CARRIES-AT-d128"),
               ("A narrow funnel", v["R-C1-5"] == "NARROW-FUNNEL"), ("A portfolio below", v["R-C1-6"] == "PORTFOLIO-BELOW-B-M3"),
               ("A X0 reproduced", v["R-C1-7"] == "REPRODUCED"), ("A regime neutral", v["R-C1-8"] == "REGIME-NEUTRAL"),
               ("A FNC1 measured", v["FNC1"] == "2.00pp" and v["CNC1"] == "2.00pp"), ("A all clean", v["STABILITY"] == "ALL-CLEAN")]
    def wB(r):  # lottery + norm fails + X0 partial + regime explains + HRM beaten by R0
        _mk(r, "A0", vc=.46, fc=.45, b1v=.44, v128v=.60); _mk(r, "A1", vc=.30, fc=.30, b1v=.30, v128v=.40, stopped_="STOPPED final step 25000 (NaN halt)")
        _mk(r, "B0", vc=.47, fc=.47, b1v=.46, v128v=.60); _mk(r, "B1", vc=.47, fc=.47, b1v=.46, v128v=.60, census_=.05)
        _mk(r, "R0", vc=.56, fc=.56, b1v=.50, v128v=.70, val_rise=.03)
        _mk(r, "X0", vc=.70, fc=.70, b1v=.60, v128v=.9); _mk(r, "X0n", vc=.66, fc=.66)
    v = run(wB)
    checks += [("B RI lottery", v["R-C1-1"] == "RI-LOTTERY"), ("B norm fails (B1 exploded)", v["R-C1-2"] == "NORM-FAILS"),
               ("B champion R0 HRM-beaten", v["R-C1-3"] == "HRM-BEATEN:R0"), ("B regime explains", v["R-C1-8"] == "REGIME-EXPLAINS"),
               ("B X0 partial", v["R-C1-7"] == "PARTIAL"), ("B unclean named", "A1" in v["STABILITY"] and "B1" in v["STABILITY"]),
               ("B FNC1 floor (no clean pair)", v["FNC1"] == "2.00pp")]
    def wC(r):  # memorized A0, drift B1, norm costs, R0 undertrained, X0 not reproduced, portfolio hits
        _mk(r, "A0", vc=.40, fc=.20, b1v=.40, v128v=.9, ce=.005, decor=True); _mk(r, "A1", vc=.44, fc=.44, b1v=.43, v128v=.9, decor=True)
        _mk(r, "B0", vc=.43, fc=.43, b1v=.38, v128v=.9, decor=True); _mk(r, "B1", vc=.44, fc=.40, b1v=.39, v128v=.9, decor=True)
        _mk(r, "R0", vc=.40, fc=.40, b1v=.35, v128v=.9, val_rise=.03, decor=True)
        _mk(r, "X0", vc=.50, fc=.50, b1v=.40, v128v=.8); _mk(r, "X0n", vc=.45, fc=.45)
    v = run(wC)
    checks += [("C memorized A0 + drift B1", v["R-C1-0"] == "MEMORIZED:A0 DRIFT:B1"), ("C norm costs", v["R-C1-2"] == "NORM-COSTS"),
               ("C A0 unclean by CE", "A0" in v["STABILITY"]), ("C regime undertrained", v["R-C1-8"] == "REGIME-UNDERTRAINED-AT-50k"),
               ("C X0 not reproduced", v["R-C1-7"] == "NOT-REPRODUCED"), ("C portfolio B-M3", v["R-C1-6"] == "B-M3-BY-PORTFOLIO"),
               ("C champion A1 plateau", v["R-C1-3"] == "PLATEAU:A1")]
    def wD(r):  # regime hurts (no rise), RI dead, X0 dead
        _mk(r, "A0", vc=.3, fc=.3, b1v=.3, v128v=.5, stopped_="STOPPED final step 5000 (NaN halt)"); _mk(r, "A1", vc=.3, fc=.3, b1v=.3, v128v=.5, retfm_=.5)
        _mk(r, "B0", vc=.46, fc=.46, b1v=.45, v128v=.6); _mk(r, "B1", vc=.45, fc=.45, b1v=.44, v128v=.6)
        _mk(r, "R0", vc=.40, fc=.40, b1v=.35, v128v=.5, val_rise=.0); _mk(r, "X0", vc=.1, fc=.1, stopped_="STOPPED final step 3000 (NaN halt)"); _mk(r, "X0n", vc=.7, fc=.7)
    v = run(wD)
    checks += [("D RI dead", v["R-C1-1"] == "RI-DEAD-AT-d128"), ("D norm works unreferenced", v["R-C1-2"] == "NORM-WORKS-UNREFERENCED"),
               ("D regime hurts", v["R-C1-8"] == "REGIME-HURTS-OUR-CELL"), ("D X0 dead", v["R-C1-7"] == "X0-DEAD"),
               ("D champion B0 on-track", v["R-C1-3"] == "ON-TRACK:B0")]
    def wE(r):  # integrity breach: scan n wrong on B0 + vote identity broken on A1
        wA(r); _mk(r, "B0", vc=.48, fc=.48, b1v=.47, v128v=.61, n_scan=19999); _mk(r, "A1", vc=.45, fc=.45, b1v=.44, v128v=.58, break_vote=True)
    v = run(wE)
    checks += [("E breach withholds", v["INTEGRITY"] == "FAIL" and "R-C1-3" not in v)]
    v = run(lambda r: None)
    checks += [("F no data", v["R-C1-3"] == "NO-DATA" and v["R-C1-7"] == "NO-DATA" and v["R-C1-2"] == "NO-DATA")]
    ok = 0
    for name, passed in checks:
        print(("  PASS  " if passed else "  FAIL  ") + name); ok += bool(passed)
    print(f"selftest: {ok}/{len(checks)}")
    return ok == len(checks)


if __name__ == "__main__":
    if "--selftest" in sys.argv: sys.exit(0 if selftest() else 1)
    analyze()
