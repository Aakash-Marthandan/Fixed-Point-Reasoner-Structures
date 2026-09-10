# Ledger: DEC-ARC NIGHT — the FROZEN analyzer (Plan_2026-09-10_DEC-ARC_Build §4 → the registration entry; built
# 2026-09-10, the bands are locked at the registration commit after the pilot). Registered decision rules only;
# every number is read from the evaluator's summary.json files (tools/eval_decarc.py; tools/arc_suite.py for the
# native control) and the arms' metrics.jsonl monitors; nothing here is descriptive. `--selftest` runs the rules on
# hand-built records with known letters. NO edit after the registration commit (the analysis pass diffs this file
# against it).
"""
  .venv/bin/python tools/analyze_decarc.py --root runs [--out runs/analysis/decarc_verdict]    # the verdict
  .venv/bin/python tools/analyze_decarc.py --selftest
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

# ---------- THE REGISTRY (locked at registration; the pilot fills the [PILOT] bands) ----------
ARMS = {"D0": "DEC-ARC w160 RI+FPA seed 0", "D1": "DEC-ARC w160 RI+FPA seed 1", "D2": "DEC-ARC w160 plain (no RI, no FPA)",
        "N0": "native rg d96 A5-class @40k (the d96 rung's arm; the control)"}
DEC_ARMS = ("D0", "D1", "D2"); SEED_PAIR = ("D0", "D1")
NATIVE_D64_VALHARD = 0.0764          # the banked d64 plain twin's val-hard per-output clean exact (arc_suite, 144 queries)
NATIVE_D64_RI_VALHARD = 0.0486
NATIVE_RETENTION_BAND = (0.1875, 0.2778)   # the natives' handed-truth retention on val-hard (RI, plain)
R = {
    "exact_margin": 0.02,             # R-DA-1: colour flip rate <= the fit-seed floor + margin
    "parity_band": (0.05, 0.12),      # R-DA-2 [PILOT]: val-hard clean exact band for PARITY; below the native twin's 7.64 % by > 2 pp = BELOW
    "depth_pp": 0.02,                 # R-DA-3: exact at T=16 minus exact at step 2: >= +2 pp PROPAGATES, <= -2 pp REGRESSES, else FLAT
    "selector_auc": 0.85,             # R-DA-4: the residual-on-z AUC for exactness on the RI + FPA arms
    "selector_mech_margin": 0.10,     # R-DA-4: the plain twin's AUC below the seed pair's mean by >= this margin = MECHANISM
    "retention_solved": 0.90,         # R-DA-5: FPA-start retention (eps 0) on the solved queries
    "retention_all_above": NATIVE_RETENTION_BAND[1],   # R-DA-5: retention on ALL queries above the natives' upper band = BASINS-HELD
    "frozen_cw": 0.90, "frozen_flips": 1.5,   # R-DA-6: converged-wrong of converged >= .9 AND flips per cell <= 1.5 = FROZEN
    "control_rg96": 0.235,            # R-DA-9: the d96 rung's registered rg-96 band (>= 23.5 % HIT; < 21 % KILL)
    "control_rg96_kill": 0.21,
    "memorized_drop": 0.05,           # MEMORIZATION: the selected grid's monitor minus the final grid's > 5 pp
    "seed_floor_max": 0.03,           # SEEDS: |D0 - D1| on val-hard clean exact <= 3 pp = TIGHT
}


def read_json(p: Path):
    return json.load(open(p)) if p.exists() else None


def load_records(root: Path):
    """runs/pretraindecarc_<arm>/{metrics.jsonl, vsel.json}; runs/decarceval_<arm>/<set>/summary.json (the chain's layout)."""
    rec = {}
    for arm in ARMS:
        d = root / f"pretraindecarc_{arm}"
        r = {"present": d.exists(), "stopped": (d / "STOPPED.txt").exists() or (d / "NAN_ABORT.txt").exists(), "monitor": [], "evals": {}}
        if (d / "metrics.jsonl").exists():
            for line in (d / "metrics.jsonl").read_text().splitlines():
                try:
                    j = json.loads(line)
                except Exception:
                    continue
                if "monitor" in j:
                    r["monitor"].append(j["monitor"])
        r["vsel"] = read_json(d / "vsel.json")
        ed = root / f"decarceval_{arm}"
        for e in (sorted(ed.iterdir()) if ed.exists() else []):
            s = read_json(e / "summary.json")
            if s is not None:
                r["evals"][e.name] = s
        rec[arm] = r
    return rec


# ---------- the rules ----------
def letters(rec):
    L = {}
    # INTEGRITY: every arm present with a val-hard summary from ONE checkpoint per arm; the xcheck on every set
    probs = []
    for arm in ARMS:
        r = rec[arm]
        if not r["present"]: probs.append(f"{arm}: missing")
        ck = {s.get("ckpt") for s in r["evals"].values() if "ckpt" in s}
        if len(ck) > 1: probs.append(f"{arm}: {len(ck)} checkpoints across its evals")
        if any(s.get("xcheck_all") is False for s in r["evals"].values()): probs.append(f"{arm}: trace cross-check failed")
        if "valhard" not in r["evals"]: probs.append(f"{arm}: no val-hard eval")
    L["INTEGRITY"] = "PASS" if not probs else "FAIL: " + "; ".join(probs)
    L["STABILITY"] = "ALL-STABLE" if not any(rec[a]["stopped"] for a in ARMS if rec[a]["present"]) else "STOPPED: " + ",".join(a for a in ARMS if rec[a]["stopped"])
    ev = lambda arm, s="valhard": rec[arm]["evals"].get(s) or {}
    # R-DA-1 EXACTNESS (the colour-flip row on the deployed protocol vs the fit-seed floor)
    out = []
    for arm in DEC_ARMS:
        f = ev(arm).get("flip")
        if f is None: out.append(f"{arm} NO-DATA"); continue
        out.append(f"{arm} " + ("EXACT" if f["flip_rate_colour"] <= f["flip_rate_floor"] + R["exact_margin"] else "NOT-EXACT"))
    L["R-DA-1 EXACTNESS"] = " | ".join(out)
    # R-DA-2 PARITY (val-hard clean exact vs the banked native d64 twin)
    out = []
    for arm in SEED_PAIR:
        c = ev(arm).get("clean_exact_limit")
        if c is None: out.append(f"{arm} NO-DATA"); continue
        lo, hi = R["parity_band"]
        out.append(f"{arm} " + ("BELOW" if c < NATIVE_D64_VALHARD - 0.02 else "ABOVE" if c > hi else "PARITY"))
    L["R-DA-2 PARITY"] = " | ".join(out)
    # R-DA-3 DEPTH on one cell across domains
    out = []
    for arm in SEED_PAIR:
        ebs = ev(arm).get("exact_by_step")
        if not ebs or len(ebs) < 3: out.append(f"{arm} NO-DATA"); continue
        d = ebs[-1] - ebs[1]
        out.append(f"{arm} " + ("PROPAGATES" if d >= R["depth_pp"] else "REGRESSES" if d <= -R["depth_pp"] else "FLAT") + f" (lost {ev(arm).get('lost')})")
    L["R-DA-3 DEPTH"] = " | ".join(out)
    # R-DA-4 SELECTOR: mechanism (AUC) and value (coverage)
    aucs = {}
    out = []
    for arm in DEC_ARMS:
        s = ev(arm)
        if s.get("auc_res_z") is None: out.append(f"{arm} NO-DATA"); continue
        aucs[arm] = s["auc_res_z"]
        cov = (s.get("oracle") or 0) - (s.get("clean_exact_limit") or 0)
        clean = s["auc_res_z"] >= R["selector_auc"]
        out.append(f"{arm} " + ("CLEAN+COVERAGE" if clean and cov > 0 else "CLEAN-NO-COVERAGE" if clean else "DIRTY") + f" (auc {s['auc_res_z']:.2f}, oracle-cold {100*cov:+.1f} pp)")
    if all(a in aucs for a in DEC_ARMS):
        mech = np.mean([aucs[a] for a in SEED_PAIR]) - aucs["D2"] >= R["selector_mech_margin"]
        out.append("MECHANISM" if mech else "NO-MECHANISM")
    L["R-DA-4 SELECTOR"] = " | ".join(out)
    # R-DA-5 RETENTION (the FPA start at eps 0)
    out = []
    for arm in SEED_PAIR:
        s = ev(arm)
        if s.get("retention_gt") is None: out.append(f"{arm} NO-DATA"); continue
        held = (s.get("retention_gt_solved") or 0) >= R["retention_solved"] and s["retention_gt"] > R["retention_all_above"]
        cond = s.get("oracle_given_retains"), s.get("oracle_given_not")
        out.append(f"{arm} " + ("BASINS-HELD" if held else "BASINS-NOT-HELD") + f" (ret {100*s['retention_gt']:.1f} %, solved {100*(s.get('retention_gt_solved') or 0):.0f} %; oracle|ret {cond[0]} vs {cond[1]})")
    L["R-DA-5 RETENTION"] = " | ".join(out)
    # R-DA-6 FAILURE TEXTURE
    out = []
    for arm in SEED_PAIR:
        s = ev(arm)
        if s.get("converged_wrong_of_converged") is None: out.append(f"{arm} NO-DATA"); continue
        frozen = s["converged_wrong_of_converged"] >= R["frozen_cw"] and (s.get("flips_per_cell") or 9) <= R["frozen_flips"]
        out.append(f"{arm} " + ("FROZEN" if frozen else "CHURN") + f" (cw {s['converged_wrong_of_converged']:.2f}, flips {s.get('flips_per_cell')})")
    L["R-DA-6 FAILURE"] = " | ".join(out)
    # R-DA-7 VOTE (descriptive numbers under one letter: does the vote collect anything)
    out = []
    for arm in SEED_PAIR:
        v = ev(arm).get("vote")
        if not v: out.append(f"{arm} NO-DATA"); continue
        out.append(f"{arm} " + ("VOTE-PAYS" if v["pass2"] > (ev(arm).get("clean_exact_limit") or 0) else "VOTE-FLAT") + f" (pass1 {100*v['pass1']:.1f}, pass2 {100*v['pass2']:.1f}, any {100*v['any_view']:.1f})")
    L["R-DA-7 VOTE"] = " | ".join(out)
    # R-DA-8 PUBLIC (a number, no rule)
    out = []
    for arm in SEED_PAIR:
        s = ev(arm, "arc1eval")
        if not s: out.append(f"{arm} NO-DATA"); continue
        v = s.get("vote") or {}
        out.append(f"{arm} pass@1 {100*(s.get('clean_exact_limit') or 0):.2f} | vote pass@2 {100*v.get('pass2', float('nan')):.2f} (n {s.get('n_queries')})")
    L["R-DA-8 PUBLIC"] = " | ".join(out)
    # R-DA-9 CONTROL (the d96 rung's registered band on rg-96)
    s = ev("N0", "rg96")
    if s and s.get("clean_exact_limit") is not None:
        c = s["clean_exact_limit"]
        L["R-DA-9 CONTROL"] = ("HIT" if c >= R["control_rg96"] else "KILL" if c < R["control_rg96_kill"] else "BETWEEN") + f" (rg-96 {100*c:.1f} %)"
    else:
        L["R-DA-9 CONTROL"] = "NO-DATA"
    # SEEDS and MEMORIZATION
    c0, c1 = ev("D0").get("clean_exact_limit"), ev("D1").get("clean_exact_limit")
    L["SEEDS"] = ("TIGHT" if abs(c0 - c1) <= R["seed_floor_max"] else "WIDE") + f" ({100*abs(c0-c1):.1f} pp)" if (c0 is not None and c1 is not None) else "NO-DATA"
    out = []
    for arm in DEC_ARMS:
        mon = [m for m in rec[arm]["monitor"] if "val_t16_ema" in m or "val_t16" in m]
        vs = rec[arm]["vsel"]
        if not mon or not vs: out.append(f"{arm} NO-DATA"); continue
        key = "val_t16_ema" if "val_t16_ema" in mon[-1] else "val_t16"
        sel = [m for m in mon if m["step"] == vs.get("step")]
        if not sel: out.append(f"{arm} NO-DATA"); continue
        drop = sel[0][key] - mon[-1][key]
        out.append(f"{arm} " + ("MEMORIZED" if drop > R["memorized_drop"] else "AT-PEAK") + f" (sel {vs.get('step')} {100*sel[0][key]:.1f} → final {100*mon[-1][key]:.1f})")
    L["MEMORIZATION"] = " | ".join(out)
    return L


# ---------- selftest ----------
def _fake(clean, auc, oracle, ret_all, ret_solved, cw, flips, ebs, flip_c, flip_f, vote_p2=None, rg96=None, ckpt="c"):
    s = {"ckpt": ckpt, "xcheck_all": True, "clean_exact_limit": clean, "auc_res_z": auc, "oracle": oracle, "retention_gt": ret_all,
         "retention_gt_solved": ret_solved, "oracle_given_retains": 0.3, "oracle_given_not": 0.0, "converged_wrong_of_converged": cw,
         "flips_per_cell": flips, "exact_by_step": ebs, "lost": 1, "flip": {"flip_rate_colour": flip_c, "flip_rate_floor": flip_f},
         "n_queries": 144}
    if vote_p2 is not None: s["vote"] = {"pass1": clean, "pass2": vote_p2, "any_view": vote_p2 + 0.05}
    return s


def selftest():
    def rec_of(d0, d1, d2, n0_rg, mon_drop=0.0):
        mon = [{"step": 20000, "val_t16_ema": 0.30}, {"step": 30000, "val_t16_ema": 0.30 - mon_drop}]
        base = lambda ev: {"present": True, "stopped": False, "monitor": mon, "vsel": {"step": 20000}, "evals": ev}
        return {"D0": base({"valhard": d0, "arc1eval": _fake(0.10, .9, .12, .5, .95, .95, 1.0, [.05, .1, .1], .02, .02, vote_p2=.15)}),
                "D1": base({"valhard": d1}), "D2": base({"valhard": d2}),
                "N0": base({"rg96": {"ckpt": "n", "xcheck_all": True, "clean_exact_limit": n0_rg}, "valhard": {"ckpt": "n", "clean_exact_limit": 0.08}})}
    ok = 0
    # 1. the good night: exact, parity, propagates, clean+coverage with mechanism, basins held, frozen, vote pays, control hit, tight seeds
    r = rec_of(_fake(.09, .90, .15, .40, .95, .95, 1.0, [.05, .07, .07, .10], .02, .02, vote_p2=.12),
               _fake(.08, .88, .13, .38, .92, .93, 1.1, [.04, .06, .06, .09], .03, .02, vote_p2=.11),
               _fake(.07, .60, .16, .30, .80, .90, 1.0, [.04, .07, .07, .07], .03, .02), 0.25)
    L = letters(r)
    exp = {"INTEGRITY": "PASS", "STABILITY": "ALL-STABLE"}
    for k, v in exp.items(): assert L[k] == v, (k, L[k]); ok += 1
    assert L["R-DA-1 EXACTNESS"] == "D0 EXACT | D1 EXACT | D2 EXACT", L["R-DA-1 EXACTNESS"]; ok += 1
    assert L["R-DA-2 PARITY"] == "D0 PARITY | D1 PARITY"; ok += 1
    assert L["R-DA-3 DEPTH"].startswith("D0 PROPAGATES") and "D1 PROPAGATES" in L["R-DA-3 DEPTH"]; ok += 1
    assert L["R-DA-4 SELECTOR"].startswith("D0 CLEAN+COVERAGE") and L["R-DA-4 SELECTOR"].endswith("MECHANISM") and "D2 DIRTY" in L["R-DA-4 SELECTOR"]; ok += 1
    assert L["R-DA-5 RETENTION"].startswith("D0 BASINS-HELD"); ok += 1
    assert L["R-DA-6 FAILURE"].startswith("D0 FROZEN"); ok += 1
    assert L["R-DA-7 VOTE"].startswith("D0 VOTE-PAYS"); ok += 1
    assert L["R-DA-9 CONTROL"].startswith("HIT"); ok += 1
    assert L["SEEDS"].startswith("TIGHT"); ok += 1
    assert L["MEMORIZATION"] == "D0 AT-PEAK (sel 20000 30.0 → final 30.0) | D1 AT-PEAK (sel 20000 30.0 → final 30.0) | D2 AT-PEAK (sel 20000 30.0 → final 30.0)", L["MEMORIZATION"]; ok += 1
    # 2. the bad night: not exact, below parity, flat depth, clean without coverage / no mechanism, basins not held, churn, vote flat, control kill, wide seeds, memorized
    r = rec_of(_fake(.03, .90, .03, .20, .50, .50, 5.0, [.03, .03, .03, .03], .20, .02, vote_p2=.03),
               _fake(.10, .90, .10, .20, .50, .50, 5.0, [.10, .10, .10, .10], .20, .02, vote_p2=.10),
               _fake(.07, .85, .16, .30, .80, .90, 1.0, [.04, .07, .07, .07], .03, .02), 0.15, mon_drop=0.10)
    L = letters(r)
    assert L["R-DA-1 EXACTNESS"] == "D0 NOT-EXACT | D1 NOT-EXACT | D2 EXACT"; ok += 1
    assert L["R-DA-2 PARITY"] == "D0 BELOW | D1 PARITY"; ok += 1
    assert L["R-DA-3 DEPTH"].startswith("D0 FLAT"); ok += 1
    assert L["R-DA-4 SELECTOR"].startswith("D0 CLEAN-NO-COVERAGE") and L["R-DA-4 SELECTOR"].endswith("NO-MECHANISM"); ok += 1
    assert L["R-DA-5 RETENTION"].startswith("D0 BASINS-NOT-HELD"); ok += 1
    assert L["R-DA-6 FAILURE"].startswith("D0 CHURN"); ok += 1
    assert L["R-DA-7 VOTE"].startswith("D0 VOTE-FLAT"); ok += 1
    assert L["R-DA-9 CONTROL"].startswith("KILL"); ok += 1
    assert L["SEEDS"].startswith("WIDE"); ok += 1
    assert all("MEMORIZED" in part for part in L["MEMORIZATION"].split(" | ")); ok += 1
    # 3. integrity: two checkpoints in one arm's evals, a missing arm
    r = rec_of(_fake(.09, .9, .15, .4, .95, .95, 1.0, [.05, .07, .07, .10], .02, .02, ckpt="a"),
               _fake(.08, .88, .13, .38, .92, .93, 1.1, [.04, .06, .06, .09], .03, .02), _fake(.07, .6, .16, .3, .8, .9, 1.0, [.04, .07, .07, .07], .03, .02), 0.25)
    r["D0"]["evals"]["arc1eval"]["ckpt"] = "b"; r["N0"]["present"] = False
    L = letters(r)
    assert L["INTEGRITY"].startswith("FAIL") and "D0: 2 checkpoints" in L["INTEGRITY"] and "N0: missing" in L["INTEGRITY"]; ok += 1
    # 4. no data anywhere -> NO-DATA letters, no exception
    r = {a: {"present": False, "stopped": False, "monitor": [], "vsel": None, "evals": {}} for a in ARMS}
    L = letters(r)
    assert L["R-DA-1 EXACTNESS"].startswith("D0 NO-DATA") and L["R-DA-9 CONTROL"] == "NO-DATA" and L["SEEDS"] == "NO-DATA"; ok += 1
    print(f"selftest OK: {ok}/{ok} checks")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="runs"); ap.add_argument("--out", default="runs/analysis/decarc_verdict")
    ap.add_argument("--selftest", action="store_true"); a = ap.parse_args()
    if a.selftest: return selftest()
    rec = load_records(Path(a.root)); L = letters(rec)
    lines = ["DEC-ARC NIGHT — REGISTERED VERDICT (tools/analyze_decarc.py; the registry at the top of this file)"]
    lines += [f"  {k:22s} {v}" for k, v in L.items()]
    txt = "\n".join(lines); print(txt)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out + ".txt").write_text(txt + "\n"); Path(a.out + ".json").write_text(json.dumps({"letters": L, "registry": R, "arms": ARMS}, indent=1))


if __name__ == "__main__":
    main()
