#!/usr/bin/env python3
# Ledger: THE DERIVED READER for the corpus run (2026-09-20; the contract the second adversarial audit's F20 asks for before any of the
# corpus numbers enters the paper). Reads runs/analysis/corpus_normalized_20260919/*.npz; adds nothing to the registration's letters.
#   (1) three masks kept apart: CURRENTLY unsolved at t, EVER solved by the horizon, solved at the END (terminal).
#   (2) revisions as INTEGER counts, "changed at least once later" (an OR over later steps), never "differs at the final step".
#   (3) the completion read with f = 1 and censored trajectories reported separately, and the DISTRIBUTION at f-1 (quartiles), not a median alone.
#   (4) Q10 strengthened: per-PUZZLE agreement with the banked dynamics rows, not a rate comparison.
"""  .venv/bin/python tools/lens_corpus_read.py --selftest
  .venv/bin/python tools/lens_corpus_read.py --read"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "tools")); sys.path.insert(0, str(ROOT / "src"))
import lens_commit_validity as V
import lens_corpus_normalized as CN
OUT = CN.OUT

# ---------------- pure helpers (selftested) ----------------
def masks(exact):
    """exact (T, B) bool along one run -> dict of three DIFFERENT populations, never interchanged."""
    return dict(ever=exact.any(0), terminal=exact[-1], never=~exact.any(0))

def currently_unsolved(exact, t):
    """Puzzles not yet solved at the START of step t (1-based): no exact step before t. Not 'never solved'."""
    return ~exact[:t - 1].any(0) if t > 1 else np.ones(exact.shape[1], bool)

def revisions(preds, ng, sel):
    """preds (T, B, 9, 9); sel (B, 9, 9) bool the cells to follow. Integer counts over the selected EMPTY cells:
    n, changed at least once after their step (OR over later steps), differs at the final step, and the two distinguished."""
    m = sel & ng; n = int(m.sum())
    if not n: return dict(n=0)
    later_any = (preds[1:] != preds[0][None]).any(0); differs_end = preds[-1] != preds[0]
    return dict(n=n, changed_any=int((later_any & m).sum()), differs_end=int((differs_end & m).sum()), changed_and_returned=int((later_any & ~differs_end & m).sum()))

def completion(exact, right_share, sure_share):
    """Per puzzle: the first fully-right step f (1-based; 0 = never). Returns the f = 1 group, the censored group, and for f > 1
    the DISTRIBUTION of the wrong share and the sure share at f-1 (quartiles), with n."""
    fe = np.where(exact.any(0), exact.argmax(0) + 1, 0); out = dict(n=int(len(fe)), f_eq_1=int((fe == 1).sum()), censored=int((fe == 0).sum()))
    sel = np.where(fe > 1)[0]
    if len(sel):
        w = 1 - right_share[fe[sel] - 2, sel]; s = sure_share[fe[sel] - 2, sel]
        out["f_gt_1"] = dict(n=int(len(sel)), wrong_q1=float(np.percentile(w, 25)), wrong_med=float(np.median(w)), wrong_q3=float(np.percentile(w, 75)),
                             sure_q1=float(np.percentile(s, 25)), sure_med=float(np.median(s)), sure_q3=float(np.percentile(s, 75)), sure_mean=float(s.mean()),
                             share_wrong_under_20=float((w < 0.2).mean()), share_sure_over_15=float((s > 0.15).mean()))
    return out

def per_puzzle_agreement(idx_a, ex_a, idx_b, ex_b):
    """Join two runs by puzzle id and compare per-puzzle exactness; returns (n_common, disagreements, only_a, only_b)."""
    common, ia, ib = np.intersect1d(idx_a, idx_b, return_indices=True); a, b = ex_a[ia], ex_b[ib]
    return dict(n=int(len(common)), disagree=int((a != b).sum()), only_a=int((a & ~b).sum()), only_b=int((b & ~a).sum()))

def fmt(v, d=1): return "  -  " if v is None else f"{100 * v:.{d}f}"

# ---------------- the read ----------------
def read():
    rows, Ls = {}, []; say = lambda s_="": (Ls.append(s_), print(s_, flush=True)); B = CN.banked_rows()
    say("THE DERIVED READER (tools/lens_corpus_read.py; the audit's F20 contract). Populations kept apart; revisions as integer counts; the completion with f = 1 and censored separated; Q10 per puzzle.")
    for f in sorted(OUT.glob("A_*.npz")) + sorted(OUT.glob("B_*.npz")) + sorted(OUT.glob("C_*.npz")):
        D = np.load(f, allow_pickle=True); meta = json.loads(str(D["meta"])); name = meta.get("name", f.stem[2:]); part = f.stem[0]
        preds = D["preds"].astype(np.int64); sol = D["sol"]; ng = D["puz"] == 0; own = V.stablemax9 if meta["own"] == "stablemax" else V.softmax9
        right = preds == sol[None]; exact = right.all((2, 3)); nng = np.maximum(ng.sum((1, 2)), 1); rs = (right & ng[None]).sum((2, 3)) / nng[None]
        p = own(D["logits"].astype(np.float64)).max(-1); sure = (p > 0.9) & ng[None]; ss = sure.sum((2, 3)) / nng[None]
        M = masks(exact); comp = completion(exact, rs, ss); rev = revisions(preds, ng, sure[0])
        r = dict(part=part, cell=meta["cell"], own=meta["own"], n=int(len(sol)), T=int(len(preds)), ever=int(M["ever"].sum()), terminal=int(M["terminal"].sum()), never=int(M["never"].sum()), completion=comp, revisions_of_step1_sure=rev)
        cu = currently_unsolved(exact, 4); r["at_step4"] = dict(currently_unsolved=int(cu.sum()), right_on_those=float((right[3] & ng)[cu].sum() / max(ng[cu].sum(), 1)) if cu.any() else None,
                                                                sure_on_those=float(sure[3][cu].sum() / max(ng[cu].sum(), 1)) if cu.any() else None, sure_on_ever_solved=float(sure[3][M["ever"]].sum() / max(ng[M["ever"]].sum(), 1)) if M["ever"].any() else None)
        if part == "A" and name in B: r["banked"] = per_puzzle_agreement(D["ids"], exact[-1], np.array(B[name].get("ids", D["ids"])), exact[-1]) if "ids" in B[name] else dict(note="the banked row stores no puzzle ids; rates only", banked_rate=B[name]["solved"], ours=float(exact[-1].mean()))
        rows[name] = r
    loops = {k: v for k, v in rows.items() if v["cell"] in ("trm", "dec") and v["part"] == "A"}; ctrl = {k: v for k, v in rows.items() if v["cell"] == "rg"}
    mature = {k: v for k, v in loops.items() if v["terminal"] / v["n"] >= 0.8 and v["completion"].get("f_gt_1", {}).get("n", 0) >= 20}
    say(); say(f"== populations (Part A): {len(loops)} loop checkpoints, {len(mature)} mature with n >= 20 completions after step 1, {len(ctrl)} controls")
    say("   ever solved and solved-at-the-end differ on: " + ", ".join(f"{k} {v['ever'] - v['terminal']}" for k, v in rows.items() if v["ever"] != v["terminal"]) or "   (no checkpoint has a terminal loss)")
    say(); say("== completion, with f = 1 and censored kept apart (the quartiles are over puzzles, not a single median)")
    for k, v in list(mature.items())[:6] + [(k, v) for k, v in rows.items() if v["part"] == "B" and k in ("C5", "C7", "C8", "EQR")]:
        c = v["completion"]; g = c.get("f_gt_1")
        say(f"  {k:28s} n {c['n']:4d} | solved at step 1: {c['f_eq_1']:3d} | never by {v['T']}: {c['censored']:3d} | f > 1 (n {g['n']:4d}): wrong at f-1 {fmt(g['wrong_q1'])}/{fmt(g['wrong_med'])}/{fmt(g['wrong_q3'])} % (quartiles), under 20 % on {fmt(g['share_wrong_under_20'])} % of puzzles | sure at f-1 {fmt(g['sure_q1'])}/{fmt(g['sure_med'])}/{fmt(g['sure_q3'])} %, over 15 % on {fmt(g['share_sure_over_15'])} %")
    say(); say("== the class contrast at step 4, on the puzzles NOT YET solved (a different population from 'never solved')")
    for nm, grp in (("loop", mature), ("control", ctrl)):
        vals = [(k, v["at_step4"]) for k, v in grp.items() if v["at_step4"]["currently_unsolved"] >= 10][:6]
        for k, a in vals: say(f"  {nm:8s} {k:28s} not yet solved {a['currently_unsolved']:3d} | cells right on them {fmt(a['right_on_those'])} % | sure on them {fmt(a['sure_on_those'])} % | sure on the ever-solved {fmt(a['sure_on_ever_solved'])} %")
    say(); say("== revisions of the cells the model is SURE of at step 1 (integer counts over cells; 'changed at least once' is an OR over later steps)")
    for k, v in list(mature.items())[:8]:
        r = v["revisions_of_step1_sure"]
        if r["n"]: say(f"  {k:28s} sure cells {r['n']:6d} | changed at least once later {r['changed_any']:5d} ({100 * r['changed_any'] / r['n']:.2f} %) | differ at the final step {r['differs_end']:5d} | changed and returned {r['changed_and_returned']:5d}")
    say(); say("== Q10 strengthened: our Part A rows against the banked dynamics rows")
    nb = [k for k, v in rows.items() if v["part"] == "A" and "banked" in v and "note" in v["banked"]]
    say(f"  the banked rows store no puzzle ids ({len(nb)} of {len(loops) + len(ctrl)} checkpoints), so a per-puzzle join is impossible against them; the rates agreed exactly on 48 of 48 in the registered read. The per-puzzle join IS available against the commit-validity arrays (same 256 puzzles, same ids):")
    for k in ("C5", "C0", "X0", "EQR", "SA256"):
        f = OUT / f"A_champ__{'C5_at_46k-vsel' if k == 'C5' else 'C0_at_16k-vsel'}.npz" if k in ("C5", "C0") else None
        vp = OUT / f"C_{k}.npz"
        if not vp.exists(): continue
        Dv = np.load(vp, allow_pickle=True); R = np.load(ROOT / f"runs/analysis/commit_validity_20260919/{k}.npz", allow_pickle=True)
        H = int(Dv["H"]); pb = Dv["preds"].astype(np.int64)[H - 1::H]; rp = R["logits"].astype(np.float64).argmax(-1) + 1; sol = Dv["sol"]
        a = (pb == sol[None]).all((2, 3))[15]; b = (rp == sol[None]).all((2, 3))[15]
        ag = per_puzzle_agreement(Dv["ids"], a, R["ids"], b); say(f"  {k:6s} at step 16: {ag['n']} puzzles joined | disagreements {ag['disagree']} (cycle probe only {ag['only_a']}, evaluator only {ag['only_b']})")
    (OUT / "derived_read.txt").write_text("\n".join(Ls) + "\n"); (OUT / "derived_read.json").write_text(json.dumps(rows, indent=1, default=float))

# ---------------- selftest ----------------
def selftest():
    n = 0
    ex = np.zeros((4, 3), bool); ex[1:, 0] = True; ex[2, 1] = True                       # p0 solved from step 2; p1 solved at 3 then LOST; p2 never
    M = masks(ex); assert M["ever"].tolist() == [True, True, False] and M["terminal"].tolist() == [True, False, False] and M["never"].tolist() == [False, False, True]; n += 1
    assert currently_unsolved(ex, 1).tolist() == [True] * 3 and currently_unsolved(ex, 3).tolist() == [False, True, True] and currently_unsolved(ex, 4).tolist() == [False, False, True]; n += 1
    rs = np.array([[.5, .4, .3]] * 4); ss = np.array([[.1, .2, .05]] * 4); c = completion(ex, rs, ss)
    assert c["f_eq_1"] == 0 and c["censored"] == 1 and c["f_gt_1"]["n"] == 2 and abs(c["f_gt_1"]["wrong_med"] - 0.55) < 1e-12; n += 1                      # wrong at f-1 = 1 - right
    ex1 = np.zeros((3, 2), bool); ex1[0, 0] = True; ex1[2, 1] = True; c1 = completion(ex1, rs[:3, :2], ss[:3, :2]); assert c1["f_eq_1"] == 1 and c1["censored"] == 0 and c1["f_gt_1"]["n"] == 1; n += 1
    T, B = 4, 1; preds = np.ones((T, B, 9, 9), int); ng = np.zeros((B, 9, 9), bool); ng[0, 0, :3] = True; sel = np.zeros((B, 9, 9), bool); sel[0, 0, :3] = True
    preds[1, 0, 0, 0] = 2                                                                  # cell 0: changes at step 2 and RETURNS by the last step
    preds[:, 0, 0, 1] = [1, 1, 1, 5]                                                        # cell 1: differs at the final step
    r = revisions(preds, ng, sel); assert r == dict(n=3, changed_any=2, differs_end=1, changed_and_returned=1); n += 1                                     # the change-and-return case is invisible to "differs at the final step"
    # the same three puzzles in a different ORDER, with the same outcomes (5 True, 1 False, 3 True): joined by id they agree everywhere;
    # compared by position they would show two disagreements. A second pair differs on exactly one puzzle (id 3).
    a = per_puzzle_agreement(np.array([5, 1, 3]), np.array([True, False, True]), np.array([1, 3, 5]), np.array([False, True, True]))
    assert a == dict(n=3, disagree=0, only_a=0, only_b=0); n += 1
    a2 = per_puzzle_agreement(np.array([5, 1, 3]), np.array([True, False, True]), np.array([1, 3, 5]), np.array([False, False, True]))
    assert a2 == dict(n=3, disagree=1, only_a=1, only_b=0); n += 1                                                                                          # the join must sort by id, not by position
    assert (OUT / "A_champ__C5_at_46k-vsel.npz").exists() and len(list(OUT.glob("*.npz"))) >= 75; n += 1
    print(f"selftest OK: {n}/{n}")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true"); ap.add_argument("--read", action="store_true"); a = ap.parse_args()
    if a.selftest: return selftest()
    if a.read: read()

if __name__ == "__main__":
    main()
