#!/usr/bin/env python3
"""THE RESTART COLUMN AS A LETTER — analyzer (Documentation/Note_2026-09-20_Restart_Column.md; FROZEN at the registration commit).

The paper's restart column is the k128, 64-iteration scan on 5,000 test puzzles: from each of 128 randomised starts the loop is run, the draw
with the SMALLEST residual is taken (the registered selection), and "verified" means at least one draw is exact. The width-192 triple
(C5 / C7 / C8) is the paper's row; this reader compares an arm against it on the IDENTICAL puzzles, with a floor computed from those same three
seeds on this same instrument — never from memory.

RULES (frozen; the note's §"The rule"):
  INTEGRITY   every row: k_init 128, t_total 64, EMA, n 5,000, on its arm's registered selected grid, and the puzzle ids identical across
              every row compared (the arms' and the three seeds').
  FLOOR       FLOOR_SEL = 2 x (max - min) over the three seeds' residual-SELECTED accuracies; FLOOR_VER likewise over their VERIFIED ones.
              Both printed with the per-seed values that produce them.
  R-RS-1      each arm's selected accuracy vs the triple's mean: ABOVE / INSIDE / BELOW by FLOOR_SEL.
  R-RS-2      each arm vs each seed, paired on the identical puzzles (exact McNemar): difference, discordant counts, p; and whether the arm is
              ahead of ALL THREE seeds.
  R-RS-3      the verified column, the same two readings with FLOOR_VER.
  R-RS-4      descriptive: the smallest k at which an arm's selected accuracy is within 0.05 pp of its own k128 value.
  A letter is claimable for the paper only when an arm is ABOVE (or BELOW) its floor AND ahead of (behind) all three seeds.
  NOTE ON THAT CLAUSE (found by mutation, 2026-09-20): with FLOOR = 2 x (max - min), "above the mean by more than the floor" already
  implies "above every seed" (the largest seed is at most 2/3 of the spread above the mean), so the per-seed clause is REDUNDANT for
  these rows and a mutant that drops it cannot be killed by any fair fixture. It is kept because it is not redundant if the rows ever
  stop sharing puzzle ids exactly (then the paired differences and the mean differences part company), and because the per-seed
  numbers are what a reader wants; it is reported, never relied on alone.

  .venv/bin/python tools/analyze_restarts.py --root <stage>/runs --refs <dir with the seeds' k128 rows> --out <dir>
  .venv/bin/python tools/analyze_restarts.py --selftest
"""
from __future__ import annotations
import argparse, json, sys, tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_paperfinal as PF   # exact_mcnemar, selected_exact, say, pp, LINES

ARMS = ("SA128", "SA192", "SA256")
SEEDS = ("C5", "C7", "C8")
N, K, T = 5000, 128, 64


def row(d):
    """(selected bits, verified bits, ids, summary) of a k128 scan directory, or None."""
    d = Path(d); s = d / "summary_all.json"; r = d / "records_all.npz"
    if not (s.exists() and r.exists()): return None
    z = dict(np.load(r, allow_pickle=True)); sel, ver = PF.selected_exact(z)
    return dict(sel=sel, ver=ver, idx=np.asarray(z["idx"]), summ=json.loads(s.read_text()), ex=np.asarray(z["mi_exact_k"]).astype(bool),
                res=np.asarray(z["mi_resid_k"]).astype(np.float64))


def floor_of(vals): return 2.0 * (max(vals) - min(vals))


def label(diff, floor): return "n/a" if diff is None else ("INSIDE" if abs(diff) <= floor else ("ABOVE" if diff > 0 else "BELOW"))


def paired(a, b, ids_a, ids_b):
    """exact McNemar on the identical ids: (diff, only_a, only_b, n, p)."""
    common, pa, pb = np.intersect1d(ids_a, ids_b, return_indices=True)
    x, y = a[pa], b[pb]; only_a = int((x & ~y).sum()); only_b = int((~x & y).sum())
    return dict(diff=float(x.mean() - y.mean()), only_a=only_a, only_b=only_b, n=int(len(common)), p=PF.exact_mcnemar(only_a, only_b))


def plateau(r, tol=0.0005):
    """the smallest k whose residual-selected accuracy is within tol of the k128 value (the reader's own re-derivation from the draws)."""
    ex, res = r["ex"], np.where(np.isfinite(r["res"]), r["res"], np.inf)
    full = float(ex[np.arange(len(ex)), np.argmin(res, axis=1)].mean())
    for k in (1, 2, 4, 8, 16, 32, 64, 128):
        acc = float(ex[np.arange(len(ex)), np.argmin(res[:, :k], axis=1)].mean())
        if abs(acc - full) <= tol: return k, acc, full
    return None, None, full


def analyze(root, refs):
    root, refs = Path(root), Path(refs); J = {}; bad = []
    A = {a: row(root / f"filler_sxscan128_pchamp{a}") for a in ARMS}
    S = {s: row(refs / f"filler_sxscan128_pchamp{s}") for s in SEEDS}
    A = {k: v for k, v in A.items() if v}; S = {k: v for k, v in S.items() if v}
    PF.say("THE RESTART COLUMN AS A LETTER — analyzer (frozen rules; the floor from the width-192 seeds on this same instrument)")
    PF.say(f"arms {sorted(A) or 'none'} | seeds {sorted(S) or 'none'}")
    if not A or len(S) < 3: PF.say("NO-DATA"); return {"INTEGRITY": "NO-DATA"}
    ids0 = None
    for nm, r in list(A.items()) + list(S.items()):
        s = r["summ"]; got = (s.get("n"), s.get("t_total"), int(s.get("k_init") or 0), bool(s.get("ema")))
        if got != (N, T, K, True): bad.append(f"{nm}: n/t/k/ema {got} != {(N, T, K, True)}")
        ids = np.sort(r["idx"])
        if ids0 is None: ids0 = ids
        elif len(ids) != len(ids0) or (ids != ids0).any(): bad.append(f"{nm}: puzzle ids differ from the first row's")
        if r["ex"].shape[1] != K: bad.append(f"{nm}: {r['ex'].shape[1]} draws, not {K}")
    J["INTEGRITY"] = "PASS" if not bad else "FAIL: " + "; ".join(bad); PF.say(f"INTEGRITY             {J['INTEGRITY']}")
    sel_s = {s: float(S[s]["sel"].mean()) for s in S}; ver_s = {s: float(S[s]["ver"].mean()) for s in S}
    fsel, fver = floor_of(sel_s.values()), floor_of(ver_s.values())
    msel, mver = sum(sel_s.values()) / 3, sum(ver_s.values()) / 3
    J["FLOOR"] = (f"FLOOR_SEL {100*fsel:.3f} pp from the seeds' selected {{" + ", ".join(f"{k}: {100*v:.2f}" for k, v in sorted(sel_s.items())) + f"}} (mean {100*msel:.2f}); "
                  f"FLOOR_VER {100*fver:.3f} pp from their verified {{" + ", ".join(f"{k}: {100*v:.2f}" for k, v in sorted(ver_s.items())) + f"}} (mean {100*mver:.2f})")
    PF.say(f"FLOOR                 {J['FLOOR']}")
    for a in sorted(A):
        for tag, key, m, fl in (("R-RS-1 SELECTED", "sel", msel, fsel), ("R-RS-3 VERIFIED", "ver", mver, fver)):
            x = float(A[a][key].mean()); d = x - m
            J[f"{tag} {a}"] = f"{label(d, fl)} ({PF.pp(x)} vs the triple {100*m:.2f}; {100*d:+.3f} pp; floor {100*fl:.3f})"
            PF.say(f"{tag} {a:6s}  {J[f'{tag} {a}']}")
            per = {s: paired(A[a][key], S[s][key], A[a]["idx"], S[s]["idx"]) for s in sorted(S)}
            ahead = all(q["diff"] > 0 for q in per.values()); behind = all(q["diff"] < 0 for q in per.values())
            J[f"{tag.replace('R-RS-1', 'R-RS-2').replace('R-RS-3', 'R-RS-3b')} {a} per seed"] = (
                ("AHEAD-OF-ALL-THREE" if ahead else "BEHIND-ALL-THREE" if behind else "MIXED") + " (" +
                ", ".join(f"{s} {100*q['diff']:+.3f} pp (only-arm {q['only_a']}, only-{s} {q['only_b']}, p {q['p']:.2g})" for s, q in per.items()) + ")")
            PF.say(f"    per seed          {J[f'{tag.replace('R-RS-1', 'R-RS-2').replace('R-RS-3', 'R-RS-3b')} {a} per seed']}")
            claim = (label(d, fl) in ("ABOVE", "BELOW")) and (ahead or behind)
            J[f"CLAIMABLE {tag} {a}"] = "YES" if claim else "NO"
            PF.say(f"    claimable         {J[f'CLAIMABLE {tag} {a}']} (the floor AND all three seeds must agree)")
        k, acc, full = plateau(A[a])
        J[f"R-RS-4 {a}"] = f"plateau at k {k} ({PF.pp(acc)} vs k128 {PF.pp(full)})" if k else f"no plateau within 0.05 pp before k128 ({PF.pp(full)})"
        PF.say(f"R-RS-4 {a:6s}        {J[f'R-RS-4 {a}']}")
    e = row(refs / "filler_sxscan128_pport_eqr")
    J["EqR reference"] = "absent" if not e else f"selected {PF.pp(float(e['sel'].mean()))}, verified {PF.pp(float(e['ver'].mean()))} (released weights, the same 5,000; a reference, never a floor)"
    PF.say(f"EqR reference         {J['EqR reference']}")
    return J


def _mk(tmp, arms, seeds, arm_shift=0, nbad=None, late=0.0):
    """rows whose SELECTED and VERIFIED columns are exactly the targets: (selected, verified) per name.
    puzzles 0..n_sel: an exact draw with the smallest residual (found by the selection; a `late` share of them at draw 5, so the
    plateau is k 8); n_sel..n_ver: an exact draw whose residual is WORSE than a wrong draw (verified but not selected); the rest: none."""
    ids = np.arange(N) * 3
    def w(d, sel, ver, n=N, idx=None):
        d.mkdir(parents=True, exist_ok=True); idx = ids if idx is None else idx
        ex = np.zeros((N, K), bool); res = np.full((N, K), 0.5)
        n_sel, n_ver = int(round(sel * N)), int(round(ver * N)); n_late = int(round(late * n_sel))
        for i in range(n_sel):
            j = 5 if i < n_late else 0
            ex[i, j] = True; res[i, j] = 0.001
        for i in range(n_sel, n_ver):
            ex[i, 3] = True; res[i, 3] = 0.9
        np.savez(d / "records_all.npz", idx=idx, mi_exact_k=ex, mi_resid_k=res, cold_exact=ex[:, 0])
        (d / "summary_all.json").write_text(json.dumps(dict(n=n, t_total=T, k_init=K, ema=True, exact_acc=sel, ckpt="x")))
    for a, (sl, vr) in arms.items(): w(tmp / "stage" / f"filler_sxscan128_pchamp{a}", sl, vr, n=(nbad or N), idx=ids + arm_shift)
    for s, (sl, vr) in seeds.items(): w(tmp / "refs" / f"filler_sxscan128_pchamp{s}", sl, vr)
    return tmp / "stage", tmp / "refs"


def selftest():
    n = 0
    assert abs(floor_of([0.9985, 0.9989, 0.9981]) - 0.0016) < 1e-9 and label(0.002, 0.0016) == "ABOVE" and label(-0.002, 0.0016) == "BELOW" and label(0.001, 0.0016) == "INSIDE"; n += 1
    a = np.array([1, 1, 0, 0], bool); b = np.array([1, 0, 1, 0], bool); q = paired(a, b, np.arange(4), np.arange(4))
    assert q["only_a"] == 1 and q["only_b"] == 1 and q["n"] == 4 and abs(q["diff"]) < 1e-12; n += 1
    with tempfile.TemporaryDirectory() as t_:
        t = Path(t_); r, f = _mk(t, {"SA128": (.9990, .9994), "SA192": (.9996, .9998), "SA256": (.9985, .9990)},
                                 {"C5": (.9985, .9989), "C7": (.9983, .9987), "C8": (.9987, .9991)}, late=0.25)
        PF.LINES.clear(); J = analyze(r, f)
        assert J["INTEGRITY"] == "PASS", J["INTEGRITY"]; n += 1
        assert J["FLOOR"].startswith("FLOOR_SEL 0.080 pp") and "FLOOR_VER 0.080 pp" in J["FLOOR"], J["FLOOR"]; n += 1     # 2 x (99.87 - 99.83)
        assert J["R-RS-1 SELECTED SA192"].startswith("ABOVE") and J["R-RS-1 SELECTED SA256"].startswith("INSIDE") and J["R-RS-1 SELECTED SA128"].startswith("INSIDE"), (J["R-RS-1 SELECTED SA192"], J["R-RS-1 SELECTED SA128"]); n += 1   # +0.11 beyond the 0.08 floor; +0.05 and -0.01 inside it
        assert J["R-RS-2 SELECTED SA192 per seed"].startswith("AHEAD-OF-ALL-THREE") and J["CLAIMABLE R-RS-1 SELECTED SA192"] == "YES"; n += 1
        assert J["CLAIMABLE R-RS-1 SELECTED SA256"] == "NO", J["R-RS-2 SELECTED SA256 per seed"]; n += 1
        assert J["R-RS-3 VERIFIED SA192"].startswith("ABOVE") and J["R-RS-4 SA128"].startswith("plateau at k 8"), (J["R-RS-3 VERIFIED SA192"], J["R-RS-4 SA128"]); n += 1
        assert J["EqR reference"] == "absent"; n += 1
    with tempfile.TemporaryDirectory() as t_:   # the id and shape guards
        t = Path(t_); r, f = _mk(t, {"SA128": (.999, .9994)}, {"C5": (.9985, .9989), "C7": (.9983, .9987), "C8": (.9987, .9991)}, arm_shift=1)
        PF.LINES.clear(); J = analyze(r, f); assert J["INTEGRITY"].startswith("FAIL") and "puzzle ids differ" in J["INTEGRITY"], J["INTEGRITY"]; n += 1
    with tempfile.TemporaryDirectory() as t_:
        t = Path(t_); r, f = _mk(t, {"SA128": (.999, .9994)}, {"C5": (.9985, .9989), "C7": (.9983, .9987), "C8": (.9987, .9991)}, nbad=4999)
        PF.LINES.clear(); J = analyze(r, f); assert J["INTEGRITY"].startswith("FAIL") and "n/t/k/ema" in J["INTEGRITY"]; n += 1
    with tempfile.TemporaryDirectory() as t_:
        PF.LINES.clear(); assert analyze(Path(t_), Path(t_)) == {"INTEGRITY": "NO-DATA"}; n += 1
    print(f"selftest OK: {n}/{n}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true"); ap.add_argument("--root"); ap.add_argument("--refs"); ap.add_argument("--out"); a = ap.parse_args()
    if a.selftest: return selftest()
    J = analyze(a.root, a.refs)
    if a.out:
        o = Path(a.out); o.mkdir(parents=True, exist_ok=True); (o / "restart_verdict.txt").write_text("\n".join(PF.LINES) + "\n"); (o / "restart_verdict.json").write_text(json.dumps(J, indent=1))


if __name__ == "__main__":
    main()
