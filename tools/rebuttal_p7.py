#!/usr/bin/env python3
# Ledger: DISCUSSION-PERIOD ANALYSIS P7 (registration Documentation/Note_2026-10-01_Rebuttal_P7_P8_P9_Registration.md, written before
# any row). Saved records only, $0, nothing run. Is correction local refutation? Per transition t -> t+1, every wrong empty-cell digit
# of the displayed grid is given-refuted / peer-refuted / unrefuted; every correct empty-cell digit is conflicted (a wrong peer
# duplicates it) or clear. R7a: injected grids, first transition (attention receivers; study, P1, P1c chunks). R7b / R7c: the models'
# own fixed-start trajectories (the study's 512 stratified runs at 16 iterations; the inside-completion records at outer readouts).
"""  .venv/bin/python tools/rebuttal_p7.py --selftest
  .venv/bin/python tools/rebuttal_p7.py --run          (reads saved records; writes runs/analysis/rebuttal_20261001e/report.{txt,json})"""
from __future__ import annotations
import argparse, glob, json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/analysis/rebuttal_20261001e"
STUDY = ROOT / "runs/analysis/attention_transfer_20260923"
P1 = ROOT / "runs/analysis/rebuttal_20260926"
P1C = ROOT / "runs/analysis/rebuttal_20260926b"
LENS = ROOT / "runs/analysis/inside_completion_20260925"
WIDTHS = (128, 192, 256)
H_HIGH, H_LOW, S_HIGH, S_LOW, Q_MIN, MIN_OBS = 2.0, 1.2, 0.7, 0.4, 0.5, 200
BOOT, BOOT_SEED = 2000, [20261001, 7]

def utc(): return datetime.now(timezone.utc).isoformat()

# ---------------- pure helpers (selftested) ----------------
def unit_peers(self_peer=False):
    r, c = np.divmod(np.arange(81), 9)
    same = (r[:, None] == r[None]) | (c[:, None] == c[None]) | ((r[:, None] // 3 == r[None] // 3) & (c[:, None] // 3 == c[None] // 3))
    if not self_peer: np.fill_diagonal(same, False)
    return same.astype(np.int32)

UNITS = unit_peers()

def classify(g, puz, sol, units=UNITS):
    """g, puz, sol (N, 81) ints (g: the displayed grid, 0 = no digit; puz: 0 = empty). Returns a dict of (N, 81) bool masks over EMPTY
    cells: wrong, given_ref, peer_ref, unref (wrong digits), correct, conflicted (correct digit duplicated by a wrong non-given peer),
    wconf (wrong digit that duplicates a correct non-given peer), nodigit (empty cell showing no digit; excluded from the classes)."""
    g = np.asarray(g, np.int64); puz = np.asarray(puz, np.int64); sol = np.asarray(sol, np.int64)
    empty = puz == 0; has = (g >= 1) & (g <= 9)
    oh = lambda m: (np.eye(10, dtype=np.int32)[np.where(m, g, 0)][..., 1:])            # (N, 81, 9) one-hot of g where m
    ohp = np.eye(10, dtype=np.int32)[puz][..., 1:]                                     # givens one-hot (0 -> none)
    wrong = empty & has & (g != sol); correct = empty & has & (g == sol)
    dig = np.clip(g - 1, 0, 8)[..., None]
    pick = lambda cnt: np.take_along_axis(cnt, dig, axis=-1)[..., 0]
    giv_cnt = np.einsum("ij,njd->nid", units, ohp); cur_cnt = np.einsum("ij,njd->nid", units, oh(empty & has))
    wr_cnt = np.einsum("ij,njd->nid", units, oh(wrong)); co_cnt = np.einsum("ij,njd->nid", units, oh(correct))
    given_ref = wrong & (pick(giv_cnt) > 0)
    peer_ref = wrong & ~given_ref & (pick(cur_cnt) > 0)
    unref = wrong & ~given_ref & ~peer_ref
    conflicted = correct & (pick(wr_cnt) > 0)
    wconf = wrong & (pick(co_cnt) > 0)
    return dict(wrong=wrong, given_ref=given_ref, peer_ref=peer_ref, unref=unref, correct=correct, conflicted=conflicted, wconf=wconf,
                nodigit=empty & ~has)

def transition_counts(g0, g1, puz, sol, require_both=True):
    """Per grid (N rows) counts for one transition g0 -> g1: refuted / unrefuted wrong cells and how many were repaired; given vs peer;
    correct cells, conflicted ones and breaks; wconf cells and their repairs. Rows where the grid lacks either class (refuted, unrefuted)
    get zero H-counts when require_both (the registered within-grid restriction)."""
    C = classify(g0, puz, sol); sol = np.asarray(sol); g1 = np.asarray(g1)
    rep = (g1 == sol); brk = (g1 != sol)
    ref = C["given_ref"] | C["peer_ref"]
    both = (ref.sum(1) > 0) & (C["unref"].sum(1) > 0) if require_both else np.ones(len(sol), bool)
    s = lambda m: m.sum(1)
    out = dict(n_ref=s(ref) * both, r_ref=s(ref & rep) * both, n_unref=s(C["unref"]) * both, r_unref=s(C["unref"] & rep) * both,
               n_giv=s(C["given_ref"]) * both, r_giv=s(C["given_ref"] & rep) * both, n_peer=s(C["peer_ref"]) * both, r_peer=s(C["peer_ref"] & rep) * both,
               n_corr=s(C["correct"]), n_conf=s(C["conflicted"]), b_conf=s(C["conflicted"] & brk), b_corr=s(C["correct"] & brk),
               n_wconf=s(C["wconf"]), r_wconf=s(C["wconf"] & rep))
    return {k: np.asarray(v, np.int64) for k, v in out.items()}

def H_of(c):
    pr = c["r_ref"].sum() / max(c["n_ref"].sum(), 1); pu = c["r_unref"].sum() / max(c["n_unref"].sum(), 1)
    return float(pr / pu) if pu > 0 else float("inf"), float(pr), float(pu)

def boot_H(per_puzzle, reps=BOOT, seed=BOOT_SEED):
    """per_puzzle: dict of (P,) count arrays aggregated per puzzle (cluster) -> (2.5 %, 97.5 %) of H over puzzle resamples."""
    rng = np.random.default_rng(seed); P = len(per_puzzle["n_ref"]); hs = []
    for _ in range(reps):
        i = rng.integers(0, P, P); c = {k: v[i] for k, v in per_puzzle.items()}; hs.append(H_of(c)[0])
    return tuple(float(x) for x in np.percentile(hs, [2.5, 97.5]))

def letter_H(H, lo, n_ref, n_unref, check_min=False):
    if check_min and (n_ref < MIN_OBS or n_unref < MIN_OBS): return "UNDEFINED"
    if H >= H_HIGH and lo > 1: return "LOCAL-REFUTATION"
    if H <= H_LOW: return "NO-ASYMMETRY"
    return "MIXED"

def combine(letters):
    """All-receivers rule: a letter holds only if every receiver carries it."""
    s = set(letters); return letters[0] if len(s) == 1 else "MIXED"

def letter_c(s, q):
    if s >= S_HIGH and q >= Q_MIN: return "CONFLICT-DRIVEN"
    if s <= S_LOW: return "NOT-CONFLICT-DRIVEN"
    return "MIXED"

# ---------------- record readers ----------------
def chunks(base, width, cond):
    fs = sorted(glob.glob(str(base / f"attention_{width}/repair/{cond}/batch_*.npz")))
    ids, pred = [], []
    for f in fs:
        with np.load(f, allow_pickle=False) as d: ids.append(d["ids"]); pred.append(d["pred"])
    return np.concatenate(ids), np.concatenate(pred, axis=1)                         # (n,), (16, n, 9, 9)

def injected_rows(width):
    """Per family: (ids, puz, sol, g0 injected grid with clues restored, g1 = prediction after iteration 1), restricted to the 438."""
    with np.load(STUDY / "inputs/repair.npz", allow_pickle=False) as d: R = {k: d[k] for k in d.files}
    with np.load(P1 / "inputs/p1_sources.npz", allow_pickle=False) as d: Q = {k: d[k] for k in d.files}
    with np.load(P1C / "inputs/p1c_sources.npz", allow_pickle=False) as d: Qc = {k: d[k] for k in d.files}
    assert np.array_equal(R["ids"], Q["ids"]) and np.array_equal(R["ids"], Qc["ids"])
    ids0, puz, sol = R["ids"], R["puz"], R["sol"]; elig = R["wrong_count"] > 0
    fam = {"eqr": (STUDY, R["eqr"])}
    for j in range(4):
        fam[f"random_{j}"] = (STUDY, R["random"][j]); fam[f"legal_random_{j}"] = (P1, Q[f"legal_random_{j}"])
        fam[f"legal_at_eqr_{j}"] = (P1, Q[f"legal_at_eqr_{j}"]); fam[f"consistent_random_{j}"] = (P1C, Qc[f"consistent_random_{j}"])
    rows = {}
    for name, (base, grid) in fam.items():
        ids, pred = chunks(base, width, name)
        assert np.array_equal(ids, ids0), (width, name)
        g0 = np.where(puz != 0, puz, grid).reshape(-1, 81)
        rows[name] = (np.flatnonzero(elig), puz.reshape(-1, 81), sol.reshape(-1, 81), g0, pred[0].reshape(-1, 81).astype(np.int64))
    return rows

def natural_study(width):
    ids, pred = chunks(STUDY, width, "fixed")
    with np.load(STUDY / "inputs/repair.npz", allow_pickle=False) as d: assert np.array_equal(d["ids"], ids); puz, sol = d["puz"], d["sol"]
    return puz.reshape(-1, 81), sol.reshape(-1, 81), pred.reshape(16, -1, 81).astype(np.int64)

def natural_lens(key):
    D = np.load(LENS / f"{key}.npz", allow_pickle=True)
    traj = D["slow_digit"][2::3].astype(np.int64)                                     # outer readouts j = 3, 6, ..., 48 -> (16, N, 81)
    return D["puz"].reshape(-1, 81), D["sol"].reshape(-1, 81), traj

def natural_counts(puz, sol, traj, exclude_completing=False):
    """Transitions t -> t+1 whose start grid precedes the first exact iteration (all of them if never exact); per-puzzle aggregated."""
    T, N, _ = traj.shape; exact = (traj == sol[None]).all(-1); first = np.where(exact.any(0), exact.argmax(0), T)   # 0-based index
    agg = None
    for t in range(T - 1):
        use = t < first                                                                # the start grid is not yet exact
        if exclude_completing: use &= (t + 1) < first
        c = transition_counts(traj[t], traj[t + 1], puz, sol)
        c = {k: v * use for k, v in c.items()}
        agg = c if agg is None else {k: agg[k] + c[k] for k in agg}
    return agg

# ---------------- the run ----------------
def run():
    OUT.mkdir(parents=True, exist_ok=True); Ls = []; JS = dict(created=utc(), registration="Documentation/Note_2026-10-01_Rebuttal_P7_P8_P9_Registration.md")
    say = lambda s_="": (Ls.append(s_), print(s_))
    say(f"P7 report ({JS['created']}); saved records only")
    say("\nR7a — injected grids, first transition (pooled over the 17 families, 438 source-error puzzles, clusters = puzzles)")
    la = []
    JS["R7a"] = {}
    for w in WIDTHS:
        rows = injected_rows(w); per = None; fam_rows = {}
        for name, (el, puz, sol, g0, g1) in rows.items():
            c = transition_counts(g0[el], g1[el], puz[el], sol[el]); fam_rows[name] = H_of(c)
            per = c if per is None else {k: per[k] + c[k] for k in per}
        H, pr, pu = H_of(per); lo, hi = boot_H(per); L_ = letter_H(H, lo, per["n_ref"].sum(), per["n_unref"].sum()); la.append(L_)
        hg = per["r_giv"].sum() / max(per["n_giv"].sum(), 1); hp = per["r_peer"].sum() / max(per["n_peer"].sum(), 1)
        JS["R7a"][w] = dict(H=H, ci=[lo, hi], p_repair_refuted=pr, p_repair_unrefuted=pu, n_refuted=int(per["n_ref"].sum()), n_unrefuted=int(per["n_unref"].sum()),
                            p_repair_given=float(hg), p_repair_peer=float(hp), letter=L_, by_family={k: dict(H=v[0], p_ref=v[1], p_unref=v[2]) for k, v in fam_rows.items()})
        say(f"  Attention {w}: P(repair | refuted) {pr:.3f} (given {hg:.3f}, peer {hp:.3f}) vs P(repair | unrefuted) {pu:.3f} -> H {H:.2f} "
            f"[{lo:.2f}, {hi:.2f}]; n {int(per['n_ref'].sum())} / {int(per['n_unref'].sum())} -> {L_}")
    JS["R7a"]["letter"] = combine(la); say(f"  R7a (all three receivers): {JS['R7a']['letter']}")
    say("\nR7b / R7c — the models' own fixed-start trajectories (transitions starting before first completion)")
    JS["R7b"] = {}; JS["R7c"] = {}; lb = []
    models = [(f"A{w} (study, 512 strat, 16 it)", lambda w=w: natural_study(w)) for w in WIDTHS] + \
             [(f"{k} (lens, 256 strat, 16 outer readouts)", lambda k=k: natural_lens(k)) for k in ("C5", "SA128", "EQR")]
    for name, loader in models:
        puz, sol, traj = loader(); per = natural_counts(puz, sol, traj)
        H, pr, pu = H_of(per); lo, hi = boot_H(per); L_ = letter_H(H, lo, per["n_ref"].sum(), per["n_unref"].sum(), check_min=True); lb.append(L_)
        ex = natural_counts(puz, sol, traj, exclude_completing=True); He, pre, pue = H_of(ex)
        b_tot = per["b_corr"].sum(); s = per["b_conf"].sum() / max(b_tot, 1); base = per["n_conf"].sum() / max(per["n_corr"].sum(), 1)
        pb = per["b_conf"].sum() / max(per["n_conf"].sum(), 1); pw = per["r_wconf"].sum() / max(per["n_wconf"].sum(), 1); q = pb / pw if pw > 0 else float("inf")
        Lc = letter_c(s, q)
        JS["R7b"][name] = dict(H=H, ci=[lo, hi], p_repair_refuted=pr, p_repair_unrefuted=pu, n_refuted=int(per["n_ref"].sum()), n_unrefuted=int(per["n_unref"].sum()),
                               letter=L_, H_excluding_completing_transition=He, p_ref_excl=pre, p_unref_excl=pue)
        JS["R7c"][name] = dict(s=float(s), base_rate_conflicted=float(base), p_break_conflicted=float(pb), p_repair_wconf=float(pw), q=float(q),
                               right_to_wrong_events=int(b_tot), letter=Lc)
        say(f"  {name}: H {H:.2f} [{lo:.2f}, {hi:.2f}] (P repair refuted {pr:.3f} vs unrefuted {pu:.3f}; n {int(per['n_ref'].sum())} / {int(per['n_unref'].sum())}) -> {L_}; "
            f"excluding the completing transition H {He:.2f} ({pre:.3f} vs {pue:.3f})")
        say(f"      R7c: {int(b_tot)} right-to-wrong events, share at conflicted cells s {s:.3f} (base rate {base:.3f}); "
            f"P(break | conflicted) {pb:.3f} vs P(repair | wrong cell duplicating a correct one) {pw:.3f} -> q {q:.2f} -> {Lc}")
    JS["R7b"]["letter"] = combine([x for x in lb if x != "UNDEFINED"]) if any(x != "UNDEFINED" for x in lb) else "UNDEFINED"
    say(f"  R7b (all models with data): {JS['R7b']['letter']}")
    (OUT / "report.txt").write_text("\n".join(Ls) + "\n"); (OUT / "report.json").write_text(json.dumps(JS, indent=1, default=float))

# ---------------- selftest ----------------
def _solution():
    return np.array([[((r * 3 + r // 3 + c) % 9) + 1 for c in range(9)] for r in range(9)], np.int64)

def selftest():
    n = 0; sol = _solution().reshape(81)
    assert all(len(set(sol.reshape(9, 9)[r])) == 9 for r in range(9)) and all(len(set(sol.reshape(9, 9)[:, c])) == 9 for c in range(9)); n += 1
    puz = sol.copy(); puz[[0, 1, 2, 9, 10, 11, 40, 41, 50, 60, 70, 80]] = 0              # a few empty cells (givens elsewhere)
    # (a) given-refuted: cell 0 (row 0) takes the digit of given cell 3 (row 0, a given)
    g = sol.copy(); g[0] = sol[3]; C = classify(g[None], puz[None], sol[None])
    assert C["given_ref"][0, 0] and not C["peer_ref"][0, 0] and not C["unref"][0, 0]; n += 1
    # (b) peer-refuted: cell 1 takes the digit of non-given cell 2 (same row, a correct empty cell) -> cell 2 is conflicted
    puz_b = puz.copy(); g = sol.copy(); d2 = sol[2]
    if any(puz_b[j] == d2 for j in np.flatnonzero(UNITS[1])): puz_b[[j for j in np.flatnonzero(UNITS[1]) if puz_b[j] == d2]] = 0
    g[1] = d2; C = classify(g[None], puz_b[None], sol[None])
    assert C["peer_ref"][0, 1] and C["conflicted"][0, 2] and C["wconf"][0, 1]; n += 1
    # (c) a mutually consistent error set: digits 1 and 2 exchanged in all 18 of their (emptied) cells -> every wrong digit duplicates nothing
    cells = np.flatnonzero((sol == 1) | (sol == 2)); assert len(cells) == 18
    puz_c = sol.copy(); puz_c[cells] = 0; g = sol.copy(); g[cells] = 3 - sol[cells]
    C = classify(g[None], puz_c[None], sol[None])
    assert C["unref"][0, cells].all() and C["wrong"][0].sum() == 18 and not C["peer_ref"][0].any() and not C["given_ref"][0].any(); n += 1
    # (d) the mutant: counting a cell as its own peer makes every wrong digit "peer-refuted" — the deadly pattern must then fail
    Cm = classify(g[None], puz_c[None], sol[None], units=unit_peers(self_peer=True))
    assert not Cm["unref"][0, cells].all(); n += 1
    # (e) transition counts and H: the 18 consistent errors plus one given-refuted cell; only the given-refuted cell is repaired
    e = next(i for i in range(81) if i not in set(cells.tolist()))
    f_ = next(j for j in np.flatnonzero(UNITS[e]) if j not in set(cells.tolist()) and j // 9 == e // 9)
    p2 = puz_c.copy(); p2[e] = 0; g0 = g.copy(); g0[e] = sol[f_]; g1 = g0.copy(); g1[e] = sol[e]
    c = transition_counts(g0[None], g1[None], p2[None], sol[None])
    assert c["n_ref"][0] == 1 and c["r_ref"][0] == 1 and c["n_unref"][0] == 18 and c["r_unref"][0] == 0; n += 1
    H, pr, pu = H_of(c); assert H == float("inf") and pr == 1.0 and pu == 0.0; n += 1
    # (f) letters at the boundaries
    assert letter_H(2.0, 1.01, 500, 500) == "LOCAL-REFUTATION" and letter_H(2.0, 0.99, 500, 500) == "MIXED" and letter_H(1.2, 0.9, 500, 500) == "NO-ASYMMETRY"; n += 1
    assert letter_H(5.0, 2.0, 199, 500, check_min=True) == "UNDEFINED" and combine(["MIXED", "MIXED"]) == "MIXED" and combine(["LOCAL-REFUTATION", "MIXED"]) == "MIXED"; n += 1
    assert letter_c(0.7, 0.5) == "CONFLICT-DRIVEN" and letter_c(0.69, 0.9) == "MIXED" and letter_c(0.4, 0.1) == "NOT-CONFLICT-DRIVEN"; n += 1
    print(f"selftest OK ({n} checks)")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true"); ap.add_argument("--run", action="store_true"); a = ap.parse_args()
    if a.selftest: selftest()
    if a.run: run()

if __name__ == "__main__":
    main()
