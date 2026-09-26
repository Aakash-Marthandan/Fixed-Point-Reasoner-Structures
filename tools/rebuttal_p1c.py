#!/usr/bin/env python3
# Ledger: THE REVIEW-PERIOD EXPERIMENT P1c — mutually consistent random corruptions (registration
# Documentation/Note_2026-09-26_Review_Period_P1c_P2b_Registration.md, written before any row). MEASUREMENT, $0, inference only.
#   Do random wrong digits that are COORDINATED like the model's own (each wrong digit's solution-holders in its row, column and box are
#   themselves wrong, so the digit conflicts with nothing correct) repair as slowly as the model's own errors? Same receivers, puzzles,
#   counts and pipeline as P1 (tools/rebuttal_p1p2.py); one new source family, four draws.
"""  .venv/bin/python tools/rebuttal_p1c.py --selftest
  .venv/bin/python tools/rebuttal_p1c.py prepare
  JAX_PLATFORMS=cpu .venv/bin/python tools/rebuttal_p1c.py run --width 128|192|256 [--smoke]
  JAX_PLATFORMS=cpu .venv/bin/python tools/rebuttal_p1c.py port --receiver C5|EQR [--smoke]
  .venv/bin/python tools/rebuttal_p1c.py report"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import rebuttal_p1p2 as P
from rebuttal_p1p2 import UNITS, legal_wrong_digits, clue_conflict_share, gap_closure, paired_bootstrap, mcnemar_exact, utc

OUT = ROOT / "runs/analysis/rebuttal_20260926b"
INPUTS = OUT / "inputs/p1c_sources.npz"          # fixed: the smoke redirects outputs only
P1 = P.OUT
SEED, DRAWS, BATCH, STEPS = 20260926, 4, 128, 16
CONDS = tuple(f"consistent_random_{j}" for j in range(DRAWS))
PORT_SOURCES = ("eqr",) + CONDS
C_HIGH, C_LOW = 0.67, 0.33


# ---------------- pure helpers (selftested) ----------------
def holders(sol, cell, d):
    """Flat indices of the unit peers of `cell` whose SOLUTION digit is d (up to three distinct cells)."""
    s = np.asarray(sol).reshape(81)
    return [int(h) for h in np.where(UNITS[cell] & (s == d))[0]]

def sample_consistent(puz, sol, k, rng):
    """The solution with k empty cells made wrong by legal digits chosen so that, wherever possible, each wrong digit's solution-holders
    in its row, column and box are themselves among the wrong cells (a chain-closure procedure: a holder that would make the new digit
    a duplicate is corrupted next, with the digit that opens the fewest further holders). Returns (grid, assigned)."""
    puz, sol = np.asarray(puz), np.asarray(sol); p, s = puz.reshape(81), sol.reshape(81); g = s.copy(); W = {}; queue = []
    empty = [c for c in range(81) if p[c] == 0]
    legal = lambda c: legal_wrong_digits(puz, sol, c)
    while len(W) < k:
        if queue:
            c = queue.pop(0)
            if c in W: continue
            opts = legal(c)
            if not opts: continue
            open_counts = [sum(1 for h in holders(sol, c, d) if h not in W) for d in opts]
            m = min(open_counts); d = int(rng.choice([o for o, oc in zip(opts, open_counts) if oc == m]))
        else:
            free = [c for c in empty if c not in W and legal(c)]
            if not free: break
            c = int(rng.choice(free)); opts = legal(c)
            closed = [d for d in opts if all(h in W for h in holders(sol, c, d))]
            d = int(rng.choice(closed)) if closed else int(rng.choice(opts))
        W[c] = d; g[c] = d
        for h in holders(sol, c, d):
            if h not in W and p[h] == 0: queue.append(h)
    # closing sweeps over the chosen cells only: each wrong cell takes the legal wrong digit that duplicates the fewest digits
    # currently in its row, column and box (ties random), until no cell changes; the set W and its size never change
    for _ in range(12):
        changed = False
        for c in rng.permutation(list(W)):
            opts = legal(int(c)); peers = g[UNITS[int(c)]]
            costs = [int((peers == d).sum()) for d in opts]; m = min(costs)
            if int((peers == g[c]).sum()) > m:
                g[c] = int(rng.choice([o for o, k_ in zip(opts, costs) if k_ == m])); W[int(c)] = int(g[c]); changed = True
        if not changed: break
    return g.reshape(9, 9), len(W)

def sample_consistent_best(puz, sol, k, rng, restarts=8):
    """The most consistent of `restarts` seeded runs of sample_consistent (selection on the grid's conflict-free share only)."""
    best = None
    for r in range(restarts):
        g, kk = sample_consistent(puz, sol, k, np.random.default_rng(rng.integers(2**31)))
        f = conflict_free_share(g, puz, sol)[0] if kk else -1.0
        if best is None or f > best[0]: best = (f, g, kk)
    return best[1], best[2]

def conflict_free_share(grid, puz, sol):
    """Share of wrong empty-cell digits that duplicate NO digit in their row, column or box (the complement of the paper's general
    duplicate measure); also the count of wrong digits."""
    import lens_repair_radius as RR
    grid, puz, sol = (np.asarray(a).reshape(-1, 9, 9) for a in (grid, puz, sol))
    cm = RR.conflict_mask(grid); wrong = (grid != sol) & (puz == 0); n = int(wrong.sum())
    return (float(1 - (cm & wrong).sum() / n) if n else float("nan")), n

def letter_p1c(G):
    if G is None: return "UNDEFINED"
    if G >= C_HIGH: return "CONSISTENCY-EXPLAINS"
    if G <= C_LOW: return "MODEL-DIGITS-SPECIAL"
    return "MIXED"


# ---------------- inputs ----------------
def prepare():
    dst = OUT / "inputs/p1c_sources.npz"
    if dst.exists(): print("prepared inputs already present"); return
    with np.load(P1 / "inputs/p1_sources.npz", allow_pickle=False) as d:
        ids, puz, sol, eqr, uni, leg, counts, bins = (d[k] for k in ("ids", "puz", "sol", "eqr", "uniform_0", "legal_random_0", "wrong_count", "rating_bin"))
        g128, g256 = d["guess_sa128"], d["guess_sa256"]
    n = len(ids); src = {}; stats = {}
    for j in range(DRAWS):
        g = np.zeros_like(sol); assigned = np.zeros(n, int)
        for i in range(n):
            g[i], assigned[i] = sample_consistent_best(puz[i], sol[i], int(counts[i]), np.random.default_rng([SEED, int(ids[i]), 31, j]))
        src[f"consistent_random_{j}"] = g
        stats[f"consistent_random_{j}_deficit_puzzles"] = int((assigned < counts).sum())
        assert np.array_equal(g[puz != 0], puz[puz != 0]) and np.isin(g, np.arange(1, 10)).all()
        assert np.all(((g != sol) & (puz == 0)).sum((1, 2)) == assigned)
        share, nw = clue_conflict_share(g, puz, sol); assert share == 0.0, "a consistent draw conflicts with a given"
        stats[f"consistent_random_{j}_conflict_free_share"] = conflict_free_share(g, puz, sol)[0]
    el = counts > 0
    per = lambda g: np.array([conflict_free_share(g[i], puz[i], sol[i])[0] for i in range(n)])
    for name, g in (("eqr", eqr), ("uniform_0", uni), ("legal_random_0", leg), ("guess_sa128", g128), ("guess_sa256", g256), ("consistent_random_0", src["consistent_random_0"])):
        f = per(g); stats[f"{name}_conflict_free_share_pooled"] = conflict_free_share(g, puz, sol)[0]
        stats[f"{name}_conflict_free_share_per_puzzle_mean"] = float(np.nanmean(f[el])); stats[f"{name}_conflict_free_share_quartiles"] = [float(x) for x in np.nanpercentile(f[el], [25, 50, 75])]
    P.write_npz if hasattr(P, "write_npz") else None
    import attention_transfer_study as ATS
    ATS.write_npz(dst, ids=ids, puz=puz, sol=sol, eqr=eqr, wrong_count=counts, rating_bin=bins, **src)
    ATS.write_json(OUT / "inputs/p1c_sources.json", dict(created=utc(), seed=SEED, draws=DRAWS, conditions=list(CONDS), stats=stats,
                                                           sha256_p1_inputs=ATS.sha(P1 / "inputs/p1_sources.npz")))
    print(json.dumps(stats, indent=1))


# ---------------- the receivers ----------------
class R3(P.R2):
    def __init__(self, width): super().__init__(width, out=OUT)
    def repair_group(self, smoke):
        group = "repair"
        with np.load(INPUTS, allow_pickle=False) as d: data = {k: d[k] for k in d.files}
        ids, puz, sol = data["ids"], data["puz"], data["sol"]
        n = 16 if smoke else len(ids); steps = 2 if smoke else STEPS
        for c in CONDS:
            for b in range(0, n, BATCH):
                sl = slice(b, min(b + BATCH, n)); ii, pp, ss, gg = ids[sl], puz[sl], sol[sl], data[c][sl]
                path = self.dir / group / c / f"batch_{b:04d}.npz"
                if self.cached(path, ii, group, c): continue
                started, cpu = time.monotonic(), time.process_time()
                values, _, extra = self.trajectory(pp, ii, initial=self.starting_state(gg), steps=steps, progress=dict(group=group, condition=c, batch_start=b))
                self.save(path, values, pp, ss, ii, group, c, started, cpu, extra, supplied=gg)
    def run(self, smoke):
        try:
            self.gate(smoke); self.repair_group(smoke)
            self.ATS.verify_sources(); self.status("inference_complete", completed=utc()); self.ATS.log("model_complete", width=self.width, group="repair_p1c")
        except BaseException as exc:
            self.status("failed", error=repr(exc)); raise

def port(receiver, smoke):
    import lens_repair_radius as RR
    RR.T_TOTAL = 2 if smoke else STEPS
    dst = OUT / f"port_{receiver}.npz"
    if dst.exists(): print(f"SKIP {dst.name} (done)"); return
    with np.load(INPUTS, allow_pickle=False) as d: data = {k: d[k] for k in d.files}
    ids, puz, sol = data["ids"], data["puz"], data["sol"]; n = 16 if smoke else len(ids)
    m = RR.Model(receiver); f = RR.embedder(m); t0 = time.time()
    Pz = np.tile(np.arange(n), len(PORT_SOURCES)); SRC = np.repeat(np.arange(len(PORT_SOURCES)), n)
    start = np.concatenate([data[s][:n] for s in PORT_SOURCES]).astype(np.int32); e0 = RR.err_frac(start, sol[Pz], puz[Pz] == 0)
    a = RR.run_rows(m, puz[Pz].astype(np.int32), sol[Pz].astype(np.int32), lambda rows: f(start[rows]), None, f"{receiver} P1c")
    meta = dict(receiver=receiver, ckpt=RR.MODELS[receiver][0], step=m.step, n=n, t_total=RR.T_TOTAL, sources=list(PORT_SOURCES), created=utc(), smoke=smoke)
    np.savez(OUT / f"port_{receiver}.tmp.npz", idx=ids[:n], puzzle=Pz, source=SRC, sources=np.asarray(PORT_SOURCES), e0=e0, meta=json.dumps(meta), wall=time.time() - t0, **a)
    os.replace(OUT / f"port_{receiver}.tmp.npz", dst); print(f"DONE {dst.name} ({time.time() - t0:.0f}s)", flush=True)


# ---------------- report (the registered rule) ----------------
def report():
    import attention_transfer_study as ATS
    with np.load(OUT / "inputs/p1c_sources.npz", allow_pickle=False) as d: el = d["wrong_count"] > 0; bins = d["rating_bin"]; n_all = len(el)
    p1 = json.loads((P1 / "report.json").read_text()); stats = json.loads((OUT / "inputs/p1c_sources.json").read_text())["stats"]
    lines = ["P1c REPORT (rule registered 2026-09-26; consistent random corruptions vs the model's own errors)",
             f"conflict-free share of wrong digits (pooled): eqr {stats['eqr_conflict_free_share_pooled']:.3f}, consistent draw 0 {stats['consistent_random_0_conflict_free_share_pooled']:.3f}, legal_random_0 {stats['legal_random_0_conflict_free_share_pooled']:.3f}, uniform_0 {stats['uniform_0_conflict_free_share_pooled']:.3f}"]
    out = {}
    for w in P.WIDTHS:
        new = P.load_group(w, "repair", CONDS, OUT); ref = P.load_group(w, "repair", ("eqr",), ATS.OUT)
        if len(new) < DRAWS or not ref: lines.append(f"attention_{w}: not complete"); continue
        e_eqr, e_uni = p1[f"attention_{w}"]["e1_eqr"], p1[f"attention_{w}"]["e1_uniform"]
        e1 = lambda r: float(r["e1"][el].mean()); e_con = float(np.mean([e1(new[c]) for c in CONDS])); G = gap_closure(e_eqr, e_uni, e_con)
        exc = np.mean([new[c]["exact"][el] for c in CONDS], axis=0); m, lo, hi = paired_bootstrap(exc, ref["eqr"]["exact"][el].astype(float), bins[el], reps=10000)
        b = int((new[CONDS[0]]["exact"][el] & ~ref["eqr"]["exact"][el]).sum()); c_ = int((~new[CONDS[0]]["exact"][el] & ref["eqr"]["exact"][el]).sum())
        out[f"attention_{w}"] = dict(e1_eqr=e_eqr, e1_uniform=e_uni, e1_consistent=e_con, per_draw=[e1(new[c]) for c in CONDS], G=G, letter=letter_p1c(G), endpoint_consistent_minus_eqr_pp=100*m, ci=[100*lo, 100*hi], draw0_mcnemar_p=mcnemar_exact(b, c_), exact_eqr=float(ref["eqr"]["exact"][el].mean()), exact_consistent=float(exc.mean()))
        lines.append(f"attention_{w}: E1 eqr {100*e_eqr:.2f} uniform {100*e_uni:.2f} consistent {100*e_con:.2f} (draws {[round(100*x,1) for x in out[f'attention_{w}']['per_draw']]}) | G {G:.3f} -> {letter_p1c(G)} | exact16 eqr {100*out[f'attention_{w}']['exact_eqr']:.1f} % consistent {100*float(exc.mean()):.1f} %; diff {100*m:+.2f} pp [{100*lo:+.2f}, {100*hi:+.2f}], draw0 McNemar p {mcnemar_exact(b, c_):.3f}")
    for rec in ("C5", "EQR"):
        p = OUT / f"port_{rec}.npz"; q = P1 / f"port_{rec}.npz"
        if not p.exists() or not q.exists(): lines.append(f"port {rec}: not complete"); continue
        d = np.load(p, allow_pickle=True); srcs = list(d["sources"]); Pz, S = d["puzzle"], d["source"]; e1v = d["e"][:, 0]; ex = d["ex16"]
        d0 = np.load(q, allow_pickle=True); s0 = list(d0["sources"]); P0, S0 = d0["puzzle"], d0["source"]; e10 = d0["e"][:, 0]
        m1 = lambda name, arr: float(np.nanmean(arr[(S == srcs.index(name)) & el[Pz]])); m0 = lambda name, arr: float(np.nanmean(arr[(S0 == s0.index(name)) & el[P0]]))
        e_eqr, e_uni = m1("eqr", e1v), m0("uniform_0", e10); e_con = float(np.mean([m1(c, e1v) for c in CONDS])); G = gap_closure(e_eqr, e_uni, e_con)
        out[f"port_{rec}"] = dict(e1_eqr=e_eqr, e1_uniform=e_uni, e1_consistent=e_con, G=G, letter=letter_p1c(G), exact_eqr=m1("eqr", ex.astype(float)), exact_consistent=float(np.mean([m1(c, ex.astype(float)) for c in CONDS])), eqr_in_process_vs_p1=m0("eqr", e10))
        lines.append(f"port {rec}: E1 eqr {100*e_eqr:.2f} (P1 run {100*m0('eqr', e10):.2f}) uniform {100*e_uni:.2f} consistent {100*e_con:.2f} | G {G:.3f} -> {letter_p1c(G)} | exact16 eqr {100*out[f'port_{rec}']['exact_eqr']:.1f} % consistent {100*out[f'port_{rec}']['exact_consistent']:.1f} %")
    (OUT / "report.txt").write_text("\n".join(lines) + "\n"); (OUT / "report.json").write_text(json.dumps(out, indent=1, default=float)); print("\n".join(lines))


# ---------------- selftest ----------------
def selftest():
    sol = np.array([[(3 * (r % 3) + r // 3 + c) % 9 + 1 for c in range(9)] for r in range(9)])
    rng = np.random.default_rng(0); puz = np.where(rng.random((9, 9)) < 0.35, sol, 0)
    # holders: exactly the peers holding d, none of them the cell itself, at most three
    for cell in range(0, 81, 7):
        for d in range(1, 10):
            hs = holders(sol, cell, d); assert cell not in hs and len(hs) <= 3
            assert all(sol.reshape(81)[h] == d and UNITS[cell][h] for h in hs)
            if d != sol.reshape(81)[cell]: assert len(hs) >= 1          # a complete grid holds every other digit somewhere in the units
    g, k = sample_consistent(puz, sol, 20, np.random.default_rng(1))
    assert k == 20 and ((g != sol) & (puz == 0)).sum() == 20 and np.array_equal(g[puz != 0], puz[puz != 0]) and clue_conflict_share(g, puz, sol)[0] == 0.0
    # consistency: a large share of the wrong digits conflict with nothing (their holders are wrong too), unlike a legal random draw
    from rebuttal_p1p2 import sample_legal_random
    gl, _, _ = sample_legal_random(puz, sol, 20, np.random.default_rng(2))
    fc, _ = conflict_free_share(g, puz, sol); fl, _ = conflict_free_share(gl, puz, sol)
    assert fc > fl + 0.1, (fc, fl)          # the toy puzzle has 35 % givens; the real target is checked at prepare against EqR
    # the closure property holds for every wrong digit that is conflict-free: all its holders are wrong
    import lens_repair_radius as RR
    cm = RR.conflict_mask(g[None])[0].reshape(81); wrong = ((g != sol) & (puz == 0)).reshape(81)
    for c in np.where(wrong & ~cm)[0]:
        assert all(wrong[h] for h in holders(sol, int(c), int(g.reshape(81)[c])))
    # a mutant that ignores holders (plain legal random) fails the consistency check
    assert not (fl > fc)
    # letters at the boundaries
    assert letter_p1c(gap_closure(0.36, 0.04, 0.30)) == "CONSISTENCY-EXPLAINS" and letter_p1c(gap_closure(0.36, 0.04, 0.10)) == "MODEL-DIGITS-SPECIAL"
    assert letter_p1c(gap_closure(0.36, 0.04, 0.20)) == "MIXED" and letter_p1c(None) == "UNDEFINED"
    print(f"selftest OK: holders, consistent sampler (conflict-free share {fc:.2f} vs legal {fl:.2f}), closure property, mutant, letters")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", nargs="?", choices=["prepare", "run", "port", "report"]); ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--width", type=int, choices=P.WIDTHS); ap.add_argument("--receiver", choices=["C5", "EQR"]); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.selftest: selftest(); return
    if a.action == "prepare": prepare()
    elif a.action == "run":
        if a.width is None: ap.error("--width required")
        global OUT
        if a.smoke: OUT = Path(os.environ.get("P1C_SMOKE_OUT", str(OUT) + "_smoke")); OUT.mkdir(parents=True, exist_ok=True)
        R3(a.width).run(a.smoke)
    elif a.action == "port":
        if a.receiver is None: ap.error("--receiver required")
        if a.smoke: OUT = Path(os.environ.get("P1C_SMOKE_OUT", str(OUT) + "_smoke")); OUT.mkdir(parents=True, exist_ok=True)
        port(a.receiver, a.smoke)
    elif a.action == "report": report()
    else: ap.print_help()


if __name__ == "__main__":
    main()
