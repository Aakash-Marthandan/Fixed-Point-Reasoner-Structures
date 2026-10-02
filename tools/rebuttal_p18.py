#!/usr/bin/env python3
# Ledger: THE DISCUSSION-PERIOD EXPERIMENT P18 — coordinated revision: are errors held in place by their mutually consistent partners?
# (registration Documentation/Note_2026-10-02_Rebuttal_P18_Registration.md, written before any row). MEASUREMENT, $0, inference only on
# the banked attention checkpoints; imports tools/rebuttal_p16.py (cell relabel) and tools/rebuttal_p15.py (states, rollouts, split) unchanged.
"""P18: in a state after iteration t, the closure of a wrong cell i is the smallest set containing i that is closed under 'a wrong peer
displays the solution digit of a member'; correcting a closure creates no duplicate. Edits (each cell's slow and fast field vectors
exchanged between its displayed digit and its solution digit): a small closure (2-6 cells) jointly; a size-matched random set of wrong
cells jointly; one cell of the closure alone; the closure minus one cell; a large closure (>= 7) and a size-matched random set; every
wrong cell. Rules: COORDINATED-REVISION (closure kept beyond size-matched random sets) and COMPLETES-THE-MOVE (the omitted cell of a
partial closure is corrected by the network)."""
import argparse, json, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import rebuttal_p15 as Q
import rebuttal_p16 as R
import rebuttal_p1p2 as P

OUT = ROOT / "runs/analysis/rebuttal_20261002c"
SEED = 20261004
TS = (1, 2)
K = Q.K
SMALL = (2, 6)
KINDS = ("noedit", "closure", "random", "random2", "single", "partial", "large_closure", "large_random", "all_wrong")
KIND = {k: i for i, k in enumerate(KINDS)}
D_HIGH, D_LOW, M_HIGH = 0.20, 0.05, 0.20


def utc(): return datetime.now(timezone.utc).isoformat()


def closure(i, grid, sol, W):
    """Smallest set containing i, closed under: a wrong peer displaying the solution digit of a member."""
    seen, st = set(), [int(i)]
    while st:
        u = st.pop()
        if u in seen: continue
        seen.add(u); st.extend(int(j) for j in np.where(P.UNITS[u] & (grid == sol[u]))[0] if int(j) in W)
    return frozenset(seen)


def creates_duplicate(grid, sol, cells):
    """True if setting the given cells to their solution digits leaves any of them duplicating a peer's displayed digit."""
    g = grid.copy(); cells = list(cells); g[cells] = sol[cells]
    return any(bool((g[P.UNITS[c]] == g[c]).any()) for c in cells)


def relabel_cells(h, l, cells, grid, sol):
    for c in cells: h, l = R.cell_relabel(h, l, int(c), int(grid[c]), int(sol[c]))
    return h, l


def norm_gain(c_edit, c_no):
    a, b = float(np.mean(c_edit)), float(np.mean(c_no))
    return (a - b) / (1 - b) if b < 1 else float("nan")


def letter_coord(e_cl, e_rd):
    if not (np.isfinite(e_cl) and np.isfinite(e_rd)): return "UNDEFINED"
    d = e_cl - e_rd
    if d >= D_HIGH: return "COORDINATED-REVISION"
    if d <= D_LOW: return "INDEPENDENT"
    return "MIXED"


def letter_move(m):
    if not np.isfinite(m): return "UNDEFINED"
    return "COMPLETES-THE-MOVE" if m >= M_HIGH else ("DOES-NOT-COMPLETE" if m <= D_LOW else "MIXED")


class R9(R.R8):
    def __init__(self, width, out=OUT):
        super().__init__(width, out)
        self.meta.update(study="rebuttal_20261002c", pipeline="tools/rebuttal_p18.py over tools/rebuttal_p16.py over tools/rebuttal_p15.py")
        self.ATS.write_json(self.dir / "runtime.json", self.meta)

    def run(self, smoke=False):
        try:
            ATS, STUDY = P.study_paths()
            with np.load(STUDY / "inputs/interventions.npz", allow_pickle=False) as d: ids_all, puz_all, sol_all = d["ids"], d["puz"], d["sol"]
            disc = Q.discovery_ids(); dirr = self.dir / ("moves_smoke" if smoke else "moves")
            for b0 in range(0, len(ids_all), 128):
                sl = slice(b0, b0 + 128); ids, puz, sol = ids_all[sl], puz_all[sl], sol_all[sl]
                paths = {t: dirr / f"b{b0:04d}_t{t}.npz" for t in TS}
                if not smoke and all(self.cached_any(paths[t]) for t in TS): continue
                x, st = self.release_states(puz)
                with np.load(STUDY / f"attention_{self.width}/interventions/intact/batch_{b0:04d}.npz", allow_pickle=False) as d: ref, refpred = d["logits"], d["pred"]
                g1 = all(np.array_equal(st[t]["logits"], ref[t]) for t in range(max(Q.TS))); assert g1, "gate 1: release states differ from the study's logits"
                n = 16 if smoke else len(ids)
                for t in (TS[:1] if smoke else TS):
                    if not smoke and self.cached_any(paths[t]): continue
                    started = time.monotonic()
                    rec = self.move_chunk(ids[:n], puz[:n], sol[:n], st[t - 1]["z"][:n], np.asarray(x)[:n], refpred[t:t + K, :n], t, disc, progress=dict(batch=b0, t=t))
                    rec["meta"] = json.dumps(dict(self.meta, t=t, batch_start=b0, n=n, seconds=time.monotonic() - started, created=utc(), smoke=smoke))
                    if smoke:
                        print(json.dumps(dict(release_states_bitwise=g1, **json.loads(rec["gate_summary"])))); return
                    self.ATS.write_npz(paths[t], **rec); self.ATS.write_json(paths[t].with_suffix(".sha256.json"), dict(sha256=self.ATS.sha(paths[t])))
                    self.ATS.log("chunk_complete", width=self.width, batch=b0, t=t, states=int(len(rec["state_ids"])), variants=int(len(rec["var_kind"])), seconds=round(time.monotonic() - started, 1))
            self.ATS.verify_sources(); self.status("inference_complete", completed=utc()); self.ATS.log("model_complete", width=self.width, group="p18")
        except BaseException as exc:
            self.status("failed", error=repr(exc)); raise

    def move_chunk(self, ids, puz, sol, z, xcan, refpred, t, disc, progress=None):
        lm = self.lm; specs, V = [], dict(state=[], kind=[], move=[], cells=[], omitted=[])
        S = dict(ids=[], t=[], split=[], grid=[], sol=[], puz=[], n_wrong=[]); per = {}; dup_fail = 0; disp_fail = 0
        for b in range(len(ids)):
            pid = int(ids[b]); pz, so = puz[b].reshape(81), sol[b].reshape(81); h, l = z[b, 0], z[b, 1]
            grid = np.einsum("fsw,w->sf", h.astype(np.float64), lm).argmax(1) + 1
            W = set(int(i) for i in np.where((pz == 0) & (grid != so))[0])
            if not W: continue
            clos = sorted(set(closure(i, grid, so, W) for i in W), key=lambda c: (len(c), sorted(c)))
            small = [c for c in clos if SMALL[0] <= len(c) <= SMALL[1]]; large = [c for c in clos if len(c) > SMALL[1]]
            if not small and not large: continue
            si = len(S["ids"]); per[si] = dict(b=b, h=h, l=l, grid=grid, sol=so)
            for key, val in (("ids", pid), ("t", t), ("split", 0 if pid in disc else 1), ("grid", grid.astype(np.int8)), ("sol", so.astype(np.int8)), ("puz", pz.astype(np.int8)), ("n_wrong", len(W))):
                S[key].append(val)
            rng = np.random.default_rng([SEED, pid, t])
            def add(kind, mv, cells, omitted=-1):
                specs.append((si, cells)); V["state"].append(si); V["kind"].append(KIND[kind]); V["move"].append(mv)
                cc = np.full(81, -1, np.int16); cc[:len(cells)] = sorted(cells); V["cells"].append(cc); V["omitted"].append(omitted)
            add("noedit", -1, [])
            picks = [small[i] for i in rng.permutation(len(small))[:2]] if small else []
            for mv, c in enumerate(picks):
                c = sorted(c)
                if creates_duplicate(grid, so, c): dup_fail += 1
                add("closure", mv, c)
                rest = sorted(W - set(c)); pool = rest if len(rest) >= len(c) else sorted(W)
                for kind in ("random", "random2"): add(kind, mv, sorted(int(x) for x in rng.choice(pool, size=len(c), replace=False)))
                add("single", mv, [int(rng.choice(c))])
                om = int(rng.choice(c)); add("partial", mv, [x for x in c if x != om], om)
            if large:
                c = sorted(large[int(rng.integers(len(large)))])
                if creates_duplicate(grid, so, c): dup_fail += 1
                add("large_closure", 9, c); add("large_random", 9, sorted(int(x) for x in rng.choice(sorted(W), size=len(c), replace=False)))
            add("all_wrong", 10, sorted(W))

        def build(spec):
            nonlocal disp_fail
            si, cells = spec; q = per[si]
            if not cells: return q["h"], q["l"]
            h2, l2 = relabel_cells(q["h"], q["l"], cells, q["grid"], q["sol"])
            want = q["grid"].copy(); want[list(cells)] = q["sol"][list(cells)]
            if not np.array_equal(np.einsum("fsw,w->sf", h2, self.lm32).argmax(1) + 1, want): disp_fail += 1
            return h2, l2

        out = np.zeros((len(specs), K, 81, 9), np.float32)
        for c0 in range(0, len(specs), Q.VBATCH):
            chunk = specs[c0:c0 + Q.VBATCH]; pad = Q.VBATCH - len(chunk); built = [build(sp) for sp in chunk]
            zz = np.stack([np.stack([u[0], u[1]]) for u in built] + [np.stack([built[0][0], built[0][1]])] * pad)
            xb = np.stack([xcan[per[sp[0]]["b"]] for sp in chunk] + [xcan[per[chunk[0][0]]["b"]]] * pad)
            out[c0:c0 + len(chunk)] = self.roll_release(zz, xb)[:len(chunk)]
            if progress: self.status("running", **progress, variants_done=c0 + len(chunk), variants=len(specs))
        assert dup_fail == 0, f"gate: {dup_fail} closures create a duplicate"
        assert disp_fail == 0, f"gate: {disp_fail} relabels did not produce the designed display"
        pred = (out.argmax(-1) + 1).astype(np.int8); st = np.asarray(V["state"]); kd = np.asarray(V["kind"]); g_fail = 0
        for v in np.where(kd == KIND["noedit"])[0]:
            if not np.array_equal(pred[v], refpred[:, per[int(st[v])]["b"]].reshape(K, 81).astype(np.int8)): g_fail += 1
        assert g_fail == 0, f"gate: {g_fail} no-edit rollouts differ from the study's displayed grids"
        summary = dict(states=len(S["ids"]), variants=len(specs), closure_duplicate_failures=dup_fail, relabel_display_failures=disp_fail, noedit_gate_failures=g_fail)
        return dict(state_ids=np.asarray(S["ids"], np.int64), state_t=np.asarray(S["t"], np.int8), state_split=np.asarray(S["split"], np.int8),
                    state_grid=np.stack(S["grid"]), state_sol=np.stack(S["sol"]), state_puz=np.stack(S["puz"]), state_nwrong=np.asarray(S["n_wrong"], np.int16),
                    var_state=st.astype(np.int32), var_kind=kd.astype(np.int8), var_move=np.asarray(V["move"], np.int8), var_cells=np.stack(V["cells"]),
                    var_omitted=np.asarray(V["omitted"], np.int16), pred=pred, gate_summary=json.dumps(summary))


# ---------------- the read ----------------
def load_rows(w, base=OUT):
    parts = sorted((base / f"attention_{w}/moves").glob("b*_t*.npz"))
    if not parts: return None
    rows = []
    for pth in parts:
        import hashlib
        side = pth.with_suffix("").with_suffix(".sha256.json")
        assert hashlib.sha256(pth.read_bytes()).hexdigest() == json.loads(side.read_text())["sha256"], f"digest {pth}"
        with np.load(pth, allow_pickle=False) as d: rows.append({k_: d[k_] for k_ in d.files if k_ not in ("meta", "gate_summary")})
    return rows


def tables(rows, k):
    """Per edited cell: correct at t+k under the edit and under no edit; per partial edit: the omitted cell's correctness; per variant: exact."""
    cells, omit, exact = [], [], []
    for r in rows:
        kd, st = r["var_kind"], r["var_state"]; Pk = r["pred"][:, k - 1].astype(int)
        for s in range(len(r["state_ids"])):
            m = st == s; sol = r["state_sol"][s].astype(int); split = int(r["state_split"][s]); t = int(r["state_t"][s])
            v0 = np.where(m & (kd == KIND["noedit"]))[0][0]
            exact.append(dict(split=split, t=t, kind="noedit", ex=int((Pk[v0] == sol).all()), size=0))
            for v in np.where(m & (kd != KIND["noedit"]))[0]:
                kind = KINDS[int(kd[v])]; cc = [int(c) for c in r["var_cells"][v] if c >= 0]
                for c in cc: cells.append(dict(split=split, t=t, kind=kind, size=len(cc), c_edit=int(Pk[v, c] == sol[c]), c_no=int(Pk[v0, c] == sol[c])))
                if kind == "partial":
                    o = int(r["var_omitted"][v]); omit.append(dict(split=split, t=t, size=len(cc) + 1, c_edit=int(Pk[v, o] == sol[o]), c_no=int(Pk[v0, o] == sol[o])))
                exact.append(dict(split=split, t=t, kind=kind, ex=int((Pk[v] == sol).all()), size=len(cc)))
    return cells, omit, exact


def summarize(cells, omit, exact, split):
    sel = lambda kind: [x for x in cells if x["split"] == split and x["kind"] == kind]
    e = {kind: norm_gain([x["c_edit"] for x in sel(kind)], [x["c_no"] for x in sel(kind)]) if sel(kind) else float("nan") for kind in KINDS[1:]}
    rnd = sel("random") + sel("random2"); e["random_pooled"] = norm_gain([x["c_edit"] for x in rnd], [x["c_no"] for x in rnd]) if rnd else float("nan")
    keep = {kind: float(np.mean([x["c_edit"] for x in sel(kind)])) if sel(kind) else float("nan") for kind in KINDS[1:]}
    om = [x for x in omit if x["split"] == split]
    m = norm_gain([x["c_edit"] for x in om], [x["c_no"] for x in om]) if om else float("nan")
    ex = {kind: float(np.mean([x["ex"] for x in exact if x["split"] == split and x["kind"] == kind])) if any(x["split"] == split and x["kind"] == kind for x in exact) else float("nan") for kind in KINDS}
    n = {kind: len(sel(kind)) for kind in KINDS[1:]}
    return dict(e=e, keep=keep, move_completion=m, n_partial=len(om), exact=ex, n_cells=n,
                letter_a=letter_coord(e["closure"], e["random_pooled"]), letter_b=letter_move(m))


def report(base=OUT, widths=Q.WIDTHS):
    lines = ["P18 REPORT (rules registered 2026-10-02): coordinated revision; letters on CONFIRMATION, k = 1"]; out = {}
    for w in widths:
        rows = load_rows(w, base)
        if rows is None: lines.append(f"attention_{w}: not run"); continue
        res = {}
        for k in range(1, K + 1):
            cells, omit, exact = tables(rows, k); res[k] = {}
            for split, name in ((1, "confirmation"), (0, "discovery")):
                s = summarize(cells, omit, exact, split); res[k][name] = s
                lines.append(f"attention_{w} k={k} {name}: e closure {s['e']['closure']:.3f} random {s['e']['random_pooled']:.3f} single {s['e']['single']:.3f} partial-cells {s['e']['partial']:.3f} large closure {s['e']['large_closure']:.3f} large random {s['e']['large_random']:.3f} all wrong {s['e']['all_wrong']:.3f}"
                             f" | omitted cell completed {s['move_completion']:.3f} (n {s['n_partial']}) | exact: none {s['exact']['noedit']:.3f} closure {s['exact']['closure']:.3f} all-wrong {s['exact']['all_wrong']:.3f}"
                             + (f" || A {s['letter_a']} B {s['letter_b']}" if (k == 1 and split == 1) else ""))
        out[f"attention_{w}"] = res
    (base / "report.txt").write_text("\n".join(lines) + "\n"); (base / "report.json").write_text(json.dumps(out, indent=1, default=float)); print("\n".join(lines))


def selftest():
    base = np.array([[(3 * (r % 3) + r // 3 + c) % 9 + 1 for c in range(9)] for r in range(9)]).reshape(81)
    puz = base.copy(); puz[:30] = 0; grid = base.copy()
    # a 2-cycle: cells 0 and 1 (same row) exchange their digits -> each displays the other's solution digit
    grid[0], grid[1] = base[1], base[0]
    W = set(int(i) for i in np.where((puz == 0) & (grid != base))[0]); assert W == {0, 1}
    c = closure(0, grid, base, W); assert c == frozenset({0, 1}) and not creates_duplicate(grid, base, c) and creates_duplicate(grid, base, [0])
    # a wrong cell whose solution digit nobody displays elsewhere: closure of size 1
    g2 = base.copy(); g2[5] = 9 if base[5] != 9 else 8
    W2 = set(int(i) for i in np.where((puz == 0) & (g2 != base))[0]); c2 = closure(5, g2, base, W2); assert len(c2) >= 1 and 5 in c2
    rng = np.random.default_rng(0); w = 8; lm = rng.standard_normal(w); h = rng.standard_normal((9, 81, w)).astype(np.float32); l = rng.standard_normal((9, 81, w)).astype(np.float32)
    h2, l2 = relabel_cells(h, l, [0, 1], grid, base); assert np.array_equal(h2[grid[0] - 1, 0], h[base[0] - 1, 0]) and np.array_equal(l2[grid[1] - 1, 1], l[base[1] - 1, 1])
    assert norm_gain([1, 1, 0, 0], [0, 0, 0, 0]) == 0.5 and letter_coord(0.5, 0.2) == "COORDINATED-REVISION" and letter_coord(0.2, 0.18) == "INDEPENDENT" and letter_coord(0.3, 0.2) == "MIXED"
    assert letter_move(0.3) == "COMPLETES-THE-MOVE" and letter_move(0.0) == "DOES-NOT-COMPLETE"
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", nargs="?", choices=["run", "report"]); ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--width", type=int, choices=Q.WIDTHS); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.selftest: selftest(); return
    if a.action == "report": report(Path(a.out) if a.out else OUT); return
    if a.action == "run":
        if a.width is None: ap.error("--width required")
        R9(a.width, Path(a.out) if a.out else OUT).run(a.smoke)
    else: ap.print_help()


if __name__ == "__main__":
    main()
