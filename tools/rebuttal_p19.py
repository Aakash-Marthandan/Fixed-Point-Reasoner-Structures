#!/usr/bin/env python3
# Ledger: THE DISCUSSION-PERIOD EXPERIMENT P19 — a collective threshold: how much of the error must be corrected at once for the
# network to accept the correction and complete the rest? (registration Documentation/Note_2026-10-03_Rebuttal_P19_Registration.md,
# written before any row). MEASUREMENT, $0, inference only on the banked attention checkpoints; imports tools/rebuttal_p18.py (R9,
# cell relabels) and tools/rebuttal_p15.py (states, rollouts, split) unchanged.
"""P19: in a state after iteration t, a random fraction f of the wrong cells is corrected at once (cell relabels: slow and fast field
vectors exchanged between the displayed and the solution digit), f in {0.1, ..., 1.0}, two draws per f; four outer iterations through
the release step. Primary: whole-grid completion at t+1 as a function of f, normalized between no edit and the full correction;
the network's repair of the wrong cells left uncorrected; the sharpness of the transition."""
import argparse, json, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import rebuttal_p15 as Q
import rebuttal_p18 as R18
import rebuttal_p1p2 as P

OUT = ROOT / "runs/analysis/rebuttal_20261003"
SEED = 20261005
TS = (1, 2)
K = Q.K
FRACS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
DRAWS = 2
SHARP, GRADUAL, MIN_RANGE = 0.30, 0.60, 0.20


def utc(): return datetime.now(timezone.utc).isoformat()


def crossing(fs, z, level):
    """Smallest f at which the piecewise-linear curve z(f) reaches `level` (fs ascending, z at fs); nan if never."""
    for a in range(len(fs)):
        if z[a] >= level:
            if a == 0: return float(fs[0])
            f0, f1, z0, z1 = fs[a - 1], fs[a], z[a - 1], z[a]
            return float(f0 + (level - z0) * (f1 - f0) / (z1 - z0)) if z1 != z0 else float(f1)
    return float("nan")


def letter(fs, y, y0):
    """y: completion at each f (fs includes 1.0 last); y0: completion with no edit. Normalized z = (y - y0)/(y(1) - y0)."""
    rng = y[-1] - y0
    if not np.isfinite(rng) or rng < MIN_RANGE: return "UNDEFINED", dict(f10=float("nan"), f50=float("nan"), f90=float("nan"), range=rng)
    z = [(v - y0) / rng for v in y]
    f10, f50, f90 = crossing(fs, z, 0.1), crossing(fs, z, 0.5), crossing(fs, z, 0.9)
    span = f90 - f10
    lt = "SHARP" if span <= SHARP else ("GRADUAL" if span >= GRADUAL else "MIXED")
    return lt, dict(f10=f10, f50=f50, f90=f90, span=span, range=rng, z=z)


class R10(R18.R9):
    def __init__(self, width, out=OUT):
        super().__init__(width, out)
        self.meta.update(study="rebuttal_20261003", pipeline="tools/rebuttal_p19.py over tools/rebuttal_p18.py over tools/rebuttal_p16.py over tools/rebuttal_p15.py")
        self.ATS.write_json(self.dir / "runtime.json", self.meta)

    def run(self, smoke=False):
        try:
            ATS, STUDY = P.study_paths()
            with np.load(STUDY / "inputs/interventions.npz", allow_pickle=False) as d: ids_all, puz_all, sol_all = d["ids"], d["puz"], d["sol"]
            disc = Q.discovery_ids(); dirr = self.dir / ("sweep_smoke" if smoke else "sweep")
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
                    rec = self.sweep_chunk(ids[:n], puz[:n], sol[:n], st[t - 1]["z"][:n], np.asarray(x)[:n], refpred[t:t + K, :n], t, disc, progress=dict(batch=b0, t=t))
                    rec["meta"] = json.dumps(dict(self.meta, t=t, batch_start=b0, n=n, seconds=time.monotonic() - started, created=utc(), smoke=smoke))
                    if smoke:
                        print(json.dumps(dict(release_states_bitwise=g1, **json.loads(rec["gate_summary"])))); return
                    self.ATS.write_npz(paths[t], **rec); self.ATS.write_json(paths[t].with_suffix(".sha256.json"), dict(sha256=self.ATS.sha(paths[t])))
                    self.ATS.log("chunk_complete", width=self.width, batch=b0, t=t, states=int(len(rec["state_ids"])), variants=int(len(rec["var_frac"])), seconds=round(time.monotonic() - started, 1))
            self.ATS.verify_sources(); self.status("inference_complete", completed=utc()); self.ATS.log("model_complete", width=self.width, group="p19")
        except BaseException as exc:
            self.status("failed", error=repr(exc)); raise

    def sweep_chunk(self, ids, puz, sol, z, xcan, refpred, t, disc, progress=None):
        lm = self.lm; specs, V = [], dict(state=[], frac=[], draw=[], cells=[])
        S = dict(ids=[], t=[], split=[], grid=[], sol=[], puz=[], n_wrong=[]); per = {}; disp_fail = 0
        for b in range(len(ids)):
            pid = int(ids[b]); pz, so = puz[b].reshape(81), sol[b].reshape(81); h, l = z[b, 0], z[b, 1]
            grid = np.einsum("fsw,w->sf", h.astype(np.float64), lm).argmax(1) + 1
            W = sorted(int(i) for i in np.where((pz == 0) & (grid != so))[0])
            if not W: continue
            si = len(S["ids"]); per[si] = dict(b=b, h=h, l=l, grid=grid, sol=so)
            for key, val in (("ids", pid), ("t", t), ("split", 0 if pid in disc else 1), ("grid", grid.astype(np.int8)), ("sol", so.astype(np.int8)), ("puz", pz.astype(np.int8)), ("n_wrong", len(W))):
                S[key].append(val)
            rng = np.random.default_rng([SEED, pid, t])
            def add(frac, draw, cells):
                specs.append((si, cells)); V["state"].append(si); V["frac"].append(frac); V["draw"].append(draw)
                cc = np.full(81, -1, np.int16); cc[:len(cells)] = sorted(cells); V["cells"].append(cc)
            add(0.0, 0, [])
            for f in FRACS:
                m = max(1, int(round(f * len(W))))
                for d in range(1 if f == 1.0 else DRAWS):
                    add(f, d, sorted(int(c) for c in rng.choice(W, size=m, replace=False)))

        def build(spec):
            nonlocal disp_fail
            si, cells = spec; q = per[si]
            if not cells: return q["h"], q["l"]
            h2, l2 = R18.relabel_cells(q["h"], q["l"], cells, q["grid"], q["sol"])
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
        assert disp_fail == 0, f"gate: {disp_fail} relabels did not produce the designed display"
        pred = (out.argmax(-1) + 1).astype(np.int8); st = np.asarray(V["state"]); fr = np.asarray(V["frac"]); g_fail = 0
        for v in np.where(fr == 0.0)[0]:
            if not np.array_equal(pred[v], refpred[:, per[int(st[v])]["b"]].reshape(K, 81).astype(np.int8)): g_fail += 1
        assert g_fail == 0, f"gate: {g_fail} no-edit rollouts differ from the study's displayed grids"
        summary = dict(states=len(S["ids"]), variants=len(specs), relabel_display_failures=disp_fail, noedit_gate_failures=g_fail)
        return dict(state_ids=np.asarray(S["ids"], np.int64), state_t=np.asarray(S["t"], np.int8), state_split=np.asarray(S["split"], np.int8),
                    state_grid=np.stack(S["grid"]), state_sol=np.stack(S["sol"]), state_puz=np.stack(S["puz"]), state_nwrong=np.asarray(S["n_wrong"], np.int16),
                    var_state=st.astype(np.int32), var_frac=fr.astype(np.float32), var_draw=np.asarray(V["draw"], np.int8), var_cells=np.stack(V["cells"]),
                    pred=pred, gate_summary=json.dumps(summary))


# ---------------- the read ----------------
def load_rows(w, base=OUT):
    parts = sorted((base / f"attention_{w}/sweep").glob("b*_t*.npz"))
    if not parts: return None
    rows = []
    for pth in parts:
        import hashlib
        side = pth.with_suffix("").with_suffix(".sha256.json")
        assert hashlib.sha256(pth.read_bytes()).hexdigest() == json.loads(side.read_text())["sha256"], f"digest {pth}"
        with np.load(pth, allow_pickle=False) as d: rows.append({k_: d[k_] for k_ in d.files if k_ not in ("meta", "gate_summary")})
    return rows


def curves(rows, k, split, t=None):
    """Per f: whole-grid completion at t+k; the edited cells kept; the network's repair of the uncorrected wrong cells (normalized)."""
    ex = {f: [] for f in (0.0,) + FRACS}; kept = {f: [] for f in FRACS}; rest = {f: ([], []) for f in FRACS}
    for r in rows:
        fr, st = r["var_frac"], r["var_state"]; Pk = r["pred"][:, k - 1].astype(int)
        for s in range(len(r["state_ids"])):
            if int(r["state_split"][s]) != split or (t is not None and int(r["state_t"][s]) != t): continue
            m = st == s; sol = r["state_sol"][s].astype(int); grid = r["state_grid"][s].astype(int); puz = r["state_puz"][s].astype(int)
            W = np.where((puz == 0) & (grid != sol))[0]; v0 = np.where(m & (fr == 0.0))[0][0]
            ex[0.0].append(int((Pk[v0] == sol).all()))
            for v in np.where(m & (fr > 0))[0]:
                f = float(np.round(fr[v], 1)); cc = [int(c) for c in r["var_cells"][v] if c >= 0]; un = [c for c in W if c not in set(cc)]
                ex[f].append(int((Pk[v] == sol).all())); kept[f] += [int(Pk[v, c] == sol[c]) for c in cc]
                rest[f][0].extend(int(Pk[v, c] == sol[c]) for c in un); rest[f][1].extend(int(Pk[v0, c] == sol[c]) for c in un)
    y = {f: float(np.mean(v)) if v else float("nan") for f, v in ex.items()}
    kp = {f: float(np.mean(v)) if v else float("nan") for f, v in kept.items()}
    rs = {f: R18.norm_gain(a, b) if a else float("nan") for f, (a, b) in rest.items()}
    return y, kp, rs, {f: len(v) for f, v in ex.items()}


def report(base=OUT, widths=Q.WIDTHS):
    lines = ["P19 REPORT (rules registered 2026-10-03): the collective threshold; letters on CONFIRMATION, k = 1 (t = 1 and 2 pooled)"]; out = {}
    for w in widths:
        rows = load_rows(w, base)
        if rows is None: lines.append(f"attention_{w}: not run"); continue
        res = {}
        for k in range(1, K + 1):
            res[k] = {}
            for split, name in ((1, "confirmation"), (0, "discovery")):
                y, kp, rs, n = curves(rows, k, split)
                lt, st = letter(list(FRACS), [y[f] for f in FRACS], y[0.0])
                res[k][name] = dict(completion=y, kept=kp, rest_repair=rs, n=n, letter=lt, stats=st)
                lines.append(f"attention_{w} k={k} {name}: completion " + " ".join(f"{f:.1f}:{y[f]:.2f}" for f in (0.0,) + FRACS)
                             + f" | rest repaired " + " ".join(f"{f:.1f}:{rs[f]:.2f}" for f in FRACS) + f" | f10 {st['f10']:.2f} f50 {st['f50']:.2f} f90 {st['f90']:.2f}"
                             + (f" || {lt}" if (k == 1 and split == 1) else ""))
        out[f"attention_{w}"] = res
    (base / "report.txt").write_text("\n".join(lines) + "\n"); (base / "report.json").write_text(json.dumps(out, indent=1, default=float)); print("\n".join(lines))


def selftest():
    fs = list(FRACS)
    sharp = [0.0, 0.0, 0.0, 0.0, 0.1, 0.9, 1.0, 1.0, 1.0, 1.0]
    lt, st = letter(fs, sharp, 0.0); assert lt == "SHARP" and 0.4 < st["f50"] < 0.6, (lt, st)
    grad = [0.1 * i for i in range(1, 11)]
    lt2, st2 = letter(fs, grad, 0.0); assert lt2 == "GRADUAL", (lt2, st2)
    assert letter(fs, [0.1] * 10, 0.05)[0] == "UNDEFINED"
    assert abs(crossing([0.1, 0.2, 0.3], [0.0, 0.5, 1.0], 0.25) - 0.15) < 1e-9 and np.isnan(crossing([0.1, 0.2], [0.0, 0.2], 0.5))
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
        R10(a.width, Path(a.out) if a.out else OUT).run(a.smoke)
    else: ap.print_help()


if __name__ == "__main__":
    main()
