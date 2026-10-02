#!/usr/bin/env python3
# Ledger: THE DISCUSSION-PERIOD EXPERIMENT P16 (revised) — where a decision is held, and whether retained state controls how a
# changed decision propagates (registration: Documentation/Note_2026-10-02_Rebuttal_P15_P16_P17_Registration.md, Amendment 2,
# written before any P16 row). MEASUREMENT, $0, inference only on the banked attention checkpoints; imports tools/rebuttal_p15.py
# (R7: states, release-step rollouts, flip selection and split) unchanged.
"""P16: belief edits in the network's own coordinates. Digit fields share every parameter, so swapping two digits' vectors at one
cell relabels that cell's belief between the two digits. Channels: the display (readout-parallel) parts only, the kernel
(readout-orthogonal slow) parts only, the whole slow vectors, and slow plus fast. For helpful edits that create a duplicate with a
wrong peer j, a dial on j's retained commitment (kernel parts moved toward or away from the cell's field mean, scores unchanged)
tests whether retained state controls whether the change propagates into j; the same dial at a matched unrelated cell is the control."""
import argparse, json, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import rebuttal_p15 as Q
import rebuttal_p1p2 as P

OUT = ROOT / "runs/analysis/rebuttal_20261002b"
SEED = 20261003
CHANNELS = ("display_swap", "kernel_swap", "slow_swap", "cell_relabel")
BETAS = (-0.5, 0.5, 1.0)
K = Q.K
KINDS = ("noedit",) + tuple(f"edit_{c}" for c in CHANNELS) + tuple(f"dialj_{b}" for b in BETAS) + tuple(f"edit_dialj_{b}" for b in BETAS) + ("dialk_1.0", "edit_dialk_1.0")
KIND = {k: i for i, k in enumerate(KINDS)}


def utc(): return datetime.now(timezone.utc).isoformat()


# ---------------- pure edits (selftested); h, l are (F, S, w) float32; lm is (w,) ----------------
def split_par(v, lm):
    """v (..., w) -> (parallel, perpendicular) in float64."""
    v = np.asarray(v, np.float64); c = (v @ lm) / (lm @ lm); par = c[..., None] * lm
    return par, v - par


def display_swap(h, i, a, b, lm):
    h = h.copy(); pa, qa = split_par(h[a - 1, i], lm); pb, qb = split_par(h[b - 1, i], lm)
    h[a - 1, i] = (pb + qa).astype(h.dtype); h[b - 1, i] = (pa + qb).astype(h.dtype); return h


def kernel_swap(h, i, a, b, lm):
    h = h.copy(); pa, qa = split_par(h[a - 1, i], lm); pb, qb = split_par(h[b - 1, i], lm)
    h[a - 1, i] = (pa + qb).astype(h.dtype); h[b - 1, i] = (pb + qa).astype(h.dtype); return h


def slow_swap(h, i, a, b):
    h = h.copy(); h[[a - 1, b - 1], i] = h[[b - 1, a - 1], i]; return h


def cell_relabel(h, l, i, a, b):
    return slow_swap(h, i, a, b), slow_swap(l, i, a, b)


def dial(h, j, beta, lm):
    """Kernel parts at cell j moved by beta toward their field mean (beta=1: identical kernels, no hidden digit preference; beta<0: away)."""
    h = h.copy(); par, q = split_par(h[:, j], lm); m = q.mean(0, keepdims=True)
    h[:, j] = (par + q + beta * (m - q)).astype(h.dtype); return h


def commitment(h, grid, lm):
    """Per cell: distance of the displayed digit's kernel from the mean of the other fields' kernels, relative to the mean kernel norm."""
    _, q = split_par(h, lm); out = np.zeros(h.shape[1])
    for s in range(h.shape[1]):
        a = int(grid[s]) - 1; others = np.delete(q[:, s], a, axis=0)
        out[s] = np.linalg.norm(q[a, s] - others.mean(0)) / max(np.linalg.norm(q[:, s], axis=1).mean(), 1e-12)
    return out


def interaction(change_both, change_dial, change_edit, change_none):
    """The part of j's change attributable to the edit at i under a dial: P(change | edit, dial) - P(change | dial)."""
    return float(np.mean(change_both)) - float(np.mean(change_dial))


# ---------------- the runner ----------------
class R8(Q.R7):
    def __init__(self, width, out=OUT):
        super().__init__(width, out)
        self.meta.update(study="rebuttal_20261002b", pipeline="tools/rebuttal_p16.py over tools/rebuttal_p15.py over tools/rebuttal_p1p2.py")
        self.ATS.write_json(self.dir / "runtime.json", self.meta)

    def symmetry_gate(self, z1, xcan, puz):
        """Global digit relabel (swap fields 1 and 2 everywhere in slow and fast state, swap digits 1 and 2 in the givens) permutes the
        next scores: largest deviation recorded (exact in real arithmetic)."""
        b = 0; h, l = z1[b, 0].copy(), z1[b, 1].copy(); x = np.asarray(xcan[b]).copy()
        hs, ls = h.copy(), l.copy(); hs[[0, 1]] = h[[1, 0]]; ls[[0, 1]] = l[[1, 0]]
        xs = x.copy(); xs[x == 1] = 2; xs[x == 2] = 1
        sc = self.roll_release(np.stack([np.stack([h, l]), np.stack([hs, ls])]), np.stack([x, xs]))[:, 0]   # (2, 81, 9) at the next iteration
        perm = sc[1][:, [1, 0] + list(range(2, 9))]
        return float(np.abs(perm - sc[0]).max()), float(np.abs(sc[0]).max())

    def run(self, smoke=False):
        try:
            ATS, STUDY = P.study_paths()
            with np.load(STUDY / "inputs/interventions.npz", allow_pickle=False) as d: ids_all, puz_all, sol_all = d["ids"], d["puz"], d["sol"]
            disc = Q.discovery_ids(); dirr = self.dir / ("edits_smoke" if smoke else "edits")
            for b0 in range(0, len(ids_all), 128):
                sl = slice(b0, b0 + 128); ids, puz, sol = ids_all[sl], puz_all[sl], sol_all[sl]
                paths = {t: dirr / f"b{b0:04d}_t{t}.npz" for t in Q.TS}
                if not smoke and all(self.cached_any(paths[t]) for t in Q.TS): continue
                x, st = self.release_states(puz)
                with np.load(STUDY / f"attention_{self.width}/interventions/intact/batch_{b0:04d}.npz", allow_pickle=False) as d: ref, refpred = d["logits"], d["pred"]
                g1 = all(np.array_equal(st[t]["logits"], ref[t]) for t in range(max(Q.TS))); assert g1, "gate 1: release states differ from the study's logits"
                sym, scale = self.symmetry_gate(st[0]["z"], np.asarray(x), puz)
                assert sym <= 1e-3 * scale, f"gate 6: digit relabel symmetry deviates by {sym} (scale {scale})"
                n = 16 if smoke else len(ids)
                for t in (Q.TS[:1] if smoke else Q.TS):
                    if not smoke and self.cached_any(paths[t]): continue
                    started = time.monotonic()
                    rec = self.edit_chunk(ids[:n], puz[:n], sol[:n], st[t - 1]["z"][:n], np.asarray(x)[:n], refpred[t:t + K, :n], t, disc, progress=dict(batch=b0, t=t))
                    gates = dict(release_states_bitwise=g1, symmetry_max_dev=sym, symmetry_scale=scale)
                    rec["meta"] = json.dumps(dict(self.meta, t=t, batch_start=b0, n=n, seconds=time.monotonic() - started, created=utc(), gates=gates, smoke=smoke))
                    if smoke:
                        print(json.dumps(dict(gates=gates, **json.loads(rec["gate_summary"])))); return
                    self.ATS.write_npz(paths[t], **rec); self.ATS.write_json(paths[t].with_suffix(".sha256.json"), dict(sha256=self.ATS.sha(paths[t])))
                    self.ATS.log("chunk_complete", width=self.width, batch=b0, t=t, states=int(len(rec["state_ids"])), variants=int(len(rec["var_kind"])), seconds=round(time.monotonic() - started, 1))
            self.ATS.verify_sources(); self.status("inference_complete", completed=utc()); self.ATS.log("model_complete", width=self.width, group="p16")
        except BaseException as exc:
            self.status("failed", error=repr(exc)); raise

    def edit_chunk(self, ids, puz, sol, z, xcan, refpred, t, disc, progress=None):
        lm = self.lm; specs, V = [], dict(state=[], kind=[], cls=[], cell=[], frm=[], to=[], flip=[], j=[], k=[])
        S = dict(ids=[], t=[], split=[], grid=[], sol=[], puz=[], commit=[], margin=[]); per = {}; gate_fail = dict(display=0, kernel_scores=0, swap_display=0, dial_scores=0)
        for b in range(len(ids)):
            pid = int(ids[b]); pz, so = puz[b].reshape(81), sol[b].reshape(81); h, l = z[b, 0], z[b, 1]
            s = np.einsum("fsw,w->sf", h.astype(np.float64), lm); grid = s.argmax(1) + 1
            if t > 1 and not bool(((grid != so) & (pz == 0)).any()): continue
            fl = Q.select(Q.eligible(grid, so, pz), pid, t)
            if not fl: continue
            si = len(S["ids"]); srt = np.sort(s, axis=1)
            per[si] = dict(b=b, h=h, l=l, grid=grid, s=s)
            for key, val in (("ids", pid), ("t", t), ("split", 0 if pid in disc else 1), ("grid", grid.astype(np.int8)), ("sol", so.astype(np.int8)), ("puz", pz.astype(np.int8)),
                             ("commit", commitment(h, grid, lm).astype(np.float32)), ("margin", (srt[:, -1] - srt[:, -2]).astype(np.float32))):
                S[key].append(val)
            def add(kind, cls=0, cell=-1, frm=0, to=0, fi=-1, j=-1, k=-1):
                specs.append((si, kind, cell, frm, to, j, k)); V["state"].append(si); V["kind"].append(KIND[kind]); V["cls"].append(cls); V["cell"].append(cell)
                V["frm"].append(frm); V["to"].append(to); V["flip"].append(fi); V["j"].append(j); V["k"].append(k)
            add("noedit")
            for fi, (c, i, bb) in enumerate(fl):
                a = int(grid[i])
                for ch in CHANNELS: add(f"edit_{ch}", Q.CODE[c], i, a, bb, fi)
                if c == "HA":
                    prt = Q.partners(grid, i, bb); rng = np.random.default_rng([SEED, pid, t, i, 7])
                    j = int(prt[rng.integers(len(prt))])
                    excl = Q.PEERS[i] | Q.PEERS[j]; cand = [c_ for c_ in np.where(pz == 0)[0] if not excl[c_] and c_ not in (i, j)]
                    k = int(cand[rng.integers(len(cand))]) if cand else -1
                    for beta in BETAS:
                        add(f"dialj_{beta}", Q.CODE[c], i, a, bb, fi, j, k); add(f"edit_dialj_{beta}", Q.CODE[c], i, a, bb, fi, j, k)
                    if k >= 0:
                        add("dialk_1.0", Q.CODE[c], i, a, bb, fi, j, k); add("edit_dialk_1.0", Q.CODE[c], i, a, bb, fi, j, k)

        def build(spec):
            si, kind, i, a, bb, j, k = spec; q = per[si]; h, l = q["h"], q["l"]
            if kind == "noedit": return h, l
            want = q["grid"].copy()
            if kind == "edit_display_swap":
                h2 = display_swap(h, i, a, bb, lm); want[i] = bb
                if not np.array_equal(np.einsum("fsw,w->sf", h2, self.lm32).argmax(1) + 1, want): gate_fail["display"] += 1
                return h2, l
            if kind == "edit_kernel_swap":
                h2 = kernel_swap(h, i, a, bb, lm)
                if np.abs(np.einsum("fsw,w->sf", h2.astype(np.float64), lm) - q["s"]).max() > 1e-3: gate_fail["kernel_scores"] += 1
                return h2, l
            if kind == "edit_slow_swap":
                h2 = slow_swap(h, i, a, bb); want[i] = bb
                if not np.array_equal(np.einsum("fsw,w->sf", h2, self.lm32).argmax(1) + 1, want): gate_fail["swap_display"] += 1
                return h2, l
            h2, l2 = (cell_relabel(h, l, i, a, bb) if kind.startswith("edit_") else (h, l))      # every remaining edit_* kind uses the full cell relabel
            if "dialj" in kind or "dialk" in kind:
                beta = float(kind.split("_")[-1]); tgt = j if "dialj" in kind else k
                h3 = dial(h2, tgt, beta, lm)
                if np.abs(np.einsum("fsw,w->sf", h3.astype(np.float64), lm) - np.einsum("fsw,w->sf", h2.astype(np.float64), lm)).max() > 1e-3: gate_fail["dial_scores"] += 1
                return h3, l2
            return h2, l2

        out = np.zeros((len(specs), K, 81, 9), np.float32)
        for c0 in range(0, len(specs), Q.VBATCH):
            chunk = specs[c0:c0 + Q.VBATCH]; pad = Q.VBATCH - len(chunk); built = [build(sp) for sp in chunk]
            zz = np.stack([np.stack([u[0], u[1]]) for u in built] + [np.stack([built[0][0], built[0][1]])] * pad)
            xb = np.stack([xcan[per[sp[0]]["b"]] for sp in chunk] + [xcan[per[chunk[0][0]]["b"]]] * pad)
            out[c0:c0 + len(chunk)] = self.roll_release(zz, xb)[:len(chunk)]
            if progress: self.status("running", **progress, variants_done=c0 + len(chunk), variants=len(specs))
        assert not any(gate_fail.values()), f"gate 3-5: {gate_fail}"
        pred = (out.argmax(-1) + 1).astype(np.int8); st = np.asarray(V["state"]); kd = np.asarray(V["kind"]); g2_fail = 0
        for v in np.where(kd == KIND["noedit"])[0]:
            if not np.array_equal(pred[v], refpred[:, per[int(st[v])]["b"]].reshape(K, 81).astype(np.int8)): g2_fail += 1
        assert g2_fail == 0, f"gate 2: {g2_fail} no-edit rollouts differ from the study's displayed grids"
        summary = dict(states=len(S["ids"]), variants=len(specs), **{f"gate_{k_}": v_ for k_, v_ in gate_fail.items()}, noedit_gate_failures=g2_fail)
        return dict(state_ids=np.asarray(S["ids"], np.int64), state_t=np.asarray(S["t"], np.int8), state_split=np.asarray(S["split"], np.int8),
                    state_grid=np.stack(S["grid"]), state_sol=np.stack(S["sol"]), state_puz=np.stack(S["puz"]), state_commit=np.stack(S["commit"]), state_margin=np.stack(S["margin"]),
                    var_state=st.astype(np.int32), var_kind=kd.astype(np.int8), var_cls=np.asarray(V["cls"], np.int8), var_cell=np.asarray(V["cell"], np.int16),
                    var_from=np.asarray(V["frm"], np.int8), var_to=np.asarray(V["to"], np.int8), var_flip=np.asarray(V["flip"], np.int8),
                    var_j=np.asarray(V["j"], np.int16), var_k=np.asarray(V["k"], np.int16), pred=pred, scores=out.astype(np.float16), gate_summary=json.dumps(summary))


# ---------------- the read ----------------
A_HIGH, A_LOW = 0.50, 0.10
B_GAIN, B_MONO = 0.15, 0.05
C_AUC, C_GAIN, C_WEAK, C_SAME = 0.75, 0.05, 0.65, 0.02


def load_rows(w, base=OUT):
    parts = sorted((base / f"attention_{w}/edits").glob("b*_t*.npz"))
    if not parts: return None
    rows = []
    for pth in parts:
        import hashlib
        side = pth.with_suffix("").with_suffix(".sha256.json")
        assert hashlib.sha256(pth.read_bytes()).hexdigest() == json.loads(side.read_text())["sha256"], f"digest {pth}"
        with np.load(pth, allow_pickle=False) as d: rows.append({k_: d[k_] for k_ in d.files if k_ not in ("scores", "meta", "gate_summary")})
    return rows


def where_letter(lett):
    """lett: channel -> R1-style letter. KERNEL-HOLDS / SLOW-HOLDS / FAST-NEEDED / NOT-LOCAL / MIXED."""
    holds = lambda ch: lett[ch] in ("FOLLOWER", "SELECTIVE")
    if lett["display_swap"] == "DISPLAY-INERT" and holds("kernel_swap"): return "KERNEL-HOLDS"
    if not holds("display_swap") and not holds("kernel_swap") and holds("slow_swap"): return "SLOW-HOLDS"
    if not holds("slow_swap") and holds("cell_relabel"): return "FAST-NEEDED"
    if not any(holds(c) for c in CHANNELS): return "NOT-LOCAL"
    return "MIXED"


def letter_b(r0, rp, rm, rk):
    if not all(np.isfinite(v) for v in (r0, rp, rm, rk)): return "UNDEFINED"
    gain = rp - r0
    if gain >= B_GAIN and (r0 - rm) >= B_MONO and abs(rk - r0) <= gain / 3: return "CONTROLS-PROPAGATION"
    if abs(gain) <= B_MONO: return "NO-CONTROL"
    return "MIXED"


def letter_c(auc_d, auc_dh):
    if not (np.isfinite(auc_d) and np.isfinite(auc_dh)): return "UNDEFINED"
    if auc_dh >= C_AUC and auc_dh - auc_d >= C_GAIN: return "COMPACT-HIDDEN"
    if auc_d >= C_AUC and auc_dh - auc_d < C_SAME: return "DISPLAY-SUFFICES"
    if auc_dh < C_WEAK: return "WEAK"
    return "MIXED"


def auc(y, p):
    y = np.asarray(y, bool); p = np.asarray(p, float); n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0: return float("nan")
    order = np.argsort(p, kind="mergesort"); ranks = np.empty(len(p)); ranks[order] = np.arange(1, len(p) + 1)
    for v in np.unique(p):                                                  # average ranks for ties
        m = p == v
        if m.sum() > 1: ranks[m] = ranks[m].mean()
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def logistic_fit(X, y, l2=1e-2, iters=50):
    X = np.column_stack([np.ones(len(X)), X]); w = np.zeros(X.shape[1]); y = np.asarray(y, float)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w)); g = X.T @ (p - y) + l2 * np.r_[0, w[1:]]
        H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.diag(np.r_[0, np.ones(len(w) - 1)])
        w -= np.linalg.solve(H + 1e-9 * np.eye(len(w)), g)
    return w


def logistic_predict(w, X): return 1 / (1 + np.exp(-(np.column_stack([np.ones(len(X)), X]) @ w)))


def tables(rows, k):
    """Part A records (per flip and channel), Part B records (per HA flip), Part C pair records (per relabel edit and other empty cell)."""
    A, B, C = [], [], []
    R_, Cc, Bx = np.divmod(np.arange(81), 9)[0], np.divmod(np.arange(81), 9)[1], None
    box = (np.arange(81) // 27) * 3 + (np.arange(81) % 9) // 3
    for r in rows:
        kd, st, fi = r["var_kind"], r["var_state"], r["var_flip"]; Pk = r["pred"][:, k - 1].astype(int)
        for s in range(len(r["state_ids"])):
            m = st == s; sol = r["state_sol"][s].astype(int); grid = r["state_grid"][s].astype(int); puz = r["state_puz"][s].astype(int)
            split = int(r["state_split"][s]); t = int(r["state_t"][s]); cm, mg = r["state_commit"][s], r["state_margin"][s]
            v0 = np.where(m & (kd == KIND["noedit"]))[0][0]
            for f in sorted(set(fi[m & (fi >= 0)].tolist())):
                vv = {KINDS[int(kd[v])]: v for v in np.where(m & (fi == f))[0]}
                ve = vv["edit_display_swap"]; i, a, b = int(r["var_cell"][ve]), int(r["var_from"][ve]), int(r["var_to"][ve]); cls = Q.CLASSES[int(r["var_cls"][ve]) - 1]
                for ch in CHANNELS:
                    v = vv[f"edit_{ch}"]
                    A.append(dict(split=split, t=t, cls=cls, ch=ch, c_edit=int(Pk[v, i] == sol[i]), c_no=int(Pk[v0, i] == sol[i]), keep=int(Pk[v, i] == b)))
                vr = vv["edit_cell_relabel"]
                for jj in np.where(puz == 0)[0]:
                    if jj == i: continue
                    rel = (int(R_[jj] == R_[i]), int(Cc[jj] == Cc[i]), int(box[jj] == box[i]))
                    C.append(dict(split=split, y=int(Pk[vr, jj] != Pk[v0, jj]), peer=int(any(rel)), row=rel[0], col=rel[1], box=rel[2],
                                  shows_b=int(grid[jj] == b), shows_a=int(grid[jj] == a), margin_j=float(mg[jj]), margin_i=float(mg[i]), commit_j=float(cm[jj]), commit_i=float(cm[i])))
                if cls == "HA" and "dialj_1.0" in vv:
                    j = int(r["var_j"][vv["dialj_1.0"]]); left = lambda v: int(Pk[v, j] != b)
                    rec = dict(split=split, t=t, i_keeps=int(Pk[vr, i] == b), left_no=left(v0), left_edit=left(vr))
                    for beta in BETAS: rec[f"left_dial_{beta}"] = left(vv[f"dialj_{beta}"]); rec[f"left_editdial_{beta}"] = left(vv[f"edit_dialj_{beta}"])
                    if "dialk_1.0" in vv: rec["left_dialk"] = left(vv["dialk_1.0"]); rec["left_editdialk"] = left(vv["edit_dialk_1.0"])
                    B.append(rec)
    return A, B, C


def summarize_a(A):
    out = {}
    for ch in CHANNELS:
        H = [x for x in A if x["ch"] == ch and x["cls"] in ("HA", "HN")]; X = [x for x in A if x["ch"] == ch and x["cls"] in ("XB", "XN")]
        ep = Q.ratio_gain([x["c_edit"] for x in H], [x["c_no"] for x in H]) if H else float("nan")
        em = Q.ratio_loss([x["c_edit"] for x in X], [x["c_no"] for x in X]) if X else float("nan")
        out[ch] = dict(e_plus=ep, e_minus=em, letter=Q.letter_r1(ep, em), n_H=len(H), n_X=len(X),
                       keep_H=float(np.mean([x["keep"] for x in H])) if H else None, keep_X=float(np.mean([x["keep"] for x in X])) if X else None)
    out["WHERE"] = where_letter({ch: out[ch]["letter"] for ch in CHANNELS}); return out


def summarize_b(B):
    if not B: return dict(n=0, letter="UNDEFINED")
    mean = lambda key: float(np.mean([x[key] for x in B if key in x])) if any(key in x for x in B) else float("nan")
    r0 = mean("left_edit") - mean("left_no"); rp = mean("left_editdial_1.0") - mean("left_dial_1.0"); rh = mean("left_editdial_0.5") - mean("left_dial_0.5")
    rm = mean("left_editdial_-0.5") - mean("left_dial_-0.5"); rk = mean("left_editdialk") - mean("left_dialk")
    return dict(n=len(B), resp_0=r0, resp_plus05=rh, resp_plus1=rp, resp_minus05=rm, resp_k=rk, i_keeps=mean("i_keeps"),
                drift_no=mean("left_no"), drift_dial1=mean("left_dial_1.0"), letter=letter_b(r0, rp, rm, rk))


def summarize_c(C):
    disc = [x for x in C if x["split"] == 0]; conf = [x for x in C if x["split"] == 1]
    if not disc or not conf: return dict(letter="UNDEFINED")
    sets = dict(static=["peer", "row", "col", "box"], display=["peer", "row", "col", "box", "shows_b", "shows_a", "margin_j", "margin_i"])
    sets["display_hidden"] = sets["display"] + ["commit_j", "commit_i"]
    res = {}
    for name, cols in sets.items():
        Xd = np.array([[x[c] for c in cols] for x in disc], float); Xc = np.array([[x[c] for c in cols] for x in conf], float)
        mu, sd = Xd.mean(0), Xd.std(0) + 1e-9; w_ = logistic_fit((Xd - mu) / sd, [x["y"] for x in disc])
        res[name] = auc([x["y"] for x in conf], logistic_predict(w_, (Xc - mu) / sd))
    res["base_rate_conf"] = float(np.mean([x["y"] for x in conf])); res["n_disc"], res["n_conf"] = len(disc), len(conf)
    res["letter"] = letter_c(res["display"], res["display_hidden"]); return res


def report(base=OUT, widths=Q.WIDTHS):
    lines = ["P16 REPORT (rules registered 2026-10-02, Amendment 2): belief edits in the network's own coordinates; letters on CONFIRMATION, k = 1"]; out = {}
    for w in widths:
        rows = load_rows(w, base)
        if rows is None: lines.append(f"attention_{w}: not run"); continue
        res = {}
        for k in range(1, K + 1):
            A, B, C = tables(rows, k); res[k] = {}
            for split, name in ((1, "confirmation"), (0, "discovery")):
                sa = summarize_a([x for x in A if x["split"] == split]); sb = summarize_b([x for x in B if x["split"] == split])
                res[k][name] = dict(A=sa, B=sb)
                lines.append(f"attention_{w} k={k} {name}: " + " | ".join(f"{ch} e+ {sa[ch]['e_plus']:.3f} e- {sa[ch]['e_minus']:.3f} ({sa[ch]['letter']})" for ch in CHANNELS)
                             + f" || WHERE {sa['WHERE']} || B: resp0 {sb.get('resp_0', float('nan')):.3f} resp+0.5 {sb.get('resp_plus05', float('nan')):.3f} resp+1 {sb.get('resp_plus1', float('nan')):.3f} resp-0.5 {sb.get('resp_minus05', float('nan')):.3f} resp_k {sb.get('resp_k', float('nan')):.3f} (n {sb['n']}) -> {sb['letter']}")
            sc = summarize_c(C); res[k]["C"] = sc
            lines.append(f"attention_{w} k={k} C (fit on discovery, AUC on confirmation): static {sc.get('static', float('nan')):.3f} display {sc.get('display', float('nan')):.3f} display+hidden {sc.get('display_hidden', float('nan')):.3f} -> {sc['letter']}")
        out[f"attention_{w}"] = res
    (base / "report.txt").write_text("\n".join(lines) + "\n"); (base / "report.json").write_text(json.dumps(out, indent=1, default=float)); print("\n".join(lines))


def reader_selftest():
    """Synthetic rows through the reader: shapes, finiteness and the letter logic (no model output is read)."""
    rng = np.random.default_rng(1); rows = []
    for split in (0, 1):
        nS = 6; var = dict(state=[], kind=[], cls=[], cell=[], frm=[], to=[], flip=[], j=[], k=[])
        for s in range(nS):
            def add(kind, cls=0, cell=-1, frm=0, to=0, fi=-1, j=-1, k=-1):
                for key, val in (("state", s), ("kind", KIND[kind]), ("cls", cls), ("cell", cell), ("frm", frm), ("to", to), ("flip", fi), ("j", j), ("k", k)): var[key].append(val)
            add("noedit")
            for fi, cls in enumerate((1, 3)):
                for ch in CHANNELS: add(f"edit_{ch}", cls, 10, 1, 2, fi)
                if cls == 1:
                    for beta in BETAS: add(f"dialj_{beta}", cls, 10, 1, 2, fi, 11, 60); add(f"edit_dialj_{beta}", cls, 10, 1, 2, fi, 11, 60)
                    add("dialk_1.0", cls, 10, 1, 2, fi, 11, 60); add("edit_dialk_1.0", cls, 10, 1, 2, fi, 11, 60)
        nV = len(var["state"]); pz = np.zeros((nS, 81), np.int8); pz[:, :20] = 5
        rows.append(dict(state_ids=np.arange(nS) + 100 * split, state_t=np.ones(nS, np.int8), state_split=np.full(nS, split, np.int8),
                         state_grid=rng.integers(1, 10, (nS, 81)).astype(np.int8), state_sol=rng.integers(1, 10, (nS, 81)).astype(np.int8), state_puz=pz,
                         state_commit=rng.random((nS, 81)).astype(np.float32), state_margin=rng.random((nS, 81)).astype(np.float32),
                         var_state=np.asarray(var["state"]), var_kind=np.asarray(var["kind"], np.int8), var_cls=np.asarray(var["cls"], np.int8), var_cell=np.asarray(var["cell"], np.int16),
                         var_from=np.asarray(var["frm"], np.int8), var_to=np.asarray(var["to"], np.int8), var_flip=np.asarray(var["flip"], np.int8),
                         var_j=np.asarray(var["j"], np.int16), var_k=np.asarray(var["k"], np.int16), pred=rng.integers(1, 10, (nV, K, 81)).astype(np.int8)))
    A, B, C = tables(rows, 1); sa = summarize_a(A); sb = summarize_b(B); sc = summarize_c(C)
    assert set(sa) == set(CHANNELS) | {"WHERE"} and sb["n"] == 12 and "display_hidden" in sc
    assert where_letter(dict(display_swap="DISPLAY-INERT", kernel_swap="SELECTIVE", slow_swap="SELECTIVE", cell_relabel="SELECTIVE")) == "KERNEL-HOLDS"
    assert where_letter(dict(display_swap="DISPLAY-INERT", kernel_swap="MIXED", slow_swap="FOLLOWER", cell_relabel="FOLLOWER")) == "SLOW-HOLDS"
    assert where_letter(dict(display_swap="DISPLAY-INERT", kernel_swap="MIXED", slow_swap="MIXED", cell_relabel="SELECTIVE")) == "FAST-NEEDED"
    assert where_letter(dict(display_swap="DISPLAY-INERT", kernel_swap="DISPLAY-INERT", slow_swap="MIXED", cell_relabel="MIXED")) == "NOT-LOCAL"
    assert letter_b(0.0, 0.2, -0.1, 0.02) == "CONTROLS-PROPAGATION" and letter_b(0.0, 0.03, 0.0, 0.0) == "NO-CONTROL" and letter_b(0.0, 0.2, -0.1, 0.15) == "MIXED"
    assert letter_c(0.70, 0.78) == "COMPACT-HIDDEN" and letter_c(0.80, 0.81) == "DISPLAY-SUFFICES" and letter_c(0.55, 0.60) == "WEAK"
    assert abs(auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) - 0.75) < 1e-9
    print("reader selftest OK")

# ---------------- selftest ----------------
def selftest():
    rng = np.random.default_rng(0); w = 16; lm = rng.standard_normal(w); h = rng.standard_normal((9, 81, w)).astype(np.float32); l = rng.standard_normal((9, 81, w)).astype(np.float32)
    sc = lambda hh: np.einsum("fsw,w->sf", hh.astype(np.float64), lm)
    s0 = sc(h); a, b, i = 3, 7, 10
    hd = display_swap(h, i, a, b, lm); s1 = sc(hd)
    assert np.allclose(s1[i, a - 1], s0[i, b - 1], atol=1e-4) and np.allclose(s1[i, b - 1], s0[i, a - 1], atol=1e-4)
    _, q0 = split_par(h[[a - 1, b - 1], i], lm); _, q1 = split_par(hd[[a - 1, b - 1], i], lm); assert np.allclose(q0, q1, atol=1e-5)   # kernels untouched
    hk = kernel_swap(h, i, a, b, lm); assert np.allclose(sc(hk), s0, atol=1e-4)                                                    # scores untouched
    _, qk = split_par(hk[[a - 1, b - 1], i], lm); assert np.allclose(qk[0], q0[1], atol=1e-5) and np.allclose(qk[1], q0[0], atol=1e-5)
    hs = slow_swap(h, i, a, b); assert np.array_equal(hs[a - 1, i], h[b - 1, i]) and np.array_equal(hs[b - 1, i], h[a - 1, i])
    mask = np.ones((9, 81), bool); mask[[a - 1, b - 1], i] = False; assert np.array_equal(hs[mask], h[mask])
    hr, lr = cell_relabel(h, l, i, a, b); assert np.array_equal(lr[a - 1, i], l[b - 1, i])
    for beta in (-0.5, 0.5, 1.0):
        hh = dial(h, 20, beta, lm); assert np.allclose(sc(hh), s0, atol=1e-4)
        _, q = split_par(hh[:, 20], lm)
        if beta == 1.0: assert np.allclose(q, q.mean(0, keepdims=True), atol=1e-5)
    assert np.array_equal(dial(h, 20, 0.5, lm)[:, 19], h[:, 19])
    grid = s0.argmax(1) + 1; cm = commitment(h, grid, lm); assert cm.shape == (81,) and np.all(cm >= 0)
    assert np.allclose(commitment(dial(h, 5, 1.0, lm), grid, lm)[5], 0, atol=1e-5)
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", nargs="?", choices=["run", "report"]); ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--width", type=int, choices=Q.WIDTHS); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.selftest: selftest(); reader_selftest(); return
    if a.action == "report": report(Path(a.out) if a.out else OUT); return
    if a.action == "run":
        if a.width is None: ap.error("--width required")
        R8(a.width, Path(a.out) if a.out else OUT).run(a.smoke)
    else: ap.print_help()


if __name__ == "__main__":
    main()
