#!/usr/bin/env python3
# Ledger: THE DISCUSSION-PERIOD EXPERIMENT P25 — a state-dependent collective repair law (registration
# Documentation/Note_2026-10-10_Rebuttal_P25_Registration.md, written before any row). MEASUREMENT, $0, inference only on the banked
# attention checkpoints. Imports the P15 runner (release states, rollouts), P16's relabel and commitment, and Stage 1's statistics unchanged.
"""P25. Part 1: two laws frozen on P19 (conditional L_c, present-state L_p) predict repair at four fixed, never-edited wrong cells per state
on fresh puzzles, across correction doses, across states, and for joint corrections A+B against an additive model on the margin scale.
Part 2: hybrids that keep a recipient's readout-parallel slow part (every score) and take a same-puzzle donor history's readout-invisible
slow part, with self-transfer, restoration and an angle- and norm-matched random rotation, test whether retained state changes the dose
response beyond baseline (B baseline-only vs D dose-response models) and carries the donor's response (TRANSFER-CARRIES).
Actions: freeze (fit and write the frozen laws), run --width, report, --selftest."""
import argparse, hashlib, json, math, os, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import rebuttal_p1p2 as P                                   # project(), UNITS
import rebuttal_p15 as Q                                    # R7: release_states, roll_release, step_rel, void
import rebuttal_p16 as R16                                  # cell_relabel, commitment
import rebuttal_p17 as R17
import rebuttal_p20 as R20
import rebuttal_p22 as R22
import analyze_seqA as SA                                   # logit_fit / logit_pred / auc / cv_predict / rowloss / boot_diff / p16_state_features

OUT = ROOT / "runs/analysis/rebuttal_20261011"
FROZEN = ROOT / "tools/p25_frozen_laws.json"
SEED = 20261011
WIDTHS = (256, 192, 128)
N_PER_BIN, N_BINS, MIN_WRONG, E_SIZE = 16, 8, 10, 4
DOSES, SETS = (0.1, 0.3, 0.5, 0.7, 0.9), 3
PANEL_FR, PANELS = (0.15, 0.25), 2
P2_DOSES, P2_SETS = (0.1, 0.5, 0.9), 2
K, VB = Q.K, Q.VBATCH
CONDS = ("intact", "transfer", "self", "restore", "random")
ARM = dict(noedit=0, dose=1, panel_A=2, panel_B=3, panel_AB=4)
LAWS = {"L_c": ["y0", "f", "f2", "y0f", "nwrong", "t"], "L_p": ["f", "f2", "margin", "commit", "conflicts", "wrong_peers", "nwrong", "t"],
        "B_c": ["y0", "nwrong", "t"], "B_p": ["nwrong", "t"]}          # B_* = each law's frozen baseline (M0)
N_BOOT, CAL_TOL, CAL_MIN, RHO_MIN, ADD_DELTA = 2000, 0.05, 4, 0.30, 0.10


def utc(): return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(2 ** 20), b""): h.update(b)
    return h.hexdigest()


# ---------------- features, laws (pure; selftested) ----------------
def cell_features(h, grid, sol, puz, lm):
    """Present-state per-cell features, P16's definitions: margin = top minus second readout score; commitment = R16.commitment;
    conflicts = peers displaying the same digit; wrong peers = peers whose display is wrong."""
    s = np.einsum("fsw,w->sf", h.astype(np.float64), lm); srt = np.sort(s, 1)
    wrong = (puz == 0) & (grid != sol)
    conflicts = np.array([int((grid[P.UNITS[c]] == grid[c]).sum()) for c in range(81)])
    wrong_peers = np.array([int(wrong[P.UNITS[c]].sum()) for c in range(81)])
    return srt[:, -1] - srt[:, -2], R16.commitment(h, grid, lm), conflicts, wrong_peers


def law_matrix(feats, d):
    """d: dict of 1-D arrays for y0, f, margin, commit, conflicts, wrong_peers, nwrong, t -> the design matrix for `feats`."""
    col = dict(y0=d.get("y0"), f=d["f"], f2=d["f"] ** 2, y0f=(d["y0"] * d["f"]) if "y0" in d else None, margin=d.get("margin"), commit=d.get("commit"),
               conflicts=d.get("conflicts"), wrong_peers=d.get("wrong_peers"), nwrong=d["nwrong"], t=d["t"])
    return np.stack([np.asarray(col[c], np.float64) for c in feats], 1)


def law_predict(law, d):
    X = law_matrix(law["features"], d); Xs = (X - np.array(law["mean"])) / np.array(law["sd"]); w = np.array(law["coef"])
    return 1 / (1 + np.exp(-np.clip(w[0] + Xs @ w[1:], -30, 30)))


def margin_to_solution(scores, sol_digit):
    """scores (..., 9), sol_digit (...) in 1..9 -> score of the solution digit minus the best other digit."""
    s = np.asarray(scores, np.float64); idx = np.asarray(sol_digit) - 1
    own = np.take_along_axis(s, idx[..., None], -1)[..., 0]; other = s.copy(); np.put_along_axis(other, idx[..., None], -np.inf, -1)
    return own - other.max(-1)


def rotate_matched(perp_r, perp_d, lm, rng):
    """Per vector: a random readout-orthogonal direction at the same angle from perp_r as perp_d is, scaled to |perp_d| (angle- and norm-matched)."""
    vhat = lm / np.linalg.norm(lm); nr = np.linalg.norm(perp_r, axis=-1, keepdims=True); nd = np.linalg.norm(perp_d, axis=-1, keepdims=True)
    ur = perp_r / np.maximum(nr, 1e-30); cos = np.clip((perp_r * perp_d).sum(-1, keepdims=True) / np.maximum(nr * nd, 1e-30), -1, 1); th = np.arccos(cos)
    g = rng.standard_normal(perp_r.shape); g -= (g @ vhat)[..., None] * vhat; g -= (g * ur).sum(-1, keepdims=True) * ur
    g /= np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-30)
    return nd * (np.cos(th) * ur + np.sin(th) * g), th[..., 0]


def interaction_letter(mean, lo, hi, delta):
    if lo > 0: return "SUPER-ADDITIVE"
    if hi < 0: return "SUB-ADDITIVE"
    if lo >= -delta and hi <= delta: return "ADDITIVE"
    return "UNRESOLVED"


def slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float); xc = x - x.mean(); return float((xc * (y - y.mean())).sum() / max((xc * xc).sum(), 1e-30))


# ---------------- populations ----------------
def study_ids():
    ATS, STUDY = P.study_paths()
    with np.load(STUDY / "inputs/repair.npz", allow_pickle=False) as d: return set(int(i) for i in d["ids"])


def pool(width):
    """16 fresh test puzzles per full-test rating octile (rank-based bins, as the study's 512 were stratified), excluding the study's 512 and
    the P12 / P17 / P20 / P22 attention pools."""
    import rebuttal_p12 as P12
    rec = np.load(ROOT / f"paper/code/evidence/benchmark/attention_{width}_d16.npz", allow_pickle=False)
    r128 = np.load(ROOT / "paper/code/evidence/benchmark/attention_128_d16.npz", allow_pickle=False)
    excl = study_ids() | set(int(i) for i in P12.pool_ids(r128["cold_exact"], r128["idx"])) | set(int(i) for i in R17.pool(width)) \
        | set(int(i) for i in R20.pool(width)) | set(int(i) for i in R22.pool(width))
    idx, rating = rec["idx"], rec["rating"]; order = np.lexsort((idx, rating)); bins = np.empty(len(idx), int); bins[order] = (np.arange(len(idx)) * N_BINS) // len(idx)
    out = []
    for b in range(N_BINS):
        cand = np.array(sorted(int(i) for i in idx[bins == b] if int(i) not in excl))
        out += list(np.random.default_rng([SEED, 25, width, b]).choice(cand, N_PER_BIN, replace=False))
    return np.sort(np.array(out, np.int64))


# ---------------- freezing the laws on P19 ----------------
def p19_law_rows(width, k=1):
    feat = SA.p16_state_features(width); D = {c: [] for c in ("y0", "f", "margin", "commit", "conflicts", "wrong_peers", "nwrong", "t", "y", "g")}; files = []
    import glob
    for fpath in sorted(glob.glob(str(ROOT / f"runs/analysis/rebuttal_20261003/attention_{width}/sweep/*.npz"))):
        if "smoke" in fpath: continue
        files.append((str(Path(fpath).relative_to(ROOT)), sha(fpath))); d = np.load(fpath, allow_pickle=False); fr = np.round(d["var_frac"].astype(np.float64), 1)
        for s in range(len(d["state_ids"])):
            sid, t = int(d["state_ids"][s]), int(d["state_t"][s]); grid, sol, puz = (d[k_][s].astype(int) for k_ in ("state_grid", "state_sol", "state_puz"))
            W = np.where((puz == 0) & (grid != sol))[0]; vs = np.where(d["var_state"] == s)[0]; pred0 = d["pred"][[v for v in vs if fr[v] == 0.0][0], k - 1].astype(int)
            margin, commit = feat[(sid, t)]; wrong = np.zeros(81, bool); wrong[W] = True
            conflicts = np.array([int((grid[P.UNITS[c]] == grid[c]).sum()) for c in range(81)]); wpeers = np.array([int(wrong[P.UNITS[c]].sum()) for c in range(81)])
            for v in vs:
                if fr[v] == 0.0: continue
                E = set(int(c) for c in d["var_cells"][v] if c >= 0); pv = d["pred"][v, k - 1].astype(int)
                for c in W:
                    c = int(c)
                    if c in E: continue
                    for key, val in (("y0", int(pred0[c] == sol[c])), ("f", len(E) / len(W)), ("margin", margin[c]), ("commit", commit[c]), ("conflicts", conflicts[c]),
                                     ("wrong_peers", wpeers[c]), ("nwrong", len(W)), ("t", t), ("y", int(pv[c] == sol[c])), ("g", sid)): D[key].append(val)
    return {k_: np.array(v, np.float64) for k_, v in D.items()}, files


def freeze():
    out = dict(created=utc(), tool="tools/rebuttal_p25.py", tool_sha256_at_freeze=sha(Path(__file__)), l2=SA.L2,
               definition="logistic regression on standardized features (training mean / sd), L2 1e-3 on slopes, intercept unpenalized; rows = every (P19 variant with f > 0, wrong cell it does not edit) at k = 1; f = edited / wrong cells",
               uncertainty=dict(bootstrap="puzzle", resamples=N_BOOT, seed_base=[SEED, 25]), widths={})
    for w in WIDTHS:
        D, files = p19_law_rows(w); y, g = D["y"], D["g"].astype(int); laws = {}
        for name, feats in LAWS.items():
            X = law_matrix(feats, D); mu, sd = X.mean(0), X.std(0) + 1e-9; coef = SA.logit_fit((X - mu) / sd, y)
            p_cv = SA.cv_predict(X, y, g)
            laws[name] = dict(features=feats, mean=mu.tolist(), sd=sd.tolist(), coef=coef.tolist(), cv_logloss=float(SA.rowloss(p_cv, y).mean()), cv_auc=SA.auc(p_cv, y))
        out["widths"][str(w)] = dict(rows=int(len(y)), puzzles=int(len(np.unique(g))), fit_files=files, laws=laws)
        print(f"attention_{w}: rows {len(y)} | " + " | ".join(f"{n} cv logloss {laws[n]['cv_logloss']:.4f} AUC {laws[n]['cv_auc']:.3f}" for n in LAWS))
    FROZEN.write_text(json.dumps(out, indent=1)); print("frozen:", FROZEN, sha(FROZEN))


# ---------------- the runner ----------------
class R19(Q.R7):
    def __init__(self, width, out=OUT):
        super().__init__(width, out)
        self.meta.update(study="rebuttal_20261011", pipeline="tools/rebuttal_p25.py over tools/rebuttal_p15.py / rebuttal_p16.py")
        self.dir = out / f"attention_{width}"; self.dir.mkdir(parents=True, exist_ok=True); self.ATS.write_json(self.dir / "runtime.json", self.meta)

    def history(self, z0, x, steps):
        """The release loop from an explicit carry (as release_states, with the canvas update): the carried state after `steps`."""
        jax, jnp = self.jax, self.jnp; x = jnp.asarray(x); y = jnp.broadcast_to(self.void, (x.shape[0],) + self.void.shape); z = jnp.asarray(z0)
        for _ in range(steps):
            lg_dev, zf = self.step_rel(self.params, x, y, self.tv, z); z = z + self.eta_z * (zf - z)
            p = jax.nn.softmax(lg_dev, axis=-1).transpose(0, 3, 1, 2); y = y + self.eta * (p - y)
        return np.asarray(z)

    def roll(self, specs, xcan, sol81, build, tag, counter):
        """specs: list of variant dicts; build(spec) -> (h, l). Returns scores (V, K, 81, 9) at float32 via roll_release in padded chunks."""
        out = np.zeros((len(specs), K, 81, 9), np.float32)
        for c0 in range(0, len(specs), VB):
            chunk = specs[c0:c0 + VB]; pad = VB - len(chunk); built = [build(sp) for sp in chunk]
            zz = np.stack([np.stack([b_[0], b_[1]]) for b_ in built] + [np.stack([built[0][0], built[0][1]])] * pad)
            xb = np.stack([xcan[sp["b"]] for sp in chunk] + [xcan[chunk[0]["b"]]] * pad)
            out[c0:c0 + len(chunk)] = self.roll_release(zz, xb)[:len(chunk)]; counter[0] += len(chunk)
            self.status("running", part=tag, variants_done=counter[0])
        return out

    def run(self, smoke=False):
        try:
            lm = self.lm; ids = pool(self.width)
            if smoke: ids = ids[:4]
            with np.load(R17.DATA, allow_pickle=False) as d: puz_all, sol_all = d["test_q"][ids].astype(np.int32), d["test_a"][ids].astype(np.int32)
            xcan_all = []; states = {1: [], 2: []}
            for b0 in range(0, len(ids), VB):
                x, st = self.release_states(puz_all[b0:b0 + VB]); xcan_all.append(np.asarray(x))
                if b0 == 0:                                                           # gate: the release loop reproduces itself bitwise
                    _, st2 = self.release_states(puz_all[b0:b0 + VB]); assert all(np.array_equal(st[t]["z"], st2[t]["z"]) for t in range(2)), "gate: release states not reproducible"
                for t in (1, 2): states[t].append(st[t - 1]["z"])
            xcan = np.concatenate(xcan_all, 0); Z = {t: np.concatenate(states[t], 0) for t in (1, 2)}
            gates = dict(display=0, e_edited=0, score_shift=0.0, self_bitwise=0, restore_dev=0.0, angle_dev=0.0, norm_dev=0.0)
            # ---- Part 1 ----
            S1, V1 = [], []
            for b, pid in enumerate(ids):
                sol, puz = sol_all[b].reshape(81), puz_all[b].reshape(81)
                for t in (1, 2):
                    h, l = Z[t][b, 0], Z[t][b, 1]; grid = np.einsum("fsw,w->sf", h.astype(np.float64), lm).argmax(1) + 1
                    W = np.where((puz == 0) & (grid != sol))[0]
                    if len(W) < MIN_WRONG: continue
                    rng = np.random.default_rng([SEED, int(pid), t, 1]); E = np.sort(rng.choice(W, E_SIZE, replace=False)); Rr = np.array(sorted(set(W) - set(E)))
                    mg, cm, cf, wp = cell_features(h, grid, sol, puz, lm); si = len(S1)
                    S1.append(dict(pid=int(pid), t=t, b=b, grid=grid.astype(np.int8), sol=sol.astype(np.int8), puz=puz.astype(np.int8), nwrong=len(W), E=E,
                                   margin=mg[E], commit=cm[E], conflicts=cf[E], wrong_peers=wp[E]))
                    V1.append(dict(si=si, b=b, arm=ARM["noedit"], dose=-1, set=-1, cells=np.array([], int)))
                    for i, f in enumerate(DOSES):
                        n = max(1, int(round(f * len(Rr))))
                        for j in range(SETS): V1.append(dict(si=si, b=b, arm=ARM["dose"], dose=i, set=j, cells=np.sort(np.random.default_rng([SEED, int(pid), t, 2, i, j]).choice(Rr, n, replace=False))))
                    for i, fr in enumerate(PANEL_FR):
                        n = max(1, int(round(fr * len(Rr))))
                        for j in range(PANELS):
                            rr = np.random.default_rng([SEED, int(pid), t, 3, i, j]); A = rr.choice(Rr, n, replace=False); B = rr.choice(np.array(sorted(set(Rr) - set(A))), n, replace=False)
                            for arm, cells in (("panel_A", A), ("panel_B", B), ("panel_AB", np.concatenate([A, B]))): V1.append(dict(si=si, b=b, arm=ARM[arm], dose=i, set=j, cells=np.sort(cells)))
            def build1(sp):
                s_ = S1[sp["si"]]; h, l = Z[s_["t"]][sp["b"], 0], Z[s_["t"]][sp["b"], 1]; cells = sp["cells"]
                if np.isin(cells, s_["E"]).any(): gates["e_edited"] += 1
                if len(cells) == 0: return h, l
                for c in cells: h, l = R16.cell_relabel(h, l, int(c), int(s_["grid"][c]), int(s_["sol"][c]))
                want = s_["grid"].astype(int).copy(); want[cells] = s_["sol"][cells]
                if not np.array_equal(np.einsum("fsw,w->sf", h, self.lm32).argmax(1) + 1, want): gates["display"] += 1
                return h, l
            cnt = [0]; sc1 = self.roll(V1, xcan, None, build1, "part1", cnt)
            # ---- Part 2 (t = 1 recipients; donor = the same puzzle's Gaussian-start history at iteration 1) ----
            z0 = np.stack([self.EV.mi_z0(SEED, int(pid), 7, Z[1].shape[1:], 1.0, "gauss") for pid in ids]).astype(np.float32); Zd = []
            for b0 in range(0, len(ids), VB):
                zb = z0[b0:b0 + VB]; xb = xcan[b0:b0 + VB]; pad = VB - len(zb)
                zp = np.concatenate([zb, np.repeat(zb[:1], pad, 0)]) if pad else zb; xp = np.concatenate([xb, np.repeat(xb[:1], pad, 0)]) if pad else xb
                Zd.append(self.history(zp, xp, 1)[:len(zb)])
            Zd = np.concatenate(Zd, 0); S2, V2, HIST = [], [], {}
            for b, pid in enumerate(ids):
                sol, puz = sol_all[b].reshape(81), puz_all[b].reshape(81)
                hs = {"A": (Z[1][b, 0], Z[1][b, 1]), "B": (Zd[b, 0], Zd[b, 1])}
                grids = {k_: np.einsum("fsw,w->sf", v[0].astype(np.float64), lm).argmax(1) + 1 for k_, v in hs.items()}
                Ws = {k_: np.where((puz == 0) & (g_ != sol))[0] for k_, g_ in grids.items()}
                if min(len(Ws["A"]), len(Ws["B"])) < MIN_WRONG: continue
                for di, (r, d_) in enumerate((("A", "B"), ("B", "A"))):
                    hr, lr = hs[r]; hd = hs[d_][0]; W = Ws[r]; grid = grids[r]
                    rng = np.random.default_rng([SEED, int(pid), 4, di]); E = np.sort(rng.choice(W, E_SIZE, replace=False)); Rr = np.array(sorted(set(W) - set(E)))
                    par_r, perp_r = P.project(hr.astype(np.float64), lm); perp_d = P.project(hd.astype(np.float64), lm)[1]
                    h_tr = (hr.astype(np.float64) + (perp_d - perp_r)).astype(np.float32)
                    h_self = (hr.astype(np.float64) + (perp_r - perp_r)).astype(np.float32)
                    h_res = (h_tr.astype(np.float64) + (perp_r - P.project(h_tr.astype(np.float64), lm)[1])).astype(np.float32)
                    newp, th = rotate_matched(perp_r, perp_d, lm, np.random.default_rng([SEED, int(pid), 6, di])); h_rand = (par_r + newp).astype(np.float32)
                    s0 = np.einsum("fsw,w->sf", hr.astype(np.float64), lm)
                    for hh in (h_tr, h_res, h_rand): gates["score_shift"] = max(gates["score_shift"], float(np.abs(np.einsum("fsw,w->sf", hh.astype(np.float64), lm) - s0).max()))
                    gates["self_bitwise"] += int(not np.array_equal(h_self, hr)); gates["restore_dev"] = max(gates["restore_dev"], float(np.abs(h_res.astype(np.float64) - hr.astype(np.float64)).max()))
                    pr = P.project(h_rand.astype(np.float64), lm)[1]; ok = (th > math.radians(0.5)) & (np.linalg.norm(perp_r, axis=-1) > 1e-6) & (np.linalg.norm(perp_d, axis=-1) > 1e-6)
                    if ok.any():
                        cr = np.clip((pr * perp_r).sum(-1) / np.maximum(np.linalg.norm(pr, axis=-1) * np.linalg.norm(perp_r, axis=-1), 1e-30), -1, 1)
                        gates["angle_dev"] = max(gates["angle_dev"], float((np.abs(np.arccos(cr) - th) / th)[ok].max()))
                        gates["norm_dev"] = max(gates["norm_dev"], float((np.abs(np.linalg.norm(pr, axis=-1) - np.linalg.norm(perp_d, axis=-1)) / np.linalg.norm(perp_d, axis=-1))[ok].max()))
                    si = len(S2); HIST[si] = dict(intact=hr, transfer=h_tr, self=h_self, restore=h_res, random=h_rand, l=lr, grid=grid, sol=sol)
                    S2.append(dict(pid=int(pid), dir=di, b=b, grid=grid.astype(np.int8), sol=sol.astype(np.int8), nwrong=len(W), E=E))
                    masks = [(i, j, np.sort(np.random.default_rng([SEED, int(pid), 5, di, i, j]).choice(Rr, max(1, int(round(f * len(Rr)))), replace=False)))
                             for i, f in enumerate(P2_DOSES) for j in range(P2_SETS)]
                    for ci_, cond in enumerate(CONDS):
                        V2.append(dict(si=si, b=b, cond=ci_, arm=ARM["noedit"], dose=-1, set=-1, cells=np.array([], int)))
                        for i, j, cells in masks: V2.append(dict(si=si, b=b, cond=ci_, arm=ARM["dose"], dose=i, set=j, cells=cells))
            def build2(sp):
                H = HIST[sp["si"]]; h, l = H[CONDS[sp["cond"]]], H["l"]; cells = sp["cells"]
                if np.isin(cells, S2[sp["si"]]["E"]).any(): gates["e_edited"] += 1
                if len(cells) == 0: return h, l
                for c in cells: h, l = R16.cell_relabel(h, l, int(c), int(H["grid"][c]), int(H["sol"][c]))
                want = H["grid"].astype(int).copy(); want[cells] = H["sol"][cells]
                if not np.array_equal(np.einsum("fsw,w->sf", h, self.lm32).argmax(1) + 1, want): gates["display"] += 1
                return h, l
            sc2 = self.roll(V2, xcan, None, build2, "part2", cnt)
            assert gates["display"] == 0 and gates["e_edited"] == 0, f"gate: {gates}"
            assert gates["score_shift"] < 1e-3 and gates["self_bitwise"] == 0 and gates["restore_dev"] < 1e-4, f"gate: {gates}"
            assert gates["angle_dev"] < 0.01 and gates["norm_dev"] < 0.01, f"gate: {gates}"
            def pack(S, V, sc, part):
                E = np.stack([s_["E"] for s_ in S]); vs = np.array([v["si"] for v in V])
                escores = np.stack([sc[i][:, E[vs[i]]] for i in range(len(V))]).astype(np.float32)          # (V, K, 4, 9)
                disp = (sc.argmax(-1) + 1).astype(np.int8)                                                 # (V, K, 81)
                cells = np.full((len(V), 81), -1, np.int16)
                for i, v in enumerate(V): cells[i, :len(v["cells"])] = v["cells"]
                base = dict(var_state=vs.astype(np.int32), var_arm=np.array([v["arm"] for v in V], np.int8), var_dose=np.array([v["dose"] for v in V], np.int8),
                            var_set=np.array([v["set"] for v in V], np.int8), var_cells=cells, var_nedit=np.array([len(v["cells"]) for v in V], np.int16),
                            escores=escores, disp=disp, state_pid=np.array([s_["pid"] for s_ in S], np.int64), state_grid=np.stack([s_["grid"] for s_ in S]),
                            state_sol=np.stack([s_["sol"] for s_ in S]), state_nwrong=np.array([s_["nwrong"] for s_ in S], np.int16), state_E=E.astype(np.int16))
                if part == 1:
                    base.update(state_t=np.array([s_["t"] for s_ in S], np.int8), state_puz=np.stack([s_["puz"] for s_ in S]),
                                **{f"state_{k_}": np.stack([s_[k_] for s_ in S]).astype(np.float32) for k_ in ("margin", "commit", "conflicts", "wrong_peers")})
                else:
                    base.update(var_cond=np.array([v["cond"] for v in V], np.int8), state_dir=np.array([s_["dir"] for s_ in S], np.int8))
                return {f"p{part}_{k_}": v for k_, v in base.items()}
            dst = self.dir / ("p25_smoke.npz" if smoke else "p25.npz")
            np.savez_compressed(dst, ids=ids, gates=json.dumps(gates), meta=json.dumps(dict(self.meta, created=utc(), n=len(ids), smoke=smoke, part1_states=len(S1), part2_recipients=len(S2),
                                                                                          frozen_laws_sha256=sha(FROZEN))), **pack(S1, V1, sc1, 1), **pack(S2, V2, sc2, 2))
            if not smoke: self.ATS.write_json(dst.with_suffix(".sha256.json"), dict(sha256=self.ATS.sha(dst)))
            print(json.dumps(dict(width=self.width, puzzles=len(ids), part1_states=len(S1), part1_variants=len(V1), part2_recipients=len(S2), part2_variants=len(V2), gates=gates)))
            self.ATS.verify_sources(); self.status("inference_complete", completed=utc()); self.ATS.log("model_complete", width=self.width, group="p25")
        except BaseException as exc:
            self.status("failed", error=repr(exc)); raise


# ---------------- the frozen reader ----------------
def part1_rows(d, laws, k=1):
    """Dose-arm rows (state, arm, E cell) with y, y0 and both laws' feature sets, plus the state index and puzzle id."""
    st = d["p1_var_state"]; arm = d["p1_var_arm"]; E = d["p1_state_E"]; sol = d["p1_state_sol"]; disp = d["p1_disp"]; nw = d["p1_state_nwrong"].astype(float)
    noedit = {int(s): i for i, s in enumerate(st) if arm[i] == ARM["noedit"]}
    R = {c: [] for c in ("y", "y0", "f", "margin", "commit", "conflicts", "wrong_peers", "nwrong", "t", "g", "s", "dose")}
    for i in np.where(arm == ARM["dose"])[0]:
        s = int(st[i]); e = E[s]; y = disp[i, k - 1, e] == sol[s, e]; y0 = disp[noedit[s], k - 1, e] == sol[s, e]
        for q in range(E_SIZE):
            for key, val in (("y", y[q]), ("y0", y0[q]), ("f", d["p1_var_nedit"][i] / nw[s]), ("margin", d["p1_state_margin"][s, q]), ("commit", d["p1_state_commit"][s, q]),
                             ("conflicts", d["p1_state_conflicts"][s, q]), ("wrong_peers", d["p1_state_wrong_peers"][s, q]), ("nwrong", nw[s]), ("t", d["p1_state_t"][s]),
                             ("g", d["p1_state_pid"][s]), ("s", s), ("dose", d["p1_var_dose"][i])): R[key].append(val)
    return {k_: np.array(v, np.float64) for k_, v in R.items()}


def spearman(a, b):
    from scipy.stats import rankdata
    ra, rb = rankdata(a), rankdata(b); return float(np.corrcoef(ra, rb)[0, 1])


def report_width(w, base=OUT):
    f = base / f"attention_{w}/p25.npz"
    if not f.exists(): return None
    d = np.load(f, allow_pickle=False); FZ = json.loads(FROZEN.read_text()); laws = FZ["widths"][str(w)]["laws"]; res = {}
    # P1a / P1b
    R = part1_rows(d, laws); y, g = R["y"], R["g"].astype(int)
    for law, basel in (("L_c", "B_c"), ("L_p", "B_p")):
        p = law_predict(laws[law], R); pb = law_predict(laws[basel], R); dd = SA.rowloss(pb, y) - SA.rowloss(p, y); lo, hi = SA.boot_diff(dd, g, n=N_BOOT, seed=[SEED, 25, w, 1])
        cal = {f"{DOSES[i]:.1f}": float(abs(y[R["dose"] == i].mean() - p[R["dose"] == i].mean())) for i in range(len(DOSES))}
        ncal = sum(v <= CAL_TOL for v in cal.values())
        letter = "CALIBRATED" if (lo > 0 and ncal >= CAL_MIN) else ("MISCALIBRATED" if lo > 0 else "NO-PREDICTION")
        sids = np.unique(R["s"]).astype(int); obs = np.array([y[R["s"] == s].mean() for s in sids]); pre = np.array([p[R["s"] == s].mean() for s in sids])
        sp_g = np.array([int(g[R["s"] == s][0]) for s in sids]); rho = spearman(pre, obs); rng = np.random.default_rng([SEED, 25, w, 2]); ug = np.unique(sp_g); bs = []
        for _ in range(N_BOOT):
            pick = rng.choice(ug, len(ug)); m = np.concatenate([np.where(sp_g == u)[0] for u in pick]); bs.append(spearman(pre[m], obs[m]))
        rlo, rhi = float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))
        res[law] = dict(logloss_gain=float(dd.mean()), ci=(lo, hi), calibration=cal, calibrated_doses=int(ncal), letter=letter, rho=rho, rho_ci=(rlo, rhi),
                        state_letter="STATE-PREDICTED" if (rho >= RHO_MIN and rlo > 0) else "NOT-SHOWN", auc=SA.auc(p, y), base_auc=SA.auc(pb, y))
    # P1c: margin interaction on the panels
    st, arm, E, sol, es = d["p1_var_state"], d["p1_var_arm"], d["p1_state_E"], d["p1_state_sol"], d["p1_escores"]
    noedit = {int(s): i for i, s in enumerate(st) if arm[i] == ARM["noedit"]}; key = {}
    for i in np.where(np.isin(arm, [ARM["panel_A"], ARM["panel_B"], ARM["panel_AB"]]))[0]: key.setdefault((int(st[i]), int(d["p1_var_dose"][i]), int(d["p1_var_set"][i])), {})[int(arm[i])] = i
    I, scale, gI, accL, accA, gA = [], [], [], [], [], []
    for (s, i_, j_), v in key.items():
        if len(v) < 3: continue
        sd = sol[s, E[s]]; m0 = margin_to_solution(es[noedit[s], 0], sd); mA = margin_to_solution(es[v[ARM["panel_A"]], 0], sd)
        mB = margin_to_solution(es[v[ARM["panel_B"]], 0], sd); mAB = margin_to_solution(es[v[ARM["panel_AB"]], 0], sd)
        dA, dB, dAB = mA - m0, mB - m0, mAB - m0; I += list(dAB - dA - dB); scale += list(np.abs(dA) + np.abs(dB)); gI += [int(d["p1_state_pid"][s])] * E_SIZE
        yAB = (mAB > 0).astype(int); y0 = (m0 > 0).astype(float); fAB = d["p1_var_nedit"][v[ARM["panel_AB"]]] / d["p1_state_nwrong"][s]
        pL = law_predict(laws["L_c"], dict(y0=y0, f=np.full(E_SIZE, fAB), nwrong=np.full(E_SIZE, float(d["p1_state_nwrong"][s])), t=np.full(E_SIZE, float(d["p1_state_t"][s]))))
        accL += list(((pL > 0.5).astype(int) == yAB).astype(float)); accA += list((((m0 + dA + dB) > 0).astype(int) == yAB).astype(float)); gA += [int(d["p1_state_pid"][s])] * E_SIZE
    I, gI = np.array(I), np.array(gI); lo, hi = SA.boot_diff(I, gI, n=N_BOOT, seed=[SEED, 25, w, 3]); delta = ADD_DELTA * float(np.mean(scale))
    dacc = np.array(accL) - np.array(accA); alo, ahi = SA.boot_diff(dacc, np.array(gA), n=N_BOOT, seed=[SEED, 25, w, 4])
    res["P1c"] = dict(interaction=float(I.mean()), ci=(lo, hi), delta=delta, letter=interaction_letter(I.mean(), lo, hi, delta), cells=int(len(I)),
                      acc_law=float(np.mean(accL)), acc_additive=float(np.mean(accA)), acc_diff_ci=(alo, ahi))
    # Part 2: model B (baseline only) vs model D (condition changes the dose response)
    st2, cond, arm2, E2, sol2, disp2 = d["p2_var_state"], d["p2_var_cond"], d["p2_var_arm"], d["p2_state_E"], d["p2_state_sol"], d["p2_disp"]
    no2 = {(int(s), int(c)): i for i, (s, c) in enumerate(zip(st2, cond)) if arm2[i] == ARM["noedit"]}; rows = []
    for i in np.where((arm2 == ARM["dose"]) & np.isin(cond, [CONDS.index("intact"), CONDS.index("transfer"), CONDS.index("random")]))[0]:
        s, c = int(st2[i]), int(cond[i]); e = E2[s]; y = disp2[i, 0, e] == sol2[s, e]; y0 = disp2[no2[(s, c)], 0, e] == sol2[s, e]
        for q in range(E_SIZE): rows.append((int(d["p2_state_pid"][s]), s, c, int(d["p2_var_dose"][i]), d["p2_var_nedit"][i] / d["p2_state_nwrong"][s], float(d["p2_state_nwrong"][s]), int(y0[q]), int(y[q])))
    A2 = np.array(rows, np.float64); gp, cc, ff, nwr, yy0, yy = A2[:, 0].astype(int), A2[:, 2].astype(int), A2[:, 4], A2[:, 5], A2[:, 6], A2[:, 7]
    it, ir = (cc == CONDS.index("transfer")).astype(float), (cc == CONDS.index("random")).astype(float)
    XB = np.stack([yy0, ff, ff ** 2, nwr], 1); XD = np.stack([yy0, ff, ff ** 2, nwr, it, it * ff, ir, ir * ff], 1)
    pB, pD = SA.cv_predict(XB, yy, gp), SA.cv_predict(XD, yy, gp); dd = SA.rowloss(pB, yy) - SA.rowloss(pD, yy); lo, hi = SA.boot_diff(dd, gp, n=N_BOOT, seed=[SEED, 25, w, 5])
    res["P2_state"] = dict(rows=int(len(yy)), logloss_gain_D_over_B=float(dd.mean()), ci=(lo, hi), D_better=bool(lo > 0))
    # TRANSFER-CARRIES: response beyond baseline, transfer vs intact against donor-intact vs intact, across (pair, direction, dose)
    resp = {}
    for (pid, s, c, di, f_, nw_, y0_, y_) in rows: resp.setdefault((int(pid), int(s), int(c), int(di)), []).append((y_, y0_))
    sdir = {int(s): int(d["p2_state_dir"][s]) for s in range(len(d["p2_state_dir"]))}; byp = {}
    for (pid, s, c, di), v in resp.items(): byp.setdefault((pid, sdir[s], di), {})[c] = float(np.mean([a for a, _ in v]) - np.mean([b_ for _, b_ in v]))
    X_, Ytr, Yrd, G_ = [], [], [], []
    for (pid, dr, di), v in byp.items():
        donor = byp.get((pid, 1 - dr, di), {}); ii = CONDS.index("intact")
        if ii in v and ii in donor and CONDS.index("transfer") in v and CONDS.index("random") in v:
            X_.append(donor[ii] - v[ii]); Ytr.append(v[CONDS.index("transfer")] - v[ii]); Yrd.append(v[CONDS.index("random")] - v[ii]); G_.append(pid)
    X_, Ytr, Yrd, G_ = map(np.array, (X_, Ytr, Yrd, G_)); rng = np.random.default_rng([SEED, 25, w, 6]); ug = np.unique(G_); b1, b2 = [], []
    for _ in range(N_BOOT):
        m = np.concatenate([np.where(G_ == u)[0] for u in rng.choice(ug, len(ug))]); s1, s2 = slope(X_[m], Ytr[m]), slope(X_[m], Yrd[m]); b1.append(s1); b2.append(s1 - s2)
    st_, sr_ = slope(X_, Ytr), slope(X_, Yrd); c1, c2 = np.percentile(b1, [2.5, 97.5]), np.percentile(b2, [2.5, 97.5])
    res["P2_transfer"] = dict(points=int(len(X_)), slope_transfer=st_, slope_random=sr_, ci_transfer=tuple(map(float, c1)), ci_difference=tuple(map(float, c2)),
                              letter="TRANSFER-CARRIES" if (c1[0] > 0 and c2[0] > 0) else "NOT-SHOWN")
    # descriptive: induced repair (cells the condition's own no-edit continuation does not repair) by dose and condition
    res["P2_dose_response"] = {CONDS[c]: {f"{P2_DOSES[di]:.1f}": float(yy[(cc == c) & (A2[:, 3] == di) & (yy0 == 0)].mean()) if ((cc == c) & (A2[:, 3] == di) & (yy0 == 0)).any() else None
                                          for di in range(len(P2_DOSES))} for c in (0, 1, 4)}
    res["gates"] = json.loads(str(d["gates"])); res["meta"] = {k_: json.loads(str(d["meta"]))[k_] for k_ in ("part1_states", "part2_recipients", "frozen_laws_sha256")}
    return res


def report(base=OUT, widths=WIDTHS):
    lines = ["P25 REPORT (rules registered 2026-10-10 before any row; laws frozen in tools/p25_frozen_laws.json)"]; js = {}
    for w in widths:
        r = report_width(w, base)
        if r is None: lines.append(f"attention_{w}: not run"); continue
        js[f"attention_{w}"] = r
        for law in ("L_c", "L_p"):
            a = r[law]; lines.append(f"attention_{w} P1a/P1b {law}: logloss gain over baseline {a['logloss_gain']:+.4f} [{a['ci'][0]:+.4f}, {a['ci'][1]:+.4f}], AUC {a['auc']:.3f} (baseline {a['base_auc']:.3f}), "
                                     f"calibration by dose " + " ".join(f"{k_}:{v:.3f}" for k_, v in a["calibration"].items()) + f" -> {a['letter']} | per-state rho {a['rho']:.3f} [{a['rho_ci'][0]:.3f}, {a['rho_ci'][1]:.3f}] -> {a['state_letter']}")
        c = r["P1c"]; lines.append(f"attention_{w} P1c: margin interaction {c['interaction']:+.4f} [{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}] (delta {c['delta']:.4f}, cells {c['cells']}) -> {c['letter']} | "
                                   f"A+B accuracy: law {c['acc_law']:.3f} vs additive {c['acc_additive']:.3f}, difference CI [{c['acc_diff_ci'][0]:+.3f}, {c['acc_diff_ci'][1]:+.3f}]")
        s, t_ = r["P2_state"], r["P2_transfer"]
        lines.append(f"attention_{w} P2: D over B logloss gain {s['logloss_gain_D_over_B']:+.4f} [{s['ci'][0]:+.4f}, {s['ci'][1]:+.4f}] (rows {s['rows']}) | transfer slope {t_['slope_transfer']:+.3f} "
                     f"[{t_['ci_transfer'][0]:+.3f}, {t_['ci_transfer'][1]:+.3f}], random {t_['slope_random']:+.3f}, difference CI [{t_['ci_difference'][0]:+.3f}, {t_['ci_difference'][1]:+.3f}] -> {t_['letter']} | "
                     f"induced repair by dose: " + "; ".join(f"{k_} " + " ".join(f"{dd_}:{v:.2f}" if v is not None else f"{dd_}:-" for dd_, v in vv.items()) for k_, vv in r["P2_dose_response"].items()))
        lines.append(f"attention_{w} gates {r['gates']} | states {r['meta']['part1_states']} part-1, {r['meta']['part2_recipients']} part-2 recipients")
    have = [w for w in (256, 192) if f"attention_{w}" in js]
    if len(have) == 2:
        both = all(js[f"attention_{w}"]["P2_state"]["D_better"] for w in have)
        lines.append(f"STATE-SETS-RESPONSE (D beats B on 256 and 192): {'STATE-SETS-RESPONSE' if both else ('BASELINE-ONLY' if not any(js[f'attention_{w}']['P2_state']['D_better'] for w in have) else 'UNRESOLVED')}")
    (base / "report.txt").write_text("\n".join(lines) + "\n"); (base / "report.json").write_text(json.dumps(js, indent=1, default=float)); print("\n".join(lines))


def selftest():
    s = np.array([[1.0, 3.0, 2.0], [5.0, 1.0, 0.0]]); s9 = np.zeros((2, 9)); s9[:, :3] = s; s9[:, 3:] = -10
    assert np.allclose(margin_to_solution(s9, np.array([2, 1])), [1.0, 4.0])
    law = dict(features=["f", "nwrong", "t"], mean=[0.0, 0.0, 0.0], sd=[1.0, 1.0, 1.0], coef=[0.5, 2.0, 0.0, 0.0])
    assert np.allclose(law_predict(law, dict(f=np.array([0.0, 1.0]), nwrong=np.zeros(2), t=np.zeros(2))), 1 / (1 + np.exp(-np.array([0.5, 2.5]))))
    rng = np.random.default_rng(0); lm = rng.standard_normal(16); pr = rng.standard_normal((50, 16)); pdn = rng.standard_normal((50, 16))
    vh = lm / np.linalg.norm(lm); pr -= (pr @ vh)[:, None] * vh; pdn -= (pdn @ vh)[:, None] * vh
    new, th = rotate_matched(pr, pdn, lm, rng); cr = (new * pr).sum(-1) / (np.linalg.norm(new, axis=-1) * np.linalg.norm(pr, axis=-1))
    assert np.allclose(np.arccos(np.clip(cr, -1, 1)), th, atol=1e-8) and np.allclose(np.linalg.norm(new, axis=-1), np.linalg.norm(pdn, axis=-1)) and np.allclose(new @ vh, 0, atol=1e-10)
    h = rng.standard_normal((9, 81, 16)).astype(np.float32); _, pp = P.project(h.astype(np.float64), lm)
    assert np.array_equal((h.astype(np.float64) + (pp - pp)).astype(np.float32), h)                      # self-transfer is bitwise
    assert interaction_letter(0.2, 0.1, 0.3, 0.05) == "SUPER-ADDITIVE" and interaction_letter(0.0, -0.01, 0.01, 0.05) == "ADDITIVE" and interaction_letter(-0.2, -0.3, -0.1, 0.05) == "SUB-ADDITIVE"
    assert interaction_letter(0.0, -0.2, 0.2, 0.05) == "UNRESOLVED" and abs(slope([0, 1, 2, 3], [1, 3, 5, 7]) - 2.0) < 1e-12
    for w in (256,):
        p = pool(w); assert len(p) == N_PER_BIN * N_BINS and np.all(np.diff(p) > 0) and np.array_equal(p, pool(w)) and not (set(int(i) for i in p) & study_ids())
        assert not (set(int(i) for i in p) & (set(int(i) for i in R22.pool(w)) | set(int(i) for i in R20.pool(w)) | set(int(i) for i in R17.pool(w))))
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", nargs="?", choices=["freeze", "run", "report"]); ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--width", type=int, choices=WIDTHS); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.selftest: selftest(); return
    if a.action == "freeze": freeze(); return
    if a.action == "report": report(Path(a.out) if a.out else OUT); return
    if a.action == "run":
        if a.width is None: ap.error("--width required")
        if not FROZEN.exists(): sys.exit("the laws are not frozen: run `freeze` and commit tools/p25_frozen_laws.json first")
        R19(a.width, Path(a.out) if a.out else OUT).run(a.smoke)
    else: ap.print_help()


if __name__ == "__main__":
    main()
