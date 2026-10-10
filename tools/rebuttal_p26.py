#!/usr/bin/env python3
# Ledger: THE DISCUSSION-PERIOD EXPERIMENT P26 — does a compact readout-invisible component carry a history's response to identical
# corrections? (registration Documentation/Note_2026-10-10_Rebuttal_P26_Registration.md, written before any row). MEASUREMENT, $0, inference
# only on the banked attention checkpoints. Reuses P25's runner (release states, the donor history loop, variant rollouts, the angle- and
# norm-matched rotation) and P16's relabel unchanged.
"""P26, two stages per receiver (192 identification, 256 frozen-procedure replication). Pairs = a puzzle's fixed-start state and its Gaussian-start
history after iteration 1, both unsolved, sharing >= 12 wrong cells; aligned challenges (4 never-edited evaluation cells and 12 correction masks,
4 of equal size per dose, drawn from the shared wrong cells and identical for both histories). Stage D (discovery puzzles): intact / full
transfer / matched random; then `learn` freezes a k = w/8 basis from the selected discovery pair-directions' donor-minus-recipient
readout-invisible differences. Stage C (confirmation puzzles): adds the common-basis compact transfer and an equal-rank random projection scaled
to its magnitude, and (secondary) the adaptive per-pair compression with its own scaled random projection. Exact restoration (the saved tensors
written back) must reproduce the intact rollouts bitwise. Rules: R1 RESPONSE-TRANSFER, R2 CHALLENGE-SPECIFIC, R3 COMPACT (on R2)."""
import argparse, hashlib, json, math, os, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import rebuttal_p1p2 as P
import rebuttal_p15 as Q
import rebuttal_p16 as R16
import rebuttal_p17 as R17
import rebuttal_p25 as P25

OUT = ROOT / "runs/analysis/rebuttal_20261012"
SEED = 20261012
WIDTHS = (192, 256)
N_PER_BIN, N_BINS, MIN_SHARED, E_SIZE = 48, 8, 12, 4
DOSES, MASKS = (0.25, 0.5, 0.75), 4
SELECT_FRAC = 0.5
COND_D = ("intact", "full", "full_random")
COND_C = ("intact", "full", "full_random", "global", "global_rand", "adaptive", "adaptive_rand")
K, VB, N_BOOT = Q.K, Q.VBATCH, 2000


def utc(): return datetime.now(timezone.utc).isoformat()


def basis_path(width, smoke_out=None):
    """The frozen basis (tracked in tools/); a smoke's learned stand-in lives in its own output folder and is never read by a real run."""
    return Path(smoke_out) / f"p26_basis_{width}_smoke.json" if smoke_out else ROOT / f"tools/p26_basis_{width}.json"


# ---------------- pure helpers (selftested) ----------------
def pool(width):
    """48 fresh puzzles per full-test rating octile (rank-based bins), excluding the study's 512, the P12/P17/P20/P22 pools and P25's pool."""
    import rebuttal_p12 as P12, rebuttal_p20 as R20, rebuttal_p22 as R22
    rec = np.load(ROOT / f"paper/code/evidence/benchmark/attention_{width}_d16.npz", allow_pickle=False)
    r128 = np.load(ROOT / "paper/code/evidence/benchmark/attention_128_d16.npz", allow_pickle=False)
    excl = P25.study_ids() | set(int(i) for i in P12.pool_ids(r128["cold_exact"], r128["idx"])) | set(int(i) for i in R17.pool(width)) \
        | set(int(i) for i in R20.pool(width)) | set(int(i) for i in R22.pool(width)) | set(int(i) for i in P25.pool(width))
    idx, rating = rec["idx"], rec["rating"]; order = np.lexsort((idx, rating)); bins = np.empty(len(idx), int); bins[order] = (np.arange(len(idx)) * N_BINS) // len(idx)
    out = []
    for b in range(N_BINS):
        cand = np.array(sorted(int(i) for i in idx[bins == b] if int(i) not in excl))
        out += list(np.random.default_rng([SEED, 26, width, b]).choice(cand, N_PER_BIN, replace=False))
    return np.sort(np.array(out, np.int64))


def split(pids, width):
    """Puzzle-level split of the eligible pairs: the first half of a seeded permutation = discovery, the rest = confirmation."""
    perm = np.random.default_rng([SEED, 26, width, 0]).permutation(np.sort(np.asarray(pids)))
    h = len(perm) // 2; return set(int(x) for x in perm[:h]), set(int(x) for x in perm[h:])


def orth_basis_excluding(vectors, lm, k):
    """Top-k right singular vectors of `vectors` (n, w), made orthogonal to lm (Gram-Schmidt) and orthonormal."""
    vhat = lm / np.linalg.norm(lm); X = vectors - (vectors @ vhat)[:, None] * vhat
    _, _, Vt = np.linalg.svd(X, full_matrices=False); U = Vt[:k].T
    U = U - np.outer(vhat, vhat @ U); U, _ = np.linalg.qr(U); return U                           # (w, k)


def random_basis(w, k, lm, rng):
    vhat = lm / np.linalg.norm(lm); G = rng.standard_normal((w, k)); G -= np.outer(vhat, vhat @ G); V, _ = np.linalg.qr(G); return V


def project(D, U):
    """D (..., w) -> its projection onto span(U) (U orthonormal, (w, k))."""
    return (D @ U) @ U.T


def contrasts(Y):
    """Y: (MASKS,) responses to the 4 equal-size masks of one (pair, direction, dose) -> the 2 independent contrasts (m0 - m1, m2 - m3)."""
    return np.array([Y[0] - Y[1], Y[2] - Y[3]])


def ols(y, X):
    Xa = np.hstack([X, np.ones((len(X), 1))]); return np.linalg.lstsq(Xa, y, rcond=None)[0]


# ---------------- the runner ----------------
class R20p26(P25.R19):
    def __init__(self, width, out=OUT):
        super().__init__(width, out)
        self.meta.update(study="rebuttal_20261012", pipeline="tools/rebuttal_p26.py over tools/rebuttal_p25.py")
        self.ATS.write_json(self.dir / "runtime.json", self.meta)

    def pairs(self, smoke=False):
        """All eligible aligned pairs for this receiver (deterministic; recomputed by each stage)."""
        lm = self.lm; ids = pool(self.width)
        if smoke: ids = ids[:6]
        with np.load(R17.DATA, allow_pickle=False) as d: puz_all, sol_all = d["test_q"][ids].astype(np.int32), d["test_a"][ids].astype(np.int32)
        xs, ZA, ZB = [], [], []
        for b0 in range(0, len(ids), VB):
            x, st = self.release_states(puz_all[b0:b0 + VB]); xs.append(np.asarray(x)); ZA.append(st[0]["z"])
            if b0 == 0:
                _, st2 = self.release_states(puz_all[b0:b0 + VB]); assert np.array_equal(st[0]["z"], st2[0]["z"]), "gate: release states not reproducible"
            z0 = np.stack([self.EV.mi_z0(SEED, int(p), 7, st[0]["z"].shape[1:], 1.0, "gauss") for p in ids[b0:b0 + VB]]).astype(np.float32)
            pad = VB - len(z0); xp = np.asarray(x)
            zp = np.concatenate([z0, np.repeat(z0[:1], pad, 0)]) if pad else z0; xpp = np.concatenate([xp, np.repeat(xp[:1], pad, 0)]) if pad else xp
            ZB.append(self.history(zp, xpp, 1)[:len(z0)])
        xcan = np.concatenate(xs, 0); ZA = np.concatenate(ZA, 0); ZB = np.concatenate(ZB, 0); out = []
        for b, pid in enumerate(ids):
            sol, puz = sol_all[b].reshape(81), puz_all[b].reshape(81)
            hA, lA, hB, lB = ZA[b, 0], ZA[b, 1], ZB[b, 0], ZB[b, 1]
            gA = np.einsum("fsw,w->sf", hA.astype(np.float64), lm).argmax(1) + 1; gB = np.einsum("fsw,w->sf", hB.astype(np.float64), lm).argmax(1) + 1
            WA, WB = np.where((puz == 0) & (gA != sol))[0], np.where((puz == 0) & (gB != sol))[0]; W = np.array(sorted(set(WA) & set(WB)))
            if len(W) < MIN_SHARED: continue
            E = np.sort(np.random.default_rng([SEED, int(pid), 1]).choice(W, E_SIZE, replace=False)); Rr = np.array(sorted(set(W) - set(E)))
            masks = [(i, m, np.sort(np.random.default_rng([SEED, int(pid), 2, i, m]).choice(Rr, max(1, int(round(f * len(Rr)))), replace=False)))
                     for i, f in enumerate(DOSES) for m in range(MASKS)]
            out.append(dict(pid=int(pid), b=b, h=(hA, hB), l=(lA, lB), grid=(gA, gB), W=(WA, WB), Wshared=W, sol=sol, E=E, masks=masks))
        return out, xcan

    def roll_variants(self, specs, xcan, sol81, build, tag, counter):
        """P25's padded-chunk rollouts, with per-chunk wall times recorded for the compute record."""
        import time
        out = np.zeros((len(specs), K, 81, 9), np.float32); self.chunk_times = []
        for c0 in range(0, len(specs), VB):
            t0 = time.time(); out[c0:c0 + VB] = super().roll_variants(specs[c0:c0 + VB], xcan, sol81, build, tag, counter); self.chunk_times.append(time.time() - t0)
            print(json.dumps(dict(tag=tag, done=counter[0], of=len(specs), chunk_s=round(self.chunk_times[-1], 2), at=utc())), flush=True)
        return out

    def condition_states(self, pr, di, conds, U=None):
        """The recipient's slow state under each condition (direction di: 0 = A <- B, 1 = B <- A). Every change is readout-orthogonal."""
        lm = self.lm; r, d = (0, 1) if di == 0 else (1, 0)
        hr = pr["h"][r].astype(np.float64); hd = pr["h"][d].astype(np.float64); par_r, perp_r = P.project(hr, lm); perp_d = P.project(hd, lm)[1]; D = perp_d - perp_r
        out, info = {}, {}
        for c in conds:
            if c == "intact": h = pr["h"][r]
            elif c == "full": h = (hr + D).astype(np.float32)
            elif c == "full_random":
                newp, _ = P25.rotate_matched(perp_r, perp_d, lm, np.random.default_rng([SEED, pr["pid"], 6, di])); h = (par_r + newp).astype(np.float32)
            elif c in ("global", "adaptive"):
                Ub = U if c == "global" else orth_basis_excluding(D.reshape(-1, D.shape[-1]), lm, self.width // 8)     # adaptive: this pair's own top-k directions
                Dc = project(D, Ub); info[f"energy_{c}"] = float((Dc ** 2).sum() / max((D ** 2).sum(), 1e-30)); info[f"norm_{c}"] = float(np.linalg.norm(Dc)); h = (hr + Dc).astype(np.float32)
            elif c in ("global_rand", "adaptive_rand"):
                assert f"norm_{c[:-5]}" in info, "the compact condition precedes its random control"
                V = random_basis(D.shape[-1], self.width // 8, lm, np.random.default_rng([SEED, pr["pid"], 9, di])); Dv = project(D, V)    # one random subspace per pair-direction, both controls
                info["energy_random"] = float((Dv ** 2).sum() / max((D ** 2).sum(), 1e-30)); Dv = Dv * (info[f"norm_{c[:-5]}"] / max(np.linalg.norm(Dv), 1e-30))
                info[f"scale_ratio_{c}"] = float(np.linalg.norm(Dv) / max(info[f"norm_{c[:-5]}"], 1e-30)); h = (hr + Dv).astype(np.float32)
            out[c] = h
        return out, info

    def run(self, stage, smoke=False):
        try:
            lm = self.lm; prs, xcan = self.pairs(smoke); disc, conf = split([p["pid"] for p in prs], self.width)
            use = disc if stage == "D" else conf; conds = COND_D if stage == "D" else COND_C; U = None
            bp = basis_path(self.width, self.dir.parent if smoke else None)
            if stage == "C" and smoke and not bp.exists():
                U = random_basis(self.width, self.width // 8, lm, np.random.default_rng([SEED, 26, self.width, 99]))          # smoke only: a stand-in, never read
            elif stage == "C":
                if not bp.exists(): sys.exit(f"the basis for width {self.width} is not frozen: run `learn` first")
                B = json.loads(bp.read_text()); U = np.array(B["U"], np.float64); assert U.shape == (self.width, self.width // 8)
                assert np.allclose(U.T @ U, np.eye(U.shape[1]), atol=1e-8) and np.abs(U.T @ (lm / np.linalg.norm(lm))).max() < 1e-8, "gate: basis not orthonormal or not readout-orthogonal"
                assert set(B["discovery_pids"]) == disc, "gate: the discovery split differs from the one the basis was learned on"
            if smoke: use = set(p["pid"] for p in prs)
            gates = dict(display=0, e_edited=0, outside_shared=0, score_shift=0.0, self_bitwise=0, scale_ratio_dev=0.0, restore_tensor=0, restore_rollout=0, restore_maxdev=0.0)
            S, V, H, RST = [], [], {}, []; energies = []
            for pr in prs:
                if pr["pid"] not in use: continue
                for c in pr["masks"]: gates["outside_shared"] += int(not set(c[2].tolist()) <= set(pr["Wshared"].tolist()) or bool(set(c[2].tolist()) & set(pr["E"].tolist())))
                for di in (0, 1):
                    r = 0 if di == 0 else 1; hs, info = self.condition_states(pr, di, conds, U); energies.append(info)
                    s0 = np.einsum("fsw,w->sf", pr["h"][r].astype(np.float64), lm)
                    for c, h in hs.items():
                        if c != "intact": gates["score_shift"] = max(gates["score_shift"], float(np.abs(np.einsum("fsw,w->sf", h.astype(np.float64), lm) - s0).max()))
                    for k_ in [k_ for k_ in info if k_.startswith("scale_ratio")]: gates["scale_ratio_dev"] = max(gates["scale_ratio_dev"], abs(info[k_] - 1.0))
                    hp, pp = P.project(pr["h"][r].astype(np.float64), lm); gates["self_bitwise"] += int(not np.array_equal((pr["h"][r].astype(np.float64) + (pp - pp)).astype(np.float32), pr["h"][r]))
                    si = len(S); H[si] = dict(hs=hs, l=pr["l"][r], grid=pr["grid"][r], sol=pr["sol"], E=pr["E"], saved=(pr["h"][r].copy(), pr["l"][r].copy()))
                    S.append(dict(pid=pr["pid"], dir=di, b=pr["b"], E=pr["E"], sol=pr["sol"].astype(np.int8), grid=pr["grid"][r].astype(np.int8), nwrong=len(pr["W"][r]), nwrong_donor=len(pr["W"][1 - r])))
                    for ci, c in enumerate(conds):
                        V.append(dict(si=si, b=pr["b"], cond=ci, arm=0, dose=-1, mask=-1, cells=np.array([], int)))
                        for i, m, cells in pr["masks"]: V.append(dict(si=si, b=pr["b"], cond=ci, arm=1, dose=i, mask=m, cells=cells))
                    RST.append((si, len(V) - len(conds) * (1 + len(pr["masks"]))))       # the state's intact block: no-edit, then masks in (dose, mask) order
            nmain = len(V)
            for si, v0 in RST:                              # exact restoration: the saved recipient tensors written back after every other variant is built
                for j in [v0] + [v0 + 1 + i * MASKS for i in range(len(DOSES))]:
                    assert V[j]["si"] == si and V[j]["cond"] == 0 and (j == v0 or V[j]["mask"] == 0)
                    V.append(dict(si=si, b=V[j]["b"], cond=-1, arm=V[j]["arm"], dose=V[j]["dose"], mask=V[j]["mask"], cells=V[j]["cells"], twin=j))
            def build(sp):
                Hh = H[sp["si"]]
                if sp["cond"] == -1:
                    h, l = Hh["saved"]; gates["restore_tensor"] += int(not (np.array_equal(h, Hh["hs"]["intact"]) and np.array_equal(l, Hh["l"])))
                else: h, l = Hh["hs"][conds[sp["cond"]]], Hh["l"]
                cells = sp["cells"]
                if np.isin(cells, Hh["E"]).any(): gates["e_edited"] += 1
                if len(cells) == 0: return h, l
                for c in cells: h, l = R16.cell_relabel(h, l, int(c), int(Hh["grid"][c]), int(Hh["sol"][c]))
                want = Hh["grid"].astype(int).copy(); want[cells] = Hh["sol"][cells]
                if not np.array_equal(np.einsum("fsw,w->sf", h, self.lm32).argmax(1) + 1, want): gates["display"] += 1
                return h, l
            cnt = [0]; sc = self.roll_variants(V, xcan, None, build, f"stage{stage}", cnt)
            for q in range(nmain, len(V)):
                j = V[q]["twin"]; gates["restore_rollout"] += int(not np.array_equal(sc[q], sc[j])); gates["restore_maxdev"] = max(gates["restore_maxdev"], float(np.abs(sc[q] - sc[j]).max()))
            V, sc = V[:nmain], sc[:nmain]
            assert gates["restore_tensor"] == 0 and gates["restore_rollout"] == 0, f"gate: {gates}"
            assert gates["display"] == 0 and gates["e_edited"] == 0 and gates["outside_shared"] == 0, f"gate: {gates}"
            assert gates["score_shift"] < 1e-3 and gates["self_bitwise"] == 0 and gates["scale_ratio_dev"] < 1e-6, f"gate: {gates}"
            Earr = np.stack([s_["E"] for s_ in S]); vs = np.array([v["si"] for v in V])
            esc = np.stack([sc[i][:, Earr[vs[i]]] for i in range(len(V))]).astype(np.float32)
            disp_e = np.stack([(sc[i][:, Earr[vs[i]]].argmax(-1) + 1) for i in range(len(V))]).astype(np.int8)
            cells = np.full((len(V), 81), -1, np.int16)
            for i, v in enumerate(V): cells[i, :len(v["cells"])] = v["cells"]
            dst = self.dir / (f"p26_{stage}_smoke.npz" if smoke else f"p26_{stage}.npz")
            np.savez_compressed(dst, conds=np.array(conds), var_state=vs.astype(np.int32), var_cond=np.array([v["cond"] for v in V], np.int8), var_arm=np.array([v["arm"] for v in V], np.int8),
                                var_dose=np.array([v["dose"] for v in V], np.int8), var_mask=np.array([v["mask"] for v in V], np.int8), var_cells=cells, var_nedit=np.array([len(v["cells"]) for v in V], np.int16),
                                escores=esc, disp_e=disp_e, state_pid=np.array([s_["pid"] for s_ in S], np.int64), state_dir=np.array([s_["dir"] for s_ in S], np.int8), state_E=Earr.astype(np.int16),
                                state_sol=np.stack([s_["sol"] for s_ in S]), state_grid=np.stack([s_["grid"] for s_ in S]), state_nwrong=np.array([s_["nwrong"] for s_ in S], np.int16),
                                state_nwrong_donor=np.array([s_["nwrong_donor"] for s_ in S], np.int16), energies=json.dumps(energies), gates=json.dumps(gates),
                                meta=json.dumps(dict(self.meta, created=utc(), stage=stage, smoke=smoke, eligible_pairs=len(prs), discovery=len(disc), confirmation=len(conf), used=len(S) // 2,
                                                     basis_sha256=(P25.sha(bp) if stage == "C" and bp.exists() else None), basis_stand_in=bool(stage == "C" and not bp.exists()))))
            if not smoke: self.ATS.write_json(dst.with_suffix(".sha256.json"), dict(sha256=self.ATS.sha(dst)))
            print(json.dumps(dict(width=self.width, stage=stage, eligible_pairs=len(prs), discovery=len(disc), confirmation=len(conf), recipients=len(S), variants=len(V), gates=gates)))
            self.ATS.verify_sources(); self.status("inference_complete", completed=utc()); self.ATS.log("model_complete", width=self.width, group=f"p26{stage}")
        except BaseException as exc:
            self.status("failed", error=repr(exc)); raise


# ---------------- the frozen reader ----------------
def unit_tables(d):
    """Per (state = pair-direction, condition): the no-edit response b and per (dose, mask) the response Y, on the evaluation cells at k = 1."""
    st, cond, arm, E, sol, de = d["var_state"], d["var_cond"], d["var_arm"], d["state_E"], d["state_sol"], d["disp_e"]
    Y, Bm = {}, {}
    for i in range(len(st)):
        s, c = int(st[i]), int(cond[i]); y = float((de[i, 0] == sol[s, E[s]]).mean())
        if arm[i] == 0: Bm[(s, c)] = y
        else: Y[(s, c, int(d["var_dose"][i]), int(d["var_mask"][i]))] = y
    return Y, Bm


def donor_state(d):
    pid, dr = d["state_pid"], d["state_dir"]; key = {(int(p), int(x)): s for s, (p, x) in enumerate(zip(pid, dr))}
    return {s: key.get((int(pid[s]), 1 - int(dr[s]))) for s in range(len(pid))}


def r1_rows(d, cx, cref):
    """R1 rows for condition cx vs reference cref: per (state, dose, mask): dependent = (Y_cx - b_cx) - (Y_cref - b_cref)... returned separately."""
    Y, Bm = unit_tables(d); don = donor_state(d); conds = [str(x) for x in d["conds"]]; ix = conds.index("intact"); rows = []
    for s in range(len(d["state_pid"])):
        ds = don[s]
        if ds is None: continue
        nr, nd = float(d["state_nwrong"][s]), float(d["state_nwrong_donor"][s])
        for i in range(len(DOSES)):
            for m in range(MASKS):
                ne = d["var_nedit"][np.where((d["var_state"] == s) & (d["var_arm"] == 1) & (d["var_dose"] == i) & (d["var_mask"] == m))[0][0]]
                fr, fd = ne / nr, ne / nd
                rows.append(dict(pid=int(d["state_pid"][s]), s=s, i=i, m=m, YR=Y[(s, ix, i, m)] - Bm[(s, ix)], YD=Y[(ds, ix, i, m)] - Bm[(ds, ix)], bR=Bm[(s, ix)], bD=Bm[(ds, ix)], fR=fr, fD=fd,
                                 Yx=Y[(s, conds.index(cx), i, m)] - Bm[(s, conds.index(cx))], bx=Bm[(s, conds.index(cx))],
                                 Yref=Y[(s, conds.index(cref), i, m)] - Bm[(s, conds.index(cref))], bref=Bm[(s, conds.index(cref))]))
    return rows


def r1_beta(rows, which):
    """The donor coefficient in the prespecified nuisance model for the dependent response of `which` ('x' or 'ref')."""
    y = np.array([r["Y" + which] for r in rows]); b = np.array([r["b" + which] for r in rows])
    col = lambda k_: np.array([r[k_] for r in rows]); fR, fD, bR, bD = col("fR"), col("fD"), col("bR"), col("bD")
    X = np.stack([col("YD"), col("YR"), b, bR, bD, fR, fD, fR ** 2, fD ** 2, b * fR, bR * fR, bD * fD], 1)      # the donor coefficient first; the registered nuisance set
    return ols(y, X)[0]


def baseline_rows(d):
    """Per state: the recipient's, donor's and each condition's no-edit repair on the evaluation cells (P25's baseline transfer, aligned cells)."""
    Y, Bm = unit_tables(d); don = donor_state(d); conds = [str(x) for x in d["conds"]]; rows = []
    for s in range(len(d["state_pid"])):
        if don[s] is None: continue
        rows.append(dict(pid=int(d["state_pid"][s]), bR=Bm[(s, 0)], bD=Bm[(don[s], 0)], **{f"b_{c}": Bm[(s, j)] for j, c in enumerate(conds)}))
    return rows


def baseline_slope(rows, c):
    x = np.array([r["bD"] - r["bR"] for r in rows]); y = np.array([r[f"b_{c}"] - r["bR"] for r in rows]); return float(ols(y, x[:, None])[0])


def r2_rows(d, cx, cref):
    """R2 units per (state, dose, contrast): dependent d_cx - d_cref; regressors d_donor, d_recipient (intact)."""
    Y, Bm = unit_tables(d); don = donor_state(d); conds = [str(x) for x in d["conds"]]; ix = conds.index("intact"); rows = []
    for s in range(len(d["state_pid"])):
        ds = don[s]
        if ds is None: continue
        for i in range(len(DOSES)):
            get = lambda st_, c: np.array([Y[(st_, c, i, m)] for m in range(MASKS)])
            dX, dR_, dD = contrasts(get(s, conds.index(cx))) - contrasts(get(s, conds.index(cref))), contrasts(get(s, ix)), contrasts(get(ds, ix))
            for q in range(2): rows.append(dict(pid=int(d["state_pid"][s]), dep=dX[q], dD=dD[q], dR=dR_[q]))
    return rows


def r2_beta(rows):
    return ols(np.array([r["dep"] for r in rows]), np.stack([[r["dD"] for r in rows], [r["dR"] for r in rows]], 1))[0]


def boot(fn, rows_list, seed, n=N_BOOT):
    """Puzzle bootstrap keeping both directions and all challenges of a puzzle together; fn(*row_lists) -> scalar."""
    pids = np.unique([r["pid"] for r in rows_list[0]]); rng = np.random.default_rng(seed); idx = [{} for _ in rows_list]
    for j, rl in enumerate(rows_list):
        for k_, r in enumerate(rl): idx[j].setdefault(r["pid"], []).append(k_)
    vals = []
    for _ in range(n):
        pick = rng.choice(pids, len(pids)); sub = [[rl[k_] for p in pick for k_ in idx[j].get(p, [])] for j, rl in enumerate(rows_list)]; vals.append(fn(*sub))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def report_stage(width, stage, base=OUT, smoke=False):
    f = base / f"attention_{width}/p26_{stage}{'_smoke' if smoke else ''}.npz"
    if not f.exists(): return None
    d = np.load(f, allow_pickle=False); res = dict(meta={k_: json.loads(str(d["meta"]))[k_] for k_ in ("eligible_pairs", "discovery", "confirmation", "used")}, gates=json.loads(str(d["gates"])))
    bl = baseline_rows(d); eb = lambda r: baseline_slope(r, "full") - baseline_slope(r, "full_random")
    res["BASELINE"] = dict(slope_full=baseline_slope(bl, "full"), slope_random=baseline_slope(bl, "full_random"), diff=eb(bl), ci=boot(eb, [bl], [SEED, 26, width, 5]), states=len(bl))
    res["BASELINE"]["letter"] = "BASELINE-TRANSFER" if res["BASELINE"]["ci"][0] > 0 else "NOT-RESOLVED"
    rows = r1_rows(d, "full", "full_random"); e1 = lambda r: r1_beta(r, "x") - r1_beta(r, "ref")
    res["R1"] = dict(beta_full=r1_beta(rows, "x"), beta_full_random=r1_beta(rows, "ref"), diff=e1(rows), ci=boot(e1, [rows], [SEED, 26, width, 1]), units=len(rows))
    res["R1"]["letter"] = "RESPONSE-TRANSFER" if res["R1"]["ci"][0] > 0 else "NOT-RESOLVED"
    r2 = r2_rows(d, "full", "full_random"); res["R2"] = dict(beta=r2_beta(r2), ci=boot(r2_beta, [r2], [SEED, 26, width, 2]), units=len(r2))
    res["R2"]["letter"] = "CHALLENGE-SPECIFIC" if res["R2"]["ci"][0] > 0 else "NOT-RESOLVED"
    conds = [str(x) for x in d["conds"]]
    if "global" in conds:
        rc = r2_rows(d, "global", "global_rand"); half = lambda a, b: r2_beta(a) - 0.5 * r2_beta(b)
        c = dict(E_full=res["R2"]["beta"], E_full_ci=res["R2"]["ci"], E_compact=r2_beta(rc), E_compact_ci=boot(r2_beta, [rc], [SEED, 26, width, 3]),
                 half_diff=half(rc, r2), half_ci=boot(half, [rc, r2], [SEED, 26, width, 4]))
        if c["E_full_ci"][0] <= 0: c["letter"] = "NOT-APPLICABLE"
        elif c["E_compact_ci"][0] > 0 and c["half_ci"][0] > 0: c["letter"] = "COMPACT"
        elif c["E_compact_ci"][0] > 0: c["letter"] = "COMPACT-PARTIAL"
        else: c["letter"] = "NOT-COMPACT"
        ra = r2_rows(d, "adaptive", "adaptive_rand"); c["adaptive_secondary"] = r2_beta(ra); c["adaptive_half_diff"] = half(ra, r2)
        en = json.loads(str(d["energies"])); c["retained_energy"] = {k_: float(np.mean([e[k_] for e in en if k_ in e])) for k_ in ("energy_global", "energy_adaptive", "energy_random")}
        r1c = r1_rows(d, "global", "global_rand"); c["R1_compact"] = r1_beta(r1c, "x") - r1_beta(r1c, "ref"); res["R3"] = c
    return res


def report(base=OUT, smoke=False):
    lines = ["P26 REPORT (rules registered before any row; bases frozen in tools/p26_basis_<width>.json before stage C)"]; js = {}
    for w in WIDTHS:
        for stage in ("D", "C"):
            r = report_stage(w, stage, base, smoke)
            if r is None: lines.append(f"attention_{w} stage {stage}: not run"); continue
            js[f"attention_{w}_{stage}"] = r; tag = "(discovery, exploratory)" if stage == "D" else "(confirmation)"
            b_ = r["BASELINE"]; lines.append(f"attention_{w} stage {stage}: baseline transfer slope full {b_['slope_full']:+.3f} vs random {b_['slope_random']:+.3f}, difference {b_['diff']:+.3f} [{b_['ci'][0]:+.3f}, {b_['ci'][1]:+.3f}] -> {b_['letter']}")
            lines.append(f"attention_{w} stage {stage} {tag}: pairs used {r['meta']['used']} of {r['meta']['eligible_pairs']} | R1 full {r['R1']['beta_full']:+.3f} vs random {r['R1']['beta_full_random']:+.3f}, "
                         f"difference {r['R1']['diff']:+.3f} [{r['R1']['ci'][0]:+.3f}, {r['R1']['ci'][1]:+.3f}] -> {r['R1']['letter']} | R2 beta {r['R2']['beta']:+.3f} [{r['R2']['ci'][0]:+.3f}, {r['R2']['ci'][1]:+.3f}] -> {r['R2']['letter']} | gates {r['gates']}")
            if "R3" in r:
                c = r["R3"]; lines.append(f"   R3: E_compact {c['E_compact']:+.3f} [{c['E_compact_ci'][0]:+.3f}, {c['E_compact_ci'][1]:+.3f}] vs E_full {c['E_full']:+.3f}; E_compact - E_full/2 {c['half_diff']:+.3f} [{c['half_ci'][0]:+.3f}, {c['half_ci'][1]:+.3f}] -> {c['letter']} | "
                                         f"adaptive (secondary) {c['adaptive_secondary']:+.3f}, minus E_full/2 {c['adaptive_half_diff']:+.3f} | retained energy {c['retained_energy']} | R1 compact {c['R1_compact']:+.3f}")
    have = [w for w in WIDTHS if f"attention_{w}_C" in js]
    if len(have) == 2:
        for key, letter in (("BASELINE", "BASELINE-TRANSFER"), ("R1", "RESPONSE-TRANSFER"), ("R2", "CHALLENGE-SPECIFIC")):
            lines.append(f"{letter} on both receivers' confirmation: {all(js[f'attention_{w}_C'][key]['letter'] == letter for w in have)}")
        lines.append(f"R3 letters: " + ", ".join(f"{w}: {js[f'attention_{w}_C']['R3']['letter']}" for w in have))
    sfx = "_smoke" if smoke else ""; (base / f"report{sfx}.txt").write_text("\n".join(lines) + "\n"); (base / f"report{sfx}.json").write_text(json.dumps(js, indent=1, default=float)); print("\n".join(lines))


# ---------------- learning the global basis (after stage D, before stage C) ----------------
def learn(width, out=OUT, smoke=False):
    """Select the top SELECT_FRAC of discovery pair-directions by s = mean over their contrasts of (d_full - d_full_random) * d_donor, then
    U = the top-k (k = w/8) right singular vectors of their pooled donor-minus-recipient readout-invisible difference vectors, orthogonal to the readout.
    Deterministic given the stage-D file and the registered procedure (a recount re-runs it and compares hashes)."""
    src = out / f"attention_{width}/p26_D{'_smoke' if smoke else ''}.npz"; d = np.load(src, allow_pickle=False); r2 = {}
    for s_, v in r2_rows_by_state(d): r2.setdefault(s_, []).append(v)
    score = {s_: float(np.mean(v)) for s_, v in r2.items()}; order = sorted(score, key=lambda s_: (-score[s_], s_)); keep = order[:int(round(SELECT_FRAC * len(order)))]
    r = R20p26(width, out / "learn"); prs, _ = r.pairs(smoke); disc, conf = split([p["pid"] for p in prs], width); key = {(p["pid"], di): p for p in prs for di in (0, 1)}; rows = []
    pidd, dirr = d["state_pid"], d["state_dir"]
    for s_ in keep:
        p, di = key[(int(pidd[s_]), int(dirr[s_]))], int(dirr[s_]); rr, dd = (0, 1) if di == 0 else (1, 0)
        D = P.project(p["h"][dd].astype(np.float64), r.lm)[1] - P.project(p["h"][rr].astype(np.float64), r.lm)[1]; rows.append(D.reshape(-1, D.shape[-1]))
    U = orth_basis_excluding(np.concatenate(rows, 0), r.lm, width // 8); dst = basis_path(width, out if smoke else None)
    js = dict(created=utc(), width=width, k=width // 8, select_frac=SELECT_FRAC, selected_states=sorted(int(s_) for s_ in keep), selected_scores=[score[s_] for s_ in keep],
              discovery_pids=sorted(disc), stage_D_sha256=P25.sha(src), smoke=smoke,
              rule="top SELECT_FRAC of discovery pair-directions by mean (d_full - d_full_random) * d_donor; U = top-k right singular vectors of pooled perp_donor - perp_recipient, orthogonalized to the readout", U=U.tolist())
    dst.write_text(json.dumps(js)); print(f"attention_{width}: basis frozen from {len(keep)} of {len(order)} discovery pair-directions -> {dst} {P25.sha(dst)}")


def r2_rows_by_state(d):
    Y, Bm = unit_tables(d); don = donor_state(d); conds = [str(x) for x in d["conds"]]; ix, fu, fr = conds.index("intact"), conds.index("full"), conds.index("full_random"); out = []
    for s in range(len(d["state_pid"])):
        ds = don[s]
        if ds is None: continue
        for i in range(len(DOSES)):
            get = lambda st_, c: np.array([Y[(st_, c, i, m)] for m in range(MASKS)])
            dep, dD = contrasts(get(s, fu)) - contrasts(get(s, fr)), contrasts(get(ds, ix))
            for q in range(2): out.append((s, float(dep[q] * dD[q])))
    return out


def selftest():
    rng = np.random.default_rng(0); w = 32; lm = rng.standard_normal(w); vh = lm / np.linalg.norm(lm)
    X = rng.standard_normal((500, w)) @ np.diag(np.linspace(3, 0.1, w)); U = orth_basis_excluding(X, lm, 4)
    assert U.shape == (w, 4) and np.allclose(U.T @ U, np.eye(4), atol=1e-10) and np.allclose(vh @ U, 0, atol=1e-10)
    V = random_basis(w, 4, lm, rng); assert np.allclose(V.T @ V, np.eye(4), atol=1e-10) and np.allclose(vh @ V, 0, atol=1e-10)
    D = rng.standard_normal((10, w)); assert np.allclose(project(project(D, U), U), project(D, U))
    assert np.allclose(contrasts(np.array([0.5, 0.25, 1.0, 0.0])), [0.25, 1.0])
    yy = rng.standard_normal(200); XX = rng.standard_normal((200, 2)); assert np.allclose(ols(2 * XX[:, 0] - XX[:, 1] + 0.5, XX), [2, -1, 0.5])
    a, b = split(np.arange(100), 192); assert len(a) == 50 and len(b) == 50 and not (a & b) and split(np.arange(100), 192) == (a, b)
    p = pool(192); assert len(p) == N_PER_BIN * N_BINS and np.array_equal(p, pool(192)) and not (set(int(i) for i in p) & set(int(i) for i in P25.pool(192)))
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", nargs="?", choices=["run", "learn", "report"]); ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--width", type=int, choices=WIDTHS); ap.add_argument("--stage", choices=["D", "C"]); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--out", default=None)
    a = ap.parse_args(); out = Path(a.out) if a.out else OUT
    if a.selftest: selftest(); return
    if a.action == "report": report(out, a.smoke); return
    if a.action == "learn":
        if a.width is None: ap.error("--width required")
        learn(a.width, out, a.smoke); return
    if a.action == "run":
        if a.width is None or a.stage is None: ap.error("--width and --stage required")
        R20p26(a.width, out).run(a.stage, a.smoke)
    else: ap.print_help()


if __name__ == "__main__":
    main()
