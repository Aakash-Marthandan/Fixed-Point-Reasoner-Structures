#!/usr/bin/env python3
# Ledger: THE DISCUSSION-PERIOD EXPERIMENT P27 — where in the readout-invisible difference does the transferred correction response live?
# (registration Documentation/Note_2026-10-11_Rebuttal_P27_Registration.md, written before any row). MEASUREMENT, $0, inference only on the
# banked attention checkpoints. Reuses P26's runner (release states, donor histories, padded rollouts with chunk timing, exact restoration),
# P26's reader helpers and P26's frozen common bases unchanged.
"""P27, one confirmation stage per receiver (192 and 256; no discovery: the leading subspace is P26's frozen basis, applied out of sample).
Pairs and aligned challenges as P26 on 384 fresh puzzles per receiver. Per pair-direction, delta = perp_d - perp_r splits into L (its projection on
the frozen basis) and M (the orthogonal remainder). Eight conditions: intact, full (L + M), P26's matched rotation (the replication control),
L, xi_L, M, xi_M and the combined control xi_L + xi_M, where each xi_C is drawn inside C's subspace with ||xi_C|| = ||C|| and <h_r, xi_C> =
<h_r, C> per (field, cell) row, drawn once per pair-direction and reused across all challenges. Primary statistic: the all-four-mask centered
donor coefficient. Rules: S (replication vs P26's control), component effects vs their controls, the total vs the combined control, half-strength
against that total, direct differences, and the interaction against the corresponding random interaction."""
import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import rebuttal_p1p2 as P
import rebuttal_p16 as R16
import rebuttal_p17 as R17
import rebuttal_p25 as P25
import rebuttal_p26 as T

OUT = ROOT / "runs/analysis/rebuttal_20261013"
SEED = 20261013
WIDTHS = (192, 256)
N_PER_BIN, N_BINS, MIN_SHARED, E_SIZE = 48, 8, 12, 4
DOSES, MASKS = T.DOSES, T.MASKS
CONDS = ("intact", "full", "full_random", "lead", "lead_ctrl", "rem", "rem_ctrl", "comb_ctrl")
BASIS_SHA = {192: "d2a5e489e5bcda3be3e70e63e9da14beb22d3698a2eae2922785272a9730a941", 256: "420752a5d5baded38529c2196dab6d50c4403236d98d7728b00a6f52f130f88f"}
EPS_A, EPS_C = 1e-8, 1e-12
K, VB, N_BOOT = T.K, T.VB, 2000


def utc(): return datetime.now(timezone.utc).isoformat()


# ---------------- pure helpers (selftested) ----------------
def pool(width):
    """48 fresh puzzles per full-test rating octile (rank-based bins), excluding the study's 512, the P12/P17/P20/P22 pools, P25's and P26's pools."""
    import rebuttal_p12 as P12, rebuttal_p20 as R20, rebuttal_p22 as R22
    rec = np.load(ROOT / f"paper/code/evidence/benchmark/attention_{width}_d16.npz", allow_pickle=False)
    r128 = np.load(ROOT / "paper/code/evidence/benchmark/attention_128_d16.npz", allow_pickle=False)
    excl = P25.study_ids() | set(int(i) for i in P12.pool_ids(r128["cold_exact"], r128["idx"])) | set(int(i) for i in R17.pool(width)) \
        | set(int(i) for i in R20.pool(width)) | set(int(i) for i in R22.pool(width)) | set(int(i) for i in P25.pool(width)) | set(int(i) for i in T.pool(width))
    idx, rating = rec["idx"], rec["rating"]; order = np.lexsort((idx, rating)); bins = np.empty(len(idx), int); bins[order] = (np.arange(len(idx)) * N_BINS) // len(idx)
    out = []
    for b in range(N_BINS):
        cand = np.array(sorted(int(i) for i in idx[bins == b] if int(i) not in excl))
        out += list(np.random.default_rng([SEED, 27, width, b]).choice(cand, N_PER_BIN, replace=False))
    return np.sort(np.array(out, np.int64))


def load_basis(width, lm):
    f = T.basis_path(width); assert P25.sha(f) == BASIS_SHA[width], "gate: P26's frozen basis changed"
    U = np.array(json.loads(f.read_text())["U"], np.float64); vh = lm / np.linalg.norm(lm)
    assert U.shape == (width, width // 8) and np.allclose(U.T @ U, np.eye(U.shape[1]), atol=1e-8) and np.abs(U.T @ vh).max() < 1e-8
    return U


def proj_lead(U): return lambda x: (x @ U) @ U.T


def proj_rem(U, lm):
    vh = lm / np.linalg.norm(lm); return lambda x: x - (x @ vh)[..., None] * vh - (x @ U) @ U.T


def matched_control(hr, C, proj, rng):
    """Rows (n, w). xi inside C's subspace (proj) with ||xi|| = ||C|| and <hr, xi> = <hr, C> per row; the rest of its direction random, drawn once.
    Registered degenerate rows: ||C|| < EPS_C -> xi = 0; ||proj(hr)|| < EPS_A * ||hr|| -> no inner-product constraint (xi = ||C|| times a random unit
    vector of the subspace). The resulting row norm ||hr + xi|| equals ||hr + C||."""
    a = proj(hr); na = np.linalg.norm(a, axis=-1, keepdims=True); n = np.linalg.norm(C, axis=-1, keepdims=True); t = (hr * C).sum(-1, keepdims=True)
    deg_a = na[:, 0] < EPS_A * np.maximum(np.linalg.norm(hr, axis=-1), 1e-30); deg_c = n[:, 0] < EPS_C
    ahat = np.where(deg_a[:, None], 0.0, a / np.maximum(na, 1e-300))
    z = proj(rng.standard_normal(hr.shape)); g = z - (z * ahat).sum(-1, keepdims=True) * ahat; g = g / np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-300)
    cpar = np.where(deg_a[:, None], 0.0, t / np.maximum(na, 1e-300)); s = np.sqrt(np.maximum(n ** 2 - cpar ** 2, 0.0))
    xi = np.where(deg_c[:, None], 0.0, cpar * ahat + s * g)
    return xi, dict(deg_a=int(deg_a.sum()), deg_c=int(deg_c.sum()))


def centered_rows(Y, don, conds, pids, cx, cref):
    """All-four-mask statistic: within each (state, dose) the four masks' outcomes are centered; dependent c_X - c_ref, regressors c_donor and
    c_recipient (both intact). Four rows per (state, dose), in a fixed (state, dose, mask) order shared by every condition pair."""
    ix, jx, jr = conds.index("intact"), conds.index(cx), conds.index(cref); P_, A, B_, C_ = [], [], [], []
    for s in range(len(pids)):
        ds = don[s]
        if ds is None: continue
        for i in range(len(DOSES)):
            g = lambda st_, c: np.array([Y[(st_, c, i, m)] for m in range(MASKS)]); cen = lambda v: v - v.mean()
            dep = cen(g(s, jx)) - cen(g(s, jr)); dD = cen(g(ds, ix)); dR = cen(g(s, ix))
            P_ += [int(pids[s])] * MASKS; A += list(dep); B_ += list(dD); C_ += list(dR)
    return dict(pid=np.array(P_, np.int64), dep=np.array(A), dD=np.array(B_), dR=np.array(C_))


def beta(r):
    X = np.column_stack([r["dD"], r["dR"], np.ones(len(r["dep"]))]); return float(np.linalg.lstsq(X, r["dep"], rcond=None)[0][0])


def boot_arr(fn, lists, seed, n=N_BOOT):
    """Puzzle bootstrap (both directions and every challenge of a puzzle together), joint across row lists that share one row order."""
    pid = lists[0]["pid"]; assert all(np.array_equal(pid, l_["pid"]) for l_ in lists)
    up, inv = np.unique(pid, return_inverse=True); groups = [np.where(inv == g)[0] for g in range(len(up))]; rng = np.random.default_rng(seed); vals = []
    for _ in range(n):
        idx = np.concatenate([groups[g] for g in rng.integers(0, len(up), len(up))]); vals.append(fn(*[{k_: v[idx] for k_, v in l_.items()} for l_ in lists]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ---------------- the runner ----------------
class R21(T.R20p26):
    def __init__(self, width, out=OUT):
        super().__init__(width, out)
        self.meta.update(study="rebuttal_20261013", pipeline="tools/rebuttal_p27.py over tools/rebuttal_p26.py")
        self.ATS.write_json(self.dir / "runtime.json", self.meta)

    def pairs(self, smoke=False):
        """All eligible aligned pairs (P26's construction with P27's pool and seeds)."""
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

    def condition_states(self, pr, di, U):
        """The recipient's slow state under each of the eight conditions (direction di: 0 = A <- B, 1 = B <- A); every change readout-orthogonal."""
        lm = self.lm; r, d = (0, 1) if di == 0 else (1, 0); shp = pr["h"][r].shape
        hr = pr["h"][r].astype(np.float64); hd = pr["h"][d].astype(np.float64); par_r, perp_r = P.project(hr, lm); perp_d = P.project(hd, lm)[1]
        D = perp_d - perp_r; L = (D @ U) @ U.T; M = D - L; fl = lambda x: x.reshape(-1, shp[-1])
        xiL, dgL = matched_control(fl(hr), fl(L), proj_lead(U), np.random.default_rng([SEED, pr["pid"], 8, di]))
        xiM, dgM = matched_control(fl(hr), fl(M), proj_rem(U, lm), np.random.default_rng([SEED, pr["pid"], 9, di]))
        xiL, xiM = xiL.reshape(shp), xiM.reshape(shp); vh = lm / np.linalg.norm(lm)
        newp, _ = P25.rotate_matched(perp_r, perp_d, lm, np.random.default_rng([SEED, pr["pid"], 6, di]))
        rel = lambda a, b: float((np.abs(a - b) / np.maximum(np.abs(b), 1e-12)).max())
        nrm = lambda x: np.linalg.norm(x, axis=-1); ip = lambda x: (hr * x).sum(-1)
        ok = lambda C: nrm(C) > 1e-9 * np.maximum(nrm(hr), 1e-30)                                     # rows where relative checks are meaningful
        info = dict(energy_lead=float((L ** 2).sum() / max((D ** 2).sum(), 1e-30)), energy_rem=float((M ** 2).sum() / max((D ** 2).sum(), 1e-30)),
                    split_dev=float(np.abs(L + M - D).max()), rem_in_U=float(np.abs(M @ U).max()), rem_in_lm=float(np.abs(M @ vh).max()),
                    xiM_in_U=float(np.abs(xiM @ U).max()), xiM_in_lm=float(np.abs(xiM @ vh).max()), xiL_out_U=float(np.abs(xiL - (xiL @ U) @ U.T).max()),
                    norm_dev=max(rel(nrm(xiL)[ok(L)], nrm(L)[ok(L)]), rel(nrm(xiM)[ok(M)], nrm(M)[ok(M)])),
                    ip_dev=max(float(np.abs(ip(xiL) - ip(L)).max() / max(np.abs(ip(L)).max(), 1e-30)), float(np.abs(ip(xiM) - ip(M)).max() / max(np.abs(ip(M)).max(), 1e-30))),
                    rownorm_dev=max(rel(nrm(hr + xiL), nrm(hr + L)), rel(nrm(hr + xiM), nrm(hr + M)), rel(nrm(hr + xiL + xiM), nrm(hr + D))),
                    deg=dict(L=dgL, M=dgM))
        f32 = lambda x: x.astype(np.float32)
        hs = dict(intact=pr["h"][r], full=f32(hr + D), full_random=f32(par_r + newp), lead=f32(hr + L), lead_ctrl=f32(hr + xiL),
                  rem=f32(hr + M), rem_ctrl=f32(hr + xiM), comb_ctrl=f32(hr + xiL + xiM))
        return hs, info

    def run(self, smoke=False):
        try:
            lm = self.lm; prs, xcan = self.pairs(smoke); U = load_basis(self.width, lm)
            gates = dict(display=0, e_edited=0, outside_shared=0, score_shift=0.0, restore_tensor=0, restore_rollout=0, restore_maxdev=0.0,
                         split_dev=0.0, rem_in_U=0.0, rem_in_lm=0.0, xiM_in_U=0.0, xiM_in_lm=0.0, xiL_out_U=0.0, norm_dev=0.0, ip_dev=0.0, rownorm_dev=0.0, deg_rows=0)
            S, V, H, RST, info_all = [], [], {}, [], []
            for pr in prs:
                for c in pr["masks"]: gates["outside_shared"] += int(not set(c[2].tolist()) <= set(pr["Wshared"].tolist()) or bool(set(c[2].tolist()) & set(pr["E"].tolist())))
                for di in (0, 1):
                    r = 0 if di == 0 else 1; hs, info = self.condition_states(pr, di, U); info_all.append(info)
                    for k_ in ("split_dev", "rem_in_U", "rem_in_lm", "xiM_in_U", "xiM_in_lm", "xiL_out_U", "norm_dev", "ip_dev", "rownorm_dev"): gates[k_] = max(gates[k_], info[k_])
                    gates["deg_rows"] += sum(v for dd in info["deg"].values() for v in dd.values())
                    s0 = np.einsum("fsw,w->sf", pr["h"][r].astype(np.float64), lm)
                    for c, h in hs.items():
                        if c != "intact": gates["score_shift"] = max(gates["score_shift"], float(np.abs(np.einsum("fsw,w->sf", h.astype(np.float64), lm) - s0).max()))
                    si = len(S); H[si] = dict(hs=hs, l=pr["l"][r], grid=pr["grid"][r], sol=pr["sol"], E=pr["E"], saved=(pr["h"][r].copy(), pr["l"][r].copy()))
                    S.append(dict(pid=pr["pid"], dir=di, b=pr["b"], E=pr["E"], sol=pr["sol"].astype(np.int8), grid=pr["grid"][r].astype(np.int8), nwrong=len(pr["W"][r]), nwrong_donor=len(pr["W"][1 - r])))
                    for ci, c in enumerate(CONDS):
                        V.append(dict(si=si, b=pr["b"], cond=ci, arm=0, dose=-1, mask=-1, cells=np.array([], int)))
                        for i, m, cells in pr["masks"]: V.append(dict(si=si, b=pr["b"], cond=ci, arm=1, dose=i, mask=m, cells=cells))
                    RST.append((si, len(V) - len(CONDS) * (1 + len(pr["masks"]))))
            nmain = len(V)
            for si, v0 in RST:                              # exact restoration: the saved recipient tensors written back after every other variant is built
                for j in [v0] + [v0 + 1 + i * MASKS for i in range(len(DOSES))]:
                    assert V[j]["si"] == si and V[j]["cond"] == 0 and (j == v0 or V[j]["mask"] == 0)
                    V.append(dict(si=si, b=V[j]["b"], cond=-1, arm=V[j]["arm"], dose=V[j]["dose"], mask=V[j]["mask"], cells=V[j]["cells"], twin=j))
            def build(sp):
                Hh = H[sp["si"]]
                if sp["cond"] == -1:
                    h, l = Hh["saved"]; gates["restore_tensor"] += int(not (np.array_equal(h, Hh["hs"]["intact"]) and np.array_equal(l, Hh["l"])))
                else: h, l = Hh["hs"][CONDS[sp["cond"]]], Hh["l"]
                cells = sp["cells"]
                if np.isin(cells, Hh["E"]).any(): gates["e_edited"] += 1
                if len(cells) == 0: return h, l
                for c in cells: h, l = R16.cell_relabel(h, l, int(c), int(Hh["grid"][c]), int(Hh["sol"][c]))
                want = Hh["grid"].astype(int).copy(); want[cells] = Hh["sol"][cells]
                if not np.array_equal(np.einsum("fsw,w->sf", h, self.lm32).argmax(1) + 1, want): gates["display"] += 1
                return h, l
            cnt = [0]; sc = self.roll_variants(V, xcan, None, build, "p27", cnt)
            for q in range(nmain, len(V)):
                j = V[q]["twin"]; gates["restore_rollout"] += int(not np.array_equal(sc[q], sc[j])); gates["restore_maxdev"] = max(gates["restore_maxdev"], float(np.abs(sc[q] - sc[j]).max()))
            V, sc = V[:nmain], sc[:nmain]
            assert gates["restore_tensor"] == 0 and gates["restore_rollout"] == 0, f"gate: {gates}"
            assert gates["display"] == 0 and gates["e_edited"] == 0 and gates["outside_shared"] == 0 and gates["score_shift"] < 1e-3, f"gate: {gates}"
            assert gates["split_dev"] < 1e-12 and max(gates["rem_in_U"], gates["rem_in_lm"], gates["xiM_in_U"], gates["xiM_in_lm"], gates["xiL_out_U"]) < 1e-9, f"gate: {gates}"
            assert gates["norm_dev"] < 1e-6 and gates["ip_dev"] < 1e-6 and gates["rownorm_dev"] < 1e-6, f"gate: {gates}"
            Earr = np.stack([s_["E"] for s_ in S]); vs = np.array([v["si"] for v in V])
            esc = np.stack([sc[i][:, Earr[vs[i]]] for i in range(len(V))]).astype(np.float32)
            disp_e = np.stack([(sc[i][:, Earr[vs[i]]].argmax(-1) + 1) for i in range(len(V))]).astype(np.int8)
            cells = np.full((len(V), 81), -1, np.int16)
            for i, v in enumerate(V): cells[i, :len(v["cells"])] = v["cells"]
            dst = self.dir / ("p27_smoke.npz" if smoke else "p27.npz")
            np.savez_compressed(dst, conds=np.array(CONDS), var_state=vs.astype(np.int32), var_cond=np.array([v["cond"] for v in V], np.int8), var_arm=np.array([v["arm"] for v in V], np.int8),
                                var_dose=np.array([v["dose"] for v in V], np.int8), var_mask=np.array([v["mask"] for v in V], np.int8), var_cells=cells, var_nedit=np.array([len(v["cells"]) for v in V], np.int16),
                                escores=esc, disp_e=disp_e, state_pid=np.array([s_["pid"] for s_ in S], np.int64), state_dir=np.array([s_["dir"] for s_ in S], np.int8), state_E=Earr.astype(np.int16),
                                state_sol=np.stack([s_["sol"] for s_ in S]), state_grid=np.stack([s_["grid"] for s_ in S]), state_nwrong=np.array([s_["nwrong"] for s_ in S], np.int16),
                                state_nwrong_donor=np.array([s_["nwrong_donor"] for s_ in S], np.int16), info=json.dumps(info_all), gates=json.dumps(gates),
                                meta=json.dumps(dict(self.meta, created=utc(), smoke=smoke, eligible_pairs=len(prs), basis_sha256=BASIS_SHA[self.width])))
            if not smoke: self.ATS.write_json(dst.with_suffix(".sha256.json"), dict(sha256=self.ATS.sha(dst)))
            print(json.dumps(dict(width=self.width, eligible_pairs=len(prs), recipients=len(S), variants=len(V), gates=gates)))
            self.ATS.verify_sources(); self.status("inference_complete", completed=utc()); self.ATS.log("model_complete", width=self.width, group="p27")
        except BaseException as exc:
            self.status("failed", error=repr(exc)); raise


# ---------------- the frozen reader ----------------
def report_width(width, base=OUT, smoke=False):
    f = base / f"attention_{width}/p27{'_smoke' if smoke else ''}.npz"
    if not f.exists(): return None
    d = np.load(f, allow_pickle=False); conds = [str(x) for x in d["conds"]]; Y, Bm = T.unit_tables(d); don = T.donor_state(d); pids = d["state_pid"]
    rw = lambda cx, cref: centered_rows(Y, don, conds, pids, cx, cref); bs = lambda fn, rl, j: boot_arr(fn, rl, [SEED, 27, width, j]); bt = lambda fn, rl, j: T.boot(fn, rl, [SEED, 27, width, j])
    res = dict(meta={k_: json.loads(str(d["meta"]))[k_] for k_ in ("eligible_pairs",)}, gates=json.loads(str(d["gates"])), states=int(len(pids)))
    res["beta"] = {c: beta(rw(c, "intact")) for c in conds[1:]}
    r_rep = rw("full", "full_random"); res["S"] = dict(E=beta(r_rep), ci=bs(beta, [r_rep], 1)); res["S"]["letter"] = "CHALLENGE-SPECIFIC-REPLICATES" if res["S"]["ci"][0] > 0 else "NOT-RESOLVED"
    r2 = T.r2_rows(d, "full", "full_random"); res["S_two_contrast"] = dict(E=T.r2_beta(r2), ci=bt(T.r2_beta, [r2], 2))
    rL, rM, rT = rw("lead", "lead_ctrl"), rw("rem", "rem_ctrl"), rw("full", "comb_ctrl")
    res["E_L"] = dict(E=beta(rL), ci=bs(beta, [rL], 3)); res["E_M"] = dict(E=beta(rM), ci=bs(beta, [rM], 4)); res["E_total"] = dict(E=beta(rT), ci=bs(beta, [rT], 5))
    half = lambda a, t: beta(a) - 0.5 * beta(t)
    res["half_L"] = dict(E=half(rL, rT), ci=bs(half, [rL, rT], 6)); res["half_M"] = dict(E=half(rM, rT), ci=bs(half, [rM, rT], 7))
    rDL, rDM = rw("full", "lead"), rw("full", "rem"); res["direct_full_minus_L"] = dict(E=beta(rDL), ci=bs(beta, [rDL], 8)); res["direct_full_minus_M"] = dict(E=beta(rDM), ci=bs(beta, [rDM], 9))
    rF, rLi, rMi, rC, rXL, rXM = rw("full", "intact"), rw("lead", "intact"), rw("rem", "intact"), rw("comb_ctrl", "intact"), rw("lead_ctrl", "intact"), rw("rem_ctrl", "intact")
    isig = lambda a, b, c: beta(a) - beta(b) - beta(c); dI = lambda a, b, c, e, f, g: isig(a, b, c) - isig(e, f, g)
    res["I_signal"] = dict(E=isig(rF, rLi, rMi), ci=bs(isig, [rF, rLi, rMi], 10)); res["I_control"] = dict(E=isig(rC, rXL, rXM), ci=bs(isig, [rC, rXL, rXM], 11))
    res["I_diff"] = dict(E=dI(rF, rLi, rMi, rC, rXL, rXM), ci=bs(dI, [rF, rLi, rMi, rC, rXL, rXM], 12))
    tot = res["E_total"]["ci"][0] > 0
    res["flags"] = dict(TOTAL="TOTAL-RESOLVED" if tot else "TOTAL-NOT-RESOLVED",
                        LEADING_CARRIES=res["E_L"]["ci"][0] > 0, REMAINDER_CARRIES=res["E_M"]["ci"][0] > 0,
                        LEADING_HALF=(res["half_L"]["ci"][0] > 0) if tot else "n/a", REMAINDER_HALF=(res["half_M"]["ci"][0] > 0) if tot else "n/a",
                        INTERACTION="POSITIVE" if res["I_diff"]["ci"][0] > 0 else ("NEGATIVE" if res["I_diff"]["ci"][1] < 0 else "UNRESOLVED"))
    fl = res["flags"]
    if fl["INTERACTION"] == "POSITIVE" and not fl["LEADING_CARRIES"] and not fl["REMAINDER_CARRIES"]:
        res["description"] = "positive interaction beyond the random interaction; individual component effects unresolved"
    bl = T.baseline_rows(d); eb = lambda r: T.baseline_slope(r, "full") - T.baseline_slope(r, "full_random"); res["BASELINE"] = dict(E=eb(bl), ci=bt(eb, [bl], 13))
    r1 = T.r1_rows(d, "full", "full_random"); e1 = lambda r: T.r1_beta(r, "x") - T.r1_beta(r, "ref"); res["R1"] = dict(E=e1(r1), ci=bt(e1, [r1], 14))
    inf = json.loads(str(d["info"])); res["energy"] = dict(lead=float(np.mean([x["energy_lead"] for x in inf])), rem=float(np.mean([x["energy_rem"] for x in inf])))
    return res


def report(base=OUT, smoke=False, quiet=False):
    lines = ["P27 REPORT (rules registered before any row; primary = the all-four-mask centered donor coefficient at k = 1)"]; js = {}
    fmt = lambda x: f"{x['E']:+.3f} [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}]"
    for w in WIDTHS:
        r = report_width(w, base, smoke)
        if r is None: lines.append(f"attention_{w}: not run"); continue
        js[f"attention_{w}"] = r
        lines += [f"attention_{w}: pairs {r['meta']['eligible_pairs']} (states {r['states']}) | gates {r['gates']}",
                  f"   per-condition donor coefficients: " + ", ".join(f"{c} {v:+.3f}" for c, v in r["beta"].items()),
                  f"   S (full vs P26's control, four masks) {fmt(r['S'])} -> {r['S']['letter']} | two-contrast {fmt(r['S_two_contrast'])}",
                  f"   E_L {fmt(r['E_L'])} | E_M {fmt(r['E_M'])} | E_total (vs combined control) {fmt(r['E_total'])}",
                  f"   half: E_L - E_total/2 {fmt(r['half_L'])} | E_M - E_total/2 {fmt(r['half_M'])}",
                  f"   direct: full - L {fmt(r['direct_full_minus_L'])} | full - M {fmt(r['direct_full_minus_M'])}",
                  f"   interaction: I_signal {fmt(r['I_signal'])} | I_control {fmt(r['I_control'])} | difference {fmt(r['I_diff'])}",
                  f"   flags {r['flags']}" + (f" | {r['description']}" if "description" in r else ""),
                  f"   secondary: BASELINE {fmt(r['BASELINE'])} | R1 {fmt(r['R1'])} | energy {r['energy']}"]
    have = [w for w in WIDTHS if f"attention_{w}" in js]
    if len(have) == 2:
        lines.append(f"CHALLENGE-SPECIFIC-REPLICATES on both receivers: {all(js[f'attention_{w}']['S']['letter'] == 'CHALLENGE-SPECIFIC-REPLICATES' for w in have)}")
        for k_ in ("LEADING_CARRIES", "REMAINDER_CARRIES", "LEADING_HALF", "REMAINDER_HALF"):
            lines.append(f"{k_} on both receivers: {all(js[f'attention_{w}']['flags'][k_] is True for w in have)}")
        lines.append("INTERACTION: " + ", ".join(f"{w}: {js[f'attention_{w}']['flags']['INTERACTION']}" for w in have))
    sfx = "_smoke" if smoke else ""
    (base / f"report{sfx}.txt").write_text("\n".join(lines) + "\n"); (base / f"report{sfx}.json").write_text(json.dumps(js, indent=1, default=float))
    print("\n".join(lines) if not quiet else f"report written ({len(lines)} lines; output suppressed)")


def selftest():
    rng = np.random.default_rng(0); w, k = 40, 5; lm = rng.standard_normal(w); vh = lm / np.linalg.norm(lm)
    U = T.random_basis(w, k, lm, rng); pL, pM = proj_lead(U), proj_rem(U, lm)
    hr = rng.standard_normal((300, w)); D = pM(rng.standard_normal((300, w))) + pL(rng.standard_normal((300, w))); L = pL(D); M = D - L
    assert np.abs(L + M - D).max() < 1e-12 and np.abs(M @ U).max() < 1e-10 and np.abs(M @ vh).max() < 1e-10
    for C, pj in ((L, pL), (M, pM)):
        xi, dg = matched_control(hr, C, pj, np.random.default_rng(1))
        assert np.allclose(np.linalg.norm(xi, axis=-1), np.linalg.norm(C, axis=-1), rtol=1e-9) and np.allclose((hr * xi).sum(-1), (hr * C).sum(-1), atol=1e-9)
        assert np.abs(xi - pj(xi)).max() < 1e-10 and np.allclose(np.linalg.norm(hr + xi, axis=-1), np.linalg.norm(hr + C, axis=-1), rtol=1e-9)
        assert np.abs((xi * C).sum(-1) / (np.linalg.norm(xi, axis=-1) * np.linalg.norm(C, axis=-1))).mean() < 0.9            # directions not copied
        xi2, _ = matched_control(hr, C, pj, np.random.default_rng(1)); assert np.array_equal(xi, xi2)                       # drawn once, reproducible
    Cz = L.copy(); Cz[0] = 0.0; xi, dg = matched_control(hr, Cz, pL, np.random.default_rng(2)); assert dg["deg_c"] == 1 and np.all(xi[0] == 0)
    hz = hr.copy(); hz[1] = pM(hz[1:2])[0]; xi, dg = matched_control(hz, L, pL, np.random.default_rng(3)); assert dg["deg_a"] >= 1 and np.isclose(np.linalg.norm(xi[1]), np.linalg.norm(L[1]))
    # the centered statistic recovers a planted donor coefficient
    pids = np.repeat(np.arange(60), 2); don = {s: s + 1 if s % 2 == 0 else s - 1 for s in range(120)}; conds = ["intact", "full"]; Y = {}
    for s in range(120):
        for i in range(3):
            for m in range(4): Y[(s, 0, i, m)] = rng.normal()
    for s in range(120):
        for i in range(3):
            for m in range(4): Y[(s, 1, i, m)] = Y[(s, 0, i, m)] + 0.3 * Y[(don[s], 0, i, m)] + 0.01 * rng.normal()
    assert abs(beta(centered_rows(Y, don, conds, pids, "full", "intact")) - 0.3) < 0.02
    p = pool(192); assert len(p) == N_PER_BIN * N_BINS and np.array_equal(p, pool(192)) and not (set(int(i) for i in p) & set(int(i) for i in T.pool(192)))
    assert P25.sha(T.basis_path(192)) == BASIS_SHA[192] and P25.sha(T.basis_path(256)) == BASIS_SHA[256]
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", nargs="?", choices=["run", "report"]); ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--width", type=int, choices=WIDTHS); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--quiet", action="store_true"); ap.add_argument("--out", default=None)
    a = ap.parse_args(); out = Path(a.out) if a.out else OUT
    if a.selftest: selftest(); return
    if a.action == "report": report(out, a.smoke, a.quiet); return
    if a.action == "run":
        if a.width is None: ap.error("--width required")
        R21(a.width, out).run(a.smoke)
    else: ap.print_help()


if __name__ == "__main__":
    main()
