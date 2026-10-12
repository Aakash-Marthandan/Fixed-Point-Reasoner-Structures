#!/usr/bin/env python3
# Ledger: THE DISCUSSION-PERIOD EXPERIMENT P29 — does a score-preserving retained-state transplant transfer differential responses to
# identical corrections on ARC? (the public TRM ARC-AGI-1 checkpoint in its own vendored code; registration
# Documentation/Note_2026-10-12_Rebuttal_P29_Registration_DRAFT.md). MEASUREMENT, $0, inference only on the Mac's CPU (float32).
"""P29: the response-transfer test of P26/P27 on a second task and architecture (TRM, ARC-AGI-1 evaluation queries).

Pairs: per query, history A = the fixed start (H_init, L_init) after one update; history B = a Gaussian start (std 1) after one update. Kept when
both are unsolved and share enough COLOUR-TO-COLOUR wrong tokens (inside the target grid, both histories display a colour, both wrong).
Aligned challenges: never-edited evaluation tokens E and 4 equal-size masks per dose from the remaining shared tokens, identical for both histories.
A correction at a token swaps the recipient's displayed-colour and target-colour logits, realized by the minimum-norm change of the slow state zH
(the readout's pseudoinverse); it is built ONCE per (recipient, mask) from the recipient's intact state and added unchanged to its intact,
transplanted and control states. Conditions: intact; full = the donor's readout-invisible zH (the 500-dimensional nullspace component of every
one of the 900 answer positions, padding included, and the whole vectors of the 16 prefix positions), the recipient's fast state zL and inputs
kept; full_ctrl = a per-token angle- and norm-matched rotation of the recipient's invisible part (drawn once per pair-direction). Fixed-depth
continuation through the adapter's update (verified against the native inner recurrence); the halting head does not act.

Phases: `pilot` (separate tasks; intact rollouts only; every gate; timing), `pilot-report` (intact-only diagnostics that set the frozen
measurement), `run` (confirmation; sharded by query blocks, checkpointed, resumable), `merge`, `report` (the frozen reader). Pure helpers are
numpy-only so the reader and the selftest run in either environment; torch is imported by the runner only."""
import argparse, hashlib, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "runs/field_ckpts"
CKPT = ASSETS / "ckpts/trm_arc1/step_259320_arc_ag1_attn_type_h3l4"
ADAPTER = ROOT / "paper/code/adapters/arc"
OUT = ROOT / "runs/analysis/rebuttal_20261015"
SEED = 20261015
N_PILOT_TASKS = 48
D_INT, K_MAX = 1, 4
DOSES, MASKS = (0.25, 0.5, 0.75), 4
E_CAND, R_MIN = (4, 8, 16), 8              # pilot: evaluation-set sizes compared (nested); eligibility |W| >= |E| + R_MIN
BLOCK, VB = 8, 8                           # queries per checkpointed block; variants per forward batch
P_PREFIX, HID, NPOS, VOCAB = 16, 512, 900, 12
CLUSTER_FACTOR, MDE_Z = 1.5, 2.8           # pilot design proxy: bootstrap SE ~ 1.5 x proxy SE (P27 calibration); MDE = 2.8 SE (80 %, two-sided 5 %)
N_BOOT = 20000
# Frozen by the pilot (Documentation/Note_2026-10-12_Rebuttal_P29_Registration.md, "Pilot outcome and frozen settings"); the run refuses while unset.
TOL_TRANSPLANT, TOL_CORRECTION = 3e-3, 1.5e-3   # the PI's second review: executed float32 logit caps (pilot maxima 1.59e-3 and 7.1e-4)
FROZEN = dict(E=4, measure="margin_std", tol_transplant=TOL_TRANSPLANT, tol_correction=TOL_CORRECTION)   # option B; the calibration's frozen rule: LAUNCH
SCALE_MIN = 0.01                                # option B: a query whose pooled intact margin-response RMS is below this (logits) is excluded, counted


def utc(): return datetime.now(timezone.utc).isoformat()


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


# ---------------- pure helpers (selftested) ----------------
def task_split(tasks):
    """Frozen seeded order of the task names; the first N_PILOT_TASKS are the feasibility pilot, never used for confirmation."""
    t = sorted(set(tasks)); perm = np.random.default_rng([SEED, 29, 0]).permutation(len(t)); o = [t[i] for i in perm]
    return o[:N_PILOT_TASKS], o[N_PILOT_TASKS:]


def colour(x): return (x >= 2) & (x <= 11)


def shared_colour_wrong(disp_a, disp_b, labels):
    """Positions inside the target grid (label a colour) where BOTH histories display a colour and both are wrong."""
    m = colour(labels) & colour(disp_a) & colour(disp_b) & (disp_a != labels) & (disp_b != labels)
    return np.where(m)[0]


def structural_errors(disp, labels):
    """In-grid positions displaying a non-colour token, and out-of-grid positions (EOS/padding labels) displaying a colour."""
    return int((colour(labels) & ~colour(disp)).sum()), int((~colour(labels) & colour(disp)).sum())


def draw_challenges(qi, W, e_excl, e_eval):
    """E = the first e_eval tokens of a seeded permutation of W (never edited); masks from W minus its first e_excl tokens (e_excl >= max e_eval)."""
    perm = np.random.default_rng([SEED, int(qi), 1]).permutation(np.asarray(W)); Ex = perm[:e_excl]; R = np.array(sorted(set(W.tolist()) - set(Ex.tolist())))
    masks = [(i, m, np.sort(np.random.default_rng([SEED, int(qi), 2, i, m]).choice(R, max(1, int(round(f * len(R)))), replace=False)))
             for i, f in enumerate(DOSES) for m in range(MASKS)]
    return perm[:e_eval], masks


def nullspace_projector(W):
    """W (vocab, hidden) float64 of full row rank -> (N = I - W^+ W, Wp = W^+), N symmetric idempotent with W N = 0."""
    Wp = np.linalg.pinv(W); N = np.eye(W.shape[1]) - Wp @ W; return 0.5 * (N + N.T), Wp


def rotate_matched(pr, pd, proj, rng):
    """Rows: a random direction inside the subspace `proj` at the same angle from pr as pd is, scaled to |pd| (P25/P26's matched rotation)."""
    nr = np.linalg.norm(pr, axis=-1, keepdims=True); nd = np.linalg.norm(pd, axis=-1, keepdims=True); ur = pr / np.maximum(nr, 1e-300)
    cos = np.clip((pr * pd).sum(-1, keepdims=True) / np.maximum(nr * nd, 1e-300), -1, 1); th = np.arccos(cos)
    g = proj(rng.standard_normal(pr.shape)); g = g - (g * ur).sum(-1, keepdims=True) * ur; g = g / np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-300)
    return nd * (np.cos(th) * ur + np.sin(th) * g)


def transplant_states(hr, hd, N, P, rng):
    """float64 (S, H) slow states of recipient and donor -> (full, ctrl, info). full: answer rows hr - hr N + hd N, prefix rows = hd; ctrl: answer
    rows hr - hr N + rot(hr N -> |hd N| at the donor's angle, inside the nullspace), prefix rows rot(hr -> |hd| at the donor's angle, full space)."""
    pr, pd = hr[P:] @ N, hd[P:] @ N
    full = hr.copy(); full[P:] = hr[P:] - pr + pd; full[:P] = hd[:P]
    ctrl = hr.copy(); newp = rotate_matched(pr, pd, lambda x: x @ N, rng); ctrl[P:] = hr[P:] - pr + newp
    ctrl[:P] = rotate_matched(hr[:P], hd[:P], lambda x: x, rng)
    nrm = lambda x: np.linalg.norm(x, axis=-1)
    info = dict(rel_change_answer=float(nrm(pd - pr).mean() / nrm(hr[P:]).mean()), rel_change_prefix=float(nrm(hd[:P] - hr[:P]).mean() / nrm(hr[:P]).mean()),
                ctrl_norm_dev=float(np.abs(nrm(newp) - nrm(pd)).max() / nrm(pd).max()), ctrl_change_dev=float(np.abs(nrm(newp - pr) - nrm(pd - pr)).max() / nrm(pd - pr).max()),
                ctrl_in_null=float(np.abs(newp - newp @ N).max()))
    return full, ctrl, info


def correction_delta(h_tokens, W, Wp, labels_tok):
    """float64 answer rows (n, H) at the masked tokens -> the minimum-norm change realizing the swap of each token's displayed and target logits."""
    lg = h_tokens @ W.T; pred = lg.argmax(-1); tgt = lg.copy(); j = np.arange(len(lg))
    tgt[j, labels_tok] = lg[j, pred]; tgt[j, pred] = lg[j, labels_tok]
    return (tgt - lg) @ Wp.T, tgt, lg


def crop(tokens):
    """TRM's decoder: the maximal top-left rectangle free of tokens outside [2, 11], shifted to colours 0..9."""
    g = np.asarray(tokens).reshape(30, 30); max_area = max_r = max_c = 0; num_c = 30
    for r in range(1, 31):
        for c in range(1, num_c + 1):
            if not (2 <= g[r - 1, c - 1] <= 11): num_c = c - 1; break
        if r * num_c > max_area: max_area, max_r, max_c = r * num_c, r, num_c
    return (g[:max_r, :max_c] - 2).astype(np.uint8)


def margins(elog, labels_e):
    """E logits (..., n, 12) and labels (n,) -> correct-token margin: label logit minus the largest other logit."""
    lab = np.take_along_axis(elog, np.broadcast_to(labels_e[..., None], elog.shape[:-1] + (1,)), -1)[..., 0]
    other = np.where(np.arange(elog.shape[-1]) == labels_e[..., None], -np.inf, elog).max(-1)
    return lab - other


# ---- the statistic (P27's all-four-mask centered donor coefficient), with task-clustered bootstrap ----
def centered_units(Y, don, cx, cref, ix=0):
    """Y[(state, cond, dose, mask)] -> rows of (dep, cD, cR) per (state, dose, mask): within each (state, dose) the four masks are centered;
    dep = c_cx - c_cref (recipient), cD = c_intact(donor), cR = c_intact(recipient). Rows in (state, dose, mask) order."""
    dep, cD, cR, st = [], [], [], []
    for s in sorted(don):
        ds = don[s]
        if ds is None: continue
        for i in range(len(DOSES)):
            g = lambda s_, c: np.array([Y[(s_, c, i, m)] for m in range(MASKS)]); cen = lambda v: v - v.mean()
            dep += list(cen(g(s, cx)) - cen(g(s, cref))); cD += list(cen(g(ds, ix))); cR += list(cen(g(s, ix))); st += [s] * MASKS
    return dict(dep=np.array(dep), cD=np.array(cD), cR=np.array(cR), state=np.array(st, np.int64))


ID_TOL = 1e-9                              # the donor column's residual (after cR and the intercept) must keep > ID_TOL of its own sum of squares


def donor_identified(XX):
    """XX = X'X for columns (cD, cR, 1): the donor coefficient is identified iff cD has a residual after projecting out (cR, 1), relative to its
    own sum of squares, above ID_TOL (and cD is not identically zero)."""
    a = XX[0, 0]
    if not a > 1e-300: return False
    B = XX[1:, 1:]; c = XX[1:, 0]; res = a - c @ np.linalg.pinv(B) @ c
    return bool(res > ID_TOL * a)


def beta(u):
    """The donor coefficient; NaN when it is not identified (flat donor responses, or donor responses collinear with the recipient's)."""
    X = np.column_stack([u["cD"], u["cR"], np.ones(len(u["dep"]))])
    if len(u["dep"]) == 0 or not donor_identified(X.T @ X): return float("nan")
    return float(np.linalg.lstsq(X, u["dep"], rcond=None)[0][0])


def cluster_stats(u, clusters):
    """Per-cluster sufficient statistics (X'X, X'y) for the donor-coefficient regression -> (G, 3, 3), (G, 3) with clusters' order."""
    X = np.column_stack([u["cD"], u["cR"], np.ones(len(u["dep"]))]); y = u["dep"]; cl = np.unique(clusters)
    XX = np.stack([X[clusters == c].T @ X[clusters == c] for c in cl]); Xy = np.stack([X[clusters == c].T @ y[clusters == c] for c in cl]); return XX, Xy


def boot_betas(unit_lists, signs, clusters_list, seed, n=N_BOOT):
    """Task-clustered bootstrap of sum_j signs[j] * beta(unit_lists[j]); the lists share one cluster order (joint resampling). Returns draws;
    a draw in which any coefficient is not identified is NaN (registered: counted, and placed outside both ends of the interval by ci())."""
    cls = [np.unique(c) for c in clusters_list]; assert all(np.array_equal(cls[0], c) for c in cls)
    st = [cluster_stats(u, c) for u, c in zip(unit_lists, clusters_list)]; G = len(cls[0]); rng = np.random.default_rng(seed); out = np.empty(n)
    for b in range(n):
        w = np.bincount(rng.integers(0, G, G), minlength=G).astype(np.float64); v = 0.0
        for sg, (XX, Xy) in zip(signs, st):
            A = np.tensordot(w, XX, 1)
            if not donor_identified(A): v = float("nan"); break
            v += sg * np.linalg.lstsq(A, np.tensordot(w, Xy, 1), rcond=None)[0][0]
        out[b] = v
    return out


def ci(draws):
    """95 % percentile interval; unidentified (NaN) draws count as below every value for the lower bound and above every value for the upper,
    so they can only widen the interval (all draws NaN -> (-inf, +inf))."""
    d = np.asarray(draws, float); lo = np.sort(np.where(np.isnan(d), -np.inf, d)); hi = np.sort(np.where(np.isnan(d), np.inf, d))

    def pct(x, q, side):                                   # numpy's linear percentile for finite neighbours; the conservative neighbour otherwise
        pos = q / 100 * (len(x) - 1); i = int(np.floor(pos)); f = pos - i; a, b_ = x[i], x[min(i + 1, len(x) - 1)]
        if f == 0 or a == b_: return float(a)
        if not (np.isfinite(a) and np.isfinite(b_)): return float(a if side == "lo" else b_)
        return float(a + f * (b_ - a))
    return pct(lo, 2.5, "lo"), pct(hi, 97.5, "hi")


def n_unidentified(draws): return int(np.isnan(np.asarray(draws, float)).sum())


def proxy_se(cD, cR):
    """The Sudoku-calibrated planning proxy sqrt(2 mean(cR^2) / sum(rD^2)); None (UNINFORMATIVE) when the responses are flat or the donor's
    responses have no variation beyond the recipient's. A planning heuristic, not an ARC power guarantee: with both directions included it is
    nearly scale-free and mostly reflects the row count and the donor-recipient correlation."""
    n = len(cD)
    if n == 0: return None, None
    X = np.column_stack([cD, cR, np.ones(n)])
    if not (cR @ cR > 1e-300) or not donor_identified(X.T @ X): return None, None
    Xr = np.column_stack([cR, np.ones(n)]); rD = cD - Xr @ np.linalg.lstsq(Xr, cD, rcond=None)[0]
    return float(np.sqrt(2 * np.mean(cR ** 2) / (rD @ rD))), rD


def design_diagnostics(Y, don, task_of_state, ix=0):
    """Intact-only diagnostics: varying-unit share; R^2 of the donor's centered responses on the recipient's; the planning proxy (None =
    UNINFORMATIVE); task concentration of the residual donor information sum(rD^2) (tasks with any, the largest task's share, the effective
    number 1 / sum(share^2)); the proxy with each task omitted."""
    u = centered_units(Y, don, ix, ix, ix); cD, cR = u["cD"], u["cR"]; n = len(cD)
    if n == 0: return None
    tk = np.array([task_of_state[s_] for s_ in u["state"]]); vary = float(np.mean([np.ptp(cR[i:i + MASKS]) > 1e-9 for i in range(0, n, MASKS)]))
    se, rD = proxy_se(cD, cR); out = dict(units=n // MASKS, tasks=int(len(np.unique(tk))), vary_share=vary, mean_var=float(np.mean(cR ** 2)),
                                         R2=float(1 - (rD @ rD) / max(cD @ cD, 1e-300)) if rD is not None else None, se_proxy=se, informative=se is not None)
    if se is None: return out
    tot = rD @ rD; sh = {t: float((rD[tk == t] ** 2).sum() / tot) for t in np.unique(tk)}; shv = np.array(sorted(sh.values(), reverse=True))
    top = max(sh, key=sh.get); loo = {}
    for t in np.unique(tk):
        m = tk != t; loo[t] = proxy_se(cD[m], cR[m])[0]
    lv = [v for v in loo.values() if v is not None]
    out.update(tasks_with_info=int((shv > 1e-9).sum()), top_task=str(top), top_share=float(shv[0]), top3_share=float(shv[:3].sum()), eff_tasks=float(1 / (shv ** 2).sum()),
               loo_se_max=float(max(lv)) if lv else None, loo_se_without_top=loo[top], loo_uninformative=int(sum(v is None for v in loo.values())))
    return out


def eligibility_interval(screen, e, n_conf_q, n_boot=4000):
    """Projected eligible confirmation queries with a task-bootstrap 95 % interval on the pilot's eligibility rate (queries of a task together)."""
    tasks = sorted(set(r["task"] for r in screen)); by = {t: [r for r in screen if r["task"] == t] for t in tasks}
    el = np.array([sum((not any(r["solved"])) and r["W"] >= e + R_MIN for r in by[t]) for t in tasks], float); nq = np.array([len(by[t]) for t in tasks], float)
    rng = np.random.default_rng([SEED, 29, 9, e]); rates = []
    for _ in range(n_boot):
        i = rng.integers(0, len(tasks), len(tasks)); rates.append(el[i].sum() / nq[i].sum())
    r = el.sum() / nq.sum(); return dict(rate=float(r), projected=float(r * n_conf_q), lo=float(np.percentile(rates, 2.5) * n_conf_q), hi=float(np.percentile(rates, 97.5) * n_conf_q))


# ---------------- the runner (torch; the TRM environment) ----------------
class Runner:
    def __init__(self, out, threads):
        import torch
        self.torch = torch; torch.set_num_threads(threads); self.threads = threads
        os.environ["RR_ARC_ASSETS"] = str(ASSETS); sys.path.insert(0, str(ADAPTER))
        import field_models_arc as FA
        FA.FAMILIES["trm_arc1"]["ckpt"] = CKPT
        self.FA = FA; self.m = FA.load("trm_arc1", "cpu", torch.float32); self.inner = self.m.inner
        from trm.evaluation.arc import _crop_np
        self.crop_np = _crop_np
        assert self.m.P == P_PREFIX and self.m.hid == HID and self.m.seq == P_PREFIX + NPOS
        self.rows = self.m.dataset(); self.W32 = self.inner.lm_head.weight.detach().clone()
        self.W = self.W32.double().numpy(); assert self.W.shape == (VOCAB, HID) and np.linalg.matrix_rank(self.W) == VOCAB
        self.N, self.Wp = nullspace_projector(self.W)
        self.out = Path(out); self.out.mkdir(parents=True, exist_ok=True)
        src = ASSETS / "src/TRM_alphaxiv"
        self.files = [Path(__file__), ADAPTER / "field_models_arc.py", ADAPTER / "arc_shims/mps_patches.py", src / "src/trm/models/architectures/trm.py",
                      src / "src/trm/models/layers.py", src / "src/trm/models/sparse_embedding.py", src / "src/trm/evaluation/arc.py",
                      ROOT / "paper/code/evidence/arc/model_data/canonical.npz", CKPT]
        self.meta = dict(torch=torch.__version__, numpy=np.__version__, threads=threads, dtype="float32", device="cpu", config=self.m.cfg,
                         sources={str(f.relative_to(ROOT)): sha(f) for f in self.files})

    # -- model access --
    def batch_inputs(self, qis):
        T = self.torch; r = [self.rows[q] for q in qis]
        return {"inputs": T.as_tensor(np.stack([x["inputs"] for x in r]).astype(np.int32)), "puzzle_identifiers": T.as_tensor(np.array([x["puzzle_identifier"] for x in r], np.int32))}

    def step(self, zH, zL, qis):
        (zH, zL), logits, _q = self.m.step((zH, zL), self.batch_inputs(qis)); return zH, zL, logits

    def logits32(self, zH):
        with self.torch.inference_mode(): return self.inner.lm_head(zH)[..., P_PREFIX:, :]

    def gauss(self, qi):
        z = np.random.default_rng([SEED, int(qi), 7]).standard_normal((2, P_PREFIX + NPOS, HID)).astype(np.float32); return self.torch.from_numpy(z[0]), self.torch.from_numpy(z[1])

    # -- gates --
    def native_gate(self, qis, states):
        """The adapter's update vs the native inner recurrence (TinyRecursiveReasoningModel_ACTV1_Inner.forward) from identical states, K_MAX updates."""
        from trm.models.architectures.trm import TinyRecursiveReasoningModel_ACTV1InnerCarry as Carry
        T = self.torch; worst = 0.0; bit = True; b = self.batch_inputs(qis)
        with T.inference_mode():
            zH, zL = states; aH, aL, nH, nL = zH.clone(), zL.clone(), zH.clone(), zL.clone()
            for _ in range(K_MAX):
                aH, aL, la = self.step(aH, aL, qis)
                c, ln, _q = self.inner(Carry(z_H=nH, z_L=nL), b); nH, nL = c.z_H, c.z_L
                for x, y in ((aH, nH), (aL, nL), (la, ln)):
                    worst = max(worst, float((x - y).abs().max())); bit &= bool(T.equal(x, y))
        return dict(bitwise=bit, max_abs=worst)

    def histories(self, qis):
        """A = fixed start after D_INT updates, B = Gaussian start after D_INT updates; float32 tensors per query."""
        T = self.torch; B = len(qis)
        with T.inference_mode():
            zH0, zL0 = self.m.init_state(B, "fixed"); g = [self.gauss(q) for q in qis]; gH, gL = T.stack([x[0] for x in g]), T.stack([x[1] for x in g])
            aH, aL, bH, bL = zH0.clone(), zL0.clone(), gH.clone(), gL.clone()
            for _ in range(D_INT):
                aH, aL, la = self.step(aH, aL, qis); bH, bL, lb = self.step(bH, bL, qis)
        return (aH, aL, la), (bH, bL, lb), (zH0, zL0, gH, gL)

    def roll(self, specs, build):
        """Roll each spec K_MAX updates in batches of VB (padded with the batch's first variant); record displays and E logits per update."""
        T = self.torch; disp = np.zeros((len(specs), K_MAX, NPOS), np.uint8); elog = np.zeros((len(specs), K_MAX, max(E_CAND), VOCAB), np.float32)
        with T.inference_mode():
            for b0 in range(0, len(specs), VB):
                sp = specs[b0:b0 + VB]; built = [build(s) for s in sp]; pad = VB - len(sp)
                zH = T.stack([x[0] for x in built] + [built[0][0]] * pad); zL = T.stack([x[1] for x in built] + [built[0][1]] * pad); qis = [s["qi"] for s in sp] + [sp[0]["qi"]] * pad
                for k in range(K_MAX):
                    zH, zL, lg = self.step(zH, zL, qis); lg = lg[:len(sp)]
                    disp[b0:b0 + len(sp), k] = lg.argmax(-1).numpy().astype(np.uint8)
                    for j, s in enumerate(sp): elog[b0 + j, k, :len(s["Etok"])] = lg[j, s["Etok"]].numpy()
        return disp, elog

    # -- one block of queries --
    def block(self, qis, conds, e_excl, e_eval_max, gates, info_all, t, exclude_flips=False):
        T = self.torch; P = P_PREFIX; t0 = time.time()
        (aH, aL, la), (bH, bL, lb), starts = self.histories(qis); t["histories"] += time.time() - t0
        states, H = [], {}
        for j, qi in enumerate(qis):
            lab = self.rows[qi]["labels"].astype(np.int64); gt = self.rows[qi]["gt"]; dA = la[j].argmax(-1).numpy(); dB = lb[j].argmax(-1).numpy()
            gates["crop_mismatch"] += int(not (np.array_equal(crop(dA), self.crop_np(dA)) and np.array_equal(crop(dB), self.crop_np(dB))))
            solA, solB = self.m.exact(crop(dA), gt), self.m.exact(crop(dB), gt); W = shared_colour_wrong(dA, dB, lab)
            Wany = np.where(colour(lab) & (dA != lab) & (dB != lab))[0]
            rec = dict(qi=int(qi), task=self.rows[qi]["task"], solved=(bool(solA), bool(solB)), W=len(W), W_any=len(Wany), n_ingrid=int(colour(lab).sum()), struct=(structural_errors(dA, lab), structural_errors(dB, lab)),
                       acc=(float((dA == lab)[colour(lab)].mean()), float((dB == lab)[colour(lab)].mean())), shape_ok=(crop(dA).shape == gt.shape, crop(dB).shape == gt.shape))
            info_all["screen"].append(rec)
            if solA or solB or len(W) < e_excl + R_MIN: continue
            Etok, masks = draw_challenges(qi, W, e_excl, e_eval_max)
            hs = [(aH[j], aL[j], dA), (bH[j], bL[j], dB)]; built = []
            for di in (0, 1):                                    # direction 0: recipient A, donor B; 1: recipient B, donor A
                r, d = (0, 1) if di == 0 else (1, 0); hr32, lr32, dr = hs[r]; hd32 = hs[d][0]
                hr, hd = hr32.double().numpy(), hd32.double().numpy()
                full, ctrl, tinfo = transplant_states(hr, hd, self.N, P, np.random.default_rng([SEED, int(qi), 6, di]))
                cs = dict(intact=hr32, full=T.from_numpy(full.astype(np.float32)), full_ctrl=T.from_numpy(ctrl.astype(np.float32)))
                if "ctrl_b" in conds:                              # the calibration's independent second draw of the same control construction
                    ctrl2, _, tinfo2 = (lambda x: (x[1], None, x[2]))(transplant_states(hr, hd, self.N, P, np.random.default_rng([SEED, int(qi), 16, di])))
                    cs["ctrl_b"] = T.from_numpy(ctrl2.astype(np.float32)); tinfo = dict(tinfo, **{k_: max(tinfo[k_], tinfo2[k_]) for k_ in ("ctrl_norm_dev", "ctrl_change_dev", "ctrl_in_null")})
                TC = [c for c in cs if c != "intact"]
                L0 = self.logits32(hr32); nflip = sum(int((self.logits32(cs[c]).argmax(-1) != L0.argmax(-1)).sum()) for c in TC)
                built.append((di, r, d, hr32, lr32, dr, hd32, hr, cs, tinfo, L0, nflip))
            if exclude_flips and any(x[-1] > 0 for x in built):    # registered rule: the whole query is excluded and counted
                gates["excluded_flip_queries"] += 1; rec["excluded_flip"] = True; continue
            for (di, r, d, hr32, lr32, dr, hd32, hr, cs, tinfo, L0, nflip) in built:
                TC = [c for c in cs if c != "intact"]
                for c in TC:                                       # executed float32 checks of the transplants (all 900 answer positions)
                    Lc = self.logits32(cs[c]); gates[f"{c}_logit_maxdev"] = max(gates.get(f"{c}_logit_maxdev", 0.0), float((Lc - L0).abs().max()))
                    gates[f"{c}_flips"] = gates.get(f"{c}_flips", 0) + int((Lc.argmax(-1) != L0.argmax(-1)).sum())
                gates["full_prefix_not_donor"] += int(not T.equal(cs["full"][:P], hd32[:P]))
                top2 = L0.topk(2, -1).values; gates["min_top2_gap"] = min(gates["min_top2_gap"], float((top2[..., 0] - top2[..., 1]).min()))
                for k_ in ("ctrl_norm_dev", "ctrl_change_dev", "ctrl_in_null"): gates[k_] = max(gates[k_], tinfo[k_])
                # corrections: built once per (recipient, mask) from the recipient's intact state; the same float32 change for every condition
                cors, cinfo = {}, []
                for (i, m_, cells) in masks:
                    dl, tgt, lg = correction_delta(hr[P + cells], self.W, self.Wp, lab[cells]); d32 = T.from_numpy(dl.astype(np.float32)); cors[(i, m_)] = (cells, d32)
                    cinfo.append(dict(dose=i, mask=m_, n=len(cells), dnorm=float(np.linalg.norm(dl, axis=-1).mean()), hnorm=float(np.linalg.norm(hr[P + cells], axis=-1).mean())))
                    for c in ["intact"] + TC:
                        x = cs[c].clone(); x[P + cells] += d32; Lx = self.logits32(x); want = L0.argmax(-1).numpy().copy(); want[cells] = lab[cells]
                        gates["corr_display"] += int(not np.array_equal(Lx.argmax(-1).numpy(), want))
                        gates[f"corr_logit_maxdev_{c}"] = max(gates.get(f"corr_logit_maxdev_{c}", 0.0), float(np.abs(Lx[cells].double().numpy() - tgt).max()))
                        other = np.setdiff1d(np.arange(NPOS), cells); Lc0 = self.logits32(cs[c])
                        gates["corr_other_changed"] += int(not T.equal(Lx[other], Lc0[other]))
                si = len(states); H[si] = dict(cs=cs, l=lr32, cors=cors, saved=(hr32.clone(), lr32.clone()), qi=int(qi))
                states.append(dict(qi=int(qi), task=self.rows[qi]["task"], dir=di, Etok=Etok, Wn=len(W), lab=lab, disp_d1=dr.astype(np.uint8), masks=masks,
                                   tinfo=tinfo, cinfo=cinfo))
        # variants
        V = []
        for si, s in enumerate(states):
            for c in conds:
                V.append(dict(si=si, qi=s["qi"], cond=c, arm=0, dose=-1, mask=-1, Etok=s["Etok"]))
                for (i, m_, cells) in s["masks"]: V.append(dict(si=si, qi=s["qi"], cond=c, arm=1, dose=i, mask=m_, Etok=s["Etok"]))
        nmain = len(V)
        for si, s in enumerate(states):                           # exact restoration, built after every other variant
            V.append(dict(si=si, qi=s["qi"], cond="restore", arm=0, dose=-1, mask=-1, Etok=s["Etok"]))
            for i in range(len(DOSES)): V.append(dict(si=si, qi=s["qi"], cond="restore", arm=1, dose=i, mask=0, Etok=s["Etok"]))

        def build(sp):
            Hh = H[sp["si"]]
            if sp["cond"] == "restore":
                h, l = Hh["saved"]; gates["restore_tensor"] += int(not (T.equal(h, Hh["cs"]["intact"]) and T.equal(l, Hh["l"])))
            else: h, l = Hh["cs"][sp["cond"]], Hh["l"]
            if sp["arm"] == 0: return h, l
            cells, d32 = Hh["cors"][(sp["dose"], sp["mask"])]; x = h.clone(); x[P + cells] += d32; return x, l
        t0 = time.time(); disp, elog = self.roll(V, build); t["rollouts"] += time.time() - t0; t["variants"] += len(V)
        for q in range(nmain, len(V)):
            v = V[q]; j = next(i for i in range(nmain) if V[i]["si"] == v["si"] and V[i]["cond"] == "intact" and V[i]["arm"] == v["arm"] and V[i]["dose"] == v["dose"] and V[i]["mask"] == v["mask"])
            gates["restore_rollout"] += int(not (np.array_equal(disp[q], disp[j]) and np.array_equal(elog[q], elog[j])))
        return states, V[:nmain], disp[:nmain], elog[:nmain]

    def run_blocks(self, qlist, conds, e_excl, e_eval_max, tag, shard=0, nshard=1, gate_native=True, exclude_flips=False, tol=None):
        """Process query blocks (BLOCK queries) of qlist; block b belongs to shard b % nshard; each block is checkpointed to its own npz."""
        T = self.torch; bdir = self.out / f"{tag}_blocks"; bdir.mkdir(exist_ok=True)
        blocks = [qlist[i:i + BLOCK] for i in range(0, len(qlist), BLOCK)]
        for b, qis in enumerate(blocks):
            if b % nshard != shard: continue
            dst = bdir / f"block_{b:03d}.npz"
            if dst.exists(): continue
            gates = dict(crop_mismatch=0, restore_tensor=0, restore_rollout=0, corr_display=0, corr_other_changed=0, full_flips=0, full_ctrl_flips=0, full_prefix_not_donor=0,
                         full_logit_maxdev=0.0, full_ctrl_logit_maxdev=0.0, corr_logit_maxdev_intact=0.0, corr_logit_maxdev_full=0.0, corr_logit_maxdev_full_ctrl=0.0,
                         ctrl_norm_dev=0.0, ctrl_change_dev=0.0, ctrl_in_null=0.0, min_top2_gap=float("inf"), excluded_flip_queries=0)
            info = dict(screen=[]); t = dict(histories=0.0, rollouts=0.0, variants=0); t0 = time.time()
            if gate_native and b == shard:                        # native recurrence and batch invariance, on the shard's first block
                (aH, aL, _), (bH, bL, _), _ = self.histories(qis[:2])
                gates["native"] = [self.native_gate(qis[:2], (aH, aL)), self.native_gate(qis[:2], (bH, bL))]
                full = T.from_numpy(transplant_states(aH[0].double().numpy(), bH[0].double().numpy(), self.N, P_PREFIX, np.random.default_rng(0))[0].astype(np.float32))
                gates["native"].append(self.native_gate(qis[:1], (full[None], aL[:1])))
                with T.inference_mode():                           # the same row alone vs inside a batch of VB
                    z1 = self.step(aH[:1], aL[:1], qis[:1])[2]; zb = self.step(T.cat([aH[:1]] + [bH[:1]] * (VB - 1)), T.cat([aL[:1]] + [bL[:1]] * (VB - 1)), [qis[0]] * VB)[2][:1]
                gates["batch_invariant"] = bool(T.equal(z1, zb)); gates["batch_maxdev"] = float((z1 - zb).abs().max())
            states, V, disp, elog = self.block(qis, conds, e_excl, e_eval_max, gates, info, t, exclude_flips); t["wall"] = time.time() - t0
            ok = gates["crop_mismatch"] == 0 and gates["restore_tensor"] == 0 and gates["restore_rollout"] == 0 and gates["corr_display"] == 0 and gates["corr_other_changed"] == 0 \
                and gates["full_flips"] == 0 and gates["full_ctrl_flips"] == 0 and gates.get("ctrl_b_flips", 0) == 0 and gates["full_prefix_not_donor"] == 0 \
                and all(g_["bitwise"] for g_ in gates.get("native", [dict(bitwise=True)])) and gates.get("batch_invariant", True) \
                and gates["ctrl_norm_dev"] < 1e-9 and gates["ctrl_change_dev"] < 1e-9 and gates["ctrl_in_null"] < 1e-9
            if tol is not None:                                   # the registered executed-float32 tolerances
                ok = ok and max(v for k_, v in gates.items() if k_.endswith("_logit_maxdev") and not k_.startswith("corr")) <= tol[0] \
                    and max(v for k_, v in gates.items() if k_.startswith("corr_logit_maxdev")) <= tol[1]
            S = len(states); nE = max(E_CAND)
            np.savez_compressed(dst, gates=json.dumps(gates), gates_ok=ok, timing=json.dumps(t), screen=json.dumps(info["screen"]), conds=np.array(conds),
                                state_qi=np.array([s["qi"] for s in states], np.int32), state_task=np.array([s["task"] for s in states], dtype="U32"), state_dir=np.array([s["dir"] for s in states], np.int8),
                                state_Etok=np.stack([s["Etok"] for s in states]).astype(np.int16) if S else np.zeros((0, nE), np.int16),
                                state_W=np.array([s["Wn"] for s in states], np.int16), state_lab=np.stack([s["lab"] for s in states]).astype(np.uint8) if S else np.zeros((0, NPOS), np.uint8),
                                state_disp_d1=np.stack([s["disp_d1"] for s in states]) if S else np.zeros((0, NPOS), np.uint8),
                                state_masks=json.dumps([[[int(i), int(m_), c.tolist()] for (i, m_, c) in s["masks"]] for s in states]),
                                state_tinfo=json.dumps([s["tinfo"] for s in states]), state_cinfo=json.dumps([s["cinfo"] for s in states]),
                                var_state=np.array([v["si"] for v in V], np.int32), var_cond=np.array([conds.index(v["cond"]) for v in V], np.int8),
                                var_arm=np.array([v["arm"] for v in V], np.int8), var_dose=np.array([v["dose"] for v in V], np.int8), var_mask=np.array([v["mask"] for v in V], np.int8),
                                disp=disp, elog=elog, meta=json.dumps(dict(self.meta, created=utc(), block=b, queries=[int(q) for q in qis], tag=tag)))
            print(json.dumps(dict(block=b, queries=len(qis), states=S, variants=len(V), wall=round(t["wall"], 1), per_variant=round(t["rollouts"] / max(t["variants"], 1), 3), gates_ok=bool(ok))), flush=True)
            if not ok: raise SystemExit(f"gate failed in block {b}: {gates}")
        for f in self.files: assert sha(f) == self.meta["sources"][str(f.relative_to(ROOT))], f"source changed during the run: {f}"


def query_order(rows, tasks):
    """Queries of the given tasks in the frozen task order, then by test index."""
    pos = {t: i for i, t in enumerate(tasks)}; q = [i for i, r in enumerate(rows) if r["task"] in pos]
    return sorted(q, key=lambda i: (pos[rows[i]["task"]], rows[i]["test_idx"]))


def merge(tag, base=OUT):
    """Concatenate the block files of a tag in block order (state indices offset), with per-block gates kept."""
    fs = sorted((base / f"{tag}_blocks").glob("block_*.npz")); assert fs, "no blocks"
    acc = {}; off = 0; gates, timing, screen, metas = [], [], [], []
    for f in fs:
        d = np.load(f, allow_pickle=False); assert bool(d["gates_ok"]), f
        gates.append(json.loads(str(d["gates"]))); timing.append(json.loads(str(d["timing"]))); screen += json.loads(str(d["screen"])); metas.append(json.loads(str(d["meta"])))
        for k in ("state_qi", "state_task", "state_dir", "state_Etok", "state_W", "state_lab", "state_disp_d1", "var_cond", "var_arm", "var_dose", "var_mask", "disp", "elog"):
            acc.setdefault(k, []).append(d[k])
        acc.setdefault("var_state", []).append(d["var_state"] + off)
        for k in ("state_masks", "state_tinfo", "state_cinfo"): acc.setdefault(k, []).extend(json.loads(str(d[k])))
        conds = [str(x) for x in d["conds"]]; off += len(d["state_qi"])
    out = {k: (json.dumps(v) if k in ("state_masks", "state_tinfo", "state_cinfo") else np.concatenate(v)) for k, v in acc.items()}
    dst = base / f"p29_{tag}.npz"
    np.savez_compressed(dst, conds=np.array(conds), gates=json.dumps(gates), timing=json.dumps(timing), screen=json.dumps(screen), meta=json.dumps(metas), **out)
    (base / f"p29_{tag}.sha256.json").write_text(json.dumps(dict(sha256=sha(dst))) + "\n"); print(f"merged {len(fs)} blocks -> {dst} ({off} states)")


# ---------------- tables from a merged file ----------------
def load(tag, base=OUT):
    d = np.load(base / f"p29_{tag}.npz", allow_pickle=False); return d


def donor_map(d):
    key = {(int(q), int(x)): s for s, (q, x) in enumerate(zip(d["state_qi"], d["state_dir"]))}
    return {s: key.get((int(q), 1 - int(x))) for s, (q, x) in enumerate(zip(d["state_qi"], d["state_dir"]))}


def unit_table(d, measure, n_e, k):
    """Y[(state, cond, dose, mask)] and B[(state, cond)] at update k (1-based) on the first n_e evaluation tokens. measure: 'correct' (share of E
    tokens displaying the label) or 'margin' (mean correct-token margin over E)."""
    st, cond, arm, dose, mask = d["var_state"], d["var_cond"], d["var_arm"], d["var_dose"], d["var_mask"]
    Et = d["state_Etok"][:, :n_e].astype(np.int64); lab = d["state_lab"]; Y, Bm = {}, {}
    labE = np.take_along_axis(lab.astype(np.int64), Et, 1)
    for i in range(len(st)):
        s = int(st[i])
        if measure == "correct": y = float((d["disp"][i, k - 1, Et[s]] == labE[s]).mean())
        else: y = float(margins(d["elog"][i, k - 1, :n_e].astype(np.float64), labE[s]).mean())
        if arm[i] == 0: Bm[(s, int(cond[i]))] = y
        else: Y[(s, int(cond[i]), int(dose[i]), int(mask[i]))] = y
    return Y, Bm


def margin_scales(d, n_e, don):
    """Option B's scale per query: the RMS of the pooled centered intact responses (mean E-margin at k = 1) of both histories, centered within each
    (history, dose) over the four masks; fixed before any transplant outcome and shared by every condition, direction and horizon."""
    Y, _ = unit_table(d, "margin", n_e, 1); ix = [str(x) for x in d["conds"]].index("intact"); qs = d["state_qi"]; acc = {}
    for s_ in sorted(don):
        for i in range(len(DOSES)):
            a = np.array([Y[(s_, ix, i, m)] for m in range(MASKS)]); acc.setdefault(int(qs[s_]), []).extend(list(a - a.mean()))
    return {q: float(np.sqrt(np.mean(np.square(v)))) for q, v in acc.items()}


def tables(d, measure, n_e, k, don):
    """(Y, B, donor map, excluded queries) for 'correct', 'margin' or 'margin_std' (option B: the margin divided by the query's scale; queries with a
    scale below SCALE_MIN excluded from this measure and counted)."""
    if measure != "margin_std":
        Y, Bm = unit_table(d, measure, n_e, k); return Y, Bm, don, set()
    sc = margin_scales(d, n_e, don); exc = {q for q, v in sc.items() if v < SCALE_MIN}; qs = d["state_qi"]; Y, Bm = unit_table(d, "margin", n_e, k)
    keep = lambda s_: int(qs[s_]) not in exc
    Y2 = {key: v / sc[int(qs[key[0]])] for key, v in Y.items() if keep(key[0])}; B2 = {key: v / sc[int(qs[key[0]])] for key, v in Bm.items() if keep(key[0])}
    return Y2, B2, {s_: (ds if keep(s_) else None) for s_, ds in don.items()}, exc


def jackknife_se(fn, clusters):
    """Task jackknife SE of a statistic fn(mask over rows) -> scalar (unidentified omissions dropped and counted)."""
    cl = np.unique(clusters); th = np.array([fn(clusters != c) for c in cl]); ok = ~np.isnan(th); G = int(ok.sum())
    return (float(np.sqrt((G - 1) / G * np.sum((th[ok] - th[ok].mean()) ** 2))) if G > 1 else float("nan")), th, int((~ok).sum())


def pilot_report(base=OUT, tag="pilot", sudoku_ref=None):
    """Intact-only diagnostics on the pilot tasks: each candidate evaluation-set size rolled with its own design on its own eligible queries;
    horizons k = 1..K_MAX; token correctness and the correct-token margin; eligibility; costs; magnitudes; gates."""
    rows = [f"P29 FEASIBILITY PILOT (intact responses only; {N_PILOT_TASKS} pilot tasks, excluded from confirmation)"]; js = dict(diag={}, gates={}, cost={})
    d4 = load(f"{tag}_E{E_CAND[0]}", base); scr = json.loads(str(d4["screen"])); nq = len(scr); unsolved = [s for s in scr if not any(s["solved"])]
    rows.append(f"screened queries {nq} | solved after one update: fixed {sum(s['solved'][0] for s in scr)}, Gaussian {sum(s['solved'][1] for s in scr)} | both unsolved {len(unsolved)}")
    pc = lambda a, q: float(np.percentile(a, q)) if len(a) else 0.0
    Wq = np.array([s["W"] for s in unsolved]); Wa = np.array([s["W_any"] for s in unsolved])
    rows.append(f"shared wrong in-grid tokens among both-unsolved: colour-to-colour median {pc(Wq, 50):.0f} (quartiles {pc(Wq, 25):.0f}/{pc(Wq, 75):.0f}); any display median {pc(Wa, 50):.0f} (quartiles {pc(Wa, 25):.0f}/{pc(Wa, 75):.0f})")
    elig = {e: sum(s["W"] >= e + R_MIN for s in unsolved) for e in E_CAND}; elig_any = {e: sum(s["W_any"] >= e + R_MIN for s in unsolved) for e in E_CAND}
    rows.append("eligible queries (|W| >= |E| + %d), colour-to-colour [any display]: " % R_MIN + ", ".join(f"|E|={e}: {elig[e]} [{elig_any[e]}] of {nq}" for e in E_CAND))
    st_in = np.array([[s["struct"][0][0], s["struct"][1][0]] for s in unsolved]); st_out = np.array([[s["struct"][0][1], s["struct"][1][1]] for s in unsolved])
    if len(st_in):
        rows.append(f"structural tokens at d=1 among both-unsolved (fixed, Gaussian): in-grid non-colour mean {st_in.mean(0).round(1).tolist()}, out-of-grid colour mean {st_out.mean(0).round(1).tolist()}; decoded shape correct {sum(s['shape_ok'][0] for s in unsolved)} and {sum(s['shape_ok'][1] for s in unsolved)} of {len(unsolved)}")
    js["eligibility"] = dict(screened=nq, both_unsolved=len(unsolved), eligible=elig, eligible_any=elig_any, W_median=pc(Wq, 50), W_any_median=pc(Wa, 50))
    n_conf_q = 419 - nq; rate = {e: elig[e] / max(nq, 1) for e in E_CAND}; ei = {e: eligibility_interval(scr, e, n_conf_q) for e in E_CAND}; js["eligibility"]["projection"] = ei
    rows.append(f"confirmation queries {n_conf_q}; projected eligible (task-bootstrap 95 % interval) " + ", ".join(f"|E|={e}: {ei[e]['projected']:.0f} [{ei[e]['lo']:.0f}, {ei[e]['hi']:.0f}]" for e in E_CAND))
    for e in E_CAND:
        d = load(f"{tag}_E{e}", base); don = donor_map(d); ns = len(d["state_qi"]); gates = json.loads(str(d["gates"])); tim = json.loads(str(d["timing"])); tos = [str(x) for x in d["state_task"]]
        g0 = {k_: (min if k_ == "min_top2_gap" else max)(g[k_] for g in gates) for k_ in gates[0] if k_ not in ("native", "batch_invariant", "batch_maxdev")}
        js["gates"][f"E{e}"] = dict(max=g0, native=gates[0].get("native"), batch_invariant=gates[0].get("batch_invariant"))
        vt = sum(x["variants"] for x in tim); rt = sum(x["rollouts"] for x in tim); ht = sum(x["histories"] for x in tim)
        js["cost"][f"E{e}"] = dict(variants=vt, rollout_s=rt, per_variant_s=rt / max(vt, 1), histories_s=ht)
        rows.append(f"|E|={e}: states rolled {ns}; {vt} variants x {K_MAX} updates in {rt:.0f} s = {rt / max(vt, 1):.3f} s per variant; gates {g0}; native {gates[0].get('native')}; batch-invariant {gates[0].get('batch_invariant')}")
        if ns == 0: continue
        for meas in ("correct", "margin"):
            for k in range(1, K_MAX + 1):
                Y, Bm = unit_table(d, meas, e, k); dg = design_diagnostics(Y, don, tos)
                dg.update(noedit=float(np.mean([v for (s, c), v in Bm.items() if c == 0])), by_dose=[float(np.mean([v for (s, c, i, m), v in Y.items() if c == 0 and i == dd])) for dd in range(len(DOSES))])
                if not dg["informative"]:
                    js["diag"][f"{meas}|E{e}|k{k}"] = dg; rows.append(f"  {meas:7s} |E|={e:2d} k={k}: units {dg['units']}, varying share {dg['vary_share']:.3f} -> UNINFORMATIVE (donor coefficient not identified)"); continue
                f_ = np.sqrt(ns / max(2 * ei[e]["projected"], 1)) * CLUSTER_FACTOR; dg.update(se_conf_projected=float(dg["se_proxy"] * f_), mde_projected_heuristic=float(MDE_Z * dg["se_proxy"] * f_))
                js["diag"][f"{meas}|E{e}|k{k}"] = dg
                rows.append(f"  {meas:7s} |E|={e:2d} k={k}: units {dg['units']} ({dg['tasks']} tasks), varying share {dg['vary_share']:.3f}, R2 {dg['R2']:.3f} | residual donor information: tasks with any {dg['tasks_with_info']}, top task {dg['top_share']:.3f}, top 3 {dg['top3_share']:.3f}, effective tasks {dg['eff_tasks']:.1f} | proxy SE {dg['se_proxy']:.4f}; leave-one-task-out max {dg['loo_se_max'] if dg['loo_se_max'] is None else round(dg['loo_se_max'], 4)}, without the top task {dg['loo_se_without_top'] if dg['loo_se_without_top'] is None else round(dg['loo_se_without_top'], 4)}, uninformative omissions {dg['loo_uninformative']} | heuristic MDE {dg['mde_projected_heuristic']:.3f} | no-edit {dg['noedit']:.3f}, by dose {[round(x, 3) for x in dg['by_dose']]}")
        tin = json.loads(str(d["state_tinfo"])); cin = json.loads(str(d["state_cinfo"])); dirs = d["state_dir"]
        ra = np.array([x["rel_change_answer"] for x in tin]); rp = np.array([x["rel_change_prefix"] for x in tin])
        dn = np.array([np.mean([c["dnorm"] / c["hnorm"] for c in ci_]) for ci_ in cin]); rr = []
        for s_, ds in don.items():
            if ds is not None and dirs[s_] == 0: rr += list(np.array([c["dnorm"] for c in cin[s_]]) / np.array([c["dnorm"] for c in cin[ds]]))
        Ym, _ = unit_table(d, "margin", e, 1); um = centered_units(Ym, don, 0, 0, 0); sdm = np.array([np.std(um["cR"][i:i + MASKS]) for i in range(0, len(um["cR"]), MASKS)])
        dev = max(g0["full_logit_maxdev"], g0["full_ctrl_logit_maxdev"])
        rows.append(f"  executed transplant logit deviation {dev:.2e} vs the across-mask SD of the intact k=1 E-margin: median {np.median(sdm):.3f}, 10th percentile {np.percentile(sdm, 10):.4f} (ratio to the median {dev / max(np.median(sdm), 1e-30):.1e})")
        js["magnitudes_E%d" % e] = dict(margin_sd_median=float(np.median(sdm)), margin_sd_p10=float(np.percentile(sdm, 10)), transplant_answer=float(ra.mean()), transplant_prefix=float(rp.mean()), corr_fixed=float(dn[dirs == 0].mean()), corr_gauss=float(dn[dirs == 1].mean()), ratio_median=float(np.median(rr)))
        rows.append(f"  magnitudes: transplant ||h_d - h_r||/||h_r|| answer {ra.mean():.3f}, prefix {rp.mean():.3f}; correction ||delta||/||h|| fixed-start {dn[dirs == 0].mean():.3f}, Gaussian-start {dn[dirs == 1].mean():.3f}; same-mask norm ratio fixed/Gaussian median {np.median(rr):.3f} (quartiles {np.percentile(rr, 25):.3f}/{np.percentile(rr, 75):.3f})")
    if sudoku_ref: rows.append(f"Sudoku reference (P27 intact rows, correctness, |E|=4, k=1): {sudoku_ref}")
    (base / "pilot_report.txt").write_text("\n".join(rows) + "\n"); (base / "pilot_report.json").write_text(json.dumps(js, indent=1, default=float)); print("\n".join(rows))


# ---------------- the frozen confirmation reader ----------------
def wrong_counts(d):
    """Each state's number of wrong in-grid tokens after one update (any display), for P26's R1 nuisance fractions."""
    lab, dd = d["state_lab"].astype(np.int64), d["state_disp_d1"].astype(np.int64); return (colour(lab) & (dd != lab)).sum(1)


def r1_units(d, Y, Bm, don, ix, cx, cref):
    """P26's R1 rows per (state, dose, mask): (Y - b) of cx and cref; the donor's and recipient's intact (Y - b); the registered nuisance set."""
    nw = wrong_counts(d); masks = json.loads(str(d["state_masks"])); rows = []
    for s in sorted(don):
        ds = don[s]
        if ds is None: continue
        size = {(i, m_): len(c) for i, m_, c in masks[s]}
        for i in range(len(DOSES)):
            for m in range(MASKS):
                ne = size[(i, m)]; rows.append(dict(state=s, YR=Y[(s, ix, i, m)] - Bm[(s, ix)], YD=Y[(ds, ix, i, m)] - Bm[(ds, ix)], bR=Bm[(s, ix)], bD=Bm[(ds, ix)],
                                                     fR=ne / nw[s], fD=ne / nw[ds], Yx=Y[(s, cx, i, m)] - Bm[(s, cx)], bx=Bm[(s, cx)], Yref=Y[(s, cref, i, m)] - Bm[(s, cref)], bref=Bm[(s, cref)]))
    return {k_: np.array([r[k_] for r in rows]) for k_ in rows[0]}


def r1_beta(u, which):
    y, b = u["Y" + which], u["b" + which]; fR, fD, bR, bD = u["fR"], u["fD"], u["bR"], u["bD"]
    X = np.stack([u["YD"], u["YR"], b, bR, bD, fR, fD, fR ** 2, fD ** 2, b * fR, bR * fR, bD * fD, np.ones(len(y))], 1)
    return float(np.linalg.lstsq(X, y, rcond=None)[0][0])


def baseline_slope(Bm, don, ix, cx):
    s_ = [s for s in sorted(don) if don[s] is not None]; x = np.array([Bm[(don[s], ix)] - Bm[(s, ix)] for s in s_]); y = np.array([Bm[(s, cx)] - Bm[(s, ix)] for s in s_])
    X = np.column_stack([x, np.ones(len(x))]); return float(np.linalg.lstsq(X, y, rcond=None)[0][0]), np.array(s_)


def boot_generic(fn, n_states, clusters, seed, n):
    """Task-clustered bootstrap for statistics without cluster sufficient statistics: fn(state_weights) -> scalar (weights per state)."""
    cl = np.unique(clusters); inv = np.searchsorted(cl, clusters); rng = np.random.default_rng(seed); out = np.empty(n)
    for b in range(n):
        w = np.bincount(rng.integers(0, len(cl), len(cl)), minlength=len(cl)).astype(np.float64); out[b] = fn(w[inv])
    return out


def wls_beta(X, y, w):
    sw = np.sqrt(w); return np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0][0]


def report(base=OUT, tag="conf", quiet=False):
    """The frozen reader: the registered measure and |E| (FROZEN), k = 1 primary; S-ARC needs both lower bounds > 0."""
    assert FROZEN["E"] is not None and FROZEN["measure"] is not None, "settings not frozen"
    d = load(tag, base); conds = [str(x) for x in d["conds"]]; ix, jf, jc = conds.index("intact"), conds.index("full"), conds.index("full_ctrl")
    don = donor_map(d); task = d["state_task"]; tcl = np.unique(task, return_inverse=True)[1]; E = FROZEN["E"]; meas = FROZEN["measure"]
    order = [meas] + [m_ for m_ in ("margin_std", "margin", "correct") if m_ != meas]; res = dict(gates=json.loads(str(d["gates"])), states=int(len(d["state_qi"])), queries=int(len(np.unique(d["state_qi"]))),
                                                                     tasks=int(len(np.unique(task))), screen=len(json.loads(str(d["screen"]))))
    lines = [f"P29 REPORT (registered before any confirmation row; measure {meas}, |E| = {E}, k = 1 primary; {N_BOOT} task-clustered draws)",
             f"screened {res['screen']} queries; eligible {res['queries']} queries ({res['states']} pair-directions) from {res['tasks']} tasks"]
    fmt = lambda e, c: f"{e:+.3f} [{c[0]:+.3f}, {c[1]:+.3f}]"
    for k in range(1, K_MAX + 1):
        for mi, mz in enumerate(order):
            Y, Bm, dk, exc = tables(d, mz, E, k, don); res[f"excluded_small_scale|{mz}"] = sorted(exc)
            uF, uC = centered_units(Y, dk, jf, ix, ix), centered_units(Y, dk, jc, ix, ix); cl = tcl[uF["state"]]
            bF, bC = beta(uF), beta(uC); drF = boot_betas([uF], [1.0], [cl], [SEED, 29, k, 1 + mi, 1], N_BOOT); drD = boot_betas([uF, uC], [1.0, -1.0], [cl, cl], [SEED, 29, k, 1 + mi, 2], N_BOOT)
            drC = boot_betas([uC], [1.0], [cl], [SEED, 29, k, 1 + mi, 3], N_BOOT)
            r = dict(beta_full=bF, ci_full=ci(drF), beta_ctrl=bC, ci_ctrl=ci(drC), diff=bF - bC, ci_diff=ci(drD)); res[f"{mz}|k{k}"] = r
            tagk = "PRIMARY " if (k == 1 and mz == meas) else ""
            lines.append(f"  {tagk}{mz} k={k}: beta(full) {fmt(bF, r['ci_full'])} | beta(full_ctrl) {fmt(bC, r['ci_ctrl'])} | full - ctrl {fmt(bF - bC, r['ci_diff'])}")
            if k == 1 and mz == meas:
                al, bc = r["ci_full"][0] > 0, r["ci_diff"][0] > 0
                res["S_ARC"] = "S-ARC" if (al and bc) else ("BEATS-CONTROL-DONOR-ALIGNMENT-UNRESOLVED" if bc else ("DONOR-ALIGNED-NOT-BEYOND-CONTROL" if al else "NOT-RESOLVED"))
                # secondary: P26's BASELINE and R1 (primary measure, k = 1), task-clustered
                sF, st_ = baseline_slope(Bm, dk, ix, jf); sC, _ = baseline_slope(Bm, dk, ix, jc); bcl = tcl[st_]
                def bfn(w):
                    x = np.array([Bm[(dk[s], ix)] - Bm[(s, ix)] for s in st_]); X = np.column_stack([x, np.ones(len(x))])
                    return wls_beta(X, np.array([Bm[(s, jf)] - Bm[(s, ix)] for s in st_]), w) - wls_beta(X, np.array([Bm[(s, jc)] - Bm[(s, ix)] for s in st_]), w)
                res["BASELINE"] = dict(E=sF - sC, ci=ci(boot_generic(bfn, len(st_), bcl, [SEED, 29, 4], N_BOOT)))
                u1 = r1_units(d, Y, Bm, dk, ix, jf, jc); e1 = r1_beta(u1, "x") - r1_beta(u1, "ref"); c1 = tcl[u1["state"]]
                def rfn(w):
                    y1, y2, ww = u1["Yx"], u1["Yref"], w
                    mk = lambda which: np.stack([u1["YD"], u1["YR"], u1["b" + which], u1["bR"], u1["bD"], u1["fR"], u1["fD"], u1["fR"] ** 2, u1["fD"] ** 2, u1["b" + which] * u1["fR"], u1["bR"] * u1["fR"], u1["bD"] * u1["fD"], np.ones(len(y1))], 1)
                    return wls_beta(mk("x"), y1, ww) - wls_beta(mk("ref"), y2, ww)
                res["R1"] = dict(E=e1, ci=ci(boot_generic(rfn, len(u1["state"]), c1, [SEED, 29, 5], N_BOOT)))
                lines.append(f"     -> {res['S_ARC']} | secondary BASELINE (full - ctrl slope) {fmt(res['BASELINE']['E'], res['BASELINE']['ci'])} | R1 {fmt(e1, res['R1']['ci'])}")
                # registered task-influence sensitivity: (i) the primary contrasts without the task holding the largest share of the intact-only
                # residual donor information (defined without any transplant outcome); (ii) leave-one-task-out ranges of the three point estimates
                tk = task[uF["state"]]; dg = design_diagnostics(Y, dk, [str(x) for x in task], ix); top = dg.get("top_task") if dg else None
                sub = lambda u, m: {k_: v[m] for k_, v in u.items()}
                if top is not None:
                    m = tk != top; vF, vC = sub(uF, m), sub(uC, m); c2 = cl[m]
                    oF = ci(boot_betas([vF], [1.0], [c2], [SEED, 29, 6], N_BOOT)); oD = ci(boot_betas([vF, vC], [1.0, -1.0], [c2, c2], [SEED, 29, 7], N_BOOT))
                    res["omit_top_task"] = dict(task=top, share=dg["top_share"], beta_full=beta(vF), ci_full=oF, diff=beta(vF) - beta(vC), ci_diff=oD,
                                                letter="S-ARC" if (oF[0] > 0 and oD[0] > 0) else "NOT-S-ARC")
                    jk = [(beta(sub(uF, tk != t)), beta(sub(uC, tk != t))) for t in np.unique(tk)]; jF = np.array([x[0] for x in jk]); jD = np.array([x[0] - x[1] for x in jk])
                    res["jackknife"] = dict(beta_full=(float(np.nanmin(jF)), float(np.nanmax(jF))), diff=(float(np.nanmin(jD)), float(np.nanmax(jD))), tasks=int(len(jk)),
                                            unidentified=int(np.isnan(jF).sum()))
                    lines.append(f"     task influence: without the top-information task ({dg['top_share']:.3f} of it): beta(full) {fmt(beta(vF), oF)}, full - ctrl {fmt(beta(vF) - beta(vC), oD)} -> {res['omit_top_task']['letter']}; "
                                 f"leave-one-task-out ranges beta(full) [{np.nanmin(jF):+.3f}, {np.nanmax(jF):+.3f}], full - ctrl [{np.nanmin(jD):+.3f}, {np.nanmax(jD):+.3f}]; information: tasks with any {dg['tasks_with_info']}, effective {dg['eff_tasks']:.1f}")
                res["unidentified_draws"] = dict(full=n_unidentified(drF), diff=n_unidentified(drD))
    # decoded shape and exactness by condition (descriptive)
    lab = d["state_lab"]; gts = [crop(l) for l in lab]; st = d["var_state"]; cond = d["var_cond"]
    for k in (1, K_MAX):
        sh = {c: [] for c in conds}; ex = {c: [] for c in conds}
        for i in range(len(st)):
            g = crop(d["disp"][i, k - 1]); sh[conds[cond[i]]].append(g.shape == gts[st[i]].shape); ex[conds[cond[i]]].append(g.shape == gts[st[i]].shape and np.array_equal(g, gts[st[i]]))
        lines.append(f"  decoded shape correct at k={k}: " + ", ".join(f"{c} {np.mean(sh[c]):.3f}" for c in conds) + " | exact: " + ", ".join(f"{c} {np.mean(ex[c]):.3f}" for c in conds))
    tin = json.loads(str(d["state_tinfo"])); cin = json.loads(str(d["state_cinfo"])); dirs = d["state_dir"]
    dn = np.array([np.mean([c["dnorm"] / c["hnorm"] for c in x]) for x in cin])
    lines.append(f"  magnitudes: transplant answer {np.mean([x['rel_change_answer'] for x in tin]):.3f}, prefix {np.mean([x['rel_change_prefix'] for x in tin]):.3f}; correction ||delta||/||h|| fixed {dn[dirs == 0].mean():.3f}, Gaussian {dn[dirs == 1].mean():.3f}")
    (base / f"report_{tag}.txt").write_text("\n".join(lines) + "\n"); (base / f"report_{tag}.json").write_text(json.dumps(res, indent=1, default=float))
    print("\n".join(lines) if not quiet else f"report written ({len(lines)} lines; output suppressed)")


LAUNCH_MDE = 0.15                               # the PI's third review: the frozen launch rule (Note, "Launch rule")


def calibration_report(base=OUT, tag="calib", quiet=False):
    """The control-only precision calibration on the pilot's eligible queries (|E| = 4, k = 1): two independent matched controls, no donor
    transplant. Measured clustered and jackknife SEs of the control-vs-intact and control-vs-control coefficients, projected to the confirmation;
    the frozen launch rule applied to option B (the standardized margin); raw margin and correctness reported beside it."""
    d = load(tag, base); conds = [str(x) for x in d["conds"]]; ix, ja, jb = conds.index("intact"), conds.index("full_ctrl"), conds.index("ctrl_b")
    don = donor_map(d); task = d["state_task"]; tcl = np.unique(task, return_inverse=True)[1]; gates = json.loads(str(d["gates"]))
    pj = json.loads((base / "pilot_report.json").read_text())["eligibility"]["projection"]["4"]; ns = {"central": pj["projected"], "low": pj["lo"], "high": pj["hi"]}
    # gate: the re-rolled intact rollouts reproduce the pilot's bitwise
    p_ = load("pilot_E4", base); pk = {}
    for i in range(len(p_["var_state"])):
        s_ = int(p_["var_state"][i]); pk[(int(p_["state_qi"][s_]), int(p_["state_dir"][s_]), int(p_["var_arm"][i]), int(p_["var_dose"][i]), int(p_["var_mask"][i]))] = i
    rep_ok, rep_n = True, 0
    for i in range(len(d["var_state"])):
        if int(d["var_cond"][i]) != ix: continue
        s_ = int(d["var_state"][i]); j = pk.get((int(d["state_qi"][s_]), int(d["state_dir"][s_]), int(d["var_arm"][i]), int(d["var_dose"][i]), int(d["var_mask"][i])))
        rep_n += 1; rep_ok &= j is not None and np.array_equal(d["disp"][i], p_["disp"][j]) and np.array_equal(d["elog"][i], p_["elog"][j])
    gates_ok = all(not k_.endswith("_flips") or g[k_] == 0 for g in gates for k_ in g) and rep_ok
    res = dict(states=int(len(d["state_qi"])), queries=int(len(np.unique(d["state_qi"]))), intact_reproduces_pilot=bool(rep_ok), intact_variants_compared=rep_n,
               excluded_flip_queries=sum(g.get("excluded_flip_queries", 0) for g in gates), projection_n=ns, gates_max={k_: max(g.get(k_, 0) for g in gates) for k_ in gates[0] if k_.endswith("maxdev")})
    lines = [f"P29 CONTROL-ONLY PRECISION CALIBRATION (pilot queries, |E| = 4, k = 1; two matched controls, no donor transplant; {N_BOOT} task-clustered draws)",
             f"queries {res['queries']} ({res['states']} pair-directions); intact re-roll reproduces the pilot bitwise: {rep_ok} ({rep_n} variants); flip-excluded queries {res['excluded_flip_queries']}; executed maxima {res['gates_max']}"]
    fmt = lambda e, c: f"{e:+.3f} [{c[0]:+.3f}, {c[1]:+.3f}]"; sub = lambda u, m: {k_: v[m] for k_, v in u.items()}; sew = lambda c: (c[1] - c[0]) / 3.92
    for mi, mz in enumerate(("margin_std", "margin", "correct")):
        Y, Bm, dk, exc = tables(d, mz, 4, 1, don); uA, uB = centered_units(Y, dk, ja, ix, ix), centered_units(Y, dk, jb, ix, ix); cl = tcl[uA["state"]]; tk = task[uA["state"]]
        npil = int(len(np.unique(d["state_qi"][uA["state"]]))); r = dict(excluded_small_scale=sorted(exc), queries_used=npil)
        for nm, (fn, lists, sg) in dict(A=(lambda m: beta(sub(uA, m)), [uA], [1.0]), B=(lambda m: beta(sub(uB, m)), [uB], [1.0]),
                                        D=(lambda m: beta(sub(uA, m)) - beta(sub(uB, m)), [uA, uB], [1.0, -1.0])).items():
            dr = boot_betas(lists, sg, [cl] * len(lists), [SEED, 29, 30, mi, ord(nm)], N_BOOT); c_ = ci(dr); sj, th, nu = jackknife_se(fn, tk)
            r[nm] = dict(E=fn(np.ones(len(tk), bool)), ci=c_, se_boot=sew(c_), se_jack=sj, unidentified_draws=n_unidentified(dr), jack_unidentified=nu, se=max(sew(c_), sj))
        seA = max(r["A"]["se"], r["B"]["se"]); seD = r["D"]["se"]
        r["mde"] = {k_: dict(A=MDE_Z * seA * np.sqrt(npil / n), D=MDE_Z * seD * np.sqrt(npil / n)) for k_, n in ns.items()}
        thD = jackknife_se(lambda m: beta(sub(uA, m)) - beta(sub(uB, m)), tk)[1]; ut = np.unique(tk); inf_ = (thD - r["D"]["E"]) ** 2
        top = str(ut[int(np.nanargmax(inf_))]); m_ = tk != top; oD = ci(boot_betas([sub(uA, m_), sub(uB, m_)], [1.0, -1.0], [cl[m_], cl[m_]], [SEED, 29, 31, mi], N_BOOT))
        r["influence"] = dict(top_task=top, top_share=float(np.nanmax(inf_) / np.nansum(inf_)), D_without_top=float(beta(sub(uA, m_)) - beta(sub(uB, m_))), ci_without_top=oD)
        rr = {}
        for c, j in (("intact", ix), ("full_ctrl", ja), ("ctrl_b", jb)):
            v = [np.sqrt(np.mean([(Y[(s_, j, i, m)] - np.mean([Y[(s_, j, i, mm)] for mm in range(MASKS)])) ** 2 for i in range(len(DOSES)) for m in range(MASKS)])) for s_ in sorted(dk) if dk[s_] is not None]
            rr[c] = dict(median=float(np.median(v)), p90=float(np.percentile(v, 90)), max=float(np.max(v)))
        r["response_rms"] = rr; res[mz] = r
        lines.append(f"  {mz:10s}: queries used {npil} (small-scale excluded {len(exc)}) | beta(full_ctrl) {fmt(r['A']['E'], r['A']['ci'])} SE boot {r['A']['se_boot']:.3f} jack {r['A']['se_jack']:.3f} | "
                     f"beta(ctrl_b) {fmt(r['B']['E'], r['B']['ci'])} SE boot {r['B']['se_boot']:.3f} jack {r['B']['se_jack']:.3f} | control - control {fmt(r['D']['E'], r['D']['ci'])} SE boot {r['D']['se_boot']:.3f} jack {r['D']['se_jack']:.3f}")
        lines.append(f"      projected MDE (control-vs-intact, control-vs-control) at n = {ns['central']:.0f}: {r['mde']['central']['A']:.3f}, {r['mde']['central']['D']:.3f}; at n = {ns['low']:.0f}: {r['mde']['low']['A']:.3f}, {r['mde']['low']['D']:.3f}; "
                     f"at n = {ns['high']:.0f}: {r['mde']['high']['A']:.3f}, {r['mde']['high']['D']:.3f} | influence: top task {top} {r['influence']['top_share']:.3f} of the jackknife influence, D without it {fmt(r['influence']['D_without_top'], oD)}")
        lines.append(f"      centered response RMS per pair-direction (units of the measure): " + "; ".join(f"{c} median {v['median']:.3f}, p90 {v['p90']:.3f}, max {v['max']:.3f}" for c, v in rr.items()))
    b = res["margin_std"]; crit = {"1 MDE control-vs-control <= 0.15": b["mde"]["central"]["D"] <= LAUNCH_MDE, "2 MDE control-vs-intact <= 0.15": b["mde"]["central"]["A"] <= LAUNCH_MDE,
                                   "3 control-vs-control not resolved": b["D"]["ci"][0] <= 0 <= b["D"]["ci"][1], "4 small-scale exclusions < 20 %": len(b["excluded_small_scale"]) < 0.2 * res["queries"],
                                   "5 gates and bitwise intact reproduction": bool(gates_ok)}
    res["launch_rule"] = crit; res["LAUNCH"] = all(crit.values())
    lines.append("LAUNCH RULE (frozen before this calibration; option B): " + "; ".join(f"{k_}: {v}" for k_, v in crit.items()) + f" -> {'LAUNCH B' if res['LAUNCH'] else 'DO NOT LAUNCH (to the PI)'}")
    (base / f"report_{tag}.txt").write_text("\n".join(lines) + "\n"); (base / f"report_{tag}.json").write_text(json.dumps(res, indent=1, default=float))
    print("\n".join(lines) if not quiet else f"report written ({len(lines)} lines; output suppressed)")


def reader_selftest():
    """Synthetic confirmation file: Y_full = Y_intact + 0.4 * donor's intact (centered per state-dose), Y_ctrl = Y_intact + noise. The reader must
    recover beta(full) ~ 0.4, beta(ctrl) ~ 0, letter S-ARC; and NOT-RESOLVED when nothing is planted."""
    import tempfile
    for plant, want in ((0.4, "S-ARC"), (0.0, "NOT-RESOLVED")):
        rng = np.random.default_rng(7); nq, e = 60, 4; conds = ["intact", "full", "full_ctrl"]; S_ = 2 * nq
        Yi = rng.normal(size=(S_, 3, 4)); bi = rng.normal(size=S_); don = lambda s: s ^ 1
        V = []
        for s in range(S_):
            for c in range(3):
                V.append((s, c, 0, -1, -1, bi[s] + (0.1 * rng.normal() if c else 0.0)))
                for i in range(3):
                    for m in range(4):
                        y = Yi[s, i, m] + (plant * Yi[don(s), i, m] + 0.1 * rng.normal() if c == 1 else (0.1 * rng.normal() if c == 2 else 0.0)); V.append((s, c, 1, i, m, y))
        lab = np.full((S_, NPOS), 3, np.uint8); lab[:, 30:] = 0; Et = np.tile(np.arange(e), (S_, 1)).astype(np.int16)
        elog = np.zeros((len(V), K_MAX, max(E_CAND), VOCAB), np.float32)
        for q, v in enumerate(V): elog[q, :, :e, 3] = v[5]
        disp = np.zeros((len(V), K_MAX, NPOS), np.uint8); disp[:, :, :30] = 3
        masks = [[[i, m, [10 + m]] for i in range(3) for m in range(4)] for _ in range(S_)]
        dd = np.full((S_, NPOS), 0, np.uint8); dd[:, :30] = 4
        with tempfile.TemporaryDirectory() as td:
            np.savez_compressed(Path(td) / "p29_conf.npz", conds=np.array(conds), gates=json.dumps([{}]), screen=json.dumps([{}] * nq), state_qi=np.repeat(np.arange(nq), 2).astype(np.int32),
                                state_task=np.array([f"t{q // 2}" for q in range(nq) for _ in range(2)]), state_dir=np.tile([0, 1], nq).astype(np.int8), state_Etok=Et, state_W=np.full(S_, 20, np.int16),
                                state_lab=lab, state_disp_d1=dd, state_masks=json.dumps(masks), state_tinfo=json.dumps([dict(rel_change_answer=1.0, rel_change_prefix=1.0)] * S_),
                                state_cinfo=json.dumps([[dict(dnorm=1.0, hnorm=1.0)]] * S_), var_state=np.array([v[0] for v in V], np.int32), var_cond=np.array([v[1] for v in V], np.int8),
                                var_arm=np.array([v[2] for v in V], np.int8), var_dose=np.array([v[3] for v in V], np.int8), var_mask=np.array([v[4] for v in V], np.int8), disp=disp, elog=elog)
            old = dict(FROZEN); FROZEN.update(E=e, measure="margin"); nb = globals()["N_BOOT"]; globals()["N_BOOT"] = 300
            try: report(Path(td), "conf", quiet=True); res = json.loads((Path(td) / "report_conf.json").read_text())
            finally: FROZEN.clear(); FROZEN.update(old); globals()["N_BOOT"] = nb
        r1 = res["margin|k1"]; assert abs(r1["beta_full"] - plant) < 0.05 and abs(r1["beta_ctrl"]) < 0.05 and res["S_ARC"] == want, (plant, r1, res["S_ARC"])
    print("reader selftest OK (planted 0.4 recovered; null not resolved)")


def selftest():
    rng = np.random.default_rng(0); W = rng.standard_normal((VOCAB, 64)); N, Wp = nullspace_projector(W)
    assert np.abs(W @ N).max() < 1e-10 and np.abs(N @ N - N).max() < 1e-10 and abs(np.trace(N) - (64 - VOCAB)) < 1e-8
    hr, hd = rng.standard_normal((40, 64)) * 3, rng.standard_normal((40, 64)) * 3; P = 5
    full, ctrl, inf = transplant_states(hr, hd, N, P, np.random.default_rng(1))
    assert np.abs(full[P:] @ W.T - hr[P:] @ W.T).max() < 1e-10 and np.abs(ctrl[P:] @ W.T - hr[P:] @ W.T).max() < 1e-10       # scores kept exactly
    assert np.abs(full[P:] @ N - hd[P:] @ N).max() < 1e-10 and np.array_equal(full[:P], hd[:P])                             # donor's invisible part
    nr = lambda x: np.linalg.norm(x, axis=-1); pr, pd, pc = hr[P:] @ N, hd[P:] @ N, ctrl[P:] @ N
    assert np.allclose(nr(pc), nr(pd)) and np.allclose((pr * pc).sum(-1), (pr * pd).sum(-1)) and inf["ctrl_in_null"] < 1e-10    # angle + norm matched
    assert np.allclose(nr(ctrl[:P]), nr(hd[:P])) and np.allclose((hr[:P] * ctrl[:P]).sum(-1), (hr[:P] * hd[:P]).sum(-1))
    assert np.abs((pc * pd).sum(-1) / (nr(pc) * nr(pd))).mean() < 0.95                                                     # direction not copied
    full2, ctrl2, _ = transplant_states(hr, hd, N, P, np.random.default_rng(1)); assert np.array_equal(ctrl, ctrl2)        # one draw, reproducible
    h = rng.standard_normal((7, 64)); labs = rng.integers(2, 12, 7); dl, tgt, lg = correction_delta(h, W, Wp, labs)
    assert np.abs((h + dl) @ W.T - tgt).max() < 1e-9 and np.all(((h + dl) @ W.T).argmax(-1) == np.where(lg.argmax(-1) == labs, lg.argmax(-1), labs))
    assert np.abs(dl @ N).max() < 1e-9                                                                                       # minimum norm: no nullspace part
    lab = np.zeros(900, np.int64); lab[:12] = 5; a = lab.copy(); b = lab.copy(); a[:4] = 6; b[2:6] = 7; b[1] = 0
    assert shared_colour_wrong(a, b, lab).tolist() == [2, 3] and structural_errors(b, lab) == (1, 0)
    g = np.zeros(900, np.int64); g[:3] = 4; g[30:33] = 4; g[3] = 1; g[33] = 1; g[60:63] = 1; assert crop(g).shape == (2, 3) and np.all(crop(g) == 2)
    el = rng.standard_normal((3, 5, VOCAB)); lb = rng.integers(0, VOCAB, 5); mg = margins(el, lb)
    for i in range(3):
        for j in range(5): assert np.isclose(mg[i, j], el[i, j, lb[j]] - np.delete(el[i, j], lb[j]).max())
    Wset = np.arange(40); Et, masks = draw_challenges(3, Wset, 16, 16); assert len(Et) == 16 and all(not (set(c.tolist()) & set(Et.tolist())) for _, _, c in masks)
    assert all(len(c) == len(masks[4 * i][2]) for i, _, c in masks) and draw_challenges(3, Wset, 16, 16)[0].tolist() == Et.tolist()
    # the centered statistic recovers a planted donor coefficient; the clustered bootstrap brackets it; P27's definition reproduced
    don = {s: s + 1 if s % 2 == 0 else s - 1 for s in range(240)}; Y = {}
    for s in range(240):
        for i in range(3):
            for m in range(4): Y[(s, 0, i, m)] = rng.normal()
    for s in range(240):
        for i in range(3):
            for m in range(4): Y[(s, 1, i, m)] = Y[(s, 0, i, m)] + 0.3 * Y[(don[s], 0, i, m)] + 0.05 * rng.normal()
    u = centered_units(Y, don, 1, 0); assert abs(beta(u) - 0.3) < 0.02
    cl = u["state"] // 2; dr = boot_betas([u], [1.0], [cl], 1, n=400); lo, hi = ci(dr); assert lo < 0.3 < hi and hi - lo < 0.1
    assert np.allclose(ci(dr), (np.percentile(dr, 2.5), np.percentile(dr, 97.5)))                                         # equals numpy's when all identified
    XX, Xy = cluster_stats(u, cl); assert np.isclose(np.linalg.solve(XX.sum(0), Xy.sum(0))[0], beta(u))
    try:
        sys.path.insert(0, str(ROOT / "tools")); import rebuttal_p27 as P27                                             # cross-check against P27's code
        pids = np.repeat(np.arange(120), 2); r27 = P27.centered_rows(Y, don, ["intact", "full"], pids, "full", "intact"); assert np.isclose(P27.beta(r27), beta(u))
        print("cross-check vs P27 OK")
    except ImportError: print("cross-check vs P27 skipped (P27 not importable in this environment)")
    tos = [f"t{s_ // 2}" for s_ in range(240)]; dg = design_diagnostics(Y, don, tos)
    assert dg["vary_share"] == 1.0 and dg["R2"] < 0.05 and dg["informative"] and dg["tasks_with_info"] == 120 and dg["eff_tasks"] > 60 and dg["loo_uninformative"] == 0
    # the PI's flat-response bug: constant responses must be UNINFORMATIVE, never zero uncertainty
    Yc = {k_: 0.5 for k_ in Y}; dc = design_diagnostics(Yc, don, tos); assert dc["informative"] is False and dc["se_proxy"] is None
    uc = centered_units(Yc, don, 1, 0); assert np.isnan(beta(uc)) and n_unidentified(boot_betas([uc], [1.0], [uc["state"] // 2], 2, n=50)) == 50
    assert ci(boot_betas([uc], [1.0], [uc["state"] // 2], 2, n=50)) == (-np.inf, np.inf)
    # donor responses collinear with the recipient's (identical histories) -> unidentified
    Yd = {(s_, c, i, m): Y[(s_ - s_ % 2, c, i, m)] for (s_, c, i, m) in Y}; assert design_diagnostics(Yd, don, tos)["informative"] is False
    # information concentrated in one task: only puzzle 0's two states vary
    Yk = {(s_, c, i, m): (Y[(s_, c, i, m)] if s_ < 2 else 0.5) for (s_, c, i, m) in Y}; dk = design_diagnostics(Yk, don, tos)
    assert dk["informative"] and dk["tasks_with_info"] == 1 and abs(dk["top_share"] - 1) < 1e-9 and dk["loo_uninformative"] == 1 and dk["loo_se_without_top"] is None
    # partly unidentified bootstrap draws widen the interval and are counted
    dr = np.array([0.1, 0.2, np.nan, 0.3] * 25); assert n_unidentified(dr) == 25 and ci(dr) == (-np.inf, np.inf)
    dr2 = np.r_[np.linspace(0.1, 0.3, 99), np.nan]; lo2, hi2 = ci(dr2); assert lo2 < np.percentile(dr2[:-1], 2.5) + 1e-12 and hi2 >= np.percentile(dr2[:-1], 97.5)
    scr = [dict(task=f"t{i}", solved=(False, False), W=12 if i % 2 else 5) for i in range(40)]; ei = eligibility_interval(scr, 4, 100, n_boot=500)
    assert abs(ei["rate"] - 0.5) < 1e-12 and ei["lo"] < 50 < ei["hi"]
    p, c = task_split([f"t{i}" for i in range(400)]); assert len(p) == N_PILOT_TASKS and not set(p) & set(c) and len(c) == 400 - N_PILOT_TASKS and task_split([f"t{i}" for i in range(400)])[0] == p
    reader_selftest()
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", nargs="?", choices=["pilot", "pilot-report", "calibrate", "calibration-report", "run", "merge", "report"]); ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--threads", type=int, default=6); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--out", default=None); ap.add_argument("--tag", default="pilot"); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(); out = Path(a.out) if a.out else OUT
    if a.selftest: selftest(); return
    if a.action in ("run", "report") and a.tag == "pilot": a.tag = "conf"
    if a.action == "pilot":
        r = Runner(out, a.threads); pilot, _ = task_split([x["task"] for x in r.rows]); q = query_order(r.rows, pilot)
        if a.limit: q = q[:a.limit]
        for e in E_CAND: r.run_blocks(q, ["intact"], e, e, f"{a.tag}_E{e}", a.shard, a.nshard)
    elif a.action == "merge":
        for t_ in ([f"{a.tag}_E{e}" for e in E_CAND] if a.tag in ("pilot", "smoke") else [a.tag]): merge(t_, out)
    elif a.action == "pilot-report": pilot_report(out, a.tag)
    elif a.action == "run":
        assert None not in FROZEN.values(), "settings not frozen"
        r = Runner(out, a.threads); _, conf = task_split([x["task"] for x in r.rows]); q = query_order(r.rows, conf)
        if a.limit: q = q[:a.limit]
        r.run_blocks(q, ["intact", "full", "full_ctrl"], FROZEN["E"], FROZEN["E"], a.tag, a.shard, a.nshard, exclude_flips=True,
                     tol=(FROZEN["tol_transplant"], FROZEN["tol_correction"]))
    elif a.action == "report": report(out, a.tag, a.quiet)
    elif a.action == "calibrate":
        r = Runner(out, a.threads); pilot, _ = task_split([x["task"] for x in r.rows]); q = query_order(r.rows, pilot)
        if a.limit: q = q[:a.limit]
        r.run_blocks(q, ["intact", "full_ctrl", "ctrl_b"], 4, 4, a.tag if a.tag != "pilot" else "calib", a.shard, a.nshard, exclude_flips=True, tol=(TOL_TRANSPLANT, TOL_CORRECTION))
    elif a.action == "calibration-report": calibration_report(out, a.tag if a.tag != "pilot" else "calib", a.quiet)
    else: ap.print_help()


if __name__ == "__main__":
    main()
