#!/usr/bin/env python3
# Ledger: THE DISCUSSION-PERIOD EXPERIMENT P15, exploratory linear layer (registration
# Documentation/Note_2026-10-02_Rebuttal_P15_P16_P17_Registration.md, "Exploratory": the linear operator on 16 DISCOVERY states per receiver).
# MEASUREMENT, $0, inference only; imports tools/rebuttal_p15.py (R7, unchanged) for the model, the gradient-stop-free outer iteration and the split.
"""The one-iteration Jacobian of next digit scores with respect to the current scores (pushes through the readout direction, kernel
and fast state fixed), on empty cells, for 16 DISCOVERY states per receiver; and the sensitivity of next scores to equal-norm pushes
in the readout-parallel slow component, the readout-orthogonal slow component and the fast state. Exploratory: no letter."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import rebuttal_p15 as Q
import rebuttal_p1p2 as P

OUT = Q.OUT
N_STATES, JB, N_SENS = 16, 64, 64


def softmax_like(s, kind):
    """Digit-conditional probabilities and their per-cell Jacobian dp/ds (9x9), for 'stablemax' (the training normalization) or 'softmax'."""
    if kind == "softmax":
        e = np.exp(s - s.max()); p = e / e.sum(); return p, np.diag(p) - np.outer(p, p)
    phi = np.where(s >= 0, 1.0 + s, 1.0 / (1.0 - s)); dphi = np.where(s >= 0, 1.0, 1.0 / (1.0 - s) ** 2)
    Z = phi.sum(); p = phi / Z
    J = (np.diag(dphi) * Z - np.outer(phi, dphi)) / Z ** 2          # dp_d/ds_e
    return p, J


def occupancy_jacobian(scores, puz, kind="stablemax"):
    """A = dr/ds over empty-cell digit scores: r = unit-digit occupancy (27 units x 9 digits) minus 1, givens one-hot (constant)."""
    empty = np.where(puz == 0)[0]; col = {int(c): k for k, c in enumerate(empty)}
    U = []
    for r in range(9): U.append([9 * r + c for c in range(9)])
    for c in range(9): U.append([9 * r + c for r in range(9)])
    for b in range(9): U.append([9 * (3 * (b // 3) + a) + 3 * (b % 3) + q for a in range(3) for q in range(3)])
    A = np.zeros((27 * 9, 9 * len(empty)))
    Js = {int(c): softmax_like(scores[c], kind)[1] for c in empty}
    for u, cells in enumerate(U):
        for c in cells:
            if c in col: A[u * 9:(u + 1) * 9, 9 * col[c]:9 * col[c] + 9] += Js[c]
    return A, empty


def measure(width, out=OUT):
    r = Q.R7(width, out); jax, jnp = r.jax, r.jnp
    ATS, STUDY = P.study_paths()
    with np.load(STUDY / "inputs/interventions.npz", allow_pickle=False) as d: ids_all, puz_all, sol_all = d["ids"], d["puz"], d["sol"]
    disc = Q.discovery_ids()
    x, st = r.release_states(puz_all[:128])
    with np.load(STUDY / f"attention_{width}/interventions/intact/batch_0000.npz", allow_pickle=False) as d: assert np.array_equal(st[0]["logits"], d["logits"][0])
    emb = np.asarray(r.emb_b(x)); z1 = st[0]["z"]
    grid = (np.einsum("bfsw,w->bsf", z1[:, 0].astype(np.float64), r.lm).argmax(-1) + 1)
    cand = [b for b in range(128) if int(ids_all[b]) in disc and ((grid[b] != sol_all[b].reshape(81)) & (puz_all[b].reshape(81) == 0)).any()]
    pick = sorted(np.random.default_rng([Q.SEED, 61]).permutation(cand)[:N_STATES].tolist())
    lmj = jnp.asarray(r.lm32); lmu = jnp.asarray(r.lm_unit); outer = r.outer_fn
    res = []
    for b in pick:
        zH, zL, e = jnp.asarray(z1[b, 0]), jnp.asarray(z1[b, 1]), jnp.asarray(emb[b])
        def from_state(dH, dL):
            fH, _ = outer(zH + dH, zL + dL, e); return jnp.einsum("fsw,w->sf", fH, lmj)
        zero = jnp.zeros_like(zH)
        jv = jax.jit(jax.vmap(lambda dH, dL: jax.jvp(from_state, (zero, zero), (dH, dL))[1]))
        empty = np.where(puz_all[b].reshape(81) == 0)[0]
        cols = [(int(c), f) for c in empty for f in range(9)]
        C = np.zeros((len(cols), 81, 9), np.float32); t0 = time.time()
        for c0 in range(0, len(cols), JB):
            chunk = cols[c0:c0 + JB]; pad = JB - len(chunk)
            dH = np.zeros((JB,) + z1.shape[2:], np.float32)
            for k, (c, f) in enumerate(chunk): dH[k, f, c] = r.lm_unit
            C[c0:c0 + len(chunk)] = np.asarray(jv(jnp.asarray(dH), jnp.zeros_like(jnp.asarray(dH))))[:len(chunk)]
        # equal-norm pushes: readout-parallel, readout-orthogonal (slow) and fast, one field at one empty cell, norm 1/|lm|
        rng = np.random.default_rng([Q.SEED, 62, int(ids_all[b])]); nrm = 1.0 / np.linalg.norm(r.lm)
        lmh = r.lm / np.linalg.norm(r.lm); sens = {}
        for kind in ("parallel", "orthogonal", "fast"):
            dH = np.zeros((N_SENS,) + z1.shape[2:], np.float32); dL = np.zeros_like(dH)
            for k in range(N_SENS):
                c = int(rng.choice(empty)); f = int(rng.integers(9))
                if kind == "parallel": v = lmh * rng.choice([-1.0, 1.0])
                else:
                    v = rng.standard_normal(len(lmh)); v -= (v @ lmh) * lmh if kind == "orthogonal" else 0.0; v /= np.linalg.norm(v)
                (dL if kind == "fast" else dH)[k, f, c] = (nrm * v).astype(np.float32)
            R = np.asarray(jv(jnp.asarray(dH), jnp.asarray(dL)))
            sens[kind] = np.linalg.norm(R.reshape(N_SENS, -1), axis=1)
        sc = np.einsum("fsw,w->sf", z1[b, 0].astype(np.float64), r.lm)
        res.append(dict(b=b, pid=int(ids_all[b]), empty=empty, C=C, sens=sens, scores=sc, puz=puz_all[b].reshape(81), sol=sol_all[b].reshape(81), seconds=time.time() - t0))
        r.ATS.log("linear_state", width=width, pid=int(ids_all[b]), cols=len(cols), seconds=round(time.time() - t0, 1))
    dst = out / f"attention_{width}/linear.npz"
    np.savez_compressed(dst, pids=np.array([q["pid"] for q in res]), **{f"C_{k}": q["C"] for k, q in enumerate(res)}, **{f"empty_{k}": q["empty"] for k, q in enumerate(res)},
                        **{f"scores_{k}": q["scores"] for k, q in enumerate(res)}, **{f"puz_{k}": q["puz"] for k, q in enumerate(res)}, **{f"sol_{k}": q["sol"] for k, q in enumerate(res)},
                        **{f"sens_{kind}_{k}": q["sens"][kind] for k, q in enumerate(res) for kind in q["sens"]})
    r.ATS.write_json(dst.with_suffix(".sha256.json"), dict(sha256=r.ATS.sha(dst))); r.ATS.log("linear_complete", width=width, states=len(res))


def report(widths=(256, 192, 128), out=OUT):
    lines = ["P15 EXPLORATORY LINEAR LAYER: one-iteration score Jacobian on 16 DISCOVERY states per receiver (no letter)"]; js = {}
    for w in widths:
        f = out / f"attention_{w}/linear.npz"
        if not f.exists(): lines.append(f"attention_{w}: not run"); continue
        d = np.load(f, allow_pickle=False); n = len(d["pids"]); rows = []
        for k in range(n):
            C, empty, sc, puz = d[f"C_{k}"], d[f"empty_{k}"], d[f"scores_{k}"], d[f"puz_{k}"]
            m = len(empty); Ce = C[:, empty, :].reshape(9 * m, 9 * m).T          # rows: empty-cell outputs, cols: empty-cell inputs
            col_norm = np.linalg.norm(C.reshape(9 * m, -1), axis=1)
            onsite = np.array([np.linalg.norm(C[j, empty[j // 9], :]) for j in range(9 * m)])
            peer = np.array([np.linalg.norm(C[j][P.UNITS[empty[j // 9]]]) for j in range(9 * m)])
            nonpeer = np.sqrt(np.maximum(col_norm ** 2 - onsite ** 2 - peer ** 2, 0))
            A, _ = occupancy_jacobian(sc, puz, "stablemax"); _, sv, Vt = np.linalg.svd(A, full_matrices=True)
            rank = int((sv > sv.max() * 1e-8).sum()); Vr, Vn = Vt[:rank], Vt[rank:]
            Kmat = np.eye(9 * m) - Ce
            g_res = np.einsum("ij,jk,ik->i", Vr, Kmat, Vr); g_null = np.einsum("ij,jk,ik->i", Vn, Kmat, Vn) if len(Vn) else np.array([np.nan])
            c_res = np.einsum("ij,jk,ik->i", Vr, Ce, Vr); c_null = np.einsum("ij,jk,ik->i", Vn, Ce, Vn) if len(Vn) else np.array([np.nan])
            rows.append(dict(m=m, col_norm_med=float(np.median(col_norm)), onsite_share=float(np.median(onsite ** 2 / np.maximum(col_norm ** 2, 1e-30))),
                             peer_share=float(np.median(peer ** 2 / np.maximum(col_norm ** 2, 1e-30))), rank_A=rank, null_dim=int(len(Vn)),
                             gain_res_med=float(np.median(g_res)), gain_null_med=float(np.nanmedian(g_null)), C_res_med=float(np.median(c_res)), C_null_med=float(np.nanmedian(c_null)),
                             sens_parallel=float(np.median(d[f"sens_parallel_{k}"])), sens_orth=float(np.median(d[f"sens_orthogonal_{k}"])), sens_fast=float(np.median(d[f"sens_fast_{k}"]))))
        agg = {key: float(np.median([r_[key] for r_ in rows])) for key in rows[0]}
        js[f"attention_{w}"] = dict(per_state=rows, median=agg)
        lines.append(f"attention_{w} (n={n}): response norm per unit score push {agg['col_norm_med']:.2e}; share on-site {agg['onsite_share']:.2f}, at peers {agg['peer_share']:.2f}; "
                     f"rank A {agg['rank_A']:.0f} of {9*agg['m']:.0f}; diag of C along residual modes {agg['C_res_med']:.2e}, along null modes {agg['C_null_med']:.2e}; "
                     f"next-score sensitivity to equal-norm pushes: readout-parallel {agg['sens_parallel']:.2e}, readout-orthogonal slow {agg['sens_orth']:.2e}, fast {agg['sens_fast']:.2e}")
    (out / "linear_report.txt").write_text("\n".join(lines) + "\n"); (out / "linear_report.json").write_text(json.dumps(js, indent=1, default=float)); print("\n".join(lines))


def selftest():
    s = np.array([3.0, -1.0, 0.5, -4.0, 2.0, -2.0, 0.0, 1.0, -0.5])
    for kind in ("stablemax", "softmax"):
        p, J = softmax_like(s, kind); eps = 1e-6
        Jn = np.stack([(softmax_like(s + eps * np.eye(9)[e], kind)[0] - softmax_like(s - eps * np.eye(9)[e], kind)[0]) / (2 * eps) for e in range(9)], axis=1)
        assert np.allclose(J, Jn, atol=1e-6) and np.isclose(p.sum(), 1.0)
    base = np.array([[(3 * (r % 3) + r // 3 + c) % 9 + 1 for c in range(9)] for r in range(9)]).reshape(81)
    puz = base.copy(); puz[:20] = 0; sc = np.random.default_rng(0).standard_normal((81, 9))
    A, empty = occupancy_jacobian(sc, puz); assert A.shape == (243, 9 * 20) and len(empty) == 20
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("action", nargs="?", choices=["run", "report"]); ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--width", type=int, choices=Q.WIDTHS); a = ap.parse_args()
    if a.selftest: selftest(); return
    if a.action == "run": measure(a.width)
    elif a.action == "report": report()
    else: ap.print_help()


if __name__ == "__main__":
    main()
