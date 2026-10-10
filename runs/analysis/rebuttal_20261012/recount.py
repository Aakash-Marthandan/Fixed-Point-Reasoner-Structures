"""Independent recount of P26 from p26_{D,C}.npz, written apart from the tool's reader: own unit tables (arrays, not dicts), own design
matrices and least squares (QR), own contrasts, own selection score; the basis re-derived from regenerated states with numpy's SVD and compared
as a subspace; the stored retained energies recomputed for the confirmation states. Point estimates must agree exactly (to float rounding);
bootstrap intervals are recomputed with an independent resampler (close, not bitwise)."""
import json, hashlib, sys
from pathlib import Path
import numpy as np
ROOT = Path("/Users/aakash/Projects/HRRN"); B = ROOT / "runs/analysis/rebuttal_20261012"
sys.path.insert(0, str(ROOT / "tools"))
REGEN = "--regen" in sys.argv; SMOKE = "--smoke" in sys.argv                     # --smoke DIR: a code check on the smoke files (already seen)
if SMOKE: B = Path(sys.argv[sys.argv.index("--smoke") + 1])
SFX = "_smoke" if SMOKE else ""
rep = json.loads((B / f"report{SFX}.json").read_text())


def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()


def qr_ls(y, cols):
    X = np.column_stack(list(cols) + [np.ones(len(y))]); Q, R = np.linalg.qr(X)
    try: return np.linalg.solve(R, Q.T @ y)
    except np.linalg.LinAlgError: return np.linalg.lstsq(X, y, rcond=None)[0]           # a degenerate bootstrap draw (tiny samples only)


def tables(d):
    S, C = len(d["state_pid"]), len(d["conds"]); Y = np.full((S, C, 3, 4), np.nan); Bm = np.full((S, C), np.nan); NE = np.zeros((S, 3, 4), int)
    hit = (d["disp_e"][:, 0, :] == d["state_sol"][d["var_state"][:, None], d["state_E"][d["var_state"]]]).mean(1)
    for i in range(len(hit)):
        s, c, a = d["var_state"][i], d["var_cond"][i], d["var_arm"][i]
        if a == 0: Bm[s, c] = hit[i]
        else: Y[s, c, d["var_dose"][i], d["var_mask"][i]] = hit[i]; NE[s, d["var_dose"][i], d["var_mask"][i]] = d["var_nedit"][i]
    assert not np.isnan(Y).any() and not np.isnan(Bm).any()
    don = np.array([np.where((d["state_pid"] == d["state_pid"][s]) & (d["state_dir"] == 1 - d["state_dir"][s]))[0][0] for s in range(S)])
    assert np.array_equal(d["state_nwrong_donor"], d["state_nwrong"][don]), "donor wrong-cell counts disagree with the reciprocal states"
    return Y, Bm, NE, don


def baseline(d, Y, Bm, don, c, rows=None):
    conds = [str(x) for x in d["conds"]]; s = np.arange(len(don)) if rows is None else rows
    x = Bm[don[s], 0] - Bm[s, 0]; y = Bm[s, conds.index(c)] - Bm[s, 0]; return qr_ls(y, [x])[0]


def r1(d, Y, Bm, NE, don, c, rows=None):
    conds = [str(x) for x in d["conds"]]; j = conds.index(c); s = np.arange(len(don)) if rows is None else rows
    ss = np.repeat(s, 12); ii = np.tile(np.repeat(np.arange(3), 4), len(s)); mm = np.tile(np.arange(4), 3 * len(s))
    fR = NE[ss, ii, mm] / d["state_nwrong"][ss]; fD = NE[ss, ii, mm] / d["state_nwrong"][don[ss]]
    bX, bR, bD = Bm[ss, j], Bm[ss, 0], Bm[don[ss], 0]
    y = Y[ss, j, ii, mm] - bX
    return qr_ls(y, [Y[don[ss], 0, ii, mm] - bD, Y[ss, 0, ii, mm] - bR, bX, bR, bD, fR, fD, fR ** 2, fD ** 2, bX * fR, bR * fR, bD * fD])[0]


def r2(d, Y, don, cx, cr, rows=None):
    conds = [str(x) for x in d["conds"]]; s = np.arange(len(don)) if rows is None else rows
    con = lambda A: np.concatenate([A[..., 0] - A[..., 1], A[..., 2] - A[..., 3]], -1).ravel()        # per state: 3 doses x 2 contrasts
    dep = con(Y[s, conds.index(cx)]) - con(Y[s, conds.index(cr)]); dD = con(Y[don[s], 0]); dR = con(Y[s, 0])
    return qr_ls(dep, [dD, dR])[0]


def ci(fn, d, seed, n=1000):
    pid = d["state_pid"]; up = np.unique(pid); groups = {p: np.where(pid == p)[0] for p in up}; rng = np.random.default_rng(seed); v = []
    for _ in range(n):
        rows = np.concatenate([groups[p] for p in rng.choice(up, len(up))]); v.append(fn(rows))
    return np.percentile(v, [2.5, 97.5])


for w in (192, 256):
    for st in ("D", "C"):
        f = B / f"attention_{w}/p26_{st}{SFX}.npz"
        if not f.exists(): print(w, st, "not run"); continue
        if not SMOKE: assert sha(f) == json.loads(f.with_suffix(".sha256.json").read_text())["sha256"], "npz hash mismatch"
        d = np.load(f, allow_pickle=False); r = rep[f"attention_{w}_{st}"]; Y, Bm, NE, don = tables(d); S = len(don)
        print(f"attention_{w} stage {st}: states {S} (pairs {S // 2}), variants {len(d['var_state'])}, gates {json.loads(str(d['gates']))}")
        bd = baseline(d, Y, Bm, don, "full") - baseline(d, Y, Bm, don, "full_random")
        e1 = r1(d, Y, Bm, NE, don, "full") - r1(d, Y, Bm, NE, don, "full_random"); b2 = r2(d, Y, don, "full", "full_random")
        print(f"   BASELINE diff {bd:+.6f} (report {r['BASELINE']['diff']:+.6f}) | R1 diff {e1:+.6f} (report {r['R1']['diff']:+.6f}) | R2 beta {b2:+.6f} (report {r['R2']['beta']:+.6f})")
        print(f"   own intervals: BASELINE {ci(lambda q: baseline(d, Y, Bm, don, 'full', q) - baseline(d, Y, Bm, don, 'full_random', q), d, 11).round(3)} (report {np.round(r['BASELINE']['ci'], 3)}) | "
              f"R1 {ci(lambda q: r1(d, Y, Bm, NE, don, 'full', q) - r1(d, Y, Bm, NE, don, 'full_random', q), d, 12).round(3)} (report {np.round(r['R1']['ci'], 3)}) | "
              f"R2 {ci(lambda q: r2(d, Y, don, 'full', 'full_random', q), d, 13).round(3)} (report {np.round(r['R2']['ci'], 3)})")
        if st == "C":
            ec = r2(d, Y, don, "global", "global_rand"); hd = ec - 0.5 * b2; ea = r2(d, Y, don, "adaptive", "adaptive_rand")
            print(f"   R3 E_compact {ec:+.6f} (report {r['R3']['E_compact']:+.6f}) | half {hd:+.6f} (report {r['R3']['half_diff']:+.6f}) | adaptive {ea:+.6f} (report {r['R3']['adaptive_secondary']:+.6f})")
            print(f"   own intervals: E_compact {ci(lambda q: r2(d, Y, don, 'global', 'global_rand', q), d, 14).round(3)} (report {np.round(r['R3']['E_compact_ci'], 3)}) | "
                  f"half {ci(lambda q: r2(d, Y, don, 'global', 'global_rand', q) - 0.5 * r2(d, Y, don, 'full', 'full_random', q), d, 15).round(3)} (report {np.round(r['R3']['half_ci'], 3)})")
        if st == "D":
            # the identification procedure, re-derived: own score, own ranking, compared with the frozen basis file
            con = lambda A: np.concatenate([A[..., 0] - A[..., 1], A[..., 2] - A[..., 3]], -1)
            sc = ((con(Y[:, 1]) - con(Y[:, 2])) * con(Y[don, 0])).reshape(S, -1).mean(1)
            order = sorted(range(S), key=lambda s: (-sc[s], s)); keep = sorted(order[:int(round(0.5 * S))])
            bf = (B / f"p26_basis_{w}_smoke.json") if SMOKE else ROOT / f"tools/p26_basis_{w}.json"; Bj = json.loads(bf.read_text())
            print(f"   basis {bf.name} sha {sha(bf)[:12]}: stage-D sha matches {Bj['stage_D_sha256'] == sha(f)}; selected states identical {keep == Bj['selected_states']}")
            if REGEN:
                import rebuttal_p26 as T, rebuttal_p1p2 as P
                rr = T.R20p26(w, Path("/private/tmp/claude-501/-Users-aakash-Projects-HRRN/29fa6425-c970-40e0-b70e-5aa7c3c811d3/scratchpad/recount_p26_runner")); prs, _ = rr.pairs(SMOKE)
                key = {p["pid"]: p for p in prs}; rows = []
                for s in keep:
                    p = key[int(d["state_pid"][s])]; a, b_ = (0, 1) if d["state_dir"][s] == 0 else (1, 0)
                    D = P.project(p["h"][b_].astype(np.float64), rr.lm)[1] - P.project(p["h"][a].astype(np.float64), rr.lm)[1]; rows.append(D.reshape(-1, w))
                X = np.concatenate(rows); v = rr.lm / np.linalg.norm(rr.lm); X = X - np.outer(X @ v, v)
                Vt = np.linalg.svd(X, full_matrices=False)[2]; U2 = Vt[: w // 8].T; U2 = U2 - np.outer(v, v @ U2); U2, _ = np.linalg.qr(U2); U = np.array(Bj["U"])
                print(f"   basis subspace re-derived: ||UU' - U2U2'||_F = {np.linalg.norm(U @ U.T - U2 @ U2.T):.2e}")
                fc = B / f"attention_{w}/p26_C{SFX}.npz"
                if fc.exists():
                    dc = np.load(fc, allow_pickle=False); en = json.loads(str(dc["energies"])); dev = 0.0
                    for s in range(len(dc["state_pid"])):
                        p = key[int(dc["state_pid"][s])]; a, b_ = (0, 1) if dc["state_dir"][s] == 0 else (1, 0)
                        D = P.project(p["h"][b_].astype(np.float64), rr.lm)[1] - P.project(p["h"][a].astype(np.float64), rr.lm)[1]
                        Dg = (D @ U) @ U.T; dev = max(dev, abs((Dg ** 2).sum() / (D ** 2).sum() - en[s]["energy_global"]))
                    print(f"   stage-C retained energy recomputed for {len(dc['state_pid'])} states: max deviation {dev:.2e}")
