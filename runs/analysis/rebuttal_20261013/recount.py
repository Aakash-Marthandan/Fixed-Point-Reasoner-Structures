"""Independent recount of P27 from p27.npz, written apart from the tool's reader: own unit tables (arrays), own four-mask centering over whole
arrays, own QR least squares, own donor mapping; every registered point estimate must agree exactly (to float rounding); intervals recomputed
with an independent resampler (close, not bitwise). With --regen: the first six pool puzzles' pairs are regenerated and their leading /
remainder energies recomputed with numpy's own projections and compared with the stored values."""
import json, hashlib, sys
from pathlib import Path
import numpy as np
ROOT = Path("/Users/aakash/Projects/HRRN"); B = ROOT / "runs/analysis/rebuttal_20261013"; sys.path.insert(0, str(ROOT / "tools"))
rep = json.loads((B / "report.json").read_text()); REGEN = "--regen" in sys.argv


def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()


def qr_ls(y, cols):
    X = np.column_stack(list(cols) + [np.ones(len(y))]); Q, R = np.linalg.qr(X)
    try: return np.linalg.solve(R, Q.T @ y)
    except np.linalg.LinAlgError: return np.linalg.lstsq(X, y, rcond=None)[0]


def load(w):
    f = B / f"attention_{w}/p27.npz"; assert sha(f) == json.loads(f.with_suffix(".sha256.json").read_text())["sha256"], "npz hash mismatch"
    d = np.load(f, allow_pickle=False); conds = [str(x) for x in d["conds"]]; S, C = len(d["state_pid"]), len(conds)
    hit = (d["disp_e"][:, 0, :] == d["state_sol"][d["var_state"][:, None], d["state_E"][d["var_state"]]]).mean(1)
    Y = np.full((S, C, 3, 4), np.nan)
    for i in np.where(d["var_arm"] == 1)[0]: Y[d["var_state"][i], d["var_cond"][i], d["var_dose"][i], d["var_mask"][i]] = hit[i]
    assert not np.isnan(Y).any()
    don = np.array([np.where((d["state_pid"] == d["state_pid"][s]) & (d["state_dir"] == 1 - d["state_dir"][s]))[0][0] for s in range(S)])
    return d, conds, Y, don


def coef(Y, don, conds, cx, cref, rows=None):
    s = np.arange(len(don)) if rows is None else rows; Yc = Y - Y.mean(-1, keepdims=True); j = conds.index
    dep = (Yc[s, j(cx)] - Yc[s, j(cref)]).ravel(); dD = Yc[don[s], 0].ravel(); dR = Yc[s, 0].ravel()
    return qr_ls(dep, [dD, dR])[0]


def stats(Y, don, conds, rows=None):
    b = lambda cx, cr="intact": coef(Y, don, conds, cx, cr, rows)
    out = dict(S=b("full", "full_random"), E_L=b("lead", "lead_ctrl"), E_M=b("rem", "rem_ctrl"), E_total=b("full", "comb_ctrl"),
               direct_full_minus_L=b("full", "lead"), direct_full_minus_M=b("full", "rem"))
    out["half_L"] = out["E_L"] - 0.5 * out["E_total"]; out["half_M"] = out["E_M"] - 0.5 * out["E_total"]
    out["I_signal"] = b("full") - b("lead") - b("rem"); out["I_control"] = b("comb_ctrl") - b("lead_ctrl") - b("rem_ctrl"); out["I_diff"] = out["I_signal"] - out["I_control"]
    return out


for w in (192, 256):
    d, conds, Y, don = load(w); r = rep[f"attention_{w}"]; st = stats(Y, don, conds)
    print(f"attention_{w}: states {len(don)}, variants {len(d['var_state'])}, gates {json.loads(str(d['gates']))}")
    beta_own = {c: coef(Y, don, conds, c, "intact") for c in conds[1:]}
    print("   per-condition: " + ", ".join(f"{c} {beta_own[c]:+.6f} (report {r['beta'][c]:+.6f})" for c in conds[1:]))
    for k_ in ("S", "E_L", "E_M", "E_total", "half_L", "half_M", "direct_full_minus_L", "direct_full_minus_M", "I_signal", "I_control", "I_diff"):
        print(f"   {k_}: {st[k_]:+.6f} (report {r[k_]['E']:+.6f})")
    pid = d["state_pid"]; up = np.unique(pid); groups = {p: np.where(pid == p)[0] for p in up}; rng = np.random.default_rng([27, w, 99]); draws = []
    for _ in range(1000): draws.append(stats(Y, don, conds, np.concatenate([groups[p] for p in rng.choice(up, len(up))])))
    for k_ in ("S", "E_L", "E_M", "E_total", "half_L", "half_M", "I_diff"):
        lo, hi = np.percentile([x[k_] for x in draws], [2.5, 97.5]); print(f"   own interval {k_}: [{lo:+.3f}, {hi:+.3f}] (report [{r[k_]['ci'][0]:+.3f}, {r[k_]['ci'][1]:+.3f}])")
    if REGEN:
        import rebuttal_p27 as T27, rebuttal_p1p2 as P
        rr = T27.R21(w, Path("/private/tmp/claude-501/-Users-aakash-Projects-HRRN/29fa6425-c970-40e0-b70e-5aa7c3c811d3/scratchpad/recount_p27_runner")); prs, _ = rr.pairs(smoke=True)
        U = np.array(json.loads((ROOT / f"tools/p26_basis_{w}.json").read_text())["U"]); info = json.loads(str(d["info"])); key = {(int(p), int(x)): s for s, (p, x) in enumerate(zip(d["state_pid"], d["state_dir"]))}; dev = 0.0; n = 0
        for p in prs:
            for di in (0, 1):
                a, b_ = (0, 1) if di == 0 else (1, 0); D = P.project(p["h"][b_].astype(np.float64), rr.lm)[1] - P.project(p["h"][a].astype(np.float64), rr.lm)[1]
                Lp = np.einsum("fsw,wk,vk->fsv", D, U, U); s = key[(p["pid"], di)]; dev = max(dev, abs((Lp ** 2).sum() / (D ** 2).sum() - info[s]["energy_lead"])); n += 1
        print(f"   regenerated {n} pair-directions: leading-energy max deviation {dev:.2e}")
