"""Independent recount of P28 from the merged p28.npz, the block files and the screen chunks, written apart from the tool's reader: own
coverage tables (dense arrays built from the raw rows), own budget curves, own paired differences, own resampler. Registered point estimates
must agree exactly; intervals close (independent draws). Also re-derives the eligible list from the chunk files (prefix rule), checks every
block's restoration flag and mismatch counts, and with --regen re-runs two eligible puzzles' banks to confirm no readout verifies."""
import json, hashlib, sys
from pathlib import Path
import numpy as np
ROOT = Path("/Users/aakash/Projects/HRRN"); B = ROOT / "runs/analysis/rebuttal_20261014"; sys.path.insert(0, str(ROOT / "tools"))
rep = json.loads((B / "report.json").read_text()); REGEN = "--regen" in sys.argv
PAIRS = [(i, j) for i in range(4) for j in range(4) if i != j]; ARMS = ["lead", "lead_ctrl", "full", "full_ctrl"]; BUD = [1, 2, 4, 8, 12]


def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()


for w in (192, 256):
    d_dir = B / f"attention_{w}"; f = d_dir / "p28.npz"
    assert sha(f) == json.loads(f.with_suffix(".sha256.json").read_text())["sha256"], "npz hash mismatch"
    d = np.load(f, allow_pickle=False); r = rep[f"attention_{w}"]; nP = len(d["pids"])
    # the eligible list from the chunk files (contiguous prefix, first 128 eligible in the frozen order)
    rows = []
    for k in range(16):
        cf = d_dir / "screen_chunks" / f"chunk_{k:03d}.json"
        if not cf.exists() or sum(x["eligible"] for x in rows) >= 128: break
        rows += json.loads(cf.read_text())["rows"]
    elig = [x["pid"] for x in rows if x["eligible"]][:128]
    import rebuttal_p28 as T28
    cand = T28.candidates(w); assert [x["pid"] for x in rows] == cand[:len(rows)].tolist(), "screen order differs from the frozen order"
    print(f"attention_{w}: eligible list re-derived equal {elig == d['pids'].tolist()} ({len(elig)}); screened {len(rows)}")
    blocks = sorted(d_dir.glob("block_[0-9][0-9].npz")); bk = [np.load(b) for b in blocks]
    print(f"   blocks {len(blocks)}: restore_ok all {all(bool(x['restore_ok']) for x in bk)}; verification-vs-reference mismatches {sum(int(x['mism']) + int(x['ext_mism']) for x in bk)}")
    # dense success tables from the raw rows
    S = {a: np.full((nP, 12), -9, np.int64) for a in ARMS}; Rs = np.full((nP, 12), -9, np.int64); SH = np.full((nP, 4), -9, np.int64); EX = np.full((nP, 4), -9, np.int64)
    for kind, p, i, j, fi in zip(d["kind"], d["p"], d["i"], d["j"], d["first"]):
        kind = str(kind)
        if kind in S: S[kind][p, PAIRS.index((int(i), int(j)))] = fi
        elif kind == "restart": Rs[p, i] = fi
        elif kind == "ext1": SH[p, i] = fi
    for p, i, fi in zip(d["ext_p"], d["ext_i"], d["ext_first"]): EX[p, i] = fi
    assert all((S[a] != -9).all() for a in ARMS) and (Rs != -9).all() and (SH != -9).all() and (EX != -9).all(), "missing rows"
    order = np.stack([np.random.default_rng([20261014, int(pid), 3]).permutation(12) for pid in d["pids"]])
    cov = {a: {b: (np.take_along_axis(S[a], order, 1)[:, :b] >= 0).any(1) for b in BUD} for a in ARMS}
    cov["restart"] = {b: (Rs[:, :b] >= 0).any(1) for b in BUD}; cov["extended"] = {b: ((EX >= 0) & (EX < 4 * b)).any(1) for b in BUD}; short = (SH >= 0).any(1)
    pri = {a: np.mean(np.stack([(S[a][:, k] >= 0).astype(float) - (SH[:, PAIRS[k][0]] >= 0).astype(float) for k in range(12)], 1), 1) for a in ARMS}
    def stats(ix):
        m = lambda v: float(np.mean(v[ix])); o = {}
        for a in ARMS:
            o[f"{a}_additional"] = m(cov[a][12] & ~short); o[f"{a}_reverse"] = m(short & ~cov[a][12]); o[f"{a}_per_rollout"] = m(pri[a])
        for a, ctl in (("lead", "lead_ctrl"), ("full", "full_ctrl")):
            for bn in (ctl, "restart", "extended"): o[f"{a}_vs_{bn}"] = m(cov[a][12].astype(float) - cov[bn][12].astype(float))
            o[f"{a}_headroom"] = m(cov[a][12].astype(float) - np.maximum.reduce([cov["restart"][1], cov["extended"][1], cov[a][1]]).astype(float))
            o[f"{a}_unique"] = m(cov[a][12] & ~cov["restart"][12] & ~cov["extended"][12])
        return o
    full = stats(np.arange(nP))
    for k_, v in full.items():
        key, sub = (k_[:-len("_reverse")] + "_additional", "reverse_losses") if k_.endswith("_reverse") else (k_, "E")
        rv = r[key][sub] if key in r else None
        print(f"   {k_}: {v:+.4f} (report {rv:+.4f})" if rv is not None else f"   {k_}: {v:+.4f} (not in report)")
    rng = np.random.default_rng([28, w, 99]); draws = [stats(rng.integers(0, nP, nP)) for _ in range(4000)]
    for k_ in ("lead_additional", "lead_per_rollout", "lead_vs_lead_ctrl", "lead_vs_restart", "lead_vs_extended", "lead_headroom", "full_vs_full_ctrl", "full_vs_restart", "full_vs_extended"):
        lo, hi = np.percentile([x[k_] for x in draws], [2.5, 97.5]); print(f"   own interval {k_}: [{lo:+.3f}, {hi:+.3f}] (report [{r[k_]['ci'][0]:+.3f}, {r[k_]['ci'][1]:+.3f}])")
    print("   coverage at budgets (own): " + " | ".join(f"{a} " + "/".join(f"{float(np.mean(cov[a][b])):.3f}" for b in BUD) for a in cov))
    if REGEN:
        rr = T28.R22(w, Path("/private/tmp/claude-501/-Users-aakash-Projects-HRRN/29fa6425-c970-40e0-b70e-5aa7c3c811d3/scratchpad/recount_p28_runner"), tag="recount")
        b2 = rr.bank(np.array(d["pids"][:2], np.int64)); print(f"   regenerated banks of 2 eligible puzzles: any verified readout {bool(b2['valid'].any())}")
