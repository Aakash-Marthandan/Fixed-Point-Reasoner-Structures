#!/usr/bin/env python3
# Ledger: THE DISCUSSION-PERIOD ANALYSIS P6 — the common accounting of returned-answer failure (registration
# Documentation/Note_2026-10-01_Rebuttal_P6_Registration.md, written before this build). SAVED RECORDS ONLY; no model is run.
"""P6: for every configuration, U (ever correct in the observed trajectories), C (a correct candidate available to the return rule)
and A (returned correct), with 1 - A = (1 - U) + (U - C) + (C - A). Endpoint-only banks carry U = unmeasured. Every cell is traced
to a record under paper/code/evidence; the manuscript's counts are reproduced as gates before anything is written."""
import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVID = ROOT / "paper/code/evidence"
OUT = ROOT / "runs/analysis/rebuttal_20261001d"
KS = (1, 2, 4, 8, 16, 32, 64, 128)
BANKS = {"trm_baseline": "Our TRM, baseline seed 0 (50k)", "trm_baseline_seed1": "Our TRM, baseline seed 1 (40k)", "trm_anchors": "Our TRM, answer anchors only (42k)",
         "trm_random_init": "Our TRM, random starts only (48k)", "attention_128_k128": "Attention 128, 5,000 puzzles", "attention_192_k128": "Attention 192, 5,000 puzzles",
         "attention_256_k128": "Attention 256, 5,000 puzzles", "attention_256_30k_reference": "Earlier Attention 256, reference (28k)",
         "attention_256_30k_no_random_init_or_anchors": "Earlier Attention 256, no starts or anchors (30k)",
         # the two file names below are the evidence export's; the protocols' checkpoint tags (SA256O at 22k, SA256L at 20k) identify the arms
         "attention_256_30k_optimizer": "Earlier Attention 256, changed batch and rate (22k)", "attention_256_30k_batch_lr": "Earlier Attention 256, no damping or noise (20k)"}


def utc(): return datetime.now(timezone.utc).isoformat()


# ---------------- banks: C_k, A_k (minimum stored residual among the first k, ties -> earliest), vote_k from the protocol ----------------
def bank_curve(ex, res, ks):
    n, K = ex.shape; out = {}
    for k in ks:
        if k > K: continue
        e, r = ex[:, :k].astype(bool), res[:, :k]
        C = int(e.any(1).sum()); j = r.argmin(1); A = int(e[np.arange(n), j].sum())
        out[k] = dict(C=C, A=A, missed=C - A, uncovered=n - C, selected_idx=j)
    return out


def load_bank(name):
    with np.load(EVID / f"selection/{name}.npz", allow_pickle=False) as d:
        ex, res, cold = d["mi_exact_k"], d["mi_resid_k"], d["cold_exact"].astype(bool)
    proto = json.loads((EVID / f"selection/{name}.protocol.json").read_text())
    return ex, res, cold, proto


def unpack_bits(packed, t_total):
    """exact_by_step bytes -> (n, t_total) bool; bit b of byte i is step 8 i + b (zero-based), confirmed against first_exact."""
    bits = np.unpackbits(packed.astype(np.uint8), axis=1, bitorder="little")
    return bits[:, :t_total].astype(bool)


def fixed_start_rows():
    rows = {}
    for w in (128, 192, 256):
        with np.load(EVID / f"benchmark/attention_{w}_d64.npz", allow_pickle=False) as d:
            ex = unpack_bits(d["exact_by_step"], 64); cold = d["cold_exact"].astype(bool); first = d["first_exact"]
        assert np.array_equal(ex[:, -1], cold), "endpoint bit disagrees with cold_exact"
        fe = np.where(ex.any(1), ex.argmax(1), -1); assert np.array_equal(fe[fe >= 0], first[fe >= 0]), "first_exact disagrees with the bits"
        n = len(cold); U = int(ex.any(1).sum()); A = int(cold.sum()); terminal = int((ex.any(1) & ~cold).sum())
        temp = int(((ex[:, :-1] & ~ex[:, 1:]).any(1)).sum())
        rows[f"attention_{w}_fulltest_d64"] = dict(label=f"Attention {w}, full test, one fixed-start trajectory, depth 64", n=n, U=U, C=A, A=A, terminal_losses=terminal, any_loss=temp,
                                                   record=f"benchmark/attention_{w}_d64.npz")
    return rows


def mlp_rows():
    rows = {}
    for step in (46000, 94000):
        with np.load(EVID / f"initialization/s{step:06d}.npz", allow_pickle=False) as d:
            for s in ("cold", "ri", "rifix", "sym", "anchor"):
                if f"{s}_ex" not in d.files: continue
                ex = d[f"{s}_ex"]; U = int(ex.any(0).sum()); A = int(ex[-1].sum())
                rows[f"mlp192_{step}_{s}"] = dict(label=f"MLP 192 at {step // 1000}k, 128 validation puzzles, start '{s}', depth 16", n=int(d["n"]), U=U, C=A, A=A, terminal_losses=U - A,
                                                  record=f"initialization/s{step:06d}.npz")
    return rows


def arc_rows():
    cold = [json.loads(l) for l in (EVID / "arc/cold.jsonl").open()]; rs = [json.loads(l) for l in (EVID / "arc/restarts.jsonl").open()]
    assert len(cold) == len(rs) == 419
    key = lambda r: (r["task"], r["test_idx"], r["aug_id"]); rs_by = {key(r): r for r in rs}
    exb = np.array([r["exact_by_step"] for r in cold], bool); U16 = int(exb.any(1).sum()); A16 = int(exb[:, -1].sum())
    assert all(r["exact"] == bool(exb[i, -1]) for i, r in enumerate(cold))
    rows = {"arc_fixed_d16": dict(label="Released TRM on ARC, 419 queries, fixed start, depth 16", n=419, U=U16, C=A16, A=A16, terminal_losses=U16 - A16, record="arc/cold.jsonl")}
    # the eight-start Gaussian bank at depth 16: C_k and minimum-residual selection among the first k draws
    ks = (1, 2, 4, 8); ex = np.array([[d["exact"] for d in rs_by[key(r)]["draws"]] for r in cold], bool); res = np.array([[d["resid_final"] for d in rs_by[key(r)]["draws"]] for r in cold])
    fixed_ex = np.array([rs_by[key(r)]["fixed_exact"] for r in cold], bool)
    curve = {}
    for k in ks:
        e, rr = ex[:, :k], res[:, :k]; C = int(e.any(1).sum()); j = rr.argmin(1); A = int(e[np.arange(419), j].sum()); curve[k] = dict(C=C, A=A, missed=C - A, uncovered=419 - C)
    union = int((exb.any(1) | ex.any(1)).sum())
    rows["arc_bank_d16"] = dict(label="Released TRM on ARC, eight Gaussian starts, depth 16 endpoints", n=419, U="unmeasured (endpoints)", curve=curve,
                                union_with_fixed_trajectory_readouts=union, fixed_endpoint_exact=int(fixed_ex.sum()), record="arc/restarts.jsonl")
    return rows


def gates(bank_curves, fixed, mlp, arc):
    g = {}
    b = bank_curves["trm_baseline"]; g["baseline k128"] = (b[128]["C"], b[128]["A"], b[128]["missed"]) == (19941, 18496, 1445)
    g["baseline k1"] = (b[1]["C"], b[1]["A"]) == (18425, 18425)
    g["anchors k128"] = (bank_curves["trm_anchors"][128]["C"], bank_curves["trm_anchors"][128]["A"]) == (19972, 7645)
    g["random starts k128"] = (bank_curves["trm_random_init"][128]["C"], bank_curves["trm_random_init"][128]["A"]) == (19663, 19659)
    for w, (gain, loss) in zip((128, 192, 256), ((2, 0), (6, 1), (1, 2))):
        c = bank_curves[f"attention_{w}_k128"]; ex, res, cold, _ = load_bank(f"attention_{w}_k128"); e = ex.astype(bool); n = len(cold)
        s8 = e[np.arange(n), c[8]["selected_idx"]]; s128 = e[np.arange(n), c[128]["selected_idx"]]
        g[f"attention {w} 8->128 gained/lost"] = (int((~s8 & s128).sum()), int((s8 & ~s128).sum())) == (gain, loss)
    for name, miss, unc, step in (("attention_256_30k_reference", 1, 3, "028000"), ("attention_256_30k_no_random_init_or_anchors", 69, 2, "030000"),
                                  ("attention_256_30k_optimizer", 1, 132, "022000"), ("attention_256_30k_batch_lr", 0, 2, "020000")):
        _, _, _, proto = load_bank(name)
        g[f"{name} missed/uncovered k32 and checkpoint"] = (bank_curves[name][32]["missed"], bank_curves[name][32]["uncovered"]) == (miss, unc) and proto["ckpt"].endswith(f"ckpt_{step}.pkl")
    g["full-test terminal losses 0/0/0"] = all(fixed[f"attention_{w}_fulltest_d64"]["terminal_losses"] == 0 for w in (128, 192, 256))
    g["full-test any-loss 0/4/7"] = tuple(fixed[f"attention_{w}_fulltest_d64"]["any_loss"] for w in (128, 192, 256)) == (0, 4, 7)
    g["mlp 94k fixed 68 ever / 11 endpoint"] = (mlp["mlp192_94000_cold"]["U"], mlp["mlp192_94000_cold"]["A"]) == (68, 11)
    g["arc 138 ever / 128 endpoint"] = (arc["arc_fixed_d16"]["U"], arc["arc_fixed_d16"]["A"]) == (138, 128)
    g["arc bank coverage 134/134/135/136, selection 134/133/133/131"] = tuple(arc["arc_bank_d16"]["curve"][k]["C"] for k in (1, 2, 4, 8)) == (134, 134, 135, 136) and tuple(arc["arc_bank_d16"]["curve"][k]["A"] for k in (1, 2, 4, 8)) == (134, 133, 133, 131)
    g["arc union of fixed readouts and bank endpoints 146"] = arc["arc_bank_d16"]["union_with_fixed_trajectory_readouts"] == 146
    return g


def run(out=OUT):
    out.mkdir(parents=True, exist_ok=True)
    bank_curves, bank_rows = {}, {}
    for name, label in BANKS.items():
        ex, res, cold, proto = load_bank(name); K = ex.shape[1]; n = len(cold)
        curve = bank_curve(ex, res, KS); bank_curves[name] = curve
        # the protocol's "vote_at_k" is verifier selection: the fixed start exact OR a verified draw among the first k (eval_sudoku_extreme.py, vote_curve); not a majority vote
        verif = {int(k): round(float(v) * n) for k, v in proto.get("vote_at_k", {}).items()}
        bank_rows[name] = dict(label=label, n=n, K=K, fixed_start_endpoint=int(cold.sum()), U="unmeasured (endpoints only)",
                               curve={k: dict(C=v["C"], A=v["A"], missed=v["missed"], uncovered=v["uncovered"], verifier=verif.get(k)) for k, v in curve.items()},
                               record=f"selection/{name}.npz (+ .protocol.json vote_at_k = verifier selection over the fixed start and the first k draws)")
    fixed = fixed_start_rows(); mlp = mlp_rows(); arc = arc_rows()
    g = gates(bank_curves, fixed, mlp, arc)
    failed = [k for k, v in g.items() if not v]
    assert not failed, f"GATE FAILED: {failed}"
    acc = dict(created=utc(), identity="1 - A = (1 - U) + (U - C) + (C - A); U unmeasured where only endpoints were saved", gates=g, banks=bank_rows, fixed_start=fixed, mlp=mlp, arc=arc)
    (out / "accounting.json").write_text(json.dumps(acc, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    L = ["# P6 — the common accounting (saved records; every gate reproduced)", "", "Identity: 1 − A = (1 − U) + (U − C) + (C − A). Fractions of the configuration's problems. 'unmeasured' = only endpoints were saved.", ""]
    L.append("## Fixed-start trajectories (one candidate: C = A)"); L.append("| configuration | n | 1 − U | U − C | C − A | 1 − A | any temporary loss |"); L.append("|---|---|---|---|---|---|---|")
    for key, r in {**fixed, **mlp, **arc}.items():
        if "curve" in r: continue
        n = r["n"]; L.append(f"| {r['label']} | {n} | {100*(n-r['U'])/n:.2f} % | {100*(r['U']-r['C'])/n:.2f} % | 0 | {100*(n-r['A'])/n:.2f} % | {r.get('any_loss', '—')} |")
    L.append(""); L.append("## Candidate banks (U unmeasured; the residual selector's gap C − A; beside it the verifier selector, which returns the fixed start if exact or else any draw among the first k that passes the constraint check)")
    L.append("| configuration | k | 1 − C | C − A | 1 − A | verifier: 1 − A_verif |"); L.append("|---|---|---|---|---|---|")
    for name, r in bank_rows.items():
        n = r["n"]
        for k in (1, 8, 32, 128):
            if k not in r["curve"]: continue
            c = r["curve"][k]; v = c["verifier"]; L.append(f"| {r['label']} | {k} | {100*c['uncovered']/n:.2f} % | {100*c['missed']/n:.2f} % | {100*(n-c['A'])/n:.2f} % | {('%.2f %%' % (100*(n-v)/n)) if v is not None else '—'} |")
    a = arc["arc_bank_d16"]; L.append(""); L.append("## ARC bank (eight Gaussian starts, depth 16)"); L.append("| k | 1 − C | C − A | 1 − A |"); L.append("|---|---|---|---|")
    for k, c in a["curve"].items(): L.append(f"| {k} | {100*c['uncovered']/419:.2f} % | {100*c['missed']/419:.2f} % | {100*(419-c['A'])/419:.2f} % |")
    L.append(f"\nUnion of the fixed trajectory's 16 readouts and the eight endpoints: {a['union_with_fixed_trajectory_readouts']} of 419 covered.")
    L.append(f"\nGates ({len(g)}): all reproduced.")
    (out / "accounting.md").write_text("\n".join(L) + "\n"); print("\n".join(L))


def selftest():
    ex = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0], [1, 1, 1]]); res = np.array([[0.5, 0.1, 0.9], [0.2, 0.2, 0.1], [0.3, 0.1, 0.2], [0.4, 0.4, 0.4]])
    c = bank_curve(ex, res, (1, 2, 3))
    assert (c[1]["C"], c[1]["A"]) == (2, 2) and (c[2]["C"], c[2]["A"]) == (3, 1) and (c[3]["C"], c[3]["A"]) == (3, 1)   # ties -> earliest (row 1 at k=2 picks draw 0, wrong)
    for k in c: assert c[k]["A"] <= c[k]["C"] <= 4 and c[k]["missed"] + c[k]["uncovered"] + c[k]["A"] == 4
    packed = np.array([[128, 255], [240, 0], [0, 0]], np.uint8); b = unpack_bits(packed, 16)
    assert b[0].tolist() == [False] * 7 + [True] + [True] * 8 and b[1].tolist() == [False] * 4 + [True] * 4 + [False] * 8 and not b[2].any()
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("action", nargs="?", choices=["run"]); ap.add_argument("--selftest", action="store_true"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.selftest: selftest(); return
    if a.action == "run": run(Path(a.out) if a.out else OUT)
    else: ap.print_help()


if __name__ == "__main__":
    main()
