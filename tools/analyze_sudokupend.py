#!/usr/bin/env python3
# Ledger: THE PENDING SUDOKU RUNS — the FROZEN analyzer (Documentation/Plan_2026-09-17_Sudoku_Pending_Runs.md §4; locked at the registration
# commit). Registered decision rules only; every number is read from the champion battery's artifacts (tools/chain_champ.sh's layout under
# --root: sxeval_pchamp<arm>/*/summary_all.json, sxscan_pchamp<arm>, filler_sxscan128_pchamp<arm>, filler_sxscan128_pport_eqr,
# pretrainchamp_<arm>/{metrics.jsonl, config.json}); the readers are tools/analyze_paperfinal.py's (imported, not copied). `--selftest`
# runs the rules on hand-built records with known letters. NO edit after the registration commit (the analysis pass diffs this file).
"""
  .venv/bin/python tools/analyze_sudokupend.py --root <stage>/runs [--arm X5] [--out runs/analysis/sudokupend_verdict]
  .venv/bin/python tools/analyze_sudokupend.py --selftest
"""
from __future__ import annotations
import argparse, json, pickle, sys, tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_paperfinal as PF   # Root, jload, exact_mcnemar, paired, selected_exact, spurious, metrics, excursion, argv_of

# ---------- THE REGISTRY (locked at the registration commit) ----------
TRIPLE = ("C5", "C7", "C8")                 # the width-192 symmetric DEC triple (the paper's headline recipe, 50k each, selected at 46k)
R = {
    "params_dec": 789_122,                  # the width-192 DEC's parameter count (dec_cell.init_params at hw 81)
    "params_tol": 0.15,                     # R-SP-1: |n_params(X) / params_dec - 1| <= tol -> MATCHED
    "floor16": 0.0054, "floor64": 0.0018,   # R-SP-2 fallback floors = the triple's half-spreads (Table 1: 95.41 +- 0.54, 99.05 +- 0.18); READ from the root when the triple is present
    "spur_clean": 0.01,                     # R-SP-3: the k32 scan's spurious rate <= .01 -> CLEAN
    "mcnemar_p": 0.01,                      # R-SP-4: AHEAD / BEHIND at p < .01 on the identical 5k, else PARITY
    "onset_pp": 0.02,                       # R-SP-5: the plateau's end = the first grid >= 2 pp under the running maximum for the rest of the run
    "flags": ["--cell", "trm", "--fpa-k", "1", "--trm-ri-sigma", "1.0", "--sudoku-digit-aug"],   # INTEGRITY: the X arm's registered levers
}


def n_params_of(root: PF.Root, arm: str):
    cfg = PF.jload(root.pdir(arm) / "config.json") or {}
    if isinstance(cfg.get("n_params_bulk"), int): return cfg["n_params_bulk"] + int(cfg.get("n_params_table") or 0), "config.json (bulk + the task table)"   # the trainer's keys (the CPU smoke: bulk 795,909, table 32)
    if isinstance(cfg.get("n_params"), int): return cfg["n_params"], "config.json"
    s = root.summ(root.chain(arm, "full_vsel_t16")) or {}
    ck = s.get("ckpt")
    if ck and (root.r.parent / ck).exists():
        obj = pickle.load(open(root.r.parent / ck, "rb"))
        tree = obj.get("state", {}).get("model", obj) if isinstance(obj, dict) else obj
        n = 0
        def walk(x):
            nonlocal n
            if isinstance(x, dict): [walk(v) for v in x.values()]
            elif isinstance(x, (list, tuple)): [walk(v) for v in x]
            elif hasattr(x, "size"): n += int(x.size)
        walk(tree)
        return n, "the selected checkpoint"
    return None, "absent"


def onset(root: PF.Root, arm: str):
    """the plateau's end on the EMA monitor: the first grid >= onset_pp under the running maximum for the rest of the run."""
    mon = sorted({m["step"]: m for m in (r["monitor"] for r in PF.metrics(root, arm) if "monitor" in r)}.values(), key=lambda m: m["step"])
    if not mon: return None
    v = [(m["step"], m.get("val_t16_ema", m.get("val_t16")) or 0.0) for m in mon]
    for i, (s, x) in enumerate(v):
        mx = max(y for _, y in v[: i + 1])
        if mx - x >= R["onset_pp"] and all(mx - y >= R["onset_pp"] for _, y in v[i:]): return s
    return "NOT-BY-END"


def argv_missing(root: PF.Root, arm: str):
    """the registered levers absent from the arm's recorded argv.
    READER-FORMAT ADDENDUM 2026-09-17 (after the frozen run printed a false FAIL): the trainer writes config.json["argv"] as vars(args), a DICT, where the
    frozen reader expected the token list. The dict form is read pairwise (flag -> its value), which is stricter than the registered token membership;
    the list form is read exactly as registered. The registry, every rule and every threshold are unchanged."""
    av = (PF.jload(root.pdir(arm) / "config.json") or {}).get("argv")
    if not isinstance(av, dict): return [f for f in R["flags"] if f not in (av if isinstance(av, list) else [])]
    f, miss, i = R["flags"], [], 0
    while i < len(f):
        key = f[i][2:].replace("-", "_")
        if i + 1 < len(f) and not f[i + 1].startswith("--"):           # a valued lever: the recorded value equals the registered one
            got, want = av.get(key), f[i + 1]
            try: same = got is not None and not isinstance(got, bool) and (str(got) == want or float(got) == float(want))
            except (TypeError, ValueError): same = False
            if not same: miss += [f[i], want]
            i += 2
        else:                                                           # a switch: recorded True
            if av.get(key) is not True: miss.append(f[i])
            i += 1
    return miss


def letters(root: PF.Root, arm: str):
    L, probs = {}, []
    a16 = root.acc(root.chain(arm, "full_vsel_t16")); a64 = root.acc(root.chain(arm, "full_vsel_t64"))
    s16 = root.summ(root.chain(arm, "full_vsel_t16")) or {}; s64 = root.summ(root.chain(arm, "full_vsel_t64")) or {}
    if s16.get("n") != 422786: probs.append(f"{arm} D16 n {s16.get('n')} != 422786")
    if s64.get("n") != 100000: probs.append(f"{arm} D64 n {s64.get('n')} != 100000")
    cks = {s.get("ckpt") for s in (s16, s64, root.summ(root.scan32(arm)) or {}) if s.get("ckpt")}
    if len(cks) > 1: probs.append(f"{arm} {len(cks)} checkpoints across its rows")
    miss = argv_missing(root, arm)
    if miss: probs.append(f"{arm} argv lacks {miss}")
    e = root.recs(root.eqr128())
    if e is None: probs.append("EqR k128 row absent")
    for a in TRIPLE + (arm,):
        z = root.recs(root.scan128(a)); s = root.summ(root.scan128(a)) or {}
        if z is None:
            if a != arm: probs.append(f"{a} k128 row absent")
            continue
        if s.get("n") != 5000 or s.get("k_init") != 128 or s.get("t_total") != 64: probs.append(f"{a} k128 row n/k/t {s.get('n')}/{s.get('k_init')}/{s.get('t_total')}")
        if e is not None and set(np.asarray(z["idx"]).tolist()) != set(np.asarray(e["idx"]).tolist()): probs.append(f"{a} k128 idx set != EqR's")
    L["INTEGRITY"] = "PASS" if not probs else "FAIL: " + "; ".join(probs)
    # R-SP-1 PARAMS
    n, src = n_params_of(root, arm)
    L["R-SP-1 PARAMS"] = "NO-DATA" if n is None else (("MATCHED" if abs(n / R["params_dec"] - 1) <= R["params_tol"] else "UNMATCHED") + f" ({n:,} vs {R['params_dec']:,}, x{n / R['params_dec']:.3f}; {src})")
    # R-SP-2 PARITY at 16 and 64 against the triple's mean, the floor = the triple's half-spread (read; the registry's fallback)
    for d, key, fl in (("16", "full_vsel_t16", "floor16"), ("64", "full_vsel_t64", "floor64")):
        tri = [root.acc(root.chain(t, key)) for t in TRIPLE]
        x = a16 if d == "16" else a64
        if x is None or any(v is None for v in tri):
            L[f"R-SP-2 PARITY-{d}"] = "NO-DATA" if x is None else f"NO-TRIPLE (X {100*x:.2f})"; continue
        m = float(np.mean(tri)); floor = (max(tri) - min(tri)) / 2; src = "read"
        if floor == 0: floor, src = R[fl], "registry"
        dd = x - m
        L[f"R-SP-2 PARITY-{d}"] = ("BELOW" if dd < -2 * floor else "ABOVE" if dd > 2 * floor else "PARITY") + f" (X {100*x:.2f} vs the triple's mean {100*m:.2f}, d {100*dd:+.2f} pp, floor {100*floor:.2f} {src})"
    # R-SP-3 SELECTOR (k32, and k128 when present)
    z32 = root.recs(root.scan32(arm)); sp = PF.spurious(z32) if z32 is not None else None
    z128 = root.recs(root.scan128(arm)); sp128 = PF.spurious(z128) if z128 is not None else None
    L["R-SP-3 SELECTOR"] = "NO-DATA" if sp is None else (("CLEAN" if sp <= R["spur_clean"] else "DIRTY") + f" (spurious k32 {100*sp:.2f} %" + (f"; k128 {100*sp128:.2f} %" if sp128 is not None else "") + ")")
    # R-SP-4 the k128 TRIPLE against EqR on the identical 5k
    out, ahead, sel_v, ver_v = [], 0, [], []
    if e is not None:
        es, ev = PF.selected_exact(e); eq = dict(idx=e["idx"], sel=es)
        for a in TRIPLE:
            z = root.recs(root.scan128(a))
            if z is None: out.append(f"{a} NO-DATA"); continue
            s, v = PF.selected_exact(z); sel_v.append(float(s.mean())); ver_v.append(float(v.mean()))
            pr = PF.paired(dict(idx=z["idx"], sel=s), eq, key="sel")
            let = "PARITY" if pr is None or pr["p"] >= R["mcnemar_p"] else ("AHEAD" if pr["diff"] > 0 else "BEHIND")
            ahead += let == "AHEAD"
            out.append(f"{a} {let} (selected {100*s.mean():.2f} vs EqR {100*es.mean():.2f}, p {pr['p']:.1e}; verified {100*v.mean():.2f})" if pr else f"{a} {let}")
        if len(sel_v) == 3:
            out.append(("TRIPLE-AHEAD" if ahead == 3 else f"TRIPLE-AHEAD {ahead}/3") + f" (selected {100*min(sel_v):.2f}–{100*max(sel_v):.2f}, verified {100*min(ver_v):.2f}–{100*max(ver_v):.2f})")
    L["R-SP-4 K128-TRIPLE"] = " | ".join(out) if out else "NO-DATA"
    # R-SP-5 ONSET of the X arm
    on = onset(root, arm)
    L["R-SP-5 ONSET"] = "NO-DATA" if on is None else (f"MEMORIZES-BY {on}" if isinstance(on, int) else "NOT-BY-END")
    # descriptive
    ex = PF.excursion(root, arm)
    L["EXCURSION (descriptive)"] = "NO-DATA" if ex is None else f"{ex['letter']} {ex.get('steps') or ''}".strip()
    if z128 is not None:
        s, v = PF.selected_exact(z128); L["X k128 (descriptive)"] = f"selected {100*s.mean():.2f}, verified {100*v.mean():.2f}"
    return L


def report(root: Path, arm: str, out: Path | None):
    L = letters(PF.Root(root), arm)
    lines = [f"PENDING SUDOKU RUNS — REGISTERED VERDICT (tools/analyze_sudokupend.py; arm {arm}; the registry at the top of this file)"]
    lines += [f"  {k:24s} {v}" for k, v in L.items()]
    text = "\n".join(lines); print(text)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True); out.with_suffix(".txt").write_text(text + "\n"); json.dump(L, open(out.with_suffix(".json"), "w"), indent=1)
    return L


# ---------- selftest: hand-built records ----------
def _root(tmp: Path, x16, x64, tri16, tri64, x_sel=0.998, tri_sel=(0.998, 0.997, 0.998), eqr_sel=0.988, spur_x=0.004, n_params=795906, onset_drop=False, flags=True, argv_dict=None):
    r = tmp / "runs"; rng = np.random.default_rng(0)
    def row(d, n, acc, ck):
        d.mkdir(parents=True, exist_ok=True); (d / "summary_all.json").write_text(json.dumps({"n": n, "exact_acc": acc, "ckpt": ck, "k_init": 32, "t_total": 64}))
    def scan(d, n, k, sel_rate, ck, idx, spur=0.0):
        d.mkdir(parents=True, exist_ok=True)
        ex = np.zeros((n, k), bool); res = rng.uniform(0.5, 1.0, (n, k))
        good = rng.random(n) < sel_rate; ex[good, 0] = True; res[good, 0] = 0.01           # the best draw is exact and has the smallest residual
        wrong = ~ex; res[wrong & (rng.random((n, k)) < spur)] = 0.005                        # a fraction spur of the WRONG draws look converged (the reader's definition)
        np.savez(d / "records_all.npz", idx=idx, cold_exact=good, mi_exact_k=ex, mi_resid_k=res)
        (d / "summary_all.json").write_text(json.dumps({"n": n, "k_init": k, "t_total": 64, "ckpt": ck}))
    idx = np.arange(5000)
    for a, a16, a64, s in zip(TRIPLE, tri16, tri64, tri_sel):
        ck = f"runs/pretrainchamp_{a}/ckpt_046000.pkl"
        row(r / f"sxeval_pchamp{a}" / "full_vsel_t16", 422786, a16, ck); row(r / f"sxeval_pchamp{a}" / "full_vsel_t64", 100000, a64, ck)
        scan(r / f"filler_sxscan128_pchamp{a}", 5000, 128, s, ck, idx)
    ckx = "runs/pretrainchamp_X5/ckpt_040000.pkl"
    row(r / "sxeval_pchampX5" / "full_vsel_t16", 422786, x16, ckx); row(r / "sxeval_pchampX5" / "full_vsel_t64", 100000, x64, ckx)
    scan(r / "sxscan_pchampX5", 5000, 32, x_sel, ckx, idx, spur=spur_x); scan(r / "filler_sxscan128_pchampX5", 5000, 128, x_sel, ckx, idx, spur=spur_x)
    scan(r / "filler_sxscan128_pport_eqr", 5000, 128, eqr_sel, "runs/frontier_ckpts/eqr.pkl", idx)
    p = r / "pretrainchamp_X5"; p.mkdir(parents=True, exist_ok=True)
    av = ["--cell", "trm", "--trm-hidden", "160", "--sudoku-digit-aug", "--fpa-k", "1", "--trm-ri-sigma", "1.0"] if flags else ["--cell", "dec"]
    if argv_dict is not None: av = argv_dict                                                  # the trainer's real format: vars(args)
    (p / "config.json").write_text(json.dumps({"argv": av, "n_params_bulk": n_params, "n_params_table": 32}))
    with open(p / "metrics.jsonl", "w") as f:
        for i, st in enumerate(range(2000, 50001, 2000)):
            v = min(0.95, 0.1 + 0.1 * i); v = v - (0.06 if (onset_drop and st >= 30000) else 0.0)   # a plateau from 20k; the drop from 30k = the onset
            f.write(json.dumps({"monitor": {"step": st, "val_t16": v, "val_t16_ema": v}}) + "\n")
    return r


def selftest():
    ok = 0; bad = []
    def chk(name, cond):
        nonlocal ok
        if cond: ok += 1
        else: bad.append(name)
    with tempfile.TemporaryDirectory() as td:
        # case A: TRM's cell BELOW the triple at both depths, CLEAN, the triple AHEAD of EqR, no onset
        r = _root(Path(td) / "a", 0.930, 0.980, (0.954, 0.9595, 0.9487), (0.9905, 0.9923, 0.9887))
        L = letters(PF.Root(r), "X5")
        chk("A integrity", L["INTEGRITY"] == "PASS")
        chk("A params", L["R-SP-1 PARAMS"].startswith("MATCHED (795,938") and "bulk + the task table" in L["R-SP-1 PARAMS"])
        chk("A parity16 BELOW", L["R-SP-2 PARITY-16"].startswith("BELOW") and "floor 0.54 read" in L["R-SP-2 PARITY-16"])
        chk("A parity64 BELOW", L["R-SP-2 PARITY-64"].startswith("BELOW"))
        chk("A selector CLEAN", L["R-SP-3 SELECTOR"].startswith("CLEAN"))
        chk("A triple ahead", "TRIPLE-AHEAD (" in L["R-SP-4 K128-TRIPLE"] and L["R-SP-4 K128-TRIPLE"].count("AHEAD") >= 4)
        chk("A onset none", L["R-SP-5 ONSET"] == "NOT-BY-END")
        # case B: PARITY at 16, ABOVE at 64, DIRTY, one BEHIND, an onset at 30000, unmatched params, a missing lever
        r = _root(Path(td) / "b", 0.955, 0.9960, (0.954, 0.9595, 0.9487), (0.9905, 0.9923, 0.9887), tri_sel=(0.998, 0.980, 0.998), spur_x=0.5, n_params=1_227_586, onset_drop=True, flags=False)
        L = letters(PF.Root(r), "X5")
        chk("B integrity flags", L["INTEGRITY"].startswith("FAIL") and "argv lacks" in L["INTEGRITY"])
        chk("B params UNMATCHED", L["R-SP-1 PARAMS"].startswith("UNMATCHED"))
        chk("B parity16 PARITY", L["R-SP-2 PARITY-16"].startswith("PARITY"))
        chk("B parity64 ABOVE", L["R-SP-2 PARITY-64"].startswith("ABOVE"))
        chk("B selector DIRTY", L["R-SP-3 SELECTOR"].startswith("DIRTY"))
        chk("B one behind", "C7 BEHIND" in L["R-SP-4 K128-TRIPLE"] and "TRIPLE-AHEAD 2/3" in L["R-SP-4 K128-TRIPLE"])
        chk("B onset 30000", L["R-SP-5 ONSET"] == "MEMORIZES-BY 30000")
        # case C: the fallback floor when the triple's rows are identical (spread 0) and a missing k128 row
        r = _root(Path(td) / "c", 0.940, 0.985, (0.954, 0.954, 0.954), (0.9905, 0.9905, 0.9905))
        import shutil; shutil.rmtree(r / "filler_sxscan128_pchampC8")
        L = letters(PF.Root(r), "X5")
        chk("C fallback floor", "floor 0.54 registry" in L["R-SP-2 PARITY-16"] and L["R-SP-2 PARITY-16"].startswith("BELOW"))
        chk("C missing k128 row", "C8 k128 row absent" in L["INTEGRITY"] and "C8 NO-DATA" in L["R-SP-4 K128-TRIPLE"])
        # case D (the reader-format addendum): the trainer's dict argv — the registered levers PASS; a wrong value, a false switch and an absent key each FAIL
        good = {"cell": "trm", "trm_hidden": 160, "sudoku_digit_aug": True, "fpa_k": 1, "fpa_eps": 0.2, "trm_ri_sigma": 1.0, "seed": 1, "steps": 50000}
        tri = ((0.954, 0.9595, 0.9487), (0.9905, 0.9923, 0.9887))
        L = letters(PF.Root(_root(Path(td) / "d1", 0.930, 0.980, *tri, argv_dict=good)), "X5")
        chk("D dict argv PASS", L["INTEGRITY"] == "PASS")
        L = letters(PF.Root(_root(Path(td) / "d2", 0.930, 0.980, *tri, argv_dict={**good, "cell": "dec"})), "X5")
        chk("D dict wrong cell", L["INTEGRITY"] == "FAIL: X5 argv lacks ['--cell', 'trm']")
        L = letters(PF.Root(_root(Path(td) / "d3", 0.930, 0.980, *tri, argv_dict={**good, "sudoku_digit_aug": False, "trm_ri_sigma": 0.0})), "X5")
        chk("D dict false switch + wrong sigma", L["INTEGRITY"] == "FAIL: X5 argv lacks ['--trm-ri-sigma', '1.0', '--sudoku-digit-aug']")
        L = letters(PF.Root(_root(Path(td) / "d4", 0.930, 0.980, *tri, argv_dict={k: v for k, v in good.items() if k != "fpa_k"})), "X5")
        chk("D dict absent key", L["INTEGRITY"] == "FAIL: X5 argv lacks ['--fpa-k', '1']")
        L = letters(PF.Root(_root(Path(td) / "d5", 0.930, 0.980, *tri, argv_dict={**good, "fpa_k": True})), "X5")
        chk("D dict bool is not 1", "'--fpa-k', '1'" in L["INTEGRITY"])
    print(f"selftest {'OK' if not bad else 'FAILED'}: {ok}/{ok + len(bad)} checks" + (f"; failed: {bad}" if bad else ""))
    return 0 if not bad else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="runs"); ap.add_argument("--arm", default="X5"); ap.add_argument("--out", default=None); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest: sys.exit(selftest())
    report(Path(a.root), a.arm, Path(a.out) if a.out else None)
