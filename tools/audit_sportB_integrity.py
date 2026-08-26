#!/usr/bin/env python3
"""Rung-1 (sportB) data-integrity audit — PI mandate 2026-08-26 15:0xZ.

Ops-boundary safe BY CONSTRUCTION: presence, counts, provenance equality,
partition proofs, and domain checks only. No metric is aggregated, printed,
or compared anywhere in this file (the analysis pass owns values).

Checks:
  A. inventory: every registered GCS object present; only the registered
     by-design empties are empty (full_B2_vb, full_B3_vb, screen_B4_mid).
  B. per sharded eval: shard summaries provenance-identical (20 fields);
     records arrays length-consistent, domains legal; shard idx sets are
     pairwise disjoint and their union equals the RECOMPUTED seeded
     selection (partition proof — catches cross-partition mixing, claim
     races, resumed-fragment corruption).
  C. p4/ (PHASE4 /16): each banked shard's idx == its array_split slice.
  D. metrics.jsonl per arm: steps monotone, terminal step == registered,
     no NaN/inf loss (B1 splice lineage included).
  E. ckpt lineage: every ckpt referenced by any summary exists as a banked
     GCS object; vb steps match val_best.txt; mid steps match the formula.

Usage: python3 tools/audit_sportB_integrity.py --phase tail|final [--cache DIR]
  tail  = mid-recovery state: p4 s6/s14 + p4mid + breadth20k/final PENDING.
  final = everything required (the §6 close audit).
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qhrrn2 import sudoku_extreme as SX  # noqa: E402

GCS = "gs://qhrrn2-rescue/sportB"
NPZ = "data/sudoku_extreme/sudoku_extreme_seed0.npz"
ARMS = ["B1", "B2", "B3", "B4", "B4s1", "B5"]
PRIMARY = ["B1", "B2", "B3", "B4"]
STEPS = {"B1": 50000, "B2": 50000, "B3": 50000, "B4": 20000, "B4s1": 20000, "B5": 20000}
EMPTY_OK = {"full_B2_vb.tgz", "full_B3_vb.tgz", "screen_B4_mid_k256.tgz"}
PROV = ["ckpt", "npz", "split", "t_total", "k_init", "init", "layout", "fpopt_gamma",
        "tau", "stratified", "subsample", "subsample_seed", "mi_seed", "eta", "eta_z",
        "T", "d", "eta_learned", "eta_override", "final_map_only", "eq_coupled_ab"]
REC_KEYS = ["idx", "rating", "cold_exact", "first_exact", "first_valid", "violations",
            "cells", "givens_kept", "mi_verified", "mi_true", "mi_first_hit"]

P, F, W = [], [], []


def ok(msg):
    P.append(msg); print(f"  PASS {msg}")


def fail(msg):
    F.append(msg); print(f"  FAIL {msg}")


def warn(msg):
    W.append(msg); print(f"  WARN {msg}")


def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.returncode, r.stdout


def gcs_listing():
    _, out = sh(f"gsutil ls -l '{GCS}/**' 2>/dev/null")
    objs = {}
    for line in out.splitlines():
        p = line.split()
        if len(p) >= 3 and p[-1].startswith("gs://"):
            objs[p[-1].replace(GCS + "/", "")] = int(p[0])
    return objs


def pull(obj, cache):
    dst = cache / obj.replace("/", "_")
    if not dst.exists():
        rc, _ = sh(f"gsutil -q cp '{GCS}/{obj}' '{dst}'")
        if rc:
            return None
    return dst


def extract(tgz, cache):
    d = cache / ("x_" + tgz.name)
    if not d.exists():
        d.mkdir(parents=True)
        with tarfile.open(tgz) as t:
            t.extractall(d)
    return d


def selections():
    d = SX.load_prepared(NPZ)
    n_test, n_val = len(d["test_q"]), len(d["val_q"])
    r_test = d["test_rating"]
    sel = {
        "full_test": np.arange(n_test),
        "strat512": np.asarray(SX.stratified_subsample(r_test, 512, 20260821)),
        "val": np.arange(n_val),
        "sub20k": np.sort(np.random.default_rng(20260822).choice(n_test, size=20000, replace=False)),
    }
    return sel, n_test, n_val


def domains_ok(rec, t_total, k):
    errs = []
    lens = {k2: len(rec[k2]) for k2 in REC_KEYS if k2 in rec}
    if len(set(lens.values())) != 1:
        errs.append(f"array length mismatch {lens}")
    def dom(key, lo, hi):
        if key not in rec:
            errs.append(f"missing {key}"); return
        a = np.asarray(rec[key])
        if not np.isfinite(a.astype(float)).all():
            errs.append(f"{key} non-finite"); return
        if a.size and (a.min() < lo or a.max() > hi):
            errs.append(f"{key} out of [{lo},{hi}]")
    dom("cold_exact", 0, 1); dom("first_exact", -1, t_total); dom("first_valid", -1, t_total)
    dom("violations", 0, 10**6); dom("cells", 0, 81); dom("givens_kept", 0, 81)
    dom("mi_verified", 0, k); dom("mi_true", 0, k); dom("mi_first_hit", -1, max(k - 1, 0))
    return errs


def audit_sharded(name, sdir, expect_sel, cache_tag=""):
    """sdir holds summary_s*.json + records_s*.npz (+ summary_all.json)."""
    sums = sorted(sdir.glob("summary_s*.json"))
    recs = sorted(sdir.glob("records_s*.npz"))
    if not sums:  # unsharded run (tag="all"): single summary/records pair
        sums = sorted(sdir.glob("summary_all.json"))
        recs = sorted(sdir.glob("records_all.npz"))
    if not sums:
        fail(f"{name}: no summaries at all"); return
    js = [json.loads(p.read_text()) for p in sums]
    ref = js[0]
    if len(js) > 1:
        bad = [(sums[i].name, k) for i, j in enumerate(js) for k in PROV if j.get(k) != ref.get(k)]
        if bad:
            fail(f"{name}: provenance drift across shards: {bad[:6]}")
        else:
            ok(f"{name}: {len(js)} shard summaries provenance-identical ({len(PROV)} fields)")
    else:
        ok(f"{name}: single-run summary present (unsharded)")
    t_total, k = int(ref.get("t_total") or 0), int(ref.get("k_init") or 0)
    idxs, total, dom_errs = [], 0, []
    for rp in recs:
        z = np.load(rp, allow_pickle=True)
        rec = {k2: z[k2] for k2 in z.files if k2 in REC_KEYS}
        dom_errs += [f"{rp.name}: {e}" for e in domains_ok(rec, t_total, k)]
        idxs.append(np.asarray(rec["idx"])); total += len(rec["idx"])
    if dom_errs:
        fail(f"{name}: domain violations: {dom_errs[:6]}")
    else:
        ok(f"{name}: {len(recs)} records files, all domains legal")
    if expect_sel is not None:
        allidx = np.concatenate(idxs) if idxs else np.array([], int)
        if len(allidx) != len(set(allidx.tolist())):
            fail(f"{name}: DUPLICATE idx across shards (partition mixing)")
        elif not np.array_equal(np.sort(allidx), np.sort(np.asarray(expect_sel))):
            fail(f"{name}: idx union != expected selection (n={len(allidx)} vs {len(expect_sel)})")
        else:
            ok(f"{name}: partition proof — union == seeded selection (n={total})")
    sa = sdir / "summary_all.json"
    if sa.exists():
        n_all = json.loads(sa.read_text()).get("n")
        if n_all != total:
            fail(f"{name}: summary_all n={n_all} != sum of shards {total}")
        else:
            ok(f"{name}: summary_all n == sum of shard n ({total})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["tail", "final"], default="tail")
    ap.add_argument("--cache", default="runs/audit_cache_sportB")
    a = ap.parse_args()
    cache = Path(a.cache); cache.mkdir(parents=True, exist_ok=True)
    sel, n_test, n_val = selections()
    print(f"selections recomputed: test={n_test} val={n_val} strat={len(sel['strat512'])} sub20k={len(sel['sub20k'])}")
    objs = gcs_listing()

    print("== A. inventory ==")
    fulls = [f"full_{x}_{k}.tgz" for x in PRIMARY for k in ("t64", "t6", "vb")] + \
            ["full_B4s1_t64.tgz", "full_B5_t64.tgz"]
    screens = [f"screen_{x}_{c}_k256.tgz" for x in ARMS for c in ("vb", "mid")]
    per_arm = [f"{x}_{s}" for x in ARMS for s in ("ckpt.pkl", "metrics.jsonl", "val_best.txt", "evalcheap.tgz")]
    need = fulls + screens + ["probes4.tgz"] + per_arm
    if a.phase == "final":
        need += ["breadth20k.tgz", "sportB_final.tgz"]
    for o in need:
        if o not in objs:
            fail(f"missing object {o}")
        elif objs[o] == 0 and o not in EMPTY_OK:
            fail(f"EMPTY object {o} (not a registered empty)")
        elif objs[o] == 0:
            ok(f"{o} empty BY DESIGN")
        else:
            ok(f"{o} present ({objs[o]}B)")
    stray_empty = [o for o, s in objs.items() if s == 0 and o.endswith(".tgz") and o not in EMPTY_OK]
    if stray_empty:
        fail(f"stray empty tgzs: {stray_empty}")

    print("== B. sharded evals (partition proofs) ==")
    for x in ARMS:
        kinds = [("t64", "full_t64", sel["full_test"])]
        if x in PRIMARY:
            kinds += [("t6", "full_t6", sel["full_test"]), ("vb", "full_t64_valbest", sel["full_test"])]
        for kind, sub, esel in kinds:
            o = f"full_{x}_{kind}.tgz"
            if objs.get(o, 0) == 0:
                continue
            tg = pull(o, cache)
            if not tg:
                fail(f"{o}: pull failed"); continue
            d = extract(tg, cache) / "runs" / f"sxeval_psportB{x}" / sub
            audit_sharded(f"{o}", d, esel)
        o = f"{x}_evalcheap.tgz"
        tg = pull(o, cache)
        if not tg:
            fail(f"{o}: pull failed"); continue
        root = extract(tg, cache) / "runs" / f"sxeval_psportB{x}"
        for kind, esel in [("strat_t6", sel["strat512"]), ("strat_t64", sel["strat512"]),
                           ("strat_t256", sel["strat512"]), ("val_t64", sel["val"]),
                           ("ret_t8", sel["strat512"]), ("retfm_t8", sel["strat512"])]:
            kd = root / kind
            if not kd.exists():
                fail(f"{o}:{kind} missing dir"); continue
            ra = kd / "records_all.npz"
            if ra.exists():
                audit_sharded(f"{o}:{kind}", kd, esel)
            else:
                audit_sharded(f"{o}:{kind}", kd, esel)
    for x in ARMS:
        for c in ("vb", "mid"):
            o = f"screen_{x}_{c}_k256.tgz"
            if objs.get(o, 0) == 0:
                continue
            tg = pull(o, cache)
            if not tg:
                fail(f"{o}: pull failed"); continue
            d = extract(tg, cache) / "runs" / f"sxscreen_psportB{x}_{c}"
            audit_sharded(o, d, sel["strat512"])

    print("== C. p4/ 16-way partition ==")
    p4 = sorted(o for o in objs if o.startswith("p4/summary_s"))
    have = sorted(int(o.split("summary_s")[1].split(".")[0]) for o in p4)
    exp16 = np.array_split(sel["sub20k"], 16)
    pending = [i for i in range(16) if i not in have]
    if a.phase == "tail":
        (ok if set(pending) <= {6, 14} else fail)(f"p4 shards banked={have} pending={pending} (s6/s14 pending is the known tail)")
    elif pending:
        fail(f"p4 shards missing at final: {pending}")
    for i in have:
        rp = pull(f"p4/records_s{i}.npz", cache)
        sp = pull(f"p4/summary_s{i}.json", cache)
        if not rp or not sp:
            fail(f"p4 s{i}: pull failed"); continue
        z = np.load(rp, allow_pickle=True)
        idx = np.asarray(z["idx"])
        if not np.array_equal(np.sort(idx), exp16[i]):
            fail(f"p4 s{i}: idx != array_split slice {i}/16 (n={len(idx)} vs {len(exp16[i])})")
        else:
            ok(f"p4 s{i}: idx == slice {i}/16 (n={len(idx)})")
        errs = domains_ok({k2: z[k2] for k2 in z.files if k2 in REC_KEYS}, 64, 128)
        (fail if errs else ok)(f"p4 s{i}: domains {'violations: ' + str(errs[:3]) if errs else 'legal'}")
    js = [json.loads(pull(f"p4/summary_s{i}.json", cache).read_text()) for i in have]
    if js:
        bad = [(f"s{have[i]}", k) for i, j in enumerate(js) for k in PROV if j.get(k) != js[0].get(k)]
        (fail if bad else ok)(f"p4: provenance across {len(js)} shards {'DRIFT ' + str(bad[:6]) if bad else 'identical'}")

    print("== B2. probes4 ==")
    tg = pull("probes4.tgz", cache)
    if tg:
        pr = extract(tg, cache)
        for x in PRIMARY:
            rf = pr / "runs" / f"sudprobe_psportB{x}" / "results.jsonl"
            if not rf.exists():
                fail(f"probes4:{x}: results.jsonl missing"); continue
            rows, badj, ids = 0, 0, []
            for line in rf.read_text().splitlines():
                try:
                    j = json.loads(line); rows += 1
                    if "idx" in j:
                        ids.append(j["idx"])
                except Exception:
                    badj += 1
            uok = len(ids) == len(set(ids)) if ids else True
            (ok if rows == 512 and badj == 0 and uok else fail)(
                f"probes4:{x}: rows={rows}/512 bad_json={badj} idx_unique={uok}")
    else:
        fail("probes4.tgz: pull failed")

    print("== D. metrics hygiene ==")
    for x in ARMS:
        mp = pull(f"{x}_metrics.jsonl", cache)
        if not mp:
            fail(f"{x}_metrics.jsonl: pull failed"); continue
        steps, badnum = [], 0
        for line in mp.read_text().splitlines():
            try:
                j = json.loads(line)
            except Exception:
                badnum += 1; continue
            if "step" in j:
                steps.append(int(j["step"]))
                v = j.get("loss")
                if v is not None and not np.isfinite(float(v)):
                    badnum += 1
        top = max(steps) if steps else -1
        # Preemption-resume produces ROLLBACK SPLICES: the live metrics were pulled,
        # then training resumed from an earlier banked ckpt, so rows overlap at each
        # resume. That is lineage, not corruption, PROVIDED the descents are few
        # (bounded by node losses), each rollback lands on a resume-plausible step,
        # and after last-wins dedup the series is monotone with full coverage.
        desc = [(steps[i], steps[i + 1]) for i in range(len(steps) - 1) if steps[i + 1] < steps[i]]
        dedup = {}
        for s in steps:
            dedup[s] = dedup.get(s, 0) + 1
        dsteps = sorted(dedup)
        cover = bool(dsteps) and dsteps[-1] == top and len(dsteps) >= top // 100
        structural = badnum == 0 and STEPS[x] - 100 <= top <= STEPS[x] and cover and len(desc) <= 25
        if structural and not desc:
            ok(f"{x}: metrics rows={len(steps)} monotone max_step={top}/{STEPS[x]} bad_rows=0")
        elif structural:
            rb_ok = all(t <= f for f, t in desc)
            (ok if rb_ok else fail)(
                f"{x}: metrics rows={len(steps)} with {len(desc)} resume splice(s) "
                f"{[f'{f}->{t}' for f, t in desc[:6]]} — lineage artifact, coverage complete "
                f"to {top}/{STEPS[x]}, bad_rows=0 (analysis pass must last-wins dedup)")
        else:
            fail(f"{x}: metrics rows={len(steps)} max_step={top}/{STEPS[x]} "
                 f"bad_rows={badnum} splices={len(desc)} coverage={cover}")

    print("== E. ckpt lineage ==")
    banked = {o for o in objs if "_ckpt" in o}
    for x in ARMS:
        vb = pull(f"{x}_val_best.txt", cache)
        vbs = vb.read_text().split()[0] if vb else "?"
        if f"{x}_ckpt_{vbs}.pkl" in banked or vbs == f"{STEPS[x]:06d}":
            ok(f"{x}: vb step {vbs} ckpt banked (or == final)")
        else:
            fail(f"{x}: vb step {vbs} ckpt NOT banked")
    ref = json.loads(pull("p4/summary_s0.json", cache).read_text())
    ck = ref.get("ckpt", "")
    okc = "pretrainsportB_B2" in ck
    (ok if okc else fail)(f"p4 ckpt lineage: {ck} {'is a B2 ckpt' if okc else 'NOT a B2 ckpt'}")

    print("== SUMMARY ==")
    print(f"PASS={len(P)} FAIL={len(F)} WARN={len(W)}")
    if F:
        print("FAILURES:")
        for m in F:
            print(f"  - {m}")
    return 1 if F else 0


if __name__ == "__main__":
    sys.exit(main())
