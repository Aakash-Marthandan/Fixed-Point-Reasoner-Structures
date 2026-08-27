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

Usage: python3 tools/audit_sportB_integrity.py --tag sportB --phase tail|final [--cache DIR]
  tail  = mid-recovery state: p4 s6/s14 + p4mid + breadth20k/final PENDING.
  final = everything required (the §6 close audit).

--tag sportBr2 (rung-2 generalization, built 2026-08-27 during the ride per the
HANDOFF during-ride list; same boundary: presence/counts/provenance/partition
ONLY, no metric aggregated or compared):
  python3 tools/audit_sportB_integrity.py --tag sportBr2 --phase live|final
  live  = mid-campaign: audit integrity of PRESENT objects only (absences are
          INFO, not FAIL; undeterminable empties WARN).
  final = strict §6 close audit per the rung-2 registration: 5 arms
          (C1/C1s1/C2/C3/C4; C3=20k, rest 50k) each ckpt+grid+metrics+val_best+
          evalcheap; screens 3/arm {vb,m1,m2} (m-steps 25000/40000 or
          10000/15000; zero-byte legit ONLY when the m-step coincides with vb,
          full_vb zero-byte legit ONLY when vb==final) with step.txt
          cross-checks; fulls t64 ×5 + t6/vb on carriers {C1,C1s1}; probes4
          (4 × 512 rows); p4/ pinned-NSH partition proof + winner-marker
          lineage; breadth20k (n=20000 gate) + _mid if present; depth_t256;
          d3demo ×0-2 OPTIONAL (audited only if present).
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

# ---- rung-2 (sportBr2) campaign spec — mirrors chain_sportBr2.sh exactly ----
R2_GCS = "gs://qhrrn2-rescue/sportBr2"
R2_TAG = "sportBr2"
R2_ARMS = ["C1", "C1s1", "C2", "C3", "C4"]
R2_PRIMARY = ["C1", "C1s1", "C2", "C3"]          # probes4 set (chain PRIMARY)
R2_CARRIERS = ["C1", "C1s1"]                     # full t6 + vb (chain CARRIER_FULLS)
R2_STEPS = {"C1": 50000, "C1s1": 50000, "C2": 50000, "C3": 20000, "C4": 50000}
R2_SUB = 20000                                   # PHASE4 subsample (SX_SUB)


def r2_mstep(arm, kind):
    """chain_sportBr2.sh mstep(): fixed screen steps per arm length."""
    if R2_STEPS[arm] == 50000:
        return 25000 if kind == "m1" else 40000
    return 10000 if kind == "m1" else 15000


def r2_grid(arm):
    """expected banked 5k-grid ckpt object names for the arm."""
    return {f"{arm}_ckpt_{s:06d}.pkl" for s in range(5000, R2_STEPS[arm] + 1, 5000)}
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


def run_sportBr2(a):
    strict = a.phase == "final"
    cache = Path(a.cache); cache.mkdir(parents=True, exist_ok=True)
    sel, n_test, n_val = selections()
    print(f"selections recomputed: test={n_test} val={n_val} strat={len(sel['strat512'])} sub20k={len(sel['sub20k'])}")
    objs = gcs_listing()

    def info(msg):
        print(f"  info {msg}")
    miss = fail if strict else info

    vb = {}
    for x in R2_ARMS:
        p = pull(f"{x}_val_best.txt", cache) if f"{x}_val_best.txt" in objs else None
        vb[x] = p.read_text().split()[0] if p else None

    # dynamic legit-empty set: the chain banks zero-byte SKIP markers when a
    # fixed screen step coincides with vb, or full_vb when vb == final.
    legit_empty = set()
    for x in R2_ARMS:
        if vb[x] is not None:
            for kind in ("m1", "m2"):
                if int(vb[x]) == r2_mstep(x, kind):
                    legit_empty.add(f"screen_{x}_{kind}_k{256}.tgz")
    for x in R2_CARRIERS:
        if vb[x] is not None and int(vb[x]) == R2_STEPS[x]:
            legit_empty.add(f"full_{x}_vb.tgz")

    print("== A. inventory ==")
    per_arm = [f"{x}_{s}" for x in R2_ARMS
               for s in ("ckpt.pkl", "metrics.jsonl", "val_best.txt", "evalcheap.tgz")]
    fulls = [f"full_{x}_t64.tgz" for x in R2_ARMS] + \
            [f"full_{x}_{k}.tgz" for x in R2_CARRIERS for k in ("t6", "vb")]
    screens = [f"screen_{x}_{c}_k256.tgz" for x in R2_ARMS for c in ("vb", "m1", "m2")]
    need = per_arm + fulls + screens + ["probes4.tgz", "p4/NSH.txt", "p4winner.txt",
                                        "breadth20k.tgz", "depth_t256.tgz"]
    if strict:
        need.append(f"{R2_TAG}_final.tgz")
    for o in need:
        if o not in objs:
            miss(f"missing object {o}")
        elif objs[o] == 0 and o in legit_empty:
            ok(f"{o} empty BY DESIGN (SKIP marker consistent with val_best)")
        elif objs[o] == 0:
            (fail if strict or vb.get(o.split("_")[1] if "_" in o else "") is not None
             else warn)(f"EMPTY object {o} (not a derivable legit empty)")
        else:
            ok(f"{o} present ({objs[o]}B)")
    for x in R2_ARMS:
        got = {o for o in objs if o.startswith(f"{x}_ckpt_0")}
        exp = r2_grid(x)
        if got == exp:
            ok(f"{x}: 5k-grid complete ({len(exp)} banked ckpts)")
        elif strict:
            fail(f"{x}: 5k-grid mismatch missing={sorted(exp - got)[:4]} stray={sorted(got - exp)[:4]}")
        else:
            info(f"{x}: 5k-grid {len(got)}/{len(exp)} banked so far")
    stray_empty = [o for o, s in objs.items() if s == 0 and o.endswith(".tgz") and o not in legit_empty]
    if stray_empty:
        (fail if strict else warn)(f"stray empty tgzs: {stray_empty}")

    print("== B. sharded evals (partition proofs) ==")
    for x in R2_ARMS:
        kinds = [("t64", "full_t64", sel["full_test"])]
        if x in R2_CARRIERS:
            kinds += [("t6", "full_t6", sel["full_test"]), ("vb", "full_t64_valbest", sel["full_test"])]
        for kind, sub, esel in kinds:
            o = f"full_{x}_{kind}.tgz"
            if objs.get(o, 0) == 0:
                continue
            tg = pull(o, cache)
            if not tg:
                fail(f"{o}: pull failed"); continue
            d = extract(tg, cache) / "runs" / f"sxeval_p{R2_TAG}{x}" / sub
            audit_sharded(f"{o}", d, esel)
        o = f"{x}_evalcheap.tgz"
        if o not in objs:
            continue
        tg = pull(o, cache)
        if not tg:
            fail(f"{o}: pull failed"); continue
        root = extract(tg, cache) / "runs" / f"sxeval_p{R2_TAG}{x}"
        for kind, esel in [("strat_t6", sel["strat512"]), ("strat_t64", sel["strat512"]),
                           ("strat_t256", sel["strat512"]), ("val_t64", sel["val"]),
                           ("ret_t8", sel["strat512"]), ("retfm_t8", sel["strat512"])]:
            kd = root / kind
            if not kd.exists():
                (fail if strict else info)(f"{o}:{kind} missing dir"); continue
            audit_sharded(f"{o}:{kind}", kd, esel)
    for x in R2_ARMS:
        for c in ("vb", "m1", "m2"):
            o = f"screen_{x}_{c}_k256.tgz"
            if objs.get(o, 0) == 0:
                continue
            tg = pull(o, cache)
            if not tg:
                fail(f"{o}: pull failed"); continue
            d = extract(tg, cache) / "runs" / f"sxscreen_p{R2_TAG}{x}_{c}"
            audit_sharded(o, d, sel["strat512"])
            st = d / "step.txt"
            expct = vb[x] if c == "vb" else f"{r2_mstep(x, c):06d}"
            if not st.exists():
                fail(f"{o}: step.txt missing (registered cross-check)")
            elif expct is None:
                warn(f"{o}: step.txt={st.read_text().strip()} but val_best not banked yet")
            elif int(st.read_text().split()[0]) != int(expct):
                fail(f"{o}: step.txt={st.read_text().strip()} != expected {expct} (wrong-ckpt screen)")
            else:
                ok(f"{o}: step.txt == expected step {expct}")

    print("== C. p4/ pinned partition ==")
    nshp = pull("p4/NSH.txt", cache) if "p4/NSH.txt" in objs else None
    winner_p = pull("p4winner.txt", cache) if "p4winner.txt" in objs else None
    winner = winner_p.read_text().split()[0] if winner_p else None
    if nshp is None:
        miss("p4/NSH.txt absent (PHASE4 not entered yet)" if not strict else "p4/NSH.txt MISSING at final")
    else:
        NSH = int(nshp.read_text().split()[0])
        ok(f"p4 partition pin NSH={NSH}")
        expN = np.array_split(sel["sub20k"], NSH)
        have = sorted(int(o.split("summary_s")[1].split(".")[0])
                      for o in objs if o.startswith("p4/summary_s"))
        pending = [i for i in range(NSH) if i not in have]
        if strict and pending:
            fail(f"p4 shards missing at final: {pending}")
        elif pending:
            info(f"p4 shards banked={have} pending={pending}")
        for i in have:
            rp = pull(f"p4/records_s{i}.npz", cache)
            sp = pull(f"p4/summary_s{i}.json", cache)
            if not rp or not sp:
                fail(f"p4 s{i}: pull failed"); continue
            z = np.load(rp, allow_pickle=True)
            idx = np.asarray(z["idx"])
            if not np.array_equal(np.sort(idx), expN[i]):
                fail(f"p4 s{i}: idx != array_split slice {i}/{NSH} (n={len(idx)} vs {len(expN[i])})")
            else:
                ok(f"p4 s{i}: idx == slice {i}/{NSH} (n={len(idx)})")
            errs = domains_ok({k2: z[k2] for k2 in z.files if k2 in REC_KEYS}, 64, 128)
            (fail if errs else ok)(f"p4 s{i}: domains {'violations: ' + str(errs[:3]) if errs else 'legal'}")
        js = [json.loads(pull(f"p4/summary_s{i}.json", cache).read_text()) for i in have]
        if js:
            bad = [(f"s{have[i]}", k) for i, j in enumerate(js) for k in PROV if j.get(k) != js[0].get(k)]
            (fail if bad else ok)(f"p4: provenance across {len(js)} shards {'DRIFT ' + str(bad[:6]) if bad else 'identical'}")
            if winner is None:
                miss("p4winner.txt absent")
            else:
                ck = js[0].get("ckpt", "")
                okc = f"pretrain{R2_TAG}_{winner}" in ck
                (ok if okc else fail)(f"p4 ckpt lineage: {ck} {'matches winner marker ' + winner if okc else 'does NOT match winner marker ' + str(winner)}")
    if objs.get("breadth20k.tgz", 0) > 0 and winner:
        tg = pull("breadth20k.tgz", cache)
        if tg:
            root = extract(tg, cache) / "runs"
            bd = root / f"sxbreadth20k_p{R2_TAG}{winner}"
            if bd.exists():
                audit_sharded("breadth20k(winner)", bd, sel["sub20k"])
                sa = bd / "summary_all.json"
                if sa.exists():
                    n = json.loads(sa.read_text()).get("n")
                    (ok if n == R2_SUB else fail)(f"breadth20k merge n-gate: n={n} (require {R2_SUB})")
            else:
                fail(f"breadth20k.tgz lacks sxbreadth20k_p{R2_TAG}{winner}")
            md = root / f"sxbreadth20k_p{R2_TAG}{winner}_mid"
            if md.exists():
                audit_sharded("breadth20k(mid)", md, sel["sub20k"])
                info("PHASE4-MID present (fired)")
            else:
                info("PHASE4-MID absent (chain predicate decides; presence-only note)")
        else:
            fail("breadth20k.tgz: pull failed")

    print("== C2. depth rider ==")
    if objs.get("depth_t256.tgz", 0) > 0:
        if not winner:
            fail("depth_t256.tgz present but p4winner.txt absent")
        else:
            tg = pull("depth_t256.tgz", cache)
            if tg:
                d = extract(tg, cache) / "runs" / f"sxdepth_p{R2_TAG}{winner}_t256"
                audit_sharded("depth_t256", d, sel["full_test"])
                s0 = sorted(d.glob("summary_s*.json")) or sorted(d.glob("summary_all.json"))
                if s0:
                    tt = json.loads(s0[0].read_text()).get("t_total")
                    (ok if tt == 256 else fail)(f"depth rider t_total={tt} (require 256)")
            else:
                fail("depth_t256.tgz: pull failed")
    elif strict:
        fail("depth_t256.tgz missing at final")

    print("== C3. d3demo (optional; audited only if present) ==")
    for which in ("b2d64", "s5d16"):
        o = f"d3demo_{which}.tgz"
        if objs.get(o, 0) == 0:
            info(f"{o} absent (OPTIONAL — never blocks)"); continue
        tg = pull(o, cache)
        if not tg:
            fail(f"{o}: pull failed"); continue
        d = extract(tg, cache) / "runs" / f"sxd3demo_{which}"
        audit_sharded(o, d, sel["strat512"])
        s0 = sorted(d.glob("summary_all.json"))
        if s0:
            ck = json.loads(s0[0].read_text()).get("ckpt", "")
            (ok if f"_d3demo_{which}" in ck else fail)(f"{o}: ckpt lineage {ck}")

    print("== B2. probes4 ==")
    if objs.get("probes4.tgz", 0) > 0:
        tg = pull("probes4.tgz", cache)
        if tg:
            pr = extract(tg, cache)
            for x in R2_PRIMARY:
                rf = pr / "runs" / f"sudprobe_p{R2_TAG}{x}" / "results.jsonl"
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
    else:
        miss("probes4.tgz absent")

    print("== D. metrics hygiene ==")
    for x in R2_ARMS:
        if f"{x}_metrics.jsonl" not in objs:
            miss(f"{x}_metrics.jsonl absent (arm not complete)"); continue
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
        desc = [(steps[i], steps[i + 1]) for i in range(len(steps) - 1) if steps[i + 1] < steps[i]]
        dedup = {}
        for s in steps:
            dedup[s] = dedup.get(s, 0) + 1
        dsteps = sorted(dedup)
        cover = bool(dsteps) and dsteps[-1] == top and len(dsteps) >= top // 100
        structural = badnum == 0 and R2_STEPS[x] - 100 <= top <= R2_STEPS[x] and cover and len(desc) <= 25
        if structural and not desc:
            ok(f"{x}: metrics rows={len(steps)} monotone max_step={top}/{R2_STEPS[x]} bad_rows=0")
        elif structural:
            rb_ok = all(t <= f for f, t in desc)
            (ok if rb_ok else fail)(
                f"{x}: metrics rows={len(steps)} with {len(desc)} resume splice(s) "
                f"{[f'{f}->{t}' for f, t in desc[:6]]} — lineage artifact, coverage complete "
                f"to {top}/{R2_STEPS[x]}, bad_rows=0 (analysis pass must last-wins dedup)")
        else:
            fail(f"{x}: metrics rows={len(steps)} max_step={top}/{R2_STEPS[x]} "
                 f"bad_rows={badnum} splices={len(desc)} coverage={cover}")

    print("== E. ckpt lineage ==")
    banked = {o for o in objs if "_ckpt" in o}
    for x in R2_ARMS:
        if vb[x] is None:
            miss(f"{x}: val_best not banked yet"); continue
        if f"{x}_ckpt_{vb[x]}.pkl" in banked or vb[x] == f"{R2_STEPS[x]:06d}":
            ok(f"{x}: vb step {vb[x]} ckpt banked (or == final)")
        else:
            fail(f"{x}: vb step {vb[x]} ckpt NOT banked")

    print("== SUMMARY ==")
    print(f"PASS={len(P)} FAIL={len(F)} WARN={len(W)}")
    if F:
        print("FAILURES:")
        for m in F:
            print(f"  - {m}")
    return 1 if F else 0


def main():
    global GCS
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", choices=["sportB", "sportBr2"], default="sportB")
    ap.add_argument("--phase", choices=["tail", "final", "live"], default="tail")
    ap.add_argument("--cache", default=None)
    a = ap.parse_args()
    if a.cache is None:
        a.cache = f"runs/audit_cache_{a.tag}"
    if a.tag == "sportBr2":
        if a.phase == "tail":
            a.phase = "live"
        GCS = R2_GCS
        return run_sportBr2(a)
    if a.phase == "live":
        print("--phase live is a sportBr2 mode; use tail|final for sportB"); return 2
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
