# Ledger: wave-2 PHYSICS PASS (2026-08-14, PI-directed) — the same banked data
# re-read against the program's physics claims (S1/S2 throats & floors, the
# cluster-O two-fixed-point spectral collapse, anti-water-filling IR shares,
# code-geometry ladders/lifetimes, packing-plane coordinates) plus the
# curiosity sweeps (per-family structure, RI endpoint condensation, the
# trained flow-constant table, the rg-vs-width dilution curve).
# $0, disk only. Artifact: runs/analysis/p13w2_physics_20260814.txt
"""
  .venv/bin/python tools/analyze_p13w2_physics.py
"""
from __future__ import annotations
import glob
import json
import pickle
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "analysis" / "p13w2_physics_20260814.txt"

W2 = ["C53", "C80", "Dcoup", "Dfloor", "Dri", "B"]
PILOT = ["C", "A"]
FLOORS = np.array([350.0, 75.0, 50.0, 15.0, 30.0])
KNEE = np.array([.69, .14, .085, .035, .048])   # cluster-O knee profile
FREE = np.array([.76, .18, .045, .012, .003])   # cluster-O free profile

LINES: list[str] = []


def say(s=""):
    LINES.append(s)
    print(s)


def rows_of(tag, prefix="lad"):
    name = f"{prefix}_p13{tag}" if tag in W2 + PILOT else f"{prefix}_{tag}"
    p = RUNS / name / "results.jsonl"
    if not p.exists():
        return None
    return [json.loads(l) for l in p.read_text().splitlines()]


def fam(task):
    return re.sub(r"\d+$", "", task.replace("ca_", ""))


# ---------- A. spectral profiles at d64 (two-fixed-point test) ----------
def section_spectra():
    say("A. SPECTRAL PROFILES AT d64 — the two-fixed-point collapse, out-of-sample")
    say(f'   {"arm":8s} {"I_med":>6s}  {"profile I_hat(s)":34s} {"d(knee)":>8s} {"d(free)":>8s} {"IR3+4":>6s}')
    say(f'   {"knee":8s} {"":>6s}  {np.array2string(KNEE, precision=3):34s} {"-":>8s} {"-":>8s} {KNEE[3:].sum():>6.3f}')
    profs = {}
    for tag in PILOT + W2:
        rows = rows_of(tag)
        if not rows:
            continue
        spec = np.median(np.array([q["I_s"] for r in rows for q in r["queries"]]), axis=0)
        prof = spec / spec.sum()
        profs[tag] = (spec, prof)
        dk = float(np.abs(prof - KNEE).sum())
        df = float(np.abs(prof - FREE).sum())
        say(f'   {tag:8s} {spec.sum():>6.0f}  {np.array2string(prof, precision=3):34s} '
            f'{dk:>8.3f} {df:>8.3f} {prof[3:].sum():>6.3f}')
    fl = FLOORS / FLOORS.sum()
    say(f'   {"floors":8s} {FLOORS.sum():>6.0f}  {np.array2string(fl, precision=3):34s} '
        f'{np.abs(fl - KNEE).sum():>8.3f} {"-":>8s} {fl[3:].sum():>6.3f}   (the floor VECTOR, normalized)')
    if "Dfloor" in profs:
        spec = profs["Dfloor"][0]
        say(f'   Dfloor per-scale medians vs floors: '
            f'{np.array2string(spec, precision=1)} vs {np.array2string(FLOORS, precision=1)}')
    return profs


# ---------- B. mechanism flux deltas (where does each mechanism spend?) ----------
def section_deltas(profs):
    say()
    say("B. MECHANISM FLUX DELTAS vs C53 (per-scale median I_s; + = extra nats)")
    base = profs["C53"][0]
    for tag in ["C80", "Dcoup", "Dfloor", "Dri", "B"]:
        if tag not in profs:
            continue
        d = profs[tag][0] - base
        say(f'   {tag:8s} {np.array2string(d, precision=1, floatmode="fixed"):44s} total {d.sum():>+7.1f}')
    say("   (B-C53 = the NI thermal tax and its locus; Dri-C53 = the RI re-encoding cost)")


# ---------- C. absolute ladders + exact-vs-retention 2x2 + lifetime ----------
def section_geometry():
    say()
    say("C. CODE GEOMETRY, survivor-bias-free — absolute S(eps) + decode-vs-hold + lifetime")
    say(f'   {"arm":8s} {"S0":>4s} {"S.05":>5s} {"S.1":>4s} {"S.2":>4s} {"S.4":>4s} | '
        f'{"ex":>3s} {"ex&ret":>6s} {"ex&!ret":>7s} | {"life8":>6s} {"nonmono":>7s} {"rbar":>5s}')
    for tag in PILOT + W2 + ["p1248c40k"]:
        rows = rows_of(tag)
        if not rows:
            continue
        S = {e: 0 for e in ("0", "0.05", "0.1", "0.2", "0.4")}
        ex = exr = exnr = 0
        life = []
        nonmono = 0
        rbars = []
        for r in rows:
            for q in r["queries"]:
                ret = q["gt_retention"]
                S["0"] += ret
                lad = q["q_ladder"]
                for e in ("0.05", "0.1", "0.2", "0.4"):
                    S[e] += lad[e]
                ex += q["exact_T"]
                exr += q["exact_T"] and ret
                exnr += q["exact_T"] and not ret
                if ret:
                    life.append(sum(q["retained_per_step"]) / 8.0)
                    top = 0.0
                    for e in ("0.05", "0.1", "0.2", "0.4"):
                        if lad[e]:
                            top = float(e)
                    rbars.append(top)
                seq = [ret] + [lad[e] for e in ("0.05", "0.1", "0.2", "0.4")]
                if any(not a and b for a, b in zip(seq, seq[1:])):
                    nonmono += 1
        say(f'   {tag:8s} {S["0"]:>4d} {S["0.05"]:>5d} {S["0.1"]:>4d} {S["0.2"]:>4d} {S["0.4"]:>4d} | '
            f'{ex:>3d} {exr:>6d} {exnr:>7d} | {np.mean(life) if life else 0:>6.2f} {nonmono:>7d} '
            f'{np.mean(rbars) if rbars else 0:>5.2f}')
    say("   (life8 = mean fraction of 8 stab steps held, retained pairs; nonmono = pairs")
    say("    failing eps but passing a larger eps — basin-edge fuzz; rbar = mean max-survived")
    say("    eps on retained pairs, the packing-plane radius coordinate)")


# ---------- D. per-family matrix ----------
def section_families():
    say()
    say("D. PER-FAMILY RETENTION (of 9 pairs/family except where noted)")
    tags = ["p1248c40k"] + PILOT + W2
    fams = {}
    per = {}
    for tag in tags:
        rows = rows_of(tag)
        if not rows:
            continue
        by = {}
        for r in rows:
            f = fam(r["task"])
            a, b = by.get(f, (0, 0))
            by[f] = (a + sum(q["gt_retention"] for q in r["queries"]),
                     b + len(r["queries"]))
        per[tag] = by
        for f, (_, n) in by.items():
            fams[f] = n
    hdr = "   " + f'{"family":22s}' + "".join(f"{t:>7s}" for t in per)
    say(hdr)
    order = sorted(fams, key=lambda f: -sum(per[t].get(f, (0, 0))[0] for t in per))
    for f in order:
        row = f'   {f:22s}'
        for t in per:
            g, n = per[t].get(f, (0, 0))
            row += f"{g:>7d}"
        say(row + f"   /{fams[f]}")
    # Dcoup surviving core vs capture-class
    if "Dcoup" in per and "C53" in per:
        core = {f for f, (g, n) in per["Dcoup"].items() if g > 0}
        say(f'   Dcoup surviving-core families: {sorted(core)}')


# ---------- E. RI endpoint condensation (mode-collapse check) ----------
def section_ri():
    say()
    say("E. RI ENDPOINT CONDENSATION — invariance or collapse?")
    for tag in ("Dri", "C53"):
        p = RUNS / f"samp_p13{tag}_mi" / "results.jsonl"
        nd1_d, ndm_d, nd = [], [], []
        cov_detfail = [0, 0]
        for line in p.read_text().splitlines():
            r = json.loads(line)
            for q in r["queries"]:
                s = q["sigmas"]["0.0"]
                nd.append(s["n_distinct"])
                (nd1_d if s["n_distinct"] == 1 else ndm_d).append(s["best_dist"])
                if q["det_dist"] > 0:          # det point missed truth
                    cov_detfail[1] += 1
                    cov_detfail[0] += s["within_radius"] and s["best_dist"] < q["det_dist"]
        ks, vs = np.unique(nd, return_counts=True)
        say(f'   {tag:5s} nd histogram: ' + " ".join(f"{k}:{v}" for k, v in zip(ks, vs)))
        say(f'         nd=1 pairs: n={len(nd1_d)}, best_dist med {np.median(nd1_d) if nd1_d else 0:.3f} | '
            f'nd>1: n={len(ndm_d)}, med {np.median(ndm_d) if ndm_d else 0:.3f} | '
            f'det-failed pairs where an alt init IMPROVES into radius: {cov_detfail[0]}/{cov_detfail[1]}')


# ---------- F. flow-constant table ----------
def section_eta():
    say()
    say("F. THE TRAINED FLOW CONSTANT eta ACROSS EVERY LOCAL CKPT (sigmoid(eq.eta))")
    sig = lambda x: 1 / (1 + np.exp(-x))
    out = []
    for p in sorted(glob.glob(str(RUNS / "pretrain*" / "ckpt_latest.pkl"))):
        try:
            ck = pickle.loads(Path(p).read_bytes())
            eq = ck["state"]["model"].get("eq")
            if eq is None:
                continue
            c = ck["config"]
            e = float(sig(float(np.asarray(eq["eta"]))))
            extra = ""
            if "alpha1" in eq:
                a1 = float(sig(float(np.asarray(eq["alpha1"]))))
                a2 = float(sig(float(np.asarray(eq["alpha2"]))))
                extra = f" a1+a2={a1 + a2:.3f} (a2={a2:.3f})"
            out.append((c.get("d"), c.get("T"), int(ck["step"]),
                        Path(p).parent.name, e, extra))
        except Exception:
            continue
    for d, T, step, name, e, extra in sorted(out):
        say(f'   d{d:<3} T{T} steps {step:>6d}  {name:22s} eta={e:.3f}{extra}')


# ---------- G. rg-vs-width dilution curve ----------
def section_rg_width():
    say()
    say("G. UNSEEN-FAMILY (rg) RETENTION vs WIDTH — priced arms, the dilution curve")
    cells = [
        ("d16", ["p10c"]), ("d24", ["p1124c", "p1124cs1", "p1124cs2"]),
        ("d32", ["p1132c", "p1132cs1", "p1132cs2"]), ("d32T6", ["p1132cT6", "p1132cT6s1"]),
        ("d48@20k", ["p1148c", "p1148cs1"]), ("d48@40k", ["p1248c40k"]),
        ("d64@53k", ["p13C", "p13C53"]), ("d64@80k", ["p13C80"]),
    ]
    for label, tags in cells:
        rets, s2s = [], []
        for t in tags:
            name = f"ladrg_{t}"
            p = RUNS / name / "results.jsonl"
            if not p.exists():
                continue
            ret = s2 = 0
            for line in p.read_text().splitlines():
                r = json.loads(line)
                for q in r["queries"]:
                    ret += q["gt_retention"]; s2 += q["q_ladder"]["0.2"]
            rets.append(ret); s2s.append(s2)
        if rets:
            say(f'   {label:8s} rg_ret {np.mean(rets):>5.1f} (n={len(rets)}: {rets})  S(.2) {np.mean(s2s):.1f}')


def main():
    say("=" * 96)
    say("P13 WAVE-2 — PHYSICS PASS (RG / holography / information / code geometry) 2026-08-14")
    say("=" * 96)
    profs = section_spectra()
    section_deltas(profs)
    section_geometry()
    section_families()
    section_ri()
    section_eta()
    section_rg_width()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(LINES) + "\n")
    say()
    say(f"artifact -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
