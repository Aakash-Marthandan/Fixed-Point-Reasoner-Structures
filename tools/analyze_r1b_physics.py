# Ledger: RUNG-1b PHYSICS PASS (2026-08-20) — supplementary to the registered
# analyze_r1b verdicts (which come ONLY from tools/analyze_r1b.py, run untouched).
# Beta-series spectra + the IR-conservation test (S1), McNemar supplement for the
# floors verdict, the eta ladder, the packing plane, and the width-vs-budget
# decomposition (H-40's evidence). Artifact: runs/analysis/r1b_physics_20260820.txt.
# Beta-series spectra + IR-conservation test, McNemar for the floors verdict,
# eta ladder, packing plane, budget-vs-width decomposition. Disk-only.
import json, math, pickle
from pathlib import Path
import numpy as np
R = Path("runs")
KNEE = np.array([.69, .14, .085, .035, .048]); FREE = np.array([.76, .18, .045, .012, .003])
EPS = ["0.05", "0.1", "0.2", "0.4"]

def qs(prefix, tag):
    p = R / f"{prefix}_{tag}" / "results.jsonl"
    if not p.exists(): return None
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return [(r["task"], qi, q) for r in rows for qi, q in enumerate(r["queries"])]

def spec(tag):
    Q = qs("lad", tag); I = np.array([q["I_s"] for _, _, q in Q])
    med = np.median(I, axis=0); return med, med / med.sum()

def eta(pdir):
    ck = pickle.loads((R / pdir / "ckpt_latest.pkl").read_bytes())
    return float(1 / (1 + np.exp(-float(np.asarray(ck["state"]["model"]["eq"]["eta"])))))

def rgpairs(tag, rt):
    out = {}
    for pref in ("ladrg", "ladrgb"):
        for t, qi, q in qs(pref, f"{rt}{tag}") or []:
            out[(pref, t, qi)] = bool(q["gt_retention"])
    return out

def mcn(pa, pb):
    k = set(pa) & set(pb); b = sum(pa[x] and not pb[x] for x in k); c = sum(pb[x] and not pa[x] for x in k)
    n = b + c; p = 1.0 if n == 0 else min(1.0, sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n * 2)
    return b, c, p

print("=" * 100)
print("A. BETA SERIES at d64 (global+NI): spectra, profiles, IR conservation  [A5 3e-5 n=3 | A6 1e-5 n=2 | A7 3e-6 n=2 | plain n=2]")
print(f'  {"arm":10s} {"I_med":>6s}  {"I_s means (s0..s4)":38s} {"profile":36s} {"dknee":>6s} {"dfree":>6s} {"UV":>5s} {"IR=s3+s4":>8s}')
groups = {"A5 3e-5": ["pr1A5s0", "pr1bA5s1", "pr1bA5s2"], "A6 1e-5": ["pr1bA6s0", "pr1bA6s1"], "A7 3e-6": ["pr1bA7s0", "pr1bA7s1"],
          "A4 floors": ["pr1A4s0", "pr1A4s1", "pr1A4s2"], "A8 d48@53k": ["pr1bA8s0"], "plain": ["pr1A3s0", "pr1A3s1"], "*A4 d48@40k": ["p13fA4s0", "p13fA4s1", "p13fA4s2"]}
sp = {}
for g, tags in groups.items():
    S = np.mean([spec(t)[0] for t in tags], axis=0); P = S / S.sum(); sp[g] = S
    I = float(np.median([sum(q["I_s"]) for t in tags for _, _, q in qs("lad", t)]))
    print(f'  {g:10s} {I:>6.0f}  {np.array2string(S, precision=1, max_line_width=80):38s} {np.array2string(P, precision=3):36s} {np.abs(P-KNEE).sum():>6.3f} {np.abs(P-FREE).sum():>6.3f} {P[0]:>5.2f} {S[3]+S[4]:>8.1f}')
print("  IR-CONSERVATION TEST (S1 floor candidate): s3+s4 across the beta decade:",
      " | ".join(f'{g}: {sp[g][3]+sp[g][4]:.0f}' for g in ("A5 3e-5", "A6 1e-5", "A7 3e-6")), " | plain:", f'{sp["plain"][3]+sp["plain"][4]:.0f}')
print("  UV inflation across the decade: s0:", " | ".join(f'{g}: {sp[g][0]:.0f}' for g in ("A5 3e-5", "A6 1e-5", "A7 3e-6")))

print(); print("B. FLOORS VERDICT SUPPLEMENT — paired McNemar A5 vs A4 on rg-96 (same seed where possible; all 9 pairs)")
for s5 in ("pr1A5s0", "pr1bA5s1", "pr1bA5s2"):
    for s4, tag4 in (("pr1A4s0", "A4s0"), ("pr1A4s1", "A4s1"), ("pr1A4s2", "A4s2")):
        a = rgpairs("", s5); b_ = rgpairs("", s4)
        b, c, p = mcn(a, b_)
        print(f'  {s5.replace("pr1b","").replace("pr1",""):5s} vs {tag4}: A5-only {b:2d}  A4-only {c:2d}  p={p:.3f}')

print(); print("C. ETA LADDER — the (width, budget, beta) decomposition")
rows = [("d48@40k A4 (n=3)", ["pretrain13f_A4s0", "pretrain13f_A4s1", "pretrain13f_A4s2"]),
        ("d48@53k A4 = A8 (n=1)", ["pretrainr1b_A8s0"]),
        ("d64@53k A4 (n=3)", ["pretrainr1_A4s0", "pretrainr1_A4s1", "pretrainr1_A4s2"]),
        ("d64@53k A5 3e-5 (n=3)", ["pretrainr1_A5s0", "pretrainr1b_A5s1", "pretrainr1b_A5s2"]),
        ("d64@53k A6 1e-5 (n=2)", ["pretrainr1b_A6s0", "pretrainr1b_A6s1"]),
        ("d64@53k A7 3e-6 (n=2)", ["pretrainr1b_A7s0", "pretrainr1b_A7s1"]),
        ("d64@53k plain (n=2)", ["pretrainr1_A3s0", "pretrainr1_A3s1"])]
for name, dirs in rows:
    es = [eta(d) for d in dirs]; print(f'  {name:24s} eta {np.mean(es):.3f}  {[round(e,3) for e in es]}')

print(); print("D. THE WIDTH-vs-BUDGET DECOMPOSITION of the rung-1 'width ceiling' (A8 = the deconfounder, n=1)")
def band(tags, pref="lad"):
    N = [sum(q["gt_retention"] for _, _, q in qs("lad", t)) for t in tags]
    G = [sum(rgpairs("", t).values()) for t in tags]
    RT = [sum(q["gt_retention"] for _, _, q in (qs("ladrt", t) or [])) if qs("ladrt", t) else None for t in tags]
    return N, G, RT
N48, G48, _ = band(["p13fA4s0", "p13fA4s1", "p13fA4s2"])
N8, G8, RT8 = band(["pr1bA8s0"])
N64, G64, _ = band(["pr1A4s0", "pr1A4s1", "pr1A4s2"])
print(f'  d48@40k A4 anchor (n=3):  N {N48} mean {np.mean(N48):.1f}   rg96 {G48} mean {np.mean(G48):.1f}')
print(f'  d48@53k A4 = A8   (n=1):  N {N8}          rg96 {G8}        -> BUDGET effect at fixed width: ΔN {np.mean(N8)-np.mean(N48):+.1f}, Δrg96 {np.mean(G8)-np.mean(G48):+.1f}')
print(f'  d64@53k A4        (n=3):  N {N64} mean {np.mean(N64):.1f}   rg96 {G64} mean {np.mean(G64):.1f}  -> width+budget total: ΔN {np.mean(N64)-np.mean(N48):+.1f}, Δrg96 {np.mean(G64)-np.mean(G48):+.1f}')
print(f'  -> the A8 cell reproduces {abs(np.mean(G8)-np.mean(G48))/abs(np.mean(G64)-np.mean(G48))*100:.0f}% of the rg-96 deficit and {abs(np.mean(N8)-np.mean(N48))/abs(np.mean(N64)-np.mean(N48))*100:.0f}% of the N deficit AT FIXED WIDTH (n=1, directional)')
b, c, p = mcn(rgpairs("", "pr1bA8s0"), rgpairs("", "p13fA4s0"))
print(f'  paired A8 vs *A4s0 rg-96 flips: A8-only {b}, anchor-only {c}, p={p:.3f}')

print(); print("E. PACKING PLANE (N, rbar) — beta series vs anchors")
def nr(tags):
    Ns, Rs = [], []
    for t in tags:
        Q = qs("lad", t); radii = [max([float(e) for e in EPS if q["q_ladder"][e]] or [0.0]) for _, _, q in Q if q["gt_retention"]]
        Ns.append(len(radii)); Rs.append(np.mean(radii))
    return np.mean(Ns), np.mean(Rs)
for g, tags in [("d48@40k A4", ["p13fA4s0", "p13fA4s1", "p13fA4s2"]), ("d64 A4", ["pr1A4s0", "pr1A4s1", "pr1A4s2"]),
                ("d64 A5", ["pr1A5s0", "pr1bA5s1", "pr1bA5s2"]), ("d64 A6", ["pr1bA6s0", "pr1bA6s1"]), ("d64 A7", ["pr1bA7s0", "pr1bA7s1"]), ("A8 d48@53k", ["pr1bA8s0"])]:
    n, r = nr(tags); print(f'  {g:12s} ({n:.1f}, {r:.3f})')
