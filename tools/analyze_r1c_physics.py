# Ledger: RUNG-1c PHYSICS PASS (2026-08-21) — supplementary to the registered
# analyze_r1c verdicts (which come ONLY from tools/analyze_r1c.py, run untouched).
# The 2x2 decomposition with per-seed detail + McNemar, the FLOORS x CELL
# interaction surface, the A5-class budget axis, the beta-transfer frontier at
# d>=48, spectra/profiles/IR-conservation, the eta ladder, e1e3 landscape detail,
# the OOD gradient (rt/vh/rg) along the BUDGET axis, per-family slices (vacancy
# floor, W-alpha Copy watch, Center), and the packing plane. Disk-only, $0.
# Artifact: runs/analysis/r1c_physics_20260821.txt.
import json, math, pickle
from pathlib import Path
import numpy as np

R = Path("runs")
OUT = R / "analysis" / "r1c_physics_20260821.txt"
KNEE = np.array([.69, .14, .085, .035, .048]); FREE = np.array([.76, .18, .045, .012, .003])
FLOORS = np.array([350., 75., 50., 15., 30.])
EPS = ["0.05", "0.1", "0.2", "0.4"]
LINES = []
def say(s=""): LINES.append(str(s)); print(s)

def qs(prefix, tag):
    p = R / f"{prefix}_{tag}" / "results.jsonl"
    if not p.exists(): return None
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return [(r["task"], qi, q) for r in rows for qi, q in enumerate(r["queries"])]

def eta(pdir):
    ck = pickle.loads((R / pdir / "ckpt_latest.pkl").read_bytes())
    return float(1 / (1 + np.exp(-float(np.asarray(ck["state"]["model"]["eq"]["eta"])))))

def rgpairs(tag):
    out = {}
    for pref in ("ladrg", "ladrgb"):
        for t, qi, q in qs(pref, tag) or []:
            out[(pref, t, qi)] = bool(q["gt_retention"])
    return out

def vhpairs(tag):
    return {(t, qi): bool(q["gt_retention"]) for t, qi, q in qs("lad", tag)}

def mcn(pa, pb):
    k = set(pa) & set(pb); b = sum(pa[x] and not pb[x] for x in k); c = sum(pb[x] and not pa[x] for x in k)
    n = b + c; p = 1.0 if n == 0 else min(1.0, sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n * 2)
    return b, c, p

def cellrow(tag, rt_tag=None):
    """All instruments for one battery cell."""
    Q = qs("lad", tag)
    N = sum(q["gt_retention"] for _, _, q in Q)
    S2 = sum(1 for _, _, q in Q if q["gt_retention"] and q["q_ladder"]["0.2"])
    S4 = sum(1 for _, _, q in Q if q["gt_retention"] and q["q_ladder"]["0.4"])
    EX = sum(q["exact_T"] for _, _, q in Q)
    I = np.array([q["I_s"] for _, _, q in Q]); Imed = float(np.median(I.sum(axis=1)))
    Smed = np.median(I, axis=0); P = Smed / Smed.sum()
    radii = [max([float(e) for e in EPS if q["q_ladder"][e]] or [0.0]) for _, _, q in Q if q["gt_retention"]]
    rbar = float(np.mean(radii)) if radii else 0.0
    fuzz = sum(1 for _, _, q in Q if q["gt_retention"] and any(
        (not q["q_ladder"][EPS[i]]) and q["q_ladder"][EPS[i+1]] for i in range(3)))
    g48 = sum(q["gt_retention"] for _, _, q in (qs("ladrg", tag) or []))
    b48 = sum(q["gt_retention"] for _, _, q in (qs("ladrgb", tag) or []))
    rt = sum(q["gt_retention"] for _, _, q in (qs("ladrt", rt_tag or tag) or [])) if qs("ladrt", rt_tag or tag) else None
    return dict(N=N, S2=S2, S4=S4, EX=EX, I=Imed, spec=Smed, prof=P, rbar=rbar, fuzz=fuzz,
                rg=g48 + b48, g48=g48, b48=b48, rt=rt)

def fam(tag, pref="lad"):
    out = {}
    for t, qi, q in qs(pref, tag) or []:
        f = t.split("_")[1] if "_" in t else t
        f = "".join(ch for ch in f if not ch.isdigit())
        out.setdefault(f, [0, 0]); out[f][0] += q["gt_retention"]; out[f][1] += 1
    return out

def e13(tag):
    p = R / f"e1e3_{tag}" / "results.jsonl"
    if not p.exists(): return None
    ws = conv = tot = nd1 = lex = 0
    for l in p.read_text().splitlines():
        r = json.loads(l)
        for e3 in r["e3"]:
            tot += 1
            if e3["converged_at"] is not None:
                conv += 1; ws += (not e3["limit_exact"]); lex += bool(e3["limit_exact"])
            nd1 += (e3["n_distinct"] == 1)
    return dict(ws=ws, conv=conv, tot=tot, nd1=nd1, lex=lex)

# battery tag, pretrain dir, rt battery tag (aliases for legacy seeds)
GROUPS = {
    "a d48@40k A4":   [("p13fA4s0", "pretrain13f_A4s0", "p13fA4s0"), ("p13fA4s1", "pretrain13f_A4s1", "pr1bA9s1"), ("p13fA4s2", "pretrain13f_A4s2", "pr1bA9s2")],
    "b d48@53k A8":   [("pr1bA8s0", "pretrainr1b_A8s0", None), ("pr1cA8s1", "pretrainr1c_A8s1", None), ("pr1cA8s2", "pretrainr1c_A8s2", None)],
    "c d64@40k A10":  [("pr1cA10s0", "pretrainr1c_A10s0", None), ("pr1cA10s1", "pretrainr1c_A10s1", None)],
    "d d64@53k A4":   [("pr1A4s0", "pretrainr1_A4s0", "pr1A4s0"), ("pr1A4s1", "pretrainr1_A4s1", "pr1bA4s1"), ("pr1A4s2", "pretrainr1_A4s2", "pr1bA4s2")],
    "A11 d48@40k A5": [("pr1cA11s0", "pretrainr1c_A11s0", None), ("pr1cA11s1", "pretrainr1c_A11s1", None)],
    "A12 d64@40k A5": [("pr1cA12s0", "pretrainr1c_A12s0", None), ("pr1cA12s1", "pretrainr1c_A12s1", None)],
    "A13 d48@53k 6e-5": [("pr1cA13s0", "pretrainr1c_A13s0", None), ("pr1cA13s1", "pretrainr1c_A13s1", None)],
    "A5 d64@53k":     [("pr1A5s0", "pretrainr1_A5s0", "pr1A5s0"), ("pr1bA5s1", "pretrainr1b_A5s1", "pr1bA5s1"), ("pr1bA5s2", "pretrainr1b_A5s2", "pr1bA5s2")],
}

say("=" * 110)
say("RUNG-1c PHYSICS PASS — supplementary to runs/analysis/r1c_verdict.txt (registered verdicts live THERE)")
say("=" * 110)

# ---------- A. Full instrument table ----------
say(); say("A. FULL TABLE — every cell of the (substrate, width, budget, beta) surface")
say(f'  {"cell":18s} {"tag":10s} {"rg96":>4s} {"g/b":>7s} {"N":>3s} {"S.2":>4s} {"S.4":>4s} {"ex":>3s} {"rt":>4s} {"I_med":>6s} {"rbar":>5s} {"fuzz":>4s} {"dknee":>6s} {"UV":>5s} {"IR34":>5s} {"eta":>5s}')
C = {}
for g, cells in GROUPS.items():
    C[g] = []
    for tag, pdir, rtt in cells:
        r = cellrow(tag, rtt); r["eta"] = eta(pdir); r["tag"] = tag; C[g].append(r)
        say(f'  {g:18s} {tag:10s} {r["rg"]:>4d} {r["g48"]:>3d}/{r["b48"]:>3d} {r["N"]:>3d} {r["S2"]:>4d} {r["S4"]:>4d} {r["EX"]:>3d} {(str(r["rt"]) if r["rt"] is not None else "-"):>4s} {r["I"]:>6.0f} {r["rbar"]:>5.3f} {r["fuzz"]:>4d} {np.abs(r["prof"]-KNEE).sum():>6.3f} {r["prof"][0]:>5.2f} {r["spec"][3]+r["spec"][4]:>5.1f} {r["eta"]:>5.3f}')
def gm(g, k): v = [r[k] for r in C[g] if r[k] is not None]; return float(np.mean(v)) if v else float("nan")
say()
say(f'  {"group means":18s} {"":10s} {"rg96":>5s} {"N":>5s} {"S.4":>5s} {"ex":>5s} {"rt":>5s} {"I":>6s} {"rbar":>5s} {"eta":>5s}')
for g in GROUPS:
    say(f'  {g:18s} {"":10s} {gm(g,"rg"):>5.1f} {gm(g,"N"):>5.1f} {gm(g,"S4"):>5.1f} {gm(g,"EX"):>5.1f} {gm(g,"rt"):>5.1f} {gm(g,"I"):>6.0f} {gm(g,"rbar"):>5.3f} {gm(g,"eta"):>5.3f}')

# ---------- B. The 2x2 decomposition ----------
say(); say("B. THE A4-CLASS 2x2 — per-seed detail, both instruments, paired McNemar (identical task sets)")
a, b, c, d = "a d48@40k A4", "b d48@53k A8", "c d64@40k A10", "d d64@53k A4"
for name, g1, g2 in [("BUDGET|d48 (a vs b)", a, b), ("WIDTH|40k (a vs c)", a, c),
                     ("BUDGET|d64 (c vs d)", c, d), ("WIDTH|53k (b vs d)", b, d)]:
    r1 = [r["rg"] for r in C[g1]]; r2 = [r["rg"] for r in C[g2]]
    say(f'  {name:22s} rg96 {r1} vs {r2}: Δ {np.mean(r1)-np.mean(r2):+.1f}   N {[r["N"] for r in C[g1]]} vs {[r["N"] for r in C[g2]]}: Δ {gm(g1,"N")-gm(g2,"N"):+.1f}   seed-overlap: {"NONE" if min(r1) > max(r2) or min(r2) > max(r1) else "YES"}')
say("  paired McNemar on rg-96 (cross-run, identical 288-pair sets; per seed-pair):")
for s1 in [r["tag"] for r in C[a]]:
    for s2 in [r["tag"] for r in C[b]]:
        x, y, p = mcn(rgpairs(s1), rgpairs(s2))
        say(f'    {s1} vs {s2}: a-only {x:2d}  b-only {y:2d}  p={p:.4f}')
say("  width pairs (a vs c):")
for s1 in [r["tag"] for r in C[a]]:
    for s2 in [r["tag"] for r in C[c]]:
        x, y, p = mcn(rgpairs(s1), rgpairs(s2))
        say(f'    {s1} vs {s2}: a-only {x:2d}  c-only {y:2d}  p={p:.4f}')

# ---------- C. Floors x cell interaction ----------
say(); say("C. FLOORS x CELL INTERACTION — floors-minus-global (NI present everywhere) at each measured (d, steps)")
pairs = [("d48@40k", a, "A11 d48@40k A5"), ("d64@40k", c, "A12 d64@40k A5"), ("d64@53k", d, "A5 d64@53k")]
for cell, gf, gg in pairs:
    d1 = gm(gf, "rg") - gm(gg, "rg")
    say(f'  {cell}: floors {gm(gf,"rg"):.1f} ({[r["rg"] for r in C[gf]]}) vs global {gm(gg,"rg"):.1f} ({[r["rg"] for r in C[gg]]}) -> floors effect {d1:+.1f} rg96   [I: {gm(gf,"I"):.0f} vs {gm(gg,"I"):.0f}, floors add {gm(gf,"I")-gm(gg,"I"):+.0f} nats]')
say(f'  d48@53k: floors A8 {gm(b,"rg"):.1f} vs A13 {gm("A13 d48@53k 6e-5","rg"):.1f} — CONFOUNDED (A13 is at 2x beta); via beta-inelasticity (E) reads {gm(b,"rg")-gm("A13 d48@53k 6e-5","rg"):+.1f}')
say("  floors-binding check (spectrum vs floor vector 350/75/50/15/30; floors arms should sit AT/above, global below):")
for g in (a, b, c, d, "A11 d48@40k A5", "A12 d64@40k A5", "A5 d64@53k", "A13 d48@53k 6e-5"):
    S = np.mean([r["spec"] for r in C[g]], axis=0)
    rel = ["AT" if abs(x - f) <= 0.1 * f else ("ABOVE" if x > f else "BELOW") for x, f in zip(S, FLOORS)]
    say(f'    {g:18s} spec {np.array2string(S, precision=0, max_line_width=60):32s} vs floors: {"/".join(rel)}')

# ---------- D. The A5-class budget axis ----------
say(); say("D. THE A5-CLASS (OPERATING SUBSTRATE) BUDGET AXIS")
say(f'  d64: A12 (40k, n=2) rg96 {[r["rg"] for r in C["A12 d64@40k A5"]]}={gm("A12 d64@40k A5","rg"):.1f} vs A5 (53k, n=3) {[r["rg"] for r in C["A5 d64@53k"]]}={gm("A5 d64@53k","rg"):.1f} -> budget effect {gm("A5 d64@53k","rg")-gm("A12 d64@40k A5","rg"):+.1f} (SIGN vs A4-class: {"same" if (gm("A5 d64@53k","rg")-gm("A12 d64@40k A5","rg"))<0 else "REVERSED"})')
say(f'  d48: A11 (40k, n=2) {gm("A11 d48@40k A5","rg"):.1f} vs A13 (53k @2x beta, n=2) {gm("A13 d48@53k 6e-5","rg"):.1f} -> {gm("A13 d48@53k 6e-5","rg")-gm("A11 d48@40k A5","rg"):+.1f} (beta-bridge, see E)')

# ---------- E. The beta-transfer frontier at d>=48 ----------
say(); say("E. THE BETA AXIS — transfer vs price at fixed (substrate=A5-class+deriv, budget)")
say("  d48@53k: A8-class is floors so the clean beta pair is A13 vs A8 with the floors caveat;")
say(f'    A8 (3e-5, floors): rg96 {gm(b,"rg"):.1f}, I {gm(b,"I"):.0f} | A13 (6e-5, no floors): rg96 {gm("A13 d48@53k 6e-5","rg"):.1f}, I {gm("A13 d48@53k 6e-5","I"):.0f} -> 2x beta compresses {gm(b,"I")-gm("A13 d48@53k 6e-5","I"):.0f} nats, transfer Δ {gm("A13 d48@53k 6e-5","rg")-gm(b,"rg"):+.1f}')
say("  d64@53k (rung-1b, global+NI): 3e-6: 35.5/1604 | 1e-5: 27.0/793 | 3e-5: 32.0/455  (n=2/2/3)")
say("  -> the transfer response is FLAT across a 20x beta range (3e-6..6e-5) at both widths; throat moves ~4.7x.")
A13s = np.mean([r["spec"] for r in C["A13 d48@53k 6e-5"]], axis=0)
A8s = np.mean([r["spec"] for r in C[b]], axis=0)
say(f'  A13 spectrum {np.array2string(A13s, precision=1)} vs A8 {np.array2string(A8s, precision=1)} (2x beta cuts UV {A8s[0]-A13s[0]:+.0f}, IR34 {A8s[3]+A8s[4]-A13s[3]-A13s[4]:+.1f})')

# ---------- F. eta ladder ----------
say(); say("F. ETA LADDER — the (steps, width) progress clock, beta- and floors-independence")
for g in GROUPS:
    say(f'  {g:18s} eta {gm(g,"eta"):.3f}  {[round(r["eta"],3) for r in C[g]]}')
e40_48 = (gm(a, "eta") + gm("A11 d48@40k A5", "eta")) / 2; e53_48 = (gm(b, "eta") + gm("A13 d48@53k 6e-5", "eta")) / 2
e40_64 = (gm(c, "eta") + gm("A12 d64@40k A5", "eta")) / 2; e53_64 = (gm(d, "eta") + gm("A5 d64@53k", "eta")) / 2
say(f'  means by (steps, d): d48@40k {e40_48:.3f} | d48@53k {e53_48:.3f} | d64@40k {e40_64:.3f} | d64@53k {e53_64:.3f}')
say(f'  -> steps effect at d48 {e53_48-e40_48:+.3f}, at d64 {e53_64-e40_64:+.3f}; width effect at 40k {e40_64-e40_48:+.3f}, at 53k {e53_64-e53_48:+.3f}')

# ---------- G. e1e3 landscape detail ----------
say(); say("G. E1E3 LANDSCAPE — wrong-stable, endpoint condensation, limit-exact, quench/codebook (R1c-4 detail)")
E = {"a (d48@40k)": ["p13fA4s0", "p13fA4s1", "p13fA4s2"], "b (d48@53k)": ["pr1cA8s1", "pr1cA8s2"],
     "c (d64@40k)": ["pr1cA10s0", "pr1cA10s1"], "d (d64@53k)": ["pr1A4s0", "pr1A4s1", "pr1A4s2"]}
gmap = {"a (d48@40k)": a, "b (d48@53k)": b, "c (d64@40k)": c, "d (d64@53k)": d}
for g, tags in E.items():
    es = [e13(t) for t in tags]; es = [e for e in es if e]
    ws = np.mean([e["ws"] / max(e["conv"], 1) for e in es]); nd = np.mean([e["nd1"] for e in es])
    lex = np.mean([e["lex"] for e in es]); exN = gm(gmap[g], "EX") / gm(gmap[g], "N")
    say(f'  {g:14s} wrong-stable {ws:.3f}  nd1 {nd:.1f}/144  limit-exact {lex:.1f}  exact/N {exN:.2f}   (n={len(es)})')
say("  -> deltas: budget|d48 ws +{:.3f} nd1 {:+.1f} | width|40k ws +{:.3f} nd1 {:+.1f}".format(
    np.mean([e13(t)["ws"]/max(e13(t)["conv"],1) for t in E["b (d48@53k)"]]) - np.mean([e13(t)["ws"]/max(e13(t)["conv"],1) for t in E["a (d48@40k)"]]),
    np.mean([e13(t)["nd1"] for t in E["b (d48@53k)"]]) - np.mean([e13(t)["nd1"] for t in E["a (d48@40k)"]]),
    np.mean([e13(t)["ws"]/max(e13(t)["conv"],1) for t in E["c (d64@40k)"]]) - np.mean([e13(t)["ws"]/max(e13(t)["conv"],1) for t in E["a (d48@40k)"]]),
    np.mean([e13(t)["nd1"] for t in E["c (d64@40k)"]]) - np.mean([e13(t)["nd1"] for t in E["a (d48@40k)"]])))

# ---------- H. OOD gradient along the budget axis ----------
say(); say("H. OOD GRADIENT — rt (trained fams) / vh (held-out hard) / rg96 (never-trained), relative to anchor a")
for g in (b, c, d, "A11 d48@40k A5", "A12 d64@40k A5", "A13 d48@53k 6e-5", "A5 d64@53k"):
    rt0, n0, g0 = gm(a, "rt"), gm(a, "N"), gm(a, "rg")
    say(f'  {g:18s} rt {gm(g,"rt"):.1f} ({(gm(g,"rt")-rt0)/rt0*100:+.0f}%)   vh N {gm(g,"N"):.1f} ({(gm(g,"N")-n0)/n0*100:+.0f}%)   rg96 {gm(g,"rg"):.1f} ({(gm(g,"rg")-g0)/g0*100:+.0f}%)')

# ---------- I. Per-family slices ----------
say(); say("I. PER-FAMILY (9-pair resolution, directional only)")
say("  vacancy floor on the 7 NEW substrates (Copy / ExtractObjects / Count, vh):")
for g in (b, c, "A11 d48@40k A5", "A12 d64@40k A5", "A13 d48@53k 6e-5"):
    for r in C[g]:
        F = fam(r["tag"])
        cop = [f"{k}:{v[0]}/{v[1]}" for k, v in sorted(F.items()) if k.lower().startswith(("copy", "extract", "count"))]
        say(f'    {r["tag"]:10s} {", ".join(cop)}')
say("  Center (H-35 refutation watch) + budget-contrast families (a-mean vs b-mean, vh):")
Fa = [fam(r["tag"]) for r in C[a]]; Fb = [fam(r["tag"]) for r in C[b]]
allf = sorted(set().union(*[set(f) for f in Fa + Fb]))
for f in allf:
    ma = np.mean([F.get(f, [0, 9])[0] for F in Fa]); mb = np.mean([F.get(f, [0, 9])[0] for F in Fb])
    if abs(ma - mb) >= 1.5 or f.lower().startswith("center"):
        say(f'    {f:22s} a {ma:.1f} -> b {mb:.1f}  ({mb-ma:+.1f})')

# ---------- J. Packing plane ----------
say(); say("J. PACKING PLANE (N, rbar)")
for g in GROUPS:
    say(f'  {g:18s} ({gm(g,"N"):.1f}, {gm(g,"rbar"):.3f})')

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(LINES) + "\n")
print(f"\nartifact -> {OUT}")
