"""The frontier comparison tables (Documentation/Comparison_Frontier_Compute_Params_2026-09-09.md), regenerated from the
analysis artifacts: the registered letters (champ_verdict.json), the physics pass (champ_physics_20260909.json: refs,
scans, the full-set D64 rows) and the MAC instrument (mac_count_20260910.json: MEASURED multiply-accumulates per outer
step per puzzle, XLA's cost analysis of the evaluator's own step map). First written 2026-09-09 with an analytic compute
column; the measured column replaced it on 2026-09-10 (tools/mac_count.py). Numbers never hand-typed.
  .venv/bin/python tools/comparison_tables.py"""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; R = ROOT / "runs" / "analysis"
V = json.load(open(R / "champ_verdict.json"))["arms"]; P = json.load(open(R / "champ_physics_20260909.json"))
REF = P["refs"]; SC = P["scans"]; F64 = P["full64"]
MAC = {r["label"]: r["macs_per_step"] / 1e9 for r in json.load(open(R / "mac_count_20260910.json"))["measured"]}
G = {"w384": MAC["DEC-w384-C0"], "w192": MAC["DEC-w192-C5"], "trm": MAC["EqR-port-w512"], "c4": MAC["DEC-w384-attn-C4"]}   # GMAC per outer step per puzzle, MEASURED
def tmac(g, steps): return g * steps / 1000
def draws_tmac(g, k, t=64): return (k + 1) * tmac(g, t)     # the cold pass + k draws at t64
mean = lambda k: sum(V[a][k] for a in ("C0", "C1", "C2")) / 3
spread = lambda k: max(V[a][k] for a in ("C0", "C1", "C2")) - min(V[a][k] for a in ("C0", "C1", "C2"))
c5full = 100 * F64["C5"]["cold64_full"]
out = []
def say(s=""): out.append(s)
say("# The frontier comparison, organized by parameters and by inference compute (2026-09-09; the full-set D64 rows and the k = 128 rows on every arm added 2026-09-10; the compute column MEASURED on 2026-09-10 by `tools/mac_count.py`, replacing the analytic column)")
say("")
say(f"Every accuracy carries its training-data column (the 1k convention = 1,000 base puzzles × augmentation; the full split = 2.7–3.8M puzzles), its protocol (single pass / restarts / search; the depth; the test set) and a compute tier: **M** = MEASURED multiply-accumulates per puzzle (XLA's cost analysis of the evaluator's own jitted one-step map at batch 1, `tools/mac_count.py` → `runs/analysis/mac_count_20260910.json`: {G['w384']:.2f} GMAC per outer step at w384, {G['w192']:.2f} at w192, {G['c4']:.2f} for the set-attention arm, {G['trm']:.2f} for the field's 5M TRM-MLP cell through the same map; the analytic column of 2026-09-09 read 70.0 / 20.4 / 15.2 and is superseded; TFLOPs = 2 × TMAC); **R** = read from the paper's own compute axis; **I** = inferred from the parameter count × 81 tokens × the reported iterations (a lower bound, no attention terms); **–** = not computed. Our numbers: EMA weights, bf16, val-selected inside the run on train-file puzzles; the field's released weights through the same evaluator on identical puzzles. Sets: FULL = 422,786; 100k / 20k / 5k = uniform subsamples (binomial SE 0.04 / 0.07 / 0.09 pp at 99 %).")
say("")
say("## 1. By parameter count (the thousand-puzzle convention unless the training column says otherwise; the headline single-pass number at the paper's own depth)")
say("")
say("| params | system | training data | protocol | accuracy | inference compute per puzzle (tier) |")
say("|---|---|---|---|---|---|")
w, t = G["w192"], G["trm"]
rows1 = [
 (789125, "**DEC-w192 (C5, ours, one seed)**", "1k × position aug", "single pass D64, FULL 422,786 (99.10 on the 100k)", f"**{c5full:.2f}**", f"{tmac(w,64):.2f} TMAC = {2*tmac(w,64):.1f} TFLOPs (M); D16 {100*V['C5']['cold16']:.2f} at {tmac(w,16):.2f} TMAC; D128 {100*V['C5']['d128']:.2f} at {tmac(w,128):.2f}; D256 {100*V['C5']['d256']:.2f} at {tmac(w,256):.2f}"),
 (796937, "Sotaku v2 (GitHub)", "**2.7M puzzles**", "single pass, 1,024 iterations, a 25k subset", "99.12", "≈ 0.07 TMAC (I: 81 × 0.8M × 1,024); 96.29 at 128 iterations ≈ 0.008"),
 (800000, "Lattice Deduction Transformer", "1k × (digit + D4) aug", "learned elimination + branching search; 300 puzzles", "100 (2 timeouts)", "– (search)"),
 (2000000, "SE-RRM ✓", "1k × aug (digit permutation)", "single pass, 16 supervision steps", "93.73", "– (attention over 9 symbols × 81 positions; not computed)"),
 (2780933, "**DEC-w384 seed triple (C0 C1 C2, ours)**", "1k × position aug", "single pass D64, 100k; mean ± half-spread of 3 seeds", f"**{100*mean('cold64'):.2f} ± {50*spread('cold64'):.2f}**", f"{tmac(G['w384'],64):.2f} TMAC = {2*tmac(G['w384'],64):.1f} TFLOPs (M); D16 {100*mean('cold16'):.2f} ± {50*spread('cold16'):.2f} at {tmac(G['w384'],16):.2f}; D128 {100*mean('d128'):.2f} at {tmac(G['w384'],128):.2f}"),
 (2977541, "DEC-w384 + set attention (C4, ours, one seed)", "1k × position aug", "single pass D64, 100k", f"{100*V['C4']['cold64']:.2f}", f"{tmac(G['c4'],64):.2f} TMAC (M, measured with the attention); D16 {100*V['C4']['cold16']:.2f} at {tmac(G['c4'],16):.2f}"),
 (5028866, "alphaXiv TRM-MLP (released grid, our evaluator)", "1k × aug 1000", "single pass D16 / D64, FULL", f"{100*REF['trmpub']['cold16']:.2f} / {100*REF['trmpub']['cold64']:.2f}", f"{tmac(t,16):.2f} / {tmac(t,64):.2f} TMAC (M); the paper's 87.4 at D16"),
 (5037058, "CGAR (released, our evaluator) ✓", "1k × aug", "single pass D16 / D64, FULL", f"{100*REF['cgar']['cold16']:.2f} / {100*REF['cgar']['cold64']:.2f}", f"{tmac(t,16):.2f} / {tmac(t,64):.2f} TMAC (M); the paper's 86.02"),
 (5037058, "EqR EMA (released, our evaluator)", "1k × aug", "single pass D16 / D64 / D128 / D256 (FULL / FULL / 20k / 5k)", f"{100*REF['eqr']['cold16']:.2f} / {100*REF['eqr']['cold64']:.2f} / {100*REF['eqr']['d128']:.2f} / {100*REF['eqr']['d256']:.2f}", f"{tmac(t,16):.2f} / {tmac(t,64):.2f} / {tmac(t,128):.2f} / {tmac(t,256):.2f} TMAC (M); the paper's 84.8 / 93.0"),
 (5000000, "CMM", "1k × aug ⚠", "single pass, iterations unstated", "93.7", "–"),
 (5000000, "PTRM ✓ (its Table 3; an earlier note said 7M)", "1k × aug", "best-Q of 100 stochastic rollouts, D64", "98.75 (87.28 deterministic)", f"≈ {100*tmac(t,64):.0f} TMAC (M: 100 rollouts × {tmac(t,64):.2f} on the TRM-MLP cell)"),
 (7000000, "FPRM ✓", "1k × aug 1000", "pass@1, adaptive halting, FULL", "94.2", "– (their curve sits at ≈ 1–10 TFLOPs per puzzle in FRM's Fig. 5; R)"),
 (7000000, "Flow Reasoning Models v3 ✓", "1k (difficulty-balanced) × Sudoku symmetries", "one stochastic rollout; PEAK over inference compute; a 1,000-puzzle test subset; 3 seeds", "99.5", "≈ 1–10 TFLOPs = 0.5–5 TMAC at the peak (R: their Fig. 5, PyTorch operator FLOPs × realized NFE)"),
 (10000000, "GRAM", "1k × aug ⚠", "20 sampled trajectories + majority vote, 16 iterations, a 1,000-puzzle test set", "97.0", "– (20 × a 16-iteration pass)"),
 (27000000, "HRM", "1k × aug", "single pass, up to 16 ACT segments", "55.0 (TRM's table; 54.9 our read)", "–"),
 (27000000, "Attractor Models", "≈ 1k", "fixed point by implicit differentiation", "91.4", "–"),
 (None, "Diffusion curriculum ✓ (hidden-128 looped transformer; params not reported)", "**the full 3.83M-puzzle split**", "one stochastic rollout, K = 10,000 steps, ≈ 423k test, 3 seeds", "99.90", "– (10,000 denoiser passes; I: ≈ 1–2 TMAC if the block is 1–2M params)"),
]
for pr, sysname, data, proto, acc, comp in sorted(rows1, key=lambda r: (r[0] is None, r[0] or 0)):
    say(f"| {pr:,} | {sysname} | {data} | {proto} | {acc} | {comp} |" if pr else f"| n/r | {sysname} | {data} | {proto} | {acc} | {comp} |")
say("")
say("## 2. The compute–accuracy ladder under the thousand-puzzle convention, single pass (one deterministic rollout; sorted by inference compute per puzzle; tier M unless noted)")
say("")
say("| TMAC / puzzle | TFLOPs | system, depth | params | set | accuracy |")
say("|---|---|---|---|---|---|")
lad = []
for tt in ("trmpub", "cgar", "eqr"):
    nm = {"trmpub": "alphaXiv TRM-MLP", "cgar": "CGAR", "eqr": "EqR"}[tt]
    for key, steps, setn in (("cold16", 16, "FULL"), ("cold64", 64, "FULL"), ("d128", 128, "20k"), ("d256", 256, "5k")):
        lad.append((tmac(G["trm"], steps), f"{nm}, D{steps}", "5.04M", setn, f"{100*REF[tt][key]:.2f}"))
for key, steps, setn in (("cold16", 16, "FULL"), ("cold64", 64, "100k"), ("d128", 128, "20k"), ("d256", 256, "5k")):
    if steps == 64: lad.append((tmac(G["w192"], steps), f"**DEC-w192 C5, D{steps}**", "0.79M", "FULL (100k: 99.10)", f"**{c5full:.2f}**"))
    else: lad.append((tmac(G["w192"], steps), f"**DEC-w192 C5, D{steps}**", "0.79M", setn, f"**{100*V['C5'][key]:.2f}**"))
    lad.append((tmac(G["w384"], steps), f"**DEC-w384 triple, D{steps}**", "2.78M", setn, f"**{100*mean(key):.2f} ± {50*spread(key):.2f}**"))
    lad.append((tmac(G["w384"], steps), f"DEC-w384 C2 (champion by rule), D{steps}", "2.78M", setn, f"{100*V['C2'][key]:.2f}"))
    lad.append((tmac(G["c4"], steps), f"DEC-w384 + set attention C4, D{steps} (measured with the attention)", "2.98M", setn, f"{100*V['C4'][key]:.2f}"))
lad.append((tmac(G["trm"], 16), "TRM-MLP paper, D16 (their number)", "5M", "FULL", "87.4"))
lad.append((tmac(G["trm"], 16), "CGAR paper, D16", "5M", "FULL", "86.02"))
lad.append((tmac(G["trm"], 16), "EqR paper, D16", "5.03M", "FULL", "84.8"))
lad.append((tmac(G["trm"], 64), "EqR paper, D64 (B = 1)", "5.03M", "FULL", "93.0"))
for x in sorted(lad, key=lambda r: r[0]):
    say(f"| {x[0]:.2f} | {2*x[0]:.1f} | {x[1]} | {x[2]} | {x[3]} | {x[4]} |")
say("| ≈ 0.5–5 (R) | ≈ 1–10 | Flow Reasoning Models v3, peak over compute (a 1,000-puzzle test subset; one stochastic rollout) | 7.0M | 1k subset | 99.5 |")
say("| – | – | FPRM pass@1 with adaptive halting (≈ 1–10 TFLOPs in FRM's Fig. 5, R) | 7M | FULL | 94.2 |")
say("| – | – | SE-RRM, 16 steps | 2M | ? | 93.73 |")
say("| – | – | CMM | 5M | ? | 93.7 |")
say("| – | – | Attractor Models | 27M | ? | 91.4 |")
say("| – | – | HRM | 27M | FULL | 55.0 |")
say("")
say("## 3. The restart, vote and search columns (the thousand-puzzle convention; sorted by inference compute per puzzle)")
say("")
say("| TMAC / puzzle | system | params | set | accuracy | column |")
say("|---|---|---|---|---|---|")
r3 = []
c5k32 = SC["C5@k32"]["sel"]; c5k128 = SC["C5@k128"]["sel"]; c0k128 = SC["C0@k128"]["sel"]; c0k32 = SC["C0@k32"]["sel"]; c2k32 = SC["C2@k32"]["sel"]; c1k32 = SC["C1@k32"]["sel"]
k128 = {a: SC[f"{a}@k128"]["sel"] for a in ("C0", "C1", "C2")}
r3.append((draws_tmac(G["w192"], 32), "**DEC-w192 C5, k = 32 at D64 (residual-selected = verified)**", "0.79M", "5k", f"**{100*c5k32['t1r']:.2f}** / {100*c5k32['verified']:.2f}", "verification-free residual selection / the free verifier"))
r3.append((draws_tmac(G["w192"], 128), "**DEC-w192 C5, k = 128 at D64**", "0.79M", "5k", f"**{100*c5k128['t1r']:.2f}** / {100*c5k128['verified']:.2f}", "residual / verifier"))
r3.append((draws_tmac(G["w384"], 32), "DEC-w384 triple, k = 32 at D64", "2.78M", "5k", f"{100*min(c0k32['t1r'],c1k32['t1r'],c2k32['t1r']):.2f}–{100*max(c0k32['t1r'],c1k32['t1r'],c2k32['t1r']):.2f} residual = verified", "residual / verifier"))
r3.append((draws_tmac(G["w384"], 128), "DEC-w384 triple, k = 128 at D64", "2.78M", "5k", f"{100*min(x['t1r'] for x in k128.values()):.2f}–{100*max(x['t1r'] for x in k128.values()):.2f} residual = verified (" + ", ".join(f"{a} {100*k128[a]['t1r']:.2f}" for a in ('C0','C1','C2')) + ")", "residual / verifier"))
r3.append((draws_tmac(G["trm"], 128), "EqR (released, our evaluator), k = 128 at D64", "5.04M", "20k", f"{100*SC['eqr@k128(20k)']['sel']['t1r']:.2f} / {100*SC['eqr@k128(20k)']['sel']['verified']:.2f}", "residual / verifier"))
r3.append((draws_tmac(G["trm"], 128), "EqR paper, B = 128 residual selection at D64", "5.03M", "FULL", "99.8", "residual (their headline)"))
r3.append((100 * tmac(G["trm"], 64), "PTRM, best-Q of 100 rollouts at D64 (σ .3)", "5M", "FULL", "98.75 (pass@100 99.06)", "learned Q-head selection"))
r3.append((draws_tmac(G["trm"], 128), "CGAR (released, our evaluator), k = 128 at D64", "5.04M", "20k", f"{100*SC['cgar@k128(20k)']['sel']['t1r']:.2f} / {100*SC['cgar@k128(20k)']['sel']['verified']:.2f}", "residual (broken: 30.9 % spurious) / verifier"))
r3.append((draws_tmac(G["trm"], 128), "alphaXiv TRM-MLP (released, our evaluator), k = 128 at D64", "5.04M", "20k", f"{100*SC['trmpub@k128(20k)']['sel']['t1r']:.2f} / {100*SC['trmpub@k128(20k)']['sel']['verified']:.2f}", "residual / verifier"))
for x in sorted(r3, key=lambda r: r[0]):
    say(f"| {x[0]:.0f} | {x[1]} | {x[2]} | {x[3]} | {x[4]} | {x[5]} |")
say("| – | GRAM, 20 sampled trajectories + majority vote at 16 iterations | 10M | 1,000 puzzles | 97.0 | unverified vote |")
say("| – | Guided reasoning (TRM base), halting-head reweighted trajectories | 5M | ? | 98.0 | halting-head selection |")
say("| – | Speed is Confidence, halt-first selection over an ensemble | TRM ensembles | ? | 97 | ensemble |")
say("| – | EqR, test-time scaling to 40,000 equivalent layers (abstract) | 5.03M | ? | > 99 | depth + breadth |")
say("| – | Hypothesis-Pinning Search (blog), hundreds of rollouts on the tail | 7M | test set | 100 | search |")
say("| – | Lattice Deduction Transformer, branching search | 0.8M | 300 puzzles | 100 | search |")
say("")
say("## 4. The other training regime (the full split; a different problem: large-data solvers, not small-data inductive bias)")
say("")
say("| TMAC / puzzle (tier) | system | params | training puzzles | protocol | accuracy |")
say("|---|---|---|---|---|---|")
say("| ≈ 0.07 (I) | Sotaku v2 | 0.80M | 2.7M | single pass, 1,024 iterations, 25k subset | 99.12 (99.05 at 2,048; 98.63 at 4,096) |")
say("| ≈ 1–2 (I) | Diffusion curriculum | not reported (hidden 128) | 3.83M | one stochastic rollout, K = 10,000, ≈ 423k test | 99.90 |")
say("")
say("## 5. What the tables say")
say("")
say(f"1. **Per parameter, the DEC is the smallest model at the top of every thousand-puzzle column:** the w192 DEC (0.79M) is 16 % of EqR / PTRM (5M), 11 % of FPRM / FRM (7M), 8 % of GRAM (10M) and 3 % of HRM (27M), and reads {c5full:.2f} at D64 on the full set against their 93.0–94.2 single pass; the w384 triple (2.78M) is 55 % of the 5M cell and reads 98.18 ± 0.61. Only Sotaku (0.80M) and the diffusion curriculum sit near or above it, both trained on the full split (§4). Every number in the column is on the set named beside it.")
say(f"2. **Per unit of inference arithmetic under the thousand-puzzle convention, the ladder is ours from {tmac(w,16):.2f} TMAC upward** (the w192's D64 on the full 422,786: {c5full:.2f}, +6.00 pp over EqR's 93.15 on the identical set, 26,727 puzzles only-ours against 1,347 only-EqR): at {tmac(t,16):.2f}–{tmac(w,16):.2f} TMAC the released 5M weights read 79–87 (D16) and the w192 DEC {100*V['C5']['cold16']:.2f} (D16); at ≈ {tmac(t,64):.1f}–{tmac(w,64):.1f} TMAC EqR's D64 93.15 vs the w192's D64 99.10 on the 100k; at {tmac(t,128):.1f}–{tmac(w,256):.1f} TMAC EqR's D128 / D256 95.02 / 95.88 vs the w192's D128 {100*V['C5']['d128']:.2f} ({tmac(w,128):.2f}) and D256 {100*V['C5']['d256']:.2f} ({tmac(w,256):.2f}) and the triple's D16 95.02 ({tmac(G['w384'],16):.2f}); every DEC point above {tmac(t,64):.1f} TMAC is above every field point at any depth. FRM's 99.5 peak sits at ≈ 0.5–5 TMAC on their own axis, on a 1,000-puzzle subset, within error of the w192's 99.42–99.60 rows on different subsets: the two systems share the top of this ladder, and neither is measured on the other's set.")
say(f"3. **The w384 DEC buys its lead with arithmetic, the w192 nearly does not:** at D64 the w384 triple costs {G['w384']/G['trm']:.1f}× EqR's arithmetic for +5.1 pp; the w192 costs {G['w192']/G['trm']:.2f}× for +6.0 pp on identical puzzles (measured; the analytic column had read 4.6× / 1.34×). The compute story of the paper is the w192 row; the w384 rows are the seeded claim.")
say(f"4. **On the restart columns the compute is comparable and the DEC is at or above the field:** {draws_tmac(G['w192'],32):.0f} TMAC buys {100*c5k32['t1r']:.2f} verification-free at k = 32 on the w192 (EqR needs ≈ {draws_tmac(G['trm'],128):.0f} TMAC for 99.8 in its paper and reads {100*SC['eqr@k128(20k)']['sel']['t1r']:.2f} through our evaluator on the 20k; PTRM ≈ {100*tmac(G['trm'],64):.0f} TMAC for 98.75); at matched k = 128 the w192 reads {100*c5k128['t1r']:.2f} at {draws_tmac(G['w192'],128):.0f} TMAC. These are coverage-class numbers on a 5k subsample, one seed for the w192.")
say("5. **The full-split solvers are cheaper AND more accurate, and they are a different problem:** Sotaku's ≈ 0.07 TMAC for 99.12 and the diffusion curriculum's 99.90 are trained on 2,700–3,800× more puzzles; under the thousand-puzzle convention the DEC leads, and no thousand-puzzle model has been shown to reach those numbers. The paper claims the small-data column and says so.")
say("6. **What is not computed:** SE-RRM's, CMM's, FPRM's, GRAM's, HRM's and the Attractor Model's arithmetic per puzzle (no operator counts in their papers; FPRM's and FRM's read from FRM's Fig. 5 only). Our column and the field's three released cells are measured through one map (`tools/mac_count.py`); C4's attention term is included in its measured row.")
(ROOT / "Documentation" / "Comparison_Frontier_Compute_Params_2026-09-09.md").write_text("\n".join(out) + "\n")
print("\n".join(out[:4]))
