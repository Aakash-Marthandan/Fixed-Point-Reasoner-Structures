# Reflective Pass — Full-Program Audit and Reorientation (2026-08-21)

*PI-directed bird's-eye review after the rung-1c verdict. Companions: `Design_Ledger.md` (all evidence cited here), `Report_2026-08-21_Rung1c_Verdict.md`, `Related_Work_EqR_FPRM.md`, the 2026-08-14 budget architecture + one-plan amendment. Deadlines verified today against the CFPs (§4.3). This document is analysis + a proposed plan; nothing is registered until the PI ratifies.*

---

## 0. The goal, operationalized

The PI's statement: a top-tier peer-reviewed paper whose solid proof is **frontier-class scores with a clear Sudoku-vs-ARC study that continues beyond EqR and FPRM, interpretable and physics-informed, with mechanisms that just work and serve the convert phase and the community.**

As testable claims, the paper needs:
- **(G1)** A frontier-class *number* on at least one benchmark, at our efficiency point, under a comparable protocol.
- **(G2)** The cross-domain landscape study (one architecture, one instrument suite, both domains) that *explains* when equilibrium-reasoning mechanisms pay — going beyond EqR/FPRM rather than trailing them.
- **(G3)** The physics/interpretability layer (priced information, basin measurement) doing real explanatory work on (G1)+(G2), not decorating them.
- **(G4)** Conversion mechanisms with measured, reusable value.

## 1. Where we actually stand (full-stack audit)

### 1.1 The goal-facing numbers (the honest scoreboard)

| metric | value | last moved | comparator frontier |
|---|---|---|---|
| ARC val-hard (48 curated) task-level | **2/48** (C.3′: keyhole+snap+vote, old-arch candidates) | **2026-08-09** | — (internal gate) |
| ARC val-hard per-output | 17/144 (oracle union 24/144) | 2026-08-09 | — |
| dev-30 | 1–2/30, flat across 5 protocols | 2026-08-05 | — |
| ARC-1 public eval | **unmeasured** | never | CompressARC 20% @76k · TRM 45% @7M · FPRM **47.5% @7M no-TTT** · LLMs 93–98% |
| Sudoku (generator, ours) | 97.5 / 90 / 10 % at 50/40/30 givens @ 77k, T6; **+16/40 at t=64** on the hard stratum | 2026-08-15 | EqR **99.8% @5.03M (D=64, B=128)** on Sudoku-Extreme; TRM-lineage ≈87% (verify at vendoring) |
| Sudoku-Extreme (the benchmark) | **unmeasured** (not vendored) | never | ↑ |

**The last goal-facing motion anywhere was twelve days ago.** Everything since (rungs 0→1c, ~$367, Aug 14–21) measured substrate quality.

### 1.2 What is genuinely banked (defensible, seeded, ours alone)

1. **The four laws** (L1 throat 785→343 nats monotone; L2 N/r̄ separability, 6 confirmations; L3 restated; L4 priced≫plain transfer, 6/6 p<.05 at three widths) + the two-profile spectral structure (H-34, floors-enforcement scoped).
2. **The cross-domain dissociation** — the program's cleanest two-sided result: the same extended-iteration instrument reads ARC as *landscape-limited* (t=96 → +0) and Sudoku as *depth-limited* (t=64 → +16/40, near-misses at median 1 violation); retention=100% on Sudoku at every difficulty while cold-start solve falls 97.5→10%. Basin existence vs reachability vs depth, decomposed.
3. **H-37, the init-distribution law** — the strongest statistics in the program (sudB>sudA 38/40 paired, p≈1e-10; the 50-given reversal 35/36; ARC Dri condensation): RI pays exactly on its training init distribution. This *explains EqR's biggest lever* (Maze 44.9→68.6) and predicts where it transfers.
4. **The conversion physics**: basins exist but cold starts miss them (exact/N ≈ .25–.30, wrong-stable ≈ .90 — invariant under width/price/budget/floors/NI/RI/coupling); candidate supply × basin decoding is what converts (C.3′); anti-Arrhenius (cold candidates only); 31× basin-existence conditioning; TTT must be basin-preserving (spine motion erodes, 3× confirmed).
5. **This week's additions**: budget tax confirmed-and-scoped (H-40 on floors substrates; H-42 free-bits memorization pocket — S2 measured *inside* priced arms); β-inelasticity of transfer over 20×; H-36 resolved (width re-opens); η progress clock; e1e3 width/budget dissociation; A11/A12 baselines.
6. **Assets**: 54 pretrained substrates on disk (d16–d64, all arms), complete S-port harness (`sudoku.py`, `probe_sudoku.py`, `analyze_sport.py`, sudA/sudB), DP trainer validated on v6e-8, the one-pod ops stack + durability ladder (preemption loss ≤5 min), exact-S9 equivariance (= Sudoku's digit symmetry, by construction — neither EqR nor FPRM has it), the pre-registration/analyzer discipline itself.

### 1.3 What has NOT moved, despite everything

- **The reachability wall**: ~90% of converged limits are wrong at every scale ever measured. No substrate intervention touched it. The rung-1 report said it plainly: *"pricing buys existence, not reachability, at every width... untouched by width or price."*
- **The vacancy floor**: ExtractObjects 0/9 on ~48 substrates (Copy now cracking at 1–2/9 on A5-class cells — watch). Architectural.
- **Task-level conversion**: 2/48 for twelve days, and the champion stack still runs on *pretrain8-era d16 substrates + old-arch candidates* — **the ladder's product (44–48-codeword A5-class substrates at d48–d64) has never been fed into the conversion stack at all.**

### 1.4 Budget and calendar position

Spend: **$582** lifetime ($215 ledger-carried + $367.11 measured). Remaining: **≈$3.3k**. The ladder consumed $367 and 7 calendar days for rungs 0–1c. Money is not the binding constraint anywhere in what follows; **calendar and analysis attention are.**

---

## 2. The drift diagnosis

### 2.1 The hypothesis scoreboard discriminates the two research lines

Substrate-geometry line (H-32, 34, 35, 36, 38, 39, 40, 41, 42): **2 falsified, 2 indeterminate, the rest confirmed-then-rescoped**; no design lever that moved a goal metric. Cross-domain/dynamics line (H-33→H-37, depth-limitation): **the program's strongest effect sizes and cleanest mechanisms** (p≈1e-10 paired; +40% absolute solve from inference depth; 3.8× rate ratios).

The mechanism of the difference is not luck. The substrate surface's effect sizes (5–10 counts on 288) sit **at the measurement-noise floor** (seed spreads 1–13), so every round needs n≥3 to say anything and even confirmed effects are modest shifts in a *leading indicator* — while Sudoku's single-attractor, verifiable, resamplable structure yields 10× cleaner signal. **Mechanisms visibly "just work" where the domain lets them.**

### 2.2 This is a named, recurring failure mode

2026-07-30 named it *measurement-refinement gravity / falsification-chain gravity*: every result spawns cheap, legitimate follow-ups that outcompete the lumpy item deciding whether a working result exists. The rung ladder relapsed into it at program scale: H-36's confound → rung 1b; rung 1b's n=1 → rung 1c; rung 1c's indeterminate corner → (the offered A10s2). Each step locally correct, pre-registered, rigorous — and the sequence optimized the leading indicator while the goal metric sat still. *The optimizer took transport over computation, again.*

### 2.3 What stays true from the ladder

The ladder was not waste: L4 seeded at three widths is paper-load-bearing; H-36's resolution (width re-opens; budgets held at 40k) prevented a wrong depth-lean commitment; A11/A12 are the anchors any scaling claim needs; H-42 is a genuinely novel S2 instance. The error was **sequencing and stopping rules**, not the science. The correction is structural (§6), not a repudiation.

---

## 3. Goal-gap analysis: where is frontier-class actually reachable by the deadlines?

### 3.1 Sudoku-Extreme — reachable, and the evidence says how

- EqR's 99.8% config is **D=64 iterations, B=128 multi-init** at 5.03M params. Our T6-trained 77k map already recovers half the hard stratum at **t=64** — *their winning depth, found independently by our depth sweep*. The remaining gap decomposes into known, cheap mechanisms, each with our own or their measured effect size: training-depth T (our T6 vs their D=64 regime), **RI rows** (their +24pp lever; H-37 says it pays exactly under multi-init deployment), **verify-and-vote multi-init** (Sudoku solutions are checkable ⇒ verification is free; sudB's per-puzzle hit-rate median 11/16 at 40 givens is waiting to be harvested), NI (spurious-attractor suppression in its home regime), FPOpt inference damping.
- **Exact-S9 equivariance is Sudoku's digit-relabeling symmetry by construction** — an inductive bias neither comparator has, worth the most in exactly the benchmark's small-data regime (TRM-lineage: ~1k training puzzles; verify protocol at vendoring).
- Scale: 9×9 canvases at d16–d32 → pretrains in minutes, campaigns in $1–5. A full sprint is ~$40–80.
- Honest banding (pre-registered, §4.1): **M1 ≥50%** = mechanisms work · **M2 ≥85%** = TRM-class at ~10–100× fewer params · **M3 ≥95%** = EqR-class. The paper's frontier claim survives at **M2** (efficiency frontier + inductive-bias story + the law); M3 is the stretch. Named risk if M1 fails: the 3-adic/dyadic pooling mismatch (registered at the S-port launch) — box-aligned pooling is the pre-named fix.

### 3.2 ARC — frontier-parity is NOT credible by Sept; the honest lane is efficiency + explanation

From 2/48 val-hard, FPRM-parity (47.5% @7M) by the freeze is not a plan, it is a wish. What *is* credible: (i) feed the ladder's substrates into the conversion stack for the first time; (ii) run the two registered-but-unlaunched supply mechanisms (portfolio-multi-init; repulsion); (iii) take the eval-6 gate; (iv) post the program's first **ARC-1 public-eval number** under the registered matched-scoring protocol, positioned honestly on the ≤1M-param efficiency lane (CompressARC-class comparator — a lane the 07-30 review already identified as ours: mdlARC's 75M "leaves the ≤400k frontier untouched"). Target band: double per-output (17→25–30/144), 4–8/48 task-level, ARC-1-eval ~10–20% — valuable at any point in the band because the *mechanism study* (what converts and what measurably cannot) is the contribution, with the wall and the vacancy floor as named limits.

### 3.3 The paper this makes (the contribution statement)

1. **The landscape-class law** (G2): one architecture + one instrument suite across Sudoku and ARC — basin existence / reachability / depth decomposed; H-37; anti-Arrhenius; *explains* both fields' architecture choices (deep unrolls justified for CSP, useless for inventory-limited ARC) and predicts where EqR-class mechanisms transfer.
2. **The frontier point** (G1): Sudoku-Extreme at M2+ with ~10–100× parameter efficiency and the S9 mechanism story.
3. **The priced-code physics** (G3): the four laws, two-profile structure, H-42's free-bits memorization — the information-theoretic reading that makes 1–2 interpretable rather than empirical.
4. **The conversion study** (G4): candidate-supply × basin-decoding as the working converter; basin-preserving adaptation as the constraint; the honest ARC efficiency number.

This is coherent at AAMAS grade with M1 + (1)(3)(4); it is ICLR-grade if M2 lands by the abstract.

---

## 4. The reoriented plan (proposed for PI ratification)

### 4.1 Three workstreams

**WS-A — S-port frontier sprint (the main quest).**
A0 vendor Sudoku-Extreme + protocol verification (train-set size, augmentation, scoring) + comparability registration: arm (i) strict-comparable (their training data only; S9 covers digit relabeling free), arm (ii) generator-pretrained, reported separately. A1 calibration battery of sudA/sudB on the real test distribution. A2 **depth ladder**: T ∈ {6,12,24} × inference t ∈ {64, 256, 1024} + FPOpt damping. A3 **RI + NI arms** (H-37's deployment condition satisfied by design). A4 **verify-and-vote multi-init** (k=16–64; free verification). A5 priced-vs-plain (Law-4 on CSP + the throat/profile of a *solving* substrate — the interpretability figure). Milestones M1/M2/M3 as §3.1, each with its gate; instruments (retention, ladders, e1e3, spectra) ride every arm — the law data and the score come from the same runs. Cost ≈ $40–80, v5e/v6e lanes (different quota family; precedented alongside the pod).

**WS-B — ARC convert phase (the solve-rate flank).**
B1 **S2-cell portfolio-multi-init** (registered 08-12, unlaunched): cold candidates from the 54-substrate portfolio through the best decoders. B2 **conversion on the new substrates** (A11/A12/A5-class + anchor-class as decoders — first time the ladder's product reaches the converter). B3 branch on what binds: supply → **repulsion build** (cluster I, registered-unbuilt); selection → PoE/residual re-adjudication. B4 **eval-6 dev-30 single shot (PI gate)** → **ARC-1 public-eval matched-scoring run** (the 08-07 registered protocol) — the external number. B1/B2 spend val-hard adjudications (~#16–17) — counted, steering-only as always. Cost ≈ $60–150.

**WS-C — d96 confirmatory (time-boxed, non-blocking).**
The pre-committed rung 2 runs as a **background confirmatory**, not a quest: one pod-day (A5-class @40k ×2–3 seeds + plain ×1, the already-drafted predictions: rg-96 ≥23.5 kill <21, N 42–50, I<499, knee L1≤.15, η .21–.22, IR34 30–35, ExtractObjects 0), half-day analysis, one report section. Optional $8 piggybacks (A10s2; A5-d48@53k = H-42's test) ride the same pod. Hard cap: if churn threatens >1.5 pod-days, it defers to the AAMAS window. Cost ≈ $60–90.

### 4.2 What we STOP (explicit)

- **Rungs 3–4 (d128/d192) and the FPRM param-parity chase**: descoped from this paper. The ≤1M efficiency lane is the claim.
- **Substrate-geometry side quests as auto-spawned follow-ups** (A10s2 etc. only as piggybacks; H-42's test is a $8 rider, not a campaign).
- **B1-full as a prerequisite**: stays registered; B1-lite (query flux-toll at fit time, ~30 lines) is admitted only as a WS-B arm if B1/B2 show selection binding.
- The old architecture retires from the *story* (heirloom candidates remain a labeled component where used).

### 4.3 Calendar (deadlines verified against the CFPs today)

**ICLR 2027: abstract Sep 18 AOE, full paper Sep 25 AOE. AAMAS 2027: abstract Oct 2, full Oct 9 (Hanoi, May 3–7).** Today is Aug 21 — 28 days to the ICLR abstract, 35 to full; one more week than the ledger's assumption.

| window | WS-A (Sudoku) | WS-B (ARC convert) | WS-C / writing |
|---|---|---|---|
| Aug 21–27 | A0–A3 (vendor, depth ladder, RI/NI) | B1–B2 (portfolio + new substrates) | d96 piggyback if weather allows |
| Aug 28–Sep 3 | A4–A5; **M1 gate** | B3 branch; **eval-6 gate** | d96 analysis (half-day) |
| Sep 4–10 | M2 climb (best config × seeds) | ARC-1-eval matched run | figures start; **Sep 10 venue checkpoint** |
| Sep 11–18 | freeze ~Sep 15; seeds/replications | freeze | **ICLR abstract Sep 18** |
| Sep 19–25 | — | — | full paper (ICLR) |
| (AAMAS mode) | results may run to ~Sep 26 | — | freeze Sep 28 · abstract Oct 2 · full Oct 9 |

Venue rule unchanged (one-plan amendment): same science plan; the Sep-10 checkpoint reads the state; M2-by-Sep-15 ⇒ write for ICLR, else the identical cadence lands AAMAS with three extra result-weeks.

### 4.4 Budget

WS-A $40–80 + WS-B $60–150 + WS-C $60–90 + churn ≈ **$250–370 total** against ~$3.3k remaining. Spot-first, one-pod-per-rung stands (Sudoku lanes are the precedented parallel track); on-demand only under the standing freeze-protection clause; the $550 rebuttal reserve is untouched.

---

## 5. Risks, honestly

| risk | odds | mitigation |
|---|---|---|
| M2 not reached by Sep 15 (ICLR miss on G1) | real — this is a climb, not a lay-up | AAMAS mode gains 11 result-days; paper stands at M1 on (G2)+(G3)+(G4); the banding is pre-registered so a partial climb is still a clean result |
| 3-adic pooling mismatch caps Sudoku | named since the S-port launch | box-aligned pooling variant is the pre-named fix; itself a finding (RG arity must match constraint arity) |
| ARC conversion stalls at ~2/48 even with supply mechanisms | possible | the paper's ARC section is the *mechanism study + limits*; the wall is a result we understand and state |
| Sudoku-Extreme protocol surprises (train-set/scoring conventions) | moderate | A0 verifies before any claim; comparability honesty is already binding policy |
| attention split across three streams | the real constraint | WS-C is hard-capped; weekly bird's-eye (§6) enforces the allocation |

## 6. Process guardrails (proposed standing policy — the anti-drift mechanism)

1. **Goal-metric gate on registrations**: every campaign registration names the end metric (Sudoku-Extreme %, val-hard/dev-30 task-level, ARC-1-eval %) it can move and by what mechanism. Substrate-metrology campaigns need explicit PI opt-in tied to a named paper figure.
2. **Indeterminates do not auto-spawn follow-ups**: a follow-up cell competes under guardrail 1 like any other proposal (r1b→r1c happened by auto-spawn).
3. **Power the decisive contrast**: no decision rule whose primary leg rests on an n=2 corner at known ~10-count noise; buy n=3 up front or pair within-run.
4. **Weekly bird's-eye**: one short reflective read at each week boundary against §0, one paragraph in the ledger.

---

*If ratified: this becomes a course-correction ledger entry; WS-A/WS-B launch registrations (decision rules pre-data, analyzers pre-written) follow per the standing discipline.*
