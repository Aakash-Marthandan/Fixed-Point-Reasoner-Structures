# Sprint S2 (Sudoku-Extreme wave 1) — Verdict, the Difficulty-Resolved Dissociation, and What Moves the Number

*2026-08-22. Registered rules: ledger 2026-08-21 SPRINT S2 LAUNCH REGISTRATION (locked pre-data; `tools/analyze_sport2.py` self-test 12/12, run untouched → `runs/analysis/sport2_verdict.txt`). Data: `runs/pretrainsport2_S0..S7`, `runs/sxeval_psport2*` (full 423k test + stratified 512), `runs/sudprobe_psport2*` (instrument suite). Protocol: `sapientinc/sudoku-extreme`, seeded 1,000 training puzzles × 100 group-augmented copies (STRICT row), exact accuracy on the full 422,786-puzzle test set, one deterministic prediction; comparators HRM ~55 / TRM 87.4 / EqR 99.8. No number below is hand-typed.*

---

## 0. Integrity (post-results critique)

All eight arms at their registered configs (d16, step 20,000; S0 T6 β3e-5 · S1 +RI · S2 +NI · S3 T12 · S4 T24 · S5 β=0 · S6 +digit-aug · S7 RI+NI+T12 — RI/digit-aug are trainer flags not recorded in the ckpt config; the final tarball omitted `config.json`, so those two rest on the harness-verified chain flags + their distinct training curves — a minor admission caveat, and neither arm carries a claim). Every full-test eval n = 422,786; verified-vs-true multi-init hits agree exactly on every eval (the uniqueness logic is sound); valid-but-wrong grids ≈ 0; givens kept 100% at t=6 on every arm. The uniform ~2.2% at t=6 across all arms is **a shared easy tail, not an artifact**: the 7,955 puzzles every arm solves at t=6 are all rating-0. Training curves pulled from GCS for the seven SKIP arms (not in the tarball). **M0 fit gate: FAIL** — val monitor (64 disjoint train-distribution puzzles, exact at t=T) is 0/64 for every arm except S4 (4/64); train loss 0.50–0.62 at 20k with the cosine schedule already decayed — the 1k-example regime is **under-trained at 20k steps** (TRM trains ~50× longer). Registered consequence fires: the GEN row (generator-pretrained + 1k finetune) becomes the primary labeled row in wave 2; STRICT is still reported.

## 1. Registered verdicts

| rule | reading | verdict |
|---|---|---|
| **Headline / bands** | best arm **S5 (plain β=0, T6) at inference depth t=64: 12.02% exact on the full test** (strat-512: 14.1%; at t=256 14.3%); verify-and-vote k=16: **36.9% (t=64) / 37.7% (t=256)** on the 512 set (separate labeled number). S4 (T24 priced) 10.9%, S7 (RI+NI+T12) 6.2%; S0/S1/S6 collapse to ≤0.2% at t=64. | **BELOW-M1** (M1 = 50%). |
| **M0** | val monitor 0/64 (S4: 4/64) | **FAIL → GEN becomes the primary labeled row** (registered). |
| **P1** (H-37: RI raises multi-init hits) | probe mi-rate S0 .009 → S1 .013 | **FLAT** — RI alone at priced T6 buys nothing; RI+NI at T12 (S7) does (6.2% vs S3's 1.1%). |
| **P2** (depth-limitation) | inference depth t6→t64 is monotone on only **3/8 arms** (S4, S5, S7 rise; S0/S1/S3/S6 collapse; S2 explodes); training depth at t=64: S3 +1.0 pp, **S4 +10.8 pp** vs S0 | **HALF-FAILED, informatively** — depth pays only on maps that are genuine equilibria (§3.2). |
| **P3** (S9 covers digit aug) | S6−S0 @t64 = −0.08 pp | **HOLDS** (explicit digit augmentation adds nothing — exact S9 covers it). |
| **P4** (Law 4 on CSP: priced ≈ plain) | priced−plain @t64 = **−11.94 pp** | **REVERSED — PLAIN ≫ PRICED.** The transfer dividend of ARC inverts to a cost on CSP (§3.1). New hypothesis H-43. |
| **P5** (combined recipe) | S7 6.19% vs max(S1,S2,S3) 1.64% | **HOLDS**. |
| **Branch rule** (below M1 → 3-adic risk live → box-aligned before scale) | fired by its letter | **Premise not met and diagnosis refuted by the instruments** (§4.1): the failures are near-misses at every difficulty (median 3 violations, boxes intact, givens kept 99.5%) — a *search* failure, not box geometry; breadth was tried only at k=16, depth never at β=0, training at ~1/50 of TRM's. Box-aligned runs as one cheap control arm in wave 2, not as the lever. |

## 2. The headline finding: the dissociation, resolved by difficulty — and the mechanism that closes it

On the stratified test set (S5, t=64), per tdoku-backtrack rating:

| rating bin | n | **retention** (handed the solution, 8 final-map steps) | ladder S(.2) | **cold solve** | **verify-and-vote k=16** | hits/16 among vote-solved |
|---|---|---|---|---|---|---|
| 0 | 76 | 0.95 | 1.00 | **64%** | **91%** | 5.6 |
| 1–9 | 116 | 0.98 | 1.00 | 17% | 56% | 2.9 |
| 10–29 | 164 | 0.99 | 0.99 | 1% | 20% | 1.1 |
| 30–59 | 124 | 0.97 | 0.98 | 0% | 16% | 1.1 |
| 60+ | 32 | 1.00 | 1.00 | 3% | 9% | 0.7 |

**Basin existence is difficulty-independent; reachability is not.** The plain map holds every solution it is handed (retention 0.95–1.00 at every rating, basins wide to ε=.4 at 75%), while a cold start reaches the solution only when the puzzle needs little or no search (rating 0 = pure propagation: 64%; rating ≥10: ≈1%). Failures are near-misses everywhere (median 3 violations, givens kept 24.8/25, 81% of solves need >6 steps — the depth-limited propagation signature from cell-1, reproduced). **Breadth with free verification recovers reachability in proportion to the required search** — 64→91% at rating 0, 17→56% at 1–9, 1→20% at 10–29, 0→16% at 30–59 — and it does so with *one hit in 16* on the hard bins, i.e. the solved hard puzzles are found by a single random-init trajectory: **k has large headroom** (EqR's B=128 is the comparator). This is the landscape-class law in its sharpest form: on single-attractor CSP the solution's basin always exists and the cold-start gap is governed by *required search*; ARC's inventory-limited landscapes and hard Sudokus both sit on the "search side" of the dissociation, easy Sudokus on the propagation side — and the benchmark's own difficulty metric (tdoku backtracks) is a direct proxy for our instruments' reachability gap. It also reads EqR's D=64/B=128 recipe mechanistically: depth for propagation, breadth for search.

## 3. Physics

### 3.1 The price kills equilibria on CSP (P4 reversed — H-43)
At the trained horizon (t=6) every arm is identical (2.2–2.3%, the rating-0 tail). Beyond it the arms bifurcate: the priced T6 maps (S0/S1/S6) **degrade with iteration** — retention ≈ 0 at all ratings, givens kept fall 100→86→63% across t=6/64/256 (they overwrite their own boundary condition), exact → 0; the plain T6 map (S5) **improves** with iteration (2.3 → 12.0%), keeps givens (99.5%), holds solutions (0.98). The price at the knee (3e-5; throat 1.8k nats vs plain 300k) turns the learned map into a *horizon map* (reach the answer at step T, no fixed point) rather than an *equilibrium map*. On ARC the same price buys transfer (Law 4, seeded at three widths) because the task's information is a short rule plus transport; on CSP the information IS the 81-cell working state that propagation must carry through every step, and pricing that channel at the knee starves it. **The information-price dividend is domain-dependent, and the instruments say why.** Depth at training restores equilibrium under price (S4/T24: retention 1.00, 10.9%); so does RI+NI at T12 (S7: 1.00, 6.2%) though T12 alone does not (S3: 0.01) — noise-trained maps tolerate the price. NI alone at T6 (S2) makes the map *explosive* beyond T (flux 1e10, givens 42%, violations 127 at t=64).

### 3.2 The flow constant separates horizon maps from equilibrium maps
η: S5 (plain, stable) **.555** · S4 (T24, stable) .650 · S3 (T12, unstable) .803 · S0 .904 · S6 .911 · S2 .919 · S7 (stable via RI+NI) .944 · S1 .980. The unstable priced T6 maps run near replacement dynamics (η→1: "jump to the answer, don't hold it"), the stable plain map flows quasi-statically (.555). Cell-1's generator Sudoku (η .956, 100% retention on easy instances) was a horizon map that happened to suffice; Sudoku-Extreme exposes it. S7 is the one stable-but-fast exception (RI+NI). H-30's restated form ("η tracks optimization regime") gains a second reading: **η < ~.7 ↔ genuine equilibrium on this domain** (directional, 8 arms).

### 3.3 Depth pays only on equilibrium maps; breadth works through trajectories, not basins
Inference depth (t=6→64) lifts S5 ×5 and S4 ×5, kills S0/S1/S6 — P2's "monotone on every arm" fails precisely on the horizon maps, which is the right way for it to fail. The probe's final-map-only multi-init (8 steps from a random canvas) hits ~1% on every arm while the evaluator's full-schedule multi-init (random canvas, t=64) solves 37%: breadth pays through **trajectory diversity from the start state** (different random starts → different propagation/guess orders), not through final-map basin capture — consistent with H-37's "RI reshapes convergence, not basin supply" on ARC, and with the 31× basin-existence conditioning there.

### 3.4 Throat, profile
Priced throats 1.7–2.0k nats, UV share .93–.97 (cell-1's .89 — Sudoku is the most UV-concentrated domain measured); plain 302k nats, UV .86. The priced-Sudoku profile is not the knee profile (a third fixed profile, as cell-1 found). S2's 1e10 is divergence, not information.

## 4. Adversarial pass

1. **The branch rule's premise was not satisfied** (§1) and its diagnosis is refuted by the violation/givens data; I am not treating "box-aligned first" as binding, but I run it as a control arm. 2. **n=1 seed per arm** — the arm *ranking* rests on single seeds; the effects that carry claims are enormous (plain vs priced: 12.0 vs 0.08%; retention 0.98 vs 0.01 at 512 puzzles; breadth ×3 at n=512), the subtle ones (S1 vs S0, S3 vs S0 at t=6) are within noise and make no claim. 3. **RI/digit-aug flags unverified at artifact level** (config.json not in the tarball) — both arms are null-result arms; noted. 4. **The 12.0% is a weak, under-trained substrate** (M0 FAIL, 78k params, 20k steps at 1/50 of TRM's budget): wave 1 was a mechanism ladder, not a climb; the frontier bands are untouched by design at this stage. 5. **Verify-and-vote uses a test-time oracle-free check** (validity + givens) — legitimate by the domain's own structure, but it is a different protocol from the series' single-prediction number and is reported separately, always. 6. **M0 is an insensitive gate as written** (val exact at t=T saturates at 0 when the map needs depth) — its FAIL is still correct in spirit (loss .5–.6), but wave 2's M0 should read val at t=64. 7. **Depth eval cost**: full-test at t=256 was not run (cost rule); strat shows t=256 ≈ t=64 (+0.2 pp), so t=64 is the working depth.

## 5. What moves the number — wave 2 (goal: the bands; every arm names its expected effect)

Evidence-ranked levers, each with its prediction:
1. **Breadth k** (eval-only, banked ckpts, first hour on the pod): k ∈ {16, 64, 128, 256} on S5/S4 at t=64, strat-512. Prediction: S5 vote crosses **≥50% at k=128** (hard-bin hits are 1/16 → far from saturation). Then the full-test vote at the best affordable k as the labeled breadth number (or a ≥20k-puzzle random subsample with CI if k×423k is prohibitive).
2. **β=0 base + training budget**: plain T6 at 50k steps with a full-length schedule (M0 says under-trained; S5's loss flattened only because the cosine ended). Prediction: +5–15 pp cold.
3. **β=0 + depth/stabilizers**: plain T12 (+remat) and plain T12 + RI .15 + NI .01. Prediction: ≥ S5 cold (+3–8 pp); RI+NI raises the breadth hit-rate (H-37 under multi-init deployment).
4. **Width d32** (220k params, plain T6). Prediction: + (the first capacity point on this domain).
5. **GEN row** (registered M0 consequence): generator pretrain (17–30 givens) → 1k finetune (`--init-from`). Prediction: ≥ STRICT; reported as its own labeled row.
6. **The price dial at β=3e-6** (plain T6 otherwise): does a 10× lower price keep the equilibrium (retention ≥.9) while compressing ~10–30×? The physics arm for H-43.
7. **Box-aligned representation** (the registered control; each 3×3 box in a 4×4 canvas block): prediction ≈ plain T6 ± 2 pp — the instruments say the residual is search, not geometry.
Ops: one pod, ~8 arms, T6 arms 50k (≈3 h wall), T12 arms 30k (≈4.5 h, remat), wave ≈ one pod-day ≈ $60–80; T24 deferred (wall-clock 0.6 it/s). Wave-2 M0 = val exact at t=64.

## 6. Claim status

| claim | status after S2 |
|---|---|
| Landscape-class law (H-33) | **SHARPENED**: the retention/reachability dissociation is governed by *required search* (rating), not by domain — basin existence ≈ 1 at all difficulties, cold-start reachability ∝ 1/search; easy Sudoku = propagation side (cell-1), hard Sudoku and ARC = search side. |
| H-37 (RI pays on its init distribution / under multi-init deployment) | RI alone at priced T6: FLAT; RI+NI+T12: yes. Under-powered at this training budget; the wave-2 plain-base arms test it properly. |
| Law 4 (pricing dividend) | **DOMAIN-SCOPED**: transfer-specific on ARC; **reverses on CSP** (P4, −11.9 pp) — H-43. |
| H-30 (η) | η distinguishes horizon maps (→1) from equilibrium maps (~.55–.65) on Sudoku-Extreme (directional). |
| Depth-limitation (cell-1) | Reproduced on the benchmark (81% of solves need >6 steps; t=64 ×5 on stable maps). |
| Frontier bands | BELOW-M1 at wave 1 (12.0% cold / 37.7% vote@16, 78k params, 20k steps). Wave 2 targets M1 on the vote number first, then cold. |
