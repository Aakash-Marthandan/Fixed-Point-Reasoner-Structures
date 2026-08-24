# Sprint S2 Wave 3a (Sudoku-Extreme) — Verdict: Breadth M1 Banked, the Collapse Scoped, and What the Ladder Carries

**Date:** 2026-08-24 · **Analyzer:** `tools/analyze_sport3a.py` (byte-untouched since the 2026-08-23 launch registration; self-test 18/18) · **Artifacts:** `runs/analysis/sport3a_verdict.txt`, `runs/analysis/sport3a_physics_20260824.txt` (physics pass, descriptive) · **Data:** `sport3a_final.tgz` (16 arms × 8 eval kinds + 13 val-best full evals, 160 banked ckpts + trajectories, 14 probes, 3 S5 breadth confirmations, 3 PHASE4 breadth20k, 4 filler scans).

Goal metric (guardrail 1): Sudoku-Extreme **full-test exact, cold** (M1 ≥ .50 / M2 ≥ .85 / M3 ≥ .95), with verify-and-vote breadth **labeled separately** — and, per the 2026-08-23 reframe, the EqR like-for-like number is **vote@128 @t64** (EqR 99.8 % is a D=64, B=128 figure at 5.03 M params).

## 0. Integrity (post-results critique)

- **Admission:** all 16 arms at step 50,000 with the registered Config flags exact at ckpt level (d16/d32, T12/T6, β knee vs 0, NI σ, `eq_coupled`, `fpa_k/eps`). Corpus-side flags (RI .5 on A2/A9/A10; aug 500 + wd 1e-3 on A8) are **chain-source-verified** (the committed launcher's flag table; config.json corroborates where it shipped — A2/A2s1/A3/A3s1, seeds correct; A9's live `fpa_ce` stream corroborates FPA) — labeled, not ckpt-verified.
- Every full-test summary n = 422,786; strat n = 512; val n = 64; retention n = 512. The 12 missing `val_best.txt` were regenerated deterministically (`select_ckpt.py`); the 13 shipped val-best full evals cross-check **13/13** against the regenerated selections. A2/A5/A6s1 have no val-best eval because their val-best step = 50k = final.
- The **W9 filler scan exists** (n=512, banked 00:36 IST) — the ops close's "skipped" was a tarball-manifest artifact; all four wave-2 funnel scans (W2/W3/W8/W9, k=256) are in evidence.
- The k=1024 scan's summary lacked vote keys above 256 (evaluator doubling-list cap). vote@512/@1024 were **recomputed exactly** from the per-draw `mi_first_hit` records and bit-match the summary at 128/256. The analyzer's "saturation 0.00 pp" line is this artifact; the recomputed tail is the honest read (evaluator fix owed, $0).
- **Rules applied as registered, no post-hoc edits.** Two rules fired in ways the mechanism data reinterpret (both discussed in §2): COLLAPSE CONTROL fired on its cold leg with the H-45 mechanism absent, and SEED NOISE = 15.32 pp makes every cold contrast WITHIN-SEED-NOISE — correctly reported, and itself a finding (H-46).

## 1. Registered verdicts

| Rule | Outcome | Numbers |
|---|---|---|
| BREADTH-CONFIRM | **M1-ON-BREADTH (full-test-grade)** | S5 20k vote@128 = **68.62 ± 0.64 %**, vote@256 = **80.68 %** (prediction 65–72 % hit); recomputed strat tail @512 = 89.6 %, **@1024 = 96.1 %** (unsaturated, +6.4 pp last doubling); hardest octile 87.5 % @1024 |
| HEADLINE | **BELOW-M1** | best cold = **A4s1 (plain T12 eq_coupled s1): 21.20 %** — program record (W2 19.37); A7s1 20.18; val-selected best A7s1 19.97 [35k] (labeled) |
| BREADTH (wave-3a arms, labeled) | BELOW-M1-ON-BREADTH | PHASE4 A4s1 20k vote@128 = 35.83 %; A2 27.7; A7s1 25.7 — all 50k T12 arms are **narrow funnels** vs S5's 68.6 |
| COLLAPSE CONTROL | COLLAPSED (by the cold leg) — **mechanism ≠ H-45** (§2.1) | A7 11.18 % (W2 − 8.2 pp) but retfm 1.00, λ<1 throughout, val-best 14.87 @35k; A7s1 = 20.18 % > W2 |
| FIX ARMS | A2 **FIX-HOLDS** · A3 FIX-PARTIAL · A4 FIX-FAILS · A9 FIX-FAILS | A2 17.73 retfm 1.00; A3 15.44 retfm 1.00 (pair spread **1.66 pp**, basins widened S(.4) .96); A4 7.86 (seed 21.20!); A9 2.79 (val-best 15.46 @15k) — **fixes do not compose** |
| PRICE CEILING | **PRICED-DEPTH-PAYS** · PRICE-FLAT (d32 budget) | A5 13.50 vs W9 5.24 (+8.3); A6 11.09 vs W8 9.35 (+1.7); all six priced arms retfm 1.00 at I ≈ 1.0–1.6 k nats |
| REGULARISER-OF-CHOICE | **A2** (margin 4.2 pp, within-noise) | best fix A2 17.73 vs best priced A5 13.50 |
| CONVENTIONAL | **CONVENTIONAL-FAILS** | A8 (aug×500 + wd×10): 0.03 %, retfm .27 — the wave's one clean H-45 collapse (§2.2) |
| SEED | NOISE = 15.32 pp | by mechanism: FPA 1.7 / priced-d32 0.7 / RI+NI+FPA 2.7 vs RI+NI 15.3 / eq_coupled 13.3 / plain 9.0 / priced-T12 7.1 |
| TRAJECTORY LAW | Spearman(λ_joint, retfm) = −0.42 (n=160) | range-restricted (8/16 arms never degrade); within-arm where degradation exists: A8 −.79, A2 −.78, A4 −.84, A10 −.95 |
| RECIPE DECISION | **A2 — plain T12 + RI .5 + NI .01** | best val-selected cold among non-collapsed arms (17.73) |

Per-arm cold @t64 (final ckpt): A4s1 21.20 · A7s1 20.18 · A2 17.73 · A3s1 17.09 · A3 15.44 · A5 13.50 · A6s1 11.78 · A7 11.18 · A6 11.09 · A10 9.04 · A4 7.86 · A5s1 6.37 · A9s1 5.51 · A9 2.79 · A2s1 2.40 · A8 0.03.

## 2. The two headline findings

### 2.1 The H-45 collapse is horizon-scoped; what remains at T12 is a different, milder pathology (H-46)

Wave 2's law said plain maps lose final-map contractivity with budget/capacity. Wave 3a **scopes it**: at T12, no plain arm ever loses the fixed point — A7 and A7s1 hold final-map retention 1.00 and λ<1 at all ten monitors through 50k. What actually moves is **cold reachability at fixed contractivity**: A7's monitor val runs 10.9 % @15k → 3.1 % @50k while its seed twin holds ~11 %; checkpoint-to-checkpoint val ranges are 3–8 pp over the last 25k steps; final-ckpt seed spreads reach 15.3 pp. The fixed points hold; the VOID-start watershed rearranges. Registered as **H-46 (cold-basin wander)** with its Phase-B test. Consequences: (i) the wave-2 W2 = 19.37 % @30k reading survives, but as a draw from a fluctuating band this recipe class occupies (A7s1 20.18 @50k, A7 14.87 @35k) rather than a stationary property of "T12@30k"; (ii) cold single-checkpoint numbers on plain-base T12 arms are weak estimators — val-selected reporting and trajectories are the instruments of record, exactly what the wave-3a protocol added.

### 2.2 The conventional control is the cleanest H-45 confirmation, and the precursor ordering holds

A8 (plain T6, aug ×500, wd ×10, 50k) collapses on the wave-2 schedule despite five times the data and ten times the weight decay: λ_joint crosses 1 at 10k, final-map retention drops below .9 at 15k (**λ precedes retention loss — the registered precursor ordering, confirmed**), ending at retfm .27, cold 0.03 %. Conventional regularisation does not touch the pathology; NI, price, FPA, and training depth all do. This is a strong paper datum: the collapse is a dynamical-stability failure, not a data-quantity or weight-norm problem.

## 3. Physics — four lenses

### 3.1 RG lens: the contractivity flow and the η endpoint spectrum
η endpoints stratify exactly by mechanism: **NI arms train to η = 1.000/0.999/0.984** — literal replacement dynamics, the damping crutch removed, so contraction is forced into the map f itself. A2 (seed 0) passes through a chaotic transient (λ-spikes to 4.35 at 25–30k, retfm dipping .39–.94) and **settles self-contractive** (λ .78–.83, retfm 1.00, 35–50k); A2s1 never settles (λ 1.0–1.6, retfm .52–.72 throughout). The 15.3 pp seed spread is a **phase-boundary phenomenon at d16**, not noise — RI+NI at η=1 sits at the edge of the contractive phase. FPA holds η mid-range (.71–.75; the anchor loss penalizes overshoot); plain T12 .61–.76; priced .87–.93. The priced arms' trajectories make the H-43→H-44 transition visible **inside single runs**: a horizon phase at 5–15k (retfm .02–.77) that converts to a perfect equilibrium map and then climbs monotonically — the price first starves the working state, then the map reorganizes into a compressed contractive code.

### 3.2 Quantum-geometry lens: λ as metric contraction, and where the instrument bends
Where an equilibrium exists and degrades, λ_joint is the right readout (within-arm Spearman −.78 to −.95 on the four degrading arms). Two contract limits, both anticipated: (i) **eq_coupled's λ reads the mass mode** — trained a1+a2 = 1.35/1.23 (mass fixed point m* = a2/(1−a1) ≈ 1.9/1.6), so A4s1 shows λ ≈ 2.0–2.5 while retaining perfectly and posting the cold record — the 2026-08-14 gauge analysis realized as an instrument caveat; (ii) **pre-equilibrium λ is not a stability readout** (A5/A10's positive within-arm correlations are the formation phase). New diagnostic from the retention pair: **sign(retfm − ret_sched)** separates ramp-path turbulence (positive: A2 +.46, A9 +.90 — the early-t_norm maps at η≈1 are violent but the terminal map contracts; benign) from true final-map degradation (negative: A8 −.19, A2s1 −.32 — the H-45 direction). Basin radii: FPA measurably **widens basins** (probe S(.4) = .96 on both seeds vs .62–.64 for plain T12) — the ladder-as-loss mechanism does what it was designed to do, generalizing beyond its ε=.2 training radius.

### 3.3 Information lens: 250× compression buys stability, not (yet) cold
Plain/NI/FPA/coupled arms run I_total ≈ 2.2–4.2×10⁵ nats (streams) + 1.4–5.9×10⁶ (free attention); priced arms run **1.0–1.6×10³ nats with the attention channel priced to zero** — ~250× compression — and are uniformly stable (retfm 1.00, 6/6) with the tightest d32 seed pair of the wave (0.68 pp). Depth pays under price (+8.3 pp, A5 vs W9); budget does not at d32 (+1.7, A6 vs W8). The priced cold ceiling at d16–d32 remains below the plain-T12 band: at these scales **the bottleneck is the stability regulariser, not the accuracy lever** — whether that inverts at d64+ is precisely a ladder question (H-44's remaining axis is width).

### 3.4 Holography lens: funnel spectroscopy — who keeps the hard tail
Per-draw verified hit-rate vs rating (log-slopes): S5 **−0.058**/rating-unit with a fat tail (0.7 %/draw on the hardest octile — the funnel never closes; vote@1024 by octile = 98/100/100/95/95/97/95/**87.5 %**); W2 −0.074; W3 −0.113 (enormous easy-octile rates, .55/draw, then a cliff — the informational prediction "W3 ≥ S5 at k=256" is **refuted**: 30.3 vs 80.7; "W2 < S5" confirmed at 44.7); priced ≤ −0.25 (no measurable tail at k=256). **Depth, RI/NI, and price all concentrate the watershed onto easy instances; the wide funnel belongs to the shallow-horizon, moderate-budget plain map.** The cold gains at T12 live almost entirely in the easiest octiles (A4s1 80.6 % on octile 1 vs S5's 50.8; ≈0 beyond octile 3) — depth widens the canonical-start basin where propagation suffices, and buys nothing where search is required. Echo of H-37 at 20k scale: a single random-init draw beats the VOID start on all three PHASE4 arms (vote@1 = 21.0/23.2/21.4 vs cold 17.7/21.2/20.2).

## 4. Adversarial pass

- *Is the A4s1 21.2 % record real?* The eval is a standard full-test run (n = 422,786) on an admitted ckpt; but it is n=1 at a 13.3 pp pair spread (A4 7.86), the arm's λ readout is mass-confounded, and it has no probe (eq_coupled skip). Report it as the record with the variance label; it does not drive the recipe (narrow funnel, 35.8 @128).
- *Does "COLLAPSED" overstate A7?* Yes, by mechanism — the rule's letter fired on the cold leg. §2.1 is the honest reading; the analyzer output stands as registered.
- *Is the seed-noise rule too blunt?* It did its job: it prevented every within-noise cold contrast from being over-read. The mechanism-resolved spreads (FPA 1.7 vs RI+NI 15.3) carry the real information and are reported descriptively.
- *Could the S5 breadth confirmation be subsample luck?* n = 20,000 seeded (registered seed), CI ±0.64 pp at the vote@128 point; strat-512 agrees (70.7); wave-2's independent k=128 run agreed (70.7). Solid.
- *λ instrument validity:* power iteration on 16 puzzles, joint (y,z) modes; validated on the banked-ckpt check at registration (healthy .73–.85, collapsed 2.3). Its two failure modes (mass-mode, pre-equilibrium) are now documented contract limits, not silent confounds.
- *Statistics:* every cold contrast n=1–2 at a measured 15.3 pp ceiling of pair spreads; the claims that carry weight are instrument-level (retention/λ/funnels, n=512 rows) and the full-test-grade breadth CI. Phase B seeds the headline rung ×2 as registered.

## 5. What moves the number — Phase B (the parity ladder)

Goal: **vote@128 @t64, full-test-grade, at ≤7 M params** vs EqR's 99.8 % @5.03 M (D=64, B=128), with cold M-bands alongside. Breadth bands registered: **B-M1 ≥ .50 (BANKED at 0.078 M) / B-M2 ≥ .85 / B-M3 ≥ .95**.

1. **Rungs d64 → d96 → d128** (full-width scaling via the `--width-scale` flag, to build; ≈5 M at d128), T12, 50k, DP-8, one pod.
2. **Rung-1 (d64) arms:** A2-class (the registered recipe) · A3-class (FPA — seed-stable alternate; one of the two continues after rung 1 by val-selected cold + funnel width) · priced T12 (stability/compression twin; PRICED-DEPTH-PAYS earned the slot). Rung 2 carries two arms; d128 carries the winner ×2 seeds.
3. **Instruments per rung:** monitors every 2k (checkpoint variance is the dominant noise), banked ckpts every 5k, val-selected reporting; λ/RG spectrum + per-scale flux along the solve; bits/step vs rating; **hit-rate-vs-rating slope vs d** (does width flatten the funnel decay? — the holographic capacity question); Fisher contraction instrument built at d64.
4. **Breadth per rung on two checkpoints** (val-selected + one mid-training) — H-46's test: does the funnel narrow with training at scale?
5. **Pre-named contingency:** if rung-1 T12 funnels read below S5's 68.6 @128, a T6@20k-class arm (the S5 recipe, width-scaled) joins rung 2 — the wide-funnel insurance.
6. Owed before launch ($0): evaluator vote_at_k list past 256; the Phase-B launch registration with locked decision rules + pre-registered predictions per rung.

Cost ≈ $700–1,000 of the ≈$2,725 remaining; calendar: ICLR abstract Sep 18 / full Sep 25 — the ladder fits with margin if launched promptly.

## 6. Claim status

- **Breadth (labeled): M1-ON-BREADTH, full-test-grade — S5 vote@128 = 68.6 ± 0.6 %, vote@256 = 80.7 % on the seeded 20k subsample at 0.078 M params**; instrument-set tail 96.1 % @k=1024 (unsaturated; hardest octile 87.5 %).
- Cold: **21.2 % full-test (A4s1) — program record, BELOW-M1**; context HRM 55 / TRM 87 / EqR 99.8 at 5–27 M params.
- H-45: scoped (T6/horizon) + mechanism-confirmed (λ-precursor on A8; conventional regularisation does not prevent it; NI/price/FPA/depth do). **H-46 registered** (cold-basin wander at fixed contractivity). H-44: priced-depth-pays; price = stability regulariser at ≈250× compression, cold ceiling below plain-T12 at d≤32.
- Funnel law (H-33 lineage, sharpened): depth/RI/NI/price concentrate the watershed on easy instances; the shallow-horizon moderate-budget plain map keeps the hard tail; vote@k stays log-linear to k=1024.
- Recipe for the ladder (mechanical rule): **A2 — plain T12 + RI .5 + NI .01**, carried with FPA as the stability alternate and priced T12 as the twin.
