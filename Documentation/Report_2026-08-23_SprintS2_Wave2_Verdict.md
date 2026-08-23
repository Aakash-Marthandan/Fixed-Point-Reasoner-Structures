# Sprint S2 Wave 2 (Sudoku-Extreme) — Verdict, the Contractivity Collapse, and the Price as a Contractive Regularizer

**Date:** 2026-08-23 · **Analyzer:** `tools/analyze_sport2w2.py` (byte-identical to the 2026-08-22 launch commit `f164feb`; self-test 21/21) · **Artifact:** `runs/analysis/sport2w2_verdict.txt` · **Data:** `gs://qhrrn2-rescue/sport2w2/sport2w2_final.tgz` (13 ckpts, 12 eval arms × 7 summaries, 11 probes, 4 breadth scans, 1 breadth20k).

Goal metric (guardrail 1): Sudoku-Extreme **full-test exact, cold** (bands M1 ≥ .50 / M2 ≥ .85 / M3 ≥ .95), with the verify-and-vote **breadth number labeled separately**. Every number below is on the full 422,786-puzzle test set unless marked strat-512 (the rating-stratified 512-puzzle instrument set, `strat-seed 20260821`) or 20k (the seeded 20,000-puzzle subsample, `subsample-seed 20260822`).

## 0. Integrity (post-results critique)

- Every full-test summary spans n = 422,786; every strat summary n = 512; val n = 64; retention n = 512. Every checkpoint sits at its registered step (W13 100k; W1/W9/W1s1 50k; W2/W3 30k; the rest 20k). W4's and W9's full-test evals were computed **sharded over idle chips and merged** (same evaluator, same flags, `--shard/--merge` — the PHASE4 mechanism); each merge asserted n == sum of shards, and sharded+merged was pre-verified bit-identical to a single run on every summary key and every per-puzzle record. Those summaries carry `shard: "merged"`. No registered number was altered.
- W4 (plain d32) suffered one **transient libtpu `SLICE_FAILURE_WORKER_UNAVAILABLE` SIGABRT** at step 19,550 (training steady, no Python exception, W8 at the same width trained clean) and finished via the standard checkpoint-exact resume (model + optimizer + rng at step 19,000). Four spot preemptions across the campaign, all resumed near-losslessly; the v6e-16 was demoted after two strikes and the full 16-job suite ran on the v6e-8.
- **Rules applied as registered, no post-hoc edits.** Two registered rules turn out to have been written for a monotone world the data refute (§2): P1's "BUDGET-NOT-BINDING" and P3's "WIDTH-NULL" are literally correct and substantively understated — budget and width are *anti*-binding on the plain base at these scales. The seed-noise rule fired at NOISE = 5.58 pp because W1 vs W1s1 differ by the collapse dynamics (0.82 vs 6.40 %), not by stationary seed jitter; the "WITHIN-SEED-NOISE" labels on P2/P3/P5/P6 are reported as registered and discussed.
- Multi-init draws are seeded per (puzzle, draw) from this wave (nested k-curves); wave-1 k=16 numbers stand as reported and are statistically, not bit-wise, comparable.
- One diagnostic instrument was added **after** the verdict and is labeled as such (§2, the η-override test); it contributes no registered number.

## 1. Registered verdicts (all cold full-test exact @t64 unless noted)

| Rule | Outcome | Numbers |
|---|---|---|
| M0-W2 | **GEN-PRIMARY** | W1 (plain T6 @50k) val@t64 = 0.00 — but for the opposite reason to the one registered (§2) |
| HEADLINE | **BELOW-M1** | best cold = **W2 plain T12 d16 @30k: 19.37 %** (t=64; 20.1 % strat @t256) |
| BREADTH (labeled) | **BELOW-M1-ON-BREADTH** (registered rule: best-cold arm's 20k vote@128) | W2 20k: cold 19.5 % → vote@128 **37.5 ± 0.7 %** |
| P7 BREADTH SCAN | **HOLDS** (S5 strat vote@128 ≥ .50) | S5 strat-512: vote@16/64/128/256 = 35.0 / 55.9 / **70.7 / 80.7 %**; not saturated (+10 pp per doubling at the top) |
| P1 BUDGET | BUDGET-NOT-BINDING · MONOTONE | W1 − S5 = −11.2 pp (0.82 vs 12.02); W13 (@100k) 0.04 % |
| P2 DEPTH | **DEPTH-PAYS** (retention 1.00) [within-seed-noise label] · **RI+NI-RAISES-BREADTH** | W2 − S5 = +7.35 pp; vote16 W3 − W2 = +3.9 pp (cold W3 − W2 = −4.4) |
| P3 WIDTH | WIDTH-NULL | W4 − S5 = −10.2 pp (1.79 %) |
| P4 PRICE × SCALE (H-44) | **SUPPORTED** — W8 (priced d32 @20k) and W9 (priced d16 @50k) both **RECOVERED** | W8 9.35 % ret 1.00; W9 5.24 % ret 1.00; S0 (priced d16 @20k) 0.08 % ret .01 HORIZON |
| DOSE (H-43 test) | **DOSE-KILLS** | W6 (β 3e-6): 0.02 %, retention .14, I_total compressed 65× |
| P5 GEN | GEN-NO-TRANSFER | W5 0.01 % (confounded — §2) |
| P6 BOX4 | BOX4-HURTS | W7 0.34 %, retention .61 (confounded — §2) |
| P7b | NO-RI+NI-BREADTH-GAIN on the priced base | S7 per-draw hit-rate .004 vs S5 .079 |
| SEED | NOISE = 5.58 pp | W1 0.82 vs W1s1 6.40; W4 1.79 vs W4s1 0.55 |

Per-arm table (cold @t64 / strat vote@16 / retention / η / I_total): S5 12.02 / 36.9 / .98 / .555 / 3.0e5 · **W2 19.37 / 23.2 / 1.00 / .607 / 3.1e5** · W3 14.95 / 27.2 / .87 / .972 / 2.4e5 · W8 9.35 / 16.6 / 1.00 / .919 / 1.1e3 · W1s1 6.40 / 22.1 / .92 / .682 / 3.3e5 · W9 5.24 / 7.2 / 1.00 / .924 / 1.3e3 · W4 1.79 / 14.6 / .93 / .700 / 2.2e5 · W1 0.82 / 14.3 / 1.00 / .900 / 2.7e5 · W4s1 0.55 / 23.0 / .99 / .723 / 2.8e5 · W7 0.34 / 2.2 / .61 / .677 / 3.5e5 · W13 0.04 / 0.4 / .98 / .984 / 3.2e5 · W6 0.02 / 0.2 / .14 / .906 / 4.7e3 · W5 0.01 / 0.6 / .86 / .808 / 3.2e5.

## 2. The headline finding: the plain map loses contractivity as training continues (η drifts as the visible proxy) — and the price keeps the map contractive

**The wave-2 design assumed monotone scaling; the data are non-monotone.** Every plain-base arm that added budget or capacity beyond S5's (T6, d16, 20k) *lost* cold accuracy — 50k steps 0.82 % (seed 1: 6.40 %), 100k steps 0.04 %, d32 1.79 % (seed 1: 0.55 %), box4 0.34 %, GEN-init 0.01 % — while train loss barely moved (ce 0.45–0.50 for all; this is **not** memorisation) and full-schedule retention stayed ≈1. Three instruments locate what broke:

1. **The learned equilibrium step η drifts up with training and capacity**: S5 .555 → W1s1 .68 → W4 .70 / W4s1 .72 / W7 .68 → W5 .81 → W1 .90 → W13 .98. Among plain arms without noise injection, η ≤ .61 ↔ 12–19 % cold; η .68–.72 ↔ 0.3–6.4 %; η ≥ .8 ↔ ≤ 0.8 %. η is the *damping* of `y ← y + η(p − y)`; the loss only ever sees the first T steps, so gradient pressure rewards a larger step (reach the answer inside the horizon) and nothing in the objective protects the long-horizon dynamics that the t=64 accuracy lives on.
2. **The solution becomes a slowly-repelling fixed point of the final map.** Full-schedule retention (solution-init, 8 steps including the t_norm ramp) reads 1.00 for W1 / .93 W4 / .99 W4s1, but the final map applied alone (probe, 8 steps at t_norm = 1) retains only **.11 / .05 / .05**; healthy maps agree across both instruments (S5 .98/.98, W2 1.00/1.00, W8 1.00/1.00, W9 1.00/1.00). And the collapsed arms' cold accuracy *falls* from t64 to t256 (W1s1 5.5 → 1.2, W4 2.3 → 1.8, W1 0.6 → 0.0) where healthy maps plateau or rise (W2 19.3 → 20.1, W3 14.5 → 18.0) — the iteration approaches the solution and leaves it. Stability of `y_{t+1} = y_t + η(f(y_t) − y_t)` at the fixed point needs |1 − η(1 − λ_f)| < 1 for every Jacobian eigenvalue λ_f; raising η destabilises oscillatory modes first. W13's map even overwrites its own clues (givens kept .85).
3. **η-override causality test** (post-hoc diagnostic, labeled; strat-512 @t64, k=0; `runs/diag_eta/`): forcing the collapsed maps back to S5's damping does **not** restore them (W13 η .984→.55: 0.00 → 0.98 %; W1 η .900→.55: 0.59 → 1.76 %), and forcing the healthy maps up to η = .90 does **not** collapse them (S5 14.1 → 11.3 %; W2 19.3 → 15.4 %). **η is a symptom, not the cause.** The learned map f itself lost contractivity at the solution — damping cannot stabilise a real expansive eigenvalue (|1 − η(1 − λ_f)| > 1 for all η > 0 when λ_f > 1), which is exactly what the final-map probe sees (retention .05–.11) — and η drifts alongside because the same short-horizon gradient pressure drives both. Knowledge is not "intact behind a bad step size"; the fixed-point structure degraded.

**H-44 is supported, and with a mechanism.** At d16/20k the knee price kills the equilibrium (S0 0.08 %, retention .01 — H-43); at d32/20k (W8) and at d16/50k (W9) the priced map is a *stable* equilibrium map (retention 1.00 on both instruments) with 9.35 % and 5.24 % cold — while the plain map at the *same* scale has collapsed (W4 1.79 %, W1 0.82 %). The priced maps sit at η ≈ .92 — the same high damping-less regime that destroys the plain maps — and are stable there; so is the NI-trained W3 (η .97, 14.95 %). **The information bottleneck produces a contractive map; the plain map's stability rests on damping that short-horizon training erodes.** The sign of the price effect therefore flips with scale exactly as the PI anticipated: priced < plain at d16/20k (0.08 vs 12.02), priced ≫ plain at d32/20k (9.35 vs 1.79) and at d16/50k (5.24 vs 0.82). The priced maps are slower to reach equilibrium (S0 → W9: 0.08 → 5.24 with budget) and their ceiling is unmeasured — the wave-3 question. The dose arm (β 3e-6, W6) is still a horizon map at d16/20k (retention .14, 65× compression), so the transition to stability is a function of scale, not of a gentler price.

**The cold number and the breadth number dissociate across substrates.** W2 (T12, η .61) is the best cold map (19.4 %) but a narrow funnel: per-draw verified hit-rate .022, vote@128 on 20k = 37.5 %. S5 (T6, η .555) is a wide funnel: hit-rate .079/draw, vote@128 = 70.7 % and vote@256 = 80.7 % on strat-512, by difficulty octile 99/89/70/79/86/72/**55 %** at k=256 (hardest octile, rating ≥ 48, cold 0.5 %). The per-draw hit-rate falls 40× from the easiest to the hardest octile (.27 → .007) but the funnel never closes; the curve is still rising at k=256 (+10 pp per doubling). The registered breadth band was evaluated on the best-*cold* arm (W2) and reads BELOW-M1; P7, registered on S5, HOLDS at k=128. Both stand; the full-test confirmation for S5 (20k subsample, k=128/256) is the first job of wave 3 — that is the M1-on-breadth claim.

**What the confounded arms do and do not say.** W5 (GEN), W7 (box4) and the plain scale arms all sit in the collapse regime at their final checkpoint, so their registered labels (GEN-NO-TRANSFER, BOX4-HURTS, WIDTH-NULL) cannot be read as properties of the lever; they are properties of *where the run was on its η trajectory at the stop*. W7 additionally has weaker retention (.61 on both instruments) — the box-aligned layout changed the dynamics, not only the timing. These levers are re-tested only under the wave-3 protocol that tracks η and val@t64 along the trajectory.

### 2a. η-override causality test — numbers
| ckpt | learned η | forced η | strat cold @t64 (own η → forced) | read |
|---|---|---|---|---|
| W13 plain T6 @100k | .984 | .55 | 0.00 → 0.98 % | not restored |
| W1 plain T6 @50k | .900 | .55 | 0.59 → 1.76 % | not restored |
| S5 plain T6 @20k | .555 | .90 | 14.06 → 11.33 % | not collapsed (−2.7 pp) |
| W2 plain T12 @30k | .607 | .90 | 19.34 → 15.43 % | not collapsed (−3.9 pp) |

Damping matters at the few-pp level; contractivity of the learned map decides collapse. The NI map (W3) and the priced maps (W8, W9) are contractive at η ≈ .92–.97; the plain maps stop being contractive somewhere between 20k and 50k steps at d16 and before 20k at d32.

## 3. Physics

### 3.1 The contractivity law (new, H-45): the t=64 dividend lives on a contractive final map, and short-horizon training erodes it
Cold accuracy at the trained horizon is flat across all arms (strat t6 ≈ 1.4 %; full t6 ≈ 2.2–2.4 %): nothing is solved inside T. The entire benchmark number is the refinement dividend of iterating the equilibrium map beyond T, and that dividend exists only while the final map is contractive at the solution. Plain maps lose that contractivity with continued training (d16: between 20k and 50k steps) and with capacity (d32 by 20k); the observable proxy is the learned η drifting toward 1 (the same gradient pressure — progress inside the horizon — drives both), but the η-override test shows the damping is not the lever: the map must be kept contractive. Three fixes, each a registered wave-3 arm: supervise at random horizons up to t ≫ T (the objective then sees the repelling fixed point), FPRM's contractive coupled residual (`eq_coupled`, the Theorem-1 recipe, already in Config and never used), and the regularisers that empirically keep maps contractive here — the price (W8/W9) and noise injection (W3). An η cap is retained only as a secondary control.

### 3.2 Breadth works through trajectories: the k-curve by difficulty
The landscape-class law's "required search governs" now has its quantitative form: hit-rate per draw decays geometrically with rating while vote@k climbs log-linearly in k on every octile. Retention ≈ 1 everywhere (every trained map fixes every solution), cold solve concentrates in the easiest octile (S5 44.5 %, W2 74.6 % of octile 1; ≤ 1 % beyond octile 3), and search recovers the rest. Depth (T12) buys cold accuracy by widening the canonical-start basin (W2 74.6 % on octile 1) at the cost of the random-start basins (hit-rate .022 vs .079); it does not buy hard-octile cold solves (0.1 % beyond octile 3 for W2 too).

### 3.3 The price at scale: contractivity and the 250× compression
Priced maps carry I_total ≈ 1.1–1.3 k nats against 2.2–3.5 × 10⁵ for plain maps and hold retention 1.00 at η ≈ .92 with cold 5–9 %. Their per-draw hit-rate on strat-512 (.078 for W8) equals S5's, and their vote@16 (16.6 %) is a third of W2's cold number — the priced map at d32/20k is a *searchable* substrate. Whether the priced ceiling rises with budget (W9 at 100k; W8 at 50k; priced T12) decides whether the price is the regulariser of choice or one of several; the plain-with-η-cap arm is the direct competitor.

### 3.4 What did not move
GEN pretraining (givens-matched, difficulty-unmatched corpus) and the box-aligned layout are unresolved (collapse-confounded); RI+NI on the priced base does not raise breadth (S7 hit-rate .004) while on the plain base it does modestly (W3 vote16 +3.9 pp over W2, at −4.4 pp cold). T24 priced (S4) remains a narrow funnel (vote@256 24 %).

## 4. Adversarial pass
- *Could the collapse be an evaluation artefact (e.g. the evaluator's η readout)?* No: the probe (independent code path, the ARC instruments) shows the same final-map instability; W13 destroys its own givens; the t64→t256 decay is monotone in η.
- *Could W2's 19.4 % be a lucky pre-collapse snapshot rather than a depth effect?* Possible in part — W2 is at 30k with η .607 and may itself drift; the seed-noise label is honest. But both T12 arms beat every T6 arm at every horizon, and W3's NI map holds 15 % at η .97, which no T6 plain map does. Depth pays; its *durability* is a wave-3 measurement (η and val@t64 along the trajectory).
- *Is "priced > plain at scale" just "plain collapsed"?* Yes — and that is the claim: the price's dividend at scale is stability under the drift. Whether priced beats *regularised* plain (η-cap, horizon supervision, aug ×1000, EMA) is open and registered as wave 3's central contrast.
- *Is the breadth number a free lunch?* It costs k × 64 forward passes per puzzle and relies on Sudoku's free verification (uniqueness); it is reported separately and never as the cold number. The 20k-subsample confirmation for S5 is pending.
- *Seeds:* every wave-2 contrast is n = 1 (n = 2 at two cells); the collapse makes single checkpoints unreliable estimators of a configuration. Wave 3 measures along trajectories and seeds the headline cells.

## 5. What moves the number — wave 3 (goal: M1 cold and M1 breadth; every arm names its expected effect)
1. **Instrument the trajectory** (prerequisite, cheap): log η and a final-map contractivity readout (solution-init, 8 final-map steps on the 64 monitor puzzles) every 1000 steps; val@t64 via the batched evaluator every 2000 steps; bank intermediate checkpoints; select reported checkpoints by val@t64 (legitimate early stopping — val is train-file, test-disjoint).
2. **Horizon-supervised plain T12 @50k** (loss at a random t ∈ [T, 32] per step) — the objective-side fix (EqR/FPRM-adjacent); expected: final-map retention stays ≈ 1 through 50k, cold ≥ W2 and rising with budget.
3. **FPRM-contractive plain T12 @50k** (`eq_coupled`, a1/a2 init .75/.25) — the architectural fix; expected as arm 2; the two together test whether contractivity is sufficient.
4. **Priced T12 @50k, priced d32 @50k, priced d16 @100k** — the H-44 ceiling; expected: monotone rise with budget; decisive vs arms 2–3 for "regulariser of choice".
5. **Breadth confirmation**: S5 20k-subsample k=128 and k=256 (the M1-on-breadth claim); S5 k=1024 on strat-512 (saturation); breadth on the best wave-3 map.
6. **Seeds ×2** on arms 2–4; **aug ×1000 + wd ×10** on plain T6 as the conventional-regulariser control; **η-cap (≤ .60) plain T12** as the secondary control.
7. Deferred: GEN/box4 re-tests under the new protocol; T24 plain; d48/d64 priced cells.

## 6. Claim status
- Sudoku-Extreme cold: **19.4 % (plain T12 d16, 30k, ~0.2 M params) — BELOW-M1.** Context: HRM 55 %, TRM 87 %, EqR 99.8 % at 5–27 M params; our substrate is small and the number is a floor under an identified, fixable pathology.
- Breadth (labeled): **S5 vote@128 = 70.7 %, vote@256 = 80.7 % on strat-512** (M1-on-breadth on the instrument set; full-test 20k confirmation pending); W2 20k vote@128 = 37.5 %.
- H-44 (price × scale): **SUPPORTED** at d32/20k and d16/50k with the contractivity mechanism; H-43 scoped to small scale; dose (3e-6) does not rescue at d16/20k.
- New law candidate (**H-45, contractivity collapse**): plain equilibrium maps lose contractivity at the solution under continued short-horizon training (final-map retention → .05–.11 while full-schedule retention stays ≈ 1; t64→t256 decay; cold collapses); η drifts toward 1 as the observable proxy but is not the lever (override test); priced and NI maps stay contractive — three fixes registered for wave 3.
- Landscape-class law (H-33): quantitative form — per-draw hit-rate decays geometrically with rating; vote@k climbs log-linearly on every octile; hardest octile 55 % at k=256.
