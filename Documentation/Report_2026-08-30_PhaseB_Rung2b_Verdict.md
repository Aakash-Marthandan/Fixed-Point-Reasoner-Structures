# Phase B Rung 2b (d96 follow-up) — Verdict: the Dosed Deep Lane Shatters the Cold Record, the Continuation Owns Breadth, and the Free Channel Is Confirmed as Pure Cost

**Date:** 2026-08-30 · **Analyzer:** `tools/analyze_sportBr2b.py` (byte-untouched since registration commit b49c970, `git diff` = 0; self-test 24/24 incl. the C3X addendum rules) · **Artifacts:** `runs/analysis/sportBr2b_verdict.txt`, `runs/analysis/sportBr2b_physics_20260830.txt`, lens updates `funnel_model_20260830.txt` / `graded_ladder_20260830.txt` / `puzzle_panel_20260830.txt` / `ignition_dynamics_20260830.txt` · **Data:** `sportBr2b_final.tgz` (852 MB; STRICT AUDIT 286/0/0 = `runs/audit_sportBr2b_final_20260830.txt`).

Goal metric: 20k vote@128 @t64 → B-M2 (≥.85) — **not measured this rung** (the registered PHASE4 gate was D1-keyed and read FAIL; see §4.2). Mechanism questions: H-48 (deep trainability under the β_nl dose) + H-45-with-anchor (long-horizon stability) + R2b-5 (continuation).

## 0. Integrity (post-results critique)
- Admission at ckpt level: all five arms verified (d96 ws6, widths/params exact; **D2's FPA flags read directly from the raw checkpoint** — fpa_k=4/eps=.2/w=1.0, T6, all-β=0 — the exact C3 recipe; seed via chain argv, C1s1-corroboration class). Resumes labeled (D3 @32000 [DP-4→8] + @49050; D4 @32000/@44050/wall-recycle; RNG-resplit valid-draw class, gradient-equivalence-tested DP); metrics splices audited, last-wins dedup applied.
- D1 = STOPPED@10000 labeled (one-shot rule; amputation audited; its screens beyond 10k are audited legit-empties).
- No chain-argmax this rung (no winner marker class); carriers are analyzer-derived.
- **The records-vs-summary investigation (owed from lens B) CLOSES with an exact mechanism:** summary `vote_at_k` = **cold ∪ first-k-draws**; records `mi_first_hit` = draws-only. D4-vb: 8 cold-only puzzles = exactly the 1.56 pp gap; C3X-vb: 0 extras = exact match; B2-d64's −1.17 pp = same class. No defect; the paper names the two statistics ("vote@k incl. cold" vs "draw-funnel@k"; the (ρ,r) instrument uses draws-only by construction).
- C3X comparability exact: same strat-512 seed (20260821), k=256, t=64 as C3's rung-2 screens.

## 1. Registered verdicts

| Rule | Outcome | Numbers |
|---|---|---|
| R2b-1 FUNNEL | **LONGER-TRAINING-HURTS** (stop-censored — see §2.4) | D1 vb 29.49 vs C3 88.48 (FN2b 3.71) |
| R2b-2 H-45-WITH-ANCHOR | **H-45-BREAKS-ANCHOR@10000** | D1 retfm 0.875 at its final monitor, ret_sched 0.61 — the anchor broke BEFORE the NaN |
| R2b-3 H-48 | **H-48-SUPPORTED** · D4 → **REGISTERED-LR-RESTORED** | D3 complete retfm 1.00; D4 complete retfm 1.00; late-train A_total 1.4 / 5.5 nats (predicted ≤1e5; free ref ~1.1e7) |
| R2b-4 DEEP BILLING | **DEEP-COLD-PAYS** | D3 vsel 30.36 ≥ bar 25.36 |
| R2b-5 C3X | **CONTINUATION-GROWS** + **FLOOR-LR-STABLE** | vb 94.53 vs C3 88.48 (> FN2b); retfm 1.00; cold 24.59 (labeled) |
| NOISE | FN2b **3.71 pp** · CNOISE **3.08 pp** | the matched-pair instruments, now measured |
| STABILITY | UNSTABLE: D1 (named) | all others retfm 1.00 |
| CARRIERS (mechanical, d128) | breadth = **C3X-class ×2 seeds**; deep = **D3-lane** (lr note §5); priced: none | |

Per-arm (cold | vsel | retfm | screen-vb v256 | curve): D1 13.16 | 13.16 | **0.88** | 29.49 | s010000=vb=29.49 (pre-ignition; stopped) · D2 25.36 | 25.36 | 1.00 | 84.77 | 73.2→81.6→84.8 · D3 29.54 | **30.36 (vsel@40k)** | 1.00 | 92.77 | 51.2→92.8 (vb=40k) · **D4 33.53** | 33.53 | 1.00 | 89.06 | 68.2→79.1→89.1 (rising at 50k) · C3X 24.59 | 24.59 | 1.00 | **94.53** | 81.5→91.0→94.5.

**Prediction scoreboard:** HIT — D3 clean + A≤1e5 by 10k (over-hit: 2.3 nats) + λ_joint ≤2.5 (over-hit: max 1.25) + η∈[.80,.90] (D3 .827, D4 .844, D2 .803); D4 clean (60% credence — and it took the record); C3X stable + cold [22.5,25] (24.59); D2 vb [84,92] (84.77). EXCEEDED — D3 cold vsel [24.5,28] → 30.36 ("lands ⇒ reclaims the d96 cold record" — over-delivered); D3 screen [70,85] → 92.77; C3X vb [88,93] → 94.53 (0.5 above band). MISSED — D1 everything (stopped@10k; retfm 1.0 → 0.88 break); D2 cold [21,23.5] → 25.36 (high side). The census: every dosed-arm prediction hit or over-delivered; the fresh-hot-T6 arm failed harder than predicted.

## 2. The findings

### 2.1 The deep lane delivers at d96 — and the cold record moves +8.3 pp
**D4 (T12, registered lr 1e-3, β_nl 1e-6): 33.53 % full-test cold** — the program record (19.4 d16 → 21.2 d16 → 25.27 d64 → 33.53 d96), paired-certain vs the d64 record (only-D4 48,176 vs only-B2 13,243, p≈0). D3 (5e-4 twin): 30.36 vsel. Both dose cells land 30+: the deep-lane claim rides n=2 (different lrs, different seeds — the lr-vs-seed split inside the pair is confounded and labeled, but the LANE is not). Rung-2's censoring reading is vindicated: C1@20k was pacing ahead of B2's mid-training; the dosed completions realize it and more. The t64 cold ladder's width points now read 25.27 (d64) → 33.53 (d96) with the recipe *unchanged except the dose*.

### 2.2 The dose is channel CLOSURE, and the free channel was pure cost
The β_nl 1e-6 toll shut the attention channel **during warmup**: A_total peaked ~0.6–0.9 M nats, fell to ~500–1,100 by 2k, **1–12 nats by 10k** (seven orders below the free arms' 10–20 M). At that closure: the cold record (33.53), the *widest* basins measured at d96 (graded S(.6) = .80/.79 vs C1/C3's .69/.79; S(.8) .15; leak 0), non-flat funnels (v256/v16 1.7–1.8), retfm 1.00, λ_joint max 1.13–1.25 (vs C1's 4.55 — the z-modes calmed exactly as predicted). **Measured capability cost of closing the free channel: zero-to-negative.** Combined with rung-2's survival table (free T12 0/4) and D1 (free T6 on the 50k cosine: retfm break 0.875 → A→1.8e14 → NaN, the C1 signature at HALF the depth), the free-channel law sharpens: unpriced attention flux is S2 deadweight that becomes *lethal* when composed through depth × width × hot-schedule lr. H-48 SUPPORTED with the strongest possible form: the channel wasn't carrying load-bearing information at all.

### 2.3 T12 funnels ignite too — rung-2's "narrow deep funnels" was death-censoring
D3's funnel: 51.2 @25k → **92.77 @40k**; D4: 68.2 → 79.1 → 89.1 *still rising at 50k*. With survival, the deep maps ignite exactly like the shallow ones — later on the schedule, and D3's peaks at 40k (=vb). The ignition signatures replicate on every 2b arm: mixer displacement dominance 72–84 % every window, Fisher-rotation consolidation rising monotone (D3 .53→.86), the η-surge→flatten precursor (D3 Δη +.138 → +.014 before its 25→40k ignition; D2 replicates the whole sequence at seed 1 with ignition <10k), rule_H ≈ 0 (committed codebook — no C4-style de-commitment under the dose). The funnel-ownership story revises again: at d96 **every stable arm's funnel is wide** (84.8–94.5 v256); ownership was never about T — it was about *who survives to ignite*.

### 2.4 The T6 long-budget route is CONTINUATION, not fresh-long
D1 (fresh 50k cosine) died in the hot phase with the anchor already broken (retfm 0.875, ret_sched 0.61 at 10k — H-45-BREAKS-ANCHOR: the first anchored-map break measured at width). C3X (cool continuation: C3's 20k ckpt + 30k at floor lr) sailed (FLOOR-LR-STABLE at ~20 M free nats) and GREW: vb 94.53 = the best funnel measured, +6.05 pp over C3 (> FN2b 3.71), paired cold gain +21,810/−12,036 (24.59, labeled continuation). R2b-1's letter (LONGER-TRAINING-HURTS) is recorded with its stop-censoring label: what hurts is the *hot fresh schedule*, not length — the lr-phase bracket the C3X addendum was registered to measure, measured. The T6 lane's scaling recipe = two-phase (cosine 20k → floor-lr continuation).

### 2.5 Seed structure at d96
FN2b = 3.71 pp (funnel) and CNOISE = 3.08 pp (cold) — the registered equivalence bands, now measured on a true matched pair. The paired panel adds texture: the seeds differ *systematically*, not as jitter (D2 > C3 on 27,804 vs 14,794 discordant puzzles, p≈0) — band-sized but real, re-affirming ×2 seeds on every d128 claim-bearing cell (already the carrier rule). D2's own cold (25.36) incidentally edges the old d64 record.

### 2.6 Coverage keeps expanding; the hard core halves-ish; conventions named
(ρ,r) on the vb screens (strat-512, labeled small-n): C3X's hard-octile reachable fractions ρ = .62–.82 (C3-d96 was .43–.59 on the 20k set) — the continuation buys coverage, exactly the "route to B-M2 is ρ via training" strategy conclusion. The model's biases bracket actuals as known (2p −9.5 pp / 1p +6.1 pp on C3X). The 22-arm panel: **cold hard core 49.59 %** (was 61.4 % over 17 arms — the dose cells alone un-cored ~50k puzzles); d96 union-cold 46.04 % vs best-single 33.53 (portfolio law grows: +12.5 pp); solve-multiplicity now spans the full 0–22 range. Convention ruling (§0): vote@k-incl.-cold vs draw-funnel@k are distinct named statistics.

## 3. Instrument notes
1. **The graded ladder discriminates at d96** and lands a clean dissociation: the **attention toll does NOT narrow basins** (D3/D4 = widest S(ε) curves at d96) — the toll-narrows-basins law (D8-extension, 08-29) belongs to the *stream* toll (β_flux) alone. The two tolls now have separated measured roles: streams-toll = abstraction/transfer shaping (with basin cost on CSP); attention-toll = stability/trainability (no basin cost, no capability cost).
2. D1's broken-map fingerprint: ret 0.885, fragmented S-curve, **leak 63** (the per-row map-class fingerprint working as designed).
3. λ_joint stays a non-alarm: C3X finishes at 1.07 (benign-watch, retfm 1.00); D3's 1.25 max under the dose vs C1's 4.55 free is a *dose readout*, not a stability readout.
4. vsel ≠ final appears at width for the first time on a healthy arm (D3 vb=40k on BOTH cold and funnel — mild late decay past 40k; H-46-adjacent, small, labeled).
5. Screen contrasts ride n=512 (CI ≈ ±2 pp): the C3X−C3 gain (6.05) and D3/D4 funnel levels (89–93) are 2–3× CI and carry paired-cold corroboration.

## 4. Adversarial pass
1. **Is D4's record the dose, the seed, or the resumes?** The registered one-variable cell is D3 (vs C1: dose only) — it alone un-censors the deep lane (30.36, p≈0 vs C1@20k). D4 adds the registered-lr restoration at a *named* seed confound. Both cells landing 30+ makes the LANE claim n=2-robust; the 33.53-vs-30.36 split (lr? seed?) is unattributed by design and stated so. Resumes are the registered valid-draw class (and D2 — zero resumes — replicates the recipe class cleanly at 25.36).
2. **B-M2 is NOT claimed.** The strat-512 v128 (C3X 89.45) sits above .85, and the rung-2 strat→20k mapping (88.48→80.57) *suggests* ~86–88 @128 on the 20k set — but that is an extrapolation, labeled. The registered gate did its job (D1-keyed, FAIL); the un-run C3X 20k scan is the one measurement standing between this rung and a B-M2 adjudication (~$25 one-shot; PI decision, recommended before/with d128).
3. **R2b-1's letter vs its meaning**: LONGER-TRAINING-HURTS is recorded with the stop-censoring label — the funnel comparison reads a 10k-stopped arm; the un-censored long-T6 question is answered by C3X instead (grows). Both recorded; neither over-claimed.
4. C3X's λ_joint 1.07 tail and D3's post-40k decay are named watch-items for the d128 monitors.
5. The d96 union (46.04) and hard-core (49.59) are portfolio facts, not deployable claims — though on Sudoku verification is free, so the model-portfolio × verified-attempts protocol remains a registrable lever (unregistered).
6. Contamination: analyzer 0-diff; no values read during ops (transcript-verifiable); D2's recipe pinned at raw-ckpt level; audit 286/0/0 before any read.

## 5. Consequences — d128 (PI decisions)
**Carried by rule:** breadth = **C3X-class ×2 seeds** (two-phase: 20k cosine → +30k floor-lr continuation) — at d128 this is the direct scaling of the recipe (its 20k base = C3-class fresh at d128); deep lane = **D3-lane** (DEEP-COLD-PAYS). **Recommendation on the deep lane's lr:** run it at the registered 1e-3 (D4's record + REGISTERED-LR-RESTORED; the 5e-4 label was always the contingency, now unneeded) — ×2 seeds if it is claim-bearing for cold.
**Recommended pre-d128 riders (~$25–65 total):** (i) C3X 20k k128 scan — the B-M2 adjudication this rung earned but couldn't run (one-shot, C4_vb-repair pattern); (ii) optional D4 20k scan (does the cold-record arm also carry a full-test-grade funnel? its strat v128 76.8 suggests below C3X — cheap to know).
**Not carried:** priced-stream lanes (dead), free deep/hot-T6 arms (0/5), lr surgery. **Stabilizer escalation ladder** (z-clamp → η_z damping → RMSNorm → FPRM kit): REGISTERED-UNNEEDED at d96 — the minimal dose sufficed; it rides as d128 contingency only. **Ops posture for d128:** US-first zones, weekend window, v6e-8/16 per availability at the corrected rates; ≈$300–470.

## 6. Claim status (paper-facing)
- **Cold:** **33.53 %** full-test @t64 at 2.11 M params (D4, dosed registered-lr; paired p≈0 vs every prior arm) — the d16→d64→d96 cold ladder now 21.2 → 25.3 → 33.5; deep-lane n=2 (30.4–33.5).
- **Breadth (labeled, instrument set):** C3X strat-512 v256 = 94.53 / v128 = 89.45; 20k-grade number pending the recommended scan; BREADTH-SCALES stands from rung 2 (80.57 @128 20k-grade).
- **Laws:** the free attention channel at scale = S2 deadweight + instability fuel — closing it (β_nl 1e-6) costs nothing measurable and unlocks the deep lane (H-48 SUPPORTED; H-45 extended: hot-schedule breaks even anchored maps — the anchor is necessary, not sufficient, at long horizon); ignition is universal on survivors (mixer-dominance / Fisher-clock / η-flatten signatures replicate ×5 arms); the two tolls dissociate (stream-toll shapes transfer/basins; attention-toll buys stability free); coverage (ρ) grows with continuation training — the registered route to B-M2.
- The measure → understand → improve loop, twice in one rung: the forensics-designed dose cell took the program record; the forensics-designed continuation took the funnel record.
