# Champion Track Pilot (sportC0, native9 d96) — Verdict: Every Registered Rule Read, and the Mechanism Behind Each Letter

**Date:** 2026-09-02 (analysis pass; ops closed 2026-09-01 by the Opus ride) · **Analyzer:** `tools/analyze_sportC0.py` byte-untouched since the registration commit 6679ac2 (`git diff` = 0; selftest 16/16) → `runs/analysis/sportC0_verdict.txt` · **Physics pass (analysis-time, descriptive):** `tools/analyze_sportC0_physics.py` → `runs/analysis/sportC0_physics_20260902.txt`; explosion census → `runs/analysis/sportC0_explosion_census_20260902.json`; memorization check (train-split cold, CPU) and val-selected 20k evals → §7 addendum · **Data:** `gs://qhrrn2-rescue/sportC0` pulled serially with `gcloud storage cp` and crc32c-verified per object (a first `gsutil cp` stalled at full size with a wrong checksum — the -m stall class now has a serial sibling; `gcloud storage cp` is the tool of record), extracted into `runs/`. Spend ≈ $105 hand-derived from watchdog spans (US 16 0.13h + US 8 1.4h + Mumbai 8 11.6h).

Registration: ledger §5 2026-09-01 "CHAMPION TRACK PILOT — LAUNCH REGISTRATION"; design `Plan_2026-09-01_Champion_Track.md`. Nothing here edits a rule; every letter below is the analyzer's, and every reading is labeled.

## 0. Integrity (post-results critique, before any value was read)

- **Admission (artifact level, 7/7):** every `ckpt_latest.pkl` reads d96 / ws6 (d_task 192, d_code 192, d_b=d_a 36) / 1,713,822 bulk params / scales 2 / canvas 9 / pool_arity 3 / mixer group9 / layout native9 / attn_max_hw 9 / β_nl 1e-6 / β_flux 0 / FPA k4 ε.2 / T12 (P1 P2 P2s1 P5) or T16 (P3 P3s1 P6) / ni_sigma .01 on P5 only. `config.json` corroborates ri_p .5 / aug 100 / seed on P1 P2 P2s1 P5 P3s1(stage B); P3 and P6 died in stage A and the chain did not copy the stage-A `config.json` (their RI/aug flags are chain-source-only — labeled; the ckpt configs pin T16 and ni 0). P3s1's stage-B `config.json` reads steps 15000, lr 3e-5→3e-5, warmup 100, init_from = the banked stage-A ckpt (35k), as registered.
- **Counts:** 7 full-test summaries merged n=422,786; 6 scans n=20,000 (P6's scan never ran — the PI closed the node early; **R-C0-4 therefore reads NO-DATA by construction**); 7 retfm n=512; 20 screens n=512. Canvas riders (C3X/D4 EqR-statistic evals) were skipped at the PI's early close — **the program-review §1 riders remain owed.**
- **STOPPED semantics:** P2s1 amputated to 20k, P3 to 10k, P5 to 40k, P6 to 10k (one-shot rule, labels verbatim). Two integrity facts the reader must carry: (i) the amputation **truncated each stopped arm's `metrics.jsonl` to ≤ its last finite grid**, so the exact NaN step is not on disk — deaths are bounded by the banked-grid census to (20k,25k] P2s1, (10k,15k] P3, (40k,45k] P5, (10k,15k] P6; the trainer kept running on non-finite parameters to 50k/35k (25–30k wasted steps per death; the early-NaN-abort hardening is confirmed owed); (ii) the fixed-step screens of stopped arms at steps beyond the death ran on **non-finite grids and are garbage** (243 violations = every group empty, 0 cells, 0 givens kept) — P3 s015000/s035000, P6 s015000/s035000, P2s1 s035000 are EXCLUDED everywhere below (the analyzer's rules never read them).
- **Checkpoint provenance of the evals:** fulls ran on `ckpt_latest` (final or amputated) — the registered convention; scans ran on the val-selected ckpt (P2 30k, P2s1 20k, P3 10k, P3s1 stage-B 10k, P5 20k) **except P1, whose scan ran on the 50k final** (its evals resumed on the Mumbai node after the US preemption, where the banked grids and `metrics.jsonl` had not been re-pulled, so `select_ckpt` had nothing to read and `VBCK` fell back to `ckpt_latest`; P1's vb screen at 20k had already run on the US node). Labeled wherever it matters.
- **Pairing:** the six native scans and the canvas D4/C3X scans share the identical 20,000 puzzles (subsample seed 20260822; idx sets verified equal); all fulls cover the same 422,786 puzzles — every contrast below is exact McNemar on identical sets.
- **Noise instruments:** FNC0 = 11.45pp and CNC0 = 9.65pp are **contaminated** — the analyzer computed them from |P3 − P3s1|, but P3 is a 10k stage-A amputation and P3s1 a complete 50k two-phase run: this is not a matched seed pair. No sub-noise claim below leans on them; the champion registration must re-measure its noise on a clean pair.
- **Resume confound named:** P5 resumed at ~37k after the 18:55Z wall-ceiling recycle and died in (40k,45k] — the rng-resplit-adjacent death class (C1's ~1.5k-post-resume death). P5's NaN cannot be attributed cleanly to NI, RI, or width. The other three deaths (P2s1, P3, P6) were single-node, no-resume runs.

## 1. Registered verdicts (the analyzer's letters)

| Rule | Letter | Numbers (analyzer) | Reading (physics pass, labeled) |
|---|---|---|---|
| STABILITY | **UNCLEAN: P2s1, P3, P5, P6** | 4/7 arms stopped; every death an RI arm; the only no-RI arm clean | §3 |
| R-C0-1 ARITY | **ARITY-HURTS** | P1 final cold 19.38 vs D4 33.53 (bar 29.82); P1 vb 73.24 vs D4 89.06 (bar 85.35) | The letter fires on a **memorization-collapsed endpoint** (§2); at the val-selected checkpoints native ≥ canvas at matched recipe (§2.3). Geometry exonerated at mechanism level; the *training regime* is convicted |
| R-C0-2 RI | **RI-FRAGILE-AT-WIDTH** | P2s1 STOPPED; b1(P2) 34.48 vs b1(P1)+10 = 10.39 (the payoff half would have read RI-PAYS by +24pp) | RI is THE per-draw lever and the death class is now mechanistically characterized: η→1 replacement dynamics with inference-time state explosions (§3) |
| R-C0-3 NI | **NI-NEUTRAL** | P5 unclean, but the RI pair was unclean too (CONVICTED requires a clean pair); b1(P5) 34.89 vs b1(P2) 34.48 | NI adds nothing measurable on per-draw; P5's death is resume-adjacent (confounded). NI stays out |
| R-C0-4 AUG | **NO-DATA** | P6's scan never ran | The mechanism read is unambiguous and paired: aug1000 +5.70pp at 10k (§2.4) — the champion goes aug ≥ 1000 as a **registered deviation from the NO-DATA letter**, stated as such |
| R-C0-5 CHAMPION | **GO-FALLBACK: P3s1** | P3 pair unclean (P3 died in stage A); best clean b1 = P3s1 38.47 | The fallback arm IS the champion recipe (T16 + RI .5 + FPA + dose + two-phase 35k/15k), seed 1 — carried by the letter, with its 1/2 seed survival as the datum |

Per-arm (analyzer Section 1; final grids unless noted):

| arm | recipe | cold (full, final) | b1 (EqR B=1) | verified@128 | t1r@128 | majority@128 (vb screen) | retfm | screen-vb v256 | status |
|---|---|---|---|---|---|---|---|---|---|
| P1 | T12 FPA dose, no RI (D4-strategy verbatim) | 19.38 | 0.39† | 36.55† | 0.00† | 23.83 | 1.00 | 73.24 (@20k) | complete; **memorized** |
| P2 | + RI .5 s0 | 15.62 | 34.48 (@30k) | 49.19 | 48.92 | 36.13 | 1.00 | 52.54 (@30k) | complete; **memorized after 30k** |
| P2s1 | + RI .5 s1 | 36.22 (@20k) | 35.59 | 43.85 | 43.12 | 35.55 | 1.00 | 44.14 (@20k) | STOPPED 20k |
| P3 | champion s0 | 27.70 (@10k) | 27.02 | 29.98 | 29.68 | 28.12 | 1.00 | 30.08 (@10k) | STOPPED 10k (stage A) |
| **P3s1** | **champion s1** | **37.35** | **38.47** | **52.85** | **52.49** | **43.95** | 1.00 | 55.86 | **complete** |
| P5 | P2 + NI .01 | 28.09 (@40k) | 34.89 (@20k) | 46.44 | 46.07 | 36.33 | 1.00 | 48.63 (@20k) | STOPPED 40k (resume-adjacent) |
| P6 | champion + aug1000 s0 | 33.40 (@10k) | — | — | — | 34.18 | 1.00 | 39.06 (@10k) | STOPPED 10k (stage A); scan absent |
| D4 canvas | T12 FPA dose (2.11M) | 33.53 | — | 81.28 | — | — | 1.00 | 89.06 | ref |
| C3X canvas | T6 FPA two-phase (2.11M) | 24.59 | — | 88.89 | — | — | 1.00 | 94.53 | ref |

† P1's scan ran on the 50k final grid (provenance note in §0); its 20k val-selected grid's per-draw statistics were never measured.

**Prediction scoreboard (registration bands):** P1 cold [29,37] MISSED (19.38 final; the strat-512 val-selected read 38.1 sits inside the band — the miss is the collapse, §2); P1 b1 [28,40] MISSED badly (0.39 — the no-RI native map has essentially no random-init capability); P1 vb [85,92] MISSED (73.2 at 20k; 92.0 at 35k — the funnel ignited late and then the map memorized); P1 verified@128 [78,86] MISSED (36.55); P1 retfm 1.0 HIT. P2 b1 [42,65] MISSED LOW (34.5) but the direction HIT (+34pp over no-RI, the registration's "biggest unknown" resolved as a large, real, per-draw effect); P2 cold ≥ P1−5 HIT (34.66 vs 19.53 on the scan set); P2 pair-clean (60% credence) MISSED (P2s1 died). P3 b1 ≥ P2 HIT at seed 1 (38.47 vs 34.48, paired p=9e-34); P3 verified@128 ≥ 90 MISSED by far (52.85 — RI funnels are narrow, §3.2); λ_J ≤ 1.5 MISSED (P3s1 lives at 1.2–2.1 in stage B and is clean — λ_J again not a stability readout). P5 unclean (55%) HIT (confounded). P6 ≥ P3 on b1 (50%) — unmeasurable (scan absent); on full cold HIT (+5.70pp paired). Arity signature (s1 flux share ≥ 2× canvas) NOT OBSERVED and the object was ill-posed for a two-cut pyramid (§4). FNC0 ≤ 6pp — the instrument was contaminated (§0). Wall ≤ 2.5h/arm HIT (≈1.75–2h per arm on a v6e-8 incl. evals; pretrain 43 min for 50k T12).

## 2. Finding 1 — the native regime MEMORIZES; the D4-strategy's collapse produced the ARITY letter

### 2.1 The signature, on every native 50k-cosine arm
Train CE (`ce_in`, the last-step masked CE on training rows) — canvas D4 stays 0.56→0.28 across 5k→50k; canvas B2-d64 0.37–0.48; C3X 0.41–0.48. **Native P1: 0.59 (2k) → 0.49 (10k) → 0.25 (20k) → 0.086 (30k) → 0.000 (40k) → 0.000 (50k). Native P2 identical shape (0.000 at 50k); P5 0.011 at 40k.** The held-out monitor (val@t64, 64 train-file puzzles) peaks and then halves: P1 .42 @22k → .22 @38k–50k; P2 .39 @28k → .19–.20 @46–50k; P5 .42 @16k → .30–.33. The strat-512 screens read the same arc on test puzzles: P1 cold 36.9 (15k) → **38.1 (20k, val-selected)** → 27.9 (35k) → 19.38 full-test at 50k; P2 35.0/35.4/33.6 through 35k → 15.62 at 50k.

### 2.2 Puzzle-level proof (train-split cold, CPU eval, §7 addendum table)
P1@50k solves **99.9 % of its 1,000 training puzzles cold** (test 19.38 %); P1@20k solves 73.9 % (a generalizing regime). P2@30k 95.6 %, P2@50k 72.8 % with mean violations 53.7 (its final map also explodes — §3.3). P3s1@50k 95.0 % (test 37.35) — the floor phase held it short of total memorization. Canvas D4@50k: see addendum.

### 2.3 Mechanism (measured, not asserted)
The canvas regime placed the 9×9 grid at a random offset on the 32×32 canvas at every sample (576 positions): an implicit ×576 augmentation that the native geometry deliberately removed ("translation breaks box alignment"). With aug 100 the native corpus is 101,000 distinct pairs; 50k steps × B64 = 3.2M samples = **32 epochs** — the 1.7M-parameter map memorizes by 40k. The "arity" comparison P1-vs-D4 therefore compared a memorized endpoint to a generalizing one. **At the val-selected checkpoints the native arm is at or above the canvas at matched recipe:** P1@20k strat-cold 38.1 vs D4 34.6 (n=512, CI ±2pp); P2@30k scan-cold 34.66 vs D4 33.28 on the identical 20k (paired p=3.9e-5, P2 ahead); P3s1 full-test 37.35 vs 33.53 (p≈0). The val-selected 20k evals (§7) put P1@20k on the identical set.

### 2.4 The two protective factors, both measured
- **aug1000: +5.70pp at equal steps, paired.** P6 vs P3 are the same recipe (T16 RI seed 0), both amputated at 10k: full-test cold 33.40 vs 27.70, only-P6 37,416 vs only-P3 13,310, p≈0, every rating octile (82.2/77.7 easiest … 12.4/8.7 hardest). Data diversity pays immediately (10k = 0.6 epochs at aug1000), not only through anti-memorization.
- **Two-phase (35k cosine → 15k floor 3e-5): the surviving champion arm never memorized.** P3s1's stage-B CE plateaus at 0.04–0.055 (not 0), its monitor val holds .34–.41, its screens are flat (43.8 stage-A end / 42.6 / 42.6 strat cold), and its full-test cold 37.35 is the **program cold record** (+3.82pp over canvas D4; paired only-P3s1 96,288 vs only-D4 … see §7). The C3X law (floor-lr continuation is stable and grows) extends to native with a new role: it is the anti-memorization phase.
- Val-selection is the third instrument: it caught every peak (P1 20k, P2 30k, P5 20k). The chain ran full tests on the final grids only; **full-on-vsel becomes mandatory** in the champion chain.

### 2.5 The memorized map's failure texture (D-catalog addition)
P1@50k's failures carry a median of **5 violations** at 51/81 correct cells (valid_wrong 0.0009) — near-valid, confidently wrong complete grids; healthy maps' failures are stuck partial propagations (P3s1 median 32 violations, D4 27, at the same 51/81). A memorizing Sudoku map drifts toward the ARC-like wrong-stable texture (D5). Its residual selector inverts too: on P1's final grid t1r@k FALLS with k (0.39 → 0.01 → 0.00): the memorized wrong states are "more converged" than the rare correct draws.

## 3. Finding 2 — RI is the per-draw lever, and its fragility has a mechanism

### 3.1 What RI buys (paired, identical 20k set)
- **Per-draw (EqR's B=1 statistic):** b1 P2 34.48 / P2s1 35.59 / P5 34.89 / P3s1 **38.47** vs no-RI P1 **0.39** (P2 vs P1 only-P2 6,847 / only-P1 30, p≈0). The registration's biggest unknown resolves: RI at width under the dose is a +34–38pp per-draw effect — the H-37 deployment-init law at its strongest.
- **b1 ≈ cold at every rating octile on the RI arms** (P3s1 78.4/28.5/23.8/22.8/25.2/26.4/26.7 vs cold 79.0/28.7/23.7/22.5/24.8/27.4/26.0): the RI map is **init-invariant** — a random-init draw does exactly what the VOID start does. This is EqR's path-independence, trained (and the reason breadth buys little, §3.2).
- **Native+RI flattens the difficulty curve:** hardest-octile cold **26.0 (P3s1) / 22.2 (P2)** vs canvas D4 15.0 / C3X 2.7; the canvas wins the easiest octile (D4 84.3 vs P3s1 79.0). The 3-adic constraint geometry pays exactly where search binds (the S1/H-33 prediction) — labeled: RI × geometry are confounded (no canvas-RI arm exists at d96).
- **EqR's selector works on RI maps:** Top-1-by-residual@128 ≈ verified vote@128 on every RI arm (P3s1 52.49 vs 52.85; P2 48.92 vs 49.19; P2s1 43.12 vs 43.85; P5 46.07 vs 46.44) — the verification-free selection captures ≥98 % of the free verifier's gain, and unverified majority@128 = 43.95 (P3s1) sits +5.5pp over b1. **The program's first numbers in the field's own columns:** b1 38.47 (EqR B=1 93.0), Top-1-residual@128 52.49 (EqR 99.8), majority@128 43.95 (aug-HRM 96.9), single-prediction cold 37.35 (HRM 55.0 / TRM 87.4) — at 1.71M params and 3.2M training samples.

### 3.2 What RI costs: a deep-narrow funnel
(ρ,r) per hard octile: P3s1 ρ .36–.44, r .18–.23 (reachable set small, reached fast); P2 ρ .34–.40, r .16–.20; canvas C3X ρ .70–.82, r .03–.06 (wide-slow); canvas D4 ρ .58–.70, r .02–.06. Verified@128 adds only +14pp over b1 on RI arms (P3s1 38.47 → 52.85) vs +57pp on the canvas (D4 cold 33.3 → 81.3). Init-invariance is the mechanism: draws are redundant. **Portfolio fact (labeled): P2 ∪ C3X verified@128 = 92.31 %** on the identical 20k; P1 ∪ P2 ∪ P3s1 cold union 50.72 / vote 69.24. The coverage route (our column) and the per-draw route (their columns) want different maps.

### 3.3 Why RI arms die: η → 1 and state explosions, not the free attention channel
- The attention channel was **closed at every death** (A_total 13–372 nats at the last finite row; peaks 6e4–1e5 during warmup, closed by 2k as in the canvas dose cells): these are not H-48 free-attention events. The free STREAM channel inflates with training (I_total 1.2–1.4e5 nats on RI arms vs 8.6e4 on P1; canvas ~1e6), unpriced (β_flux 0).
- **η (the learned damping) rides to replacement dynamics under RI:** P2 .980, P5 .992, P2s1 .924 at its death, vs P1 .799 (no RI) and P3s1 .893 (T16 two-phase, clean). The wave-3a/rung-1 fragility class (A2s1 mildly expansive at d16; B1-d64 NaN at η→.995) reappears at width with RI as the driver.
- **Inference-time explosion census (CPU f32, cold trajectories, strat-512):** P2@50k's final map goes non-finite on 5/64 puzzles by step ~57 (max |z| 2.8e19; max |logit| 9.5e18); P5@40k reaches |z| 2.9e18 within 64 steps; P1 (η .80) and P3s1 (η .89) stay bounded (|z| ≤ 120). Full census in §7 (all grids, t=64 and t=256). The training deaths are the same class caught mid-batch: an RI row far from the trained manifold under a near-replacement map produces an expansive excursion; one such excursion in a batch is a NaN gradient. The one-shot rule is vindicated again, and so is the lens-C finding — no monitored precursor (λ_J at the last monitor before death: P2s1 0.76, P3 1.51, P5 1.01, P6 0.91; P3s1 lives at 1.2–2.1 and is clean; λ_J is not a stability readout on these maps).
- **A mechanism-derived stabilizer exists and is one variable:** cap η (the damped update's step) below replacement — e.g. η ≤ 0.9 — or penalize its approach to 1. P3s1 (η .89, clean, bounded, the record arm) is the existence proof at n=1; the champion registration proposes it as a single-toggle arm (Plan_2026-09-02 §3). Alternatives noted, not chosen: lower ri_p, RI only in the hot phase, a stream dose (priced arms have never died in any campaign — 0/14 — but the stream toll narrows Sudoku basins, D8).
- **T16 vs T12** on per-draw: P3s1 38.47 > P2 34.48 (paired p=9e-34) — confounded by seed and schedule; T16 pace 32 vs 39 it/s printed.

## 4. Finding 3 — native flux and the "arity signature"
Offline CPU forward (cold trajectory, t=64, the probe's strat-512 puzzles): native P1 I_s = [97,796, 15,038] nats (shares .869/.131), P3s1 [87,614, 7,963] (.918/.082); A_s 68–100 nats total (closed). Canvas D4 probe: I_s [930,279, 258,634, 81,339, 33,683, 6,310] (shares .712/.197/.062/.025/.005), total 1.31M. The native pyramid routes **~12× less stream flux** (81 cells, 2 cuts) and concentrates 87–92 % at its finest cut. The registered signature ("mass migrates s0→s1, s1 share ≥ 2× the canvas value") is NOT observed — and was ill-posed: in native9 the box-scale tokens are the OUTPUT of the s0 pooling, so the cell-level information must cross the first cut regardless of alignment; there is no cut "below the boxes" to migrate from. Descriptive only, as registered. P2's forward returned NaN on 128 puzzles (its final map's explosions, §3.3) — a flux profile for P2 is read at its 30k grid in §7.

## 5. Instrument and ops notes
1. **Memorization tripwire** = train `ce_in` (→0) + monitor-val decline; both were logged and neither was a rule. The champion registers CE ≥ 0.02 at end-of-training as a hard integrity gate and reads every performance number at the val-selected checkpoint with a full test (not final-only).
2. **Post-death screens** are garbage; the chain must skip fixed-step screens beyond a stopped arm's last finite grid; **early-NaN abort** in the trainer (nan check at monitor cadence) saves 25–30k wasted steps per death (~30 min each here).
3. **Two-stage arms** shipped only the stage-B directory (stage-A metrics lost; the stage-A ckpt survived in GCS as `P3s1_stageA_ckpt.pkl`) and no `config.json` on stage-A death — ship both.
4. **`select_ckpt` after a node change** needs the banked grids + metrics re-pulled, else `VBCK` silently falls back to the final (P1's scan) — make the fallback loud (`VB-FALLBACK-FINAL` echo) and pull the grids.
5. **Pulls:** `gcloud storage cp` per object + crc32c verification against `gcloud storage ls -L` (the `gsutil cp` serial stall at full size with a wrong hash is now recorded); macOS `/bin/bash` is 3.2 (no associative arrays) — portable scripts only.
6. **Pace (measured):** native d96 T12 50k in 43 min on a v6e-8 (≈19 it/s wall; 39 printed); T16 ≈ 0.83×; full test 8 min sharded-8; 20k scan k128 ≈ 27 min; arm ≈ 1.75–2h all-in. The ~12× step-compute estimate realized as ~6× wall (per-step overhead binds at these sizes, the d64 lesson again).
7. **λ_J** stays a non-alarm; the explosion census (CPU, $0, minutes) is a NEW inference-side stability instrument: it separates η→1 maps (P2/P5 explode) from bounded ones (P1/P3s1) on a final grid, and is the natural pre-registration readout for the η-cap arm.

## 6. Consequences for the champion (Plan_2026-09-02_Champion_sportC1.md carries the registration draft)
- **Geometry:** native9 stays (exonerated at val-selected checkpoints; the flattest difficulty curve measured; cold record). The §2 fallbacks (concat mixers / dyadic 16×16) are NOT triggered — a stated departure from the ARITY-HURTS letter's registered consequence, for the PI to ratify.
- **Regime (mandatory by mechanism):** aug ≥ 1000; two-phase (hot cosine → floor continuation); full test at vsel AND final; CE tripwire; explosion census on every final grid.
- **RI:** kept as the per-draw lever with a one-variable stabilizer arm (η cap) and ×3 seeds on the recipe (one-shot; survival is part of the readout).
- **Portfolio:** one wide-funnel arm (no-RI native and/or canvas C3X-class at d128) rides alongside for the coverage column and the union.
- **Width:** d128 (3.00M) primary; one d160 (4.66M) arm if the PI wants the ladder point toward HRM's 55.
- **Frontier honesty:** at d96/1.71M the field-column numbers are b1 38.5 / t1r@128 52.5 / majority@128 44.0 / cold 37.4. Beating HRM's 55.0 on single prediction is the realistic target of the d128–d160 round; TRM's 87.4 and EqR's 93.0/99.8 are not within this round's reach and the paper says so; B-M3 (.95) is plausibly reachable in our own coverage column via portfolio × verified attempts (92.3 at d96, labeled).

## 7. Addendum — measurements completed after the main pass (same day, CPU, $0)

### 7.1 Explosion census (`runs/analysis/sportC0_explosion_census_20260902.json`; strat-512 test puzzles, cold trajectories, CPU float32)
| grid | η | n | t | exploded (non-finite or \|z\|>1e6) | first blow-up step (median) | \|z\|max median / p99 |
|---|---|---|---|---|---|---|
| P1@20k(vb) | 0.698 | 512 | 64 | **0/512 (0.0 %)** | — | 45.5 / 66.8 |
| P1@50k | 0.799 | 512 | 64 | **0/512 (0.0 %)** | — | 54.1 / 71 |
| P2@30k(vb) | 0.966 | 512 | 64 | **0/512 (0.0 %)** | — | 39.7 / 58.8 |
| P2@50k | 0.980 | 512 | 64 | **97/512 (18.9 %)** | 40.0 | 67.5 / 2.5e+19 |
| P2s1@20k | 0.924 | 512 | 64 | **4/512 (0.8 %)** | 35.0 | 71.5 / 172 |
| P5@20k(vb) | 0.968 | 512 | 64 | **0/512 (0.0 %)** | — | 42.4 / 68.7 |
| P5@40k | 0.992 | 512 | 64 | **5/512 (1.0 %)** | 52.0 | 47.9 / 799 |
| P3s1@35k(stageA) | 0.884 | 512 | 64 | **0/512 (0.0 %)** | — | 44.1 / 65.9 |
| P3s1@50k | 0.893 | 512 | 64 | **0/512 (0.0 %)** | — | 45.5 / 71 |
| P6@10k | 0.805 | 512 | 64 | **0/512 (0.0 %)** | — | 48.2 / 90.7 |

Reading: every val-selected and every two-phase grid is bounded — including P2@30k at η .966 and P5@20k at η .968 — while the LATE grids of the T12 RI arms explode (P2@50k 18.9 % at η .980; P5@40k 1.0 % at η .992) and P2s1's last finite grid shows the first excursions (0.8 %) just before its death. The threshold sits sharply between η ≈ .97 and .98; the proposed champion cap η ≤ .90 (Plan §3, arm B0) is comfortably below it, and P3s1 (η .884–.893, the record arm) is bounded at both its stage-A and final grids. (Remaining rows — canvas D4 control and the t=256 subset — append below when they finish.)

### 7.2 Train-split memorization check (cold exact on the 1,000 seeded training puzzles, un-augmented; `scratchpad/memcheck`)
| grid | TRAIN cold exact | mean violations on train | TEST cold (full 422,786 / strat-512) |
|---|---|---|---|
| P1@50k (final) | **99.9 %** | 0.02 | 19.38 (full) |
| P1@20k (val-selected) | 73.9 % | 4.12 | 38.1 (strat-512) |
| P2@30k (val-selected) | 95.6 % | 1.03 | 34.66 (20k scan) |
| P2@50k (final) | 72.8 % | 53.7 (exploding map, §7.1) | 15.62 (full) |
| P3s1@50k (final, two-phase) | 95.0 % | 1.85 | 37.35 (full) |
| canvas D4@50k | (pending) | | 33.53 (full) |

Reading: the native 50k-cosine arms memorize their training set completely (P1) or nearly (P2@30k 95.6 %) while their test cold collapses; the two-phase arm sits at 95 % train / 37.35 test — memorization is well advanced even on the record arm, so the champion's aug1000 regime (10× the corpus, ~5 epochs at 80k steps) is not optional.

### 7.3 Val-selected 20k cold evals (identical puzzle set as the canvas D4 scan; k=0) — in flight on CPU; results appended below with paired McNemar vs D4 (33.28 on this set) and vs each arm's own final-grid scan.
