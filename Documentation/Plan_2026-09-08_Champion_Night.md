# THE CHAMPION NIGHT — design and registration DRAFT (2026-09-08; NOT registered: rules lock in `tools/analyze_champ.py --selftest` and the chain passes its harness before any launch; every band below is a draft the PI amends)

**Purpose (the PI's ask, 2026-09-07):** the final Sudoku runs, designed to maximize the score with interpretations and laws that push the frontier, from everything measured through the instrument suite on our arms, the field's public checkpoints and the corpus. **Inputs:** `Report_2026-09-08_Frontier_Analysis.md` (the field's design choices; pass two of Night A; the seven laws), `Report_2026-09-07_NightA_Verdict.md`, `Plan_2026-09-07_Instrument_Suite.md` (the battery), the ledger §3/§5.

## §1. What the measurements say the score is made of

| lever | measured effect | consequence for the night |
|---|---|---|
| the cell class | the DEC (the field's loop on our nine-field S9-exact state) reads 90.0–95.1 @D16 / 94.7–98.1 @D64 / 95.9–98.7 @D128 at 2.78M; the best public single pass 86.5 / 93.2 / 95.0 (EqR) | the DEC-w384 is the champion cell; no width change (A8 w512 −1.45 pp at matched seed, memorizes fastest) |
| the objectives | A5 (FPA k1 + RI σ1) = A7 (plain, seed 1) on cold (95.08 / 98.11 vs 95.12 / 98.04) and buys a clean selector (spurious .04 % vs 4.9–68.6 % on RI-free seeds; t1r 99.44 = verified 99.46); B0 := A5 by the registered rule | the champion recipe carries FPA k1 + RI σ1 |
| the trajectory | every DEC peak at 10–28k (A3 ties 16–42k); finals 12–23 pp under; the 64-puzzle monitor's 1.6 pp ties selected the memorization side | a 512-puzzle train-file monitor, grids every 2k, earliest tie, the raw monitor as the second key; 30k steps for the plain arms |
| the orbit | digit aug cannot help an equivariant cell (A4 inside the floor); position aug 1000 is the DEC's only orbit; the DEC memorizes 1k × 1000 at the field's rate | position aug 10,000 as one variable (the untested anti-memorization lever; 50k steps) |
| depth | +0.5 pp per doubling on the DEC at D128 (A7 98.66, A5 98.58), zero regressions; the field +1 pp to D256 | D64 headline row, D128 on 20k, D256 on 5k on every arm |
| the operator | the DeepSets mean coupling reaches parity (L3 resolved benignly); SE-RRM's attention over symbols reads 93.7 @2M with digit aug | set attention over the nine fields per cell as one variable (the L3 disentangling arm) |
| compute | the DEC-w384 = 4.6× X0's MACs per step; no point at or below the field's compute | w192 (1.34×) as the accuracy-per-MAC point |
| the class-intrinsic gap | at stalls 54–68 % of committed cells wrong on every decimating grid (18 grids incl. the four public models); the DEC stalls on 2–5 % | the calibrated commit head as the one mechanism arm (the verifier-dependence table's row) |
| selection / test-time columns | the verifier reaches 99.5–99.9 at k32 on every DEC arm; A5's residual selector 99.44 verification-free | the labeled columns ride on every arm's 5k × k32 scan at batch 128 |

## §2. Arms (tag `champ`; every arm the field regime: batch 768, wd 1.0, lr 1e-4 constant after 2k warmup, β2 .95, EMA .999, stablemax, SOT + ACT, bf16, `--grid-every 2000`, the 512-puzzle monitor; DEC-w384 unless noted; NO digit augmentation)

| arm | one variable (from its base) | base | steps | reads |
|---|---|---|---|---|
| **C0 / C1 / C2** | seeds 0 / 1 / 2 of the champion recipe: DEC-w384 + FPA k1 ε.2 frac.25 + RI σ1 (= A5's recipe) | — | 30k | the claim-bearing headline; the seed floor (n = 3); FLOOR_C = max spread |
| **C3** | position augmentation 10,000 | C0 | 50k | the orbit lever: peak height and the memorization onset (T1) |
| **C4** | the SE-RRM operator: equivariant set attention over the nine fields per cell in place of the mean coupling (build §5.2) | C0 | 30k | the L3 arm: does a stronger cross-symbol operator beat the mean? |
| **C5** | DEC-w192 (1.34× X0's MACs) | C0 | 30k | the accuracy-per-MAC point |
| **C6** | the calibrated commit head with selective hardening τ .9 (build §5.3) | C0 | 50k | E3 wrong-among-committed at stalls; the verifier-dependence row |
| rider | the evaluator's sync-policy A/B pace test (one 512-row batch of a k32 scan under `--sync-per-step` and the default) as the FIRST job; the reference-decoder DFS rung on CPU | — | — | ops + Fig. 1 |

Battery per arm (the suite): full D16 at vsel / final / alt (the wide-arm labeled subsamples for final/alt: 50k), full D64 on 100k (+ exact-by-step + q), D128 on the 20k scan set, D256 on 5k, the 5k × k32 scan at t64 (batch 128), the strat-512 k256 screen with the unverified majority, census (vsel + final), calibration, init radius (5 ε), the dynamics row (CPU, at analysis); screens at 2k multiples {10k, 20k, vb} (+ 30k, 40k on the 50k arms); monitors every 2k on 512 train-file puzzles at D16 (EMA + raw).

## §3. Decision rules (to lock verbatim in `tools/analyze_champ.py --selftest`; letters only)

- **INTEGRITY:** one checkpoint path per (arm, vsel) across every vsel-labeled eval; the per-class protocol gate (wide arms scanned at 5k × k32, never at 20k); n gates; identical idx on every paired contrast.
- **CLEAN split:** STABILITY {STOPPED, census > .02} separate from MEMORIZATION {end segment-CE < .02, vsel − final > .05}; a memorized arm's val-selected grid remains a measurement, labeled.
- **NOISE FLOOR:** FLOOR_C = max(spread of cold16 over the clean seed triple C0/C1/C2 at their selected grids, .005); contrasts read beyond 2 × FLOOR_C; the noise floor is also read at the arms' BEST grids (the offline re-selection on strat-512) and both are reported.
- **R-CH-0 SEEDS:** SPREAD-C ≤ 1.5 pp → TIGHT; ≤ 3 → LOOSE; else WIDE (the selection instrument re-examined before any other letter).
- **R-CH-1 ORBIT-10k:** C3 cold16 − mean(C0..C2) ≥ 2 × FLOOR_C → AUG10k-LIFTS; the memorization onset (CE < .05) later than C0's by ≥ 10k steps → +DELAYS; else FLAT.
- **R-CH-2 OPERATOR:** C4 − mean(C0..C2) beyond 2 × FLOOR_C → ATTENTION-HELPS / HURTS; else MEAN-SUFFICES.
- **R-CH-3 WIDTH-DOWN:** C5 cold64 ≥ 92.0 → PARITY-AT-1.3x; ≥ 87.4 → TRM-CLASS-AT-1.3x; else BELOW.
- **R-CH-4 COMMIT:** C6 E3 wrong-among-committed at stalls (τ .9, strat-128 dynamics) < 30 % (from 54–68) AND yield + ≥ 1 pp → CALIBRATED; wrong < 30 % without the yield → CALIBRATED-INERT; else UNCALIBRATED.
- **R-CH-5 SELECTOR:** every RI arm's spurious ≤ .01 at k32 → SELECTOR-CLEAN; the t1r/verified ratio reported.
- **R-CH-6 DEPTH:** D128 − D64 and D256 − D128 on each arm; regressions ≤ .05 %.
- **R-CH-7 PARITY:** single pass, EMA, D64 on the 100k subsample: ≥ 92.0 PARITY (all three seeds) ; the headline = the seed triple's mean and spread at D64 and D128.
- **CHAMPION BY RULE:** the highest CLEAN single-pass D64 among C0..C2 (seeded); "program-record" prefix only; the labeled best across every arm reported beside it.

## §4. Predictions (draft bands; credences to set at the registration)

C0..C2 cold16 ∈ [93.5, 96.5] each, spread ≤ 1.5 pp (60 %); cold64 ∈ [97.0, 98.5]; D128 ∈ [97.5, 99.0]; D256 ∈ [97.8, 99.2]; spurious ≤ .1 % on all three (75 %); verified@32 ≥ 99.5 (80 %). C3 (aug 10k): cold16 ∈ [95.0, 97.5], onset ≥ 10k later, final within 5 pp of vsel (50 %). C4 (attention): within ±1.5 of the triple's mean (55 %; HELPS 25 %). C5 (w192): cold16 ∈ [86, 93], cold64 ∈ [91, 96] (55 %). C6 (commit head): E3 wrong-among-committed < 30 % (45 %), cold within ±1 (60 %). Depth: +0.3–0.8 per doubling on every arm (65 %). The sync-policy A/B: the default policy at least 1.5× the per-step policy's row-draw pace on the chip (55 %).

## §5. Builds (each with its harness scenario and CI test; estimates in Fable days)

1. **B5 the selection instrument (0.25 d, required):** `pretrain.py --monitor-n 512` (the seeded 512 train-file puzzles at D16, EMA + raw rows), `select_ckpt.py --tie earliest --second-key <raw monitor>`; grids every 2k; the chain's screens at 2k multiples.
2. **The SE-RRM operator (0.5 d, C4):** `dec_cell` coupling = per-cell multi-head self-attention over the nine field tokens (queries/keys/values from the field's own token; shared weights over fields = equivariant by construction; `test_dec_exact_s9` extended); `--dec-coupling {mean, attn}`; param count pinned.
3. **The commit head (0.5 d, C6):** per cell per outer step `c = σ(v · z_H)` trained with BCE on "argmax correct"; selective hardening of the feedback where `c > τ` (straight-through); evaluator emits the head's calibration rows; CI: hardening is identity at τ = 1; the head's target is available at training only.
4. **The aug-10,000 corpus (0.1 d, C3):** the builder at 10,000 position variants (memory ≈ 10M × 81 bytes ≈ 0.8 GB; the node has 1.4 TB); a CPU smoke of the build time.
5. **The sync-policy A/B rider (0.1 d):** the chain's first job runs one 512-row batch of a k32 scan on A3's grid under both policies and banks the two walls (`SYNC-AB` marker).
6. **`analyze_champ.py` + `chain_champ.sh` + `harness_champ.sh` (0.5 d):** the Night A chain's mechanics (static maps, preflight, live bank, remat retry, idempotent markers, the DEC scan at batch 128) with the arm registry above; scenarios per resume path and per new flag; the negative scenarios assert the staged failure fired.
7. **Owed ops fixes (0.1 d):** `pod.sh` lists the zone after a create timeout (the stray CREATING record of 2026-09-07); the DMS pushed past the deadline at every launch (the 09-04 rule, automated).

## §6. Ops, walls, budget (inferred from Night A's measured paces; the preflight re-prices)

DEC-w384 trains at ≈ 1.3 it/s on a 4-chip worker (remat) and ≈ 2.2 it/s on 8 chips: 30k steps ≈ 6.4 h (4 chips) / 3.8 h (8 chips); 50k ≈ 10.7 / 6.3 h; the battery per wide arm ≈ 3 h on 4 chips (US pace; Mumbai's multi-draw loop runs at 0.42× — the A/B rider decides whether the sync policy recovers it). **Shapes:** a v6e-32 (8 workers × 4 chips): w0 C0 · w1 C1 · w2 C2 · w3 C3 · w4 C4 · w5 C5 · w6 C6 · w7 riders, long poles C3/C6 ≈ 14 h → cap launch + 18 h, ≈ $27.3/h × 15 h ≈ **$410 + weather** (the 32 was never granted in Night A; US weekend window Sep 12–13); a v6e-16 (4 workers) over TWO nights: night 1 w0 C0 C5 · w1 C1 C4 · w2 C2 · w3 C3; night 2 C6 + the riders ≈ 2 × 12 h × $13.64 ≈ **$330 + weather**; a v6e-8 sequential ≈ 45 h ≈ $310 (three nights; the calendar binds). **Policy:** one spot pod, US zones first, Mumbai last; the deadline knob bumped before launch; the launchd watchdog; `plant_guard` per node with the DMS past the deadline; the live 5-min bank; the ops phase reads no accuracy values. **Budget after the frontier run:** ≈ $874; the night ≈ $330–410; the ARC-d96 rung ≈ $150–250 after it; reserve ≈ $200–300.

## §7. Calendar

Sep 8 (Tue): this design; the PI's amendments; builds 1, 4, 5, 7 · Sep 9 (Wed): builds 2, 3, 6; smokes; pre-mortem; registration (ledger + analyzer selftest + harness green) · Sep 10 (Thu) evening IST: launch (a 16, two nights) or Sep 12 (Sat, US weekend) for the 32 · Sep 12–13: analysis (the analyzer untouched; the physics pass; the lens; the corpus tables) · Sep 13–15: the ARC-d96 rung · Sep 16: freeze; Sep 18 abstract; Sep 25 full paper.

## §8. Adversarial review (pre-registration)

1. *"Thirty thousand steps is early stopping by another name."* The peak is a measured property of the trajectory law on this cell (five arms, 10–28k); the monitor selects inside the run, the finals are reported beside the peaks, and the aug-10k arm tests whether the peak moves. 2. *"One seed per treatment arm."* The treatment arms read against a seed triple's floor, on identical puzzles, paired; contrasts inside 2 × FLOOR read FLAT by rule. 3. *"The commit head is ACT per cell."* Yes, and E3 measures whether it is calibrated where the halting head is not asked to be; the arm exists to move a class-intrinsic gap no field model has moved. 4. *"Mumbai's pace."* The A/B rider is the first job; if the sync policy does not recover the multi-draw pace, the scans stay at 5k × k32 and the walls above hold. 5. *"Verified columns are oracles."* Coverage columns only; the headline is the single pass; the residual-selected column is verification-free and reported at the verifier's level where it reaches it. 6. *"Where is ARC?"* After this night, inside the remaining envelope, with the DEC's ARC form and the verifier-dependence table.
