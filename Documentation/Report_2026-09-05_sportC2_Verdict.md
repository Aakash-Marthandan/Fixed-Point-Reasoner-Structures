# sportC2 (the pre-parity graft night at d128) — Verdict: Every Registered Rule Read, the Seven Questions Answered in Plain Words, and Where We Stand Against the Field

**Date:** 2026-09-05 (analysis pass; ops closed 2026-09-05 07:45Z by the Opus ride, `tools/HANDOFF.md` STATUS) · **Analyzer:** `tools/analyze_sportC2.py` byte-untouched since its creation commit eea3c32 and the registration commit 15735c3 (`git diff` empty for both; selftest 21/21) → `runs/analysis/sportC2_verdict.txt` · **Physics pass (analysis-time, descriptive, no rules):** `tools/analyze_sportC2_physics.py` → `runs/analysis/sportC2_physics_20260905.txt` · **Decoder lens (analysis-time, descriptive, no rules; the PI's error-correcting-code emphasis):** `tools/analyze_sportC2_ecc.py` → `runs/analysis/sportC2_ecc_20260905.{txt,json}` · **Data:** `gs://qhrrn2-rescue/sportC2/sportC2_final.tgz` (5,309,494,906 bytes) pulled with `gcloud storage cp`, crc32c re-verified `DG8G0A==` against the GCS object, extracted into `runs/`; every registered count present (7 arms + 4 stage-A dirs with `ckpt_latest.pkl`; 24 full tests at n=422,786; 5 retfm at 512; 7 scans at 20,000; 26 screens at 512; 14 censuses with 2 rows; 8 calibration rows; 5 riders). **Code provenance:** science code (`src/`, the trainer, evaluator, census, calibration, `select_ckpt`, the analyzer) byte-untouched since the registration commit (`git diff 15735c3..HEAD` on those paths empty); the ops-side changes since registration are the live-bank tool, the snapshot tool, the harness and the chain's restore call. **Spend ≈ $306** (Mumbai v6e-8 25.25 h ≈ $202 · US v6e-16 2.5 h $34 · US v6e-8 10.25 h $70; hand-derived from watchdog spans) vs the registered $170–230 band (1.3–1.8×, spot weather: 8 node incarnations, 3 preemptions, two Mumbai stints); program ≈ $2,860–2,960 of the $3,700 envelope. **Test suite:** 151 green at HEAD before the analysis.

Registration: ledger §5 2026-09-04 "sportC2 — LAUNCH REGISTRATION"; design `Plan_2026-09-04_sportC2.md`. Nothing here edits a rule; every letter is the analyzer's; every reading is labeled measured / inferred.

**The night in one paragraph.** Seven arms ran clean to their full battery — no death, no NaN, no out-of-memory retry, every trajectory bounded, every eval on one checkpoint per arm. None of the four levers we grafted onto our cell delivered the mechanism it was built to deliver: the persistent carry made the map reach fewer puzzles, not more; trained inner cycles did not make propagation deeper; hard-decision rows improved the map's confidence only a little; and weight decay at our learning rate held memorization off for longer but cost eight points of accuracy. Continuing our cell for fifty thousand more steps in the field's regime bought nothing. On the field's side the two additive arms produced the night's two largest effects, both against our predictions in size: removing digit augmentation collapses the field's recipe by sixty-five points because the TRM cell then memorizes the thousand training puzzles, and grafting our constraint-group mixer into their loop costs forty-seven points. The champion by rule is R3 (hard rows) at 43.57 % single-prediction on the full test, the program's best number by rule, with the same late collapse the un-decayed arms showed. The gap to the field's reproduced recipe (86.0 / 92.8) stands at about 45–50 points on single prediction and is not moved by any single graft at d128.

## 0. Integrity (post-results critique, before any value was read)

- **Admission (artifact level, 7/7):** every `ckpt_latest.pkl` carries the registered config — natives d128 / T16 / native9 / arity 3 / group9 / β_nl 1e-6 / FPA k4 ε.2 / z-norm rms, parameter count 3,004,658 (the z-norm pin) on all five; the graft flags exactly as registered (R1 `sot` with 4 segments, R2 `inner_k 3`, R3 `hard_p 0.5`, R4 `init_from` sportC1 R0's checkpoint under the field regime with remat); X1 = the X0 flags with `digit_aug False` (5,037,061 = the 5,037,058 pin + the loop's three unused damping scalars); X2 = X0 + `trm_token_mixer group9` at 5,839,877 (= the registered 5,839,874 + 3). `config.json` argv corroborates every regime flag (W0–R3: batch 64, wd 1.0, lr 1e-3 → 3e-5 two-stage, aug 1000; R4: batch 384, lr 1e-4 constant, wd 1.0, β2 .95, EMA; X1/X2: batch 768, wd 1.0, lr 1e-4, SOT + ACT).
- **Stability:** no `STOPPED.txt`, no `NAN_ABORT.txt`, no `RETRY_REMAT.txt` anywhere; every banked grid finite; retfm 1.00 at every monitor on W0/R3/R4, one dip to .88 on R2 and an early transient on R1 (.06 at 6k, .66 at 10k, 1.00 from 14k — the carried-state loop's settle); explosion census 0 % at t=64 and t=256 on every vsel and final grid of every arm.
- **Resumes (labeled draws, valid):** R2 stage A at 29,000, R3 stage A at 35,000, R4 at 10,000 (the live-bank restore proven on the Mumbai switch). **Labeled data gaps:** R2's stage-A grids at 5k–25k and R4's 5k/10k grids were lost with the preempted v6e-16, so R2's stage-A selection candidates start at 30k and R4's at 15k (monitors intact).
- **Val-selection provenance — the sportC1 lesson holds:** every vsel-labeled eval of every arm (full, alt, D64 row, scan, census, vb screen, retfm, calibration, and R3's two hard-feedback rows) reports ONE checkpoint path; the offline re-selection over both stages reproduces the chain's choice on all seven arms (W0 A:50k, R1 B:10k = effective 60k, R2 A:40k, R3 A:40k, R4 20k = effective 70k on R0's weights, X1 20k, X2 50k = final). **Instrument note (new):** grids are banked every 5k and monitors run every 2k, so only the 10k multiples are selection candidates (5k / 15k / 25k / … grids can never be chosen); R2's best monitor (54k, .453) and W0's (54k, .422) sat on non-candidate steps. This held in sportC1 too (B0's A:20k). A fix is owed (monitor cadence dividing the grid cadence).
- **Counts and pairing:** fulls n=422,786 on every row; all seven 20k scans share the identical puzzle set (subsample seed 20260822; idx sets equal; vote identity exact) with the sportC1 scans, the correct-grid riders, the canvas D4/C3X scans and the pilot's — every contrast below is exact McNemar on identical sets. The canvas sel-5k riders are on a DIFFERENT 5,000 (the seeded choice of 5k is not nested in the 20k); their pairings run on the 256-puzzle intersection and are labeled.
- **Noise instruments:** CNC2 = 8.20pp, because the rule defines it as |vcold(W0) − vcold(B0)| with a floor of 1pp on the assumption that wd is the only variable in a matched pair; W0 landed 8.2pp below B0, so the "matched-pair noise" is really the wd effect itself. Consequence for the mechanical rules: the R-C2-7 admission band (vcold ≥ W0 − CNC2) became lenient (any arm ≥ 26.4 qualifies) and the WD-HOLDS bar (≥ B0 + CNC2 = 51) unreachable. The true seed noise on this recipe class is the sportC1 z-norm pair's 0.12pp (full test) and this round's correct-grid pair B0c/B1c: 0.30pp cold, 0.44pp b1, 0.16pp verified@128 (all p > .18). FNC2 sits at its 2pp floor.
- **Registered deviations carried:** grafts on the W0 base (not R0's regime); R4's fresh optimizer and EMA restart with the step counter from 0; the SOT carry not checkpointed (R1 never resumed, so unexercised); R3 evaluated soft on every registered row with two labeled hard-feedback rows; W0's decoupled decay rate lr × wd ≈ 5e-4 in the hot phase (5× the field's 1e-4) and 3e-5 in the floor (0.3×) — the over-regularization reading the registration pre-labeled fired (§2.1).

## 1. Registered verdicts (the analyzer's letters)

| Rule | Letter | Numbers (analyzer) | Plain reading (physics pass, labeled) |
|---|---|---|---|
| INTEGRITY | **PASS** | n, idx, protocol, vote identity, one-ckpt-per-(arm, vsel) all pass | — |
| STABILITY | **ALL-STABLE** | no STOPPED; retfm ≥ .9 on every native's vsel grid; census 0 % | the first round of the champion track with zero deaths |
| MEMORIZATION | **NONE** (natives, by the CE letter) | end CE W0 .041 / R1 .567 / R2 .314 / R3 .075 / R4 .126; X1 .011 (labeled MEMORIZED in its row) | W0 and R3 did reach CE ≈ 0 mid-run (minima .0019 at 56k and .0080 at 70k) and their val-selected grids precede that; R3's final collapsed (§2.4) |
| R-C2-0 REGIME (W0) | **WD-PARTIAL** | CE .041 (< .05), val peak 54k (≥ 25k), vcold 34.63 vs 42.83 + CNC2 | wd 1.0 at lr 1e-3 = over-regularized: memorization delayed, not prevented, at −8.2pp cold (§2.1) |
| R-C2-1 CARRY (R1) | **CARRY-FLAT + SELECTOR-INTACT** | reach 43.77 (W0 56.85); t1r/verified .984 | the carry NARROWED the reachable set by 13pp while raising single-shot accuracy 2–3pp (§2.2) |
| R-C2-2 DEPTH (R2) | **DEPTH-FLAT** | first_exact median 9 (W0 11); reach at 21–25 givens 19.3 (W0 40.0) | trained inner cycles do not deepen propagation; the hardest-erasure puzzles got LESS reachable (§2.3) |
| R-C2-3 COMMIT (R3) | **COMMIT-PARTIAL** | top-5 correct at stalls 76.4 (W0 70.6; bar 90); entropy at step 1 .57 | calibration moved a little; the map does not commit earlier; but R3 is the best cold by rule (§2.4) |
| R-C2-4 CONTINUATION (R4) | **REGIME-FLAT** | vcold 37.12 vs R0 37.33 | fifty thousand more steps in the field regime buy nothing on single prediction (§2.5) |
| R-C2-5 ORBIT (X1) | **ORBIT-LOAD-BEARING** | cold@D16 21.20 vs X0 86.03 | without digit augmentation the field's cell memorizes; −65pp (§2.6) |
| R-C2-6 GEOMETRY (X2) | **GEOMETRY-HURTS** | cold@D16 38.91 vs 86.03 | our group mixer as their token mixer: −47pp at 50k, slow learning, no init-robustness (§2.7) |
| R-C2-7 CHAMPION-RECIPE | **CARRY: wd1.0 \| CHAMP: R3 (43.57)** | no graft earned its primary letter; wd carried by PARTIAL | the mechanical champion is R3 at its val-selected 40k grid (§2.4, §3) |

Per-arm table (headline weights: raw on W0–R3, EMA on R4/X1/X2; vsel grid unless named; the 20k scan for b1 / verified / t1r; the strat-512 vb screen for majority):

| arm | recipe | vsel cold (full) | final cold | alt-weights cold | b1 (B=1) | verified@128 | t1r@128 | majority@128 | retfm | end CE | funnel ρ / r (hard octiles) | vsel grid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| W0 | B0 + wd 1.0 | 34.63 | 30.12 | 36.22 (EMA) | 34.31 | 56.85 | 56.42 | 35.94 | 1.00 | .041 | .40–.46 / .13–.14 | A:50k |
| R1 | W0 + carry | 37.56 | **38.41** (rising) | 38.22 (EMA) | 37.13 | 43.77 | 43.05 | 39.65 | 1.00 | .567 (mixed rows) | .22–.34 / .28–.32 | B:10k (60k) |
| R2 | W0 + inner cycles K3 | 34.03 | 30.60 | 39.37 (EMA) | 32.16 | 42.68 | 41.48 | 33.59 | 1.00 | .314 | .24–.32 / .23–.25 | A:40k |
| **R3** | W0 + hard rows | **43.57** | 21.78 | **44.65** (EMA) | 43.11 | 51.66 | 50.97 | 43.95 | 1.00 | .075 | .36–.44 / .28–.36 | A:40k |
| R4 | R0 + 50k, field regime | 37.12 | 32.18 | 34.57 (raw) | 37.34 | 56.12 | 55.53 | 34.96 | 1.00 | .126 | .40–.48 / .14–.18 | 20k (= 70k) |
| X1 | X0 − digit aug | 21.20 @D16 / 24.23 @D64 | 19.47 | 13.90 (raw) | 26.55 @D64 | 69.53 | 62.76 | 22.46 | — | .011 | .44–.62 / .05–.08 | 20k |
| X2 | X0 + group9 mixer | 38.91 @D16 / 43.74 @D64 | = | 29.62 (raw) | 0.01 | 43.58 | 0.03 | 0.00 | — | .708 | .02 / .002 | 50k = final |
| refs | sportC1 X0 (field baseline) · R0 (our cell, field regime) · B0/B1 correct A:20k grids (this night's riders) | 86.03 @D16 / 92.81 @D64 · 37.33 · 42.83 / 42.71 (EMA 45.42 / 45.65) | | | 92.12 · 36.84 · 42.75 / 42.31 | 99.70 · 50.80 · 52.97 / 52.81 | 92.48 · 50.18 · 52.78 / 52.65 | | | | .98 / .5–.9 · .32–.42 / .16–.20 · .36–.44 / .25–.32 | |

**Prediction scoreboard (bands locked pre-data):** W0 vcold [43, 50]: MISS-BELOW (34.63); CE ≥ .05: MISS (.041). R1 reach [.65, .85]: MISS-BELOW (43.77); vcold [42, 50]: MISS-BELOW (37.56); selector intact (80 %): HIT. R2 first_exact median ≤ 6: ABOVE (9); vcold [44, 52]: MISS-BELOW (34.03). R3 top-5 ≥ .90: MISS-BELOW (76.4); entropy at step 1 < .3: MISS (.57); vcold [42, 50]: **HIT (43.57)**. R4 vcold [39, 44]: MISS-BELOW (37.12). X1 cold@D16 [80, 85], "a drop ≥ 1pp": the direction HIT, the size MISSED by a factor of twenty (21.20). X2 [84, 88], neutral: MISS-BELOW (38.91). Ops: wall ≤ 16 h on a 16 / the 8-shape's 20–22 h: the mixed campaign took 41 h wall across 8 incarnations (the sequential 8-shape ran ~25 h of Mumbai time), inside the extended 44 h cap; spend 1.3–1.8× the band (weather). Ten of eleven scientific predictions missed; the night was informative because every miss carries its mechanism (§2).

## 2. The seven questions, answered in plain words

### 2.1 Does weight decay at our learning rate stop memorization? (W0 = B0 + wd 1.0) — Partly, at a large cost

What we did: the exact recipe that produced last round's 42.8 % (B0: our learning rate 1e-3, two-phase schedule, augmentation 1000, z-norm, seed 0), with weight decay raised from 1e-4 to the field's 1.0 as the only change.

What happened: training error no longer collapsed during the hot phase — at 20k steps the training cross-entropy was still .66 where B0's was .21, and the map's held-out monitor rose slowly (.27 at 24k, .34 at 32k, .39 at 50k) instead of peaking at 20k. The selected checkpoint is the END of the hot phase (50k) and scores **34.63 %** on the full test — **8.2 points below B0's 42.83 at its own peak** (paired on identical puzzles: only-W0 35,820 / only-B0 70,481, p ≈ 0). Then, in the floor phase (learning rate 3e-5), the training error fell to .0019 at 56k and the map memorized after all: final 30.12 %.

Why (the twin table in the physics pass, matched step by step against B0): with decoupled weight decay the regularization strength is the product lr × wd. At our hot-phase learning rate that product is 5e-4, five times the field's 1e-4 — the map is over-regularized and learns slowly (the free-stream flux stays at 23–37k nats, the codebook never commits, rule_H 1.9 at the end); in the floor phase the product drops to 3e-5, a third of the field's, and memorization returns. Weight decay is the lever (last round's reading stands: R4 in the field regime never memorized in the same sense), but the quantity to match is the decay rate, not the decay coefficient. The registration pre-labeled exactly this outcome ("an over-regularized W0 reads WD-PARTIAL by the letter") and named the matched-rate arm (wd 0.1 at lr 1e-3) as the next one-variable test. Side effect worth keeping: W0 has the WIDEST reachable set of any native arm (56.85 % of puzzles reachable with 128 verified restarts vs B0's 52.97, p 4e-26) — slow regularized training keeps restart diversity that the sharper B0 lost.

Letter: WD-PARTIAL. Carried to sportC3 by the mechanical rule, with the rate correction as the recommended form.

### 2.2 Does training on the map's own late states widen what it can reach? (R1 = W0 + persistent carry) — No; it narrowed it

What we did: the field's "state-over-time" loop on our cell — every optimizer step advances every training row one 16-step segment from where it left off; rows are replaced when the verifier says they are solved or after four segments (64 steps).

What happened: single-shot accuracy rose (37.56 % at the selected grid, 38.41 % at the end — the only native arm still improving at 80k; +2.9 points over W0 paired), but the reachable set SHRANK: with 128 verified restarts R1 solves 43.77 % of puzzles where W0 solves 56.85 % (paired: only-W0 3,888 / only-R1 1,272, p 6e-304). On every rating octile above the easiest the funnel model reads ρ .22–.34 (W0 .40–.46) with a per-restart hit rate r .28–.32 (W0 .13–.14): the carried map converges more reliably on the puzzles it can solve and reaches fewer of them.

Why: the carry trains the map on its own stationary states; it becomes more path-independent (restarts agree, b1 = cold to 0.4pp), which is the same property that made our RI maps stop exploring last round, now stronger. New, and important for the decoder reading in §4: R1's convergence residual no longer separates correct from wrong restarts (AUC .62; 36 % of wrong restarts sit below the correct-restart median residual) — the carried map has learned STATIONARY WRONG STATES, soft high-entropy fixed points (entropy .45 on unsolved puzzles at t=64, commitment 7 %) that look "converged". The verification-free selector still reads .984 of the verifier only because outcomes barely depend on the restart. The carry's early transient (retfm .06 at 6k, settled by 14k) and its late Jacobian rise (λ_J 1.6–2.0 from 62k with retfm 1.00) are the trajectory signatures.

Letter: CARRY-FLAT + SELECTOR-INTACT. The freethink's prediction (ρ ≥ .7 by 50k) failed in the opposite direction. Not carried.

### 2.3 Does more computation per step make propagation deeper? (R2 = W0 + 3 trained latent passes per readout) — No

What we did: three passes of the cell through the carried latent before each readout, trained that way from scratch (the evaluator, census and calibration mirror it), tripling the compute per outer step.

What happened: the outer step at which solved puzzles first become exact moved from a median of 11 (W0) to 9 — not the halving the rule required — and the puzzles with the fewest givens (21–25) became LESS reachable (19.3 % vs 40.0 %). Single-shot accuracy is unchanged (34.03 vs 34.63, paired p 1e-16 in W0's favor), the reachable set is narrower (42.68 vs 56.85), and the EMA weights are worth a large +5.3 points on this arm (39.37). The K=3 unroll also produced the wildest Jacobian estimate of the round (λ_J 40.8 at one monitor) with retention intact (min .88).

Why: last round's freethink measured that inner cycles at inference destroy the map (1.6 %) and predicted they would work if trained in. Trained in, they do not add propagation: our cell's step is a 5-layer scale-hierarchy pass, and running it three times per readout does not turn it into the field's 42-layer flat step. The "RG time per readout" hypothesis is dead at K=3 on our cell at d128.

Letter: DEPTH-FLAT. Not carried.

### 2.4 Do hard decisions during training make the map's confidence trustworthy? (R3 = W0 + hard-decision rows on half the steps) — A little; but R3 is the best single-shot number the program has by its rule, and it collapses late

What we did: on a random half of the outer steps during training, the feedback the cell receives is its own argmax one-hot instead of its soft readout (straight-through gradient); every registered evaluation still runs the shared soft loop, and two labeled rows read the hard-feedback mode.

What happened: at stalls, the five most confident cells are right 76.4 % of the time (W0 70.6 %; the bar was 90 %) at a mean confidence of .96 — better, still badly over-confident; the map does not commit earlier (entropy at step 1 .57, same as W0). But R3's val-selected grid (40k) scores **43.57 %** on the full test (44.65 with EMA weights), **+8.9 points over W0 paired** (only-R3 75,153 / only-W0 37,371, p ≈ 0), +6.2 over R0 and +10.0 over the canvas record D4 — the highest single-prediction number the program has produced by its rule (the sportC1 B0/B1 grids at 42.8 were disqualified by the CE gate; their EMA rows 45.4/45.7 remain the highest under any convention, labeled). The selector is intact (t1r/verified .987, no spurious attractors: 0.0 % of wrong restarts below the correct median). In hard-feedback inference the same grid scores 43.16 on the strat set with failures that are near-valid (3.2 violations vs 7.4 soft) — the commitment changes the failure texture, not the count.

Then it collapsed: the monitor peaked at 40k (.4375) and fell (.328 at 50k, .19 at 54k); training error reached .008 at 70k; the final grid scores 21.78 %. The hard rows accelerate fitting — the training error of R3 fell faster than W0's at every step from 20k (.53 vs .66 at 20k, .31 vs .36 at 40k) — and the peak arrives at 40k instead of B0's 20k because of the decay; then the same memorization follows. So the +8.9 points are a training-signal effect (the straight-through argmax is a stronger, sharper supervision on the answer register), not calibrated commitment, and they come with B0's failure mode delayed by 20k steps. Val-selection caught the peak exactly (the chain's A:40k = the offline argmax).

Letter: COMMIT-PARTIAL. Not carried by the mechanical rule (its primary letter is not CALIBRATED); champion-so-far by the rule's argmax.

### 2.5 Was our cell simply under-trained in the field's regime? (R4 = last round's R0 continued for 50k more steps) — No

What we did: restart from R0's 50k weights (a fresh optimizer, EMA restarted) and train 50k more steps under the field's optimizer (batch 384, lr 1e-4 constant, wd 1.0) — 100k steps and 38 epochs in total.

What happened: the monitor was flat for the whole continuation (.28–.34), the selected grid (20k into the continuation, 70k effective) scores **37.12 %** vs R0's 37.33 (paired on the full test −0.21pp, p 3e-4 — a tie in practice; on the 20k set +0.08, p .78), the final grid 32.18 % (a mild downward drift while training error kept falling, .32 → .13: the first sign of memorization even at the field's decay rate at this epoch count). What did move: the reachable set widened from 50.80 to 56.12 % with restarts (p 1e-80; ρ .40–.48 vs .32–.42) — continued training in this regime keeps opening reachability slowly without converting it into single-shot accuracy.

Why: our cell in the field's regime plateaus at 37 % single prediction. Last round's labeled read ("REGIME-UNDERTRAINED-AT-50k", R0's monitor was still rising) is refuted; the plateau was already there. The regime question is now closed in both directions: their regime cures our memorization and instability and buys no accuracy; their accuracy comes from their cell and loop.

Letter: REGIME-FLAT.

### 2.6 Does the field's recipe need digit augmentation? (X1 = X0 without digit relabeling) — Yes, enormously

What we did: the reproduced field baseline (TRM cell, state-over-time loop, ACT, batch 768, wd 1.0, 50k steps — 86.03 % last round) with the digit-permutation augmentation switched off (position augmentation 1000 kept).

What happened: the cell memorized the thousand training puzzles almost immediately (training cross-entropy .015 at 10k, train-exact .99, the halting head halting 96 % of rows) and scores **21.20 % at D16 / 24.23 % at D64** on the full test — a drop of **64.8 / 68.6 points**, paired (only-X0 278,715 / only-X1 4,651). Its failures include 5.9 % fully valid grids that contradict the givens (givens kept .9975): the memorized cell recalls a solution it has seen instead of solving the puzzle in front of it — the wrong-stable texture we know from ARC. Restarts still explore (a wide, slow funnel: 69.5 % reachable with 128 verified restarts, hit rate .05–.08 per restart) and the residual selector on this map is calibrated (AUC .98, spurious 1.6 %): what died is the single-shot decoder, not the landscape.

Why, and what it means for us: the digit relabeling makes each training puzzle up to 9! different puzzles for a cell that treats digits as arbitrary tokens; without it, a 5M-parameter cell stores the thousand solutions. Our cell is exactly equivariant to digit relabeling by construction (S9) and has never used digit augmentation. This is the cleanest number the program has for what that symmetry is worth in the field's own units: **≈ 65 points of single-prediction accuracy at matched everything else**. It also sharpens last round's memorization law: position-group augmentation (which the mixer can learn) adds nothing, but the digit orbit (which the TRM cell cannot shortcut) is the field's actual anti-memorization device — weight decay, ACT and EMA together do not replace it.

Letter: ORBIT-LOAD-BEARING (predicted; the size was not: the registered band expected a 1–6 point drop).

### 2.7 Does our constraint-group mixer help their loop? (X2 = X0 with the token-mixing MLP replaced by our factorized group mixer) — No; it hurts badly at 50k

What we did: TRM's token mixer (a learned all-to-all mixing across the 81 cell tokens) replaced by our shared all-different operator over the 27 constraint groups on a 64-dimensional projection, the prefix tokens fed a pooled cell summary; 5.84M parameters vs 5.04M.

What happened: **38.91 % at D16 / 43.74 % at D64** (−47.1 / −49.1 points paired). The map is still learning at 50k (training cross-entropy .71 vs X0's .60; the monitor rising +9pp over the last 10k; train-exact .03; rows carried for 8.5 segments on average) — it is slow, not dead — and it has lost the field cell's restart robustness completely: a single random-init restart solves 0.01 % of puzzles (X0: 92.1 %), so restarts add nothing (43.58 with 128 verified restarts ≈ cold) and the majority vote over restarts is 0. The residual selector is broken on it (AUC .73).

Why (labeled inference): the all-to-all mixer is load-bearing in TRM — it carries the two-hop shortcuts between cells that share no constraint group, and the loop's path-independence rides on it; our group mixer, restricted to the constraint graph on a narrow projection, is a far weaker per-step operator inside a flat 42-layer stack, and on a random-init latent it never finds the solution path. The prediction ("the 27 groups are the graph already; ±1pp") was wrong; the geometry that makes our cell's difficulty curve flat does not transplant into their loop as a drop-in mixer.

Letter: GEOMETRY-HURTS.

## 3. Where we stand against the field (all measured; protocol and regime columns named)

| statistic (Sudoku-Extreme, full 422,786 test unless named) | the field's recipe reproduced on our stack (X0, 5.0M; last round) | our best by rule this night (R3 at its val-selected 40k grid, 3.0M) | our best under any convention (labeled) | published comparators |
|---|---|---|---|---|
| single prediction, one cold pass | 86.03 (16 outer steps) / 92.81 (64) | **43.57** raw / 44.65 EMA (64 outer steps) | 45.65 (B1 EMA on its A:20k grid; the arm memorized after) | EqR 84.8 (D16 base) / 93.0 (D64 B=1); TRM-MLP 87.4; HRM 55.0; CMM 93.7 @5M, 85.4 @0.26M |
| one random-init draw (their B=1) | 92.12 @D64 | 43.11 | 42.75 (B0 correct grid) | EqR 93.0 |
| 128 verified restarts (our coverage column) | 99.70 | 51.66 (R3) · 56.85 (W0, the widest native) | 52.97 (B0c) | PTRM pass@100 99.06 (noise rollouts + Q selector) |
| 128 restarts, residual-selected, no verifier (their 99.8 column) | 92.48 (base, no RI/NI) | 50.97 | **88.16 on the canvas C3X map** (sel-5k rider, ratio to the verifier .988) | EqR 99.8 (RI/NI-trained) |
| majority over 128 restarts, no verifier | 88.87 | 43.95 | — | aug-HRM 96.9 (orbit majority) |
| portfolio, verified@128 (labeled protocol) | — | W0 ∪ R1 ∪ R2 ∪ R3 ∪ R4 = 78.03 · + canvas C3X/D4 = 98.25 | + X0 = 99.83 | — |
| the same recipe without digit augmentation | 21.20 / 24.23 | n/a — exact S9 | | |
| inference compute per puzzle (inferred from layer counts) | 16 × 42 = 672 layer-applications at D16 (2,688 at D64) | 64 × ≈5–6 ≈ 350 | | a measured MAC count is still owed |

Plainly: the field's reproduced recipe solves 86–93 % of test puzzles in one pass and essentially all of them with restarts; our best cell solves 43–46 % in one pass and 52–57 % with restarts, with a portfolio of our own maps reaching 98 % when restarts are verified. The 45–50 point single-prediction gap survived every graft we tried at d128. Two things this night adds to the comparison that no series paper reports: their recipe depends on digit augmentation for 65 points (our symmetry gives that for free), and their all-to-all token mixer is load-bearing (our constraint-graph mixer is not a drop-in). Our maps remain the cleaner decoders on selector reliability (no spurious attractors on W0/R3/R4/B0c/B1c) and hold the frontier-adjacent verification-free number on the canvas (88.2).

## 4. The decoder lens (error-correcting-code reading; `tools/analyze_sportC2_ecc.py`, analysis-time, descriptive)

**The frame.** A Sudoku puzzle is a codeword of the constraint code (81 cells, 27 all-different checks) received with 81 − givens cells erased — 57–79 % of the codeword at 17–35 givens. Every map is a decoder for that erasure channel, and the test set is the channel. Three reference points make the reading quantitative: the maximum-likelihood decoder solves every puzzle (unique solutions); a pure propagation decoder (naked and hidden singles iterated to a fixpoint, computed here) solves **11.1 %** of the natural test distribution — 77 % of tdoku's rating-0 class and none of the puzzles that need a guess; tdoku's rating-0 class is 14.5 % of the distribution. A learned map sits between the two curves, and how far it climbs above the propagation decoder on puzzles that need guesses is its measured search (decimation) yield. All rows below are on the identical 20,000-puzzle scan set (record level) or the rating-stratified 256-puzzle cold trajectories (dynamics), descriptive, from `runs/analysis/sportC2_ecc_20260905.{txt,json}`.

### 4.1 Erasure thresholds — where each decoder's single-shot success crosses 50 %

| map | g50 cold (givens) | 10–90 % width | g50 with 128 verified restarts | solves rating-0 (propagation class) | solves rating > 0 (search yield) | solve rate by rating band 0 / 1–9 / 10–29 / 30–59 / 60+ |
|---|---|---|---|---|---|---|
| W0 | 27.0 | 11.8 | 24.2 | 82.6 | 27.1 | 83 / 40 / 21 / 24 / 20 |
| R1 | 26.6 | 11.2 | 25.8 | 95.4 | 27.5 | 95 / 47 / 18 / 23 / 19 |
| R2 | 27.0 | 10.5 | 25.9 | 85.5 | 24.6 | 86 / 36 / 19 / 23 / 18 |
| **R3** | **25.9** | 11.3 | 25.0 | 89.9 | **35.5** | 90 / 46 / 30 / 34 / 28 |
| R4 | 27.0 | 14.6 | 24.2 | 87.5 | 28.6 | 88 / 42 / 24 / 25 / 20 |
| B0c / B1c (sportC1, correct grids) | 26.1 / 26.1 | 12.6 / 12.0 | 24.8 / 24.8 | 89.2 / 88.6 | 34.9 / 34.6 | 89 / 45 / 29 / 34 / 31 |
| R0 (our cell, field regime) | 27.1 | 14.8 | 25.1 | 90.6 | 28.0 | 91 / 42 / 22 / 25 / 18 |
| **X0 (the field's recipe)** | **10.9 (no threshold in 17–35)** | 24.0 | none | 100.0 | **91.7** | 100 / 95 / 89 / 91 / 92 |
| X1 (X0 − digit aug) | 29.0 | 14.0 | 19.0 | 67.5 | 17.3 | 68 / 37 / 11 / 9 / 7 |
| X2 (X0 + group mixer) | 26.1 | 13.8 | 26.0 | 97.8 | 34.3 | 98 / 59 / 22 / 28 / 32 |
| canvas D4 / C3X | 27.2 / 29.5 | 11.8 / 15.9 | **9.7 / 9.6** | 93.5 / 89.8 | 23.1 / 13.6 | 94 / 44 / 15 / 17 / 12 · 90 / 37 / 5 / 5 / 2 |

Reading in plain words: every one of our native maps stops working somewhere between 26 and 27 givens (about 55 erased cells) with a wide, gentle waterfall; they solve most puzzles that pure propagation can solve and a flat 18–35 % of the puzzles that need guesses, at every difficulty band. The field's map has no threshold anywhere in the tested range and solves 92 % of the guess-requiring puzzles. R3 is the strongest native decoder on both counts (threshold 25.9, search yield 35.5), matching the sportC1 B0/B1 grids, and the carry (R1) is the best pure propagator among ours (95 % of the rating-0 class) without any search gain. Two more readings: the canvas maps are the weakest single-shot decoders but lose their threshold entirely under list decoding with the verifier (a wide funnel plus a verifier is maximum-likelihood-like on this channel), and memorization removes decoding radius from the low-givens end first (R3 from 71 % to 30 % at 27–29 givens between its peak and its final grid; B0 from 50 to 10 % at 17–21).

### 4.2 List decoding — what 128 verified restarts buy, and why it differs by map

| map | verified@1/8/32/128 | decodable fraction ρ | per-puzzle per-trial rate: never / (0,.05] / (.05,.2] / (.2,.5] / (.5,1] | restarts for 50 % / 90 % of the decodable set | rescue of a cold failure by 1 draw / by any of 128 |
|---|---|---|---|---|---|
| W0 | 38.7 / 46.5 / 52.2 / 56.9 | 56.9 | 43 / 9 / 7 / 7 / 34 | 0 / 26 | 5.5 / 33.5 |
| R1 | 38.7 / 41.2 / 42.7 / 43.8 | 43.8 | 56 / 2 / 2 / 3 / 37 | 0 / 2 | 2.2 / 10.3 |
| R2 | 35.0 / 38.6 / 40.9 / 42.7 | 42.7 | 57 / 3 / 3 / 4 / 33 | 0 / 8 | 2.4 / 13.9 |
| R3 | 44.8 / 48.0 / 50.2 / 51.7 | 51.7 | 48 / 3 / 3 / 3 / 43 | 0 / 4 | 2.6 / 14.6 |
| R4 | 41.0 / 47.7 / 52.2 / 56.1 | 56.1 | 44 / 7 / 6 / 6 / 37 | 0 / 18 | 6.0 / 30.2 |
| B0c | 44.7 / 48.5 / 51.0 / 53.0 | 53.0 | 47 / 4 / 3 / 4 / 43 | 0 / 6 | 3.3 / 17.8 |
| X0 | 95.0 / 97.6 / 98.8 / 99.7 | 99.7 | 0.3 / 2 / 2 / 3 / 94 | 0 / 0 | 29.9 / 95.9 |
| X1 | 30.7 / 42.5 / 54.6 / 69.5 | 69.5 | 31 / 27 / 10 / 7 / 25 | 3 / 68 | 8.2 / 59.6 |
| X2 | 43.5 / 43.5 / 43.5 / 43.6 | 43.6 | 99.6 / 0.4 / 0 / 0 / 0 | 0 / 0 | 0.0 / 0.1 |

Reading: on our maps each puzzle's "channel" is nearly all-or-nothing — 43–57 % of puzzles are never decoded from any start, 33–43 % are decoded on almost every trial, and only 6–23 % sit in between, where a list of restarts helps. That is what init-invariance looks like in decoder terms, and the carry sharpens it (R1: 56 / 37 / 6; two restarts already cover 90 % of what it can ever reach). W0 and R4 are the only natives with a real list-decoding regime (26 and 18 restarts to cover 90 %; a third of cold failures rescued). The field's decoder reaches 94 % of puzzles at a per-trial rate above .5 and rescues 96 % of its cold failures with restarts; the memorized field cell keeps a broad channel (restarts still explore) while its single-shot decoder is dead; the group-mixer cell has no channel at all (no restart ever succeeds).

### 4.3 Decoding dynamics — soft decoders that revise, hard decoders that freeze (strat-256 cold trajectories, 64 steps)

| map | solved | non-given cells correct after step 1 / 8 / 64 (solved ‖ unsolved) | cells committed (p > .9) at step 1 / 64 (solved ‖ unsolved) | confidently wrong at step 64 (unsolved) | monotone solved trajectories | flips-to-wrong per cell in the last 32 steps (unsolved) | syndrome of the argmax grid on unsolved puzzles at step 1 → 64 (late oscillation) | first-exact median / p90 |
|---|---|---|---|---|---|---|---|---|
| W0 | 29.7 | 51 / 82 / 100 ‖ 42 / 51 / 47 | 5 / 100 ‖ 1 / 40 | 15 | 7.9 | 2.27 | 62 → 14 (11.5) | 12 / 28 |
| R1 | 42.6 | 50 / 77 / 100 ‖ 41 / 50 / 52 | 4 / 65 ‖ 1 / 7 | 1 | 2.8 | 0.81 | 61 → 24 (5.3) | 16 / 38 |
| R2 (K=3) | 33.6 | 58 / 90 / 100 ‖ 45 / 52 / 51 | 12 / 99 ‖ 3 / 12 | 2 | 9.3 | 0.13 | 52 → 25 (3.8) | 10 / 16 |
| R3 | 45.3 | 50 / 79 / 100 ‖ 43 / 51 / 49 | 4 / 100 ‖ 1 / 34 | 14 | 3.4 | 0.22 | 57 → 14 (6.9) | 14 / 29 |
| R3, hard-feedback inference | 40.2 | 50 / 80 / 100 ‖ 43 / 51 / 48 | 4 / 100 ‖ 1 / 47 | 19 | 3.9 | 1.43 | 57 → 6 (8.4) | 13 / 28 |
| R4 | 40.2 | 48 / 77 / 100 ‖ 41 / 50 / 45 | 3 / 100 ‖ 1 / 45 | 21 | 4.9 | 1.65 | 62 → 9 (9.9) | 12 / 31 |
| R0 / B0 (sportC1) | 41.8 / 44.5 | 49 / 79 / 100 ‖ 41 / 51 / 46 · 50 / 84 / 100 ‖ 42 / 49 / 45 | 4 / 100 ‖ 1 / 37 · 4 / 100 ‖ 1 / 26 | 15 / 11 | 2.8 / 6.1 | 1.90 / 0.81 | 62 → 12 (10.6) · 60 → 13 (8.4) | 12 / 32 · 10 / 25 |
| **X0 (field)** | 92.2 | 71 / 93 / 100 ‖ 45 / 43 / 46 | 96 / 100 ‖ 94 / 90 | 47 | **62.7** | 3.27 | 22 → 34 (20.5) | 1 / 10 |
| X1 (memorized) | 22.7 | 73 / 92 / 100 ‖ 41 / 41 / 40 | 99 / 100 ‖ 98 / 98 | 59 | 43.1 | 5.54 | 15 → 10 (16.3) | 2 / 15 |
| X2 | 44.1 | 68 / 92 / 100 ‖ 49 / 52 / 51 | 92 / 100 ‖ 86 / 90 | 42 | 42.5 | 1.46 | 35 → 30 (9.3) | 2 / 22 |

Reading. Our maps are soft-decision decoders: at step 1 they commit 3–12 % of the cells, keep the readout at half its maximum entropy, and reach their solutions by revising — only 3–9 % of solved trajectories are monotone (a decided-correct cell that never flips back), and on the way to a solution a typical cell is un-decided about twice. The field's map is a hard-decision decoder: 96 % of cells committed after the first step, first-exact at step 1, 63 % of its solves monotone (one-shot, then a peeling-like clean-up). At stalls the syndrome (violations of the current best guess) plateaus on every map — no map's syndrome is monotone non-increasing on unsolved puzzles, the signature of a trapping set rather than a slow decode — but the two families stall differently: W0, R3, R4 and the sportC1 maps stall committed (34–45 % of cells at p > .9, 14–22 % of cells confidently wrong, the syndrome oscillating by 7–12), the field's map stalls in confident oscillation (90 % committed, 47 % confidently wrong, oscillation 20.5), and the two quiet grafts R1 and R2 stall as soft stationary states (7–12 % committed, 1–2 % confidently wrong, oscillation 4–5, the lowest churn of the round). The memorized field cell is frozen from step 1 (41 % of cells correct and never moving, entropy 0, 58 % of cells confidently wrong) — the wrong-stable class we know from ARC, reproduced on Sudoku by removing the digit orbit; the group-mixer cell commits like the field's and freezes at 49–52 %. One more thing the dynamics settle: R2's trained cycles do propagate faster per step on the puzzles it solves (90 % of cells by step 8 against W0's 82, 99 % by step 16 against 93) while its stalled set is unchanged — per-step depth was never the binding term; the stopping sets are.

### 4.4 Decimation quality at stalls — can any map's confident cells be handed to a propagation decoder?

| map | committed cells (p > .9) as a fraction of non-given | wrong among them | stalled puzzles with every committed cell correct | peeling from givens + committed cells: solved / stuck / contradiction | at p > .99: committed / wrong / solved / contradiction |
|---|---|---|---|---|---|
| W0 | 40 % | 39 % | 10 % | 1 / 42 / 57 | 11 / 17 / 4 / 13 |
| R1 | 7 % | 7 % | 80 % | 0 / 99 / 1 | 2 / 0 / 1 / 0 |
| R2 | 12 % | 10 % | 75 % | 0 / 89 / 11 | 2 / 4 / 0 / 4 |
| R3 | 34 % | 33 % | 27 % | 1 / 56 / 44 | 10 / 15 / 1 / 16 |
| R3 hard-feedback | 47 % | 40 % | 8 % | 0 / 26 / 75 | 13 / 17 / 1 / 18 |
| R4 | 45 % | 42 % | 7 % | 0 / 31 / 69 | 6 / 19 / 2 / 9 |
| R0 / B0 (sportC1) | 37 / 26 % | 35 / 33 % | 20 / 27 % | 1 / 50 / 50 · 0 / 72 / 28 | 5 / 11 / 1 / 7 · 6 / 15 / 0 / 4 |
| X0 (field) | 91 % | 52 % | 0 % | 0 / 0 / 100 | 82 / 51 / 0 / 100 |
| X1 / X2 | 98 / 90 % | 60 / 47 % | 0 / 0 % | 0 / 0 / 100 | 97 / 59 · 80 / 45 |

Peeling from the givens alone solves 0–3.5 % of the stalled puzzles on every map (the stalls are the search-class puzzles). Reading: at a stopping set nobody's confidence is a usable hard decision. The committed natives are wrong on a third of the cells they are surest about (and hand the propagation decoder a contradiction on half to two-thirds of puzzles); the quiet grafts commit so little that propagation has nothing to work with; the field-class cells commit nearly everything with half of it wrong. R3's hard rows did not change this (33 % wrong at p > .9; 15 % at .99), and the calibration rows say the same in their own units (five most confident cells at stalls: 68–76 % correct at confidence .96–.98 on W0/R3/R4; 82–83 % at .87 on R1/R2, better calibrated only because they commit less; 41–63 % at 1.00 on the field-class cells). The freethink's "calibrated commitment" therefore remains unbuilt: R3 trained hard decisions without the contradiction signal that would make them trustworthy.

### 4.5 The decoder scorecard, and what the gap is in decoder terms

| map | threshold g50 | search yield (rating > 0) | decodable ρ (128 restarts) | selector AUC / spurious | top-5 correct at stalls (gap) | monotone solves | churn at stalls | syndrome oscillation |
|---|---|---|---|---|---|---|---|---|
| W0 | 27.0 | 27.1 | 56.9 | .997 / 0.1 | 70.6 (.27) | 7.9 | 2.27 | 11.5 |
| R1 | 26.6 | 27.5 | 43.8 | .622 / 35.8 | 83.0 (.04) | 2.8 | 0.81 | 5.3 |
| R2 | 27.0 | 24.6 | 42.7 | .977 / 0.7 | 82.0 (.05) | 9.3 | 0.13 | 3.8 |
| R3 | 25.9 | 35.5 | 51.7 | .986 / 0.0 | 76.4 (.19) | 3.4 | 0.22 | 6.9 |
| R4 | 27.0 | 28.6 | 56.1 | .996 / 0.0 | 67.8 (.30) | 4.9 | 1.65 | 9.9 |
| B0c | 26.1 | 34.9 | 53.0 | .997 / 0.0 | — | 6.1 | 0.81 | 8.4 |
| X0 | none | 91.7 | 99.7 | .925 / 7.8 | — | 62.7 | 3.27 | 20.5 |
| X1 | 29.0 | 17.3 | 69.5 | .979 / 1.5 | 40.9 (.59) | 43.1 | 5.54 | 16.3 |
| X2 | 26.1 | 34.3 | 43.6 | .732 / 22.4 | 62.9 (.37) | 42.5 | 1.46 | 9.3 |

In decoder terms the gap to the field is not the threshold's height on propagation-class puzzles (both families solve them) but the search-class yield: 25–36 % against 92 %. Our cell behaves like a soft iterative decoder that halts at its stopping sets — in the language of the coding literature, a decoder stuck at its belief-propagation threshold while the field's behaves like a decimating (guess-and-revise) decoder that reaches the maximum-likelihood threshold on this channel. The four grafts left that class unchanged: the carry made the decoder quieter and more deterministic (fewer decodable puzzles), the inner cycles made it faster within its class, the hard rows made it fit faster without making its guesses trustworthy, and weight decay changed when it memorizes. What would change the class is the decision rule at stopping sets — a decimation step trained against the verifier's contradiction signal (the freethink's unbuilt form), or the field's loop — and the lens now gives that rule its instrument: the E3 table (fraction of confident cells that are wrong, and whether propagation from them solves, stalls or contradicts).

## 5. Physics notes (descriptive)

1. **Regularization as a rate.** At matched steps the wd-1.0 twin (W0) keeps the free-stream flux at 23–37k nats where the memorizing twin B0 inflates to 160–220k, keeps η at .56–.65 where B0 rides to .98, and leaves the codebook uncommitted (rule_H 1.9 at the end vs 0.0) — the S2 regularization-side instance again — but the price is slow learning until the decay rate falls in the floor phase. Curvature (Adam-v): W0 spreads the curvature mass over 4.8 % of its parameters (PR/n 1.5e-3) against B0's 1.3 % (7e-6) — a 200× spread; R3 1.5 % (4.6e-4); the field regime R4 1.9 % (3.0e-4); the memorized X1 6.6 % (6e-3, half of X0's 1.3e-2); X2 22.6 % (broad, still learning).
2. **The carry's fixed points.** R1 is the first native map whose convergence residual does not certify correctness (AUC .62 vs ≥ .98 on every other native): training on self-generated late states creates stationary soft states on unsolved puzzles — a D5 continuum point between our exact contractive solutions and the field's confident churn.
3. **The field regime's plateau.** R4's flat monitor at 100k total steps, CE falling .32 → .13 and val drifting down at the end, is the onset of memorization at the field's decay rate at 38 epochs — the field's own 50k / 768-batch budget (38 segment-passes) sits at the same epoch count; ACT's early halting keeps their per-sample passes ≈3× lower.
4. **Val-selection at 10k resolution** (§0) is the one instrument limitation found; two monitor peaks (W0 54k, R2 54k) were not candidates.
5. **Pace (measured):** two-stage native 80k ≈ 1 h 50 min on a 4-chip worker (30 it/s printed); R2 (K=3) 6 h 24 min (13 it/s); R4 4.0 it/s at batch 384 with remat (16.4 h on the sequential 8 including the resume); X1 41 min (58 it/s); X2 46 min (52 it/s); the canvas sel-5k riders ≈ 1.5 h each sharded.

## 6. Instrument and ops notes (each a lesson or a fix)

1. **The one-checkpoint INTEGRITY gate and the CLEAN split worked as designed:** every vsel eval consistent; the memorization reading (CE) separated from stability; R3's val-selected grid admissible although its final collapsed.
2. **Noise definitions must not assume the effect is small:** CNC2 = |W0 − B0| was meant as a matched-pair noise floor and became the wd effect (8.2pp); the next analyzer takes the noise from a seed pair or a fixed floor, never from a contrast that is itself a treatment.
3. **Selection cadence:** bank grids at the monitor cadence (or monitor at 5k too); the 5k / 15k / 25k grids are dead weight for selection today.
4. **The riders closed the sportC1 provenance gap:** B0/B1 correct-grid scans (42.76 / 42.45 cold, b1 42.75 / 42.31, verified@128 52.97 / 52.81 — the strat-512 estimates of last round, 45.1 / 45.5 b1, were 2.4–3.2pp high, a strat-vs-20k offset) and B0's EMA full on the correct grid (45.42). The canvas riders landed the EqR-statistic rows: C3X t1r@128 88.16 / verified 89.20 / majority 15.76; D4 58.44 / 80.50 / 20.42 (n=5,000; the same checkpoints' 20k scans agree exactly on the 256-puzzle overlap).
5. **Ops:** ~41 h wall, 8 incarnations, 3 preemptions, two Mumbai stints; the live 5-minute bank + fixed restore cost ≤ 5 min per switch (R4 resumed at 10,000; R2/R3 stage A at 29k/35k); the DMS/deadline invariant and the Mac-on-battery check are standing rules; spend 1.3–1.8× the band — the band assumed a 16-node night and the night became a sequential 8.
6. **Owed ($0):** MAC-count instrument; the spend report's zone-aware rates; the grid-cadence fix; promote the decoder lens and the physics pass to the analyzer template as standing outputs.

## 7. Consequences (PI decisions; nothing launched; fleet zero; deadline knob moot)

1. **What the night settles for the paper.** (a) The champion single-prediction number by rule is **43.57 % (R3, val-selected)**, 44.65 with EMA; the labeled best under any convention stays 45.65 (B1 EMA). The ladder reads 21.2 → 25.3 → 33.5 → 37.35 → 42.8 → 43.6 (by rule). (b) The regime question is closed both ways (§2.5). (c) The two field-side results are frontier content: digit augmentation is worth 65 points to the field's recipe and our symmetry supplies it exactly; the all-to-all mixer is load-bearing in their loop. (d) Each graft's failure carries its mechanism (carry → narrower, more deterministic maps with stationary wrong states; cycles → no depth; hard rows → a fitting accelerator, not calibration; wd → a rate, not a coefficient).
2. **sportC3 as registered ("parity at d160 with the surviving levers") has no surviving lever by the mechanical rule.** The evidence-based options, each one night:
   - **(A) Seed-and-rate night at d128 (recommended; ≈ $100–150 on a v6e-8 sequential, 12–16 h):** R3's recipe ×2 more seeds (a claim-bearing 43–45 by the measurement law), R3 + the matched decay rate (wd 0.1 at lr 1e-3), W0 at the matched rate (the clean one-variable wd test the registration named), and the selection-cadence fix. Expected: a seeded R3 number in [42, 46]; the rate arms decide whether the late collapse can be held off without the 8-point over-regularization tax.
   - **(B) d160 parity night (≈ $150–250, 16–20 h):** the R3 recipe at ws10 (4.66M) ×2 seeds with matched inference compute. Expected 45–50 if the d96 → d128 slope (+5.5) continues; risk: the late collapse at width and a slope that is unmeasured beyond d128.
   - **(C) No more Sudoku nights: the ARC-d96 extension (≈ $60–100) and drafting from Sep 8** on the numbers in hand.
   Recommendation: (A) tonight or tomorrow, then (C); (B) only if (A) reads a seeded ≥ 45. The calendar binds: freeze Sep 10–12, abstract Sep 18, full Sep 25.
3. **Registration hygiene for whichever runs next:** noise from seed pairs or fixed floors (§6.2); the grid cadence fix; the decoder lens and the calibration row as standing outputs; R3's hard rows carry a CE tripwire in the chain (its 40k peak → 50k collapse is inside one stage).
4. **Candidate hypotheses for the ledger (nothing registered here):** the decay-rate law (lr × wd is the memorization lever; predicts W0 at wd 0.1 ≈ B0's peak cold with the peak held ≥ 30k); the carry-creates-stationary-wrong-states reading (test: NI on the carry, or a verifier-driven kick in the loop); hard rows as a fitting accelerator (test: W0 early-stopped at R3's CE vs R3 at matched CE).
