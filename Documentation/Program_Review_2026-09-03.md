# Program Review #3 — 2026-09-03: Where the Program and the Paper Stand After sportC1, Why the Field's Recipe Works (Read Through Our Instruments), the ARC Comparison, and the Road to the Frontier

**Trigger (PI):** "Explain the results in detail … where we are in the program and in terms of the paper … the next steps and the justifications … use our measurement tools and our accumulated knowledge to understand why things work and interpret them clearly … compare it with ARC … advance our architecture to beat the frontier scores using our champion track, in an interpretable way." · **Inputs:** `Report_2026-09-03_Champion_sportC1_Verdict.md` (every number below is cited there or in the lens artifacts), `runs/analysis/sportC1_lensG_records_20260903.txt` (lens G1/G2/G5, $0, disk), `runs/analysis/sportC1_lensG_dynamics_20260903.{log,json}` (lens G3, CPU), the complete ledger, the two prior program reviews, the D-catalog (`Sudoku_vs_ARC_Instrument_Map.md`). · **Stance:** every claim is labeled measured / inferred / hypothesis; nothing here is a registered rule.

---

## §1. Where we are — the results, read in detail

### 1.1 What the round was designed to decide, and what it decided
The round asked four questions and answered every one, three of them against the expectation the registration carried.

| Question (registration) | Expected | Measured | Status |
|---|---|---|---|
| Does aug 1000 + the two-phase floor cure memorization at d128? (H-49's test) | yes (60–70 %) | **No.** Both arms that ran to 80k memorized in the hot phase (train CE → 0 by 30–40k; vsel 42.8 → final 23–25); the floor phase from a memorized grid recovered nothing. The field's weight decay (R0: wd 1.0) did cure it (CE .42 at 19 epochs, val still rising). | kill fired; lever found |
| Does normalizing the carried z remove the RI death class? (H-50's bridge) | NORM-WORKS (55 %) | **Yes on the mechanism** (both no-norm twins died at 34,550 and 5,100; both z-norm twins ran 80k clean at η .98 with 0 % explosions at t=64/256 on every grid) — but the rule's CLEAN gate also demands end-CE ≥ .02, so the letter reads NORM-FAILS. | mechanism confirmed; letter fires on memorization |
| Does the field's base recipe reproduce on our stack? (R-C1-7) | REPRODUCED (65 %) | **Yes, to ~1pp:** X0 86.03 @D16 (EqR 84.8), 92.81 @D64 (93.0), b1@D64 92.12 (93.0), verified@128 99.70 (their residual-selected 99.8), EMA +5.2, ACT +13.95 (theirs +8.3). | prerequisite for every additive claim met |
| Is the gap to the field a regime gap or an architecture gap? (R-C1-8; Program Review #2's thesis) | regime explains (55 %) | **Architecture/loop.** At matched data, augmentation, wd, lr, batch and EMA on the identical 20k puzzles: their cell + SOT/ACT loop 92.9 vs our cell 37.1 (only-X0 11,249 / only-R0 94). Our cell in OUR regime at its val-selected grid (42.8) beats our cell in THEIRS at 50k (37.3, undertrained). | Review #2's explanation refuted as the main cause |

### 1.2 The numbers that define our position (all measured; protocol in brackets)
- **Cold single-prediction ladder (full 422,786 test, t=64, val-selected, raw weights):** d16 21.2 → d64 25.3 → d96 canvas 33.5 → d96 native 37.35 → **d128 native 42.83 / 42.71** (B0/B1 at their hot-phase 20k grids; seed spread 0.12pp) — with the labels that the champion-by-rule is **R0 37.33** (the only non-memorizing clean arm; exactly P3s1's 37.35, paired p = .84; undertrained, val +6.25pp over its last 10k) and that B0/B1 memorized AFTER their selected grids. **45.65 with EMA weights on B1's grid** — the program's highest full-test cold under any convention (labeled).
- **The field's own columns, on our stack:** single random-init draw (EqR B=1) X0 92.12 @D64; verified coverage@128 99.70; Top-1-residual@128 92.48; unverified majority@128 88.87; the D16→D64 depth dividend +6.78pp with zero regressions (28,666 / 0). Ours in the same columns: b1 36.8–45.5, verified@128 50–56, t1r ≈ verified.
- **Coverage column (ours, labeled protocol):** R0 ∪ C3X ∪ D4 verified@128 = 96.17 % (B-M3 by portfolio); B0 ∪ B1 ∪ R0 72.6; + canvas 97.7; X0 alone 99.7.
- **Stability:** every native arm retfm 1.00 at every monitor (FPA, fourth regime); the attention dose closed on all five natives; z-norm 2/2 survived vs 0/2 without; explosion census 0 % everywhere.
- **Spend/calendar:** $215 for the round (band $160–185; weather), program ≈ $2.6k of $3.7k; today Sep 3; freeze for champion numbers ~Sep 10–12; ICLR abstract Sep 18, full Sep 25.

### 1.3 The laws after the round (what strengthened, what was scoped, what fired)
- **H-49 memorization (FIRED, sharpened):** memorization is not corpus-size-bound when the augmentation group is learnable — the memorization step is the same at aug 100/d96 and aug 1000/d128; the lever is weight decay (measured on R0), with lr/batch confounded inside R0; ACT on the field cell acts as an anti-memorization device (X0n memorizes, X0 does not).
- **H-50 z-channel instability (CONFIRMED):** the death class lives in the free z recurrence; normalization removes it at η → .98 at no per-draw cost; a second, early low-η flavor exists (A1 @5,100, η .66).
- **H-37 init-invariance + the selector law (SHARPENED):** every RI+FPA map is init-invariant with ≤ 0.1 % spurious attractors → residual selection ≈ verification; the field's RI-free base has 3–5 % spurious attractors per wrong draw → its selector caps at 92.75 % of hits (lens G1, §3).
- **H-45 FPA, H-48 dose, H-46 drift:** hold (4th regime; closed channel at every death; the only vsel≠final cases are memorization).
- **The cold ladder is monotone through d128** at val-selected grids; the funnel (reachability) is NOT — it is flat at ρ ≈ .5 for our cell at every d128 checkpoint (§3).

---

## §2. What this means for the paper

**The object of paper 1 is unchanged** (the physics-informed architecture, the equilibrium core with basin objectives, exact S9, the measured-in mechanisms, the measure → understand → improve loop, the Sudoku-vs-ARC catalog). What the round changes is the **performance section and the positioning**, in three ways.

1. **The comparator numbers are now ours to cite, with columns.** We reproduced EqR's base (86/93/99.7) on our stack; the table can carry protocol (best-test vs val-selected), training-regime (aug/wd/lr/batch/EMA/normalization) and inference-compute (equivalent layers: X0 672 at D16, 2,688 at D64; our native ≈ 5–6 layer-equivalents per step at d128 → ≈ 350 at t=64, inferred from layer counts × width²; a measured MAC count is owed) columns — the reviewer's "your 37 vs their 87" objection is answered by measurement, not by argument.
2. **The honest performance sentence.** Our cell reaches 42.8 % val-selected (45.7 with EMA) at 3.0M params and ~30–100× less inference compute per puzzle (inferred) than the field's 93 %; the gap is the cell and its training loop, not the optimizer regime (§1.1 row 4). The frontier-class single-prediction score is not within our cell's reach at this compute, and the paper says so with the decomposition as the reason. What the paper CAN claim as frontier-adjacent: (a) the reproduction itself (the first independent reproduction of EqR's base on a different codebase, with instruments), (b) the coverage column (our verified breadth 96–100 % by portfolio; X0's own 99.7 with a free verifier — i.e., EqR's 99.8 residual-selection result is a selection result the verifier makes unnecessary), (c) the mechanism account of why their recipe works (§3) and why ours dies/memorizes without their conventions — this is the "field recipe read through our instruments" section, which no series paper has.
3. **The laws carry their regime scope, as Review #2 required, but the scope is now measured rather than conceded:** memorization (H-49) and the RI death class (H-50) are pathologies of our regime that the field's conventions (wd 1.0, EMA, ACT, normalization) prevent — and we measured each convention doing exactly that on our stack (R0 for wd; X0/X0n for ACT; the z-norm twins for normalization; the EMA rows). The sentence "our laws explain the field's conventions from first principles" is now backed by one-variable measurements on our own cell, not only by literature.

**What we cannot say** (and the adversarial reviewer will check): that the physics-informed architecture is competitive on Sudoku single-prediction accuracy; that pricing/tolls buy accuracy on CSP (they buy stability and compression); that verified breadth is comparable to residual-selected breadth (it is a different, stronger statistic — measured on the same map, §3.3).

**Calendar consequence.** The abstract (Sep 18) can be written on today's numbers. The champion section moves only if a night of the §5 grafts lifts reachability; the "field recipe under our instruments" section is complete in its first form today and gains the additive ledger if that night runs.

---

## §3. Why the field's recipe works — read through our instruments (lens G; descriptive, from banked records + one CPU pass)

Three record-level lenses and one dynamics lens were run on the reproduced field model (X0) and, side by side, on our best maps (R0, B0, A0, the pilot's P3s1) and the canvas references. Every quantity is defined in the ledger's instrument suite; nothing new was invented.

### 3.1 Reachability is the binding difference (funnel ρ, lens G5 + physics pass §F)
The (ρ, r) funnel model — ρ = fraction of puzzles reachable from ANY of 128 random inits, r = per-draw hit rate on reachable puzzles — reads:

| map | ρ (all) | ρ (hardest quartile) | r (hard octiles) | cold | verified@128 |
|---|---|---|---|---|---|
| X0 @15k / @35k / @50k | .84 / .98 / .98 | .84 / .98 / .98 | .21 → .33 → .57 | 67 → 86 → 87.5 (strat) | 86.5 → 99.0 → 99.6 |
| R0 @15k / @35k / @50k | .24 / .38 / .48 | .02 / .20 / .40 | .16–.23 (50k) | 24 → 31 → 36 | 24 → 38 → 51 |
| B0 @A:20k (vsel) / @50k / @B+15k | .5 / .52 / .5 | .46–.50 | .10–.17 | 44 → 31 → 29 | 55 → 55 → 52 |
| P3s1 d96 | .36–.44 (hard) | | .18–.23 | 37.4 | 52.9 |
| canvas C3X | .70–.82 (hard) | | .03–.06 | 24.6 | 88.9 |

Reading: the field's map reaches essentially every puzzle from any init by 35k steps and gets faster per draw thereafter; our maps plateau at ρ ≈ .5 (B0 at every checkpoint, A0 .54, P3s1 ≈ .4) — cold ≈ b1 ≈ ρ × (draw saturation), so **half the test set is unreachable for our cell from any start**, and no amount of attempts (verified@128 saturates at 50–56) or cold refinement changes that. R0 (our cell in the field regime) is the one native arm whose ρ is still rising at 50k (.24 → .48): in the non-memorizing regime, reachability is training-limited; in ours it is architecture-limited (B0's ρ never moved while its cold went 44 → 31). The canvas C3X reaches ρ .8 with r .04 (wide-slow): breadth ownership and cold ownership dissociate exactly as the D-catalog says (D9), and the field's map is the first we have measured that is wide AND fast.

### 3.2 Propagation depth: the field solves in one or two outer steps; ours propagate (lens G2 + G3)
`first_exact` (the outer step at which the cold trajectory first equals the solution), medians on the full test:

| map | median first_exact by rating octile o2…o8 | p90 (all solved) |
|---|---|---|
| X0 @D64 | 1 2 2 2 2 1 2 | 11 |
| X0 @D16 | 1 1 1 1 1 1 1 | 6 |
| R0 | 10 15 15 14 14 14 15 | 32 |
| B0 (vsel) | 8 11 10 10 10 10 11 | 24 |
| P3s1 d96 | 8 11 9 9 9 9 10 | 20 |
| canvas D4 / C3X | 7–10 / 7–33 | 16 / 33 |

The field's 42-layer outer step (2 blocks × 3 H-cycles × 6 L-cycles + 3) propagates most puzzles to the solution within one or two steps at every difficulty; the +6.78pp it gains from D16 → D64 is a long tail (the puzzles solved only at D64 have median first_exact 28, p90 52 — genuine long propagation, with zero regressions). Our shallow steps (≈ 5–6 layer-equivalents at d128) need 8–15 outer steps at median and 22–32 at p90: our maps are propagation engines, theirs is closer to one-shot with a propagation tail. The dynamics lens (G3, a rating-stratified 84-puzzle cold run; `sportC1_lensG_dynamics_20260903.json`) reads the curves directly — fraction solved after outer step 1 / 2 / 4 / 8 / 16 / 32 / 64:

| map | 1 | 2 | 4 | 8 | 16 | 32 | 64 | residual at t=64, solved / unsolved | Jacobian radius λ at t=64, solved / unsolved |
|---|---|---|---|---|---|---|---|---|---|
| X0 (field) | 20 | 50 | 70 | 77 | 81 | 87 | 94 | .009 / .27 | 1.07 / 8.7 (finite-difference estimate; only 21 % of endpoints below 1) |
| R0 (ours, field regime) | 0 | 0 | 2 | 4 | 25 | 33 | 37 | 5e-10 / 3e-3 | .73 / 1.05 |
| B0 (ours, vsel) | 0 | 1 | 2 | 14 | 33 | 38 | 40 | 2e-12 / 1e-3 | .67 / .95 |

Half of the field's solves are complete after two outer steps; ours need eight to sixteen. On our side the fixed-point structure is exact: solved endpoints are fixed points to 1e-10 with a contractive Jacobian (λ .67–.73), and unsolved endpoints are NOT fixed points (residual 1e-3, λ ≈ 1, still moving) — Sudoku failures on our cell are unfinished propagation, never capture by a wrong attractor (0/103 converged-wrong on this subsample). The field's unsolved endpoints also keep moving (residual .27 vs .009 solved; 0/5 converged-wrong here) — the spurious attractors of §3.3 live among the random-init draws, not on the cold path. **The field's solved endpoints are not contractive fixed points in the latent:** autodiff through the cell returned NaN, so the Jacobian radius was estimated by finite-difference power iteration of the latent step with the readout input held at its start value (8 iterations, drift ≈ .03; indicative on a non-smooth map — the deployed step also re-embeds the current answer, which moves solve rates by ~3pp on this subsample): median 1.07 on solved endpoints (p10 .96, p90 1.4), 5.6–8.7 on unsolved ones, with the solved residual flat at .009 from t=16 to t=64 rather than decaying. The field's "stabilized attractor" is a **readout-stable, latent-drifting** state (the answer's argmax is stable while the latent keeps moving at neutral gain), whereas our basin objective produces exact contractive fixed points (residual → 1e-10, λ .7). This is the Ren & Liu "fixed-point violation" measured on the reproduced EqR base — and it is the mechanism behind §3.3: a neutral-gain latent can settle into a converged-wrong state after a random init, which is what RI/NI training must remove for their residual selector to work; our contractive fixed points cannot.

### 3.3 Attractor structure: the one instrument where our maps read cleaner (lens G1)
Per draw (20k puzzles × 128 random-init draws), the convergence residual EqR selects on, versus correctness:

| map | P(resid correct < resid wrong) | spurious-attractor rate per wrong draw (o3–o8) | hittable puzzles whose min-residual draw is correct | t1r@128 / verified@128 |
|---|---|---|---|---|
| X0 (field, no RI/NI) | .925 | 3–6 % (54 % on the easiest nonzero octile) | 92.75 % | 92.48 / 99.70 |
| R0, A0, P3s1, B0† (ours, RI+FPA) | .994–.999 | 0.0–0.1 % | 98.6–99.7 % | .975–.997 of verified |
| P1 (d96, no RI, memorized) | .950 | 3–16 % | 81.9 % | 20.3 / 36.6 |

Reading: on the field's base map a few percent of wrong draws converge to spurious fixed points (residual as low as correct draws), so with 128 draws the minimum-residual pick is wrong 7 % of the time and t1r@k falls with k (95.5 @8 → 92.5 @128). EqR's 99.8 requires RI/NI training to remove those spurious attractors; with a free verifier the base already covers 99.7. Our RI+FPA maps have essentially no spurious attractors (the solved state is an exact fixed point, every stuck state keeps moving — §3.2) — which is why residual selection ≈ verification on our maps without NI. Memorized maps (P1) grow spurious attractors (3–16 %): the D5 "wrong-stable" texture is a memorization readout on Sudoku, as the pilot said. The failure texture otherwise coincides: stuck partial propagations with ~50/81 cells correct and 30+ violations on X0 and on our healthy maps alike (lens G2; the verdict's first draft mis-read a mean over all puzzles — corrected).

### 3.4 Training dynamics: what each of the field's conventions does (one variable each, measured on our stack)
- **Weight decay 1.0** (R0 vs our regime): no memorization at 19 epochs (CE .42 vs 0.000), 5× less free stream flux (30k vs 104–223k nats — the S2 "excess free flux is memorization" statement in its regularization form), codebook uncommitted (rule_H .27 vs 0), η held at .64 (never the replacement dynamics), λ_J benign; cost: slow (val still rising at 50k, ρ still rising).
- **EMA .999:** +5.2pp on the field cell, +2–3pp on ours (paired, full test); the field's number includes it.
- **ACT:** +13.95pp at D16 / +16.1 at D64 on the field cell; the no-ACT loop memorizes (train_exact .999 at 38 segment-passes) while the ACT loop never drives its segment CE below .55 — ACT replaces halted rows with fresh samples (≈ 3× fewer passes per sample) and its halting head changes what is learned; the gain exceeds anti-memorization alone (labeled hypothesis; test = X0n early-stopped vs X0 at matched passes).
- **Normalization inside the cell** (z-norm on ours): removes the RI death class; the field cell (post-norm RMSNorm everywhere) has no such class to remove.
- **The SOT online loop** (persistent carry, one segment per optimizer step, rows replaced when done): the field's map is trained on its OWN trajectory states for up to 16 segments × 21 passes — a basin-widening objective on the natural state distribution (our FPA is the same idea on ε-corrupted solutions; RI rows are the same idea at t=0). This is the untested candidate for the reachability gap (§5, graft 1).
- **Digit augmentation** on a non-equivariant cell: their model trains on 9× relabeled copies; our S9 gives the orbit by construction. Untested on their cell (§5, ledger arm X1).

### 3.5 The mechanism story in one paragraph (labeled inference)
The field's recipe works because (i) each outer step is a deep, normalized, globally-mixing computation that carries most puzzles to the solution in one pass (§3.2), (ii) the SOT loop trains the map on the states it actually visits, so almost every puzzle sits in the solution's basin from any start (§3.1), (iii) weight decay + EMA + ACT keep the 1k-puzzle corpus from being memorized (§3.4), and (iv) the residual selector is a post-hoc device that works only after RI/NI removes the few-percent spurious attractors the base leaves behind (§3.3). Our cell has (iii) partly (val-selection, z-norm, FPA) and beats them on (iv) (no spurious attractors), but lacks (i) and, probably, (ii): reachability ρ ≈ .5 is our wall.

---

## §4. Sudoku vs ARC — what the round adds to the catalog

**Same wall, different heights.** The binding limit in both domains is reachability, not precision: on ARC the equilibrium maps hold basins for only ~29–34 % of pairs and no iteration/temperature/attempt helps where none exists (D4); on Sudoku our cell's basin covers ~50 % of puzzles from any init (§3.1), and the field's cell covers 98 %. The wall is therefore a property of the cell and its training loop, not of the domain — the first cross-domain statement of D4 with a counterexample on the Sudoku side.

**Spurious attractors form a continuum measurable with one instrument (converged-wrong rate).** ARC: 92 % of converged limits are wrong-stable (E3). Sudoku: the memorized native map 3–16 % of wrong draws, the field's base 3–6 %, our RI+FPA maps ≤ 0.1 %. The E3 battery's Sudoku form is the per-draw residual-vs-correctness table (§3.3) — it now exists on both domains and reads the same object. Consequence for ARC's conversion stack: verification-free residual selection (EqR's) is useless on a 92 %-wrong-stable landscape and works on Sudoku only after RI/NI cleans the base — ARC's selection problem is an inventory problem (no basin for the right rule), which no init trick removes; our basin-snap/vote stack remains the right tool there.

**One-shot vs propagation.** ARC maps solve at step 0–1 or never (the horizon probe: all first-exact at steps 0–1; +0 solves at t=96); the field's Sudoku map solves mostly in 1–2 outer steps with a genuine propagation tail (D64 +6.8pp); our Sudoku maps propagate over 8–15 shallow steps. "Depth = propagation" (D9) is true of OUR Sudoku maps and of the field's tail, false for ARC and for most of the field's solves — the per-step computation decides which regime a map is in.

**Memorization vs transfer erosion (D11/D12).** Sudoku memorization (train CE → 0, test collapse) and ARC's optimization-length transfer tax (H-40: unseen-family basins erode while trained families sharpen) look like one pressure with two observables. The Sudoku round measured the lever (weight decay); ARC's runs were never made with wd 1.0. Prediction for the ARC extension: an A5-class d96 arm at wd 1.0 shows a reduced or absent budget tax at long optimization (H-40's mechanism = memorization in ARC clothing) — a cheap, registrable arm on the planned ARC-d96 rung.

**What transfers from the field's loop to ARC conversion (hypotheses):** the SOT-carry (train on self-generated states) is a basin-widening objective compatible with the E10 core (FPA is our version on corrupted solutions); ACT-style halting is an anti-memorization device where the corpus is small; EMA is free. What does not transfer: the residual selector (D3); digit-style augmentation (ARC's orbit is D₄ × palette, already used).

---

## §5. Advancing the architecture toward the frontier, interpretably (the champion track after this round)

The three measured deficits map to three grafts, and each graft has an instrument that reads its effect before any headline number moves:

| deficit (measured) | graft (one variable from R0 or B0) | instrument that reads it | prediction (pre-registrable) |
|---|---|---|---|
| reachability ρ ≈ .5 (§3.1) | **SOT-carry training on our cell** (persistent (y, z) carry, one T=16 segment per optimizer step, rows replaced when solved or after 16 segments; deep supervision per segment) — build ≈ 1 day (the trainer's `--sot` exists for the field cell) | ρ per checkpoint (screens k256), first_exact, spurious rate | ρ rises above .7 by 50k; spurious rate stays ≤ .1 %; retfm 1.00 |
| per-step computation (§3.2) | **inner cycles**: K passes of the cell with the carried z per outer step before the y update (their L-cycles) — build ≈ 0.5 day | first_exact (median ↓), ρ, per-step cost | first_exact median halves at K=3; cold +5–10pp at matched wall |
| undertrained in the safe regime (§1.1) | **R0 @100k+** and **B0 + wd 1.0** (our lr/schedule; one variable) | CE, val peak step, ρ trajectory | wd stops the memorization (CE plateau ≥ .05) and lets the hot phase run past 20k → cold > 42.8 |
| (their side) what our geometry adds | **X0 with our factorized group mixer as the token mixer** (the graft in the other direction) | cold, ρ, first_exact on their loop | small effect (the constraint graph = the 27 groups; all-to-all adds 2-hop shortcuts) |
| (their side) is digit augmentation load-bearing for them? | **X0 − digit aug** | cold, per-draw | a drop of several pp — the value of exact S9 in their units |

Our ingredients that stay in the champion on the round's evidence: z-norm (the death class is gone at η .98), FPA (fixed points exact; no spurious attractors), exact S9 (free orbit), RI (init-invariance; possibly redundant once SOT-carry is in), the attention toll (stability at zero cost), val-selection + EMA (the anti-memorization instrument + a free +2–3pp), the two-phase floor (protective only when the hot phase ends before CE → 0 — under wd 1.0 this becomes testable again).

Honest expectation (inferred from the funnel arithmetic): cold ≈ ρ × draw-saturation on our maps; lifting ρ from .5 to .8–.9 with r unchanged would put cold in the 60–70 range; matching the field's 93 needs both ρ ≈ .98 and r ≈ .5+ — i.e. their per-step depth. The grafts are the way to find out which of the two our cell can buy at ≤ 5M params, and every intermediate result is a mechanism statement the paper can use whether or not the headline moves.

---

## §6. Next steps, with justifications (PI decisions)

1. **Night 1 (build Sep 4, launch Sep 4 evening / Sep 5; ≈ $150–220 on a v6e-16 4×4, ≈ $120–170 on an 8 over a longer night): the two-directional graft ledger.** Arms: X1 = X0 − digit aug · X2 = X0 + our group mixer · R1 = R0 + SOT-carry · R2 = R0 + inner cycles K=3 · R3 = R0 @100k (continuation) · W0 = B0 + wd 1.0 · + the owed riders (B0/B1 20k scans and B0's EMA full on the A:20k grids; canvas C3X/D4 EqR-statistic evals). *Justification:* each arm isolates one of the three measured deficits (§5) or one of the field's conventions (§3.4) with an instrument that reads the mechanism; the reproduction (X0) is the control every arm hangs on; the riders close the only integrity gap of the round. Registration hygiene from the verdict applies (grid-consistency gate; CLEAN split into stability vs memorization; harness scenario per resume path).
2. **Night 2 (Sep 6–7, ≈ $80–120, if night 1's ρ moves): the champion assembly** — the surviving grafts combined on our cell in the non-memorizing regime, ×2 seeds, the full battery. *Justification:* the champion by rule must be a clean arm; a combined recipe needs its own registration and seeds.
3. **ARC-d96 extension (Sep 7–8, ≈ $60–100)** with one added arm: A5-class at wd 1.0 (the H-40 = memorization test, §4). *Justification:* scale-matching for the catalog was already owed; the wd arm converts the Sudoku lesson into an ARC prediction the paper can report.
4. **Paper 1 drafting from Sep 8** on whatever has landed; abstract Sep 18; full Sep 25. The freeze rule (Sep 10–12) stands; anything not landed ships in the AAMAS version.
5. **$0 items this week:** the MAC-count instrument for the inference-compute column; the finite-difference λ on the field cell (running); the residual-vs-correctness table as a standing instrument in the analyzer template; the D-catalog rows above into the Instrument Map.

Budget: nights 1 + 2 + ARC ≈ $290–440 → program lands ≈ $2.9–3.1k of $3.7k, inside every envelope rule; the calendar, not the budget, binds.
