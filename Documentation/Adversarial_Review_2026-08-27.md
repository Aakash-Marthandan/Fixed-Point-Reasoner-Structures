# Adversarial Review — All Standing Assertions (2026-08-27)

**Mandate (PI):** a skeptical pass over every assertion currently steering the program, to counter politeness/confirmation drift before the rung-2 registration hardens. Method: each load-bearing claim is attacked with its strongest alternative explanation or confound; verdicts are SURVIVES / SHARPENED (attack produced a stronger form) / WEAKENED (scope reduced) / LABELED (stands only with its stated caveat). Attacks on the PI's own new conjecture and on the analyst's own recommendations are included deliberately. New $0 data checks run for this review are marked ★.

## A. The rung-1 verdict's claims

**A1. "B2/FPA is the best recipe" — WEAKENED to "B2 is the mechanical winner; the FPA-vs-RI/NI ordering is open."** B2-over-B1 on screens (+4.9 pp) is within the measured funnel-noise floor (7.03); on cold (+1.6 pp) it is at the d16 FPA pair-spread scale (1.66). B1 moreover ran at half the registered learning rate (the labeled retry) — the registered B1 config never completed at d64. What IS established: B2 held stability from the first monitor while B1's registered config diverged — "FPA is the width-scaling stabilizer" survives; "FPA beats RI/NI on accuracy/breadth" does not yet exist as a claim. Consequence: none for rung-2 arms (carrier by rule), but the paper must not write "FPA wins" — it writes "FPA is the stable carrier; the accuracy ordering is within noise."

**A2. "The funnel changed owners at width" — SHARPENED by its strongest attack ★.** Attack: the d16 "narrow T12 funnels" set (26–36 @128) contained no FPA measurement — if FPA was always wide, the flip is recipe-discovery, not width. Check run (★, banked cheap evals, same 512-puzzle strat set, shared k=16 point): FPA-d16 vote@16 = 22.1/21.9 (both seeds) — the *narrowest* of the d16 deep arms (RI/NI 27.0, plain 24.8) — vs FPA-d64 vote@16 = 41.6. Within-recipe, same-k, same-puzzles: width ≈ doubled the anchored funnel. The interaction is real and recipe-dependent (FPA +19.5 pp at k=16, RI/NI +6.4, priced +9.2-then-condensed). Bonus: this retro-explains the registration's wrong winner prediction — the FPA funnel advantage is width-emergent and was invisible in every d16 measurement. Residual confound LABELED: steps (50k vs 20k) still differ between the deep winner and the shallow loser; bounded by the mid-checkpoint (25k) already at 57.3 @128 ≈ 2× every d16 deep funnel.

**A3. "Width raises per-draw rates 4–5× and flattens the difficulty slope" — LABELED (n=1 per cell, sign-consistent across all 8 octiles).** The censored-geometric estimator is noisy on near-zero d16 hard-octile rates; the cross-scale comparison uses the identical estimator and identical 20k puzzles, and the direction is uniform. Rung-2 registration carries a numeric slope prediction to convert this from descriptive to tested.

**A4. "H-45 collapse accelerates with width" — SURVIVES.** d64: n=2 seeds, both broken from the first monitor, val-selection independently catching the early peak; d32 precedent n=1; d16 stable. Three widths, one direction, instrument-level evidence (n=512 retention rows). The 5k-cadence limitation (can't resolve the λ-vs-retention ordering) is a monitoring note, not a threat to the claim.

**A5. "The price prevents the collapse (single-toggle B4/B5)" — SURVIVES, n-LABELED.** The toggle is clean (identical config but β) but B5 is n=1. Supporting mass: priced arms are 8/8 stable across waves 2/3a/rung-1 at three widths. Rung 2's priced arms extend the record; no dedicated replication cell needed yet.

**A6. "H-47: precision and diversity compete for the priced budget" — SHARPENED with a scope correction ★.** Attack 1: is terminal flat-k condensation width-new? No ★ — the d16 priced arm's k-curve was already flat-ish (12.3→15.4 over k=1→16; d64: 22.1→27.0 over k=1→256). Correction folded into H-47's framing: the *terminal condensed state is the priced steady-state at every width measured*; the width-new observable is the **wide mid-training window** (71.3 @256 at 25k) — i.e., width delays the condensation long enough to see the competition happen. Attack 2: is the condensation just "training stopped changing the map"? No — cold doubled (9.4→19.1) across the same window; the map kept improving on precision while diversity died. Attack 3 (the sharpest): the "competition for nats" mechanism is an *interpretation* — the measured throat is training-objective flux, not an inference-time diversity ledger; nothing yet measures diversity *in* nats. LABELED as mechanism-reading; the β-ladder cell is the discriminator either way, and an inference-time flux-vs-draw-diversity measurement is noted as a future instrument idea.

**A7. "Cold-wander is d16-finite-size" — SURVIVES with a resolution label.** The d64 evidence: val-selected ≈ final on all three healthy deep arms with full-test-grade deltas (≤0.06 pp). The 64-puzzle monitor's ±5 pp noise means small wander below that resolution is not excluded; the claim is "the d16-magnitude (15 pp) chaos is absent," not "training is perfectly monotone."

**A8. "Law 1 extends: task-set-constant throat across 12× params" — SURVIVES, n-LABELED (1–2 per point).** Consistent with the seeded ARC version of the same law; the Sudoku points are directional. S2's free-flux inflation at width: same label.

**A9. "Fisher migration eq→readout" — LABELED exploratory.** First use; block attribution rides parameter-name parsing; no baseline for the participation ratio. No verdict or decision rides it; keep descriptive until it earns a registered prediction.

**A10. The G-B3/G-B4 rule-letter reinterpretations — PROCEDURALLY CLEAN.** The registered verdicts stand as issued; the mid-screen data that reinterprets them came from registered instruments, and the reinterpretation is flagged as post-hoc reading, feeding new registrations (H-47) rather than editing old ones. This is the append-only discipline working as intended.

## B. Standing laws feeding the rung-2 design

**B1. Breadth vote@k "log-linear" — CORRECTED phrasing.** At d64 the curve is *convex* in log-k (accelerating per doubling: +1.9→+9.7), not flat log-linear as at d16. The operative claim — unsaturated at k=128 — survives; the paper should not write "log-linear" for d64.

**B2. "FPA widens basins" — stands at d16; UNVERIFIABLE at d64 until the ladder extends** (saturation). The rung-2 extended rungs re-open the measurement; until then the width-basin claim cites d16 only.

**B3. The landscape-class law, the two-sided iteration result, the two throats, the three profiles — SURVIVE** (seeded or multi-instrument; no new attack found beyond the recorded caveats).

**B4. Night-window stability — HEURISTIC, not science.** Supported by the whole campaign's churn record but unquantified; treated as a scheduling policy with zero evidentiary weight.

## C. The PI's new conjecture, attacked before registration

**Claim:** toll effects correlate positively with scale — the precision/diversity competition disappears as the network "carries more information through the toll"; an optimal (toll, scale) point may exist.

**Attack (the strongest one available):** at fixed toll, the measured throat does **not** grow with scale — it is constant-to-declining (Law 1, both domains; d64 Sudoku ≈ 967 nats ≈ d16's). So the literal mechanism "more information through the toll at scale" contradicts our own strongest law *at fixed β*.

**Steel-man (two survivable forms):** (i) **code-efficiency form** — at fixed β and fixed nats, capacity improves the code, so the same budget encodes more usable structure; the competition eases with scale without the throat moving. (ii) **knee-shift form** — the optimal β falls with capacity (the rate–distortion knee moves), so a *re-tuned* toll admits an effectively richer code at scale; the optimum the PI conjectures is a point on the (β, d) surface.

**Registered as falsifiable predictions (rung 2):** **P-A (scale axis, fixed β):** the d96 priced arm's condensation attenuates vs d64 — screen-vb @256 > 27.0 (d64's) with a non-flat k-curve at ≥3 checkpoints. **P-B (budget axis, fixed scale):** the β/3 arm at d96 holds a wide funnel through late training. Outcomes: P-A alone → code-efficiency form; P-B alone → the competition is budget-mediated (H-47 as first registered); both → the conjecture's strong form, and the (β, d) optimum becomes a mapped object (four priced corners exist after rung 2); neither → priced breadth is dead at these scales and the toll's role stays stability/compression. Either way the conjecture leaves the conversation as a measurement, not a mood.

## D. The analyst's own recommendations, attacked

**D1. The T6+FPA insurance arm — challenged and retained.** Alternative use of the slot: a third carrier seed (statistics) or an inference-depth/steps cell. Retained because: the goal metric is breadth; the funnel question at d128 is whether a *cheap-training* wide-funnel recipe exists (T6@20k trains ~2.5× cheaper than T12@50k — real money at d128); and rung 2's carrier pair already supplies the funnel-noise measurement the third seed would buy. The alternative was weighed, not ignored.

**D2. Dropping the random-start recipe — retained with the honest label from A1:** it is dropped for *stability and redundancy* reasons, not because it measurably lost.

**D3. The report's own prediction scoreboard (4 hits / 6 misses) — verified as stated;** no miss was softened in the writeup. The G-B1 "FLAT" call and the insurance-contingency condition firing are reported against interest of the width narrative.

## E. Consequences folded into the rung-2 registration

1. H-47's framing corrected (terminal condensation = priced steady-state; the mid-training window is the width-new observable) — the registration's screens-at-≥3-checkpoints are the right instrument either way.
2. PI-conjecture predictions P-A/P-B registered verbatim (§C).
3. A numeric slope prediction at d96 (converts A3 from descriptive to tested).
4. Paper-language constraints: "mechanical winner," not "best recipe" (A1); no "log-linear" at d64 (B1); FPA basin-widening cited at d16 pending ladder extension (B2).
5. The diversity-in-nats instrument idea noted (A6) — unbuilt, no slot this rung.
