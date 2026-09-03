# sportC2 — the pre-parity graft night at d128 (registration draft; locks in `tools/analyze_sportC2.py --selftest` before launch)

**Date:** 2026-09-04 · **Status:** BUILD IN PROGRESS → registration → launch on the explicit PI go · **PI decisions already taken (2026-09-03 evening):** the arm set as tabled in the freethink is APPROVED; the ARC comparison builds/rung are DEFERRED until after the Sudoku champion work (sportC2 → sportC3), carrying everything learned. · **Inputs:** `Research_Brainstorm.md` "Freethink 2026-09-03" (X-4 the amalgamation principle, X-7 the design, §0 the nine groundings), `Report_2026-09-03_Champion_sportC1_Verdict.md`, `Program_Review_2026-09-03.md`.

## §1. Purpose
Test, one variable at a time and with an instrument that reads the mechanism, the four levers the freethink derived from the measured deficits of our cell — reachability (ρ ≈ .5 vs the field's .98), per-step propagation (8–15 outer steps vs 1–2), calibrated commitment (top-5 confident cells at stalls only 70–72 % correct at p .96), and the memorization regime (wd 1e-4 collapses curvature and erases the low-givens capability) — plus the two additive reads on the field's cell (its digit orbit; our geometry on its loop) and the undertrained read on R0, BEFORE the parity run commits a recipe at d160.

## §2. Arms (d128 ws8 native9 unless X; every measured champion ingredient kept: z-norm, FPA k4 ε.2, β_nl 1e-6, exact S9, aug 1000, RI .5, val-selection, EMA row)

| arm | one variable (from its base) | base | steps / regime | worker |
|---|---|---|---|---|
| **W0** | `--wd 1.0` | sportC1 B0 (our regime: lr 1e-3, B64, two-phase 50k cosine + 30k floor) | 80k, two-stage | w0 |
| **R1** | `--sot --sot-segments 4` (the persistent (y, z) carry; rows replaced when EXACT or after 4 segments = t 64) | W0 | 80k, two-stage | w1 |
| **R2** | `--inner-k 3` (3 latent passes per readout, trained; evaluator/census/monitor/calib mirror it) | W0 | 80k, two-stage (≈ 3× per-step cost) | w0 |
| **R3** | `--hard-p 0.5` (per outer step, the feedback is the argmax one-hot with a straight-through gradient; deep supervision at later steps IS the revision signal) | W0 | 80k, two-stage | w1 |
| **R4** | `--init-from` sportC1 R0's 50k ckpt, +50k steps (fresh optimizer, EMA restarted — labeled) | sportC1 R0 (field regime: wd 1.0, lr 1e-4 const, batch 384, remat) | 50k continuation | w2 |
| **X1** | no `--sudoku-digit-aug` | sportC1 X0 (TRM cell + SOT + ACT, field regime) | 50k | w2 |
| **X2** | `--trm-token-mixer group9` (our group mixer on a 64-dim projection replaces the token-mixing MLP; prefix tokens get a pooled cell summary; params ≈ 5.9M, labeled) | X0 | 50k | w3 |
| riders | B0/B1 20k scans + B0's EMA full on the CORRECT A:20k grids (the sportC1 provenance gap); canvas C3X/D4 EqR-statistic evals (owed since the pilot); `stall_calibration` on every arm's vsel grid (standing) | — | — | w3 |

**Deviation from the freethink table, stated:** R1/R2/R3 are grafted on the W0 base (our lr/schedule with wd 1.0) rather than on R0's field regime — the field regime at batch 384 puts an 80k K=3 arm at ~21 h on a 4-chip worker (infeasible in one night) and a common fast base makes the three grafts comparable; R4 keeps the field-regime read. If W0 memorizes anyway, the grafts read at their val-selected grids (the sportC1 convention) and R-C2-0 says so.

## §3. Instruments per arm (all standing; the three freethink lenses run at analysis time from records)
Screens strat-512 k256 at {stage-A end, B+5k, B+15k, vb} (natives) / {15k, 35k, vb} (R4, X); retfm; full tests at vsel AND final (+ the alternate-weights row); the 20k k128 scan at vsel (b1, verified, t1r, per-draw records); explosion census on vsel AND final (t64 + t256); **stall calibration** (`tools/stall_calibration.py`: top-5 correctness / confidence / entropy on stalled puzzles, strat-512) on the vsel grid; at analysis: reachability-vs-givens, the flip spectrum, curvature concentration.

## §4. Decision rules (locked verbatim in `tools/analyze_sportC2.py`; selftest green before launch)
INTEGRITY adds the sportC1 lesson: every vsel-labeled eval of an arm on ONE checkpoint path (breach withholds). STABILITY (admission) = not STOPPED ∧ retfm ≥ .9 ∧ census ≤ .02; MEMORIZATION (end CE < .02) is reported separately and never disqualifies the val-selected grid. CNC2 = max(|vcold(W0) − .4283|, .01). R-C2-0 REGIME (W0): CE ≥ .05 ∧ val-peak ≥ 25k ∧ vcold ≥ .4283 + CNC2 → WD-HOLDS / CE < .02 → WD-MEMORIZES / else WD-PARTIAL. R-C2-1 CARRY (R1): reach (verified@128) ≥ .70 WIDENS / ≤ .55 FLAT / PARTIAL; + SELECTOR-INTACT iff t1r/verified ≥ .97. R-C2-2 DEPTH (R2): median first_exact ≤ ½ W0's → PROPAGATES; reach at 21–25 givens ≥ .50 → REACHES; neither → FLAT. R-C2-3 COMMIT (R3): calib top-5 on stalled ≥ .90 CALIBRATED / ≤ .75 FLAT / PARTIAL; + SPURIOUS-APPEAR iff t1r/verified < .97. R-C2-4 (R4): vcold − .3733 ≥ .03 CLIMBS / ≥ .01 SLOW / FLAT. R-C2-5 (X1): .8603 − cold@D16 ≥ .03 ORBIT-LOAD-BEARING / ≤ .01 ORBIT-LEARNED / PARTIAL. R-C2-6 (X2): cold@D16 − .8603 ≥ .02 HELPS / ≤ −.02 HURTS / NEUTRAL. R-C2-7 CHAMPION-RECIPE (mechanical): levers with their primary letter (WIDENS / PROPAGATES-or-REACHES / CALIBRATED) on a STABLE arm with vcold ≥ vcold(W0) − CNC2 are carried to sportC3; wd 1.0 carried iff HOLDS/PARTIAL; champion-so-far = argmax vcold over stable non-memorized natives.

## §5. Predictions (numeric, pre-data)
W0 vcold [43, 50], CE end ≥ .05 (55 %) · R1 reach [.65, .85] (50 %), vcold [42, 50], selector intact (80 %) · R2 first_exact median ≤ 6 vs W0 ≈ 10 (55 %), vcold [44, 52] · R3 top-5 ≥ .90 (45 %), entropy at step 1 < .3, vcold [42, 50] · R4 vcold [39, 44] (60 %) · X1 cold@D16 [80, 85] (a drop of ≥ 1pp, 60 %) · X2 [84, 88] (neutral, 60 %). Kills as in the freethink X-7 table.

## §6. Cost, wall, ops
v6e-16 4×4 static map w0 W0→R2 · w1 R1→R3 · w2 R4→X1 · w3 X2→riders; estimated walls (4-chip workers, measured sportC1 paces): W0 2 h + R2 ≈ 5.5 h (+ evals ≈ 4 h) ≈ 12 h; w1 ≈ 8 h; w2 R4 ≈ 7 h + X1 2.5 h + evals ≈ 12 h; w3 X2 ≈ 3 h + riders ≈ 2 h + evals ≈ 7 h → ≈ 12–14 h ≈ $170–230 (US 16 $13.64/h; Mumbai ≈ $16/h). 8-shape fallback (sequential, PI-approved deadline): W0 R1 R3 X1 R2 R4 X2 in priority order. Deadline launch + 16 h (16) / PI-gated on the 8. **`runs/tpu_deadline.txt` is in the past — the launch procedure bumps it first.** Chain `tools/chain_sportC2.sh` through `tools/harness_sportC2.sh` (all sportC1 scenarios incl. S8 node-change restore) + CPU smokes on real code (carry loop single + DP, inner cycles, hard rows, evaluator hard mode, calibration tool, X2 forward) before silicon; pre-mortem after the harness.

## §7. Labeled deviations
Grafts on the W0 base (§2); R4's fresh optimizer + EMA restart; the SOT carry not checkpointed (a resume restarts rows); X2's parameter count (≈ 5.9M vs X0's 5.04M) and prefix summary path; hard rows are a per-step whole-grid commitment (not per cell); inner cycles multiply per-step compute ×3 (the matched-inference-compute row is the parity run's, not this night's).
