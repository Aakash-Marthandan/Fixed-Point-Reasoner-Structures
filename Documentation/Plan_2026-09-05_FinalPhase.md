# FINAL PHASE — the Decimating Equilibrium Cell (DEC): three nights from the sportC2 verdict to the ICLR freeze (DRAFT for the PI, 2026-09-05 evening; NOT registered — registration follows the PI's go, rules locked in `tools/analyze_final.py --selftest`)

**Status:** plan, pre-registration. Nothing built beyond what sportC2 left; nothing launched; fleet zero; `runs/tpu_deadline.txt` in the past. Companion texts: the sportC2 verdict (`Report_2026-09-05_sportC2_Verdict.md`), the decoder lens (`runs/analysis/sportC2_ecc_20260905.*`), Freethink 2026-09-05 (evening) in `Research_Brainstorm.md` (P-0…P-6), Program Review #3, the Instrument Map.

## §0. Purpose and the thesis this phase must make true or false

**Purpose.** Convert everything the program has measured into one cell that reaches the field's single-pass accuracy on Sudoku-Extreme, delivered with an instrument for every mechanism and with the field's frontier loop read through those instruments — and settle, with numbers, which of the mechanisms depend on Sudoku's local verifiability (the Sudoku-vs-ARC law paper 2 needs).

**The thesis (paper 1, after this phase).** A learned recursive reasoner is a measurable decoder. Its decoder CLASS — soft iterative (halts at stopping sets) versus decimating (commits and revises) — is set by the timescale structure of its state, not by depth, regime, or operator; the field's frontier loop is decimating because it carries a slow decision field trained to be decodable at every step. A cell that carries exact symmetry, equilibrium objectives and a calibrated commit head into that class reaches the field's accuracy with no digit augmentation, a verifier-free selector at the verifier's level, and a physics reading (erasure threshold = phase transition; trapping sets = metastable states; decimation = symmetry breaking) with the numbers to back it.

**What "differentiated, not redundant" means here (the test every deliverable passes):** the field reports accuracies and ablations; EqR reports fixed-point residuals and Jacobian norms; CMM reports an equilibrium loss. We do not re-report those as ours (we cite them). Ours are the instruments in §5 that no series paper has: the decoder-class ladder against reference decoders, erasure thresholds and search-class yield, per-puzzle channel structure and list-decoding cost, trapping-set dynamics, decimation quality at stalls (E3), the converged-wrong rate as the selector law, the timescale-separation reading of the state, exact symmetry versus augmentation, the decay-rate law of memorization, the price's sign arc, and the verifier-dependence table.

## §1. The knowledge inventory the design obeys (fact → source → design consequence)

| # | measured fact | source | consequence for the DEC |
|---|---|---|---|
| 1 | Our natives are SOFT iterative decoders (g50 ≈ 26–27 givens; search-class yield 25–36 %; 3–9 % monotone solves; trapping-set stalls); the field's loop is DECIMATING (no threshold; 92 %; 63 % monotone; commit 96 % at step 1) | decoder lens 09-05 | the wall is the decoder class → the DEC adopts the field's two-timescale loop (fast messages / slow decisions) as its base |
| 2 | Per-step depth is not binding: our operator ×3 per readout leaves the stalled set unchanged | R2 DEPTH-FLAT | L_cycles stays at the field's 6; no compute spent on more inner iterations |
| 3 | Digit augmentation is worth 65 pp on the field cell (21.2 without it) | X1 ORBIT-LOAD-BEARING | exact S9 by weight sharing over the nine digit fields replaces it (our C2 construction) — the DEC trains with NO digit augmentation |
| 4 | The all-to-all token mixer is load-bearing (our group mixer in its place: −47 pp, no restart ever succeeds) | X2 GEOMETRY-HURTS | the DEC keeps the MLP-mixer over the 81 cell tokens (per field); our shared all-different operator enters only as an additive branch, tested last |
| 5 | The field regime (wd 1.0 · lr 1e-4 · batch ≥ 384 · EMA) buys stability and anti-memorization on our cell, not accuracy; the lever is the decay RATE lr·wd | R4 REGIME-FLAT, W0 WD-PARTIAL, H-49 | the DEC trains in the field regime; no regime arms in this phase |
| 6 | ACT = +14/+16 pp on the field cell and prevents memorization; the halting head is a learned verifier trained on correctness | sportC1 X0/X0n | ACT on; its calibration is recorded as the first row of the verifier-dependence table |
| 7 | The SOT carry on a single-field cell learns soft stationary WRONG states and narrows reachability (ρ 57 → 44) | R1 CARRY-FLAT | the carry exists only together with a slow decision field trained by deep supervision (never alone) |
| 8 | Hard-decision rows without a contradiction signal = a fitting accelerator; confidence at stalls stays 33–42 % wrong at p > .9 | R3 COMMIT-PARTIAL, E3 | the commit head is trained on a per-cell correctness/consistency target with a proper scoring rule, and hardening is SELECTIVE (only where the head is confident) |
| 9 | Our RI+FPA maps: converged-wrong draws ≤ .1 % (residual ≈ verifier); the field's base 3–6 % (caps its residual selector at 92.75 % of hits) | program review #3, lens G | FPA/NI on the field loop = the SELECTOR LIFT (a verifier-free statistic at the verifier's level) |
| 10 | RI buys convergence robustness on its own init distribution (H-37); on our cell it stopped draws exploring; EqR's RI is z0 ~ N(0, σ) | H-37, freethink X-3 | RI on the field loop is a single toggle (EqR's own σ = 1), measured, not assumed |
| 11 | Our solved endpoints are exact contractive fixed points (λ .67–.73); the field's are readout-stable, latent-drifting states (λ 1.07) | lens G | the DEC's fixed-point census is an instrument (is a decimating decoder a fixed point at all?), not a design constraint |
| 12 | Memorization = curvature collapse (PR/n 7e-6 vs 4e-4) that erases low-givens capability first; wd 1.0 protects at the field's rate | freethink grounding, H-49 | EMA row + val-selection stay; the memorization gate is in the analyzer's CLEAN split |
| 13 | The champion cell is UNPRICED (β_flux 0; β_nl 1e-6 only); the price's sign arc: kills at d16 (H-43) → recovers with scale (H-44) → precondition for deep training at d96 (H-48) → transfer dividend on ARC (Law 4) | ledger §3, `chain_sportC2.sh` | one priced-coupling arm on the DEC if the port costs ≤ 0.5 day (§3 B4); otherwise the arc's fifth point waits for paper 2 |
| 14 | Verifier-selected restarts are a coverage column (the field's pass@k), never a headline; the canvas maps lose their threshold entirely under list decoding | protocol table; lens E1 | the headline is the single pass with ACT; residual-selected and verified columns labeled |
| 15 | 10/11 predictions missed (4/5 the round before); the CNC2 "noise" read 8.2 pp because it was a treatment | sportC2 verdict | predictions as bands; the noise floor from SEED PAIRS only; paired McNemar on identical puzzles; one variable per arm |
| 16 | Paces: X0 (5.03M) 27 it/s at batch 768 on a v6e-8 (50k in 86 min); full test ≈ 8 min sharded; 20k k128 scan ≈ 30 min; d128 native 21 it/s on 4 chips | sportC1 report §7 | walls in §7 |

## §2. The architecture: DEC — the Decimating Equilibrium Cell (`src/qhrrn2/dec_cell.py`, `--cell dec`)

**Representation (ours, C2).** The state lives on the 81-cell lattice × 9 digit FIELDS × width w: `z_L`, `z_H` each of shape (9, 81, w). Every parameter is shared over the field axis, so a permutation of digits permutes the state and the logits exactly (CI test: `test_dec_equivariance`, bit-exact under a random S9 element). Givens enter as a per-cell, per-field role embedding ("this cell is given AS this digit" / "given as another digit" / "empty"), position-group augmentation (band/stack/transpose orbit, `--sudoku-aug 1000`) kept, NO digit augmentation. The 16-token puzzle prefix of TRM is dropped (one identifier on Sudoku; zero-init; measured inert).

**The block (the field's).** TRM's block applied per field with shared weights: MLP-mixer token mixing over the 81 cells, SwiGLU (expansion 4), post-norm parameter-free RMSNorm, sqrt(w)-scaled embeddings — exactly `trm_cell._block` vmapped over the field axis. Plus the one thing 81-token TRM cannot have and our cell has by construction: an equivariant FIELD COUPLING per cell — the mean over the other eight fields (the DeepSets form of the all-different message: "what the other digits claim here") added to each field's token before the block. Optional additive branch (B-night only): our shared all-different unit operator (`cell.group_mixer`) on a w/4 projection.

**The loop (the field's, verbatim).** `z_L ← block(z_L + z_H + emb)` L_cycles = 6 times, `z_H ← block(z_H + z_L)` once, H_cycles = 3 per segment with the gradient through the last; deep supervision per segment (cross-entropy on `readout(z_H)`, stablemax); ACT q-halt head (TRM no-ACT-continue variant, as X0); SOT carry of (z_H, z_L) across segments; EqR damping λ .05 and per-pass noise β .01 (the field's NI). Readout: equivariant — a shared vector reads each field's cell token to one logit → 9 logits per cell (givens pass through).

**Our objectives on the loop (new on this path; `objective.py` gains a `dec` branch).** FPA anchor rows (k 4, ε .2): corrupt ε of the solution's cells, run the loop from the anchored state, supervise every step toward the solution (the H-45 mechanism that gave our maps their contractive fixed points and ≤ .1 % converged-wrong draws). RI as EqR's σ = 1 (`--trm-ri-sigma`, exists). NI stays the field's β.

**The commit head (new; the decimation rule).** Per cell, per outer step, a scalar `c = σ(v·z_H)` trained with a proper scoring rule (binary CE) on the target "the argmax of this cell is correct" (available at training; a learned per-cell verifier at test, the same status as the halting head). In the loop it selects the feedback: where `c > τ` (τ .9) the feedback to `z_L` is the hardened one-hot (straight-through), elsewhere the soft readout — commitment where the head is calibrated, revision elsewhere. The instrument is E3 (wrong-among-committed at stalls) and the calibration rows; the bar is 90 % (freethink X-3).

**The syndrome-feedback arm (instrument only, Sudoku-only, labeled).** The per-cell violation count of the current argmax grid (computable from the rules alone) fed back as a scalar channel. This is the explicit local verifier inside the loop; it does not transfer to ARC by construction, which is exactly why it is measured: its gain is the verifier-dependence of the decimating class.

**Sizes and compute.** `w = 256` per field: ≈ 1.3M parameters (TRM's block scales as w²), activations 9 × 256 = 2304 per cell (4.5 × X0's 512); expected pace ≈ 6–9 it/s at batch 768 (inferred from X0's 27; the canary re-prices) → 50k steps ≈ 1.6–2.3 h. `w = 512`: 5.0M parameters (X0's count), ≈ 3 it/s → 50k ≈ 4.6 h (the scale arm, B-night, remat if the HBM asks). Every cross-system number carries its parameter count AND its inference compute per puzzle (MAC count instrument, owed since 09-02 — built this phase, §6).

**Why this is ours and not a re-skin.** Three of its five ingredients come from our measurements and do not exist in the field's cell (exact S9 by field sharing; FPA anchoring; the calibrated commit head), the fourth (the field coupling) is our C2 construction on their block, and the loop is adopted BECAUSE the lens measured it as the decoder class — the paper says so, and the additive ledger (§3) attributes every point to its ingredient.

## §3. The three nights (additive ledgers; one variable per arm; bands pre-data; the seed pair is the only noise floor)

### Night A — Sep 7 → 8: "the class transfer and the field ledger" (v6e-8 US first; ≈ $120–200)

| arm | one variable (from its base) | base | steps / regime | wall (inferred) | worker |
|---|---|---|---|---|---|
| **A0** | seed 2 | sportC1 X0 (TRM cell + SOT + ACT + digit aug; field regime, batch 768) | 50k | 1.5 h + evals | w0 |
| **A1** | `--fpa-k 4 --fpa-eps 0.2` on the field cell (build §6.1) | X0 | 50k | 1.7 h + evals | w0 |
| **A2** | `--trm-ri-sigma 1` (EqR's RI; exists) | X0 | 50k | 1.5 h + evals | w0 |
| **A3** | **DEC-w256, no digit aug** (build §6.2) | X0's loop and regime | 50k | 2–2.5 h + evals | w1 |
| **A4** | DEC-w256 **+ digit aug** (the redundancy control) | A3 | 50k | 2–2.5 h + evals | w1 |
| **A5** | DEC-w256 + FPA + RI (the assembled objectives) | A3 | 50k | 2–2.5 h + evals | w1 |
| rider | BP and BP+decimation reference decoders (CPU, §6.5); the decoder lens on every A arm at analysis | — | — | — | Mac |

Evals per arm (standing battery): full test at D16 and D64 (cold, EMA), 20k stratified k128 scan (b1, t1r@k, verified@k, per-draw exact bits + residuals), calibration rows (`stall_calibration.py`), census, monitors; the lens's thresholds / yield / dynamics / E3 at analysis time.

**Predictions (bands locked at registration):** A0 within ±1.0 of X0 at D16 (the noise floor); A1 converged-wrong 3–6 % → < 1 %, t1r@128 92.5 → ≥ 97, cold within ±1.5 (SELECTOR-LIFT); A2 cold within ±1, ρ within ±1, selector AUC up (RI-ON-FIELD ∈ {FLAT, HELPS, HURTS}); **A3 cold@D16 ∈ [70, 90]** (X1's 21.2 is the floor; ≥ 61 = +40 over X1 reads ORBIT-EXACT; ≥ 84.8 = the EqR base → PARITY-AT-D16); A4 − A3 ∈ [−2, +2] (AUG-REDUNDANT); A5 − A3 ∈ [−1, +3] with converged-wrong < 1 %.

### Night B — Sep 10 → 11: "commitment" (≈ $200–300)

| arm | one variable | base | steps | wall (inferred) | worker |
|---|---|---|---|---|---|
| **B0a / B0b** | seeds 1 and 2 (the claim-bearing pair) | A's best DEC objective set (A3 or A5 by rule) | 50k | 2–2.5 h each + evals | w0 / w1 |
| **B1** | + commit head, selective hardening τ .9 (build §6.3) | B0 | 50k | 2.5 h + evals | w0 |
| **B2** | + syndrome feedback (instrument arm; Sudoku-only, labeled) | B0 | 50k | 2.5 h + evals | w1 |
| **B3** | **DEC-w512** (the scale arm; X0's parameter count) | B0 | 50k | 4.6 h + evals | w2 (16-node) or after B1 on w0 |
| **B4** | + priced field coupling `--beta-flux` on the field-mean message (ONLY if §6.4's port ≤ 0.5 day) | B0 | 50k | 2.5 h | w1 |

**Predictions:** B0 pair within ±1.5 of each other (the floor); B1: E3 wrong-among-committed at stalls < 10 % (from 33–52), search-class yield + ≥ 3 pp, cold + [0, 4] (COMMIT-CALIBRATED); B2: cold + [2, 8] — if ≥ 3 the decimating class is VERIFIER-DRIVEN in part (the row that goes to the verifier-dependence table); B3 ≥ 92 at D64 single pass = PARITY (EqR 93.0 / TRM 87.4 / X0 92.8); B4 cold ±2 with throat compressed (sign-arc point 5, either way).

### Night C — Sep 13 → 14 (the PI's choice on B's read; ≈ $100–250)

- **C-i (parity headline):** the B-winner at w512 × 2 seeds at matched inference compute, D64 and D128 columns, the full protocol table (single pass with ACT / B = 1 / residual-selected @128 / verified @128) — the paper's headline pair, seeded.
- **C-ii (paper-2 bridge):** ARC-d96 with the DEC (exact color-permutation + dihedral symmetry; the commit head as the verifier-free commit rule; the wd 1.0 arm owed since review #3) — the ARC column of the verifier-dependence table; the converged-wrong rate per mechanism per domain.
- If A3 reads below 50: B pivots to "X0 + our objectives + commit head" on the flat cell with digit aug (the amalgamation without the symmetry claim) — the paper's claim shrinks to the decomposition, the selector lift and the decoder-class account; Night C = C-ii.

## §4. Decision rules (locked verbatim in `tools/analyze_final.py --selftest` at registration; letters only)

- **INTEGRITY:** one checkpoint path per (arm, selection); every summary carries `ckpt`; grids at the monitor cadence (§6.7 fix).
- **CLEAN split:** STABILITY {STOPPED, retfm < .9, census} separate from MEMORIZATION {end train-CE < .02, vsel-vs-final drop > 5 pp}; a memorized arm's val-selected grid remains a measurement, labeled.
- **NOISE FLOOR:** |A0 − X0| at D16 and |B0a − B0b|; a contrast is READ only if it exceeds 2 × the floor; else FLAT. Never a treatment as the floor (the CNC2 lesson).
- **ORBIT-EXACT:** A3 − X1 ≥ 40 pp AND |A4 − A3| ≤ 2 → EXACT-S9-REPLACES-AUG; A3 − X1 ≥ 40 AND A4 − A3 > 2 → PARTIAL; A3 − X1 < 40 → FAILS.
- **SELECTOR-LIFT:** A1 converged-wrong < 1 % AND t1r@128 ≥ 97 → LIFT; cold change within the floor required for LIFT-FREE, else LIFT-AT-COST.
- **RI-ON-FIELD:** HELPS / FLAT / HURTS by the floor on cold and ρ.
- **DECODER-CLASS (per arm, from the lens):** DECIMATING if g50 is absent in 17–35 AND search-class yield ≥ 70 % AND monotone solves ≥ 40 %; SOFT if g50 ≥ 24 AND yield ≤ 45 %; MIXED otherwise.
- **COMMIT-CALIBRATED:** B1 E3 wrong-among-committed (p > .9) < 10 % AND yield + ≥ 3 pp over B0 → CALIBRATED; wrong < 10 % without the yield → CALIBRATED-INERT; else UNCALIBRATED.
- **VERIFIER-DRIVEN:** B2 − B0 ≥ 3 pp cold → VERIFIER-DRIVEN; within the floor → CLASS-INTRINSIC.
- **PARITY:** single pass with ACT at D64, EMA labeled: ≥ 92.0 → PARITY; [87.4, 92) → TRM-CLASS; [84.8, 87.4) → EQR-BASE-CLASS; < 84.8 → BELOW.
- **CHAMPION BY RULE:** the highest CLEAN single-pass D64 number among DEC arms, seeded where a pair exists; "program-record" prefix only.
- **PRICE (B4):** cold within the floor AND throat down ≥ 3× → FREE-COMPRESSION; cold down > floor → COSTS; cold up > floor → PAYS.

## §5. The measurement spine of the paper (what is ours; what it shows; the figure)

| instrument | what it measures | the field reports instead | figure / table |
|---|---|---|---|
| Decoder-class ladder | peeling (11.1 %) < BP (§6.5) < our natives (≈ 40) < BP+decimation < the field's loop (92 search-yield) < ML (100), with each learned cell's stall set overlapped with BP's | accuracies | Fig. 1: the ladder; the "learned cell ≈ BP class" overlap |
| Erasure thresholds + search-class yield | g50, waterfall width, solve rate by rating band; threshold removal under list decoding | none | Fig. 2: threshold curves per cell (ours / field / DEC) |
| Per-puzzle channel + list-decoding cost | never / always / intermediate fractions; k50/k90; rescue by draws | pass@k only | Table: channel structure per cell |
| Trapping-set dynamics | monotone-solve fraction, un-peel events, syndrome plateau and oscillation, first-exact | none | Fig. 3: dynamics of a soft vs a decimating decoder |
| Decimation quality at stalls (E3) | wrong-among-committed, peeling from committed cells: solve / stuck / contradiction; the commit head's calibration | none | Table: E3 before/after the commit head |
| The selector law | converged-wrong rate vs residual-selected cap; ≤ .1 % (ours) vs 3–6 % (theirs) → the FPA lift | Top-1-residual @128 (EqR) | Fig. 4: selector cap vs converged-wrong rate |
| Timescale separation (new on the DEC) | relaxation time of z_L at fixed z_H vs the z_H update; commitment per outer step; the fixed-point census of a decimating decoder | fixed-point residuals (EqR) | Fig. 5: fast/slow relaxation and commitment |
| Exact symmetry vs augmentation | 65 pp (X1) and the DEC's no-aug parity; A4's redundancy control | augmentation ablations | Table: symmetry row |
| The decay-rate law | lr·wd as the memorization lever; PR/n curvature collapse | wd as a fixed convention | Fig. 6 (supplement) |
| The price's sign arc | H-43 → H-44 → H-48 → Law 4 (+ B4) | none | Table: sign across scale × domain |
| Verifier-dependence table | per mechanism: gain on Sudoku vs gain without local verifiability (ACT head, residual selection, FPA, RI, commit head, syndrome) with the ARC column (C-ii) | none | Table: the Sudoku-vs-ARC law |
| Sudoku-vs-ARC catalog | the D-rows of the Instrument Map, closed by this phase (D13 decoder class; D14 verifier dependence) | none | Section |

Not claimed (cited): fixed-point residual selection (EqR), the equilibrium/contraction losses (CMM), pass@k with a Q-head (PTRM), stablemax and ACT (HRM/TRM).

## §6. Builds owed (each with its harness scenario and its CI test; estimates in Fable/Opus days)

1. **FPA on the field-cell path (0.5 d).** `objective.py`: a `trm`/`dec` branch that runs the anchor rows through `segment` (corrupt ε of the solution → anchored (z_H, z_L) via the embedding of the corrupted grid → k steps → per-step CE), weight `fpa_w`; bit-exact when `fpa_k` = 0 (`tests/test_fpa.py` extended); smoke: anchor-row CE falls in 200 CPU steps.
2. **The DEC cell (1–1.5 d).** `dec_cell.py`: field-batched TRM block (vmap over the field axis; shared params), equivariant embed/readout, field-mean coupling, `--cell dec`, `--dec-width`, the group-operator branch flag (off by default); reuse `segment`/ACT/SOT by making them field-agnostic; CI: `test_dec_equivariance` (bit-exact logits under a random S9 element), param count assertion, forward shape; evaluator/census/monitor/calibration mirrors (`cell_kind == "dec"`: z0, latent residual over the last 3 outer steps as for trm); smoke: single + DP-2, 300 steps, evaluator full/screen styles read the checkpoint.
3. **The commit head + selective hardening (0.5 d)** and **syndrome feedback (0.25 d).** Flags `--commit-head`, `--commit-tau`, `--syndrome-feedback`; the head's labels from the solution; the syndrome from the argmax grid (the lens's `syndrome` function, jitted); evaluator emits the head's calibration rows; CI: hardening is identity when τ = 1; syndrome of a valid grid is 0.
4. **Priced field coupling (0.5 d; optional — B4 only if it fits Sep 9).** A VIB toll on the field-mean message (our C14 form on one cut); flux logged; `--beta-flux` honored on `dec`.
5. **Reference decoders (0.5 d, CPU).** `tools/reference_decoders.py`: sum-product BP on the Sudoku factor graph (cells × candidates; all-different checks as row/col/box factors with the standard candidate-elimination messages), BP + decimation (commit the most polarized candidate, re-run), both on the 20k scan set with the same erasure sets; outputs the solved sets for the ladder and the stall-set overlaps with each learned cell.
6. **`tools/analyze_final.py` (0.5 d)** with the §4 rules and a selftest; `chain_final.sh` + `harness_final.sh` with one scenario per resume path and per new flag (S1 fresh, S2 preempt-resume with the carry, S3 torn ckpt, S4 dec-specific evaluator, S5 the negative scenario per new flag asserting the staged failure fired); pre-mortem after the harness, before registration.
7. **Owed fixes rolled in:** val-selection cadence (grids at every monitor step, 2k), the MAC-count instrument (inference compute per puzzle per cell; the protocol table's compute column), `spend_report.py` zone-aware rates (ops), early-NaN abort in the trainer.

Build calendar: Sep 6 (Sat/Sun): items 1, 2, 6 (chain + harness), 7 (cadence); Sep 7 (Mon) morning: smokes, pre-mortem, registration, campaign env, deadline bump, launch on the PI's go (evening IST); Sep 8–9: analysis of A, items 3 (and 4 if it fits), 5; Sep 9 evening: B's registration; Sep 10 launch; Sep 12: B's analysis; Sep 13: C's registration and launch; Sep 15: C's analysis; Sep 16: freeze.

## §7. Ops, walls, cost

- **Policy:** one spot pod per night (v6e-8 US first, us-east1-d; Mumbai only when the US is dry); `pod.sh` supervise + campaign env + `runs/tpu_deadline.txt` bumped before launch; launchd watchdog + session Monitor + hourly heartbeat; Mac on AC (`pmset -g batt` at every handoff); live 5-min GCS banking at every chain start; node-side guard re-planted per node and per deadline; the dispatcher DMS past the deadline; canary before the campaign; the analyzer never run in the ops phase.
- **Walls (inferred from §1.16; the canary re-prices):** Night A sequential on a v6e-8 ≈ 6 arms × (train 1.5–2.5 h + evals ≈ 0.7 h) ≈ 14–17 h → run A as two workers on a v6e-16 (≈ 8–9 h, ≈ $120) or accept the 8's wall (≈ $100–115); Night B ≈ 5 arms with B3 at 4.6 h → v6e-16, two workers, ≈ 9–11 h (≈ $130–160) plus weather; Night C ≈ $100–250. Total ≈ $350–600 of the ≈ $750–850 left; the ARC extension (C-ii) inside it.
- **Riders on idle chips:** the sportC1/C2 protocol-table gaps (matched inference compute columns), the DEC's timescale instrument at D128, canvas EqR-statistic evals.

## §8. Deliverables to the paper by date

- **Sep 8 (drafting starts, independent of the nights):** the reproduction section (X0 86.0 / 92.8 / 99.7 with protocol + regime columns), the decomposition (cell vs regime; the two load-bearing ingredients), the decoder-lens section (thresholds, dynamics, E3 on the sportC2 corpus), the decay-rate law, the price's sign arc, the Sudoku-vs-ARC catalog, the comparator table with its protocol and compute columns.
- **Sep 9:** Night A's rows — the symmetry result, the selector lift, the field ledger; the decoder ladder with BP/BPD.
- **Sep 12:** Night B's rows — the commit head (E3 before/after), the verifier-dependence row, the parity number if B3 reads it.
- **Sep 15–16:** Night C — the seeded headline pair or the ARC column; freeze Sep 16; abstract Sep 18; full paper Sep 25 AOE. AAMAS (Oct 2 / Oct 9) remains the fallback for paper 1 if the parity number is not clean; paper 2 (ARC) → AAMAS Oct 9 / ICML Jan.

## §9. Adversarial review (pre-registration; the objections we expect and the answers we can measure)

1. *"It is TRM with a symmetric embedding."* The additive ledger attributes every point: the symmetry row (A3/A4 vs X0/X1), the objective rows (A1/A2/A5), the commit head (B1) — each a measured contrast on identical puzzles with the seed pair as the floor. The paper's object is the decoder-class account and the instruments; the DEC is its constructive demonstration.
2. *"Nine fields cost nine times the activations."* Reported at matched parameters (w512 = X0's 5.03M) AND matched inference compute (MAC instrument); the w256 arm shows the accuracy-per-MAC curve, not a single point.
3. *"The commit head is ACT's halting head per cell."* Yes — that is the mechanism: the per-cell learned verifier is the decimation rule, and E3 measures whether it is calibrated where it matters (at stalls), which the halting head is not asked to be.
4. *"The syndrome arm uses the rules."* Labeled instrument, never in a headline; its whole purpose is to price the class's dependence on local verifiability.
5. *"Single seeds."* Pairs on every claim-bearing arm (B0a/b, C-i); paired McNemar on the identical 20k; contrasts under 2 × the floor read FLAT.
6. *"Verified@128 is an oracle."* Coverage column only, alongside the field's pass@k, and the FPA lift is stated on the residual-selected (verifier-free) statistic.
7. *"The field regime is not your regime."* Measured: it buys stability and anti-memorization on our cell and nothing else (R4/W0); the DEC inherits it and the paper says why.
8. *"Three nights are too few."* The contingency in §3 shrinks the claim, never the discipline; a slipped night drops C, never the freeze.
9. *"Where is ARC?"* Night C-ii and the verifier-dependence table; paper 2 carries the conversion.

## §10. After the final phase (next steps, in order)

1. **Paper 2 (ARC conversion):** the DEC with exact color-permutation × dihedral symmetry, the commit head as the verifier-free commit rule, wd 1.0 in the regime; the verifier-dependence law as its opening; the ARC-d96 rung first (≈ $60–100), then d128.
2. **The rate term (B1, C19)** — the theory-completing objective, on the DEC's priced coupling once B4 has read.
3. **Scale:** d160/d192 stretch after the paper, seeded, matched compute.
4. **Instrument back-ports:** the lens as a standing analyzer module for every campaign (thresholds, dynamics, E3 as registered letters); the reference-decoder ladder for any CSP the program touches next.

## §11. NIGHT A — REGISTERED 2026-09-05 (build complete; launch on the explicit PI go) + PRE-MORTEM AUDIT

**Locked:** rules in `tools/analyze_finalA.py` (selftest 25/25); chain `tools/chain_final.sh`; harness `tools/harness_final.sh` 52/52; env `tools/campaign_final.env`; ledger §5 entry of this date. Gates: `tests/test_final.py` 5/5, suite 156 green, CPU smokes rc 0 on every tool, the X0 path bit-exact vs HEAD.

**Pre-mortem (run after the harness, before the go; each item names its countermeasure):**
- **PM-A-1 HBM at w256 under DP-4 (v6e-16 workers, 192 rows/chip).** Activations per stack pass ≈ 192 × 729 tokens × ~2.3k floats ≈ 1.3 GB, ~14 live passes under the last-H-cycle gradient ≈ 18 GB of 32 GB: borderline. Countermeasure: `pt_run`'s ONE `--remat` retry (harness S9), now honoured by the DEC's segment (`jax.checkpoint` per stack pass); on the 1×8 fallback the per-chip batch halves.
- **PM-A-2 Pace unknown for the DEC.** The canary + the first A3 log lines re-price the walls; the 8.5 h wall recycles idempotently (markers + live bank); the 1×8 sequential order puts the DEC arms first so their pace is read early.
- **PM-A-3 The evaluator's RI draws on the DEC carry.** `mi_z0` draws the cell's own shape (2, 9, 81, 256) per row per draw (≈ 373k floats per row; ≈ 380 MB at batch 256): fine; smoke 2b exercised the screen-style scan with k = 2.
- **PM-A-4 A4's digit augmentation on an exactly equivariant model** changes the augmented data stream, not the function class: the A4 − A3 contrast carries data-order noise; read only beyond 2×FLOOR (the rule) and labeled.
- **PM-A-5 The spurious rate needs `mi_resid_k` in the scan records.** The evaluator writes the latent residual for cell_kind in {trm, dec} on the same code path as X0's sportC1 scan (lens E5 read it); X0's reference records are on disk (`runs/sxscan_psportC1X0`) — the analyzer reads them, else the labeled constant .078.
- **PM-A-6 Disk and bank volume at the 2k grid cadence:** 25 grids × ≈ 80 MB per X0-class arm (≈ 2 GB; the DEC-w256 ≈ 0.6 GB) → ≈ 8 GB per campaign on the node and in `*_pretrain.tgz`; the 5-min live bank syncs deltas only.
- **PM-A-7 bf16 numerics of the padded logits:** −1e4 is representable in bf16; stablemax at −1e4 gives s ≈ 1e-4 (finite gradient); the smokes ran in f32 on CPU — the first TPU log lines (finite loss at step 5) are the check.
- **PM-A-8 The launch procedure (owed since sportC2):** `cp tools/campaign_final.env tools/campaign.env` → bump `runs/tpu_deadline.txt` (in the PAST now) → `bash tools/pod.sh supervise` with `caffeinate -s` on AC → `plant_guard` per node → Monitor + hourly heartbeat; the DMS past the deadline; the analyzer never runs in the ops phase.
- **PM-A-9 The harness stub cannot see a TPU-only runtime failure of the new cell** (compile, sharding, bf16): the chain's canary + the first PRETRAIN-START log of A3 are read at the source within 15 min of launch, per the standing verify-at-source rule.
- **PM-A-10 Seeds:** A0 is the only pair; every A-night contrast is single-seed by design and the floor rule reads FLAT inside 2×FLOOR — Night B seeds the claim-bearing arm.
