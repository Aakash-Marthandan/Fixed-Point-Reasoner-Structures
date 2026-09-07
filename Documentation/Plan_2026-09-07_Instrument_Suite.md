# THE COMPREHENSIVE INSTRUMENT SUITE — design for the frontier models on a pod, the corpus-wide backfill on the banked data, and the final Sudoku night (DRAFT for the PI, 2026-09-07; NOT a registration — the frontier registration and the champion-night registration follow the freethink)

**Status:** design + inventory + the $0 backfill launched. Budget after the Professor's approval: **$1,000** for (i) the frontier battery on a pod, (ii) the final Sudoku champion night, (iii) the ARC runs, (iv) reserve. Calendar: freeze Sep 16, ICLR abstract Sep 18, full paper Sep 25 AOE. Companion texts: `Report_2026-09-07_NightA_Verdict.md` (the DEC verdict and the trajectory law), `Report_2026-09-06_FieldCheckpoints_Instruments.md` (the Mac run on the public checkpoints), `Note_2026-09-05_Field_KnowledgeBase.md` (SE-RRM, PTRM, CMM, GRAM, CGAR), `Report_2026-09-05_sportC2_Verdict.md` §4 (the decoder lens), `Sudoku_vs_ARC_Instrument_Map.md` (the catalog), `Plan_2026-09-05_FinalPhase.md` §5 (the measurement spine as first drafted). Inventory artifact: `runs/analysis/inventory_20260907.txt` (694 eval artifacts; ~170 checkpoints; the 44 Mac-study outputs).

## §0. What the suite is for, and the three consumers

The paper's claim is not an accuracy; it is that a learned recursive reasoner is a **measurable decoder**, and that the same instruments read our cells, the field's public checkpoints and the DEC — and separate Sudoku from ARC. Three consumers use one suite:

1. **The frontier pod run** — the public HRM / TRM (alphaXiv, CGAR) / EqR checkpoints at the FULL protocol (the Mac run used k = 8 on 5k, strat-256 dynamics, strat-84 Jacobians, fp32) — so every comparator row in the paper is at the same protocol as our arms, plus the rows the Mac could not afford (depth ladder to D128, the 20k × k128 selector column, the PTRM test-time protocol on the same checkpoint, the orbit defect at 20k).
2. **The corpus-wide backfill on banked data** ($0, the Mac) — every instrument that reads records or checkpoints applied to every banked map: the cold ladder d16 → d128 → canvas → field cells → DEC through ONE lens, so the paper's figures are corpus-wide and the freethink starts from a complete table rather than from the last night's arms.
3. **The final Sudoku night** — the suite's decision rules become the night's registered letters (seeds, the selection instrument, the compute column, the decoder-class letters, the selector law), and the night's battery is the suite by construction.

The test every instrument passes before it enters: it answers a named question of §1, it has a definition with a formula, a set and a seed, a protocol column, a cost, a tool that exists or a build item with an estimate, and a consumer.

## §1. The questions the paper asks, and the instruments that answer them

| # | question | instruments (§2) | figure / table |
|---|---|---|---|
| Q1 | What decoder class is a cell, and where does its accuracy come from? | R1 thresholds + yield, R2 first-exact, C1 dynamics, C2 decimation at stalls, X1 reference decoders | Fig. 1 the decoder-class ladder; Fig. 2 threshold curves; Fig. 3 soft vs decimating dynamics |
| Q2 | When does verification-free selection work, and what is the verifier worth? | R5 list decoding, R6 selector law, R7 b1 / majority, R10 halting head, C11 test-time protocols | Fig. 4 the selector law; the verifier-dependence table |
| Q3 | What is the attractor structure — retention, contraction, spurious limits, explosions? | C5 retention + ε-ladder, C6 fixed-point census, C7 init-basin radius, C4 explosion census | Fig. 5 fixed points and basins |
| Q4 | What is exact symmetry worth, and what does augmentation buy a non-equivariant cell? | C8 orbit defect, the A3/A4/X0/X1 rows, C8's exactness control on S9 cells | the symmetry table |
| Q5 | What happens along training — memorization, peaks, the selection instrument? | T1 trajectories, T2 selection, T3 curvature, R9 memorization radius, C12 train-set readout | Fig. 6 the trajectory law |
| Q6 | What does it cost? | M1–M4 params, MACs, measured walls, accuracy-per-MAC | the comparator table's compute column |
| Q7 | How does accuracy scale with inference depth, and does it regress? | R4 depth ladder | the depth row |
| Q8 | Are the comparator numbers reproducible and numerically stable? | C10 bf16/fp32 floor, the port verification (I12), the reproduction column | the comparator table's reproduction column |
| Q9 | Which mechanisms depend on Sudoku's local verifiability (the Sudoku-vs-ARC law)? | the D-catalog rows: C5, R6, C1's commit/revision axis, R4 — read on ARC by the back-port list (§8) | the Sudoku-vs-ARC section |

## §2. The instrument catalog (definitive; ID · definition · set/protocol · tool/status · cost · consumer)

Conventions used throughout: **cold** = one deterministic pass from the cell's own start (the benchmark number); **D** = outer steps at inference; **draw** = a random-init restart (N(0, σ = 1) on the carried latent for field-class cells; a uniform random canvas for our natives; EqR's own truncated normal as a labeled variant); **verified** = a draw whose completion is valid and consistent with the givens (Sudoku's free verifier); **residual** = the evaluator's convergence signal (mean |Δ latent| over the last 3 outer steps for field-class cells; mean |Δy| for natives); **stall** = a cold trajectory unsolved at t = 64. Sets in §3.

### 2.1 Record-level instruments (read from the evaluator's per-puzzle records; $0; `tools/suite_records.py`)

| ID | name | definition | set / protocol | status |
|---|---|---|---|---|
| **R1** | Erasure thresholds and yield | logistic fit P(cold ∣ givens) → g50 (the 50 % crossing) and the 10–90 % width; "no threshold" iff no crossing in 17–35; P(cold ∣ rating 0) = the propagation class; P(cold ∣ rating > 0) = the search-class yield; solve rate by rating band 0 / 1–9 / 10–29 / 30–59 / 60+ and by givens bin; per source file | any records; the 20k scan and the fulls | exists (lens E1; analyzer R-A-4) |
| **R2** | One-shot vs propagation | first_exact on solved puzzles: median, p90, fraction exact at outer step 1 and ≤ 2; by octile | any records with `first_exact` | exists (physics H) |
| **R3** | Failure texture | mean violations on failed puzzles, cells correct, givens kept, valid-but-wrong fraction, given-violating recalls | any records | exists (summaries) |
| **R4** | Depth ladder | exact at D ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256} from one rollout's per-step exact bits; per-puzzle regressions (solved at D, not at D′ > D) | the 20k scan set (one cold run recording exact-by-step); fulls at D16 / D64 | exists on the field harness (exact_by_step); our evaluator records first_exact only → **build B1a** (`--record-by-step`) |
| **R5** | List decoding (the funnel) | verified@k for k ≤ K; ρ = decodable fraction; per-puzzle per-draw rate r_i and its spectrum (never / (0,.05] / (.05,.2] / (.2,.5] / (.5,1]); k50 / k90 of the reachable set; rescue of cold failures by 1 / any draw; the (ρ, r) fit per octile (censored geometric on draws ≤ 64); the coverage exponent (sBG, optional) | 20k × k128 (the registered column); strat-512 × k256 (screens, labeled) | exists (lens E4; physics F) |
| **R6** | The selector law | residual AUC P(resid correct < resid wrong); spurious rate = wrong draws below the correct-draw median residual; t1r@k (Top-1-by-residual) vs verified@k and their ratio; EqR's logit-delta score as the second residual (field harness) | same as R5 | exists (lens E5; analyzer SPURIOUS) |
| **R7** | b1 and majority | b1 = exact of draw 1 (EqR's B = 1 statistic); unverified cellwise majority@k (the aug-HRM column) | R5's sets (majority needs `--vote-unverified`, strat-scale) | exists |
| **R8** | Paired contrasts | exact McNemar on identical idx (only-A / only-B / p); unions and Jaccard of solved sets; nesting P(A fails ∣ B fails) | every pair on an identical set | exists (physics E; field J) |
| **R9** | Memorization radius | vsel vs final cold by givens bin (the low-givens capability erased first) | fulls at vsel and final | exists (lens E1) |
| **R10** | The halting head as verifier | AUC(q_halt, exact) at the last step; q-halt accuracy; precision / recall at 0; reliability curve | any cold run of a cell with a Q head (field cells, the DEC) | exists on the field harness (q_by_step); our evaluator does not record q → **build B1b** (`--record-q`) |

### 2.2 Checkpoint-level instruments (need the model; Mac CPU or pod; `tools/suite_ckpt.py` + the standing tools)

| ID | name | definition | set / protocol | status |
|---|---|---|---|---|
| **C1** | Decoder dynamics (E2) | per outer step on cold trajectories: non-given cells correct, readout entropy (/ln 9), commitment (p > .9), confidently wrong, the flip spectrum (to correct / to wrong), monotone solves (a decided-correct cell never flips back), un-peel events, the syndrome (violations of the argmax grid) and its late oscillation, first_exact | strat-512 at t = 64 (pod), strat-128 (Mac) | exists (`analyze_finalA_ecc.py` E2 → `suite_ckpt.py --mode dyn`) |
| **C2** | Decimation quality at stalls (E3) | committed cells (p > τ ∈ {.9, .99}) handed to the peeling decoder as hard decisions: committed fraction, wrong-among-committed, P(all committed correct), peeling solves / stuck / contradiction | stalled puzzles of C1's set | exists (same tool) |
| **C3** | Calibration at stalls (E6) | on stalled puzzles: top-k (k = 5) most confident cells' correctness vs mean confidence (the gap), entropy at step 1 and at the stall, confidently-wrong fraction | strat-512 | exists (`stall_calibration.py`) |
| **C4** | Explosion census | fraction of cold trajectories non-finite or with |z| > 1e6 by t = 64 (n 512) and t = 256 (n 64) | strat-512 | exists (`explosion_census.py`) |
| **C5** | Retention and the ε-ladder | the solution injected into the state — natives: y := solution one-hot (`--init solution`, final-map variant); field-class cells: z_H := the cell's own embedding of the solution (TRM's tok_emb; the DEC's role table, `embed_answer`) — then exact at every step for 8 steps (retention); the same from ε-corrupted solutions (ε ∈ {.05, .1, .2, .4, .6, .8}) → S(ε) and the graded first-failure rung | strat-512 | natives: exists (`probe_sudoku.py`, evaluator `--init solution`); field cells: exists on the field harness (`retain`), **build B1c** in our evaluator (`--init-latent solution --corrupt-eps`) |
| **C6** | Fixed-point census | latent residual at the t = 16 and t = 64 endpoints, solved vs unsolved; finite-difference Jacobian spectral radius of the outer-step map at the endpoint (8 power iterations); converged-wrong fraction (residual ≤ the solved median on an unsolved endpoint) | strat-512 (pod) / strat-128 (Mac) | exists (`analyze_sportC1_lensG_dynamics_fd.py`, field `jac`) → fold into `suite_ckpt.py --mode jac` (**build B3c**) |
| **C7** | Init-basin radius | exact@D vs ε for z0 = trained init + ε·N(0, 1), ε ∈ {0, .03, .1, .3, 1, 3} (ε = 0 the cold pass; ε → ∞ the random draw); the ε at which cold halves | strat-512 at D16 | exists on the field harness (`initrad`); **build B1d** in our evaluator (`--z0-mode perturb --z0-eps`) |
| **C8** | Symmetry defect | the digit-relabeling orbit: 9 random relabelings (+ transpose at p .5) of every puzzle; orbit-consistency (all 9 agree), any-of-9, orbit-vote gain (Ren & Liu's +18.2 pp on HRM); exactness control = 0 by construction on S9-exact cells (`test_dec_exact_s9`) | strat-512 (Mac), the 20k scan set at D16 (pod) | exists on the field harness (`sym`); **build B1e** (`--digit-perm`) for our evaluator (field cells X0/X1/X2; the DEC and natives read 0 by construction — the control row) |
| **C9** | Prefix inertness | the puzzle-prefix tokens zeroed → Δ exact (HRM −55.9 pp; TRM inert) | strat-512 → 20k | exists on the field harness (`prefix`); **build B1f** (`--zero-prefix`, trm cell only) |
| **C10** | Numerics floor | bf16 vs fp32 forward: rates and per-puzzle agreement (TRM-pub: 79.13 vs 79.12, 95.4 % agreement) | the 20k scan set at D16 | exists (protocol; `JAX_DEFAULT_MATMUL_PRECISION`) |
| **C11** | Test-time protocol rows | on ONE checkpoint: (a) cold single pass; (b) B = 1 random-init draw; (c) verified@k; (d) Top-1-by-residual@k (EqR); (e) EqR's top-k convergence selection; (f) PTRM's protocol: K noisy rollouts (σ = .3 Gaussian on the latent input at each outer step) with Q-head selection = best-Q@K and pass@K; (g) unverified majority@k | the 20k scan (k128) / SUB5k (K100 for PTRM) | (a)–(e), (g) exist; (f) **build B1g** (`--noise-rollouts σ --select q`) |
| **C12** | Train-set memorization readout | cold solve on the model's own 1k training puzzles (EqR seed-42 set exists; ours = the seeded 1k of the corpus); given-violating recalls on test | the model's train 1k; the 20k test | exists (field `train`; our corpus builder) |
| **C13** | Flux / throat (priced natives only) | I_s, A_s per cut from the probe trace; the throat; the profile | strat-512 via `probe_sudoku.py` | exists (natives); not applicable to field-class cells (no channels) — labeled ARC-side |
| **X1** | Reference decoders | peeling (naked + hidden singles), undamped exact-factor BP (40 it), BP + guided decimation; **build B6**: a DFS/backtracking classical upper rung | REF500 (rating-stratified) + the natural 20k for peeling | exists (`reference_decoders.py`; lens E0) |

### 2.3 Trajectory-level instruments (from `metrics.jsonl`; $0; `tools/suite_trajectory.py`, **build B3b**)

| ID | name | definition |
|---|---|---|
| **T1** | The trajectory law | segment/train CE at 5k…50k and its minimum step; train exact; halt fraction and mean segments (ACT loops); the monitor's peak step and value; vsel − final; EMA vs raw at the selected grid; the memorization onset (CE < .05 / train exact > .9); the low-givens erasure (R9) |
| **T2** | The selection instrument | monitor resolution (n puzzles), the tie set at the maximum, the tie-break rule, the offline re-selection; the 512-puzzle monitor (**build B5**) with an earliest-tie rule and the raw monitor as the second key |
| **T3** | Stability monitors and curvature | η / η_z, λ_J, retfm / ret_sched (natives); the Adam-v participation ratio PR/n and k90 (curvature concentration); the attention/stream flux A/I on priced cells |

### 2.4 Compute instruments (`tools/mac_count.py`, **build B3d**)

| ID | definition |
|---|---|
| **M1** | parameter count (pinned by test) |
| **M2** | analytic MACs per outer step and per puzzle at D: TRM block = token SwiGLU (S tokens, expansion 4) per channel + channel SwiGLU (w, expansion 4) per token, 2 blocks per pass, H × (L + 1) = 21 passes per outer step; the DEC = 9 fields × (the block at S = 81 + the (w, w) coupling); natives from the layer census. Verified against `jax.jit(...).lower().compile().cost_analysis()` on the real graph |
| **M3** | measured wall per puzzle-step on the pod from the evaluator's timestamps (it/s printed by the trainer reads 2.8× high on the field loop — never printed rates) |
| **M4** | accuracy-per-MAC: cold at (D, cell) vs MACs per puzzle — the curve, not a point (w192 / w256 / w384 / w512 DEC points + X0 at D16 / D64 / D128) |

## §3. Sets, seeds and protocol columns (binding for every reading)

- **FULL** = the 422,786 test puzzles (the headline's set). **SCAN20k** = the seeded 20,000 natural subsample (subsample seed 20260822) — identical on every scan since rung 2b, every field-checkpoint reading and every finalA X0-class scan; **SUB5k** = the seeded 5,000 of it (the Mac study's draws). **STRAT512 / 256 / 128 / 84** = `stratified_subsample(test_rating, n, 20260821)` (64 per rating octile at 512) — the screens', the calibration's and the dynamics' set. **REF500** = the reference decoders' rating-stratified 500 (seed 20260905). **TRAIN-1k** = the model's own training puzzles (EqR seed 42 reproducible; ours the seeded 1k of `sudoku_extreme_seed0.npz`; HRM unknown; alphaXiv's builder seed checked at run time). **ORBIT-512** = STRAT512 × 9 relabelings (seed 20260905).
- **Draws:** per-(puzzle, draw) seeding (mi_seed 4242) → nested k-curves, shard-invariant; k = 128 on SCAN20k at t = 64 (the registered column); k = 256 on STRAT512 at the headline depth (screens); the draw distribution per cell class as in §2's conventions, with EqR's truncated normal as a labeled variant (`--z0-mode trunc`, **build B1d**).
- **Depth:** D16 = the headline (EqR's base column); D64 = the depth row; D128 / D256 riders on SCAN20k. **Weights:** EMA headline where trained with EMA; raw = the alt row. **Selection:** ours val-selected on a train-file monitor (the 512-puzzle monitor from the next night; the 64-puzzle monitor's limit is labeled on every earlier arm); the field's = "best test checkpoint" (a comparator column); public checkpoints as released. **Numerics:** the field's bf16 matmul on the pod for field-class cells with an fp32 control on SCAN20k (C10).
- **Every cross-system number carries:** the protocol column (single pass / B = 1 / verified@k / residual@k / majority@k / PTRM best-Q@K), the training-regime column (data × aug, wd, lr, batch, EMA, normalization, ACT), the selection column, the compute column (M1–M3), the reproduction column (paper / reproduction / ours), and the numerics floor.

## §4. The frontier pod protocol (the comparator rows at the full protocol)

### 4.1 Models and routes

| checkpoint | params | route | why |
|---|---|---|---|
| TRM-alphaXiv (`step_32550_sudoku_epoch50k`) | 5.03M | **JAX port → our evaluator on the TPU** (`tools/port_field_ckpt.py`, **build B2**): the weight conversion of `verify_port.py` (token rows shifted, prefix row 0, heads transposed, their H_init/L_init) into a `ckpt_latest.pkl` in our format; verified fp32 step-1 max |Δlogit| ≤ 2e-3 (relative 1e-4) and D16 exact agreement ≥ 95 % (the chaos caveat, control = torch fp64 vs fp32) | the identical code path, records and sets as our arms; the whole standing battery for free |
| TRM-CGAR (`pytorch_model.bin`) | 5.03M | JAX port (same keys) | same |
| EqR-pub (EMA shadow) | 5.03M | JAX port: the TRM-MLP block + λ_ = .95 ↔ our `trm_lambda` .05; per-pass noise β = 0 (the paper's B = 1 column) and .5 (their released eval; labeled); `--z0-mode trunc` for their reset | same; the noise arc is the protocol finding of the Mac run |
| HRM-pub | 27.3M | **the torch harness on a GPU VM** (`tools/field_ckpts/run_field.py --device cuda`; an L4 or A100 spot VM, ≈ $0.35–1.3/h): a different architecture (H/L modules, attention + RoPE, one prefix token); a JAX port is ≈ 1 day and buys nothing the harness lacks | the harness already implements every mode at the Mac's reduced protocol; the GPU lifts it to the full protocol |
| X0 (ours, sportC1) | 5.04M | already at the full protocol | the port-integrity control (the public EqR reads like X0 on every instrument: Jaccard .886) |
| TRM-alphaXiv on BOTH routes | | the JAX port on the TPU and the torch harness on the GPU on SCAN20k at D16 and on the draws | the cross-route row: per-puzzle agreement ≥ the bf16 floor (95 %) |
| SE-RRM (code, no weights) | 2M | **optional, GPU:** their training recipe (1k × aug 1000, batch 272, 10k epochs ≈ 37k steps) reproduced on a GPU VM (≈ 6–12 h, ≈ $10–20), then instrumented through the harness (their architecture needs its own loader, ≈ 0.5 d) | the closest published relative of the DEC; a reproduction row of 93.7 with our instruments (retention, selector, orbit, dynamics) — decide at the freethink |
| PTRM / CMM / GRAM | — | no code: cited; PTRM's PROTOCOL reproduced on the alphaXiv checkpoint (C11f) | the test-time-protocol column |

### 4.2 The battery per model (v6e-8; walls inferred from X0's measured paces on the 8; one-shot idempotent evals with 300 s partial banking)

| eval | set / protocol | wall (8 chips) | reads |
|---|---|---|---|
| cold FULL D16 (+ exact-by-step) | 422,786 | 8 min | R1–R4, R10 |
| cold FULL D64 | 422,786 | 30 min | R1–R4 depth row |
| cold SCAN20k D128 (+ D256 on 5k) | 20k | 5 + 5 min | R4 |
| scan SCAN20k k128 t64 | 20k × 128 | 30 min | R5–R7 |
| screen STRAT512 k256 D16 + majority | 512 × 256 | 20 min | R5–R7 at the screen protocol (the DEC's comparable column) |
| dynamics STRAT512 t64 (+ E3) | 512 | 5 min (CPU-side stats from per-step records) | C1, C2 |
| calibration STRAT512 | 512 | 5 min | C3 |
| census STRAT512 t64 / 64 at t256 | | 5 min | C4 |
| retention + ε-ladder STRAT512 | 512 × 7 | 5 min | C5 |
| fixed-point census STRAT512 (residuals + FD Jacobian) | 512 | 10 min | C6 |
| init radius STRAT512 × 6 ε at D16 | 3,072 | 5 min | C7 |
| orbit ×9 SCAN20k D16 | 180k | 10 min | C8 |
| prefix zeroed SCAN20k D16 | 20k | 2 min | C9 |
| bf16 / fp32 control SCAN20k D16 | 20k | 3 min | C10 |
| PTRM protocol SUB5k K100 σ.3 D64 (alphaXiv; EqR optional) | 5k × 100 | 40 min | C11f |
| EqR noise-.5 variants (FULL D16/D64, scan) | | +45 min | the noise arc |
| train-1k readout (EqR; alphaXiv if its seed reproduces) | 1k | 1 min | C12 |
| **per TRM-class model** | | **≈ 2.5–3 h** | |

Three TRM-class models ≈ 8–9 h on a US v6e-8 ≈ **$60–70 + weather**; the DEC's five wide-arm scans first (≈ 1 h, pass two of Night A) and A3/A7's D128 rows (10 min); HRM on a GPU VM at the same battery ≈ 6–10 h at ≈ $0.35–1.3/h ≈ **$5–15**; SE-RRM optional ≈ $10–20. **Frontier total ≈ $80–110.**

### 4.3 Ops

One v6e-8 in the US first (the one-pod policy), `chain_frontier.sh` = the eval helpers of `chain_final.sh` (eval_one / eval_sharded / census_one / calib_one with EVAL_TIMEOUT, idempotent markers, the live bank of partials) over a static job list, no training; the offline harness with one scenario per eval kind and the negative (a timed-out eval); the canary on the DEC scan at `--batch 128` on ONE chip as the first job (the deadlock diagnosis at the source: `py-spy dump` if it hangs; then `--batch 64`); the GPU VM runs the torch harness's `night_gpu.sh` pattern with `--device cuda` and banks `out/*` to `gs://qhrrn2-rescue/frontier/`; the analyzer for this run is descriptive (`analyze_frontier.py`, the field study's `analyze_field.py` generalized to the full protocol + our records) with pre-registered predictions (§4.4) and NO champion rule.

### 4.4 Predictions to lock at the frontier registration (drafted; the freethink finalizes)

From the Mac run, extrapolated to the full protocol (credence): HRM FULL D16 ∈ [53, 57] / D64 [58, 62] (75 %); alphaXiv D16 [78, 80] / D64 [82.5, 84.5] (75 %); CGAR D16 [85.5, 86.7] / D64 [91, 92.5] (70 %); EqR noise 0 D16 [86, 87.5] / D64 [92.5, 93.7] (70 %); the JAX-port rows within the bf16 floor of the torch rows (95 % per-puzzle agreement, rates within .3 pp) (70 %); spurious rates at k128 EqR ≤ 1 %, alphaXiv 2–5 %, CGAR 20–30 %, HRM ≥ 10 % (60 %); retention 1.00 on the three TRMs, HRM < .05 at step 1 (75 %); orbit-consistency 50–80 % on all four with any-of-9 gains ≥ 10 pp (65 %); halting AUC ≥ .995 on all four (75 %); the D128 rows: alphaXiv + 1–2 pp over D64, EqR + 0.5–1.5, HRM + 1–3 (55 %); PTRM's best-Q@100 on alphaXiv within 2 pp of pass@100 and ≥ 95 (50 %); the depth-regression count ≤ .1 % on every model (70 %); every public model DECIMATING at D64 except HRM MIXED (70 %).

## §5. The corpus-wide backfill on the banked data (the $0 Mac program, launched 2026-09-07)

### 5.1 What is on disk (from `inventory_20260907.txt`)

- **694 eval artifacts** across 13 Sudoku campaigns: sport2 (S0–S7, d16 wave 1: fulls t6/t64, strat k16), sport2w2 (W1–W13), sport3a (A2–A10 + seeds), sportB (B1–B5 d64: fulls + screens), sportBr2 (C1–C4 d96), sportBr2b (D1–D4, C3X: + the 20k scans of C3X / D4), sportC0 (P1–P6: 20k k128 scans + k256 screens with majority), sportC1 (A0, A1, B0, B1, R0, X0, X0n: fulls at vsel/final/alt, 20k k128 scans, screens, censuses, the correct-grid riders), sportC2 (W0, R1, R2, R3, R4, X1, X2: + calibration), finalA (A0–A8: fulls at D16 vsel/final/alt + D64, the three X0-class scans, k256 screens with majority, censuses, calibration), the d3demo cells (B2-d64, S5-d16 at k128 with majority) and the canvas sel-5k riders.
- **Per-draw records (mi_exact_k / mi_resid_k):** 24 scans at k128 on the identical 20k (sportC0 P1–P5, sportC1 ×8 incl. the correct grids, sportC2 ×7, finalA A0–A2) and 30+ screens at k256 with the unverified majority (sportC0+, finalA); the rung-2b and earlier screens carry first-hit bits only.
- **Checkpoints with `ckpt_latest.pkl`:** ≈ 170 — the ARC-era rungs (pretrain1…13f, r1/r1b/r1c: ~70) and the Sudoku era (sport2 8, sport2w2 13, sport3a 16, sportB 6, sportBr2 5, sportBr2b 5, sportC0 8, sportC1 11, sportC2 11, finalA 8), with banked grids and 2k monitors from sportC1 on.
- **The Mac field study:** 44 output dirs (`runs/field_ckpts/out/`): cold SCAN20k D64 (+ EqR noise 0/.5, TRM bf16), STRAT512 D16, draws SUB5k k8 D16 (HRM also SUB1k k2 D64), dynamics STRAT256 D64, retention STRAT512 D8, orbit STRAT512 D16, prefix STRAT512, init radius STRAT512, Jacobian STRAT84 D64, EqR train-1k.

### 5.2 Record-level backfill — `tools/suite_records.py` (this session; `runs/analysis/suite_records_20260907.{txt,json,csv}`)

One row per (campaign, arm, eval): R1 (g50 / width / in-range flag, yield, propagation class, rating bands, givens bins), R2, R3, and for records with draws R5–R7 (ρ, b1, r_i spectrum, k50 / k90, rescue, AUC, spurious, t1r@k / verified@k, majority@k where recorded). The output is the corpus ladder through one lens: d16 (S5, W2, A-arms) → d64 (B2) → d96 (C3, D3, D4, C3X) → d128 natives (R0, B0c, R3, W0, R1, R2, R4) → the field cells (X0, X0n, X1, X2, A0–A2) → the DEC (A3–A8) → the public checkpoints (the Mac rows from `runs/field_ckpts/out`).

**First read (2026-09-07; 647 rows: 488 fulls, 26 scans, 113 screens, 2 riders, 2 demo cells, 16 public-checkpoint records; `suite_records_20260907.txt`).** The decoder class is set by the loop, not by the scale: every one of the ~60 native maps from d16 to d128 (78k → 3.0M parameters, T6 → T16, canvas and native, priced and free, every regime) sits in the SOFT class — an erasure threshold g50 between 25.7 and 38 givens at D64 and a search-class yield between 0 and 37.6 % (the d128 champion line R3 / B0 / B1 at 36.5–37.6; the d96 record D4 25.2; d64 B2 15.1; d16 S5 5.3) — while every field-loop cell reads DECIMATING or MIXED with g50 ≤ 20 or none and yields 73–98 % (X0 91.7, X0n 72.8, A0 93.5, A1 93.0, A2 90.8; the DEC A3 93.8, A4 97.2, A5 97.8, A7 97.7, A8 96.5; the public EqR 92.1, CGAR 90.4, alphaXiv 80.7, HRM 53.9), the two exceptions being the two field-loop cells whose operator or orbit was removed (X2 with our group mixer: 26.1 / 34.6; X1 without the digit orbit, memorized: 29.0 / 17.3). First-exact medians are 1 (p90 6–11) on every field-loop cell and 7–26 (p90 15–46) on every native. Selector rows: our RI + FPA natives ≤ .4 % spurious with t1r/verified ≥ .99 (P2, P3s1, P5, R4); the no-RI native P1 7.1 % / .56; the field cells 1.5–52 % by seed and recipe (X0 7.8, A0 38, A1 52, X1 1.5). The three orders of magnitude of the native ladder never crossed the class boundary; the loop did in one night.

### 5.3 Checkpoint-level backfill — `tools/suite_ckpt.py --mode dyn` (dynamics C1 + C2 on strat-128 at t = 64) as a background driver, prioritized

| priority | maps (grid) | why |
|---|---|---|
| P0 (done in the Night A lens) | finalA A3, A7, A5, A8, A0, X0 | the paper's frontier |
| P0′ | finalA A1, A2, A4; sportC2 X1 (20k grid), X2 (50k) | the field-cell ledger and the memorized field cell |
| P1 | sportC1 R0 (50k), B0 (A:20k), sportC2 W0 (A:50k), R1 (B:10k), R2 (A:40k), R3 (A:40k), R4 (20k), sportC0 P3s1 (final), P1 (20k) | the native champion line (the sportC2 lens ran these at strat-256; this pass puts them on the same n = 128 t = 64 protocol as the DEC) |
| P2 | canvas D4 (50k), C3X (30k), D3 (40k), sportBr2 C3 (20k), sportB B2 (50k), sport2 S5 (20k), sport3a A2 / A3 / A7 (50k) | the scale ladder and the mechanism arms (H-43 / H-45 dynamics as decoder dynamics) |

CPU walls: 3 min (d16) to 20 min (w512) per map; ≈ 25 maps ≈ 4–6 h → runs across the freethink. Calibration (C3) is missing on sportC0 / C1 / the canvas — the driver runs `stall_calibration.py` on the P1 / P2 grids after the dynamics (≈ 5 min each). Explosion censuses exist on sportC0+ (skip). C5–C9 on our cells wait for the evaluator builds (B1c–B1f), then run in the same driver.

### 5.4 Trajectory backfill — `tools/suite_trajectory.py` (**build B3b**) over every `metrics.jsonl` with monitors (sport3a on): T1's peak / onset / drop table across regimes — the trajectory law's corpus (H-45 / H-46 / H-49 / Night A in one table).

## §6. Builds (owed; ordered; estimates in Fable days)

| # | build | est. | consumer |
|---|---|---|---|
| **B1** | evaluator flags: (a) `--record-by-step` (exact bits per outer step → R4 from one run); (b) `--record-q` (halting logits → R10); (c) `--init-latent solution --corrupt-eps e` (C5 for field-class cells through `embed_answer`); (d) `--z0-mode {gauss,trunc,perturb} --z0-eps` (EqR's reset; C7); (e) `--digit-perm k` (C8; permute the puzzle, un-permute the prediction); (f) `--zero-prefix` (C9; trm cell); (g) `--noise-rollouts σ --select q` (C11f); (h) per-class `--batch` in the chain (the DEC scan) | 1.0 | frontier + backfill |
| **B2** | `tools/port_field_ckpt.py` (alphaXiv / CGAR / EqR → our ckpt format incl. EMA; the DEC-style config) + `tests/test_port_field.py` (fp32 step-1 agreement on 64 puzzles ≤ 2e-3) | 0.5 | frontier |
| **B3** | (a) `suite_records.py` ✓ (this session); (b) `suite_trajectory.py`; (c) `suite_ckpt.py` modes `dyn` ✓ (this session), `jac`, `calib`, `census` wrappers; (d) `mac_count.py` (M2 analytic + `cost_analysis()` check) | 0.75 | all three |
| **B4** | `chain_frontier.sh` + `harness_frontier.sh` (eval-only chain; the timed-out negative scenario; the DEC scan canary at `--batch 128`) + `analyze_frontier.py` with the §4.4 predictions | 0.5 | frontier |
| **B5** | the 512-puzzle train-file monitor + the earliest-tie rule + the raw monitor as the second key in `pretrain.py` / `select_ckpt.py`; screens at 2k multiples | 0.25 | the champion night |
| **B6** | the DFS / backtracking classical rung in `reference_decoders.py` (REF500) | 0.25 | Fig. 1 |
| **B7** | SE-RRM reproduction (optional; their code on a GPU VM; a loader in `field_models.py`) | 0.5 + GPU | positioning |
| **B8** | gradient accumulation for the w512 DEC (owed since Night A); the champion night's chain from `chain_final.sh` | 0.5 | the champion night |

## §7. Calendar and budget (proposal)

| day | work | $ |
|---|---|---|
| Sep 7 (today) | this design; the record-level backfill (done in the session); the dynamics backfill launched; the freethink on the suite → the frontier registration's predictions | 0 |
| Sep 8 | B1, B2, B3, B4 built and smoked; the frontier registration (ledger, predictions, harness green); launch the v6e-8 evening US window + the GPU VM for HRM | ≈ 80–110 |
| Sep 9 | the frontier analysis pass (descriptive; the comparator table at full protocol; the DEC's pass-two letters); the backfill consolidated; the second freethink → the champion night registration (B5, B8 built) | 0 |
| Sep 10–11 | the champion night (one v6e-8; seeds ×3 on the plain DEC, the aug-10000 lever, w192, the SE-RRM operator; the suite as the battery) | ≈ 200–300 |
| Sep 12 | the champion verdict; drafting | 0 |
| Sep 13–14 | the ARC runs (the d96 rung with the DEC's ARC form + the instrument back-port; the wd-1.0 arm) | ≈ 150–250 |
| Sep 15–16 | the ARC verdict; freeze | 0 |
| reserve | weather, a re-run, the SE-RRM reproduction | ≈ 250–350 |

## §8. The ARC back-port list (what the suite carries to paper 2's runs)

Retention on injected solutions (C5, the E3b class), the converged-wrong rate as the selector law's ARC end (R6 on multi-init limits: .92 wrong-stable), the commit / revision axis (C1's flip spectrum on ARC equilibrium maps), the depth ladder with regressions (R4: the two-sided iteration figure), the calibration gap at stalls (C3), the explosion census (C4), the trajectory law (T1 with the optimization-length tax), the compute column (M1–M3), and the DEC's ARC form (exact color-permutation × dihedral symmetry; the commit head as the verifier-free commit rule).

## §9. Adversarial checks on the suite itself (the ways a reading goes wrong, and the guard)

1. **Set identity:** every paired contrast asserts identical idx (the analyzer's gate); intersections are labeled with their n. 2. **The strat-512 ↔ 20k offset** is arm-dependent with both signs (C3 −7.9 / C3X −0.6 / D4 +4.5 pp): screen-protocol numbers never share a column with scan-protocol numbers. 3. **k = 8 vs k = 128:** ρ and spurious rates at small k are lower bounds; the pod run lifts every Mac row to k128. 4. **The residual's definition differs by cell** (latent vs y): the selector law is stated per cell class, and EqR's logit-delta score is reported beside ours on the field cells. 5. **The 64-puzzle monitor** (Night A's lesson): selection ties and the later-tie rule select the memorization side; every earlier arm's selection is labeled, the next night runs the 512-monitor. 6. **Chaos on unsolved trajectories** (10× per outer step): per-puzzle agreement across implementations has a ≈ 5 % floor; rates are the comparable statistic. 7. **The records-vs-summary vote convention** (cold ∪ draws vs draws-only): both named where they differ. 8. **The DEC scan deadlock:** batch × carry size; the canary at `--batch 128` before any wide-arm draw job. 9. **EqR's reset:** truncated normal (their code) vs N(0, 1) (ours): the `trunc` mode is a labeled variant and the Mac run reproduced their paper with it. 10. **The printed it/s** on the field loop is 2.8× high: M3 from timestamps only. 11. **Memorized checkpoints:** every instrument row names the grid (vsel / final / step) and the memorization state (T1) — a memorized grid's selector, retention and dynamics are a different object.

## §10. What the freethink decides (staged)

1. Which instruments become the champion night's decision rules (letters) and which stay descriptive: R1/R2 (class), R5–R7 (selector, coverage, majority), C5 (retention), T1/T2 (the trajectory and the selection instrument), M4 (the compute column) — and the noise floor from three seeds at their best grids.
2. The frontier registration's predictions (§4.4) and the SE-RRM decision (reproduce or cite).
3. The champion night's arm set under the $1,000 plan (§7): seeds ×3 of the plain DEC-w384; the aug-10000 position-orbit lever; w192 (1.34× X0's MACs); the SE-RRM operator; the D128 / D256 rows; what the ARC runs need from the same night (the wd-1.0 native arm? the DEC's ARC form smoke?).
4. The paper's figure list against the suite (Fig. 1–6 + the four tables) and which backfill rows each needs.
