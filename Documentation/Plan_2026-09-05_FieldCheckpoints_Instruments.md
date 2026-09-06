# Instrument portability on the field's OWN weights — public HRM / TRM / EqR Sudoku-Extreme checkpoints read through our suite (REGISTRATION, pre-data, 2026-09-05 ~21:30Z; Fable, parallel science session)

**Purpose (PI, 2026-09-05):** use the accumulated knowledge and the instrument suite on the public HRM, TRM and EqR checkpoints, to understand why their recipes work outside our codebase, generalize the measurement suite, and harvest paper-1 insights. This is Program Review #1's R4 and Review #2 §3's "load-bearing external-validity test". Descriptive study with PRE-REGISTERED numeric predictions; no champion rule; every number labeled measured / inferred; nothing here touches Night A.

## §1. The checkpoints (downloaded 2026-09-05 ~20:40Z; sources and sizes)
| tag | source | file | size | trained as | reported |
|---|---|---|---|---|---|
| **HRM-pub** | HF `sapientinc/HRM-checkpoint-sudoku-extreme` (official, 2025-07-21) | `checkpoint` (+ `all_config.yaml`) | 109.1 MB, 27,276,802 params (fp32; H_init/L_init bf16) | HRM ACT-v1: H_cycles 2, L_cycles 2, H_layers 4, L_layers 4, hidden 512, 8 heads, RoPE, puzzle_emb 512 (1 prefix token), halt_max_steps 16; data `sudoku-extreme-1k-aug-1000`, epochs 40000, batch 768, lr 1e-4, wd 1.0, seed 0 | README "nearly perfect"; TRM's Table: 55.0 (⚠ the discrepancy `Related_Work_Series.md` §6.1 names) |
| **TRM-pub** | HF `alphaXiv/trm-model-sudoku` (independent reproduction by alphaXiv, 2025-10-22; the official Samsung repo is archived without weights) | `step_32550_sudoku_epoch50k` (MLP-T) | 20.1 MB, 5,030,402 params (= our port's 5,037,058 minus the 3 unused damping scalars + vocab 11 vs 12 rows) | TRM-MLP: H_cycles 3, L_cycles 6, L_layers 2, hidden 512, puzzle_emb_len 16, no pos-enc, no_ACT_continue; alphaXiv README recipe: batch 1536, lr 2e-4, wd 1.0, 50k epochs (32,550 steps) | alphaXiv model card 79.37 ± 0.12 (paper 87.4 ± 2) |
| **EqR-pub** | HF `locuslab/EqR-model` (official, 2026-05-20) | `sudoku-extreme/eqr.pth` | 40.3 MB (model + EMA shadow; config embedded) | EqR on TRM-MLP: H_cycles 3, L_cycles 6, L_layers 2, hidden 512, puzzle_emb 512 × 16, λ_ = .95 (their convention: 95 % new state), noise .01 (train), RI trunc-normal std 1 at every reset; batch 768, 50k epochs, lr 1e-4, wd 1.0, EMA .999; data seed 42 | paper: 84.8 base / 86.0 +RI / 86.4 +RI+NI @D16 B1; 93.0 @D64 B1; 99.8 @D64 B128 Top-1-residual; README eval = "depth 64, breadth 128, convergence-based top-4 selection, Langevin noise scale 0.5" |
Also read for the comparator table (no weights): CMM 93.7 @5M (arXiv 2603.22871, no code found), PTRM 98.75 (2605.19943, no code found; "same checkpoints as TRM"), GRAM 97.0 @10M (new, 2026).

## §2. Protocol facts established from the code (binding for every reading)
- Token convention (HRM/TRM/EqR builders): vocab 11 = PAD 0, blank 1, digits 2..10; labels = solution + 1; test puzzle identifier 0 (`<blank>`), a single trained prefix embedding shared by all puzzles. Ours: blank 0, digits 1..9 → theirs = ours + 1. The test set is the same HF `sapientinc/sudoku-extreme` test.csv (422,786 rows, same order as our `test_q`).
- **HRM's 1k training subsample is NOT reproducible** (`np.random.choice` unseeded in `dataset/build_sudoku_dataset.py`); EqR's is (`--seed 42` → `np.random.seed(42)` before the choice over 3,831,994 rows); TRM-alphaxiv's = see the builder's seed (checked at run time; labeled).
- Evaluation semantics: HRM/TRM eval ALWAYS runs `halt_max_steps` outer steps (halting only in training) and scores the LAST step; `q_halt_accuracy` = P((q_halt ≥ 0) == exact) is their own verifier-calibration metric. EqR eval: every reset draws z_H, z_L ~ trunc-normal(std 1) (no deterministic cold start exists), per-pass Langevin noise (default 0.5 in the eval yamls; .01 at training), `different_init` B draws, convergence score = mean over the last 3 outer steps of the per-token L2 norm of the LOGIT delta, top-k selection by that score.
- Numerics: all three trained in bf16 forward; we run fp32 (labeled) and reproduce the headline in bf16 on the same subsample to bound the numerics effect.

## §3. Puzzle sets (identical to the lens's; idx into our `test_q`)
- **SCAN20k**: the 20,000-puzzle natural subsample (seed 20260822) = `runs/sxscan_psportC2W0/records_all.npz` idx — thresholds, yields, list decoding, selectors (X0's sportC1 scan and every native scan are on the identical set).
- **STRAT512** = `stratified_subsample(test_rating, 512, 20260821)`; **STRAT256** = the same call with 256 (the lens's E2/E3 dynamics set); **REF500** = the reference-decoder slice (`runs/analysis/reference_decoders.npz` idx).
- **TRAIN-EqR**: the exact seed-42 1k subsample of train.csv (re-derived with their builder's RNG call; row indices recorded).

## §4. Instruments (every one an existing suite member unless marked NEW), with the compute plan (Mac M5 Pro, Metal; fp32)
I1 Headline reproduction (their protocol): exact@D16 on SCAN20k (CI ±0.35 pp); EqR also @D64, noise ∈ {0, .5}, RI std 1 (B=1), plus their top-4 convergence selection at B=8 (their statistic at a smaller B, labeled).
I2 Erasure thresholds (E1): g50 + 10–90 width (logistic on cold vs givens), search-class yield P(cold | rating > 0), by-givens and by-rating bands.
I3 List decoding + selector (E4/E5) with k = 8 random-init draws (N(0,1) carries; for EqR its own trunc-normal): ρ@8, per-puzzle rate spectrum, rescue of cold failures, residual AUC / spurious rate / t1r@8 vs verified@8 — residual = our evaluator's definition (mean |Δ carry| over the last 3 outer steps) AND EqR's logit-delta score (both reported).
I4 Decoder dynamics (E2) on STRAT256 at t = 64: cells-correct curves, readout entropy, commitment (p > .9), confidently-wrong, monotone solves, un-peel events, flips, syndrome oscillation, first_exact; E3 decimation quality at stalls (τ .9 / .99 → peeling) and calibration at stalls (top-5 confidence vs correctness).
I5 Fixed-point census (lens G3): latent residual at t = 16 / 64 on solved vs unsolved; finite-difference Jacobian radius of the outer step at the endpoint on a strat-84 subset (X0: 1.07 solved).
I6 Solution retention (E3b / H-45 ported; NEW on the field's weights): z_H := their embedding of the SOLUTION (sqrt(hid)-scaled tok_emb), z_L := L_init; 8 outer steps; retained = exact at every step. Ren & Liu's "fixed-point violation" made a number.
I7 Depth scaling: exact at D = 16 / 32 / 64 with per-puzzle regressions counted (X0: +6.8 pp, 0 regressions).
I8 Halting head as verifier (NEW as a table row): AUC(q_halt logit, exact) at the last step; their q_halt_accuracy; precision/recall at 0; reliability curve.
I9 Symmetry defect (NEW; the C2 exactness in their units): STRAT512 × 8 random digit relabelings (+ transpose): P(solve) mean, orbit-consistency (all 9 agree), orbit-vote gain (Ren & Liu: +18.2 pp on HRM-27M).
I10 Memorization readout: cold solve on TRAIN-EqR's 1k (EqR only, labeled; HRM's set unknown); given-violating recalls on test (X1's 5.9 % signature).
I11 Puzzle-prefix inertness: prefix zeroed → Δexact on STRAT512.
I12 Port verification: alphaXiv TRM weights converted into our JAX `trm_cell` (tok_emb rows shifted by one, prefix row 0 = their vector, rows 1–15 zero; lm_head/q_head transposed; mlp_t/mlp gate_up/down transposed; H_init/L_init from the checkpoint) → fp32 logits on 64 puzzles at D16 vs their PyTorch forward.

## §5. Predictions (locked pre-data; credences in parentheses)
- P1 HRM-pub exact@D16 on SCAN20k ∈ [48, 62] (60 %) — TRM's 55.0 stands; "nearly perfect" was another protocol/set. ≥ 90 (10 %) would falsify TRM's table row.
- P2 TRM-pub exact@D16 ∈ [77, 82] (75 %); @D64 ∈ [82, 90] with ≤ 0.5 % regressions (60 %).
- P3 EqR-pub @D16 B1 (noise .5) ∈ [82, 90] (60 %); @D64 ∈ [90, 95] (60 %); |noise 0 − noise .5| ≤ 2 pp at D16 (55 %); top-4 convergence at B=8 @D64 ≥ 96 (55 %).
- P4 Decoder class (R-A-4 letters, descriptive here): TRM-pub and EqR-pub DECIMATING (no g50 in 17–35; yield ≥ .70) (70 %); HRM-pub MIXED (a g50 inside the range, yield in [.45, .70]) (55 %).
- P5 Dynamics: commit@step1 ≥ 85 % of non-given cells on all three (75 %); monotone solves ≥ 40 % (65 %); confidently-wrong at stalls ≥ 30 % (X0: 47 %) (65 %) — the hard-decision class of X0 is the class of the public weights too.
- P6 Selector: TRM-pub spurious rate per wrong draw ∈ [3, 12] % (60 %), EqR-pub ≤ 1 % (65 %) → t1r/verified@8: EqR ≥ .98, TRM ≤ .96 (60 %) — the H-37/D3 law on the field's own weights.
- P7 Retention (I6): EqR ≥ .95 (70 %); TRM ∈ [.60, .95] (55 %); HRM < .80 (60 %).
- P8 Fixed points: EqR's median solved-endpoint latent residual ≤ 1/5 of TRM-pub's (60 %); FD Jacobian radius on solved endpoints TRM-pub ∈ [0.95, 1.15] (60 %), EqR < 1 (55 %), HRM > 1 (55 %).
- P9 Halting head: AUC ≥ .90 for TRM-pub and EqR-pub (70 %); HRM-pub ≥ .80 (55 %).
- P10 Symmetry: HRM-pub orbit-vote gain ≥ +10 pp with orbit-consistency < .80 (60 %); TRM-pub / EqR-pub gain ≤ +5 pp (60 %).
- P11 EqR train-solve on its own 1k ≥ .99 (70 %); given-violating recalls on test ≤ .2 % for all three (70 %).
- P12 Depth: TRM-pub / EqR-pub regressions D16→D64 ≤ 0.5 % (60 %); HRM-pub regressions ≥ 2 % (55 %).
- P13 Puzzle prefix inert on all three (|Δ| ≤ 1 pp; 70 %).
- P14 Port: our JAX port reproduces the alphaXiv PyTorch fp32 logits to max |Δ| ≤ 1e-3 and ≥ 99.5 % exact-match agreement on 64 puzzles (75 %).
- P15 Cross-cell: the public TRM-MLP at 79 sits ABOVE every native of ours on search-class yield and BELOW X0 on the SAME 20k (85 %); its g50 is absent or ≤ 20 (60 %).

## §6. Outputs
`runs/field_ckpts/out/<tag>/…` records in the evaluator's format (`records_all.npz` with idx / cold_exact / first_exact / mi_first_hit / mi_exact_k / mi_resid_k), per-set summaries, and `runs/analysis/field_ckpts_<date>.{txt,json}`; report `Documentation/Report_<date>_FieldCheckpoints_Instruments.md`; ledger §5 entry at the checkpoint; the harness promoted to `tools/field_ckpts/` at that checkpoint. Compute: Mac only (the pod is Night A's); every reduced protocol (k = 8, subsample sizes) labeled beside the number.

## §7. AMENDMENT (2026-09-05 ~21:25Z, pre-data for this checkpoint): a fourth public checkpoint — TRM-CGAR
`Kaleemullah/trm-cgar-sudoku/pytorch_model.bin` (20.1 MB; HF config.yaml = the standard TRM-MLP arch trained batch 768, lr 1e-4, wd 1.0, 50k epochs with CGAR's depth curriculum + supervision decay .7; card 86.02 %). Same instruments (tag `trmc`), same sets. Predictions: **P16** exact@D16 on SCAN20k ∈ [84, 88] (70 %); **P17** its decoder class DECIMATING and its dynamics signature (commit@1, monotone, churn) within 10 pp of TRM-pub's — training quality moves yield and thresholds, not the class (60 %); **P18** spurious rate ≤ TRM-pub's (55 %); **P19** the two TRM-MLP checkpoints' solved sets nest: P(CGAR solves | alphaXiv solves) ≥ .95 (65 %). Nothing else in the registration changes. The knowledge base for the surrounding literature = `Note_2026-09-05_Field_KnowledgeBase.md`.

## §8. EXPLORATORY ADDENDUM (2026-09-06 ~09:50Z; post-hoc, labeled — motivated by a result, so NOT a registered prediction)
The k=8 draws read HRM-pub's single random-init draw at 3.2 % (cold 54.6 %): the public HRM has no random-init capability at all (our P1-class), while TRM-pub (b1 77.8 vs cold 79.2) and EqR-pub (76.8 vs 77.4) are init-invariant. Two exploratory lenses follow: (a) **init-basin radius** — exact@16 on STRAT512 vs the size ε ∈ {0, .03, .1, .3, 1, 3} of a Gaussian perturbation of the trained fixed init (ε = 0 the cold pass, ε → ∞ the random draw), for HRM / TRM / CGAR / EqR(noise 0); (b) HRM random-init draws at D = 64 (k = 2, 1k puzzles): does the random-init trajectory recover with depth? Readings enter the report as exploratory lenses, never as scoreboard items.
