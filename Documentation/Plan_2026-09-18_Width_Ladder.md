# The width ladder and the attention-mixer arms — REGISTRATION (2026-09-18; built and harnessed; nothing launched until the PI's go)

**THE PI'S GO (2026-09-18 ~11:00Z, to the ops session):** "There are runs we need for the paper that's already registered. I want this session to run the ops on it." The ops session re-ran every gate before the registration commit (§5) and launched both pods on that commit; §3 and §4 lock there, the analyzer is frozen there.

## Page one: the goal, the expected accuracy, the cost (the PI confirms this page before any build)

**The PI (2026-09-18):** "We already have w192 symmetric DEC. Let's just do it with 30k training budget and evals on a 5k subsample for the width ladder as we can't keep spending much more on the sudoku project ... SE-RRM 128 and 192 is also okay. Let's do it over two v6e-8 pods."

**Goal: a MEASUREMENT for paper 1, not a better score.** A reviewer will ask why a 0.8M symbol-equivariant model beats SE-RRM's 2M one. Read at the source (2026-09-18), SE-RRM is at parity with our 2.8M width-384 model at every iteration count (98.2 vs 98.2 at 64), so our answer is "the narrowing", which today rests on ONE contrast (192 vs 384, three seed pairs, +0.5 to +1.3 at 64). The ladder asks whether accuracy against hidden size has an interior optimum near 192 on our model, and the mixer arms ask whether an attention-mixed symmetric model (SE-RRM's design) narrows the same way. No headline number changes whatever these read.

**Lane 1 (pod 0), the width ladder:** the symmetric DEC under the registered recipe, seed 0, FIXED 30,000 steps (no extension), hidden size 96, 128, 256. Existing points complete it at no cost: 192 (seed 0's banked grids up to 30,000, one eval-only row) and 384 (the 30k-budget seeds). Evaluation on the identical 5,000 test puzzles only: 16 and 64 iterations from the fixed start, and the 32-restart scan (random-start single pass, selected, verified). Selection as registered (512 validation puzzles, earliest tie).
**Lane 2 (pod 1), the mixer arms:** the same state, loop, recipe, seed, budget and rows with the 81-cell MLP mixer replaced by self-attention over the cells with 2D rotary positions, and attention across the nine fields in place of the mean (both SE-RRM's choices), at hidden size 128, 192 and 256. This is OUR reimplementation of their mixers inside our loop and recipe, labeled so everywhere; 256 is the fidelity anchor against their published 93.7 / 98.2.

**Expected accuracy (16 / 64 iterations, 5,000 puzzles; one seed, so ± 1.1 at 16 and ± 0.5 at 64 is noise):** width 256 ≈ 95 / 98.5; width 128 ≈ 94–96 / 98.5–99.2 (credence 0.35 that it beats 192 at 64); width 96 ≈ 90–95 / 97–99 with a real risk of the narrow model's instability inside 30k (the width-192 seeds left their basin from the fixed start between 14k and 28k; that is why the random-start row is in the battery). Mixer arms: 256 ≈ 92–95 / 97.5–98.5 if the port is faithful; 0.5 that 192 ≥ 256 at 64. For reference: ours 95.4 ± 0.5 / 99.0 ± 0.2 (50k budget), SE-RRM 93.7 / 98.2, EqR's released weights 86.5 / 93.2.
**What one seed can and cannot show:** a 2-point collapse at 96, or a monotone trend across five widths, yes; a 0.5-point difference between neighbours, no. The paper would report the curve as one seed per new width beside the seeded 192 / 384 points.

**Cost and time (planned on WALL pace, rule 18d):** ≈ 1.5–2.5 h per arm on a spot v6e-8 (training 0.6–1.5 h + the reduced battery); three arms per pod ≈ 6–7 h each, in parallel; ≈ $100–120 at the Mumbai rate plus churn, cap $150. Mac: the 192 / 384 points restricted to the 5,000 ids from banked records, $0.

**Build before launch (≈ half a day):** width and mixer arms + a reduced-battery knob in the chain (defaults inert), the attention mixer in the cell with its exact-equivariance test, harness scenarios, the CPU smoke with the launch env, a frozen analyzer with its selftest, the pace probe as the first stage on the chip.

## The PI's decisions on page one (2026-09-18, before the build)
1. The SE-RRM route: **our JAX reimplementation** of their mixers inside our loop and recipe, at hidden 128, 192 and 256 (256 = the fidelity anchor). Their released PyTorch code needs a GPU (quota zero).
2. Page one **confirmed with width 96 dropped**: the ladder's new points are 128 and 256.

## 2. The arms (one variable each from the registered champion recipe at seed 0; `tools/chain_champ.sh` `arm_flags`)
| arm | hidden | mixer over the cells | across the fields | params | pod |
|---|---|---|---|---|---|
| W256 | 256 | the 81-cell SwiGLU | mean | ≈ 1.3M | 0 |
| W128 | 128 | the 81-cell SwiGLU | mean | ≈ 0.4M | 0 |
| SA128 | 128 | self-attention, 2D rotary, 4 heads of 32 | attention, 4 × 32 | ≈ 0.5M | 0 |
| SA256 | 256 | self-attention, 2D rotary, 8 heads of 32 | attention, 4 × 32 | ≈ 1.8M | 1 |
| SA192 | 192 | self-attention, 2D rotary, 6 heads of 32 | attention, 4 × 32 | 1,057,925 (the CPU smoke) | 1 |

Everything else is the champion recipe verbatim (TRM's loop H 3 / L 6 / 2 blocks, EqR's damping 0.05 and noise 0.01, randomized-init σ 1, anchor rows k 1 / ε 0.2 / a quarter of the batch, 1,000 position copies, stablemax, batch 768, AdamW lr 1e-4 after 2,000 warm-up steps, wd 1.0, β2 0.95, EMA 0.999, ACT in training only). **Fixed 30,000 steps, never extended** (`fixed_budget`). Selection: the 512 validation puzzles every 2,000 steps, the maximum with the earliest tie, the raw monitor as the second key. **The reduced battery (`LADDER_BATTERY=1`)** on the selected grid, EMA weights, the identical 5,000 test puzzles (`--subsample 5000`, seed 20260822): 16 iterations and 64 iterations from the fixed start with the exact bit per step, and the 5,000 × 32-restart scan at 64 iterations (one random start, selected by convergence, verified). No full-set row, no 128 / 256-iteration row, no census, calibration or screens.
**References at no pod cost:** width 192 = seed 0's grid selected inside 30,000 steps (`runs/pretrainchamp_C5/ckpt_028000.pkl`), read on the Mac with the real evaluator at 64 iterations with the exact bit per step on the same 5,000 (f32 on the CPU against bf16 on the chip: labeled; the paper's measured numerics floor applies); width 384 = seed 0 of the 30k-budget triple, its banked full-set / 100k records intersected with the 5,000 ids.
**What the port is and is not.** SA* = SE-RRM's two mixers (their paper and README, read 2026-09-18) inside OUR block, loop and recipe. NOT ported: their layer layout (two position + two symbol layers per block), hidden-256 MLP ratios, dropout 0.2, lr 5e-4, batch 272, the 5 % random early stop, 10,000 epochs. The paper calls it "our reimplementation of SE-RRM's mixers"; R-WL-3 says how close it lands to their published numbers.

## 3. The rules (locked; `tools/analyze_wladder.py`, selftest 16/16, frozen at the registration commit)
Floors = twice the largest seeded spread of this recipe: 2.58 pp at 16 iterations, 2.44 pp at 64. INTEGRITY (config / seed / fixed budget / one grid = the selection / n 5,000 / identical ids) gates every letter. R-WL-1 each new width against the width-192 reference, paired: INSIDE / BEYOND. R-WL-2 the shape at 64 over 128 / 192 / 256 / 384: INTERIOR-192 / NARROWER-BETTER / WIDER-BETTER / MIXED, RESOLVED only if the range exceeds the floor. R-WL-3 the port's fidelity: SA256 within the floors of SE-RRM's 93.73 / 98.22 → FAITHFUL, else ABOVE / BELOW per depth. R-WL-4 narrowing on the attention-mixed model: SA192 − SA256 and SA128 − SA256 at 64. R-WL-5 the mixer at matched hidden size. STABILITY descriptive (selected step, validation maximum and end, the fixed start against one random start).

## 4. Predictions (credences; written before any run)
| | prediction | credence |
|---|---|---|
| P1 | INTEGRITY PASS on all five arms | 0.85 |
| P2 | W256 INSIDE at both depths against width 192 | 0.75 |
| P3 | W128 INSIDE at 64; W128 ahead of width 192 at 64 (any margin) | 0.70; 0.35 |
| P4 | R-WL-2 reads UNRESOLVED-AT-ONE-SEED | 0.70 |
| P5 | the shape letter, if any: INTERIOR-192 0.40 · NARROWER-BETTER 0.30 · MIXED 0.20 · WIDER-BETTER 0.10 | |
| P6 | R-WL-3 FAITHFUL | 0.50 (BELOW@16 0.35: no dropout, another layer layout) |
| P7 | SA192 NARROWER-AHEAD of SA256 at 64 (any margin); BEYOND | 0.55; 0.10 |
| P8 | the attention mixer BELOW the 81-cell SwiGLU at 16 iterations at matched size on at least two of three sizes | 0.60 |
| P9 | a fixed-start excursion (validation EMA under 60 % after first passing 85 %) on at least one narrow arm (128) | 0.50 |
| P10 | the random-start row within 0.5 pp of the fixed start at 64 on every arm's selected grid | 0.70 |

**What one seed settles:** a collapse or a monotone trend; the port's fidelity to within the floors; whether the narrow attention model is at least not worse. **What it does not:** any difference under 2.4 pp at 64; those go to the paper as "one seed, inside the seed spread".

## 5. Ops
Two spot v6e-8 pods in the ARC project (the PI: "two v6e-8 pods"), Mumbai, runtime `v2-alpha-tpuv6e`, each under its own fresh prefix (`gs://qhrrn2-arc/rescue/wladder_p0`, `_p1`), envs `tools/campaign_wladder_p{0,1}.env`, chain `tools/chain_wladder.sh` → `tools/chain_champ.sh` (arms + the fixed budget + `LADDER_BATTERY`, every default inert). The chain's own 60-step preflight per arm is the on-chip pace and memory probe before any training (rule 13a); an OOM retries once with rematerialization (labeled). Pod 0: W256 → W128 → SA128; pod 1: SA256 → SA192. Planned wall (rule 18d: wall pace, not the instantaneous one): ≈ 5–7 h per pod; deadline = launch + 14 h; cap $150. Gates before the launch: `harness_wladder.sh` 18/18; the champion / pending-runs / X5-long harnesses re-run after the shared-chain edit; `tests/test_final.py` (the attention mixer's exact S9 equivariance, the default parameter tree unchanged); the real trainer and evaluator on the Mac with SA192's launch flags (rc 0; 1,057,925 parameters); the analyzer's selftest.
