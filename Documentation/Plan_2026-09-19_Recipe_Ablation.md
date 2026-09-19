# The recipe ablation: where does the advantage over SE-RRM's published number live? — REGISTRATION (2026-09-19)

**THE PI'S GO (2026-09-19, to the ops session):** "Let's do the ablation for now: Change the three things to locate the advantage we have." The three things are the three arms below, as proposed to the PI with their cost before this word. §2–§4 lock at the registration commit; the analyzer is frozen there.

## Page one: the goal, the expected accuracy, the cost

**Goal: a MEASUREMENT that locates a difference, not a better score.** The width ladder's SA256 arm is SE-RRM's two mixers (self-attention over the cells, attention across the nine fields) inside OUR block, loop and recipe at hidden 256 (1,967,621 parameters). It reads **98.24 / 99.60** at 16 / 64 iterations on 5,000 test puzzles (selected at 28k of a fixed 30k). SE-RRM's paper reports **93.73 / 98.22** for its 2M model (one run). So the mixer is not the difference; something in the recipe is, or their run is. A reviewer will ask which. This run moves ONE recipe item at a time from our side to SE-RRM's published side and reads what each costs.

**The question is posed at 16 iterations.** There the difference is 4.51 points, beyond the 2.58-point floor. At 64 iterations it is 1.38 points, INSIDE the 2.44-point floor: one seed cannot locate a difference that one seed cannot resolve, and the analyzer says so.

| arm | the one item moved | flags after SA256's | what it tests |
|---|---|---|---|
| SA256L | **the loop**: EqR's damping and path noise off | `--trm-lambda 0 --trm-beta 0` | SE-RRM has neither; both are EqR's, not ours |
| SA256S | **the start-up levers**: randomized initialization and anchor rows off | `--trm-ri-sigma 0 --fpa-k 0` | ours; flat on accuracy on the MLP-mixer model (Night A: 95.08 vs 95.04–95.12) |
| SA256O | **the optimizer**: SE-RRM's batch and learning rate | `--batch 272 --lr 5e-4 --lr-end 5e-4` | TRM's 768 / 1e-4 against SE-RRM's 272 / 5e-4 |

Everything else is SA256 verbatim: seed 0, hidden 256, both attention mixers, TRM's loop (H 3 / L 6 / 2 blocks, T 16), ACT in training, 1,000 position copies, stablemax, AdamW wd 1.0 / β2 0.95 after 2,000 warm-up steps, EMA 0.999, **fixed 30,000 steps, never extended**, selection on the 512 validation puzzles with the earliest tie, the reduced battery on the identical 5,000 test puzzles (16 and 64 iterations from the fixed start with the exact bit per step; the 32-restart scan). **The reference is the ladder's banked SA256 row; it is read, never re-run.**

**Expected accuracy (16 / 64 iterations; one seed, floors 2.58 / 2.44):**
- SA256L ≈ 96.5–98.5 / 99.0–99.6. TRM's own loop has no damping and trains; the risk is a late instability at 64 iterations.
- SA256S ≈ 97.5–98.5 / 99.3–99.6 from the fixed start (the levers were flat on the MLP model), with a worse random-start pass in the scan.
- SA256O ≈ 93–98 / 98–99.5, the widest band: a five-times-larger learning rate under wd 1.0 may destabilize our model (NaN or abort, credence 0.15), and at a fixed step budget this arm also sees 2.8× fewer training rows.
- For reference: SA256 98.24 / 99.60; SE-RRM's published 93.73 / 98.22; our width-192 MLP-mixer model 95.4 ± 0.5 / 99.0 ± 0.2.

**What each outcome would mean for the paper (one sentence in the SE-RRM paragraph either way; no claim of ours depends on it):**
- An arm reads BELOW SA256 beyond the floor at 16 → the difference is LOCATED in that item at one seed. If it is the loop, the credit is EqR's. If it is the optimizer, the credit is TRM's settings and the sentence is "SE-RRM's mixers under TRM's optimizer settings reach 98.2". If it is the start-up levers, it is ours, and new (they were flat on the MLP model).
- No arm does → NOT-LOCATED: the paper says only what is measured ("their mixers inside our recipe reach 98.2; three single-item reversions do not account for the difference") and names what was not tested.

**What this run cannot show.** One seed per arm. One item at a time, so an interaction (for example the larger learning rate WITHOUT damping) is invisible. SA256O holds the steps at 30,000, so it moves batch, learning rate AND the training rows seen (8.2M against 23.0M) together; if it is located, a rows-matched follow-up separates them. NOT ablatable with today's trainer: dropout 0.2, their block layout (two position + two symbol layers per block), the 5 % random early stop in place of the Q-head halting, their hidden-256 MLP ratios, their selection rule. SE-RRM's number is one run on their evaluation; ours is the 5,000-puzzle subsample (measured against the full set on five banked models at 16 iterations: +0.04 points on average, −0.17 to +0.22; `tools/lens_ladder_adversarial.py`).

**Cost and time (wall pace, rule 18d):** one arm per spot v6e-8 pod, three pods in parallel in the ARC project (Mumbai, $8/h). SA256 ran at 1.96 wall steps/s: L and S ≈ 4.3 h of training + ≈ 1.3 h of battery ≈ 5.5–6 h each; O is faster per step (a third of the batch) ≈ 3–4 h. **≈ $120–135, cap $170; ≈ 6 h wall.** Mac: $0.

## 2. The arms (`tools/chain_champ.sh` `arm_flags`; each = `$(arm_flags SA256)` + its trailing override, so argparse's last word is the arm's)
| arm | pod | prefix | differs from SA256 in (trainer argv) | in (model config) |
|---|---|---|---|---|
| SA256L | qhrrn2-arc-pod0 | `gs://qhrrn2-arc/rescue/sablate_p0` | `trm_lambda` 0, `trm_beta` 0 | the same two |
| SA256S | qhrrn2-arc-pod1 | `gs://qhrrn2-arc/rescue/sablate_p1` | `trm_ri_sigma` 0, `fpa_k` 0 | the same two |
| SA256O | qhrrn2-arc-pod2 | `gs://qhrrn2-arc/rescue/sablate_p2` | `batch` 272, `lr` 5e-4, `lr_end` 5e-4 | none |

The damping lives in the model config, so the evaluator inherits it: SA256L is evaluated undamped, as it was trained, which is SE-RRM's condition. The path noise is training-only. `--fpa-k 0` makes the anchor rows' ε and fraction inert. All three are in `fixed_budget`. Batch 272 divides over the 8 chips as 34 rows each; its 68 anchor rows are not a multiple of 8. The CPU smoke exercises that uneven slice on 8 forced host devices with `--dp` at a small batch with the same property (24 rows = 3 per device, 6 anchor rows: three finite steps); at the real batch 272 the Mac passes the 8-device split (`34 rows/device`) and is then killed for memory, as it is at 768, so the chain's 60-step on-chip preflight is the first full-batch run of this arm and aborts before any training if it fails.

## 3. The rules (locked; `tools/analyze_sablate.py`, selftest 25/25, three mutants killed; frozen at the registration commit)
Floors = twice the largest seeded spread of this recipe: 2.58 pp at 16 iterations, 2.44 pp at 64. **INTEGRITY** gates every letter: each arm's trainer argv AND model config differ from SA256's banked ones in EXACTLY the registered keys at the registered values (`out`, `resume` ignored; `remat` reported as math-neutral); seed 0; 30,000 steps with no extension marker; rows n 5,000 / EMA / the right depth on puzzle ids identical to SA256's; all of an arm's rows and its scan on one grid = its selection. **R-AB-1** arm vs SA256 at 16 and at 64, paired exact McNemar on the identical 5,000: INSIDE / BELOW-BEYOND / ABOVE-BEYOND; the floor is absolute and a p-value alone never moves a label. **R-AB-2** arm vs SE-RRM's 93.73 / 98.22: WITHIN the floor / ABOVE / BELOW (different evaluation sets, their one run). **R-AB-3** LOCATED at 16 / at 64 = the arms reading BELOW-BEYOND, else NOT-LOCATED; an arm's share of the SA256-to-SE-RRM difference is printed only when it is BELOW-BEYOND and the difference itself exceeds the floor. **R-AB-4** SELECTOR per arm from its k32 scan: CLEAN at a spurious rate ≤ 1 %, else DIRTY. **STABILITY** (descriptive): the selected grid, flagged EDGE when it is the budget's last grid (the arm is then a lower bound); validation maximum and end; the scan's fixed start against one random start; verified accuracy; non-finite loss rows.
A NaN abort (the chain's one-shot amputation, then `NAN-ABORT`) on SA256O is a RESULT ("unstable under SE-RRM's optimizer settings in our recipe"), reported as such with whatever grids were banked; it is not retried with different settings.

## 4. Predictions (credences; written before any run)
| | prediction | credence |
|---|---|---|
| P1 | INTEGRITY PASS on all three arms | 0.85 |
| P2 | SA256S INSIDE at both depths | 0.70 |
| P2b | SA256S: one random start ≥ 5 points below its fixed start in the scan | 0.65 |
| P2c | SA256S selector DIRTY | 0.55 |
| P3 | SA256L INSIDE at both depths | 0.60 |
| P4 | SA256O BELOW-BEYOND at 16 | 0.50 |
| P4b | SA256O aborts on a NaN or never trains | 0.15 |
| P4c | SA256O selected at the budget EDGE | 0.40 |
| P5 | at least one arm WITHIN the floor of SE-RRM's 93.73 at 16 (i.e. ≤ 96.31) | 0.35 |
| P6 | NOT-LOCATED at 16 | 0.30 |
| P7 | NOT-LOCATED at 64 (the difference there is inside the floor to begin with) | 0.80 |

## 5. Ops
Three spot v6e-8 pods in the ARC project, Mumbai, runtime `v2-alpha-tpuv6e`, one arm each under a FRESH prefix, envs `tools/campaign_sablate_p{0,1,2}.env`, chain `tools/chain_sablate.sh` → `tools/chain_champ.sh` (`LADDER_BATTERY=1`, `C1_STEPS_X=30000`, sentinel `CHAIN-SABLATE-COMPLETE`, final object `sablate_final.tgz`). The compile cache seeds from the ladder's `wladder_p1` (read-only; the harness asserts it is untouched); the monitor file from `champ/sets`. The chain's 60-step preflight is the on-chip pace and memory probe (rule 13a); an OOM retries once with rematerialization (labeled, math-neutral). WALL 30,600 s per chain life; a recycle = relaunch + resume from the live bank. `runs/tpu_deadline.txt` is reset BEFORE the supervisors start (launch + 14 h). Only `qhrrn2-*` resources; spot only. The ops phase reads no accuracy; monitor values are masked in the ticks.
**Gates before the launch:** `harness_sablate.sh` 25/25 (three mutants killed: a dropped override, an arm outside the fixed budget, a widened whitelist); every chain harness re-run after the shared-chain edit (champion, ladder, X5-long, pending runs, paper-final, C8 extension, C5 long); the CPU smokes of the three arms' real flags through the real trainer (three finite steps each at a small batch, the override read back from `config.json` in both the argv and the model config), SA256O on 8 forced host devices with `--dp`; the analyzer's selftest; the fresh prefixes verified empty.
**Close:** pull with crc32c verification, structure check without reading accuracy, fleet to zero, then the frozen analyzer FIRST (content-hash check against the registration commit), any lens after it and labeled EXPLORATORY.
