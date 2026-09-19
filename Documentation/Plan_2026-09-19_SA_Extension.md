# The attention arms to the paper's budget: SA128 / SA192 / SA256 extended 30k → 50k, then the full champion battery — REGISTRATION (2026-09-19)

**The PI (2026-09-19, to the ops session):** "Now let's extend SA128 / SA192 / SA256 from 30k to 50k, then the full champion battery." And: "We've been auditing and finding several problems recently, I want you to be extra careful and do these runs with utmost attention to detail as we cannot afford any more errors." Nothing launches before the PI confirms page one (the goal-disclosure rule); §2–§5 lock at the registration commit, the analyzer is frozen there.

## Page one: the goal, the expected accuracy, the cost

**Goal: full rows for the three attention arms, on the paper's terms.** The width ladder trained SE-RRM's two mixers inside our block, loop and recipe (self-attention over the cells with 2D rotary positions; attention across the nine fields) to a FIXED 30,000 steps and read them on 5,000 puzzles: SA128 94.92 / 99.34, SA192 96.90 / 99.60, SA256 98.24 / 99.60 at 16 / 64 iterations (one seed). Two of the three were selected at the budget's last grid, so they are lower bounds, and none was read on the full test set. The paper's width-192 model was trained to 50,000 steps and read on all 422,786 puzzles. This run gives the attention arms the same budget and the same rows, so they can stand as full rows beside it (the PI's open decision D7), and it reads them on SE-RRM's own evaluation set.

**What is run.** Each arm's banked 30k state (its optimizer state, EMA weights and data RNG) is RESUMED to 50,000 steps. The learning rate is constant at 1e-4, so this is an exact continuation of the ladder run, not a new run; the only discontinuity is the SOT carry reset at the resume, which the trainer labels (SA192 also carries its ladder resume at 25,500). This is also the champion recipe's registered one-time extension (+20,000). The registered selection (the 512 validation puzzles every 2,000 steps, the maximum with the earliest tie, the raw monitor as the second key) runs over ALL 25 grids, 2,000 → 50,000, so a grid from the first 30,000 steps can still win. Then, on the selected grid:

| part | rows | the width-192 seeds carry it | per-arm wall (measured on the width-192 model × the attention multiplier) |
|---|---|---|---|
| **A. the champion battery** | screens (10k / 20k / the selection); D16 on all 422,786 (+ the exact bit per step + the halting logits); the final grid and the raw weights on 50k; D64 on 100k; D128 on 20k; D256 on 5k; the 5k × k32 restart scan; census; calibration | yes (C5, C7, C8) | 2.5 h / 3.5 h / 3.9 h |
| **B. the four filler rows** | D64 on ALL 422,786; D128 and D256 on the 50k; the 5k × k128 restart scan | yes (these are the paper's Table 1 cells at 64 / 128 / 256 iterations and its restart column) | 5.4 h / 7.4 h / 8.4 h |

**Expected accuracy** (on all 422,786 at 16 / 64 iterations; one seed per arm; the seed floors are 2.58 / 2.44 points):

| | expected | for comparison |
|---|---|---|
| SA128 (0.62M parameters) | 95.0–96.5 / 99.3–99.6 | our width-192 triple (0.79M, 50k): **95.41 ± 0.54 / 99.05 ± 0.18** (C5 / C7 / C8 = 95.91 / 94.82 / 95.49 and 99.16 / 98.82 / 99.18) |
| SA192 (1.06M) | 96.8–97.8 / 99.6–99.75 | SE-RRM's published (2M, one run): **93.73 / 98.22** (main text, RoPE2D); **95.4 at 16** (their appendix Table A6, 1-D RoPE) |
| SA256 (1.97M) | 98.0–98.6 / 99.6–99.8 | the ladder's 5,000-puzzle subsample reads within ±0.2 of the full set on five banked models |

What one seed can show: whether an arm is beyond the seed floor of the width-192 triple or of SE-RRM's numbers, and whether the extension moved an arm by more than a floor. It cannot rank two arms that are within a floor of each other.

**Cost and time** (one spot v6e-8 per arm, three pods in parallel, Mumbai $8/h; training at the ladder's measured WALL paces: SA128 3.25, SA192 2.32, SA256 1.96 steps/s → 1.7 / 2.4 / 2.8 h for the 20,000 steps; the battery walls are the width-192 model's measured rows × the attention arms' measured inference multiplier over it, from the ladder's k32 scans: 1.6 / 2.2 / 2.45; ±20 % until the first rows are measured on the chip, then re-projected):

| option | per arm (SA128 / SA192 / SA256) | wall | cost |
|---|---|---|---|
| **A** the champion battery | 4.4 / 6.0 / 6.9 h | ≈ 7 h | ≈ **$140** (cap $180) |
| **A + B** the rows the paper's width-192 seeds carry | 10.2 / 13.7 / 15.9 h (one chain recycle per arm at 8.5 h, two for SA256) | ≈ 16 h | ≈ **$320** (cap $380) |

**My earlier ETA for this request (≈ 7 h, ≈ $140) covered option A only.** I did not say so at the time; the four filler rows are what Table 1's width-192 cells at 64 (full set), 128 and 256 iterations and its restart column are built from, so a like-for-like attention row needs them. **Recommendation: A + B** if the attention arms are to sit in Table 1 cell for cell; A alone if they stay in a comparison table with 16 iterations on the full set and 64 on 100k.

## 2. The arms and the build (`tools/chain_saext.sh`, one arm per pod)

| arm | pod | source (read-only) | fresh prefix | banked 30k state |
|---|---|---|---|---|
| SA128 | qhrrn2-arc-pod0 | `gs://qhrrn2-arc/rescue/wladder_p0/SA128_pretrain.tgz` (crc32c rS3mvA==) | `gs://qhrrn2-arc/rescue/saext_p0` | step 30,000; 15 grids; ladder-selected 30k (EDGE) |
| SA192 | qhrrn2-arc-pod1 | `…/wladder_p1/SA192_pretrain.tgz` (C06WZg==) | `…/saext_p1` | step 30,000; 15 grids; one ladder resume at 25,500; selected 30k (EDGE) |
| SA256 | qhrrn2-arc-pod2 | `…/wladder_p1/SA256_pretrain.tgz` (XPvuvA==) | `…/saext_p2` | step 30,000; 15 grids; selected 28k |

Verified before registration, at the source: each banked `ckpt_latest.pkl` is step 30,000 with the optimizer state, the EMA weights and the RNG, and is identical to its `ckpt_030000.pkl`; the GCS objects are byte-identical (crc32c) to the copies verified at the ladder's close; no trainer, model, evaluator, census, calibration, selection, filler or live-bank code has changed since the ladder's registration (fd19514), committed or not; the arm-flag registry is identical to the ladder's; and **the chain's exact argv for the extension, parsed by the trainer's own parser, equals each ladder run's saved argv in all 91 keys except `steps` 30,000 → 50,000** (`tools/resume_flags_guard.py`).

**The resume guard (new, in `tools/chain_champ.sh` `run_pretrain`, inert unless the wrapper stages `$D/config_banked.json`):** a resumed run continues the banked state with whatever flags it is launched with, so a differing flag would change the run silently. Before the trainer starts, the exact argv is parsed with the trainer's own parser and compared key by key with the banked run's config; any difference but the budget refuses the arm (`PRETRAIN-RESUME-FLAGS-ABORT`, INCOMPLETE — the heartbeat alerts). Its comparator's selftest caught a real defect before use (Python's `True == 1`).

**Every command of the battery on an attention checkpoint.** The ladder ran only the reduced battery; the screens (k256 + the vote), the halting logits, the census, the calibration and the long-depth rows had never run on an attention arm. A CPU smoke runs every command of the chain's battery and the four filler rows with the chain's exact flags on the real SA128 and SA256 checkpoints (tiny shards) before registration (§5).

## 3. The rules (locked; `tools/analyze_saext.py`, frozen at the registration commit)

Floors = twice the largest seeded spread of the recipe: 2.58 pp at 16 iterations, 2.44 pp at 64. **INTEGRITY** gates every letter, per arm: the run's argv equals the banked (ladder) argv in every key but `steps` (30,000 → 50,000), and the banked config is the ladder's own; the `EXTENDED` record reads 30,000 → 50,000; the resume record contains 30,000; the monitor rows are the 25 grids 2,000 … 50,000 and the first 15 are the ladder's rows unchanged; the last training step is 50,000; the selection re-derived from the monitor rows with the registered rule equals `val_best.txt`; every row sits on the selected grid (the final row on the last grid, the raw row without EMA) with its registered size, depth, restarts and EMA flag; the rows on 5,000 puzzles use the ladder's 5,000 ids.
**R-SE-1 THE EXTENSION'S EFFECT** per arm, paired on the ladder's identical 5,000: the selected 50k-budget grid against the ladder's selected 30k grid at 16 (the full D16 row restricted to the 5,000) and at 64 (the full D64 row restricted to the 5,000; without part B, the 100k row's overlap, n printed): INSIDE / ABOVE-BEYOND / BELOW-BEYOND; SAME-GRID when the selection is the ladder's own grid.
**R-SE-2 WHERE THE SELECTION LANDS:** LADDER-GRID (≤ 30,000: the extension found nothing better on the monitor) / INTERIOR (32,000–48,000) / EDGE (50,000: a lower bound).
**R-SE-3 AGAINST THE PAPER'S WIDTH-192 TRIPLE** (95.406 / 99.050, all 422,786, 50k budget): ABOVE / INSIDE / BELOW by the floor, at 16 and at 64 (at 64: the full-set row; without part B, the 100k row, labelled).
**R-SE-4 AGAINST SE-RRM'S PUBLISHED NUMBERS ON THEIR SET** (all 422,786): 16 iterations against 93.73 (main text) and 95.4 (appendix Table A6); 64 against 98.22: ABOVE / WITHIN / BELOW by the floor.
**R-SE-5 SELECTOR:** the k32 scan (and the k128 with part B): CLEAN at a spurious rate ≤ 1 %, else DIRTY.
**R-SE-6 THE WIDTH ORDER AT THE PAPER'S BUDGET** (paired on all 422,786 at 16; at 64 on the full row, or on the 100k row without part B): SA192 vs SA256 and SA128 vs SA192: NARROWER-AHEAD / NARROWER-BEHIND, INSIDE / BEYOND.
**DESCRIPTIVE** (no letters): the depth rows (D128 / D256), the final and raw rows, the census, the calibration, the screens, the validation curve across the join.

## 4. Predictions (credences; written before any run)

| | prediction | credence |
|---|---|---|
| P1 | INTEGRITY PASS on all three | 0.85 |
| P2 | SA128 selected past 30,000 (INTERIOR or EDGE) | 0.75 |
| P3 | SA192 selected past 30,000 | 0.65 |
| P4 | SA256 selected past 30,000 | 0.45 |
| P5 | R-SE-1 INSIDE at both depths on SA192 and SA256 | 0.70 |
| P6 | R-SE-1 ABOVE-BEYOND at 16 on SA128 | 0.25 |
| P7 | R-SE-3: SA256 ABOVE the width-192 triple at 16 (> 97.99) | 0.60 |
| P8 | R-SE-3: SA128 INSIDE at 16 | 0.75 |
| P9 | R-SE-3: all three INSIDE at 64 | 0.90 |
| P10 | R-SE-4: SA256 and SA192 ABOVE SE-RRM's 93.73 at 16 (> 96.31) | 0.80 |
| P11 | R-SE-4: SA128 WITHIN the floor of 95.4 at 16 | 0.70 |
| P12 | R-SE-5: CLEAN on all three at k32 | 0.75 |
| P13 | no NaN, no SCAN-DEADLOCK, no resume guard refusal | 0.90 |

## 5. Ops

Three spot v6e-8 pods in the ARC project (Mumbai, `v2-alpha-tpuv6e`), one arm each under a FRESH prefix (verified empty), envs `tools/campaign_saext_p{0,1,2}.env`, chain `tools/chain_saext.sh` → `tools/chain_champ.sh` (the full battery; `LADDER_BATTERY` unset) → `tools/filler_full.sh` (part B; `SE_JOBS=""` = option A). WALL 30,600 s per chain life (a recycle = relaunch + resume from the live bank; evaluation rows resume from their 300-s partials); `runs/tpu_deadline.txt` reset BEFORE the supervisors (launch + 24 h). Only `qhrrn2-*` resources; spot only. The ops phase reads no accuracy; the tick mask is `sed -E` (tested on a VALBEST line). The first measured battery rows re-project the ETA.
**Gates before the launch (all green at registration):** `harness_saext.sh` **62/62** (the three arms fresh with the peak inside the extension, inside the ladder's 30k and at the last grid; the guard refusing an integer, a float and a boolean difference — each refusal names its key; no / short resume state; a preemption; bad arm / job; option A; a relaunch after the pretrain is banked), **four mutants caught** (the guard removed, the budget 60k, the banked config not staged, k128 dropped); every chain harness re-run after the shared-chain edit: champion 72/72, ladder 18/18, ablation 25/25, X5-long 28/28, pending runs 18/18, paper-final 20/20, C8 extension 22/22, C5 long 21/21; the resume guard 5/5 and its offline pass on the three real banked configs (91 keys identical, `steps` the only change); **the CPU smoke of every battery command and the four filler rows with the chain's exact flags on the real SA128 checkpoint (14/14) and of the four never-exercised commands on SA256 (screen, the halting logits, census, calibration: 4/4)**, outputs checked for content (per-step bits and halting logits present, vote statistics, census and calibration fields); `analyze_saext.py` selftest **30/30, five mutants killed**; the sources' crc32c against the ladder close; the fresh prefixes empty; both ladder prefixes' scan recipe `BATCHONLY` (= the pre-staged one).
**Found and fixed while building (none reached a launch):** my first smoke script handed a shell function to perl's `exec` (nothing ran; the pass rule refused it); the guard's comparator treated `True == 1` (its selftest caught it); the analyzer's rule for the final row was wrong (a selected 50,000 grid still runs the final row on `ckpt_latest` over 50k — the chain copies the row only in the no-monitor fallback) and its selftest could not see it because the fixture was built from the rule itself — the fixture is now an independent table and a mutant on that rule dies; the harness carried the same final-row assumption (fixed), and its first fixture carried the stub trainer's half-budget grid and a late stub patch (fixed: the fixture now mirrors the real banked state, 15 grids at 2,000-step spacing).
**Close:** pull with crc32c verification, structure check without reading accuracy, fleet to zero, the frozen analyzer FIRST (content hash against the registration commit), any lens after it, labelled EXPLORATORY.
