# The width-192 long run — verdict and what we learn (2026-09-15)

**Registration:** `Documentation/Plan_2026-09-14_W192_Long.md` (commit 57b5656); §8 the early look at 100k (adb1310); §9 the 100k reading and two notes fixed before any instrument row existed (3b7bcad).

**Tools:**
- **The frozen analyzer:** `tools/analyze_c5l.py` + `tools/c5l_curves.py`, 0-diff against 57b5656 (and `tools/analyze_paperfinal.py` 0-diff against 7a99d9e) → `runs/analysis/c5l_20260915/c5l_verdict.{txt,json}`.
- **The §9 Note 1 labels:** `tools/c5l_mechanism.py` (selftest 9/9), committed before the pull (5fed757) → `runs/analysis/c5l_20260915/c5l_mechanism.{txt,json}`.
- **Roots:** champion `runs/_paperfinal_pull/stage`; reused held-out rows `runs/_c8x_pull/x/runs`; this run `runs/_c5l_pull/x/runs` (extracted from the crc32c-verified tarballs into a new directory).

## 1. The run and the close

**Node and training.** One spot v6e-8 in asia-south1-c, CREATE 16:04:07Z, no preemption.
- **The resume, verified on the node:** C5's banked 50k state restored read-only from `champ/C5_pretrain.tgz`; `RESUMED from runs/pretrainchamp_C5/ckpt_latest.pkl at step 50000` (8 devices × 96 rows); training reached 150k at 22:00:40Z.
- **The early look at 100k** (§8) read NOT-YET at 19:14Z; the run went on to 150k as registered.
- **Instruments:** C5's 127 rows (53 held-out + 74 train-1k grids) 22:09–23:10Z; the two test rows on the 50k subsample 23:10–23:21Z; the wide runs' 152 rows 23:29–02:25Z.
- **Completion:** `c5l_final.tgz` by 02:25:37Z; the supervisor's teardown `Deleted tpu`, `down rc=0` at 02:30:02Z.

**Fleet zero** at the source in six zones, queued 0. **Spend:** 10.43 node-hours ≈ **$83** at about $8/h (the registration's estimate ≈ $85).

**Data integrity (the PI's directive).**
- **Pulls:** 283 objects under `c5l/` (val 129, tr1k 150, test 2, `C5_pretrain.tgz`, `c5l_final.tgz`), crc32c 283/283, into `runs/_c5l_pull/tgz/`; extracted only into `runs/_c5l_pull/x/`. The 285 files the final tarball shares with the rows and the training state are sha256-identical.
- **Earlier results untouched:** the `champ/` (2,044 objects; 2,182 listing lines), `finalA/` (1,491) and `c8x/` (875) listings are identical before launch and after the close, object by object (size and timestamp); none missing, changed or new.
- **The PI's evaluation rule:** both test rows ran with `--split test --subsample 50000` (seed 20260822); nothing touched the full test set.

## 2. Integrity

**The frozen analyzer prints INTEGRITY FAIL on one check, I1** ("argv differs from the champion C5's beyond out/steps/remat"). I2–I6 all hold: the monitor rows ≤ 50k are identical, the long run's rows are complete, `resumes.txt` holds 50000, the 46k and 50k checkpoints are sha256-identical, every row's n / split / EMA / depth / grid / run matches, and the test rows sit on one puzzle set inside the paper's 46k rows.

**What I1 caught is schema growth, not a changed run** (verified at the source; the PI's call whether it is trivial, as with the paper-final reader-format addendum):
- **The differing keys:** `decarc_heads` (absent → 4), `table_lr` and `table_wd` (absent → None), `w_void` (absent → None). All four arrived in one commit, 04f995c (2026-09-10, the DEC-ARC build), after the champion night; C7, C8 and the C8 extension ran with the same schema (their argv keys equal this run's; values differ only in out, seed, steps).
- **Each is inert here:**
  - `decarc_heads` is read only by `src/qhrrn2/decarc_cell.py`, dispatched when `cell_kind == "decarc"`; this run's cell is `dec`.
  - `table_lr = table_wd = None` takes the original single-optimizer branch (`clip_by_global_norm(1.0)` + `adamw(sched, b2, wd)`), the line 04f995c moved into that branch unchanged.
  - `w_void = None` is not passed, so the model config keeps 0.1, identical in both `config.json` files.
- **The addendum check (labeled, not a registered rule):** I1 with only those four keys dropped where the champion lacks them and the long run holds the inert value → **PASS**, no remaining difference, seed 0. The model dataclass differs only in `decarc_heads` (and `remat`, which I1 excludes by design).

**Disclosures carried beside R-L1 and R-L6** (plan §9 Note 2): C5's first 50k steps ran on 4 devices × 192 rows with rematerialization; the extension ran on 8 × 96 without. The global batch (768), recipe, schedule (flat 1e-4) and per-row computation are the same; the random streams and float reduction order are not.

## 3. The registered reading (verbatim from the frozen analyzer)

| rule | reading |
|---|---|
| **R-L1 LATER-BETTER** | **EARLIER-BEST** — smoothed late max at 54k 94.17 % vs early max at 48k 95.88 %: −1.71 pp; raw held-out peak 46k 95.98 %; 150k 81.36 % |
| **R-L2 ONSET-W192** | **EARLY** — held-out onset 58k; peak 46k; fit onset 140k |
| **R-L3 CLOCK-SCALING** | w192 vs w384: **WIDTH-LR-CLOCK** (ρ192 1.61 = 58k / 36k) · w384 vs w512: **PARAM-CLOCK** (ρ512 2.38 = 38k / 16k) · onsets C5 58k, C1 34k, A5 38k, A7 38k, A8 16k · peaks C5 46k, C1 16k, A5 20k, A7 26k, A8 12k |
| **R-L4 WINDOW** | **NOT-SUPPORTED** — onset/peak: w512 A8 1.33 · w384 median 1.90 (C1 2.12, A5 1.90, A7 1.46) · w192 C5 1.26 |
| **R-L5 GAP-ORDER** | **GAP-WIDTH-ORDERED** — train-1k minus held-out at 50k: w512 +27.12 · w384 median +17.88 · w192 +1.89 pp |
| **R-L6 TEST-50K** | g_val = 46k **NO-MOVE** · g_mon = 46k **NO-MOVE** (replay 46k) · final 150k **WORSE**: D16 −15.01 pp (584 solved only at 150k / 8,087 only at 46k, n 50,000, exact p underflows to 0) DOWN; D64 −13.48 pp (86 / 6,824) DOWN |

**Predictions:** the frozen scoring gives 4 of 8 (R-L1 SAME-PLATEAU MISS · R-L2 LATE-BAND MISS · R-L3 ρ192 PARAM MISS · ρ512 PARAM HIT · R-L4 TWO-RATES MISS · R-L5 ORDERED HIT · R-L6 g_val SAME HIT · final WORSE HIT). The scoring counts g_val NO-MOVE as a hit for SAME; the plan's credence table listed NO-MOVE as its own outcome (10 %), so strictly 3 of 8.

## 4. The §9 Note 1 labels (fixed before any instrument row existed)

At each onset grid and the next three held-out grids, dH and dT are the drops under each curve's running maximum (pp):

| run | onset | label | dH / dT at the four grids |
|---|---|---|---|
| C5 (w192) | 58k | **COLLAPSE-TYPE** | 3.25/3.70 · 3.79/4.50 · 5.88/5.70 · 3.68/4.40 |
| C1 (w384) | 34k | MEMORIZATION-TYPE | 1.61/0 · 2.87/0 · 4.29/0 · 8.45/0 |
| A5 (w384) | 38k | MEMORIZATION-TYPE | 1.93/0 · 2.71/0 · 6.61/0 · 10.33/0.10 |
| A7 (w384) | 38k | MEMORIZATION-TYPE | 2.61/0 · 5.91/0 · 10.92/0 · 14.86/0 |
| A8 (w512) | 16k | MEMORIZATION-TYPE | 1.81/0 · 4.84/0 · 10.79/0 · 14.40/0 |

**Flags:** ρ192 (C5 vs C1, A5) **mixed-mechanism** → its WIDTH-LR-CLOCK letter is not support for either clock hypothesis. ρ512 (A7 vs A8) **same-mechanism MEMORIZATION-TYPE** → PARAM-CLOCK stands as read (one seed pair, plain recipe). R-L4 (all runs) **mixed-mechanism**.

**Held-out drops ≥ 10 pp under the running maximum** (descriptive):
- **C5:** 18k–24k, deepest 22k: dH 76.63, dT 77.80, recovered · 76k–150k, deepest 94k: dH 89.05, dT 90.00, not recovered by 150k.
- **C1** 50k: 13.43 / 0 · **A5** 50k: 10.33 / 0.10 · **A7** 46k–50k: 14.86 / 0 · **A8** 20k–50k: 21.07 / 0 — none recovered by 50k.

## 5. What the reading means (the mappings fixed in plan §5 and §9)

- **The paper:** unchanged. The width-192 row and the abstract stand as they are whatever this run reads (§5); the 46k pick is confirmed by both selectors over 10k–150k.
- **The ARC DEC port, EARLIER-BEST with an EARLY onset (§5):** the 50k budget was near the optimum for width 192 on 1k puzzles, and the clock picture needs revising before it guides the port.
- **The ARC DEC port, C5's drop is COLLAPSE-TYPE (§9):** select on a held-out monitor at dense grids; keep the EMA; do not treat a fixed long budget at constant learning rate as safe.
- **Not triggered:** WIDTH-LR-CLOCK (§5's μP recommendation) is flagged mixed-mechanism; PARAM-CLOCK + TWO-RATES-SUPPORTED (§5's "try narrower cells") fails on R-L4.

## 6. What we learn (descriptive; no rules)

**Held-out / train-1k at D16 for C5 (%):** 12k 90.4/92.2 · 16k 88.6/90.2 · 20k 20.8/23.2 · 22k 15.2/15.4 · 26k 94.2/95.1 · 46k 96.0/97.1 · 50k 95.9/97.8 · 58k 92.7/94.1 · 70k 90.6/93.0 · 76k 85.8/87.4 · 84k 49.3/50.7 · 90k 12.5/12.6 · 94k 6.9/7.8 · 100k 48.4/51.5 · 120k 19.9/21.1 · 130k 54.9/62.8 · 140k 78.1/86.7 · 150k 81.4/89.8.

1. **Width 192 does not memorize within 150k at this lr; it destabilizes.** Its training puzzles fail with its held-out puzzles, point for point: the transient collapse at 18k–24k, a slide of 3–6 pp from 58k, then repeated collapses (6.9 % at 94k, 19.9 % at 120k) with recoveries (81.4 % at 150k). The training loss kept falling through all of it. It never fits its own 1,000 puzzles at D16 (best 97.8 %).
2. **The wide cells memorize, cleanly.** Each fits all 1,000 training puzzles and then loses held-out accuracy while the training puzzles stay at 100 %. Each onset falls at or before the grid where the run first fits every training puzzle (A8 16k vs 18k; A7 38k vs 38k; A5 38k vs 42k; C1 34k vs 46k). The memorization gap at 50k orders with width (27 / 18 / 1.9 pp).
3. **The "optimum parameter scale" needs a second axis.** Narrowing from 384 to 192 removed memorization within the budget but exposed a different limit: the stability of training the recursion at constant learning rate. The best point is where width 192 is both unmemorized and not yet destabilized, around 46k.
4. **A late memorization component appears after the second collapse.** The training puzzles recover faster than the held-out ones (gap 1–3 pp through 120k, 7.9–8.6 pp at 130k–150k; the fit onset at 140k).
5. **The in-training monitor saw the collapse, not the slide.** The 512-puzzle monitor read 91.8–96.5 % over 52k–74k, where the 10k held-out set already dated the onset (58k).

**Caveats:** one seed for the long run and single wide runs (the clock readings are descriptive in strength, plan §7); the onset's mechanism is labeled by a rule written after the in-training monitor showed the 76k collapse but before any instrument row existed.

## 7. Open questions (proposed, nothing launched)

- **Is the collapse a learning-rate instability?** A registered test from the banked 46k state: the same recipe with a cooldown (or lr 5e-5) to 100k, instruments as here.
- **What breaks inside the recursion?** A zero-cloud CPU lens on the 75 banked C5 checkpoints: weight and spectral norms, the relative update per 2k steps, and a fixed-point probe (start at the solution: held or lost?) across 46k–150k.
