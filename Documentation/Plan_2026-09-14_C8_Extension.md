# The C8 extension — registration (2026-09-14)

**The PI's request (2026-09-14):** "I want to extend C8, the w192 seed with the lowest score, and extend it to 50k steps. I suspect that the noise played a part in the extender's precision and maybe the true best point is at around 46k. Let's run this on the pod and check if we get a better optimal point than the current step." And, during the build: "Make sure not to delete or change previous results and data. Science integrity needs to be top priority."

**Why.** The paper-final runs (registration 7a99d9e; verdict `Report_2026-09-14_Paper_Final_Verdict.md`, verified §9) trained the width-192 recipe on three seeds. The registered extension rule gave C5 and C7 50k steps and C8 30k. C8's 26k grid read 484/512 on the monitor against 485/512 at its selected 22k grid, so one puzzle decided the budget. C8 is the triple's low seed: D16 full 92.98, D64 full 97.97.

The rows that exist cannot say whether C8's deficit is its seed or its budget. Nor can they say whether the 512-puzzle monitor, at about 1 pp resolution, picked a good point. This run answers both with two instruments:
1. **The counterfactual budget:** C8 trained on to 50k, exactly as the rule would have done had it fired.
2. **A selection instrument about 20× tighter than the monitor:** 10,000 held-out puzzles drawn like the monitor, from the training file and never from the test set.

This registration is locked at its commit. `tools/analyze_c8x.py` and `tools/c8x_valcurve.py` adjudicate byte-untouched from that commit.

## 1. The run (one spot v6e-8 in asia-south1-c; `tools/chain_c8x.sh`)

| lane | what |
|---|---|
| 0 | The resume point: C8's banked 30k state (`champ/C8_pretrain.tgz`: model, EMA, optimizer and RNG state at step 30,000), read-only. The chain refuses to start below step 30,000 (`C8X-NO-RESUME-STATE`): the extension never trains from scratch. |
| 1 | `tools/chain_champ.sh` for C8 alone, through the registered extension path, the same code that extended C5 and C7. The pre-staged marker `c8x/C8_EXTENDED` sets the budget to 50k; the trainer resumes from `ckpt_latest` at 30,000 with C8's flags unchanged (seed 2, width 192, the constant field learning rate). Then the registered selection over every banked 2k grid (512-puzzle monitor, EMA key, raw second key, earliest tie) → **g_mon**, and the full champion battery on g_mon. |
| 2 | **The low-noise curves.** Every banked grid from 10k to 50k of C8, C5 and C7 (21 each; C5 and C7 read-only from their paper-final tarballs) at D16 with EMA on `sudoku_extreme_seed0_val10k.npz`: 10,000 puzzles drawn with seed 20260914 from the Sudoku-Extreme training file, outside the 1,000 training puzzles and all 512 monitor puzzles, and string-disjoint from the full test set (`tools/sx_make_valset.py`; sha256 4f5973b4…). **g_val** = the maximum exact count over C8's grids, earliest tie (`tools/c8x_valcurve.py`). |
| 3 | The filler rows for g_mon (D64 full, D128 and D256 on the 50k). Extra test rows: g_val, if it is neither g_mon nor 22k, gets D16 full and D64 full. The PI's 46k, if not among those grids, gets D16 full and D64 on the 100k. Rows for the 22k grid are never recomputed; the paper-final rows stand. |

**Walls,** measured on the paper-final pod for this cell on 8 chips:
- Training 20k steps: about 66 min.
- The battery: about 80 min (D16 full 14.5 min, D64 on the 100k 13.7, D128 5.8, D256 2.9, the k32 scan 22, plus screens, census and calibration).
- The 63 validation grids: about 35 min.
- C8's filler rows: 106 min (59.5 + 15 + 31).
- The extra rows: 0–105 min.
- Bring-up and preflight: about 25 min.

**Total about 5.5–7 h ≈ $45–56** at about $8/h (inferred). Cap launch + 13 h.

## 2. What this run does not touch (the PI's directive)

- **Read-only sources.** Nothing under `gs://qhrrn2-rescue/champ` is written or deleted. `C8_pretrain.tgz`, `C5_pretrain.tgz`, `C7_pretrain.tgz`, `jax_cache.tgz` and `sets/` are read. Every marker, tarball, live bank and manifest lives under the fresh prefix `gs://qhrrn2-rescue/c8x`.
- **The paper-final record.** Its rows, letters, verdict and report are unchanged by this run whatever it reads. This run adds a counterfactual beside them.
- **The Mac.** Pulls land in `runs/_c8x_pull/`, never over `runs/`. The analyzer reads the paper-final staging root read-only.
- **Harness proof.** `tools/harness_c8x.sh` asserts the whole `champ/` tree is byte-identical after a full run (X1) and that the 30k state is never extracted over extended grids (X5).
- **Shared-script edit.** The one edit to a shared script, `chain_champ.sh`'s `CHAMP_ALL_ARMS`, is default-inert: `harness_champ.sh` and `harness_paperfinal.sh` are re-run.

## 3. The decision rules (locked verbatim at the top of `tools/analyze_c8x.py`)

- **INTEGRITY.**
  - **I1:** the extension's argv equals the paper-final C8's except `--out`, `--steps` and `--remat` (seed 2).
  - **I2:** every monitor row at steps ≤ 30,000 equals the paper-final C8's; rows exist at every 2k from 32k to 50k; `resumes.txt` holds 30000; `EXTENDED.txt` exists.
  - **I3:** the 22k and 30k grids are byte-identical (sha256) to the paper-final's.
  - **I4:** n-gates on every row.
  - **I5:** one checkpoint path over the g_mon rows, and each validation or extra row's grid equals its name.
  - **I6:** 21 validation grids per model.
  - **I7:** the paired puzzle sets are identical.
- **R-X1 MONITOR-PICK.** g_mon = 22k → SAME, else MOVED(g_mon), cross-checked by a replay of the selector.
- **R-X2 VAL-PEAK.**
  - **Peak position:** g_val ≥ 40k → PEAK-LATE; between 30k and 40k → PEAK-MID; ≤ 30k → PEAK-EARLY.
  - **Plateau:** grids within 50 puzzles (0.5 pp) of the maximum.
  - **22k vs g_val,** paired on the 10k with exact McNemar: p < .01 and g_val ahead → 22K-BELOW, else 22K-IN-PLATEAU.
  - **The PI's 46k:** HYP-IN-PLATEAU or HYP-OUT, and against 22k HYP-ABOVE-22K or HYP-NOT-ABOVE.
- **R-X3 BETTER-POINT.** Per candidate (g_val = the letter of record; g_mon and 46k reported), on the test set paired with the paper-final 22k rows, at D16 full and at D64 (full where the candidate has it, else the 100k).
  - **Per depth:** p < .01 and Δ > 0 → UP; p < .01 and Δ < 0 → DOWN; else FLAT.
  - **Candidate letter:** both UP → BETTER; both DOWN → WORSE; both FLAT → SAME; else MIXED. A candidate equal to 22k → NO-MOVE.
- **R-X4 SEED-OR-BUDGET.** R = (D64 full at g_val − D64 full at 22k) / (C7's D64 full − C8's at 22k), where C7's gap to C8 is 0.85 pp. R ≥ .75 → BUDGET-MOSTLY; .25 < R < .75 → BOTH; R ≤ .25 → SEED-MOSTLY.
- **R-X5 TRIPLE-SENS.** The width-192 D64 full mean with C8 replaced by g_mon's row: ≥ 99.0 → HOLDS; [98.5, 99.0) → SOFTENS; < 98.5 → FALLS. Descriptive; the paper-final letters stand.
- **R-X6 W192-PLATEAU.** g_val per model (C5, C7, C8): span ≤ 8k → SHARED-PEAK, else SEED-PEAKS. MONITOR-IN-PLATEAU k/3 counts the models whose registered monitor pick lies inside their own validation plateau.

**Validation before registration:**
- `c8x_valcurve.py` selftest 4/4.
- `analyze_c8x.py` selftest 19/19: every letter both ways, the integrity failures, and the no-data reading.
- The analyzer's run on the real paper-final root with no extension data exits cleanly, all NO-DATA.
- The held-out evaluation path was exercised for real on the Mac: two shards, merge, n 96, unique idx.
- The Mac CPU re-evaluation earlier today checked the checkpoints' identity against the pods' records.

## 4. Predictions (credences)

| prediction | credence |
|---|---|
| R-X1 MOVED (the monitor picks a grid past 30k) | 45 % |
| R-X2 PEAK-LATE · 22K-BELOW · HYP-IN-PLATEAU | 60 · 70 · 55 % |
| R-X3 for g_val: BETTER · SAME · MIXED · WORSE | 55 · 35 · 5 · 5 % |
| R-X4: BUDGET-MOSTLY · BOTH · SEED-MOSTLY | 30 · 40 · 30 % |
| R-X5: HOLDS · SOFTENS · FALLS | 15 · 80 · 5 % |
| R-X6: SHARED-PEAK; MONITOR-IN-PLATEAU ≥ 2/3 | 50 %; 70 % |
| C8 at g_val: D16 full in [93.0, 95.0]; D64 full in [98.0, 98.9] | 60 %; 65 % |

**The PI's hypothesis, g_val within 44–48k:** 30 %. The curve is expected to be flat across a broad plateau, so an earliest-tie argmax could land anywhere on it.

## 5. Reporting rules (fixed now)

- **(A) The default.** The paper's width-192 row and the abstract's 98.6 ± 0.6 remain the registered protocol's (the extension rule as registered, including its noise).
  - C8 trained to 50k is reported beside the row as the counterfactual that equalizes the width-192 budgets, with its letters R-X3–R-X5: a footnote on Table 1 and an appendix paragraph with the validation curves.
  - The selection-precision finding (R-X2, R-X6) enters the checkpoint-selection discussion whatever it reads.
- **(B) Only if the PI chooses it before the analysis reads this run's rows.** The budget-matched triple (C8 := g_mon's rows) becomes the paper's width-192 row and the abstract's number, rounded to one decimal whatever it reads. The registered 30k-budget triple moves to the appendix, labeled.
- **If INTEGRITY fails,** no letter is read and the paper keeps (A) without the counterfactual.

## 6. Ops

One spot v6e-8 in asia-south1-c (the PI's standing choice).
- **Config:** env `tools/campaign_c8x.env`, `tools/pod.sh supervise`, deadline knob launch + 13 h with the node guard and the DMS past it.
- **Monitoring:** the 5-minute live bank to `c8x/live`; the 15-minute heartbeat (`HB_SENTINEL=CHAIN-C8X-COMPLETE HB_FINAL=gs://qhrrn2-rescue/c8x/c8x_final.tgz`).
- **Teardown:** `SELF_TEARDOWN=0` (the supervisor tears the node down on the final object or the sentinel).
- **Harness:** `tools/harness_c8x.sh` X1–X7 22/22, plus `harness_champ.sh` and `harness_paperfinal.sh` after the shared edit.
- **Before launch:** the validation set is banked at `c8x/sets/`.

## 7. What could fool us, and the guard on each

- **A resume that silently restarts from step 0.**
  - Lane 0 refuses to start below 30,000 (X4).
  - I2 and I3 check that the monitor rows and grids up to 30k are the paper-final's.
- **The per-row carry reset at resume** (the trainer does not checkpoint the carry): the same reset happened in C5's and C7's registered extensions, so it is part of the extension path, not a new variable.
- **Selecting on test.** g_val comes from held-out training-file puzzles disjoint from the test set; the test rows only adjudicate the pre-registered candidates.
- **A flat plateau makes the argmax arbitrary.** R-X2 reports the plateau window and the paired 22k test; R-X3 tests the candidates rather than trusting the argmax.
- **Extra budget alone might help any seed.** R-X4 compares against C7's measured gap, and R-X6 shows whether C5 and C7 also peak late on the same instrument.
- **The one-seed counterfactual is still one seed:** labeled.
- **The analysis overwrites local data.** Pulls land in a new directory; the paper-final staging root is read-only.
