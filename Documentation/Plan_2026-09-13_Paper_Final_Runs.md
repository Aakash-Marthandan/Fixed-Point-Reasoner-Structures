# The paper's final Sudoku runs — registration (2026-09-13)

**The PI's decision (2026-09-13):** "Let's finish the recommended runs for the paper first. Then we'll close off Sudoku and focus on ARC." And, during the build: "Make sure to not delete anything important that we worked hard to acquire. Make sure to run the ops smooth without sacrificing the science."

**Why these runs.** The Sudoku half of the paper waits on no run except three single-point exposures:
1. The installed abstract states the width-192 cell's 99.2 % at 64 iterations, and Table 1's width-192 row carries one seed. The paper's best numbers and its whole compute story rest on that seed.
2. Table 1's restart column pairs our 5,000-puzzle scans with EqR's 20,000-puzzle scan: the comparison is on different puzzles.
3. The set-attention arm's D64 cell is the last 100,000-puzzle cell in Table 1's D64 column.

This registration is locked at its commit. The analyzer `tools/analyze_paperfinal.py` adjudicates byte-untouched from that commit.

## 1. The runs (one spot v6e-8 in asia-south1-c; `tools/chain_paperfinal.sh`)

| label | what | rows |
|---|---|---|
| **C7** | C5's registered recipe (champion registration 7ada7c7: the symmetric DEC at width 192, FPA k1 ε .2 frac .25, RI σ 1, the field regime at batch 768, 30k + the extension rule, the 512-puzzle monitor with the earliest tie) at **seed 1** | the champion battery (`chain_champ.sh` `run_arm`, unchanged) + D64 on the full 422,786 + D128 and D256 on the 50k |
| **C8** | the same at **seed 2** | the same |
| **EqR port** | EqR's released EMA weights (the frontier run's port, `gs://qhrrn2-rescue/frontier/ckpts/eqr.pkl`) | k = 128 at t 64 on the champion arms' identical 5,000-puzzle set (the 512-monitor file's test split, subsample seed 20260822; the frontier headline scan's flags otherwise) |
| **C4** | the set-attention arm's selected grid | D64 on the full 422,786 |

The flag strings of C7 and C8 were verified to equal C5's 66 tokens except `--seed`. The champion night trained C5 with data parallelism over four chips; C7 and C8 train over eight chips of one host at the same global batch of 768, so the gradient is the same batch mean up to the order of reductions (labeled in the report, not a variable).

**Order:** C7, then C8, then their full-set and 50k rows, then the EqR scan, then C4's full row. The claim-bearing pair runs first; the riders run last.

## 2. What this run does not touch (the PI's directive)

The champion night's GCS state is read-only: `champ/champ_final.tgz`, every C0–C6 marker and evaluation tarball, the filler's banked rows, the 6.3 GB `champ/live` prefix (never restored; this run banks to `champ_paper/live`), and the stale `champ/filler/d64full_C4_CLAIM_w2`, which the filler reads as its own claim by running as worker 2 rather than by deleting it. The one object rewritten is `champ/jax_cache.tgz`, a compile cache re-uploaded as a superset of what was restored from it. Everything else this run writes is new: the C7/C8 markers and tarballs, `champ/filler/{d64full,d128sub,d256sub}_{C7,C8}`, `k128port_eqr`, `d64full_C4`, `paperfinal_final.tgz`. On the Mac the pulls land in a new directory, the analyzer reads a staging root, and nothing is extracted over `runs/`. The harness asserts the GCS half (P1).

## 3. The decision rules (locked verbatim at the top of `tools/analyze_paperfinal.py`)

- **INTEGRITY.** C7/C8 argv equals C5's except `--seed` (`--out`, `--steps`, `--remat` ignored; `--remat` labeled). Every selected-grid row of an arm on one checkpoint path. n-gates: D16 full 422,786; final 50,000 or 422,786; raw 50,000; D64 100,000 in the chain and 422,786 in the filler; D128 20,000 and 50,000; D256 5,000 and 50,000; the k32 scan n 5,000 at t 64; the k128 scans n 5,000, k 128, t 64. EqR's 5k idx set equals every champion arm's k128 idx set.
- **R-PF-1 SEEDS-W192.** The spread of {C5, C7, C8} at D16 on the full set ≤ 1.5 pp → TIGHT, else WIDE. The D64 full-set spread is reported beside it.
- **R-PF-2 HEADLINE-W192.** The width-192 triple's D64 full-set mean ≥ 99.0 % → HOLDS; [98.5, 99.0) → SOFTENS; < 98.5 → FALLS. The paper's number for the 0.8M cell becomes the triple mean ± half-spread whatever the letter.
- **R-PF-3 WIDTH-PAIRED.** Per seed, Δ = width 192 − width 384 at D64 on the full set with exact pairing (C5−C0, C7−C1, C8−C2): mean Δ ≥ +0.5 pp and all three Δ > 0 → WIDTH-DOWN-PAYS; mean Δ ≤ −0.5 pp and all three < 0 → WIDTH-DOWN-COSTS; |mean Δ| < 0.5 pp → PARITY; else MIXED. Reported at D16 too, descriptively.
- **R-PF-4 SELECTOR-W192.** Per arm: spurious ≤ .005 and residual-selected@32 ≥ verified@32 − .001 → CLEAN; both → SELECTOR-CLEAN.
- **R-PF-5 DEPTH-W192.** Per arm: every depth row's regressions ≤ .0005 × n → MONOTONE; puzzles solved at D16 and unsolved at D64 on the full set ≤ 2 → +ZERO-REGRESSION.
- **R-PF-6 MEMORIZATION.** Per arm: D16 full − final on 50k > 5 pp → VSEL-FINAL-DROP; mean segment CE of the last five rows < .02 → END-CE; neither → NOT-MEMORIZED (the champion analyzer's definitions).
- **R-PF-7 EXTENSION.** The registered rule fired → EXTENDED; the selected step beyond the original budget → +PAID.
- **R-PF-8 EXCURSION.** After the first 2k monitor grid with the EMA row ≥ .90, at least two consecutive grids with the raw row < .50 → EXCURSION, else NONE. Valid only if C5 reads EXCURSION. Calibrated before registration on C0–C6: C5 fires at 16k–28k; C4's single dip at 12k does not.
- **R-PF-9 RESTART-PAIRED.** EqR on the identical 5k at k 128, residual-selected and verified accuracy recomputed from the per-draw records and cross-checked to the summary. CONSISTENT with its 20k reading of 98.85 if within three combined binomial standard errors, else SET-SHIFT. Per champion arm with a k128 row: Δ in residual-selected accuracy with an exact McNemar test on the selected draw's exact bit; p < .01 and Δ > 0 → AHEAD, p < .01 and Δ < 0 → BEHIND, else PARITY.
- **R-PF-10 C4-FULL.** C4's full-set D64 within .25 pp of its 100k row → CONSISTENT, else SUBSET-SHIFT; the paired contrast against C0 on the full set is reported.

**Validation before registration (real rows):** the analyzer's residual selection on C5's k128 records reproduces the evaluator's summary (99.88 residual-selected, 99.88 verified, spurious .20 %), and its paired contrast of C5 against C0 at D64 on the full set reproduces the champion report's addendum (+0.68 pp; 4,651 and 1,757 discordant puzzles). Selftest 25/25 exercises every letter both ways and each integrity failure.

## 4. Predictions (credences; the scoreboard is in the analyzer)

| prediction | band | credence |
|---|---|---|
| C7, C8 D16 on the full set | [94.9, 96.9] each | 70 % |
| C7, C8 D64 on the full set | [98.6, 99.5] each | 70 % |
| SEEDS-W192 | TIGHT | 65 % |
| HEADLINE-W192 | HOLDS · SOFTENS · FALLS | 55 · 35 · 10 % |
| WIDTH-PAIRED | PAYS · PARITY · MIXED · COSTS | 45 · 40 · 10 · 5 % |
| SELECTOR-W192 | SELECTOR-CLEAN | 85 % |
| DEPTH-W192 | MONOTONE + ZERO-REGRESSION on both | 85 % |
| MEMORIZATION | NOT-MEMORIZED on both | 70 % |
| EXTENSION | fires on at least one of C7, C8 | 60 % |
| EXCURSION | on at least one of C7, C8 | 35 % |
| EqR on the 5k | residual-selected in [98.3, 99.4]; CONSISTENT; spurious ≤ .5 % | 75 · 80 · 85 % |
| RESTART-PAIRED | C5 AHEAD of EqR | 60 % |
| C4 full-set D64 | in [98.85, 99.10] and CONSISTENT | 75 % |

## 5. What each outcome changes in the paper (reporting rules, fixed now)

- Table 1's width-192 row becomes three seeds: mean ± half-spread at D16 and D64 on the full set and at D128 and D256 on the 50k. The abstract's 0.8M sentence uses the triple's D64 full-set mean to one decimal whatever it reads; the letter decides the sentence's verb, never its presence.
- The restart column's EqR entry becomes its identical-5k row with the paired letters; the 20k row moves to Appendix D, labeled.
- C4's dagger goes; its full-set row replaces the 100k row.
- WIDTH-PAIRED decides §4's compute sentence: PAYS "leads its seed-matched width-384 twin on every seed", PARITY "matches", COSTS "trails".
- MEMORIZATION and EXCURSION give the memorization-clock finding and the excursion paragraph their seed counts.
- If INTEGRITY fails no letter is read, and the paper keeps the one-seed labels.

## 6. Ops

- **Pod.** One spot v6e-8 in asia-south1-c (the PI's standing choice), env `tools/campaign_paperfinal.env`, `tools/pod.sh supervise`, deadline knob launch + 36 h with the node guard and the DMS past it, the 5-minute live bank to `champ_paper/live`, the 15-minute heartbeat with ALERTs, `SELF_TEARDOWN=0` (the supervisor tears the node down on `paperfinal_final.tgz` or `CHAIN-PAPERFINAL-COMPLETE`).
- **Walls, inferred and labeled.** C7 and C8 about 7–9 h each: training 2.5–4.2 h depending on the extension (the champion night's four-chip hours, halved for eight chips), the champion battery about 2.8 h, the full-set and 50k rows about 2 h at the evaluator pace measured on 2026-09-12. The EqR scan about 4 h, scaled from C5's measured k128 wall. C4's full row about 3 h. Total about 22–25 h, **≈ $180–230** at about $8 per hour. The preflight measures the training pace in its first 20 minutes and the first partial banks measure the evaluation pace; the ETA is re-priced from those.
- **Harness.** `tools/harness_paperfinal.sh` P1–P5 (fresh completion with nothing of the champion night touched; idempotent rerun; the ported checkpoint missing; a seed-arm preflight failure; the extension on one arm) and the unchanged `harness_champ.sh` 65/65 and `harness_decarc.sh` 50/50 after the shared-script edits (`chain_champ.sh` C7/C8 and `CHAMP_EXTRA_ARMS`; `live_bank.sh` `LIVE_PREFIX`; `filler_full.sh` `k128port`), each default-inert.

## 7. What could fool us, and the guard on each

- *Eight-chip data parallelism changes the recipe.* The global batch and its mean gradient are unchanged; a seed pair below C5 on every row is a legitimate seed outcome and is reported as one, with this note beside it.
- *The extension fires on both arms.* About 1.8 h each, inside the cap.
- *The EqR scan's multi-draw path stalls.* The filler has no in-band stall watchdog; the heartbeat's 30-minute evaluation-stall ALERT catches it and the partials resume.
- *A preemption.* The supervisor re-hunts Mumbai and the chain resumes from the fresh live prefix; the evaluator's partials resume bit-identically.
- *The excursion detector misreads.* It is valid only if it fires on C5.
- *The analysis overwrites local data.* The analyzer reads a staging root; the pulls never touch `runs/`.
