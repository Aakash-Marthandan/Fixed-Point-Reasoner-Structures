# The X5 long run — verdict (2026-09-17)

**Registration:** `Plan_2026-09-17_X5_Long.md` (§3 rules and §4 credences locked at 47c642a; page one confirmed by the PI). **Analyzer:** `tools/analyze_x5long.py`, frozen at 47c642a, run byte-untouched (its imports `analyze_sudokupend.py` and `analyze_paperfinal.py` likewise). **Data:** `runs/_x5long_pull/` (19 objects, crc32c 19/19). **Spend:** ≈ $27.7. Every number below is printed by a script; the outputs are in `runs/_x5long_pull/analysis/`.

## 0. The short version

| Question | Answer |
|---|---|
| Given the DEC's training compute, where does TRM's cell land at the DEC's parameter count? | **60.26 % at 16 iterations (all 422,786) and 62.40 % at 64 (100k)** — up from 45.04 / 47.47 at 50k, and 35 points below the DEC triple's 95.41 / 99.01. The letters: BELOW at both depths, GAINED (+15.22 pp on the identical 422,786), PLATEAUED. |
| Did the budget bind this time? | **No.** The selector chose grid 660,000 of 960,000, and the monitor sat within one point of its maximum from 440k to 740k. The 50k row's defect does not recur; this is a reading, not a floor. |
| What did the extra training do to the machinery? | Two things I did not predict. **The residual selector broke:** 73.9 % of wrong draws now look at least as converged as the right ones (0.01 % at 50k), so restarts no longer help — selected 60.66 vs cold 62.44 at k = 128, while a verifier would reach 76.94. **And the curve declines late:** the onset rule fired at 750k, and the final grid reads 7.55 points under the selected one on the identical 50,000 test puzzles. |
| Is the paper's attribution safe? | **Yes, with its number.** At matched parameters, recipe and training compute (×1.00 by the registered measure), TRM's cell plateaus 35 points below the symmetric DEC. The gap is far outside the ten-point band that would have owed seeds 1–2. |

## 1. Integrity and provenance

- Pull 19/19 crc32c; the manifest agrees with the per-row banks on every shared file (0 differing). All 116 grids also survive individually under the live prefix.
- The staging root is symlinks only: the long run's rows from the pull, the triple's rows on `ckpt_046000` (the paper's), EqR's k128 row from the paper-final pull; the reference root is the 50k campaign's stage.
- **INTEGRITY PASS** on every registered sub-check: the last monitor row at 960,000; the 25 rows up to 50k byte-equal to the 50k run's (it is X5's continuation); 91 cadence rows at 10k with none off cadence and none missing; the first and only resume at 50,000; the long and the 50k D16 rows on the identical 422,786 puzzles; one grid across the D16, D64, k32 and k128 rows; the registered levers in the recorded arguments; the k128 rows at 5,000 / k 128 / t 64 on EqR's puzzle set.
- The mid-run audit (HANDOFF, 20:35Z): the node's seven science-bearing tools sha256-identical to 47c642a; `pretrain.py` restores `state_ema`, so the .999 EMA that drives selection is continuous across the resume; the SOT carry reset at the resume is labeled and is what C5 (two resumes), C7 and C8 (one each) also received.
- The fixed-step screens at 10k–40k re-ran under the fresh prefix on the same grids and reproduce the 50k campaign's values **exactly** (21.09 / 17.38 / 16.80 / 41.02) — a free determinism check on the evaluator.
- **Ops disclosure:** the `VALBEST` provenance line carries the monitor value, so X5's EMA at the selected grid (0.6445 on the 512-puzzle validation monitor) was read during ops. No test-set value was read before the analyzer ran. The reading has no path into any letter.
- The `ckpt_0*` glob in two recovery paths (`amputate`, `sanitize`) was found mid-run and left unfixed until the close; it never fired (0 occurrences; the curve is untruncated) and it has no path into a result (selection uses `ckpt_[0-9]*`; the analyzers hold no checkpoint globs). Fixed after this verdict, with the harnesses re-run (§7).

## 2. The registered letters (verbatim)

```
INTEGRITY                    PASS
R-SP-1 PARAMS                MATCHED (795,941 vs 789,122, x1.009; config.json (bulk + the task table))
R-XL-1 PLATEAU               PLATEAUED (the EMA monitor's running maximum 64.45 first at 660000; 64.45 by 660000; gain +0.00 pp over the last 300,000 steps, the line 2 pp)
R-XL-2 PARITY-16             BELOW (X 60.26 vs the triple's mean 95.41, d -35.14 pp, floor 0.54 read)
R-XL-2 PARITY-64             BELOW (X 62.40 vs the triple's mean 99.01, d -36.61 pp, floor 0.18 read)
R-XL-3 GAIN-16               GAINED (d +15.22 pp on 422,786; only-long 71,356, only-50k 7,018; p 0.0e+00; the floor 1 pp)
R-XL-4 SELECTOR              DIRTY (spurious k32 73.94 %; k128 73.91 %)
R-XL-5 ONSET                 MEMORIZES-BY 750000
X k128 (descriptive)         selected 60.66, verified 76.94
R-XL-6 DEC-VS-X              C5 AHEAD (99.88 vs X 60.66; only-DEC 1961, only-X 0; p 0.0e+00) | C7 AHEAD (99.88 vs X 60.66; only-DEC 1961, only-X 0; p 0.0e+00) | C8 AHEAD (99.80 vs X 60.66; only-DEC 1959, only-X 2; p 0.0e+00) | TRIPLE-AHEAD
COMPUTE (descriptive)        median pace 273.4 steps/s past 50,000; 960,000 steps = x1.00 of one DEC seed's training compute (50,000 steps at 14.2 steps/s, the same pod type)
SELECTED GRID (descriptive)  runs/pretrainchamp_X5/ckpt_660000.pkl
```

## 3. Prediction scoreboard (credences locked before the data)

| Prediction | Credence | Outcome |
|---|---|---|
| INTEGRITY PASS | 90 % | hit |
| R-XL-1 PLATEAUED | 60 % | hit |
| R-XL-2 BELOW at 16 | 93 % | hit |
| R-XL-2 BELOW at 64 | 92 % | hit |
| R-XL-3 GAINED | 92 % | hit |
| R-XL-4 CLEAN | 75 % | **miss** — DIRTY, 73.9 % |
| R-XL-5 NOT-BY-END | 65 % | **miss** — an onset at 750k |
| R-XL-6 TRIPLE-AHEAD | 95 % | hit |
| D16 inside 52–84 | 80 % | hit (60.26; the median was 66) |
| D64 inside 54–87 | 80 % | hit (62.40; the median was 69) |

Seven letters of nine, both bands. The two misses are the informative part of the run: I expected the extra training to move the accuracy and leave the machinery alone, and it did the opposite — the accuracy plateaued five to six points under my medians while the selector and the late curve changed character. Both misses were credenced at 25 % and 35 %, so neither was ruled out; but I had no mechanism in mind for either, and §4 supplies one only for the first.

## 4. Post-results critique

**C1. The plateau letter is not an artifact of the window edge.** The running maximum sits at 660k, which is exactly where the 300k window begins, so the registered subtraction is 64.45 − 64.45. Excluding that grid, the maximum over ≤ 650k is 64.06 (at 450k) and the gain is +0.39 pp — still under the 2-point line. The monitor is within one point of its maximum at 440k, 450k, 470k, 550k, 660k, 670k and 740k: the "peak" is a broad plateau from about 190k onward, and 660k is where the selector's tie-breaking landed on it.

**C2. The selector inverted, and that is a finding about the cell, not noise.** Spurious is the fraction of wrong draws whose residual is at or under the exact draws' median. At 50k the exact draws had residual 0.002 (p50) and the wrong ones 0.027: the right answers were true fixed points and the selector was clean (0.01 %). At 660k the exact draws read 0.149 and the wrong ones 0.101 (p10 0.026): **the wrong endpoints are now more stationary than the right ones**. On the DEC (C5, k = 128) the separation is the other way round and wide — 0.112 vs 0.359 — and spurious is 0.20 %. The consequence is concrete: at k = 128 the minimum-residual draw is right on 60.66 % of puzzles, some draw is right on 76.94 %, and a verifier would rescue 814 of 5,000; a single cold start (62.44 %) now beats the restart-plus-selector machinery (60.66 %). This is Night A's R-A-1 arriving with training time — the anchor row on TRM's cell at 5 M read spurious .524 at 42k ("the anchor teaches the loop to be stationary near corrupted-solution states") — and it is the sharpest contrast in the campaign: the DEC's selector property (H-45, contractive fixed points on the answer register) is a property of the cell, not of the recipe. One seed, one cell; the effect is 0.01 % → 73.9 %, far outside anything seed noise does.

**C3. The onset letter's name is the rule's; the mechanism is not shown.** R-XL-5 fires when the EMA monitor sits ≥ 2 points under its running maximum for the rest of the run, which happens from 750k. The decline is real on test data: the final grid (960k) reads 7.55 points under the selected one on the identical 50,000 (only-final 2,044, only-selected 5,817). But the training side shows no memorization signature — training-exact rises slowly and monotonically from 3.6 % to 7.5 %, the loss from 0.769 to 0.704, and the halting statistics barely move. Whatever drives the late decline, the train-exact clock the paper uses for the DEC's memorization story is not it. The paper should call this a late decline, report the peak and the onset step as registered, and not name a mechanism.

**C4. One seed; what that costs.** The BELOW letters are safe: a 35-point gap against a 0.54-point floor. The plateau *level* is where seed uncertainty lives — 60.26 against my median of 66 — and the registration's criterion (c) owed seeds 1–2 only inside ten points of the triple, which this is not. A second seed would tighten the number, not change the letter.

**C5. The EMA row is the headline; the raw weights sit 7.5 points under it — on the DEC too.** On the identical 50,000, raw − EMA at 660k is −7.54 pp for X5, and the raw monitor swings between 24 and 60 across adjacent grids late in the run. I first read this as a TRM-cell peculiarity; the same pairing on C5 gives −6.95 pp, so the raw-versus-EMA gap is a property of the constant-lr regime shared by both cells, and it carries no contrast. The .999 EMA is the registered instrument for every row. What *does* separate the two is the tail: C5's final grid reads +0.18 pp against its selected one (p = .08, no decline), X5's −7.55.

**C6. Compute.** The median pace past 50k was 273.4 steps/s, so 960k steps is ×1.00 of one DEC seed's training compute by the registered measure. Inference compute is not matched by this run, as registered; the DEC's iteration costs about nineteen times more, and the paper says so where it states the row.

**C7. Which letters could not have failed, and the count floor.** R-SP-1 (the architecture is unchanged) and R-XL-6 (only-X 0, 0 and 2 against only-DEC ≈ 1,960). R-XL-3's discordants are 71,356 : 7,018 — far above the floor; R-XL-6's are far above it. R-XL-4's DIRTY rests on 73.9 % of ≈ 60,000 wrong draws.

**C8. Descriptive readings I did not register.** The selector anatomy (C2) and the final-versus-selected pairing (C3) were built after the letters were on record; they are labeled exploratory, each has a hand-built-records selftest (`tools/lens_sudokupend_read.py`, 16/16), and neither changes a letter.

## 5. Descriptive read (EXPLORATORY — `tools/lens_sudokupend_read.py`, 16/16)

**The curve.** 46.7 at 50k → 60 by 190k → a plateau of 59–64 to 740k (maximum 64.45 at 660k) → 59.8 at 750k, 53.1 at 960k. Mean EMA 62.2 over 400–650k, 60.2 over 700–960k.

**Depth.** 60.26 → 62.40 → 62.82 → 63.28 at 16 / 64 / 128 / 256 iterations: +3.0 points, as flat as at 50k (+3.9). The DEC's +3.6 from 95.4 comes from the other end of the scale.

**Coverage.** The per-draw solve rate at k = 128 rose from 47.4 % to 62.6 %, and the puzzles with no exact draw in 128 fell from 2,084 to 1,153 (23 %). Against EqR on the identical 5,000: only-X 3, only-EqR 1,912.

**The training side.** Loss 0.769 → 0.704 over the run, training-exact 3.6 → 7.5 %, mean halting steps 7.8 → 7.0. Slow, monotone, no break where the monitor declines.

## 6. What this means for the paper

The registration's committed wording for BELOW + PLATEAUED, with the onset clause:

> At matched parameters, recipe and training compute (960k steps; the same measured compute as one width-192 seed), TRM's cell plateaus at 60.3 % at 16 iterations and 62.4 % at 64 (one seed; selected at 660k of 960k on the validation monitor; the curve declines after 740k, the final grid reading 7.6 points under the selected one). The symmetric DEC reads 95.4 ± 0.5 and 99.0 ± 0.2 at 46k.

The attribution of the accuracy to the cell stands, with its number. What the paper may add, labeled as one seed: at k = 128 restarts the cell's residual selector no longer separates right from wrong endpoints (spurious 73.9 %; selected 60.7, verified 76.9, cold 62.4), where the DEC's does (0.2–0.4 %; selected 99.85 ± 0.04). What the paper must not say: that TRM's cell "memorizes" — the train-side clock does not show it; and nothing about inference compute, which this run does not match.

Seeds 1–2 of X5 are **not** owed under the registration's criterion; they would tighten 60.3 without moving the letter.

## 7. Owed after this verdict

- The `ckpt_0*` → `ckpt_[0-9]*` fix in `chain_champ.sh` `amputate()` and `live_bank.sh` `sanitize()`, with the champion, paper-final and x5long harnesses re-run (a widening that is identical for every arm under 100k steps).
- The paper hand-off: the sentence above; the k128 row for X5 with selected and verified side by side; the corrected compute ratio (19×) wherever the 50k row's context is stated.
