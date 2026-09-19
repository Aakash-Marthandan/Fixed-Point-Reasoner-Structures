# The width ladder under an adversarial pass, and decimation on the attention-mixed model (2026-09-19; EXPLORATORY)

**Status: descriptive, post-verdict, zero cloud.** Written after `Report_2026-09-19_Width_Ladder_Verdict.md`; nothing here was registered before the data, so nothing here is a letter. Numbers come from two scripts on banked records: `tools/lens_ladder_adversarial.py` (selftest 5/5; `runs/analysis/ladder_adversarial_20260919/report.txt`) and `tools/lens_ladder_decimation.py` (selftest 11/11; a driver around the paper session's `tools/lens_repair_radius.py`, imported and never edited; `runs/analysis/ladder_decimation_20260919/report.txt`).

**The PI's questions (2026-09-19):** why is SA256 (SE-RRM's mixers inside our recipe, 98.24 / 99.60) above SE-RRM's published 93.73 / 98.22; what changed in the ladder and what did not; why does our MLP-mixer model peak at 0.8M parameters while the attention arms keep rising to 2M; did an ops-phase look at the width-192 reference bias the runs; and "does decimation still happen with the attention mixer since that is our central claim".

## 1. Is the SA256 number an artifact? Four checks, all negative
| check | result |
|---|---|
| **An easy subsample?** Five banked models, full set (422,786) against the ladder's 5,000 ids, 16 iterations | +0.04 pp on average (−0.17 to +0.22). At 64 iterations only 1,201 of the ids fall inside the banked 100,000-puzzle rows: +0.19 to +0.34 pp, inside that sample's own noise. The subsample does not explain 4.5 points. |
| **Leakage through the selector?** The 512 validation puzzles and the 1,000 training puzzles against the 422,786 test puzzles | 0 exact matches; 0 up to a relabeling of the digits; validation against training 0 / 0. Position symmetries were not canonicalized (stated, not tested). |
| **A look at the reference during ops?** The ops session read one partial value of the width-192 REFERENCE row (disclosed in the ledger) | It could not move a run: the arms' code and flags were fixed and harnessed before that read, the analyzer was frozen at the registration commit, and the value belonged to the reference, not to an arm. |
| **EMA or the mixer?** | Both sides use EMA; the mixers are SE-RRM's. Neither is a difference. |

**What is NOT ruled out** (the recipe ablation, `Plan_2026-09-19_Recipe_Ablation.md`, tests the first three): EqR's damping and path noise; our randomized initialization and anchor rows (flat on the MLP-mixer model in Night A: 95.08 against 95.04–95.12); TRM's batch 768 / lr 1e-4 against SE-RRM's 272 / 5e-4; and, not ablatable with today's trainer, their dropout 0.2, their block layout (two position + two symbol layers), the 5 % random early stop in place of the Q-head halting, their selection rule, and one run on each side.

## 2. Why the MLP-mixer model peaks near 0.8M while the attention arms keep rising (a HYPOTHESIS with one supporting count)
| selected checkpoint (EMA) | across the cells | across the fields | per-cell MLP | total |
|---|---|---|---|---|
| W128, MLP mixer | 124,416 | 0 | 393,216 | 551,205 |
| W256, MLP mixer | 124,416 | 0 | 1,179,648 | 1,436,709 |
| SA128, attention | 131,072 | 65,536 | 393,216 | 623,397 |
| SA192, attention | 294,912 | 98,304 | 589,824 | 1,057,957 |
| SA256, attention | 524,288 | 131,072 | 1,179,648 | 1,967,653 |

The 81-cell SwiGLU that mixes across the cells has **124,416 parameters at every hidden size**: widening our model buys per-cell capacity and no cross-cell communication. In the attention arms the cross-cell parameters grow with the square of the hidden size. That is consistent with "wider is not better" on the MLP-mixer model (R-WL-2 read INTERIOR-192 but UNRESOLVED at one seed) and "wider is better" on the attention arms, and it is a count, not an experiment: no arm varied the cell mixer's size at a fixed hidden size. **For the paper: the narrowing statement is scoped to the MLP-mixer design.**

**A free decomposition from banked rows:** attention across the fields ALONE (C4: hidden 384, the MLP cell mixer kept) against C0, paired on the full rows: **+1.61 pp at 16 iterations (n 422,786), +0.48 pp at 64 (n 100,000)**. Both are far beyond the PUZZLE-sampling chance (exact McNemar) and both are INSIDE the seed floors (2.58 / 2.44 pp; one seed per arm), so by the measurement law this stays what the champion verdict called it: not resolved at one seed. It is a direction, not a finding: the field coupling MAY carry part of the attention arms' gain, and the ladder has no arm that separates the two mixers at hidden 256.

**Compute beside any attention row.** The attention arms cost ×1.5 / ×2.1 / ×4.9 per training step at hidden 128 / 192 / 256 against the width-192 MLP-mixer model (14.2 steps/s on the same pod type; the verdict report's C2), and their 32-restart scans ran 39 / 53 / 60 min against 17 / 22.7 / 31 min for the MLP mixer at the same hidden size (×1.9–2.3 at inference; walls from the ops logs' timestamps). Three of the five ladder arms (W128, SA128, SA192) were selected at the 30k budget's last grid and are lower bounds.

## 3. Decimation on the attention-mixed model: it holds
Panel A of the repair-radius lens (the fixed start, 64 iterations, no halting, no noise, EMA weights) on the identical 512 rating-stratified test puzzles as the paper's width-192 panel. Expectations were written in the driver before any row.

| model | solved @16 / @64 | committed after iteration 1 | committed AND wrong (solved puzzles) | first exact: median / p90 |
|---|---|---|---|---|
| SA256 (attention, 28k) | 98.2 / 99.8 | 93.7 % | 23.3 % | 2 / 6 |
| SA192 (attention, 30k) | 96.1 / 99.8 | 95.2 % | 25.6 % | 2 / 8 |
| W256 (MLP mixer, 18k) | 94.7 / 98.2 | 88.7 % | 30.9 % | 3 / 9 |
| C5 (the paper's model: MLP mixer, hidden 192, 46k) | 96.1 / 99.8 | 92.8 % | 31.9 % | 3 / 9 |
| C0 (MLP mixer, hidden 384) | 94.1 / 99.6 | 95.8 % | 26.3 % | 2 / 9 |

Wrong share of the empty cells on solved puzzles, by iteration (1, 2, 4, 8, 16, 64): SA256 27.3 → 15.8 → 7.9 → 3.5 → 0.9 → 0; SA192 28.7 → 17.4 → 10.4 → 5.4 → 2.1 → 0; C5 36.4 → 25.4 → 12.7 → 5.5 → 2.0 → 0.

**Reading.** The signature the paper names is present with the attention mixers: after ONE iteration 94–95 % of the empty cells are committed, about a quarter of the empty cells are committed and WRONG on puzzles that end up solved, and the wrong share is repaired to under 4–6 % by iteration 8. The attention model starts from a better first guess and repairs faster (E1, E2, E3 hold on both arms). E4 (failed puzzles stay flat) is **uninformative**: SA256, SA192 and C5 each fail ONE of the 512 puzzles at 64 iterations; on W256 (9 failures) the wrong share goes 56.5 → 50.3, outside the 5-point band by 1.2. W256 misses E1 narrowly (88.7 % committed against the 90 % line). **For the paper: "commit, then repair" is a property of the recursion on this task under both mixers, at one seed per attention arm; the claim does not depend on the MLP mixer.**

## 4. What goes to the paper session
1. The SE-RRM paragraph: their mixers inside our recipe reach 98.2 / 99.6 at 2.0M parameters (one seed, 5,000 puzzles); the difference from their published number is not the mixer; the recipe ablation's outcome adds one sentence.
2. The narrowing statement scoped to the MLP-mixer design, with the 124,416-parameter count as the stated reason to expect it.
3. Compute stated beside any attention row.
4. Decimation: one sentence that the signature holds under attention mixers (this note's §3), if the PI wants it in.
