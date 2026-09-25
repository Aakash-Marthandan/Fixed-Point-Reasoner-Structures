# Registration: the Night-A symmetric-cell arms re-selected on the 512-puzzle monitor (2026-09-15, before the data)

**Why.** The paper's Finding 6 (digit augmentation on the symmetric cell, A4 − A3 = +3.17 pp at D16) and its width-512 contrast (A8 − A7 = −1.45 pp) are read as flat against the seed spread of A3 / A7 (+5.08 pp), which the paper attributes to the 64-puzzle monitor those runs were selected on. The second review round (area chair; statistics reviewer) called the verdict circular: the same spread is disowned in Finding 4 and used as the floor in Finding 6. The PI approved analysis on the banked data and no new pod runs.

**What.** `tools/lens_finalA_reselect.py`: the banked checkpoints of A3, A4, A7 (2k–50k, 24–25 each, `gs://qhrrn2-rescue/finalA/`), EMA weights, cold D16 on the champion night's 512 monitor puzzles (`sudoku_extreme_seed0_mon512.npz`, outside the shared 1,000-puzzle training subsample); the maximum selects (ties to the earliest); the selected checkpoint is read cold at D16 on the first 5,000 puzzles of the evaluator's seeded 20,000-puzzle test subsample (seed 20260822; binomial SE ≈ 0.3 pp), extended to 20,000 if the CPU allows. DESCRIPTIVE: no registered rule adjudicates it; the registered rows of Table 4 stand, and this reading is reported beside them, labeled.

**Predictions (credences), written before any checkpoint was evaluated.**
1. Under the 512 monitor the A3 / A7 seed spread at D16 falls to ≤ 2.58 pp (twice the champion floor): 0.55.
2. A4 − A3 under the 512 monitor lies inside twice the re-measured spread (flat by the paper's rule): 0.55; inside 2.58 pp: 0.45.
3. The 512-monitor selection moves A3's selected step by ≥ 4,000 steps from the 64-monitor pick: 0.6.

**What settles what.** If (1) and (2) hold, Finding 6 keeps "does the digit augmentation's work" with a floor the paper stands behind. If (2) fails with the contrast beyond the floor, Finding 6 reads "digit augmentation is worth +X pp to the symmetric cell, a fraction of the 64.83 it is worth to TRM's cell", and the exactness argument stays an argument. Either way the paper reports the number.

## Outcome (2026-09-17 00:58 IST; `runs/analysis/finalA_reselect_20260915/report.txt`, `reselect.json`, `read/*.npz`)

Descriptive, as registered. A3 re-selects at 30,000 (512-set 95.9 %), A7 at 22,000 (96.5 %), A4 at 12,000 (95.7 %); read at those steps on the first 5,000 of the seeded 20,000 test subsample, cold D16: A3 95.04 %, A7 95.08 %, A4 93.14 % (binomial SE 0.31–0.36 pp).

- Prediction 1 (spread ≤ 2.58 pp at 0.55): HIT. The A3/A7 spread is 0.04 pp (paired: only-A7 136, only-A3 134) against 5.08 under the 64-puzzle set.
- Prediction 2 (A4 − A3 inside twice the re-measured spread at 0.55): MISSED; A4 − A3 = −1.90 pp (only-A4 122, only-A3 217) against a band of 0.08. Inside 2.58 pp (0.45): HIT. Under the paper's registered rule the augmentation is flat; its sign reversed from the 64-puzzle selection's +3.17.
- Prediction 3 (A3's selected step moves ≥ 4,000 at 0.6): HIT; 42,000 (the later-tie pick of a four-way tie) → 30,000.

What it settles: the 5.08-pp spread was the 64-puzzle selection; digit augmentation on the symmetric cell is flat (−1.90, inside 2.58, one seed pair); the two kinds of examples' +5.03 accuracy gain over A3 was the same artifact (A3 re-selected reads 95.04 against A5's 95.08, different sets) while their selector effect stands; the symmetric state without augmentation reads 95.04 % on 5,000 puzzles against TRM's augmented cell's 86.03 on the full set. What it does not settle: width 512 (A8 was not re-selected; −1.45 is inside the rule and one seed pair). Entered into the paper 2026-09-17 (§4.1, §4.3, Finding 6's body, Tables 3–4, Appendix D) through `tools/paper_numbers.py` keys `resel.*`.
