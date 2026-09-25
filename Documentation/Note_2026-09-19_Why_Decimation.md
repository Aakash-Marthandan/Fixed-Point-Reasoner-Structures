# Registration: why the loop commits (2026-09-19, before the data)

> **Adversarial audit addendum (2026-09-19):** The title mechanism is not established by dense argmax or confidence. Use the separate decimation definition and audit F02–F03; the softmax confidence columns cannot support training-normalized claims. See [the audit](Report_2026-09-19_Adversarial_Writing_Audit.md).

**Why.** The PI: "what is still unexplained is why decimation happens in the best models, it's a property of the loop but why is it?" and "No more runs for sudoku on the pod ... whatever can be run locally should be done." The account to test (stated to the PI before any of these rows): (1) every puzzle has ONE solution, so the Bayes-optimal readout is one-hot and cross-entropy never rewards expressing computational doubt; each iteration is supervised as a complete solver on 1,000 puzzles the model increasingly fits, so its confident one-hot output carries over to test puzzles right or wrong; (2) what the loop adds is COMPUTE PER SUPERVISED STEP (21 applications, near-full state replacement, a normalized state) where the control models move by a small damped step. Goal: a MEASUREMENT for paper 1; $0, the Mac's CPU, banked checkpoints, nothing trained.
**Known before this registration:** iteration-1 commitment and first-guess error on TEST puzzles for every model below; the ignition's test-side rows (commitment 36 → 88–94 % between 2k and 4k steps on one width-384 seed while 24 % of puzzles solve); the trainer's logged `train_exact` (whole-batch exactness, not per-cell, not at iteration 1).

**What.** `tools/lens_why_decimation.py` (imports the repair-radius lens's model loader and recorder; its own report files; selftest). Evaluation as the paper's (EMA, no halting, no noise, the fixed start). TRAIN = the first 512 of the 1,000 training puzzles, un-augmented; TEST = the 512 rating-stratified test puzzles of the repair-radius lens.
- **F1, train against test at iteration 1** (16 iterations run): the width-192 and width-384 symmetric models (seed 0, selected grids), TRM's network reproduced (5M), and TRM's loop WITHOUT digit augmentation (solves 21 % of the test set; `runs/pretrainsportC2_X1/ckpt_020000.pkl`). Reads: committed share, first-guess wrong share, wrong among the committed, readout entropy, exact at 16, on both sets.
- **F2, inside the first two iterations:** the state is read out after EACH outer cycle (three per iteration; z_H changes only there), on 256 test puzzles, for the width-192 model and EqR's released weights; the probe's full-iteration grid must equal the evaluator's (asserted). Reads: committed share and wrong share after cycles 1–6.
- **F3, along training:** the width-192 seed-0 run at 2k, 4k, 6k, 8k, 10k, 14k, 18k, 22k, 26k, 30k, 38k, 46k, 50k and the width-384 seed-0 run at 2k, 4k, 6k, 8k, 10k, 16k, 30k, on 128 train and 128 test puzzles: iteration-1 committed share and wrong share, exact at 16.

DESCRIPTIVE; nothing here adjudicates a registered row.

**Predictions (credences).**
- P-F1a commitment on train and test within 3 points of each other on all four models: 0.80.
- P-F1b the first guess is better on train than on test by at least 10 points of wrong share on the two symmetric models: 0.60; on the no-augmentation TRM loop the train first guess is under 10 % wrong while its test first guess is over 40 % wrong, both committed over 90 %: 0.65.
- P-F1c exact at 16 on train exceeds test on every model: 0.85.
- P-F2a on the width-192 model commitment after the FIRST outer cycle is under 60 % and rises over the three cycles of iteration 1: 0.50; the same on EqR: 0.50.
- P-F2b the wrong share falls across the cycles of iteration 1 on both models (the guess is being built, not only sharpened): 0.70.
- P-F3a at the first checkpoint where test commitment exceeds 80 %, the train first guess is at least 10 points better than the test first guess (confidence arrives with the fit of the training cells): 0.55.
- P-F3b across checkpoints, iteration-1 commitment correlates at least as strongly with train first-guess accuracy as with test exact accuracy at 16 (Spearman, the width-192 run): 0.55.

**What settles what.** F1b is the core of part (1): equal confidence on both sets with a much better guess on the training set means the confidence is inherited from the fit. If train and test first guesses are EQUALLY wrong, part (1) is wrong as stated and commitment is not a memorization effect. F2a says whether compute per supervised step builds the commitment. F3 dates the confidence against the fit.
