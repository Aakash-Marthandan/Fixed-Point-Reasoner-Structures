# Note 2026-10-03 — P22: is stalled correction memoryless? Repeated answer-preserving re-rolls of the same stalled state (registration, before any row and before P21's results)

**Goal (page one).** A MEASUREMENT, with no accuracy target and no cost. It runs on the Mac's CPU, inference only, on the manuscript's banked attention checkpoints; nothing is trained.
- Tool: `tools/rebuttal_p22.py`. It imports P17's runner and the P17 and P20 pools unchanged.
- Outputs: `runs/analysis/rebuttal_20261003e/`.
- Registered before any P22 row and before P21's results were seen.

## Why

P20 found that one score-preserving rotation of the hidden state neither helps nor hurts stalled trajectories on average. Instead it re-rolls them:
- about a third of stalled puzzles change outcome, equally often in each direction;
- each perturbed arm's 16-iteration solve rate matches intact continuation's;
- four parallel 16-iteration perturbed runs solve as many puzzles as 64 iterations of continuation.

Two further observations from P20:
- Intact continuation's per-iteration solve rate falls over time (0.42 → 0.10 per 16 iterations on Attention 256).
- On puzzles it had not solved by 32, its rate over iterations 33–48 roughly equals the re-rolls' rate (0.37 vs 0.30 pooled; 0.32 vs 0.34 in P17's data).

Those observations fit a simple law that would unify the program's null results: at the level of the instance, stalled correction is **memoryless**. Each stalled state completes at its own constant rate per iteration, whichever trajectory it follows, and the falling rate across puzzles comes from mixing easy and hard instances.

- If the law holds:
  - restarting cannot shorten the tail;
  - compute can be spent as parallel re-rolls or as sequential iterations with the same result, given a validity check that needs no answer key;
  - the tail measures difficulty, not stuck trajectories.
- If trajectories age instead (rates fall within an instance), periodic restarts would help.

Classic restart theory turns on exactly this distinction (decreasing hazards within an instance make restarts useful). It has not been measured for a learned recursive reasoner.

## Design

**Receivers.** Attention 128, 192 and 256: three independent initializations of one recipe.

**Population.** Per receiver:
- a fresh seeded uniform draw of 160 test indices from the saved full-test record's unsolved-at-16 pool, excluding P12's pool and that receiver's P17 and P20 pools (seed `[20261004, 22, width]`);
- sixteen fixed-start iterations through the release step on the CPU.

Trigger (no answer key): the displayed grid at iteration 16 is invalid. This is asserted equivalent to CPU-unsolved. In P17 and P20 about 37–42 % of drawn puzzles triggered.

**Arms.** All start from each triggered puzzle's iteration-16 state.
- **A0**: intact, 32 iterations.
- **Eight 60° re-rolls**: eight independent score- and norm-preserving rotations of the readout-invisible slow state, P20's operation, each with its own random direction (seed `[20261004, puzzle ID, 101, k]`, k = 0…7). Each runs 32 iterations.
- **Four 1° re-rolls**: the same operation at one degree (seed `[…, 102, k]`, k = 0…3). Each runs 16 iterations.
- **Four 0.1° re-rolls**: exploratory (seed `[…, 103, k]`). Each runs 16 iterations.

**Gates.**
- The trigger's equivalence with CPU-unsolved.
- Every rotation changes no digit score by more than 1e-2.
- Every rotation's realized angle, per vector, is within 1 % of its nominal angle.
- A gates-only smoke on 16 puzzles of Attention 128 is run before launch.

## Quantities

For each re-roll, the first hit is the first iteration after the kick at which the display equals the solution. No solved grid has ever un-solved in P17 or P20; this is counted again here.

**Windows.** Iterations after the kick: early E = 1–16, late L = 17–32.

**Memory ratio (rule A).** Computed over the eight 60° re-rolls of each puzzle i. Let d_E,i be the re-rolls completing in E, and d_L,i those completing in L (all of them survivors of E).

R = Σ_i d_L,i ÷ Σ_i (K − d_E,i) · d_E,i / (K − 1), with K = 8.

Under memorylessness each survivor completes in L with the same probability p_i as a fresh re-roll does in E, so E[d_L,i] = K p_i (1 − p_i). For d_E,i ~ Binomial(K, p_i), E[(K − d_E,i) d_E,i / (K − 1)] has the same value, so numerator and denominator estimate the same quantity.
- R < 1: survivors complete less often than fresh re-rolls; trajectories age, and restarts would help.
- R > 1: survivors complete more often; trajectories accumulate progress.

Pooling over puzzles removes differences in difficulty between them. The interval is a 95 % percentile bootstrap over puzzles (2,000 resamples, fixed seed).

*Design check (synthetic data only, before any row).* A Mantel–Haenszel ratio of window rates, drafted first, is biased upward when exposure in L depends on events in E, as it does here. In simulation it gave intervals centred near 1.2 under a memoryless truth and read ACCUMULATING in 3–4 of 12 runs. It was replaced by R.

Simulated with per-puzzle hazards spread like P20's, 60 puzzles × 8 re-rolls:
- under a memoryless truth, R has median 1.01 and reads MEMORYLESS in 28 of 30 runs, with a median interval of [0.83, 1.21];
- an aging truth, half the re-rolls stuck, reads AGING in 12 of 12 (median R 0.56);
- an accumulating truth, the hazard doubling after 16, reads ACCUMULATING in 12 of 12 (median R 1.45).

**Chaos ratio (rule B).** The within-puzzle pairwise disagreement of outcomes at +16, compared between angles:
- D(1°) over the four 1° re-rolls;
- D(60°) over the first four 60° re-rolls;
- the ratio is D(1°) / D(60°), with a bootstrap interval over puzzles.

If the dynamics decorrelate fully even at one degree, the ratio is near 1.

## Rules

Per receiver; a letter is confirmed when it holds on at least two of the three.

**A — memory.**

| letter | condition |
|---|---|
| AGING | R's interval has its upper end below 1 (survivors complete less often than fresh re-rolls) |
| ACCUMULATING | its lower end is above 1 (survivors complete more often) |
| MEMORYLESS | otherwise, if the interval lies within [0.67, 1.5] |
| UNRESOLVED | otherwise |

**B — chaos.**

| letter | condition |
|---|---|
| CHAOTIC | D(1°) / D(60°) ≥ 0.8 |
| REGULAR | ≤ 0.5 |
| INTERMEDIATE | otherwise |

**Exploratory, reported beside the rules.**
- D(0.1°) / D(60°).
- Exchangeability of intact continuation with the re-rolls: A0's solved count at +16 against the re-rolls' expected count, and the re-rolls' mean rate on puzzles A0 solves vs does not solve.
- Parallel against sequential at matched compute: two 60° re-rolls of 16 iterations, combined with a validity check, against A0 at 32 iterations and against one re-roll at 32 iterations (exact McNemar).
- Solved-to-unsolved transitions.

**Predictions and credences (before any row).**

| rule | letter | credence |
|---|---|---|
| A | MEMORYLESS | 0.45 |
| A | AGING | 0.25 |
| A | UNRESOLVED | 0.20 |
| A | ACCUMULATING | 0.10 |
| B | CHAOTIC | 0.55 |
| B | INTERMEDIATE | 0.30 |
| B | REGULAR | 0.15 |

**Wording under each outcome.**
- **MEMORYLESS:** "For a fixed stalled state, re-rolled trajectories complete at a rate that does not change over the following thirty-two iterations. At the level of the instance, stalled correction is memoryless: restarting cannot shorten it, and the falling completion rate across puzzles reflects which puzzles are hard."
- **AGING:** "For a fixed stalled state, re-rolled trajectories complete less often the longer they run. Trajectories get stuck, and periodic restarts would shorten the tail."
- **ACCUMULATING:** "... more often the longer they run. Trajectories accumulate progress, and restarts would lengthen the tail."
- **CHAOTIC:** "A one-degree rotation of the readout-invisible state, with every score kept, changes the outcome sixteen iterations later about as often as a sixty-degree rotation. The stalled dynamics are chaotic."
- **REGULAR:** "Small rotations leave the outcome largely unchanged while large ones re-roll it. The stalled dynamics are not chaotic at this scale."

Under every outcome:
- re-roll directions are random;
- the trigger uses no answer key;
- the result holds for the tested checkpoints and population, at the tested horizon (32 iterations after iteration 16).

## Plan

1. Commit this note and the tool before any row. The commit stays local; pushing waits for the other session's local-only commit to clear.
2. Run the gates-only smoke.
3. Queue the experiment behind P21's queue, waiting on its PID.
4. Run the three widths in parallel at nice 5.
5. Produce the report with the tool.
6. Append the Outcome here and write a ledger line when it is read.
