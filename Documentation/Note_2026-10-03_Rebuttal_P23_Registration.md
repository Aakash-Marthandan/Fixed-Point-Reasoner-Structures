# Note 2026-10-03 — P23: does the memoryless chaotic tail transfer to another token mixer? P22's protocol on MLP 192, three seeds (registration, before any row)

**Goal (page one).** A MEASUREMENT. No accuracy target, no cost, and nothing trained.
- Inference only, on the Mac's CPU, with the manuscript's banked MLP 192 checkpoints (seeds 0, 1, 2).
- Tool: `tools/rebuttal_p23.py`. It imports P22's statistics, letters and reader helpers and P21's MLP loader unchanged. The run loop is P22's, with the MLP pool and its own seed.
- Outputs: `runs/analysis/rebuttal_20261003f/`.
- Tests prediction 4 of plan §7 (transfer) for the law itself.

## Why

P22 confirmed on all three attention widths that stalled correction is **memoryless** and **chaotic**:
- R = 1.08, 0.91 and 1.08;
- D(1°)/D(60°) = 0.93, 0.97 and 0.82.

P21 showed, exploratorily, the same pattern on MLP 192: survivors of 32 completed at the fresh re-roll rate (0.257 vs 0.256). The question now is whether the law transfers to a different token mixer and three more independent initializations.

## Design

**Identical to P22's**, except for the following.
- **Receivers:** MLP 192 seeds 0, 1 and 2, each loaded with P21's MLP loader, whose checkpoint hash is gated against the release manifest.
- **Population:** per seed, a fresh seeded uniform draw of 160 test indices from that model's own record's unsolved-at-16 pool, excluding P21's pool for that seed (seed `[20261005, 23, seed]`).
- **Rotation seeds:** `[20261005, puzzle ID, 101 / 102 / 103, k]`.

As in P22:
- Trigger: an invalid display at iteration 16 on the CPU, asserted equivalent to CPU-unsolved.
- Arms: A0 intact (32 iterations); eight 60° re-rolls (32 iterations); four 1° and four 0.1° re-rolls (16 iterations).
- Gates: rotation score shift ≤ 1e-2; realized angle within 1 % of nominal; trigger equivalence. A gates-only smoke on the first 16 drawn puzzles of seed 0 (10:02Z, after the note and tool were committed and pushed, before launch) passed every gate: the trigger's equivalence (8 triggered), the largest rotation score shift 7.2e-7, the largest realized-angle deviation 0.001 %, and every arm with its designed length. It printed no outcome counts. Those 16 puzzles are part of the registered population.

## Rules

P22's rules, unchanged, with each seed as a receiver. Each letter is confirmed when it holds on at least two of the three seeds.

- **A — memory.** From R = Σ d_L ÷ Σ (K − d_E) d_E / (K − 1), with a puzzle bootstrap 95 % interval:
  - AGING if the interval's upper end is below 1;
  - ACCUMULATING if its lower end is above 1;
  - otherwise MEMORYLESS if the interval lies within [0.67, 1.5];
  - otherwise UNRESOLVED.
- **B — chaos.** From D(1°)/D(60°): CHAOTIC if ≥ 0.8; REGULAR if ≤ 0.5; INTERMEDIATE otherwise.
- **Exploratory:** P22's list, the same as there.

## Predictions and credences (before any row)

| rule | letter | credence |
|---|---|---|
| A | MEMORYLESS | 0.65 |
| A | AGING | 0.10 |
| A | ACCUMULATING | 0.10 |
| A | UNRESOLVED | 0.15 |
| B | CHAOTIC | 0.65 |
| B | INTERMEDIATE | 0.25 |
| B | REGULAR | 0.10 |

## Wording under each outcome

- **MEMORYLESS / CHAOTIC:** P22's sentences, with "in a recursive reasoner with a different token mixer, across three independent initializations" added.
- **Any other letter:** "The memoryless chaotic tail found on the attention models does not transfer to the MLP mixer as a [letter] pattern", together with the measured R and ratio.
- Under every outcome: P22's qualifications.

## Plan

1. Commit this note and the tool before any row; push.
2. Run the gates-only smoke.
3. Launch the three seeds in parallel at nice 5. The other session has said the Mac's CPU is free until about 15:00Z.
4. Produce the report with the tool.
5. When it is read, append the Outcome here and write a ledger line.

---

## P23 Outcome (2026-10-03 11:00Z; `runs/analysis/rebuttal_20261003f/report.{txt,json}`; every gate passed; no shared tool edited during the runs)

**Integrity.**
- Every checkpoint's hash equalled the release manifest's.
- On every seed the trigger was asserted equivalent to CPU-unsolved. Triggered: 66 / 84 / 74 of 160 (seeds 0 / 1 / 2).
- Every rotation changed no score by more than 1e-2 (realized maximum 8.5e-7), and every realized angle was within 0.002 % of nominal.
- No solved grid un-solved.
- An independent recount agrees with the report on R, both ratios, the event counts, every letter and the aggregate. A separately written bootstrap gives R intervals of [0.949, 1.391], [0.716, 1.018] and [0.761, 1.156].

**Results.**

| MLP 192 | triggered | completions early / late | R [95 %] | A | D(60°) | D(1°) | D(0.1°) | D(1°)/D(60°) [95 %] | B |
|---|---|---|---|---|---|---|---|---|---|
| seed 0 | 66 | 207 / 101 | 1.16 [0.95, 1.40] | **MEMORYLESS** | 0.343 | 0.273 | 0.295 | 0.79 [0.59, 1.06] | INTERMEDIATE |
| seed 1 | 84 | 269 / 96 | 0.86 [0.71, 1.02] | **MEMORYLESS** | 0.337 | 0.248 | 0.230 | 0.74 [0.59, 0.90] | INTERMEDIATE |
| seed 2 | 74 | 273 / 88 | 0.95 [0.76, 1.15] | **MEMORYLESS** | 0.302 | 0.227 | 0.286 | 0.75 [0.56, 0.97] | INTERMEDIATE |

**Letters.**
- A: **MEMORYLESS** on every seed, so the memory law **transfers**.
- B: **INTERMEDIATE** on every seed, so CHAOTIC is not confirmed on this mixer. A 1° rotation produces 74–79 % of the outcome disagreement that 60° does, against 82–97 % on the attention models. Every interval includes the 0.8 threshold.

**Predictions scored.** MEMORYLESS (0.65): HIT. CHAOTIC (0.65): MISS. INTERMEDIATE (0.25) is the letter that holds.

**The registered sentences that apply.**
- MEMORYLESS: P22's sentence, with "in a recursive reasoner with a different token mixer, across three independent initializations" added: "For a fixed stalled state, re-rolled trajectories complete at a rate that does not change over the following thirty-two iterations. At the level of the instance, stalled correction is memoryless: restarting cannot shorten it, and the falling completion rate across puzzles reflects which puzzles are hard."
- B, the "any other letter" sentence: "The memoryless chaotic tail found on the attention models does not transfer to the MLP mixer as an INTERMEDIATE pattern." Measured: on the MLP mixer, sixteen iterations after a 1° or 0.1° score-preserving rotation the outcome is re-rolled substantially but not fully (ratios 0.68–0.95 across the two small angles, intervals including 0.8). The stalled dynamics are strongly sensitive, somewhat less than the attention models'.

**Exploratory.**
1. Intact continuation compared with the re-rolls:
   - completed within 16 iterations: 32 / 28 / 31, against 25.9 / 33.6 / 34.1 expected;
   - within 32 iterations: 42 / 47 / 40, against a memoryless prediction of 35.4 / 45.8 / 44.3.
2. Parallel against sequential at matched compute: two 16-iteration re-rolls solved 35 / 51 / 39, against 42 / 47 / 40 for 32 iterations of intact continuation (McNemar 7/14, 11/7, 8/9; none significant).

**Reading.** The instance-level memorylessness of stalled correction holds on a second token mixer and three more independent initializations. Together with P22, that makes six receivers and two mixers. The degree of chaos at small angles is somewhat lower on the MLP mixer, so the robust statement across architectures is the memoryless law, with strong sensitivity to the readout-invisible state.
