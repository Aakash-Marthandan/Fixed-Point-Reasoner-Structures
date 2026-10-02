# Note 2026-10-01 — P12: hidden-state nudges as test-time exploration (registration, before any row)

**Goal (page one).** A MEASUREMENT. No accuracy target; $0; the Mac's CPU; inference only on the banked Attention 128 checkpoint; nothing trained.

**Provenance.**
- Started autonomously by session d378a5 under the PI's instruction to run the Mac work autonomously. It goes beyond the plan agreed at 15:20 IST, and the PI can stop it.
- Tool: `tools/rebuttal_p12.py`, which imports the other session's P3 runner (`tools/rebuttal_p3.py`, unchanged) for its one-edit loop.
- Outputs: `runs/analysis/rebuttal_20261001j/`.
- Runs at nice 10. Written before the tool is built and before any row exists.

## Why

P10 found that two to three iterations before completion, a 15° score-preserving rotation of the slow state's readout-invisible part moves the completion time in most puzzles, earlier in 23–43 % of them (SENSITIVE-TRANSIENT). P3's interim read showed that on the hardest grids a one-time random replacement sometimes lowers error one iteration later.

If completion is a sensitive transient, a small hidden-state nudge could unstick a trajectory that is still unsolved, without changing its displayed answer. That would be a cheap exploration step, distinct from a restart, which discards the trajectory. It could equally just add noise. This tests which, at a fixed extra budget.

## Design

**Population.**
- Draw 256 test puzzles uniformly, seed `[20261001, 12]`, from those Attention 128 did not solve at 16 iterations from the fixed start in the manuscript's full-test record (`paper/code/evidence/benchmark/attention_128_d16.npz`, `cold_exact == False`).
- Each is labelled by the full-test 64-iteration record (`…_d64.npz`): solved by 64 when continued intact ("transient") or not ("trapped").
- That record came from the accelerator; this run is on CPU float32. The CPU run's own iteration-16 flags are reported, and the analysis uses only puzzles unsolved at 16 on this run.

**Receiver.** Attention 128 at its benchmark weights, through the P3 / P1-P2 pipeline (release code), batch 128, fixed start.

**Arms.**

| arm | what runs | compute |
|---|---|---|
| intact | 32 iterations unedited | 32 iterations |
| sham_t17 | the edit path, nothing changed, before iteration 17 | 32 iterations |
| rot15_t17 | a 15° score- and norm-preserving rotation of the readout-orthogonal slow part before iteration 17 | 32 iterations |
| random_t17 | the readout-orthogonal part replaced by a norm-matched random vector before iteration 17 | 32 iterations |
| restart | a fresh independent Gaussian start, seed `[4242, test ID, 1, 7]` through the release's `mi_z0`, run 16 iterations | the same extra 16 iterations, without the first 16 |

The nudge seeds follow P3's convention, `[20261001, test ID, 42 or 41, t]`.

## Gates

- (1) `sham_t17` reproduces intact's exact flags at every iteration on every puzzle.
- (2) Every edit keeps every digit score: the float32 realized shift is below 1e-3.

## Rules (confirmatory)

U is the set of pool puzzles unsolved at iteration 16 on this run. For each nudge arm, Δ = (solved by 32 under the arm) − (solved by 32 intact), over U, as a share of |U|. The test is a paired exact McNemar test against intact.

- **NUDGE-HELPS** if Δ ≥ +0.05 and p < 0.01.
- **NUDGE-HURTS** if Δ ≤ −0.05 and p < 0.01.
- **NEUTRAL** otherwise.
- The primary letter is `rot15_t17`; `random_t17` gets the same rule as the secondary letter.

**Descriptive, no letter.**
- The restart's solved-at-16 share over U, against intact's extra-16 gain (solved by 32 among U).
- The union coverage of intact and restart.
- Every arm by label (transient vs trapped).
- Losses: puzzles solved by intact at 32 but not under the nudge, and the reverse.

## Predictions and credences

- `rot15_t17`: NEUTRAL 0.45, NUDGE-HURTS 0.30, NUDGE-HELPS 0.25.
- `random_t17`: NEUTRAL 0.40, NUDGE-HURTS 0.40, NUDGE-HELPS 0.20.
- On trapped puzzles, the nudge solves at least 3 more than intact (exploratory): 0.35.
- The restart solves more of U at 16 than intact gains from 16 to 32: 0.5.

## Wording under each outcome

- **NUDGE-HELPS:** "On puzzles a trajectory has not solved after 16 iterations, one small score-preserving nudge to the slow state's invisible features before continuing solves more by 32 than continuing unchanged: the sensitivity of completion can be used for exploration without discarding the trajectory."
- **NUDGE-HURTS:** "A nudge costs more completions than it creates: the sensitivity randomizes timing and does not usefully explore."
- **NEUTRAL:** "At this budget a nudge neither helps nor hurts on net; it reshuffles which puzzles complete." The gains and losses are reported.

Under every outcome:
- one checkpoint, 256 puzzles, and one nudge per puzzle;
- the nudge directions are random, not learned;
- the restart comparison is descriptive.

---

## P12 Outcome (2026-10-01 15:46Z; `runs/analysis/rebuttal_20261001j/report.{txt,json}`; tool hash unchanged from its relaunch freeze; run log exit 0; wall 68 min at nice 10)

**Disclosure.** The first launch crashed at model load, before any row: the tool imported the repository's `qhrrn2` before the P3 runner set the release import path. The tool was fixed (no repository `src` on the path; puzzles read from the data file directly), its selftest re-run, its hash re-frozen, and it was relaunched at 14:38Z.

**Gates, both passed.**
- (1) The sham reproduces intact's exact flags at every iteration on every puzzle.
- (2) The largest realized score shift is 5.0e-5, against a limit of 1e-3.

**Population.**
- 256 puzzles drawn from the full-test record's unsolved-at-16 pool. On this CPU run, 105 of them are unsolved at 16 (U), as registered; 22 of U are labelled "trapped".
- *Exploratory, post-hoc:* on the 512 unselected stratified puzzles, this pipeline and the full-test record agree on exact-at-16 for 484 of 512. The 28 disagreements split 17 one way and 11 the other. For borderline puzzles, completion by 16 flips between backends in about two-thirds of cases. That matches the 151 of 256 here and is consistent with P10's sensitivity.

**Results over U** (n = 105; solved by 32):

| arm | solved | arm-only / intact-only | Δ | McNemar p | letter |
|---|---|---|---|---|---|
| intact (continue to 32) | 42 | — | — | — | — |
| rot15_t17 (15° score-preserving rotation before 17) | 53 | 22 / 11 | +0.105 | 0.080 | NEUTRAL |
| random_t17 (norm-matched random replacement before 17) | 64 | 32 / 10 | +0.210 | 0.00094 | **NUDGE-HELPS** |
| restart (fresh Gaussian start, 16 iterations) | 43 | — | — | — | descriptive |

- **By label:** among the trapped (22), the nudges solve 4 and 5 against intact's 5. Among the transient (83), they solve 49 and 59 against intact's 37.
- **Restart:** the union of the restart and intact continuation covers 64; the restart alone adds 22.

**Predictions scored.**
- rot15 NEUTRAL (0.45): HIT.
- random NUDGE-HELPS (0.20): HIT.
- Trapped gains ≥ 3 (0.35): MISS (5 vs 5).
- The restart solves more than intact's 16→32 gain (0.5): HIT by one puzzle (43 vs 42), which is effectively a tie.

**The registered wording that applies (random_t17, the secondary letter).** "On puzzles a trajectory has not solved after 16 iterations, one small score-preserving nudge to the slow state's invisible features before continuing solves more by 32 than continuing unchanged: the sensitivity of completion can be used for exploration without discarding the trajectory." The primary letter (rot15_t17) is NEUTRAL, with the same direction: +11 net, p = 0.08.

**Reading (descriptive).** For trajectories stalled at iteration 16, one random replacement of the readout-invisible slow state — with the displayed answer and every digit score unchanged — completes 21 points more of them by 32 than continuing unchanged. It also completes more than a fresh Gaussian restart given the same extra 16 iterations (64 vs 43). The gain comes from puzzles that would complete later anyway: it is a speed-up of a sensitive transient, not new capability on trapped puzzles.

**Limits.**
- One checkpoint, 105 puzzles, one nudge per puzzle.
- Puzzles were selected by being unsolved, not by a deployable trigger. In Sudoku, a constraint-violating grid is a reference-free trigger, but it was not used here.
- One draw per arm.
- The 15° arm is not significant at the registered threshold.
- Scale, triggers and repeated nudges are for P14.
