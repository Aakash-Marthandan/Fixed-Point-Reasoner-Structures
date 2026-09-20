# The restart column as a letter — REGISTRATION (2026-09-20; the PI: "Do the registered re-read with the floor")

**What this is.** The extension's verdict reports the restart columns descriptively (§5b): on the identical 5,000 puzzles at 64 iterations, the attention arms read 99.92–99.98 with the paper's selection rule at 128 restarts against the width-192 k128 triple's 99.85 ± 0.04. There is no registered letter for that column, so the comparison cannot carry a claim. This note registers one, with its floor, and a frozen reader (`tools/analyze_restarts.py`).

**DISCLOSURE, first, because it bounds what this can be.** The arms' restart values above are ALREADY READ (verdict §5b, committed 4529c28). A registration written after the numbers are known is not a prediction test of those numbers, and nothing here should be read as one. What is genuinely unread at the time of writing, and therefore what the credences below are about: (a) the width-192 triple's per-seed k128 values and hence **the floor itself**; (b) the paired per-seed comparisons (exact McNemar on the identical puzzles); (c) the verified (any-draw) column's floor and labels; (d) where each arm's restart curve reaches its plateau. The rule and the floor definition are fixed before those are computed, in this file, at this commit.

## The rule (locked at this commit; `tools/analyze_restarts.py`, frozen here)

**The floor.** As everywhere else in this program, twice the largest seeded spread of the recipe, computed on the SAME instrument: `FLOOR_SEL = 2 × (max − min)` over the width-192 seeds' k128 **residual-selected** accuracies (C5 / C7 / C8, the paper's restart column), and `FLOOR_VER` likewise over their **verified** (any-draw-exact) accuracies. Both are printed with the per-seed values that produce them; a floor is never taken from memory.
**INTEGRITY** gates the letters: every row is a k128, 64-iteration scan with EMA weights on the identical 5,000 puzzle ids (the arms' and the three seeds' ids compared pairwise), each on its arm's registered selected grid, `n = 5000`, `k_init = 128`.
**R-RS-1 SELECTED vs the triple:** each arm's residual-selected accuracy against the triple's mean: ABOVE / INSIDE / BELOW by `FLOOR_SEL`.
**R-RS-2 SELECTED, paired per seed:** each arm against each of C5 / C7 / C8 on the identical puzzles (exact McNemar): the difference, the discordant counts and p, and whether the arm is ahead of ALL THREE seeds (the ladder's and the pending runs' convention for a one-seed row: "ahead of every seed" is reported, never "ahead of the mean" alone).
**R-RS-3 VERIFIED vs the triple:** the same two readings on the any-draw column with `FLOOR_VER`.
**R-RS-4 THE PLATEAU (descriptive):** the smallest k at which an arm's selected accuracy is within 0.05 pp of its own k128 value.
**EqR** (released weights, k128 on the same 5,000) is printed as an external reference, never as a floor.
**No letter is claimed for the paper unless it is ABOVE or BELOW its floor AND ahead of / behind all three seeds.** (Mutation-tested at registration: three of four planted defects die. The fourth — dropping the per-seed clause — SURVIVES, and that is a property of the rule, not a gap in the test: with `FLOOR = 2 × (max − min)` an arm above the mean by more than the floor is necessarily above every seed, so the clause is redundant while the rows share puzzle ids exactly. It is kept for the case where they do not, and reported for the reader.)

## Predictions (credences, on the parts not yet computed)

| | prediction | credence |
|---|---|---|
| Q1 | `FLOOR_SEL` ≤ 0.20 pp (the triple's k128 spread is small) | 0.75 |
| Q2 | R-RS-1: every arm INSIDE `FLOOR_SEL` | 0.55 |
| Q3 | R-RS-1: at least one arm ABOVE `FLOOR_SEL` | 0.45 |
| Q4 | R-RS-2: at least one arm ahead of ALL THREE seeds | 0.60 |
| Q5 | R-RS-2: the arm ahead of all three (if any) is SA192 | 0.45 |
| Q6 | R-RS-3: every arm INSIDE `FLOOR_VER` | 0.65 |
| Q7 | R-RS-4: every arm's plateau is at k ≤ 16 | 0.70 |
| Q8 | INTEGRITY PASS (ids, grids, k, n, EMA on all six rows) | 0.90 |

## What each outcome licenses

- **Every arm INSIDE the floor:** the paper's restart column gains an attention row that is *not distinguishable* from the width-192 triple at one seed. That is the honest sentence, and it is the one I expect to be able to write.
- **An arm ABOVE the floor AND ahead of all three seeds:** the paper may state that the attention arm's restart column exceeds the width-192 triple's at one seed, with the compute ratio beside it (1.6–2.45×) and the one-seed caveat.
- **Anything else** (ABOVE the floor but not ahead of every seed, or the reverse): report both readings, claim neither.

## Outcome (2026-09-20, the frozen reader run on the banked rows; `runs/_saext_pull/analysis/restart_verdict.txt`)

**INTEGRITY PASS** (six rows: k128, 64 iterations, EMA, n 5,000, identical puzzle ids, each on its registered grid).
**The floor, from the width-192 seeds on this instrument:** selected C5 99.88 · C7 99.88 · C8 99.80 → mean 99.85, **FLOOR_SEL 0.160 pp**; verified C5 99.88 · C7 99.92 · C8 99.86 → mean 99.89, **FLOOR_VER 0.120 pp**. (The paper's "99.85 ± 0.04" is this mean and half-spread.)

| arm | selected | vs the triple | R-RS-1 | verified | vs the triple | R-RS-3 | per seed (both columns) | claimable | plateau |
|---|---|---|---|---|---|---|---|---|---|
| SA128 | 99.96 | +0.107 | **INSIDE** | 99.96 | +0.073 | **INSIDE** | ahead of all three | NO | k 8 |
| SA192 | 99.98 | +0.127 | **INSIDE** | 100.00 | +0.113 | **INSIDE** | ahead of all three | NO | k 16 |
| SA256 | 99.92 | +0.067 | **INSIDE** | 99.98 | +0.093 | **INSIDE** | ahead of all three | NO | k 8 |

EqR's released weights on the same 5,000: 98.84 selected and verified (a reference, not a floor).

**The reading: no letter for the paper.** Every arm is INSIDE both floors, so the restart column carries no claim — although every arm is ahead of every width-192 seed pairwise on both columns (SA192 by +0.10 / +0.10 / +0.18 pp selected). Consistent direction, differences smaller than the seed spread: exactly the case this rule was registered to adjudicate. The paper's restart column keeps the width-192 triple; an attention row may be printed beside it as a one-seed row with its compute ratio, and with no claim of difference.
**Correction this supersedes:** the verdict's §5b called the gaps "about two to three times that triple's seed half-spread", which read as more margin than the rule allows. Against the registered floor (twice the FULL spread, 0.160 pp) they are inside it. §5b now points here.
**A property of the rule, found by mutation at registration:** the "ahead of all three seeds" clause is redundant under this floor (above the mean by more than twice the spread implies above every seed); kept for rows that might not share ids, reported for the reader, never relied on alone.

| | prediction | credence | outcome |
|---|---|---|---|
| Q1 | FLOOR_SEL ≤ 0.20 pp | 0.75 | **held** (0.160) |
| Q2 | every arm INSIDE FLOOR_SEL | 0.55 | **held** |
| Q3 | at least one arm ABOVE FLOOR_SEL | 0.45 | failed |
| Q4 | at least one arm ahead of all three seeds | 0.60 | **held** (all three are) |
| Q5 | the arm ahead of all three is SA192 | 0.45 | **failed on wording** — every arm is ahead of all three, so "the arm" had no referent; a prediction must name the letter it is scored on (the same defect as the ablation's P5) |
| Q6 | every arm INSIDE FLOOR_VER | 0.65 | **held** |
| Q7 | every plateau at k ≤ 16 | 0.70 | **held** (8 / 16 / 8) |
| Q8 | INTEGRITY PASS | 0.90 | **held** |

**Brier 0.132** over the eight.
