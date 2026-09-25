# The width ladder and the attention-mixer arms — verdict (2026-09-19)

> **Adversarial audit addendum (2026-09-19):** The saved accuracy tables reproduce, but the asynchronous pace calculation does not. Audit F08 replaces SA256’s 4.9× claim with an approximately 2.45× logged step-time estimate; it is not inference cost. Small residuals do not establish true fixed points (F14). Frozen verdict text below is preserved. See [the audit](Report_2026-09-19_Adversarial_Writing_Audit.md).

**Registration:** `Plan_2026-09-18_Width_Ladder.md` (§3 rules and §4 credences locked at fd19514; page one confirmed by the PI). **Analyzer:** `tools/analyze_wladder.py`, frozen at fd19514 — verified by content hash, run byte-untouched. **Data:** `runs/_wladder_pull/` (27 objects, crc32c 27/27). **Spend:** $147.11 (cap $150). Every number below is printed by a script; the outputs are in `runs/_wladder_pull/analysis/`.

## 0. The short version

The ladder was registered to answer a reviewer's question: *why does our 0.8 M symbol-equivariant model beat SE-RRM's 2 M one?* The paper's working answer was "the narrowing". The measurement says something else.

| Question | Answer |
|---|---|
| Is it the mixer? | **No — the opposite.** SE-RRM's mixers (attention over the cells with 2D rotary positions, attention across the fields), reimplemented inside our loop and recipe, read **98.24 % at 16 iterations and 99.60 % at 64** at hidden 256 (one seed, the identical 5,000 puzzles, 30k steps). That is above SE-RRM's published 93.73 / 98.22 (R-WL-3: ABOVE@16) **and above our own MLP-mixer model at every matched width** (R-WL-5: +5.26 / +3.68 pp at 16 for widths 128 / 256, both BEYOND the seed floor). Whatever separates SE-RRM's published numbers from ours lives in the loop and recipe, not in their mixers. |
| Does the ladder show an interior optimum near 192? | **INTERIOR-192, but UNRESOLVED at one seed** (128: 96.84, 192: 99.16, 256: 98.66, 384: 98.75 at 64). And the narrow end is budget-limited: width 128's best grid is the budget's last. The ladder does not strengthen the narrowing claim beyond the three seed pairs the paper already has. |
| Does narrowing help the attention-mixed model? | **No.** At 64 iterations 128 / 192 / 256 read 99.34 / 99.60 / 99.60 (an exact tie, and −0.26 pp, both inside the floor). At 16 iterations wider is better: 94.92 → 96.90 → 98.24. |
| What does it cost? | The attention arms' gains come at matched *steps*, not matched *compute*: per training step they cost 1.5× (128), 2.1× (192) and 4.9× (256) of the width-192 DEC. At roughly matched compute (SA128, 0.6 M parameters) the attention model is at parity with the paper's 0.8 M model (−0.24 / +0.18 pp). |

The registration's commitment holds: no headline number changes. But the paper's answer to the SE-RRM question does.

## 1. Integrity and provenance

- The analyzer, its import `analyze_paperfinal.py` and the plan are identical by content hash to the registration commit. (A first check by `git diff` printed FROZEN vacuously — two parallel shell calls shared a working directory and the pathspec resolved against the wrong one; redone with `git show … | shasum`, which cannot pass on a missing file.)
- The pull: 27/27 crc32c; the two manifests agree with the per-row banks on every shared file (0 differing). The staging root is symlinks only.
- **INTEGRITY PASS** on all five arms: config (cell, width, mixer, coupling), seed 0, 30,000 steps, no extension marker, the last training step at the budget, n = 5,000 with `subsample 5000` and EMA weights on both rows, one grid per arm equal to the registered selection, identical puzzle ids across every row.
- Selected grids: W256 18k · W128 **30k** · SA128 **30k** · SA256 28k · SA192 **30k**. Resumes: none, except **SA192 at 25,500** (pod 1's wall ceiling; verified clean at the source: one resume, no leftover evaluators). Its selected grid was trained through that reset; the curve shows no dip after it (96.7 → 96.9 → 97.5 at 26k / 28k / 30k).
- References: W192R = seed 0's grid selected inside 30k (`ckpt_028000`), read by the paper session on the Mac with the real evaluator (f32 on CPU against bf16 on the chip; labelled in the registration). Width 384 = C0 at its selected `ckpt_016000`. **The frozen analyzer's width-384 point at 64 iterations rests on 1,201 of the 5,000 ids** (C0's banked 64-iteration row is a 100k subsample); the full-set row restricted to all 5,000 reads 98.52 against 98.75, and no letter moves under either.
- Ops disclosure: one progress line of the W192R reference's Mac evaluation (16 iterations, 0.9507 at 4,096 of 5,000) was seen while checking the gates. No ladder-arm value was read before the analyzer ran; every selection line was masked.

## 2. The registered letters (verbatim)

```
INTEGRITY                PASS
  W128   16: 89.66 (n 5000)   64: 96.84 (n 5000)
  W256   16: 94.56 (n 5000)   64: 98.66 (n 5000)
  SA128  16: 94.92 (n 5000)   64: 99.34 (n 5000)
  SA192  16: 96.90 (n 5000)   64: 99.60 (n 5000)
  SA256  16: 98.24 (n 5000)   64: 99.60 (n 5000)
  W192R  16: 95.16 (n 5000)   64: 99.16 (n 5000)
  C0     16: 94.92 (n 5000)   64: 98.75 (n 1201)
R-WL-1 W128 vs W192R @16  BEYOND (-5.50 pp, only-W128 114, only-W192R 389, n 5000, p 3.7e-36)
R-WL-1 W128 vs W192R @64  INSIDE (-2.32 pp, only-W128 18, only-W192R 134, n 5000, p 4.1e-23)
R-WL-1 W256 vs W192R @16  INSIDE (-0.60 pp, only-W256 145, only-W192R 175, n 5000, p 0.1)
R-WL-1 W256 vs W192R @64  INSIDE (-0.50 pp, only-W256 21, only-W192R 46, n 5000, p 0.0031)
R-WL-2 SHAPE@64          INTERIOR-192 / UNRESOLVED-AT-ONE-SEED (128: 96.84, 192: 99.16, 256: 98.66, 384: 98.75)
R-WL-3 FIDELITY          ABOVE@16 (SA256 98.24 / 99.60 vs SE-RRM's published 93.73 / 98.22; +4.51 / +1.38 pp; different evaluation sets)
R-WL-4 SA192 vs SA256 @64  NARROWER-BEHIND / INSIDE (+0.00 pp, p 1, n 5000)
R-WL-4 SA128 vs SA256 @64  NARROWER-BEHIND / INSIDE (-0.26 pp, p 0.041, n 5000)
R-WL-5 SA128 vs W128 @16  BEYOND (+5.26 pp, p 5.6e-29, n 5000)
R-WL-5 SA128 vs W128 @64  BEYOND (+2.50 pp, p 1.5e-23, n 5000)
R-WL-5 SA256 vs W256 @16  BEYOND (+3.68 pp, p 1.2e-31, n 5000)
R-WL-5 SA256 vs W256 @64  INSIDE (+0.94 pp, p 4e-09, n 5000)
R-WL-5 SA192 vs W192R @16  INSIDE (+1.74 pp, p 7.8e-07, n 5000)
R-WL-5 SA192 vs W192R @64  INSIDE (+0.44 pp, p 0.0021, n 5000)
STABILITY W128           selected 030000; validation max 91.80, end 91.80; scan cold 96.84, one random start 96.66 (gap -0.18 pp), verified 99.62
STABILITY W256           selected 018000; validation max 96.88, end 95.70; scan cold 98.58, one random start 98.86 (gap +0.28 pp), verified 99.76
STABILITY SA128          selected 030000; validation max 95.90, end 95.90; scan cold 99.16, one random start 99.24 (gap +0.08 pp), verified 99.94
STABILITY SA192          selected 030000; validation max 97.46, end 97.46; scan cold 99.60, one random start 99.54 (gap -0.06 pp), verified 99.96
STABILITY SA256          selected 028000; validation max 98.63, end 97.85; scan cold 99.60, one random start 99.58 (gap -0.02 pp), verified 99.94
```

## 3. Prediction scoreboard (credences written before any run)

| | Prediction | Credence | Outcome |
|---|---|---|---|
| P1 | INTEGRITY PASS on all five | 0.85 | hit |
| P2 | W256 INSIDE at both depths | 0.75 | hit |
| P3 | W128 INSIDE at 64 · W128 ahead of 192 at 64 | 0.70 · 0.35 | hit (−2.32 against a 2.44 floor: by 0.12 pp) · not ahead |
| P4 | R-WL-2 UNRESOLVED-AT-ONE-SEED | 0.70 | hit |
| P5 | the shape letter INTERIOR-192 | 0.40 (the plurality) | hit |
| P6 | R-WL-3 FAITHFUL | 0.50 (BELOW@16 0.35) | **miss — ABOVE@16**, the outcome given the least credence |
| P7 | SA192 NARROWER-AHEAD of SA256 at 64 | 0.55 | **miss — an exact tie** (4,980 of 5,000 each) |
| P8 | the attention mixer BELOW the SwiGLU at 16 on ≥ 2 of 3 sizes | 0.60 | **miss — ABOVE on all three, two BEYOND** |
| P9 | a fixed-start excursion on a narrow arm | 0.50 | **miss — none on any arm** |
| P10 | the random-start row within 0.5 pp of the fixed start on every arm | 0.70 | hit (largest gap 0.28) |

Six hits, four misses. P6 and P8 share one wrong belief: I expected the attention mixer to be a handicap at 16 iterations inside our loop (no dropout, another layer layout), and it is the largest gain in the campaign. The page-one expectations show the same error: mixer-256 "≈ 92–95 / 97.5–98.5 if the port is faithful" against 98.24 / 99.60, and width 128 "≈ 94–96 / 98.5–99.2" against 89.66 / 96.84.

## 4. Post-results critique

**C1. The fixed budget binds on three of five arms, and on the narrow end of both ladders.** W128, SA128 and SA192 select the budget's last grid with the monitor at its running maximum: SA128 and SA192 are still rising (95.1 → 95.5 → 95.9; 96.7 → 96.9 → 97.5, the maximum only at the edge), and W128 has just flattened (90.2 → 91.8 → 91.8, a tie between 28k and 30k). The wide arms peak inside it (W256 at 18k, SA256 at 28k). Narrow models learn more slowly in steps — W128 first passes 85 % at 14k, W256 at 8k, the attention arms at 6–10k — so at 30k the narrow rows are lower bounds and the wide rows are not. This is the X5 lesson again, and it points one way: **R-WL-1's W128 collapse (−5.50 at 16), R-WL-2's interior optimum and R-WL-4's "narrower is not ahead" all lean on rows that had not finished training.** None of them licenses a statement about what width 128 reaches.

**C2. Matched steps are not matched compute — and this time it flatters the winner.** The trainer's per-step cost against the width-192 DEC (14.2 steps/s on the same pod type): W128 ×0.6, W256 ×1.2, SA128 ×1.5, SA192 ×2.1, SA256 ×4.9. The attention arms' advantage is real at matched steps and comes with 1.5–5× the training compute, and attention over 81 cells per field costs correspondingly more per iteration at inference. The per-iteration table makes the trade concrete: SA256 at 16 iterations (98.24) equals W192R at 32 (98.18) — half the iterations at about five times the cost of each. At roughly matched compute and fewer parameters (SA128, 0.62 M, ×1.5) the attention model sits at parity with the paper's 0.79 M model: −0.24 pp at 16, +0.18 at 64, neither significant. The paper cannot present the attention mixer as a free improvement.

**C3. What R-WL-3 does and does not say.** ABOVE@16 compares one seed on 5,000 puzzles with SE-RRM's published full-test-set number, and the arm is our reimplementation of their two mixers inside our block, loop and recipe — not their layer layout, dropout 0.2, lr 5e-4, batch 272, random early stop or epoch count. The reading is therefore narrow and firm: **their mixers are not what holds SE-RRM's published accuracy below ours; inside our loop they do better than our own.** It does not say which of the unported differences accounts for the published gap; we changed all of them at once.

**C4. The narrowing claim is not strengthened.** On the MLP model every difference among 192, 256 and 384 is inside the one-seed floor (192 over 256 by 0.60 / 0.50; over 384 by 0.24 / 0.64 on these 5,000, consistent in sign with the paper's three seed pairs), and 128 is budget-limited (C1). On the attention model narrowing buys nothing at 64 and costs accuracy at 16 (128 → 256: +3.32 pp, beyond the 2.58 floor, with two of the three rows lower bounds). The paper's statement should stay where its evidence is — three seed pairs of 192 against 384 on the MLP-mixer model — and must not be written as a property of symmetric models in general.

**C5. A label artefact in R-WL-4.** SA192 and SA256 solve exactly 4,980 of 5,000 at 64 iterations. The rule's strict inequality prints NARROWER-BEHIND for a difference of +0.00 (p = 1); the reading is a tie. The frozen analyzer is untouched; the report states the tie.

**C6. Which letters could not have failed, and what the floors are worth here.** INTEGRITY's sub-checks were all under the chain's control. The floors (2.58 / 2.44) are twice the largest seeded spread of this recipe; on these same 5,000 the width-384 seeds read 94.92 / 94.28 / 95.56 at 16 (half-spread 0.64), so the floors are conservative. The BEYOND letters that rest on finished rows are SA256 over W256 at 16 (+3.68; selections at 28k and 18k) — the cleanest contrast in the campaign — and, by the exploratory pair, SA256 over W192R at 16 (+3.08, 202 : 48). SA128 over W128 (+5.26 / +2.50) compares two unfinished rows.

**C7. SA192's reset and W192R's numerics.** One SOT-carry reset at 25,500 on SA192 and none on the other four ladder arms; the W192R grid (28k) precedes seed 0's first resume (30,000), so the reference carries none either. SA192 is thus the only row here trained through a reset. Its curve shows no visible effect, and every SA192 letter is INSIDE, so nothing rests on it. W192R was evaluated in f32 on CPU; the numerics floor is far below every BEYOND margin.

## 5. Descriptive read (EXPLORATORY — `tools/lens_wladder_read.py`)

**By iteration, on the identical 5,000** (the packed per-step bits, unpacking verified per row against the row's own exact bit and `first_exact`):

| arm | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| W128 | 4.06 | 25.18 | 58.88 | 79.90 | 89.66 | 94.48 | 96.84 |
| W192R | 8.56 | 39.52 | 69.20 | 87.34 | 95.16 | 98.18 | 99.16 |
| W256 | 6.76 | 46.50 | 75.30 | 88.90 | 94.56 | 97.66 | 98.66 |
| SA128 | 15.84 | 41.72 | 70.14 | 88.04 | 94.92 | 97.94 | 99.34 |
| SA192 | 27.36 | 62.08 | 79.42 | 90.92 | 96.90 | 99.16 | 99.60 |
| SA256 | 25.60 | 65.18 | 83.76 | 94.00 | 98.24 | 99.16 | 99.60 |

The attention arms are ahead from the first iteration (25–27 % solved in one pass against 7–9 %) and the gap closes with depth: ≤ 0.5 pp by 64. They start from a better first guess and the loop needs fewer repairs.

**The selector stays clean under both mixers.** Spurious 0.00–0.52 % on every arm; the restart scan lifts every arm to 99.4–99.9 selected and 99.6–99.96 verified. On the attention arms the exact draws are true fixed points (residual p50 0.002–0.005; the MLP arms 0.02–0.18). Set beside the X5 long run, where TRM's cell under the same recipe lost its selector entirely (73.9 % spurious), this is two more one-seed cases of the clean selector travelling with the symmetric state rather than with a particular mixer.

**The random start changes nothing at the selected grids** (gaps −0.18 to +0.28 pp) and no arm shows an excursion of the fixed start.

## 6. What this means for the paper

1. **The SE-RRM paragraph needs a different answer.** "Why does a 0.8 M symbol-equivariant model beat SE-RRM's 2 M one?" — not because of the mixer, and the ladder does not show it is the narrowing. What the paper can say, labelled as our reimplementation, one seed, 5,000 puzzles: *SE-RRM's two mixers inside our loop and recipe read 98.2 / 99.6 at hidden 256 — above their published 93.7 / 98.2 and above our MLP-mixer model at matched width — so the published gap lies in the loop and recipe, which we did not decompose.* This supports the paper's broader reading (the symmetric state carries the accuracy; both mixers reach ≥ 99.3 at 64 inside it) better than the narrowing argument did.
2. **Keep the narrowing statement at its existing evidence** (three seed pairs, 192 over 384, MLP mixer). Add, if anything, that a one-seed ladder at a 30k budget reads an interior optimum at 192 that one seed cannot resolve and whose narrow end is budget-limited. Do not extend it to the attention-mixed model, where it does not hold.
3. **State the compute beside any attention row**: ×1.5 / ×2.1 / ×4.9 per training step; SA128 at parity with the 0.8 M model at about 1.5× its compute.
4. **No headline number changes**, as registered. SA256 is one seed on 5,000 puzzles at a 30k budget.

**For the PI — a decision this run creates but does not make.** There is now a measured configuration that reads 3 points above the paper's headline model at 16 iterations (98.24 against 95.16 on the identical 5,000, 202 : 48 discordant). Making it a claim would need its own registration — seeds, the full test set, the 50k budget — at roughly 7 h of a v6e-8 per seed for SA256 (≈ $170 for a triple before batteries), or about half that for SA192. Whether paper 1 wants that row, or notes it as the next step, is yours to decide.
