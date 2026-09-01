# The Champion Track — 3-adic Native Sudoku Architecture, d96 Pilot, and the d128 Champion

**Date:** 2026-09-01 (planned; ratified in the 2026-08-31 session) · **PI decisions ratified:** the champion track REPLACES the canvas-d128 law rung (laws freeze at the 27× d16→d96 span, stated in the paper; canvas-d128 = post-abstract option); RI returns under the H-48 dose; the architecture goes Sudoku-native. · **Purpose:** one paper section ("the laws, applied") that fields a competitive, protocol-honest number built entirely from measured program laws — the direct answer to review concern R2. · **Status:** DESIGN DOC — the launch registration (locked rules, analyzer selftest, harness) follows the build per house discipline. Nothing here is a result.

---

## §1. Why 3-adic (the design's spine)

- Dyadic (2×2) pooling fits power-of-2 canvases; Sudoku is 9×9 = 3², and its 27 constraints (9 rows, 9 cols, 9 boxes) are base-3 objects: boxes tile the grid as a 3×3 array of 3×3 blocks.
- **Box-aligned 3×3 pooling makes each level-1 block exactly one box**; the level-1 lattice is the box grid whose rows/cols are the bands/stacks. All 27 constraints become within-block (boxes) or axial (rows/cols) — none require generic long-range transport across misaligned cuts.
- Thesis form (S1): the task's dependency graph embeds in the computation tree — the min-cut regime where the code can be shortest. Measured motivation: the Sudoku flux profile is extreme-UV (~.9 at s0) on the dyadic canvas — consistent with misaligned coarse scales forcing all structure to the finest cut.
- Plain unpadded 9×9 is not a dyadic option (9 is odd → pad to 16 → 68% waste, boxes still straddle blocks). Native ⇒ choose an arity; 3 is unique and constraint-matched.
- Registered-risk-to-finding: the 3-adic/dyadic mismatch was named 2026-08-14 as the S-port's structural risk and "itself a finding about RG arity"; W7 (box4) was collapse-confounded. The pilot runs the clean test with a pre-registrable instrument signature: **flux mass migrates s0 → s1 (box scale) under alignment** (band to be locked at registration: s1 share ≥ 2× the canvas value), throat direction read alongside.
- Efficiency: pyramid tokens 81+9+1 = 91 vs 1,364 (canvas scales) ⇒ **~12–15× step-compute reduction (INFERRED until canary)**; params are width-driven and barely move — savings fund depth (T16), steps (50k+ two-phase), and width (d128-class ≈ 3.7M, under EqR's 5.03M).

## §2. Native architecture spec (build decisions, with the param trap named)

1. **Canvas 9×9, no padding, no placement offsets.** Cell states: categorical blank(=black) + digits 1–9 + VOID (query/init), exactly the current palette machinery. Exact S9 digit equivariance by weight sharing (CI-1 ported to 9×9, bit-exact test).
2. **RG levels:** s0 = 9×9 → s1 = 3×3 (box tokens) → s2 = 1×1. Kept ⊕ streamed split with VIB-priced streams at BOTH cuts (flux ledger I_s0→s1, I_s1→s2 as now; β_flux knob present, champion sets 0). Priced attention (C14) at s0 (81 tokens — cheap) and s1, d_a=6, β_nl knob.
3. **THE PARAM TRAP + FIX:** naive 3×3-window concat mixers scale ~(9d)² — ~5× the current mixer params (~12M at d96): unacceptable. **Design of record: FACTORIZED CONSTRAINT MIXING at s0 — three parallel residual operators: row-mix (per-row 9-token map), col-mix (per-column), box-mix (per-box)** — Sudoku's constraint basis as the operator set, applied every equilibrium step; ~1–2M mixer params at d96 (measured + test-pinned at build). s1 mixes factorized the same way (band-mix/stack-mix over the 3×3 box grid; tiny). Aligned/offset seam alternation is inherited by box-mix vs row/col-mix (which straddle boxes by construction). FALLBACKS in order: (a) concat 3×3 mixers at expansion 2; (b) 9×9-on-16×16 dyadic canvas (hours; 4× savings; no arity claim).
4. **Axial summaries (C7):** subsumed by row/col-mix (they ARE axial operators); keep the pooled row/col summary channels if the build finds them load-bearing in smokes — decide at build, note in the registration.
5. **Kept machinery, unchanged:** equilibrium core (learned η, η_z, deep supervision, T knob, remat), rule codebook + e_t single-task row (S-port convention; ~4–8% param overhead accepted — no surgery on the champion), FPA anchor rows, RI rows (yprev = random canvas at ri_p), NI knob, aug group acting natively (transpose/band/row-in-band/stack/col-in-stack; digit perm FREE via S9; placement-offset machinery deleted).
6. **Evaluator/probes:** native paths for eval_sudoku_extreme (no placement), retention/ladder probes on 81 cells; **NEW STANDING STATS (the review's riders, implemented once): vote@1 (B=1), unverified-majority@k, Top-1-by-residual@k (EqR's L=3 average-residual selector, verbatim), verified coverage@k** — every future scan emits all four.
7. **CI gates before silicon (house law):** S9 bit-exact (9×9); box-pool alignment named test (constructed grid → box summaries exact); factorized-mixer equivariance + inertness knobs; ledger present per cut + S9-invariant; corpus/aug validity (perms preserve solutions + uniqueness); evaluator givens-kept/exactness fixture; CPU smoke pretrain→eval→probe; offline chain harness (fresh/idempotent/NaN-amputation/both shapes) green; canary on-node with projection gate (B1).

## §3. The d96-native pilot (tag `sportC0`; one pod, one night, ≈$90–130 INFERRED)

All arms d96-class native, β_nl 1e-6, β_flux 0, lr 1e-3, B64, warmup 500, seed 0 unless noted, 50k steps (single-phase cosine unless noted), one-shot amputation, monitors 2k, ckpts 5k, screens ≥3 ckpts, full eval battery + new stats:

| Arm | Delta from P1 | Purpose |
|---|---|---|
| P1 | T12 + FPA k4 ε.2, NO RI (the D4-strategy verbatim, native) | arity control vs D4-canvas (cold 33.53 / vote@1 36.0 / verified@128 81.3) + profile-shift signature |
| P2, P2s1 | + RI .5 | THE RI LEVER at width under the dose; ×2 seeds because seed-split is the fragility signature (A2 17.73 vs A2s1 2.40) |
| P3, P3s1 | THE CHAMPION RECIPE (§4) @d96 | dress rehearsal; sets d128 bands |
| P5 | P2 + NI .01 | convicts/exonerates NI (A2s1/A9/B1 history); EqR-faithful row |
| P6 | P3 + aug 1000 | the untested series-standard lever (TRM convention; we run 100) — tested at pilot so it never enters the champion blind |

**Decision rules (to be LOCKED verbatim in the analyzer pre-data; drafted):**
- **R-C0-1 (arity):** P1 within FN2b (3.71pp) of D4-canvas on cold AND screen-vb → NATIVE-CARRIES; below both → ARITY-HURTS → champion falls back per §2 fallbacks (either way + the flux-profile read = the registered arity finding).
- **R-C0-2 (RI):** vote@1(P2) ≥ vote@1(P1) + 10pp with retfm ≥ .9 on BOTH seeds → RI-PAYS (champion keeps RI); gain < 10pp → RI-WEAK (champion drops RI; the H-37 per-draw story is scoped); retfm < .9 or NaN on either seed → RI-FRAGILE-AT-WIDTH (dose insufficient for the settle mode; champion = P1-class scaled; the fragility datum is a finding vs EqR's recipe).
- **R-C0-3 (NI):** P5 unstable while P2 pair clean → NI CONVICTED (stays out); P5 clean AND ≥ P2 + FN2b on vote@1 → NI earns the champion slot.
- **R-C0-4 (aug):** P6 ≥ P3 + FN2b on the composite → champion goes aug-1000; else aug 100.
- **R-C0-5 (champion GO):** P3 pair clean AND P3 ≥ max(P1, P2) − FN2b on the composite (cold, vote@1, verified@128) → d128 GO with P3's config verbatim (+R-C0-4's aug); else the best surviving arm's config carries; NO arm clean → PI consult (no d128 launch).
- STABILITY named per arm; FN2b/CNOISE re-measured natively from the P3/P3s1 pair.

**Draft prediction bands (finalized at registration):** P1 cold [29, 37], vote@1 [30, 40] (native ≈ canvas at matched recipe); P2 vote@1 [42, 65] (the pilot's biggest unknown; EqR's RI = +24pp on Maze at their scale), P2 cold ≥ P1 − 5pp (VOID-dilution cost); P3 ≥ P2 on vote@1 (T16 + two-phase add); best-arm verified@128 ≥ 90; s1 flux share ≥ 2× canvas; wall ≤ 2.5h/arm on v6e-8 (else the efficiency estimate was wrong — projection gate).

**Riders on idle chips (eval-only, informational):** the review's EqR-statistic + majority evals on the banked CANVAS C3X/D4 ckpts (~$10–15; the paper's §4 table rows); freethink orbit-vote@8 and ckpt-split evals on the best pilot arm; optional k=1024 coverage on canvas-C3X (PI menu, ~$15–40).

## §4. The champion design (P3 exact; scales to d128 as `sportC1`)

Every component cites its law:
- **Geometry:** 3-adic native factorized-constraint core (§2) — [S1 embed + the arity finding].
- **Width/depth:** pilot d96 → champion d128-class (~3.7M, measured+pinned) · **T16** (T12 proven under the dose; native compute affords +33% depth; λ/retfm watch; NOT T24 this round).
- **Rows:** VOID rows + **RI .5** [H-37: train the deployment init-distribution; VOID kept for the cold statistic] + **FPA k4 ε.2** [H-45's fix; retfm 1.00 ×3 scales] · **NI off** pending R-C0-3.
- **Toll:** β_nl 1e-6, β_flux 0 [H-48 closure at zero capability cost; stream toll stays off CSP basins/funnels].
- **Optimizer/schedule:** lr 1e-3 registered [D4-vindicated], B64, **two-phase: cosine 35k → floor 3e-5 ×15k at pilot; 50k → +30k at d128** [C3X continuation law: the floor phase grows ρ], EMA not used (TRM's lever — noted, unadopted, one-line in limitations).
- **Aug:** 100 at pilot baseline; d128 per R-C0-4.
- **Ops:** one pod, one-shot amputation, ×2 seeds at d128 MANDATORY, banked 5k grids (the temporal-ensemble asset), screens at ≥3 ckpts + vb, deadline caps + node-side delete guard per the standing runbook.
- **Reporting (the §4 protocol table, every row):** cold · B=1 (vote@1) · Top-1-residual@128 · unverified-majority@128 · verified coverage@128 (+ curve/records) · t=256 depth read once. Benchmark conventions untouched: the SAME seeded 1k training rows, test set touched only by the evaluator, vsel from train-split val monitors, final AND vsel reported labeled.
- **d128 cost:** ≈$150–250 native (INFERRED; canary re-prices). **Champion bands set from pilot data at its own registration** — the floor claim we expect to defend: beat HRM (55.0) on single-prediction at ~10–20× fewer params; TRM-class (87.4) per-draw = stretch; EqR-class not promised.

## §5. Risks, fallbacks, calendar

- **RI composition risk** (A9-d16 fail; B1-d64 fragile transient): that is WHY P3 runs at pilot scale ×2 seeds before any d128 spend; R-C0-2/-5 route every outcome.
- **New-operator risk** (factorized mixers): CI gates + smokes + harness before silicon; fallbacks §2.
- **Efficiency estimate risk:** all costs INFERRED until canary; projection gate holds the launch if >1.2× band.
- **Comparability:** the champion is a Sudoku-specialized model (precedent: TRM's MLP-mixer Sudoku variant) and its numbers DO NOT enter the law tables (different architecture class); the law corpus stays on the shared canvas models — one sentence in the paper.
- **Calendar (freeze rule):** build Sep 1–2 → pilot night Sep 2–3 → verdict + champion registration Sep 3 → champion night Sep 3–4 → ARC-d96 extension Sep 4–6 → drafting from ~Sep 6. Champion numbers in by ~Sep 10–12 or the section ships in the AAMAS version; the ICLR paper stands on the review's structure regardless.
- **Budget:** pilot + champion + ARC ext + riders ≈ $300–480 → program lands ≈$2.6–2.8k of $3.7k; rebuttal reserve intact.

## §6. PI decisions embedded here (ratify/adjust at registration)

1. Seven pilot arms as listed (P5/P6 are the droppable two if the wall binds).
2. ri_p = .5 (A2's value; the only prior) — alternative .25 if cold-dilution worry dominates.
3. The champion's T16 (vs staying T12) and the two-phase split points (35k/15k, 50k/30k).
4. The k=1024 canvas-C3X coverage rider: ride the pilot node or skip.
5. Confirmation that canvas-d128 is deferred post-abstract (ratified in-session 2026-08-31; restated for the record).
