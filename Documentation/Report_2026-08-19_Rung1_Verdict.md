# Rung 1 (d64 × seeds) — Verdict, Physics Reading, Adversarial Pass, and Where Every Claim Stands

*2026-08-19. Companion to `Design_Ledger.md` (the 2026-08-15 RUNG-1 LAUNCH REGISTRATION is the pre-registration; the 2026-08-19 VERDICT entry is the record) and to `Basin_Measurement_Note.pdf` (the Professor's review log — its claims are re-graded in §6). Every number below is computed by `tools/analyze_r1.py` (registered rules R1-1..R1-6, written before the data existed; artifact `runs/analysis/r1_verdict.txt`) or `tools/analyze_r1_physics.py` (artifact `runs/analysis/r1_physics_20260819.txt`); none is hand-typed.*

---

## 0. What was measured

**Campaign.** Rung 1 of the width ladder: d64 / T6 / 53,333 steps (the d48-proportional budget), batch 64, C20+ConceptARC+orbit×4 corpus (1,803 tasks / 12,619 pairs), knee price β=3e-5 / β_nl=1e-5, floors 350/75/50/15/30 where used; 763,420 bulk parameters. Arms: **A4** floors+NI (σ=.01) seeds {0,1,2} — the rung-0 operating substrate; **A3** plain seeds {0,1}; **A5** global-priced+NI seed 0 (the floors decomposition owed by rung 0). Anchor: the rung-0 seeded d48 cells (A4 ×3, A3 ×3, A2 global ×3, A1 floors ×3 at d48/T6/40k). Comparators (n=1): the wave-2 d64 cells C53 (global, 53k), C80 (global, 80k), Dfloor (floors, 53k), Dri (RI), B.

**Instruments** (Basin note §2): retention N=S(0) (codebook size, truth-restricted), corruption ladder S(ε), ε∈{.05,.1,.2,.4} (code distance; r̄ = mean max-survived ε), exact@T (quench reachability), e1e3 (convergence / wrong-stable / endpoint condensation), throat I_s per RG cut (priced information), and the gates val-hard (144 pairs, 16 families × 9), rg-96 = rg-48 ∪ rb-48 (288 pairs, never-trained families), rt-48 (fresh instances of trained families).

**Admission.** All six ckpts are the registered config at artifact level (d=64, T=6, step 53,333, arm flags exact) — admitted. All 24 battery files 48/48 tasks. Resume caveat (ledger 08-18→19): A3s0, A5s0, A4s1 resumed from live ckpts; not bit-identical to uninterrupted runs; rung 0 symmetric.

**Measurement law** (binding): n≥3 or within-run pairing for claim-bearing contrasts; single-run cross-run deltas under ~10 counts are non-evidence. Cells are labeled n=3 / n=2 / n=1 throughout.

---

## 1. The registered verdicts (R1-1..R1-6, as locked 2026-08-15)

| rule | reading | verdict |
|---|---|---|
| **R1-1 (H-36) width ceiling** | d64 A4 vs d48 A4, per seed-pair: rg-96 retention 27/23/24 vs 34/35/34 (d64 lower 3/3); rg-96 S(.2) 8/11/6 vs 7/12/11 (lower 2/3); val-hard S(.4) 15/21/18 vs 21/24/19 (lower 3/3). "d64<d48 on both transfer metrics" in 2/3 pairs; no pair has d64≥d48 on both. | **WIDTH CEILING CONFIRMED at d48** (per the registered rule) → ladder goes **depth-lean** (T-scaling at d≤48). See §4 for the two alternative mechanisms that the rule cannot separate and the cheap cell that can. |
| **R1-2 (H-30 restated)** | priced A4 η = .229/.227/.245 (mean .234), A5 .235; plain A3 .246/.245 (mean .246); \|plain−priced\|/priced = .05 | **HOLDS**: priced lands in .20–.28, plain within 15% (the price effect on η is gone at d64 — in fact plain ≥ priced). |
| **R1-3 (Law 4 at d64)** | A4 vs A3 on rg-96 retention: 27/23/24 vs 11/9 — priced > plain in **6/6** seed comparisons, McNemar p = .002/.001/.043/.001/.024/.001 | **Law 4 EXTENDS to d64** (transfer dividend seeded at width; ~2.5×). |
| **R1-4 (NI attribution, directional n=1)** | A5 (global+NI) rg-96 = **34** vs A4 band [23,27] (spread 4) | **A5 ABOVE the A4 band** — at d64 the floors *cost* unseen-family transfer with NI present (n=1; 9.6 counts, at the measurement-law threshold). On val-hard A5 ≈ A4 (N 43 vs 40.7; S(.4) 22 vs 18; exact 11 vs 10). |
| **R1-5 (A4 seed-invariance)** | rg-96 {23,24,27} spread 4 (rung 0: {34,35,34} spread 1) | **INTERMEDIATE**: the stabilizer claim is weaker than rung 0's spread-1 (partly luck) but not refuted; A4 is still far tighter than plain (A3 {9,11}) and than rung-0's A1 floors {17,28,32}. |
| **R1-6 (throat / profile)** | A4 I_med 535/521/541 (mean 532) vs d48 A4 611/560/588 (586): **−54 nats**; A5 468; all priced d64 cells on the **knee** profile (L1 .035–.061), plain on **free** (.242/.301 vs free .031/.163) | **Throat declines with width (Law 1 extends); H-34 two-profile structure HOLDS** (second seeded test). |

---

## 2. The hypotheses and laws — stated cleanly, and what rung 1 did to each

Legend: **HOLDS** (new data consistent, often strengthened) · **EXTENDS** (holds at a new scale) · **RESTATED** (data forced a sharper form) · **STRAINED** (data push against it without killing it) · **FALSIFIED** · **NO NEW DATA**.

### 2.1 Thesis statements (S1–S4, ledger §3c)

| | statement (clean) | rung-1 reading | status |
|---|---|---|---|
| **S1 Flux-Floor Law** | Per-cut flux I_s is bounded below by the distributional information complexity of the cut function; under a price β>0 the total I_total is task-determined and converges toward that bound as capacity grows (the "throat"). | Priced throat at matched arm 586 → 532 nats (d48→d64, n=3 each; −9%); global-priced series now 785/657/635/602 (d16–48@20k) → 519 (d48/T6/40k) → 452 (C53 d64/53k) → 432 (C80 d64/80k); A5 (global+NI) 468. The decline is **UV-led**: per-cut Δ(d64−d48) = [−41.5, −7.4, −6.2, −4.0, +0.4] nats — the coarse cuts (s3+s4 ≈ 45–50 nats) are **width-invariant**. | **EXTENDS** — and acquires its first structural detail: capacity improves the UV code; the IR content behaves as a task-determined constant (the floor S1 predicts lives in the IR). |
| **S2 Excess-Flux Criterion** | Excess flux above the floor tracks memorization (LoO gap / fewer, narrower basins), causally under β-intervention. | Plain d64 carries 313k nats (**1.50× d48's 209k**) and holds N 22.5 (vs 28.3 at d48; spread-limited) with rg-96 10 (vs 16.3) — free capacity goes into flux, not codebook. Priced vs plain at d64: 600× less flux, 1.8× more codewords, 2.5× more transfer. | **HOLDS** (its widest-point instance yet). Caveat: within the *priced* family across width, less flux came with *fewer* basins — S2 governs the excess, not the priced-regime width effect (§3.3). |
| **S3 Locality-Class Law** | The per-cut envelope selects the minimal architecture class (priced attention at all scales). | No attention-class arm this rung. | **NO NEW DATA** |
| **S4 Rule Code Distance** | Erasure-robustness of discrete rule selection. | Not probed. | **NO NEW DATA** |

### 2.2 The four laws (Basin note §4; ledger 2026-08-12)

| | law (clean) | rung-1 reading | status |
|---|---|---|---|
| **L1 Throat** | I_total is task-determined and mildly decreasing in capacity under β>0; free models carry orders of magnitude more with no benefit. | See S1: priced −9% at d48→d64 (n=3); free +50%. | **EXTENDS** |
| **L2 Separability** | Codebook size N and code distance r have separate physical controls (N: depth, corpus, dials; r: β at mid-width, budget). | At d64 (priced, linear budget) **N fell (47.3→40.7, 3/3 seeds) while r̄ stayed flat (.235→.236)**, radius ratio S(.2)/N .69→.71 — count moved, distance did not. Plain: r̄ .262→.266, N 28.3→22.5. | **HOLDS** (a new instance: width past d48 moves N down without touching r). |
| **L3 (amended) Radius is budget-limited; with steps scaled to capacity width keeps buying code distance** | — | With steps scaled linearly to d (53,333), r̄ did **not** rise (.236 vs .235; rg radius ratio .34 vs .29 on fewer codewords); the d48/40k "record radius" was the anchor, and seeded d64 matches but does not beat it. C80 (80k, n=1) r̄ .23. | **RESTATED**: code distance **saturates by d48** at this price/depth; beyond d48, linear-in-d budget buys neither distance nor count (§3.3 for why, and the β caveat). |
| **L4 Transfer-specific dividend** | Priced > free on never-trained families at every width; in-distribution differences are seed noise. | 6/6 at d64, all p<.05 (27/23/24 vs 11/9). In-distribution: priced N 40.7 vs plain 22.5 is *not* noise at d64 (plain is worse in-distribution too, but plain's d48 spread was 12 — the in-distribution half of L4 stays "seed-limited", the transfer half is now seeded at three widths). | **EXTENDS** |

### 2.3 Tracked hypotheses (ledger §3)

| | statement (clean) | rung-1 reading | status |
|---|---|---|---|
| **H-30 (restated 08-15) flow-velocity law** | Trained damping η = f(width, depth); the pricing effect on η decays to nil with width; deeper T lowers η. | Priced .234 (n=3), A5 .235, plain .246 (n=2): price effect **gone** (ratio 1.05, sign reversed vs narrow substrates); prediction η≈.24 hit. Depth half untested (no T4 seed at rung 1). Across cells η tracks total optimization: d48/40k .16–.19 → d64/53k .23–.25 → d64/80k .285 (C80, n=1). | **HOLDS (width half)**; depth half still owed. Honest form: η is set by optimization length (and/or width) and is price-independent at d≥48. |
| **H-31 NI thermal tax is a UV-localized Shannon rent, monotone in σ** | — | At d64: A5−C53 (global+NI vs global, n=1 each) = **+18 nats, UV-concentrated** ([+9.6, +8.5, +1.3, +3.5, −4.5]); rung-0 clean tax at d48 was +12 ([+5.3, +2.4, +0.5, +3.2, +0.1], n=3). σ-monotonicity untested. | **HOLDS (localization, 2nd sighting)**; dose half NO NEW DATA. |
| **H-32 two-sided S2 (sub-floor nats load-bearing)** | Global-β compression below the per-scale floors erodes deep-tail basins; floors protect them. | Rung 0 killed it (sub-floor nats not load-bearing). Rung 1: A5 compresses **below the floor vector on 4/5 cuts** (314/77/39/11/21 vs 350/75/50/15/30, total 468) and is the **best** d64 cell on rg-96 (34) with the best deep tail (S(.4) 22); A4 sits AT the floors (365/67/47/16/31) and is worse on transfer. | **KILL STANDS, and the sign may be reversed** at d64 (n=1): forced UV nats above the natural code may cost transfer. → H-39 below. |
| **H-33 / H-37 landscape-class / init-distribution laws** | — | S-port only. | **NO NEW DATA** |
| **H-34 knee-profile as scaling diagnostic** | Cells on the knee profile (L1≲.10) scale healthily; deviation ≳.15 localizes a capability cost to the starved channel's families. | All priced d64 cells on-profile (.035–.061); plain on free. The d64 cells show a capability cost (N −6.7) that is **diffuse** (CompleteShape −2.3, HorizontalVertical −2.0, Order −2.0, TopBottom2D −1.3, InsideOutside −1.0; AboveBelow +1.7 — all at 9-pair family resolution) — no starved-channel signature. | **HOLDS in its precise form** (no deviation ↔ no *localized* cost) — but a **LIMITATION is now explicit**: an on-profile cell can lose codebook size diffusely; the profile is a necessary, not sufficient, health indicator. |
| **H-35 Center width-casualty (watch)** | Center-family retention declines with width past d48 (d48 5/9 → d64 0–2/9). | Seeded: d48 A4 {2,0,1} → d64 A4 {2,0,2}; A5 2; plain {2,1}. The "5/9 at d48" was the single p1248c40k cell. | **FALSIFIED (as a width effect)** — Center is simply low (~1–2/9) at both widths; the wave-2 reading was a comparator artifact. |
| **H-36 width ceiling** | Priced-basin geometry (radius, rg transfer) stops improving past d48. | R1-1 fired CEILING: N, exact, rg-96 all lower at d64 in 3/3 seed-pairs; r̄ flat. | **CONFIRMED as registered** (with the mechanism question open — §4.1/§5). |
| NI as operating substrate (rung-0 R6) | NI regularizes transfer geometry and collapses its seed variance. | A4 spread 4 (rg-96) vs plain 2 (on n=2) and rung-0 A1 15; A4 wins every val-hard metric vs plain; A5 (NI without floors) ≥ A4. | **HOLDS (NI)**; the *floors* half of A4 is now the weakest component (n=1) → A5-class candidate. |
| Vacancy floor (W-α) | Copy and ExtractObjects have no basins on any substrate (content-verbatim families are architecture-vacant). | 0/9 on all six seeded d64 arms → **0 on 37 substrates**; Count 1/9 on A4s0, 0 elsewhere. | **HOLDS** (the program's hardest invariant, now at seeded d64). |
| Reachability wall | ~86–90% of converged limits are wrong; exact@T ≪ N. | d64 A4: wrong-stable .91–.94 of converged (d48 .88–.90); exact/N .20–.31 (d48 .28–.34); exact-but-not-retained = 0 in every d64 cell (exact-when-reached is a fixed point, rung-0 finding). | **HOLDS** — slightly worse at d64 (§5.5). |
| Two-fixed-point spectral structure (cluster O) | Priced codes collapse on the knee profile, free on the free profile, independent of width/seed. | d48 and d64 priced profiles coincide (UV share .68–.71, IR .085–.091); plain .78–.84 UV. | **HOLDS** (width-invariant scaling function, third seeded test). |

---

## 3. The physics reading

### 3.1 RG / holography — the profile is a fixed point; capacity improves the UV code, the IR is conserved
The per-cut flux of every priced cell lies on the knee profile [.69, .14, .085, .035, .048] at d48 **and** d64 (L1 ≤ .061 across six seeded cells); the floor vector itself is the knee profile (d=.048). Width changes the *amount* of information, not its distribution across cuts: the d64−d48 delta at the matched arm is −41.5 nats at the finest cut and ≈0 at the two coarsest (s3+s4: 47→47). Read as an RG statement: the coarse-grained (IR) content a task forces through the network is a width-invariant constant of the task distribution (S1's floor lives there), and additional capacity is spent making the fine-scale (UV) code cheaper. Free codes, by contrast, inflate (×1.5) with width and stay on the free profile — there is no fixed point pulling them toward the task's information content.

### 3.2 Code geometry — a ceiling in codebook size, not in code distance
In the (N, r̄) plane the seeded d64 A4 point is (40.7, .236) against d48's (47.3, .235): fewer codewords at identical typical distance; d64 is frontier-interior exactly as the wave-2 n=1 cells suggested (C53 (42,.22), C80 (38,.23)). Unseen-family transfer falls further (rg-96 34.3 → 24.7) while fresh instances of *trained* families are flat (rt-48: 27 vs 26, n=1). Width at fixed price and linear budget therefore does not dilute the code's *distance*; it shrinks the *inventory* and does so in proportion to distance from the training distribution (§5.1).

### 3.3 Information theory — rate–distortion at fixed β, and the two readings of the ceiling
Two mechanisms produce "fewer codewords at d64" and the registered rule cannot separate them:
- **Geometric/dimensional:** the task set supports a finite truth-codebook; more dimensions do not create more true fixed points but do create more spurious ones (§5.5) — a capacity-independent inventory (the 08-14 "inventory-limited" reading).
- **Pricing (rate–distortion at fixed β):** the knee β=3e-5 was calibrated at d16/20k. Larger, longer-trained models are more compressible (L1), so at fixed β the optimum migrates to lower rate; the extra compression is paid in unseen-family basins. Evidence in both directions: C53→C80 (same arm, 53k→80k, n=1) compresses 452→432 nats and loses N 42→38, rg-48 12→6 — *more budget at fixed β makes transfer worse*; yet A5 compresses to 468 (below the floors) and has the best transfer of any d64 cell. The discriminating cell is cheap and named in §7 (β-rescale at d64).

### 3.4 Dynamics — the flow constant
η rises with optimization length/width and no longer depends on price: .16–.19 (d48/40k, all arms) → .23–.25 (d64/53k, all arms) → .285 (C80). A weak negative rank correlation with N across 22 cells (ρ=−.34) is fully confounded (width, budget, RI, B) and is *not* claimed; the dials arm at d16 (η floor .228) had the most basins of its grid, which refutes any simple "fast flow sheds codewords" reading.

### 3.5 Landscape inventory — width adds attractors faster than truth basins
E1/E3 at d64 (A4 ×3 vs d48 A4 ×3): convergence 140/144 (vs 134–143), wrong-stable fraction .91–.94 (vs .88–.90), limit-exact 8–14 (vs 14–17), and endpoint condensation from random inits weaker (nd=1 pairs 33/37/44 vs 49/49/52). The d64 map converges at least as reliably but to a larger set of *distinct* endpoints, more of them wrong — the inventory grows, the truth-restricted part does not. This is the same landscape geometry H-36 reports, seen from the dynamical side.

---

## 4. Adversarial pass — what these numbers do not show

1. **The ceiling is protocol-bound.** "d64 < d48" holds at: fixed β=3e-5 (calibrated at d16/20k), fixed floor vector (calibrated at d64 wave-2), T=6, batch 64, steps linear in d. Each is a dial; the β dial is the one with a plausible mechanism (§3.3) and a cheap test. The registered planning call (depth-lean) stands because it was the pre-registered consequence of the rule — but it should be read as "depth-lean *at this price*", and the β-rescale cell should run before the ladder commits d96+ resources.
2. **Budget.** The linear steps law was assumed (53,333). C80 (80k ≈ the quadratic budget, n=1) did not recover d48's numbers — against a budget explanation — but it is one cell of a different arm (global, no NI).
3. **A5 is n=1.** Every R1-4 statement and the "floors cost transfer at d64" reading rest on one seed. The 9.6-count rg-96 margin over A4's band sits at the measurement-law threshold; its rg-48 half (17 vs 9.3) is 7.7 counts. Directional, registered for seeds.
4. **Family-level numbers are 9-pair counts.** Per-family deltas of ±2 are within binomial noise; only the aggregate N/rg contrasts are claim-bearing. H-35's refutation rests on seeded *aggregates* at both widths (Center ≈1–2/9 everywhere), not on a per-family delta.
5. **Plain at d64 has n=2, and d48 plain's spread was 12.** The in-distribution plain deficit (22.5 vs 28.3) is suggestive, not established; L4's in-distribution clause is unchanged.
6. **Cross-rung comparison is between-run.** The measurement law's 10-count single-run floor applies to individual pairs; the verdict rests on three same-sign seed-pairs with spreads 1–4 and mean gaps 6.7 (N), 4.3 (exact), 9.6 (rg-96) — above the floor — but the rg-96 S(.2) clause (2/3) is the weakest leg of R1-1.
7. **Resumes.** A4s1 (resumed at 6k and 35k) is the best A4 seed on S(.4) and within band elsewhere; no resume signature is visible, but the caveat stands: resumed arms are valid draws, not bit-identical replicates.
8. **Two I_med conventions coexist in the ledger**: median of per-pair totals (rung 0/1 analyzers: C53 452, C80 432) vs sum of per-cut medians (the 08-14 physics pass: C53 443). This report uses the former throughout; the 08-14 C53 figure of 443 is the latter. Differences are ≤16 nats and never change a reading.
9. **rg-96 vs rg-48.** The wave-2 comparators predate the rb-48 gate; comparisons with them use rg-48 only (d64 A4 9.3, A5 17, C53 12, C80 6, Dfloor 9, d48 A4 13.3).
10. **The knee profile is not a sufficient health diagnostic** (H-34 nuance): on-profile cells lost codebook size. Any future "on-profile, therefore healthy" inference is now known to be unsafe.

---

## 5. Curiosity pass — patterns the data offer (each with its n and its test)

1. **The width deficit grows with distance from the training distribution.** rt-48 (trained families, fresh instances) 27 vs 26 (+4%, n=1); val-hard (held-out hard tasks of trained families) 40.7 vs 47.3 (−14%, n=3); rg-96 (never-trained families) 24.7 vs 34.3 (−28%, n=3), both halves (rg-48 −30%, rb-48 −27%). Reading: at fixed β, extra capacity specializes the priced code to the trained family structure — generalization *within* families is preserved, generalization *across* families pays. Test: rt on all seeds at the next cell (one wave-hour); if the gradient replicates at n=3 it belongs in the paper as the shape of the width ceiling.
2. **The floors became binding at d64.** A4's spectrum sits at the floor vector (at/above on 3/5 cuts; s1/s2 slightly below), A5's sits below it on 4/5 cuts — the natural d64 code wants ~314 nats at the UV, the floor forces ~365. The floor vector was measured at d64 wave-2 on a global-priced cell *without* NI; with NI and width it is no longer the code's own fixed point. Together with R1-4 (A5 ≥ A4) this says the floors are a d48/wave-2 constant, not a law of the code — H-39.
3. **Over-compression at long budget (n=1).** C53→C80: −20 nats, N −4, rg-48 −6, exact +3, η +.04. At fixed β the rate keeps falling and unseen-family basins keep falling with it; reachability does not. If the β-rescale cell (§7) shows d64 at β=1e-5 recovering rg-96 toward 34, the "ceiling" is a knee that moves with capacity — the most consequential open question the rung leaves.
4. **IR conservation.** s3+s4 flux ≈ 45–50 nats at d48 and d64 for every priced arm and ≈ 30 nats even for A5 (the most compressed) — a candidate measurement of the task distribution's coarse information content, the quantity S1 predicts is bounded below. Test: does it stay put at d96 and at T8? If yes it is the paper's cleanest "floor" figure.
5. **Spurious-attractor proliferation with capacity.** e3: nd=1 pairs 49–52 → 33–44, wrong-stable .89 → .92, limit-exact 15 → 11 (n=3 each). Width adds distinct endpoints, few of them true — the dynamical face of the codebook ceiling, and a reason the quench reaches a *smaller* fraction of the codebook at d64 (.25 vs .30).
6. **Pricing buys existence, not reachability, at every width.** Absolute exact@T is ~equal for priced and plain (d64: 10.0 vs 9.5; d48: 14.3 vs 9.0 seed-limited) while priced holds 1.7–1.8× more codewords; plain reaches 42% of its small codebook, priced 25% of its large one. The conversion problem (basins exist, the cold start doesn't land in them) is untouched by width or price — consistent with everything since 2026-08-09.
7. **NI's variance collapse is real but smaller than rung 0 suggested**: A4 rg-96 spread 4 (vs plain's 2 on n=2, floors-only 15, global 5 at d48). The stabilizer is a factor ~2–4 in spread, not ~15.
8. **η sign reversal.** At d≤32 priced > plain in η (up to 3.2×); at d64 plain ≥ priced (.246 vs .234). Whatever set the flow constant on narrow substrates (price-induced coarseness) is gone at width; η now reads as an optimization-length clock. A d48@53k cell would separate "length" from "width" at zero design cost.

---

## 6. The Basin Measurement Note (2026-08-12) — claim-by-claim status with rung-0 + rung-1 data

| note item | then | now |
|---|---|---|
| Setup: learned damping η "measured 0.058" | pretrain8 (plain, d16) | **Update the dictionary**: η is substrate-dependent — .035–.10 plain-narrow, .14–.19 priced-narrow/20k, .16–.19 all arms d48/40k, .23–.25 all arms d64/53k, .285 at 80k; the EqR 0.05 coincidence is with our *plain narrow* regime only (ledger 08-14 E; rung 1 R1-2). |
| Def. 1 retention N = codebook size (truth-restricted) | — | Instrument unchanged; K=8 binary (life8=1.00, ledger 08-14 G-iv). Seeded d48 A4 N 47.3±1; d64 A4 40.7±3. |
| Def. 2 corruption ladder, r, S(ε), radius ratio; ε*≈.2 | — | Unchanged; basin-edge fuzz 9–15 non-monotone pairs/arm quantified (4–10%). r̄ flat across d48→d64 (.235/.236). |
| Def. 3 reachability (quench) ≪ retention | "central dissociation" | **Holds at every width**: exact/N .20–.34; exact-but-not-retained 0/144 in all d64 cells. |
| Def. 4 thermal capture (hops), Arrhenius decomposition | cluster Q | No new data this rung (anti-Arrhenius stands from 08-12). |
| Def. 5 throat I_s, prices as Lagrange multipliers | — | Unchanged; note the convention (median of per-pair totals). |
| Dictionary: β, β_nl "the code-distance dial" | — | **Amend**: at d≥48 the price dial moves *codebook size and transfer*, not typical distance (r̄ is flat across widths and arms ≈ .22–.28); "code-distance dial" held at d24–32 (L3 original). |
| Dictionary: rt/rg "conditioned on codeword-never-trained" | — | rg-96 now = rg-48 ∪ rb-48 (G4, rung 0); rt-48 = trained-family fresh instances. The width deficit orders rt (flat) < val-hard < rg (§5.1). |
| **L1** Throat | 785→602 (d16–48@20k); "free ~200×" | **Extends**: → 519 (d48/T6/40k) → 532/468 (d64 A4/A5) → 452/432 (global 53k/80k); free 200k→313k (×600 at d64); decline is UV-led, IR conserved. |
| **L2** Separability | N ↔ dynamics; r ↔ optimization under constraint | **Holds**: new instance (width past d48: N ↓, r flat). |
| **L3 (amended)** width keeps buying code distance with steps ∝ capacity | radius .80 record at d48/40k | **Restate**: code distance saturates by d48 at this price/depth; at d64 with linear budget r̄ is flat and N falls; whether a rescaled β moves the knee is the open cell (§7). |
| **L4** transfer-specific dividend | priced > free on rg at 4 widths | **Extends** to a 5th width, seeded: 6/6 at d64, p<.05 each. |
| Remark (ii) K=8 finite-horizon | stress-checked at 16 | unchanged |
| Remark (iv) oracle-diagnostic instruments | — | unchanged; rung 1 ran them on rg-96 + rt-48 without contamination (gates never enter any solve path). |

---

## 7. Registered consequences and the next cells (for PI decision; nothing launched)

1. **β-rescale at d64 (the discriminating cell for H-36's mechanism; ~1.5 pod-h):** A5-class (global+NI) at β ∈ {1e-5, 3e-6} (+ the existing 3e-5), seeds {0,1}, full battery. If rg-96 recovers toward the d48 level (≥30) at lower β → the width ceiling is a **fixed-β over-compression** (the knee scales with capacity; the ladder re-tunes β per rung and width is back on the table); if not → **geometric** (depth-lean stands). Registered as **H-38** (below).
2. **A5 seeds (×2, ~1.3 pod-h):** decides floors-vs-global at d64 (R1-4 at n=3). Registered as **H-39**.
3. **Depth lean at d48** (rung 2 candidate if H-38 reads geometric): d48/T8 and d48/T12 at steps ∝ T·d, A5-class, seeds ×3 — the count axis per L2; also the depth half of H-30 (η falls with T).
4. **rt on every seed** at the next cell (cheap, one wave) to seed the OOD-gradient pattern (§5.1).
5. **d48@53k** (one arm, one seed): separates optimization-length from width in η (§5.8) — piggyback, not a cell.
6. Rung-2 registration is **re-opened**: the 08-14 ladder's "insert d72 / d96" branch is not taken; the PI chooses between (1)→(3) sequencing or (3) directly.

**New registered hypotheses (added to §3):**
- **H-38 knee-β scales with capacity (fixed-β over-compression):** at fixed β=3e-5 the rate–distortion optimum migrates to lower rate as d and steps grow (L1), and the lost nats are unseen-family basins; the d64 "ceiling" is therefore partly a pricing artifact. *Test:* cell 1 above. *Falsified if:* d64 at β≤1e-5 does not raise rg-96 above the A4 band (≤27) at n=2, or raises throat without raising transfer.
- **H-39 floors are a width-specific constant, not a law:** the floor vector (measured at d64 wave-2, global, no NI) is above the natural d64+NI code; forcing the code up to it costs transfer; global+NI ≥ floors+NI at d64. *Test:* cell 2. *Falsified if:* seeded A5 falls inside or below the A4 rg-96 band.

---

*Artifacts: `runs/analysis/r1_verdict.txt`, `runs/analysis/r1_physics_20260819.txt`; data `runs/pretrainr1_*`, `runs/*_pr1*` (+ GCS `rr1`). Analyzers: `tools/analyze_r1.py` (pre-registered), `tools/analyze_r1_physics.py`.*
