# Program Review 2026-09-02 — After the Champion Pilot: Why EqR's Numbers Are Out of Reach, What the Pilot Did to the Paper, and the Experiment That Decides It

**Trigger (PI):** "The pilot runs were really disappointing… does this put our work on shakier ground? Why are EqR numbers unreachable? With all the results and knowledge we have, we should be able to do strictly better than them by adding our knowledge over the recipe. Why isn't that the case?" · **Inputs:** the complete ledger, `Report_2026-09-02_ChampionPilot_Verdict.md`, `Program_Review_2026-08-31.md` (claim inventory + adversarial pass, updated here), the series notes, and three protocol fetches run for this review (EqR full text; TRM and HRM training READMEs). · **Stance:** the answer is mechanistic and unflattering in one specific way; it does not retract any registered verdict, and it changes what the next round must contain.

---

## §1. The direct answer: it is a regime gap, not a knowledge gap

### 1.1 What the field's recipe actually is (fetched, quoted)
| | HRM (Sudoku-Extreme 1k) | TRM (MLP-T) | EqR | **ours (pilot)** |
|---|---|---|---|---|
| training puzzles × augmentations | 1,000 × **1000** (`--subsample-size 1000 --num-aug 1000`) | 1,000 × **1000** | HRM lineage; appendix C.1 (⚠ still not extracted) | 1,000 × **100** |
| distinct training examples | 1.0M | 1.0M | (as HRM) | **101k** |
| samples seen | `epochs=20000` × 1k ≈ **20M** (batch 384, ~52k steps) | `epochs=50000` × 1k ≈ **50M** | (as HRM-class) | 50k × 64 = **3.2M** |
| passes per augmented example | ~20 | ~50 | — | ~32 |
| weight decay | **1.0** (`weight_decay=1.0 puzzle_emb_weight_decay=1.0`) | **1.0** | (as HRM) | **1e-4** |
| learning rate | 7e-5 (1 GPU) / 1e-4 (8 GPU) | 1e-4 | — | **1e-3** |
| batch | 384 | (README silent; ≥384 in the lineage) | — | **64** |
| normalization in the cell | RMSNorm pre-norm transformer blocks | pre-norm blocks (+ EMA) | transformer blocks; damping λ=.05, noise β=.01 | **none** (FiLM only; residual GELU mixers; unnormalized z carry) |
| params | 27M | 5M | 5.03M (2 weight-tied blocks) | 1.71M |
| single-prediction Sudoku-Extreme | 55.0 (per TRM's table ⚠) | 87.4 ± 2 | 84.8 (D=16, no RI/NI) · 86.4 (RI+NI) · 93.0 (D=64, B=1) | 37.35 (P3s1, RI) |
| wall | ~10h on one RTX 4070 | < 20h on one L40S | — | 43 min pretrain + ~1.3h evals on a v6e-8 |

Three things in this table decide the question.

1. **EqR's per-draw number does not come from RI/NI.** Their Table 3: baseline map 84.8 % → +RI 86.0 → +RI+NI 86.4 at D=16, B=1. Randomized init and noise add **+1.6pp on Sudoku** (they add +37pp on Maze). Depth to D=64 adds +6.6 (93.0); breadth with the residual selector adds +6.8 (99.8). **The 85 % is the base recipe** — a normalized transformer cell trained on 1M distinct augmented examples for tens of millions of samples under weight decay 1.0 at lr 1e-4. Every mechanism we "walked" from the series (RI, NI, depth, breadth) is a small increment on top of a base we never built.

2. **We never ran their recipe.** The series-as-ablation-ladder plan (2026-08-21) walked their *mechanisms* on **our substrate under our regime**: our RG cell (no normalization, a compressive shared group mixer, the attention channel dosed shut), lr 1e-3 (10× theirs), weight decay 1e-4 (10⁴× weaker than theirs), batch 64 (6× smaller), aug 100 (10× fewer distinct examples), 3.2M samples (6–16× fewer). The pilot's two pathologies are exactly the ones that regime predicts and theirs prevents: **memorization** (aug 100 + wd 1e-4 + 32 passes per example → train CE 0.000; the grokking literature's memorization regime — Power et al. 2022, Nanda et al. 2023 — where weight decay is the known lever that converts memorization into generalization) and **state explosion** (no normalization anywhere; our attention toll closed one runaway channel and RI found another).

3. **Our knowledge fixed our pathologies, not theirs.** FPA repairs contractivity collapse (H-45); the β_nl dose closes the free attention channel (H-48); two-phase prevents memorization (H-49); the η cap will address the RI explosions (H-50). Each is a measured law of *our* regime. A pre-norm transformer at wd 1.0, aug 1000, lr 1e-4 does not enter most of those regimes in the first place — so "adding our knowledge over their recipe" adds remedies for diseases their recipe does not have. What our knowledge can add on top of their recipe is (a) **efficiency** (exact S9 makes digit permutation free — Ren & Liu spend ~10² orbit forwards recovering exactly this; the 3-adic constraint operator has the flattest difficulty curve we have measured; 1.7M params vs 5M), (b) **the coverage axis** (verified breadth; portfolio × attempts — our own column), and (c) **understanding** (the instruments explain *why* their recipe works — basins, funnels, flux, contraction — which none of the series papers can). None of those three is "strictly better single-prediction accuracy," and the honest expectation is that per-draw accuracy on top of a 93 % base moves by low single digits from our ingredients.

### 1.2 So why are EqR's numbers unreachable *for us right now*?
Because our substrate has never been trained in the regime that produces them, and the pilot proved the two consequences: at aug 100 / wd 1e-4 the map memorizes before it generalizes (P1: 99.9 % train / 19.4 % test), and without normalization the replacement-dynamics map RI needs is one excursion away from NaN (P2: 19 % of test trajectories explode at t=64). The remaining gap after fixing both — capacity (1.7M vs 5M) and depth of inference chains — is the ordinary ladder, and the canvas ladder's +8pp per rung suggests d128–d160 at the right regime lands in the 45–55 range on single prediction, not at 87–93. **Reaching 93 % per draw requires running the field's regime, not adding to ours.**

### 1.3 The implicit assumption that failed
The 2026-08-14 registration framed the comparison as "our stack on top of an FPRM-class substrate — strictly additive." That frame was right; the execution inverted it: the substrate under test has always been ours, and the field's substrate was never reproduced. The pilot is the first campaign whose readout makes the inversion visible, because it is the first where the *training regime* (not the mechanism) was the binding variable.

---

## §2. What the pilot did to the paper — claim by claim (updates the 2026-08-31 inventory)

| # | Claim (2026-08-31 grade) | Pilot effect | Grade now |
|---|---|---|---|
| 1 | Contractivity collapse (A) | retfm 1.00 on all 7 native arms under FPA — no collapse observed (FPA everywhere); no new evidence either way | **A**, scope note added (§3) |
| 2 | FPA repairs it (A) | 7/7 arms retfm 1.00 across regimes incl. RI and two-phase native — a 4th scale/regime replication | **A** (stronger) |
| 3 | H-48 free-channel lethality + closure-is-free (A−) | dose closed A on 7/7 (13–372 nats); zero capability cost again; **and the RI deaths occurred with the channel closed** → the scope sharpens (H-48 is the attention channel only; H-50 is a different class) | **A−** unchanged; scope statement is now mandatory |
| 4 | Law 1 / throat (B+) | native total flux ~12× lower with a two-cut pyramid — a geometry effect, not a law test; the "arity signature" was ill-posed (retired) | **B+**, plus a stated limit: the throat comparison needs matched cut structure |
| 5 | S2 spectrum (B+) | free stream flux inflates with training again (RI 1.2–1.4e5 vs 8.6e4); memorization = the S2 "excess flux buys memorization" statement realized at the puzzle level | **B+** (a new supporting instance) |
| 6 | Landscape-class law / D-catalog (A) | D5 extended (a memorizing Sudoku map acquires ARC's wrong-stable texture); D12 new (data-diversity dependence); D3 sharpened (verifier value is a property of the map class: ≈0 on init-invariant maps) | **A** (richer) |
| 7 | Cold ladder (A) | ladder continues: 21.2 → 25.3 → 33.5 → **37.35** (native RI, 1.71M); the pilot also exposes that the ladder's absolute level sits far below a regime-matched field baseline | **A** as a ladder; positioning per §1 |
| 8 | Verified breadth = coverage; verifier value regime-dependent (A as measurement, framing per review §1) | **the pilot answers the verification question mechanistically:** on RI (init-invariant) maps EqR's own residual selector recovers ≥98 % of the verifier's gain (t1r@128 52.49 vs 52.85) and the funnel is deep-narrow; on wide-funnel maps the verifier is worth +57pp. Our numbers in EqR's column now exist (b1 38.47; t1r@128 52.49) | **A**, and the review §1 repair is DONE for the native arms (canvas riders still owed) |
| 9 | Ignition (B) | not tested (no multi-ckpt screen curve survived the deaths/collapse); P1's funnel ignited late (60.7 → 92.0 v256 by 35k) before memorization killed it | **B** unchanged |
| 10 | Toll dissociation (B+) | attention toll again stability-only at zero cost | **B+** |
| 11 | Two-phase T6 route (B, n=1) | **replicated in a new role**: the floor phase prevented memorization on the record arm (P3s1 CE plateau .04–.05) while every 50k-cosine native arm memorized | **B+** (n=2 across regimes; the mechanism now has two readouts) |
| 12 | Equivariance by construction (A) | unchanged; aug 1000 is still needed for the *position* group — the pilot measured it (+5.70pp at 10k) | **A** with the position-group caveat made explicit |
| 13 | Portfolio law (B) | P2 ∪ C3X verified@128 = 92.31 on the identical 20k; RI-native and canvas-breadth maps are complementary by mechanism (per-draw vs coverage) | **B+** (mechanism for the decorrelation) |
| 14 | Instrument suite + loop (A) | +2 laws (H-49, H-50), +1 instrument (explosion census), the memorization diagnosis from CE/val/train-split in one pass; the pre-registered pilot rules read every branch honestly (letters + labeled mechanisms) | **A** |
| **NEW 15** | Memorization / data-diversity law (H-49) | measured with a clean paired remedy (aug1000 +5.70pp) and puzzle-level proof; connects to grokking's memorization regime and to the field's aug-1000 + wd-1.0 convention (it explains *why* the convention exists) | **B+** (one campaign; sportC1 R-C1-0 is its test) |
| **NEW 16** | RI init-invariance + per-draw law (H-37 sharpened) and the η→1 explosion class (H-50) | b1 ≈ cold per octile; +34–38pp per draw; explosion census threshold η ≈ .97–.98; EqR's own +1.6pp RI gain on Sudoku vs our +34pp is explained: RI matters when the base map has no random-init capability (ours), not when it already converges from anywhere (theirs) | **B+** (mechanism; the η-cap arm is its test) |

**Net:** no standing law weakened; three strengthened (2, 8, 11); two added; the D-catalog gained three rows. The science is on *firmer* ground than on 08-31. What weakened is the **positioning**: the field-column gap is now measured and large (38.5 vs 93.0; 52.5 vs 99.8; 37.4 vs 55/87.4), and §1 shows the reason is a regime we have not entered — which a reviewer will also see.

---

## §3. The risk the pilot made visible: regime-specificity of the laws
The sharper reviewer objection is no longer R2 ("you are far below the leaders") but this: **"Your laws describe pathologies of an under-regularized, un-normalized, small-batch regime that the standard recipe never enters — collapse, free-channel explosions, memorization, RI blow-ups. What do they say about models trained the normal way?"** Our evidence for transfer, honestly graded:

- **Transfers (external evidence exists):** the fixed-point violation / truth-erasure family (H-45 / E3b) is reported inside HRM by Ren & Liu (their "fixed-point violation": correct solutions unstable, latents corrupt found answers) — a normalized, wd-1.0, aug-1000 model shows the H-45 class. The depth-before-breadth interaction, the RI effect direction, and the residual-selection mechanism all replicate across the two substrates in the expected directions.
- **Plausibly regime-specific:** H-48's free-attention explosion (pre-norm + QK-norm/z-loss are the field's standing fix — our toll is our version of it); H-49's memorization (aug 1000 + wd 1.0 are the field's standing fix); H-50's η→1 explosion (normalization again). These are still findings — they *explain the field's conventions from first principles* (why aug 1000, why wd 1.0, why pre-norm/QK-norm) — but the paper must state them as laws of a regime, with the field's recipe as the counterfactual.
- **Unknown until measured:** whether the basin/funnel/flux instruments read anything *new* on the field's own models. This is the 08-31 review's R4 (instrument portability on a public HRM/TRM checkpoint). After the pilot it is no longer a post-abstract nicety; it is the paper's load-bearing external-validity test.

---

## §4. The experiment that decides it — and the only route to "strictly better"
**Reproduce the field's base recipe on our infrastructure, then add our ingredients one at a time.** Concretely (a PI decision; cost is small on the native/TPU stack):

1. **Baseline reproduction (control):** an EqR/TRM-class cell in our codebase — 81 tokens, pre-norm transformer blocks (RMSNorm, RoPE or learned positions), the damped iterate with λ=.05, D_train=16 with SOT-style detach, RI/NI optional — trained in **their regime**: aug 1000, weight decay 1.0, lr 1e-4, batch 384, ~20M samples. Target: reproduce ≈85 % single-shot at D=16 / ≈93 % at D=64. If we cannot reproduce it, the gap is in our understanding of their recipe and nothing else matters until it closes. (Compute: a 5M transformer at batch 384 for ~52k steps is ~1–3 h on a v6e-8 — cheaper than the pilot.)
2. **Regime-matched native arm:** our native9 recipe under their optimizer regime (aug 1000, wd 1.0, lr 1e-4, batch 384, RMSNorm added to the cell as the FPRM kit the ledger registered-unneeded at d96). This separates "our architecture" from "our regime" — the question the pilot could not answer.
3. **Additive toggles on the reproduced baseline** (each one variable): + FPA anchor rows (does it move retfm/funnel on their map?), + attention toll (stability at zero cost?), + exact S9 (drop digit-aug: same accuracy at 9× fewer augmented forwards?), + the 3-adic group mixer (hard-octile curve?), + val-selection + two-phase, + verified breadth (our column). **"Strictly better" is only claimable from this design**, and it is also the design that produces the paper's most defensible sentence: *the field's recipe read through our instruments*.
4. **Instrument portability** (R4) falls out of (1): retention, ret_sched, λ, the funnel (ρ,r), the explosion census, and the flux ledger run on the reproduced baseline for free.

**Calendar (freeze rule stands):** the baseline build is ~1–2 days; its training is hours; the additive toggles are one night. It fits before ~Sep 10–12 **only if it replaces, not follows, the d128 native round as currently drafted** — see §5.

---

## §5. Recommendation to the PI (three options, one recommended)
- **Option A — run sportC1 as drafted** (native d128, aug 1000, two-phase, RI ×3 + η cap, canvas partner). Delivers a better native ladder point (likely 45–52 cold) and the champion section, but leaves §3's objection unanswered and cannot support any "better than EqR" sentence.
- **Option B — replace sportC1 with the §4 program** (baseline reproduction + regime-matched native + additive toggles). Answers the PI's question experimentally, closes R2/R4 together, and reframes paper 1's performance section as *the field's recipe under our instruments, plus the measured value of each of our ingredients*. Risk: reproduction may fall short of 85 % on the first try (their appendix hyperparameters remain partly unextracted — the ⚠ on EqR's dataset must be closed by reading appendix C.1/D.1 before building).
- **Option C (recommended) — sportC1 amended:** keep the native d128 champion arms (A0–A2 + η-cap B0; they are cheap and the ladder point matters) **and add the two decisive arms now**: the **baseline reproduction** (TRM-MLP-class or EqR-class cell in their regime; ~$20–40) and the **regime-matched native arm** (our cell in their regime; ~$15). Drop C0/E0/F0 to make room. One night, ≈ $150–200. The verdict then reads three things at once: the native ladder, whether our regime or our architecture explains the gap, and whether the field's number reproduces on our stack — which is the prerequisite for every additive claim in the paper.

Whatever the choice, two paper-level edits are no longer optional: (i) every law carries its regime scope and the field's convention as the counterfactual (aug 1000, wd 1.0, pre-norm) — stated as *what the law explains*; (ii) the comparator table gains a "training regime" column (examples, augmentations, wd, lr, batch, normalization) beside the protocol column — the reader must see that the 37 vs 87 comparison spans a 16× sample gap, a 10× augmentation gap, a 10⁴× weight-decay gap, and a normalization gap, not only a params gap.

---

## §6. Literature placements this review adds
- **Memorization vs generalization on small algorithmic datasets** (Power et al. 2022 "Grokking", arXiv:2201.02177; Nanda et al. 2023, arXiv:2301.05217; Liu et al. "Omnigrok" arXiv:2210.01117): weight decay and data-set size/diversity govern whether training memorizes or generalizes. H-49 is this regime measured with basin/dynamics instruments on a reasoning model; the field's wd 1.0 + aug 1000 is the standard remedy, and our two-phase floor is a schedule-side remedy the grokking literature does not report.
- **Normalization as the stabilizer of deep unrolls** (pre-norm/RMSNorm; QK-norm and z-loss, arXiv:2309.14322; σReparam, arXiv:2303.06296; FPRM's stabilizer theorems): H-48 and H-50 are the un-normalized substrate's failure modes; the toll is a measured, information-theoretic *alternative* to normalization for one channel, and the explosion census is a portable readout the normalization literature lacks.
- **Test-time compute** (self-consistency, Wang et al. 2022; Large Language Monkeys, arXiv:2407.21787; AlphaCode filtering): the pilot's result that selector value depends on the map class (init-invariant vs wide-funnel) is a mechanism these papers do not have.
- **The series** (HRM → TRM → EqR → FPRM): our measured per-mechanism effects now sit beside theirs — RI +34pp (ours) vs +1.6pp (EqR, Sudoku) with the explanation (base-map random-init capability); depth +3.6pp at t=256 (ours) vs +6.6 at D=64 (theirs); breadth +14pp (ours, init-invariant) vs +6.8 (theirs, residual-selected).

---

## §7. Bottom line for the PI
1. **Shakier ground? No for the science, yes for the story we cannot yet tell.** No law weakened; two were added; the verification question the 08-31 review flagged is now answered mechanistically. But the pilot exposed that our numbers and the field's were produced in different training regimes, and until we run their regime we cannot say whether the remaining gap is architecture, regime, or scale.
2. **Why EqR is unreachable today:** their 85–93 % per draw is the base recipe (normalized cell, aug 1000, wd 1.0, lr 1e-4, batch ≥384, 20–50M samples); RI/NI are +1.6pp for them. We trained 6–16× fewer samples on 10× fewer distinct examples with 10⁴× weaker weight decay, a 10× hotter learning rate, and no normalization — and the pilot showed the two pathologies that regime produces. Our knowledge cured *our* diseases; their recipe never had them.
3. **How to be strictly better, honestly:** reproduce their base on our stack, then add our ingredients one variable at a time (§4). The realistic wins are efficiency (params, S9-free augmentation), the coverage column, and the mechanistic account — not a large per-draw jump. Option C makes that test part of the next night's spend.
