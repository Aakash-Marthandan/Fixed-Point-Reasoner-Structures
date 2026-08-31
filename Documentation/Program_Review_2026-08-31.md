# Program Review — Claims vs. Literature, the Verification Question, and the Adversarial Pass

**Date:** 2026-08-31 · **Trigger:** PI directive at the post-freethink checkpoint ("evaluate all the original findings and results we have for the paper, set against the existing literature… my concern is that we're using verification and that can be problematic since EqR doesn't… I want an adversarial pass"). · **Inputs:** the complete ledger (all §5, July 16 → the B-M2 scan verdict), every verdict report, the D-catalog, the 2026-08-31 freethink, the related-work deep-reads, and three source-verifications run for this review (EqR full-text protocol; TRM/HRM series notes re-read; Ren & Liu). · **Companions:** `Related_Work_Series.md`, `Sudoku_vs_ARC_Instrument_Map.md`, `Research_Brainstorm.md` (Freethink 2026-08-31). Nothing here changes a registered verdict; recommendations are PI decisions.

---

## §1. THE VERIFICATION QUESTION — verdict first

**The PI's concern is CONFIRMED, and the source-verified facts make it precise.**

### 1.1 The facts

- **EqR's headline statistic is verification-free selection.** Full-text quote (arXiv:2605.21488, fetched 2026-08-31): *"convergence-based selection picks the run with the smallest average residual over the final few iterations (L=3) and checks whether its prediction is correct."* The 99.8% (D=64, B=128) = **Top-1-Converged**: one trajectory chosen by an internal signal (fixed-point residual), then scored. No validity/constraint checking anywhere in the paper; accuracy is token-exact on the selected answer.
- **EqR's per-draw accuracy is ~93%.** Their Table 4: **B=1, D=64 = 93.0%** on Sudoku-Extreme. Breadth + residual selection buys 93.0 → 99.8 (+6.8pp). Their B=1 draw is a single random-init trajectory — statistically the same object as our `vote@1`.
- **Our per-draw accuracy is ~29–36%.** C3X vote@1 = 28.9%, D4 vote@1 = 36.0% (20k scan; note D4's single *random-init* draw beats its VOID-start cold 33.3 — the H-37 deployment-distribution effect at d96). Our verified-vote@128 (88.9% C3X) is a **coverage** statistic: a draw counts iff it is a valid, givens-consistent completion (= the unique solution).
- **The verifier's value is regime-dependent, and we measured that.** D3 demo cell: verified vote vs. unverified majority @k=128 = 68.8 vs 18.8 (B2-d64), 70.7 vs 5.1 (S5-d16) — worth +50–65pp in our per-draw regime. In EqR's 93%-per-draw regime a verifier is worth ≈0 (they are near ceiling without one). LLMonkeys (arXiv:2407.21787) reports the same law at LLM scale: unverified selection plateaus; verified coverage keeps scaling.
- The series' other Sudoku numbers are **single deterministic predictions** (HRM 55.0 per TRM's table; TRM-MLP 87.4), and augmented-HRM's 96.9 (Ren & Liu) is **orbit majority voting** (~10² forwards, non-halting discarded, verification-free).

### 1.2 The verdict

1. **Our verified-vote numbers are NOT comparable to EqR's 99.8, TRM's 87.4, or aug-HRM's 96.9 — and the paper must never place them in the same column.** Three different statistics are in play across the field (single-prediction; internal-signal selection over draws; orbit majority), and ours is a fourth (verified coverage). B-M2 (88.9 ≥ .85) stands as what it always was in the ledger: an **internal milestone on a named statistic**, pre-registered and honestly labeled. The early band nickname "M2 = TRM-class" must be struck from all paper language — TRM's 87.4 is a single-prediction number and our 88.9 is 128 verified draws.
2. **The honest EqR-comparable rows are:** (a) their B=1 (93.0) ↔ our vote@1 (28.9–36.0) — we are ~57–64pp below at 2.4× fewer params; (b) their Top-1-residual@128 ↔ **unmeasured on our carriers** — the gap this review's repair plan closes. Using their own selector on our draws will land far below our verified number (selection among mostly-wrong draws by an internal signal); measuring it is cheap and makes the comparison bulletproof in both directions.
3. **Using verification is not a flaw — hiding the statistic would be.** On Sudoku the verifier is part of the problem structure (uniqueness ⇒ validity = correctness; our valid_wrong = 0.0000 × 2.1M makes it sound *exactly*), the sample-then-filter protocol has an established lineage (AlphaCode arXiv:2203.07814; LLMonkeys), and the D3 cell quantifies precisely what it buys. The paper's move is to present verified breadth as **its own axis (coverage under free verification)** with the verifier-value law as a *finding*, while reporting our numbers on the field's statistics alongside.

### 1.3 The repair plan (cheap, before the abstract)

| Rider | What it measures | Cost | Status |
|---|---|---|---|
| **EqR-statistic eval**: Top-1-by-residual selection over k=128 draws (their L=3 average-residual rule, implemented verbatim) on C3X + D4 ckpts, 5k subsample | our number in EqR's column | ~$5–10, eval-only | recommend REGISTER |
| **Unverified majority@k** on the same run (the Ren & Liu / self-consistency statistic) | our number in aug-HRM's column | same run, $0 extra | recommend REGISTER |
| vote@1 (their B=1 column) | already measured | $0 (in scan summaries) | done |
| Optional: verified-vote **union curve to k=1024** (5k subsample) | B-M3 adjudication + coverage-law model test (freethink Y) | ~$15–40 | PI menu |

All four numbers then live in one protocol-columned table (§4), and every cross-system sentence in the paper cites its column.

---

## §2. Claim inventory — evidence grade × literature position

Grades: **A** = paper-ready as stated (pre-registered, replicated or n-adequate, caveats named). **B** = paper-ready with a load-bearing caveat in the text. **C** = needs a measurement or a downgrade before print. Scale span and the nearest literature are given for each; "PRE" marks pre-registered decision rules.

| # | Claim (paper form) | Evidence | Grade | Nearest literature / differentiation |
|---|---|---|---|---|
| 1 | **Contractivity collapse**: plain equilibrium maps lose final-map contractivity with training/scale (retfm → .05–.5) while schedule retention stays ≈1; collapse accelerates with width | d16→d64 ×3 widths, both seeds where run; PRE (H-45 rules); precursor ordering λ>1 before retfm<.9 confirmed once (A8) | **A** | Ren & Liu's "fixed-point violation" = independent confirmation (they never fix it; we do). Exposure-bias/denoising-attractor lineage (Alain-Bengio). DEQ/monotone-DEQ (Bai et al. arXiv:1909.01377; Winston & Kolter arXiv:2006.08591) get contraction *by construction* — cite as the architectural alternative to our *trained* contraction |
| 2 | **FPA (anchor rows) repairs it**: retfm 1.00 at d16/d64/d96; seed-stable (1.66pp); widens basins (S(.4) .96 vs .62); rescues the shallow recipe at width | 3 scales; PRE; the one-variable control at d64 (B4 broken / B5-priced whole / FPA whole) | **A** | Our basin-objective attribution vs Ren & Liu's one-step-gradient attribution: our full-BPTT arms had the disease too — the objective is the ingredient. Growing-NCA damage training, SUNDAE lineage acknowledged |
| 3 | **Free-channel lethality (H-48) + closure-is-free**: unpriced attention flux (~10⁷ nats) × depth × width × hot-lr = state-explosion NaNs (0/5 free arms); β_nl=1e-6 closes it 7 orders at zero-to-negative capability cost (record cold rides the dose) | deaths 0/5 across 2 lrs/2 seeds/2 T; dosed clean 4/4 + priced 2/2; ONE-variable D3-vs-C1 PRE; no-precursor case-control (3 deaths/1052 windows) | **A−** (composition law depth×width×lr inferred across cells, not factorial — say so) | z-loss/QK-norm/σReparam (arXiv:2303.06296, 2309.14322) = the same fix as engineering trick; instability monitors (arXiv:2606.28116) = the contrast to our no-precursor result. **Ours adds the measurement**: the closed channel carried no load-bearing information (the ledger) + the mass-term reading |
| 4 | **Law 1 / throat**: priced information total is a task-set property — flat-to-declining across 27× params (ARC 785→432; Sudoku ~970–1600) — while free flux inflates | many cells, most n=1–2, three task sets; estimator = variational upper bound (named) | **B+** | Intrinsic-dimension line (Li et al., Aghajanyan et al.); IB-controversy dodged by enforced channels (vs Saxe critique); Generalization Ridge (depth-only, observational) — ours is per-cut, enforced, cross-domain. The S1 "sandwich" (analytic floors) closed only on constructed families — scope stated |
| 5 | **S2 spectrum**: excess (free) flux is deadweight → basin-narrowing → lethal; priced flux is the code (400× compression at equal-or-better retention; anti-water-filling allocation) | β-intervention direction ×many cells; PRE at several points; the registered LoO-gap correlation half was superseded by retention/transfer readouts (say so) | **B+** | CompressARC (global MDL, no cuts); VIB water-filling (refuted form = the finding); our enforced-channel MI is well-defined where IB's wasn't |
| 6 | **Landscape-class law / D-catalog**: same instruments, opposite verdicts — ARC inventory-limited (retention ~29–34%, depth +0, temperature anti-Arrhenius, 92% wrong-stable) vs Sudoku search-limited (retention ≈1 everywhere, depth +16/40, valid_wrong 0.0000, funnels) | replicated across eras; the D4 two-sided iteration figure; D3 verifier quantification; D5 exact | **A** (with the scale-match caveat: ARC frozen at d64 — extension registered) | HRM's own unexplained observation ("extra compute pays on Sudoku, minimal on ARC") — we explain the series' own anomaly. Nobody else separates the domains the benchmarks lump |
| 7 | **Cold ladder**: 21.2 → 25.3 → 33.5% full-test single-prediction cold at 78k → 0.96M → 2.11M params (no TTT, no verifier, 1k-convention) | paired-McNemar p≈0 between rungs; PRE bands; D4-record seed/lr confound named (lane n=2: 30.4–33.5) | **A** (as a ladder; the absolute level is positioning, §5) | The series' protocol (HRM/TRM 1k rows); our single-prediction number is the only strictly-comparable row we publish vs HRM 55/TRM 87.4 |
| 8 | **Verified breadth = coverage**: 88.9% @128 verified (B-M2, n=20k, integrity-audited); funnel = power-law coverage (miss ~ k^−0.9); verifier worth +50–65pp at our per-draw rate | scan n=20000 ×2 arms; D3 cell; sBG fits (unregistered, labeled model-based) | **A** as a measurement; **framing per §1** (never EqR-column) | LLMonkeys coverage laws + verifier/selection split; AlphaCode sample-filter; pass@k lineage (Codex). Ours: per-instance rates + training-time funnel evolution — no LLM work has the training axis |
| 9 | **Ignition**: funnels ignite (51→93 etc.) as smooth basin growth crosses the init measure — percolation-style transition; η-flatten × no-stream-toll ⇒ ignition (7/7); three-act displacement/Fisher arc | multi-ckpt screens ×5 arms + replication at seed 1; two-factor rule post-hoc (7/7); freethink groundings | **B** → A-able by pre-registering the d128 ignition predictions | Grokking progress-measures line (arXiv:2301.05217, 2408.08944); Ren & Liu's per-sample leaps; Achille's information-plasticity two-phase |
| 10 | **Toll dissociation**: stream toll shapes transfer/basins (ARC dividend Law 4; CSP basin cost), attention toll buys stability free; the price's sign arc across domain × scale (kill → recovery → precondition) | Law 4 seeded ×3 widths (ARC); H-43/44 arc PRE across 4 scales; graded-ladder d96 n=2/cell; the S(ε)-reader bug history disclosed + corrected | **B+** | The scale/domain-scoped mechanism story is unique to us; reviewers get the corrected-instrument disclosure up front |
| 11 | **Two-phase T6 route**: fresh-hot 50k breaks the anchored map (H-45 ext); cosine→floor continuation grows (+6.05pp funnel, paired cold +21.8k/−12.0k) to the funnel record and B-M2 | C3X n=1 (+ its scan n=20k); D1 n=1 (one-shot rule); PRE addendum | **B** (n=1 each side; C3X2 would replicate) | WSD/cooldown schedules (arXiv:2405.18392, 2508.01483; river-valley 2410.05192) — convergent schedule physics, discovered here via survival |
| 12 | **Equivariance-by-construction**: exact S9 = Sudoku's digit symmetry with zero augmentation or test-time orbit compute | [P-C] by construction + CI-1 | **A** | Ren & Liu spend ~10² orbit forwards for +18.2pp recovering exactly this — the cleanest external quantification of an inductive bias we get for free |
| 13 | **Portfolio law**: union-over-arms > best-single at every scale (+6–13pp); mechanism diversity decorrelates (Jaccard .34–.52 cross vs .74–.80 same-recipe); pair union 95.08% @128 | 22-arm panel ($0 lens, labeled exploratory); scan pair | **B** (facts, not a deployed protocol — the portfolio × verified-attempts protocol is registrable, unrun) | Ren & Liu's +9.2pp checkpoint bootstrapping = external convergence (temporal axis); ensemble-diversity literature |
| 14 | **Instrument suite + the loop** (measure → understand → improve, ×3: E3b→E10; NaN forensics→dose→record; funnel model→continuation→B-M2) | the ledger itself (append-only, pre-registration discipline, audits 286–339/0/0) | **A** | No counterpart in the series (they infer attractors; we measure them); Ren & Liu's convergence on 2 instruments (violations, fixed-point checks) supports external validity |

**Registered-but-unfinished claims a reviewer could ask for** (do not claim; list as future/limitations): S2's LoO-gap correlation in its original form; S3's NL-class predictive power (split failed → weakened form, only constructed families); S4 code-distance (exploratory, thin); H-9 flux-family clustering (never run); B-M3 (open); the ARC d96 extension (registered, unrun).

---

## §3. The adversarial pass — ranked reviewer concerns

**R1 (blocking-class). "Your headline breadth number uses an oracle verifier; EqR/TRM don't."** — Confirmed and repaired per §1: protocol-columned table, our numbers on their statistics (riders), verified breadth reframed as coverage with the D3/LLMonkeys law. *Owed: the two rider evals; strike "TRM-class" band language.*

**R2. "Single-prediction accuracy is far below the series leaders (33.5 vs 87.4/93.0 at 2.4–13× params)."** — True; the paper does not claim benchmark leadership. Positioning (§5): a science-of-reasoning-models paper whose performance spine is the *ladder* (cold 21.2→25.3→33.5 with d128 pending, at 78k→3.7M params) and whose contributions are laws + instruments + the cross-domain separation. The honest efficiency row: our vote@1 per-forward-pass at 2.11M vs EqR's 93.0 at 5.03M — we are behind; RI training (EqR's lever, our H-37) is the named closing mechanism, with our checkered RI history at width stated. *Decision: whether a d128 RI-on-FPA arm is worth a slot (risk: B1-d64 NaN history) — PI call at registration.*

**R3. "n=1–2 on many claim-bearing cells; the record cell has a seed×lr confound."** — The measurement law (n≥3 or within-run pairing) is enforced and every sub-threshold contrast is labeled; the paired-McNemar upgrades (422,786-puzzle pairing) carry the load-bearing contrasts at p≈0; D4-vs-D3 confound is named in the ledger and the *lane* claim rides n=2. Weakest spots: FN2b/CNOISE from one pair; C3X/D1 n=1 each; funnel-vs-training curves on 512-puzzle screens (CI ±2pp) — all named. *Owed: nothing mandatory; d128 ×2 seeds per the carrier rule strengthens the recipe class.*

**R4. "Nonstandard 2M-param architecture — do the laws transfer?"** — Partially externally validated (Ren & Liu convergence on the pathology + instruments; z-loss/QK-norm as the field's version of the toll; EqR's damping = our learned η). The decisive response is cheap and unrun: **run our instruments (retfm, violations, funnel/(ρ,r)) on a public HRM/TRM/EqR checkpoint** — instrument portability demonstrated on the field's own models. *Recommend registering as a post-abstract cell (one lane-day).*

**R5. "The physics language is decorative."** — Defense already built: rigor-class discipline since July (S1 anchored in classical MFMC/IC, no duality claims; S3's split FAILED its stability test and was weakened *by the pre-registered rule* — the discipline visibly bites); two freethink kills this week (mixer-dominance died at normalization) show the same. Keep every physics term paired with its measured quantity; the mass-term reading of the toll is presented as a reading, with the A*(β) check as its test.

**R6. "Instrument validity: 8-step retention, λ_J contract, the S(ε) reader bug."** — All three are disclosed with named scopes/fixes in the ledger and Instrument Map; the reader bug affected zero registered rules and the correction is its own documented lesson (instrument-readers need named tests — now practice). Reviewers respect disclosed instrument archaeology; hiding it would be the risk.

**R7. "Test set is the same generator distribution — where's OOD?"** — D2 row: on Sudoku the generalization axis is instance difficulty (1k train → 423k test including far harder ratings); rule-family OOD exists only on ARC (rg gates, Law 4) — stated as the domains' structural difference, which is the paper's point. Contamination hygiene is strong (test touched only by the evaluator; registered subsample seeds; aug group from train rows only).

**R8. "Comparator hygiene: whose numbers are these?"** — All comparator numbers are *reported artifacts* (HRM 55.0 via TRM's table — provenance ⚠ noted; EqR fetched at source this review; aug-HRM per Ren & Liu; FPRM ARC-only). None independently reproduced. The paper labels each; the ⚠ items in `Related_Work_Series.md` §6 (HRM provenance, EqR train-set size) are owed at writing time. *EqR's Sudoku training-set size remains unverified — flag prominently until read from their appendix.*

**R9. "Why β=3e-5/1e-5 knee and β_nl=1e-6? Sensitivity?"** — ARC: the β-ladder history (H-38/H-41: transfer β-inelastic over a decade; knee calibrated then held). Sudoku: knee inherited; the 1e-6 dose was forensics-derived with dose arithmetic pre-registered, and the A*(β) ~ 1/β relation (freethink Z) is the compact sensitivity story; the β/3 arm (C4) supplies one more point. Adequate if presented as designed doses, not tuned hyperparameters.

**R10. "The ARC half of the paper is a pile of negative results at 2/48."** — Reframe as designed: ARC appears as (i) the contrast domain in the D-catalog (its *instrument* results are strong and replicated), (ii) the falsification-map methodology (H-2 falsified-as-deployed → repair → measured recovery), (iii) the laws' second domain. The ARC solver story is paper 2, and paper 1 must say so in one sentence. The scale-match gap (ARC frozen at d64) is the one real exposure → the registered d96 ARC extension (~$60–100, ~Sep 3–6) is the fix and fits the calendar.

**R11. "Classical solvers do Sudoku perfectly; RRN-class nets did hard Sudoku years ago — why is any of this interesting?"** — Standard series defense (the benchmark tests few-shot-data constraint reasoning, not Sudoku): 1k training rows, exact-match on 423k far-harder instances; classical solvers don't learn; RRN-class results (⚠ verify numbers at writing: Palm et al. arXiv:1711.08028, ~96.6% on 17-given hardest, different dataset/protocol, 10⁵× more training data) are a different data regime. Our claims are about the *landscape physics* of the learned solvers, which is solver-agnostic content.

**R12. "Resumes, preemptions, TPU nondeterminism."** — The rng-resplit valid-draw class is registered and labeled per arm; the audits (286–339/0/0) + fingerprint gates + same-seed replicate noise floor (the measurement law's origin) turn this into a reproducibility *strength*. State it in the appendix.

**R13. "Post-hoc lenses (panel, funnel model, forensics) risk garden-of-forking-paths."** — The house rule already separates registered verdicts (analyzers frozen pre-data, 0-diff verified) from exploratory lenses (labeled, no rules read from them; their claims either upgraded existing PRE contrasts by pairing or spawned *new registrations*). Say this in methods; it is a genuine differentiator.

**R14. "Band thresholds (.50/.85/.95) are arbitrary."** — Internal milestones defined pre-data for sequencing decisions; never claims. With R1's language fix (no comparator-class nicknames) this is inert.

**R15. "What would falsify the framework?"** — The register answers directly: every law carries its kill (several fired and are reported as such: H-41 falsified, S3-split failed, water-filling refuted, mixer-dominance killed at normalization this week). Include the "kills that fired" table — it is the program's credibility asset.

---

## §4. The comparator table for the paper (protocol column mandatory)

| System | Params | Train rows | Statistic | Forwards/puzzle | Selection signal | Verifier | Sudoku-Extreme |
|---|---|---|---|---|---|---|---|
| HRM (per TRM table ⚠) | 27M | 1k | single prediction | 1 | — | none | 55.0 |
| TRM-MLP | 5M | 1k | single prediction | 1 | — | none | 87.4 |
| aug-HRM (Ren & Liu) | 27M | 1k (+aug) | orbit majority vote | ~10² | self-consistency | none | 96.9 |
| EqR | 5.03M | ⚠ unverified | **B=1 single draw** | 64 (D) | — | none | **93.0** |
| EqR | 5.03M | ⚠ | Top-1 by residual, B=128 | ~10³·8 | conv. residual (L=3) | none | **99.8** |
| **ours (D4)** | 2.11M | 1k | single prediction (cold) | 64 (t) | — | none | **33.5** |
| **ours (D4)** | 2.11M | 1k | single random-init draw (vote@1) | 64 | — | none | **36.0** |
| **ours (C3X)** | 2.11M | 1k | Top-1 by residual @128 | ~10³·8 | conv. residual | none | **⟵ RIDER (owed)** |
| **ours (C3X)** | 2.11M | 1k | unverified majority @128 | ~10³·8 | self-consistency | none | **⟵ RIDER (owed)** |
| **ours (C3X)** | 2.11M | 1k | **verified coverage @128** | ~10³·8 | validity filter | **free (uniqueness)** | **88.9** — *no external counterpart; ours alone* |

Reading order for the paper: rows 1–7 are the field's game and we are honest about where we sit in it; the last row is a different, named game whose value the D3 cell quantifies and whose law (coverage ~ k^−a) is part of the contribution.

---

## §5. Paper-1 restructure recommendation

1. **Contribution ordering** (replaces any parity framing): (i) the instrument suite + the measured laws (H-45/FPA, H-48/closure-free, throat/S2 spectrum, toll dissociation); (ii) the cross-domain separation catalog (D1–D11) resolving the series' own anomaly; (iii) the funnel/coverage law + the verifier-value quantification (with the §4 table); (iv) the recipe results (FPA, dose, two-phase) with the cold ladder as the performance spine and d128 as its next point; (v) limitations = the unfinished-claims list + R2's efficiency gap, stated plainly.
2. **Language rules:** every cross-system number cites its §4 column; "record" only with "program-" prefix; band names without comparator nicknames; every physics term paired with its measured quantity; lenses labeled exploratory; the kills-that-fired table included.
3. **Riders before the abstract (Sep 18):** the two §1 rider evals (~$10–15 total, one short eval session, can share a node with the d128 launch); optionally k=1024 coverage (~$15–40). The ARC d96 extension (~$60–100) closes R10's scale gap and is already PI-slotted ~Sep 3–6. The instrument-portability cell (R4) is the highest-leverage post-abstract item.
4. **d128 registration inputs from this review:** the rider stats become standing eval outputs for every d128 arm ($0 marginal); freethink pre-registrations (ignition step, two-factor rule, sBG bands, IRT cold band) attach; the RI-arm question (R2) is a named PI decision with its risk history.

## §6. Decision summary for the PI

- **Verification concern: confirmed; repair = measurement + framing, not retraction.** B-M2 stands as registered (internal, named statistic); the paper reframes it as coverage and adds our numbers on the field's three statistics (two cheap riders + vote@1 already in hand).
- **Claim inventory: 6 A-grade, 6 B-grade, 1 A−; nothing at C after the riders.** The weakest legs are all named (n=1 two-phase; post-hoc ignition rule; ARC scale-match) and each has a cheap registered fix.
- **Calendar:** riders + d128 (+2 seeds carriers) + ARC-d96 extension fit before ~Sep 8–10; drafting window Sep 5–14 holds for the ICLR abstract Sep 18 per the standing plan.
