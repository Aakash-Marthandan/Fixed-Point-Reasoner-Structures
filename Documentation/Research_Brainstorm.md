# Research & Brainstorming — QG/Information-Theory Lens on the Architecture

**Concision pass 2026-09-02 (PI-directed): the four earlier freethinks (the initial QG/IT survey with clusters A–H; Freethink 2026-08-08 attractor engineering; 2026-08-10 post-pivot clusters I–M; 2026-08-12 post-grid clusters N–S; 2026-08-13 consolidation clusters T–W) moved VERBATIM to `Research_Brainstorm_Archive_2026-07_to_2026-08-13.md` (pre-move state = git d25b728). This file keeps the newest freethink (2026-08-31, clusters X–AF) and the staged inputs it consumed. Lettering of clusters continues across both files.**

## Archived sections (open the archive for any of these)

- Research & Brainstorming — QG/Information-Theory Lens on the Architecture
- 1. Field scan (what the outside world measured, distilled to mechanisms)
- 2. Physics-lens idea clusters
- 3. Distilled shortlist (merged with the registered pipeline, cheap→heavy)
- 4. Sources
- Freethink 2026-08-08 — after the E1/E3 verdicts: attractor engineering
- 0. New same-day data
- 1. The convergent diagnosis (ours ⊕ the field's)
- 2. Mechanism shortlist for basin construction (from the survey, ranked)
- 3. What is publishable regardless of solve-rate
- Freethink 2026-08-10 — post-pivot: the landscape is real; now govern it
- New clusters (I–M)
- Updated shortlist (merges with the standing queue)
- Freethink 2026-08-12 — post-grid: the landscape is a code; now BOUND it
- 0. The organizing observation
- N. The packing frontier: are priced landscapes near-optimal codes? **[T1 — zero new compute]**
- O. Spectral shape universality: an RG scaling function **[T1 — zero new compute + one big-d check]**
- P. The retention metric: anisotropy of basin-destruction **[T2 — the principled TTT]**
- Q. Barrier spectroscopy: hop thermodynamics **[T1.5 — one sampling battery]**
- R. Boundary sufficiency at scale: the holographic dictionary test **[T2 — rides pretrain-13]**
- S. The all-eq conversion attempt: self-decoding at record radius **[T1.5 — the convert-phase opener]**
- Literature anchors (tonight)
- Priority for the scaling sessions
- Freethink 2026-08-13 — consolidation-pass design round (clusters T-W)
- T. Coarsening thermodynamics: is consolidation a ripening flow? **[$0]**
- U. Water-filling optimality of the knee profile **[$0 — theory-grade if it fits]**
- V. Fisher-anchored TTT — P2 in buildable form **[convert-phase arm]**
- W. Representational basin creation for the vacancy floor **[the family-structure consequence]**
- Scale-regression flag (rides wave-2 analysis)
- Lit anchors (this pass)

# Freethink 2026-08-31 — the B-M2 synthesis: percolation funnels, tolls as mass, and the three-act code

*The long-owed synthesis (PI-sequenced after the rung-2b close + rider-scan
verdict). Inputs: the COMPLETE program corpus (full ledger ingestion July 16 →
today; the 2b verdict + lens suite + today's B-M2 scan), the staged-inputs
section below (now CONSUMED by this entry), a literature sweep (8 searches +
the PI-supplied Ren & Liu paper), and FOUR grounding computations run before
writing (scratch: `<scratchpad>/freethink/ground{1,2,3}.py`; all $0, disk-only,
UNREGISTERED — numbers quoted from them are labeled model-based/descriptive).
Nothing here is registered until it moves to the ledger with a named test.
Lettering continues from W.*

## 0. What the grounding pass measured before any idea was written

1. **Block parameter census (d96):** mixer 70.0% of params, e_cb 9.7%, w 5.4%,
   rule_query 3.5%, pool 3.1%, attn 2.8%, film 0.6%, codebook 0.6%. The mixers
   are weight-tied ACROSS scales (enc/dec × l1/l2 only — scale-specificity
   lives in film's 13k params).
2. **Displacement DENSITY per window** (share of ‖Δθ‖² ÷ share of params):
   mixers run 1.0–1.2× in every window of every arm — the 72–85% displacement
   "dominance" is parameter-proportional. The density signal is elsewhere:
   attn 1.3–1.6× EARLY (highest on D1, the arm that died), e_cb 1.2–1.5×
   MID, w + rule_query 1.2–1.9× LATE; B3's condensation window 35–40k spikes
   **rule_query to 4.8×**; C4's late de-commit windows show rule_query
   1.4–1.9× exactly as rule_H rises 0.043→0.133.
3. **Per-puzzle ignition grading** (screen records, strat-512): D3 25k→40k
   newly-hit median rating 25.5 vs already-hit 3.0 (the wave sweeps outward in
   difficulty) but the front is DIFFUSE in rating (still-miss median 22!), and
   D4 shows watershed CHURN (40 puzzles lost 25k→40k, 13 lost 40k→50k) — the
   boundary is a probabilistic shell, not a sharp set.
4. **Shifted-Beta-geometric funnel fits** (scan records, n=20000/arm; fit
   draws ≤64, validate on the held-out 65–128 window): per-octile shape
   a ≈ 0.9–1.0 OCTILE-INVARIANT on C3X with only b scaling 21→33 (difficulty
   = pure rate rescaling); unreachable atom π₀ ≈ 0 on every octile; the fits
   UNDER-predict the held-out window by 2–4pp everywhere (the true rate tail
   is heavier than Beta) — so extrapolations are lower bounds. **C3X model
   forecast: vote@256 ≈ 92.3, @512 ≈ 95.6, @1024 ≈ 97.5, @4096 ≈ 99.2; D4 one
   generation behind (93.3 @1024).** Model err at the 128 checkpoint: −2.1pp
   (C3X) / −2.8pp (D4), i.e. reality beats the model at the tail.
5. **η-flatten vs ignition across 12 arms:** every igniting arm's η flattens
   before/during its ignition window (D3 flat@30k inside 25–40k; C3 flat@14k
   inside 10–15k) — but C2 flattens @32k and NEVER ignites. λ_J-settle does
   NOT order events (D4 ignites while λ_J>1.05 until 44k). The clean rule in
   the corpus: **η-flat × no-stream-toll ⇒ ignition (7/7); stream-tolled arms
   flatten into condensation instead** (C2, B3; C4 = the β/3 boundary case).

## X. Ignition is watershed percolation against the init measure **[the RG/phase-transition frame; partially grounded]**

The funnel's ρ-expansion now has enough structure to name the transition:
the trained map's solution basin B(θ) grows smoothly under training (no
weight-space burst — displacement is schedule-ordinary; Fisher consolidation
monotone), while the multi-init draw measure is fixed. ρ = init-measure(B).
A smoothly moving boundary crossing a concentrated measure produces a SHARP
observable — ignition is a percolation-style transition of REACHABILITY, not
a training-dynamics event. The grounding adds three facts the frame must and
does carry: (i) the front sweeps outward in difficulty (new-hit median rating
25.5 vs 3.0) but is diffuse in RATING — the sharp coordinate should be the
model's own difficulty axis (→ cluster AB); (ii) the boundary is a
probabilistic SHELL (D4's 40-lost churn = puzzles with p ≈ 1/k flickering in
and out of a finite-draw census); (iii) C3X grows +13pp of funnel at
0.023/window displacement with Fisher rotation already ≈0.9 — **basin-boundary
micro-motion without reorganization**, the strongest evidence that ρ-growth is
boundary geometry, not code rebuilding. Literature: this is the grokking
phenomenology transplanted to reachability — internal progress measures
(here: η-flatten, Fisher-J, the three-act density arc) move continuously
through a discontinuous-looking capability onset (Nanda et al.
arXiv:2301.05217; grokking-as-phase-transition arXiv:2408.08944); Ren & Liu's
per-sample "long plateau → sudden orthogonal leap" (arXiv:2601.10679) is the
same transition seen along a single inference trajectory instead of along
training. **Predictions worth pre-registering at d128:** (1) ignition step
predicted BEFORE screens land, from η-flatten + the act-2→act-3 density
handoff (both computable from monitors/grids); (2) the two-factor rule holds:
dosed/free arms that flatten ignite, any priced-stream arm flattens without
igniting; (3) per-puzzle ignition order is sharp in model-difficulty θ
(cluster AB) and diffuse in tdoku rating. Kill: an arm that flattens, carries
no stream toll, and still fails to ignite by end-of-schedule.

## Y. The funnel is a power-law coverage law; B-M3 is plausibly within attempts-reach of the EXISTING C3X checkpoint **[grounded; the sharpest new consequence]**

The sBG fits say the per-puzzle hit-rate distribution has (i) no detectable
unreachable atom (π₀≈0 at k=128 resolution), (ii) a near-invariant shape
a≈0.9 with difficulty entering only as rate scale b, and (iii) a tail heavier
than Beta (systematic −2–4pp under-prediction of the held-out window). A
Beta(a,b) rate mixture gives miss@k ~ k^(−a) at large k — a POWER LAW with
exponent ≈0.9 — which is exactly the empirical form of LLM repeated-sampling
coverage ("Large Language Monkeys," arXiv:2407.21787: log-linear coverage
over four orders of k, exponentiated-power-law fits). Our verified-funnel is
their coverage curve with the verifier free by construction; their second
finding (majority voting / reward-model selection PLATEAUS after ~10² samples
without a verifier) is our D3 demo cell (+50–65pp verifier worth; unverified
majority flat in k) measured at LLM scale. Two consequences: **(1) the (ρ,r)
instrument acquires an external law-family** — report the coverage exponent a
per arm/octile alongside (ρ,r); (2) quantitatively, C3X's own curve forecasts
B-M3 (95%) at k ≈ 384–512 draws and ≈97.5% at k=1024, with the model biased
LOW — so the attempts-route to B-M3 is live on the existing checkpoint,**
distinct from the training route (a d128 C3X-class). The lens-B "plateau
~82.5%" forecast for C3@20k came from the 2-parameter model whose ρ is a
lower bound; the sBG family says "no plateau below ~99" for C3X — the model
CLASS is now the question. **Registrable adjudication (PI menu): a one-shot
k=1024 scan on C3X's ckpt, subsample 5k (CI ±0.6pp), ≈$15–40 — simultaneously
(a) the B-M3 attempts-adjudication and (b) an 8× out-of-sample test of the
coverage-law class (power-law-climb vs saturation), the cleanest model-test
the funnel program has ever had.** Kill for the power-law reading: measured
vote@1024 lands ≤ the Beta forecast (tail is NOT heavy; ceilings real).

## Z. The attention toll is a mass term; the cliffs are massless-mode runaways **[IT/field-theory frame; semi-grounded]**

The toll is literally a mass: KL(N(μ,σ)‖N(0,1)) = ½(μ² + σ² − 1 − ln σ²) is a
quadratic well on every attention message — β_nl multiplies a mass term for
the nonlocal field. β_nl = 0 leaves a MASSLESS direction: message magnitudes
can grow at zero loss cost (S2 deadweight), and T-fold composition amplifies
excursions until the state explodes — the measured cliff (A→1e14 in ≤50
steps, no y-side precursor, λ_J useless as alarm). The mass explains the
phenomenology: (i) WHY closure is free — the channel carried no load-bearing
information, so gapping it costs nothing (B-M2 through the dosed/closed
regime); (ii) WHY depth × width × hot-lr compose — more composition, more
modes, bigger kicks (D1: the steepest η-surge +0.273 AND the highest early
attn displacement density 1.6× AND death — the hot schedule kicks the
massless mode hardest); (iii) WHY there is no precursor — a marginal mode's
runaway is triggered, not accumulated. Rough equilibrium check from disk:
β·A_late ≈ 1e-6–1e-5 across the dosed (1e-6 → 1.4–5.5 nats) and knee-priced
(1e-5/3.3e-6 → 0.3–1.3) arms — an A*(β) ~ 1/β-family consistent with a soft
well; **[T1, $0]: fit A*(β) properly across all priced arms; it predicts the
β_nl=1e-8 barrier-form variant sits at ~10²–10³ nats and stays stable — the
d128 contingency design derived rather than guessed.** Literature: this is
the family the LLM world converged on as tricks — attention-entropy collapse
and σReparam (arXiv:2303.06296), QK-layernorm/z-loss and logit-growth
instabilities (small-scale proxies, arXiv:2309.14322); mechanism-driven
instability monitors (arXiv:2606.28116) watch logit magnitudes — our
case-control found NO precursor at our observables, and the structural fix
(gap the mode) removes the need for monitoring entirely. Our addition to that
literature: the LEDGER — we measure the channel's information and show the
gapped channel carried none; the trick is explained by the theory it
instantiates (S1/S2), and one number (seven orders closed, zero capability
cost) is the whole argument.

## AA. Learning is param-proportional on the seams — the real signal is a three-act displacement drama **[grounded: one honest kill + one upgrade]**

**The kill:** the staged "mixer dominance as holography" reading dies at
normalization — mixers hold 70.0% of parameters and their displacement share
is 72–85% ≈ 1.0–1.2× density in EVERY window of EVERY arm. Learning does not
concentrate on the cuts beyond capacity share; the ARCHITECTURE concentrated
capacity on the cuts (a design statement, decided in July, not a training
discovery), and the mixers are scale-SHARED, so "which-scale mixers match the
flux profile" is ill-posed at parameter level (scale-specificity lives in
film, 0.6%). The correct holographic statement stays architectural: one seam
operator serves every scale, (s,t)-modulated — the code is scale-shared by
construction. **The upgrade:** per-param DENSITY has clean temporal structure
— a three-act arc: **act 1 attention-led** (1.3–1.6× early; the A-flux peak;
the doomed arm highest), **act 2 coding-led** (e_cb 1.2–1.5× mid), **act 3
readout-led** (w + rule_query 1.2–1.9× late) — and it corroborates the
Adam-v Fisher eq→coding→readout migration from an entirely independent
instrument. Events read in density: **B3's condensation window carries a
rule_query spike at 4.8×** (H-47's diversity condensation is a rule-PATH
reorganization, not diffuse forgetting); C4's late funnel-opening carries
rule_query 1.4–1.9× while rule_H de-commits. Anchor: Achille et al.'s
information-plasticity two-phase (Fisher rises then consolidates,
arXiv:1711.08856) — ours is the block-resolved version with a third act and
event markers. **[T1, $0]: formalize the density instrument (tool + artifact)
and add the act-2→act-3 handoff step to the d128 monitor watch; the staged
"mixer-targeted lr" lever is DOWNGRADED** (no per-param evidence mixers are
special); its replacement candidate — act-aware lr on rule_query/w late — is
noted, unregistered, and needs a mechanism first.

## AB. Difficulty is a one-parameter rate family; the model's own θ beats rating **[grounded + buildable]**

Three measurements now say tdoku rating is the wrong coordinate: it explains
~35% of model-difficulty variance (puzzle panel); the ignition front is
diffuse in rating (grounding 3); and the sBG fits collapse difficulty to a
single rate scale b with shape invariant (grounding 4). The natural upgrade
is psychometric: fit a Rasch/IRT model P(solve_ai) = σ(b_a − θ_i) on the
22-arm × 422,786 paired panel — puzzle ability-scale θ from the models
themselves. **Predictions:** (1) the percolation front is SHARP in θ where it
is diffuse in rating; (2) sBG b ∝ exp(c·θ) (the one-parameter family in its
natural coordinate); (3) the 49.6% cold hard core is the high-θ tail, and its
funnels are rate-suppressed, not absent (π₀≈0 — the scan pair's 4.92%
unsolved-at-128 should keep yielding at higher k); (4) the arm-ability ladder
b_a across d16→d64→d96 gives a d128 cold FORECAST to pre-register (the cold
ladder 21.2→25.3→33.5 as an ability trend, not a curve-fit). **[T1, $0,
lens-E-class]:** one analyzer over banked records; IRT-for-benchmarks is
standard practice in the eval literature, and θ becomes the difficulty axis
of record for the paper's spectroscopy figures.

## AC. Draw-diversity has two routes: state-space and code-space **[data-scoped; watch]**

The staged H[q]↔diversity hypothesis survives only in scoped form. The dosed
arms hold rule_H ≈ 0 (fully committed codebooks) with the WIDEST funnels
measured (89–93) — de-commitment is NOT necessary for draw diversity; the
init-measure over states supplies it (route 1, the carrier route). C4 shows
the second route: under stream-budget pressure the codebook PARTIALLY
de-commits late (H[q] 0→0.13, rule_query density rising) exactly as its
funnel opens and de-flattens — diversity bought in CODE space when state
space is priced (route 2). B4's collapse-decommitment is the pathological
third face. SSB reading, honestly scoped: partial symmetry restoration is A
diversity mechanism, not THE mechanism. Consequence: the H[q]-floor
regularizer stays PARKED (the carrier class doesn't need it); rule_H stays a
free logged observable at d128; if a d128 arm's funnel opens WITH rule_H
motion, route 2 has appeared at width and earns its own cell.

## AD. Verification is the scaling axis; equivariance is inference-compute you don't have to spend **[Ren & Liu × LLMonkeys × our D3]**

The PI-supplied paper (Ren & Liu, arXiv:2601.10679: mechanistic HRM analysis)
independently reproduces the program's core pathology findings and completes
a triangle with the coverage literature: (1) their "fixed-point violation"
(correct solutions unstable; latents corrupt found answers) is our E3b
truth-erasure / H-45 contractivity collapse, measured here three weeks
earlier and FIXED (FPA: retfm 1.00 at three scales) — and the cross-paper
synthesis sharpens the attribution: they blame HRM's one-step gradient, but
our full-BPTT arms had the same disease until anchor rows — **the missing
ingredient is the basin OBJECTIVE, not the gradient estimator**; notably they
never repair retention — they route around it with test-time voting. (2)
Their conflict-count energy E(ŷ) = Σ ReLU(count−1) is exactly our violations
instrument; their "spurious fixed points" on our substrate are sharpened by
D5's valid_wrong = 0.0000 × 2.1M — stuck states are INVALID partial
propagations, never wrong-valid attractors. (3) Their four per-sample modes
(trivial/non-trivial success/failure) map onto our (cold-solve, draw-hit,
rate-suppressed, stuck) taxonomy with the funnel as the quantitative form.
(4) Their test-time recipe — 55.0→96.9% on Sudoku-Extreme at 27M params via
digit-relabeling votes (+18.2pp), checkpoint bootstrapping (+9.2pp), and
augmentation (+4.9pp) — reads through our instruments as: **the biggest
lever (+18.2) is the S9 orbit, which our architecture internalizes EXACTLY
by weight-sharing** (their 9× relabeled forward passes recover what our
equivariance gives in one — inference compute spent recovering a missing
inductive bias); their checkpoint bootstrapping is the temporal-portfolio
idea (cluster F, 08-06) validated externally (+9.2pp from HIGHLY correlated
members — consistent with our portfolio law's diversity-decorrelates
finding); and their majority voting works where LLMonkeys says it plateaus
because orbit-views are semantically EQUIVALENT (self-consistency over a
group action), a different object from selecting among heterogeneous draws.
Comparator table updates: HRM 55 → augmented-HRM 96.9 (27M, ~10² forwards)
→ TRM 87.4 → EqR 99.8 (5.03M) → ours 88.9 verified@128 (2.11M).
**Levers for us, cheap and registrable:** (i) POSITION-orbit voting at
inference (band/row/stack swaps — trained-in as augmentation, never voted;
unlike ARC's eval-4 failure there is no TTT view-specialization on Sudoku,
so orbit votes should compose with draws as a decorrelated diversity source;
eval-only rider, ~$5); (ii) checkpoint-ensemble draw-splitting (spend the
same k across 2–3 banked ckpts vs one — $0 extra at eval; the banked 5k
grids are the ensemble); both are (ρ,r)-instrumentable: does orbit/temporal
diversity raise ρ at fixed k where draw-diversity alone cannot?

## AE. The two-phase route is the river valley; the continuation moves along the floor **[data + literature]**

C3X's grounding signature — funnel +13pp with displacement flat at
0.023/window and Fisher already consolidated — is the river-valley picture
of WSD schedules (arXiv:2410.05192): the decayed phase travels ALONG the
valley floor, refining position (watershed micro-motion, ρ-led gains) rather
than rebuilding the code. The T6 scaling recipe the program measured into
existence (cosine 20k → floor-lr continuation; fresh-hot 50k kills even the
anchored map) is the WSD/cooldown family (arXiv:2405.18392; cooldown
bias-variance arXiv:2508.01483) discovered independently via SURVIVAL
constraints — worth stating in the paper as convergent schedule physics.
Open, cheap, and PI-menu-able: **C3X2 (+30k more floor, ~$25)** measures the
continuation's diminishing-returns curve against the sBG ceiling — the
training-route mirror of the k=1024 attempts-cell; the cooldown literature's
bias-variance trade suggests a deeper-floor variant (lr→1e-5) as the
alternative arm if C3X2 reads flat.

## AF. The unified S2 statement for paper 1 **[synthesis; the physics section's spine]**

Today's scan completed an arc the ledger has been building since July 27
(the C14 smoke: A→9.3e7 free vs 5.5e3 priced — the first sighting): **free
(unpriced) information flux is never load-bearing on these tasks, and its
cost escalates with scale: deadweight at small scale (S2 classic, 400×
compression at equal accuracy), basin-narrowing through the stream channel
on CSP (graded-ladder inversion of ARC Law 3/4), and LETHAL through the
attention channel under depth × width × hot-lr (H-48; the mass-term reading
of Z).** Priced flux, conversely, IS the code: throat = task-set constant
(785→432 nats across 27× params on ARC; instance-conditional on Sudoku —
the ledger reads required computation), profiles = task-geometry
fingerprints, and closure of the free channel is not merely safe but the
enabling condition of the deep lane and the record cold. One mechanism
(pricing), three measured regimes, two domains, plus a literature bridge in
each direction (z-loss/QK-norm as the trick our theory explains; MDL/VIB as
the theory our instrument operationalizes). This is the physics section's
one-paragraph thesis, now fully evidence-backed.

## Priority shortlist (maps to the PI's next decisions; nothing registered here)

1. **d128 registration inputs (pre-data predictions this freethink offers):**
   ignition-step from η-flatten + act-handoff; the two-factor ignition rule;
   sBG vote@k bands per arm; cold band from the IRT ability ladder; rule_H
   route-2 watch; A*(β) check on the dose.
2. **PI menu of cheap cells (one night, ~$45–90 total):** (i) C3X k=1024
   subsample scan — B-M3 attempts-adjudication + coverage-law model test
   (~$15–40); (ii) C3X2 +30k floor continuation (~$25); (iii) position-orbit
   voting rider (~$5, eval-only); (iv) checkpoint-ensemble draw-split rider
   (~$5). Each is one-shot, banked-partial, harness-trivial.
3. **$0 lens-E completion (this week, disk):** IRT/θ fit (AB) + η(t)
   universality + the density instrument formalized (AA) + A*(β) fit (Z) +
   per-puzzle ignition-in-θ regrade (X).
4. **Lens F / ARC extension (~Sep 3–6, pre-paper):** d96 ARC rung with the
   Sudoku-era instrument back-port — scale-matches D6/D8 and tests whether
   ARC ignition exists (the funnels there were never multi-ckpt-screened).
5. **Paper 1:** AF as the physics spine; AD's comparator + related-work
   updates (Ren & Liu, LLMonkeys, WSD, σReparam/z-loss, Achille); the
   two-sided D4 figure stays the flagship.

## Sources (this pass)
- Ren & Liu, mechanistic HRM analysis (PI-supplied): [arXiv:2601.10679](https://arxiv.org/abs/2601.10679)
- Large Language Monkeys, repeated-sampling coverage laws: [arXiv:2407.21787](https://arxiv.org/abs/2407.21787)
- Progress measures for grokking: [arXiv:2301.05217](https://arxiv.org/abs/2301.05217); grokking as emergent phase transition: [arXiv:2408.08944](https://arxiv.org/abs/2408.08944)
- Attention entropy collapse / σReparam: [arXiv:2303.06296](https://arxiv.org/abs/2303.06296); small-scale proxies for training instabilities (QK-norm, z-loss): [arXiv:2309.14322](https://arxiv.org/abs/2309.14322); instability monitors: [arXiv:2606.28116](https://arxiv.org/abs/2606.28116)
- Critical learning periods / information plasticity: [arXiv:1711.08856](https://arxiv.org/abs/1711.08856)
- WSD/cooldown schedules: [arXiv:2405.18392](https://arxiv.org/abs/2405.18392), [arXiv:2508.01483](https://arxiv.org/abs/2508.01483); river-valley landscape: [arXiv:2410.05192](https://arxiv.org/abs/2410.05192)
- Edge of stability (lr × sharpness): [arXiv:2103.00065](https://arxiv.org/abs/2103.00065)
- sBG lineage: Fader & Hardie, "How to Project Customer Retention" (2007) — the shifted-Beta-geometric family
- Verified large-scale sampling precedent: AlphaCode [arXiv:2203.07814](https://arxiv.org/abs/2203.07814)

---

# STAGED INPUTS for the next freethink (2026-08-29 session close — NOT a freethink;
# CONSUMED by the 2026-08-31 freethink above — every numbered item is addressed
# there (X/Y/Z/AA/AB/AC/AE), with two honest kills recorded (mixer-dominance
# normalization; λ_J-settle ordering). Kept verbatim for the record.
# Lettering continues from W. PI lenses requested: quantum geometry, information
# theory, RG, holography; literature research; big-picture outlook.)

## The day's measured findings (each committed; artifacts in runs/analysis/*_20260829.txt)
1. MIXER DOMINANCE: the seam mixers carry 72-85% of ALL parameter displacement
   in every 5k window of every arm at every scale (e_cb second, 5-14%). PI flag:
   this reads as a HOLOGRAPHY statement — the boundary-acting disentangler
   descendants are where the code physically forms. Freethink question: is
   learning-lives-on-the-cuts a derivable consequence of the priced-cut
   architecture, and does it predict WHICH mixers (scale profile of mixer
   displacement) match the flux profile?
2. IGNITION = REORGANIZATION, NOT BURST: funnel rho-expansion (C3 10k->15k)
   shows schedule-ordinary displacement + monotone Fisher consolidation; the
   signature is the eq->coding Fisher-share migration. SEQUENCE: eta-surge +
   lambda-settle (5-10k, +0.25) PRECEDES ignition; eta flattens through it.
   PI wants the eta<->ignition relationship understood. Candidate frames:
   eta as inverse-temperature/quench-rate of the flow; settle = the ordered
   phase forming; ignition = basin-of-attraction percolation after ordering.
   Readiness-signal test staged on C3X/D1-redux screens.
3. rule_H <-> DIVERSITY (new observable): the codebook DE-commits (H[q] 0->.13)
   exactly while C4's beta/3 funnel opens late; distinct from B4's collapse-
   decommitment. On a ONE-rule domain. Speculative lever noted (unregistered):
   a mild H[q] floor as a diversity-preserving term. SSB reading: draw-
   diversity requires partial symmetry restoration?
4. (rho, r) FUNNEL PLANE (lens B, validated 7/7 out-of-sample): reachable-
   fraction x per-draw-rate; deep-narrow = tiny-rho/high-r, wide = high-rho/
   low-r; C3-d96 raises both (coverage-led); mid-training = rate-limited,
   final = coverage-limited; C3@20k plateaus ~82.5% => B-M2 via TRAINING (rho)
   not attempts (k) — the strategy conclusion the PI endorsed. Next refinement:
   Beta-geometric (rate heterogeneity); EqR's coverage-x-refinement conjecture
   is exactly this plane — we measure what they conjecture.
5. TOLL NARROWS BASINS, growing with scale (corrected-reader discovery):
   priced-vs-anchored S(eps) gap d16 .80-vs-.96 -> d96 .18-vs-.69-.79 @ .6.
   DOMAIN INVERSION of ARC Law 3/4 (pricing widened transfer basins there).
   Coding frame: the ~1.2k-nat code lacks redundancy for basin width; ARC's
   many-rule codebook spends redundancy differently. Also: collapse signature
   reads in basin space; leak-count = per-row map-class fingerprint; the
   graded first-failure distribution = radius-vs-scale curve from disk.
6. COLD HARD CORE 61.4% (17 arms x 422,786, paired panel) + PORTFOLIO LAW on
   Sudoku (+6-13pp union-over-best; mechanism diversity decorrelates, J .34-.52
   cross-mechanism) + tdoku rating = only ~35% of model difficulty (solve
   multiplicity the better axis). Registrable lever: model-portfolio x verified
   attempts. Paired-McNemar upgrades landed program-wide (BREADTH-SCALES
   p<1e-300; B2>B1 cold p~7e-161; priced-cold-rises-with-width p~2e-228).
7. THE CLIFFS ARE UNHERALDED (NaN case-control, 3 deaths vs 1,052 survivor
   windows): no precursor at any logged resolution; C1's calmest-ever window
   preceded death by 100 steps; lam_J is not an alarm (survivors run 0.4-0.5
   frac>2; two deaths launched from ~0.95). H-48: free-attention-flux x
   effective-lr x width; the 1e-6 dose = ~5 orders flux reduction, both dosed
   arms clean through all historical death territory (live tonight). Prevention
   is STRUCTURAL — a d128 design constant. D1's death sharpened the law:
   free-T6 dies on the 50k cosine where the 20k cosine trains clean (lr-phase,
   not depth). Freethink angle: the toll as a soft Lyapunov barrier in the
   information metric (KL restoring force ~ beta*A at excursions); relation to
   NI's accidental renorm; the z-channel as the free-vector sector where
   FPRM-class stabilizers legitimately apply (coupled-scaling legal on z).
8. FISHER ROTATION = CONSOLIDATION CLOCK (monotone .5->.93; priced consolidate
   furthest); Fisher block shares: stable deep arms migrate eq->readout,
   priced arms live in code blocks, collapsing arms never consolidate.

## Discussed-but-unregistered ideas parked for the freethink/next registrations
- Stabilizer escalation ladder (dose -> z-dials -> z-clamp/renorm -> message
  norms -> full FPRM kit w/ twin-arm bridge); beta_nl 1e-8 barrier-form variant.
- H[q]-floor diversity regularizer (3 above). Mixer-targeted lr/capacity as
  the next recipe axis (1 above). Watershed interpolation probe (demoted to
  optional top-tail; graded reader restored the ladder).
- B2 p4 records-vs-summary -1.17pp reconciliation (named investigation, 2b analysis).
- D11 (overtraining costs by domain: ARC=transfer erosion vs Sudoku=stability+
  diversity) — Part 2d of the Instrument Map holds the catalog updates.

## Remaining lens queue
- LENS E (NOT yet run as its own pass): the trajectory CORPUS — eta(t)
  universality across ~60 arms (is the progress clock one curve? quantify),
  fp_drift corpus (H-46 wander directly), rule_H corpus beyond the windows
  read, loss-curve shapes vs mechanism class. Cheap, feeds the freethink.
- LENS F: ARC graded-ladder + leak fingerprints + Fisher back-port on banked
  ARC d48/d64 grids — the ARC-extension prelude (planned ~Sep 3-6 window).
