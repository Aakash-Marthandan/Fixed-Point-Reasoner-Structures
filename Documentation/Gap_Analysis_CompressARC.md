# Why QHRRN-2 is not solving ARC: a roots-first failure analysis

**Status: PI-directed post-mortem (2026-08-08). Method: documentation review →
ledger audit → code verification → test evidence, in that order — the
question is why WE fail, on our own theory's terms. CompressARC appears only
as a calibration benchmark (§5); nothing here is a proposal to copy it.
Every claim is labeled measured (file/ledger-linked), code-verified
(file:line), or hypothesis. Companion ledger entry: 2026-08-08 post-mortem.**

---

## 0. Summary of findings

1. **The theory's two load-bearing solve-time mechanisms have never been
   exercised.** [H-2] (solved task = fixed point of the recursion map — the
   "reasoning = RG flow to a stable fixed point" claim) and [H-6] (rule
   selection = symmetry breaking under a τ-annealed discrete code, with H[q]
   collapse as the order parameter) were registered with named tests on
   2026-07-19/20 and neither test has ever been run. Code-verified: τ = 1.0
   at every pretraining step and every TTT fit and every prediction ever
   made (tools/pretrain.py:55, tools/eval_pop.py, train.py defaults);
   recursion depth T = 4 fixed (config.py); β = 0 at every TTT fit.
   **The architecture we measured all season is the scaffolding of the
   theory with its phase transition and its fixed-point semantics switched
   off.** The PI's "grokking-type lock" and "inference-time recursion"
   instincts are, respectively, H-6 and H-2 — the roots asking to be tested.
2. **The rule is re-inferred per input, never committed per task**
   (code-verified, model.py:222-237): rule_q = softmax over the codebook of
   the *current input's* pooled encoding (+ the 64-float e_t bias). At query
   time the rule is re-derived from the query's own encoding. Nothing
   applies "the rule that fit the supports" to the query beyond e_t's soft
   bias. This one mechanism explains the season's signature failure — 
   support-side (LoO) exactness with query-side incoherence — better than
   any lever we tested.
3. **The season's falsification record indicts commitment, transport, and
   convergence — not expressivity, corpus, or width** (§3). Oracle-member
   analysis (2026-08-08): even perfect selection over everything our fits
   generate solves ~1/48 curated-hard tasks. The failure is in what the
   solve-time procedure computes, not in how we pick among its outputs.
4. **Scale verdict** (§4): every scale axis we tested (width d16→d32, corpus
   ×5.6, boundary width, budget to 2000 steps) is measured-flat. The axes
   the theory itself emphasizes — recursion depth, per-task adapted
   capacity ([H-7], deferred since July), fit-time annealing length — were
   never scaled. "More compute" is useful only on the untested axes, and
   they are cheap.
5. **The program** (§6): six experiments, all of them ledger hypotheses from
   the roots finally getting their named tests, ordered so the first two
   days are pure instruments that adjudicate the PI's phase-transition
   hypothesis before any architecture change.

---

## 1. What the roots actually specify (documentation review)

From the four source documents as cataloged in ledger §1 (claims quoted from
the ledger's provenance table, which is the audited reading of the PDFs):

- *Renormalization of Thought*: reasoning is an RG flow UV→IR **to a stable
  fixed point** — "applying the transformation again yields an invariant
  state" → sharpened at registration into **[H-2]: solved task = fixed
  point of the iterate map Y_{t+1} = F(Y_t)**, with the named test
  "convergence of recursion iterates on solved vs unsolved tasks."
- Rule selection is **SSB**: "melting → cooling → phase transition;
  translation symmetry wins over rotational symmetry" → **[H-6]: τ-annealed
  codebook; H[q] collapse correlates with correctness; ablation
  discrete-vs-continuous rule code.**
- TTT itself is **annealing on a free energy** ([H-5], Langevin form) — the
  solve procedure is supposed to be a cooling schedule, not a fixed-
  temperature fit.
- The original holographic loss is a **capacity penalty** (bond dimension ×
  ln χ) — structural rate control while solving; [S-1] records our
  deliberate switch to usage pricing (the flux ledger), which is the right
  MDL reading — but usage pricing at β=0 is no rate control at all.
- **[H-7]** (adaptive capacity: "if error is high, increase χ") — per-task
  capacity escalation, explicitly deferred to Phase 3 and never reached.

The theory's solve-time story, assembled: *cool the rule distribution until
one symmetry wins (H-6), under rate pressure (capacity/flux), and iterate
the coarse-grain/fine-grain map until the answer is invariant (H-2),
escalating capacity if the task resists (H-7).* That is a phase-transition-
plus-attractor solver — precisely the "lock better answers with a reasoning
jump" the PI describes.

## 2. What the implementation does at solve time (code audit)

Verified against source this session:

| roots mechanism | implementation state at solve time |
|---|---|
| τ-anneal / SSB commitment (H-6) | **Never fires.** τ=1.0 at all pretraining (pretrain.py:55 default, run configs), all TTT fits, all predictions. The codebook is a permanent soft mixture; H[q] is logged in aux (objective.py:55) but no fit trajectory has ever recorded it. |
| Fixed-point recursion (H-2) | **Never tested.** T=4 constant (config.py); recursion feedback is the argmax canvas (model.py iterate); no convergence check, no iterate-until-invariant mode, no measurement of whether solved answers are stable under F. |
| Rate pressure while solving (S-1/H-4) | **Off.** β_flux = β_flux_nl = 0 at every TTT fit (measured worse for CE-fitting at 1e-5 — the tell that the objective, not the pricing, was misaligned); pricing used only as a post-hoc instrument. |
| Annealed solve dynamics (H-5) | AdamW at fixed lr; SGLD relegated to an unrun ablation row. Selection compensates *backward* (earliest-LoO-exact) because later steps memorize — the anti-grokking signature: the generalizing solution is a waypoint the trajectory passes through, not an attractor it settles into (eval-2's measured selection pathology). |
| Rule as the task's committed code | **Per-input re-inference** (model.py:222-237): h_ir is pooled from the current input's fields; rule_q = softmax((h_ir·W_m)·Eᵀ/τ). Query forwards re-derive the rule from the query's own encoding; the only cross-input carriers are the 64-float e_t bias and codebook geometry. |
| Adaptive capacity (H-7) | Deferred since 2026-07-19; per-task adapted DOF has been ≤64 floats (e_t) in every eval. |

## 3. The measured failure signature, mapped to mechanisms

Season evidence (all from results files / ledger; assembled 2026-08-08):

| observation (measured) | indicts |
|---|---|
| 104/1536 members LoO-exact; ≥1 exact member on 19/48 val-hard tasks; **0 task solves** — supports compress, queries don't follow | **Transport** (per-input rule re-inference): held-out supports are distribution-close to training supports, so re-inference lands in the same basin; queries are not, so it doesn't. |
| Earliest-exact selection beats final params; MDL tie-break "walks past the generalizing solution into support-memorization" (eval-2/3, measured) | **No attractor** (H-5/H-6 off): nothing makes the generalizing solution stable; we fish it out of the trajectory in hindsight. |
| Failure taxonomy: 72% content-level — within-object inconsistency, deformed/deleted objects (eval-3r renders); near-miss mass with 0.91 cellwise agreement on wrong answers | **No convergence/commitment**: outputs are superposition-like mixtures — colors assigned inconsistently *within one bar* is exactly what an uncommitted soft rule produces. C17 raised object-coherence without raising solves: binding was the symptom, commitment the cause (measured 2026-08-04). |
| Candidate aliasing (sel probe): multiple support-consistent hypotheses; fit picks train-consistent-but-wrong; B2 (direct sel optimization) failed to flip it | **No arbitration pressure**: with β=0 there is no rate term to prefer the shorter hypothesis; aliases are equally cheap. |
| Oracle-member analysis: true answer among 32 members for 29/144 pairs but oracle *task*-coverage 1/48; member error correlated 0.64-0.74 | **Generation, not selection**: the fits do not produce task-complete correct candidate sets under any diversity axis we built (views, seeds, bulks, agreement). |
| Five dev-30 evals flat at 1-2/30 across every protocol lever; portfolio union 2× best single but voting dilutes it | The levers all operated *downstream* of the switched-off mechanisms. |

The through-line: **every headline pathology is what the theory predicts if
you run its architecture without the phase transition (H-6), without the
fixed point (H-2), without rate pressure (β=0), and without rule transport.**
The season falsified the downstream levers honestly; the upstream
mechanisms were never in the loop.

## 4. Is it a scale problem?

Measured-flat axes (falsified as levers this season): field width d16→d24→
d32 (H-17 killed strong-form; d24 helped val-40 by +2, d32 by 0); corpus
370→1928 tasks (val-hard unmoved); boundary/task-vector width (dt64: same
zeros); fit budget 600→2000 (no unlock); population size 16→32.

Never-scaled axes (the theory's own dials): recursion depth T (=4 always;
[H-2] untested at any other value), per-task adapted capacity (≤64 floats
always; [H-7] deferred), fit length under commitment pressure (grokking-
class schedules are 5-20× our budgets, and our budgets were sized for a
selection frame that fishes backward), rule codebook temperature schedule.

**Verdict: the tested notion of scale is falsified; the untested notion is
exactly where the PI proposes to spend compute, and all four dials are
cheap** (T and τ are inference/fit-time knobs on existing checkpoints; long
fits are minutes-per-task on v5e).

## 5. The benchmark calibration (CompressARC, kept to its proper role)

What the benchmark establishes (their artifacts, our audit — per-task list
vendored, scorer reproduced at 80/400 = 20.0%): ~83.5K weights (our
instrumented count; their "76K" is unreproducible from their code) plus
~10²-10⁵ fitted latent parameters per task, all adapted per task from
scratch, MDL objective, ~21 min/task, 20% eval pass@2. Three calibration
facts matter for us; the rest is their business:

1. **The bar at this compute class**: 20% of eval tasks are reachable by
   ~2000 steps of per-task gradient descent on an ~80K-param model with the
   right objective — no pretraining required. Our 0/48 is not "ARC is
   impossible without LLMs"; it is our solve-time procedure.
2. **Their failure list overlaps ours** (counting, repeated serial ops,
   image-level rigid motions, long-range extension — their Appendix H, our
   taxonomy): those families resist both systems' continuous optimization.
   Roots reading: those are exactly the tasks whose programs need *deep*
   recursion/iteration — evidence FOR the H-2 direction, not for copying
   their layers.
3. **They also lose ~1/3 of achievable headroom to selection** (their own
   released logs: 137/400 in-pool vs 80/400 top-2), and they ship **no
   component ablations at all** (verified). The mechanism-level questions —
   what commitment, convergence, and rate pressure each contribute — are
   open in the field. Our program (§6) answers them on our architecture;
   that is a paper contribution, not a chase.

## 6. The experiment program: run the roots' own tests

Ordered by information per TPU-hour. E1-E3 are instruments on EXISTING
checkpoints — two lane-days total, no architecture changes — and they
adjudicate the phase-transition hypothesis before any build. Gates stay
val-hard(48) + dev-30. Each item: hypothesis → test → kill condition.

**E1 — The order-parameter instrument ([H-6]'s test, part 1). Hypothesis:
solved outputs are produced under low-entropy (committed) rule
distributions; failures under high-entropy mixtures.** Test: log H[q] per
forward on every val-hard fit trajectory + at prediction (aux already
computes it; one plumbing change); compare solved-output vs failed-output
H[q] distributions. Kill: no separation ⇒ commitment is not the missing
ingredient and E2's premise weakens. Cost: ~0.5 lane-day (rerun one
protocol with logging).

**E2 — The commitment intervention ([H-6] part 2 + [H-5]; the PI's
"grokking lock"). Hypothesis: τ-annealing during the fit (1.0 → ~0.1 on a
cooling schedule), with β_flux > 0 as the rate pressure and weight decay
held, converts near-misses by forcing discrete rule commitment — a
measurable phase transition (H[q] collapse mid-fit) followed by stable
generalization (final params ≥ earliest-exact selection).** Long-fit rider:
5-10k steps at d16 to test for delayed generalization (grokking-class
budgets; our 600-2000 were sized for the backward-selection frame). Named
metric: LoO-exact→query-exact conversion rate vs the τ=1 baseline rows (all
eight already measured). Kill: annealed+priced fits convert nothing the τ=1
fits don't ⇒ H-6 falsified at TTT; the discrete-code design itself goes to
[R]. Cost: 1-2 lane-days (48 tasks × 2 arms).

**E3 — The fixed-point probe ([H-2]'s named test; the PI's inference-time
recursion, part 1). Hypothesis: correct answers are fixed points of the
iterate map; wrong answers drift or cycle.** Test: on existing checkpoints,
run predict with T = 1..16 on all val-hard pairs; measure per-pair
convergence (Y_{t+1}=Y_t reached? at what depth?) and its correlation with
exactness; separately check whether the 27-29 oracle-member answers are
stabler under iteration than the wrong majority (an *unsupervised
verifier* candidate — attractor-stability as answer scoring, which would
attack the selection half for free). Kill: convergence uncorrelated with
correctness ⇒ H-2 falsified as stated; deepening T is then pure capacity,
not semantics. Cost: ~0.5 lane-day, no training.

**E4 — Rule transport (the code-smallest intervention with the largest
measured target). Hypothesis: freezing the rule at the task level —
pooling rule_q over support forwards after the fit and decoding the query
under that frozen distribution (or its argmax token) — closes part of the
LoO-exact→query-fail gap (19 task targets exist).** ~50-line change:
forward_fields already takes the pieces; add a frozen-rule decode path.
Kill: frozen-rule decode ≤ re-inference on those 19 ⇒ transport is not
separable from representation, pushing weight onto E2/E5. Cost: 0.5
lane-day.

**E5 — Inference-time MERA recursion ([H-2] part 2; build). Hypothesis:
coarse-solve-then-refine — predict at a coarsened resolution, condition the
fine pass on the coarse answer (renormalization-native scaffolding), iterate
to convergence — reaches the exactness that single-shot decoding misses.**
Staged after E3's verdict (E3 tells us whether iteration has the right
semantics before we deepen it). Cost: build ~3-4 days + 1 lane-day gate.

**E6 — Per-task capacity ladder ([H-7], finally). Hypothesis: query
transfer is limited by adapted-parameter capacity.** Arms at matched budget
and matched E2-winning schedule: e_t(64) → +decoder-LoRA(~2-8K) → full
fine-tune from pretrained init → full fit from random init (the no-prior
control; also the honest answer to "does our pretraining help at all").
Kill for the pretraining bet: random-init ≥ pretrained ⇒ the bulk is dead
weight for solve-rate and the physics program decouples from it. Cost: ~2
lane-days.

**Sequencing**: E1+E3 immediately (instruments, existing checkpoints,
~1-2 lane-days, ~$1-2) → E2+E4 interventions guided by them → E6 with the
E2-winning schedule → E5 build if E3 supports the semantics. Every verdict
lands in the ledger under its original [H-n]. The Aug-10 memo = this
document + E1/E3 first numbers if they land in time.

## 7. What survives unconditionally

The measurement program and its findings (flux ledgers → now candidate rate
terms; coherence instruments; the H-18 matrix; the falsification map as
method), S9 equivariance, the canvas/size machinery, val-hard + dev-30 +
the run-execution standard, and the ledger discipline that made this
post-mortem writable from evidence rather than memory.
