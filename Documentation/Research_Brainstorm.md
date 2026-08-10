# Research & Brainstorming — QG/Information-Theory Lens on the Architecture

*2026-08-06. A deliberately less-precise document: idea fodder, not commitments.
Nothing here is registered until it moves to the Design Ledger with a named
test. Tiers: **[T1]** buildable this week against existing instruments ·
**[T2]** designable, needs a build cycle · **[T3]** speculative physics
mapping, valuable if it survives PI scrutiny. ❓-boxes mark places where PI
intuition is explicitly requested.*

---

## 1. Field scan (what the outside world measured, distilled to mechanisms)

| finding | mechanism that mattered | what it means for us |
|---|---|---|
| **Product of Experts / "perspectives"** (ICML 2025; 71.6% ARC-1 public eval at ~2¢/task) | Augmented *views* used in generation **and in scoring**: candidates are scored by their likelihood under every view, combined multiplicatively (product of experts), not by majority vote | **Our members already ARE perspectives — but we vote cellwise instead of scoring candidates.** A candidate grid's log-likelihood under member m's view is one forward pass. PoE selection over the candidate set {member predictions} uses fits we already have. [T1] |
| **ARC Prize 2025 report**: refinement loops named the year's defining theme; 2nd place used masked-diffusion + recursive self-refinement + perspective scoring; 1st place leaned on synthetic data + TTT | Iterate-and-refine with a feedback signal, everywhere | Our T-loop refines during *training* (deep supervision) but inference decodes once. **Iterate-until-self-consistent at inference** (stop when yprev fixed-point) is unused. Synthetic data (RE-ARC-class generators) remains our known long pole. [T1/T2] |
| **TRM** (45% ARC-1 at 7M params, paper award) | Recursion depth + tiny width, aggressive supervision | Confirms our recursion bet; our T=6 failure was optimization (gradient dilution at 10k steps), not the idea. Worth one retest at 20k with per-step loss weighting. [T2] |
| **CompressARC** (20%, no pretraining, paper award) | Pure description-length minimization per puzzle | Our thesis's nearest sibling; their edge over us today = transduction (query in the compression problem) + heavy augmentation. Both in our pipeline now. |
| Frontier context: LLMs at 93% ARC-1 / ~54% ARC-2 at $15+/task | Scale + search | Not our lane. Our lane: the params×cost frontier — PoE-ARC's 2¢/task at 71.6% is the *efficiency* rival to watch, not o3. |

**Sharpest import: perspective-scoring (PoE) over member candidates.** We built
the perspectives (views×seeds×bulks); we've been aggregating them with the
weakest possible combiner (cellwise majority). Every strong system aggregates
with *likelihood products*. This costs one forward pass per member×candidate
and no new training.

---

## 2. Physics-lens idea clusters

### A. Bit threads: from cut-sums to flow-paths **[T2, high interpretability value]**
Freedman–Headrick rewrote RT: entanglement = max *flow* of divergenceless
"bit threads," not min *cut*. Our ledger prices **cuts** (I_s, A_s per scale).
The thread picture suggests the dual instrument: **explicit information
*paths* from input cells to output cells through the channel graph** —
computable as attribution flows (which input cells, via which channel class,
feed each output decision). Uses: (i) an interpretability map per solved task
("the thread bundle" — where the rule's information physically routes);
(ii) a *flow-consistency regularizer*: support-pair thread bundles for one
task should be *isomorphic* (same routing topology, different data) — a
transductive signal distinct from output agreement. ❓ **PI: is there a
natural discrete max-flow formulation on our RG graph whose optimum we could
train against, i.e., a trainable Freedman–Headrick dual of our KL prices?**

### B. Holographic QEC / HaPPY: the rule as a bulk logical operator **[T2→T3, the deepest reframe]**
In holographic codes, a bulk logical operator is reconstructible from *any
sufficient boundary subregion* — redundancy is the defining property of the
encoding. Map: rule ↔ bulk logical; members (views/bulk-variants/support
subsets) ↔ boundary subregions. Then:
- **Agreement-regularized populations = enforcing code-like redundancy** (the
  registered eval-7 design acquires a precise physics reading: we are training
  the encoding so the rule lies in every member's entanglement wedge).
- **Erasure test [T1]:** drop random member subsets at scoring; a properly
  "encoded" rule survives ablation gracefully; memorized solutions shatter.
  This is a *measurable code-distance for programs* — instrument exists.
- **[T3] Design speculation**: make the e_t table literally an erasure code —
  per-task programs stored redundantly across K sub-embeddings with a decoder
  that tolerates missing shares; TTT then fits shares, and code distance
  becomes a regularizer. ❓ **PI: does the subregion-duality picture suggest
  which member sets should suffice (wedge inclusion condition) — e.g., any
  4 of 8 views, or specific complementary pairs?**

### C. Entanglement wedges and the query-transfer wall **[T3 framing, T1 consequence]**
Our measured wall: support-consistent programs underdetermine the query.
Holographic language: **the query sits outside the entanglement wedge of the
support-boundary region** — no reconstruction without enlarging the boundary.
Transduction (query input in the fit) *is* wedge enlargement. This framing
predicts an ordering we can test cheaply: transductive signal should help
most on tasks where support-support LoO succeeds but query geometry differs
most from supports (largest "wedge gap" — measurable as support↔query
statistics distance). [T1 analysis on existing predictions.]

### D. c-theorem for training: coherence decay as monotone crossover **[T1 instrument]**
Every bulk shows raw coherence decaying while loss falls — we called it the
memorization transition. RG language: propose a **c-function for training**
— e.g., priced-flux-at-matched-accuracy, or objcons×fit — and check
monotonicity/crossover across all 10+ bulk trajectories we already logged.
If a single monotone organizes all curves, checkpoint selection ("temporal
members") gets a principled criterion instead of a heuristic. ❓ **PI: your
instinct on the right c-function candidate — entropy-like (flux totals) or
order-parameter-like (rule-selection entropy H[q], which we log)?**

### E. Geometry from entanglement: task-adaptive RG trees **[T3, the big architecture swing]**
Van Raamsdonk's dictum — geometry *is* entanglement structure. Our RG tree is
frozen (dyadic) plus C17's measured clusters. The radical version: **build
the coarse-graining tree per task from the measured mutual-information graph
of the support grids** (MI between cell neighborhoods, estimated from the
support set), so the network's geometry reorganizes to each task's
correlation structure. C17 was the first step (clusters); this is the full
step (learned/measured *tree*). Expensive, beautiful, post-deadline unless
eval-6 stalls hard. The cheap probe [T1]: measure MI-graph structure for
solved vs unsolved tasks — if unsolved tasks' MI graphs are systematically
non-dyadic/non-cluster (long-range MI), the frozen geometry is the binding
constraint and E earns its build.

### F. Replicas and wormholes: correlated member error **[T1, already registered as H-18]**
Ensemble members = replicas; shared-bulk error correlation = the "wormhole"
coupling replicas. The physics adds one prediction beyond H-18: error
correlation should *grow with fit sharpness* (later checkpoints = deeper
shared basin). Testable on saved snapshots: member-agreement-on-wrong vs
snapshot step. If confirmed, **early-checkpoint members are not just more
coherent but more independent** — doubling the case for temporal ensembles.

### G. Complexity ≠ information: a second ledger **[T2, likely paper-grade]**
Holography splits entropy (RT) from *complexity* (=volume/action conjectures):
some bulk properties cost computation, not information. Our fractal-tile
failure smells complexity-limited, not information-limited (density right,
structure absent — the rule is short but its *execution* is deep). Proposal:
a **complexity ledger** alongside the flux ledger — per-iteration utilization
(how much the state changes per T-step, `‖z_t − z_{t−1}‖` per scale), priced
or just measured. Prediction: task families stratify on (information,
complexity) as a 2D plane — identity low/low, counting high/low, fractal
low/HIGH. If the plane separates families, we have a second measurable
taxonomy nobody else has, and a principled reason some tasks need more T,
not more d. ❓ **PI: complexity=volume vs complexity=action — which discrete
analog would you trust on our lattice (state-change volume vs action-like
sum over gates)?**

### H. Measurement-induced dynamics: the argmax channel **[T3, watch-item]**
Our T-loop feeds back `argmax(logits)` — a projective measurement between
iterations. Measurement-induced phase transitions suggest the *rate* of
projection matters: hard argmax each step may collapse useful superposition
(candidate ambiguity) too early. Cheap variant [T2]: soft feedback (temperature
on yprev one-hots) annealed across T — keep candidate mixtures alive until
late iterations. Connects directly to the aliasing findings (size candidates,
color assignments): ambiguity we currently destroy at t=1 could be resolved
by later evidence.

---

## 3. Distilled shortlist (merged with the registered pipeline, cheap→heavy)

1. **[T1] PoE member scoring** — replace cellwise vote with product-of-experts
   log-likelihood scoring of member candidates across views. Slots into
   `score_population` unchanged elsewhere. *The field's strongest aggregator,
   free for us.*
2. **[T1] H-18 check + F's sharpness prediction** — member agreement-on-wrong
   vs checkpoint step, from saved predictions.
3. **[T1] Erasure test (B)** — member-subset ablation curves on the existing
   population results; "code distance for programs."
4. **[T1] Wedge-gap analysis (C)** — support↔query statistical distance vs
   transfer success, existing predictions.
5. **[Registered] Agreement-regularized population + cross-bulk members**
   (eval-7 design; B gives it its physics reading).
6. **[T2] Complexity ledger (G)** — one StepOutput field + analysis; then the
   (I, C) plane over dev-30/val-hard.
7. **[T2] Soft-feedback annealing (H)** and **inference-time
   iterate-to-fixed-point** (field scan).
8. **[T2] Thread/flow attribution (A)** — interpretability map + flow-topology
   transduction.
9. **[T3] Adaptive-geometry RG (E)** — gated on the MI-graph probe.
10. **Long pole unchanged**: RE-ARC-class generators (every 2025 winner leaned
    on synthetic data).

## 4. Sources
- [Bit threads and holographic entanglement (Freedman & Headrick)](https://arxiv.org/pdf/1604.00354) · [quantum bit threads of MERA](https://arxiv.org/pdf/1804.00441)
- [HaPPY holographic codes (Pastawski et al.)](https://arxiv.org/abs/1503.06237) · [infinite-dimensional HaPPY / wedge dynamics](https://arxiv.org/pdf/2005.05971)
- [Product of Experts with LLMs for ARC (ICML 2025)](https://arxiv.org/abs/2505.07859) · [code](https://github.com/da-fr/Product-of-Experts-ARC-Paper)
- [ARC Prize 2025 technical report](https://arxiv.org/html/2601.10904v1) · [ARC Prize 2024 report](https://arxiv.org/pdf/2412.04604)
- [CompressARC — ARC-AGI without pretraining](https://arxiv.org/pdf/2512.06104)
- [ARC-AGI-2 benchmark](https://arxiv.org/pdf/2505.11831)

---

# Freethink 2026-08-08 — after the E1/E3 verdicts: attractor engineering

*Inputs: the E1/E3/E3b measurements (ledger 2026-08-08), two literature
sweeps (attractor-shaping mechanisms; ARC TTT/recursion mechanics 2024-26),
and two same-day analyses from saved data. Nothing below is registered until
it moves to the ledger; the companion ledger entry registers the subset we
commit to.*

## 0. New same-day data
- **Dynamics collapse (cluster G's datum):** 53-58% of val-hard pairs
  produce exactly ONE distinct state across 16 iterations; median 1; ≥4
  states only 10-20%. The recursion does no computation for most inputs.
- **Wedge-gap probe (cluster C): NEGATIVE.** Solved outputs sit at *higher*
  support↔query statistical distance (med 1.63 vs 1.00); no low-gap
  enrichment (5/14 vs 7 uniform). The crude distance is wrong or the wedge
  framing is; cluster C demoted until someone proposes a better metric.

## 1. The convergent diagnosis (ours ⊕ the field's)
Our E3b (truth erased 92-94%; own wrong answers stable 72-84%) is the
literature's inverted-exposure-bias pathology, measured directly: **no loss
term ever demanded F(x, y*) = y*** — deep supervision only maps
self-produced states → y*, so basins form around trajectory states, never
around targets. The field hit the same wall and named its repairs:
- **TRM measured HRM's residuals never vanishing** (the fixed-point gradient
  was a fiction); TRM abandoned equilibria; **FPRM instead made them real**
  (pre-norm + learnable residual scaling + damped iteration +
  residual-gated halt) and beats TRM at 7M params: ARC-1 47.5%. [H-2] is
  not wrong physics; it was unbuilt engineering.
- **ARChitects-25**: refinement loop never trained ("not trained to
  re-iterate upon its own first guess" — their words, our measurement);
  works only because random-rate masked training accidentally builds basins
  at all corruption scales; limits untrustworthy → they vote by
  **trajectory visit counts**.
- **ARC Prize HRM ablation**: the outer refinement loop IS the driver
  (+13pp from one extra loop; train-with-16-loops > test-with-16). value of
  iteration is front-loaded and largely a *training-time* signal.
- **Answer/latent split**: every recursive winner carries a decodable
  answer register y AND a separate latent z; merging costs ~15pp (TRM
  ablation). We carry only argmax(y) — no latent survives between steps.
- **Per-task capacity**: task-embedding-only adaptation is measured-dead
  across the field (TRM: blank puzzle_id → 0%; TRM-TTA: full-net FT works,
  LoRA/embedding-only fails; CompressARC fits everything). Our 64-float e_t
  is the field's known-dead configuration — independent confirmation of E6's
  premise before we run it.
- **Aggregation**: PoE's ladder (+26 TTT, +18 vote, +5 PoE, +4 DFS) — the
  augmentation orbit used three ways (TTT data / candidate generator /
  product-of-experts verifier). Selection-side, for when generation moves.

## 2. Mechanism shortlist for basin construction (from the survey, ranked)
1. **Corrupted-target anchoring**: with prob p per supervision sample, set
   yprev = corrupt(y*) at rate ε ~ U[0, ε_max] **including ε=0** (explicit
   idempotence F(x,y*)=y*). Basin radius programmable via ε_max. Near-zero
   cost. (Alain-Bengio 1211.4246; GNS 2002.09405; Growing-NCA damage
   training; the ARChitects' accidental version.)
2. **Self-rollout restarts** (SUNDAE 2112.06749; deep-thinking
   incremental-progress 2202.05826): start supervised segments from k
   gradient-free self-iterations (k random, sometimes from converged wrong
   states) — trains ESCAPE from own attractors; the only mechanism aimed at
   the 72-84% wrong-stability half.
3. **Local contraction + path independence** (Jacobian penalty at (x,y*),
   2106.14342; init-randomization for path independence, 2211.09961) — the
   2000-5000-step arm.
4. **FPRM stabilizer kit** (2606.18206) as the architecture-side fix when we
   next touch src/: residual scaling, damped iteration, residual-gated halt.
5. Schedule multipliers, untested-at-TTT anywhere (open niche): weight
   decay + shrunk-init + Grokfast-EMA (2405.20233) on per-task fits.

## 3. What is publishable regardless of solve-rate
- **The E3b instrument + repair curve**: nobody in the ARC literature
  *measures* basin structure directly (the field infers it from residuals or
  oscillations). GT-retention and the corruption-ladder = a quantitative
  "code distance for reasoning solutions" — thesis S4 made operational —
  with before/after curves under each training mechanism.
- **The order-parameter split** (E1): commitment level vs commitment
  identity — a mechanistic account of why per-input rule re-inference
  fails, with H[q] as the measured variable.
- **The falsification map**: H-2 as-deployed falsified → FPRM-informed
  restatement → repaired-or-not, all pre-registered. Method paper spine.

---

# Freethink 2026-08-10 — post-pivot: the landscape is real; now govern it

*Inputs: the pivot week's measured facts (basins trainable 29→38-49%; β
IMPROVES retention; portfolio law thrice-confirmed; C.3‴ near-parity with
the Center2 loss; transport still open). QG/holography/information theory as
the lens on what to build next. Nothing registered until it moves to the
ledger.*

## New clusters (I–M)

### I. Replica repulsion: train the portfolio FOR complementarity **[T2, high leverage]**
The Center2 lesson: different pretrains are different CODES with different
correctable-error sets; the union decodes more because the sets are
complementary — and we currently get complementarity by ACCIDENT (seeds).
Gravity read: replica wormholes couple copies; we want the opposite — an
explicit ANTI-coupling on errors. Design: pretrain portfolio members with a
pairwise error-overlap penalty (the H-18 matrix machinery, promoted from
instrument to LOSS — members may agree on answers but must not agree on
MISTAKES). Prediction: portfolio union grows faster than seed-replication
(union 26 → ?), snap yield rises. Cheapest form: fine-tune P9-A/B briefly
with the repulsion term against each other's frozen errors.

### J. Basin thermodynamics: the equation of state **[T1 NOW — memo figure from existing data]**
The corruption ladder IS a microcanonical entropy measurement: S(ε) =
log #(ε-corruptions retained). We hold S(ε) curves for five substrates and
two β values. Analysis (zero compute): plot S(ε) per arm; extract dS/dβ
(pricing's effect on basin geometry — measured +7 retention at fixed ε) and
dS/dη (the floor's effect). If the curves collapse under a scaling, we have
an empirical equation of state for reasoning landscapes — the physics
centerpiece figure, and a principled way to choose β/η for pretrain-10
instead of grid search.

### K. Pool mutual information: buy the right portfolio **[T1, from saved preds]**
The candidate pool's value saturates as members correlate (C.3″/‴ measured
it qualitatively). Island-formula flavor: pool information = Σ members −
correlation corrections. Operational: estimate pool→answer MI proxies
(coverage-within-ε vs pool size/composition, marginal gain per added bulk)
from ALL saved member preds. Deliverable: a pre-run predictor of snap yield
per candidate source — decides portfolio purchases (which pretrain arm to
replicate) before spending lanes.

### L. Stationary-flux transport test **[T1→T2; E4's instrument]**
At equilibrium the information flow is stationary — the flux spectrum on
the QUERY forward should MATCH the supports' spectrum when the same rule
transports ([H-15] reborn with equilibrium semantics; bit-threads: same
source, same flow). Instrument: |I_s(query) − I_s(supports)| per scale as a
transport-failure meter; then E4's frozen-rule decode should REDUCE it
where it converts. Doubles as candidate-rule selection (min flux mismatch).

### M. Langevin candidates: the decoder as its own generator **[T2]**
Still missing a native sampling axis (H-5 unbuilt at inference). With real
basins, noise-injected trajectories (Langevin at temperature T, annealed)
sample candidates FROM the decoder itself — possibly recovering
Center2-class diversity without the old architecture. One cell: T>0
trajectory ensembles × snap, vs the old-member pool. Kill: eq-sampled
candidates no more diverse than eq-members ⇒ diversity must come from
weights (cluster I), not dynamics.

## Updated shortlist (merges with the standing queue)
1. **[queued] Attribution cell** — P9-D decoder × OLD candidates (isolates
   the C.3‴ confound; one lane-hour).
2. **[T1 now] J: S(ε) equation-of-state analysis** + **K: pool-MI
   predictor** — both from disk, memo-grade figures.
3. **[builds, unchanged top] B1 (MDL-native objective, query co-compression)
   and E4/L (transport + its flux instrument)** — the foundational debt.
4. **[T2] I: replica-repulsion portfolio** — the first portfolio-as-code
   design.
5. **[T2] M: Langevin candidates.**
6. **Protocol: next task-level claims move OFF val-hard** (≈12
   adjudications; evidentiary decay) — eval-6 dev-30 single shot per the
   holdout law once the attribution cell picks the champion.
