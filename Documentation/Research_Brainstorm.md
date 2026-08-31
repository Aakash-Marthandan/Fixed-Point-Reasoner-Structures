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

---

# Freethink 2026-08-12 — post-grid: the landscape is a code; now BOUND it

*Inputs: the complete seeded grid (four laws + the budget amendment), the
[H-12] hard-negative, the v6e pod, and tonight's literature sweep
(equilibrium reasoners; modern-Hopfield capacity theory; kernel-Hopfield
basin-boundary geometry). Nothing here is registered until it moves to the
ledger with a named test. Lettering continues from cluster M.*

## 0. The organizing observation

Every measured regularity now factors through TWO code parameters with
SEPARATE physical controls: **codebook size N** (count movers: depth,
corpus, dials — "more dynamical capacity") and **code distance d** (radius
movers: pricing, width×budget — "more optimization under constraint").
Coding theory says these two cannot be independent forever: sphere packing
bounds N·V(d/2) by the volume of state space. The freethink's spine: find
the frontier, test whether pricing pushes us toward it, and use the bound's
SHAPE as the next generation of falsifiable predictions.

## N. The packing frontier: are priced landscapes near-optimal codes? **[T1 — zero new compute]**
From every substrate's ladder we hold (N, d) estimates. Coding theory gives
Hamming/GV-style bounds on achievable (N, d) at fixed blocklength. Analysis:
plot all 17 substrates in the (log N, d) plane against the packing bound for
the task-relevant state space; measure "packing efficiency" per substrate.
**Prediction: priced substrates sit measurably closer to the frontier than
free ones at matched N; dials/depth move N along the frontier, not off it.**
If confirmed: priced training = approximate optimal-code construction — the
paper's strongest theoretical claim, from data already on disk. Kill: no
ordering, or efficiency anti-correlates with transfer (frontier proximity
would then be memorization-flavored, not generalization-flavored).

## O. Spectral shape universality: an RG scaling function **[T1 — zero new compute + one big-d check]**
Priced per-scale spectra at different widths look SHAPE-similar (d24:
458/103/62/26/22; d32: 433/92/58/24/28 — near-identical after I_total
normalization) while free spectra steepen with d. **Hypothesis: under
pricing, the normalized information distribution across RG cuts
Î(s) = I_s/I_total is a d-INVARIANT scaling function** — universality of
the information geometry under the flow, the holographic reading being that
the boundary theory (task) fixes the bulk profile regardless of bulk
resolution. Test: collapse plot over all priced substrates (disk); then the
5-75M models must land on the SAME curve — deviation becomes a scaling
diagnostic ("when big models break, the spectrum says where"). Kill: no
collapse at matched β.

## P. The retention metric: anisotropy of basin-destruction **[T2 — the principled TTT]**
[H-12]'s lesson: spine motion at fit-scale destroys basins; e_t motion does
not. Information-geometric reading: retention defines a metric on weight
space (sensitivity of basin structure to perturbation direction); TTT should
move in its NULL directions. Probe (cheap): random unit perturbations of
fixed norm in different parameter subspaces (spine mixers / heads / gates /
e_t / codebook) × retention drop ⇒ the anisotropy spectrum. **Prediction:
orders-of-magnitude anisotropy; the low-sensitivity subspace ⊇ {e_t, gates,
biases} and excludes mixers.** Then the convert-phase TTT = adapt ONLY in
measured-flat directions (a natural-gradient with the retention metric; the
KL-anchor arm is its crude isotropic shadow). This turns the [H-12] negative
into a design principle: *fit where the code isn't stored.*

## Q. Barrier spectroscopy: hop thermodynamics **[T1.5 — one sampling battery]**
Kernel-Hopfield theory (2605.00366) finds attractors separated by sharp,
phase-transition-like barriers with critical slowing — our substrate should
show the same. We hold hop rates at exactly two temperatures (T0=0.1, 0.4 —
nearly equal rates). A proper T-sweep {0.05..1.6, k=64} on the record-radius
40k substrate: **if hop rate follows Arrhenius exp(−ΔF/T), the slope
measures per-task-family BARRIER HEIGHTS** — "reasoning barrier
spectroscopy," a measurable nobody else has. Prediction: barrier heights
anti-correlate with snap-convertibility (low-barrier families are where
candidates/self-decoding work). Kill: rate flat in T (barriers either ≪ or
≫ the accessible T range — itself informative for sampler design).

## R. Boundary sufficiency at scale: the holographic dictionary test **[T2 — rides pretrain-13]**
The throat law says the task fixes information regardless of bulk capacity —
the boundary (e_t, 64 floats) has never been re-examined on the equilibrium
line (H-17 was falsified on the OLD architecture only). **At d64+: sweep
d_task {16, 32, 64, 128}. Prediction (holographic): radius/transfer flat in
d_task — the boundary is task-determined, not capacity-determined; the
minimal sufficient boundary rank per task ≈ the code dimension (measurable
via rank-restricted e_t fits).** Kill: d_task binds at scale ⇒ a
bulk-boundary dictionary constraint (equally interesting: the dictionary
has a measurable capacity law).

## S. The all-eq conversion attempt: self-decoding at record radius **[T1.5 — the convert-phase opener]**
The .80-radius / exact-14 substrate (pretrain12_48c_40k) is the best decoder
candidate ever measured. One cell: the C.3′ stack with THIS decoder ×
eq-native candidates (multi-init trajectories per EqR — registered variant —
plus Langevin at the Q-informed temperature), scored by PoE. **Prediction:
first all-eq task-level conversions; the heirloom (old-arch pool) becomes
optional.** This is the experiment that decides whether the geometry program
cashes out as solves — run FIRST in the convert phase, before eval-6.

## Literature anchors (tonight)
- Equilibrium reasoners (2605.21488): attractor-reasoning at scale, multi-init
  breadth — our differentiation = instruments + pricing + family gates.
- Modern Hopfield capacity (exponential attractors, basins stay large) — the
  count/radius separability has associative-memory theory kin; connect the
  count-movers to capacity-scaling results in the paper's related work.
- Kernel Hopfield basin boundaries (2605.00366): sharp barriers + critical
  slowing — cluster Q's theoretical twin.
- Reasoning-as-compression CIB (2603.08462): B1's LLM-world cousin.

## Priority for the scaling sessions
1. **N + O from disk** (the frontier + the collapse — paper-grade, $0).
2. **S** (the all-eq conversion opener) with **Q** riding its sampler.
3. **P** (retention-metric probe) before ANY 5-75M TTT commitment.
4. **R** rides pretrain-13's d64 pilot; steps(d) calibration from the
   20k/40k pair is its sibling calibration.

# Freethink 2026-08-13 — consolidation-pass design round (clusters T-W)

*Inputs: the frontier-consolidation report (same date), the family-vacancy
table, wave-2 in flight, and a four-query literature sweep. Lettering
continues from S. Nothing registered until it moves to the ledger.*

## T. Coarsening thermodynamics: is consolidation a ripening flow? **[$0]**
Count falls monotonically under budget/scale (51→42→32) while radius and
compression improve — the signature of coarsening (fewer, larger domains).
Lit twin: dynamical condensation / cluster-coarsening in self-attention
(arXiv 2608.08922, this month). Analysis from disk: retention-count vs
effective optimization (steps × width) across all 20+ substrates — power-law
or saturating? PREDICTION: coarsening saturates at a task-set-determined
terminal codebook (the "true" rule count), not N→1. KILL: count keeps
falling through the 80k/quadratic cells — consolidation would then be a
pathology to bound, not a flow to ride.

## U. Water-filling optimality of the knee profile **[$0 — theory-grade if it fits]**
VIB theory: optimal per-channel capacity under a total constraint is a
WATER-FILLING allocation. We hold the universal knee profile
(.69/.14/.085/.035/.048) and the free-arm spectra (the unconstrained
"channel demands"). Fit: is the knee profile the water-filling solution
given the free spectrum's per-scale levels and the measured throat total?
If yes: (i) the priced objective performs constrained rate allocation
optimally — S1/S2's strongest theoretical statement; (ii) FLOORS AT ANY
SCALE become derivable (water-fill at the target throat) instead of copied
from the nearest measured profile. KILL: allocation matches neither
water-filling nor any monotone transform of it.

## V. Fisher-anchored TTT — P2 in buildable form **[convert-phase arm]**
P's verdict: damage is gradient-directed. The Fisher diagonal IS the
gradient second-moment — and Adam's v accumulator in every saved
opt_state is a FREE Fisher approximation (the "Fishers for Free" trick,
arXiv 2507.18807). Build: probe_lora --ewc — parameter-space anchor
lambda * sum v_i (theta_i - theta0_i)^2 alongside the function-space --kl
arm. PREDICTION: Fisher-weighted >= KL >= none on retention-preserving
adaptation (it penalizes exactly the directions training uses to destroy).
KILL: EWC <= plain at matched exactness — curvature-then != damage-now,
and the KL arm stands alone.

## W. Representational basin creation for the vacancy floor **[the family-structure consequence]**
ExtractObjects 0%/11 substrates, Copy ~2%, Count ~7% — no amount of scale,
price, corpus, or dials creates these basins. W-alpha ($0): correlate
vacancy with output geometry (extent ratio, content-cell count, palette
size) — the anchor-row mechanism plausibly under-trains SMALL-output
families (corrupt-target rows on tiny extents carry little basin signal;
masked CE weights shrink with extent). W-beta (C22 candidate,
registered-unbuilt): basins BY CONSTRUCTION — a learned potential/Lyapunov
term shaping the update (Ghost Attractor Networks, arXiv 2606.18315, build
basin structure architecturally rather than by training alone). Gate:
W-alpha's correlate decides whether the lever is loss-side (reweight small
extents) or architectural (C22).

## Scale-regression flag (rides wave-2 analysis)
HorizontalVertical: 78-89% retention at d16-d48 → 0% on BOTH d64 arms. A
family lost to scale for the first time. Slice the d64 spectra/gate values
on HV tasks vs d48's when wave-2 lands; if d64's IR profile starved the
axial summaries (C7's channel), the 5-75M track needs a floor on that
channel class.

## Lit anchors (this pass)
- Ghost Attractor Networks 2606.18315 — basin-structured decoders by
  construction (W-beta/C22).
- Clustered attractor manifolds / dynamical condensation 2608.08922 —
  coarsening theory twin (T).
- Solve the Loop 2605.12466 — implicit-diff attractor module (Neumann-K
  cousin; deep-T memory).
- Fishers for Free 2507.18807 — Adam-v as Fisher diagonal (V).
- FIESTA 2503.23257 — Fisher-selective test-time adaptation (V's TTA kin).
- VIB water-filling allocation (classic) — U's frame.

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
