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
