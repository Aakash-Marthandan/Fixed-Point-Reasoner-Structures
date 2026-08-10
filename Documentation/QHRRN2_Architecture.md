# QHRRN-2 Architecture Specification

**Status:** v0.3 — v0.2 added the expressivity audit §16 and Amendments A/B/C; v0.3 registers **Amendment D** (ledger C14, 2026-07-21): KL-priced attention channels at *all* scales ("wormhole tolls"), each with measured flux A_s and price β_nl — required by thesis statement S3 (`Thesis_Information_Holography.md` §6), making nonlocal information demand a per-task measurable. Implemented 2026-07-27 (§2.2b; measured net **−144 params** at d=12 — the VIB messages replace the heavier full-width value path) · **Date:** 2026-07-18/27 · **Working paper name candidates:** HoloMERA / FluxRG / RG-Adapt
**Companion:** `Design_Ledger.md` (the epistemic record) · **Target:** AAMAS 2027, submission early October 2026 · **Budget:** ≤ $4000 GCP

---

## 0. Design contract

Every mechanism in this spec exists to discharge a specific measured failure mode or documented requirement, each tracked in the ledger. §9 states the design-motivation principle; §10 gives the CI gates that make each claim falsifiable *before* any cloud spend. Nothing in this architecture is decorative: if a component can't be measured or ablated, it doesn't ship.

The three load-bearing mechanisms, in one breath each:

1. **Priced holographic streams** — every coarse-graining step splits its input into a *kept* channel (flows up the RG hierarchy) and a *boundary stream* (retained, bottlenecked, re-injected during decoding at the matching scale). Nothing is destroyed; everything transmitted is *priced* via an information penalty. The optimizer chooses, per task, how much UV detail to pay for. This resolves the identity-vs-abstraction tension that no funnel architecture can.
2. **Symmetry stack** — exact color-permutation equivariance (colors as a set axis), translation equivariance (convolutional structure), D₄ handled by orbit augmentation + test-time symmetrization (strict D₄-tied kernels as a stretch ablation). Symmetry divides parameters and sample complexity simultaneously: this is the parameter-golf engine.
3. **Discrete rule selection as measurable symmetry breaking** — a small codebook of rule tokens at the IR point, selected by temperature-annealed categorical attention. The entropy of the selection distribution is an *order parameter*; its collapse during test-time training is the honest version of the SSB story, and it's plottable.

---

## 1. Data representation

- **Canvas:** fixed 32×32 (covers ARC's 30×30 max). One JIT shape, compile once.
- **Cell alphabet:** 12 states = 10 ARC colors + `VOID` (outside the true grid) + implicit "empty within grid" = color 0 as usual. `VOID ≠ black`: padding can never be confused with background at the representation level — the "vacuum" is an actual distinguished state.
- **Color axis as a set (AMENDED v0.2 — Amendment A):** the state carries a **per-color field** `Z⁽ᶜ⁾ ∈ ℝ^{H×W×d}` with *weights shared across colors* plus permutation-equivariant interaction terms (DeepSets/mean-pool style). The symmetric group is **S₉ over colors 1–9**; **black (0) and `VOID` are distinguished fields** with private weights — matching ARC's empirical convention (black ≈ background; standard palette augmentation permutes 1–9 and fixes 0).
  **Color symmetry breaking at TTT:** a strictly S₉-equivariant network cannot represent color-*constant* rules ("always paint it red") — permute the palette and the output must permute, contradiction. Fix: θ_task includes **per-color bias vectors** (10 × d ≈ 160 params, initialized 0, ~0 during pretraining). The equivariant core is the symmetric phase; the support pairs are the explicit breaking field; the TTT biases are the order parameter. Color-constant rules become learnable *precisely and only when the evidence demands them* — the SSB story extends to the color sector, and this is the correct prior (most ARC rules are color-relational; the exceptions break the symmetry through data, not weights).
- **Input embedding:** per-color occupancy map `o⁽ᶜ⁾ ∈ {0,1}^{H×W}` → 3×3 conv (shared across colors) → `Z₀⁽ᶜ⁾`. Each color field starts as "where my color is, locally."
- **Episodes:** a training example is a whole episode (support pairs + query), matching the TTT deployment condition.

**Why this matters:** the network cannot even *represent* a color-specific rule in its weights — rules become functions of color *relations* (same/different, majority, containment), which is what ARC rules actually are. Palette-swap generalization is exact by construction instead of hoped-for. Parameter cost of color processing drops from O((10d)²) to O(d²).

---

## 2. The RG cell (shared core)

One cell `R_θ` implements one renormalization step `H_s → H_{s+1} = H_s/2`. The same cell (with scale/iteration modulation) is applied at every scale — one core, five scales.

### 2.1 Seam mixer (the disentangler, wired correctly this time)

- Partition the grid into 2×2 blocks **offset by (1,1)** relative to the pooling blocks that follow.
- Within each offset block, the 4 site features interact through a residual GELU MLP:
  `u = concat(z₁..z₄, z̄₁..z̄₄) → MLP → (Δz₁..Δz₄)`, `zᵢ += Δzᵢ`,
  where `z̄ᵢ` is the color-mean at site i (the permutation-equivariant cross-color term).
- Optionally a Cayley-orthogonal linear component for TTT stability — an *ablation*, not a pillar.

**Why:** this is MERA's staggered disentangler layout (deep-learning twin: Swin's shifted windows — a known-working pattern, which de-risks it). It is the D2 fix: information at pooling-block boundaries is mixed *before* pooling can sever it, and it is **nonlinear** — the E1 linearity collapse is dead at this layer. It is also the D1 fix: these block-local interactions *are* the lattice bonds; locality (Area Law) is enforced because each cell only ever talks to its neighborhood, at every scale.

### 2.2 Coarse-graining with conservation (isometry → kept ⊕ streamed)

Aligned 2×2 blocks pool to one coarse site. The block vector `u ∈ ℝ^{4d}` splits:

- **Kept channel:** `k = GELU(W_k u) ∈ ℝ^d` — flows to scale s+1.
- **Boundary stream:** `(μ_s, log σ_s) = W_b u ∈ ℝ^{2·d_b}` — a variational code `b_s ~ N(μ_s, σ_s)` retained at the coarse resolution of scale s+1, `d_b ≈ 6`.

**Why:** this is the honest version of the documents' "isometry keeps Signal, discards Noise." Nothing is discarded — the complement of the kept channel is shunted to a *priced* stream. The cell is (softly) information-conserving, which is what the unitarity/reversibility language in the original docs was actually reaching for. Funnels delete; QHRRN-2 *files*.

### 2.2b Priced attention at every scale — "wormhole tolls" (AMENDED v0.2 — Amendment B; v0.3 — Amendment D)

Attention runs at **all** scales (32×32 and 16×16 included — Amendment D upgrades Amendment B's coarse-only placement), and every attention channel is **priced**. The mechanism (`cell.attention`): the pattern is computed S9-safely from the field-mean (`q, k` from z̄); each site then emits a **variational message** `(μ, log σ) ∈ ℝ^{2·d_a}` (d_a ≈ 6, projections shared across fields); the sampled messages `m ~ N(μ, σ)` are transported by the attention pattern and injected residually. The per-scale toll is `A_s = Σ KL(N(μ,σ) ‖ N(0,1))` — encoder + decoder contributions at the same resolution — reported in `StepOutput.flux_attn` and priced by its own coefficient **β_nl**.

**Pricing at emission, deliberately:** messages are sampled *before* the convex attention mixing, so by data processing Σ A_s upper-bounds the information that actually crosses the nonlocal channel — the same variational status as the stream ledger I_s, and the accounting point is where "wormhole toll" literally applies: at the mouth. Emission-side coding also transports d_a dims instead of d, which is what keeps 1024-token attention affordable (~12M MACs for the pattern at 32×32; measured ~2 s/step CPU at the CI-3a batch, negligible on TPU).

**Why attention at all (Amendment B's case, unchanged):** long-range *correspondence* — symmetry completion about an arbitrary axis, "copy patch A onto marker B", same/different comparison across distance — is the known structural weakness of purely local hierarchies. Strictly local seam mixers route such correlations through many layers with positional blur; attention computes correspondence directly while the streams supply fine registration.

**Why all scales, priced (Amendment D's case):** with the channel present but *tolled* everywhere, the Area-Law prior stops being an architectural axiom and becomes a **measured default the optimizer pays to relax** — the exact same move C5 made for UV detail, now for nonlocality. Each task's solution yields the decomposition **(I_local = Σ I_s, A_nonlocal = Σ A_s)**: hierarchical vs nonlocal information demand, the measurable that thesis statement S3 (Locality-Class Law) is built on. The EFT reading of Amendment B ("relevant nonlocal operators enter at the IR") is no longer imposed — it becomes a *prediction*: if it's true, the optimizer will spend A_s at coarse scales and starve the fine-scale channels; if a task family needs fine-scale wormholes, the ledger will say so. Ablation axes: β_nl (priced vs free) and `attn_max_hw` ∈ {32, 8, 0} (all-scales vs Amendment-B coarse-only vs absent).

### 2.2c Axial summaries (AMENDED v0.2 — Amendment C)

At each scale, append to every site's features the permutation-equivariant pooled summaries of its **row and column** (mean/max, per color, ~0 params, one matmul). 

**Why:** row/column-global rules — gravity/compaction ("everything falls"), ray casting, "move to the last empty cell in the column" — are prefix/aggregate computations along an axis. The isotropic 2×2 pyramid never materializes 1×32 reductions, and pure seam-mixing advances a falling object ~2 cells per pass. Axial summaries + two recursion passes express compaction directly (position = prefix count of occupied cells — verified constructively on pretrain task `1e0a9b12`, a column-compaction task). Cheap, translation-equivariant, and closes a whole family.

### 2.3 Scale–iteration modulation (the hypernetwork, kept from the original design)

Cell weights are modulated by (scale s, recursion step t): `θ(s,t) = θ_base + A(s,t)B(s,t)ᵀ` (LoRA rank ~4) or FiLM gains. One tiny embedding-MLP generates the modulation.

**Why:** the θ(s) = θ_base + Δ(s) idea from the January proposal was good — it's how one ~5k-param core covers UV texture handling *and* IR object logic. Modulating by t as well lets early recursion passes do coarse hypothesizing and late passes do refinement.

---

## 3. Encoder, IR latent, rule codebook

- **Encoder:** 5 cell applications: 32→16→8→4→2→1, emitting streams `b₁..b₅` (at resolutions 16,8,4,2,1) and the IR state.
- **IR summary:** color-pooled, spatially pooled invariant vector `h ∈ ℝ^{64}` + per-color IR features (for color-binding decisions).
- **Rule codebook:** `K = 128` learned rule tokens `e_k ∈ ℝ^{64}`, `M = 2` selection slots (compositional rules: "move AND recolor"). Each slot attends: `q_m(k) = softmax(⟨W_m h, e_k⟩ / τ)`.
  - Pretraining: τ moderate (soft mixtures), Gumbel-softmax gradients, codebook usage regularization (EMA updates, usage-entropy bonus — standard VQ hygiene).
  - TTT: τ annealed 1 → 0.05. **The collapse of H[q] is the symmetry-breaking event.** Degenerate low-τ vacua = rule hypotheses consistent with the support set; support evidence breaks the degeneracy. We log H[q] per step — the phase-transition plot is a paper figure, and it *means* something: the order parameter is a distribution over discrete hypotheses, not an optimizer artifact.

**Why K=128, M=2:** ARC's rule inventory at the granularity a conditioning code needs is O(10²) (transformation families × arguments get handled by the decoder conditioning, not the code). M=2 covers the common "two-step" compositions; M is an ablation axis. If discrete selection underperforms, the fallback is a continuous rule vector — everything else survives, we lose one figure.

---

## 4. Decoder: holographic reconstruction with gated, priced streams

Mirror pyramid, separate decode core `D_θ` (sharing with encode core is elegant but empirically brittle; separate is +~5k params):

At each upscale s+1 → s:
1. Upsample coarse state (2×2 nearest + linear).
2. **Stream injection:** `z_s += G_s ⊙ MLP(b_s)` where the gate `G_s = σ(g(rule code, s, t)) ∈ ℝ^{d_b}` is rule- and scale-conditioned.
3. Seam-mix (offset blocks, as encoder).

Output heads:
- **Grid head:** per-color logits via the equivariant readout `logit_c = ψ(Z⁽ᶜ⁾, Z̄, V)` + void logit → 11-way softmax/cell.
- **Canvas head:** `(H_out, W_out)` as two 30-way classifications from `h` + rule code. Train-time loss masks by *true* canvas; test-time output uses *predicted* canvas (no E13/D13 ground-truth-size leak). Report exact-match and size-given accuracy separately.

**Why the gates matter:** the identity task needs `G ≈ 1` at fine scales (full UV transmission — your point that ARC needs UV data is architecturally honored here); "fill everything with the majority color" needs `G ≈ 0` everywhere (pure IR). The rule code chooses. A pure funnel has no UV→output path at all; a U-Net has a free unpriced path (lazy copying); QHRRN-2 has a *tollgated* path.

---

## 5. Recursion (depth without parameters)

TRM-style outer loop, T ≈ 6 iterations:

- State per iteration: `(X, Y_{t-1})` embedded jointly (previous prediction as extra per-color occupancy channels; `Y_0` = blank).
- Full encode→rule→decode pass emits `Y_t`.
- **Deep supervision:** the loss hits every `Y_t` (weights `w_t` uniform or ramped). Prevents drift, gives gradient signal at effective depth ~ T × 10 cell applications ≈ 60 layer-equivalents from two ~5k-param cores.

**TRM correspondence** (useful for the paper's positioning): TRM's answer `y` ↔ our `Y_t`; TRM's latent scratchpad `z` ↔ our stream stack + IR state; TRM's weight-shared recursion ↔ our radial re-flow. The difference: our recursion has *scale semantics*, exact symmetry structure, and priced information routing — TRM is a volume-law MLP/attention block at 7M params; we target ≤0.5M with structure doing the work parameters do in TRM.

---

## 6. Objective: the flux ledger

Per episode:

```
L = Σ_t w_t · CE_masked(Y_t, Y*)          (masked to true canvas; void excluded from CE weighting pathology)
  + β · Σ_s I_s                            I_s = Σ_sites,colors KL( N(μ_s,σ_s) ‖ N(0,1) )   [nats crossing cut s]
  + β_nl · Σ_s A_s                         A_s = attention-message KL at scale s (Amendment D — wormhole tolls)
  + λ_size · [CE(H_out) + CE(W_out)]
  + κ · (codebook commitment / usage terms)
```

- `I_s` is a variational upper bound on the information crossing RG cut s — the **flux ledger**. β sets the exchange rate between fit and abstraction. `A_s` is the same bound for the nonlocal channels (§2.2b); β_nl prices nonlocality separately, and the pair (Σ I_s, Σ A_s) is the S3 locality decomposition.
- **The RT-flavored claim (stated as analogy, tested as mechanism):** complexity of the inferred rule = total boundary information flux through the RG cuts. Simple rules ⇒ low total flux; texture-carrying rules ⇒ flux concentrated at fine cuts. The per-task **flux spectrum {I_s}** at the TTT solution is a new, quantitative, per-task interpretability object — "which scales does this task's rule live at." Clustering ARC by flux spectra is a headline figure candidate no other ARC system can produce.
- β schedule: warmup 0 → β* during pretraining (let reconstruction work first, then price it). β* swept in Phase 2; per-scale β_s if needed.

**Design note:** the entropy penalty acts on *states crossing cuts* — the object the theory documents specified — not on weight-matrix spectra — and CE is masked to the true canvas.

---

## 7. Test-time training: boundary-condition adaptation

**Frozen:** both cores, codebook, embeddings, readouts (the "bulk geometry").
**Adapted per task θ_task (~10–25k params):** LoRA modulators (rank 4) on cell linear maps per scale, rule-slot queries `W_m h`, stream gate biases. In the holographic frame: the task fixes *boundary couplings*; the bulk is universal.

Protocol:
1. Augment support pairs by the symmetry orbit: D₄ × color permutations (subsampled, ~50–200 variants) × small translations. With 3 support pairs → effective support ~500–2000.
2. AdamW on θ_task, ~300–800 steps; τ annealed 1 → 0.05 across the run (the SSB anneal).
3. **Leave-one-out validation:** hold out each support pair in rotation; LoO accuracy = early-stopping signal + candidate score.
4. **pass@2:** two attempts = top-2 distinct predictions by LoO score (across the TTT trajectory and 2 seeds).
5. Optional ablation: SGLD with the *correct* noise scale √(2·lr·T) — an honest nod to the Langevin story, reported as an ablation row, not load-bearing.

**Vectorized population TTT (systems contribution):** θ_task is tiny and identically-shaped across tasks ⇒ `vmap` the entire TTT loop over ~64 tasks simultaneously on one TPU. Full 400-task ARC-1 eval becomes a few TPU-hours instead of a serial day. This is what makes the β × d × T sweep grid affordable inside $4000, and it's a reportable engineering novelty.

**Design note:** adapting *all* parameters against 2–5 examples overfits by construction; here ≤25k structured params adapt against 500–2000 orbit-augmented examples, with validation-based stopping.

---

## 8. Pretraining

- **Corpus:** RE-ARC-style procedural generators over the 400 public training tasks' rule families (Hodel's re-arc as base; extend with symmetry/object-dynamics generators), target 100k–400k episodes. Optionally BARC-class synthetic tasks for diversity. **No evaluation-split contact, ever.**
- **Regime:** episodic (support + query per example) so the pretrained network is already few-shot-shaped; augmentation orbit always on; AdamW, cosine decay; deep supervision on all T iterates.
- **Scale:** d ∈ {12, 16, 24, 32} sweep. Parameter table (both cores + hypernet + codebook + heads):

| d  | approx. params | note |
|----|----------------|------|
| 12 | ~60k  | sanity scale (Mac-trainable) |
| 16 | ~100k | expected operating point |
| 24 | ~220k | if capacity-limited |
| 32 | ~400k | ceiling; still 17× under TRM |

(All ≪ 10M spec cap; parameter golf headline target: **TRM-competitive accuracy at 10–30× fewer parameters**, with the honest fallback framing "X% of TRM's score at Y× fewer params" if we land short.)

---

## 9. Design-motivation principle

Every mechanism above answers a measured failure mode of naive designs (lossy funnels, unmasked padded losses, unstructured whole-network adaptation) or implements a documented requirement. Per-mechanism rationale is inline in §§1–7; each component's epistemic status and named test live in the ledger (`Design_Ledger.md` §2). The full derivation trail is preserved in the local archive.

## 10. CI gates (all local/CPU, all before cloud spend)

1. **Equivariance exactness:** permute palette → logits permute bit-exactly; translate input → output translates (within canvas).
2. **Nonlinearity/rank probe:** superposition violated; output rank > 1000 over random inputs.
3. **Sanity triad via TTT:** identity, color-swap, translation — 3/3 exact match.
4. **Seam test:** a task whose rule crosses pooling-block boundaries (e.g., 2-px-offset checkerboard completion) must not degrade vs. block-aligned variant by more than ε.
5. **Flux behavior:** identity's fine-scale flux ≫ constant-fill's total flux; β=0 vs β>0 changes spectra in the predicted direction.
6. **Canvas:** size prediction exact on size-preserving tasks; no use of GT size anywhere in the predict path.

## 11. Ablation matrix (paper table, each row = one mechanism claim)

| Ablation | Tests the claim |
|---|---|
| streams off (pure funnel) | UV transmission is necessary (funnel control) |
| β = 0 (free streams / U-Net mode) | pricing improves generalization, not just compresses |
| aligned (non-offset) seam blocks | staggered disentanglers matter (MERA vs tree) |
| attention: priced (β_nl>0) vs free (β_nl=0) vs coarse-only vs absent | Amendment D: tolls select genuine nonlocality; S3 architecture-selection law |
| color-set → plain embedding | exact color equivariance beats learned embedding |
| discrete code → continuous vector | SSB selection helps (or is just pretty) |
| T = 1 vs 6 | recursion depth contribution |
| TTT: frozen-core LoRA vs full fine-tune | the D6 lesson, quantified |
| + strict D₄ G-conv (stretch) | symmetrization vs strict equivariance |

## 12. Compute plan (≤ $4000)

| Item | Est. |
|---|---|
| Phase 1–2 dev (Mac + v5e-1 spot spot-checks) | ~$50 |
| Pretraining sweeps: d × β grid, short runs, v5e-8 spot | $300–600 |
| Full pretrains (3–4 configs × ~30–60 h v5e-8 spot) | $400–900 |
| Vectorized TTT evals (ARC-1 400-task, ~10–15 full evals) | $500–1200 |
| ARC-2 attempts + final reruns + seeds | $300–600 |
| **Total** | **~$1600–3300** (headroom ~$700–2400) |

Spot instances + the dispatcher's unconditional-teardown discipline (add `--spot`); results to GCS, not stdout-scraping.

## 13. Risks & fallbacks

| Risk | Mitigation / fallback |
|---|---|
| Streams enable lazy copying; abstraction never forms | β warmup + per-scale β_s; monitor flux spectra; worst case, β fixed high at fine scales |
| Codebook collapse / dead codes | EMA + usage entropy (standard VQ hygiene); fallback: continuous rule vector |
| Color-set axis underperforms | A/B in Phase 2 week 1 — cheap to swap to plain embedding |
| Capacity short at d=16 | dial d (params ∝ d²); 32 still 17× under TRM |
| TTT wall-clock for 400 tasks | vmapped population TTT (64-wide); measured before Phase 4 |
| AAMAS fit | frame TTT as an *adaptive agent* (online adaptation under few-shot supervision); learning-and-adaptation track; confirm venue with Prof |
| Deadline | stretch items (strict G-conv, M>2 slots, ARC-2 depth) are severable; gates are go/no-go dates per report §7 |

## 14. Related work positioning (novelty defense)

- **RG-Flow** (Hu, Li, You et al.): MERA-topology normalizing flow for *generation*; no task adaptation, no priced skips, no discrete rule selection. We are the task-adaptive, discriminative-with-reconstruction counterpart.
- **Neural RG** (Li & Wang 2018): learned RG for sampling physics models; same family tree, different task.
- **U-Net / Swin:** skip connections and shifted windows exist — *unpriced* and without scale-semantics; our contribution is the flux ledger + rule-gated routing, and the RG framing that makes {I_s} meaningful.
- **HRM (27M) / TRM (7M):** recursion + deep supervision at small scale, volume-law cores; we add exact symmetry structure, scale semantics, priced information routing, at 10–30× fewer params.
- **TN classifiers (Stoudenmire et al.):** multilinear TN models; no locality-hierarchy-adaptivity triple, no ARC.
- **TTT for ARC (MindsAI lineage, Akyürek et al.):** validated TTT + augmentation orbit; we adopt the protocol but adapt only structured boundary parameters, and vectorize the population.

**Claimed novel contributions (paper):** (1) priced per-scale boundary streams as an adaptive abstraction mechanism (holographic Occam); (2) per-task RG flux spectra as an interpretability object; (3) rule selection as measurable annealed symmetry breaking; (4) the equivariance-stacked recursive tiny-core, placing a new point on the ARC params-accuracy frontier; (5) vectorized population TTT.

## 15. Open questions for PI review

1. **Strict D₄ G-convs now or later?** My call: symmetrization first (cheap, 90% of the benefit), strict equivariance as ablation. Override if you want the cleaner theory story from day one.
2. **M = 2 rule slots** — enough compositionality, or do you want a small program-like chain (M=3-4 with ordering)?
3. **Episodic pretraining** (my default) vs TRM-style supervised-with-augmentation on the training set — episodic matches deployment; supervised is closer to TRM comparability. Could run both at d=16 in Phase 3 if budget allows (~$300).
4. **How physics-forward to pitch the RT analogy** in the paper: "inspired-by" framing (safe) vs a formal section mapping flux ledger ↔ entanglement-entropy bounds (riskier, more distinctive). Your hep-th judgment call.
5. Canvas head: direct (H,W) classification vs size-transform taxonomy — direct is simpler; taxonomy is more interpretable. Default: direct.

---

## 16. Expressivity audit (v0.2) — can it represent the ARC function class?

**Method:** enumerate the functional requirements ARC-1 empirically demands (Chollet's core-knowledge priors + task-family analyses), and for each give either a constructive representability argument in this architecture or an honest flag. The audit produced Amendments A (S₉ + TTT color biases), B (coarse attention), C (axial summaries) — v0.1 could not represent color-constant rules at all, and was weak on correspondence and axial-global families.

### 16.1 Family-by-family

| ARC requirement | Expressing mechanism | Confidence |
|---|---|---|
| Identity / texture-preserving edits | Constructive: mixers ≈ id (residual → 0), gates open, streams carry 4-subcell occupancy (needs ~4 bits ≤ d_b=6 dims), decoder re-places. Exact identity exists in weight space; CI gate tests TTT *finds* it | **High (proved)** |
| Geometry: translate / rotate / reflect / tile | Translation native to conv structure; D₄ via orbit + symmetrization; tiling via rule-conditioned decoder + canvas head | High |
| Object formation (connected components) | Hierarchical assembly: a component of diameter k is assembled by scale log₂k; seam mixers prevent boundary splitting; pretraining auxiliaries (predict component ids/sizes/bboxes — labels free from generators) reinforce | Med-high |
| Counting | Components-minus-holes is a *local* quad-count (Gray's identity for the Euler characteristic — a topological invariant computed as a local density, exactly in the network's wheelhouse); hole-free counting exact; general counting via hierarchy + IR pooling | Med-high |
| Select / argmax over objects ("largest", "the odd one") | Coarse-scale competition via Amendment-B attention (soft argmax over ≤64 tokens) + iterative sharpening across T passes + fine-scale masks from streams | **Medium — top learnability risk (R1)** |
| Long-range correspondence (symmetry completion, copy A→B) | Amendment-B attention computes correspondence at IR; streams provide content; recursion refines registration | Medium (R2) |
| Row/column-global rules (gravity, rays, compaction) | Amendment-C axial summaries: prefix/aggregate statistics per axis; compaction = prefix-count (constructive on `1e0a9b12`) | Med-high with C |
| Color-constant rules ("paint it red") | Amendment-A TTT per-color biases (symmetry breaking by evidence) | High with A |
| Relational color logic (recolor by rank/count/mapping) | Native to the set-equivariant color axis | High |
| Conditional branching (if P then A else B) | Rule slots (M=2) + gates + FiLM; per-object conditions through object features | Medium |
| Grid-size transforms (crop/bbox/scale/tile) | Canvas head + rule-conditioned decode | High |
| Small arithmetic (×N, N from a count) | Count features → canvas/tiling conditioning | Medium |

**Residual, knowingly out of reach:** long *sequential* simulations with no hierarchical shortcut (multi-bounce trajectories, order-dependent stacking of many objects) — bounded by circuit depth ≈ T×(10 cells + attention) ≈ 60–100 nonlinear steps with log-depth shortcuts; tasks needing genuinely serial O(N²) simulation exceed it. TRM has the same ceiling; these families are a small ARC-1 minority. T is a TTT-time dial (compute at 100k params is cheap; T up to 16 if a task's LoO score wants it).

### 16.2 Representable ≠ learnable — the honest gap, ranked

Representational failure classes are closed by construction (and CI-gated). The remaining risk is *learnability*: whether pretraining + TTT actually finds the representations. Ranked, with the gate that catches each early:

| # | Risk | Mitigation | Early gate |
|---|---|---|---|
| R1 | Object-argmax/selection doesn't emerge from pixels | Auxiliary object heads during pretraining (component ids, sizes, bboxes — free labels from generators); Amendment-B competition substrate | dev-30 stratified: object-selection family, Aug gate |
| R2 | Correspondence tasks stay unlearned | Amendment B; symmetry-completion family over-sampled in generators | dev-30: symmetry family |
| R3 | Stream/VIB collapse (β kills streams → funnel redux) or the reverse (β→0 lazy copying) | Free-bits floor per stream; β warmup; per-scale β_s; {I_s} monitored every run | identity CI + flux-direction sanity |
| R4 | Codebook pathologies (collapse/dead codes) | EMA + usage-entropy hygiene; fallback: continuous rule vector (loses one figure, not the system) | code-usage histograms, Phase 2 |
| R5 | Generator-distribution gap (RE-ARC ≠ eval rules) | Mix RE-ARC + BARC-style + hand-written families; dev-30 uses *real* tasks only | dev-30 vs generator-holdout delta |
| R6 | TTT overfits 2-pair tasks | LoO stopping; θ_task kept ≤25k; orbit augmentation | LoO-vs-test correlation on training split |

### 16.3 Why a decent solve rate is credible (the argument's load-bearing structure)

The solve-rate bet deliberately does **not** ride on the novel physics. It decomposes:

1. **Proven engine, kept:** recursion + deep supervision + orbit-augmented TTT is TRM's demonstrated 45%-at-7M recipe (and the MindsAI-lineage TTT results before it). We keep that engine intact.
2. **Params removed by exactness, not hope:** TRM spends capacity learning approximate invariances from its augmentation orbit; we impose translation + S₉ exactly and D₄ by symmetrization. Equivariant nets matching or beating augmented baselines at a fraction of the parameters is a repeatedly replicated finding (G-CNN literature) — this is where the 10–70× parameter reduction comes from without a solve-rate sacrifice.
3. **The bottleneck problem TRM never had, solved structurally:** TRM's full-width recurrent state gives it UV access by brute width; streams give the same access at a fraction of the state and parameters — plus the Occam pressure and the measurables. Streams are how we stay tiny *while keeping* TRM-class capability; the flux machinery's accuracy upside (better composition/generalization) is a bonus bet, priced as ablations.
4. **Honest expectation:** target band **30–50% ARC-1 public-eval** at ≤400k params (TRM-competitive at 17–70× fewer params); floor scenario ~15–25% if R1/R2 bite hard — detectable by Aug 31 at the dev-30 gate with time to descope; upside >50% if priced composition genuinely helps. Any point in the target band is a new params-accuracy frontier point, which is the paper's efficiency claim; the physics deliverables (flux spectra, SSB transitions) do not depend on beating TRM.

**Efficiency, quantified:** ~100k params (d=16); pretraining ≈ one v5e-8-spot day; vectorized TTT ≈ $0.10–0.20 per task at evaluation (compare: frontier-LLM refinement harnesses at ~$31/task on ARC-2) — three orders of magnitude cheaper per task, at 70× fewer parameters than the smallest strong baseline.

---

## 17. Amendment F — B1: the MDL-native episode objective (registered 2026-08-10, unbuilt)

**Debt being paid.** S1/S2 have been instruments, not objectives, since v0.2: the
flux ledgers *measure* description length but the loss optimizes distortion
(masked CE) plus, since pretrain-9C, a small channel toll (β_flux=3e-5 — the
first load-bearing rate term, +5–7 retained pairs over unpriced seeds, arm-pair
overlap showing a *reshuffled* retained set: pricing perturbs the landscape, it
does not yet organize it). B1 makes the objective itself a two-part code, the
CompressARC-convergent move on our substrate with our instruments.

**Form (per episode, K supports + query x_q):**

    L_B1 = Σ_k CE(y_k | x_k, rule)                    [data | rule]
         + λ_rule · CL(rule)                          [rule cost]
         + Σ β_s I_s + β_nl Σ A_s   (all forwards)    [channel usage]
         + λ_tx · T(x_q)                              [query co-compression]

- **CL(rule)** — two-part-code rule cost: −log P(codes) under a *learned
  usage prior* over the codebook (M×K logit table, EMA-updated at corpus
  scale). A rule used by many episodes is cheap; an episode-private rule is
  expensive — memorization priced where it actually lives (H-4's original
  intent, finally at the rule level rather than only the channel level).
- **T(x_q)** — transduction terms needing no query labels:
  (a) the query forward's channel usage priced identically (the rule must
  route the query cheaply too), and (b) **rule-transport consistency**
  KL(q(x_q) ‖ q̄_supports) — the E4/CI-10 committed rule and the cluster-L
  stationary-flux meter promoted from instrument to loss term.

**Staging (each stage carries its named test; ledger discipline):**

| Stage | What | Named test | Kill condition |
|---|---|---|---|
| **B1-lite** [H-25] | TTT-side only: add T(x_q) (query flux toll + rule-consistency KL) to keyhole fits; ~30 lines in the fit, cfg-gated | keyhole battery ± B1-lite on one eq substrate: GT-retention, exact@sel, ε-ladder; mechanism variable pre-measured by probe_e4 (fraction of pairs where query self-codes ≠ committed codes) | retention or exact drop vs baseline fits |
| **B1-full** [H-26] | pretrain-10 objective: CL(rule) with learned usage prior + β from the P9-C knee + T(x_q) on episodic queries | the pretrain-9 battery protocol (retention, ladders, spectra) vs P9-C; flux spectra must compress at equal-or-better retention | retention transfer worsens vs P9-C, or codebook collapses (R4 hygiene gates) |

**Why now.** Three 2026-08-10 measurements make B1 the ranked next build: the
priced landscape survived its first load-bearing test (P9-C); the pool-MI
analysis found a 26pp correlation cost with a hard core of 36/144 pairs no
candidate machinery covers (better *selection* cannot reach them — only a
better landscape or transport can); and probe_e4 gives B1-lite's mechanism
variable for free. B1-lite is one fit-code change measurable in a day locally;
B1-full is pretrain-10's headline arm.
