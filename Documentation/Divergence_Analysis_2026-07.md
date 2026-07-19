# QHRRN Post-Mortem: Divergence Analysis & Redesign Review

**Date:** 2026-07-16 · **Scope:** commit `d874427` (April 29, 2026) vs. the four design documents in `Documentation/` · **Method:** full code read + numerical probes against the trained `core_checkpoint.pkl` on CPU (JAX 0.10.2)

---

## 1. Executive summary

The April implementation did not fail because the physics ideas were wrong. It failed because **the physics was never implemented**. The delivered network is, provably, a **single exactly-linear map with a rank-8 bottleneck** — no PEPS contraction, no MERA seam disentanglers, no state entanglement anywhere, no nonlinearity of any kind between input and logits. A model of this class cannot express the identity rule ("copy the grid") on unseen inputs, let alone conditional reasoning. Every downstream mechanism (Langevin annealing, holographic penalty, block-sparse kernels) was either mis-derived, disconnected from the model, or reduced to a no-op — while the tests and README asserted success in increasingly emphatic language.

The core ideas — reasoning as RG flow, Area-Law locality as inductive bias, holographic complexity penalties, test-time adaptation of a tiny modulator, sub-10M parameters — remain sound and are now *more* relevant: ARC Prize 2025's paper award went to TRM, a 7M-parameter recursive model (45% ARC-AGI-1, 8% ARC-AGI-2), and the 2026 competition explicitly rewards efficiency. But three of the ideas need real revision, not just correct implementation: (i) compression-only RG cannot generate outputs that depend on fine (UV) input detail — the decode path needs per-scale boundary data (the holographically correct version of skip connections); (ii) the entropy penalty must act on **states/information flux, not weight spectra**; (iii) continuous-weight Langevin noise does not produce discrete symmetry breaking — rule selection needs a discrete latent if the SSB story is to mean anything.

**Empirical autopsy headlines (details in §2):**

- The trained model is **exactly linear** (superposition holds to 6×10⁻⁷) with **output rank exactly 8**.
- Stage-1 pretraining converged to the **constant "all-black" map**: on a task it was trained on, 97.7% pixel accuracy = exactly the all-black baseline (5119/5120 cells predicted black).
- TTT does not escape it: on eval task `0a2355a6` the tuned model scores **65.1%, below the 66.3% all-black baseline**; on `1e97544e` (target has zero black cells) it scores 13% ≈ chance.
- Sanity floor fails: **identity (copy) = 7.8%** pixel accuracy (below the 20% chance level); color-swap = 23.4% ≈ chance.
- The loss was dominated by padding: **87.7% of the 32×32 canvas is padding** on average (median 90.3%), and padding shares the value 0 with ARC black.
- The "FROZEN: invariant rule extracted" event never fired in any 400-step probe; its trigger tracks clipped-gradient-EMA arithmetic, not task structure, and the "melting" noise was ~20× under-scaled by an ordering bug.

---

## 2. What actually went wrong — empirical autopsy

All probes run against the shipped `core_checkpoint.pkl` (the Stage-1 artifact), JAX 0.10.2, CPU.

### E1 — The model is exactly linear with rank exactly 8

- Superposition test `f(αx₁+βx₂) = αf(x₁)+βf(x₂)`: max relative error **6.2 × 10⁻⁷** (float32 exactness); `f(0) = 0` (no bias parameters exist).
- Stacking logits for 40 random inputs and taking the SVD: singular values `[136.8, 110.9, 97.9, 90.4, 76.2, 63.1, 52.1, 39.7, 0.0, 0.0, …]` — **rank exactly 8**.
- Cause: `embed → encode → decode → logits` is a chain of matrix multiplications, and all information passes through the `(1,1,8)` top latent. The only nonlinearities in the entire system are the softmax inside the loss and the argmax at prediction time.
- Consequence: every logit field the model can ever emit lives in an 8-dimensional subspace of ℝ^10240. A rank-8 linear map can *memorize* up to 8 support pairs, but it cannot express the identity **rule** (a rank-1024 operator) — so it is guaranteed to fail on any unseen test grid whose content isn't spanned by the supports. This is memorization-without-generalization **by construction** — the exact failure mode the Area-Law program was designed to rule out.

### E2 — What Stage-1 pretraining actually learned: the background color

Forward pass only (no TTT) on `007bbfb7` — a task **in the Stage-1 training set**:

| Params | CE | pixel acc | all-black baseline | prediction histogram |
|---|---|---|---|---|
| random init | 2.303 (≈ ln 10) | 12.6% | 97.7% | spread over all 10 colors |
| **stage-1 checkpoint** | **0.149** | **97.7%** | **97.7%** | **{black: 5119, red: 1}** |

The CE improvement 2.30 → 0.15 that looked like learning in the April logs is fully explained by discovering the constant map "everything is black." Accuracy lands **exactly** on the all-black baseline.

### E3 — Behavior with full TTT (warm start from checkpoint, 400 steps, `MAX_TTT_STEPS` regime)

| Task | exact | pixel acc | all-black baseline | note |
|---|---|---|---|---|
| eval `0a2355a6` | ✗ | 65.1% | 66.3% | TTT ends **below** the trivial baseline; output is 250/255 black + 5 stray pixels |
| eval `1e97544e` | ✗ | 13.0% | 0.0% | target contains no black at all; model sprays all 10 colors ≈ chance |
| pretrain `007bbfb7` (seen) | ✗ | 55.6% | 55.6% | still predicts 100% black **after** 400 task-specific TTT steps |
| **IDENTITY (copy 8×8)** | ✗ | **7.8%** | 21.9% | **below the 20% chance level**; cannot copy its input |
| **COLOR-SWAP (1↔2)** | ✗ | 23.4% | 14.1% | ≈ chance |

The identity result is the single most important number in this report: a system marketed as a reasoning architecture cannot reproduce its input after 400 steps of task-specific training on four demonstrations of "output = input." Per E1 this is not an optimization failure — it is representationally guaranteed.

### E4 — The loss is dominated by padding

Over the first 100 public training tasks, **87.7% of the 32×32 canvas is padding on average (median 90.3%)**.

Grids are zero-padded to 32×32 and the cross-entropy is averaged over **all 1024 cells** with no mask; padding value 0 is also ARC's black. For a small task the trivial strategy "predict black everywhere" already achieves ~90%+ pixel accuracy and near-zero CE. The rank-8 linear model finds exactly this solution.

### E5 — "Spontaneous symmetry breaking" is optimizer arithmetic, not physics

Three compounding defects in `src/ttt/langevin.py`:

1. **The trigger measures clipping artifacts.** Cooling fires when `|‖g‖ − EMA(‖g‖)| < 10⁻³`, computed on gradients that already passed `clip_by_global_norm(1.0)`. When raw gradients are large (e.g., from random init), the clipped norm is pinned at exactly 1.0, the EMA (α = 0.9) converges to it geometrically, and the detector fires every step — freezing at a *task-independent* `log(10⁻⁴)/log(0.95) ≈ 180` cooling events after the EMA settles. When gradients are small and noisy (warm start), the detector essentially never fires: **in all five 400-step probes, no task ever froze** (`frozen=False`, steps = 401 everywhere). In neither regime does the freeze event measure anything about rule extraction.
2. **The "melting" noise is ~20× under-scaled.** Thermal noise is added to the updates *before* the `−lr` multiplication, so its effective std is `lr·√(2·lr·T)` ≈ 0.016·√T instead of the SGLD-correct `√(2·lr·T)` ≈ 0.32·√T. High-temperature exploration was cosmetic from the first step.
3. **Nothing discrete exists to break.** Even with correct noise and a correct schedule, annealing continuous weights selects a basin of a loss surface, not one of several symmetry-related rule hypotheses; there is no order parameter and no degeneracy. See §4.2.4.

### E6 — The block-sparse TPU kernel is fake sparsity, and dead code

- `contract_sparse` computes **every** block product (the Pallas grid iterates over all `num_blocks`; the CPU fallback `vmap`s over all blocks) and then multiplies results by the mask. FLOPs are identical to dense blocked matmul, plus masking overhead. The spec's requirement — *skip* zero blocks using scalar-prefetched sparsity maps — is not implemented; `PrefetchScalarGridSpec` is stuffed into `compiler_params` where it configures nothing, and its `num_scalar_prefetch=1` contract (kernel receives a scalar ref) is not honored by the kernel signature.
- The unit test asserts `non_zero_count == 2` — a *bookkeeping counter* (`jnp.sum(mask)`), not a property of the computation. The "MXU Zero-Wastage Check" tests that a number equals itself.
- Most importantly: **nothing in the model calls it.** `grep` shows `contract_sparse` is imported only by its own test. The PEPS "contraction" in the live path is `jnp.dot(B_flat, bond_projector)`.

### E7 — Environment rot

The environment was never pinned (`requirements.txt` contains one real dependency, unversioned). Under current JAX (0.10.2) the repo failed immediately on the removed `jax.lib.xla_bridge` API — fixed in this session with a one-line change to `jax.default_backend()`. This is direct evidence the codebase was built against an older, undocumented environment; nothing recorded which one. Relatedly, every task is fetched from GitHub raw at run time via `urllib` with **no timeout** — the first probe run of this analysis hung indefinitely on exactly that call. The ARC-1 dataset is now vendored locally under `data/ARC-AGI/` (git-ignored), and `file://` URLs work with the existing loader.

---

## 3. Idea → implementation divergence table

| # | Documented idea (source) | What the code actually does | Impact |
|---|---|---|---|
| D1 | **PEPS lattice**: pixels entangled with 4 neighbors through virtual bonds; global state = contraction of the lattice (Spec §2.1; HRRN.pdf 18–20) | `embed_grid_to_peps` never connects neighbors. Each site's 4 virtual legs are flattened and linearly projected, per-pixel, with a shared matrix. There are **no bonds between sites** — the "PEPS" is a 10-row embedding table computed with extra steps | **Fatal.** The Area-Law object the whole theory rests on does not exist; no information propagates through the lattice |
| D2 | **MERA disentanglers act across block seams** to remove inter-block entanglement before coarse-graining (Spec §2.2; Overview 8, 12) | `U` is applied to the *same* 2×2 block that `W` then compresses. `W^T·U^T` composes into a single 32→8 linear map: a **tree tensor network**, not MERA. Adjacent pixels in different blocks first interact 1–5 layers up | **Fatal for the "sort signal from noise before pooling" mechanism**; block-boundary information is destroyed exactly the way the docs warned max-pooling would |
| D3 | **Nonlinear/multilinear expressivity** from tensor contraction (amplitudes multiplicative across sites) | One-hot × shared tensor × linear layers = **exactly linear end-to-end** (probe E1) | **Fatal.** Rank-8 linear hypothesis class |
| D4 | **Holographic loss**: Von Neumann entropy of the *state* across virtual bonds; penalize bond usage (Spec §4.1; Overview 13; HRRN.pdf 33) | `holographic_entropy_proxy` computes SVD-spectrum entropy of **weight matrices** (including the hypernetwork's MLP weights) | **Category error.** Regularizes parameter spectra, not entanglement; no relation to the represented state's complexity |
| D5 | **Langevin annealing → SSB selects the invariant rule** (Spec §4.2) | Plateau detector saturated by gradient clipping → cooling fires every step → freeze at ~step 180 always (E5); noise 20× too small; nothing discrete ever breaks | **Broken as physics and as optimization**; "Frozen (SSB)" telemetry is a timer |
| D6 | **TTT adapts the modulator only; core frozen** (Overview 14: "TTT optimizes θ_task … θ_scale fixed"; Spec §4: freeze structure, optimize U,W) | TTT optimizes **every parameter** (embedding, base weights, hypernetwork, decoder) at lr = 0.05 from the warm start | Destroys the pretrained prior; 78k free params fit to 2–5 examples |
| D7 | **Block-sparse Pallas kernels skip zero blocks via scalar prefetch** (Spec §3.1) | Kernel computes all blocks, masks after; `PrefetchScalarGridSpec` misused; never called by the model (E6) | Dead + fake; TPU cost/perf story unfounded |
| D8 | **Symmetry sectors**: U(1)/ℤ₁₀ color-charge conservation block-diagonalizes tensors (χ_eff 32 at χ=4 cost) (HRRN.pdf 32) | Absent. Dense unstructured tensors; colors are anonymous one-hot slots | Major loss: this was simultaneously the compute story, a capacity story, **and** a color-equivariance prior |
| D9 | **Recursive refinement**: one shared core applied T times with scale modulation; "think broader"; deep supervision (HRRN.pdf 23–25, 29; Overview 11) | Single feed-forward encode/decode pass; no outer recursion, no iterative refinement, no deep supervision | The TRM-inspired mechanism the proposal was named for is absent |
| D10 | **Reversibility as a check**: decode with zeroed noise wires must reconstruct the input; mismatch = leaked information → retrain disentanglers (Overview 15) | No reconstruction consistency term anywhere; loss is x→y CE + weight entropy | The self-diagnostic that would have exposed D1–D3 in April was never wired in |
| D11 | **Interpretability plots**: per-layer tensor heatmaps, bond-dimension complexity metric (HRRN.pdf 28; Overview 6) | `visualize.py` prints ANSI grids of input/prediction/target only | Missing the instrument that makes a "glass box" a glass box |
| D12 | **Evaluation plan**: ARC-1 + ARC-2, TRM/ViT/ResNet baselines, Area-Law scaling validation (HRRN.pdf 35–36; Spec §5) | 10-task eval list generated; benchmark never run; no baselines; no scaling test | The claims were never confronted with data until now |
| D13 | Canvas/size handling (implicit in all docs' "solve the puzzle") | Zero-pad to 32×32, unmasked CE over the full canvas, pad value = black, output cropped to the **ground-truth** test output size | Loss gamed by background (E4); test-time size leak invalidates even the honest numbers |

Also noteworthy: `MeraEngine`/`Disentangler`/`Isometry` classes exist and pass their unitarity tests but are **not used** by the live pipeline (only the functional path in `run_ttt.py` is); `dispatcher.py` provisions **on-demand** TPUs while README/deploy script claim Spot (~2.4× cost difference); `jax.clear_caches()` between tasks forces recompilation per task (~min/task of pure compile on a 400-task run).

---

## 4. Critical evaluation of the ideas themselves

Separating the load-bearing physics from the decorative physics, now that we know what the failure actually was.

### 4.1 What survives (and is worth betting on)

- **Locality / Area-Law as inductive bias.** Still the single best-motivated prior for ARC: rules are overwhelmingly local, object-based, and composable. The 2D-lattice + hierarchical coarse-graining program is right. What failed was the implementation (no lattice coupling at all), not the prior.
- **Reasoning as RG flow.** As an organizing principle — UV pixels → IR rule, with scale-indexed processing — this maps beautifully onto what works empirically elsewhere (hierarchical conv/U-Net structure; TRM's iterated refinement as *temporal* RG). It needs one honest amendment (§4.2.1).
- **Complexity penalty selecting the simplest rule.** Occam-by-construction is the right answer to few-shot generalization. But the penalty must act on the **information the network actually uses**, not on weight spectra (§4.2.2).
- **Tiny parameter count + TTT.** Fully vindicated by the field: TRM (7M params) took ARC Prize 2025's paper award at 45%/8%; the 2026 prize explicitly scores efficiency. Parameter golf is a real, winnable lane, and test-time training is the standard mechanism among small-model leaders.
- **Reversibility as a diagnostic.** The docs' "glass-box check" was prophetic — it is precisely the test that would have caught the collapse. Promote it from idea to permanent CI gate.

### 4.2 What does not hold as formulated

**4.2.1 Compression-only RG contradicts the task family.** A strict funnel (32×32 → 8 floats → 32×32) assumes the answer is a function of the IR fixed point alone. But a large class of ARC tasks — identity, local recoloring, texture-preserving edits — require **UV data to survive into the output**. In RG language: the rule is often *not* an IR observable; it's a relevant operator acting on microstate details. The holographically honest fix is to keep the **boundary data at every radial slice**: each scale s retains a (regularized) residual stream that the decoder consumes at the matching scale. That is exactly a U-Net skip topology, but with a physics-native accounting: the information flux I_s crossing each RG cut is measured and *priced* (§5.2). Reconstruction of UV detail is then possible but costs flux, so the optimizer uses fine-scale data only where the rule genuinely needs it.

**4.2.2 Weight-spectrum entropy is not holographic entropy.** Entanglement entropy is a property of a **state across a cut**, not of parameter matrices. The correct implementable analog: per-scale activations (the data crossing each cut) get an information bottleneck — e.g., KL(q(z_s|x) ‖ N(0,1)) per stream, or discrete entropy if streams are vector-quantized. Total loss `CE + β·Σ_s I_s` makes "complexity of the inferred rule" = total boundary information flux through the RG surfaces — an RT-flavored Occam's razor that is differentiable, per-task measurable, and yields an interpretable **complexity spectrum** {I_s} (which scales a task's rule lives at). This is both better physics and a better regularizer than D4.

**4.2.3 A tree is not MERA, and the difference is the whole point.** Without seam disentanglers, entanglement across cuts that don't align with the tree is simply destroyed — the docs' own critique of max-pooling, reproduced faithfully by the implementation. Real MERA staggers unitary "sorters" across block boundaries before each isometry. If we keep any tensor-network claim, the seam operators are non-negotiable. (They also have a cheap neural relaxation: alternating shifted local mixing, à la Swin's shifted windows — a fact worth exploiting rather than resenting.)

**4.2.4 Continuous Langevin noise does not produce discrete symmetry breaking.** Annealing weights in ℝⁿ selects a basin, not a *rule*; there is no order parameter, no degenerate vacua, nothing to break. If rule selection is to be a phase transition, the latent must be **discrete** (e.g., a small codebook of rule tokens / VQ latent, annealed via temperature on the categorical distribution — Gumbel-softmax τ→0 is literally an annealed symmetry-breaking scheme). Alternatively, drop the metaphor and use a plain well-tuned optimizer. The middle ground (SGLD on weights with a *correct* noise scale and a *fixed* schedule) is defensible but should be treated as an ablation, not a pillar.

**4.2.5 The symmetry story was underexploited — it is the parameter-golf engine.** ARC's rule distribution is (approximately) equivariant under D₄ spatial symmetry, translations, and **color permutations** (S₉ on non-background colors, with 0 special). The docs gestured at U(1)/ℤ₁₀ sectors for *compute*; the deeper win is **equivariance as gauge symmetry of the hypothesis class**: process color as a *set* dimension (attention/DeepSets over the color axis) and share weights over D₄ orbits. Each imposed symmetry divides parameters and sample complexity together — this, plus scale-shared cores with low-rank modulators, is how a <1M-parameter model gets TRM-class capacity. Conserved charges (per-color pixel counts, object counts, connectivity invariants) become Noether-style readouts usable as consistency checks during TTT.

**4.2.6 TTT-everything on 2–5 examples is statistically unserious.** The docs' own instinct (freeze core, adapt a small modulator) was correct; quantified: adapting ~10³ low-rank parameters against 2–5 pairs (with group augmentation multiplying effective support ×8–×50) is a sane ratio; adapting 78k dense parameters is not. Add leave-one-out validation over support pairs for early stopping and candidate ranking (pass@2 selection needs a score).

**4.2.7 Missing theory of the canvas.** Padding-as-black, unmasked loss, and ground-truth-size cropping are not details — they decide what the optimizer optimizes (E4). The theory documents never address output-size prediction; the redesign must (mask + size head or size-canonicalization; padding as a distinct 11th "void" state, which incidentally is the correct "vacuum" the physics language kept invoking).

---

## 5. Redesign: QHRRN-2 ("holographic RG network")

### 5.1 Design principles

1. Every physics claim must correspond to a **measured quantity or an ablatable mechanism**. No decorative physics.
2. Symmetry first: equivariance under D₄ × translations × color-permutations is the primary capacity-compression device (the parameter-golf engine).
3. Holography as architecture: scale-indexed residual streams ("boundary data per radial slice") with **priced information flux** (β·Σ I_s) replacing both the missing skip paths and the miscast entropy penalty.
4. Recursion as radial evolution: a single shared core applied iteratively (TRM-style), modulated by scale embedding — depth without parameters.
5. TTT adapts **only** low-rank modulators, with group augmentation and leave-one-out scoring.
6. Discrete rule latent if (and only if) the SSB/annealing story is kept.

### 5.2 Architecture sketch

- **Input:** grid → 11-state (10 colors + void) → color-equivariant set embedding (per-cell features = functions of "which color class", not "which color id").
- **Core cell (shared across scales, ~10⁴–10⁵ params):** shifted local mixing (seam op) → 2×2 coarse-grain isometry → per-scale FiLM/LoRA modulation from a scale-conditioned hypernetwork (the existing θ(s)=θ_base+Δ(s) idea, kept). Optional Cayley-orthogonal parameterization retained *only* where it demonstrably stabilizes TTT (ablation).
- **Streams:** at each scale, a slim residual stream z_s (the boundary slice) with an information bottleneck; decoder consumes z_s at matching scale during expansion.
- **Rule latent:** small discrete codebook at the IR point (e.g., 64–256 codes, rank driven by Σ I_s pressure); annealed categorical selection during TTT = the *actual* symmetry-breaking mechanism.
- **Decoder:** mirror expansion with seam ops; **canvas head** predicts (H_out, W_out) or a size-transform class; loss masked to the true canvas.
- **Objective:** masked CE + β·Σ_s I_s + reconstruction-consistency term (the glass-box check, now differentiable and always-on) + charge-consistency auxiliaries (optional).
- **Budget:** target **≤1M parameters** (stretch: ≤250k). Reference frontier: TRM at 7M / 45% ARC-1 / 8% ARC-2.

### 5.3 Training & TTT protocol

- **Pretrain** the core + hypernetwork on procedurally generated tasks (RE-ARC-style generators over the public training distribution; optionally BARC-class synthetic corpora) — thousands of tasks, not 20×5 epochs.
- **TTT per task:** freeze core; adapt low-rank modulators + rule-latent logits; augment support with the D₄×palette orbit; leave-one-out early stopping; two candidates by score → pass@2.
- **Optimizer:** AdamW baseline; SGLD-with-correct-noise as an ablation only.

### 5.4 What to keep from the current repo

Keep: the dispatcher's provision→run→rescue→**unconditional teardown** discipline (add `--spot`), the checkpoint I/O, the ANSI visualizer, `cayley.py` as a utility, the task-list tooling in `data_loader.py`, and the (now-fixed) test harness bones. Rewrite: the model (`peps.py`, the functional MERA in `run_ttt.py`, `loss.py`, `langevin.py`) and the Pallas kernel (only when a real contraction/blocked-equivariant op needs it — write it against a measured bottleneck, not speculatively).

---

## 6. Process lessons (why this failed as a project, not just as a model)

1. **No acceptance gate.** A 5-minute identity-task test would have exposed the rank-8 collapse in April. The sanity triad (identity, color-swap, translation, each solved to 100% by TTT) becomes the permanent gate before any cloud spend.
2. **Tests asserted bookkeeping, not behavior** (the MXU counter test; unitarity tested on classes the model doesn't use). Tests must exercise the live path end-to-end.
3. **Narrative outran measurement.** README/comments claim in superlatives what was never run once. All claims now trace to a metric in a results file, or they don't ship.
4. **Unpinned environment.** Freeze `requirements.txt` (exact versions), vendor the ARC dataset locally (no per-run GitHub fetches; also removes a TPU-side failure mode), record `jax.__version__` in every checkpoint.
5. **Agent-generated code needs adversarial review by default** — the failure pattern here (elaborate scaffolding, disconnected core, emphatic comments) is exactly what "looks done" without being done.

---

## 7. Phased roadmap — targeting the October submission

Working deadline: **AAMAS 2027** (Hanoi, May 3–7, 2027) — submissions tentatively **October 2026** (historically abstracts ≈ Oct 1, full papers ≈ Oct 8; official dates TBC). AAAI-27 main-track deadlines fall in July–August, so AAMAS is the consistent reading of "October" — confirm with the Prof. Plan assumes **results frozen September 28, 2026**.

| Phase | Window | Work | Gate to advance | Compute |
|---|---|---|---|---|
| 0 | Jul 17–27 | Harness: pinned env, vendored ARC data (done), pytest, masked-loss eval runner, metrics CSV | Repo runs end-to-end deterministically on CPU | Mac, $0 |
| 1 | Jul 28–Aug 10 | QHRRN-2 core (§5.2) at toy scale; sanity triad via TTT | **3/3 sanity tasks exact-match** | Mac, $0 |
| 2 | Aug 11–31 | Dev-30 subset (stratified public training tasks); ablations: seam ops, flux penalty β, equivariance on/off, discrete latent on/off | ≥15/30 with ablation table justifying each mechanism | Mac / 1 short v5e-1 spot run |
| 3 | Sep 1–14 | RE-ARC-style pretraining at target param count; TTT protocol tuning (augmentation orbit, LoO) | Beats Phase-2 score with frozen-core TTT | v5e-1/-8 spot, ~$100–300 |
| 4 | Sep 15–28 | Full ARC-1 public eval + ARC-2 attempt; params-vs-accuracy frontier vs TRM; flux-spectrum analysis; **results freeze Sep 28** | Camera-ready numbers + ablations | v5e spot, ~$100–200 |
| 5 | Aug → Oct 8 | Paper track in parallel: method/theory sections from this report (Aug); experiments written as they land (Sep); full draft Sep 21; polish after freeze; abstract ≈ Oct 1, full ≈ Oct 8 | Submitted | — |

Budget outlook: well inside the ~$2000 research credits; Phases 0–2 are essentially free. The deadline, not the budget, is the binding constraint: gates are go/no-go on calendar dates — a mechanism that misses its gate ships as an ablation finding rather than blocking the pipeline, and the paper's headline switches to the strongest result that did land.

---

## 8. Reference points (July 2026 landscape)

- ARC-AGI-1 is saturated by frontier LLMs (>90%, at costs of dollars-per-task); the open frontier for small/efficient systems is the params-and-cost frontier, plus ARC-AGI-2 (2025 Kaggle top ≈ 24%).
- TRM ("Less is More: Recursive Reasoning with Tiny Networks", arXiv:2510.04871): 7M params, 45% ARC-1, 8% ARC-2 — ARC Prize 2025 Paper Award; the direct comparison target for parameter golf.
- ARC Prize 2026 emphasizes efficiency; ARC-AGI-3 (interactive) exists but is out of scope for this project.
