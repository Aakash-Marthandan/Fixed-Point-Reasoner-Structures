# QHRRN-2 Design Ledger — Hypotheses vs. Proven Ground

**Purpose:** the single source of epistemic truth for this project. Every design element carries a status; statuses change only with dated, linked evidence. This is the discipline April lacked: claims outran measurement, and nothing recorded which was which.

**Created:** 2026-07-19 · **Maintainers:** PI + Claude · **Companions:** `Divergence_Analysis_2026-07.md` (evidence for all REFUTED entries), `QHRRN2_Architecture.md` v0.2 (the design the ledger governs)

---

## 0. Rules

**Status vocabulary:**

| Tag | Meaning | What it takes to claim it |
|---|---|---|
| **[P-C]** | Proven — constructive/mathematical | A proof or explicit construction in our setting (e.g., "identity is in the weight space", with the construction written down) |
| **[P-L]** | Proven — literature | A replicated, load-bearing published result we are *reusing, not extending* (cited) |
| **[P-M]** | Proven — measured by us | Our own probe/CI/experiment, with the artifact (script + date + number) linked |
| **[H-n]** | Hypothesis | An assumption we are betting on. **Must name its test** (CI gate or ablation row) and its falsification condition |
| **[R-n]** | Refuted | Contradicted by measurement or proof (April post-mortem or later) |
| **[S-n]** | Superseded | Dropped by design decision, not disproof — the reason recorded |

**Discipline (binding on all future work):**
1. No code implements an [H] without its named test existing in the same change.
2. Status changes are **append-only** in §5 with date + evidence link. Nobody edits history.
3. Every module header cites the ledger IDs it implements (e.g., `# Ledger: H3, P-C2`).
4. A failed [H] is a *reported negative result* in the paper, not a silent deletion.
5. Solve-rate claims live only in §3 (hypotheses) until eval day. Nothing in a README asserts performance.

---

## 1. Provenance — every load-bearing claim from the four source documents

### 1.1 *Renormalization of Thought* (Dec 8, 2025 — the theory notes)

| Claim (as stated) | Status now | Disposition in QHRRN-2 |
|---|---|---|
| Grid ≅ 2D lattice; pixels ≅ Potts fields ℤ_q | **[P-C]** trivially (a representation choice) | Kept: canvas + categorical fields |
| Rule ≅ Hamiltonian / transfer matrix | **[H-1]** (a modeling metaphor — useful, untestable as stated; operationalized via H-4/H-5) | Informs, doesn't constrain |
| Reasoning ≅ RG flow UV→IR to a stable fixed point; "applying the transformation again yields an invariant state" | **[H-2]**, sharpened: solved task = fixed point of the iterate map Y_{t+1}=F(Y_t). Test: convergence of recursion iterates on solved vs unsolved tasks (do solved tasks reach Y_{t+1}=Y_t?) | Kept as the recursion-with-deep-supervision design; fixed-point convergence becomes a *measured diagnostic* |
| Proposition 1: "All valid ARC objects are Area-Law states" (solid shapes carry information only on boundaries) | **[H-3]** — plausible for solid shapes; *false in general* (texture/noise-carrying tasks exist; identity requires full UV transmission — proven by the April E3 analysis + rank argument). Weakened form kept: *fine-scale locality is the right prior; information demand varies per task* | Replaced by the priced-stream mechanism: Area-Law is the **default the optimizer relaxes for a fee**, not an axiom |
| Disentangler Û acts **on the boundary between two blocks**, turning entanglement into local information | **[P-L]** as an architecture pattern (MERA staggering; Swin shifted windows) — note: **April violated this exact sentence** (same-block U∘W, [R-2]) | Seam mixers, offset by (1,1) — the spec's own wiring, finally |
| Isometry Ŵ compresses blocks to conceptual tokens | **[P-L]** (coarse-graining works: all conv pyramids) — but *lossy-funnel sufficiency* is **[R-1]** | Kept as the *kept-channel*; complement goes to streams, not to /dev/null |
| Holographic loss L = −log P(Y\|X) + λ Σ_l \|γ_l\|·ln χ_l (RT: entropy = bonds cut × ln χ; G_N ∝ 1/χ) | Original form = **capacity** penalty on the architecture. **[S-1]**: superseded by the **usage** penalty (flux ledger I_s = measured nats crossing cut s, VIB bound) — the tighter MDL reading: capacity bounds usage, usage is what memorization actually spends. April implemented *neither* (weight-SVD entropy, [R-3]) | **[H-4]** (the central bet): pricing information flux across RG cuts improves few-shot generalization vs β=0. Tests: β-sweep, streams-off, flux-spectrum sanity (CI-5) |
| "Memorized solution = massive wormhole throat (high χ); reasoned = narrow throat" | Same as above — the qualitative prediction of **[H-4]**; directly measurable now: flux(identity) ≫ flux(constant-fill) is CI gate 5 | The tollgate design thesis |
| TTT = simulated annealing on free energy; P(z) ∝ e^{−βH_S(z)}; correct Langevin dz = −∇H dt + √(2T) dW | Annealing-as-frame: **[H-5]**. Note the doc's SGLD noise is the *correct* scale — April's implementation was ~20× off its own theory ([R-4]) | SGLD (correct scale) kept as an **ablation row only**; primary optimizer AdamW + LoO early stop |
| Melting → cooling → phase transition; "Translation Symmetry wins over Rotational Symmetry" (rule selection = SSB among candidate symmetries) | **[H-6]**: SSB is meaningful only with a discrete order parameter. Test: τ-annealed codebook; H[q] collapse correlates with correctness; ablation discrete-vs-continuous rule code | The rule codebook. April's continuous-weight version is [R-5] |
| "Do not use PyTorch Linear; use a TN library" | **[S-2]**: superseded. The failure mode wasn't the library, it was unfalsified structure. We use JAX with *structural constraints + tests* instead of TN-library orthodoxy | — |
| Adaptive capacity: "if error is high, increase χ slightly" | **[H-7]** (nice idea, unimplemented in April): per-task capacity escalation (raise T, widen gates) when LoO fails. Test: escalation policy vs fixed, Phase 3 | Deferred to Phase 3; logged so it isn't lost |

### 1.2 *HRRN.pdf* (Dec 2025 proposal deck)

| Claim | Status | Disposition |
|---|---|---|
| TRM insight: parameter efficiency > depth; recursion with shared weights; deep supervision prevents drift | **[P-L]** — TRM: 7M params, 45% ARC-1 / 8% ARC-2 (arXiv:2510.04871, ARC Prize 2025 paper award); HRM before it | Kept intact — this is the proven engine the solve-rate bet rides on |
| Modulated recursion θ(s) = θ_base + Δ(s) via tiny hypernetwork (capability fixed, attention/modulation varies) | **[P-L]** for FiLM/LoRA-style modulation generally; scale-conditioning specifically = **[H-8]** (test: ablate θ(s,t) modulation vs plain shared core) | Kept: LoRA-rank-4 modulation by (s,t) |
| "Thinking broader" (spatial recursion) parallels "thinking deeper" (temporal) | **[H-2]** again (same family) | Recursion = radial re-flow |
| Bond dimension χ as complexity/interpretability metric | Reborn as flux spectrum {I_s} — **[H-9]**: {I_s} clusters align with human task-family labels. Test: Phase-4 clustering figure | The paper's interpretability claim |
| Block-sparse ℤ₁₀/U(1) color symmetry sectors (χ_eff via block-diagonal) | The *compute* claim is **[S-3]** (XLA handles our shapes; no kernel until profiled). The *symmetry* insight is **[P-C1]**: exact S₉ color equivariance implemented as weight sharing — the honest descendant | Color-set axis |
| χ ≤ 32 keeps contraction tractable | Moot in current design ([S-3]); capacity dial is d | — |
| Interpretability: latents are 2D grids, visualizable | **[P-C]** trivially (states are spatial by construction) + gate maps/flux/H[q] as first-class outputs | Tooling requirement, Phase 2 |

### 1.3 *HRRN_Overview* (Jan 2026 deck)

| Claim | Status | Disposition |
|---|---|---|
| "TTT optimizes θ_task (context); θ_scale fixed after pretraining" | **[P-L]** as protocol family (TTT-for-ARC lineage: MindsAI, Akyürek et al.; LoRA-style adapters standard). April violated it ([R-6]: all 78k params adapted) | Frozen bulk + ~10–25k boundary params |
| Reversibility "glass-box check": decode with zeroed noise wires must reconstruct input; mismatch = leaked information | The *diagnostic instinct* is vindicated (**[P-M]**: exactly such a check would have caught April in five minutes — see post-mortem E1/E3). Exact-unitarity *requirement* is **[S-4]**: conservation now = kept ⊕ streamed, softly, with the flux ledger as the leak meter | CI gates 2–3 are its descendants |
| Two orthogonal bottlenecks: spatial (isometry) + logical (χ) | Survives as: spatial pyramid + priced streams ([H-4]) | — |
| Causal cone / "teleportation is local at depth log L" | **[P-C]** for the pyramid's receptive-field arithmetic; *sufficiency* for correspondence tasks is **[R-7]**-adjacent (local hierarchies are weak at correspondence — literature + our audit) → Amendment B | Coarse-scale attention at IR (≤64 tokens) |

### 1.4 *QHRR Master Specification* (Apr 2026 — the Antigravity directive)

| Claim | Status | Disposition |
|---|---|---|
| "QHRR does not rely on massive pre-training; adapts per puzzle" | Half-refuted: pure TTT-from-scratch fails (April, all probes); pretrained-prior + light TTT is the working recipe (**[P-L]**, TRM/TTT lineage) | Episodic generator-scale pretraining + boundary TTT |
| <10M parameter constraint | Kept and sharpened: **[H-10]** ≤400k suffices for ≥30% ARC-1 (see §3) | The parameter-golf headline |
| Pallas block-sparse kernels with scalar prefetch are necessary for budget | **[R-8]**: the shipped kernel was fake sparsity + dead code (post-mortem E7); necessity claim refuted at our scales | Revisit only on a profiled bottleneck |
| Staged deploy / teardown discipline / checkpoint rescue | **[P-M]** (the one April subsystem that worked as designed) | Kept; add `--spot` |

---

## 2. QHRRN-2 component ledger

| # | Component | Status | Evidence / Test that moves it |
|---|---|---|---|
| C1 | Canvas 32×32, VOID as real state, masked CE, canvas head (no GT-size at predict) | **[P-M]** that the alternative fails (E4: 87.7% padding domination; D13 size leak); correctness of the fix = construction + CI-6 | CI-6 |
| C2 | Exact S₉ color equivariance (colors 1–9 set axis; black, VOID distinguished) | **[P-C1]** exactness by construction (weight sharing); **benefit** = [P-L] (G-CNN literature: equivariance ≥ augmentation at fewer params) + **[H-11]** for ARC specifically (ablation: color-set vs plain embedding) | CI-1 (bit-exact permutation test); ablation row 4 |
| C3 | Per-color TTT bias vectors (color symmetry broken by evidence) | **[P-C2]**: strict equivariance provably cannot represent color-constant rules (permutation contradiction); biases restore representability. Learnability = part of [H-12] | CI-3 includes a color-constant task |
| C4 | Seam mixers, offset (1,1), GELU residual | **[P-L]** (Swin/shifted-window lineage; MERA staggering) + faithful to the Dec-2025 spec's own sentence | CI-4 (seam task) |
| C5 | Kept ⊕ streamed split; priced streams; rule-conditioned gates | **[H-4]** — *the central novel bet.* Falsified if: β>0 never beats β=0 on dev-30, or streams-off matches full model | Ablations 1–2; CI-5 |
| C6 | Coarse-scale attention (s≥3, ≤64 tokens) | **[P-L]** attention does correspondence; IR-only placement = **[H-13]** (test: move/remove attention scales) | Ablation row 8 |
| C7 | Axial row/col summaries | **[P-C3]**: compaction = prefix count (explicit construction, task `1e0a9b12`) | Ablation row 9 |
| C8 | Rule codebook K=128, M=2, τ-annealed; H[q] order parameter | **[H-6]** (discrete beats continuous rule code); VQ hygiene = [P-L] | Ablation row 5; H[q]-vs-correctness plot |
| C9 | Recursion T≈6, deep supervision, Y_{t−1} feedback | **[P-L]** (TRM/HRM) | Ablation row 6 (T=1 vs 6) |
| C10 | TTT: frozen bulk, LoRA-rank-4 + gates + rule queries + color biases (~10–25k), D₄×palette orbit, LoO early stop, pass@2 | Protocol = **[P-L]** (TTT-for-ARC + LoRA); our exact budget = **[H-12]** (≤25k adapted params suffice; test: LoRA-TTT vs full-FT ablation) | Ablation row 7 |
| C11 | Vectorized population TTT (vmap ~64 tasks) | Engineering claim, verifiable by wall-clock measurement | Phase-3 benchmark |
| C12 | Episodic pretraining on RE-ARC-style generators | **[P-L]** that generator pretraining transfers (TRM/BARC lineage); episodic-vs-supervised = open question #3 for PI | Phase-3 A/B if budgeted |
| C13 | Nonlinearity/rank floor (GELU everywhere; no linear collapse) | **[P-C]** by construction; guarded forever by CI-2 (rank > 1000, superposition must fail) | CI-2 |

**CI gates (all local, all pre-cloud):** CI-1 equivariance bit-exact · CI-2 anti-linearity/rank · CI-3 sanity triad + color-constant task via TTT · CI-4 seam-boundary task · CI-5 flux-direction sanity · CI-6 canvas/no-GT-size. Each maps to ledger IDs above.

---

## 3. The hypothesis register (what we are actually betting, ranked by risk × centrality)

| ID | Hypothesis | Test | Falsified if | Decision date |
|---|---|---|---|---|
| **H-4** | Pricing information flux across RG cuts (β>0) improves few-shot generalization over free streams (β=0) | β-sweep on dev-30; streams-off control | β=0 ≥ all β>0 across families | Aug 31 (dev-30 gate) |
| **H-10** | ≤400k params reach ≥30% ARC-1 public eval (params–accuracy frontier point) | Phase-4 full eval | <20% at d=32 after Phase-3 tuning | Sep 28 (freeze) |
| **H-12** | ≤25k adapted boundary params + orbit augmentation suffice at TTT (frozen bulk) | LoRA-TTT vs full-FT | full-FT beats by >5 pts | Phase 3 |
| **H-6** | Discrete τ-annealed rule selection ≥ continuous rule vector; H[q] collapse tracks correctness | Ablation 5 + correlation plot | no gain and no correlation | Phase 3 |
| **H-11** | Exact color equivariance beats learned color embedding at equal params | Ablation 4 | embedding wins clearly | Aug 31 |
| **H-9** | Flux spectra {I_s} cluster into human-recognizable task families | Phase-4 clustering | spectra are noise | Sep (paper figure) |
| **H-13** | IR-only attention suffices for correspondence families | dev-30 symmetry/copy families | those families need fine-scale attention | Aug 31 |
| **H-2** | Solved tasks behave as fixed points of the iterate map (Y_{t+1}=Y_t at convergence) | convergence diagnostic on solved vs unsolved | no relationship | Phase 3–4 (diagnostic, not gate) |
| **H-8** | (s,t)-modulation of the shared core beats unmodulated sharing | ablation | no gain | Phase 3 |
| **H-5/H-7** | SGLD-correct annealing helps; adaptive capacity escalation helps | ablation rows (stretch) | no gain | Phase 3 stretch |

**Explicitly not hypotheses (settled):** that a rank-8 linear funnel can't reason [R-1, P-M]; that weight-spectrum entropy is not holographic complexity [R-3, P-C]; that identity requires UV transmission [P-C, post-mortem §4.2.1]; that unmasked padded CE is gamed by background [P-M, E4].

## 4. Refuted & superseded register (carried from the post-mortem, IDs stable)

[R-1] lossy funnel sufficiency (E1/E3) · [R-2] same-block disentangler = MERA (D2) · [R-3] weight-SVD entropy as holographic loss (D4) · [R-4] April's Langevin noise/trigger implementation (E5/E6) · [R-5] continuous-weight SSB (E6 + no-order-parameter argument) · [R-6] TTT-everything (D6) · [R-7] local-only hierarchies handle correspondence (expressivity audit) · [R-8] speculative sparse kernels (E7). Superseded: [S-1] capacity→usage penalty · [S-2] TN-library mandate · [S-3] symmetry *sectors* → symmetry *equivariance* · [S-4] exact unitarity → conservation-with-priced-leak.

## 5. Status-change log (append-only)

- **2026-07-16** — [R-1..R-8] established by measurement; see `Divergence_Analysis_2026-07.md` (probes: `probe_linearity.py`, `probe_ttt.py`, checkpoint @ `d874427`).
- **2026-07-18** — Spec v0.1 drafted; C1–C13 initial statuses assigned.
- **2026-07-19** — Expressivity audit: [P-C2] established (strict S₁₀ equivariance excludes color-constant rules → Amendment A); [R-7] recorded → Amendments B, C; H-13 opened. Spec bumped to v0.2.
- **2026-07-19** — Ledger created; repo reset to clean slate (old implementation preserved in git history @ `d874427`; post-mortem snapshot committed).
- **2026-07-20** — Refinement: CI-3 split into **CI-3a** (Phase-1 *trainability* gate: triad solved by full-parameter fit at toy scale — tests expressivity + optimizability) and **CI-3b** (Phase-3 gate: triad under the frozen-core TTT protocol, which only makes sense once a pretrained bulk exists). Rationale: a random frozen bulk with ~20k adapters cannot be expected to solve anything; conflating the two would make the Phase-1 gate untestable.
- **2026-07-20** — [P-M] Core model implemented (`src/qhrrn2/`): **CI-2 passing** — superposition fails (rel. residual > 1e-2) and output rank ≥ 38/40 on random probes (April E1 inverted into a permanent regression test); **full-model S9 equivariance exact at init** (max err < 1e-4) and `color_bias` demonstrably able to break it (Amendment A mechanism verified); param count **45,653 at d=12** (153× under TRM).
- **2026-07-20** — CI-3a **first run: 0/3 FAIL** (honest record). Signature: support loss → 0.0000 on all tasks, query pixel-acc identity 98.4% / color-swap 82.8% / translate-right 45.3% ⇒ memorization-not-rule under a naive fit protocol (no LoO selection, τ trained 1.0 but predicted 0.05, half the D₄ orbit, no placement offsets, no weight decay). Expressivity is NOT indicted (support fit is perfect; identity misses by ~1 cell); the *fit protocol* is. Fixes: LoO-validated selection (the C10 mechanism, now pulled into the gate), τ-consistent prediction, full orbit + placement-offset augmentation, AdamW. Re-run recorded below. Translate-right flagged as the watch-item: if it stays low after protocol fixes, that is a real block-alignment bias signal (would revisit fine-scale shift handling).

## 6. What this buys the paper

The ledger *is* the experimental section's skeleton: §3's rows become the ablation table; [P-M] entries become the motivation section's evidence; the append-only log documents that hypotheses were registered **before** results existed — pre-registration discipline that reviewers reward and that April's process made impossible.
