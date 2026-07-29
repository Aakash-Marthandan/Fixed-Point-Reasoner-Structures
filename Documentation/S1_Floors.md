# S1 Analytic Floors & the State of the Sandwich — 2026-07-29

**Purpose:** the lower slice of the S1 sandwich (`Thesis_Information_Holography.md` §2), computed for the five constructed families; comparison against the measured envelopes (ledger 2026-07-29 correction); a candid account of what is and is not rigorously boundable; and the concrete program that follows. **Status:** desk analysis, no new compute. Numbers use the sweep-1/1b data (d=12, from-scratch regime, 8×8 content, palette {0..4} uniform).

---

## 1. What our instrument measures, precisely

`StepOutput.flux/flux_attn` sum the KL of every **priced** channel on the final recursion pass of one query: per-scale stream codes (bypass across scale cuts) and per-scale attention messages (nonlocal channel). Three definitional facts that any floor comparison must respect:

1. **The kept spine is unpriced and continuous.** Information can flow UV→IR→decoder through the kept channel without appearing in the ledger. (Thesis §1 caveat; arXiv:2505.24668's skip-path warning.)
2. **The Y_{t−1} feedback does not bypass the cuts** — it re-enters as input fields and crosses the same priced channels; per-pass flux at convergence measures the cost of re-deriving (or re-transporting) the output each pass. We report the final pass, uniformly.
3. **Emission-side pricing upper-bounds crossings** (data processing through the convex attention mix), so measured totals are certified *upper* estimates of what the solver actually moved. Good for upper bounds; irrelevant to lower bounds.

## 2. The rigor boundary: scale cuts vs spatial cuts

**Scale cuts (what we currently measure): no unconditional floor exists.** Because the kept spine is a continuous channel of unbounded formal capacity, an exact solver could in principle route *all* content through it, paying zero ledger. Any scale-cut floor is therefore **conditional on a kept-capacity model** — an assumption, or better, a measurement (see §6, the IR-only probe). This must be stated in the paper; pretending otherwise would not survive review.

**Spatial cuts: floors are rigorous and computable.** For a bipartition of the grid, the induced two-party function has a distributional information-complexity lower bound by the standard data-processing argument — no capacity assumptions, exactly the classical machinery the thesis cites. The catch: **our current ledger does not resolve spatial cuts.** The attention flux is decomposable by source→target side (maskable post-hoc from the pattern and message KLs), and stream/mixer transport is spatially local (crosses a spatial cut only in the seam neighborhood), so a spatial-cut ledger is buildable from the existing mechanisms. **This is the missing instrument, and it is where S1's theorem-grade half actually lives.**

## 3. Analytic reference scales and floors, per family

Content model: X uniform over {0..4}^64 (make_task), H(X) = 64·ln 5 ≈ **103.0 nats**.

| family | output content H(Y \| rule params) | rule params H | rigorous spatial IC (vertical mid-cut) | measured envelope (scale cuts) |
|---|---|---|---|---|
| identity | 103.0 (Y = X) | 0 | **0** (each side emits its own half — identity is spatially diagonal) | 5,617 |
| colorswap | 103.0 (bijective relabel) | ≈ ln 72 ≈ 4.3 | **0** (per-side relabel) | 12,438 |
| translate-right | ≈ 56·ln 5 ≈ 90.1 (one column exits, one enters deterministically) | ≈ ln 30 (shift id) | ≈ **8·ln 5 ≈ 12.9** (one column crosses) | 13,247 |
| constfill | ≈ 0 (+ extent) | ≈ ln 9 ≈ 2.2 | **0** | 326 |
| checkerboard | ≈ 0 given params (pattern is determined) | ≈ ln 56 ≈ **4.0** (ordered color pair; phase fixed by colors-at-parity) | O(boundary) ≈ a few nats (completion needs cross-cut neighbors only near the cut) | 6,237 |

Three observations with teeth:

**(a) The measured envelopes do not track the analytic scales — and the mismatch is a finding.** Checkerboard's output is ~4 nats of parameters, yet its envelope (6,237) sits *above* identity's (5,617), whose content is 103 nats. SGD found the *transport* solution (move the pattern through the cuts) rather than the *computation* solution (infer 4 nats of parameters at the IR and repaint). **The frontier measures learned solvers, not information floors; its gap to the floor decomposes into coding overhead × redundancy × optimization gap.** This reframes S1's empirical section: the sandwich width is itself a measurable object with structure, not an embarrassment.

**(b) Coding overhead currently dominates the ledger.** Identity's fine-scale stream flux I₀ = 3,293 nats spreads over ~256 sites × 11 fields × 6 dims ≈ 16.9k dimensions → **~0.2 nats/dim** — the channels pay per-dimension "rent" far exceeding content. Levers with predicted large effects: free-bits floors, per-dimension gating/sparsity, longer priced phases, β_s per scale. A ~10–50× envelope compression is plausibly available *without touching the floors* — this is the路 to closing the sandwich from above.

**(c) Identity's spatial IC is zero — so its entire cost is scale-structural.** The families separate cleanly in *which* cut family charges them: identity/colorswap are spatial-free but scale-heavy (UV transport); translate pays a rigorous, computable spatial toll (one column); checkerboard pays O(boundary) spatially — literal area law. **The per-family (spatial, scale) cost signature is a sharper taxonomy than either alone and directly feeds S3's locality classes.**

## 4. What S1 can honestly claim, restated

- **Rigorous half:** spatial-cut floors (IC / counting arguments, as above) vs a *spatial* ledger to be built — this is the theorem-anchored sandwich, and translate/checkerboard give nonzero, hand-computable floors.
- **Conditional half:** scale-cut envelopes vs reference scales under a stated kept-capacity model, with the IR-probe measurement (§6) converting the assumption into data.
- **The gap-structure result:** envelope − floor decomposed (coding rent, field redundancy, optimization gap), with the checkerboard transport-vs-computation inversion as the flagship example.
- Kill condition unchanged but now properly aimed: the sandwich must *visibly close* under the levers in (b) on at least the cheap families (constfill is already within ~300 nats of its ~2-nat reference — the closest sandwich we own).

## 5. Corrections this analysis forces

1. Ledger F1 already corrected to envelope estimates (2026-07-29 entry). This document supersedes any content-entropy-only reading of the ordering.
2. The phrase "flux floor" is reserved for analytic lower bounds henceforth; measured quantities are "envelopes" (upper estimates). The spec/thesis text should be swept for this usage before submission.

## 6. New experiments this analysis demands (queued, cheap)

1. **IR-only probe** (kept-capacity measurement): decode with all priced channels zeroed (gates closed, b=0, messages=0) and measure reconstruction accuracy per family — quantifies unpriced-spine leakage; turns the §2 caveat into a number. ~Zero new code (a forward-pass flag).
2. **Spatial-cut ledger**: decompose A_s by source→target side of a bipartition; report seam-local stream crossings — the rigorous-floor comparison instrument. Modest code.
3. **Envelope-tightening sweep**: free-bits + per-scale β on identity/constfill — tests the coding-rent hypothesis directly (predict ≥10× envelope drop at equal accuracy).
4. Stability + knee data (running today) feeds the same analysis unchanged.

## 7. S3 decomposition semantics (fixed BEFORE the stability data arrives)

**Definition (binding for the stability analysis):** for a task at its accuracy-qualified priced optimum, **I_local ≡ Σ_s I_s** (stream/bypass ledger) and **A_nonlocal ≡ Σ_s A_s** (attention-message ledger), both on the final recursion pass, MDL-selected params, deterministic decode. Per-scale vectors retained; enc/dec contributions summed per scale (a definitional choice — flagged, revisit only with cause).

**Identifiability threats, named in advance:**
1. **Substitution degeneracy** (measured 2026-07-28: identity pays A₀ = 41k > I₀ = 11k in one fit, 5.6k vs 3.3k in another): copy-type content can route through either currency. The decomposition may reflect optimizer choice, not task demand.
2. **Kept-spine leakage**: both ledgers under-count if content rides the unpriced spine (the §2 conditionality; IR-only probe quantifies it).
3. **β-path dependence**: the (I, A) split at the optimum may depend on the warmup trajectory.

**What the running stability sweep adjudicates (pre-registered predictions):** across 8 seeds × 2 β per family — if the (I, A) *split* varies wildly across seeds while I+A is stable, S3's decomposition is substitutional and the claim must weaken to the total; if the split is family-stable, the decomposition is intrinsic and S3 stands as stated. Either outcome is reportable; the kill condition (thesis §6-S3) triggers only if *neither* the split *nor* the total stratifies families stably.

**The class test the graded measurement cannot give:** S3's architecture-selection law is about *minimal nonlocality demand*. The discriminating instrument is the **attention-absent ablation** (`attn_max_hw=0`): families that still solve are locality-class L; those that fail are class NL; then (I, A) grades within class. Queued as a 10-row sweep (5 families × 2 seeds, one β) — cheap, next session, after the stability verdict defines what to compare it against.

---

*Companion ledger entries added same date. This document is the S1 section's skeleton for the paper and the pre-registration of the S3 stability analysis.*
