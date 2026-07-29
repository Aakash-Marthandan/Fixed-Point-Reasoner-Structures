# Reflection & Re-orientation — 2026-07-29

**Occasion:** PI-called halt after the first Phase-2 measurement days (Jul 27–29). Purpose: fine-detail audit of what we actually measured, big-picture check against the thesis and the deadline, and a corrected plan. **Companions:** `Design_Ledger.md` (evidence), `Thesis_Information_Holography.md` (claims), `QHRRN2_Architecture.md` (design).

---

## 1. A correction found during this reflection (fine detail first)

The ledger's sweep-1 finding **F1 mis-stated the floor estimates**. We reported the family ordering at the fixed point β=10⁻⁵ (constfill 1.4k < checkerboard 6.2k < identity 8.7k < …). But the S1 estimator is the **envelope**: per family, the *minimum* flux over all accuracy-qualified rows, over the whole β grid. Under the envelope:

| family | envelope upper-estimate (nats) | achieved at | F1 said |
|---|---|---|---|
| constfill | **326** | β=10⁻³ | 1.4k |
| identity | **5,617** | β=10⁻⁴ | 8.7k |
| checkerboard | **6,237** | β=10⁻⁵ | 6.2k |
| colorswap | **12,438** | β=10⁻⁵ | 12.4k |
| translate | **13,247** | β=10⁻⁵ | 13.2k |

Two consequences. (a) **Identity moves below checkerboard** — the "content-entropy ordering" story was partly an artifact of fixing β. The corrected reading: checkerboard's flux is not pure content transport; completion is a neighbor-*dependent computation*, and its inability to compress further at 10⁻⁴ (accuracy breaks first) leaves its envelope above identity's. Computation structure, not just content entropy, prices the cut. This is a *better* S1 story, and it is what the data actually says. (b) **Frontier protocol amended:** frontier points are per-family envelopes over accuracy-qualified rows; single-β columns are never quoted as floors. Ledger correction entry appended alongside this memo.

The meta-lesson is the point of this halt: our own headline finding survived one day before an internal review corrected it. The ledger caught it because the raw rows were preserved. This is the process working — and a warning about how quickly "measured" hardens into "known."

## 2. What we actually know (evidence-grade audit)

**Established, would survive review:**
- **Trainability of the full C1–C14 stack**: 6/6 pre-cloud gates, CI-3a reproduced cross-backend (CPU + TPU, float32-parity). The April failure class is closed and regression-gated.
- **The priced-channel mechanism works as designed**: tolls measured at every scale; β steers flux over 3–5 orders of magnitude; unpriced channels blow up (up to 10⁹ nats) — β, β_nl > 0 is now a standing requirement.
- **Operational area law (weak form)**: geometric I_s decay at moderate β across all five constructed families.
- **Binding-before-transport fragility**: identity solves exactly at every β tested (incl. 10⁻³); relabel/shift/completion families lose exactness at 10⁻⁴. Novel, crisp, and ours. *Scope caveat: measured at d=12, from-scratch fits, 8×8 content.*
- **Frontier terminus**: at β=10⁻³ transport families never reach compressed solutions (doubling the priced phase does not fix it) — the knee for binding families lies in (10⁻⁴, 10⁻³) *for this optimizer and budget*.
- **S3 identifiability caution (measured)**: identity's priced solution routes more through fine-scale attention than streams (A₀ > I₀) — copy-tasks can pay in either currency. Any S3 decomposition claim must survive this degeneracy.

**Suggestive but NOT yet evidence-grade:**
- The envelope table above: **upper bounds only** — 2 seeds, one steps-budget, no β-grid refinement near knees, and *no analytic lower bounds computed yet*. The S1 sandwich currently has one slice of bread.
- Zero LoO gap on constructed families: consistent with "these tasks generalize once fit," but also means **the S2 instrument has returned no signal yet** — we have not yet measured the thing the PI cares most about. The gap metric itself (mean support pix − held-out pix at selected params) may saturate; worth revisiting the metric alongside harder tasks.

**Known unknowns / untouched:**
- Analytic IC floors (thesis §2 proposition): unproven even for identity/constfill worked examples. Desk work; no compute.
- d-dependence of every Phase-2 result (everything is d=12).
- The entire pretraining arm: generators, corpus, pretrain runs, CI-3b, TTT-protocol measurements, H-10. **No code exists for generators yet — this is the longest unstarted pole.**
- S3 stability data (lost overnight; re-run pending). S4 untouched (by design, exploratory).
- Owed hygiene: mechanical citation re-verification; S1 proof write-up.

## 3. Regime caveat that must be said out loud

Every Phase-2 measurement so far is a **from-scratch full-parameter fit at toy scale** — not the deployment condition (pretrained bulk + boundary-only TTT). The sandwich logic survives this (any solver's achieved flux is a valid *upper* bound on the floor; pretrained solvers should only tighten it), and the fits are honest solvers. But: knee locations, binding fragility, and stability structure may all shift under the deployment protocol. Framing rule going forward: **Phase-2 constructed-family results are stated in the from-scratch regime**; deployment-regime versions arrive with Phase 3. The paper's S1/S2 narrative should present the from-scratch sandwich as the *first* instantiation, then (if time allows) the TTT-regime replication.

## 4. Process reflection (two days of operations)

**What worked:** the ledger discipline — every failure became a finding with a dated diagnosis (0/3 gate run → three protocol discoveries; harsh-β "artifact" → terminus result; today's F1 correction). Gates as instruments, not ceremonies. Honest FAIL records are becoming paper content.

**What cost us:** infrastructure assumptions taken on faith. The tally: SSH-attached execution (reset, double-launch), the launch shell-precedence bug, the unbounded kill call (7 h hang), the DMS-stops-billing assumption (false — **only node deletion stops TPU billing**), quota assumed from a stale note (real preemptible cap: 4 chips/zone), ceilings sized by hope rather than measured row-times. Rough overnight cost: ~$30 and the stability dataset. Total TPU spend to date ≈ $35–45 — trivial against $2,000, but the *pattern* (assumption → surprise → fix) consumed roughly 40% of two working days.

**Standing rules adopted:** (1) every unattended chain ends in node deletion, sized to finish inside the DMS window; (2) ceilings = measured row-time × rows/shard × 1.5; (3) periodic data rescue during long sweeps (per-N-rows scp or GCS), never only at the end; (4) no run is planned on hardware whose quota hasn't been verified that day; (5) protocol changes to gates/estimators land as dated ledger entries *before* the re-run they enable (we did this — keep doing it; it is the answer to "how many analyst degrees of freedom did you spend?").

## 5. Deadline math and the shape of the paper

Today → freeze (Sep 28) = **61 days**. Working backward:

- **Phase 4** (full eval + figures + writing buffer): last 2 weeks (Sep 15–28).
- **Phase 3** (generators → pretrain → CI-3b → TTT protocol → dev-30 gate → ARC-1 eval): realistically 4 weeks if generators take one; **must start by ~Aug 18** to fit. Dev-30 gate slips from Aug 31 to ~Sep 5 unless generators start sooner.
- **Phase 2 completion window: now → ~Aug 17** — S1 sandwich (proofs + envelope measurement), S2 first signal (dev-30 from-scratch or early-TTT), S3 stability + architecture-selection test, d-sweep spot-checks.

**Decision point (put it in the calendar): Aug 15.** If pretraining is not on rails by then, the paper's center of mass moves to the physics program (S1–S3 at toy scale, honestly framed, with CompressARC as the efficiency baseline comparison) plus whatever solve-rate point exists — rather than risking both halves. The physics deliverables do not depend on beating TRM; that was always the design.

## 6. The re-oriented queue (proposed, PI to confirm)

1. **Desk (no compute, this week):** S1 analytic floors for the five constructed families (the data-processing lower bound + per-family counting arguments); citation re-verification pass; S3 decomposition semantics write-up (what exactly (I, A) measures given emission-side pricing and the unpriced kept channel — must be crisp before the stability figure).
2. **Cheap compute:** stability sweep re-run (4-wide, 6 h ceiling, periodic rescue — ~$10); β-grid refinement near knees (3×10⁻⁵, 3×10⁻⁴) to sharpen envelopes.
3. **PI console items:** quota bump (8-chip spot lane); optional Cloud Scheduler delete-backstop.
4. **Dev-30 selection** to 30 verified tasks (renderer pipeline exists; ~2–3 focused hours) → first real S2 attempt in the from-scratch regime.
5. **Start generators (the long pole) no later than Aug 4** — even at half attention, so Phase 3 can start on schedule.

---

*Committed as the orientation artifact for the restart. The ledger correction entry accompanies this memo.*
