# State of the Program — 2026-08-10

**PI report: where we are, what stands against us, and the next best steps.
Every number recomputed from disk this session; adversarial checks run on the
headline before it appears here. Companions: Design_Ledger.md (the evidence
chain), Gap_Analysis_CompressARC.md (the post-mortem),
Research_Brainstorm.md (idea provenance).**

---

## 1. Where we are

### The result, stress-tested
Basin-snapped population voting (C.3′) produced the **first task-level solves
on the 48-task curated-hard holdout in project history: 2/48**
(`ca_Center2`, `ca_InsideOutside4`), with per-output 17/144 — the best
realized protocol of the season (prior best 14; 8-protocol oracle union 26).
Adversarial checks on the headline:

- **3 of the 17 exact pairs lie beyond the entire season's 8-protocol
  union** — and both task conversions required one of them. This is not
  re-aggregation of known answers.
- Mechanism split, measured: on `Center2-p0` **no input candidate equaled
  the truth — the snap corrected a near-miss into exactness** (the basin
  acting as an error-correcting decoder, thesis S4 operational). On the
  other two new pairs the truth existed among members and every previous
  vote failed to select it; snapping collapsed wrong candidates onto shared
  attractors and the correct one won. Correction and selection, 1:2.
- No ground truth touches any solve path (candidates, fits, snapping, and
  votes are all GT-free; scoring happens after attempts are fixed).

### The verified causal chain behind it
Each link measured-in individually this week, none assumed:
1. The old architecture's argmax feedback was a gradient wall → basins
   untrainable at any budget (E8 matrix + B4/B5 controls).
2. Continuous carried state (E10) made the trajectory differentiable →
   corpus-scale anchor pretraining built genuine near-idempotence at truth
   (GT-retention 6%→29%, η-confound checked and the claim sharpened, old
   substrate control at 5% closes attribution).
3. Full-capacity TTT *erodes* those basins (29%→12.5%, C.2) → keyhole fits
   preserve them.
4. Iteration alone cannot reach them (horizon null at t=96, twice-measured
   across substrates) — but the season's population machinery supplies
   candidates within the measured pull radius on 69% of failed pairs.
5. Snap-then-vote converts: basins as decoder over near-miss candidates.

### Position against the frontier (honest)
CompressARC 20% ARC-1-eval @ ~80K/task-fitted; FPRM 47.5% ARC-1 @ 7M with
genuine equilibria; PoE 71.6% @ ~2¢/task. **We are far below frontier solve
rates.** What we hold that the field does not: direct basin instrumentation
(retention, corruption ladders = code-distance spectra), a fully attributed
repair chain from measured pathology to conversion, and the decoder reading
of test-time reasoning. The paper's spine is real regardless of where the
solve rate lands by freeze; the solve-rate credibility now rests on scaling
the mechanism, not finding it.

## 2. Problems and challenges

**Foundational (the physics debt — the deepest items are still open):**
- **The rate side of the thesis has never shaped the landscape.** Basins are
  *denoising-shaped* (restoration around completions — score-matching
  physics), not *compression-shaped*: β=0 at every fit and pretrain-8;
  S1/S2 remain instruments, not objectives; B1 (MDL-native loss — the
  CompressARC-convergent, theory-completing move) is registered and
  unbuilt. Risk: restoration basins may transfer weakly to novel rules
  precisely where compression-shaped basins would generalize — the 29%
  retention on held-out tasks is encouraging but unexplained by our own
  theory until the rate term is load-bearing.
- **Transport ([H-6′]) unaddressed**: the rule is still re-inferred from
  each input; the committed-rule boundary condition (E4, ~50 lines) remains
  the architecture's known open wound. The snap sidesteps it via candidate
  diversity; it does not fix it.
- **The latent scratchpad is dead**: α_z ≈ 0 — the A.2 z-carry never
  engaged (anchor training is solvable through the y-channel alone, so no
  gradient pressure opened the gate). The field's evidence (TRM ablation,
  ~15pp) says this axis matters; ours is currently decorative.
- **No stochastic/annealing axis**: η=0.058 quasi-static flow, no
  temperature, no trajectory sampling — H-5's annealing and the field's
  candidate-diversity engines (CompressARC's latent noise, the ARChitects'
  trajectory harvest) are unexploited on our substrate. Candidate diversity
  currently comes entirely from *old-architecture* members — an external
  crutch.

**Statistical / methodological:**
- 2/48 at n=1 seed, one bulk, one candidate pool: binomially fragile.
- **Val-hard's evidentiary value has decayed** through ~9 protocol
  adjudications this week. It remains a fine gate; it is no longer a clean
  claim set. Final numbers must come from the untouched sets (dev-30 under
  the last-shots law; ARC-1 eval split still virgin).
- Candidate-source asymmetry is an open question, not a free win:
  same-substrate members may be *more* correlated with the map's own wrong
  attractors (candidates and basins sharing biases) — the next experiment
  tests this both ways.

**Mechanistic limits:**
- Pull radius is bounded (ladder: strong to ε≈0.2, partial at 0.4) and pull
  rates are a minority of pairs; the snap converts the near-miss shell, not
  the 74/144 pairs where no member comes close.
- TTT capacity remains unusable (C.2 erosion) until basin-preserving
  adaptation exists (LoRA-scale / KL-anchored-to-pretrained — unbuilt).

## 3. Next best steps (ranked; each with its registered question)

1. **Same-substrate populations × snap** (cheap, one lane): members fitted
   on pretrain8 as candidates for its own basins. Decides the
   candidate-source question in both directions (better-aligned vs
   more-correlated) and removes the old-arch crutch if it wins.
2. **Transport/E4** (~50 lines): frozen support-rule conditioning the
   equilibrium. The oldest measured wound; now testable with basin
   instruments (does committed-rule decoding enlarge the reachable basin
   set?).
3. **Pretrain-9** (one lane-day): η floor/schedule (quasi-static was
   learned, not chosen), z-gate engagement signal (auxiliary loss or warm
   α_z init — wake the scratchpad), seed ×2 (variance), **and a β>0 arm —
   the first compression-shaped-basin experiment (B1-lite)**: does pricing
   change retention *transfer* to novel rules? This is where the thesis
   starts paying for itself or gets billed.
4. **Basin-preserving TTT** (LoRA/KL-anchored) — unlocks per-task capacity
   without the C.2 erosion.
5. **Protocol hygiene**: freeze the C.3′ stack; after (1) lands, take the
   ONE registered dev-30 shot (eval-6) under the holdout law; keep the
   eval split virgin for the final. The Aug-10 memo ships on this report.
6. **B1 proper** (MDL-native objective with query co-compression) — the
   next big build; the convergence of our theory with the strongest
   published mechanism, now with basin instrumentation to measure exactly
   what it changes.

## 4. One-paragraph bird's eye

The project spent a season proving that its solver's fixed points were the
wrong ones, one week rebuilding the substrate so that truth could be a
fixed point at all, and one day showing that those basins — used as an
error-correcting decoder over the population's near-misses — convert real
held-out tasks. The mechanism is attributed end-to-end and every component
carries its measurement. What stands between this and a competitive solve
rate is no longer a mystery: the rate term must shape the landscape (B1),
the rule must transport (E4), the scratchpad must wake (z), diversity must
become native (sampling), and adaptation must preserve what pretraining
builds. Each has an instrument waiting for it. The physics stopped being
decoration this week; the discipline is what made that checkable.
