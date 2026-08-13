# Frontier Consolidation — 2026-08-13

**PI report: everything the frontier sessions (08-12/13) measured, organized
as a single picture of the object under study — the priced equilibrium
landscape — plus what remains undecided and what the pending wave-2 readouts
decide. Every number recomputed from disk by a named analyzer; companions:
Design_Ledger.md §5 (the dated evidence), Research_Brainstorm.md (idea
provenance), runs/analysis/ (artifacts).**

---

## 1. The object, as the instruments now describe it

Two days of measurement collapse into one coherent description. The priced
equilibrium substrate is a **sparse, coarsening, spectrally-universal code**
whose accessibility is **family-structured**:

### 1a. The code shortens with everything (the throat curve)
Priced per-episode information declines **785 → 657 → 635 → 602 → 520 → 459
nats** across d16→d64 and 20k→53k steps (78k→763k params, six substrate
generations; `analyze_spectra`, `analyze_steps`, pilot batteries). The task
sets the information; capacity *and* optimization budget both improve the
code toward it. S1's cleanest empirical statement.

### 1b. Two universal spectral profiles, β-selected (cluster O)
Normalized per-scale information Î(s) is **d-invariant within each pricing
regime** — free arms collapse onto (.76/.18/.045/.012/.003), knee-priced arms
onto (.69/.14/.085/.035/.048) — across d16-d48, T4/T6, and both corpora. The
toll does not merely shrink the throat ~250×: it **reshapes information
toward the IR**, and both shapes behave like fixed profiles of the flow. The
free-steepening prediction was refuted; the two-fixed-point reading replaced
it. Deviation from the knee profile is now a scaling diagnostic for the
5-75M track.

### 1c. Geometry: count and radius trade against *optimization*, not volume
- The (N, r̄) frontier trades only ~0.5 cells of Hamming volume per basin —
  the state space is packing-unsaturated by ~2 orders (`analyze_packing`).
  Count is volume-cheap; radius is optimization-expensive.
- **Budget/scale consolidate**: d48 count 51→42 under 2× budget; d64 count
  32 with radius held at the record band (.78) and throat at the record low.
  Fewer, wider, cheaper basins — a coarsening flow.
- The steps(d) law: linear bracket c ∈ (417, 625] steps/d from the
  sufficiency constraints; quadratic unexcluded; the d64@53k readout fired
  *on the resolution boundary* (.286 vs .30 at n=7) — the 80k decider (in
  wave-2) settles the branch.

### 1d. Dynamics: capture is cold settling; walls are vacancies (cluster Q)
Hop rate **declines** monotonically with temperature (anti-Arrhenius,
.0129→.0062 across a 32× T range); no activated-crossing regime exists in
the accessible window. Conditioning on basin existence is decisive: **31×
enrichment** (3.1% vs 0.1% per sample) — where pretraining built a basin,
cold noise finds it; where it didn't, no temperature helps. Sampling is a
decoder, not a search engine (third and sharpest form of the
no-inference-time-rescue law).

### 1e. NEW — vacancy is family-structured and substrate-robust
Per-family GT-retention across **eleven substrates** (`analyze_vacancy`,
runs/analysis/family_vacancy_20260813.txt): ExtractObjects **0% on all
eleven**; Copy ~2%, Count ~7% — while CleanUp/CompleteShape/
HorizontalVertical hold 53-60%. Basin-formability is largely a **task-family
property**, nearly invariant to scale, pricing, corpus, and dials. Two
consequences: (i) the basin-creation lever for the floor families is
*representational* (architecture/corpus mechanism), not more-of-the-same
scaling; (ii) a **scale-regression flag**: HorizontalVertical collapsed
78-89% → 0% on *both* d64 arms — the first family lost to scale, wanting an
explanation before 5-75M commits.

### 1f. Adaptation: damage is gradient-directed (cluster P + [H-12])
Random perturbations at fit-scale norms produce only 1.6-2× spine-vs-boundary
anisotropy — nowhere near [H-12]'s catastrophe. Training destroys basins
along *chosen* directions, not generic ones. KL-anchored TTT is therefore
the supported tool (promoted to convert-phase default); film — the (s,t)
modulation — is the one standout-fragile subspace (freeze it); gates/
codebook/boundary are confirmed-safe.

### 1g. Conversion: supply-bound at parity (cluster S)
The record decoder × its own cold candidates lands **16/144, 0/48** —
exactly the all-eq parity line, kill unfired, prediction missed, with the
attribution pre-paid by the 62%-coverage measurement: **candidate supply
binds**; the decoder over-delivers per unit coverage (radius partially
compensates). The heirloom old-arch pool stays; Center2-p1 remains the
concrete retirement blocker. One genuine positive: EqR-style residual
ranking — dead pre-shaping — now performs at vote-parity (att1 13 vs 12);
shaping made stability informative. The portfolio law now stands in three
independent forms (temperature / thermal settling / inits): single-substrate
diversity mechanisms cap below the heterogeneous pool.

### 1h. The trained scalars tell their own story
The chosen flow constant **scales**: η = .058 (d16/T4) → .234 (d64/T6, arm
C), with arm A's coupled α₂ independently at .286. The EqR damping
"convergence" was a small-substrate coincidence. And the coupled pair
drifted **expansive** (α₁+α₂ = 1.049 from contractive init) — the prime
suspect for arm A's retention halving (18 vs 32) alongside its radius
record (.94): an expansive map sheds shallow basins and keeps wide ones.
Deep structure (S(.4), exact@T) was untouched — the A-bundle's damage is
confined to the shallow shell. Wave-2's single-toggle cells attribute this.

## 2. What is currently undecided (and what decides it)

| Open question | Decider | Status |
|---|---|---|
| steps*(d): linear vs quadratic | C80 (d64@80k) rg-radius ≥.30 at n≥9 | in wave-2 |
| A's retention halving: coupling vs floors vs RI | Dcoup/Dfloor/Dri vs C53 | in wave-2 |
| NI's spurious-attractor claim | B vs A wrong-stable + battery | in wave-2 |
| RI trains multi-init breadth (EqR's core claim) | samp Dri vs C53, coverage jump | in wave-2 |
| Cross-substrate init diversity ≈ heirloom? | S2-cell (portfolio multi-init) | registered |
| Metric-TTT revival | P2 (gradient-direction perturbations) | registered |
| HV family's d64 collapse | needs a probe (see brainstorm) | new |
| Vacancy-floor families (ExtractObjects/Copy/Count) | representational lever TBD | new, § brainstorm |

## 3. Honest position

Solve conversion remains the frontier: the deployed champion is still
C.3′-class (2/48 val-hard task-level, old stack), and nothing measured this
week moved task-level conversion — the week's value is that the *reasons*
are now measured (supply-bound candidates; family-structured vacancy; no
inference-time rescue). Statistics: the frontier cells are n=1 seed; the
four laws are seeded but the pilot readouts are not; val-hard carries ~15
adjudications and steers only. Ops: v6e spot churn cost ~$10 and a night —
countered with 5-minute GCS durability; the door map has seven zones.

## 4. What this buys the next phase

Pretrain-13-full (seeded, post-wave-2) inherits: steps(d) from the decided
branch; the knee profile + frontier coordinates as diagnostics; floors as
standard (they behaved exactly as designed); the A-decomposition's surviving
mechanisms; KL-anchored + film-frozen TTT for the convert phase; and two
new experiment classes from the vacancy finding (representational basin
creation for floor families; the HV scale-regression probe). The paper's
empirical spine — throat curve, two profiles, coarsening geometry,
anti-Arrhenius dynamics, family-structured accessibility — is coherent,
attributed, and already figure-ready.
