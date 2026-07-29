# The Physics of QHRRN-2

*The inspiration, the mathematical tools, and — importantly — the exact
boundary between what is load-bearing mathematics and what is metaphor.
Every physics claim here carries one of the project's rigor classes:*
**[theorem]** *(proven, cited),* **[solvable model]** *(exact in a stated
limit),* **[construction]** *(exact by our own design),* **[hypothesis]**
*(registered with a named test and kill condition in the
[Design Ledger](Documentation/Design_Ledger.md)),* **[metaphor]**
*(organizing language only — never evidence).*

---

## 1. The lattice picture: reasoning as renormalization

An ARC grid is a small 2D lattice whose cells take one of ~10 discrete
states — formally a Potts-like configuration $x \in \{0..9\}^{H\times W}$
**[construction]**. An ARC *rule* is a map $Y = R(X)$ that is usually local
on this lattice (objects, neighborhoods, rows/columns) with a small number
of global degrees of freedom (the rule's parameters). The physics instinct:
solving such a task is a **renormalization-group flow** — coarse-grain the
grid UV→IR, identify the relevant operators (the rule), discard irrelevant
microstructure, then flow back down to paint the answer **[hypothesis:
operationalized]**. The architecture literalizes this: five 2×2
coarse-grainings ($32^2 \to 1$), a discrete rule-selection event at the IR
point, and a mirrored decoder.

The tensor-network correspondence is precise at the level of *wiring*
**[construction]**: our seam mixers are MERA's **disentanglers** — they act
on 2×2 blocks *offset by (1,1)* from the pooling blocks, so entanglement
across pooling boundaries is absorbed *before* the isometry can sever it
(deep-learning twin: shifted windows). The pool-and-split step is the
**isometry** — except that where a lossy funnel would discard the
complement, we *file* it (see §3). The recursion with shared weights is a
radial re-flow of the same RG map, and a solved task is a **fixed point of
the iterate map** $Y_{t+1} = F(Y_t)$ — a measurable diagnostic, not an
axiom **[hypothesis H-2]**.

## 2. Holography: entropy as minimal cut, information as flow

The Ryu–Takayanagi formula computes the entanglement entropy of a boundary
region as the area of a minimal bulk surface. Three mathematical results
make this usable outside gravity:

- In **random tensor networks** at large bond dimension, the entropy of a
  boundary region equals the weight of the minimal cut through the network —
  RT exactly, via an exact Ising mapping (Hayden–Nezami–Qi–Thomas–Walter–
  Yang, arXiv:1601.01694) **[solvable model]**.
- RT can be rewritten with no surfaces at all: entropy equals the **maximum
  flux of a divergenceless, norm-bounded flow** — "bit threads"
  (Freedman–Headrick, arXiv:1604.00354), i.e., a Riemannian max-flow/min-cut
  theorem **[theorem, conditional on RT]**.
- On **Bruhat–Tits trees** (the geometry our hierarchical network actually
  has), RT-like minimal-cut entropy formulas are *proven* for perfect-tensor
  holographic codes, including multi-interval regions
  (Heydeman–Marcolli–Parikh–Saberi, arXiv:1605.07639, 1801.09623)
  **[theorem]**.

The transfer to machine learning needs care. Quantum max-flow = min-cut is
**false** for general tensor networks (explicit counterexamples), but true
when all bond dimensions are powers of a fixed integer — the
uniform-bond-dimension regime (Cui et al.) **[theorem + counterexample]**.
Our design responds in two ways **[construction]**: uniform width per scale,
and — decisively — an information ledger that is **classical Shannon
information through explicit noisy channels**, where the unconditional
classical max-flow/min-cut theorem applies. We import the *mathematics* of
holography (cuts, flows, floors), never its gravitational content.

## 3. The flux ledger: a variational bound as a toll booth

Every boundary stream and every attention message is a Gaussian variational
code: the encoder emits $(\mu, \log\sigma)$, the sample
$b \sim \mathcal N(\mu, \sigma)$ crosses the cut, and the price is

$$I \;=\; \mathrm{KL}\!\left(\mathcal N(\mu,\sigma)\,\|\,\mathcal N(0,1)\right)
\;=\; \tfrac12 \sum \left(\mu^2 + \sigma^2 - 2\log\sigma - 1\right)\ \text{nats},$$

summed per scale into the stream ledger $I_s$ and the attention ledger
$A_s$. This is the variational information bottleneck bound: the KL is a
certified **upper bound on the mutual information** the channel carries
about its input **[theorem, standard]**. Two design subtleties:

- **Usage, not capacity.** An architectural bond-dimension penalty (the
  original theory documents' $\lambda \sum \|\gamma_l\| \ln \chi_l$) prices
  what the network *could* carry. We price what it *does* carry — the
  tighter MDL reading, and the one that makes memorization measurable
  **[construction; ledger S-1]**.
- **Emission-side pricing.** Attention messages are sampled *before* the
  convex attention mix, so by the data-processing inequality $\sum A_s$
  upper-bounds what actually crosses the nonlocal channel
  **[construction]**. The toll is charged at the wormhole's mouth.

The objective is a **priced free energy** **[construction]**:

$$\mathcal L \;=\; \underbrace{\textstyle\sum_t w_t\,\mathrm{CE}(Y_t, Y^*)}_{\text{fit}}
\;+\; \beta \underbrace{\textstyle\sum_s I_s}_{\text{hierarchical flux}}
\;+\; \beta_{nl} \underbrace{\textstyle\sum_s A_s}_{\text{nonlocal flux}}
\;+\;\dots$$

$\beta$ is an exchange rate between fit and abstraction. Measured behavior
(2026-07): with $\beta = 0$ the free channels inflate to $10^7$–$10^9$
nats; $\beta > 0$ compresses them by 3–5 orders of magnitude at equal
accuracy. An **operational area law** — geometric decay of $I_s$ from fine
to coarse — appears at moderate $\beta$ in every family measured so far
**[measured; from-scratch regime, toy scale]**.

## 4. Floors, envelopes, and the sandwich

For a task family and a cut, define the **floor** as the least information
any exact solver must move across that cut. The project's central empirical
object is a two-sided *sandwich*:

- **From above:** the measured accuracy–flux frontier as $\beta \to 0^+$ —
  our KL ledger certifies these as **upper estimates ("envelopes")**.
- **From below:** analytic lower bounds. Here the mathematics is honest
  about a boundary (`Documentation/S1_Floors.md`): for **spatial cuts**
  (grid bipartitions), floors are rigorous — the induced two-party function
  has a distributional **communication/information-complexity** lower bound
  by data processing **[theorem, classical]**; translate-right must move
  exactly one column ($\approx 8\ln 5 \approx 12.9$ nats across a vertical
  mid-cut), checkerboard-completion pays $O(\text{boundary})$ — a literal
  area law. For **scale cuts**, no unconditional floor exists (the kept
  spine is a continuous unpriced channel), so scale-cut claims are
  conditional on a measured kept-capacity model **[stated limitation]**.

First measurements produced a finding we did not predict: checkerboard's
output is ~4 nats of *parameters*, yet its measured envelope exceeds
identity's, whose content is 103 nats — the optimizer learned to
*transport* the pattern rather than *compute* it. **The frontier measures
learned solvers, not information floors**, and the gap between them
decomposes into coding rent, redundancy, and optimization gap — each
separately measurable **[measured + analysis]**.

## 5. Wormhole tolls: nonlocality as a priced resource

Purely local hierarchies are provably weak at long-range correspondence;
attention fixes this but ordinarily as an *unpriced* shortcut. Amendment D
prices it: attention runs at **every** scale, its messages bottlenecked and
metered ($A_s$). The physics-flavored reading **[metaphor, flagged as
such]**: attention edges are wormholes threading the hierarchical geometry,
and $\beta_{nl}$ is the toll. What is *not* metaphor is the measurable it
creates **[construction]**: the decomposition $(\,\Sigma I_s,\ \Sigma
A_s\,)$ — hierarchical vs nonlocal information demand per task — with a
pre-registered stability analysis (are the two currencies intrinsic to the
task, or substitutable by the optimizer? Identity, measured twice, can pay
in either — the "attention-copy degeneracy" — which is exactly why the
stability claim is part of statement S3, not an afterthought).

## 6. Symmetry breaking: rule selection as a phase transition

The rule codebook gives discrete rule tokens; a temperature-annealed
softmax selects among them. The entropy $H[q]$ of the selection
distribution is an **order parameter**: early in per-task adaptation the
distribution is broad (symmetric phase); as $\tau$ anneals, support
evidence breaks the degeneracy and $H[q]$ collapses (selected phase)
**[construction + hypothesis H-6]**. The honest scope: this is a
finite-size, large-deviations statement about an annealed categorical
variable — thermodynamic SSB language beyond that was explicitly retired.
The same logic extends to the color sector **[construction, Amendment A]**:
the core is exactly $S_9$-equivariant (a color-*relational* prior — most
ARC rules are), and strictly equivariant networks *cannot represent*
color-constant rules; per-color bias vectors, trained only at adaptation
time, are the symmetry-breaking field that evidence switches on.

## 7. Memorization as excess flux

The generalization statement (S2): a solver's **excess flux** above the
floor is its memorization, predicted to track the leave-one-out
generalization gap and to respond causally to $\beta$-intervention
**[hypothesis, kill-conditioned]**. The theoretical anchor is the
information-theoretic generalization literature (Xu–Raginsky / CMI): those
bounds are often vacuous for deterministic nets because $I(S;W)$ is
ill-defined — our channels are *stochastic by construction*, so the
measured quantity exists and is non-vacuous; the formal adaptation is
labeled conjecture-with-proof-owed. Constructed families show ~zero gap
(they generalize once fit) — the S2 signal requires real, memorization-
prone tasks; that campaign is queued.

## 8. What we explicitly do NOT claim

- No AdS/CFT duality for neural networks: no large-N limit, no
  diffeomorphisms, no gravitational dynamics. The Mehta–Schwab
  variational-RG↔RBM episode — contested, then defended on an exactness
  condition — is our cautionary tale: **exactness conditions stated, or the
  claim is labeled heuristic.**
- Engineered correspondences (holography-as-ansatz networks) are cited as
  motivation, never evidence. The random-tensor-network theorems hold under
  conditions never verified for trained networks — that verification gap is
  precisely the measurement program this project runs.
- The dead metaphors stay dead: no claim survives here without a test or a
  theorem attached.

## 9. Reading list (the load-bearing citations)

RT on trees: Heydeman, Marcolli, Parikh, Saberi — arXiv:1605.07639,
1801.09623 · Random TNs: Hayden et al. — 1601.01694 · Bit threads:
Freedman, Headrick — 1604.00354 · Quantum MFMC and its failure: Cui et al.
· ConvAC separation-rank/min-cut: Levine et al. — 1704.01552 (scope-limited
to ConvACs; we construct rather than rely) · VIB: Alemi et al. — 1612.00410
· Information complexity: Braverman–Rao lineage · Successive refinement:
Equitz–Cover · Generalization: Xu–Raginsky — 1705.07809 · TRM (the
efficiency baseline lineage): 2510.04871 · CompressARC (nearest neighbor,
mandatory baseline): 2512.06104 · Skip-path confound in bottleneck
measurement: 2505.24668.

*Every claim above is tracked with its status and test in the
[Design Ledger](Documentation/Design_Ledger.md); the full statements S1–S4
with kill conditions live in
[`Documentation/Thesis_Information_Holography.md`](Documentation/Thesis_Information_Holography.md).*
