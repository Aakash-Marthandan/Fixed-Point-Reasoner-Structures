# Terminology: the paper's fixed vocabulary

One term per concept, defined once here and used identically everywhere. A synonym in the "never" column is deleted on sight. Definitions match the ledger (`Documentation/Design_Ledger.md`) and the instrument suite plan (`Documentation/Plan_2026-09-07_Instrument_Suite.md`); when the ledger and this file disagree, the ledger wins and this file is corrected.

## The objects

| term (use) | definition | never |
|---|---|---|
| **recursive reasoner** | a network applied repeatedly to a carried state to produce an answer (the class HRM, TRM, EqR and the DEC belong to) | "iterative model", "looped transformer" |
| **cell** | the network applied at each outer step (the field's TRM block stack; our nine-field DEC) | "module", "backbone" |
| **loop** | the outer-step schedule that carries the state, including the two-timescale H/L cycles, deep supervision, ACT halting and the segment carry ("the field loop" = HRM/TRM/EqR's) | "recurrence", "unrolling" |
| **outer step, D** | one application of the cell to the carried state; D = the number of outer steps at evaluation (D16, D64, D128, D256) | "iteration", "T" in prose (T is the training segment count) |
| **carried state, z** | the latent carried between outer steps (z_H, z_L); the object the instruments read | "hidden state", "memory" |
| **the DEC** | the Decimating Equilibrium Cell: the field's block and loop on a nine-field state with every parameter shared over the digit fields, so a digit permutation permutes the state and the logits exactly (exact S9 equivariance) | "our model", "QHRRN" in the paper's body |
| **the field cell / X0** | the TRM-MLP cell under the field's recipe reproduced on our stack (the reproduction control) | "baseline" |
| **native cell** | our earlier equilibrium cells with an answer register (the SOFT decoders of the corpus) | "old model" |
| **nine-field state** | the (F = 9, S = 81, w) state, one field per digit | "one-hot channels" |
| **exact S9 equivariance** | a relabeling of the nine digits permutes the state's field axis and the logits' digit axis exactly (tested bit-for-bit up to reduction order) | "symmetry-aware", "invariant" (the logits are equivariant, the halting head invariant) |
| **the orbit / position orbit** | the HRM/TRM position group of Sudoku (transpose, band/stack permutations, row/column permutations within them); "digit orbit" = the digit relabelings | "data augmentation" without the noun it augments |
| **regime** | the training recipe column: corpus (1k puzzles), augmentation, batch, wd, lr, EMA, steps, halting, selection | "hyperparameters" |
| **protocol** | the evaluation column: set, D, draws k, weights (EMA/raw), numerics (bf16/fp32), selection column | "setting", "config" |

## The measurements

| term (use) | definition | never |
|---|---|---|
| **cold / single pass** | one rollout from the cell's own start state (no draws) | "greedy", "zero-shot" |
| **exact accuracy** | the fraction of puzzles whose final grid equals the solution exactly | "solve rate" (allowed only in the decoder-class definition of yield), "accuracy" alone |
| **draw** | one rollout from a random start state (RI draw); k draws per puzzle | "sample", "restart" |
| **verified@k** | the fraction of puzzles solved by at least one of k draws, checked by the verifier (a coverage column) | "pass@k" |
| **residual-selected, t1r@k** | the draw with the smallest latent residual over the last three outer steps, selected without a verifier; t1r@k is its exact accuracy | "top-1", "best-of-k" |
| **spurious rate** | among wrong draws, the fraction whose residual lies below the median residual of the correct draws (a converged wrong grid) | "false positive rate" |
| **selector AUC** | the rank-AUC of the residual as a classifier of correctness over draws | — |
| **breadth** | the multi-draw coverage curve over k | "diversity" |
| **decoder class** | the two-axis reading of a grid: (i) the erasure threshold g50 and the search yield on the D64 records; (ii) the commitment dynamics (commitment at step 1, confidently-wrong, monotone solves, first-exact); letters SOFT / DECIMATING / MIXED by the registered rule | "solver type" |
| **erasure threshold g50** | the number of givens at which the logistic fit of exact accuracy vs givens crosses 50 % (None when no crossing in the tested range) | — |
| **search yield** | exact accuracy on puzzles with rating > 0 (the rated, hard subset) | — |
| **commitment at step 1** | the fraction of free cells whose readout confidence exceeds τ after the first outer step | — |
| **confidently wrong** | the fraction of committed cells that are wrong, read on failures at a given step | — |
| **stall** | an unsolved puzzle whose grid stops changing before D | "failure" (a failure is any unsolved puzzle) |
| **calibration at stalls** | among the most confident free cells at a stall, the fraction that are correct (with their mean confidence) | — |
| **decimation quality / E3** | at stalls: the committed fraction, the wrong-among-committed fraction, the peeling contradiction rate | — |
| **calibrated-commitment gap** | the difference between a decimating decoder's stated confidence at stalls and the correctness of its committed cells | — |
| **init radius** | exact accuracy from starts perturbed at scale ε, as a fraction of cold | "robustness" |
| **prefix inertness** | the change in accuracy when the puzzle-prefix embedding is zeroed | — |
| **halting head as verifier** | the AUC of the halting logit as a classifier of exactness at the last step | — |
| **depth regression** | a puzzle solved at some outer step and unsolved at the last | — |
| **monitor** | the held-out train-file puzzles (64 or 512) evaluated every 2k steps during training; the selection instrument | "validation set" (the test set is never used for selection) |
| **val-selected grid (vsel)** | the banked checkpoint with the best monitor value under the registered tie rule; "final" = the last checkpoint | "best checkpoint" |
| **memorization** | end segment-CE below the threshold or a vsel-minus-final drop above it (labeled, never disqualifying) | "overfitting" in claims |
| **noise floor** | the spread of a seed pair or triple at the selected grids; contrasts are read only beyond twice the floor | "variance" |
| **one-variable arm** | an arm that differs from its base by exactly one registered change | "ablation" (allowed in the section title only) |
| **law** | a regularity stated in one sentence with its scope (cells, regimes, sets) and its evidence table; numbered | "finding", "insight" |
| **MACs per outer step** | the analytic multiply-accumulate count of one outer step on one puzzle (the compute column) | "FLOPs" (unless converted) |

## Comparators (protocol column always attached)

| name | what it denotes |
|---|---|
| HRM | Wang et al. 2025, the 27M hierarchical model; its Sudoku-Extreme numbers are TRM's table row unless stated |
| TRM-MLP | Jolicoeur-Martineau 2025; the released alphaXiv reproduction checkpoint when a grid is meant |
| EqR | the Equilibrium Reasoners checkpoint (EMA weights = the headline; raw = the alt row) |
| CGAR | the curriculum-trained TRM-MLP checkpoint |
| SE-RRM | the symbol-equivariant recurrent reasoner (concurrent work; numbers as published, with digit augmentation) |

## House rules on numbers and names

- "program-record" is the only permitted use of the word record; "record" alone is deleted.
- Comparator nicknames ("the field's champion") never appear in band names; the paper says "EqR at D64, single pass".
- A number in prose carries its protocol in the same sentence; a number in a table carries it in the columns.
- The word "parity" means "within the noise floor of the field's single-pass number at the same protocol", and the protocol is named.

## The ARC-era objects and instruments (used in §6 and the appendices; definitions from the ledger's archived entries and the instrument map)

| term (use) | definition | never |
|---|---|---|
| **ARC** | the Abstraction and Reasoning Corpus (ARC-1): few-shot rule inference from support pairs, evaluated on held-out tasks; a rule inventory | "ARC-AGI" without the version |
| **verification game** | a task whose candidate answer can be checked exactly (Sudoku: uniqueness makes any valid completion the solution) | "puzzle" as a category |
| **generalization task** | a task whose answer cannot be checked and whose evaluation is transfer to unseen rules or instances (ARC) | "open-ended task" |
| **retention, Ret(g*)** | handed the correct answer as the start state, the fraction the map keeps as a fixed point after the schedule; the basin-existence probe (both domains) | "stability" alone |
| **final-map retention, retfm** | retention under the final map alone (t_norm = 1); the contractivity-at-the-solution readout | — |
| **basin existence vs reachability** | existence: the map holds the solution as a fixed point (retention); reachability: a cold or random start reaches it (cold, breadth); the decomposition that reads ARC as inventory-limited and Sudoku as search-limited | "convergence" for either |
| **truth-erasure** | a map that does not keep a handed correct answer (retention below 1 on that pair); repaired on ARC by the equilibrium core (retention 6 % → 29 %) | — |
| **wrong-stable** | a converged limit that is confidently wrong (ARC: 0.92 of converged limits); the ARC end of the converged-wrong continuum | "hallucination" |
| **converged-wrong rate** | among wrong endpoints, the fraction that are converged (the one instrument that reads the failure-class continuum on both domains: ARC .92 → memorized Sudoku maps .03–.16 → the field's RI-free base .03–.06 → RI + FPA maps ≤ .001) | — |
| **reachability wall** | the finding that on ARC most converged limits are wrong at every scale measured and no substrate intervention moved it; sharpened on Sudoku to a cell- and loop-dependent property | — |
| **basin-existence conditioning** | the enrichment of a hop rate on pairs whose fitted map holds a truth basin (31× on ARC): the walls are absent basins, not barrier heights | — |
| **anti-Arrhenius** | the ARC finding that the hop rate into truth basins falls with temperature (cold candidates only) | — |
| **codebook size N; the packing plane (N, r̄)** | the number of distinct rule basins a substrate holds and their typical radius; ARC-only (Sudoku has one rule) | — |
| **corruption ladder, S(ε), r̄** | the fraction of corrupted answers the basin absorbs at corruption ε; the basin radius (code distance) | "robustness" |
| **transfer gates rg / rt / val-hard** | generalization to never-trained rule families (rg), fresh instances of trained families (rt), the curated hard set (val-hard); ARC's generalization axis | — |
| **flux ledger, throat I_total, toll β** | the information crossing each internal cut of the map and its sum; the price on it (the priced information flow through RG cuts) | "regularizer" |
| **knee profile** | the universal per-scale flux shape a priced code sits on at the knee price; IR-fattened on ARC, extreme-UV on Sudoku | — |
| **the landscape-class law** | convergence-discipline mechanisms convert to accuracy on single-attractor constraint-satisfaction landscapes and not on inventory-limited generative landscapes; basin-existence conditioning is the discriminating instrument (H-33, sharpened) | — |
| **the init-distribution law** | randomized-init training buys convergence robustness on its training init distribution and nothing on other inits, so it pays at deployment iff inference starts from that distribution (H-37; both domains) | — |
| **the optimization-length tax** | on ARC, longer optimization at fixed price erodes unseen-family basins while trained families sharpen (H-40); its Sudoku observable is memorization (one pressure, two observables) | — |
| **TTT** | test-time training: the per-task adaptation layer ARC requires and Sudoku does not have | "fine-tuning" |
| **basin-snap voting / candidate supply × basin decoding** | the ARC conversion stack: candidates from cold starts, snapped to the nearest held basin and voted; the substitute for a verifier | — |
| **vacancy floor** | rule families for which the architecture holds no basin on any substrate measured (ARC) | — |
| **inventory-limited / search-limited** | the two regimes: performance bounded by which basins exist (ARC) vs by the search an instance requires (Sudoku) | "hard" / "easy" |
| **the two-sided iteration result** | the same extended-iteration instrument reads +0 solves on ARC at t = 96 and +16/40 on Sudoku's hard stratum at t = 64: the flagship figure of §6 | — |
| **the verifier's worth** | verified vote minus unverified majority on the same draws at the same k (Sudoku: +50–65 pp on wide-funnel maps, ≈ 0 on init-invariant maps) | — |

## Terms adopted from concurrent work (2026-09-08; cite the source at first use; see `CONCURRENT_GAPS.md` Part B)

| term (use) | definition | source |
|---|---|---|
| **symbol-equivariant** | the standard adjective for a cell whose output permutes exactly under a relabeling of the symbols; "exact digit-relabeling equivariance" when precision is needed | SE-RRM |
| **convergence residual** | the change of the carried state over the last outer steps; the selector's and the halting rule's signal | EqR, FPRM |
| **task-conditioned attractor** | the solution as a stable fixed point of the map conditioned on the puzzle; retention asks whether it exists, the cold pass whether it is reached | EqR |
| **transient near a saddle** | a stall: a still-moving state near a nearly-correct grid, never a fixed point | Lai et al. |
| **trapping-time distribution** | the distribution of first-exact steps; its tail is what basin entropy indexes | Lai et al. |
| **escape** | a restart leaving a wrong basin; its probability is the per-draw success rate | PTRM |
| **selection problem / irreducible failure** | a failure some restart solves / a failure no restart solves | Speed is Confidence |
| **guide alignment** | the halting head's AUC against exactness | guided reasoning |
| **where the noise enters** | memory (RI on the start state, Langevin noise on the carried state), observable (fresh noise on free cells), the inner recursion, or none | diffusion curriculum |
| **sound / unsound commitment** | collapsing a cell's candidate set to the correct / a wrong singleton; the calibrated-commitment gap is the unsound rate at stalls | LDT |
| **plausible but invalid** | the wrong-valid failure class | GRAM |
| **anytime solver** | a decoder whose accuracy rises with steps and never regresses (the depth-regression count is its test) | Sotaku, the diffusion curriculum |
