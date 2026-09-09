# Decimation, equilibrium, and the decoder view: the references and the explanation (2026-09-08)

Purpose: the lineage the paper cites when it says "decimating equilibrium cell" and "decoder class", with the explanation in the paper's voice. ⚠ marks bibliographic details to verify at writing.

## 1. The three meanings of "decimation" the paper draws on

**In the renormalization group (physics).** Decimation is the real-space RG step that eliminates a subset of degrees of freedom and renormalizes the couplings of the survivors: Kadanoff's block spins (1966), Wilson's real-space RG (1975), and the Migdal–Kadanoff scheme, in which bonds are moved away from the sites to be eliminated and the eliminated sites are summed out (Migdal 1976; Kadanoff 1976). After each decimation the system is described by fewer variables with effective couplings; a fixed point of the map from couplings to renormalized couplings is a scale-invariant theory. The word carries the meaning the paper needs: **a decimating step removes variables from play and rewrites the problem on the rest.**

**In constraint satisfaction (statistical physics of hard problems).** Belief propagation (BP) computes marginals on a factor graph; its fixed points are stationary points of the Bethe free energy (Yedidia, Freeman, Weiss 2005). BP-guided decimation solves a constraint-satisfaction problem by alternating two moves: run BP to a fixed point, then **fix** (decimate) the variables whose marginals are most polarized, reduce the problem, and repeat (Montanari, Ricci-Tersenghi, Semerjian 2007; Ricci-Tersenghi and Semerjian 2009 for the cavity analysis of the decimated problem; Mézard and Montanari 2009, chapters on BP and decimation, for the textbook treatment). Survey propagation is the same scheme with messages that carry the cluster structure of solutions near the satisfiability threshold (Mézard, Parisi, Zecchina 2002; Braunstein, Mézard, Zecchina 2005). Two properties of decimation are the ones our instruments measure: it is **irreversible** (a fixed variable is never revisited without backtracking, which is why decimation fails where the wrong variable is fixed early), and its **failure is a contradiction** downstream of a wrong fix (the reduced problem becomes unsatisfiable while every message looks confident).

**In coding theory (the erasure channel).** A Sudoku puzzle is a codeword (the solution grid) sent through an erasure channel that erases the blank cells; the givens are the received symbols. The peeling decoder for the binary erasure channel (Luby, Mitzenmacher, Shokrollahi, Spielman 2001) repeatedly finds a constraint with exactly one erased variable and fills it in; it stops at a **stopping set**, a set of erased variables every one of whose constraints touches the set at least twice (Di, Proietti, Telatar, Richardson, Urbanke 2002; Richardson and Urbanke 2008). On Sudoku, peeling is naked and hidden singles; a stopping set is a puzzle state where no cell is forced and a guess (a decimation) is required. Iterative decoding has a **threshold**: below a critical erasure fraction the peeling decoder succeeds with high probability, above it it stops (density evolution; Richardson and Urbanke 2008). The Sudoku analog is the givens count: our erasure-threshold instrument g50 is the number of givens at which a learned map's single pass crosses 50 %.

## 2. What "decimating equilibrium cell" means, and how it relates to a decoder

A recursive reasoner applies one map to a carried state until the readout stops changing; the readout at a fixed point is its answer. Read as a decoder, the map takes the received word (the givens) and the erased positions (the blanks) to a completion. Two kinds of map appear in the corpus:

- **A soft decoder** keeps every cell's distribution open and moves all of them together: after one step 3–13 % of cells are committed, the readout entropy is about .5, solves arrive after 7–17 steps, and the single pass has an erasure threshold like peeling's (g50 at 26–27 givens on our native cells) with a low search yield on the rated puzzles (25–36 %). This is BP without decimation: it converges to a fixed point that is a marginal, not a completion, wherever the constraints do not force the cells.
- **A decimating decoder** commits: after one outer step 91–99 % of the cells are at confidence 1.000 and the later steps propagate from the committed cells; solves arrive at step 1–3; there is no erasure threshold in the tested range and the search yield is 90–98 %. This is BP-guided decimation with the decimation folded into the map: the field's two-timescale loop (H/L cycles with the gradient through the last cycle) is what makes the map commit, which is why the loop, not the operator or the scale, sets the class (Law 1).

The **equilibrium** in the name is the fixed point the decimating map reaches: on a solved puzzle the solution is a fixed point of the map (retention 1.0; the Jacobian at the solution contractive, λ .67–.73 on our cells), and the decimated cells are the survivors of the RG-like reduction. The **decimating equilibrium cell** (the DEC) is the field's decimating loop placed on a state with one field per digit and every parameter shared across the fields, so the map is exactly equivariant under digit relabeling: the decoder cannot tell the digits apart, which is the symmetry of the code.

The decoder view is what makes the failure class measurable, and it inherits decimation's two properties:

1. **Irreversibility.** A decimating map does not revise a committed cell: at a stall 88–93 % of the cells are committed, 54–68 % of the committed cells are wrong, and peeling from the committed cells contradicts the givens on 100 % of stalls (the decimation-quality row E3). The stall is the "contradiction downstream of a wrong fix" of BP-guided decimation, read on a learned map.
2. **Confidence carries no information at the failure.** The five most confident cells at a stall are right 41–62 % of the time at confidence 1.000 on every decimating grid measured, ours and the released models' (Law 2). In decimation language: the polarization that guides the fix is uninformative exactly where the fix is wrong. This is the gap the commit head arm (C6) is designed to move: a learned, calibrated decimation signal in place of the readout's confidence.

The reference decoders anchor the axis: peeling (naked and hidden singles) solves 11 % of the natural test distribution, BP 17 % and BP + decimation 22 % on the reference slice, our natives 35–46, HRM 62, TRM 84, CGAR 92 ≈ X0 92 ≈ EqR 93 (the ladder of `Report_2026-09-06_FieldCheckpoints_Instruments.md`), with the stall-set overlap with BP falling monotonically along it: the learned decimating decoders are not BP, but they fail where decimation fails.

## 3. The references (bib keys in `draft/refs.bib`)

- Kadanoff, L. P. (1966). Scaling laws for Ising models near T_c. *Physics* 2, 263–272. ⚠ verify pages.
- Wilson, K. G. (1975). The renormalization group: critical phenomena and the Kondo problem. *Reviews of Modern Physics* 47, 773–840. ⚠ verify.
- Migdal, A. A. (1976). Phase transitions in gauge and spin-lattice systems. *Sov. Phys. JETP* 42, 743.
- Kadanoff, L. P. (1976). Notes on Migdal's recursion formulas. *Annals of Physics* 100, 359–394.
- Yedidia, J. S., Freeman, W. T., Weiss, Y. (2005). Constructing free-energy approximations and generalized belief propagation algorithms. *IEEE Transactions on Information Theory* 51(7), 2282–2312.
- Mézard, M., Parisi, G., Zecchina, R. (2002). Analytic and algorithmic solution of random satisfiability problems. *Science* 297, 812–815.
- Braunstein, A., Mézard, M., Zecchina, R. (2005). Survey propagation: an algorithm for satisfiability. *Random Structures & Algorithms* 27(2), 201–226.
- Montanari, A., Ricci-Tersenghi, F., Semerjian, G. (2007). Solving constraint satisfaction problems through belief propagation-guided decimation. *Proc. 45th Allerton Conference*; arXiv:0709.1667.
- Ricci-Tersenghi, F., Semerjian, G. (2009). On the cavity method for decimated random constraint satisfaction problems and the analysis of belief propagation guided decimation algorithms. *J. Stat. Mech.* P09001; arXiv:0904.3395. ⚠ verify journal reference.
- Mézard, M., Montanari, A. (2009). *Information, Physics, and Computation*. Oxford University Press (BP, decimation, and the erasure decoder in one book).
- Luby, M., Mitzenmacher, M., Shokrollahi, M. A., Spielman, D. A. (2001). Efficient erasure correcting codes. *IEEE Transactions on Information Theory* 47(2), 569–584.
- Di, C., Proietti, D., Telatar, İ. E., Richardson, T. J., Urbanke, R. L. (2002). Finite-length analysis of low-density parity-check codes on the binary erasure channel. *IEEE Transactions on Information Theory* 48(6), 1570–1579.
- Richardson, T., Urbanke, R. (2008). *Modern Coding Theory*. Cambridge University Press (the peeling decoder, stopping sets, density evolution, thresholds).
- Sudoku-specific: Moon, T. K., Gunther, J. H. (2006), BP on Sudoku; Moon, Gunther, Kupin (2009), Sinkhorn solves Sudoku, *IEEE T-IT* 55(4), 1741–1746; Goldberger (2007), combined message passing for Sudoku ⚠; Davis et al. (2026), Lattice Deduction Transformers (a learned, sound candidate-set decimation with search).

## 4. The sentences the paper can carry

- "A recursive reasoner read as a decoder is either soft or decimating: it keeps the cells open and moves them together, or it commits and propagates. The loop decides which."
- "Decimation is irreversible and its failure is a contradiction downstream of a wrong fix; a learned decimating decoder inherits both, and its confidence does not report the fix that was wrong."
- "The solution is a fixed point of the map; the decimated cells are the survivors of the reduction; the decimating equilibrium cell is that map made symbol-equivariant."
