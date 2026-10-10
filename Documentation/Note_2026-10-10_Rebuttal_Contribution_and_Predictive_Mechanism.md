# Rebuttal contribution and predictive mechanism experiments

2026-10-10. Saved at the PI's request for the discussion-period revision. This preserves the substance of the preceding contribution assessment and adds a proposed next experiment sequence. The proposal is not a preregistration, and no new inference or training was launched for it. The manuscript is unchanged.

**The research target.** Explain when additional recursion produces useful correction, identify a compact structure that predicts previously untested correction responses, and manipulate that structure to improve success. The current experiments establish causal constraints on this explanation; they have not yet recovered the proposed mechanism or a reliable inference controller.

**The original contribution supported by the current evidence.**

The current answer does not determine how repairable a computation is. Repairability depends on retained state, the configuration of decisions, and their continuing interaction. Interventions can affect the production and preservation of solutions differently. Here repairability means outcomes under a specified continuation budget and intervention protocol, rather than an internal quality assigned after observing success.

Four results jointly support that statement:

| Evidence | Scientific meaning |
| --- | --- |
| Score-preserving hidden-state changes alter subsequent success; selected same-puzzle transfers rescue preservation. | States displaying the same answer can have different computational futures. Holding the recipient's scores fixed separates this effect from changing its starting prediction. |
| Structured synthetic errors reproduce much of the repair difficulty of model-generated errors. | The organization of the error configuration matters; model-generated origin is not necessary. This does not identify the internal representation of that configuration. |
| Frozen communication can preserve existing solutions while eliminating or sharply reducing new completions. | Producing and maintaining a solution impose different demands on the recurrent computation. The intervention does not identify the semantic content of individual messages. |
| Grouped corrections induce repair at untouched cells, whereas random hidden perturbations change outcomes without a confirmed net improvement. | Coordinated correction is consequential; trajectory sensitivity alone does not provide useful control. The grouped corrections are oracle-guided diagnostic edits, not a deployable solver. |

The strongest causal control is preservation of current output scores. The strongest response measurement is beneficial change at decisions that were not edited. Together, these go beyond demonstrating that deleting memory is harmful or that perturbations propagate through a network.

The local commitment index did not provide a useful control variable. A tested simple factorization of the linear response operator left substantial structure unexplained in the earlier exploratory audit; that fit needs reproduction before manuscript use. Random-perturbation gains did not survive fresh confirmation. These findings constrain possible mechanisms but do not prove that no compact mechanism exists.

The architecture-by-dropout comparisons additionally connect productive recursion to both architecture and training. Dropout substantially reduces the one-carry versus two-carry performance gap, while paired shared failures still show differences in later completion. These are single-seed comparisons per training arm, not a complete attribution of the architectural advantage.

Broad priority claims about memory, iterative refinement, state interventions, causal propagation, attractors, or noisy search are inappropriate. Close work already includes [recurrent-state interventions](https://arxiv.org/html/2609.22197), [interaction locality across tasks](https://arxiv.org/html/2605.20784v1), and [stochastic recurrent search](https://arxiv.org/html/2605.19943). Our strongest originality claim is the particular combination of controlled causal contrasts and measured outcomes, not being the first to study these general ideas. This is a scoped literature assessment rather than an exhaustive priority guarantee.

**The generalizable hypothesis and the standard for establishing it.**

Write the recurrence as z_next = F_x(z), with current output scores s = R(z). The finite response to a controlled intervention I_u is

    C(z, u, k) = R(F_x^k(I_u(z))) - R(F_x^k(z)).

For interventions intended to isolate retained information beyond the answer, require R(I_u(z)) = R(z), to a declared and checked numerical tolerance. For linear readouts, a null-space projection supplies this control. Nonlinear readouts require verification after the finite intervention; a local derivative constraint alone is insufficient.

The proposed hypothesis is that recurrence learns a state-dependent rule for coordinating revisions across dependent decisions. Productive continuation depends on that rule supporting compatible corrections beyond what current answer quality predicts. State dependence and interacting corrections are supported; a compact representation of the coordination rule is not established.

Defining C is automatic for any recurrent system. The discovery would be compact, interpretable structure that predicts unseen finite responses and can be manipulated causally. A complete demonstration would connect:

1. A component or feature measured before the outcome.
2. Predictions of which new edits will be retained and which other decisions will change.
3. A controlled change to that feature that changes responses as predicted.
4. Better completion or preservation on fresh examples, with the cost of identification and intervention included.

Across architectures, the experimental procedure and predictive relationship should transfer. Hidden coordinates and fitted weights need not match. A successful score predictor alone would remain a predictor; an intervention that changes only its score without changing measured success would fail the control claim.

**What the saved bank actually contains.**

An inventory on 2026-10-10 found the following on Attention 256, excluding smoke files:

| Bank | Saved coverage | Reusable information |
| --- | --- | --- |
| P15 | 256 puzzles, 395 states, 11,521 variants | Finite display-edit responses, four subsequent predictions and score arrays, intervention metadata, own no-edit references. |
| P16 | 256 puzzles, 395 states, 22,119 variants | Responses to different state-edit channels, current margins, the tested commitment index, four subsequent predictions and score arrays. |
| P18 | 173 puzzles, 262 states, 2,539 variants | Group-edit comparisons and their recorded outcomes. |
| P19 | 177 puzzles, 268 states, 5,360 variants | Edited masks, initial grids, wrong-cell counts and four subsequent predicted grids. |
| P22 and its replications | Eight large-angle continuations and four continuations at each smaller angle per retained puzzle | Per-iteration exactness flags, IDs, seeds and protocol metadata. |

These are correlated variants of relatively few puzzles, not tens of thousands of independent observations. P19 saves predicted grids rather than hidden states or continuous response scores. P22 saves success flags rather than the branching hidden states. Prefixes can be replayed to extract state features, but this is additional computation with replay checks. The old outcomes have been inspected, so any new fitting on them is exploratory even if an earlier column was called CONFIRMATION. Prospective confirmation needs fresh puzzles and frozen analysis choices.

P22 estimates a local success frequency around one base state per puzzle under a particular perturbation distribution. It cannot by itself distinguish intrinsic puzzle difficulty from a manipulable state property. Eight branches also give noisy per-state estimates. Treat the binomial outcomes directly rather than treating their empirical fraction as an exact label.

**Recommended sequence A: identify the smallest predictive explanation from existing responses.**

Use the existing finite responses to compare three explicit explanations:

| Candidate explanation | Predictive content |
| --- | --- |
| Independent local correction | Group effects are predicted from individual edit effects, where the bank contains the matching single edits. |
| Global amount of correction | Initial quality, edit count or fraction, time and visible compatibility explain the response; which equally sized set was edited adds little. |
| State-dependent interactions | The response depends on which edits occur together, and compact state features improve prediction beyond the visible configuration and edit count. |

Primary targets should include the signed change in correctness at unedited cells and next-step completion, each relative to its own no-edit continuation. Use continuous score contrasts where stored, alongside discrete outcomes, to distinguish an interaction in computation from an argmax threshold effect. The sparse single-edit bank does not identify every entry of a full interaction matrix; missing paired probes must be measured prospectively.

Keep every state and variant of a puzzle in one split. Use puzzle-level nested validation within this exploratory analysis; report uncertainty by resampling puzzles. Compare with current-margin, visible-conflict, iteration and edit-count baselines. Separate early correction states from late unsolved states. Ground-truth error counts are permitted in a diagnostic model but must be excluded from a proposed inference controller. Reconstruct prefixes only if an output-based model leaves a predictive gap worth testing with hidden features.

Fit a small model before an expressive encoder: an additive model, a monotone count-response model, then a regularized model with a few interaction factors. Test their predictions on held-out edited sets and held-out puzzles. Any low-rank representation must predict new interventions; a compact visualization of already observed responses is insufficient.

Interpretation: if amount of correction explains the data, pursue a collective reconstruction law. If edit identity and state-dependent interactions add reproducible predictive information, pursue a compact coupling operator. If neither generalizes, do not relabel a flexible fit as a theory. This first analysis selects the next intervention rather than producing another catalogue of descriptive metrics.

**Recommended sequence B: measure and transfer a correction response.**

This is the highest-priority new causal experiment. The earlier P16 response-transfer objective was withdrawn or deferred; P9/P11 transferred natural continuation behavior, not a response rule evaluated on new challenges.

Begin with a small discovery pilot on Attention 256. Use Attention 192 for a frozen replication and a banked MLP checkpoint for a subsequent architecture check. The P9/P11 same-puzzle pairs are a useful positive-control setting for preservation, but the main correction experiment must include unsolved states.

For a given state, choose disjoint edit sets A and B and run no edit, A alone, B alone and A union B. Evaluate every arm on the same cells outside A union B. The interaction is

    I(A, B) = Y(A union B) - Y(A) - Y(B) + Y(no edit),

where Y is the same response measure on that common evaluation set. This supplies the paired interaction contrasts that two independent random draws at each correction fraction do not provide. Match edit volume and, where feasible, perturbation magnitude; include correct and incorrect edits, plus controls that disrupt comparable amounts of state. Do not assume Sudoku peer relations define the learned interaction groups: P18 did not support that shortcut.

Then cross the probe panel with a controlled hidden-state transfer between histories of the same puzzle. Keep the recipient's output scores fixed. Test the invisible slow component first, with a limited fast-state factorial if the discovery evidence requires it. Prefer histories with matched current outputs and time where available. Hybrids can still be atypical states, so include self-transfer, restoration, unrelated or shuffled-component, and magnitude-matched controls. Verify the complete designated readout after every intervention.

Crucially, subtract each hidden condition's own unchallenged continuation. A transfer that changes the baseline drift is not, by itself, evidence that it transfers the response to an edit. Also avoid testing only binary preservation where a recipient already fails without a challenge: that creates a floor. Use the full signed response profile and report completion separately.

Identify a candidate component using one set of probes. Freeze it, then ask whether it predicts and transfers the responses to different cells, digits and grouped edits on fresh puzzles. Reduce the transferred component to a small rank or a small set of factors, compare against matched random components, and report the amount needed. Transfer of a whole hidden tensor establishes a coarse functional dependence; compact selective transfer would be the stronger mechanism result.

A useful initial scope is 16–32 discovery states and a bounded panel of finite probes, with one- and four-iteration horizons. Freeze the precise selection, sample size and stopping conditions in a separate registration before collecting outcomes. Add more states, rather than more variants of the same few states, for confirmation.

Interpretation: successful transfer would show that a retained component controls how new evidence or a changed decision is used. Failure would separate the known effect on natural continuation from the stronger claim that a compact response rule can be transferred.

**Recommended sequence C: turn the predicted response into beneficial control.**

Once a compact predictor survives new probes, choose a state intervention from that predictor without consulting the solution. A natural initial implementation is a bounded edit within the identified score-preserving subspace; use a finite candidate set or a trust region before attempting unrestricted gradient optimization.

Compare the predicted beneficial direction with its opposite, a magnitude-matched random direction, intact continuation and restoration. Measure both the predicted intermediate change in the correction response and actual completion. This distinguishes successful manipulation of the proposed mechanism from generic disruption or a misleading predictor increase.

For statistical continuation measurements, define

    q_B(z; D) = P(complete within B iterations after an initial perturbation drawn from D around z).

The probability is induced by the specified initial perturbation distribution; the unperturbed model remains deterministic. Match D and the horizon across conditions. Crucially, compare multiple manipulated states of the SAME puzzle with the same current readout. This addresses the puzzle-difficulty confound in fitting a predictor to P22 alone. Use fresh independent branches for evaluation, including unperturbed continuation as a separate endpoint; never select a favorable direction using the branches that score it. A small branch count is suitable for discovery, with confirmation size chosen from the paired variance and a prespecified useful effect.

The decisive chain is: measured component -> predicted response change -> observed response change -> greater success. A rise in q_B can be a scientific control result before it becomes an efficient solver. Claim an inference improvement only if the complete policy beats intact continuation and a strong random-branch baseline at matched model work, including probe, branching and controller overhead. Report runtime as well as arithmetic, and keep discovery compute separate from deployment cost.

Simply optimizing a correctness head is not an unexplored alternative: [Probabilistic TRM Appendix C](https://arxiv.org/html/2605.19943#A3) reports no additional benefit from Q-guided gradients over matched noise. Our proposed test targets experimentally identified future response structure and verifies its causal effect.

**Replication and the move to ARC.**

Freeze the identification procedure before replication. Replicating across widths is useful but does not substitute for a different recurrence, token mixer or training seed. Use banked MLP seeds and the one-carry/two-carry models where their interfaces permit the same intervention definition. New training is warranted only after a result needs an independent-seed test unavailable in the bank.

For ARC, preserve the procedure: matched present outputs, finite structured probes, predictions of changes at untouched decisions, response transfer, then useful control. Begin with controlled task families whose generating transformation is known. Compare random pixel errors with a coherent alternative transformation; measure whether changing one object alters untouched objects appropriately. A task interpretation should not be assumed to occupy a particular hidden vector before evidence identifies it.

Ground truth may define diagnostic edits and evaluate them. A deployed policy cannot use a held-out query answer. Keep demonstration exposure, held-out queries and unseen task families explicit. Transfer of an identification-and-control procedure is the objective; identical latent coordinates or Sudoku constraint matrices are not required.

**Compute and decision gates.**

The first model comparisons need only saved arrays. Prefix replay, a bounded finite-response pilot and a first intervention test should be feasible locally. Existing Attention 256 P19 chunks recorded about 2,230 seconds for 5,360 four-iteration variants; the three-width P22 queue ran for about 102 minutes. These are historical timings under their own batch and load conditions, not an ETA for new code. Time a numerical smoke before estimating the new workload.

Use a pod for a locked large confirmation set, additional training seeds, or repeated long continuations that exceed the practical local budget. There is no scientific reason to train new models before testing whether the existing checkpoints contain a compact predictive and controllable response.

Before any new scientific run, register the populations, discovery/confirmation division, hypotheses and credences, primary contrasts, effect thresholds, stopping rules and wording for each outcome. Use fresh confirmation puzzles because the old bank has already informed the hypothesis. Keep runners immutable during runs; derivative instrumentation must separately pass forward and finite-difference checks. The known training-time gradient stops make unverified automatic derivatives unsuitable. Gate each backend and batch configuration against its own baseline rather than assuming CPU trajectories reproduce accelerator trajectories.

Do not prioritize another undirected perturbation sweep, a larger collection of attractor visualizations, a Sudoku-only dependency closure guessed in advance, or a success classifier evaluated only across different puzzles. Each could consume substantial compute without closing the missing causal link.

**Existing records.**

- [Current experiment synthesis](Note_2026-10-03_Rebuttal_Synthesis_P15_P22.md)
- [P15–P17 designs, amendments and outcomes](Note_2026-10-02_Rebuttal_P15_P16_P17_Registration.md)
- [P19 grouped corrections](Note_2026-10-03_Rebuttal_P19_Registration.md)
- [P22 continuation statistics and scope correction](Note_2026-10-03_Rebuttal_P22_Registration.md)
- [P24 recurrence comparison and corrections](Note_2026-10-03_Rebuttal_P24_Registration.md)
- [P9/P11 corrected control accounting](../runs/analysis/rebuttal_20261001g/controls_correction_20261008.txt)
