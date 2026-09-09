# Style checklist (run on every paragraph, every section, the whole paper)

## A. Every paragraph

- [ ] The first sentence states the paragraph's one claim or move, in plain words, ideally under twelve words.
- [ ] Each definition or equation is followed by a sentence saying what it means (the dissertation's "Non-trivial elements of E_AB represent ..." move).
- [ ] Each abstraction is followed by a concrete example or check ("As a concrete example, ...", "As a simple check, ...").
- [ ] The paragraph names its move when it starts one: "To resolve ..., we ...", "To extract ..., we ...".
- [ ] Every hedge is graded and specific ("it seems plausible", "we expect", "remains to be seen"); no bare "may".
- [ ] Every assumption is owned where it is made ("We will proceed assuming this is the case.").
- [ ] No filler adverbs (interestingly, notably, importantly, significantly, remarkably) unless the same sentence gives the reason.
- [ ] No hype words (novel, powerful, state of the art, impressive, breakthrough, outperform, beat).
- [ ] No program jargon (load-bearing, the field, suite, arm, battery, lens, verdict, stall, attractor, cold, port, lever, funnel, bank): see `VOICE_AND_STYLE.md` §5b for the replacements.
- [ ] No em-dashes; consequences joined with so / thus / therefore / because / which means; colons for consequence.
- [ ] Contractions: none in claims, results, abstract, captions; at most one per page in explanation.
- [ ] Citations are agents credited for a specific result ("Fredenhagen [45] proved that ..."), never bracket dumps.
- [ ] One analogy at most, one sentence long, precise, dropped immediately.
- [ ] Parentheses are short and stay inside the sentence; technical asides go to footnotes.
- [ ] The last sentence hands the reader to the next paragraph or section.

## B. Every number

- [ ] It came from an analysis script (`runs/analysis/*`); the appendix names the script and file.
- [ ] Its protocol (set, D, k, weights, numerics) is in the sentence or in the table's columns.
- [ ] A cross-system number also carries the training-regime column and the selection column.
- [ ] Coverage columns (verified@k, majority) are labeled coverage; the headline is the single pass.
- [ ] A contrast is stated with its noise floor and read only beyond twice the floor; inside the floor it is "flat".
- [ ] "record" appears only as "program-record".
- [ ] A number in prose changes what the reader concludes; otherwise it lives in the table.

## C. Every definition, instrument and law

- [ ] The definition is numbered (`definition` environment), names the set, the depth, the draws and the weights it is computed on.
- [ ] A meaning sentence follows ("a value of ... means the decoder ...").
- [ ] One known reading on a corpus grid is given as the calibration example.
- [ ] The law it feeds is named; the law is one sentence in a `law` environment with its scope and its evidence table or figure.
- [ ] The independent check is stated (a reproduction row, a cross-route row, a registered prediction, a test).
- [ ] The term appears in `TERMINOLOGY.md` with the same wording.

## D. Every section

- [ ] The opening paragraph says what the section does and how it is organized (two to three sentences).
- [ ] The closing paragraph connects forward ("This is the subject of §5.").
- [ ] Results sections use the bold run-in ledger for enumerated results ("**EMA.** ...").
- [ ] Speculation lives in its own labeled paragraph or subsection, with the evidence for and against and what would settle it.
- [ ] Every physics word in the section is in the RG correspondence table with an instrument beside it.
- [ ] Every figure caption lets the figure stand alone (what is plotted, on which set, what to see).

## E. The whole paper

- [ ] The abstract: two sentences of context, the objects named, at most three numbers with protocols, the laws named.
- [ ] The introduction opens with two facts and one inference, then the numbered contributions with section and table pointers, then Figure 1.
- [ ] The protocol-and-regime table appears once in §2 and every later number cites its columns.
- [ ] Related work is a narrative of results that ends at the gap.
- [ ] Limitations are the honest ledger with what would settle each item.
- [ ] The conclusion has one thesis sentence and one suggestive result.
- [ ] Conventions are fixed once and named as conventions.
- [ ] Symbols are defined in the "where" clause in the order they appear; no symbol is reused with two meanings.
- [ ] The compiled PDF builds on both toolchains (local `paper/build.sh`; Overleaf pdfLaTeX) with no warnings that change layout.

## F. LaTeX conventions (Overleaf-safe)

- Environments: `definition` and `law` (numbered, from `macros.tex`); refer with `\cref`.
- Tables: `booktabs` rules only; protocol columns first (set, D, k, weights), then the numbers; captions above tables, below figures.
- Numbers: percentages with two decimals in tables, one in prose; parameter counts with thousands separators; "pp" for percentage-point differences via `\pp`.
- Macros: `\dec`, `\trm`, `\hrm`, `\eqr`, `\sudext`, `\Dsteps{64}`; add new macros to `macros.tex` before use.
- Cross-references: sections `\cref{sec:...}`, tables `\cref{tab:...}`, figures `\cref{fig:...}`, laws `\cref{law:...}`; never hard-coded numbers.
- Figures as PDF (vector) under `figures/`, under 5 MB each; fonts embedded.
- No custom `.sty` files, no shell-escape, no fontspec; the venue switch `\venuetrue` swaps the style file only.
