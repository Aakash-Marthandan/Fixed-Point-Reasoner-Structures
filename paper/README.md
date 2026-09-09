# paper/ — the ICLR 2027 draft (AAMAS fallback)

- `draft/main.tex` is the root document; sections live in `draft/sections/`, figures in `draft/figures/`, references in `draft/refs.bib`, macros in `draft/macros.tex`.
- Build locally with `bash paper/build.sh` (or `make -C paper`); the PDF lands at `paper/paper.pdf`, intermediates in `paper/build/`.
- Overleaf: `make -C paper overleaf` zips `draft/` (upload as a new project, root document `main.tex`, compiler pdfLaTeX). Every package is in TeX Live; no shell-escape, no custom style files, no local fonts.
- Venue style: drop `iclr2027_conference.sty` (+ `.bst`) into `draft/` and set `\venuetrue` in `main.tex`; the preprint layout (geometry + natbib) is the default until then.
- `documentation/claims.tex` is the claim ledger the abstract and introduction draw from (not compiled).
- `documentation/` holds the writing documentation: the voice/style/tone markers from the PI's dissertation (`VOICE_AND_STYLE.md`), the paper blueprint (`PAPER_BLUEPRINT.md`), the fixed vocabulary (`TERMINOLOGY.md`) and the pre-commit checklist (`STYLE_CHECKLIST.md`). Every section is drafted and revised against them.
