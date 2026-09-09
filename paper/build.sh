#!/bin/bash
# Compile paper/draft/main.tex -> paper/paper.pdf. Prefers TeX Live's latexmk (pdfLaTeX, = Overleaf's toolchain);
# falls back to tectonic (XeTeX-based, self-contained, pulls packages on first use). Build files stay in paper/build/.
set -uo pipefail
cd "$(dirname "$0")" || exit 1
mkdir -p build
if command -v latexmk >/dev/null 2>&1 && command -v pdflatex >/dev/null 2>&1; then
  (cd draft && latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=../build main.tex) || exit 1
  cp build/main.pdf paper.pdf
elif command -v tectonic >/dev/null 2>&1; then
  tectonic --keep-logs --keep-intermediates -o build draft/main.tex || exit 1
  cp build/main.pdf paper.pdf
else
  echo "no TeX engine: brew install tectonic (no sudo) or install MacTeX/BasicTeX"; exit 2
fi
echo "built paper.pdf ($(du -h paper.pdf | cut -f1))"
