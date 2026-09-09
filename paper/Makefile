.PHONY: all clean overleaf
all:
	bash build.sh
clean:
	rm -rf build paper.pdf
overleaf:
	rm -f overleaf_draft.zip && cd draft && zip -r ../overleaf_draft.zip . -x '*.aux' '*.log' '*.out' '*.bbl' '*.blg' '*.fls' '*.fdb_latexmk' '*.synctex.gz'
	@echo "upload paper/overleaf_draft.zip to Overleaf (New project -> Upload); root document: main.tex"
