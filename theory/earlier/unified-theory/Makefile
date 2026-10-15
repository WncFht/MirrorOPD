BOOK = 2026-10-07-MirrorOPD-统一理论

.PHONY: all clean
all:
	xelatex -interaction=nonstopmode -halt-on-error "$(BOOK).tex"
	biber "$(BOOK)"
	xelatex -interaction=nonstopmode -halt-on-error "$(BOOK).tex"
	xelatex -interaction=nonstopmode -halt-on-error "$(BOOK).tex"

clean:
	rm -f *.aux *.bbl *.bcf *.blg *.log *.out *.run.xml *.toc
