DOCUMENT := 2026-10-15-从密度比到α散度

.PHONY: all clean
all:
	latexmk -xelatex -interaction=nonstopmode -halt-on-error $(DOCUMENT).tex

clean:
	latexmk -c $(DOCUMENT).tex
