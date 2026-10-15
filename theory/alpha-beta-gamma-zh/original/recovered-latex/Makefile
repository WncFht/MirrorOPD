DOCUMENT := 2026-10-15-families-of-alpha-beta-and-gamma-divergences

.PHONY: all clean
all:
	latexmk -xelatex -interaction=nonstopmode -halt-on-error $(DOCUMENT).tex

clean:
	latexmk -c $(DOCUMENT).tex
