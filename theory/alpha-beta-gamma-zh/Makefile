DOCUMENT := 2026-10-15-α、β和γ散度族

.PHONY: all english clean
all:
	latexmk -xelatex -interaction=nonstopmode -halt-on-error $(DOCUMENT).tex

english:
	$(MAKE) -C original/recovered-latex

clean:
	latexmk -c $(DOCUMENT).tex
	$(MAKE) -C original/recovered-latex clean
