NAME := 2026-10-15-MirrorOPD核心推导

.PHONY: all full core clean
all: full core
full:
	latexmk -xelatex -interaction=nonstopmode -halt-on-error $(NAME).tex
core: full
	latexmk -xelatex -interaction=nonstopmode -halt-on-error $(NAME)-正文.tex
clean:
	latexmk -c $(NAME).tex $(NAME)-正文.tex
