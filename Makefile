PYTHON ?= python
BOOK_DIR := docs/book
BOOK := 2026-10-08-固定测度Token级MirrorOPD

.PHONY: book test upstream data bootstrap check
book:
	cd $(BOOK_DIR) && latexmk -xelatex -interaction=nonstopmode -halt-on-error $(BOOK).tex

upstream:
	PYTHONPATH=src $(PYTHON) -m mirroropd backend --output upstream/TA-OPD-v2

data:
	python3 scripts/restore-data.py

test:
	$(PYTHON) -m pytest -q

bootstrap:
	npm ci
	bash scripts/agent-links.sh
	bash scripts/crossnote-links.sh
	pre-commit install
	git config blame.ignoreRevsFile .git-blame-ignore-revs
	python3 scripts/restore-data.py

check:
	npm run format:check
	npm run lint:md
	python3 scripts/check-yaml-comments.py
	ruff check .
	ruff format --check .
	pre-commit run --all-files
