PYTHON ?= python
BOOK_DIR := theory/book
BOOK := 2026-10-08-固定测度Token级MirrorOPD

.PHONY: book theory-core test upstream data bootstrap check check-docs check-maintenance test-maintenance
book:
	cd $(BOOK_DIR) && latexmk -bibtex- -xelatex -interaction=nonstopmode -halt-on-error $(BOOK).tex

theory-core:
	$(MAKE) -C theory/core

upstream:
	PYTHONPATH=src $(PYTHON) -m mirroropd backend --output upstream/TA-OPD-v2

data:
	python3 scripts/restore-data.py

test:
	$(PYTHON) -m pytest -q

bootstrap:
	npm ci
	$(PYTHON) -m pip install -r requirements-dev.txt
	$(PYTHON) scripts/setup.py
	bash scripts/crossnote-links.sh
	python3 scripts/restore-data.py

check-docs:
	$(PYTHON) scripts/run_checks.py --config scripts/checks.json --mode docs

test-maintenance:
	$(PYTHON) -m pytest -q tests/maintenance

check-maintenance:
	$(PYTHON) scripts/run_checks.py --config scripts/checks.json --mode maintenance

check: check-docs
	npm run format:check
	npm run lint:md
	python3 scripts/check-yaml-comments.py
	ruff check .
	ruff format --check .
	pre-commit run --all-files
