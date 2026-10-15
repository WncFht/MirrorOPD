PYTHON ?= python
BACKEND ?= upstream/TA-OPD-v2
PACKAGE ?= transfers/code
RUN ?=
DEST ?=
MANIFEST ?=
HOPE_FILE ?= hope/mirroropd.hope

.PHONY: book theory-core test upstream data bootstrap check check-docs check-maintenance test-maintenance
book: theory-core

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

.PHONY: code-pack results-pack wandb-pack unpack hope-submit
code-pack:
	$(PYTHON) -m mirroropd pack code --backend "$(BACKEND)" --output "$(PACKAGE)"

results-pack:
	$(PYTHON) -m mirroropd pack results --run "$(RUN)" --output "$(PACKAGE)"

wandb-pack:
	$(PYTHON) -m mirroropd pack wandb --run "$(RUN)" --output "$(PACKAGE)"

unpack:
	$(PYTHON) -m mirroropd unpack "$(MANIFEST)" --output "$(DEST)"

hope-submit:
	@test -f "$(HOPE_FILE)" || { echo "Hope file not found: $(HOPE_FILE)" >&2; exit 1; }
	@! grep -q 'REPLACE' "$(HOPE_FILE)" || { echo "Edit all REPLACE paths in $(HOPE_FILE) before submitting" >&2; exit 1; }
	cd "$(dir $(HOPE_FILE))" && hope run "$(notdir $(HOPE_FILE))"
