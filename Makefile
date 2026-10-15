.DEFAULT_GOAL := help

PYTHON ?= python
BACKEND ?= upstream/TA-OPD-v2
PACKAGE ?= transfers/code
RUN ?=
DEST ?=
MANIFEST ?=
HOPE_FILE ?= hope/mirroropd.hope
WANDB_RUN ?=

.PHONY: help
help:
	@printf '%s\n' \
	  '代码、日志与结果：' \
	  '  make code-pack BACKEND=... PACKAGE=transfers/code-001' \
	  '  make results-pack RUN=runs/... PACKAGE=transfers/result-001' \
	  '  make wandb-pack RUN=runs/... PACKAGE=transfers/wandb-001' \
	  '  make unpack MANIFEST=.../manifest.json DEST=imported/snapshot-001' \
	  '  make wandb-sync WANDB_RUN=.../offline-run-...  （显式上传到 W&B）' \
	  '集群与开发：' \
	  '  make hope-submit HOPE_FILE=hope/mirroropd.hope' \
	  '  make upstream | data | test | check | book' \
	  '通过 PYTHON=/path/to/python 选择环境；每个包和解包目录使用新路径。'

.PHONY: book theory-core test upstream data bootstrap check check-docs check-maintenance test-maintenance
book: theory-core

theory-core:
	$(MAKE) -C theory/core

upstream:
	PYTHONPATH=src $(PYTHON) -m mirroropd backend --output "$(BACKEND)"

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

.PHONY: code-pack results-pack wandb-pack wandb-sync unpack hope-submit
code-pack:
	$(PYTHON) -m mirroropd pack code --backend "$(BACKEND)" --output "$(PACKAGE)"

results-pack:
	$(PYTHON) -m mirroropd pack results --run "$(RUN)" --output "$(PACKAGE)"

wandb-pack:
	$(PYTHON) -m mirroropd pack wandb --run "$(RUN)" --output "$(PACKAGE)"

wandb-sync:
	@test -n "$(WANDB_RUN)" && test -d "$(WANDB_RUN)" || { echo "Set WANDB_RUN to an unpacked offline-run directory" >&2; exit 1; }
	$(PYTHON) -m wandb sync "$(WANDB_RUN)"

unpack:
	$(PYTHON) -m mirroropd unpack "$(MANIFEST)" --output "$(DEST)"

hope-submit:
	@test -f "$(HOPE_FILE)" || { echo "Hope file not found: $(HOPE_FILE)" >&2; exit 1; }
	@! grep -q 'REPLACE' "$(HOPE_FILE)" || { echo "Edit all REPLACE paths in $(HOPE_FILE) before submitting" >&2; exit 1; }
	cd "$(dir $(HOPE_FILE))" && hope run "$(notdir $(HOPE_FILE))"
