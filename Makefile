BOOK_DIR := docs/book
BOOK := 2026-10-08-固定测度Token级MirrorOPD

.PHONY: book test upstream data bootstrap check
book:
	cd $(BOOK_DIR) && latexmk -xelatex -interaction=nonstopmode -halt-on-error $(BOOK).tex

upstream:
	bash scripts/bootstrap-upstream.sh

data:
	python3 scripts/restore-data.py

test:
	PYTHONPATH=src uv run --with torch --with numpy python tests/test_math_evaluation.py
	python3 tests/test_kl_theory_identities.py
	python3 tests/test_kl_unified_framework.py
	PYTHONPATH=src python3 tests/test_gsm8k_evaluation.py
	PYTHONPATH=src uv run --with pytest python -m pytest tests/test_live_wandb_logging.py -q
	PYTHONPATH=src python3 tests/test_tracking_layout.py
	PYTHONPATH=src uv run --with torch --with numpy python tests/test_mdpo_eta_limits.py
	PYTHONPATH=src uv run --with torch --with numpy python tests/test_checkpoint_diagnostics.py
	PYTHONPATH=src uv run --with torch --with numpy python tests/test_topk_overlap.py
	PYTHONPATH=src uv run --with torch --with numpy python tests/test_training_metrics.py
	PYTHONPATH=src uv run --with torch --with numpy python tests/test_global_inner_writer.py
	PYTHONPATH=src uv run --with torch --with numpy --with pytest python -m pytest tests/test_rollout_support_alignment.py -q
	PYTHONPATH=src uv run --with torch --with numpy python tests/test_mean_bpo.py
	PYTHONPATH=src uv run --with torch --with numpy python tests/test_kl_estimators.py
	PYTHONPATH=src uv run --with torch --with numpy python tests/test_opd_outer_inner.py
	PYTHONPATH=src uv run --with torch --with numpy python tests/test_upstream_opd_semantics.py
	PYTHONPATH=src uv run --with torch --with numpy python tests/test_upstream_mirror_wiring.py
	PYTHONPATH=src uv run --with torch --with numpy --with transformers python tests/test_upstream_mirror_lm.py
	PYTHONPATH=src uv run --with torch --with numpy python tests/test_opd_mean_analysis.py
	PYTHONPATH=src uv run --with torch --with numpy --with transformers python tests/test_opd_lm.py

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
