BOOK_DIR := docs/book
BOOK := 2026-10-08-固定测度Token级MirrorOPD

.PHONY: book test upstream bootstrap check
book:
	cd $(BOOK_DIR) && latexmk -xelatex -interaction=nonstopmode -halt-on-error $(BOOK).tex

upstream:
	bash scripts/bootstrap-upstream.sh

test:
	uv run --with torch --with numpy python experiments/test_mean_bpo.py
	uv run --with torch --with numpy python experiments/test_kl_estimators.py
	uv run --with torch --with numpy python experiments/test_opd_outer_inner.py
	uv run --with torch --with numpy python experiments/test_upstream_opd_semantics.py
	uv run --with torch --with numpy python experiments/test_upstream_mirror_wiring.py
	uv run --with torch --with numpy --with transformers python experiments/test_upstream_mirror_lm.py

bootstrap:
	npm ci
	bash scripts/agent-links.sh
	bash scripts/crossnote-links.sh
	pre-commit install
	git config blame.ignoreRevsFile .git-blame-ignore-revs

check:
	npm run format:check
	npm run lint:md
	python3 scripts/check-yaml-comments.py
	ruff check .
	ruff format --check .
	pre-commit run --all-files
