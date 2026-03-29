PYTHON ?= python

SOURCE_DIRS := moozy
CLEAN_DIRS := .mypy_cache .ruff_cache build dist .eggs
CLEAN_FILES :=

.DEFAULT_GOAL := help

.PHONY: help bootstrap install-dev install-hooks doctor format format-check lint typecheck compile pre-commit check ci fix clean

help:
	@printf '%s\n' \
		'Available targets:' \
		'  make bootstrap      Install dev dependencies and git hooks' \
		'  make install-dev    Install the package with dev dependencies into the active env' \
		'  make install-hooks  Install pre-commit git hooks' \
		'  make doctor         Show python and tool versions from the active env' \
		'  make format         Auto-format code with Ruff' \
		'  make format-check   Verify formatting without changing files' \
		'  make lint           Run Ruff lint checks' \
		'  make typecheck      Run mypy on configured typed boundaries' \
		'  make compile        Verify project imports compile' \
		'  make pre-commit     Run configured pre-commit hooks on all files' \
		'  make check          Run the full local validation suite' \
		'  make fix            Apply Ruff fixes, then rerun the full check suite' \
		'  make clean          Remove caches and local build artifacts'

bootstrap: install-dev install-hooks

install-dev:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

install-hooks:
	$(PYTHON) -m pre_commit install

doctor:
	@printf 'python: '
	@$(PYTHON) --version
	@printf 'pip: '
	@$(PYTHON) -m pip --version
	@printf 'ruff: '
	@$(PYTHON) -m ruff --version
	@printf 'mypy: '
	@$(PYTHON) -m mypy --version
	@printf 'pre-commit: '
	@$(PYTHON) -m pre_commit --version

format:
	$(PYTHON) -m ruff format .

format-check:
	$(PYTHON) -m ruff format . --check

lint:
	$(PYTHON) -m ruff check .

typecheck:
	$(PYTHON) -m mypy --config-file pyproject.toml

compile:
	$(PYTHON) -m compileall $(SOURCE_DIRS)

pre-commit:
	$(PYTHON) -m pre_commit run --all-files

check: format-check lint typecheck compile

ci: check

fix:
	$(PYTHON) -m ruff check . --fix
	$(PYTHON) -m ruff format .
	$(MAKE) check

clean:
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	find . -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
	rm -rf $(CLEAN_DIRS) $(CLEAN_FILES) *.egg-info
