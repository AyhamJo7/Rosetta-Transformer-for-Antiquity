# Makefile for Rosetta-Transformer-for-Antiquity

# --- Variables ---
PYTHON = python3
.PHONY: help install-dev format lint test ci

# --- Commands ---

help:
	@echo "Commands:"
	@echo "  install-dev   : Install development dependencies and pre-commit hooks."
	@echo "  format        : Format code with black and isort (via ruff)."
	@echo "  lint          : Lint code with ruff and mypy."
	@echo "  test          : Run all tests."
	@echo "  ci            : Run all CI checks (format, lint, test)."

install-dev:
	@echo "Installing development dependencies and pre-commit hooks..."
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements-dev.in
	$(PYTHON) -m pip install -r requirements.in
	pre-commit install

format:
	@echo "Formatting code..."
	black .
	ruff check --fix .
	ruff format .

lint:
	@echo "Linting code..."
	ruff check .
#	@echo "Running mypy for static type checking..."
#	mypy .

test:
	@echo "Running tests..."
	# pytest will be configured in pyproject.toml later
	$(PYTHON) -m pytest

ci:
	@echo "Running all CI checks..."
	$(MAKE) format
	$(MAKE) lint
	$(MAKE) test
