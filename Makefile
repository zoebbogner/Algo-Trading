.PHONY: help install setup test lint format clean run-backtest run-live run-setup validate

help: ## Show this help message
	@echo "Algo-Trading Development Commands"
	@echo "=================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies with Poetry
	poetry install --with dev,llm

setup: ## Setup the trading system
	poetry run python -m src.app.main setup

test: ## Run tests with coverage
	poetry run pytest tests/ --cov=src --cov-report=term-missing --cov-report=html

test-fast: ## Run tests without coverage
	poetry run pytest tests/ -v

lint: ## Run all linting and type checking
	poetry run black --check --diff src/ tests/
	poetry run ruff check src/ tests/
	poetry run isort --check-only --diff src/ tests/
	poetry run mypy src/

format: ## Format code with Black and isort
	poetry run black src/ tests/
	poetry run isort src/ tests/

format-check: ## Check code formatting
	poetry run black --check --diff src/ tests/
	poetry run isort --check-only --diff src/ tests/

security: ## Run security checks
	poetry run bandit -r src/ -f json -o bandit-report.json
	poetry run detect-secrets scan --baseline .secrets.baseline

pre-commit: ## Run pre-commit hooks
	poetry run pre-commit run --all-files

clean: ## Clean up generated files
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

run-setup: ## Run system setup
	poetry run python -m src.app.main setup

run-validate: ## Validate configuration
	poetry run python -m src.app.main validate-config

run-status: ## Show system status
	poetry run python -m src.app.main status

run-backtest: ## Run backtesting (SIM mode)
	poetry run python -m src.app.main backtest --mode SIM --symbols "BTC/USD,ETH/USD" --duration 1d

run-live-sim: ## Start live trading in SIM mode
	poetry run python -m src.app.main live --mode SIM --symbols "BTC/USD,ETH/USD"

run-live-paper: ## Start live trading in PAPER mode
	poetry run python -m src.app.main live --mode PAPER --symbols "BTC/USD,ETH/USD"

test-setup: ## Run setup verification script
	python scripts/test_setup.py

dev-install: ## Install development dependencies
	poetry install --with dev

llm-install: ## Install LLM dependencies
	poetry install --with llm

all-install: ## Install all dependencies
	poetry install --with dev,llm

check-deps: ## Check dependency status
	poetry show --tree

update-deps: ## Update dependencies
	poetry update

lock-deps: ## Lock dependencies
	poetry lock

build: ## Build the package
	poetry build

publish: ## Publish to PyPI (dry run)
	poetry publish --dry-run

docs: ## Generate documentation
	@echo "Documentation generation not yet implemented"

docker-build: ## Build Docker image
	docker build -t algo-trading .

docker-run: ## Run Docker container
	docker run -it --rm algo-trading

docker-test: ## Run tests in Docker
	docker run -it --rm algo-trading make test
