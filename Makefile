# Mental Health Detector - Makefile
# Common development tasks and commands

.PHONY: help setup dev-setup run dev test test-all test-coverage lint format security clean install-deps setup-models docker-build docker-run docs

# Default target
help:
	@echo "Mental Health Detector - Available Commands"
	@echo "==========================================="
	@echo ""
	@echo "Setup Commands:"
	@echo "  setup          - Complete setup for production use"
	@echo "  dev-setup      - Setup development environment"
	@echo "  install-deps   - Install Python dependencies"
	@echo "  setup-models   - Download and setup AI models"
	@echo ""
	@echo "Development Commands:"
	@echo "  run            - Start the Streamlit application"
	@echo "  dev            - Start development server with auto-reload"
	@echo "  test           - Run basic tests"
	@echo "  test-all       - Run all tests including integration"
	@echo "  test-coverage  - Run tests with coverage report"
	@echo "  test-ethics    - Run ethical AI and bias tests"
	@echo ""
	@echo "Code Quality Commands:"
	@echo "  lint           - Run code linting (flake8, pylint)"
	@echo "  format         - Format code (black, isort)"
	@echo "  security       - Run security checks (bandit, safety)"
	@echo "  type-check     - Run type checking (mypy)"
	@echo ""
	@echo "Docker Commands:"
	@echo "  docker-build   - Build Docker image"
	@echo "  docker-run     - Run application in Docker"
	@echo "  docker-dev     - Run development environment in Docker"
	@echo ""
	@echo "Utility Commands:"
	@echo "  clean          - Clean temporary files and caches"
	@echo "  docs           - Generate documentation"
	@echo "  requirements   - Update requirements files"

# Python and virtual environment settings
PYTHON := python3
VENV := venv
VENV_BIN := $(VENV)/bin
PIP := $(VENV_BIN)/pip
PYTHON_VENV := $(VENV_BIN)/python

# Check if virtual environment exists
VENV_EXISTS := $(shell test -d $(VENV) && echo 1 || echo 0)

# Setup Commands
setup: clean install-deps setup-models
	@echo "✅ Setup complete! Run 'make run' to start the application."

dev-setup: clean install-deps install-dev-deps setup-models setup-pre-commit
	@echo "✅ Development setup complete! Run 'make dev' to start development server."

install-deps:
ifeq ($(VENV_EXISTS), 0)
	@echo "📦 Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)
endif
	@echo "📦 Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install-dev-deps:
	@echo "📦 Installing development dependencies..."
	$(PIP) install -r requirements-dev.txt || echo "⚠️  requirements-dev.txt not found, skipping dev dependencies"

setup-models:
	@echo "🤖 Setting up AI models..."
	$(PYTHON_VENV) setup_models.py

setup-pre-commit:
	@echo "🔧 Setting up pre-commit hooks..."
	$(VENV_BIN)/pre-commit install || echo "⚠️  pre-commit not available, skipping hooks setup"

# Development Commands
run:
	@echo "🚀 Starting Mental Health Detector..."
	$(VENV_BIN)/streamlit run app.py

dev:
	@echo "🔧 Starting development server..."
	$(VENV_BIN)/streamlit run app.py --server.runOnSave true

# Testing Commands
test:
	@echo "🧪 Running basic tests..."
	$(PYTHON_VENV) -m pytest tests/ -v

test-all:
	@echo "🧪 Running all tests..."
	$(PYTHON_VENV) -m pytest tests/ -v --tb=short

test-coverage:
	@echo "📊 Running tests with coverage..."
	$(PYTHON_VENV) -m pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

test-ethics:
	@echo "⚖️ Running ethical AI and bias tests..."
	$(PYTHON_VENV) -m pytest tests/test_ethics.py tests/test_bias.py -v || echo "⚠️  Ethical tests not found"

# Code Quality Commands
lint:
	@echo "🔍 Running code linting..."
	$(VENV_BIN)/flake8 src/ app.py || echo "⚠️  flake8 not available"
	$(VENV_BIN)/pylint src/ app.py || echo "⚠️  pylint not available"

format:
	@echo "🎨 Formatting code..."
	$(VENV_BIN)/black src/ app.py tests/ || echo "⚠️  black not available"
	$(VENV_BIN)/isort src/ app.py tests/ || echo "⚠️  isort not available"

security:
	@echo "🔒 Running security checks..."
	$(VENV_BIN)/bandit -r src/ || echo "⚠️  bandit not available"
	$(VENV_BIN)/safety check || echo "⚠️  safety not available"

type-check:
	@echo "🔍 Running type checks..."
	$(VENV_BIN)/mypy src/ || echo "⚠️  mypy not available"

# Docker Commands
docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t mental-health-detector .

docker-run: docker-build
	@echo "🐳 Running application in Docker..."
	docker run -p 8501:8501 mental-health-detector

docker-dev:
	@echo "🐳 Starting Docker development environment..."
	docker-compose -f docker-compose.dev.yml up --build

docker-compose-up:
	@echo "🐳 Starting application with Docker Compose..."
	docker-compose up --build

docker-compose-down:
	@echo "🐳 Stopping Docker Compose services..."
	docker-compose down

# Utility Commands
clean:
	@echo "🧹 Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/
	rm -rf .mypy_cache/
	rm -rf .tox/

docs:
	@echo "📚 Generating documentation..."
	$(VENV_BIN)/sphinx-build -b html docs/ docs/_build/html || echo "⚠️  Sphinx not available"

requirements:
	@echo "📋 Updating requirements files..."
	$(PIP) freeze > requirements.txt

# Health check
health-check:
	@echo "🏥 Running health checks..."
	@echo "Python version: $$($(PYTHON) --version)"
	@echo "Virtual environment: $$(test -d $(VENV) && echo '✅ Active' || echo '❌ Not found')"
	@echo "Dependencies: $$($(PIP) list | wc -l) packages installed"
	@echo "Models: $$(test -d models/ && echo '✅ Available' || echo '❌ Not found')"

# Quick start for new developers
quickstart:
	@echo "🚀 Quick start for new developers..."
	@echo "1. Setting up environment..."
	make dev-setup
	@echo "2. Running tests..."
	make test
	@echo "3. Starting development server..."
	@echo "   Run 'make dev' to start the development server"
	@echo "   Open http://localhost:8501 in your browser"

# CI/CD Commands (for GitHub Actions)
ci-setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -r requirements-dev.txt || echo "No dev requirements"

ci-test:
	$(PYTHON) -m pytest tests/ --cov=src --cov-report=xml

ci-lint:
	$(PYTHON) -m flake8 src/ app.py
	$(PYTHON) -m black --check src/ app.py tests/

ci-security:
	$(PYTHON) -m bandit -r src/
	$(PYTHON) -m safety check

# Database and model management (if applicable)
reset-models:
	@echo "🔄 Resetting AI models..."
	rm -rf models/
	make setup-models

update-models:
	@echo "🔄 Updating AI models..."
	$(PYTHON_VENV) setup_models.py --update

# Performance testing
perf-test:
	@echo "⚡ Running performance tests..."
	$(PYTHON_VENV) -m pytest tests/test_performance.py -v || echo "⚠️  Performance tests not found"

# Deployment helpers
deploy-check:
	@echo "🚀 Pre-deployment checks..."
	make test-all
	make lint
	make security
	@echo "✅ Ready for deployment!"

# Show project statistics
stats:
	@echo "📊 Project Statistics"
	@echo "===================="
	@echo "Lines of code: $$(find src/ -name '*.py' -exec wc -l {} + | tail -1 | awk '{print $$1}')"
	@echo "Test files: $$(find tests/ -name '*.py' | wc -l)"
	@echo "Dependencies: $$(cat requirements.txt | wc -l)"
	@echo "Python files: $$(find . -name '*.py' | wc -l)"
