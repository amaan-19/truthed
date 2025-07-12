.PHONY: install install-dev setup test lint format clean docker-build docker-up docker-down

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install
	python -m spacy download en_core_web_lg

setup: install-dev
	python scripts/setup_database.py
	python -m spacy download en_core_web_lg

# Testing
test:
	pytest tests/ -v --cov=src/truthed

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

# Code quality
lint:
	flake8 src/ tests/
	mypy src/

format:
	black src/ tests/
	isort src/ tests/

# Data and models
collect-data:
	python scripts/collect_data.py

train-models:
	python scripts/train_models.py

evaluate:
	python scripts/evaluate_system.py

# Docker
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

# Database
db-migrate:
	alembic upgrade head

db-reset:
	alembic downgrade base
	alembic upgrade head

# Development
dev:
	uvicorn src.truthed.api.main:app --reload --host 0.0.0.0 --port 8000

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +

# Web development
run-web:
	cd web && python app.py

dev-web:
	cd web && FLASK_ENV=development FLASK_DEBUG=1 python app.py

build-web:
	cd web && python -m py_compile app.py

# Combined development (your existing pipeline + web)
dev-full:
	make install-dev
	make setup
	make run-web

# Testing web interface
test-web:
	cd web && python -c "import app; print('✅ Web app imports successfully')"

# Clean web cache
clean-web:
	find web -name "*.pyc" -delete
	find web -name "__pycache__" -delete

# Update existing clean target
clean: clean-web
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +