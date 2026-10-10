.PHONY: help install download train api test lint docker-up docker-down clean

help:  ## Show this help message
	@echo "🎯 SentimentScope - Available commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	pip install -r requirements.txt

download:  ## Download dataset (default: imdb, 5000 samples)
	python src/data/download.py --dataset imdb --num-samples 5000

train:  ## Train the model
	python src/train.py --dataset imdb --epochs 3 --batch-size 16

api:  ## Start the API server
	uvicorn app.main:app --reload --port 8000

mlflow:  ## Start MLflow UI
	mlflow server --host 0.0.0.0 --port 5000

dashboard:  ## Start Streamlit dashboard
	streamlit run dashboard/streamlit_app.py

test:  ## Run all tests with coverage
	pytest tests/ -v --cov=src --cov=app --cov-report=term-missing

lint:  ## Run code linting
	black --check src/ app/ tests/
	isort --check-only src/ app/ tests/
	flake8 src/ app/ tests/

format:  ## Format code
	black src/ app/ tests/
	isort src/ app/ tests/

docker-up:  ## Start all services with Docker
	docker compose -f docker/docker-compose.yml up -d

docker-down:  ## Stop all Docker services
	docker compose -f docker/docker-compose.yml down

clean:  ## Clean up generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .coverage htmlcov
