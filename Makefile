.PHONY: up down etl dashboard test lint clean

up:
	docker-compose up -d

down:
	docker-compose down

etl:
	python -m src.main

dashboard:
	python -m src.dashboard.app

test:
	pytest

lint:
	ruff check . && black --check .

format:
	black . && ruff check --fix .

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
