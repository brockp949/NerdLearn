.PHONY: help install dev build clean docker-up docker-down docker-logs worker-dev worker-logs

help:
	@echo "NerdLearn - AI-Powered Adaptive Learning Platform"
	@echo ""
	@echo "Available commands:"
	@echo "  make install      - Install all dependencies"
	@echo "  make dev          - Start development servers"
	@echo "  make build        - Build all applications"
	@echo "  make docker-up    - Start all Docker services"
	@echo "  make docker-down  - Stop all Docker services"
	@echo "  make docker-logs  - View Docker logs"
	@echo "  make worker-dev   - Start Celery worker locally"
	@echo "  make worker-logs  - View worker logs"
	@echo "  make clean        - Clean build artifacts"

install:
	@echo "Installing dependencies..."
	npm install
	cd apps/api && python -m venv venv && . venv/bin/activate && pip install -r requirements.txt
	cd apps/worker && pip install -r requirements.txt

dev:
	@echo "Starting development servers..."
	npm run dev

build:
	@echo "Building applications..."
	npm run build

docker-up:
	@echo "Starting Docker services..."
	docker-compose up -d

docker-down:
	@echo "Stopping Docker services..."
	docker-compose down

docker-logs:
	@echo "Viewing Docker logs..."
	docker-compose logs -f

worker-dev:
	@echo "Starting Celery worker..."
	cd apps/worker && celery -A app.celery_app worker --loglevel=info --concurrency=2

worker-logs:
	@echo "Viewing worker logs..."
	docker-compose logs -f worker

clean:
	@echo "Cleaning build artifacts..."
	rm -rf node_modules
	rm -rf apps/*/node_modules
	rm -rf apps/*/.next
	rm -rf apps/*/dist
	rm -rf apps/api/venv
	find . -type d -name "__pycache__" -exec rm -rf {} +
