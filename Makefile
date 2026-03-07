.PHONY: setup up up-tools up-monitoring down dev-backend dev-frontend ingest test lint

# First-time setup
setup:
	@if [ ! -f .env ]; then cp .env.example .env && echo "Created .env — fill in OPENAI_API_KEY and JWT_SECRET_KEY"; fi
	@if [ ! -f chatbot-ui/.env.local ]; then cp chatbot-ui/.env.example chatbot-ui/.env.local && echo "Created chatbot-ui/.env.local"; fi
	pip install -e ".[dev]"
	cd chatbot-ui && npm install

# Docker services
up:
	docker-compose -f deployment/docker-compose.yml up -d

up-tools:
	docker-compose -f deployment/docker-compose.yml --profile tools up -d

up-monitoring:
	docker-compose -f deployment/docker-compose.monitoring.yml up -d

down:
	docker-compose -f deployment/docker-compose.yml down
	docker-compose -f deployment/docker-compose.monitoring.yml down 2>/dev/null || true

# Local dev servers
dev-backend:
	uvicorn src.api.main:app --reload

dev-frontend:
	cd chatbot-ui && npm run dev

# Data ingestion
ingest:
	python scripts/run_ingestion.py data/sample/ --skip-embeddings

# Testing & quality
test:
	pytest tests/ -v --cov=src

lint:
	ruff check . --fix
	mypy src/
