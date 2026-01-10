# Fraud Detection Demo v2 - Makefile

.PHONY: help generate-data up down build logs clean

# Default number of rows for data generation
ROWS ?= 1000000

help:
	@echo "Fraud Detection Demo v2"
	@echo ""
	@echo "Usage:"
	@echo "  make generate-data    Generate synthetic transaction data"
	@echo "  make up               Start all containers"
	@echo "  make down             Stop all containers"
	@echo "  make build            Build all containers"
	@echo "  make logs             View container logs"
	@echo "  make clean            Remove containers and data"
	@echo ""
	@echo "Options:"
	@echo "  ROWS=1000000          Number of rows to generate (default: 1M)"
	@echo ""
	@echo "Example:"
	@echo "  make generate-data ROWS=5000000"
	@echo "  make up"

# Generate synthetic data
generate-data:
	@echo "Generating $(ROWS) synthetic transactions..."
	@mkdir -p data
	python scripts/generate_data.py --rows $(ROWS) --output-dir ./data
	@echo "Data generation complete!"

# Start all containers
up:
	docker compose up --build -d
	@echo ""
	@echo "Dashboard available at: http://localhost:8080"
	@echo ""

# Stop all containers
down:
	docker compose down

# Build containers without starting
build:
	docker compose build

# View logs
logs:
	docker compose logs -f

logs-dashboard:
	docker compose logs -f dashboard

logs-cpu:
	docker compose logs -f worker-cpu

logs-gpu:
	docker compose logs -f worker-gpu

logs-triton:
	docker compose logs -f triton

# Clean up
clean:
	docker compose down -v --rmi local
	rm -rf data/cpu data/gpu models/fraud_xgboost_*

# Status
status:
	docker compose ps

# Restart a specific service
restart-%:
	docker compose restart $*

# Health check
health:
	@echo "Checking service health..."
	@curl -s http://localhost:8080/api/health | python -m json.tool || echo "Dashboard not responding"
