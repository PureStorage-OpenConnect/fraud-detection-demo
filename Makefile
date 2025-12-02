.PHONY: help build up down logs clean test status

help:
	@echo "NVIDIA Financial Fraud Detection Pipeline"
	@echo ""
	@echo "Available commands:"
	@echo "  make build    - Build all Docker containers"
	@echo "  make up       - Start all services"
	@echo "  make down     - Stop all services"
	@echo "  make logs     - View logs from all services"
	@echo "  make status   - Check status of all services"
	@echo "  make clean    - Remove all containers and images"
	@echo "  make test     - Run test inference request"
	@echo ""

build:
	@echo "Building all containers..."
	@chmod +x scripts/build_all.sh
	@./scripts/build_all.sh

up:
	@echo "Starting all services..."
	docker-compose up -d
	@echo ""
	@echo "Services started. Check status with: make status"
	@echo "View logs with: make logs"

down:
	@echo "Stopping all services..."
	docker-compose down

logs:
	docker-compose logs -f

status:
	@echo "Service Status:"
	@docker-compose ps
	@echo ""
	@echo "GPU Usage:"
	@docker exec fraud-detection-prep nvidia-smi 2>/dev/null || echo "GPU containers not running"

clean:
	@echo "Removing all containers and images..."
	docker-compose down -v --rmi all
	@echo "Cleanup complete"

test:
	@echo "Testing service endpoints..."
	@echo ""
	@echo "1. Testing Notification Service:"
	@curl -s -X GET http://localhost:5000/health | jq '.' || echo "Notification service not ready"
	@echo ""
	@echo "2. Testing Triton Inference Server:"
	@curl -s -X GET http://localhost:8002/v2/health/ready || echo "Inference service not ready"
	@echo ""
	@echo "3. Checking models loaded:"
	@curl -s -X GET http://localhost:8000/v2/models || echo "Cannot query models"
