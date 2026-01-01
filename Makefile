# Financial Fraud Detection Pipeline
# Pure Storage FlashBlade/FlashArray + NVIDIA GPU Demo

.PHONY: help build pipeline clean-data clean-all test inference stop

# Default paths (override with environment variables)
FB_DATA ?= /mnt/fsaai-shared/ebiser/fraud-data
FB_PREP ?= /mnt/fsaai-shared/ebiser/prep-output
MODEL_REPO ?= ./model_repository

# Pipeline settings
DURATION ?= 60
NUM_WORKERS ?= 64
MAX_FILES ?= 50

help:
	@echo "Financial Fraud Detection Pipeline"
	@echo ""
	@echo "Usage:"
	@echo "  make build       Build all containers"
	@echo "  make pipeline    Run full pipeline (pods 1-4)"
	@echo "  make inference   Start inference server"
	@echo "  make test        Test inference endpoint"
	@echo "  make stop        Stop all containers"
	@echo "  make clean-data  Remove generated data"
	@echo "  make clean-all   Full cleanup (data + images)"
	@echo ""
	@echo "Individual pods:"
	@echo "  make pod1        Run data generator"
	@echo "  make pod2        Run feature engineering"
	@echo "  make pod3        Run model training"
	@echo ""
	@echo "Configuration (set via environment):"
	@echo "  FB_DATA=$(FB_DATA)"
	@echo "  FB_PREP=$(FB_PREP)"
	@echo "  MODEL_REPO=$(MODEL_REPO)"
	@echo "  DURATION=$(DURATION)s NUM_WORKERS=$(NUM_WORKERS)"

build:
	@echo "Building all containers..."
	docker compose build

# Full pipeline: data generation → feature engineering → model training
pipeline: build
	@echo ""
	@echo "=========================================="
	@echo "Starting Full Pipeline"
	@echo "=========================================="
	@mkdir -p $(MODEL_REPO)
	@echo ""
	@echo "[1/3] Data Generation ($(DURATION)s)..."
	docker compose run --rm \
		-e DURATION_SECONDS=$(DURATION) \
		-e NUM_WORKERS=$(NUM_WORKERS) \
		data-gather
	@echo ""
	@echo "[2/3] Feature Engineering..."
	docker compose run --rm \
		-e MAX_FILES_PER_RUN=$(MAX_FILES) \
		data-prep
	@echo ""
	@echo "[3/3] Model Training..."
	docker compose run --rm model-build
	@echo ""
	@echo "=========================================="
	@echo "Pipeline Complete!"
	@echo "=========================================="
	@echo "Model saved to: $(MODEL_REPO)"
	@echo "Start inference: make inference"

# Individual pods
pod1:
	docker compose run --rm \
		-e DURATION_SECONDS=$(DURATION) \
		-e NUM_WORKERS=$(NUM_WORKERS) \
		data-gather

pod2:
	docker compose run --rm \
		-e MAX_FILES_PER_RUN=$(MAX_FILES) \
		data-prep

pod3:
	docker compose run --rm model-build

# Start inference server
inference:
	@echo "Starting Triton Inference Server..."
	docker compose up -d inference
	@echo "Waiting for server to be ready..."
	@sleep 10
	@curl -s http://localhost:8000/v2/health/ready && echo " Server ready!" || echo " Server starting..."
	@echo ""
	@echo "Endpoints:"
	@echo "  HTTP:    http://localhost:8000"
	@echo "  gRPC:    localhost:8001"
	@echo "  Metrics: http://localhost:8002"

# Test inference
test:
	@echo "Testing inference endpoint..."
	@curl -s -X POST http://localhost:8000/v2/models/fraud_xgboost/infer \
		-H "Content-Type: application/json" \
		-d '{"inputs": [{"name": "input__0", "shape": [1, 21], "datatype": "FP32", "data": [100.0, 35.0, -90.0, 50000, 1704067200, 35.1, -90.1, 12345, 30301, 4.6, 0.5, 12, 3, 0, 0, 10.5, 1, 10, 1, 10.8, 3]}]}' \
		| python3 -m json.tool
	@echo ""

# Stop all containers
stop:
	docker compose down

# Clean generated data (preserves images)
clean-data:
	@echo "Cleaning generated data..."
	sudo rm -rf $(FB_DATA)/run_*
	sudo rm -rf $(FB_PREP)/features_*.parquet
	sudo rm -rf $(FB_PREP)/metadata_*.json
	sudo rm -rf $(FB_PREP)/.prep_state.json
	rm -rf $(MODEL_REPO)
	@echo "Data cleaned"

# Full cleanup
clean-all: stop clean-data
	@echo "Removing Docker images..."
	docker compose down --rmi all -v
	docker builder prune -f
	@echo "Full cleanup complete"

# Quick demo (1 minute)
demo:
	@echo "Running quick demo (1 minute data generation)..."
	$(MAKE) pipeline DURATION=60 NUM_WORKERS=64 MAX_FILES=50