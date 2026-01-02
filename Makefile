# Fraud Detection Demo
# Pure Storage FlashBlade/FlashArray + NVIDIA GPU
#
# Configuration: All paths defined in .env (single source of truth)

# Load .env if it exists
-include .env
export

# Fallback defaults (if .env is missing)
FB_MOUNT ?= /mnt/fsaai-shared/ebiser
FB_DATA ?= $(FB_MOUNT)/fraud-data
FB_PREP ?= $(FB_MOUNT)/prep-output
FA_MODEL_REPO ?= $(PWD)/model_repository

# Pipeline settings (can override via command line or .env)
DURATION_SECONDS ?= 60
NUM_WORKERS ?= 128
MAX_FILES_PER_RUN ?= 100
CHUNK_SIZE ?= 1000000
FRAUD_RATE ?= 0.005

.PHONY: help build pipeline clean-data clean-all test inference stop env-check

help:
	@echo "Fraud Detection Demo"
	@echo ""
	@echo "Usage:"
	@echo "  make build       Build all containers"
	@echo "  make pipeline    Run full pipeline (pods 1-3)"
	@echo "  make inference   Start inference server (pod 4)"
	@echo "  make test        Test inference endpoint"
	@echo "  make stop        Stop all containers"
	@echo "  make clean-data  Remove generated data"
	@echo "  make clean-all   Full cleanup (data + images)"
	@echo "  make env-check   Verify path configuration"
	@echo ""
	@echo "Individual pods:"
	@echo "  make pod1        Run data generator"
	@echo "  make pod2        Run feature engineering (CPU vs GPU comparison)"
	@echo "  make pod3        Run model training"
	@echo ""
	@echo "Configuration (from .env):"
	@echo "  FB_DATA=$(FB_DATA)"
	@echo "  FB_PREP=$(FB_PREP)"
	@echo "  FA_MODEL_REPO=$(FA_MODEL_REPO)"
	@echo "  DURATION_SECONDS=$(DURATION_SECONDS)s NUM_WORKERS=$(NUM_WORKERS)"

# Verify environment and paths
env-check:
	@echo "=== Path Configuration Check ==="
	@echo ""
	@echo "FlashBlade paths:"
	@echo "  FB_DATA: $(FB_DATA)"
	@test -d $(FB_DATA) && echo "    ✓ exists" || echo "    ✗ MISSING - run: sudo mkdir -p $(FB_DATA)"
	@echo "  FB_PREP: $(FB_PREP)"
	@test -d $(FB_PREP) && echo "    ✓ exists" || echo "    ✗ MISSING - run: sudo mkdir -p $(FB_PREP)"
	@echo ""
	@echo "Model repository:"
	@echo "  FA_MODEL_REPO: $(FA_MODEL_REPO)"
	@test -d $(FA_MODEL_REPO) && echo "    ✓ exists" || echo "    ✗ MISSING - will be created during pipeline"
	@echo ""
	@echo "Data flow:"
	@echo "  Pod 1 → $(FB_DATA)/run_*/*.parquet"
	@echo "  Pod 2 → $(FB_PREP)/features_*.parquet"
	@echo "  Pod 3 → $(FA_MODEL_REPO)/fraud_xgboost/"
	@echo "  Pod 4 ← $(FA_MODEL_REPO)/fraud_xgboost/"

build:
	@echo "Building all containers..."
	docker compose build

# Full pipeline: data generation → feature engineering → model training
pipeline: build
	@echo ""
	@echo "=========================================="
	@echo "Starting Full Pipeline"
	@echo "=========================================="
	@echo "Paths:"
	@echo "  FB_DATA:    $(FB_DATA)"
	@echo "  FB_PREP:    $(FB_PREP)"
	@echo "  FA_MODEL_REPO: $(FA_MODEL_REPO)"
	@echo ""
	@mkdir -p $(FA_MODEL_REPO)
	@echo "[1/3] Data Generation ($(DURATION_SECONDS)s)..."
	docker compose run --rm data-gather
	@echo ""
	@echo "[2/3] Feature Engineering (CPU vs GPU comparison)..."
	docker compose run --rm data-prep
	@echo ""
	@echo "[3/3] Model Training..."
	docker compose run --rm model-build
	@echo ""
	@echo "=========================================="
	@echo "Pipeline Complete!"
	@echo "=========================================="
	@echo ""
	@echo "Verify model output:"
	@ls -la $(FA_MODEL_REPO)/fraud_xgboost/ 2>/dev/null || echo "  Warning: Model not found at $(FA_MODEL_REPO)/fraud_xgboost/"
	@echo ""
	@echo "Start inference: make inference"

# Individual pods
pod1:
	@mkdir -p $(FB_DATA)
	docker compose run --rm data-gather

pod2:
	@mkdir -p $(FB_PREP)
	docker compose run --rm data-prep

pod3:
	@mkdir -p $(FA_MODEL_REPO)
	docker compose run --rm model-build

# Start inference server
inference:
	@echo "Starting Triton Inference Server..."
	@echo "Model repository: $(FA_MODEL_REPO)"
	@if [ ! -d "$(FA_MODEL_REPO)/fraud_xgboost" ]; then \
		echo "ERROR: Model not found at $(FA_MODEL_REPO)/fraud_xgboost/"; \
		echo "Run 'make pipeline' first to train the model."; \
		exit 1; \
	fi
	@ls -la $(FA_MODEL_REPO)/fraud_xgboost/
	docker compose up -d inference
	@echo ""
	@echo "Waiting for server to be ready..."
	@sleep 10
	@curl -s http://localhost:8000/v2/health/ready && echo " Server ready!" || echo " Server still starting (check logs: docker compose logs inference)"
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
		| python3 -m json.tool 2>/dev/null || echo "Error: Inference server not responding. Run 'make inference' first."
	@echo ""

# Check inference server status
status:
	@echo "=== Container Status ==="
	@docker compose ps
	@echo ""
	@echo "=== Model Repository ==="
	@ls -la $(FA_MODEL_REPO)/ 2>/dev/null || echo "  No models found"
	@echo ""
	@echo "=== Triton Health ==="
	@curl -s http://localhost:8000/v2/health/ready && echo "Ready" || echo "Not ready"

# Stop all containers
stop:
	docker compose down

# Clean generated data (preserves images)
clean-data:
	@echo "Cleaning generated data..."
	sudo rm -rf $(FB_DATA)/run_* 2>/dev/null || true
	sudo rm -rf $(FB_PREP)/features_*.parquet 2>/dev/null || true
	sudo rm -rf $(FB_PREP)/metadata_*.json 2>/dev/null || true
	sudo rm -rf $(FB_PREP)/.prep_state.json 2>/dev/null || true
	rm -rf $(FA_MODEL_REPO) 2>/dev/null || true
	@echo "Data cleaned"

# Full cleanup
clean-all: stop clean-data
	@echo "Removing Docker images..."
	docker compose down --rmi all -v 2>/dev/null || true
	docker builder prune -f
	@echo "Full cleanup complete"

# Quick demo (1 minute)
demo:
	@echo "Running quick demo (1 minute data generation)..."
	$(MAKE) pipeline DURATION_SECONDS=60 NUM_WORKERS=64 MAX_FILES_PER_RUN=50