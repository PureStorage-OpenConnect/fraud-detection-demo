# Fraud Detection Demo v2 - Solution Architecture

## Overview

This solution demonstrates a **CPU vs GPU comparison** for a fraud detection ML pipeline using Pure Storage. The demo showcases how GPU acceleration can dramatically speed up data processing and model training compared to traditional CPU-based approaches.

The pipeline processes financial transaction data through 4 stages, running both CPU and GPU workers in parallel to provide real-time performance comparison.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              EC2 Instance                                    │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                         Docker Compose                                   ││
│  │                                                                          ││
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              ││
│  │  │   Dashboard  │    │  CPU Worker  │    │  GPU Worker  │              ││
│  │  │   (Flask)    │◄──►│   (Python)   │    │  (Python +   │              ││
│  │  │   Port 5000  │    │   Port 5001  │    │   RAPIDS)    │              ││
│  │  │              │◄──►│              │    │   Port 5002  │              ││
│  │  └──────────────┘    └──────────────┘    └──────────────┘              ││
│  │         │                   │                   │                       ││
│  │         │                   │                   │                       ││
│  │         └───────────────────┴───────────────────┘                       ││
│  │                             │                                            ││
│  │                    ┌────────▼────────┐                                  ││
│  │                    │  /data volume   │                                  ││
│  │                    │  (NFS mount to  │                                  ││
│  │                    │   FlashBlade)   │                                  ││
│  │                    └─────────────────┘                                  ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

### Components

| Component | Port | Description |
|-----------|------|-------------|
| **Dashboard** | 8080 (external) → 5000 (internal) | Flask web UI, orchestrates stages, displays real-time metrics |
| **CPU Worker** | 5001 | Processes data using pandas, scikit-learn, XGBoost |
| **GPU Worker** | 5002 | Processes data using cuDF (RAPIDS), XGBoost with GPU |
| **Triton** | 8000-8002 | NVIDIA Triton Inference Server (optional, for production inference) |

---

## Pipeline Stages

The demo runs through 4 sequential stages, with both CPU and GPU workers processing in parallel:

### Stage 1: Ingest
- **Purpose**: Load raw transaction data from storage
- **CPU**: Uses pandas `read_csv()`
- **GPU**: Uses cuDF `read_csv()` for GPU-accelerated loading
- **Data**: ~1M transaction records from IEEE fraud detection dataset

### Stage 2: Data Prep
- **Purpose**: Feature engineering and data transformation
- **Creates 21 features**:
  - Transaction amount statistics (mean, std by card)
  - Time-based features (hour, day of week, time since last transaction)
  - Categorical encodings (ProductCD, card types, email domains)
  - Velocity features (transaction counts per time window)
  - Device/browser fingerprinting features

### Stage 3: Model Train
- **Purpose**: Train XGBoost gradient boosting classifier
- **Algorithm**: XGBoost with 100 boosting rounds
- **Parameters**:
  - `max_depth`: 6
  - `learning_rate`: 0.1
  - `early_stopping_rounds`: 10
  - `eval_metric`: AUC
- **CPU**: Standard XGBoost with `tree_method='hist'`
- **GPU**: XGBoost with `tree_method='gpu_hist'` and `device='cuda'`
- **Training split**: 80% train, 20% validation

### Stage 4: Inference
- **Purpose**: Run fraud predictions on test dataset
- **Batch processing**: Processes records in batches of 10,000
- **Output**: Fraud probability scores for each transaction

---

## Container Configurations

### Docker Compose (GPU Mode)
**File**: `docker-compose.yaml`

```yaml
services:
  dashboard:
    build: ./dashboard
    ports: ["8080:5000"]
    environment:
      - CPU_WORKER_URL=http://worker-cpu:5001
      - GPU_WORKER_URL=http://worker-gpu:5002

  worker-cpu:
    build: ./worker-cpu
    environment:
      - WORKER_PORT=5001
    volumes:
      - ${DATA_PATH:-./data}:/data:ro
      - ${MODEL_PATH:-./models}:/models

  worker-gpu:
    build: ./worker-gpu
    environment:
      - WORKER_PORT=5002
    volumes:
      - ${DATA_PATH:-./data}:/data:ro
      - ${MODEL_PATH:-./models}:/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    shm_size: '8gb'

  triton:
    image: nvcr.io/nvidia/tritonserver:24.02-py3
    # Used for production inference serving
```

### Docker Compose (Dual-CPU Mode)
**File**: `docker-compose.dual-cpu.yaml`

When GPU is not available (e.g., waiting for AWS quota), both workers run as CPU:

```yaml
services:
  worker-gpu:
    build:
      context: ./worker-cpu  # Uses CPU worker code
      dockerfile: Dockerfile
    environment:
      - WORKER_TYPE=cpu      # Runs as simulated "GPU"
```

---

## Real-Time Metrics & Training Indicator

### Metrics Collected
- **Rows processed** / **Total rows**
- **Bytes processed**
- **Throughput** (MB/s)
- **Max throughput** (peak performance)
- **Elapsed time**

### Training Progress
During the Model Train stage, a special **"Training..."** indicator appears with:
- Spinning animation circle
- Pulsing glow effect
- XGBoost progress callback reporting each boosting round

The `is_training` flag is set in the worker's `TrainingProgressCallback`:

```python
class TrainingProgressCallback(xgb.callback.TrainingCallback):
    def after_iteration(self, model, epoch, evals_log):
        self.report_metrics({'is_training': True, ...})
        return False  # Continue training
```

---

## FlashBlade Storage Integration

### NFS Mount Architecture

```
┌─────────────────┐         ┌─────────────────────────┐
│   FlashBlade    │   NFS   │      EC2 Instance       │
│                 │◄───────►│                         │
│  /fraud-data    │         │  /mnt/flashblade        │
│                 │         │         │               │
└─────────────────┘         │         ▼               │
                            │  Docker Volume Mount    │
                            │  /mnt/flashblade:/data  │
                            └─────────────────────────┘
```

### Setup Steps

1. **Mount FlashBlade on EC2**:
```bash
# Install NFS client
sudo apt-get install nfs-common

# Create mount point
sudo mkdir -p /mnt/flashblade

# Mount FlashBlade (replace with your FlashBlade IP)
sudo mount -t nfs -o vers=3,tcp <flashblade-ip>:/fraud-data /mnt/flashblade

# Add to /etc/fstab for persistence
echo "<flashblade-ip>:/fraud-data /mnt/flashblade nfs vers=3,tcp 0 0" | sudo tee -a /etc/fstab
```

2. **Configure Docker Compose**:
```bash
# Set environment variable
export DATA_PATH=/mnt/flashblade

# Start containers
docker compose up -d
```

3. **Or specify inline**:
```bash
DATA_PATH=/mnt/flashblade docker compose up -d
```

### Data Directory Structure
```
/mnt/flashblade/
├── cpu/
│   ├── train_transaction.csv
│   ├── train_identity.csv
│   └── test_transaction.csv
└── gpu/
    ├── train_transaction.csv
    ├── train_identity.csv
    └── test_transaction.csv
```

---

## Deployment Options

### Option 1: Single GPU Instance (Recommended for Demo)

Run everything on one GPU-enabled EC2 instance:

```bash
# Instance type: g4dn.xlarge or p3.2xlarge
# All 3 containers on same instance
# Simpler networking, easier demo

docker compose -f docker-compose.yaml up -d
```

**Pros**:
- Simple setup
- No network latency between containers
- Easier to demonstrate

**Cons**:
- Containers share GPU memory
- May need larger instance for production

### Option 2: Dual-CPU Mode (Current)

When waiting for GPU quota approval:

```bash
docker compose -f docker-compose.dual-cpu.yaml up -d
```

Both workers run CPU code - the "GPU" worker is just a second CPU worker for demonstration purposes.

### Option 3: Separate Instances (Production)

For production or larger datasets:
- Dashboard on t3.medium
- CPU Worker on c5.4xlarge (compute optimized)
- GPU Worker on p3.2xlarge or g4dn.xlarge

Use Docker Swarm or Kubernetes for orchestration.

---

## Running the Demo

### Prerequisites
- Docker and Docker Compose installed
- NVIDIA drivers and nvidia-container-toolkit (for GPU mode)
- Data files in the data directory

### Quick Start

```bash
# Clone repository
git clone https://github.com/PureStorage-OpenConnect/fraud-detection-demo
cd fraud-detection-demo/v2

# Start containers (GPU mode)
docker compose up -d

# Or dual-CPU mode
docker compose -f docker-compose.dual-cpu.yaml up -d

# Access dashboard
open http://localhost:8080
```

### Demo Flow

1. **Open Dashboard** at `http://<instance-ip>:8080`
2. **Click "Start Demo"** to begin Stage 1 (Ingest)
3. **Watch real-time metrics** as both workers process data
4. **Click "Continue"** after each stage completes
5. **View Summary** after all 4 stages showing speedup comparison

### Expected Results

| Stage | CPU Time | GPU Time | Speedup |
|-------|----------|----------|---------|
| Ingest | ~30s | ~5s | ~6x |
| Data Prep | ~45s | ~8s | ~5-6x |
| Model Train | ~120s | ~15s | ~8x |
| Inference | ~20s | ~3s | ~7x |

*Note: Actual times depend on data size and instance type*

---

## Key Files

```
v2/
├── docker-compose.yaml          # GPU mode orchestration
├── docker-compose.dual-cpu.yaml # Dual-CPU mode
├── SOLUTION.md                  # This document
│
├── dashboard/
│   ├── Dockerfile
│   ├── app.py                   # Flask orchestrator
│   ├── templates/
│   │   └── index.html           # Main UI
│   └── static/
│       ├── css/dashboard.css    # Styling + animations
│       └── js/dashboard.js      # Real-time updates
│
├── worker-cpu/
│   ├── Dockerfile
│   └── worker.py                # CPU pipeline (pandas, XGBoost)
│
├── worker-gpu/
│   ├── Dockerfile
│   └── worker.py                # GPU pipeline (cuDF, RAPIDS)
│
├── data/                        # Transaction data (mounted)
└── models/                      # Saved models
```

---

## Troubleshooting

### Container won't start
```bash
# Check logs
docker compose logs worker-gpu

# Common issues:
# - NVIDIA driver not installed
# - nvidia-container-toolkit missing
# - GPU already in use
```

### No GPU detected
```bash
# Verify NVIDIA driver
nvidia-smi

# Verify Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

### Data not loading
```bash
# Check volume mounts
docker compose exec worker-cpu ls -la /data

# Verify NFS mount
mount | grep flashblade
```

---

## Summary

This demo showcases:
- **Pure Storage FlashBlade** as high-performance data source via NFS
- **NVIDIA RAPIDS (cuDF)** for GPU-accelerated data processing
- **XGBoost** with GPU support for fast model training
- **Real-time comparison** dashboard showing CPU vs GPU performance
- **Container orchestration** with Docker Compose

The combination of Pure Storage's high-throughput storage with GPU acceleration demonstrates dramatic speedups for fraud detection ML pipelines.

---

## Appendix A: Original v1 Pipeline (6 Pods)

The original v1 solution used a 6-pod architecture designed for production-scale processing:

### Pod Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Production Pipeline (v1)                            │
│                                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │   Pod 1     │    │   Pod 2     │    │   Pod 3     │    │   Pod 4     │      │
│  │ Data Gather │───►│  Data Prep  │───►│ Model Build │───►│  Inference  │      │
│  │   (CPU)     │    │ (Multi-GPU) │    │   (GPU)     │    │  (Triton)   │      │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘      │
│         │                  │                  │                  │              │
│         ▼                  ▼                  ▼                  ▼              │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         FlashBlade (NFS)                                 │   │
│  │   /fraud-data          /prep-output        /model_repository            │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  ┌─────────────┐    ┌─────────────┐                                            │
│  │   Pod 5     │    │   Pod 6     │                                            │
│  │Notification │    │  Benchmark  │                                            │
│  │   (REST)    │    │ (I/O Stress)│                                            │
│  └─────────────┘    └─────────────┘                                            │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Pod Specifications

| Pod | Container Name | Purpose | Resources | Ports |
|-----|----------------|---------|-----------|-------|
| **1** | `fraud-data-gather` | Generate synthetic transactions at 2+ GB/s | 32 CPUs, 64GB RAM | - |
| **2** | `fraud-data-prep` | RAPIDS cuDF feature engineering | All GPUs, 16GB shm | - |
| **3** | `fraud-model-build` | XGBoost GPU training | 1 GPU | - |
| **4** | `fraud-inference` | Triton Inference Server | 1 GPU | 8000, 8001, 8002 |
| **5** | `fraud-notification` | REST API for alerts | CPU only | 5000 |
| **6** | `fraud-benchmark` | FlashArray I/O stress test | Privileged | - |

### Pod 1: Data Generator
```yaml
data-gather:
  build: ./pods/data-gather
  container_name: fraud-data-gather
  volumes:
    - ${FB_DATA}:/data/output           # FlashBlade mount
  environment:
    - OUTPUT_DIR=/data/output
    - NUM_WORKERS=128                    # Parallel writers
    - DURATION_SECONDS=60                # Generation duration
    - CHUNK_SIZE=1000000                 # Rows per chunk
    - FRAUD_RATE=0.005                   # 0.5% fraud rate
  deploy:
    resources:
      limits:
        cpus: '32'
        memory: 64G
  ulimits:
    nofile:
      soft: 65536
      hard: 65536
```

### Pod 2: Feature Engineering (Multi-GPU)
```yaml
data-prep:
  build: ./pods/data-prep
  container_name: fraud-data-prep
  volumes:
    - ${FB_DATA}:/data/input:ro          # Read from Pod 1
    - ${FB_PREP}:/data/output            # Write features
  environment:
    - INPUT_DIR=/data/input
    - OUTPUT_DIR=/data/output
    - BATCH_MODE=true
    - LATEST_ONLY=true
    - MAX_FILES_PER_RUN=100
    - USE_MULTI_GPU=true
  shm_size: '16gb'                       # GPU shared memory
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: all                    # Use ALL GPUs
            capabilities: [gpu]
```

### Pod 3: Model Training
```yaml
model-build:
  build: ./pods/model-build
  container_name: fraud-model-build
  volumes:
    - ${FB_PREP}:/data/input:ro          # Read from Pod 2
    - ${FA_MODEL_REPO}:/data/models      # Write to Triton repo
  environment:
    - PREP_OUTPUT_DIR=/data/input
    - FA_MOUNT=/data/models
  user: root
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

### Pod 4: Triton Inference Server
```yaml
inference:
  image: nvcr.io/nvidia/tritonserver:24.02-py3
  container_name: fraud-inference
  ports:
    - "8000:8000"                        # HTTP API
    - "8001:8001"                        # gRPC API
    - "8002:8002"                        # Metrics
  volumes:
    - ${FA_MODEL_REPO}:/models:ro
  command: ["tritonserver", "--model-repository=/models", "--strict-model-config=false"]
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:8000/v2/health/ready"]
    interval: 10s
    timeout: 5s
    retries: 5
```

### Pod 5: Notification Service
```yaml
notification:
  build: ./pods/notification
  container_name: fraud-notification
  ports:
    - "5000:5000"
  environment:
    - HOST=0.0.0.0
    - PORT=5000
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
    interval: 10s
```

### Pod 6: Benchmark (I/O Stress Test)
```yaml
benchmark:
  build: ./pods/benchmark
  container_name: fraud-benchmark
  privileged: true                       # For drop_caches
  volumes:
    - ${FA_MODEL_REPO}:/data/models
    - ${FB_DATA}:/data/input:ro
  environment:
    # FlashArray stress test
    - RUN_FA=true
    - FA_WORKERS=64
    - FA_FILE_SIZE_MB=50
    - FA_NUM_FILES=200
    - FA_MIXED=false
    # Triton inference test
    - RUN_TRITON=true
    - TRITON_URL=inference:8001
    - TRITON_WORKERS=8
    - TRITON_BATCH_SIZE=1000
```

---

## Appendix B: FlashBlade Configuration

### Environment Variables (.env)

```bash
# =============================================================================
# FlashBlade Storage Paths (High-Throughput)
# =============================================================================
# Base FlashBlade mount point
FB_MOUNT=/mnt/flashblade

# Pod 1 Output: Generated transaction data
FB_DATA=${FB_MOUNT}/fraud-data

# Pod 2 Output: Prepared features
FB_PREP=${FB_MOUNT}/prep-output

# =============================================================================
# FlashArray / Model Repository
# =============================================================================
# Triton model repository (Pod 3 Output → Pod 4 Input)
FA_MODEL_REPO=${PWD}/model_repository

# =============================================================================
# Pod 1: Data Generation Settings
# =============================================================================
NUM_WORKERS=128              # Parallel data generators
DURATION_SECONDS=60          # How long to generate data
CHUNK_SIZE=1000000           # Rows per parquet file
FRAUD_RATE=0.005             # 0.5% fraud transactions

# =============================================================================
# Pod 2: Feature Engineering Settings
# =============================================================================
MAX_FILES_PER_RUN=100        # Max parquet files to process
USE_MULTI_GPU=true           # Enable multi-GPU processing
LATEST_ONLY=true             # Only process latest run

# =============================================================================
# Pod 6: Benchmark Settings
# =============================================================================
BENCHMARK_DURATION=60        # Stress test duration

# FlashArray I/O stress test
BENCHMARK_WORKERS=8          # Concurrent workers
BENCHMARK_COPIES=100         # Model copies (defeats cache)

# Triton inference test
TRITON_WORKERS=8             # Concurrent inference workers
TRITON_BATCH_SIZE=1000       # Records per batch

# =============================================================================
# S3 Configuration (Optional - FlashBlade S3)
# =============================================================================
#S3_ENDPOINT=https://flashblade.example.com
#S3_ACCESS_KEY=your_access_key
#S3_SECRET_KEY=your_secret_key
#S3_BUCKET=fraud-detection
```

### FlashBlade Mount Commands

```bash
# 1. Install NFS client
sudo apt-get update && sudo apt-get install -y nfs-common

# 2. Create mount points
sudo mkdir -p /mnt/flashblade/fraud-data
sudo mkdir -p /mnt/flashblade/prep-output

# 3. Mount FlashBlade exports
sudo mount -t nfs -o vers=3,tcp,rsize=1048576,wsize=1048576 \
    <flashblade-ip>:/fraud-data /mnt/flashblade/fraud-data

sudo mount -t nfs -o vers=3,tcp,rsize=1048576,wsize=1048576 \
    <flashblade-ip>:/prep-output /mnt/flashblade/prep-output

# 4. Verify mounts
df -h | grep flashblade

# 5. Add to /etc/fstab for persistence
cat >> /etc/fstab << EOF
<flashblade-ip>:/fraud-data   /mnt/flashblade/fraud-data   nfs  vers=3,tcp,rsize=1048576,wsize=1048576  0 0
<flashblade-ip>:/prep-output  /mnt/flashblade/prep-output  nfs  vers=3,tcp,rsize=1048576,wsize=1048576  0 0
EOF
```

### FlashBlade NFS Tuning Options

| Option | Value | Purpose |
|--------|-------|---------|
| `vers=3` | NFSv3 | Better performance for parallel I/O |
| `tcp` | TCP | Reliable transport |
| `rsize=1048576` | 1MB | Read buffer size |
| `wsize=1048576` | 1MB | Write buffer size |
| `noatime` | - | Skip access time updates |
| `nodiratime` | - | Skip directory access time |

### Directory Structure on FlashBlade

```
/mnt/flashblade/
├── fraud-data/                    # Pod 1 output
│   └── run_20240115_143022/       # Timestamped run
│       ├── worker_000.parquet
│       ├── worker_001.parquet
│       └── ...
│
├── prep-output/                   # Pod 2 output
│   └── features_run_20240115.parquet
│
└── models/                        # Optional: model storage
    └── fraud_xgboost/
        └── 1/
            └── xgboost.json
```

### v2 Simplified Configuration

For the v2 demo, FlashBlade integration is simplified:

```bash
# .env.example for v2
DATA_PATH=/mnt/flashblade/fraud-demo/data
MODEL_PATH=/mnt/flashblade/fraud-demo/models

# Or local for testing
DATA_PATH=./data
MODEL_PATH=./models
```

```bash
# Start v2 with FlashBlade
DATA_PATH=/mnt/flashblade/fraud-demo docker compose up -d
```
