# Fraud Detection Demo v2

CPU vs GPU comparison demo with Pure Storage FlashBlade integration.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Dashboard / Orchestrator                      │
│                   (Web UI + PureStorage branding)                │
│                      http://localhost:8080                       │
└──────────────┬────────────────────────────────┬─────────────────┘
               │                                │
               ▼                                ▼
┌─────────────────────┐          ┌─────────────────────┐
│      CPU Worker     │          │      GPU Worker     │
│   (python/pandas)   │          │      (RAPIDS)       │
└──────────┬──────────┘          └──────────┬──────────┘
           │                                │
           └──────────────┬─────────────────┘
                          ▼
                ┌───────────────────┐
                │   Triton Server   │
                │  (CPU + GPU models)│
                └───────────────────┘
```

## Quick Start

### 1. Generate Synthetic Data (One-time setup)

```bash
# Generate 1 million transactions
python scripts/generate_data.py --rows 1000000 --output-dir ./data

# Or use the Makefile
make generate-data ROWS=1000000
```

This creates:
- `data/cpu/transactions.parquet`
- `data/gpu/transactions.parquet`

### 2. Start the Demo

```bash
# Build and start all containers
make up

# Or manually
docker compose up --build -d
```

### 3. Open the Dashboard

Navigate to: **http://localhost:8080**

### 4. Run the Pipeline

1. Click **Start Demo** to begin Stage 1 (Ingest)
2. Watch CPU and GPU progress in parallel
3. Click **Continue** after each stage completes
4. View the final **Summary** after all 4 stages

## Pipeline Stages

| Stage | CPU Path | GPU Path | FlashBlade Showcase |
|-------|----------|----------|---------------------|
| **Ingest** | pandas.read_parquet | cudf.read_parquet | Read throughput |
| **Data Prep** | pandas/numpy | cuDF/cupy | Read + processing |
| **Model Train** | XGBoost CPU | XGBoost GPU | Data loading |
| **Inference** | Triton (CPU model) | Triton (GPU model) | Batch scoring |

## Configuration

Environment variables in `.env` or docker-compose:

```bash
# Data paths
DATA_PATH=/path/to/flashblade/data
MODEL_PATH=./models

# Dashboard
DASHBOARD_PORT=8080
```

## Development

### Rebuild a single service

```bash
docker compose build worker-gpu
docker compose up -d worker-gpu
```

### View logs

```bash
docker compose logs -f dashboard
docker compose logs -f worker-cpu
docker compose logs -f worker-gpu
```

### Stop all services

```bash
make down
# or
docker compose down
```

## Files

```
v2/
├── docker-compose.yaml      # Container orchestration
├── Makefile                 # Build/run commands
├── scripts/
│   └── generate_data.py     # Offline data generation
├── dashboard/
│   ├── app.py               # Flask orchestrator
│   ├── templates/index.html # Dashboard UI
│   └── static/              # CSS/JS assets
├── worker-cpu/
│   └── worker.py            # CPU pipeline (pandas)
├── worker-gpu/
│   └── worker.py            # GPU pipeline (RAPIDS)
└── data/                    # Generated data
    ├── cpu/
    └── gpu/
```
