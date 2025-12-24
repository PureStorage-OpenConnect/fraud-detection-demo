# NVIDIA Financial Fraud Detection Pipeline

[![NVIDIA](https://img.shields.io/badge/NVIDIA-L40S-76B900?logo=nvidia)](https://www.nvidia.com/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![RAPIDS](https://img.shields.io/badge/RAPIDS-cuDF%20%7C%20cuGraph-blueviolet)](https://rapids.ai/)
[![Pure Storage](https://img.shields.io/badge/Pure_Storage-FlashBlade-FF6600)](https://www.purestorage.com/)

## Overview

A containerized fraud detection pipeline optimized for dual NVIDIA L40S GPUs and Pure Storage. This project re-architects the [NVIDIA Financial Fraud Detection AI Blueprint](https://github.com/NVIDIA-AI-Blueprints/Financial-Fraud-Detection) into 5 independent Docker containers that work together to process transactions, train models, and detect fraud in real-time.

---

## System Architecture

```mermaid
graph LR
    A[Pod 1<br/>Data Gather] -->|Generated Data| B[Pod 2<br/>Data Prep]
    B -->|Features| C[Pod 3<br/>Model Build]
    C -->|Models| D[Pod 4<br/>Inference]
    D -->|Alerts| E[Pod 5<br/>Notification]
    
    style A fill:#76B900,stroke:#333,stroke-width:2px,color:#fff
    style B fill:#d85e00,stroke:#333,stroke-width:2px,color:#fff
    style C fill:#d85e00,stroke:#333,stroke-width:2px,color:#fff
    style D fill:#d85e00,stroke:#333,stroke-width:2px,color:#fff
    style E fill:#1a5490,stroke:#333,stroke-width:2px,color:#fff
```

---

## 5-Pod Architecture

| Pod | Container | GPU | Purpose |
|-----|-----------|-----|---------|
| 1 | `data-gather` | No | High-throughput synthetic transaction data generation |
| 2 | `data-prep` | 2x L40S | GPU-accelerated feature engineering (RAPIDS) |
| 3 | `model-build` | 2x L40S | Train GNN and XGBoost models |
| 4 | `inference` | 2x L40S | Real-time fraud detection (Triton Server) |
| 5 | `notification` | No | Handle fraud alerts via webhook |

---

## Storage

This demo uses Pure Storage for high-performance data access:

- **FlashBlade (FB)**: High-throughput parallel I/O for data generation and feature processing
- **FlashArray (FA)**: Low-latency storage for model serving

### Mount Points

| Mount Point | Purpose |
|-------------|---------|
| `/mnt/datasets/kaggle/creditcardfraud` | Input template (Kaggle creditcard.csv) |
| `/mnt/fsaai-shared/ebiser/fraud-data` | Generated transaction data output |
| `~/ebiser/nvidia.financial.fraud.detection` | Model repository |

---

## Quick Start

### Prerequisites

- 2x NVIDIA L40S GPUs
- Docker >= 24.x with NVIDIA Container Toolkit
- Pure Storage FlashBlade and FlashArray mounts configured
- [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

### Installation

```bash
# Clone repository
git clone https://github.com/PureStorage-OpenConnect/financial-fraud-demo.git
cd financial-fraud-demo

# Build containers
docker-compose build

# Start the pipeline
docker-compose up
```

---

## Data Generation Demo

Pod 1 demonstrates high-throughput data generation using 128 parallel workers.

### Running the Demo

```bash
# Run with defaults (5 minutes, 128 workers, Parquet format)
docker-compose up data-gather

# Custom configuration
NUM_WORKERS=256 DURATION_SECONDS=600 OUTPUT_FORMAT=binary docker-compose up data-gather
```

### Output Formats

| Format | Description |
|--------|-------------|
| `parquet` | Apache Parquet (default) - good balance of speed and compatibility |
| `binary` | Raw numpy arrays - maximum throughput |
| `csv` | CSV text format - most compatible |

### Reading the Output

```
[   30s] Files:  128 | Size:  45.00 GB | Speed:  1.01 GB/s | ~8.3M rec/s | Workers: 128
```

| Metric | Description |
|--------|-------------|
| **Files** | Number of output files generated |
| **Size** | Total data written |
| **Speed** | Current write throughput |
| **rec/s** | Records generated per second |
| **Workers** | Active worker processes |

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `NUM_WORKERS` | 128 | Parallel worker processes |
| `DURATION_SECONDS` | 300 | Demo duration (5 minutes) |
| `CHUNK_SIZE` | 50000 | Rows per write operation |
| `OUTPUT_FORMAT` | parquet | Output format (parquet/binary/csv) |

---

## Usage

### Run Individual Pods

```bash
# Pod 1: Data generation
docker-compose up data-gather

# Pod 2: Feature preparation
docker-compose up data-prep

# Pod 3: Model training
docker-compose up model-build

# Pods 4 & 5: Inference and notifications
docker-compose up inference notification
```

### Monitor

```bash
# View logs
docker-compose logs -f data-gather

# Check GPU usage
watch -n 1 nvidia-smi

# Container stats
docker stats
```

---

## Troubleshooting

### Low throughput

```bash
# Verify template file exists
ls -la /mnt/datasets/kaggle/creditcardfraud/creditcard.csv

# Check output directory permissions
ls -la /mnt/fsaai-shared/ebiser/fraud-data/
```

### GPU not detected

```bash
# Verify NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

# Reconfigure runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Container issues

```bash
# Check logs
docker-compose logs <service-name>

# Rebuild
docker-compose build --no-cache <service-name>
```

---

## License

Apache License 2.0