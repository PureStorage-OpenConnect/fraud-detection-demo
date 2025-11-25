# financial-fraud-demo
NVIDIA Financial Fraud Detection Pipeline (Dual L40S Optimized)

🌟 Overview

This project re-architects the NVIDIA Financial Fraud Detection AI Blueprint into a highly scalable, containerized, 5-tier microservice pipeline optimized for dual NVIDIA L40S GPUs within a Kubernetes environment.

This project is a brownfield redevelopment of the original solution, which can be found here:

Original Blueprint Documentation: NVIDIA Financial Fraud Detection

Original GitHub Repository: NVIDIA-AI-Blueprints/Financial-Fraud-Detection

The design focuses on decoupling the core machine learning workflow—data gathering, high-speed data preparation, distributed training, and real-time inference—into distinct, independently deployable services, ideal for demonstrating performance and scalability, particularly concerning high-speed I/O with Pure Storage mounts.

⚙️ Architecture: 5-Tier Microservices

The workflow is divided into five sequential or continuously running containers, with specific GPU allocation designed to maximize utilization of the dual L40S system.

Stage

Service Name

Container Type

Primary GPU Allocation

Core Responsibility

0. Gather

data-gather-service

K8s Job

None

Generates/Ingests raw transactional data.

1. Prep

data-prep-service

K8s Job

Dual L40S (2x)

High-speed, GPU-accelerated feature engineering and graph creation via RAPIDS (cuDF/cuGraph).

2. Build

model-build-service

K8s Job

Dual L40S (2x)

Distributed GNN and cuXGBoost training, saving models to the Triton Model Repository.

3. Serve

inference-service

K8s Deployment

Dual L40S (2x)

Real-time, low-latency fraud scoring using NVIDIA Triton Inference Server and TensorRT.

4. Notify

notification-service

K8s Deployment

None

Provides a webhook for receiving and simulating the streaming of high-risk transaction alerts.

🗺️ System Architecture

This diagram illustrates the data flow and container orchestration across the 5 distinct microservices, highlighting the GPU allocation for performance-critical stages.

🚀 Key Technology Stack

Component

Technology

Rationale

GPUs

NVIDIA L40S (2x)

High-performance compute for all intensive stages. Explicitly configured for dual-GPU utilization.

Data Prep

RAPIDS (cuDF, cuGraph)

Maximizes data processing speed and I/O saturation using the GPU memory.

Model Training

RAPIDS (cuXGBoost), PyTorch/TensorFlow

Enables distributed and accelerated training of the GNN and final XGBoost classifier.

Inference

NVIDIA Triton Inference Server

Production-grade model serving for high-throughput, low-latency real-time scoring.

Orchestration

Docker Compose, Kubernetes

Provides a clear path from local validation (Docker) to scalable, resilient deployment (K8s).

Persistence

Shared PVC

Simulates the connection to high-speed Pure Storage mounts (/data) for artifact sharing.

📦 Data Flow and Persistence

All intermediate data and final models are stored on a shared Persistent Volume mounted at /data across all containers. This design ensures artifacts are readily available for subsequent stages without needing to copy data between service executions.

data-gather writes raw CSV to /data/raw_data/.

data-prep reads the raw CSV and writes features/graphs to /data/artifacts/prep_output/.

model-build reads the prepared features and writes deployable models to /data/model_repository/ (Triton's required format).

triton-server reads the models directly from /data/model_repository/ and serves them.

triton-server sends high-risk alerts to the notification-service endpoint (e.g., http://notification-service:5000/notify/fraud).

🛠️ Deployment Strategy

The project follows a two-phase deployment strategy:

Phase 1: Local Validation (Docker Compose)

Before deploying to Kubernetes, the entire pipeline can be built and executed locally using docker-compose.yaml. This validates container images, script execution order, volume mounting, and the inter-service data dependencies on a single host with GPU access.

Phase 2: Scalable Deployment (Kubernetes)

Once local validation is complete, the k8s_manifests.yaml file defines the full production environment, including:

K8s Jobs for the batch stages (Gather, Prep, Build).

K8s Deployments and Services for the continuous services (Inference, Notification).

Explicit GPU resource requests (nvidia.com/gpu: 2) to leverage the dual-L40S system effectively.