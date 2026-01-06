# Pod 6: Storage & Inference Stress Test
# Demonstrates Pure Storage FlashArray + NVIDIA Triton performance

FROM python:3.11-slim

WORKDIR /app

# Install dependencies including Triton gRPC client
RUN pip install --no-cache-dir \
    numpy>=1.26.0 \
    pandas>=2.1.0 \
    pyarrow>=14.0.0 \
    xgboost>=2.0.0 \
    tritonclient[grpc]>=2.42.0

COPY benchmark.py .

# FlashArray test configuration
ENV MODEL_DIR=/data/models
ENV DATA_DIR=/data/input
ENV DURATION_SECONDS=60
ENV NUM_WORKERS=8
ENV NUM_MODEL_COPIES=100
ENV RUN_INFERENCE=false
ENV INFERENCE_BATCH_SIZE=1000

# Test selection
ENV RUN_FLASHARRAY=true
ENV RUN_TRITON=false

# Triton test configuration
ENV TRITON_GRPC_URL=inference:8001
ENV TRITON_WORKERS=8
ENV TRITON_BATCH_SIZE=1000

ENV PYTHONUNBUFFERED=1

CMD ["python", "benchmark.py"]