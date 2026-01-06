# Pod 6: FlashArray Model Reload Stress Test
# Demonstrates low-latency model serving from Pure Storage FlashArray

FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    numpy>=1.26.0 \
    pandas>=2.1.0 \
    pyarrow>=14.0.0 \
    xgboost>=2.0.0

COPY benchmark.py .

# Configuration
ENV MODEL_DIR=/data/models
ENV DATA_DIR=/data/input
ENV DURATION_SECONDS=60
ENV NUM_WORKERS=8
ENV RUN_INFERENCE=true
ENV INFERENCE_BATCH_SIZE=1000
ENV PYTHONUNBUFFERED=1

CMD ["python", "benchmark.py"]