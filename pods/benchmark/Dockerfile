# Pod 6: Sustained Throughput Benchmark
# Compares CPU (XGBoost) vs GPU (Triton) inference over 60 seconds

FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    numpy>=1.26.0 \
    pandas>=2.1.0 \
    pyarrow>=14.0.0 \
    xgboost>=2.0.0 \
    requests>=2.31.0

COPY benchmark.py .

# Configuration
ENV DATA_DIR=/data/input
ENV MODEL_DIR=/data/models
ENV TRITON_URL=http://inference:8000
ENV DURATION_SECONDS=60
ENV BATCH_SIZE=10000
ENV PYTHONUNBUFFERED=1

CMD ["python", "benchmark.py"]