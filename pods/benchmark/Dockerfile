# Pod 6: FlashArray + Triton Benchmark
# Aggressive I/O stress test

FROM python:3.11-slim

WORKDIR /app

# Install coreutils for dd with direct I/O
RUN apt-get update && \
    apt-get install -y --no-install-recommends coreutils procps && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    numpy>=1.26.0 \
    tritonclient[grpc]>=2.42.0

COPY benchmark.py .

# Common
ENV MODEL_DIR=/data/models
ENV DATA_DIR=/data/input
ENV DURATION_SECONDS=60

# FlashArray test - AGGRESSIVE DEFAULTS
ENV RUN_FA=true
ENV FA_WORKERS=64
ENV FA_FILE_SIZE_MB=50
ENV FA_NUM_FILES=200
ENV FA_MIXED=false

# Triton test
ENV RUN_TRITON=true
ENV TRITON_URL=inference:8001
ENV TRITON_WORKERS=8
ENV TRITON_BATCH_SIZE=1000

ENV PYTHONUNBUFFERED=1

CMD ["python", "benchmark.py"]