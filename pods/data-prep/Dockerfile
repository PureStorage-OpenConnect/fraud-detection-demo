# Pod 2: Data Prep - RAPIDS cuDF (GPU-accelerated)
FROM nvcr.io/nvidia/rapidsai/base:24.02-cuda12.0-py3.10

LABEL description="GPU-accelerated ETL using RAPIDS cuDF"

WORKDIR /app

# Application
COPY prepare.py /app/

ENV INPUT_DIR=/mnt/fsaai-shared/ebiser/fraud-data
ENV OUTPUT_DIR=/mnt/fsaai-shared/ebiser/prep_output
ENV BATCH_MODE=false
ENV POLL_INTERVAL=5

ENTRYPOINT ["python", "/app/prepare.py"]