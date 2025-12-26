# Pod 2: Data Prep - RAPIDS cuDF (GPU-accelerated)
FROM nvcr.io/nvidia/rapidsai/base:24.02-cuda12.0-py3.10

LABEL description="GPU-accelerated ETL using RAPIDS cuDF"

# Suppress duplicate logging from RAPIDS/cuDF
ENV RAPIDS_NO_INITIALIZE=1
ENV CUDF_LOGGING_LEVEL=ERROR
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Application
COPY prepare.py /app/

ENV INPUT_DIR=/mnt/fsaai-shared/ebiser/fraud-data
ENV OUTPUT_DIR=/mnt/fsaai-shared/ebiser/prep-output
ENV BATCH_MODE=false
ENV POLL_INTERVAL=5
ENV MAX_FILES_PER_RUN=50
ENV LATEST_ONLY=true
ENV FILE_STABLE_SECONDS=10

# Run directly without shell to avoid duplication
CMD ["python", "-u", "/app/prepare.py"]