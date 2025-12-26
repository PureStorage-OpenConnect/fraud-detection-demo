# Pod 2: Data Prep - Multi-GPU RAPIDS (Dask-cuDF)
FROM nvcr.io/nvidia/rapidsai/base:24.02-cuda12.0-py3.10

LABEL description="Multi-GPU feature engineering using RAPIDS Dask-cuDF"

# Suppress duplicate logging from RAPIDS/cuDF
ENV RAPIDS_NO_INITIALIZE=1
ENV CUDF_LOGGING_LEVEL=ERROR
ENV PYTHONUNBUFFERED=1

# Dask configuration
ENV DASK_DISTRIBUTED__WORKER__MEMORY__TARGET=0.85
ENV DASK_DISTRIBUTED__WORKER__MEMORY__SPILL=0.90
ENV DASK_DISTRIBUTED__WORKER__MEMORY__PAUSE=0.95

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
ENV USE_MULTI_GPU=true

# Run directly without shell to avoid duplication
CMD ["python", "-u", "/app/prepare.py"]