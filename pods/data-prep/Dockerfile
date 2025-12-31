FROM nvcr.io/nvidia/rapidsai/base:24.02-cuda12.0-py3.10

ENV RAPIDS_NO_INITIALIZE=1
ENV CUDF_LOGGING_LEVEL=CRITICAL
ENV PYTHONUNBUFFERED=1
ENV DASK_DISTRIBUTED__LOGGING__DISTRIBUTED=critical

WORKDIR /app
COPY prepare.py /app/

# Default configuration
ENV INPUT_DIR=/mnt/fsaai-shared/ebiser/fraud-data
ENV OUTPUT_DIR=/mnt/fsaai-shared/ebiser/prep-output
ENV BATCH_MODE=false
ENV POLL_INTERVAL=5
ENV MAX_FILES_PER_RUN=50
ENV LATEST_ONLY=true
ENV FILE_STABLE_SECONDS=10
ENV USE_MULTI_GPU=true

CMD ["python", "prepare.py"]