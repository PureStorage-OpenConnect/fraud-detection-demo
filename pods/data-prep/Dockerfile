# Pod 2: Feature Engineering
# Multi-GPU RAPIDS for data preparation

FROM nvcr.io/nvidia/rapidsai/base:24.02-cuda12.0-py3.10

ENV RAPIDS_NO_INITIALIZE=1
ENV CUDF_LOGGING_LEVEL=CRITICAL
ENV DASK_DISTRIBUTED__LOGGING__DISTRIBUTED=critical
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY prepare.py .

ENV INPUT_DIR=/data/input
ENV OUTPUT_DIR=/data/output
ENV BATCH_MODE=true
ENV LATEST_ONLY=true
ENV MAX_FILES_PER_RUN=100
ENV USE_MULTI_GPU=true

CMD ["python", "prepare.py"]