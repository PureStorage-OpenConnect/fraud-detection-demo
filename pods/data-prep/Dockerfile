# Pod 2: Data Prep - RAPIDS Accelerator for Spark
#
# NGC Authentication required before build:
#   docker login nvcr.io -u '$oauthtoken' -p <NGC_API_KEY>
#
# Get your API key from: https://ngc.nvidia.com/setup/api-key
#
FROM nvcr.io/nvidia/spark:3.5.0-py3

LABEL description="GPU-accelerated ETL using RAPIDS Accelerator for Apache Spark"

ENV RAPIDS_VERSION=24.02.0
ENV SPARK_HOME=/opt/spark

# Download RAPIDS JARs
RUN mkdir -p ${SPARK_HOME}/jars/rapids && \
    cd ${SPARK_HOME}/jars/rapids && \
    wget -q https://repo1.maven.org/maven2/com/nvidia/rapids-4-spark_2.12/${RAPIDS_VERSION}/rapids-4-spark_2.12-${RAPIDS_VERSION}.jar && \
    wget -q https://repo1.maven.org/maven2/ai/rapids/cudf/${RAPIDS_VERSION}/cudf-${RAPIDS_VERSION}-cuda12.jar

# Python dependencies
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Application
COPY prepare.py /app/
WORKDIR /app

# Default environment
ENV INPUT_DIR=/mnt/fsaai-shared/ebiser/fraud-data
ENV OUTPUT_DIR=/mnt/fsaai-shared/ebiser/prep_output
ENV NUM_GPUS=2
ENV BATCH_MODE=false
ENV POLL_INTERVAL=5

ENTRYPOINT ["spark-submit", \
    "--master", "local[*]", \
    "--driver-memory", "16g", \
    "--conf", "spark.plugins=com.nvidia.spark.SQLPlugin", \
    "--conf", "spark.rapids.sql.enabled=true", \
    "--conf", "spark.rapids.memory.pinnedPool.size=4g", \
    "--conf", "spark.rapids.sql.concurrentGpuTasks=2", \
    "--conf", "spark.executor.resource.gpu.amount=2", \
    "--conf", "spark.task.resource.gpu.amount=0.5", \
    "--conf", "spark.rapids.sql.format.parquet.read.enabled=true", \
    "--conf", "spark.rapids.sql.format.parquet.write.enabled=true", \
    "--conf", "spark.sql.adaptive.enabled=true", \
    "--jars", "/opt/spark/jars/rapids/rapids-4-spark_2.12-24.02.0.jar,/opt/spark/jars/rapids/cudf-24.02.0-cuda12.jar", \
    "/app/prepare.py"]