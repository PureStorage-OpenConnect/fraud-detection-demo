# Pod 2: Data Prep - RAPIDS Accelerator for Spark
FROM apache/spark:3.5.0-python3

USER root

LABEL description="GPU-accelerated ETL using RAPIDS Accelerator for Apache Spark"

ENV SPARK_HOME=/opt/spark

# Install wget
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*

# Download RAPIDS Accelerator JAR (includes cudf dependency)
RUN mkdir -p ${SPARK_HOME}/jars/rapids && \
    cd ${SPARK_HOME}/jars/rapids && \
    wget https://repo1.maven.org/maven2/com/nvidia/rapids-4-spark_2.12/24.02.0/rapids-4-spark_2.12-24.02.0.jar

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
    "--jars", "/opt/spark/jars/rapids/rapids-4-spark_2.12-24.02.0.jar", \
    "/app/prepare.py"]