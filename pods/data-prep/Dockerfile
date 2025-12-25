# Pod 2: Data Prep - Spark with RAPIDS Accelerator on CUDA
# Based on NVIDIA's recommended approach for GPU-accelerated Spark
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

LABEL description="GPU-accelerated Spark ETL using RAPIDS Accelerator"

# Versions
ENV SPARK_VERSION=3.5.0
ENV HADOOP_VERSION=3
ENV RAPIDS_VERSION=24.02.0
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-11-jdk-headless \
    python3 \
    python3-pip \
    wget \
    procps \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

# Install Spark
RUN wget -q https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
    && tar -xzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz -C /opt \
    && mv /opt/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} /opt/spark \
    && rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz

ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin

# Download RAPIDS Accelerator JAR
RUN wget -q https://repo1.maven.org/maven2/com/nvidia/rapids-4-spark_2.12/${RAPIDS_VERSION}/rapids-4-spark_2.12-${RAPIDS_VERSION}.jar \
    -P ${SPARK_HOME}/jars/

# Python dependencies
COPY requirements.txt /tmp/
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

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
    "/app/prepare.py"]