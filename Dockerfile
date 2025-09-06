
FROM apache/airflow:2.9.2

USER root
COPY --chown=airflow:root requirements.txt /requirements.txt

ARG AIRFLOW_VERSION=2.9.2
ARG PYTHON_VERSION=3.12
ENV AIRFLOW_CONSTRAINTS_LOCATION="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"


USER airflow
RUN python -m pip install --no-cache-dir -r /requirements.txt --constraint "${AIRFLOW_CONSTRAINTS_LOCATION}"
