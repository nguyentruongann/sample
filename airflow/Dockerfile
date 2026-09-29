ARG AIRFLOW_VERSION=3.3.2

FROM apache/airflow:${AIRFLOW_VERSION}-python3.12

# Khai báo lại để sử dụng ARG sau FROM
ARG AIRFLOW_VERSION

ENV PROJECT_ROOT=/opt/demand_forecasting

# LightGBM cần thư viện OpenMP libgomp.so.1
USER root

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Pip của Airflow phải chạy bằng user airflow
USER airflow

COPY --chown=airflow:root requirements-ml.txt /tmp/requirements-ml.txt

RUN python -m pip install --no-cache-dir \
    "apache-airflow==${AIRFLOW_VERSION}" \
    "apache-airflow-providers-fab>=2.0,<4.0" \
    -r /tmp/requirements-ml.txt

COPY --chown=airflow:root pyproject.toml README.md /opt/demand_forecasting/

COPY --chown=airflow:root src /opt/demand_forecasting/src

RUN python -m pip install --no-cache-dir --no-deps /opt/demand_forecasting \
    && mkdir -p \
        /opt/demand_forecasting/data \
        /opt/demand_forecasting/models \
        /opt/demand_forecasting/outputs