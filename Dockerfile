FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PROJECT_ROOT=/opt/demand_forecasting

WORKDIR /opt/demand_forecasting

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 50000 appuser

COPY requirements-ml.txt requirements-api.txt pyproject.toml README.md ./
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements-api.txt

COPY src ./src
COPY scripts ./scripts
RUN python -m pip install --no-cache-dir --no-deps . \
    && python -c "from importlib.resources import files; assert files('demand_forecasting').joinpath('dashboard_assets/monitoring.html').is_file(), 'Missing packaged monitoring.html'" \
    && mkdir -p data models outputs \
    && chown -R 50000:0 /opt/demand_forecasting

USER 50000

EXPOSE 8000

CMD ["uvicorn", "demand_forecasting.api:app", "--host", "0.0.0.0", "--port", "8000"]
