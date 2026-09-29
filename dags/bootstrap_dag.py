"""Optional manual bootstrap; never replace existing raw observations."""

from __future__ import annotations

from datetime import timedelta

import pendulum
from airflow.sdk import dag, task


@dag(
    dag_id="demand_bootstrap",
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz="Asia/Ho_Chi_Minh"),
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "demand-forecasting", "retries": 1},
    tags=["demand", "bootstrap", "training"],
)
def demand_bootstrap():
    @task(pool="ml_cpu", execution_timeout=timedelta(hours=3))
    def ensure_historical_dataset() -> dict:
        from demand_forecasting.config import get_settings
        from demand_forecasting.database import (
            copy_parquet_to_postgres,
            database_has_rows,
            ensure_runtime_tables,
        )
        from demand_forecasting.fake_data import generate_fake_data

        settings = get_settings()
        ensure_runtime_tables(settings)
        if database_has_rows(settings):
            return {"seeded": False, "table": f"{settings.postgres_schema}.{settings.postgres_table}"}
        path, _ = generate_fake_data(settings.data_dir / "raw", overwrite=True)
        summary = copy_parquet_to_postgres(path, settings=settings)
        return {"seeded": True, "rows": summary.loaded_rows, "table": f"{summary.schema}.{summary.table}"}

    @task(pool="ml_cpu", execution_timeout=timedelta(hours=12))
    def train_models(load_summary: dict) -> dict:
        from demand_forecasting.config import get_settings
        from demand_forecasting.jobs import train_from_postgres

        settings = get_settings()
        summary, feature_path, feature_rows = train_from_postgres(settings=settings, rolling=True)
        if not bool(summary["promoted"].all()):
            raise RuntimeError("Incomplete LightGBM model set; see outputs/training_alerts.json")
        return {
            "seeded": bool(load_summary["seeded"]),
            "feature_rows": int(feature_rows),
            "trained_models": len(summary),
            "feature_path": str(feature_path),
        }

    @task(execution_timeout=timedelta(hours=2))
    def evaluate_models(training_summary: dict) -> dict:
        from demand_forecasting.config import get_settings
        from demand_forecasting.jobs import evaluate_saved_predictions

        paths = evaluate_saved_predictions(get_settings(), rolling=True)
        return {
            "trained_models": training_summary["trained_models"],
            "diagnostics": sorted(paths),
        }

    loaded = ensure_historical_dataset()
    trained = train_models(loaded)
    evaluate_models(trained)


demand_bootstrap()
