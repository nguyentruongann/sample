"""Weekly CPU retraining with rolling validation and held-out WMAPE."""

from __future__ import annotations

import os
from datetime import timedelta

import pendulum
from airflow.sdk import dag, task


@dag(
    dag_id="demand_weekly_retrain",
    schedule="0 2 * * 1",
    start_date=pendulum.datetime(2026, 1, 1, tz="Asia/Ho_Chi_Minh"),
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "demand-forecasting", "retries": 1},
    tags=["demand", "training", "weekly"],
)
def demand_weekly_retrain():
    @task(execution_timeout=timedelta(minutes=10))
    def validate_training_data() -> dict:
        from demand_forecasting.config import get_settings
        from demand_forecasting.database import database_row_count

        rows = database_row_count(get_settings())
        minimum_rows = int(os.getenv("MIN_TRAIN_ROWS", "100000"))
        if rows < minimum_rows:
            raise ValueError(f"Need at least {minimum_rows:,} rows; found {rows:,}")
        return {"source_rows": rows}

    @task(pool="ml_cpu", execution_timeout=timedelta(hours=12))
    def train_models(data_check: dict) -> dict:
        from demand_forecasting.config import get_settings
        from demand_forecasting.jobs import train_from_postgres

        summary, feature_path, feature_rows = train_from_postgres(get_settings(), rolling=True)
        if not bool(summary["promoted"].all()):
            raise RuntimeError("Incomplete LightGBM model set; see outputs/training_alerts.json")
        return {
            "source_rows": int(data_check["source_rows"]),
            "feature_rows": int(feature_rows),
            "trained_models": len(summary),
            "feature_path": str(feature_path),
            "promoted": bool(summary["promoted"].all()),
            "validation_status": "searched_before_refit",
            "val_wmape": dict(zip(
                summary["travel_mode"] + "_H" + summary["horizon"].astype(str),
                summary["VAL_WMAPE_%"].astype(float),
            )),
            "test_wmape": dict(zip(
                summary["travel_mode"] + "_H" + summary["horizon"].astype(str),
                summary["WMAPE_%"].astype(float),
            )),
        }

    train_models(validate_training_data())


demand_weekly_retrain()
