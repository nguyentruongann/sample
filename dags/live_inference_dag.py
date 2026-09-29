"""Every 10 minutes: ingest one closed fake bucket and predict the next bucket."""

from __future__ import annotations

from datetime import timedelta

import pendulum
from airflow.sdk import dag, get_current_context, task


LOCAL_TIMEZONE = "Asia/Ho_Chi_Minh"
BUCKET_MINUTES = 10


def _resolve_closed_bucket_start_utc() -> pendulum.DateTime:
    """Xác định bucket 10 phút vừa đóng cho scheduled và manual run."""

    context = get_current_context()

    # Scheduled run thường có data_interval_end.
    # Manual run có thể chỉ có logical_date hoặc không có cả hai.
    scheduled_anchor = (
        context.get("data_interval_end")
        or context.get("logical_date")
        or pendulum.now(LOCAL_TIMEZONE)
    )

    # A delayed task must catch up to wall time rather than forecast an old run.
    anchor_local = max(
        pendulum.instance(scheduled_anchor).in_timezone(LOCAL_TIMEZONE),
        pendulum.now(LOCAL_TIMEZONE),
    )

    # Làm tròn xuống mốc 10 phút hiện tại.
    current_boundary = anchor_local.replace(
        minute=(anchor_local.minute // BUCKET_MINUTES) * BUCKET_MINUTES,
        second=0,
        microsecond=0,
    )

    # Lấy bucket vừa đóng, không lấy bucket đang diễn ra.
    closed_bucket_start = current_boundary.subtract(
        minutes=BUCKET_MINUTES
    )

    return closed_bucket_start.in_timezone("UTC")


@dag(
    dag_id="demand_live_inference",
    schedule="*/10 * * * *",
    start_date=pendulum.datetime(
        2026,
        1,
        1,
        tz=LOCAL_TIMEZONE,
    ),
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner": "demand-forecasting",
        "retries": 2,
    },
    tags=["demand", "inference", "10-minute"],
)
def demand_live_inference():
    @task(
        pool="db_write",
        execution_timeout=timedelta(minutes=45),  # Allows catch-up after host downtime.
    )
    def ingest_closed_bucket() -> dict:
        from demand_forecasting.incremental_data import (
            backfill_live_history,
            generate_and_store_bucket,
        )

        bucket_start_utc = _resolve_closed_bucket_start_utc()

        # Heal a missed scheduled interval before generating the latest bucket.
        # end_utc is exclusive, so the just-closed bucket is handled below.
        backfill_live_history(end_utc=bucket_start_utc)

        summary = generate_and_store_bucket(
            bucket_start_utc
        )

        return summary.to_dict()

    @task(
        pool="db_write",
        retries=1,
        execution_timeout=timedelta(minutes=45),
    )
    def forecast_next_bucket(ingestion: dict) -> dict:
        import pandas as pd

        from demand_forecasting.jobs import forecast_live_after_catchup

        summary = forecast_live_after_catchup()

        result = summary.to_dict()
        result["source_bucket_start_utc"] = (
            pd.Timestamp(result["forecast_start_utc"])
            - pd.Timedelta(minutes=BUCKET_MINUTES)
        ).isoformat()
        result["ingested_bucket_start_utc"] = ingestion["bucket_start_utc"]

        return result

    ingestion = ingest_closed_bucket()
    forecast_next_bucket(ingestion)


demand_live_inference()
