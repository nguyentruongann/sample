"""A delayed forecast must use the current database head, not an old DAG XCom."""

from types import SimpleNamespace

import pandas as pd

import demand_forecasting.database as database
import demand_forecasting.incremental_data as incremental_data
import demand_forecasting.jobs as jobs
import demand_forecasting.predict as predict


def test_delayed_forecast_repairs_missing_buckets_and_reads_latest_again(monkeypatch):
    settings = object()
    boundary = pd.Timestamp("2026-09-26T03:00:00Z")
    state = {"latest": pd.Timestamp("2026-09-25T19:10:00Z")}
    repaired = []

    monkeypatch.setattr(incremental_data, "normalize_bucket_start", lambda: boundary)
    monkeypatch.setattr(database, "latest_observation_timestamp", lambda _settings: state["latest"])

    def backfill(*, end_utc, settings):
        assert end_utc == boundary
        state["latest"] = boundary - pd.Timedelta(minutes=10)
        repaired.append(end_utc)
        return {"buckets": 46}

    def forecast(*, settings):
        assert state["latest"] == boundary - pd.Timedelta(minutes=10)
        return SimpleNamespace(forecast_start_utc=state["latest"] + pd.Timedelta(minutes=10))

    monkeypatch.setattr(incremental_data, "backfill_live_history", backfill)
    monkeypatch.setattr(predict, "predict_and_store", forecast)

    result = jobs.forecast_live_after_catchup(settings)
    assert repaired == [boundary]
    assert result.forecast_start_utc == boundary


def test_current_forecast_does_not_rebuild_live_history(monkeypatch):
    settings = object()
    boundary = pd.Timestamp("2026-09-26T03:00:00Z")
    monkeypatch.setattr(incremental_data, "normalize_bucket_start", lambda: boundary)
    monkeypatch.setattr(database, "latest_observation_timestamp",
                        lambda _settings: boundary - pd.Timedelta(minutes=10))
    monkeypatch.setattr(incremental_data, "backfill_live_history",
                        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected backfill")))
    monkeypatch.setattr(predict, "predict_and_store",
                        lambda *, settings: SimpleNamespace(forecast_start_utc=boundary))
    assert jobs.forecast_live_after_catchup(settings).forecast_start_utc == boundary
