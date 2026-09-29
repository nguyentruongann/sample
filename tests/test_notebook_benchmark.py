"""The historical notebook comparison must not mutate or include live data."""

from types import SimpleNamespace

import pandas as pd

import demand_forecasting.database as database
import demand_forecasting.feature_engineering as feature_engineering
import demand_forecasting.incremental_data as incremental_data
import demand_forecasting.train as train
from demand_forecasting.jobs import train_from_postgres


def test_historical_benchmark_only_reads_original_history(tmp_path, monkeypatch):
    raw = pd.DataFrame({
        "period_datetime_utc": pd.to_datetime([
            "2026-09-10T16:50:00Z", "2026-09-11T00:00:00Z"
        ]),
        "travel_mode": ["Car", "Car"],
        "total_demand": [4, 8],
    })
    monkeypatch.setattr(database, "load_raw_demand", lambda settings: raw.copy())
    monkeypatch.setattr(incremental_data, "backfill_live_history", lambda **kwargs: (
        _ for _ in ()).throw(AssertionError("Historical benchmark must not write live rows"))
    )
    monkeypatch.setattr(feature_engineering, "build_features", lambda frame, **kwargs: frame.copy())
    seen = []

    def record_training(features, **kwargs):
        seen.append((features.copy(), kwargs["rolling"]))
        return pd.DataFrame({"WMAPE_%": [15.0]})

    monkeypatch.setattr(train, "train_all", record_training)
    settings = SimpleNamespace(data_dir=tmp_path, local_timezone="Asia/Ho_Chi_Minh")
    summary, path, rows = train_from_postgres(settings=settings, rolling=False)
    assert rows == 1
    assert seen[0][1] is False
    assert seen[0][0]["total_demand"].tolist() == [4]
    assert path.name == "notebook_benchmark_features.parquet"
    assert summary["WMAPE_%"].tolist() == [15.0]
