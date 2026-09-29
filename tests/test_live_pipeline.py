import numpy as np
import pandas as pd
import pandas.testing as pdt

from demand_forecasting.fake_data import COMMON_HEX, CONTRACT, N_DAYS, LOCAL_END
from demand_forecasting.evaluate import wmape
from demand_forecasting.feature_engineering import FEATURE_COLUMNS
from demand_forecasting.incremental_data import generate_live_bucket
from demand_forecasting.incremental_data import backfill_live_history, live_expected_rows, _observed_days
from demand_forecasting.predict import build_online_features


def test_live_bucket_is_complete_unique_and_deterministic():
    timestamp = "2026-09-22T10:03:17Z"
    first = generate_live_bucket(timestamp)
    second = generate_live_bucket(timestamp)

    assert len(first) == live_expected_rows(timestamp)
    assert 0 < first["hex_id_7"].nunique() <= len(COMMON_HEX)
    assert first["period_datetime_utc"].nunique() == 1
    assert first["period_datetime_utc"].iloc[0] == pd.Timestamp("2026-09-22T10:00:00Z")
    assert not first.duplicated(["travel_mode", "hex_id_7", "period_datetime_utc"]).any()
    pdt.assert_frame_equal(first, second)


def test_live_coverage_and_marks_follow_notebook_contract():
    for mode in ("Car", "Motorcycle"):
        observed = _observed_days(mode)
        assert observed.shape == (N_DAYS, len(COMMON_HEX))
        assert 0 <= 144 * observed.sum() - CONTRACT[mode]["n"] < 144
    sample = pd.concat([
        generate_live_bucket(t) for t in pd.date_range(
            "2026-09-11", periods=7 * 144, freq="10min", tz="Asia/Ho_Chi_Minh"
        )
    ], ignore_index=True)
    for mode, group in sample.groupby("travel_mode"):
        demand = group["total_demand"].to_numpy()
        contract = CONTRACT[mode]
        assert abs(demand.mean() / contract["mean"] - 1) < .15
        assert abs(np.quantile(demand, .95) / contract["p95"] - 1) < .20
        assert .001 < np.mean(demand == 0) < .005
        indexed = group.set_index(["hex_id_7", "period_datetime_utc"])["total_demand"]
        lag_lookup = pd.MultiIndex.from_arrays([
            group["hex_id_7"],
            group["period_datetime_utc"] - pd.Timedelta(minutes=30),
        ])
        lag_30m = indexed.reindex(lag_lookup).to_numpy()
        available = np.isfinite(lag_30m)
        assert wmape(demand[available], lag_30m[available]) < 35
        zeros = group.loc[group["total_demand"] == 0, ["hex_id_7", "period_datetime_utc"]]
        for _, cell in zeros.groupby("hex_id_7"):
            assert (cell["period_datetime_utc"].sort_values().diff().dropna()
                    >= pd.Timedelta(minutes=20)).all()


def test_online_features_use_exact_historical_offsets():
    periods = 8 * 24 * 6
    timestamps = pd.date_range("2026-01-01", periods=periods, freq="10min", tz="UTC")
    demand = np.arange(periods, dtype=np.int16)
    history = pd.DataFrame(
        {
            "period_datetime_utc": timestamps,
            "hex_id_7": "8765b56e9ffffff",
            "travel_mode": "Car",
            "total_demand": demand,
            "is_holiday": np.int8(0),
        }
    )
    forecast_start = timestamps[-1] + pd.Timedelta(minutes=10)
    feature_columns = list(FEATURE_COLUMNS) + [
        "naive_recent_h30",
        "naive_1d_h30",
        "naive_7d_h30",
    ]
    artifact = {
        "travel_mode": "Car",
        "hex_categories": ["8765b56e9ffffff"],
        "training_origin": "2026-01-01T07:00:00+07:00",
        "feature_columns": feature_columns,
    }

    features, returned_columns = build_online_features(
        history, forecast_start, artifact, horizon_minutes=30
    )
    row = features.iloc[0]

    assert returned_columns == feature_columns
    assert row["lag_30m"] == demand[-3]
    assert row["lag_1d"] == demand[-144]
    assert row["lag_7d"] == demand[-1008]
    assert row["naive_recent_h30"] == demand[-3] + demand[-4] + demand[-5]


def test_backfill_only_generates_missing_closed_buckets(monkeypatch):
    import demand_forecasting.database as db
    from demand_forecasting.config import get_settings

    latest = pd.Timestamp("2026-09-25T02:00:00Z")
    seen = []
    monkeypatch.setattr(db, "latest_observation_timestamp", lambda settings: latest)
    monkeypatch.setattr(db, "live_bucket_row_counts", lambda start, end, settings: {
        timestamp: live_expected_rows(timestamp) for timestamp in
        pd.date_range(start, latest, freq="10min")
    })
    monkeypatch.setattr(db, "upsert_raw_demand", lambda frame, settings: seen.append(frame.copy()) or len(frame))
    summary = backfill_live_history(
        end_utc="2026-09-25T02:30:00Z", settings=get_settings(), batch_buckets=2
    )
    assert summary["buckets"] == 2
    assert len(seen) == 1
    assert list(seen[0]["period_datetime_utc"].drop_duplicates()) == [
        latest + pd.Timedelta(minutes=10), latest + pd.Timedelta(minutes=20)
    ]


def test_backfill_repairs_missing_and_partial_buckets_without_replacing_existing(monkeypatch):
    import demand_forecasting.database as db
    from demand_forecasting.config import get_settings

    start = pd.Timestamp("2026-09-25T02:00:00Z")
    seen = []
    monkeypatch.setattr(db, "latest_observation_timestamp", lambda settings: start + pd.Timedelta(minutes=10))
    def bucket_counts(scan_start, scan_end, settings):
        counts = {t: live_expected_rows(t) for t in
                  pd.date_range(scan_start, scan_end, freq="10min", inclusive="left")}
        counts[start] = 1
        counts.pop(start + pd.Timedelta(minutes=20))
        return counts

    monkeypatch.setattr(db, "live_bucket_row_counts", bucket_counts)
    monkeypatch.setattr(db, "upsert_raw_demand", lambda frame, settings: seen.append(frame.copy()) or len(frame))
    summary = backfill_live_history(end_utc=start + pd.Timedelta(minutes=30),
                                    settings=get_settings(), batch_buckets=2)
    assert summary["buckets"] == 2
    assert list(seen[0]["period_datetime_utc"].drop_duplicates()) == [
        start, start + pd.Timedelta(minutes=20)
    ]


def test_full_scan_upgrades_live_generator_and_rebuilds_only_live_buckets(monkeypatch):
    import demand_forecasting.database as db
    from demand_forecasting.config import get_settings
    from demand_forecasting.incremental_data import LIVE_GENERATOR_VERSION

    live_start = LOCAL_END.tz_convert("UTC")
    upgrades, writes = [], []
    monkeypatch.setattr(db, "ensure_live_generator_version", lambda version, cutoff, settings:
                        upgrades.append((version, cutoff)) or True)
    monkeypatch.setattr(db, "latest_observation_timestamp", lambda settings: None)
    monkeypatch.setattr(db, "live_bucket_row_counts", lambda start, end, settings: {})
    monkeypatch.setattr(db, "upsert_raw_demand", lambda frame, settings:
                        writes.append(frame.copy()) or len(frame))
    result = backfill_live_history(end_utc=live_start + pd.Timedelta(minutes=20),
                                   settings=get_settings(), full_scan=True)
    assert upgrades == [(LIVE_GENERATOR_VERSION, live_start)]
    assert result["buckets"] == 2
    assert min(writes[0]["period_datetime_utc"]) == live_start
