import numpy as np
import pandas as pd

from demand_forecasting.feature_engineering import (
    build_features,
    prepare_horizon_frame,
)


def make_raw(periods=8 * 24 * 6):
    timestamps = pd.date_range("2026-01-01", periods=periods, freq="10min", tz="UTC")
    return pd.DataFrame(
        {
            "period_datetime_utc": timestamps,
            "hex_id_7": "8765b56e9ffffff",
            "travel_mode": "Car",
            "total_demand": np.arange(periods, dtype=np.int64) % 100 + 1,
            "is_holiday": np.zeros(periods, dtype=np.int8),
        }
    )


def test_exact_lags_match_real_timestamps():
    raw = make_raw()
    features = build_features(raw, drop_incomplete=False)
    row = features.iloc[7 * 24 * 6]
    source = raw["total_demand"].to_numpy()
    position = 7 * 24 * 6

    assert row["lag_30m"] == source[position - 3]
    assert row["lag_1d"] == source[position - 144]
    assert row["lag_7d"] == source[position - 1008]


def test_missing_bucket_does_not_turn_into_wrong_shift():
    raw = make_raw(periods=30).drop(index=5).reset_index(drop=True)
    features = build_features(raw, drop_incomplete=False)
    target_time = pd.Timestamp("2026-01-01 01:30:00", tz="UTC")
    row = features.loc[features["period_datetime_utc"] == target_time].iloc[0]

    assert np.isnan(row["lag_40m"])
    expected_lag_30m = raw.loc[
        raw["period_datetime_utc"] == target_time - pd.Timedelta(minutes=30),
        "total_demand",
    ].iloc[0]
    assert row["lag_30m"] == expected_lag_30m


def test_h30_target_is_three_exact_buckets():
    raw = make_raw(periods=40)
    features = build_features(raw, drop_incomplete=False)
    prepared, _, target = prepare_horizon_frame(features, 30)
    first = prepared.iloc[0]
    start = first["period_datetime_utc"]
    expected = raw.loc[
        raw["period_datetime_utc"].isin(
            [start + pd.Timedelta(minutes=offset) for offset in (0, 10, 20)]
        ),
        "total_demand",
    ].sum()
    assert target == "demand_h30"
    assert first[target] == expected
