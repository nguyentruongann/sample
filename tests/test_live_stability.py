"""The fake extension must preserve the notebook's short-lag predictability."""

import pandas as pd

from demand_forecasting.fake_data import LOCAL_END
from demand_forecasting.jobs import check_live_lag_stability


def _features(live_lag: float) -> pd.DataFrame:
    historical = pd.date_range(LOCAL_END - pd.Timedelta(days=7),
                               periods=1008, freq="10min")
    live = pd.date_range(LOCAL_END, periods=1009, freq="10min")
    return pd.DataFrame({
        "bucket_start": historical.append(live),
        "travel_mode": ["Car"] * (len(historical) + len(live)),
        "demand_h10": [10.0] * (len(historical) + len(live)),
        "lag_30m": [8.0] * len(historical) + [live_lag] * len(live),
    })


def test_stable_live_extension_passes_preflight():
    scores = check_live_lag_stability(_features(7.5))
    assert scores["Car"] == (20.0, 25.0)


def test_shuffled_live_extension_is_rejected_before_training():
    try:
        check_live_lag_stability(_features(0.0))
    except RuntimeError as exc:
        assert "temporal pattern drifted" in str(exc)
    else:
        raise AssertionError("Fake data with 100% lag WMAPE must be rejected")
