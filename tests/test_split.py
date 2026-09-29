import numpy as np
import pandas as pd

from demand_forecasting.data_split import resolve_boundaries, time_based_split


def make_frame():
    return pd.DataFrame({
        "bucket_start": pd.date_range(
            "2026-05-01", periods=50 * 24 * 6, freq="10min", tz="Asia/Ho_Chi_Minh"
        ),
    })


def test_all_horizons_share_cutoff_and_purge_targets_at_both_boundaries():
    frame = make_frame()
    cutoff = pd.Timestamp("2026-06-08T09:40:00+07:00")
    boundaries = resolve_boundaries(frame, validation_days=10, test_start=cutoff)
    for horizon in (10, 30, 60):
        split = time_based_split(frame, horizon, boundaries=boundaries)
        assert split.test_start == cutoff
        assert split.val_start == cutoff - pd.Timedelta(days=10)
        assert (frame.iloc[split.train_index]["bucket_start"] + pd.Timedelta(minutes=horizon) <= split.val_start).all()
        assert (frame.iloc[split.val_index]["bucket_start"] + pd.Timedelta(minutes=horizon) <= cutoff).all()
        assert (frame.iloc[split.test_index]["bucket_start"] >= cutoff).all()
        assert len(split.purged_index) == (horizon // 10 - 1) * 2


def test_sampling_is_reproducible_and_does_not_count_as_purge():
    frame = make_frame()
    args = dict(horizon_minutes=30, test_start="2026-06-08T09:40:00+07:00", validation_days=10,
                max_train_rows=20, seed=7)
    first = time_based_split(frame, **args)
    second = time_based_split(frame, **args)
    assert len(first.train_index) == 20
    assert np.array_equal(first.train_index, second.train_index)
    assert len(first.purged_index) == 4


def test_rolling_split_holds_out_the_latest_seven_days():
    frame = make_frame()
    split = time_based_split(frame, 60, validation_days=10, rolling=True, rolling_test_days=7)
    assert len(split.train_index) and len(split.val_index) and len(split.test_index)
    assert split.test_start == frame["bucket_start"].max() + pd.Timedelta(minutes=10) - pd.Timedelta(days=7)
    assert (frame.iloc[split.val_index]["bucket_start"] + pd.Timedelta(minutes=60) <= split.test_start).all()
    assert (frame.iloc[split.test_index]["bucket_start"] >= split.test_start).all()


def test_default_rolling_window_has_90_train_30_val_30_test_days():
    frame = pd.DataFrame({"bucket_start": pd.date_range(
        "2026-05-01", periods=150 * 144, freq="10min", tz="Asia/Ho_Chi_Minh"
    )})
    split = time_based_split(frame, 60, rolling=True)
    assert split.val_start - split.train_start == pd.Timedelta(days=90)
    assert split.test_start - split.val_start == pd.Timedelta(days=30)
    assert split.test_end - split.test_start == pd.Timedelta(days=30)
    assert split.test_end == frame["bucket_start"].max() + pd.Timedelta(minutes=10)
    assert (frame.iloc[split.train_index]["bucket_start"] >= split.train_start).all()
    assert (frame.iloc[split.test_index]["bucket_start"] + pd.Timedelta(minutes=60) <= split.test_end).all()


def test_notebook_test_ends_before_live_generator_begins():
    frame = pd.DataFrame({"bucket_start": pd.date_range(
        "2026-05-01", "2026-09-26", freq="10min", tz="Asia/Ho_Chi_Minh", inclusive="left"
    )})
    cutoff = pd.Timestamp("2026-08-15T09:40:00+07:00")
    end = pd.Timestamp("2026-09-11T00:00:00+07:00")
    split = time_based_split(frame, 60, boundaries=(cutoff-pd.Timedelta(days=30), cutoff),
                             historical_end=end)
    test_times = frame.iloc[split.test_index]["bucket_start"]
    assert split.test_end == end
    assert test_times.min() == cutoff
    assert (test_times + pd.Timedelta(minutes=60) <= end).all()
    assert test_times.max() == end - pd.Timedelta(minutes=60)
