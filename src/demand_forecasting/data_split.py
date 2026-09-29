"""Shared chronological train/validation/test boundaries for every mode and horizon."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimeSplit:
    train_start: pd.Timestamp
    val_start: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_index: np.ndarray
    val_index: np.ndarray
    test_index: np.ndarray
    train_val_index: np.ndarray
    purged_index: np.ndarray

    @property
    def cutoff(self) -> pd.Timestamp:
        return self.test_start


def resolve_boundaries(
    frame: pd.DataFrame,
    *,
    train_ratio: float = 0.80,
    validation_days: int = 30,
    test_start: str | pd.Timestamp | None = None,
    rolling: bool = False,
    rolling_test_days: int = 30,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Historical cutoff is locked; rolling retrain holds out the latest days."""
    if not 0 < train_ratio < 1 or validation_days <= 0 or rolling_test_days <= 0:
        raise ValueError("Invalid train_ratio, validation_days or rolling_test_days")
    times = pd.DatetimeIndex(pd.to_datetime(frame["bucket_start"], errors="raise").unique()).sort_values()
    if len(times) < 3 or times.tz is None:
        raise ValueError("At least three timezone-aware timestamps are required")
    if rolling:
        end = times[-1] + pd.Timedelta(minutes=10) - pd.Timedelta(days=rolling_test_days)
    elif test_start is not None:
        end = pd.Timestamp(test_start)
        if end.tzinfo is None:
            raise ValueError("TEST_START must include a timezone")
        end = end.tz_convert(times.tz)
        if end < times[0] or end > times[-1]:
            raise ValueError(f"Locked TEST_START {end} is outside the available data")
    else:
        end = times[min(max(int(len(times) * train_ratio), 1), len(times) - 1)]
    start = end - pd.Timedelta(days=validation_days)
    if start <= times[0]:
        raise ValueError("Insufficient history before validation; extend data or shorten VALIDATION_DAYS")
    return start, end


def time_based_split(
    frame: pd.DataFrame,
    horizon_minutes: int,
    train_ratio: float = 0.80,
    max_train_rows: int | None = None,
    seed: int = 20260916,
    *,
    validation_days: int = 30,
    test_start: str | pd.Timestamp | None = None,
    rolling: bool = False,
    rolling_test_days: int = 30,
    train_history_days: int = 90,
    boundaries: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    historical_end: pd.Timestamp | None = None,
) -> TimeSplit:
    if horizon_minutes not in (10, 30, 60):
        raise ValueError("Only H10, H30 and H60 are supported")
    if train_history_days <= 0:
        raise ValueError("TRAIN_HISTORY_DAYS must be positive")
    val_start, cutoff = boundaries or resolve_boundaries(
        frame,
        train_ratio=train_ratio,
        validation_days=validation_days,
        test_start=test_start,
        rolling=rolling,
        rolling_test_days=rolling_test_days,
    )
    ts = pd.to_datetime(frame["bucket_start"])
    target_end = ts + pd.Timedelta(minutes=horizon_minutes)
    train_start = val_start - pd.Timedelta(days=train_history_days)
    if rolling:
        test_end = cutoff + pd.Timedelta(days=rolling_test_days)
    else:
        # Historical notebook calls provide its exact endpoint. A generic
        # caller can still request a bounded test window without that cutoff.
        test_end = (pd.Timestamp(historical_end) if historical_end is not None
                    else cutoff + pd.Timedelta(days=rolling_test_days))
        if test_end.tzinfo is None:
            raise ValueError("Historical data end must include a timezone")
        test_end = test_end.tz_convert(ts.dt.tz)
        if not cutoff < test_end:
            raise ValueError("Historical test end must follow its start")
    train = (ts >= train_start) & (ts < val_start) & (target_end <= val_start)
    val = (ts >= val_start) & (ts < cutoff) & (target_end <= cutoff)
    test = (ts >= cutoff) & (target_end <= test_end)
    train_index = np.flatnonzero(train.to_numpy())
    if max_train_rows is not None and len(train_index) > max_train_rows:
        rng = np.random.default_rng(seed)
        train_index = np.sort(rng.choice(train_index, max_train_rows, replace=False))
    val_index = np.flatnonzero(val.to_numpy())
    test_index = np.flatnonzero(test.to_numpy())
    if not len(train_index) or not len(val_index) or not len(test_index):
        raise ValueError(f"H{horizon_minutes}: train, validation or test is empty")
    purged = ((ts < val_start) & (target_end > val_start)) | (
        (ts >= val_start) & (ts < cutoff) & (target_end > cutoff)
    )
    return TimeSplit(
        train_start=max(train_start, ts.min()),
        val_start=val_start,
        test_start=cutoff,
        test_end=test_end,
        train_index=train_index,
        val_index=val_index,
        test_index=test_index,
        train_val_index=np.sort(np.r_[train_index, val_index]),
        purged_index=np.flatnonzero(purged.to_numpy()),
    )
