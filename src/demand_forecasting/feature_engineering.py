"""Leakage-safe feature and multi-horizon target construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

RAW_COLUMNS = {
    "period_datetime_utc",
    "hex_id_7",
    "travel_mode",
    "total_demand",
    "is_holiday",
}

LAG_MINUTES = (30, 40, 50, 60, 90, 120, 1440, 10080)

FEATURE_COLUMNS = [
    "hex_code",
    "slot_10m",
    "day_of_week",
    "is_weekend",
    "is_holiday",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "trend_day",
    "lag_30m",
    "lag_40m",
    "lag_50m",
    "lag_60m",
    "lag_90m",
    "lag_120m",
    "lag_1d",
    "lag_7d",
    "recent_mean_30_60m",
    "recent_std_30_60m",
    "recent_trend",
]


def validate_raw_demand(frame: pd.DataFrame) -> None:
    missing = sorted(RAW_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Raw demand is missing columns: {missing}")

    duplicate_mask = frame.duplicated(
        ["travel_mode", "hex_id_7", "period_datetime_utc"], keep=False
    )
    if duplicate_mask.any():
        sample = frame.loc[
            duplicate_mask,
            ["travel_mode", "hex_id_7", "period_datetime_utc"],
        ].head(5)
        raise ValueError(f"Duplicate raw keys detected:\n{sample}")


def _history_series(frame: pd.DataFrame, value_column: str) -> pd.Series:
    index = pd.MultiIndex.from_arrays(
        [frame["travel_mode"], frame["hex_id_7"], frame["bucket_start"]],
        names=["travel_mode", "hex_id_7", "bucket_start"],
    )
    history = pd.Series(frame[value_column].to_numpy(), index=index)
    if not history.index.is_unique:
        raise ValueError("History key must be unique by mode, hex and timestamp")
    return history


def exact_offset_values(
    frame: pd.DataFrame,
    history: pd.Series,
    offset_minutes: int,
) -> np.ndarray:
    """Return values at T + offset for the same travel mode and H3 cell."""

    lookup = pd.MultiIndex.from_arrays(
        [
            frame["travel_mode"],
            frame["hex_id_7"],
            frame["bucket_start"] + pd.Timedelta(minutes=offset_minutes),
        ]
    )
    return history.reindex(lookup).to_numpy(dtype=np.float32)


def build_features(
    raw_frame: pd.DataFrame,
    local_timezone: str = "Asia/Ho_Chi_Minh",
    drop_incomplete: bool = True,
) -> pd.DataFrame:
    """Build the same base feature set used by the original notebook."""

    validate_raw_demand(raw_frame)
    frame = raw_frame.copy()
    frame["period_datetime_utc"] = pd.to_datetime(
        frame["period_datetime_utc"], utc=True, errors="raise"
    )
    frame["bucket_start"] = frame["period_datetime_utc"].dt.tz_convert(local_timezone)
    frame["demand_h10"] = pd.to_numeric(frame["total_demand"], errors="raise").astype("int16")
    frame["is_holiday"] = pd.to_numeric(frame["is_holiday"], errors="raise").astype("int8")

    frame = frame.sort_values(
        ["travel_mode", "hex_id_7", "bucket_start"], kind="stable"
    ).reset_index(drop=True)

    hex_categories = sorted(frame["hex_id_7"].astype(str).unique())
    frame["hex_id_7"] = pd.Categorical(frame["hex_id_7"], categories=hex_categories)
    frame["hex_code"] = frame["hex_id_7"].cat.codes.astype("int16")

    frame["slot_10m"] = (
        frame["bucket_start"].dt.hour * 6 + frame["bucket_start"].dt.minute // 10
    ).astype("int16")
    frame["day_of_week"] = frame["bucket_start"].dt.dayofweek.astype("int8")
    frame["is_weekend"] = (frame["day_of_week"] >= 5).astype("int8")

    hour = frame["bucket_start"].dt.hour + frame["bucket_start"].dt.minute / 60.0
    frame["hour_sin"] = np.sin(2 * np.pi * hour / 24).astype("float32")
    frame["hour_cos"] = np.cos(2 * np.pi * hour / 24).astype("float32")
    frame["dow_sin"] = np.sin(2 * np.pi * frame["day_of_week"] / 7).astype("float32")
    frame["dow_cos"] = np.cos(2 * np.pi * frame["day_of_week"] / 7).astype("float32")

    local_start = frame["bucket_start"].min()
    frame["trend_day"] = ((frame["bucket_start"] - local_start).dt.total_seconds() / 86_400).astype(
        "float32"
    )

    history = _history_series(frame, "demand_h10")
    for minutes in LAG_MINUTES:
        if minutes == 1440:
            column = "lag_1d"
        elif minutes == 10080:
            column = "lag_7d"
        else:
            column = f"lag_{minutes}m"
        frame[column] = exact_offset_values(frame, history, -minutes)

    recent_columns = ["lag_30m", "lag_40m", "lag_50m", "lag_60m"]
    frame["recent_mean_30_60m"] = frame[recent_columns].mean(axis=1).astype("float32")
    frame["recent_std_30_60m"] = frame[recent_columns].std(axis=1, ddof=0).astype("float32")
    frame["recent_trend"] = (frame["lag_30m"] - frame["lag_60m"]).astype("float32")

    if drop_incomplete:
        frame = frame.loc[frame["lag_30m"].notna()].copy().reset_index(drop=True)

    # trend_day is measured from the first raw bucket, not the first retained row.
    frame.attrs["training_origin"] = local_start.isoformat()
    return frame


def _sum_exact_offsets(
    frame: pd.DataFrame,
    history: pd.Series,
    offsets_minutes: list[int],
) -> np.ndarray:
    values = [exact_offset_values(frame, history, offset) for offset in offsets_minutes]
    matrix = np.column_stack(values)
    valid = np.isfinite(matrix).all(axis=1)
    result = np.full(len(frame), np.nan, dtype=np.float32)
    result[valid] = matrix[valid].sum(axis=1, dtype=np.float32)
    return result


def prepare_horizon_frame(
    base_frame: pd.DataFrame,
    horizon_minutes: int,
) -> tuple[pd.DataFrame, list[str], str]:
    """Add an exact future target and horizon-specific naive features."""

    if horizon_minutes <= 0 or horizon_minutes % 10:
        raise ValueError("horizon_minutes must be a positive multiple of 10")

    frame = base_frame.copy()
    target_column = f"demand_h{horizon_minutes}"

    if horizon_minutes == 10:
        return frame, list(FEATURE_COLUMNS), "demand_h10"

    history = _history_series(frame, "demand_h10")
    target_offsets = list(range(0, horizon_minutes, 10))
    number_of_buckets = horizon_minutes // 10

    frame[target_column] = _sum_exact_offsets(frame, history, target_offsets)

    recent_column = f"naive_recent_h{horizon_minutes}"
    day_column = f"naive_1d_h{horizon_minutes}"
    week_column = f"naive_7d_h{horizon_minutes}"

    frame[recent_column] = _sum_exact_offsets(
        frame,
        history,
        [-(30 + 10 * index) for index in range(number_of_buckets)],
    )
    frame[day_column] = _sum_exact_offsets(
        frame,
        history,
        [-1440 + offset for offset in target_offsets],
    )
    frame[week_column] = _sum_exact_offsets(
        frame,
        history,
        [-10080 + offset for offset in target_offsets],
    )

    feature_columns = list(FEATURE_COLUMNS) + [recent_column, day_column, week_column]
    frame = frame.loc[frame[target_column].notna() & frame[recent_column].notna()].reset_index(
        drop=True
    )
    return frame, feature_columns, target_column
