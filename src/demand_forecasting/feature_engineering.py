"""Leakage-safe feature and multi-horizon target construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

RAW_COLUMNS = {
    "period_datetime_utc",
    "hex_id_7",
    "travel_mode",
    "total_demand",
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
    frame["demand_h10"] = pd.to_numeric(frame["total_demand"], errors="raise").astype("float32")
    if not np.isfinite(frame["demand_h10"]).all() or (frame["demand_h10"] < 0).any():
        raise ValueError("Demand must be finite and nonnegative")
    if not frame["bucket_start"].eq(frame["bucket_start"].dt.floor("10min")).all():
        raise ValueError("Demand timestamps must be aligned to 10-minute buckets")
    # Notebook recomputes Vietnamese holidays from local dates, not the raw flag.
    import holidays
    calendar = holidays.country_holidays("VN", years=frame["bucket_start"].dt.year.unique().tolist(), observed=True)
    frame["is_holiday"] = frame["bucket_start"].dt.date.isin(calendar).astype("int8")

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

    # Notebook fresh/neighbor history is MODE_FRAMES after lag_30m filtering.
    frame = add_fresh_neighbor_features(frame)
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

    if horizon_minutes not in (10, 30, 60):
        raise ValueError("horizon_minutes must be a positive multiple of 10")

    from .notebook_contract import feature_columns
    frame = base_frame.copy()
    columns = feature_columns(horizon_minutes, FEATURE_COLUMNS)
    target_column = f"demand_h{horizon_minutes}"

    if horizon_minutes == 10:
        return frame, columns, "demand_h10"

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

    frame = frame.loc[frame[target_column].notna() & frame[recent_column].notna()].reset_index(
        drop=True
    )
    # Notebook factorizes numeric hex codes AFTER dropping incomplete horizon rows.
    # H10 keeps the global categorical code; H30/H60 use mode/horizon numeric codes.
    frame["hex_code"] = pd.factorize(frame["hex_code"], sort=True)[0].astype(np.int32)
    for column in columns:
        if column != "hex_code":
            frame[column] = frame[column].astype(np.float32)
    return frame, columns, target_column


def fresh_neighbor_features(source: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Notebook _h60_extra, using MODE_FRAMES history after lag_30m filtering.

    One mode per call. Missing neighbors are ignored (never filled with demand=0).
    Self is excluded. An isolated cell has NaN neighbor features.
    """
    import h3
    from scipy.sparse import csr_matrix

    raw = source[["hex_id_7", "bucket_start", "demand_h10"]].copy()
    raw["hex_id_7"] = raw["hex_id_7"].astype(str)
    if raw.duplicated(["hex_id_7", "bucket_start"]).any():
        raise ValueError("Duplicate history keys")
    wide = raw.pivot(index="bucket_start", columns="hex_id_7", values="demand_h10")
    times, cells = pd.DatetimeIndex(wide.index), pd.Index(wide.columns)
    values = wide.to_numpy(dtype=np.float32)
    lookup = {cell: i for i, cell in enumerate(cells)}
    edges = [(i, lookup[n]) for cell, i in lookup.items()
             for n in h3.grid_disk(cell, 1) if n != cell and n in lookup]
    if edges:
        rr, cc = np.asarray(edges, dtype=np.int32).T
        adjacency = csr_matrix((np.ones(len(rr), dtype=np.float32), (rr, cc)),
                               shape=(len(cells), len(cells)))
        sums = (adjacency @ np.nan_to_num(values, nan=0).T).T
        counts = (adjacency @ np.isfinite(values).astype(np.float32).T).T
        neighbors = np.full(values.shape, np.nan, dtype=np.float32)
        np.divide(sums, counts, out=neighbors, where=counts > 0)
    else:
        neighbors = np.full(values.shape, np.nan, dtype=np.float32)
    cell_pos = cells.get_indexer(rows["hex_id_7"].astype(str))

    def past(minutes, neighbor=False):
        time_pos = times.get_indexer(pd.DatetimeIndex(rows["bucket_start"] - pd.Timedelta(minutes=minutes)))
        found = (time_pos >= 0) & (cell_pos >= 0)
        out = np.full(len(rows), np.nan, dtype=np.float32)
        grid = neighbors if neighbor else values
        out[found] = grid[time_pos[found], cell_pos[found]]
        return out

    result = {"lag_10m": past(10), "lag_20m": past(20)}
    recent = np.column_stack([result["lag_10m"], result["lag_20m"],
                              rows[["lag_30m", "lag_40m", "lag_50m", "lag_60m"]].to_numpy(np.float32)])
    count = np.isfinite(recent).sum(axis=1)
    result["mean_10_60m"] = np.divide(np.nansum(recent, axis=1), count,
        out=np.full(len(rows), np.nan, dtype=np.float32), where=count > 0)
    result["delta_10_30m"] = result["lag_10m"] - rows["lag_30m"].to_numpy(np.float32)
    for minutes in (10, 20, 30, 60):
        result[f"n{minutes}"] = past(minutes, neighbor=True)
    result["neighbor_trend_10_30"] = result["n10"] - result["n30"]
    return pd.DataFrame(result, index=rows.index, dtype=np.float32)


def add_fresh_neighbor_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute once per mode, before making any horizon-specific target."""
    from .notebook_contract import EXTRA_FEATURES

    result = frame.copy()
    for name in EXTRA_FEATURES[60]:
        result[name] = np.float32(np.nan)
    for positions in frame.groupby("travel_mode", observed=True, sort=False).indices.values():
        rows = frame.iloc[positions]
        extra = fresh_neighbor_features(rows, rows)
        result.loc[rows.index, extra.columns] = extra
    return result
