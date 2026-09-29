"""Error, stability, peak, weakness and trend diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import calculate_metrics

GROUP_KEYS = ["horizon", "travel_mode", "model"]


def _validate_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "bucket_start",
        "hex_id_7",
        "travel_mode",
        "horizon",
        "model",
        "actual",
        "prediction",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Predictions are missing columns: {missing}")

    result = frame.copy()
    result["bucket_start"] = pd.to_datetime(result["bucket_start"], errors="raise")
    result["actual"] = pd.to_numeric(result["actual"], errors="coerce")
    result["prediction"] = pd.to_numeric(result["prediction"], errors="coerce")
    valid = np.isfinite(result["actual"]) & np.isfinite(result["prediction"])
    result = result.loc[valid].copy()
    result["error"] = result["prediction"] - result["actual"]
    result["abs_error"] = result["error"].abs()
    result["date"] = result["bucket_start"].dt.floor("D")
    result["hour"] = result["bucket_start"].dt.hour
    result["day_of_week"] = result["bucket_start"].dt.dayofweek
    return result


def error_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = _validate_predictions(predictions)
    rows = []
    for keys, group in frame.groupby(GROUP_KEYS, observed=True, sort=True):
        metrics = calculate_metrics(group["actual"], group["prediction"])
        metrics.update(dict(zip(GROUP_KEYS, keys)))
        rows.append(metrics)
    return pd.DataFrame(rows).sort_values(GROUP_KEYS + ["WMAPE_%"])


def demand_band_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = _validate_predictions(predictions)
    rows = []
    for keys, group in frame.groupby(GROUP_KEYS, observed=True, sort=True):
        p50, p90 = group["actual"].quantile([0.50, 0.90])
        band = np.select(
            [group["actual"] <= p50, group["actual"] < p90],
            ["Low", "Medium"],
            default="Peak",
        )
        for label in ("Low", "Medium", "Peak"):
            selected = group.loc[band == label]
            metrics = calculate_metrics(selected["actual"], selected["prediction"])
            metrics.update(dict(zip(GROUP_KEYS, keys)))
            metrics.update({"demand_band": label, "P50_demand": p50, "P90_demand": p90})
            rows.append(metrics)
    return pd.DataFrame(rows)


def daily_stability(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = _validate_predictions(predictions)
    rows = []
    daily_parts = []
    for keys, group in frame.groupby(GROUP_KEYS, observed=True, sort=True):
        daily_rows = []
        for date, day in group.groupby("date", observed=True, sort=True):
            metrics = calculate_metrics(day["actual"], day["prediction"])
            daily_rows.append({"date": date, **metrics})
        daily = pd.DataFrame(daily_rows)
        for key, value in zip(GROUP_KEYS, keys):
            daily[key] = value
        daily_parts.append(daily)

        midpoint = max(1, len(daily) // 2)
        first = daily.iloc[:midpoint]["WMAPE_%"].mean()
        second = daily.iloc[midpoint:]["WMAPE_%"].mean()
        rows.append(
            {
                **dict(zip(GROUP_KEYS, keys)),
                "days": len(daily),
                "daily_WMAPE_mean_%": daily["WMAPE_%"].mean(),
                "daily_WMAPE_std_pp": daily["WMAPE_%"].std(ddof=1),
                "daily_WMAPE_min_%": daily["WMAPE_%"].min(),
                "daily_WMAPE_max_%": daily["WMAPE_%"].max(),
                "drift_pp": second - first,
            }
        )
    return pd.DataFrame(rows), pd.concat(daily_parts, ignore_index=True)


def peak_summary(predictions: pd.DataFrame, quantile: float = 0.90) -> pd.DataFrame:
    frame = _validate_predictions(predictions)
    rows = []
    for keys, group in frame.groupby(GROUP_KEYS, observed=True, sort=True):
        threshold = float(group["actual"].quantile(quantile))
        actual_peak = group["actual"] >= threshold
        predicted_peak = group["prediction"] >= threshold
        true_positive = int((actual_peak & predicted_peak).sum())
        precision = true_positive / max(1, int(predicted_peak.sum()))
        recall = true_positive / max(1, int(actual_peak.sum()))
        peak_metrics = calculate_metrics(
            group.loc[actual_peak, "actual"], group.loc[actual_peak, "prediction"]
        )
        rows.append(
            {
                **dict(zip(GROUP_KEYS, keys)),
                "peak_threshold_P90": threshold,
                "peak_rows": int(actual_peak.sum()),
                "peak_WMAPE_%": peak_metrics["WMAPE_%"],
                "peak_bias_%": peak_metrics["bias_%"],
                "peak_precision_%": 100.0 * precision,
                "peak_recall_%": 100.0 * recall,
            }
        )
    return pd.DataFrame(rows)


def weakness_summary(
    predictions: pd.DataFrame,
    group_column: str,
    minimum_rows: int,
    top_n: int = 10,
) -> pd.DataFrame:
    frame = _validate_predictions(predictions)
    if group_column not in frame:
        raise KeyError(group_column)
    rows = []
    keys = GROUP_KEYS + [group_column]
    for values, group in frame.groupby(keys, observed=True, sort=True):
        if len(group) < minimum_rows:
            continue
        metrics = calculate_metrics(group["actual"], group["prediction"])
        metrics.update(dict(zip(keys, values)))
        rows.append(metrics)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return (
        result.sort_values(GROUP_KEYS + ["WMAPE_%"], ascending=[True, True, True, False])
        .groupby(GROUP_KEYS, observed=True, sort=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def trend_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = _validate_predictions(predictions)
    rows = []
    for keys, group in frame.groupby(GROUP_KEYS, observed=True, sort=True):
        daily = group.groupby("date", observed=True, sort=True).agg(
            actual_volume=("actual", "sum"),
            predicted_volume=("prediction", "sum"),
        )
        x = np.arange(len(daily), dtype=np.float64)
        actual_slope = np.polyfit(x, daily["actual_volume"], 1)[0] if len(daily) > 1 else 0.0
        prediction_slope = np.polyfit(x, daily["predicted_volume"], 1)[0] if len(daily) > 1 else 0.0
        rows.append(
            {
                **dict(zip(GROUP_KEYS, keys)),
                "days": len(daily),
                "actual_slope_per_day": actual_slope,
                "prediction_slope_per_day": prediction_slope,
                "daily_volume_correlation": daily["actual_volume"].corr(daily["predicted_volume"]),
            }
        )
    return pd.DataFrame(rows)


def export_diagnostics(predictions: pd.DataFrame, output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stability, daily = daily_stability(predictions)
    tables = {
        "error_summary.csv": error_summary(predictions),
        "error_by_demand_band.csv": demand_band_summary(predictions),
        "stability_summary.csv": stability,
        "daily_stability.csv": daily,
        "peak_summary.csv": peak_summary(predictions),
        "weak_hours.csv": weakness_summary(predictions, "hour", 100, 5),
        "weak_hex.csv": weakness_summary(predictions, "hex_id_7", 200, 10),
        "trend_summary.csv": trend_summary(predictions),
    }
    paths = {}
    for filename, table in tables.items():
        path = output_dir / filename
        table.to_csv(path, index=False)
        paths[filename] = path
    return paths
