"""Reusable batch jobs called by both the CLI and Airflow tasks."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from .config import Settings, get_settings


def check_live_lag_stability(features: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Catch a fake live generator that destroys per-H3 temporal continuity."""
    from .evaluate import wmape
    from .fake_data import LOCAL_END

    live_end = features["bucket_start"].max()
    scores: dict[str, tuple[float, float]] = {}
    if live_end < LOCAL_END + pd.Timedelta(days=7):
        return scores
    for mode in ("Car", "Motorcycle"):
        frame = features.loc[features["travel_mode"] == mode]
        notebook = frame.loc[
            (frame["bucket_start"] >= LOCAL_END - pd.Timedelta(days=7))
            & (frame["bucket_start"] < LOCAL_END)
        ]
        live = frame.loc[
            (frame["bucket_start"] >= live_end - pd.Timedelta(days=7))
            & (frame["bucket_start"] >= LOCAL_END)
        ]
        if min(len(notebook), len(live)) < 1_000:
            continue
        historical_score = wmape(notebook["demand_h10"], notebook["lag_30m"])
        live_score = wmape(live["demand_h10"], live["lag_30m"])
        scores[mode] = (historical_score, live_score)
        print(f"{mode} lag-30m baseline WMAPE | notebook={historical_score:.2f}% "
              f"live={live_score:.2f}%", flush=True)
        if not pd.notna(live_score) or live_score > max(1.5 * historical_score,
                                                       historical_score + 8):
            raise RuntimeError(
                f"{mode} fake live temporal pattern drifted: lag-30m WMAPE "
                f"{historical_score:.2f}% -> {live_score:.2f}%. "
                "Previous active models retained; check the live generator."
            )
    return scores


def forecast_live_after_catchup(settings: Settings | None = None):
    """Forecast from the current database head, even when an Airflow task is late."""
    from .database import latest_observation_timestamp
    from .incremental_data import backfill_live_history, normalize_bucket_start
    from .predict import predict_and_store

    settings = settings or get_settings()
    current_boundary = normalize_bucket_start()
    latest = latest_observation_timestamp(settings)
    last_closed = current_boundary - pd.Timedelta(minutes=10)
    if latest is None or latest < last_closed:
        repaired = backfill_live_history(end_utc=current_boundary, settings=settings)
        print(f"Forecast: filled demand through the last closed bucket: {repaired}", flush=True)
    # Resolve latest again inside predict_and_store. The upstream Airflow XCom
    # may be hours old by the time this task starts or is retried.
    return predict_and_store(settings=settings)


def train_from_postgres(
    settings: Settings | None = None,
    model_names: tuple[str, ...] | None = None,
    horizons: tuple[int, ...] | None = None,
    rolling: bool = False,
) -> tuple[pd.DataFrame, Path, int]:
    from .database import load_raw_demand
    from .fake_data import LOCAL_END
    from .feature_engineering import build_features
    from .incremental_data import backfill_live_history
    from .train import train_all

    settings = settings or get_settings()
    started = time.perf_counter()
    if rolling:
        backfill = backfill_live_history(settings=settings, full_scan=True)
        print(f"Training: live fake history synchronized: {backfill}", flush=True)
    print("Training: loading demand from PostgreSQL", flush=True)
    raw = load_raw_demand(settings)
    if not rolling:
        raw = raw.loc[
            pd.to_datetime(raw["period_datetime_utc"], utc=True) < LOCAL_END.tz_convert("UTC")
        ].copy()
    print(
        f"Training: loaded {len(raw):,} rows in {time.perf_counter() - started:.1f}s; "
        "building lag features",
        flush=True,
    )
    features = build_features(raw, local_timezone=settings.local_timezone)
    if rolling:
        check_live_lag_stability(features)
    feature_path = settings.data_dir / "processed" / (
        "demand_features.parquet" if rolling else "notebook_benchmark_features.parquet"
    )
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(feature_path, index=False)
    print(
        f"Training: saved {len(features):,} feature rows after "
        f"{time.perf_counter() - started:.1f}s; starting LightGBM",
        flush=True,
    )
    summary = train_all(
        features,
        settings=settings,
        model_names=model_names,
        horizons=horizons,
        rolling=rolling,
    )
    return summary, feature_path, len(features)


def evaluate_saved_predictions(
    settings: Settings | None = None,
    *,
    rolling: bool = False,
) -> dict[str, Path]:
    from .diagnostics import export_diagnostics

    settings = settings or get_settings()
    summary_path = settings.output_dir / (
        "weekly_retrain_summary.csv" if rolling else "notebook_benchmark_summary.csv"
    )
    if not summary_path.exists():
        raise FileNotFoundError("Run historical LightGBM backtest before exporting diagnostics")
    summary = pd.read_csv(summary_path)
    paths = [Path(path) for path in summary["predictions_path"] if isinstance(path, str)]
    expected_mode = "rolling" if rolling else "backtest"
    if (summary["model"] != "lightgbm").any() or (summary["mode"] != expected_mode).any():
        raise ValueError(f"Diagnostics must use the LightGBM {expected_mode} test")
    if not paths:
        raise FileNotFoundError(f"No prediction files found in {settings.output_dir}")
    predictions = pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)
    return export_diagnostics(predictions, settings.output_dir / "diagnostics")
