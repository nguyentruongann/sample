"""Online CPU inference from the latest PostgreSQL demand history."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import holidays
import joblib
import numpy as np
import pandas as pd

from .config import Settings, get_settings
from .feature_engineering import FEATURE_COLUMNS, LAG_MINUTES, fresh_neighbor_features
from .notebook_contract import FEATURE_SCHEMA, EVALUATION_SCHEMA, feature_columns as contract_columns


@dataclass(frozen=True)
class PredictionSummary:
    forecast_start_utc: str
    rows: int
    models: int
    stored_rows: int

    def to_dict(self) -> dict[str, str | int]:
        return asdict(self)


def normalize_forecast_start(value) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.floor("10min")


def _model_version(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def _lookup_history(
    hex_ids: list[str],
    timestamp: pd.Timestamp,
    history: pd.Series,
) -> np.ndarray:
    lookup = pd.MultiIndex.from_arrays(
        [hex_ids, np.repeat(timestamp, len(hex_ids))],
        names=["hex_id_7", "period_datetime_utc"],
    )
    return history.reindex(lookup).to_numpy(dtype=np.float32)


def build_online_features(
    raw_history: pd.DataFrame,
    forecast_start_utc,
    artifact: dict,
    horizon_minutes: int,
    local_timezone: str = "Asia/Ho_Chi_Minh",
) -> tuple[pd.DataFrame, list[str]]:
    """Build exactly the feature columns expected by one saved model."""

    if raw_history.empty:
        raise ValueError("Demand history is empty")
    forecast_start = normalize_forecast_start(forecast_start_utc)
    mode = str(artifact["travel_mode"])
    if artifact.get("local_timezone", local_timezone) != local_timezone:
        raise ValueError("Serving timezone differs from training")
    schema = artifact.get("feature_schema")
    if schema is not None and schema != FEATURE_SCHEMA:
        raise ValueError("Unsupported feature schema; retrain with the current pipeline")
    history_frame = raw_history.loc[raw_history["travel_mode"] == mode].copy()
    if history_frame.empty:
        raise ValueError(f"No demand history is available for {mode}")

    history_frame["period_datetime_utc"] = pd.to_datetime(
        history_frame["period_datetime_utc"], utc=True, errors="raise"
    )
    history_frame["hex_id_7"] = history_frame["hex_id_7"].astype(str)
    # Never use current/future labels; reject ambiguous duplicates rather than silently overwrite.
    history_frame = history_frame.loc[history_frame["period_datetime_utc"] < forecast_start].copy()
    if history_frame.duplicated(["hex_id_7", "period_datetime_utc"]).any():
        raise ValueError("Duplicate online demand keys")
    history = pd.Series(
        history_frame["total_demand"].to_numpy(dtype=np.float32),
        index=pd.MultiIndex.from_frame(history_frame[["hex_id_7", "period_datetime_utc"]]),
    )

    hex_ids = [str(value) for value in artifact.get("hex_categories", [])]
    if not hex_ids:
        hex_ids = sorted(history_frame["hex_id_7"].unique())
    if not hex_ids:
        raise ValueError("Model has no H3 categories")

    code_map = artifact.get("hex_code_mapping")
    if schema == FEATURE_SCHEMA and (not code_map or any(hx not in code_map for hx in hex_ids)):
        raise ValueError("Artifact is missing the trained hex-code mapping")
    local_start = forecast_start.tz_convert(local_timezone)
    training_origin = pd.Timestamp(
        artifact.get("training_origin", history_frame["period_datetime_utc"].min())
    )
    if training_origin.tzinfo is None:
        training_origin = training_origin.tz_localize(local_timezone)
    else:
        training_origin = training_origin.tz_convert(local_timezone)

    frame = pd.DataFrame(
        {
            "period_datetime_utc": forecast_start,
            "bucket_start": local_start,
            "hex_id_7": hex_ids,
            "travel_mode": mode,
            "hex_code": (np.asarray([code_map[hx] for hx in hex_ids], dtype=np.int32)
                         if code_map else np.arange(len(hex_ids), dtype=np.int16)),
        }
    )
    frame["slot_10m"] = np.int16(local_start.hour * 6 + local_start.minute // 10)
    frame["day_of_week"] = np.int8(local_start.dayofweek)
    frame["is_weekend"] = np.int8(local_start.dayofweek >= 5)
    vn_holidays = holidays.country_holidays("VN", years=[local_start.year], observed=True)
    frame["is_holiday"] = np.int8(local_start.date() in vn_holidays)

    decimal_hour = local_start.hour + local_start.minute / 60.0
    frame["hour_sin"] = np.float32(np.sin(2 * np.pi * decimal_hour / 24))
    frame["hour_cos"] = np.float32(np.cos(2 * np.pi * decimal_hour / 24))
    frame["dow_sin"] = np.float32(np.sin(2 * np.pi * local_start.dayofweek / 7))
    frame["dow_cos"] = np.float32(np.cos(2 * np.pi * local_start.dayofweek / 7))
    frame["trend_day"] = np.float32((local_start - training_origin).total_seconds() / 86_400)

    for minutes in LAG_MINUTES:
        if minutes == 1440:
            column = "lag_1d"
        elif minutes == 10080:
            column = "lag_7d"
        else:
            column = f"lag_{minutes}m"
        frame[column] = _lookup_history(
            hex_ids, forecast_start - pd.Timedelta(minutes=minutes), history
        )

    recent_columns = ["lag_30m", "lag_40m", "lag_50m", "lag_60m"]
    frame["recent_mean_30_60m"] = frame[recent_columns].mean(axis=1).astype("float32")
    frame["recent_std_30_60m"] = frame[recent_columns].std(axis=1, ddof=0).astype("float32")
    frame["recent_trend"] = (frame["lag_30m"] - frame["lag_60m"]).astype("float32")

    # MODE_FRAMES in final.ipynb contains only historical rows with lag_30m present.
    # Base lag features above use RAW history; naive/fresh/neighbor use this retained history.
    retained_history = history
    retained_frame = history_frame
    if schema == FEATURE_SCHEMA:
        past_index = pd.MultiIndex.from_arrays([history_frame["hex_id_7"],
            history_frame["period_datetime_utc"] - pd.Timedelta(minutes=30)])
        retained_frame = history_frame.loc[np.isfinite(history.reindex(past_index).to_numpy(float))].copy()
        retained_history = pd.Series(retained_frame["total_demand"].to_numpy(np.float32),
            index=pd.MultiIndex.from_frame(retained_frame[["hex_id_7", "period_datetime_utc"]]))
    if horizon_minutes > 10:
        bucket_count = horizon_minutes // 10
        target_offsets = range(0, horizon_minutes, 10)
        groups = {
            f"naive_recent_h{horizon_minutes}": [
                -(30 + 10 * index) for index in range(bucket_count)
            ],
            f"naive_1d_h{horizon_minutes}": [-1440 + offset for offset in target_offsets],
            f"naive_7d_h{horizon_minutes}": [-10080 + offset for offset in target_offsets],
        }
        for column, offsets in groups.items():
            values = [
                _lookup_history(hex_ids, forecast_start + pd.Timedelta(minutes=offset), retained_history)
                for offset in offsets
            ]
            matrix = np.column_stack(values)
            valid = np.isfinite(matrix).all(axis=1)
            total = np.full(len(frame), np.nan, dtype=np.float32)
            total[valid] = matrix[valid].sum(axis=1, dtype=np.float32)
            frame[column] = total

    if schema == FEATURE_SCHEMA:
        neighbor_ids = artifact.get("neighbor_hex_ids")
        if not neighbor_ids:
            raise ValueError("Artifact is missing the training neighbor universe")
        source = retained_frame.loc[retained_frame["hex_id_7"].isin(neighbor_ids)].copy()
        source["bucket_start"] = source["period_datetime_utc"].dt.tz_convert(local_timezone)
        source["demand_h10"] = source["total_demand"].astype(np.float32)
        extra = fresh_neighbor_features(source, frame)
        frame[extra.columns] = extra
    feature_columns = list(artifact.get("feature_columns", FEATURE_COLUMNS))
    if schema == FEATURE_SCHEMA and feature_columns != contract_columns(horizon_minutes, FEATURE_COLUMNS):
        raise ValueError("Artifact feature order/count does not match final.ipynb")
    if schema == FEATURE_SCHEMA and horizon_minutes > 10:
        frame[feature_columns] = frame[feature_columns].astype(np.float32)
    missing = sorted(set(feature_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Online feature builder is missing model columns: {missing}")
    return frame, feature_columns


def predict_and_store(
    forecast_start_utc=None,
    settings: Settings | None = None,
    model_names: tuple[str, ...] | None = None,
    horizons: tuple[int, ...] | None = None,
) -> PredictionSummary:
    """Load active model files, predict all H3 cells and persist the batch."""

    from .database import (
        ensure_runtime_tables,
        latest_observation_timestamp,
        load_demand_history,
        upsert_predictions,
    )

    settings = settings or get_settings()
    ensure_runtime_tables(settings)
    latest = latest_observation_timestamp(settings)
    if latest is None:
        raise RuntimeError("No demand observations exist; run bootstrap or live ingestion first")

    forecast_start = (
        normalize_forecast_start(forecast_start_utc)
        if forecast_start_utc is not None
        else latest + pd.Timedelta(minutes=10)
    )
    if forecast_start <= latest:
        raise ValueError(
            f"Forecast start {forecast_start.isoformat()} must be after latest observation "
            f"{latest.isoformat()}"
        )

    history_start = forecast_start - pd.Timedelta(days=settings.prediction_history_days)
    raw_history = load_demand_history(history_start, forecast_start, settings=settings)
    model_names = model_names or settings.model_names
    if tuple(model_names) != ("lightgbm",):
        raise ValueError("Only the notebook LightGBM models are available")
    horizons = horizons or settings.horizons
    active_path = settings.model_dir / "active.json"
    if not active_path.is_file():
        raise FileNotFoundError("No approved LightGBM run. Run the training pipeline first.")
    manifest = json.loads(active_path.read_text(encoding="utf-8"))
    if (manifest.get("feature_schema") != FEATURE_SCHEMA or
            manifest.get("evaluation_schema") != EVALUATION_SCHEMA):
        raise ValueError("Active models use the old 21/24-feature contract; run bootstrap to retrain")
    if manifest.get("model_name") != "lightgbm":
        raise ValueError("Active model manifest does not contain LightGBM")
    from .incremental_data import LIVE_GENERATOR_VERSION

    if manifest.get("live_generator_version") != LIVE_GENERATOR_VERSION:
        raise RuntimeError(
            "Active models were trained on an older fake live generator; "
            "wait for bootstrap retraining before serving predictions"
        )
    outputs: list[pd.DataFrame] = []
    loaded_models = 0
    missing_paths: list[str] = []

    for mode in ("Car", "Motorcycle"):
        for horizon in horizons:
            relative = manifest.get("model_files", {}).get(mode, {}).get(str(horizon))
            if not relative:
                missing_paths.append(f"{mode} H{horizon} in active LightGBM run")
                continue
            model_path = (settings.model_dir / relative).resolve()
            if not model_path.is_relative_to(settings.model_dir.resolve()) or not model_path.is_file():
                missing_paths.append(str(model_path))
                continue
            artifact = joblib.load(model_path)
            if (artifact.get("feature_schema") != FEATURE_SCHEMA
                    or artifact.get("model_name") != "lightgbm"
                    or artifact.get("strategy") != "direct"
                    or str(artifact.get("travel_mode")) != mode
                    or int(artifact.get("horizon", -1)) != int(horizon)):
                raise ValueError(f"LightGBM model metadata mismatch in {model_path}")
            frame, feature_columns = build_online_features(
                raw_history, forecast_start, artifact, horizon, settings.local_timezone,
            )
            from .feature_monitor import measure, persist
            persist(settings, measure(frame, feature_columns, raw_history, forecast_start,
                                      mode, horizon, _model_version(model_path)))
            # The notebook observes only a subset of H3 cells on some days.
            # LightGBM handles missing history for dormant/newly active cells.
            # All-fresh-history-missing is an ingest failure, not a silent fallback.
            if frame["lag_10m"].isna().all() or frame["lag_20m"].isna().all():
                raise ValueError(f"No closed T-10/T-20 demand history for {mode}; backfill before serving")
            if frame["lag_30m"].isna().all():
                raise ValueError(f"No closed T-30m demand history for {mode}; backfill before serving")
            model_input = frame[feature_columns]
            if artifact.get("input_format") == "numeric_float32":
                model_input = model_input.to_numpy(dtype=np.float32, copy=False)
            prediction = np.rint(
                np.clip(artifact["model"].predict(model_input), 0, None)
            ).astype(np.float64)
            if not np.isfinite(prediction).all():
                raise ValueError("Nonfinite model output")
            outputs.append(pd.DataFrame({
                "forecast_start_utc": forecast_start,
                "hex_id_7": frame["hex_id_7"].astype(str),
                "travel_mode": mode,
                "horizon_minutes": int(horizon),
                "model_name": "lightgbm",
                "model_version": _model_version(model_path),
                "predicted_demand": prediction,
            }))
            loaded_models += 1

    if missing_paths:
        missing = "\n".join(f"- {path}" for path in missing_paths)
        raise FileNotFoundError(
            "Missing trained model files. Run the bootstrap/train command first:\n" + missing
        )
    if not outputs:
        raise RuntimeError("No predictions were produced")

    predictions = pd.concat(outputs, ignore_index=True)
    stored = upsert_predictions(predictions, settings=settings)
    return PredictionSummary(
        forecast_start_utc=forecast_start.isoformat(),
        rows=len(predictions),
        models=loaded_models,
        stored_rows=stored,
    )
