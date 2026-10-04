"""Direct CPU LightGBM with validation search and held-out test WMAPE."""

from __future__ import annotations

import gc
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import Settings, get_settings
from .data_split import resolve_boundaries, time_based_split
from .evaluate import calculate_metrics, wmape
from .fake_data import LOCAL_END
from .feature_engineering import prepare_horizon_frame
from .notebook_contract import (FEATURE_SCHEMA, EVALUATION_SCHEMA, FEATURE_POLICY,
    EXPECTED_FEATURE_COUNT, LEAF_TRIAL_ORDER, LOCKED_PARAMS)

# The notebook searches these leaf counts independently for each mode/horizon.
# It does not cap max_depth. Rounds are learned with validation early stopping.
TREE_CANDIDATES = (31, 63, 130)


def lightgbm_params(horizon: int, seed: int, leaves: int, rounds: int, n_jobs: int = 4) -> dict:
    """CPU Poisson LightGBM; tree shape and rounds are selected on validation."""
    if horizon not in (10, 30, 60):
        raise ValueError(f"Unsupported horizon H{horizon}")
    return dict(
        objective="poisson",
        metric="l1",
        device_type="cpu",
        max_bin=255,
        n_estimators=rounds,
        learning_rate=0.05 if horizon == 10 else 0.06,
        num_leaves=leaves,
        min_child_samples=100,
        subsample=0.90,
        subsample_freq=1,
        colsample_bytree=0.90,
        reg_lambda=2.0,
        verbosity=-1,
        n_jobs=n_jobs,
        random_state=seed,
    )


def _model_input(frame: pd.DataFrame, features: list[str], horizon: int):
    x = frame[features]
    # H10 in the notebook treats hex_code as categorical. H30/H60 train on
    # float32 NumPy arrays, where hex_code is an ordinary numeric feature.
    return x if horizon == 10 else x.to_numpy(dtype=np.float32, copy=False)


def _fit(model, x, y: pd.Series, horizon: int):
    if horizon == 10:
        model.fit(x, y, categorical_feature=["hex_code"])
    else:
        model.fit(x, y)
    return model


def _atomic_joblib(value: dict, path: Path) -> None:
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        joblib.dump(value, temp, compress=3)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _atomic_json(value: dict, path: Path) -> None:
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _train_one(
    mode_frame: pd.DataFrame,
    mode: str,
    horizon: int,
    boundaries: tuple[pd.Timestamp, pd.Timestamp],
    run_dir: Path,
    settings: Settings,
    *,
    rolling: bool,
    training_origin: str,
) -> tuple[dict, str, list[dict]]:
    """Use locked/search parameters, refit TRAIN+VAL, score integer TEST predictions."""
    from lightgbm import LGBMRegressor, early_stopping

    print(f"{mode} H{horizon}: preparing target and features", flush=True)
    frame, features, target = prepare_horizon_frame(mode_frame, horizon)
    split = time_based_split(
        frame, horizon, settings.train_ratio, settings.max_train_rows, settings.seed,
        validation_days=settings.validation_days, rolling=rolling,
        rolling_test_days=settings.rolling_test_days,
        train_history_days=settings.train_history_days, boundaries=boundaries,
        historical_end=None if rolling else LOCAL_END,
    )
    print(
        f"{mode} H{horizon}: train={len(split.train_index):,} "
        f"val={len(split.val_index):,} test={len(split.test_index):,}; "
        f"strategy={settings.training_strategy}; full train/validation",
        flush=True,
    )
    assert len(features) == EXPECTED_FEATURE_COUNT[horizon]
    started = time.perf_counter()
    strategy = settings.training_strategy
    parameter_source = "final_notebook_locked" if strategy == "locked" else "validation_tree_search"
    validation_status = "locked_params_evaluated_before_refit" if strategy == "locked" else "searched_before_refit"
    # Full TRAIN and full VAL, exactly as final.ipynb; no search sampling.
    search_train, search_val = split.train_index, split.val_index
    x_train = _model_input(frame.iloc[search_train], features, horizon)
    y_train = frame.iloc[search_train][target].to_numpy(dtype=np.float32)
    x_val = _model_input(frame.iloc[search_val], features, horizon)
    val_actual = frame.iloc[search_val][target].to_numpy(dtype=np.float64)
    trials: list[dict] = []
    best_prediction = None
    best_score = float("inf")
    if strategy == "locked":
        configs = [LOCKED_PARAMS[(mode, horizon)]]
    else:
        # Stable first-min tie rule in notebook trial order (H10 starts at 130).
        order = [n for n in LEAF_TRIAL_ORDER[horizon] if n in TREE_CANDIDATES]
        order += [n for n in TREE_CANDIDATES if n not in order]
        configs = [(leaves, settings.search_estimators) for leaves in order]
    for leaves, budget in configs:
        params = lightgbm_params(horizon, settings.seed, leaves, budget, settings.train_n_jobs)
        trial_start = time.perf_counter()
        candidate = LGBMRegressor(**params)
        fit_options = {}
        if strategy == "notebook_search":
            fit_options.update(eval_set=[(x_val, val_actual)],
                callbacks=[early_stopping(stopping_rounds=50, verbose=False)])
        if horizon == 10:
            fit_options["categorical_feature"] = ["hex_code"]
        candidate.fit(x_train, y_train, **fit_options)
        rounds = int(candidate.best_iteration_ or budget) if strategy == "notebook_search" else budget
        val_prediction = np.clip(candidate.predict(x_val, num_iteration=rounds), 0, None)
        if not np.isfinite(val_prediction).all():
            raise ValueError("Nonfinite validation prediction")
        score = wmape(val_actual, val_prediction)
        if not np.isfinite(score):
            raise ValueError(f"{mode} H{horizon}: validation WMAPE is unavailable")
        midpoint = split.val_start + (split.test_start-split.val_start)/2
        first = (frame.iloc[search_val]["bucket_start"] < midpoint).to_numpy()
        trials.append({
            "travel_mode": mode, "horizon": horizon, "num_leaves": leaves,
            "max_depth": -1, "best_iteration": rounds, "VAL_WMAPE_%": score,
            "VAL_WMAPE_integer_%": wmape(val_actual, np.rint(val_prediction)),
            "first_val_WMAPE_integer_%": wmape(val_actual[first], np.rint(val_prediction[first])),
            "second_val_WMAPE_integer_%": wmape(val_actual[~first], np.rint(val_prediction[~first])),
            "search_train_rows": len(search_train), "search_val_rows": len(search_val),
            "full_val_rows": len(split.val_index), "parameter_source": parameter_source,
            "feature_count": len(features), "hit_tree_limit": strategy == "notebook_search" and rounds == budget,
            "search_seconds": time.perf_counter() - trial_start,
        })
        if score < best_score:
            best_score, best_prediction = score, np.rint(val_prediction).astype(np.int64)
        print(f"{mode} H{horizon} {strategy} | leaves={leaves} trees={rounds} "
              f"VAL float WMAPE={score:.4f}%", flush=True)
        del candidate, val_prediction
        gc.collect()
    del x_train, y_train, x_val
    best = min(trials, key=lambda item: item["VAL_WMAPE_%"])
    for trial in trials:
        trial["selected"] = trial is best
    params = lightgbm_params(
        horizon, settings.seed, best["num_leaves"], best["best_iteration"],
        settings.train_n_jobs,
    )
    model = _fit(
        LGBMRegressor(**params),
        _model_input(frame.iloc[split.train_val_index], features, horizon),
        frame.iloc[split.train_val_index][target].to_numpy(dtype=np.float32),
        horizon,
    )
    fit_seconds = time.perf_counter() - started
    test_float = np.clip(model.predict(
        _model_input(frame.iloc[split.test_index], features, horizon)), 0, None)
    if not np.isfinite(test_float).all():
        raise ValueError("Nonfinite test prediction")
    test_prediction = np.rint(test_float).astype(np.int64)
    actual = frame.iloc[split.test_index][target].to_numpy(dtype=np.float64)
    test_metrics = calculate_metrics(actual, test_prediction)
    if not np.isfinite(test_metrics["WMAPE_%"]):
        raise ValueError(f"{mode} H{horizon}: test WMAPE is unavailable; active models unchanged")
    rounded_test_wmape = wmape(actual, np.rint(test_prediction))
    test_times = frame.iloc[split.test_index]["bucket_start"]
    notebook_test = (test_times + pd.Timedelta(minutes=horizon) <= LOCAL_END).to_numpy()
    live_test = (test_times >= LOCAL_END).to_numpy()
    notebook_wmape = wmape(actual[notebook_test], test_prediction[notebook_test])
    live_wmape = wmape(actual[live_test], test_prediction[live_test])
    stem = f"{mode.lower()}_h{horizon}_lightgbm"
    model_path = run_dir / f"{stem}.joblib"
    _atomic_joblib({
        "model": model, "feature_columns": features, "target_column": target,
        "travel_mode": mode, "horizon": horizon, "model_name": "lightgbm",
        "strategy": "direct", "model_params": params,
        "parameter_source": parameter_source, "search_trials": trials,
        "validation_status": validation_status,
        "train_start": split.train_start.isoformat(),
        "val_start": split.val_start.isoformat(),
        "test_start": split.test_start.isoformat(),
        "test_end": split.test_end.isoformat(),
        "training_origin": training_origin,
        "hex_categories": sorted(frame["hex_id_7"].astype(str).unique()),
        "hex_code_mapping": {str(hx): int(code) for hx, code in
            frame[["hex_id_7", "hex_code"]].drop_duplicates().itertuples(index=False, name=None)},
        "neighbor_hex_ids": sorted(mode_frame["hex_id_7"].astype(str).unique()),
        "local_timezone": settings.local_timezone,
        "feature_schema": FEATURE_SCHEMA, "feature_count": len(features),
        "variant": FEATURE_POLICY[horizon], "postprocessing": "clip_rint_only",
        "fit_scope": "train_plus_validation_purged",
        "availability_requirement": "T-10/T-20 closed and ingested at forecast time T",
        "target_semantics": "sum demand over [T,T+H)",
        "input_format": "categorical_dataframe" if horizon == 10 else "numeric_float32",
    }, model_path)

    baseline_column = "lag_30m" if horizon == 10 else f"naive_recent_h{horizon}"
    baseline_val = wmape(frame.iloc[split.val_index][target], frame.iloc[split.val_index][baseline_column])
    baseline_test = wmape(frame.iloc[split.test_index][target], frame.iloc[split.test_index][baseline_column])
    prediction_dir = settings.output_dir / "weekly" if rolling else settings.output_dir
    prediction_dir.mkdir(parents=True, exist_ok=True)
    path = prediction_dir / f"predictions_{stem}.parquet"
    predictions = frame.iloc[split.test_index][["bucket_start", "hex_id_7", "travel_mode"]].copy()
    predictions["horizon"] = f"H{horizon}"
    predictions["model"] = "lightgbm"
    predictions["actual"] = actual.astype(np.float32)
    predictions["prediction"] = test_prediction
    predictions["prediction_float"] = test_float
    # Save validation predictions separately: model trained only on TRAIN.
    val_rows = frame.iloc[split.val_index][["bucket_start", "hex_id_7", "travel_mode"]].copy()
    val_rows["actual"], val_rows["prediction"] = val_actual, best_prediction
    val_rows.to_parquet(run_dir / f"validation_{stem}.parquet", index=False)
    predictions.to_parquet(path, index=False)
    row = {
        "travel_mode": mode, "horizon": horizon, "model": "lightgbm",
        "mode": "rolling" if rolling else "backtest",
        "train_start": split.train_start.isoformat(),
        "val_start": split.val_start.isoformat(),
        "test_start": split.test_start.isoformat(),
        "test_end": split.test_end.isoformat(),
        "train_rows": len(split.train_index), "val_rows": len(split.val_index),
        "fit_rows": len(split.train_val_index), "test_rows": len(split.test_index),
        "purged_rows": len(split.purged_index),
        "num_leaves": params["num_leaves"], "max_depth": -1,
        "best_iteration": params["n_estimators"],
        "parameter_source": parameter_source,
        "validation_status": validation_status,
        "VAL_WMAPE_%": best["VAL_WMAPE_%"],
        "VAL_baseline_WMAPE_%": baseline_val,
        "WMAPE_%": test_metrics["WMAPE_%"], "baseline_WMAPE_%": baseline_test,
        "WMAPE_rounded_%": rounded_test_wmape,
        "WMAPE_float_%": wmape(actual, test_float),
        "feature_count": len(features), "variant": FEATURE_POLICY[horizon],
        "VAL_WMAPE_integer_%": best["VAL_WMAPE_integer_%"],
        "notebook_test_rows": int(notebook_test.sum()),
        "notebook_test_WMAPE_%": notebook_wmape,
        "live_test_rows": int(live_test.sum()),
        "live_test_WMAPE_%": live_wmape,
        "fit_seconds": fit_seconds, "fit_count": len(trials) + 1,
        "model_path": str(model_path), "predictions_path": str(path),
        **{k: v for k, v in test_metrics.items() if k != "WMAPE_%"},
    }
    del model, frame
    gc.collect()
    return row, model_path.name, trials


def train_all(
    feature_frame: pd.DataFrame,
    settings: Settings | None = None,
    model_names: tuple[str, ...] | None = None,
    horizons: tuple[int, ...] | None = None,
    *,
    rolling: bool = False,
) -> pd.DataFrame:
    settings = settings or get_settings()
    settings.ensure_directories()
    if tuple(model_names or settings.model_names) != ("lightgbm",):
        raise ValueError("Only notebook LightGBM is supported")
    if getattr(settings, "max_train_rows", None) is not None:
        raise ValueError("MAX_TRAIN_ROWS must be empty: final.ipynb trains on all rows")
    if getattr(settings, "training_strategy", "locked") not in ("locked", "notebook_search"):
        raise ValueError("TRAINING_STRATEGY must be locked or notebook_search")
    if not getattr(settings, "fresh_buckets_closed_at_t", True):
        raise ValueError("The chosen feature set requires T-10/T-20 closed and ingested at T")
    selected_horizons = tuple(dict.fromkeys(horizons or settings.horizons))
    if not selected_horizons or set(selected_horizons) - {10, 30, 60}:
        raise ValueError("Horizons must be selected from 10,30,60")
    boundaries = resolve_boundaries(
        feature_frame, train_ratio=settings.train_ratio,
        validation_days=settings.validation_days, test_start=settings.test_start,
        rolling=rolling, rolling_test_days=settings.rolling_test_days,
    )
    training_origin = feature_frame.attrs.get(
        "training_origin", pd.Timestamp(feature_frame["bucket_start"].min()).isoformat()
    )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    run_dir = settings.model_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    search_trials = []
    model_files: dict[str, dict[str, str]] = {}
    for mode in ("Car", "Motorcycle"):
        source = feature_frame.loc[feature_frame["travel_mode"] == mode].copy().reset_index(drop=True)
        if source.empty:
            raise ValueError(f"Missing {mode} history")
        model_files[mode] = {}
        for horizon in selected_horizons:
            row, filename, trials = _train_one(
                source, mode, horizon, boundaries, run_dir, settings,
                rolling=rolling, training_origin=training_origin,
            )
            rows.append(row)
            search_trials.extend(trials)
            model_files[mode][str(horizon)] = str(Path("runs") / run_id / filename)
            print(
                f"{mode} H{horizon} LightGBM | leaves={row['num_leaves']} "
                f"trees={row['best_iteration']} | VAL WMAPE={row['VAL_WMAPE_%']:.2f}% "
                f"| TEST WMAPE={row['WMAPE_%']:.2f}% "
                f"| TEST WMAPE rounded={row['WMAPE_rounded_%']:.2f}% "
                f"(notebook={row['notebook_test_WMAPE_%']:.2f}%, "
                f"live={row['live_test_WMAPE_%']:.2f}%) "
                f"| fit={row['fit_seconds']:.1f}s",
                flush=True,
            )
        del source
        gc.collect()

    summary = pd.DataFrame(rows)
    complete_set = set(selected_horizons) == {10, 30, 60}
    alert_path = settings.output_dir / "training_alerts.json"
    quality_passed = bool(summary["WMAPE_%"].le(30.0).all())
    # Historical notebook reproduction is a separate report. Only the
    # rolling run is eligible for live serving.
    promoted = rolling and complete_set and quality_passed
    if rolling and not promoted:
        _atomic_json({
            "run_id": run_id,
            "reason": ("notebook_benchmark_only" if not rolling else
                       "incomplete_horizon_set" if not complete_set else
                       "test_wmape_above_30_pct"),
            "requested_horizons": list(selected_horizons),
            "test_wmape_pct": dict(zip(
                summary["travel_mode"] + "_H" + summary["horizon"].astype(str),
                summary["WMAPE_%"].astype(float),
            )),
        }, alert_path)
        print(f"WARNING: run not promoted; see {alert_path}")
    elif promoted:
        alert_path.unlink(missing_ok=True)
    if complete_set:
        if len(summary) != 6 or not np.isfinite(summary[["VAL_WMAPE_%", "WMAPE_%"]].to_numpy()).all():
            raise RuntimeError("Missing validation/test WMAPE; previous active model set retained")
    summary["promoted"] = promoted
    summary_filename = "weekly_retrain_summary.csv" if rolling else "notebook_benchmark_summary.csv"
    summary.to_csv(settings.output_dir / summary_filename, index=False)
    summary.to_csv(run_dir / "metrics.csv", index=False)
    pd.DataFrame(search_trials).to_csv(run_dir / "model_search_trials.csv", index=False)
    shared_trials = "model_search_trials.csv" if rolling else "notebook_benchmark_search_trials.csv"
    pd.DataFrame(search_trials).to_csv(settings.output_dir / shared_trials, index=False)
    # Always keep a complete candidate manifest, including historical benchmark runs.
    _atomic_json({"run_id": run_id, "model_name": "lightgbm", "model_files": model_files,
        "feature_schema": FEATURE_SCHEMA, "evaluation_schema": EVALUATION_SCHEMA,
        "training_strategy": getattr(settings, "training_strategy", "locked"),
        "rolling": rolling, "promoted": promoted, "postprocessing": "clip_rint_only"},
        run_dir / "manifest.json")
    if promoted:
        # Publish the whole set together; online requests never see a mixed run.
        from .incremental_data import LIVE_GENERATOR_VERSION

        _atomic_json({
            "run_id": run_id,
            "model_name": "lightgbm",
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "rolling": rolling,
            "evaluation_schema": EVALUATION_SCHEMA,
            "feature_schema": FEATURE_SCHEMA,
            "training_strategy": getattr(settings, "training_strategy", "locked"),
            "postprocessing": "clip_rint_only",
            "deployment_scope": "synthetic_pipeline_demo_not_production_approval",
            "live_generator_version": LIVE_GENERATOR_VERSION,
            "model_files": model_files,
        }, settings.model_dir / "active.json")
    return summary
