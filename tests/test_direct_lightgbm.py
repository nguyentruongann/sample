"""Exercise the actual LightGBM direct train-to-serve path on small, continuous data."""

import json

import joblib
import lightgbm
import numpy as np
import pandas as pd

import demand_forecasting.database as database
import demand_forecasting.train as training
from demand_forecasting.config import Settings
from demand_forecasting.feature_engineering import build_features, prepare_horizon_frame
from demand_forecasting.predict import build_online_features, predict_and_store


def raw_history():
    local = pd.date_range("2026-08-01", periods=17 * 144, freq="10min", tz="Asia/Ho_Chi_Minh")
    slots = np.arange(len(local))
    parts = []
    for mode, base in (("Car", 5), ("Motorcycle", 9)):
        demand = (base + (slots % 144 < 65).astype(int) * 3 + (slots % 144 > 100).astype(int) * 2
                  + slots % 7).astype(np.int16)
        parts.append(pd.DataFrame({
            "period_datetime_utc": local.tz_convert("UTC"),
            "hex_id_7": "8765b56e9ffffff",
            "travel_mode": mode,
            "total_demand": demand,
            "is_holiday": np.zeros(len(local), dtype=np.int8),
        }))
    return pd.concat(parts, ignore_index=True)


def small_settings(tmp_path):
    return Settings(
        database_url="", postgres_schema="public", postgres_table="fake_demand_10min",
        predictions_table="demand_predictions", seed=42, local_timezone="Asia/Ho_Chi_Minh",
        train_ratio=0.80, max_train_rows=900, model_names=("lightgbm",),
        horizons=(10, 30, 60), validation_days=2, rolling_test_days=2,
        train_history_days=10, search_train_rows=200, search_val_rows=100,
        search_estimators=12,
        test_start="2026-08-13T00:00:00+07:00", data_dir=tmp_path / "data",
        model_dir=tmp_path / "models", output_dir=tmp_path / "outputs",
        prediction_history_days=8,
    )


def test_direct_horizons_train_and_serve_without_h10_as_a_feature(tmp_path, monkeypatch):
    raw = raw_history()
    feature_frame = build_features(raw)
    settings = small_settings(tmp_path)
    monkeypatch.setattr(training, "TREE_CANDIDATES", (7, 15))
    original = training.lightgbm_params
    monkeypatch.setattr(training, "lightgbm_params", lambda h, seed, leaves, rounds, n_jobs: {
        **original(h, seed, leaves, rounds, n_jobs), "n_jobs": 1, "min_child_samples": 5,
    })
    fit_calls = []
    original_fit = lightgbm.LGBMRegressor.fit

    def count_fit(model, x, y, **kwargs):
        fit_calls.append((model.get_params()["num_leaves"], model.get_params()["n_estimators"]))
        if isinstance(x, pd.DataFrame):
            assert kwargs["categorical_feature"] == ["hex_code"]
        else:
            assert isinstance(x, np.ndarray) and x.dtype == np.float32
            assert "categorical_feature" not in kwargs
        return original_fit(model, x, y, **kwargs)

    monkeypatch.setattr(lightgbm.LGBMRegressor, "fit", count_fit)

    summary = training.train_all(feature_frame, settings=settings, rolling=True)
    assert len(summary) == 6
    assert len(fit_calls) == 18
    assert {leaves for leaves, _ in fit_calls} == {7, 15}
    assert (summary["fit_count"] == 3).all()
    assert summary["promoted"].all()
    assert set(summary["horizon"]) == {10, 30, 60}
    assert summary["test_start"].nunique() == 1
    assert summary["val_start"].nunique() == 1
    assert summary["best_iteration"].ge(1).all()
    assert np.isfinite(summary["VAL_WMAPE_%"]).all()
    assert (summary["validation_status"] == "searched_before_refit").all()
    assert (summary["parameter_source"] == "validation_tree_search").all()
    assert summary["WMAPE_%"].notna().all()
    assert (summary["model"] == "lightgbm").all()
    trials = pd.read_csv(settings.output_dir / "model_search_trials.csv")
    assert len(trials) == 12
    assert trials["selected"].sum() == 6
    assert np.allclose(
        summary.sort_values(["travel_mode", "horizon"])["VAL_WMAPE_%"].to_numpy(),
        trials.groupby(["travel_mode", "horizon"])["VAL_WMAPE_%"].min().to_numpy(),
    )
    active = json.loads((settings.model_dir / "active.json").read_text())
    from demand_forecasting.incremental_data import LIVE_GENERATOR_VERSION
    assert active["live_generator_version"] == LIVE_GENERATOR_VERSION
    for mode in ("Car", "Motorcycle"):
        for horizon in (10, 30, 60):
            artifact = joblib.load(settings.model_dir / active["model_files"][mode][str(horizon)])
            assert artifact["strategy"] == "direct"
            assert "pred_h10" not in artifact["feature_columns"]
            assert artifact["model_params"]["device_type"] == "cpu"
            assert artifact["model_params"]["num_leaves"] in {7, 15}
            assert "max_depth" not in artifact["model_params"]
            assert 1 <= artifact["model_params"]["n_estimators"] <= 12
            assert artifact["parameter_source"] == "validation_tree_search"
            assert artifact["input_format"] == ("categorical_dataframe" if horizon == 10 else "numeric_float32")

    # The exact same historical offsets must be produced offline and online.
    t = pd.Timestamp("2026-08-14T12:00:00+07:00")
    mode_frame = feature_frame[feature_frame["travel_mode"] == "Car"].copy()
    offline, feature_columns, _ = prepare_horizon_frame(mode_frame, 60)
    expected = offline.loc[offline["bucket_start"] == t, feature_columns].iloc[0]
    artifact = joblib.load(settings.model_dir / active["model_files"]["Car"]["60"])
    online, cols = build_online_features(
        raw.loc[raw["period_datetime_utc"] < t.tz_convert("UTC")],
        t.tz_convert("UTC"), artifact, 60,
    )
    assert cols == feature_columns
    np.testing.assert_allclose(online[cols].iloc[0].to_numpy(dtype=float),
                               expected.to_numpy(dtype=float), rtol=1e-5, atol=1e-5,
                               equal_nan=True)

    stored = []
    monkeypatch.setattr(database, "ensure_runtime_tables", lambda _settings: None)
    monkeypatch.setattr(database, "latest_observation_timestamp", lambda _settings: t.tz_convert("UTC") - pd.Timedelta(minutes=10))
    monkeypatch.setattr(database, "load_demand_history", lambda start, end, settings: raw.loc[
        (raw["period_datetime_utc"] >= start) & (raw["period_datetime_utc"] < end)
    ].copy())
    monkeypatch.setattr(database, "upsert_predictions", lambda frame, settings: stored.append(frame.copy()) or len(frame))
    output = predict_and_store(t.tz_convert("UTC"), settings=settings)
    assert output.models == 6
    assert output.stored_rows == 6
    assert set(stored[0]["horizon_minutes"]) == {10, 30, 60}
    assert set(stored[0]["model_name"]) == {"lightgbm"}


def test_training_rejects_a_second_model(tmp_path):
    settings = small_settings(tmp_path)
    try:
        training.train_all(pd.DataFrame(), settings=settings, model_names=("lightgbm", "xgboost"))
    except ValueError as error:
        assert "only" in str(error).lower()
    else:
        raise AssertionError("Other models must be rejected")


def test_weekly_retrain_searches_validation_then_refits_six_models(tmp_path, monkeypatch):
    settings = small_settings(tmp_path)
    features = build_features(raw_history())
    original_params = training.lightgbm_params
    original_fit = training._fit
    fit_calls = []

    monkeypatch.setattr(training, "TREE_CANDIDATES", (7, 15))
    monkeypatch.setattr(training, "lightgbm_params", lambda h, seed, leaves, rounds, n_jobs: {
        **original_params(h, seed, leaves, rounds, n_jobs), "n_jobs": 1, "min_child_samples": 5,
    })

    def count_fit(model, x, y, horizon):
        fit_calls.append((model.get_params()["num_leaves"], model.get_params()["n_estimators"]))
        return original_fit(model, x, y, horizon)

    monkeypatch.setattr(training, "_fit", count_fit)
    summary = training.train_all(features, settings=settings, rolling=True)

    assert len(fit_calls) == 6
    assert all(leaves in {7, 15} and 1 <= trees <= 12 for leaves, trees in fit_calls)
    assert len(summary) == 6
    assert summary["promoted"].all()
    assert summary["test_rows"].gt(0).all()
    assert np.isfinite(summary["WMAPE_%"]).all()
    assert np.isfinite(summary["VAL_WMAPE_%"]).all()
    assert (summary["fit_count"] == 3).all()
    assert (summary["mode"] == "rolling").all()
    assert (summary["parameter_source"] == "validation_tree_search").all()
    assert (settings.model_dir / "active.json").is_file()
