"""A live-data regression must never silently replace the serving model set."""

import json
from types import SimpleNamespace

import pandas as pd

import demand_forecasting.train as training


def test_43_pct_test_wmape_keeps_previous_active_models(tmp_path, monkeypatch):
    model_dir = tmp_path / "models"
    output_dir = tmp_path / "outputs"
    model_dir.mkdir()
    output_dir.mkdir()
    active = model_dir / "active.json"
    active.write_text('{"run_id":"previous"}', encoding="utf-8")
    settings = SimpleNamespace(
        model_dir=model_dir, output_dir=output_dir,
        model_names=("lightgbm",), horizons=(10, 30, 60),
        train_ratio=.8, validation_days=30, test_start=None, rolling_test_days=30,
        ensure_directories=lambda: None,
    )
    features = pd.DataFrame({
        "travel_mode": ["Car", "Motorcycle"],
        "bucket_start": pd.to_datetime(["2026-09-10", "2026-09-10"], utc=True),
    })
    monkeypatch.setattr(training, "resolve_boundaries", lambda *_args, **_kwargs: (
        pd.Timestamp("2026-07-25T00:00:00Z"), pd.Timestamp("2026-08-25T00:00:00Z")
    ))

    def fake_train_one(source, mode, horizon, *_args, **_kwargs):
        return {
            "travel_mode": mode, "horizon": horizon, "VAL_WMAPE_%": 15.0,
            "WMAPE_%": 43.0, "WMAPE_rounded_%": 43.0, "num_leaves": 31, "best_iteration": 100,
            "fit_seconds": .1, "notebook_test_WMAPE_%": 15.0,
            "live_test_WMAPE_%": 70.0,
        }, f"{mode}_h{horizon}.joblib", [{"mode": mode, "horizon": horizon}]

    monkeypatch.setattr(training, "_train_one", fake_train_one)
    summary = training.train_all(features, settings=settings, rolling=True)
    assert not summary["promoted"].any()
    assert json.loads(active.read_text(encoding="utf-8")) == {"run_id": "previous"}
    alert = json.loads((output_dir / "training_alerts.json").read_text(encoding="utf-8"))
    assert alert["reason"] == "test_wmape_above_30_pct"


def test_notebook_benchmark_never_replaces_live_models(tmp_path, monkeypatch):
    model_dir = tmp_path / "models"
    output_dir = tmp_path / "outputs"
    model_dir.mkdir()
    output_dir.mkdir()
    active = model_dir / "active.json"
    active.write_text('{"run_id":"serving"}', encoding="utf-8")
    settings = SimpleNamespace(
        model_dir=model_dir, output_dir=output_dir,
        model_names=("lightgbm",), horizons=(10, 30, 60),
        train_ratio=.8, validation_days=30, test_start="2026-08-15T09:40:00+07:00",
        rolling_test_days=30, ensure_directories=lambda: None,
    )
    features = pd.DataFrame({
        "travel_mode": ["Car", "Motorcycle"],
        "bucket_start": pd.to_datetime(["2026-09-10", "2026-09-10"], utc=True),
    })
    monkeypatch.setattr(training, "resolve_boundaries", lambda *_args, **_kwargs: (
        pd.Timestamp("2026-07-16T09:40:00+07:00"),
        pd.Timestamp("2026-08-15T09:40:00+07:00"),
    ))

    def fake_train_one(source, mode, horizon, *_args, **_kwargs):
        return {
            "travel_mode": mode, "horizon": horizon, "VAL_WMAPE_%": 14.0,
            "WMAPE_%": 15.0, "WMAPE_rounded_%": 15.0, "num_leaves": 63, "best_iteration": 400,
            "fit_seconds": .1, "notebook_test_WMAPE_%": 15.0,
            "live_test_WMAPE_%": float("nan"),
        }, f"{mode}_h{horizon}.joblib", [{"mode": mode, "horizon": horizon}]

    monkeypatch.setattr(training, "_train_one", fake_train_one)
    summary = training.train_all(features, settings=settings, rolling=False)
    assert not summary["promoted"].any()
    assert json.loads(active.read_text(encoding="utf-8")) == {"run_id": "serving"}
    assert (output_dir / "notebook_benchmark_summary.csv").is_file()
    assert not (output_dir / "training_alerts.json").exists()
