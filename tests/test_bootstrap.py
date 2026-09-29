from argparse import Namespace
from types import SimpleNamespace

import pandas as pd

import demand_forecasting.database as db
import demand_forecasting.incremental_data as incremental_data
import demand_forecasting.pipeline as pipeline
import demand_forecasting.predict as predict
import demand_forecasting.provision as provision


def test_init_runtime_provisions_before_creating_tables(monkeypatch):
    calls = []
    settings = SimpleNamespace(
        require_database_url=lambda: "postgresql://app:pass@db/demand_db",
        postgres_schema="public",
        postgres_table="fake_demand_10min",
        predictions_table="demand_predictions",
    )
    monkeypatch.setattr(pipeline, "get_settings", lambda: settings)
    monkeypatch.setenv("AIRFLOW_METADATA_URL", "postgresql+psycopg2://airflow:pass@db/airflow")
    monkeypatch.setenv("POSTGRES_ADMIN_URL", "postgresql://postgres:pass@db/postgres")
    monkeypatch.setattr(provision, "provision_external_databases", lambda *args: calls.append(("provision", args)))
    monkeypatch.setattr(db, "wait_for_database", lambda _settings: calls.append(("wait",)))
    monkeypatch.setattr(db, "ensure_runtime_tables", lambda _settings: calls.append(("tables",)))

    pipeline.command_init_runtime(Namespace())
    assert [event[0] for event in calls] == ["provision", "wait", "tables"]
    assert calls[0][1][2].endswith("/postgres")


def _setup(monkeypatch, tmp_path, has_rows):
    calls = []
    settings = SimpleNamespace(
        data_dir=tmp_path, model_dir=tmp_path, horizons=(10, 30, 60)
    )
    monkeypatch.setattr(pipeline, "get_settings", lambda: settings)
    monkeypatch.setattr(pipeline, "command_init_runtime", lambda _args: calls.append("init"))
    monkeypatch.setattr(db, "database_has_rows", lambda _settings: has_rows)
    monkeypatch.setattr(
        pipeline, "command_generate", lambda _args: calls.append("generate") or tmp_path / "raw.parquet"
    )
    monkeypatch.setattr(
        db, "copy_parquet_to_postgres", lambda *_args, **_kwargs: calls.append("load")
    )
    monkeypatch.setattr(
        pipeline, "command_train", lambda _args: calls.append("train") or pd.DataFrame({"promoted": [True]})
    )
    monkeypatch.setattr(pipeline, "command_evaluate", lambda _args: calls.append("evaluate"))
    monkeypatch.setattr(
        incremental_data, "backfill_live_history", lambda **_kwargs: calls.append("backfill") or {}
    )
    monkeypatch.setattr(
        predict, "predict_and_store", lambda **_kwargs: calls.append("predict") or SimpleNamespace(to_dict=lambda: {})
    )
    return calls


def test_first_start_auto_seeds_trains_backfills_and_predicts(monkeypatch, tmp_path):
    calls = _setup(monkeypatch, tmp_path, has_rows=False)
    pipeline.command_bootstrap(Namespace(batch_size=200_000))
    assert calls == ["init", "generate", "load", "backfill", "train", "evaluate", "backfill", "predict"]


def test_rerun_preserves_existing_rows_and_trained_models(monkeypatch, tmp_path):
    calls = _setup(monkeypatch, tmp_path, has_rows=True)
    monkeypatch.setattr(pipeline, "_active_model_files_exist", lambda _settings: True)
    pipeline.command_bootstrap(Namespace(batch_size=200_000))
    assert calls == ["init", "backfill", "backfill", "predict"]
