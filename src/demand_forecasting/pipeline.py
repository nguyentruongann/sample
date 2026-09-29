"""Command-line orchestration for generation, PostgreSQL, training and evaluation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from .config import get_settings


def _parse_csv_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip().lower() for item in value.split(",") if item.strip())


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def command_generate(args) -> Path:
    from .fake_data import generate_fake_data

    path, summary = generate_fake_data(output_dir=args.output_dir, overwrite=args.overwrite)
    print(summary.to_string(index=False))
    print(f"Parquet: {path}")
    return path


def command_load_postgres(args):
    from .database import copy_parquet_to_postgres, wait_for_database

    settings = get_settings()
    wait_for_database(settings)
    path = Path(args.parquet).resolve()
    summary = copy_parquet_to_postgres(path, settings=settings, batch_size=args.batch_size)
    print(
        f"Loaded {summary.loaded_rows:,} rows into "
        f"{summary.schema}.{summary.table} in {summary.elapsed_seconds:.1f}s"
    )
    return summary


def command_train(args) -> pd.DataFrame:
    from .jobs import train_from_postgres

    settings = get_settings()
    summary, feature_path, feature_rows = train_from_postgres(
        settings=settings,
        model_names=_parse_csv_strings(args.models),
        horizons=_parse_csv_ints(args.horizons),
        rolling=args.rolling,
    )
    print(f"Features: {feature_path} | rows={feature_rows:,}")
    print(summary.to_string(index=False))
    return summary


def _load_prediction_files(output_dir: Path) -> pd.DataFrame:
    paths = sorted(output_dir.glob("predictions_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No prediction files found in {output_dir}")
    return pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)


def command_evaluate(args):
    from .jobs import evaluate_saved_predictions

    settings = get_settings()
    paths = evaluate_saved_predictions(settings, rolling=getattr(args, "rolling", False))
    for name, path in paths.items():
        print(f"{name}: {path}")
    return paths


def command_provision_db(_args) -> None:
    from .provision import provision_external_databases

    settings = get_settings()
    provision_external_databases(
        settings.require_database_url(),
        os.getenv("AIRFLOW_METADATA_URL", "").strip() or None,
        os.getenv("POSTGRES_ADMIN_URL", "").strip() or None,
    )
    print("External PostgreSQL roles and databases are ready")


def command_init_runtime(args) -> None:
    from .database import ensure_runtime_tables, wait_for_database

    settings = get_settings()
    command_provision_db(args)
    wait_for_database(settings)
    ensure_runtime_tables(settings)
    print(
        f"Runtime tables ready: {settings.postgres_schema}.{settings.postgres_table}, "
        f"{settings.postgres_schema}.{settings.predictions_table}"
    )


def command_ingest_live(args):
    from .incremental_data import generate_and_store_bucket

    summary = generate_and_store_bucket(args.timestamp, settings=get_settings())
    print(summary.to_dict())
    return summary


def command_backfill_live(args):
    from .incremental_data import backfill_live_history

    result = backfill_live_history(end_utc=args.end, settings=get_settings())
    print(result)
    return result


def command_predict(args):
    from .predict import predict_and_store

    model_names = _parse_csv_strings(args.models) if args.models else None
    horizons = _parse_csv_ints(args.horizons) if args.horizons else None
    summary = predict_and_store(
        forecast_start_utc=args.forecast_start,
        settings=get_settings(),
        model_names=model_names,
        horizons=horizons,
    )
    print(summary.to_dict())
    return summary


def _active_model_files_exist(settings) -> bool:
    from .incremental_data import LIVE_GENERATOR_VERSION

    manifest_path = settings.model_dir / "active.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest["model_files"]
        if (manifest.get("model_name") != "lightgbm"
                or manifest.get("evaluation_schema") != 3
                or manifest.get("live_generator_version") != LIVE_GENERATOR_VERSION):
            return False
        for mode in ("Car", "Motorcycle"):
            for horizon in (10, 30, 60):
                file_path = (settings.model_dir / files[mode][str(horizon)]).resolve()
                if not file_path.is_relative_to(settings.model_dir.resolve()) or not file_path.is_file():
                    return False
        return True
    except (KeyError, ValueError, TypeError, OSError):
        return False


def command_bootstrap(args) -> None:
    """First start and restarts are automatic; existing rows are never replaced."""
    from .database import copy_parquet_to_postgres, database_has_rows
    from .incremental_data import backfill_live_history
    from .predict import predict_and_store

    settings = get_settings()
    command_init_runtime(args)
    seeded = not database_has_rows(settings)
    if seeded:
        generated_path = command_generate(
            argparse.Namespace(output_dir=settings.data_dir / "raw", overwrite=True)
        )
        copy_parquet_to_postgres(generated_path, settings=settings, batch_size=args.batch_size)
        print("Historical fake demand automatically loaded into PostgreSQL")
    else:
        print("Existing demand rows found; skipping historical seed to preserve data")

    result = backfill_live_history(settings=settings, full_scan=True)
    print(f"Live fake history synchronized: {result}")

    if seeded or not _active_model_files_exist(settings):
        summary = command_train(args)
        if not bool(summary["promoted"].all()):
            raise RuntimeError(
                "Initial training did not produce a complete model set; "
                "see outputs/training_alerts.json. No model was activated."
            )
        command_evaluate(args)
    else:
        print("Complete active LightGBM model set found; skipping initial retraining")

    result = backfill_live_history(settings=settings, full_scan=True)
    print(f"Live fake history after training synchronized: {result}")
    print(f"Initial forecast: {predict_and_store(settings=settings).to_dict()}")


def command_all(args) -> None:
    """Maintain the existing CLI alias, now safe to rerun without dropping data."""
    command_bootstrap(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate the calibrated fake dataset")
    generate.add_argument("--output-dir", type=Path, default=None)
    generate.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    generate.set_defaults(handler=command_generate)

    load = subparsers.add_parser("load-postgres", help="COPY Parquet into PostgreSQL")
    load.add_argument("--parquet", required=True)
    load.add_argument("--batch-size", type=int, default=200_000)
    load.set_defaults(handler=command_load_postgres)

    train = subparsers.add_parser("train", help="Train notebook direct LightGBM CPU models")
    train.add_argument("--models", default="lightgbm")
    train.add_argument("--horizons", default="10,30,60")
    train.add_argument("--rolling", action="store_true", help="Rolling 90-day train, 30-day validation and 30-day test")
    train.set_defaults(handler=command_train)

    evaluate = subparsers.add_parser("evaluate", help="Export model diagnostics")
    evaluate.add_argument("--rolling", action="store_true", help="Use rolling test predictions")
    evaluate.set_defaults(handler=command_evaluate)

    init_runtime = subparsers.add_parser(
        "init-runtime", help="Provision missing external PostgreSQL databases and create runtime tables"
    )
    init_runtime.set_defaults(handler=command_init_runtime)

    provision_db = subparsers.add_parser(
        "provision-db", help="Create missing roles/databases only (safe before restoring a backup)"
    )
    provision_db.set_defaults(handler=command_provision_db)

    bootstrap = subparsers.add_parser(
        "bootstrap", help="Idempotent first start: provision, seed, train, backfill and forecast"
    )
    bootstrap.add_argument("--models", default="lightgbm")
    bootstrap.add_argument("--horizons", default="10,30,60")
    bootstrap.add_argument("--batch-size", type=int, default=200_000)
    bootstrap.set_defaults(rolling=True, handler=command_bootstrap)

    ingest_live = subparsers.add_parser(
        "ingest-live", help="Generate and upsert one deterministic 10-minute bucket"
    )
    ingest_live.add_argument(
        "--timestamp",
        default=None,
        help="Bucket start as an ISO-8601 timestamp; defaults to the last closed UTC bucket",
    )
    ingest_live.set_defaults(handler=command_ingest_live)

    backfill = subparsers.add_parser("backfill-live", help="Fill history through current closed bucket")
    backfill.add_argument("--end", default=None, help="Exclusive UTC ISO timestamp; default current 10-minute boundary")
    backfill.set_defaults(handler=command_backfill_live)

    predict = subparsers.add_parser(
        "predict", help="Use trained models to predict and store the next horizons"
    )
    predict.add_argument(
        "--forecast-start",
        default=None,
        help="Forecast start as ISO-8601; defaults to latest observation plus 10 minutes",
    )
    predict.add_argument("--models", default=None)
    predict.add_argument("--horizons", default=None)
    predict.set_defaults(handler=command_predict)

    all_command = subparsers.add_parser("all", help="Run the idempotent bootstrap pipeline")
    all_command.add_argument("--models", default="lightgbm")
    all_command.add_argument("--horizons", default="10,30,60")
    all_command.add_argument("--batch-size", type=int, default=200_000)
    all_command.set_defaults(rolling=True)
    all_command.set_defaults(handler=command_all)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
