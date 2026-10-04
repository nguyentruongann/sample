"""Central configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # Keeps lightweight unit tests importable before installation.
    load_dotenv = None


PROJECT_ROOT = (
    Path(os.getenv("PROJECT_ROOT", str(Path(__file__).resolve().parents[2]))).expanduser().resolve()
)


def _optional_int(value: str | None) -> int | None:
    if value is None or not value.strip():
        return None
    return int(value)


def _csv_ints(value: str, default: tuple[int, ...]) -> tuple[int, ...]:
    if not value.strip():
        return default
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _csv_strings(value: str, default: tuple[str, ...]) -> tuple[str, ...]:
    if not value.strip():
        return default
    return tuple(item.strip().lower() for item in value.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    """Immutable application settings."""

    database_url: str
    postgres_schema: str
    postgres_table: str
    predictions_table: str
    seed: int
    local_timezone: str
    train_ratio: float
    max_train_rows: int | None
    model_names: tuple[str, ...]
    horizons: tuple[int, ...]
    validation_days: int
    test_start: str
    data_dir: Path
    model_dir: Path
    output_dir: Path
    prediction_history_days: int
    rolling_test_days: int = 30
    train_n_jobs: int = 4
    train_history_days: int = 90
    search_train_rows: int = 300_000
    search_val_rows: int = 100_000
    search_estimators: int = 800
    training_strategy: str = "locked"
    fresh_buckets_closed_at_t: bool = True

    @classmethod
    def from_env(cls) -> Settings:
        if load_dotenv is not None:
            load_dotenv(PROJECT_ROOT / ".env")

        train_ratio = float(os.getenv("TRAIN_RATIO", "0.80"))
        if not 0 < train_ratio < 1:
            raise ValueError("TRAIN_RATIO must be between 0 and 1.")

        horizons = _csv_ints(os.getenv("HORIZONS", "10,30,60"), (10, 30, 60))
        if not horizons or any(horizon not in (10, 30, 60) for horizon in horizons):
            raise ValueError("Only H10, H30 and H60 are supported.")
        model_names = _csv_strings(os.getenv("MODEL_NAMES", "lightgbm"), ("lightgbm",))
        if model_names != ("lightgbm",):
            raise ValueError("This project supports only the notebook's LightGBM models.")
        validation_days = int(os.getenv("VALIDATION_DAYS", "30"))
        if validation_days < 1:
            raise ValueError("VALIDATION_DAYS must be positive.")
        rolling_test_days = int(os.getenv("ROLLING_TEST_DAYS", "30"))
        if rolling_test_days < 1:
            raise ValueError("ROLLING_TEST_DAYS must be positive.")
        train_n_jobs = int(os.getenv("TRAIN_N_JOBS", "4"))
        if train_n_jobs < 1:
            raise ValueError("TRAIN_N_JOBS must be positive.")
        train_history_days = int(os.getenv("TRAIN_HISTORY_DAYS", "90"))
        search_train_rows = int(os.getenv("SEARCH_TRAIN_ROWS", "300000"))
        search_val_rows = int(os.getenv("SEARCH_VAL_ROWS", "100000"))
        search_estimators = int(os.getenv("SEARCH_ESTIMATORS", "800"))
        if min(train_history_days, search_train_rows, search_val_rows, search_estimators) < 1:
            raise ValueError("Training window and search budgets must be positive.")

        training_strategy = os.getenv("TRAINING_STRATEGY", "locked").strip().lower()
        if training_strategy not in ("locked", "notebook_search"):
            raise ValueError("TRAINING_STRATEGY must be locked or notebook_search")
        fresh_closed = os.getenv("FRESH_BUCKETS_CLOSED_AT_T", "true").strip().lower() in ("1", "true", "yes")
        if not fresh_closed:
            raise ValueError("final.ipynb features require closed/ingested T-10/T-20 buckets")
        prediction_history_days = int(os.getenv("PREDICTION_HISTORY_DAYS", "8"))
        if prediction_history_days < 8:
            raise ValueError("PREDICTION_HISTORY_DAYS must be at least 8.")

        return cls(
            database_url=os.getenv("DATABASE_URL", "").strip(),
            postgres_schema=os.getenv("POSTGRES_SCHEMA", "public").strip(),
            postgres_table=os.getenv("POSTGRES_TABLE", "fake_demand_10min").strip(),
            predictions_table=os.getenv("PREDICTIONS_TABLE", "demand_predictions").strip(),
            seed=int(os.getenv("SEED", "20260916")),
            local_timezone=os.getenv("LOCAL_TIMEZONE", "Asia/Ho_Chi_Minh").strip(),
            train_ratio=train_ratio,
            max_train_rows=_optional_int(os.getenv("MAX_TRAIN_ROWS")),
            model_names=model_names,
            horizons=horizons,
            validation_days=validation_days,
            test_start=os.getenv("TEST_START", "2026-08-15T09:40:00+07:00").strip(),
            data_dir=PROJECT_ROOT / "data",
            model_dir=PROJECT_ROOT / "models",
            output_dir=PROJECT_ROOT / "outputs",
            prediction_history_days=prediction_history_days,
            rolling_test_days=rolling_test_days,
            train_n_jobs=train_n_jobs,
            train_history_days=train_history_days,
            search_train_rows=search_train_rows,
            search_val_rows=search_val_rows,
            search_estimators=search_estimators,
            training_strategy=training_strategy,
            fresh_buckets_closed_at_t=fresh_closed,
        )

    def require_database_url(self) -> str:
        if not self.database_url:
            raise RuntimeError(
                "DATABASE_URL is missing. Copy .env.example to .env and configure PostgreSQL."
            )
        return self.database_url

    def ensure_directories(self) -> None:
        for path in (self.data_dir, self.model_dir, self.output_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings.from_env()
    settings.ensure_directories()
    return settings
